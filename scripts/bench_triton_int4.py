#!/usr/bin/env python3
"""Triton W4A16 (packed int4 weight, bf16 activation, groupwise dequant) GEMM,
tuned for BATCHED decode (M=16-64) -- the regime where PyTorch's M=1-tuned
_weight_int4pack_mm loses. The int4 weight is 1/4 the bytes of bf16, so for the
memory-bound decode GEMM the floor is ~4x lower; a well-pipelined kernel should
beat cuBLAS bf16 here. If so, it is the unblocked path to faster batched decode
(no torchao/vllm cutlass ABI dependency) -> flips tree/multi_turn TPOT.

Design: BLOCK_K == group (one scale per block). Weight byte holds two int4 along
K: low nibble = even-K column, high nibble = odd-K column. So split A into
even/odd K halves and do TWO tl.dot()s -- no per-element nibble branching.

  python scripts/bench_triton_int4.py
"""

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bn in (64, 128, 256):
        for bm in (16, 32, 64):
            for ns in (3, 4, 5):
                for nw in (4, 8):
                    cfgs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=ns, num_warps=nw))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _w4a16_kernel(
    a_ptr, wq_ptr, s_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,      # wq: [N, K//2] uint8
    stride_sn, stride_sg,      # scales: [N, K//GROUP]
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    BLOCK_K: tl.constexpr = GROUP
    HALF: tl.constexpr = GROUP // 2
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)        # half-K index within a block
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_blocks = tl.cdiv(K, BLOCK_K)
    for kb in range(0, n_blocks):
        k_even = kb * BLOCK_K + 2 * offs_h            # even K columns
        k_odd = k_even + 1
        a_even = tl.load(a_ptr + offs_m[:, None] * stride_am + k_even[None, :] * stride_ak,
                         mask=m_mask & (k_even[None, :] < K), other=0.0)
        a_odd = tl.load(a_ptr + offs_m[:, None] * stride_am + k_odd[None, :] * stride_ak,
                        mask=m_mask & (k_odd[None, :] < K), other=0.0)
        byte_col = kb * HALF + offs_h
        wb = tl.load(wq_ptr + offs_n[:, None] * stride_wn + byte_col[None, :] * stride_wk,
                     mask=n_mask & (byte_col[None, :] < (K // 2)), other=0).to(tl.int32)
        low = ((wb & 0xF) - 8).to(tl.float32)         # even-K weights [BLOCK_N, HALF]
        high = (((wb >> 4) & 0xF) - 8).to(tl.float32)  # odd-K weights
        sc = tl.load(s_ptr + offs_n[:, None] * stride_sn + kb * stride_sg,
                     mask=n_mask, other=0.0).to(tl.float32)  # [BLOCK_N,1] one group/block
        low = (low * sc).to(tl.bfloat16)
        high = (high * sc).to(tl.bfloat16)
        acc += tl.dot(a_even, low.trans(), allow_tf32=False)
        acc += tl.dot(a_odd, high.trans(), allow_tf32=False)
    c = acc.to(tl.bfloat16)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c,
             mask=m_mask & (offs_n[None, :] < N))


def w4a16(a, wq, scales, group, N, K):
    M = a.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _w4a16_kernel[grid](
        a, wq, scales, c, M, N, K,
        a.stride(0), a.stride(1), wq.stride(0), wq.stride(1),
        scales.stride(0), scales.stride(1), GROUP=group,
    )
    return c


def quantize_int4(w, group=128):
    N, K = w.shape
    wg = w.reshape(N, K // group, group).float()
    scales = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-6)
    q = (wg / scales).round().clamp(-8, 7).to(torch.int32).reshape(N, K) + 8
    lo = q[:, 0::2]; hi = q[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed.contiguous(), scales.reshape(N, K // group).to(torch.bfloat16).contiguous()


def bench(fn, it=100, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000


def main():
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    group = 128
    N, K = 512, 1024
    a = torch.randn(48, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, sc = quantize_int4(w, group)
    ref = a.float() @ w.float().t()
    out = w4a16(a, wq, sc, group, N, K).float()
    print(f"correctness rel-err: {((out-ref).abs().mean()/ref.abs().mean()).item():.4f}\n")
    shapes = {"qkv": (1280, 8192), "o_proj": (8192, 1024), "gate_up": (7168, 8192),
              "down": (8192, 3584), "lm_head": (16032, 8192)}
    for M in [16, 48, 64]:
        print(f"=== M={M} ==="); tb = tt = 0.0
        for nm, (N, K) in shapes.items():
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            wq, sc = quantize_int4(w, group)
            t_bf = bench(lambda: torch.mm(a, w.t()))
            try:
                t_tt = bench(lambda: w4a16(a, wq, sc, group, N, K))
            except Exception as ex:
                print(f"  {nm}: triton FAIL {ex!r}"); continue
            tb += t_bf; tt += t_tt
            print(f"  {nm:8s} [{M},{K}]x[{K},{N}] bf16={t_bf:6.1f}us int4={t_tt:6.1f}us speedup={t_bf/t_tt:.2f}x")
        if tt: print(f"  TOTAL bf16={tb:.1f}us int4={tt:.1f}us speedup={tb/tt:.2f}x")


if __name__ == "__main__":
    main()
