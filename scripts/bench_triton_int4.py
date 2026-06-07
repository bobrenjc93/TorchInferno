#!/usr/bin/env python3
"""Triton W4A16 (packed int4 weight, bf16 activation, groupwise dequant) GEMM,
tuned for BATCHED decode (M=16-64) -- the regime where PyTorch's M=1-tuned
_weight_int4pack_mm loses. If this beats bf16 at M=48, it is the unblocked path
to faster batched decode (no torchao/vllm cutlass ABI dependency).

  python scripts/bench_triton_int4.py
"""

import torch
import triton
import triton.language as tl

FP_DTYPE = tl.bfloat16


@triton.jit
def _w4a16_kernel(
    a_ptr, wq_ptr, s_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,      # wq: [N, K//2] uint8
    stride_sn, stride_sg,      # scales: [N, K//G]
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # iterate K in BLOCK_K chunks; weight byte holds 2 int4 along K (k, k+1)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0,
        ).to(tl.float32)
        # packed weight: column (k0+kk) lives in byte (k0+kk)//2, nibble (k0+kk)%2
        kk = k0 + offs_k
        byte_col = kk // 2
        nib = kk % 2
        wb = tl.load(
            wq_ptr + offs_n[:, None] * stride_wn + byte_col[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (byte_col[None, :] < (K // 2)), other=0,
        ).to(tl.int32)
        nibble = tl.where(nib[None, :] == 0, wb & 0xF, (wb >> 4) & 0xF)
        wv = (nibble - 8).to(tl.float32)  # int4 value in [-8,7]
        g = kk // GROUP
        sc = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + g[None, :] * stride_sg,
            mask=(offs_n[:, None] < N), other=0.0,
        ).to(tl.float32)
        w = wv * sc  # [BLOCK_N, BLOCK_K] dequant
        acc += tl.dot(a, w.trans(), allow_tf32=False)
    c = acc.to(FP_DTYPE)
    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def w4a16(a, wq, scales, group, N, K):
    M = a.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BM = 16 if M <= 16 else (32 if M <= 32 else 64)
    BN, BK = 64, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w4a16_kernel[grid](
        a, wq, scales, c, M, N, K,
        a.stride(0), a.stride(1),
        wq.stride(0), wq.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP=group, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return c


def quantize_int4(w, group=128):
    # w: [N,K] bf16 -> packed uint8 [N,K//2], scales [N,K//G]
    N, K = w.shape
    wg = w.reshape(N, K // group, group).float()
    scales = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-6)
    q = (wg / scales).round().clamp(-8, 7).to(torch.int32).reshape(N, K) + 8  # 0..15
    lo = q[:, 0::2]; hi = q[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed.contiguous(), scales.reshape(N, K // group).to(torch.bfloat16).contiguous()


def bench(fn, it=100, wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000


def main():
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    shapes = {"qkv": (1280, 8192), "o_proj": (8192, 1024), "gate_up": (7168, 8192),
              "down": (8192, 3584), "lm_head": (16032, 8192)}
    group = 128
    # correctness check first
    N, K = 256, 512
    a = torch.randn(48, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
    wq, sc = quantize_int4(w, group)
    ref = (a.float() @ w.float().t())
    out = w4a16(a, wq, sc, group, N, K).float()
    rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
    print(f"correctness rel-err (int4 quant vs fp32 ref): {rel:.4f}\n")
    for M in [16, 48, 64]:
        print(f"=== M={M} ==="); tb = tt = 0.0
        for nm, (N, K) in shapes.items():
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
            wq, sc = quantize_int4(w, group)
            t_bf = bench(lambda: torch.mm(a, w.t()))
            try:
                t_tt = bench(lambda: w4a16(a, wq, sc, group, N, K))
            except Exception as ex:
                print(f"  {nm}: triton FAIL {ex!r}"); continue
            tb += t_bf; tt += t_tt
            print(f"  {nm:8s} [{M},{K}]x[{K},{N}] bf16={t_bf:6.1f}us triton-int4={t_tt:6.1f}us speedup={t_bf/t_tt:.2f}x")
        if tt: print(f"  TOTAL bf16={tb:.1f}us int4={tt:.1f}us speedup={tb/tt:.2f}x")


if __name__ == "__main__":
    main()
