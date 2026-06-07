#!/usr/bin/env python3
"""Microbenchmark: bf16 vs FP8 (e4m3) GEMM at Llama3-70B TP=8 DECODE shapes.

Decode is memory-bound on the weight read; FP8 weights are half the bytes, so
this measures whether torch._scaled_mm actually delivers the ~2x we expect for
the skinny (M=batch) decode GEMMs before committing to an FP8 model change.

  python scripts/bench_fp8_decode.py
"""

import torch

FP8 = torch.float8_e4m3fn


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    dev = torch.device("cuda:0")
    # Llama3.1-70B: hidden=8192, n_heads=64, n_kv=8, head_dim=128,
    # intermediate=28672, vocab=128256. TP=8 -> shard the inner/outer dims.
    H = 8192
    shapes = {
        # name: (N_out, K_in) for a [M, K] x [K, N] -> [M, N] decode GEMM (TP8 sharded)
        "qkv_proj":  ((64 + 2 * 8) * 128 // 8, H),     # fused qkv, heads sharded
        "o_proj":    (H, 64 * 128 // 8),               # out proj, input sharded
        "gate_up":   (2 * 28672 // 8, H),              # fused gate+up, sharded
        "down_proj": (H, 28672 // 8),                  # down, input sharded
        "lm_head":   (128256 // 8, H),                 # vocab sharded
    }
    batches = [16, 32, 48, 64]
    print(f"H100 decode GEMM bf16 vs fp8-e4m3 (us/call); TP8-sharded weights\n")
    for bs in batches:
        print(f"=== batch M={bs} ===")
        tot_bf16 = tot_fp8 = 0.0
        for name, (N, K) in shapes.items():
            x = torch.randn(bs, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)  # row-major [N,K]

            def bf16():
                return torch.mm(x, w.t())

            # FP8: quantize x and w to e4m3 with per-tensor scales; _scaled_mm
            # wants w as [K, N] column-major (w_fp8.t()) and fp32 scales.
            xs = (x.abs().max() / 448.0).clamp(min=1e-4)
            ws = (w.abs().max() / 448.0).clamp(min=1e-4)
            x_fp8 = (x / xs).to(FP8)
            w_fp8 = (w / ws).to(FP8)
            w_fp8_t = w_fp8.t().contiguous().t()  # ensure [N,K] then .t() -> [K,N] view
            scale_a = xs.float().reshape(1)
            scale_b = ws.float().reshape(1)

            def fp8():
                return torch._scaled_mm(
                    x_fp8, w_fp8.t(),
                    scale_a=scale_a, scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                )

            try:
                t_bf16 = bench(bf16)
            except Exception as e:
                t_bf16 = float("nan"); print(f"  {name}: bf16 FAIL {e!r}")
            try:
                t_fp8 = bench(fp8)
            except Exception as e:
                t_fp8 = float("nan"); print(f"  {name}: fp8 FAIL {e!r}"); continue
            tot_bf16 += t_bf16
            tot_fp8 += t_fp8
            spd = t_bf16 / t_fp8 if t_fp8 else 0.0
            print(f"  {name:10s} [{bs},{K}]x[{K},{N}]  bf16={t_bf16:7.1f}us  fp8={t_fp8:7.1f}us  speedup={spd:.2f}x")
        spd_tot = tot_bf16 / tot_fp8 if tot_fp8 else 0.0
        print(f"  {'TOTAL':10s} bf16={tot_bf16:7.1f}us  fp8={tot_fp8:7.1f}us  speedup={spd_tot:.2f}x\n")


if __name__ == "__main__":
    main()
