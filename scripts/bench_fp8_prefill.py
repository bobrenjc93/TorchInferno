#!/usr/bin/env python3
"""bf16 vs FP8 (e4m3) GEMM at Llama3-70B TP=8 PREFILL shapes -- the TTFT lever.

Prefill is COMPUTE-bound (large M = tokens), so FP8 tensor cores (~2x bf16 peak)
are the biggest TTFT lever. This benchmark establishes the REALISTIC ceiling and,
critically, how much of it the activation quant eats. All timings are CUDA-GRAPHED
(prefill runs graphed in the server), which is the only fair comparison.

DECISIVE FINDINGS (real 70B TP8 shapes, H100):
  1. SCALES MUST BE TENSORWISE (scalar), not rowwise. torch._scaled_mm with rowwise
     scales (scale_a [M,1], scale_b [1,N]) hits a SLOW kernel (~0.3x, a fixed ~160us
     floor). Tensorwise (scalar scale_a, scale_b) hits the fast cublasLt path = 2.0x.
  2. FP8 GEMM excl quant (fused-into-rmsnorm ceiling) = 2.00x at M>=1024 (gate_up
     113us vs bf16 224us @ M=1024).
  3. FP8 GEMM incl dynamic tensorwise quant, GRAPHED = only ~1.07x @ M=1024
     (209us): the amax+div+cast (~96us even graphed, 3 passes over [M,K]) nearly
     eats the GEMM win. At M=2048 the GEMM is bigger so quant amortizes -> ~1.37x.
  => FP8 prefill needs the quant FUSED into the rmsnorm/swiglu epilogue (output fp8 +
     a scalar scale in the same kernel that already streams the activation) to realize
     the 2x. Naive eager/graphed dynamic quant is ~1.0-1.4x (M-dependent) and REGRESSES
     the small GEMMs (qkv/o_proj). Even the 2x ceiling is ~1.44x overall prefill (GEMMs
     are 61%; see prefill-is-gemm-bound) -> insufficient ALONE to flip the closest TTFT
     cell (few_shot 1.85x); must pair with a custom allreduce (the 24% allreduce slice).

  PYTHONPATH=src python scripts/bench_fp8_prefill.py
"""
import torch

from torchinferno.kernels.fp8 import quantize_activation_fp8

FP8 = torch.float8_e4m3fn
SHAPES = [("qkv", 8192, 1280), ("o_proj", 1024, 8192),
          ("gate_up", 8192, 7168), ("down", 3584, 8192)]


def gbench(fn, it=100):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0  # us


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    print(
        f"{'shape':10s} {'M':>5s} {'bf16':>8s} {'dynamic':>8s} {'static':>8s} "
        f"{'fp8-q':>8s} {'dyn':>6s} {'stat':>6s} {'excl':>6s}  (graphed, tensorwise)"
    )
    for M in (1, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048):
        tb = td = ts = te = 0.0
        for nm, K, N in SHAPES:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02
            sb = (w.abs().amax() / 448.0).to(torch.float32)
            wq = (w.t() / sb).to(FP8).contiguous().t()  # [K,N] col-major
            sa0 = (a.abs().amax() / 448.0).to(torch.float32)
            aq0 = (a / sa0).to(FP8)
            static_scale = sa0 * 4.0
            static_inverse_scale = static_scale.reciprocal()

            def f_bf16(a=a, w=w):
                return torch.mm(a, w)

            def f_dynamic(a=a, wq=wq, sb=sb):
                sa = (a.abs().amax() / 448.0).to(torch.float32)
                aq = (a / sa).to(FP8)
                return torch._scaled_mm(aq, wq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

            def f_static(
                a=a,
                wq=wq,
                sb=sb,
                static_scale=static_scale,
                static_inverse_scale=static_inverse_scale,
            ):
                aq, sa = quantize_activation_fp8(
                    a,
                    scale=static_scale,
                    inverse_scale=static_inverse_scale,
                )
                return torch._scaled_mm(aq, wq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

            def f_excl(aq0=aq0, sa0=sa0, wq=wq, sb=sb):
                return torch._scaled_mm(aq0, wq, scale_a=sa0, scale_b=sb, out_dtype=torch.bfloat16)

            try:
                t_b = gbench(f_bf16)
                t_d = gbench(f_dynamic)
                t_s = gbench(f_static)
                t_e = gbench(f_excl)
            except Exception as exc:
                print(f"  {nm}: _scaled_mm failed: {exc}")
                return
            tb += t_b
            td += t_d
            ts += t_s
            te += t_e
            print(
                f"{nm:10s} {M:5d} {t_b:7.1f}u {t_d:7.1f}u {t_s:7.1f}u {t_e:7.1f}u "
                f"{t_b / t_d:5.2f}x {t_b / t_s:5.2f}x {t_b / t_e:5.2f}x"
            )
        print(
            f"{'TOTAL':10s} {M:5d} {tb:7.1f}u {td:7.1f}u {ts:7.1f}u {te:7.1f}u "
            f"{tb / td:5.2f}x {tb / ts:5.2f}x {tb / te:5.2f}x   <-- M={M}\n"
        )


if __name__ == "__main__":
    main()
