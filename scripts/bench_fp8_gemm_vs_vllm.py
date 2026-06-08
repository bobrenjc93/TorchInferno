#!/usr/bin/env python3
"""Is our torch._scaled_mm fp8 prefill GEMM as fast as vLLM's cutlass_scaled_mm?

Direct head-to-head at Llama-70B TP8 gate_up prefill shapes, CUDA-graphed. Answers
whether the residual TTFT gap vs vLLM (after our fp8 prefill lands) is the GEMM KERNEL
or something else (prefix-reuse / scheduling).

RESULT (real H100): our _scaled_mm MATCHES vLLM cutlass (M=512 55.9 vs 62.6us; M=1024
109 vs 119us; M=2048 206 vs 200us) -- parity. So our fp8 prefill GEMM is NOT the
bottleneck; the residual few_shot TTFT gap is prefix-reuse (vLLM caches the 5 shared
few-shot shots; we re-prefill) + scheduling, NOT kernel quality.

  PYTHONPATH=src python scripts/bench_fp8_gemm_vs_vllm.py
"""
import torch

FP8 = torch.float8_e4m3fn


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
    return s.elapsed_time(e) / it * 1000


def main():
    import vllm  # noqa: F401
    from vllm import _custom_ops as ops

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    for M, K, N in [(512, 8192, 7168), (1024, 8192, 7168), (2048, 8192, 7168)]:
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02
        sb = (w.abs().amax() / 448.0).to(torch.float32)
        wq = (w.t() / sb).to(FP8).contiguous().t()
        wq_v = (w.t() / sb).to(FP8).contiguous()
        sa = (a.abs().amax() / 448.0).to(torch.float32)
        aq = (a / sa).to(FP8)

        def bf16(a=a, w=w):
            return torch.mm(a, w)

        def ours(aq=aq, wq=wq, sa=sa, sb=sb):
            return torch._scaled_mm(aq, wq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

        def vll(aq=aq, wq_v=wq_v, sa=sa, sb=sb):
            return ops.cutlass_scaled_mm(aq, wq_v.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

        tb, to, tv = gbench(bf16), gbench(ours), gbench(vll)
        print(f"gate_up M={M}: bf16={tb:.1f}u  ours(_scaled_mm)={to:.1f}u ({tb/to:.2f}x)  "
              f"vllm(cutlass)={tv:.1f}u ({tb/tv:.2f}x)")


if __name__ == "__main__":
    main()
