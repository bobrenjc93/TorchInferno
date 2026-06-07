#!/usr/bin/env python3
"""Is decode weight-bound (flat TPOT vs batch) or compute-bound (scales)?

This decides whether raising effective concurrency (KV-token-bounded admission)
is a clean throughput+TTFT win. Times the per-GPU Llama3-70B TP8 decode
projection GEMMs at batch M = 32..256 inside a CUDA graph. If total time is ~flat
across M, decode is weight-bound and high concurrency is ~free TPOT-wise.

  PYTHONPATH=src python scripts/bench_decode_batch_scaling.py
"""
import torch

# per-GPU (TP8) decode GEMM shapes: (name, K, N) for a [M,K]@[K,N] projection
SHAPES = [
    ("qkv", 8192, 1280),
    ("o_proj", 1024, 8192),
    ("gate_up", 8192, 7168),
    ("down", 3584, 8192),
]


def gbench(fn, it=300):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000  # us


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    weights = {n: torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02 for n, K, N in SHAPES}
    print(f"{'M':>5} " + " ".join(f"{n:>9}" for n, _, _ in SHAPES) + f"{'TOTAL':>10} {'us/row':>8}")
    base = None
    for M in (32, 48, 64, 96, 128, 192, 256):
        acts = {n: torch.randn(M, K, device=dev, dtype=torch.bfloat16) for n, K, _N in SHAPES}
        times = {}
        for n, K, N in SHAPES:
            a, w = acts[n], weights[n]
            times[n] = gbench(lambda a=a, w=w: torch.mm(a, w))
        total = sum(times.values())
        if base is None:
            base = total
        per_row = total / M
        print(f"{M:>5} " + " ".join(f"{times[n]:9.1f}" for n, _, _ in SHAPES)
              + f"{total:10.1f} {per_row:8.2f}  ({total/base:.2f}x vs M=32)")


if __name__ == "__main__":
    main()
