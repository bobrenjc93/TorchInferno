#!/usr/bin/env python3
"""Marlin int4 (W4A16) GEMM at Llama3-70B TP8 DECODE shapes, via vLLM's prebuilt
kernel -- the cell-flipping batched-int4 decode lever, FINALLY unblocked.

KEY: vLLM's _C is built against torch's STABLE libtorch ABI, so `import vllm`
registers torch.ops._C.marlin_gemm and it works with our custom torch 2.13.0a0
WITHOUT rebuilding (unlike torchao 0.17's cutlass .so, which was CUDA13-ABI and
failed to load; and unlike torchao 0.18, whose int4 needs the fbcode-only `mslk`
dep and whose cutlass kernels are FP8-sparse, not int4-marlin).

Run (LD_PRELOAD fixes a soxr/libstdc++ CXXABI clash in the vllm->transformers
import chain; the marlin op itself does not need it once registered):
  LD_PRELOAD=/home/bobren/local/d/pytorch-env/lib/libstdc++.so.6 \
    python scripts/bench_marlin_int4.py

Result (CUDA-graph floor, M=48): gate_up 1.52x, down 1.44x vs fp16 (marlin wins
the big, K-large GEMMs); tiny qkv/o_proj lose (marlin's fixed overhead vs a
~10us bf16 GEMM). A HYBRID -- marlin for gate_up+down, bf16 for qkv/o_proj/lm_head
(lm_head N=16032 is not divisible by marlin's min_thread_n=64) -- is ~1.38x on the
decode projection GEMMs. Correctness: maxdiff ~0.008 vs fp16 (RTN int4 quant
error; real integration needs GPTQ/AWQ calibration to hold the 98% bench bar).
"""

import functools
import sys

import torch

sys.path.insert(0, "/data/users/bobren/d/vllm")
import vllm  # noqa: F401  (registers torch.ops._C.marlin_gemm)
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)

QT = scalar_types.uint4b8
GROUP = 128


def bench(fn, it=100, wu=30):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000


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
    return s.elapsed_time(e) / it * 1000


def make(M, N, K, dev):
    # weight is [K, N] (the GEMM is a[M,K] @ w[K,N]); marlin_quantize packs it.
    w = torch.randn(K, N, device=dev, dtype=torch.float16) * 0.02
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    _w_ref, q_w, s, g_idx, srt, _ = marlin_quantize(w, QT, GROUP, False)
    wsp = marlin_make_workspace_new(dev)
    out = torch.empty((M, N), dtype=torch.float16, device=dev)
    marlin = functools.partial(
        ops.marlin_gemm, a, out, q_w, None, s, None, None, None, g_idx, srt, wsp,
        QT, M, N, K, is_k_full=True,
    )
    fp16 = functools.partial(torch.mm, a, w)
    return fp16, marlin


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    # marlin needs N % 64 == 0; lm_head (N=16032) does NOT qualify -> stays bf16.
    shapes = {"qkv": (1280, 8192), "o_proj": (8192, 1024),
              "gate_up": (7168, 8192), "down": (8192, 3584)}
    for use_graph, label in ((False, "eager"), (True, "CUDA-graph floor")):
        b = gbench if use_graph else bench
        for M in (16, 48, 64):
            print(f"=== M={M} ({label}) ===")
            tb = tm = 0.0
            for nm, (N, K) in shapes.items():
                fp16, marlin = make(M, N, K, dev)
                t_bf, t_mar = b(fp16), b(marlin)
                tb += t_bf
                tm += t_mar
                print(f"  {nm:8s} [{K}x{N}] fp16={t_bf:6.1f}us marlin-int4={t_mar:6.1f}us speedup={t_bf/t_mar:.2f}x")
            print(f"  TOTAL fp16={tb:.1f}us marlin={tm:.1f}us speedup={tb/tm:.2f}x")


if __name__ == "__main__":
    main()
