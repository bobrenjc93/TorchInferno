#!/usr/bin/env python3
"""Isolate triton-vs-aten RoPE cost at DECODE vs PREFILL shapes.

The rope-fusion commit routed BOTH the single-token decode path and the
multi-token ragged-prefill path through one batched triton kernel. The decode win
was profiled (13x), but the prefill regime (large tokens-per-row) was never
isolated -- and the benchmark regressed exactly the prefill-heavy workloads. This
times both impls at Llama-3.1-70B TP8 local dims (q_heads=8, kv_heads=1, hd=128).
"""
import time

import torch

from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_batched_inplace


def _rotate(x, cos, sin):
    half = x.size(-1) // 2
    if cos.size(-1) == half:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def bench(name, fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"  {name:28s}: {us:8.1f} us/call")
    return us


def run_shape(label, batch, tokens):
    dev = "cuda"
    dt = torch.bfloat16
    qh, kvh, hd = 8, 1, 128
    print(f"\n== {label}: batch={batch} tokens={tokens} (q[{batch},{qh},{tokens},{hd}]) ==")
    q = torch.randn(batch, qh, tokens, hd, device=dev, dtype=dt)
    k = torch.randn(batch, kvh, tokens, hd, device=dev, dtype=dt)
    cos = torch.randn(batch, tokens, hd, device=dev, dtype=dt)
    sin = torch.randn(batch, tokens, hd, device=dev, dtype=dt)

    def triton_call():
        triton_apply_rotary_llama_batched_inplace(q.clone(), k.clone(), cos, sin)

    cos_b = cos[:, None, :, :]
    sin_b = sin[:, None, :, :]

    def aten_call():
        _rotate(q, cos_b, sin_b)
        _rotate(k, cos_b, sin_b)

    t = bench("triton batched inplace", triton_call)
    a = bench("aten rotate-half", aten_call)
    print(f"  -> triton/aten ratio: {t/a:.2f}x  ({'triton WINS' if t < a else 'ATEN WINS'})")


def main():
    # decode: one token per row, high batch (concurrency)
    run_shape("DECODE 64-row", 64, 1)
    run_shape("DECODE 128-row", 128, 1)
    # prefill ragged suffix: many tokens per row
    run_shape("PREFILL 8x512", 8, 512)
    run_shape("PREFILL 16x256", 16, 256)
    run_shape("PREFILL 64x256", 64, 256)
    run_shape("PREFILL 1x2048", 1, 2048)


if __name__ == "__main__":
    main()
