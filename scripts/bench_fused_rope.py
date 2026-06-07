#!/usr/bin/env python3
"""Validate + benchmark the fused decode RoPE kernel vs the aten rotate-half path.

The FlashInfer decode applies rotary via _rotate_llama_eager (aten cat/neg/mul):
the batch=32 decode profiler showed ~2.6ms of the step in rope elementwise ops.
triton_apply_rotary_llama_batched_inplace fuses the whole rotate-half into one
in-place kernel with per-(batch,token) cos. This script checks it matches the aten
reference within bf16 tolerance and times both at Llama3-70B TP8 decode shapes.

  PYTHONPATH=src python scripts/bench_fused_rope.py
"""

import torch

from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_batched_inplace

Q_HEADS = 8   # 64 attn heads / TP8
K_HEADS = 1   # 8 KV heads / TP8 (GQA)
HEAD_DIM = 128
HALF = HEAD_DIM // 2


def rotate_half_eager(x, cos, sin):
    # Mirror _rotate_llama_eager with pre-expanded (full head_dim) cos/sin.
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def make_inputs(batch, tokens, dev):
    # q/k come from _qkv as [b, t, h, d].transpose(1,2) -> non-contiguous strides.
    q = torch.randn(batch, tokens, Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(batch, tokens, K_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    pos = torch.randint(0, 4096, (batch, tokens), device=dev)
    inv = 1.0 / (500000.0 ** (torch.arange(0, HALF, device=dev).float() / HALF))
    ang = pos.float()[..., None] * inv[None, None, :]            # [b, t, half]
    cos_h = torch.cos(ang).to(torch.bfloat16)
    sin_h = torch.sin(ang).to(torch.bfloat16)
    cos = torch.cat((cos_h, cos_h), dim=-1)                      # [b, t, head_dim]
    sin = torch.cat((sin_h, sin_h), dim=-1)
    return q, k, cos, sin


def check(batch, tokens, dev):
    q, k, cos, sin = make_inputs(batch, tokens, dev)
    # reference (broadcast cos over heads): cos [b,t,d] -> [b,1,t,d]
    cb = cos[:, None, :, :]
    sb = sin[:, None, :, :]
    q_ref = rotate_half_eager(q.float(), cb.float(), sb.float())
    k_ref = rotate_half_eager(k.float(), cb.float(), sb.float())
    qf = q.clone()
    kf = k.clone()
    triton_apply_rotary_llama_batched_inplace(qf, kf, cos, sin)
    qd = (qf.float() - q_ref).abs().max().item()
    kd = (kf.float() - k_ref).abs().max().item()
    qrel = qd / q_ref.abs().max().item()
    krel = kd / k_ref.abs().max().item()
    print(f"  b={batch:3d} t={tokens} q max|d|={qd:.4e} (rel {qrel:.2e})  k max|d|={kd:.4e} (rel {krel:.2e})")
    assert qrel < 1e-2 and krel < 1e-2, "fused rope diverges from aten reference"


def bench(fn, it=200, wu=50):
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
    return s.elapsed_time(e) / it * 1000  # us


def gbench(build_iter, layers=80, it=200):
    # Capture `layers` rope applications (the per-step rope cost) into a CUDA
    # graph so launch overhead is amortized -- this is the real in-decode-graph
    # GPU time, unlike the eager microbench which is python-dispatch bound.
    steps = build_iter()
    for fn in steps:
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for fn in steps:
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
    return s.elapsed_time(e) / it  # ms per step (all `layers` applications)


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    print("=== correctness (fused vs aten rotate-half) ===")
    for b in (16, 32, 48, 64):
        check(b, 1, dev)
    check(32, 8, dev)   # ragged suffix
    print("=== in-CUDA-graph per-step rope time (80 layers x 2 q/k applications) ===")
    for b in (16, 32, 48, 64):
        q, k, cos, sin = make_inputs(b, 1, dev)
        cb, sb = cos[:, None, :, :], sin[:, None, :, :]
        # Distinct q/k buffers per layer so the graph does real work each call.
        qa = [q.clone() for _ in range(80)]
        ka = [k.clone() for _ in range(80)]
        qfb = [q.clone() for _ in range(80)]
        kfb = [k.clone() for _ in range(80)]

        def build_aten():
            return [lambda i=i: (rotate_half_eager(qa[i], cb, sb), rotate_half_eager(ka[i], cb, sb)) for i in range(80)]

        def build_fused():
            return [lambda i=i: triton_apply_rotary_llama_batched_inplace(qfb[i], kfb[i], cos, sin) for i in range(80)]

        t_aten = gbench(build_aten)
        t_fused = gbench(build_fused)
        print(f"  b={b:3d}: aten={t_aten:.3f}ms fused={t_fused:.3f}ms "
              f"({t_aten/t_fused:.2f}x)  save={t_aten-t_fused:.3f}ms/step")


if __name__ == "__main__":
    main()
