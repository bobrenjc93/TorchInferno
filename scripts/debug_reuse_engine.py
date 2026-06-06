#!/usr/bin/env python3
"""Synchronous single-GPU repro of the ContinuousBatchEngine prefix-reuse path,
matching the SERVER path: captured FlashInfer prefill graphs + online stepping +
reuse enabled. Under CUDA_LAUNCH_BLOCKING=1 the server's index assert localizes.

Run: CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
     TORCHINFERNO_CONTINUOUS_FI_REUSE=1 TORCHINFERNO_COMPILED_UNIFIED_FORWARD=0 \
     PYTHONPATH=src python scripts/debug_reuse_engine.py
"""

import tempfile
from pathlib import Path

import torch

from torchinferno.models.llama3 import (
    Llama3TensorParallelForCausalLM,
    Llama3V0ForCausalLM,
    tiny_llama3_config,
)
from torchinferno.runtime.serving import ContinuousBatchEngine, ServingRequest
from torchinferno.variant_validation import _write_llama3_hf_checkpoint


def main():
    dev = torch.device("cuda:0")
    max_seq = 512
    max_active = 16
    prefix_rows = 8
    cfg = tiny_llama3_config(
        vocab_size=128, hidden_size=512, num_attention_heads=4,
        num_key_value_heads=1, max_position_embeddings=max_seq,
    )
    eager = Llama3V0ForCausalLM(cfg).to(dev, torch.bfloat16).eval()
    with tempfile.TemporaryDirectory() as tmp:
        _write_llama3_hf_checkpoint(eager, Path(tmp))
        model = Llama3TensorParallelForCausalLM.from_pretrained(Path(tmp), dtype="bfloat16").eval()
    print("[ok] model built", flush=True)

    # External cache shared by graph capture and the engine (as the server does).
    cache = model.allocate_cache(max_active + prefix_rows, max_seq, cache_backend="flashinfer")
    print(f"[ok] cache {max_active + prefix_rows} rows x {max_seq}", flush=True)

    # Capture prefill graphs (server warmup) -- joint (batch,q) under token budget.
    pairs = [(b, q) for q in (256, 512) for b in (1, 2, 4, 8) if b * q <= 8192]
    captured = 0
    for b, q in pairs:
        try:
            for lc in cache.layers:
                if hasattr(lc, "_seq_lens"):
                    for r in range(len(lc._seq_lens)):
                        lc._seq_lens[r] = 0
            if model.capture_flashinfer_prefill_graph(cache, b, q):
                captured += 1
        except Exception as exc:
            print(f"[warn] capture ({b},{q}) failed: {exc!r}", flush=True)
    print(f"[ok] captured {captured} prefill graphs", flush=True)
    model._flashinfer_jit_warmed = True

    engine = ContinuousBatchEngine(
        model,
        device=dev,
        cache_backend="flashinfer",
        max_active_requests=max_active,
        prefix_cache_capacity=prefix_rows,
        temperature=0.7,
        store_reusable_prefixes=True,
        store_full_prompt_prefixes=True,
        pin_shared_prefix=False,
    )
    print("[ok] engine built", flush=True)

    # ~400-token shared prompt so wave-1 uses the (b,512) prefill graph and
    # wave-2 reuses it. Online stepping (submit/step) mirrors the server.
    shared = tuple(i % cfg.vocab_size for i in range(400))
    engine.start_online(max_seq_len=max_seq, external_cache=cache)
    for j in range(6):
        engine.submit_online(ServingRequest(f"w0-{j}", shared, 4, arrival_step=0))
    print("[ok] wave1 submitted; stepping ...", flush=True)
    steps = 0
    while engine.has_online_work() and steps < 50:
        engine.step_online()
        steps += 1
    torch.cuda.synchronize()
    print(f"[ok] wave1 done in {steps} steps; reuse_reqs={engine.stats.prefix_reuse_requests}", flush=True)

    for j in range(6):
        engine.submit_online(ServingRequest(f"w1-{j}", shared, 4, arrival_step=steps))
    print("[ok] wave2 submitted (should REUSE); stepping ...", flush=True)
    while engine.has_online_work() and steps < 120:
        engine.step_online()
        steps += 1
    torch.cuda.synchronize()
    print(f"[ok] wave2 done; reuse_batches={engine.stats.prefill_prefix_reuse_batches} "
          f"reuse_reqs={engine.stats.prefix_reuse_requests} "
          f"prefill_calls={engine.stats.prefill_model_calls}", flush=True)
    print("ENGINE+GRAPH REUSE OK", flush=True)


if __name__ == "__main__":
    main()
