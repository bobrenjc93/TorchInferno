#!/usr/bin/env python3
"""Synchronous single-GPU repro of the ContinuousBatchEngine prefix-reuse path.

Drives the SAME _prefill_many / _prefill_flashinfer_reuse code the server uses,
on a tiny TP model + FlashInfer cache, under CUDA_LAUNCH_BLOCKING=1 so the
index assert that crashes the server localizes synchronously.

Run: CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
     TORCHINFERNO_COMPILED_UNIFIED_FORWARD=0 PYTHONPATH=src \
     python scripts/debug_reuse_engine.py
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
    cfg = tiny_llama3_config(
        vocab_size=128, hidden_size=512, num_attention_heads=4,
        num_key_value_heads=1, max_position_embeddings=512,
    )
    eager = Llama3V0ForCausalLM(cfg).to(dev, torch.bfloat16).eval()
    with tempfile.TemporaryDirectory() as tmp:
        _write_llama3_hf_checkpoint(eager, Path(tmp))
        model = Llama3TensorParallelForCausalLM.from_pretrained(Path(tmp), dtype="bfloat16").eval()
    print("[ok] model built", flush=True)

    engine = ContinuousBatchEngine(
        model,
        device=dev,
        cache_backend="flashinfer",
        max_active_requests=16,
        prefix_cache_capacity=8,
        temperature=0.7,
        store_reusable_prefixes=True,
        store_full_prompt_prefixes=True,
        pin_shared_prefix=False,  # store per-request prefixes for reuse
    )
    print("[ok] engine built", flush=True)

    # Identical prompts (self_consistency): wave 1 prefills + stores, wave 2 reuses.
    shared = tuple(i % cfg.vocab_size for i in range(40))
    reqs = []
    for wave in range(2):
        for j in range(6):
            reqs.append(ServingRequest(f"w{wave}-{j}", shared, 4, arrival_step=wave))

    print("[ok] running engine (reuse path) ...", flush=True)
    results = engine.run(reqs)
    torch.cuda.synchronize()
    print(f"[ok] {len(results)} results", flush=True)
    print(f"    prefix_reuse_batches={engine.stats.prefill_prefix_reuse_batches} "
          f"prefix_reuse_requests={engine.stats.prefix_reuse_requests} "
          f"prefill_model_calls={engine.stats.prefill_model_calls}", flush=True)
    for r in results[:3]:
        print(f"    {r.request_id}: {len(r.tokens)} tokens, prefix_hit={r.prefix_hit_tokens}", flush=True)
    print("ENGINE REUSE OK", flush=True)


if __name__ == "__main__":
    main()
