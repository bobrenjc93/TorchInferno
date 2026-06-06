#!/usr/bin/env python3
"""Synchronous single-GPU repro of the FlashInfer prefix-reuse primitives.

Run with: CUDA_LAUNCH_BLOCKING=1 TORCHINFERNO_COMPILED_UNIFIED_FORWARD=0 \
          PYTHONPATH=src python scripts/debug_reuse.py

Reproduces, without the 70B server or distributed init:
  1) full-prompt prefill into row A (forward_step_flashinfer, seq_lens=0)
  2) copy row A's KV into a prefix row P (copy_prefix_from)
  3) copy prefix row P into a fresh active row B (the reuse copy)
  4) suffix prefill on row B with seq_lens=prefix_len (the reuse forward)
So any IndexKernel out-of-bounds asserts synchronously at the exact op.
"""

import tempfile
from pathlib import Path

import torch

from torchinferno.models.llama3 import (
    Llama3TensorParallelForCausalLM,
    Llama3V0ForCausalLM,
    tiny_llama3_config,
)
from torchinferno.variant_validation import _write_llama3_hf_checkpoint


def step(msg):
    torch.cuda.synchronize()
    print(f"[ok] {msg}", flush=True)


def main():
    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    # head_dim must be 64/128/256 for FlashInfer SM90 prefill; use 128 (= 70B).
    cfg = tiny_llama3_config(
        vocab_size=128, hidden_size=512, num_attention_heads=4,
        num_key_value_heads=1, max_position_embeddings=512,
    )
    eager = Llama3V0ForCausalLM(cfg).to(dev, dt).eval()
    with tempfile.TemporaryDirectory() as tmp:
        _write_llama3_hf_checkpoint(eager, Path(tmp))
        model = Llama3TensorParallelForCausalLM.from_pretrained(Path(tmp), dtype="bfloat16").eval()
    step("model built")

    max_seq = 512
    n_rows = 8
    cache = model.allocate_cache(n_rows, max_seq, cache_backend="flashinfer")
    step(f"cache allocated rows={n_rows} max_seq={max_seq}")
    vocab = cfg.vocab_size

    def prefill(row, tokens, seq_start):
        b, qlen = 1, len(tokens)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=dev)
        seq_lens = torch.tensor([seq_start], dtype=torch.long, device=dev)
        q_lens = torch.tensor([qlen], dtype=torch.long, device=dev)
        wpos = torch.arange(seq_start, seq_start + qlen, device=dev, dtype=torch.long).unsqueeze(0)
        lpos = torch.tensor([qlen - 1], dtype=torch.long, device=dev)
        ris = torch.tensor([row], dtype=torch.long, device=dev)
        with torch.inference_mode():
            return model.forward_step_flashinfer(
                input_ids, cache, seq_lens=seq_lens, q_lens=q_lens,
                write_positions=wpos, logit_positions=lpos, row_indices=ris,
            )

    prompt = [i % vocab for i in range(40)]
    prefix_len = 40

    # 1) full prefill into row 0
    _ = prefill(0, prompt, 0)
    step("1) full prefill row0 seq_lens=0")
    for lc in cache.layers:
        if hasattr(lc, "_seq_lens"):
            lc._seq_lens[0] = prefix_len

    # 2) copy row0 -> prefix row 6
    cache.copy_prefix_from(cache, prefix_len, source_row=0, dest_row=6)
    step("2) copy_prefix_from row0 -> row6 (prefix store)")

    # 3) copy prefix row6 -> fresh active row 1 (the reuse copy)
    cache.copy_prefix_from(cache, prefix_len, source_row=6, dest_row=1)
    step("3) copy_prefix_from row6 -> row1 (reuse copy)")
    for lc in cache.layers:
        if hasattr(lc, "_seq_lens"):
            lc._seq_lens[1] = prefix_len

    # 4) suffix prefill on row1 with seq_lens=prefix_len (the reuse forward)
    suffix = [(i + 7) % vocab for i in range(10)]
    _ = prefill(1, suffix, prefix_len)
    step("4) suffix prefill row1 seq_lens=prefix_len  <-- reuse forward")

    print("ALL REUSE PRIMITIVES OK", flush=True)


if __name__ == "__main__":
    main()
