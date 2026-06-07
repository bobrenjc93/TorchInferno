#!/usr/bin/env python3
"""Is paged-vs-dense token divergence a STRUCTURAL bug or bf16 greedy noise?

Compares LOGITS (not argmax) of the paged prefill->decode path vs dense, step by
step, on the real 70B. If per-step rel is ~1% (bf16 level, matching the known
forward_decode_paged rel=1.2%), the token divergence is just greedy argmax flipping
on close logits -- the paged path is correct. If rel is large, it's a real bug.

  torchrun ... --nproc-per-node 8 scripts/cmp_paged_dense_logits.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import _plan_decode, _plan_prefill

PAGE = 16


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    import flashinfer

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")
    nqo = model.layers[0].local_attention_heads
    nkv = model.layers[0].local_key_value_heads
    hd = model.config.head_dim

    prompt = [1, 5, 9, 13, 2, 6]
    steps = 5
    rid = "r"
    max_seq = len(prompt) + steps + 2
    pages_per = (max_seq + PAGE - 1) // PAGE
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=pages_per + 4, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    cache.reserve(rid, max_seq)
    cache._sequences[rid].length = len(prompt)

    with torch.inference_mode():
        # paged prefill -> first token (compare last-token logits to dense)
        pw = _plan_prefill(flashinfer, cache, [rid], [len(prompt)], nqo, nkv, hd, PAGE)
        plog = model.forward_prefill_paged(
            torch.tensor([prompt], dtype=torch.long, device=dev), cache, request_ids=[rid], prefill_wrapper=pw)
        p_last = plog[0, -1, :].float()

        c = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        dlog, _ = model.forward(torch.tensor([prompt], dtype=torch.long, device=dev), cache=c, use_cache=True)
        # dense forward returns GATHERED full vocab; paged returns this rank's shard.
        lv = p_last.shape[-1]
        d_last = dlog[0, -1, :].float()[rank * lv:(rank + 1) * lv]
        rel0 = (p_last - d_last).abs().max().item() / d_last.abs().max().item()
        log(rank, f"[prefill last-token] paged vs dense rel={rel0:.4f} "
                  f"argmax_paged={int(p_last.argmax())} argmax_dense_shard={int(d_last.argmax())}")

        # follow the PAGED greedy path; at each step compare paged-decode logits to a
        # dense forward over the SAME paged-chosen sequence (isolates per-step rel).
        tok = int(p_last.argmax())  # this rank's shard argmax (not the real token, but fine for rel)
        # use a deterministic token to advance both identically:
        seq = list(prompt)
        # pick the true next token via dense gathered logits so both advance the same
        tok = int(dlog[0, -1, :].argmax())
        for step in range(steps):
            seq.append(tok)
            pos = len(seq) - 1
            cache._sequences[rid].length = pos + 1
            dw = _plan_decode(flashinfer, cache, [rid], nqo, nkv, hd, PAGE)
            pd = model.forward_decode_paged(
                torch.tensor([[tok]], dtype=torch.long, device=dev), cache,
                request_ids=[rid], positions=torch.tensor([pos], device=dev), decode_wrapper=dw)
            p = pd[0, -1, :].float()
            cc = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
            dl, _ = model.forward(torch.tensor([seq], dtype=torch.long, device=dev), cache=cc, use_cache=True)
            d_full = dl[0, -1, :].float()
            d = d_full[rank * lv:(rank + 1) * lv]
            rel = (p - d).abs().max().item() / d.abs().max().item()
            log(rank, f"  step {step} pos={pos}: paged-decode vs dense rel={rel:.4f} "
                      f"argmax_match={int(p.argmax())==int(d.argmax())}")
            tok = int(d_full.argmax())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
