"""Paged serving driver: chains FlashInfer-paged prefill + decode over a
LayeredPagedKVCache with admission-by-free-pages.

This is the engine-logic core for migrating llama3-TP serving off the dense
[batch, kv_heads, max_seq, head_dim] cache (which caps concurrency at ~48 rows)
to true paging (memory scales with actual tokens -> far more concurrent rows ->
the queueing-bound TTFT/throughput cells). It composes the validated model methods
forward_prefill_paged / forward_decode_paged and the LayeredPagedKVCache
primitives (reserve/slot_mapping/scatter_write/flashinfer_page_table) into a
prefill->decode generation loop. The OpenAI online-batcher wiring (request queue,
TP command protocol) builds on this; the scheduling is deterministic so it stays
consistent across TP ranks without extra broadcasts.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from torchinferno.runtime.paged import LayeredPagedKVCache


def _plan_decode(flashinfer, cache, request_ids, nqo, nkv, hd, page_size):
    indptr, indices, lpl = cache.flashinfer_page_table(request_ids)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=cache.kv.device)
    dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    dw.plan(indptr=indptr, indices=indices, last_page_len=lpl, num_qo_heads=nqo,
            num_kv_heads=nkv, head_dim=hd, page_size=page_size, q_data_type=cache.kv.dtype)
    return dw


def _plan_prefill(flashinfer, cache, request_ids, lengths, nqo, nkv, hd, page_size):
    indptr, indices, lpl = cache.flashinfer_page_table(request_ids)
    qo_indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=cache.kv.device)
    qo_indptr[1:] = torch.tensor(lengths, dtype=torch.int32, device=cache.kv.device).cumsum(0)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=cache.kv.device)
    pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    pw.plan(qo_indptr=qo_indptr, paged_kv_indptr=indptr, paged_kv_indices=indices,
            paged_kv_last_page_len=lpl, num_qo_heads=nqo, num_kv_heads=nkv,
            head_dim_qk=hd, page_size=page_size, causal=True, q_data_type=cache.kv.dtype)
    return pw


def generate_paged(
    model: object,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    page_size: int = 16,
    cache: LayeredPagedKVCache | None = None,
) -> list[list[int]]:
    """Greedy paged generation: prefill a uniform-length prompt batch into a page
    pool, then decode max_new_tokens steps, returning the generated token ids per
    request. Validates the chained prefill->decode paged path; the production
    online batcher adds variable arrival/admission + the TP protocol on top.
    """
    import flashinfer

    if not prompts:
        return []
    T = len(prompts[0])
    if any(len(p) != T for p in prompts):
        raise ValueError("generate_paged currently expects uniform-length prompts")
    batch = len(prompts)
    dev = model.device
    layer0 = model.layers[0]
    nqo, nkv, hd = layer0.local_attention_heads, layer0.local_key_value_heads, model.config.head_dim
    rids = [f"g{i}" for i in range(batch)]
    max_seq = T + max_new_tokens

    if cache is None:
        pages_per = math.ceil(max_seq / page_size)
        cache = LayeredPagedKVCache(
            num_layers=len(model.layers), num_pages=batch * pages_per + 8, page_size=page_size,
            num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
        )
    for rid in rids:
        cache.reserve(rid, max_seq)
        cache._sequences[rid].length = T

    out: list[list[int]] = [[] for _ in range(batch)]
    with torch.inference_mode():
        # prefill: populate pages for the prompt, predict the first new token
        pw = _plan_prefill(flashinfer, cache, rids, [T] * batch, nqo, nkv, hd, page_size)
        logits = model.forward_prefill_paged(
            torch.tensor(prompts, dtype=torch.long, device=dev), cache,
            request_ids=rids, prefill_wrapper=pw,
        )  # [batch, T, vocab]
        tok = logits[:, -1, :].argmax(-1)  # [batch]
        for i in range(batch):
            out[i].append(int(tok[i]))

        # decode loop: each generated token occupies the next position
        for step in range(max_new_tokens - 1):
            for rid in rids:
                cache._sequences[rid].length = T + step + 1
            dw = _plan_decode(flashinfer, cache, rids, nqo, nkv, hd, page_size)
            positions = torch.full((batch,), T + step, device=dev)
            logits = model.forward_decode_paged(
                tok.view(batch, 1), cache, request_ids=rids,
                positions=positions, decode_wrapper=dw,
            )  # [batch, 1, vocab]
            tok = logits[:, -1, :].argmax(-1)
            for i in range(batch):
                out[i].append(int(tok[i]))
    return out
