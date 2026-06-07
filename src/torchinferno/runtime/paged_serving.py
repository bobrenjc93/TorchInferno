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


class PagedDecodeGraphRunner:
    """CUDA-graphed paged decode step over a LayeredPagedKVCache.

    The reusable decode heart of the paged-KV serving path. Captures ONE CUDA graph
    around model.forward_decode_paged (block_table path -- fully on-device, no host
    sync in the captured region) and replays it every step; the static buffers
    (tokens / positions / block_table / FlashInfer page table) are refilled OUTSIDE
    the graph each step and the wrapper re-planned, so growing context (new pages,
    page-boundary crossings) is handled by buffer updates, not recapture. Real-70B:
    graphed paged decode is BIT-IDENTICAL to eager at 18.6 ms/step (8.3x), and
    context-FLAT -- so it beats the dense graphed decode at long context.
    """

    def __init__(self, model, cache, *, batch, max_pages, workspace_bytes=256 * 1024 * 1024):
        import flashinfer

        self.model = model
        self.cache = cache
        self.batch = batch
        self.max_pages = max_pages
        dev = cache.kv.device
        self.dev = dev
        layer0 = model.layers[0]
        self.nqo = layer0.local_attention_heads
        self.nkv = layer0.local_key_value_heads
        self.hd = model.config.head_dim
        self.page_size = cache.page_size
        # static input/output buffers (graph reads/writes these by address)
        self.s_ids = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.s_pos = torch.zeros(batch, dtype=torch.long, device=dev)
        self.s_bt = torch.zeros(batch, max_pages, dtype=torch.long, device=dev)
        # FlashInfer CUDAGraph decode wrapper + its static page-table buffers
        self._ws = torch.empty(workspace_bytes, dtype=torch.uint8, device=dev)
        self._ind = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        self._idx = torch.empty(batch * max_pages, dtype=torch.int32, device=dev)
        self._lp = torch.empty(batch, dtype=torch.int32, device=dev)
        self.dw = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self._ws, self._ind, self._idx, self._lp, kv_layout="NHD"
        )
        self.graph = None
        self.out = None

    def _plan(self, request_ids):
        # Pass FRESH page-table tensors to plan(); the CUDAGraph wrapper copies them
        # into the fixed buffers it was constructed with (matching the dense decode
        # warmup). Pre-copying into those buffers and passing slices back aliases the
        # wrapper's own storage and yields wrong results.
        indptr, indices, last = self.cache.flashinfer_page_table(request_ids)
        self.dw.plan(
            indptr=indptr, indices=indices, last_page_len=last,
            num_qo_heads=self.nqo, num_kv_heads=self.nkv, head_dim=self.hd,
            page_size=self.page_size, q_data_type=self.cache.kv.dtype,
        )

    def _fill(self, tokens, positions, request_ids):
        self.s_ids.copy_(tokens.view(self.batch, 1))
        self.s_pos.copy_(positions)
        self.s_bt.zero_()
        bt = self.cache.block_table(request_ids, max_pages=self.max_pages)
        self.s_bt[:, : bt.size(1)].copy_(bt)

    def capture(self, tokens, positions, request_ids):
        """Fill buffers for the current step, warm up on a side stream, capture."""
        self._fill(tokens, positions, request_ids)
        self._plan(request_ids)
        self.model.forward_decode_paged(
            self.s_ids, self.cache, positions=self.s_pos, decode_wrapper=self.dw, block_table=self.s_bt
        )
        torch.cuda.synchronize(self.dev)
        stream = torch.cuda.Stream(device=self.dev)
        stream.wait_stream(torch.cuda.current_stream(self.dev))
        self._plan(request_ids)
        with torch.cuda.stream(stream):
            self.model.forward_decode_paged(
                self.s_ids, self.cache, positions=self.s_pos, decode_wrapper=self.dw, block_table=self.s_bt
            )
        torch.cuda.current_stream(self.dev).wait_stream(stream)
        torch.cuda.synchronize(self.dev)
        self.graph = torch.cuda.CUDAGraph()
        self._plan(request_ids)
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = self.model.forward_decode_paged(
                self.s_ids, self.cache, positions=self.s_pos, decode_wrapper=self.dw, block_table=self.s_bt
            )
        self._stream = stream

    def step(self, tokens, positions, request_ids):
        """Replay the captured graph for the current step; returns its logits.

        ALWAYS replays (even right after capture): the output buffer is only
        guaranteed valid after a replay, not from the capture-time execution -- the
        same pattern the dense decode warmup uses. (Reading self.out straight from
        capture silently diverges from eager.)
        """
        if self.graph is None:
            self.capture(tokens, positions, request_ids)
        self._fill(tokens, positions, request_ids)
        self._plan(request_ids)
        self.graph.replay()
        return self.out


def generate_paged(
    model: object,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    page_size: int = 16,
    cache: LayeredPagedKVCache | None = None,
    use_graph: bool = False,
) -> list[list[int]]:
    """Greedy paged generation: prefill a uniform-length prompt batch into a page
    pool, then decode max_new_tokens steps, returning the generated token ids per
    request. Validates the chained prefill->decode paged path; the production
    online batcher adds variable arrival/admission + the TP protocol on top.

    use_graph=True drives the decode loop through PagedDecodeGraphRunner (the
    CUDA-graphed paged decode); the token stream must be identical to the eager
    path -- the multi-step (growing context / page-boundary) correctness check.
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
        runner = None
        if use_graph:
            runner = PagedDecodeGraphRunner(
                model, cache, batch=batch, max_pages=math.ceil(max_seq / page_size)
            )
        for step in range(max_new_tokens - 1):
            for rid in rids:
                cache._sequences[rid].length = T + step + 1
            positions = torch.full((batch,), T + step, device=dev)
            if runner is not None:
                logits = runner.step(tok.view(batch, 1), positions, rids)
            else:
                dw = _plan_decode(flashinfer, cache, rids, nqo, nkv, hd, page_size)
                logits = model.forward_decode_paged(
                    tok.view(batch, 1), cache, request_ids=rids,
                    positions=positions, decode_wrapper=dw,
                )  # [batch, 1, vocab]
            tok = logits[:, -1, :].argmax(-1)
            for i in range(batch):
                out[i].append(int(tok[i]))
    return out
