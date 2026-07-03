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

import time
from dataclasses import dataclass, field
import math

import torch

from torchinferno.runtime.options import env_flag, env_int
from torchinferno.runtime.paged import LayeredPagedKVCache, PagedSequence
from torchinferno.runtime.prefix import PrefixAwareRouter


@dataclass
class PagedEngineStats:
    prefill_model_calls: int = 0
    prefill_batches: int = 0
    prefill_tokens: int = 0
    prefill_admitted_requests: int = 0
    prefill_plain_batches: int = 0
    prefill_prefix_reuse_batches: int = 0
    prefill_wall_ms: float = 0.0
    prefill_forward_ms: float = 0.0
    decode_model_calls: int = 0
    decode_batches: int = 0
    decode_tokens: int = 0
    decode_active_tokens: int = 0
    prefix_reuse_requests: int = 0
    prefix_reuse_tokens: int = 0
    queued_requests: int = 0
    scheduler_steps: int = 0
    max_model_batch_size: int = 0
    prefill_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_wall_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_forward_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_active_requests: dict[str, int] = field(default_factory=dict)
    prefill_shape_model_rows: dict[str, int] = field(default_factory=dict)
    prefill_shape_active_tokens: dict[str, int] = field(default_factory=dict)
    prefill_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    decode_shape_counts: dict[str, int] = field(default_factory=dict)
    decode_shape_model_ms: dict[str, float] = field(default_factory=dict)
    prefix_reuse_route_counts: dict[str, int] = field(default_factory=dict)
    prefix_reuse_hit_token_counts: dict[str, int] = field(default_factory=dict)


def _add_count(mapping: dict[str, int], key: str, value: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + int(value)


def _add_time(mapping: dict[str, float], key: str, value: float) -> None:
    mapping[key] = float(mapping.get(key, 0.0)) + float(value)


class PagedPrefixCache:
    """Zero-copy block-level prefix cache over a LayeredPagedKVCache (vllm-style).

    remember(rid, tokens) RETAINS a completed sequence's page-aligned KV pages (holds
    an extra refcount so the cache's free() won't release them) and indexes its tokens
    in a radix router. share_into(new_rid, tokens) finds the longest cached
    PAGE-ALIGNED prefix of the new prompt and shares those pages ZERO-COPY into the new
    request (refcount++), returning the shared token count -- so the engine prefills
    ONLY the suffix. LRU-bounded; eviction releases the retained refs (and rebuilds the
    router, which has no remove). This is the cross-request KV reuse that closes the
    multi_turn TTFT gap (turn k shares turns 1..k-1's pages instead of re-prefilling).
    Collective-safe on TP: every rank runs the same deterministic remember/match on the
    same broadcast tokens -> identical share decisions -> identical page tables.
    """

    def __init__(self, cache: LayeredPagedKVCache, *, capacity: int) -> None:
        self.cache = cache
        self.page_size = cache.page_size
        self.capacity = max(0, int(capacity))
        self._router = PrefixAwareRouter(default_route=None)
        self._entries: dict[str, tuple[tuple[int, ...], list[int]]] = {}
        self._lru: list[str] = []

    def remember(self, request_id: str, tokens) -> None:
        if self.capacity <= 0:
            return
        seq = self.cache._sequences.get(request_id)
        if seq is None:
            if request_id in self._entries:
                self._touch(request_id)
            return
        if request_id in self._entries:
            self._drop_entry(request_id)
        full_pages = min(len(seq.page_ids), len(tokens) // self.page_size)
        if full_pages <= 0:
            return
        keep = list(seq.page_ids[:full_pages])
        for pid in keep:
            self.cache.retain_page(pid)
        toks = tuple(int(t) for t in tokens[: full_pages * self.page_size])
        self._entries[request_id] = (toks, keep)
        self._index_entry(request_id, toks)
        self._lru.append(request_id)
        self._evict()

    def _index_entry(self, request_id: str, toks: tuple[int, ...]) -> None:
        # Insert a route endpoint at EVERY page boundary so a query that shares only a
        # LEADING run of pages (then diverges) still matches that run -- the radix
        # router matches an inserted sequence only when it is a full prefix of the
        # query. A boundary shared by two sequences maps to whichever was indexed last;
        # that is correct because identical leading tokens have identical KV, so either
        # sequence's pages for that prefix are interchangeable.
        for p in range(1, len(toks) // self.page_size + 1):
            self._router.add_prefix(toks[: p * self.page_size], request_id)

    def share_into(self, new_request_id: str, tokens) -> int:
        if not self._entries:
            return 0
        match = self._router.route(tokens)
        rid = match.route_id
        if rid is None or rid not in self._entries:
            return 0
        depth = (match.depth // self.page_size) * self.page_size
        if depth < self.page_size:
            return 0
        _toks, ent_pages = self._entries[rid]
        n_pages = depth // self.page_size
        dst = self.cache._sequences.setdefault(new_request_id, PagedSequence(new_request_id))
        if dst.page_ids:
            return 0  # target must be fresh
        for pid in ent_pages[:n_pages]:
            dst.page_ids.append(pid)
            self.cache.retain_page(pid)
        dst.length = max(dst.length, depth)
        self._touch(rid)
        return depth

    def _touch(self, request_id: str) -> None:
        if request_id in self._lru:
            self._lru.remove(request_id)
            self._lru.append(request_id)

    def _drop_entry(self, request_id: str) -> None:
        _toks, pages = self._entries.pop(request_id)
        if request_id in self._lru:
            self._lru.remove(request_id)
        for pid in pages:
            self.cache.release_page_ref(pid)
        self._rebuild_router()

    def _evict(self) -> None:
        while len(self._lru) > self.capacity:
            self._drop_entry(self._lru[0])

    def _rebuild_router(self) -> None:
        # PrefixAwareRouter has no remove(); rebuild from the surviving entries
        # (eviction is rare and capacity is bounded).
        self._router = PrefixAwareRouter(default_route=None)
        for rid, (toks, _pages) in self._entries.items():
            self._index_entry(rid, toks)


def _greedy_tokens(model, logits_2d):
    """Greedy next-token from [batch, vocab] logits. On TP the logits are VOCAB-
    SHARDED, so a plain argmax gives a per-rank LOCAL index -> every rank picks a
    different token -> desync/garbage. Use the model's TP-aware greedy sampler
    (all-reduce to the consistent GLOBAL token) when sharded; plain argmax for
    world_size==1 (full vocab)."""
    if getattr(model, "world_size", 1) > 1 and hasattr(model, "_sample_next_token_greedy"):
        return model._sample_next_token_greedy(logits_2d)
    return logits_2d.argmax(-1)


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

    def _pad(self, tokens, positions, request_ids):
        # Pad an ACTIVE set (active <= batch) up to the captured batch by repeating
        # the last active row -- the dummy rows redo a valid (already-reserved)
        # request's work, so their slots/page-table are valid and their output is
        # simply ignored (step() slices to active). This lets ONE captured graph
        # serve any active count <= batch, the continuous-batching requirement.
        active = len(request_ids)
        if active == self.batch:
            return tokens, positions, request_ids, active
        if active == 0 or active > self.batch:
            raise ValueError(f"active {active} must be in 1..{self.batch}")
        pad = self.batch - active
        tokens_b = torch.cat([tokens.view(active, 1), tokens.view(active, 1)[-1:].expand(pad, 1)], 0)
        positions_b = torch.cat([positions, positions[-1:].expand(pad)], 0)
        request_ids_b = list(request_ids) + [request_ids[-1]] * pad
        return tokens_b, positions_b, request_ids_b, active

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
        """Replay the captured graph for the current step; returns logits[:active].

        Accepts an ACTIVE set of any size 1..batch (padded internally to the
        captured batch). ALWAYS replays (even right after capture): the output
        buffer is only valid after a replay, not from the capture-time execution --
        the same pattern the dense decode warmup uses. (Reading self.out straight
        from capture silently diverges from eager.)
        """
        tokens, positions, request_ids, active = self._pad(tokens, positions, request_ids)
        if self.graph is None:
            self.capture(tokens, positions, request_ids)
        self._fill(tokens, positions, request_ids)
        self._plan(request_ids)
        self.graph.replay()
        return self.out[:active]


class PagedPrefillGraphRunner:
    """CUDA-graphed paged prefill at a fixed (batch, T) bucket.

    Captures ONE CUDA graph around model.forward_prefill_paged (on-device block_table
    slots -- no host sync in the captured region) for a fixed batch size and prompt
    length T, then replays it for any same-shape fresh prefill. Eliminates the
    ~245ms/call eager TP launch overhead (80 layers x per-layer allreduce, not
    graph-amortized) -- the multi_turn TTFT killer. The FlashInfer prefill page table
    (which pages each request occupies) varies per call, so it lives in static
    buffers the wrapper re-plans OUTSIDE the graph before each replay; the captured
    region runs only the embedding + 80-layer stack + lm_head. Mirrors
    PagedDecodeGraphRunner (decode) and capture_flashinfer_prefill_graph (dense).
    """

    def __init__(self, model, cache, *, batch, T, workspace_bytes=128 * 1024 * 1024):
        import flashinfer

        self.model = model
        self.cache = cache
        self.batch = batch
        self.T = T
        dev = cache.kv.device
        self.dev = dev
        layer0 = model.layers[0]
        self.nqo = layer0.local_attention_heads
        self.nkv = layer0.local_key_value_heads
        self.hd = model.config.head_dim
        self.page_size = cache.page_size
        self.pages_per = math.ceil(T / self.page_size)
        # static input buffers (graph reads these by address)
        self.s_ids = torch.zeros(batch, T, dtype=torch.long, device=dev)
        self.s_bt = torch.zeros(batch, self.pages_per, dtype=torch.long, device=dev)
        # FlashInfer CUDAGraph prefill wrapper + its static page-table buffers
        self._ws = torch.empty(workspace_bytes, dtype=torch.uint8, device=dev)
        self._qo = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        self._kv_indptr = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        self._kv_indices = torch.empty(batch * self.pages_per, dtype=torch.int32, device=dev)
        self._kv_lpl = torch.empty(batch, dtype=torch.int32, device=dev)
        self.pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._ws, kv_layout="NHD", use_cuda_graph=True,
            qo_indptr_buf=self._qo, paged_kv_indptr_buf=self._kv_indptr,
            paged_kv_indices_buf=self._kv_indices, paged_kv_last_page_len_buf=self._kv_lpl,
        )
        self.graph = None
        self.out = None

    def _plan(self, request_ids):
        # Pass FRESH page-table tensors to plan(); the CUDAGraph wrapper copies them
        # into the fixed buffers it was constructed with (same as the decode runner).
        indptr, indices, lpl = self.cache.flashinfer_page_table(request_ids)
        qo_indptr = torch.zeros(self.batch + 1, dtype=torch.int32, device=self.dev)
        qo_indptr[1:] = torch.full(
            (self.batch,), self.T, dtype=torch.int32, device=self.dev
        ).cumsum(0)
        self.pw.plan(
            qo_indptr=qo_indptr, paged_kv_indptr=indptr, paged_kv_indices=indices,
            paged_kv_last_page_len=lpl, num_qo_heads=self.nqo, num_kv_heads=self.nkv,
            head_dim_qk=self.hd, page_size=self.page_size, causal=True,
            q_data_type=self.cache.kv.dtype,
        )

    def _fill(self, input_ids, request_ids):
        self.s_ids.copy_(input_ids.view(self.batch, self.T))
        self.s_bt.zero_()
        # Build the block table from the FIRST pages_per pages of each request: a
        # fresh [batch, T] prefill writes positions 0..T-1, which span exactly
        # ceil(T/page_size)=pages_per pages, even when the request reserved more
        # pages for future decode (real serving reserves len(prompt)+max_new).
        for i, rid in enumerate(request_ids):
            pids = self.cache._sequences[rid].page_ids[: self.pages_per]
            if pids:
                self.s_bt[i, : len(pids)].copy_(
                    torch.tensor(pids, dtype=torch.long, device=self.dev)
                )

    def capture(self, input_ids, request_ids):
        self._fill(input_ids, request_ids)
        self._plan(request_ids)
        self.model.forward_prefill_paged(
            self.s_ids, self.cache, request_ids=request_ids,
            prefill_wrapper=self.pw, block_table=self.s_bt,
        )
        torch.cuda.synchronize(self.dev)
        stream = torch.cuda.Stream(device=self.dev)
        stream.wait_stream(torch.cuda.current_stream(self.dev))
        self._plan(request_ids)
        with torch.cuda.stream(stream):
            self.model.forward_prefill_paged(
                self.s_ids, self.cache, request_ids=request_ids,
                prefill_wrapper=self.pw, block_table=self.s_bt,
            )
        torch.cuda.current_stream(self.dev).wait_stream(stream)
        torch.cuda.synchronize(self.dev)
        self.graph = torch.cuda.CUDAGraph()
        self._plan(request_ids)
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = self.model.forward_prefill_paged(
                self.s_ids, self.cache, request_ids=request_ids,
                prefill_wrapper=self.pw, block_table=self.s_bt,
            )
        self._stream = stream

    def step(self, input_ids, request_ids):
        """Replay the captured prefill graph; returns logits [batch, T, vocab].

        ALWAYS replays (even right after capture): the output buffer is valid only
        after a replay -- the same pattern PagedDecodeGraphRunner uses.
        """
        if self.graph is None:
            self.capture(input_ids, request_ids)
        self._fill(input_ids, request_ids)
        self._plan(request_ids)
        self.graph.replay()
        return self.out


class PagedSpecGraphRunner:
    """CUDA-graphed SPECULATIVE-decode verify step: forward_prefill_paged of T=1+K
    tokens per request at PER-REQUEST start positions (the requests are desynced --
    each accepted a different number of drafts last step). Mirrors PagedPrefillGraphRunner
    but (a) feeds a [batch] start_position tensor, (b) uses the FULL block table (the T
    query tokens attend to the whole prefix via the paged-kv page table -- causal aligns
    them as the last T of each request's kv). Eager spec is ~220ms (80-layer launch floor);
    graphed is ~29ms -> with ~3x fewer steps a ~2x decode win on echo workloads. The page
    table + start grow each step (context extends) -> static buffers re-planned/refilled
    OUTSIDE the graph, captured region = embedding + 80-layer stack + lm_head.
    """

    def __init__(self, model, cache, *, batch, T, max_pages, workspace_bytes=128 * 1024 * 1024):
        import flashinfer

        self.model = model
        self.cache = cache
        self.batch = batch
        self.T = T
        self.max_pages = max_pages
        dev = cache.kv.device
        self.dev = dev
        layer0 = model.layers[0]
        self.nqo = layer0.local_attention_heads
        self.nkv = layer0.local_key_value_heads
        self.hd = model.config.head_dim
        self.page_size = cache.page_size
        self.s_ids = torch.zeros(batch, T, dtype=torch.long, device=dev)
        self.s_start = torch.zeros(batch, dtype=torch.long, device=dev)
        self.s_bt = torch.zeros(batch, max_pages, dtype=torch.long, device=dev)
        self._ws = torch.empty(workspace_bytes, dtype=torch.uint8, device=dev)
        self._qo = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        self._kv_indptr = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        self._kv_indices = torch.empty(batch * max_pages, dtype=torch.int32, device=dev)
        self._kv_lpl = torch.empty(batch, dtype=torch.int32, device=dev)
        self.pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._ws, kv_layout="NHD", use_cuda_graph=True,
            qo_indptr_buf=self._qo, paged_kv_indptr_buf=self._kv_indptr,
            paged_kv_indices_buf=self._kv_indices, paged_kv_last_page_len_buf=self._kv_lpl,
        )
        self.graph = None
        self.out = None

    def _plan(self, request_ids, max_kv=False):
        # max_kv=True (CAPTURE only): plan the attention schedule for each request's MAX
        # reserved kv (all allocated pages, last page full) so the captured kernel covers
        # ALL of growing context; step() re-plans with the ACTUAL (smaller) length. Without
        # this the kernel is sized for the small capture-time kv -> wrong as context grows
        # (the FlashInfer prefill CUDAGraph wrapper, unlike the decode one, bakes the kv
        # schedule at capture).
        if max_kv:
            saved = {}
            for rid in request_ids:
                seq = self.cache._sequences[rid]
                if rid not in saved:
                    saved[rid] = seq.length
                seq.length = len(seq.page_ids) * self.page_size
        indptr, indices, lpl = self.cache.flashinfer_page_table(request_ids)
        if max_kv:
            for rid, length in saved.items():
                self.cache._sequences[rid].length = length
        qo_indptr = torch.zeros(self.batch + 1, dtype=torch.int32, device=self.dev)
        qo_indptr[1:] = torch.full((self.batch,), self.T, dtype=torch.int32, device=self.dev).cumsum(0)
        self.pw.plan(
            qo_indptr=qo_indptr, paged_kv_indptr=indptr, paged_kv_indices=indices,
            paged_kv_last_page_len=lpl, num_qo_heads=self.nqo, num_kv_heads=self.nkv,
            head_dim_qk=self.hd, page_size=self.page_size, causal=True,
            q_data_type=self.cache.kv.dtype,
        )

    def _pad(self, input_ids, starts, request_ids):
        active = len(request_ids)
        if active == self.batch:
            return input_ids, starts, request_ids, active
        if active == 0 or active > self.batch:
            raise ValueError(f"active {active} must be in 1..{self.batch}")
        pad = self.batch - active
        ids_b = torch.cat([input_ids.view(active, self.T), input_ids.view(active, self.T)[-1:].expand(pad, self.T)], 0)
        starts_b = torch.cat([starts, starts[-1:].expand(pad)], 0)
        rids_b = list(request_ids) + [request_ids[-1]] * pad
        return ids_b, starts_b, rids_b, active

    def _fill(self, input_ids, starts, request_ids):
        self.s_ids.copy_(input_ids.view(self.batch, self.T))
        self.s_start.copy_(starts)
        self.s_bt.zero_()
        bt = self.cache.block_table(request_ids, max_pages=self.max_pages)
        self.s_bt[:, : bt.size(1)].copy_(bt)

    def capture(self, input_ids, starts, request_ids):
        # capture the kernel sized for MAX kv (max_kv=True) so growing context is covered.
        self._fill(input_ids, starts, request_ids)
        self._plan(request_ids, max_kv=True)
        fwd = lambda: self.model.forward_prefill_paged(
            self.s_ids, self.cache, request_ids=request_ids,
            prefill_wrapper=self.pw, block_table=self.s_bt, start_position=self.s_start)
        fwd()
        torch.cuda.synchronize(self.dev)
        stream = torch.cuda.Stream(device=self.dev)
        stream.wait_stream(torch.cuda.current_stream(self.dev))
        self._plan(request_ids, max_kv=True)
        with torch.cuda.stream(stream):
            fwd()
        torch.cuda.current_stream(self.dev).wait_stream(stream)
        torch.cuda.synchronize(self.dev)
        self.graph = torch.cuda.CUDAGraph()
        self._plan(request_ids, max_kv=True)
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = fwd()
        self._stream = stream

    def step(self, input_ids, starts, request_ids):
        input_ids, starts, request_ids, active = self._pad(input_ids, starts, request_ids)
        if self.graph is None:
            self.capture(input_ids, starts, request_ids)
        self._fill(input_ids, starts, request_ids)
        self._plan(request_ids)
        self.graph.replay()
        return self.out[:active]


def generate_paged_continuous(
    model: object,
    requests: list[tuple[list[int], int]],
    *,
    page_size: int = 16,
    max_active: int = 8,
    use_graph: bool = True,
) -> list[list[int]]:
    """Continuous-batching paged greedy generation -- the full paged serving loop.

    requests: list of (prompt_token_ids, max_new_tokens). Admits up to max_active
    concurrent requests bounded by FREE PAGES (no fixed row cap), prefills each
    arrival (batch=1, eager), decodes the active set via a graphed
    PagedDecodeGraphRunner (padded to max_active), and FREES a request's pages on
    completion so a waiting one can be admitted. Returns generated tokens per
    request in input order. This composes every validated paged piece into the
    engine logic the OpenAI online batcher will wrap (queue + TP protocol on top).
    """
    import flashinfer

    if not requests:
        return []
    dev = model.device
    layer0 = model.layers[0]
    nqo, nkv, hd = layer0.local_attention_heads, layer0.local_key_value_heads, model.config.head_dim
    max_seq = max(len(p) + n for p, n in requests)
    pages_per = math.ceil(max_seq / page_size)
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=max_active * pages_per + 8, page_size=page_size,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    runner = PagedDecodeGraphRunner(model, cache, batch=max_active, max_pages=pages_per) if use_graph else None

    results: list[list[int] | None] = [None] * len(requests)
    pending = list(range(len(requests)))
    active: list[dict] = []
    next_rid = 0

    with torch.inference_mode():
        while pending or active:
            # admit by free pages (replaces the dense 48-row cap)
            while len(active) < max_active and pending:
                idx = pending[0]
                prompt, max_new = requests[idx]
                need = math.ceil((len(prompt) + max_new) / page_size)
                if len(cache.free_pages) < need:
                    break
                pending.pop(0)
                rid = f"r{next_rid}"
                next_rid += 1
                cache.reserve(rid, len(prompt) + max_new)
                cache._sequences[rid].length = len(prompt)
                pw = _plan_prefill(flashinfer, cache, [rid], [len(prompt)], nqo, nkv, hd, page_size)
                logits = model.forward_prefill_paged(
                    torch.tensor([prompt], dtype=torch.long, device=dev), cache,
                    request_ids=[rid], prefill_wrapper=pw,
                )
                tok = int(_greedy_tokens(model, logits[:, -1, :])[0])
                active.append({"idx": idx, "rid": rid, "plen": len(prompt), "gen": [tok], "max_new": max_new, "last": tok})
            # retire finished requests (free their pages)
            finished = [a for a in active if len(a["gen"]) >= a["max_new"]]
            for a in finished:
                results[a["idx"]] = a["gen"]
                cache.free(a["rid"])
            active = [a for a in active if len(a["gen"]) < a["max_new"]]
            if not active:
                continue
            # graphed paged decode of the active set (one token each)
            for a in active:
                cache._sequences[a["rid"]].length = a["plen"] + len(a["gen"])
            tokens = torch.tensor([[a["last"]] for a in active], dtype=torch.long, device=dev)
            positions = torch.tensor([a["plen"] + len(a["gen"]) - 1 for a in active], dtype=torch.long, device=dev)
            rids_active = [a["rid"] for a in active]
            if runner is not None:
                logits = runner.step(tokens, positions, rids_active)
            else:
                dw = _plan_decode(flashinfer, cache, rids_active, nqo, nkv, hd, page_size)
                logits = model.forward_decode_paged(
                    tokens, cache, request_ids=rids_active, positions=positions, decode_wrapper=dw)
            nxt = _greedy_tokens(model, logits[:, -1, :])
            for i, a in enumerate(active):
                t = int(nxt[i])
                a["gen"].append(t)
                a["last"] = t
    return [r if r is not None else [] for r in results]


class PagedEngine:
    """Incremental (submit/step) paged continuous-batching engine -- the bridge the
    OpenAI online batcher needs (it streams per-token events, so it drives a stepper,
    not a batch run()). One step() = one iteration of the validated
    generate_paged_continuous loop: admit+prefill new arrivals by free pages, decode
    the active set one token via the graphed runner, retire finished (freeing pages).
    Each step returns (request_id, token, finished) events to stream. Same logic as
    generate_paged_continuous, so it inherits its bf16-vs-dense correctness; a
    step-driven run is token-identical to the batch driver (tests).
    """

    def __init__(self, model, *, page_size=16, max_active=8, max_seq=2048, use_graph=True):
        self.model = model
        self.page_size = page_size
        self.max_active = max_active
        self.max_seq = max_seq  # exposed for persistent-engine fit/rebuild checks
        dev = model.device
        self.dev = dev
        layer0 = model.layers[0]
        self.nqo = layer0.local_attention_heads
        self.nkv = layer0.local_key_value_heads
        self.hd = model.config.head_dim
        pages_per = math.ceil(max_seq / page_size)
        self.pages_per = pages_per
        self.cache = LayeredPagedKVCache(
            num_layers=len(model.layers), num_pages=max_active * pages_per + 8, page_size=page_size,
            num_key_value_heads=self.nkv, head_dim=self.hd, device=dev, dtype=model.dtype,
        )
        self.runner = (
            PagedDecodeGraphRunner(model, self.cache, batch=max_active, max_pages=pages_per)
            if use_graph else None
        )
        self._pending: list[tuple] = []  # (request_id, prompt, max_new, eos, stop_ids)
        self._active: list[dict] = []
        self._next_rid = 0
        self._step_no = 0
        self._prefill_ws = None       # persistent prefill workspace (avoid per-call 128MB alloc)
        self._prefill_wrapper = None  # persistent prefill wrapper (re-plan, no realloc)
        # CUDA-graphed prefill: cache one PagedPrefillGraphRunner per (batch, T) shape.
        # DEFAULT-OFF: local A/B (real 70B TP8, VARLEN ~1500-tok 32-conc PIPELINE)
        # showed default-on is a REGRESSION on the varied-length high-concurrency load
        # (= multi_turn): TTFT 8.7s->21.2s, tput 144->66. Cause: exact-T keying + inline
        # capture means ~16 distinct shapes each capture (~500ms) BLOCKING in the
        # serving loop while concurrent requests queue (capture-thrash). The graph IS
        # 1.5-2.5x when shapes RECUR (few distinct lengths), but varied-length multi_turn
        # never reaches steady reuse. Needs T-BUCKETING (round T to coarse buckets + pad,
        # logits read at real_len-1) before it can be default-on.
        self._use_prefill_graph = use_graph and env_flag("TORCHINFERNO_PAGED_PREFILL_GRAPH", False)
        self._prefill_runners: dict[tuple, PagedPrefillGraphRunner] = {}
        self._prefill_runner_cap = env_int("TORCHINFERNO_PAGED_PREFILL_GRAPH_MAX", 16, minimum=1)
        self._prefill_graph_max_t = env_int("TORCHINFERNO_PAGED_PREFILL_GRAPH_MAX_T", 4096, minimum=1)
        self.stats = PagedEngineStats(max_model_batch_size=max_active)
        # Zero-copy COW prefix cache (default-off until end-to-end validated): a new
        # request shares the longest cached page-aligned prefix and prefills only its
        # suffix (the multi_turn TTFT lever -- reuse prior turns' KV). Gated +
        # capacity-bounded. Persistence across bursts is increment 4.
        self.prefix_cache = (
            PagedPrefixCache(self.cache, capacity=env_int("TORCHINFERNO_PAGED_PREFIX_CACHE_CAP", 256, minimum=1))
            if env_flag("TORCHINFERNO_PAGED_PREFIX_CACHE", False) else None
        )

    def _prefill(self, rid, prompt):
        # Persistent-workspace prefill: re-plan the cached BatchPrefill wrapper per
        # request instead of allocating a fresh 128MB workspace + wrapper each time
        # (the per-call alloc dominated TTFT at concurrency). Returns last-token logits.
        import flashinfer

        if self._prefill_ws is None:
            self._prefill_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.dev)
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._prefill_ws, kv_layout="NHD")
        T = len(prompt)
        indptr, indices, lpl = self.cache.flashinfer_page_table([rid])
        qo = torch.zeros(2, dtype=torch.int32, device=self.dev)
        qo[1] = T
        self._prefill_wrapper.plan(
            qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=indices, paged_kv_last_page_len=lpl,
            num_qo_heads=self.nqo, num_kv_heads=self.nkv, head_dim_qk=self.hd, page_size=self.page_size,
            causal=True, q_data_type=self.cache.kv.dtype,
        )
        return self.model.forward_prefill_paged(
            torch.tensor([prompt], dtype=torch.long, device=self.dev), self.cache,
            request_ids=[rid], prefill_wrapper=self._prefill_wrapper,
        )

    def _graphed_prefill(self, rids, prompts):
        # Replay (or capture-then-replay) a CUDA-graphed prefill for this exact
        # (batch, T) shape. Drop-in for forward_prefill_paged: same [B,T] input,
        # same [B,T,vocab] output, so the caller's logits[:, -1, :] is unchanged.
        # Exact-T keying means no padding -> the last-token logit position is correct
        # by construction (no garbage-token risk). NO-EVICT policy: once the cache is
        # full a NEW shape returns None so the caller stays EAGER -- this bounds total
        # captures + memory and guarantees no capture-thrash regression vs all-eager
        # if the workload cycles through many shapes.
        B, T = len(rids), len(prompts[0])
        key = (B, T)
        runner = self._prefill_runners.get(key)
        if runner is None:
            if len(self._prefill_runners) >= self._prefill_runner_cap:
                return None
            runner = PagedPrefillGraphRunner(self.model, self.cache, batch=B, T=T)
            self._prefill_runners[key] = runner
        input_ids = torch.tensor(prompts, dtype=torch.long, device=self.dev)
        return runner.step(input_ids, rids)

    def _prefill_batch(self, rids, prompts):
        # Batched (uniform-length) prefill: ONE forward_prefill_paged for all rids
        # of the same prompt length T, instead of one call per request -- the fix for
        # the sequential-prefill TTFT bottleneck. qo_indptr = [0,T,2T,...,B*T].
        import flashinfer

        # Graphed fast path for the LAUNCH-BOUND regime (small batch): the eager TP
        # prefill is ~2.5x slower here (per-layer allreduces not amortized). Large
        # batches are compute-bound, so the graph is skipped (B>4) -- it would only
        # add capture cost. Same-length group only (uniform T).
        T0 = len(prompts[0])
        if (
            self._use_prefill_graph
            and len(rids) <= 4
            and T0 <= self._prefill_graph_max_t
            and all(len(p) == T0 for p in prompts)
        ):
            graphed = self._graphed_prefill(rids, prompts)
            if graphed is not None:
                return graphed

        if len(rids) == 1:
            return self._prefill(rids[0], prompts[0])
        if self._prefill_ws is None:
            self._prefill_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.dev)
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._prefill_ws, kv_layout="NHD")
        B, T = len(rids), len(prompts[0])
        indptr, indices, lpl = self.cache.flashinfer_page_table(rids)
        qo = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=self.dev)
        self._prefill_wrapper.plan(
            qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=indices, paged_kv_last_page_len=lpl,
            num_qo_heads=self.nqo, num_kv_heads=self.nkv, head_dim_qk=self.hd, page_size=self.page_size,
            causal=True, q_data_type=self.cache.kv.dtype,
        )
        return self.model.forward_prefill_paged(
            torch.tensor(prompts, dtype=torch.long, device=self.dev), self.cache,
            request_ids=rids, prefill_wrapper=self._prefill_wrapper,
        )

    def _prefill_suffix(self, rid, prompt, shared):
        return self._prefill_suffix_batch([rid], [prompt], [shared])

    def _prefill_suffix_batch(self, rids, prompts, shared_tokens):
        # COW suffix prefill: rid already holds the shared prefix pages (positions
        # 0..shared-1 KV present); prefill ONLY prompt[shared:] at start_position=shared
        # over the FULL page table (prefix + suffix pages), causal -> attention reads
        # the shared prefix for free. Returns [1, suffix_len, vocab]. Mirrors the
        # GPU-validated scripts/validate_paged_suffix_prefill.py path.
        import flashinfer

        if self._prefill_ws is None:
            self._prefill_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.dev)
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._prefill_ws, kv_layout="NHD")
        if len(rids) != len(prompts) or len(rids) != len(shared_tokens):
            raise ValueError("paged suffix prefill batch inputs must have matching lengths")
        suffixes = [
            list(prompt[int(shared):])
            for prompt, shared in zip(prompts, shared_tokens)
        ]
        suffix_lens = {len(suffix) for suffix in suffixes}
        if not suffix_lens or min(suffix_lens) <= 0:
            raise ValueError("paged suffix prefill requires non-empty suffixes")
        if len(suffix_lens) != 1:
            raise ValueError("paged suffix prefill batch requires uniform suffix length")
        suffix_len = suffix_lens.pop()
        indptr, indices, lpl = self.cache.flashinfer_page_table(list(rids))
        qo = torch.arange(0, (len(rids) + 1) * suffix_len, suffix_len, dtype=torch.int32, device=self.dev)
        self._prefill_wrapper.plan(
            qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=indices, paged_kv_last_page_len=lpl,
            num_qo_heads=self.nqo, num_kv_heads=self.nkv, head_dim_qk=self.hd, page_size=self.page_size,
            causal=True, q_data_type=self.cache.kv.dtype,
        )
        starts = [int(shared) for shared in shared_tokens]
        start_position = starts[0] if len(set(starts)) == 1 else torch.tensor(starts, dtype=torch.long, device=self.dev)
        return self.model.forward_prefill_paged(
            torch.tensor(suffixes, dtype=torch.long, device=self.dev), self.cache,
            request_ids=list(rids), prefill_wrapper=self._prefill_wrapper, start_position=start_position,
        )

    def submit(
        self,
        request_id,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id=None,
        stop_token_ids=(),
    ) -> None:
        stop_ids = {int(token_id) for token_id in stop_token_ids if int(token_id) >= 0}
        if eos_token_id is not None and int(eos_token_id) >= 0:
            stop_ids.add(int(eos_token_id))
        self._pending.append((request_id, list(prompt), max_new_tokens, eos_token_id, tuple(sorted(stop_ids))))
        self.stats.queued_requests += 1

    def _allocate_request_id(self) -> str:
        rid = f"p{self._next_rid}"
        self._next_rid += 1
        return rid

    def _record_prefill_stats(
        self,
        shape_key: str,
        *,
        batch: int,
        tokens_per_request: int,
        elapsed_ms: float,
        prefix_reuse: bool,
    ) -> None:
        model_tokens = int(batch) * int(tokens_per_request)
        self.stats.prefill_model_calls += 1
        self.stats.prefill_batches += 1
        self.stats.prefill_tokens += model_tokens
        self.stats.prefill_wall_ms += elapsed_ms
        self.stats.prefill_forward_ms += elapsed_ms
        if prefix_reuse:
            self.stats.prefill_prefix_reuse_batches += 1
        else:
            self.stats.prefill_plain_batches += 1
        _add_count(self.stats.prefill_shape_counts, shape_key)
        _add_time(self.stats.prefill_shape_wall_ms, shape_key, elapsed_ms)
        _add_time(self.stats.prefill_shape_forward_ms, shape_key, elapsed_ms)
        _add_count(self.stats.prefill_shape_active_requests, shape_key, batch)
        _add_count(self.stats.prefill_shape_model_rows, shape_key, batch)
        _add_count(self.stats.prefill_shape_active_tokens, shape_key, model_tokens)
        _add_count(self.stats.prefill_shape_model_tokens, shape_key, model_tokens)

    def _record_decode_stats(self, *, batch: int, elapsed_ms: float) -> None:
        shape_key = f"paged_decode:b{int(batch)}"
        self.stats.decode_model_calls += 1
        self.stats.decode_batches += 1
        self.stats.decode_tokens += int(batch)
        self.stats.decode_active_tokens += int(batch)
        _add_count(self.stats.decode_shape_counts, shape_key)
        _add_time(self.stats.decode_shape_model_ms, shape_key, elapsed_ms)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)

    def _ensure_decode_runner_capacity(self, request_ids: list[str]) -> None:
        if self.runner is None:
            return
        required_pages = self._required_pages_for(request_ids)
        if required_pages <= self.runner.max_pages:
            return
        target_pages = max(required_pages, self.runner.max_pages * 2)
        self.runner = PagedDecodeGraphRunner(
            self.model,
            self.cache,
            batch=self.max_active,
            max_pages=target_pages,
        )

    def _required_pages_for(self, request_ids: list[str]) -> int:
        return max(
            (len(self.cache._sequences[rid].page_ids) for rid in request_ids),
            default=1,
        )

    def _ensure_spec_runner_capacity(self, request_ids: list[str], T: int) -> object:
        required_pages = self._required_pages_for(request_ids)
        runner = getattr(self, "_spec_runner", None)
        if runner is not None and runner.T == T and required_pages <= runner.max_pages:
            return runner
        base_pages = math.ceil(self.max_seq / self.page_size) + 1
        previous_pages = runner.max_pages if runner is not None and runner.T == T else 0
        target_pages = max(required_pages, base_pages, previous_pages * 2)
        runner = PagedSpecGraphRunner(
            self.model,
            self.cache,
            batch=self.max_active,
            T=T,
            max_pages=target_pages,
        )
        self._spec_runner = runner
        return runner

    def step(self) -> list[tuple]:
        """One continuous-batching iteration. Returns [(request_id, token, finished)]."""
        import flashinfer

        events: list[tuple] = []
        with torch.inference_mode():
            self.stats.scheduler_steps += 1
            # admit new arrivals (reserve pages), then BATCHED prefill grouped by
            # prompt length (forward_prefill_paged is uniform-length, so one call per
            # length-group instead of one per request -- avoids serializing a herd,
            # which dominated TTFT). Deterministic across TP ranks (same pending +
            # free pages -> same admits + grouping -> matching collectives).
            admitted: list[dict] = []
            while len(self._active) + len(admitted) < self.max_active and self._pending:
                ext_id, prompt, max_new, eos, stop_ids = self._pending[0]
                need = math.ceil((len(prompt) + max_new) / self.page_size)
                if len(self.cache.free_pages) < need:
                    break
                self._pending.pop(0)
                rid = self._allocate_request_id()
                # COW: share the longest cached page-aligned prefix (zero-copy) so we
                # prefill only the suffix. Deterministic across TP ranks (same cache +
                # tokens -> same share), keeping per-request prefill collectives aligned.
                shared = self.prefix_cache.share_into(rid, prompt) if self.prefix_cache is not None else 0
                if shared >= len(prompt):  # whole prompt cached -> keep >=1 token to prefill
                    shared = max(0, (len(prompt) - 1) // self.page_size * self.page_size)
                self.cache.reserve(rid, len(prompt) + max_new)
                self.cache._sequences[rid].length = len(prompt)
                admitted.append({"ext": ext_id, "rid": rid, "prompt": prompt,
                                 "max_new": max_new, "eos": eos, "stop": stop_ids, "shared": shared})
            if admitted:
                self.stats.prefill_admitted_requests += len(admitted)
                for a in admitted:
                    shared = int(a["shared"])
                    if shared > 0:
                        self.stats.prefix_reuse_requests += 1
                        self.stats.prefix_reuse_tokens += shared
                        _add_count(self.stats.prefix_reuse_route_counts, "paged_prefix")
                        _add_count(self.stats.prefix_reuse_hit_token_counts, str(shared))

            def _retire_or_activate(a, tok):
                fin = 1 >= a["max_new"] or tok in a["stop"]
                events.append((a["ext"], tok, fin))
                if fin:
                    if self.prefix_cache is not None:
                        self.prefix_cache.remember(a["rid"], list(a["prompt"]) + [tok])
                    self.cache.free(a["rid"])
                else:
                    self._active.append({
                        "ext": a["ext"], "rid": a["rid"], "plen": len(a["prompt"]),
                        "prompt": a["prompt"], "gen": [tok], "max_new": a["max_new"],
                        "last": tok, "eos": a["eos"], "stop": a["stop"],
                    })

            if admitted:
                # COW-shared requests prefill ONLY their suffix. Group uniform
                # suffix lengths so a turn wave reuses prefix pages without
                # serializing one FlashInfer prefill per request.
                shared_groups: dict[int, list[dict]] = {}
                for a in (x for x in admitted if x["shared"] > 0):
                    shared_groups.setdefault(len(a["prompt"]) - int(a["shared"]), []).append(a)
                for _suffix_len, grp in shared_groups.items():
                    rids = [a["rid"] for a in grp]
                    prompts = [a["prompt"] for a in grp]
                    starts = [int(a["shared"]) for a in grp]
                    start_s = time.perf_counter()
                    logits = self._prefill_suffix_batch(rids, prompts, starts)
                    elapsed_ms = (time.perf_counter() - start_s) * 1000.0
                    shape_key = (
                        f"paged_prefix:b{len(rids)}:s{_suffix_len}:"
                        f"p{min(starts)}-{max(starts)}"
                    )
                    self._record_prefill_stats(
                        shape_key,
                        batch=len(rids),
                        tokens_per_request=_suffix_len,
                        elapsed_ms=elapsed_ms,
                        prefix_reuse=True,
                    )
                    toks = _greedy_tokens(self.model, logits[:, -1, :])
                    for i, a in enumerate(grp):
                        _retire_or_activate(a, int(toks[i]))
                plain = [x for x in admitted if x["shared"] == 0]
                groups: dict[int, list[dict]] = {}
                for a in plain:
                    groups.setdefault(len(a["prompt"]), []).append(a)
                for _T, grp in groups.items():
                    rids = [a["rid"] for a in grp]
                    prompts = [a["prompt"] for a in grp]
                    start_s = time.perf_counter()
                    logits = self._prefill_batch(rids, prompts)  # [B, T, vocab]
                    elapsed_ms = (time.perf_counter() - start_s) * 1000.0
                    self._record_prefill_stats(
                        f"paged_prefill:b{len(rids)}:t{_T}",
                        batch=len(rids),
                        tokens_per_request=_T,
                        elapsed_ms=elapsed_ms,
                        prefix_reuse=False,
                    )
                    toks = _greedy_tokens(self.model, logits[:, -1, :])  # [B]
                    for i, a in enumerate(grp):
                        _retire_or_activate(a, int(toks[i]))
            if not self._active:
                return events
            if env_flag("TORCHINFERNO_PAGED_SPEC_DECODE", False):
                self._decode_spec(events)
                return events
            # decode the active set one token
            for a in self._active:
                self.cache._sequences[a["rid"]].length = a["plen"] + len(a["gen"])
            tokens = torch.tensor([[a["last"]] for a in self._active], dtype=torch.long, device=self.dev)
            positions = torch.tensor([a["plen"] + len(a["gen"]) - 1 for a in self._active], dtype=torch.long, device=self.dev)
            rids_active = [a["rid"] for a in self._active]
            if self.runner is not None:
                self._ensure_decode_runner_capacity(rids_active)
                start_s = time.perf_counter()
                logits = self.runner.step(tokens, positions, rids_active)
            else:
                dw = _plan_decode(flashinfer, self.cache, rids_active, self.nqo, self.nkv, self.hd, self.page_size)
                start_s = time.perf_counter()
                logits = self.model.forward_decode_paged(
                    tokens, self.cache, request_ids=rids_active, positions=positions, decode_wrapper=dw)
            self._record_decode_stats(
                batch=len(rids_active),
                elapsed_ms=(time.perf_counter() - start_s) * 1000.0,
            )
            nxt = _greedy_tokens(self.model, logits[:, -1, :])
            still = []
            for i, a in enumerate(self._active):
                t = int(nxt[i])
                a["gen"].append(t)
                a["last"] = t
                fin = len(a["gen"]) >= a["max_new"] or t in a["stop"]
                events.append((a["ext"], t, fin))
                if fin:
                    # Remember the full prompt+generated sequence so a later request
                    # (e.g. the next conversation turn) can share this turn's KV pages.
                    if self.prefix_cache is not None and "prompt" in a:
                        self.prefix_cache.remember(a["rid"], list(a["prompt"]) + list(a["gen"]))
                    self.cache.free(a["rid"])
                else:
                    still.append(a)
            self._active = still
        return events

    @staticmethod
    def _propose_ngram(seq: list[int], ng: int, k: int) -> list[int]:
        # Prompt-lookup proposal: find the most recent earlier occurrence of the last
        # `ng` tokens, propose the `k` tokens that followed it (pad to k with 0 = a
        # non-match the verifier rejects -> 0 accepted, normal decode). No draft model.
        if len(seq) < ng:
            return [0] * k
        last = tuple(seq[-ng:])
        for i in range(len(seq) - ng - 1, -1, -1):
            if tuple(seq[i:i + ng]) == last:
                m = seq[i + ng:i + ng + k]
                return list(m) + [0] * (k - len(m))
        return [0] * k

    def _decode_spec(self, events: list[tuple]) -> None:
        # Batched prompt-lookup speculative decode of the active set. Each request
        # proposes K tokens (n-gram), we forward [N, 1+K] (last_token + props) at the
        # per-request position (start_position tensor), verify per-row (argmax chain),
        # accept the matching prefix, and emit accepted+1 tokens in ONE forward.
        # Greedy-EXACT (verification) -- validate vs the 1-token path. Default OFF.
        import flashinfer

        act = self._active
        n = len(act)
        k = env_int("TORCHINFERNO_PAGED_SPEC_K", 8, minimum=1)
        ng = env_int("TORCHINFERNO_PAGED_SPEC_NGRAM", 3, minimum=1)
        props = [self._propose_ngram(list(a["prompt"]) + list(a["gen"]), ng, k) for a in act]
        starts = [a["plen"] + len(a["gen"]) - 1 for a in act]
        inp = [[act[i]["last"]] + props[i] for i in range(n)]  # [N, 1+K]
        rids = [a["rid"] for a in act]
        for i, a in enumerate(act):
            self.cache.reserve(a["rid"], starts[i] + 1 + k)
            self.cache._sequences[a["rid"]].length = starts[i] + 1 + k
        inp_t = torch.tensor(inp, dtype=torch.long, device=self.dev)
        starts_t = torch.tensor(starts, dtype=torch.long, device=self.dev)
        # GRAPHED spec verify (PagedSpecGraphRunner) is the wall-win: eager forward is
        # ~220ms (80-layer launch floor) vs ~29ms graphed. Eager fallback for A/B / no-graph.
        # GRAPH path (default ON): PagedSpecGraphRunner runs ~29ms vs eager ~165ms (eager
        # is launch-bound). VALIDATED greedy-EXACT: graph==eager argmax for all requests over
        # 60 steps incl padding to max_active (dbg_div). [The earlier "MATCH=False" was spec-
        # verify-vs-baseline-decode near-tie flips -- the inherent spec numerical property,
        # not a graph bug.] capture-at-max sizes the kernel for the full kv (growing context).
        if self.runner is not None and env_flag("TORCHINFERNO_PAGED_SPEC_GRAPH", True):
            sr = self._ensure_spec_runner_capacity(rids, 1 + k)
            out = sr.step(inp_t, starts_t, rids)  # [n, 1+k, vocab]
        else:
            bt = self.cache.block_table(rids)
            pw = _plan_prefill(flashinfer, self.cache, rids, [1 + k] * n, self.nqo, self.nkv, self.hd, self.page_size)
            out = self.model.forward_prefill_paged(
                inp_t, self.cache, request_ids=rids, prefill_wrapper=pw, block_table=bt, start_position=starts_t)
        # TP-aware greedy: logits are VOCAB-SHARDED across ranks, so a plain argmax gives
        # a per-rank LOCAL index -> divergent tokens -> collective DESYNC. Use
        # _greedy_tokens (all-reduces to the global token) for ALL N*(1+K) positions.
        flat = _greedy_tokens(self.model, out.reshape(-1, out.size(-1)).float())  # [N*(1+K)]
        preds = flat.reshape(n, -1)  # [N, 1+K] global argmax
        still = []
        for i, a in enumerate(act):
            p = [int(x) for x in preds[i]]
            prop = props[i]
            emit = [p[0]]  # the real next token (always)
            j = 0
            while j < k and prop[j] == p[j]:  # draft j matched -> p[j+1] is the next real token
                emit.append(p[j + 1])
                j += 1
            finished = False
            for t in emit:
                a["gen"].append(t)
                a["last"] = t
                fin = len(a["gen"]) >= a["max_new"] or t in a["stop"]
                events.append((a["ext"], t, fin))
                if fin:
                    finished = True
                    break
            # KV valid up to the second-to-last emitted token; the last emitted token's
            # KV (written at a rejected/unverified slot) is rewritten next step. Set
            # length = plen + gen - 1 (mirrors the validated single-request prototype).
            self.cache._sequences[a["rid"]].length = a["plen"] + len(a["gen"]) - 1
            if finished:
                if self.prefix_cache is not None and "prompt" in a:
                    self.prefix_cache.remember(a["rid"], list(a["prompt"]) + list(a["gen"]))
                self.cache.free(a["rid"])
            else:
                still.append(a)
        self._active = still

    # --- ContinuousBatchEngine-compatible interface (drop-in for the OpenAI online
    # batcher, which drives start_online / submit_online / has_online_work /
    # step_online -> ServingTokenEvent). Lets the batcher swap the dense engine for
    # this paged one behind a flag without changing its queue/streaming/TP loop. ---

    def start_online(self, *, max_seq_len: int, external_cache=None) -> None:
        # Reset per-run state. The cache/runner are sized at construction (max_seq);
        # an external paged cache may be adopted (e.g. a persistent TP-shared pool).
        if external_cache is not None:
            self.cache = external_cache
            if self.runner is not None:
                self.runner.cache = external_cache
        self._pending = []
        self._active = []
        if self.prefix_cache is None:
            self._next_rid = 0
        self._step_no = 0
        self._spec_runner = None  # rebuilt lazily (cache may have been swapped above)

    def submit_online(self, request) -> None:
        self.submit(
            request.request_id,
            list(request.prompt),
            request.max_new_tokens,
            request.eos_token_id,
            request.stop_token_ids,
        )

    def has_online_work(self) -> bool:
        return self.has_work()

    def step_online(self) -> list:
        from torchinferno.runtime.serving import ServingTokenEvent

        raw = self.step()
        gen_count: dict = getattr(self, "_gen_count", {})
        self._gen_count = gen_count
        events = []
        for ext_id, tok, fin in raw:
            gen_count[ext_id] = gen_count.get(ext_id, 0) + 1
            events.append(ServingTokenEvent(
                request_id=str(ext_id), token=int(tok), step=self._step_no,
                generated=gen_count[ext_id], finished=bool(fin),
            ))
        self._step_no += 1
        return events


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
        tok = _greedy_tokens(model, logits[:, -1, :])  # [batch]
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
            tok = _greedy_tokens(model, logits[:, -1, :])
            for i in range(batch):
                out[i].append(int(tok[i]))
    return out
