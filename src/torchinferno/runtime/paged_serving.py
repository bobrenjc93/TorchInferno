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
        self._pending: list[tuple] = []  # (request_id, prompt, max_new, eos)
        self._active: list[dict] = []
        self._next_rid = 0
        self._step_no = 0
        self._prefill_ws = None       # persistent prefill workspace (avoid per-call 128MB alloc)
        self._prefill_wrapper = None  # persistent prefill wrapper (re-plan, no realloc)

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

    def _prefill_batch(self, rids, prompts):
        # Batched (uniform-length) prefill: ONE forward_prefill_paged for all rids
        # of the same prompt length T, instead of one call per request -- the fix for
        # the sequential-prefill TTFT bottleneck. qo_indptr = [0,T,2T,...,B*T].
        import flashinfer

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

    def submit(self, request_id, prompt: list[int], max_new_tokens: int, eos_token_id=None) -> None:
        self._pending.append((request_id, list(prompt), max_new_tokens, eos_token_id))

    def has_work(self) -> bool:
        return bool(self._pending or self._active)

    def step(self) -> list[tuple]:
        """One continuous-batching iteration. Returns [(request_id, token, finished)]."""
        import flashinfer

        events: list[tuple] = []
        with torch.inference_mode():
            # admit new arrivals (reserve pages), then BATCHED prefill grouped by
            # prompt length (forward_prefill_paged is uniform-length, so one call per
            # length-group instead of one per request -- avoids serializing a herd,
            # which dominated TTFT). Deterministic across TP ranks (same pending +
            # free pages -> same admits + grouping -> matching collectives).
            admitted: list[dict] = []
            while len(self._active) + len(admitted) < self.max_active and self._pending:
                ext_id, prompt, max_new, eos = self._pending[0]
                need = math.ceil((len(prompt) + max_new) / self.page_size)
                if len(self.cache.free_pages) < need:
                    break
                self._pending.pop(0)
                rid = f"p{self._next_rid}"
                self._next_rid += 1
                self.cache.reserve(rid, len(prompt) + max_new)
                self.cache._sequences[rid].length = len(prompt)
                admitted.append({"ext": ext_id, "rid": rid, "prompt": prompt, "max_new": max_new, "eos": eos})
            if admitted:
                groups: dict[int, list[dict]] = {}
                for a in admitted:
                    groups.setdefault(len(a["prompt"]), []).append(a)
                for _T, grp in groups.items():
                    rids = [a["rid"] for a in grp]
                    prompts = [a["prompt"] for a in grp]
                    logits = self._prefill_batch(rids, prompts)  # [B, T, vocab]
                    toks = _greedy_tokens(self.model, logits[:, -1, :])  # [B]
                    for i, a in enumerate(grp):
                        tok = int(toks[i])
                        fin = 1 >= a["max_new"] or tok == a["eos"]
                        events.append((a["ext"], tok, fin))
                        if fin:
                            self.cache.free(a["rid"])
                        else:
                            self._active.append({
                                "ext": a["ext"], "rid": a["rid"], "plen": len(a["prompt"]),
                                "gen": [tok], "max_new": a["max_new"], "last": tok, "eos": a["eos"],
                            })
            if not self._active:
                return events
            # decode the active set one token
            for a in self._active:
                self.cache._sequences[a["rid"]].length = a["plen"] + len(a["gen"])
            tokens = torch.tensor([[a["last"]] for a in self._active], dtype=torch.long, device=self.dev)
            positions = torch.tensor([a["plen"] + len(a["gen"]) - 1 for a in self._active], dtype=torch.long, device=self.dev)
            rids_active = [a["rid"] for a in self._active]
            if self.runner is not None:
                logits = self.runner.step(tokens, positions, rids_active)
            else:
                dw = _plan_decode(flashinfer, self.cache, rids_active, self.nqo, self.nkv, self.hd, self.page_size)
                logits = self.model.forward_decode_paged(
                    tokens, self.cache, request_ids=rids_active, positions=positions, decode_wrapper=dw)
            nxt = _greedy_tokens(self.model, logits[:, -1, :])
            still = []
            for i, a in enumerate(self._active):
                t = int(nxt[i])
                a["gen"].append(t)
                a["last"] = t
                fin = len(a["gen"]) >= a["max_new"] or t == a["eos"]
                events.append((a["ext"], t, fin))
                if fin:
                    self.cache.free(a["rid"])
                else:
                    still.append(a)
            self._active = still
        return events

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
        self._next_rid = 0
        self._step_no = 0

    def submit_online(self, request) -> None:
        self.submit(request.request_id, list(request.prompt), request.max_new_tokens, request.eos_token_id)

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
