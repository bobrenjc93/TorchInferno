# Paged KV Integration Plan (llama3 TP serving)

Status: foundation built + fully de-risked this session; integration NOT yet wired.
This is the actionable, file-level plan for the focused effort.

## Why (the one lever for the most cells)

Profiling (docs/PERF_GAP_ANALYSIS.md) showed the multi_turn/long_output TPOT gaps
AND the TTFT/throughput rows (3-9x, ~13 of the 18 lost cells) are all
QUEUEING-bound: the dense per-layer cache caps concurrent rows at ~48 for long
contexts, so 64-125 client-concurrent benchmarks queue heavily, which inflates
TTFT and (via prefill-interleaving) TPOT. Decode itself is GEMM-bound and
weight-bound (flat to 256 rows), and paged decode ATTENTION is sub-linear at high
concurrency (bench_paged_decode_concurrency.py: 48->512 rows @2048 ctx = 3.95x
attention, ~5% of the step). So packing more concurrent rows via paging is a pure
win with no hidden wall.

## Key structural insight (makes this tractable, not a rewrite)

`Llama3TensorParallelLayerKVCache.paged_kv` is ALREADY in FlashInfer NHD paged
format: `[batch, 2, max_seq_len, kv_heads, head_dim]` -- i.e. "dense-as-paged"
(batch pages, each of size max_seq). The FI decode (`wrapper.run(q, cache.paged_kv)`)
and the CUDA-graph path (serving.py ~2940, decode_runner.py) ALREADY consume the
paged format with the page table (`indptr`/`indices`/`last_page_len`) as graph
inputs. So the consumer side is DONE. The migration is: swap the dense-as-paged
buffer for a real small-page pool + block tables + admission-by-pages.

## Foundation already in place (this session, tested)

- `runtime/paged.py::LayeredPagedKVCache` -- multi-layer pool, ONE block table per
  request shared across layers, per-layer NHD `[num_pages,2,page_size,kv_heads,head_dim]`,
  `reserve/extend/write_layer/layer_kv/flashinfer_page_table/free`.
- Validated FlashInfer-correct vs dense SDPA
  (tests/test_scaffolding.py::test_layered_paged_kv_cache_flashinfer_decode_matches_dense).
- Concurrency-viable + no CUDA-graph blocker (CUDAGraphBatchDecodeWithPagedKVCacheWrapper).

## Integration steps (ordered; flag-gated, default dense, validate each on 8xH100)

1. CACHE: add a paged mode to the llama3 TP cache (or a sibling class) backed by a
   per-layer page pool `[num_pages, 2, page_size, kv_heads, head_dim]` (small
   page_size, e.g. 16) + a shared block table per row. `num_pages` from a KV-token
   budget (TP-sharded: local kv_heads). Keep the dense path as default behind a flag
   (TORCHINFERNO_OPENAI_TP_ONLINE_PAGED_KV).
2. KV WRITE: replace `paged_kv[row, :, seq_len:end]` writes (model.py ~563 and the
   ragged-append paths) with block-table scatter (extend the row's pages, write to
   the new page slots). Reuse LayeredPagedKVCache.write_layer logic.
3. PAGE TABLE per step: build `(indptr, indices, last_page_len)` from the active
   rows' block tables (LayeredPagedKVCache.flashinfer_page_table) and feed the FI
   plan -- replacing the current (arange, row_indices, seq_len+1, page_size=max_seq).
   Keep it GPU-resident / minimal-CPU; it's shared across all 80 layers per step.
4. GRAPH BUFFERS: the captured decode graph's `indices` buffer must size to
   max-total-pages (active_rows * max_pages_per_row) instead of `bs`; update before
   replay. indptr/last_page_len already updated per replay.
5. ADMISSION: change `_admit_ready_requests` (runtime/serving.py) to admit by FREE
   PAGES (cache.free_pages >= ceil(ctx/page_size)) instead of the row cap. This is
   what unlocks high concurrency for short/growing contexts. The shipped KV-bounded
   concurrency is the dense-cache approximation of this; paged makes it exact +
   long-context-capable.
6. PREFILL: prefill also writes KV -> page-allocate the prompt span, write suffix to
   pages (ragged prefill graph already exists; route its KV write through pages).
7. EVICTION/preemption when the pool is full (start simple: reject admission when
   no free pages, like today's row cap; add preemption later).

## Validation (self, on 8xH100, before default-on)

- Correctness: greedy output of paged path == dense path on a few prompts.
- Concurrency: closed-loop long_output/multi_turn-like load -> confirm many more
  concurrent rows admitted, TTFT down, throughput up, TPOT not regressed beyond the
  allreduce-scaling (which is the only batch-scaling cost; GEMMs flat).
- Then default-on; the live benchmark scores TTFT/E2E/throughput across
  multi_turn/long_output/tree/few_shot/self_consistency.

## Risk control

Flag-gated (default dense) -> zero risk to the benchmarked path until validated.
Each step above is independently testable. The hard part is step 4 (graph buffer
sizing for variable page counts); if it stalls, an eager paged decode path
validates 1-3+5 correctness first, then add the graph.

## UPDATE 2026-06-07: existing paged backend (oracle) + FlashInfer-paged decode built

DISCOVERY: the model ALREADY has a true-paged backend --
`allocate_cache(cache_backend="paged", page_size=N)` ->
`PagedLlama3TensorParallelLayerKVCache` (uses PagedKVCache + torch-SDPA
`append_and_attend_ragged`), validated against dense by
test_llama3_tensor_parallel_paged_cache_matches_dense_forward. BUT it uses torch
SDPA (no FlashInfer, no decode graph) -> slow decode, and serving uses the
FlashInfer dense-as-paged backend instead. So it's a correctness ORACLE + proves
paged prefill/decode plumbing, but is not the fast serving path.

PROGRESS (branch `paged-kv-decode-wiring`, commit eeddd9e): built + validated the
FlashInfer-paged decode -- `Llama3TensorParallelForCausalLM.forward_decode_paged`
(slot_mapping+scatter_write write, layer_kv+flashinfer_page_table read), matches the
dense full-forward reference on a tiny model
(test_llama3_tensor_parallel_forward_decode_paged_matches_dense). This is the
FAST (FlashInfer) true-paged decode the serving needs, distinct from the existing
torch-SDPA "paged" backend.

REMAINING (branch): serving wiring -- allocate the NHD LayeredPagedKVCache pool at
start, admission-by-free-pages (replace the 48-row cap), per-step
flashinfer_page_table + wrapper plan, paged prefill (can mirror the existing paged
backend's prefill or extend the ragged prefill), dynamic page-tables as
decode-graph inputs. Then merge, flag-gated default-dense, 8xH100 validate, default-on.

## UPDATE 2026-06-07 (later): MODEL-SIDE PAGED FORWARD COMPLETE + VALIDATED

Branch `paged-kv-decode-wiring` now has BOTH halves of the FlashInfer-paged model
forward, each validated vs a dense reference on a real tiny model:
- forward_decode_paged (commit eeddd9e) -- test_..._forward_decode_paged_matches_dense
- forward_prefill_paged (commit 0d924a8) -- test_..._forward_prefill_paged_matches_dense
  (note: FlashInfer's PREFILL hopper kernel requires head_dim in {64,128,256}; the
   DECODE wrapper accepts smaller -- the prefill test uses a head_dim=64 config.)

Both use slot_mapping()+scatter_write() to write the NHD pool and a paged FlashInfer
wrapper (decode/prefill) over layer_kv(); the GEMM/rope/norm flow is identical to the
dense path. So paged KV is de-risked end to end at the model level: pool/block-table,
write (CPU+GPU validated), read (FlashInfer-correct), and now the FULL forward
(prefill+decode match dense on a real model). An independent torch-SDPA "paged"
backend remains as a cross-check oracle.

REMAINING = serving-engine wiring ONLY (runtime/serving.py + openai_server.py):
1. allocate a TP-sharded LayeredPagedKVCache pool at serving start (size by KV-token
   budget; small page_size).
2. online-batcher step: build page table once/step (flashinfer_page_table) + plan the
   decode/prefill wrappers; call forward_prefill_paged on admit, forward_decode_paged
   on step; extend()/reserve() per sequence; free() on finish.
3. admission by FREE PAGES (replace the 48-row cap) -- the actual concurrency win.
4. decode CUDA graph with the page table as graph inputs (CUDAGraphBatchDecode...
   already supports it; size the indices buffer to max-total-pages).
This needs the live 8xH100 server to validate (correctness + concurrency); the
model-side groundwork above makes it assembly, not new algorithms. Merge flag-gated
default-dense, validate on 8xH100, then default-on.

## UPDATE 2026-06-07 (later): forward_decode_paged VALIDATED ON THE REAL 70B

scripts/bench_paged_serving_70b.py (torchrun, branch) loaded the actual
Llama-3.1-70B TP8 and ran forward_decode_paged through a LayeredPagedKVCache:
paged decode logits match the dense forward -- max|d|=0.125 rel=0.0118 (bf16-level).
So the FlashInfer-paged decode is now validated on the REAL model, not just a tiny
one. (Decode throughput vs concurrency is in the same script but needs free GPUs --
a co-tenant 8-GPU job took the machine; the concurrency win is already kernel-benched
in bench_paged_decode_concurrency.py.) The model-side paged forward is now
real-model-validated; only the serving-engine wiring remains.

## UPDATE 2026-06-07 (later): real-70B THROUGHPUT win demonstrated (8.7x)

scripts/bench_paged_serving_70b.py completed on the real Llama-3.1-70B TP8:
- correctness: paged decode == dense, rel=0.0118 (bf16-level).
- decode throughput vs concurrency (paged):
    N= 48: 330  tok/s (145.7 ms/step)   <- the dense cache's ~48-row cap
    N=128: 817  tok/s (156.6 ms/step)   2.5x
    N=256: 1698 tok/s (150.8 ms/step)   5.1x
    N=512: 2880 tok/s (177.8 ms/step)   8.7x
So 10.7x concurrency -> 8.7x decode throughput with step time only +22% (nearly
flat = weight-bound, as predicted). The dense cache caps at ~48 rows; paging unlocks
512 long-context rows at ~8.7x throughput. This is the capstone validation of the
paged-KV thesis ON THE REAL MODEL -- the queueing-bound throughput/TTFT cells
(long_output throughput 20 vs vllm 58, multi_turn/tree TTFT, etc.) are exactly what
this unlocks. The model-side paged forward is fully validated (correctness + the
concurrency win) on the actual 70B; only the serving-engine wiring remains.
