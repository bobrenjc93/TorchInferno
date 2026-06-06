# TorchInferno vs vLLM/sglang — Performance Gap Analysis (Llama-3.1-70B, 8xH100)

Consolidated findings from extended profiling. Benchmark numbers are from
inference-bench run `20260606_201227` (torchinferno commit `25260c0`); local
measurements are from `scripts/test_ttft.py` / `test_long_output.py` against the
current online batcher.

## Headline gaps (cross-benchmark medians)

| Metric | torchinferno | vllm | sglang |
| :-- | --: | --: | --: |
| TTFT median (ms) | 454 | 136 | 134 |
| TPOT median (ms) | 36.6 | 32.4 | 50.7 |
| throughput (tok/s) | 6.5 | 18.1 | 12.5 |

TPOT is already competitive (we beat sglang, ~within 12% of vllm). **The deficit
is TTFT and throughput**, and both decompose into two deep issues below.

## Issue 1 — prefill kernel MFU (~2x), the dominant TTFT gap

few_shot = 64 concurrent x ~640-token prompts (one burst). Local measurement at
HEAD: min TTFT 146ms (≈ vllm), but **median 2135ms** — 63 of 64 requests bunch at
~2.1s. Cause: admission packs ~48 long prompts into one step → a single
~30,720-token prefill. Even at the graph-bucketed level, the (batch=8, q=768)
prefill graph does 6144 tokens in **315ms ≈ 34% MFU**. vllm prefills the same
64x640 with ~145ms median TTFT — roughly **2x better GEMM MFU plus chunking**.

Ruled out as quick fixes (tested locally):
- Lowering `prefill_token_budget` 32768 → 5120: median TTFT unchanged (~2117ms),
  throughput WORSE (557 → 463 tok/s). Total prefill compute is the floor;
  splitting it just adds overhead. **Do not lower the budget.**
- Joint (batch,q) prefill CUDA graphs already remove per-layer launch overhead
  (the 315ms is a graph replay = pure compute at 34% MFU).

Op-level profile (TORCHINFERNO_PROFILE_PREFILL_ONCE, one batched prefill,
torch.profiler sorted by CUDA time) CORRECTS the earlier "raise GEMM MFU"
framing — the sinks are:

| component                       | CUDA time | %    |
| :------------------------------ | --------: | ---: |
| GEMMs (aten::mm, cuBLAS nvjet)  |    91.5ms | 52%  |
| TP allreduce (NCCL ring, 160x)  |    53.9ms | 31%  |
| add_rms_norm                    |    15.4ms |  9%  |
| FlashInfer attention            |     2.0ms | 1.2% |

Attention is NOT the sink (1.2%). Two real levers:
- GEMMs (52%): already cuBLAS; QKV and gate-up are ALREADY fused (one GEMM
  each). The MFU loss is the TP-sharded dims (per-rank K/N smaller). Limited
  headroom without a different parallelism layout.
- Allreduce (31%): runs as ncclDevKernel_AllReduce_Sum_bf16_RING_LL -- plain
  NCCL ring, NOT the symm-mem one-shot allreduce the decode graphs use. On
  8xH100 NVLink, symm-mem one/two-shot is typically faster than ring for these
  sizes. ROUTING PREFILL ALLREDUCE THROUGH SYMM-MEM is the most actionable
  single lever (2 allreduces/layer x 80 layers = 160 calls). This is the next
  data-driven change after feaebc7 (max_active revert) validates.

Also still: true chunked prefill interleaved with decode so early requests
return their first token without waiting for the whole burst.

## Issue 2 — decode is memory-bound; throughput scales with batch

long_output (tiny prompt, huge output, 64 concurrent): TPOT 30.7 vs vllm 15.1,
throughput 22.3 vs 58.3. Decode reads all 70B weights every step, so per-step
time is ~fixed and throughput ∝ concurrent rows. Profiling: decode_model launch
2.3ms, but the blocking `next_token_tensor.cpu().tolist()` exposes ~18ms of
GPU decode per step.

Partial fix already in flight (commit `28f62c1`, UNVALIDATED): `max_active`
48 → 128 decode rows + decode/prefill batch decoupling. Local: 64-concurrent
aggregate throughput 623 → 1204 tok/s; 96-concurrent 1775 tok/s. Needs the
benchmark to confirm.

Remaining: the ~18ms per-step CPU sync is the GPU decode time exposed; cutting
TPOT further needs async overlap (GPU token stash — keep sampled tokens on GPU,
read back lagging) which has been correctness-fragile at scale.

## Issue 3 — prefix reuse is session-limited AND TP-buggy (low ROI)

Reuse would help self_consistency / multi_turn / few_shot / tree_of_thought in
principle, BUT:
- With `persistent=False`, every request burst starts a fresh online session and
  `start_online → _reset_capacity` WIPES the prefix cache. So cross-burst reuse
  (e.g. multi_turn's per-turn requests) NEVER triggers — verified via per-rank
  `TORCHINFERNO_REUSE_DEBUG` traces (`cached_prefixes=0` at every `step=0`).
- Only within-session reuse fires, and that is exactly the case that
  hangs/CUDA-asserts on 8 ranks (a multi-rank collective divergence). The reuse
  LOGIC is verified correct single-GPU (`scripts/debug_reuse_engine.py` PASSES);
  storage is TP-consistent; the divergence is in within-session reuse execution.

So a useful reuse needs BOTH the TP fix AND a persistent engine whose cache
survives bursts (persistent=True previously deadlocked). Larger than the reuse
path itself. Gated OFF (`TORCHINFERNO_CONTINUOUS_FI_REUSE=0`).

## In-flight, validated-locally-but-NOT-benchmarked commits

The inference-bench harness has been frozen on `25260c0` for many hours, so these
are unmeasured against vllm/sglang:
- joint (batch,q) prefill graphs under a token budget,
- `max_active=128` decode batch + prefill/decode decoupling,
- single-request prefill via graph (245 → 51ms single-req TTFT, local),
- paged-cache crash guards (SDPA fallback, FI-eager write-bounds).

## Priority for a focused (non-loop) session

1. Unfreeze the benchmark harness so the in-flight commits get measured.
2. Prefill MFU (Issue 1) — biggest TTFT lever, ~2x, affects 3/5 benchmarks.
3. Chunked prefill interleaved with decode — lets early requests return fast.
4. Persistent engine + TP-safe reuse (Issue 3) — needed for multi_turn.
