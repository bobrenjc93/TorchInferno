# TorchInferno Roadmap

This document is the working readiness map for the original TorchInferno plan.
The CLI view is [`torchinferno audit`](../src/torchinferno/audit.py).

## Readiness Levels

- `integrated`: working code is exercised by tests and at least one CLI path.
- `bridge`: the execution shape is present, but still uses reference/copy-heavy
  mechanics rather than the final production implementation.
- `reference`: correctness contract exists; optimized implementation is not
  complete.
- `scaffold`: API shape exists, but production execution is future work.
- `simulated`: planner or replay behavior exists without production transport.
- `minimal`: intentionally small harness that covers repeatable local
  experiments without a full production system.
- `optional`: feature is wired behind optional dependencies or hardware paths.
- `experimental`: optional feature is available and still gated from production
  promotion.

> [!NOTE]
> The status vocabulary mirrors [`torchinferno audit`](../src/torchinferno/audit.py).
> Optional backend-provider work reports `experimental` when the current
> provider dependency is installed and `optional` otherwise.

## Current Status

The table separates model families from execution workbenches. A family owns
the torch-native tensor contract and provenance; workbenches own serving,
profiling, scheduling, parallelism, and offline graph optimization around those
contracts. Runtime consumes concrete promoted variants. Compiler, partitioning,
provider search, and promotion happen offline as described in
[`OFFLINE_OPTIMIZATION.md`](OFFLINE_OPTIMIZATION.md).

| Area | Status | Notes |
| --- | --- | --- |
| DSv4 model family | integrated | Compact torch-native decoder-only causal LM, local checkpoint save/load, CLI smoke, tracing, serving, profiling, and conversion compatibility paths exist. |
| DeepSeek-V3.2 model family | integrated | Torch-native config/model/cache/checkpoint path mirrors production tensor contracts. |
| Llama3 model family | reference | Torch-native config plus make_fx v0 and fused v1 variants exist for tiny/full planning and logit validation. |
| Llama3 parallel execution adapters | bridge | Llama 70B config, pipeline and tensor-parallel safetensor loaders/generate paths, torchrun TP compatibility coverage, and optional Triton/CUDA-graph fast paths exist; production scheduler integration remains open. |
| Model provenance variants | reference | DSv4, DeepSeek-V3.2, and Llama3 have make_fx v0 references, traceable model wrappers, v1 variants, and registry lineage. |
| V0-vs-optimized logit validation | integrated | `validate-model-variants` compares tiny make_fx v0 logits against optimized variants with a 1% default tolerance and optional JSON reports. |
| Text IO and known-logit validation | integrated | Auto model loading, tokenizer-backed generation, `capture-logits`, and `validate-logits` cover checkpoint bringup without a server. |
| Checkpoint conversion | integrated | Shape/key audit plus sharded safetensor writer; needs real-weight golden validation. |
| Offline `torch.compile` experiments | integrated | Shared compile policy and smoke path exist for explicit experiments and comparisons, not runtime hot-path compilation. |
| Offline graph capture | integrated | `make_fx` plus FakeTensorMode trace helper and tests support reference graph capture. |
| Profile artifact loop | integrated | Whole-run, time-sliced replay, CPU-offload replay, node-id subgraph, and focused-region commands write graph/profile/timeline/memory JSON, Chrome traces, and repro scripts; graph-pattern profiling adds pass reports and reference/optimized comparisons. |
| Fake process groups | integrated | Deterministic single-process collectives. |
| Flex attention | bridge | Dispatches to torch flex attention when available, with eager q/k/v fallback. |
| Piecewise CUDA graphs | bridge | Named runner captures static CUDA tensor inputs and recaptures on shape changes. |
| Paged attention | integrated | Native DeepSeek paged prefill/decode attends over request page tables with torch/Triton decode fallback. |
| Agent rank files | bridge | `disagg-init` emits editable prefill/decode rank files with local RPC wrappers. |
| Prefix KV reuse | integrated | Serving aliases reusable prefix pages across model layer cache rows with copy-on-write protection. |
| Continuous batching | bridge | Persistent row-assigned cache batches same-length prefill/decode groups without rebuilding temporary caches; OpenAI serving now microbatches same-shape requests. |
| OpenAI serving API | bridge | HTTP server exposes `/health`, `/v1/models`, and streaming/non-streaming `/v1/chat/completions`, can auto-launch Llama tensor-parallel workers, and has direct plus HTTP microbench loops. |
| Benchmark suites | integrated | `vllm-bench-suite`, `vllm-bench-plot`, `llama-bench-suite`, `openai-microbench`, and `openai-server-microbench` write repeatable commands, JSON summaries, and plots for vLLM-compatible comparisons. |
| Time-sliced virtual GPU profiling | integrated | Representative generation profiles can be scaled and replayed across virtual ranks on one physical device. |
| CPU offload profiling | bridge | Module-at-a-time CPU/device staging records movement overhead separately from compute; decode-cache offload and mmap streaming remain open. |
| Disaggregated prefill/decode | simulated | Planner models rank assignment and transfer latency. |
| Offline graph replacement | bridge | Leaf target swaps plus a multi-node symbolic/make_fx fused-op example exist for candidate generation before promotion. |
| Backend candidate providers | optional | Helion search is the first optional provider; future providers can include CuteDSL/CUTLASS, Triton, custom CUDA/C++, PyTorch custom ops, or pure PyTorch rewrites under the same promotion flow. |
| NVFP4 graph passes | reference | Quantized tensor contract and graph hook exist; production fused kernel remains open. |
| Research harness | minimal | Named experiments and metric comparison exist. |

<details>
<summary>Implementation and verification index</summary>

| Area | Code | Verification |
| --- | --- | --- |
| DSv4 model family | [`models/dsv4/`](../src/torchinferno/models/dsv4/) | `dsv4-smoke`, `dsv4-hf-smoke`, `tests/test_dsv4_e2e.py` |
| DeepSeek-V3.2 model family | [`models/deepseek_v32/`](../src/torchinferno/models/deepseek_v32/), compatibility [`models/deepseek.py`](../src/torchinferno/models/deepseek.py) | `deepseek-smoke`, `deepseek-hf-smoke`, `tests/test_deepseek_native.py` |
| Llama3 model family | [`models/llama3/`](../src/torchinferno/models/llama3/) | `validate-model-variants --family llama3`, `tests/test_model_variants.py` |
| Llama3 parallel execution adapters | [`models/llama3/pipeline.py`](../src/torchinferno/models/llama3/pipeline.py), [`models/llama3/tensor_parallel.py`](../src/torchinferno/models/llama3/tensor_parallel.py), [`benchmarks/torchinferno_llama.py`](../src/torchinferno/benchmarks/torchinferno_llama.py) | `llama-bench-suite`, `tests/test_llama3_tensor_parallel_distributed.py` |
| Model provenance and validation | [`models/variants.py`](../src/torchinferno/models/variants.py), [`variant_validation.py`](../src/torchinferno/variant_validation.py) | `model-variants`, `validate-model-variants`, `tests/test_model_variants.py` |
| Text IO and known-logit validation | [`tokenization.py`](../src/torchinferno/tokenization.py), [`validation.py`](../src/torchinferno/validation.py), [`models/auto.py`](../src/torchinferno/models/auto.py) | `text-generate`, `capture-logits`, `validate-logits`, `tests/test_production_workflows.py` |
| Checkpoint conversion | [`models/conversion.py`](../src/torchinferno/models/conversion.py) | `dsv4-audit`, `dsv4-convert`, `deepseek-audit`, `deepseek-convert`, `tests/test_conversion_and_kernels.py` |
| Offline graph optimization | [`compiler.py`](../src/torchinferno/compiler.py), [`graph/`](../src/torchinferno/graph/), [`profiling.py`](../src/torchinferno/profiling.py), [`OFFLINE_OPTIMIZATION.md`](OFFLINE_OPTIMIZATION.md) | `trace-smoke`, `profile-pattern`, `tests/test_scaffolding.py` |
| Profile artifact loops | [`profiling.py`](../src/torchinferno/profiling.py), [`runtime/offload.py`](../src/torchinferno/runtime/offload.py) | `profile-run`, `profile-timeslice`, `profile-offload`, `profile-region`, `profile-pattern`, `profile-subgraph`, `profile-nodes`, `tests/test_profile_artifacts.py` |
| Runtime policy scaffolds | [`runtime/`](../src/torchinferno/runtime/) | `sim-smoke`, `traffic-smoke`, `serve-smoke`, `disagg-init`, `disagg-smoke`, `tests/test_serving_engine.py`, `tests/test_disagg_ranks.py` |
| OpenAI-compatible serving | [`openai_server.py`](../src/torchinferno/openai_server.py), [`openai_http.py`](../src/torchinferno/openai_http.py), [`openai_warmup.py`](../src/torchinferno/openai_warmup.py) | `openai-server`, `openai-microbench`, `openai-server-microbench`, `tests/test_openai_server.py`, `tests/test_openai_server_microbench.py`, `tests/test_openai_warmup.py` |
| Backend replacement providers | [`kernels/`](../src/torchinferno/kernels/), [`research/helion.py`](../src/torchinferno/research/helion.py) | `perf-smoke`, `helion-candidate`, `helion-search-fx`, `helion-search-region`, `tests/test_performance_specialization.py` |
| Benchmark comparisons | [`benchmarks/`](../src/torchinferno/benchmarks/) | `vllm-bench-suite`, `vllm-bench-plot`, `llama-bench-suite`, `tests/test_vllm_benchmarks.py` |

</details>

## Next Production Milestones

1. Add fused CUDA kernels for batched paged prefill/decode request/page tables.
2. Promote OpenAI serving from same-shape microbatching to full continuous
   batching with admission, cancellation, and prefix-aware routing across live
   traffic.
3. Add decode-shape static buffer planning policies on top of the CUDA graph
   capture runner.
4. Validate native checkpoint conversion against real production weights with
   committed logit references.
5. Formalize provider-neutral candidate artifacts for graph partitions, then
   use them to promote MLA, grouped MoE, and NVFP4 linear replacements with
   architecture/shape guards.
