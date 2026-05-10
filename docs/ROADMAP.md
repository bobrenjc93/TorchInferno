# TorchInferno Roadmap

This document is the working readiness map for the original TorchInferno plan.
The CLI view is `torchinferno audit`.

## Readiness Levels

- `integrated`: working code is exercised by tests and at least one CLI path.
- `bridge`: the execution shape is present, but still uses reference/copy-heavy
  mechanics rather than the final production implementation.
- `reference`: correctness contract exists; optimized implementation is not
  complete.
- `scaffold`: API shape exists, but production execution is future work.

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Native DeepSeek-V3.2-style model | integrated | Torch-native config/model/cache/checkpoint path exists. |
| Native Llama3 production-scale paths | bridge | Llama 70B config, pipeline and tensor-parallel safetensor loaders/generate paths, and torchrun TP compatibility coverage exist; production scheduler integration remains open. |
| Model provenance variants | reference | DSv4, DeepSeek-V3.2, and Llama3 have raw/fused v0/v1 ladders and registry lineage. |
| Eager-vs-optimized logit validation | integrated | `validate-model-variants` compares tiny eager v0 logits against optimized variants with a 1% default tolerance and optional JSON reports. |
| Checkpoint conversion | integrated | Shape/key audit plus sharded safetensor writer; needs real-weight golden validation. |
| `torch.compile` | integrated | Shared compile policy and smoke path. |
| `make_fx` and fake tensors | integrated | FakeTensorMode trace helper and tests. |
| Profile artifact loop | integrated | Whole-run, time-sliced replay, CPU-offload replay, node-id subgraph, and focused-region commands write graph/profile/timeline/memory JSON, Chrome traces, and repro scripts; graph-pattern profiling adds pass reports and reference/optimized comparisons. |
| Fake process groups | integrated | Deterministic single-process collectives. |
| Flex attention | bridge | Dispatches to torch flex attention when available, with eager q/k/v fallback. |
| Piecewise CUDA graphs | bridge | Named runner captures static CUDA tensor inputs and recaptures on shape changes. |
| Paged attention | integrated | Native DeepSeek paged prefill/decode attends over request page tables with torch/Triton decode fallback. |
| Agent rank files | bridge | `disagg-init` emits editable prefill/decode rank files with local RPC wrappers. |
| Prefix KV reuse | integrated | Serving aliases reusable prefix pages across model layer cache rows with copy-on-write protection. |
| Continuous batching | bridge | Persistent row-assigned cache batches same-length prefill/decode groups without rebuilding temporary caches; OpenAI serving now microbatches same-shape requests. |
| OpenAI serving API | bridge | HTTP server exposes `/v1/models` and streaming/non-streaming `/v1/chat/completions`, can auto-launch Llama tensor-parallel workers, and has direct plus HTTP microbench loops. |
| Benchmark suites | integrated | `vllm-bench-suite`, `vllm-bench-plot`, `llama-bench-suite`, and `openai-server-microbench` write repeatable commands, JSON summaries, and plots for vLLM-compatible comparisons. |
| Time-sliced virtual GPU profiling | integrated | Representative generation profiles can be scaled and replayed across virtual ranks on one physical device. |
| CPU offload profiling | bridge | Module-at-a-time CPU/device staging records movement overhead separately from compute; decode-cache offload and mmap streaming remain open. |
| Disaggregated prefill/decode | simulated | Planner models rank assignment and transfer latency. |
| Graph pattern replacement | bridge | Leaf target swaps plus a multi-node symbolic/make_fx fused-op example exist. |
| Helion candidate kernels | optional/experimental | `helion-search-fx` and `helion-search-region` enumerate supported FX windows and benchmark generated kernels before production promotion. |
| NVFP4 graph passes | reference | Quantized tensor contract and graph hook exist; production fused kernel remains open. |
| Research harness | minimal | Named experiments and metric comparison exist. |

## Next Production Milestones

1. Add fused CUDA kernels for batched paged prefill/decode request/page tables.
2. Promote OpenAI serving from same-shape microbatching to full continuous
   batching with admission, cancellation, and prefix-aware routing across live
   traffic.
3. Add decode-shape static buffer planning policies on top of the CUDA graph
   capture runner.
4. Validate native checkpoint conversion against real production weights with
   committed logit references.
5. Apply subgraph captures to production MLA, grouped MoE, and NVFP4 linear
   regions with architecture/shape guards.
