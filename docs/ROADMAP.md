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
| Model provenance variants | reference | DSv4, DeepSeek-V3.2, and Llama3 have raw/fused v0/v1 ladders and registry lineage. |
| Checkpoint conversion | integrated | Shape/key audit plus sharded safetensor writer; needs real-weight golden validation. |
| `torch.compile` | integrated | Shared compile policy and smoke path. |
| `make_fx` and fake tensors | integrated | FakeTensorMode trace helper and tests. |
| Profile artifact loop | integrated | Whole-run, time-sliced replay, CPU-offload replay, node-id subgraph, and focused-region commands write graph/profile/timeline/memory JSON, Chrome traces, and repro scripts; graph-pattern profiling adds pass reports and reference/optimized comparisons. |
| Fake process groups | integrated | Deterministic single-process collectives. |
| Monarch | scaffold | Adapter detects Monarch and falls back to fake world. |
| Flex attention | reference | Eager q/k/v fallback exists; real flex dispatch remains open. |
| Piecewise CUDA graphs | scaffold | Named runner exists; static CUDA capture buffers remain open. |
| Paged attention | integrated | Native DeepSeek paged decode path plus torch/Triton fallback. |
| Agent rank files | bridge | `disagg-init` emits editable prefill/decode rank files with local RPC wrappers. |
| Prefix KV reuse | integrated | Serving aliases reusable prefix pages across model layer cache rows with copy-on-write protection. |
| Continuous batching | bridge | Persistent row-assigned cache batches same-length prefill/decode groups without rebuilding temporary caches. |
| Time-sliced virtual GPU profiling | integrated | Representative generation profiles can be scaled and replayed across virtual ranks on one physical device. |
| CPU offload profiling | bridge | Module-at-a-time CPU/device staging records movement overhead separately from compute; decode-cache offload and mmap streaming remain open. |
| Disaggregated prefill/decode | simulated | Planner models rank assignment and transfer latency. |
| Graph pattern replacement | bridge | Leaf target swaps plus a multi-node symbolic/make_fx fused-op example exist. |
| NVFP4 graph passes | reference | Quantized tensor contract and graph hook exist; production fused kernel remains open. |
| Research harness | minimal | Named experiments and metric comparison exist. |

## Next Production Milestones

1. Add batched paged prefill and decode kernels that accept request/page tables.
2. Replace multi-token prefill materialization with paged/flex attention over
   request page tables.
3. Add static buffer planning and CUDA graph capture for decode pieces.
4. Implement real flex-attention dispatch behind the current fallback contract.
5. Add Monarch-backed distributed execution behind the fake-world interface.
6. Validate native checkpoint conversion against real production weights with
   committed logit references.
7. Apply subgraph captures to production MLA, grouped MoE, and NVFP4 linear
   regions with architecture/shape guards.
