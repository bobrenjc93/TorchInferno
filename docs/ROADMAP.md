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
| Checkpoint conversion | integrated | Shape/key audit plus sharded safetensor writer; needs real-weight golden validation. |
| `torch.compile` | integrated | Shared compile policy and smoke path. |
| `make_fx` and fake tensors | integrated | FakeTensorMode trace helper and tests. |
| Fake process groups | integrated | Deterministic single-process collectives. |
| Monarch | scaffold | Adapter detects Monarch and falls back to fake world. |
| Flex attention | reference | Eager q/k/v fallback exists; real flex dispatch remains open. |
| Piecewise CUDA graphs | scaffold | Named runner exists; static CUDA capture buffers remain open. |
| Paged attention | integrated | Native DeepSeek paged decode path plus torch/Triton fallback. |
| Prefix KV reuse | integrated | Serving copies reusable prefix KV into new request caches. |
| Continuous batching | bridge | Same-shape prefill/decode groups are batched through temporary caches. |
| Disaggregated prefill/decode | simulated | Planner models rank assignment and transfer latency. |
| NVFP4 graph passes | reference | Quantized tensor contract and graph hook exist; fused kernel remains open. |
| Research harness | minimal | Named experiments and metric comparison exist. |

## Next Production Milestones

1. Replace temporary-cache decode batching with persistent row-assigned paged
   cache state and per-row sequence lengths.
2. Replace prefix KV copy with shared page aliasing across model layer caches.
3. Add batched paged prefill and decode kernels that accept request/page tables.
4. Add static buffer planning and CUDA graph capture for decode pieces.
5. Implement real flex-attention dispatch behind the current fallback contract.
6. Add Monarch-backed distributed execution behind the fake-world interface.
7. Validate native checkpoint conversion against real production weights with
   committed logit references.
8. Expand graph pattern matching from call-target replacement to subgraph
   captures for MLA, grouped MoE, and NVFP4 linear regions.
