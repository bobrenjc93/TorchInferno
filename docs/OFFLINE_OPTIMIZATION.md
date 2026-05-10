# Offline Graph Optimization

TorchInferno treats model execution and compiler exploration as separate
phases. Runtime code loads a concrete model variant and executes it directly.
Graph capture, partitioning, backend candidate generation, candidate
comparison, and promotion happen offline in explicit workbenches.

## Runtime Boundary

Runtime model paths must not invoke compilers, graph tracers, partitioners,
kernel generators, or search loops in the hot path.

Allowed at runtime:

- Instantiate a checked-in model variant such as `v0`, `v1`, or a promoted
  family-specific adapter.
- Call checked-in PyTorch, Triton, CUDA, or custom-op implementations through a
  stable API.
- Select between already-implemented backends and deterministic fallbacks.
- Validate shape and architecture guards before using a promoted fast path.

Not allowed at runtime:

- Calling `torch.compile`, `make_fx`, Helion, CuteDSL/CUTLASS generators,
  Triton code generation, CUDA extension builds, or graph partition search from
  model constructors, `forward`, `generate`, serving request loops, or OpenAI
  engine paths.
- Rewriting model submodules after load as part of request handling.
- Searching for a faster partition during server startup unless the command is
  explicitly an offline benchmark or promotion tool.
- Making Helion or any other generator a required serving dependency.

Environment toggles may choose among committed implementations, but they should
not generate new implementations.

## Intended Workflow

1. Create a readable `v0` reference.
   The baseline for each family should be torch-native Python/PyTorch with
   stable tensor, cache, and checkpoint contracts. Prefer one readable reference
   graph surface per family; use a small `raw_ops.py` only when it keeps
   replacement boundaries explicit. Provider-specific code does not belong in
   `v0`.

2. Capture the offline graph.
   Use workbench commands to trace the `v0` model, a module region, or a node
   range. The capture should write graph JSON/text, a runnable repro, input
   summaries, output references, and shape/architecture guards.

3. Partition candidate regions.
   Partitioning is an offline decision. Partitions can be whole modules, FX node
   ranges, symbolic patterns, or request/page-table-shaped kernels. Each
   partition should record its inputs, outputs, aliases, mutation behavior,
   dtype/device assumptions, and fallback path.

4. Generate or write replacements.
   A replacement provider can be Helion, CuteDSL/CUTLASS, Triton, handwritten
   CUDA/C++, a PyTorch custom op, a `torch.compile` experiment, or a pure
   PyTorch rewrite. TorchInferno should treat these as providers behind one
   candidate contract, not as special model concepts.

5. Validate and benchmark candidates.
   Compare `v0` and every candidate on correctness, logits, latency, memory,
   trace shape, fallback behavior, and reproducibility. Candidate names should
   be explicit, for example `candidate-v1-01`, and reports should be
   machine-readable.

6. Promote deliberately.
   A candidate becomes `v1` or another checked-in variant only after it has
   passed the relevant correctness and benchmark gates. Promotion updates the
   variant registry, tests, docs, and any checked-in kernel/provider artifacts.
   Runtime then selects the promoted variant directly.

## Candidate Artifact Contract

Offline candidate directories should be self-contained. A complete candidate
run should aim to include:

- `source_graph.json` and `source_graph.txt`: captured `v0` graph.
- `partition.json`: selected nodes, boundary inputs, outputs, guards, aliases,
  and mutation assumptions.
- `provider.json`: provider name, version, command, device, dtype, and
  generated artifact paths.
- `candidate_graph.json` and `candidate_graph.txt`: graph after replacement.
- `correctness.json`: reference outputs, candidate outputs, tolerances, and
  pass/fail reasons.
- `benchmark.json`: latency, memory, profiler summaries, and comparison to the
  current best baseline.
- `manifest.json`: artifact index.
- `repro.py`: standalone command for rerunning the exact comparison.

Existing profile commands already write many of these pieces. New optimization
tools should extend that artifact shape rather than inventing unrelated output
formats.

## Backend Provider Neutrality

The graph partition is the stable unit. The replacement provider is pluggable.

Examples:

- Helion can search generated CUDA candidates for a partition.
- CuteDSL/CUTLASS can target tensor-core kernels where that is the right fit.
- Triton can provide handwritten or generated CUDA kernels with torch
  fallbacks.
- A custom CUDA/C++ op can be checked in when the interface is stable.
- A PyTorch custom op can preserve graph boundaries and fake-tensor behavior.
- `torch.compile` can be used as an offline comparator or exploratory backend,
  but it should not become a hidden runtime requirement.

Provider reports should normalize into the same candidate schema so benchmark
and promotion tooling can compare them fairly.

## Current Repo Mapping

- Model families live under `src/torchinferno/models/`.
- Provenance-tracked variants live under `models/*_family/`.
- Reference operation surfaces are `raw_ops.py`; promoted operation hooks are
  currently represented by `fused_ops.py`.
- Offline graph capture lives in `src/torchinferno/graph/` and
  `src/torchinferno/profiling.py`.
- Kernel APIs and graph replacement passes live in `src/torchinferno/kernels/`.
- Helion search is currently one optional provider in
  `src/torchinferno/research/helion.py`.
- Runtime serving, scheduling, cache, and traffic policy live under
  `src/torchinferno/runtime/` and should consume promoted variants rather than
  perform optimization.

## Promotion Checklist

Before promoting a candidate:

- `v0` remains readable and provider-independent.
- The candidate has a reproducible artifact directory and standalone repro.
- Correctness passes against `v0` for the target tolerances and shapes.
- Benchmarks beat the configured baseline by enough margin to justify the added
  surface area.
- Shape, dtype, device, architecture, cache, and aliasing guards are explicit.
- CPU or torch fallback behavior remains available where required.
- Tests cover the promoted path and the fallback path.
- The model variant registry records parentage and provider provenance.
- Runtime code selects the promoted variant directly and does not call offline
  compiler or generator tools.

## Current Commands

Useful existing commands for the offline loop:

```bash
PYTHONPATH=src python3 -m torchinferno.cli model-variants
PYTHONPATH=src python3 -m torchinferno.cli validate-model-variants --device cpu
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu --tokens 2 --fake
PYTHONPATH=src python3 -m torchinferno.cli profile-run .torchinferno_runs/dsv4-cpu --device cpu --warmup 0 --no-profiler
PYTHONPATH=src python3 -m torchinferno.cli profile-region .torchinferno_runs/attn0-cpu --region layers.0.attn --device cpu --warmup 0 --iters 1
PYTHONPATH=src python3 -m torchinferno.cli profile-subgraph .torchinferno_runs/subgraph-cpu --source-run .torchinferno_runs/dsv4-cpu --nodes 3 --device cpu --warmup 0 --iters 1
PYTHONPATH=src python3 -m torchinferno.cli profile-pattern .torchinferno_runs/swiglu-pattern-cpu --device cpu --warmup 0 --iters 1
```

Provider-specific commands, such as Helion search, should feed the same offline
candidate and promotion flow rather than defining a separate path:

```bash
PYTHONPATH=src python3 -m torchinferno.cli helion-search-fx \
  --candidate swiglu \
  --device cuda \
  --remember .torchinferno_runs/helion-decisions.jsonl
```
