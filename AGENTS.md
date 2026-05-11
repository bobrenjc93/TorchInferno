# TorchInferno Agent Notes

TorchInferno is designed so agents can make focused contributions without
learning a serving stack first.

## Project Overview

TorchInferno is a torch-native inference workbench for making inference systems
easy to inspect, trace, optimize, simulate, and serve without booting a
heavyweight production stack for every experiment. The repo is organized around
separate planes:

- Model families in `src/torchinferno/models/` own tensor contracts,
  checkpoint rules, cache interfaces, and readable torch-native `v0`
  implementations.
- Runtime code in `src/torchinferno/runtime/`, `src/torchinferno/engine/`, and
  `src/torchinferno/server/` owns generation, batching, paged/prefix KV cache
  policy, OpenAI-compatible serving, distributed simulation, and graph runners.
- Offline optimization code in `src/torchinferno/compiler.py`,
  `src/torchinferno/graph/`, `src/torchinferno/kernels/`,
  `src/torchinferno/profiling.py`, and `src/torchinferno/research/` captures
  reference graphs, evaluates candidate replacements, and produces evidence
  before anything is promoted into runtime.
- Documentation and readiness tracking live in `README.md`,
  `docs/OFFLINE_OPTIMIZATION.md`, `docs/ROADMAP.md`, and
  `src/torchinferno/audit.py`.

Current model coverage includes compact DSv4, native DeepSeek-V3.2-style, and
Llama3 paths. DSv4 and DeepSeek are intentionally CPU-friendly correctness and
cache bringup surfaces; Llama3 carries the larger production-scale pipeline and
tensor-parallel benchmark work.

## Working Philosophy

- Keep the readable torch-native model path as the reference contract.
- Keep model families separate from serving, profiling, parallelism,
  disaggregation, and backend replacement workbenches.
- Keep runtime hot paths concrete: do not generate graphs, compile kernels, or
  run search loops from constructors, `forward`, `generate`, serving engines, or
  request handling.
- Promote optimizations only after they are validated against the reference path
  with reproducible artifacts.
- Prefer small CPU-runnable APIs and tests first, then add CUDA-only or
  distributed behavior behind clear interfaces.
- Treat benchmark and profile numbers as evidence to explain, not as answers to
  chase.

## Benchmark and Evaluation Integrity

Never hard-code benchmark answers, expected outputs, request shapes, prompt
strings, token ids, fixture names, dataset identifiers, hashes, random seeds, or
test-case positions to pass a benchmark or evaluation. Do not add logic that
detects a benchmark harness, public test, hidden test, or known prompt and
returns a special result.

Prohibited patterns include prompt lookup tables, fixture-specific branches,
`if benchmark_name == ...` behavior changes, request-length fingerprints, and
shortcuts that bypass normal model/runtime logic for known eval cases.

Benchmark-driven work is welcome when it improves the general implementation:
better scheduling, cache policy, graph reuse, kernel selection, batching,
sampling, IO, or validation. Such changes must be expressed as normal runtime or
offline optimization behavior, must preserve correctness on unseen inputs, and
must have focused tests or profile artifacts that explain why the improvement is
real.

If a benchmark exposes a gap, fix the underlying model, runtime, conversion,
kernel, or measurement issue. If a benchmark itself is flawed, document the
limitation and add a fairer check instead of training the code to the benchmark.

## Tooling Snapshot

- `python3 -m pytest` runs the test suite; `make test` is the shortcut.
- `make lint` runs `pyflakes` and `ruff`; `make dead-code` runs `vulture`.
- `torchinferno.cli audit` and `model-variants` summarize readiness and variant
  provenance.
- Smoke commands cover DSv4, DeepSeek, tracing, serving, simulation, traffic,
  perf, and profiling paths.
- Profile commands write artifacts under `.torchinferno_runs/` for whole-model,
  time-sliced, offload, node, subgraph, region, and pattern analysis.
- Optional extras are split in `pyproject.toml`: `serve` and `text` for
  tokenizer-backed serving, `kernels` for Triton, `helion` for Helion
  experiments, and `dev` for lint/test tools.

## Start Here

Run the local checks before and after non-trivial changes:

```bash
PYTHONPATH=src python3 -m torchinferno.cli audit
PYTHONPATH=src python3 -m torchinferno.cli model-variants
python3 -m pytest
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu --tokens 2
PYTHONPATH=src python3 -m torchinferno.cli sim-smoke
PYTHONPATH=src python3 -m torchinferno.cli traffic-smoke
PYTHONPATH=src python3 -m torchinferno.cli serve-smoke --device cpu --cache-backend paged
PYTHONPATH=src python3 -m torchinferno.cli perf-smoke --device cpu --heads 2 --seq-len 8 --head-dim 8 --value-dim 8
PYTHONPATH=src python3 -m torchinferno.cli profile-run .torchinferno_runs/dsv4-cpu --device cpu --warmup 0 --no-profiler
PYTHONPATH=src python3 -m torchinferno.cli profile-timeslice .torchinferno_runs/dsv4-timeslice-cpu --device cpu --warmup 0 --iters 1 --no-profiler
PYTHONPATH=src python3 -m torchinferno.cli profile-offload .torchinferno_runs/dsv4-offload-cpu --device cpu --prompt-tokens 3 --new-tokens 1 --warmup 1 --iters 3
PYTHONPATH=src python3 -m torchinferno.cli profile-nodes .torchinferno_runs/dsv4-cpu --grep embedding
PYTHONPATH=src python3 -m torchinferno.cli profile-subgraph .torchinferno_runs/embedding-cpu --source-run .torchinferno_runs/dsv4-cpu --nodes 3 --device cpu --warmup 0 --iters 1
PYTHONPATH=src python3 -m torchinferno.cli profile-region .torchinferno_runs/attn0-cpu --region layers.0.attn --device cpu --warmup 0 --iters 1
PYTHONPATH=src python3 -m torchinferno.cli profile-pattern .torchinferno_runs/swiglu-pattern-cpu --device cpu --warmup 0 --iters 1
```

Makefile shortcuts cover the common pieces as `make audit`, `make variants`,
`make lint`, `make dead-code`, `make test`, `make smoke`, `make perf`,
`make openai-server-bench`, and `make disagg`. Focused profiling shortcuts
are available as `make profile`, `make profile-timeslice`,
`make profile-offload`, `make profile-nodes`, `make profile-subgraph`,
`make profile-region`, and `make profile-pattern`; benchmark planning is
available as `make vllm-bench-plan`.

## Git Pushes

When asked to push, use `agentexec git push ...`. Direct `git push` may be
blocked by local command policy or lack interactive HTTPS credentials in agent
sessions.

## Fast Profile Loops

- Use `profile-run` when you need a full generation graph, memory profile,
  Chrome trace, and standalone `repro.py`.
- Use `profile-nodes` and `profile-subgraph` when a module boundary is too
  coarse. `profile-subgraph` accepts integer node ids, comma lists, or ranges
  from a prior `profile-run` graph and emits a callable FX slice with boundary
  inputs and source-output comparison.
- Use `profile-region` when you want to optimize one module path such as
  `layers.0.attn`, `layers.0.moe`, or `model.layers.0.self_attn`.
- Use `profile-pattern` when you are working on graph replacement. It writes
  reference and optimized graphs, `pass_report.json`, profiler JSON for both
  callables, and `comparison.json`.

## Offline Optimization Contract

- Do not put compilers, graph tracers, partitioners, Helion/CuteDSL/CUTLASS
  generators, Triton codegen, CUDA extension builds, or graph-search loops in
  model constructors, `forward`, `generate`, serving engines, or request hot
  paths.
- Treat `v0` as the readable torch-native reference contract. Capture and
  partition it offline, then compare candidates such as `candidate-v1-01`
  against `v0` with reproducible artifacts.
- Keep replacement providers pluggable. A candidate may come from Helion,
  CuteDSL/CUTLASS, Triton, custom CUDA/C++, PyTorch custom ops,
  `torch.compile` experiments, or pure PyTorch rewrites, but all providers
  should feed the same validation and benchmark flow.
- Promote only checked-in, validated candidates into runtime-selectable
  variants. Runtime should load a concrete variant directly, not generate or
  rewrite one.
- See `docs/OFFLINE_OPTIMIZATION.md` before adding graph optimization or kernel
  promotion workflows.

For conversion or kernel changes, also run:

```bash
python3 -m pytest tests/test_conversion_and_kernels.py
python3 -m pytest tests/test_deepseek_native.py
python3 -m pytest tests/test_production_workflows.py
python3 -m pytest tests/test_performance_specialization.py
python3 -m pytest tests/test_serving_engine.py
```

## Where To Put Work

- Model architecture changes: `src/torchinferno/models/`.
- Checkpoint audit/conversion work: `src/torchinferno/models/conversion.py`.
- Custom kernels and backend provider hooks: `src/torchinferno/kernels/`.
- Compiler and tracing helpers: `src/torchinferno/compiler.py` and
  `src/torchinferno/graph/`; these are offline workbench tools, not runtime
  model dependencies.
- Scheduling, batching, cache, routing, and simulation policies:
  `src/torchinferno/runtime/`.
- Parallel and distributed execution adapters: shared policy in
  `src/torchinferno/runtime/`; family-specific checkpoint loading or sharding
  adapters beside the affected model family.
- Native paged-cache serving integration: `src/torchinferno/runtime/serving.py`
  and `src/torchinferno/models/deepseek_v32/model.py`.
- Feature readiness/DX status: `src/torchinferno/audit.py` and
  `docs/ROADMAP.md`.
- Profile artifact loops and repro generators: `src/torchinferno/profiling.py`.
- Text IO and validation workflows: `src/torchinferno/tokenization.py` and
  `src/torchinferno/validation.py`.
- Repeatable experiments and comparisons: `src/torchinferno/research/`.
- End-to-end behavioral tests: `tests/`.

## Contribution Shape

- Keep model code torch-native and easy to trace.
- Keep model families separate from serving, profiling, parallelism, and other
  execution workbenches in docs and APIs.
- Keep graph optimization offline: capture `v0`, generate provider-neutral
  candidates, benchmark them, then promote concrete variants for runtime.
- Add focused tests for every new scaffold or policy.
- Prefer small APIs that can run on CPU before introducing CUDA-only code.
- Use fake process groups and time-sliced simulation for distributed policy
  tests.
- Put custom-kernel experiments behind offline graph passes or stable kernel
  APIs so baseline DSv4 remains readable.
