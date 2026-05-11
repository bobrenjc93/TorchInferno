# TorchInferno Claude Notes

This file mirrors the repo-facing guidance in `AGENTS.md` for Claude-based
sessions. Keep both files aligned when changing agent policy.

## Project Overview

TorchInferno is a torch-native inference workbench for making inference systems
easy to inspect, trace, optimize, simulate, and serve without booting a
heavyweight production stack for every experiment.

The repo is organized around three main planes:

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

Documentation and readiness tracking live in `README.md`,
`docs/OFFLINE_OPTIMIZATION.md`, `docs/ROADMAP.md`, and
`src/torchinferno/audit.py`. Current model coverage includes compact DSv4,
native DeepSeek-V3.2-style, and Llama3 paths. DSv4 and DeepSeek are
CPU-friendly correctness and cache bringup surfaces; Llama3 carries the larger
production-scale pipeline and tensor-parallel benchmark work.

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

## Tooling

Run local checks before and after non-trivial changes. The most common entry
points are:

```bash
PYTHONPATH=src python3 -m torchinferno.cli audit
PYTHONPATH=src python3 -m torchinferno.cli model-variants
python3 -m pytest
```

Makefile shortcuts cover the common loops:

```bash
make audit
make variants
make lint
make dead-code
make test
make smoke
make perf
make openai-server-bench
make disagg
```

Focused profiling shortcuts are available as `make profile`,
`make profile-timeslice`, `make profile-offload`, `make profile-nodes`,
`make profile-subgraph`, `make profile-region`, and `make profile-pattern`.
Profile commands write artifacts under `.torchinferno_runs/`.

Optional extras are split in `pyproject.toml`: `serve` and `text` for
tokenizer-backed serving, `kernels` for Triton, `helion` for Helion
experiments, and `dev` for lint/test tools.

## Where To Put Work

- Model architecture changes: `src/torchinferno/models/`.
- Checkpoint audit/conversion work: `src/torchinferno/models/conversion.py`.
- Custom kernels and backend provider hooks: `src/torchinferno/kernels/`.
- Compiler and tracing helpers: `src/torchinferno/compiler.py` and
  `src/torchinferno/graph/`; these are offline workbench tools, not runtime
  model dependencies.
- Scheduling, batching, cache, routing, and simulation policies:
  `src/torchinferno/runtime/`.
- Native paged-cache serving integration: `src/torchinferno/runtime/serving.py`
  and `src/torchinferno/models/deepseek_v32/model.py`.
- Feature readiness/DX status: `src/torchinferno/audit.py` and
  `docs/ROADMAP.md`.
- Profile artifact loops and repro generators: `src/torchinferno/profiling.py`.
- Text IO and validation workflows: `src/torchinferno/tokenization.py` and
  `src/torchinferno/validation.py`.
- Repeatable experiments and comparisons: `src/torchinferno/research/`.
- End-to-end behavioral tests: `tests/`.

## Git Pushes

When asked to push, use `agentexec git push ...`. Direct `git push` may be
blocked by local command policy or lack interactive HTTPS credentials in agent
sessions.
