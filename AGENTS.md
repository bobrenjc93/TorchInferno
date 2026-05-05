# TorchInferno Agent Notes

TorchInferno is designed so agents can make focused contributions without
learning a serving stack first.

## Start Here

Run the local checks before and after non-trivial changes:

```bash
PYTHONPATH=src python3 -m torchinferno.cli audit
python3 -m pytest
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu --tokens 2
PYTHONPATH=src python3 -m torchinferno.cli sim-smoke
PYTHONPATH=src python3 -m torchinferno.cli traffic-smoke
PYTHONPATH=src python3 -m torchinferno.cli serve-smoke --device cpu --cache-backend paged
PYTHONPATH=src python3 -m torchinferno.cli perf-smoke --device cpu --heads 2 --seq-len 8 --head-dim 8 --value-dim 8
```

The same core checks are available as `make audit`, `make test`, and
`make smoke`.

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
- Custom kernels and kernel graph passes: `src/torchinferno/kernels/`.
- Compiler and tracing helpers: `src/torchinferno/compiler.py` and
  `src/torchinferno/graph/`.
- Scheduling, batching, cache, routing, and simulation policies:
  `src/torchinferno/runtime/`.
- Native paged-cache serving integration: `src/torchinferno/runtime/serving.py`
  and `src/torchinferno/models/deepseek.py`.
- Feature readiness/DX status: `src/torchinferno/audit.py` and
  `docs/ROADMAP.md`.
- Text IO and validation workflows: `src/torchinferno/tokenization.py` and
  `src/torchinferno/validation.py`.
- Repeatable experiments and comparisons: `src/torchinferno/research/`.
- End-to-end behavioral tests: `tests/`.

## Contribution Shape

- Keep model code torch-native and easy to trace.
- Add focused tests for every new scaffold or policy.
- Prefer small APIs that can run on CPU before introducing CUDA-only code.
- Use fake process groups and time-sliced simulation for distributed policy
  tests.
- Put custom-kernel experiments behind graph passes or runtime interfaces so
  baseline DSv4 remains readable.
