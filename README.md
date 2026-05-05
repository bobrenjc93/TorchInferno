# TorchInferno

TorchInferno is a torch-native inference playground in the spirit of TorchTitan:
small enough to hack on locally, but structured around the same surfaces needed
for serious model-serving work.

This repository currently ships an end-to-end DSv4-style inference path with:

- A decoder-only causal LM in pure PyTorch.
- MLA-like latent KV projection.
- Rotary causal attention with grouped KV heads.
- Routed top-k MoE feed-forward blocks.
- Explicit append-only KV cache for prefill and decode.
- Greedy and temperature sampling.
- A ragged request batching harness.
- A deterministic time-sliced multi-GPU simulator.
- `make_fx`/FakeTensor tracing utilities and a graph pass registry.
- CLI smoke tests for local CPU or CUDA execution.

The DSv4 model here is an inference harness with random local weights by
default. It is meant to make architecture, cache, batching, tracing, compiler,
and kernel-replacement work easy to develop and test before wiring in production
weights or custom kernels.

## Quickstart

From the repo root:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
torchinferno dsv4-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
```

Without installing the package, use `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cuda
PYTHONPATH=src python3 -m torchinferno.cli batch-smoke --device cpu
```

## DSv4 End-To-End Path

The main model lives in `src/torchinferno/models/dsv4.py`.

```python
import torch

from torchinferno import DSv4ForCausalLM, tiny_dsv4_config

config = tiny_dsv4_config(vocab_size=128, max_seq_len=32)
model = DSv4ForCausalLM(config).eval()
prompt = torch.tensor([[1, 2, 3, 4]])

with torch.inference_mode():
    output = model.generate(prompt, max_new_tokens=4)

print(output)
```

The cache path is covered by tests that compare incremental decode logits
against a full causal forward pass.

## Torch Compile

The CLI exposes a small compile smoke:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke \
  --device cpu \
  --batch-size 1 \
  --prompt-tokens 2 \
  --new-tokens 1 \
  --compile
```

First compile is expected to be much slower than eager execution. The current
goal is to keep the forward path traceable enough for compiler experiments, not
to claim a tuned compiled serving path yet.

## Runtime Harnesses

`torchinferno.runtime.batching.run_continuous_batch` accepts ragged requests at
the API boundary, buckets compatible prompts, runs dense model calls, and returns
results in request order.

`torchinferno.runtime.simulation.TimeSlicedSimulator` provides a deterministic
single-process harness for virtual GPU ranks. It is intentionally simple so
network latency, request bursts, and disaggregated prefill/decode policies can
be modeled without booting a full distributed stack.

## Graph Work

`torchinferno.graph.trace_with_make_fx` wraps `make_fx` and optional
`FakeTensorMode`.

`torchinferno.graph.PassRegistry` is the landing point for pattern-match graph
passes. This is where replacements for specialized kernels, NVFP4 MoE paths,
radix attention, paged attention, or flex-attention experiments should be
registered without complicating model code.

## Test Matrix

Current local checks:

```bash
python3 -m pytest
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cuda
```

The test suite covers:

- Cached decode matching full forward logits.
- End-to-end token generation.
- Ragged request batching.
- Time-sliced virtual GPU simulation.
- CLI execution.

## Roadmap

Near-term work that fits the design direction:

- Real DSv4/DeepSeek checkpoint loading and tokenizer integration.
- Paged KV allocation instead of contiguous append-only cache.
- Prefix-aware routing and prefix cache reuse.
- Piecewise CUDA graph capture for prefill and decode.
- Flex attention and custom attention kernel swaps.
- Monarch and fake process group simulation for distributed policies.
- Multi-GPU single-GPU time slicing with richer latency and bandwidth models.
- Disaggregated prefill/decode scheduling experiments.
- Pattern-matched graph replacements for NVFP4 and other model-specific kernels.
- Auto research harnesses for comparing cache, batching, and routing policies.

## Repository Layout

```text
src/torchinferno/
  cli.py                 CLI smoke runners.
  models/dsv4.py         DSv4-style causal LM, MoE, attention, and KV cache.
  runtime/batching.py    Ragged request batching harness.
  runtime/simulation.py  Time-sliced virtual GPU simulator.
  graph/export.py        make_fx and FakeTensor tracing helper.
  graph/passes.py        Graph pass registry.
tests/
  test_dsv4_e2e.py       End-to-end DSv4 and runtime tests.
```

## Development Principles

- Keep model code torch-native and easy to trace.
- Prefer simulation-first workflows over heavyweight serving bootstraps.
- Make small, focused harnesses that let optimization work zoom into one system
  boundary at a time.
- Keep graph and kernel replacement hooks explicit so experimental SOTA paths do
  not make baseline code hard to read.
- Upstream generic PyTorch improvements where possible, and keep TorchInferno
  thin when the platform can carry the abstraction cleanly.
