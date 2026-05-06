# TorchInferno

TorchInferno is a torch-native inference workbench in the spirit of
TorchTitan. The goal is to make inference systems easy to trace, optimize,
simulate, and extend without booting a heavyweight serving stack for every
experiment.

The current repo has a working DSv4-style end-to-end path plus scaffolding for
the compiler, runtime, scheduling, cache, routing, and research surfaces needed
to grow toward production-grade SOTA inference.

## What Works Today

- DSv4-style decoder-only causal LM in pure PyTorch.
- MLA-like latent KV projection, grouped KV heads, rotary causal attention, and
  routed top-k MoE feed-forward blocks.
- Native DeepSeek-V3.2-style causal LM with query LoRA, split RoPE/nope QK
  heads, latent MQA KV projection, independent value head dimensions,
  dense-to-MoE layer transitions, shared experts, grouped top-k routing, and
  score correction bias support.
- Explicit dense and paged KV cache backends for prefill and decode correctness
  tests in the native DeepSeek path.
- Hugging Face-style local and Hub checkpoint save/load for TorchInferno DSv4
  key names and tensor shapes.
- DeepSeek-style checkpoint audit and exact conversion for checkpoints whose
  tensor contracts match TorchInferno DSv4.
- Auto model loading, tokenizer-backed text generation, and known-logit
  validation for checkpoint bringup.
- Greedy and temperature sampling.
- Ragged request API with dense continuous-batching execution buckets.
- Token-step continuous serving harness with admission, paged-cache policy, and
  prefix-hit accounting, shared prefix-page reuse, persistent row-assigned
  paged cache state, and same-shape prefill/decode microbatching.
- Deterministic time-sliced virtual GPU simulation.
- Disaggregated prefill/decode planner with network latency modeling.
- Fake process groups and fake collectives for single-process distributed
  policy tests.
- Paged KV cache allocator integrated into native DeepSeek decode and available
  as a standalone kernel workbench.
- Functional paged causal attention reference for kernel replacement work.
- Radix/prefix tree, prefix-aware router, and prefix cache lookup.
- `torch.compile` helper and CLI smoke path.
- `make_fx` tracing helper with FakeTensorMode support.
- Pattern-based FX graph pass registry with call-target and multi-node
  subgraph replacement helpers.
- Flex-attention-shaped q/k/v API with eager fallback.
- Piecewise CUDA graph runner API with CPU/eager fallback.
- Optional Monarch adapter point with fake-world fallback.
- Bursty traffic simulation for request diversity and latency modeling.
- Triton CUDA kernels for RMSNorm and SwiGLU activation, with eager torch
  fallbacks on CPU or unsupported devices.
- Triton-backed paged decode attention specialization with a torch reference
  fallback.
- NVFP4 quantized-linear reference surface and graph-pass hook for future fused
  DeepSeek-V3.2-NVFP4 kernels.
- Local performance benchmark smoke for reference versus specialized paths.
- Minimal auto research harness for comparing scheduler/cache/routing policies.

## Quickstart

From the repo root:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
torchinferno audit
torchinferno dsv4-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
```

Without installing the package:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli audit
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli batch-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli sim-smoke
PYTHONPATH=src python3 -m torchinferno.cli traffic-smoke
PYTHONPATH=src python3 -m torchinferno.cli serve-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli perf-smoke
PYTHONPATH=src python3 -m torchinferno.cli profile-run .torchinferno_runs/dsv4-cpu --device cpu
PYTHONPATH=src python3 -m torchinferno.cli research-smoke
```

CUDA is optional for the tests, but the DSv4 smoke can run on GPU:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke --device cuda
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cuda
```

## DSv4 End-To-End Path

The compact model lives in `src/torchinferno/models/dsv4.py`. It is the fast
local harness for compiler, cache, batching, and graph-pass experiments.

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

The tests compare incremental cached decode logits against a full causal
forward pass, then exercise generation, local checkpoint save/load, CLI
execution, batching, tracing, and runtime scaffolds.

## Native DeepSeek Path

The native model lives in `src/torchinferno/models/deepseek.py`. It mirrors the
production DeepSeek-style tensor contracts rather than compressing them into the
compact DSv4 harness.

```python
import torch

from torchinferno import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config

config = tiny_deepseek_v32_config(vocab_size=128, max_position_embeddings=32)
model = DeepSeekV32ForCausalLM(config).eval()
prompt = torch.tensor([[1, 2, 3, 4]])

with torch.inference_mode():
    output = model.generate(prompt, max_new_tokens=4)

print(output)
```

Run a native architecture smoke:

```bash
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu --no-q-lora
PYTHONPATH=src python3 -m torchinferno.cli deepseek-smoke --device cpu --cache-backend paged
```

## Compiler And Graph Work

TorchInferno keeps compiler hooks explicit:

- `torchinferno.compiler.compile_forward` wraps `torch.compile` policy.
- `torchinferno.graph.trace_with_make_fx` wraps `make_fx` and optional fake
  tensor tracing.
- `torchinferno.graph.PassRegistry` registers ordered FX graph passes.
- `replace_call_function_targets` handles leaf target swaps.
- `replace_subgraph_pattern` handles multi-node symbolic FX and make_fx/ATen
  replacements for fused custom-kernel regions.
- `torchinferno.kernels.passes.register_kernel_replacement_passes` wires
  reference leaf calls and example fused subgraphs to TorchInferno kernel APIs.

Trace a DSv4 attention slice:

```bash
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu --tokens 2
PYTHONPATH=src python3 -m torchinferno.cli trace-smoke --device cpu --tokens 2 --fake --print-graph
```

Compile the DSv4 forward path:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-smoke \
  --device cpu \
  --batch-size 1 \
  --prompt-tokens 2 \
  --new-tokens 1 \
  --compile
```

First compile is expected to be much slower than eager execution.

## Profile Artifact Runs

`profile-run` is the one-command loop for agent and kernel work. It runs one
generation, captures a forward `make_fx` graph, collects profiler and memory
data, and writes a standalone repro script into one artifact directory.

```bash
PYTHONPATH=src python3 -m torchinferno.cli profile-run .torchinferno_runs/dsv4-gpu \
  --model-kind dsv4 \
  --device cuda \
  --batch-size 1 \
  --prompt-tokens 8 \
  --new-tokens 8
```

The output directory contains:

- `manifest.json`: artifact index.
- `run_config.json` and `environment.json`: exact run setup.
- `input_ids.json` and `output.json`: deterministic repro inputs and tokens.
- `graph.json`, `graph.txt`, and `graph_module.py`: captured forward graph.
- `operator_profile.json`: `torch.profiler` key averages as JSON.
- `chrome_trace.json`: Chrome/Perfetto trace from `torch.profiler`.
- `memory_profile.json`: CUDA allocator stats when running on GPU.
- `repro.py`: standalone repro for rerunning the same workload.

Useful variants:

```bash
PYTHONPATH=src python3 -m torchinferno.cli profile-run .torchinferno_runs/deepseek-paged \
  --model-kind deepseek \
  --device cuda \
  --cache-backend paged

python3 .torchinferno_runs/dsv4-gpu/repro.py --device cuda
```

## Runtime Scaffolds

The runtime package is where simulation-first serving work belongs:

- `runtime.batching`: ragged request boundary and continuous batching buckets.
- `runtime.simulation`: simple virtual GPU time slicing.
- `runtime.scheduler`: disaggregated prefill/decode planning.
- `runtime.fake_dist`: fake process groups and collectives.
- `runtime.paged`: page-table-shaped KV cache allocation and materialization.
- `runtime.paged_attention`: correct torch paged causal attention reference.
- `runtime.prefix`: radix prefix tree and prefix-aware routing.
- `runtime.prefix_cache`: prefix cache lookup for reusable KV entries.
- `runtime.serving`: token-step continuous serving engine for native DeepSeek
  cache-policy and prefix-aware request experiments.
- `runtime.traffic`: deterministic request burst/diversity simulation.
- `runtime.flex`: flex-attention-shaped API with an eager fallback.
- `runtime.cudagraphs`: named piecewise CUDA graph execution API.
- `runtime.monarch`: optional Monarch adapter with fake process world fallback.

Example disaggregated simulation:

```bash
PYTHONPATH=src python3 -m torchinferno.cli sim-smoke \
  --prefill-us-per-token 2 \
  --decode-us-per-token 4 \
  --network-latency-us 10
PYTHONPATH=src python3 -m torchinferno.cli traffic-smoke --requests 32 --burst-size 8
PYTHONPATH=src python3 -m torchinferno.cli serve-smoke --device cpu --cache-backend paged
```

## Text And Validation

`load_model_auto` chooses the compact DSv4 or native DeepSeek model from the
checkpoint config. `load_text_tokenizer` loads a local `tokenizer.json` or a
Hugging Face tokenizer behind a minimal encode/decode interface.

```bash
PYTHONPATH=src python3 -m torchinferno.cli text-generate /path/to/checkpoint "hello world" \
  --tokenizer /path/to/tokenizer \
  --device cuda
```

Known-logit references make checkpoint bringup reproducible without a serving
stack:

```bash
PYTHONPATH=src python3 -m torchinferno.cli capture-logits /path/to/checkpoint reference.json \
  --input-ids 1 2 3 \
  --device cuda
PYTHONPATH=src python3 -m torchinferno.cli validate-logits /path/to/checkpoint reference.json \
  --device cuda
```

## Research Harnesses

`torchinferno.research.ResearchHarness` is intentionally small: register named
experiments, run them, and compare metrics. It is enough for agents and humans
to add repeatable policy experiments without inventing a new harness each time.

```bash
PYTHONPATH=src python3 -m torchinferno.cli research-smoke
```

## Custom Kernels

`torchinferno.kernels` exposes production-facing kernel APIs with deterministic
torch fallbacks:

- `rms_norm`: Triton CUDA RMSNorm when CUDA/Triton are available, torch fallback
  otherwise.
- `swiglu_activation`: Triton CUDA SwiGLU activation when available, torch
  fallback otherwise.
- `fused_rmsnorm_swiglu`: graph-friendly custom op for residual-add, RMSNorm,
  weighted gate/up projection, and SwiGLU with Triton CUDA and torch fallback.
- `paged_decode_attention`: decode-token paged attention that uses Triton for
  CUDA-compatible single-token decode and falls back to the torch reference.
- `quantize_nvfp4`, `dequantize_nvfp4`, and `nvfp4_linear_reference`: a stable
  NVFP4 quantized-linear contract for graph passes and future fused kernels.

The DSv4 model calls these APIs from `RMSNorm` and `SwiGLUExpert`, so the model
keeps one torch-native path while the kernel backend can evolve behind a narrow
interface. Tests compare CUDA Triton outputs against torch references when CUDA
is available.

Benchmark reference versus specialized paged attention:

```bash
PYTHONPATH=src python3 -m torchinferno.cli perf-smoke --device cuda \
  --heads 8 \
  --seq-len 1024 \
  --head-dim 64 \
  --value-dim 64
```

## Agent Workflow

`AGENTS.md` gives coding agents a short map of where to put model, compiler,
runtime, scheduler, and experiment changes. The repo is intentionally organized
so an agent can add one focused harness or graph pass and verify it with CPU
tests before touching CUDA-only code.

`docs/ROADMAP.md` and `torchinferno audit` keep the implementation status
explicit so scaffolds are not confused for finished production paths.

## Checkpoints

TorchInferno can save and load DSv4-compatible checkpoints with a Hugging
Face-style layout:

```python
model.save_pretrained("/tmp/tiny-dsv4")
loaded = DSv4ForCausalLM.from_pretrained("/tmp/tiny-dsv4")
```

The CLI can load a local checkpoint directory or a Hub repo ID:

```bash
torchinferno dsv4-hf-smoke /tmp/tiny-dsv4 --device cpu
```

For private or gated Hub repos, provide credentials through the environment,
not command-line flags:

```bash
export HF_TOKEN=...
torchinferno dsv4-hf-smoke org-or-user/repo-name --device cuda
```

TorchInferno also has DeepSeek-style checkpoint audit and conversion commands:

```bash
PYTHONPATH=src python3 -m torchinferno.cli dsv4-audit /path/to/deepseek-checkpoint
PYTHONPATH=src python3 -m torchinferno.cli dsv4-convert /path/to/deepseek-checkpoint /tmp/torchinferno-dsv4
PYTHONPATH=src python3 -m torchinferno.cli deepseek-audit /path/to/deepseek-checkpoint
PYTHONPATH=src python3 -m torchinferno.cli deepseek-convert /path/to/deepseek-checkpoint /tmp/torchinferno-native
```

The converters index safetensor shard metadata without loading the whole model,
check every tensor shape against the target model, write sharded safetensors,
and store `torchinferno_conversion_report.json` in the output directory.

Use `deepseek-audit` and `deepseek-convert` for native production architecture
checkpoints. The older `dsv4-*` conversion path remains intentionally
conservative for the compact DSv4 harness and still refuses tensors DSv4 cannot
represent exactly.

## Repository Layout

```text
src/torchinferno/
  audit.py                Environment and feature readiness audit.
  compiler.py             torch.compile policy helper.
  cli.py                  CLI smoke runners.
  tokenization.py         Tokenizer adapters for text IO.
  validation.py           Known-logit capture and validation.
  kernels/ops.py          Kernel APIs with torch fallbacks.
  kernels/triton_ops.py   Triton CUDA RMSNorm and SwiGLU kernels.
  kernels/paged_attention.py
                           Paged decode attention kernel API.
  kernels/nvfp4.py        NVFP4 quantized-linear reference contract.
  kernels/passes.py       Graph-pass registration for kernel replacements.
  models/dsv4.py          DSv4-style causal LM, MoE, attention, and KV cache.
  models/auto.py          Config-driven model loader.
  models/deepseek.py      Native DeepSeek-V3.2-style architecture.
  models/conversion.py    DeepSeek-style checkpoint audit and conversion.
  models/hf.py            Hugging Face-style config and weights IO.
  graph/export.py         make_fx and FakeTensor tracing helper.
  graph/passes.py         FX graph pass registry and replacement helpers.
  runtime/batching.py     Ragged request batching harness.
  runtime/cudagraphs.py   Piecewise CUDA graph runner API.
  runtime/fake_dist.py    Fake process groups and collectives.
  runtime/flex.py         Flex-attention-shaped fallback.
  runtime/monarch.py      Monarch adapter point.
  runtime/paged.py        Paged KV cache scaffold.
  runtime/paged_attention.py
                           Paged causal attention reference.
  runtime/prefix.py       Radix prefix and prefix-aware routing.
  runtime/prefix_cache.py Prefix cache lookup.
  runtime/scheduler.py    Disaggregated prefill/decode planner.
  runtime/serving.py      Token-step continuous serving engine.
  runtime/simulation.py   Time-sliced virtual GPU simulator.
  runtime/traffic.py      Bursty traffic simulation.
  research/harness.py     Auto research experiment harness.
  research/benchmarks.py  Local benchmark helpers.
tests/
  test_dsv4_e2e.py        DSv4 e2e and original runtime tests.
  test_deepseek_native.py Native architecture, cache, conversion, and CLI tests.
  test_conversion_and_kernels.py
                           Checkpoint conversion and custom kernel tests.
  test_scaffolding.py     Compiler/runtime/research scaffold tests.
  test_production_workflows.py
                           Text, validation, paged attention, prefix, traffic tests.
  test_performance_specialization.py
                           Triton paged decode, NVFP4, and benchmark tests.
  test_serving_engine.py   Native paged-cache serving engine tests.
AGENTS.md                 Short contribution map for coding agents.
docs/ROADMAP.md           Feature readiness map and production milestones.
```

## Design Coverage

Implemented as working code and tests:

- DSv4 local inference path.
- Native DeepSeek-V3.2-style inference path.
- DeepSeek-style checkpoint audit and exact compatible conversion.
- Auto model loading.
- Tokenizer-backed text generation.
- Known-logit capture and validation.
- One-command profile artifact capture with graph JSON, profiler JSON, memory
  JSON, Chrome trace export, and standalone repro generation.
- `torch.compile` smoke path.
- `make_fx` and fake tensor trace helper.
- Fake process groups.
- Time-sliced multi-rank simulation.
- Disaggregated prefill/decode planning.
- Ragged and continuous batching.
- Pattern-match graph replacement entry point, including a multi-node
  make_fx/ATen subgraph replacement example.
- Native DeepSeek dense/paged cache backend selection.
- Paged KV allocation and paged causal attention reference.
- Token-step continuous serving engine with prefix-hit accounting, shared
  prefix-page reuse, persistent row-assigned paged cache state, and same-shape
  prefill/decode microbatching.
- Radix/prefix-aware routing and prefix cache lookup.
- Bursty traffic simulation.
- Piecewise CUDA graph API scaffold.
- Flex-attention-shaped fallback.
- Auto research harness.
- Optional Monarch integration point.
- Triton CUDA RMSNorm and SwiGLU kernels with torch fallbacks.
- Triton CUDA fused residual-add/RMSNorm/weighted-SwiGLU custom op with torch
  fallback and fake-tensor trace support.
- Triton-backed paged decode attention with torch fallback.
- NVFP4 quantized-linear reference and graph-pass hook.
- Local benchmark harness and CLI.

Still intentionally future work:

- Validated reference files for downloaded production weights.
- Fully ragged persistent batching and production paged/radix/flex attention
  kernels without prefill materialization.
- Fused MoE/NVFP4 kernels beyond the current correct references and
  decode-focused Triton specialization.
- Piecewise CUDA graph capture with static device buffers.
- Monarch-backed distributed execution instead of the fake fallback.
- Full production serving scheduler with cancellation and async transport.

## Development Principles

- Keep model code torch-native and easy to trace.
- Put experimental runtime policy in explicit harnesses, not hidden globals.
- Prefer simulation-first workflows over heavyweight serving bootstraps.
- Make it easy to zoom into a layer, graph region, cache policy, or scheduler.
- Keep graph and kernel replacement hooks explicit so SOTA paths do not make
  baseline code hard to read.
- Upstream generic PyTorch improvements where possible, and keep TorchInferno
  thin when the platform can carry the abstraction cleanly.
