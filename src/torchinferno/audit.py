from __future__ import annotations

from dataclasses import dataclass

import torch

from torchinferno.kernels.ops import helion_available, triton_available
from torchinferno.runtime.flex import flex_attention_available


@dataclass(frozen=True)
class FeatureAudit:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class EnvironmentAudit:
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    triton_available: bool
    helion_available: bool
    flex_attention_available: bool


@dataclass(frozen=True)
class TorchInfernoAudit:
    environment: EnvironmentAudit
    features: tuple[FeatureAudit, ...]

    def format(self) -> str:
        env = self.environment
        lines = [
            "TorchInferno audit",
            f"torch={env.torch_version}",
            f"cuda_available={env.cuda_available} cuda_device_count={env.cuda_device_count}",
            f"triton_available={env.triton_available}",
            f"helion_available={env.helion_available}",
            f"flex_attention_available={env.flex_attention_available}",
            "features:",
        ]
        for feature in self.features:
            lines.append(f"- {feature.name}: {feature.status} - {feature.detail}")
        return "\n".join(lines)


def build_audit_report() -> TorchInfernoAudit:
    environment = EnvironmentAudit(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        triton_available=triton_available(),
        helion_available=helion_available(),
        flex_attention_available=flex_attention_available(),
    )
    return TorchInfernoAudit(
        environment=environment,
        features=(
            FeatureAudit(
                "DSv4 model family",
                "integrated",
                "Compact torch-native decoder-only model covers local generation, checkpoints, tracing, serving, profiling, and conversion compatibility.",
            ),
            FeatureAudit(
                "DeepSeek-V3.2 model family",
                "integrated",
                "Native config/model/cache/checkpoint path mirrors production tensor contracts.",
            ),
            FeatureAudit(
                "Llama3 model family",
                "reference",
                "Torch-native config plus make_fx v0 and fused v1 variants cover tiny/full planning and logit validation.",
            ),
            FeatureAudit(
                "Llama3 parallel execution adapters",
                "bridge",
                "Llama 70B config plus pipeline and tensor-parallel safetensor loaders/generate paths exist; scheduler integration remains open.",
            ),
            FeatureAudit(
                "offline torch.compile experiments",
                "integrated",
                "CompileConfig and CLI smoke cover explicit compile experiments without making compilation a runtime model requirement.",
            ),
            FeatureAudit(
                "offline graph capture",
                "integrated",
                "Trace helper supports make_fx with FakeTensorMode for offline graph capture.",
            ),
            FeatureAudit(
                "profile artifact loop",
                "integrated",
                "profile-run/profile-timeslice/profile-offload/profile-subgraph/profile-region/profile-pattern write focused graphs, timelines, profiles, traces, comparisons, and repros.",
            ),
            FeatureAudit("fake process groups", "integrated", "Single-process fake collectives cover distributed policy tests."),
            FeatureAudit(
                "time-sliced profile replay",
                "integrated",
                "Representative generation profiles can be scaled and replayed across virtual GPU ranks.",
            ),
            FeatureAudit(
                "CPU offload profiling",
                "bridge",
                "Weights can be staged CPU-to-device one module at a time with movement overhead reported separately from compute.",
            ),
            FeatureAudit("flex attention", "bridge", "Runtime dispatches to torch flex attention when available and keeps the q/k/v eager fallback."),
            FeatureAudit("piecewise cudagraphs", "bridge", "Named pieces can capture static CUDA tensor inputs and recapture on shape changes."),
            FeatureAudit("paged attention", "integrated", "Native DeepSeek paged prefill/decode attend over request page tables with torch/Triton decode fallback."),
            FeatureAudit(
                "agent rank files",
                "bridge",
                "disagg-init emits editable prefill/decode rank files with local RPC wrappers.",
            ),
            FeatureAudit(
                "offline graph replacement",
                "bridge",
                "Leaf swaps and multi-node symbolic/make_fx subgraph replacement produce candidate optimized graphs before promotion.",
            ),
            FeatureAudit(
                "backend candidate providers",
                "experimental" if helion_available() else "optional",
                "Helion search is the first optional provider; promotion flow is intended to also support Triton, CuteDSL/CUTLASS, custom CUDA/C++, PyTorch custom ops, and pure PyTorch rewrites.",
            ),
            FeatureAudit(
                "prefix KV reuse",
                "integrated",
                "Serving aliases reusable prefix pages inside persistent native paged caches.",
            ),
            FeatureAudit(
                "continuous batching",
                "bridge",
                "Persistent row-assigned cache batches same-length prefill/decode groups; OpenAI serving microbatches same-shape live requests.",
            ),
            FeatureAudit(
                "OpenAI serving API",
                "bridge",
                "openai-server exposes /health, /v1/models, batched streaming /v1/chat/completions, and auto-launched Llama TP worker mode.",
            ),
            FeatureAudit(
                "benchmark suites",
                "integrated",
                "vLLM-compatible, native Llama, direct OpenAI, and HTTP OpenAI benchmark loops write repeatable JSON summaries and plots.",
            ),
            FeatureAudit(
                "text IO and known-logit validation",
                "integrated",
                "Auto loading, tokenizer-backed generation, capture-logits, and validate-logits cover checkpoint bringup without a server.",
            ),
            FeatureAudit(
                "disaggregated prefill/decode",
                "bridge",
                "Llama OpenAI serving can split one CUDA launch into TP prefill/decode roles with live NCCL KV handoff; cross-request overlap and DeepSeek support remain open.",
            ),
            FeatureAudit("NVFP4 graph passes", "reference", "NVFP4 tensor contract and pass hook exist; production fused kernel remains open."),
            FeatureAudit("research harness", "minimal", "Named experiments and metric comparison are available."),
            FeatureAudit(
                "model variant provenance",
                "reference",
                "DSv4, DeepSeek-V3.2, and Llama3 have make_fx v0 references, traceable model wrappers, v1 variants, and registry lineage.",
            ),
            FeatureAudit(
                "v0-vs-optimized logit validation",
                "integrated",
                "validate-model-variants compares tiny make_fx v0 logits against optimized variants with a 1% default tolerance and JSON reports for agent loops.",
            ),
        ),
    )
