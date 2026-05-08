from __future__ import annotations

from dataclasses import dataclass

import torch

from torchinferno.kernels.ops import helion_available, triton_available
from torchinferno.runtime.flex import flex_attention_available
from torchinferno.runtime.monarch import monarch_available


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
    monarch_available: bool


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
            f"monarch_available={env.monarch_available}",
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
        monarch_available=monarch_available(),
    )
    return TorchInfernoAudit(
        environment=environment,
        features=(
            FeatureAudit("torch.compile", "integrated", "CompileConfig and CLI smoke cover eager-to-compiled policy."),
            FeatureAudit("make_fx/fake tensors", "integrated", "Trace helper supports make_fx with FakeTensorMode."),
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
            FeatureAudit("monarch", "adapter", "Runtime detects Monarch and otherwise routes to FakeProcessWorld."),
            FeatureAudit("flex attention", "bridge", "Runtime dispatches to torch flex attention when available and keeps the q/k/v eager fallback."),
            FeatureAudit("piecewise cudagraphs", "bridge", "Named pieces can capture static CUDA tensor inputs and recapture on shape changes."),
            FeatureAudit("paged attention", "integrated", "Native DeepSeek paged prefill/decode attend over request page tables with torch/Triton decode fallback."),
            FeatureAudit(
                "agent rank files",
                "bridge",
                "disagg-init emits editable prefill/decode rank files with local RPC wrappers.",
            ),
            FeatureAudit(
                "graph pattern replacement",
                "bridge",
                "Leaf swaps and multi-node symbolic/make_fx subgraph replacement route into fused custom ops.",
            ),
            FeatureAudit(
                "Helion candidate kernels",
                "experimental" if helion_available() else "optional",
                "helion-search-fx enumerates FX windows and benchmarks generated kernels before any production swap.",
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
                "openai-server exposes /v1/models, batched streaming /v1/chat/completions, and explicit torchrun Llama TP worker mode.",
            ),
            FeatureAudit("disaggregated prefill/decode", "simulated", "Planner models rank assignment and network latency."),
            FeatureAudit("NVFP4 graph passes", "reference", "NVFP4 tensor contract and pass hook exist; production fused kernel remains open."),
            FeatureAudit("research harness", "minimal", "Named experiments and metric comparison are available."),
            FeatureAudit(
                "model variant provenance",
                "reference",
                "DSv4, DeepSeek-V3.2, and Llama3 have v0/v1 raw/fused variant registries.",
            ),
            FeatureAudit(
                "eager-vs-optimized logit validation",
                "integrated",
                "validate-model-variants compares tiny eager v0 logits against optimized variants with a 1% default tolerance and JSON reports for agent loops.",
            ),
        ),
    )
