from __future__ import annotations

from dataclasses import dataclass

import torch

from torchinferno.kernels.ops import triton_available
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
                "profile-run/profile-subgraph/profile-region/profile-pattern write focused graphs, profiles, traces, comparisons, and repros.",
            ),
            FeatureAudit("fake process groups", "integrated", "Single-process fake collectives cover distributed policy tests."),
            FeatureAudit("monarch", "adapter", "Runtime detects Monarch and otherwise routes to FakeProcessWorld."),
            FeatureAudit("flex attention", "reference", "Fallback q/k/v contract exists; real flex dispatch is still future work."),
            FeatureAudit("piecewise cudagraphs", "scaffold", "Named pieces exist; static CUDA capture buffers are not implemented."),
            FeatureAudit("paged attention", "integrated", "Native DeepSeek can use paged decode with torch/Triton fallback."),
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
                "prefix KV reuse",
                "integrated",
                "Serving aliases reusable prefix pages inside persistent native paged caches.",
            ),
            FeatureAudit(
                "continuous batching",
                "bridge",
                "Persistent row-assigned cache batches same-length prefill/decode groups without temporary cache rebuilds.",
            ),
            FeatureAudit("disaggregated prefill/decode", "simulated", "Planner models rank assignment and network latency."),
            FeatureAudit("NVFP4 graph passes", "reference", "NVFP4 tensor contract and pass hook exist; production fused kernel remains open."),
            FeatureAudit("research harness", "minimal", "Named experiments and metric comparison are available."),
        ),
    )
