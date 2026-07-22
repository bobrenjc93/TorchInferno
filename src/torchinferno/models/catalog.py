from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamilySpec:
    """TorchTitan-style model family package metadata."""

    name: str
    package: str
    model_module: str
    registry_module: str
    checkpoint_adapter_module: str | None
    notes: str


MODEL_FAMILIES: tuple[ModelFamilySpec, ...] = (
    ModelFamilySpec(
        name="dsv4",
        package="torchinferno.models.dsv4",
        model_module="torchinferno.models.dsv4.model",
        registry_module="torchinferno.models.dsv4.registry",
        checkpoint_adapter_module="torchinferno.models.conversion",
        notes="Compact DeepSeek-style CPU-friendly reference family.",
    ),
    ModelFamilySpec(
        name="deepseek-v3.2",
        package="torchinferno.models.deepseek_v32",
        model_module="torchinferno.models.deepseek_v32.model",
        registry_module="torchinferno.models.deepseek_v32.registry",
        checkpoint_adapter_module="torchinferno.models.conversion",
        notes="Native DeepSeek-V3.2 tensor-contract family.",
    ),
    ModelFamilySpec(
        name="deepseek-v4",
        package="torchinferno.models.deepseek_v4",
        model_module="torchinferno.models.deepseek_v4.model",
        registry_module="torchinferno.models.deepseek_v4.registry",
        checkpoint_adapter_module="torchinferno.models.deepseek_v4.checkpoint",
        notes="Native DeepSeek-V4 mHC, heterogeneous compressed-attention, and hash-MoE family.",
    ),
    ModelFamilySpec(
        name="llama3",
        package="torchinferno.models.llama3",
        model_module="torchinferno.models.llama3.model",
        registry_module="torchinferno.models.llama3.registry",
        checkpoint_adapter_module=None,
        notes="Llama3 reference family plus production-scale load/generate adapters.",
    ),
)

_MODEL_FAMILIES_BY_NAME = {family.name: family for family in MODEL_FAMILIES}
FAMILY_ALIASES = {
    "deepseek-v32": "deepseek-v3.2",
    "deepseek_v32": "deepseek-v3.2",
    "dsv3.2": "deepseek-v3.2",
    "deepseek_v4": "deepseek-v4",
    "dsv4-flash": "deepseek-v4",
}


def list_model_families() -> tuple[ModelFamilySpec, ...]:
    return MODEL_FAMILIES


def get_model_family(name: str) -> ModelFamilySpec:
    return _MODEL_FAMILIES_BY_NAME[FAMILY_ALIASES.get(name, name)]
