from __future__ import annotations

from torchinferno.models.deepseek_v32.registry import VARIANTS as DEEPSEEK_V32_VARIANTS
from torchinferno.models.dsv4.registry import VARIANTS as DSV4_VARIANTS
from torchinferno.models.llama3.registry import VARIANTS as LLAMA3_VARIANTS
from torchinferno.models.provenance import ModelVariantRegistry, ModelVariantSpec


MODEL_VARIANTS: tuple[ModelVariantSpec, ...] = (
    *DSV4_VARIANTS,
    *DEEPSEEK_V32_VARIANTS,
    *LLAMA3_VARIANTS,
)

MODEL_VARIANT_REGISTRY = ModelVariantRegistry(MODEL_VARIANTS)
FAMILY_ALIASES = {
    "dsv3.2": "deepseek-v3.2",
    "deepseek-v32": "deepseek-v3.2",
    "deepseek_v32": "deepseek-v3.2",
}


def list_model_variants(family: str | None = None) -> tuple[ModelVariantSpec, ...]:
    family = _normalize_family(family)
    return MODEL_VARIANT_REGISTRY.list(family)


def get_model_variant(family: str, variant: str) -> ModelVariantSpec:
    family = _normalize_family(family) or family
    return MODEL_VARIANT_REGISTRY.get(family, variant)


def model_variant_lineage(family: str, variant: str) -> tuple[ModelVariantSpec, ...]:
    family = _normalize_family(family) or family
    return MODEL_VARIANT_REGISTRY.lineage(family, variant)


def _normalize_family(family: str | None) -> str | None:
    if family is None:
        return None
    return FAMILY_ALIASES.get(family, family)
