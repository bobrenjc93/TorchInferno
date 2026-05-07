from __future__ import annotations

from torchinferno.models.provenance import ModelVariantSpec


VARIANTS = (
    ModelVariantSpec(
        family="llama3",
        variant="v0",
        stage="raw-python-reference",
        parents=(),
        module="torchinferno.models.llama3_family.v0",
        class_name="Llama3V0ForCausalLM",
        ops_module="torchinferno.models.llama3_family.raw_ops",
        status="reference",
        notes="Torch-native Llama3 full-prefix reference with raw Python/PyTorch ops.",
    ),
    ModelVariantSpec(
        family="llama3",
        variant="v1",
        stage="fused-ops",
        parents=("v0",),
        module="torchinferno.models.llama3_family.v1",
        class_name="Llama3V1ForCausalLM",
        ops_module="torchinferno.models.llama3_family.fused_ops",
        status="reference",
        notes="Same parameter layout as v0 with kernel-backed norm/SwiGLU hooks.",
    ),
)
