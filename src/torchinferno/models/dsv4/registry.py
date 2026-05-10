from __future__ import annotations

from torchinferno.models.provenance import ModelVariantSpec


VARIANTS = (
    ModelVariantSpec(
        family="dsv4",
        variant="v0",
        stage="raw-python-reference",
        parents=(),
        module="torchinferno.models.dsv4.v0",
        class_name="DSv4V0ForCausalLM",
        ops_module="torchinferno.models.dsv4.raw_ops",
        status="reference",
        notes="Full-prefix recompute baseline; raw op module is the provenance anchor.",
    ),
    ModelVariantSpec(
        family="dsv4",
        variant="v1",
        stage="fused-cached",
        parents=("v0",),
        module="torchinferno.models.dsv4.v1",
        class_name="DSv4V1ForCausalLM",
        ops_module="torchinferno.models.dsv4.fused_ops",
        status="integrated",
        notes="Current DSv4 cached implementation with TorchInferno kernel API hooks.",
    ),
)
