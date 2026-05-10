from __future__ import annotations

from torchinferno.models.provenance import ModelVariantSpec


VARIANTS = (
    ModelVariantSpec(
        family="deepseek-v3.2",
        variant="v0",
        stage="make-fx-reference",
        parents=(),
        module="torchinferno.models.deepseek_v32.v0",
        class_name="DeepSeekV32V0ForCausalLM",
        ops_module="torchinferno.models.deepseek_v32.raw_ops",
        status="reference",
        notes="Shape-specialized make_fx graph of the traceable full-prefix DeepSeek-V3.2 model; print_readable is available on the variant.",
    ),
    ModelVariantSpec(
        family="deepseek-v3.2",
        variant="v1",
        stage="fused-cached",
        parents=("v0",),
        module="torchinferno.models.deepseek_v32.v1",
        class_name="DeepSeekV32V1ForCausalLM",
        ops_module="torchinferno.models.deepseek_v32.fused_ops",
        status="integrated",
        notes="Native cached DeepSeek-V3.2 implementation with kernel and paged-cache hooks.",
    ),
)
