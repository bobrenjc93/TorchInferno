from __future__ import annotations

from torchinferno.models.provenance import ModelVariantSpec


VARIANTS = (
    ModelVariantSpec(
        family="deepseek-v4",
        variant="v0",
        stage="torch-reference",
        parents=(),
        module="torchinferno.models.deepseek_v4.model",
        class_name="DeepSeekV4ForCausalLM",
        ops_module="torchinferno.models.deepseek_v4.ops",
        status="reference",
        notes="Pure torch DeepSeek-V4 mHC, sparse compressed attention, hash routing, and heterogeneous cache reference.",
    ),
    ModelVariantSpec(
        family="deepseek-v4",
        variant="tp-v0",
        stage="tensor-expert-parallel-production-scale",
        parents=("v0",),
        module="torchinferno.models.deepseek_v4.tensor_parallel",
        class_name="DeepSeekV4TensorParallelForCausalLM",
        ops_module="torchinferno.models.deepseek_v4.ops",
        status="experimental",
        notes="Streaming public-checkpoint loader with TP/EP collectives, heterogeneous cache, TileLang kernels, and optional grouped MXFP4 Marlin experts.",
    ),
)
