from __future__ import annotations

from torchinferno.models.provenance import ModelVariantSpec


VARIANTS = (
    ModelVariantSpec(
        family="llama3",
        variant="v0",
        stage="make-fx-reference",
        parents=(),
        module="torchinferno.models.llama3.v0",
        class_name="Llama3V0ForCausalLM",
        ops_module="torchinferno.models.llama3.raw_ops",
        status="reference",
        notes="Shape-specialized make_fx graph of the traceable full-prefix Llama3 model; print_readable is available on the variant.",
    ),
    ModelVariantSpec(
        family="llama3",
        variant="v1",
        stage="fused-ops",
        parents=("v0",),
        module="torchinferno.models.llama3.v1",
        class_name="Llama3V1ForCausalLM",
        ops_module="torchinferno.models.llama3.fused_ops",
        status="reference",
        notes="Same parameter layout as v0 with kernel-backed norm/SwiGLU hooks.",
    ),
    ModelVariantSpec(
        family="llama3",
        variant="pipeline-v0",
        stage="pipeline-sharded-production-scale",
        parents=("v0",),
        module="torchinferno.models.llama3.pipeline",
        class_name="Llama3PipelineForCausalLM",
        ops_module="torchinferno.models.llama3.raw_ops",
        status="experimental",
        notes="Direct safetensors loader with per-layer device placement and KV-cached decode for Llama70B.",
    ),
    ModelVariantSpec(
        family="llama3",
        variant="tp-v0",
        stage="tensor-parallel-production-scale",
        parents=("pipeline-v0",),
        module="torchinferno.models.llama3.tensor_parallel",
        class_name="Llama3TensorParallelForCausalLM",
        ops_module="torchinferno.models.llama3.raw_ops",
        status="experimental",
        notes="Torchrun/NCCL tensor-parallel Llama70B path with sharded QKV/MLP weights and KV-cached decode.",
    ),
)
