from __future__ import annotations

from torchinferno.graph import PassRegistry, replace_call_function_targets
from torchinferno.kernels.nvfp4 import nvfp4_linear_reference
from torchinferno.kernels.ops import swiglu_activation, swiglu_activation_reference


def register_kernel_replacement_passes(registry: PassRegistry) -> None:
    """Register graph replacements that route reference ops to kernel APIs."""

    registry.register(
        "swiglu-reference-to-kernel",
        replace_call_function_targets({swiglu_activation_reference: swiglu_activation}),
        "Replace leaf SwiGLU reference calls with the TorchInferno kernel API.",
    )
    registry.register(
        "nvfp4-linear-reference-marker",
        replace_call_function_targets({nvfp4_linear_reference: nvfp4_linear_reference}),
        "Mark NVFP4 linear call sites as the stable target for future fused kernels.",
    )
