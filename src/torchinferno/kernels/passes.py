from __future__ import annotations

from torchinferno.graph import PassRegistry, replace_call_function_targets
from torchinferno.kernels.ops import swiglu_activation, swiglu_activation_reference


def register_kernel_replacement_passes(registry: PassRegistry) -> None:
    """Register graph replacements that route reference ops to kernel APIs."""

    registry.register(
        "swiglu-reference-to-kernel",
        replace_call_function_targets({swiglu_activation_reference: swiglu_activation}),
        "Replace leaf SwiGLU reference calls with the TorchInferno kernel API.",
    )
