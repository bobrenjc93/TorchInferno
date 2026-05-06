from torchinferno.kernels.ops import (
    KernelBackend,
    KernelConfig,
    fused_rmsnorm_swiglu,
    fused_rmsnorm_swiglu_reference,
    rms_norm,
    swiglu_activation,
    swiglu_activation_reference,
)
from torchinferno.kernels.nvfp4 import NVFP4Tensor, dequantize_nvfp4, nvfp4_linear_reference, quantize_nvfp4


def paged_decode_attention(query, cache, request_id, position, *, config=None):
    from torchinferno.kernels.paged_attention import paged_decode_attention as _paged_decode_attention

    return _paged_decode_attention(query, cache, request_id, position, config=config)

__all__ = [
    "KernelBackend",
    "KernelConfig",
    "NVFP4Tensor",
    "dequantize_nvfp4",
    "fused_rmsnorm_swiglu",
    "fused_rmsnorm_swiglu_reference",
    "nvfp4_linear_reference",
    "paged_decode_attention",
    "quantize_nvfp4",
    "rms_norm",
    "swiglu_activation",
    "swiglu_activation_reference",
]
