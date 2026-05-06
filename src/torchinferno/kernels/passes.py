from __future__ import annotations

import torch
import torch.nn.functional as F

from torchinferno.graph import PassRegistry, replace_call_function_targets, replace_subgraph_pattern
from torchinferno.kernels.nvfp4 import nvfp4_linear_reference
from torchinferno.kernels.ops import fused_rmsnorm_swiglu
from torchinferno.kernels.ops import swiglu_activation, swiglu_activation_reference


_FUSED_RMSNORM_SWIGLU_EPS = 1e-6


def _rmsnorm_swiglu_pattern(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    hidden = x + residual
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(variance + _FUSED_RMSNORM_SWIGLU_EPS).to(dtype=hidden.dtype) * norm_weight
    gate = normed * gate_weight
    up = normed * up_weight
    return F.silu(gate) * up


def _rmsnorm_swiglu_replacement(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    return fused_rmsnorm_swiglu(
        x,
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        eps=_FUSED_RMSNORM_SWIGLU_EPS,
    )


def _make_aten_rmsnorm_swiglu_pass():
    from torch.fx.experimental.proxy_tensor import make_fx

    sample_args = (
        torch.ones(2, 8),
        torch.full((2, 8), 2.0),
        torch.full((8,), 3.0),
        torch.full((8,), 4.0),
        torch.full((8,), 5.0),
    )
    pattern = make_fx(_rmsnorm_swiglu_pattern)(*sample_args)
    replacement = make_fx(_rmsnorm_swiglu_replacement)(*sample_args)
    return replace_subgraph_pattern(
        pattern,
        replacement,
        metadata_key="fused_rmsnorm_swiglu_aten_matches",
    )


def register_kernel_replacement_passes(registry: PassRegistry) -> None:
    """Register graph replacements that route reference ops to kernel APIs."""

    registry.register(
        "fused-rmsnorm-swiglu-symbolic-subgraph",
        replace_subgraph_pattern(
            _rmsnorm_swiglu_pattern,
            _rmsnorm_swiglu_replacement,
            metadata_key="fused_rmsnorm_swiglu_symbolic_matches",
        ),
        "Replace symbolic residual-add/RMSNorm/weighted-SwiGLU subgraphs with a fused TorchInferno custom op.",
    )
    registry.register(
        "fused-rmsnorm-swiglu-aten-subgraph",
        _make_aten_rmsnorm_swiglu_pass(),
        "Replace make_fx ATen residual-add/RMSNorm/weighted-SwiGLU subgraphs with a fused TorchInferno custom op.",
    )
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
