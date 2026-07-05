from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


class KernelBackend(str, Enum):
    AUTO = "auto"
    TORCH = "torch"
    TRITON = "triton"


@dataclass(frozen=True)
class KernelConfig:
    backend: KernelBackend = KernelBackend.AUTO
    eps: float = 1e-6


def triton_available() -> bool:
    return find_spec("triton") is not None and find_spec("triton.language") is not None


def helion_available() -> bool:
    return find_spec("helion") is not None and find_spec("helion.language") is not None


def swiglu_activation_reference(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


def fused_rmsnorm_swiglu_reference(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Reference residual-add + RMSNorm + weighted SwiGLU region."""

    _validate_fused_rmsnorm_swiglu_inputs(x, residual, norm_weight, gate_weight, up_weight)
    hidden = x + residual
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(variance + eps).to(dtype=hidden.dtype) * norm_weight
    gate = normed * gate_weight
    up = normed * up_weight
    return F.silu(gate) * up


try:
    torch.fx.wrap("swiglu_activation_reference")
except Exception:
    pass


@torch.library.custom_op("torchinferno::fused_rmsnorm_swiglu", mutates_args=())
def _fused_rmsnorm_swiglu_custom_op(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    eps: float,
) -> Tensor:
    return _fused_rmsnorm_swiglu_impl(x, residual, norm_weight, gate_weight, up_weight, eps=eps)


@_fused_rmsnorm_swiglu_custom_op.register_fake
def _fused_rmsnorm_swiglu_fake(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    eps: float,
) -> Tensor:
    _validate_fused_rmsnorm_swiglu_inputs(x, residual, norm_weight, gate_weight, up_weight)
    return torch.empty_like(x)


def swiglu_activation(
    gate: Tensor,
    up: Tensor,
    *,
    out: Tensor | None = None,
    config: Optional[KernelConfig] = None,
) -> Tensor:
    """SwiGLU activation with a Triton CUDA implementation and torch fallback."""

    config = KernelConfig() if config is None else config
    if _should_use_triton(gate, up, config):
        from torchinferno.kernels.triton_ops import triton_swiglu_activation

        return triton_swiglu_activation(gate, up, out=out)
    result = swiglu_activation_reference(gate, up)
    if out is not None:
        out.copy_(result)
        return out
    return result


def fused_rmsnorm_swiglu(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    eps: float,
    config: Optional[KernelConfig] = None,
) -> Tensor:
    """Residual-add + RMSNorm + weighted SwiGLU with a graph-friendly fused op."""

    if config is None:
        return torch.ops.torchinferno.fused_rmsnorm_swiglu(
            x,
            residual,
            norm_weight,
            gate_weight,
            up_weight,
            float(eps),
        )
    return _fused_rmsnorm_swiglu_impl(x, residual, norm_weight, gate_weight, up_weight, eps=eps, config=config)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    *,
    eps: float,
    config: Optional[KernelConfig] = None,
) -> Tensor:
    """RMSNorm with a Triton CUDA implementation and torch fallback."""

    config = KernelConfig(eps=eps) if config is None else config
    if _should_use_triton(x, weight, config) and x.ndim >= 2:
        from torchinferno.kernels.triton_ops import triton_rms_norm

        return triton_rms_norm(x, weight, eps)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps).to(dtype=x.dtype) * weight


def _should_use_triton(left: Tensor, right: Tensor, config: KernelConfig) -> bool:
    if config.backend == KernelBackend.TORCH:
        return False
    if config.backend == KernelBackend.TRITON and not triton_available():
        raise RuntimeError("Triton backend requested but Triton is not available")
    return left.is_cuda and right.is_cuda and triton_available()


def _fused_rmsnorm_swiglu_impl(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    eps: float,
    config: Optional[KernelConfig] = None,
) -> Tensor:
    config = KernelConfig(eps=eps) if config is None else config
    _validate_fused_rmsnorm_swiglu_inputs(x, residual, norm_weight, gate_weight, up_weight)
    if _should_use_triton(x, norm_weight, config) and x.ndim >= 1:
        from torchinferno.kernels.triton_ops import triton_fused_rmsnorm_swiglu

        return triton_fused_rmsnorm_swiglu(x, residual, norm_weight, gate_weight, up_weight, eps)
    return fused_rmsnorm_swiglu_reference(x, residual, norm_weight, gate_weight, up_weight, eps=eps)


def _validate_fused_rmsnorm_swiglu_inputs(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> None:
    if x.shape != residual.shape:
        raise ValueError("x and residual tensors must have the same shape")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    hidden_size = x.size(-1)
    expected_weight_shape = (hidden_size,)
    for name, weight in (
        ("norm_weight", norm_weight),
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
    ):
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(f"{name} shape must be {expected_weight_shape}")
