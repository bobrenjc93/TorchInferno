from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


def swiglu_activation_reference(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


try:
    torch.fx.wrap("swiglu_activation_reference")
except Exception:
    pass


def swiglu_activation(
    gate: Tensor,
    up: Tensor,
    *,
    config: Optional[KernelConfig] = None,
) -> Tensor:
    """SwiGLU activation with a Triton CUDA implementation and torch fallback."""

    config = KernelConfig() if config is None else config
    if _should_use_triton(gate, up, config):
        from torchinferno.kernels.triton_ops import triton_swiglu_activation

        return triton_swiglu_activation(gate, up)
    return swiglu_activation_reference(gate, up)


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
