from __future__ import annotations

from torch import Tensor

from torchinferno.kernels import rms_norm as kernel_rms_norm
from torchinferno.kernels import swiglu_activation
from torchinferno.models.llama3 import raw_ops


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return kernel_rms_norm(x, weight, eps=eps)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return swiglu_activation(gate, up)


def rotary_cache(head_dim: int, positions: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    return raw_ops.rotary_cache(head_dim, positions, theta)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return raw_ops.apply_rotary(x, cos, sin)


def repeat_kv(x: Tensor, repeats: int) -> Tensor:
    return raw_ops.repeat_kv(x, repeats)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    return raw_ops.causal_attention(q, k, v, positions)
