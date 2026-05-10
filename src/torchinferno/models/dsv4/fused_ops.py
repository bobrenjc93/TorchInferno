from __future__ import annotations

from torch import Tensor

from torchinferno.kernels import rms_norm as kernel_rms_norm
from torchinferno.kernels import swiglu_activation
from torchinferno.models.dsv4 import raw_ops


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return kernel_rms_norm(x, weight, eps=eps)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return swiglu_activation(gate, up)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return raw_ops.apply_rotary(x, cos, sin)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    # Keep the same visible contract as raw_ops. A future v2 can replace this
    # with flex/paged attention while retaining v1 as provenance.
    return raw_ops.causal_attention(q, k, v, positions)
