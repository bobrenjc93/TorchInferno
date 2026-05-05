from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guarded by caller
    raise RuntimeError("Triton is required for torchinferno.kernels.triton_ops") from exc


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr) -> None:
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
    out = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swiglu_activation(gate: Tensor, up: Tensor) -> Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n_elements = out.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _swiglu_kernel[grid](gate, up, out, n_elements, block_size, num_warps=4)
    return out


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(variance + eps) * weight
    tl.store(out_ptr + row * n_cols + offsets, y, mask=mask)


def triton_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if x.size(-1) != weight.numel():
        raise ValueError("RMSNorm weight size must match x.size(-1)")
    x_2d = x.contiguous().view(-1, x.size(-1))
    weight = weight.contiguous()
    out = torch.empty_like(x_2d)
    n_cols = x_2d.size(1)
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 8192:
        raise ValueError("Triton RMSNorm fallback supports hidden sizes up to 8192")
    _rms_norm_kernel[(x_2d.size(0),)](
        x_2d,
        weight,
        out,
        n_cols,
        eps,
        block_size,
        num_warps=8,
    )
    return out.view_as(x)
