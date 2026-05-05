from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


NVFP4_CODEBOOK = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class NVFP4Tensor:
    packed: Tensor
    scale: Tensor
    shape: tuple[int, ...]
    block_size: int


def quantize_nvfp4(tensor: Tensor, *, block_size: int = 16) -> NVFP4Tensor:
    """Quantize a tensor into a small NVFP4-like packed representation.

    This is a torch-native reference path for graph rewrites and checkpoint
    experiments. The codebook is intentionally explicit so kernel replacements
    can preserve the contract while specializing the implementation.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    flat = tensor.detach().float().flatten()
    original_numel = flat.numel()
    padded_numel = _round_up(original_numel, block_size)
    if padded_numel != original_numel:
        flat = torch.cat([flat, flat.new_zeros(padded_numel - original_numel)])
    blocks = flat.view(-1, block_size)
    max_code = NVFP4_CODEBOOK.abs().max().to(device=blocks.device)
    scale = (blocks.abs().amax(dim=-1) / max_code).clamp_min(1e-12)
    normalized = blocks / scale[:, None]
    codebook = NVFP4_CODEBOOK.to(device=blocks.device, dtype=normalized.dtype)
    distances = (normalized[:, :, None] - codebook[None, None, :]).abs()
    codes = torch.argmin(distances, dim=-1).to(torch.uint8).flatten()
    if codes.numel() % 2 != 0:
        codes = torch.cat([codes, codes.new_zeros(1)])
    packed = codes[0::2] | (codes[1::2] << 4)
    return NVFP4Tensor(packed=packed.contiguous(), scale=scale.contiguous(), shape=tuple(tensor.shape), block_size=block_size)


def dequantize_nvfp4(tensor: NVFP4Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    codes = torch.empty(tensor.packed.numel() * 2, device=tensor.packed.device, dtype=torch.long)
    codes[0::2] = (tensor.packed & 0x0F).long()
    codes[1::2] = (tensor.packed >> 4).long()
    total = _round_up(_numel(tensor.shape), tensor.block_size)
    codes = codes[:total]
    codebook = NVFP4_CODEBOOK.to(device=tensor.packed.device)
    values = codebook[codes].view(-1, tensor.block_size) * tensor.scale[:, None].float()
    values = values.flatten()[: _numel(tensor.shape)].view(tensor.shape)
    return values.to(dtype=dtype) if dtype is not None else values


def nvfp4_linear_reference(x: Tensor, weight: NVFP4Tensor, bias: Tensor | None = None) -> Tensor:
    dequantized = dequantize_nvfp4(weight, dtype=x.dtype)
    output = torch.matmul(x, dequantized.t())
    if bias is not None:
        output = output + bias
    return output


try:
    torch.fx.wrap("nvfp4_linear_reference")
except Exception:
    pass


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
