from __future__ import annotations

import math

import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(dtype=x.dtype)) * weight


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return torch.nn.functional.silu(gate) * up


def rotary_cache(head_dim: int, positions: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    sin = sin.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


def repeat_kv(x: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    key_positions = torch.arange(k.size(-2), device=q.device)
    allowed = key_positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)
