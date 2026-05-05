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


@triton.jit
def _paged_decode_attention_kernel(
    query_ptr,
    key_pages_ptr,
    value_pages_ptr,
    page_ids_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    position: tl.constexpr,
    page_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    value_block = tl.program_id(1)
    offs_s = tl.arange(0, block_s)
    offs_d = tl.arange(0, block_d)
    page = tl.load(page_ids_ptr + (offs_s // page_size), mask=offs_s < seq_len, other=0)
    page_offset = offs_s % page_size
    key_offsets = (((page[:, None] * n_heads + head) * page_size + page_offset[:, None]) * head_dim) + offs_d[None, :]
    q = tl.load(query_ptr + head * head_dim + offs_d, mask=offs_d < head_dim, other=0.0).to(tl.float32)
    k = tl.load(key_pages_ptr + key_offsets, mask=(offs_s[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * scale
    valid = (offs_s < seq_len) & (offs_s <= position)
    scores = tl.where(valid, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)

    offs_v = value_block * block_v + tl.arange(0, block_v)
    value_offsets = (((page[:, None] * n_heads + head) * page_size + page_offset[:, None]) * value_dim) + offs_v[None, :]
    values = tl.load(
        value_pages_ptr + value_offsets,
        mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim),
        other=0.0,
    )
    out = tl.sum(values.to(tl.float32) * probs[:, None], axis=0)
    tl.store(out_ptr + head * value_dim + offs_v, out, mask=offs_v < value_dim)


def triton_paged_decode_attention(query: Tensor, cache, request_id: str, position: int) -> Tensor:
    seq = cache.sequence(request_id)
    page_ids = torch.tensor(seq.page_ids, device=query.device, dtype=torch.int64)
    out = torch.empty((query.size(0), cache.value_head_dim), device=query.device, dtype=query.dtype)
    block_s = triton.next_power_of_2(seq.length)
    block_d = triton.next_power_of_2(query.size(-1))
    block_v = min(64, triton.next_power_of_2(cache.value_head_dim))
    grid = (query.size(0), triton.cdiv(cache.value_head_dim, block_v))
    _paged_decode_attention_kernel[grid](
        query.contiguous(),
        cache.keys,
        cache.values,
        page_ids,
        out,
        seq.length,
        int(position),
        cache.page_size,
        cache.num_key_value_heads,
        cache.head_dim,
        cache.value_head_dim,
        1.0 / (query.size(-1) ** 0.5),
        block_s,
        block_d,
        block_v,
        num_warps=4,
    )
    return out
