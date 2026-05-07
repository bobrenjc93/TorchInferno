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
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
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
def _rotary_interleaved_inplace_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    total_pairs,
    heads: tl.constexpr,
    tokens: tl.constexpr,
    half_dim: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_head: tl.constexpr,
    stride_token: tl.constexpr,
    stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    pair_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = pair_offsets < total_pairs
    rotary_dim = pair_offsets % half_dim
    token_head_batch = pair_offsets // half_dim
    token = token_head_batch % tokens
    head_batch = token_head_batch // tokens
    head = head_batch % heads
    batch = head_batch // heads

    x_offset = (
        batch * stride_batch
        + head * stride_head
        + token * stride_token
        + rotary_dim * 2 * stride_dim
    )
    cos_sin_offset = token * half_dim + rotary_dim
    x_even = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + x_offset + stride_dim, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + cos_sin_offset, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + cos_sin_offset, mask=mask, other=0.0).to(tl.float32)

    tl.store(x_ptr + x_offset, x_even * cos - x_odd * sin, mask=mask)
    tl.store(x_ptr + x_offset + stride_dim, x_even * sin + x_odd * cos, mask=mask)


def triton_apply_rotary_interleaved_inplace(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply interleaved RoPE to tensor-parallel q/k views in place."""

    if q.size(-1) != k.size(-1):
        raise ValueError("q and k head dimensions must match")
    if q.size(-2) != k.size(-2):
        raise ValueError("q and k token dimensions must match")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    if cos.shape != (q.size(-2), q.size(-1) // 2):
        raise ValueError("rotary cache shape must be [tokens, head_dim / 2]")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must have contiguous head dimensions")
    cos = cos.contiguous()
    sin = sin.contiguous()
    _triton_rotate_one_interleaved_inplace(q, cos, sin)
    _triton_rotate_one_interleaved_inplace(k, cos, sin)
    return q, k


def _triton_rotate_one_interleaved_inplace(x: Tensor, cos: Tensor, sin: Tensor) -> None:
    batch, heads, tokens, head_dim = x.shape
    half_dim = head_dim // 2
    total_pairs = batch * heads * tokens * half_dim
    block_size = 256
    grid = (triton.cdiv(total_pairs, block_size),)
    _rotary_interleaved_inplace_kernel[grid](
        x,
        cos,
        sin,
        total_pairs,
        heads,
        tokens,
        half_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        block_size,
        num_warps=4,
    )


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
def _fused_rmsnorm_swiglu_kernel(
    x_ptr,
    residual_ptr,
    norm_weight_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_offset = row * n_cols + offsets
    x = tl.load(x_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
    hidden = x + residual
    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    scale = tl.rsqrt(variance + eps)
    norm_weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_weight = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up_weight = tl.load(up_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = hidden * scale * norm_weight
    gate = normed * gate_weight
    up = normed * up_weight
    out = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + row_offset, out, mask=mask)


def triton_fused_rmsnorm_swiglu(
    x: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    eps: float,
) -> Tensor:
    if x.shape != residual.shape:
        raise ValueError("x and residual tensors must have the same shape")
    hidden_size = x.size(-1)
    for name, weight in (
        ("norm_weight", norm_weight),
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
    ):
        if tuple(weight.shape) != (hidden_size,):
            raise ValueError(f"{name} shape must be {(hidden_size,)}")
    x_2d = x.contiguous().view(-1, hidden_size)
    residual_2d = residual.contiguous().view(-1, hidden_size)
    norm_weight = norm_weight.contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    out = torch.empty_like(x_2d)
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 8192:
        raise ValueError("Triton fused RMSNorm+SwiGLU supports hidden sizes up to 8192")
    _fused_rmsnorm_swiglu_kernel[(x_2d.size(0),)](
        x_2d,
        residual_2d,
        norm_weight,
        gate_weight,
        up_weight,
        out,
        hidden_size,
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
