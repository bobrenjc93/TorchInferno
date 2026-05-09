from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guarded by caller
    raise RuntimeError("Triton is required for torchinferno.kernels.triton_ops") from exc


@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    cols: tl.constexpr,
    gate_stride_row: tl.constexpr,
    up_stride_row: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < cols
    gate = tl.load(gate_ptr + row * gate_stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + row * up_stride_row + offsets, mask=mask, other=0.0)
    out = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + row * cols + offsets, out, mask=mask)


def triton_swiglu_activation(gate: Tensor, up: Tensor) -> Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("gate and up tensors must have contiguous last dimensions")
    if gate.ndim == 1:
        gate_2d = gate[None, :]
        up_2d = up[None, :]
    else:
        gate_2d = gate.flatten(0, -2)
        up_2d = up.flatten(0, -2)
    out = torch.empty_like(gate_2d)
    cols = gate_2d.size(1)
    block_size = triton.next_power_of_2(cols)
    _swiglu_kernel[(gate_2d.size(0),)](
        gate_2d,
        up_2d,
        out,
        cols,
        gate_2d.stride(0),
        up_2d.stride(0),
        block_size,
        num_warps=8,
    )
    return out.view_as(gate)


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


@triton.jit
def _rotary_llama_qk_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    total_elements,
    total_q,
    total_k,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    tokens: tl.constexpr,
    half_dim: tl.constexpr,
    cache_dim: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    rotary_dim = offsets % half_dim
    token_head_batch = offsets // half_dim

    q_token = token_head_batch % tokens
    q_head_batch = token_head_batch // tokens
    q_head = q_head_batch % q_heads
    q_batch = q_head_batch // q_heads
    q_offset = (
        q_batch * q_stride_batch
        + q_head * q_stride_head
        + q_token * q_stride_token
        + rotary_dim * q_stride_dim
    )
    q_mask = mask & (offsets < total_q)
    q_first = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + q_offset + half_dim * q_stride_dim, mask=q_mask, other=0.0).to(tl.float32)
    q_cos = tl.load(cos_ptr + q_token * cache_dim + rotary_dim, mask=q_mask, other=1.0).to(tl.float32)
    q_sin = tl.load(sin_ptr + q_token * cache_dim + rotary_dim, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + q_offset, q_first * q_cos - q_second * q_sin, mask=q_mask)
    tl.store(q_ptr + q_offset + half_dim * q_stride_dim, q_second * q_cos + q_first * q_sin, mask=q_mask)

    k_token = token_head_batch % tokens
    k_head_batch = token_head_batch // tokens
    k_head = k_head_batch % k_heads
    k_batch = k_head_batch // k_heads
    k_offset = (
        k_batch * k_stride_batch
        + k_head * k_stride_head
        + k_token * k_stride_token
        + rotary_dim * k_stride_dim
    )
    k_mask = mask & (offsets < total_k)
    k_first = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_offset + half_dim * k_stride_dim, mask=k_mask, other=0.0).to(tl.float32)
    k_cos = tl.load(cos_ptr + k_token * cache_dim + rotary_dim, mask=k_mask, other=1.0).to(tl.float32)
    k_sin = tl.load(sin_ptr + k_token * cache_dim + rotary_dim, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(k_ptr + k_offset, k_first * k_cos - k_second * k_sin, mask=k_mask)
    tl.store(k_ptr + k_offset + half_dim * k_stride_dim, k_second * k_cos + k_first * k_sin, mask=k_mask)


def triton_apply_rotary_llama_inplace(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Llama/HF rotate-half RoPE to tensor-parallel q/k views in place."""

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, tokens, head_dim]")
    if q.size(0) != k.size(0):
        raise ValueError("q and k batch dimensions must match")
    if q.size(-1) != k.size(-1):
        raise ValueError("q and k head dimensions must match")
    if q.size(-2) != k.size(-2):
        raise ValueError("q and k token dimensions must match")
    if q.size(-1) % 2 != 0:
        raise ValueError("q and k head dimensions must be even")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    half_dim = q.size(-1) // 2
    if cos.shape not in {(q.size(-2), q.size(-1)), (q.size(-2), half_dim)}:
        raise ValueError("rotary cache shape must be [tokens, head_dim] or [tokens, head_dim / 2]")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must have contiguous head dimensions")
    cos = cos.contiguous()
    sin = sin.contiguous()
    batch, q_heads, tokens, _ = q.shape
    k_heads = k.size(1)
    total_q = batch * q_heads * tokens * half_dim
    total_k = batch * k_heads * tokens * half_dim
    total_elements = max(total_q, total_k)
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _rotary_llama_qk_inplace_kernel[grid](
        q,
        k,
        cos,
        sin,
        total_elements,
        total_q,
        total_k,
        q_heads,
        k_heads,
        tokens,
        half_dim,
        cos.size(-1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        block_size,
        num_warps=4,
    )
    return q, k


@triton.jit
def _rotary_llama_append_kv_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_key_ptr,
    cache_value_ptr,
    seq_start_ptr,
    cos_ptr,
    sin_ptr,
    total_elements,
    q_pairs,
    k_pairs,
    v_elements,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    cache_dim: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_batch: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    cache_stride_batch: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    seq_start = tl.load(seq_start_ptr)
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements

    q_dim = offsets % half_dim
    q_head_batch = offsets // half_dim
    q_head = q_head_batch % q_heads
    q_batch = q_head_batch // q_heads
    q_mask = mask & (offsets < q_pairs)
    q_offset = q_batch * q_stride_batch + q_head * q_stride_head + q_dim * q_stride_dim
    q_first = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + q_offset + half_dim * q_stride_dim, mask=q_mask, other=0.0).to(tl.float32)
    q_cos = tl.load(cos_ptr + q_dim, mask=q_mask, other=1.0).to(tl.float32)
    q_sin = tl.load(sin_ptr + q_dim, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + q_offset, q_first * q_cos - q_second * q_sin, mask=q_mask)
    tl.store(q_ptr + q_offset + half_dim * q_stride_dim, q_second * q_cos + q_first * q_sin, mask=q_mask)

    k_dim = offsets % half_dim
    k_head_batch = offsets // half_dim
    k_head = k_head_batch % kv_heads
    k_batch = k_head_batch // kv_heads
    k_mask = mask & (offsets < k_pairs)
    k_offset = k_batch * k_stride_batch + k_head * k_stride_head + k_dim * k_stride_dim
    cache_k_offset = (
        k_batch * cache_stride_batch
        + k_head * cache_stride_head
        + seq_start * cache_stride_token
        + k_dim * cache_stride_dim
    )
    k_first = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_offset + half_dim * k_stride_dim, mask=k_mask, other=0.0).to(tl.float32)
    k_cos = tl.load(cos_ptr + k_dim, mask=k_mask, other=1.0).to(tl.float32)
    k_sin = tl.load(sin_ptr + k_dim, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(cache_key_ptr + cache_k_offset, k_first * k_cos - k_second * k_sin, mask=k_mask)
    tl.store(
        cache_key_ptr + cache_k_offset + half_dim * cache_stride_dim,
        k_second * k_cos + k_first * k_sin,
        mask=k_mask,
    )

    v_dim = offsets % head_dim
    v_head_batch = offsets // head_dim
    v_head = v_head_batch % kv_heads
    v_batch = v_head_batch // kv_heads
    v_mask = mask & (offsets < v_elements)
    v_offset = v_batch * v_stride_batch + v_head * v_stride_head + v_dim * v_stride_dim
    cache_v_offset = (
        v_batch * cache_stride_batch
        + v_head * cache_stride_head
        + seq_start * cache_stride_token
        + v_dim * cache_stride_dim
    )
    tl.store(cache_value_ptr + cache_v_offset, tl.load(v_ptr + v_offset, mask=v_mask), mask=v_mask)


def triton_apply_rotary_append_kv_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    seq_start: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply one-token Llama RoPE to q and append rotated k/v into a dense KV cache."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, head_dim]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v batch dimensions must match")
    if q.size(-2) != 1 or k.size(-2) != 1 or v.size(-2) != 1:
        raise ValueError("fused decode rotary append expects exactly one token")
    if q.size(-1) != k.size(-1) or k.size(-1) != v.size(-1):
        raise ValueError("q, k, and v head dimensions must match")
    if q.size(-1) % 2 != 0:
        raise ValueError("head dimension must be even")
    if k.size(1) != v.size(1):
        raise ValueError("k and v head counts must match")
    if cache_keys.shape != cache_values.shape:
        raise ValueError("cache keys and values must have the same shape")
    if k.size(0) > cache_keys.size(0) or k.size(1) != cache_keys.size(1) or k.size(-1) != cache_keys.size(-1):
        raise ValueError("cache shape is incompatible with incoming k/v")
    if seq_start.numel() != 1:
        raise ValueError("dynamic KV append position must be a scalar tensor")
    if seq_start.device != cache_keys.device:
        raise ValueError("dynamic KV append position must be on the cache device")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    half_dim = q.size(-1) // 2
    if cos.shape not in {(1, half_dim), (half_dim,)}:
        raise ValueError("decode rotary cache shape must be [1, head_dim / 2]")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must have contiguous head dimensions")
    if cache_keys.stride(-1) != 1 or cache_values.stride(-1) != 1:
        raise ValueError("cache keys and values must have contiguous head dimensions")
    cos = cos.reshape(-1).contiguous()
    sin = sin.reshape(-1).contiguous()
    batch, q_heads, _, head_dim = q.shape
    kv_heads = k.size(1)
    q_pairs = batch * q_heads * half_dim
    k_pairs = batch * kv_heads * half_dim
    v_elements = batch * kv_heads * head_dim
    total_elements = max(q_pairs, k_pairs, v_elements)
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _rotary_llama_append_kv_decode_kernel[grid](
        q,
        k,
        v,
        cache_keys,
        cache_values,
        seq_start,
        cos,
        sin,
        total_elements,
        q_pairs,
        k_pairs,
        v_elements,
        q_heads,
        kv_heads,
        head_dim,
        half_dim,
        cos.numel(),
        q.stride(0),
        q.stride(1),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(3),
        cache_keys.stride(0),
        cache_keys.stride(1),
        cache_keys.stride(2),
        cache_keys.stride(3),
        block_size,
        num_warps=4,
    )
    return q


@triton.jit
def _kv_cache_append_kernel(
    key_ptr,
    value_ptr,
    cache_key_ptr,
    cache_value_ptr,
    total_elements,
    seq_start,
    heads: tl.constexpr,
    tokens: tl.constexpr,
    head_dim: tl.constexpr,
    key_stride_batch: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_batch: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_dim: tl.constexpr,
    cache_stride_batch: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    dim = offsets % head_dim
    token_head_batch = offsets // head_dim
    token = token_head_batch % tokens
    head_batch = token_head_batch // tokens
    head = head_batch % heads
    batch = head_batch // heads

    key_offset = (
        batch * key_stride_batch
        + head * key_stride_head
        + token * key_stride_token
        + dim * key_stride_dim
    )
    value_offset = (
        batch * value_stride_batch
        + head * value_stride_head
        + token * value_stride_token
        + dim * value_stride_dim
    )
    cache_offset = (
        batch * cache_stride_batch
        + head * cache_stride_head
        + (seq_start + token) * cache_stride_token
        + dim * cache_stride_dim
    )
    tl.store(cache_key_ptr + cache_offset, tl.load(key_ptr + key_offset, mask=mask), mask=mask)
    tl.store(cache_value_ptr + cache_offset, tl.load(value_ptr + value_offset, mask=mask), mask=mask)


@triton.jit
def _kv_cache_append_dynamic_kernel(
    key_ptr,
    value_ptr,
    cache_key_ptr,
    cache_value_ptr,
    seq_start_ptr,
    total_elements,
    heads: tl.constexpr,
    tokens: tl.constexpr,
    head_dim: tl.constexpr,
    key_stride_batch: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_batch: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_dim: tl.constexpr,
    cache_stride_batch: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    seq_start = tl.load(seq_start_ptr)
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    dim = offsets % head_dim
    token_head_batch = offsets // head_dim
    token = token_head_batch % tokens
    head_batch = token_head_batch // tokens
    head = head_batch % heads
    batch = head_batch // heads

    key_offset = (
        batch * key_stride_batch
        + head * key_stride_head
        + token * key_stride_token
        + dim * key_stride_dim
    )
    value_offset = (
        batch * value_stride_batch
        + head * value_stride_head
        + token * value_stride_token
        + dim * value_stride_dim
    )
    cache_offset = (
        batch * cache_stride_batch
        + head * cache_stride_head
        + (seq_start + token) * cache_stride_token
        + dim * cache_stride_dim
    )
    tl.store(cache_key_ptr + cache_offset, tl.load(key_ptr + key_offset, mask=mask), mask=mask)
    tl.store(cache_value_ptr + cache_offset, tl.load(value_ptr + value_offset, mask=mask), mask=mask)


def triton_append_kv_cache(
    keys: Tensor,
    values: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    seq_start: int | Tensor,
) -> None:
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape")
    if keys.ndim != 4:
        raise ValueError("keys and values must have shape [batch, heads, tokens, head_dim]")
    if cache_keys.shape != cache_values.shape:
        raise ValueError("cache keys and values must have the same shape")
    if keys.stride(-1) != 1 or values.stride(-1) != 1:
        raise ValueError("keys and values must have contiguous head dimensions")
    batch, heads, tokens, head_dim = keys.shape
    if batch > cache_keys.size(0) or heads != cache_keys.size(1) or head_dim != cache_keys.size(3):
        raise ValueError("cache shape is incompatible with incoming keys")
    if isinstance(seq_start, Tensor):
        if seq_start.numel() != 1:
            raise ValueError("dynamic KV append position must be a scalar tensor")
        if seq_start.device != cache_keys.device:
            raise ValueError("dynamic KV append position must be on the cache device")
    elif seq_start < 0 or seq_start + tokens > cache_keys.size(2):
        raise ValueError("KV cache capacity exceeded")
    total_elements = batch * heads * tokens * head_dim
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    if isinstance(seq_start, Tensor):
        _kv_cache_append_dynamic_kernel[grid](
            keys,
            values,
            cache_keys,
            cache_values,
            seq_start,
            total_elements,
            heads,
            tokens,
            head_dim,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            keys.stride(3),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            values.stride(3),
            cache_keys.stride(0),
            cache_keys.stride(1),
            cache_keys.stride(2),
            cache_keys.stride(3),
            block_size,
            num_warps=4,
        )
        return
    _kv_cache_append_kernel[grid](
        keys,
        values,
        cache_keys,
        cache_values,
        total_elements,
        seq_start,
        heads,
        tokens,
        head_dim,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        cache_keys.stride(0),
        cache_keys.stride(1),
        cache_keys.stride(2),
        cache_keys.stride(3),
        block_size,
        num_warps=4,
    )


@triton.jit
def _dense_gqa_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_batch: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (q_heads // kv_heads)
    offs_s = tl.arange(0, block_s)
    offs_d = tl.arange(0, block_d)
    q_offsets = batch * q_stride_batch + q_head * q_stride_head + offs_d * q_stride_dim
    k_offsets = (
        batch * k_stride_batch
        + kv_head * k_stride_head
        + offs_s[:, None] * k_stride_token
        + offs_d[None, :] * k_stride_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=offs_d < head_dim, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_offsets, mask=(offs_s[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * scale
    scores = tl.where(offs_s < seq_len, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)

    offs_v = tl.arange(0, block_v)
    v_offsets = (
        batch * v_stride_batch
        + kv_head * v_stride_head
        + offs_s[:, None] * v_stride_token
        + offs_v[None, :] * v_stride_dim
    )
    values = tl.load(v_ptr + v_offsets, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim), other=0.0)
    out = tl.sum(values.to(tl.float32) * probs[:, None], axis=0)
    out_offsets = batch * out_stride_batch + q_head * out_stride_head + offs_v * out_stride_dim
    tl.store(out_ptr + out_offsets, out, mask=offs_v < value_dim)


@triton.jit
def _dense_gqa_decode_attention_dynamic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len_ptr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_batch: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    seq_len = tl.load(seq_len_ptr)
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (q_heads // kv_heads)
    offs_s = tl.arange(0, block_s)
    offs_d = tl.arange(0, block_d)
    q_offsets = batch * q_stride_batch + q_head * q_stride_head + offs_d * q_stride_dim
    k_offsets = (
        batch * k_stride_batch
        + kv_head * k_stride_head
        + offs_s[:, None] * k_stride_token
        + offs_d[None, :] * k_stride_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=offs_d < head_dim, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_offsets, mask=(offs_s[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * scale
    scores = tl.where(offs_s < seq_len, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)

    offs_v = tl.arange(0, block_v)
    v_offsets = (
        batch * v_stride_batch
        + kv_head * v_stride_head
        + offs_s[:, None] * v_stride_token
        + offs_v[None, :] * v_stride_dim
    )
    values = tl.load(v_ptr + v_offsets, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim), other=0.0)
    out = tl.sum(values.to(tl.float32) * probs[:, None], axis=0)
    out_offsets = batch * out_stride_batch + q_head * out_stride_head + offs_v * out_stride_dim
    tl.store(out_ptr + out_offsets, out, mask=offs_v < value_dim)


def triton_dense_gqa_decode_attention(q: Tensor, k: Tensor, v: Tensor, seq_len: int | Tensor | None = None) -> Tensor:
    """Single-token dense-cache GQA attention for Llama tensor-parallel decode."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, dim]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v batch dimensions must match")
    if q.size(-2) != 1:
        raise ValueError("decode attention expects q to contain exactly one token")
    if k.size(-2) != v.size(-2):
        raise ValueError("k and v sequence lengths must match")
    if k.size(1) != v.size(1):
        raise ValueError("k and v head counts must match")
    if q.size(1) % k.size(1) != 0:
        raise ValueError("q head count must be divisible by kv head count")
    if q.size(-1) != k.size(-1):
        raise ValueError("q and k head dimensions must match")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must have contiguous head dimensions")
    batch, q_heads, _, head_dim = q.shape
    _, kv_heads, cache_tokens, _ = k.shape
    value_dim = v.size(-1)
    if seq_len is None:
        seq_len = cache_tokens
    if isinstance(seq_len, Tensor):
        if seq_len.numel() != 1:
            raise ValueError("dynamic decode attention sequence length must be a scalar tensor")
        if seq_len.device != k.device:
            raise ValueError("dynamic decode attention sequence length must be on the cache device")
        block_s = triton.next_power_of_2(cache_tokens)
    else:
        if seq_len < 1 or seq_len > cache_tokens:
            raise ValueError("decode attention sequence length is outside the cache shape")
        block_s = triton.next_power_of_2(seq_len)
    block_d = triton.next_power_of_2(head_dim)
    block_v = triton.next_power_of_2(value_dim)
    if block_s > 2048:
        raise ValueError("Triton dense decode attention supports sequence lengths up to 2048")
    if block_d > 256 or block_v > 256:
        raise ValueError("Triton dense decode attention supports head dimensions up to 256")
    out = torch.empty((batch, q_heads, 1, value_dim), device=q.device, dtype=q.dtype)
    if isinstance(seq_len, Tensor):
        _dense_gqa_decode_attention_dynamic_kernel[(batch, q_heads)](
            q,
            k,
            v,
            out,
            seq_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            1.0 / (head_dim**0.5),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            block_s,
            block_d,
            block_v,
            num_warps=4,
        )
        return out
    _dense_gqa_decode_attention_kernel[(batch, q_heads)](
        q,
        k,
        v,
        out,
        seq_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        1.0 / (head_dim**0.5),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_s,
        block_d,
        block_v,
        num_warps=4,
    )
    return out


@triton.jit
def _grouped_gqa_decode_attention_dynamic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len_ptr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_batch: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_q: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    seq_len = tl.load(seq_len_ptr)
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_q = tl.arange(0, block_q)
    offs_s = tl.arange(0, block_s)
    offs_d = tl.arange(0, block_d)

    q_head = kv_head * group_size + offs_q
    q_offsets = (
        batch * q_stride_batch
        + q_head[:, None] * q_stride_head
        + offs_d[None, :] * q_stride_dim
    )
    k_offsets = (
        batch * k_stride_batch
        + kv_head * k_stride_head
        + offs_d[:, None] * k_stride_dim
        + offs_s[None, :] * k_stride_token
    )
    q = tl.load(q_ptr + q_offsets, mask=(offs_q[:, None] < group_size) & (offs_d[None, :] < head_dim), other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=(offs_d[:, None] < head_dim) & (offs_s[None, :] < seq_len), other=0.0)
    scores = tl.dot(q, k) * scale
    scores = tl.where((offs_q[:, None] < group_size) & (offs_s[None, :] < seq_len), scores, -float("inf"))
    scores = scores - tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=1)[:, None]

    offs_v = tl.arange(0, block_v)
    v_offsets = (
        batch * v_stride_batch
        + kv_head * v_stride_head
        + offs_s[:, None] * v_stride_token
        + offs_v[None, :] * v_stride_dim
    )
    values = tl.load(v_ptr + v_offsets, mask=(offs_s[:, None] < seq_len) & (offs_v[None, :] < value_dim), other=0.0)
    out = tl.dot(probs.to(values.dtype), values)
    out_offsets = (
        batch * out_stride_batch
        + q_head[:, None] * out_stride_head
        + offs_v[None, :] * out_stride_dim
    )
    tl.store(out_ptr + out_offsets, out, mask=(offs_q[:, None] < group_size) & (offs_v[None, :] < value_dim))


@triton.jit
def _grouped_gqa_decode_attention_streaming_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len_ptr,
    cache_tokens: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_batch: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_batch: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_q: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    seq_len = tl.load(seq_len_ptr)
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_q = tl.arange(0, block_q)
    offs_s = tl.arange(0, block_s)
    offs_d = tl.arange(0, block_d)
    offs_v = tl.arange(0, block_v)
    q_head = kv_head * group_size + offs_q
    q_offsets = (
        batch * q_stride_batch
        + q_head[:, None] * q_stride_head
        + offs_d[None, :] * q_stride_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=(offs_q[:, None] < group_size) & (offs_d[None, :] < head_dim), other=0.0)
    running_max = tl.full((block_q,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, block_v), dtype=tl.float32)
    for start in range(0, cache_tokens, block_s):
        seq_offsets = start + offs_s
        k_offsets = (
            batch * k_stride_batch
            + kv_head * k_stride_head
            + offs_d[:, None] * k_stride_dim
            + seq_offsets[None, :] * k_stride_token
        )
        keys = tl.load(
            k_ptr + k_offsets,
            mask=(offs_d[:, None] < head_dim) & (seq_offsets[None, :] < seq_len),
            other=0.0,
        )
        scores = tl.dot(q, keys) * scale
        scores = tl.where((offs_q[:, None] < group_size) & (seq_offsets[None, :] < seq_len), scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        probs = tl.exp(scores - next_max[:, None])
        scale_old = tl.exp(running_max - next_max)
        v_offsets = (
            batch * v_stride_batch
            + kv_head * v_stride_head
            + seq_offsets[:, None] * v_stride_token
            + offs_v[None, :] * v_stride_dim
        )
        values = tl.load(
            v_ptr + v_offsets,
            mask=(seq_offsets[:, None] < seq_len) & (offs_v[None, :] < value_dim),
            other=0.0,
        )
        acc = acc * scale_old[:, None] + tl.dot(probs.to(values.dtype), values)
        running_sum = running_sum * scale_old + tl.sum(probs, axis=1)
        running_max = next_max
    out = acc / running_sum[:, None]
    out_offsets = (
        batch * out_stride_batch
        + q_head[:, None] * out_stride_head
        + offs_v[None, :] * out_stride_dim
    )
    tl.store(out_ptr + out_offsets, out, mask=(offs_q[:, None] < group_size) & (offs_v[None, :] < value_dim))


def triton_grouped_gqa_decode_attention(q: Tensor, k: Tensor, v: Tensor, seq_len: Tensor) -> Tensor:
    """Single-token GQA decode attention that shares K/V loads across a query-head group."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, dim]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v batch dimensions must match")
    if q.size(-2) != 1:
        raise ValueError("decode attention expects q to contain exactly one token")
    if k.size(-2) != v.size(-2):
        raise ValueError("k and v sequence lengths must match")
    if k.size(1) != v.size(1):
        raise ValueError("k and v head counts must match")
    if q.size(1) % k.size(1) != 0:
        raise ValueError("q head count must be divisible by kv head count")
    if q.size(-1) != k.size(-1):
        raise ValueError("q and k head dimensions must match")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must have contiguous head dimensions")
    if seq_len.numel() != 1:
        raise ValueError("dynamic decode attention sequence length must be a scalar tensor")
    if seq_len.device != k.device:
        raise ValueError("dynamic decode attention sequence length must be on the cache device")
    batch, q_heads, _, head_dim = q.shape
    _, kv_heads, cache_tokens, _ = k.shape
    value_dim = v.size(-1)
    group_size = q_heads // kv_heads
    block_q = triton.next_power_of_2(group_size)
    block_s = triton.next_power_of_2(cache_tokens)
    block_d = triton.next_power_of_2(head_dim)
    block_v = triton.next_power_of_2(value_dim)
    if block_q > 16:
        raise ValueError("grouped GQA decode attention supports up to 16 query heads per KV head")
    if block_s > 2048:
        raise ValueError("grouped GQA decode attention supports sequence lengths up to 2048")
    if block_d > 256 or block_v > 256:
        raise ValueError("grouped GQA decode attention supports head dimensions up to 256")
    out = torch.empty((batch, q_heads, 1, value_dim), device=q.device, dtype=q.dtype)
    if os.environ.get("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION", "1") != "0":
        block_s = int(os.environ.get("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S", "64"))
        if block_s <= 0 or block_s > 2048 or block_s & (block_s - 1) != 0:
            raise ValueError("streaming decode attention block size must be a power of two")
        _grouped_gqa_decode_attention_streaming_kernel[(batch, kv_heads)](
            q,
            k,
            v,
            out,
            seq_len,
            cache_tokens,
            group_size,
            head_dim,
            value_dim,
            1.0 / (head_dim**0.5),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            block_q,
            block_s,
            block_d,
            block_v,
            num_warps=4,
        )
        return out
    num_warps = int(os.environ.get("TORCHINFERNO_TRITON_GROUPED_DECODE_ATTENTION_WARPS", "4"))
    _grouped_gqa_decode_attention_dynamic_kernel[(batch, kv_heads)](
        q,
        k,
        v,
        out,
        seq_len,
        q_heads,
        kv_heads,
        group_size,
        head_dim,
        value_dim,
        1.0 / (head_dim**0.5),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_q,
        block_s,
        block_d,
        block_v,
        num_warps=num_warps,
    )
    return out


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
def _add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    hidden_out_ptr,
    norm_out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_offset = row * n_cols + offsets
    hidden = (
        tl.load(x_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
        + tl.load(residual_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    normed = hidden * tl.rsqrt(variance + eps) * weight
    tl.store(hidden_out_ptr + row_offset, hidden, mask=mask)
    tl.store(norm_out_ptr + row_offset, normed, mask=mask)


def triton_add_rms_norm(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    if x.shape != residual.shape:
        raise ValueError("x and residual tensors must have the same shape")
    if x.size(-1) != weight.numel():
        raise ValueError("RMSNorm weight size must match x.size(-1)")
    x_2d = x.contiguous().view(-1, x.size(-1))
    residual_2d = residual.contiguous().view(-1, residual.size(-1))
    weight = weight.contiguous()
    hidden_out = torch.empty_like(x_2d)
    norm_out = torch.empty_like(x_2d)
    n_cols = x_2d.size(1)
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 8192:
        raise ValueError("Triton add+RMSNorm supports hidden sizes up to 8192")
    _add_rms_norm_kernel[(x_2d.size(0),)](
        x_2d,
        residual_2d,
        weight,
        hidden_out,
        norm_out,
        n_cols,
        eps,
        block_size,
        num_warps=8,
    )
    return hidden_out.view_as(x), norm_out.view_as(x)


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
