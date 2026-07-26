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
    out_stride_row: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < cols
    gate = tl.load(gate_ptr + row * gate_stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + row * up_stride_row + offsets, mask=mask, other=0.0)
    out = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + row * out_stride_row + offsets, out, mask=mask)


def triton_swiglu_activation(gate: Tensor, up: Tensor, *, out: Tensor | None = None) -> Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    if out is not None and out.shape != gate.shape:
        raise ValueError("out tensor must have the same shape as gate and up")
    if out is not None and (out.device != gate.device or out.dtype != gate.dtype):
        raise ValueError("out tensor must have the same device and dtype as gate")
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("gate and up tensors must have contiguous last dimensions")
    if out is not None and out.stride(-1) != 1:
        raise ValueError("out tensor must have a contiguous last dimension")
    if gate.ndim == 1:
        gate_2d = gate[None, :]
        up_2d = up[None, :]
        out_2d = out[None, :] if out is not None else None
    else:
        gate_2d = gate.flatten(0, -2)
        up_2d = up.flatten(0, -2)
        out_2d = out.flatten(0, -2) if out is not None else None
    out_2d = torch.empty_like(gate_2d) if out_2d is None else out_2d
    cols = gate_2d.size(1)
    block_size = triton.next_power_of_2(cols)
    _swiglu_kernel[(gate_2d.size(0),)](
        gate_2d,
        up_2d,
        out_2d,
        cols,
        gate_2d.stride(0),
        up_2d.stride(0),
        out_2d.stride(0),
        block_size,
        num_warps=8,
    )
    return out if out is not None else out_2d.view_as(gate)


@triton.jit
def _copy_ragged_prefix_kv_kernel(
    keys_ptr,
    values_ptr,
    lengths_ptr,
    dest_rows_ptr,
    source_rows_ptr,
    key_stride_row,
    key_stride_head,
    key_stride_token,
    value_stride_row,
    value_stride_head,
    value_stride_token,
    total_elements,
    kv_heads: tl.constexpr,
    prefix_capacity: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    in_bounds = offsets < total_elements
    dim = offsets % head_dim
    logical = offsets // head_dim
    token = logical % prefix_capacity
    logical //= prefix_capacity
    head = logical % kv_heads
    batch = logical // kv_heads
    length = tl.load(lengths_ptr + batch, mask=in_bounds, other=0)
    mask = in_bounds & (token < length)
    dest_row = tl.load(dest_rows_ptr + batch, mask=mask, other=0)
    source_row = tl.load(source_rows_ptr + batch, mask=mask, other=0)
    key_source = (
        source_row * key_stride_row
        + head * key_stride_head
        + token * key_stride_token
        + dim
    )
    key_dest = (
        dest_row * key_stride_row
        + head * key_stride_head
        + token * key_stride_token
        + dim
    )
    value_source = (
        source_row * value_stride_row
        + head * value_stride_head
        + token * value_stride_token
        + dim
    )
    value_dest = (
        dest_row * value_stride_row
        + head * value_stride_head
        + token * value_stride_token
        + dim
    )
    tl.store(keys_ptr + key_dest, tl.load(keys_ptr + key_source, mask=mask), mask=mask)
    tl.store(
        values_ptr + value_dest,
        tl.load(values_ptr + value_source, mask=mask),
        mask=mask,
    )


def triton_copy_ragged_prefix_kv(
    keys: Tensor,
    values: Tensor,
    lengths: Tensor,
    dest_rows: Tensor,
    source_rows: Tensor,
    *,
    prefix_capacity: int,
) -> None:
    """Copy per-row dense KV prefixes without materializing indexed tensors."""

    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("keys and values must have matching [rows, heads, tokens, dim] shapes")
    if not keys.is_cuda or not values.is_cuda:
        raise ValueError("ragged prefix KV copy requires CUDA tensors")
    if keys.device != values.device or keys.dtype != values.dtype:
        raise ValueError("keys and values must have matching devices and dtypes")
    batch = int(lengths.numel())
    if lengths.ndim != 1 or dest_rows.shape != (batch,) or source_rows.shape != (batch,):
        raise ValueError("lengths, dest_rows, and source_rows must be equal-length vectors")
    if any(tensor.device != keys.device for tensor in (lengths, dest_rows, source_rows)):
        raise ValueError("prefix metadata must be on the KV cache device")
    capacity = int(prefix_capacity)
    if capacity < 0 or capacity > keys.size(2):
        raise ValueError("prefix capacity exceeds the KV cache")
    if capacity == 0 or batch == 0:
        return
    if keys.stride(-1) != 1 or values.stride(-1) != 1:
        raise ValueError("KV cache head dimensions must be contiguous")
    total = batch * keys.size(1) * capacity * keys.size(3)
    block_size = 256
    _copy_ragged_prefix_kv_kernel[(triton.cdiv(total, block_size),)](
        keys,
        values,
        lengths,
        dest_rows,
        source_rows,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        total,
        keys.size(1),
        capacity,
        keys.size(3),
        block_size,
        num_warps=4,
    )


@triton.jit
def _copy_ragged_prefix_kv_layers_kernel(
    keys_ptr,
    values_ptr,
    lengths_ptr,
    dest_rows_ptr,
    source_rows_ptr,
    key_stride_layer,
    key_stride_row,
    key_stride_head,
    key_stride_token,
    value_stride_layer,
    value_stride_row,
    value_stride_head,
    value_stride_token,
    total_elements,
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    prefix_capacity: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    in_bounds = offsets < total_elements
    dim = offsets % head_dim
    logical = offsets // head_dim
    token = logical % prefix_capacity
    logical //= prefix_capacity
    head = logical % kv_heads
    logical //= kv_heads
    row = logical % batch
    layer = logical // batch
    length = tl.load(lengths_ptr + row, mask=in_bounds, other=0)
    mask = in_bounds & (token < length)
    dest_row = tl.load(dest_rows_ptr + row, mask=mask, other=0)
    source_row = tl.load(source_rows_ptr + row, mask=mask, other=0)
    key_base = layer * key_stride_layer + head * key_stride_head + token * key_stride_token + dim
    value_base = (
        layer * value_stride_layer + head * value_stride_head + token * value_stride_token + dim
    )
    key_source = key_base + source_row * key_stride_row
    key_dest = key_base + dest_row * key_stride_row
    value_source = value_base + source_row * value_stride_row
    value_dest = value_base + dest_row * value_stride_row
    tl.store(keys_ptr + key_dest, tl.load(keys_ptr + key_source, mask=mask), mask=mask)
    tl.store(
        values_ptr + value_dest,
        tl.load(values_ptr + value_source, mask=mask),
        mask=mask,
    )


def triton_copy_ragged_prefix_kv_layers(
    keys: Tensor,
    values: Tensor,
    lengths: Tensor,
    dest_rows: Tensor,
    source_rows: Tensor,
    *,
    prefix_capacity: int,
) -> None:
    """Copy per-row dense KV prefixes across all layers in one launch."""

    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError(
            "keys and values must have matching [layers, rows, heads, tokens, dim] shapes"
        )
    if not keys.is_cuda or not values.is_cuda:
        raise ValueError("layered ragged prefix KV copy requires CUDA tensors")
    if keys.device != values.device or keys.dtype != values.dtype:
        raise ValueError("keys and values must have matching devices and dtypes")
    batch = int(lengths.numel())
    if lengths.ndim != 1 or dest_rows.shape != (batch,) or source_rows.shape != (batch,):
        raise ValueError("lengths, dest_rows, and source_rows must be equal-length vectors")
    if any(tensor.device != keys.device for tensor in (lengths, dest_rows, source_rows)):
        raise ValueError("prefix metadata must be on the KV cache device")
    capacity = int(prefix_capacity)
    if capacity < 0 or capacity > keys.size(3):
        raise ValueError("prefix capacity exceeds the KV cache")
    if capacity == 0 or batch == 0 or keys.size(0) == 0:
        return
    if keys.stride(-1) != 1 or values.stride(-1) != 1:
        raise ValueError("KV cache head dimensions must be contiguous")
    total = keys.size(0) * batch * keys.size(2) * capacity * keys.size(4)
    block_size = 256
    _copy_ragged_prefix_kv_layers_kernel[(triton.cdiv(total, block_size),)](
        keys,
        values,
        lengths,
        dest_rows,
        source_rows,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        total,
        batch,
        keys.size(2),
        capacity,
        keys.size(4),
        block_size,
        num_warps=4,
    )


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
def _rotary_llama_qk_batched_inplace_kernel(
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
    cos_stride_batch: tl.constexpr,
    cos_stride_token: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    # Fused Llama rotate-half RoPE for q/k where the rotary tables vary PER
    # (batch_row, token) -- the decode/ragged-suffix case, unlike the prefill
    # kernel above whose positions are shared across the batch. cos/sin are
    # indexed [batch, token, dim]; q/k broadcast it over their head dim. Math is
    # identical to _rotate_llama_eager (q_first*cos - q_second*sin in the low
    # half, q_second*cos + q_first*sin in the high half).
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
    q_cos_offset = q_batch * cos_stride_batch + q_token * cos_stride_token + rotary_dim
    q_mask = mask & (offsets < total_q)
    q_first = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + q_offset + half_dim * q_stride_dim, mask=q_mask, other=0.0).to(tl.float32)
    q_cos = tl.load(cos_ptr + q_cos_offset, mask=q_mask, other=1.0).to(tl.float32)
    q_sin = tl.load(sin_ptr + q_cos_offset, mask=q_mask, other=0.0).to(tl.float32)
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
    k_cos_offset = k_batch * cos_stride_batch + k_token * cos_stride_token + rotary_dim
    k_mask = mask & (offsets < total_k)
    k_first = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_offset + half_dim * k_stride_dim, mask=k_mask, other=0.0).to(tl.float32)
    k_cos = tl.load(cos_ptr + k_cos_offset, mask=k_mask, other=1.0).to(tl.float32)
    k_sin = tl.load(sin_ptr + k_cos_offset, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(k_ptr + k_offset, k_first * k_cos - k_second * k_sin, mask=k_mask)
    tl.store(k_ptr + k_offset + half_dim * k_stride_dim, k_second * k_cos + k_first * k_sin, mask=k_mask)


def triton_apply_rotary_llama_batched_inplace(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-(batch,token) Llama rotate-half RoPE on q/k views, in place.

    q/k: [batch, heads, tokens, head_dim] (any strides, contiguous head dim).
    cos/sin: [batch, tokens, head_dim] or [batch, tokens, head_dim/2]. This is the
    decode / ragged-suffix analogue of triton_apply_rotary_llama_inplace (whose
    rotary tables are shared across the batch). Fuses the rotate-half cat/neg/mul
    that _rotate_llama_eager otherwise spends ~2.6ms on across the decode step.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, tokens, head_dim]")
    if q.size(0) != k.size(0) or q.size(-1) != k.size(-1) or q.size(-2) != k.size(-2):
        raise ValueError("q and k must share batch, token, and head dimensions")
    if q.size(-1) % 2 != 0:
        raise ValueError("head dimension must be even")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must have contiguous head dimensions")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    batch, q_heads, tokens, head_dim = q.shape
    half_dim = head_dim // 2
    if cos.shape not in {(batch, tokens, head_dim), (batch, tokens, half_dim)}:
        raise ValueError("rotary cache must be [batch, tokens, head_dim] or [..., head_dim/2]")
    cos = cos.contiguous()
    sin = sin.contiguous()
    k_heads = k.size(1)
    total_q = batch * q_heads * tokens * half_dim
    total_k = batch * k_heads * tokens * half_dim
    total_elements = max(total_q, total_k)
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _rotary_llama_qk_batched_inplace_kernel[grid](
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
        cos.stride(0),
        cos.stride(1),
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
def _rotary_llama_append_kv_ragged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_key_ptr,
    cache_value_ptr,
    positions_ptr,
    row_indices_ptr,
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
    has_row_indices: tl.constexpr,
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
    q_cos = tl.load(cos_ptr + q_batch * cache_dim + q_dim, mask=q_mask, other=1.0).to(tl.float32)
    q_sin = tl.load(sin_ptr + q_batch * cache_dim + q_dim, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + q_offset, q_first * q_cos - q_second * q_sin, mask=q_mask)
    tl.store(q_ptr + q_offset + half_dim * q_stride_dim, q_second * q_cos + q_first * q_sin, mask=q_mask)

    k_dim = offsets % half_dim
    k_head_batch = offsets // half_dim
    k_head = k_head_batch % kv_heads
    k_batch = k_head_batch // kv_heads
    k_mask = mask & (offsets < k_pairs)
    k_offset = k_batch * k_stride_batch + k_head * k_stride_head + k_dim * k_stride_dim
    k_row = k_batch
    if has_row_indices:
        k_row = tl.load(row_indices_ptr + k_batch, mask=k_mask, other=0)
    k_pos = tl.load(positions_ptr + k_batch, mask=k_mask, other=0)
    cache_k_offset = (
        k_row * cache_stride_batch
        + k_head * cache_stride_head
        + k_pos * cache_stride_token
        + k_dim * cache_stride_dim
    )
    k_first = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_offset + half_dim * k_stride_dim, mask=k_mask, other=0.0).to(tl.float32)
    k_cos = tl.load(cos_ptr + k_batch * cache_dim + k_dim, mask=k_mask, other=1.0).to(tl.float32)
    k_sin = tl.load(sin_ptr + k_batch * cache_dim + k_dim, mask=k_mask, other=0.0).to(tl.float32)
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
    v_row = v_batch
    if has_row_indices:
        v_row = tl.load(row_indices_ptr + v_batch, mask=v_mask, other=0)
    v_pos = tl.load(positions_ptr + v_batch, mask=v_mask, other=0)
    cache_v_offset = (
        v_row * cache_stride_batch
        + v_head * cache_stride_head
        + v_pos * cache_stride_token
        + v_dim * cache_stride_dim
    )
    tl.store(cache_value_ptr + cache_v_offset, tl.load(v_ptr + v_offset, mask=v_mask), mask=v_mask)


def triton_apply_rotary_append_kv_ragged_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    row_indices: Tensor | None = None,
) -> Tensor:
    """Apply per-row Llama RoPE to decode q and append rotated k/v to dense KV cache."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, head_dim]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v batch dimensions must match")
    if q.size(-2) != 1 or k.size(-2) != 1 or v.size(-2) != 1:
        raise ValueError("ragged decode rotary append expects exactly one token")
    if q.size(-1) != k.size(-1) or k.size(-1) != v.size(-1):
        raise ValueError("q, k, and v head dimensions must match")
    if q.size(-1) % 2 != 0:
        raise ValueError("head dimension must be even")
    if k.size(1) != v.size(1):
        raise ValueError("k and v head counts must match")
    if cache_keys.shape != cache_values.shape:
        raise ValueError("cache keys and values must have the same shape")
    if k.size(1) != cache_keys.size(1) or k.size(-1) != cache_keys.size(-1):
        raise ValueError("cache shape is incompatible with incoming k/v")
    batch, q_heads, _, head_dim = q.shape
    if positions.shape != (batch,):
        raise ValueError("ragged decode positions must have shape [batch]")
    if positions.device != cache_keys.device:
        raise ValueError("ragged decode positions must be on the cache device")
    if row_indices is not None:
        if row_indices.shape != (batch,):
            raise ValueError("ragged decode row indices must have shape [batch]")
        if row_indices.device != cache_keys.device:
            raise ValueError("ragged decode row indices must be on the cache device")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    half_dim = q.size(-1) // 2
    if cos.shape not in {(batch, half_dim), (batch, head_dim)}:
        raise ValueError("ragged decode rotary cache shape must be [batch, head_dim / 2]")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must have contiguous head dimensions")
    if cache_keys.stride(-1) != 1 or cache_values.stride(-1) != 1:
        raise ValueError("cache keys and values must have contiguous head dimensions")
    cos = cos.contiguous()
    sin = sin.contiguous()
    positions = positions.contiguous()
    if row_indices is not None:
        row_indices = row_indices.contiguous()
    kv_heads = k.size(1)
    q_pairs = batch * q_heads * half_dim
    k_pairs = batch * kv_heads * half_dim
    v_elements = batch * kv_heads * head_dim
    total_elements = max(q_pairs, k_pairs, v_elements)
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _rotary_llama_append_kv_ragged_decode_kernel[grid](
        q,
        k,
        v,
        cache_keys,
        cache_values,
        positions,
        row_indices,
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
        cos.size(-1),
        row_indices is not None,
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
def _rotary_llama_append_kv_packed_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_key_ptr,
    cache_value_ptr,
    positions_ptr,
    rows_ptr,
    cos_ptr,
    sin_ptr,
    total_elements,
    q_pairs,
    k_pairs,
    v_elements,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    tokens: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    cache_dim: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    cache_stride_batch: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements

    q_dim = offsets % half_dim
    q_logical = offsets // half_dim
    q_token = q_logical % tokens
    q_head = (q_logical // tokens) % q_heads
    q_mask = mask & (offsets < q_pairs)
    q_offset = q_head * q_stride_head + q_token * q_stride_token + q_dim * q_stride_dim
    q_first = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0).to(tl.float32)
    q_second = tl.load(
        q_ptr + q_offset + half_dim * q_stride_dim,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    q_cos = tl.load(cos_ptr + q_token * cache_dim + q_dim, mask=q_mask, other=1.0).to(tl.float32)
    q_sin = tl.load(sin_ptr + q_token * cache_dim + q_dim, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + q_offset, q_first * q_cos - q_second * q_sin, mask=q_mask)
    tl.store(
        q_ptr + q_offset + half_dim * q_stride_dim,
        q_second * q_cos + q_first * q_sin,
        mask=q_mask,
    )

    k_dim = offsets % half_dim
    k_logical = offsets // half_dim
    k_token = k_logical % tokens
    k_head = (k_logical // tokens) % kv_heads
    k_mask = mask & (offsets < k_pairs)
    k_offset = k_head * k_stride_head + k_token * k_stride_token + k_dim * k_stride_dim
    k_row = tl.load(rows_ptr + k_token, mask=k_mask, other=0)
    k_pos = tl.load(positions_ptr + k_token, mask=k_mask, other=0)
    cache_k_offset = (
        k_row * cache_stride_batch
        + k_head * cache_stride_head
        + k_pos * cache_stride_token
        + k_dim * cache_stride_dim
    )
    k_first = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(
        k_ptr + k_offset + half_dim * k_stride_dim,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    k_cos = tl.load(cos_ptr + k_token * cache_dim + k_dim, mask=k_mask, other=1.0).to(tl.float32)
    k_sin = tl.load(sin_ptr + k_token * cache_dim + k_dim, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(cache_key_ptr + cache_k_offset, k_first * k_cos - k_second * k_sin, mask=k_mask)
    tl.store(
        cache_key_ptr + cache_k_offset + half_dim * cache_stride_dim,
        k_second * k_cos + k_first * k_sin,
        mask=k_mask,
    )

    v_dim = offsets % head_dim
    v_logical = offsets // head_dim
    v_token = v_logical % tokens
    v_head = (v_logical // tokens) % kv_heads
    v_mask = mask & (offsets < v_elements)
    v_offset = v_head * v_stride_head + v_token * v_stride_token + v_dim * v_stride_dim
    v_row = tl.load(rows_ptr + v_token, mask=v_mask, other=0)
    v_pos = tl.load(positions_ptr + v_token, mask=v_mask, other=0)
    cache_v_offset = (
        v_row * cache_stride_batch
        + v_head * cache_stride_head
        + v_pos * cache_stride_token
        + v_dim * cache_stride_dim
    )
    tl.store(cache_value_ptr + cache_v_offset, tl.load(v_ptr + v_offset, mask=v_mask), mask=v_mask)


def triton_apply_rotary_append_kv_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    positions: Tensor,
    rows: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply Llama RoPE and append a packed token stream to arbitrary cache rows."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.size(0) != 1:
        raise ValueError("packed q, k, and v must have shape [1, heads, tokens, dim]")
    if k.size(0) != 1 or v.size(0) != 1 or k.size(1) != v.size(1):
        raise ValueError("packed k and v shapes must match")
    if q.size(-2) != k.size(-2) or k.size(-2) != v.size(-2):
        raise ValueError("packed q, k, and v token counts must match")
    if q.size(-1) != k.size(-1) or k.size(-1) != v.size(-1) or q.size(-1) % 2:
        raise ValueError("packed q, k, and v require matching even head dimensions")
    if cache_keys.shape != cache_values.shape:
        raise ValueError("cache keys and values must have matching shapes")
    tokens = q.size(-2)
    if positions.shape != (tokens,) or rows.shape != (tokens,):
        raise ValueError("packed positions and rows must have shape [tokens]")
    if k.size(1) != cache_keys.size(1) or k.size(-1) != cache_keys.size(-1):
        raise ValueError("cache shape is incompatible with packed k/v")
    if any(tensor.device != cache_keys.device for tensor in (q, k, v, positions, rows, cos, sin)):
        raise ValueError("packed rotary append tensors must share one device")
    half_dim = q.size(-1) // 2
    if cos.shape != sin.shape or cos.numel() != tokens * half_dim:
        raise ValueError("packed rotary cache must have shape [tokens, head_dim / 2]")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v, cache_keys, cache_values)):
        raise ValueError("packed rotary append requires contiguous head dimensions")

    cos = cos.reshape(tokens, half_dim).contiguous()
    sin = sin.reshape(tokens, half_dim).contiguous()
    positions = positions.contiguous()
    rows = rows.contiguous()
    q_heads = q.size(1)
    kv_heads = k.size(1)
    head_dim = q.size(-1)
    q_pairs = tokens * q_heads * half_dim
    k_pairs = tokens * kv_heads * half_dim
    v_elements = tokens * kv_heads * head_dim
    total_elements = max(q_pairs, k_pairs, v_elements)
    block_size = 256
    _rotary_llama_append_kv_packed_kernel[(triton.cdiv(total_elements, block_size),)](
        q,
        k,
        v,
        cache_keys,
        cache_values,
        positions,
        rows,
        cos,
        sin,
        total_elements,
        q_pairs,
        k_pairs,
        v_elements,
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        half_dim,
        half_dim,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
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
    row_indices_ptr,
    seq_len_stride: tl.constexpr,
    has_row_indices: tl.constexpr,
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
    batch = tl.program_id(0)
    cache_batch = batch
    if has_row_indices:
        cache_batch = tl.load(row_indices_ptr + batch)
    seq_len = tl.load(seq_len_ptr + batch * seq_len_stride)
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
        cache_batch * k_stride_batch
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
        cache_batch * v_stride_batch
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
    row_indices_ptr,
    seq_len_stride: tl.constexpr,
    has_row_indices: tl.constexpr,
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
    batch = tl.program_id(0)
    cache_batch = batch
    if has_row_indices:
        cache_batch = tl.load(row_indices_ptr + batch)
    seq_len = tl.load(seq_len_ptr + batch * seq_len_stride)
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
            cache_batch * k_stride_batch
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
            cache_batch * v_stride_batch
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


def triton_grouped_gqa_decode_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_len: Tensor,
    row_indices: Tensor | None = None,
) -> Tensor:
    """Single-token GQA decode attention that shares K/V loads across a query-head group."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, dim]")
    if row_indices is None and (q.size(0) != k.size(0) or q.size(0) != v.size(0)):
        raise ValueError("q, k, and v batch dimensions must match")
    if row_indices is not None:
        if row_indices.shape != (q.size(0),):
            raise ValueError("decode attention row indices must have shape [batch]")
        if row_indices.device != k.device:
            raise ValueError("decode attention row indices must be on the cache device")
        if k.size(0) != v.size(0):
            raise ValueError("k and v batch dimensions must match")
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
    if seq_len.numel() not in {1, q.size(0)}:
        raise ValueError("dynamic decode attention sequence length must be scalar or have shape [batch]")
    if seq_len.device != k.device:
        raise ValueError("dynamic decode attention sequence length must be on the cache device")
    if row_indices is not None:
        row_indices = row_indices.contiguous()
    row_indices_arg = row_indices if row_indices is not None else seq_len
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
    seq_len_stride = 0 if seq_len.numel() == 1 else 1
    out = torch.empty((batch, q_heads, 1, value_dim), device=q.device, dtype=q.dtype)
    if os.environ.get("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION", "1") != "0":
        block_s = _streaming_decode_attention_block_s(batch)
        if block_s <= 0 or block_s > 2048 or block_s & (block_s - 1) != 0:
            raise ValueError("streaming decode attention block size must be a power of two")
        _grouped_gqa_decode_attention_streaming_kernel[(batch, kv_heads)](
            q,
            k,
            v,
            out,
            seq_len,
            row_indices_arg,
            seq_len_stride,
            row_indices is not None,
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
            num_warps=_streaming_decode_attention_num_warps(),
        )
        return out
    num_warps = int(os.environ.get("TORCHINFERNO_TRITON_GROUPED_DECODE_ATTENTION_WARPS", "4"))
    _grouped_gqa_decode_attention_dynamic_kernel[(batch, kv_heads)](
        q,
        k,
        v,
        out,
        seq_len,
        row_indices_arg,
        seq_len_stride,
        row_indices is not None,
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


def _streaming_decode_attention_block_s(batch: int) -> int:
    del batch
    return int(os.environ.get("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S", "64"))


def _streaming_decode_attention_num_warps() -> int:
    return int(os.environ.get("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_WARPS", "4"))


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
def _rms_norm_fp8_per_token_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.rsqrt(variance + eps) * weight
    scale = tl.maximum(tl.max(tl.abs(normed), axis=0) / 448.0, 1e-10)
    tl.store(output_ptr + row_offsets, normed / scale, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _add_rms_norm_fp8_per_token_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    hidden_out_ptr,
    output_ptr,
    scale_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets
    hidden = (
        tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    normed = hidden * tl.rsqrt(variance + eps) * weight
    scale = tl.maximum(tl.max(tl.abs(normed), axis=0) / 448.0, 1e-10)
    tl.store(hidden_out_ptr + row_offsets, hidden, mask=mask)
    tl.store(output_ptr + row_offsets, normed / scale, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _swiglu_fp8_per_token_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    scale_ptr,
    gate_stride_row: tl.constexpr,
    up_stride_row: tl.constexpr,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    output_offsets = row * n_cols + offsets
    gate = tl.load(
        gate_ptr + row * gate_stride_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + row * up_stride_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # Match the unfused path's BF16 activation materialization before dynamic
    # FP8 quantization. Keeping the FP32 intermediate changes decode decisions
    # enough to fail the held-out serving-quality gate.
    activated = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(activated), axis=0) / 448.0, 1e-10)
    tl.store(output_ptr + output_offsets, activated / scale, mask=mask)
    tl.store(scale_ptr + row, scale)


def _fp8_per_token_outputs(x: Tensor) -> tuple[Tensor, Tensor, int, int]:
    if not x.is_cuda or x.ndim < 2:
        raise ValueError("per-token FP8 fusion requires a CUDA activation matrix")
    n_cols = x.size(-1)
    rows = x.numel() // n_cols
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 8192:
        raise ValueError("per-token FP8 fusion supports hidden sizes up to 8192")
    output = torch.empty((rows, n_cols), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    return output, scale, n_cols, block_size


def triton_rms_norm_fp8_per_token(
    x: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """RMS-normalize and dynamically quantize each row to e4m3 in one pass."""

    if x.size(-1) != weight.numel() or weight.device != x.device:
        raise ValueError("RMSNorm weight must match the activation width and device")
    output, scale, n_cols, block_size = _fp8_per_token_outputs(x)
    x_2d = x.contiguous().view(-1, n_cols)
    _rms_norm_fp8_per_token_kernel[(x_2d.size(0),)](
        x_2d,
        weight.contiguous(),
        output,
        scale,
        n_cols,
        eps,
        block_size,
        num_warps=8,
        num_stages=1,
    )
    return output.view_as(x), scale


def triton_add_rms_norm_fp8_per_token(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Add residual, RMS-normalize, and quantize the normalized rows in one pass."""

    if x.shape != residual.shape or x.device != residual.device:
        raise ValueError("x and residual tensors must have matching shapes and devices")
    if x.size(-1) != weight.numel() or weight.device != x.device:
        raise ValueError("RMSNorm weight must match the activation width and device")
    output, scale, n_cols, block_size = _fp8_per_token_outputs(x)
    x_2d = x.contiguous().view(-1, n_cols)
    residual_2d = residual.contiguous().view(-1, n_cols)
    hidden = torch.empty_like(x_2d)
    _add_rms_norm_fp8_per_token_kernel[(x_2d.size(0),)](
        x_2d,
        residual_2d,
        weight.contiguous(),
        hidden,
        output,
        scale,
        n_cols,
        eps,
        block_size,
        num_warps=8,
        num_stages=1,
    )
    return hidden.view_as(x), output.view_as(x), scale


def triton_swiglu_fp8_per_token(gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    """Apply SwiGLU and dynamically quantize each output row to e4m3 in one pass."""

    if gate.shape != up.shape or gate.device != up.device:
        raise ValueError("gate and up tensors must have matching shapes and devices")
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("gate and up tensors must have contiguous last dimensions")
    output, scale, n_cols, block_size = _fp8_per_token_outputs(gate)
    gate_2d = gate.view(-1, n_cols)
    up_2d = up.view(-1, n_cols)
    _swiglu_fp8_per_token_kernel[(gate_2d.size(0),)](
        gate_2d,
        up_2d,
        output,
        scale,
        gate_2d.stride(0),
        up_2d.stride(0),
        n_cols,
        block_size,
        num_warps=8,
        num_stages=1,
    )
    return output.view_as(gate), scale


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


@triton.jit
def _batched_paged_gqa_decode_attention_streaming_kernel(
    q_ptr,
    key_pages_ptr,
    value_pages_ptr,
    page_table_ptr,
    seq_lens_ptr,
    out_ptr,
    cache_tokens: tl.constexpr,
    page_table_stride_batch: tl.constexpr,
    page_size: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_dim: tl.constexpr,
    key_stride_page: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_page: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_q: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch)
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
        page_slots = seq_offsets // page_size
        page_offsets = seq_offsets - page_slots * page_size
        page_ids = tl.load(
            page_table_ptr + batch * page_table_stride_batch + page_slots,
            mask=seq_offsets < seq_len,
            other=0,
        )
        key_offsets = (
            page_ids[None, :] * key_stride_page
            + kv_head * key_stride_head
            + page_offsets[None, :] * key_stride_token
            + offs_d[:, None] * key_stride_dim
        )
        keys = tl.load(
            key_pages_ptr + key_offsets,
            mask=(offs_d[:, None] < head_dim) & (seq_offsets[None, :] < seq_len),
            other=0.0,
        )
        scores = tl.dot(q, keys) * scale
        scores = tl.where((offs_q[:, None] < group_size) & (seq_offsets[None, :] < seq_len), scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        probs = tl.exp(scores - next_max[:, None])
        scale_old = tl.exp(running_max - next_max)
        value_offsets = (
            page_ids[:, None] * value_stride_page
            + kv_head * value_stride_head
            + page_offsets[:, None] * value_stride_token
            + offs_v[None, :] * value_stride_dim
        )
        values = tl.load(
            value_pages_ptr + value_offsets,
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


def triton_batched_paged_gqa_decode_attention(
    query: Tensor,
    cache,
    request_ids: tuple[str, ...] | list[str],
    positions: Tensor,
) -> Tensor:
    """Batched single-token paged GQA decode attention over a request page table."""

    if query.ndim == 3:
        query = query[:, :, None, :]
    if query.ndim != 4 or query.size(2) != 1:
        raise ValueError("query must have shape [batch, heads, head_dim] or [batch, heads, 1, head_dim]")
    if len(request_ids) != query.size(0):
        raise ValueError("request_ids length must match query batch")
    if query.size(1) % cache.num_key_value_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if positions.ndim == 0:
        positions = positions.expand(len(request_ids))
    if positions.shape != (len(request_ids),):
        raise ValueError("positions must have shape [batch]")
    page_table, seq_lens, cache_tokens = _batched_paged_decode_page_table(cache, request_ids, device=query.device)
    return triton_batched_paged_gqa_decode_attention_with_table(
        query,
        cache.keys,
        cache.values,
        page_table,
        seq_lens,
        page_size=cache.page_size,
        cache_tokens=cache_tokens,
        num_key_value_heads=cache.num_key_value_heads,
        value_head_dim=cache.value_head_dim,
    )


def triton_batched_paged_gqa_decode_attention_with_table(
    query: Tensor,
    key_pages: Tensor,
    value_pages: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    *,
    page_size: int,
    cache_tokens: int,
    num_key_value_heads: int | None = None,
    value_head_dim: int | None = None,
) -> Tensor:
    """Batched single-token paged GQA decode attention over an existing device page table."""

    if query.ndim == 3:
        query = query[:, :, None, :]
    if query.ndim != 4 or query.size(2) != 1:
        raise ValueError("query must have shape [batch, heads, head_dim] or [batch, heads, 1, head_dim]")
    if page_table.ndim != 2 or page_table.size(0) != query.size(0):
        raise ValueError("page_table must have shape [batch, pages]")
    if seq_lens.shape != (query.size(0),):
        raise ValueError("seq_lens must have shape [batch]")
    if cache_tokens <= 0:
        value_dim = value_pages.size(3) if value_head_dim is None else value_head_dim
        return query.new_zeros((query.size(0), query.size(1), 1, value_dim))
    batch, q_heads, _tokens, head_dim = query.shape
    kv_heads = key_pages.size(1) if num_key_value_heads is None else num_key_value_heads
    value_dim = value_pages.size(3) if value_head_dim is None else value_head_dim
    if q_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    group_size = q_heads // kv_heads
    block_q = triton.next_power_of_2(group_size)
    block_s = _streaming_decode_attention_block_s(batch)
    block_d = triton.next_power_of_2(head_dim)
    block_v = triton.next_power_of_2(value_dim)
    if block_q > 16:
        raise ValueError("batched paged GQA decode attention supports up to 16 query heads per KV head")
    if block_s <= 0 or block_s > 2048 or block_s & (block_s - 1) != 0:
        raise ValueError("streaming decode attention block size must be a power of two up to 2048")
    if block_d > 256 or block_v > 256:
        raise ValueError("batched paged GQA decode attention supports head/value dimensions up to 256")
    if query.stride(-1) != 1:
        query = query.contiguous()
    out = torch.empty((batch, q_heads, 1, value_dim), device=query.device, dtype=query.dtype)
    _batched_paged_gqa_decode_attention_streaming_kernel[(batch, kv_heads)](
        query,
        key_pages,
        value_pages,
        page_table,
        seq_lens,
        out,
        cache_tokens,
        page_table.stride(0),
        page_size,
        group_size,
        head_dim,
        value_dim,
        1.0 / (head_dim**0.5),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key_pages.stride(0),
        key_pages.stride(1),
        key_pages.stride(2),
        key_pages.stride(3),
        value_pages.stride(0),
        value_pages.stride(1),
        value_pages.stride(2),
        value_pages.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_q,
        block_s,
        block_d,
        block_v,
        num_warps=_streaming_decode_attention_num_warps(),
    )
    return out


@triton.jit
def _paged_append_kv_cache_ragged_kernel(
    keys_ptr,
    values_ptr,
    key_pages_ptr,
    value_pages_ptr,
    page_table_ptr,
    positions_ptr,
    page_table_stride_batch: tl.constexpr,
    page_size: tl.constexpr,
    key_in_stride_batch: tl.constexpr,
    key_in_stride_head: tl.constexpr,
    key_in_stride_token: tl.constexpr,
    key_in_stride_dim: tl.constexpr,
    value_in_stride_batch: tl.constexpr,
    value_in_stride_head: tl.constexpr,
    value_in_stride_token: tl.constexpr,
    value_in_stride_dim: tl.constexpr,
    key_page_stride_page: tl.constexpr,
    key_page_stride_head: tl.constexpr,
    key_page_stride_token: tl.constexpr,
    key_page_stride_dim: tl.constexpr,
    value_page_stride_page: tl.constexpr,
    value_page_stride_head: tl.constexpr,
    value_page_stride_token: tl.constexpr,
    value_page_stride_dim: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    block_dim: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    offsets = tl.arange(0, block_dim)
    position = tl.load(positions_ptr + batch)
    page_slot = position // page_size
    page_offset = position - page_slot * page_size
    page_id = tl.load(page_table_ptr + batch * page_table_stride_batch + page_slot)
    key_values = tl.load(
        keys_ptr
        + batch * key_in_stride_batch
        + kv_head * key_in_stride_head
        + offsets * key_in_stride_dim,
        mask=offsets < head_dim,
        other=0.0,
    )
    tl.store(
        key_pages_ptr
        + page_id * key_page_stride_page
        + kv_head * key_page_stride_head
        + page_offset * key_page_stride_token
        + offsets * key_page_stride_dim,
        key_values,
        mask=offsets < head_dim,
    )
    value_values = tl.load(
        values_ptr
        + batch * value_in_stride_batch
        + kv_head * value_in_stride_head
        + offsets * value_in_stride_dim,
        mask=offsets < value_dim,
        other=0.0,
    )
    tl.store(
        value_pages_ptr
        + page_id * value_page_stride_page
        + kv_head * value_page_stride_head
        + page_offset * value_page_stride_token
        + offsets * value_page_stride_dim,
        value_values,
        mask=offsets < value_dim,
    )


def triton_paged_append_kv_cache_ragged(
    keys: Tensor,
    values: Tensor,
    key_pages: Tensor,
    value_pages: Tensor,
    page_table: Tensor,
    positions: Tensor,
    *,
    page_size: int,
) -> None:
    """Append one ragged decode token per row using an existing device page table."""

    if keys.ndim != 4 or values.ndim != 4 or keys.size(2) != 1 or values.size(2) != 1:
        raise ValueError("paged ragged append expects keys/values with shape [batch, heads, 1, dim]")
    if page_table.ndim != 2 or page_table.size(0) != keys.size(0):
        raise ValueError("page_table must have shape [batch, pages]")
    if positions.shape != (keys.size(0),):
        raise ValueError("positions must have shape [batch]")
    if keys.size(0) != values.size(0) or keys.size(1) != values.size(1):
        raise ValueError("keys and values must have matching batch/head dimensions")
    kv_heads = keys.size(1)
    head_dim = keys.size(3)
    value_dim = values.size(3)
    block_dim = triton.next_power_of_2(max(head_dim, value_dim))
    if block_dim > 256:
        raise ValueError("paged ragged append supports head/value dimensions up to 256")
    _paged_append_kv_cache_ragged_kernel[(keys.size(0), kv_heads)](
        keys,
        values,
        key_pages,
        value_pages,
        page_table,
        positions.to(device=keys.device, dtype=torch.long),
        page_table.stride(0),
        page_size,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        key_pages.stride(0),
        key_pages.stride(1),
        key_pages.stride(2),
        key_pages.stride(3),
        value_pages.stride(0),
        value_pages.stride(1),
        value_pages.stride(2),
        value_pages.stride(3),
        head_dim,
        value_dim,
        block_dim,
        num_warps=4,
    )


def _batched_paged_decode_page_table(cache, request_ids: tuple[str, ...] | list[str], *, device: torch.device):
    lengths = [cache.sequence_length(request_id) for request_id in request_ids]
    cache_tokens = max(lengths, default=0)
    max_pages = max(1, (cache_tokens + cache.page_size - 1) // cache.page_size)
    page_rows: list[list[int]] = []
    for request_id, seq_len in zip(request_ids, lengths):
        seq = cache.sequence(request_id)
        required_pages = (seq_len + cache.page_size - 1) // cache.page_size if seq_len > 0 else 0
        pages = [int(page_id) for page_id in seq.page_ids[:required_pages]]
        page_rows.append(pages + [0] * (max_pages - len(pages)))
    page_table = torch.tensor(page_rows, dtype=torch.long, device=device)
    seq_lens = torch.tensor(lengths, dtype=torch.long, device=device)
    return page_table, seq_lens, cache_tokens
