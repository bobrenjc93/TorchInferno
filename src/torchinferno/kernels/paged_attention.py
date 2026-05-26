from __future__ import annotations

import torch
from torch import Tensor

from torchinferno.kernels.ops import KernelBackend, KernelConfig, triton_available
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import batched_paged_decode_attention_reference, paged_causal_attention


def paged_decode_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_id: str,
    position: int,
    *,
    config: KernelConfig | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Decode-token paged attention with Triton when available and safe."""

    config = KernelConfig() if config is None else config
    if query.ndim == 3:
        if query.size(1) != 1:
            return paged_causal_attention(
                query,
                cache,
                request_id,
                torch.tensor([position], device=query.device),
                enable_gqa=enable_gqa,
            )
        query_2d = query[:, 0, :]
    elif query.ndim == 2:
        query_2d = query
    else:
        raise ValueError("query must have shape [heads, head_dim] or [heads, 1, head_dim]")

    seq = cache.sequence(request_id)
    if _should_use_triton(query_2d, config, seq.length, cache, enable_gqa=enable_gqa):
        from torchinferno.kernels.triton_ops import triton_paged_decode_attention

        return triton_paged_decode_attention(query_2d, cache, request_id, position)[:, None, :]
    return paged_causal_attention(
        query_2d[:, None, :],
        cache,
        request_id,
        torch.tensor([position], device=query.device),
        enable_gqa=enable_gqa,
    )


def _should_use_triton(
    query: Tensor,
    config: KernelConfig,
    seq_len: int,
    cache: PagedKVCache,
    *,
    enable_gqa: bool = False,
) -> bool:
    if config.backend == KernelBackend.TORCH:
        return False
    if config.backend == KernelBackend.TRITON and not triton_available():
        raise RuntimeError("Triton backend requested but Triton is not available")
    if enable_gqa or query.size(0) != cache.num_key_value_heads:
        return False
    if not (query.is_cuda and cache.keys.is_cuda and cache.values.is_cuda and triton_available()):
        return False
    if seq_len < 1:
        return False
    return query.size(-1) <= 256 and cache.value_head_dim <= 256 and seq_len <= 4096


def batched_paged_decode_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_ids: tuple[str, ...] | list[str],
    positions: Tensor,
    *,
    config: KernelConfig | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Batched decode-token paged attention over independent request page tables."""

    if query.ndim == 4:
        if query.size(2) != 1:
            raise ValueError("batched decode query must have one token")
        query_rows = query[:, :, 0, :]
    elif query.ndim == 3:
        query_rows = query
    else:
        raise ValueError("query must have shape [batch, heads, head_dim] or [batch, heads, 1, head_dim]")
    if len(request_ids) != query_rows.size(0):
        raise ValueError("request_ids length must match query batch")
    if positions.ndim == 0:
        positions_tensor = positions.to(device=query_rows.device).expand(len(request_ids))
    elif positions.ndim == 1 and positions.numel() == len(request_ids):
        positions_tensor = positions.to(device=query_rows.device)
    else:
        raise ValueError("positions must be scalar or have shape [batch]")
    config = KernelConfig() if config is None else config
    query_4d = query_rows[:, :, None, :]
    if _should_use_batched_triton(query_4d, config, cache, request_ids, positions_tensor, enable_gqa=enable_gqa):
        from torchinferno.kernels.triton_ops import triton_batched_paged_gqa_decode_attention

        return triton_batched_paged_gqa_decode_attention(
            query_4d,
            cache,
            request_ids,
            positions_tensor,
        )
    row_positions = [int(value) for value in positions_tensor.detach().cpu().tolist()]
    return batched_paged_decode_attention_reference(
        query_4d,
        cache,
        request_ids,
        row_positions,
        enable_gqa=enable_gqa,
    )


def _should_use_batched_triton(
    query: Tensor,
    config: KernelConfig,
    cache: PagedKVCache,
    request_ids: tuple[str, ...] | list[str],
    positions: Tensor,
    *,
    enable_gqa: bool,
) -> bool:
    if config.backend == KernelBackend.TORCH:
        return False
    if config.backend == KernelBackend.TRITON and not triton_available():
        raise RuntimeError("Triton backend requested but Triton is not available")
    if not (query.is_cuda and cache.keys.is_cuda and cache.values.is_cuda and triton_available()):
        return False
    if query.ndim != 4 or query.size(2) != 1:
        return False
    if len(request_ids) != query.size(0) or positions.shape != (query.size(0),):
        return False
    if query.size(1) % cache.num_key_value_heads != 0:
        return False
    if not enable_gqa and query.size(1) != cache.num_key_value_heads:
        return False
    group_size = query.size(1) // cache.num_key_value_heads
    if group_size > 16:
        return False
    if query.size(-1) > 256 or cache.value_head_dim > 256:
        return False
    if query.stride(-1) != 1 or cache.keys.stride(-1) != 1 or cache.values.stride(-1) != 1:
        return False
    return True
