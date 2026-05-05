from __future__ import annotations

import torch
from torch import Tensor

from torchinferno.kernels.ops import KernelBackend, KernelConfig, triton_available
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import paged_causal_attention


def paged_decode_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_id: str,
    position: int,
    *,
    config: KernelConfig | None = None,
) -> Tensor:
    """Decode-token paged attention with Triton when available and safe."""

    config = KernelConfig() if config is None else config
    if query.ndim == 3:
        if query.size(1) != 1:
            return paged_causal_attention(query, cache, request_id, torch.tensor([position], device=query.device))
        query_2d = query[:, 0, :]
    elif query.ndim == 2:
        query_2d = query
    else:
        raise ValueError("query must have shape [heads, head_dim] or [heads, 1, head_dim]")

    seq = cache.sequence(request_id)
    if _should_use_triton(query_2d, config, seq.length, cache):
        from torchinferno.kernels.triton_ops import triton_paged_decode_attention

        return triton_paged_decode_attention(query_2d, cache, request_id, position)[:, None, :]
    return paged_causal_attention(query_2d[:, None, :], cache, request_id, torch.tensor([position], device=query.device))


def _should_use_triton(query: Tensor, config: KernelConfig, seq_len: int, cache: PagedKVCache) -> bool:
    if config.backend == KernelBackend.TORCH:
        return False
    if config.backend == KernelBackend.TRITON and not triton_available():
        raise RuntimeError("Triton backend requested but Triton is not available")
    if not (query.is_cuda and cache.keys.is_cuda and cache.values.is_cuda and triton_available()):
        return False
    if seq_len < 1:
        return False
    return query.size(-1) <= 256 and cache.value_head_dim <= 256 and seq_len <= 4096
