from __future__ import annotations

import math

import torch
from torch import Tensor

from torchinferno.runtime.paged import PagedKVCache


def paged_causal_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_id: str,
    positions: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Reference paged causal attention over request pages without dense KV materialization."""

    if query.ndim != 3:
        raise ValueError("query must have shape [heads, tokens, head_dim]")
    if positions.ndim != 1 or positions.numel() != query.size(1):
        raise ValueError("positions must have shape [tokens]")
    if not cache.has_sequence(request_id):
        raise KeyError(request_id)
    seq = cache.sequence(request_id)
    if cache.num_key_value_heads != query.size(0):
        raise ValueError("query and cache must have the same number of heads")
    scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    if seq.length == 0:
        return query.new_zeros((query.size(0), query.size(1), cache.value_head_dim))

    positions = positions.to(device=query.device)
    query_float = query.float()
    global_max = torch.full(
        (query.size(0), query.size(1)),
        torch.finfo(torch.float32).min,
        device=query.device,
        dtype=torch.float32,
    )

    page_spans = _page_spans(cache, request_id)
    for page_id, key_start, take in page_spans:
        keys = cache.keys[page_id, :, :take, :].to(device=query.device)
        scores = torch.matmul(query_float, keys.float().transpose(-1, -2)) * scale
        key_positions = torch.arange(key_start, key_start + take, device=query.device)
        allowed = key_positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(torch.float32).min)
        global_max = torch.maximum(global_max, scores.max(dim=-1).values)

    denom = torch.zeros_like(global_max)
    output = torch.zeros(
        (query.size(0), query.size(1), cache.value_head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    for page_id, key_start, take in page_spans:
        keys = cache.keys[page_id, :, :take, :].to(device=query.device)
        values = cache.values[page_id, :, :take, :].to(device=query.device)
        scores = torch.matmul(query_float, keys.float().transpose(-1, -2)) * scale
        key_positions = torch.arange(key_start, key_start + take, device=query.device)
        allowed = key_positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(torch.float32).min)
        weights = torch.exp(scores - global_max[:, :, None])
        denom = denom + weights.sum(dim=-1)
        output = output + torch.matmul(weights, values.float())
    return (output / denom.clamp_min(1e-20)[:, :, None]).to(dtype=query.dtype)


def batched_paged_causal_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_ids: tuple[str, ...] | list[str],
    positions: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, tokens, head_dim]")
    if len(request_ids) != query.size(0):
        raise ValueError("request_ids length must match query batch")
    if positions.ndim == 1:
        row_positions = [positions for _ in request_ids]
    elif positions.ndim == 2 and positions.size(0) == query.size(0):
        row_positions = [positions[row] for row in range(query.size(0))]
    else:
        raise ValueError("positions must have shape [tokens] or [batch, tokens]")
    return torch.stack(
        [
            paged_causal_attention(query[row], cache, request_id, row_positions[row], scale=scale)
            for row, request_id in enumerate(request_ids)
        ],
        dim=0,
    )


def _page_spans(cache: PagedKVCache, request_id: str) -> tuple[tuple[int, int, int], ...]:
    if not cache.has_sequence(request_id):
        raise KeyError(request_id)
    seq = cache.sequence(request_id)
    spans: list[tuple[int, int, int]] = []
    remaining = seq.length
    key_start = 0
    for page_id in seq.page_ids:
        take = min(cache.page_size, remaining)
        if take <= 0:
            break
        spans.append((page_id, key_start, take))
        key_start += take
        remaining -= take
    return tuple(spans)
