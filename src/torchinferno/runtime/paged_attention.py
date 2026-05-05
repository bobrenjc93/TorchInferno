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
    """Reference paged causal attention over a request's materialized pages."""

    if query.ndim != 3:
        raise ValueError("query must have shape [heads, tokens, head_dim]")
    keys, values = cache.materialize(request_id)
    if keys.size(0) != query.size(0):
        raise ValueError("query and cache must have the same number of heads")
    scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
    key_positions = torch.arange(keys.size(-2), device=query.device)
    allowed = key_positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    return torch.matmul(probs, values)
