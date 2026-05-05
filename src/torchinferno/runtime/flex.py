from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor


MaskMod = Callable[[int, int], bool]


def flex_attention_available() -> bool:
    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except Exception:
        return False
    return True


def causal_mask_mod(query_position: int, key_position: int) -> bool:
    return key_position <= query_position


def flex_attention_or_fallback(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask_mod: Optional[MaskMod] = None,
) -> Tensor:
    """Tiny attention surface that can be swapped for torch flex attention.

    The fallback intentionally keeps the same q/k/v contract used by model
    attention: [batch, heads, tokens, head_dim].
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, head_dim]")
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if mask_mod is not None:
        mask = torch.empty((q.size(-2), k.size(-2)), device=q.device, dtype=torch.bool)
        for query_position in range(q.size(-2)):
            for key_position in range(k.size(-2)):
                mask[query_position, key_position] = mask_mod(query_position, key_position)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)
