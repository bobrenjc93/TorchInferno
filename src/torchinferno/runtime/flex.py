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
    flex_output = _try_flex_attention(q, k, v, mask_mod=mask_mod)
    if flex_output is not None:
        return flex_output
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if mask_mod is not None:
        mask = torch.empty((q.size(-2), k.size(-2)), device=q.device, dtype=torch.bool)
        for query_position in range(q.size(-2)):
            for key_position in range(k.size(-2)):
                mask[query_position, key_position] = mask_mod(query_position, key_position)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)


def _try_flex_attention(q: Tensor, k: Tensor, v: Tensor, *, mask_mod: Optional[MaskMod]) -> Tensor | None:
    if not flex_attention_available():
        return None
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        return None
    try:
        block_mask = None
        if mask_mod is not None:
            def flex_mask_mod(batch: Tensor, head: Tensor, query_position: Tensor, key_position: Tensor) -> Tensor:
                del batch, head
                return key_position <= query_position if mask_mod is causal_mask_mod else _evaluate_mask_mod(
                    mask_mod,
                    query_position,
                    key_position,
                )

            block_mask = create_block_mask(
                flex_mask_mod,
                q.size(0),
                q.size(1),
                q.size(-2),
                k.size(-2),
                device=q.device,
            )
        return flex_attention(q, k, v, block_mask=block_mask)
    except Exception:
        return None


def _evaluate_mask_mod(mask_mod: MaskMod, query_position: Tensor, key_position: Tensor) -> Tensor:
    result = torch.empty_like(query_position, dtype=torch.bool)
    flat_result = result.reshape(-1)
    flat_query = query_position.reshape(-1)
    flat_key = key_position.reshape(-1)
    for index in range(flat_result.numel()):
        flat_result[index] = bool(mask_mod(int(flat_query[index]), int(flat_key[index])))
    return result
