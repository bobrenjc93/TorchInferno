from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from torchinferno.runtime.paged import PagedKVCache


def paged_causal_attention(
    query: Tensor,
    cache: PagedKVCache,
    request_id: str,
    positions: Tensor,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Reference paged causal attention over request pages without dense KV materialization."""

    if query.ndim != 3:
        raise ValueError("query must have shape [heads, tokens, head_dim]")
    if positions.ndim != 1 or positions.numel() != query.size(1):
        raise ValueError("positions must have shape [tokens]")
    if not cache.has_sequence(request_id):
        raise KeyError(request_id)
    seq = cache.sequence(request_id)
    query_heads = query.size(0)
    if cache.num_key_value_heads != query_heads:
        if not enable_gqa or query_heads % cache.num_key_value_heads != 0:
            raise ValueError("query heads must match KV heads unless enable_gqa=True with an integer ratio")
    scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    if seq.length == 0:
        return query.new_zeros((query_heads, query.size(1), cache.value_head_dim))

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
        keys = _expand_kv_heads(
            cache.keys[page_id, :, :take, :].to(device=query.device),
            query_heads,
        )
        scores = torch.matmul(query_float, keys.float().transpose(-1, -2)) * scale
        key_positions = torch.arange(key_start, key_start + take, device=query.device)
        allowed = key_positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(torch.float32).min)
        global_max = torch.maximum(global_max, scores.max(dim=-1).values)

    denom = torch.zeros_like(global_max)
    output = torch.zeros(
        (query_heads, query.size(1), cache.value_head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    for page_id, key_start, take in page_spans:
        keys = _expand_kv_heads(
            cache.keys[page_id, :, :take, :].to(device=query.device),
            query_heads,
        )
        values = _expand_kv_heads(
            cache.values[page_id, :, :take, :].to(device=query.device),
            query_heads,
        )
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
    enable_gqa: bool = False,
) -> Tensor:
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, tokens, head_dim]")
    if len(request_ids) != query.size(0):
        raise ValueError("request_ids length must match query batch")
    if positions.ndim == 1:
        if positions.numel() != query.size(2):
            raise ValueError("positions must match query tokens")
        row_positions = positions.to(device=query.device).expand(query.size(0), -1)
    elif positions.ndim == 2 and positions.size(0) == query.size(0):
        if positions.size(1) != query.size(2):
            raise ValueError("positions must match query tokens")
        row_positions = positions.to(device=query.device)
    else:
        raise ValueError("positions must have shape [tokens] or [batch, tokens]")
    seq_lens = [cache.sequence_length(request_id) for request_id in request_ids]
    max_seq_len = max(seq_lens, default=0)
    if max_seq_len <= 0:
        return query.new_zeros((query.size(0), query.size(1), query.size(2), cache.value_head_dim))
    keys, values = _materialize_batch(cache, request_ids, seq_lens, max_seq_len, device=query.device, dtype=query.dtype)
    key_positions = torch.arange(max_seq_len, device=query.device)
    causal_mask = key_positions[None, None, :] <= row_positions[:, :, None]
    if all(seq_len == max_seq_len for seq_len in seq_lens):
        attn_mask = causal_mask
    else:
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=query.device)
        length_mask = key_positions[None, None, :] < seq_lens_tensor[:, None, None]
        attn_mask = causal_mask & length_mask
    sdp_kwargs: dict[str, object] = {
        "attn_mask": attn_mask[:, None, :, :],
        "dropout_p": 0.0,
        "is_causal": False,
        "enable_gqa": enable_gqa,
    }
    if scale is not None:
        sdp_kwargs["scale"] = scale
    return F.scaled_dot_product_attention(query, keys, values, **sdp_kwargs)


def paged_decode_page_table(
    cache: PagedKVCache,
    request_ids: Sequence[str],
    positions: Tensor | Sequence[int] | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Build a dense page table and decode lengths for a request batch."""

    if not request_ids:
        raise ValueError("request_ids must be non-empty")
    if positions is None:
        position_values: list[int | None] = [None for _ in request_ids]
    elif isinstance(positions, Tensor):
        if positions.ndim == 0:
            position_values = [int(positions.item()) for _ in request_ids]
        elif positions.ndim == 1 and positions.numel() == len(request_ids):
            position_values = [int(value) for value in positions.detach().cpu().tolist()]
        else:
            raise ValueError("positions must be scalar or have shape [batch]")
    else:
        if len(positions) != len(request_ids):
            raise ValueError("positions length must match request_ids")
        position_values = [int(value) for value in positions]

    table_device = cache.keys.device if device is None else device
    page_rows: list[list[int]] = []
    lengths: list[int] = []
    max_pages = 1
    for request_id, position in zip(request_ids, position_values):
        if not cache.has_sequence(request_id):
            raise KeyError(request_id)
        seq = cache.sequence(request_id)
        length = int(seq.length)
        if position is not None:
            if position < 0:
                raise ValueError("decode position must be non-negative")
            length = min(length, position + 1)
        required_pages = math.ceil(length / cache.page_size) if length > 0 else 0
        pages = [int(page_id) for page_id in seq.page_ids[:required_pages]]
        page_rows.append(pages)
        lengths.append(length)
        max_pages = max(max_pages, len(pages))

    page_table = torch.zeros((len(request_ids), max_pages), dtype=torch.long, device=table_device)
    for row, pages in enumerate(page_rows):
        if pages:
            page_table[row, : len(pages)] = torch.tensor(pages, dtype=torch.long, device=table_device)
    seq_lens = torch.tensor(lengths, dtype=torch.long, device=table_device)
    return page_table, seq_lens


def batched_paged_decode_attention_reference(
    query: Tensor,
    cache: PagedKVCache,
    request_ids: Sequence[str],
    positions: Tensor | Sequence[int],
    *,
    enable_gqa: bool = False,
) -> Tensor:
    """Vectorized torch fallback for batched paged single-token decode attention."""

    query_rows = _normalize_decode_query(query)
    if len(request_ids) != query_rows.size(0):
        raise ValueError("request_ids length must match query batch")
    _page_table, seq_lens = paged_decode_page_table(cache, request_ids, positions, device=query_rows.device)
    if seq_lens.numel() == 0:
        return query_rows.new_empty((0, query_rows.size(1), 1, cache.value_head_dim))
    max_seq_len = int(seq_lens.max().item())
    if max_seq_len <= 0:
        return query_rows.new_zeros((query_rows.size(0), query_rows.size(1), 1, cache.value_head_dim))
    keys, values = _materialize_batch(cache, request_ids, seq_lens, max_seq_len, device=query_rows.device, dtype=query_rows.dtype)
    mask = torch.arange(max_seq_len, device=query_rows.device)[None, :] < seq_lens.to(device=query_rows.device)[:, None]
    # Padding slots (position >= seq_len) are gathered from unused cache rows that
    # may hold uninitialized memory. SDPA masks their attention weight to zero, but
    # 0 * NaN = NaN, so a stale NaN in a padded KV slot poisons the output even
    # though it is logically masked out. Zero the masked positions so the masked
    # weighted sum stays finite (padding contributes nothing either way).
    kv_keep = mask[:, None, :, None]
    keys = torch.where(kv_keep, keys, keys.new_zeros(()))
    values = torch.where(kv_keep, values, values.new_zeros(()))
    return F.scaled_dot_product_attention(
        query_rows,
        keys,
        values,
        attn_mask=mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=enable_gqa,
    )


def _materialize_batch(
    cache: PagedKVCache,
    request_ids: Sequence[str],
    seq_lens: Tensor | Sequence[int],
    max_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not _is_cuda_graph_capturing(device):
        return _materialize_batch_gather(
            cache,
            request_ids,
            seq_lens,
            max_seq_len,
            device=device,
            dtype=dtype,
        )
    return _materialize_batch_loop(
        cache,
        request_ids,
        seq_lens,
        max_seq_len,
        device=device,
        dtype=dtype,
    )


def _materialize_batch_gather(
    cache: PagedKVCache,
    request_ids: Sequence[str],
    seq_lens: Tensor | Sequence[int],
    max_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    lengths = _seq_lens_to_list(seq_lens)
    max_pages = max(1, math.ceil(max_seq_len / cache.page_size))
    page_rows: list[list[int]] = []
    for request_id, seq_len in zip(request_ids, lengths):
        seq = cache.sequence(request_id)
        required_pages = math.ceil(seq_len / cache.page_size) if seq_len > 0 else 0
        pages = [int(page_id) for page_id in seq.page_ids[:required_pages]]
        page_rows.append(pages + [0] * (max_pages - len(pages)))
    page_table = torch.tensor(page_rows, dtype=torch.long, device=cache.keys.device)
    flat_pages = page_table.reshape(-1)
    keys = _gather_pages(cache.keys, flat_pages, len(request_ids), max_pages, max_seq_len)
    values = _gather_pages(cache.values, flat_pages, len(request_ids), max_pages, max_seq_len)
    if keys.device != device or keys.dtype != dtype:
        keys = keys.to(device=device, dtype=dtype)
    if values.device != device or values.dtype != dtype:
        values = values.to(device=device, dtype=dtype)
    return keys, values


def _materialize_batch_loop(
    cache: PagedKVCache,
    request_ids: Sequence[str],
    seq_lens: Tensor | Sequence[int],
    max_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    keys = torch.zeros(
        (len(request_ids), cache.num_key_value_heads, max_seq_len, cache.head_dim),
        device=device,
        dtype=dtype,
    )
    values = torch.zeros(
        (len(request_ids), cache.num_key_value_heads, max_seq_len, cache.value_head_dim),
        device=device,
        dtype=dtype,
    )
    for row, request_id in enumerate(request_ids):
        seq = cache.sequence(request_id)
        remaining = int(seq_lens[row].item()) if isinstance(seq_lens, Tensor) else int(seq_lens[row])
        offset = 0
        for page_id in seq.page_ids:
            if remaining <= 0:
                break
            take = min(cache.page_size, remaining)
            page_keys = cache.keys[page_id, :, :take, :]
            page_values = cache.values[page_id, :, :take, :]
            if page_keys.device != device or page_keys.dtype != dtype:
                page_keys = page_keys.to(device=device, dtype=dtype)
            if page_values.device != device or page_values.dtype != dtype:
                page_values = page_values.to(device=device, dtype=dtype)
            keys[row, :, offset : offset + take, :].copy_(page_keys)
            values[row, :, offset : offset + take, :].copy_(page_values)
            offset += take
            remaining -= take
    return keys, values


def _gather_pages(page_storage: Tensor, flat_pages: Tensor, batch: int, max_pages: int, max_seq_len: int) -> Tensor:
    gathered = page_storage.index_select(0, flat_pages)
    gathered = gathered.view(batch, max_pages, page_storage.size(1), page_storage.size(2), page_storage.size(3))
    gathered = gathered.permute(0, 2, 1, 3, 4).reshape(
        batch,
        page_storage.size(1),
        max_pages * page_storage.size(2),
        page_storage.size(3),
    )
    return gathered[:, :, :max_seq_len, :]


def _seq_lens_to_list(seq_lens: Tensor | Sequence[int]) -> list[int]:
    if isinstance(seq_lens, Tensor):
        return [int(value) for value in seq_lens.detach().cpu().tolist()]
    return [int(value) for value in seq_lens]


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def _normalize_decode_query(query: Tensor) -> Tensor:
    if query.ndim == 4:
        if query.size(2) != 1:
            raise ValueError("batched decode query must have one token")
        return query
    if query.ndim == 3:
        return query[:, :, None, :]
    raise ValueError("query must have shape [batch, heads, head_dim] or [batch, heads, 1, head_dim]")


def _expand_kv_heads(tensor: Tensor, query_heads: int) -> Tensor:
    kv_heads = tensor.size(0)
    if kv_heads == query_heads:
        return tensor
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be an integer multiple of KV heads")
    return tensor.repeat_interleave(query_heads // kv_heads, dim=0)


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
