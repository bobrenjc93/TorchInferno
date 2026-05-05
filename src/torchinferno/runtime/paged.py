from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class PagedSequence:
    request_id: str
    page_ids: list[int] = field(default_factory=list)
    length: int = 0


class PagedKVCache:
    """Deterministic page-table-shaped KV cache.

    The DSv4 model still uses its compact append-only cache for the hot path.
    This class is the runtime scaffold for paged attention work: request-level
    allocation, page reuse, and materialization for focused kernel tests.
    """

    def __init__(
        self,
        *,
        num_pages: int,
        page_size: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_pages < 1 or page_size < 1:
            raise ValueError("num_pages and page_size must be positive")
        self.page_size = page_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.keys = torch.empty((num_pages, num_key_value_heads, page_size, head_dim), device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self._free_pages = list(range(num_pages))
        self._sequences: dict[str, PagedSequence] = {}

    @property
    def free_pages(self) -> tuple[int, ...]:
        return tuple(self._free_pages)

    def sequence(self, request_id: str) -> PagedSequence:
        return self._sequences.setdefault(request_id, PagedSequence(request_id))

    def append(self, request_id: str, keys: Tensor, values: Tensor) -> PagedSequence:
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        expected_prefix = (self.num_key_value_heads,)
        if keys.ndim != 3 or keys.shape[:1] != expected_prefix or keys.shape[-1] != self.head_dim:
            raise ValueError("keys must have shape [kv_heads, tokens, head_dim]")

        seq = self.sequence(request_id)
        tokens = keys.size(1)
        self._ensure_capacity(seq, seq.length + tokens)
        for token in range(tokens):
            position = seq.length + token
            page_id = seq.page_ids[position // self.page_size]
            offset = position % self.page_size
            self.keys[page_id, :, offset, :].copy_(keys[:, token, :])
            self.values[page_id, :, offset, :].copy_(values[:, token, :])
        seq.length += tokens
        return seq

    def materialize(self, request_id: str) -> tuple[Tensor, Tensor]:
        seq = self._sequences[request_id]
        keys = []
        values = []
        remaining = seq.length
        for page_id in seq.page_ids:
            take = min(self.page_size, remaining)
            keys.append(self.keys[page_id, :, :take, :])
            values.append(self.values[page_id, :, :take, :])
            remaining -= take
            if remaining == 0:
                break
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def free(self, request_id: str) -> None:
        seq = self._sequences.pop(request_id)
        self._free_pages.extend(seq.page_ids)
        self._free_pages.sort()

    def _ensure_capacity(self, seq: PagedSequence, tokens: int) -> None:
        required_pages = math.ceil(tokens / self.page_size)
        while len(seq.page_ids) < required_pages:
            if not self._free_pages:
                raise RuntimeError("paged KV cache is out of pages")
            seq.page_ids.append(self._free_pages.pop(0))
