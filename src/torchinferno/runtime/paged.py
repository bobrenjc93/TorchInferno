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
        value_head_dim: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_pages < 1 or page_size < 1:
            raise ValueError("num_pages and page_size must be positive")
        self.page_size = page_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.value_head_dim = head_dim if value_head_dim is None else value_head_dim
        self.keys = torch.empty((num_pages, num_key_value_heads, page_size, head_dim), device=device, dtype=dtype)
        self.values = torch.empty(
            (num_pages, num_key_value_heads, page_size, self.value_head_dim),
            device=device,
            dtype=dtype,
        )
        self._free_pages = list(range(num_pages))
        self._page_refcounts = [0 for _ in range(num_pages)]
        self._sequences: dict[str, PagedSequence] = {}

    @property
    def free_pages(self) -> tuple[int, ...]:
        return tuple(self._free_pages)

    def sequence(self, request_id: str) -> PagedSequence:
        return self._sequences.setdefault(request_id, PagedSequence(request_id))

    def append(self, request_id: str, keys: Tensor, values: Tensor) -> PagedSequence:
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("keys and values must have shape [kv_heads, tokens, head_dim]")
        if keys.shape[:2] != values.shape[:2]:
            raise ValueError("keys and values must have the same head and token dimensions")
        if keys.shape[0] != self.num_key_value_heads or keys.shape[-1] != self.head_dim:
            raise ValueError("keys must have shape [kv_heads, tokens, head_dim]")
        if values.shape[-1] != self.value_head_dim:
            raise ValueError("values must have shape [kv_heads, tokens, value_head_dim]")

        seq = self.sequence(request_id)
        tokens = keys.size(1)
        self._ensure_capacity(seq, seq.length + tokens)
        for token in range(tokens):
            position = seq.length + token
            page_index = position // self.page_size
            page_id = self._prepare_page_for_write(seq, page_index)
            offset = position % self.page_size
            self.keys[page_id, :, offset, :].copy_(keys[:, token, :])
            self.values[page_id, :, offset, :].copy_(values[:, token, :])
        seq.length += tokens
        return seq

    def alias_prefix(self, source_request_id: str, target_request_id: str, tokens: int) -> PagedSequence:
        """Alias a prefix page span into a new request sequence.

        Shared pages are reference counted and copy-on-write during append, so
        this is safe for prefix reuse experiments even when the target extends a
        partially filled last page.
        """

        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        if target_request_id in self._sequences:
            raise ValueError(f"target request already exists: {target_request_id}")
        source = self._sequences[source_request_id]
        if tokens > source.length:
            raise ValueError("cannot alias more tokens than the source sequence contains")
        required_pages = math.ceil(tokens / self.page_size) if tokens else 0
        target = PagedSequence(target_request_id, list(source.page_ids[:required_pages]), tokens)
        for page_id in target.page_ids:
            self._page_refcounts[page_id] += 1
        self._sequences[target_request_id] = target
        return target

    def materialize(self, request_id: str) -> tuple[Tensor, Tensor]:
        seq = self._sequences[request_id]
        if seq.length == 0:
            return (
                self.keys.new_empty((self.num_key_value_heads, 0, self.head_dim)),
                self.values.new_empty((self.num_key_value_heads, 0, self.value_head_dim)),
            )
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
        for page_id in seq.page_ids:
            self._release_page(page_id)

    def _ensure_capacity(self, seq: PagedSequence, tokens: int) -> None:
        required_pages = math.ceil(tokens / self.page_size)
        while len(seq.page_ids) < required_pages:
            seq.page_ids.append(self._allocate_page())

    def _prepare_page_for_write(self, seq: PagedSequence, page_index: int) -> int:
        page_id = seq.page_ids[page_index]
        if self._page_refcounts[page_id] <= 1:
            return page_id

        new_page_id = self._allocate_page()
        self.keys[new_page_id].copy_(self.keys[page_id])
        self.values[new_page_id].copy_(self.values[page_id])
        self._release_page(page_id)
        seq.page_ids[page_index] = new_page_id
        return new_page_id

    def _allocate_page(self) -> int:
        if not self._free_pages:
            raise RuntimeError("paged KV cache is out of pages")
        page_id = self._free_pages.pop(0)
        self._page_refcounts[page_id] = 1
        return page_id

    def _release_page(self, page_id: int) -> None:
        self._page_refcounts[page_id] -= 1
        if self._page_refcounts[page_id] < 0:
            raise RuntimeError("paged KV cache page refcount went negative")
        if self._page_refcounts[page_id] == 0:
            self._free_pages.append(page_id)
            self._free_pages.sort()
