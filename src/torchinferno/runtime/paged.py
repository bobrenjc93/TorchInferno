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
        self._free_pages = list(reversed(range(num_pages)))
        self._page_refcounts = [0 for _ in range(num_pages)]
        self._sequences: dict[str, PagedSequence] = {}

    @property
    def free_pages(self) -> tuple[int, ...]:
        return tuple(sorted(self._free_pages))

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._sequences)

    def has_sequence(self, request_id: str) -> bool:
        return request_id in self._sequences

    def sequence_length(self, request_id: str) -> int:
        return self._sequences[request_id].length if request_id in self._sequences else 0

    def sequence(self, request_id: str) -> PagedSequence:
        return self._sequences.setdefault(request_id, PagedSequence(request_id))

    def truncate(self, request_id: str, tokens: int) -> PagedSequence:
        return self.set_length(request_id, tokens, release_pages=True)

    def set_length(self, request_id: str, tokens: int, *, release_pages: bool = False) -> PagedSequence:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        if tokens == 0 and release_pages:
            self.free(request_id)
            return self.sequence(request_id)
        seq = self.sequence(request_id) if tokens == 0 else self._sequences.get(request_id)
        if seq is None:
            raise KeyError(request_id)
        if tokens > seq.length and tokens > len(seq.page_ids) * self.page_size:
            raise ValueError("cannot extend a paged sequence beyond allocated pages")
        required_pages = math.ceil(tokens / self.page_size)
        if release_pages:
            for page_id in seq.page_ids[required_pages:]:
                self._release_page(page_id)
            del seq.page_ids[required_pages:]
        seq.length = tokens
        return seq

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
        token_offset = 0
        while token_offset < tokens:
            position = seq.length + token_offset
            page_index = position // self.page_size
            page_id = self._prepare_page_for_write(seq, page_index)
            page_offset = position % self.page_size
            take = min(self.page_size - page_offset, tokens - token_offset)
            self.keys[page_id, :, page_offset : page_offset + take, :].copy_(
                keys[:, token_offset : token_offset + take, :]
            )
            self.values[page_id, :, page_offset : page_offset + take, :].copy_(
                values[:, token_offset : token_offset + take, :]
            )
            token_offset += take
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
        if request_id not in self._sequences:
            return
        seq = self._sequences.pop(request_id)
        for page_id in seq.page_ids:
            self._release_page(page_id)

    def flashinfer_page_table(
        self, request_ids: list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the FlashInfer paged-KV plan tensors for a batch of sequences.

        Returns ``(kv_indptr, kv_indices, kv_last_page_len)`` as int32 tensors on
        the cache device, exactly the CSR layout
        ``BatchDecodeWithPagedKVCacheWrapper.plan`` expects: ``kv_indptr`` is the
        ``[B+1]`` cumulative per-request page count, ``kv_indices`` concatenates
        each request's page ids in order, and ``kv_last_page_len[i]`` is the number
        of valid tokens in request i's final page (1..page_size; 0 for an empty
        sequence). This is the bridge from the page pool to FlashInfer paged
        attention -- the missing piece for migrating the dense llama3-TP KV cache
        to true paged allocation (higher long-context concurrency).
        """
        indptr: list[int] = [0]
        indices: list[int] = []
        last_page_len: list[int] = []
        for request_id in request_ids:
            seq = self._sequences[request_id]
            num_pages = math.ceil(seq.length / self.page_size) if seq.length else 0
            indices.extend(seq.page_ids[:num_pages])
            indptr.append(indptr[-1] + num_pages)
            if seq.length == 0:
                last_page_len.append(0)
            else:
                last_page_len.append(seq.length - (num_pages - 1) * self.page_size)
        device = self.keys.device
        return (
            torch.tensor(indptr, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int32, device=device),
            torch.tensor(last_page_len, dtype=torch.int32, device=device),
        )

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
        page_id = self._free_pages.pop()
        self._page_refcounts[page_id] = 1
        return page_id

    def _release_page(self, page_id: int) -> None:
        self._page_refcounts[page_id] -= 1
        if self._page_refcounts[page_id] < 0:
            raise RuntimeError("paged KV cache page refcount went negative")
        if self._page_refcounts[page_id] == 0:
            self._free_pages.append(page_id)


class LayeredPagedKVCache:
    """Multi-layer paged KV pool with ONE block table per request shared across
    all layers.

    A token occupies the same logical page slot in every layer, so admission and
    eviction are decided once per sequence (a single block table) while each layer
    keeps its own NHD storage ``[num_pages, 2, page_size, kv_heads, head_dim]`` (2 =
    K,V) -- the vLLM-standard layout that FlashInfer's
    BatchDecodeWithPagedKVCacheWrapper consumes directly via layer_kv(). This is the
    core structure for migrating the dense per-layer llama3-TP cache
    (``[batch, kv_heads, max_seq_len, head_dim]``, which caps concurrency at ~48
    rows for long contexts) to true paging, where memory scales with ACTUAL tokens
    so far more long-context rows fit -> the queueing-bound multi_turn/long_output
    TPOT and TTFT/throughput gaps. COW prefix aliasing is intentionally left to
    PagedKVCache / a later step; this focuses on the core allocate/append/plan path.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_layers < 1 or num_pages < 1 or page_size < 1:
            raise ValueError("num_layers, num_pages and page_size must be positive")
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        # NHD paged layout per layer: [num_pages, 2, page_size, kv_heads, head_dim].
        self.kv = torch.empty(
            (num_layers, num_pages, 2, page_size, num_key_value_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._free_pages = list(reversed(range(num_pages)))
        self._sequences: dict[str, PagedSequence] = {}

    @property
    def free_pages(self) -> tuple[int, ...]:
        return tuple(sorted(self._free_pages))

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._sequences)

    def sequence_length(self, request_id: str) -> int:
        seq = self._sequences.get(request_id)
        return seq.length if seq is not None else 0

    def layer_kv(self, layer: int) -> Tensor:
        """Per-layer paged KV tensor [num_pages, 2, page_size, kv_heads, head_dim]
        to pass straight to a FlashInfer paged-decode wrapper for that layer."""
        return self.kv[layer]

    def reserve(self, request_id: str, total_tokens: int) -> PagedSequence:
        """Ensure the request's block table covers `total_tokens` (allocates pages
        from the pool as needed). Block table is shared by all layers."""
        seq = self._sequences.setdefault(request_id, PagedSequence(request_id))
        required_pages = math.ceil(total_tokens / self.page_size)
        while len(seq.page_ids) < required_pages:
            if not self._free_pages:
                raise RuntimeError("LayeredPagedKVCache is out of pages")
            seq.page_ids.append(self._free_pages.pop())
        return seq

    def extend(self, request_id: str, num_new_tokens: int) -> int:
        """Advance the sequence length by `num_new_tokens` (allocating pages),
        returning the START position the new tokens occupy. Call ONCE per step;
        then write_layer() each layer's K/V at the returned start."""
        seq = self._sequences.setdefault(request_id, PagedSequence(request_id))
        start = seq.length
        self.reserve(request_id, start + num_new_tokens)
        seq.length = start + num_new_tokens
        return start

    def write_layer(
        self, layer: int, request_id: str, keys: Tensor, values: Tensor, *, start: int
    ) -> None:
        """Scatter a layer's K/V (NHD ``[tokens, kv_heads, head_dim]``) into the
        request's pages starting at logical position `start`."""
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("keys/values must be [tokens, kv_heads, head_dim] (NHD)")
        seq = self._sequences[request_id]
        tokens = keys.size(0)
        if start + tokens > len(seq.page_ids) * self.page_size:
            raise ValueError("write exceeds reserved pages; call extend/reserve first")
        offset = 0
        while offset < tokens:
            position = start + offset
            page_id = seq.page_ids[position // self.page_size]
            page_offset = position % self.page_size
            take = min(self.page_size - page_offset, tokens - offset)
            self.kv[layer, page_id, 0, page_offset : page_offset + take].copy_(
                keys[offset : offset + take]
            )
            self.kv[layer, page_id, 1, page_offset : page_offset + take].copy_(
                values[offset : offset + take]
            )
            offset += take

    def materialize_layer(self, layer: int, request_id: str) -> tuple[Tensor, Tensor]:
        """Gather a layer's contiguous K/V (NHD ``[length, kv_heads, head_dim]``)
        for tests / reference attention."""
        seq = self._sequences[request_id]
        keys: list[Tensor] = []
        values: list[Tensor] = []
        remaining = seq.length
        for page_id in seq.page_ids:
            if remaining <= 0:
                break
            take = min(self.page_size, remaining)
            keys.append(self.kv[layer, page_id, 0, :take])
            values.append(self.kv[layer, page_id, 1, :take])
            remaining -= take
        if not keys:
            empty = self.kv.new_empty((0, self.num_key_value_heads, self.head_dim))
            return empty, empty.clone()
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def flashinfer_page_table(
        self, request_ids: list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """CSR (kv_indptr, kv_indices, kv_last_page_len) for a batch -- shared by
        every layer (the block table is layer-independent), so it is built once per
        decode step and reused across all layer wrappers."""
        indptr: list[int] = [0]
        indices: list[int] = []
        last_page_len: list[int] = []
        for request_id in request_ids:
            seq = self._sequences[request_id]
            num_pages = math.ceil(seq.length / self.page_size) if seq.length else 0
            indices.extend(seq.page_ids[:num_pages])
            indptr.append(indptr[-1] + num_pages)
            last_page_len.append(
                0 if seq.length == 0 else seq.length - (num_pages - 1) * self.page_size
            )
        device = self.kv.device
        return (
            torch.tensor(indptr, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int32, device=device),
            torch.tensor(last_page_len, dtype=torch.int32, device=device),
        )

    def free(self, request_id: str) -> None:
        seq = self._sequences.pop(request_id, None)
        if seq is None:
            return
        for page_id in seq.page_ids:
            self._free_pages.append(page_id)
