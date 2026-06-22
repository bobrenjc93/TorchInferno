from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from torchinferno.models.hf import HF_CONFIG_NAME
from torchinferno.models.llama3.config import Llama3Config
from torchinferno.models.llama3.pipeline import (
    LLAMA3_70B_REPO_ID,
    _CheckpointTensorLoader,
    _build_inv_freq,
    _resolve_dtype,
    _rms_norm as _torch_rms_norm,
    resolve_llama3_checkpoint,
)
from torchinferno.runtime.options import env_flag, env_int, warn_optional_failure
from torchinferno.runtime.sampling import sample_next_token


_COMPILED_ROTATE_LLAMA = None
_COMPILED_ROTATE_LLAMA_CHECKED = False
_COMPILED_ROTATE_LLAMA_FAILED = False
_SYMM_REDUCE_BUFFERS: dict[tuple[str, int, str, str, tuple[int, ...]], Tensor] = {}
_SYMM_REDUCE_PROBED: set[tuple[str, int, str, str, tuple[int, ...]]] = set()
# Distinct eager-prefill symm-mem allreduce shapes seen, to bound buffer memory.
_SYMM_PREFILL_SHAPES: set[tuple[str, tuple[int, ...], int]] = set()
_SYMM_REDUCE_DISABLED = False
_SYMM_MEM_ALLREDUCE_MAX_BATCH_OVERRIDE: list[int | None] = [None]
_SYMM_MEM_ALLREDUCE_ENABLED_OVERRIDE: list[bool | None] = [None]
_DEFAULT_DECODE_STEP_MAX_BATCH = 64


@contextmanager
def symm_mem_allreduce_max_batch(max_batch: int | None, *, enabled: bool | None = None) -> Iterator[None]:
    if max_batch is not None and max_batch < 1:
        raise ValueError("max_batch must be positive")
    previous_batch = _SYMM_MEM_ALLREDUCE_MAX_BATCH_OVERRIDE[0]
    previous_enabled = _SYMM_MEM_ALLREDUCE_ENABLED_OVERRIDE[0]
    _SYMM_MEM_ALLREDUCE_MAX_BATCH_OVERRIDE[0] = max_batch
    _SYMM_MEM_ALLREDUCE_ENABLED_OVERRIDE[0] = enabled
    try:
        yield
    finally:
        _SYMM_MEM_ALLREDUCE_MAX_BATCH_OVERRIDE[0] = previous_batch
        _SYMM_MEM_ALLREDUCE_ENABLED_OVERRIDE[0] = previous_enabled


def _tp_flag(name: str, default: bool = True) -> bool:
    return env_flag(name, default)


def _tp_int(name: str, default: int, *, minimum: int | None = None) -> int:
    return env_int(name, default, minimum=minimum)


def _tp_env_set(name: str) -> bool:
    return name in os.environ


def _capture_needed_on_any_rank(needs_capture: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return needs_capture
    flag = torch.tensor([1 if needs_capture else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _capture_succeeded_on_all_ranks(succeeded: bool, device: torch.device) -> bool:
    # After a coordinated graph capture, every rank must agree it succeeded. If
    # capture throws on even one rank, that rank would fall back to eager while
    # the others replay, and the next collective (allreduce / sampler broadcast)
    # mismatches and the run hangs. MIN-reduce the success flag so a single
    # failure forces every rank to abandon the graph and run eager together.
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return succeeded
    flag = torch.tensor([1 if succeeded else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@dataclass(frozen=True)
class Llama3TensorParallelLoadReport:
    checkpoint: str
    dtype: str
    device: str
    rank: int
    world_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
        }


@dataclass
class _StaticCudaGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    output: Tensor


@dataclass
class _StaticQKVRotaryGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    static_cos: Tensor
    static_sin: Tensor
    q: Tensor
    k: Tensor
    v: Tensor


@dataclass
class _StaticPrefillActivationGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    output: Tensor


class Llama3TensorParallelLayerKVCache:
    cache_backend = "dense"

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        local_key_value_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, local_key_value_heads, max_seq_len, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._seq_lens = [0 for _ in range(batch_size)]
        self._uniform_seq_len: list[int | None] = [0]
        self._row_indices: tuple[int, ...] | None = None
        self._row_indices_tensor: Tensor | None = None

    @property
    def seq_len(self) -> int:
        return self.seq_len_for_rows(self._selected_rows())

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self.set_seq_len(seq_len)

    def seq_len_for_rows(self, rows: tuple[int, ...]) -> int:
        if not rows:
            return 0
        if any(row < 0 or row >= len(self._seq_lens) for row in rows):
            raise ValueError("cache row out of range")
        uniform_seq_len = self._uniform_seq_len[0]
        if uniform_seq_len is not None:
            return uniform_seq_len
        seq_len = self._seq_lens[rows[0]]
        if any(self._seq_lens[row] != seq_len for row in rows):
            raise ValueError("selected cache rows must have the same sequence length")
        return seq_len

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > self.max_seq_len:
            raise ValueError("seq_len exceeds KV cache capacity")
        self._set_rows_seq_len(self._selected_rows(), seq_len)

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, tokens, _ = keys.shape
        if batch > self.batch_size:
            raise ValueError("cache batch is smaller than incoming batch")
        uniform_seq_len = self._uniform_seq_len[0]
        if self._row_indices is None and uniform_seq_len is not None:
            rows: tuple[int, ...] | None = None
            start = uniform_seq_len
        else:
            rows = self._selected_rows(batch)
            start = self.seq_len_for_rows(rows)
        end = start + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        triton_target: tuple[Tensor, Tensor] | None = None
        if keys.is_cuda and values.is_cuda and _tp_flag("TORCHINFERNO_TRITON_KV_APPEND"):
            if self._row_indices is None:
                triton_target = (self.keys, self.values)
            else:
                if rows is None:
                    rows = self._selected_rows(batch)
                span = _contiguous_row_span(rows)
                if span is not None:
                    row_start, row_end = span
                    triton_target = (self.keys[row_start:row_end], self.values[row_start:row_end])
        if triton_target is not None:
            try:
                from torchinferno.kernels.triton_ops import triton_append_kv_cache

                triton_append_kv_cache(keys, values, triton_target[0], triton_target[1], start)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.triton_kv_append", exc)
                if self._row_indices is None:
                    self.keys[:batch, :, start:end, :].copy_(keys)
                    self.values[:batch, :, start:end, :].copy_(values)
                else:
                    if rows is None:
                        rows = self._selected_rows(batch)
                    span = _contiguous_row_span(rows)
                    if span is not None:
                        row_start, row_end = span
                        self.keys[row_start:row_end, :, start:end, :].copy_(keys)
                        self.values[row_start:row_end, :, start:end, :].copy_(values)
                    else:
                        index = torch.tensor(rows, dtype=torch.long, device=self.keys.device)
                        self.keys[index, :, start:end, :] = keys
                        self.values[index, :, start:end, :] = values
        else:
            if self._row_indices is None:
                self.keys[:batch, :, start:end, :].copy_(keys)
                self.values[:batch, :, start:end, :].copy_(values)
            else:
                if rows is None:
                    rows = self._selected_rows(batch)
                span = _contiguous_row_span(rows)
                if span is not None:
                    row_start, row_end = span
                    self.keys[row_start:row_end, :, start:end, :].copy_(keys)
                    self.values[row_start:row_end, :, start:end, :].copy_(values)
                else:
                    index = torch.tensor(rows, dtype=torch.long, device=self.keys.device)
                    self.keys[index, :, start:end, :] = keys
                    self.values[index, :, start:end, :] = values
        if rows is None:
            self._set_root_prefix_seq_len(batch, end)
        else:
            self._set_rows_seq_len(rows, end)
        if self._row_indices is None:
            return self.keys[:batch, :, :end, :], self.values[:batch, :, :end, :]
        if rows is None:
            rows = self._selected_rows(batch)
        span = _contiguous_row_span(rows)
        if span is not None:
            row_start, row_end = span
            return self.keys[row_start:row_end, :, :end, :], self.values[row_start:row_end, :, :end, :]
        index = torch.tensor(rows, dtype=torch.long, device=self.keys.device)
        return self.keys.index_select(0, index)[:, :, :end, :], self.values.index_select(0, index)[:, :, :end, :]

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "Llama3TensorParallelLayerKVCache":
        if not rows:
            raise ValueError("row view must select at least one cache row")
        physical_rows = tuple(self._physical_row(int(row)) for row in rows)
        view = object.__new__(Llama3TensorParallelLayerKVCache)
        view.keys = self.keys
        view.values = self.values
        view.max_seq_len = self.max_seq_len
        view.batch_size = len(physical_rows)
        view._seq_lens = self._seq_lens
        view._uniform_seq_len = self._uniform_seq_len
        view._row_indices = physical_rows
        view._row_indices_tensor = torch.tensor(physical_rows, dtype=torch.long, device=self.keys.device)
        return view

    def clear_row(self, row: int) -> None:
        physical_row = self._physical_row(row)
        self._set_rows_seq_len((physical_row,), 0)

    def copy_prefix_from(
        self,
        source: "Llama3TensorParallelLayerKVCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        source_physical = source._physical_row(source_row)
        dest_physical = self._physical_row(dest_row)
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        if tokens > source._seq_lens[source_physical] or tokens > self.max_seq_len:
            raise ValueError("prefix length exceeds cache row capacity")
        self.keys[dest_physical : dest_physical + 1, :, :tokens, :].copy_(
            source.keys[source_physical : source_physical + 1, :, :tokens, :]
        )
        self.values[dest_physical : dest_physical + 1, :, :tokens, :].copy_(
            source.values[source_physical : source_physical + 1, :, :tokens, :]
        )
        self._set_rows_seq_len((dest_physical,), tokens)

    def _selected_rows(self, batch: int | None = None) -> tuple[int, ...]:
        if self._row_indices is None:
            size = self.batch_size if batch is None else batch
            return tuple(range(size))
        if batch is None:
            return self._row_indices
        return self._row_indices[:batch]

    def _selected_row_indices_tensor(self, batch: int | None = None) -> Tensor | None:
        if self._row_indices_tensor is None:
            return None
        if batch is None:
            return self._row_indices_tensor
        return self._row_indices_tensor[:batch]

    def _contiguous_key_value_storage(self, batch: int) -> tuple[Tensor, Tensor] | None:
        rows = self._selected_rows(batch)
        span = _contiguous_row_span(rows)
        if span is None:
            return None
        row_start, row_end = span
        return self.keys[row_start:row_end], self.values[row_start:row_end]

    def _physical_row(self, row: int) -> int:
        if row < 0 or row >= self.batch_size:
            raise ValueError("cache row out of range")
        return row if self._row_indices is None else self._row_indices[row]

    def _set_root_prefix_seq_len(self, batch: int, seq_len: int) -> None:
        if batch == len(self._seq_lens):
            self._seq_lens[:] = [seq_len] * len(self._seq_lens)
            self._uniform_seq_len[0] = seq_len
            return
        prior_uniform = self._uniform_seq_len[0]
        self._seq_lens[:batch] = [seq_len] * batch
        self._uniform_seq_len[0] = self._partial_update_uniform_seq_len(
            seq_len,
            updated_rows=batch,
            prior_uniform=prior_uniform,
        )

    def _set_rows_seq_len(self, rows: tuple[int, ...], seq_len: int) -> None:
        if len(rows) == len(self._seq_lens) and set(rows) == set(range(len(self._seq_lens))):
            self._seq_lens[:] = [seq_len] * len(self._seq_lens)
            self._uniform_seq_len[0] = seq_len
            return
        prior_uniform = self._uniform_seq_len[0]
        for row in rows:
            self._seq_lens[row] = seq_len
        self._uniform_seq_len[0] = self._partial_update_uniform_seq_len(
            seq_len,
            updated_rows=len(rows),
            prior_uniform=prior_uniform,
        )

    def _partial_update_uniform_seq_len(
        self,
        seq_len: int,
        *,
        updated_rows: int,
        prior_uniform: int | None,
    ) -> int | None:
        if prior_uniform == seq_len:
            return seq_len
        if updated_rows * 2 >= len(self._seq_lens):
            return seq_len if all(value == seq_len for value in self._seq_lens) else None
        return None


class FlashInferLayerKVCache:
    """KV cache stored in FlashInfer's paged format for fused attention.

    Physical layout: [batch, 2, max_seq, kv_heads, head_dim] (NHD paged).
    Provides .keys/.values views in [batch, kv_heads, max_seq, head_dim] (BHSD)
    so existing model code (QKV projection, rotary, MLP) works unchanged.
    FlashInfer's attention kernel reads the paged tensor directly.
    """
    cache_backend = "flashinfer"

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        local_key_value_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.paged_kv = torch.zeros(
            batch_size, 2, max_seq_len, local_key_value_heads, head_dim,
            device=device, dtype=dtype,
        )
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._seq_lens = [0 for _ in range(batch_size)]
        self._uniform_seq_len: list[int | None] = [0]

    @property
    def keys(self) -> Tensor:
        # [batch, max_seq, kv_heads, head_dim] → [batch, kv_heads, max_seq, head_dim]
        return self.paged_kv[:, 0].transpose(1, 2)

    @property
    def values(self) -> Tensor:
        return self.paged_kv[:, 1].transpose(1, 2)

    @keys.setter
    def keys(self, value: Tensor) -> None:
        self.paged_kv[:, 0] = value.transpose(1, 2)

    @values.setter
    def values(self, value: Tensor) -> None:
        self.paged_kv[:, 1] = value.transpose(1, 2)

    @property
    def seq_len(self) -> int:
        u = self._uniform_seq_len[0]
        if u is not None:
            return int(u)
        if not self._seq_lens:
            return 0
        first = self._seq_lens[0]
        if all(s == first for s in self._seq_lens):
            return first
        raise ValueError("seq_len is not uniform across rows")

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self.set_seq_len(seq_len)

    def set_seq_len(self, seq_len: int) -> None:
        self._seq_lens[:] = [seq_len] * len(self._seq_lens)
        self._uniform_seq_len[0] = seq_len

    def reset(self) -> None:
        self.set_seq_len(0)

    def clear_row(self, row: int) -> None:
        if 0 <= row < len(self._seq_lens):
            self._seq_lens[row] = 0

    def copy_prefix_from(
        self,
        source: "FlashInferLayerKVCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if tokens <= 0:
            return
        self.keys[dest_row : dest_row + 1, :, :tokens, :].copy_(
            source.keys[source_row : source_row + 1, :, :tokens, :]
        )
        self.values[dest_row : dest_row + 1, :, :tokens, :].copy_(
            source.values[source_row : source_row + 1, :, :tokens, :]
        )
        if 0 <= dest_row < len(self._seq_lens):
            self._seq_lens[dest_row] = tokens
            self._uniform_seq_len[0] = None

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = self.seq_len
        tokens = k.size(2)
        end = seq_len + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        batch = min(k.size(0), self.batch_size)
        self.keys[:batch, :, seq_len:end, :] = k[:batch]
        self.values[:batch, :, seq_len:end, :] = v[:batch]
        self.set_seq_len(end)
        return self.keys[:batch, :, :end, :], self.values[:batch, :, :end, :]

    def _selected_rows(self, batch: int | None = None) -> tuple[int, ...]:
        size = self.batch_size if batch is None else batch
        return tuple(range(size))

    def _physical_row(self, row: int) -> int:
        return row

    def _selected_row_indices_tensor(self, batch: int | None = None) -> Tensor | None:
        return None

    def seq_len_for_rows(self, rows: tuple[int, ...]) -> int:
        if not rows:
            return 0
        first = self._seq_lens[rows[0]] if rows[0] < len(self._seq_lens) else 0
        return first

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "_FlashInferLayerKVCacheView":
        rows = tuple(rows) if not isinstance(rows, tuple) else rows
        return _FlashInferLayerKVCacheView(self, rows)


class _FlashInferLayerKVCacheView:
    cache_backend = "flashinfer"

    def __init__(self, parent: FlashInferLayerKVCache, rows: tuple[int, ...]) -> None:
        self._parent = parent
        self._rows = rows
        self._row_list = list(rows)
        self.max_seq_len = parent.max_seq_len
        self.batch_size = len(rows)
        self._uniform_seq_len: list[int | None] = [None]

    @property
    def paged_kv(self) -> Tensor:
        return self._parent.paged_kv[self._row_list]

    @property
    def _seq_lens(self) -> list[int]:
        return [self._parent._seq_lens[r] if 0 <= r < len(self._parent._seq_lens) else 0 for r in self._rows]

    @property
    def keys(self) -> Tensor:
        return self._parent.paged_kv[self._row_list, 0].transpose(1, 2)

    @keys.setter
    def keys(self, value: Tensor) -> None:
        self._parent.paged_kv[self._row_list, 0] = value.transpose(1, 2)

    @property
    def values(self) -> Tensor:
        return self._parent.paged_kv[self._row_list, 1].transpose(1, 2)

    @values.setter
    def values(self, value: Tensor) -> None:
        self._parent.paged_kv[self._row_list, 1] = value.transpose(1, 2)

    @property
    def seq_len(self) -> int:
        sl = self._seq_lens
        if not sl:
            return 0
        first = sl[0]
        if all(s == first for s in sl):
            return first
        raise ValueError("seq_len is not uniform across rows")

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self.set_seq_len(seq_len)

    def set_seq_len(self, seq_len: int) -> None:
        for r in self._rows:
            if 0 <= r < len(self._parent._seq_lens):
                self._parent._seq_lens[r] = seq_len
        self._parent._uniform_seq_len[0] = None

    def reset(self) -> None:
        self.set_seq_len(0)

    def clear_row(self, row: int) -> None:
        if 0 <= row < len(self._rows):
            physical = self._rows[row]
            if 0 <= physical < len(self._parent._seq_lens):
                self._parent._seq_lens[physical] = 0
            self._parent._uniform_seq_len[0] = None

    def _selected_rows(self, batch: int | None = None) -> tuple[int, ...]:
        size = self.batch_size if batch is None else batch
        return tuple(range(size))

    def _physical_row(self, row: int) -> int:
        return row

    def _selected_row_indices_tensor(self, batch: int | None = None) -> Tensor | None:
        return None

    def seq_len_for_rows(self, rows: tuple[int, ...]) -> int:
        sl = self._seq_lens
        if not rows:
            return 0
        return sl[rows[0]] if rows[0] < len(sl) else 0

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        sl = self._seq_lens
        seq_len = sl[0] if sl else 0
        tokens = k.size(2)
        end = seq_len + tokens
        rl = self._row_list
        self._parent.paged_kv[rl, 0, seq_len:end, :, :] = k.permute(0, 2, 1, 3)
        self._parent.paged_kv[rl, 1, seq_len:end, :, :] = v.permute(0, 2, 1, 3)
        self.set_seq_len(end)
        return self.keys[:self.batch_size, :, :end, :], self.values[:self.batch_size, :, :end, :]

    def copy_prefix_from(
        self,
        source: "FlashInferLayerKVCache | _FlashInferLayerKVCacheView",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if tokens <= 0:
            return
        src_parent = source._parent if isinstance(source, _FlashInferLayerKVCacheView) else source
        src_physical = source._rows[source_row] if isinstance(source, _FlashInferLayerKVCacheView) else source_row
        dst_physical = self._rows[dest_row]
        self._parent.paged_kv[dst_physical, :, :tokens, :, :] = src_parent.paged_kv[src_physical, :, :tokens, :, :]
        if 0 <= dst_physical < len(self._parent._seq_lens):
            self._parent._seq_lens[dst_physical] = tokens
            self._parent._uniform_seq_len[0] = None


class PagedLlama3TensorParallelLayerKVCache:
    cache_backend = "paged"

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        local_key_value_heads: int,
        head_dim: int,
        *,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be positive")
        from torchinferno.runtime.paged import PagedKVCache

        pages_per_row = max(1, (max_seq_len + page_size - 1) // page_size)
        self.pages = PagedKVCache(
            num_pages=max(1, batch_size * pages_per_row),
            page_size=page_size,
            num_key_value_heads=local_key_value_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self.request_ids = tuple(f"batch-{idx}" for idx in range(batch_size))
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._row_indices: tuple[int, ...] | None = None

    @property
    def seq_len(self) -> int:
        return self.seq_len_for_rows(self._selected_rows())

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        self.set_seq_len(seq_len)

    def seq_len_for_row(self, row: int) -> int:
        physical_row = self._physical_row(row)
        return self.pages.sequence_length(self.request_ids[physical_row])

    def seq_len_for_rows(self, rows: tuple[int, ...]) -> int:
        if not rows:
            return 0
        seq_len = self.pages.sequence_length(self.request_ids[rows[0]])
        if any(self.pages.sequence_length(self.request_ids[row]) != seq_len for row in rows):
            raise ValueError("selected cache rows must have the same sequence length")
        return seq_len

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > self.max_seq_len:
            raise ValueError("seq_len exceeds KV cache capacity")
        rows = self._selected_rows()
        if all(self.pages.sequence_length(self.request_ids[row]) == seq_len for row in rows):
            return
        for row in rows:
            request_id = self.request_ids[row]
            current = self.pages.sequence_length(request_id)
            if seq_len > current:
                allocated = len(self.pages.sequence(request_id).page_ids) * self.pages.page_size
                if seq_len > allocated:
                    raise ValueError("paged Llama3 cache seq_len cannot extend missing KV state")
            if seq_len < current:
                self.pages.set_length(request_id, seq_len)
            elif seq_len > current:
                self.pages.set_length(request_id, seq_len)

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        rows = self._append_pages(keys, values)
        return self.materialize(rows)

    def append_and_attend(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        positions: Tensor,
        *,
        enable_gqa: bool,
    ) -> Tensor:
        rows = self._append_pages(keys, values)
        from torchinferno.runtime.paged_attention import batched_paged_causal_attention

        return batched_paged_causal_attention(
            query,
            self.pages,
            tuple(self.request_ids[row] for row in rows),
            positions,
            enable_gqa=enable_gqa,
        )

    def append_and_attend_ragged(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        positions: Tensor,
        row_indices: Tensor | None,
        *,
        enable_gqa: bool,
    ) -> Tensor:
        if keys.ndim != 4 or values.ndim != 4 or keys.size(2) != 1 or values.size(2) != 1:
            raise ValueError("ragged paged KV append expects one token")
        if getattr(self, "_torchinferno_paged_decode_graph_active", False):
            page_table = getattr(self, "_torchinferno_paged_decode_page_table", None)
            decode_seq_lens = getattr(self, "_torchinferno_paged_decode_seq_lens", None)
            cache_tokens = getattr(self, "_torchinferno_paged_decode_cache_tokens", None)
            if page_table is not None and decode_seq_lens is not None and cache_tokens is not None and query.is_cuda:
                from torchinferno.kernels.triton_ops import (
                    triton_batched_paged_gqa_decode_attention_with_table,
                    triton_paged_append_kv_cache_ragged,
                )

                triton_paged_append_kv_cache_ragged(
                    keys,
                    values,
                    self.pages.keys,
                    self.pages.values,
                    page_table,
                    positions,
                    page_size=self.pages.page_size,
                )
                return triton_batched_paged_gqa_decode_attention_with_table(
                    query,
                    self.pages.keys,
                    self.pages.values,
                    page_table,
                    decode_seq_lens,
                    page_size=self.pages.page_size,
                    cache_tokens=int(cache_tokens),
                    num_key_value_heads=self.pages.num_key_value_heads,
                    value_head_dim=self.pages.value_head_dim,
                )
        rows = getattr(self, "_torchinferno_ragged_decode_rows", None)
        positions_cpu = getattr(self, "_torchinferno_ragged_decode_positions", None)
        if rows is None:
            rows = self._selected_rows(keys.size(0)) if row_indices is None else tuple(
                self._physical_row(int(row)) for row in row_indices.detach().cpu().tolist()
            )
        if positions_cpu is None:
            positions_cpu = tuple(int(position) for position in positions.detach().cpu().tolist())
        if len(rows) != len(positions_cpu):
            raise ValueError("ragged paged KV positions must match batch")
        append_rows: list[int] = []
        append_pages: list[int] = []
        append_offsets: list[int] = []
        for incoming_row, (cache_row, position) in enumerate(zip(rows, positions_cpu)):
            request_id = self.request_ids[cache_row]
            current = self.pages.sequence_length(request_id)
            if current == position:
                if position + 1 > self.max_seq_len:
                    raise ValueError("KV cache capacity exceeded")
                seq = self.pages.sequence(request_id)
                self.pages._ensure_capacity(seq, position + 1)
                page_index = position // self.pages.page_size
                page_id = self.pages._prepare_page_for_write(seq, page_index)
                append_rows.append(incoming_row)
                append_pages.append(page_id)
                append_offsets.append(position % self.pages.page_size)
                seq.length += 1
            elif current == position + 1:
                # Static bucket padding rows may be replayed without advancing seq_lens.
                pass
            else:
                raise ValueError("paged Llama3 ragged decode position does not match row state")
        if append_rows:
            row_index = torch.tensor(append_rows, dtype=torch.long, device=keys.device)
            page_index = torch.tensor(append_pages, dtype=torch.long, device=keys.device)
            page_offsets = torch.tensor(append_offsets, dtype=torch.long, device=keys.device)
            self.pages.keys[page_index, :, page_offsets, :] = keys.index_select(0, row_index)[:, :, 0, :]
            self.pages.values[page_index, :, page_offsets, :] = values.index_select(0, row_index)[:, :, 0, :]
        from torchinferno.kernels import batched_paged_decode_attention

        return batched_paged_decode_attention(
            query,
            self.pages,
            tuple(self.request_ids[row] for row in rows),
            positions,
            enable_gqa=enable_gqa,
        )

    def materialize(self, rows: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        keys = []
        values = []
        for row in rows:
            row_keys, row_values = self.pages.materialize(self.request_ids[row])
            keys.append(row_keys)
            values.append(row_values)
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def materialize_row(self, row: int) -> tuple[Tensor, Tensor]:
        physical_row = self._physical_row(row)
        return self.pages.materialize(self.request_ids[physical_row])

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "PagedLlama3TensorParallelLayerKVCache":
        if not rows:
            raise ValueError("row view must select at least one cache row")
        physical_rows = tuple(self._physical_row(int(row)) for row in rows)
        view = object.__new__(PagedLlama3TensorParallelLayerKVCache)
        view.pages = self.pages
        view.request_ids = self.request_ids
        view.max_seq_len = self.max_seq_len
        view.batch_size = len(physical_rows)
        view._row_indices = physical_rows
        return view

    def clear_row(self, row: int) -> None:
        physical_row = self._physical_row(row)
        self.pages.free(self.request_ids[physical_row])

    def copy_prefix_from(
        self,
        source: "Llama3TensorParallelLayerKVCache | PagedLlama3TensorParallelLayerKVCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        source_physical = source._physical_row(source_row)
        dest_physical = self._physical_row(dest_row)
        if tokens > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        if isinstance(source, PagedLlama3TensorParallelLayerKVCache):
            source_len = source.pages.sequence_length(source.request_ids[source_physical])
            if tokens > source_len:
                raise ValueError("source cache range is invalid")
            if source.pages is self.pages and source_physical == dest_physical:
                return
            self.clear_row(dest_row)
            if tokens and source.pages is self.pages:
                self.pages.alias_prefix(
                    source.request_ids[source_physical],
                    self.request_ids[dest_physical],
                    tokens,
                )
                return
            source_keys, source_values = source.pages.materialize(source.request_ids[source_physical])
        else:
            source_len = source._seq_lens[source_physical]
            if tokens > source_len:
                raise ValueError("source cache range is invalid")
            source_keys = source.keys[source_physical, :, :tokens, :]
            source_values = source.values[source_physical, :, :tokens, :]
            self.clear_row(dest_row)
        if tokens:
            self.pages.append(
                self.request_ids[dest_physical],
                source_keys[:, :tokens, :],
                source_values[:, :tokens, :],
            )

    def _append_pages(self, keys: Tensor, values: Tensor) -> tuple[int, ...]:
        batch, _, tokens, _ = keys.shape
        if batch > self.batch_size:
            raise ValueError("cache batch is smaller than incoming batch")
        rows = self._selected_rows(batch)
        start = self.seq_len_for_rows(rows)
        end = start + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        for incoming_row, cache_row in enumerate(rows):
            self.pages.append(self.request_ids[cache_row], keys[incoming_row], values[incoming_row])
        return rows

    def _selected_rows(self, batch: int | None = None) -> tuple[int, ...]:
        if self._row_indices is None:
            size = self.batch_size if batch is None else batch
            return tuple(range(size))
        if batch is None:
            return self._row_indices
        return self._row_indices[:batch]

    def _physical_row(self, row: int) -> int:
        if row < 0 or row >= self.batch_size:
            raise ValueError("cache row out of range")
        return row if self._row_indices is None else self._row_indices[row]


def _contiguous_row_span(rows: tuple[int, ...]) -> tuple[int, int] | None:
    if not rows:
        return None
    start = rows[0]
    for offset, row in enumerate(rows):
        if row != start + offset:
            return None
    return start, start + len(rows)


class Llama3TensorParallelCache:
    def __init__(
        self,
        layers: list[Llama3TensorParallelLayerKVCache | PagedLlama3TensorParallelLayerKVCache],
        *,
        cache_backend: str = "dense",
    ) -> None:
        self.layers = layers
        self.cache_backend = cache_backend
        self._graph_cache_id = id(self)

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        for layer in self.layers:
            if seq_len > layer.max_seq_len:
                raise ValueError("seq_len exceeds KV cache capacity")
        for layer in self.layers:
            layer.set_seq_len(seq_len)

    def reset(self) -> None:
        self.set_seq_len(0)

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "Llama3TensorParallelCache":
        view = Llama3TensorParallelCache(
            [layer.for_rows(rows) for layer in self.layers],
            cache_backend=self.cache_backend,
        )
        view._graph_cache_id = self._graph_cache_id
        view._parent_cache = self
        view._rows = tuple(rows)
        if getattr(self, "_skip_capture_sync", False):
            view._skip_capture_sync = True
        if getattr(self, "_block_decode_graph_captures", False):
            view._block_decode_graph_captures = True
        if getattr(self, "_compiled_prefill_ready", False):
            view._compiled_prefill_ready = True
        return view

    def clear_row(self, row: int) -> None:
        for layer in self.layers:
            layer.clear_row(row)

    def copy_prefix_from(
        self,
        source: "Llama3TensorParallelCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if len(self.layers) != len(source.layers):
            raise ValueError("source cache must have the same number of layers")
        for target_layer, source_layer in zip(self.layers, source.layers):
            target_layer.copy_prefix_from(source_layer, tokens, source_row=source_row, dest_row=dest_row)


def _set_paged_ragged_decode_context(
    cache: Llama3TensorParallelCache,
    *,
    batch: int,
    cache_positions: Tensor,
    row_indices: Tensor | None,
    device: torch.device,
) -> bool:
    if cache.cache_backend != "paged" or _cuda_stream_is_capturing(device):
        return False
    positions = tuple(int(position) for position in cache_positions.detach().cpu().tolist())
    if len(positions) != batch:
        return False
    row_values = None if row_indices is None else tuple(int(row) for row in row_indices.detach().cpu().tolist())
    if row_values is not None and len(row_values) != batch:
        return False
    for layer in cache.layers:
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        rows = layer._selected_rows(batch) if row_values is None else tuple(layer._physical_row(row) for row in row_values)
        layer._torchinferno_ragged_decode_rows = rows
        layer._torchinferno_ragged_decode_positions = positions
    return True


def _clear_paged_ragged_decode_context(cache: Llama3TensorParallelCache) -> None:
    for layer in cache.layers:
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        for attr in ("_torchinferno_ragged_decode_rows", "_torchinferno_ragged_decode_positions"):
            if hasattr(layer, attr):
                delattr(layer, attr)


def _prepare_paged_ragged_decode_graph_state(
    cache: Llama3TensorParallelCache,
    *,
    batch: int,
    cache_positions: Tensor,
    row_indices: Tensor | None,
    device: torch.device,
) -> None:
    if getattr(cache, "cache_backend", "dense") != "paged":
        return
    positions = tuple(int(position) for position in cache_positions.detach().cpu().tolist())
    if len(positions) != batch:
        return
    row_values = None if row_indices is None else tuple(int(row) for row in row_indices.detach().cpu().tolist())
    if row_values is not None and len(row_values) != batch:
        return
    decode_lengths = cache_positions.to(device=device, dtype=torch.long) + 1
    for layer in cache.layers:
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        rows = layer._selected_rows(batch) if row_values is None else tuple(layer._physical_row(row) for row in row_values)
        pages_per_row = max(1, (layer.max_seq_len + layer.pages.page_size - 1) // layer.pages.page_size)
        page_rows: list[list[int]] = []
        for row, position in zip(rows, positions):
            if position < 0 or position >= layer.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            request_id = layer.request_ids[row]
            seq = layer.pages.sequence(request_id)
            layer.pages._ensure_capacity(seq, position + 1)
            layer.pages._prepare_page_for_write(seq, position // layer.pages.page_size)
            pages = [int(page_id) for page_id in seq.page_ids[:pages_per_row]]
            page_rows.append(pages + [0] * (pages_per_row - len(pages)))
        page_table = getattr(layer, "_torchinferno_paged_decode_page_table", None)
        if not isinstance(page_table, Tensor) or page_table.shape != (batch, pages_per_row):
            page_table = torch.empty((batch, pages_per_row), dtype=torch.long, device=device)
            layer._torchinferno_paged_decode_page_table = page_table
        page_table.copy_(torch.tensor(page_rows, dtype=torch.long, device=device))
        seq_lens = getattr(layer, "_torchinferno_paged_decode_seq_lens", None)
        if not isinstance(seq_lens, Tensor) or seq_lens.shape != (batch,):
            seq_lens = torch.empty((batch,), dtype=torch.long, device=device)
            layer._torchinferno_paged_decode_seq_lens = seq_lens
        seq_lens.copy_(decode_lengths)
        layer._torchinferno_paged_decode_cache_tokens = int(layer.max_seq_len)


def _advance_paged_ragged_decode_cache_lengths(
    cache: Llama3TensorParallelCache,
    *,
    batch: int,
    cache_positions: Tensor,
    row_indices: Tensor | None,
) -> None:
    if getattr(cache, "cache_backend", "dense") != "paged":
        return
    positions = tuple(int(position) for position in cache_positions.detach().cpu().tolist())
    if len(positions) != batch:
        return
    row_values = None if row_indices is None else tuple(int(row) for row in row_indices.detach().cpu().tolist())
    if row_values is not None and len(row_values) != batch:
        return
    for layer in cache.layers:
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        rows = layer._selected_rows(batch) if row_values is None else tuple(layer._physical_row(row) for row in row_values)
        for row, position in zip(rows, positions):
            request_id = layer.request_ids[row]
            seq = layer.pages.sequence(request_id)
            if seq.length < position + 1:
                seq.length = position + 1


def _set_paged_ragged_decode_graph_active(cache: Llama3TensorParallelCache, active: bool) -> None:
    if getattr(cache, "cache_backend", "dense") != "paged":
        return
    for layer in cache.layers:
        if isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            layer._torchinferno_paged_decode_graph_active = bool(active)


def _cuda_stream_is_capturing(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_current_stream_capturing())


@dataclass
class _StaticDecodeGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_position: Tensor
    static_cache_positions: Tensor | None
    static_row_indices: Tensor | None
    static_attention_length: Tensor
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_token: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    attention_block_size: int


@dataclass
class _StaticDecodeLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_position: Tensor
    static_cache_positions: Tensor | None
    static_row_indices: Tensor | None
    static_attention_length: Tensor
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    attention_block_size: int


@dataclass
class _StaticRaggedDecodeLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_positions: Tensor
    static_row_indices: Tensor | None
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int


@dataclass
class _StaticRaggedDecodeGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_positions: Tensor
    static_row_indices: Tensor | None
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_token: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int


@dataclass
class _StaticRaggedPrefillLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor  # [batch, suffix_bucket]
    static_start_positions: Tensor  # [batch] per-row prefix length (write start)
    static_write_positions: Tensor  # [batch, suffix_bucket] absolute KV write columns
    static_row_indices: Tensor | None  # [batch] scattered physical rows
    static_rotary_cos: Tensor  # [batch, suffix_bucket, rotary_dim]
    static_rotary_sin: Tensor  # [batch, suffix_bucket, rotary_dim]
    static_logit_positions: Tensor  # [batch] real last-token index per row
    static_src_prefix_row: Tensor | None  # [1] shared-prefix source row (folded copy)
    output_logits: Tensor  # [batch, 1, local_vocab_size]
    cache: Llama3TensorParallelCache
    max_seq_len: int
    suffix_bucket: int
    context_len: int | None


@dataclass
class _StaticPrefillGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    output_token: Tensor
    cache: Llama3TensorParallelCache
    prompt_tokens: int
    initial_seq_len: int
    max_seq_len: int


@dataclass
class _StaticPrefillLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_logit_positions: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    prompt_tokens: int
    initial_seq_len: int
    max_seq_len: int


@dataclass
class _StaticPrefillSelectedLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_logit_positions: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    prompt_tokens: int
    initial_seq_len: int
    max_seq_len: int


class _Llama3TensorParallelLayer:
    def __init__(
        self,
        config: Llama3Config,
        layer_id: int,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        weights: dict[str, Tensor],
    ) -> None:
        if config.num_attention_heads % world_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel world size")
        if config.num_key_value_heads % world_size != 0:
            raise ValueError("num_key_value_heads must be divisible by tensor parallel world size")
        if config.intermediate_size % world_size != 0:
            raise ValueError("intermediate_size must be divisible by tensor parallel world size")
        self.config = config
        self.layer_id = layer_id
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_attention_heads = config.num_attention_heads // world_size
        self.local_key_value_heads = config.num_key_value_heads // world_size
        self.local_hidden_size = self.local_attention_heads * config.head_dim
        self.local_key_value_size = self.local_key_value_heads * config.head_dim
        self.local_intermediate_size = config.intermediate_size // world_size
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        self.qkv_proj_weight = weights["self_attn.qkv_proj.weight"]
        self.o_proj_weight = weights["self_attn.o_proj.weight"]
        self.gate_up_proj_weight = weights["mlp.gate_up_proj.weight"]
        self.down_proj_weight = weights["mlp.down_proj.weight"]
        self.qkv_proj_weight_decode = _maybe_decode_weight_t(self.qkv_proj_weight)
        self.o_proj_weight_decode = _maybe_decode_weight_t(self.o_proj_weight)
        self.gate_up_proj_weight_decode = _maybe_decode_weight_t(self.gate_up_proj_weight)
        self.down_proj_weight_decode = _maybe_decode_weight_t(self.down_proj_weight)
        self.inv_freq = _build_inv_freq(config, device)
        self.profile_seconds: dict[str, float] | None = None
        self.profile_counts: dict[str, int] | None = None
        self._mlp_project_graph: _StaticCudaGraphCall | None = None
        self._mlp_project_graph_failed = False
        self._qkv_rotary_graphs: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            _StaticQKVRotaryGraphCall,
        ] = {}
        self._qkv_rotary_graph_failed = False
        self._input_qkv_rotary_graphs: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            _StaticQKVRotaryGraphCall,
        ] = {}
        self._input_qkv_rotary_graph_failed = False
        self._post_mlp_project_graph: _StaticCudaGraphCall | None = None
        self._post_mlp_project_graph_failed = False
        self._attention_o_graph: _StaticCudaGraphCall | None = None
        self._attention_o_graph_failed = False
        self._prefill_gate_up_activation_graphs: dict[
            tuple[int, ...],
            _StaticPrefillActivationGraphCall,
        ] = {}
        self._prefill_gate_up_activation_graph_failed = False
        self._symm_reduce_failed = False

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        if self.profile_seconds is None or self.profile_counts is None:
            residual = hidden
            hidden = residual + self._attention_from_hidden(hidden, positions, rotary, cache)
            residual = hidden
            return residual + self._post_attention_mlp_project(hidden)

        residual = hidden
        attn_in = self._profile_block(
            "norm.input",
            lambda: _tp_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._attention(attn_in, positions, rotary, cache)
        residual = hidden
        mlp_in = self._profile_block(
            "norm.post_attention",
            lambda: _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._mlp(mlp_in)
        return hidden

    def forward_prefill_fast(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
        next_norm_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden
        attention = self._attention_from_hidden(hidden, positions, rotary, cache, attn_in=attn_in)
        return self._post_attention_forward(
            attention, residual, next_norm_weight,
        )

    def _post_attention_forward(
        self,
        attention: Tensor,
        residual: Tensor,
        next_norm_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        compiled = getattr(self, "_compiled_post_attn", None)
        if compiled is not None:
            return compiled(attention, residual, next_norm_weight)
        return self._post_attention_forward_impl(attention, residual, next_norm_weight)

    def _post_attention_forward_impl(
        self,
        attention: Tensor,
        residual: Tensor,
        next_norm_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention, residual,
            self.post_attention_layernorm_weight, self.config.rms_norm_eps,
        )
        projected = self._mlp_project_eager(mlp_in)
        _all_reduce(projected)
        if next_norm_weight is None:
            return hidden + projected, None
        return _tp_decode_add_rms_norm(projected, hidden, next_norm_weight, self.config.rms_norm_eps)

    def append_prefill_cache(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
    ) -> None:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        if attn_in is None:
            _, k, v = self._profile_block(
                "cache_only_prefill.input_norm_qkv_rotary",
                lambda: self._input_norm_qkv_rotary(hidden, batch, tokens, head_dim, rotary),
            )
        else:
            _, k, v = self._profile_block(
                "cache_only_prefill.qkv_rotary",
                lambda: self._qkv_rotary(attn_in, batch, tokens, head_dim, rotary),
            )
        self._profile_block("cache_only_prefill.cache_append", lambda: cache.append(k, v))

    def _attention(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        if self.profile_seconds is None or self.profile_counts is None:
            q, k, v = self._qkv_rotary(hidden, batch, tokens, head_dim, rotary)
            enable_gqa = self.local_attention_heads != self.local_key_value_heads
            if cache is not None:
                append_and_attend = getattr(cache, "append_and_attend", None)
                if callable(append_and_attend):
                    out = append_and_attend(q, k, v, positions, enable_gqa=enable_gqa)
                    out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
                    return self._attention_o_project_reduce(out)
                k, v = cache.append(k, v)
            out = self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa)
            out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
            return self._attention_o_project_reduce(out)

        q, k, v = self._profile_block(
            "attention.qkv",
            lambda: self._qkv(hidden, batch, tokens, head_dim),
        )
        q, k = self._profile_block("attention.rotary", lambda: _apply_rotary_cached(q, k, rotary))
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        if cache is not None:
            append_and_attend = getattr(cache, "append_and_attend", None)
            if callable(append_and_attend):
                out = self._profile_block(
                    "attention.paged_append_attention",
                    lambda: append_and_attend(q, k, v, positions, enable_gqa=enable_gqa),
                )
                out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
                projected = self._profile_block("attention.o_proj", lambda: F.linear(out, self.o_proj_weight))
                self._profile_block("attention.all_reduce", lambda: _all_reduce(projected))
                return projected
            k, v = self._profile_block("attention.cache_append", lambda: cache.append(k, v))

        out = self._profile_block(
            "attention.sdp",
            lambda: self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa),
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        projected = self._profile_block("attention.o_proj", lambda: F.linear(out, self.o_proj_weight))
        self._profile_block("attention.all_reduce", lambda: _all_reduce(projected))
        return projected

    def _attention_from_hidden(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
        *,
        attn_in: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        if attn_in is None:
            q, k, v = self._profile_block(
                "fast_prefill.attention.input_norm_qkv_rotary",
                lambda: self._input_norm_qkv_rotary(hidden, batch, tokens, head_dim, rotary),
            )
        else:
            q, k, v = self._profile_block(
                "fast_prefill.attention.qkv_rotary",
                lambda: self._qkv_rotary(attn_in, batch, tokens, head_dim, rotary),
            )
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        if cache is not None:
            append_and_attend = getattr(cache, "append_and_attend", None)
            if callable(append_and_attend):
                out = self._profile_block(
                    "fast_prefill.attention.paged_append_attention",
                    lambda: append_and_attend(q, k, v, positions, enable_gqa=enable_gqa),
                )
                out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
                return self._profile_block(
                    "fast_prefill.attention.o_project_reduce",
                    lambda: self._attention_o_project_reduce(out),
                )
            k, v = self._profile_block("fast_prefill.attention.cache_append", lambda: cache.append(k, v))
        out = self._profile_block(
            "fast_prefill.attention.sdp",
            lambda: self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa),
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._profile_block(
            "fast_prefill.attention.o_project_reduce",
            lambda: self._attention_o_project_reduce(out),
        )

    def forward_decode_static(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_position: Tensor,
        cache_positions: Tensor | None,
        row_indices: Tensor | None,
        attention_length: Tensor,
        attention_block_size: int | None,
        next_norm_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_decode_static(
            hidden,
            attn_in,
            rotary,
            cache,
            cache_position,
            cache_positions,
            row_indices,
            attention_length,
            attention_block_size,
        )
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention,
            residual,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        residual = hidden
        projected = self._mlp_project_decode_reduce(mlp_in)
        return _tp_decode_add_rms_norm(projected, residual, next_norm_weight, self.config.rms_norm_eps)

    def forward_decode_ragged(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_positions: Tensor,
        row_indices: Tensor | None,
        next_norm_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_decode_ragged(
            hidden,
            attn_in,
            rotary,
            cache,
            cache_positions,
            row_indices,
        )
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention,
            residual,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        residual = hidden
        projected = self._mlp_project_decode_reduce(mlp_in)
        return _tp_decode_add_rms_norm(projected, residual, next_norm_weight, self.config.rms_norm_eps)

    def _attention_decode_static(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_position: Tensor,
        cache_positions: Tensor | None,
        row_indices: Tensor | None,
        attention_length: Tensor,
        attention_block_size: int | None,
    ) -> Tensor:
        # Paged caches (cache_backend=paged) have no dense `keys`/`values` tensors --
        # the static decode body below assumes the dense [rows, 2, max_seq, ...] layout
        # and crashes with AttributeError('...no attribute keys'). The online batcher's
        # eager decode for paged reaches here; route it to the paged-aware ragged decode
        # (triton append_and_attend_ragged). cache_positions/row_indices are required for
        # the paged path and are always provided in the indexed-rows (paged) case.
        if callable(getattr(cache, "append_and_attend_ragged", None)) and cache_positions is not None:
            return self._attention_decode_ragged(hidden, attn_in, rotary, cache, cache_positions, row_indices)
        from torchinferno.kernels.triton_ops import (
            triton_apply_rotary_append_kv_decode,
            triton_apply_rotary_append_kv_ragged_decode,
            triton_append_kv_cache,
            triton_dense_gqa_decode_attention,
            triton_grouped_gqa_decode_attention,
        )

        batch, tokens, _ = hidden.shape
        storage = cache._contiguous_key_value_storage(batch)
        indexed_rows = storage is None
        if indexed_rows:
            if cache_positions is None or row_indices is None:
                raise ValueError("static decode graph requires cache positions and row indices for sparse rows")
            cache_keys = cache.keys
            cache_values = cache.values
        else:
            cache_keys, cache_values = storage
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        if _tp_flag("TORCHINFERNO_TRITON_DECODE_ROTARY_APPEND"):
            q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
            if indexed_rows:
                q = triton_apply_rotary_append_kv_ragged_decode(
                    q,
                    k,
                    v,
                    cache_keys,
                    cache_values,
                    cache_positions,
                    rotary[0],
                    rotary[1],
                    row_indices,
                )
            else:
                q = triton_apply_rotary_append_kv_decode(
                    q,
                    k,
                    v,
                    cache_keys,
                    cache_values,
                    cache_position,
                    rotary[0],
                    rotary[1],
                )
        else:
            q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
            if indexed_rows:
                q, k = _apply_rotary_ragged(q, k, rotary)
                _append_ragged_kv_cache(cache, k, v, cache_positions, row_indices)
            else:
                q, k = _apply_rotary_cached(q, k, rotary)
                triton_append_kv_cache(k, v, cache_keys, cache_values, cache_position)
        attention_keys = cache_keys
        attention_values = cache_values
        if attention_block_size is not None and attention_block_size < cache_keys.size(2):
            attention_keys = cache_keys[:, :, :attention_block_size, :]
            attention_values = cache_values[:, :, :attention_block_size, :]
        if (
            self.local_attention_heads > self.local_key_value_heads
            and _tp_flag("TORCHINFERNO_TRITON_GROUPED_DECODE_ATTENTION")
        ):
            out = triton_grouped_gqa_decode_attention(
                q,
                attention_keys,
                attention_values,
                attention_length,
                row_indices=row_indices if indexed_rows else None,
            )
        elif indexed_rows:
            out = _ragged_scaled_dot_product_attention(
                q,
                attention_keys,
                attention_values,
                attention_length.expand(batch),
                row_indices=row_indices,
                enable_gqa=False,
            )
        else:
            out = triton_dense_gqa_decode_attention(q, attention_keys, attention_values, seq_len=attention_length)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)

    def _attention_decode_ragged(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_positions: Tensor,
        row_indices: Tensor | None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        if tokens != 1:
            raise ValueError("ragged decode expects exactly one token")
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        append_and_attend_ragged = getattr(cache, "append_and_attend_ragged", None)
        if callable(append_and_attend_ragged):
            q, k = _apply_rotary_ragged(q, k, rotary)
            out = append_and_attend_ragged(
                q,
                k,
                v,
                cache_positions,
                row_indices,
                enable_gqa=enable_gqa,
            )
            out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
            return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)
        if (
            q.is_cuda
            and k.is_cuda
            and v.is_cuda
            and _tp_flag("TORCHINFERNO_TRITON_RAGGED_DECODE_ROTARY_APPEND")
        ):
            try:
                from torchinferno.kernels.triton_ops import triton_apply_rotary_append_kv_ragged_decode

                q = triton_apply_rotary_append_kv_ragged_decode(
                    q,
                    k,
                    v,
                    cache.keys,
                    cache.values,
                    cache_positions,
                    rotary[0],
                    rotary[1],
                    row_indices,
                )
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.ragged_decode_rotary_append", exc)
                q, k = _apply_rotary_ragged(q, k, rotary)
                _append_ragged_kv_cache(cache, k, v, cache_positions, row_indices)
        else:
            q, k = _apply_rotary_ragged(q, k, rotary)
            _append_ragged_kv_cache(cache, k, v, cache_positions, row_indices)
        attention_lengths = cache_positions + 1
        if row_indices is None:
            attention_keys = cache.keys[:batch]
            attention_values = cache.values[:batch]
        else:
            attention_keys = cache.keys
            attention_values = cache.values
        out = _ragged_scaled_dot_product_attention(
            q,
            attention_keys,
            attention_values,
            attention_lengths,
            row_indices=row_indices,
            enable_gqa=enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)

    def forward_prefill_ragged(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor | None,
        next_norm_weight: Tensor,
        context_len: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_prefill_ragged(
            hidden,
            attn_in,
            rotary,
            cache,
            start_positions,
            write_positions,
            row_indices,
            context_len,
        )
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention,
            residual,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        residual = hidden
        projected = self._mlp_project_decode_reduce(mlp_in)
        return _tp_decode_add_rms_norm(projected, residual, next_norm_weight, self.config.rms_norm_eps)

    def _attention_prefill_ragged(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor | None,
        context_len: int | None = None,
    ) -> Tensor:
        # Multi-token ragged prefill of a suffix into scattered cache rows. Mirror
        # of _attention_decode_ragged but for T query tokens with per-token rotary
        # and offset-causal attention (flash via context_len). Dense backend only.
        batch, tokens, _ = hidden.shape
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        q, k = _apply_rotary_ragged_prefill(q, k, rotary)
        _append_ragged_kv_prefill(cache, k, v, write_positions, row_indices)
        out = _ragged_prefill_scaled_dot_product_attention(
            q,
            cache.keys,
            cache.values,
            start_positions,
            suffix_tokens=tokens,
            row_indices=row_indices,
            enable_gqa=enable_gqa,
            context_len=context_len,
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)

    def forward_flashinfer(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: FlashInferLayerKVCache,
        write_positions: Tensor,
        flashinfer_wrapper: object,
        next_norm_weight: Tensor,
        *,
        row_indices: Tensor | None = None,
        q_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        batch, tokens, _ = hidden.shape
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
        q, k = _apply_rotary_ragged_prefill(q, k, rotary)
        _append_ragged_kv_prefill(cache, k, v, write_positions, row_indices)
        if hasattr(cache, 'paged_kv'):
            paged_kv = cache.paged_kv
        else:
            _fi_buf = getattr(cache, '_fi_paged_kv', None)
            if _fi_buf is None:
                _fi_buf = torch.empty(
                    cache.keys.size(0), 2, cache.keys.size(2),
                    cache.keys.size(1), cache.keys.size(3),
                    device=cache.keys.device, dtype=cache.keys.dtype,
                )
                cache._fi_paged_kv = _fi_buf
            _fi_buf[:, 0].copy_(cache.keys.transpose(1, 2))
            _fi_buf[:, 1].copy_(cache.values.transpose(1, 2))
            paged_kv = _fi_buf
        q_permuted = q.permute(0, 2, 1, 3)
        if q_lens is not None and not (q_lens == tokens).all():
            valid_mask = torch.arange(tokens, device=q.device).unsqueeze(0) < q_lens.unsqueeze(1)
            q_packed = q_permuted[valid_mask]
            out_packed = flashinfer_wrapper.run(q_packed, paged_kv)
            out = torch.zeros(batch, tokens, self.local_hidden_size, device=q.device, dtype=q.dtype)
            out[valid_mask] = out_packed.view(-1, self.local_hidden_size)
        else:
            q_packed = q_permuted.reshape(-1, self.local_attention_heads, self.config.head_dim)
            out_packed = flashinfer_wrapper.run(q_packed, paged_kv)
            out = out_packed.view(batch, tokens, self.local_hidden_size)
        attention = self._decode_linear_all_reduce(
            out, self.o_proj_weight, "attention", self.o_proj_weight_decode
        )
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention, residual,
            self.post_attention_layernorm_weight, self.config.rms_norm_eps,
        )
        residual = hidden
        projected = self._mlp_project_decode_reduce(mlp_in)
        return _tp_decode_add_rms_norm(projected, residual, next_norm_weight, self.config.rms_norm_eps)

    def _marlin_proj(self, hidden: Tensor, key: str, weight: Tensor) -> Tensor | None:
        # Generic COLUMN-parallel (output-sharded, NO allreduce) projection via
        # Marlin int4. Lazily quantizes `weight` ([N,K]) to int4 on first call (eager,
        # BEFORE any decode CUDA-graph capture), caches per `key` on self, then runs
        # marlin_int4_mm. hidden is bf16; marlin scales are bf16 (quantize default)
        # so NO fp16 conversion. Returns [..., N] or None (caller falls back to bf16).
        # Default-on for decode-sized M: gate_up int4 saves decode wall time on
        # short-row serving paths, while structural graph-vs-eager tests pin it
        # off because int4 != fp32/bf16 exactness on tiny fixtures. Disable with
        # TORCHINFERNO_MARLIN_INT4_DECODE=0 when comparing pure bf16 paths.
        # marlin int4 is a WEIGHT-ONLY-quant kernel (int4 dequant + bf16 compute): it
        # WINS at SMALL M (decode, weight-read/memory-bound: 1.52x at M=48, ~2ms/step)
        # but LOSES at LARGE M (prefill, compute-bound -- bf16 tensor cores run
        # full-throughput, dequant is pure overhead; measured real 70B: marlin-prefill
        # REGRESSED few_shot TTFT 1080->1404ms, and MARLIN_INT4_DECODE=1 without the
        # M-gate slowed multi_turn-shaped wall 22.9->25.8s via the shared prefill GEMM
        # path _mlp_project_decode_reduce). So it MUST be M-CONDITIONAL: engage only at
        # small M (decode); skip large M (prefill) -> bf16. Decode M = rows (<=~128);
        # prefill M = rows*prompt (thousands), so MAX_M=256 cleanly separates them.
        # gate_up int4 is greedy-EXACT vs bf16 (5/5 short + 4/4 long-ctx paged). The
        # large-M prefill GEMM lever is separately FP8 _scaled_mm (no infra yet).
        if not _tp_flag("TORCHINFERNO_MARLIN_INT4_DECODE", True):
            return None
        if not hidden.is_cuda:  # marlin_gemm is CUDA-only; CPU falls back to bf16
            return None
        if hidden.numel() // hidden.size(-1) > _tp_int("TORCHINFERNO_MARLIN_INT4_MAX_M", 256, minimum=1):
            return None  # large-M (prefill): marlin loses -> stay bf16
        if getattr(self, f"_marlin_{key}_failed", False):
            return None
        if getattr(self, f"_marlin_{key}_q", None) is None:
            try:
                from torchinferno.kernels import marlin as _marlin
                n, k = int(weight.size(0)), int(weight.size(1))
                if not _marlin.load_marlin_ops() or not _marlin.marlin_supports_shape(n, k):
                    setattr(self, f"_marlin_{key}_failed", True)
                    return None
                q, s = _marlin.quantize_to_marlin_int4(weight.t().contiguous(), 128)
                setattr(self, f"_marlin_{key}_q", q)
                setattr(self, f"_marlin_{key}_s", s)
                setattr(self, f"_marlin_{key}_ws", _marlin.make_workspace(n, weight.device))
                setattr(self, f"_marlin_{key}_n", n)
                setattr(self, f"_marlin_{key}_k", k)
            except Exception as exc:
                warn_optional_failure(f"llama3_tensor_parallel.marlin_{key}", exc)
                setattr(self, f"_marlin_{key}_failed", True)
                return None
        from torchinferno.kernels import marlin as _marlin
        n = getattr(self, f"_marlin_{key}_n")
        k = getattr(self, f"_marlin_{key}_k")
        x2d = hidden.reshape(-1, k).contiguous()
        out = _marlin.marlin_int4_mm(
            x2d, getattr(self, f"_marlin_{key}_q"), getattr(self, f"_marlin_{key}_s"),
            getattr(self, f"_marlin_{key}_ws"), n, k,
        )
        return out.reshape(*hidden.shape[:-1], n)

    def _fp8_proj(self, hidden: Tensor, key: str, weight: Tensor) -> Tensor | None:
        # FP8 e4m3 W8A8 GEMM for COMPUTE-bound PREFILL (LARGE M). The complement of
        # _marlin_proj: marlin (int4, weight-read-bound) wins SMALL-M decode; fp8 (tensor
        # cores) wins LARGE-M prefill. M-gated to M > FP8_PREFILL_MIN_M (256) so it ONLY
        # touches prefill -- decode (small M) stays bf16/marlin (fp8 there is slower AND
        # fp8 decode is lossy). FP8 prefill is greedy-EXACT vs bf16
        # (validate_fp8_prefill_correctness.py); the fused-quant kernel gives 1.4-2x on
        # the big GEMMs (bench_fp8_prefill.py). Lazily tensorwise-quantizes the weight to
        # fp8 (one-time, in EAGER context only -- guarded vs graph capture so the alloc
        # never lands inside a CUDA graph; callers run fp8 prefill eager). Returns [..,N]
        # or None (caller falls back to bf16). DEFAULT OFF -- REVERTED after a measured
        # benchmark REGRESSION: cron run 20260608_121124 (built 0c65c8b, fp8 on) showed
        # few_shot TTFT 288.9->515.4ms (+78%), E2E 384->600, tput 3.0->2.4; other cells
        # within noise. CAUSE: fp8 adds ~44ms/forward of LAUNCH-BOUND quant overhead
        # (abs+amax fixed ~235us x 160 calls); at few_shot's HIGH-CONCURRENCY SMALL
        # prefills that overhead EXCEEDS the GEMM savings -> net slower. My local A/B that
        # justified default-on used 400-tok prompts (larger M, where fp8 wins) and MISSED
        # few_shot's small-prefill regime. The cron OVERRIDES the local A/B. To re-enable:
        # fuse the absmax into the rms_norm/swiglu epilogue (kill the quant overhead) AND
        # raise the M-gate so fp8 fires only where GEMM savings >> overhead, validated on
        # a live cron. Revert=this flag default False (= TORCHINFERNO_FP8_PREFILL unset).
        if not _tp_flag("TORCHINFERNO_FP8_PREFILL", False):
            return None
        if not hidden.is_cuda:
            return None
        m = hidden.numel() // hidden.size(-1)
        if m <= _tp_int("TORCHINFERNO_FP8_PREFILL_MIN_M", 256, minimum=1):
            return None  # small-M (decode): fp8 loses + is lossy -> bf16/marlin
        if getattr(self, f"_fp8_{key}_failed", False):
            return None
        if getattr(self, f"_fp8_{key}_wq", None) is None:
            if _cuda_stream_is_capturing(hidden.device):
                return None  # never quantize (alloc) inside a graph capture; bf16 this call
            try:
                from torchinferno.kernels import fp8 as _fp8
                if not _fp8.fp8_available():
                    setattr(self, f"_fp8_{key}_failed", True)
                    return None
                wq, sb = _fp8.quantize_weight_fp8(weight)
                setattr(self, f"_fp8_{key}_wq", wq)
                setattr(self, f"_fp8_{key}_sb", sb)
            except Exception as exc:
                warn_optional_failure(f"llama3_tensor_parallel.fp8_{key}", exc)
                setattr(self, f"_fp8_{key}_failed", True)
                return None
        from torchinferno.kernels import fp8 as _fp8
        return _fp8.fp8_prefill_linear(
            hidden, getattr(self, f"_fp8_{key}_wq"), getattr(self, f"_fp8_{key}_sb")
        )

    def _mlp_project_decode_reduce(self, hidden: Tensor) -> Tensor:
        # This path serves BOTH decode (small M) and paged PREFILL (large M, via
        # forward_prefill_paged). marlin wins small-M; fp8 wins large-M (M-gates are
        # complementary, so at most one fires).
        gu = self._marlin_proj(hidden, "gu", self.gate_up_proj_weight)
        if gu is None:
            gu = self._fp8_proj(hidden, "gu", self.gate_up_proj_weight)
        if gu is not None:
            gate, up = gu.split((self.local_intermediate_size, self.local_intermediate_size), dim=-1)
        else:
            gate, up = _decode_linear(hidden, self.gate_up_proj_weight, self.gate_up_proj_weight_decode).split(
                (self.local_intermediate_size, self.local_intermediate_size),
                dim=-1,
            )
        activated = _tp_decode_swiglu(gate, up)
        # down_proj is the OTHER big MLP GEMM (N=hidden 8192, K=local_intermediate);
        # at decode M it is weight-read/memory-bound where marlin int4 wins the GEMM
        # (~1.44x, bench_marlin_int4). BUT default-OFF: unlike gate_up (greedy-EXACT vs
        # bf16) its output lands straight in the residual, so RTN-int4 error accumulates
        # across 80 layers and FLIPS the greedy argmax from the FIRST decode token
        # (validate_marlin_down: down=0 -> 128009..., down=1 -> 27... both coherent but
        # divergent) -- too lossy for the tight 98-100% bench correctness bar. And its
        # ~1ms TPOT saving cannot flip a cell anyway (long_output/multi_turn TPOT gaps
        # are 7-9ms). Wiring kept behind the flag for future calibrated (GPTQ/AWQ) quant.
        down_key = "down" if _tp_flag("TORCHINFERNO_MARLIN_INT4_DOWN", False) else None
        return self._decode_linear_all_reduce(
            activated, self.down_proj_weight, "mlp", self.down_proj_weight_decode,
            marlin_key=down_key, fp8_key="down",
        )

    def _mlp_project_prefill_reduce(self, hidden: Tensor) -> Tensor | None:
        if not _should_use_symm_mem_prefill_all_reduce(hidden, self.down_proj_weight, self.world_size):
            return None
        activated = self._profile_block(
            "fast_prefill.mlp_prefill.gate_up_activation",
            lambda: self._prefill_gate_up_activation(hidden),
        )
        return self._prefill_linear_all_reduce(activated, self.down_proj_weight, "mlp-prefill", fp8_key="down")

    def _prefill_gate_up_activation(self, hidden: Tensor) -> Tensor:
        # FP8 prefill (large-M, compute-bound) runs EAGER: the fused-quant fp8 GEMM win
        # (1.4-2x) exceeds the activation-graph's launch-overhead savings, and running
        # eager keeps the lazy weight-quant out of any graph capture. Falls through to
        # the bf16 graph/eager path when fp8 is off or the M-gate declines.
        gu = self._fp8_proj(hidden, "gu", self.gate_up_proj_weight)
        if gu is not None:
            gate, up = gu.split((self.local_intermediate_size, self.local_intermediate_size), dim=-1)
            return _tp_swiglu(gate, up)
        if (
            self.world_size > 1
            and _should_use_prefill_gate_up_activation_graph(hidden)
            and not self._prefill_gate_up_activation_graph_failed
        ):
            try:
                return self._run_prefill_gate_up_activation_graph(hidden)
            except Exception:
                self._prefill_gate_up_activation_graph_failed = True
        gate, up = F.linear(hidden, self.gate_up_proj_weight).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        return _tp_swiglu(gate, up)

    def _run_prefill_gate_up_activation_graph(self, hidden: Tensor) -> Tensor:
        key = tuple(hidden.shape)
        captured = self._prefill_gate_up_activation_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._prefill_gate_up_activation_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._prefill_gate_up_activation_eager(static_input)
            captured = _StaticPrefillActivationGraphCall(graph=graph, static_input=static_input, output=output)
            self._prefill_gate_up_activation_graphs[key] = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _prefill_gate_up_activation_eager(self, hidden: Tensor) -> Tensor:
        gate, up = F.linear(hidden, self.gate_up_proj_weight).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        return _tp_swiglu(gate, up)

    def _decode_linear_all_reduce(
        self,
        hidden: Tensor,
        weight: Tensor,
        buffer_name: str,
        weight_t: Tensor | None = None,
        marlin_key: str | None = None,
        fp8_key: str | None = None,
    ) -> Tensor:
        # marlin_key/fp8_key set => try a quantized GEMM for this row-parallel proj
        # (down_proj). Both kernels are reduce-agnostic; we all-reduce the output after.
        # marlin (int4) wins small-M decode; fp8 wins large-M prefill (M-gates are
        # complementary). The op returns its own tensor, so for the symm-mem fast path we
        # copy it into the symm buffer (small [M,N] copy) then multimem-all-reduce.
        marlin_out = self._marlin_proj(hidden, marlin_key, weight) if marlin_key is not None else None
        if marlin_out is None and fp8_key is not None:
            marlin_out = self._fp8_proj(hidden, fp8_key, weight)
        use_sm = _should_use_symm_mem_all_reduce(hidden, weight, self.world_size)
        sm_name = buffer_name
        if not use_sm and _should_use_symm_mem_prefill_all_reduce(hidden, weight, self.world_size):
            # Eager prefill: bound distinct prefill shapes so per-shape ~0.5GB
            # symm-mem buffers cannot churn into OOM; new shapes past the cap fall
            # back to NCCL. few_shot's shape repeats, so it stays cached.
            sm_name = f"{buffer_name}-pf"
            cap = _tp_int("TORCHINFERNO_SYMM_MEM_PREFILL_MAX_BUFFERS", 6, minimum=1)
            shape_key = (sm_name, tuple(hidden.shape[:-1]), int(weight.size(0)))
            if shape_key in _SYMM_PREFILL_SHAPES or len(_SYMM_PREFILL_SHAPES) < cap:
                use_sm = True
        if use_sm and not self._symm_reduce_failed:
            try:
                expected_shape = (*hidden.shape[:-1], weight.size(0))
                buffer, group_name = self._symm_reduce_buffer(sm_name, hidden, expected_shape)
                if sm_name != buffer_name:
                    _SYMM_PREFILL_SHAPES.add((sm_name, tuple(hidden.shape[:-1]), int(weight.size(0))))
                hidden_2d = hidden.reshape(-1, hidden.size(-1))
                output_2d = buffer.reshape(-1, weight.size(0))
                if marlin_out is not None:
                    output_2d.copy_(marlin_out.reshape(-1, weight.size(0)))
                elif weight_t is not None:
                    torch.mm(hidden_2d, weight_t, out=output_2d)
                else:
                    torch.mm(hidden_2d, weight.t(), out=output_2d)
                torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
                return buffer
            except Exception:
                self._symm_reduce_failed = True
                _disable_symm_reduce()
        if marlin_out is not None:
            _all_reduce(marlin_out)
            return marlin_out
        projected = _decode_linear(hidden, weight, weight_t)
        _all_reduce(projected)
        return projected

    def _prefill_linear_all_reduce(
        self, hidden: Tensor, weight: Tensor, buffer_name: str, fp8_key: str | None = None
    ) -> Tensor | None:
        if not _should_use_symm_mem_prefill_all_reduce(hidden, weight, self.world_size):
            return None
        # fp8_key set => large-M prefill down_proj via fp8 (reduce-agnostic GEMM); copy
        # its output into the symm buffer then multimem-all-reduce. None on the small-M
        # gate -> bf16 mm into the buffer.
        fp8_out = self._fp8_proj(hidden, fp8_key, weight) if fp8_key is not None else None
        try:
            expected_shape = (*hidden.shape[:-1], weight.size(0))
            buffer, group_name = self._symm_reduce_buffer(buffer_name, hidden, expected_shape)
            if fp8_out is not None:
                buffer.reshape(-1, weight.size(0)).copy_(fp8_out.reshape(-1, weight.size(0)))
            else:
                self._profile_block(
                    f"fast_prefill.{buffer_name}.mm",
                    lambda: torch.mm(hidden.reshape(-1, hidden.size(-1)), weight.t(), out=buffer.reshape(-1, weight.size(0))),
                )
            self._profile_block(
                f"fast_prefill.{buffer_name}.all_reduce",
                lambda: torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name),
            )
            return buffer
        except Exception:
            _disable_symm_reduce()
            return None

    def _symm_reduce_buffer(self, name: str, hidden: Tensor, expected_shape: tuple[int, ...]) -> tuple[Tensor, str]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("symmetric-memory allreduce requires an initialized process group")
        group_name = dist.group.WORLD.group_name
        device_index = hidden.device.index if hidden.device.index is not None else torch.cuda.current_device()
        key = (group_name, device_index, name, str(hidden.dtype), expected_shape)
        buffer = _SYMM_REDUCE_BUFFERS.get(key)
        if buffer is None:
            import torch.distributed._symmetric_memory as symm_mem

            buffer = symm_mem.empty(expected_shape, device=hidden.device, dtype=hidden.dtype)
            symm_mem.rendezvous(buffer, group_name)
            _SYMM_REDUCE_BUFFERS[key] = buffer
        if key not in _SYMM_REDUCE_PROBED:
            self._probe_symm_reduce_buffer(key, buffer, group_name)
        return buffer, group_name

    def _probe_symm_reduce_buffer(
        self,
        key: tuple[str, int, str, str, tuple[int, ...]],
        buffer: Tensor,
        group_name: str,
    ) -> None:
        buffer.zero_()
        torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        torch.cuda.synchronize(buffer.device)
        _SYMM_REDUCE_PROBED.add(key)

    def _mlp(self, hidden: Tensor) -> Tensor:
        if self.profile_seconds is None or self.profile_counts is None:
            projected = self._mlp_project(hidden)
            _all_reduce(projected)
            return projected

        gate, up = self._profile_block(
            "mlp.gate_up",
            lambda: F.linear(hidden, self.gate_up_proj_weight).split(
                (self.local_intermediate_size, self.local_intermediate_size),
                dim=-1,
            ),
        )
        activated = self._profile_block("mlp.activation", lambda: _tp_swiglu(gate, up))
        projected = self._profile_block("mlp.down", lambda: F.linear(activated, self.down_proj_weight))
        self._profile_block("mlp.all_reduce", lambda: _all_reduce(projected))
        return projected

    def _mlp_project(self, hidden: Tensor) -> Tensor:
        if self.world_size > 1 and _should_use_mlp_project_graph(hidden) and not self._mlp_project_graph_failed:
            try:
                return self._run_mlp_project_graph(hidden)
            except Exception:
                self._mlp_project_graph_failed = True
        return self._mlp_project_eager(hidden)

    def _mlp_project_eager(self, hidden: Tensor) -> Tensor:
        gate, up = _decode_linear(hidden, self.gate_up_proj_weight, self.gate_up_proj_weight_decode).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        activated = _tp_swiglu(gate, up)
        return _decode_linear(activated, self.down_proj_weight, self.down_proj_weight_decode)

    def _run_mlp_project_graph(self, hidden: Tensor) -> Tensor:
        captured = self._mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._mlp_project_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._mlp_project_eager(static_input)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _post_attention_mlp_project(self, hidden: Tensor) -> Tensor:
        mlp_in = _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        reduced = self._mlp_project_prefill_reduce(mlp_in)
        if reduced is not None:
            return reduced
        if _should_graph_all_reduce() and self.world_size > 1 and _should_use_mlp_project_graph(hidden):
            try:
                return self._run_post_mlp_project_reduce_graph(hidden)
            except Exception:
                self._post_mlp_project_graph_failed = True
        if self.world_size > 1 and _should_use_mlp_project_graph(hidden) and not self._post_mlp_project_graph_failed:
            try:
                projected = self._run_post_mlp_project_graph(hidden)
                _all_reduce(projected)
                return projected
            except Exception:
                self._post_mlp_project_graph_failed = True
        projected = self._mlp_project_eager(mlp_in)
        _all_reduce(projected)
        return projected

    def _run_post_mlp_project_reduce_graph(self, hidden: Tensor) -> Tensor:
        captured = self._post_mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                projected = self._post_mlp_project_eager(static_input)
                _all_reduce(projected)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._post_mlp_project_eager(static_input)
                _all_reduce(output)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._post_mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _post_mlp_project_eager(self, hidden: Tensor) -> Tensor:
        mlp_in = _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        return self._mlp_project_eager(mlp_in)

    def _run_post_mlp_project_graph(self, hidden: Tensor) -> Tensor:
        captured = self._post_mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._post_mlp_project_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._post_mlp_project_eager(static_input)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._post_mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _attention_o_project(self, hidden: Tensor) -> Tensor:
        if self.world_size > 1 and _should_use_attention_o_graph(hidden) and not self._attention_o_graph_failed:
            try:
                return self._run_attention_o_graph(hidden)
            except Exception:
                self._attention_o_graph_failed = True
        return _decode_linear(hidden, self.o_proj_weight, self.o_proj_weight_decode)

    def _run_attention_o_graph(self, hidden: Tensor) -> Tensor:
        captured = self._attention_o_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._attention_o_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _attention_o_project_reduce(self, hidden: Tensor) -> Tensor:
        reduced = self._prefill_linear_all_reduce(hidden, self.o_proj_weight, "attention-prefill")
        if reduced is not None:
            return reduced
        if _should_graph_all_reduce() and self.world_size > 1 and _should_use_attention_o_graph(hidden):
            try:
                return self._run_attention_o_reduce_graph(hidden)
            except Exception:
                self._attention_o_graph_failed = True
        projected = self._attention_o_project(hidden)
        _all_reduce(projected)
        return projected

    def _run_attention_o_reduce_graph(self, hidden: Tensor) -> Tensor:
        captured = self._attention_o_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                projected = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
                _all_reduce(projected)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
                _all_reduce(output)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._attention_o_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _qkv(self, hidden: Tensor, batch: int, tokens: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        # qkv stays bf16: it is a SMALL GEMM (N=local_qkv ~1280) where marlin's fixed
        # overhead loses (measured: marlin qkv 0.43x, and end-to-end it REGRESSED the
        # decode step). marlin only wins the big GEMMs (gate_up, down).
        qkv = _decode_linear(hidden, self.qkv_proj_weight, self.qkv_proj_weight_decode)
        q, k, v = qkv.split(
            (self.local_hidden_size, self.local_key_value_size, self.local_key_value_size),
            dim=-1,
        )
        q = q.view(
            batch,
            tokens,
            self.local_attention_heads,
            head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        return q, k, v

    def _qkv_rotary(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.world_size > 1 and _should_use_qkv_rotary_graph(hidden) and not self._qkv_rotary_graph_failed:
            try:
                return self._run_qkv_rotary_graph(hidden, batch, tokens, head_dim, rotary)
            except Exception:
                self._qkv_rotary_graph_failed = True
        return self._qkv_rotary_eager(hidden, batch, tokens, head_dim, rotary)

    def _qkv_rotary_eager(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = self._qkv(hidden, batch, tokens, head_dim)
        q, k = _apply_rotary_cached(q, k, rotary)
        return q, k, v

    def _run_qkv_rotary_graph(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        cos, sin = rotary
        key = (tuple(hidden.shape), tuple(cos.shape), tuple(sin.shape))
        captured = self._qkv_rotary_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_cos = torch.empty_like(cos)
            static_sin = torch.empty_like(sin)
            static_input.copy_(hidden)
            static_cos.copy_(cos)
            static_sin.copy_(sin)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, k, v = self._qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            captured = _StaticQKVRotaryGraphCall(
                graph=graph,
                static_input=static_input,
                static_cos=static_cos,
                static_sin=static_sin,
                q=q,
                k=k,
                v=v,
            )
            self._qkv_rotary_graphs[key] = captured
            captured.graph.replay()
            return captured.q, captured.k, captured.v
        captured.static_input.copy_(hidden)
        captured.static_cos.copy_(cos)
        captured.static_sin.copy_(sin)
        captured.graph.replay()
        return captured.q, captured.k, captured.v

    def _input_norm_qkv_rotary(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.world_size > 1 and _should_use_qkv_rotary_graph(hidden) and not self._input_qkv_rotary_graph_failed:
            try:
                return self._run_input_qkv_rotary_graph(hidden, batch, tokens, head_dim, rotary)
            except Exception:
                self._input_qkv_rotary_graph_failed = True
        return self._input_qkv_rotary_eager(hidden, batch, tokens, head_dim, rotary)

    def _input_qkv_rotary_eager(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn_in = _tp_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        return self._qkv_rotary_eager(attn_in, batch, tokens, head_dim, rotary)

    def _run_input_qkv_rotary_graph(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        cos, sin = rotary
        key = (tuple(hidden.shape), tuple(cos.shape), tuple(sin.shape))
        captured = self._input_qkv_rotary_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_cos = torch.empty_like(cos)
            static_sin = torch.empty_like(sin)
            static_input.copy_(hidden)
            static_cos.copy_(cos)
            static_sin.copy_(sin)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._input_qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, k, v = self._input_qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            captured = _StaticQKVRotaryGraphCall(
                graph=graph,
                static_input=static_input,
                static_cos=static_cos,
                static_sin=static_sin,
                q=q,
                k=k,
                v=v,
            )
            self._input_qkv_rotary_graphs[key] = captured
            captured.graph.replay()
            return captured.q, captured.k, captured.v
        captured.static_input.copy_(hidden)
        captured.static_cos.copy_(cos)
        captured.static_sin.copy_(sin)
        captured.graph.replay()
        return captured.q, captured.k, captured.v

    @staticmethod
    def _scaled_dot_product(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        positions: Tensor,
        device: torch.device,
        enable_gqa: bool,
    ) -> Tensor:
        if q.size(-2) == 1:
            if q.is_cuda and k.size(-2) <= 2048 and _tp_flag("TORCHINFERNO_TRITON_DECODE_ATTENTION"):
                try:
                    from torchinferno.kernels.triton_ops import triton_dense_gqa_decode_attention

                    return triton_dense_gqa_decode_attention(q, k, v)
                except Exception as exc:
                    warn_optional_failure("llama3_tensor_parallel.decode_attention", exc)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        if k.size(-2) == q.size(-2):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=enable_gqa,
            )
        try:
            from torch.nn.attention.bias import causal_lower_right

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_lower_right(q.size(-2), k.size(-2)),
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.causal_lower_right", exc)
        key_positions = torch.arange(k.size(-2), device=device)
        allowed = key_positions[None, :] <= positions[:, None]
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=enable_gqa,
        )

    def _profile_block(self, name: str, fn):
        if self.profile_seconds is None or self.profile_counts is None:
            return fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.profile_seconds[name] = self.profile_seconds.get(name, 0.0) + (time.perf_counter() - start)
        self.profile_counts[name] = self.profile_counts.get(name, 0) + 1
        return result


class Llama3TensorParallelForCausalLM:
    """Tensor-parallel Llama3 inference path launched with torchrun."""

    provenance_variant = "llama3:tp-v0"

    def __init__(
        self,
        config: Llama3Config,
        *,
        embed_tokens_weight: Tensor,
        norm_weight: Tensor,
        lm_head_weight: Tensor,
        layers: list[_Llama3TensorParallelLayer],
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        checkpoint: str | Path,
    ) -> None:
        if config.vocab_size % world_size != 0:
            raise ValueError("vocab_size must be divisible by tensor parallel world size")
        self.config = config
        self.embed_tokens_weight = embed_tokens_weight
        self.norm_weight = norm_weight
        self.lm_head_weight = lm_head_weight
        self.lm_head_weight_decode = _maybe_decode_weight_t(lm_head_weight)
        self.layers = layers
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.devices = (device,)
        self.dtype = dtype
        self.checkpoint = Path(checkpoint)
        self.embed_device = device
        self.output_device = device
        self.local_vocab_size = config.vocab_size // world_size
        self.vocab_start = rank * self.local_vocab_size
        self.inv_freq = _build_inv_freq(config, device)
        self.rotary_cos_cache, self.rotary_sin_cache = _build_llama_rotary_cache(
            config.max_position_embeddings,
            self.inv_freq,
            device=device,
            dtype=dtype,
        )
        self.profile_seconds: dict[str, float] = {}
        self.profile_counts: dict[str, int] = {}
        self.training = False
        self._prefill_graphs: dict[tuple[object, ...], _StaticPrefillGraphCall] = {}
        self._prefill_logits_graphs: dict[tuple[object, ...], _StaticPrefillLogitsGraphCall] = {}
        self._prefill_selected_logits_graphs: dict[
            tuple[object, ...],
            _StaticPrefillSelectedLogitsGraphCall,
        ] = {}
        self._prefill_graph_failed = False
        self._prefill_logits_graph_failed = False
        self._prefill_selected_logits_graph_failed = False
        self._decode_graphs: dict[tuple[int, int, int, int], _StaticDecodeGraphCall] = {}
        self._decode_logits_graphs: dict[tuple[int, int, int, int], _StaticDecodeLogitsGraphCall] = {}
        self._ragged_decode_graphs: dict[
            tuple[int, int, int, bool, int],
            _StaticRaggedDecodeGraphCall,
        ] = {}
        self._ragged_decode_logits_graphs: dict[
            tuple[int, int, int, bool, int],
            _StaticRaggedDecodeLogitsGraphCall,
        ] = {}
        self._decode_graph_failed = False
        self._decode_logits_graph_failed = False
        self._ragged_decode_graph_failed = False
        self._ragged_decode_logits_graph_failed = False
        self._ragged_prefill_logits_graphs: dict[
            tuple[int, int, int, int, bool, int],
            _StaticRaggedPrefillLogitsGraphCall,
        ] = {}
        self._ragged_prefill_logits_graph_failed = False
        self._temperature_gumbel_generators: dict[str, torch.Generator] = {}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = LLAMA3_70B_REPO_ID,
        *,
        dtype: torch.dtype | str | None = None,
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "Llama3TensorParallelForCausalLM":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        _init_distributed_if_needed()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        root = _resolve_tensor_parallel_checkpoint(
            checkpoint,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
        )
        config = Llama3Config.from_dict(json.loads((root / HF_CONFIG_NAME).read_text()))
        torch_dtype = _resolve_dtype(dtype, root)
        loader = _CheckpointTensorLoader(root)
        embed_tokens_weight = loader.get_tensor("model.embed_tokens.weight", device=device, dtype=torch_dtype)
        norm_weight = loader.get_tensor("model.norm.weight", device=device, dtype=torch_dtype)
        lm_head_weight = loader.get_tensor_shard(
            "lm_head.weight",
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=torch_dtype,
        )

        layers: list[_Llama3TensorParallelLayer] = []
        for layer_id in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}."
            q_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.q_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            k_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.k_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            v_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.v_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            gate_proj_weight = loader.get_tensor_shard(
                prefix + "mlp.gate_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            up_proj_weight = loader.get_tensor_shard(
                prefix + "mlp.up_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            weights = {
                "input_layernorm.weight": loader.get_tensor(
                    prefix + "input_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                ),
                "post_attention_layernorm.weight": loader.get_tensor(
                    prefix + "post_attention_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                ),
                "self_attn.qkv_proj.weight": torch.cat(
                    (q_proj_weight, k_proj_weight, v_proj_weight),
                    dim=0,
                ).contiguous(),
                "self_attn.o_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.o_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "mlp.gate_up_proj.weight": torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous(),
                "mlp.down_proj.weight": loader.get_tensor_shard(
                    prefix + "mlp.down_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
            }
            layers.append(
                _Llama3TensorParallelLayer(
                    config,
                    layer_id,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    weights=weights,
                )
            )
        model = cls(
            config,
            embed_tokens_weight=embed_tokens_weight,
            norm_weight=norm_weight,
            lm_head_weight=lm_head_weight,
            layers=layers,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            dtype=torch_dtype,
            checkpoint=root,
        )
        model.load_report = Llama3TensorParallelLoadReport(
            checkpoint=str(root),
            dtype=str(torch_dtype).replace("torch.", ""),
            device=str(device),
            rank=rank,
            world_size=world_size,
        )
        _barrier()
        return model

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def eval(self) -> "Llama3TensorParallelForCausalLM":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "Llama3TensorParallelForCausalLM":
        self.training = mode
        return self

    def enable_profile(self) -> None:
        self.profile_seconds = {}
        self.profile_counts = {}
        for layer in self.layers:
            layer.profile_seconds = self.profile_seconds
            layer.profile_counts = self.profile_counts

    def disable_profile(self) -> None:
        for layer in self.layers:
            layer.profile_seconds = None
            layer.profile_counts = None

    def profile_summary(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "seconds": dict(sorted(self.profile_seconds.items())),
            "counts": dict(sorted(self.profile_counts.items())),
        }

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        cache_backend: str = "dense",
        page_size: int = 16,
        device: torch.device | None = None,
    ) -> Llama3TensorParallelCache:
        if cache_backend not in {"dense", "paged", "flashinfer"}:
            raise ValueError("cache_backend must be 'dense', 'paged', or 'flashinfer'")
        if device is not None and torch.device(device) != self.device:
            raise ValueError("Llama3 tensor-parallel cache must be allocated on the model device")
        local_kv_heads = self.config.num_key_value_heads // self.world_size
        if cache_backend == "flashinfer":
            layer_cls = FlashInferLayerKVCache
        elif cache_backend == "paged":
            layer_cls = PagedLlama3TensorParallelLayerKVCache
        else:
            layer_cls = Llama3TensorParallelLayerKVCache
        def _make_layer_cache():
            if layer_cls is PagedLlama3TensorParallelLayerKVCache:
                return layer_cls(
                    batch_size, max_seq_len, local_kv_heads, self.config.head_dim,
                    page_size=page_size, device=self.device, dtype=self.dtype,
                )
            return layer_cls(
                batch_size, max_seq_len, local_kv_heads, self.config.head_dim,
                device=self.device, dtype=self.dtype,
            )
        return Llama3TensorParallelCache(
            [_make_layer_cache() for _ in self.layers],
            cache_backend=cache_backend,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Llama3TensorParallelCache | None = None,
        use_cache: bool = True,
        return_last_logits_only: bool = False,
        return_sharded_logits: bool = False,
        logit_positions: Tensor | None = None,
    ) -> tuple[Tensor, Llama3TensorParallelCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = self.allocate_cache(batch, tokens)
        past_len = active_cache.seq_len if active_cache is not None else 0
        positions = torch.arange(past_len, past_len + tokens, device=self.device)
        rotary = self._rotary_cache(past_len, tokens)
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        profile_fast_prefill = _tp_flag("TORCHINFERNO_PROFILE_FAST_PREFILL", False)
        if tokens > 1 and (profile_fast_prefill or all(layer.profile_seconds is None for layer in self.layers)):
            use_compiled = False
            cache_ready = (
                getattr(active_cache, "_compiled_prefill_ready", False)
                or getattr(getattr(active_cache, "_parent_cache", None), "_compiled_prefill_ready", False)
            )
            if active_cache is not None and cache_ready:
                self._ensure_compiled_prefill()
                compiled = getattr(self, "_compiled_forward_prefill", None)
                if compiled is not None:
                    try:
                        hidden = compiled(hidden, positions, rotary, active_cache)
                        use_compiled = True
                    except Exception:
                        self._compiled_forward_prefill = None
            if not use_compiled:
                attn_in: Tensor | None = None
                for layer_id, layer in enumerate(self.layers):
                    layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
                    next_norm_weight = (
                        self.layers[layer_id + 1].input_layernorm_weight
                        if layer_id + 1 < len(self.layers)
                        else None
                    )
                    hidden, attn_in = layer.forward_prefill_fast(
                        hidden,
                        attn_in,
                        positions,
                        rotary,
                        layer_cache,
                        next_norm_weight,
                    )
        else:
            for layer_id, layer in enumerate(self.layers):
                layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
                hidden = layer.forward(hidden, positions, rotary, layer_cache)
        if logit_positions is not None:
            if logit_positions.ndim != 1 or logit_positions.numel() != batch:
                raise ValueError("logit_positions must have shape [batch]")
            gather_positions = logit_positions.to(self.device, non_blocking=True).view(batch, 1, 1)
            gather_positions = gather_positions.expand(-1, 1, hidden.size(-1))
            hidden = torch.gather(hidden, 1, gather_positions)
        elif return_last_logits_only:
            hidden = hidden[:, -1:, :]
        hidden = _tp_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head_weight)
        if return_sharded_logits:
            return logits, active_cache
        logits = self._gather_logits(logits)
        return logits, active_cache

    def _ensure_compiled_prefill(self) -> None:
        if hasattr(self, "_compiled_forward_prefill"):
            return
        if not env_flag("TORCHINFERNO_COMPILED_PREFILL", False):
            self._compiled_forward_prefill = None
            return
        try:
            self._compiled_forward_prefill = torch.compile(
                self._forward_prefill_body,
                dynamic=True,
                fullgraph=False,
            )
        except Exception:
            self._compiled_forward_prefill = None

    def _forward_prefill_body(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: "Llama3TensorParallelCache",
    ) -> Tensor:
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            layer_cache = cache.layers[layer_id]
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else None
            )
            hidden, attn_in = layer.forward_prefill_fast(
                hidden, attn_in, positions, rotary, layer_cache, next_norm_weight,
            )
        return hidden

    @torch.inference_mode()
    def prefill_cache_only(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> Llama3TensorParallelCache:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, tokens = input_ids.shape
        if tokens <= 0:
            return cache
        past_len = cache.seq_len
        positions = torch.arange(past_len, past_len + tokens, device=self.device)
        rotary = self._rotary_cache(past_len, tokens)
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        if tokens > 1 and all(layer.profile_seconds is None for layer in self.layers):
            attn_in: Tensor | None = None
            last_layer_id = len(self.layers) - 1
            for layer_id, layer in enumerate(self.layers):
                if layer_id == last_layer_id:
                    layer.append_prefill_cache(hidden, attn_in, rotary, cache.layers[layer_id])
                    break
                next_norm_weight = (
                    self.layers[layer_id + 1].input_layernorm_weight
                    if layer_id + 1 < len(self.layers)
                    else None
                )
                hidden, attn_in = layer.forward_prefill_fast(
                    hidden,
                    attn_in,
                    positions,
                    rotary,
                    cache.layers[layer_id],
                    next_norm_weight,
                )
        else:
            last_layer_id = len(self.layers) - 1
            for layer_id, layer in enumerate(self.layers):
                if layer_id == last_layer_id:
                    layer.append_prefill_cache(hidden, None, rotary, cache.layers[layer_id])
                    break
                hidden = layer.forward(hidden, positions, rotary, cache.layers[layer_id])
        return cache

    def try_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if getattr(cache, "_skip_capture_sync", False):
            capture_on_miss = False
        if (
            not _tp_env_set("TORCHINFERNO_CUDAGRAPH_PREFILL")
            and int(getattr(self.config, "hidden_size", 0)) < 1024
        ):
            return None
        if self._prefill_graph_failed or not _should_use_prefill_graph(input_ids, cache, temperature):
            return None
        try:
            return self._run_prefill_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.prefill_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} prefill_graph_failed={exc!r}", flush=True)
            self._prefill_graph_failed = True
            return None

    def try_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if getattr(cache, "_skip_capture_sync", False):
            capture_on_miss = False
        if self._prefill_logits_graph_failed or not _should_use_prefill_logits_graph(input_ids, cache):
            return None
        try:
            return self._run_prefill_logits_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.prefill_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} prefill_logits_graph_failed={exc!r}", flush=True)
            self._prefill_logits_graph_failed = True
            return None

    def try_prefill_selected_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        logit_positions: Tensor,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if self._prefill_selected_logits_graph_failed or not _should_use_prefill_logits_graph(input_ids, cache):
            return None
        if logit_positions.ndim != 1 or logit_positions.numel() != input_ids.size(0):
            return None
        try:
            return self._run_prefill_selected_logits_graph(
                input_ids,
                cache,
                logit_positions=logit_positions,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.prefill_selected_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} prefill_selected_logits_graph_failed={exc!r}", flush=True)
            self._prefill_selected_logits_graph_failed = True
            return None

    def _run_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        if end_seq_len > cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        key = (
            *_prefill_graph_cache_key(cache, input_ids.size(0)),
            initial_seq_len,
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            tuple(input_ids.shape),
        )
        captured = self._prefill_graphs.get(key)
        needs_capture = (
            captured is None
            or not _same_prefill_graph_cache(captured.cache, cache, input_ids.size(0))
            or captured.prompt_tokens != input_ids.size(1)
            or captured.initial_seq_len != initial_seq_len
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
        )
        if needs_capture and not capture_on_miss:
            return None
        if capture_on_miss:
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            try:
                captured = self._capture_prefill_graph(input_ids, cache)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if key not in self._prefill_graphs and len(self._prefill_graphs) >= max_graphs:
                self._prefill_graphs.clear()
            self._prefill_graphs[key] = captured
        else:
            captured.static_input_ids.copy_(input_ids)
            self._set_cache_seq_len(cache, captured.initial_seq_len)
            captured.graph.replay()
            self._set_cache_seq_len(cache, end_seq_len)
        return captured.output_token

    def _capture_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> _StaticPrefillGraphCall:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        static_input_ids = torch.empty_like(input_ids)
        static_input_ids.copy_(input_ids)
        captured = _StaticPrefillGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            output_token=torch.empty((input_ids.size(0),), device=self.device, dtype=torch.long),
            cache=cache,
            prompt_tokens=input_ids.size(1),
            initial_seq_len=initial_seq_len,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._set_cache_seq_len(cache, initial_seq_len)
            logits = self._forward_prefill_static(captured.static_input_ids, cache)
            self._sample_next_token(logits[:, -1, :], 0.0)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._set_cache_seq_len(cache, initial_seq_len)
        with torch.cuda.graph(captured.graph):
            logits = self._forward_prefill_static(captured.static_input_ids, cache)
            captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
        captured.graph.replay()
        self._set_cache_seq_len(cache, end_seq_len)
        return captured

    def _run_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        initial_seq_len = cache.seq_len
        prompt_tokens = input_ids.size(1)
        logit_positions = torch.full(
            (input_ids.size(0),),
            prompt_tokens - 1,
            device=self.device,
            dtype=torch.long,
        )
        bucket = _prefill_bucket_size(prompt_tokens) if _tp_flag(
            "TORCHINFERNO_CUDAGRAPH_PREFILL_BUCKETING", True
        ) else None
        if bucket is not None and bucket > prompt_tokens:
            pad_len = bucket - prompt_tokens
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        end_seq_len = initial_seq_len + input_ids.size(1)
        if end_seq_len > cache.layers[0].max_seq_len:
            if bucket is not None:
                input_ids = input_ids[:, :prompt_tokens]
                end_seq_len = initial_seq_len + prompt_tokens
                bucket = None
            if end_seq_len > cache.layers[0].max_seq_len:
                raise ValueError("KV cache capacity exceeded")
        key = (
            *_prefill_graph_cache_key(cache, input_ids.size(0)),
            initial_seq_len,
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            tuple(input_ids.shape),
        )
        captured = self._prefill_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or not _same_prefill_graph_cache(captured.cache, cache, input_ids.size(0))
            or captured.prompt_tokens != input_ids.size(1)
            or captured.initial_seq_len != initial_seq_len
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or captured.static_logit_positions.shape != logit_positions.shape
        )
        if needs_capture and not capture_on_miss:
            return None
        if capture_on_miss:
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            try:
                captured = self._capture_prefill_logits_graph(input_ids, cache, logit_positions)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if key not in self._prefill_logits_graphs and len(self._prefill_logits_graphs) >= max_graphs:
                self._prefill_logits_graphs.clear()
            self._prefill_logits_graphs[key] = captured
        else:
            captured.static_input_ids.copy_(input_ids)
            captured.static_logit_positions.copy_(logit_positions)
            self._set_cache_seq_len(cache, captured.initial_seq_len)
            captured.graph.replay()
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
        return captured.output_logits

    def _capture_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        logit_positions: Tensor,
    ) -> _StaticPrefillLogitsGraphCall:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        static_input_ids = torch.empty_like(input_ids)
        static_input_ids.copy_(input_ids)
        static_logit_positions = torch.empty_like(logit_positions, device=self.device)
        static_logit_positions.copy_(logit_positions)
        captured = _StaticPrefillLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_logit_positions=static_logit_positions,
            output_logits=torch.empty(
                (input_ids.size(0), 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            prompt_tokens=input_ids.size(1),
            initial_seq_len=initial_seq_len,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._set_cache_seq_len(cache, initial_seq_len)
            self._forward_prefill_selected_static(
                captured.static_input_ids,
                cache,
                captured.static_logit_positions,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._set_cache_seq_len(cache, initial_seq_len)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_prefill_selected_static(
                captured.static_input_ids,
                cache,
                captured.static_logit_positions,
            )
        captured.graph.replay()
        self._set_cache_seq_len(cache, end_seq_len)
        return captured

    def _run_prefill_selected_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        logit_positions: Tensor,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        initial_seq_len = cache.seq_len
        prompt_tokens = input_ids.size(1)
        bucket = _prefill_bucket_size(prompt_tokens) if _tp_flag(
            "TORCHINFERNO_CUDAGRAPH_PREFILL_BUCKETING", True
        ) else None
        if bucket is not None and bucket > prompt_tokens:
            pad_len = bucket - prompt_tokens
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        end_seq_len = initial_seq_len + input_ids.size(1)
        if end_seq_len > cache.layers[0].max_seq_len:
            if bucket is not None:
                input_ids = input_ids[:, :prompt_tokens]
                end_seq_len = initial_seq_len + prompt_tokens
                bucket = None
            if end_seq_len > cache.layers[0].max_seq_len:
                raise ValueError("KV cache capacity exceeded")
        key = (
            *_prefill_graph_cache_key(cache, input_ids.size(0)),
            initial_seq_len,
            cache.layers[0].max_seq_len,
            tuple(input_ids.shape),
            tuple(logit_positions.shape),
        )
        captured = self._prefill_selected_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or not _same_prefill_graph_cache(captured.cache, cache, input_ids.size(0))
            or captured.prompt_tokens != input_ids.size(1)
            or captured.initial_seq_len != initial_seq_len
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or captured.static_logit_positions.shape != logit_positions.shape
        )
        if needs_capture and not capture_on_miss:
            return None
        if capture_on_miss:
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            try:
                captured = self._capture_prefill_selected_logits_graph(input_ids, cache, logit_positions)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if (
                key not in self._prefill_selected_logits_graphs
                and len(self._prefill_selected_logits_graphs) >= max_graphs
            ):
                self._prefill_selected_logits_graphs.clear()
            self._prefill_selected_logits_graphs[key] = captured
        else:
            captured.static_input_ids.copy_(input_ids)
            captured.static_logit_positions.copy_(logit_positions.to(self.device, non_blocking=True))
            self._set_cache_seq_len(cache, captured.initial_seq_len)
            captured.graph.replay()
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
        return captured.output_logits

    def _capture_prefill_selected_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        logit_positions: Tensor,
    ) -> _StaticPrefillSelectedLogitsGraphCall:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        static_input_ids = torch.empty_like(input_ids)
        static_input_ids.copy_(input_ids)
        static_logit_positions = torch.empty_like(logit_positions, device=self.device)
        static_logit_positions.copy_(logit_positions.to(self.device, non_blocking=True))
        captured = _StaticPrefillSelectedLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_logit_positions=static_logit_positions,
            output_logits=torch.empty(
                (input_ids.size(0), 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            prompt_tokens=input_ids.size(1),
            initial_seq_len=initial_seq_len,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._set_cache_seq_len(cache, initial_seq_len)
            self._forward_prefill_selected_static(
                captured.static_input_ids,
                cache,
                captured.static_logit_positions,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._set_cache_seq_len(cache, initial_seq_len)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_prefill_selected_static(
                captured.static_input_ids,
                cache,
                captured.static_logit_positions,
            )
        captured.graph.replay()
        self._set_cache_seq_len(cache, end_seq_len)
        return captured

    def _forward_prefill_static(self, input_ids: Tensor, cache: Llama3TensorParallelCache) -> Tensor:
        logits, _ = self.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        return logits

    def _forward_prefill_selected_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        logit_positions: Tensor,
    ) -> Tensor:
        logits, _ = self.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=False,
            return_sharded_logits=True,
            logit_positions=logit_positions,
        )
        return logits

    def try_decode_one_token_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if getattr(cache, "_block_decode_graph_captures", False):
            capture_on_miss = False
        if self._decode_graph_failed or not _should_use_decode_step_graph(input_ids, cache, temperature):
            return None
        try:
            return self._run_decode_step_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_step_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} decode_step_graph_failed={exc!r}", flush=True)
            self._decode_graph_failed = True
            return None

    def try_decode_one_token_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if self._decode_logits_graph_failed or not _should_use_decode_step_logits_graph(input_ids, cache):
            return None
        try:
            return self._run_decode_step_logits_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_step_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} decode_step_logits_graph_failed={exc!r}", flush=True)
            self._decode_logits_graph_failed = True
            return None

    def try_decode_ragged_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if self._ragged_decode_logits_graph_failed or not _should_use_ragged_decode_logits_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
        ):
            return None
        try:
            return self._run_ragged_decode_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} ragged_decode_logits_graph_failed={exc!r}", flush=True)
            self._ragged_decode_logits_graph_failed = True
            return None

    def try_decode_ragged_token_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if getattr(cache, "_block_decode_graph_captures", False):
            capture_on_miss = False
        if self._ragged_decode_graph_failed or not _should_use_ragged_decode_token_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            temperature,
        ):
            return None
        try:
            return self._run_ragged_decode_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_token_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} ragged_decode_token_graph_failed={exc!r}", flush=True)
            self._ragged_decode_graph_failed = True
            return None

    def release_decode_graphs_for_cache(self, cache: Llama3TensorParallelCache) -> None:
        cache_ids = {id(cache), _cache_graph_root_id(cache)}
        for graph_map in (
            self._prefill_graphs,
            self._prefill_logits_graphs,
            getattr(self, "_prefill_selected_logits_graphs", {}),
            self._decode_graphs,
            self._decode_logits_graphs,
            getattr(self, "_ragged_decode_graphs", {}),
            self._ragged_decode_logits_graphs,
        ):
            for key, captured in list(graph_map.items()):
                if key[0] in cache_ids or getattr(captured, "cache", None) is cache:
                    graph_map.pop(key, None)

    def _run_decode_step_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if cache.seq_len >= cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        attention_block_size = _decode_attention_block_size(cache.seq_len + 1, cache.layers[0].max_seq_len)
        symm_reduce_key = _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self))
        key = (id(cache), input_ids.size(0), attention_block_size, symm_reduce_key)
        captured = self._decode_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.attention_block_size != attention_block_size
            or captured.static_input_ids.shape != input_ids.shape
        )
        if not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            if not capture_on_miss:
                return None
            captured = self._capture_decode_step_graph(input_ids, cache, attention_block_size)
        else:
            self._copy_decode_graph_inputs(captured, input_ids, cache)
            captured.graph.replay()
        self._advance_decode_graph_cache(cache)
        return captured.output_token

    def _capture_decode_step_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        attention_block_size: int,
    ) -> _StaticDecodeGraphCall:
        batch = input_ids.size(0)
        indexed_rows = not _static_decode_cache_rows_are_contiguous(cache, batch)
        static_row_indices = _static_decode_row_indices(cache, batch) if indexed_rows else None
        if indexed_rows and static_row_indices is None:
            raise ValueError("static decode graph requires row-indexed cache views for sparse rows")
        static_input_ids = torch.empty_like(input_ids)
        static_cache_position = torch.empty((), device=self.device, dtype=torch.int64)
        static_cache_positions = torch.empty((batch,), device=self.device, dtype=torch.int64) if indexed_rows else None
        static_attention_length = torch.empty((), device=self.device, dtype=torch.int64)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        rotary_rows = batch if indexed_rows else 1
        static_rotary_cos = torch.empty((rotary_rows, rotary_cache_dim), device=self.device, dtype=self.dtype)
        static_rotary_sin = torch.empty((rotary_rows, rotary_cache_dim), device=self.device, dtype=self.dtype)
        captured = _StaticDecodeGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_cache_position=static_cache_position,
            static_cache_positions=static_cache_positions,
            static_row_indices=static_row_indices,
            static_attention_length=static_attention_length,
            static_rotary_cos=static_rotary_cos,
            static_rotary_sin=static_rotary_sin,
            output_token=torch.empty((batch,), device=self.device, dtype=torch.long),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            attention_block_size=attention_block_size,
        )
        self._copy_decode_graph_inputs(captured, input_ids, cache)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_cache_positions,
                captured.static_row_indices,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
            self._sample_next_token(logits[:, -1, :], 0.0)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_cache_positions,
                captured.static_row_indices,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
            captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
        captured.graph.replay()
        symm_reduce_key = _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self))
        key = (id(cache), input_ids.size(0), attention_block_size, symm_reduce_key)
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._decode_graphs and len(self._decode_graphs) >= max_graphs:
            self._decode_graphs.clear()
        self._decode_graphs[key] = captured
        return captured

    def _run_decode_step_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if cache.seq_len >= cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        attention_block_size = _decode_attention_block_size(cache.seq_len + 1, cache.layers[0].max_seq_len)
        symm_reduce_key = _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self))
        key = (id(cache), input_ids.size(0), attention_block_size, symm_reduce_key)
        captured = self._decode_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.attention_block_size != attention_block_size
            or captured.static_input_ids.shape != input_ids.shape
        )
        needs_capture = _capture_needed_on_any_rank(needs_capture, self.device) if not getattr(cache, "_skip_capture_sync", False) else needs_capture
        if needs_capture:
            if not capture_on_miss:
                return None
            captured = self._capture_decode_step_logits_graph(input_ids, cache, attention_block_size)
        else:
            self._copy_decode_logits_graph_inputs(captured, input_ids, cache)
            captured.graph.replay()
        self._advance_decode_graph_cache(cache)
        return captured.output_logits

    def _run_ragged_decode_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if not cache.layers:
            raise ValueError("ragged decode requires a non-empty KV cache")
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_decode_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
        )
        needs_capture = _capture_needed_on_any_rank(needs_capture, self.device) if not getattr(cache, "_skip_capture_sync", False) else needs_capture
        if needs_capture:
            if not capture_on_miss:
                return None
            captured = self._capture_ragged_decode_graph(input_ids, cache, seq_lens, row_indices)
        else:
            self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
            captured.graph.replay()
        _advance_paged_ragged_decode_cache_lengths(
            cache,
            batch=input_ids.size(0),
            cache_positions=captured.static_cache_positions,
            row_indices=captured.static_row_indices,
        )
        return captured.output_token

    def _run_ragged_decode_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if not cache.layers:
            raise ValueError("ragged decode requires a non-empty KV cache")
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_decode_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
        )
        needs_capture = _capture_needed_on_any_rank(needs_capture, self.device) if not getattr(cache, "_skip_capture_sync", False) else needs_capture
        if needs_capture:
            if not capture_on_miss:
                return None
            captured = self._capture_ragged_decode_logits_graph(input_ids, cache, seq_lens, row_indices)
        else:
            self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
            captured.graph.replay()
        captured_cache_positions = getattr(captured, "static_cache_positions", None)
        if isinstance(captured_cache_positions, Tensor):
            _advance_paged_ragged_decode_cache_lengths(
                cache,
                batch=input_ids.size(0),
                cache_positions=captured_cache_positions,
                row_indices=captured.static_row_indices,
            )
        return captured.output_logits

    def _capture_ragged_decode_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        seq_lens: Tensor,
        row_indices: Tensor | None,
    ) -> _StaticRaggedDecodeGraphCall:
        batch = input_ids.size(0)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_row_indices = torch.empty_like(row_indices) if row_indices is not None else None
        captured = _StaticRaggedDecodeGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_cache_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_row_indices=static_row_indices,
            static_rotary_cos=torch.empty((batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_rotary_sin=torch.empty((batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            output_token=torch.empty((batch,), device=self.device, dtype=torch.long),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        _set_paged_ragged_decode_graph_active(cache, True)
        try:
            with torch.cuda.stream(stream):
                logits = self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    (captured.static_rotary_cos, captured.static_rotary_sin),
                )
                self._sample_next_token(logits[:, -1, :], 0.0)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(captured.graph):
                logits = self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    (captured.static_rotary_cos, captured.static_rotary_sin),
                )
                captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
            captured.graph.replay()
        finally:
            _set_paged_ragged_decode_graph_active(cache, False)
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._ragged_decode_graphs and len(self._ragged_decode_graphs) >= max_graphs:
            self._ragged_decode_graphs.clear()
        self._ragged_decode_graphs[key] = captured
        return captured

    def _capture_ragged_decode_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        seq_lens: Tensor,
        row_indices: Tensor | None,
    ) -> _StaticRaggedDecodeLogitsGraphCall:
        batch = input_ids.size(0)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_row_indices = torch.empty_like(row_indices) if row_indices is not None else None
        captured = _StaticRaggedDecodeLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_cache_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_row_indices=static_row_indices,
            static_rotary_cos=torch.empty((batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_rotary_sin=torch.empty((batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            output_logits=torch.empty(
                (batch, 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        _set_paged_ragged_decode_graph_active(cache, True)
        try:
            with torch.cuda.stream(stream):
                self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    (captured.static_rotary_cos, captured.static_rotary_sin),
                )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(captured.graph):
                captured.output_logits = self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    (captured.static_rotary_cos, captured.static_rotary_sin),
                )
            captured.graph.replay()
        finally:
            _set_paged_ragged_decode_graph_active(cache, False)
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._ragged_decode_logits_graphs and len(self._ragged_decode_logits_graphs) >= max_graphs:
            self._ragged_decode_logits_graphs.clear()
        self._ragged_decode_logits_graphs[key] = captured
        return captured

    def _copy_ragged_decode_graph_inputs(
        self,
        captured: _StaticRaggedDecodeGraphCall | _StaticRaggedDecodeLogitsGraphCall,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor | None,
    ) -> None:
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        if row_indices is None:
            cache_positions = seq_lens[: input_ids.size(0)]
        else:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if captured.static_row_indices is None:
                raise RuntimeError("captured ragged decode graph does not accept row indices")
            captured.static_row_indices.copy_(row_indices)
            cache_positions = seq_lens.index_select(0, row_indices)
        captured.static_input_ids.copy_(input_ids)
        captured.static_cache_positions.copy_(cache_positions)
        captured.static_rotary_cos.copy_(self.rotary_cos_cache.index_select(0, cache_positions))
        captured.static_rotary_sin.copy_(self.rotary_sin_cache.index_select(0, cache_positions))
        _prepare_paged_ragged_decode_graph_state(
            captured.cache,
            batch=input_ids.size(0),
            cache_positions=captured.static_cache_positions,
            row_indices=captured.static_row_indices,
            device=self.device,
        )

    def _capture_decode_step_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        attention_block_size: int,
    ) -> _StaticDecodeLogitsGraphCall:
        batch = input_ids.size(0)
        indexed_rows = not _static_decode_cache_rows_are_contiguous(cache, batch)
        static_row_indices = _static_decode_row_indices(cache, batch) if indexed_rows else None
        if indexed_rows and static_row_indices is None:
            raise ValueError("static decode graph requires row-indexed cache views for sparse rows")
        static_input_ids = torch.empty_like(input_ids)
        static_cache_position = torch.empty((), device=self.device, dtype=torch.int64)
        static_cache_positions = torch.empty((batch,), device=self.device, dtype=torch.int64) if indexed_rows else None
        static_attention_length = torch.empty((), device=self.device, dtype=torch.int64)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        rotary_rows = batch if indexed_rows else 1
        static_rotary_cos = torch.empty((rotary_rows, rotary_cache_dim), device=self.device, dtype=self.dtype)
        static_rotary_sin = torch.empty((rotary_rows, rotary_cache_dim), device=self.device, dtype=self.dtype)
        captured = _StaticDecodeLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_cache_position=static_cache_position,
            static_cache_positions=static_cache_positions,
            static_row_indices=static_row_indices,
            static_attention_length=static_attention_length,
            static_rotary_cos=static_rotary_cos,
            static_rotary_sin=static_rotary_sin,
            output_logits=torch.empty(
                (batch, 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            attention_block_size=attention_block_size,
        )
        self._copy_decode_logits_graph_inputs(captured, input_ids, cache)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_cache_positions,
                captured.static_row_indices,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_cache_positions,
                captured.static_row_indices,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
        captured.graph.replay()
        symm_reduce_key = _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self))
        key = (id(cache), input_ids.size(0), attention_block_size, symm_reduce_key)
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._decode_logits_graphs and len(self._decode_logits_graphs) >= max_graphs:
            self._decode_logits_graphs.clear()
        self._decode_logits_graphs[key] = captured
        return captured

    def _copy_decode_graph_inputs(
        self,
        captured: _StaticDecodeGraphCall,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> None:
        position = cache.seq_len
        captured.static_input_ids.copy_(input_ids)
        captured.static_cache_position.fill_(position)
        if captured.static_cache_positions is not None:
            captured.static_cache_positions.fill_(position)
        captured.static_attention_length.fill_(position + 1)
        captured.static_rotary_cos.copy_(
            self.rotary_cos_cache[position : position + 1].expand_as(captured.static_rotary_cos)
        )
        captured.static_rotary_sin.copy_(
            self.rotary_sin_cache[position : position + 1].expand_as(captured.static_rotary_sin)
        )

    def _copy_decode_logits_graph_inputs(
        self,
        captured: _StaticDecodeLogitsGraphCall,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> None:
        position = cache.seq_len
        captured.static_input_ids.copy_(input_ids)
        captured.static_cache_position.fill_(position)
        if captured.static_cache_positions is not None:
            captured.static_cache_positions.fill_(position)
        captured.static_attention_length.fill_(position + 1)
        captured.static_rotary_cos.copy_(
            self.rotary_cos_cache[position : position + 1].expand_as(captured.static_rotary_cos)
        )
        captured.static_rotary_sin.copy_(
            self.rotary_sin_cache[position : position + 1].expand_as(captured.static_rotary_sin)
        )

    def _advance_decode_graph_cache(self, cache: Llama3TensorParallelCache) -> None:
        next_seq_len = cache.seq_len + 1
        self._set_cache_seq_len(cache, next_seq_len)

    @staticmethod
    def _set_cache_seq_len(cache: Llama3TensorParallelCache, seq_len: int) -> None:
        cache.set_seq_len(seq_len)

    def _forward_decode_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        cache_position: Tensor,
        cache_positions: Tensor | None,
        row_indices: Tensor | None,
        attention_length: Tensor,
        rotary: tuple[Tensor, Tensor],
        attention_block_size: int | None = None,
    ) -> Tensor:
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_decode_static(
                hidden,
                attn_in,
                rotary,
                cache.layers[layer_id],
                cache_position,
                cache_positions,
                row_indices,
                attention_length,
                attention_block_size,
                next_norm_weight,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    def _forward_decode_ragged_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        cache_positions: Tensor,
        row_indices: Tensor | None,
        rotary: tuple[Tensor, Tensor],
    ) -> Tensor:
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        context_set = _set_paged_ragged_decode_context(
            cache,
            batch=input_ids.size(0),
            cache_positions=cache_positions,
            row_indices=row_indices,
            device=self.device,
        )
        try:
            for layer_id, layer in enumerate(self.layers):
                next_norm_weight = (
                    self.layers[layer_id + 1].input_layernorm_weight
                    if layer_id + 1 < len(self.layers)
                    else self.norm_weight
                )
                hidden, attn_in = layer.forward_decode_ragged(
                    hidden,
                    attn_in,
                    rotary,
                    cache.layers[layer_id],
                    cache_positions,
                    row_indices,
                    next_norm_weight,
                )
        finally:
            if context_set:
                _clear_paged_ragged_decode_context(cache)
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def decode_ragged_logits(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError("ragged decode expects input_ids with shape [batch, 1]")
        if not cache.layers:
            raise ValueError("ragged decode requires a non-empty KV cache")
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        if row_indices is not None:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if row_indices.ndim != 1 or row_indices.numel() != input_ids.size(0):
                raise ValueError("row_indices must have shape [batch]")
            cache_positions = seq_lens.index_select(0, row_indices)
        else:
            cache_positions = seq_lens[: input_ids.size(0)]
            if cache_positions.numel() != input_ids.size(0):
                raise ValueError("seq_lens must cover the ragged decode batch")
        if bool(torch.any(cache_positions < 0)):
            raise ValueError("seq_lens must be non-negative")
        if bool(torch.any(cache_positions >= cache.layers[0].max_seq_len)):
            raise ValueError("KV cache capacity exceeded")

        rotary = (
            self.rotary_cos_cache.index_select(0, cache_positions),
            self.rotary_sin_cache.index_select(0, cache_positions),
        )
        hidden = F.embedding(input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        context_set = _set_paged_ragged_decode_context(
            cache,
            batch=input_ids.size(0),
            cache_positions=cache_positions,
            row_indices=row_indices,
            device=self.device,
        )
        try:
            for layer_id, layer in enumerate(self.layers):
                next_norm_weight = (
                    self.layers[layer_id + 1].input_layernorm_weight
                    if layer_id + 1 < len(self.layers)
                    else self.norm_weight
                )
                hidden, attn_in = layer.forward_decode_ragged(
                    hidden,
                    attn_in,
                    rotary,
                    cache.layers[layer_id],
                    cache_positions,
                    row_indices,
                    next_norm_weight,
                )
        finally:
            if context_set:
                _clear_paged_ragged_decode_context(cache)
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    def _forward_prefill_ragged_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
    ) -> Tensor:
        batch = input_ids.size(0)
        # Fold the prefix KV broadcast INTO the graph: copy [0:prefix_len] from
        # each source row into its active row, per layer, via advanced indexing.
        # Captured once, replayed in one launch -- removes the ~80 per-layer
        # index_copy launches/batch that the engine used to issue eagerly.
        # src_prefix_row is a copied-in device tensor so replay re-targets source
        # rows; row_indices re-targets dest rows. A single source row broadcasts
        # for common-prefix reuse, and one source per request handles full-prompt
        # reuse with equal prefix lengths.
        if src_prefix_row is not None and row_indices is not None:
            if context_len is not None:
                prefix_len = context_len - input_ids.size(1)
                if prefix_len > 0:
                    for layer in cache.layers:
                        layer.keys[row_indices, :, :prefix_len, :] = layer.keys.index_select(
                            0, src_prefix_row
                        )[:, :, :prefix_len, :]
                        layer.values[row_indices, :, :prefix_len, :] = layer.values.index_select(
                            0, src_prefix_row
                        )[:, :, :prefix_len, :]
            else:
                prefix_len = int(start_positions.max().item()) if start_positions.numel() else 0
                if prefix_len > 0:
                    source_rows = src_prefix_row
                    if source_rows.numel() == 1 and row_indices.numel() > 1:
                        source_rows = source_rows.expand(row_indices.numel())
                    mask = (
                        torch.arange(prefix_len, device=self.device)[None, :]
                        < start_positions[:, None]
                    )
                    for layer in cache.layers:
                        source_keys = layer.keys.index_select(0, source_rows)[:, :, :prefix_len, :]
                        source_values = layer.values.index_select(0, source_rows)[:, :, :prefix_len, :]
                        zero_key = torch.zeros((), dtype=source_keys.dtype, device=source_keys.device)
                        zero_value = torch.zeros((), dtype=source_values.dtype, device=source_values.device)
                        layer.keys[row_indices, :, :prefix_len, :] = torch.where(
                            mask[:, None, :, None],
                            source_keys,
                            zero_key,
                        )
                        layer.values[row_indices, :, :prefix_len, :] = torch.where(
                            mask[:, None, :, None],
                            source_values,
                            zero_value,
                        )
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_prefill_ragged(
                hidden,
                attn_in,
                rotary,
                cache.layers[layer_id],
                start_positions,
                write_positions,
                row_indices,
                next_norm_weight,
                context_len,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        gather_positions = logit_positions.view(batch, 1, 1).expand(-1, 1, attn_in.size(-1))
        gathered = torch.gather(attn_in, 1, gather_positions)
        return _decode_linear(gathered, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def forward_step_flashinfer(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        write_positions: Tensor,
        logit_positions: Tensor,
        row_indices: Tensor | None = None,
    ) -> Tensor:
        if (
            getattr(cache, "_compiled_prefill_ready", False)
            and not hasattr(self, "_compiled_step_flashinfer")
            and env_flag("TORCHINFERNO_COMPILED_UNIFIED_FORWARD", False)
        ):
            try:
                _compile_mode = "reduce-overhead" if env_flag(
                    "TORCHINFERNO_COMPILED_UNIFIED_REDUCE_OVERHEAD", True
                ) else "max-autotune"
                self._compiled_step_flashinfer = torch.compile(
                    self._forward_step_flashinfer_body,
                    mode=_compile_mode,
                    fullgraph=False,
                )
            except Exception:
                self._compiled_step_flashinfer = None
        compiled = getattr(self, "_compiled_step_flashinfer", None)
        if compiled is not None:
            return compiled(
                input_ids, cache,
                seq_lens=seq_lens, q_lens=q_lens,
                write_positions=write_positions,
                logit_positions=logit_positions,
                row_indices=row_indices,
            )
        return self._forward_step_flashinfer_body(
            input_ids, cache,
            seq_lens=seq_lens, q_lens=q_lens,
            write_positions=write_positions,
            logit_positions=logit_positions,
            row_indices=row_indices,
        )

    def _flashinfer_prefill_wrapper(self) -> object:
        import flashinfer
        workspace = getattr(self, '_flashinfer_workspace', None)
        if workspace is None:
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self._flashinfer_workspace = workspace
        wrapper = getattr(self, '_flashinfer_wrapper', None)
        if wrapper is None:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout='NHD')
            self._flashinfer_wrapper = wrapper
        return wrapper

    def _plan_flashinfer_prefill(
        self,
        wrapper: object,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        row_indices: Tensor | None,
    ) -> None:
        # Host-side scheduling. Must run OUTSIDE any CUDA graph capture/replay.
        batch = q_lens.size(0)
        max_seq = cache.layers[0].max_seq_len
        qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        qo_indptr[1:] = q_lens.to(torch.int32).cumsum(0)
        paged_kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=self.device)
        if row_indices is not None:
            paged_kv_indices = row_indices.to(dtype=torch.int32, device=self.device)
        else:
            paged_kv_indices = torch.arange(batch, dtype=torch.int32, device=self.device)
        paged_kv_last_page_len = (seq_lens + q_lens).to(torch.int32)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.layers[0].local_attention_heads,
            num_kv_heads=self.layers[0].local_key_value_heads,
            head_dim_qk=self.config.head_dim,
            page_size=max_seq,
            causal=True,
            q_data_type=self.dtype,
        )

    def _forward_step_flashinfer_compute(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        wrapper: object,
        *,
        q_lens: Tensor | None,
        write_positions: Tensor,
        logit_positions: Tensor,
        row_indices: Tensor | None = None,
    ) -> Tensor:
        # The graphable portion of the FlashInfer prefill forward: embedding,
        # the 80-layer stack (with FlashInfer attention against the already-
        # planned wrapper), and the per-request logit gather. No host-side
        # scheduling here, so this whole region can be captured into a CUDA
        # graph to eliminate per-layer kernel-launch overhead.
        #
        # q_lens=None selects the uniform (non-ragged) attention path in the
        # layer, which avoids the `(q_lens == tokens).all()` tensor->bool host
        # sync that is illegal during CUDA graph capture. The graph path always
        # pads to a uniform q bucket, so None is correct there; the eager body
        # passes real per-request q_lens for the ragged path.
        batch = input_ids.size(0)
        max_q_len = input_ids.size(1)
        flat_positions = write_positions.reshape(-1).clamp(0, self.rotary_cos_cache.size(0) - 1)
        rotary = (
            self.rotary_cos_cache.index_select(0, flat_positions).view(batch, max_q_len, -1),
            self.rotary_sin_cache.index_select(0, flat_positions).view(batch, max_q_len, -1),
        )
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_flashinfer(
                hidden, attn_in, rotary,
                cache.layers[layer_id],
                write_positions,
                wrapper,
                next_norm_weight,
                row_indices=row_indices,
                q_lens=q_lens,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        gather_positions = logit_positions.view(batch, 1, 1).expand(-1, 1, attn_in.size(-1))
        gathered = torch.gather(attn_in, 1, gather_positions)
        return _decode_linear(gathered, self.lm_head_weight, self.lm_head_weight_decode)

    def _forward_step_flashinfer_body(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        write_positions: Tensor,
        logit_positions: Tensor,
        row_indices: Tensor | None = None,
    ) -> Tensor:
        wrapper = self._flashinfer_prefill_wrapper()
        self._plan_flashinfer_prefill(
            wrapper, cache, seq_lens=seq_lens, q_lens=q_lens, row_indices=row_indices,
        )
        if not getattr(self, '_flashinfer_jit_warmed', False):
            self._flashinfer_jit_warmed = True
        # One-shot op-level profile of a batched prefill to locate the prefill
        # MFU sink (GEMMs vs FlashInfer attention vs allreduce). Gated; rank 0
        # only; fires once for batch>1 so it does not perturb steady state.
        if (
            env_flag("TORCHINFERNO_PROFILE_PREFILL_ONCE", False)
            and input_ids.size(0) > 1
            and not getattr(self, "_prefill_profiled", False)
            and getattr(self, "rank", 0) == 0
        ):
            self._prefill_profiled = True
            import sys as _pp
            from torch.profiler import profile as _tprof, ProfilerActivity as _PA
            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as _prof:
                _out = self._forward_step_flashinfer_compute(
                    input_ids, cache, wrapper,
                    q_lens=q_lens, write_positions=write_positions,
                    logit_positions=logit_positions, row_indices=row_indices,
                )
                torch.cuda.synchronize(self.device)
            print(
                f"[PREFILL_PROF] batch={input_ids.size(0)} q={input_ids.size(1)}\n"
                + _prof.key_averages().table(sort_by="cuda_time_total", row_limit=18),
                file=_pp.stderr, flush=True,
            )
            return _out
        return self._forward_step_flashinfer_compute(
            input_ids, cache, wrapper,
            q_lens=q_lens, write_positions=write_positions,
            logit_positions=logit_positions, row_indices=row_indices,
        )

    def capture_flashinfer_prefill_graph(
        self,
        cache: Llama3TensorParallelCache,
        batch: int,
        q_len: int,
    ) -> bool:
        # Capture a CUDA graph of the FlashInfer prefill compute at a fixed
        # (batch, q_len) bucket. plan() runs OUTSIDE the graph; only the
        # embedding + 80-layer stack + logit gather are captured, eliminating
        # the ~245ms of per-layer kernel-launch overhead measured in the eager
        # path. KV writes target the static row buffer, so replays can retarget
        # arbitrary cache rows by updating it before replay.
        import flashinfer

        device = self.device
        max_seq = cache.layers[0].max_seq_len
        graphs = getattr(self, "_fi_prefill_graphs", None)
        if graphs is None:
            graphs = {}
            self._fi_prefill_graphs = graphs

        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        qo_buf = torch.empty(batch + 1, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.empty(batch + 1, dtype=torch.int32, device=device)
        kv_indices_buf = torch.empty(batch, dtype=torch.int32, device=device)
        kv_lpl_buf = torch.empty(batch, dtype=torch.int32, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", use_cuda_graph=True,
            qo_indptr_buf=qo_buf, paged_kv_indptr_buf=kv_indptr_buf,
            paged_kv_indices_buf=kv_indices_buf, paged_kv_last_page_len_buf=kv_lpl_buf,
        )

        s_ids = torch.zeros(batch, q_len, dtype=torch.long, device=device)
        s_wp = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1).contiguous()
        s_ri = torch.arange(batch, dtype=torch.long, device=device)
        s_qlens = torch.full((batch,), q_len, dtype=torch.long, device=device)
        s_seq = torch.zeros(batch, dtype=torch.long, device=device)
        s_logit = torch.full((batch,), q_len - 1, dtype=torch.long, device=device)

        def plan() -> None:
            self._plan_flashinfer_prefill(
                wrapper, cache, seq_lens=s_seq, q_lens=s_qlens, row_indices=s_ri,
            )

        try:
            plan()
            self._forward_step_flashinfer_compute(
                s_ids, cache, wrapper, q_lens=None, write_positions=s_wp,
                logit_positions=s_logit, row_indices=s_ri,
            )
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                plan()
                self._forward_step_flashinfer_compute(
                    s_ids, cache, wrapper, q_lens=None, write_positions=s_wp,
                    logit_positions=s_logit, row_indices=s_ri,
                )
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            plan()
            with torch.cuda.graph(graph, stream=stream):
                s_out = self._forward_step_flashinfer_compute(
                    s_ids, cache, wrapper, q_lens=None, write_positions=s_wp,
                    logit_positions=s_logit, row_indices=s_ri,
                )
        except Exception as _cap_exc:
            if getattr(self, "rank", 0) == 0:
                import sys as _cs, traceback as _ct
                print(f"[FI_PREFILL_GRAPH] capture bs={batch} q={q_len} failed: {_cap_exc!r}", file=_cs.stderr, flush=True)
                _ct.print_exc(file=_cs.stderr)
            return False

        graphs[(batch, q_len)] = (
            graph, wrapper, s_ids, s_wp, s_ri, s_seq, s_qlens, s_logit, s_out,
        )
        return True

    def try_prefill_flashinfer_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        write_positions: Tensor,
        logit_positions: Tensor,
        row_indices: Tensor,
    ) -> Tensor | None:
        # Replay a captured prefill graph if a (batch_bucket, q_bucket) covers
        # this request batch. Requests are right-padded to the q bucket and run
        # with uniform q (the graphable attention path); per-request logits are
        # gathered at the real last-token position so padding never corrupts the
        # sampled token. Returns sharded logits [batch, 1, vocab] or None.
        graphs = getattr(self, "_fi_prefill_graphs", None)
        if not graphs:
            return None
        n = input_ids.size(0)
        real_q = int(input_ids.size(1))
        # Pick the captured (batch_bucket, q_bucket) that covers this request with
        # the LEAST padded compute (batch_bucket * q_bucket). Power-of-two batch
        # padding wastes more compute than the launch overhead it saves, so the
        # capture set is fine-grained and we choose the minimal-area fit.
        batch_bucket = None
        q_bucket = None
        best_area = None
        for (b, q) in graphs:
            if b >= n and q >= real_q:
                area = b * q
                if best_area is None or area < best_area:
                    best_area = area
                    batch_bucket = b
                    q_bucket = q
        if batch_bucket is None:
            return None
        entry = graphs.get((batch_bucket, q_bucket))
        if entry is None:
            return None
        graph, wrapper, s_ids, s_wp, s_ri, s_seq, s_qlens, s_logit, s_out = entry

        s_ids.zero_()
        s_ids[:n, :real_q].copy_(input_ids)
        s_ri[:n].copy_(row_indices.to(torch.long))
        if n < batch_bucket:
            s_ri[n:] = s_ri[0]
        s_logit[:n].copy_(logit_positions.to(torch.long))
        self._plan_flashinfer_prefill(
            wrapper, cache, seq_lens=s_seq, q_lens=s_qlens, row_indices=s_ri,
        )
        graph.replay()
        return s_out[:n]

    def forward_decode_flashinfer(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        write_positions: Tensor,
        row_indices: Tensor,
        decode_wrapper: object,
    ) -> Tensor:
        # One-shot op-level profile of a full decode step (per-kernel CUDA time:
        # GEMMs vs FlashInfer attention vs allreduce vs norms). Gated; rank 0;
        # fires once on the eager call (before graph capture) so it does not
        # perturb steady state. Locates the decode-step bottleneck definitively.
        if (
            env_flag("TORCHINFERNO_PROFILE_DECODE_ONCE", False)
            and input_ids.size(0) >= 32  # serving-representative batch, not the bs=1 warmup
            and not getattr(self, "_decode_profiled", False)
            and getattr(self, "rank", 0) == 0
        ):
            self._decode_profiled = True
            import sys as _dp
            from torch.profiler import profile as _tprof, ProfilerActivity as _PA
            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as _prof:
                _out = self._forward_decode_flashinfer_body(
                    input_ids, cache, write_positions, row_indices, decode_wrapper,
                )
                torch.cuda.synchronize(self.device)
            print(
                f"[DECODE_PROF] batch={input_ids.size(0)}\n"
                + _prof.key_averages().table(sort_by="cuda_time_total", row_limit=22),
                file=_dp.stderr, flush=True,
            )
            return _out
        return self._forward_decode_flashinfer_body(
            input_ids, cache, write_positions, row_indices, decode_wrapper,
        )

    def _forward_decode_flashinfer_body(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        write_positions: Tensor,
        row_indices: Tensor,
        decode_wrapper: object,
    ) -> Tensor:
        batch = input_ids.size(0)

        flat_positions = write_positions.reshape(-1).clamp(0, self.rotary_cos_cache.size(0) - 1)
        cos = self.rotary_cos_cache.index_select(0, flat_positions).view(batch, 1, -1)
        sin = self.rotary_sin_cache.index_select(0, flat_positions).view(batch, 1, -1)
        # Pre-expand half-dim cos/sin to full head_dim ONCE here instead of inside
        # _rotate_llama_eager every layer x {q,k}: rotary is identical across all 80
        # layers, so this hoists ~320 redundant cat()s out of the decode step
        # (profile: rope cat/neg/mul were ~2.6ms of the step). Exact -- same math.
        if cos.size(-1) * 2 == self.config.head_dim:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        rotary = (cos, sin)

        hidden = F.embedding(input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None

        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_flashinfer(
                hidden, attn_in, rotary,
                cache.layers[layer_id],
                write_positions,
                decode_wrapper,
                next_norm_weight,
                row_indices=row_indices,
            )

        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def forward_decode_paged(
        self,
        input_ids: Tensor,
        paged_cache: object,
        *,
        request_ids: list[str] | None = None,
        positions: Tensor,
        decode_wrapper: object,
        block_table: Tensor | None = None,
    ) -> Tensor:
        # TRUE paged-KV decode (WIP, feature branch): same GEMM/rope/norm flow as
        # _forward_decode_flashinfer_body, but the KV write goes to a small-page
        # pool via slot_mapping()+scatter_write() and attention reads the pool via
        # layer_kv() -- instead of the dense [rows, 2, max_seq, ...] cache. This is
        # the model-side half of the paged-KV migration (the lever for the
        # queueing-bound multi_turn/long_output/TTFT/throughput cells); the serving
        # loop owns admission-by-pages + the decode_wrapper plan (built once per
        # step from paged_cache.flashinfer_page_table(request_ids), shared by all
        # layers). input_ids: [batch, 1]; positions: [batch] absolute position of
        # each decoded token (its pages must already be reserved).
        batch = input_ids.size(0)
        head_dim = self.config.head_dim
        rms_eps = self.config.rms_norm_eps
        flat_positions = positions.reshape(-1).clamp(0, self.rotary_cos_cache.size(0) - 1)
        cos = self.rotary_cos_cache.index_select(0, flat_positions).view(batch, 1, -1)
        sin = self.rotary_sin_cache.index_select(0, flat_positions).view(batch, 1, -1)
        if cos.size(-1) * 2 == head_dim:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        rotary = (cos, sin)
        # Slot computation. The GRAPHABLE path takes a pre-built block_table tensor
        # (static buffer the serving loop fills outside the graph) and computes slots
        # ENTIRELY on-device via slots_from_block_table -- no host work inside the
        # captured region, so paged decode can hit the dense graphed ~21ms instead of
        # the eager ~146ms (scripts/bench_decode_context_scaling.py). The eager path
        # (request_ids given) builds the table inline; both match the host
        # slot_mapping exactly (tests/test_scaffolding.py).
        if block_table is not None:
            slots = type(paged_cache).slots_from_block_table(
                block_table, positions, paged_cache.page_size
            )
        elif request_ids is not None:
            slots = paged_cache.slot_mapping_device(request_ids, positions)
        else:
            raise ValueError("forward_decode_paged needs request_ids or block_table")

        hidden = F.embedding(input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            residual = hidden
            if attn_in is None:
                attn_in = _tp_decode_rms_norm(hidden, layer.input_layernorm_weight, rms_eps)
            q, k, v = layer._qkv(attn_in, batch, 1, head_dim)
            q, k = _apply_rotary_ragged_prefill(q, k, rotary)
            # paged write: k/v are [batch, kv_heads, 1, head_dim] -> NHD [batch, kv_heads, head_dim]
            paged_cache.scatter_write(
                layer_id, slots, k[:, :, 0, :].contiguous(), v[:, :, 0, :].contiguous()
            )
            q_packed = q.permute(0, 2, 1, 3).reshape(-1, layer.local_attention_heads, head_dim)
            out_packed = decode_wrapper.run(q_packed, paged_cache.layer_kv(layer_id))
            out = out_packed.view(batch, 1, layer.local_hidden_size)
            attention = layer._decode_linear_all_reduce(
                out, layer.o_proj_weight, "attention", layer.o_proj_weight_decode
            )
            hidden, mlp_in = _tp_decode_add_rms_norm(
                attention, residual, layer.post_attention_layernorm_weight, rms_eps
            )
            residual = hidden
            projected = layer._mlp_project_decode_reduce(mlp_in)
            hidden, attn_in = _tp_decode_add_rms_norm(projected, residual, next_norm_weight, rms_eps)

        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, rms_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def forward_prefill_paged(
        self,
        input_ids: Tensor,
        paged_cache: object,
        *,
        request_ids: list[str],
        prefill_wrapper: object,
        block_table: Tensor | None = None,
        start_position: int | Tensor = 0,
    ) -> Tensor:
        # TRUE paged-KV prefill (WIP, feature branch): fresh-sequence prefill of a
        # uniform [batch, T] prompt block into the small-page pool. Same per-layer
        # GEMM/rope/norm flow as the FlashInfer prefill (forward_flashinfer's
        # prefill branch), but writes all T tokens' K/V via slot_mapping()+
        # scatter_write() and attends via a paged FlashInfer PREFILL wrapper over
        # layer_kv(). Each request must be reserved with length T; prefill_wrapper
        # is pre-planned with qo_indptr (T per request), paged_cache's
        # flashinfer_page_table(request_ids), and causal=True. Returns logits for
        # every position [batch, T, vocab]. This is the prefill half that populates
        # the pool so forward_decode_paged can extend each sequence.
        batch, tokens = input_ids.shape
        head_dim = self.config.head_dim
        rms_eps = self.config.rms_norm_eps
        # start_position > 0 = SUFFIX prefill over a shared/cached prefix (COW prefix
        # reuse): these `tokens` query positions are [start, start+tokens), their KV is
        # written at those absolute slots, and rotary uses the absolute positions. The
        # caller plans prefill_wrapper with qo=suffix over the FULL paged_kv (shared
        # prefix pages + new suffix pages), seq_lens=start, causal -> attention reads
        # the shared prefix for free.
        # start_position is a SCALAR (all rows share a start -- fresh/COW-suffix prefill)
        # OR a [batch] int64 tensor of PER-REQUEST starts (batched speculative-decode
        # verify, where desynced requests sit at different positions). Per-row:
        # positions[i, j] = start[i] + j.
        if isinstance(start_position, Tensor):
            start_col = start_position.to(self.device).view(batch, 1)
            positions = start_col + torch.arange(tokens, device=self.device).view(1, tokens)
        else:
            positions = (
                torch.arange(start_position, start_position + tokens, device=self.device)
                .unsqueeze(0)
                .expand(batch, tokens)
            )
        flat = positions.reshape(-1)
        cos = self.rotary_cos_cache.index_select(0, flat).view(batch, tokens, -1)
        sin = self.rotary_sin_cache.index_select(0, flat).view(batch, tokens, -1)
        if cos.size(-1) * 2 == head_dim:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        rotary = (cos, sin)
        if block_table is not None:
            # On-device, CUDA-graph-capturable slots (no host loop): for a fresh
            # [batch, T] prefill the token at (request i, position j) writes slot
            # block_table[i, j//page_size]*page_size + j%page_size. positions is the
            # [batch, T] absolute-position grid (arange(T) per row, since prefill
            # starts at 0); flatten row-major to match the [batch*T] scatter.
            page_size = paged_cache.page_size
            page_slot = torch.div(positions, page_size, rounding_mode="floor")
            page_id = block_table.gather(1, page_slot)
            slots = (page_id * page_size + (positions - page_slot * page_size)).reshape(-1)
        else:
            ids: list[str] = []
            row_pos = positions.tolist()  # per-row absolute positions (scalar or per-request start)
            for request_id in request_ids:
                ids.extend([request_id] * tokens)
            pos = [p for row in row_pos for p in row]
            slots = paged_cache.slot_mapping(ids, pos)  # [batch*T], row-major (request, position)

        hidden = F.embedding(input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            residual = hidden
            if attn_in is None:
                attn_in = _tp_decode_rms_norm(hidden, layer.input_layernorm_weight, rms_eps)
            q, k, v = layer._qkv(attn_in, batch, tokens, head_dim)
            q, k = _apply_rotary_ragged_prefill(q, k, rotary)
            # paged write: k/v [batch, kv_heads, T, head_dim] -> NHD [batch*T, kv_heads, head_dim]
            k_nhd = k.permute(0, 2, 1, 3).reshape(
                batch * tokens, layer.local_key_value_heads, head_dim
            ).contiguous()
            v_nhd = v.permute(0, 2, 1, 3).reshape(
                batch * tokens, layer.local_key_value_heads, head_dim
            ).contiguous()
            paged_cache.scatter_write(layer_id, slots, k_nhd, v_nhd)
            q_packed = q.permute(0, 2, 1, 3).reshape(
                batch * tokens, layer.local_attention_heads, head_dim
            )
            out_packed = prefill_wrapper.run(q_packed, paged_cache.layer_kv(layer_id))
            out = out_packed.view(batch, tokens, layer.local_hidden_size)
            attention = layer._decode_linear_all_reduce(
                out, layer.o_proj_weight, "attention", layer.o_proj_weight_decode
            )
            hidden, mlp_in = _tp_decode_add_rms_norm(
                attention, residual, layer.post_attention_layernorm_weight, rms_eps
            )
            residual = hidden
            projected = layer._mlp_project_decode_reduce(mlp_in)
            hidden, attn_in = _tp_decode_add_rms_norm(projected, residual, next_norm_weight, rms_eps)

        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, rms_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def prefill_ragged_logits(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
    ) -> Tensor:
        # Eager reference for the ragged-prefill graph: prefill a [batch, suffix]
        # block of tokens into scattered cache rows with per-row prefix offsets,
        # returning one (sharded) logit row per request at logit_positions. This
        # is the oracle the CUDA graph captures and the CPU test compares against.
        if input_ids.ndim != 2:
            raise ValueError("ragged prefill expects input_ids [batch, suffix]")
        if not cache.layers:
            raise ValueError("ragged prefill requires a non-empty KV cache")
        batch, suffix = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        logit_positions = logit_positions.to(self.device, non_blocking=True)
        if logit_positions.ndim != 1 or logit_positions.numel() != batch:
            raise ValueError("logit_positions must have shape [batch]")
        if row_indices is not None:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if row_indices.ndim != 1 or row_indices.numel() != batch:
                raise ValueError("row_indices must have shape [batch]")
            start_positions = seq_lens.index_select(0, row_indices)
        else:
            start_positions = seq_lens[:batch]
            if start_positions.numel() != batch:
                raise ValueError("seq_lens must cover the ragged prefill batch")
        max_seq = cache.layers[0].max_seq_len
        if bool(torch.any(start_positions < 0)):
            raise ValueError("seq_lens must be non-negative")
        if bool(torch.any(start_positions + logit_positions >= max_seq)):
            raise ValueError("KV cache capacity exceeded")
        query_offsets = torch.arange(suffix, device=self.device)
        write_positions = (start_positions[:, None] + query_offsets[None, :]).clamp(max=max_seq - 1)
        rotary = (
            self.rotary_cos_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1),
            self.rotary_sin_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1),
        )
        if src_prefix_row is not None:
            src_prefix_row = src_prefix_row.to(self.device, non_blocking=True)
        return self._forward_prefill_ragged_static(
            input_ids,
            cache,
            start_positions,
            write_positions,
            row_indices,
            rotary,
            logit_positions,
            context_len,
            src_prefix_row,
        )

    def try_prefill_ragged_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if self._ragged_prefill_logits_graph_failed or not _should_use_ragged_prefill_logits_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            logit_positions,
        ):
            return None
        try:
            return self._run_ragged_prefill_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                logit_positions=logit_positions,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} ragged_prefill_logits_graph_failed={exc!r}", flush=True)
            self._ragged_prefill_logits_graph_failed = True
            return None

    def _run_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if not cache.layers:
            raise ValueError("ragged prefill requires a non-empty KV cache")
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        key = (
            id(cache),
            input_ids.size(0),
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            context_len if context_len is not None else -1,
            src_prefix_rows,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_prefill_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
            or captured.context_len != context_len
            or (captured.static_src_prefix_row is None) != (src_prefix_row is None)
            or (
                captured.static_src_prefix_row is not None
                and src_prefix_row is not None
                and captured.static_src_prefix_row.shape != src_prefix_row.shape
            )
        )
        skip_sync = bool(getattr(cache, "_skip_capture_sync", False))
        needs_capture = needs_capture if skip_sync else _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            if not capture_on_miss:
                return None
            succeeded = True
            new_captured: _StaticRaggedPrefillLogitsGraphCall | None = None
            try:
                new_captured = self._capture_ragged_prefill_logits_graph(
                    input_ids, cache, seq_lens, row_indices, logit_positions, context_len, src_prefix_row,
                )
            except Exception:
                succeeded = False
            if not skip_sync:
                succeeded = _capture_succeeded_on_all_ranks(succeeded, self.device)
            if not succeeded or new_captured is None:
                raise RuntimeError("ragged prefill graph capture failed on at least one rank")
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if key not in self._ragged_prefill_logits_graphs and len(self._ragged_prefill_logits_graphs) >= max_graphs:
                self._ragged_prefill_logits_graphs.clear()
            self._ragged_prefill_logits_graphs[key] = new_captured
            captured = new_captured
        else:
            self._copy_ragged_prefill_graph_inputs(
                captured, input_ids, seq_lens, row_indices, logit_positions, src_prefix_row
            )
            captured.graph.replay()
        return captured.output_logits

    def _capture_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
    ) -> _StaticRaggedPrefillLogitsGraphCall:
        batch, suffix = input_ids.shape
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_row_indices = torch.empty_like(row_indices) if row_indices is not None else None
        static_src_prefix_row = torch.empty_like(src_prefix_row) if src_prefix_row is not None else None
        captured = _StaticRaggedPrefillLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_start_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_write_positions=torch.empty((batch, suffix), device=self.device, dtype=torch.int64),
            static_row_indices=static_row_indices,
            static_rotary_cos=torch.empty((batch, suffix, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_rotary_sin=torch.empty((batch, suffix, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_logit_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_src_prefix_row=static_src_prefix_row,
            output_logits=torch.empty((batch, 1, self.local_vocab_size), device=self.device, dtype=self.dtype),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            suffix_bucket=suffix,
            context_len=context_len,
        )
        self._copy_ragged_prefill_graph_inputs(
            captured, input_ids, seq_lens, row_indices, logit_positions, src_prefix_row
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._forward_prefill_ragged_static(
                captured.static_input_ids,
                cache,
                captured.static_start_positions,
                captured.static_write_positions,
                captured.static_row_indices,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                captured.static_logit_positions,
                context_len,
                captured.static_src_prefix_row,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_prefill_ragged_static(
                captured.static_input_ids,
                cache,
                captured.static_start_positions,
                captured.static_write_positions,
                captured.static_row_indices,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                captured.static_logit_positions,
                context_len,
                captured.static_src_prefix_row,
            )
        captured.graph.replay()
        return captured

    def _copy_ragged_prefill_graph_inputs(
        self,
        captured: _StaticRaggedPrefillLogitsGraphCall,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None = None,
    ) -> None:
        batch, suffix = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        logit_positions = logit_positions.to(self.device, non_blocking=True)
        if captured.static_src_prefix_row is not None and src_prefix_row is not None:
            captured.static_src_prefix_row.copy_(src_prefix_row.to(self.device, non_blocking=True))
        if row_indices is None:
            start_positions = seq_lens[:batch]
        else:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if captured.static_row_indices is None:
                raise RuntimeError("captured ragged prefill graph does not accept row indices")
            captured.static_row_indices.copy_(row_indices)
            start_positions = seq_lens.index_select(0, row_indices)
        query_offsets = torch.arange(suffix, device=self.device)
        write_positions = (start_positions[:, None] + query_offsets[None, :]).clamp(max=captured.max_seq_len - 1)
        captured.static_input_ids.copy_(input_ids)
        captured.static_start_positions.copy_(start_positions)
        captured.static_write_positions.copy_(write_positions)
        captured.static_logit_positions.copy_(logit_positions)
        captured.static_rotary_cos.copy_(
            self.rotary_cos_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1)
        )
        captured.static_rotary_sin.copy_(
            self.rotary_sin_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1)
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids.to(self.device, non_blocking=True)
        cache = self.allocate_cache(input_ids.size(0), input_ids.size(1) + max_new_tokens)
        logits, cache = self.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        next_token = self._sample_next_token(logits[:, -1, :], temperature)
        output = [input_ids.to(self.device, non_blocking=True), next_token[:, None]]
        for _ in range(1, max_new_tokens):
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self.forward(
                next_token[:, None],
                cache=cache,
                use_cache=True,
                return_last_logits_only=True,
                return_sharded_logits=True,
            )
            next_token = self._sample_next_token(logits[:, -1, :], temperature)
            output.append(next_token[:, None])
        return torch.cat(output, dim=1)

    def _sample_next_token(self, logits: Tensor, temperature: float) -> Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return sample_next_token(logits, temperature).to(self.device)
        if temperature <= 0:
            return self._sample_next_token_greedy(logits)
        if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER", False):
            try:
                return self._sample_next_token_temperature_gather(logits, temperature)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.temperature_sample_gather", exc)
                if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER_STRICT", False):
                    raise
        if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GUMBEL"):
            try:
                return self._sample_next_token_temperature_gumbel(logits, temperature)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.temperature_sample_gumbel", exc)
                if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GUMBEL_STRICT", False):
                    raise
        return self._sample_next_token_temperature(logits, temperature)

    def sample_repeated_next_token(self, logits: Tensor, batch_size: int, temperature: float) -> Tensor:
        if batch_size <= 1:
            return self._sample_next_token(logits, temperature)
        if temperature <= 0:
            return self._sample_next_token(logits, temperature).expand(batch_size).contiguous()
        if logits.size(0) != 1:
            expanded = logits.expand(batch_size, logits.size(-1)).contiguous()
            return self._sample_next_token(expanded, temperature)
        repeated_gumbel_min_batch = _tp_int(
            "TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL_MIN_BATCH",
            128,
            minimum=1,
        )
        if (
            batch_size >= repeated_gumbel_min_batch
            and _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL", True)
        ):
            try:
                return self._sample_next_token_temperature_repeated_gumbel(logits, batch_size, temperature)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.temperature_sample_repeated_gumbel", exc)
                if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL_STRICT", False):
                    raise
        if not dist.is_available() or not dist.is_initialized():
            cumulative = torch.cumsum(torch.softmax(logits.float() / temperature, dim=-1)[0], dim=-1).contiguous()
            threshold = torch.rand((batch_size,), dtype=cumulative.dtype, device=logits.device) * cumulative[-1]
            return torch.searchsorted(cumulative, threshold)
        return self._sample_next_token_temperature_repeated(logits, batch_size, temperature)

    def _sample_next_token_greedy(self, logits: Tensor) -> Tensor:
        if _tp_flag("TORCHINFERNO_GREEDY_SAMPLE_GATHER", False):
            try:
                return self._sample_next_token_greedy_gather(logits)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.greedy_sample_gather", exc)
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        sentinel = torch.full_like(local_indices, self.config.vocab_size)
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(local_values == global_values, local_tokens, sentinel)
        dist.all_reduce(next_token, op=dist.ReduceOp.MIN)
        return next_token

    def _sample_next_token_greedy_gather(self, logits: Tensor) -> Tensor:
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        local_tokens = (local_indices + self.vocab_start).to(torch.float32)
        local_pairs = torch.stack((local_values, local_tokens), dim=-1).contiguous()
        gathered = torch.empty(
            (self.world_size, *local_pairs.shape),
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered, local_pairs)
        values = gathered[..., 0]
        tokens = gathered[..., 1].to(torch.long)
        global_values = values.max(dim=0).values
        sentinel = torch.full_like(tokens, self.config.vocab_size)
        candidate_tokens = torch.where(values == global_values[None, :], tokens, sentinel)
        return candidate_tokens.min(dim=0).values

    def _sample_next_token_temperature_gather(self, logits: Tensor, temperature: float) -> Tensor:
        local_logits = logits.contiguous()
        gathered = torch.empty(
            (self.world_size, *local_logits.shape),
            dtype=local_logits.dtype,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered, local_logits)
        next_token = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        if self.is_primary:
            full_logits = gathered.permute(1, 0, 2).reshape(logits.size(0), self.world_size * logits.size(1))
            probs = torch.softmax(full_logits.float() / temperature, dim=-1)
            next_token.copy_(torch.multinomial(probs, num_samples=1).squeeze(-1))
        dist.broadcast(next_token, src=0)
        return next_token

    def _sample_next_token_temperature_gumbel(self, logits: Tensor, temperature: float) -> Tensor:
        logits_float = logits.float() / temperature
        gumbel = -torch.empty_like(logits_float).exponential_(
            generator=self._temperature_gumbel_generator(logits.device)
        ).log()
        local_values, local_indices = torch.max(logits_float + gumbel, dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        sentinel = torch.full_like(local_indices, self.config.vocab_size)
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(local_values == global_values, local_tokens, sentinel)
        dist.all_reduce(next_token, op=dist.ReduceOp.MIN)
        return next_token

    def _temperature_gumbel_generator(self, device: torch.device) -> torch.Generator:
        generators = getattr(self, "_temperature_gumbel_generators", None)
        if generators is None:
            generators = {}
            self._temperature_gumbel_generators = generators
        key = str(device)
        generator = generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            rank = int(getattr(self, "rank", 0))
            seed = (torch.initial_seed() + (rank + 1) * 0x9E3779B97F4A7C15) % ((1 << 63) - 1)
            generator.manual_seed(seed)
            generators[key] = generator
        return generator

    def _sample_next_token_temperature(self, logits: Tensor, temperature: float) -> Tensor:
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        weights = torch.exp(logits_float - global_max[:, None])
        local_sum = weights.sum(dim=-1)
        gathered_sums = torch.empty(
            (self.world_size, *local_sum.shape),
            dtype=local_sum.dtype,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered_sums, local_sum.contiguous())

        sample_payload = torch.empty((2, logits.size(0)), dtype=torch.float32, device=self.device)
        if self.is_primary:
            cumulative = torch.cumsum(gathered_sums, dim=0)
            total = cumulative[-1]
            target = torch.rand_like(total) * total
            selected_rank = (cumulative < target[None, :]).sum(dim=0).to(torch.long)
            row = torch.arange(logits.size(0), device=self.device)
            previous = torch.zeros_like(target)
            has_previous = selected_rank > 0
            previous[has_previous] = cumulative[selected_rank[has_previous] - 1, row[has_previous]]
            sample_payload[0].copy_(selected_rank.to(sample_payload.dtype))
            sample_payload[1].copy_(target - previous)
        dist.broadcast(sample_payload, src=0)
        selected_rank = sample_payload[0].to(torch.long)
        local_threshold = sample_payload[1]

        cumulative_local = torch.cumsum(weights, dim=-1)
        local_threshold = torch.minimum(local_threshold, cumulative_local[:, -1])
        local_index = torch.searchsorted(cumulative_local.contiguous(), local_threshold[:, None]).squeeze(-1)
        local_index = torch.clamp(local_index, max=self.local_vocab_size - 1)
        selected = selected_rank == self.rank
        local_token = torch.where(
            selected,
            local_index + self.vocab_start,
            torch.zeros_like(local_index),
        )
        dist.all_reduce(local_token, op=dist.ReduceOp.SUM)
        return local_token

    def _sample_next_token_temperature_repeated(
        self,
        logits: Tensor,
        batch_size: int,
        temperature: float,
    ) -> Tensor:
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        weights = torch.exp(logits_float - global_max[:, None])
        local_sum = weights.sum(dim=-1)
        gathered_sums = torch.empty(
            (self.world_size, *local_sum.shape),
            dtype=local_sum.dtype,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered_sums, local_sum.contiguous())

        sample_payload = torch.empty((2, batch_size), dtype=torch.float32, device=self.device)
        if self.is_primary:
            cumulative = torch.cumsum(gathered_sums[:, 0], dim=0)
            total = cumulative[-1]
            target = torch.rand((batch_size,), dtype=cumulative.dtype, device=self.device) * total
            selected_rank = (cumulative[:, None] < target[None, :]).sum(dim=0).to(torch.long)
            previous = torch.zeros_like(target)
            has_previous = selected_rank > 0
            previous[has_previous] = cumulative[selected_rank[has_previous] - 1]
            sample_payload[0].copy_(selected_rank.to(sample_payload.dtype))
            sample_payload[1].copy_(target - previous)
        dist.broadcast(sample_payload, src=0)
        selected_rank = sample_payload[0].to(torch.long)
        local_threshold = sample_payload[1]

        cumulative_local = torch.cumsum(weights[0], dim=-1).contiguous()
        local_threshold = torch.minimum(local_threshold, cumulative_local[-1].expand_as(local_threshold))
        local_index = torch.searchsorted(cumulative_local, local_threshold)
        local_index = torch.clamp(local_index, max=self.local_vocab_size - 1)
        selected = selected_rank == self.rank
        local_token = torch.where(
            selected,
            local_index + self.vocab_start,
            torch.zeros_like(local_index),
        )
        dist.all_reduce(local_token, op=dist.ReduceOp.SUM)
        return local_token

    def _sample_next_token_temperature_repeated_gumbel(
        self,
        logits: Tensor,
        batch_size: int,
        temperature: float,
    ) -> Tensor:
        if not dist.is_available() or not dist.is_initialized():
            cumulative = torch.cumsum(torch.softmax(logits.float() / temperature, dim=-1)[0], dim=-1).contiguous()
            threshold = torch.rand((batch_size,), dtype=cumulative.dtype, device=logits.device) * cumulative[-1]
            return torch.searchsorted(cumulative, threshold)
        logits_float = logits.float() / temperature
        gumbel = -torch.empty(
            (batch_size, logits_float.size(-1)),
            dtype=logits_float.dtype,
            device=logits_float.device,
        ).exponential_(generator=self._temperature_gumbel_generator(logits.device)).log()
        local_values, local_indices = torch.max(logits_float.expand_as(gumbel) + gumbel, dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        sentinel = torch.full_like(local_indices, self.config.vocab_size)
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(local_values == global_values, local_tokens, sentinel)
        dist.all_reduce(next_token, op=dist.ReduceOp.MIN)
        return next_token

    def _gather_logits(self, logits: Tensor) -> Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return logits
        gathered = [torch.empty_like(logits) for _ in range(self.world_size)]
        dist.all_gather(gathered, logits)
        return torch.cat(gathered, dim=-1)

    def _rotary_cache(self, start: int, tokens: int) -> tuple[Tensor, Tensor]:
        end = start + tokens
        if end <= self.rotary_cos_cache.size(0):
            return self.rotary_cos_cache[start:end], self.rotary_sin_cache[start:end]
        positions = torch.arange(start, end, device=self.device)
        freqs = torch.outer(positions.float(), self.inv_freq)
        return freqs.cos().to(dtype=self.dtype), freqs.sin().to(dtype=self.dtype)


def _init_distributed_if_needed() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _resolve_tensor_parallel_checkpoint(
    checkpoint: str | Path,
    *,
    token: str | None,
    revision: str | None,
    cache_dir: str | Path | None,
    device: torch.device,
) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.exists():
        return candidate
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return resolve_llama3_checkpoint(checkpoint, token=token, revision=revision, cache_dir=cache_dir)

    rank = dist.get_rank()
    resolved = [""]
    if rank == 0:
        print(f"[Llama3TP] resolving checkpoint {checkpoint}", flush=True)
        resolved[0] = str(resolve_llama3_checkpoint(checkpoint, token=token, revision=revision, cache_dir=cache_dir))
        print(f"[Llama3TP] resolved checkpoint {resolved[0]}", flush=True)
    _broadcast_object_list(resolved, src=0, device=device)
    if not resolved[0]:
        raise RuntimeError("rank 0 did not broadcast a resolved checkpoint path")
    return Path(resolved[0])


def _broadcast_object_list(objects: list[object], *, src: int, device: torch.device) -> None:
    if device.type == "cuda":
        try:
            dist.broadcast_object_list(objects, src=src, device=device)
            return
        except TypeError:
            pass
    dist.broadcast_object_list(objects, src=src)


def _all_reduce(tensor: Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _build_llama_rotary_cache(
    max_position_embeddings: int,
    inv_freq: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.float())
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def _apply_rotary_cached(q: Tensor, k: Tensor, rotary: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    cos, sin = rotary
    if q.is_cuda and k.is_cuda and _tp_flag("TORCHINFERNO_TRITON_ROTARY"):
        try:
            from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_inplace

            return triton_apply_rotary_llama_inplace(q, k, cos, sin)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rotary", exc)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return _rotate_llama(q, cos, sin), _rotate_llama(k, cos, sin)


def _apply_rotary_ragged(q: Tensor, k: Tensor, rotary: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    cos, sin = rotary
    # Same fused-kernel fast path as _apply_rotary_ragged_prefill; here cos/sin
    # are [batch, rotary_dim] (one position per row), reshaped to [batch, 1, dim]
    # for the per-(batch,token) kernel.
    if q.is_cuda and k.is_cuda and _tp_flag("TORCHINFERNO_TRITON_ROTARY"):
        try:
            from torchinferno.kernels.triton_ops import (
                triton_apply_rotary_llama_batched_inplace,
            )

            return triton_apply_rotary_llama_batched_inplace(
                q, k, cos.unsqueeze(1), sin.unsqueeze(1)
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rotary_ragged", exc)
    cos = cos[:, None, None, :]
    sin = sin[:, None, None, :]
    return _rotate_llama(q, cos, sin), _rotate_llama(k, cos, sin)


def _rotate_llama(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if x.is_cuda and _tp_flag("TORCHINFERNO_COMPILE_ROTARY", False):
        compiled = _load_compiled_rotate_llama()
        if compiled is not None:
            try:
                return compiled(x, cos, sin)
            except Exception:
                global _COMPILED_ROTATE_LLAMA_FAILED
                _COMPILED_ROTATE_LLAMA_FAILED = True
    return _rotate_llama_eager(x, cos, sin)


def _load_compiled_rotate_llama():
    global _COMPILED_ROTATE_LLAMA, _COMPILED_ROTATE_LLAMA_CHECKED
    if not _COMPILED_ROTATE_LLAMA_CHECKED:
        _COMPILED_ROTATE_LLAMA_CHECKED = True
        try:
            _COMPILED_ROTATE_LLAMA = torch.compile(
                _rotate_llama_eager,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        except Exception:
            _COMPILED_ROTATE_LLAMA = None
    if _COMPILED_ROTATE_LLAMA_FAILED:
        return None
    return _COMPILED_ROTATE_LLAMA


def _rotate_llama_eager(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    if cos.size(-1) == half:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _append_ragged_kv_cache(
    cache: Llama3TensorParallelLayerKVCache,
    keys: Tensor,
    values: Tensor,
    positions: Tensor,
    row_indices: Tensor | None,
) -> None:
    if keys.ndim != 4 or values.ndim != 4 or keys.size(2) != 1 or values.size(2) != 1:
        raise ValueError("ragged KV append expects one token")
    if row_indices is None:
        rows = torch.arange(keys.size(0), device=keys.device, dtype=torch.long)
    else:
        rows = row_indices.to(device=keys.device, dtype=torch.long)
    positions = positions.to(device=keys.device, dtype=torch.long)
    _parent = getattr(cache, '_parent', None)
    paged_kv = getattr(cache, 'paged_kv', None)
    if _parent is not None:
        physical_rows = torch.tensor(cache._row_list, device=keys.device, dtype=torch.long)
        phys_rows = physical_rows[rows]
        _parent.paged_kv[phys_rows, 0, positions, :, :] = keys[:, :, 0, :]
        _parent.paged_kv[phys_rows, 1, positions, :, :] = values[:, :, 0, :]
    elif paged_kv is not None:
        paged_kv[rows, 0, positions, :, :] = keys[:, :, 0, :]
        paged_kv[rows, 1, positions, :, :] = values[:, :, 0, :]
    else:
        cache.keys[rows, :, positions, :] = keys[:, :, 0, :]
        cache.values[rows, :, positions, :] = values[:, :, 0, :]


def _ragged_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_lengths: Tensor,
    *,
    row_indices: Tensor | None = None,
    enable_gqa: bool,
) -> Tensor:
    if (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and enable_gqa
        and _tp_flag("TORCHINFERNO_TRITON_GROUPED_DECODE_ATTENTION")
    ):
        try:
            from torchinferno.kernels.triton_ops import triton_grouped_gqa_decode_attention

            return triton_grouped_gqa_decode_attention(
                q,
                k,
                v,
                attention_lengths.to(device=k.device),
                row_indices=row_indices,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_attention", exc)
    if row_indices is not None:
        k = k.index_select(0, row_indices)
        v = v.index_select(0, row_indices)
    if not q.is_cuda:
        max_attention_len = int(attention_lengths.max().item()) if attention_lengths.numel() else 0
        if max_attention_len > 0:
            k = k[:, :, :max_attention_len, :]
            v = v[:, :, :max_attention_len, :]
    max_seq_len = k.size(2)
    key_positions = torch.arange(max_seq_len, device=q.device)
    mask = key_positions[None, :] < attention_lengths.to(device=q.device)[:, None]
    # Cache rows beyond attention_lengths are unwritten (torch.empty) and may hold
    # NaN; SDPA masks their weight to zero but NaN*0 == NaN poisons the output.
    # Zero the masked keys/values first, matching the guard in the ragged-prefill
    # fallback above. (Padding contributes nothing either way.)
    zero = torch.zeros((), dtype=k.dtype, device=k.device)
    k = torch.where(mask[:, None, :, None], k, zero)
    v = torch.where(mask[:, None, :, None], v, zero)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=enable_gqa,
    )


def _apply_rotary_ragged_prefill(
    q: Tensor,
    k: Tensor,
    rotary: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    # Per-row, PER-TOKEN rotary for a ragged prefill suffix. cos/sin are
    # [batch, T, rotary_dim] (each row's tokens carry their own absolute
    # positions, unlike decode's single position per row); broadcast over the
    # head dim. _rotate_llama handles the half-width cos/sin via a cat.
    cos, sin = rotary
    # Fused-kernel fast path: the aten rotate-half below is cat/neg/mul that the
    # decode profiler showed at ~1.3ms/step (after the cos/sin pre-expand hoist);
    # the batched triton kernel does it in one in-place launch (~0.1ms/step, 13x).
    # Used by BOTH the graph-captured and eager decode, so graph-vs-eager stays
    # bit-identical (same kernel both sides). Same flag as the uniform-prefill
    # fused rope in _apply_rotary_cached.
    if q.is_cuda and k.is_cuda and _tp_flag("TORCHINFERNO_TRITON_ROTARY"):
        try:
            from torchinferno.kernels.triton_ops import (
                triton_apply_rotary_llama_batched_inplace,
            )

            return triton_apply_rotary_llama_batched_inplace(q, k, cos, sin)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rotary_ragged_prefill", exc)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return _rotate_llama(q, cos, sin), _rotate_llama(k, cos, sin)


def _append_ragged_kv_prefill(
    cache: Llama3TensorParallelLayerKVCache,
    keys: Tensor,
    values: Tensor,
    positions: Tensor,
    row_indices: Tensor | None,
) -> None:
    # Multi-token generalization of _append_ragged_kv_cache: scatter-write a
    # T-token suffix into arbitrary (scattered) physical rows at per-row write
    # columns. keys/values are [batch, kv_heads, T, head_dim]; positions is
    # [batch, T] of absolute KV columns. Pure advanced-index assignment so it is
    # legal inside a CUDA graph capture (no host->device list->tensor copy).
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError("ragged KV prefill append expects 4D keys/values")
    batch, _kv_heads, tokens, _head_dim = keys.shape
    if row_indices is None:
        rows = torch.arange(batch, device=keys.device, dtype=torch.long)
    else:
        rows = row_indices.to(device=keys.device, dtype=torch.long)
    positions = positions.to(device=keys.device, dtype=torch.long)
    row_idx = rows[:, None].expand(batch, tokens)
    paged_kv = getattr(cache, 'paged_kv', None)
    _parent = getattr(cache, '_parent', None)
    if _parent is not None:
        physical_rows = torch.tensor(cache._row_list, device=keys.device, dtype=torch.long)
        phys_row_idx = physical_rows[row_idx]
        _parent.paged_kv[phys_row_idx, 0, positions, :, :] = keys.permute(0, 2, 1, 3)
        _parent.paged_kv[phys_row_idx, 1, positions, :, :] = values.permute(0, 2, 1, 3)
    elif paged_kv is not None:
        paged_kv[row_idx, 0, positions, :, :] = keys.permute(0, 2, 1, 3)
        paged_kv[row_idx, 1, positions, :, :] = values.permute(0, 2, 1, 3)
    else:
        cache.keys[row_idx, :, positions, :] = keys.permute(0, 2, 1, 3)
        cache.values[row_idx, :, positions, :] = values.permute(0, 2, 1, 3)


def _ragged_prefill_scaled_dot_product_attention(
    q: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    start_positions: Tensor,
    *,
    suffix_tokens: int,
    row_indices: Tensor | None,
    enable_gqa: bool,
    context_len: int | None = None,
) -> Tensor:
    # Attention for a ragged prefill suffix on scattered cache rows.
    #
    # FAST PATH (context_len given, uniform prefix): keys [0:context_len] hold
    # prefix+suffix for every row, so causal_lower_right(T, context_len) (query j
    # attends keys [0, context_len - T + j] = [0, start + j]) is exactly the
    # offset-causal mask AND uses a flash kernel with NO materialized
    # [batch,heads,T,context_len] attention matrix -- essential because the
    # boolean-mask math backend OOMs at large suffix x context. This is how the
    # batch worker prefills fast; here we keep it with scattered rows.
    #
    # FALLBACK (context_len None, possibly MIXED per-row prefixes -- the eager
    # CPU oracle): explicit per-row offset-causal boolean mask over the full
    # cache. Correct for mixed starts; only used off the hot serving path.
    if row_indices is not None:
        k = cache_keys.index_select(0, row_indices)
        v = cache_values.index_select(0, row_indices)
    else:
        k = cache_keys[: q.size(0)]
        v = cache_values[: q.size(0)]
    if context_len is not None:
        k = k[:, :, :context_len, :]
        v = v[:, :, :context_len, :]
        from torch.nn.attention.bias import causal_lower_right

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_lower_right(suffix_tokens, context_len),
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
    # Eager mask path only (never inside a graph -- the graph path always passes
    # context_len). Supports MIXED per-row prefixes. Replace every key/value that
    # a row has not written (column >= start+suffix) with zero via torch.where so
    # the masked math-backend SDPA cannot propagate NaN out of uninitialized
    # cache memory -- multiply-by-mask fails because NaN*0 == NaN, and the
    # additive -inf bias fails because NaN + -inf == NaN.
    start_positions = start_positions.to(device=q.device)
    eff_len = int((start_positions + suffix_tokens).max().item())
    eff_len = max(1, min(eff_len, k.size(2)))
    k = k[:, :, :eff_len, :]
    v = v[:, :, :eff_len, :]
    key_positions = torch.arange(eff_len, device=q.device)
    query_offsets = torch.arange(suffix_tokens, device=q.device)
    written = key_positions[None, :] < (start_positions[:, None] + suffix_tokens)  # [batch, eff_len]
    zero = torch.zeros((), dtype=k.dtype, device=k.device)
    k = torch.where(written[:, None, :, None], k, zero)
    v = torch.where(written[:, None, :, None], v, zero)
    q_abs = start_positions[:, None] + query_offsets[None, :]  # [batch, T]
    mask = key_positions[None, None, :] <= q_abs[:, :, None]  # [batch, T, eff_len]
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask[:, None, :, :],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=enable_gqa,
    )


def _decode_linear(x: Tensor, weight: Tensor, weight_t: Tensor | None = None) -> Tensor:
    if (
        x.is_cuda
        and x.ndim == 3
        and x.size(0) == 1
        and x.size(1) == 1
        and _tp_flag("TORCHINFERNO_DECODE_LINEAR_MV")
    ):
        if weight_t is not None:
            return torch.mm(x.reshape(1, -1), weight_t).view(1, 1, weight.size(0))
        return torch.mv(weight, x.reshape(-1)).view(1, 1, weight.size(0))
    if (
        x.is_cuda
        and x.ndim == 3
        and x.size(1) == 1
        and weight_t is not None
        and _tp_flag("TORCHINFERNO_DECODE_LINEAR_MM")
        and x.size(0) <= _tp_int("TORCHINFERNO_DECODE_LINEAR_MM_MAX_BATCH", _DEFAULT_DECODE_STEP_MAX_BATCH, minimum=1)
    ):
        return torch.mm(x.reshape(-1, x.size(-1)), weight_t).view(x.size(0), 1, weight.size(0))
    return F.linear(x, weight)


def _maybe_decode_weight_t(weight: Tensor) -> Tensor | None:
    if (
        not weight.is_cuda
        or not _tp_flag("TORCHINFERNO_DECODE_TRANSPOSED_WEIGHTS")
        or weight.ndim != 2
    ):
        return None
    try:
        return weight.t().contiguous()
    except Exception as exc:
        warn_optional_failure("llama3_tensor_parallel.decode_weight_transpose", exc)
        return None


def _tp_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if x.is_cuda and weight.is_cuda and _tp_flag("TORCHINFERNO_TRITON_RMS_NORM", False):
        try:
            from torchinferno.kernels import rms_norm as kernel_rms_norm

            return kernel_rms_norm(x, weight, eps=eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rms_norm", exc)
    return _torch_rms_norm(x, weight, eps)


def _tp_decode_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if x.is_cuda and weight.is_cuda and _tp_flag("TORCHINFERNO_TRITON_DECODE_RMS_NORM"):
        try:
            from torchinferno.kernels import rms_norm as kernel_rms_norm

            return kernel_rms_norm(x, weight, eps=eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_rms_norm", exc)
    return _torch_rms_norm(x, weight, eps)


def _tp_decode_add_rms_norm(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    if (
        x.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and _tp_flag("TORCHINFERNO_TRITON_DECODE_ADD_RMS_NORM")
    ):
        try:
            from torchinferno.kernels.triton_ops import triton_add_rms_norm

            return triton_add_rms_norm(x, residual, weight, eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_add_rms_norm", exc)
    hidden = residual + x
    return hidden, _torch_rms_norm(hidden, weight, eps)


def _tp_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_SWIGLU", False):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.swiglu", exc)
    return F.silu(gate) * up


def _tp_decode_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_DECODE_SWIGLU"):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_swiglu", exc)
    return F.silu(gate) * up


def _should_use_mlp_project_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) == 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_MLP")
    )


def _should_use_qkv_rotary_graph(hidden: Tensor) -> bool:
    prefill_tokens = hidden.size(1) > 1 and _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_QKV_ROTARY", False)
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and (hidden.size(1) == 1 or prefill_tokens)
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_QKV_ROTARY")
    )


def _should_use_prefill_gate_up_activation_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) > 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_GATE_UP")
    )


def _should_use_attention_o_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) == 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_ATTENTION_O")
    )


def _should_graph_all_reduce() -> bool:
    return _tp_flag("TORCHINFERNO_CUDAGRAPH_ALLREDUCE", False)


def _should_use_decode_step_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    temperature: float,
) -> bool:
    cache_keys = getattr(cache.layers[0], "keys", None) if cache.layers else None
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_STEP", True)
        and temperature <= 0.0
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and input_ids.size(1) == 1
        and cache_keys is not None
        and cache_keys.is_cuda
        and (
            _static_decode_cache_rows_are_contiguous(cache, input_ids.size(0))
            or _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_SPARSE_ROWS", False)
        )
    )


def _should_use_decode_step_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
) -> bool:
    cache_keys = getattr(cache.layers[0], "keys", None) if cache.layers else None
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_STEP", True)
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and input_ids.size(1) == 1
        and cache_keys is not None
        and cache_keys.is_cuda
        and (
            _static_decode_cache_rows_are_contiguous(cache, input_ids.size(0))
            or _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_SPARSE_ROWS", False)
        )
    )


def _static_decode_cache_rows_are_contiguous(cache: Llama3TensorParallelCache, batch: int) -> bool:
    for layer in cache.layers:
        if not hasattr(layer, '_selected_rows'):
            return False
    return all(_contiguous_row_span(layer._selected_rows(batch)) is not None for layer in cache.layers)


def _static_decode_row_indices(cache: Llama3TensorParallelCache, batch: int) -> Tensor | None:
    if not cache.layers:
        return None
    return cache.layers[0]._selected_row_indices_tensor(batch)


def _should_use_ragged_decode_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    seq_lens: Tensor,
    row_indices: Tensor | None,
) -> bool:
    cache_keys = _prefill_graph_cache_storage(cache)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP", True)
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and input_ids.size(1) == 1
        and seq_lens.is_cuda
        and seq_lens.ndim == 1
        and (row_indices is None or (row_indices.is_cuda and row_indices.shape == (input_ids.size(0),)))
        and cache_keys is not None
        and cache_keys.is_cuda
    )


def _should_use_ragged_decode_token_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    temperature: float,
) -> bool:
    return temperature <= 0.0 and _should_use_ragged_decode_logits_graph(input_ids, cache, seq_lens, row_indices)


def _should_use_ragged_prefill_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    logit_positions: Tensor,
) -> bool:
    cache_keys = _prefill_graph_cache_storage(cache)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_PREFILL", True)
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and input_ids.size(1) >= 1
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and seq_lens.is_cuda
        and seq_lens.ndim == 1
        and logit_positions.is_cuda
        and logit_positions.shape == (input_ids.size(0),)
        and (row_indices is None or (row_indices.is_cuda and row_indices.shape == (input_ids.size(0),)))
        and cache_keys is not None
        and cache_keys.is_cuda
    )


def _decode_step_max_batch() -> int:
    return _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", _DEFAULT_DECODE_STEP_MAX_BATCH, minimum=1)


def _should_use_prefill_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    temperature: float,
) -> bool:
    max_cache_tokens = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_CACHE_TOKENS", 8192, minimum=1)
    cache_keys = _prefill_graph_cache_storage(cache)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL")
        and temperature <= 0.0
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_BATCH", 64, minimum=1)
        and input_ids.size(1) > 1
        and cache_keys is not None
        and cache_keys.is_cuda
        and cache.layers[0].max_seq_len <= max_cache_tokens
    )


def _should_use_prefill_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
) -> bool:
    max_cache_tokens = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_CACHE_TOKENS", 8192, minimum=1)
    cache_keys = _prefill_graph_cache_storage(cache)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL")
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_BATCH", 64, minimum=1)
        and input_ids.size(1) > 1
        and cache_keys is not None
        and cache_keys.is_cuda
        and cache.layers[0].max_seq_len <= max_cache_tokens
    )


_PREFILL_TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024)


def _prefill_bucket_size(prompt_tokens: int) -> int | None:
    for bucket in _PREFILL_TOKEN_BUCKETS:
        if prompt_tokens <= bucket:
            return bucket
    return None


def _prefill_graph_cache_storage(cache: Llama3TensorParallelCache) -> Tensor | None:
    if not cache.layers:
        return None
    layer = cache.layers[0]
    cache_keys = getattr(layer, "keys", None)
    if isinstance(cache_keys, Tensor):
        return cache_keys
    pages = getattr(layer, "pages", None)
    page_keys = getattr(pages, "keys", None)
    return page_keys if isinstance(page_keys, Tensor) else None


def _cache_graph_root_id(cache: Llama3TensorParallelCache) -> int:
    return int(getattr(cache, "_graph_cache_id", id(cache)))


def _prefill_graph_cache_rows(cache: Llama3TensorParallelCache, batch: int) -> tuple[int, ...]:
    if not cache.layers:
        return tuple(range(batch))
    return cache.layers[0]._selected_rows(batch)


def _prefill_graph_cache_key(cache: Llama3TensorParallelCache, batch: int) -> tuple[int, tuple[int, ...]]:
    return _cache_graph_root_id(cache), _prefill_graph_cache_rows(cache, batch)


def _same_prefill_graph_cache(
    left: Llama3TensorParallelCache,
    right: Llama3TensorParallelCache,
    batch: int,
) -> bool:
    return _prefill_graph_cache_key(left, batch) == _prefill_graph_cache_key(right, batch)


def _decode_attention_block_size(attention_length: int, max_seq_len: int) -> int:
    if not _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_ATTENTION_BLOCKS", True):
        return max_seq_len
    if attention_length <= 1:
        return 1
    return min(max_seq_len, 1 << (attention_length - 1).bit_length())


def _should_use_symm_mem_all_reduce(hidden: Tensor, weight: Tensor, world_size: int) -> bool:
    max_batch = _symm_mem_allreduce_max_batch()
    return (
        world_size > 1
        and not _SYMM_REDUCE_DISABLED
        and _symm_mem_allreduce_enabled()
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) <= max_batch
        and hidden.size(1) == 1
    )


def _symm_mem_allreduce_max_batch() -> int:
    # 256 (was 1): apply symm-mem multimem_all_reduce to BATCHED decode, not just
    # single-row. Decode does 160 small allreduces/step (2/layer x 80); symm-mem
    # is ~3x faster than NCCL ring for these latency-bound sizes. Measured on
    # 8xH100: 16-concurrent decode TPOT 38 -> 27 ms (-29%), 32-conc 29 -> 26 ms,
    # correct output, decode graphs still capture. The per-shape symm-mem buffer
    # is [batch, hidden] (tiny), so a high cap costs ~nothing; 256 covers any
    # realistic decode batch (cache rows).
    max_batch = _SYMM_MEM_ALLREDUCE_MAX_BATCH_OVERRIDE[0]
    if max_batch is not None:
        return max_batch
    return _tp_int("TORCHINFERNO_SYMM_MEM_ALLREDUCE_MAX_BATCH", 256, minimum=1)


def _symm_mem_allreduce_enabled() -> bool:
    override = _SYMM_MEM_ALLREDUCE_ENABLED_OVERRIDE[0]
    if override is not None:
        return override
    # DEFAULT ON, made safe by validate_symm_mem_allreduce_collective() at warmup.
    # History: the per-rank runtime fallback could deadlock when multicast init failed
    # on a subset of ranks (rank6 "CUDA driver error" -> some ranks NCCL, others
    # multimem -> mismatched collectives -> 1800s warmup timeout = the e211b4b dash
    # cause). The warmup handshake now votes COLLECTIVELY: any rank that cannot init
    # multicast (a clean exception) makes ALL ranks disable symm-mem together, so there
    # is no divergence. Good hosts keep the measured allreduce wins (same-session full
    # bench A/B 2026-06-09: few_shot ttft -20% / tpot -7%, self_consistency ttft -6%,
    # multi_turn tpot -7%, all correct -- the earlier "symm-mem breaks few_shot" was the
    # SEPARATE cross-benchmark cache bug, fixed in e9d8299). Force off with the env flag
    # set to 0 if a host HANGS (rather than cleanly errors) in symm_mem.rendezvous.
    return _tp_flag("TORCHINFERNO_SYMM_MEM_ALLREDUCE", True)


def _symm_mem_allreduce_graph_key(batch_size: int, world_size: int) -> int:
    if world_size <= 1 or _SYMM_REDUCE_DISABLED or not _symm_mem_allreduce_enabled():
        return 0
    max_batch = _symm_mem_allreduce_max_batch()
    return max_batch if batch_size <= max_batch else 0


def _model_world_size(model: object) -> int:
    return int(getattr(model, "world_size", 1))


def _should_use_symm_mem_prefill_all_reduce(hidden: Tensor, weight: Tensor, world_size: int) -> bool:
    # symm-mem multimem_all_reduce beats NCCL ring 1.38-3.16x on 8xH100 for
    # prefill-sized [tokens, hidden] bf16 (scripts/bench_allreduce.py). Allreduce
    # is ~31% of prefill (profiled), so this is the top prefill lever. ON by
    # default. Restricted to EAGER execution: during CUDA-graph capture the
    # per-shape symm-mem buffer would need a rendezvous (a collective) inside the
    # graph, which is illegal -- so captured (bucketed) prefill stays on NCCL and
    # only the eager path (e.g. few_shot's 48x640 graph-miss) uses symm-mem.
    max_tokens = _tp_int("TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE_MAX_TOKENS", 40960, minimum=1)
    return (
        world_size > 1
        and not _SYMM_REDUCE_DISABLED
        # DEFAULT OFF for the same per-rank-fallback divergence-deadlock reason as
        # the decode path (see _symm_mem_allreduce_enabled): a subset-rank multicast
        # init failure on a fresh remote host hangs warmup -> bench dash runs.
        and _tp_flag("TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE", False)
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.ndim == 3
        and hidden.size(1) > 1
        and hidden.size(0) * hidden.size(1) <= max_tokens
        and not torch.cuda.is_current_stream_capturing()
    )


def _disable_symm_reduce() -> None:
    global _SYMM_REDUCE_DISABLED
    _SYMM_REDUCE_DISABLED = True


def validate_symm_mem_allreduce_collective(model: object, device: object) -> None:
    # COLLECTIVE handshake (all ranks must call): probe symm-mem multicast once at
    # warmup, then NCCL-vote so that if ANY rank cannot init multicast (the observed
    # "init_multicast_for_block: CUDA driver error" on a fresh remote host, a clean
    # exception), ALL ranks disable symm-mem together and fall back to NCCL. This
    # makes symm-mem safe to default-ON: good hosts get the measured allreduce wins
    # (few_shot ttft -20%), bad hosts fall back collectively with no mismatched-
    # collective deadlock (the e211b4b dash cause). No-op if symm-mem is off / TP=1.
    import torch.distributed as dist

    world = _model_world_size(model)
    if world <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    if _SYMM_REDUCE_DISABLED or not _symm_mem_allreduce_enabled():
        return
    ok = 1
    try:
        import torch.distributed._symmetric_memory as symm_mem

        group_name = dist.group.WORLD.group_name
        hidden = int(getattr(getattr(model, "config", object()), "hidden_size", 0)) or 8192
        buffer = symm_mem.empty((1, hidden), device=device, dtype=torch.bfloat16)
        symm_mem.rendezvous(buffer, group_name)
        buffer.zero_()
        torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        torch.cuda.synchronize(device)
    except Exception:
        ok = 0
    flag = torch.tensor([ok], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) == 0:
        _disable_symm_reduce()


def _rotate_interleaved_eager(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    out = torch.empty_like(x)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            dist.barrier()
