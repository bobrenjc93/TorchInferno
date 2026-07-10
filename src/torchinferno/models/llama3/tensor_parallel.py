from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
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
_SYMM_MEM_PREFILL_ALLREDUCE_ENABLED_OVERRIDE: list[bool | None] = [None]
_DEFAULT_DECODE_STEP_MAX_BATCH = 64
_DEFAULT_PREFILL_GRAPH_MAX_GRAPHS = 192
# Keep a small allocator cushion; CUDA graph private pools can otherwise consume
# all free memory on tight 70B tensor-parallel hosts before the next prefill.
_DEFAULT_PREFILL_GRAPH_MIN_FREE_MB = 1024
_PackedPrefillAttentionGroup = tuple[int, int, Tensor, tuple[int, ...], tuple[int, ...]]


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


@contextmanager
def symm_mem_prefill_allreduce(enabled: bool | None = None) -> Iterator[None]:
    previous_enabled = _SYMM_MEM_PREFILL_ALLREDUCE_ENABLED_OVERRIDE[0]
    _SYMM_MEM_PREFILL_ALLREDUCE_ENABLED_OVERRIDE[0] = enabled
    try:
        yield
    finally:
        _SYMM_MEM_PREFILL_ALLREDUCE_ENABLED_OVERRIDE[0] = previous_enabled


def _tp_flag(name: str, default: bool = True) -> bool:
    return env_flag(name, default)


def _tp_int(name: str, default: int, *, minimum: int | None = None) -> int:
    return env_int(name, default, minimum=minimum)


def _prefill_graph_max_graphs() -> int:
    return _tp_int(
        "TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS",
        _DEFAULT_PREFILL_GRAPH_MAX_GRAPHS,
        minimum=1,
    )


def _prefill_graph_min_free_bytes() -> int:
    min_free_mb = _tp_int(
        "TORCHINFERNO_CUDAGRAPH_PREFILL_MIN_FREE_MB",
        _DEFAULT_PREFILL_GRAPH_MIN_FREE_MB,
        minimum=0,
    )
    return min_free_mb * 1024 * 1024


def _tp_env_set(name: str) -> bool:
    return name in os.environ


def _ragged_prefill_precision_graph_key(
    token_count: int,
    *,
    is_cuda: bool,
    layers: Sequence[object],
) -> tuple[bool, ...]:
    if not is_cuda:
        return (False,)
    env_configured = _tp_env_set("TORCHINFERNO_FP8_PREFILL")
    env_enabled = _tp_flag("TORCHINFERNO_FP8_PREFILL", False) if env_configured else False
    if env_configured and not env_enabled:
        return (False,)
    env_min_m = (
        _tp_int("TORCHINFERNO_FP8_PREFILL_MIN_M", 256, minimum=1)
        if _tp_env_set("TORCHINFERNO_FP8_PREFILL_MIN_M")
        else None
    )
    layer_modes: list[bool] = []
    for layer in layers:
        runtime_enabled = bool(getattr(layer, "_runtime_fp8_prefill_enabled", False))
        if not env_enabled and not runtime_enabled:
            layer_modes.append(False)
            continue
        if env_min_m is not None:
            min_m = env_min_m
        elif runtime_enabled and not env_enabled:
            min_m = int(getattr(layer, "_runtime_fp8_prefill_min_m", 2048))
        else:
            min_m = 256
        layer_modes.append(token_count > max(1, min_m))
    if not layer_modes or not any(layer_modes):
        return (False,)
    if all(layer_modes):
        return (True,)
    return tuple(layer_modes)


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


def _tp_positive_int_csv(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value > 0:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _ragged_decode_cache_token_bucket(
    cache: Llama3TensorParallelCache,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    *,
    batch: int,
) -> int:
    if not cache.layers:
        return 0
    max_seq_len = int(cache.layers[0].max_seq_len)
    explicit_limit = _ragged_decode_cache_token_limit(cache, max_seq_len, batch=batch)
    if explicit_limit is not None:
        return explicit_limit
    if not _tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKETS", False):
        return max_seq_len
    if row_indices is None:
        selected = seq_lens[:batch]
    else:
        selected = seq_lens.index_select(0, row_indices.to(device=seq_lens.device, dtype=torch.long))
    needed = int(selected.max().item()) + 1 if selected.numel() else 1
    needed = max(1, min(needed, max_seq_len))
    buckets = sorted(
        set(
            _tp_positive_int_csv(
                "TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKET_VALUES",
                "128,256,512,1024,2048,4096,8192",
            )
        )
    )
    for bucket in buckets:
        if bucket >= needed:
            return min(int(bucket), max_seq_len)
    return max_seq_len


def _ragged_decode_cache_token_limit(
    cache: Llama3TensorParallelCache,
    max_seq_len: int,
    *,
    batch: int,
) -> int | None:
    raw_limit = getattr(cache, "_torchinferno_ragged_decode_cache_token_limit", None)
    if raw_limit is None:
        return None
    try:
        min_batch = int(getattr(cache, "_torchinferno_ragged_decode_cache_token_min_batch", 1))
    except (TypeError, ValueError):
        min_batch = 1
    if batch < max(1, min_batch):
        return None
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    return max(1, min(limit, max_seq_len))


def _prepare_paged_ragged_decode_graph_state(
    cache: Llama3TensorParallelCache,
    *,
    batch: int,
    cache_positions: Tensor,
    row_indices: Tensor | None,
    device: torch.device,
    cache_token_bucket: int | None = None,
    steps: int = 1,
    page_tables: Sequence[Tensor] | None = None,
    seq_lens_buffers: Sequence[Tensor] | None = None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]] | None:
    if getattr(cache, "cache_backend", "dense") != "paged":
        return None
    positions = tuple(int(position) for position in cache_positions.detach().cpu().tolist())
    if len(positions) != batch:
        return None
    row_values = None if row_indices is None else tuple(int(row) for row in row_indices.detach().cpu().tolist())
    if row_values is not None and len(row_values) != batch:
        return None
    max_seq_len = int(cache.layers[0].max_seq_len) if cache.layers else 0
    target_cache_tokens = max_seq_len if cache_token_bucket is None else int(cache_token_bucket)
    target_cache_tokens = max(1, min(target_cache_tokens, max_seq_len)) if max_seq_len > 0 else 1
    decode_steps = max(1, int(steps))
    decode_lengths = cache_positions.to(device=device, dtype=torch.long) + 1
    shared_seq_lens: Tensor | None = None
    if seq_lens_buffers is not None:
        for candidate in seq_lens_buffers:
            if (
                isinstance(candidate, Tensor)
                and candidate.shape == (batch,)
                and candidate.device == device
                and candidate.dtype == torch.long
            ):
                shared_seq_lens = candidate
                break
    if shared_seq_lens is None:
        for layer in cache.layers:
            candidate = getattr(layer, "_torchinferno_paged_decode_seq_lens", None)
            if (
                isinstance(candidate, Tensor)
                and candidate.shape == (batch,)
                and candidate.device == device
                and candidate.dtype == torch.long
            ):
                shared_seq_lens = candidate
                break
    if shared_seq_lens is None:
        shared_seq_lens = torch.empty((batch,), dtype=torch.long, device=device)
    shared_seq_lens.copy_(decode_lengths)
    prepared_page_tables: list[Tensor] = []
    prepared_seq_lens: list[Tensor] = []
    for layer_index, layer in enumerate(cache.layers):
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        rows = layer._selected_rows(batch) if row_values is None else tuple(layer._physical_row(row) for row in row_values)
        pages_per_row = max(1, (target_cache_tokens + layer.pages.page_size - 1) // layer.pages.page_size)
        page_rows: list[list[int]] = []
        for row, position in zip(rows, positions):
            if position < 0 or position >= layer.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            request_id = layer.request_ids[row]
            seq = layer.pages.sequence(request_id)
            decode_end = position + decode_steps
            if decode_end > layer.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            layer.pages._ensure_capacity(seq, decode_end)
            start_page = position // layer.pages.page_size
            end_page = (decode_end - 1) // layer.pages.page_size
            for page_index in range(start_page, end_page + 1):
                layer.pages._prepare_page_for_write(seq, page_index)
            pages = [int(page_id) for page_id in seq.page_ids[:pages_per_row]]
            page_rows.append(pages + [0] * (pages_per_row - len(pages)))
        page_table = None
        if page_tables is not None and layer_index < len(page_tables):
            candidate = page_tables[layer_index]
            if (
                isinstance(candidate, Tensor)
                and candidate.shape == (batch, pages_per_row)
                and candidate.device == device
                and candidate.dtype == torch.long
            ):
                page_table = candidate
        if page_table is None:
            candidate = getattr(layer, "_torchinferno_paged_decode_page_table", None)
            if (
                isinstance(candidate, Tensor)
                and candidate.shape == (batch, pages_per_row)
                and candidate.device == device
                and candidate.dtype == torch.long
            ):
                page_table = candidate
        if page_table is None:
            page_table = torch.empty((batch, pages_per_row), dtype=torch.long, device=device)
        page_table.copy_(torch.tensor(page_rows, dtype=torch.long, device=device))
        layer._torchinferno_paged_decode_page_table = page_table
        layer._torchinferno_paged_decode_seq_lens = shared_seq_lens
        layer._torchinferno_paged_decode_cache_tokens = int(target_cache_tokens)
        prepared_page_tables.append(page_table)
        prepared_seq_lens.append(shared_seq_lens)
    return tuple(prepared_page_tables), tuple(prepared_seq_lens)


def _advance_paged_ragged_decode_cache_lengths(
    cache: Llama3TensorParallelCache,
    *,
    batch: int,
    cache_positions: Tensor,
    row_indices: Tensor | None,
    steps: int = 1,
) -> None:
    if getattr(cache, "cache_backend", "dense") != "paged":
        return
    positions = tuple(int(position) for position in cache_positions.detach().cpu().tolist())
    if len(positions) != batch:
        return
    row_values = None if row_indices is None else tuple(int(row) for row in row_indices.detach().cpu().tolist())
    if row_values is not None and len(row_values) != batch:
        return
    decode_steps = max(1, int(steps))
    for layer in cache.layers:
        if not isinstance(layer, PagedLlama3TensorParallelLayerKVCache):
            continue
        rows = layer._selected_rows(batch) if row_values is None else tuple(layer._physical_row(row) for row in row_values)
        for row, position in zip(rows, positions):
            request_id = layer.request_ids[row]
            seq = layer.pages.sequence(request_id)
            target_length = position + decode_steps
            if seq.length < target_length:
                seq.length = target_length


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
    cache_token_bucket: int
    static_paged_decode_page_tables: tuple[Tensor, ...] | None = None
    static_paged_decode_seq_lens: tuple[Tensor, ...] | None = None
    rotary_in_graph: bool = False


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
    cache_token_bucket: int
    static_paged_decode_page_tables: tuple[Tensor, ...] | None = None
    static_paged_decode_seq_lens: tuple[Tensor, ...] | None = None
    rotary_in_graph: bool = False


@dataclass
class _StaticRaggedDecodeManyGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_positions: Tensor
    static_row_indices: Tensor | None
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_tokens: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    cache_token_bucket: int
    steps: int
    static_paged_decode_page_tables: tuple[Tensor, ...] | None = None
    static_paged_decode_seq_lens: tuple[Tensor, ...] | None = None
    rotary_in_graph: bool = True


@dataclass
class _StaticRaggedPrefillLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor  # [batch, suffix_bucket]
    static_start_positions: Tensor  # [batch] per-row prefix length (write start)
    static_write_positions: Tensor  # [batch, suffix_bucket] absolute KV write columns
    static_query_offsets: Tensor  # [suffix_bucket] offsets added to start positions
    static_row_indices: Tensor | None  # [batch] scattered physical rows
    static_rotary_cos: Tensor  # [batch, suffix_bucket, rotary_dim]
    static_rotary_sin: Tensor  # [batch, suffix_bucket, rotary_dim]
    static_logit_positions: Tensor | None  # [batch] real last-token index per row
    static_src_prefix_row: Tensor | None  # [1] shared-prefix source row (folded copy)
    output_logits: Tensor  # [batch, 1, local_vocab_size], or empty for no-logits prefill
    output_token: Tensor | None  # [batch] greedy sampled token when captured with token output
    cache: Llama3TensorParallelCache
    max_seq_len: int
    suffix_bucket: int
    context_len: int | None
    prefix_copy_len: int | None
    emit_logits: bool = True
    emit_tokens: bool = False
    rotary_in_graph: bool = False
    write_positions_in_graph: bool = False


@dataclass
class _StaticPackedRaggedPrefillLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_start_positions: Tensor
    static_q_lens: Tensor
    static_row_indices: Tensor
    static_logit_positions: Tensor
    static_src_prefix_row: Tensor | None
    static_request_offsets: Tensor
    static_flat_token_indices: Tensor
    static_flat_request_indices: Tensor
    packed_attention_groups: tuple[_PackedPrefillAttentionGroup, ...]
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    suffix_bucket: int
    q_lens_key: tuple[int, ...]
    start_positions_key: tuple[int, ...]
    prefix_copy_len: int | None


@dataclass
class _RepeatedTemperatureSampleState:
    temperature: float
    cumulative_local: Tensor
    rank_cumulative: Tensor | None
    prefetch: int
    cached_tokens: Tensor | None = None
    cached_offset: int = 0


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
        self._runtime_fp8_prefill_enabled = False
        self._runtime_fp8_prefill_min_m = 2048
        self._runtime_marlin_int4_decode_enabled = True
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
        self._decode_scratch_buffers: dict[tuple[str, torch.device, torch.dtype, tuple[int, ...]], Tensor] = {}

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
        projected = self._mlp_project_fast_prefill(mlp_in)
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
        attention_cache_tokens: int | None = None,
        attention_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_decode_ragged(
            hidden,
            attn_in,
            rotary,
            cache,
            cache_positions,
            row_indices,
            attention_cache_tokens=attention_cache_tokens,
            attention_lengths=attention_lengths,
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
        *,
        attention_cache_tokens: int | None = None,
        attention_lengths: Tensor | None = None,
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
        if attention_lengths is None:
            attention_lengths = cache_positions + 1
        if row_indices is None:
            attention_keys = cache.keys[:batch]
            attention_values = cache.values[:batch]
        else:
            attention_keys = cache.keys
            attention_values = cache.values
        if (
            attention_cache_tokens is not None
            and attention_cache_tokens > 0
            and attention_cache_tokens < attention_keys.size(2)
        ):
            attention_keys = attention_keys[:, :, :attention_cache_tokens, :]
            attention_values = attention_values[:, :, :attention_cache_tokens, :]
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

    def forward_prefill_packed_eager(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor,
        flat_rows: Tensor,
        q_lens: Tensor,
        request_offsets: Tensor,
        packed_attention_groups: tuple[_PackedPrefillAttentionGroup, ...],
        next_norm_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_prefill_packed_eager(
            hidden,
            attn_in,
            rotary,
            cache,
            start_positions,
            write_positions,
            row_indices,
            flat_rows,
            q_lens,
            request_offsets,
            packed_attention_groups,
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

    def forward_prefill_packed_flashinfer(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: FlashInferLayerKVCache,
        write_positions: Tensor,
        flat_rows: Tensor,
        flashinfer_wrapper: object,
        next_norm_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_prefill_packed_flashinfer(
            hidden,
            attn_in,
            rotary,
            cache,
            write_positions,
            flat_rows,
            flashinfer_wrapper,
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

    def _attention_prefill_packed_eager(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor,
        flat_rows: Tensor,
        q_lens: Tensor,
        request_offsets: Tensor,
        packed_attention_groups: tuple[_PackedPrefillAttentionGroup, ...],
    ) -> Tensor:
        # Packed eager oracle for ragged prefill: projections run on the compact
        # real-token stream, while attention is sliced back per request so tokens
        # from different rows never attend to each other.
        batch, tokens, _ = hidden.shape
        if batch != 1:
            raise ValueError("packed ragged prefill expects a single flattened token row")
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        q, k = _apply_rotary_ragged_prefill(q, k, rotary)
        _append_ragged_kv_prefill(
            cache,
            k.permute(2, 1, 0, 3).contiguous(),
            v.permute(2, 1, 0, 3).contiguous(),
            write_positions.view(tokens, 1),
            flat_rows,
        )
        out = _packed_prefill_scaled_dot_product_attention(
            q,
            cache.keys,
            cache.values,
            start_positions,
            q_lens=q_lens,
            row_indices=row_indices,
            request_offsets=request_offsets,
            packed_attention_groups=packed_attention_groups,
            enable_gqa=enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)

    def _attention_prefill_packed_flashinfer(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: FlashInferLayerKVCache,
        write_positions: Tensor,
        flat_rows: Tensor,
        flashinfer_wrapper: object,
    ) -> Tensor:
        # True packed ragged prefill for FlashInfer caches: projections, MLP, and
        # collectives run on only the real suffix tokens, while FlashInfer handles
        # the per-request varlen causal attention plan.
        batch, tokens, _ = hidden.shape
        if batch != 1:
            raise ValueError("packed FlashInfer prefill expects a single flattened token row")
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
        q, k = _apply_rotary_ragged_prefill(q, k, rotary)
        _append_ragged_kv_prefill(
            cache,
            k.permute(2, 1, 0, 3).contiguous(),
            v.permute(2, 1, 0, 3).contiguous(),
            write_positions.view(tokens, 1),
            flat_rows,
        )
        paged_kv = getattr(cache, "paged_kv", None)
        if paged_kv is None:
            raise ValueError("packed FlashInfer prefill requires FlashInfer KV storage")
        q_packed = q.permute(0, 2, 1, 3).reshape(
            tokens,
            self.local_attention_heads,
            self.config.head_dim,
        )
        out_packed = flashinfer_wrapper.run(q_packed, paged_kv)
        out = out_packed.reshape(1, tokens, self.local_hidden_size)
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

    def _marlin_proj(
        self,
        hidden: Tensor,
        key: str,
        weight: Tensor,
        *,
        out: Tensor | None = None,
    ) -> Tensor | None:
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
        if not bool(getattr(self, "_runtime_marlin_int4_decode_enabled", True)):
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
        out_2d = out.reshape(-1, n) if out is not None else None
        out = _marlin.marlin_int4_mm(
            x2d, getattr(self, f"_marlin_{key}_q"), getattr(self, f"_marlin_{key}_s"),
            getattr(self, f"_marlin_{key}_ws"), n, k, out=out_2d,
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
        # few_shot's small-prefill regime. The broad env flag stays default-off. The
        # online server may opt in for deterministic 401-512 token sessions with a
        # higher runtime M-gate after live few_shot and multi_turn guards.
        env_configured = _tp_env_set("TORCHINFERNO_FP8_PREFILL")
        env_enabled = _tp_flag("TORCHINFERNO_FP8_PREFILL", False) if env_configured else False
        runtime_enabled = bool(getattr(self, "_runtime_fp8_prefill_enabled", False))
        if env_configured and not env_enabled:
            runtime_enabled = False
        if not env_enabled and not runtime_enabled:
            return None
        if not hidden.is_cuda:
            return None
        m = hidden.numel() // hidden.size(-1)
        if _tp_env_set("TORCHINFERNO_FP8_PREFILL_MIN_M"):
            min_m = _tp_int("TORCHINFERNO_FP8_PREFILL_MIN_M", 256, minimum=1)
        elif runtime_enabled and not env_enabled:
            min_m = int(getattr(self, "_runtime_fp8_prefill_min_m", 2048))
        else:
            min_m = 256
        if m <= max(1, min_m):
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
        gu_buffer = self._decode_scratch_buffer(
            "mlp-gate-up",
            hidden,
            self.gate_up_proj_weight.size(0),
        )
        gu = self._marlin_proj(hidden, "gu", self.gate_up_proj_weight, out=gu_buffer)
        if gu is None:
            gu = self._fp8_proj(hidden, "gu", self.gate_up_proj_weight)
        if gu is not None:
            gate, up = gu.split((self.local_intermediate_size, self.local_intermediate_size), dim=-1)
        else:
            gate, up = _decode_linear(hidden, self.gate_up_proj_weight, self.gate_up_proj_weight_decode).split(
                (self.local_intermediate_size, self.local_intermediate_size),
                dim=-1,
            )
        activation_buffer = self._decode_scratch_buffer(
            "mlp-activation",
            hidden,
            self.local_intermediate_size,
        )
        activated = _tp_decode_swiglu(gate, up, out=activation_buffer)
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

    def _decode_scratch_buffer(self, name: str, hidden: Tensor, width: int) -> Tensor | None:
        if hidden.ndim != 3 or hidden.size(1) != 1:
            return None
        if _cuda_stream_is_capturing(hidden.device):
            key = (name, hidden.device, hidden.dtype, (*hidden.shape[:-1], int(width)))
            return self._decode_scratch_buffers.get(key)
        expected_shape = (*hidden.shape[:-1], int(width))
        key = (name, hidden.device, hidden.dtype, expected_shape)
        buffer = self._decode_scratch_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(expected_shape, device=hidden.device, dtype=hidden.dtype)
            self._decode_scratch_buffers[key] = buffer
        return buffer

    def _mlp_project_prefill_reduce(self, hidden: Tensor) -> Tensor | None:
        if not _should_use_symm_mem_prefill_all_reduce(hidden, self.down_proj_weight, self.world_size):
            return None
        activated = self._profile_block(
            "fast_prefill.mlp_prefill.gate_up_activation",
            lambda: self._prefill_gate_up_activation(hidden),
        )
        return self._prefill_linear_all_reduce(activated, self.down_proj_weight, "mlp-prefill", fp8_key="down")

    def _mlp_project_fast_prefill(self, hidden: Tensor) -> Tensor:
        reduced = self._mlp_project_prefill_reduce(hidden)
        if reduced is not None:
            return reduced
        gu = self._fp8_proj(hidden, "gu", self.gate_up_proj_weight)
        if gu is None:
            projected = self._mlp_project_eager(hidden)
            _all_reduce(projected)
            return projected
        gate, up = gu.split((self.local_intermediate_size, self.local_intermediate_size), dim=-1)
        activated = _tp_swiglu(gate, up)
        fp8_down = self._fp8_proj(activated, "down", self.down_proj_weight)
        if fp8_down is not None:
            _all_reduce(fp8_down)
            return fp8_down
        projected = _decode_linear(activated, self.down_proj_weight, self.down_proj_weight_decode)
        _all_reduce(projected)
        return projected

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
        # complementary). In the symm-mem path, let Marlin write directly into the
        # reduce buffer so decode does not allocate an intermediate and copy it.
        use_sm = _should_use_symm_mem_all_reduce(hidden, weight, self.world_size)
        sm_name = buffer_name
        if not use_sm and _should_use_symm_mem_prefill_all_reduce(hidden, weight, self.world_size):
            # Eager prefill: bound distinct prefill shapes so per-shape ~0.5GB
            # symm-mem buffers cannot churn into OOM; new shapes past the cap fall
            # back to NCCL. During CUDA graph capture, only reuse a buffer that an
            # eager warmup already allocated and probed; rendezvous/allocation inside
            # capture is not legal.
            sm_name = f"{buffer_name}-pf"
            cap = _tp_int("TORCHINFERNO_SYMM_MEM_PREFILL_MAX_BUFFERS", 6, minimum=1)
            shape_key = (sm_name, tuple(hidden.shape[:-1]), int(weight.size(0)))
            if shape_key in _SYMM_PREFILL_SHAPES or len(_SYMM_PREFILL_SHAPES) < cap:
                expected_shape = (*hidden.shape[:-1], weight.size(0))
                use_sm = (
                    not _cuda_stream_is_capturing(hidden.device)
                    or self._symm_reduce_buffer_ready(sm_name, hidden, expected_shape)
                )
        if use_sm and not self._symm_reduce_failed:
            try:
                expected_shape = (*hidden.shape[:-1], weight.size(0))
                buffer, group_name = self._symm_reduce_buffer(sm_name, hidden, expected_shape)
                if sm_name != buffer_name:
                    _SYMM_PREFILL_SHAPES.add((sm_name, tuple(hidden.shape[:-1]), int(weight.size(0))))
                hidden_2d = hidden.reshape(-1, hidden.size(-1))
                output_2d = buffer.reshape(-1, weight.size(0))
                wrote_output = False
                if marlin_key is not None:
                    wrote_output = self._marlin_proj(hidden, marlin_key, weight, out=output_2d) is not None
                if not wrote_output:
                    fp8_out = self._fp8_proj(hidden, fp8_key, weight) if fp8_key is not None else None
                    if fp8_out is not None:
                        output_2d.copy_(fp8_out.reshape(-1, weight.size(0)))
                    elif weight_t is not None:
                        torch.mm(hidden_2d, weight_t, out=output_2d)
                    else:
                        torch.mm(hidden_2d, weight.t(), out=output_2d)
                torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
                return buffer
            except Exception:
                self._symm_reduce_failed = True
                _disable_symm_reduce()
        marlin_out = self._marlin_proj(hidden, marlin_key, weight) if marlin_key is not None else None
        if marlin_out is None and fp8_key is not None:
            marlin_out = self._fp8_proj(hidden, fp8_key, weight)
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
            if (
                _cuda_stream_is_capturing(hidden.device)
                and not self._symm_reduce_buffer_ready(buffer_name, hidden, expected_shape)
            ):
                return None
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

    def _symm_reduce_buffer_ready(self, name: str, hidden: Tensor, expected_shape: tuple[int, ...]) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return False
        group_name = dist.group.WORLD.group_name
        device_index = hidden.device.index if hidden.device.index is not None else torch.cuda.current_device()
        key = (group_name, device_index, name, str(hidden.dtype), expected_shape)
        return key in _SYMM_REDUCE_BUFFERS and key in _SYMM_REDUCE_PROBED

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
            tuple[int, int, int, int, bool, int],
            _StaticRaggedDecodeGraphCall,
        ] = {}
        self._ragged_decode_logits_graphs: dict[
            tuple[int, int, int, int, bool, int],
            _StaticRaggedDecodeLogitsGraphCall,
        ] = {}
        self._ragged_decode_many_graphs: dict[
            tuple[int, int, int, int, int, int],
            _StaticRaggedDecodeManyGraphCall,
        ] = {}
        self._decode_graph_failed = False
        self._decode_logits_graph_failed = False
        self._ragged_decode_graph_failed = False
        self._ragged_decode_logits_graph_failed = False
        self._ragged_decode_many_graph_failed = False
        self._ragged_prefill_logits_graphs: dict[
            tuple[object, ...],
            _StaticRaggedPrefillLogitsGraphCall,
        ] = {}
        self._packed_ragged_prefill_logits_graphs: dict[
            tuple[object, ...],
            _StaticPackedRaggedPrefillLogitsGraphCall,
        ] = {}
        self._packed_ragged_prefill_logits_graph_seen: dict[tuple[object, ...], int] = {}
        self._ragged_prefill_logits_graph_evictions = 0
        self._ragged_prefill_logits_graph_evicted_entries = 0
        self._ragged_prefill_logits_graph_max_entries = 0
        self._ragged_prefill_logits_graph_failed = False
        self._ragged_prefill_token_logits_graph_failed = False
        self._packed_ragged_prefill_logits_graph_failed = False
        self._ragged_prefill_mixed_logits_graph_failed = False
        self._ragged_prefill_capture_on_miss_failed = False
        self._temperature_gumbel_generators: dict[str, torch.Generator] = {}
        self._temperature_gumbel_scratch: Tensor | None = None
        self._temperature_sample_profile_ms: dict[str, float] = {}
        self._temperature_sample_profile_calls = 0
        self._temperature_sample_profile_rows = 0
        self._temperature_sample_gumbel_profile_calls = 0
        self._temperature_sample_gumbel_profile_rows = 0

    def set_runtime_fp8_prefill(self, enabled: bool, *, min_m: int = 2048) -> None:
        min_m = max(1, int(min_m))
        for layer in self.layers:
            setattr(layer, "_runtime_fp8_prefill_enabled", bool(enabled))
            setattr(layer, "_runtime_fp8_prefill_min_m", min_m)

    def set_runtime_marlin_int4_decode(self, enabled: bool) -> None:
        for layer in self.layers:
            setattr(layer, "_runtime_marlin_int4_decode_enabled", bool(enabled))

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
        load_start = time.perf_counter()
        checkpoint_broadcast = _rank0_checkpoint_broadcast_enabled(
            device=device,
            world_size=world_size,
            dtype=torch_dtype,
        )
        checkpoint_replicated_broadcast = _rank0_replicated_checkpoint_broadcast_enabled(
            device=device,
            world_size=world_size,
            dtype=torch_dtype,
        )
        checkpoint_replicated_page_cache_warm = _rank0_replicated_checkpoint_page_cache_warm_enabled(
            device=device,
            world_size=world_size,
            dtype=torch_dtype,
        )
        checkpoint_shard_scatter = _rank0_checkpoint_shard_scatter_enabled(
            device=device,
            world_size=world_size,
            dtype=torch_dtype,
        )
        checkpoint_direct_scatter = checkpoint_shard_scatter and _rank0_checkpoint_direct_scatter_enabled()
        if rank == 0:
            print(
                "[Llama3TP] loading checkpoint tensors "
                f"world_size={world_size} dtype={str(torch_dtype).replace('torch.', '')} "
                f"rank0_broadcast={int(checkpoint_broadcast)} "
                f"rank0_replicated_broadcast={int(checkpoint_replicated_broadcast)} "
                f"rank0_replicated_page_cache_warm={int(checkpoint_replicated_page_cache_warm)} "
                f"rank0_direct_scatter={int(checkpoint_direct_scatter)} "
                f"rank0_shard_scatter={int(checkpoint_shard_scatter)}",
                flush=True,
            )
            print("[Llama3TP] loading initial embedding/norm/head tensors", flush=True)
        embed_tokens_weight = _load_checkpoint_tensor(
            loader,
            "model.embed_tokens.weight",
            device=device,
            dtype=torch_dtype,
            rank=rank,
            world_size=world_size,
        )
        norm_weight = _load_checkpoint_tensor(
            loader,
            "model.norm.weight",
            device=device,
            dtype=torch_dtype,
            rank=rank,
            world_size=world_size,
        )
        lm_head_weight = _load_checkpoint_tensor_shard(
            loader,
            "lm_head.weight",
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=torch_dtype,
        )
        if rank == 0:
            print(
                f"[Llama3TP] loaded initial embedding/norm/head tensors "
                f"in {time.perf_counter() - load_start:.1f}s",
                flush=True,
            )

        layers: list[_Llama3TensorParallelLayer] = []
        for layer_id in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}."
            q_proj_weight = _load_checkpoint_tensor_shard(
                loader,
                prefix + "self_attn.q_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            k_proj_weight = _load_checkpoint_tensor_shard(
                loader,
                prefix + "self_attn.k_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            v_proj_weight = _load_checkpoint_tensor_shard(
                loader,
                prefix + "self_attn.v_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            gate_proj_weight = _load_checkpoint_tensor_shard(
                loader,
                prefix + "mlp.gate_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            up_proj_weight = _load_checkpoint_tensor_shard(
                loader,
                prefix + "mlp.up_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            weights = {
                "input_layernorm.weight": _load_checkpoint_tensor(
                    loader,
                    prefix + "input_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                    rank=rank,
                    world_size=world_size,
                ),
                "post_attention_layernorm.weight": _load_checkpoint_tensor(
                    loader,
                    prefix + "post_attention_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                    rank=rank,
                    world_size=world_size,
                ),
                "self_attn.qkv_proj.weight": torch.cat(
                    (q_proj_weight, k_proj_weight, v_proj_weight),
                    dim=0,
                ).contiguous(),
                "self_attn.o_proj.weight": _load_checkpoint_tensor_shard(
                    loader,
                    prefix + "self_attn.o_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "mlp.gate_up_proj.weight": torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous(),
                "mlp.down_proj.weight": _load_checkpoint_tensor_shard(
                    loader,
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
            if rank == 0 and ((layer_id + 1) % 10 == 0 or layer_id + 1 == config.num_hidden_layers):
                print(
                    f"[Llama3TP] loaded {layer_id + 1}/{config.num_hidden_layers} layers "
                    f"in {time.perf_counter() - load_start:.1f}s",
                    flush=True,
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
        if rank == 0:
            print(f"[Llama3TP] checkpoint load complete in {time.perf_counter() - load_start:.1f}s", flush=True)
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

    def _temperature_sample_profile_enabled(self) -> bool:
        cached = getattr(self, "_temperature_sample_profile_enabled_cached", None)
        if isinstance(cached, bool):
            return cached
        if _tp_env_set("TORCHINFERNO_TEMPERATURE_SAMPLE_PROFILE"):
            enabled = _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_PROFILE", False)
        else:
            enabled = bool(
                (
                    os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL")
                    or os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE")
                )
                and env_flag("TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS", False)
            )
        self._temperature_sample_profile_enabled_cached = enabled
        return enabled

    def _temperature_sample_counts_enabled(self) -> bool:
        cached = getattr(self, "_temperature_sample_counts_enabled_cached", None)
        if isinstance(cached, bool):
            return cached
        enabled = bool(
            os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL")
            or os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE")
        )
        self._temperature_sample_counts_enabled_cached = enabled
        return enabled

    def _record_temperature_sample_profile(
        self,
        *,
        rows: int,
        total_ms: float,
        max_ms: float,
        weights_ms: float,
        rank_ms: float,
        cdf_ms: float,
        reduce_ms: float,
        record_timings: bool = True,
    ) -> None:
        if record_timings:
            totals = getattr(self, "_temperature_sample_profile_ms", None)
            if not isinstance(totals, dict):
                totals = {}
                self._temperature_sample_profile_ms = totals
            for name, value in (
                ("total", total_ms),
                ("max", max_ms),
                ("weights", weights_ms),
                ("rank", rank_ms),
                ("cdf", cdf_ms),
                ("reduce", reduce_ms),
            ):
                totals[name] = float(totals.get(name, 0.0)) + float(value)
        self._temperature_sample_profile_calls = int(
            getattr(self, "_temperature_sample_profile_calls", 0)
        ) + 1
        self._temperature_sample_profile_rows = int(
            getattr(self, "_temperature_sample_profile_rows", 0)
        ) + max(0, int(rows))

    def _record_temperature_sample_gumbel_profile(
        self,
        *,
        rows: int,
        total_ms: float,
        noise_ms: float,
        max_ms: float,
        reduce_ms: float,
        record_timings: bool = True,
    ) -> None:
        if record_timings:
            totals = getattr(self, "_temperature_sample_profile_ms", None)
            if not isinstance(totals, dict):
                totals = {}
                self._temperature_sample_profile_ms = totals
            for name, value in (
                ("total", total_ms),
                ("gumbel_total", total_ms),
                ("gumbel_noise", noise_ms),
                ("gumbel_max", max_ms),
                ("gumbel_reduce", reduce_ms),
            ):
                totals[name] = float(totals.get(name, 0.0)) + float(value)
        self._temperature_sample_profile_calls = int(
            getattr(self, "_temperature_sample_profile_calls", 0)
        ) + 1
        self._temperature_sample_profile_rows = int(
            getattr(self, "_temperature_sample_profile_rows", 0)
        ) + max(0, int(rows))
        self._temperature_sample_gumbel_profile_calls = int(
            getattr(self, "_temperature_sample_gumbel_profile_calls", 0)
        ) + 1
        self._temperature_sample_gumbel_profile_rows = int(
            getattr(self, "_temperature_sample_gumbel_profile_rows", 0)
        ) + max(0, int(rows))

    def temperature_sample_profile_summary(self) -> dict[str, float | int]:
        calls = int(getattr(self, "_temperature_sample_profile_calls", 0))
        if calls <= 0:
            return {}
        totals = getattr(self, "_temperature_sample_profile_ms", None)
        if not isinstance(totals, dict):
            totals = {}
        rows = int(getattr(self, "_temperature_sample_profile_rows", 0))
        summary: dict[str, float | int] = {
            "temperature_sample_calls": calls,
            "temperature_sample_rows": rows,
        }
        for name in ("total", "max", "weights", "rank", "cdf", "reduce"):
            if name in totals:
                summary[f"temperature_sample_{name}_ms"] = float(totals.get(name, 0.0))
        gumbel_calls = int(getattr(self, "_temperature_sample_gumbel_profile_calls", 0))
        if gumbel_calls > 0:
            summary["temperature_sample_gumbel_calls"] = gumbel_calls
            summary["temperature_sample_gumbel_rows"] = int(
                getattr(self, "_temperature_sample_gumbel_profile_rows", 0)
            )
            if "gumbel_total" in totals:
                summary["temperature_sample_gumbel_ms"] = float(
                    totals.get("gumbel_total", 0.0)
                )
            if "gumbel_noise" in totals:
                summary["temperature_sample_gumbel_noise_ms"] = float(
                    totals.get("gumbel_noise", 0.0)
                )
            if "gumbel_max" in totals:
                summary["temperature_sample_gumbel_max_ms"] = float(
                    totals.get("gumbel_max", 0.0)
                )
            if "gumbel_reduce" in totals:
                summary["temperature_sample_gumbel_reduce_ms"] = float(
                    totals.get("gumbel_reduce", 0.0)
                )
        return summary

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
            memory_ok = self._trim_ragged_prefill_logits_graphs_for_memory()
            if capture_on_miss:
                memory_ok = _capture_succeeded_on_all_ranks(memory_ok, self.device)
            if not memory_ok:
                return None
            try:
                captured = self._capture_prefill_graph(input_ids, cache)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            max_graphs = _prefill_graph_max_graphs()
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
            memory_ok = self._trim_ragged_prefill_logits_graphs_for_memory()
            if capture_on_miss:
                memory_ok = _capture_succeeded_on_all_ranks(memory_ok, self.device)
            if not memory_ok:
                return None
            try:
                captured = self._capture_prefill_logits_graph(input_ids, cache, logit_positions)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
            max_graphs = _prefill_graph_max_graphs()
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
            memory_ok = self._trim_ragged_prefill_logits_graphs_for_memory()
            if capture_on_miss:
                memory_ok = _capture_succeeded_on_all_ranks(memory_ok, self.device)
            if not memory_ok:
                return None
            try:
                captured = self._capture_prefill_selected_logits_graph(input_ids, cache, logit_positions)
            except Exception:
                self._set_cache_seq_len(cache, initial_seq_len)
                raise
            real_end = initial_seq_len + prompt_tokens
            self._set_cache_seq_len(cache, real_end)
            max_graphs = _prefill_graph_max_graphs()
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
        self._last_ragged_decode_logits_graph_captured = None
        self._last_ragged_decode_logits_graph_key = None
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
            self._last_ragged_decode_logits_graph_captured = None
            self._last_ragged_decode_logits_graph_key = None
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
        self._last_ragged_decode_graph_captured = None
        self._last_ragged_decode_graph_key = None
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
            self._last_ragged_decode_graph_captured = None
            self._last_ragged_decode_graph_key = None
            return None

    def try_decode_ragged_token_graph_many(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        steps: int = 1,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        self._last_ragged_decode_many_graph_captured = None
        cache_backend = getattr(cache, "cache_backend", "dense")
        if (
            steps <= 1
            or cache_backend not in {"dense", "paged"}
            or getattr(cache, "_block_decode_graph_captures", False)
        ):
            return None
        if self._ragged_decode_many_graph_failed or not _should_use_ragged_decode_token_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            temperature,
        ):
            return None
        try:
            return self._run_ragged_decode_many_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                steps=int(steps),
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_many_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} ragged_decode_many_graph_failed={exc!r}", flush=True)
            self._ragged_decode_many_graph_failed = True
            self._last_ragged_decode_many_graph_captured = None
            return None

    def release_decode_graphs_for_cache(self, cache: Llama3TensorParallelCache) -> None:
        cache_ids = {id(cache), _cache_graph_root_id(cache)}
        for graph_map in (
            self._prefill_graphs,
            self._prefill_logits_graphs,
            getattr(self, "_prefill_selected_logits_graphs", {}),
            getattr(self, "_ragged_prefill_logits_graphs", {}),
            self._decode_graphs,
            self._decode_logits_graphs,
            getattr(self, "_ragged_decode_graphs", {}),
            self._ragged_decode_logits_graphs,
            getattr(self, "_ragged_decode_many_graphs", {}),
        ):
            for key, captured in list(graph_map.items()):
                if key[0] in cache_ids or getattr(captured, "cache", None) is cache:
                    graph_map.pop(key, None)

    def _evict_one_ragged_prefill_logits_graph(self, *, protected_key: tuple[object, ...] | None = None) -> bool:
        graphs = getattr(self, "_ragged_prefill_logits_graphs", None)
        if not isinstance(graphs, dict):
            return False
        evicted_key = next((candidate for candidate in graphs if candidate != protected_key), None)
        if evicted_key is None:
            return False
        self._ragged_prefill_logits_graph_evictions = int(
            getattr(self, "_ragged_prefill_logits_graph_evictions", 0)
        ) + 1
        self._ragged_prefill_logits_graph_evicted_entries = int(
            getattr(self, "_ragged_prefill_logits_graph_evicted_entries", 0)
        ) + 1
        del graphs[evicted_key]
        return True

    def _trim_ragged_prefill_logits_graphs_for_memory(
        self,
        *,
        protected_key: tuple[object, ...] | None = None,
    ) -> bool:
        graphs = getattr(self, "_ragged_prefill_logits_graphs", None)
        if not isinstance(graphs, dict):
            return True
        max_graphs = _prefill_graph_max_graphs()
        self._ragged_prefill_logits_graph_max_entries = max_graphs
        while len(graphs) > max_graphs:
            if not self._evict_one_ragged_prefill_logits_graph(protected_key=protected_key):
                break
        min_free_bytes = _prefill_graph_min_free_bytes()
        if min_free_bytes <= 0 or self.device.type != "cuda":
            return True
        while graphs:
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
            except Exception:
                return True
            if int(free_bytes) >= min_free_bytes:
                return True
            if not self._evict_one_ragged_prefill_logits_graph(protected_key=protected_key):
                return False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return True
        return int(free_bytes) >= min_free_bytes

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
        if needs_capture and not capture_on_miss:
            return None
        if capture_on_miss and not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
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
        if needs_capture and not capture_on_miss:
            return None
        if capture_on_miss and not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
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
        cache_token_bucket = _ragged_decode_cache_token_bucket(
            cache,
            seq_lens,
            row_indices,
            batch=input_ids.size(0),
        )
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_decode_graphs.get(key)
        self._last_ragged_decode_graph_key = key
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.cache_token_bucket != cache_token_bucket
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
        )
        if needs_capture and not capture_on_miss:
            self._last_ragged_decode_graph_captured = None
            self._last_ragged_decode_graph_key = None
            return None
        if capture_on_miss and not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            captured = self._capture_ragged_decode_graph(
                input_ids,
                cache,
                seq_lens,
                row_indices,
                cache_token_bucket=cache_token_bucket,
            )
        else:
            self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
            if not self._maybe_profile_ragged_decode_graph_replay_once(
                captured,
                input_ids,
                cache_token_bucket=cache_token_bucket,
                row_indices=row_indices,
            ):
                captured.graph.replay()
        self._last_ragged_decode_graph_captured = bool(needs_capture)
        _advance_paged_ragged_decode_cache_lengths(
            cache,
            batch=input_ids.size(0),
            cache_positions=captured.static_cache_positions,
            row_indices=captured.static_row_indices,
        )
        return captured.output_token

    def _run_ragged_decode_many_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        steps: int,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if not cache.layers:
            raise ValueError("ragged decode requires a non-empty KV cache")
        steps = max(1, int(steps))
        cache_token_bucket = _ragged_decode_cache_token_bucket(
            cache,
            seq_lens,
            row_indices,
            batch=input_ids.size(0),
        )
        selected = (
            seq_lens[: input_ids.size(0)]
            if row_indices is None
            else seq_lens.index_select(0, row_indices.to(device=seq_lens.device))
        )
        if selected.numel():
            needed = int(selected.max().item()) + steps
            if needed > cache_token_bucket:
                return None
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
            steps,
            row_indices is not None,
            _tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_MANY_ROTARY_IN_GRAPH", False),
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_decode_many_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.cache_token_bucket != cache_token_bucket
            or captured.static_input_ids.shape != input_ids.shape
            or captured.steps != steps
            or (captured.static_row_indices is None) != (row_indices is None)
        )
        if needs_capture and not capture_on_miss:
            self._last_ragged_decode_many_graph_captured = None
            return None
        if capture_on_miss and not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            captured = self._capture_ragged_decode_many_graph(
                input_ids,
                cache,
                seq_lens,
                row_indices,
                steps=steps,
                cache_token_bucket=cache_token_bucket,
            )
        else:
            self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
            if not self._maybe_profile_ragged_decode_many_graph_replay_once(
                captured,
                input_ids,
                cache_token_bucket=cache_token_bucket,
                row_indices=row_indices,
            ):
                captured.graph.replay()
        self._last_ragged_decode_many_graph_captured = bool(needs_capture)
        _advance_paged_ragged_decode_cache_lengths(
            cache,
            batch=input_ids.size(0),
            cache_positions=captured.static_cache_positions,
            row_indices=captured.static_row_indices,
            steps=steps,
        )
        return captured.output_tokens

    def _capture_ragged_decode_many_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        *,
        steps: int,
        cache_token_bucket: int,
    ) -> _StaticRaggedDecodeManyGraphCall:
        batch = input_ids.size(0)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_row_indices = torch.empty_like(row_indices) if row_indices is not None else None
        captured = _StaticRaggedDecodeManyGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_cache_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_row_indices=static_row_indices,
            static_rotary_cos=torch.empty((steps, batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_rotary_sin=torch.empty((steps, batch, rotary_cache_dim), device=self.device, dtype=self.dtype),
            output_tokens=torch.empty((steps, batch), device=self.device, dtype=torch.long),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            cache_token_bucket=cache_token_bucket,
            steps=int(steps),
            rotary_in_graph=_tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_MANY_ROTARY_IN_GRAPH", False),
        )
        self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        _set_paged_ragged_decode_graph_active(cache, True)
        try:
            with torch.cuda.stream(stream):
                self._forward_decode_ragged_many_static(captured)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(captured.graph):
                self._forward_decode_ragged_many_static(captured)
            captured.graph.replay()
        finally:
            _set_paged_ragged_decode_graph_active(cache, False)
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
            int(steps),
            row_indices is not None,
            bool(captured.rotary_in_graph),
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._ragged_decode_many_graphs and len(self._ragged_decode_many_graphs) >= max_graphs:
            self._ragged_decode_many_graphs.clear()
        self._ragged_decode_many_graphs[key] = captured
        return captured

    def _maybe_profile_ragged_decode_many_graph_replay_once(
        self,
        captured: _StaticRaggedDecodeManyGraphCall,
        input_ids: Tensor,
        *,
        cache_token_bucket: int,
        row_indices: Tensor | None,
    ) -> bool:
        if (
            not env_flag("TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_REPLAY_ONCE", False)
            or getattr(self, "_ragged_decode_many_replay_profiled", False)
            or input_ids.device.type != "cuda"
        ):
            return False
        min_batch = env_int("TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_MIN_BATCH", 32, minimum=1)
        if input_ids.size(0) < min_batch:
            return False
        configured_bucket = os.environ.get("TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_CACHE_BUCKET")
        if configured_bucket is not None and configured_bucket.strip():
            if int(configured_bucket) != int(cache_token_bucket):
                return False
        configured_steps = os.environ.get("TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_STEPS")
        if configured_steps is not None and configured_steps.strip():
            if int(configured_steps) != int(captured.steps):
                return False
        skip_matches = env_int(
            "TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_REPLAY_SKIP_MATCHES",
            0,
            minimum=0,
        )
        profile_matches = int(getattr(self, "_ragged_decode_many_replay_profile_matches", 0)) + 1
        self._ragged_decode_many_replay_profile_matches = profile_matches
        if profile_matches <= skip_matches:
            return False
        self._ragged_decode_many_replay_profiled = True
        rank = getattr(self, "rank", 0)
        if rank != 0:
            captured.graph.replay()
            return True
        replayed = False
        try:
            import sys as _rdmp
            from torch.profiler import ProfilerActivity as _PA
            from torch.profiler import profile as _tprof

            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as prof:
                captured.graph.replay()
                replayed = True
                torch.cuda.synchronize(self.device)
            rows = int(row_indices.numel()) if row_indices is not None else input_ids.size(0)
            row_limit = env_int("TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_ROW_LIMIT", 32, minimum=1)
            print(
                f"[RAGGED_DECODE_MANY_REPLAY_PROF] batch={input_ids.size(0)} "
                f"steps={captured.steps} "
                f"match={profile_matches} "
                f"cache_bucket={cache_token_bucket} "
                f"rows={rows}\n"
                + prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit),
                file=_rdmp.stderr,
                flush=True,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_many_replay_profile", exc)
            if not replayed:
                captured.graph.replay()
        return True

    def _forward_decode_ragged_many_static(
        self,
        captured: _StaticRaggedDecodeManyGraphCall,
    ) -> None:
        token_ids = captured.static_input_ids
        for step in range(captured.steps):
            cache_positions = (
                captured.static_cache_positions
                if step == 0
                else captured.static_cache_positions + step
            )
            self._copy_paged_ragged_decode_step_lengths(captured, cache_positions)
            rotary = (
                (
                    self.rotary_cos_cache.index_select(0, cache_positions),
                    self.rotary_sin_cache.index_select(0, cache_positions),
                )
                if captured.rotary_in_graph
                else (
                    captured.static_rotary_cos[step],
                    captured.static_rotary_sin[step],
                )
            )
            logits = self._forward_decode_ragged_static(
                token_ids,
                captured.cache,
                cache_positions,
                captured.static_row_indices,
                rotary,
                captured.cache_token_bucket,
            )
            next_token = self._sample_next_token(logits[:, -1, :], 0.0)
            captured.output_tokens[step].copy_(next_token)
            token_ids = next_token.view(next_token.size(0), 1)

    @staticmethod
    def _copy_paged_ragged_decode_step_lengths(
        captured: _StaticRaggedDecodeManyGraphCall,
        cache_positions: Tensor,
    ) -> None:
        seq_lens_buffers = getattr(captured, "static_paged_decode_seq_lens", None)
        if not seq_lens_buffers:
            return
        decode_lengths = cache_positions + 1
        copied: set[int] = set()
        for seq_lens in seq_lens_buffers:
            key = id(seq_lens)
            if key in copied:
                continue
            seq_lens.copy_(decode_lengths)
            copied.add(key)

    def _maybe_profile_ragged_decode_graph_replay_once(
        self,
        captured: _StaticRaggedDecodeGraphCall,
        input_ids: Tensor,
        *,
        cache_token_bucket: int,
        row_indices: Tensor | None,
    ) -> bool:
        if (
            not env_flag("TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_ONCE", False)
            or getattr(self, "_ragged_decode_replay_profiled", False)
            or input_ids.device.type != "cuda"
        ):
            return False
        min_batch = env_int("TORCHINFERNO_PROFILE_RAGGED_DECODE_MIN_BATCH", 32, minimum=1)
        if input_ids.size(0) < min_batch:
            return False
        configured_bucket = os.environ.get("TORCHINFERNO_PROFILE_RAGGED_DECODE_CACHE_BUCKET")
        if configured_bucket is not None and configured_bucket.strip():
            if int(configured_bucket) != int(cache_token_bucket):
                return False
        skip_matches = env_int(
            "TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_SKIP_MATCHES", 0, minimum=0
        )
        profile_matches = int(getattr(self, "_ragged_decode_replay_profile_matches", 0)) + 1
        self._ragged_decode_replay_profile_matches = profile_matches
        if profile_matches <= skip_matches:
            return False
        self._ragged_decode_replay_profiled = True
        rank = getattr(self, "rank", 0)
        if rank != 0:
            captured.graph.replay()
            return True
        replayed = False
        try:
            import sys as _rdp
            from torch.profiler import ProfilerActivity as _PA
            from torch.profiler import profile as _tprof

            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as prof:
                captured.graph.replay()
                replayed = True
                torch.cuda.synchronize(self.device)
            rows = int(row_indices.numel()) if row_indices is not None else input_ids.size(0)
            row_limit = env_int("TORCHINFERNO_PROFILE_RAGGED_DECODE_ROW_LIMIT", 24, minimum=1)
            print(
                f"[RAGGED_DECODE_REPLAY_PROF] batch={input_ids.size(0)} "
                f"match={profile_matches} "
                f"cache_bucket={cache_token_bucket} "
                f"rows={rows}\n"
                + prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit),
                file=_rdp.stderr,
                flush=True,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_decode_replay_profile", exc)
            if not replayed:
                captured.graph.replay()
        return True

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
        cache_token_bucket = _ragged_decode_cache_token_bucket(
            cache,
            seq_lens,
            row_indices,
            batch=input_ids.size(0),
        )
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
            row_indices is not None,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
        )
        captured = self._ragged_decode_logits_graphs.get(key)
        self._last_ragged_decode_logits_graph_key = key
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.cache_token_bucket != cache_token_bucket
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
        )
        if needs_capture and not capture_on_miss:
            self._last_ragged_decode_logits_graph_captured = None
            self._last_ragged_decode_logits_graph_key = None
            return None
        if capture_on_miss and not getattr(cache, "_skip_capture_sync", False):
            needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            captured = self._capture_ragged_decode_logits_graph(
                input_ids,
                cache,
                seq_lens,
                row_indices,
                cache_token_bucket=cache_token_bucket,
            )
        else:
            self._copy_ragged_decode_graph_inputs(captured, input_ids, seq_lens, row_indices)
            captured.graph.replay()
        self._last_ragged_decode_logits_graph_captured = bool(needs_capture)
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
        *,
        cache_token_bucket: int,
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
            cache_token_bucket=cache_token_bucket,
            rotary_in_graph=_tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_ROTARY_IN_GRAPH", True),
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
                    self._ragged_decode_graph_rotary(captured),
                    captured.cache_token_bucket,
                )
                self._sample_next_token(logits[:, -1, :], 0.0)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(captured.graph):
                logits = self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    self._ragged_decode_graph_rotary(captured),
                    captured.cache_token_bucket,
                )
                captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
            captured.graph.replay()
        finally:
            _set_paged_ragged_decode_graph_active(cache, False)
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
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
        *,
        cache_token_bucket: int,
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
            cache_token_bucket=cache_token_bucket,
            rotary_in_graph=_tp_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_ROTARY_IN_GRAPH", False),
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
                    self._ragged_decode_graph_rotary(captured),
                    captured.cache_token_bucket,
                )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(captured.graph):
                captured.output_logits = self._forward_decode_ragged_static(
                    captured.static_input_ids,
                    cache,
                    captured.static_cache_positions,
                    captured.static_row_indices,
                    self._ragged_decode_graph_rotary(captured),
                    captured.cache_token_bucket,
                )
            captured.graph.replay()
        finally:
            _set_paged_ragged_decode_graph_active(cache, False)
        key = (
            id(cache),
            input_ids.size(0),
            cache.layers[0].max_seq_len,
            cache_token_bucket,
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
        captured: (
            _StaticRaggedDecodeGraphCall
            | _StaticRaggedDecodeLogitsGraphCall
            | _StaticRaggedDecodeManyGraphCall
        ),
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
        if not getattr(captured, "rotary_in_graph", False):
            steps = int(getattr(captured, "steps", 1))
            if steps > 1:
                step_offsets = torch.arange(steps, device=self.device, dtype=cache_positions.dtype)
                positions = cache_positions.unsqueeze(0) + step_offsets[:, None]
                flat_positions = positions.reshape(-1)
                captured.static_rotary_cos.copy_(
                    self.rotary_cos_cache.index_select(0, flat_positions).view(
                        steps,
                        input_ids.size(0),
                        -1,
                    )
                )
                captured.static_rotary_sin.copy_(
                    self.rotary_sin_cache.index_select(0, flat_positions).view(
                        steps,
                        input_ids.size(0),
                        -1,
                    )
                )
            else:
                captured.static_rotary_cos.copy_(self.rotary_cos_cache.index_select(0, cache_positions))
                captured.static_rotary_sin.copy_(self.rotary_sin_cache.index_select(0, cache_positions))
        paged_state = _prepare_paged_ragged_decode_graph_state(
            captured.cache,
            batch=input_ids.size(0),
            cache_positions=captured.static_cache_positions,
            row_indices=captured.static_row_indices,
            device=self.device,
            cache_token_bucket=captured.cache_token_bucket,
            steps=int(getattr(captured, "steps", 1)),
            page_tables=captured.static_paged_decode_page_tables,
            seq_lens_buffers=captured.static_paged_decode_seq_lens,
        )
        if paged_state is not None:
            captured.static_paged_decode_page_tables = paged_state[0]
            captured.static_paged_decode_seq_lens = paged_state[1]

    def _ragged_decode_graph_rotary(
        self,
        captured: _StaticRaggedDecodeGraphCall | _StaticRaggedDecodeLogitsGraphCall,
    ) -> tuple[Tensor, Tensor]:
        if getattr(captured, "rotary_in_graph", False):
            positions = captured.static_cache_positions
            return (
                self.rotary_cos_cache.index_select(0, positions),
                self.rotary_sin_cache.index_select(0, positions),
            )
        return captured.static_rotary_cos, captured.static_rotary_sin

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
        attention_cache_tokens: int | None = None,
    ) -> Tensor:
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        attention_lengths = (
            None
            if callable(getattr(cache.layers[0], "append_and_attend_ragged", None))
            else cache_positions + 1
        )
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
                    attention_cache_tokens=attention_cache_tokens,
                    attention_lengths=attention_lengths,
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
        attention_cache_tokens = _ragged_decode_cache_token_bucket(
            cache,
            seq_lens,
            row_indices,
            batch=input_ids.size(0),
        )

        rotary = (
            self.rotary_cos_cache.index_select(0, cache_positions),
            self.rotary_sin_cache.index_select(0, cache_positions),
        )
        hidden = F.embedding(input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        attention_lengths = (
            None
            if callable(getattr(cache.layers[0], "append_and_attend_ragged", None))
            else cache_positions + 1
        )
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
                    attention_cache_tokens=attention_cache_tokens,
                    attention_lengths=attention_lengths,
                )
        finally:
            if context_set:
                _clear_paged_ragged_decode_context(cache)
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

    def _copy_ragged_prefill_prefix(
        self,
        cache: Llama3TensorParallelCache,
        *,
        input_tokens: int,
        start_positions: Tensor,
        row_indices: Tensor | None,
        context_len: int | None,
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None = None,
    ) -> None:
        # Fold the prefix KV broadcast INTO the graph: copy [0:prefix_len] from
        # each source row into its active row, per layer, via advanced indexing.
        # Captured once, replayed in one launch -- removes the ~80 per-layer
        # index_copy launches/batch that the engine used to issue eagerly.
        # src_prefix_row is a copied-in device tensor so replay re-targets source
        # rows; row_indices re-targets dest rows. A single source row broadcasts
        # for common-prefix reuse, and one source per request handles full-prompt
        # reuse with equal prefix lengths.
        if src_prefix_row is None:
            return
        dest_rows = (
            torch.arange(start_positions.numel(), device=self.device)
            if row_indices is None
            else row_indices
        )
        if context_len is not None:
            if context_len < 0:
                prefix_len = (-context_len) - input_tokens
            else:
                prefix_len = context_len - input_tokens
            if prefix_len > 0:
                for layer in cache.layers:
                    layer.keys[dest_rows, :, :prefix_len, :] = layer.keys.index_select(
                        0, src_prefix_row
                    )[:, :, :prefix_len, :]
                    layer.values[dest_rows, :, :prefix_len, :] = layer.values.index_select(
                        0, src_prefix_row
                    )[:, :, :prefix_len, :]
            return
        prefix_len = (
            int(prefix_copy_len)
            if prefix_copy_len is not None
            else int(start_positions.max().item()) if start_positions.numel() else 0
        )
        if prefix_len <= 0:
            return
        source_rows = src_prefix_row
        if source_rows.numel() == 1 and dest_rows.numel() > 1:
            source_rows = source_rows.expand(dest_rows.numel())
        mask = torch.arange(prefix_len, device=self.device)[None, :] < start_positions[:, None]
        for layer in cache.layers:
            source_keys = layer.keys.index_select(0, source_rows)[:, :, :prefix_len, :]
            source_values = layer.values.index_select(0, source_rows)[:, :, :prefix_len, :]
            zero_key = torch.zeros((), dtype=source_keys.dtype, device=source_keys.device)
            zero_value = torch.zeros((), dtype=source_values.dtype, device=source_values.device)
            layer.keys[dest_rows, :, :prefix_len, :] = torch.where(
                mask[:, None, :, None],
                source_keys,
                zero_key,
            )
            layer.values[dest_rows, :, :prefix_len, :] = torch.where(
                mask[:, None, :, None],
                source_values,
                zero_value,
            )

    def _forward_prefill_ragged_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        start_positions: Tensor,
        write_positions: Tensor,
        row_indices: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        logit_positions: Tensor | None,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        *,
        emit_logits: bool = True,
    ) -> Tensor:
        batch = input_ids.size(0)
        self._copy_ragged_prefill_prefix(
            cache,
            input_tokens=input_ids.size(1),
            start_positions=start_positions,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
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
        if not emit_logits:
            return attn_in.new_empty((0,))
        if logit_positions is None:
            raise ValueError("logit_positions are required when emit_logits is true")
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

    def _plan_packed_flashinfer_prefill(
        self,
        wrapper: object,
        cache: Llama3TensorParallelCache,
        *,
        start_positions: Tensor,
        q_lens: Tensor,
        row_indices: Tensor,
    ) -> None:
        # Host-side varlen plan for packed suffix prefill. Each dense cache row is
        # exposed to FlashInfer as one large page, matching forward_flashinfer.
        batch = q_lens.size(0)
        max_seq = cache.layers[0].max_seq_len
        qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        qo_indptr[1:] = q_lens.to(torch.int32).cumsum(0)
        paged_kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=self.device)
        paged_kv_indices = row_indices.to(dtype=torch.int32, device=self.device)
        paged_kv_last_page_len = (start_positions + q_lens).to(torch.int32)
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
        prefix_copy_len: int | None = None,
    ) -> Tensor:
        # Eager reference for the ragged-prefill graph: prefill a [batch, suffix]
        # block of tokens into scattered cache rows with per-row prefix offsets,
        # returning one (sharded) logit row per request at logit_positions. This
        # is the oracle the CUDA graph captures and the CPU test compares against.
        return self._prefill_ragged_impl(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            emit_logits=True,
        )

    @torch.inference_mode()
    def prefill_ragged_logits_packed_eager(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
    ) -> Tensor:
        # Dense eager oracle for a future packed ragged-prefill kernel. It skips
        # padded suffix columns entirely while preserving the padded oracle's KV
        # row/position contract and one-logit-per-request output contract.
        if input_ids.ndim != 2:
            raise ValueError("packed ragged prefill expects input_ids [batch, suffix_bucket]")
        if getattr(cache, "cache_backend", "dense") != "dense":
            raise ValueError("packed eager ragged prefill currently requires a dense KV cache")
        if not cache.layers:
            raise ValueError("packed ragged prefill requires a non-empty KV cache")
        batch, suffix_bucket = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        q_lens = q_lens.to(self.device, non_blocking=True)
        logit_positions = logit_positions.to(self.device, non_blocking=True)
        if q_lens.ndim != 1 or q_lens.numel() != batch:
            raise ValueError("q_lens must have shape [batch]")
        if logit_positions.ndim != 1 or logit_positions.numel() != batch:
            raise ValueError("logit_positions must have shape [batch]")
        if row_indices is not None:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if row_indices.ndim != 1 or row_indices.numel() != batch:
                raise ValueError("row_indices must have shape [batch]")
        else:
            row_indices = torch.arange(batch, dtype=torch.long, device=self.device)
        if src_prefix_row is not None:
            src_prefix_row = src_prefix_row.to(self.device, non_blocking=True)
        if bool(torch.any(q_lens <= 0)):
            raise ValueError("q_lens must be positive")
        if bool(torch.any(q_lens > suffix_bucket)):
            raise ValueError("q_lens cannot exceed the input suffix bucket")
        if bool(torch.any(logit_positions < 0)) or bool(torch.any(logit_positions >= q_lens)):
            raise ValueError("logit_positions must select real packed suffix tokens")
        start_positions = seq_lens.index_select(0, row_indices)
        if bool(torch.any(start_positions < 0)):
            raise ValueError("seq_lens must be non-negative")
        max_seq = cache.layers[0].max_seq_len
        if bool(torch.any(start_positions + q_lens > max_seq)):
            raise ValueError("KV cache capacity exceeded")
        q_lens_cpu = tuple(int(v) for v in q_lens.detach().cpu().tolist())
        start_positions_cpu = tuple(int(v) for v in start_positions.detach().cpu().tolist())
        request_offsets_cpu = _packed_prefill_offsets_from_q_lens(q_lens_cpu)
        request_offsets = torch.tensor(
            request_offsets_cpu,
            dtype=torch.long,
            device=self.device,
        )
        packed_attention_groups = _build_packed_prefill_attention_groups(
            q_lens_cpu,
            start_positions_cpu,
            request_offsets_cpu,
            device=self.device,
        )
        flat_token_indices, flat_request_indices = _packed_prefill_flat_indices(
            q_lens_cpu,
            suffix_bucket,
            device=self.device,
        )

        return self._prefill_ragged_logits_packed_eager_compute(
            input_ids,
            cache,
            start_positions=start_positions,
            q_lens=q_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            request_offsets=request_offsets,
            flat_token_indices=flat_token_indices,
            flat_request_indices=flat_request_indices,
            packed_attention_groups=packed_attention_groups,
        )

    def _prefill_ragged_logits_packed_eager_compute(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        start_positions: Tensor,
        q_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None,
        request_offsets: Tensor,
        flat_token_indices: Tensor,
        flat_request_indices: Tensor,
        packed_attention_groups: tuple[_PackedPrefillAttentionGroup, ...],
    ) -> Tensor:
        batch, suffix_bucket = input_ids.shape
        query_offsets = torch.arange(suffix_bucket, device=self.device)
        flat_input_ids = input_ids.reshape(-1).index_select(0, flat_token_indices).view(1, -1)
        total_tokens = flat_input_ids.size(1)
        if total_tokens <= 0:
            raise ValueError("packed ragged prefill requires at least one token")
        bucket_write_positions = start_positions[:, None] + query_offsets[None, :]
        flat_write_positions = bucket_write_positions.reshape(-1).index_select(
            0,
            flat_token_indices,
        )
        flat_rows = row_indices.index_select(0, flat_request_indices)

        self._copy_ragged_prefill_prefix(
            cache,
            input_tokens=suffix_bucket,
            start_positions=start_positions,
            row_indices=row_indices,
            context_len=None,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
        )
        rotary = (
            self.rotary_cos_cache.index_select(0, flat_write_positions).view(1, total_tokens, -1),
            self.rotary_sin_cache.index_select(0, flat_write_positions).view(1, total_tokens, -1),
        )
        hidden = F.embedding(flat_input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_prefill_packed_eager(
                hidden,
                attn_in,
                rotary,
                cache.layers[layer_id],
                start_positions,
                flat_write_positions,
                row_indices,
                flat_rows,
                q_lens,
                request_offsets,
                packed_attention_groups,
                next_norm_weight,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        flat_logit_positions = request_offsets + logit_positions
        gathered = attn_in.squeeze(0).index_select(0, flat_logit_positions).view(batch, 1, -1)
        return _decode_linear(gathered, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def try_prefill_ragged_logits_packed_eager_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if (
            self._packed_ragged_prefill_logits_graph_failed
            or input_ids.device.type != "cuda"
            or getattr(cache, "cache_backend", "dense") != "dense"
            or not cache.layers
        ):
            return None
        try:
            return self._run_packed_ragged_prefill_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                q_lens=q_lens,
                row_indices=row_indices,
                logit_positions=logit_positions,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.packed_ragged_prefill_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} packed_ragged_prefill_logits_graph_failed={exc!r}", flush=True)
            self._packed_ragged_prefill_logits_graph_failed = True
            return None

    def _run_packed_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        batch, suffix_bucket = input_ids.shape
        if batch <= 0 or suffix_bucket <= 0:
            return None
        q_lens = q_lens.to(self.device, non_blocking=True)
        logit_positions = logit_positions.to(self.device, non_blocking=True)
        if row_indices is None:
            row_indices = torch.arange(batch, dtype=torch.long, device=self.device)
        else:
            row_indices = row_indices.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        start_positions = seq_lens.index_select(0, row_indices)
        q_lens_key = tuple(int(value) for value in q_lens.detach().cpu().tolist())
        start_positions_key = tuple(
            int(value) for value in start_positions.detach().cpu().tolist()
        )
        logit_positions_key = tuple(
            int(value) for value in logit_positions.detach().cpu().tolist()
        )
        if not q_lens_key or all(q_len >= suffix_bucket for q_len in q_lens_key):
            return None
        if any(q_len <= 0 or q_len > suffix_bucket for q_len in q_lens_key):
            return None
        if logit_positions_key != tuple(q_len - 1 for q_len in q_lens_key):
            return None
        if any(start < 0 for start in start_positions_key):
            return None
        max_seq_len = cache.layers[0].max_seq_len
        if any(start + q_len > max_seq_len for start, q_len in zip(start_positions_key, q_lens_key)):
            return None
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        precision_key = _ragged_prefill_precision_graph_key(
            sum(q_lens_key),
            is_cuda=input_ids.is_cuda,
            layers=self.layers,
        )
        key = (
            id(cache),
            batch,
            suffix_bucket,
            max_seq_len,
            q_lens_key,
            start_positions_key,
            src_prefix_rows,
            prefix_copy_len if prefix_copy_len is not None else -1,
            precision_key,
            _symm_mem_allreduce_graph_key(1, _model_world_size(self)),
        )
        captured = self._packed_ragged_prefill_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or captured.q_lens_key != q_lens_key
            or captured.start_positions_key != start_positions_key
            or captured.prefix_copy_len != prefix_copy_len
            or (captured.static_src_prefix_row is None) != (src_prefix_row is None)
        )
        needs_capture = _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            if not capture_on_miss:
                return None
            if captured is None:
                min_capture_calls = _tp_int(
                    "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH_CAPTURE_MIN_CALLS",
                    2,
                    minimum=1,
                )
                seen = self._packed_ragged_prefill_logits_graph_seen.get(key, 0) + 1
                self._packed_ragged_prefill_logits_graph_seen[key] = seen
                if seen < min_capture_calls:
                    return None
            succeeded = True
            new_captured: _StaticPackedRaggedPrefillLogitsGraphCall | None = None
            try:
                new_captured = self._capture_packed_ragged_prefill_logits_graph(
                    input_ids,
                    cache,
                    start_positions=start_positions,
                    q_lens=q_lens,
                    row_indices=row_indices,
                    logit_positions=logit_positions,
                    src_prefix_row=src_prefix_row,
                    prefix_copy_len=prefix_copy_len,
                    q_lens_key=q_lens_key,
                    start_positions_key=start_positions_key,
                )
            except Exception:
                succeeded = False
            succeeded = _capture_succeeded_on_all_ranks(succeeded, self.device)
            if not succeeded or new_captured is None:
                return None
            captured = new_captured
            self._packed_ragged_prefill_logits_graphs[key] = captured
        else:
            self._copy_packed_ragged_prefill_graph_inputs(
                captured,
                input_ids,
                start_positions=start_positions,
                row_indices=row_indices,
                src_prefix_row=src_prefix_row,
            )
            captured.graph.replay()
        return captured.output_logits

    def _capture_packed_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        start_positions: Tensor,
        q_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None,
        q_lens_key: tuple[int, ...],
        start_positions_key: tuple[int, ...],
    ) -> _StaticPackedRaggedPrefillLogitsGraphCall:
        batch, suffix_bucket = input_ids.shape
        request_offsets_cpu = _packed_prefill_offsets_from_q_lens(q_lens_key)
        flat_token_indices, flat_request_indices = _packed_prefill_flat_indices(
            q_lens_key,
            suffix_bucket,
            device=self.device,
        )
        packed_attention_groups = _build_packed_prefill_attention_groups(
            q_lens_key,
            start_positions_key,
            request_offsets_cpu,
            device=self.device,
        )
        captured = _StaticPackedRaggedPrefillLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_start_positions=torch.empty((batch,), dtype=torch.long, device=self.device),
            static_q_lens=torch.tensor(q_lens_key, dtype=torch.long, device=self.device),
            static_row_indices=torch.empty((batch,), dtype=torch.long, device=self.device),
            static_logit_positions=torch.tensor(
                [q_len - 1 for q_len in q_lens_key],
                dtype=torch.long,
                device=self.device,
            ),
            static_src_prefix_row=(
                torch.empty_like(src_prefix_row) if src_prefix_row is not None else None
            ),
            static_request_offsets=torch.tensor(
                request_offsets_cpu,
                dtype=torch.long,
                device=self.device,
            ),
            static_flat_token_indices=flat_token_indices,
            static_flat_request_indices=flat_request_indices,
            packed_attention_groups=packed_attention_groups,
            output_logits=torch.empty(
                (batch, 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            suffix_bucket=suffix_bucket,
            q_lens_key=q_lens_key,
            start_positions_key=start_positions_key,
            prefix_copy_len=prefix_copy_len,
        )
        self._copy_packed_ragged_prefill_graph_inputs(
            captured,
            input_ids,
            start_positions=start_positions,
            row_indices=row_indices,
            src_prefix_row=src_prefix_row,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._prefill_ragged_logits_packed_eager_compute(
                captured.static_input_ids,
                cache,
                start_positions=captured.static_start_positions,
                q_lens=captured.static_q_lens,
                row_indices=captured.static_row_indices,
                logit_positions=captured.static_logit_positions,
                src_prefix_row=captured.static_src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                request_offsets=captured.static_request_offsets,
                flat_token_indices=captured.static_flat_token_indices,
                flat_request_indices=captured.static_flat_request_indices,
                packed_attention_groups=captured.packed_attention_groups,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._prefill_ragged_logits_packed_eager_compute(
                captured.static_input_ids,
                cache,
                start_positions=captured.static_start_positions,
                q_lens=captured.static_q_lens,
                row_indices=captured.static_row_indices,
                logit_positions=captured.static_logit_positions,
                src_prefix_row=captured.static_src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                request_offsets=captured.static_request_offsets,
                flat_token_indices=captured.static_flat_token_indices,
                flat_request_indices=captured.static_flat_request_indices,
                packed_attention_groups=captured.packed_attention_groups,
            )
        captured.graph.replay()
        return captured

    def _copy_packed_ragged_prefill_graph_inputs(
        self,
        captured: _StaticPackedRaggedPrefillLogitsGraphCall,
        input_ids: Tensor,
        *,
        start_positions: Tensor,
        row_indices: Tensor,
        src_prefix_row: Tensor | None,
    ) -> None:
        captured.static_input_ids.copy_(input_ids.to(self.device, non_blocking=True))
        captured.static_start_positions.copy_(
            start_positions.to(self.device, non_blocking=True)
        )
        captured.static_row_indices.copy_(row_indices.to(self.device, non_blocking=True))
        if captured.static_src_prefix_row is not None and src_prefix_row is not None:
            captured.static_src_prefix_row.copy_(
                src_prefix_row.to(self.device, non_blocking=True)
            )

    @torch.inference_mode()
    def prefill_ragged_logits_packed_flashinfer(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        q_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
    ) -> Tensor:
        # FlashInfer-backed packed ragged prefill. Unlike forward_step_flashinfer,
        # this removes padded suffix columns from the whole transformer block, not
        # just from the attention query tensor. It is intentionally a concrete
        # opt-in path so dense graph prefill remains the default reference.
        if input_ids.ndim != 2:
            raise ValueError("packed FlashInfer prefill expects input_ids [batch, suffix_bucket]")
        if getattr(cache, "cache_backend", "dense") != "flashinfer":
            raise ValueError("packed FlashInfer prefill requires a FlashInfer KV cache")
        if not cache.layers:
            raise ValueError("packed FlashInfer prefill requires a non-empty KV cache")
        batch, suffix_bucket = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        q_lens = q_lens.to(self.device, non_blocking=True)
        logit_positions = logit_positions.to(self.device, non_blocking=True)
        if q_lens.ndim != 1 or q_lens.numel() != batch:
            raise ValueError("q_lens must have shape [batch]")
        if logit_positions.ndim != 1 or logit_positions.numel() != batch:
            raise ValueError("logit_positions must have shape [batch]")
        if row_indices is not None:
            row_indices = row_indices.to(self.device, non_blocking=True)
            if row_indices.ndim != 1 or row_indices.numel() != batch:
                raise ValueError("row_indices must have shape [batch]")
            start_positions = seq_lens.index_select(0, row_indices)
        else:
            row_indices = torch.arange(batch, dtype=torch.long, device=self.device)
            start_positions = seq_lens[:batch]
            if start_positions.numel() != batch:
                raise ValueError("seq_lens must cover the ragged prefill batch")
        if src_prefix_row is not None:
            src_prefix_row = src_prefix_row.to(self.device, non_blocking=True)
        if bool(torch.any(q_lens <= 0)):
            raise ValueError("q_lens must be positive")
        if bool(torch.any(q_lens > suffix_bucket)):
            raise ValueError("q_lens cannot exceed the input suffix bucket")
        if bool(torch.any(logit_positions < 0)) or bool(torch.any(logit_positions >= q_lens)):
            raise ValueError("logit_positions must select real packed suffix tokens")
        if bool(torch.any(start_positions < 0)):
            raise ValueError("seq_lens must be non-negative")
        max_seq = cache.layers[0].max_seq_len
        if bool(torch.any(start_positions + q_lens > max_seq)):
            raise ValueError("KV cache capacity exceeded")

        query_offsets = torch.arange(suffix_bucket, device=self.device)
        real_token_mask = query_offsets[None, :] < q_lens[:, None]
        flat_input_ids = input_ids[real_token_mask].view(1, -1)
        total_tokens = flat_input_ids.size(1)
        if total_tokens <= 0:
            raise ValueError("packed FlashInfer prefill requires at least one token")
        bucket_write_positions = start_positions[:, None] + query_offsets[None, :]
        flat_write_positions = bucket_write_positions[real_token_mask]
        flat_rows = row_indices[:, None].expand(batch, suffix_bucket)[real_token_mask]
        request_offsets = torch.empty(batch, dtype=torch.long, device=self.device)
        request_offsets[0] = 0
        if batch > 1:
            request_offsets[1:] = torch.cumsum(q_lens[:-1], dim=0)

        self._copy_ragged_prefill_prefix(
            cache,
            input_tokens=suffix_bucket,
            start_positions=start_positions,
            row_indices=row_indices,
            context_len=None,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
        )
        wrapper = self._flashinfer_prefill_wrapper()
        self._plan_packed_flashinfer_prefill(
            wrapper,
            cache,
            start_positions=start_positions,
            q_lens=q_lens,
            row_indices=row_indices,
        )
        rotary = (
            self.rotary_cos_cache.index_select(0, flat_write_positions).view(1, total_tokens, -1),
            self.rotary_sin_cache.index_select(0, flat_write_positions).view(1, total_tokens, -1),
        )
        hidden = F.embedding(flat_input_ids, self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_prefill_packed_flashinfer(
                hidden,
                attn_in,
                rotary,
                cache.layers[layer_id],
                flat_write_positions,
                flat_rows,
                wrapper,
                next_norm_weight,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        flat_logit_positions = request_offsets + logit_positions
        gathered = attn_in.squeeze(0).index_select(0, flat_logit_positions).view(batch, 1, -1)
        return _decode_linear(gathered, self.lm_head_weight, self.lm_head_weight_decode)

    @torch.inference_mode()
    def prefill_ragged_cache(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
    ) -> bool:
        # No-logits variant for intermediate chunked prefill: populate KV for a
        # suffix chunk, but skip the final gather/lm_head projection because no
        # request emits a token until its last prompt chunk.
        self._prefill_ragged_impl(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=None,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            emit_logits=False,
        )
        return True

    def _prefill_ragged_impl(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor | None,
        context_len: int | None,
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None,
        emit_logits: bool,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("ragged prefill expects input_ids [batch, suffix]")
        if not cache.layers:
            raise ValueError("ragged prefill requires a non-empty KV cache")
        batch, suffix = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        if logit_positions is not None:
            logit_positions = logit_positions.to(self.device, non_blocking=True)
            if logit_positions.ndim != 1 or logit_positions.numel() != batch:
                raise ValueError("logit_positions must have shape [batch]")
        elif emit_logits:
            raise ValueError("logit_positions are required when emit_logits is true")
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
        if logit_positions is not None:
            if bool(torch.any(start_positions + logit_positions >= max_seq)):
                raise ValueError("KV cache capacity exceeded")
        elif bool(torch.any(start_positions + max(0, suffix - 1) >= max_seq)):
            raise ValueError("KV cache capacity exceeded")
        query_offsets = torch.arange(suffix, device=self.device)
        write_positions = (start_positions[:, None] + query_offsets[None, :]).clamp(max=max_seq - 1)
        rotary = (
            self.rotary_cos_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1),
            self.rotary_sin_cache.index_select(0, write_positions.reshape(-1)).view(batch, suffix, -1),
        )
        if src_prefix_row is not None:
            src_prefix_row = src_prefix_row.to(self.device, non_blocking=True)
        if emit_logits and logit_positions is not None:
            profiled = self._maybe_profile_ragged_prefill_once(
                input_ids,
                cache,
                start_positions,
                write_positions,
                row_indices,
                rotary,
                logit_positions,
                context_len,
                src_prefix_row,
                prefix_copy_len,
            )
            if profiled is not None:
                return profiled
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
            prefix_copy_len,
            emit_logits=emit_logits,
        )

    @staticmethod
    def _is_mixed_prefix_ragged_prefill_graph(
        *,
        context_len: int | None,
        src_prefix_rows: int,
        prefix_copy_len: int | None,
    ) -> bool:
        return context_len is None and prefix_copy_len is not None and src_prefix_rows > 1

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
        prefix_copy_len: int | None = None,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        self._last_ragged_prefill_graph_captured = False
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        if getattr(self, "_ragged_prefill_mixed_logits_graph_failed", False) and (
            self._is_mixed_prefix_ragged_prefill_graph(
                context_len=context_len,
                src_prefix_rows=src_prefix_rows,
                prefix_copy_len=prefix_copy_len,
            )
        ):
            return None
        if getattr(self, "_ragged_prefill_capture_on_miss_failed", False):
            capture_on_miss = False
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
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} ragged_prefill_logits_graph_failed={exc!r}", flush=True)
            self._ragged_prefill_logits_graph_failed = True
            return None

    def try_prefill_ragged_token_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> tuple[Tensor, Tensor] | None:
        if float(temperature) > 0.0:
            return None
        if getattr(self, "_ragged_prefill_token_logits_graph_failed", False):
            return None
        self._last_ragged_prefill_graph_captured = False
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        if getattr(self, "_ragged_prefill_mixed_logits_graph_failed", False) and (
            self._is_mixed_prefix_ragged_prefill_graph(
                context_len=context_len,
                src_prefix_rows=src_prefix_rows,
                prefix_copy_len=prefix_copy_len,
            )
        ):
            return None
        if getattr(self, "_ragged_prefill_capture_on_miss_failed", False):
            capture_on_miss = False
        if self._ragged_prefill_logits_graph_failed or not _should_use_ragged_prefill_logits_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            logit_positions,
        ):
            return None
        try:
            output = self._run_ragged_prefill_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                logit_positions=logit_positions,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
                emit_tokens=True,
            )
            if not isinstance(output, tuple):
                return None
            return output
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_token_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} ragged_prefill_token_logits_graph_failed={exc!r}", flush=True)
            self._ragged_prefill_token_logits_graph_failed = True
            return None

    def try_prefill_ragged_token_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if float(temperature) > 0.0:
            return None
        if getattr(self, "_ragged_prefill_token_logits_graph_failed", False):
            return None
        self._last_ragged_prefill_graph_captured = False
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        if getattr(self, "_ragged_prefill_mixed_logits_graph_failed", False) and (
            self._is_mixed_prefix_ragged_prefill_graph(
                context_len=context_len,
                src_prefix_rows=src_prefix_rows,
                prefix_copy_len=prefix_copy_len,
            )
        ):
            return None
        if getattr(self, "_ragged_prefill_capture_on_miss_failed", False):
            capture_on_miss = False
        if self._ragged_prefill_logits_graph_failed or not _should_use_ragged_prefill_logits_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            logit_positions,
        ):
            return None
        try:
            output = self._run_ragged_prefill_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                logit_positions=logit_positions,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
                emit_logits=False,
                emit_tokens=True,
            )
            if isinstance(output, tuple):
                return output[1]
            return output
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_token_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} ragged_prefill_token_graph_failed={exc!r}", flush=True)
            self._ragged_prefill_token_logits_graph_failed = True
            return None

    def try_prefill_ragged_cache_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None = None,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        capture_on_miss: bool = True,
    ) -> bool | None:
        self._last_ragged_prefill_graph_captured = False
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        if getattr(self, "_ragged_prefill_mixed_logits_graph_failed", False) and (
            self._is_mixed_prefix_ragged_prefill_graph(
                context_len=context_len,
                src_prefix_rows=src_prefix_rows,
                prefix_copy_len=prefix_copy_len,
            )
        ):
            return None
        if getattr(self, "_ragged_prefill_capture_on_miss_failed", False):
            capture_on_miss = False
        if self._ragged_prefill_logits_graph_failed or not _should_use_ragged_prefill_graph(
            input_ids,
            cache,
            seq_lens,
            row_indices,
            None,
        ):
            return None
        try:
            output = self._run_ragged_prefill_logits_graph(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
                logit_positions=None,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
                emit_logits=False,
            )
            return True if output is not None else None
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_cache_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} ragged_prefill_cache_graph_failed={exc!r}", flush=True)
            self._ragged_prefill_logits_graph_failed = True
            return None

    def _run_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor | None,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        capture_on_miss: bool = True,
        emit_logits: bool = True,
        emit_tokens: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | None:
        if not cache.layers:
            raise ValueError("ragged prefill requires a non-empty KV cache")
        src_prefix_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        mixed_prefix_graph = self._is_mixed_prefix_ragged_prefill_graph(
            context_len=context_len,
            src_prefix_rows=src_prefix_rows,
            prefix_copy_len=prefix_copy_len,
        )
        precision_key = _ragged_prefill_precision_graph_key(
            input_ids.numel(),
            is_cuda=input_ids.is_cuda,
            layers=self.layers,
        )
        rotary_in_graph = _tp_flag(
            "TORCHINFERNO_CUDAGRAPH_RAGGED_PREFILL_ROTARY_IN_GRAPH",
            True,
        )
        write_positions_in_graph = _tp_flag(
            "TORCHINFERNO_CUDAGRAPH_RAGGED_PREFILL_WRITE_POSITIONS_IN_GRAPH",
            True,
        )
        key = (
            id(cache),
            input_ids.size(0),
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            row_indices is not None,
            context_len if context_len is not None else -1,
            prefix_copy_len if prefix_copy_len is not None else -1,
            src_prefix_rows,
            precision_key,
            _symm_mem_allreduce_graph_key(input_ids.size(0), _model_world_size(self)),
            int(bool(emit_logits)),
            int(bool(emit_tokens)),
            int(bool(rotary_in_graph)),
            int(bool(write_positions_in_graph)),
        )
        memory_ok = self._trim_ragged_prefill_logits_graphs_for_memory(protected_key=key)
        captured = self._ragged_prefill_logits_graphs.get(key)
        needs_capture = (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
            or (captured.static_row_indices is None) != (row_indices is None)
            or captured.emit_logits != bool(emit_logits)
            or captured.emit_tokens != bool(emit_tokens)
            or captured.context_len != context_len
            or captured.prefix_copy_len != prefix_copy_len
            or captured.rotary_in_graph != bool(rotary_in_graph)
            or captured.write_positions_in_graph != bool(write_positions_in_graph)
            or (captured.static_src_prefix_row is None) != (src_prefix_row is None)
            or (captured.static_logit_positions is None) != (logit_positions is None)
            or (
                captured.static_src_prefix_row is not None
                and src_prefix_row is not None
                and captured.static_src_prefix_row.shape != src_prefix_row.shape
            )
            or (
                captured.static_logit_positions is not None
                and logit_positions is not None
                and captured.static_logit_positions.shape != logit_positions.shape
            )
        )
        skip_sync = bool(getattr(cache, "_skip_capture_sync", False))
        needs_capture = needs_capture if skip_sync else _capture_needed_on_any_rank(needs_capture, self.device)
        if needs_capture:
            if not capture_on_miss:
                return None
            if not skip_sync:
                memory_ok = _capture_succeeded_on_all_ranks(memory_ok, self.device)
            if not memory_ok:
                return None
            succeeded = True
            new_captured: _StaticRaggedPrefillLogitsGraphCall | None = None
            try:
                new_captured = self._capture_ragged_prefill_logits_graph(
                    input_ids,
                    cache,
                    seq_lens,
                    row_indices,
                    logit_positions,
                    context_len,
                    src_prefix_row,
                    prefix_copy_len,
                    emit_logits,
                    emit_tokens,
                    rotary_in_graph,
                    write_positions_in_graph,
                )
            except Exception:
                succeeded = False
            if not skip_sync:
                succeeded = _capture_succeeded_on_all_ranks(succeeded, self.device)
            if not succeeded or new_captured is None:
                if emit_tokens:
                    self._ragged_prefill_token_logits_graph_failed = True
                    return None
                if mixed_prefix_graph:
                    self._ragged_prefill_mixed_logits_graph_failed = True
                    self._ragged_prefill_capture_on_miss_failed = True
                    exc = RuntimeError("mixed-prefix ragged prefill graph capture failed on at least one rank")
                    warn_optional_failure("llama3_tensor_parallel.ragged_prefill_mixed_logits_graph", exc)
                    if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                        print(f"rank={self.rank} ragged_prefill_mixed_logits_graph_failed={exc!r}", flush=True)
                    return None
                raise RuntimeError("ragged prefill graph capture failed on at least one rank")
            max_graphs = _prefill_graph_max_graphs()
            self._ragged_prefill_logits_graph_max_entries = max_graphs
            if (
                key not in self._ragged_prefill_logits_graphs
                and len(self._ragged_prefill_logits_graphs) >= max_graphs
            ):
                self._evict_one_ragged_prefill_logits_graph(protected_key=key)
            self._ragged_prefill_logits_graphs[key] = new_captured
            self._trim_ragged_prefill_logits_graphs_for_memory(protected_key=key)
            captured = new_captured
            self._last_ragged_prefill_graph_captured = True
        else:
            self._ragged_prefill_logits_graphs.pop(key, None)
            self._ragged_prefill_logits_graphs[key] = captured
            self._copy_ragged_prefill_graph_inputs(
                captured, input_ids, seq_lens, row_indices, logit_positions, src_prefix_row
            )
            if self._maybe_profile_ragged_prefill_graph_replay_once(
                captured,
                input_ids,
                context_len,
                src_prefix_row,
                prefix_copy_len,
            ):
                return self._ragged_prefill_graph_output(captured, emit_tokens=emit_tokens)
            captured.graph.replay()
        return self._ragged_prefill_graph_output(captured, emit_tokens=emit_tokens)

    @staticmethod
    def _ragged_prefill_graph_output(
        captured: _StaticRaggedPrefillLogitsGraphCall,
        *,
        emit_tokens: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not emit_tokens:
            return captured.output_logits
        if captured.output_token is None:
            raise RuntimeError("ragged prefill token graph did not produce tokens")
        if not captured.emit_logits:
            return captured.output_token
        return captured.output_logits, captured.output_token

    def _maybe_profile_ragged_prefill_graph_replay_once(
        self,
        captured: _StaticRaggedPrefillLogitsGraphCall,
        input_ids: Tensor,
        context_len: int | None,
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None,
    ) -> bool:
        if (
            not env_flag("TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE", False)
            or getattr(self, "_ragged_prefill_replay_profiled", False)
            or input_ids.device.type != "cuda"
        ):
            return False
        min_batch = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH", 32, minimum=1)
        if input_ids.size(0) < min_batch:
            return False
        min_suffix = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX", 1, minimum=1)
        if input_ids.size(1) < min_suffix:
            return False
        if not self._ragged_prefill_profile_context_matches(context_len):
            return False
        skip_matches = env_int(
            "TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_SKIP_MATCHES", 0, minimum=0
        )
        profile_matches = int(getattr(self, "_ragged_prefill_replay_profile_matches", 0)) + 1
        self._ragged_prefill_replay_profile_matches = profile_matches
        if profile_matches <= skip_matches:
            return False
        self._ragged_prefill_replay_profiled = True
        rank = getattr(self, "rank", 0)
        if rank != 0:
            captured.graph.replay()
            return True
        replayed = False
        try:
            import sys as _rpp
            from torch.profiler import ProfilerActivity as _PA
            from torch.profiler import profile as _tprof

            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as prof:
                captured.graph.replay()
                replayed = True
                torch.cuda.synchronize(self.device)
            src_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
            row_limit = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_ROW_LIMIT", 24, minimum=1)
            print(
                f"[RAGGED_PREFILL_REPLAY_PROF] batch={input_ids.size(0)} "
                f"suffix={input_ids.size(1)} "
                f"match={profile_matches} "
                f"context_len={context_len if context_len is not None else 'none'} "
                f"src_rows={src_rows} "
                f"prefix_copy_len={prefix_copy_len if prefix_copy_len is not None else 'none'}\n"
                + prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit),
                file=_rpp.stderr,
                flush=True,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_replay_profile", exc)
            if not replayed:
                captured.graph.replay()
        return True

    def _capture_ragged_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor | None,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        emit_logits: bool = True,
        emit_tokens: bool = False,
        rotary_in_graph: bool = False,
        write_positions_in_graph: bool = False,
    ) -> _StaticRaggedPrefillLogitsGraphCall:
        batch, suffix = input_ids.shape
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_row_indices = torch.empty_like(row_indices) if row_indices is not None else None
        static_src_prefix_row = torch.empty_like(src_prefix_row) if src_prefix_row is not None else None
        static_logit_positions = torch.empty_like(logit_positions) if logit_positions is not None else None
        captured = _StaticRaggedPrefillLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=torch.empty_like(input_ids),
            static_start_positions=torch.empty((batch,), device=self.device, dtype=torch.int64),
            static_write_positions=torch.empty((batch, suffix), device=self.device, dtype=torch.int64),
            static_query_offsets=torch.arange(suffix, device=self.device, dtype=torch.int64),
            static_row_indices=static_row_indices,
            static_rotary_cos=torch.empty((batch, suffix, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_rotary_sin=torch.empty((batch, suffix, rotary_cache_dim), device=self.device, dtype=self.dtype),
            static_logit_positions=static_logit_positions,
            static_src_prefix_row=static_src_prefix_row,
            output_logits=(
                torch.empty((batch, 1, self.local_vocab_size), device=self.device, dtype=self.dtype)
                if emit_logits
                else torch.empty((0,), device=self.device, dtype=self.dtype)
            ),
            output_token=(
                torch.empty((batch,), device=self.device, dtype=torch.long)
                if emit_tokens
                else None
            ),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            suffix_bucket=suffix,
            context_len=context_len,
            prefix_copy_len=prefix_copy_len,
            emit_logits=bool(emit_logits),
            emit_tokens=bool(emit_tokens),
            rotary_in_graph=bool(rotary_in_graph),
            write_positions_in_graph=bool(write_positions_in_graph),
        )
        self._copy_ragged_prefill_graph_inputs(
            captured, input_ids, seq_lens, row_indices, logit_positions, src_prefix_row
        )
        if (emit_logits or emit_tokens) and captured.static_logit_positions is not None:
            write_positions = self._ragged_prefill_graph_write_positions(captured)
            self._maybe_profile_ragged_prefill_once(
                captured.static_input_ids,
                cache,
                captured.static_start_positions,
                write_positions,
                captured.static_row_indices,
                self._ragged_prefill_graph_rotary(captured, write_positions),
                captured.static_logit_positions,
                context_len,
                captured.static_src_prefix_row,
                prefix_copy_len,
            )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            write_positions = self._ragged_prefill_graph_write_positions(captured)
            logits = self._forward_prefill_ragged_static(
                captured.static_input_ids,
                cache,
                captured.static_start_positions,
                write_positions,
                captured.static_row_indices,
                self._ragged_prefill_graph_rotary(captured, write_positions),
                captured.static_logit_positions,
                context_len,
                captured.static_src_prefix_row,
                prefix_copy_len,
                emit_logits=emit_logits or emit_tokens,
            )
            if emit_tokens:
                self._sample_next_token(logits[:, -1, :], 0.0)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            write_positions = self._ragged_prefill_graph_write_positions(captured)
            logits = self._forward_prefill_ragged_static(
                captured.static_input_ids,
                cache,
                captured.static_start_positions,
                write_positions,
                captured.static_row_indices,
                self._ragged_prefill_graph_rotary(captured, write_positions),
                captured.static_logit_positions,
                context_len,
                captured.static_src_prefix_row,
                prefix_copy_len,
                emit_logits=emit_logits or emit_tokens,
            )
            if emit_logits:
                captured.output_logits = logits
            if emit_tokens:
                captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
        captured.graph.replay()
        return captured

    def _maybe_profile_ragged_prefill_once(
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
        prefix_copy_len: int | None = None,
    ) -> Tensor | None:
        if (
            not env_flag("TORCHINFERNO_PROFILE_RAGGED_PREFILL_ONCE", False)
            or getattr(self, "_ragged_prefill_profiled", False)
            or input_ids.device.type != "cuda"
        ):
            return None
        min_batch = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH", 32, minimum=1)
        if input_ids.size(0) < min_batch:
            return None
        min_suffix = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX", 1, minimum=1)
        if input_ids.size(1) < min_suffix:
            return None
        if not self._ragged_prefill_profile_context_matches(context_len):
            return None
        skip_matches = env_int(
            "TORCHINFERNO_PROFILE_RAGGED_PREFILL_SKIP_MATCHES", 0, minimum=0
        )
        profile_matches = int(getattr(self, "_ragged_prefill_profile_matches", 0)) + 1
        self._ragged_prefill_profile_matches = profile_matches
        if profile_matches <= skip_matches:
            return None
        self._ragged_prefill_profiled = True
        rank = getattr(self, "rank", 0)
        if rank != 0:
            # The profiled body contains TP collectives. Every rank must execute
            # the same extra forward; only rank 0 pays for profiler collection.
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
                prefix_copy_len,
            )
        try:
            import sys as _rpp
            from torch.profiler import ProfilerActivity as _PA
            from torch.profiler import profile as _tprof

            torch.cuda.synchronize(self.device)
            with _tprof(activities=[_PA.CPU, _PA.CUDA]) as prof:
                output = self._forward_prefill_ragged_static(
                    input_ids,
                    cache,
                    start_positions,
                    write_positions,
                    row_indices,
                    rotary,
                    logit_positions,
                    context_len,
                    src_prefix_row,
                    prefix_copy_len,
                )
                torch.cuda.synchronize(self.device)
            src_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
            row_limit = env_int("TORCHINFERNO_PROFILE_RAGGED_PREFILL_ROW_LIMIT", 24, minimum=1)
            print(
                f"[RAGGED_PREFILL_PROF] batch={input_ids.size(0)} "
                f"suffix={input_ids.size(1)} "
                f"match={profile_matches} "
                f"context_len={context_len if context_len is not None else 'none'} "
                f"src_rows={src_rows} "
                f"prefix_copy_len={prefix_copy_len if prefix_copy_len is not None else 'none'}\n"
                + prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit),
                file=_rpp.stderr,
                flush=True,
            )
            return output
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.ragged_prefill_profile", exc)
            return None

    @staticmethod
    def _ragged_prefill_profile_context_matches(context_len: int | None) -> bool:
        configured = os.environ.get("TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN")
        if configured is None or not configured.strip():
            return True
        return context_len is not None and context_len == int(configured)

    def _copy_ragged_prefill_graph_inputs(
        self,
        captured: _StaticRaggedPrefillLogitsGraphCall,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        logit_positions: Tensor | None,
        src_prefix_row: Tensor | None = None,
    ) -> None:
        batch, suffix = input_ids.shape
        input_ids = input_ids.to(self.device, non_blocking=True)
        seq_lens = seq_lens.to(self.device, non_blocking=True)
        if logit_positions is not None:
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
        write_positions: Tensor | None = None

        def derived_write_positions() -> Tensor:
            nonlocal write_positions
            if write_positions is None:
                query_offsets = torch.arange(suffix, device=self.device, dtype=start_positions.dtype)
                write_positions = (start_positions[:, None] + query_offsets[None, :]).clamp(
                    max=captured.max_seq_len - 1
                )
            return write_positions

        captured.static_input_ids.copy_(input_ids)
        captured.static_start_positions.copy_(start_positions)
        if not getattr(captured, "write_positions_in_graph", False):
            captured.static_write_positions.copy_(derived_write_positions())
        if captured.static_logit_positions is not None:
            if logit_positions is None:
                raise RuntimeError("captured ragged prefill graph requires logit positions")
            captured.static_logit_positions.copy_(logit_positions)
        if not getattr(captured, "rotary_in_graph", False):
            positions = derived_write_positions()
            captured.static_rotary_cos.copy_(
                self.rotary_cos_cache.index_select(0, positions.reshape(-1)).view(batch, suffix, -1)
            )
            captured.static_rotary_sin.copy_(
                self.rotary_sin_cache.index_select(0, positions.reshape(-1)).view(batch, suffix, -1)
            )

    def _ragged_prefill_graph_write_positions(
        self,
        captured: _StaticRaggedPrefillLogitsGraphCall,
    ) -> Tensor:
        if getattr(captured, "write_positions_in_graph", False):
            return (captured.static_start_positions[:, None] + captured.static_query_offsets[None, :]).clamp(
                max=captured.max_seq_len - 1
            )
        return captured.static_write_positions

    def _ragged_prefill_graph_rotary(
        self,
        captured: _StaticRaggedPrefillLogitsGraphCall,
        write_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if getattr(captured, "rotary_in_graph", False):
            if write_positions is None:
                write_positions = self._ragged_prefill_graph_write_positions(captured)
            batch, suffix = write_positions.shape
            positions = write_positions.reshape(-1)
            return (
                self.rotary_cos_cache.index_select(0, positions).view(batch, suffix, -1),
                self.rotary_sin_cache.index_select(0, positions).view(batch, suffix, -1),
            )
        return captured.static_rotary_cos, captured.static_rotary_sin

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

    def prepare_repeated_next_token_state(
        self,
        logits: Tensor,
        temperature: float,
    ) -> _RepeatedTemperatureSampleState | None:
        if temperature <= 0.0 or logits.size(0) != 1:
            return None
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        if dist.is_available() and dist.is_initialized():
            global_max = local_max.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
            weights = torch.exp(logits_float - global_max[:, None])
            local_sum = weights.sum(dim=-1)
            gathered_sums = torch.empty(
                (self.world_size, *local_sum.shape),
                dtype=local_sum.dtype,
                device=logits.device,
            )
            dist.all_gather_into_tensor(gathered_sums, local_sum.contiguous())
            rank_cumulative = torch.cumsum(gathered_sums[:, 0], dim=0).contiguous()
        else:
            weights = torch.exp(logits_float - local_max[:, None])
            rank_cumulative = None
        cumulative_local = torch.cumsum(weights[0], dim=-1).contiguous()
        return _RepeatedTemperatureSampleState(
            temperature=float(temperature),
            cumulative_local=cumulative_local,
            rank_cumulative=rank_cumulative,
            prefetch=_tp_int(
                "TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_STATE_PREFETCH",
                128,
                minimum=0,
            ),
        )

    def sample_repeated_next_token_from_state(
        self,
        state: object,
        batch_size: int,
        temperature: float,
    ) -> Tensor | None:
        if (
            not isinstance(state, _RepeatedTemperatureSampleState)
            or batch_size < 1
            or abs(float(temperature) - state.temperature) > 1e-12
        ):
            return None
        cumulative_local = state.cumulative_local
        prefetch = max(0, int(state.prefetch))
        if prefetch > 0 and state.cached_tokens is not None:
            cached_offset = max(0, int(state.cached_offset))
            cached_end = cached_offset + batch_size
            if cached_end <= int(state.cached_tokens.numel()):
                state.cached_offset = cached_end
                return state.cached_tokens[cached_offset:cached_end]
            state.cached_tokens = None
            state.cached_offset = 0
        sample_count = max(batch_size, prefetch) if prefetch > 0 else batch_size
        if not dist.is_available() or not dist.is_initialized():
            threshold = torch.rand(
                (sample_count,),
                dtype=cumulative_local.dtype,
                device=cumulative_local.device,
            ) * cumulative_local[-1]
            sampled = torch.searchsorted(cumulative_local, threshold)
            if sample_count > batch_size:
                state.cached_tokens = sampled
                state.cached_offset = batch_size
                return sampled[:batch_size]
            return sampled
        rank_cumulative = state.rank_cumulative
        if rank_cumulative is None:
            return None
        sample_payload = torch.empty((2, sample_count), dtype=torch.float32, device=cumulative_local.device)
        if self.is_primary:
            target = torch.rand(
                (sample_count,),
                dtype=rank_cumulative.dtype,
                device=rank_cumulative.device,
            ) * rank_cumulative[-1]
            selected_rank = (rank_cumulative[:, None] < target[None, :]).sum(dim=0).to(torch.long)
            previous = torch.zeros_like(target)
            has_previous = selected_rank > 0
            previous[has_previous] = rank_cumulative[selected_rank[has_previous] - 1]
            sample_payload[0].copy_(selected_rank.to(sample_payload.dtype))
            sample_payload[1].copy_(target - previous)
        dist.broadcast(sample_payload, src=0)
        selected_rank = sample_payload[0].to(torch.long)
        local_threshold = sample_payload[1].to(cumulative_local.dtype)
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
        if sample_count > batch_size:
            state.cached_tokens = local_token
            state.cached_offset = batch_size
            return local_token[:batch_size]
        return local_token

    def _sample_next_token_greedy(self, logits: Tensor) -> Tensor:
        if _tp_flag("TORCHINFERNO_GREEDY_SAMPLE_GATHER", logits.is_cuda):
            try:
                return self._sample_next_token_greedy_gather(logits)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.greedy_sample_gather", exc)
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(
            local_values == global_values, local_tokens, self.config.vocab_size
        )
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
        candidate_tokens = torch.where(
            values == global_values[None, :], tokens, self.config.vocab_size
        )
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
        profile_sample = self._temperature_sample_profile_enabled()
        count_sample = profile_sample or self._temperature_sample_counts_enabled()
        total_start_s = time.perf_counter() if profile_sample else 0.0
        phase_start_s = total_start_s
        logits_float = logits.float() / temperature
        gumbel = self._temperature_gumbel_noise(logits_float)
        noise_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
        phase_start_s = time.perf_counter() if profile_sample else 0.0
        local_values, local_indices = torch.max(logits_float + gumbel, dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        max_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(
            local_values == global_values, local_tokens, self.config.vocab_size
        )
        phase_start_s = time.perf_counter() if profile_sample else 0.0
        dist.all_reduce(next_token, op=dist.ReduceOp.MIN)
        if count_sample:
            reduce_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
            self._record_temperature_sample_gumbel_profile(
                rows=int(logits.size(0)),
                total_ms=(time.perf_counter() - total_start_s) * 1000.0 if profile_sample else 0.0,
                noise_ms=noise_ms,
                max_ms=max_ms,
                reduce_ms=reduce_ms,
                record_timings=profile_sample,
            )
        return next_token

    def _temperature_gumbel_noise(self, logits_float: Tensor) -> Tensor:
        if not _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GUMBEL_SCRATCH", True):
            return -torch.empty_like(logits_float).exponential_(
                generator=self._temperature_gumbel_generator(logits_float.device)
            ).log()
        scratch = getattr(self, "_temperature_gumbel_scratch", None)
        if (
            not isinstance(scratch, Tensor)
            or scratch.device != logits_float.device
            or scratch.dtype != logits_float.dtype
            or scratch.dim() != logits_float.dim()
            or scratch.size(-1) != logits_float.size(-1)
            or scratch.size(0) < logits_float.size(0)
        ):
            scratch = torch.empty_like(logits_float)
            self._temperature_gumbel_scratch = scratch
        noise = scratch[: logits_float.size(0), : logits_float.size(1)]
        noise.exponential_(generator=self._temperature_gumbel_generator(logits_float.device))
        noise.log_().neg_()
        return noise

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
        profile_sample = self._temperature_sample_profile_enabled()
        count_sample = profile_sample or self._temperature_sample_counts_enabled()
        total_start_s = time.perf_counter() if profile_sample else 0.0
        phase_start_s = total_start_s
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        max_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
        phase_start_s = time.perf_counter() if profile_sample else 0.0
        weights = torch.exp(logits_float - global_max[:, None])
        local_sum = weights.sum(dim=-1)
        weights_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
        phase_start_s = time.perf_counter() if profile_sample else 0.0
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
        rank_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0

        phase_start_s = time.perf_counter() if profile_sample else 0.0
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
        cdf_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
        phase_start_s = time.perf_counter() if profile_sample else 0.0
        dist.all_reduce(local_token, op=dist.ReduceOp.SUM)
        if count_sample:
            reduce_ms = (time.perf_counter() - phase_start_s) * 1000.0 if profile_sample else 0.0
            self._record_temperature_sample_profile(
                rows=int(logits.size(0)),
                total_ms=(time.perf_counter() - total_start_s) * 1000.0 if profile_sample else 0.0,
                max_ms=max_ms,
                weights_ms=weights_ms,
                rank_ms=rank_ms,
                cdf_ms=cdf_ms,
                reduce_ms=reduce_ms,
                record_timings=profile_sample,
            )
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
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(
            local_values == global_values, local_tokens, self.config.vocab_size
        )
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
    dist.init_process_group(backend=backend, timeout=_tensor_parallel_process_group_timeout())


def _tensor_parallel_process_group_timeout() -> timedelta:
    return timedelta(seconds=_tp_int("TORCHINFERNO_TP_PROCESS_GROUP_TIMEOUT_S", 1800, minimum=1))


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


def _rank0_checkpoint_broadcast_enabled(
    *,
    device: torch.device,
    world_size: int,
    dtype: torch.dtype | None,
) -> bool:
    if world_size <= 1 or dtype is None:
        return False
    if device.type != "cuda":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    # Rank-0 tensor broadcast can reduce shared-storage pressure, but very
    # large startup broadcasts are brittle across NCCL/CUDA/host environments.
    # Keep the portable per-rank checkpoint reader as the default and let
    # operators opt into broadcast where it is known to be healthy.
    return _tp_flag("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", False)


def _rank0_replicated_checkpoint_broadcast_enabled(
    *,
    device: torch.device,
    world_size: int,
    dtype: torch.dtype | None,
) -> bool:
    if world_size <= 1 or dtype is None:
        return False
    if device.type != "cuda":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    if "TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST" in os.environ:
        checkpoint_broadcast = _rank0_checkpoint_broadcast_enabled(
            device=device,
            world_size=world_size,
            dtype=dtype,
        )
        if not checkpoint_broadcast:
            return False
        if "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST" not in os.environ:
            return True
    if "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST" in os.environ:
        return _tp_flag("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST", True)
    # Public CUDA 13 benchmark hosts have repeatedly made large startup NCCL
    # checkpoint collectives much slower than direct per-rank safetensor reads.
    # Keep the portable path as the default; operators can still opt in where
    # rank-0 loading is known to be healthy.
    return False


def _rank0_replicated_checkpoint_page_cache_warm_enabled(
    *,
    device: torch.device,
    world_size: int,
    dtype: torch.dtype | None,
) -> bool:
    del dtype
    env_name = "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM"
    if world_size <= 1 or device.type != "cuda":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    if env_name in os.environ:
        return _tp_flag(env_name, True)
    if (
        "TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST" in os.environ
        or "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST" in os.environ
    ):
        return False
    # This only orders per-rank reads through rank 0; it is not a collective
    # broadcast. It helps on local cached filesystems, but public submit hosts
    # have shown 500s+ initial tensor loads with this path. Keep concurrent
    # per-rank reads as the portable default.
    return False


def _rank0_checkpoint_shard_scatter_enabled(
    *,
    device: torch.device,
    world_size: int,
    dtype: torch.dtype | None,
) -> bool:
    if world_size <= 1 or dtype is None:
        return False
    if device.type != "cuda":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    if "TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST" in os.environ:
        return _rank0_checkpoint_broadcast_enabled(
            device=device,
            world_size=world_size,
            dtype=dtype,
        )
    # Large rank-0 checkpoint scatters are fast on healthy local hosts but have
    # timed out repeatedly on public submit hosts when rank 0 stalls reading or
    # packing a tensor while other ranks enqueue SCATTER work. Keep the
    # per-rank safetensor shard reader as the default and require an explicit
    # opt-in for collective shard loading.
    return _tp_flag("TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER", False)


def _rank0_checkpoint_scatter_enabled() -> bool:
    return _tp_flag("TORCHINFERNO_TP_RANK0_CHECKPOINT_SCATTER", True)


def _rank0_checkpoint_direct_scatter_enabled() -> bool:
    if not hasattr(dist, "scatter"):
        return False
    return _tp_flag("TORCHINFERNO_TP_RANK0_CHECKPOINT_DIRECT_SCATTER", True)


def _checkpoint_broadcast_chunk_bytes() -> int:
    return _tp_int(
        "TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST_CHUNK_BYTES",
        256 * 1024 * 1024,
        minimum=1,
    )


def _replicated_checkpoint_page_cache_warm_min_bytes() -> int:
    return _tp_int(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM_MIN_BYTES",
        64 * 1024 * 1024,
        minimum=0,
    )


def _checkpoint_tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype | None) -> int:
    elements = 1
    for size in shape:
        elements *= int(size)
    element_dtype = dtype if dtype is not None else torch.float32
    return elements * torch.empty((), dtype=element_dtype).element_size()


def _broadcast_tensor_in_chunks(tensor: Tensor, *, src: int) -> Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    flat = tensor.view(-1)
    if flat.numel() == 0:
        return tensor
    max_chunk_bytes = _checkpoint_broadcast_chunk_bytes()
    elems_per_chunk = max(1, max_chunk_bytes // tensor.element_size())
    if flat.numel() <= elems_per_chunk:
        dist.broadcast(flat, src=src)
        return tensor
    for start in range(0, flat.numel(), elems_per_chunk):
        length = min(elems_per_chunk, flat.numel() - start)
        dist.broadcast(flat.narrow(0, start, length), src=src)
    return tensor


def _load_checkpoint_tensor(
    loader: _CheckpointTensorLoader,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    rank: int,
    world_size: int,
) -> Tensor:
    if not _rank0_replicated_checkpoint_broadcast_enabled(
        device=device,
        world_size=world_size,
        dtype=dtype,
    ):
        if _rank0_replicated_checkpoint_page_cache_warm_enabled(
            device=device,
            world_size=world_size,
            dtype=dtype,
        ):
            shape = loader.get_tensor_shape(name)
            if _checkpoint_tensor_nbytes(shape, dtype) >= _replicated_checkpoint_page_cache_warm_min_bytes():
                if rank == 0:
                    tensor = loader.get_tensor(name, device=device, dtype=dtype)
                    _barrier()
                else:
                    _barrier()
                    tensor = loader.get_tensor(name, device=device, dtype=dtype)
                _barrier()
                return tensor
        return loader.get_tensor(name, device=device, dtype=dtype)
    shape = loader.get_tensor_shape(name)
    if rank == 0:
        tensor = loader.get_tensor(name, device=device, dtype=dtype)
    else:
        tensor = torch.empty(shape, device=device, dtype=dtype)
    return _broadcast_tensor_in_chunks(tensor, src=0)


def _load_checkpoint_tensor_shard(
    loader: _CheckpointTensorLoader,
    name: str,
    *,
    dim: int,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> Tensor:
    rank0_broadcast = _rank0_checkpoint_broadcast_enabled(
        device=device,
        world_size=world_size,
        dtype=dtype,
    )
    rank0_shard_scatter = _rank0_checkpoint_shard_scatter_enabled(
        device=device,
        world_size=world_size,
        dtype=dtype,
    )
    if not rank0_broadcast and not rank0_shard_scatter:
        return loader.get_tensor_shard(
            name,
            dim=dim,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=dtype,
        )
    shape = loader.get_tensor_shape(name)
    if shape[dim] % world_size != 0:
        raise ValueError(f"cannot shard {name} shape={shape} dim={dim} across {world_size} ranks")
    if (
        (rank0_broadcast or rank0_shard_scatter)
        and _rank0_checkpoint_scatter_enabled()
        and hasattr(dist, "reduce_scatter_tensor")
    ):
        return _load_checkpoint_tensor_shard_scatter(
            loader,
            name,
            shape=shape,
            dim=dim,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=dtype,
        )
    if not rank0_broadcast:
        return loader.get_tensor_shard(
            name,
            dim=dim,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=dtype,
        )
    if rank == 0:
        tensor = loader.get_tensor(name, device=device, dtype=dtype)
    else:
        tensor = torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(tensor, src=0)
    shard = shape[dim] // world_size
    start = rank * shard
    return tensor.narrow(dim, start, shard).clone(memory_format=torch.contiguous_format)


def _load_checkpoint_tensor_shard_scatter(
    loader: _CheckpointTensorLoader,
    name: str,
    *,
    shape: tuple[int, ...],
    dim: int,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shard = shape[dim] // world_size
    output_shape = list(shape)
    output_shape[dim] = shard
    output = torch.empty(tuple(output_shape), device=device, dtype=dtype)

    if _rank0_checkpoint_direct_scatter_enabled():
        scatter_list = None
        if rank == 0:
            tensor = loader.get_tensor(name, device=device, dtype=dtype)
            scatter_list = [
                chunk if chunk.is_contiguous() else chunk.contiguous()
                for chunk in tensor.split(shard, dim=dim)
            ]
            del tensor
        dist.scatter(output, scatter_list=scatter_list, src=0)
        return output if output.is_contiguous() else output.contiguous()

    if rank == 0:
        tensor = loader.get_tensor(name, device=device, dtype=dtype)
        if dim == 0:
            scatter_input = tensor.contiguous()
        else:
            chunks = tensor.split(shard, dim=dim)
            scatter_input = torch.cat([chunk.contiguous() for chunk in chunks], dim=0).contiguous()
    else:
        input_shape = list(shape)
        if dim != 0:
            input_shape[0] *= world_size
            input_shape[dim] = shard
        scatter_input = torch.zeros(tuple(input_shape), device=device, dtype=dtype)
    dist.reduce_scatter_tensor(output, scatter_input, op=dist.ReduceOp.SUM)
    return output if output.is_contiguous() else output.contiguous()


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
    # BUCKET PATH (context_len < 0): keys [0:-context_len] hold a static context
    # bucket while start_positions stays dynamic. This sacrifices the lower-right
    # flash mask for a captured boolean mask, but it lets one graph replay across
    # different prefix lengths inside the same context bucket.
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
    if context_len is not None and context_len > 0:
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
    if context_len is not None:
        context_bucket = -context_len
        if context_bucket < suffix_tokens:
            raise ValueError("ragged prefill context bucket must cover the suffix")
        context_bucket = min(context_bucket, k.size(2))
        k = k[:, :, :context_bucket, :]
        v = v[:, :, :context_bucket, :]
        start_positions = start_positions.to(device=q.device)
        key_positions = torch.arange(context_bucket, device=q.device)
        query_offsets = torch.arange(suffix_tokens, device=q.device)
        written = key_positions[None, :] < (start_positions[:, None] + suffix_tokens)
        zero = torch.zeros((), dtype=k.dtype, device=k.device)
        k = torch.where(written[:, None, :, None], k, zero)
        v = torch.where(written[:, None, :, None], v, zero)
        q_abs = start_positions[:, None] + query_offsets[None, :]
        mask = key_positions[None, None, :] <= q_abs[:, :, None]
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, :, :],
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


def _packed_prefill_offsets_from_q_lens(q_lens: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for q_len in q_lens:
        if q_len <= 0:
            raise ValueError("packed prefill q_lens must be positive")
        offsets.append(cursor)
        cursor += int(q_len)
    return tuple(offsets)


def _packed_prefill_flat_indices(
    q_lens: tuple[int, ...],
    suffix_bucket: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    token_indices: list[int] = []
    request_indices: list[int] = []
    for request_idx, q_len in enumerate(q_lens):
        if q_len <= 0:
            raise ValueError("packed prefill q_lens must be positive")
        if q_len > suffix_bucket:
            raise ValueError("packed prefill q_lens cannot exceed suffix bucket")
        base = request_idx * suffix_bucket
        token_indices.extend(base + offset for offset in range(q_len))
        request_indices.extend([request_idx] * q_len)
    return (
        torch.tensor(token_indices, dtype=torch.long, device=device),
        torch.tensor(request_indices, dtype=torch.long, device=device),
    )


def _build_packed_prefill_attention_groups(
    q_lens: tuple[int, ...],
    start_positions: tuple[int, ...],
    request_offsets: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[_PackedPrefillAttentionGroup, ...]:
    if not (len(q_lens) == len(start_positions) == len(request_offsets)):
        raise ValueError("packed prefill metadata must have one entry per request")
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for request_idx, (q_len, start, offset) in enumerate(zip(q_lens, start_positions, request_offsets)):
        if q_len <= 0:
            raise ValueError("packed prefill q_lens must be positive")
        grouped.setdefault((int(q_len), int(start)), []).append((request_idx, int(offset)))
    return tuple(
        (
            q_len,
            start,
            torch.tensor([request_idx for request_idx, _offset in entries], dtype=torch.long, device=device),
            tuple(request_idx for request_idx, _offset in entries),
            tuple(offset for _request_idx, offset in entries),
        )
        for (q_len, start), entries in grouped.items()
    )


def _packed_prefill_scaled_dot_product_attention(
    q: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    start_positions: Tensor,
    *,
    q_lens: Tensor,
    row_indices: Tensor,
    request_offsets: Tensor,
    packed_attention_groups: tuple[_PackedPrefillAttentionGroup, ...] | None = None,
    enable_gqa: bool,
) -> Tensor:
    if q.size(0) != 1:
        raise ValueError("packed prefill attention expects q batch size 1")
    if packed_attention_groups is None:
        q_lens_cpu = tuple(int(v) for v in q_lens.detach().cpu().tolist())
        starts_cpu = tuple(int(v) for v in start_positions.detach().cpu().tolist())
        offsets_cpu = tuple(int(v) for v in request_offsets.detach().cpu().tolist())
        packed_attention_groups = _build_packed_prefill_attention_groups(
            q_lens_cpu,
            starts_cpu,
            offsets_cpu,
            device=q.device,
        )
    if (
        sum(
            int(group_indices.numel())
            for _q_len, _start, group_indices, _request_indices, _offsets in packed_attention_groups
        )
        != row_indices.numel()
    ):
        raise ValueError("packed prefill metadata must have one entry per request")
    outs: list[Tensor | None] = [None] * int(row_indices.numel())
    causal_lower_right = None
    try:
        from torch.nn.attention.bias import causal_lower_right as _causal_lower_right

        causal_lower_right = _causal_lower_right
    except Exception as exc:
        warn_optional_failure("llama3_tensor_parallel.packed_causal_lower_right", exc)
    for q_len, start, group_indices, request_indices, offsets in packed_attention_groups:
        context = start + q_len
        if context > cache_keys.size(2):
            raise ValueError("KV cache capacity exceeded")
        rows = row_indices.index_select(0, group_indices)
        qi = torch.cat([q[:, :, offset : offset + q_len, :] for offset in offsets], dim=0)
        ki = cache_keys.index_select(0, rows)[:, :, :context, :]
        vi = cache_values.index_select(0, rows)[:, :, :context, :]
        if causal_lower_right is not None:
            out_group = F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=causal_lower_right(q_len, context),
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        else:
            key_positions = torch.arange(context, device=q.device)
            query_positions = start + torch.arange(q_len, device=q.device)
            mask = key_positions[None, :] <= query_positions[:, None]
            out_group = F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=mask[None, None, :, :],
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        for group_pos, request_idx in enumerate(request_indices):
            outs[int(request_idx)] = out_group[group_pos : group_pos + 1]
    if any(out is None for out in outs):
        raise RuntimeError("packed prefill attention did not produce every request output")
    return torch.cat([out for out in outs if out is not None], dim=2)


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


def _tp_swiglu(gate: Tensor, up: Tensor, *, out: Tensor | None = None) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_SWIGLU", False):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up, out=out)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.swiglu", exc)
    activated = F.silu(gate) * up
    if out is not None:
        out.copy_(activated)
        return out
    return activated


def _tp_decode_swiglu(gate: Tensor, up: Tensor, *, out: Tensor | None = None) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_DECODE_SWIGLU"):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up, out=out)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_swiglu", exc)
    activated = F.silu(gate) * up
    if out is not None:
        out.copy_(activated)
        return out
    return activated


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
    return _should_use_ragged_prefill_graph(
        input_ids,
        cache,
        seq_lens,
        row_indices,
        logit_positions,
    )


def _should_use_ragged_prefill_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    logit_positions: Tensor | None,
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
        and (
            logit_positions is None
            or (logit_positions.is_cuda and logit_positions.shape == (input_ids.size(0),))
        )
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
    # is ~31% of prefill (profiled), so this is the top prefill lever. Callers must
    # avoid allocating/probing the per-shape symm-mem buffer during CUDA graph
    # capture; captured prefill may reuse only buffers prepared by eager warmup.
    max_tokens = _tp_int("TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE_MAX_TOKENS", 40960, minimum=1)
    override = _SYMM_MEM_PREFILL_ALLREDUCE_ENABLED_OVERRIDE[0]
    prefill_enabled = (
        override
        if override is not None
        else _tp_flag("TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE", False)
    )
    return (
        world_size > 1
        and not _SYMM_REDUCE_DISABLED
        and prefill_enabled
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.ndim == 3
        and hidden.size(1) > 1
        and hidden.size(0) * hidden.size(1) <= max_tokens
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
