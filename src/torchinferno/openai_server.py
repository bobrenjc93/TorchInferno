from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import ContextManager, Iterable, Iterator, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelForCausalLM,
    symm_mem_allreduce_max_batch,
)
from torchinferno.engine.loader import (
    distributed_env_requested as _engine_distributed_env_requested,
    distributed_server_command as _engine_distributed_server_command,
    infer_model_kind as _engine_infer_model_kind,
    llama_parallelism as _engine_llama_parallelism,
    load_model_for_engine,
    primary_device as _engine_primary_device,
    resolve_dtype as _engine_resolve_dtype,
    server_devices as _engine_server_devices,
    should_reexec_distributed_server as _engine_should_reexec_distributed_server,
)
from torchinferno.openai_http import (
    FastOpenAIHTTPServer as _FastOpenAIServer,
    OpenAIHTTPServer as _OpenAIServer,
)
from torchinferno.openai_warmup import (
    parse_positive_int_csv as _parse_positive_int_csv,
    warmup_prefill_cache_token_counts as _warmup_prefill_cache_token_counts,
    warmup_prefix_suffix_cache_token_counts as _warmup_prefix_suffix_cache_token_counts,
    warmup_prefix_suffix_token_counts as _warmup_prefix_suffix_token_counts,
    warmup_prompt_token_counts as _warmup_prompt_token_counts,
    warmup_ragged_decode_batch_sizes as _warmup_ragged_decode_batch_sizes,
    warmup_ragged_decode_cache_token_counts as _warmup_ragged_decode_cache_token_counts,
    warmup_ragged_decode_extra_cache_specs as _warmup_ragged_decode_extra_cache_specs,
    warmup_ragged_decode_prompt_tokens as _warmup_ragged_decode_prompt_tokens,
    warmup_ragged_decode_row_counts as _warmup_ragged_decode_row_counts,
    warmup_temperature_batch_sizes as _warmup_temperature_batch_sizes,
    warmup_temperature_prompt_token_counts as _warmup_temperature_prompt_token_counts,
)
from torchinferno.runtime.options import env_flag, env_float, env_int, warn_optional_failure
from torchinferno.runtime.prefix_cache import (
    TensorPrefixCacheEntry,
    cache_sequence_length,
    reset_cache_sequence,
    restore_tensor_prefix_cache,
    set_cache_sequence_length,
    snapshot_tensor_prefix_cache,
)
from torchinferno.runtime.sampling import sample_next_token
from torchinferno.runtime.scheduler import (
    PersistentBatchPlan as _PersistentBatchPlan,
    PersistentBatchRequest as _PersistentBatchRequest,
    PersistentBatchScheduler as _PersistentBatchScheduler,
    TokenBudgetPlan as _TokenBudgetPlan,
    TokenBudgetRequest as _TokenBudgetRequest,
    TokenBudgetScheduler as _TokenBudgetScheduler,
    token_budget_model_step_command as _token_budget_model_step_command,
)
from torchinferno.runtime.serving import (
    ContinuousBatchEngine as _RuntimeContinuousBatchEngine,
    ServingRequest as _RuntimeServingRequest,
)


@dataclass(frozen=True)
class OpenAIServerConfig:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    model_kind: str = "auto"
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    devices: tuple[str, ...] = ()
    device: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    token: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    cache_backend: str = "dense"
    page_size: int = 16
    max_batch_size: int = 128
    batch_wait_ms: float = 2.0
    single_request_admission_wait_ms: float | None = None
    llama_parallelism: str = "auto"


@dataclass(frozen=True)
class CompletionResult:
    tokens: list[int]
    prompt_tokens: int


def _startup_warmup_enabled_for_cache_backend(cache_backend: str) -> bool:
    if cache_backend.lower() == "dense":
        return True
    return env_flag("TORCHINFERNO_OPENAI_STARTUP_WARMUP_NON_DENSE_CACHE", True)


@dataclass
class _QueuedGeneration:
    prompt: list[int]
    max_tokens: int
    temperature: float
    stream: bool
    responses: "queue.Queue[object] | queue.SimpleQueue[object]"
    queued_at_s: float = 0.0
    queue_sequence: int = -1
    done: bool = False


@dataclass
class _StreamRow:
    request_id: str
    request: _QueuedGeneration
    row: int
    generated_tokens: int = 0


class _StreamRowState:
    def __init__(self) -> None:
        self._rows: dict[int, _StreamRow] = {}
        self._request_rows: dict[str, int] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(row.request_id for _row_index, row in sorted(self._rows.items()))

    def is_admitted(self, request_id: str) -> bool:
        return str(request_id) in self._request_rows

    def generated_tokens(self, request_id: str) -> int:
        row = self._row_for_request(request_id)
        return row.generated_tokens

    def admit(
        self,
        request_id: str,
        row: int,
        request: _QueuedGeneration,
        *,
        generated_tokens: int = 0,
    ) -> None:
        request_id = str(request_id)
        row = int(row)
        if generated_tokens < 0:
            raise ValueError("stream row generated_tokens must be non-negative")
        existing_row = self._request_rows.get(request_id)
        if existing_row is not None:
            if existing_row != row:
                raise ValueError("stream request is already assigned to a different row")
            current = self._rows[existing_row]
            if current.request is not request:
                raise ValueError("stream request_id is already assigned to another request")
            return
        current = self._rows.get(row)
        if current is not None:
            raise ValueError("stream row is already occupied")
        self._rows[row] = _StreamRow(
            request_id=request_id,
            request=request,
            row=row,
            generated_tokens=int(generated_tokens),
        )
        self._request_rows[request_id] = row

    def emit(
        self,
        request_id: str,
        token_id: int | None,
        *,
        stop_token_ids: frozenset[int] = frozenset(),
    ) -> bool:
        row = self._row_for_request(request_id)
        if token_id is not None:
            row.generated_tokens += 1
        _emit_stream_token(
            row.request,
            token_id,
            generated_tokens=row.generated_tokens,
            stop_token_ids=stop_token_ids,
        )
        if row.request.done:
            self.release(request_id)
            return True
        return False

    def finish(self, request_id: str) -> bool:
        row_index = self._request_rows.get(str(request_id))
        if row_index is None:
            return False
        row = self._rows[row_index]
        _finish_stream_request(row.request)
        self.release(request_id)
        return True

    def release(self, request_id: str) -> bool:
        request_id = str(request_id)
        row_index = self._request_rows.pop(request_id, None)
        if row_index is None:
            return False
        self._rows.pop(row_index, None)
        return True

    def _row_for_request(self, request_id: str) -> _StreamRow:
        row_index = self._request_rows.get(str(request_id))
        if row_index is None:
            raise ValueError("stream request is not admitted")
        return self._rows[row_index]


@dataclass
class _PersistentPromptListStepState:
    cache: object
    prefix_caches: Mapping[tuple[int, ...], object]
    active: list[bool]
    per_row_limits: list[int]
    generated_tokens: list[int]
    seq_lens: Tensor
    next_token_tensor: Tensor
    row_request_ids: list[str | None]
    cache_batch_size: int
    logical_cache_batch_size: int = 0
    ephemeral_graph_allowed: bool = False
    ephemeral_graph_scope: bool = False


@dataclass(frozen=True)
class _PersistentPromptListStepResult:
    decode_tokens: dict[str, int | None]
    prefill_tokens: dict[str, int | None]
    finished_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PersistentPromptListDecodeRunResult:
    step_results: tuple[_PersistentPromptListStepResult, ...]

    @property
    def finished_request_ids(self) -> tuple[str, ...]:
        finished: list[str] = []
        for result in self.step_results:
            finished.extend(result.finished_request_ids)
        return tuple(finished)


@dataclass(frozen=True)
class _PersistentPromptListLocalRunStats:
    scheduler_steps: int
    step_commands: int
    decode_run_commands: int
    empty_plans: int
    decode_steps: int
    max_decode_run_steps: int
    prefill_admissions: int
    emitted_tokens: int
    finished_events: int
    first_emit_s: float | None
    closed: bool


@dataclass
class _TokenBudgetStepState:
    cache: object
    prefix_caches: Mapping[tuple[int, ...], object]
    active: list[bool]
    row_request_ids: list[str | None]
    generated_tokens: list[int]
    seq_lens: Tensor
    next_token_tensor: Tensor
    cache_batch_size: int


@dataclass(frozen=True)
class _TokenBudgetStepResult:
    decode_tokens: dict[str, int | None]
    prefill_tokens: dict[str, int | None]
    finished_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TokenBudgetDecodeRunResult:
    step_results: tuple[_TokenBudgetStepResult, ...]

    @property
    def finished_request_ids(self) -> tuple[str, ...]:
        finished: list[str] = []
        for result in self.step_results:
            finished.extend(result.finished_request_ids)
        return tuple(finished)


@dataclass(frozen=True)
class _TokenBudgetLocalRunStats:
    scheduler_steps: int
    step_commands: int
    decode_run_commands: int
    empty_plans: int
    emitted_tokens: int
    finished_events: int
    max_decode_run_steps: int
    closed: bool


def _merge_token_budget_finished_ids(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for request_id in group:
            request_id_text = str(request_id)
            if request_id_text in seen:
                continue
            seen.add(request_id_text)
            merged.append(request_id_text)
    return tuple(merged)


def _persistent_prompt_list_scheduler_for_group(
    group: Sequence[_QueuedGeneration],
    *,
    max_active_rows: int,
    prefill_token_budget: int | None = None,
    prefix_tokens: int = 0,
) -> _PersistentBatchScheduler:
    scheduler = _PersistentBatchScheduler(
        max_rows=max_active_rows,
        prefill_token_budget=prefill_token_budget,
    )
    prompts = [request.prompt for request in group]
    prefix_key: tuple[int, ...] | None = None
    if prefix_tokens > 0 and prompts:
        candidate = tuple(int(token_id) for token_id in prompts[0][:prefix_tokens])
        if len(candidate) == prefix_tokens and all(tuple(prompt[:prefix_tokens]) == candidate for prompt in prompts):
            prefix_key = candidate
    for index, request in enumerate(group):
        if request.max_tokens <= 0:
            continue
        prefix_hit_tokens = min(prefix_tokens, len(request.prompt)) if prefix_key is not None else 0
        scheduler.submit(
            _PersistentBatchRequest(
                str(index),
                prompt_tokens=len(request.prompt),
                max_new_tokens=request.max_tokens,
                prefix_hit_tokens=prefix_hit_tokens,
                prefix_key=prefix_key,
            )
        )
    return scheduler


def _persistent_prompt_list_step_payload(
    plan: _PersistentBatchPlan,
    group: Sequence[_QueuedGeneration],
) -> dict[str, object]:
    admitted: list[dict[str, object]] = []
    for admission in plan.prefill_admissions:
        request_index = int(admission.request_id)
        request = group[request_index]
        prefix_tokens = int(admission.prefix_hit_tokens)
        admitted.append(
            {
                "request_id": admission.request_id,
                "row": int(admission.row),
                "prompt": list(request.prompt),
                "max_tokens": int(request.max_tokens),
                "prefix_hit_tokens": prefix_tokens,
                "prefix": list(request.prompt[:prefix_tokens]),
                "suffix": list(request.prompt[prefix_tokens:]),
                "prefill_tokens": int(admission.prefill_tokens),
            }
        )
    request_by_id = {str(index): request for index, request in enumerate(group)}
    prefill_groups: list[dict[str, object]] = []
    for prefill_group in plan.prefill_groups:
        request_ids = list(prefill_group.request_ids)
        first_request = request_by_id.get(request_ids[0]) if request_ids else None
        prefix_tokens = int(prefill_group.prefix_hit_tokens)
        prefill_groups.append(
            {
                "request_ids": request_ids,
                "rows": [int(row) for row in prefill_group.rows],
                "prefix_hit_tokens": prefix_tokens,
                "prefix": [] if first_request is None else list(first_request.prompt[:prefix_tokens]),
                "suffix_tokens": [int(tokens) for tokens in prefill_group.suffix_tokens],
            }
        )
    return {
        "op": "persistent_prompt_list_step",
        "step": int(plan.step),
        "decode_request_ids": list(plan.decode_request_ids),
        "decode_rows": list(plan.decode_rows),
        "prefill": admitted,
        "prefill_groups": prefill_groups,
        "finished_after_prefill": list(plan.finished_after_prefill),
    }


def _persistent_prompt_list_decode_run_payload(
    *,
    start_step: int,
    step_count: int,
    temperature: float,
    static_graph_buckets: bool = False,
) -> dict[str, object]:
    if step_count < 1:
        raise ValueError("persistent prompt-list decode-run requires positive step_count")
    payload: dict[str, object] = {
        "op": "persistent_prompt_list_decode_run",
        "start_step": int(start_step),
        "step_count": int(step_count),
        "temperature": float(temperature),
    }
    if static_graph_buckets:
        payload["static_graph_buckets"] = True
    return payload


def _persistent_prompt_list_decode_run_tensor_payload(
    payload: Mapping[str, object],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    step_count = int(payload.get("step_count", 0))
    if step_count < 1:
        raise ValueError("persistent prompt-list decode-run tensor payload requires positive step_count")
    meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
    meta[0] = _TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN
    meta[1] = int(payload.get("start_step", 0))
    meta[2] = step_count
    meta[8] = int(bool(payload.get("static_graph_buckets", False)))
    temperature = torch.tensor([float(payload.get("temperature", 0.0))], dtype=torch.float64, device=device)
    return meta, temperature


def _persistent_prompt_list_decode_run_payload_from_tensor_payload(
    meta: Tensor,
    temperature: Tensor,
) -> dict[str, object]:
    meta_cpu = meta.detach().cpu().to(torch.long)
    if int(meta_cpu[0].item()) != _TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN:
        raise ValueError("persistent prompt-list decode-run tensor payload has wrong command kind")
    payload = _persistent_prompt_list_decode_run_payload(
        start_step=int(meta_cpu[1].item()),
        step_count=int(meta_cpu[2].item()),
        temperature=float(temperature.detach().cpu().to(torch.float64).item()),
        static_graph_buckets=bool(int(meta_cpu[8].item())),
    )
    return payload


def _persistent_prompt_list_step_tensor_payload(
    payload: Mapping[str, object],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    decode_ids_obj = payload.get("decode_request_ids", [])
    decode_rows_obj = payload.get("decode_rows", [])
    if not isinstance(decode_ids_obj, list) or not isinstance(decode_rows_obj, list):
        raise ValueError("persistent prompt-list tensor payload requires decode lists")
    if len(decode_ids_obj) != len(decode_rows_obj):
        raise ValueError("persistent prompt-list tensor payload decode lists differ")
    decode_rows = [
        [_token_budget_request_ordinal(request_id), int(row)]
        for request_id, row in zip(decode_ids_obj, decode_rows_obj)
    ]
    decode_tensor = torch.tensor(decode_rows, dtype=torch.long, device=device)
    if not decode_rows:
        decode_tensor = torch.empty(
            (0, _TP_PROMPT_LIST_PERSISTENT_DECODE_FIELDS),
            dtype=torch.long,
            device=device,
        )

    prefill_obj = payload.get("prefill", [])
    if not isinstance(prefill_obj, list):
        raise ValueError("persistent prompt-list tensor payload requires prefill list")
    prefill_rows: list[list[int]] = []
    prompt_rows: list[list[int]] = []
    for item in prefill_obj:
        if not isinstance(item, Mapping):
            raise ValueError("persistent prompt-list tensor prefill entries must be mappings")
        prompt_obj = item.get("prompt", [])
        if not isinstance(prompt_obj, list):
            raise ValueError("persistent prompt-list tensor prefill prompt must be a list")
        prompt = [int(token_id) for token_id in prompt_obj]
        prefill_rows.append(
            [
                _token_budget_request_ordinal(item.get("request_id")),
                int(item["row"]),
                int(item["max_tokens"]),
                int(item.get("prefix_hit_tokens", 0)),
                int(item.get("prefill_tokens", max(1, len(prompt) - int(item.get("prefix_hit_tokens", 0))))),
                int(item.get("start_token", item.get("prefix_hit_tokens", 0))),
            ]
        )
        prompt_rows.append(prompt)
    prefill_tensor = torch.tensor(prefill_rows, dtype=torch.long, device=device)
    if not prefill_rows:
        prefill_tensor = torch.empty(
            (0, _TP_PROMPT_LIST_PERSISTENT_PREFILL_FIELDS),
            dtype=torch.long,
            device=device,
        )
    prompt_token_rows, prompt_lengths = _prompt_list_tensor_payload(prompt_rows, device)

    finished_obj = payload.get("finished_after_prefill", [])
    if not isinstance(finished_obj, list):
        raise ValueError("persistent prompt-list tensor payload requires finished_after_prefill list")
    finished_ids = torch.tensor(
        [_token_budget_request_ordinal(request_id) for request_id in finished_obj],
        dtype=torch.long,
        device=device,
    )
    meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
    meta[0] = _TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP
    meta[1] = int(payload.get("step", 0))
    meta[2] = int(decode_tensor.size(0))
    meta[3] = int(prefill_tensor.size(0))
    meta[4] = int(prompt_token_rows.size(1)) if prompt_token_rows.ndim == 2 else 0
    meta[5] = int(finished_ids.numel())
    meta[6] = _TP_PROMPT_LIST_PERSISTENT_PREFILL_FIELDS
    meta[7] = _TP_PROMPT_LIST_PERSISTENT_DECODE_FIELDS
    meta[8] = int(bool(payload.get("static_graph_buckets", False)))
    temperature = torch.tensor([float(payload.get("temperature", 0.0))], dtype=torch.float64, device=device)
    return meta, temperature, decode_tensor, prefill_tensor, prompt_lengths, prompt_token_rows, finished_ids


def _persistent_prompt_list_step_payload_from_tensor_payload(
    meta: Tensor,
    temperature: Tensor,
    decode_tensor: Tensor,
    prefill_tensor: Tensor,
    prompt_lengths: Tensor,
    prompt_token_rows: Tensor,
    finished_ids: Tensor,
) -> dict[str, object]:
    meta_cpu = meta.detach().cpu().to(torch.long)
    if int(meta_cpu[0].item()) != _TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP:
        raise ValueError("persistent prompt-list tensor payload has wrong command kind")
    if int(meta_cpu[6].item()) != _TP_PROMPT_LIST_PERSISTENT_PREFILL_FIELDS:
        raise ValueError("persistent prompt-list tensor payload has wrong prefill width")
    if int(meta_cpu[7].item()) != _TP_PROMPT_LIST_PERSISTENT_DECODE_FIELDS:
        raise ValueError("persistent prompt-list tensor payload has wrong decode width")

    decode_cpu = decode_tensor.detach().cpu().to(torch.long)
    if decode_cpu.ndim != 2 or decode_cpu.size(1) != _TP_PROMPT_LIST_PERSISTENT_DECODE_FIELDS:
        raise ValueError("persistent prompt-list tensor decode tensor has invalid shape")
    if int(meta_cpu[2].item()) != int(decode_cpu.size(0)):
        raise ValueError("persistent prompt-list tensor decode count mismatch")
    prefill_cpu = prefill_tensor.detach().cpu().to(torch.long)
    if prefill_cpu.ndim != 2 or prefill_cpu.size(1) != _TP_PROMPT_LIST_PERSISTENT_PREFILL_FIELDS:
        raise ValueError("persistent prompt-list tensor prefill tensor has invalid shape")
    if int(meta_cpu[3].item()) != int(prefill_cpu.size(0)):
        raise ValueError("persistent prompt-list tensor prefill count mismatch")
    prompt_lengths_list = [int(value) for value in prompt_lengths.detach().cpu().to(torch.long).tolist()]
    if len(prompt_lengths_list) != int(prefill_cpu.size(0)):
        raise ValueError("persistent prompt-list tensor prompt length count mismatch")
    prompt_rows_cpu = prompt_token_rows.detach().cpu().to(torch.long)
    if prompt_rows_cpu.ndim != 2:
        raise ValueError("persistent prompt-list tensor prompt rows have invalid shape")
    if int(meta_cpu[4].item()) != int(prompt_rows_cpu.size(1)):
        raise ValueError("persistent prompt-list tensor prompt width mismatch")
    finished_cpu = finished_ids.detach().cpu().to(torch.long)
    if int(meta_cpu[5].item()) != int(finished_cpu.numel()):
        raise ValueError("persistent prompt-list tensor finished count mismatch")

    decode_request_ids = [str(int(values[0])) for values in decode_cpu.tolist()]
    decode_rows = [int(values[1]) for values in decode_cpu.tolist()]
    prefill_items: list[dict[str, object]] = []
    for index, values in enumerate(prefill_cpu.tolist()):
        prompt_len = prompt_lengths_list[index]
        prompt = [int(token_id) for token_id in prompt_rows_cpu[index, :prompt_len].tolist()]
        prefix_hit_tokens = int(values[3])
        prefill_tokens = int(values[4])
        start_token = int(values[5])
        prompt_complete = start_token + prefill_tokens >= len(prompt)
        prompt_chunk = prompt[start_token : start_token + prefill_tokens]
        item = {
            "request_id": str(int(values[0])),
            "row": int(values[1]),
            "prompt": prompt,
            "max_tokens": int(values[2]),
            "prefix_hit_tokens": prefix_hit_tokens,
            "prefix": prompt[:prefix_hit_tokens],
            "suffix": prompt_chunk,
            "prefill_tokens": prefill_tokens,
        }
        if start_token != prefix_hit_tokens or not prompt_complete:
            item.update(
                {
                    "start_token": start_token,
                    "prompt_chunk": prompt_chunk,
                    "prompt_complete": prompt_complete,
                    "emits_token": prompt_complete,
                }
            )
        prefill_items.append(item)

    groups: dict[tuple[tuple[int, ...], int], list[dict[str, object]]] = {}
    for item in prefill_items:
        prompt_complete = bool(item.get("prompt_complete", True))
        emits_token = bool(item.get("emits_token", prompt_complete))
        if not prompt_complete or not emits_token:
            continue
        if int(item.get("start_token", item["prefix_hit_tokens"])) != int(item["prefix_hit_tokens"]):
            continue
        prefix = tuple(int(token_id) for token_id in item["prefix"])  # type: ignore[arg-type]
        prefix_hit_tokens = int(item["prefix_hit_tokens"])
        groups.setdefault((prefix, prefix_hit_tokens), []).append(item)
    prefill_groups: list[dict[str, object]] = []
    for (prefix, prefix_hit_tokens), items in groups.items():
        prefill_groups.append(
            {
                "request_ids": [str(item["request_id"]) for item in items],
                "rows": [int(item["row"]) for item in items],
                "prefix_hit_tokens": prefix_hit_tokens,
                "prefix": list(prefix),
                "suffix_tokens": [max(1, int(item["prefill_tokens"])) for item in items],
            }
        )

    result: dict[str, object] = {
        "op": "persistent_prompt_list_step",
        "step": int(meta_cpu[1].item()),
        "decode_request_ids": decode_request_ids,
        "decode_rows": decode_rows,
        "prefill": prefill_items,
        "prefill_groups": prefill_groups,
        "finished_after_prefill": [str(int(value)) for value in finished_cpu.tolist()],
        "temperature": float(temperature.detach().cpu().to(torch.float64).item()),
    }
    if bool(int(meta_cpu[8].item())):
        result["static_graph_buckets"] = True
    return result


def _token_budget_scheduler_for_group(
    group: Sequence[_QueuedGeneration],
    *,
    max_active_rows: int,
    max_scheduled_tokens: int,
    prefill_chunk_size: int | None = None,
    prefix_tokens: int = 0,
    arrival_steps: Sequence[int] | None = None,
) -> _TokenBudgetScheduler:
    if arrival_steps is not None and len(arrival_steps) != len(group):
        raise ValueError("arrival_steps must match group size")
    scheduler = _TokenBudgetScheduler(
        max_rows=max_active_rows,
        max_scheduled_tokens=max_scheduled_tokens,
        prefill_chunk_size=prefill_chunk_size,
    )
    prompts = [request.prompt for request in group]
    prefix_key: tuple[int, ...] | None = None
    if prefix_tokens > 0 and prompts:
        candidate = tuple(int(token_id) for token_id in prompts[0][:prefix_tokens])
        if len(candidate) == prefix_tokens and all(tuple(prompt[:prefix_tokens]) == candidate for prompt in prompts):
            prefix_key = candidate
    for index, request in enumerate(group):
        if request.max_tokens <= 0:
            continue
        prefix_hit_tokens = min(prefix_tokens, len(request.prompt)) if prefix_key is not None else 0
        arrival_step = 0 if arrival_steps is None else int(arrival_steps[index])
        scheduler.submit(
            _TokenBudgetRequest(
                str(index),
                prompt_tokens=len(request.prompt),
                max_new_tokens=request.max_tokens,
                arrival_step=arrival_step,
                prefix_hit_tokens=prefix_hit_tokens,
                prefix_key=prefix_key,
            )
        )
    return scheduler


def _token_budget_step_payload(
    plan: _TokenBudgetPlan,
    request_by_id: Mapping[str, _QueuedGeneration],
) -> dict[str, object]:
    command = _token_budget_model_step_command(plan)
    chunks: list[dict[str, object]] = []
    for chunk in command.chunks:
        request = request_by_id.get(chunk.request_id)
        if request is None:
            raise ValueError(f"missing token-budget request {chunk.request_id}")
        item: dict[str, object] = {
            "request_id": chunk.request_id,
            "row": int(chunk.row),
            "kind": chunk.kind,
            "start_token": int(chunk.start_token),
            "token_count": int(chunk.token_count),
            "prompt_complete": bool(chunk.prompt_complete),
            "emits_token": bool(chunk.emits_token),
        }
        if chunk.kind == "prefill":
            start = int(chunk.start_token)
            end = start + int(chunk.token_count)
            if start < 0 or end > len(request.prompt):
                raise ValueError("token-budget prefill chunk is outside the prompt")
            item.update(
                {
                    "prompt_chunk": list(request.prompt[start:end]),
                    "prompt_tokens": len(request.prompt),
                    "max_tokens": int(request.max_tokens),
                }
            )
            if chunk.prefix_key is not None:
                prefix = chunk.prefix_key
                if not isinstance(prefix, tuple) or not all(isinstance(token_id, int) for token_id in prefix):
                    raise ValueError("token-budget prefix_key must be a tuple of token ids")
                item["prefix"] = [int(token_id) for token_id in prefix]
        chunks.append(item)
    return {
        "op": "token_budget_step",
        "step": int(command.step),
        "chunks": chunks,
        "decode_rows": [int(row) for row in command.decode_rows],
        "prefill_rows": [int(row) for row in command.prefill_rows],
        "emit_request_ids": list(command.emit_request_ids),
        "emit_rows": [int(row) for row in command.emit_rows],
        "finished_request_ids": list(command.finished_request_ids),
        "scheduled_tokens": int(command.scheduled_tokens),
    }


def _token_budget_plan_is_decode_only(plan: _TokenBudgetPlan) -> bool:
    return bool(plan.chunks) and all(chunk.kind == "decode" for chunk in plan.chunks)


def _token_budget_decode_run_payload(
    plans: Sequence[_TokenBudgetPlan],
    request_by_id: Mapping[str, _QueuedGeneration],
) -> dict[str, object]:
    if not plans:
        raise ValueError("token-budget decode run requires at least one plan")
    steps: list[dict[str, object]] = []
    for plan in plans:
        if not _token_budget_plan_is_decode_only(plan):
            raise ValueError("token-budget decode run only accepts decode-only plans")
        steps.append(_token_budget_step_payload(plan, request_by_id))
    return {
        "op": "token_budget_decode_run",
        "steps": steps,
        "step_count": len(steps),
    }


def _token_budget_prompt_list_step_payload(
    plan: _TokenBudgetPlan,
    request_by_id: Mapping[str, _QueuedGeneration],
) -> dict[str, object] | None:
    command = _token_budget_model_step_command(plan)
    decode_request_ids: list[str] = []
    decode_rows: list[int] = []
    prefill_items: list[dict[str, object]] = []
    for chunk in command.chunks:
        request = request_by_id.get(chunk.request_id)
        if request is None:
            raise ValueError(f"missing token-budget request {chunk.request_id}")
        if chunk.kind == "decode":
            decode_request_ids.append(chunk.request_id)
            decode_rows.append(int(chunk.row))
            continue
        if chunk.kind != "prefill":
            raise ValueError(f"unsupported token-budget chunk kind: {chunk.kind}")
        start_token = int(chunk.start_token)
        end = start_token + int(chunk.token_count)
        if start_token < 0 or end > len(request.prompt):
            return None
        prefix: tuple[int, ...] = tuple(int(token_id) for token_id in request.prompt[:start_token])
        prefix_key = chunk.prefix_key
        if isinstance(prefix_key, tuple) and all(isinstance(token_id, int) for token_id in prefix_key):
            prefix = prefix_key
        elif prefix_key is not None:
            return None
        prefix_tokens = len(prefix)
        if tuple(request.prompt[:prefix_tokens]) != prefix:
            return None
        prompt_chunk = list(request.prompt[start_token:end])
        if not prompt_chunk:
            return None
        item = {
            "request_id": chunk.request_id,
            "row": int(chunk.row),
            "prompt": list(request.prompt),
            "max_tokens": int(request.max_tokens),
            "prefix_hit_tokens": prefix_tokens,
            "prefix": list(prefix),
            "suffix": prompt_chunk,
            "prefill_tokens": int(chunk.token_count),
        }
        if (
            start_token != prefix_tokens
            or not bool(chunk.prompt_complete)
            or not bool(chunk.emits_token)
            or end != len(request.prompt)
        ):
            item.update(
                {
                    "start_token": start_token,
                    "prompt_chunk": prompt_chunk,
                    "prompt_complete": bool(chunk.prompt_complete),
                    "emits_token": bool(chunk.emits_token),
                }
            )
        prefill_items.append(item)

    grouped: dict[tuple[tuple[int, ...], int], list[dict[str, object]]] = {}
    for item in prefill_items:
        prompt_complete = bool(item.get("prompt_complete", True))
        emits_token = bool(item.get("emits_token", prompt_complete))
        if not prompt_complete or not emits_token:
            continue
        if int(item.get("start_token", item["prefix_hit_tokens"])) != int(item["prefix_hit_tokens"]):
            continue
        prefix_values = item.get("prefix", [])
        if not isinstance(prefix_values, list):
            raise ValueError("token-budget prompt-list prefill prefix must be a list")
        prefix = tuple(int(token_id) for token_id in prefix_values)
        prefix_tokens = int(item["prefix_hit_tokens"])
        grouped.setdefault((prefix, prefix_tokens), []).append(item)
    prefill_groups: list[dict[str, object]] = []
    for (prefix, prefix_tokens), items in grouped.items():
        prefill_groups.append(
            {
                "request_ids": [str(item["request_id"]) for item in items],
                "rows": [int(item["row"]) for item in items],
                "prefix_hit_tokens": prefix_tokens,
                "prefix": list(prefix),
                "suffix_tokens": [max(1, int(item["prefill_tokens"])) for item in items],
            }
        )
    return {
        "op": "persistent_prompt_list_step",
        "step": int(command.step),
        "decode_request_ids": decode_request_ids,
        "decode_rows": decode_rows,
        "prefill": prefill_items,
        "prefill_groups": prefill_groups,
        "finished_after_prefill": list(command.finished_request_ids),
    }


def _token_budget_step_tensor_payload(
    payload: Mapping[str, object],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    chunks_obj = payload.get("chunks", [])
    if not isinstance(chunks_obj, list):
        raise ValueError("token-budget tensor payload requires chunk list")
    chunk_rows: list[list[int]] = []
    prefill_chunks: list[list[int]] = []
    emit_count = 0
    for chunk_obj in chunks_obj:
        if not isinstance(chunk_obj, Mapping):
            raise ValueError("token-budget tensor payload chunk must be a mapping")
        request_ordinal = _token_budget_request_ordinal(chunk_obj.get("request_id"))
        kind_text = str(chunk_obj.get("kind"))
        if kind_text == "prefill":
            kind_id = _TP_TOKEN_BUDGET_KIND_PREFILL
        elif kind_text == "decode":
            kind_id = _TP_TOKEN_BUDGET_KIND_DECODE
        else:
            raise ValueError(f"unsupported token-budget tensor payload kind: {kind_text}")
        prefill_index = -1
        prompt_tokens = -1
        max_tokens = -1
        if kind_id == _TP_TOKEN_BUDGET_KIND_PREFILL:
            prompt_chunk_obj = chunk_obj.get("prompt_chunk", [])
            if not isinstance(prompt_chunk_obj, list):
                raise ValueError("token-budget prefill tensor payload requires prompt_chunk")
            prefill_index = len(prefill_chunks)
            prefill_chunks.append([int(token_id) for token_id in prompt_chunk_obj])
            prompt_tokens = int(chunk_obj.get("prompt_tokens", -1))
            max_tokens = int(chunk_obj.get("max_tokens", -1))
        emits_token = int(bool(chunk_obj.get("emits_token", False)))
        emit_count += emits_token
        chunk_rows.append(
            [
                request_ordinal,
                int(chunk_obj["row"]),
                kind_id,
                int(chunk_obj["start_token"]),
                int(chunk_obj["token_count"]),
                int(bool(chunk_obj.get("prompt_complete", False))),
                emits_token,
                prefill_index,
                prompt_tokens,
                max_tokens,
            ]
        )
    chunk_tensor = torch.tensor(chunk_rows, dtype=torch.long, device=device)
    if not chunk_rows:
        chunk_tensor = torch.empty((0, _TP_TOKEN_BUDGET_CHUNK_FIELDS), dtype=torch.long, device=device)
    prefill_token_rows, prefill_lengths = _prompt_list_tensor_payload(prefill_chunks, device)
    finished_obj = payload.get("finished_request_ids", [])
    if not isinstance(finished_obj, list):
        raise ValueError("token-budget tensor payload requires finished_request_ids list")
    finished_ids = torch.tensor(
        [_token_budget_request_ordinal(request_id) for request_id in finished_obj],
        dtype=torch.long,
        device=device,
    )
    meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
    meta[0] = _TP_COMMAND_TOKEN_BUDGET_STEP
    meta[1] = int(payload.get("step", 0))
    meta[2] = int(chunk_tensor.size(0))
    meta[3] = _TP_TOKEN_BUDGET_CHUNK_FIELDS
    meta[4] = int(payload.get("scheduled_tokens", int(chunk_tensor[:, 4].sum().item()) if chunk_rows else 0))
    meta[5] = int(prefill_lengths.numel())
    meta[6] = int(prefill_token_rows.size(1)) if prefill_token_rows.ndim == 2 else 0
    meta[7] = emit_count
    meta[8] = int(finished_ids.numel())
    return meta, chunk_tensor, prefill_lengths, prefill_token_rows, finished_ids


def _token_budget_step_payload_from_tensor_payload(
    meta: Tensor,
    chunk_tensor: Tensor,
    prefill_lengths: Tensor,
    prefill_token_rows: Tensor,
    finished_ids: Tensor,
) -> dict[str, object]:
    meta_cpu = meta.detach().cpu().to(torch.long)
    if int(meta_cpu[0].item()) != _TP_COMMAND_TOKEN_BUDGET_STEP:
        raise ValueError("token-budget tensor payload has wrong command kind")
    expected_fields = int(meta_cpu[3].item())
    if expected_fields != _TP_TOKEN_BUDGET_CHUNK_FIELDS:
        raise ValueError("token-budget tensor payload has wrong chunk width")
    chunk_cpu = chunk_tensor.detach().cpu().to(torch.long)
    if chunk_cpu.ndim != 2 or chunk_cpu.size(1) != _TP_TOKEN_BUDGET_CHUNK_FIELDS:
        raise ValueError("token-budget tensor payload chunk tensor has invalid shape")
    if int(meta_cpu[2].item()) != int(chunk_cpu.size(0)):
        raise ValueError("token-budget tensor payload chunk count mismatch")
    prefill_lengths_list = [int(value) for value in prefill_lengths.detach().cpu().to(torch.long).tolist()]
    prefill_rows_cpu = prefill_token_rows.detach().cpu().to(torch.long)
    finished_list = [str(int(value)) for value in finished_ids.detach().cpu().to(torch.long).tolist()]

    chunks: list[dict[str, object]] = []
    decode_rows: list[int] = []
    prefill_rows: list[int] = []
    emit_request_ids: list[str] = []
    emit_rows: list[int] = []
    for row_values in chunk_cpu.tolist():
        request_id = str(int(row_values[0]))
        row = int(row_values[1])
        kind_id = int(row_values[2])
        if kind_id == _TP_TOKEN_BUDGET_KIND_PREFILL:
            kind = "prefill"
            prefill_rows.append(row)
        elif kind_id == _TP_TOKEN_BUDGET_KIND_DECODE:
            kind = "decode"
            decode_rows.append(row)
        else:
            raise ValueError(f"unsupported token-budget tensor payload kind id: {kind_id}")
        item: dict[str, object] = {
            "request_id": request_id,
            "row": row,
            "kind": kind,
            "start_token": int(row_values[3]),
            "token_count": int(row_values[4]),
            "prompt_complete": bool(row_values[5]),
            "emits_token": bool(row_values[6]),
        }
        if item["emits_token"]:
            emit_request_ids.append(request_id)
            emit_rows.append(row)
        if kind == "prefill":
            prefill_index = int(row_values[7])
            if prefill_index < 0 or prefill_index >= len(prefill_lengths_list):
                raise ValueError("token-budget tensor payload has invalid prefill index")
            length = prefill_lengths_list[prefill_index]
            item["prompt_chunk"] = prefill_rows_cpu[prefill_index, :length].tolist()
            prompt_tokens = int(row_values[8])
            max_tokens = int(row_values[9])
            if prompt_tokens >= 0:
                item["prompt_tokens"] = prompt_tokens
            if max_tokens >= 0:
                item["max_tokens"] = max_tokens
        chunks.append(item)
    return {
        "op": "token_budget_step",
        "step": int(meta_cpu[1].item()),
        "chunks": chunks,
        "decode_rows": decode_rows,
        "prefill_rows": prefill_rows,
        "emit_request_ids": emit_request_ids,
        "emit_rows": emit_rows,
        "finished_request_ids": finished_list,
        "scheduled_tokens": int(meta_cpu[4].item()),
    }


def _token_budget_decode_run_tensor_payload(
    payload: Mapping[str, object],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    steps_obj = payload.get("steps", [])
    if not isinstance(steps_obj, list) or not steps_obj:
        raise ValueError("token-budget decode-run tensor payload requires non-empty steps")
    step_rows: list[list[int]] = []
    chunk_rows: list[list[int]] = []
    finished_values: list[int] = []
    for step_obj in steps_obj:
        if not isinstance(step_obj, Mapping):
            raise ValueError("token-budget decode-run tensor step must be a mapping")
        chunks_obj = step_obj.get("chunks", [])
        if not isinstance(chunks_obj, list) or not chunks_obj:
            raise ValueError("token-budget decode-run tensor step requires chunks")
        chunk_start = len(chunk_rows)
        finished_start = len(finished_values)
        for chunk_obj in chunks_obj:
            if not isinstance(chunk_obj, Mapping):
                raise ValueError("token-budget decode-run tensor chunk must be a mapping")
            if str(chunk_obj.get("kind")) != "decode":
                raise ValueError("token-budget decode-run tensor payload only accepts decode chunks")
            request_ordinal = _token_budget_request_ordinal(chunk_obj.get("request_id"))
            chunk_rows.append(
                [
                    request_ordinal,
                    int(chunk_obj["row"]),
                    _TP_TOKEN_BUDGET_KIND_DECODE,
                    int(chunk_obj["start_token"]),
                    int(chunk_obj["token_count"]),
                    int(bool(chunk_obj.get("prompt_complete", False))),
                    int(bool(chunk_obj.get("emits_token", False))),
                    -1,
                    -1,
                    -1,
                ]
            )
        finished_obj = step_obj.get("finished_request_ids", [])
        if not isinstance(finished_obj, list):
            raise ValueError("token-budget decode-run tensor step requires finished_request_ids list")
        finished_values.extend(_token_budget_request_ordinal(request_id) for request_id in finished_obj)
        step_rows.append(
            [
                int(step_obj.get("step", 0)),
                chunk_start,
                len(chunk_rows) - chunk_start,
                finished_start,
                len(finished_values) - finished_start,
            ]
        )
    meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
    meta[0] = _TP_COMMAND_TOKEN_BUDGET_DECODE_RUN
    meta[1] = len(step_rows)
    meta[2] = len(chunk_rows)
    meta[3] = _TP_TOKEN_BUDGET_CHUNK_FIELDS
    meta[4] = len(finished_values)
    meta[5] = _TP_TOKEN_BUDGET_DECODE_RUN_STEP_FIELDS
    step_tensor = torch.tensor(step_rows, dtype=torch.long, device=device)
    chunk_tensor = torch.tensor(chunk_rows, dtype=torch.long, device=device)
    if not chunk_rows:
        chunk_tensor = torch.empty((0, _TP_TOKEN_BUDGET_CHUNK_FIELDS), dtype=torch.long, device=device)
    finished_ids = torch.tensor(finished_values, dtype=torch.long, device=device)
    return meta, step_tensor, chunk_tensor, finished_ids


def _token_budget_decode_run_payload_from_tensor_payload(
    meta: Tensor,
    step_tensor: Tensor,
    chunk_tensor: Tensor,
    finished_ids: Tensor,
) -> dict[str, object]:
    meta_cpu = meta.detach().cpu().to(torch.long)
    if int(meta_cpu[0].item()) != _TP_COMMAND_TOKEN_BUDGET_DECODE_RUN:
        raise ValueError("token-budget decode-run tensor payload has wrong command kind")
    expected_chunk_fields = int(meta_cpu[3].item())
    if expected_chunk_fields != _TP_TOKEN_BUDGET_CHUNK_FIELDS:
        raise ValueError("token-budget decode-run tensor payload has wrong chunk width")
    expected_step_fields = int(meta_cpu[5].item())
    if expected_step_fields != _TP_TOKEN_BUDGET_DECODE_RUN_STEP_FIELDS:
        raise ValueError("token-budget decode-run tensor payload has wrong step width")
    step_cpu = step_tensor.detach().cpu().to(torch.long)
    if step_cpu.ndim != 2 or step_cpu.size(1) != _TP_TOKEN_BUDGET_DECODE_RUN_STEP_FIELDS:
        raise ValueError("token-budget decode-run tensor step tensor has invalid shape")
    if int(meta_cpu[1].item()) != int(step_cpu.size(0)):
        raise ValueError("token-budget decode-run tensor step count mismatch")
    chunk_cpu = chunk_tensor.detach().cpu().to(torch.long)
    if chunk_cpu.ndim != 2 or chunk_cpu.size(1) != _TP_TOKEN_BUDGET_CHUNK_FIELDS:
        raise ValueError("token-budget decode-run tensor chunk tensor has invalid shape")
    if int(meta_cpu[2].item()) != int(chunk_cpu.size(0)):
        raise ValueError("token-budget decode-run tensor chunk count mismatch")
    finished_cpu = finished_ids.detach().cpu().to(torch.long)
    if int(meta_cpu[4].item()) != int(finished_cpu.numel()):
        raise ValueError("token-budget decode-run tensor finished count mismatch")

    steps: list[dict[str, object]] = []
    for row_values in step_cpu.tolist():
        step = int(row_values[0])
        chunk_start = int(row_values[1])
        chunk_count = int(row_values[2])
        finished_start = int(row_values[3])
        finished_count = int(row_values[4])
        if chunk_start < 0 or chunk_count < 1 or chunk_start + chunk_count > chunk_cpu.size(0):
            raise ValueError("token-budget decode-run tensor step has invalid chunk slice")
        if finished_start < 0 or finished_count < 0 or finished_start + finished_count > finished_cpu.numel():
            raise ValueError("token-budget decode-run tensor step has invalid finished slice")
        chunks: list[dict[str, object]] = []
        decode_rows: list[int] = []
        emit_request_ids: list[str] = []
        emit_rows: list[int] = []
        scheduled_tokens = 0
        for chunk_values in chunk_cpu[chunk_start : chunk_start + chunk_count].tolist():
            if int(chunk_values[2]) != _TP_TOKEN_BUDGET_KIND_DECODE:
                raise ValueError("token-budget decode-run tensor payload only supports decode chunks")
            request_id = str(int(chunk_values[0]))
            row = int(chunk_values[1])
            emits_token = bool(chunk_values[6])
            chunks.append(
                {
                    "request_id": request_id,
                    "row": row,
                    "kind": "decode",
                    "start_token": int(chunk_values[3]),
                    "token_count": int(chunk_values[4]),
                    "prompt_complete": bool(chunk_values[5]),
                    "emits_token": emits_token,
                }
            )
            decode_rows.append(row)
            scheduled_tokens += int(chunk_values[4])
            if emits_token:
                emit_request_ids.append(request_id)
                emit_rows.append(row)
        finished_slice = finished_cpu[finished_start : finished_start + finished_count]
        steps.append(
            {
                "op": "token_budget_step",
                "step": step,
                "chunks": chunks,
                "decode_rows": decode_rows,
                "prefill_rows": [],
                "emit_request_ids": emit_request_ids,
                "emit_rows": emit_rows,
                "finished_request_ids": [str(int(value)) for value in finished_slice.tolist()],
                "scheduled_tokens": scheduled_tokens,
            }
        )
    return {
        "op": "token_budget_decode_run",
        "steps": steps,
        "step_count": len(steps),
    }


def _token_budget_request_ordinal(request_id: object) -> int:
    try:
        value = int(str(request_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("token-budget tensor payload requires numeric request ids") from exc
    if value < 0:
        raise ValueError("token-budget tensor payload request ids must be non-negative")
    return value


_TENSOR_PARALLEL_CONTROL_GROUP: object | None = None
_TENSOR_PARALLEL_CONTROL_GROUP_LOCK = threading.Lock()


@dataclass(frozen=True)
class _GenerationDone:
    pass


@dataclass(frozen=True)
class _GenerationResult:
    tokens: list[int]


@dataclass(frozen=True)
class _PrefixCachedPrompt:
    index: int
    prompt: list[int]
    entry: TensorPrefixCacheEntry
    prefix_tokens: int


@runtime_checkable
class _IncrementalGenerationModel(Protocol):
    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs: object) -> object: ...

    def forward(self, input_ids: Tensor, **kwargs: object) -> tuple[Tensor, object]: ...


@runtime_checkable
class _GenerateModel(Protocol):
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        eos_token_id: int | None,
    ) -> Tensor: ...


class _ByteFallbackTokenizer:
    eos_token_id: int | None = None

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = max(2, vocab_size)
        self.stop_token_ids: frozenset[int] = frozenset()

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        return self.encode(_format_messages(messages))

    def encode(self, text: str) -> list[int]:
        limit = min(self.vocab_size, 256)
        return [min(ord(ch), limit - 1) for ch in text] or [1]

    def decode_token(self, token_id: int) -> str:
        if 32 <= token_id <= 126:
            return chr(token_id)
        return chr(32 + (token_id % 95))

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.decode_token(int(token_id)) for token_id in token_ids)


class _TransformersChatTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.stop_token_ids = _chat_stop_token_ids(tokenizer)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            encoded = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return _coerce_token_ids(encoded)
        return self.encode(_format_messages(messages))

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return [int(token_id) for token_id in encoded] or [self.eos_token_id or 0]

    @lru_cache(maxsize=8192)
    def decode_token(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=True))  # type: ignore[attr-defined]

    def decode(self, token_ids: Iterable[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))  # type: ignore[attr-defined]


def load_chat_tokenizer(
    config: OpenAIServerConfig,
    vocab_size: int,
) -> _ByteFallbackTokenizer | _TransformersChatTokenizer:
    tokenizer_name = config.tokenizer or config.model
    if tokenizer_name in {"byte", "bytes", "fallback", "tiny"} or (
        config.tokenizer is None and config.model_kind.startswith("tiny")
    ):
        return _ByteFallbackTokenizer(vocab_size)
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "transformers is required for OpenAI-compatible text serving. "
            "Install TorchInferno with the 'serve' extra or pass --tokenizer byte for smoke tests."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
        token=config.token,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )
    return _TransformersChatTokenizer(tokenizer)


def _chat_stop_token_ids(tokenizer: object) -> frozenset[int]:
    token_ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    token_ids.update(_coerce_optional_token_ids(eos_token_id))
    token_ids.update(_coerce_optional_token_ids(getattr(tokenizer, "eos_token_ids", None)))
    token_ids.update(_known_chat_terminator_ids(tokenizer))
    token_ids.update(_added_chat_control_token_ids(tokenizer))
    return frozenset(token_ids)


def _known_chat_terminator_ids(tokenizer: object) -> set[int]:
    token_ids: set[int] = set()
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return token_ids
    vocab_size = getattr(tokenizer, "vocab_size", None)
    all_special_ids = _coerce_optional_token_ids(getattr(tokenizer, "all_special_ids", None))
    for token in ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"):
        try:
            token_id = convert(token)
        except Exception:
            continue
        for coerced in _coerce_optional_token_ids(token_id):
            if vocab_size is not None and coerced >= int(vocab_size):
                token_ids.add(coerced)
            elif coerced in all_special_ids:
                token_ids.add(coerced)
            elif token in getattr(tokenizer, "all_special_tokens", ()):
                token_ids.add(coerced)
    return token_ids


def _added_chat_control_token_ids(tokenizer: object) -> set[int]:
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return set()
    try:
        added_vocab = get_added_vocab()
    except Exception:
        return set()
    if not isinstance(added_vocab, Mapping):
        return set()
    token_ids: set[int] = set()
    for token, token_id in added_vocab.items():
        if isinstance(token, str) and token.startswith("<|") and token.endswith("|>"):
            token_ids.update(_coerce_optional_token_ids(token_id))
    return token_ids


def _coerce_optional_token_ids(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, set, str, bytes)):
        value = value.tolist()  # type: ignore[assignment]
    if isinstance(value, (list, tuple, set)):
        token_ids: set[int] = set()
        for item in value:
            token_ids.update(_coerce_optional_token_ids(item))
        return token_ids
    try:
        token_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return set()
    if token_id < 0:
        return set()
    return {token_id}


def _tokenizer_stop_token_ids(tokenizer: object) -> frozenset[int]:
    token_ids = _coerce_optional_token_ids(getattr(tokenizer, "stop_token_ids", None))
    token_ids.update(_coerce_optional_token_ids(getattr(tokenizer, "eos_token_id", None)))
    return frozenset(token_ids)


def _coerce_token_ids(encoded: object) -> list[int]:
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    if input_ids is not None:
        encoded = input_ids
    if isinstance(encoded, Tensor):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist") and not isinstance(encoded, (list, tuple, str, bytes)):
        encoded = encoded.tolist()  # type: ignore[assignment]
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and _is_token_sequence(encoded[0]):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]  # type: ignore[union-attr]


def _is_token_sequence(value: object) -> bool:
    if isinstance(value, Tensor):
        return value.ndim == 1
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, str, bytes)):
        value = value.tolist()
    return isinstance(value, (list, tuple))


def _supports_incremental_generation(model: object) -> bool:
    return (
        isinstance(model, _IncrementalGenerationModel)
        and callable(getattr(model, "allocate_cache", None))
        and callable(getattr(model, "forward", None))
    )


def _generate_with_model(
    model: object,
    input_ids: Tensor,
    *,
    max_tokens: int,
    temperature: float,
    eos_token_id: int | None,
) -> Tensor:
    if not isinstance(model, _GenerateModel):
        raise TypeError("model must expose either allocate_cache/forward or generate")
    return model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )


def _generated_rows_with_model(
    model: object,
    input_ids: Tensor,
    *,
    max_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    stop_token_ids: frozenset[int],
) -> list[list[int]]:
    generated = _generate_with_model(
        model,
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    rows = generated[:, input_ids.size(1) :].detach().cpu().tolist()
    return _trim_rows_at_stop(rows, stop_token_ids)


def _iter_generated_steps(
    rows: Sequence[Sequence[int]],
    max_tokens: int,
    stop_token_ids: frozenset[int],
) -> Iterator[list[int | None]]:
    finished = [False for _ in rows]
    for step in range(max_tokens):
        step_tokens: list[int | None] = []
        for row_index, row in enumerate(rows):
            if finished[row_index] or step >= len(row):
                step_tokens.append(None)
                continue
            token_id = int(row[step])
            step_tokens.append(token_id)
            if token_id in stop_token_ids:
                finished[row_index] = True
        if all(token is None for token in step_tokens):
            break
        yield step_tokens


class OpenAICompletionEngine:
    def __init__(
        self,
        model: object,
        tokenizer: _ByteFallbackTokenizer | _TransformersChatTokenizer,
        *,
        model_id: str,
        device: torch.device,
        cache_backend: str = "dense",
        page_size: int = 16,
        max_model_len: int | None = None,
        max_batch_size: int = 128,
        batch_wait_ms: float = 10.0,
        single_request_admission_wait_ms: float | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stop_token_ids = _tokenizer_stop_token_ids(tokenizer)
        self.model_id = model_id
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.max_model_len = max_model_len
        self.max_batch_size = _effective_openai_max_batch_size(model, device, max_batch_size)
        self.batch_wait_s = max(0.0, batch_wait_ms / 1000.0)
        default_single_request_wait_ms = (
            single_request_admission_wait_ms
            if single_request_admission_wait_ms is not None
            else 0.0
        )
        self.single_request_admission_wait_s = (
            env_float(
                "TORCHINFERNO_OPENAI_SINGLE_ADMISSION_WAIT_MS",
                float(default_single_request_wait_ms),
                minimum=0.0,
            )
            / 1000.0
        )
        # Tensor-parallel workers consume one command stream from rank 0. Keep
        # rank 0 on the queued path so direct requests cannot interleave with
        # batched requests in a different order from the worker ranks.
        self.single_request_fast_path = not _is_tensor_parallel_primary_model(model)
        self._generation_queue: "queue.Queue[_QueuedGeneration | None]" = queue.Queue()
        self._model_lock = threading.Lock()
        self._live_request_condition = threading.Condition()
        self._live_requests = 0
        self._closed = False
        self._worker: threading.Thread | None = None
        self._cache_pool: dict[tuple[int, int, str, int, str], object] = {}
        self._microbatch_cache_pool: dict[tuple[int, int, int, str, int, str], object] = {}
        self._single_prefill_capture_seen: dict[tuple[int, int, int, int, bool, str], int] = {}
        self._batched_prefill_capture_seen: dict[tuple[int, int, int, int, bool, bool, str], int] = {}
        self._completed_queue_batches = 0
        self._idle_since_s = time.perf_counter()
        self._cleanup_after_idle = False
        self._prefix_cache_entry: TensorPrefixCacheEntry | None = None
        self._prefix_cache_entries: dict[tuple[int, ...], TensorPrefixCacheEntry] = {}
        self._prompt_logits_cache: dict[tuple[int, ...], Tensor] = {}
        self._prompt_token_cache: dict[str, list[int]] = {}
        self._prompt_token_cache_lock = threading.Lock()
        self._phase_timing_enabled = env_flag("TORCHINFERNO_OPENAI_PHASE_TIMINGS")
        self._phase_records: list[dict[str, float]] = []
        self._phase_records_lock = threading.Lock()
        self._queue_profile_path = os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", "")
        self._queue_profile_lock = threading.Lock()
        self._queue_profile_next_sequence = 0
        self._persistent_prompt_list_step_state: _PersistentPromptListStepState | None = None
        self._persistent_prompt_list_step_last_result: _PersistentPromptListStepResult | None = None
        self._token_budget_step_state: _TokenBudgetStepState | None = None
        self._token_budget_step_last_result: _TokenBudgetStepResult | None = None
        self._warmup_tokenizer()
        self._warmup_tensor_parallel_control_group()
        self._warmup_tensor_parallel_model()
        if not _is_tensor_parallel_worker_model(model):
            worker_fn = self._unified_scheduler_worker if self._should_use_unified_scheduler() else self._batch_worker
            self._worker = threading.Thread(target=worker_fn, name="torchinferno-openai-batcher", daemon=True)
            self._worker.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._generation_queue.put(None)
            self._worker.join(timeout=10)
        _broadcast_tensor_parallel_stop(self.model)

    def generate_chat_tokens(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[int]:
        phase = self._new_phase_record()
        self._enter_live_request()
        try:
            self._mark_phase(phase, "entered_live_request")
            prompt = self._encode_chat_prompt(messages, max_tokens=max_tokens)
            self._mark_phase(phase, "encoded_prompt")
            if self._try_acquire_single_request_model(temperature=temperature):
                try:
                    self._mark_phase(phase, "acquired_model")
                    yield from self._generate_prompt_tokens_with_phase(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        phase=phase,
                    )
                finally:
                    self._model_lock.release()
            else:
                self._mark_phase(phase, "queued_generation")
                yield from self._submit_generation(prompt, max_tokens=max_tokens, temperature=temperature)
        finally:
            self._exit_live_request()

    def complete_chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        self._enter_live_request()
        try:
            prompt = self._encode_chat_prompt(messages, max_tokens=max_tokens)
            if self._try_acquire_single_request_model(temperature=temperature):
                try:
                    tokens = self._generate_prompt_token_list(prompt, max_tokens=max_tokens, temperature=temperature)
                finally:
                    self._model_lock.release()
            else:
                tokens = self._submit_completion(prompt, max_tokens=max_tokens, temperature=temperature)
        finally:
            self._exit_live_request()
        return CompletionResult(tokens=tokens, prompt_tokens=len(prompt))

    def _encode_chat_prompt(self, messages: list[dict[str, object]], *, max_tokens: int) -> list[int]:
        prompt = self._cached_encode_chat_prompt(messages)
        if self.max_model_len is not None and len(prompt) + max_tokens > self.max_model_len:
            prompt_budget = max(1, self.max_model_len - max_tokens)
            prompt = prompt[-prompt_budget:]
        return prompt

    def _cached_encode_chat_prompt(self, messages: list[dict[str, object]]) -> list[int]:
        max_entries = env_int("TORCHINFERNO_OPENAI_PROMPT_TOKEN_CACHE_MAX_ENTRIES", 4096, minimum=0)
        if max_entries <= 0:
            return self.tokenizer.encode_messages(messages)
        cache_key = _chat_prompt_cache_key(messages)
        cache = self._prompt_token_cache_map()
        with self._prompt_token_cache_lock:
            cached = cache.get(cache_key)
            if cached is not None:
                cache.pop(cache_key, None)
                cache[cache_key] = cached
                return list(cached)
        prompt = self.tokenizer.encode_messages(messages)
        with self._prompt_token_cache_lock:
            cache.pop(cache_key, None)
            cache[cache_key] = list(prompt)
            while len(cache) > max_entries:
                cache.pop(next(iter(cache)))
        return prompt

    def _prompt_token_cache_map(self) -> dict[str, list[int]]:
        cache = getattr(self, "_prompt_token_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._prompt_token_cache = cache
        if getattr(self, "_prompt_token_cache_lock", None) is None:
            self._prompt_token_cache_lock = threading.Lock()
        return cache

    def _submit_generation(self, prompt: list[int], *, max_tokens: int, temperature: float) -> Iterator[int]:
        if self._closed:
            raise RuntimeError("OpenAI completion engine is closed")
        responses: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        profile_queue = bool(self._queue_profile_path_value())
        queued_at_s = time.perf_counter() if profile_queue else 0.0
        queue_sequence = self._next_queue_profile_sequence() if profile_queue else -1
        self._generation_queue.put(
            _QueuedGeneration(
                prompt,
                max_tokens,
                temperature,
                True,
                responses,
                queued_at_s=queued_at_s,
                queue_sequence=queue_sequence,
            )
        )
        while True:
            item = responses.get()
            if isinstance(item, _GenerationDone):
                break
            if isinstance(item, BaseException):
                raise item
            yield int(item)

    def _submit_completion(self, prompt: list[int], *, max_tokens: int, temperature: float) -> list[int]:
        if self._closed:
            raise RuntimeError("OpenAI completion engine is closed")
        responses: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        profile_queue = bool(self._queue_profile_path_value())
        queued_at_s = time.perf_counter() if profile_queue else 0.0
        queue_sequence = self._next_queue_profile_sequence() if profile_queue else -1
        self._generation_queue.put(
            _QueuedGeneration(
                prompt,
                max_tokens,
                temperature,
                False,
                responses,
                queued_at_s=queued_at_s,
                queue_sequence=queue_sequence,
            )
        )
        item = responses.get()
        if isinstance(item, BaseException):
            raise item
        if not isinstance(item, _GenerationResult):
            raise RuntimeError(f"unexpected completion response: {item!r}")
        return item.tokens

    def _batch_worker(self) -> None:
        while True:
            first = self._generation_queue.get()
            if first is None:
                return
            if self._should_use_tensor_parallel_online_batcher(first):
                with self._model_lock:
                    self._maybe_cleanup_runtime_after_idle()
                    self._run_tensor_parallel_online_batcher(first)
                    self._completed_queue_batches += 1
                continue
            batch = [first]
            batch_limit = self._queued_batch_limit(first)
            initial_wait_s = self._queued_initial_batch_wait_s(first)
            if initial_wait_s > 0.0:
                self._collect_batch_until_deadline(
                    batch,
                    limit=batch_limit,
                    wait_s=initial_wait_s,
                    reset_deadline_on_item=False,
                    stop_when_live_batch_ready=False,
                )
            self._drain_ready_requests(batch, limit=batch_limit)
            if len(batch) > 1 or self._has_multiple_live_requests():
                secondary_wait_s = max(
                    self._queued_batch_wait_s(first) if batch else self.batch_wait_s,
                    self.batch_wait_s,
                )
                self._collect_batch_until_deadline(batch, limit=batch_limit, wait_s=secondary_wait_s)
            with self._model_lock:
                self._drain_ready_requests(batch, limit=batch_limit)
                self._maybe_cleanup_runtime_after_idle()
                self._run_queued_batch(batch)
                self._completed_queue_batches += 1
                while self._has_multiple_live_requests() or not self._generation_queue.empty():
                    next_batch: list[_QueuedGeneration] = []
                    self._drain_ready_requests(next_batch, limit=batch_limit)
                    if not next_batch:
                        try:
                            item = self._generation_queue.get(timeout=self.batch_wait_s)
                        except queue.Empty:
                            break
                        if item is None:
                            self._generation_queue.put(None)
                            break
                        next_batch.append(item)
                    self._drain_ready_requests(next_batch, limit=batch_limit)
                    if self._has_multiple_live_requests() and len(next_batch) < batch_limit:
                        self._collect_batch_until_deadline(
                            next_batch, limit=batch_limit, wait_s=self.batch_wait_s,
                        )
                    if (
                        len(next_batch) < batch_limit
                        and self._has_multiple_live_requests()
                        and not self._generation_queue.empty()
                    ):
                        self._drain_ready_requests(next_batch, limit=batch_limit)
                    if not next_batch:
                        break
                    self._run_queued_batch(next_batch)
                    self._completed_queue_batches += 1

    def _warmup_unified_scheduler_cache(self, vocab_size: int) -> None:
        max_active = self._online_serving_max_active()
        # Persistent cache holds active rows plus extra rows for shared prompt
        # prefixes, so the continuous batcher can prefill only suffixes.
        total_rows = max_active + self._online_serving_prefix_rows()
        cache_batch = total_rows
        max_seq_len = env_int(
            "TORCHINFERNO_OPENAI_UNIFIED_MAX_SEQ_LEN",
            getattr(self, "max_model_len", None) or 768,
            minimum=64,
        )
        cache = _allocate_cache(
            self.model, cache_batch, _generation_cache_capacity(self.model, max_seq_len),
            device=self.device, cache_backend=self.cache_backend, page_size=self.page_size,
        )
        _reset_generation_cache(cache)
        try:
            cache._skip_capture_sync = True
        except Exception:
            pass
        prompt_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKENS", 32, minimum=1)
        # Decode touches only active rows (<= max_active), addressed via row
        # indices, so warm the active-range buckets, not the full cache size.
        batch_sizes = sorted(
            {1, 2, 4, 8, 16, 32, 48, 64, max_active, cache_batch}
            & set(range(1, cache_batch + 1))
        )
        with _tensor_parallel_symm_mem_allreduce_scope(
            self.model, self.device, max_tokens=1, temperature=0.0,
        ):
            for bs in batch_sizes:
                _set_generation_cache_seq_len(cache, prompt_tokens)
                decode_input_ids = torch.zeros(bs, 1, dtype=torch.long, device=self.device)
                row_indices = torch.arange(bs, dtype=torch.long, device=self.device)
                seq_lens_tensor = torch.full((bs,), prompt_tokens, dtype=torch.long, device=self.device)
                try:
                    _try_decode_ragged_token_graph(
                        self.model, decode_input_ids, cache, seq_lens=seq_lens_tensor,
                        row_indices=row_indices, temperature=0.0, allow_capture=True,
                    )
                except Exception as _warmup_exc:
                    import sys as _wsys
                    print(f"[WARMUP] graph capture failed bs={bs}: {_warmup_exc}", file=_wsys.stderr, flush=True)
                _reset_generation_cache(cache)
        self._persistent_serving_cache = cache

    def _warmup_token_budget_prefill_graphs(self, prefill_chunk_size: int) -> None:
        state = self._token_budget_step_state
        if state is None:
            return
        vocab_size = max(1, int(getattr(getattr(self.model, "config", object()), "vocab_size", 1)))
        from torchinferno.models.llama3.tensor_parallel import _PREFILL_TOKEN_BUCKETS
        buckets = [b for b in _PREFILL_TOKEN_BUCKETS if b <= max(prefill_chunk_size, 256)]
        if not buckets:
            return
        with torch.inference_mode():
            for bucket in buckets:
                cache_view = _cache_row_slice(state.cache, 0, 1)
                if cache_view is None:
                    continue
                _reset_generation_cache(cache_view)
                input_ids = (torch.arange(bucket, device=self.device, dtype=torch.long) % vocab_size)[None, :]
                try:
                    _try_prefill_logits_graph(self.model, input_ids, cache_view, allow_capture=True)
                except Exception:
                    pass
                _reset_generation_cache(cache_view)

    def _should_use_unified_scheduler(self) -> bool:
        if not env_flag("TORCHINFERNO_OPENAI_UNIFIED_SCHEDULER", False):
            return False
        if not _is_tensor_parallel_primary_model(self.model):
            return False
        if self.device.type != "cuda":
            return False
        return hasattr(self.model, "allocate_cache")

    def _unified_scheduler_worker(self) -> None:
        max_active = min(64, _effective_openai_max_batch_size(self.model, self.device, self.max_batch_size))
        max_tokens_per_step = env_int("TORCHINFERNO_OPENAI_UNIFIED_MAX_TOKENS", 2048, minimum=1)
        prefill_chunk = env_int("TORCHINFERNO_OPENAI_UNIFIED_PREFILL_CHUNK", 512, minimum=1)
        decode_run_steps = env_int("TORCHINFERNO_OPENAI_UNIFIED_DECODE_RUN_STEPS", 8, minimum=1)
        scheduler = _TokenBudgetScheduler(
            max_rows=max_active,
            max_scheduled_tokens=max_tokens_per_step,
            prefill_chunk_size=prefill_chunk,
            decode_first=True,
        )
        request_map: dict[str, _QueuedGeneration] = {}
        next_id = 0
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        max_model_len = env_int(
            "TORCHINFERNO_OPENAI_UNIFIED_MAX_SEQ_LEN",
            getattr(self, "max_model_len", None) or 512,
            minimum=64,
        )
        cache_batch = _generation_cache_batch_capacity(self.model, max_active)
        finished_ids: tuple[str, ...] = ()
        shutdown = False

        def _drain_queue() -> bool:
            nonlocal next_id, shutdown
            admitted = False
            while True:
                try:
                    item = self._generation_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._generation_queue.put(None)
                    shutdown = True
                    break
                if not item.stream:
                    item.responses.put(RuntimeError("unified scheduler only supports streaming"))
                    continue
                rid = str(next_id)
                next_id += 1
                request_map[rid] = item
                scheduler.submit(_TokenBudgetRequest(
                    request_id=rid,
                    prompt_tokens=len(item.prompt),
                    max_new_tokens=max(1, item.max_tokens),
                ))
                admitted = True
            return admitted

        def _emit_result(result: object) -> tuple[str, ...]:
            new_finished: list[str] = []
            all_tokens: dict[str, int | None] = {}
            if hasattr(result, "decode_tokens"):
                all_tokens.update(result.decode_tokens)
            if hasattr(result, "prefill_tokens"):
                all_tokens.update(result.prefill_tokens)
            if hasattr(result, "finished_request_ids"):
                for rid in result.finished_request_ids:
                    if rid not in new_finished:
                        new_finished.append(rid)
            for rid, token_id in all_tokens.items():
                req = request_map.get(rid)
                if req is None or req.done:
                    continue
                if token_id is None:
                    continue
                tok = int(token_id)
                if tok in stop_token_ids:
                    _finish_stream_request(req)
                    if rid not in new_finished:
                        new_finished.append(rid)
                else:
                    req.responses.put(tok)
            for rid in new_finished:
                req = request_map.pop(rid, None)
                if req is not None and not req.done:
                    _finish_stream_request(req)
            return tuple(new_finished)

        with self._model_lock:
            try:
                _broadcast_tensor_parallel_token_budget_start(
                    self.model,
                    max_seq_len=max_model_len,
                    max_active_rows=cache_batch,
                    temperature=0.0,
                    max_tokens=max_tokens_per_step,
                )
                ext_cache = getattr(self, "_persistent_serving_cache", None)
                self._start_token_budget_step_state(
                    cache_batch_size=cache_batch,
                    max_seq_len=max_model_len,
                    temperature=0.0,
                    external_cache=ext_cache,
                )
                symm_scope = _tensor_parallel_symm_mem_allreduce_scope(
                    self.model,
                    self.device,
                    max_tokens=max_tokens_per_step,
                    temperature=0.0,
                )
                symm_scope.__enter__()
                _sync_tensor_parallel_command(self.model, self.device)

                while not shutdown:
                    _drain_queue()
                    if shutdown:
                        break

                    if not scheduler.has_work():
                        try:
                            item = self._generation_queue.get(timeout=0.005)
                        except queue.Empty:
                            continue
                        if item is None:
                            break
                        rid = str(next_id)
                        next_id += 1
                        request_map[rid] = item
                        scheduler.submit(_TokenBudgetRequest(
                            request_id=rid,
                            prompt_tokens=len(item.prompt),
                            max_new_tokens=max(1, item.max_tokens),
                        ))
                        _drain_queue()
                        continue

                    prior_finished = finished_ids
                    plan = scheduler.step(finished_request_ids=finished_ids)
                    finished_ids = ()
                    if not plan.chunks:
                        continue

                    if decode_run_steps > 1 and _token_budget_plan_is_decode_only(plan):
                        plans = [plan]
                        while (
                            len(plans) < decode_run_steps
                            and not plans[-1].finished_request_ids
                            and scheduler.has_work()
                        ):
                            next_plan = scheduler.step(finished_request_ids=())
                            if not next_plan.chunks:
                                if next_plan.finished_request_ids:
                                    finished_ids = next_plan.finished_request_ids
                                    break
                                continue
                            if not _token_budget_plan_is_decode_only(next_plan):
                                plan = next_plan
                                break
                            plans.append(next_plan)
                        else:
                            plan = None
                        if plans:
                            payload = _token_budget_decode_run_payload(plans, request_map)
                            payload["temperature"] = 0.0
                            if prior_finished:
                                steps = payload.get("steps", [])
                                if steps and isinstance(steps[0], dict):
                                    existing = list(steps[0].get("finished_request_ids", []))
                                    existing.extend(prior_finished)
                                    steps[0]["finished_request_ids"] = existing
                                prior_finished = ()
                            _broadcast_tensor_parallel_token_budget_decode_run(self.model, payload)
                            result = self._handle_token_budget_decode_run_payload(payload)
                            run_finished: list[str] = []
                            for step_result in result.step_results:
                                run_finished.extend(_emit_result(step_result))
                            finished_ids = tuple(run_finished) + (finished_ids or ())
                            _drain_queue()
                            self._completed_queue_batches += 1
                        if plan is None:
                            continue

                    payload = _token_budget_step_payload(plan, request_map)
                    payload["temperature"] = 0.0
                    if prior_finished:
                        existing = list(payload.get("finished_request_ids", []))
                        existing.extend(prior_finished)
                        payload["finished_request_ids"] = existing
                    _broadcast_tensor_parallel_token_budget_step(self.model, payload)
                    result = self._handle_token_budget_step_payload(payload)
                    finished_ids = _emit_result(result)
                    _drain_queue()
                    self._completed_queue_batches += 1

            except BaseException as exc:
                for req in request_map.values():
                    if not req.done:
                        req.responses.put(exc)
                        req.done = True
            finally:
                symm_scope.__exit__(None, None, None)
                handler = getattr(self, "_handle_token_budget_close_payload", None)
                if callable(handler):
                    handler({})
                _broadcast_tensor_parallel_token_budget_close(self.model)
                _sync_tensor_parallel_command(self.model, self.device)
                self._token_budget_step_state = None

    def _online_serving_max_active(self) -> int:
        cap = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_ACTIVE", 64, minimum=1)
        effective = _effective_openai_max_batch_size(self.model, self.device, self.max_batch_size)
        return max(1, min(cap, effective))

    def _online_serving_prefix_rows(self) -> int:
        # Extra persistent-cache rows that hold shared prompt prefixes so the
        # continuous batcher prefills only per-request suffixes (like the batch
        # worker), instead of re-prefilling the full prompt each time.
        return env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", 8, minimum=0)

    def _should_use_tensor_parallel_online_batcher(self, first: _QueuedGeneration) -> bool:
        explicit = "TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER" in os.environ
        if not env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", False):
            return False
        if not first.stream:
            return False
        if self.device.type != "cuda" and not explicit:
            return False
        if not _is_tensor_parallel_primary_model(self.model):
            return False
        return hasattr(self.model, "allocate_cache")

    def _run_tensor_parallel_online_batcher(self, first: _QueuedGeneration) -> None:
        requested_max_batch = int(getattr(self, "max_batch_size", 1))
        enable_ragged_decode = env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_RAGGED_DECODE", True)
        store_reusable_prefixes = env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_CACHE_STORE", True)
        store_full_prompt_prefixes = env_flag(
            "TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_CACHE_STORE_FULL_PROMPTS",
            True,
        )
        max_active = self._online_serving_max_active()
        prefix_rows = self._online_serving_prefix_rows()
        prefill_budget = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", 2048, minimum=0)
        request_by_id: dict[str, _QueuedGeneration] = {}
        next_request_id = 0
        deferred: list[_QueuedGeneration] = []
        started = False
        step = 0
        online_step_commands = 0
        emitted_events = 0
        finished_events = 0
        initial_wait_s = (
            env_float("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", 5.0, minimum=0.0) / 1000.0
        )
        idle_wait_s = env_float("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", 2.0, minimum=0.0) / 1000.0
        profile_start_s = time.perf_counter()
        phase_ms: dict[str, float] = {}

        def add_phase(name: str, started_at_s: float) -> None:
            phase_ms[name] = phase_ms.get(name, 0.0) + (time.perf_counter() - started_at_s) * 1000.0

        def same_online_class(request: _QueuedGeneration) -> bool:
            return request.stream and request.temperature == first.temperature

        initial_batch = [first]
        initial_batch_start_s = time.perf_counter()

        def drain_initial_ready() -> int:
            admitted = 0
            while len(initial_batch) < max_active:
                try:
                    item = self._generation_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._generation_queue.put(None)
                    break
                if same_online_class(item):
                    initial_batch.append(item)
                    admitted += 1
                else:
                    deferred.append(item)
            return admitted

        drain_initial_ready()
        if initial_wait_s > 0.0 and len(initial_batch) < max_active:
            deadline = time.perf_counter() + initial_wait_s
            while len(initial_batch) < max_active:
                timeout = deadline - time.perf_counter()
                if timeout <= 0.0:
                    break
                try:
                    item = self._generation_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    self._generation_queue.put(None)
                    break
                if same_online_class(item):
                    initial_batch.append(item)
                else:
                    deferred.append(item)
        add_phase("initial_batch_ms", initial_batch_start_s)

        default_max_seq_len = self._tp_online_default_max_seq_len(initial_batch)
        max_seq_len = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN", default_max_seq_len, minimum=1)
        max_seq_len = max(max_seq_len, len(first.prompt) + first.max_tokens)
        sized_initial_batch = [
            request
            for request in initial_batch
            if len(request.prompt) + request.max_tokens <= max_seq_len
        ]
        sized_initial_ids = {id(request) for request in sized_initial_batch}
        for request in initial_batch:
            if id(request) not in sized_initial_ids:
                deferred.append(request)
        initial_batch = sized_initial_batch or [first]
        run_max_tokens = max(request.max_tokens for request in initial_batch)
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        eos_token_id = next(iter(stop_token_ids)) if stop_token_ids else None
        engine_create_start_s = time.perf_counter()
        runtime_engine = _RuntimeContinuousBatchEngine(
            self.model,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
            temperature=first.temperature,
            max_active_requests=max_active,
            prefix_cache_capacity=prefix_rows,
            prefill_token_budget=prefill_budget if prefill_budget > 0 else None,
            enable_ragged_decode=enable_ragged_decode,
            store_reusable_prefixes=store_reusable_prefixes,
            store_full_prompt_prefixes=store_full_prompt_prefixes,
            pin_shared_prefix=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_PIN_SHARED_PREFIX", True),
            graph_prefill=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_GRAPH_PREFILL", True),
            prefill_chunk_size=(env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK", 0, minimum=0) or None),
            profile_timings=bool(self._queue_profile_path_value()),
        )
        add_phase("engine_create_ms", engine_create_start_s)

        def compatible(request: _QueuedGeneration) -> bool:
            return same_online_class(request) and len(request.prompt) + request.max_tokens <= max_seq_len

        def submit_batch(requests: Sequence[_QueuedGeneration], *, arrival_step: int) -> None:
            nonlocal next_request_id
            if not requests:
                return
            submit_start_s = time.perf_counter()
            request_id_start = next_request_id
            prompts = [request.prompt for request in requests]
            row_max_tokens = [request.max_tokens for request in requests]
            max_tokens = max(row_max_tokens, default=0)
            _broadcast_tensor_parallel_online_submit_prompt_lists(
                self.model,
                prompts,
                max_tokens=max_tokens,
                row_max_tokens=row_max_tokens,
                arrival_step=arrival_step,
                eos_token_id=eos_token_id,
                request_id_start=request_id_start,
            )
            for request in requests:
                request_id = str(next_request_id)
                next_request_id += 1
                request_by_id[request_id] = request
                runtime_engine.submit_online(
                    _RuntimeServingRequest(
                        request_id,
                        tuple(request.prompt),
                        request.max_tokens,
                        arrival_step=arrival_step,
                        eos_token_id=eos_token_id,
                    )
                )
            _sync_tensor_parallel_command(self.model, self.device)
            add_phase("submit_sync_ms", submit_start_s)

        def drain_ready(arrival_step: int) -> int:
            drain_start_s = time.perf_counter()
            ready: list[_QueuedGeneration] = []
            while len(ready) < max_active:
                try:
                    item = self._generation_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._generation_queue.put(None)
                    break
                if compatible(item):
                    ready.append(item)
                else:
                    deferred.append(item)
            add_phase("drain_ready_poll_ms", drain_start_s)
            submit_batch(ready, arrival_step=arrival_step)
            return len(ready)

        def wait_and_drain(arrival_step: int, wait_s: float) -> int:
            wait_start_s = time.perf_counter()
            admitted = drain_ready(arrival_step)
            if wait_s <= 0.0 or admitted >= max_active:
                add_phase("idle_wait_drain_ms", wait_start_s)
                return admitted
            deadline = time.perf_counter() + wait_s
            while admitted < max_active:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.0001))
                admitted += drain_ready(arrival_step)
            add_phase("idle_wait_drain_ms", wait_start_s)
            return admitted

        try:
            with _tensor_parallel_symm_mem_allreduce_scope(
                self.model,
                self.device,
                max_tokens=run_max_tokens,
                temperature=first.temperature,
            ):
                start_sync_start_s = time.perf_counter()
                _broadcast_tensor_parallel_online_start(
                    self.model,
                    max_seq_len=max_seq_len,
                    max_active_requests=max_active,
                    prefix_cache_capacity=prefix_rows,
                    prefill_token_budget=prefill_budget if prefill_budget > 0 else None,
                    temperature=first.temperature,
                    enable_ragged_decode=enable_ragged_decode,
                    store_reusable_prefixes=store_reusable_prefixes,
                    store_full_prompt_prefixes=store_full_prompt_prefixes,
                    max_tokens=run_max_tokens,
                )
                shared_cache = getattr(self, "_persistent_serving_cache", None)
                if shared_cache is None:
                    try:
                        total_online_rows = max_active + prefix_rows
                        shared_cache = self._generation_cache(
                            total_online_rows,
                            max_seq_len,
                            model=self.model,
                            batch_capacity=_generation_cache_batch_capacity(self.model, total_online_rows),
                        )
                        _reset_generation_cache(shared_cache)
                    except Exception:
                        shared_cache = None
                runtime_engine.start_online(max_seq_len=max_seq_len, external_cache=shared_cache)
                started = True
                _sync_tensor_parallel_command(self.model, self.device)
                add_phase("start_sync_ms", start_sync_start_s)
                submit_batch(initial_batch, arrival_step=0)
                decode_quantum = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", 8, minimum=1)
                while True:
                    drain_ready(step)
                    if not runtime_engine.has_online_work():
                        if wait_and_drain(step, idle_wait_s) == 0:
                            break
                        continue
                    step_broadcast_start_s = time.perf_counter()
                    _broadcast_tensor_parallel_online_step(self.model, decode_quantum)
                    add_phase("step_broadcast_ms", step_broadcast_start_s)
                    online_step_commands += 1
                    for _ in range(decode_quantum):
                        if not runtime_engine.has_online_work():
                            break
                        runtime_step_start_s = time.perf_counter()
                        events = runtime_engine.step_online()
                        add_phase("runtime_step_ms", runtime_step_start_s)
                        event_emit_start_s = time.perf_counter()
                        for event in events:
                            emitted_events += 1
                            request = request_by_id[event.request_id]
                            if request.done:
                                continue
                            if event.token in stop_token_ids:
                                _finish_stream_request(request)
                                finished_events += 1
                                continue
                            request.responses.put(event.token)
                            if event.finished:
                                _finish_stream_request(request)
                                finished_events += 1
                        add_phase("event_emit_ms", event_emit_start_s)
                        step += 1
                    if env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC", True):
                        step_sync_start_s = time.perf_counter()
                        _sync_tensor_parallel_command(self.model, self.device)
                        add_phase("step_sync_ms", step_sync_start_s)
        except BaseException as exc:
            for request in request_by_id.values():
                if not request.done:
                    request.responses.put(exc)
                    request.done = True
            raise
        finally:
            add_phase("total_ms", profile_start_s)
            phase_fields = {f"phase_{name}": round(value, 3) for name, value in phase_ms.items()}
            self._record_runtime_engine_queue_profile(
                "online_batcher",
                runtime_engine,
                submitted_requests=len(request_by_id),
                deferred_requests=len(deferred),
                initial_batch_size=len(initial_batch),
                max_seq_len=max_seq_len,
                run_max_tokens=run_max_tokens,
                max_active=max_active,
                prefix_rows=prefix_rows,
                decode_quantum=env_int("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", 1, minimum=1),
                online_steps=step,
                online_step_commands=online_step_commands,
                emitted_events=emitted_events,
                finished_events=finished_events,
                **phase_fields,
            )
            if started:
                _broadcast_tensor_parallel_online_close(self.model)
                _sync_tensor_parallel_command(self.model, self.device)
            for request in deferred:
                self._generation_queue.put(request)

    def _tp_online_default_max_seq_len(self, requests: Sequence[_QueuedGeneration]) -> int:
        default_max_seq_len = max((len(request.prompt) + request.max_tokens for request in requests), default=1)
        default_max_seq_len += env_int(
            "TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS",
            0,
            minimum=0,
        )
        max_model_len = getattr(self, "max_model_len", None)
        if max_model_len is not None:
            default_max_seq_len = min(default_max_seq_len, int(max_model_len))
        return max(1, default_max_seq_len)

    def _enter_live_request(self) -> None:
        with self._live_request_condition:
            if self._live_requests == 0 and self._completed_queue_batches > 0:
                idle_s = time.perf_counter() - self._idle_since_s
                min_idle_s = env_float("TORCHINFERNO_OPENAI_IDLE_CLEANUP_MIN_IDLE_MS", 250.0, minimum=0.0) / 1000.0
                if idle_s >= min_idle_s:
                    self._cleanup_after_idle = True
            self._live_requests += 1
            self._live_request_condition.notify_all()

    def _exit_live_request(self) -> None:
        with self._live_request_condition:
            self._live_requests -= 1
            if self._live_requests <= 0:
                self._live_requests = 0
                self._idle_since_s = time.perf_counter()
            self._live_request_condition.notify_all()

    def _maybe_cleanup_runtime_after_idle(self) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_IDLE_CLEANUP", False):
            self._cleanup_after_idle = False
            return
        if not self._cleanup_after_idle:
            return
        self._cleanup_after_idle = False
        self._record_queue_profile(
            {
                "event": "idle_cleanup",
                "completed_queue_batches": self._completed_queue_batches,
            }
        )
        _broadcast_tensor_parallel_cleanup(self.model)
        self._clear_runtime_state_after_idle()
        _sync_tensor_parallel_command(self.model, self.device, cuda_sync=False)

    def _clear_runtime_state_after_idle(self) -> None:
        if env_flag("TORCHINFERNO_OPENAI_IDLE_CLEANUP_CACHE_POOLS", True):
            self._clear_cache_pool(self._cache_pool, model=self.model)
            self._clear_cache_pool(self._microbatch_cache_pool, model=self.model)
        self._single_prefill_capture_seen.clear()
        self._batched_prefill_capture_seen.clear()
        self._persistent_prompt_list_step_state = None
        self._persistent_prompt_list_step_last_result = None
        self._token_budget_step_state = None
        self._token_budget_step_last_result = None
        if env_flag("TORCHINFERNO_OPENAI_IDLE_CLEANUP_PREFIX_CACHE", False):
            self._clear_prefix_cache()
        if env_flag("TORCHINFERNO_OPENAI_IDLE_CLEANUP_GRAPH_CACHES", True):
            _clear_model_graph_caches(self.model)
        if self.device.type == "cuda" and env_flag("TORCHINFERNO_OPENAI_IDLE_CLEANUP_EMPTY_CACHE", False):
            torch.cuda.empty_cache()

    def pop_phase_records(self) -> list[dict[str, float]]:
        with self._phase_records_lock:
            records = list(self._phase_records)
            self._phase_records.clear()
        return records

    def _new_phase_record(self) -> dict[str, float] | None:
        if not self._phase_timing_enabled:
            return None
        return {"request_start": time.perf_counter()}

    def _mark_phase(self, phase: dict[str, float] | None, name: str) -> None:
        if phase is not None:
            phase[name] = time.perf_counter()

    def _record_phase(self, phase: dict[str, float] | None) -> None:
        if phase is None:
            return
        with self._phase_records_lock:
            self._phase_records.append(dict(phase))

    def _queue_profile_path_value(self) -> str:
        path = getattr(self, "_queue_profile_path", None)
        if path is None:
            path = os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", "")
            self._queue_profile_path = path
        return str(path)

    def _record_queue_profile(self, record: Mapping[str, object]) -> None:
        profile_path = self._queue_profile_path_value()
        if not profile_path:
            return
        try:
            line = json.dumps(record, sort_keys=True) + "\n"
            lock = getattr(self, "_queue_profile_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._queue_profile_lock = lock
            with lock:
                with open(profile_path, "a", encoding="utf-8") as profile_file:
                    profile_file.write(line)
        except Exception as exc:
            warn_optional_failure("openai.queue_profile", exc)

    def _next_queue_profile_sequence(self) -> int:
        lock = getattr(self, "_queue_profile_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._queue_profile_lock = lock
        with lock:
            value = int(getattr(self, "_queue_profile_next_sequence", 0))
            self._queue_profile_next_sequence = value + 1
            return value

    def _record_runtime_engine_queue_profile(
        self,
        event: str,
        runtime_engine: object,
        **fields: object,
    ) -> None:
        if not self._queue_profile_path_value():
            return
        stats = getattr(runtime_engine, "stats", None)
        record: dict[str, object] = {"event": event, **fields}
        for name in (
            "prefill_model_calls",
            "prefill_batches",
            "prefill_tokens",
            "decode_model_calls",
            "decode_batches",
            "decode_tokens",
            "ragged_decode_batches",
            "ragged_decode_tokens",
            "decode_graph_hits",
            "decode_graph_misses",
            "prefix_reuse_requests",
            "prefix_reuse_tokens",
            "queued_requests",
            "scheduler_steps",
            "max_model_batch_size",
            "persistent_cache_rows",
            "prefill_admitted_requests",
            "prefill_single_batches",
            "prefill_plain_batches",
            "prefill_prefix_reuse_batches",
            "prefill_common_prefix_batches",
            "prefill_padded_suffix_batches",
            "prefill_graph_hits",
            "prefill_graph_misses",
            "prefill_wall_ms",
            "prefill_copy_ms",
            "prefill_forward_ms",
            "prefill_setup_ms",
            "prefill_sample_ms",
            "prefill_state_ms",
            "decode_ragged_prepare_ms",
            "decode_ragged_model_ms",
            "decode_ragged_cpu_tokens_ms",
            "decode_ragged_state_update_ms",
        ):
            value = getattr(stats, name, None)
            if isinstance(value, (int, float)):
                record[f"runtime_{name}"] = value
        self._record_queue_profile(record)

    def _reset_stream_group_profile_extra(self) -> None:
        if self._queue_profile_path_value():
            self._stream_group_profile_extra = {}

    def _add_stream_group_profile_extra(self, **fields: object) -> None:
        if not self._queue_profile_path_value():
            return
        extra = getattr(self, "_stream_group_profile_extra", None)
        if not isinstance(extra, dict):
            extra = {}
            self._stream_group_profile_extra = extra
        extra.update(fields)

    def _stream_group_profile_start_s(self) -> float:
        return time.perf_counter() if self._queue_profile_path_value() else 0.0

    def _add_stream_group_profile_elapsed(self, field: str, start_s: float) -> None:
        if start_s <= 0.0 or not self._queue_profile_path_value():
            return
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        self._add_stream_group_profile_value(field, elapsed_ms)

    def _add_stream_group_profile_value(self, field: str, value: float) -> None:
        if not self._queue_profile_path_value():
            return
        extra = getattr(self, "_stream_group_profile_extra", None)
        if not isinstance(extra, dict):
            extra = {}
            self._stream_group_profile_extra = extra
        previous = extra.get(field)
        elapsed_ms = float(value)
        if isinstance(previous, (int, float)):
            elapsed_ms += float(previous)
        extra[field] = elapsed_ms

    def _try_acquire_single_request_model(self, *, temperature: float = 0.0) -> bool:
        if not self.single_request_fast_path:
            return False
        self._wait_for_single_request_admission(temperature=temperature)
        with self._live_request_condition:
            is_only_live_request = self._live_requests == 1
        if not is_only_live_request or not self._generation_queue.empty():
            return False
        return self._model_lock.acquire(blocking=False)

    def _wait_for_single_request_admission(self, *, temperature: float = 0.0) -> None:
        wait_s = min(
            self.batch_wait_s,
            max(self.single_request_admission_wait_s, self._temperature_single_request_admission_wait_s(temperature)),
        )
        if wait_s <= 0.0 or self.max_batch_size <= 1:
            return
        if self._model_lock.locked() or not self._generation_queue.empty():
            return
        deadline = time.perf_counter() + wait_s
        with self._live_request_condition:
            while self._live_requests == 1 and self._generation_queue.empty():
                timeout = deadline - time.perf_counter()
                if timeout <= 0.0:
                    break
                self._live_request_condition.wait(timeout=timeout)

    def _temperature_single_request_admission_wait_s(self, temperature: float) -> float:
        if temperature <= 0.0:
            return 0.0
        return env_float("TORCHINFERNO_OPENAI_TEMPERATURE_ADMISSION_WAIT_MS", 0.5, minimum=0.0) / 1000.0

    def _has_multiple_live_requests(self) -> bool:
        with self._live_request_condition:
            return self._live_requests > 1

    def _queued_batch_has_current_live_requests(self, batch_size: int, batch_limit: int) -> bool:
        with self._live_request_condition:
            live_requests = self._live_requests
        return live_requests > 0 and batch_size >= min(batch_limit, live_requests)

    def _queued_batch_limit(self, first: _QueuedGeneration) -> int:
        limit = self.max_batch_size
        if (
            first.stream
            and _is_tensor_parallel_model(self.model)
            and self.device.type == "cuda"
        ):
            short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
            long_prompt_short_max_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_LONG_PROMPT_SHORT_STREAM_MAX_TOKENS",
                128,
                minimum=1,
            )
            long_prompt_min_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_LONG_PROMPT_SHORT_STREAM_MIN_PROMPT_TOKENS",
                96,
                minimum=1,
            )
            if (
                first.temperature <= 0.0
                and first.max_tokens <= long_prompt_short_max_tokens
                and len(first.prompt) >= long_prompt_min_tokens
                and env_flag("TORCHINFERNO_OPENAI_TP_LONG_PROMPT_SHORT_STREAM_BATCH_CAP", False)
            ):
                long_prompt_limit = env_int(
                    "TORCHINFERNO_OPENAI_TP_LONG_PROMPT_SHORT_STREAM_MAX_BATCH_SIZE",
                    min(limit, 56),
                    minimum=1,
                )
                return min(limit, long_prompt_limit)
            if first.max_tokens <= short_max_tokens:
                if first.temperature > 0.0:
                    default_short_limit = env_int(
                        "TORCHINFERNO_OPENAI_TP_SAMPLED_SHORT_STREAM_MAX_BATCH_SIZE",
                        min(limit, 64),
                        minimum=1,
                    )
                else:
                    default_short_limit = 64
                short_limit = env_int(
                    "TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE",
                    min(limit, default_short_limit),
                    minimum=1,
                )
                return min(limit, short_limit)
            large_min_tokens = env_int("TORCHINFERNO_OPENAI_LARGE_STREAM_MIN_TOKENS", 512, minimum=1)
            if first.temperature <= 0.0 and first.max_tokens >= large_min_tokens:
                large_limit = env_int(
                    "TORCHINFERNO_OPENAI_TP_LARGE_STREAM_MAX_BATCH_SIZE",
                    min(limit, 32),
                    minimum=1,
                )
                return min(limit, large_limit)
        return limit

    def _drain_ready_requests(self, batch: list[_QueuedGeneration], *, limit: int | None = None) -> None:
        batch_limit = self.max_batch_size if limit is None else max(1, min(self.max_batch_size, limit))
        while len(batch) < batch_limit:
            try:
                item = self._generation_queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._generation_queue.put(None)
                return
            batch.append(item)

    def _collect_batch_until_deadline(
        self,
        batch: list[_QueuedGeneration],
        *,
        limit: int | None = None,
        wait_s: float | None = None,
        reset_deadline_on_item: bool = True,
        stop_when_live_batch_ready: bool = True,
    ) -> None:
        batch_wait_s = (
            wait_s
            if wait_s is not None
            else (self._queued_batch_wait_s(batch[0]) if batch else self.batch_wait_s)
        )
        if batch_wait_s == 0.0:
            return
        batch_limit = self.max_batch_size if limit is None else max(1, min(self.max_batch_size, limit))
        if stop_when_live_batch_ready and self._queued_batch_has_current_live_requests(len(batch), batch_limit):
            return
        deadline = time.perf_counter() + batch_wait_s
        while len(batch) < batch_limit:
            timeout = max(0.0, deadline - time.perf_counter())
            if timeout == 0.0:
                break
            try:
                item = self._generation_queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is None:
                self._generation_queue.put(None)
                break
            batch.append(item)
            if stop_when_live_batch_ready and self._queued_batch_has_current_live_requests(len(batch), batch_limit):
                break
            if reset_deadline_on_item:
                deadline = time.perf_counter() + batch_wait_s

    def _queued_batch_wait_s(self, first: _QueuedGeneration) -> float:
        if first.temperature <= 0.0:
            deterministic_wait_s = self._queued_deterministic_stream_batch_wait_s(first)
            if deterministic_wait_s is not None:
                return deterministic_wait_s
            return self.batch_wait_s
        max_tokens = env_int("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", 512, minimum=1)
        if first.max_tokens > max_tokens:
            return self.batch_wait_s
        short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
        default_wait_ms = 1.0 if first.max_tokens <= short_max_tokens else 2.0
        temperature_wait_s = (
            env_float("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MS", default_wait_ms, minimum=0.0)
            / 1000.0
        )
        return max(self.batch_wait_s, temperature_wait_s)

    def _queued_deterministic_stream_batch_wait_s(self, first: _QueuedGeneration) -> float | None:
        if not (
            first.stream
            and _is_tensor_parallel_model(getattr(self, "model", None))
            and getattr(self, "device", torch.device("cpu")).type == "cuda"
        ):
            return None
        min_tokens = env_int(
            "TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MIN_TOKENS",
            1,
            minimum=1,
        )
        max_tokens = env_int(
            "TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MAX_TOKENS",
            400,
            minimum=min_tokens,
        )
        if not (min_tokens <= first.max_tokens <= max_tokens):
            return None
        wait_ms = env_float("TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_BATCH_WAIT_MS", 0.0, minimum=0.0)
        return wait_ms / 1000.0

    def _queued_initial_batch_wait_s(self, first: _QueuedGeneration) -> float:
        if not first.stream or self.max_batch_size <= 1:
            return 0.0
        if not (
            _is_tensor_parallel_model(self.model)
            and self.device.type == "cuda"
        ):
            return 0.0
        if first.temperature <= 0.0:
            min_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MIN_TOKENS",
                1,
                minimum=1,
            )
            max_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MAX_TOKENS",
                400,
                minimum=min_tokens,
            )
            if not (min_tokens <= first.max_tokens <= max_tokens):
                return 0.0
            short_output_max_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_GREEDY_SHORT_OUTPUT_INITIAL_BATCH_MAX_TOKENS",
                128,
                minimum=1,
            )
            default_wait_ms = (
                env_float(
                    "TORCHINFERNO_OPENAI_TP_GREEDY_SHORT_OUTPUT_INITIAL_BATCH_WAIT_MS",
                    5.0,
                    minimum=0.0,
                )
                if first.max_tokens <= short_output_max_tokens
                else 5.0
            )
            wait_ms = env_float(
                "TORCHINFERNO_OPENAI_TP_GREEDY_INITIAL_BATCH_WAIT_MS",
                default_wait_ms,
                minimum=0.0,
            )
            return min(self.batch_wait_s, wait_ms / 1000.0)
        max_temperature_tokens = env_int("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", 512, minimum=1)
        if first.max_tokens > max_temperature_tokens:
            return 0.0
        short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
        if first.max_tokens <= short_max_tokens:
            wait_ms = env_float(
                "TORCHINFERNO_OPENAI_TP_SHORT_SAMPLED_INITIAL_BATCH_WAIT_MS",
                1.0,
                minimum=0.0,
            )
        else:
            wait_ms = env_float("TORCHINFERNO_OPENAI_TP_SAMPLED_INITIAL_BATCH_WAIT_MS", 1.0, minimum=0.0)
        return min(self._queued_batch_wait_s(first), wait_ms / 1000.0)

    def _run_queued_batch(self, batch: list[_QueuedGeneration]) -> None:
        groups: dict[tuple[float, bool], list[_QueuedGeneration]] = {}
        for request in batch:
            key = (request.temperature, request.stream)
            groups.setdefault(key, []).append(request)
        for group in groups.values():
            is_stream = group[0].stream
            try:
                if is_stream:
                    self._run_queued_stream_group(group)
                else:
                    self._run_queued_completion_group(group)
            except BaseException as exc:
                for request in group:
                    if not request.done:
                        request.responses.put(exc)
            finally:
                if is_stream:
                    for request in group:
                        _finish_stream_request(request)

    def _run_queued_stream_group(self, group: list[_QueuedGeneration]) -> None:
        if self._should_use_tensor_parallel_online_stream_group(group):
            self._run_queued_stream_group_tensor_parallel_online(group)
            return
        if self._should_use_runtime_continuous_stream_group(group):
            self._run_queued_stream_group_runtime_continuous(group)
            return
        if self._should_use_flashinfer_stream_group(group):
            self._run_flashinfer_stream_group(group)
            return
        prompts = [request.prompt for request in group]
        max_tokens = max((request.max_tokens for request in group), default=0)
        row_max_tokens = [request.max_tokens for request in group]
        use_prompt_list_batch = self._should_use_prompt_list_stream_group(prompts)
        if use_prompt_list_batch and _prefer_tensor_parallel_stream_group(prompts, self.model):
            if self._shared_prefix_prompt_list_tokens(prompts) <= 0:
                use_prompt_list_batch = False
        shared_prefix_tokens = self._shared_prefix_prompt_list_tokens(prompts) if use_prompt_list_batch else 0
        profile_queue = bool(self._queue_profile_path_value())
        with _tensor_parallel_symm_mem_allreduce_scope(
            self.model,
            self.device,
            max_tokens=max_tokens,
            temperature=group[0].temperature,
        ):
            if use_prompt_list_batch:
                completed_steps = 0
                group_start_s = time.perf_counter() if profile_queue else 0.0
                first_emit_s: float | None = None
                emitted_tokens = 0
                if profile_queue:
                    self._reset_stream_group_profile_extra()
                    self._add_stream_group_profile_extra(
                        **_model_graph_cache_profile_fields(self.model, "graph_before_")
                    )
                try:
                    early_restart = (lambda: True) if env_flag(
                        "TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", False
                    ) else None
                    step_iter = self._generate_prompt_list_batch_steps(
                        prompts,
                        max_tokens=max_tokens,
                        temperature=group[0].temperature,
                        row_max_tokens=row_max_tokens,
                        early_restart_check=early_restart,
                    )
                    for step, step_tokens in enumerate(step_iter):
                        completed_steps = step + 1
                        if profile_queue and first_emit_s is None:
                            first_emit_s = time.perf_counter()
                        emitted_tokens += sum(token is not None for token in step_tokens)
                        emit_start_s = self._stream_group_profile_start_s()
                        _emit_stream_step(group, step, step_tokens, getattr(self, "stop_token_ids", frozenset()))
                        self._add_stream_group_profile_elapsed("stream_emit_ms", emit_start_s)
                finally:
                    _sync_tensor_parallel_command(
                        self.model,
                        self.device,
                        cuda_sync=_tp_command_cuda_sync_for_steps(
                            completed_steps,
                            emitted_tokens=emitted_tokens,
                        ),
                    )
                    if profile_queue:
                        self._record_stream_group_queue_profile(
                            group,
                            group_start_s=group_start_s,
                            first_emit_s=first_emit_s,
                            completed_steps=completed_steps,
                            emitted_tokens=emitted_tokens,
                            use_prompt_list_batch=True,
                            group_kind="prompt_list",
                        )
                return
            for same_length_group in _queued_groups_by_prompt_length(group):
                same_length_max_tokens = max(request.max_tokens for request in same_length_group)
                group_start_s = time.perf_counter() if profile_queue else 0.0
                first_emit_s = None
                emitted_tokens = 0
                if profile_queue:
                    self._reset_stream_group_profile_extra()
                    self._add_stream_group_profile_extra(
                        **_model_graph_cache_profile_fields(self.model, "graph_before_")
                    )
                input_ids = torch.tensor(
                    [request.prompt for request in same_length_group],
                    dtype=torch.long,
                    device=self.device,
                )
                completed_steps = 0
                try:
                    step_iter = self._generate_batch_steps(
                        input_ids,
                        max_tokens=same_length_max_tokens,
                        temperature=same_length_group[0].temperature,
                        row_max_tokens=[request.max_tokens for request in same_length_group],
                    )
                    for step, step_tokens in enumerate(step_iter):
                        completed_steps = step + 1
                        if profile_queue and first_emit_s is None:
                            first_emit_s = time.perf_counter()
                        emitted_tokens += sum(token is not None for token in step_tokens)
                        emit_start_s = self._stream_group_profile_start_s()
                        _emit_stream_step(
                            same_length_group,
                            step,
                            step_tokens,
                            getattr(self, "stop_token_ids", frozenset()),
                        )
                        self._add_stream_group_profile_elapsed("stream_emit_ms", emit_start_s)
                finally:
                    _sync_tensor_parallel_command(
                        self.model,
                        self.device,
                        cuda_sync=_tp_command_cuda_sync_for_steps(
                            completed_steps,
                            emitted_tokens=emitted_tokens,
                        ),
                    )
                    if profile_queue:
                        self._record_stream_group_queue_profile(
                            same_length_group,
                            group_start_s=group_start_s,
                            first_emit_s=first_emit_s,
                            completed_steps=completed_steps,
                            emitted_tokens=emitted_tokens,
                            use_prompt_list_batch=False,
                            group_kind="same_length_tensor",
                        )

    def _should_use_tensor_parallel_online_stream_group(self, group: Sequence[_QueuedGeneration]) -> bool:
        if not env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS", False):
            return False
        if not group or any(not request.stream for request in group):
            return False
        if not _is_tensor_parallel_primary_model(self.model):
            return False
        if len(getattr(self, "stop_token_ids", frozenset())) > 1:
            return False
        return hasattr(self.model, "allocate_cache")

    def _run_queued_stream_group_tensor_parallel_online(self, group: Sequence[_QueuedGeneration]) -> None:
        max_tokens = max((request.max_tokens for request in group), default=0)
        if max_tokens <= 0:
            return
        enable_ragged_decode = env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_RAGGED_DECODE", True)
        store_reusable_prefixes = env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_CACHE_STORE", True)
        store_full_prompt_prefixes = env_flag(
            "TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_CACHE_STORE_FULL_PROMPTS",
            True,
        )
        default_max_active = min(max(1, len(group)), int(getattr(self, "max_batch_size", max(1, len(group)))))
        if not enable_ragged_decode:
            default_max_active = min(
                default_max_active,
                env_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", 64, minimum=1),
            )
        max_active = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_ACTIVE", default_max_active, minimum=1)
        prefix_rows = env_int(
            "TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS",
            min(max_active, max(0, len(group))),
            minimum=0,
        )
        prefill_budget = (
            env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", 0, minimum=0)
            if "TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET" in os.environ
            else 0
        )
        max_seq_len = env_int(
            "TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN",
            max(len(request.prompt) + request.max_tokens for request in group),
            minimum=1,
        )
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        eos_token_id = next(iter(stop_token_ids)) if stop_token_ids else None
        runtime_engine = _RuntimeContinuousBatchEngine(
            self.model,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
            temperature=group[0].temperature,
            max_active_requests=max_active,
            prefix_cache_capacity=prefix_rows,
            prefill_token_budget=prefill_budget if prefill_budget > 0 else None,
            enable_ragged_decode=enable_ragged_decode,
            store_reusable_prefixes=store_reusable_prefixes,
            store_full_prompt_prefixes=store_full_prompt_prefixes,
            graph_prefill=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_GRAPH_PREFILL", True),
            prefill_chunk_size=(env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK", 0, minimum=0) or None),
            profile_timings=bool(self._queue_profile_path_value()),
        )
        request_by_id = {str(index): request for index, request in enumerate(group)}
        row_max_tokens = [request.max_tokens for request in group]
        prompts = [request.prompt for request in group]
        started = False
        online_steps = 0
        online_step_commands = 0
        emitted_events = 0
        finished_events = 0
        try:
            with _tensor_parallel_symm_mem_allreduce_scope(
                self.model,
                self.device,
                max_tokens=max_tokens,
                temperature=group[0].temperature,
            ):
                _broadcast_tensor_parallel_online_start(
                    self.model,
                    max_seq_len=max_seq_len,
                    max_active_requests=max_active,
                    prefix_cache_capacity=prefix_rows,
                    prefill_token_budget=prefill_budget if prefill_budget > 0 else None,
                    temperature=group[0].temperature,
                    enable_ragged_decode=enable_ragged_decode,
                    store_reusable_prefixes=store_reusable_prefixes,
                    store_full_prompt_prefixes=store_full_prompt_prefixes,
                    max_tokens=max_tokens,
                )
                runtime_engine.start_online(max_seq_len=max_seq_len)
                started = True
                _sync_tensor_parallel_command(self.model, self.device)
                _broadcast_tensor_parallel_online_submit_prompt_lists(
                    self.model,
                    prompts,
                    max_tokens=max_tokens,
                    row_max_tokens=row_max_tokens,
                    arrival_step=0,
                    eos_token_id=eos_token_id,
                    request_id_start=0,
                )
                for index, request in enumerate(group):
                    runtime_engine.submit_online(
                        _RuntimeServingRequest(
                            str(index),
                            tuple(request.prompt),
                            request.max_tokens,
                            arrival_step=0,
                            eos_token_id=eos_token_id,
                        )
                )
                _sync_tensor_parallel_command(self.model, self.device)
                decode_quantum = env_int("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", 8, minimum=1)
                while runtime_engine.has_online_work():
                    _broadcast_tensor_parallel_online_step(self.model, decode_quantum)
                    online_step_commands += 1
                    for _ in range(decode_quantum):
                        if not runtime_engine.has_online_work():
                            break
                        events = runtime_engine.step_online()
                        for event in events:
                            emitted_events += 1
                            request = request_by_id[event.request_id]
                            if request.done:
                                continue
                            if event.token in stop_token_ids:
                                _finish_stream_request(request)
                                finished_events += 1
                                continue
                            request.responses.put(event.token)
                            if event.finished:
                                _finish_stream_request(request)
                                finished_events += 1
                        online_steps += 1
                    if env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC", True):
                        _sync_tensor_parallel_command(self.model, self.device)
        finally:
            self._record_runtime_engine_queue_profile(
                "online_stream_group",
                runtime_engine,
                submitted_requests=len(group),
                max_active=max_active,
                prefix_rows=prefix_rows,
                decode_quantum=env_int("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", 1, minimum=1),
                online_steps=online_steps,
                online_step_commands=online_step_commands,
                emitted_events=emitted_events,
                finished_events=finished_events,
            )
            if started:
                _broadcast_tensor_parallel_online_close(self.model)
                _sync_tensor_parallel_command(self.model, self.device)

    def _should_use_flashinfer_stream_group(self, group: Sequence[_QueuedGeneration]) -> bool:
        if not env_flag("TORCHINFERNO_OPENAI_FLASHINFER", False):
            return False
        if not group or any(not request.stream for request in group):
            return False
        if not _is_tensor_parallel_primary_model(self.model):
            return False
        if self.device.type != "cuda":
            return False
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            return False
        return hasattr(self.model, "forward_step_flashinfer") and hasattr(self.model, "allocate_cache")

    def _run_flashinfer_stream_group(self, group: Sequence[_QueuedGeneration]) -> None:
        # Step-at-a-time scheduler using FlashInfer for fused prefill+decode.
        # Each step: prefill new + decode active in ONE model forward.
        # TP coordination: broadcast step tensors to workers via dist.broadcast_object_list
        # before each forward. Workers receive and run the same forward_step_flashinfer.
        import torch
        import torch.distributed as dist
        model = self.model
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        max_batch = min(
            int(getattr(self, "max_batch_size", 64)),
            env_int("TORCHINFERNO_OPENAI_FLASHINFER_MAX_BATCH", 64, minimum=1),
        )
        max_seq_len = env_int("TORCHINFERNO_OPENAI_FLASHINFER_MAX_SEQ_LEN", 768, minimum=64)

        # Broadcast "flashinfer_start" to workers so they allocate the same cache
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list([{
                "op": "flashinfer_start",
                "max_batch": max_batch,
                "max_seq_len": max_seq_len,
                "temperature": float(group[0].temperature),
                "max_tokens": max((r.max_tokens for r in group), default=1),
            }], src=0)

        # Allocate FlashInfer cache
        cache = model.allocate_cache(max_batch, max_seq_len, cache_backend="flashinfer")

        # Active rows: row_index → (request, seq_len, generated, last_token)
        active: dict[int, tuple[_QueuedGeneration, int, int, int]] = {}
        free_rows = list(range(max_batch))

        # Admit initial group
        prompts_to_prefill: list[tuple[int, _QueuedGeneration]] = []
        for request in group:
            if not free_rows:
                break
            row = free_rows.pop(0)
            prompts_to_prefill.append((row, request))

        with _tensor_parallel_symm_mem_allreduce_scope(
            model, self.device,
            max_tokens=max((r.max_tokens for r in group), default=1),
            temperature=group[0].temperature,
        ):
            # Main step loop
            while prompts_to_prefill or active:
                # Build step tensors
                step_rows = []
                step_q_lens = []
                step_input_ids_list = []
                step_write_positions_list = []
                step_logit_positions = []
                max_q_len = 1

                # Decode active rows (1 query token each)
                for row, (request, seq_len, generated, last_token) in active.items():
                    step_rows.append(row)
                    step_q_lens.append(1)
                    step_input_ids_list.append([last_token])
                    step_write_positions_list.append([seq_len])
                    step_logit_positions.append(0)

                # Prefill new rows (variable query tokens)
                for row, request in prompts_to_prefill:
                    prompt = request.prompt
                    suffix_len = len(prompt)
                    step_rows.append(row)
                    step_q_lens.append(suffix_len)
                    step_input_ids_list.append(list(prompt))
                    step_write_positions_list.append(list(range(suffix_len)))
                    step_logit_positions.append(suffix_len - 1)
                    max_q_len = max(max_q_len, suffix_len)
                    active[row] = (request, 0, 0, 0)  # will be updated after forward

                batch = len(step_rows)
                if batch == 0:
                    break

                # Pad to max_q_len
                for i in range(len(step_input_ids_list)):
                    while len(step_input_ids_list[i]) < max_q_len:
                        step_input_ids_list[i].append(0)
                    while len(step_write_positions_list[i]) < max_q_len:
                        step_write_positions_list[i].append(0)

                # Build tensors
                input_ids = torch.tensor(step_input_ids_list, device=self.device, dtype=torch.long)
                q_lens = torch.tensor(step_q_lens, device=self.device, dtype=torch.long)
                write_positions = torch.tensor(step_write_positions_list, device=self.device, dtype=torch.long)
                logit_positions = torch.tensor(step_logit_positions, device=self.device, dtype=torch.long)

                # Set seq_lens on the cache (FlashInfer needs them)
                seq_lens_list = [0] * max_batch
                for i, row in enumerate(step_rows):
                    req, sl, gen, lt = active[row]
                    seq_lens_list[row] = sl
                seq_lens = torch.tensor(seq_lens_list[:batch], device=self.device, dtype=torch.long)

                # Reorder cache rows to match step order
                # Actually, FlashInfer's paged API handles this via paged_kv_indices.
                # But our forward_step_flashinfer uses rows 0..batch-1 contiguously.
                # We need to remap step_rows to physical cache rows.
                # For simplicity, use row indices directly (the cache has max_batch rows).

                # Broadcast step tensors to workers
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast_object_list([{
                        "op": "flashinfer_step",
                        "batch": batch,
                        "max_q_len": max_q_len,
                    }], src=0)
                    dist.broadcast(input_ids, src=0)
                    dist.broadcast(q_lens, src=0)
                    dist.broadcast(write_positions, src=0)
                    dist.broadcast(seq_lens, src=0)
                    dist.broadcast(logit_positions, src=0)

                # Forward (all ranks run this in lockstep via NCCL)
                logits = model.forward_step_flashinfer(
                    input_ids, cache,
                    seq_lens=seq_lens,
                    q_lens=q_lens,
                    write_positions=write_positions,
                    logit_positions=logit_positions,
                )

                # Sample
                next_tokens = model._sample_next_token(logits[:, -1, :], group[0].temperature)
                next_tokens_cpu = next_tokens.detach().cpu().tolist()

                # Process results
                prompts_to_prefill = []
                finished_rows = []
                for i, row in enumerate(step_rows):
                    request, seq_len, generated, last_token = active[row]
                    next_token = int(next_tokens_cpu[i])
                    new_seq_len = seq_len + step_q_lens[i]
                    new_generated = generated + 1

                    # Emit token (skip for prefill first token if it's a stop token)
                    if next_token in stop_token_ids:
                        _finish_stream_request(request)
                        finished_rows.append(row)
                    elif new_generated > request.max_tokens:
                        _finish_stream_request(request)
                        finished_rows.append(row)
                    else:
                        if not request.done:
                            request.responses.put(next_token)
                        if new_generated >= request.max_tokens:
                            _finish_stream_request(request)
                            finished_rows.append(row)
                        else:
                            active[row] = (request, new_seq_len, new_generated, next_token)

                # Free finished rows and check queue for new requests
                for row in finished_rows:
                    if row in active:
                        del active[row]
                    free_rows.append(row)
                    # Set cache seq_len to 0 for freed row
                    for layer in cache.layers:
                        if 0 <= row < len(layer._seq_lens):
                            layer._seq_lens[row] = 0

                # Admit new requests from queue
                while free_rows and not self._generation_queue.empty():
                    try:
                        new_request = self._generation_queue.get_nowait()
                    except Exception:
                        break
                    if new_request is None:
                        self._generation_queue.put(None)
                        break
                    row = free_rows.pop(0)
                    prompts_to_prefill.append((row, new_request))

        # Tell workers the FlashInfer session is done
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list([{"op": "flashinfer_close"}], src=0)

        # Finish any remaining active rows
        for row, (request, _, _, _) in active.items():
            if not request.done:
                _finish_stream_request(request)

    def _should_use_runtime_continuous_stream_group(self, group: Sequence[_QueuedGeneration]) -> bool:
        if not env_flag("TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_STREAM", False):
            return False
        if not group or any(not request.stream for request in group):
            return False
        if _is_tensor_parallel_model(self.model) and _tensor_parallel_world_size(self.model) > 1:
            return False
        if len(getattr(self, "stop_token_ids", frozenset())) > 1:
            return False
        return hasattr(self.model, "allocate_cache")

    def _run_queued_stream_group_runtime_continuous(self, group: Sequence[_QueuedGeneration]) -> None:
        max_active = env_int(
            "TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_MAX_ACTIVE",
            min(max(1, len(group)), int(getattr(self, "max_batch_size", max(1, len(group))))),
            minimum=1,
        )
        prefix_rows = env_int(
            "TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_PREFIX_ROWS",
            min(max_active, max(0, len(group))),
            minimum=0,
        )
        prefill_budget = (
            env_int("TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_PREFILL_TOKEN_BUDGET", 0, minimum=0)
            if "TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_PREFILL_TOKEN_BUDGET" in os.environ
            else 0
        )
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        eos_token_id = next(iter(stop_token_ids)) if stop_token_ids else None
        enable_ragged_decode = env_flag("TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_RAGGED_DECODE", True)
        store_full_prompt_prefixes = env_flag(
            "TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_PREFIX_CACHE_STORE_FULL_PROMPTS",
            True,
        )
        runtime_engine = _RuntimeContinuousBatchEngine(
            self.model,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
            temperature=group[0].temperature,
            max_active_requests=max_active,
            prefix_cache_capacity=prefix_rows,
            prefill_token_budget=prefill_budget if prefill_budget > 0 else None,
            enable_ragged_decode=enable_ragged_decode,
            store_full_prompt_prefixes=store_full_prompt_prefixes,
            graph_prefill=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_GRAPH_PREFILL", True),
            prefill_chunk_size=(env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK", 0, minimum=0) or None),
            profile_timings=bool(self._queue_profile_path_value()),
        )
        request_by_id = {str(index): request for index, request in enumerate(group)}
        runtime_requests = [
            _RuntimeServingRequest(
                str(index),
                tuple(request.prompt),
                request.max_tokens,
                arrival_step=0,
                eos_token_id=eos_token_id,
            )
            for index, request in enumerate(group)
        ]
        for event in runtime_engine.iter_events(runtime_requests):
            request = request_by_id[event.request_id]
            if request.done:
                continue
            if event.token in stop_token_ids:
                _finish_stream_request(request)
                continue
            request.responses.put(event.token)
            if event.finished:
                _finish_stream_request(request)

    def _record_stream_group_queue_profile(
        self,
        group: Sequence[_QueuedGeneration],
        *,
        group_start_s: float,
        first_emit_s: float | None,
        completed_steps: int,
        emitted_tokens: int,
        use_prompt_list_batch: bool,
        group_kind: str,
    ) -> None:
        if not self._queue_profile_path_value():
            return
        group_end_s = time.perf_counter()
        queued_times = [request.queued_at_s for request in group if request.queued_at_s > 0.0]
        first_queued_s = min(queued_times, default=group_start_s)
        prompt_lengths = [len(request.prompt) for request in group]
        row_max_tokens = [request.max_tokens for request in group]
        queue_sequences = [request.queue_sequence for request in group if request.queue_sequence >= 0]
        record: dict[str, object] = {
            "event": "stream_group",
            "group_kind": group_kind,
            "batch_size": len(group),
            "use_prompt_list_batch": use_prompt_list_batch,
            "temperature": float(group[0].temperature) if group else 0.0,
            "max_tokens": max(row_max_tokens, default=0),
            "min_row_max_tokens": min(row_max_tokens, default=0),
            "max_row_max_tokens": max(row_max_tokens, default=0),
            "min_prompt_tokens": min(prompt_lengths, default=0),
            "max_prompt_tokens": max(prompt_lengths, default=0),
            "shared_prefix_tokens": self._shared_prefix_prompt_list_tokens(
                [request.prompt for request in group]
            ),
            "queue_wait_ms": (group_start_s - first_queued_s) * 1000.0,
            "run_to_first_emit_ms": None
            if first_emit_s is None
            else (first_emit_s - group_start_s) * 1000.0,
            "queued_to_first_emit_ms": None
            if first_emit_s is None
            else (first_emit_s - first_queued_s) * 1000.0,
            "group_elapsed_ms": (group_end_s - group_start_s) * 1000.0,
            "completed_steps": completed_steps,
            "emitted_tokens": emitted_tokens,
        }
        if queue_sequences:
            record["queue_sequence_min"] = min(queue_sequences)
            record["queue_sequence_max"] = max(queue_sequences)
            record["queue_sequence_count"] = len(queue_sequences)
        extra = getattr(self, "_stream_group_profile_extra", None)
        if isinstance(extra, dict):
            record.update(extra)
        record.update(_model_graph_cache_profile_fields(self.model, "graph_after_"))
        self._stream_group_profile_extra = {}
        self._record_queue_profile(record)

    def _should_use_prompt_list_stream_group(self, prompts: Sequence[Sequence[int]]) -> bool:
        if self._shared_prefix_prompt_list_tokens(prompts) > 0:
            return True
        return (
            len(prompts) == 1
            and _is_tensor_parallel_model(self.model)
            and _prefix_cache_enabled_for_model(self.model)
            and env_flag("TORCHINFERNO_OPENAI_TP_SINGLE_PROMPT_LIST_STREAM", True)
        )

    def _run_queued_completion_group(self, group: list[_QueuedGeneration]) -> None:
        for same_length_group in _queued_groups_by_prompt_length(group):
            max_tokens = max(request.max_tokens for request in same_length_group)
            input_ids = torch.tensor(
                [request.prompt for request in same_length_group],
                dtype=torch.long,
                device=self.device,
            )
            with _tensor_parallel_symm_mem_allreduce_scope(
                self.model,
                self.device,
                max_tokens=max_tokens,
                temperature=same_length_group[0].temperature,
            ):
                rows = self._generate_batch_tokens(
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=same_length_group[0].temperature,
                )
                _sync_tensor_parallel_command(self.model, self.device)
            for request, tokens in zip(same_length_group, rows):
                request.responses.put(_GenerationResult(tokens[: request.max_tokens]))

    @torch.inference_mode()
    def _generate_batch_tokens(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
    ) -> list[list[int]]:
        if max_tokens <= 0:
            return [[] for _ in range(input_ids.size(0))]
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_generate(
                self.model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            return _generated_rows_with_model(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
            )

        cache = self._generation_cache(
            input_ids.size(0),
            input_ids.size(1) + max_tokens,
            model=model,
        )
        next_token, cache = _prefill_next_token(
            model,
            input_ids,
            cache,
            temperature,
            allow_capture=self._batched_prefill_graph_capture_enabled(
                model,
                input_ids,
                cache,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        next_token = next_token.to(self.device)
        generated_tokens: list[Tensor] = []
        active = (
            torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device)
            if stop_token_ids
            else None
        )
        for step in range(max_tokens):
            generated_tokens.append(next_token[:, None])
            should_continue = step + 1 < max_tokens
            if active is not None:
                active &= _tokens_not_in_stop(next_token, stop_token_ids)
                # Keep non-stream decode mostly async; exact output is trimmed after the final transfer.
                should_check_stop = (step + 1) % 8 == 0 or step + 1 == max_tokens
                if should_check_stop and not bool(active.any()):
                    should_continue = False
            should_continue = _sync_tensor_parallel_continue(model, should_continue, next_token.device)
            if not should_continue:
                break
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)
        rows = torch.cat(generated_tokens, dim=1).detach().cpu().tolist()
        return _trim_rows_at_stop(rows, stop_token_ids)

    def _generation_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        model: object,
        pool: bool = True,
        batch_capacity: int | None = None,
    ) -> object:
        cache_batch_size = max(batch_size, int(batch_capacity)) if batch_capacity is not None else batch_size
        cache_capacity = _generation_cache_capacity(model, max_seq_len)
        if not pool:
            cache = _allocate_cache(
                model,
                cache_batch_size,
                cache_capacity,
                device=self.device,
                cache_backend=self.cache_backend,
                page_size=self.page_size,
            )
            try:
                setattr(cache, "_torchinferno_ephemeral_cache", True)
            except Exception:
                pass
            _reset_generation_cache(cache)
            return cache
        exact_capacity = _prefers_exact_generation_cache(model)
        key = (cache_batch_size, cache_capacity, self.cache_backend, self.page_size, str(self.device))
        for cached_key, cached in list(self._cache_pool.items()):
            cached_batch, cached_max_seq_len, cached_backend, cached_page_size, cached_device = cached_key
            capacity_matches = cached_max_seq_len == cache_capacity if exact_capacity else cached_max_seq_len >= max_seq_len
            if (
                cached_batch == cache_batch_size
                and capacity_matches
                and cached_backend == self.cache_backend
                and cached_page_size == self.page_size
                and cached_device == str(self.device)
            ):
                # Preserve the prefix mark before reset so we can check it after.
                # The dense cache reset only zeroes seq_len counters, not KV data,
                # so the prefix KV physically survives in the cache. After reset,
                # restore the repeated-prefix marker if the prefix tokens match,
                # enabling the fast-path skip in _copy_generation_cache_first_row
                # (~12ms/batch savings for shared-prefix workloads).
                prefix_before_reset = _generation_cache_prefix_tokens(cached)
                repeated_before_reset = getattr(cached, "_torchinferno_repeated_prefix", None)
                if not _reset_generation_cache(cached):
                    continue
                if (
                    prefix_before_reset
                    and isinstance(repeated_before_reset, tuple)
                    and len(repeated_before_reset) == 3
                    and repeated_before_reset[0] == prefix_before_reset
                    and int(repeated_before_reset[2]) >= cache_batch_size
                ):
                    _mark_generation_cache_repeated_prefix(
                        cached,
                        prefix_before_reset,
                        int(repeated_before_reset[1]),
                        int(repeated_before_reset[2]),
                    )
                _set_ragged_decode_graph_disabled(cached, False)
                self._cache_pool.pop(cached_key, None)
                self._cache_pool[cached_key] = cached
                return cached
        max_entries = _cache_pool_max_entries()
        self._prepare_cache_pool_insert(self._cache_pool, key, max_entries, model=model)
        cache = _allocate_cache(
            model,
            cache_batch_size,
            cache_capacity,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
        )
        if _reset_generation_cache(cache):
            self._store_cache_pool_entry(self._cache_pool, key, cache, max_entries, model=model)
        return cache

    def _generation_microbatch_cache(
        self,
        slot: int,
        batch_size: int,
        max_seq_len: int,
        *,
        model: object,
    ) -> object:
        cache_capacity = _generation_cache_capacity(model, max_seq_len)
        exact_capacity = _prefers_exact_generation_cache(model)
        key = (slot, batch_size, cache_capacity, self.cache_backend, self.page_size, str(self.device))
        for cached_key, cached in list(self._microbatch_cache_pool.items()):
            (
                cached_slot,
                cached_batch,
                cached_max_seq_len,
                cached_backend,
                cached_page_size,
                cached_device,
            ) = cached_key
            capacity_matches = cached_max_seq_len == cache_capacity if exact_capacity else cached_max_seq_len >= max_seq_len
            if (
                cached_slot == slot
                and cached_batch == batch_size
                and capacity_matches
                and cached_backend == self.cache_backend
                and cached_page_size == self.page_size
                and cached_device == str(self.device)
                and _reset_generation_cache(cached)
            ):
                _set_ragged_decode_graph_disabled(cached, False)
                self._microbatch_cache_pool.pop(cached_key, None)
                self._microbatch_cache_pool[cached_key] = cached
                return cached
        max_entries = _microbatch_cache_pool_max_entries()
        self._evict_microbatch_slot(slot, keep_key=key, model=model)
        self._prepare_cache_pool_insert(self._microbatch_cache_pool, key, max_entries, model=model)
        cache = _allocate_cache(
            model,
            batch_size,
            cache_capacity,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
        )
        if _reset_generation_cache(cache):
            self._store_cache_pool_entry(
                self._microbatch_cache_pool,
                key,
                cache,
                max_entries,
                model=model,
            )
        return cache

    def _evict_microbatch_slot(
        self,
        slot: int,
        *,
        keep_key: tuple[int, int, int, str, int, str],
        model: object,
    ) -> None:
        for cached_key in list(self._microbatch_cache_pool):
            if cached_key != keep_key and cached_key[0] == slot:
                self._evict_cache_pool_key(self._microbatch_cache_pool, cached_key, model=model)

    def _prepare_cache_pool_insert(
        self,
        pool: dict[object, object],
        key: object,
        max_entries: int,
        *,
        model: object,
    ) -> None:
        self._evict_cache_pool_key(pool, key, model=model)
        if max_entries <= 0:
            self._clear_cache_pool(pool, model=model)
            return
        while len(pool) >= max_entries:
            self._evict_cache_pool_key(pool, _cache_pool_eviction_key(pool, model=model), model=model)

    def _store_cache_pool_entry(
        self,
        pool: dict[object, object],
        key: object,
        cache: object,
        max_entries: int,
        *,
        model: object,
    ) -> None:
        if max_entries <= 0:
            return
        existing = pool.pop(key, None)
        if existing is not None and existing is not cache:
            _release_decode_graphs_for_cache(model, existing)
        pool[key] = cache
        while len(pool) > max_entries:
            self._evict_cache_pool_key(pool, _cache_pool_eviction_key(pool, model=model), model=model)

    def _evict_cache_pool_key(self, pool: dict[object, object], key: object, *, model: object) -> None:
        cache = pool.pop(key, None)
        if cache is not None:
            _sync_before_decode_graph_release(
                model,
                cache,
                device=self.device,
                label="openai.cache_pool.evict_graph_cache_sync",
            )
            _release_decode_graphs_for_cache(model, cache)

    def _clear_cache_pool(self, pool: dict[object, object], *, model: object) -> None:
        for key in list(pool):
            self._evict_cache_pool_key(pool, key, model=model)

    def _restore_prefix_cache(self, input_ids: Tensor, cache: object) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return 0
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return 0
        if input_ids.size(0) != 1:
            return 0
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        min_prefix_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1)
        min_suffix_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", 8, minimum=1)
        short_suffix_max_prefix_tokens = env_int(
            "TORCHINFERNO_OPENAI_PREFIX_CACHE_SHORT_SUFFIX_MAX_PREFIX_TOKENS",
            256,
            minimum=1,
        )
        for _, entry in self._prefix_cache_restore_candidates(input_tokens, min_prefix_tokens):
            prefix_tokens = _matching_prefix_tokens(entry.tokens, input_tokens, min_prefix_tokens)
            suffix_tokens = len(input_tokens) - prefix_tokens
            if suffix_tokens < min_suffix_tokens and prefix_tokens < short_suffix_max_prefix_tokens:
                continue
            restored = restore_tensor_prefix_cache(
                entry,
                input_tokens,
                cache,
                min_prefix_tokens=min_prefix_tokens,
                device=str(self.device),
                backend=self.cache_backend,
                page_size=self.page_size,
                on_seq_len_restore_error=lambda exc: warn_optional_failure("openai.prefix_cache.seq_len_restore", exc),
            )
            if restored > 0:
                self._mark_prefix_cache_entry_used(entry)
                return restored
        return 0

    def _save_prefix_cache(self, input_ids: Tensor, generated_tokens: list[int], cache: object) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return
        if input_ids.size(0) != 1 or not generated_tokens:
            return
        cache_layers = tuple(getattr(cache, "layers", ()) or ())
        if not cache_layers:
            return
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        materialize_generated = env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE_MATERIALIZE_GENERATED")
        tokens = (
            (*input_tokens, *(int(token_id) for token_id in generated_tokens))
            if materialize_generated
            else input_tokens
        )
        max_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            self._prefix_cache_entry = None
            return
        if materialize_generated:
            self._materialize_generated_cache_tokens(input_ids, generated_tokens, cache)
        seq_len = min(len(tokens), _cache_row_seq_len(cache, 0))
        if seq_len < len(tokens):
            self._prefix_cache_entry = None
            return
        self._store_prefix_cache_entry(snapshot_tensor_prefix_cache(
            cache,
            tokens,
            seq_len=seq_len,
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
        ))

    def _restore_exact_prefix_cache(self, input_ids: Tensor, cache: object) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return 0
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return 0
        if input_ids.size(0) != 1:
            return 0
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        entry = self._exact_prefix_cache_entry(input_tokens)
        if entry is None:
            return 0
        if entry.device != str(self.device) or entry.backend != self.cache_backend or entry.page_size != self.page_size:
            return 0
        cache_layers = tuple(getattr(cache, "layers", ()) or ())
        if not cache_layers or len(cache_layers) != len(entry.layers):
            return 0
        seq_len = len(input_tokens)
        for layer, (keys, values) in zip(cache_layers, entry.layers):
            layer_keys = getattr(layer, "keys", None)
            layer_values = getattr(layer, "values", None)
            if not isinstance(layer_keys, Tensor) or not isinstance(layer_values, Tensor):
                return 0
            if layer_keys.size(0) < 1 or layer_keys.size(2) < seq_len:
                return 0
            layer_keys[:1, :, :seq_len, :].copy_(keys[:, :, :seq_len, :])
            layer_values[:1, :, :seq_len, :].copy_(values[:, :, :seq_len, :])
        _set_generation_cache_seq_len(cache, seq_len)
        self._mark_prefix_cache_entry_used(entry)
        return seq_len

    def _save_prompt_prefix_cache(self, input_ids: Tensor, cache: object) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return
        if input_ids.size(0) != 1:
            return
        tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        max_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            self._prefix_cache_entry = None
            return
        seq_len = min(len(tokens), _cache_row_seq_len(cache, 0))
        if seq_len < len(tokens):
            return
        self._store_prefix_cache_entry(snapshot_tensor_prefix_cache(
            cache,
            tokens,
            seq_len=seq_len,
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
        ))

    def _save_prompt_prefix_cache_row(
        self,
        tokens: Sequence[int],
        cache: object,
        *,
        row: int,
        seq_len: int,
    ) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return
        token_tuple = tuple(int(token_id) for token_id in tokens)
        max_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        row_max_tokens = env_int(
            "TORCHINFERNO_OPENAI_PREFIX_CACHE_ROW_MAX_TOKENS",
            16,
            minimum=1,
        )
        if len(token_tuple) > min(max_tokens, row_max_tokens) or seq_len < len(token_tuple):
            return
        self._store_prefix_cache_entry(snapshot_tensor_prefix_cache(
            cache,
            token_tuple,
            seq_len=seq_len,
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
            row=row,
        ))

    def _save_prompt_prefix_cache_rows(
        self,
        prompts: Sequence[Sequence[int]],
        cache: object,
        *,
        row_indices: Sequence[int] | None = None,
    ) -> None:
        indices = range(len(prompts)) if row_indices is None else row_indices
        for row, prompt_index in enumerate(indices):
            prompt = prompts[int(prompt_index)]
            self._save_prompt_prefix_cache_row(
                prompt,
                cache,
                row=row,
                seq_len=len(prompt),
            )

    def _prefix_cached_prompt_groups(
        self,
        prompts: Sequence[Sequence[int]],
    ) -> list[list[_PrefixCachedPrompt]]:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE"):
            return []
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return []
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return []
        min_prefix_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1)
        use_ragged_suffixes = _ragged_decode_enabled_for_model(self.model)
        groups: dict[tuple[int, int], list[_PrefixCachedPrompt]] = {}
        for index, prompt in enumerate(prompts):
            input_tokens = tuple(int(token_id) for token_id in prompt)
            for prefix_tokens, entry in self._prefix_cache_restore_candidates(input_tokens, min_prefix_tokens):
                if (
                    entry.device != str(self.device)
                    or entry.backend != self.cache_backend
                    or entry.page_size != self.page_size
                ):
                    continue
                suffix_len = len(input_tokens) - prefix_tokens
                if suffix_len <= 0:
                    continue
                suffix_key = -1 if use_ragged_suffixes else suffix_len
                groups.setdefault((prefix_tokens, suffix_key), []).append(
                    _PrefixCachedPrompt(index, list(prompt), entry, prefix_tokens)
                )
                break
        min_rows = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE_MIN_ROWS", 2, minimum=1)
        grouped = [group for group in groups.values() if len(group) >= min_rows]
        grouped.sort(key=lambda group: (group[0].prefix_tokens, len(group)), reverse=True)
        signature = _prefix_cached_prompt_groups_signature(grouped) if grouped else 0
        if not _tensor_parallel_all_ranks_same_int(self.model, signature, self.device):
            return []
        return grouped

    def _clear_prefix_cache(self) -> None:
        self._prefix_cache_entry = None
        entries = getattr(self, "_prefix_cache_entries", None)
        if isinstance(entries, dict):
            entries.clear()
        else:
            self._prefix_cache_entries = {}
        logits_cache = getattr(self, "_prompt_logits_cache", None)
        if isinstance(logits_cache, dict):
            logits_cache.clear()
        else:
            self._prompt_logits_cache = {}

    def _prefix_cache_entry_map(self) -> dict[tuple[int, ...], TensorPrefixCacheEntry]:
        entries = getattr(self, "_prefix_cache_entries", None)
        if not isinstance(entries, dict):
            entries = {}
            self._prefix_cache_entries = entries
        return entries

    def _prompt_logits_cache_map(self) -> dict[tuple[int, ...], Tensor]:
        entries = getattr(self, "_prompt_logits_cache", None)
        if not isinstance(entries, dict):
            entries = {}
            self._prompt_logits_cache = entries
        return entries

    def _store_prefix_cache_entry(self, entry: TensorPrefixCacheEntry | None) -> None:
        self._prefix_cache_entry = entry
        if entry is None:
            return
        entries = self._prefix_cache_entry_map()
        entries.pop(entry.tokens, None)
        entries[entry.tokens] = entry
        max_entries = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", 128, minimum=1)
        while len(entries) > max_entries:
            evicted_tokens = next(iter(entries))
            entries.pop(evicted_tokens)
            self._prompt_logits_cache_map().pop(evicted_tokens, None)

    def _store_prompt_logits_cache(self, input_ids: Tensor, logits: Tensor) -> None:
        if not _prompt_logits_cache_enabled():
            return
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return
        if input_ids.size(0) != 1:
            return
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        if logits.ndim != 2 or logits.size(0) != 1:
            return
        tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        max_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            self._prompt_logits_cache_map().pop(tokens, None)
            return
        if self._exact_prefix_cache_entry(tokens) is None:
            return
        entries = self._prompt_logits_cache_map()
        entries.pop(tokens, None)
        entries[tokens] = logits.detach().clone()
        max_entries = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", 128, minimum=1)
        while len(entries) > max_entries:
            entries.pop(next(iter(entries)))

    def _restore_exact_prompt_logits(
        self,
        input_ids: Tensor,
        cache: object,
        *,
        restore_cache: bool = True,
    ) -> Tensor | None:
        if not _prompt_logits_cache_enabled():
            return None
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return None
        if not _prefix_cache_enabled_for_model(self.model):
            self._clear_prefix_cache()
            return None
        if input_ids.size(0) != 1:
            return None
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        entries = self._prompt_logits_cache_map()
        logits = entries.get(input_tokens)
        restored = 0
        if logits is not None:
            restored = (
                self._restore_exact_prefix_cache(input_ids, cache)
                if restore_cache
                else input_ids.size(1)
            )
        local_hit = logits is not None and restored == input_ids.size(1)
        all_ranks_hit = _tensor_parallel_all_ranks_true(self.model, local_hit, self.device)
        if not all_ranks_hit:
            if restore_cache and restored:
                _reset_generation_cache(cache)
            return None
        entries.pop(input_tokens, None)
        entries[input_tokens] = logits
        return logits.to(self.device, non_blocking=True)

    def _mark_prefix_cache_entry_used(self, entry: TensorPrefixCacheEntry) -> None:
        self._prefix_cache_entry = entry
        entries = self._prefix_cache_entry_map()
        if entries.get(entry.tokens) is entry:
            entries.pop(entry.tokens, None)
            entries[entry.tokens] = entry

    def _exact_prefix_cache_entry(self, input_tokens: tuple[int, ...]) -> TensorPrefixCacheEntry | None:
        entry = self._prefix_cache_entry_map().get(input_tokens)
        if entry is not None:
            return entry
        latest = getattr(self, "_prefix_cache_entry", None)
        if isinstance(latest, TensorPrefixCacheEntry) and latest.tokens == input_tokens:
            return latest
        return None

    def _prefix_cache_restore_candidates(
        self,
        input_tokens: tuple[int, ...],
        min_prefix_tokens: int,
    ) -> list[tuple[int, TensorPrefixCacheEntry]]:
        candidates: list[tuple[int, TensorPrefixCacheEntry]] = []
        seen: set[tuple[int, ...]] = set()
        for entry in self._prefix_cache_entry_map().values():
            seen.add(entry.tokens)
            prefix_tokens = _matching_prefix_tokens(entry.tokens, input_tokens, min_prefix_tokens)
            if prefix_tokens > 0:
                candidates.append((prefix_tokens, entry))
        latest = getattr(self, "_prefix_cache_entry", None)
        if isinstance(latest, TensorPrefixCacheEntry) and latest.tokens not in seen:
            prefix_tokens = _matching_prefix_tokens(latest.tokens, input_tokens, min_prefix_tokens)
            if prefix_tokens > 0:
                candidates.append((prefix_tokens, latest))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return candidates

    def _materialize_generated_cache_tokens(
        self,
        input_ids: Tensor,
        generated_tokens: list[int],
        cache: object,
    ) -> None:
        cached_generated_tokens = max(0, _generation_cache_seq_len(cache) - input_ids.size(1))
        missing = generated_tokens[cached_generated_tokens:]
        if not missing:
            return
        token_tensor = torch.tensor([missing], dtype=torch.long, device=self.device)
        _forward(self.model, token_tensor, cache)

    def _warmup_tokenizer(self) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_TOKENIZER_WARMUP", True):
            return
        if not isinstance(self.tokenizer, _TransformersChatTokenizer):
            return
        try:
            warmup_text = " ".join(f"tok{idx:02d}" for idx in range(32))
            self.tokenizer.encode_messages(
                [{"role": "user", "content": warmup_text}]
            )
            self.tokenizer.encode_messages(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": warmup_text},
                ]
            )
        except Exception as exc:
            warn_optional_failure("openai.tokenizer_warmup", exc)

    def _warmup_tensor_parallel_control_group(self) -> None:
        if not (
            _is_tensor_parallel_model(self.model)
            and _tensor_parallel_world_size(self.model) > 1
            and env_flag("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", True)
        ):
            return
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            _tensor_parallel_control_group(dist)

    def _warmup_tensor_parallel_model(self) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_STARTUP_WARMUP", True):
            return
        if not _is_tensor_parallel_model(self.model) or self.device.type != "cuda":
            return
        if not hasattr(self.model, "generate"):
            return
        if not _startup_warmup_enabled_for_cache_backend(str(getattr(self, "cache_backend", "dense"))):
            return
        prompt_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKENS", 32, minimum=1)
        prompt_token_counts = _warmup_prompt_token_counts(prompt_tokens)
        new_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_NEW_TOKENS", 2, minimum=1)
        vocab_size = max(1, int(getattr(getattr(self.model, "config", object()), "vocab_size", 1)))
        with torch.inference_mode():
            with _tensor_parallel_symm_mem_allreduce_scope(
                self.model,
                self.device,
                max_tokens=new_tokens,
                temperature=0.0,
            ):
                for count in prompt_token_counts:
                    input_ids = (torch.arange(count, device=self.device, dtype=torch.long) % vocab_size)[None, :]
                    for _ in self._generate_single_tokens(
                        input_ids,
                        max_tokens=new_tokens,
                        temperature=0.0,
                        broadcast_tensor_parallel=False,
                        update_prefix_cache=False,
                    ):
                        pass
            self._warmup_tensor_parallel_prefill_graphs(prompt_token_counts, vocab_size)
            self._warmup_tensor_parallel_prefix_suffix_graphs(vocab_size)
            self._warmup_tensor_parallel_temperature_graphs(vocab_size)
            self._warmup_tensor_parallel_resident_temperature_graphs(vocab_size)
            self._warmup_tensor_parallel_ragged_decode_graphs(vocab_size)
            if env_flag("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_SYMM_GRAPHS", True):
                with _tensor_parallel_symm_mem_allreduce_scope(
                    self.model,
                    self.device,
                    max_tokens=1,
                    temperature=0.0,
                ):
                    self._warmup_tensor_parallel_ragged_decode_graphs(vocab_size)
            self._warmup_tensor_parallel_batched_prefix_suffix_graphs(vocab_size)
            warmup_cache_tokens = max(
                max(prompt_token_counts) + new_tokens,
                env_int("TORCHINFERNO_OPENAI_WARMUP_CACHE_TOKENS", 256, minimum=1),
            )
            self._generation_cache(1, warmup_cache_tokens, model=self.model, pool=False)
            _warmup_tensor_parallel_decode_attention(self.model)
            if (
                env_flag("TORCHINFERNO_OPENAI_UNIFIED_SCHEDULER", False)
                or env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", False)
            ) and hasattr(self.model, "allocate_cache"):
                self._warmup_unified_scheduler_cache(vocab_size)
        torch.cuda.synchronize(self.device)

    def _warmup_tensor_parallel_prefill_graphs(
        self,
        prompt_token_counts: Sequence[int],
        vocab_size: int,
    ) -> None:
        if not env_flag("TORCHINFERNO_CUDAGRAPH_PREFILL", True):
            return
        cache_token_counts = _warmup_prefill_cache_token_counts()
        if not cache_token_counts:
            return
        for cache_tokens in cache_token_counts:
            cache = self._generation_cache(1, cache_tokens, model=self.model)
            for count in prompt_token_counts:
                if count > cache_tokens:
                    continue
                input_ids = (torch.arange(count, device=self.device, dtype=torch.long) % vocab_size)[None, :]
                _try_prefill_graph(self.model, input_ids, cache, 0.0, allow_capture=True)
                _reset_generation_cache(cache)

    def _warmup_tensor_parallel_prefix_suffix_graphs(self, vocab_size: int) -> None:
        if not env_flag("TORCHINFERNO_CUDAGRAPH_PREFILL", True):
            return
        specs = _warmup_prefix_suffix_token_counts()
        cache_token_counts = _warmup_prefix_suffix_cache_token_counts()
        if not specs or not cache_token_counts:
            return
        for cache_tokens in cache_token_counts:
            cache = self._generation_cache(1, cache_tokens, model=self.model)
            for prefix_tokens, suffix_tokens in specs:
                if prefix_tokens + suffix_tokens > cache_tokens:
                    continue
                _set_generation_cache_seq_len(cache, prefix_tokens)
                input_ids = (
                    (torch.arange(suffix_tokens, device=self.device, dtype=torch.long) + prefix_tokens)
                    % vocab_size
                )[None, :]
                _try_prefill_graph(self.model, input_ids, cache, 0.0, allow_capture=True)
                _reset_generation_cache(cache)

    def _warmup_tensor_parallel_temperature_graphs(self, vocab_size: int) -> None:
        if not env_flag("TORCHINFERNO_CUDAGRAPH_PREFILL", True):
            return
        prompt_counts = _warmup_temperature_prompt_token_counts()
        batch_sizes = _warmup_temperature_batch_sizes()
        cache_token_counts = _warmup_prefill_cache_token_counts()
        if not prompt_counts or not batch_sizes or not cache_token_counts:
            return
        for batch_size in batch_sizes:
            for cache_tokens in cache_token_counts:
                cache = self._generation_cache(batch_size, cache_tokens, model=self.model)
                for count in prompt_counts:
                    if count > cache_tokens:
                        continue
                    base = torch.arange(count, device=self.device, dtype=torch.long) % vocab_size
                    input_ids = base[None, :].expand(batch_size, count).contiguous()
                    self._warmup_temperature_prefill_decode_graphs(input_ids, cache, batch_size)
                    if batch_size > 1:
                        _reset_generation_cache(cache)
                        self._warmup_temperature_prefill_decode_graphs(base[None, :], cache, batch_size)
                    _reset_generation_cache(cache)

    def _warmup_temperature_prefill_decode_graphs(
        self,
        input_ids: Tensor,
        cache: object,
        batch_size: int,
    ) -> None:
        logits = _try_prefill_logits_graph(self.model, input_ids, cache, allow_capture=True)
        if logits is None:
            return
        if logits.size(0) == batch_size:
            next_token = _sample(self.model, logits[:, -1, :], 0.7).to(self.device)
            decode_input = next_token[:, None]
        elif _shared_prefix_sample_enabled(0.7):
            next_token = _sample(self.model, logits[:, -1, :], 0.7).to(self.device)
            decode_input = next_token[:1, None]
        else:
            sample_logits = logits[:, -1, :].expand(batch_size, logits.size(-1)).contiguous()
            next_token = _sample(self.model, sample_logits, 0.7).to(self.device)
            decode_input = next_token[:, None]
        _repeat_generation_cache_first_batch(cache, batch_size)
        _try_decode_one_token_logits_graph(self.model, decode_input, cache)

    def _warmup_tensor_parallel_resident_temperature_graphs(self, vocab_size: int) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_WARMUP_RESIDENT_TEMPERATURE_GRAPHS", True):
            return
        if not env_flag("TORCHINFERNO_CUDAGRAPH_PREFILL", True):
            return
        prompt_counts = _warmup_temperature_prompt_token_counts()
        cache_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_RESIDENT_TEMPERATURE_CACHE_TOKENS", 512, minimum=1)
        if not prompt_counts:
            return
        batch_sizes = [
            batch_size
            for batch_size in _warmup_temperature_batch_sizes()
            if batch_size <= env_int("TORCHINFERNO_OPENAI_WARMUP_RESIDENT_TEMPERATURE_MAX_BATCH", 16, minimum=1)
        ]
        for batch_size in batch_sizes:
            cache = self._generation_cache(batch_size, cache_tokens, model=self.model)
            for count in prompt_counts:
                if count > cache_tokens:
                    continue
                base = torch.arange(count, device=self.device, dtype=torch.long) % vocab_size
                input_ids = base[None, :].expand(batch_size, count).contiguous()
                self._warmup_temperature_prefill_decode_graphs(input_ids, cache, batch_size)
                if batch_size > 1:
                    _reset_generation_cache(cache)
                    self._warmup_temperature_prefill_decode_graphs(base[None, :], cache, batch_size)
                _reset_generation_cache(cache)

    def _warmup_tensor_parallel_batched_prefix_suffix_graphs(self, vocab_size: int) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_WARMUP_BATCHED_PREFIX_SUFFIX", False):
            return
        if not (
            _is_tensor_parallel_model(self.model)
            and _tensor_parallel_world_size(self.model) > 1
            and self.device.type == "cuda"
        ):
            return
        effective_max = _effective_openai_max_batch_size(self.model, self.device, self.max_batch_size)
        batch_size = _generation_cache_batch_capacity(self.model, effective_max)
        if batch_size <= 1:
            return
        for prefix_tokens, suffix_tokens in _warmup_prefix_suffix_token_counts():
            cache_tokens = prefix_tokens + suffix_tokens + 16
            try:
                cache = self._generation_cache(
                    batch_size, cache_tokens, model=self.model, pool=False,
                    batch_capacity=batch_size,
                )
            except Exception:
                continue
            try:
                _set_generation_cache_seq_len(cache, prefix_tokens)
                input_ids = (
                    torch.arange(suffix_tokens, device=self.device, dtype=torch.long) % vocab_size
                )[None, :].expand(batch_size, -1).contiguous()
                logit_positions = torch.full(
                    (batch_size,), suffix_tokens - 1,
                    dtype=torch.long, device=self.device,
                )
                _forward_selected_logits(
                    self.model, input_ids, cache, logit_positions,
                    allow_capture=True,
                )
            except Exception:
                pass
            finally:
                _reset_generation_cache(cache)

    def _warmup_tensor_parallel_ragged_decode_graphs(self, vocab_size: int) -> None:
        if not env_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP", True):
            return
        if not _ragged_decode_enabled_for_model(self.model):
            return
        prompt_tokens = _warmup_ragged_decode_prompt_tokens(64)
        batch_sizes = _warmup_ragged_decode_batch_sizes()
        row_counts = _warmup_ragged_decode_row_counts()
        cache_token_counts = _warmup_ragged_decode_cache_token_counts()
        if not batch_sizes or not row_counts or not cache_token_counts:
            return
        cache_specs = tuple(
            dict.fromkeys(
                [(batch_size, cache_tokens) for batch_size in batch_sizes for cache_tokens in cache_token_counts]
                + list(_warmup_ragged_decode_extra_cache_specs())
            )
        )
        force_row_indices = _force_tp_shared_prefix_ragged_row_indices(self.model) or env_flag(
            "TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_FORCE_ROW_INDICES",
            False,
        )
        for batch_size, cache_tokens in cache_specs:
            if prompt_tokens >= cache_tokens:
                continue
            cache = self._generation_cache(batch_size, cache_tokens, model=self.model)
            base = torch.arange(prompt_tokens, device=self.device, dtype=torch.long) % vocab_size
            input_ids = base[None, :].expand(batch_size, prompt_tokens).contiguous()
            next_token, cache = _prefill_next_token(
                self.model,
                input_ids,
                cache,
                0.0,
                allow_capture=True,
            )
            next_token = next_token.to(self.device)
            seq_lens = torch.full((batch_size,), prompt_tokens, dtype=torch.long, device=self.device)
            seen_shapes: set[tuple[int, bool]] = set()
            for row_count in row_counts:
                rows = min(batch_size, int(row_count))
                if rows <= 0:
                    continue
                use_row_indices = force_row_indices or rows != batch_size
                shape = (rows, use_row_indices)
                if shape in seen_shapes:
                    continue
                seen_shapes.add(shape)
                row_indices = torch.arange(rows, dtype=torch.long, device=self.device) if use_row_indices else None
                decode_input = next_token[:rows, None] if row_indices is not None else next_token[:, None]
                warm_token_graph = env_flag("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_TOKEN_GRAPHS", True)
                warm_logits_graph = env_flag("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_LOGITS_GRAPHS", True)
                graph_token = None
                if warm_token_graph:
                    graph_token = _try_decode_ragged_token_graph(
                        self.model,
                        decode_input,
                        cache,
                        seq_lens=seq_lens,
                        row_indices=row_indices,
                        temperature=0.0,
                    )
                if warm_logits_graph or (warm_token_graph and graph_token is None):
                    _try_decode_ragged_logits_graph(
                        self.model,
                        decode_input,
                        cache,
                        seq_lens=seq_lens,
                        row_indices=row_indices,
                    )
            _reset_generation_cache(cache)

    def _generate_prompt_token_list(self, prompt: list[int], *, max_tokens: int, temperature: float) -> list[int]:
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        rows = self._generate_batch_tokens(input_ids, max_tokens=max_tokens, temperature=temperature)
        return rows[0] if rows else []

    def _generate_prompt_tokens_with_phase(
        self,
        prompt: list[int],
        *,
        max_tokens: int,
        temperature: float,
        phase: dict[str, float] | None,
    ) -> Iterator[int]:
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        self._mark_phase(phase, "built_input_tensor")
        yield from self._generate_single_tokens(input_ids, max_tokens=max_tokens, temperature=temperature, phase=phase)

    @torch.inference_mode()
    def _generate_single_tokens(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        phase: dict[str, float] | None = None,
        update_prefix_cache: bool = True,
    ) -> Iterator[int]:
        if max_tokens <= 0:
            return
        if input_ids.size(0) != 1:
            raise ValueError("single-request generation expects batch size 1")
        if broadcast_tensor_parallel:
            self._mark_phase(phase, "broadcast_start")
            _broadcast_tensor_parallel_generate(
                self.model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            self._mark_phase(phase, "broadcast_done")
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            generated = _generate_with_model(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
            for token in generated[0, input_ids.size(1) :].detach().cpu().tolist():
                token_id = int(token)
                yield token_id
                if token_id in stop_token_ids:
                    break
            return

        self._mark_phase(phase, "cache_start")
        cache = self._generation_cache(
            1,
            input_ids.size(1) + max_tokens,
            model=model,
        )
        self._mark_phase(phase, "cache_done")
        prefix_len = self._restore_prefix_cache(input_ids, cache) if update_prefix_cache else 0
        if phase is not None:
            phase["prefix_cache_tokens"] = float(prefix_len)
        if prefix_len > 0:
            self._mark_phase(phase, "prefix_cache_done")
        prefill_input_ids = input_ids[:, prefix_len:] if 0 < prefix_len < input_ids.size(1) else input_ids
        if phase is not None:
            phase["prefill_tokens"] = float(prefill_input_ids.size(1))
        self._mark_phase(phase, "prefill_start")
        allow_capture = self._single_prefill_graph_capture_enabled(
            model,
            prefill_input_ids,
            cache,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prefill_token = _try_prefill_graph(model, prefill_input_ids, cache, temperature, allow_capture=allow_capture)
        if prefill_token is None:
            prefill_logits = _try_prefill_logits_graph(model, prefill_input_ids, cache, allow_capture=allow_capture)
        else:
            prefill_logits = None
        if prefill_token is None and prefill_logits is None:
            self._mark_phase(phase, "first_forward_start")
            logits, cache = _forward(model, prefill_input_ids, cache)
            self._mark_phase(phase, "first_forward_done")
            next_token = _sample(model, logits[:, -1, :], temperature).to(self.device)
            self._mark_phase(phase, "prefill_sample_done")
        elif prefill_logits is not None:
            next_token = _sample(model, prefill_logits[:, -1, :], temperature).to(self.device)
            self._mark_phase(phase, "prefill_logits_graph_done")
        else:
            next_token = prefill_token.to(self.device)
            self._mark_phase(phase, "prefill_graph_done")
        generated_tokens: list[int] = []
        drained_after_close = False
        for step in range(max_tokens):
            if step == 0:
                self._mark_phase(phase, "first_token_sync_start")
            token_tensor = next_token
            token_id = int(token_tensor.item())
            generated_tokens.append(token_id)
            if step == 0:
                self._mark_phase(phase, "first_token_ready")
                self._record_phase(phase)
            try:
                yield token_id
            except GeneratorExit:
                if _is_tensor_parallel_primary_model(model):
                    drained_after_close = True
                    self._drain_closed_single_generation(
                        model,
                        token_tensor,
                        cache,
                        temperature,
                        stop_token_ids,
                        step,
                        max_tokens,
                        generated_tokens,
                    )
                    if update_prefix_cache:
                        self._save_prefix_cache(input_ids, generated_tokens, cache)
                raise
            if token_id in stop_token_ids:
                break
            if step + 1 == max_tokens:
                break
            next_token, cache = _decode_next_token(model, token_tensor[:, None], cache, temperature)
            next_token = next_token.to(self.device)
        if update_prefix_cache and not drained_after_close:
            self._save_prefix_cache(input_ids, generated_tokens, cache)

    def _single_prefill_graph_capture_enabled(
        self,
        model: object,
        input_ids: Tensor,
        cache: object,
        *,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
            return env_flag(
                "TORCHINFERNO_OPENAI_SINGLE_RUNTIME_PREFILL_CAPTURE",
                False,
            ) and _runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens)
        if "TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE" in os.environ:
            return _runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens)
        if not env_flag("TORCHINFERNO_OPENAI_TP_SINGLE_RUNTIME_PREFILL_CAPTURE", False):
            return False
        if temperature > 0.0 and not env_flag(
            "TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE",
            True,
        ):
            return False
        token_limit = env_int("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", 1024, minimum=1)
        if max_tokens > token_limit:
            return False
        layers = tuple(getattr(cache, "layers", ()) or ())
        max_seq_len = int(getattr(layers[0], "max_seq_len", 0)) if layers else 0
        key = (
            int(input_ids.size(0)),
            int(input_ids.size(1)),
            _generation_cache_seq_len(cache),
            max_seq_len,
            temperature > 0.0,
            str(self.device),
        )
        seen = self._single_prefill_capture_seen
        count = seen.get(key, 0) + 1
        seen[key] = count
        max_entries = env_int("TORCHINFERNO_OPENAI_TP_SINGLE_RUNTIME_PREFILL_CAPTURE_MAX_ENTRIES", 256, minimum=1)
        while len(seen) > max_entries:
            seen.pop(next(iter(seen)))
        min_hits = env_int("TORCHINFERNO_OPENAI_TP_SINGLE_RUNTIME_PREFILL_CAPTURE_MIN_HITS", 2, minimum=1)
        return count >= min_hits

    def _batched_prefill_graph_capture_enabled(
        self,
        model: object,
        input_ids: Tensor,
        cache: object,
        *,
        temperature: float,
        max_tokens: int,
        selected_logits: bool = False,
    ) -> bool:
        tensor_parallel_cuda = (
            _is_tensor_parallel_model(model)
            and _tensor_parallel_world_size(model) > 1
            and self.device.type == "cuda"
        )
        if _runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens) and not (
            selected_logits and tensor_parallel_cuda
        ):
            return True
        if not tensor_parallel_cuda:
            return False
        if selected_logits:
            selected_capture_env_set = "TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE" in os.environ
            if selected_capture_env_set:
                if not env_flag("TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE", False):
                    return False
            else:
                skip_max_tokens = env_int(
                    "TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE_SKIP_MAX_TOKENS",
                    128,
                    minimum=0,
                )
                if max_tokens <= skip_max_tokens:
                    return False
                min_selected_batch = env_int(
                    "TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE_MIN_BATCH",
                    48,
                    minimum=1,
                )
                if input_ids.size(0) < min_selected_batch:
                    return False
        if (
            "TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE" in os.environ
            and not env_flag("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", True)
        ):
            return False
        if not env_flag("TORCHINFERNO_OPENAI_TP_REPEATED_RUNTIME_PREFILL_CAPTURE", True):
            return False
        if input_ids.ndim != 2 or input_ids.size(0) <= 1 or input_ids.size(1) <= 1:
            return False
        if temperature > 0.0 and not env_flag(
            "TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE",
            True,
        ):
            return False
        token_limit = env_int("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", 1024, minimum=1)
        if max_tokens > token_limit:
            return False
        layers = tuple(getattr(cache, "layers", ()) or ())
        cache_seq_len = _generation_cache_seq_len_if_uniform(cache)
        if cache_seq_len is None:
            sliced_cache = _cache_row_slice(cache, 0, int(input_ids.size(0)))
            if sliced_cache is None:
                return False
            cache = sliced_cache
            cache_seq_len = _generation_cache_seq_len_if_uniform(cache)
            if cache_seq_len is None:
                return False
            layers = tuple(getattr(cache, "layers", ()) or ())
        max_seq_len = int(getattr(layers[0], "max_seq_len", 0)) if layers else 0
        if max_seq_len <= 0 or max_seq_len > token_limit:
            return False
        key = (
            int(input_ids.size(0)),
            int(input_ids.size(1)),
            cache_seq_len,
            max_seq_len,
            temperature > 0.0,
            bool(selected_logits),
            str(self.device),
        )
        seen = getattr(self, "_batched_prefill_capture_seen", None)
        if not isinstance(seen, dict):
            seen = {}
            self._batched_prefill_capture_seen = seen
        count = seen.get(key, 0) + 1
        seen[key] = count
        max_entries = env_int("TORCHINFERNO_OPENAI_TP_BATCHED_RUNTIME_PREFILL_CAPTURE_MAX_ENTRIES", 512, minimum=1)
        while len(seen) > max_entries:
            seen.pop(next(iter(seen)))
        min_hits = env_int("TORCHINFERNO_OPENAI_TP_BATCHED_RUNTIME_PREFILL_CAPTURE_MIN_HITS", 2, minimum=1)
        return count >= min_hits

    @torch.inference_mode()
    def _generate_batch_steps(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: Sequence[int] | None = None,
        prefix_cache_prompts: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[list[int | None]]:
        if max_tokens <= 0:
            return
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_generate(
                self.model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                row_max_tokens=row_max_tokens,
            )
        if input_ids.size(0) > 1 and _tensor_rows_are_identical(input_ids):
            yield from self._generate_identical_prompt_batch_steps(
                input_ids[:1],
                batch_size=input_ids.size(0),
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=row_max_tokens,
            )
            return
        microbatch_size = self._stream_microbatch_size(input_ids.size(0))
        shared_prefix_tokens = self._shared_prefix_batch_tokens(input_ids)
        if shared_prefix_tokens > 0:
            yield from self._generate_shared_prefix_batch_steps(
                input_ids,
                prefix_tokens=shared_prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                microbatch_size=microbatch_size,
                row_max_tokens=row_max_tokens,
                prefix_cache_prompts=prefix_cache_prompts,
            )
            return
        if 0 < microbatch_size < input_ids.size(0):
            yield from self._generate_batch_steps_microbatched(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                microbatch_size=microbatch_size,
                row_max_tokens=row_max_tokens,
                prefix_cache_prompts=prefix_cache_prompts,
            )
            return
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            rows = _generated_rows_with_model(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
            )
            yield from _iter_generated_steps(rows, max_tokens, stop_token_ids)
            return

        cache = self._generation_cache(
            input_ids.size(0),
            input_ids.size(1) + max_tokens,
            model=model,
        )
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, input_ids.size(0), max_tokens)
        active = [limit > 0 for limit in per_row_limits]
        next_token, cache = _prefill_next_token(
            model,
            input_ids,
            cache,
            temperature,
            allow_capture=self._batched_prefill_graph_capture_enabled(
                model,
                input_ids,
                cache,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        next_token = next_token.to(self.device)
        seq_lens = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long, device=self.device)
        for step in range(max_tokens):
            token_ids = next_token.detach().cpu().tolist()
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(token_ids):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if token_id in stop_token_ids or step + 1 >= per_row_limits[row]:
                    active[row] = False
            yield step_tokens
            if step == 0 and prefix_cache_prompts is not None:
                self._save_prompt_prefix_cache_rows(prefix_cache_prompts, cache)
            should_continue = step + 1 < max_tokens and any(active)
            should_continue = _sync_tensor_parallel_continue(model, should_continue, next_token.device)
            if not should_continue:
                break
            if _ragged_decode_enabled_for_model(model) and not all(active):
                active_indices = [index for index, is_active in enumerate(active) if is_active]
                row_indices = torch.tensor(active_indices, dtype=torch.long, device=self.device)
                decode_input = next_token.index_select(0, row_indices)[:, None]
                decoded_token, cache = _decode_next_token_ragged(
                    model,
                    decode_input,
                    cache,
                    seq_lens,
                    row_indices,
                    temperature,
                )
                decoded_token = decoded_token.to(self.device)
                next_token = next_token.clone()
                next_token[row_indices] = decoded_token
                seq_lens[row_indices] = seq_lens.index_select(0, row_indices) + 1
            else:
                next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
                next_token = next_token.to(self.device)
                seq_lens += 1

    @torch.inference_mode()
    def _generate_identical_prompt_batch_steps(
        self,
        input_ids: Tensor,
        *,
        batch_size: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int] | None = None,
    ) -> Iterator[list[int | None]]:
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            expanded = input_ids.expand(batch_size, input_ids.size(1)).contiguous()
            rows = _generated_rows_with_model(
                model,
                expanded,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
            )
            yield from _iter_generated_steps(rows, max_tokens, stop_token_ids)
            return

        decode_batch_size = _sampled_batch_shape_bucket_size(
            model,
            self.device,
            batch_size,
            temperature,
        )
        cache = self._generation_cache(
            decode_batch_size,
            input_ids.size(1) + max_tokens,
            model=model,
            pool=_identical_prompt_cache_pool_enabled(model, temperature),
        )
        cache_materialized = True
        cache_materialized_input_ids = input_ids

        def ensure_cache_materialized() -> None:
            nonlocal cache, cache_materialized
            if cache_materialized:
                return
            restored = self._restore_exact_prefix_cache(cache_materialized_input_ids, cache)
            if restored != cache_materialized_input_ids.size(1):
                cache = _prefill_cache_only(
                    model,
                    cache_materialized_input_ids,
                    cache,
                    allow_capture=_identical_prompt_prefill_graph_capture_enabled(
                        model,
                        temperature,
                        max_tokens=max_tokens,
                    ),
                )
            _repeat_generation_cache_first_batch(cache, decode_batch_size)
            cache_materialized = True

        use_prompt_logits_cache = batch_size > 1 and _prompt_logits_cache_enabled()
        if use_prompt_logits_cache:
            cached_logits = self._restore_exact_prompt_logits(input_ids, cache, restore_cache=False)
            if cached_logits is None:
                next_token, cache, last_logits = _prefill_repeated_prefix_next_token_with_logits(
                    model,
                    input_ids,
                    cache,
                    decode_batch_size,
                    temperature,
                    allow_capture=_identical_prompt_prefill_graph_capture_enabled(
                        model,
                        temperature,
                        max_tokens=max_tokens,
                    ),
                )
                self._save_prompt_prefix_cache(input_ids, cache)
                self._store_prompt_logits_cache(input_ids, last_logits)
            else:
                next_token = _sample_repeated_prefix_logits(
                    model,
                    cached_logits,
                    decode_batch_size,
                    temperature,
                )
                cache_materialized = False
                cache_materialized_input_ids = input_ids
        else:
            next_token, cache = _prefill_repeated_prefix_next_token(
                model,
                input_ids,
                cache,
                decode_batch_size,
                temperature,
                allow_capture=_identical_prompt_prefill_graph_capture_enabled(
                    model,
                    temperature,
                    max_tokens=max_tokens,
                ),
            )
        next_token = next_token.to(self.device)
        if cache_materialized:
            _repeat_generation_cache_first_batch(cache, decode_batch_size)
        shared_sample = _shared_prefix_sample_enabled(temperature)
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, batch_size, max_tokens)
        decode_row_limits = [*per_row_limits, *([0] * (decode_batch_size - batch_size))]
        active = [limit > 0 for limit in decode_row_limits]
        rows_share_state = True
        for step in range(max_tokens):
            token_ids = next_token.detach().cpu().tolist()
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(token_ids[:batch_size]):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if token_id in stop_token_ids or step + 1 >= decode_row_limits[row]:
                    active[row] = False
            yield step_tokens
            should_continue = step + 1 < max_tokens and any(active)
            should_continue = _sync_tensor_parallel_continue(model, should_continue, next_token.device)
            if not should_continue:
                break
            if shared_sample:
                ensure_cache_materialized()
                next_token, cache = _decode_next_token(model, next_token[:1, None], cache, temperature)
                next_token = next_token.expand(decode_batch_size).contiguous()
                _repeat_generation_cache_first_batch(cache, decode_batch_size)
                rows_share_state = True
            else:
                uniform_token = _uniform_active_token_id(token_ids, active, batch_size)
                if rows_share_state and uniform_token is not None:
                    extended_input_ids = torch.cat(
                        (input_ids, input_ids.new_tensor([[uniform_token]])),
                        dim=1,
                    )
                    cached_logits = self._restore_exact_prompt_logits(extended_input_ids, cache, restore_cache=False)
                    if cached_logits is not None:
                        next_token = _sample_repeated_prefix_logits(
                            model,
                            cached_logits,
                            decode_batch_size,
                            temperature,
                        )
                        cache_materialized = False
                        cache_materialized_input_ids = extended_input_ids
                    else:
                        ensure_cache_materialized()
                        decode_input = next_token.new_tensor([[uniform_token]])
                        next_token, cache, last_logits = _decode_repeated_prefix_next_token_with_logits(
                            model,
                            decode_input,
                            cache,
                            decode_batch_size,
                            temperature,
                        )
                        _repeat_generation_cache_first_batch(cache, decode_batch_size)
                        self._save_prompt_prefix_cache(extended_input_ids, cache)
                        self._store_prompt_logits_cache(extended_input_ids, last_logits)
                else:
                    ensure_cache_materialized()
                    next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
                    rows_share_state = False
            next_token = next_token.to(self.device)

    def _stream_microbatch_size(self, batch_size: int) -> int:
        if batch_size <= 1:
            return batch_size
        if "TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE" in os.environ:
            return min(batch_size, env_int("TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE", batch_size, minimum=1))
        if _is_tensor_parallel_model(self.model) and self.device.type == "cuda":
            return min(batch_size, env_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", 64, minimum=1))
        return batch_size

    @torch.inference_mode()
    def _generate_batch_steps_microbatched(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        microbatch_size: int,
        row_max_tokens: Sequence[int] | None = None,
        prefix_cache_prompts: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        states: list[dict[str, object]] = []
        batch_size = input_ids.size(0)
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, batch_size, max_tokens)
        first_step_tokens = [None for _ in range(batch_size)]
        for slot, start in enumerate(range(0, batch_size, microbatch_size)):
            end = min(batch_size, start + microbatch_size)
            chunk_input_ids = input_ids[start:end]
            chunk_limits = per_row_limits[start:end]
            cache = self._generation_microbatch_cache(
                slot,
                chunk_input_ids.size(0),
                chunk_input_ids.size(1) + max_tokens,
                model=model,
            )
            active = [limit > 0 for limit in chunk_limits]
            next_token, cache = _prefill_next_token(
                model,
                chunk_input_ids,
                cache,
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    chunk_input_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            next_token = next_token.to(self.device)
            for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                token_id = int(token_id)
                if not active[offset]:
                    continue
                first_step_tokens[start + offset] = token_id
                if token_id in stop_token_ids or chunk_limits[offset] <= 1:
                    active[offset] = False
            states.append(
                {
                    "start": start,
                    "active": active,
                    "cache": cache,
                    "next_token": next_token,
                    "row_max_tokens": chunk_limits,
                }
            )
            if prefix_cache_prompts is not None:
                self._save_prompt_prefix_cache_rows(prefix_cache_prompts[start:end], cache)
        yield first_step_tokens
        for step in range(1, max_tokens):
            emitted = False
            step_tokens = [None for _ in range(batch_size)]
            for state in states:
                active = state["active"]
                local_should_decode = isinstance(active, list) and any(active)
                should_decode = _sync_tensor_parallel_continue(model, local_should_decode, self.device)
                if not should_decode:
                    continue
                next_token = state["next_token"]
                cache = state["cache"]
                if not isinstance(next_token, Tensor):
                    raise RuntimeError("invalid microbatch token state")
                next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
                next_token = next_token.to(self.device)
                state["cache"] = cache
                state["next_token"] = next_token
                start = int(state["start"])
                chunk_limits = state["row_max_tokens"]
                if not isinstance(chunk_limits, list):
                    raise RuntimeError("invalid microbatch row limit state")
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    if not active[offset]:
                        continue
                    token_id = int(token_id)
                    step_tokens[start + offset] = token_id
                    if token_id in stop_token_ids or step + 1 >= int(chunk_limits[offset]):
                        active[offset] = False
                emitted = True
            if not emitted:
                break
            yield step_tokens

    def _drain_closed_single_generation(
        self,
        model: object,
        token_tensor: Tensor,
        cache: object,
        temperature: float,
        stop_token_ids: frozenset[int],
        completed_step: int,
        max_tokens: int,
        generated_tokens: list[int],
    ) -> None:
        token_id = int(token_tensor.item())
        if token_id in stop_token_ids or completed_step + 1 >= max_tokens:
            return
        next_token = token_tensor
        for _ in range(completed_step + 1, max_tokens):
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)
            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            if token_id in stop_token_ids:
                break

    def _shared_prefix_batch_tokens(self, input_ids: Tensor) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_BATCH", True):
            return 0
        if not _shared_prefix_batch_enabled_for_model(self.model):
            return 0
        if input_ids.size(0) <= 1 or input_ids.size(1) <= 1:
            return 0
        prefix_tokens = min(_common_prefix_token_count(input_ids), input_ids.size(1) - 1)
        min_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1)
        return prefix_tokens if prefix_tokens >= min_tokens else 0

    def _shared_prefix_prompt_list_tokens(self, prompts: Sequence[Sequence[int]]) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_BATCH", True):
            return 0
        if not _shared_prefix_batch_enabled_for_model(self.model):
            return 0
        if len(prompts) <= 1:
            return 0
        min_prompt_len = min((len(prompt) for prompt in prompts), default=0)
        if min_prompt_len <= 1:
            return 0
        prefix_tokens = min(_common_prefix_list_token_count(prompts), min_prompt_len - 1)
        min_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1)
        return prefix_tokens if prefix_tokens >= min_tokens else 0

    @torch.inference_mode()
    def _generate_prompt_list_batch_steps(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: Sequence[int] | None = None,
        allow_prefix_cache_restore: bool = True,
        early_restart_check: object = None,
    ) -> Iterator[list[int | None]]:
        if max_tokens <= 0:
            return
        if not prompts:
            return
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_generate_prompt_lists(
                self.model,
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                row_max_tokens=row_max_tokens,
            )
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, len(prompts), max_tokens)
        segment_rows = self._prompt_list_segment_rows(len(prompts))
        if 0 < segment_rows < len(prompts):
            segments: list[tuple[Sequence[int], Iterator[list[int | None]]]] = []
            for start in range(0, len(prompts), segment_rows):
                end = min(len(prompts), start + segment_rows)
                segment_indices = list(range(start, end))
                segments.append(
                    (
                        segment_indices,
                        self._generate_prompt_list_batch_steps(
                            [prompts[index] for index in segment_indices],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            broadcast_tensor_parallel=False,
                            row_max_tokens=per_row_limits[start:end],
                            allow_prefix_cache_restore=allow_prefix_cache_restore,
                        ),
                    )
                )
            yield from _interleave_prompt_segments(len(prompts), segments)
            return
        if len(prompts) == 1:
            input_ids = torch.tensor([prompts[0]], dtype=torch.long, device=self.device)
            for token_id in self._generate_single_tokens(
                input_ids,
                max_tokens=per_row_limits[0],
                temperature=temperature,
                broadcast_tensor_parallel=False,
            ):
                yield [token_id]
            return
        if _prompt_rows_are_identical(prompts):
            input_ids = torch.tensor([prompts[0]], dtype=torch.long, device=self.device)
            yield from self._generate_identical_prompt_batch_steps(
                input_ids,
                batch_size=len(prompts),
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
            )
            return
        if allow_prefix_cache_restore:
            cached_groups = self._prefix_cached_prompt_groups(prompts)
            if cached_groups:
                cached_indices = {item.index for group in cached_groups for item in group}
                segments: list[tuple[list[int], Iterator[list[int | None]]]] = [
                    (
                        [item.index for item in group],
                        self._generate_prefix_cached_prompt_group_steps(
                            group,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            row_max_tokens=[per_row_limits[item.index] for item in group],
                        ),
                    )
                    for group in cached_groups
                ]
                remaining = [
                    (index, prompt)
                    for index, prompt in enumerate(prompts)
                    if index not in cached_indices
                ]
                if remaining:
                    remaining_indices = [index for index, _prompt in remaining]
                    segments.append(
                        (
                            remaining_indices,
                            self._generate_prompt_list_batch_steps(
                                [prompt for _index, prompt in remaining],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                broadcast_tensor_parallel=False,
                                row_max_tokens=[per_row_limits[index] for index in remaining_indices],
                                allow_prefix_cache_restore=False,
                            ),
                        )
                )
                yield from _interleave_prompt_segments(len(prompts), segments)
                return
        prompt_lengths = {len(prompt) for prompt in prompts}
        if len(prompt_lengths) == 1:
            input_ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
            yield from self._generate_batch_steps(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                broadcast_tensor_parallel=False,
                row_max_tokens=per_row_limits,
                prefix_cache_prompts=prompts,
            )
            return
        prefix_tokens = self._shared_prefix_prompt_list_tokens(prompts)
        if prefix_tokens > 0:
            yield from self._generate_shared_prefix_prompt_list_steps(
                prompts,
                prefix_tokens=prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
                early_restart_check=early_restart_check,
            )
            return
        indexed = [
            (index, list(prompt))
            for index, prompt in enumerate(prompts)
        ]
        for same_length in _indexed_prompts_by_length(indexed):
            original_indices = [index for index, _prompt in same_length]
            input_ids = torch.tensor([prompt for _index, prompt in same_length], dtype=torch.long, device=self.device)
            for group_step in self._generate_batch_steps(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                broadcast_tensor_parallel=False,
                row_max_tokens=[per_row_limits[index] for index in original_indices],
                prefix_cache_prompts=[prompt for _index, prompt in same_length],
            ):
                step_tokens: list[int | None] = [None for _ in prompts]
                for original_index, token_id in zip(original_indices, group_step):
                    step_tokens[original_index] = token_id
                yield step_tokens

    def _prompt_list_segment_rows(self, prompt_count: int) -> int:
        if prompt_count <= 1:
            return prompt_count
        if "TORCHINFERNO_OPENAI_PROMPT_LIST_SEGMENT_ROWS" in os.environ:
            rows = env_int("TORCHINFERNO_OPENAI_PROMPT_LIST_SEGMENT_ROWS", prompt_count, minimum=0)
            return prompt_count if rows <= 0 else min(prompt_count, rows)
        model = getattr(self, "model", None)
        device = getattr(self, "device", torch.device("cpu"))
        if _is_tensor_parallel_model(model) and device.type == "cuda":
            default_rows = env_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", 64, minimum=1)
            rows = env_int("TORCHINFERNO_OPENAI_TP_PROMPT_LIST_SEGMENT_ROWS", default_rows, minimum=1)
            return min(prompt_count, rows)
        return prompt_count

    @torch.inference_mode()
    def _generate_prefix_cached_prompt_group_steps(
        self,
        group: Sequence[_PrefixCachedPrompt],
        *,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int] | None = None,
        allow_suffix_buckets: bool = True,
    ) -> Iterator[list[int | None]]:
        if max_tokens <= 0 or not group:
            return
        model = self.model
        if not _supports_incremental_generation(model):
            input_ids = torch.tensor([item.prompt for item in group], dtype=torch.long, device=self.device)
            rows = _generated_rows_with_model(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_token_ids=self.stop_token_ids,
            )
            yield from _iter_generated_steps(rows, max_tokens, self.stop_token_ids)
            return

        prefix_tokens = group[0].prefix_tokens
        prompt_lengths = [len(item.prompt) for item in group]
        suffix_rows = [item.prompt[prefix_tokens:] for item in group]
        max_suffix_len = max((len(row) for row in suffix_rows), default=0)
        if max_suffix_len <= 0:
            return
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, len(group), max_tokens)
        suffix_buckets = (
            _prefix_cached_prompt_suffix_buckets(group)
            if (
                allow_suffix_buckets
                and _ragged_decode_enabled_for_model(model)
                and min(len(row) for row in suffix_rows) < max_suffix_len
            )
            else []
        )
        if suffix_buckets and _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
            same_buckets = _tensor_parallel_all_ranks_same_int(
                model,
                _prefix_cached_prompt_suffix_buckets_signature(suffix_buckets),
                self.device,
            )
            if not same_buckets:
                suffix_buckets = []
        if suffix_buckets:
            states: list[dict[str, object]] = []
            first_tokens: list[int | None] = [None for _item in group]
            for slot, bucket in enumerate(suffix_buckets):
                state = self._prefill_prefix_cached_prompt_suffix_bucket(
                    bucket,
                    prefix_tokens=prefix_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    row_max_tokens=per_row_limits,
                    model=model,
                    slot=slot,
                )
                if state is None:
                    states = []
                    break
                local_indices = state["indices"]
                next_token = state["next_token"]
                if not isinstance(local_indices, list) or not isinstance(next_token, Tensor):
                    states = []
                    break
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    first_tokens[int(local_indices[offset])] = int(token_id)
                states.append(state)
            if not states:
                return
            yield first_tokens
            group_prompts = [item.prompt for item in group]
            for state in states:
                cache = state["cache"]
                local_indices = state["indices"]
                if isinstance(local_indices, list):
                    self._save_prompt_prefix_cache_rows(group_prompts, cache, row_indices=local_indices)

            should_continue = max_tokens > 1 and any(
                isinstance(state["active"], list) and any(state["active"])
                for state in states
            )
            should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
            if should_continue:
                combined_cache = self._shared_prefix_prompt_list_ragged_cache(
                    states,
                    prompt_lengths=prompt_lengths,
                    prompt_count=len(group),
                    max_tokens=max_tokens,
                    model=model,
                )
                use_combined_cache = combined_cache is not None
                if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
                    use_combined_cache = _tensor_parallel_all_ranks_true(
                        model,
                        use_combined_cache,
                        self.device,
                    )
                if use_combined_cache and combined_cache is not None:
                    yield from self._decode_shared_prefix_prompt_list_ragged(
                        cache=combined_cache,
                        active=[
                            token_id is not None and int(token_id) not in self.stop_token_ids
                            for token_id in first_tokens
                        ],
                        prompt_lengths=prompt_lengths,
                        max_tokens=max_tokens,
                        next_tokens=first_tokens,
                        temperature=temperature,
                        row_max_tokens=per_row_limits,
                    )
                else:
                    yield from self._decode_shared_prefix_prompt_list_state_steps(
                        states,
                        prompt_count=len(group),
                        prompt_lengths=prompt_lengths,
                        max_tokens=max_tokens,
                        next_tokens=first_tokens,
                        temperature=temperature,
                        row_max_tokens=per_row_limits,
                    )
            return

        cache = self._generation_cache(
            len(group),
            max(prompt_lengths) + max_tokens,
            model=model,
        )
        for row, item in enumerate(group):
            restored = restore_tensor_prefix_cache(
                item.entry,
                item.prompt,
                cache,
                min_prefix_tokens=prefix_tokens,
                device=str(self.device),
                backend=self.cache_backend,
                page_size=self.page_size,
                row=row,
                restore_seq_len=False,
                on_seq_len_restore_error=lambda exc: warn_optional_failure("openai.prefix_cache.seq_len_restore", exc),
            )
            if restored != prefix_tokens:
                return
        _set_generation_cache_seq_len(cache, prefix_tokens)

        if min(len(row) for row in suffix_rows) == max_suffix_len:
            suffix_ids = torch.tensor(suffix_rows, dtype=torch.long, device=self.device)
            next_token, cache = _prefill_next_token(
                model,
                suffix_ids,
                cache,
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    suffix_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = _padded_token_rows_tensor(suffix_rows, self.device, pad_token_id)
            next_token, cache = _prefill_padded_suffix_next_token(
                model,
                suffix_ids,
                cache,
                [len(suffix) for suffix in suffix_rows],
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    suffix_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    selected_logits=True,
                ),
            )
        next_token = next_token.to(self.device)

        stop_token_ids = self.stop_token_ids
        first_tokens: list[int | None] = []
        active: list[bool] = []
        for row, token_id in enumerate(next_token.detach().cpu().tolist()):
            if per_row_limits[row] <= 0:
                first_tokens.append(None)
                active.append(False)
                continue
            token = int(token_id)
            first_tokens.append(token)
            active.append(token not in stop_token_ids and per_row_limits[row] > 1)
        yield first_tokens
        self._save_prompt_prefix_cache_rows([item.prompt for item in group], cache)

        should_continue = max_tokens > 1 and any(active)
        should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
        if not should_continue:
            return
        if _ragged_decode_enabled_for_model(model):
            yield from self._decode_shared_prefix_prompt_list_ragged(
                cache=cache,
                active=active,
                prompt_lengths=prompt_lengths,
                max_tokens=max_tokens,
                next_tokens=first_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
            )
            return

        for step in range(1, max_tokens):
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(next_token.detach().cpu().tolist()):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token = int(token_id)
                step_tokens.append(token)
                if token in stop_token_ids or step + 1 >= per_row_limits[row]:
                    active[row] = False
            yield step_tokens
            should_continue = step + 1 < max_tokens and any(active)
            should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
            if not should_continue:
                break

    def _prefill_prefix_cached_prompt_suffix_bucket(
        self,
        bucket: Sequence[tuple[int, _PrefixCachedPrompt]],
        *,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int],
        model: object,
        slot: int,
    ) -> dict[str, object] | None:
        if not bucket:
            return None
        local_indices = [local_index for local_index, _item in bucket]
        prompt_rows = [item.prompt for _local_index, item in bucket]
        suffix_rows = [prompt[prefix_tokens:] for prompt in prompt_rows]
        suffix_lengths = [len(row) for row in suffix_rows]
        max_suffix_len = max(suffix_lengths, default=0)
        if max_suffix_len <= 0:
            return None
        cache = self._generation_microbatch_cache(
            slot,
            len(bucket),
            max((len(prompt) for prompt in prompt_rows), default=0) + max_tokens,
            model=model,
        )
        for row, (_local_index, item) in enumerate(bucket):
            restored = restore_tensor_prefix_cache(
                item.entry,
                item.prompt,
                cache,
                min_prefix_tokens=prefix_tokens,
                device=str(self.device),
                backend=self.cache_backend,
                page_size=self.page_size,
                row=row,
                restore_seq_len=False,
                on_seq_len_restore_error=lambda exc: warn_optional_failure("openai.prefix_cache.seq_len_restore", exc),
            )
            if restored != prefix_tokens:
                return None
        _set_generation_cache_seq_len(cache, prefix_tokens)

        if min(suffix_lengths) == max_suffix_len:
            suffix_ids = torch.tensor(suffix_rows, dtype=torch.long, device=self.device)
            next_token, cache = _prefill_next_token(
                model,
                suffix_ids,
                cache,
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    suffix_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            ragged_decode = False
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = _padded_token_rows_tensor(suffix_rows, self.device, pad_token_id)
            next_token, cache = _prefill_padded_suffix_next_token(
                model,
                suffix_ids,
                cache,
                suffix_lengths,
                temperature,
                allow_capture=(
                    _shared_prefix_suffix_bucket_selected_logits_capture_enabled(
                        model,
                        batch_size=len(prompt_rows),
                        max_tokens=max_tokens,
                        device=self.device,
                    )
                    and self._batched_prefill_graph_capture_enabled(
                        model,
                        suffix_ids,
                        cache,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        selected_logits=True,
                    )
                ),
            )
            ragged_decode = True
        next_token = next_token.to(self.device)

        active: list[bool] = []
        stop_token_ids = self.stop_token_ids
        for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
            token = int(token_id)
            active.append(token not in stop_token_ids and row_max_tokens[local_indices[offset]] > 1)
        return {
            "indices": local_indices,
            "active": active,
            "cache": cache,
            "next_token": next_token,
            "ragged_decode": ragged_decode,
        }

    @torch.inference_mode()
    def _generate_shared_prefix_prompt_list_steps(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int] | None = None,
        early_restart_check: object = None,
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            indexed = [
                (index, list(prompt))
                for index, prompt in enumerate(prompts)
            ]
            for same_length in _indexed_prompts_by_length(indexed):
                original_indices = [index for index, _prompt in same_length]
                input_ids = torch.tensor([prompt for _index, prompt in same_length], dtype=torch.long, device=self.device)
                rows = _generated_rows_with_model(
                    model,
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_token_ids=stop_token_ids,
                )
                for group_step in _iter_generated_steps(rows, max_tokens, stop_token_ids):
                    step_tokens: list[int | None] = [None for _ in prompts]
                    for original_index, token_id in zip(original_indices, group_step):
                        step_tokens[original_index] = token_id
                    yield step_tokens
            return

        indexed = [
            (index, list(prompt))
            for index, prompt in enumerate(prompts)
        ]
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, len(prompts), max_tokens)
        length_groups = _indexed_prompts_by_length(indexed)
        prompt_lengths = [len(prompt) for prompt in prompts]
        prefix_cache = self._generation_microbatch_cache(
            -1,
            1,
            prefix_tokens,
            model=model,
        )
        prefix_profile_start_s = self._stream_group_profile_start_s()
        prefix_ids = torch.tensor([length_groups[0][0][1][:prefix_tokens]], dtype=torch.long, device=self.device)
        restored_prefix_tokens = self._restore_exact_prefix_cache(prefix_ids, prefix_cache)
        restored_prefix = restored_prefix_tokens == prefix_tokens
        if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
            restored_prefix = _tensor_parallel_all_ranks_true(model, restored_prefix, self.device)
        if not restored_prefix:
            _reset_generation_cache(prefix_cache)
            prefix_cache = _prefill_cache_only(
                model,
                prefix_ids,
                prefix_cache,
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
            )
            self._save_prompt_prefix_cache(prefix_ids, prefix_cache)
        _mark_generation_cache_prefix(prefix_cache, _tensor_row_tokens(prefix_ids))
        _set_generation_cache_seq_len(prefix_cache, prefix_tokens)
        self._add_stream_group_profile_elapsed("shared_prefix_restore_ms", prefix_profile_start_s)
        self._add_stream_group_profile_extra(shared_prefix_restored=bool(restored_prefix))

        prefer_dense_group_decode = _prefer_shared_prefix_dense_group_decode(
            length_groups,
            prompt_lengths=prompt_lengths,
        )
        tensor_parallel = _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1
        prefer_padded_suffix_prefill = (
            _ragged_decode_enabled_for_model(model)
            and not prefer_dense_group_decode
            and _prefer_shared_prefix_padded_suffix_prefill(
                length_groups,
                prefix_tokens=prefix_tokens,
                prompt_lengths=prompt_lengths,
                max_tokens=max_tokens,
            )
        )
        use_padded_suffix_prefill = prefer_padded_suffix_prefill
        if tensor_parallel:
            use_padded_suffix_prefill = _tensor_parallel_all_ranks_true(
                model,
                use_padded_suffix_prefill,
                self.device,
            )
        if use_padded_suffix_prefill:
            self._add_stream_group_profile_extra(prompt_list_path="padded_suffix")
            prefill_profile_start_s = self._stream_group_profile_start_s()
            padded_state = self._prefill_shared_prefix_prompt_list_padded_suffixes(
                prompts,
                prefix_cache=prefix_cache,
                prefix_tokens=prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
                model=model,
            )
            self._add_stream_group_profile_elapsed(
                "shared_prefix_padded_suffix_prefill_ms",
                prefill_profile_start_s,
            )
            if padded_state is not None:
                combined_cache, first_tokens, active = padded_state
                yield first_tokens
                self._save_prompt_prefix_cache_rows(prompts, combined_cache)
                should_continue = max_tokens > 1 and any(active)
                should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
                if should_continue:
                    yield from self._decode_shared_prefix_prompt_list_ragged(
                        cache=combined_cache,
                        active=active,
                        prompt_lengths=prompt_lengths,
                        max_tokens=max_tokens,
                        next_tokens=first_tokens,
                        temperature=temperature,
                        row_max_tokens=per_row_limits,
                        early_restart_check=early_restart_check,
                )
                return

        suffix_buckets = (
            _shared_prefix_padded_suffix_buckets(
                length_groups,
                prefix_tokens=prefix_tokens,
                prompt_lengths=prompt_lengths,
            )
            if (
                _shared_prefix_ragged_cache_enabled_for_model(model)
                and not prefer_dense_group_decode
                and not prefer_padded_suffix_prefill
            )
            else []
        )
        if suffix_buckets and tensor_parallel:
            same_buckets = _tensor_parallel_all_ranks_same_int(
                model,
                _shared_prefix_padded_suffix_buckets_signature(suffix_buckets),
                self.device,
            )
            if not same_buckets:
                suffix_buckets = []
        if suffix_buckets:
            self._add_stream_group_profile_extra(
                prompt_list_path="suffix_buckets",
                suffix_bucket_count=len(suffix_buckets),
            )
            bucket_states: list[dict[str, object]] = []
            first_tokens: list[int | None] = [None for _ in prompts]
            prefill_profile_start_s = self._stream_group_profile_start_s()
            for slot, bucket in enumerate(suffix_buckets):
                state = self._prefill_shared_prefix_prompt_list_suffix_bucket(
                    bucket,
                    prefix_cache=prefix_cache,
                    prefix_tokens=prefix_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    row_max_tokens=per_row_limits,
                    model=model,
                    slot=slot,
                )
                if state is None:
                    bucket_states = []
                    break
                original_indices = state["indices"]
                next_token = state["next_token"]
                if not isinstance(original_indices, list) or not isinstance(next_token, Tensor):
                    bucket_states = []
                    break
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    first_tokens[int(original_indices[offset])] = int(token_id)
                bucket_states.append(state)
            self._add_stream_group_profile_elapsed(
                "shared_prefix_suffix_bucket_prefill_ms",
                prefill_profile_start_s,
            )
            if bucket_states:
                yield first_tokens
                for state in bucket_states:
                    cache = state["cache"]
                    original_indices = state["indices"]
                    if isinstance(original_indices, list):
                        self._save_prompt_prefix_cache_rows(prompts, cache, row_indices=original_indices)
                should_continue = max_tokens > 1 and any(
                    isinstance(state["active"], list) and any(state["active"])
                    for state in bucket_states
                )
                should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
                if should_continue:
                    ragged_cache_profile_start_s = self._stream_group_profile_start_s()
                    combined_cache = self._shared_prefix_prompt_list_ragged_cache(
                        bucket_states,
                        prompt_lengths=prompt_lengths,
                        prompt_count=len(prompts),
                        max_tokens=max_tokens,
                        model=model,
                    )
                    self._add_stream_group_profile_elapsed(
                        "shared_prefix_ragged_cache_ms",
                        ragged_cache_profile_start_s,
                    )
                    use_combined_cache = combined_cache is not None
                    if tensor_parallel:
                        use_combined_cache = _tensor_parallel_all_ranks_true(
                            model,
                            use_combined_cache,
                            self.device,
                        )
                    if use_combined_cache and combined_cache is not None:
                        yield from self._decode_shared_prefix_prompt_list_ragged(
                            cache=combined_cache,
                            active=[
                                token_id is not None and int(token_id) not in stop_token_ids
                                for token_id in first_tokens
                            ],
                            prompt_lengths=prompt_lengths,
                            max_tokens=max_tokens,
                            next_tokens=first_tokens,
                            temperature=temperature,
                            row_max_tokens=per_row_limits,
                        )
                    else:
                        yield from self._decode_shared_prefix_prompt_list_state_steps(
                            bucket_states,
                            prompt_count=len(prompts),
                            prompt_lengths=prompt_lengths,
                            max_tokens=max_tokens,
                            next_tokens=first_tokens,
                            temperature=temperature,
                            row_max_tokens=per_row_limits,
                        )
                return

        states: list[dict[str, object]] = []
        self._add_stream_group_profile_extra(
            prompt_list_path="length_groups",
            length_group_count=len(length_groups),
        )
        first_tokens: list[int | None] = [None for _ in prompts]
        prefill_profile_start_s = self._stream_group_profile_start_s()
        for slot, same_length in enumerate(length_groups):
            state = self._prefill_shared_prefix_prompt_list_suffix_bucket(
                same_length,
                prefix_cache=prefix_cache,
                prefix_tokens=prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
                model=model,
                slot=slot,
            )
            if state is None:
                continue
            original_indices = state["indices"]
            next_token = state["next_token"]
            active = state["active"]
            if (
                not isinstance(original_indices, list)
                or not isinstance(next_token, Tensor)
                or not isinstance(active, list)
            ):
                raise RuntimeError("invalid shared-prefix prompt-list state")
            next_token = next_token.to(self.device)
            for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                token_id = int(token_id)
                first_tokens[original_indices[offset]] = token_id
            states.append(state)
        self._add_stream_group_profile_elapsed(
            "shared_prefix_length_group_prefill_ms",
            prefill_profile_start_s,
        )
        yield first_tokens
        for state in states:
            cache = state["cache"]
            original_indices = state["indices"]
            if isinstance(original_indices, list):
                self._save_prompt_prefix_cache_rows(prompts, cache, row_indices=original_indices)
        should_continue = max_tokens > 1 and any(
            isinstance(state["active"], list) and any(state["active"])
            for state in states
        )
        should_continue = _sync_tensor_parallel_continue(model, should_continue, self.device)
        if not should_continue:
            return

        if _shared_prefix_ragged_cache_enabled_for_model(model) and not prefer_dense_group_decode:
            ragged_cache_profile_start_s = self._stream_group_profile_start_s()
            combined_cache = self._shared_prefix_prompt_list_ragged_cache(
                states,
                prompt_lengths=prompt_lengths,
                prompt_count=len(prompts),
                max_tokens=max_tokens,
                model=model,
            )
            self._add_stream_group_profile_elapsed(
                "shared_prefix_ragged_cache_ms",
                ragged_cache_profile_start_s,
            )
            use_combined_cache = combined_cache is not None
            if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
                use_combined_cache = _tensor_parallel_all_ranks_true(
                    model,
                    use_combined_cache,
                    self.device,
                )
            if use_combined_cache and combined_cache is not None:
                yield from self._decode_shared_prefix_prompt_list_ragged(
                    cache=combined_cache,
                    active=[
                        token_id is not None and int(token_id) not in stop_token_ids
                        for token_id in first_tokens
                    ],
                    prompt_lengths=prompt_lengths,
                    max_tokens=max_tokens,
                    next_tokens=first_tokens,
                    temperature=temperature,
                    row_max_tokens=per_row_limits,
                )
                return

        yield from self._decode_shared_prefix_prompt_list_state_steps(
            states,
            prompt_count=len(prompts),
            prompt_lengths=prompt_lengths,
            max_tokens=max_tokens,
            next_tokens=first_tokens,
            temperature=temperature,
            row_max_tokens=per_row_limits,
        )

    def _prefill_shared_prefix_prompt_list_suffix_bucket(
        self,
        bucket: Sequence[tuple[int, list[int]]],
        *,
        prefix_cache: object,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int],
        model: object,
        slot: int,
    ) -> dict[str, object] | None:
        if not bucket:
            return None
        original_indices = [index for index, _prompt in bucket]
        prompt_rows = [prompt for _index, prompt in bucket]
        suffix_rows = [prompt[prefix_tokens:] for prompt in prompt_rows]
        suffix_lengths = [len(row) for row in suffix_rows]
        max_suffix_len = max(suffix_lengths, default=0)
        if max_suffix_len <= 0:
            return None
        cache = self._generation_microbatch_cache(
            slot,
            len(prompt_rows),
            prefix_tokens + max_suffix_len + max_tokens,
            model=model,
        )
        _copy_generation_cache_first_row(prefix_cache, cache, len(prompt_rows))
        if min(suffix_lengths) == max_suffix_len:
            suffix_ids = torch.tensor(suffix_rows, dtype=torch.long, device=self.device)
            next_token, cache = _prefill_next_token(
                model,
                suffix_ids,
                cache,
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    suffix_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            ragged_decode = False
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = _padded_token_rows_tensor(suffix_rows, self.device, pad_token_id)
            next_token, cache = _prefill_padded_suffix_next_token(
                model,
                suffix_ids,
                cache,
                suffix_lengths,
                temperature,
                allow_capture=self._batched_prefill_graph_capture_enabled(
                    model,
                    suffix_ids,
                    cache,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    selected_logits=True,
                ),
            )
            ragged_decode = True
        next_token = next_token.to(self.device)
        active: list[bool] = []
        stop_token_ids = self.stop_token_ids
        for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
            token = int(token_id)
            original_index = original_indices[offset]
            active.append(token not in stop_token_ids and row_max_tokens[original_index] > 1)
        return {
            "indices": original_indices,
            "active": active,
            "cache": cache,
            "next_token": next_token,
            "ragged_decode": ragged_decode,
        }

    def _decode_shared_prefix_prompt_list_state_steps(
        self,
        states: Sequence[dict[str, object]],
        *,
        prompt_count: int,
        prompt_lengths: Sequence[int],
        max_tokens: int,
        next_tokens: Sequence[int | None],
        temperature: float,
        row_max_tokens: Sequence[int],
    ) -> Iterator[list[int | None]]:
        segments: list[tuple[Sequence[int], Iterator[list[int | None]]]] = []
        for state in states:
            original_indices = state["indices"]
            active = state["active"]
            if not isinstance(original_indices, list) or not isinstance(active, list):
                raise RuntimeError("invalid shared-prefix prompt-list state")
            if bool(state.get("ragged_decode")):
                segments.append(
                    (
                        original_indices,
                        self._decode_shared_prefix_prompt_list_ragged(
                            cache=state["cache"],
                            active=list(active),
                            prompt_lengths=[prompt_lengths[int(index)] for index in original_indices],
                            max_tokens=max_tokens,
                            next_tokens=[next_tokens[int(index)] for index in original_indices],
                            temperature=temperature,
                            row_max_tokens=[row_max_tokens[int(index)] for index in original_indices],
                        ),
                    )
                )
            else:
                segments.append(
                    (
                        original_indices,
                        self._decode_shared_prefix_prompt_list_dense_state_steps(
                            state,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            row_max_tokens=[row_max_tokens[int(index)] for index in original_indices],
                        ),
                    )
                )
        yield from _interleave_prompt_segments(prompt_count, segments)

    def _decode_shared_prefix_prompt_list_dense_state_steps(
        self,
        state: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int],
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        active = state["active"]
        if not isinstance(active, list):
            raise RuntimeError("invalid shared-prefix prompt-list active state")
        for step in range(1, max_tokens):
            local_should_decode = any(active)
            should_decode = _sync_tensor_parallel_continue(model, local_should_decode, self.device)
            if not should_decode:
                break
            next_token = state["next_token"]
            cache = state["cache"]
            if not isinstance(next_token, Tensor):
                raise RuntimeError("invalid shared-prefix prompt-list token state")
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)
            state["cache"] = cache
            state["next_token"] = next_token
            step_tokens: list[int | None] = []
            emitted = False
            for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                if not active[offset]:
                    step_tokens.append(None)
                    continue
                token = int(token_id)
                step_tokens.append(token)
                if token in stop_token_ids or step + 1 >= row_max_tokens[offset]:
                    active[offset] = False
                emitted = True
            if not emitted:
                break
            yield step_tokens

    def _prefill_shared_prefix_prompt_list_padded_suffix_rows(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        prefix_cache: object,
        target_cache: object,
        target_rows: Sequence[int],
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int],
        model: object,
    ) -> tuple[list[int | None], list[bool]] | None:
        prompt_count = len(prompts)
        if prompt_count <= 0 or len(target_rows) != prompt_count:
            return None
        if len(row_max_tokens) != prompt_count:
            return None
        suffix_rows = [list(prompt[prefix_tokens:]) for prompt in prompts]
        if any(not suffix for suffix in suffix_rows):
            return None
        prefix_seq_len = _cache_row_seq_len(prefix_cache, 0)
        if prefix_tokens <= 0 or prefix_seq_len < prefix_tokens:
            return None
        copy_prefix = getattr(target_cache, "copy_prefix_from", None)
        for_rows = getattr(target_cache, "for_rows", None)
        if not callable(copy_prefix) or not callable(for_rows):
            return None
        try:
            if not _copy_generation_cache_first_row_to_rows(prefix_cache, target_cache, target_rows, prefix_tokens):
                for target_row in target_rows:
                    copy_prefix(prefix_cache, prefix_tokens, source_row=0, dest_row=int(target_row))
            cache_view = for_rows(tuple(int(row) for row in target_rows))
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_padded_suffix_row_cache", exc)
            return None

        suffix_lengths = [len(suffix) for suffix in suffix_rows]
        pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
        suffix_ids = _padded_token_rows_tensor(suffix_rows, self.device, pad_token_id)
        next_token, _cache_view = _prefill_padded_suffix_next_token(
            model,
            suffix_ids,
            cache_view,
            suffix_lengths,
            temperature,
            allow_capture=self._batched_prefill_graph_capture_enabled(
                model,
                suffix_ids,
                cache_view,
                temperature=temperature,
                max_tokens=max_tokens,
                selected_logits=True,
            ),
        )
        try:
            _set_generation_cache_rows_seq_lens(
                cache_view,
                range(prompt_count),
                [prefix_tokens + length for length in suffix_lengths],
            )
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_padded_suffix_row_lengths", exc)
            return None
        next_token = next_token.to(self.device)
        stop_token_ids = self.stop_token_ids
        first_tokens: list[int | None] = []
        active: list[bool] = []
        for row, token_id in enumerate(next_token[:prompt_count].detach().cpu().tolist()):
            if row_max_tokens[row] <= 0:
                first_tokens.append(None)
                active.append(False)
                continue
            token = int(token_id)
            first_tokens.append(token)
            active.append(token not in stop_token_ids and row_max_tokens[row] > 1)
        return first_tokens, active

    def _prefill_persistent_prompt_list_payload_groups(
        self,
        payload: Mapping[str, object],
        *,
        target_cache: object,
        prefix_caches: Mapping[tuple[int, ...], object],
        model: object,
        temperature: float,
    ) -> dict[str, tuple[int | None, bool]]:
        prefill_items = payload.get("prefill", [])
        prefill_groups = payload.get("prefill_groups", [])
        if not isinstance(prefill_items, list) or not isinstance(prefill_groups, list):
            raise ValueError("persistent prompt-list payload requires prefill lists")
        item_by_id: dict[str, Mapping[str, object]] = {}
        for item in prefill_items:
            if not isinstance(item, Mapping):
                raise ValueError("persistent prompt-list prefill entries must be mappings")
            request_id = item.get("request_id")
            if request_id is None:
                raise ValueError("persistent prompt-list prefill entry requires request_id")
            item_by_id[str(request_id)] = item

        results: dict[str, tuple[int | None, bool]] = {}
        for group in prefill_groups:
            if not isinstance(group, Mapping):
                raise ValueError("persistent prompt-list prefill groups must be mappings")
            request_id_values = group.get("request_ids", [])
            if not isinstance(request_id_values, list):
                raise ValueError("persistent prompt-list prefill group request_ids must be a list")
            request_ids = [str(request_id) for request_id in request_id_values]
            prefix_values = group.get("prefix", [])
            if not isinstance(prefix_values, list):
                raise ValueError("persistent prompt-list prefill group prefix must be a list")
            prefix = tuple(int(token_id) for token_id in prefix_values)
            prefix_cache = prefix_caches.get(prefix)
            if prefix_cache is None:
                raise RuntimeError("missing persistent prompt-list prefix cache")
            prefix_tokens = int(group.get("prefix_hit_tokens", len(prefix)))
            prompts: list[list[int]] = []
            target_rows: list[int] = []
            row_max_tokens: list[int] = []
            for request_id in request_ids:
                item = item_by_id.get(request_id)
                if item is None:
                    raise ValueError(f"missing prefill entry for request {request_id}")
                prompt = item.get("prompt", [])
                if not isinstance(prompt, list):
                    raise ValueError("persistent prompt-list prefill prompt must be a list")
                prompts.append([int(token_id) for token_id in prompt])
                target_rows.append(int(item["row"]))
                row_max_tokens.append(int(item["max_tokens"]))
            prefill_result = self._prefill_shared_prefix_prompt_list_padded_suffix_rows(
                prompts,
                prefix_cache=prefix_cache,
                target_cache=target_cache,
                target_rows=target_rows,
                prefix_tokens=prefix_tokens,
                max_tokens=max(row_max_tokens, default=0),
                temperature=temperature,
                row_max_tokens=row_max_tokens,
                model=model,
            )
            if prefill_result is None:
                raise RuntimeError("persistent prompt-list row-targeted prefill failed")
            first_tokens, active_flags = prefill_result
            for request_id, token_id, active in zip(request_ids, first_tokens, active_flags):
                results[request_id] = (token_id, active)
        return results

    def _execute_persistent_prompt_list_prefill_item(
        self,
        item: Mapping[str, object],
        state: _PersistentPromptListStepState,
        *,
        temperature: float,
    ) -> tuple[int | None, bool]:
        request_id = item.get("request_id")
        if request_id is None:
            raise ValueError("persistent prompt-list prefill entry requires request_id")
        row = int(item["row"])
        if row < 0 or row >= state.cache_batch_size:
            raise ValueError("persistent prompt-list prefill row is out of range")
        prompt_obj = item.get("prompt", [])
        if not isinstance(prompt_obj, list):
            raise ValueError("persistent prompt-list prefill prompt must be a list")
        prompt = [int(token_id) for token_id in prompt_obj]
        start_token = int(item.get("start_token", item.get("prefix_hit_tokens", 0)))
        prefill_tokens = int(item.get("prefill_tokens", max(1, len(prompt) - start_token)))
        if start_token < 0 or prefill_tokens < 1 or start_token + prefill_tokens > len(prompt):
            raise ValueError("persistent prompt-list prefill chunk is outside the prompt")
        request_id_text = str(request_id)
        current_request_id = state.row_request_ids[row]
        if current_request_id is None:
            if start_token > 0:
                prefix_values = item.get("prefix", prompt[:start_token])
                if not isinstance(prefix_values, list):
                    raise ValueError("persistent prompt-list prefill prefix must be a list")
                prefix = tuple(int(token_id) for token_id in prefix_values)
                if len(prefix) != start_token:
                    raise ValueError("persistent prompt-list prefill prefix length mismatch")
                prefix_cache = state.prefix_caches.get(prefix)
                if prefix_cache is None:
                    raise RuntimeError("missing persistent prompt-list prefix cache")
                _copy_generation_cache_row(
                    prefix_cache,
                    state.cache,
                    source_row=0,
                    target_row=row,
                    seq_len=start_token,
                )
            state.row_request_ids[row] = request_id_text
            state.active[row] = False
            state.per_row_limits[row] = int(item.get("max_tokens", 0))
            state.generated_tokens[row] = 0
            state.seq_lens[row] = start_token
            state.next_token_tensor[row] = 0
        elif current_request_id != request_id_text:
            raise ValueError("persistent prompt-list prefill row is occupied by another request")
        if int(state.seq_lens[row].item()) != start_token:
            raise ValueError("persistent prompt-list prefill start_token does not match row state")

        prompt_chunk = prompt[start_token : start_token + prefill_tokens]
        cache_view = _cache_row_slice(state.cache, row, row + 1)
        if cache_view is None:
            raise RuntimeError("persistent prompt-list prefill requires row-view cache")
        input_ids = torch.tensor([prompt_chunk], dtype=torch.long, device=self.device)
        logits, _cache_view = _forward(self.model, input_ids, cache_view)
        state.seq_lens[row] = start_token + prefill_tokens
        prompt_complete = bool(item.get("prompt_complete", True))
        emits_token = bool(item.get("emits_token", prompt_complete))
        if not prompt_complete or not emits_token:
            return None, False

        next_token = _sample(self.model, logits[:, -1, :], temperature).to(self.device)
        token_id = int(next_token.item())
        state.generated_tokens[row] = 1
        state.next_token_tensor[row] = token_id
        state.per_row_limits[row] = int(item.get("max_tokens", state.per_row_limits[row]))
        active = token_id not in self.stop_token_ids and state.per_row_limits[row] > 1
        state.active[row] = active
        if not active:
            state.row_request_ids[row] = None
        return token_id, active

    def _execute_persistent_prompt_list_step_payload(
        self,
        payload: Mapping[str, object],
        state: _PersistentPromptListStepState,
        *,
        temperature: float,
        static_graph_buckets: bool = False,
    ) -> _PersistentPromptListStepResult:
        decode_rows_obj = payload.get("decode_rows", [])
        decode_ids_obj = payload.get("decode_request_ids", [])
        if not isinstance(decode_rows_obj, list) or not isinstance(decode_ids_obj, list):
            raise ValueError("persistent prompt-list decode rows and ids must be lists")
        decode_rows = [int(row) for row in decode_rows_obj]
        decode_request_ids = [str(request_id) for request_id in decode_ids_obj]
        if len(decode_rows) != len(decode_request_ids):
            raise ValueError("persistent prompt-list decode rows and ids length mismatch")
        if decode_rows:
            active_rows = [row for row, is_active in enumerate(state.active) if is_active]
            if decode_rows != active_rows:
                raise ValueError("persistent prompt-list decode rows do not match active state")

        decode_tokens: dict[str, int | None] = {}
        finished_request_ids: list[str] = []
        if decode_rows:
            _set_shared_prefix_ragged_static_graph_bucket_mode(
                self.model,
                state.cache,
                static_graph_buckets=static_graph_buckets,
            )
            self._ensure_persistent_prompt_list_ephemeral_graph_scope(
                state,
                step=int(payload.get("step", 0)),
            )
            step_result = self._decode_shared_prefix_prompt_list_ragged_step(
                cache=state.cache,
                active=state.active,
                per_row_limits=state.per_row_limits,
                generated_tokens=state.generated_tokens,
                seq_lens=state.seq_lens,
                next_token_tensor=state.next_token_tensor,
                step=int(payload.get("step", 0)),
                cache_batch_size=(
                    state.cache_batch_size
                    if static_graph_buckets
                    else max(
                        max(decode_rows, default=-1) + 1,
                        state.logical_cache_batch_size,
                    )
                ),
                temperature=temperature,
                static_graph_buckets=static_graph_buckets,
            )
            if step_result is not None:
                state.cache, step_tokens = step_result
                for request_id, row in zip(decode_request_ids, decode_rows):
                    token = step_tokens[row] if row < len(step_tokens) else None
                    decode_tokens[request_id] = token
                    if row < len(state.active) and not state.active[row]:
                        state.row_request_ids[row] = None
                        finished_request_ids.append(request_id)

        prefill_tokens: dict[str, int | None] = {}
        prefill_results = self._prefill_persistent_prompt_list_payload_groups(
            payload,
            target_cache=state.cache,
            prefix_caches=state.prefix_caches,
            model=self.model,
            temperature=temperature,
        )
        prefill_items = payload.get("prefill", [])
        if not isinstance(prefill_items, list):
            raise ValueError("persistent prompt-list prefill entries must be a list")
        for item in prefill_items:
            if not isinstance(item, Mapping):
                raise ValueError("persistent prompt-list prefill entry must be a mapping")
            request_id = item.get("request_id")
            if request_id is None:
                raise ValueError("persistent prompt-list prefill entry requires request_id")
            request_id_text = str(request_id)
            row = int(item["row"])
            if row < 0 or row >= len(state.active):
                raise ValueError("persistent prompt-list prefill row is out of range")
            state.logical_cache_batch_size = max(state.logical_cache_batch_size, row + 1)
            prompt = item.get("prompt", [])
            if not isinstance(prompt, list):
                raise ValueError("persistent prompt-list prefill prompt must be a list")
            if request_id_text in prefill_results:
                token_id, active = prefill_results[request_id_text]
            else:
                token_id, active = self._execute_persistent_prompt_list_prefill_item(
                    item,
                    state,
                    temperature=temperature,
                )
            prefill_tokens[request_id_text] = token_id
            state.per_row_limits[row] = int(item["max_tokens"])
            state.generated_tokens[row] = 0 if token_id is None else 1
            if token_id is not None:
                state.seq_lens[row] = len(prompt)
            state.next_token_tensor[row] = 0 if token_id is None else int(token_id)
            if token_id is not None:
                state.active[row] = bool(active)
                state.row_request_ids[row] = request_id_text if active else None
            if token_id is not None and not active:
                finished_request_ids.append(request_id_text)

        explicit_finished = payload.get("finished_after_prefill", [])
        if isinstance(explicit_finished, list):
            for request_id in explicit_finished:
                request_id_text = str(request_id)
                if request_id_text not in finished_request_ids:
                    finished_request_ids.append(request_id_text)
        return _PersistentPromptListStepResult(
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            finished_request_ids=tuple(finished_request_ids),
        )

    def _handle_persistent_prompt_list_step_payload(
        self,
        payload: Mapping[str, object],
    ) -> _PersistentPromptListStepResult:
        state = self._persistent_prompt_list_step_state
        if state is None:
            raise RuntimeError("persistent prompt-list step state is not installed")
        result = self._execute_persistent_prompt_list_step_payload(
            payload,
            state,
            temperature=float(payload.get("temperature", 0.0)),
            static_graph_buckets=bool(payload.get("static_graph_buckets", False)),
        )
        self._persistent_prompt_list_step_last_result = result
        return result

    def _execute_persistent_prompt_list_decode_run_payload(
        self,
        payload: Mapping[str, object],
        state: _PersistentPromptListStepState,
        *,
        temperature: float,
        static_graph_buckets: bool = False,
    ) -> _PersistentPromptListDecodeRunResult:
        step_count = int(payload.get("step_count", 0))
        if step_count < 1:
            raise ValueError("persistent prompt-list decode-run requires positive step_count")
        start_step = int(payload.get("start_step", 0))
        results: list[_PersistentPromptListStepResult] = []
        for offset in range(step_count):
            active_rows = [row for row, is_active in enumerate(state.active) if is_active]
            if not active_rows:
                break
            decode_request_ids: list[str] = []
            for row in active_rows:
                request_id = state.row_request_ids[row]
                if request_id is None:
                    raise RuntimeError("persistent prompt-list active row is missing request id")
                decode_request_ids.append(request_id)
            step_payload = {
                "op": "persistent_prompt_list_step",
                "step": start_step + offset,
                "decode_request_ids": decode_request_ids,
                "decode_rows": active_rows,
                "prefill": [],
                "prefill_groups": [],
                "finished_after_prefill": [],
            }
            results.append(
                self._execute_persistent_prompt_list_step_payload(
                    step_payload,
                    state,
                    temperature=temperature,
                    static_graph_buckets=static_graph_buckets,
                )
            )
        return _PersistentPromptListDecodeRunResult(step_results=tuple(results))

    def _handle_persistent_prompt_list_decode_run_payload(
        self,
        payload: Mapping[str, object],
    ) -> _PersistentPromptListDecodeRunResult:
        state = self._persistent_prompt_list_step_state
        if state is None:
            raise RuntimeError("persistent prompt-list decode-run state is not installed")
        result = self._execute_persistent_prompt_list_decode_run_payload(
            payload,
            state,
            temperature=float(payload.get("temperature", 0.0)),
            static_graph_buckets=bool(payload.get("static_graph_buckets", False)),
        )
        if result.step_results:
            self._persistent_prompt_list_step_last_result = result.step_results[-1]
        return result

    def _start_persistent_prompt_list_step_state(
        self,
        *,
        prefix: Sequence[int],
        cache_batch_size: int,
        max_seq_len: int,
        temperature: float,
        max_tokens: int | None = None,
    ) -> _PersistentPromptListStepState:
        prefix_tokens = tuple(int(token_id) for token_id in prefix)
        if not prefix_tokens:
            raise ValueError("persistent prompt-list state requires a non-empty prefix")
        if cache_batch_size < 1:
            raise ValueError("persistent prompt-list cache_batch_size must be positive")
        if max_seq_len < len(prefix_tokens):
            raise ValueError("persistent prompt-list max_seq_len must cover the prefix")
        model = self.model
        prefix_cache = self._generation_cache(
            1,
            len(prefix_tokens),
            model=model,
            pool=False,
        )
        prefix_ids = torch.tensor([list(prefix_tokens)], dtype=torch.long, device=self.device)
        prefix_cache = _prefill_cache_only(
            model,
            prefix_ids,
            prefix_cache,
            allow_capture=_runtime_prefill_graph_capture_enabled(
                model,
                temperature,
                max_tokens=max_seq_len - len(prefix_tokens),
            ),
        )
        _mark_generation_cache_prefix(prefix_cache, prefix_tokens)
        cache = self._generation_cache(
            cache_batch_size,
            max_seq_len,
            model=model,
            pool=False,
            batch_capacity=cache_batch_size,
        )
        ragged_graph_tokens = max_tokens if max_tokens is not None else max_seq_len - len(prefix_tokens)
        if _disable_tp_shared_prefix_ragged_decode_graph(model, max_tokens=ragged_graph_tokens):
            _set_ragged_decode_graph_disabled(cache, True)
        _set_runtime_ragged_decode_graph_capture(
            cache,
            _runtime_ragged_decode_graph_capture_allowed_for_request(
                model,
                max_tokens=ragged_graph_tokens,
            ),
        )
        state = _PersistentPromptListStepState(
            cache=cache,
            prefix_caches={prefix_tokens: prefix_cache},
            active=[False for _ in range(cache_batch_size)],
            per_row_limits=[0 for _ in range(cache_batch_size)],
            generated_tokens=[0 for _ in range(cache_batch_size)],
            seq_lens=torch.zeros(cache_batch_size, dtype=torch.long, device=self.device),
            next_token_tensor=torch.zeros(cache_batch_size, dtype=torch.long, device=self.device),
            row_request_ids=[None for _ in range(cache_batch_size)],
            cache_batch_size=cache_batch_size,
            ephemeral_graph_allowed=(
                getattr(cache, "_torchinferno_ephemeral_cache", False)
                and env_flag("TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH", True)
            ),
        )
        self._persistent_prompt_list_step_state = state
        self._persistent_prompt_list_step_last_result = None
        return state

    def _close_persistent_prompt_list_step_state(self) -> None:
        state = self._persistent_prompt_list_step_state
        if state is not None and state.ephemeral_graph_scope:
            try:
                setattr(state.cache, "_torchinferno_ephemeral_ragged_graph_scope", False)
            except Exception:
                pass
            _release_decode_graphs_for_cache(self.model, state.cache)
        self._persistent_prompt_list_step_state = None
        self._persistent_prompt_list_step_last_result = None

    def _ensure_persistent_prompt_list_ephemeral_graph_scope(
        self,
        state: _PersistentPromptListStepState,
        *,
        step: int,
    ) -> None:
        if state.ephemeral_graph_scope or not state.ephemeral_graph_allowed:
            return
        min_step = env_int(
            "TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH_MIN_STEP",
            1,
            minimum=1,
        )
        if step < min_step:
            return
        try:
            setattr(state.cache, "_torchinferno_ephemeral_ragged_graph_scope", True)
            state.ephemeral_graph_scope = True
        except Exception:
            state.ephemeral_graph_allowed = False

    def _run_persistent_prompt_list_group_local(
        self,
        group: Sequence[_QueuedGeneration],
        *,
        max_active_rows: int,
        prefill_token_budget: int | None = None,
        prefix_tokens: int = 0,
        decode_run_steps: int = 1,
        static_graph_buckets: bool = False,
        broadcast_tensor_parallel: bool = False,
        sync_tensor_parallel: bool = False,
        max_scheduler_steps: int | None = None,
    ) -> _PersistentPromptListLocalRunStats:
        if not group:
            return _PersistentPromptListLocalRunStats(
                scheduler_steps=0,
                step_commands=0,
                decode_run_commands=0,
                empty_plans=0,
                decode_steps=0,
                max_decode_run_steps=0,
                prefill_admissions=0,
                emitted_tokens=0,
                finished_events=0,
                first_emit_s=None,
                closed=True,
            )
        sync_tensor_parallel = bool(sync_tensor_parallel or broadcast_tensor_parallel)
        if prefix_tokens <= 0:
            raise ValueError("persistent prompt-list local runner requires a shared prefix")
        prompts = [request.prompt for request in group]
        prefix = tuple(int(token_id) for token_id in prompts[0][:prefix_tokens])
        if len(prefix) != prefix_tokens or any(tuple(prompt[:prefix_tokens]) != prefix for prompt in prompts):
            raise ValueError("persistent prompt-list local runner requires a common shared prefix")
        scheduler = _persistent_prompt_list_scheduler_for_group(
            group,
            max_active_rows=max_active_rows,
            prefill_token_budget=prefill_token_budget,
            prefix_tokens=prefix_tokens,
        )
        max_seq_len = max((len(request.prompt) + request.max_tokens for request in group), default=prefix_tokens)
        max_tokens = max((request.max_tokens for request in group), default=0)
        started = False
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_persistent_prompt_list_start(
                self.model,
                prefix=prefix,
                cache_batch_size=max_active_rows,
                max_seq_len=max_seq_len,
                temperature=group[0].temperature,
                max_tokens=max_tokens,
            )
            started = True
        self._start_persistent_prompt_list_step_state(
            prefix=prefix,
            cache_batch_size=max_active_rows,
            max_seq_len=max_seq_len,
            temperature=group[0].temperature,
            max_tokens=max_tokens,
        )
        if sync_tensor_parallel:
            _sync_tensor_parallel_command(self.model, self.device)
        finished_request_ids: tuple[str, ...] = ()
        scheduler_steps = 0
        step_commands = 0
        decode_run_commands = 0
        empty_plans = 0
        decode_steps = 0
        max_decode_run_size = 0
        prefill_admissions = 0
        emitted_tokens = 0
        finished_events = 0
        first_emit_s: float | None = None
        stream_rows = _StreamRowState()
        try:
            while scheduler.has_work() or finished_request_ids:
                plan = scheduler.step(finished_request_ids=finished_request_ids)
                scheduler_steps += 1
                finished_request_ids = ()
                if max_scheduler_steps is not None and scheduler_steps > max_scheduler_steps:
                    raise RuntimeError("persistent prompt-list local runner exceeded max_scheduler_steps")
                if not plan.decode_request_ids and not plan.prefill_admissions:
                    empty_plans += 1
                    continue
                if decode_run_steps > 1 and plan.decode_request_ids and not plan.prefill_admissions:
                    payload = _persistent_prompt_list_decode_run_payload(
                        start_step=plan.step,
                        step_count=decode_run_steps,
                        temperature=group[0].temperature,
                        static_graph_buckets=static_graph_buckets,
                    )
                    if broadcast_tensor_parallel:
                        _broadcast_tensor_parallel_persistent_prompt_list_decode_run(self.model, payload)
                    result = self._handle_persistent_prompt_list_decode_run_payload(payload)
                    if sync_tensor_parallel:
                        _sync_tensor_parallel_command(self.model, self.device)
                    emitted_finished_request_ids = self._stream_persistent_prompt_batch_decode_run_result(
                        group,
                        result,
                        stream_rows=stream_rows,
                    )
                    decode_run_commands += 1
                    max_decode_run_size = max(max_decode_run_size, len(result.step_results))
                    decode_steps += len(result.step_results)
                    new_tokens = sum(
                        token is not None
                        for step_result in result.step_results
                        for token in (*step_result.decode_tokens.values(), *step_result.prefill_tokens.values())
                    )
                    if new_tokens and first_emit_s is None:
                        first_emit_s = time.perf_counter()
                    emitted_tokens += new_tokens
                    finished_request_ids = _merge_token_budget_finished_ids(
                        result.finished_request_ids,
                        emitted_finished_request_ids,
                    )
                    finished_events += len(finished_request_ids)
                    continue
                payload = _persistent_prompt_list_step_payload(plan, group)
                payload["temperature"] = float(group[0].temperature)
                if static_graph_buckets:
                    payload["static_graph_buckets"] = True
                if broadcast_tensor_parallel:
                    _broadcast_tensor_parallel_persistent_prompt_list_step(self.model, payload)
                result = self._handle_persistent_prompt_list_step_payload(payload)
                if sync_tensor_parallel:
                    _sync_tensor_parallel_command(self.model, self.device)
                emitted_finished_request_ids = self._stream_persistent_prompt_batch_step_result(
                    group,
                    result,
                    payload=payload,
                    stream_rows=stream_rows,
                )
                step_commands += 1
                if plan.decode_request_ids:
                    decode_steps += 1
                prefill_admissions += len(plan.prefill_admissions)
                new_tokens = sum(token is not None for token in result.decode_tokens.values())
                new_tokens += sum(token is not None for token in result.prefill_tokens.values())
                if new_tokens and first_emit_s is None:
                    first_emit_s = time.perf_counter()
                emitted_tokens += new_tokens
                finished_request_ids = _merge_token_budget_finished_ids(
                    result.finished_request_ids,
                    emitted_finished_request_ids,
                )
                finished_events += len(finished_request_ids)
        finally:
            if broadcast_tensor_parallel and started:
                _broadcast_tensor_parallel_persistent_prompt_list_close(self.model)
                if sync_tensor_parallel:
                    _sync_tensor_parallel_command(self.model, self.device)
            self._close_persistent_prompt_list_step_state()
        return _PersistentPromptListLocalRunStats(
            scheduler_steps=scheduler_steps,
            step_commands=step_commands,
            decode_run_commands=decode_run_commands,
            empty_plans=empty_plans,
            decode_steps=decode_steps,
            max_decode_run_steps=max_decode_run_size,
            prefill_admissions=prefill_admissions,
            emitted_tokens=emitted_tokens,
            finished_events=finished_events,
            first_emit_s=first_emit_s,
            closed=self._persistent_prompt_list_step_state is None,
        )

    def _run_token_budget_prompt_list_group_local(
        self,
        group: Sequence[_QueuedGeneration],
        *,
        max_active_rows: int,
        max_scheduled_tokens: int,
        prefill_chunk_size: int | None = None,
        prefix_tokens: int = 0,
        decode_run_steps: int = 1,
        static_graph_buckets: bool = False,
        broadcast_tensor_parallel: bool = False,
        sync_tensor_parallel: bool = False,
        arrival_steps: Sequence[int] | None = None,
        max_scheduler_steps: int | None = None,
    ) -> _PersistentPromptListLocalRunStats:
        if not group:
            return _PersistentPromptListLocalRunStats(
                scheduler_steps=0,
                step_commands=0,
                decode_run_commands=0,
                empty_plans=0,
                decode_steps=0,
                max_decode_run_steps=0,
                prefill_admissions=0,
                emitted_tokens=0,
                finished_events=0,
                first_emit_s=None,
                closed=True,
            )
        if prefix_tokens <= 0:
            raise ValueError("token-budget prompt-list local runner requires a shared prefix")
        sync_tensor_parallel = bool(sync_tensor_parallel or broadcast_tensor_parallel)
        request_by_id = {str(index): request for index, request in enumerate(group)}
        prompts = [request.prompt for request in group]
        prefix = tuple(int(token_id) for token_id in prompts[0][:prefix_tokens])
        if len(prefix) != prefix_tokens or any(tuple(prompt[:prefix_tokens]) != prefix for prompt in prompts):
            raise ValueError("token-budget prompt-list local runner requires a common shared prefix")
        scheduler = _token_budget_scheduler_for_group(
            group,
            max_active_rows=max_active_rows,
            max_scheduled_tokens=max_scheduled_tokens,
            prefill_chunk_size=prefill_chunk_size,
            prefix_tokens=prefix_tokens,
            arrival_steps=arrival_steps,
        )
        max_seq_len = max((len(request.prompt) + request.max_tokens for request in group), default=prefix_tokens)
        max_tokens = max((request.max_tokens for request in group), default=0)
        started = False
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_persistent_prompt_list_start(
                self.model,
                prefix=prefix,
                cache_batch_size=max_active_rows,
                max_seq_len=max_seq_len,
                temperature=group[0].temperature,
                max_tokens=max_tokens,
            )
            started = True
        self._start_persistent_prompt_list_step_state(
            prefix=prefix,
            cache_batch_size=max_active_rows,
            max_seq_len=max_seq_len,
            temperature=group[0].temperature,
            max_tokens=max_tokens,
        )
        if sync_tensor_parallel:
            _sync_tensor_parallel_command(self.model, self.device)
        finished_request_ids: tuple[str, ...] = ()
        pending_plan: _TokenBudgetPlan | None = None
        scheduler_steps = 0
        step_commands = 0
        decode_run_commands = 0
        empty_plans = 0
        decode_steps = 0
        max_decode_run_size = 0
        prefill_admissions = 0
        emitted_tokens = 0
        finished_events = 0
        first_emit_s: float | None = None
        stream_rows = _StreamRowState()
        try:
            while scheduler.has_work() or finished_request_ids or pending_plan is not None:
                if pending_plan is None:
                    plan = scheduler.step(finished_request_ids=finished_request_ids)
                    scheduler_steps += 1
                else:
                    plan = pending_plan
                    pending_plan = None
                finished_request_ids = ()
                if max_scheduler_steps is not None and scheduler_steps > max_scheduler_steps:
                    raise RuntimeError("token-budget prompt-list local runner exceeded max_scheduler_steps")
                if not plan.chunks:
                    empty_plans += 1
                    continue
                if decode_run_steps > 1 and _token_budget_plan_is_decode_only(plan):
                    plans = [plan]
                    while (
                        len(plans) < decode_run_steps
                        and not plans[-1].finished_request_ids
                        and scheduler.has_work()
                    ):
                        next_plan = scheduler.step(finished_request_ids=())
                        scheduler_steps += 1
                        if max_scheduler_steps is not None and scheduler_steps > max_scheduler_steps:
                            raise RuntimeError("token-budget prompt-list local runner exceeded max_scheduler_steps")
                        if not next_plan.chunks:
                            if next_plan.finished_request_ids:
                                pending_plan = next_plan
                                break
                            empty_plans += 1
                            continue
                        if not _token_budget_plan_is_decode_only(next_plan):
                            pending_plan = next_plan
                            break
                        plans.append(next_plan)
                    payload = _persistent_prompt_list_decode_run_payload(
                        start_step=plans[0].step,
                        step_count=len(plans),
                        temperature=group[0].temperature,
                        static_graph_buckets=static_graph_buckets,
                    )
                    if broadcast_tensor_parallel:
                        _broadcast_tensor_parallel_persistent_prompt_list_decode_run(self.model, payload)
                    result = self._handle_persistent_prompt_list_decode_run_payload(payload)
                    if sync_tensor_parallel:
                        _sync_tensor_parallel_command(self.model, self.device)
                    emitted_finished_request_ids = self._stream_persistent_prompt_batch_decode_run_result(
                        group,
                        result,
                        stream_rows=stream_rows,
                    )
                    decode_run_commands += 1
                    max_decode_run_size = max(max_decode_run_size, len(result.step_results))
                    decode_steps += len(result.step_results)
                    new_tokens = sum(
                        token is not None
                        for step_result in result.step_results
                        for token in (*step_result.decode_tokens.values(), *step_result.prefill_tokens.values())
                    )
                    if new_tokens and first_emit_s is None:
                        first_emit_s = time.perf_counter()
                    emitted_tokens += new_tokens
                    finished_request_ids = _merge_token_budget_finished_ids(
                        result.finished_request_ids,
                        emitted_finished_request_ids,
                    )
                    finished_events += len(finished_request_ids)
                    continue
                payload = _token_budget_prompt_list_step_payload(plan, request_by_id)
                if payload is None:
                    raise RuntimeError("token-budget prompt-list local runner requires prompt-complete prefix prefill chunks")
                payload["temperature"] = float(group[0].temperature)
                if static_graph_buckets:
                    payload["static_graph_buckets"] = True
                if broadcast_tensor_parallel:
                    _broadcast_tensor_parallel_persistent_prompt_list_step(self.model, payload)
                result = self._handle_persistent_prompt_list_step_payload(payload)
                if sync_tensor_parallel:
                    _sync_tensor_parallel_command(self.model, self.device)
                emitted_finished_request_ids = self._stream_persistent_prompt_batch_step_result(
                    group,
                    result,
                    payload=payload,
                    stream_rows=stream_rows,
                )
                step_commands += 1
                if payload.get("decode_request_ids"):
                    decode_steps += 1
                prefill_items = payload.get("prefill", [])
                if isinstance(prefill_items, list):
                    prefill_admissions += len(prefill_items)
                new_tokens = sum(token is not None for token in result.decode_tokens.values())
                new_tokens += sum(token is not None for token in result.prefill_tokens.values())
                if new_tokens and first_emit_s is None:
                    first_emit_s = time.perf_counter()
                emitted_tokens += new_tokens
                finished_request_ids = _merge_token_budget_finished_ids(
                    result.finished_request_ids,
                    emitted_finished_request_ids,
                )
                finished_events += len(finished_request_ids)
        finally:
            if broadcast_tensor_parallel and started:
                _broadcast_tensor_parallel_persistent_prompt_list_close(self.model)
                if sync_tensor_parallel:
                    _sync_tensor_parallel_command(self.model, self.device)
            self._close_persistent_prompt_list_step_state()
        return _PersistentPromptListLocalRunStats(
            scheduler_steps=scheduler_steps,
            step_commands=step_commands,
            decode_run_commands=decode_run_commands,
            empty_plans=empty_plans,
            decode_steps=decode_steps,
            max_decode_run_steps=max_decode_run_size,
            prefill_admissions=prefill_admissions,
            emitted_tokens=emitted_tokens,
            finished_events=finished_events,
            first_emit_s=first_emit_s,
            closed=self._persistent_prompt_list_step_state is None,
        )

    def _handle_token_budget_prompt_list_run_payload(
        self,
        payload: Mapping[str, object],
    ) -> _PersistentPromptListLocalRunStats:
        input_id_lists = payload.get("input_id_lists")
        if not isinstance(input_id_lists, list):
            raise ValueError("token-budget prompt-list run requires input_id_lists")
        max_tokens = int(payload.get("max_tokens", 0))
        row_max_tokens = _coerce_optional_int_sequence(payload.get("row_max_tokens"))
        if row_max_tokens is None:
            row_max_tokens = [max_tokens for _ in input_id_lists]
        if len(row_max_tokens) != len(input_id_lists):
            raise ValueError("token-budget prompt-list run row_max_tokens must match input_id_lists")
        arrival_steps = _coerce_optional_int_sequence(payload.get("arrival_steps"))
        if arrival_steps is None:
            arrival_steps = [0 for _ in input_id_lists]
        if len(arrival_steps) != len(input_id_lists):
            raise ValueError("token-budget prompt-list run arrival_steps must match input_id_lists")
        temperature = float(payload.get("temperature", 0.0))
        responses: list[queue.SimpleQueue[object]] = [queue.SimpleQueue() for _ in input_id_lists]
        group = [
            _QueuedGeneration(
                [int(token_id) for token_id in prompt],
                int(row_max_tokens[index]),
                temperature,
                True,
                responses[index],
            )
            for index, prompt in enumerate(input_id_lists)
        ]
        prefill_chunk_size_value = int(payload.get("prefill_chunk_size", 0) or 0)
        return self._run_token_budget_prompt_list_group_local(
            group,
            max_active_rows=int(payload.get("max_active_rows", max(1, len(group)))),
            max_scheduled_tokens=int(payload.get("max_scheduled_tokens", 8192)),
            prefill_chunk_size=prefill_chunk_size_value if prefill_chunk_size_value > 0 else None,
            prefix_tokens=int(payload.get("prefix_tokens", 0)),
            decode_run_steps=int(payload.get("decode_run_steps", 1)),
            static_graph_buckets=bool(payload.get("static_graph_buckets", False)),
            broadcast_tensor_parallel=False,
            sync_tensor_parallel=False,
            arrival_steps=arrival_steps,
        )

    def _stream_persistent_prompt_batch_step_result(
        self,
        group: Sequence[_QueuedGeneration],
        result: _PersistentPromptListStepResult,
        *,
        payload: Mapping[str, object] | None = None,
        stream_rows: _StreamRowState | None = None,
    ) -> tuple[str, ...]:
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        finished_request_ids: list[str] = []
        payload_rows = _persistent_prompt_list_payload_request_rows(payload)
        for token_map in (result.decode_tokens, result.prefill_tokens):
            for request_id, token_id in token_map.items():
                if token_id is None:
                    continue
                request = group[int(request_id)]
                if request.done:
                    continue
                if stream_rows is not None:
                    row = payload_rows.get(str(request_id))
                    if row is not None:
                        stream_rows.admit(str(request_id), row, request)
                    if stream_rows.emit(str(request_id), int(token_id), stop_token_ids=stop_token_ids):
                        finished_request_ids.append(str(request_id))
                    continue
                if int(token_id) in stop_token_ids:
                    _finish_stream_request(request)
                    finished_request_ids.append(str(request_id))
                    continue
                request.responses.put(int(token_id))
        for request_id in result.finished_request_ids:
            request = group[int(request_id)]
            if stream_rows is not None:
                if stream_rows.finish(str(request_id)):
                    finished_request_ids.append(str(request_id))
                elif not request.done:
                    _finish_stream_request(request)
                    finished_request_ids.append(str(request_id))
            elif not request.done:
                _finish_stream_request(request)
                finished_request_ids.append(str(request_id))
        return _merge_token_budget_finished_ids(finished_request_ids)

    def _stream_persistent_prompt_batch_decode_run_result(
        self,
        group: Sequence[_QueuedGeneration],
        result: _PersistentPromptListDecodeRunResult,
        *,
        stream_rows: _StreamRowState | None = None,
    ) -> tuple[str, ...]:
        finished_request_ids: list[str] = []
        for step_result in result.step_results:
            finished_request_ids.extend(
                self._stream_persistent_prompt_batch_step_result(
                    group,
                    step_result,
                    stream_rows=stream_rows,
                )
            )
        return _merge_token_budget_finished_ids(finished_request_ids)

    def _start_token_budget_step_state(
        self,
        *,
        cache_batch_size: int,
        max_seq_len: int,
        prefix: Sequence[int] = (),
        temperature: float = 0.0,
        external_cache: object | None = None,
    ) -> _TokenBudgetStepState:
        if cache_batch_size < 1:
            raise ValueError("token-budget cache_batch_size must be positive")
        if max_seq_len < 1:
            raise ValueError("token-budget max_seq_len must be positive")
        prefix_tokens = tuple(int(token_id) for token_id in prefix)
        if prefix_tokens and max_seq_len < len(prefix_tokens):
            raise ValueError("token-budget max_seq_len must cover the prefix")
        prefix_caches: dict[tuple[int, ...], object] = {}
        if prefix_tokens:
            prefix_cache = self._generation_cache(
                1,
                len(prefix_tokens),
                model=self.model,
                pool=False,
            )
            prefix_ids = torch.tensor([list(prefix_tokens)], dtype=torch.long, device=self.device)
            prefix_cache = _prefill_cache_only(
                self.model,
                prefix_ids,
                prefix_cache,
                allow_capture=_runtime_prefill_graph_capture_enabled(
                    self.model,
                    temperature,
                    max_tokens=max_seq_len - len(prefix_tokens),
                ),
            )
            _mark_generation_cache_prefix(prefix_cache, prefix_tokens)
            prefix_caches[prefix_tokens] = prefix_cache
        if external_cache is None:
            external_cache = getattr(self, "_persistent_serving_cache", None)
        if external_cache is not None:
            cache = external_cache
            _reset_generation_cache(cache)
        else:
            cache = self._generation_cache(
                cache_batch_size,
                max_seq_len,
                model=self.model,
                batch_capacity=cache_batch_size,
            )
        state = _TokenBudgetStepState(
            cache=cache,
            prefix_caches=prefix_caches,
            active=[False for _ in range(cache_batch_size)],
            row_request_ids=[None for _ in range(cache_batch_size)],
            generated_tokens=[0 for _ in range(cache_batch_size)],
            seq_lens=torch.zeros(cache_batch_size, dtype=torch.long, device=self.device),
            next_token_tensor=torch.zeros(cache_batch_size, dtype=torch.long, device=self.device),
            cache_batch_size=cache_batch_size,
        )
        try:
            cache._skip_capture_sync = True
        except Exception:
            pass
        self._token_budget_step_state = state
        self._token_budget_step_last_result = None
        return state

    def _execute_token_budget_step_payload(
        self,
        payload: Mapping[str, object],
        state: _TokenBudgetStepState,
        *,
        temperature: float,
    ) -> _TokenBudgetStepResult:
        chunks_obj = payload.get("chunks", [])
        if not isinstance(chunks_obj, list):
            raise ValueError("token-budget step payload requires chunk list")
        chunk_rids = {str(c.get("request_id", "")) for c in chunks_obj if isinstance(c, Mapping)}
        pre_fin = payload.get("finished_request_ids", [])
        if isinstance(pre_fin, list):
            for rid_obj in pre_fin:
                rid = str(rid_obj)
                if rid in chunk_rids:
                    continue
                for row, row_rid in enumerate(state.row_request_ids):
                    if row_rid == rid:
                        state.active[row] = False
                        state.row_request_ids[row] = None
                        state.generated_tokens[row] = 0
                        state.seq_lens[row] = 0
                        try:
                            view = _cache_row_slice(state.cache, row, row + 1)
                            if view is not None:
                                _reset_generation_cache(view)
                        except Exception:
                            pass
                        break
        decode_tokens: dict[str, int | None] = {}
        prefill_tokens: dict[str, int | None] = {}
        index = 0
        while index < len(chunks_obj):
            chunk = chunks_obj[index]
            if not isinstance(chunk, Mapping):
                raise ValueError("token-budget step chunk must be a mapping")
            request_id = str(chunk.get("request_id"))
            row = int(chunk["row"])
            if row < 0 or row >= state.cache_batch_size:
                raise ValueError("token-budget step row is out of range")
            kind = str(chunk.get("kind"))
            if kind == "decode":
                decode_group: list[Mapping[str, object]] = [chunk]
                index += 1
                while index < len(chunks_obj):
                    next_chunk = chunks_obj[index]
                    if not isinstance(next_chunk, Mapping):
                        raise ValueError("token-budget step chunk must be a mapping")
                    if str(next_chunk.get("kind")) != "decode":
                        break
                    decode_group.append(next_chunk)
                    index += 1
                decode_tokens.update(
                    self._execute_token_budget_decode_chunks(
                        decode_group,
                        state,
                        temperature=temperature,
                    )
                )
                continue
            if kind == "prefill":
                prefill_group: list[Mapping[str, object]] = [chunk]
                index += 1
                while index < len(chunks_obj):
                    next_chunk = chunks_obj[index]
                    if not isinstance(next_chunk, Mapping):
                        raise ValueError("token-budget step chunk must be a mapping")
                    if str(next_chunk.get("kind")) != "prefill":
                        break
                    prefill_group.append(next_chunk)
                    index += 1
                prefill_tokens.update(
                    self._execute_token_budget_prefill_chunks(
                        prefill_group,
                        state,
                        temperature=temperature,
                    )
                )
                continue
            raise ValueError(f"unsupported token-budget step chunk kind: {kind}")

        finished_request_ids: list[str] = []
        finished_obj = payload.get("finished_request_ids", [])
        if isinstance(finished_obj, list):
            for request_id_obj in finished_obj:
                request_id = str(request_id_obj)
                for row, row_request_id in enumerate(state.row_request_ids):
                    if row_request_id != request_id:
                        continue
                    state.active[row] = False
                    state.row_request_ids[row] = None
                    state.generated_tokens[row] = 0
                    finished_request_ids.append(request_id)
                    break
        return _TokenBudgetStepResult(
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            finished_request_ids=tuple(finished_request_ids),
        )

    def _execute_token_budget_prefill_chunks(
        self,
        chunks: Sequence[Mapping[str, object]],
        state: _TokenBudgetStepState,
        *,
        temperature: float,
    ) -> dict[str, int | None]:
        tokens: dict[str, int | None] = {}
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            group_key = self._token_budget_prefill_group_key(chunk, state)
            if group_key is None:
                request_id = str(chunk.get("request_id"))
                row = int(chunk["row"])
                tokens[request_id] = self._execute_token_budget_prefill_chunk(
                    chunk,
                    state,
                    request_id=request_id,
                    row=row,
                    temperature=temperature,
                )
                index += 1
                continue
            group: list[Mapping[str, object]] = [chunk]
            index += 1
            while index < len(chunks):
                next_chunk = chunks[index]
                if self._token_budget_prefill_group_key(next_chunk, state) != group_key:
                    break
                group.append(next_chunk)
                index += 1
            if len(group) == 1:
                request_id = str(chunk.get("request_id"))
                row = int(chunk["row"])
                tokens[request_id] = self._execute_token_budget_prefill_chunk(
                    chunk,
                    state,
                    request_id=request_id,
                    row=row,
                    temperature=temperature,
                )
                continue
            grouped_tokens = self._execute_token_budget_shared_prefix_prefill_group(
                group,
                state,
                prefix=group_key,
                temperature=temperature,
            )
            if grouped_tokens is None:
                for item in group:
                    request_id = str(item.get("request_id"))
                    row = int(item["row"])
                    tokens[request_id] = self._execute_token_budget_prefill_chunk(
                        item,
                        state,
                        request_id=request_id,
                        row=row,
                        temperature=temperature,
                    )
                continue
            tokens.update(grouped_tokens)
        return tokens

    def _token_budget_prefill_group_key(
        self,
        chunk: Mapping[str, object],
        state: _TokenBudgetStepState,
    ) -> tuple[int, ...] | None:
        if str(chunk.get("kind")) != "prefill":
            return None
        if not bool(chunk.get("prompt_complete", False)) or not bool(chunk.get("emits_token", False)):
            return None
        start_token = int(chunk["start_token"])
        if start_token <= 0:
            return None
        row = int(chunk["row"])
        if row < 0 or row >= state.cache_batch_size or state.row_request_ids[row] is not None:
            return None
        prompt_tokens = int(chunk.get("prompt_tokens", 0))
        token_count = int(chunk["token_count"])
        if token_count != prompt_tokens - start_token:
            return None
        prompt_chunk_obj = chunk.get("prompt_chunk", [])
        if not isinstance(prompt_chunk_obj, list) or len(prompt_chunk_obj) != token_count:
            return None
        prefix_obj = chunk.get("prefix")
        if prefix_obj is None:
            matches = [prefix for prefix in state.prefix_caches if len(prefix) == start_token]
            if len(matches) != 1:
                return None
            prefix = matches[0]
        elif isinstance(prefix_obj, list):
            prefix = tuple(int(token_id) for token_id in prefix_obj)
        else:
            return None
        if len(prefix) != start_token or prefix not in state.prefix_caches:
            return None
        return prefix

    def _execute_token_budget_shared_prefix_prefill_group(
        self,
        chunks: Sequence[Mapping[str, object]],
        state: _TokenBudgetStepState,
        *,
        prefix: tuple[int, ...],
        temperature: float,
    ) -> dict[str, int | None] | None:
        prefix_cache = state.prefix_caches.get(prefix)
        if prefix_cache is None:
            return None
        request_ids: list[str] = []
        prompts: list[list[int]] = []
        target_rows: list[int] = []
        row_max_tokens: list[int] = []
        for chunk in chunks:
            request_id = str(chunk.get("request_id"))
            row = int(chunk["row"])
            prompt_chunk_obj = chunk.get("prompt_chunk", [])
            if not isinstance(prompt_chunk_obj, list):
                return None
            prompt_chunk = [int(token_id) for token_id in prompt_chunk_obj]
            request_ids.append(request_id)
            prompts.append([*prefix, *prompt_chunk])
            target_rows.append(row)
            row_max_tokens.append(int(chunk.get("max_tokens", 0)))
        prefill_result = self._prefill_shared_prefix_prompt_list_padded_suffix_rows(
            prompts,
            prefix_cache=prefix_cache,
            target_cache=state.cache,
            target_rows=target_rows,
            prefix_tokens=len(prefix),
            max_tokens=max(row_max_tokens, default=0),
            temperature=temperature,
            row_max_tokens=row_max_tokens,
            model=self.model,
        )
        if prefill_result is None:
            return None
        first_tokens, active_flags = prefill_result
        tokens: dict[str, int | None] = {}
        for request_id, prompt, row, max_tokens, token_id, active in zip(
            request_ids,
            prompts,
            target_rows,
            row_max_tokens,
            first_tokens,
            active_flags,
        ):
            tokens[request_id] = token_id
            state.generated_tokens[row] = 0 if token_id is None else 1
            state.seq_lens[row] = len(prompt)
            state.next_token_tensor[row] = 0 if token_id is None else int(token_id)
            state.active[row] = bool(active)
            state.row_request_ids[row] = request_id if max_tokens > 0 else None
        return tokens

    def _execute_token_budget_prefill_chunk(
        self,
        chunk: Mapping[str, object],
        state: _TokenBudgetStepState,
        *,
        request_id: str,
        row: int,
        temperature: float,
    ) -> int | None:
        prompt_chunk_obj = chunk.get("prompt_chunk", [])
        if not isinstance(prompt_chunk_obj, list):
            raise ValueError("token-budget prefill chunk requires prompt_chunk")
        prompt_chunk = [int(token_id) for token_id in prompt_chunk_obj]
        start_token = int(chunk["start_token"])
        token_count = int(chunk["token_count"])
        if token_count != len(prompt_chunk):
            raise ValueError("token-budget prefill token_count must match prompt_chunk")
        current_request_id = state.row_request_ids[row]
        if current_request_id is not None and current_request_id != request_id:
            state.active[row] = False
            state.row_request_ids[row] = None
            state.generated_tokens[row] = 0
            state.seq_lens[row] = 0
            try:
                view = _cache_row_slice(state.cache, row, row + 1)
                if view is not None:
                    _reset_generation_cache(view)
            except Exception:
                pass
            current_request_id = None
        if current_request_id is None:
            if start_token > 0:
                self._copy_token_budget_prefix_for_chunk(chunk, state, row=row, prefix_tokens=start_token)
            state.row_request_ids[row] = request_id
            state.active[row] = True
            state.generated_tokens[row] = 0
            state.seq_lens[row] = start_token
        if int(state.seq_lens[row].item()) != start_token:
            raise ValueError("token-budget prefill start_token does not match row state")
        if token_count <= 0:
            raise ValueError("token-budget prefill chunk must contain tokens")
        cache_view = _cache_row_slice(state.cache, row, row + 1)
        if cache_view is None:
            raise RuntimeError("token-budget prefill requires row-view cache")
        input_ids = torch.tensor([prompt_chunk], dtype=torch.long, device=self.device)
        graph_logits = _try_prefill_logits_graph(self.model, input_ids, cache_view, allow_capture=False)
        if graph_logits is not None:
            logits = graph_logits
        else:
            logits, _cache_view = _forward(self.model, input_ids, cache_view)
        state.seq_lens[row] = start_token + token_count
        if not bool(chunk.get("prompt_complete", False)):
            return None
        if not bool(chunk.get("emits_token", False)):
            return None
        next_token = _sample(self.model, logits[:, -1, :], temperature).to(self.device)
        token_id = int(next_token.item())
        state.next_token_tensor[row] = token_id
        state.generated_tokens[row] += 1
        return token_id

    def _copy_token_budget_prefix_for_chunk(
        self,
        chunk: Mapping[str, object],
        state: _TokenBudgetStepState,
        *,
        row: int,
        prefix_tokens: int,
    ) -> None:
        prefix_obj = chunk.get("prefix")
        if prefix_obj is None:
            matches = [prefix for prefix in state.prefix_caches if len(prefix) == prefix_tokens]
            if len(matches) != 1:
                raise RuntimeError("token-budget prefix prefill requires a unique installed prefix")
            prefix = matches[0]
        else:
            if not isinstance(prefix_obj, list):
                raise ValueError("token-budget prefix prefill requires prefix tokens")
            prefix = tuple(int(token_id) for token_id in prefix_obj)
        if len(prefix) != prefix_tokens:
            raise ValueError("token-budget prefix length does not match start_token")
        prefix_cache = state.prefix_caches.get(prefix)
        if prefix_cache is None:
            raise RuntimeError("missing token-budget prefix cache")
        _copy_generation_cache_row(
            prefix_cache,
            state.cache,
            source_row=0,
            target_row=row,
            seq_len=prefix_tokens,
        )

    def _execute_token_budget_decode_chunks(
        self,
        chunks: Sequence[Mapping[str, object]],
        state: _TokenBudgetStepState,
        *,
        temperature: float,
    ) -> dict[str, int | None]:
        if not chunks:
            return {}
        if len(chunks) == 1 or not callable(getattr(self.model, "decode_ragged_logits", None)):
            tokens: dict[str, int | None] = {}
            for chunk in chunks:
                request_id = str(chunk.get("request_id"))
                row = int(chunk["row"])
                tokens[request_id] = self._execute_token_budget_decode_chunk(
                    chunk,
                    state,
                    request_id=request_id,
                    row=row,
                    temperature=temperature,
                )
            return tokens

        request_ids: list[str] = []
        rows: list[int] = []
        input_tokens: list[int] = []
        for chunk in chunks:
            request_id = str(chunk.get("request_id"))
            row = int(chunk["row"])
            if row < 0 or row >= state.cache_batch_size:
                raise ValueError("token-budget decode row is out of range")
            if state.row_request_ids[row] != request_id or not state.active[row]:
                raise ValueError("token-budget decode requires an active matching row")
            if int(chunk["token_count"]) != 1:
                raise ValueError("token-budget decode chunk must contain one token")
            request_ids.append(request_id)
            rows.append(row)
            input_tokens.append(int(state.next_token_tensor[row].item()))

        actual_batch = len(input_tokens)
        padded_batch = _decode_graph_padded_batch(actual_batch, state.cache_batch_size)
        if padded_batch > actual_batch:
            pad_row = rows[0]
            pad_token = input_tokens[0]
            input_tokens_padded = input_tokens + [pad_token] * (padded_batch - actual_batch)
            rows_padded = rows + [pad_row] * (padded_batch - actual_batch)
        else:
            input_tokens_padded = input_tokens
            rows_padded = rows
        input_ids = torch.tensor([[token_id] for token_id in input_tokens_padded], dtype=torch.long, device=self.device)
        row_indices = torch.tensor(rows_padded, dtype=torch.long, device=self.device)
        graph_token = _try_decode_ragged_token_graph(
            self.model, input_ids, state.cache,
            seq_lens=state.seq_lens, row_indices=row_indices,
            temperature=temperature, allow_capture=False,
        )
        if graph_token is not None:
            next_tokens = graph_token[:actual_batch]
        else:
            if padded_batch > actual_batch:
                input_ids = input_ids[:actual_batch]
                row_indices = row_indices[:actual_batch]
            decode_ragged = getattr(self.model, "decode_ragged_logits", None)
            if callable(decode_ragged):
                logits = decode_ragged(input_ids, state.cache, seq_lens=state.seq_lens, row_indices=row_indices)
                next_tokens = _sample(self.model, logits[:, -1, :], temperature).to(self.device)
            else:
                next_tokens, _cache = _decode_next_token_ragged(
                    self.model,
                    input_ids,
                    state.cache,
                    state.seq_lens,
                    row_indices,
                    temperature,
                )
        token_values = [int(token_id) for token_id in next_tokens.detach().cpu().tolist()]
        result: dict[str, int | None] = {}
        for request_id, row, token_id in zip(request_ids, rows, token_values):
            state.next_token_tensor[row] = token_id
            state.generated_tokens[row] += 1
            state.seq_lens[row] = int(state.seq_lens[row].item()) + 1
            result[request_id] = token_id
        return result

    def _execute_token_budget_decode_chunk(
        self,
        chunk: Mapping[str, object],
        state: _TokenBudgetStepState,
        *,
        request_id: str,
        row: int,
        temperature: float,
    ) -> int | None:
        if state.row_request_ids[row] != request_id or not state.active[row]:
            raise ValueError("token-budget decode requires an active matching row")
        if int(chunk["token_count"]) != 1:
            raise ValueError("token-budget decode chunk must contain one token")
        cache_view = _cache_row_slice(state.cache, row, row + 1)
        if cache_view is None:
            raise RuntimeError("token-budget decode requires row-view cache")
        input_ids = state.next_token_tensor[row : row + 1].view(1, 1)
        next_token, _cache_view = _decode_next_token(self.model, input_ids, cache_view, temperature)
        token_id = int(next_token.item())
        state.next_token_tensor[row] = token_id
        state.generated_tokens[row] += 1
        state.seq_lens[row] = int(state.seq_lens[row].item()) + 1
        return token_id

    def _handle_token_budget_start_payload(self, payload: Mapping[str, object]) -> _TokenBudgetStepState:
        prefix_obj = payload.get("prefix", [])
        if not isinstance(prefix_obj, list):
            raise ValueError("token-budget start prefix must be a list")
        state = self._start_token_budget_step_state(
            cache_batch_size=int(payload["max_active_rows"]),
            max_seq_len=int(payload["max_seq_len"]),
            prefix=[int(token_id) for token_id in prefix_obj],
            temperature=float(payload.get("temperature", 0.0)),
        )
        return state

    def _handle_token_budget_step_payload(self, payload: Mapping[str, object]) -> _TokenBudgetStepResult:
        state = self._token_budget_step_state
        if state is None:
            raise RuntimeError("token-budget step state is not installed")
        result = self._execute_token_budget_step_payload(
            payload,
            state,
            temperature=float(payload.get("temperature", 0.0)),
        )
        self._token_budget_step_last_result = result
        return result

    def _execute_token_budget_decode_run_payload(
        self,
        payload: Mapping[str, object],
        state: _TokenBudgetStepState,
        *,
        temperature: float,
    ) -> _TokenBudgetDecodeRunResult:
        steps_obj = payload.get("steps", [])
        if not isinstance(steps_obj, list) or not steps_obj:
            raise ValueError("token-budget decode run requires non-empty steps")
        results: list[_TokenBudgetStepResult] = []
        for step_payload in steps_obj:
            if not isinstance(step_payload, Mapping):
                raise ValueError("token-budget decode run steps must be mappings")
            chunks = step_payload.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("token-budget decode run step requires chunks")
            if any(not isinstance(chunk, Mapping) or str(chunk.get("kind")) != "decode" for chunk in chunks):
                raise ValueError("token-budget decode run only accepts decode chunks")
            results.append(
                self._execute_token_budget_step_payload(
                    step_payload,
                    state,
                    temperature=temperature,
                )
            )
        return _TokenBudgetDecodeRunResult(step_results=tuple(results))

    def _handle_token_budget_decode_run_payload(
        self,
        payload: Mapping[str, object],
    ) -> _TokenBudgetDecodeRunResult:
        state = self._token_budget_step_state
        if state is None:
            raise RuntimeError("token-budget decode run state is not installed")
        result = self._execute_token_budget_decode_run_payload(
            payload,
            state,
            temperature=float(payload.get("temperature", 0.0)),
        )
        self._token_budget_step_last_result = result.step_results[-1]
        return result

    def _handle_token_budget_close_payload(self, payload: Mapping[str, object]) -> None:
        del payload
        self._token_budget_step_state = None
        self._token_budget_step_last_result = None

    def _run_token_budget_group_local(
        self,
        group: Sequence[_QueuedGeneration],
        *,
        max_active_rows: int,
        max_scheduled_tokens: int,
        prefill_chunk_size: int | None = None,
        prefix_tokens: int = 0,
        decode_run_steps: int = 1,
        max_scheduler_steps: int | None = None,
    ) -> _TokenBudgetLocalRunStats:
        if not group:
            return _TokenBudgetLocalRunStats(
                scheduler_steps=0,
                step_commands=0,
                decode_run_commands=0,
                empty_plans=0,
                emitted_tokens=0,
                finished_events=0,
                max_decode_run_steps=0,
                closed=True,
            )
        request_by_id = {str(index): request for index, request in enumerate(group)}
        prompts = [request.prompt for request in group]
        prefix: tuple[int, ...] = ()
        if prefix_tokens > 0 and prompts:
            candidate = tuple(int(token_id) for token_id in prompts[0][:prefix_tokens])
            if len(candidate) == prefix_tokens and all(tuple(prompt[:prefix_tokens]) == candidate for prompt in prompts):
                prefix = candidate
        scheduler = _token_budget_scheduler_for_group(
            group,
            max_active_rows=max_active_rows,
            max_scheduled_tokens=max_scheduled_tokens,
            prefill_chunk_size=prefill_chunk_size,
            prefix_tokens=prefix_tokens,
        )
        max_seq_len = max((len(request.prompt) + request.max_tokens for request in group), default=1)
        self._start_token_budget_step_state(
            cache_batch_size=max_active_rows,
            max_seq_len=max_seq_len,
            prefix=prefix,
            temperature=group[0].temperature,
        )
        finished_request_ids: tuple[str, ...] = ()
        pending_plan: _TokenBudgetPlan | None = None
        scheduler_steps = 0
        step_commands = 0
        decode_run_commands = 0
        empty_plans = 0
        emitted_tokens = 0
        finished_events = 0
        max_decode_run_size = 0
        stream_rows = _StreamRowState()
        try:
            while scheduler.has_work() or finished_request_ids or pending_plan is not None:
                if pending_plan is None:
                    plan = scheduler.step(finished_request_ids=finished_request_ids)
                    scheduler_steps += 1
                else:
                    plan = pending_plan
                    pending_plan = None
                finished_request_ids = ()
                if max_scheduler_steps is not None and scheduler_steps > max_scheduler_steps:
                    raise RuntimeError("token-budget local runner exceeded max_scheduler_steps")
                if not plan.chunks:
                    empty_plans += 1
                    continue
                if decode_run_steps > 1 and _token_budget_plan_is_decode_only(plan):
                    plans = [plan]
                    while (
                        len(plans) < decode_run_steps
                        and not plans[-1].finished_request_ids
                        and scheduler.has_work()
                    ):
                        next_plan = scheduler.step(finished_request_ids=())
                        scheduler_steps += 1
                        if max_scheduler_steps is not None and scheduler_steps > max_scheduler_steps:
                            raise RuntimeError("token-budget local runner exceeded max_scheduler_steps")
                        if not next_plan.chunks:
                            if next_plan.finished_request_ids:
                                pending_plan = next_plan
                                break
                            empty_plans += 1
                            continue
                        if not _token_budget_plan_is_decode_only(next_plan):
                            pending_plan = next_plan
                            break
                        plans.append(next_plan)
                    payload = _token_budget_decode_run_payload(plans, request_by_id)
                    result = self._handle_token_budget_decode_run_payload(payload)
                    emitted_finished_request_ids = self._emit_token_budget_decode_run_result(
                        group,
                        payload,
                        result,
                        stream_rows=stream_rows,
                    )
                    decode_run_commands += 1
                    max_decode_run_size = max(max_decode_run_size, len(plans))
                    emitted_tokens += sum(
                        token is not None
                        for step_result in result.step_results
                        for token in (*step_result.decode_tokens.values(), *step_result.prefill_tokens.values())
                    )
                    finished_request_ids = _merge_token_budget_finished_ids(
                        result.finished_request_ids,
                        emitted_finished_request_ids,
                    )
                    finished_events += len(finished_request_ids)
                    continue
                payload = _token_budget_step_payload(plan, request_by_id)
                result = self._handle_token_budget_step_payload(payload)
                emitted_finished_request_ids = self._emit_token_budget_step_result(
                    group,
                    payload,
                    result,
                    stream_rows=stream_rows,
                )
                step_commands += 1
                emitted_tokens += sum(token is not None for token in result.decode_tokens.values())
                emitted_tokens += sum(token is not None for token in result.prefill_tokens.values())
                finished_request_ids = _merge_token_budget_finished_ids(
                    result.finished_request_ids,
                    emitted_finished_request_ids,
                )
                finished_events += len(finished_request_ids)
        finally:
            self._handle_token_budget_close_payload({"op": "token_budget_close"})
        return _TokenBudgetLocalRunStats(
            scheduler_steps=scheduler_steps,
            step_commands=step_commands,
            decode_run_commands=decode_run_commands,
            empty_plans=empty_plans,
            emitted_tokens=emitted_tokens,
            finished_events=finished_events,
            max_decode_run_steps=max_decode_run_size,
            closed=self._token_budget_step_state is None,
        )

    def _emit_token_budget_decode_run_result(
        self,
        group: Sequence[_QueuedGeneration],
        payload: Mapping[str, object],
        result: _TokenBudgetDecodeRunResult,
        *,
        stream_rows: _StreamRowState | None = None,
    ) -> tuple[str, ...]:
        steps_obj = payload.get("steps", [])
        if not isinstance(steps_obj, list):
            raise ValueError("token-budget decode run emit requires step list")
        if len(steps_obj) != len(result.step_results):
            raise ValueError("token-budget decode run result length mismatch")
        finished_request_ids: list[str] = []
        for step_payload, step_result in zip(steps_obj, result.step_results):
            if not isinstance(step_payload, Mapping):
                raise ValueError("token-budget decode run emit step must be a mapping")
            finished_request_ids.extend(
                self._emit_token_budget_step_result(
                    group,
                    step_payload,
                    step_result,
                    stream_rows=stream_rows,
                )
            )
        return _merge_token_budget_finished_ids(finished_request_ids)

    def _emit_token_budget_step_result(
        self,
        group: Sequence[_QueuedGeneration],
        payload: Mapping[str, object],
        result: _TokenBudgetStepResult,
        *,
        stream_rows: _StreamRowState | None = None,
    ) -> tuple[str, ...]:
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("token-budget emit requires chunk list")
        stop_token_ids = getattr(self, "stop_token_ids", frozenset())
        finished_request_ids: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("token-budget emit chunk must be a mapping")
            request_id = str(chunk.get("request_id"))
            token_id = result.decode_tokens.get(request_id)
            if token_id is None and request_id in result.prefill_tokens:
                token_id = result.prefill_tokens[request_id]
            if token_id is None:
                continue
            request = group[int(request_id)]
            if request.done:
                continue
            if stream_rows is not None:
                row = int(chunk["row"])
                stream_rows.admit(request_id, row, request)
                if stream_rows.emit(request_id, int(token_id), stop_token_ids=stop_token_ids):
                    finished_request_ids.append(request_id)
                continue
            if int(token_id) in stop_token_ids:
                _finish_stream_request(request)
                finished_request_ids.append(request_id)
                continue
            request.responses.put(int(token_id))
        for request_id in result.finished_request_ids:
            request = group[int(request_id)]
            if stream_rows is not None:
                if stream_rows.finish(str(request_id)):
                    finished_request_ids.append(str(request_id))
                elif not request.done:
                    _finish_stream_request(request)
                    finished_request_ids.append(str(request_id))
            elif not request.done:
                _finish_stream_request(request)
                finished_request_ids.append(str(request_id))
        return _merge_token_budget_finished_ids(finished_request_ids)

    def _prefill_shared_prefix_prompt_list_padded_suffixes(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        prefix_cache: object,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: Sequence[int],
        model: object,
    ) -> tuple[object, list[int | None], list[bool]] | None:
        prompt_count = len(prompts)
        if prompt_count <= 0:
            return None
        suffix_rows = [list(prompt[prefix_tokens:]) for prompt in prompts]
        max_suffix_len = max((len(row) for row in suffix_rows), default=0)
        if max_suffix_len <= 0:
            return None
        padded_suffix_len = _shared_prefix_padded_suffix_bucketed_length(
            model,
            device=self.device,
            prompt_count=prompt_count,
            max_suffix_len=max_suffix_len,
        )
        max_prompt_len = max((len(prompt) for prompt in prompts), default=0)
        cache_batch_size = _generation_cache_batch_capacity(model, prompt_count)
        cache = self._generation_cache(
            prompt_count,
            max_prompt_len + max_tokens,
            model=model,
            batch_capacity=cache_batch_size,
            pool=_shared_prefix_ragged_cache_pool_enabled_for_model(
                    model,
                    max_tokens=max_tokens,
                ),
            )
        physical_batch_size = _cache_batch_size(cache) or prompt_count
        prefill_batch_size = prompt_count
        if _shared_prefix_padded_suffix_static_batch_enabled(
            model,
            prompt_count=prompt_count,
            physical_batch_size=physical_batch_size,
            device=self.device,
        ):
            prefill_batch_size = physical_batch_size
        self._add_stream_group_profile_extra(
            padded_suffix_physical_batch_size=physical_batch_size,
            padded_suffix_prefill_batch_size=prefill_batch_size,
            padded_suffix_max_suffix_len=max_suffix_len,
            padded_suffix_bucketed_suffix_len=padded_suffix_len,
        )
        _set_cache_physical_rows_initialized(cache, False)
        try:
            _copy_generation_cache_first_row(prefix_cache, cache, prefill_batch_size)
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_padded_suffix_cache", exc)
            return None

        pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
        if any(not suffix for suffix in suffix_rows):
            return None
        suffix_lengths = [len(suffix) for suffix in suffix_rows]
        prefill_suffix_rows = suffix_rows
        prefill_suffix_lengths = suffix_lengths
        if prefill_batch_size > prompt_count:
            dummy_suffix = list(max(suffix_rows, key=len))
            extra_rows = prefill_batch_size - prompt_count
            prefill_suffix_rows = suffix_rows + [dummy_suffix for _ in range(extra_rows)]
            prefill_suffix_lengths = suffix_lengths + [len(dummy_suffix) for _ in range(extra_rows)]
        suffix_ids = _padded_token_rows_tensor(prefill_suffix_rows, self.device, pad_token_id)
        if suffix_ids.size(1) < padded_suffix_len:
            extra = torch.full(
                (suffix_ids.size(0), padded_suffix_len - suffix_ids.size(1)),
                int(pad_token_id),
                dtype=suffix_ids.dtype,
                device=suffix_ids.device,
            )
            suffix_ids = torch.cat((suffix_ids, extra), dim=1)

        # Right padding is safe here: real suffix-token logits never attend to
        # future pad tokens, and later ragged decode uses true per-row lengths.
        next_token_start_s = self._stream_group_profile_start_s()
        next_token, cache = _prefill_padded_suffix_next_token(
            model,
            suffix_ids,
            cache,
            prefill_suffix_lengths,
            temperature,
            allow_capture=self._batched_prefill_graph_capture_enabled(
                model,
                suffix_ids,
                cache,
                temperature=temperature,
                max_tokens=max_tokens,
                selected_logits=True,
            ),
        )
        self._add_stream_group_profile_elapsed(
            "shared_prefix_padded_suffix_next_token_ms",
            next_token_start_s,
        )
        seq_len_update_start_s = self._stream_group_profile_start_s()
        try:
            _set_generation_cache_rows_seq_lens(
                cache,
                range(prompt_count),
                [prefix_tokens + length for length in suffix_lengths],
            )
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_padded_suffix_lengths", exc)
            return None
        self._add_stream_group_profile_elapsed(
            "shared_prefix_padded_suffix_seq_len_update_ms",
            seq_len_update_start_s,
        )
        _set_cache_physical_rows_initialized(cache, prefill_batch_size >= physical_batch_size)
        next_token = next_token.to(self.device)
        stop_token_ids = self.stop_token_ids
        first_tokens: list[int | None] = []
        active: list[bool] = []
        cpu_token_start_s = self._stream_group_profile_start_s()
        for row, token_id in enumerate(next_token[:prompt_count].detach().cpu().tolist()):
            if row_max_tokens[row] <= 0:
                first_tokens.append(None)
                active.append(False)
                continue
            token = int(token_id)
            first_tokens.append(token)
            active.append(token not in stop_token_ids and row_max_tokens[row] > 1)
        self._add_stream_group_profile_elapsed(
            "shared_prefix_padded_suffix_cpu_tokens_ms",
            cpu_token_start_s,
        )
        return cache, first_tokens, active

    def _shared_prefix_prompt_list_ragged_cache(
        self,
        states: Sequence[Mapping[str, object]],
        *,
        prompt_lengths: Sequence[int],
        prompt_count: int,
        max_tokens: int,
        model: object,
    ) -> object | None:
        if prompt_count <= 0:
            return None
        max_prompt_len = max(prompt_lengths, default=0)
        if max_prompt_len <= 0:
            return None
        cache_batch_size = _generation_cache_batch_capacity(model, prompt_count)
        cache = self._generation_cache(
            prompt_count,
            max_prompt_len + max_tokens,
            model=model,
            batch_capacity=cache_batch_size,
            pool=_shared_prefix_ragged_cache_pool_enabled_for_model(
                model,
                max_tokens=max_tokens,
            ),
        )
        try:
            bulk_copy_used = False
            if _shared_prefix_ragged_cache_bulk_copy_allowed(prompt_count, max_tokens=max_tokens):
                bulk_copy_used = _copy_generation_cache_state_rows_padded(
                    states,
                    cache,
                    prompt_lengths=prompt_lengths,
                    prompt_count=prompt_count,
                )
            if bulk_copy_used:
                self._add_stream_group_profile_extra(
                    shared_prefix_ragged_cache_bulk_copy=True,
                    shared_prefix_ragged_cache_bulk_rows=prompt_count,
                )
            else:
                for state in states:
                    source_cache = state.get("cache")
                    indices = state.get("indices")
                    if source_cache is None or not isinstance(indices, list):
                        return None
                    for source_row, original_index in enumerate(indices):
                        _copy_generation_cache_row(
                            source_cache,
                            cache,
                            source_row=source_row,
                            target_row=int(original_index),
                            seq_len=int(prompt_lengths[int(original_index)]),
                        )
            _set_generation_cache_seq_len(cache, max_prompt_len)
            _set_cache_physical_rows_initialized(cache, (_cache_batch_size(cache) or prompt_count) <= prompt_count)
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_ragged_cache", exc)
            return None
        return cache

    def _decode_shared_prefix_prompt_list_ragged_step(
        self,
        *,
        cache: object,
        active: list[bool],
        per_row_limits: Sequence[int],
        generated_tokens: list[int],
        seq_lens: Tensor,
        next_token_tensor: Tensor,
        step: int,
        cache_batch_size: int,
        temperature: float,
        static_graph_buckets: bool,
    ) -> tuple[object, list[int | None]] | None:
        model = self.model
        logical_batch_size = max(1, min(int(cache_batch_size), len(active)))
        active_for_plan = active[:logical_batch_size]
        active_indices = [index for index, is_active in enumerate(active_for_plan) if is_active]
        should_decode = _sync_tensor_parallel_continue(model, bool(active_indices), self.device)
        if not should_decode:
            return None
        prepare_start_s = self._stream_group_profile_start_s()
        force_row_indices = env_flag(
            "TORCHINFERNO_OPENAI_RAGGED_DECODE_FORCE_ROW_INDICES",
            _force_tp_shared_prefix_ragged_row_indices(model),
        )
        row_plan = _shared_prefix_ragged_decode_row_plan(
            active_for_plan,
            active_indices=active_indices,
            step=step,
            cache_batch_size=cache_batch_size,
            force_row_indices=force_row_indices,
            static_graph_buckets=static_graph_buckets,
            prefer_full_bucket=_prefer_paged_ragged_decode_full_bucket(
                model,
                cache,
                static_graph_buckets=static_graph_buckets,
            ),
        )
        decode_indices = list(row_plan.decode_indices)
        if self._queue_profile_path_value():
            active_count = len(active_indices)
            decode_count = len(decode_indices)
            self._add_stream_group_profile_value("shared_prefix_ragged_active_row_steps", float(active_count))
            self._add_stream_group_profile_value("shared_prefix_ragged_decode_row_steps", float(decode_count))
            if decode_count > active_count:
                self._add_stream_group_profile_value(
                    "shared_prefix_ragged_bucket_padding_row_steps",
                    float(decode_count - active_count),
                )
                self._add_stream_group_profile_value("shared_prefix_ragged_bucketed_steps", 1.0)
            if row_plan.row_indices is not None:
                self._add_stream_group_profile_value("shared_prefix_ragged_row_index_steps", 1.0)
        if row_plan.row_indices is None:
            decode_input = next_token_tensor[: len(decode_indices), None]
            decode_seq_lens = seq_lens[: len(decode_indices)]
            row_indices = None
            advance_row_indices = None
        else:
            row_indices = torch.tensor(row_plan.row_indices, dtype=torch.long, device=self.device)
            advance_row_indices = (
                torch.tensor(row_plan.advance_row_indices, dtype=torch.long, device=self.device)
                if row_plan.advance_row_indices is not None
                else None
            )
            decode_input = next_token_tensor.index_select(0, row_indices)[:, None]
            decode_seq_lens = seq_lens
        self._add_stream_group_profile_elapsed("shared_prefix_ragged_prepare_ms", prepare_start_s)
        model_start_s = self._stream_group_profile_start_s()
        model_start_event: torch.cuda.Event | None = None
        model_end_event: torch.cuda.Event | None = None
        if model_start_s > 0.0 and self.device.type == "cuda":
            model_start_event = torch.cuda.Event(enable_timing=True)
            model_end_event = torch.cuda.Event(enable_timing=True)
            model_start_event.record(torch.cuda.current_stream(self.device))
        next_token, cache = _decode_next_token_ragged(
            model,
            decode_input,
            cache,
            decode_seq_lens,
            row_indices,
            temperature,
        )
        next_token = next_token.to(self.device)
        if model_start_event is not None and model_end_event is not None:
            model_end_event.record(torch.cuda.current_stream(self.device))
            model_end_event.synchronize()
            self._add_stream_group_profile_value(
                "shared_prefix_ragged_model_ms",
                model_start_event.elapsed_time(model_end_event),
            )
        else:
            self._add_stream_group_profile_elapsed("shared_prefix_ragged_model_ms", model_start_s)
        decode_token_count = min(int(next_token.numel()), len(decode_indices))
        if decode_token_count > 0:
            decode_next_tokens = next_token[:decode_token_count].to(
                device=next_token_tensor.device,
                dtype=next_token_tensor.dtype,
            )
            if row_indices is None:
                next_token_tensor[:decode_token_count].copy_(decode_next_tokens)
            elif advance_row_indices is not None:
                logical_update_count = 0
                for original_index in decode_indices[:decode_token_count]:
                    if original_index >= len(active):
                        break
                    logical_update_count += 1
                if logical_update_count > 0:
                    next_token_tensor.index_copy_(
                        0,
                        row_indices[:logical_update_count],
                        decode_next_tokens[:logical_update_count],
                    )
            else:
                next_token_tensor.index_copy_(0, row_indices[:decode_token_count], decode_next_tokens)
        if row_indices is None:
            seq_lens[: len(decode_indices)] = seq_lens[: len(decode_indices)] + 1
        elif advance_row_indices is not None:
            seq_lens[advance_row_indices] = seq_lens.index_select(0, advance_row_indices) + 1
        else:
            seq_lens[row_indices] = seq_lens.index_select(0, row_indices) + 1
        step_tokens: list[int | None] = [None for _ in active]
        cpu_tokens_start_s = self._stream_group_profile_start_s()
        token_ids = next_token.detach().cpu().tolist()
        self._add_stream_group_profile_elapsed("shared_prefix_ragged_cpu_tokens_ms", cpu_tokens_start_s)
        state_update_start_s = self._stream_group_profile_start_s()
        for offset, token_id in enumerate(token_ids):
            original_index = decode_indices[offset]
            if original_index >= len(active):
                continue
            token_id = int(token_id)
            if not active[original_index]:
                continue
            step_tokens[original_index] = token_id
            generated_tokens[original_index] += 1
            if token_id in self.stop_token_ids or generated_tokens[original_index] >= per_row_limits[original_index]:
                active[original_index] = False
        self._add_stream_group_profile_elapsed("shared_prefix_ragged_state_update_ms", state_update_start_s)
        return cache, step_tokens

    def _decode_shared_prefix_prompt_list_ragged(
        self,
        *,
        cache: object,
        active: list[bool],
        prompt_lengths: Sequence[int],
        max_tokens: int,
        next_tokens: Sequence[int | None],
        temperature: float,
        row_max_tokens: Sequence[int] | None = None,
        early_restart_check: object = None,
    ) -> Iterator[list[int | None]]:
        model = self.model
        if _disable_tp_shared_prefix_ragged_decode_graph(model, max_tokens=max_tokens):
            _set_ragged_decode_graph_disabled(cache, True)
        _set_runtime_ragged_decode_graph_capture(
            cache,
            _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=max_tokens),
        )
        if max_tokens <= 1 or not any(active):
            return
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, len(active), max_tokens)
        for index, limit in enumerate(per_row_limits):
            if limit <= 1:
                active[index] = False
        if not any(active):
            return
        static_graph_bucket_capacity = (
            _force_tp_shared_prefix_ragged_row_indices(model)
            and not getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)
            and not _disable_tp_shared_prefix_ragged_static_buckets(model, max_tokens=max_tokens)
        )
        cache_batch_size = len(active)
        if static_graph_bucket_capacity:
            physical_cache_rows = max(len(active), _cache_batch_size(cache) or 0)
            if (
                physical_cache_rows > len(active)
                and (
                    _cache_physical_rows_initialized(cache)
                    or not _cache_requires_physical_rows_initialized(cache)
                )
            ):
                cache_batch_size = physical_cache_rows
        seq_lens = torch.empty(cache_batch_size, dtype=torch.long, device=self.device)
        prompt_len_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        seq_lens[: len(prompt_lengths)] = prompt_len_tensor
        if cache_batch_size > len(prompt_lengths):
            seq_lens[len(prompt_lengths) :] = max(prompt_lengths, default=0)
        next_token_tensor = torch.zeros(cache_batch_size, dtype=torch.long, device=self.device)
        if next_tokens:
            next_token_tensor[: len(next_tokens)] = torch.tensor(
                [0 if token_id is None else int(token_id) for token_id in next_tokens],
                dtype=torch.long,
                device=self.device,
            )
        generated_tokens = [1 if is_active else 0 for is_active in active]
        ephemeral_graph_allowed = (
            getattr(cache, "_torchinferno_ephemeral_cache", False)
            and env_flag("TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH", True)
        )
        ephemeral_graph_min_step = env_int(
            "TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH_MIN_STEP",
            1,
            minimum=1,
        )
        static_graph_bucket_max_steps = env_int(
            "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_MAX_STEPS",
            64,
            minimum=0,
        )
        ephemeral_graph_scope = False
        decode_profile_start_s = self._stream_group_profile_start_s()
        decoded_steps = 0
        initial_active_rows = sum(1 for is_active in active if is_active)
        min_active_rows = initial_active_rows
        final_active_rows = initial_active_rows
        active_full_steps = 0
        active_half_step: int | None = None
        active_row_steps = 0
        try:
            for step in range(1, max_tokens):
                active_rows_before = sum(1 for is_active in active if is_active)
                if active_rows_before <= 0:
                    break
                active_row_steps += active_rows_before
                if active_rows_before == initial_active_rows:
                    active_full_steps += 1
                if (
                    active_half_step is None
                    and initial_active_rows > 0
                    and active_rows_before * 2 <= initial_active_rows
                ):
                    active_half_step = step
                step_static_graph_buckets = bool(static_graph_bucket_capacity) and (
                    static_graph_bucket_max_steps <= 0 or step < static_graph_bucket_max_steps
                )
                _set_shared_prefix_ragged_static_graph_bucket_mode(
                    model,
                    cache,
                    static_graph_buckets=step_static_graph_buckets,
                )
                if ephemeral_graph_allowed and not ephemeral_graph_scope and step >= ephemeral_graph_min_step:
                    try:
                        setattr(cache, "_torchinferno_ephemeral_ragged_graph_scope", True)
                        ephemeral_graph_scope = True
                    except Exception:
                        ephemeral_graph_allowed = False
                step_result = self._decode_shared_prefix_prompt_list_ragged_step(
                    cache=cache,
                    active=active,
                    per_row_limits=per_row_limits,
                    generated_tokens=generated_tokens,
                    seq_lens=seq_lens,
                    next_token_tensor=next_token_tensor,
                    step=step,
                    cache_batch_size=cache_batch_size,
                    temperature=temperature,
                    static_graph_buckets=step_static_graph_buckets,
                )
                if step_result is None:
                    break
                cache, step_tokens = step_result
                decoded_steps += 1
                final_active_rows = sum(1 for is_active in active if is_active)
                min_active_rows = min(min_active_rows, final_active_rows)
                yield step_tokens
                # Mid-batch early restart: when at least half the initial batch
                # has finished, break so the batch worker can restart with the
                # remaining live rows + new arrivals. TP-safe: active[] is
                # derived from NCCL-synchronized tokens, so primary and workers
                # see the same counts and break at the same step. No extra
                # dist.broadcast (which caused NCCL desync in earlier attempts).
                if (
                    early_restart_check is not None
                    and initial_active_rows > 1
                    and final_active_rows * 2 <= initial_active_rows
                    and callable(early_restart_check)
                    and early_restart_check()
                ):
                    break
        finally:
            self._add_stream_group_profile_elapsed(
                "shared_prefix_ragged_decode_ms",
                decode_profile_start_s,
            )
            self._add_stream_group_profile_extra(
                shared_prefix_ragged_decode_steps=decoded_steps,
                shared_prefix_ragged_static_bucket_capacity=bool(static_graph_bucket_capacity),
                shared_prefix_ragged_active_initial=initial_active_rows,
                shared_prefix_ragged_active_min=min_active_rows,
                shared_prefix_ragged_active_final=final_active_rows,
                shared_prefix_ragged_active_full_steps=active_full_steps,
                shared_prefix_ragged_active_half_step=active_half_step,
                shared_prefix_ragged_active_row_steps_observed=active_row_steps,
            )
            if ephemeral_graph_scope:
                try:
                    setattr(cache, "_torchinferno_ephemeral_ragged_graph_scope", False)
                except Exception:
                    pass
                _release_decode_graphs_for_cache(model, cache)

    @torch.inference_mode()
    def _generate_shared_prefix_batch_steps(
        self,
        input_ids: Tensor,
        *,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        microbatch_size: int,
        row_max_tokens: Sequence[int] | None = None,
        prefix_cache_prompts: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        if not _supports_incremental_generation(model):
            rows = _generated_rows_with_model(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_token_ids=stop_token_ids,
            )
            yield from _iter_generated_steps(rows, max_tokens, stop_token_ids)
            return

        batch_size = input_ids.size(0)
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, batch_size, max_tokens)
        cache = self._generation_cache(
            batch_size,
            input_ids.size(1) + max_tokens,
            model=model,
        )
        prefix_ids = input_ids[:1, :prefix_tokens]
        restored_prefix_tokens = self._restore_exact_prefix_cache(prefix_ids, cache)
        if restored_prefix_tokens != prefix_tokens:
            cache = _prefill_cache_only(
                model,
                prefix_ids,
                cache,
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
            )
            self._save_prompt_prefix_cache(prefix_ids, cache)
        _mark_generation_cache_prefix(cache, _tensor_row_tokens(prefix_ids))
        _repeat_generation_cache_first_batch(cache, batch_size)
        next_token, cache = _prefill_next_token(
            model,
            input_ids[:, prefix_tokens:],
            cache,
            temperature,
            allow_capture=self._batched_prefill_graph_capture_enabled(
                model,
                input_ids[:, prefix_tokens:],
                cache,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        next_token = next_token.to(self.device)

        active = [limit > 0 for limit in per_row_limits]
        step_tokens: list[int | None] = []
        for row, token_id in enumerate(next_token.detach().cpu().tolist()):
            token_id = int(token_id)
            if not active[row]:
                step_tokens.append(None)
                continue
            step_tokens.append(token_id)
            if token_id in stop_token_ids or per_row_limits[row] <= 1:
                active[row] = False
        yield step_tokens
        if prefix_cache_prompts is not None:
            self._save_prompt_prefix_cache_rows(prefix_cache_prompts, cache)
        should_continue = max_tokens > 1 and any(active)
        should_continue = _sync_tensor_parallel_continue(model, should_continue, next_token.device)
        if not should_continue:
            return

        if 0 < microbatch_size < batch_size:
            cache_views = [
                _cache_row_slice(cache, start, min(batch_size, start + microbatch_size))
                for start in range(0, batch_size, microbatch_size)
            ]
            if all(view is not None for view in cache_views):
                yield from self._decode_shared_prefix_microbatches(
                    cache_views=[view for view in cache_views if view is not None],
                    active=active,
                    batch_size=batch_size,
                    microbatch_size=microbatch_size,
                    max_tokens=max_tokens,
                    next_token=next_token,
                    temperature=temperature,
                    row_max_tokens=per_row_limits,
                )
                return

        for step in range(1, max_tokens):
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)
            token_ids = next_token.detach().cpu().tolist()
            step_tokens = []
            for row, token_id in enumerate(token_ids):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if token_id in stop_token_ids or step + 1 >= per_row_limits[row]:
                    active[row] = False
            yield step_tokens
            should_continue = step + 1 < max_tokens and any(active)
            should_continue = _sync_tensor_parallel_continue(model, should_continue, next_token.device)
            if not should_continue:
                break

    def _decode_shared_prefix_microbatches(
        self,
        *,
        cache_views: list[object],
        active: list[bool],
        batch_size: int,
        microbatch_size: int,
        max_tokens: int,
        next_token: Tensor,
        temperature: float,
        row_max_tokens: Sequence[int] | None = None,
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        states: list[dict[str, object]] = []
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, batch_size, max_tokens)
        for slot, start in enumerate(range(0, batch_size, microbatch_size)):
            end = min(batch_size, start + microbatch_size)
            states.append(
                {
                    "start": start,
                    "active": active[start:end],
                    "cache": cache_views[slot],
                    "next_token": next_token[start:end].contiguous(),
                    "row_max_tokens": per_row_limits[start:end],
                }
            )
        for step in range(1, max_tokens):
            emitted = False
            step_tokens = [None for _ in range(batch_size)]
            for state in states:
                chunk_active = state["active"]
                local_should_decode = isinstance(chunk_active, list) and any(chunk_active)
                should_decode = _sync_tensor_parallel_continue(model, local_should_decode, self.device)
                if not should_decode:
                    continue
                chunk_token = state["next_token"]
                cache = state["cache"]
                if not isinstance(chunk_token, Tensor):
                    raise RuntimeError("invalid shared-prefix microbatch token state")
                chunk_token, cache = _decode_next_token(model, chunk_token[:, None], cache, temperature)
                chunk_token = chunk_token.to(self.device)
                state["cache"] = cache
                state["next_token"] = chunk_token
                start = int(state["start"])
                chunk_limits = state["row_max_tokens"]
                if not isinstance(chunk_limits, list):
                    raise RuntimeError("invalid shared-prefix row limit state")
                for offset, token_id in enumerate(chunk_token.detach().cpu().tolist()):
                    if not chunk_active[offset]:
                        continue
                    token_id = int(token_id)
                    step_tokens[start + offset] = token_id
                    if token_id in stop_token_ids or step + 1 >= int(chunk_limits[offset]):
                        chunk_active[offset] = False
                        active[start + offset] = False
                emitted = True
            if not emitted:
                break
            yield step_tokens


def build_engine(config: OpenAIServerConfig) -> OpenAICompletionEngine:
    model, device = _load_model(config)
    vocab_size = int(getattr(getattr(model, "config", object()), "vocab_size", 256))
    tokenizer = load_chat_tokenizer(config, vocab_size)
    return OpenAICompletionEngine(
        model,
        tokenizer,
        model_id=config.model,
        device=device,
        cache_backend=config.cache_backend,
        page_size=config.page_size,
        max_model_len=config.max_model_len,
        max_batch_size=config.max_batch_size,
        batch_wait_ms=config.batch_wait_ms,
        single_request_admission_wait_ms=config.single_request_admission_wait_ms,
    )


def serve(config: OpenAIServerConfig) -> None:
    engine = build_engine(config)
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return
    server_cls = _FastOpenAIServer if env_flag("TORCHINFERNO_OPENAI_FAST_HTTP", True) else _OpenAIServer
    server = server_cls((config.host, config.port), engine)
    print(
        f"TorchInferno OpenAI server listening on http://{config.host}:{server.server_port}/v1 "
        f"model={config.model}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


def _load_model(config: OpenAIServerConfig) -> tuple[object, torch.device]:
    return load_model_for_engine(config)


def _infer_model_kind(config: OpenAIServerConfig) -> str:
    return _engine_infer_model_kind(config)


def _primary_device(config: OpenAIServerConfig) -> torch.device:
    return _engine_primary_device(config)


def _server_devices(config: OpenAIServerConfig) -> tuple[str, ...]:
    return _engine_server_devices(config)


def _llama_parallelism(config: OpenAIServerConfig) -> str:
    return _engine_llama_parallelism(config)


def _distributed_env_requested() -> bool:
    return _engine_distributed_env_requested()


def _should_reexec_distributed_server(config: OpenAIServerConfig) -> bool:
    return _engine_should_reexec_distributed_server(config)


def _distributed_server_command(config: OpenAIServerConfig, argv: Sequence[str]) -> list[str]:
    return _engine_distributed_server_command(config, argv)


def _reexec_distributed_server(config: OpenAIServerConfig, argv: Sequence[str]) -> None:
    command = _distributed_server_command(config, argv)
    print(
        "TorchInferno OpenAI server auto-launching tensor-parallel workers: "
        + " ".join(command),
        flush=True,
    )
    os.execvpe(command[0], command, os.environ.copy())


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    return _engine_resolve_dtype(dtype)


def _is_tensor_parallel_model(model: object) -> bool:
    return isinstance(model, Llama3TensorParallelForCausalLM)


def _tensor_parallel_world_size(model: object) -> int:
    return int(getattr(model, "world_size", 1)) if _is_tensor_parallel_model(model) else 1


def _tensor_parallel_symm_mem_allreduce_scope(
    model: object,
    device: torch.device,
    *,
    max_tokens: int,
    temperature: float,
) -> ContextManager[None]:
    if (
        not _is_tensor_parallel_model(model)
        or _tensor_parallel_world_size(model) <= 1
        or device.type != "cuda"
    ):
        return nullcontext()
    max_tokens_limit = env_int("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_TOKENS", 1024, minimum=1)
    if max_tokens > max_tokens_limit:
        return nullcontext()
    if not env_flag("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", True):
        return nullcontext()
    max_batch = env_int("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_BATCH", 64, minimum=1)
    return symm_mem_allreduce_max_batch(max_batch, enabled=True)


def _effective_openai_max_batch_size(model: object, device: torch.device, requested: int) -> int:
    max_batch_size = max(1, requested)
    if _is_tensor_parallel_model(model) and device.type == "cuda":
        tp_default = env_int("TORCHINFERNO_OPENAI_TP_MAX_BATCH_SIZE", 128, minimum=1)
        return min(max_batch_size, tp_default)
    return max_batch_size


def _tensor_parallel_control_group(dist: object) -> object | None:
    global _TENSOR_PARALLEL_CONTROL_GROUP
    if _TENSOR_PARALLEL_CONTROL_GROUP is not None:
        return _TENSOR_PARALLEL_CONTROL_GROUP
    with _TENSOR_PARALLEL_CONTROL_GROUP_LOCK:
        if _TENSOR_PARALLEL_CONTROL_GROUP is None:
            new_group = getattr(dist, "new_group", None)
            if new_group is None:
                return None
            try:
                _TENSOR_PARALLEL_CONTROL_GROUP = new_group(backend="gloo")
            except Exception as exc:
                warn_optional_failure("tensor-parallel control group", exc)
                return None
        return _TENSOR_PARALLEL_CONTROL_GROUP


def _is_tensor_parallel_primary_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) == 0


def _is_tensor_parallel_worker_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) != 0


def _prefix_cache_enabled_for_model(model: object) -> bool:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_flag("TORCHINFERNO_OPENAI_TP_PREFIX_CACHE", True)
    return True


def _shared_prefix_batch_enabled_for_model(model: object) -> bool:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_BATCH", True)
    return True


def _prefer_tensor_parallel_stream_group(
    prompts: Sequence[Sequence[int]],
    model: object,
) -> bool:
    if not _tensor_parallel_tensor_commands_enabled(model):
        return False
    if len(prompts) <= 1:
        return False
    prompt_len = len(prompts[0])
    return all(len(prompt) == prompt_len for prompt in prompts)


_TP_COMMAND_STOP = 0
_TP_COMMAND_GENERATE_TENSOR = 1
_TP_COMMAND_GENERATE_PROMPT_LISTS = 2
_TP_COMMAND_ONLINE_START = 3
_TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS = 4
_TP_COMMAND_ONLINE_STEP = 5
_TP_COMMAND_ONLINE_CLOSE = 6
_TP_COMMAND_CLEANUP = 7
_TP_COMMAND_TOKEN_BUDGET_STEP = 8
_TP_COMMAND_TOKEN_BUDGET_START = 9
_TP_COMMAND_TOKEN_BUDGET_CLOSE = 10
_TP_COMMAND_TOKEN_BUDGET_DECODE_RUN = 11
_TP_COMMAND_PROMPT_LIST_PERSISTENT_START = 12
_TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP = 13
_TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE = 14
_TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN = 15
_TP_COMMAND_META_FIELDS = 11
_TP_TOKEN_BUDGET_CHUNK_FIELDS = 10
_TP_TOKEN_BUDGET_DECODE_RUN_STEP_FIELDS = 5
_TP_TOKEN_BUDGET_KIND_PREFILL = 0
_TP_TOKEN_BUDGET_KIND_DECODE = 1
_TP_PROMPT_LIST_PERSISTENT_DECODE_FIELDS = 2
_TP_PROMPT_LIST_PERSISTENT_PREFILL_FIELDS = 6


def _runtime_prefill_graph_capture_enabled(
    model: object,
    temperature: float = 0.0,
    *,
    max_tokens: int | None = None,
) -> bool:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        capture_env_set = "TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE" in os.environ
        if not env_flag("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", True):
            return False
        if not capture_env_set and temperature > 0.0 and not env_flag(
            "TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE",
            True,
        ):
            return False
        if max_tokens is not None:
            short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
            sampled_skip_max_tokens = env_int(
                "TORCHINFERNO_OPENAI_TP_TEMPERATURE_PREFILL_CAPTURE_SKIP_MAX_TOKENS",
                max(short_max_tokens, 320),
                minimum=1,
            )
            if (
                not capture_env_set
                and temperature > 0.0
                and max_tokens <= sampled_skip_max_tokens
                and not env_flag("TORCHINFERNO_OPENAI_TP_SHORT_TEMPERATURE_PREFILL_CAPTURE", False)
            ):
                return False
            if not capture_env_set and temperature <= 0.0:
                skip_min_tokens = env_int(
                    "TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_SKIP_MIN_TOKENS",
                    1,
                    minimum=1,
                )
                skip_max_tokens = env_int(
                    "TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_SKIP_MAX_TOKENS",
                    256,
                    minimum=skip_min_tokens,
                )
                if (
                    skip_min_tokens <= max_tokens <= skip_max_tokens
                    and not env_flag("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_IN_SKIP_WINDOW", False)
                ):
                    return False
            token_limit = env_int("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", 1024, minimum=1)
            if max_tokens > token_limit:
                return False
        return True
    return True


def _identical_prompt_prefill_graph_capture_enabled(
    model: object,
    temperature: float = 0.0,
    *,
    max_tokens: int | None = None,
) -> bool:
    if not _runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens):
        return False
    if (
        temperature > 0.0
        and _is_tensor_parallel_model(model)
        and _tensor_parallel_world_size(model) > 1
    ):
        return env_flag("TORCHINFERNO_OPENAI_TP_IDENTICAL_TEMPERATURE_PREFILL_CAPTURE", False)
    return True


def _openai_cuda_graph_enabled_for_model(model: object) -> bool:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_flag("TORCHINFERNO_OPENAI_TP_CUDAGRAPH", True)
    return True


def _sync_tensor_parallel_command(
    model: object,
    device: torch.device,
    *,
    cuda_sync: bool | None = None,
) -> None:
    if not _is_tensor_parallel_model(model) or _tensor_parallel_world_size(model) <= 1:
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if device.type == "cuda":
        sync_cuda = env_flag("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC") if cuda_sync is None else cuda_sync
        if sync_cuda:
            torch.cuda.synchronize(device)
        control_group = _tensor_parallel_control_group(dist)
        if control_group is not None:
            dist.barrier(group=control_group)
        if sync_cuda:
            torch.cuda.synchronize(device)
        return
    dist.barrier()


def _tp_command_cuda_sync_for_steps(completed_steps: int, *, emitted_tokens: int = 0) -> bool:
    if env_flag("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC"):
        return True
    min_steps = env_int("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", 8, minimum=1)
    if completed_steps < min_steps:
        return False
    skip_emitted_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_SKIP_MIN_EMITTED_TOKENS",
        512,
        minimum=1,
    )
    if emitted_tokens >= skip_emitted_tokens:
        return False
    if "TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MAX_STEPS" in os.environ:
        max_steps = env_int("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MAX_STEPS", 32, minimum=min_steps)
        return completed_steps <= max_steps
    return True


def _sync_tensor_parallel_continue(model: object, should_continue: bool, device: torch.device) -> bool:
    if not _is_tensor_parallel_model(model) or _tensor_parallel_world_size(model) <= 1:
        return should_continue
    if not env_flag("TORCHINFERNO_OPENAI_SYNC_TP_CONTINUE"):
        return should_continue
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return should_continue
    flag = torch.tensor([1 if should_continue else 0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _tensor_parallel_all_ranks_true(model: object, value: bool, device: torch.device) -> bool:
    if not _is_tensor_parallel_model(model) or _tensor_parallel_world_size(model) <= 1:
        return value
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    control_group = _tensor_parallel_control_group(dist)
    if control_group is not None:
        flag = torch.tensor([1 if value else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=control_group)
        return bool(flag.item())
    flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _tensor_parallel_all_ranks_same_int(model: object, value: int, device: torch.device) -> bool:
    if not _is_tensor_parallel_model(model) or _tensor_parallel_world_size(model) <= 1:
        return True
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return True
    control_group = _tensor_parallel_control_group(dist)
    if control_group is not None:
        low = torch.tensor([int(value)], dtype=torch.int64)
        high = low.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=control_group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=control_group)
        return bool(low.item() == high.item())
    low = torch.tensor([int(value)], dtype=torch.int64, device=device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return bool(low.item() == high.item())


def _tensor_parallel_tensor_commands_enabled(model: object) -> bool:
    if (
        not _is_tensor_parallel_model(model)
        or _tensor_parallel_world_size(model) <= 1
        or not env_flag("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", True)
    ):
        return False
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def _tensor_parallel_command_device(device: torch.device) -> torch.device:
    command_device = torch.device(device)
    return command_device if command_device.type == "cuda" else torch.device("cpu")


def _tensor_parallel_tensor_command_group(dist_module: object) -> object | None:
    if not env_flag("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", True):
        return None
    return _tensor_parallel_control_group(dist_module)


def _tensor_parallel_tensor_command_device(device: torch.device, group: object | None) -> torch.device:
    if group is not None:
        return torch.device("cpu")
    return _tensor_parallel_command_device(device)


def _broadcast_tensor_command(tensor: Tensor, *, src: int, group: object | None) -> None:
    import torch.distributed as dist

    if group is None:
        dist.broadcast(tensor, src=src)
    else:
        dist.broadcast(tensor, src=src, group=group)


def _broadcast_tensor_parallel_tensor_payload(
    model: object,
    *,
    command_kind: int,
    token_rows: Tensor,
    lengths: Tensor,
    max_tokens: int,
    temperature: float,
    stream: bool,
    row_max_tokens: Sequence[int] | None,
    command_handle: int = 0,
) -> bool:
    if not _tensor_parallel_tensor_commands_enabled(model):
        return False
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False
    command_group = _tensor_parallel_tensor_command_group(dist)
    command_device = _tensor_parallel_tensor_command_device(token_rows.device, command_group)
    token_rows = token_rows.to(command_device, non_blocking=True).contiguous()
    lengths = lengths.to(command_device, non_blocking=True).contiguous()
    row_max = (
        torch.tensor([int(value) for value in row_max_tokens], dtype=torch.long, device=command_device)
        if row_max_tokens is not None
        else torch.empty(0, dtype=torch.long, device=command_device)
    )
    meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
    meta[0] = int(command_kind)
    meta[1] = int(bool(stream))
    meta[2] = int(token_rows.size(0))
    meta[3] = int(token_rows.size(1))
    meta[4] = int(max_tokens)
    meta[5] = int(row_max.numel() > 0)
    meta[6] = int(command_handle)
    temp = torch.tensor([float(temperature)], dtype=torch.float64, device=command_device)
    _broadcast_tensor_command(meta, src=0, group=command_group)
    _broadcast_tensor_command(temp, src=0, group=command_group)
    _broadcast_tensor_command(lengths, src=0, group=command_group)
    _broadcast_tensor_command(token_rows, src=0, group=command_group)
    if row_max.numel() > 0:
        _broadcast_tensor_command(row_max, src=0, group=command_group)
    return True


def _receive_tensor_parallel_tensor_payload(engine: OpenAICompletionEngine) -> dict[str, object]:
    import torch.distributed as dist

    command_group = _tensor_parallel_tensor_command_group(dist)
    command_device = _tensor_parallel_tensor_command_device(engine.device, command_group)
    meta = torch.empty(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
    _broadcast_tensor_command(meta, src=0, group=command_group)
    command_kind = int(meta[0].item())
    if command_kind == _TP_COMMAND_STOP:
        return {"op": "stop"}
    if command_kind == _TP_COMMAND_ONLINE_STEP:
        return {"op": "online_step", "steps": max(1, int(meta[4].item()))}
    if command_kind == _TP_COMMAND_ONLINE_CLOSE:
        return {"op": "online_close"}
    if command_kind == _TP_COMMAND_CLEANUP:
        return {"op": "cleanup"}
    if command_kind == _TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE:
        return {"op": "persistent_prompt_list_close"}
    if command_kind == _TP_COMMAND_TOKEN_BUDGET_CLOSE:
        return {"op": "token_budget_close"}
    if command_kind == _TP_COMMAND_ONLINE_START:
        temp = torch.empty(1, dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        prefill_budget = int(meta[9].item())
        return {
            "op": "online_start",
            "max_seq_len": int(meta[6].item()),
            "max_active_requests": int(meta[7].item()),
            "prefix_cache_capacity": int(meta[8].item()),
            "prefill_token_budget": prefill_budget,
            "temperature": float(temp.item()),
            "enable_ragged_decode": bool(int(meta[5].item())),
            "store_reusable_prefixes": bool(int(meta[10].item())),
            "store_full_prompt_prefixes": bool(int(meta[3].item())),
            "max_tokens": int(meta[4].item()),
        }
    if command_kind == _TP_COMMAND_TOKEN_BUDGET_START:
        temp = torch.empty(1, dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        prefix_len = int(meta[5].item())
        prefix: list[int] = []
        if prefix_len > 0:
            prefix_tensor = torch.empty(prefix_len, dtype=torch.long, device=command_device)
            _broadcast_tensor_command(prefix_tensor, src=0, group=command_group)
            prefix = [int(token_id) for token_id in prefix_tensor.cpu().tolist()]
        return {
            "op": "token_budget_start",
            "max_seq_len": int(meta[6].item()),
            "max_active_rows": int(meta[7].item()),
            "temperature": float(temp.item()),
            "max_tokens": int(meta[4].item()),
            "prefix": prefix,
        }
    if command_kind == _TP_COMMAND_PROMPT_LIST_PERSISTENT_START:
        temp = torch.empty(1, dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        prefix_len = int(meta[5].item())
        prefix: list[int] = []
        if prefix_len > 0:
            prefix_tensor = torch.empty(prefix_len, dtype=torch.long, device=command_device)
            _broadcast_tensor_command(prefix_tensor, src=0, group=command_group)
            prefix = [int(token_id) for token_id in prefix_tensor.cpu().tolist()]
        return {
            "op": "persistent_prompt_list_start",
            "prefix": prefix,
            "cache_batch_size": int(meta[7].item()),
            "max_seq_len": int(meta[6].item()),
            "temperature": float(temp.item()),
            "max_tokens": int(meta[4].item()),
        }
    if command_kind == _TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP:
        decode_count = int(meta[2].item())
        prefill_count = int(meta[3].item())
        prompt_width = int(meta[4].item())
        finished_count = int(meta[5].item())
        decode_width = int(meta[7].item())
        prefill_width = int(meta[6].item())
        temp = torch.empty(1, dtype=torch.float64, device=command_device)
        decode_tensor = torch.empty((decode_count, decode_width), dtype=torch.long, device=command_device)
        prefill_tensor = torch.empty((prefill_count, prefill_width), dtype=torch.long, device=command_device)
        prompt_lengths = torch.empty(prefill_count, dtype=torch.long, device=command_device)
        prompt_rows = torch.empty((prefill_count, prompt_width), dtype=torch.long, device=command_device)
        finished_ids = torch.empty(finished_count, dtype=torch.long, device=command_device)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        _broadcast_tensor_command(decode_tensor, src=0, group=command_group)
        _broadcast_tensor_command(prefill_tensor, src=0, group=command_group)
        _broadcast_tensor_command(prompt_lengths, src=0, group=command_group)
        _broadcast_tensor_command(prompt_rows, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return _persistent_prompt_list_step_payload_from_tensor_payload(
            meta,
            temp,
            decode_tensor,
            prefill_tensor,
            prompt_lengths,
            prompt_rows,
            finished_ids,
        )
    if command_kind == _TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN:
        temp = torch.empty(1, dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        return _persistent_prompt_list_decode_run_payload_from_tensor_payload(meta, temp)
    if command_kind == _TP_COMMAND_TOKEN_BUDGET_STEP:
        rows = int(meta[2].item())
        width = int(meta[3].item())
        prefill_count = int(meta[5].item())
        prefill_width = int(meta[6].item())
        finished_count = int(meta[8].item())
        chunk_tensor = torch.empty((rows, width), dtype=torch.long, device=command_device)
        prefill_lengths = torch.empty(prefill_count, dtype=torch.long, device=command_device)
        prefill_token_rows = torch.empty((prefill_count, prefill_width), dtype=torch.long, device=command_device)
        finished_ids = torch.empty(finished_count, dtype=torch.long, device=command_device)
        _broadcast_tensor_command(chunk_tensor, src=0, group=command_group)
        _broadcast_tensor_command(prefill_lengths, src=0, group=command_group)
        _broadcast_tensor_command(prefill_token_rows, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return _token_budget_step_payload_from_tensor_payload(
            meta,
            chunk_tensor,
            prefill_lengths,
            prefill_token_rows,
            finished_ids,
        )
    if command_kind == _TP_COMMAND_TOKEN_BUDGET_DECODE_RUN:
        step_count = int(meta[1].item())
        rows = int(meta[2].item())
        width = int(meta[3].item())
        finished_count = int(meta[4].item())
        step_width = int(meta[5].item())
        step_tensor = torch.empty((step_count, step_width), dtype=torch.long, device=command_device)
        chunk_tensor = torch.empty((rows, width), dtype=torch.long, device=command_device)
        finished_ids = torch.empty(finished_count, dtype=torch.long, device=command_device)
        _broadcast_tensor_command(step_tensor, src=0, group=command_group)
        _broadcast_tensor_command(chunk_tensor, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return _token_budget_decode_run_payload_from_tensor_payload(
            meta,
            step_tensor,
            chunk_tensor,
            finished_ids,
        )
    if command_kind not in {
        _TP_COMMAND_GENERATE_TENSOR,
        _TP_COMMAND_GENERATE_PROMPT_LISTS,
        _TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS,
    }:
        raise ValueError(f"unsupported tensor-parallel tensor command: {command_kind}")

    stream = bool(meta[1].item())
    rows = int(meta[2].item())
    width = int(meta[3].item())
    max_tokens = int(meta[4].item())
    has_row_max_tokens = bool(meta[5].item())
    temp = torch.empty(1, dtype=torch.float64, device=command_device)
    lengths = torch.empty(rows, dtype=torch.long, device=command_device)
    token_rows = torch.empty((rows, width), dtype=torch.long, device=command_device)
    _broadcast_tensor_command(temp, src=0, group=command_group)
    _broadcast_tensor_command(lengths, src=0, group=command_group)
    _broadcast_tensor_command(token_rows, src=0, group=command_group)
    row_max_tokens = None
    if has_row_max_tokens:
        row_max = torch.empty(rows, dtype=torch.long, device=command_device)
        _broadcast_tensor_command(row_max, src=0, group=command_group)
        row_max_tokens = [int(value) for value in row_max.detach().cpu().tolist()]

    if command_kind == _TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS:
        lengths_list = [int(value) for value in lengths.detach().cpu().tolist()]
        token_rows_cpu = token_rows.detach().cpu()
        eos_token_id = int(meta[7].item())
        return {
            "op": "online_submit",
            "input_id_lists": [
                token_rows_cpu[row, :length].tolist()
                for row, length in enumerate(lengths_list)
            ],
            "max_tokens": max_tokens,
            "row_max_tokens": row_max_tokens,
            "arrival_step": int(meta[6].item()),
            "eos_token_id": None if eos_token_id < 0 else eos_token_id,
            "request_id_start": int(meta[8].item()),
        }

    payload: dict[str, object] = {
        "op": "generate",
        "max_tokens": max_tokens,
        "row_max_tokens": row_max_tokens,
        "temperature": float(temp.item()),
        "stream": stream,
    }
    if command_kind == _TP_COMMAND_GENERATE_TENSOR:
        payload["input_ids_tensor"] = token_rows.to(engine.device, non_blocking=True)
    else:
        lengths_list = [int(value) for value in lengths.detach().cpu().tolist()]
        token_rows_cpu = token_rows.detach().cpu()
        payload["input_id_lists"] = [
            token_rows_cpu[row, :length].tolist()
            for row, length in enumerate(lengths_list)
        ]
    return payload


def _broadcast_tensor_parallel_generate(
    model: object,
    input_ids: Tensor,
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
    row_max_tokens: Sequence[int] | None = None,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    token_rows = input_ids.detach().contiguous()
    lengths = torch.full(
        (token_rows.size(0),),
        token_rows.size(1),
        dtype=torch.long,
        device=token_rows.device,
    )
    if _broadcast_tensor_parallel_tensor_payload(
        model,
        command_kind=_TP_COMMAND_GENERATE_TENSOR,
        token_rows=token_rows,
        lengths=lengths,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        row_max_tokens=row_max_tokens,
    ):
        return
    command = [
        {
            "op": "generate",
            "input_ids": input_ids.detach().cpu().tolist(),
            "max_tokens": int(max_tokens),
            "row_max_tokens": None if row_max_tokens is None else [int(value) for value in row_max_tokens],
            "temperature": float(temperature),
            "stream": bool(stream),
        }
    ]
    dist.broadcast_object_list(command, src=0)


def _broadcast_tensor_parallel_generate_prompt_lists(
    model: object,
    prompts: Sequence[Sequence[int]],
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
    row_max_tokens: Sequence[int] | None = None,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if prompts:
        token_rows, lengths = _prompt_list_tensor_payload(prompts, torch.device("cpu"))
        if _broadcast_tensor_parallel_tensor_payload(
            model,
            command_kind=_TP_COMMAND_GENERATE_PROMPT_LISTS,
            token_rows=token_rows,
            lengths=lengths,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            row_max_tokens=row_max_tokens,
        ):
            return
    command = [
        {
            "op": "generate",
            "input_id_lists": [list(prompt) for prompt in prompts],
            "max_tokens": int(max_tokens),
            "row_max_tokens": None if row_max_tokens is None else [int(value) for value in row_max_tokens],
            "temperature": float(temperature),
            "stream": bool(stream),
        }
    ]
    dist.broadcast_object_list(command, src=0)


def _broadcast_tensor_parallel_token_budget_prompt_list_run(
    model: object,
    prompts: Sequence[Sequence[int]],
    *,
    max_tokens: int,
    temperature: float,
    row_max_tokens: Sequence[int] | None,
    prefix_tokens: int,
    max_active_rows: int,
    max_scheduled_tokens: int,
    prefill_chunk_size: int | None = None,
    decode_run_steps: int = 1,
    arrival_steps: Sequence[int] | None = None,
    static_graph_buckets: bool = False,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    command = [
        {
            "op": "token_budget_prompt_list_run",
            "input_id_lists": [list(prompt) for prompt in prompts],
            "max_tokens": int(max_tokens),
            "row_max_tokens": None if row_max_tokens is None else [int(value) for value in row_max_tokens],
            "temperature": float(temperature),
            "prefix_tokens": int(prefix_tokens),
            "max_active_rows": int(max_active_rows),
            "max_scheduled_tokens": int(max_scheduled_tokens),
            "prefill_chunk_size": 0 if prefill_chunk_size is None else int(prefill_chunk_size),
            "decode_run_steps": int(decode_run_steps),
            "arrival_steps": None if arrival_steps is None else [int(value) for value in arrival_steps],
            "static_graph_buckets": bool(static_graph_buckets),
        }
    ]
    dist.broadcast_object_list(command, src=0)


def _broadcast_tensor_parallel_persistent_prompt_list_step(
    model: object,
    payload: Mapping[str, object],
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta, temp, decode_rows, prefill_rows, prompt_lengths, prompt_token_rows, finished_ids = (
            _persistent_prompt_list_step_tensor_payload(payload, command_device)
        )
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        _broadcast_tensor_command(decode_rows, src=0, group=command_group)
        _broadcast_tensor_command(prefill_rows, src=0, group=command_group)
        _broadcast_tensor_command(prompt_lengths, src=0, group=command_group)
        _broadcast_tensor_command(prompt_token_rows, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return
    command = dict(payload)
    command["op"] = "persistent_prompt_list_step"
    dist.broadcast_object_list([command], src=0)


def _broadcast_tensor_parallel_persistent_prompt_list_decode_run(
    model: object,
    payload: Mapping[str, object],
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta, temp = _persistent_prompt_list_decode_run_tensor_payload(payload, command_device)
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        return
    command = dict(payload)
    command["op"] = "persistent_prompt_list_decode_run"
    dist.broadcast_object_list([command], src=0)


def _broadcast_tensor_parallel_persistent_prompt_list_start(
    model: object,
    *,
    prefix: Sequence[int],
    cache_batch_size: int,
    max_seq_len: int,
    temperature: float,
    max_tokens: int = 0,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        prefix_tensor = torch.tensor([int(token_id) for token_id in prefix], dtype=torch.long, device=command_device)
        meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
        meta[0] = _TP_COMMAND_PROMPT_LIST_PERSISTENT_START
        meta[4] = int(max_tokens)
        meta[5] = int(prefix_tensor.numel())
        meta[6] = int(max_seq_len)
        meta[7] = int(cache_batch_size)
        temp = torch.tensor([float(temperature)], dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        if prefix_tensor.numel() > 0:
            _broadcast_tensor_command(prefix_tensor, src=0, group=command_group)
        return
    dist.broadcast_object_list(
        [
            {
                "op": "persistent_prompt_list_start",
                "prefix": [int(token_id) for token_id in prefix],
                "cache_batch_size": int(cache_batch_size),
                "max_seq_len": int(max_seq_len),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
        ],
        src=0,
    )


def _broadcast_tensor_parallel_persistent_prompt_list_close(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
        meta[0] = _TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE
        _broadcast_tensor_command(meta, src=0, group=command_group)
        return
    dist.broadcast_object_list([{"op": "persistent_prompt_list_close"}], src=0)


def _broadcast_tensor_parallel_online_start(
    model: object,
    *,
    max_seq_len: int,
    max_active_requests: int,
    prefix_cache_capacity: int,
    prefill_token_budget: int | None,
    temperature: float,
    enable_ragged_decode: bool = True,
    store_reusable_prefixes: bool = True,
    store_full_prompt_prefixes: bool = True,
    max_tokens: int = 0,
) -> None:
    _broadcast_tensor_parallel_online_command(
        model,
        {
            "op": "online_start",
            "max_seq_len": int(max_seq_len),
            "max_active_requests": int(max_active_requests),
            "prefix_cache_capacity": int(prefix_cache_capacity),
            "prefill_token_budget": 0 if prefill_token_budget is None else int(prefill_token_budget),
            "temperature": float(temperature),
            "enable_ragged_decode": bool(enable_ragged_decode),
            "store_reusable_prefixes": bool(store_reusable_prefixes),
            "store_full_prompt_prefixes": bool(store_full_prompt_prefixes),
            "max_tokens": int(max_tokens),
        },
    )


def _broadcast_tensor_parallel_online_submit_prompt_lists(
    model: object,
    prompts: Sequence[Sequence[int]],
    *,
    max_tokens: int,
    row_max_tokens: Sequence[int] | None,
    arrival_step: int,
    eos_token_id: int | None,
    request_id_start: int = 0,
) -> None:
    _broadcast_tensor_parallel_online_command(
        model,
        {
            "op": "online_submit",
            "input_id_lists": [list(prompt) for prompt in prompts],
            "max_tokens": int(max_tokens),
            "row_max_tokens": None if row_max_tokens is None else [int(value) for value in row_max_tokens],
            "arrival_step": int(arrival_step),
            "eos_token_id": eos_token_id,
            "request_id_start": int(request_id_start),
        },
    )


def _broadcast_tensor_parallel_online_step(model: object, steps: int = 1) -> None:
    payload: dict[str, object] = {"op": "online_step"}
    if steps != 1:
        payload["steps"] = int(steps)
    _broadcast_tensor_parallel_online_command(model, payload)


def _broadcast_tensor_parallel_online_close(model: object) -> None:
    _broadcast_tensor_parallel_online_command(model, {"op": "online_close"})


def _broadcast_tensor_parallel_token_budget_start(
    model: object,
    *,
    max_seq_len: int,
    max_active_rows: int,
    temperature: float,
    max_tokens: int = 0,
    prefix: Sequence[int] = (),
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    payload = {
        "op": "token_budget_start",
        "max_seq_len": int(max_seq_len),
        "max_active_rows": int(max_active_rows),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "prefix": [int(token_id) for token_id in prefix],
    }
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        prefix_tensor = torch.tensor([int(token_id) for token_id in prefix], dtype=torch.long, device=command_device)
        meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
        meta[0] = _TP_COMMAND_TOKEN_BUDGET_START
        meta[4] = int(max_tokens)
        meta[5] = int(prefix_tensor.numel())
        meta[6] = int(max_seq_len)
        meta[7] = int(max_active_rows)
        temp = torch.tensor([float(temperature)], dtype=torch.float64, device=command_device)
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(temp, src=0, group=command_group)
        if prefix_tensor.numel() > 0:
            _broadcast_tensor_command(prefix_tensor, src=0, group=command_group)
        return
    dist.broadcast_object_list([payload], src=0)


def _broadcast_tensor_parallel_token_budget_close(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
        meta[0] = _TP_COMMAND_TOKEN_BUDGET_CLOSE
        _broadcast_tensor_command(meta, src=0, group=command_group)
        return
    dist.broadcast_object_list([{"op": "token_budget_close"}], src=0)


@contextmanager
def _tensor_parallel_token_budget_lifecycle(
    model: object,
    *,
    max_seq_len: int,
    max_active_rows: int,
    temperature: float,
    max_tokens: int = 0,
    prefix: Sequence[int] = (),
) -> Iterator[None]:
    _broadcast_tensor_parallel_token_budget_start(
        model,
        max_seq_len=max_seq_len,
        max_active_rows=max_active_rows,
        temperature=temperature,
        max_tokens=max_tokens,
        prefix=prefix,
    )
    try:
        yield
    finally:
        _broadcast_tensor_parallel_token_budget_close(model)


def _broadcast_tensor_parallel_token_budget_step(model: object, payload: Mapping[str, object]) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta, chunks, prefill_lengths, prefill_token_rows, finished_ids = _token_budget_step_tensor_payload(
            payload,
            command_device,
        )
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(chunks, src=0, group=command_group)
        _broadcast_tensor_command(prefill_lengths, src=0, group=command_group)
        _broadcast_tensor_command(prefill_token_rows, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return
    dist.broadcast_object_list([dict(payload)], src=0)


def _broadcast_tensor_parallel_token_budget_decode_run(model: object, payload: Mapping[str, object]) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta, step_tensor, chunk_tensor, finished_ids = _token_budget_decode_run_tensor_payload(
            payload,
            command_device,
        )
        _broadcast_tensor_command(meta, src=0, group=command_group)
        _broadcast_tensor_command(step_tensor, src=0, group=command_group)
        _broadcast_tensor_command(chunk_tensor, src=0, group=command_group)
        _broadcast_tensor_command(finished_ids, src=0, group=command_group)
        return
    dist.broadcast_object_list([dict(payload)], src=0)


def _broadcast_tensor_parallel_online_command(model: object, payload: Mapping[str, object]) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        command_device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        op = payload.get("op")
        if op == "online_start":
            meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
            meta[0] = _TP_COMMAND_ONLINE_START
            meta[3] = int(bool(payload.get("store_full_prompt_prefixes", True)))
            meta[4] = int(payload.get("max_tokens", 0))
            meta[5] = int(bool(payload.get("enable_ragged_decode", True)))
            meta[6] = int(payload["max_seq_len"])
            meta[7] = int(payload["max_active_requests"])
            meta[8] = int(payload["prefix_cache_capacity"])
            meta[9] = int(payload.get("prefill_token_budget") or 0)
            meta[10] = int(bool(payload.get("store_reusable_prefixes", True)))
            temp = torch.tensor([float(payload.get("temperature", 0.0))], dtype=torch.float64, device=command_device)
            _broadcast_tensor_command(meta, src=0, group=command_group)
            _broadcast_tensor_command(temp, src=0, group=command_group)
            return
        if op == "online_step":
            meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
            meta[0] = _TP_COMMAND_ONLINE_STEP
            meta[4] = max(1, int(payload.get("steps", 1)))
            _broadcast_tensor_command(meta, src=0, group=command_group)
            return
        if op == "online_close":
            meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
            meta[0] = _TP_COMMAND_ONLINE_CLOSE
            _broadcast_tensor_command(meta, src=0, group=command_group)
            return
        if op == "online_submit":
            prompts = payload.get("input_id_lists")
            if isinstance(prompts, list):
                token_rows, lengths = _prompt_list_tensor_payload(prompts, command_device)
                row_max = _coerce_optional_int_sequence(payload.get("row_max_tokens"))
                row_max_tensor = (
                    torch.tensor(row_max, dtype=torch.long, device=command_device)
                    if row_max is not None
                    else torch.empty(0, dtype=torch.long, device=command_device)
                )
                meta = torch.tensor(
                    [
                        _TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS,
                        1,
                        int(token_rows.size(0)),
                        int(token_rows.size(1)),
                        int(payload.get("max_tokens", 0)),
                        int(row_max_tensor.numel() > 0),
                        int(payload.get("arrival_step", 0)),
                        -1 if payload.get("eos_token_id") is None else int(payload["eos_token_id"]),
                        int(payload.get("request_id_start", 0)),
                        0,
                    ],
                    dtype=torch.long,
                    device=command_device,
                )
                temp = torch.zeros(1, dtype=torch.float64, device=command_device)
                _broadcast_tensor_command(meta, src=0, group=command_group)
                _broadcast_tensor_command(temp, src=0, group=command_group)
                _broadcast_tensor_command(lengths, src=0, group=command_group)
                _broadcast_tensor_command(token_rows, src=0, group=command_group)
                if row_max_tensor.numel() > 0:
                    _broadcast_tensor_command(row_max_tensor, src=0, group=command_group)
                return
    dist.broadcast_object_list([dict(payload)], src=0)


def _broadcast_tensor_parallel_stop(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if _tensor_parallel_tensor_commands_enabled(model):
            command_group = _tensor_parallel_tensor_command_group(dist)
            device = _tensor_parallel_tensor_command_device(
                getattr(model, "device", torch.device("cpu")),
                command_group,
            )
            meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
            meta[0] = _TP_COMMAND_STOP
            _broadcast_tensor_command(meta, src=0, group=command_group)
            return
        dist.broadcast_object_list([{"op": "stop"}], src=0)


def _broadcast_tensor_parallel_cleanup(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    if _tensor_parallel_tensor_commands_enabled(model):
        command_group = _tensor_parallel_tensor_command_group(dist)
        device = _tensor_parallel_tensor_command_device(
            getattr(model, "device", torch.device("cpu")),
            command_group,
        )
        meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
        meta[0] = _TP_COMMAND_CLEANUP
        _broadcast_tensor_command(meta, src=0, group=command_group)
        return
    dist.broadcast_object_list([{"op": "cleanup"}], src=0)


def _prompt_list_tensor_payload(
    prompts: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    lengths_values = [len(prompt) for prompt in prompts]
    token_rows = _padded_token_rows_tensor(prompts, device, 0)
    lengths = torch.tensor(lengths_values, dtype=torch.long, device=device)
    return token_rows, lengths


def _padded_token_rows_tensor(
    rows: Sequence[Sequence[int]],
    device: torch.device,
    pad_token_id: int,
) -> Tensor:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return torch.empty((len(rows), 0), dtype=torch.long, device=device)
    padded = [
        [int(token_id) for token_id in row] + [int(pad_token_id)] * (width - len(row))
        for row in rows
    ]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _tensor_parallel_worker_loop(engine: OpenAICompletionEngine) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("tensor-parallel worker loop requires an initialized process group")
    online_runtime_engine: _RuntimeContinuousBatchEngine | None = None
    online_symm_scope: ContextManager[None] | None = None
    persistent_prompt_list_symm_scope: ContextManager[None] | None = None
    token_budget_symm_scope: ContextManager[None] | None = None
    _skip_finally_sync = False
    while True:
        cuda_sync: bool | None = None
        _skip_finally_sync = False
        if _tensor_parallel_tensor_commands_enabled(getattr(engine, "model", None)):
            payload = _receive_tensor_parallel_tensor_payload(engine)
        else:
            command: list[object] = [None]
            dist.broadcast_object_list(command, src=0)
            payload = command[0]
        if not isinstance(payload, dict):
            continue
        op = payload.get("op")
        if op == "stop":
            if online_symm_scope is not None:
                online_symm_scope.__exit__(None, None, None)
            if persistent_prompt_list_symm_scope is not None:
                persistent_prompt_list_symm_scope.__exit__(None, None, None)
            if token_budget_symm_scope is not None:
                token_budget_symm_scope.__exit__(None, None, None)
            return
        cuda_sync_value = payload.get("cuda_sync")
        if isinstance(cuda_sync_value, bool):
            cuda_sync = cuda_sync_value
        try:
            if op == "cleanup":
                engine._clear_runtime_state_after_idle()
                continue
            if op == "flashinfer_start":
                _skip_finally_sync = True
                fi_max_batch = int(payload.get("max_batch", 64))
                fi_max_seq_len = int(payload.get("max_seq_len", 768))
                fi_temperature = float(payload.get("temperature", 0.0))
                fi_max_tokens = int(payload.get("max_tokens", 1))
                fi_cache = getattr(engine, "model").allocate_cache(
                    fi_max_batch, fi_max_seq_len, cache_backend="flashinfer"
                )
                fi_symm_scope = _tensor_parallel_symm_mem_allreduce_scope(
                    getattr(engine, "model", None),
                    getattr(engine, "device", torch.device("cpu")),
                    max_tokens=fi_max_tokens,
                    temperature=fi_temperature,
                )
                fi_symm_scope.__enter__()
                # Loop receiving FlashInfer step commands
                while True:
                    step_cmd: list[object] = [None]
                    dist.broadcast_object_list(step_cmd, src=0)
                    step_payload = step_cmd[0]
                    if not isinstance(step_payload, dict):
                        continue
                    step_op = step_payload.get("op")
                    if step_op == "flashinfer_close":
                        break
                    if step_op == "flashinfer_step":
                        batch_sz = int(step_payload["batch"])
                        max_q_len = int(step_payload["max_q_len"])
                        device = getattr(engine, "device", torch.device("cpu"))
                        input_ids = torch.empty(batch_sz, max_q_len, dtype=torch.long, device=device)
                        q_lens = torch.empty(batch_sz, dtype=torch.long, device=device)
                        write_positions = torch.empty(batch_sz, max_q_len, dtype=torch.long, device=device)
                        seq_lens = torch.empty(batch_sz, dtype=torch.long, device=device)
                        logit_positions = torch.empty(batch_sz, dtype=torch.long, device=device)
                        dist.broadcast(input_ids, src=0)
                        dist.broadcast(q_lens, src=0)
                        dist.broadcast(write_positions, src=0)
                        dist.broadcast(seq_lens, src=0)
                        dist.broadcast(logit_positions, src=0)
                        getattr(engine, "model").forward_step_flashinfer(
                            input_ids, fi_cache,
                            seq_lens=seq_lens, q_lens=q_lens,
                            write_positions=write_positions,
                            logit_positions=logit_positions,
                        )
                fi_symm_scope.__exit__(None, None, None)
                continue
            if op == "persistent_prompt_list_start":
                if persistent_prompt_list_symm_scope is not None:
                    persistent_prompt_list_symm_scope.__exit__(None, None, None)
                    persistent_prompt_list_symm_scope = None
                prefix = payload.get("prefix", [])
                if not isinstance(prefix, list):
                    raise ValueError("persistent prompt-list start requires prefix")
                max_seq_len = int(payload["max_seq_len"])
                temperature = float(payload.get("temperature", 0.0))
                engine._start_persistent_prompt_list_step_state(
                    prefix=[int(token_id) for token_id in prefix],
                    cache_batch_size=int(payload["cache_batch_size"]),
                    max_seq_len=max_seq_len,
                    temperature=temperature,
                    max_tokens=int(payload.get("max_tokens", max(0, max_seq_len - len(prefix)))),
                )
                max_tokens = int(payload.get("max_tokens", max(0, max_seq_len - len(prefix))))
                persistent_prompt_list_symm_scope = _tensor_parallel_symm_mem_allreduce_scope(
                    getattr(engine, "model", None),
                    getattr(engine, "device", torch.device("cpu")),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                persistent_prompt_list_symm_scope.__enter__()
                continue
            if op == "token_budget_start":
                _skip_finally_sync = True
                if token_budget_symm_scope is not None:
                    token_budget_symm_scope.__exit__(None, None, None)
                    token_budget_symm_scope = None
                handler = getattr(engine, "_handle_token_budget_start_payload", None)
                if not callable(handler):
                    raise RuntimeError("token-budget start handler is not installed")
                handler(payload)
                temperature = float(payload.get("temperature", 0.0))
                token_budget_symm_scope = _tensor_parallel_symm_mem_allreduce_scope(
                    getattr(engine, "model", None),
                    getattr(engine, "device", torch.device("cpu")),
                    max_tokens=int(payload.get("max_tokens", 0)),
                    temperature=temperature,
                )
                token_budget_symm_scope.__enter__()
                _sync_tensor_parallel_command(
                    getattr(engine, "model", None),
                    getattr(engine, "device", torch.device("cpu")),
                )
                continue
            if op == "persistent_prompt_list_step":
                handler = getattr(engine, "_handle_persistent_prompt_list_step_payload", None)
                if not callable(handler):
                    raise RuntimeError("persistent prompt-list step handler is not installed")
                handler(payload)
                continue
            if op == "persistent_prompt_list_decode_run":
                handler = getattr(engine, "_handle_persistent_prompt_list_decode_run_payload", None)
                if not callable(handler):
                    raise RuntimeError("persistent prompt-list decode-run handler is not installed")
                handler(payload)
                continue
            if op == "token_budget_step":
                _skip_finally_sync = True
                handler = getattr(engine, "_handle_token_budget_step_payload", None)
                if not callable(handler):
                    raise RuntimeError("token-budget step handler is not installed")
                handler(payload)
                continue
            if op == "token_budget_decode_run":
                _skip_finally_sync = True
                handler = getattr(engine, "_handle_token_budget_decode_run_payload", None)
                if not callable(handler):
                    raise RuntimeError("token-budget decode-run handler is not installed")
                handler(payload)
                continue
            if op == "token_budget_prompt_list_run":
                handler = getattr(engine, "_handle_token_budget_prompt_list_run_payload", None)
                if not callable(handler):
                    raise RuntimeError("token-budget prompt-list run handler is not installed")
                temperature = float(payload.get("temperature", 0.0))
                with _tensor_parallel_symm_mem_allreduce_scope(
                    getattr(engine, "model", None),
                    getattr(engine, "device", torch.device("cpu")),
                    max_tokens=int(payload.get("max_tokens", 0)),
                    temperature=temperature,
                ):
                    handler(payload)
                continue
            if op == "persistent_prompt_list_close":
                engine._close_persistent_prompt_list_step_state()
                if persistent_prompt_list_symm_scope is not None:
                    persistent_prompt_list_symm_scope.__exit__(None, None, None)
                    persistent_prompt_list_symm_scope = None
                continue
            if op == "token_budget_close":
                handler = getattr(engine, "_handle_token_budget_close_payload", None)
                if not callable(handler):
                    raise RuntimeError("token-budget close handler is not installed")
                handler(payload)
                if token_budget_symm_scope is not None:
                    token_budget_symm_scope.__exit__(None, None, None)
                    token_budget_symm_scope = None
                continue
            if op == "online_start":
                if online_symm_scope is not None:
                    online_symm_scope.__exit__(None, None, None)
                    online_symm_scope = None
                online_runtime_engine = None
                max_seq_len = int(payload["max_seq_len"])
                max_active = int(payload.get("max_active_requests", 1))
                prefix_rows = int(payload.get("prefix_cache_capacity", 0))
                prefill_budget_value = int(payload.get("prefill_token_budget", 0))
                temperature = float(payload.get("temperature", 0.0))
                online_runtime_engine = _RuntimeContinuousBatchEngine(
                    getattr(engine, "model"),
                    device=getattr(engine, "device", torch.device("cpu")),
                    cache_backend=str(getattr(engine, "cache_backend", "dense")),
                    page_size=int(getattr(engine, "page_size", 16)),
                    temperature=temperature,
                    max_active_requests=max_active,
                    prefix_cache_capacity=prefix_rows,
                    prefill_token_budget=prefill_budget_value if prefill_budget_value > 0 else None,
                    enable_ragged_decode=bool(payload.get("enable_ragged_decode", True)),
                    store_reusable_prefixes=bool(payload.get("store_reusable_prefixes", True)),
                    store_full_prompt_prefixes=bool(payload.get("store_full_prompt_prefixes", True)),
                    pin_shared_prefix=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_PIN_SHARED_PREFIX", True),
                    graph_prefill=env_flag("TORCHINFERNO_OPENAI_TP_ONLINE_GRAPH_PREFILL", True),
            prefill_chunk_size=(env_int("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK", 0, minimum=0) or None),
                )
                worker_shared_cache = getattr(engine, "_persistent_serving_cache", None)
                if worker_shared_cache is None:
                    try:
                        total_rows = max_active + prefix_rows
                        worker_shared_cache = engine._generation_cache(
                            total_rows,
                            max_seq_len,
                            model=getattr(engine, "model"),
                            batch_capacity=_generation_cache_batch_capacity(
                                getattr(engine, "model"), total_rows,
                            ),
                        )
                        _reset_generation_cache(worker_shared_cache)
                    except Exception:
                        worker_shared_cache = None
                online_runtime_engine.start_online(
                    max_seq_len=max_seq_len, external_cache=worker_shared_cache,
                )
                online_symm_scope = _tensor_parallel_symm_mem_allreduce_scope(
                    getattr(engine, "model"),
                    getattr(engine, "device", torch.device("cpu")),
                    max_tokens=int(payload.get("max_tokens", 0)),
                    temperature=temperature,
                )
                online_symm_scope.__enter__()
                continue
            if op == "online_submit":
                input_id_lists = payload.get("input_id_lists")
                if not isinstance(input_id_lists, list):
                    raise ValueError("online_submit requires input_id_lists")
                row_max_tokens = _coerce_optional_int_sequence(payload.get("row_max_tokens"))
                if row_max_tokens is None:
                    default_max_tokens = int(payload.get("max_tokens", 0))
                    row_max_tokens = [default_max_tokens for _ in input_id_lists]
                eos_token_id = payload.get("eos_token_id")
                eos = int(eos_token_id) if isinstance(eos_token_id, int) else None
                arrival_step = int(payload.get("arrival_step", 0))
                request_id_start = int(payload.get("request_id_start", 0))
                if online_runtime_engine is None:
                    raise RuntimeError("online tensor-parallel worker engine has not been started")
                for index, prompt in enumerate(input_id_lists):
                    online_runtime_engine.submit_online(
                        _RuntimeServingRequest(
                            str(request_id_start + index),
                            tuple(int(token_id) for token_id in prompt),
                            int(row_max_tokens[index]),
                            arrival_step=arrival_step,
                            eos_token_id=eos,
                        )
                    )
                continue
            if op == "online_step":
                if online_runtime_engine is None:
                    raise RuntimeError("online tensor-parallel worker engine has not been started")
                for _ in range(max(1, int(payload.get("steps", 1)))):
                    for _event in online_runtime_engine.step_online():
                        pass
                continue
            if op == "online_close":
                online_runtime_engine = None
                if online_symm_scope is not None:
                    online_symm_scope.__exit__(None, None, None)
                    online_symm_scope = None
                continue
            if op != "generate":
                raise ValueError(f"unsupported tensor-parallel worker op: {op}")
            max_tokens = int(payload["max_tokens"])
            temperature = float(payload["temperature"])
            with _tensor_parallel_symm_mem_allreduce_scope(
                getattr(engine, "model", None),
                getattr(engine, "device", torch.device("cpu")),
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                if "input_id_lists" in payload:
                    if bool(payload.get("stream", True)):
                        worker_early_restart = (lambda: True) if env_flag(
                            "TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", False
                        ) else None
                        iterator = engine._generate_prompt_list_batch_steps(
                            payload["input_id_lists"],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            broadcast_tensor_parallel=False,
                            row_max_tokens=_coerce_optional_int_sequence(payload.get("row_max_tokens")),
                            early_restart_check=worker_early_restart,
                        )
                        for _ in iterator:
                            pass
                    else:
                        for prompt_group in _indexed_prompts_by_length(
                            [
                                (index, list(prompt))
                                for index, prompt in enumerate(payload["input_id_lists"])
                            ]
                        ):
                            input_ids = torch.tensor(
                                [prompt for _index, prompt in prompt_group],
                                dtype=torch.long,
                                device=engine.device,
                            )
                            engine._generate_batch_tokens(
                                input_ids,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                broadcast_tensor_parallel=False,
                            )
                    continue
                tensor_payload = payload.get("input_ids_tensor")
                if isinstance(tensor_payload, Tensor):
                    input_ids = tensor_payload.to(engine.device, non_blocking=True)
                else:
                    input_ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=engine.device)
                if bool(payload.get("stream", True)):
                    iterator = engine._generate_batch_steps(
                        input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        broadcast_tensor_parallel=False,
                        row_max_tokens=_coerce_optional_int_sequence(payload.get("row_max_tokens")),
                    )
                    for _ in iterator:
                        pass
                else:
                    engine._generate_batch_tokens(
                        input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        broadcast_tensor_parallel=False,
                    )
        finally:
            if not _skip_finally_sync:
                _sync_tensor_parallel_command(getattr(engine, "model", None), engine.device, cuda_sync=cuda_sync)


def _allocate_cache(
    model: object,
    batch_size: int,
    max_seq_len: int,
    *,
    device: torch.device,
    cache_backend: str,
    page_size: int,
) -> object:
    allocate_cache = getattr(model, "allocate_cache")
    try:
        return allocate_cache(
            batch_size,
            max_seq_len,
            device=device,
            cache_backend=cache_backend,
            page_size=page_size,
        )
    except TypeError:
        try:
            return allocate_cache(batch_size, max_seq_len, device=device)
        except TypeError:
            return allocate_cache(batch_size, max_seq_len)


def _generation_cache_capacity(model: object, requested_tokens: int) -> int:
    if _prefers_exact_generation_cache(model):
        return 1 << (max(1, requested_tokens) - 1).bit_length()
    return requested_tokens


def _generation_cache_batch_capacity(model: object, requested_batch: int) -> int:
    requested_batch = max(1, int(requested_batch))
    if not _prefers_exact_generation_cache(model):
        return requested_batch
    if not env_flag("TORCHINFERNO_OPENAI_TP_CACHE_BATCH_BUCKETING", True):
        return requested_batch
    buckets = tuple(sorted(_parse_positive_int_csv(os.environ.get("TORCHINFERNO_OPENAI_TP_CACHE_BATCH_BUCKETS", "64"))))
    for bucket in buckets:
        if requested_batch <= bucket:
            return bucket
    return requested_batch


def _cache_pool_max_entries() -> int:
    return env_int("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", 5, minimum=0)


def _microbatch_cache_pool_max_entries() -> int:
    return env_int("TORCHINFERNO_OPENAI_MICROBATCH_CACHE_POOL_MAX_ENTRIES", 8, minimum=0)


def _generation_cache_seq_len(cache: object) -> int:
    return cache_sequence_length(cache)


def _generation_cache_seq_len_if_uniform(cache: object) -> int | None:
    try:
        return _generation_cache_seq_len(cache)
    except ValueError as exc:
        if "same sequence length" in str(exc):
            return None
        raise


def _clear_generation_cache_repeated_prefix_if_empty(cache: object) -> None:
    if _generation_cache_seq_len_if_uniform(cache) == 0:
        _clear_generation_cache_repeated_prefix(cache)


def _set_generation_cache_seq_len(cache: object, seq_len: int) -> None:
    set_cache_sequence_length(
        cache,
        seq_len,
        on_error=lambda exc: warn_optional_failure("openai.generation_cache.seq_len", exc),
    )


def _set_generation_cache_rows_seq_lens(cache: object, rows: Iterable[int], seq_lens: Sequence[int]) -> None:
    row_tuple = tuple(int(row) for row in rows)
    seq_tuple = tuple(int(seq_len) for seq_len in seq_lens)
    if len(row_tuple) != len(seq_tuple):
        raise ValueError("rows and seq_lens must have the same length")
    for layer in getattr(cache, "layers", ()) or ():
        _set_layer_rows_seq_lens(layer, row_tuple, seq_tuple)


def _prefers_exact_generation_cache(model: object) -> bool:
    return (
        _is_tensor_parallel_model(model)
        and _openai_decode_graph_enabled(model)
    )


def _reset_generation_cache(cache: object) -> bool:
    reset = reset_cache_sequence(
        cache,
        on_error=lambda exc: warn_optional_failure("openai.generation_cache.reset", exc),
    )
    if reset:
        _set_cache_physical_rows_initialized(cache, False)
    return reset


def _warmup_tensor_parallel_decode_attention(model: object) -> None:
    if not env_flag("TORCHINFERNO_TRITON_DECODE_ATTENTION", True):
        return
    if not _is_tensor_parallel_model(model):
        return
    device = getattr(model, "device", torch.device("cpu"))
    if torch.device(device).type != "cuda":
        return
    try:
        from torchinferno.kernels.triton_ops import triton_dense_gqa_decode_attention
    except Exception as exc:
        warn_optional_failure("openai.decode_attention_warmup_import", exc)
        return
    config = getattr(model, "config", None)
    if config is None:
        return
    world_size = max(1, int(getattr(model, "world_size", 1)))
    q_heads = int(config.num_attention_heads) // world_size
    kv_heads = int(config.num_key_value_heads) // world_size
    head_dim = int(config.head_dim)
    dtype = getattr(model, "dtype", torch.bfloat16)
    q = torch.zeros((1, q_heads, 1, head_dim), device=device, dtype=dtype)
    for seq_len in (64, 128, 256):
        k = torch.zeros((1, kv_heads, seq_len, head_dim), device=device, dtype=dtype)
        v = torch.zeros_like(k)
        try:
            triton_dense_gqa_decode_attention(q, k, v)
        except Exception as exc:
            warn_optional_failure("openai.decode_attention_warmup", exc)
            return


def _forward(model: object, input_ids: Tensor, cache: object) -> tuple[Tensor, object]:
    return _forward_with_logits_mode(model, input_ids, cache, return_last_logits_only=True)


def _forward_all_logits(model: object, input_ids: Tensor, cache: object) -> tuple[Tensor, object]:
    return _forward_with_logits_mode(model, input_ids, cache, return_last_logits_only=False)


def _forward_selected_logits(
    model: object,
    input_ids: Tensor,
    cache: object,
    logit_positions: Tensor,
    *,
    allow_capture: bool = False,
) -> tuple[Tensor, object]:
    graph_logits = _try_prefill_selected_logits_graph(
        model,
        input_ids,
        cache,
        logit_positions=logit_positions,
        allow_capture=allow_capture,
    )
    if graph_logits is not None:
        return graph_logits, cache
    forward = model.forward  # type: ignore[attr-defined]
    parameters = _forward_parameter_names(type(model))
    if "logit_positions" not in parameters:
        return _forward_all_logits(model, input_ids, cache)
    kwargs: dict[str, object] = {"cache": cache, "use_cache": True, "logit_positions": logit_positions}
    if "return_last_logits_only" in parameters:
        kwargs["return_last_logits_only"] = False
    if _is_tensor_parallel_model(model) and "return_sharded_logits" in parameters:
        kwargs["return_sharded_logits"] = True
    return forward(input_ids, **kwargs)


def _prefill_padded_suffix_next_token(
    model: object,
    suffix_ids: Tensor,
    cache: object,
    suffix_lengths: Sequence[int],
    temperature: float,
    *,
    allow_capture: bool = False,
) -> tuple[Tensor, object]:
    _clear_generation_cache_repeated_prefix_if_empty(cache)
    active_cache = cache
    if _generation_cache_seq_len_if_uniform(cache) is None:
        active_cache = _cache_row_slice(cache, 0, suffix_ids.size(0)) or cache
    last_positions = torch.tensor(
        [length - 1 for length in suffix_lengths],
        dtype=torch.long,
        device=suffix_ids.device,
    )
    if _selected_padded_suffix_logits_enabled(
        model,
        batch_size=suffix_ids.size(0),
        max_suffix_len=suffix_ids.size(1),
    ):
        logits, _ = _forward_selected_logits(
            model,
            suffix_ids,
            active_cache,
            last_positions,
            allow_capture=allow_capture,
        )
        return _sample(model, logits[:, -1, :], temperature), cache

    logits, _ = _forward_all_logits(model, suffix_ids, active_cache)
    row_positions = torch.arange(suffix_ids.size(0), dtype=torch.long, device=suffix_ids.device)
    return _sample(model, logits[row_positions, last_positions, :], temperature), cache


def _forward_with_logits_mode(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    return_last_logits_only: bool,
) -> tuple[Tensor, object]:
    _clear_generation_cache_repeated_prefix_if_empty(cache)
    forward = model.forward  # type: ignore[attr-defined]
    parameters = _forward_parameter_names(type(model))
    kwargs: dict[str, object] = {"cache": cache, "use_cache": True}
    if "return_last_logits_only" in parameters:
        kwargs["return_last_logits_only"] = return_last_logits_only
    if _is_tensor_parallel_model(model) and "return_sharded_logits" in parameters:
        kwargs["return_sharded_logits"] = True
    return forward(input_ids, **kwargs)


def _prefill_next_token(
    model: object,
    input_ids: Tensor,
    cache: object,
    temperature: float,
    *,
    allow_capture: bool = False,
) -> tuple[Tensor, object]:
    _clear_generation_cache_repeated_prefix_if_empty(cache)
    prefill_token = _try_prefill_graph(model, input_ids, cache, temperature, allow_capture=allow_capture)
    if prefill_token is not None:
        return prefill_token, cache
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache, allow_capture=allow_capture)
    if prefill_logits is None:
        prefill_logits, cache = _forward(model, input_ids, cache)
    return _sample(model, prefill_logits[:, -1, :], temperature), cache


def _prefill_repeated_prefix_next_token(
    model: object,
    input_ids: Tensor,
    cache: object,
    batch_size: int,
    temperature: float,
    *,
    allow_capture: bool = False,
) -> tuple[Tensor, object]:
    prefill_token = _try_prefill_graph(model, input_ids, cache, temperature, allow_capture=allow_capture)
    if prefill_token is not None:
        return prefill_token.expand(batch_size).contiguous(), cache
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache, allow_capture=allow_capture)
    if prefill_logits is None:
        prefill_logits, cache = _forward(model, input_ids, cache)
    if _shared_prefix_sample_enabled(temperature):
        token = _sample(model, prefill_logits[:, -1, :], temperature)
        return token.expand(batch_size).contiguous(), cache
    sample_repeated = getattr(model, "sample_repeated_next_token", None)
    if callable(sample_repeated):
        return sample_repeated(prefill_logits[:, -1, :], batch_size, temperature), cache
    logits = prefill_logits[:, -1, :].expand(batch_size, prefill_logits.size(-1)).contiguous()
    return _sample(model, logits, temperature), cache


def _prefill_repeated_prefix_next_token_with_logits(
    model: object,
    input_ids: Tensor,
    cache: object,
    batch_size: int,
    temperature: float,
    *,
    allow_capture: bool = False,
) -> tuple[Tensor, object, Tensor]:
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache, allow_capture=allow_capture)
    if prefill_logits is None:
        prefill_logits, cache = _forward(model, input_ids, cache)
    last_logits = prefill_logits[:, -1, :]
    return _sample_repeated_prefix_logits(model, last_logits, batch_size, temperature), cache, last_logits


def _decode_repeated_prefix_next_token_with_logits(
    model: object,
    input_ids: Tensor,
    cache: object,
    batch_size: int,
    temperature: float,
) -> tuple[Tensor, object, Tensor]:
    logits = _try_decode_one_token_logits_graph(model, input_ids, cache)
    if logits is None:
        logits, cache = _forward(model, input_ids, cache)
    last_logits = logits[:, -1, :]
    return _sample_repeated_prefix_logits(model, last_logits, batch_size, temperature), cache, last_logits


def _sample_repeated_prefix_logits(
    model: object,
    logits: Tensor,
    batch_size: int,
    temperature: float,
) -> Tensor:
    if _shared_prefix_sample_enabled(temperature):
        token = _sample(model, logits, temperature)
        return token.expand(batch_size).contiguous()
    sample_repeated = getattr(model, "sample_repeated_next_token", None)
    if callable(sample_repeated):
        return sample_repeated(logits, batch_size, temperature)
    expanded = logits.expand(batch_size, logits.size(-1)).contiguous()
    return _sample(model, expanded, temperature)


def _uniform_active_token_id(
    token_ids: Sequence[int],
    active: Sequence[bool],
    row_count: int,
) -> int | None:
    token: int | None = None
    for row in range(min(row_count, len(token_ids), len(active))):
        if not active[row]:
            continue
        row_token = int(token_ids[row])
        if token is None:
            token = row_token
        elif row_token != token:
            return None
    return token


def _decode_next_token(
    model: object,
    input_ids: Tensor,
    cache: object,
    temperature: float,
) -> tuple[Tensor, object]:
    allow_graph_capture = _runtime_decode_graph_capture_enabled(model)
    graph_token = _try_decode_one_token_graph(model, input_ids, cache, temperature, allow_capture=allow_graph_capture)
    if graph_token is not None:
        return graph_token, cache
    graph_logits = _try_decode_one_token_logits_graph(model, input_ids, cache, allow_capture=allow_graph_capture)
    if graph_logits is None:
        graph_logits, cache = _forward(model, input_ids, cache)
    return _sample(model, graph_logits[:, -1, :], temperature), cache


def _decode_next_token_ragged(
    model: object,
    input_ids: Tensor,
    cache: object,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    temperature: float,
) -> tuple[Tensor, object]:
    profile_path = _openai_decode_profile_path_for_model(model)
    profile_start_s = time.perf_counter() if profile_path else 0.0
    allow_graph_capture = _runtime_ragged_decode_graph_capture_enabled(model, cache)
    graph_token = _try_decode_ragged_token_graph(
        model,
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        temperature=temperature,
        allow_capture=allow_graph_capture,
    )
    if graph_token is not None:
        _record_openai_decode_profile(
            profile_path,
            model,
            input_ids,
            cache,
            row_indices,
            mode="token_graph",
            start_s=profile_start_s,
            allow_capture=allow_graph_capture,
        )
        return graph_token, cache
    graph_logits = _try_decode_ragged_logits_graph(
        model,
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        allow_capture=allow_graph_capture,
    )
    if graph_logits is not None:
        next_token = _sample(model, graph_logits[:, -1, :], temperature)
        _record_openai_decode_profile(
            profile_path,
            model,
            input_ids,
            cache,
            row_indices,
            mode="logits_graph",
            start_s=profile_start_s,
            allow_capture=allow_graph_capture,
        )
        return next_token, cache
    decode = getattr(model, "decode_ragged_logits", None)
    if decode is None:
        raise RuntimeError("model does not support ragged decode")
    logits = decode(input_ids, cache, seq_lens=seq_lens, row_indices=row_indices)
    next_token = _sample(model, logits[:, -1, :], temperature)
    _record_openai_decode_profile(
        profile_path,
        model,
        input_ids,
        cache,
        row_indices,
        mode="eager",
        start_s=profile_start_s,
        allow_capture=allow_graph_capture,
    )
    return next_token, cache


def _openai_decode_profile_path_for_model(model: object) -> str:
    path = os.environ.get("TORCHINFERNO_OPENAI_DECODE_PROFILE_JSONL", "")
    if not path:
        return ""
    rank = _model_rank(model)
    if not env_flag("TORCHINFERNO_OPENAI_DECODE_PROFILE_ALL_RANKS", False):
        target_rank = env_int("TORCHINFERNO_OPENAI_DECODE_PROFILE_RANK", 0, minimum=0)
        if rank != target_rank:
            return ""
    return path


def _model_rank(model: object) -> int:
    try:
        return int(getattr(model, "rank", 0))
    except Exception:
        return 0


def _model_graph_cache_profile_fields(model: object, prefix: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for name in (
        "_prefill_graphs",
        "_prefill_logits_graphs",
        "_prefill_selected_logits_graphs",
        "_decode_graphs",
        "_decode_logits_graphs",
        "_ragged_decode_graphs",
        "_ragged_decode_logits_graphs",
    ):
        graphs = getattr(model, name, None)
        if isinstance(graphs, dict):
            fields[f"{prefix}{name.removeprefix('_')}"] = len(graphs)
    return fields


def _record_openai_decode_profile(
    profile_path: str,
    model: object,
    input_ids: Tensor,
    cache: object,
    row_indices: Tensor | None,
    *,
    mode: str,
    start_s: float,
    allow_capture: bool,
) -> None:
    if not profile_path:
        return
    cuda_sync = env_flag("TORCHINFERNO_OPENAI_DECODE_PROFILE_SYNC", False)
    try:
        if cuda_sync and input_ids.is_cuda:
            torch.cuda.synchronize(input_ids.device)
        cache_layers = tuple(getattr(cache, "layers", ()) or ())
        first_layer = cache_layers[0] if cache_layers else None
        cache_max_seq_len = getattr(first_layer, "max_seq_len", None) if first_layer is not None else None
        cache_batch_size = _cache_batch_size(cache)
        cache_id = id(cache)
        ragged_graphs = getattr(model, "_ragged_decode_graphs", None)
        ragged_logits_graphs = getattr(model, "_ragged_decode_logits_graphs", None)
        token_graphs_for_cache = (
            sum(1 for key in ragged_graphs if isinstance(key, tuple) and key and key[0] == cache_id)
            if isinstance(ragged_graphs, dict)
            else None
        )
        logits_graphs_for_cache = (
            sum(1 for key in ragged_logits_graphs if isinstance(key, tuple) and key and key[0] == cache_id)
            if isinstance(ragged_logits_graphs, dict)
            else None
        )
        record = {
            "event": "ragged_decode_step",
            "mode": mode,
            "rank": _model_rank(model),
            "batch_size": int(input_ids.size(0)),
            "tokens": int(input_ids.size(1)) if input_ids.ndim > 1 else 1,
            "row_indices": row_indices is not None,
            "allow_capture": bool(allow_capture),
            "cache_batch_size": cache_batch_size,
            "cache_id": cache_id,
            "cache_max_seq_len": None if cache_max_seq_len is None else int(cache_max_seq_len),
            "cache_graph_disabled": bool(getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)),
            "cache_ephemeral": bool(getattr(cache, "_torchinferno_ephemeral_cache", False)),
            "runtime_capture": getattr(cache, "_torchinferno_runtime_ragged_decode_capture", None),
            "token_graphs_for_cache": token_graphs_for_cache,
            "logits_graphs_for_cache": logits_graphs_for_cache,
            "elapsed_ms": (time.perf_counter() - start_s) * 1000.0,
            "cuda_sync": cuda_sync,
        }
        with open(profile_path, "a", encoding="utf-8") as profile_file:
            profile_file.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        warn_optional_failure("openai.decode_profile", exc)


_DECODE_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64)


def _decode_graph_padded_batch(actual: int, max_batch: int) -> int:
    for bucket in _DECODE_GRAPH_BUCKETS:
        if bucket >= actual:
            return min(bucket, max_batch)
    return min(actual, max_batch)


def _try_decode_ragged_token_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    temperature: float,
    allow_capture: bool = True,
) -> Tensor | None:
    if getattr(cache, "_torchinferno_disable_ragged_decode_graph", False):
        return None
    if getattr(cache, "_torchinferno_ephemeral_cache", False) and not getattr(
        cache,
        "_torchinferno_ephemeral_ragged_graph_scope",
        False,
    ):
        return None
    if not _openai_ragged_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_ragged_token_graph", None)
    if decode_graph is None:
        return None
    if _callable_accepts_keyword(decode_graph, "capture_on_miss"):
        return decode_graph(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            temperature=temperature,
            capture_on_miss=allow_capture,
        )
    return decode_graph(input_ids, cache, seq_lens=seq_lens, row_indices=row_indices, temperature=temperature)


def _ragged_decode_enabled_for_model(model: object) -> bool:
    return (
        env_flag("TORCHINFERNO_OPENAI_RAGGED_DECODE", True)
        and callable(getattr(model, "decode_ragged_logits", None))
    )


def _shared_prefix_ragged_cache_enabled_for_model(model: object) -> bool:
    if not _ragged_decode_enabled_for_model(model):
        return False
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE", True)
    return True


def _shared_prefix_ragged_cache_pool_enabled_for_model(
    model: object,
    *,
    max_tokens: int | None = None,
) -> bool:
    del max_tokens
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE_POOL", True)
    return True


def _identical_prompt_cache_pool_enabled(model: object, temperature: float) -> bool:
    del temperature
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return True
    if "TORCHINFERNO_OPENAI_TP_IDENTICAL_PROMPT_CACHE_POOL" in os.environ:
        return env_flag("TORCHINFERNO_OPENAI_TP_IDENTICAL_PROMPT_CACHE_POOL", True)
    return True


def _prompt_logits_cache_enabled() -> bool:
    return env_flag("TORCHINFERNO_OPENAI_PROMPT_LOGITS_CACHE", True)


def _prefer_shared_prefix_padded_suffix_prefill(
    length_groups: Sequence[Sequence[tuple[int, Sequence[int]]]],
    *,
    prefix_tokens: int,
    prompt_lengths: Sequence[int],
    max_tokens: int | None = None,
) -> bool:
    if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", True):
        return False
    min_groups = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MIN_GROUPS", 2, minimum=2)
    if len(length_groups) < min_groups:
        return False
    min_spread = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MIN_SPREAD", 1, minimum=1)
    if not prompt_lengths or max(prompt_lengths) - min(prompt_lengths) < min_spread:
        return False
    suffix_lengths = [
        max(0, len(prompt) - prefix_tokens)
        for same_length in length_groups
        for _index, prompt in same_length
    ]
    return _shared_prefix_padded_suffix_padding_allowed(
        suffix_lengths,
        prefix_tokens=prefix_tokens,
    ) or _short_output_shared_prefix_padded_suffix_padding_allowed(
        suffix_lengths,
        prefix_tokens=prefix_tokens,
        max_tokens=max_tokens,
    )


def _short_output_shared_prefix_padded_suffix_padding_allowed(
    suffix_lengths: Sequence[int],
    *,
    prefix_tokens: int,
    max_tokens: int | None,
) -> bool:
    if not env_flag("TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_PREFILL", True):
        return False
    if max_tokens is None:
        return False
    max_output_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_TOKENS",
        128,
        minimum=1,
    )
    if max_tokens > max_output_tokens:
        return False
    min_rows = env_int(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MIN_ROWS",
        48,
        minimum=1,
    )
    if len(suffix_lengths) < min_rows:
        return False
    min_prefix_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MIN_PREFIX_TOKENS",
        64,
        minimum=0,
    )
    if prefix_tokens < min_prefix_tokens:
        return False
    if not suffix_lengths or min(suffix_lengths) <= 0:
        return False
    max_suffix_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_SUFFIX_TOKENS",
        96,
        minimum=1,
    )
    if max(suffix_lengths) > max_suffix_tokens:
        return False
    real_suffix_tokens = sum(suffix_lengths)
    padded_suffix_tokens = len(suffix_lengths) * max(suffix_lengths)
    padding_tokens = padded_suffix_tokens - real_suffix_tokens
    max_padding_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_PADDING_TOKENS",
        4096,
        minimum=0,
    )
    if padding_tokens > max_padding_tokens:
        return False
    max_padding_ratio = env_float(
        "TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_PADDING_RATIO",
        1.75,
        minimum=1.0,
    )
    return padded_suffix_tokens <= real_suffix_tokens * max_padding_ratio


def _shared_prefix_padded_suffix_static_batch_enabled(
    model: object,
    *,
    prompt_count: int,
    physical_batch_size: int,
    device: torch.device,
) -> bool:
    if physical_batch_size <= prompt_count:
        return False
    if "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_STATIC_BATCH" in os.environ:
        return env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_STATIC_BATCH", True)
    if device.type != "cuda":
        return False
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return False
    min_rows = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_STATIC_BATCH_MIN_ROWS",
        1,
        minimum=1,
    )
    if prompt_count < min_rows:
        return False
    min_occupancy = env_float(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_STATIC_BATCH_MIN_OCCUPANCY",
        0.0,
        minimum=0.0,
    )
    if min_occupancy > 0.0 and (prompt_count / physical_batch_size) < min_occupancy:
        return False
    return env_flag("TORCHINFERNO_OPENAI_TP_CACHE_BATCH_BUCKETING", True)


def _shared_prefix_suffix_bucket_selected_logits_capture_enabled(
    model: object,
    *,
    batch_size: int,
    max_tokens: int,
    device: torch.device,
) -> bool:
    del batch_size, max_tokens, device
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return False
    return env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_SUFFIX_BUCKET_SELECTED_LOGITS_PREFILL_CAPTURE", False)


def _selected_padded_suffix_logits_enabled(
    model: object,
    *,
    batch_size: int,
    max_suffix_len: int,
) -> bool:
    if not env_flag("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS", True):
        return False
    if "logit_positions" not in _forward_parameter_names(type(model)):
        return False
    skipped_logits = max(0, batch_size * max_suffix_len - batch_size)
    min_skipped_logits = env_int(
        "TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_SKIPPED_TOKENS",
        512,
        minimum=0,
    )
    if skipped_logits < min_skipped_logits:
        return False
    min_total_logits = env_int(
        "TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_TOTAL_TOKENS",
        128,
        minimum=1,
    )
    return batch_size * max_suffix_len >= min_total_logits


def _shared_prefix_padded_suffix_bucketed_length(
    model: object,
    *,
    device: torch.device,
    prompt_count: int,
    max_suffix_len: int,
) -> int:
    max_suffix_len = max(1, int(max_suffix_len))
    if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKETS", False):
        return max_suffix_len
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and device.type == "cuda"):
        return max_suffix_len
    raw_buckets = os.environ.get(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKET_VALUES",
        "16,32,48,64,80,96,128,160,192,256",
    )
    buckets = tuple(sorted(set(_parse_positive_int_csv(raw_buckets))))
    if not buckets:
        return max_suffix_len
    max_extra_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKET_MAX_EXTRA_TOKENS",
        1024,
        minimum=0,
    )
    max_ratio = env_float(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKET_MAX_RATIO",
        1.25,
        minimum=1.0,
    )
    for bucket in buckets:
        if bucket < max_suffix_len:
            continue
        extra_tokens = (bucket - max_suffix_len) * max(1, int(prompt_count))
        if extra_tokens > max_extra_tokens:
            continue
        if float(bucket) / float(max_suffix_len) > max_ratio:
            continue
        return bucket
    return max_suffix_len


def _shared_prefix_padded_suffix_buckets(
    length_groups: Sequence[Sequence[tuple[int, list[int]]]],
    *,
    prefix_tokens: int,
    prompt_lengths: Sequence[int],
) -> list[list[tuple[int, list[int]]]]:
    if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", True):
        return []
    min_groups = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MIN_GROUPS", 2, minimum=2)
    if len(length_groups) < min_groups:
        return []
    min_spread = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MIN_SPREAD", 1, minimum=1)
    if not prompt_lengths or max(prompt_lengths) - min(prompt_lengths) < min_spread:
        return []

    buckets: list[list[Sequence[tuple[int, list[int]]]]] = []
    used_padded_bucket = False
    group_index = 0
    while group_index < len(length_groups):
        bucket_groups: list[Sequence[tuple[int, list[int]]]] = [length_groups[group_index]]
        group_index += 1
        while group_index < len(length_groups):
            candidate = [*bucket_groups, length_groups[group_index]]
            if not _shared_prefix_padded_suffix_bucket_allowed(
                candidate,
                prefix_tokens=prefix_tokens,
                min_groups=min_groups,
            ):
                break
            bucket_groups = candidate
            group_index += 1
            used_padded_bucket = True
        buckets.append(bucket_groups)
    if not used_padded_bucket:
        return []
    return [[item for group in bucket for item in group] for bucket in buckets]


def _shared_prefix_padded_suffix_bucket_allowed(
    groups: Sequence[Sequence[tuple[int, list[int]]]],
    *,
    prefix_tokens: int,
    min_groups: int,
) -> bool:
    if len(groups) < min_groups:
        return False
    suffix_lengths = [
        max(0, len(prompt) - prefix_tokens)
        for group in groups
        for _index, prompt in group
    ]
    return _shared_prefix_padded_suffix_padding_allowed(
        suffix_lengths,
        prefix_tokens=prefix_tokens,
    )


def _shared_prefix_padded_suffix_padding_allowed(
    suffix_lengths: Sequence[int],
    *,
    prefix_tokens: int,
) -> bool:
    if not suffix_lengths or min(suffix_lengths) <= 0:
        return False
    real_suffix_tokens = sum(suffix_lengths)
    padded_suffix_tokens = len(suffix_lengths) * max(suffix_lengths)
    padding_tokens = padded_suffix_tokens - real_suffix_tokens
    max_padding_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_TOKENS",
        4096,
        minimum=0,
    )
    max_padding_ratio = env_float(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_RATIO",
        3.0,
        minimum=1.0,
    )
    if padding_tokens > max_padding_tokens:
        return False
    return padded_suffix_tokens <= real_suffix_tokens * max_padding_ratio


def _shared_prefix_padded_suffix_buckets_signature(
    buckets: Sequence[Sequence[tuple[int, Sequence[int]]]],
) -> int:
    value = 29
    for bucket in buckets:
        value = (value * 1_000_003 + len(bucket)) & 0x7FFFFFFF
        for index, prompt in bucket:
            value = (value * 1_000_003 + int(index)) & 0x7FFFFFFF
            value = (value * 1_000_003 + len(prompt)) & 0x7FFFFFFF
    return value


def _prefix_cached_prompt_suffix_buckets(
    group: Sequence[_PrefixCachedPrompt],
) -> list[list[tuple[int, _PrefixCachedPrompt]]]:
    if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE_BUCKET_SUFFIXES", True):
        return []
    if not group:
        return []
    prefix_tokens = group[0].prefix_tokens
    indexed_prompts = [
        (index, item.prompt)
        for index, item in enumerate(group)
    ]
    length_groups = _indexed_prompts_by_length(indexed_prompts)
    if _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=[len(item.prompt) for item in group],
    ):
        return []
    prompt_buckets = _shared_prefix_padded_suffix_buckets(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=[len(item.prompt) for item in group],
    )
    if not prompt_buckets:
        return []
    return [
        [
            (local_index, group[local_index])
            for local_index, _prompt in bucket
        ]
        for bucket in prompt_buckets
    ]


def _prefix_cached_prompt_suffix_buckets_signature(
    buckets: Sequence[Sequence[tuple[int, _PrefixCachedPrompt]]],
) -> int:
    value = 31
    for bucket in buckets:
        value = (value * 1_000_003 + len(bucket)) & 0x7FFFFFFF
        for local_index, item in bucket:
            value = (value * 1_000_003 + int(local_index)) & 0x7FFFFFFF
            value = (value * 1_000_003 + item.index) & 0x7FFFFFFF
            value = (value * 1_000_003 + item.prefix_tokens) & 0x7FFFFFFF
            value = (value * 1_000_003 + len(item.prompt)) & 0x7FFFFFFF
    return value


def _prefer_shared_prefix_dense_group_decode(
    length_groups: Sequence[Sequence[tuple[int, Sequence[int]]]],
    *,
    prompt_lengths: Sequence[int],
) -> bool:
    if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_DENSE_GROUP_DECODE"):
        return False
    if len(length_groups) <= 1 or not prompt_lengths:
        return False
    max_groups = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_DENSE_GROUP_MAX_GROUPS", 4, minimum=1)
    if len(length_groups) > max_groups:
        return False
    max_spread = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_DENSE_GROUP_MAX_SPREAD", 4, minimum=0)
    if max(prompt_lengths) - min(prompt_lengths) > max_spread:
        return False
    min_group_size = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_DENSE_GROUP_MIN_SIZE", 8, minimum=1)
    return max((len(group) for group in length_groups), default=0) >= min_group_size


def _prefer_full_batch_ragged_decode(active_count: int, batch_size: int) -> bool:
    if active_count >= batch_size:
        return True
    min_batch = env_int("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ROWS", 8, minimum=1)
    if batch_size < min_batch or active_count <= 0:
        return False
    min_fraction = env_float(
        "TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION",
        1.0,
        minimum=0.0,
    )
    return (active_count / batch_size) >= min(1.0, min_fraction)


@dataclass(frozen=True)
class _RaggedDecodeRowPlan:
    active_indices: tuple[int, ...]
    decode_indices: tuple[int, ...]
    row_indices: tuple[int, ...] | None
    advance_row_indices: tuple[int, ...] | None


def _shared_prefix_ragged_decode_row_plan(
    active: Sequence[bool],
    *,
    active_indices: Sequence[int] | None = None,
    step: int,
    cache_batch_size: int,
    force_row_indices: bool,
    static_graph_buckets: bool,
    prefer_full_bucket: bool = False,
) -> _RaggedDecodeRowPlan:
    active_tuple = tuple(
        int(index)
        for index in (
            active_indices
            if active_indices is not None
            else [index for index, is_active in enumerate(active) if is_active]
        )
    )
    decode_full_batch = _prefer_full_batch_ragged_decode(len(active_tuple), len(active))
    if (len(active_tuple) == len(active) or decode_full_batch) and not force_row_indices:
        return _RaggedDecodeRowPlan(
            active_indices=active_tuple,
            decode_indices=tuple(range(len(active))),
            row_indices=None,
            advance_row_indices=None,
        )
    decode_indices = tuple(
        _ragged_decode_bucket_indices(
            active_tuple,
            active,
            step=step,
            batch_capacity=cache_batch_size,
            static_graph_buckets=static_graph_buckets,
            prefer_full_bucket=prefer_full_bucket,
        )
    )
    if decode_indices != active_tuple:
        return _RaggedDecodeRowPlan(
            active_indices=active_tuple,
            decode_indices=decode_indices,
            row_indices=decode_indices,
            advance_row_indices=active_tuple,
        )
    return _RaggedDecodeRowPlan(
        active_indices=active_tuple,
        decode_indices=decode_indices,
        row_indices=active_tuple,
        advance_row_indices=None,
    )


def _ragged_decode_bucket_indices(
    active_indices: Sequence[int],
    active: Sequence[bool],
    *,
    step: int,
    batch_capacity: int | None = None,
    static_graph_buckets: bool = False,
    prefer_full_bucket: bool = False,
) -> list[int]:
    if not env_flag("TORCHINFERNO_OPENAI_RAGGED_DECODE_POWER2_BUCKETS", True):
        return list(active_indices)
    default_min_step = 1 if static_graph_buckets else 4
    min_step = env_int("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", default_min_step, minimum=1)
    if step < min_step:
        return list(active_indices)
    active_count = len(active_indices)
    logical_batch_size = len(active)
    capacity = max(logical_batch_size, int(batch_capacity or 0))
    if active_count <= 0 or active_count >= capacity:
        return list(active_indices)
    bucket_size = _ragged_decode_bucket_size(
        active_count,
        capacity,
        static_graph_buckets=static_graph_buckets,
        prefer_full_bucket=prefer_full_bucket,
    )
    if bucket_size <= active_count:
        return list(active_indices)
    active_set = set(active_indices)
    decode_indices = list(active_indices)
    for index, is_active in enumerate(active):
        if is_active or index in active_set:
            continue
        decode_indices.append(index)
        if len(decode_indices) >= bucket_size:
            break
    for index in range(logical_batch_size, capacity):
        if len(decode_indices) >= bucket_size:
            break
        decode_indices.append(index)
    if len(decode_indices) < bucket_size:
        return list(active_indices)
    return decode_indices


def _ragged_decode_bucket_size(
    active_count: int,
    capacity: int,
    *,
    static_graph_buckets: bool,
    prefer_full_bucket: bool = False,
) -> int:
    if prefer_full_bucket:
        return capacity
    if static_graph_buckets:
        raw_sizes = os.environ.get("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_SIZES")
        bucket_sizes = _parse_positive_int_csv(raw_sizes) if raw_sizes is not None else _warmup_ragged_decode_row_counts()
        for bucket_size in sorted(set(bucket_sizes)):
            if active_count <= bucket_size <= capacity:
                return int(bucket_size)
    return min(capacity, 1 << (active_count - 1).bit_length())


def _prefer_paged_ragged_decode_full_bucket(
    model: object,
    cache: object,
    *,
    static_graph_buckets: bool,
) -> bool:
    if "TORCHINFERNO_OPENAI_PAGED_RAGGED_DECODE_FULL_BUCKET" in os.environ:
        return env_flag("TORCHINFERNO_OPENAI_PAGED_RAGGED_DECODE_FULL_BUCKET", True)
    if "TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_SIZES" in os.environ:
        return False
    return (
        static_graph_buckets
        and getattr(cache, "cache_backend", None) == "paged"
        and not getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)
        and _openai_ragged_decode_graph_enabled(model)
    )


def _tokenizer_padding_token_id(tokenizer: object | None) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            return max(0, int(token_id))
    return 0


def _tensor_row_tokens(input_ids: Tensor, row: int = 0) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in input_ids[row].detach().cpu().tolist())


def _mark_generation_cache_prefix(cache: object, tokens: Sequence[int]) -> None:
    try:
        setattr(cache, "_torchinferno_prefix_tokens", tuple(int(token_id) for token_id in tokens))
    except Exception:
        pass


def _generation_cache_prefix_tokens(cache: object) -> tuple[int, ...] | None:
    tokens = getattr(cache, "_torchinferno_prefix_tokens", None)
    if not isinstance(tokens, tuple):
        return None
    return tuple(int(token_id) for token_id in tokens)


def _mark_generation_cache_repeated_prefix(
    cache: object,
    tokens: tuple[int, ...],
    *,
    seq_len: int,
    rows: int,
) -> None:
    try:
        setattr(cache, "_torchinferno_repeated_prefix", (tokens, int(seq_len), int(rows)))
    except Exception:
        pass


def _clear_generation_cache_repeated_prefix(cache: object) -> None:
    try:
        if hasattr(cache, "_torchinferno_repeated_prefix"):
            delattr(cache, "_torchinferno_repeated_prefix")
    except Exception:
        pass


def _cached_repeated_prefix_rows(cache: object, tokens: tuple[int, ...], seq_len: int) -> int:
    marker = getattr(cache, "_torchinferno_repeated_prefix", None)
    if not (
        isinstance(marker, tuple)
        and len(marker) == 3
        and isinstance(marker[0], tuple)
        and int(marker[1]) == int(seq_len)
    ):
        return 0
    if marker[0] != tokens:
        return 0
    return max(0, int(marker[2]))


def _repeat_generation_cache_first_batch(cache: object, batch_size: int) -> None:
    if batch_size <= 1:
        return
    layers = tuple(getattr(cache, "layers", ()) or ())
    if layers and all(_dense_layer_has_rows(layer, batch_size) for layer in layers):
        for layer in layers:
            seq_len = _layer_row_seq_len(layer, 0)
            if seq_len <= 0:
                continue
            keys = getattr(layer, "keys")
            values = getattr(layer, "values")
            keys[:batch_size, :, :seq_len, :].copy_(keys[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1))
            values[:batch_size, :, :seq_len, :].copy_(values[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1))
            _set_layer_rows_seq_len(layer, range(batch_size), seq_len)
        return
    copy_prefix = getattr(cache, "copy_prefix_from", None)
    if callable(copy_prefix):
        seq_len = _cache_row_seq_len(cache, 0)
        if seq_len <= 0:
            return
        for row in range(1, batch_size):
            copy_prefix(cache, seq_len, source_row=0, dest_row=row)


def _dense_layer_has_rows(layer: object, rows: int) -> bool:
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    return (
        isinstance(keys, Tensor)
        and isinstance(values, Tensor)
        and keys.size(0) >= rows
        and values.size(0) >= rows
    )


def _dense_cache_pair_has_rows(source: object, target: object, rows: int) -> bool:
    source_layers = tuple(getattr(source, "layers", ()) or ())
    target_layers = tuple(getattr(target, "layers", ()) or ())
    return (
        bool(source_layers)
        and len(source_layers) == len(target_layers)
        and all(_dense_layer_has_rows(source_layer, 1) for source_layer in source_layers)
        and all(_dense_layer_has_rows(target_layer, rows) for target_layer in target_layers)
    )


def _dense_cache_pair_has_row(source: object, target: object, source_row: int, target_row: int) -> bool:
    source_layers = tuple(getattr(source, "layers", ()) or ())
    target_layers = tuple(getattr(target, "layers", ()) or ())
    if not source_layers or len(source_layers) != len(target_layers):
        return False
    for source_layer, target_layer in zip(source_layers, target_layers):
        source_keys = getattr(source_layer, "keys", None)
        source_values = getattr(source_layer, "values", None)
        target_keys = getattr(target_layer, "keys", None)
        target_values = getattr(target_layer, "values", None)
        if not all(isinstance(tensor, Tensor) for tensor in (source_keys, source_values, target_keys, target_values)):
            return False
        if source_keys.size(0) <= source_row or source_values.size(0) <= source_row:
            return False
        if target_keys.size(0) <= target_row or target_values.size(0) <= target_row:
            return False
    return True


def _prefill_cache_only(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    allow_capture: bool = False,
) -> object:
    _clear_generation_cache_repeated_prefix_if_empty(cache)
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache, allow_capture=allow_capture)
    if prefill_logits is not None:
        return cache
    prefill_cache_only = getattr(model, "prefill_cache_only", None)
    if callable(prefill_cache_only) and input_ids.size(1) >= _model_cache_only_prefill_min_tokens(model):
        return prefill_cache_only(input_ids, cache)
    _, cache = _forward(model, input_ids, cache)
    return cache


def _model_cache_only_prefill_min_tokens(model: object) -> int:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return env_int("TORCHINFERNO_OPENAI_TP_CACHE_ONLY_PREFILL_MIN_TOKENS", 96, minimum=1)
    return env_int("TORCHINFERNO_OPENAI_CACHE_ONLY_PREFILL_MIN_TOKENS", 1, minimum=1)


def _shared_prefix_sample_enabled(temperature: float) -> bool:
    return (
        temperature > 0.0
        and env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE_SHARED_SAMPLE")
    )


def _sampled_batch_shape_bucket_size(
    model: object,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> int:
    if (
        batch_size <= 1
        or temperature <= 0.0
        or not env_flag("TORCHINFERNO_OPENAI_SAMPLED_BATCH_SHAPE_BUCKETING", True)
        or not _is_tensor_parallel_model(model)
        or torch.device(device).type != "cuda"
    ):
        return batch_size
    max_ratio = env_float("TORCHINFERNO_OPENAI_SAMPLED_BATCH_SHAPE_BUCKET_MAX_RATIO", 2.0, minimum=1.0)
    for candidate in sorted(_warmup_temperature_batch_sizes()):
        if batch_size <= candidate <= int(batch_size * max_ratio):
            return candidate
    return batch_size


def _tensor_rows_are_identical(input_ids: Tensor) -> bool:
    if input_ids.size(0) <= 1:
        return True
    return bool(torch.equal(input_ids, input_ids[:1].expand_as(input_ids)))


def _prompt_rows_are_identical(prompts: Sequence[Sequence[int]]) -> bool:
    if len(prompts) <= 1:
        return True
    first = tuple(int(token_id) for token_id in prompts[0])
    return all(tuple(int(token_id) for token_id in prompt) == first for prompt in prompts[1:])


def _queued_groups_by_prompt_length(group: Sequence[_QueuedGeneration]) -> list[list[_QueuedGeneration]]:
    groups: dict[int, list[_QueuedGeneration]] = {}
    for request in group:
        groups.setdefault(len(request.prompt), []).append(request)
    return list(groups.values())


def _persistent_prompt_list_payload_request_rows(
    payload: Mapping[str, object] | None,
) -> dict[str, int]:
    if payload is None:
        return {}
    request_rows: dict[str, int] = {}
    decode_ids_obj = payload.get("decode_request_ids", [])
    decode_rows_obj = payload.get("decode_rows", [])
    if isinstance(decode_ids_obj, list) and isinstance(decode_rows_obj, list):
        for request_id, row in zip(decode_ids_obj, decode_rows_obj):
            request_rows[str(request_id)] = int(row)
    prefill_items = payload.get("prefill", [])
    if isinstance(prefill_items, list):
        for item in prefill_items:
            if not isinstance(item, Mapping):
                continue
            request_id = item.get("request_id")
            if request_id is None or "row" not in item:
                continue
            request_rows[str(request_id)] = int(item["row"])
    return request_rows


def _emit_stream_step(
    group: Sequence[_QueuedGeneration],
    step: int,
    step_tokens: Sequence[int | None],
    stop_token_ids: frozenset[int] = frozenset(),
) -> None:
    for request, token_id in zip(group, step_tokens):
        _emit_stream_token(
            request,
            token_id,
            generated_tokens=step + 1,
            stop_token_ids=stop_token_ids,
        )


def _emit_stream_token(
    request: _QueuedGeneration,
    token_id: int | None,
    *,
    generated_tokens: int,
    stop_token_ids: frozenset[int] = frozenset(),
) -> None:
    if request.done:
        return
    if token_id is None or generated_tokens > request.max_tokens:
        _finish_stream_request(request)
        return
    token = int(token_id)
    if token in stop_token_ids:
        _finish_stream_request(request)
        return
    request.responses.put(token)
    if generated_tokens >= request.max_tokens:
        _finish_stream_request(request)

def _finish_stream_request(request: _QueuedGeneration) -> None:
    if request.done:
        return
    request.responses.put(_GenerationDone())
    request.done = True


def _fail_stream_request(request: _QueuedGeneration, exc: BaseException) -> None:
    if not request.done:
        request.responses.put(exc)
    _finish_stream_request(request)


def _normalize_row_max_tokens(
    row_max_tokens: Sequence[int] | None,
    batch_size: int,
    max_tokens: int,
) -> list[int]:
    if row_max_tokens is None:
        return [max_tokens for _ in range(batch_size)]
    if len(row_max_tokens) != batch_size:
        raise ValueError("row_max_tokens must match batch size")
    return [max(0, min(max_tokens, int(value))) for value in row_max_tokens]


def _interleave_prompt_segments(
    prompt_count: int,
    segments: Sequence[tuple[Sequence[int], Iterator[list[int | None]]]],
) -> Iterator[list[int | None]]:
    active_segments = list(segments)
    while active_segments:
        step_tokens: list[int | None] = [None for _ in range(prompt_count)]
        next_segments: list[tuple[Sequence[int], Iterator[list[int | None]]]] = []
        emitted = False
        for indices, iterator in active_segments:
            try:
                segment_tokens = next(iterator)
            except StopIteration:
                continue
            for index, token_id in zip(indices, segment_tokens):
                step_tokens[int(index)] = token_id
            next_segments.append((indices, iterator))
            emitted = True
        if not emitted:
            break
        yield step_tokens
        active_segments = next_segments


def _prefix_cached_prompt_groups_signature(groups: Sequence[Sequence[_PrefixCachedPrompt]]) -> int:
    value = 17
    for group in groups:
        value = (value * 1_000_003 + len(group)) & 0x7FFFFFFF
        for item in group:
            value = (value * 1_000_003 + item.index) & 0x7FFFFFFF
            value = (value * 1_000_003 + item.prefix_tokens) & 0x7FFFFFFF
            value = (value * 1_000_003 + len(item.prompt)) & 0x7FFFFFFF
    return value


def _coerce_optional_int_sequence(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, str, bytes)):
        value = value.tolist()  # type: ignore[assignment]
    if not isinstance(value, (list, tuple)):
        return None
    return [int(item) for item in value]


def _indexed_prompts_by_length(indexed: Sequence[tuple[int, list[int]]]) -> list[list[tuple[int, list[int]]]]:
    groups: dict[int, list[tuple[int, list[int]]]] = {}
    for item in indexed:
        groups.setdefault(len(item[1]), []).append(item)
    return sorted(groups.values(), key=lambda group: len(group[0][1]), reverse=True)


def _matching_prefix_tokens(
    cached_tokens: Sequence[int],
    input_tokens: Sequence[int],
    min_prefix_tokens: int,
) -> int:
    if len(input_tokens) <= len(cached_tokens):
        return 0
    max_prefix = min(len(input_tokens) - 1, len(cached_tokens))
    for prefix_tokens in range(max_prefix, min_prefix_tokens - 1, -1):
        if input_tokens[:prefix_tokens] == cached_tokens[:prefix_tokens]:
            return prefix_tokens
    return 0


def _common_prefix_token_count(input_ids: Tensor) -> int:
    if input_ids.size(0) <= 1:
        return input_ids.size(1)
    matches = (input_ids == input_ids[:1]).all(dim=0)
    mismatch = torch.nonzero(~matches, as_tuple=False)
    if mismatch.numel() == 0:
        return input_ids.size(1)
    return int(mismatch[0].item())


def _common_prefix_list_token_count(prompts: Sequence[Sequence[int]]) -> int:
    if not prompts:
        return 0
    prefix = list(prompts[0])
    for prompt in prompts[1:]:
        limit = min(len(prefix), len(prompt))
        mismatch = limit
        for index in range(limit):
            if prefix[index] != prompt[index]:
                mismatch = index
                break
        del prefix[mismatch:]
        if not prefix:
            return 0
    return len(prefix)


def _cache_row_slice(cache: object, start: int, end: int) -> object | None:
    if end <= start:
        return None
    for_rows = getattr(cache, "for_rows", None)
    if callable(for_rows):
        try:
            return for_rows(tuple(range(start, end)))
        except (TypeError, ValueError, AttributeError):
            pass
    layers = tuple(getattr(cache, "layers", ()) or ())
    if not layers:
        return None
    sliced_layers = []
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            return None
        if keys.size(0) < end or values.size(0) < end:
            return None
        layer_view = copy.copy(layer)
        layer_view.keys = keys[start:end]
        layer_view.values = values[start:end]
        if hasattr(layer_view, "batch_size"):
            layer_view.batch_size = end - start
        sliced_layers.append(layer_view)
    cache_view = copy.copy(cache)
    cache_view.layers = sliced_layers
    return cache_view


def _cache_batch_size(cache: object) -> int | None:
    layers = tuple(getattr(cache, "layers", ()) or ())
    if not layers:
        return None
    first_layer = layers[0]
    batch_size = getattr(first_layer, "batch_size", None)
    if batch_size is not None:
        try:
            return int(batch_size)
        except (TypeError, ValueError):
            pass
    keys = getattr(first_layer, "keys", None)
    if isinstance(keys, Tensor) and keys.ndim >= 1:
        return int(keys.size(0))
    return None


def _set_cache_physical_rows_initialized(cache: object, initialized: bool) -> None:
    try:
        setattr(cache, "_torchinferno_physical_rows_initialized", bool(initialized))
    except Exception:
        pass


def _cache_physical_rows_initialized(cache: object) -> bool:
    return bool(getattr(cache, "_torchinferno_physical_rows_initialized", False))


def _cache_requires_physical_rows_initialized(cache: object) -> bool:
    layers = tuple(getattr(cache, "layers", ()) or ())
    if not layers:
        return False
    return hasattr(layers[0], "batch_size")


def _copy_generation_cache_first_row(source: object, target: object, batch_size: int) -> None:
    if batch_size <= 0:
        return
    source_tokens = _generation_cache_prefix_tokens(source)
    source_seq_len = _cache_row_seq_len(source, 0) if source_tokens is not None else 0
    if (
        source_tokens is not None
        and source_seq_len > 0
        and _cached_repeated_prefix_rows(target, source_tokens, source_seq_len) >= batch_size
    ):
        for target_layer in getattr(target, "layers", ()) or ():
            _set_layer_rows_seq_len(target_layer, range(batch_size), source_seq_len)
        return
    if _dense_cache_pair_has_rows(source, target, batch_size):
        for source_layer, target_layer in zip(
            getattr(source, "layers", ()) or (),
            getattr(target, "layers", ()) or (),
        ):
            seq_len = _layer_row_seq_len(source_layer, 0)
            if seq_len <= 0:
                continue
            source_keys = getattr(source_layer, "keys")
            source_values = getattr(source_layer, "values")
            target_keys = getattr(target_layer, "keys")
            target_values = getattr(target_layer, "values")
            if source_keys.size(2) < seq_len or source_values.size(2) < seq_len:
                raise RuntimeError("source shared prefix cache row is shorter than requested")
            if target_keys.size(2) < seq_len or target_values.size(2) < seq_len:
                raise RuntimeError("target shared prefix cache row is shorter than requested")
            target_keys[:batch_size, :, :seq_len, :].copy_(
                source_keys[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1)
            )
            target_values[:batch_size, :, :seq_len, :].copy_(
                source_values[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1)
            )
            _set_layer_rows_seq_len(target_layer, range(batch_size), seq_len)
        if source_tokens is not None and source_seq_len > 0:
            _mark_generation_cache_repeated_prefix(
                target,
                source_tokens,
                seq_len=source_seq_len,
                rows=batch_size,
            )
        return
    copy_prefix = getattr(target, "copy_prefix_from", None)
    if callable(copy_prefix):
        seq_len = _cache_row_seq_len(source, 0)
        if seq_len <= 0:
            return
        for row in range(batch_size):
            copy_prefix(source, seq_len, source_row=0, dest_row=row)
        if source_tokens is not None and source_seq_len > 0:
            _mark_generation_cache_repeated_prefix(
                target,
                source_tokens,
                seq_len=source_seq_len,
                rows=batch_size,
            )
        return
    raise RuntimeError("cannot copy shared prefix cache for non-tensor KV layer")


def _copy_generation_cache_first_row_to_rows(
    source: object,
    target: object,
    rows: Sequence[int],
    seq_len: int,
) -> bool:
    target_rows = tuple(int(row) for row in rows)
    if not target_rows:
        return True
    if seq_len <= 0:
        return False
    if target_rows == tuple(range(len(target_rows))):
        _copy_generation_cache_first_row(source, target, len(target_rows))
        return True
    source_layers = tuple(getattr(source, "layers", ()) or ())
    target_layers = tuple(getattr(target, "layers", ()) or ())
    if not source_layers or len(source_layers) != len(target_layers):
        return False
    max_target_row = max(target_rows)
    for source_layer, target_layer in zip(source_layers, target_layers):
        source_keys = getattr(source_layer, "keys", None)
        source_values = getattr(source_layer, "values", None)
        target_keys = getattr(target_layer, "keys", None)
        target_values = getattr(target_layer, "values", None)
        if not all(isinstance(tensor, Tensor) for tensor in (source_keys, source_values, target_keys, target_values)):
            return False
        if source_keys.size(0) < 1 or source_values.size(0) < 1:
            return False
        if target_keys.size(0) <= max_target_row or target_values.size(0) <= max_target_row:
            return False
        if source_keys.size(2) < seq_len or source_values.size(2) < seq_len:
            return False
        if target_keys.size(2) < seq_len or target_values.size(2) < seq_len:
            return False
        if (
            source_keys.size(1) != target_keys.size(1)
            or source_values.size(1) != target_values.size(1)
            or source_keys.size(3) != target_keys.size(3)
            or source_values.size(3) != target_values.size(3)
        ):
            return False
    target_span = _contiguous_int_span(target_rows)
    for source_layer, target_layer in zip(source_layers, target_layers):
        source_keys = getattr(source_layer, "keys")
        source_values = getattr(source_layer, "values")
        target_keys = getattr(target_layer, "keys")
        target_values = getattr(target_layer, "values")
        expanded_keys = source_keys[:1, :, :seq_len, :].expand(len(target_rows), -1, -1, -1)
        expanded_values = source_values[:1, :, :seq_len, :].expand(len(target_rows), -1, -1, -1)
        if target_span is not None:
            start, end = target_span
            target_keys[start:end, :, :seq_len, :].copy_(expanded_keys)
            target_values[start:end, :, :seq_len, :].copy_(expanded_values)
        else:
            target_index = torch.tensor(target_rows, dtype=torch.long, device=target_keys.device)
            target_keys[:, :, :seq_len, :].index_copy_(0, target_index, expanded_keys)
            target_values[:, :, :seq_len, :].index_copy_(0, target_index, expanded_values)
        _set_layer_rows_seq_len(target_layer, target_rows, seq_len)
    source_tokens = _generation_cache_prefix_tokens(source)
    if source_tokens is not None and target_rows == tuple(range(len(target_rows))):
        _mark_generation_cache_repeated_prefix(target, source_tokens, seq_len=seq_len, rows=len(target_rows))
    return True


def _copy_generation_cache_row(
    source: object,
    target: object,
    *,
    source_row: int,
    target_row: int,
    seq_len: int,
) -> None:
    if seq_len <= 0:
        return
    if _dense_cache_pair_has_row(source, target, source_row, target_row):
        for source_layer, target_layer in zip(
            getattr(source, "layers", ()) or (),
            getattr(target, "layers", ()) or (),
        ):
            source_keys = getattr(source_layer, "keys")
            source_values = getattr(source_layer, "values")
            target_keys = getattr(target_layer, "keys")
            target_values = getattr(target_layer, "values")
            if source_keys.size(2) < seq_len or source_values.size(2) < seq_len:
                raise RuntimeError("source ragged cache row is shorter than requested")
            if target_keys.size(2) < seq_len or target_values.size(2) < seq_len:
                raise RuntimeError("target ragged cache row is shorter than requested")
            target_keys[target_row : target_row + 1, :, :seq_len, :].copy_(
                source_keys[source_row : source_row + 1, :, :seq_len, :]
            )
            target_values[target_row : target_row + 1, :, :seq_len, :].copy_(
                source_values[source_row : source_row + 1, :, :seq_len, :]
            )
            _set_layer_rows_seq_len(target_layer, (target_row,), seq_len)
        return
    copy_prefix = getattr(target, "copy_prefix_from", None)
    if callable(copy_prefix):
        copy_prefix(source, seq_len, source_row=source_row, dest_row=target_row)
        return
    raise RuntimeError("cannot copy ragged cache row for non-tensor KV layer")


def _shared_prefix_ragged_cache_bulk_copy_allowed(prompt_count: int, *, max_tokens: int) -> bool:
    if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_RAGGED_CACHE_BULK_COPY", True):
        return False
    min_rows = env_int("TORCHINFERNO_OPENAI_SHARED_PREFIX_RAGGED_CACHE_BULK_MIN_ROWS", 48, minimum=1)
    if prompt_count < min_rows:
        return False
    max_copy_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_RAGGED_CACHE_BULK_MAX_TOKENS",
        128,
        minimum=0,
    )
    return max_copy_tokens <= 0 or max_tokens <= max_copy_tokens


def _copy_generation_cache_state_rows_padded(
    states: Sequence[Mapping[str, object]],
    target: object,
    *,
    prompt_lengths: Sequence[int],
    prompt_count: int,
) -> bool:
    prepared: list[tuple[object, tuple[int, ...], tuple[int, ...], int]] = []
    target_rows_seen: set[int] = set()
    for state in states:
        source = state.get("cache")
        indices = state.get("indices")
        if source is None or not isinstance(indices, list):
            return False
        target_rows = tuple(int(index) for index in indices)
        if not target_rows:
            continue
        if any(row < 0 or row >= prompt_count or row >= len(prompt_lengths) for row in target_rows):
            return False
        if target_rows_seen.intersection(target_rows):
            return False
        target_rows_seen.update(target_rows)
        seq_lens = tuple(int(prompt_lengths[row]) for row in target_rows)
        max_seq_len = max(seq_lens, default=0)
        if max_seq_len <= 0:
            return False
        prepared.append((source, target_rows, seq_lens, max_seq_len))
    if len(target_rows_seen) != prompt_count or target_rows_seen != set(range(prompt_count)):
        return False
    target_layers = tuple(getattr(target, "layers", ()) or ())
    if not target_layers:
        return False
    for source, target_rows, seq_lens, max_seq_len in prepared:
        source_layers = tuple(getattr(source, "layers", ()) or ())
        if not source_layers or len(source_layers) != len(target_layers):
            return False
        row_count = len(target_rows)
        max_target_row = max(target_rows)
        for source_layer, target_layer in zip(source_layers, target_layers):
            source_keys = getattr(source_layer, "keys", None)
            source_values = getattr(source_layer, "values", None)
            target_keys = getattr(target_layer, "keys", None)
            target_values = getattr(target_layer, "values", None)
            if not all(isinstance(tensor, Tensor) for tensor in (source_keys, source_values, target_keys, target_values)):
                return False
            if source_keys.size(0) < row_count or source_values.size(0) < row_count:
                return False
            if target_keys.size(0) <= max_target_row or target_values.size(0) <= max_target_row:
                return False
            if source_keys.size(2) < max_seq_len or source_values.size(2) < max_seq_len:
                return False
            if target_keys.size(2) < max_seq_len or target_values.size(2) < max_seq_len:
                return False
            if (
                source_keys.size(1) != target_keys.size(1)
                or source_values.size(1) != target_values.size(1)
                or source_keys.size(3) != target_keys.size(3)
                or source_values.size(3) != target_values.size(3)
            ):
                return False
    for source, target_rows, seq_lens, max_seq_len in prepared:
        row_count = len(target_rows)
        for source_layer, target_layer in zip(
            getattr(source, "layers", ()) or (),
            target_layers,
        ):
            source_keys = getattr(source_layer, "keys")
            source_values = getattr(source_layer, "values")
            target_keys = getattr(target_layer, "keys")
            target_values = getattr(target_layer, "values")
            target_span = _contiguous_int_span(target_rows)
            if target_span is not None:
                start, end = target_span
                target_keys[start:end, :, :max_seq_len, :].copy_(
                    source_keys[:row_count, :, :max_seq_len, :]
                )
                target_values[start:end, :, :max_seq_len, :].copy_(
                    source_values[:row_count, :, :max_seq_len, :]
                )
            else:
                target_index = torch.tensor(target_rows, dtype=torch.long, device=target_keys.device)
                target_keys[:, :, :max_seq_len, :].index_copy_(
                    0,
                    target_index,
                    source_keys[:row_count, :, :max_seq_len, :],
                )
                target_values[:, :, :max_seq_len, :].index_copy_(
                    0,
                    target_index,
                    source_values[:row_count, :, :max_seq_len, :],
                )
            _set_layer_rows_seq_len(target_layer, target_rows, max_seq_len)
        _set_generation_cache_rows_seq_lens(target, target_rows, seq_lens)
    return True


def _contiguous_int_span(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return None
    start = int(values[0])
    for offset, value in enumerate(values):
        if int(value) != start + offset:
            return None
    return start, start + len(values)


def _cache_row_seq_len(cache: object, row: int) -> int:
    layers = tuple(getattr(cache, "layers", ()) or ())
    if not layers:
        return 0
    return _layer_row_seq_len(layers[0], row)


def _layer_row_seq_len(layer: object, row: int) -> int:
    seq_len_for_rows = getattr(layer, "seq_len_for_rows", None)
    if callable(seq_len_for_rows):
        return int(seq_len_for_rows((row,)))
    seq_len_for_row = getattr(layer, "seq_len_for_row", None)
    if callable(seq_len_for_row):
        return int(seq_len_for_row(row))
    seq_lens = getattr(layer, "seq_lens", None)
    if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
        return int(seq_lens[row])
    private_seq_lens = getattr(layer, "_seq_lens", None)
    if isinstance(private_seq_lens, list) and 0 <= row < len(private_seq_lens):
        return int(private_seq_lens[row])
    return int(getattr(layer, "seq_len", 0))


def _layer_physical_row(layer: object, row: int) -> int:
    physical_row = getattr(layer, "_physical_row", None)
    if callable(physical_row):
        try:
            return int(physical_row(int(row)))
        except Exception:
            return int(row)
    return int(row)


def _set_layer_rows_seq_lens_direct(
    layer: object,
    rows: Sequence[int],
    seq_lens: Sequence[int],
) -> bool:
    row_tuple = tuple(int(row) for row in rows)
    seq_tuple = tuple(int(seq_len) for seq_len in seq_lens)
    if not row_tuple or len(row_tuple) != len(seq_tuple):
        return False
    target = getattr(layer, "_seq_lens", None)
    if not isinstance(target, list):
        target = getattr(layer, "seq_lens", None)
    if not isinstance(target, list):
        return False
    physical_rows = tuple(_layer_physical_row(layer, row) for row in row_tuple)
    if any(row < 0 or row >= len(target) for row in physical_rows):
        return False
    for row, seq_len in zip(physical_rows, seq_tuple):
        target[row] = seq_len
    uniform = getattr(layer, "_uniform_seq_len", None)
    if isinstance(uniform, list) and uniform:
        first = target[0] if target else 0
        uniform[0] = first if all(value == first for value in target) else None
    return True


def _set_layer_rows_seq_len(layer: object, rows: Iterable[int], seq_len: int) -> None:
    row_tuple = tuple(int(row) for row in rows)
    if not row_tuple:
        return
    if _set_layer_rows_seq_lens_direct(layer, row_tuple, [int(seq_len)] * len(row_tuple)):
        return
    set_seq_len = getattr(layer, "set_seq_len", None)
    if callable(set_seq_len):
        for_rows = getattr(layer, "for_rows", None)
        if callable(for_rows):
            for_rows(row_tuple).set_seq_len(seq_len)
        else:
            set_seq_len(seq_len)
        return
    seq_lens = getattr(layer, "seq_lens", None)
    if isinstance(seq_lens, list):
        for row in row_tuple:
            seq_lens[int(row)] = seq_len
        return
    private_seq_lens = getattr(layer, "_seq_lens", None)
    if isinstance(private_seq_lens, list):
        for row in row_tuple:
            private_seq_lens[int(row)] = seq_len
        return
    try:
        setattr(layer, "seq_len", seq_len)
    except AttributeError:
        pass


def _set_layer_rows_seq_lens(layer: object, rows: Sequence[int], seq_lens: Sequence[int]) -> None:
    row_tuple = tuple(int(row) for row in rows)
    seq_tuple = tuple(int(seq_len) for seq_len in seq_lens)
    if not row_tuple or len(row_tuple) != len(seq_tuple):
        return
    if _set_layer_rows_seq_lens_direct(layer, row_tuple, seq_tuple):
        return
    if len(set(seq_tuple)) == 1:
        _set_layer_rows_seq_len(layer, row_tuple, seq_tuple[0])
        return
    for row, seq_len in zip(row_tuple, seq_tuple):
        _set_layer_rows_seq_len(layer, (row,), seq_len)


def _tokens_not_in_stop(tokens: Tensor, stop_token_ids: frozenset[int]) -> Tensor:
    active = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
    for token_id in stop_token_ids:
        active &= tokens != token_id
    return active


def _try_decode_one_token_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    temperature: float,
    *,
    allow_capture: bool = True,
) -> Tensor | None:
    if getattr(cache, "_torchinferno_ephemeral_cache", False):
        return None
    if not _openai_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_one_token_graph", None)
    if decode_graph is None:
        return None
    if _callable_accepts_keyword(decode_graph, "capture_on_miss"):
        return decode_graph(input_ids, cache, temperature=temperature, capture_on_miss=allow_capture)
    return decode_graph(input_ids, cache, temperature=temperature)


def _try_decode_one_token_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    allow_capture: bool = True,
) -> Tensor | None:
    if getattr(cache, "_torchinferno_ephemeral_cache", False):
        return None
    if not _openai_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_one_token_logits_graph", None)
    if decode_graph is None:
        return None
    if _callable_accepts_keyword(decode_graph, "capture_on_miss"):
        return decode_graph(input_ids, cache, capture_on_miss=allow_capture)
    return decode_graph(input_ids, cache)


def _try_decode_ragged_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    allow_capture: bool = True,
) -> Tensor | None:
    if getattr(cache, "_torchinferno_disable_ragged_decode_graph", False):
        return None
    if getattr(cache, "_torchinferno_ephemeral_cache", False) and not getattr(
        cache,
        "_torchinferno_ephemeral_ragged_graph_scope",
        False,
    ):
        return None
    if not _openai_ragged_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_ragged_logits_graph", None)
    if decode_graph is None:
        return None
    if _callable_accepts_keyword(decode_graph, "capture_on_miss"):
        return decode_graph(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            capture_on_miss=allow_capture,
        )
    return decode_graph(input_ids, cache, seq_lens=seq_lens, row_indices=row_indices)


def _disable_tp_shared_prefix_ragged_decode_graph(model: object, *, max_tokens: int | None = None) -> bool:
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return False
    if "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH" in os.environ:
        return not env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", True)
    if max_tokens is None:
        return False
    if "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS" in os.environ:
        max_graph_tokens = env_int(
            "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS",
            128,
            minimum=1,
        )
        return max_tokens > max_graph_tokens
    disable_min_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS",
        0,
        minimum=0,
    )
    return disable_min_tokens > 0 and max_tokens >= disable_min_tokens


def _disable_tp_shared_prefix_ragged_static_buckets(model: object, *, max_tokens: int | None = None) -> bool:
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return False
    if "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKETS" in os.environ:
        return not env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKETS", True)
    if max_tokens is None:
        return False
    if "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_MAX_TOKENS" in os.environ:
        max_bucket_tokens = env_int(
            "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_MAX_TOKENS",
            128,
            minimum=1,
        )
        return max_tokens > max_bucket_tokens
    disable_min_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS",
        80,
        minimum=0,
    )
    disable_max_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MAX_TOKENS",
        128,
        minimum=0,
    )
    return (
        disable_min_tokens > 0
        and max_tokens >= disable_min_tokens
        and (disable_max_tokens <= 0 or max_tokens <= disable_max_tokens)
    )


def _set_shared_prefix_ragged_static_graph_bucket_mode(
    model: object,
    cache: object,
    *,
    static_graph_buckets: bool,
) -> None:
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return
    attr = "_torchinferno_shared_prefix_ragged_static_graph_buckets"
    nonstatic_released_attr = "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released"
    try:
        previous_static = bool(getattr(cache, attr, False))
        already_released = bool(getattr(cache, nonstatic_released_attr, False))
        static_graph_buckets = bool(static_graph_buckets)
        setattr(cache, attr, static_graph_buckets)
        if static_graph_buckets:
            setattr(cache, nonstatic_released_attr, False)
        else:
            if previous_static and not already_released:
                _sync_before_decode_graph_release(
                    model,
                    cache,
                    label="openai.shared_prefix_ragged.static_bucket_graph_release_sync",
                )
                _release_decode_graphs_for_cache(model, cache)
            setattr(cache, nonstatic_released_attr, True)
    except Exception:
        pass


def _force_tp_shared_prefix_ragged_row_indices(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1


def _set_ragged_decode_graph_disabled(cache: object, disabled: bool) -> None:
    try:
        setattr(cache, "_torchinferno_disable_ragged_decode_graph", disabled)
    except Exception:
        pass


def _release_decode_graphs_for_cache(model: object, cache: object) -> None:
    release = getattr(model, "release_decode_graphs_for_cache", None)
    if callable(release):
        release(cache)


def _sync_before_decode_graph_release(
    model: object,
    cache: object,
    *,
    device: torch.device | str | None = None,
    label: str,
) -> None:
    if _cache_graph_ref_count(model, cache) <= 0:
        return
    sync_device = torch.device(device if device is not None else getattr(model, "device", torch.device("cpu")))
    if sync_device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(sync_device)
    except Exception as exc:
        warn_optional_failure(label, exc)


def _clear_model_graph_caches(model: object) -> None:
    for attr in (
        "_prefill_graphs",
        "_prefill_logits_graphs",
        "_prefill_selected_logits_graphs",
        "_decode_graphs",
        "_decode_logits_graphs",
        "_ragged_decode_graphs",
        "_ragged_decode_logits_graphs",
    ):
        graph_map = getattr(model, attr, None)
        if isinstance(graph_map, dict):
            graph_map.clear()


def _cache_pool_eviction_key(pool: Mapping[object, object], *, model: object) -> object:
    fallback_key: object | None = None
    fallback_refs: int | None = None
    for key, cache in pool.items():
        refs = _cache_graph_ref_count(model, cache)
        if refs == 0:
            return key
        if fallback_refs is None or refs < fallback_refs:
            fallback_key = key
            fallback_refs = refs
    return next(iter(pool)) if fallback_key is None else fallback_key


def _cache_graph_ref_count(model: object, cache: object) -> int:
    cache_id = id(cache)
    refs = 0
    for attr in (
        "_prefill_graphs",
        "_prefill_logits_graphs",
        "_prefill_selected_logits_graphs",
        "_decode_graphs",
        "_decode_logits_graphs",
        "_ragged_decode_graphs",
        "_ragged_decode_logits_graphs",
    ):
        graph_map = getattr(model, attr, None)
        if not isinstance(graph_map, dict):
            continue
        for key, captured in graph_map.items():
            if (
                (isinstance(key, tuple) and bool(key) and key[0] == cache_id)
                or getattr(captured, "cache", None) is cache
            ):
                refs += 1
    return refs


def _openai_decode_graph_enabled(model: object) -> bool:
    if "TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP" in os.environ:
        return (
            env_flag("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", True)
            and _openai_cuda_graph_enabled_for_model(model)
        )
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return (
            env_flag("TORCHINFERNO_OPENAI_TP_DECODE_CUDAGRAPH", True)
            and _openai_cuda_graph_enabled_for_model(model)
        )
    return _openai_cuda_graph_enabled_for_model(model)


def _openai_ragged_decode_graph_enabled(model: object) -> bool:
    if not _openai_decode_graph_enabled(model):
        return False
    if "TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP" in os.environ:
        return env_flag("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP", True)
    return True


def _runtime_decode_graph_capture_enabled(model: object) -> bool:
    if "TORCHINFERNO_OPENAI_RUNTIME_DECODE_CAPTURE" in os.environ:
        return env_flag("TORCHINFERNO_OPENAI_RUNTIME_DECODE_CAPTURE", False)
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return False
    return True


def _runtime_ragged_decode_graph_capture_enabled(model: object, cache: object) -> bool:
    if "TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE" in os.environ:
        return env_flag("TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE", False)
    cache_capture = getattr(cache, "_torchinferno_runtime_ragged_decode_capture", None)
    if cache_capture is not None:
        return bool(cache_capture)
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        return False
    return True


def _runtime_ragged_decode_graph_capture_allowed_for_request(model: object, *, max_tokens: int) -> bool:
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return True
    max_capture_tokens = env_int(
        "TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MAX_TOKENS",
        128,
        minimum=0,
    )
    if max_capture_tokens > 0 and max_tokens <= max_capture_tokens:
        return True
    min_capture_tokens = env_int(
        "TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MIN_TOKENS",
        512,
        minimum=0,
    )
    return min_capture_tokens > 0 and max_tokens >= min_capture_tokens


def _set_runtime_ragged_decode_graph_capture(cache: object, allow_capture: bool) -> None:
    try:
        setattr(cache, "_torchinferno_runtime_ragged_decode_capture", bool(allow_capture))
    except Exception:
        pass


def _try_prefill_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    temperature: float,
    *,
    allow_capture: bool = False,
) -> Tensor | None:
    if not _openai_cuda_graph_enabled_for_model(model):
        return None
    prefill_graph = getattr(model, "try_prefill_graph", None)
    if prefill_graph is None:
        return None
    if _callable_accepts_keyword(prefill_graph, "capture_on_miss"):
        return prefill_graph(input_ids, cache, temperature=temperature, capture_on_miss=allow_capture)
    return prefill_graph(input_ids, cache, temperature=temperature)


def _try_prefill_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    allow_capture: bool = False,
) -> Tensor | None:
    if not _openai_cuda_graph_enabled_for_model(model):
        return None
    prefill_graph = getattr(model, "try_prefill_logits_graph", None)
    if prefill_graph is None:
        return None
    if _callable_accepts_keyword(prefill_graph, "capture_on_miss"):
        return prefill_graph(input_ids, cache, capture_on_miss=allow_capture)
    return prefill_graph(input_ids, cache)


def _try_prefill_selected_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    logit_positions: Tensor,
    allow_capture: bool = False,
) -> Tensor | None:
    if not _openai_cuda_graph_enabled_for_model(model):
        return None
    prefill_graph = getattr(model, "try_prefill_selected_logits_graph", None)
    if prefill_graph is None:
        return None
    if _callable_accepts_keyword(prefill_graph, "capture_on_miss"):
        return prefill_graph(
            input_ids,
            cache,
            logit_positions=logit_positions,
            capture_on_miss=allow_capture,
        )
    return prefill_graph(input_ids, cache, logit_positions=logit_positions)


def _callable_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        return keyword in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=None)
def _forward_parameter_names(model_type: type) -> frozenset[str]:
    return frozenset(inspect.signature(model_type.forward).parameters)


def _sample(model: object, logits: Tensor, temperature: float) -> Tensor:
    sample = getattr(model, "_sample_next_token", None)
    if sample is not None:
        return sample(logits, temperature)
    return sample_next_token(logits, temperature)


def _trim_rows_at_stop(rows: Sequence[Sequence[int]], stop_token_ids: frozenset[int]) -> list[list[int]]:
    trimmed_rows: list[list[int]] = []
    for row in rows:
        trimmed: list[int] = []
        for token_id in row:
            token = int(token_id)
            trimmed.append(token)
            if token in stop_token_ids:
                break
        trimmed_rows.append(trimmed)
    return trimmed_rows


def _format_messages(messages: list[dict[str, object]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _chat_prompt_cache_key(messages: list[dict[str, object]]) -> str:
    return json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TorchInferno behind an OpenAI-compatible HTTP API.")
    parser.add_argument("--model", required=True, help="Model id or local checkpoint path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-kind", default="auto", help="auto, llama3, deepseek, dsv4, or tiny-* for smoke tests.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path. Use 'byte' for smoke tests.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--devices", default=None, help="Comma-separated device list. Defaults to cuda:0..tp-1.")
    parser.add_argument("--device", default=None, help="Single-device fallback for non-Llama models.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--cache-backend",
        choices=["dense", "paged"],
        default=os.environ.get("TORCHINFERNO_OPENAI_CACHE_BACKEND", "dense"),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=env_int("TORCHINFERNO_OPENAI_PAGE_SIZE", 16, minimum=1),
    )
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument(
        "--single-request-admission-wait-ms",
        type=float,
        default=None,
        help=(
            "Optional wait before a lone request takes the direct model path. "
            "Lower values minimize TTFT; higher values preserve a short batching window."
        ),
    )
    parser.add_argument(
        "--llama-parallelism",
        choices=["auto", "pipeline", "tensor"],
        default="auto",
        help=(
            "Use tensor parallel for --tensor-parallel-size > 1, auto-launching "
            "workers when needed; use pipeline to force single-process placement."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OpenAIServerConfig:
    devices: Sequence[str] = ()
    if args.devices:
        devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    return OpenAIServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        model_kind=args.model_kind,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=tuple(devices),
        device=args.device,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        revision=args.revision,
        cache_dir=args.cache_dir,
        cache_backend=args.cache_backend,
        page_size=args.page_size,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        single_request_admission_wait_ms=args.single_request_admission_wait_ms,
        llama_parallelism=args.llama_parallelism,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    original_argv = tuple(sys.argv[1:] if argv is None else argv)
    if _should_reexec_distributed_server(config):
        _reexec_distributed_server(config, original_argv)
    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
