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
from contextlib import nullcontext
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
    OpenAIHTTPServer as _OpenAIServer,
)
from torchinferno.openai_warmup import (
    warmup_prefill_cache_token_counts as _warmup_prefill_cache_token_counts,
    warmup_prefix_suffix_cache_token_counts as _warmup_prefix_suffix_cache_token_counts,
    warmup_prefix_suffix_token_counts as _warmup_prefix_suffix_token_counts,
    warmup_prompt_token_counts as _warmup_prompt_token_counts,
    warmup_ragged_decode_batch_sizes as _warmup_ragged_decode_batch_sizes,
    warmup_ragged_decode_cache_token_counts as _warmup_ragged_decode_cache_token_counts,
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
    batch_wait_ms: float = 10.0
    single_request_admission_wait_ms: float | None = None
    llama_parallelism: str = "auto"


@dataclass(frozen=True)
class CompletionResult:
    tokens: list[int]
    prompt_tokens: int


@dataclass
class _QueuedGeneration:
    prompt: list[int]
    max_tokens: int
    temperature: float
    stream: bool
    responses: "queue.Queue[object] | queue.SimpleQueue[object]"
    done: bool = False


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
        self._prefix_cache_entry: TensorPrefixCacheEntry | None = None
        self._prefix_cache_entries: dict[tuple[int, ...], TensorPrefixCacheEntry] = {}
        self._prompt_logits_cache: dict[tuple[int, ...], Tensor] = {}
        self._prompt_token_cache: dict[str, list[int]] = {}
        self._prompt_token_cache_lock = threading.Lock()
        self._phase_timing_enabled = env_flag("TORCHINFERNO_OPENAI_PHASE_TIMINGS")
        self._phase_records: list[dict[str, float]] = []
        self._phase_records_lock = threading.Lock()
        self._warmup_tokenizer()
        self._warmup_tensor_parallel_model()
        if not _is_tensor_parallel_worker_model(model):
            self._worker = threading.Thread(target=self._batch_worker, name="torchinferno-openai-batcher", daemon=True)
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
        self._generation_queue.put(_QueuedGeneration(prompt, max_tokens, temperature, True, responses))
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
        self._generation_queue.put(_QueuedGeneration(prompt, max_tokens, temperature, False, responses))
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
                self._collect_batch_until_deadline(batch, limit=batch_limit)
            with self._model_lock:
                self._run_queued_batch(batch)

    def _enter_live_request(self) -> None:
        with self._live_request_condition:
            self._live_requests += 1
            self._live_request_condition.notify_all()

    def _exit_live_request(self) -> None:
        with self._live_request_condition:
            self._live_requests -= 1
            self._live_request_condition.notify_all()

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
        return env_float("TORCHINFERNO_OPENAI_TEMPERATURE_ADMISSION_WAIT_MS", 5.0, minimum=0.0) / 1000.0

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
            if first.max_tokens <= short_max_tokens:
                default_short_limit = 128 if first.temperature > 0.0 else 56
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
            return self.batch_wait_s
        max_tokens = env_int("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", 512, minimum=1)
        if first.max_tokens > max_tokens:
            return self.batch_wait_s
        short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
        default_wait_ms = 10.0 if first.max_tokens <= short_max_tokens else 50.0
        temperature_wait_s = (
            env_float("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MS", default_wait_ms, minimum=0.0)
            / 1000.0
        )
        return max(self.batch_wait_s, temperature_wait_s)

    def _queued_initial_batch_wait_s(self, first: _QueuedGeneration) -> float:
        if not first.stream or first.temperature <= 0.0 or self.max_batch_size <= 1:
            return 0.0
        max_temperature_tokens = env_int("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", 512, minimum=1)
        if not (
            _is_tensor_parallel_model(self.model)
            and self.device.type == "cuda"
            and first.max_tokens <= max_temperature_tokens
        ):
            return 0.0
        short_max_tokens = env_int("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", 256, minimum=1)
        if first.max_tokens <= short_max_tokens:
            wait_ms = env_float(
                "TORCHINFERNO_OPENAI_TP_SHORT_SAMPLED_INITIAL_BATCH_WAIT_MS",
                10.0,
                minimum=0.0,
            )
        else:
            wait_ms = env_float("TORCHINFERNO_OPENAI_TP_SAMPLED_INITIAL_BATCH_WAIT_MS", 10.0, minimum=0.0)
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
        prompts = [request.prompt for request in group]
        max_tokens = max((request.max_tokens for request in group), default=0)
        row_max_tokens = [request.max_tokens for request in group]
        use_prompt_list_batch = self._should_use_prompt_list_stream_group(prompts)
        if use_prompt_list_batch and _prefer_tensor_parallel_stream_group(prompts, self.model):
            use_prompt_list_batch = False
        with _tensor_parallel_symm_mem_allreduce_scope(
            self.model,
            self.device,
            max_tokens=max_tokens,
            temperature=group[0].temperature,
        ):
            if use_prompt_list_batch:
                completed_steps = 0
                try:
                    step_iter = self._generate_prompt_list_batch_steps(
                        prompts,
                        max_tokens=max_tokens,
                        temperature=group[0].temperature,
                        row_max_tokens=row_max_tokens,
                    )
                    for step, step_tokens in enumerate(step_iter):
                        completed_steps = step + 1
                        _emit_stream_step(group, step, step_tokens, getattr(self, "stop_token_ids", frozenset()))
                finally:
                    _sync_tensor_parallel_command(
                        self.model,
                        self.device,
                        cuda_sync=_tp_command_cuda_sync_for_steps(completed_steps),
                    )
                return
            for same_length_group in _queued_groups_by_prompt_length(group):
                same_length_max_tokens = max(request.max_tokens for request in same_length_group)
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
                        _emit_stream_step(
                            same_length_group,
                            step,
                            step_tokens,
                            getattr(self, "stop_token_ids", frozenset()),
                        )
                finally:
                    _sync_tensor_parallel_command(
                        self.model,
                        self.device,
                        cuda_sync=_tp_command_cuda_sync_for_steps(completed_steps),
                    )

    def _should_use_prompt_list_stream_group(self, prompts: Sequence[Sequence[int]]) -> bool:
        if self._shared_prefix_prompt_list_tokens(prompts) > 0:
            return True
        return (
            len(prompts) == 1
            and _is_tensor_parallel_model(self.model)
            and _prefix_cache_enabled_for_model(self.model)
            and env_flag("TORCHINFERNO_OPENAI_TP_SINGLE_PROMPT_LIST_STREAM", False)
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
            allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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
    ) -> object:
        cache_capacity = _generation_cache_capacity(model, max_seq_len)
        if not pool:
            cache = _allocate_cache(
                model,
                batch_size,
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
        key = (batch_size, cache_capacity, self.cache_backend, self.page_size, str(self.device))
        for cached_key, cached in list(self._cache_pool.items()):
            cached_batch, cached_max_seq_len, cached_backend, cached_page_size, cached_device = cached_key
            capacity_matches = cached_max_seq_len == cache_capacity if exact_capacity else cached_max_seq_len >= max_seq_len
            if (
                cached_batch == batch_size
                and capacity_matches
                and cached_backend == self.cache_backend
                and cached_page_size == self.page_size
                and cached_device == str(self.device)
                and _reset_generation_cache(cached)
            ):
                _set_ragged_decode_graph_disabled(cached, False)
                self._cache_pool.pop(cached_key, None)
                self._cache_pool[cached_key] = cached
                return cached
        max_entries = _cache_pool_max_entries()
        self._prepare_cache_pool_insert(self._cache_pool, key, max_entries, model=model)
        cache = _allocate_cache(
            model,
            batch_size,
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
            self._evict_cache_pool_key(pool, next(iter(pool)), model=model)

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
            self._evict_cache_pool_key(pool, next(iter(pool)), model=model)

    def _evict_cache_pool_key(self, pool: dict[object, object], key: object, *, model: object) -> None:
        cache = pool.pop(key, None)
        if cache is not None:
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
            128,
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
        if grouped and not _tensor_parallel_all_ranks_same_int(
            self.model,
            _prefix_cached_prompt_groups_signature(grouped),
            self.device,
        ):
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

    def _warmup_tensor_parallel_model(self) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_STARTUP_WARMUP", True):
            return
        if not _is_tensor_parallel_model(self.model) or self.device.type != "cuda":
            return
        if not hasattr(self.model, "generate"):
            return
        prompt_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKENS", 32, minimum=1)
        prompt_token_counts = _warmup_prompt_token_counts(prompt_tokens)
        new_tokens = env_int("TORCHINFERNO_OPENAI_WARMUP_NEW_TOKENS", 2, minimum=1)
        vocab_size = max(1, int(getattr(getattr(self.model, "config", object()), "vocab_size", 1)))
        with torch.inference_mode():
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
            with _tensor_parallel_symm_mem_allreduce_scope(
                self.model,
                self.device,
                max_tokens=1,
                temperature=0.0,
            ):
                self._warmup_tensor_parallel_ragged_decode_graphs(vocab_size)
            warmup_cache_tokens = max(
                max(prompt_token_counts) + new_tokens,
                env_int("TORCHINFERNO_OPENAI_WARMUP_CACHE_TOKENS", 256, minimum=1),
            )
            self._generation_cache(1, warmup_cache_tokens, model=self.model)
            _warmup_tensor_parallel_decode_attention(self.model)
            self._warmup_tensor_parallel_resident_temperature_graphs(vocab_size)
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
        for batch_size in batch_sizes:
            for cache_tokens in cache_token_counts:
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
                for row_count in row_counts:
                    rows = min(batch_size, int(row_count))
                    if rows <= 0:
                        continue
                    if rows == batch_size:
                        _try_decode_ragged_logits_graph(
                            self.model,
                            next_token[:, None],
                            cache,
                            seq_lens=seq_lens,
                            row_indices=None,
                        )
                    else:
                        row_indices = torch.arange(rows, dtype=torch.long, device=self.device)
                        _try_decode_ragged_logits_graph(
                            self.model,
                            next_token[:rows, None],
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
            allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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

        def ensure_cache_materialized() -> None:
            nonlocal cache, cache_materialized
            if cache_materialized:
                return
            restored = self._restore_exact_prefix_cache(input_ids, cache)
            if restored != input_ids.size(1):
                cache = _prefill_cache_only(
                    model,
                    input_ids,
                    cache,
                    allow_capture=_runtime_prefill_graph_capture_enabled(
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
                    allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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
        else:
            next_token, cache = _prefill_repeated_prefix_next_token(
                model,
                input_ids,
                cache,
                decode_batch_size,
                temperature,
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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
                    cached_logits = self._restore_exact_prompt_logits(extended_input_ids, cache)
                    if cached_logits is not None:
                        next_token = _sample_repeated_prefix_logits(
                            model,
                            cached_logits,
                            decode_batch_size,
                            temperature,
                        )
                        cache_materialized = True
                        _repeat_generation_cache_first_batch(cache, decode_batch_size)
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
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
            )
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = torch.full(
                (len(group), max_suffix_len),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, suffix in enumerate(suffix_rows):
                suffix_ids[row, : len(suffix)] = torch.tensor(suffix, dtype=torch.long, device=self.device)
            logits, cache = _forward_all_logits(model, suffix_ids, cache)
            row_positions = torch.arange(len(group), dtype=torch.long, device=self.device)
            last_positions = torch.tensor(
                [len(suffix) - 1 for suffix in suffix_rows],
                dtype=torch.long,
                device=self.device,
            )
            next_token = _sample(model, logits[row_positions, last_positions, :], temperature)
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
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
            )
            ragged_decode = False
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = torch.full(
                (len(bucket), max_suffix_len),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, suffix in enumerate(suffix_rows):
                suffix_ids[row, : len(suffix)] = torch.tensor(suffix, dtype=torch.long, device=self.device)
            logits, cache = _forward_all_logits(model, suffix_ids, cache)
            row_positions = torch.arange(len(bucket), dtype=torch.long, device=self.device)
            last_positions = torch.tensor(
                [len(suffix) - 1 for suffix in suffix_rows],
                dtype=torch.long,
                device=self.device,
            )
            next_token = _sample(model, logits[row_positions, last_positions, :], temperature)
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
            padded_state = self._prefill_shared_prefix_prompt_list_padded_suffixes(
                prompts,
                prefix_cache=prefix_cache,
                prefix_tokens=prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                row_max_tokens=per_row_limits,
                model=model,
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
            bucket_states: list[dict[str, object]] = []
            first_tokens: list[int | None] = [None for _ in prompts]
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
                    combined_cache = self._shared_prefix_prompt_list_ragged_cache(
                        bucket_states,
                        prompt_lengths=prompt_lengths,
                        prompt_count=len(prompts),
                        max_tokens=max_tokens,
                        model=model,
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
        first_tokens: list[int | None] = [None for _ in prompts]
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
            combined_cache = self._shared_prefix_prompt_list_ragged_cache(
                states,
                prompt_lengths=prompt_lengths,
                prompt_count=len(prompts),
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
                allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
            )
            ragged_decode = False
        else:
            pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
            suffix_ids = torch.full(
                (len(prompt_rows), max_suffix_len),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, suffix in enumerate(suffix_rows):
                suffix_ids[row, : len(suffix)] = torch.tensor(suffix, dtype=torch.long, device=self.device)
            logits, cache = _forward_all_logits(model, suffix_ids, cache)
            row_positions = torch.arange(len(prompt_rows), dtype=torch.long, device=self.device)
            last_positions = torch.tensor(
                [len(suffix) - 1 for suffix in suffix_rows],
                dtype=torch.long,
                device=self.device,
            )
            next_token = _sample(model, logits[row_positions, last_positions, :], temperature)
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
        max_prompt_len = max((len(prompt) for prompt in prompts), default=0)
        cache = self._generation_cache(
            prompt_count,
            max_prompt_len + max_tokens,
            model=model,
            pool=_shared_prefix_ragged_cache_pool_enabled_for_model(
                model,
                max_tokens=max_tokens,
            ),
        )
        try:
            _copy_generation_cache_first_row(prefix_cache, cache, prompt_count)
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_padded_suffix_cache", exc)
            return None

        pad_token_id = _tokenizer_padding_token_id(getattr(self, "tokenizer", None))
        suffix_ids = torch.full(
            (prompt_count, max_suffix_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for row, suffix in enumerate(suffix_rows):
            if not suffix:
                return None
            suffix_ids[row, : len(suffix)] = torch.tensor(suffix, dtype=torch.long, device=self.device)

        # Right padding is safe here: real suffix-token logits never attend to
        # future pad tokens, and later ragged decode uses true per-row lengths.
        logits, cache = _forward_all_logits(model, suffix_ids, cache)
        row_positions = torch.arange(prompt_count, dtype=torch.long, device=self.device)
        last_positions = torch.tensor(
            [len(suffix) - 1 for suffix in suffix_rows],
            dtype=torch.long,
            device=self.device,
        )
        next_token = _sample(model, logits[row_positions, last_positions, :], temperature).to(self.device)
        stop_token_ids = self.stop_token_ids
        first_tokens: list[int | None] = []
        active: list[bool] = []
        for row, token_id in enumerate(next_token.detach().cpu().tolist()):
            if row_max_tokens[row] <= 0:
                first_tokens.append(None)
                active.append(False)
                continue
            token = int(token_id)
            first_tokens.append(token)
            active.append(token not in stop_token_ids and row_max_tokens[row] > 1)
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
        cache = self._generation_cache(
            prompt_count,
            max_prompt_len + max_tokens,
            model=model,
            pool=_shared_prefix_ragged_cache_pool_enabled_for_model(
                model,
                max_tokens=max_tokens,
            ),
        )
        try:
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
        except Exception as exc:
            warn_optional_failure("openai.shared_prefix_ragged_cache", exc)
            return None
        return cache

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
    ) -> Iterator[list[int | None]]:
        model = self.model
        if _disable_tp_shared_prefix_ragged_decode_graph(model, max_tokens=max_tokens):
            _set_ragged_decode_graph_disabled(cache, True)
        if max_tokens <= 1 or not any(active):
            return
        per_row_limits = _normalize_row_max_tokens(row_max_tokens, len(active), max_tokens)
        for index, limit in enumerate(per_row_limits):
            if limit <= 1:
                active[index] = False
        if not any(active):
            return
        seq_lens = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        next_token_tensor = torch.tensor(
            [0 if token_id is None else int(token_id) for token_id in next_tokens],
            dtype=torch.long,
            device=self.device,
        )
        ephemeral_graph_allowed = (
            getattr(cache, "_torchinferno_ephemeral_cache", False)
            and env_flag("TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH", True)
        )
        ephemeral_graph_min_step = env_int(
            "TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH_MIN_STEP",
            1,
            minimum=1,
        )
        ephemeral_graph_scope = False
        try:
            for step in range(1, max_tokens):
                active_indices = [index for index, is_active in enumerate(active) if is_active]
                should_decode = _sync_tensor_parallel_continue(model, bool(active_indices), self.device)
                if not should_decode:
                    break
                decode_full_batch = _prefer_full_batch_ragged_decode(len(active_indices), len(active))
                force_row_indices = env_flag(
                    "TORCHINFERNO_OPENAI_RAGGED_DECODE_FORCE_ROW_INDICES",
                    _force_tp_shared_prefix_ragged_row_indices(model),
                )
                advance_row_indices: Tensor | None = None
                if (len(active_indices) == len(active) or decode_full_batch) and not force_row_indices:
                    decode_indices = list(range(len(active)))
                    decode_input = next_token_tensor[:, None]
                    row_indices = None
                else:
                    decode_indices = _ragged_decode_bucket_indices(active_indices, active, step=step)
                    row_indices = torch.tensor(active_indices, dtype=torch.long, device=self.device)
                    if decode_indices != active_indices:
                        row_indices = torch.tensor(decode_indices, dtype=torch.long, device=self.device)
                        advance_row_indices = torch.tensor(active_indices, dtype=torch.long, device=self.device)
                    decode_input = next_token_tensor.index_select(0, row_indices)[:, None]
                if ephemeral_graph_allowed and not ephemeral_graph_scope and step >= ephemeral_graph_min_step:
                    try:
                        setattr(cache, "_torchinferno_ephemeral_ragged_graph_scope", True)
                        ephemeral_graph_scope = True
                    except Exception:
                        ephemeral_graph_allowed = False
                next_token, cache = _decode_next_token_ragged(
                    model,
                    decode_input,
                    cache,
                    seq_lens,
                    row_indices,
                    temperature,
                )
                next_token = next_token.to(self.device)
                if row_indices is None:
                    seq_lens += 1
                elif advance_row_indices is not None:
                    seq_lens[advance_row_indices] = seq_lens.index_select(0, advance_row_indices) + 1
                else:
                    seq_lens[row_indices] = seq_lens.index_select(0, row_indices) + 1
                step_tokens: list[int | None] = [None for _ in active]
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    original_index = decode_indices[offset]
                    token_id = int(token_id)
                    next_token_tensor[original_index] = token_id
                    if not active[original_index]:
                        continue
                    step_tokens[original_index] = token_id
                    if token_id in self.stop_token_ids or step + 1 >= per_row_limits[original_index]:
                        active[original_index] = False
                yield step_tokens
        finally:
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
        _repeat_generation_cache_first_batch(cache, batch_size)
        next_token, cache = _prefill_next_token(
            model,
            input_ids[:, prefix_tokens:],
            cache,
            temperature,
            allow_capture=_runtime_prefill_graph_capture_enabled(model, temperature, max_tokens=max_tokens),
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
    server = _OpenAIServer((config.host, config.port), engine)
    print(
        f"TorchInferno OpenAI server listening on http://{config.host}:{server.server_port}/v1 "
        f"model={config.model}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
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
        or temperature > 0.0
    ):
        return nullcontext()
    max_tokens_limit = env_int("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_TOKENS", 128, minimum=1)
    if max_tokens > max_tokens_limit:
        return nullcontext()
    max_batch = env_int("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_BATCH", 64, minimum=1)
    return symm_mem_allreduce_max_batch(max_batch)


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
_TP_COMMAND_META_FIELDS = 7


def _runtime_prefill_graph_capture_enabled(
    model: object,
    temperature: float = 0.0,
    *,
    max_tokens: int | None = None,
) -> bool:
    if _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1:
        if not env_flag("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE"):
            return False
        if temperature > 0.0 and not env_flag(
            "TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE",
            True,
        ):
            return False
        if max_tokens is not None:
            token_limit = env_int("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", 1024, minimum=1)
            if max_tokens > token_limit:
                return False
        return True
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


def _tp_command_cuda_sync_for_steps(completed_steps: int) -> bool:
    if env_flag("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC"):
        return True
    min_steps = env_int("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", 8, minimum=1)
    return completed_steps >= min_steps


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
    flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _tensor_parallel_all_ranks_same_int(model: object, value: int, device: torch.device) -> bool:
    if not _is_tensor_parallel_model(model) or _tensor_parallel_world_size(model) <= 1:
        return True
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return True
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
) -> bool:
    if not _tensor_parallel_tensor_commands_enabled(model):
        return False
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False
    command_device = token_rows.device
    row_max = (
        torch.tensor([int(value) for value in row_max_tokens], dtype=torch.long, device=command_device)
        if row_max_tokens is not None
        else torch.empty(0, dtype=torch.long, device=command_device)
    )
    meta = torch.tensor(
        [
            command_kind,
            int(bool(stream)),
            int(token_rows.size(0)),
            int(token_rows.size(1)),
            int(max_tokens),
            int(row_max.numel() > 0),
            0,
        ],
        dtype=torch.long,
        device=command_device,
    )
    temp = torch.tensor([float(temperature)], dtype=torch.float64, device=command_device)
    dist.broadcast(meta, src=0)
    dist.broadcast(temp, src=0)
    dist.broadcast(lengths, src=0)
    dist.broadcast(token_rows, src=0)
    if row_max.numel() > 0:
        dist.broadcast(row_max, src=0)
    return True


def _receive_tensor_parallel_tensor_payload(engine: OpenAICompletionEngine) -> dict[str, object]:
    import torch.distributed as dist

    command_device = _tensor_parallel_command_device(engine.device)
    meta = torch.empty(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=command_device)
    dist.broadcast(meta, src=0)
    command_kind = int(meta[0].item())
    if command_kind == _TP_COMMAND_STOP:
        return {"op": "stop"}
    if command_kind not in {_TP_COMMAND_GENERATE_TENSOR, _TP_COMMAND_GENERATE_PROMPT_LISTS}:
        raise ValueError(f"unsupported tensor-parallel tensor command: {command_kind}")

    stream = bool(meta[1].item())
    rows = int(meta[2].item())
    width = int(meta[3].item())
    max_tokens = int(meta[4].item())
    has_row_max_tokens = bool(meta[5].item())
    temp = torch.empty(1, dtype=torch.float64, device=command_device)
    lengths = torch.empty(rows, dtype=torch.long, device=command_device)
    token_rows = torch.empty((rows, width), dtype=torch.long, device=command_device)
    dist.broadcast(temp, src=0)
    dist.broadcast(lengths, src=0)
    dist.broadcast(token_rows, src=0)
    row_max_tokens = None
    if has_row_max_tokens:
        row_max = torch.empty(rows, dtype=torch.long, device=command_device)
        dist.broadcast(row_max, src=0)
        row_max_tokens = [int(value) for value in row_max.detach().cpu().tolist()]

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
        payload["input_id_lists"] = [
            token_rows[row, :length].detach().cpu().tolist()
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
    command_device = _tensor_parallel_command_device(input_ids.device)
    token_rows = input_ids.detach().to(command_device, non_blocking=True).contiguous()
    lengths = torch.full(
        (token_rows.size(0),),
        token_rows.size(1),
        dtype=torch.long,
        device=command_device,
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
        command_device = _tensor_parallel_command_device(getattr(model, "device", torch.device("cpu")))
        lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long, device=command_device)
        width = int(lengths.max().item()) if lengths.numel() > 0 else 0
        token_rows = torch.zeros((len(prompts), width), dtype=torch.long, device=command_device)
        for row, prompt in enumerate(prompts):
            if prompt:
                token_rows[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=command_device)
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


def _broadcast_tensor_parallel_stop(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if _tensor_parallel_tensor_commands_enabled(model):
            device = _tensor_parallel_command_device(getattr(model, "device", torch.device("cpu")))
            meta = torch.zeros(_TP_COMMAND_META_FIELDS, dtype=torch.long, device=device)
            meta[0] = _TP_COMMAND_STOP
            dist.broadcast(meta, src=0)
            return
        dist.broadcast_object_list([{"op": "stop"}], src=0)


def _tensor_parallel_worker_loop(engine: OpenAICompletionEngine) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("tensor-parallel worker loop requires an initialized process group")
    while True:
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
            return
        if op != "generate":
            raise ValueError(f"unsupported tensor-parallel worker op: {op}")
        try:
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
                        for _ in engine._generate_prompt_list_batch_steps(
                            payload["input_id_lists"],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            broadcast_tensor_parallel=False,
                            row_max_tokens=_coerce_optional_int_sequence(payload.get("row_max_tokens")),
                        ):
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
                    for _ in engine._generate_batch_steps(
                        input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        broadcast_tensor_parallel=False,
                        row_max_tokens=_coerce_optional_int_sequence(payload.get("row_max_tokens")),
                    ):
                        pass
                else:
                    engine._generate_batch_tokens(
                        input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        broadcast_tensor_parallel=False,
                    )
        finally:
            _sync_tensor_parallel_command(getattr(engine, "model", None), engine.device)


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


def _cache_pool_max_entries() -> int:
    return env_int("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", 4, minimum=0)


def _microbatch_cache_pool_max_entries() -> int:
    return env_int("TORCHINFERNO_OPENAI_MICROBATCH_CACHE_POOL_MAX_ENTRIES", 8, minimum=0)


def _generation_cache_seq_len(cache: object) -> int:
    return cache_sequence_length(cache)


def _set_generation_cache_seq_len(cache: object, seq_len: int) -> None:
    set_cache_sequence_length(
        cache,
        seq_len,
        on_error=lambda exc: warn_optional_failure("openai.generation_cache.seq_len", exc),
    )


def _prefers_exact_generation_cache(model: object) -> bool:
    return (
        _is_tensor_parallel_model(model)
        and _openai_decode_graph_enabled(model)
    )


def _reset_generation_cache(cache: object) -> bool:
    return reset_cache_sequence(
        cache,
        on_error=lambda exc: warn_optional_failure("openai.generation_cache.reset", exc),
    )


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


def _forward_with_logits_mode(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    return_last_logits_only: bool,
) -> tuple[Tensor, object]:
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
    graph_token = _try_decode_one_token_graph(model, input_ids, cache, temperature)
    if graph_token is not None:
        return graph_token, cache
    graph_logits = _try_decode_one_token_logits_graph(model, input_ids, cache)
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
    graph_token = _try_decode_ragged_token_graph(
        model,
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        temperature=temperature,
    )
    if graph_token is not None:
        return graph_token, cache
    graph_logits = _try_decode_ragged_logits_graph(
        model,
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
    )
    if graph_logits is not None:
        return _sample(model, graph_logits[:, -1, :], temperature), cache
    decode = getattr(model, "decode_ragged_logits", None)
    if decode is None:
        raise RuntimeError("model does not support ragged decode")
    logits = decode(input_ids, cache, seq_lens=seq_lens, row_indices=row_indices)
    return _sample(model, logits[:, -1, :], temperature), cache


def _try_decode_ragged_token_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    seq_lens: Tensor,
    row_indices: Tensor | None,
    temperature: float,
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
    if not suffix_lengths or min(suffix_lengths) <= 0:
        return False
    real_suffix_tokens = sum(suffix_lengths)
    padded_suffix_tokens = len(suffix_lengths) * max(suffix_lengths)
    padding_tokens = padded_suffix_tokens - real_suffix_tokens
    max_padding_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_TOKENS",
        1024,
        minimum=0,
    )
    if padding_tokens > max_padding_tokens:
        return False
    max_padding_ratio = env_float(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_RATIO",
        1.5,
        minimum=1.0,
    )
    return padded_suffix_tokens <= real_suffix_tokens * max_padding_ratio


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
    if not suffix_lengths or min(suffix_lengths) <= 0:
        return False
    real_suffix_tokens = sum(suffix_lengths)
    padded_suffix_tokens = len(suffix_lengths) * max(suffix_lengths)
    padding_tokens = padded_suffix_tokens - real_suffix_tokens
    max_padding_tokens = env_int(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_TOKENS",
        1024,
        minimum=0,
    )
    if padding_tokens > max_padding_tokens:
        return False
    max_padding_ratio = env_float(
        "TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_RATIO",
        1.5,
        minimum=1.0,
    )
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


def _ragged_decode_bucket_indices(active_indices: Sequence[int], active: Sequence[bool], *, step: int) -> list[int]:
    if not env_flag("TORCHINFERNO_OPENAI_RAGGED_DECODE_POWER2_BUCKETS", True):
        return list(active_indices)
    min_step = env_int("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", 4, minimum=1)
    if step < min_step:
        return list(active_indices)
    active_count = len(active_indices)
    batch_size = len(active)
    if active_count <= 0 or active_count >= batch_size:
        return list(active_indices)
    bucket_size = min(batch_size, 1 << (active_count - 1).bit_length())
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
    if len(decode_indices) < bucket_size:
        return list(active_indices)
    return decode_indices


def _tokenizer_padding_token_id(tokenizer: object | None) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            return max(0, int(token_id))
    return 0


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
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache, allow_capture=allow_capture)
    if prefill_logits is not None:
        return cache
    _, cache = _forward(model, input_ids, cache)
    return cache


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


def _emit_stream_step(
    group: Sequence[_QueuedGeneration],
    step: int,
    step_tokens: Sequence[int | None],
    stop_token_ids: frozenset[int] = frozenset(),
) -> None:
    for request, token_id in zip(group, step_tokens):
        if request.done:
            continue
        if step >= request.max_tokens or token_id is None:
            _finish_stream_request(request)
            continue
        token = int(token_id)
        if token in stop_token_ids:
            _finish_stream_request(request)
            continue
        request.responses.put(token)
        if step + 1 >= request.max_tokens:
            _finish_stream_request(request)


def _finish_stream_request(request: _QueuedGeneration) -> None:
    if request.done:
        return
    request.responses.put(_GenerationDone())
    request.done = True


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


def _copy_generation_cache_first_row(source: object, target: object, batch_size: int) -> None:
    if batch_size <= 0:
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
        return
    copy_prefix = getattr(target, "copy_prefix_from", None)
    if callable(copy_prefix):
        seq_len = _cache_row_seq_len(source, 0)
        if seq_len <= 0:
            return
        for row in range(batch_size):
            copy_prefix(source, seq_len, source_row=0, dest_row=row)
        return
    raise RuntimeError("cannot copy shared prefix cache for non-tensor KV layer")


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


def _set_layer_rows_seq_len(layer: object, rows: Iterable[int], seq_len: int) -> None:
    set_seq_len = getattr(layer, "set_seq_len", None)
    if callable(set_seq_len):
        row_tuple = tuple(int(row) for row in rows)
        if not row_tuple:
            return
        for_rows = getattr(layer, "for_rows", None)
        if callable(for_rows):
            for_rows(row_tuple).set_seq_len(seq_len)
        else:
            set_seq_len(seq_len)
        return
    seq_lens = getattr(layer, "seq_lens", None)
    if isinstance(seq_lens, list):
        for row in rows:
            seq_lens[int(row)] = seq_len
        return
    private_seq_lens = getattr(layer, "_seq_lens", None)
    if isinstance(private_seq_lens, list):
        for row in rows:
            private_seq_lens[int(row)] = seq_len
        return
    try:
        setattr(layer, "seq_len", seq_len)
    except AttributeError:
        pass


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
) -> Tensor | None:
    if getattr(cache, "_torchinferno_ephemeral_cache", False):
        return None
    if not _openai_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_one_token_graph", None)
    if decode_graph is None:
        return None
    return decode_graph(input_ids, cache, temperature=temperature)


def _try_decode_one_token_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
) -> Tensor | None:
    if getattr(cache, "_torchinferno_ephemeral_cache", False):
        return None
    if not _openai_decode_graph_enabled(model):
        return None
    decode_graph = getattr(model, "try_decode_one_token_logits_graph", None)
    if decode_graph is None:
        return None
    return decode_graph(input_ids, cache)


def _try_decode_ragged_logits_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    *,
    seq_lens: Tensor,
    row_indices: Tensor | None,
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
    return decode_graph(input_ids, cache, seq_lens=seq_lens, row_indices=row_indices)


def _disable_tp_shared_prefix_ragged_decode_graph(model: object, *, max_tokens: int | None = None) -> bool:
    if not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1):
        return False
    if "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH" in os.environ:
        return not env_flag("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", True)
    if max_tokens is None:
        return False
    max_graph_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS",
        128,
        minimum=1,
    )
    if max_tokens <= max_graph_tokens:
        return False
    large_graph_tokens = env_int(
        "TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_LARGE_MIN_TOKENS",
        512,
        minimum=max_graph_tokens + 1,
    )
    return max_tokens < large_graph_tokens


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
    parser.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=10.0)
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
