from __future__ import annotations

import argparse
import copy
import inspect
import os
import queue
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from torchinferno.models.llama3_family.tensor_parallel import Llama3TensorParallelForCausalLM
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
    max_batch_size: int = 64
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
    responses: "queue.Queue[object]"


@dataclass(frozen=True)
class _GenerationDone:
    pass


@dataclass(frozen=True)
class _GenerationResult:
    tokens: list[int]


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

    def decode_token(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=True))  # type: ignore[attr-defined]

    def decode(self, token_ids: Iterable[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))  # type: ignore[attr-defined]


def load_chat_tokenizer(
    config: OpenAIServerConfig,
    vocab_size: int,
) -> _ByteFallbackTokenizer | _TransformersChatTokenizer:
    tokenizer_name = config.tokenizer or config.model
    if tokenizer_name in {"byte", "bytes", "fallback", "tiny"} or config.model_kind.startswith("tiny"):
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
    return frozenset(token_ids)


def _known_chat_terminator_ids(tokenizer: object) -> set[int]:
    token_ids: set[int] = set()
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return token_ids
    vocab_size = getattr(tokenizer, "vocab_size", None)
    all_special_ids = _coerce_optional_token_ids(getattr(tokenizer, "all_special_ids", None))
    for token in ("<|eot_id|>",):
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
        max_batch_size: int = 64,
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
        self.max_batch_size = max(1, max_batch_size)
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
        self._prefix_cache_entry: TensorPrefixCacheEntry | None = None
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
        prompt = self.tokenizer.encode_messages(messages)
        if self.max_model_len is not None and len(prompt) + max_tokens > self.max_model_len:
            prompt_budget = max(1, self.max_model_len - max_tokens)
            prompt = prompt[-prompt_budget:]
        return prompt

    def _submit_generation(self, prompt: list[int], *, max_tokens: int, temperature: float) -> Iterator[int]:
        if self._closed:
            raise RuntimeError("OpenAI completion engine is closed")
        responses: "queue.Queue[object]" = queue.Queue()
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
        responses: "queue.Queue[object]" = queue.Queue()
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
            self._drain_ready_requests(batch)
            if len(batch) > 1 or self._has_multiple_live_requests():
                self._collect_batch_until_deadline(batch)
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

    def _drain_ready_requests(self, batch: list[_QueuedGeneration]) -> None:
        while len(batch) < self.max_batch_size:
            try:
                item = self._generation_queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._generation_queue.put(None)
                return
            batch.append(item)

    def _collect_batch_until_deadline(self, batch: list[_QueuedGeneration]) -> None:
        if self.batch_wait_s == 0.0:
            return
        deadline = time.perf_counter() + self.batch_wait_s
        while len(batch) < self.max_batch_size:
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
            deadline = time.perf_counter() + self.batch_wait_s

    def _run_queued_batch(self, batch: list[_QueuedGeneration]) -> None:
        groups: dict[tuple[int, float, bool], list[_QueuedGeneration]] = {}
        for request in batch:
            key = (request.max_tokens, request.temperature, request.stream)
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
                    request.responses.put(exc)
            finally:
                if is_stream:
                    for request in group:
                        request.responses.put(_GenerationDone())

    def _run_queued_stream_group(self, group: list[_QueuedGeneration]) -> None:
        prompts = [request.prompt for request in group]
        if self._shared_prefix_prompt_list_tokens(prompts) > 0:
            for step_tokens in self._generate_prompt_list_batch_steps(
                prompts,
                max_tokens=group[0].max_tokens,
                temperature=group[0].temperature,
            ):
                for request, token_id in zip(group, step_tokens):
                    if token_id is not None:
                        request.responses.put(int(token_id))
            return
        for same_length_group in _queued_groups_by_prompt_length(group):
            input_ids = torch.tensor(
                [request.prompt for request in same_length_group],
                dtype=torch.long,
                device=self.device,
            )
            for step_tokens in self._generate_batch_steps(
                input_ids,
                max_tokens=same_length_group[0].max_tokens,
                temperature=same_length_group[0].temperature,
            ):
                for request, token_id in zip(same_length_group, step_tokens):
                    if token_id is not None:
                        request.responses.put(int(token_id))

    def _run_queued_completion_group(self, group: list[_QueuedGeneration]) -> None:
        for same_length_group in _queued_groups_by_prompt_length(group):
            input_ids = torch.tensor(
                [request.prompt for request in same_length_group],
                dtype=torch.long,
                device=self.device,
            )
            rows = self._generate_batch_tokens(
                input_ids,
                max_tokens=same_length_group[0].max_tokens,
                temperature=same_length_group[0].temperature,
            )
            for request, tokens in zip(same_length_group, rows):
                request.responses.put(_GenerationResult(tokens))

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
        next_token, cache = _prefill_next_token(model, input_ids, cache, temperature)
        next_token = next_token.to(self.device)
        generated_tokens: list[Tensor] = []
        active = (
            torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device)
            if stop_token_ids
            else None
        )
        for step in range(max_tokens):
            generated_tokens.append(next_token[:, None])
            if active is not None:
                active &= _tokens_not_in_stop(next_token, stop_token_ids)
                # Keep non-stream decode mostly async; exact output is trimmed after the final transfer.
                should_check_stop = (step + 1) % 8 == 0 or step + 1 == max_tokens
                if should_check_stop and not bool(active.any()):
                    break
            if step + 1 == max_tokens:
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
    ) -> object:
        cache_capacity = _generation_cache_capacity(model, max_seq_len)
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
                self._cache_pool.pop(cached_key, None)
                self._cache_pool[cached_key] = cached
                return cached
        max_entries = _cache_pool_max_entries()
        self._prepare_cache_pool_insert(self._cache_pool, key, max_entries)
        cache = _allocate_cache(
            model,
            batch_size,
            cache_capacity,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
        )
        if _reset_generation_cache(cache):
            self._store_cache_pool_entry(self._cache_pool, key, cache, max_entries)
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
                self._microbatch_cache_pool.pop(cached_key, None)
                self._microbatch_cache_pool[cached_key] = cached
                return cached
        max_entries = _microbatch_cache_pool_max_entries()
        self._evict_microbatch_slot(slot, keep_key=key)
        self._prepare_cache_pool_insert(self._microbatch_cache_pool, key, max_entries)
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
            )
        return cache

    def _evict_microbatch_slot(
        self,
        slot: int,
        *,
        keep_key: tuple[int, int, int, str, int, str],
    ) -> None:
        for cached_key in list(self._microbatch_cache_pool):
            if cached_key != keep_key and cached_key[0] == slot:
                self._microbatch_cache_pool.pop(cached_key, None)

    @staticmethod
    def _prepare_cache_pool_insert(
        pool: dict[object, object],
        key: object,
        max_entries: int,
    ) -> None:
        pool.pop(key, None)
        if max_entries <= 0:
            pool.clear()
            return
        while len(pool) >= max_entries:
            pool.pop(next(iter(pool)))

    @staticmethod
    def _store_cache_pool_entry(
        pool: dict[object, object],
        key: object,
        cache: object,
        max_entries: int,
    ) -> None:
        if max_entries <= 0:
            return
        pool.pop(key, None)
        pool[key] = cache
        while len(pool) > max_entries:
            pool.pop(next(iter(pool)))

    def _restore_prefix_cache(self, input_ids: Tensor, cache: object) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return 0
        if not _prefix_cache_enabled_for_model(self.model):
            self._prefix_cache_entry = None
            return 0
        if input_ids.size(0) != 1:
            return 0
        entry = self._prefix_cache_entry
        if entry is None:
            return 0
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        return restore_tensor_prefix_cache(
            entry,
            input_tokens,
            cache,
            min_prefix_tokens=env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1),
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
            on_seq_len_restore_error=lambda exc: warn_optional_failure("openai.prefix_cache.seq_len_restore", exc),
        )

    def _save_prefix_cache(self, input_ids: Tensor, generated_tokens: list[int], cache: object) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._prefix_cache_entry = None
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
        seq_len = min(len(tokens), _generation_cache_seq_len(cache))
        if seq_len < len(tokens):
            self._prefix_cache_entry = None
            return
        self._prefix_cache_entry = snapshot_tensor_prefix_cache(
            cache,
            tokens,
            seq_len=seq_len,
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
        )

    def _restore_exact_prefix_cache(self, input_ids: Tensor, cache: object) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return 0
        if not _prefix_cache_enabled_for_model(self.model):
            self._prefix_cache_entry = None
            return 0
        if input_ids.size(0) != 1:
            return 0
        entry = self._prefix_cache_entry
        if entry is None:
            return 0
        input_tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        if input_tokens != entry.tokens:
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
        return seq_len

    def _save_prompt_prefix_cache(self, input_ids: Tensor, cache: object) -> None:
        if not env_flag("TORCHINFERNO_OPENAI_PREFIX_CACHE", True):
            return
        if not _prefix_cache_enabled_for_model(self.model):
            self._prefix_cache_entry = None
            return
        if input_ids.size(0) != 1:
            return
        tokens = tuple(int(token_id) for token_id in input_ids[0].detach().cpu().tolist())
        max_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            self._prefix_cache_entry = None
            return
        seq_len = min(len(tokens), _generation_cache_seq_len(cache))
        if seq_len < len(tokens):
            return
        self._prefix_cache_entry = snapshot_tensor_prefix_cache(
            cache,
            tokens,
            seq_len=seq_len,
            device=str(self.device),
            backend=self.cache_backend,
            page_size=self.page_size,
        )

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
            self.tokenizer.encode_messages(
                [{"role": "user", "content": " ".join(f"tok{idx:02d}" for idx in range(32))}]
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
            warmup_cache_tokens = max(
                max(prompt_token_counts) + new_tokens,
                env_int("TORCHINFERNO_OPENAI_WARMUP_CACHE_TOKENS", 256, minimum=1),
            )
            self._generation_cache(1, warmup_cache_tokens, model=self.model)
            _warmup_tensor_parallel_decode_attention(self.model)
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
        prefill_token = _try_prefill_graph(model, prefill_input_ids, cache, temperature)
        if prefill_token is None:
            prefill_logits = _try_prefill_logits_graph(model, prefill_input_ids, cache)
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

    @torch.inference_mode()
    def _generate_batch_steps(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
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
            )
        if input_ids.size(0) > 1 and _tensor_rows_are_identical(input_ids):
            yield from self._generate_identical_prompt_batch_steps(
                input_ids[:1],
                batch_size=input_ids.size(0),
                max_tokens=max_tokens,
                temperature=temperature,
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
            )
            return
        if 0 < microbatch_size < input_ids.size(0):
            yield from self._generate_batch_steps_microbatched(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                microbatch_size=microbatch_size,
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
        active = [True for _ in range(input_ids.size(0))]
        next_token, cache = _prefill_next_token(model, input_ids, cache, temperature)
        next_token = next_token.to(self.device)
        for _ in range(max_tokens):
            token_ids = next_token.detach().cpu().tolist()
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(token_ids):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if token_id in stop_token_ids:
                    active[row] = False
            yield step_tokens
            if not any(active):
                break
            next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
            next_token = next_token.to(self.device)

    @torch.inference_mode()
    def _generate_identical_prompt_batch_steps(
        self,
        input_ids: Tensor,
        *,
        batch_size: int,
        max_tokens: int,
        temperature: float,
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

        cache = self._generation_cache(
            batch_size,
            input_ids.size(1) + max_tokens,
            model=model,
        )
        next_token, cache = _prefill_repeated_prefix_next_token(
            model,
            input_ids,
            cache,
            batch_size,
            temperature,
        )
        next_token = next_token.to(self.device)
        _repeat_generation_cache_first_batch(cache, batch_size)
        shared_sample = _shared_prefix_sample_enabled(temperature)
        active = [True for _ in range(batch_size)]
        for _ in range(max_tokens):
            token_ids = next_token.detach().cpu().tolist()
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(token_ids):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if token_id in stop_token_ids:
                    active[row] = False
            yield step_tokens
            if not any(active):
                break
            decode_input = next_token[:1, None] if shared_sample else next_token[:, None]
            next_token, cache = _decode_next_token(model, decode_input, cache, temperature)
            if shared_sample:
                next_token = next_token.expand(batch_size).contiguous()
                _repeat_generation_cache_first_batch(cache, batch_size)
            next_token = next_token.to(self.device)

    def _stream_microbatch_size(self, batch_size: int) -> int:
        if batch_size <= 1:
            return batch_size
        if "TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE" in os.environ:
            return env_int("TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE", batch_size, minimum=1)
        if _is_tensor_parallel_model(self.model) and self.device.type == "cuda":
            return env_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", 64, minimum=1)
        return batch_size

    @torch.inference_mode()
    def _generate_batch_steps_microbatched(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        microbatch_size: int,
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        states: list[dict[str, object]] = []
        batch_size = input_ids.size(0)
        for slot, start in enumerate(range(0, batch_size, microbatch_size)):
            end = min(batch_size, start + microbatch_size)
            chunk_input_ids = input_ids[start:end]
            cache = self._generation_microbatch_cache(
                slot,
                chunk_input_ids.size(0),
                chunk_input_ids.size(1) + max_tokens,
                model=model,
            )
            active = [True for _ in range(chunk_input_ids.size(0))]
            next_token, cache = _prefill_next_token(model, chunk_input_ids, cache, temperature)
            next_token = next_token.to(self.device)
            step_tokens = [None for _ in range(batch_size)]
            for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                token_id = int(token_id)
                step_tokens[start + offset] = token_id
                if token_id in stop_token_ids:
                    active[offset] = False
            states.append({"start": start, "active": active, "cache": cache, "next_token": next_token})
            yield step_tokens
        for _ in range(1, max_tokens):
            emitted = False
            for state in states:
                active = state["active"]
                if not isinstance(active, list) or not any(active):
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
                step_tokens = [None for _ in range(batch_size)]
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    if not active[offset]:
                        continue
                    token_id = int(token_id)
                    step_tokens[start + offset] = token_id
                    if token_id in stop_token_ids:
                        active[offset] = False
                emitted = True
                yield step_tokens
            if not emitted:
                break

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
        if input_ids.size(0) <= 1 or input_ids.size(1) <= 1:
            return 0
        prefix_tokens = min(_common_prefix_token_count(input_ids), input_ids.size(1) - 1)
        min_tokens = env_int("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", 16, minimum=1)
        return prefix_tokens if prefix_tokens >= min_tokens else 0

    def _shared_prefix_prompt_list_tokens(self, prompts: Sequence[Sequence[int]]) -> int:
        if not env_flag("TORCHINFERNO_OPENAI_SHARED_PREFIX_BATCH", True):
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
            )
        prompt_lengths = {len(prompt) for prompt in prompts}
        if len(prompt_lengths) == 1:
            input_ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
            yield from self._generate_batch_steps(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                broadcast_tensor_parallel=False,
            )
            return
        prefix_tokens = self._shared_prefix_prompt_list_tokens(prompts)
        if prefix_tokens > 0:
            yield from self._generate_shared_prefix_prompt_list_steps(
                prompts,
                prefix_tokens=prefix_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
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
            ):
                step_tokens: list[int | None] = [None for _ in prompts]
                for original_index, token_id in zip(original_indices, group_step):
                    step_tokens[original_index] = token_id
                yield step_tokens

    @torch.inference_mode()
    def _generate_shared_prefix_prompt_list_steps(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
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
        length_groups = _indexed_prompts_by_length(indexed)
        prefix_cache = self._generation_microbatch_cache(
            -1,
            1,
            prefix_tokens,
            model=model,
        )
        prefix_ids = torch.tensor([length_groups[0][0][1][:prefix_tokens]], dtype=torch.long, device=self.device)
        restored_prefix_tokens = self._restore_exact_prefix_cache(prefix_ids, prefix_cache)
        if restored_prefix_tokens != prefix_tokens:
            prefix_cache = _prefill_cache_only(
                model,
                prefix_ids,
                prefix_cache,
                allow_capture=_runtime_prefill_graph_capture_enabled(model),
            )
            self._save_prompt_prefix_cache(prefix_ids, prefix_cache)

        states: list[dict[str, object]] = []
        first_tokens: list[int | None] = [None for _ in prompts]
        for slot, same_length in enumerate(length_groups):
            original_indices = [index for index, _prompt in same_length]
            prompt_rows = [prompt for _index, prompt in same_length]
            suffix_rows = [prompt[prefix_tokens:] for prompt in prompt_rows]
            suffix_len = len(suffix_rows[0])
            cache = self._generation_microbatch_cache(
                slot,
                len(prompt_rows),
                prefix_tokens + suffix_len + max_tokens,
                model=model,
            )
            _copy_generation_cache_first_row(prefix_cache, cache, len(prompt_rows))
            suffix_ids = torch.tensor(suffix_rows, dtype=torch.long, device=self.device)
            next_token, cache = _prefill_next_token(
                model,
                suffix_ids,
                cache,
                temperature,
                allow_capture=_runtime_prefill_graph_capture_enabled(model),
            )
            next_token = next_token.to(self.device)
            active = [True for _ in prompt_rows]
            for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                token_id = int(token_id)
                first_tokens[original_indices[offset]] = token_id
                if token_id in stop_token_ids:
                    active[offset] = False
            states.append(
                {
                    "indices": original_indices,
                    "active": active,
                    "cache": cache,
                    "next_token": next_token,
                }
            )
        yield first_tokens
        if max_tokens <= 1:
            return

        for _ in range(1, max_tokens):
            step_tokens: list[int | None] = [None for _ in prompts]
            emitted = False
            for state in states:
                active = state["active"]
                if not isinstance(active, list) or not any(active):
                    continue
                next_token = state["next_token"]
                cache = state["cache"]
                if not isinstance(next_token, Tensor):
                    raise RuntimeError("invalid shared-prefix prompt-list token state")
                next_token, cache = _decode_next_token(model, next_token[:, None], cache, temperature)
                next_token = next_token.to(self.device)
                state["cache"] = cache
                state["next_token"] = next_token
                original_indices = state["indices"]
                if not isinstance(original_indices, list):
                    raise RuntimeError("invalid shared-prefix prompt-list index state")
                for offset, token_id in enumerate(next_token.detach().cpu().tolist()):
                    if not active[offset]:
                        continue
                    token_id = int(token_id)
                    step_tokens[int(original_indices[offset])] = token_id
                    if token_id in stop_token_ids:
                        active[offset] = False
                    emitted = True
            if not emitted:
                break
            yield step_tokens

    @torch.inference_mode()
    def _generate_shared_prefix_batch_steps(
        self,
        input_ids: Tensor,
        *,
        prefix_tokens: int,
        max_tokens: int,
        temperature: float,
        microbatch_size: int,
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
                allow_capture=_runtime_prefill_graph_capture_enabled(model),
            )
            self._save_prompt_prefix_cache(prefix_ids, cache)
        _repeat_generation_cache_first_batch(cache, batch_size)
        next_token, cache = _prefill_next_token(
            model,
            input_ids[:, prefix_tokens:],
            cache,
            temperature,
            allow_capture=_runtime_prefill_graph_capture_enabled(model),
        )
        next_token = next_token.to(self.device)

        active = [True for _ in range(batch_size)]
        step_tokens: list[int | None] = []
        for row, token_id in enumerate(next_token.detach().cpu().tolist()):
            token_id = int(token_id)
            step_tokens.append(token_id)
            if token_id in stop_token_ids:
                active[row] = False
        yield step_tokens
        if max_tokens <= 1 or not any(active):
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
                )
                return

        for _ in range(1, max_tokens):
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
                if token_id in stop_token_ids:
                    active[row] = False
            yield step_tokens
            if not any(active):
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
    ) -> Iterator[list[int | None]]:
        stop_token_ids = self.stop_token_ids
        model = self.model
        states: list[dict[str, object]] = []
        for slot, start in enumerate(range(0, batch_size, microbatch_size)):
            end = min(batch_size, start + microbatch_size)
            states.append(
                {
                    "start": start,
                    "active": active[start:end],
                    "cache": cache_views[slot],
                    "next_token": next_token[start:end].contiguous(),
                }
            )
        for _ in range(1, max_tokens):
            emitted = False
            for state in states:
                chunk_active = state["active"]
                if not isinstance(chunk_active, list) or not any(chunk_active):
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
                step_tokens = [None for _ in range(batch_size)]
                for offset, token_id in enumerate(chunk_token.detach().cpu().tolist()):
                    if not chunk_active[offset]:
                        continue
                    token_id = int(token_id)
                    step_tokens[start + offset] = token_id
                    if token_id in stop_token_ids:
                        chunk_active[offset] = False
                        active[start + offset] = False
                emitted = True
                yield step_tokens
            if not emitted:
                break


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


def _is_tensor_parallel_primary_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) == 0


def _is_tensor_parallel_worker_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) != 0


def _prefix_cache_enabled_for_model(model: object) -> bool:
    return not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1)


def _runtime_prefill_graph_capture_enabled(model: object) -> bool:
    # TP ranks can legitimately have different graph-cache contents after a
    # bursty serving run. Runtime capture on only a subset of ranks changes the
    # NCCL collective stream, so TP capture is limited to coordinated warmup.
    return not (_is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1)


def _broadcast_tensor_parallel_generate(
    model: object,
    input_ids: Tensor,
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    command = [
        {
            "op": "generate",
            "input_ids": input_ids.detach().cpu().tolist(),
            "max_tokens": int(max_tokens),
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
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    command = [
        {
            "op": "generate",
            "input_id_lists": [list(prompt) for prompt in prompts],
            "max_tokens": int(max_tokens),
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
        dist.broadcast_object_list([{"op": "stop"}], src=0)


def _tensor_parallel_worker_loop(engine: OpenAICompletionEngine) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("tensor-parallel worker loop requires an initialized process group")
    while True:
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
        if "input_id_lists" in payload:
            if bool(payload.get("stream", True)):
                for _ in engine._generate_prompt_list_batch_steps(
                    payload["input_id_lists"],
                    max_tokens=int(payload["max_tokens"]),
                    temperature=float(payload["temperature"]),
                    broadcast_tensor_parallel=False,
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
                        max_tokens=int(payload["max_tokens"]),
                        temperature=float(payload["temperature"]),
                        broadcast_tensor_parallel=False,
                    )
            continue
        input_ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=engine.device)
        if bool(payload.get("stream", True)):
            if input_ids.size(0) == 1:
                for _ in engine._generate_single_tokens(
                    input_ids,
                    max_tokens=int(payload["max_tokens"]),
                    temperature=float(payload["temperature"]),
                    broadcast_tensor_parallel=False,
                    update_prefix_cache=False,
                ):
                    pass
            else:
                for _ in engine._generate_batch_steps(
                    input_ids,
                    max_tokens=int(payload["max_tokens"]),
                    temperature=float(payload["temperature"]),
                    broadcast_tensor_parallel=False,
                ):
                    pass
        else:
            engine._generate_batch_tokens(
                input_ids,
                max_tokens=int(payload["max_tokens"]),
                temperature=float(payload["temperature"]),
                broadcast_tensor_parallel=False,
            )


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
        and env_flag("TORCHINFERNO_CUDAGRAPH_DECODE_STEP", True)
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
    forward = model.forward  # type: ignore[attr-defined]
    parameters = _forward_parameter_names(type(model))
    kwargs: dict[str, object] = {"cache": cache, "use_cache": True}
    if "return_last_logits_only" in parameters:
        kwargs["return_last_logits_only"] = True
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
) -> tuple[Tensor, object]:
    prefill_token = _try_prefill_graph(model, input_ids, cache, temperature)
    if prefill_token is not None:
        return prefill_token.expand(batch_size).contiguous(), cache
    prefill_logits = _try_prefill_logits_graph(model, input_ids, cache)
    if prefill_logits is None:
        prefill_logits, cache = _forward(model, input_ids, cache)
    if _shared_prefix_sample_enabled(temperature):
        token = _sample(model, prefill_logits[:, -1, :], temperature)
        return token.expand(batch_size).contiguous(), cache
    logits = prefill_logits[:, -1, :].expand(batch_size, prefill_logits.size(-1)).contiguous()
    return _sample(model, logits, temperature), cache


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


def _repeat_generation_cache_first_batch(cache: object, batch_size: int) -> None:
    if batch_size <= 1:
        return
    for layer in getattr(cache, "layers", ()) or ():
        seq_len = int(getattr(layer, "seq_len", 0))
        if seq_len <= 0:
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, Tensor) and keys.size(0) >= batch_size:
            keys[:batch_size, :, :seq_len, :].copy_(keys[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1))
        if isinstance(values, Tensor) and values.size(0) >= batch_size:
            values[:batch_size, :, :seq_len, :].copy_(values[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1))


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


def _tensor_rows_are_identical(input_ids: Tensor) -> bool:
    if input_ids.size(0) <= 1:
        return True
    return bool(torch.equal(input_ids, input_ids[:1].expand_as(input_ids)))


def _queued_groups_by_prompt_length(group: Sequence[_QueuedGeneration]) -> list[list[_QueuedGeneration]]:
    groups: dict[int, list[_QueuedGeneration]] = {}
    for request in group:
        groups.setdefault(len(request.prompt), []).append(request)
    return list(groups.values())


def _indexed_prompts_by_length(indexed: Sequence[tuple[int, list[int]]]) -> list[list[tuple[int, list[int]]]]:
    groups: dict[int, list[tuple[int, list[int]]]] = {}
    for item in indexed:
        groups.setdefault(len(item[1]), []).append(item)
    return sorted(groups.values(), key=lambda group: len(group[0][1]), reverse=True)


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
    for source_layer, target_layer in zip(
        getattr(source, "layers", ()) or (),
        getattr(target, "layers", ()) or (),
    ):
        seq_len = int(getattr(source_layer, "seq_len", 0))
        if seq_len <= 0:
            continue
        source_keys = getattr(source_layer, "keys", None)
        source_values = getattr(source_layer, "values", None)
        target_keys = getattr(target_layer, "keys", None)
        target_values = getattr(target_layer, "values", None)
        if not all(isinstance(tensor, Tensor) for tensor in (source_keys, source_values, target_keys, target_values)):
            raise RuntimeError("cannot copy shared prefix cache for non-tensor KV layer")
        if target_keys.size(0) < batch_size or target_values.size(0) < batch_size:
            raise RuntimeError("shared prefix target cache batch is too small")
        target_keys[:batch_size, :, :seq_len, :].copy_(
            source_keys[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1)
        )
        target_values[:batch_size, :, :seq_len, :].copy_(
            source_values[:1, :, :seq_len, :].expand(batch_size, -1, -1, -1)
        )
        target_layer.seq_len = seq_len


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
    if not _openai_decode_graph_enabled():
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
    if not _openai_decode_graph_enabled():
        return None
    decode_graph = getattr(model, "try_decode_one_token_logits_graph", None)
    if decode_graph is None:
        return None
    return decode_graph(input_ids, cache)


def _openai_decode_graph_enabled() -> bool:
    return env_flag("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", True)


def _try_prefill_graph(
    model: object,
    input_ids: Tensor,
    cache: object,
    temperature: float,
    *,
    allow_capture: bool = False,
) -> Tensor | None:
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
    parser.add_argument("--max-batch-size", type=int, default=64)
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
