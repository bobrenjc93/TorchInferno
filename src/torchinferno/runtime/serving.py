from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from inspect import signature
from typing import Callable, Hashable, Iterator, Optional

import torch
from torch import Tensor

from torchinferno.runtime.options import env_flag, env_int
from torchinferno.runtime.prefix_cache import PrefixCacheIndex
from torchinferno.runtime.sampling import sample_next_token


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    arrival_step: int = 0
    eos_token_id: Optional[int] = None


@dataclass(frozen=True)
class ServingResult:
    request_id: str
    tokens: tuple[int, ...]
    prefix_hit_tokens: int
    arrival_step: int
    started_step: int
    finished_step: int


@dataclass(frozen=True)
class ServingTokenEvent:
    request_id: str
    token: int
    step: int
    generated: int
    finished: bool


@dataclass
class ServingStats:
    prefill_model_calls: int = 0
    prefill_batches: int = 0
    prefill_tokens: int = 0
    decode_model_calls: int = 0
    decode_batches: int = 0
    decode_tokens: int = 0
    ragged_decode_batches: int = 0
    ragged_decode_tokens: int = 0
    decode_graph_hits: int = 0
    decode_graph_misses: int = 0
    prefix_reuse_requests: int = 0
    prefix_reuse_tokens: int = 0
    queued_requests: int = 0
    scheduler_steps: int = 0
    max_model_batch_size: int = 0
    persistent_cache_rows: int = 0
    prefill_admitted_requests: int = 0
    prefill_single_batches: int = 0
    prefill_plain_batches: int = 0
    prefill_prefix_reuse_batches: int = 0
    prefill_common_prefix_batches: int = 0
    prefill_padded_suffix_batches: int = 0
    prefill_graph_hits: int = 0
    prefill_graph_misses: int = 0
    prefill_wall_ms: float = 0.0
    decode_ragged_prepare_ms: float = 0.0
    decode_ragged_model_ms: float = 0.0
    decode_ragged_cpu_tokens_ms: float = 0.0
    decode_ragged_state_update_ms: float = 0.0


@dataclass(frozen=True)
class _QueuedRequest:
    original_index: int
    request: ServingRequest
    sequence: int


@dataclass
class _ReusablePrefix:
    route_id: Hashable
    tokens: tuple[int, ...]
    row: int
    logits: Tensor


@dataclass
class _ActiveRequest:
    original_index: int
    request: ServingRequest
    tokens: list[int]
    generated: int
    row: int
    last_token: int
    seq_len: int
    prefix_hit_tokens: int
    started_step: int


def _contiguous_int_span(values: tuple[int, ...]) -> tuple[int, int] | None:
    if not values:
        return None
    start = values[0]
    for offset, value in enumerate(values):
        if value != start + offset:
            return None
    return start, start + len(values)


class ServingQueue:
    """Arrival-ordered queue with prefix-aware admission hooks."""

    def __init__(self, requests: list[tuple[int, ServingRequest]] | None = None) -> None:
        self._items: list[_QueuedRequest] = []
        self._next_sequence = 0
        if requests is not None:
            for original_index, request in requests:
                self.push(original_index, request)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, original_index: int, request: ServingRequest) -> None:
        self._items.append(_QueuedRequest(original_index, request, self._next_sequence))
        self._next_sequence += 1
        self._items.sort(key=self._arrival_key)

    def pop_admissible(
        self,
        *,
        step: int,
        capacity: int,
        token_budget: int | None = None,
        token_cost: Callable[[ServingRequest], int] | None = None,
        priority_key: Callable[[_QueuedRequest], tuple[object, ...]] | None = None,
    ) -> list[tuple[int, ServingRequest]]:
        if capacity <= 0:
            return []
        ready: list[_QueuedRequest] = []
        waiting: list[_QueuedRequest] = []
        for item in self._items:
            if item.request.arrival_step <= step:
                ready.append(item)
            else:
                waiting.append(item)
        if not ready:
            self._items = waiting
            return []
        if priority_key is None:
            ready.sort(key=self._arrival_key)
        else:
            ready.sort(key=priority_key)

        selected: list[_QueuedRequest] = []
        deferred: list[_QueuedRequest] = []
        remaining_budget = token_budget
        for item in ready:
            if len(selected) >= capacity:
                deferred.append(item)
                continue
            cost = max(1, token_cost(item.request) if token_cost is not None else len(item.request.prompt))
            if remaining_budget is not None and selected and cost > remaining_budget:
                deferred.append(item)
                continue
            selected.append(item)
            if remaining_budget is not None:
                remaining_budget -= cost
        self._items = [*deferred, *waiting]
        self._items.sort(key=self._arrival_key)
        return [(item.original_index, item.request) for item in selected]

    def ready_count(self, *, step: int) -> int:
        return sum(1 for item in self._items if item.request.arrival_step <= step)

    def next_arrival_step(self) -> int | None:
        if not self._items:
            return None
        return min(item.request.arrival_step for item in self._items)

    @staticmethod
    def _arrival_key(item: _QueuedRequest) -> tuple[int, int]:
        return (item.request.arrival_step, item.sequence)


class ContinuousBatchEngine:
    """Token-step continuous serving harness with persistent row-assigned cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        cache_backend: str = "dense",
        page_size: int = 16,
        temperature: float = 0.0,
        max_active_requests: int = 16,
        prefix_cache_capacity: int | None = None,
        prefill_token_budget: int | None = None,
        decode_first: bool = True,
        enable_ragged_decode: bool = True,
        store_reusable_prefixes: bool = True,
        store_full_prompt_prefixes: bool = True,
        profile_timings: bool = False,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        if prefix_cache_capacity is not None and prefix_cache_capacity < 0:
            raise ValueError("prefix_cache_capacity must be non-negative")
        if prefill_token_budget is not None and prefill_token_budget < 1:
            raise ValueError("prefill_token_budget must be positive")
        model_to = getattr(model, "to", None)
        if callable(model_to):
            model = model_to(device)
        model_eval = getattr(model, "eval", None)
        if callable(model_eval):
            model = model_eval()
        self.model = model
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.temperature = temperature
        self.max_active_requests = max_active_requests
        self.prefix_cache_capacity = max_active_requests if prefix_cache_capacity is None else prefix_cache_capacity
        self.prefill_token_budget = prefill_token_budget
        self.decode_first = decode_first
        self.enable_ragged_decode = enable_ragged_decode
        self.store_reusable_prefixes = store_reusable_prefixes
        self.store_full_prompt_prefixes = store_full_prompt_prefixes
        self.profile_timings = profile_timings
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes: dict[Hashable, _ReusablePrefix] = {}
        self.stats = ServingStats()
        self._cache: object | None = None
        self._cache_views: dict[tuple[int, ...], object] = {}
        self._reported_static_graph_miss = False
        self._free_active_rows: list[int] = []
        self._free_prefix_rows: list[int] = []
        self._row_seq_lens: list[int] = []
        self._prefix_order: list[Hashable] = []
        self._online_waiting: ServingQueue | None = None
        self._online_active: list[_ActiveRequest] = []
        self._online_step = 0
        self._online_next_index = 0

    @torch.inference_mode()
    def run(self, requests: list[ServingRequest]) -> list[ServingResult]:
        results, _events = self.run_with_events(requests, collect_events=False)
        return results

    @torch.inference_mode()
    def run_with_events(
        self,
        requests: list[ServingRequest],
        *,
        collect_events: bool = True,
    ) -> tuple[list[ServingResult], list[ServingTokenEvent]]:
        self._reset_run_state(requests)
        waiting = ServingQueue(list(enumerate(requests)))
        active: list[_ActiveRequest] = []
        indexed_results: list[tuple[int, ServingResult]] = []
        events: list[ServingTokenEvent] | None = [] if collect_events else None
        step = 0

        while waiting or active:
            self.stats.scheduler_steps += 1
            if self.decode_first and active:
                decoded_results, active = self._decode_active(active, step, events=events)
                indexed_results.extend(decoded_results)

            admitted = self._admit_ready_requests(waiting, step, len(active))
            if admitted:
                admitted_results, admitted_active = self._prefill_many(admitted, step, events=events)
                indexed_results.extend(admitted_results)
                active.extend(admitted_active)

            if not self.decode_first and active:
                decoded_results, active = self._decode_active(active, step + 1, events=events)
                indexed_results.extend(decoded_results)
            step += 1

            next_arrival_step = waiting.next_arrival_step()
            if next_arrival_step is not None and not active and next_arrival_step > step:
                step = next_arrival_step

        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])], events or []

    def iter_events(self, requests: list[ServingRequest]) -> Iterator[ServingTokenEvent]:
        with torch.inference_mode():
            self._reset_run_state(requests)
            waiting = ServingQueue(list(enumerate(requests)))
            active: list[_ActiveRequest] = []
            step = 0

            while waiting or active:
                self.stats.scheduler_steps += 1
                step_events: list[ServingTokenEvent] = []
                if self.decode_first and active:
                    _decoded_results, active = self._decode_active(active, step, events=step_events)

                admitted = self._admit_ready_requests(waiting, step, len(active))
                if admitted:
                    _admitted_results, admitted_active = self._prefill_many(admitted, step, events=step_events)
                    active.extend(admitted_active)

                if not self.decode_first and active:
                    _decoded_results, active = self._decode_active(active, step + 1, events=step_events)
                for event in step_events:
                    yield event
                step += 1

                next_arrival_step = waiting.next_arrival_step()
                if next_arrival_step is not None and not active and next_arrival_step > step:
                    step = next_arrival_step

    def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self._reset_capacity(max_seq_len=max_seq_len, queued_requests=0, external_cache=external_cache)
        self._online_waiting = ServingQueue()
        self._online_active = []
        self._online_step = 0
        self._online_next_index = 0

    def submit_online(self, request: ServingRequest) -> None:
        waiting = self._require_online_waiting()
        waiting.push(self._online_next_index, request)
        self._online_next_index += 1
        self.stats.queued_requests += 1

    def has_online_work(self) -> bool:
        waiting = self._online_waiting
        return bool(waiting) or bool(self._online_active)

    @torch.inference_mode()
    def step_online(self) -> list[ServingTokenEvent]:
        waiting = self._require_online_waiting()
        if not waiting and not self._online_active:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        active = self._online_active
        self.stats.scheduler_steps += 1
        if self.decode_first and active:
            _decoded_results, active = self._decode_active(active, step, events=events)

        admitted = self._admit_ready_requests(waiting, step, len(active))
        if admitted:
            _admitted_results, admitted_active = self._prefill_many(admitted, step, events=events)
            active.extend(admitted_active)

        if not self.decode_first and active:
            _decoded_results, active = self._decode_active(active, step + 1, events=events)
        self._online_active = active
        self._online_step = step + 1

        next_arrival_step = waiting.next_arrival_step()
        if next_arrival_step is not None and not active and next_arrival_step > self._online_step:
            self._online_step = next_arrival_step
        return events

    def _reset_run_state(self, requests: list[ServingRequest]) -> None:
        max_seq_len = max((len(request.prompt) + request.max_new_tokens for request in requests), default=1)
        self._reset_capacity(max_seq_len=max(1, max_seq_len), queued_requests=len(requests))

    def _reset_capacity(
        self,
        *,
        max_seq_len: int,
        queued_requests: int,
        external_cache: object | None = None,
    ) -> None:
        self.stats = ServingStats()
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes = {}
        self._prefix_order = []
        self._cache_views = {}
        self._reported_static_graph_miss = False
        total_rows = self.max_active_requests + self.prefix_cache_capacity
        if external_cache is not None:
            self._cache = external_cache
        else:
            self._cache = self._allocate_cache(max(1, total_rows), max_seq_len)
        if not hasattr(self._cache, "for_rows"):
            raise ValueError("model cache must support row views for persistent serving")
        self._row_seq_lens = [0 for _ in range(total_rows)]
        self._free_active_rows = list(reversed(range(self.max_active_requests)))
        self._free_prefix_rows = list(reversed(range(self.max_active_requests, total_rows)))
        self.stats.persistent_cache_rows = total_rows
        self.stats.queued_requests = queued_requests
        self._online_waiting = None
        self._online_active = []
        self._online_step = 0
        self._online_next_index = 0

    def _admit_ready_requests(
        self,
        waiting: ServingQueue,
        step: int,
        active_count: int,
    ) -> list[tuple[int, ServingRequest]]:
        capacity = self.max_active_requests - active_count
        min_free_rows = env_int("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_FREE_ROWS", 1, minimum=1)
        if active_count > 0 and capacity < min(min_free_rows, self.max_active_requests):
            return []
        min_ready_requests = env_int("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS", 1, minimum=1)
        if active_count > 0 and min_ready_requests > 1:
            min_ready_requests = min(min_ready_requests, capacity)
            if waiting.ready_count(step=step) < min_ready_requests:
                return []
        return waiting.pop_admissible(
            step=step,
            capacity=capacity,
            token_budget=self.prefill_token_budget,
            token_cost=self._prefill_token_cost,
            priority_key=self._admission_priority,
        )

    def _prefill_token_cost(self, request: ServingRequest) -> int:
        prefix_hit_tokens = self._reusable_prefix_hit_tokens(request.prompt)
        return max(1, len(request.prompt) - prefix_hit_tokens)

    def _admission_priority(self, item: _QueuedRequest) -> tuple[object, ...]:
        prefix_hit_tokens = self._reusable_prefix_hit_tokens(item.request.prompt)
        return (-prefix_hit_tokens, item.request.arrival_step, item.sequence)

    def _prefill_many(
        self,
        indexed_requests: list[tuple[int, ServingRequest]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        timing_start_s = time.perf_counter() if self.profile_timings else 0.0
        self.stats.prefill_admitted_requests += len(indexed_requests)
        indexed_results: list[tuple[int, ServingResult]] = []
        active: list[_ActiveRequest] = []
        batchable: dict[int, list[tuple[int, ServingRequest, int]]] = defaultdict(list)
        prefix_batchable: dict[tuple[int, int], list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = defaultdict(list)
        pad_prefix_suffixes = env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False)

        for original_index, request in indexed_requests:
            if not request.prompt:
                raise ValueError("request prompt must contain at least one token")
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            reusable_prefix_tokens = match.depth if reusable is not None else 0
            if request.max_new_tokens == 0:
                indexed_results.append(
                    (
                        original_index,
                        ServingResult(
                            request.request_id,
                            request.prompt,
                            0,
                            request.arrival_step,
                            step,
                            step,
                        ),
                    )
                )
                continue
            if reusable is not None and reusable_prefix_tokens > 0:
                suffix_len = len(request.prompt) - reusable_prefix_tokens
                batch_suffix_len = -1 if pad_prefix_suffixes else suffix_len
                prefix_batchable[(reusable_prefix_tokens, batch_suffix_len)].append(
                    (original_index, request, reusable_prefix_tokens, reusable)
                )
            else:
                batchable[len(request.prompt)].append((original_index, request, 0))

        for group in prefix_batchable.values():
            active.extend(self._prefill_prefix_batch(group, step, events=events))

        plain_group = [item for group in batchable.values() for item in group]
        shared_prefix_active = self._prefill_common_prefix_batch(plain_group, step, events=events)
        if shared_prefix_active is not None:
            active.extend(shared_prefix_active)
        else:
            for group in batchable.values():
                if len(group) == 1:
                    original_index, request, prefix_hit_tokens = group[0]
                    active.append(
                        self._prefill_one(
                            original_index,
                            request,
                            step,
                            prefix_hit_tokens,
                            None,
                            events=events,
                        )
                    )
                else:
                    active.extend(self._prefill_batch(group, step, events=events))
        if self.profile_timings:
            self.stats.prefill_wall_ms += (time.perf_counter() - timing_start_s) * 1000.0
        return indexed_results, active

    def _prefill_common_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if len(group) <= 1 or self.prefix_cache_capacity <= 0:
            return None
        prefix_tokens = _common_prefix_token_count([request.prompt for _index, request, _hit in group])
        min_prefix_tokens = 16
        if prefix_tokens < min_prefix_tokens:
            return None
        prefix_row = self._acquire_prefix_row()
        if prefix_row is None:
            return None
        try:
            prefix_ids = torch.tensor(
                [group[0][1].prompt[:prefix_tokens]],
                device=self.device,
                dtype=torch.long,
            )
            prefix_logits, _ = self._prefill_logits(prefix_ids, cache=self._cache_view([prefix_row]))
            self._refresh_row_seq_len_from_cache(prefix_row, prefix_tokens)
            self._record_model_call("prefill", 1, tokens=prefix_ids.numel())
            self.stats.prefill_common_prefix_batches += 1
            prefix_tuple = tuple(group[0][1].prompt[:prefix_tokens])
            self._store_reusable_prefix_tokens(
                ("common_prefix", prefix_tuple),
                "__common_prefix__",
                prefix_tuple,
                prefix_row,
                prefix_logits,
            )

            padded_active = self._prefill_common_prefix_padded_suffix_batch(
                group,
                prefix_row=prefix_row,
                prefix_tokens=prefix_tokens,
                step=step,
                events=events,
            )
            if padded_active is not None:
                return padded_active

            active: list[_ActiveRequest] = []
            suffix_groups: dict[int, list[tuple[int, ServingRequest, int]]] = defaultdict(list)
            for original_index, request, prefix_hit_tokens in group:
                del prefix_hit_tokens
                suffix_groups[len(request.prompt) - prefix_tokens].append((original_index, request, 0))
            for suffix_group in suffix_groups.values():
                rows = [self._acquire_active_row() for _ in suffix_group]
                suffixes = []
                for row, (_original_index, request, _prefix_hit_tokens) in zip(rows, suffix_group):
                    self._copy_prefix(prefix_row, row, prefix_tokens)
                    suffixes.append(request.prompt[prefix_tokens:])

                if not suffixes or not suffixes[0]:
                    logits = prefix_logits.expand(len(suffix_group), -1, -1)
                else:
                    input_ids = torch.tensor(suffixes, device=self.device, dtype=torch.long)
                    logits, _ = self._prefill_logits(input_ids, cache=self._cache_view(rows))
                    self._record_model_call("prefill", len(suffix_group), tokens=input_ids.numel())
                next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

                for row_index, (original_index, request, prefix_hit_tokens) in enumerate(suffix_group):
                    row = rows[row_index]
                    seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
                    self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
                    next_token = int(next_tokens[row_index])
                    state = _ActiveRequest(
                        original_index=original_index,
                        request=request,
                        tokens=[*request.prompt, next_token],
                        generated=1,
                        row=row,
                        last_token=next_token,
                        seq_len=seq_len,
                        prefix_hit_tokens=prefix_hit_tokens,
                        started_step=step,
                    )
                    self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                    active.append(state)
            return active
        finally:
            self._release_prefix_row(prefix_row)

    def _prefill_common_prefix_padded_suffix_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        *,
        prefix_row: int,
        prefix_tokens: int,
        step: int,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False):
            return None
        suffixes = [request.prompt[prefix_tokens:] for _original_index, request, _hit in group]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0:
            return None
        max_suffix_len = max(suffix_lengths)
        static_batch = self._prefill_static_batch_size(len(group))
        padded_batch_size = max(len(group), static_batch)
        padding_tokens = padded_batch_size * max_suffix_len - sum(suffix_lengths)
        max_padding_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_MAX_PADDING_TOKENS",
            4096,
            minimum=0,
        )
        if padding_tokens > max_padding_tokens:
            return None

        rows = [self._acquire_active_row() for _ in group]
        try:
            self._copy_prefix_to_rows(prefix_row, rows, prefix_tokens)
            padded_suffixes = [
                [*suffix, *([0] * (max_suffix_len - len(suffix)))]
                for suffix in suffixes
            ]
            pad_rows: list[int] = []
            if padded_batch_size > len(group):
                dummy_suffix = padded_suffixes[0] if padded_suffixes else [0] * max_suffix_len
                for _ in range(padded_batch_size - len(group)):
                    pad_row = self._acquire_active_row_or_none()
                    if pad_row is None:
                        break
                    pad_rows.append(pad_row)
                    self._copy_prefix(prefix_row, pad_row, prefix_tokens)
                    padded_suffixes.append(list(dummy_suffix))
            all_rows = rows + pad_rows
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(
                [length - 1 for length in suffix_lengths]
                + [max_suffix_len - 1] * len(pad_rows),
                device=self.device,
                dtype=torch.long,
            )
            logits = self._forward_selected_logits(
                input_ids,
                cache=self._cache_view(all_rows),
                logit_positions=logit_positions,
            )
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            if logits is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
            self.stats.prefill_padded_suffix_batches += 1
            next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
                row = rows[row_index]
                self._set_cache_row_seq_len(row, len(request.prompt))
                self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
                next_token = int(next_tokens[row_index])
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=1,
                    row=row,
                    last_token=next_token,
                    seq_len=self._cache_row_seq_len(row, len(request.prompt)),
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
                self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                active.append(state)
            return active
        except Exception:
            for row in rows:
                self._release_active_row(row)
            raise

    def _prefill_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        suffix_lengths = [len(request.prompt) - prefix_hit_tokens for _index, request, prefix_hit_tokens, _reusable in group]
        if len(set(suffix_lengths)) > 1:
            padded_active = self._prefill_prefix_padded_suffix_batch(group, step, events=events)
            if padded_active is not None:
                return padded_active
            active: list[_ActiveRequest] = []
            by_suffix_len: dict[int, list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = defaultdict(list)
            for item, suffix_len in zip(group, suffix_lengths):
                by_suffix_len[suffix_len].append(item)
            for suffix_group in by_suffix_len.values():
                active.extend(self._prefill_prefix_batch(suffix_group, step, events=events))
            return active

        self.stats.prefill_prefix_reuse_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        suffixes = []
        reusable_logits = []
        self._copy_reusable_prefixes_to_rows(rows, group)
        for _row, (_original_index, request, prefix_hit_tokens, reusable) in zip(rows, group):
            suffixes.append(request.prompt[prefix_hit_tokens:])
            reusable_logits.append(reusable.logits)

        if suffixes and suffixes[0]:
            input_ids = torch.tensor(suffixes, device=self.device, dtype=torch.long)
            logits, _ = self._prefill_logits(input_ids, cache=self._cache_view(rows))
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
        else:
            logits = torch.cat([item.to(self.device) for item in reusable_logits], dim=0)

        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
        active = []
        for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
            next_token = int(next_tokens[row_index])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefill_prefix_padded_suffix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False):
            return None
        suffixes = [request.prompt[prefix_hit_tokens:] for _index, request, prefix_hit_tokens, _reusable in group]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0 or len(set(suffix_lengths)) <= 1:
            return None
        max_suffix_len = max(suffix_lengths)
        padding_tokens = len(suffix_lengths) * max_suffix_len - sum(suffix_lengths)
        max_padding_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_MAX_PADDING_TOKENS",
            1024,
            minimum=0,
        )
        if padding_tokens > max_padding_tokens:
            return None

        rows = [self._acquire_active_row() for _ in group]
        try:
            self._copy_reusable_prefixes_to_rows(rows, group)

            padded_suffixes = [
                [*suffix, *([0] * (max_suffix_len - len(suffix)))]
                for suffix in suffixes
            ]
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(
                [length - 1 for length in suffix_lengths],
                device=self.device,
                dtype=torch.long,
            )
            logits = self._forward_selected_logits(
                input_ids,
                cache=self._cache_view(rows),
                logit_positions=logit_positions,
            )
            if logits is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
            self.stats.prefill_prefix_reuse_batches += 1
            self.stats.prefill_padded_suffix_batches += 1
            next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
                row = rows[row_index]
                self._set_cache_row_seq_len(row, len(request.prompt))
                self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
                next_token = int(next_tokens[row_index])
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=1,
                    row=row,
                    last_token=next_token,
                    seq_len=self._cache_row_seq_len(row, len(request.prompt)),
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
                self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                active.append(state)
            return active
        except Exception:
            for row in rows:
                self._release_active_row(row)
            raise

    def _copy_reusable_prefixes_to_rows(
        self,
        rows: list[int],
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
    ) -> None:
        copy_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row, (_original_index, _request, prefix_hit_tokens, reusable) in zip(rows, group):
            copy_groups[(reusable.row, prefix_hit_tokens)].append(row)
            self.stats.prefix_reuse_requests += 1
            self.stats.prefix_reuse_tokens += prefix_hit_tokens
        for (source_row, prefix_hit_tokens), dest_rows in copy_groups.items():
            self._copy_prefix_to_rows(source_row, dest_rows, prefix_hit_tokens)

    def _prefill_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        self.stats.prefill_plain_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        prompts = torch.tensor([request.prompt for _, request, _ in group], device=self.device, dtype=torch.long)
        cache_view = self._cache_view(rows)
        logits, _ = self._prefill_logits(prompts, cache=cache_view)
        self._record_model_call("prefill", len(group), tokens=prompts.numel())
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
            next_token = int(next_tokens[row_index])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefill_one(
        self,
        original_index: int,
        request: ServingRequest,
        step: int,
        prefix_hit_tokens: int,
        reusable: _ReusablePrefix | None,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> _ActiveRequest:
        self.stats.prefill_single_batches += 1
        row = self._acquire_active_row()
        suffix = request.prompt
        logits: Tensor
        if reusable is not None and prefix_hit_tokens > 0:
            self._copy_prefix(reusable.row, row, prefix_hit_tokens)
            suffix = request.prompt[prefix_hit_tokens:]
            self.stats.prefix_reuse_requests += 1
            self.stats.prefix_reuse_tokens += prefix_hit_tokens

        if suffix:
            input_ids = torch.tensor([suffix], device=self.device, dtype=torch.long)
            logits, _ = self._prefill_logits(input_ids, cache=self._cache_view([row]))
            self._record_model_call("prefill", 1, tokens=input_ids.numel())
        elif reusable is not None:
            logits = reusable.logits.to(self.device)
        else:
            raise RuntimeError("empty prompt suffix without a reusable prefix")

        next_token = int(self._sample_logits(logits[:, -1, :]).item())
        seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
        self._store_reusable_prefix(request.request_id, request.prompt, row, logits)
        state = _ActiveRequest(
            original_index=original_index,
            request=request,
            tokens=[*request.prompt, next_token],
            generated=1,
            row=row,
            last_token=next_token,
            seq_len=seq_len,
            prefix_hit_tokens=prefix_hit_tokens,
            started_step=step,
        )
        self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
        return state

    def _decode_active(
        self,
        active: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        indexed_results: list[tuple[int, ServingResult]] = []
        live: list[_ActiveRequest] = []
        for state in active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                indexed_results.append((state.original_index, self._finish_and_release(state, step)))
            else:
                live.append(state)

        next_active: list[_ActiveRequest] = []
        groups = [live] if self._can_decode_ragged(live) else self._decode_groups(live)
        for group in groups:
            if self._can_decode_ragged(group):
                decoded = self._decode_ragged_batch(group, step, events=events)
            else:
                decoded = (
                    self._decode_batch(group, step, events=events)
                    if len(group) > 1
                    else [self._decode_one(group[0], step, events=events)]
                )
            for item, state in zip(decoded, group):
                if isinstance(item, ServingResult):
                    indexed_results.append((state.original_index, item))
                else:
                    next_active.append(item)
        return indexed_results, next_active

    def _decode_groups(self, states: list[_ActiveRequest]) -> list[list[_ActiveRequest]]:
        grouped: dict[int, list[_ActiveRequest]] = defaultdict(list)
        for state in states:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            grouped[state.seq_len].append(state)
        return list(grouped.values())

    def _decode_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest | ServingResult]:
        for state in states:
            self._set_cache_row_seq_len(state.row, state.seq_len)
        rows = [state.row for state in states]
        input_ids = torch.tensor([[state.last_token] for state in states], device=self.device, dtype=torch.long)
        cache_view = self._cache_view(rows)
        graph_token = self._try_static_token_graph(input_ids, cache_view)
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._static_decode_logits(input_ids, cache_view)
            next_token_tensor = self._sample_logits(logits[:, -1, :])
        self._record_model_call("decode", len(states), tokens=len(states))
        next_tokens = next_token_tensor.detach().cpu().tolist()

        decoded: list[_ActiveRequest | ServingResult] = []
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
            self._set_cache_row_seq_len(state.row, next_seq_len)
            state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        return decoded

    def _decode_one(
        self,
        state: _ActiveRequest,
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> _ActiveRequest | ServingResult:
        self._set_cache_row_seq_len(state.row, state.seq_len)
        input_ids = torch.tensor([[state.last_token]], device=self.device, dtype=torch.long)
        cache_view = self._cache_view([state.row])
        graph_token = self._try_static_token_graph(input_ids, cache_view)
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._static_decode_logits(input_ids, cache_view)
            next_token_tensor = self._sample_logits(logits[:, -1, :])
        self._record_model_call("decode", 1, tokens=1)
        next_token = int(next_token_tensor.item())
        state.tokens.append(next_token)
        state.generated += 1
        state.last_token = next_token
        next_seq_len = state.seq_len + 1
        self._set_cache_row_seq_len(state.row, next_seq_len)
        state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
        finished = self._should_finish_after_decode(state)
        self._record_token_event(events, state, next_token, step, finished=finished)
        if finished:
            return self._finish_and_release(state, step)
        return state

    def _decode_ragged_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest | ServingResult]:
        prepare_start_s = time.perf_counter() if self.profile_timings else 0.0
        rows = [state.row for state in states]
        decode_rows = self._ragged_decode_bucket_rows(rows)
        pad_token = states[0].last_token
        input_tokens = [
            states[index].last_token if index < len(states) else pad_token
            for index, _row in enumerate(decode_rows)
        ]
        input_ids = torch.tensor([[token] for token in input_tokens], device=self.device, dtype=torch.long)
        row_indices = torch.tensor(decode_rows, dtype=torch.long, device=self.device)
        seq_lens = self._seq_lens_tensor(states, rows=decode_rows)
        if self.profile_timings:
            self.stats.decode_ragged_prepare_ms += (time.perf_counter() - prepare_start_s) * 1000.0
        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        graph_token = self._try_ragged_token_graph(input_ids, seq_lens, row_indices)
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._ragged_decode_logits(input_ids, seq_lens, row_indices)
            next_token_tensor = self._sample_logits(logits[:, -1, :])
        self._record_model_call("decode", len(decode_rows), tokens=len(decode_rows), ragged=True)
        if self.profile_timings:
            self.stats.decode_ragged_model_ms += (time.perf_counter() - model_start_s) * 1000.0
        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        next_tokens = next_token_tensor[: len(states)].detach().cpu().tolist()
        if self.profile_timings:
            self.stats.decode_ragged_cpu_tokens_ms += (time.perf_counter() - cpu_tokens_start_s) * 1000.0
        sync_cache_seq_lens = env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_SYNC_CACHE_SEQ_LENS", False)

        state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
        decoded: list[_ActiveRequest | ServingResult] = []
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
            if sync_cache_seq_lens:
                self._set_cache_row_seq_len(state.row, next_seq_len)
                state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
            else:
                self._remember_row_seq_len(state.row, next_seq_len)
                state.seq_len = next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        if self.profile_timings:
            self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
        return decoded

    def _ragged_decode_bucket_rows(self, rows: list[int]) -> list[int]:
        if not env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKETS", True):
            return rows
        active_count = len(rows)
        if active_count <= 1:
            return rows
        capacity = min(
            self.max_active_requests,
            env_int(
                "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKET_CAPACITY",
                self.max_active_requests,
                minimum=1,
            ),
        )
        if active_count >= capacity:
            return rows
        bucket_size = min(capacity, 1 << (active_count - 1).bit_length())
        if bucket_size <= active_count:
            return rows
        row_set = set(rows)
        bucketed = list(rows)
        for row in self._free_active_rows:
            if row in row_set:
                continue
            bucketed.append(row)
            if len(bucketed) >= bucket_size:
                return bucketed
        return rows

    def _finish_and_release(self, state: _ActiveRequest, step: int) -> ServingResult:
        result = ServingResult(
            state.request.request_id,
            tuple(state.tokens),
            state.prefix_hit_tokens,
            state.request.arrival_step,
            state.started_step,
            step,
        )
        self._release_active_row(state.row)
        return result

    def _store_reusable_prefix(self, request_id: str, tokens: tuple[int, ...], source_row: int, logits: Tensor) -> None:
        if not self.store_full_prompt_prefixes:
            return
        self._store_reusable_prefix_tokens(None, request_id, tokens, source_row, logits)

    def _store_reusable_prefix_tokens(
        self,
        route_id: Hashable | None,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor,
    ) -> None:
        if not self.store_reusable_prefixes or not env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE", True):
            return
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        old_prefix = self.reusable_prefixes.pop(entry.route_id, None)
        if old_prefix is not None:
            self._clear_physical_row(old_prefix.row)
            if entry.route_id in self._prefix_order:
                self._prefix_order.remove(entry.route_id)
            self._free_prefix_rows.append(old_prefix.row)
            self._free_prefix_rows.sort()
        prefix_row = self._acquire_prefix_row()
        if prefix_row is None:
            return
        self._copy_prefix(source_row, prefix_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            prefix_row,
            logits[:, -1:, :].detach().clone().cpu(),
        )
        self._prefix_order.append(entry.route_id)

    def _reusable_prefix_hit_tokens(self, prompt: tuple[int, ...]) -> int:
        match, entry = self.prefix_cache.lookup(prompt)
        if entry is None or entry.route_id not in self.reusable_prefixes:
            return 0
        return match.depth

    def _copy_prefix(self, source_row: int, dest_row: int, tokens: int) -> None:
        cache = self._require_cache()
        cache.copy_prefix_from(cache, tokens, source_row=source_row, dest_row=dest_row)  # type: ignore[attr-defined]
        self._remember_row_seq_len(dest_row, tokens)

    def _copy_prefix_to_rows(self, source_row: int, dest_rows: list[int], tokens: int) -> None:
        if not dest_rows:
            return
        if self._copy_prefix_to_rows_dense(source_row, dest_rows, tokens):
            return
        for row in dest_rows:
            self._copy_prefix(source_row, row, tokens)

    def _copy_prefix_to_rows_dense(self, source_row: int, dest_rows: list[int], tokens: int) -> bool:
        if tokens <= 0:
            for row in dest_rows:
                self._remember_row_seq_len(row, 0)
            return True
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if not layers:
            return False
        try:
            for layer in layers:
                keys = getattr(layer, "keys")
                values = getattr(layer, "values")
                physical = getattr(layer, "_physical_row", None)
                src = int(physical(source_row) if callable(physical) else source_row)
                dst = tuple(int(physical(row) if callable(physical) else row) for row in dest_rows)
                if tokens > keys.size(2) or tokens > values.size(2):
                    return False
                seq_len_for_rows = getattr(layer, "seq_len_for_rows", None)
                if callable(seq_len_for_rows):
                    if int(seq_len_for_rows((src,))) < tokens:
                        return False
                else:
                    seq_lens = getattr(layer, "_seq_lens", None)
                    if isinstance(seq_lens, list) and (src >= len(seq_lens) or int(seq_lens[src]) < tokens):
                        return False
                source_keys = keys[src : src + 1, :, :tokens, :].expand(len(dst), -1, -1, -1)
                source_values = values[src : src + 1, :, :tokens, :].expand(len(dst), -1, -1, -1)
                span = _contiguous_int_span(dst)
                if span is not None:
                    start, end = span
                    keys[start:end, :, :tokens, :].copy_(source_keys)
                    values[start:end, :, :tokens, :].copy_(source_values)
                else:
                    index = torch.tensor(dst, dtype=torch.long, device=keys.device)
                    keys[:, :, :tokens, :].index_copy_(0, index, source_keys)
                    values[:, :, :tokens, :].index_copy_(0, index, source_values)
                setter = getattr(layer, "_set_rows_seq_len", None)
                if callable(setter):
                    setter(dst, tokens)
                else:
                    seq_lens = getattr(layer, "_seq_lens", None)
                    if isinstance(seq_lens, list):
                        for row in dst:
                            seq_lens[row] = int(tokens)
            for row in dest_rows:
                self._remember_row_seq_len(row, tokens)
            return True
        except Exception:
            return False

    def _acquire_active_row(self) -> int:
        if not self._free_active_rows:
            raise RuntimeError("no active serving rows available")
        row = self._free_active_rows.pop()
        self._clear_physical_row(row)
        return row

    def _acquire_active_row_or_none(self) -> int | None:
        if not self._free_active_rows:
            return None
        row = self._free_active_rows.pop()
        self._clear_physical_row(row)
        return row

    def _release_active_row(self, row: int) -> None:
        self._clear_physical_row(row)
        if row not in self._free_active_rows:
            self._free_active_rows.append(row)

    def _prefill_static_batch_size(self, request_count: int) -> int:
        bucket = env_int("TORCHINFERNO_CONTINUOUS_PREFILL_STATIC_BATCH", self.max_active_requests, minimum=1)
        available = request_count + len(self._free_active_rows)
        return min(bucket, self.max_active_requests, available)

    def _acquire_prefix_row(self) -> int | None:
        if self.prefix_cache_capacity == 0:
            return None
        if self._free_prefix_rows:
            return self._free_prefix_rows.pop()
        while self._prefix_order:
            route_id = self._prefix_order.pop(0)
            prefix = self.reusable_prefixes.pop(route_id, None)
            if prefix is not None:
                self._clear_physical_row(prefix.row)
                return prefix.row
        return None

    def _release_prefix_row(self, row: int) -> None:
        self._clear_physical_row(row)
        if row not in self._free_prefix_rows:
            self._free_prefix_rows.append(row)
            self._free_prefix_rows.sort()

    def _cache_view(self, rows: list[int]) -> object:
        row_key = tuple(rows)
        view = self._cache_views.get(row_key)
        if view is None:
            view = self._require_cache().for_rows(row_key)  # type: ignore[attr-defined]
            self._cache_views[row_key] = view
        return view

    def _require_cache(self) -> object:
        if self._cache is None:
            raise RuntimeError("serving cache has not been initialized")
        return self._cache

    def _require_online_waiting(self) -> ServingQueue:
        if self._online_waiting is None:
            raise RuntimeError("online serving has not been initialized")
        return self._online_waiting

    def _clear_physical_row(self, row: int) -> None:
        self._require_cache().for_rows((row,)).clear_row(0)  # type: ignore[attr-defined]
        self._remember_row_seq_len(row, 0)

    def _allocate_cache(self, batch_size: int, max_seq_len: int) -> object:
        allocate_cache = getattr(self.model, "allocate_cache")
        if self.cache_backend != "dense":
            try:
                return allocate_cache(
                    batch_size,
                    max_seq_len=max_seq_len,
                    device=self.device,
                    cache_backend=self.cache_backend,
                    page_size=self.page_size,
                )
            except TypeError:
                try:
                    return allocate_cache(
                        batch_size,
                        max_seq_len=max_seq_len,
                        cache_backend=self.cache_backend,
                        page_size=self.page_size,
                    )
                except TypeError:
                    raise ValueError(f"model does not support cache_backend={self.cache_backend}") from None
        try:
            return allocate_cache(
                batch_size,
                max_seq_len=max_seq_len,
                device=self.device,
            )
        except TypeError:
            return allocate_cache(batch_size, max_seq_len=max_seq_len)

    def _forward_model(self, input_ids: Tensor, *, cache: object, use_cache: bool) -> tuple[Tensor, object | None]:
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if callable(forward):
            if self._prefer_sharded_logits():
                try:
                    return forward(
                        input_ids,
                        cache=cache,
                        use_cache=use_cache,
                        return_last_logits_only=True,
                        return_sharded_logits=True,
                    )
                except TypeError:
                    pass
            return forward(input_ids, cache=cache, use_cache=use_cache)
        raise TypeError("serving model must be callable or expose forward()")

    def _prefill_logits(self, input_ids: Tensor, *, cache: object) -> tuple[Tensor, object | None]:
        graph_logits = self._try_prefill_logits_graph(input_ids, cache)
        if graph_logits is not None:
            return graph_logits, cache
        return self._forward_model(input_ids, cache=cache, use_cache=True)

    def _try_prefill_logits_graph(self, input_ids: Tensor, cache: object) -> Tensor | None:
        graph = getattr(self.model, "try_prefill_logits_graph", None)
        if not callable(graph):
            return None
        capture_on_miss = env_flag("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", False)
        logits = self._call_prefill_graph(graph, input_ids, cache, capture_on_miss=capture_on_miss)
        if logits is None:
            self.stats.prefill_graph_misses += 1
            return None
        self.stats.prefill_graph_hits += 1
        return logits

    def _forward_selected_logits(
        self,
        input_ids: Tensor,
        *,
        cache: object,
        logit_positions: Tensor,
    ) -> Tensor | None:
        graph_logits = self._try_prefill_selected_logits_graph(
            input_ids,
            cache,
            logit_positions=logit_positions,
        )
        if graph_logits is not None:
            return graph_logits
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if not callable(forward):
            raise TypeError("serving model must be callable or expose forward()")
        kwargs: dict[str, object] = {
            "cache": cache,
            "use_cache": True,
            "logit_positions": logit_positions,
        }
        if self._prefer_sharded_logits():
            kwargs["return_last_logits_only"] = False
            kwargs["return_sharded_logits"] = True
        try:
            logits, _cache = forward(input_ids, **kwargs)
        except TypeError:
            return None
        return logits

    def _try_prefill_selected_logits_graph(
        self,
        input_ids: Tensor,
        cache: object,
        *,
        logit_positions: Tensor,
    ) -> Tensor | None:
        graph = getattr(self.model, "try_prefill_selected_logits_graph", None)
        if not callable(graph):
            return None
        capture_on_miss = env_flag(
            "TORCHINFERNO_CONTINUOUS_SELECTED_PREFILL_CAPTURE",
            env_flag("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", False),
        )
        logits = self._call_prefill_graph(
            graph,
            input_ids,
            cache,
            capture_on_miss=capture_on_miss,
            logit_positions=logit_positions,
        )
        if logits is None:
            self.stats.prefill_graph_misses += 1
            return None
        self.stats.prefill_graph_hits += 1
        return logits

    @staticmethod
    def _call_prefill_graph(
        graph: Callable[..., Tensor | None],
        input_ids: Tensor,
        cache: object,
        *,
        capture_on_miss: bool,
        logit_positions: Tensor | None = None,
    ) -> Tensor | None:
        try:
            parameters = signature(graph).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "capture_on_miss" in parameters:
            if logit_positions is None:
                return graph(input_ids, cache, capture_on_miss=capture_on_miss)
            return graph(
                input_ids,
                cache,
                logit_positions=logit_positions,
                capture_on_miss=capture_on_miss,
            )
        if not capture_on_miss:
            return None
        if logit_positions is None:
            return graph(input_ids, cache)
        return graph(input_ids, cache, logit_positions=logit_positions)

    def _prefer_sharded_logits(self) -> bool:
        return int(getattr(self.model, "world_size", 1)) > 1 and callable(
            getattr(self.model, "_sample_next_token", None)
        )

    def _record_model_call(self, kind: str, batch_size: int, *, tokens: int, ragged: bool = False) -> None:
        if kind == "prefill":
            self.stats.prefill_model_calls += 1
            self.stats.prefill_batches += 1
            self.stats.prefill_tokens += tokens
        elif kind == "decode":
            self.stats.decode_model_calls += 1
            self.stats.decode_batches += 1
            self.stats.decode_tokens += tokens
            if ragged:
                self.stats.ragged_decode_batches += 1
                self.stats.ragged_decode_tokens += tokens
        else:
            raise ValueError(f"unknown model call kind: {kind}")
        self.stats.max_model_batch_size = max(self.stats.max_model_batch_size, batch_size)

    def _can_decode_ragged(self, states: list[_ActiveRequest]) -> bool:
        if not self.enable_ragged_decode:
            return False
        if len(states) <= 1:
            return False
        if len({state.seq_len for state in states}) <= 1 and not env_flag(
            "TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE",
            False,
        ):
            return False
        return (
            hasattr(self.model, "decode_ragged_logits")
            or hasattr(self.model, "try_decode_ragged_logits_graph")
            or hasattr(self.model, "try_decode_ragged_token_graph")
        )

    def _seq_lens_tensor(self, states: list[_ActiveRequest], *, rows: list[int] | None = None) -> Tensor:
        required = max([state.row for state in states] + list(rows or [0])) + 1
        if len(self._row_seq_lens) >= required:
            seq_lens = list(self._row_seq_lens[:required])
        else:
            seq_lens = [0 for _ in range(required)]
            for row, seq_len in enumerate(self._row_seq_lens):
                if row < required:
                    seq_lens[row] = int(seq_len)
        pad_seq_len = 0
        active_rows: set[int] = set()
        for state in states:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            seq_lens[state.row] = state.seq_len
            active_rows.add(state.row)
            pad_seq_len = max(pad_seq_len, state.seq_len)
        for row in rows or ():
            if row not in active_rows and 0 <= row < len(seq_lens) and seq_lens[row] <= 0:
                seq_lens[row] = pad_seq_len
        return torch.tensor(seq_lens, device=self.device, dtype=torch.long)

    def _cache_row_seq_len(self, row: int, fallback: int) -> int:
        if 0 <= row < len(self._row_seq_lens):
            seq_len = int(self._row_seq_lens[row])
            if seq_len > 0 or fallback <= 0:
                return seq_len
        seq_len = self._cache_row_seq_len_from_cache(row, fallback)
        self._remember_row_seq_len(row, seq_len)
        return seq_len

    def _refresh_row_seq_len_from_cache(self, row: int, fallback: int) -> int:
        seq_len = self._cache_row_seq_len_from_cache(row, fallback)
        self._remember_row_seq_len(row, seq_len)
        return seq_len

    def _cache_row_seq_len_from_cache(self, row: int, fallback: int) -> int:
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if layers:
            layer = layers[0]
            seq_len_for_rows = getattr(layer, "seq_len_for_rows", None)
            if callable(seq_len_for_rows):
                try:
                    return int(seq_len_for_rows((row,)))
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
                try:
                    return int(seq_lens[row])
                except Exception:
                    pass
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                return int(getattr(cache_view((row,)), "seq_len"))
            except Exception:
                pass
        return int(fallback)

    def _remember_row_seq_len(self, row: int, seq_len: int) -> None:
        if 0 <= row < len(self._row_seq_lens):
            self._row_seq_lens[row] = int(seq_len)

    def _set_cache_row_seq_len(self, row: int, seq_len: int) -> None:
        self._remember_row_seq_len(row, seq_len)
        cache = self._require_cache()
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                view = cache_view((row,))
                setter = getattr(view, "set_seq_len", None)
                if callable(setter):
                    setter(int(seq_len))
                    return
            except Exception:
                pass
        layers = tuple(getattr(cache, "layers", ()) or ())
        changed = False
        for layer in layers:
            setter = getattr(layer, "_set_rows_seq_len", None)
            if callable(setter):
                try:
                    setter((row,), int(seq_len))
                    changed = True
                    continue
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
                seq_lens[row] = int(seq_len)
                uniform = getattr(layer, "_uniform_seq_len", None)
                if isinstance(uniform, list) and uniform:
                    uniform[0] = int(seq_len) if all(value == int(seq_len) for value in seq_lens) else None
                changed = True
        if changed:
            return
        seq_lens = getattr(cache, "_seq_lens", None)
        if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
            seq_lens[row] = int(seq_len)

    def _try_ragged_token_graph(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
    ) -> Tensor | None:
        decode_graph = getattr(self.model, "try_decode_ragged_token_graph", None)
        if decode_graph is None:
            return None
        token = decode_graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            temperature=self.temperature,
        )
        if token is None:
            self.stats.decode_graph_misses += 1
        return token

    def _try_static_token_graph(self, input_ids: Tensor, cache: object) -> Tensor | None:
        decode_graph = getattr(self.model, "try_decode_one_token_graph", None)
        if decode_graph is None:
            self._report_static_graph_miss(input_ids, cache, "no_token_graph")
            return None
        token = decode_graph(input_ids, cache, temperature=self.temperature)
        if token is None:
            self.stats.decode_graph_misses += 1
            self._report_static_graph_miss(input_ids, cache, "token_graph_returned_none")
        return token

    def _static_decode_logits(self, input_ids: Tensor, cache: object) -> Tensor:
        decode_graph = getattr(self.model, "try_decode_one_token_logits_graph", None)
        if decode_graph is not None:
            logits = decode_graph(input_ids, cache)
            if logits is not None:
                self.stats.decode_graph_hits += 1
                return logits
            self.stats.decode_graph_misses += 1
            self._report_static_graph_miss(input_ids, cache, "logits_graph_returned_none")
        else:
            self._report_static_graph_miss(input_ids, cache, "no_logits_graph")
        logits, _ = self._forward_model(input_ids, cache=cache, use_cache=True)
        return logits

    def _report_static_graph_miss(self, input_ids: Tensor, cache: object, reason: str) -> None:
        if self._reported_static_graph_miss or not env_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
            return
        self._reported_static_graph_miss = True
        cache_seq_len: object
        try:
            cache_seq_len = getattr(cache, "seq_len", None)
        except Exception as exc:
            cache_seq_len = f"error:{exc!r}"
        print(
            "continuous_decode_graph_miss "
            f"reason={reason} "
            f"batch={int(input_ids.size(0))} "
            f"input_cuda={bool(input_ids.is_cuda)} "
            f"temperature={float(self.temperature)} "
            f"cache_seq_len={cache_seq_len} "
            f"token_failed={getattr(self.model, '_decode_graph_failed', None)} "
            f"logits_failed={getattr(self.model, '_decode_logits_graph_failed', None)}",
            flush=True,
        )

    def _ragged_decode_logits(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        decode_graph = getattr(self.model, "try_decode_ragged_logits_graph", None)
        if decode_graph is not None:
            logits = decode_graph(
                input_ids,
                self._require_cache(),
                seq_lens=seq_lens,
                row_indices=row_indices,
            )
            if logits is not None:
                self.stats.decode_graph_hits += 1
                return logits
            self.stats.decode_graph_misses += 1
        decode = getattr(self.model, "decode_ragged_logits", None)
        if decode is None:
            raise RuntimeError("model does not support ragged decode")
        return decode(input_ids, self._require_cache(), seq_lens=seq_lens, row_indices=row_indices)

    def _sample_logits(self, logits: Tensor) -> Tensor:
        sampler = getattr(self.model, "_sample_next_token", None)
        if callable(sampler):
            return sampler(logits, self.temperature).to(self.device)
        return sample_next_token(logits, self.temperature).to(self.device)

    @staticmethod
    def _record_token_event(
        events: list[ServingTokenEvent] | None,
        state: _ActiveRequest,
        token: int,
        step: int,
        *,
        finished: bool,
    ) -> None:
        if events is None:
            return
        events.append(
            ServingTokenEvent(
                request_id=state.request.request_id,
                token=int(token),
                step=step,
                generated=state.generated,
                finished=finished,
            )
        )

    @staticmethod
    def _should_finish_before_decode(state: _ActiveRequest) -> bool:
        if state.request.eos_token_id is not None and state.last_token == state.request.eos_token_id:
            return True
        return state.generated >= state.request.max_new_tokens

    @staticmethod
    def _should_finish_after_decode(state: _ActiveRequest) -> bool:
        if state.request.eos_token_id is not None and state.last_token == state.request.eos_token_id:
            return True
        return state.generated >= state.request.max_new_tokens


def _common_prefix_token_count(prompts: list[tuple[int, ...]]) -> int:
    if len(prompts) <= 1:
        return 0
    min_len = min((len(prompt) for prompt in prompts), default=0)
    if min_len <= 1:
        return 0
    prefix_tokens = 0
    for offset in range(min_len - 1):
        token = prompts[0][offset]
        if any(prompt[offset] != token for prompt in prompts[1:]):
            break
        prefix_tokens += 1
    return prefix_tokens
