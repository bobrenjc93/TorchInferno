from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from inspect import signature
from typing import Callable, Hashable, Iterator, Optional

import torch
from torch import Tensor

from torchinferno.runtime.options import env_flag, env_int
from torchinferno.runtime.prefix_cache import PrefixCacheIndex
from torchinferno.runtime.sampling import sample_next_token


def _fi_decode_graph_mode() -> str:
    raw = os.environ.get("TORCHINFERNO_FI_DECODE_GRAPH", "sampled").strip().lower()
    if raw in {"1", "true", "yes", "on", "always"}:
        return "always"
    if raw in {"0", "false", "no", "off", "never", ""}:
        return "off"
    if raw in {"sample", "sampled", "auto"}:
        return "sampled"
    return "off"


def _preferred_prefix_rows() -> tuple[int, ...]:
    raw = os.environ.get("TORCHINFERNO_CONTINUOUS_PREFERRED_PREFIX_ROWS", "48,53,68,69,128")
    rows: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            row = int(token)
        except ValueError:
            continue
        if row < 0 or row in seen:
            continue
        rows.append(row)
        seen.add(row)
    return tuple(rows)


def _enable_runtime_cache_capture_sync(cache: object) -> None:
    try:
        delattr(cache, "_skip_capture_sync")
    except AttributeError:
        pass
    except Exception:
        pass


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    arrival_step: int = 0
    eos_token_id: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        stop_ids = {int(token_id) for token_id in self.stop_token_ids if int(token_id) >= 0}
        if self.eos_token_id is not None:
            eos_token_id = int(self.eos_token_id)
            object.__setattr__(self, "eos_token_id", eos_token_id)
            if eos_token_id >= 0:
                stop_ids.add(eos_token_id)
        object.__setattr__(self, "stop_token_ids", tuple(sorted(stop_ids)))

    def is_stop_token(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids


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
    prefill_copy_ms: float = 0.0
    prefill_forward_ms: float = 0.0
    prefill_setup_ms: float = 0.0
    prefill_sample_ms: float = 0.0
    prefill_state_ms: float = 0.0
    decode_ragged_prepare_ms: float = 0.0
    decode_ragged_model_ms: float = 0.0
    decode_ragged_model_gpu_ms: float = 0.0
    decode_ragged_cpu_tokens_ms: float = 0.0
    decode_ragged_state_update_ms: float = 0.0
    prompt_lookup_batches: int = 0
    prompt_lookup_requests: int = 0
    prompt_lookup_proposed_tokens: int = 0
    prompt_lookup_accepted_tokens: int = 0
    generated_prefix_store_requests: int = 0
    generated_prefix_reuse_requests: int = 0
    generated_prefix_reuse_tokens: int = 0
    prefill_shape_counts: dict[str, int] = field(default_factory=dict)
    decode_shape_counts: dict[str, int] = field(default_factory=dict)


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
    logits: Tensor | None


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
    # Chunked prefill: a request stays in the 'prefilling' phase, advancing
    # prompt_cursor by a bounded chunk each step (so a long prompt does not stall
    # decode in one shot), until prompt_cursor == len(prompt), then it samples its
    # first token and flips to 'decoding'. Default 'decoding' preserves the
    # one-shot-prefill path when chunking is off.
    phase: str = "decoding"
    prompt_cursor: int = 0
    prefix_source_row: int = -1  # reusable-prefix source row (first chunk folds its copy)


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
        prefill_chunk_size: int | None = None,
        decode_first: bool = True,
        enable_ragged_decode: bool = True,
        store_reusable_prefixes: bool = True,
        store_full_prompt_prefixes: bool = True,
        pin_shared_prefix: bool = False,
        graph_prefill: bool = False,
        profile_timings: bool = False,
        admit_min_ready_requests: int | None = None,
        admit_per_step_cap: int | None = None,
        enable_decode_many: bool | None = None,
        decode_many_allow_stop: bool | None = None,
        generated_prefix_cache: bool | None = None,
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
        # Chunked prefill: when set, an admitted request prefills its suffix in
        # bounded chunks of this many tokens across steps (interleaved with
        # decode) instead of in one shot, bounding how long a long prompt's
        # prefill stalls the active decode batch. None preserves one-shot prefill.
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_first = decode_first
        self.enable_ragged_decode = enable_ragged_decode
        self.store_reusable_prefixes = store_reusable_prefixes
        self.store_full_prompt_prefixes = store_full_prompt_prefixes
        # When pinning is on, the engine caches ONLY shared common prefixes and
        # pins them against eviction, skipping per-request full-prompt stores
        # that would otherwise starve the prefix-row pool and shadow the shared
        # prefix in the radix tree (preventing cross-batch reuse).
        self.pin_shared_prefix = pin_shared_prefix
        # When graph_prefill is on, suffix prefills route through the model's
        # row_indices ragged-prefill LOGITS graph (try_prefill_ragged_logits_graph):
        # the suffix KV is scatter-written into the (scattered) active rows and
        # one logit row per request is gathered, replaying the graph across
        # changing row sets. Batch and suffix are padded to power-of-two buckets
        # so graph shapes repeat. Per-row start positions handle mixed prefixes.
        self.graph_prefill = graph_prefill
        self.profile_timings = profile_timings
        self.admit_min_ready_requests = admit_min_ready_requests
        self.admit_per_step_cap = admit_per_step_cap
        self.generated_prefix_cache = generated_prefix_cache
        self.enable_decode_many = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY", False)
            if enable_decode_many is None
            else bool(enable_decode_many)
        )
        self.decode_many_allow_stop = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP", False)
            if decode_many_allow_stop is None
            else bool(decode_many_allow_stop)
        )
        self.unified_forward = bool(
            env_flag("TORCHINFERNO_CONTINUOUS_UNIFIED_FORWARD", False)
            and hasattr(model, "forward_step_flashinfer")
        )
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes: dict[Hashable, _ReusablePrefix] = {}
        self._pinned_prefix_routes: set[Hashable] = set()
        self.stats = ServingStats()
        self._cache: object | None = None
        self._cache_views: dict[tuple[int, ...], object] = {}
        self._reported_static_graph_miss = False
        self._free_active_rows: list[int] = []
        self._free_prefix_rows: list[int] = []
        self._row_seq_lens: list[int] = []
        self._device_index_tensors: dict[tuple[int, ...], Tensor] = {}
        self._prefix_order: list[Hashable] = []
        self._online_waiting: ServingQueue | None = None
        self._online_active: list[_ActiveRequest] = []
        self._online_prefilling: list[_ActiveRequest] = []
        self._online_step = 0
        self._online_next_index = 0
        self._pending_decode_ragged_model_events: list[tuple[object, object]] = []

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

    def start_online(
        self,
        *,
        max_seq_len: int,
        external_cache: object | None = None,
    ) -> None:
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

    @torch.inference_mode()
    def step_decode_only(self) -> list[ServingTokenEvent]:
        if not self._online_active:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        active = self._online_active
        self.stats.scheduler_steps += 1
        _decoded_results, active = self._decode_active(active, step, events=events)
        self._online_active = active
        self._online_step = step + 1
        return events

    @torch.inference_mode()
    def step_online_many(self, max_steps: int) -> tuple[list[ServingTokenEvent], int]:
        steps_left = max(1, int(max_steps))
        events: list[ServingTokenEvent] = []
        steps_run = 0
        while steps_left > 0 and self.has_online_work():
            if self._can_step_decode_many(steps_left):
                many_events, many_steps = self._step_decode_only_many(steps_left)
                if many_steps <= 0:
                    break
                events.extend(many_events)
                steps_run += many_steps
                steps_left -= many_steps
                continue
            step_events = self.step_online()
            events.extend(step_events)
            steps_run += 1
            steps_left -= 1
            break
        return events, steps_run

    def _can_step_decode_many(self, max_steps: int) -> bool:
        if max_steps <= 1:
            return False
        if not self.enable_decode_many:
            return False
        waiting = self._online_waiting
        if waiting or self._online_prefilling or not self._online_active:
            return False
        if self.unified_forward or not self.decode_first or self.temperature > 0.0:
            return False
        if (
            any(state.request.stop_token_ids for state in self._online_active)
            and not self.decode_many_allow_stop
        ):
            return False
        if self._generated_prefix_cache_enabled() or self._prompt_lookup_decode_enabled():
            return False
        return self._can_decode_ragged(self._online_active)

    def _step_decode_only_many(self, max_steps: int) -> tuple[list[ServingTokenEvent], int]:
        active = list(self._online_active)
        if not active:
            return [], 0

        self._sync_gpu_last_tokens_from_states(active)
        self._sync_gpu_seq_lens_from_states(active)
        records: list[tuple[list[_ActiveRequest], int, list[int], list[bool]]] = []
        token_parts: list[Tensor] = []
        steps_run = 0

        while steps_run < max_steps and active and self._can_decode_ragged(active):
            step = self._online_step + steps_run
            states = list(active)
            self.stats.scheduler_steps += 1
            next_token_tensor = self._decode_ragged_batch_token_tensor(states)
            token_parts.append(next_token_tensor[: len(states)].detach().clone())

            state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
            generated_after: list[int] = []
            finished_by_limit: list[bool] = []
            next_active: list[_ActiveRequest] = []
            for state in states:
                state.generated += 1
                next_seq_len = state.seq_len + 1
                self._remember_row_seq_len(state.row, next_seq_len)
                state.seq_len = next_seq_len
                generated_after.append(state.generated)
                done = state.generated >= state.request.max_new_tokens
                finished_by_limit.append(done)
                if not done:
                    next_active.append(state)
            if self.profile_timings:
                self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
            records.append((states, step, generated_after, finished_by_limit))
            active = next_active
            steps_run += 1

        if steps_run <= 0 or not token_parts:
            return [], 0

        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        flat_tokens = torch.cat(token_parts).detach().cpu().tolist()
        if self.profile_timings:
            self.stats.decode_ragged_cpu_tokens_ms += (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self._flush_decode_ragged_model_gpu_timers()

        events: list[ServingTokenEvent] = []
        terminated: set[int] = set()
        offset = 0
        for states, step, generated_after, finished_by_limit in records:
            row_tokens = flat_tokens[offset : offset + len(states)]
            offset += len(states)
            for state, token_value, generated, limit_finished in zip(
                states,
                row_tokens,
                generated_after,
                finished_by_limit,
            ):
                state_id = id(state)
                if state_id in terminated:
                    continue
                token = int(token_value)
                state.tokens.append(token)
                state.last_token = token
                finished = bool(limit_finished or state.request.is_stop_token(token))
                events.append(
                    ServingTokenEvent(
                        request_id=state.request.request_id,
                        token=token,
                        step=step,
                        generated=generated,
                        finished=finished,
                    )
                )
                if finished:
                    terminated.add(state_id)
                    self._finish_and_release(state, step)

        self._online_active = [state for state in active if id(state) not in terminated]
        self._online_step += steps_run
        return events, steps_run

    def has_online_work(self) -> bool:
        waiting = self._online_waiting
        return bool(waiting) or bool(self._online_active) or bool(self._online_prefilling)

    @torch.inference_mode()
    def step_online(self) -> list[ServingTokenEvent]:
        if self.unified_forward:
            return self._step_online_unified()
        waiting = self._require_online_waiting()
        if not waiting and not self._online_active and not self._online_prefilling:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        active = self._online_active
        self.stats.scheduler_steps += 1
        if self.decode_first and active:
            _da_start = time.perf_counter() if self.profile_timings else 0.0
            _decoded_results, active = self._decode_active(active, step, events=events)
            if self.profile_timings:
                self.stats._decode_active_ms = getattr(self.stats, '_decode_active_ms', 0.0) + (time.perf_counter() - _da_start) * 1000.0

        if self.prefill_chunk_size and self._online_prefilling:
            active.extend(self._advance_prefilling(step, events=events))

        _admit_start = time.perf_counter() if self.profile_timings else 0.0
        occupied = len(active) + len(self._online_prefilling)
        admitted = self._admit_ready_requests(waiting, step, occupied)
        if self.profile_timings:
            self.stats._admit_ms = getattr(self.stats, '_admit_ms', 0.0) + (time.perf_counter() - _admit_start) * 1000.0
        if admitted:
            if self.prefill_chunk_size:
                active.extend(self._admit_to_prefilling(admitted, step, events=events))
            else:
                _admitted_results, admitted_active = self._prefill_many(admitted, step, events=events)
                active.extend(admitted_active)

        if not self.decode_first and active:
            _decoded_results, active = self._decode_active(active, step + 1, events=events)
        if self.decode_first and active:
            active = self._release_online_prefill_finished(active, step)
        self._online_active = active
        self._online_step = step + 1

        next_arrival_step = waiting.next_arrival_step()
        idle = not active and not self._online_prefilling
        if next_arrival_step is not None and idle and next_arrival_step > self._online_step:
            self._online_step = next_arrival_step
        return events

    @torch.inference_mode()
    def _step_online_unified(self) -> list[ServingTokenEvent]:
        waiting = self._require_online_waiting()
        if not waiting and not self._online_active and not self._online_prefilling:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        self.stats.scheduler_steps += 1
        cache = self._require_cache()

        decode_states: list[_ActiveRequest] = []
        for state in self._online_active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                self._finish_and_release(state, step)
                self._record_token_event(events, state, state.last_token, step, finished=True)
            else:
                decode_states.append(state)

        occupied = len(decode_states) + len(self._online_prefilling)
        admitted = self._admit_ready_requests(waiting, step, occupied)

        prefill_states: list[_ActiveRequest] = []
        exact_reuse_group: list[tuple[int, ServingRequest, int, _ReusablePrefix]] = []
        for original_index, request in admitted:
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            prefix_hit = match.depth if (reusable is not None and match.depth > 0) else 0
            if reusable is not None and prefix_hit >= len(request.prompt) and reusable.logits is not None:
                exact_reuse_group.append((original_index, request, prefix_hit, reusable))
                continue
            if reusable is not None and prefix_hit >= len(request.prompt):
                reusable = None
                prefix_hit = 0
            row = self._acquire_active_row()
            if reusable is not None and prefix_hit > 0:
                self._copy_prefix(reusable.row, row, prefix_hit)
                self.stats.prefix_reuse_requests += 1
                self.stats.prefix_reuse_tokens += prefix_hit
            suffix = request.prompt[prefix_hit:]
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=list(request.prompt),
                generated=0,
                row=row,
                last_token=request.prompt[-1] if request.prompt else 0,
                seq_len=prefix_hit,
                prefix_hit_tokens=prefix_hit,
                started_step=step,
            )
            state._prefill_suffix = suffix
            prefill_states.append(state)
        self.stats.prefill_admitted_requests += len(prefill_states) + len(exact_reuse_group)

        exact_reuse_active: list[_ActiveRequest] = []
        exact_reuse_processed = False

        def finish_exact_reuse() -> list[_ActiveRequest]:
            nonlocal exact_reuse_active, exact_reuse_processed
            if exact_reuse_group and not exact_reuse_processed:
                exact_reuse_active = self._prefill_exact_prefix_batch(
                    exact_reuse_group,
                    step,
                    events=events,
                )
                exact_reuse_processed = True
            return exact_reuse_active

        if not decode_states and not prefill_states and not self._online_prefilling:
            self._online_active = finish_exact_reuse()
            self._online_step = step + 1
            return events

        if prefill_states or self._online_prefilling:
            all_prefilling = list(self._online_prefilling) + prefill_states
            batch_rows = []
            batch_q_lens = []
            batch_input_ids: list[list[int]] = []
            batch_write_pos: list[list[int]] = []
            batch_logit_pos = []
            batch_seq_lens = []
            max_q = 1
            batch_is_decode: list[bool] = []
            batch_states: list[_ActiveRequest] = []

            for state in decode_states:
                batch_rows.append(state.row)
                batch_q_lens.append(1)
                batch_input_ids.append([state.last_token])
                batch_write_pos.append([state.seq_len])
                batch_logit_pos.append(0)
                batch_seq_lens.append(state.seq_len)
                batch_is_decode.append(True)
                batch_states.append(state)

            chunk = self.prefill_chunk_size
            still_prefilling: list[_ActiveRequest] = []
            for state in all_prefilling:
                suffix = getattr(state, '_prefill_suffix', None)
                if suffix is None:
                    suffix = state.request.prompt[state.seq_len:]
                if chunk and len(suffix) > chunk:
                    cur_suffix = suffix[:chunk]
                    state._prefill_suffix = suffix[chunk:]
                    still_prefilling.append(state)
                else:
                    cur_suffix = suffix
                    state._prefill_suffix = None
                cursor = state.seq_len
                q_len = len(cur_suffix)
                if q_len == 0:
                    continue
                batch_rows.append(state.row)
                batch_q_lens.append(q_len)
                batch_input_ids.append(list(cur_suffix))
                batch_write_pos.append(list(range(cursor, cursor + q_len)))
                batch_logit_pos.append(q_len - 1)
                batch_seq_lens.append(cursor)
                batch_is_decode.append(False)
                batch_states.append(state)
                max_q = max(max_q, q_len)

            if not batch_rows:
                self._online_active = decode_states + finish_exact_reuse()
                self._online_prefilling = still_prefilling
                self._online_step = step + 1
                return events

            n = len(batch_rows)
            for i in range(n):
                last_wp = batch_write_pos[i][-1] if batch_write_pos[i] else 0
                while len(batch_input_ids[i]) < max_q:
                    batch_input_ids[i].append(0)
                while len(batch_write_pos[i]) < max_q:
                    batch_write_pos[i].append(last_wp)

            input_ids = torch.tensor(batch_input_ids, device=self.device, dtype=torch.long)
            q_lens_t = torch.tensor(batch_q_lens, device=self.device, dtype=torch.long)
            write_positions = torch.tensor(batch_write_pos, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(batch_logit_pos, device=self.device, dtype=torch.long)
            seq_lens_t = torch.tensor(batch_seq_lens, device=self.device, dtype=torch.long)
            row_indices = torch.tensor(batch_rows, device=self.device, dtype=torch.long)

            logits = self.model.forward_step_flashinfer(
                input_ids, cache,
                seq_lens=seq_lens_t, q_lens=q_lens_t,
                write_positions=write_positions,
                logit_positions=logit_positions,
                row_indices=row_indices,
            )
            self._record_model_call("unified", n, tokens=int(q_lens_t.sum().item()))
            next_tokens_cpu = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

            next_active: list[_ActiveRequest] = []
            for i, state in enumerate(batch_states):
                tok = int(next_tokens_cpu[i])
                if batch_is_decode[i]:
                    state.tokens.append(tok)
                    state.generated += 1
                    state.last_token = tok
                    state.seq_len += 1
                    self._remember_row_seq_len(state.row, state.seq_len)
                    finished = self._should_finish_after_decode(state)
                    self._record_token_event(events, state, tok, step, finished=finished)
                    if finished:
                        self._finish_and_release(state, step)
                    else:
                        next_active.append(state)
                else:
                    q_len = batch_q_lens[i]
                    new_seq_len = batch_seq_lens[i] + q_len
                    self._set_cache_row_seq_len(state.row, new_seq_len)
                    self._remember_row_seq_len(state.row, new_seq_len)
                    state.seq_len = new_seq_len
                    if state._prefill_suffix is None or len(state._prefill_suffix) == 0:
                        state.tokens.append(tok)
                        state.generated = 1
                        state.last_token = tok
                        state._prefill_suffix = None
                        self._store_reusable_prefix(
                            state.request.request_id, state.request.prompt,
                            state.row, logits[i:i+1],
                            allow_pinned=self._allow_pinned_full_prompt_store(state.request),
                        )
                        finished = self._should_finish_after_decode(state)
                        self._record_token_event(events, state, tok, step, finished=finished)
                        if finished:
                            self._finish_and_release(state, step)
                        else:
                            next_active.append(state)
                    else:
                        still_prefilling.append(state)

            next_active.extend(finish_exact_reuse())
            self._online_active = next_active
            self._online_prefilling = still_prefilling
        else:
            decoded = (
                self._decode_ragged_batch(decode_states, step, events=events)
                if self._can_decode_ragged(decode_states)
                else (
                    self._decode_batch(decode_states, step, events=events)
                    if len(decode_states) > 1
                    else ([self._decode_one(decode_states[0], step, events=events)] if decode_states else [])
                )
            )
            next_active = []
            for item, state in zip(decoded, decode_states):
                if isinstance(item, ServingResult):
                    pass
                else:
                    next_active.append(item)
            next_active.extend(finish_exact_reuse())
            self._online_active = next_active

        self._online_step = step + 1
        next_arrival_step = waiting.next_arrival_step()
        idle = not self._online_active and not self._online_prefilling
        if next_arrival_step is not None and idle and next_arrival_step > self._online_step:
            self._online_step = next_arrival_step
        return events


    def _admit_to_prefilling(
        self,
        admitted: list[tuple[int, ServingRequest]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        # Create a 'prefilling' state per admitted request without running any
        # model forward; the chunk advance does the prefill incrementally. The
        # shared-prefix KV copy is deferred to (and folded into) the first chunk.
        exact_reuse_group: list[tuple[int, ServingRequest, int, _ReusablePrefix]] = []
        for original_index, request in admitted:
            if request.max_new_tokens == 0:
                continue
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            prefix_hit = match.depth if (reusable is not None and match.depth > 0) else 0
            source_row = reusable.row if (reusable is not None and prefix_hit > 0) else -1
            if reusable is not None and prefix_hit >= len(request.prompt) and reusable.logits is not None:
                exact_reuse_group.append((original_index, request, prefix_hit, reusable))
                continue
            if reusable is not None and prefix_hit >= len(request.prompt):
                reusable = None
                prefix_hit = 0
            row = self._acquire_active_row()
            if prefix_hit > 0:
                self.stats.prefix_reuse_requests += 1
                self.stats.prefix_reuse_tokens += prefix_hit
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=list(request.prompt),
                generated=0,
                row=row,
                last_token=request.prompt[-1],
                seq_len=prefix_hit,
                prefix_hit_tokens=prefix_hit,
                started_step=step,
                phase="prefilling",
                prompt_cursor=prefix_hit,
                prefix_source_row=source_row,
            )
            self._online_prefilling.append(state)
        return self._prefill_exact_prefix_batch(
            exact_reuse_group,
            step,
            events=events,
        )

    def _advance_prefilling(self, step: int, events: list[ServingTokenEvent] | None) -> list[_ActiveRequest]:
        chunk = int(self.prefill_chunk_size or 0)
        if chunk <= 0 or not self._online_prefilling:
            return []
        # Group by (prefix length, cursor, prefix source) so every row in a group
        # shares one absolute start -> the flash context_len path applies, and the
        # first chunk of a group folds the shared-prefix copy from one source row.
        groups: dict[tuple[int, int, int], list[_ActiveRequest]] = defaultdict(list)
        for state in self._online_prefilling:
            groups[(state.prefix_hit_tokens, state.prompt_cursor, state.prefix_source_row)].append(state)
        newly_decoding: list[_ActiveRequest] = []
        still_prefilling: list[_ActiveRequest] = []
        for (prefix_hit, cursor, source_row), states in groups.items():
            finished, pending = self._prefill_chunk_group(states, cursor, source_row, chunk, step, events)
            newly_decoding.extend(finished)
            still_prefilling.extend(pending)
        self._online_prefilling = still_prefilling
        return newly_decoding

    def _prefill_chunk_group(
        self,
        states: list[_ActiveRequest],
        cursor: int,
        source_row: int,
        chunk: int,
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> tuple[list[_ActiveRequest], list[_ActiveRequest]]:
        chunk_lens = [min(chunk, len(s.request.prompt) - cursor) for s in states]
        chunk_bucket = self._suffix_bucket(max(chunk_lens))
        cache_max_seq = self._cache_max_seq_len()
        if cache_max_seq is not None:
            chunk_bucket = min(chunk_bucket, max(1, cache_max_seq - cursor))
        context_len = cursor + chunk_bucket
        chunks = [s.request.prompt[cursor : cursor + n] for s, n in zip(states, chunk_lens)]
        padded = [[*c, *([0] * (chunk_bucket - len(c)))] for c in chunks]
        rows = [s.row for s in states]
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        row_indices = torch.tensor(rows, device=self.device, dtype=torch.long)
        required = max(rows + ([source_row] if source_row >= 0 else [0])) + 1
        seq_lens_list = [0] * required
        for row in rows:
            seq_lens_list[row] = cursor
        seq_lens = torch.tensor(seq_lens_list, device=self.device, dtype=torch.long)
        logit_positions = torch.tensor([n - 1 for n in chunk_lens], device=self.device, dtype=torch.long)
        # Fold the prefix copy only on the FIRST chunk of a reused prefix.
        src_prefix_row = None
        if source_row >= 0 and cursor == states[0].prefix_hit_tokens and cursor > 0:
            src_prefix_row = torch.tensor([source_row], device=self.device, dtype=torch.long)
        logits = self._try_ragged_prefill_logits(
            input_ids, seq_lens, row_indices, logit_positions, context_len, src_prefix_row
        )
        if logits is None:
            logits = self._ragged_prefill_logits_eager(
                input_ids, seq_lens, row_indices, logit_positions, context_len, src_prefix_row
            )
        self._record_model_call("prefill", len(states), tokens=sum(chunk_lens))
        self.stats.prefill_prefix_reuse_batches += 1
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
        finished: list[_ActiveRequest] = []
        pending: list[_ActiveRequest] = []
        for index, state in enumerate(states):
            new_cursor = cursor + chunk_lens[index]
            state.prompt_cursor = new_cursor
            self._set_cache_row_seq_len(state.row, new_cursor)
            state.seq_len = new_cursor
            if new_cursor >= len(state.request.prompt):
                next_token = int(next_tokens[index])
                state.tokens.append(next_token)
                state.generated = 1
                state.last_token = next_token
                state.phase = "decoding"
                request_finished = self._should_finish_before_decode(state)
                self._record_token_event(events, state, next_token, step, finished=request_finished)
                if request_finished:
                    self._finish_and_release(state, step)
                else:
                    finished.append(state)
            else:
                pending.append(state)
        return finished, pending

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
        self._pinned_prefix_routes = set()
        self._cache_views = {}
        self._reported_static_graph_miss = False
        total_rows = self.max_active_requests + self.prefix_cache_capacity
        if external_cache is not None:
            self._cache = external_cache
            _enable_runtime_cache_capture_sync(self._cache)
        else:
            self._cache = self._allocate_cache(max(1, total_rows), max_seq_len)
        if not hasattr(self._cache, "for_rows"):
            raise ValueError("model cache must support row views for persistent serving")
        self._row_seq_lens = [0 for _ in range(total_rows)]
        self._gpu_seq_lens = None
        self._device_index_tensors = {}
        self._free_active_rows = list(reversed(range(self.max_active_requests)))
        self._free_prefix_rows = list(reversed(range(self.max_active_requests, total_rows)))
        self.stats.persistent_cache_rows = total_rows
        self.stats.queued_requests = queued_requests
        self._online_waiting = None
        self._online_active = []
        self._online_prefilling = []
        self._online_step = 0
        self._online_next_index = 0
        self._pending_decode_ragged_model_events = []

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
        # Always cap NEW admissions per step at per_step_cap. This decouples the
        # prefill batch size (<= per_step_cap, where the prefill CUDA graphs live)
        # from the decode batch size (active rows can grow to max_active across
        # several steps). A larger decode batch lifts memory-bound decode
        # throughput without forcing a giant single-step prefill.
        per_step_cap = self.admit_per_step_cap
        if per_step_cap is None:
            per_step_cap = env_int("TORCHINFERNO_CONTINUOUS_ADMIT_PER_STEP_CAP", 48, minimum=0)
        if per_step_cap > 0:
            capacity = min(capacity, per_step_cap)
        default_min_ready_requests = self.admit_min_ready_requests
        if default_min_ready_requests is None:
            default_min_ready_requests = 1
        min_ready_requests = env_int(
            "TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS",
            int(default_min_ready_requests),
            minimum=1,
        )
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
        # graph_prefill pads suffixes to a common length and buckets the batch,
        # so reuse requests must be grouped by prefix alone (suffix key -1)
        # rather than split per suffix length -- otherwise each suffix length
        # reaches the graph path as its own tiny batch and never amortizes.
        pad_prefix_suffixes = env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False) or self.graph_prefill

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
            if (
                reusable is not None
                and reusable_prefix_tokens >= len(request.prompt)
                and reusable.logits is None
            ):
                reusable = None
                reusable_prefix_tokens = 0
            if reusable is not None and reusable_prefix_tokens > 0:
                suffix_len = len(request.prompt) - reusable_prefix_tokens
                batch_suffix_len = -1 if pad_prefix_suffixes else suffix_len
                if (
                    pad_prefix_suffixes
                    and env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL", False)
                ):
                    reusable_prefix_tokens = -1
                prefix_batchable[(reusable_prefix_tokens, batch_suffix_len)].append(
                    (original_index, request, match.depth, reusable)
                )
            else:
                batchable[len(request.prompt)].append((original_index, request, 0))

        plain_group = [item for group in batchable.values() for item in group]

        # FlashInfer-native prefix reuse for cached-prefix hits; on failure the
        # group falls back into the full-prompt path below (returns None).
        # OFF by default. The reuse LOGIC is verified correct in isolation --
        # scripts/debug_reuse_engine.py reproduces the full engine path (prefill
        # graphs + online stepping + reuse) single-GPU and PASSES. Enabling it on
        # the 8-rank server hangs/CUDA-asserts. Narrowed via env-gated diagnostics
        # (TORCHINFERNO_REUSE_DEBUG): it is reuse-vs-TP, NOT the graphs. Ruled out:
        # reuse-OFF + graphs-OFF + a single large request works fine on 8 ranks
        # (so the _prefill_one -> _prefill_logits FI-eager fallback is sound). The
        # hang appears ONLY with the reuse config (FI_REUSE=1 + pin_shared_prefix
        # =False + prefix_rows>2) and resists single-GPU repro -- the standalone
        # engine harness (scripts/debug_reuse_engine.py) PASSES. This is a subtle
        # multi-rank collective-divergence bug that needs a dedicated instrumented
        # 8-GPU session, not incremental loop runs. Enable via the env flag +
        # pin_shared_prefix=False once that divergence is found.
        #
        # IMPORTANT value caveat (8-GPU REUSE_DEBUG trace): with persistent=False
        # every request burst starts a fresh online session and start_online ->
        # _reset_capacity WIPES reusable_prefixes/prefix_cache. So cross-burst
        # reuse never triggers (each _prefill_many logs cached_prefixes=0 at
        # step=0) -- e.g. multi_turn's per-turn requests are separate bursts and
        # get no reuse. Only WITHIN-session reuse fires (and that is the case that
        # hangs on TP). Making reuse actually pay off therefore ALSO needs a
        # persistent engine whose prefix cache survives across bursts -- a larger
        # change than the reuse path itself.
        _reuse_dbg = env_flag("TORCHINFERNO_REUSE_DEBUG", False)
        _reuse_handled = 0
        if env_flag("TORCHINFERNO_CONTINUOUS_FI_REUSE", False) and prefix_batchable:
            unhandled: dict[tuple[int, int], list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = {}
            for key, group in prefix_batchable.items():
                reuse_active = self._prefill_flashinfer_reuse(group, step, events=events)
                if reuse_active is not None:
                    active.extend(reuse_active)
                    _reuse_handled += len(reuse_active)
                else:
                    unhandled[key] = group
            prefix_batchable = unhandled
        if _reuse_dbg:
            import sys as _rdbg
            _rk = getattr(self.model, "rank", 0)
            _n_pref = sum(len(g) for g in prefix_batchable.values())
            print(
                f"[REUSE_DBG] rank={_rk} step={step} reuse_handled={_reuse_handled} "
                f"unhandled_prefix={_n_pref} plain={len(plain_group)} "
                f"cached_prefixes={len(self.reusable_prefixes)}",
                file=_rdbg.stderr, flush=True,
            )

        # Common-prefix fast path: prefill a shared prefix ONCE then per-request
        # suffixes. DISABLED by default -- live A/B on the real 70B (TP8) showed it
        # is a regression in EVERY regime, including the identical-prompt
        # self_consistency case it was built for. Cause: it routes the burst through
        # the EAGER _prefill_logits path (launch-overhead bound, ~245ms/call on TP8
        # because per-layer allreduces are not graph-amortized), bypassing the
        # graph-backed _try_flashinfer_prefill below (try_prefill_flashinfer_graph,
        # the warmup-captured _fi_prefill_graphs). Worse, the online batcher admits
        # in waves (initial_batch_size=1), so each wave pays its own ~245ms eager
        # prefix prefill. Measured TTFT (identical / distinct), ON vs OFF: N=8
        # 603/737 -> 122/124; N=64 1549/2378 -> 999/949 (3-7x at low N, ~2x at high
        # N, ~2x throughput). The graph-FI path handles identical and distinct
        # equally fast, so this path only wins if eager prefill is ever fixed.
        if (
            env_flag("TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_PREFILL", False)
            and len(plain_group) > 1
            and self.prefix_cache_capacity > 0
        ):
            cp_tokens = _common_prefix_token_count([req.prompt for _i, req, _h in plain_group])
            if cp_tokens >= 16:
                shared_active = self._prefill_common_prefix_batch(
                    [(idx, req, 0) for idx, req, _h in plain_group], step, events=events
                )
                if shared_active is not None:
                    active.extend(shared_active)
                    plain_group = []

        all_fi_requests: list[tuple[int, ServingRequest, int, _ReusablePrefix | None]] = []
        # FlashInfer prefill defaults OFF. MEASURED 2026-06-10 on the real 70B TP8
        # local full bench at 64-conc: FI-prefill ON regressed few_shot ttft
        # 216->759ms (3.5x), tpot 73->247ms, tput 4.1->1.1 vs identical config with
        # FI-prefill OFF. The bench's prefills are small, so FlashInfer's varlen
        # advantage does not apply and its per-wave plan/launch overhead dominates;
        # TI's graphed ragged prefill is already GEMM-bound and faster here. The
        # gate was default-ON, a footgun: it auto-enables whenever flashinfer is
        # importable, so installing flashinfer (e.g. for paged decode) would
        # silently regress serving 3.5x. Set the flag to 0 to opt back in.
        if not env_flag("TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE", True):
            for group in prefix_batchable.values():
                for idx, req, hit, reusable in group:
                    all_fi_requests.append((idx, req, hit, reusable))
            for idx, req, hit in plain_group:
                all_fi_requests.append((idx, req, hit, None))

        if all_fi_requests:
            _fi_debug = env_flag("TORCHINFERNO_FI_PREFILL_PROFILE", False)
            _fi_t0 = time.perf_counter() if _fi_debug else 0.0
            # For a single request, eager FlashInfer is launch-overhead bound
            # (~245ms regardless of prompt length), worse than the SDPA single
            # path. So batch=1 uses the prefill CUDA graph ONLY (a captured
            # batch=1 graph replays in ~25ms even padded to the q bucket, since
            # batch=1 compute is tiny); a graph miss falls through to _prefill_one.
            graph_only = len(all_fi_requests) == 1
            fi_active = self._try_flashinfer_prefill(
                all_fi_requests, step, events=events, graph_only=graph_only
            )
            if fi_active is not None:
                if _fi_debug:
                    import sys as _fpm
                    print(
                        f"[FI_PREFILL] OK batch={len(all_fi_requests)} active={len(fi_active)} "
                        f"time={(time.perf_counter()-_fi_t0)*1000:.1f}ms",
                        file=_fpm.stderr, flush=True,
                    )
                active.extend(fi_active)
                if self.profile_timings:
                    self.stats.prefill_wall_ms += (time.perf_counter() - timing_start_s) * 1000.0
                return indexed_results, active
            elif _fi_debug:
                import sys as _fpm
                print(
                    f"[FI_PREFILL] FALLBACK batch={len(all_fi_requests)} "
                    f"prefix_batchable={len(prefix_batchable)} plain={len(plain_group)}",
                    file=_fpm.stderr, flush=True,
                )

        for group in prefix_batchable.values():
            active.extend(self._prefill_prefix_batch(group, step, events=events))

        if env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_GRAPH_PREFILL", False) and plain_group and self.graph_prefill and len(plain_group) > 1:
            ragged_active = self._prefill_ragged_graph_batch(plain_group, step, events=events)
            if ragged_active is not None:
                active.extend(ragged_active)
                plain_group = []
        if plain_group:
            shared_prefix_active = self._prefill_common_prefix_batch(plain_group, step, events=events)
            if shared_prefix_active is not None:
                active.extend(shared_prefix_active)
            elif len(plain_group) > 1 and self._can_padded_batch_prefill(plain_group):
                active.extend(self._prefill_padded_batch(plain_group, step, events=events))
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
            self._record_shape_count(self.stats.prefill_shape_counts, f"common_prefix:b1:t{prefix_tokens}")
            self.stats.prefill_common_prefix_batches += 1
            prefix_tuple = tuple(group[0][1].prompt[:prefix_tokens])
            common_route = ("common_prefix", prefix_tuple)
            store_common_prefix_logits = any(
                len(request.prompt) <= prefix_tokens
                for _index, request, _hit in group
            )
            self._store_reusable_prefix_tokens(
                common_route,
                "__common_prefix__",
                prefix_tuple,
                prefix_row,
                prefix_logits if store_common_prefix_logits else None,
                store_logits=store_common_prefix_logits,
            )
            reusable = self.reusable_prefixes.get(common_route)
            if self.pin_shared_prefix and common_route in self.reusable_prefixes:
                self._pinned_prefix_routes.add(common_route)

            # Folding the common-prefix KV copy into the ragged suffix graph is
            # profitable for short greedy streams, where it removes one-off
            # prefill graph misses. Greedy-mid workloads such as few_shot stay on
            # the lower cutoff because the extra ragged suffix work regresses TPOT.
            max_ragged_prefix_tokens = self._common_prefix_ragged_suffix_max_prefix_tokens(group)
            if (
                self.graph_prefill
                and reusable is not None
                and max_ragged_prefix_tokens > 0
                and prefix_tokens <= max_ragged_prefix_tokens
                and all(len(request.prompt) > prefix_tokens for _index, request, _hit in group)
            ):
                graph_active = self._prefill_prefix_graph_batch(
                    [
                        (original_index, request, prefix_tokens, reusable)
                        for original_index, request, _prefix_hit_tokens in group
                    ],
                    step,
                    events=events,
                )
                if graph_active is not None:
                    return graph_active

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
                self._copy_prefix_to_rows(prefix_row, rows, prefix_tokens)
                suffixes = [
                    request.prompt[prefix_tokens:]
                    for _original_index, request, _prefix_hit_tokens in suffix_group
                ]

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
                    self._store_reusable_prefix(
                        request.request_id,
                        request.prompt,
                        row,
                        logits[row_index : row_index + 1],
                        allow_pinned=self._allow_pinned_full_prompt_store(request),
                    )
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

    def _common_prefix_ragged_suffix_max_prefix_tokens(
        self,
        group: list[tuple[int, ServingRequest, int]],
    ) -> int:
        configured = os.environ.get(
            "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_MAX_PREFIX_TOKENS"
        )
        if configured is not None:
            return env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_MAX_PREFIX_TOKENS",
                64,
                minimum=0,
            )
        if self.temperature <= 0.0 and group:
            max_new_tokens = max(request.max_new_tokens for _index, request, _hit in group)
            greedy_short_max = env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_SHORT_MAX_TOKENS",
                128,
                minimum=1,
            )
            if 0 < max_new_tokens <= greedy_short_max:
                return env_int(
                    "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_SHORT_PREFIX_TOKENS",
                    128,
                    minimum=0,
                )
        return 64

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
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    logits[row_index : row_index + 1],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
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

    def _prefill_batch_bucket(self, count: int) -> int:
        # Pad the prefill batch to a power of two so the model's prefill graph
        # key -- (batch, suffix_bucket, prefix_len) -- repeats across batches and
        # replays instead of recapturing on every differently-sized batch. Cap
        # at the active-row capacity so dummy padding rows never exceed the cache.
        if count <= 1:
            return 1
        bucket = 1 << (count - 1).bit_length()
        return min(bucket, self.max_active_requests)

    def _suffix_bucket(self, length: int) -> int:
        # Pad the suffix length to a power of two so the ragged-prefill graph key
        # (batch, suffix_bucket, ...) repeats across batches and replays.
        if length <= 1:
            return 1
        return 1 << (length - 1).bit_length()

    def _prefill_prefix_graph_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        # Route suffix prefill through the model's row_indices ragged-prefill
        # LOGITS graph. The shared prefix KV is copied into each (scattered) row,
        # then the suffix is scatter-written and one logit row per request is
        # gathered. Per-row start positions handle MIXED prefix lengths, and the
        # graph replays across changing scattered row sets (unlike the old
        # contiguous for_rows selected-logits path). Batch and suffix are padded
        # to power-of-two buckets so graph shapes repeat.
        prefix_hits = [prefix_hit_tokens for _i, _req, prefix_hit_tokens, _r in group]
        # _prefill_many groups reuse requests by prefix length, so the prefix is
        # uniform here; that lets the suffix attention use a flash causal_lower_right
        # over a static context_len (prefix + suffix_bucket) instead of a boolean
        # mask (which OOMs at large suffix x context). A non-uniform group (rare)
        # falls back to the eager per-suffix-length path.
        mixed_prefixes = len(set(prefix_hits)) != 1
        if mixed_prefixes and not env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL", False):
            return None
        suffixes = [request.prompt[prefix_hits[i]:] for i, (_idx, request, _h, _r) in enumerate(group)]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0:
            return None
        non_common_graph_prefill = env_flag("TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL", False)
        if not non_common_graph_prefill:
            for _index, _request, _prefix_hit_tokens, reusable in group:
                route_id = reusable.route_id
                if not (isinstance(route_id, tuple) and route_id[:1] == ("common_prefix",)):
                    return None
        elif not env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_GRAPH_PREFILL", False):
            for _index, _request, _prefix_hit_tokens, reusable in group:
                route_id = reusable.route_id
                if isinstance(route_id, tuple) and route_id[:1] == ("finished_prefix",):
                    return None
        cache_max_seq = self._cache_max_seq_len()
        suffix_bucket = self._suffix_bucket(max(suffix_lengths))
        if cache_max_seq is not None:
            suffix_bucket = min(suffix_bucket, max(1, cache_max_seq - max(prefix_hits)))
        context_len = None if mixed_prefixes else prefix_hits[0] + suffix_bucket
        count = len(group)
        batch_bucket = self._prefill_batch_bucket(count)
        rows = [self._acquire_active_row() for _ in group]
        pad_rows: list[int] = []
        try:
            # The prefix KV broadcast is FOLDED INTO the prefill graph (copy from
            # each reusable source row to its active row in one captured pass), so
            # the engine no longer issues ~80 per-layer index_copy launches per
            # batch here -- it only records reuse accounting and source rows.
            copy_start_s = time.perf_counter() if self.profile_timings else 0.0
            source_prefix_rows = [
                reusable.row
                for _index, _request, _prefix_hit_tokens, reusable in group
            ]
            for _index, _request, prefix_hit_tokens, _reusable in group:
                self.stats.prefix_reuse_requests += 1
                self.stats.prefix_reuse_tokens += prefix_hit_tokens
            padded_suffixes = [
                [*suffix, *([0] * (suffix_bucket - len(suffix)))]
                for suffix in suffixes
            ]
            start_lens = list(prefix_hits)
            if batch_bucket > count:
                dummy_suffix = padded_suffixes[0]
                for _ in range(batch_bucket - count):
                    pad_row = self._acquire_active_row_or_none()
                    if pad_row is None:
                        break
                    pad_rows.append(pad_row)
                    padded_suffixes.append(list(dummy_suffix))
                    start_lens.append(prefix_hits[0])
                    source_prefix_rows.append(source_prefix_rows[0])
            if not mixed_prefixes and len(set(source_prefix_rows)) == 1:
                source_prefix_rows = [source_prefix_rows[0]]
            self._record_shape_count(
                self.stats.prefill_shape_counts,
                "prefix_graph:"
                f"b{batch_bucket}:"
                f"s{suffix_bucket}:"
                f"p{min(prefix_hits)}-{max(prefix_hits)}:"
                f"src{len(source_prefix_rows)}:"
                f"mixed{int(mixed_prefixes)}",
            )
            if self.profile_timings:
                self.stats.prefill_copy_ms += (time.perf_counter() - copy_start_s) * 1000.0
            setup_start_s = time.perf_counter() if self.profile_timings else 0.0
            all_rows = rows + pad_rows
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            row_indices = torch.tensor(all_rows, device=self.device, dtype=torch.long)
            src_prefix_row = torch.tensor(source_prefix_rows, device=self.device, dtype=torch.long)
            required = max(all_rows + source_prefix_rows) + 1
            seq_lens_list = [0] * required
            for physical_row, start_len in zip(all_rows, start_lens):
                seq_lens_list[physical_row] = start_len
            seq_lens = torch.tensor(seq_lens_list, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(
                [length - 1 for length in suffix_lengths] + [0] * len(pad_rows),
                device=self.device,
                dtype=torch.long,
            )
            if self.profile_timings:
                self.stats.prefill_setup_ms += (time.perf_counter() - setup_start_s) * 1000.0
            forward_start_s = time.perf_counter() if self.profile_timings else 0.0
            logits = None
            if not mixed_prefixes or env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL_GRAPH", False):
                logits = self._try_ragged_prefill_logits(
                    input_ids,
                    seq_lens,
                    row_indices,
                    logit_positions,
                    context_len,
                    src_prefix_row,
                    capture_on_miss=env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_ON_MISS", True),
                )
            if logits is None:
                logits = self._ragged_prefill_logits_eager(
                    input_ids, seq_lens, row_indices, logit_positions, context_len, src_prefix_row
                )
            if self.profile_timings and logits is not None:
                # force the prefill graph/forward to complete for honest timing
                torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
                self.stats.prefill_forward_ms += (time.perf_counter() - forward_start_s) * 1000.0
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            pad_rows = []
            if logits is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", count, tokens=count * max(suffix_lengths))
            self.stats.prefill_prefix_reuse_batches += 1
            self.stats.prefill_padded_suffix_batches += 1
            sample_start_s = time.perf_counter() if self.profile_timings else 0.0
            next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
            if self.profile_timings:
                self.stats.prefill_sample_ms += (time.perf_counter() - sample_start_s) * 1000.0
            state_start_s = time.perf_counter() if self.profile_timings else 0.0
            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
                row = rows[row_index]
                self._set_cache_row_seq_len(row, len(request.prompt))
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    logits[row_index : row_index + 1],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
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
            if self.profile_timings:
                self.stats.prefill_state_ms += (time.perf_counter() - state_start_s) * 1000.0
            return active
        except Exception:
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            for row in rows:
                self._release_active_row(row)
            raise

    def _cache_max_seq_len(self) -> int | None:
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if not layers:
            return None
        return getattr(layers[0], "max_seq_len", None)

    def _try_ragged_prefill_logits(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        graph = getattr(self.model, "try_prefill_ragged_logits_graph", None)
        if graph is None:
            return None
        logits = graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            capture_on_miss=capture_on_miss,
        )
        if logits is None:
            self.stats.prefill_graph_misses += 1
            return None
        self.stats.prefill_graph_hits += 1
        return logits

    def _ragged_prefill_logits_eager(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
    ) -> Tensor | None:
        eager = getattr(self.model, "prefill_ragged_logits", None)
        if eager is None:
            return None
        return eager(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
        )

    def _prefill_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        if self.graph_prefill:
            graph_active = self._prefill_prefix_graph_batch(group, step, events=events)
            if graph_active is not None:
                return graph_active
        suffix_lengths = [len(request.prompt) - prefix_hit_tokens for _index, request, prefix_hit_tokens, _reusable in group]
        if events is not None and suffix_lengths and max(suffix_lengths) == 0:
            return self._prefill_exact_prefix_batch(group, step, events=events)
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
            if any(item is None for item in reusable_logits):
                raise RuntimeError("exact-prefix reuse requires cached logits")
            logits = torch.cat([item.to(self.device) for item in reusable_logits], dim=0)

        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
        active = []
        for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
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

    def _prefill_exact_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        if not group:
            return []
        self.stats.prefill_prefix_reuse_batches += 1
        acquired_rows: list[int] = []
        try:
            next_tokens, logits_by_index = self._sample_exact_prefix_group(group)
            self.stats.prefix_reuse_requests += len(group)
            self.stats.prefix_reuse_tokens += sum(
                prefix_hit_tokens for _original_index, _request, prefix_hit_tokens, _reusable in group
            )

            active_items: list[tuple[_ActiveRequest, int, int]] = []
            copy_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            continuation_reuse_groups: dict[Hashable, list[int]] = defaultdict(list)
            continuation_reuse: dict[int, _ReusablePrefix] = {}
            pending_rows: list[int] = []
            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
                next_token = int(next_tokens[row_index])
                generated = 1
                finished = request.is_stop_token(next_token) or generated >= request.max_new_tokens
                if finished:
                    if events is not None:
                        events.append(
                            ServingTokenEvent(
                                request_id=request.request_id,
                                token=next_token,
                                step=step,
                                generated=generated,
                                finished=True,
                        )
                    )
                    continue

                if events is not None:
                    events.append(
                        ServingTokenEvent(
                            request_id=request.request_id,
                            token=next_token,
                            step=step,
                            generated=generated,
                            finished=False,
                        )
                    )
                generated_prefix = (*request.prompt, next_token)
                continuation = self._lookup_exact_reusable_prefix(generated_prefix)
                if continuation is not None:
                    continuation_reuse[row_index] = continuation
                    continuation_reuse_groups[continuation.route_id].append(row_index)
                    continue
                pending_rows.append(row_index)

            continuation_next_tokens: dict[int, int] = {}
            for indices in continuation_reuse_groups.values():
                continuation = continuation_reuse[indices[0]]
                if continuation.logits is None:
                    continue
                logits = continuation.logits.to(self.device)
                sampled = self._sample_repeated_logits(logits[:, -1, :], len(indices))
                sampled_tokens = sampled.detach().cpu().tolist()
                for token_index, row_index in enumerate(indices):
                    continuation_next_tokens[row_index] = int(sampled_tokens[token_index])

            continuation_copy_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            continuation_active: list[_ActiveRequest] = []
            for row_index, next_token in continuation_next_tokens.items():
                original_index, request, _prefix_hit_tokens, _reusable = group[row_index]
                continuation = continuation_reuse[row_index]
                second_token = int(next_token)
                generated = 2
                generated_prefix_len = len(request.prompt) + 1
                self.stats.prefix_reuse_requests += 1
                self.stats.prefix_reuse_tokens += generated_prefix_len
                self.stats.generated_prefix_reuse_requests += 1
                self.stats.generated_prefix_reuse_tokens += generated_prefix_len
                finished = request.is_stop_token(second_token) or generated >= request.max_new_tokens
                if events is not None:
                    events.append(
                        ServingTokenEvent(
                            request_id=request.request_id,
                            token=second_token,
                            step=step,
                            generated=generated,
                            finished=finished,
                        )
                    )
                if finished:
                    continue

                row = self._acquire_active_row()
                acquired_rows.append(row)
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, int(next_tokens[row_index]), second_token],
                    generated=generated,
                    row=row,
                    last_token=second_token,
                    seq_len=generated_prefix_len,
                    prefix_hit_tokens=generated_prefix_len,
                    started_step=step,
                )
                continuation_copy_groups[(continuation.row, generated_prefix_len)].append(row)
                continuation_active.append(state)

            for (source_row, tokens), dest_rows in continuation_copy_groups.items():
                self._copy_prefix_to_rows(source_row, dest_rows, tokens)
            for state in continuation_active:
                state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
                active.append(state)

            for row_index in pending_rows:
                original_index, request, prefix_hit_tokens, _reusable = group[row_index]
                next_token = int(next_tokens[row_index])
                generated = 1
                row = self._acquire_active_row()
                acquired_rows.append(row)
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=generated,
                    row=row,
                    last_token=next_token,
                    seq_len=len(request.prompt),
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
                copy_groups[(_reusable.row, prefix_hit_tokens)].append(row)
                active_items.append((state, row_index, row))

            for (source_row, prefix_hit_tokens), dest_rows in copy_groups.items():
                self._copy_prefix_to_rows(source_row, dest_rows, prefix_hit_tokens)
            for state, row_index, row in active_items:
                state.seq_len = self._cache_row_seq_len(row, len(state.request.prompt))
                self._store_reusable_prefix(
                    state.request.request_id,
                    state.request.prompt,
                    row,
                    logits_by_index[row_index],
                    allow_pinned=self._allow_pinned_full_prompt_store(state.request),
                )
                active.append(state)
            return active
        except Exception:
            for row in acquired_rows:
                self._release_active_row(row)
            raise

    def _sample_exact_prefix_group(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
    ) -> tuple[list[int], list[Tensor]]:
        by_route: dict[Hashable, list[int]] = defaultdict(list)
        for index, (_original_index, _request, _prefix_hit_tokens, reusable) in enumerate(group):
            by_route[reusable.route_id].append(index)

        next_tokens = [0 for _ in group]
        logits_by_index: list[Tensor | None] = [None for _ in group]
        for indices in by_route.values():
            reusable = group[indices[0]][3]
            if reusable.logits is None:
                raise RuntimeError("exact-prefix sampling requires cached logits")
            logits = reusable.logits.to(self.device)
            sampled = self._sample_repeated_logits(logits[:, -1, :], len(indices))
            sampled_tokens = sampled.detach().cpu().tolist()
            for token_index, group_index in enumerate(indices):
                next_tokens[group_index] = int(sampled_tokens[token_index])
                logits_by_index[group_index] = logits

        if any(logits is None for logits in logits_by_index):
            raise RuntimeError("exact-prefix sampling did not produce logits for every request")
        return next_tokens, [logits for logits in logits_by_index if logits is not None]

    def _sample_repeated_logits(self, logits: Tensor, batch_size: int) -> Tensor:
        if batch_size <= 1:
            return self._sample_logits(logits)
        sample_repeated = getattr(self.model, "sample_repeated_next_token", None)
        if callable(sample_repeated):
            return sample_repeated(logits, batch_size, self.temperature).to(self.device)
        expanded = logits.expand(batch_size, logits.size(-1)).contiguous()
        return self._sample_logits(expanded)

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
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    logits[row_index : row_index + 1],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
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
        self._record_shape_count(self.stats.prefill_shape_counts, f"plain:b{len(group)}:t{prompts.size(1)}")
        cache_view = self._cache_view(rows)
        logits, _ = self._prefill_logits(prompts, cache=cache_view)
        self._record_model_call("prefill", len(group), tokens=prompts.numel())
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
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

    def _prefill_ragged_graph_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        ragged_graph = getattr(self.model, "try_prefill_ragged_logits_graph", None)
        if ragged_graph is None:
            return None
        rows = [self._acquire_active_row() for _ in group]
        cache = self._require_cache()
        for row in rows:
            cache.clear_row(row)
            for layer_cache in cache.layers:
                layer_cache.keys[row].zero_()
                layer_cache.values[row].zero_()
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        suffix_bucket = self._suffix_bucket(max_len)
        batch_bucket = self._prefill_batch_bucket(len(group))
        self._record_shape_count(
            self.stats.prefill_shape_counts,
            f"ragged_graph:b{batch_bucket}:t{suffix_bucket}",
        )
        padded = []
        for _, request, _ in group:
            prompt = list(request.prompt)
            prompt.extend([0] * (suffix_bucket - len(prompt)))
            padded.append(prompt)
        while len(padded) < batch_bucket:
            padded.append([0] * suffix_bucket)
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        row_indices_list = list(rows)
        pad_row = self._free_active_rows[-1] if self._free_active_rows else rows[0]
        while len(row_indices_list) < batch_bucket:
            row_indices_list.append(pad_row)
        row_indices = torch.tensor(row_indices_list, device=self.device, dtype=torch.long)
        seq_lens_list = [0] * (max(row_indices_list) + 1)
        seq_lens = torch.tensor(seq_lens_list, device=self.device, dtype=torch.long)
        logit_pos = [l - 1 for l in lengths]
        while len(logit_pos) < batch_bucket:
            logit_pos.append(0)
        logit_positions = torch.tensor(logit_pos, device=self.device, dtype=torch.long)
        try:
            logits = ragged_graph(
                input_ids, self._require_cache(),
                seq_lens=seq_lens, row_indices=row_indices,
                logit_positions=logit_positions,
                context_len=suffix_bucket,
            )
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None
        n = len(group)
        logits = logits[:n]
        self._record_model_call("prefill", n, tokens=sum(lengths))
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
        for i, (_, request, _) in enumerate(group):
            self._set_cache_row_seq_len(rows[i], len(request.prompt))
        active = []
        for i, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[i]
            seq_len = len(request.prompt)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[i:i+1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            next_token = int(next_tokens[i])
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

    def _can_padded_batch_prefill(self, group: list[tuple[int, ServingRequest, int]]) -> bool:
        # Off by default: it forces seq_len=len(prompt) and changes prefill
        # grouping, which differs from _prefill_batch for skewed-seq-len models.
        # The shared-prefix workloads we care about use the common-prefix path
        # (and cross-batch prefix pinning) instead. Available via env when a
        # workload has no shared prefix but similar suffix lengths.
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_BATCH_PREFILL", True):
            return False
        if not callable(getattr(self.model, "forward", None)):
            return False
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        min_len = min(lengths)
        return max_len > 0 and min_len >= max_len * 0.5

    def _prefill_padded_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        self.stats.prefill_plain_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        self._record_shape_count(
            self.stats.prefill_shape_counts,
            f"padded_plain:b{len(group)}:t{min(lengths)}-{max_len}",
        )
        padded = []
        logit_positions = []
        for _i, (_, request, _) in enumerate(group):
            prompt = list(request.prompt)
            logit_positions.append(len(prompt) - 1)
            prompt.extend([0] * (max_len - len(prompt)))
            padded.append(prompt)
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        logit_pos_tensor = torch.tensor(logit_positions, device=self.device, dtype=torch.long)
        cache_view = self._cache_view(rows)
        selected_logits = self._forward_selected_logits(
            input_ids, cache=cache_view, logit_positions=logit_pos_tensor,
        )
        if selected_logits is not None:
            logits = selected_logits
        else:
            full_logits, _ = self._prefill_logits(input_ids, cache=cache_view)
            if full_logits.size(1) == 1:
                logits = full_logits
            else:
                logits = full_logits[
                    torch.arange(len(group), device=self.device),
                    torch.tensor([length - 1 for length in lengths], device=self.device),
                ].unsqueeze(1)
        self._record_model_call("prefill", len(group), tokens=input_ids.numel())
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()

        for row_index, (_, request, _) in enumerate(group):
            self._set_cache_row_seq_len(rows[row_index], len(request.prompt))

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            seq_len = len(request.prompt)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
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
            self._record_shape_count(self.stats.prefill_shape_counts, f"single:b1:t{input_ids.size(1)}")
            logits, _ = self._prefill_logits(input_ids, cache=self._cache_view([row]))
            self._record_model_call("prefill", 1, tokens=input_ids.numel())
        elif reusable is not None and reusable.logits is not None:
            logits = reusable.logits.to(self.device)
        else:
            raise RuntimeError("empty prompt suffix without a reusable prefix")

        next_token_t = self._sample_logits(logits[:, -1, :])
        next_token = int(next_token_t.item())
        seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
        self._store_reusable_prefix(
            request.request_id,
            request.prompt,
            row,
            logits,
            allow_pinned=self._allow_pinned_full_prompt_store(request),
        )
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

    def _flush_pending_decode(self, events: list[ServingTokenEvent] | None = None) -> None:
        pending = getattr(self, "_pending_decode", None)
        if pending is not None:
            cpu_buf, event, states, step, pending_events = pending
            event.synchronize()
            tokens = cpu_buf[:len(states)].tolist()
            target_events = events if events is not None else pending_events
            self._finalize_decode(tokens, states, step, target_events)
            self._pending_decode = None

    def _decode_active(
        self,
        active: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        # GPU-resident decode runner (token stash + async D2H). Gated by
        # TORCHINFERNO_DECODE_GRAPH_RUNNER (default off; the runner is only built when on),
        # handed to the engine via _decode_runner. Correctness is validated (greedy output
        # is bit-identical to the baseline path on 8-rank TP -- sampling collectives already
        # sync the token across ranks, so the GPU->GPU feed is TP-safe). The path is still
        # SYNCHRONOUS (get_cpu_tokens harvests immediately after step), so it does NOT yet
        # win: shape-dependent in A/B -- helps prefill-heavy few_shot (TPOT 68->29) but
        # hurts decode-bound long_output (24->30) and tree (57->68). The actual win needs
        # PIPELINING (double-buffer the readback + lagged harvest so decode replays run
        # back-to-back and the .cpu() sync overlaps GPU compute); see docs/PERF_GAP_ANALYSIS.
        runner = getattr(self, "_decode_runner", None)
        if runner is not None and active:
            return self._decode_active_with_runner(runner, active, step, events=events)
        _p = self.profile_timings
        _t0 = time.perf_counter() if _p else 0.0
        indexed_results: list[tuple[int, ServingResult]] = []
        live: list[_ActiveRequest] = []
        for state in active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                indexed_results.append((state.original_index, self._finish_and_release(state, step)))
            else:
                live.append(state)
        if _p:
            self.stats._da_filter_ms = getattr(self.stats, '_da_filter_ms', 0.0) + (time.perf_counter() - _t0) * 1000.0

        _t1 = time.perf_counter() if _p else 0.0
        next_active: list[_ActiveRequest] = []
        groups = [live] if self._can_decode_ragged(live) else self._decode_groups(live)
        if _p:
            self.stats._da_group_ms = getattr(self.stats, '_da_group_ms', 0.0) + (time.perf_counter() - _t1) * 1000.0
        for group in groups:
            if self._can_decode_ragged(group):
                decoded = self._decode_ragged_batch(group, step, events=events)
            else:
                decoded = (
                    self._decode_batch(group, step, events=events)
                    if len(group) > 1
                    else [self._decode_one(group[0], step, events=events)]
                )
            _t2 = time.perf_counter() if _p else 0.0
            for item, state in zip(decoded, group):
                if isinstance(item, ServingResult):
                    indexed_results.append((state.original_index, item))
                else:
                    next_active.append(item)
            if _p:
                self.stats._da_collect_ms = getattr(self.stats, '_da_collect_ms', 0.0) + (time.perf_counter() - _t2) * 1000.0
        return indexed_results, next_active

    def _decode_active_with_runner(
        self,
        runner: object,
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
        if not live:
            return indexed_results, []
        rows = [s.row for s in live]
        if runner.n_active != len(live) or runner.active_rows != rows:
            runner.set_active(rows, [s.last_token for s in live], [s.seq_len for s in live])
        runner.step()
        tokens = runner.get_cpu_tokens()
        next_active: list[_ActiveRequest] = []
        any_finished = False
        for i, state in enumerate(live):
            tok = int(tokens[i]) if i < len(tokens) else 0
            state.tokens.append(tok)
            state.generated += 1
            state.last_token = tok
            state.seq_len += 1
            self._remember_row_seq_len(state.row, state.seq_len)
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, tok, step, finished=finished)
            if finished:
                self._finish_and_release(state, step)
                any_finished = True
            else:
                next_active.append(state)
        if any_finished:
            runner.set_active(
                [s.row for s in next_active],
                [s.last_token for s in next_active],
                [s.seq_len for s in next_active],
            )
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
        rows = [state.row for state in states]
        if states:
            seq_len = states[0].seq_len
            if all(state.seq_len == seq_len for state in states):
                self._set_cache_rows_seq_len(rows, seq_len)
            else:
                for state in states:
                    self._set_cache_row_seq_len(state.row, state.seq_len)
        input_ids = torch.tensor([[state.last_token] for state in states], device=self.device, dtype=torch.long)
        reuse_logits: Tensor | None = None
        need_generated_prefix_logits = self._needs_generated_prefix_logits(states)
        if not hasattr(self, "_has_fi_decode"):
            self._has_fi_decode = bool(getattr(self.model, "_fi_decode_graphs", None))
        if self._has_fi_decode:
            row_indices_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            seq_lens_t = self._seq_lens_tensor(states, rows=rows)
            if need_generated_prefix_logits:
                logits = self._ragged_decode_logits(input_ids, seq_lens_t, row_indices_t)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits(reuse_logits)
            else:
                self._last_ragged_decode_logits = None
                fi_token = self._try_ragged_token_graph(input_ids, seq_lens_t, row_indices_t)
                if fi_token is not None:
                    next_token_tensor = fi_token.to(self.device)
                    reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
                    self.stats.decode_graph_hits += 1
                else:
                    cache_view = self._cache_view(rows)
                    logits = self._static_decode_logits(input_ids, cache_view)
                    reuse_logits = logits[:, -1, :]
                    next_token_tensor = self._sample_logits(logits[:, -1, :])
        else:
            cache_view = self._cache_view(rows)
            graph_token = None if need_generated_prefix_logits else self._try_static_token_graph(input_ids, cache_view)
            if graph_token is None:
                logits = self._static_decode_logits(input_ids, cache_view)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits(logits[:, -1, :])
            else:
                next_token_tensor = graph_token.to(self.device)
                self.stats.decode_graph_hits += 1
        self._record_model_call("decode", len(states), tokens=len(states))
        next_tokens = next_token_tensor.detach().cpu().tolist()
        self._store_decoded_reusable_prefixes(states, reuse_logits)

        decoded: list[_ActiveRequest | ServingResult] = []
        batched_next_seq_len: int | None = None
        if states:
            candidate_next_seq_len = states[0].seq_len + 1
            if all(state.seq_len + 1 == candidate_next_seq_len for state in states):
                self._set_cache_rows_seq_len(rows, candidate_next_seq_len)
                batched_next_seq_len = candidate_next_seq_len
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            if batched_next_seq_len is None:
                next_seq_len = state.seq_len + 1
                self._set_cache_row_seq_len(state.row, next_seq_len)
                state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
            else:
                state.seq_len = batched_next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        return decoded

    def _finalize_decode(
        self,
        next_tokens: list[int],
        states: list[_ActiveRequest],
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> None:
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
            self._remember_row_seq_len(state.row, next_seq_len)
            state.seq_len = next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                self._finish_and_release(state, step)

    def _decode_one(
        self,
        state: _ActiveRequest,
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> _ActiveRequest | ServingResult:
        self._set_cache_row_seq_len(state.row, state.seq_len)
        input_ids = torch.tensor([[state.last_token]], device=self.device, dtype=torch.long)
        reuse_logits: Tensor | None = None
        need_generated_prefix_logits = self._needs_generated_prefix_logits([state])
        if not hasattr(self, "_has_fi_decode"):
            self._has_fi_decode = bool(getattr(self.model, "_fi_decode_graphs", None))
        if self._has_fi_decode:
            row_indices_t = torch.tensor([state.row], dtype=torch.long, device=self.device)
            seq_lens_t = self._seq_lens_tensor([state], rows=[state.row])
            if need_generated_prefix_logits:
                logits = self._ragged_decode_logits(input_ids, seq_lens_t, row_indices_t)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits(reuse_logits)
            else:
                self._last_ragged_decode_logits = None
                fi_token = self._try_ragged_token_graph(input_ids, seq_lens_t, row_indices_t)
                if fi_token is not None:
                    next_token_tensor = fi_token.to(self.device)
                    reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
                    self.stats.decode_graph_hits += 1
                else:
                    cache_view = self._cache_view([state.row])
                    logits = self._static_decode_logits(input_ids, cache_view)
                    reuse_logits = logits[:, -1, :]
                    next_token_tensor = self._sample_logits(logits[:, -1, :])
        else:
            cache_view = self._cache_view([state.row])
            graph_token = None if need_generated_prefix_logits else self._try_static_token_graph(input_ids, cache_view)
            if graph_token is None:
                logits = self._static_decode_logits(input_ids, cache_view)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits(logits[:, -1, :])
            else:
                next_token_tensor = graph_token.to(self.device)
                self.stats.decode_graph_hits += 1
        self._record_model_call("decode", 1, tokens=1)
        next_token = int(next_token_tensor.item())
        self._store_decoded_reusable_prefixes([state], reuse_logits)
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

    def _start_decode_ragged_model_gpu_timer(self) -> tuple[object, object] | None:
        if not self.profile_timings or self.device.type != "cuda":
            return None
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
            return start, end
        except Exception:
            return None

    def _stop_decode_ragged_model_gpu_timer(self, events: tuple[object, object] | None) -> None:
        if events is None:
            return
        try:
            _start, end = events
            end.record(torch.cuda.current_stream(self.device))
        except Exception:
            return
        self._pending_decode_ragged_model_events.append(events)

    def _flush_decode_ragged_model_gpu_timers(self) -> None:
        pending = getattr(self, "_pending_decode_ragged_model_events", [])
        if not pending:
            return
        remaining: list[tuple[object, object]] = []
        for start, end in pending:
            try:
                self.stats.decode_ragged_model_gpu_ms += float(start.elapsed_time(end))
            except RuntimeError:
                remaining.append((start, end))
            except Exception:
                continue
        self._pending_decode_ragged_model_events = remaining

    def _ensure_gpu_token_buf(self) -> Tensor:
        buf = getattr(self, "_gpu_last_tokens", None)
        total = self.max_active_requests + (getattr(self, "prefix_cache_capacity", 0) or 0) + 2
        if buf is None or buf.size(0) < total:
            buf = torch.zeros(total, dtype=torch.long, device=self.device)
            self._gpu_last_tokens = buf
        return buf

    def _device_index_tensor(self, values: tuple[int, ...]) -> Tensor:
        cached = self._device_index_tensors.get(values)
        if cached is None:
            cached = torch.tensor(values, device=self.device, dtype=torch.long)
            self._device_index_tensors[values] = cached
        return cached

    def _sync_gpu_last_tokens_from_states(self, states: list[_ActiveRequest]) -> None:
        if not states:
            return
        rows = self._device_index_tensor(tuple(state.row for state in states))
        tokens = torch.tensor([state.last_token for state in states], device=self.device, dtype=torch.long)
        self._ensure_gpu_token_buf().index_copy_(0, rows, tokens)

    def _ensure_gpu_seq_lens_buf(self) -> Tensor:
        total = max(1, len(self._row_seq_lens))
        buf = getattr(self, "_gpu_seq_lens", None)
        if buf is None or buf.numel() < total:
            buf = torch.zeros(total, dtype=torch.long, device=self.device)
            self._gpu_seq_lens = buf
        return buf

    def _sync_gpu_seq_lens_from_states(self, states: list[_ActiveRequest]) -> None:
        if not states:
            return
        buf = self._ensure_gpu_seq_lens_buf()
        rows = self._device_index_tensor(tuple(state.row for state in states))
        seq_lens = torch.tensor([state.seq_len for state in states], device=self.device, dtype=torch.long)
        buf.index_copy_(0, rows, seq_lens)

    def _decode_many_seq_lens_tensor(self, states: list[_ActiveRequest], rows: list[int]) -> Tensor:
        buf = self._ensure_gpu_seq_lens_buf()
        if not states:
            return buf
        active_rows = {state.row for state in states}
        pad_seq_len = max(state.seq_len for state in states)
        pad_rows = tuple(
            row
            for row in rows
            if row not in active_rows and 0 <= row < len(self._row_seq_lens) and self._row_seq_lens[row] <= 0
        )
        if pad_rows:
            buf.index_fill_(0, self._device_index_tensor(pad_rows), int(pad_seq_len))
        return buf

    def _advance_gpu_seq_lens(self, rows: Tensor) -> None:
        buf = getattr(self, "_gpu_seq_lens", None)
        if buf is None or rows.numel() == 0:
            return
        buf.index_add_(0, rows, torch.ones_like(rows, dtype=torch.long))

    def _decode_ragged_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest | ServingResult]:
        prompt_lookup_decoded = self._try_prompt_lookup_decode_batch(
            states,
            step,
            events=events,
        )
        if prompt_lookup_decoded is not None:
            return prompt_lookup_decoded
        return self._decode_ragged_batch_baseline(states, step, events=events)

    def _try_prompt_lookup_decode_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult] | None:
        if not self._prompt_lookup_decode_enabled() or not states:
            return None

        proposals: list[tuple[int, ...]] = [
            self._prompt_lookup_proposal(state)
            for state in states
        ]
        candidate_indices = [
            index for index, proposal in enumerate(proposals)
            if proposal
        ]
        if not candidate_indices:
            return None

        decoded: list[_ActiveRequest | ServingResult | None] = [None for _ in states]
        for group_indices in self._prompt_lookup_groups(states, proposals, candidate_indices).values():
            if not group_indices:
                continue
            group_states = [states[index] for index in group_indices]
            group_proposals = [proposals[index] for index in group_indices]
            group_decoded = self._decode_prompt_lookup_group(
                group_states,
                group_proposals,
                step,
                events=events,
            )
            for index, item in zip(group_indices, group_decoded):
                decoded[index] = item

        remaining = [
            states[index]
            for index, item in enumerate(decoded)
            if item is None
        ]
        if remaining:
            fallback = self._decode_ragged_batch_baseline(remaining, step, events=events)
            fallback_iter = iter(fallback)
            for index, item in enumerate(decoded):
                if item is None:
                    decoded[index] = next(fallback_iter)

        return [item for item in decoded if item is not None]

    def _prompt_lookup_decode_enabled(self) -> bool:
        if self.temperature > 0.0:
            return False
        if not env_flag("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_DECODE", False):
            return False
        cache = self._require_cache()
        if getattr(cache, "cache_backend", self.cache_backend) != "dense":
            return False
        return True

    def _prompt_lookup_proposal(self, state: _ActiveRequest) -> tuple[int, ...]:
        min_max_new_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MIN_MAX_TOKENS",
            16,
            minimum=1,
        )
        if state.request.max_new_tokens < min_max_new_tokens:
            return ()
        remaining = state.request.max_new_tokens - state.generated
        if remaining <= 1:
            return ()
        ngram = env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_NGRAM", 3, minimum=1)
        history = state.tokens
        if len(history) <= ngram:
            return ()
        max_proposal = min(
            env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MAX_PROPOSAL_TOKENS", 8, minimum=1),
            remaining - 1,
        )
        if max_proposal <= 0:
            return ()
        min_proposal = min(
            env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MIN_PROPOSAL_TOKENS", 4, minimum=1),
            max_proposal,
        )
        needle = tuple(history[-ngram:])
        for start in range(len(history) - ngram - 1, -1, -1):
            if tuple(history[start : start + ngram]) != needle:
                continue
            proposal = tuple(history[start + ngram : start + ngram + max_proposal])
            if len(proposal) >= min_proposal:
                return proposal
        return ()

    @staticmethod
    def _prompt_lookup_groups(
        states: list[_ActiveRequest],
        proposals: list[tuple[int, ...]],
        candidate_indices: list[int],
    ) -> dict[tuple[int, int], list[int]]:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in candidate_indices:
            groups[(states[index].seq_len, len(proposals[index]))].append(index)
        return groups

    def _decode_prompt_lookup_group(
        self,
        states: list[_ActiveRequest],
        proposals: list[tuple[int, ...]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        if not states:
            return []
        proposal_len = len(proposals[0])
        input_ids = torch.tensor(
            [
                [state.last_token, *proposal]
                for state, proposal in zip(states, proposals)
            ],
            device=self.device,
            dtype=torch.long,
        )
        rows = [state.row for state in states]
        for state in states:
            self._set_cache_row_seq_len(state.row, state.seq_len)

        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        logits, _ = self._prefill_full_logits(input_ids, cache=self._cache_view(rows))
        self._stop_decode_ragged_model_gpu_timer(gpu_model_events)
        if self.profile_timings:
            self.stats.decode_ragged_model_ms += (time.perf_counter() - model_start_s) * 1000.0

        if logits.ndim != 3 or logits.size(1) < proposal_len + 1:
            for state in states:
                self._set_cache_row_seq_len(state.row, state.seq_len)
            return self._decode_ragged_batch_baseline(states, step, events=events)

        self._record_model_call(
            "decode",
            len(states),
            tokens=len(states) * (proposal_len + 1),
            ragged=True,
        )
        self._record_shape_count(
            self.stats.decode_shape_counts,
            f"prompt_lookup:b{len(states)}:proposal{proposal_len}",
        )
        self.stats.prompt_lookup_batches += 1
        self.stats.prompt_lookup_requests += len(states)
        self.stats.prompt_lookup_proposed_tokens += sum(len(proposal) for proposal in proposals)

        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        flat_logits = logits[:, : proposal_len + 1, :].reshape(-1, logits.size(-1))
        predicted = self._sample_logits(flat_logits).view(len(states), proposal_len + 1)
        predicted_tokens = predicted.detach().cpu().tolist()
        if self.profile_timings:
            self.stats.decode_ragged_cpu_tokens_ms += (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self._flush_decode_ragged_model_gpu_timers()

        decoded: list[_ActiveRequest | ServingResult] = []
        state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
        for state, proposal, row_predictions in zip(states, proposals, predicted_tokens):
            old_seq_len = state.seq_len
            emitted = [int(row_predictions[0])]
            for index, proposed_token in enumerate(proposal):
                if int(proposed_token) != int(row_predictions[index]):
                    break
                emitted.append(int(row_predictions[index + 1]))
            remaining = state.request.max_new_tokens - state.generated
            emitted = emitted[:remaining]

            finished_result: ServingResult | None = None
            emitted_count = 0
            for token in emitted:
                state.tokens.append(int(token))
                state.generated += 1
                state.last_token = int(token)
                emitted_count += 1
                finished = self._should_finish_after_decode(state)
                self._record_token_event(events, state, int(token), step, finished=finished)
                if finished:
                    self._set_cache_row_seq_len(state.row, old_seq_len + emitted_count)
                    state.seq_len = old_seq_len + emitted_count
                    finished_result = self._finish_and_release(state, step)
                    break

            state.seq_len = old_seq_len + emitted_count
            self.stats.prompt_lookup_accepted_tokens += max(0, emitted_count - 1)
            if finished_result is not None:
                decoded.append(finished_result)
                continue
            self._set_cache_row_seq_len(state.row, state.seq_len)
            decoded.append(state)
        if self.profile_timings:
            self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
        return decoded

    def _decode_ragged_batch_baseline(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        prepare_start_s = time.perf_counter() if self.profile_timings else 0.0
        rows = [state.row for state in states]
        decode_rows = self._ragged_decode_bucket_rows(rows)
        n_active = len(states)
        n_padded = len(decode_rows)
        self._record_shape_count(
            self.stats.decode_shape_counts,
            f"ragged:b{n_active}/{n_padded}",
        )
        pad_token = states[0].last_token
        input_tokens = [
            states[index].last_token if index < n_active else pad_token
            for index, _row in enumerate(decode_rows)
        ]
        input_ids = torch.tensor([[token] for token in input_tokens], device=self.device, dtype=torch.long)
        row_indices = torch.tensor(decode_rows, dtype=torch.long, device=self.device)
        seq_lens = self._seq_lens_tensor(states, rows=decode_rows)
        if self.profile_timings:
            self.stats.decode_ragged_prepare_ms += (time.perf_counter() - prepare_start_s) * 1000.0
        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        self._last_ragged_decode_logits = None
        reuse_logits: Tensor | None = None
        need_generated_prefix_logits = self._needs_generated_prefix_logits(states)
        graph_token = None if need_generated_prefix_logits else self._try_ragged_token_graph(input_ids, seq_lens, row_indices)
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._ragged_decode_logits(input_ids, seq_lens, row_indices)
            reuse_logits = logits[:, -1, :]
            next_token_tensor = self._sample_logits(logits[:, -1, :])
        self._stop_decode_ragged_model_gpu_timer(gpu_model_events)
        self._record_model_call("decode", n_padded, tokens=n_padded, ragged=True)
        if self.profile_timings:
            self.stats.decode_ragged_model_ms += (time.perf_counter() - model_start_s) * 1000.0
        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        next_tokens = next_token_tensor[:n_active].detach().cpu().tolist()
        if self.profile_timings:
            self.stats.decode_ragged_cpu_tokens_ms += (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self._flush_decode_ragged_model_gpu_timers()
        self._store_decoded_reusable_prefixes(
            states,
            None if reuse_logits is None else reuse_logits[:n_active],
        )
        return self._apply_decoded_tokens(states, next_tokens, step, events)

    def _decode_ragged_batch_token_tensor(self, states: list[_ActiveRequest]) -> Tensor:
        prepare_start_s = time.perf_counter() if self.profile_timings else 0.0
        rows = [state.row for state in states]
        decode_rows = self._ragged_decode_bucket_rows(rows)
        n_active = len(states)
        n_padded = len(decode_rows)
        row_indices = self._device_index_tensor(tuple(decode_rows))
        input_ids = self._ensure_gpu_token_buf().index_select(0, row_indices).view(n_padded, 1)
        seq_lens = self._decode_many_seq_lens_tensor(states, decode_rows)
        if self.profile_timings:
            self.stats.decode_ragged_prepare_ms += (time.perf_counter() - prepare_start_s) * 1000.0

        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        self._last_ragged_decode_logits = None
        graph_token = self._try_ragged_token_graph(input_ids, seq_lens, row_indices)
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._ragged_decode_logits(input_ids, seq_lens, row_indices)
            next_token_tensor = self._sample_logits(logits[:, -1, :])
        self._stop_decode_ragged_model_gpu_timer(gpu_model_events)
        self._record_model_call("decode", n_padded, tokens=n_padded, ragged=True)
        active_row_indices = row_indices[:n_active]
        self._ensure_gpu_token_buf().index_copy_(0, active_row_indices, next_token_tensor[:n_active])
        self._advance_gpu_seq_lens(active_row_indices)
        if self.profile_timings:
            self.stats.decode_ragged_model_ms += (time.perf_counter() - model_start_s) * 1000.0
        return next_token_tensor

    def _apply_decoded_tokens(
        self,
        states: list[_ActiveRequest],
        next_tokens: list[int],
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
        decoded: list[_ActiveRequest | ServingResult] = []
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
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
        row_adopted = self._store_finished_reusable_prefix(state)
        result = ServingResult(
            state.request.request_id,
            tuple(state.tokens),
            state.prefix_hit_tokens,
            state.request.arrival_step,
            state.started_step,
            step,
        )
        if not row_adopted:
            self._release_active_row(state.row)
        return result

    def _allow_pinned_full_prompt_store(self, request: ServingRequest) -> bool:
        threshold = env_int(
            "TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_MIN_MAX_TOKENS",
            0,
            minimum=0,
        )
        return threshold > 0 and request.max_new_tokens >= threshold

    def _store_reusable_prefix(
        self,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor,
        *,
        allow_pinned: bool = False,
    ) -> None:
        if not self.store_full_prompt_prefixes:
            return
        if self.pin_shared_prefix and not allow_pinned:
            # Per-request full-prompt stores would starve the prefix-row pool.
            # Keep only pinned shared prefixes in this mode; unpinned modes
            # remove evicted routes from the radix index so shorter live prefixes
            # can still match. A long-output opt-in below enables this for
            # multi-turn style workloads where one full-prompt row can save a
            # much larger next-turn suffix prefill.
            return
        store_logits = True
        if allow_pinned:
            store_logits = env_flag(
                "TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_LOGITS",
                False,
            )
        if "TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS" in os.environ:
            store_logits = env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS", store_logits)
        self._store_reusable_prefix_tokens(
            None,
            request_id,
            tokens,
            source_row,
            logits,
            store_logits=store_logits,
        )

    def _generated_prefix_cache_base_enabled(self) -> bool:
        return (
            self.prefix_cache_capacity > 0
            and self.store_reusable_prefixes
            and self.store_full_prompt_prefixes
        )

    def _generated_prefix_cache_enabled(self) -> bool:
        if not self._generated_prefix_cache_base_enabled():
            return False
        configured = self.generated_prefix_cache
        if configured is not None:
            return bool(configured)
        if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE", False):
            return True
        if not env_flag("TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE", False):
            return False
        return any(
            isinstance(route_id, tuple) and route_id[:1] == ("generated_prefix",)
            for route_id in self.reusable_prefixes
        )

    def _should_collect_generated_prefix_logits(self, states: list[_ActiveRequest]) -> bool:
        if not states or not self._generated_prefix_cache_base_enabled():
            return False
        configured = self.generated_prefix_cache
        if configured is not None:
            return bool(configured)
        if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE", False):
            return True
        if not env_flag("TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE", False):
            return False
        waiting = self._online_waiting
        if waiting is None or not bool(waiting):
            return False
        candidate_prompts = {
            state.request.prompt
            for state in states
            if state.generated > 0 and self._state_has_full_prompt_kv(state)
        }
        if not candidate_prompts:
            return False
        pending_exact = sum(
            1
            for item in waiting._items
            if item.request.prompt in candidate_prompts
        )
        min_pending = env_int(
            "TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_MIN_PENDING",
            16,
            minimum=1,
        )
        return pending_exact >= min_pending

    def _finished_prefix_cache_enabled(self) -> bool:
        return (
            self.prefix_cache_capacity > 0
            and self.store_reusable_prefixes
            and self.store_full_prompt_prefixes
            and env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE", False)
        )

    @staticmethod
    def _generated_prefix_route_id(tokens: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        return ("generated_prefix", tokens)

    @staticmethod
    def _finished_prefix_route_id(tokens: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        return ("finished_prefix", tokens)

    def _lookup_exact_reusable_prefix(self, tokens: tuple[int, ...]) -> _ReusablePrefix | None:
        if not self._generated_prefix_cache_enabled():
            return None
        route_id = self._generated_prefix_route_id(tokens)
        reusable = self.reusable_prefixes.get(route_id)
        if reusable is not None and reusable.tokens == tokens:
            return reusable
        match, entry = self.prefix_cache.lookup(tokens)
        if entry is None or match.depth != len(tokens):
            return None
        reusable = self.reusable_prefixes.get(entry.route_id)
        if reusable is None or reusable.tokens != tokens:
            return None
        return reusable

    def _store_generated_reusable_prefix(
        self,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor,
    ) -> None:
        if not self._generated_prefix_cache_base_enabled():
            return
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) <= 0 or len(tokens) > max_tokens:
            return
        route_id = self._generated_prefix_route_id(tokens)
        if route_id in self.reusable_prefixes:
            return
        store_logits = logits[:, None, :] if logits.ndim == 2 else logits
        try:
            self._store_reusable_prefix_tokens(route_id, request_id, tokens, source_row, store_logits)
        except Exception:
            if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_STRICT", False):
                raise
            return
        if route_id in self.reusable_prefixes:
            self.stats.generated_prefix_store_requests += 1

    def _store_finished_reusable_prefix(self, state: _ActiveRequest) -> bool:
        if not self._finished_prefix_cache_enabled():
            return False
        if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
            return False
        kv_token_count = min(
            self._cache_row_seq_len(state.row, state.seq_len),
            len(state.tokens),
        )
        tokens = tuple(state.tokens[:kv_token_count])
        if len(tokens) <= len(state.request.prompt):
            return False
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            return False
        route_id = self._finished_prefix_route_id(tokens)
        if route_id in self.reusable_prefixes:
            return False
        if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_ADOPT_ROWS", True):
            try:
                if self._adopt_reusable_prefix_tokens(
                    route_id,
                    state.request.request_id,
                    tokens,
                    state.row,
                ):
                    self.stats.generated_prefix_store_requests += 1
                    return True
            except Exception:
                if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_STRICT", False):
                    raise
        try:
            self._set_cache_row_seq_len(state.row, len(tokens))
            self._store_reusable_prefix_tokens(
                route_id,
                state.request.request_id,
                tokens,
                state.row,
                None,
                store_logits=False,
            )
        except Exception:
            if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_STRICT", False):
                raise
            return False
        if route_id in self.reusable_prefixes:
            self.stats.generated_prefix_store_requests += 1
        return False

    def _store_decoded_reusable_prefixes(
        self,
        states: list[_ActiveRequest],
        logits: Tensor | None,
    ) -> None:
        if logits is None or not states or not self._should_collect_generated_prefix_logits(states):
            return
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for row_index, state in enumerate(states):
            if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
                continue
            tokens = tuple(state.tokens)
            if len(tokens) <= len(state.request.prompt):
                continue
            route_id = self._generated_prefix_route_id(tokens)
            if route_id in seen or route_id in self.reusable_prefixes:
                continue
            seen.add(route_id)
            self._set_cache_row_seq_len(state.row, len(tokens))
            self._store_generated_reusable_prefix(
                state.request.request_id,
                tokens,
                state.row,
                logits[row_index : row_index + 1],
            )

    @staticmethod
    def _state_has_full_prompt_kv(state: _ActiveRequest) -> bool:
        prompt_len = len(state.request.prompt)
        return state.prefix_hit_tokens >= prompt_len or state.seq_len >= prompt_len

    def _needs_generated_prefix_logits(self, states: list[_ActiveRequest]) -> bool:
        if not states or not self._should_collect_generated_prefix_logits(states):
            return False
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        for state in states:
            if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
                continue
            tokens = tuple(state.tokens)
            if len(tokens) <= len(state.request.prompt) or len(tokens) > max_tokens:
                continue
            if self._generated_prefix_route_id(tokens) not in self.reusable_prefixes:
                return True
        return False

    def _store_reusable_prefix_tokens(
        self,
        route_id: Hashable | None,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor | None,
        *,
        store_logits: bool = True,
    ) -> None:
        if not self.store_reusable_prefixes or not env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE", True):
            return
        actual_route = request_id if route_id is None else route_id
        old_prefix = self.reusable_prefixes.pop(actual_route, None)
        if old_prefix is not None:
            self._clear_physical_row(old_prefix.row)
            if actual_route in self._prefix_order:
                self._prefix_order.remove(actual_route)
            self._free_prefix_rows.append(old_prefix.row)
            self._free_prefix_rows.sort()
        # Drop any stale pin; the caller re-pins after a successful re-store.
        self._pinned_prefix_routes.discard(actual_route)
        self.prefix_cache.remove(actual_route)
        prefix_row = self._acquire_prefix_row()
        if env_flag("TORCHINFERNO_REUSE_DEBUG", False):
            import sys as _sd
            print(
                f"[STORE_DBG] rank={getattr(self.model, 'rank', 0)} "
                f"store={'SKIP(no_row)' if prefix_row is None else prefix_row} "
                f"ntoks={len(tokens)} free_prefix_rows={len(self._free_prefix_rows)} "
                f"cached={len(self.reusable_prefixes)}",
                file=_sd.stderr, flush=True,
            )
        if prefix_row is None:
            return
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        self._copy_prefix(source_row, prefix_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            prefix_row,
            logits[:, -1:, :].detach().clone().cpu() if store_logits and logits is not None else None,
        )
        self._prefix_order.append(entry.route_id)

    def _adopt_reusable_prefix_tokens(
        self,
        route_id: Hashable,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
    ) -> bool:
        if not self.store_reusable_prefixes or not env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE", True):
            return False
        if route_id in self.reusable_prefixes:
            return False
        replacement_active_row = self._acquire_prefix_row()
        if replacement_active_row is None:
            return False
        self.prefix_cache.remove(route_id)
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        self._set_cache_row_seq_len(source_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            source_row,
            None,
        )
        self._prefix_order.append(entry.route_id)
        self._remember_row_seq_len(replacement_active_row, 0)
        if replacement_active_row not in self._free_active_rows:
            self._free_active_rows.append(replacement_active_row)
        return True

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
        # Skip the GPU KV zero on release: _acquire_active_row already clears a
        # row before any reuse, so clearing here too is a redundant second pass
        # that runs inside the hot decode state-update loop (once per finishing
        # request). Only the seq_len reset is needed for correctness -- a free
        # row reused as decode-bucket padding has seq_len 0 so attention never
        # reads its stale KV, and its bucket output is discarded regardless.
        self._remember_row_seq_len(row, 0)
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
            for row in _preferred_prefix_rows():
                if row in self._free_prefix_rows:
                    self._free_prefix_rows.remove(row)
                    return row
            return self._free_prefix_rows.pop()
        # Evict the oldest UNPINNED reusable prefix. Pinned routes (the active
        # shared prefix) are skipped so their KV stays a valid copy source.
        for index, route_id in enumerate(self._prefix_order):
            if route_id in self._pinned_prefix_routes:
                continue
            self._prefix_order.pop(index)
            prefix = self.reusable_prefixes.pop(route_id, None)
            self.prefix_cache.remove(route_id)
            if prefix is not None:
                self._clear_physical_row(prefix.row)
                return prefix.row
            return self._acquire_prefix_row()
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
        self._cache_view([row]).clear_row(0)  # type: ignore[attr-defined]
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

    def _prefill_full_logits(self, input_ids: Tensor, *, cache: object) -> tuple[Tensor, object | None]:
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if not callable(forward):
            raise TypeError("serving model must be callable or expose forward()")
        kwargs: dict[str, object] = {
            "cache": cache,
            "use_cache": True,
        }
        if self._prefer_sharded_logits():
            kwargs["return_last_logits_only"] = False
            kwargs["return_sharded_logits"] = True
        try:
            return forward(input_ids, **kwargs)
        except TypeError:
            return forward(input_ids, cache=cache, use_cache=True)

    def _prefill_logits(self, input_ids: Tensor, *, cache: object) -> tuple[Tensor, object | None]:
        graph_logits = self._try_prefill_logits_graph(input_ids, cache)
        if graph_logits is not None:
            return graph_logits, cache
        fi_fwd = getattr(self.model, "forward_step_flashinfer", None)
        fi_ready = getattr(self.model, "_flashinfer_jit_warmed", False)
        cache_layers = getattr(self._require_cache(), "layers", None)
        has_paged = cache_layers and hasattr(cache_layers[0], "paged_kv")
        if fi_fwd is not None and fi_ready and has_paged and input_ids.device.type == "cuda":
            try:
                batch, seq_len = input_ids.shape
                full_cache = self._require_cache()
                parent = getattr(cache, "_parent_cache", None)
                rows_attr = getattr(cache, "_rows", None) or getattr(cache, "_row_list", None)
                if rows_attr is not None and parent is not None:
                    row_list = list(rows_attr) if not isinstance(rows_attr, list) else rows_attr
                else:
                    row_list = list(range(batch))
                    full_cache = cache
                seq_lens_val = 0
                try:
                    seq_lens_val = int(cache.seq_len)
                except Exception:
                    pass
                # Reject writes past the row capacity BEFORE launching the kernel:
                # a KV scatter at columns [seq_lens_val, seq_lens_val+seq_len) that
                # exceeds max_seq is a device-side index assert that kills every TP
                # rank. Raise a catchable error so one request fails instead.
                _ms = self._cache_max_seq_len()
                if _ms is not None and seq_lens_val + seq_len > _ms:
                    raise RuntimeError(
                        f"prefill write past cache: seq_lens={seq_lens_val} + "
                        f"q={seq_len} > max_seq={_ms}"
                    )
                row_indices = torch.tensor(row_list[:batch], device=input_ids.device, dtype=torch.long)
                seq_lens = torch.full((batch,), seq_lens_val, device=input_ids.device, dtype=torch.long)
                q_lens = torch.full((batch,), seq_len, device=input_ids.device, dtype=torch.long)
                write_pos = torch.arange(seq_lens_val, seq_lens_val + seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
                logit_pos = torch.full((batch,), seq_len - 1, device=input_ids.device, dtype=torch.long)
                logits = fi_fwd(
                    input_ids, full_cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_pos, logit_positions=logit_pos,
                    row_indices=row_indices,
                )
                for lc in full_cache.layers:
                    if hasattr(lc, '_seq_lens'):
                        for r in row_list[:batch]:
                            lc._seq_lens[r] = seq_lens_val + seq_len
                    if hasattr(lc, '_uniform_seq_len'):
                        lc._uniform_seq_len[0] = None
                return logits, cache
            except Exception as _fi_exc:
                import sys as _fis
                print(
                    f"[PREFILL] FlashInfer eager prefill failed: {_fi_exc!r}",
                    file=_fis.stderr, flush=True,
                )
        if has_paged:
            # The SDPA dense _forward_model below indexes the cache as a contiguous
            # [batch, ...] tensor, which CUDA index-asserts against the paged
            # FlashInfer cache and takes down EVERY tensor-parallel rank (an
            # unrecoverable device-side assert, not a Python exception). When the
            # cache is paged, fail THIS request with a catchable error instead so
            # the engine degrades one request rather than crashing the server.
            raise RuntimeError(
                "paged-cache prefill requires a FlashInfer graph/eager path; "
                "refusing to fall back to the SDPA dense forward"
            )
        return self._forward_model(input_ids, cache=cache, use_cache=True)

    def _try_flashinfer_prefill(
        self,
        requests: list[tuple[int, "ServingRequest", int, "_ReusablePrefix | None"]],
        step: int,
        *,
        events: list["ServingTokenEvent"] | None = None,
        graph_only: bool = False,
    ) -> list["_ActiveRequest"] | None:
        forward_fi = getattr(self.model, "forward_step_flashinfer", None)
        if forward_fi is None:
            return None
        try:
            __import__("flashinfer")
        except ImportError:
            return None
        cache = self._require_cache()
        active: list[_ActiveRequest] = []
        rows = []
        prompts: list[tuple[int, ...]] = []
        for original_index, request, prefix_hit_tokens, reusable in requests:
            row = self._acquire_active_row()
            rows.append(row)
            prompts.append(request.prompt)

        if not prompts:
            return None

        prompt_lens = [len(p) for p in prompts]
        max_prompt_len = max(prompt_lens)
        batch = len(prompts)
        padded = [list(p) + [0] * (max_prompt_len - len(p)) for p in prompts]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        q_lens = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        seq_lens = torch.zeros(batch, dtype=torch.long, device=self.device)
        row_indices = torch.tensor(rows, dtype=torch.long, device=self.device)
        positions = []
        for i in range(batch):
            positions.append(list(range(max_prompt_len)))
        write_positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        logit_positions = torch.tensor(
            [plen - 1 for plen in prompt_lens], dtype=torch.long, device=self.device,
        )
        try:
            logits = None
            graph_fn = getattr(self.model, "try_prefill_flashinfer_graph", None)
            if graph_fn is not None:
                logits = graph_fn(
                    input_ids, cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_positions,
                    logit_positions=logit_positions,
                    row_indices=row_indices,
                )
            if logits is None and graph_only:
                # Graph missed and the caller (single-request path) does not want
                # the launch-bound eager FlashInfer fallback; release rows so the
                # normal _prefill_one/SDPA path can handle this request instead.
                for row in rows:
                    self._release_active_row(row)
                return None
            if logits is None:
                logits = forward_fi(
                    input_ids, cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_positions,
                    logit_positions=logit_positions,
                    row_indices=row_indices,
                )
        except Exception as _fi_prefill_exc:
            import sys as _fps
            import traceback as _fptb
            print(
                f"[FI_PREFILL] FlashInfer prefill failed batch={batch}: {_fi_prefill_exc}",
                file=_fps.stderr, flush=True,
            )
            _fptb.print_exc(file=_fps.stderr)
            for row in rows:
                self._release_active_row(row)
            return None
        self._record_model_call("prefill", batch, tokens=int(q_lens.sum().item()))
        next_tokens = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
        for i in range(batch):
            self._set_cache_row_seq_len(rows[i], len(requests[i][1].prompt))
        for i, (original_index, request, prefix_hit_tokens, reusable) in enumerate(requests):
            row = rows[i]
            next_token = int(next_tokens[i])
            prompt_len = len(request.prompt)
            seq_len = prompt_len
            self._remember_row_seq_len(row, seq_len)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[i:i+1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
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

    def _prefill_flashinfer_reuse(
        self,
        group: list[tuple[int, "ServingRequest", int, "_ReusablePrefix"]],
        step: int,
        *,
        events: list["ServingTokenEvent"] | None = None,
    ) -> list["_ActiveRequest"] | None:
        # FlashInfer-native prefix reuse: copy cached prefix KV into each row,
        # then prefill only the suffix (seq_lens=prefix_len). Empty suffix (whole
        # prompt cached) just samples the cached logits. FlashInfer-cache-safe.
        forward_fi = getattr(self.model, "forward_step_flashinfer", None)
        if forward_fi is None:
            return None
        try:
            __import__("flashinfer")
        except ImportError:
            return None
        max_seq = self._cache_max_seq_len()
        for _idx, request, hit, reusable in group:
            if reusable is None or hit <= 0 or hit > len(request.prompt):
                return None
            if max_seq is not None and len(request.prompt) > max_seq:
                return None
            if reusable.row < 0:
                return None
        cache = self._require_cache()
        rows: list[int] = []
        try:
            for _idx, request, hit, reusable in group:
                row = self._acquire_active_row()
                self._copy_prefix(reusable.row, row, hit)
                rows.append(row)
                self.stats.prefix_reuse_requests += 1
                self.stats.prefix_reuse_tokens += hit
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None

        n = len(group)
        next_tokens: list[int | None] = [None] * n
        out_logits: list[Tensor | None] = [None] * n
        suffix_idx = [i for i, (_x, req, hit, _r) in enumerate(group) if len(req.prompt) > hit]
        full_idx = [i for i, (_x, req, hit, _r) in enumerate(group) if len(req.prompt) <= hit]
        try:
            if suffix_idx:
                suffixes = [list(group[i][1].prompt[group[i][2]:]) for i in suffix_idx]
                msl = max(len(s) for s in suffixes)
                padded = [s + [0] * (msl - len(s)) for s in suffixes]
                b = len(suffix_idx)
                input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
                q_lens = torch.tensor([len(s) for s in suffixes], dtype=torch.long, device=self.device)
                seq_lens = torch.tensor([group[i][2] for i in suffix_idx], dtype=torch.long, device=self.device)
                ris = torch.tensor([rows[i] for i in suffix_idx], dtype=torch.long, device=self.device)
                wpos = torch.tensor(
                    [[group[i][2] + j for j in range(msl)] for i in suffix_idx],
                    dtype=torch.long, device=self.device,
                )
                lpos = torch.tensor([len(s) - 1 for s in suffixes], dtype=torch.long, device=self.device)
                logits = forward_fi(
                    input_ids, cache, seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=wpos, logit_positions=lpos, row_indices=ris,
                )
                self._record_model_call("prefill", b, tokens=int(q_lens.sum().item()))
                toks = self._sample_logits(logits[:, -1, :]).detach().cpu().tolist()
                for k, i in enumerate(suffix_idx):
                    next_tokens[i] = int(toks[k])
                    out_logits[i] = logits[k:k + 1]
            for i in full_idx:
                if group[i][3].logits is None:
                    return None
                cached = group[i][3].logits.to(self.device)
                next_tokens[i] = int(self._sample_logits(cached[:, -1, :]).item())
                out_logits[i] = cached
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None

        self.stats.prefill_prefix_reuse_batches += 1
        active: list[_ActiveRequest] = []
        for i, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
            row = rows[i]
            self._set_cache_row_seq_len(row, len(request.prompt))
            self._remember_row_seq_len(row, len(request.prompt))
            next_token = int(next_tokens[i])
            if out_logits[i] is not None:
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    out_logits[i],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=len(request.prompt),
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

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
            env_flag("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", self.graph_prefill),
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
        elif kind == "unified":
            self.stats.prefill_model_calls += 1
            self.stats.decode_model_calls += 1
            self.stats.prefill_tokens += tokens
            self.stats.decode_tokens += tokens
        self.stats.max_model_batch_size = max(self.stats.max_model_batch_size, batch_size)

    def _record_shape_count(self, counts: dict[str, int], key: str) -> None:
        if not self.profile_timings:
            return
        counts[key] = counts.get(key, 0) + 1

    def _can_decode_ragged(self, states: list[_ActiveRequest]) -> bool:
        if not self.enable_ragged_decode:
            return False
        if len(states) <= 1:
            return False
        if not hasattr(self, "_uniform_ragged"):
            self._uniform_ragged = env_flag("TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE", False)
        if len({state.seq_len for state in states}) <= 1 and not self._uniform_ragged:
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
        # Cheap path FIRST: set the per-layer seq_len list (and per-layer setter)
        # directly. The for_rows-view path below builds an UNCACHED view with a
        # fresh GPU index tensor on every call -- when run once per row in the
        # prefill/decode state loops that allocation dominated wall time (~49% of
        # prefill). The direct list write is pure Python and equivalent for the
        # dense cache; the view path stays as a fallback for backends (e.g.
        # paged) that manage seq_len behind a view.
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
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                view = self._cache_view([row])
                setter = getattr(view, "set_seq_len", None)
                if callable(setter):
                    setter(int(seq_len))
                    return
            except Exception:
                pass
        seq_lens = getattr(cache, "_seq_lens", None)
        if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
            seq_lens[row] = int(seq_len)

    def _set_cache_rows_seq_len(self, rows: list[int], seq_len: int) -> None:
        if not rows:
            return
        seq_len = int(seq_len)
        for row in rows:
            self._remember_row_seq_len(row, seq_len)
        cache = self._require_cache()
        row_tuple = tuple(int(row) for row in rows)
        layers = tuple(getattr(cache, "layers", ()) or ())
        changed = False
        for layer in layers:
            setter = getattr(layer, "_set_rows_seq_len", None)
            if callable(setter):
                try:
                    setter(row_tuple, seq_len)
                    changed = True
                    continue
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list):
                for row in row_tuple:
                    if 0 <= row < len(seq_lens):
                        seq_lens[row] = seq_len
                        changed = True
                uniform = getattr(layer, "_uniform_seq_len", None)
                if isinstance(uniform, list) and uniform:
                    uniform[0] = seq_len if all(value == seq_len for value in seq_lens) else None
        if changed:
            return
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                view = self._cache_view(row_tuple)
                setter = getattr(view, "set_seq_len", None)
                if callable(setter):
                    setter(seq_len)
                    return
            except Exception:
                pass
        seq_lens = getattr(cache, "_seq_lens", None)
        if isinstance(seq_lens, list):
            for row in row_tuple:
                if 0 <= row < len(seq_lens):
                    seq_lens[row] = seq_len

    def _try_ragged_token_graph(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
    ) -> Tensor | None:
        fi_decode_mode = _fi_decode_graph_mode()
        use_fi_decode = fi_decode_mode == "always" or (
            fi_decode_mode == "sampled" and float(self.temperature) > 0.0
        )
        fi_graphs = (
            getattr(self.model, "_fi_decode_graphs", None)
            if use_fi_decode
            else None
        )
        if fi_graphs:
            batch = input_ids.size(0)
            bucket = 1 << (batch - 1).bit_length() if batch > 1 else 1
            entry = fi_graphs.get(bucket)
            if entry is not None:
                graph, dw, s_ids, s_wp, s_ri, s_logits, nqo, nkv, hd, ms, qd = entry
                s_ids[:batch].copy_(input_ids)
                s_ri[:batch].copy_(row_indices)
                if batch < bucket:
                    s_ids[batch:] = 0
                    s_ri[batch:] = 0
                fi_bufs = getattr(self, "_fi_bufs", {})
                if bucket not in fi_bufs:
                    fi_bufs[bucket] = (
                        torch.arange(bucket + 1, dtype=torch.int32, device=self.device),
                        torch.ones(bucket, dtype=torch.int32, device=self.device),
                    )
                    self._fi_bufs = fi_bufs
                indptr, lpl = fi_bufs[bucket]
                lpl.fill_(1)
                indices = s_ri.to(dtype=torch.int32)
                row_sl = seq_lens[row_indices[:batch].long()]
                s_wp[:batch, 0].copy_(row_sl)
                lpl[:batch] = (row_sl + 1).to(torch.int32)
                if batch < bucket:
                    s_wp[batch:] = 0
                dw.plan(indptr=indptr, indices=indices, last_page_len=lpl,
                        num_qo_heads=nqo, num_kv_heads=nkv, head_dim=hd, page_size=ms, q_data_type=qd)
                graph.replay()
                last_logits = s_logits[:batch, -1, :]
                self._last_ragged_decode_logits = last_logits
                return self._sample_logits(last_logits)
        decode_graph = getattr(self.model, "try_decode_ragged_token_graph", None)
        if decode_graph is None:
            return None
        token = self._call_decode_graph(
            decode_graph,
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            temperature=self.temperature,
            capture_on_miss=self._decode_capture_on_miss(),
        )
        if token is None:
            self.stats.decode_graph_misses += 1
        return token

    def _try_static_token_graph(self, input_ids: Tensor, cache: object) -> Tensor | None:
        decode_graph = getattr(self.model, "try_decode_one_token_graph", None)
        if decode_graph is None:
            self._report_static_graph_miss(input_ids, cache, "no_token_graph")
            return None
        token = self._call_decode_graph(
            decode_graph,
            input_ids,
            cache,
            temperature=self.temperature,
            capture_on_miss=self._decode_capture_on_miss(),
        )
        if token is None:
            self.stats.decode_graph_misses += 1
            self._report_static_graph_miss(input_ids, cache, "token_graph_returned_none")
        return token

    def _static_decode_logits(self, input_ids: Tensor, cache: object) -> Tensor:
        decode_graph = getattr(self.model, "try_decode_one_token_logits_graph", None)
        if decode_graph is not None:
            logits = self._call_decode_graph(
                decode_graph,
                input_ids,
                cache,
                capture_on_miss=self._decode_capture_on_miss(),
            )
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
            logits = self._call_decode_graph(
                decode_graph,
                input_ids,
                self._require_cache(),
                seq_lens=seq_lens,
                row_indices=row_indices,
                capture_on_miss=self._decode_capture_on_miss(),
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

    def _decode_capture_on_miss(self) -> bool:
        if "TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE" in os.environ:
            return env_flag("TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE", False)
        if self._generated_prefix_cache_enabled():
            return False
        return True

    @staticmethod
    def _call_decode_graph(
        graph: Callable[..., Tensor | None],
        input_ids: Tensor,
        cache: object,
        *,
        capture_on_miss: bool,
        temperature: float | None = None,
        seq_lens: Tensor | None = None,
        row_indices: Tensor | None = None,
    ) -> Tensor | None:
        try:
            parameters = signature(graph).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, object] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seq_lens is not None:
            kwargs["seq_lens"] = seq_lens
            kwargs["row_indices"] = row_indices
        if "capture_on_miss" in parameters:
            kwargs["capture_on_miss"] = capture_on_miss
        return graph(input_ids, cache, **kwargs)

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

    def _release_online_prefill_finished(
        self,
        active: list[_ActiveRequest],
        step: int,
    ) -> list[_ActiveRequest]:
        live: list[_ActiveRequest] = []
        for state in active:
            if self._should_finish_before_decode(state):
                self._finish_and_release(state, step)
            else:
                live.append(state)
        return live

    @staticmethod
    def _should_finish_before_decode(state: _ActiveRequest) -> bool:
        if state.request.is_stop_token(state.last_token):
            return True
        return state.generated >= state.request.max_new_tokens

    @staticmethod
    def _should_finish_after_decode(state: _ActiveRequest) -> bool:
        if state.request.is_stop_token(state.last_token):
            return True
        return state.generated >= state.request.max_new_tokens


def _common_prefix_token_count(prompts: list[tuple[int, ...]]) -> int:
    if len(prompts) <= 1:
        return 0
    min_len = min((len(prompt) for prompt in prompts), default=0)
    if min_len <= 1:
        return 0
    prefix_tokens = 0
    for offset in range(min_len):
        token = prompts[0][offset]
        if any(prompt[offset] != token for prompt in prompts[1:]):
            break
        prefix_tokens += 1
    return prefix_tokens
