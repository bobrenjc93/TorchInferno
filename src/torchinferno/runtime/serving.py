from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Optional

import torch
from torch import Tensor

from torchinferno.runtime.prefix_cache import PrefixCacheIndex


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


@dataclass
class ServingStats:
    prefill_model_calls: int = 0
    prefill_batches: int = 0
    decode_model_calls: int = 0
    decode_batches: int = 0
    prefix_reuse_requests: int = 0
    prefix_reuse_tokens: int = 0
    max_model_batch_size: int = 0
    persistent_cache_rows: int = 0


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
    prefix_hit_tokens: int
    started_step: int


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
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        if prefix_cache_capacity is not None and prefix_cache_capacity < 0:
            raise ValueError("prefix_cache_capacity must be non-negative")
        self.model = model.to(device).eval()
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.temperature = temperature
        self.max_active_requests = max_active_requests
        self.prefix_cache_capacity = max_active_requests if prefix_cache_capacity is None else prefix_cache_capacity
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes: dict[Hashable, _ReusablePrefix] = {}
        self.stats = ServingStats()
        self._cache: object | None = None
        self._free_active_rows: list[int] = []
        self._free_prefix_rows: list[int] = []
        self._prefix_order: list[Hashable] = []

    @torch.inference_mode()
    def run(self, requests: list[ServingRequest]) -> list[ServingResult]:
        self._reset_run_state(requests)
        waiting = sorted(enumerate(requests), key=lambda item: (item[1].arrival_step, item[0]))
        active: list[_ActiveRequest] = []
        indexed_results: list[tuple[int, ServingResult]] = []
        step = 0
        cursor = 0

        while cursor < len(waiting) or active:
            admitted: list[tuple[int, ServingRequest]] = []
            while (
                cursor < len(waiting)
                and waiting[cursor][1].arrival_step <= step
                and len(active) + len(admitted) < self.max_active_requests
            ):
                admitted.append(waiting[cursor])
                cursor += 1
            if admitted:
                admitted_results, admitted_active = self._prefill_many(admitted, step)
                indexed_results.extend(admitted_results)
                active.extend(admitted_active)

            decoded_results, active = self._decode_active(active, step + 1)
            indexed_results.extend(decoded_results)
            step += 1

            if cursor < len(waiting) and not active and waiting[cursor][1].arrival_step > step:
                step = waiting[cursor][1].arrival_step

        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    def _reset_run_state(self, requests: list[ServingRequest]) -> None:
        self.stats = ServingStats()
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes = {}
        self._prefix_order = []
        max_seq_len = max((len(request.prompt) + request.max_new_tokens for request in requests), default=1)
        total_rows = self.max_active_requests + self.prefix_cache_capacity
        self._cache = self._allocate_cache(max(1, total_rows), max(1, max_seq_len))
        if not hasattr(self._cache, "for_rows"):
            raise ValueError("model cache must support row views for persistent serving")
        self._free_active_rows = list(range(self.max_active_requests))
        self._free_prefix_rows = list(range(self.max_active_requests, total_rows))
        self.stats.persistent_cache_rows = total_rows

    def _prefill_many(
        self,
        indexed_requests: list[tuple[int, ServingRequest]],
        step: int,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        indexed_results: list[tuple[int, ServingResult]] = []
        active: list[_ActiveRequest] = []
        batchable: dict[int, list[tuple[int, ServingRequest, int]]] = defaultdict(list)

        for original_index, request in indexed_requests:
            if not request.prompt:
                raise ValueError("request prompt must contain at least one token")
            match, entry = self.prefix_cache.lookup(request.prompt)
            prefix_hit_tokens = match.depth
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            if request.max_new_tokens == 0:
                indexed_results.append(
                    (
                        original_index,
                        ServingResult(
                            request.request_id,
                            request.prompt,
                            prefix_hit_tokens,
                            request.arrival_step,
                            step,
                            step,
                        ),
                    )
                )
                continue
            if reusable is not None and prefix_hit_tokens > 0:
                active.append(self._prefill_one(original_index, request, step, prefix_hit_tokens, reusable))
            else:
                batchable[len(request.prompt)].append((original_index, request, prefix_hit_tokens))

        for group in batchable.values():
            if len(group) == 1:
                original_index, request, prefix_hit_tokens = group[0]
                active.append(self._prefill_one(original_index, request, step, prefix_hit_tokens, None))
            else:
                active.extend(self._prefill_batch(group, step))
        return indexed_results, active

    def _prefill_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
    ) -> list[_ActiveRequest]:
        prompt_len = len(group[0][1].prompt)
        rows = [self._acquire_active_row() for _ in group]
        prompts = torch.tensor([request.prompt for _, request, _ in group], device=self.device, dtype=torch.long)
        cache_view = self._cache_view(rows)
        logits, _ = self.model(prompts, cache=cache_view, use_cache=True)  # type: ignore[misc]
        self._record_model_call("prefill", len(group))
        next_tokens = sample_next_token(logits[:, -1, :], self.temperature).detach().cpu().tolist()

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            self._store_reusable_prefix(request.request_id, request.prompt, row, logits[row_index : row_index + 1])
            next_token = int(next_tokens[row_index])
            active.append(
                _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=1,
                    row=row,
                    last_token=next_token,
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
            )
        return active

    def _prefill_one(
        self,
        original_index: int,
        request: ServingRequest,
        step: int,
        prefix_hit_tokens: int,
        reusable: _ReusablePrefix | None,
    ) -> _ActiveRequest:
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
            logits, _ = self.model(input_ids, cache=self._cache_view([row]), use_cache=True)  # type: ignore[misc]
            self._record_model_call("prefill", 1)
        elif reusable is not None:
            logits = reusable.logits.to(self.device)
        else:
            raise RuntimeError("empty prompt suffix without a reusable prefix")

        next_token = int(sample_next_token(logits[:, -1, :], self.temperature).item())
        self._store_reusable_prefix(request.request_id, request.prompt, row, logits)
        return _ActiveRequest(
            original_index=original_index,
            request=request,
            tokens=[*request.prompt, next_token],
            generated=1,
            row=row,
            last_token=next_token,
            prefix_hit_tokens=prefix_hit_tokens,
            started_step=step,
        )

    def _decode_active(
        self,
        active: list[_ActiveRequest],
        step: int,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        indexed_results: list[tuple[int, ServingResult]] = []
        live: list[_ActiveRequest] = []
        for state in active:
            if self._should_finish_before_decode(state):
                indexed_results.append((state.original_index, self._finish_and_release(state, step)))
            else:
                live.append(state)

        next_active: list[_ActiveRequest] = []
        for group in self._decode_groups(live):
            decoded = self._decode_batch(group, step) if len(group) > 1 else [self._decode_one(group[0], step)]
            for item, state in zip(decoded, group):
                if isinstance(item, ServingResult):
                    indexed_results.append((state.original_index, item))
                else:
                    next_active.append(item)
        return indexed_results, next_active

    def _decode_groups(self, states: list[_ActiveRequest]) -> list[list[_ActiveRequest]]:
        grouped: dict[int, list[_ActiveRequest]] = defaultdict(list)
        for state in states:
            grouped[self._row_seq_len(state.row)].append(state)
        return list(grouped.values())

    def _decode_batch(self, states: list[_ActiveRequest], step: int) -> list[_ActiveRequest | ServingResult]:
        rows = [state.row for state in states]
        input_ids = torch.tensor([[state.last_token] for state in states], device=self.device, dtype=torch.long)
        logits, _ = self.model(input_ids, cache=self._cache_view(rows), use_cache=True)  # type: ignore[misc]
        self._record_model_call("decode", len(states))
        next_tokens = sample_next_token(logits[:, -1, :], self.temperature).detach().cpu().tolist()

        decoded: list[_ActiveRequest | ServingResult] = []
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            if self._should_finish_after_decode(state):
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        return decoded

    def _decode_one(self, state: _ActiveRequest, step: int) -> _ActiveRequest | ServingResult:
        input_ids = torch.tensor([[state.last_token]], device=self.device, dtype=torch.long)
        logits, _ = self.model(input_ids, cache=self._cache_view([state.row]), use_cache=True)  # type: ignore[misc]
        self._record_model_call("decode", 1)
        next_token = int(sample_next_token(logits[:, -1, :], self.temperature).item())
        state.tokens.append(next_token)
        state.generated += 1
        state.last_token = next_token
        if self._should_finish_after_decode(state):
            return self._finish_and_release(state, step)
        return state

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
        entry = self.prefix_cache.add(request_id, tokens)
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

    def _copy_prefix(self, source_row: int, dest_row: int, tokens: int) -> None:
        cache = self._require_cache()
        cache.copy_prefix_from(cache, tokens, source_row=source_row, dest_row=dest_row)  # type: ignore[attr-defined]

    def _acquire_active_row(self) -> int:
        if not self._free_active_rows:
            raise RuntimeError("no active serving rows available")
        row = self._free_active_rows.pop(0)
        self._clear_physical_row(row)
        return row

    def _release_active_row(self, row: int) -> None:
        self._clear_physical_row(row)
        if row not in self._free_active_rows:
            self._free_active_rows.append(row)
            self._free_active_rows.sort()

    def _acquire_prefix_row(self) -> int | None:
        if self.prefix_cache_capacity == 0:
            return None
        if self._free_prefix_rows:
            return self._free_prefix_rows.pop(0)
        while self._prefix_order:
            route_id = self._prefix_order.pop(0)
            prefix = self.reusable_prefixes.pop(route_id, None)
            if prefix is not None:
                self._clear_physical_row(prefix.row)
                return prefix.row
        return None

    def _cache_view(self, rows: list[int]) -> object:
        return self._require_cache().for_rows(tuple(rows))  # type: ignore[attr-defined]

    def _require_cache(self) -> object:
        if self._cache is None:
            raise RuntimeError("serving cache has not been initialized")
        return self._cache

    def _clear_physical_row(self, row: int) -> None:
        self._require_cache().for_rows((row,)).clear_row(0)  # type: ignore[attr-defined]

    def _row_seq_len(self, row: int) -> int:
        return int(self._cache_view([row]).seq_len)  # type: ignore[attr-defined]

    def _allocate_cache(self, batch_size: int, max_seq_len: int) -> object:
        allocate_cache = getattr(self.model, "allocate_cache")
        try:
            return allocate_cache(
                batch_size,
                max_seq_len=max_seq_len,
                device=self.device,
                cache_backend=self.cache_backend,
                page_size=self.page_size,
            )
        except TypeError:
            if self.cache_backend != "dense":
                raise ValueError(f"model does not support cache_backend={self.cache_backend}") from None
            return allocate_cache(batch_size, max_seq_len=max_seq_len, device=self.device)

    def _record_model_call(self, kind: str, batch_size: int) -> None:
        if kind == "prefill":
            self.stats.prefill_model_calls += 1
            self.stats.prefill_batches += 1
        elif kind == "decode":
            self.stats.decode_model_calls += 1
            self.stats.decode_batches += 1
        else:
            raise ValueError(f"unknown model call kind: {kind}")
        self.stats.max_model_batch_size = max(self.stats.max_model_batch_size, batch_size)

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


def sample_next_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
