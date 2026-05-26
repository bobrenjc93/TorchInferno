from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal


StageName = Literal["prefill", "decode"]
TokenBudgetChunkKind = Literal["prefill", "decode"]


@dataclass(frozen=True)
class InferenceJob:
    request_id: str
    prompt_tokens: int
    decode_tokens: int
    arrival_us: float = 0.0


@dataclass(frozen=True)
class ScheduledStage:
    request_id: str
    stage: StageName
    rank: int
    start_us: float
    end_us: float

    @property
    def elapsed_us(self) -> float:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class PersistentBatchRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_step: int = 0
    prefix_hit_tokens: int = 0
    prefix_key: Hashable | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")
        if self.prefix_hit_tokens < 0:
            raise ValueError("prefix_hit_tokens must be non-negative")
        if self.prefix_hit_tokens > self.prompt_tokens:
            raise ValueError("prefix_hit_tokens cannot exceed prompt_tokens")

    @property
    def prefill_tokens(self) -> int:
        return max(1, self.prompt_tokens - self.prefix_hit_tokens)


@dataclass(frozen=True)
class PersistentPrefillAdmission:
    request_id: str
    row: int
    prompt_tokens: int
    prefix_hit_tokens: int
    prefill_tokens: int
    max_new_tokens: int
    prefix_key: Hashable | None = None


@dataclass(frozen=True)
class PersistentPrefillGroup:
    prefix_key: Hashable | None
    prefix_hit_tokens: int
    request_ids: tuple[str, ...]
    rows: tuple[int, ...]
    suffix_tokens: tuple[int, ...]


@dataclass(frozen=True)
class PersistentBatchPlan:
    step: int
    decode_request_ids: tuple[str, ...]
    decode_rows: tuple[int, ...]
    prefill_admissions: tuple[PersistentPrefillAdmission, ...]
    prefill_groups: tuple[PersistentPrefillGroup, ...]
    finished_after_prefill: tuple[str, ...]


@dataclass(frozen=True)
class _QueuedPersistentRequest:
    request: PersistentBatchRequest
    sequence: int


@dataclass(frozen=True)
class _ActivePersistentRequest:
    request: PersistentBatchRequest
    row: int


class PersistentBatchScheduler:
    """CPU-only planner for persistent row admission and refill experiments."""

    def __init__(self, *, max_rows: int, prefill_token_budget: int | None = None) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        if prefill_token_budget is not None and prefill_token_budget < 1:
            raise ValueError("prefill_token_budget must be positive")
        self.max_rows = max_rows
        self.prefill_token_budget = prefill_token_budget
        self._waiting: list[_QueuedPersistentRequest] = []
        self._active: dict[str, _ActivePersistentRequest] = {}
        self._free_rows = list(range(max_rows))
        self._next_sequence = 0
        self._step = 0

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def active_rows(self) -> tuple[int, ...]:
        return tuple(sorted(item.row for item in self._active.values()))

    def submit(self, request: PersistentBatchRequest) -> None:
        if request.request_id in self._active or any(
            item.request.request_id == request.request_id for item in self._waiting
        ):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._waiting.append(_QueuedPersistentRequest(request, self._next_sequence))
        self._next_sequence += 1
        self._waiting.sort(key=self._arrival_key)

    def has_work(self) -> bool:
        return bool(self._waiting) or bool(self._active)

    def step(self, *, finished_request_ids: Iterable[str] = ()) -> PersistentBatchPlan:
        for request_id in finished_request_ids:
            self._release(request_id)
        decode_active = sorted(self._active.values(), key=lambda item: item.row)
        admissions = self._admit_ready()
        finished_after_prefill: list[str] = []
        for admission in admissions:
            if admission.max_new_tokens <= 1:
                self._release(admission.request_id)
                finished_after_prefill.append(admission.request_id)
        plan = PersistentBatchPlan(
            step=self._step,
            decode_request_ids=tuple(item.request.request_id for item in decode_active),
            decode_rows=tuple(item.row for item in decode_active),
            prefill_admissions=tuple(admissions),
            prefill_groups=_group_prefill_admissions(admissions),
            finished_after_prefill=tuple(finished_after_prefill),
        )
        self._step += 1
        return plan

    def _admit_ready(self) -> list[PersistentPrefillAdmission]:
        capacity = min(len(self._free_rows), self.max_rows - len(self._active))
        if capacity <= 0:
            return []
        ready: list[_QueuedPersistentRequest] = []
        waiting: list[_QueuedPersistentRequest] = []
        for item in self._waiting:
            if item.request.arrival_step <= self._step:
                ready.append(item)
            else:
                waiting.append(item)
        if not ready:
            self._waiting = waiting
            return []
        ready.sort(key=self._admission_key)
        selected: list[_QueuedPersistentRequest] = []
        deferred: list[_QueuedPersistentRequest] = []
        remaining_budget = self.prefill_token_budget
        for item in ready:
            if len(selected) >= capacity:
                deferred.append(item)
                continue
            cost = item.request.prefill_tokens
            if remaining_budget is not None and selected and cost > remaining_budget:
                deferred.append(item)
                continue
            selected.append(item)
            if remaining_budget is not None:
                remaining_budget -= cost
        self._waiting = [*deferred, *waiting]
        self._waiting.sort(key=self._arrival_key)
        admissions: list[PersistentPrefillAdmission] = []
        for item in selected:
            row = self._free_rows.pop(0)
            request = item.request
            self._active[request.request_id] = _ActivePersistentRequest(request, row)
            admissions.append(
                PersistentPrefillAdmission(
                    request_id=request.request_id,
                    row=row,
                    prompt_tokens=request.prompt_tokens,
                    prefix_hit_tokens=request.prefix_hit_tokens,
                    prefill_tokens=request.prefill_tokens,
                    max_new_tokens=request.max_new_tokens,
                    prefix_key=request.prefix_key,
                )
            )
        return admissions

    def _release(self, request_id: str) -> None:
        active = self._active.pop(request_id, None)
        if active is None:
            return
        self._free_rows.append(active.row)
        self._free_rows.sort()

    @staticmethod
    def _arrival_key(item: _QueuedPersistentRequest) -> tuple[int, int]:
        return (item.request.arrival_step, item.sequence)

    @staticmethod
    def _admission_key(item: _QueuedPersistentRequest) -> tuple[int, int, int]:
        return (-item.request.prefix_hit_tokens, item.request.arrival_step, item.sequence)


@dataclass(frozen=True)
class TokenBudgetRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_step: int = 0
    prefix_hit_tokens: int = 0
    prefix_key: Hashable | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")
        if self.prefix_hit_tokens < 0:
            raise ValueError("prefix_hit_tokens must be non-negative")
        if self.prefix_hit_tokens > self.prompt_tokens:
            raise ValueError("prefix_hit_tokens cannot exceed prompt_tokens")


@dataclass(frozen=True)
class TokenBudgetScheduledChunk:
    request_id: str
    row: int
    kind: TokenBudgetChunkKind
    start_token: int
    token_count: int
    prompt_complete: bool = False
    emits_token: bool = False
    prefix_key: Hashable | None = None


@dataclass(frozen=True)
class TokenBudgetPlan:
    step: int
    chunks: tuple[TokenBudgetScheduledChunk, ...]
    finished_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class TokenBudgetModelStepCommand:
    """Immutable model-work transcript for one token-budget scheduler step."""

    step: int
    chunks: tuple[TokenBudgetScheduledChunk, ...]
    decode_rows: tuple[int, ...]
    prefill_rows: tuple[int, ...]
    emit_request_ids: tuple[str, ...]
    emit_rows: tuple[int, ...]
    finished_request_ids: tuple[str, ...]
    scheduled_tokens: int

    @property
    def is_empty(self) -> bool:
        return not self.chunks


@dataclass
class TokenBudgetModelStepState:
    """CPU row-state mirror for token-budget command execution tests."""

    row_request_ids: list[str | None]
    computed_tokens: list[int]
    generated_tokens: list[int]
    prompt_complete: list[bool]

    @classmethod
    def empty(cls, max_rows: int) -> "TokenBudgetModelStepState":
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        return cls(
            row_request_ids=[None for _ in range(max_rows)],
            computed_tokens=[0 for _ in range(max_rows)],
            generated_tokens=[0 for _ in range(max_rows)],
            prompt_complete=[False for _ in range(max_rows)],
        )


@dataclass(frozen=True)
class TokenBudgetModelStepResult:
    emitted_request_ids: tuple[str, ...]
    emitted_rows: tuple[int, ...]
    finished_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class _QueuedTokenBudgetRequest:
    request: TokenBudgetRequest
    sequence: int


@dataclass
class _ActiveTokenBudgetRequest:
    request: TokenBudgetRequest
    row: int
    computed_tokens: int
    generated_tokens: int = 0


def token_budget_model_step_command(plan: TokenBudgetPlan) -> TokenBudgetModelStepCommand:
    """Normalize a token-budget plan into one rank-lockstep model command."""

    rows_seen: set[int] = set()
    request_ids_seen: set[str] = set()
    decode_rows: list[int] = []
    prefill_rows: list[int] = []
    emit_request_ids: list[str] = []
    emit_rows: list[int] = []
    scheduled_tokens = 0
    for chunk in plan.chunks:
        if chunk.row in rows_seen:
            raise ValueError("token-budget model command cannot reuse a row in one step")
        rows_seen.add(chunk.row)
        if chunk.request_id in request_ids_seen:
            raise ValueError("token-budget model command cannot schedule one request twice in one step")
        request_ids_seen.add(chunk.request_id)
        if chunk.token_count < 1:
            raise ValueError("token-budget model command chunks must schedule positive tokens")
        if chunk.kind == "decode":
            if chunk.token_count != 1:
                raise ValueError("token-budget decode chunks must schedule exactly one token")
            if not chunk.prompt_complete or not chunk.emits_token:
                raise ValueError("token-budget decode chunks must be prompt-complete and emit")
            decode_rows.append(chunk.row)
        elif chunk.kind == "prefill":
            if chunk.emits_token and not chunk.prompt_complete:
                raise ValueError("token-budget prefill chunks cannot emit before prompt completion")
            prefill_rows.append(chunk.row)
        else:
            raise ValueError(f"unsupported token-budget chunk kind: {chunk.kind}")
        scheduled_tokens += chunk.token_count
        if chunk.emits_token:
            emit_request_ids.append(chunk.request_id)
            emit_rows.append(chunk.row)
    return TokenBudgetModelStepCommand(
        step=plan.step,
        chunks=plan.chunks,
        decode_rows=tuple(decode_rows),
        prefill_rows=tuple(prefill_rows),
        emit_request_ids=tuple(emit_request_ids),
        emit_rows=tuple(emit_rows),
        finished_request_ids=plan.finished_request_ids,
        scheduled_tokens=scheduled_tokens,
    )


def apply_token_budget_model_step_command(
    state: TokenBudgetModelStepState,
    command: TokenBudgetModelStepCommand,
) -> TokenBudgetModelStepResult:
    """Apply one normalized command to a CPU row-state mirror."""

    _validate_token_budget_step_state(state)
    for chunk in command.chunks:
        row = chunk.row
        if row < 0 or row >= len(state.row_request_ids):
            raise ValueError("token-budget command row is out of range")
        current_request_id = state.row_request_ids[row]
        if current_request_id is None:
            if chunk.kind == "decode":
                raise ValueError("token-budget decode chunk requires an occupied row")
            state.row_request_ids[row] = chunk.request_id
            state.computed_tokens[row] = chunk.start_token
            state.generated_tokens[row] = 0
            state.prompt_complete[row] = False
        elif current_request_id != chunk.request_id:
            raise ValueError("token-budget command row is occupied by a different request")

        if state.computed_tokens[row] != chunk.start_token:
            raise ValueError("token-budget chunk start_token does not match row state")
        if chunk.kind == "decode" and not state.prompt_complete[row]:
            raise ValueError("token-budget decode chunk requires prompt-complete row state")

        state.computed_tokens[row] += chunk.token_count
        if chunk.prompt_complete:
            state.prompt_complete[row] = True
        if chunk.emits_token:
            state.generated_tokens[row] += 1

    active_by_request: dict[str, int] = {}
    for row, request_id in enumerate(state.row_request_ids):
        if request_id is not None:
            active_by_request[request_id] = row
    for request_id in command.finished_request_ids:
        row = active_by_request.get(request_id)
        if row is None:
            raise ValueError("token-budget command finished an unknown request")
        state.row_request_ids[row] = None
        state.computed_tokens[row] = 0
        state.generated_tokens[row] = 0
        state.prompt_complete[row] = False

    return TokenBudgetModelStepResult(
        emitted_request_ids=command.emit_request_ids,
        emitted_rows=command.emit_rows,
        finished_request_ids=command.finished_request_ids,
    )


def _validate_token_budget_step_state(state: TokenBudgetModelStepState) -> None:
    rows = len(state.row_request_ids)
    if not (
        len(state.computed_tokens)
        == rows
        == len(state.generated_tokens)
        == len(state.prompt_complete)
    ):
        raise ValueError("token-budget step state row fields must have matching lengths")


class TokenBudgetScheduler:
    """CPU planner for vLLM-style token-budget continuous batching.

    The planner does not execute a model. It tracks per-request computed-token
    progress so future runners can schedule mixed decode and chunked prefill
    work under one token budget while preserving persistent row assignment.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        max_scheduled_tokens: int,
        prefill_chunk_size: int | None = None,
        decode_first: bool = True,
    ) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        if max_scheduled_tokens < 1:
            raise ValueError("max_scheduled_tokens must be positive")
        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be positive")
        self.max_rows = max_rows
        self.max_scheduled_tokens = max_scheduled_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_first = decode_first
        self._waiting: list[_QueuedTokenBudgetRequest] = []
        self._active: dict[str, _ActiveTokenBudgetRequest] = {}
        self._free_rows = list(range(max_rows))
        self._next_sequence = 0
        self._step = 0

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def active_rows(self) -> tuple[int, ...]:
        return tuple(sorted(item.row for item in self._active.values()))

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    def submit(self, request: TokenBudgetRequest) -> None:
        if request.request_id in self._active or any(
            item.request.request_id == request.request_id for item in self._waiting
        ):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._waiting.append(_QueuedTokenBudgetRequest(request, self._next_sequence))
        self._next_sequence += 1
        self._waiting.sort(key=self._arrival_key)

    def has_work(self) -> bool:
        return bool(self._waiting) or bool(self._active)

    def step(self, *, finished_request_ids: Iterable[str] = ()) -> TokenBudgetPlan:
        for request_id in finished_request_ids:
            self._release(request_id)
        budget = self.max_scheduled_tokens
        chunks: list[TokenBudgetScheduledChunk] = []
        finished: list[str] = []
        if self.decode_first:
            budget = self._schedule_active(chunks, finished, budget)
            budget = self._admit_and_schedule(chunks, finished, budget)
        else:
            budget = self._admit_and_schedule(chunks, finished, budget)
            budget = self._schedule_active(chunks, finished, budget)
        plan = TokenBudgetPlan(
            step=self._step,
            chunks=tuple(chunks),
            finished_request_ids=tuple(finished),
        )
        for request_id in finished:
            self._release(request_id)
        self._step += 1
        return plan

    def _schedule_active(
        self,
        chunks: list[TokenBudgetScheduledChunk],
        finished: list[str],
        budget: int,
    ) -> int:
        for active in sorted(list(self._active.values()), key=lambda item: item.row):
            if budget <= 0:
                break
            budget = self._schedule_one(active, chunks, finished, budget)
        return budget

    def _admit_and_schedule(
        self,
        chunks: list[TokenBudgetScheduledChunk],
        finished: list[str],
        budget: int,
    ) -> int:
        while budget > 0 and self._free_rows:
            item = self._pop_next_ready()
            if item is None:
                break
            request = item.request
            row = self._free_rows.pop(0)
            active = _ActiveTokenBudgetRequest(
                request=request,
                row=row,
                computed_tokens=request.prefix_hit_tokens,
            )
            self._active[request.request_id] = active
            budget = self._schedule_one(active, chunks, finished, budget)
        return budget

    def _schedule_one(
        self,
        active: _ActiveTokenBudgetRequest,
        chunks: list[TokenBudgetScheduledChunk],
        finished: list[str],
        budget: int,
    ) -> int:
        request = active.request
        if active.computed_tokens < request.prompt_tokens:
            remaining = request.prompt_tokens - active.computed_tokens
            if self.prefill_chunk_size is not None:
                remaining = min(remaining, self.prefill_chunk_size)
            token_count = min(remaining, budget)
            if token_count <= 0:
                return budget
            start_token = active.computed_tokens
            active.computed_tokens += token_count
            prompt_complete = active.computed_tokens >= request.prompt_tokens
            emits_token = prompt_complete
            if emits_token:
                active.generated_tokens += 1
            chunks.append(
                TokenBudgetScheduledChunk(
                    request_id=request.request_id,
                    row=active.row,
                    kind="prefill",
                    start_token=start_token,
                    token_count=token_count,
                    prompt_complete=prompt_complete,
                    emits_token=emits_token,
                    prefix_key=request.prefix_key,
                )
            )
            budget -= token_count
            if active.generated_tokens >= request.max_new_tokens:
                finished.append(request.request_id)
            return budget

        if active.generated_tokens >= request.max_new_tokens:
            finished.append(request.request_id)
            return budget

        start_token = request.prompt_tokens + max(0, active.generated_tokens - 1)
        active.generated_tokens += 1
        active.computed_tokens += 1
        chunks.append(
            TokenBudgetScheduledChunk(
                request_id=request.request_id,
                row=active.row,
                kind="decode",
                start_token=start_token,
                token_count=1,
                prompt_complete=True,
                emits_token=True,
                prefix_key=request.prefix_key,
            )
        )
        budget -= 1
        if active.generated_tokens >= request.max_new_tokens:
            finished.append(request.request_id)
        return budget

    def _pop_next_ready(self) -> _QueuedTokenBudgetRequest | None:
        ready: list[_QueuedTokenBudgetRequest] = []
        waiting: list[_QueuedTokenBudgetRequest] = []
        for item in self._waiting:
            if item.request.arrival_step <= self._step:
                ready.append(item)
            else:
                waiting.append(item)
        if not ready:
            self._waiting = waiting
            return None
        ready.sort(key=self._admission_key)
        selected = ready.pop(0)
        self._waiting = [*ready, *waiting]
        self._waiting.sort(key=self._arrival_key)
        return selected

    def _release(self, request_id: str) -> None:
        active = self._active.pop(request_id, None)
        if active is None:
            return
        if active.row not in self._free_rows:
            self._free_rows.append(active.row)
            self._free_rows.sort()

    @staticmethod
    def _arrival_key(item: _QueuedTokenBudgetRequest) -> tuple[int, int]:
        return (item.request.arrival_step, item.sequence)

    @staticmethod
    def _admission_key(item: _QueuedTokenBudgetRequest) -> tuple[int, int, int]:
        return (-item.request.prefix_hit_tokens, item.request.arrival_step, item.sequence)


def _group_prefill_admissions(
    admissions: Iterable[PersistentPrefillAdmission],
) -> tuple[PersistentPrefillGroup, ...]:
    groups: dict[tuple[Hashable | None, int], list[PersistentPrefillAdmission]] = {}
    for admission in admissions:
        key = (admission.prefix_key, admission.prefix_hit_tokens)
        groups.setdefault(key, []).append(admission)
    return tuple(
        PersistentPrefillGroup(
            prefix_key=prefix_key,
            prefix_hit_tokens=prefix_hit_tokens,
            request_ids=tuple(item.request_id for item in group),
            rows=tuple(item.row for item in group),
            suffix_tokens=tuple(max(1, item.prompt_tokens - item.prefix_hit_tokens) for item in group),
        )
        for (prefix_key, prefix_hit_tokens), group in groups.items()
    )


class DisaggregatedPrefillDecodeSimulator:
    """Single-process planner for disaggregated prefill/decode experiments."""

    def __init__(
        self,
        *,
        prefill_ranks: Iterable[int],
        decode_ranks: Iterable[int],
        prefill_us_per_token: float = 1.0,
        decode_us_per_token: float = 1.0,
        network_latency_us: float = 0.0,
    ) -> None:
        self.prefill_ranks = tuple(prefill_ranks)
        self.decode_ranks = tuple(decode_ranks)
        if not self.prefill_ranks or not self.decode_ranks:
            raise ValueError("prefill_ranks and decode_ranks must be non-empty")
        self.prefill_us_per_token = prefill_us_per_token
        self.decode_us_per_token = decode_us_per_token
        self.network_latency_us = network_latency_us

    def plan(self, jobs: Iterable[InferenceJob]) -> list[ScheduledStage]:
        prefill_available = {rank: 0.0 for rank in self.prefill_ranks}
        decode_available = {rank: 0.0 for rank in self.decode_ranks}
        stages: list[ScheduledStage] = []
        for job in jobs:
            prefill_rank = min(prefill_available, key=prefill_available.get)
            prefill_start = max(job.arrival_us, prefill_available[prefill_rank])
            prefill_end = prefill_start + max(1, job.prompt_tokens) * self.prefill_us_per_token
            prefill_available[prefill_rank] = prefill_end
            stages.append(ScheduledStage(job.request_id, "prefill", prefill_rank, prefill_start, prefill_end))

            decode_rank = min(decode_available, key=decode_available.get)
            decode_start = max(prefill_end + self.network_latency_us, decode_available[decode_rank])
            decode_end = decode_start + max(1, job.decode_tokens) * self.decode_us_per_token
            decode_available[decode_rank] = decode_end
            stages.append(ScheduledStage(job.request_id, "decode", decode_rank, decode_start, decode_end))
        return sorted(stages, key=lambda stage: (stage.start_us, stage.request_id, stage.stage))
