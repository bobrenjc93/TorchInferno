from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


StageName = Literal["prefill", "decode"]


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
