from __future__ import annotations

import random
from dataclasses import dataclass

from torchinferno.runtime.scheduler import DisaggregatedPrefillDecodeSimulator, InferenceJob, ScheduledStage


@dataclass(frozen=True)
class TrafficPattern:
    requests: int
    prompt_min: int = 1
    prompt_max: int = 64
    decode_min: int = 1
    decode_max: int = 16
    burst_size: int = 4
    burst_gap_us: float = 1000.0
    in_burst_gap_us: float = 10.0
    seed: int = 0


def generate_traffic(pattern: TrafficPattern) -> list[InferenceJob]:
    rng = random.Random(pattern.seed)
    jobs: list[InferenceJob] = []
    for index in range(pattern.requests):
        burst = index // pattern.burst_size
        within_burst = index % pattern.burst_size
        arrival = burst * pattern.burst_gap_us + within_burst * pattern.in_burst_gap_us
        jobs.append(
            InferenceJob(
                request_id=f"req-{index}",
                prompt_tokens=rng.randint(pattern.prompt_min, pattern.prompt_max),
                decode_tokens=rng.randint(pattern.decode_min, pattern.decode_max),
                arrival_us=arrival,
            )
        )
    return jobs


@dataclass(frozen=True)
class TrafficSimulationResult:
    jobs: tuple[InferenceJob, ...]
    stages: tuple[ScheduledStage, ...]

    @property
    def makespan_us(self) -> float:
        return max((stage.end_us for stage in self.stages), default=0.0)

    @property
    def requests_per_second(self) -> float:
        if self.makespan_us <= 0:
            return 0.0
        return len(self.jobs) / (self.makespan_us / 1_000_000.0)


def simulate_traffic(
    pattern: TrafficPattern,
    scheduler: DisaggregatedPrefillDecodeSimulator,
) -> TrafficSimulationResult:
    jobs = tuple(generate_traffic(pattern))
    stages = tuple(scheduler.plan(jobs))
    return TrafficSimulationResult(jobs, stages)
