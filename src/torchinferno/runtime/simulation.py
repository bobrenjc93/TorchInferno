from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class VirtualGPU:
    rank: int
    latency_us: float = 0.0
    time_slice_us: float = 1000.0


@dataclass(frozen=True)
class SimulationEvent(Generic[T]):
    rank: int
    elapsed_us: float
    result: T


@dataclass(frozen=True)
class TimeSliceWorkload:
    rank: int
    work_us: float
    label: str = ""
    arrival_us: float = 0.0


@dataclass(frozen=True)
class TimeSliceReplayEvent:
    rank: int
    label: str
    slice_index: int
    start_us: float
    end_us: float
    work_us: float
    overhead_us: float
    remaining_us: float


@dataclass(frozen=True)
class TimeSliceReplayResult:
    time_slice_us: float
    total_elapsed_us: float
    total_work_us: float
    total_overhead_us: float
    utilization: float
    events: tuple[TimeSliceReplayEvent, ...]


class TimeSlicedSimulator:
    """Deterministic single-process simulator for multi-GPU inference flows."""

    def __init__(self, virtual_gpus: Iterable[VirtualGPU]) -> None:
        self.virtual_gpus = tuple(virtual_gpus)
        if not self.virtual_gpus:
            raise ValueError("at least one virtual GPU is required")

    def run(self, fn: Callable[[VirtualGPU], T]) -> list[SimulationEvent[T]]:
        events: list[SimulationEvent[T]] = []
        for gpu in self.virtual_gpus:
            start = time.perf_counter()
            result = fn(gpu)
            elapsed = (time.perf_counter() - start) * 1_000_000.0
            events.append(SimulationEvent(gpu.rank, elapsed + gpu.latency_us + gpu.time_slice_us, result))
        return events

    def replay(self, workloads: Iterable[TimeSliceWorkload]) -> TimeSliceReplayResult:
        """Replay measured rank work with deterministic round-robin time slicing.

        The replay is a scheduler model: it does not preempt a live CUDA kernel.
        Instead, callers pass measured or estimated work in microseconds, and
        this method emits the virtual execution timeline for a single physical
        device shared by multiple ranks.
        """

        pending = [
            _RemainingWork(
                workload.rank,
                max(0.0, workload.work_us),
                workload.label or f"rank-{workload.rank}",
                max(0.0, workload.arrival_us),
                0,
            )
            for workload in workloads
        ]
        if not pending:
            raise ValueError("at least one workload is required")
        gpus = {gpu.rank: gpu for gpu in self.virtual_gpus}
        unknown = sorted({work.rank for work in pending if work.rank not in gpus})
        if unknown:
            raise ValueError(f"workload ranks are not in virtual_gpus: {unknown}")
        for gpu in self.virtual_gpus:
            if gpu.time_slice_us <= 0:
                raise ValueError("time_slice_us must be positive")

        pending.sort(key=lambda work: (work.arrival_us, work.rank))
        active: list[_RemainingWork] = []
        events: list[TimeSliceReplayEvent] = []
        clock_us = 0.0
        total_work_us = 0.0
        total_overhead_us = 0.0

        while pending or active:
            while pending and pending[0].arrival_us <= clock_us:
                active.append(pending.pop(0))
            if not active:
                clock_us = max(clock_us, pending[0].arrival_us)
                continue

            work = active.pop(0)
            gpu = gpus[work.rank]
            if work.remaining_us <= 0:
                continue
            start_us = clock_us + gpu.latency_us
            slice_work_us = min(work.remaining_us, gpu.time_slice_us)
            end_us = start_us + slice_work_us
            remaining_us = work.remaining_us - slice_work_us
            events.append(
                TimeSliceReplayEvent(
                    rank=work.rank,
                    label=work.label,
                    slice_index=work.slice_index,
                    start_us=start_us,
                    end_us=end_us,
                    work_us=slice_work_us,
                    overhead_us=gpu.latency_us,
                    remaining_us=remaining_us,
                )
            )
            total_work_us += slice_work_us
            total_overhead_us += gpu.latency_us
            clock_us = end_us
            if remaining_us > 0:
                active.append(
                    _RemainingWork(
                        work.rank,
                        remaining_us,
                        work.label,
                        work.arrival_us,
                        work.slice_index + 1,
                    )
                )

        utilization = 0.0 if clock_us <= 0 else total_work_us / clock_us
        min_slice = min(gpu.time_slice_us for gpu in self.virtual_gpus)
        return TimeSliceReplayResult(
            time_slice_us=min_slice,
            total_elapsed_us=clock_us,
            total_work_us=total_work_us,
            total_overhead_us=total_overhead_us,
            utilization=utilization,
            events=tuple(events),
        )


@dataclass(frozen=True)
class _RemainingWork:
    rank: int
    remaining_us: float
    label: str
    arrival_us: float
    slice_index: int
