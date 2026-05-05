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
