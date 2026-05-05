from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    mean_ms: float
    iters: int
    device: str


def benchmark_callable(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int = 5,
    iters: int = 20,
    device: torch.device | None = None,
) -> BenchmarkResult:
    if iters < 1:
        raise ValueError("iters must be positive")
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    mean_ms = (time.perf_counter() - start) * 1000.0 / iters
    return BenchmarkResult(name=name, mean_ms=mean_ms, iters=iters, device=str(device or "cpu"))


def _sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
