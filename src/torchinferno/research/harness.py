from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    metrics: Mapping[str, float]
    artifacts: Mapping[str, object] = field(default_factory=dict)


ExperimentFn = Callable[[], ExperimentResult]


class ResearchHarness:
    """Minimal auto research harness for repeatable policy experiments."""

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentFn] = {}

    def register(self, name: str, fn: ExperimentFn) -> None:
        if name in self._experiments:
            raise ValueError(f"duplicate experiment: {name}")
        self._experiments[name] = fn

    def run(self, names: Iterable[str] | None = None) -> list[ExperimentResult]:
        selected = tuple(self._experiments) if names is None else tuple(names)
        results = []
        for name in selected:
            if name not in self._experiments:
                raise KeyError(name)
            result = self._experiments[name]()
            if result.name != name:
                raise ValueError(f"experiment {name} returned result named {result.name}")
            results.append(result)
        return results

    @staticmethod
    def best(results: Iterable[ExperimentResult], metric: str, *, higher_is_better: bool = False) -> ExperimentResult:
        result_list = list(results)
        if not result_list:
            raise ValueError("no experiment results")
        return sorted(result_list, key=lambda result: result.metrics[metric], reverse=higher_is_better)[0]
