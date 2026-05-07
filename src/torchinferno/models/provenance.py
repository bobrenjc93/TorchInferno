from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariantSpec:
    family: str
    variant: str
    stage: str
    parents: tuple[str, ...]
    module: str
    class_name: str
    ops_module: str
    status: str
    notes: str

    @property
    def id(self) -> str:
        return f"{self.family}:{self.variant}"

    @property
    def class_path(self) -> str:
        return f"{self.module}.{self.class_name}"


class ModelVariantRegistry:
    def __init__(self, specs: tuple[ModelVariantSpec, ...]) -> None:
        self._specs = specs
        self._by_id = {spec.id: spec for spec in specs}
        if len(self._by_id) != len(specs):
            raise ValueError("duplicate model variant ids")

    def list(self, family: str | None = None) -> tuple[ModelVariantSpec, ...]:
        if family is None:
            return self._specs
        return tuple(spec for spec in self._specs if spec.family == family)

    def get(self, family: str, variant: str) -> ModelVariantSpec:
        return self._by_id[f"{family}:{variant}"]

    def lineage(self, family: str, variant: str) -> tuple[ModelVariantSpec, ...]:
        seen: set[str] = set()
        ordered: list[ModelVariantSpec] = []

        def visit(spec: ModelVariantSpec) -> None:
            if spec.id in seen:
                return
            for parent in spec.parents:
                visit(self._by_id[f"{family}:{parent}"])
            seen.add(spec.id)
            ordered.append(spec)

        visit(self.get(family, variant))
        return tuple(ordered)
