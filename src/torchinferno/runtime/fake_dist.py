from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Sequence, TypeVar

import torch
from torch import Tensor


T = TypeVar("T")


@dataclass(frozen=True)
class FakeProcessGroup:
    """Single-process stand-in for one distributed rank."""

    rank: int
    world_size: int
    mesh_shape: tuple[int, ...] = ()

    def coordinate(self) -> tuple[int, ...]:
        if not self.mesh_shape:
            return (self.rank,)
        if self.world_size != _product(self.mesh_shape):
            raise ValueError("mesh_shape product must match world_size")
        remaining = self.rank
        coords = []
        for size in reversed(self.mesh_shape):
            coords.append(remaining % size)
            remaining //= size
        return tuple(reversed(coords))


@dataclass(frozen=True)
class FakeRankResult(Generic[T]):
    rank: int
    result: T


class FakeProcessWorld:
    """Functional fake distributed world for deterministic local tests.

    Collectives accept one tensor per rank and return one result per rank. This
    avoids threads or background rendezvous while preserving the shape of the
    distributed APIs a scheduler or model-parallel policy needs to reason about.
    """

    def __init__(self, world_size: int, *, mesh_shape: Iterable[int] = ()) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        self.world_size = world_size
        self.mesh_shape = tuple(mesh_shape)
        if self.mesh_shape and _product(self.mesh_shape) != world_size:
            raise ValueError("mesh_shape product must match world_size")

    def group(self, rank: int) -> FakeProcessGroup:
        if rank < 0 or rank >= self.world_size:
            raise ValueError("rank out of range")
        return FakeProcessGroup(rank, self.world_size, self.mesh_shape)

    def groups(self) -> tuple[FakeProcessGroup, ...]:
        return tuple(self.group(rank) for rank in range(self.world_size))

    def run(self, fn: Callable[[FakeProcessGroup], T]) -> list[FakeRankResult[T]]:
        return [FakeRankResult(group.rank, fn(group)) for group in self.groups()]

    def all_reduce(self, tensors: Sequence[Tensor], *, op: str = "sum") -> tuple[Tensor, ...]:
        self._check_tensors(tensors)
        stacked = torch.stack(tuple(tensors), dim=0)
        if op == "sum":
            reduced = stacked.sum(dim=0)
        elif op == "mean":
            reduced = stacked.mean(dim=0)
        elif op == "max":
            reduced = stacked.max(dim=0).values
        elif op == "min":
            reduced = stacked.min(dim=0).values
        else:
            raise ValueError(f"unsupported all_reduce op: {op}")
        return tuple(reduced.clone() for _ in range(self.world_size))

    def all_gather(self, tensors: Sequence[Tensor]) -> tuple[tuple[Tensor, ...], ...]:
        self._check_tensors(tensors)
        gathered = tuple(tensor.clone() for tensor in tensors)
        return tuple(gathered for _ in range(self.world_size))

    def broadcast(self, tensors: Sequence[Tensor], *, src: int = 0) -> tuple[Tensor, ...]:
        self._check_tensors(tensors)
        if src < 0 or src >= self.world_size:
            raise ValueError("src rank out of range")
        return tuple(tensors[src].clone() for _ in range(self.world_size))

    def reduce_scatter(self, tensors: Sequence[Tensor], *, op: str = "sum", dim: int = 0) -> tuple[Tensor, ...]:
        reduced = self.all_reduce(tensors, op=op)[0]
        chunks = torch.chunk(reduced, self.world_size, dim=dim)
        if len(chunks) != self.world_size:
            raise ValueError("reduced tensor cannot be evenly scattered across ranks")
        return tuple(chunk.contiguous() for chunk in chunks)

    def _check_tensors(self, tensors: Sequence[Tensor]) -> None:
        if len(tensors) != self.world_size:
            raise ValueError(f"expected {self.world_size} tensors, got {len(tensors)}")
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        for tensor in tensors:
            if tensor.shape != shape:
                raise ValueError("all fake collective tensors must have the same shape")
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError("all fake collective tensors must have the same dtype and device")


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
