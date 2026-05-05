from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class CUDAGraphPiece:
    name: str
    fn: Callable[..., object]
    warmup_iters: int = 1


class PiecewiseCUDAGraphRunner:
    """Named execution pieces with an eager fallback.

    Actual CUDA graph capture needs static buffers and device-specific policy.
    This runner gives TorchInferno a stable API for carving prefill/decode/model
    fragments today while keeping CPU and fake-device tests runnable.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._pieces: dict[str, CUDAGraphPiece] = {}

    @property
    def capture_available(self) -> bool:
        return self.enabled and torch.cuda.is_available()

    def register(self, piece: CUDAGraphPiece) -> None:
        if piece.name in self._pieces:
            raise ValueError(f"duplicate CUDA graph piece: {piece.name}")
        self._pieces[piece.name] = piece

    def run(self, name: str, *args: object, **kwargs: object) -> object:
        if name not in self._pieces:
            raise KeyError(name)
        return self._pieces[name].fn(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(self._pieces)
