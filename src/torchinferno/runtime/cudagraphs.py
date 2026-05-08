from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class CUDAGraphPiece:
    name: str
    fn: Callable[..., object]
    warmup_iters: int = 1


@dataclass
class _CapturedPiece:
    piece: CUDAGraphPiece
    graph: torch.cuda.CUDAGraph
    static_args: tuple[object, ...]
    static_kwargs: dict[str, object]
    output: object


class PiecewiseCUDAGraphRunner:
    """Named execution pieces with an eager fallback.

    Actual CUDA graph capture needs static buffers and device-specific policy.
    This runner gives TorchInferno a stable API for carving prefill/decode/model
    fragments today while keeping CPU and fake-device tests runnable.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._pieces: dict[str, CUDAGraphPiece] = {}
        self._captures: dict[str, _CapturedPiece] = {}

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
        piece = self._pieces[name]
        if not self.capture_available or not _capture_supported(args, kwargs):
            return piece.fn(*args, **kwargs)
        captured = self._captures.get(name)
        if captured is None or not _static_inputs_match(args, kwargs, captured.static_args, captured.static_kwargs):
            captured = self._capture(piece, args, kwargs)
            self._captures[name] = captured
        else:
            _copy_inputs(args, kwargs, captured.static_args, captured.static_kwargs)
        captured.graph.replay()
        return _clone_output(captured.output)

    def names(self) -> tuple[str, ...]:
        return tuple(self._pieces)

    def _capture(self, piece: CUDAGraphPiece, args: tuple[object, ...], kwargs: dict[str, object]) -> _CapturedPiece:
        static_args = _clone_inputs(args)
        static_kwargs = dict(zip(kwargs, _clone_inputs(tuple(kwargs.values()))))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(0, piece.warmup_iters)):
                piece.fn(*static_args, **static_kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = piece.fn(*static_args, **static_kwargs)
        return _CapturedPiece(piece, graph, static_args, static_kwargs, output)


def _capture_supported(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
    values = (*args, *kwargs.values())
    return bool(values) and all(isinstance(value, torch.Tensor) and value.is_cuda for value in values)


def _clone_inputs(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value for value in values)


def _copy_inputs(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    static_args: tuple[object, ...],
    static_kwargs: dict[str, object],
) -> None:
    for source, target in zip(args, static_args):
        if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
            target.copy_(source)
    for key, source in kwargs.items():
        target = static_kwargs[key]
        if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
            target.copy_(source)


def _static_inputs_match(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    static_args: tuple[object, ...],
    static_kwargs: dict[str, object],
) -> bool:
    if len(args) != len(static_args) or set(kwargs) != set(static_kwargs):
        return False
    for source, target in zip(args, static_args):
        if not _same_tensor_contract(source, target):
            return False
    for key, source in kwargs.items():
        if not _same_tensor_contract(source, static_kwargs[key]):
            return False
    return True


def _same_tensor_contract(source: object, target: object) -> bool:
    return (
        isinstance(source, torch.Tensor)
        and isinstance(target, torch.Tensor)
        and source.shape == target.shape
        and source.dtype == target.dtype
        and source.device == target.device
    )


def _clone_output(output: object) -> object:
    if isinstance(output, torch.Tensor):
        return output.clone()
    if isinstance(output, tuple):
        return tuple(_clone_output(item) for item in output)
    if isinstance(output, list):
        return [_clone_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _clone_output(value) for key, value in output.items()}
    return output
