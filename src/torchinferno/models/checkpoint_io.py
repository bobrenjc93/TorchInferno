from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from torchinferno.models.hf import SAFETENSORS_INDEX_NAME, SAFETENSORS_NAME


class CheckpointTensorLoader:
    """Lazy, sliced safetensors reader shared by production model loaders."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        index_path = self.root / SAFETENSORS_INDEX_NAME
        single_path = self.root / SAFETENSORS_NAME
        if index_path.exists():
            if not index_path.resolve().is_relative_to(self.root):
                raise ValueError("checkpoint index escapes checkpoint root")
            payload = json.loads(index_path.read_text())
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid safetensors weight_map in {index_path}")
            self.weight_map = {}
            for name, filename in weight_map.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"invalid tensor name in {index_path}")
                if not isinstance(filename, str):
                    raise ValueError(f"invalid checkpoint shard for {name!r} in {index_path}")
                self._shard_path(filename)
                self.weight_map[name] = filename
        elif single_path.exists():
            single_path = self._shard_path(single_path.name)
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self.weight_map = {name: single_path.name for name in handle.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint found in {self.root}")
        self._stack = ExitStack()
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "CheckpointTensorLoader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._handles.clear()
        self._stack.close()

    def _handle(self, filename: str) -> object:
        handle = self._handles.get(filename)
        if handle is None:
            path = self._shard_path(filename)
            if not path.exists():
                raise FileNotFoundError(f"checkpoint shard is not local: {path}")
            handle = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            self._handles[filename] = handle
        return handle

    def _shard_path(self, filename: str) -> Path:
        relative = Path(filename)
        if (
            not filename
            or relative.is_absolute()
            or relative.name != filename
            or "/" in filename
            or "\\" in filename
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"unsafe checkpoint shard filename: {filename!r}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard escapes checkpoint root: {filename!r}")
        return path

    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def shape(self, name: str) -> tuple[int, ...]:
        handle = self._handle(self._filename(name))
        return tuple(handle.get_slice(name).get_shape())

    def dtype(self, name: str) -> str:
        handle = self._handle(self._filename(name))
        return str(handle.get_slice(name).get_dtype())

    def tensor(
        self,
        name: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        handle = self._handle(self._filename(name))
        return _finish_tensor(handle.get_tensor(name), device=device, dtype=dtype)

    def shard(
        self,
        name: str,
        *,
        dim: int,
        rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        handle = self._handle(self._filename(name))
        tensor_slice = handle.get_slice(name)
        shape = tuple(tensor_slice.get_shape())
        if not 0 <= dim < len(shape):
            raise ValueError(f"invalid shard dimension {dim} for {name} shape={shape}")
        if shape[dim] % world_size:
            raise ValueError(f"cannot shard {name} shape={shape} dim={dim} across {world_size} ranks")
        width = shape[dim] // world_size
        index = [slice(None)] * len(shape)
        index[dim] = slice(rank * width, (rank + 1) * width)
        return _finish_tensor(tensor_slice[tuple(index)], device=device, dtype=dtype)

    def _filename(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError:
            raise KeyError(f"checkpoint tensor not found: {name}") from None


def _finish_tensor(tensor: Tensor, *, device: torch.device, dtype: torch.dtype | None) -> Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.to(device=device, non_blocking=True)
    return tensor if tensor.is_contiguous() else tensor.contiguous()
