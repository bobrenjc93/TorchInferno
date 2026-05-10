from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchinferno.graph import trace_with_make_fx


FXGraphKey = tuple[tuple[int, ...], str, str]


class FXGraphBackedMixin:
    """Mixin for v0 variants backed by shape-specialized make_fx graphs."""

    def _v0_graph_cache(self) -> dict[FXGraphKey, torch.fx.GraphModule]:
        cache = self.__dict__.get("_v0_fx_graphs")
        if cache is None:
            cache = {}
            self.__dict__["_v0_fx_graphs"] = cache
        return cache

    def _default_v0_input_ids(self) -> Tensor:
        raise NotImplementedError

    def _traceable_forward(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError

    def v0_graph(self, input_ids: Tensor | None = None) -> torch.fx.GraphModule:
        input_ids = self._default_v0_input_ids() if input_ids is None else input_ids
        key = _graph_key(input_ids)
        cache = self._v0_graph_cache()
        graph_module = cache.get(key)
        if graph_module is None:
            graph_module = trace_with_make_fx(lambda ids: self._traceable_forward(ids), input_ids)
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            cache[key] = graph_module
        return graph_module

    def _run_v0_graph(self, input_ids: Tensor) -> Tensor:
        return self.v0_graph(input_ids)(input_ids)

    def print_readable(
        self,
        input_ids: Tensor | None = None,
        *,
        print_output: bool = True,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        **kwargs: Any,
    ) -> str:
        return self.v0_graph(input_ids).print_readable(
            print_output=print_output,
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            **kwargs,
        )


def default_input_ids(
    *,
    vocab_size: int,
    max_seq_len: int,
    device: torch.device,
    tokens: int = 3,
) -> Tensor:
    tokens = max(1, min(tokens, max_seq_len))
    return torch.arange(tokens, device=device, dtype=torch.long)[None, :] % vocab_size


def _graph_key(input_ids: Tensor) -> FXGraphKey:
    return tuple(int(dim) for dim in input_ids.shape), str(input_ids.dtype), str(input_ids.device)
