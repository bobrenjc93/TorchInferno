from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


GraphPass = Callable[[torch.fx.GraphModule], torch.fx.GraphModule]


@dataclass(frozen=True)
class RegisteredPass:
    name: str
    pass_fn: GraphPass
    description: str


class PassRegistry:
    """Small registry for pattern-match graph rewrites.

    The intent is to keep model code simple while giving optimization work a
    stable place to register replacements such as custom attention or NVFP4 MoE
    kernels.
    """

    def __init__(self) -> None:
        self._passes: list[RegisteredPass] = []

    def register(self, name: str, pass_fn: GraphPass, description: str = "") -> None:
        if any(existing.name == name for existing in self._passes):
            raise ValueError(f"duplicate graph pass: {name}")
        self._passes.append(RegisteredPass(name, pass_fn, description))

    def run(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for registered in self._passes:
            graph_module = registered.pass_fn(graph_module)
        graph_module.recompile()
        return graph_module

    def names(self) -> list[str]:
        return [registered.name for registered in self._passes]
