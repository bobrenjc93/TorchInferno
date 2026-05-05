from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

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

    def describe(self) -> list[RegisteredPass]:
        return list(self._passes)


def replace_call_function_targets(
    replacements: Mapping[Callable[..., object], Callable[..., object]],
) -> GraphPass:
    """Build a graph pass that swaps call_function targets.

    This is intentionally small but important scaffolding: custom kernels such
    as NVFP4 MoE or specialized attention can start as target replacements
    before graduating to richer pattern matching.
    """

    def pass_fn(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in replacements:
                node.target = replacements[node.target]
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module

    return pass_fn


def replace_call_module_targets(
    replacements: Mapping[str, torch.nn.Module],
) -> GraphPass:
    """Build a graph pass that swaps named call_module targets.

    This is a pragmatic bridge for custom kernel modules. More advanced
    pattern-matching passes can lower into this once they identify a module
    subtree that should become a fused implementation.
    """

    def pass_fn(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for name, module in replacements.items():
            graph_module.add_submodule(name, module)
        graph_module.recompile()
        return graph_module

    return pass_fn


def annotate_matching_nodes(
    predicate: Callable[[torch.fx.Node], bool],
    *,
    key: str,
    value: object = True,
) -> GraphPass:
    def pass_fn(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for node in graph_module.graph.nodes:
            if predicate(node):
                node.meta[key] = value
        return graph_module

    return pass_fn
