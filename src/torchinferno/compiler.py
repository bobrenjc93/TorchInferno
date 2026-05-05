from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CompileConfig:
    """Small, explicit surface for torch.compile experiments."""

    enabled: bool = True
    backend: Optional[str] = None
    mode: Optional[str] = "reduce-overhead"
    fullgraph: bool = False
    dynamic: Optional[bool] = None


def compile_forward(module: torch.nn.Module, config: CompileConfig | None = None) -> torch.nn.Module:
    """Compile a module's forward method in-place and return the module.

    Keeping this as a tiny helper gives CLI tools, research harnesses, and
    focused model experiments one shared place for compiler policy.
    """

    config = CompileConfig() if config is None else config
    if not config.enabled:
        return module

    kwargs: dict[str, object] = {
        "fullgraph": config.fullgraph,
    }
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.mode is not None:
        kwargs["mode"] = config.mode
    if config.dynamic is not None:
        kwargs["dynamic"] = config.dynamic

    module.forward = torch.compile(module.forward, **kwargs)  # type: ignore[method-assign]
    return module
