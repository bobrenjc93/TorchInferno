from __future__ import annotations

from typing import Any

import torch


def trace_with_make_fx(module: torch.nn.Module, *args: Any, fake: bool = False, **kwargs: Any) -> torch.fx.GraphModule:
    """Trace a module with make_fx, optionally under FakeTensorMode."""

    try:
        from torch.fx.experimental.proxy_tensor import FakeTensorMode, make_fx
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError("This PyTorch build does not expose make_fx/FakeTensorMode") from exc

    if not fake:
        return make_fx(module)(*args, **kwargs)

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake_args = tuple(mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        fake_kwargs = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return make_fx(module)(*fake_args, **fake_kwargs)
