from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MODEL_TYPES = {
    "dsv4": "dsv4",
    "deepseek_v32": "deepseek-v3.2",
    "deepseek_v4": "deepseek-v4",
    "llama": "llama3",
    "llama3": "llama3",
}

_ARCHITECTURES = {
    "dsv4forcausallm": "dsv4",
    "deepseekv32forcausallm": "deepseek-v3.2",
    "deepseekv4forcausallm": "deepseek-v4",
    "deepseek_v4forcausallm": "deepseek-v4",
    "llamaforcausallm": "llama3",
    "llama3forcausallm": "llama3",
}


def detect_model_identity(config: Mapping[str, Any], *, required: bool = True) -> str | None:
    """Return a canonical family using exact config identifiers.

    A recognized model type and architecture must agree. This prevents a V4
    checkpoint from falling through to a V3.2 loader after a config relabel.
    """

    model_type = str(config.get("model_type", "")).strip().lower()
    type_family = _MODEL_TYPES.get(model_type)
    architecture_families = {
        _ARCHITECTURES[name]
        for value in config.get("architectures", ()) or ()
        if (name := str(value).strip().lower()) in _ARCHITECTURES
    }
    if len(architecture_families) > 1:
        raise ValueError(f"conflicting model architectures: {sorted(architecture_families)}")
    architecture_family = next(iter(architecture_families), None)
    if type_family is not None and architecture_family is not None and type_family != architecture_family:
        raise ValueError(
            "conflicting model identity: "
            f"model_type={model_type!r} identifies {type_family}, "
            f"architectures identify {architecture_family}"
        )
    family = type_family or architecture_family
    if family is None and required:
        architectures = [str(value) for value in config.get("architectures", ()) or ()]
        raise ValueError(f"unsupported model identity: model_type={model_type!r}, architectures={architectures}")
    return family


def require_model_identity(config: Mapping[str, Any], expected: str) -> None:
    actual = detect_model_identity(config)
    if actual != expected:
        raise ValueError(f"expected {expected} checkpoint, got {actual}")
