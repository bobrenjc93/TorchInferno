from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from torchinferno.models.deepseek import DeepSeekV32ForCausalLM
from torchinferno.models.dsv4 import DSv4ForCausalLM
from torchinferno.models.hf import load_config, resolve_pretrained_path


def load_model_auto(
    pretrained_model_name_or_path: str | Path,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Load the TorchInferno model class implied by a checkpoint config."""

    path = resolve_pretrained_path(
        pretrained_model_name_or_path,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
    )
    config = load_config(path)
    model_type = str(config.get("model_type", "")).lower()
    architectures = {str(name).lower() for name in config.get("architectures", [])}
    if model_type == "deepseek_v32" or "deepseekv32forcausallm" in architectures:
        return DeepSeekV32ForCausalLM.from_pretrained(path, map_location=map_location, strict=strict)
    if model_type == "dsv4" or "dsv4forcausallm" in architectures:
        return DSv4ForCausalLM.from_pretrained(path, map_location=map_location, strict=strict)
    raise ValueError(f"unsupported TorchInferno model type in {path}: {model_type}, {architectures}")
