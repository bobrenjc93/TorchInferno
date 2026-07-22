from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from torchinferno.models.deepseek_v32 import DeepSeekV32ForCausalLM
from torchinferno.models.deepseek_v4 import DeepSeekV4ForCausalLM
from torchinferno.models.dsv4 import DSv4ForCausalLM
from torchinferno.models.hf import load_config, resolve_pretrained_path
from torchinferno.models.identity import detect_model_identity


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
    family = detect_model_identity(config)
    if family == "deepseek-v3.2":
        return DeepSeekV32ForCausalLM.from_pretrained(path, map_location=map_location, strict=strict)
    if family == "deepseek-v4":
        return DeepSeekV4ForCausalLM.from_pretrained(path, map_location=map_location, strict=strict)
    if family == "dsv4":
        return DSv4ForCausalLM.from_pretrained(path, map_location=map_location, strict=strict)
    raise ValueError(f"unsupported TorchInferno model family in {path}: {family}")
