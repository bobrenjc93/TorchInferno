from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from torchinferno.models.dsv4 import DSv4Config, DSv4ForCausalLM


CONFIG_NAME = "dsv4_config.json"
HF_CONFIG_NAME = "config.json"
SAFETENSORS_NAME = "model.safetensors"
SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"

M = TypeVar("M", bound=DSv4ForCausalLM)


def save_dsv4_pretrained(model: DSv4ForCausalLM, save_directory: str | Path) -> None:
    path = Path(save_directory)
    path.mkdir(parents=True, exist_ok=True)
    config = model.config.to_dict()
    (path / CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (path / HF_CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, path / SAFETENSORS_NAME, metadata={"format": "pt"})


def load_dsv4_pretrained(
    model_cls: type[M],
    pretrained_model_name_or_path: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> M:
    path = _resolve_pretrained_path(
        pretrained_model_name_or_path,
        token=_resolve_token(token),
        revision=revision,
        cache_dir=cache_dir,
    )
    config = DSv4Config.from_dict(_load_config(path))
    model = model_cls(config)
    state_dict = _load_state_dict(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _resolve_pretrained_path(
    pretrained_model_name_or_path: str | Path,
    *,
    token: str | None,
    revision: str | None,
    cache_dir: str | Path | None,
) -> Path:
    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        return candidate

    snapshot = snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            CONFIG_NAME,
            HF_CONFIG_NAME,
            SAFETENSORS_NAME,
            SAFETENSORS_INDEX_NAME,
            "*.safetensors",
            PYTORCH_WEIGHTS_NAME,
        ],
    )
    return Path(snapshot)


def _load_config(path: Path) -> dict[str, object]:
    for name in (CONFIG_NAME, HF_CONFIG_NAME):
        candidate = path / name
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"no {CONFIG_NAME} or {HF_CONFIG_NAME} found in {path}")


def _load_state_dict(path: Path, *, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    safetensors_path = path / SAFETENSORS_NAME
    if safetensors_path.exists():
        return load_file(safetensors_path, device=str(map_location))

    index_path = path / SAFETENSORS_INDEX_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        state_dict: dict[str, torch.Tensor] = {}
        for shard in shards:
            state_dict.update(load_file(path / shard, device=str(map_location)))
        return state_dict

    pytorch_path = path / PYTORCH_WEIGHTS_NAME
    if pytorch_path.exists():
        loaded = torch.load(pytorch_path, map_location=map_location)
        if not isinstance(loaded, dict):
            raise TypeError(f"{pytorch_path} did not contain a state dict")
        return loaded

    raise FileNotFoundError(f"no supported DSv4 weights found in {path}")
