from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from torchinferno.models.deepseek_v4.config import DeepSeekV4Config
from torchinferno.models.hf import HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME
from torchinferno.models.identity import require_model_identity


DEEPSEEK_V4_FLASH_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"


@dataclass(frozen=True)
class DeepSeekV4CheckpointIssue:
    severity: str
    code: str
    message: str
    tensor: str | None = None


@dataclass(frozen=True)
class DeepSeekV4CheckpointReport:
    source: str
    config: dict[str, Any]
    tensor_count: int
    shard_count: int
    total_size: int | None
    weights_available: bool
    issues: tuple[DeepSeekV4CheckpointIssue, ...]

    @property
    def compatible(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "compatible": self.compatible,
            "config": self.config,
            "tensor_count": self.tensor_count,
            "shard_count": self.shard_count,
            "total_size": self.total_size,
            "weights_available": self.weights_available,
            "issues": [asdict(issue) for issue in self.issues],
        }


def resolve_deepseek_v4_manifest(
    checkpoint: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.exists():
        return candidate
    snapshot = snapshot_download(
        repo_id=str(checkpoint),
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME],
    )
    return Path(snapshot)


def audit_deepseek_v4_checkpoint(
    checkpoint: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> DeepSeekV4CheckpointReport:
    root = resolve_deepseek_v4_manifest(
        checkpoint,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
    )
    config_payload = json.loads((root / HF_CONFIG_NAME).read_text())
    require_model_identity(config_payload, "deepseek-v4")
    config = DeepSeekV4Config.from_dict(config_payload)
    index_path = root / SAFETENSORS_INDEX_NAME
    if not index_path.exists():
        return DeepSeekV4CheckpointReport(
            source=str(root),
            config=config.to_dict(),
            tensor_count=0,
            shard_count=0,
            total_size=None,
            weights_available=False,
            issues=(DeepSeekV4CheckpointIssue("error", "missing_index", "model.safetensors.index.json is required"),),
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    filenames = {str(value) for value in weight_map.values()}
    issues: list[DeepSeekV4CheckpointIssue] = []
    for filename in sorted(filenames):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            issues.append(
                DeepSeekV4CheckpointIssue(
                    "error",
                    "invalid_shard_name",
                    f"unsafe or unsupported shard filename: {filename}",
                )
            )
    expected = expected_deepseek_v4_tensor_names(config)
    missing = sorted(expected - set(weight_map))
    for name in missing:
        issues.append(
            DeepSeekV4CheckpointIssue(
                "error",
                "missing_tensor",
                "required DeepSeek V4 tensor is absent from the index",
                tensor=name,
            )
        )
    known_prefixes = tuple(f"layers.{index}." for index in range(config.num_hidden_layers))
    unexpected = sorted(
        name
        for name in weight_map
        if name not in expected
        and not name.startswith("mtp.")
        and name not in {"embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
        and not name.startswith(known_prefixes)
    )
    for name in unexpected:
        issues.append(
            DeepSeekV4CheckpointIssue(
                "warning",
                "unexpected_tensor",
                "tensor is outside the released V4 contract",
                tensor=name,
            )
        )
    weights_available = all((root / filename).exists() for filename in filenames)
    metadata = index.get("metadata")
    total_size = None
    if isinstance(metadata, dict) and metadata.get("total_size") is not None:
        total_size = int(metadata["total_size"])
    return DeepSeekV4CheckpointReport(
        source=str(root),
        config=config.to_dict(),
        tensor_count=len(weight_map),
        shard_count=len(filenames),
        total_size=total_size,
        weights_available=weights_available,
        issues=tuple(issues),
    )


def expected_deepseek_v4_tensor_names(config: DeepSeekV4Config) -> set[str]:
    names = {"embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
    for layer_idx in range(config.num_hidden_layers):
        names.update(_block_tensor_names(f"layers.{layer_idx}", layer_idx, config))
    for mtp_idx in range(config.num_nextn_predict_layers):
        prefix = f"mtp.{mtp_idx}"
        names.update(_block_tensor_names(prefix, config.num_hidden_layers + mtp_idx, config, ratio=0))
        names.update(
            {
                f"{prefix}.hc_head_fn",
                f"{prefix}.hc_head_base",
                f"{prefix}.hc_head_scale",
                f"{prefix}.e_proj.weight",
                f"{prefix}.e_proj.scale",
                f"{prefix}.h_proj.weight",
                f"{prefix}.h_proj.scale",
                f"{prefix}.enorm.weight",
                f"{prefix}.hnorm.weight",
                f"{prefix}.norm.weight",
            }
        )
    return names


def _block_tensor_names(
    prefix: str,
    layer_idx: int,
    config: DeepSeekV4Config,
    *,
    ratio: int | None = None,
) -> set[str]:
    names = {
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn.attn_sink",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.kv_norm.weight",
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.ffn.gate.weight",
    }
    for linear in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        names.add(f"{prefix}.attn.{linear}.weight")
        names.add(f"{prefix}.attn.{linear}.scale")
    actual_ratio = config.compress_ratios[layer_idx] if ratio is None else ratio
    if actual_ratio:
        names.update(_compressor_names(f"{prefix}.attn.compressor"))
    if actual_ratio == 4:
        names.add(f"{prefix}.attn.indexer.wq_b.weight")
        names.add(f"{prefix}.attn.indexer.wq_b.scale")
        names.add(f"{prefix}.attn.indexer.weights_proj.weight")
        names.update(_compressor_names(f"{prefix}.attn.indexer.compressor"))
    if layer_idx < config.num_hash_layers:
        names.add(f"{prefix}.ffn.gate.tid2eid")
    else:
        names.add(f"{prefix}.ffn.gate.bias")
    for expert_kind in ("shared_experts",):
        for projection in ("w1", "w2", "w3"):
            names.add(f"{prefix}.ffn.{expert_kind}.{projection}.weight")
            names.add(f"{prefix}.ffn.{expert_kind}.{projection}.scale")
    for expert_idx in range(config.n_routed_experts):
        for projection in ("w1", "w2", "w3"):
            names.add(f"{prefix}.ffn.experts.{expert_idx}.{projection}.weight")
            names.add(f"{prefix}.ffn.experts.{expert_idx}.{projection}.scale")
    return names


def _compressor_names(prefix: str) -> set[str]:
    return {
        f"{prefix}.ape",
        f"{prefix}.wkv.weight",
        f"{prefix}.wgate.weight",
        f"{prefix}.norm.weight",
    }
