from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from torchinferno.models.dsv4 import DSv4Config
from torchinferno.models.hf import (
    CONFIG_NAME,
    HF_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFETENSORS_INDEX_NAME,
    SAFETENSORS_NAME,
    load_config,
    resolve_pretrained_path,
)


CONVERSION_REPORT_NAME = "torchinferno_conversion_report.json"

UNSUPPORTED_NATIVE_CONFIG_FIELDS = {
    "q_lora_rank": "TorchInferno DSv4 currently has a single q_proj instead of DeepSeek query LoRA.",
    "qk_rope_head_dim": "TorchInferno DSv4 currently uses one RoPE head_dim instead of split rope/nope QK heads.",
    "qk_nope_head_dim": "TorchInferno DSv4 currently uses one RoPE head_dim instead of split rope/nope QK heads.",
    "v_head_dim": "TorchInferno DSv4 currently requires value head_dim == hidden_size / num_attention_heads.",
    "first_k_dense_replace": "TorchInferno DSv4 currently models every layer as routed MoE.",
    "n_shared_experts": "TorchInferno DSv4 currently has routed experts only, with no shared expert branch.",
    "n_group": "TorchInferno DSv4 router does not yet model grouped expert selection.",
    "topk_group": "TorchInferno DSv4 router does not yet model grouped expert selection.",
    "routed_scaling_factor": "TorchInferno DSv4 router does not yet model DeepSeek routed scaling.",
    "norm_topk_prob": "TorchInferno DSv4 router always normalizes selected top-k probabilities.",
}

_DTYPE_NAME = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    filename: str

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _dtype_size(self.dtype)


@dataclass(frozen=True)
class ConversionIssue:
    severity: str
    code: str
    message: str
    target_name: Optional[str] = None
    source_name: Optional[str] = None


@dataclass(frozen=True)
class TensorMapping:
    target_name: str
    source_name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ConversionReport:
    source: str
    model_type: str
    target_config: dict[str, object]
    mapped_tensors: tuple[TensorMapping, ...]
    issues: tuple[ConversionIssue, ...] = field(default_factory=tuple)

    @property
    def compatible(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "model_type": self.model_type,
            "compatible": self.compatible,
            "target_config": self.target_config,
            "mapped_tensors": [asdict(mapping) for mapping in self.mapped_tensors],
            "issues": [asdict(issue) for issue in self.issues],
        }

    def summary(self) -> str:
        status = "compatible" if self.compatible else "incompatible"
        lines = [
            f"DeepSeek conversion audit: {status}",
            f"source={self.source}",
            f"model_type={self.model_type}",
            f"mapped_tensors={len(self.mapped_tensors)}",
            f"issues={len(self.issues)}",
        ]
        for issue in self.issues[:20]:
            location = f" target={issue.target_name}" if issue.target_name else ""
            source = f" source={issue.source_name}" if issue.source_name else ""
            lines.append(f"- {issue.severity}:{issue.code}:{location}{source} {issue.message}")
        if len(self.issues) > 20:
            lines.append(f"- ... {len(self.issues) - 20} more issues")
        return "\n".join(lines)


class IncompatibleCheckpointError(RuntimeError):
    def __init__(self, report: ConversionReport) -> None:
        super().__init__(report.summary())
        self.report = report


def audit_deepseek_checkpoint(
    checkpoint: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> ConversionReport:
    path = resolve_pretrained_path(checkpoint, token=token, revision=revision, cache_dir=cache_dir)
    config_dict = load_config(path)
    target_config = DSv4Config.from_dict(config_dict)
    tensor_index = index_checkpoint_tensors(path)
    expected = expected_dsv4_tensors(target_config)
    native_map = deepseek_to_dsv4_key_map(target_config)

    issues: list[ConversionIssue] = []
    for field_name, message in UNSUPPORTED_NATIVE_CONFIG_FIELDS.items():
        if field_name in config_dict:
            issues.append(ConversionIssue("error", "unsupported_config", message))

    mapped: list[TensorMapping] = []
    used_sources: set[str] = set()
    for target_name, expected_shape in expected.items():
        source_name = _first_existing((target_name, *native_map.get(target_name, ())), tensor_index)
        if source_name is None or source_name not in tensor_index:
            issues.append(
                ConversionIssue(
                    "error",
                    "missing_tensor",
                    "No source tensor maps to required TorchInferno DSv4 tensor.",
                    target_name=target_name,
                    source_name=source_name,
                )
            )
            continue
        source_info = tensor_index[source_name]
        if source_info.shape != expected_shape:
            issues.append(
                ConversionIssue(
                    "error",
                    "shape_mismatch",
                    f"Expected shape {expected_shape}, found {source_info.shape}.",
                    target_name=target_name,
                    source_name=source_name,
                )
            )
            continue
        mapped.append(TensorMapping(target_name, source_name, expected_shape, source_info.dtype))
        used_sources.add(source_name)

    for source_name in sorted(set(tensor_index) - used_sources):
        if _looks_like_model_weight(source_name):
            issues.append(
                ConversionIssue(
                    "warning",
                    "unused_tensor",
                    "Source tensor is not consumed by the TorchInferno DSv4 target model.",
                    source_name=source_name,
                )
            )

    return ConversionReport(
        source=str(path),
        model_type=str(config_dict.get("model_type", "unknown")),
        target_config=target_config.to_dict(),
        mapped_tensors=tuple(mapped),
        issues=tuple(issues),
    )


def convert_deepseek_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    dtype: torch.dtype | None = None,
    max_shard_size: int | str = "5GB",
    allow_partial: bool = False,
) -> ConversionReport:
    path = resolve_pretrained_path(checkpoint, token=token, revision=revision, cache_dir=cache_dir)
    report = audit_deepseek_checkpoint(path)
    if not report.compatible and not allow_partial:
        raise IncompatibleCheckpointError(report)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = report.target_config
    (output_path / CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_path / HF_CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_path / CONVERSION_REPORT_NAME).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    tensor_index = index_checkpoint_tensors(path)
    mappings = list(report.mapped_tensors)
    if allow_partial:
        mappings = [
            mapping
            for mapping in mappings
            if mapping.source_name in tensor_index and tensor_index[mapping.source_name].shape == mapping.shape
        ]

    shard_limit = parse_size(max_shard_size)
    writer = _ShardWriter(output_path, shard_limit)
    for mapping in mappings:
        tensor = load_checkpoint_tensor(path, tensor_index[mapping.source_name])
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        writer.add(mapping.target_name, tensor)
    writer.close()
    return report


def index_checkpoint_tensors(path: str | Path) -> dict[str, TensorInfo]:
    root = Path(path)
    safetensor_files = _safetensor_files(root)
    if safetensor_files:
        index: dict[str, TensorInfo] = {}
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor_slice = handle.get_slice(key)
                    index[key] = TensorInfo(
                        name=key,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=_DTYPE_NAME.get(str(tensor_slice.get_dtype()), str(tensor_slice.get_dtype())),
                        filename=file_path.name,
                    )
        return index

    pytorch_path = root / PYTORCH_WEIGHTS_NAME
    if pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError(f"{pytorch_path} did not contain a state dict")
        return {
            name: TensorInfo(name, tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), pytorch_path.name)
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
        }

    raise FileNotFoundError(f"no supported checkpoint weights found in {root}")


def load_checkpoint_tensor(path: str | Path, info: TensorInfo) -> torch.Tensor:
    root = Path(path)
    file_path = root / info.filename
    if file_path.suffix == ".safetensors":
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(info.name)
    if file_path.name == PYTORCH_WEIGHTS_NAME:
        state_dict = torch.load(file_path, map_location="cpu")
        return state_dict[info.name]
    raise FileNotFoundError(file_path)


def expected_dsv4_tensors(config: DSv4Config) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    head_dim = config.head_dim
    kv_out = 2 * config.num_key_value_heads * head_dim
    expected = {
        "embed_tokens.weight": (config.vocab_size, hidden),
        "norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for layer in range(config.num_layers):
        prefix = f"layers.{layer}"
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (hidden,),
                f"{prefix}.attn.q_proj.weight": (hidden, hidden),
                f"{prefix}.attn.kv_a_proj.weight": (config.latent_kv_size, hidden),
                f"{prefix}.attn.kv_a_norm.weight": (config.latent_kv_size,),
                f"{prefix}.attn.kv_b_proj.weight": (kv_out, config.latent_kv_size),
                f"{prefix}.attn.o_proj.weight": (hidden, hidden),
                f"{prefix}.moe_norm.weight": (hidden,),
                f"{prefix}.moe.router.weight": (config.num_experts, hidden),
            }
        )
        for expert in range(config.num_experts):
            expert_prefix = f"{prefix}.moe.experts.{expert}"
            expected.update(
                {
                    f"{expert_prefix}.w1.weight": (config.intermediate_size, hidden),
                    f"{expert_prefix}.w3.weight": (config.intermediate_size, hidden),
                    f"{expert_prefix}.w2.weight": (hidden, config.intermediate_size),
                }
            )
    return expected


def deepseek_to_dsv4_key_map(config: DSv4Config) -> dict[str, tuple[str, ...]]:
    mapping = {
        "embed_tokens.weight": ("model.embed_tokens.weight",),
        "norm.weight": ("model.norm.weight",),
        "lm_head.weight": ("lm_head.weight", "model.lm_head.weight"),
    }
    for layer in range(config.num_layers):
        source = f"model.layers.{layer}"
        target = f"layers.{layer}"
        mapping.update(
            {
                f"{target}.attn_norm.weight": (f"{source}.input_layernorm.weight",),
                f"{target}.attn.q_proj.weight": (f"{source}.self_attn.q_proj.weight",),
                f"{target}.attn.kv_a_proj.weight": (
                    f"{source}.self_attn.kv_a_proj.weight",
                    f"{source}.self_attn.kv_a_proj_with_mqa.weight",
                ),
                f"{target}.attn.kv_a_norm.weight": (f"{source}.self_attn.kv_a_layernorm.weight",),
                f"{target}.attn.kv_b_proj.weight": (f"{source}.self_attn.kv_b_proj.weight",),
                f"{target}.attn.o_proj.weight": (f"{source}.self_attn.o_proj.weight",),
                f"{target}.moe_norm.weight": (f"{source}.post_attention_layernorm.weight",),
                f"{target}.moe.router.weight": (
                    f"{source}.mlp.gate.weight",
                    f"{source}.mlp.gate_proj.weight",
                ),
            }
        )
        for expert in range(config.num_experts):
            target_prefix = f"{target}.moe.experts.{expert}"
            mapping[f"{target_prefix}.w1.weight"] = tuple(
                f"{source}.mlp.{source_experts}.{expert}.gate_proj.weight"
                for source_experts in ("experts", "routed_experts")
            )
            mapping[f"{target_prefix}.w3.weight"] = tuple(
                f"{source}.mlp.{source_experts}.{expert}.up_proj.weight"
                for source_experts in ("experts", "routed_experts")
            )
            mapping[f"{target_prefix}.w2.weight"] = tuple(
                f"{source}.mlp.{source_experts}.{expert}.down_proj.weight"
                for source_experts in ("experts", "routed_experts")
            )
    return mapping


def parse_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)\s*", value.upper())
    if match is None:
        raise ValueError(f"invalid size: {value}")
    amount = float(match.group(1))
    suffix = match.group(2)
    scale = {
        "": 1,
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }[suffix]
    return int(amount * scale)


def dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    normalized = name.lower().replace("torch.", "")
    if normalized in {"fp32", "float32", "f32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "f16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


class _ShardWriter:
    def __init__(self, output_path: Path, max_shard_size: int) -> None:
        self.output_path = output_path
        self.max_shard_size = max_shard_size
        self.shard_id = 0
        self.current_size = 0
        self.current: dict[str, torch.Tensor] = {}
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self.shard_names: list[str] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor_size = tensor.numel() * tensor.element_size()
        if self.current and self.current_size + tensor_size > self.max_shard_size:
            self._flush()
        self.current[name] = tensor.detach().cpu().contiguous()
        self.current_size += tensor_size
        self.total_size += tensor_size

    def close(self) -> None:
        self._flush()
        shard_count = self.shard_id
        if shard_count == 0:
            return
        if shard_count == 1:
            only_shard = self.output_path / self._shard_name(1)
            final = self.output_path / SAFETENSORS_NAME
            if final.exists():
                final.unlink()
            only_shard.rename(final)
            self.weight_map = {name: SAFETENSORS_NAME for name in self.weight_map}
            index_path = self.output_path / SAFETENSORS_INDEX_NAME
            if index_path.exists():
                index_path.unlink()
            return

        final_weight_map: dict[str, str] = {}
        for shard_id, temporary_name in enumerate(self.shard_names, start=1):
            final_name = f"model-{shard_id:05d}-of-{shard_count:05d}.safetensors"
            (self.output_path / temporary_name).rename(self.output_path / final_name)
            for tensor_name, shard_name in self.weight_map.items():
                if shard_name == temporary_name:
                    final_weight_map[tensor_name] = final_name
        self.weight_map = final_weight_map
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        (self.output_path / SAFETENSORS_INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    def _flush(self) -> None:
        if not self.current:
            return
        self.shard_id += 1
        shard_name = self._shard_name(self.shard_id)
        save_file(self.current, self.output_path / shard_name, metadata={"format": "pt"})
        self.shard_names.append(shard_name)
        for name in self.current:
            self.weight_map[name] = shard_name
        self.current = {}
        self.current_size = 0

    @staticmethod
    def _shard_name(shard_id: int) -> str:
        return f"model-{shard_id:05d}-of-xxxxx.safetensors"


def _safetensor_files(root: Path) -> list[Path]:
    single = root / SAFETENSORS_NAME
    if single.exists():
        return [single]
    index = root / SAFETENSORS_INDEX_NAME
    if index.exists():
        data = json.loads(index.read_text())
        return [root / name for name in sorted(set(data["weight_map"].values()))]
    return sorted(root.glob("*.safetensors"))


def _looks_like_model_weight(name: str) -> bool:
    return any(part in name for part in ("embed_tokens", "layers", "lm_head", "norm", "model."))


def _first_existing(candidates: Iterable[str], tensor_index: dict[str, TensorInfo]) -> str | None:
    for candidate in candidates:
        if candidate in tensor_index:
            return candidate
    return None


def _dtype_size(dtype: str) -> int:
    if dtype in {"float64", "int64"}:
        return 8
    if dtype in {"float32", "int32"}:
        return 4
    if dtype in {"float16", "bfloat16", "int16"}:
        return 2
    return 1
