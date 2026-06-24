from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download, try_to_load_from_cache
from safetensors import safe_open
from torch import Tensor

from torchinferno.models.hf import HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME
from torchinferno.models.llama3.config import Llama3Config
from torchinferno.runtime.sampling import sample_next_token


LLAMA3_70B_REPO_ID = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass(frozen=True)
class Llama3PipelineLoadReport:
    checkpoint: str
    dtype: str
    devices: tuple[str, ...]
    layer_devices: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "devices": list(self.devices),
            "layer_devices": list(self.layer_devices),
        }


class Llama3LayerKVCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.seq_len = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, tokens, _ = keys.shape
        if batch > self.batch_size:
            raise ValueError("cache batch is smaller than incoming batch")
        end = self.seq_len + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        self.keys[:batch, :, self.seq_len : end, :].copy_(keys)
        self.values[:batch, :, self.seq_len : end, :].copy_(values)
        self.seq_len = end
        return self.keys[:batch, :, :end, :], self.values[:batch, :, :end, :]


class Llama3PipelineCache:
    def __init__(self, layers: list[Llama3LayerKVCache]) -> None:
        self.layers = layers

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0


class _CheckpointTensorLoader:
    def __init__(self, checkpoint: str | Path) -> None:
        self.root = resolve_llama3_checkpoint(checkpoint)
        index_path = self.root / SAFETENSORS_INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(f"no {SAFETENSORS_INDEX_NAME} found in {self.root}")
        index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._shape_cache: dict[str, tuple[int, ...]] = {}

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        cached = self._shape_cache.get(name)
        if cached is not None:
            return cached
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        path = self.root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(name).get_shape())
        self._shape_cache[name] = shape
        return shape

    def get_tensor(self, name: str, *, device: torch.device, dtype: torch.dtype | None) -> Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        path = self.root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        return _finish_checkpoint_tensor(tensor, device=device, dtype=dtype)

    def get_tensor_shard(
        self,
        name: str,
        *,
        dim: int,
        rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        path = self.root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            if shape[dim] % world_size != 0:
                raise ValueError(f"cannot shard {name} shape={shape} dim={dim} across {world_size} ranks")
            shard = shape[dim] // world_size
            start = rank * shard
            end = start + shard
            index = [slice(None)] * len(shape)
            index[dim] = slice(start, end)
            tensor = tensor_slice[tuple(index)]
        return _finish_checkpoint_tensor(tensor, device=device, dtype=dtype)


def _finish_checkpoint_tensor(tensor: Tensor, *, device: torch.device, dtype: torch.dtype | None) -> Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.to(device=device, non_blocking=True)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


class _Llama3PipelineLayer:
    def __init__(self, config: Llama3Config, layer_id: int, device: torch.device, weights: dict[str, Tensor]) -> None:
        self.config = config
        self.layer_id = layer_id
        self.device = device
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        self.q_proj_weight = weights["self_attn.q_proj.weight"]
        self.k_proj_weight = weights["self_attn.k_proj.weight"]
        self.v_proj_weight = weights["self_attn.v_proj.weight"]
        self.o_proj_weight = weights["self_attn.o_proj.weight"]
        self.gate_proj_weight = weights["mlp.gate_proj.weight"]
        self.up_proj_weight = weights["mlp.up_proj.weight"]
        self.down_proj_weight = weights["mlp.down_proj.weight"]
        self.inv_freq = _build_inv_freq(config, device)

    def forward(self, hidden: Tensor, positions: Tensor, cache: Llama3LayerKVCache | None) -> Tensor:
        hidden = hidden.to(self.device, non_blocking=True)
        positions = positions.to(self.device, non_blocking=True)
        residual = hidden
        attn_in = _rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        hidden = residual + self._attention(attn_in, positions, cache)
        residual = hidden
        mlp_in = _rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        hidden = residual + self._mlp(mlp_in)
        return hidden

    def _attention(self, hidden: Tensor, positions: Tensor, cache: Llama3LayerKVCache | None) -> Tensor:
        batch, tokens, _ = hidden.shape
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = F.linear(hidden, self.q_proj_weight).view(batch, tokens, heads, head_dim).transpose(1, 2)
        k = F.linear(hidden, self.k_proj_weight).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        v = F.linear(hidden, self.v_proj_weight).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        q, k = _apply_rotary(q, k, positions, self.inv_freq)
        if cache is not None:
            k, v = cache.append(k, v)

        enable_gqa = kv_heads != heads
        if tokens == 1:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        elif k.size(-2) == tokens:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=enable_gqa,
            )
        else:
            key_positions = torch.arange(k.size(-2), device=hidden.device)
            allowed = key_positions[None, :] <= positions[:, None]
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed[None, None, :, :],
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.config.hidden_size)
        return F.linear(out, self.o_proj_weight)

    def _mlp(self, hidden: Tensor) -> Tensor:
        gate = F.linear(hidden, self.gate_proj_weight)
        up = F.linear(hidden, self.up_proj_weight)
        return F.linear(F.silu(gate) * up, self.down_proj_weight)


class Llama3PipelineForCausalLM:
    """Pipeline-sharded Llama3 inference path for checkpoints too large for one GPU.

    This class intentionally keeps the first production-scale Llama path simple:
    one Python process owns all devices, each decoder layer lives on exactly one
    device, and activation tensors move between devices at layer boundaries.
    """

    provenance_variant = "llama3:pipeline-v0"

    def __init__(
        self,
        config: Llama3Config,
        *,
        embed_tokens_weight: Tensor,
        norm_weight: Tensor,
        lm_head_weight: Tensor,
        layers: Sequence[_Llama3PipelineLayer],
        devices: Sequence[torch.device],
        dtype: torch.dtype,
        checkpoint: str | Path,
    ) -> None:
        self.config = config
        self.embed_tokens_weight = embed_tokens_weight
        self.norm_weight = norm_weight
        self.lm_head_weight = lm_head_weight
        self.layers = list(layers)
        self.devices = tuple(devices)
        self.dtype = dtype
        self.checkpoint = Path(checkpoint)
        self.embed_device = embed_tokens_weight.device
        self.output_device = lm_head_weight.device
        self.training = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = LLAMA3_70B_REPO_ID,
        *,
        devices: Sequence[str | torch.device] | None = None,
        dtype: torch.dtype | str | None = None,
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "Llama3PipelineForCausalLM":
        root = resolve_llama3_checkpoint(checkpoint, token=token, revision=revision, cache_dir=cache_dir)
        config = Llama3Config.from_dict(json.loads((root / HF_CONFIG_NAME).read_text()))
        torch_dtype = _resolve_dtype(dtype, root)
        resolved_devices = _resolve_devices(devices)
        loader = _CheckpointTensorLoader(root)

        embed_device = resolved_devices[0]
        output_device = resolved_devices[-1]
        embed_tokens_weight = loader.get_tensor("model.embed_tokens.weight", device=embed_device, dtype=torch_dtype)
        norm_weight = loader.get_tensor("model.norm.weight", device=output_device, dtype=torch_dtype)
        lm_head_weight = loader.get_tensor("lm_head.weight", device=output_device, dtype=torch_dtype)

        layers: list[_Llama3PipelineLayer] = []
        layer_devices: list[torch.device] = []
        for layer_id in range(config.num_hidden_layers):
            device = _device_for_layer(layer_id, config.num_hidden_layers, resolved_devices)
            prefix = f"model.layers.{layer_id}."
            keys = (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
            weights = {key: loader.get_tensor(prefix + key, device=device, dtype=torch_dtype) for key in keys}
            layers.append(_Llama3PipelineLayer(config, layer_id, device, weights))
            layer_devices.append(device)

        model = cls(
            config,
            embed_tokens_weight=embed_tokens_weight,
            norm_weight=norm_weight,
            lm_head_weight=lm_head_weight,
            layers=layers,
            devices=resolved_devices,
            dtype=torch_dtype,
            checkpoint=root,
        )
        model.load_report = Llama3PipelineLoadReport(
            checkpoint=str(root),
            dtype=str(torch_dtype).replace("torch.", ""),
            devices=tuple(str(device) for device in resolved_devices),
            layer_devices=tuple(str(device) for device in layer_devices),
        )
        return model

    def eval(self) -> "Llama3PipelineForCausalLM":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "Llama3PipelineForCausalLM":
        self.training = mode
        return self

    def allocate_cache(self, batch_size: int, max_seq_len: int) -> Llama3PipelineCache:
        layers = [
            Llama3LayerKVCache(
                batch_size,
                max_seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
                device=layer.device,
                dtype=self.dtype,
            )
            for layer in self.layers
        ]
        return Llama3PipelineCache(layers)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Llama3PipelineCache | None = None,
        use_cache: bool = True,
        return_last_logits_only: bool = False,
    ) -> tuple[Tensor, Llama3PipelineCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        if tokens < 1:
            raise ValueError("input_ids must contain at least one token")

        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = self.allocate_cache(batch, input_ids.size(1))
        past_len = active_cache.seq_len if active_cache is not None else 0
        positions = torch.arange(past_len, past_len + tokens, device=self.embed_device)

        hidden = F.embedding(input_ids.to(self.embed_device, non_blocking=True), self.embed_tokens_weight)
        for layer_id, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
            hidden = layer.forward(hidden, positions, layer_cache)

        hidden = hidden.to(self.output_device, non_blocking=True)
        if return_last_logits_only:
            hidden = hidden[:, -1:, :]
        hidden = _rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head_weight)
        return logits, active_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids.to(self.embed_device, non_blocking=True)

        cache = self.allocate_cache(input_ids.size(0), input_ids.size(1) + max_new_tokens)
        logits, cache = self.forward(input_ids, cache=cache, use_cache=True, return_last_logits_only=True)
        next_token = sample_next_token(logits[:, -1, :], temperature).to(self.embed_device, non_blocking=True)
        output = [input_ids.to(self.embed_device, non_blocking=True), next_token[:, None]]

        for _ in range(1, max_new_tokens):
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self.forward(
                next_token[:, None],
                cache=cache,
                use_cache=True,
                return_last_logits_only=True,
            )
            next_token = sample_next_token(logits[:, -1, :], temperature).to(self.embed_device, non_blocking=True)
            output.append(next_token[:, None])
        return torch.cat(output, dim=1)


def resolve_llama3_checkpoint(
    checkpoint: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.exists():
        return candidate
    cached = _resolve_cached_llama3_checkpoint(
        str(checkpoint),
        revision=revision,
        cache_dir=cache_dir,
    )
    if cached is not None:
        return cached
    snapshot = snapshot_download(
        repo_id=str(checkpoint),
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            HF_CONFIG_NAME,
            SAFETENSORS_INDEX_NAME,
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )
    return Path(snapshot)


def _resolve_cached_llama3_checkpoint(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> Path | None:
    cached_files: list[Path] = []
    for filename in (HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME):
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        if not isinstance(cached, str):
            return None
        cached_files.append(Path(cached))

    snapshot = cached_files[0].parent
    if all(path.parent == snapshot for path in cached_files) and all(
        (snapshot / filename).exists() for filename in (HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME)
    ):
        return snapshot
    return None


def _resolve_devices(devices: Sequence[str | torch.device] | None) -> tuple[torch.device, ...]:
    if devices is None:
        if torch.cuda.is_available():
            return tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
        return (torch.device("cpu"),)
    resolved = tuple(torch.device(device) for device in devices)
    if not resolved:
        raise ValueError("at least one device is required")
    return resolved


def _resolve_dtype(dtype: torch.dtype | str | None, checkpoint: Path) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        config = json.loads((checkpoint / HF_CONFIG_NAME).read_text())
        dtype = str(config.get("torch_dtype", "bfloat16"))
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def _device_for_layer(layer_id: int, num_layers: int, devices: Sequence[torch.device]) -> torch.device:
    return devices[min(layer_id * len(devices) // num_layers, len(devices) - 1)]


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return F.rms_norm(x, (x.size(-1),), weight, eps)


def _build_inv_freq(config: Llama3Config, device: torch.device) -> Tensor:
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32) / config.head_dim)
    )
    scaling = config.rope_scaling or {}
    if scaling.get("rope_type") != "llama3":
        return inv_freq

    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    old_context_len = float(scaling["original_max_position_embeddings"])
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed, inv_freq_llama)


def _apply_rotary(q: Tensor, k: Tensor, positions: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype, device=q.device)[None, None, :, :]
    sin = emb.sin().to(dtype=q.dtype, device=q.device)[None, None, :, :]
    return _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)


def _rotate_half(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)
