from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from torchinferno.kernels import rms_norm, swiglu_activation


@dataclass
class DSv4Config:
    """Configuration for a compact DeepSeek-style decoder-only model.

    The implementation is intentionally torch-native and small enough to run as
    a local harness. It keeps the architectural pieces that matter for inference
    work: MLA-like latent KV projection, rotary attention, routed MoE blocks, and
    an explicit paged-cache-shaped KV object.
    """

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    latent_kv_size: int = 512
    intermediate_size: int = 11008
    num_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_type"] = "dsv4"
        data["architectures"] = ["DSv4ForCausalLM"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DSv4Config":
        aliases = {
            "num_hidden_layers": "num_layers",
            "num_heads": "num_attention_heads",
            "n_heads": "num_attention_heads",
            "num_kv_heads": "num_key_value_heads",
            "kv_lora_rank": "latent_kv_size",
            "moe_intermediate_size": "intermediate_size",
            "n_routed_experts": "num_experts",
            "num_routed_experts": "num_experts",
            "num_experts_per_tok": "top_k",
            "max_position_embeddings": "max_seq_len",
        }
        normalized = dict(data)
        is_deepseek_v32 = normalized.get("model_type") == "deepseek_v32"
        for source, target in aliases.items():
            if source in normalized and (target not in normalized or is_deepseek_v32):
                normalized[target] = normalized[source]
        if is_deepseek_v32 and "moe_intermediate_size" in normalized:
            normalized["intermediate_size"] = normalized["moe_intermediate_size"]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in allowed})


def tiny_dsv4_config(**overrides: int | float | bool) -> DSv4Config:
    """Return a tiny DSv4 config for smoke tests and local iteration."""

    config = DSv4Config(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        latent_kv_size=32,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        max_seq_len=128,
    )
    return replace(config, **overrides)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, eps=self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        if positions.numel() == 0:
            return q, k

        freqs = torch.outer(positions.to(self.inv_freq.device).float(), self.inv_freq)
        cos = freqs.cos().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)


class LayerKVCache:
    """Append-only KV cache for one decoder layer.

    The tensor layout is compatible with future paged attention work:
    [batch, kv_head, sequence, head_dim].
    """

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


class DSv4Cache:
    def __init__(self, layers: list[LayerKVCache]) -> None:
        self.layers = layers

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    @classmethod
    def allocate(
        cls,
        config: DSv4Config,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DSv4Cache":
        return cls(
            [
                LayerKVCache(
                    batch_size,
                    max_seq_len,
                    config.num_key_value_heads,
                    config.head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )


class DSv4Attention(nn.Module):
    def __init__(self, config: DSv4Config) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.kv_a_proj = nn.Linear(config.hidden_size, config.latent_kv_size, bias=False)
        self.kv_a_norm = RMSNorm(config.latent_kv_size, config.rms_norm_eps)
        kv_out = 2 * config.num_key_value_heads * config.head_dim
        self.kv_b_proj = nn.Linear(config.latent_kv_size, kv_out, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: Tensor, positions: Tensor, cache: Optional[LayerKVCache]) -> Tensor:
        batch, tokens, _ = x.shape
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = self.q_proj(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
        latent_kv = self.kv_a_norm(self.kv_a_proj(x))
        kv = self.kv_b_proj(latent_kv).view(batch, tokens, kv_heads, 2, head_dim)
        k = kv[:, :, :, 0, :].transpose(1, 2)
        v = kv[:, :, :, 1, :].transpose(1, 2)
        q, k = self.rope(q, k, positions)

        if cache is not None:
            k, v = cache.append(k, v)

        if kv_heads != heads:
            repeats = heads // kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        key_positions = torch.arange(k.size(-2), device=x.device)
        allowed = key_positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.config.hidden_size)
        return self.o_proj(out)


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(swiglu_activation(self.w1(x), self.w3(x)))


class RoutedMoE(nn.Module):
    def __init__(self, config: DSv4Config) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_weights, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_weights.float(), dim=-1).to(dtype=flat.dtype)
        output = torch.zeros_like(flat)

        for slot in range(self.top_k):
            slot_indices = top_indices[:, slot]
            slot_weights = top_weights[:, slot]
            for expert_id, expert in enumerate(self.experts):
                mask = slot_indices == expert_id
                output[mask] += expert(flat[mask]) * slot_weights[mask, None]

        return output.view(original_shape)


class DSv4DecoderLayer(nn.Module):
    def __init__(self, config: DSv4Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = DSv4Attention(config)
        self.moe_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = RoutedMoE(config)

    def forward(self, x: Tensor, positions: Tensor, cache: Optional[LayerKVCache]) -> Tensor:
        x = x + self.attn(self.attn_norm(x), positions, cache)
        x = x + self.moe(self.moe_norm(x))
        return x


class DSv4ForCausalLM(nn.Module):
    def __init__(self, config: DSv4Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DSv4DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "DSv4ForCausalLM":
        from torchinferno.models.hf import load_dsv4_pretrained

        return load_dsv4_pretrained(
            cls,
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            map_location=map_location,
            strict=strict,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        from torchinferno.models.hf import save_dsv4_pretrained

        save_dsv4_pretrained(self, save_directory)

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DSv4Cache:
        max_seq_len = self.config.max_seq_len if max_seq_len is None else max_seq_len
        device = self.embed_tokens.weight.device if device is None else device
        dtype = self.embed_tokens.weight.dtype if dtype is None else dtype
        return DSv4Cache.allocate(self.config, batch_size, max_seq_len, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Optional[DSv4Cache] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[DSv4Cache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        if tokens < 1:
            raise ValueError("input_ids must contain at least one token")

        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = self.allocate_cache(batch, self.config.max_seq_len, device=input_ids.device)
        past_len = active_cache.seq_len if active_cache is not None else 0
        if past_len + tokens > self.config.max_seq_len:
            raise ValueError("input sequence exceeds configured max_seq_len")

        positions = torch.arange(past_len, past_len + tokens, device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer_id, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
            x = layer(x, positions, layer_cache)
        logits = self.lm_head(self.norm(x))
        return logits, active_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids

        was_training = self.training
        self.eval()
        cache = self.allocate_cache(
            input_ids.size(0),
            min(self.config.max_seq_len, input_ids.size(1) + max_new_tokens),
            device=input_ids.device,
        )
        logits, cache = self(input_ids, cache=cache, use_cache=True)
        next_token = sample_next_token(logits[:, -1, :], temperature)
        output = [input_ids, next_token[:, None]]

        for step in range(1, max_new_tokens):
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self(next_token[:, None], cache=cache, use_cache=True)
            next_token = sample_next_token(logits[:, -1, :], temperature)
            output.append(next_token[:, None])

        if was_training:
            self.train()
        return torch.cat(output, dim=1)


def sample_next_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
