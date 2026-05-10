from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from torchinferno.kernels import batched_paged_decode_attention, rms_norm, swiglu_activation
from torchinferno.models.hf import HF_CONFIG_NAME, SAFETENSORS_NAME, load_config, load_state_dict, resolve_pretrained_path
from torchinferno.runtime.paged_attention import batched_paged_causal_attention
from torchinferno.runtime.sampling import sample_next_token


@dataclass
class DeepSeekV32Config:
    """Native DeepSeek-V3.2-style decoder configuration.

    This config models the production tensor contracts that the compact DSv4
    harness deliberately simplified: query LoRA, split RoPE/nope QK heads,
    latent MQA KV projection, dense-to-MoE layer transitions, shared experts,
    grouped routing, and independent value head dimensions.
    """

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "softmax"
    use_score_correction_bias: bool = False
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")
        if self.kv_lora_rank < 1:
            raise ValueError("kv_lora_rank must be positive")
        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.n_group < 1 or self.topk_group < 1 or self.topk_group > self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        if self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive")
        if self.scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError("scoring_func must be 'softmax' or 'sigmoid'")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attention_output_size(self) -> int:
        return self.num_attention_heads * self.v_head_dim

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx < self.first_k_dense_replace:
            return False
        return (layer_idx - self.first_k_dense_replace) % self.moe_layer_freq == 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_type"] = "deepseek_v32"
        data["architectures"] = ["DeepSeekV32ForCausalLM"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DeepSeekV32Config":
        aliases = {
            "num_layers": "num_hidden_layers",
            "n_layers": "num_hidden_layers",
            "num_heads": "num_attention_heads",
            "n_heads": "num_attention_heads",
            "num_routed_experts": "n_routed_experts",
            "num_experts": "n_routed_experts",
            "top_k": "num_experts_per_tok",
            "max_seq_len": "max_position_embeddings",
            "hidden_act": "_unused_hidden_act",
            "attention_bias": "_unused_attention_bias",
        }
        normalized = dict(data)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        if "topk_method" in normalized and normalized.get("topk_method") == "noaux_tc":
            normalized.setdefault("use_score_correction_bias", True)
        if "routed_score_func" in normalized and "scoring_func" not in normalized:
            normalized["scoring_func"] = normalized["routed_score_func"]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in allowed})


def tiny_deepseek_v32_config(**overrides: int | float | bool | str | None) -> DeepSeekV32Config:
    config = DeepSeekV32Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        max_position_embeddings=128,
    )
    return replace(config, **overrides)


class DeepSeekRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        if self.ops is not None:
            return self.ops.rms_norm(x, self.weight, self.eps)
        return rms_norm(x, self.weight, eps=self.eps)


class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float, ops: ModuleType | None = None) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.ops = ops

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(positions.to(self.inv_freq.device).float(), self.inv_freq)
        cos = freqs.cos().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        if self.ops is not None:
            return self.ops.apply_rotary(q, cos, sin), self.ops.apply_rotary(k, cos, sin)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class DeepSeekLayerKVCache:
    cache_backend = "dense"

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.keys = torch.empty((batch_size, num_heads, max_seq_len, qk_head_dim), device=device, dtype=dtype)
        self.values = torch.empty((batch_size, num_heads, max_seq_len, v_head_dim), device=device, dtype=dtype)
        self.seq_lens = [0 for _ in range(batch_size)]
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @property
    def seq_len(self) -> int:
        return self.seq_len_for_rows(tuple(range(self.batch_size)))

    def seq_len_for_rows(self, row_indices: tuple[int, ...]) -> int:
        if not row_indices:
            return 0
        if any(row < 0 or row >= self.batch_size for row in row_indices):
            raise ValueError("cache row out of range")
        seq_len = self.seq_lens[row_indices[0]]
        if any(self.seq_lens[row] != seq_len for row in row_indices):
            raise ValueError("selected cache rows must have the same sequence length")
        return seq_len

    def append(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        row_indices: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, tokens, _ = keys.shape
        row_indices = tuple(range(batch)) if row_indices is None else row_indices
        if batch != len(row_indices):
            raise ValueError("cache row count must match incoming batch")
        if any(row < 0 or row >= self.batch_size for row in row_indices):
            raise ValueError("cache batch is smaller than incoming batch")
        start = self.seq_len_for_rows(row_indices)
        end = start + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        for incoming_row, cache_row in enumerate(row_indices):
            self.keys[cache_row, :, start:end, :].copy_(keys[incoming_row])
            self.values[cache_row, :, start:end, :].copy_(values[incoming_row])
            self.seq_lens[cache_row] = end
        return (
            torch.stack([self.keys[row, :, :end, :] for row in row_indices], dim=0),
            torch.stack([self.values[row, :, :end, :] for row in row_indices], dim=0),
        )

    def copy_prefix_from(
        self,
        source: "DeepSeekLayerKVCache | PagedDeepSeekLayerKVCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if (
            source_row < 0
            or source_row >= source.batch_size
            or dest_row < 0
            or dest_row >= self.batch_size
        ):
            raise ValueError("cache row out of range")
        source_is_dense = isinstance(source, DeepSeekLayerKVCache)
        source_len = source.seq_lens[source_row] if source_is_dense else source.seq_len_for_row(source_row)
        if tokens < 0 or tokens > source_len:
            raise ValueError("tokens must be in the source cache range")
        if tokens > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        if tokens:
            if source_is_dense:
                source_keys = source.keys[source_row, :, :tokens, :]
                source_values = source.values[source_row, :, :tokens, :]
            else:
                source_keys, source_values = source.materialize_row(source_row)
                source_keys = source_keys[:, :tokens, :]
                source_values = source_values[:, :tokens, :]
            self.keys[dest_row, :, :tokens, :].copy_(source_keys)
            self.values[dest_row, :, :tokens, :].copy_(source_values)
        self.seq_lens[dest_row] = tokens

    def clear_row(self, row: int) -> None:
        if row < 0 or row >= self.batch_size:
            raise ValueError("cache row out of range")
        self.seq_lens[row] = 0


class PagedDeepSeekLayerKVCache:
    cache_backend = "paged"

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        *,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be positive")
        from torchinferno.runtime.paged import PagedKVCache

        num_pages = batch_size * math.ceil(max_seq_len / page_size)
        self.pages = PagedKVCache(
            num_pages=num_pages,
            page_size=page_size,
            num_key_value_heads=num_heads,
            head_dim=qk_head_dim,
            value_head_dim=v_head_dim,
            device=device,
            dtype=dtype,
        )
        self.request_ids = tuple(f"batch-{idx}" for idx in range(batch_size))
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @property
    def seq_len(self) -> int:
        return self.seq_len_for_rows(tuple(range(self.batch_size)))

    def seq_len_for_row(self, row: int) -> int:
        if row < 0 or row >= self.batch_size:
            raise ValueError("cache row out of range")
        return self.pages.sequence_length(self.request_ids[row])

    def seq_len_for_rows(self, row_indices: tuple[int, ...]) -> int:
        if not row_indices:
            return 0
        seq_len = self.seq_len_for_row(row_indices[0])
        if any(self.seq_len_for_row(row) != seq_len for row in row_indices):
            raise ValueError("selected cache rows must have the same sequence length")
        return seq_len

    def append(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        row_indices: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        row_indices = self._append_pages(keys, values, row_indices=row_indices)
        return self.materialize(row_indices)

    def append_and_attend(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        positions: Tensor,
        *,
        row_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        row_indices = self._append_pages(keys, values, row_indices=row_indices)
        request_ids = tuple(self.request_ids[row] for row in row_indices)
        if keys.size(2) == 1:
            position = positions[-1].expand(len(row_indices))
            return batched_paged_decode_attention(query, self.pages, request_ids, position)
        return batched_paged_causal_attention(query, self.pages, request_ids, positions)

    def materialize(self, row_indices: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        keys = []
        values = []
        for row in row_indices:
            row_keys, row_values = self.materialize_row(row)
            keys.append(row_keys)
            values.append(row_values)
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def materialize_row(self, row: int) -> tuple[Tensor, Tensor]:
        return self.pages.materialize(self.request_ids[row])

    def copy_prefix_from(
        self,
        source: DeepSeekLayerKVCache | "PagedDeepSeekLayerKVCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        if (
            source_row < 0
            or source_row >= source.batch_size
            or dest_row < 0
            or dest_row >= self.batch_size
        ):
            raise ValueError("cache row out of range")
        if tokens > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        if isinstance(source, PagedDeepSeekLayerKVCache):
            source_len = source.seq_len_for_row(source_row)
            if tokens > source_len:
                raise ValueError("source cache range is invalid")
            if source.pages is self.pages and source_row == dest_row:
                return
            self.clear_row(dest_row)
            if tokens and source.pages is self.pages:
                self.pages.alias_prefix(source.request_ids[source_row], self.request_ids[dest_row], tokens)
                return
            source_keys, source_values = source.materialize_row(source_row)
        else:
            source_len = source.seq_lens[source_row]
            if tokens > source_len:
                raise ValueError("source cache range is invalid")
            source_keys = source.keys[source_row, :, :tokens, :]
            source_values = source.values[source_row, :, :tokens, :]
            self.clear_row(dest_row)
        if tokens:
            self.pages.append(self.request_ids[dest_row], source_keys[:, :tokens, :], source_values[:, :tokens, :])

    def clear_row(self, row: int) -> None:
        if row < 0 or row >= self.batch_size:
            raise ValueError("cache row out of range")
        self.pages.free(self.request_ids[row])

    def _append_pages(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        row_indices: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        batch, _, tokens, _ = keys.shape
        row_indices = tuple(range(batch)) if row_indices is None else row_indices
        if batch != len(row_indices):
            raise ValueError("cache row count must match incoming batch")
        if any(row < 0 or row >= self.batch_size for row in row_indices):
            raise ValueError("cache batch is smaller than incoming batch")
        start = self.seq_len_for_rows(row_indices)
        end = start + tokens
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        for incoming_row, cache_row in enumerate(row_indices):
            self.pages.append(self.request_ids[cache_row], keys[incoming_row], values[incoming_row])
        return row_indices


class DeepSeekCache:
    def __init__(
        self,
        layers: list[DeepSeekLayerKVCache | PagedDeepSeekLayerKVCache],
        *,
        cache_backend: str,
        row_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.layers = layers
        self.cache_backend = cache_backend
        self.row_indices = row_indices

    @property
    def selected_rows(self) -> tuple[int, ...]:
        if self.row_indices is not None:
            return self.row_indices
        if not self.layers:
            return ()
        return tuple(range(self.layers[0].batch_size))

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len_for_rows(self.selected_rows) if self.layers else 0

    @property
    def max_seq_len(self) -> int:
        return self.layers[0].max_seq_len if self.layers else 0

    @property
    def batch_size(self) -> int:
        return len(self.selected_rows)

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > self.max_seq_len:
            raise ValueError("seq_len exceeds KV cache capacity")
        for layer in self.layers:
            if isinstance(layer, PagedDeepSeekLayerKVCache):
                if seq_len != 0:
                    raise ValueError("paged DeepSeek cache seq_len can only be restored through page state")
                for row in self.selected_rows:
                    layer.clear_row(row)
            else:
                for row in self.selected_rows:
                    layer.seq_lens[row] = seq_len

    def reset(self) -> None:
        self.set_seq_len(0)

    def for_rows(self, row_indices: tuple[int, ...] | list[int]) -> "DeepSeekCache":
        rows = tuple(row_indices)
        if self.layers and any(row < 0 or row >= self.layers[0].batch_size for row in rows):
            raise ValueError("cache row out of range")
        return DeepSeekCache(self.layers, cache_backend=self.cache_backend, row_indices=rows)

    def copy_prefix_from(
        self,
        source: "DeepSeekCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if len(self.layers) != len(source.layers):
            raise ValueError("source cache must have the same number of layers")
        if (
            source_row < 0
            or source_row >= len(source.selected_rows)
            or dest_row < 0
            or dest_row >= len(self.selected_rows)
        ):
            raise ValueError("cache row out of range")
        source_physical_row = source.selected_rows[source_row]
        dest_physical_row = self.selected_rows[dest_row]
        for dest_layer, source_layer in zip(self.layers, source.layers):
            dest_layer.copy_prefix_from(
                source_layer,
                tokens,
                source_row=source_physical_row,
                dest_row=dest_physical_row,
            )

    def clear_row(self, row: int) -> None:
        if row < 0 or row >= len(self.selected_rows):
            raise ValueError("cache row out of range")
        physical_row = self.selected_rows[row]
        for layer in self.layers:
            layer.clear_row(physical_row)

    @classmethod
    def allocate(
        cls,
        config: DeepSeekV32Config,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache_backend: str = "dense",
        page_size: int = 16,
    ) -> "DeepSeekCache":
        if cache_backend not in {"dense", "paged"}:
            raise ValueError("cache_backend must be 'dense' or 'paged'")
        layer_cls = PagedDeepSeekLayerKVCache if cache_backend == "paged" else DeepSeekLayerKVCache
        layers = []
        for _ in range(config.num_hidden_layers):
            if layer_cls is PagedDeepSeekLayerKVCache:
                layers.append(
                    PagedDeepSeekLayerKVCache(
                        batch_size,
                        max_seq_len,
                        config.num_attention_heads,
                        config.qk_head_dim,
                        config.v_head_dim,
                        page_size=page_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                layers.append(
                    DeepSeekLayerKVCache(
                        batch_size,
                        max_seq_len,
                        config.num_attention_heads,
                        config.qk_head_dim,
                        config.v_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
        return cls(layers, cache_backend=cache_backend)


def dense_causal_attention(query: Tensor, key: Tensor, value: Tensor, positions: Tensor) -> Tensor:
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
    key_positions = torch.arange(key.size(-2), device=query.device)
    allowed = key_positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    return torch.matmul(probs, value)


class DeepSeekAttention(nn.Module):
    def __init__(self, config: DeepSeekV32Config, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        q_out = config.num_attention_heads * config.qk_head_dim
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, q_out, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_layernorm = DeepSeekRMSNorm(config.q_lora_rank, config.rms_norm_eps, ops)
            self.q_b_proj = nn.Linear(config.q_lora_rank, q_out, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(config.kv_lora_rank, config.rms_norm_eps, ops)
        kv_b_out = config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, kv_b_out, bias=False)
        self.o_proj = nn.Linear(config.attention_output_size, config.hidden_size, bias=False)
        self.rotary_emb = DeepSeekRotaryEmbedding(config.qk_rope_head_dim, config.rope_theta, ops)

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        cache: Optional[DeepSeekLayerKVCache | PagedDeepSeekLayerKVCache],
        row_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        batch, tokens, _ = hidden_states.shape
        heads = self.config.num_attention_heads

        if self.config.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch, tokens, heads, self.config.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_pe = torch.split(kv_a, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv = self.kv_b_proj(kv_latent).view(
            batch,
            tokens,
            heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        )
        k_nope, value = torch.split(kv, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)
        k_pe = k_pe[:, None, :, :]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe, positions)
        key = torch.cat([k_nope, k_pe.expand(-1, heads, -1, -1)], dim=-1)
        query = torch.cat([q_nope, q_pe], dim=-1)

        if isinstance(cache, PagedDeepSeekLayerKVCache):
            out = cache.append_and_attend(query, key, value, positions, row_indices=row_indices)
        else:
            if cache is not None:
                key, value = cache.append(key, value, row_indices=row_indices)
            out = (
                self.ops.causal_attention(query, key, value, positions)
                if self.ops is not None
                else dense_causal_attention(query, key, value, positions)
            )

        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.config.attention_output_size)
        return self.o_proj(out)


class DeepSeekMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        if self.ops is not None:
            return self.down_proj(self.ops.swiglu(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(swiglu_activation(self.gate_proj(x), self.up_proj(x)))


class DeepSeekMoEGate(nn.Module):
    def __init__(self, config: DeepSeekV32Config, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        if config.use_score_correction_bias:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        else:
            self.register_parameter("e_score_correction_bias", None)
        self.ops = ops
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.matmul(x, self.weight.t())
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)

        selection_scores = scores
        if self.e_score_correction_bias is not None:
            selection_scores = selection_scores + self.e_score_correction_bias
        if self.ops is not None:
            _, top_indices = self.ops.grouped_topk(selection_scores, self.n_group, self.topk_group, self.top_k)
        else:
            if self.n_group > 1 and self.topk_group < self.n_group:
                group_size = selection_scores.size(-1) // self.n_group
                grouped = selection_scores.view(selection_scores.size(0), self.n_group, group_size)
                group_scores = grouped.max(dim=-1).values
                group_idx = torch.topk(group_scores, self.topk_group, dim=-1).indices
                group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
                group_mask.scatter_(1, group_idx, True)
                expert_mask = group_mask[:, :, None].expand(-1, -1, group_size).reshape_as(selection_scores)
                selection_scores = selection_scores.masked_fill(~expert_mask, torch.finfo(selection_scores.dtype).min)

            top_indices = torch.topk(selection_scores, self.top_k, dim=-1).indices
        top_weights = scores.gather(1, top_indices)
        if self.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        top_weights = top_weights * self.routed_scaling_factor
        return top_weights.to(dtype=x.dtype), top_indices


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekV32Config, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.gate = DeepSeekMoEGate(config, ops)
        self.experts = nn.ModuleList(
            [DeepSeekMLP(config.hidden_size, config.moe_intermediate_size, ops) for _ in range(config.n_routed_experts)]
        )
        shared_size = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = DeepSeekMLP(config.hidden_size, shared_size, ops) if shared_size > 0 else None
        self.top_k = config.num_experts_per_tok

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        top_weights, top_indices = self.gate(flat)
        output = torch.zeros_like(flat)
        for slot in range(self.top_k):
            slot_indices = top_indices[:, slot]
            slot_weights = top_weights[:, slot]
            for expert_id, expert in enumerate(self.experts):
                mask = slot_indices == expert_id
                output[mask] += expert(flat[mask]) * slot_weights[mask, None]
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat)
        return output.view(original_shape)


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV32Config, layer_idx: int, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps, ops)
        self.self_attn = DeepSeekAttention(config, ops)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps, ops)
        if config.is_moe_layer(layer_idx):
            self.mlp = DeepSeekMoE(config, ops)
        else:
            self.mlp = DeepSeekMLP(config.hidden_size, config.intermediate_size, ops)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: Optional[DeepSeekLayerKVCache | PagedDeepSeekLayerKVCache],
        row_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, cache, row_indices)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepSeekV32Model(nn.Module):
    def __init__(self, config: DeepSeekV32Config, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekDecoderLayer(config, layer_idx, ops) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps, ops)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Optional[DeepSeekCache],
        use_cache: bool,
    ) -> tuple[Tensor, Optional[DeepSeekCache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        if tokens < 1:
            raise ValueError("input_ids must contain at least one token")
        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = DeepSeekCache.allocate(
                self.config,
                batch,
                self.config.max_position_embeddings,
                device=input_ids.device,
                dtype=self.embed_tokens.weight.dtype,
                cache_backend="dense",
            )
        past_len = active_cache.seq_len if active_cache is not None else 0
        row_indices = active_cache.selected_rows if active_cache is not None else None
        if row_indices is not None and len(row_indices) != batch:
            raise ValueError("cache row selection must match input batch size")
        if past_len + tokens > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds configured max_position_embeddings")

        positions = torch.arange(past_len, past_len + tokens, device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_idx] if active_cache is not None else None
            hidden_states = layer(hidden_states, positions, layer_cache, row_indices)
        return self.norm(hidden_states), active_cache


class DeepSeekV32ForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV32Config, ops: ModuleType | None = None) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        self.model = DeepSeekV32Model(config, ops)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
    ) -> "DeepSeekV32ForCausalLM":
        path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
        config = DeepSeekV32Config.from_dict(load_config(path))
        model = cls(config)
        state_dict = load_state_dict(path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        return model

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        (path / HF_CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        state_dict = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        save_file(state_dict, path / SAFETENSORS_NAME, metadata={"format": "pt"})

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_backend: str = "dense",
        page_size: int = 16,
    ) -> DeepSeekCache:
        max_seq_len = self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        device = self.model.embed_tokens.weight.device if device is None else device
        dtype = self.model.embed_tokens.weight.dtype if dtype is None else dtype
        return DeepSeekCache.allocate(
            self.config,
            batch_size,
            max_seq_len,
            device=device,
            dtype=dtype,
            cache_backend=cache_backend,
            page_size=page_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Optional[DeepSeekCache] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[DeepSeekCache]]:
        hidden_states, cache = self.model(input_ids, cache=cache, use_cache=use_cache)
        return self.lm_head(hidden_states), cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        cache_backend: str = "dense",
        page_size: int = 16,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids
        was_training = self.training
        self.eval()
        try:
            cache = self.allocate_cache(
                input_ids.size(0),
                min(self.config.max_position_embeddings, input_ids.size(1) + max_new_tokens),
                device=input_ids.device,
                cache_backend=cache_backend,
                page_size=page_size,
            )
            logits, cache = self(input_ids, cache=cache, use_cache=True)
            next_token = sample_next_token(logits[:, -1, :], temperature)
            output = [input_ids, next_token[:, None]]
            for _ in range(1, max_new_tokens):
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                logits, cache = self(next_token[:, None], cache=cache, use_cache=True)
                next_token = sample_next_token(logits[:, -1, :], temperature)
                output.append(next_token[:, None])
            return torch.cat(output, dim=1)
        finally:
            if was_training:
                self.train()
