from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from torchinferno.models.hf import HF_CONFIG_NAME
from torchinferno.models.llama3_family.config import Llama3Config
from torchinferno.models.llama3_family.pipeline import (
    LLAMA3_70B_REPO_ID,
    _CheckpointTensorLoader,
    _build_inv_freq,
    _resolve_dtype,
    _rms_norm,
    resolve_llama3_checkpoint,
)
from torchinferno.models.llama3_family.v0 import sample_next_token


_COMPILED_ROTATE_LLAMA = None
_COMPILED_ROTATE_LLAMA_CHECKED = False
_COMPILED_ROTATE_LLAMA_FAILED = False


@dataclass(frozen=True)
class Llama3TensorParallelLoadReport:
    checkpoint: str
    dtype: str
    device: str
    rank: int
    world_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
        }


class Llama3TensorParallelLayerKVCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        local_key_value_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, local_key_value_heads, max_seq_len, head_dim)
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


class Llama3TensorParallelCache:
    def __init__(self, layers: list[Llama3TensorParallelLayerKVCache]) -> None:
        self.layers = layers

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0


class _Llama3TensorParallelLayer:
    def __init__(
        self,
        config: Llama3Config,
        layer_id: int,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        weights: dict[str, Tensor],
    ) -> None:
        if config.num_attention_heads % world_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel world size")
        if config.num_key_value_heads % world_size != 0:
            raise ValueError("num_key_value_heads must be divisible by tensor parallel world size")
        if config.intermediate_size % world_size != 0:
            raise ValueError("intermediate_size must be divisible by tensor parallel world size")
        self.config = config
        self.layer_id = layer_id
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_attention_heads = config.num_attention_heads // world_size
        self.local_key_value_heads = config.num_key_value_heads // world_size
        self.local_hidden_size = self.local_attention_heads * config.head_dim
        self.local_key_value_size = self.local_key_value_heads * config.head_dim
        self.local_intermediate_size = config.intermediate_size // world_size
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        self.qkv_proj_weight = weights["self_attn.qkv_proj.weight"]
        self.o_proj_weight = weights["self_attn.o_proj.weight"]
        self.gate_up_proj_weight = weights["mlp.gate_up_proj.weight"]
        self.down_proj_weight = weights["mlp.down_proj.weight"]
        self.inv_freq = _build_inv_freq(config, device)
        self.profile_seconds: dict[str, float] | None = None
        self.profile_counts: dict[str, int] | None = None

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        residual = hidden
        attn_in = self._profile_block(
            "norm.input",
            lambda: _rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._attention(attn_in, positions, rotary, cache)
        residual = hidden
        mlp_in = self._profile_block(
            "norm.post_attention",
            lambda: _rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._mlp(mlp_in)
        return hidden

    def _attention(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        q, k, v = self._profile_block(
            "attention.qkv",
            lambda: self._qkv(hidden, batch, tokens, head_dim),
        )
        q, k = self._profile_block("attention.rotary", lambda: _apply_rotary_cached(q, k, rotary))
        if cache is not None:
            k, v = self._profile_block("attention.cache_append", lambda: cache.append(k, v))

        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        out = self._profile_block(
            "attention.sdp",
            lambda: self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa),
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        projected = self._profile_block("attention.o_proj", lambda: F.linear(out, self.o_proj_weight))
        self._profile_block("attention.all_reduce", lambda: _all_reduce(projected))
        return projected

    def _mlp(self, hidden: Tensor) -> Tensor:
        gate, up = self._profile_block(
            "mlp.gate_up",
            lambda: F.linear(hidden, self.gate_up_proj_weight).split(
                (self.local_intermediate_size, self.local_intermediate_size),
                dim=-1,
            ),
        )
        activated = self._profile_block("mlp.activation", lambda: F.silu(gate) * up)
        projected = self._profile_block("mlp.down", lambda: F.linear(activated, self.down_proj_weight))
        self._profile_block("mlp.all_reduce", lambda: _all_reduce(projected))
        return projected

    def _qkv(self, hidden: Tensor, batch: int, tokens: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        qkv = F.linear(hidden, self.qkv_proj_weight)
        q, k, v = qkv.split(
            (self.local_hidden_size, self.local_key_value_size, self.local_key_value_size),
            dim=-1,
        )
        q = q.view(
            batch,
            tokens,
            self.local_attention_heads,
            head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _scaled_dot_product(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        positions: Tensor,
        device: torch.device,
        enable_gqa: bool,
    ) -> Tensor:
        if q.size(-2) == 1:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        key_positions = torch.arange(k.size(-2), device=device)
        allowed = key_positions[None, :] <= positions[:, None]
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=enable_gqa,
        )

    def _profile_block(self, name: str, fn):
        if self.profile_seconds is None or self.profile_counts is None:
            return fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.profile_seconds[name] = self.profile_seconds.get(name, 0.0) + (time.perf_counter() - start)
        self.profile_counts[name] = self.profile_counts.get(name, 0) + 1
        return result


class Llama3TensorParallelForCausalLM:
    """Tensor-parallel Llama3 inference path launched with torchrun."""

    provenance_variant = "llama3:tp-v0"

    def __init__(
        self,
        config: Llama3Config,
        *,
        embed_tokens_weight: Tensor,
        norm_weight: Tensor,
        lm_head_weight: Tensor,
        layers: list[_Llama3TensorParallelLayer],
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        checkpoint: str | Path,
    ) -> None:
        if config.vocab_size % world_size != 0:
            raise ValueError("vocab_size must be divisible by tensor parallel world size")
        self.config = config
        self.embed_tokens_weight = embed_tokens_weight
        self.norm_weight = norm_weight
        self.lm_head_weight = lm_head_weight
        self.layers = layers
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.devices = (device,)
        self.dtype = dtype
        self.checkpoint = Path(checkpoint)
        self.embed_device = device
        self.output_device = device
        self.local_vocab_size = config.vocab_size // world_size
        self.vocab_start = rank * self.local_vocab_size
        self.inv_freq = _build_inv_freq(config, device)
        self.profile_seconds: dict[str, float] = {}
        self.profile_counts: dict[str, int] = {}
        self.training = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = LLAMA3_70B_REPO_ID,
        *,
        dtype: torch.dtype | str | None = None,
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "Llama3TensorParallelForCausalLM":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        _init_distributed_if_needed()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        root = resolve_llama3_checkpoint(checkpoint, token=token, revision=revision, cache_dir=cache_dir)
        config = Llama3Config.from_dict(json.loads((root / HF_CONFIG_NAME).read_text()))
        torch_dtype = _resolve_dtype(dtype, root)
        loader = _CheckpointTensorLoader(root)
        embed_tokens_weight = loader.get_tensor("model.embed_tokens.weight", device=device, dtype=torch_dtype)
        norm_weight = loader.get_tensor("model.norm.weight", device=device, dtype=torch_dtype)
        lm_head_weight = loader.get_tensor_shard(
            "lm_head.weight",
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=torch_dtype,
        )

        layers: list[_Llama3TensorParallelLayer] = []
        for layer_id in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}."
            q_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.q_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            k_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.k_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            v_proj_weight = loader.get_tensor_shard(
                prefix + "self_attn.v_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            gate_proj_weight = loader.get_tensor_shard(
                prefix + "mlp.gate_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            up_proj_weight = loader.get_tensor_shard(
                prefix + "mlp.up_proj.weight",
                dim=0,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=torch_dtype,
            )
            weights = {
                "input_layernorm.weight": loader.get_tensor(
                    prefix + "input_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                ),
                "post_attention_layernorm.weight": loader.get_tensor(
                    prefix + "post_attention_layernorm.weight",
                    device=device,
                    dtype=torch_dtype,
                ),
                "self_attn.qkv_proj.weight": torch.cat(
                    (q_proj_weight, k_proj_weight, v_proj_weight),
                    dim=0,
                ).contiguous(),
                "self_attn.o_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.o_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "mlp.gate_up_proj.weight": torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous(),
                "mlp.down_proj.weight": loader.get_tensor_shard(
                    prefix + "mlp.down_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
            }
            layers.append(
                _Llama3TensorParallelLayer(
                    config,
                    layer_id,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    weights=weights,
                )
            )
        model = cls(
            config,
            embed_tokens_weight=embed_tokens_weight,
            norm_weight=norm_weight,
            lm_head_weight=lm_head_weight,
            layers=layers,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            dtype=torch_dtype,
            checkpoint=root,
        )
        model.load_report = Llama3TensorParallelLoadReport(
            checkpoint=str(root),
            dtype=str(torch_dtype).replace("torch.", ""),
            device=str(device),
            rank=rank,
            world_size=world_size,
        )
        _barrier()
        return model

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def eval(self) -> "Llama3TensorParallelForCausalLM":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "Llama3TensorParallelForCausalLM":
        self.training = mode
        return self

    def enable_profile(self) -> None:
        self.profile_seconds = {}
        self.profile_counts = {}
        for layer in self.layers:
            layer.profile_seconds = self.profile_seconds
            layer.profile_counts = self.profile_counts

    def disable_profile(self) -> None:
        for layer in self.layers:
            layer.profile_seconds = None
            layer.profile_counts = None

    def profile_summary(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "seconds": dict(sorted(self.profile_seconds.items())),
            "counts": dict(sorted(self.profile_counts.items())),
        }

    def allocate_cache(self, batch_size: int, max_seq_len: int) -> Llama3TensorParallelCache:
        local_kv_heads = self.config.num_key_value_heads // self.world_size
        return Llama3TensorParallelCache(
            [
                Llama3TensorParallelLayerKVCache(
                    batch_size,
                    max_seq_len,
                    local_kv_heads,
                    self.config.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in self.layers
            ]
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Llama3TensorParallelCache | None = None,
        use_cache: bool = True,
        return_last_logits_only: bool = False,
        return_sharded_logits: bool = False,
    ) -> tuple[Tensor, Llama3TensorParallelCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = self.allocate_cache(batch, tokens)
        past_len = active_cache.seq_len if active_cache is not None else 0
        positions = torch.arange(past_len, past_len + tokens, device=self.device)
        rotary = self._rotary_cache(positions)
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        for layer_id, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
            hidden = layer.forward(hidden, positions, rotary, layer_cache)
        if return_last_logits_only:
            hidden = hidden[:, -1:, :]
        hidden = _rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head_weight)
        if return_sharded_logits:
            return logits, active_cache
        logits = self._gather_logits(logits)
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
            return input_ids.to(self.device, non_blocking=True)
        cache = self.allocate_cache(input_ids.size(0), input_ids.size(1) + max_new_tokens)
        logits, cache = self.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        next_token = self._sample_next_token(logits[:, -1, :], temperature)
        output = [input_ids.to(self.device, non_blocking=True), next_token[:, None]]
        for _ in range(1, max_new_tokens):
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self.forward(
                next_token[:, None],
                cache=cache,
                use_cache=True,
                return_last_logits_only=True,
                return_sharded_logits=True,
            )
            next_token = self._sample_next_token(logits[:, -1, :], temperature)
            output.append(next_token[:, None])
        return torch.cat(output, dim=1)

    def _sample_next_token(self, logits: Tensor, temperature: float) -> Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return sample_next_token(logits, temperature).to(self.device)
        if temperature <= 0:
            return self._sample_next_token_greedy(logits)
        return self._sample_next_token_temperature(logits, temperature)

    def _sample_next_token_greedy(self, logits: Tensor) -> Tensor:
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        gathered_values = [torch.empty_like(local_values) for _ in range(self.world_size)]
        gathered_indices = [torch.empty_like(local_indices) for _ in range(self.world_size)]
        dist.all_gather(gathered_values, local_values)
        dist.all_gather(gathered_indices, local_indices)
        next_token = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        if self.is_primary:
            values = torch.stack(gathered_values, dim=0)
            indices = torch.stack(gathered_indices, dim=0)
            best_rank = torch.argmax(values, dim=0)
            row = torch.arange(logits.size(0), device=self.device)
            next_token.copy_(indices[best_rank, row] + best_rank.to(torch.long) * self.local_vocab_size)
        dist.broadcast(next_token, src=0)
        return next_token

    def _sample_next_token_temperature(self, logits: Tensor, temperature: float) -> Tensor:
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        weights = torch.exp(logits_float - global_max[:, None])
        local_sum = weights.sum(dim=-1)
        gathered_sums = [torch.empty_like(local_sum) for _ in range(self.world_size)]
        dist.all_gather(gathered_sums, local_sum)

        selected_rank = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        local_threshold = torch.empty(logits.size(0), dtype=torch.float32, device=self.device)
        if self.is_primary:
            sums = torch.stack(gathered_sums, dim=0)
            cumulative = torch.cumsum(sums, dim=0)
            total = cumulative[-1]
            target = torch.rand_like(total) * total
            selected_rank.copy_((cumulative < target[None, :]).sum(dim=0).to(torch.long))
            row = torch.arange(logits.size(0), device=self.device)
            previous = torch.zeros_like(target)
            has_previous = selected_rank > 0
            previous[has_previous] = cumulative[selected_rank[has_previous] - 1, row[has_previous]]
            local_threshold.copy_(target - previous)
        dist.broadcast(selected_rank, src=0)
        dist.broadcast(local_threshold, src=0)

        cumulative_local = torch.cumsum(weights, dim=-1)
        local_threshold = torch.minimum(local_threshold, cumulative_local[:, -1])
        local_index = (cumulative_local < local_threshold[:, None]).sum(dim=-1).to(torch.long)
        local_index = torch.clamp(local_index, max=self.local_vocab_size - 1)
        selected = selected_rank == self.rank
        local_token = torch.where(
            selected,
            local_index + self.vocab_start,
            torch.zeros_like(local_index),
        )
        dist.all_reduce(local_token, op=dist.ReduceOp.SUM)
        return local_token

    def _gather_logits(self, logits: Tensor) -> Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return logits
        gathered = [torch.empty_like(logits) for _ in range(self.world_size)]
        dist.all_gather(gathered, logits)
        return torch.cat(gathered, dim=-1)

    def _rotary_cache(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=self.dtype, device=self.device)
        sin = emb.sin().to(dtype=self.dtype, device=self.device)
        return cos, sin


def _init_distributed_if_needed() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _all_reduce(tensor: Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _apply_rotary_cached(q: Tensor, k: Tensor, rotary: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    cos, sin = rotary
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return _rotate_llama(q, cos, sin), _rotate_llama(k, cos, sin)


def _rotate_llama(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if x.is_cuda and os.environ.get("TORCHINFERNO_COMPILE_ROTARY", "1") != "0":
        compiled = _load_compiled_rotate_llama()
        if compiled is not None:
            try:
                return compiled(x, cos, sin)
            except Exception:
                global _COMPILED_ROTATE_LLAMA_FAILED
                _COMPILED_ROTATE_LLAMA_FAILED = True
    return _rotate_llama_eager(x, cos, sin)


def _load_compiled_rotate_llama():
    global _COMPILED_ROTATE_LLAMA, _COMPILED_ROTATE_LLAMA_CHECKED
    if not _COMPILED_ROTATE_LLAMA_CHECKED:
        _COMPILED_ROTATE_LLAMA_CHECKED = True
        try:
            _COMPILED_ROTATE_LLAMA = torch.compile(
                _rotate_llama_eager,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        except Exception:
            _COMPILED_ROTATE_LLAMA = None
    if _COMPILED_ROTATE_LLAMA_FAILED:
        return None
    return _COMPILED_ROTATE_LLAMA


def _rotate_llama_eager(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _rotate_interleaved_eager(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    out = torch.empty_like(x)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            dist.barrier()
