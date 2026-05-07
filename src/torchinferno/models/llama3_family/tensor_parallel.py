from __future__ import annotations

import json
import math
import os
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
    _apply_rotary,
    _build_inv_freq,
    _resolve_dtype,
    _rms_norm,
    resolve_llama3_checkpoint,
)
from torchinferno.models.llama3_family.v0 import sample_next_token


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

    def forward(self, hidden: Tensor, positions: Tensor, cache: Llama3TensorParallelLayerKVCache | None) -> Tensor:
        residual = hidden
        attn_in = _rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        hidden = residual + self._attention(attn_in, positions, cache)
        residual = hidden
        mlp_in = _rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        hidden = residual + self._mlp(mlp_in)
        return hidden

    def _attention(self, hidden: Tensor, positions: Tensor, cache: Llama3TensorParallelLayerKVCache | None) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        q = F.linear(hidden, self.q_proj_weight).view(
            batch,
            tokens,
            self.local_attention_heads,
            head_dim,
        ).transpose(1, 2)
        k = F.linear(hidden, self.k_proj_weight).view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        v = F.linear(hidden, self.v_proj_weight).view(
            batch,
            tokens,
            self.local_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        q, k = _apply_rotary(q, k, positions, self.inv_freq)
        if cache is not None:
            k, v = cache.append(k, v)

        repeats = self.local_attention_heads // self.local_key_value_heads
        if repeats != 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
        key_positions = torch.arange(k.size(-2), device=hidden.device)
        allowed = key_positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        projected = F.linear(out, self.o_proj_weight)
        _all_reduce(projected)
        return projected

    def _mlp(self, hidden: Tensor) -> Tensor:
        gate = F.linear(hidden, self.gate_proj_weight)
        up = F.linear(hidden, self.up_proj_weight)
        projected = F.linear(F.silu(gate) * up, self.down_proj_weight)
        _all_reduce(projected)
        return projected


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
        lm_head_weight = loader.get_tensor("lm_head.weight", device=device, dtype=torch_dtype)

        layers: list[_Llama3TensorParallelLayer] = []
        for layer_id in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}."
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
                "self_attn.q_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.q_proj.weight",
                    dim=0,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "self_attn.k_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.k_proj.weight",
                    dim=0,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "self_attn.v_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.v_proj.weight",
                    dim=0,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "self_attn.o_proj.weight": loader.get_tensor_shard(
                    prefix + "self_attn.o_proj.weight",
                    dim=1,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "mlp.gate_proj.weight": loader.get_tensor_shard(
                    prefix + "mlp.gate_proj.weight",
                    dim=0,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
                "mlp.up_proj.weight": loader.get_tensor_shard(
                    prefix + "mlp.up_proj.weight",
                    dim=0,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=torch_dtype,
                ),
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
    ) -> tuple[Tensor, Llama3TensorParallelCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, tokens = input_ids.shape
        active_cache = cache if use_cache else None
        if use_cache and active_cache is None:
            active_cache = self.allocate_cache(batch, tokens)
        past_len = active_cache.seq_len if active_cache is not None else 0
        positions = torch.arange(past_len, past_len + tokens, device=self.device)
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        for layer_id, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
            hidden = layer.forward(hidden, positions, layer_cache)
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
            return input_ids.to(self.device, non_blocking=True)
        cache = self.allocate_cache(input_ids.size(0), input_ids.size(1) + max_new_tokens)
        logits, cache = self.forward(input_ids, cache=cache, use_cache=True, return_last_logits_only=True)
        next_token = self._sample_next_token(logits[:, -1, :], temperature)
        output = [input_ids.to(self.device, non_blocking=True), next_token[:, None]]
        for _ in range(1, max_new_tokens):
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self.forward(next_token[:, None], cache=cache, use_cache=True, return_last_logits_only=True)
            next_token = self._sample_next_token(logits[:, -1, :], temperature)
            output.append(next_token[:, None])
        return torch.cat(output, dim=1)

    def _sample_next_token(self, logits: Tensor, temperature: float) -> Tensor:
        if self.is_primary:
            next_token = sample_next_token(logits, temperature).to(self.device)
        else:
            next_token = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(next_token, src=0)
        return next_token


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


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            dist.barrier()
