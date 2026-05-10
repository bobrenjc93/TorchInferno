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
    _rms_norm as _torch_rms_norm,
    resolve_llama3_checkpoint,
)
from torchinferno.runtime.options import env_flag, env_int, warn_optional_failure
from torchinferno.runtime.sampling import sample_next_token


_COMPILED_ROTATE_LLAMA = None
_COMPILED_ROTATE_LLAMA_CHECKED = False
_COMPILED_ROTATE_LLAMA_FAILED = False
_SYMM_REDUCE_BUFFERS: dict[tuple[str, int, str, str, tuple[int, ...]], Tensor] = {}
_SYMM_REDUCE_PROBED: set[tuple[str, int, str, str, tuple[int, ...]]] = set()
_SYMM_REDUCE_DISABLED = False
_DEFAULT_DECODE_STEP_MAX_BATCH = 16


def _tp_flag(name: str, default: bool = True) -> bool:
    return env_flag(name, default)


def _tp_int(name: str, default: int, *, minimum: int | None = None) -> int:
    return env_int(name, default, minimum=minimum)


def _tp_env_set(name: str) -> bool:
    return name in os.environ


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


@dataclass
class _StaticCudaGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    output: Tensor


@dataclass
class _StaticQKVRotaryGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    static_cos: Tensor
    static_sin: Tensor
    q: Tensor
    k: Tensor
    v: Tensor


@dataclass
class _StaticPrefillActivationGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input: Tensor
    output: Tensor


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
        if keys.is_cuda and values.is_cuda and _tp_flag("TORCHINFERNO_TRITON_KV_APPEND"):
            try:
                from torchinferno.kernels.triton_ops import triton_append_kv_cache

                triton_append_kv_cache(keys, values, self.keys, self.values, self.seq_len)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.triton_kv_append", exc)
                self.keys[:batch, :, self.seq_len : end, :].copy_(keys)
                self.values[:batch, :, self.seq_len : end, :].copy_(values)
        else:
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

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        for layer in self.layers:
            if seq_len > layer.max_seq_len:
                raise ValueError("seq_len exceeds KV cache capacity")
        for layer in self.layers:
            layer.seq_len = seq_len

    def reset(self) -> None:
        self.set_seq_len(0)


@dataclass
class _StaticDecodeGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_position: Tensor
    static_attention_length: Tensor
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_token: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    attention_block_size: int


@dataclass
class _StaticDecodeLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_cache_position: Tensor
    static_attention_length: Tensor
    static_rotary_cos: Tensor
    static_rotary_sin: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    max_seq_len: int
    attention_block_size: int


@dataclass
class _StaticPrefillGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    output_token: Tensor
    cache: Llama3TensorParallelCache
    prompt_tokens: int
    initial_seq_len: int
    max_seq_len: int


@dataclass
class _StaticPrefillLogitsGraphCall:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    output_logits: Tensor
    cache: Llama3TensorParallelCache
    prompt_tokens: int
    initial_seq_len: int
    max_seq_len: int


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
        self.qkv_proj_weight_decode = _maybe_decode_weight_t(self.qkv_proj_weight)
        self.o_proj_weight_decode = _maybe_decode_weight_t(self.o_proj_weight)
        self.gate_up_proj_weight_decode = _maybe_decode_weight_t(self.gate_up_proj_weight)
        self.down_proj_weight_decode = _maybe_decode_weight_t(self.down_proj_weight)
        self.inv_freq = _build_inv_freq(config, device)
        self.profile_seconds: dict[str, float] | None = None
        self.profile_counts: dict[str, int] | None = None
        self._mlp_project_graph: _StaticCudaGraphCall | None = None
        self._mlp_project_graph_failed = False
        self._qkv_rotary_graphs: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            _StaticQKVRotaryGraphCall,
        ] = {}
        self._qkv_rotary_graph_failed = False
        self._input_qkv_rotary_graphs: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            _StaticQKVRotaryGraphCall,
        ] = {}
        self._input_qkv_rotary_graph_failed = False
        self._post_mlp_project_graph: _StaticCudaGraphCall | None = None
        self._post_mlp_project_graph_failed = False
        self._attention_o_graph: _StaticCudaGraphCall | None = None
        self._attention_o_graph_failed = False
        self._prefill_gate_up_activation_graphs: dict[
            tuple[int, ...],
            _StaticPrefillActivationGraphCall,
        ] = {}
        self._prefill_gate_up_activation_graph_failed = False
        self._symm_reduce_failed = False

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        if self.profile_seconds is None or self.profile_counts is None:
            residual = hidden
            hidden = residual + self._attention_from_hidden(hidden, positions, rotary, cache)
            residual = hidden
            return residual + self._post_attention_mlp_project(hidden)

        residual = hidden
        attn_in = self._profile_block(
            "norm.input",
            lambda: _tp_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._attention(attn_in, positions, rotary, cache)
        residual = hidden
        mlp_in = self._profile_block(
            "norm.post_attention",
            lambda: _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps),
        )
        hidden = residual + self._mlp(mlp_in)
        return hidden

    def forward_prefill_fast(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
        next_norm_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden
        attention = self._profile_block(
            "fast_prefill.attention",
            lambda: self._attention_from_hidden(hidden, positions, rotary, cache, attn_in=attn_in),
        )
        hidden, mlp_in = self._profile_block(
            "fast_prefill.post_attention_add_norm",
            lambda: _tp_decode_add_rms_norm(
                attention,
                residual,
                self.post_attention_layernorm_weight,
                self.config.rms_norm_eps,
            ),
        )

        def project_mlp() -> Tensor:
            projected = self._mlp_project_prefill_reduce(mlp_in)
            if projected is None:
                projected = self._mlp_project_eager(mlp_in)
                _all_reduce(projected)
            return projected

        projected = self._profile_block("fast_prefill.mlp_project", project_mlp)
        if next_norm_weight is None:
            return hidden + projected, None
        return self._profile_block(
            "fast_prefill.next_input_add_norm",
            lambda: _tp_decode_add_rms_norm(projected, hidden, next_norm_weight, self.config.rms_norm_eps),
        )

    def _attention(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        if self.profile_seconds is None or self.profile_counts is None:
            q, k, v = self._qkv_rotary(hidden, batch, tokens, head_dim, rotary)
            if cache is not None:
                k, v = cache.append(k, v)
            enable_gqa = self.local_attention_heads != self.local_key_value_heads
            out = self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa)
            out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
            return self._attention_o_project_reduce(out)

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

    def _attention_from_hidden(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache | None,
        *,
        attn_in: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        head_dim = self.config.head_dim
        if attn_in is None:
            q, k, v = self._profile_block(
                "fast_prefill.attention.input_norm_qkv_rotary",
                lambda: self._input_norm_qkv_rotary(hidden, batch, tokens, head_dim, rotary),
            )
        else:
            q, k, v = self._profile_block(
                "fast_prefill.attention.qkv_rotary",
                lambda: self._qkv_rotary(attn_in, batch, tokens, head_dim, rotary),
            )
        if cache is not None:
            k, v = self._profile_block("fast_prefill.attention.cache_append", lambda: cache.append(k, v))
        enable_gqa = self.local_attention_heads != self.local_key_value_heads
        out = self._profile_block(
            "fast_prefill.attention.sdp",
            lambda: self._scaled_dot_product(q, k, v, positions, hidden.device, enable_gqa=enable_gqa),
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._profile_block(
            "fast_prefill.attention.o_project_reduce",
            lambda: self._attention_o_project_reduce(out),
        )

    def forward_decode_static(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_position: Tensor,
        attention_length: Tensor,
        attention_block_size: int | None,
        next_norm_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = hidden
        attention = self._attention_decode_static(
            hidden,
            attn_in,
            rotary,
            cache,
            cache_position,
            attention_length,
            attention_block_size,
        )
        hidden, mlp_in = _tp_decode_add_rms_norm(
            attention,
            residual,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        residual = hidden
        projected = self._mlp_project_decode_reduce(mlp_in)
        return _tp_decode_add_rms_norm(projected, residual, next_norm_weight, self.config.rms_norm_eps)

    def _attention_decode_static(
        self,
        hidden: Tensor,
        attn_in: Tensor | None,
        rotary: tuple[Tensor, Tensor],
        cache: Llama3TensorParallelLayerKVCache,
        cache_position: Tensor,
        attention_length: Tensor,
        attention_block_size: int | None,
    ) -> Tensor:
        from torchinferno.kernels.triton_ops import (
            triton_apply_rotary_append_kv_decode,
            triton_append_kv_cache,
            triton_dense_gqa_decode_attention,
            triton_grouped_gqa_decode_attention,
        )

        batch, tokens, _ = hidden.shape
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        if _tp_flag("TORCHINFERNO_TRITON_DECODE_ROTARY_APPEND"):
            q, k, v = self._qkv(attn_in, batch, tokens, self.config.head_dim)
            q = triton_apply_rotary_append_kv_decode(q, k, v, cache.keys, cache.values, cache_position, rotary[0], rotary[1])
        else:
            q, k, v = self._qkv_rotary_eager(attn_in, batch, tokens, self.config.head_dim, rotary)
            triton_append_kv_cache(k, v, cache.keys, cache.values, cache_position)
        attention_keys = cache.keys
        attention_values = cache.values
        if attention_block_size is not None and attention_block_size < cache.keys.size(2):
            attention_keys = cache.keys[:, :, :attention_block_size, :]
            attention_values = cache.values[:, :, :attention_block_size, :]
        if (
            self.local_attention_heads > self.local_key_value_heads
            and _tp_flag("TORCHINFERNO_TRITON_GROUPED_DECODE_ATTENTION")
        ):
            out = triton_grouped_gqa_decode_attention(q, attention_keys, attention_values, attention_length)
        else:
            out = triton_dense_gqa_decode_attention(q, attention_keys, attention_values, seq_len=attention_length)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.local_hidden_size)
        return self._decode_linear_all_reduce(out, self.o_proj_weight, "attention", self.o_proj_weight_decode)

    def _mlp_project_decode_reduce(self, hidden: Tensor) -> Tensor:
        gate, up = _decode_linear(hidden, self.gate_up_proj_weight, self.gate_up_proj_weight_decode).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        activated = _tp_decode_swiglu(gate, up)
        return self._decode_linear_all_reduce(activated, self.down_proj_weight, "mlp", self.down_proj_weight_decode)

    def _mlp_project_prefill_reduce(self, hidden: Tensor) -> Tensor | None:
        if not _should_use_symm_mem_prefill_all_reduce(hidden, self.down_proj_weight, self.world_size):
            return None
        activated = self._profile_block(
            "fast_prefill.mlp_prefill.gate_up_activation",
            lambda: self._prefill_gate_up_activation(hidden),
        )
        return self._prefill_linear_all_reduce(activated, self.down_proj_weight, "mlp-prefill")

    def _prefill_gate_up_activation(self, hidden: Tensor) -> Tensor:
        if (
            self.world_size > 1
            and _should_use_prefill_gate_up_activation_graph(hidden)
            and not self._prefill_gate_up_activation_graph_failed
        ):
            try:
                return self._run_prefill_gate_up_activation_graph(hidden)
            except Exception:
                self._prefill_gate_up_activation_graph_failed = True
        gate, up = F.linear(hidden, self.gate_up_proj_weight).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        return _tp_swiglu(gate, up)

    def _run_prefill_gate_up_activation_graph(self, hidden: Tensor) -> Tensor:
        key = tuple(hidden.shape)
        captured = self._prefill_gate_up_activation_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._prefill_gate_up_activation_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._prefill_gate_up_activation_eager(static_input)
            captured = _StaticPrefillActivationGraphCall(graph=graph, static_input=static_input, output=output)
            self._prefill_gate_up_activation_graphs[key] = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _prefill_gate_up_activation_eager(self, hidden: Tensor) -> Tensor:
        gate, up = F.linear(hidden, self.gate_up_proj_weight).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        return _tp_swiglu(gate, up)

    def _decode_linear_all_reduce(
        self,
        hidden: Tensor,
        weight: Tensor,
        buffer_name: str,
        weight_t: Tensor | None = None,
    ) -> Tensor:
        if _should_use_symm_mem_all_reduce(hidden, weight, self.world_size) and not self._symm_reduce_failed:
            try:
                expected_shape = (1, 1, weight.size(0))
                buffer, group_name = self._symm_reduce_buffer(buffer_name, hidden, expected_shape)
                if weight_t is not None:
                    torch.mm(hidden.reshape(1, -1), weight_t, out=buffer.reshape(1, -1))
                else:
                    torch.mv(weight, hidden.reshape(-1), out=buffer.view(-1))
                torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
                return buffer
            except Exception:
                self._symm_reduce_failed = True
                _disable_symm_reduce()
        projected = _decode_linear(hidden, weight, weight_t)
        _all_reduce(projected)
        return projected

    def _prefill_linear_all_reduce(self, hidden: Tensor, weight: Tensor, buffer_name: str) -> Tensor | None:
        if not _should_use_symm_mem_prefill_all_reduce(hidden, weight, self.world_size):
            return None
        try:
            expected_shape = (*hidden.shape[:-1], weight.size(0))
            buffer, group_name = self._symm_reduce_buffer(buffer_name, hidden, expected_shape)
            self._profile_block(
                f"fast_prefill.{buffer_name}.mm",
                lambda: torch.mm(hidden.reshape(-1, hidden.size(-1)), weight.t(), out=buffer.reshape(-1, weight.size(0))),
            )
            self._profile_block(
                f"fast_prefill.{buffer_name}.all_reduce",
                lambda: torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name),
            )
            return buffer
        except Exception:
            _disable_symm_reduce()
            return None

    def _symm_reduce_buffer(self, name: str, hidden: Tensor, expected_shape: tuple[int, ...]) -> tuple[Tensor, str]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("symmetric-memory allreduce requires an initialized process group")
        group_name = dist.group.WORLD.group_name
        device_index = hidden.device.index if hidden.device.index is not None else torch.cuda.current_device()
        key = (group_name, device_index, name, str(hidden.dtype), expected_shape)
        buffer = _SYMM_REDUCE_BUFFERS.get(key)
        if buffer is None:
            import torch.distributed._symmetric_memory as symm_mem

            buffer = symm_mem.empty(expected_shape, device=hidden.device, dtype=hidden.dtype)
            symm_mem.rendezvous(buffer, group_name)
            _SYMM_REDUCE_BUFFERS[key] = buffer
        if key not in _SYMM_REDUCE_PROBED:
            self._probe_symm_reduce_buffer(key, buffer, group_name)
        return buffer, group_name

    def _probe_symm_reduce_buffer(
        self,
        key: tuple[str, int, str, str, tuple[int, ...]],
        buffer: Tensor,
        group_name: str,
    ) -> None:
        buffer.zero_()
        torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        torch.cuda.synchronize(buffer.device)
        _SYMM_REDUCE_PROBED.add(key)

    def _mlp(self, hidden: Tensor) -> Tensor:
        if self.profile_seconds is None or self.profile_counts is None:
            projected = self._mlp_project(hidden)
            _all_reduce(projected)
            return projected

        gate, up = self._profile_block(
            "mlp.gate_up",
            lambda: F.linear(hidden, self.gate_up_proj_weight).split(
                (self.local_intermediate_size, self.local_intermediate_size),
                dim=-1,
            ),
        )
        activated = self._profile_block("mlp.activation", lambda: _tp_swiglu(gate, up))
        projected = self._profile_block("mlp.down", lambda: F.linear(activated, self.down_proj_weight))
        self._profile_block("mlp.all_reduce", lambda: _all_reduce(projected))
        return projected

    def _mlp_project(self, hidden: Tensor) -> Tensor:
        if self.world_size > 1 and _should_use_mlp_project_graph(hidden) and not self._mlp_project_graph_failed:
            try:
                return self._run_mlp_project_graph(hidden)
            except Exception:
                self._mlp_project_graph_failed = True
        return self._mlp_project_eager(hidden)

    def _mlp_project_eager(self, hidden: Tensor) -> Tensor:
        gate, up = _decode_linear(hidden, self.gate_up_proj_weight, self.gate_up_proj_weight_decode).split(
            (self.local_intermediate_size, self.local_intermediate_size),
            dim=-1,
        )
        activated = _tp_swiglu(gate, up)
        return _decode_linear(activated, self.down_proj_weight, self.down_proj_weight_decode)

    def _run_mlp_project_graph(self, hidden: Tensor) -> Tensor:
        captured = self._mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._mlp_project_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._mlp_project_eager(static_input)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _post_attention_mlp_project(self, hidden: Tensor) -> Tensor:
        mlp_in = _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        reduced = self._mlp_project_prefill_reduce(mlp_in)
        if reduced is not None:
            return reduced
        if _should_graph_all_reduce() and self.world_size > 1 and _should_use_mlp_project_graph(hidden):
            try:
                return self._run_post_mlp_project_reduce_graph(hidden)
            except Exception:
                self._post_mlp_project_graph_failed = True
        if self.world_size > 1 and _should_use_mlp_project_graph(hidden) and not self._post_mlp_project_graph_failed:
            try:
                projected = self._run_post_mlp_project_graph(hidden)
                _all_reduce(projected)
                return projected
            except Exception:
                self._post_mlp_project_graph_failed = True
        projected = self._mlp_project_eager(mlp_in)
        _all_reduce(projected)
        return projected

    def _run_post_mlp_project_reduce_graph(self, hidden: Tensor) -> Tensor:
        captured = self._post_mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                projected = self._post_mlp_project_eager(static_input)
                _all_reduce(projected)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._post_mlp_project_eager(static_input)
                _all_reduce(output)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._post_mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _post_mlp_project_eager(self, hidden: Tensor) -> Tensor:
        mlp_in = _tp_rms_norm(hidden, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        return self._mlp_project_eager(mlp_in)

    def _run_post_mlp_project_graph(self, hidden: Tensor) -> Tensor:
        captured = self._post_mlp_project_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._post_mlp_project_eager(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._post_mlp_project_eager(static_input)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._post_mlp_project_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _attention_o_project(self, hidden: Tensor) -> Tensor:
        if self.world_size > 1 and _should_use_attention_o_graph(hidden) and not self._attention_o_graph_failed:
            try:
                return self._run_attention_o_graph(hidden)
            except Exception:
                self._attention_o_graph_failed = True
        return _decode_linear(hidden, self.o_proj_weight, self.o_proj_weight_decode)

    def _run_attention_o_graph(self, hidden: Tensor) -> Tensor:
        captured = self._attention_o_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._attention_o_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _attention_o_project_reduce(self, hidden: Tensor) -> Tensor:
        reduced = self._prefill_linear_all_reduce(hidden, self.o_proj_weight, "attention-prefill")
        if reduced is not None:
            return reduced
        if _should_graph_all_reduce() and self.world_size > 1 and _should_use_attention_o_graph(hidden):
            try:
                return self._run_attention_o_reduce_graph(hidden)
            except Exception:
                self._attention_o_graph_failed = True
        projected = self._attention_o_project(hidden)
        _all_reduce(projected)
        return projected

    def _run_attention_o_reduce_graph(self, hidden: Tensor) -> Tensor:
        captured = self._attention_o_graph
        if captured is None or captured.static_input.shape != hidden.shape:
            static_input = torch.empty_like(hidden)
            static_input.copy_(hidden)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                projected = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
                _all_reduce(projected)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _decode_linear(static_input, self.o_proj_weight, self.o_proj_weight_decode)
                _all_reduce(output)
            captured = _StaticCudaGraphCall(graph=graph, static_input=static_input, output=output)
            self._attention_o_graph = captured
            captured.graph.replay()
            return captured.output
        captured.static_input.copy_(hidden)
        captured.graph.replay()
        return captured.output

    def _qkv(self, hidden: Tensor, batch: int, tokens: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        qkv = _decode_linear(hidden, self.qkv_proj_weight, self.qkv_proj_weight_decode)
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

    def _qkv_rotary(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.world_size > 1 and _should_use_qkv_rotary_graph(hidden) and not self._qkv_rotary_graph_failed:
            try:
                return self._run_qkv_rotary_graph(hidden, batch, tokens, head_dim, rotary)
            except Exception:
                self._qkv_rotary_graph_failed = True
        return self._qkv_rotary_eager(hidden, batch, tokens, head_dim, rotary)

    def _qkv_rotary_eager(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = self._qkv(hidden, batch, tokens, head_dim)
        q, k = _apply_rotary_cached(q, k, rotary)
        return q, k, v

    def _run_qkv_rotary_graph(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        cos, sin = rotary
        key = (tuple(hidden.shape), tuple(cos.shape), tuple(sin.shape))
        captured = self._qkv_rotary_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_cos = torch.empty_like(cos)
            static_sin = torch.empty_like(sin)
            static_input.copy_(hidden)
            static_cos.copy_(cos)
            static_sin.copy_(sin)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, k, v = self._qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            captured = _StaticQKVRotaryGraphCall(
                graph=graph,
                static_input=static_input,
                static_cos=static_cos,
                static_sin=static_sin,
                q=q,
                k=k,
                v=v,
            )
            self._qkv_rotary_graphs[key] = captured
            captured.graph.replay()
            return captured.q, captured.k, captured.v
        captured.static_input.copy_(hidden)
        captured.static_cos.copy_(cos)
        captured.static_sin.copy_(sin)
        captured.graph.replay()
        return captured.q, captured.k, captured.v

    def _input_norm_qkv_rotary(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.world_size > 1 and _should_use_qkv_rotary_graph(hidden) and not self._input_qkv_rotary_graph_failed:
            try:
                return self._run_input_qkv_rotary_graph(hidden, batch, tokens, head_dim, rotary)
            except Exception:
                self._input_qkv_rotary_graph_failed = True
        return self._input_qkv_rotary_eager(hidden, batch, tokens, head_dim, rotary)

    def _input_qkv_rotary_eager(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn_in = _tp_rms_norm(hidden, self.input_layernorm_weight, self.config.rms_norm_eps)
        return self._qkv_rotary_eager(attn_in, batch, tokens, head_dim, rotary)

    def _run_input_qkv_rotary_graph(
        self,
        hidden: Tensor,
        batch: int,
        tokens: int,
        head_dim: int,
        rotary: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        cos, sin = rotary
        key = (tuple(hidden.shape), tuple(cos.shape), tuple(sin.shape))
        captured = self._input_qkv_rotary_graphs.get(key)
        if captured is None:
            static_input = torch.empty_like(hidden)
            static_cos = torch.empty_like(cos)
            static_sin = torch.empty_like(sin)
            static_input.copy_(hidden)
            static_cos.copy_(cos)
            static_sin.copy_(sin)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                self._input_qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, k, v = self._input_qkv_rotary_eager(static_input, batch, tokens, head_dim, (static_cos, static_sin))
            captured = _StaticQKVRotaryGraphCall(
                graph=graph,
                static_input=static_input,
                static_cos=static_cos,
                static_sin=static_sin,
                q=q,
                k=k,
                v=v,
            )
            self._input_qkv_rotary_graphs[key] = captured
            captured.graph.replay()
            return captured.q, captured.k, captured.v
        captured.static_input.copy_(hidden)
        captured.static_cos.copy_(cos)
        captured.static_sin.copy_(sin)
        captured.graph.replay()
        return captured.q, captured.k, captured.v

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
            if q.is_cuda and k.size(-2) <= 2048 and _tp_flag("TORCHINFERNO_TRITON_DECODE_ATTENTION"):
                try:
                    from torchinferno.kernels.triton_ops import triton_dense_gqa_decode_attention

                    return triton_dense_gqa_decode_attention(q, k, v)
                except Exception as exc:
                    warn_optional_failure("llama3_tensor_parallel.decode_attention", exc)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        if k.size(-2) == q.size(-2):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=enable_gqa,
            )
        try:
            from torch.nn.attention.bias import causal_lower_right

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_lower_right(q.size(-2), k.size(-2)),
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.causal_lower_right", exc)
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
        self.lm_head_weight_decode = _maybe_decode_weight_t(lm_head_weight)
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
        self.rotary_cos_cache, self.rotary_sin_cache = _build_llama_rotary_cache(
            config.max_position_embeddings,
            self.inv_freq,
            device=device,
            dtype=dtype,
        )
        self.profile_seconds: dict[str, float] = {}
        self.profile_counts: dict[str, int] = {}
        self.training = False
        self._prefill_graphs: dict[tuple[int, int, int, tuple[int, ...]], _StaticPrefillGraphCall] = {}
        self._prefill_logits_graphs: dict[tuple[int, int, int, tuple[int, ...]], _StaticPrefillLogitsGraphCall] = {}
        self._prefill_graph_failed = False
        self._prefill_logits_graph_failed = False
        self._decode_graphs: dict[tuple[int, int, int], _StaticDecodeGraphCall] = {}
        self._decode_logits_graphs: dict[tuple[int, int, int], _StaticDecodeLogitsGraphCall] = {}
        self._decode_graph_failed = False
        self._decode_logits_graph_failed = False

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
        rotary = self._rotary_cache(past_len, tokens)
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        profile_fast_prefill = _tp_flag("TORCHINFERNO_PROFILE_FAST_PREFILL", False)
        if tokens > 1 and (profile_fast_prefill or all(layer.profile_seconds is None for layer in self.layers)):
            attn_in: Tensor | None = None
            for layer_id, layer in enumerate(self.layers):
                layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
                next_norm_weight = (
                    self.layers[layer_id + 1].input_layernorm_weight
                    if layer_id + 1 < len(self.layers)
                    else None
                )
                hidden, attn_in = layer.forward_prefill_fast(
                    hidden,
                    attn_in,
                    positions,
                    rotary,
                    layer_cache,
                    next_norm_weight,
                )
        else:
            for layer_id, layer in enumerate(self.layers):
                layer_cache = active_cache.layers[layer_id] if active_cache is not None else None
                hidden = layer.forward(hidden, positions, rotary, layer_cache)
        if return_last_logits_only:
            hidden = hidden[:, -1:, :]
        hidden = _tp_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head_weight)
        if return_sharded_logits:
            return logits, active_cache
        logits = self._gather_logits(logits)
        return logits, active_cache

    def try_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if (
            not _tp_env_set("TORCHINFERNO_CUDAGRAPH_PREFILL")
            and int(getattr(self.config, "hidden_size", 0)) < 1024
        ):
            return None
        if self._prefill_graph_failed or not _should_use_prefill_graph(input_ids, cache, temperature):
            return None
        try:
            return self._run_prefill_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.prefill_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} prefill_graph_failed={exc!r}", flush=True)
            self._prefill_graph_failed = True
            return None

    def try_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        if self._prefill_logits_graph_failed or not _should_use_prefill_logits_graph(input_ids, cache):
            return None
        try:
            return self._run_prefill_logits_graph(input_ids, cache, capture_on_miss=capture_on_miss)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.prefill_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_DEBUG", False):
                print(f"rank={self.rank} prefill_logits_graph_failed={exc!r}", flush=True)
            self._prefill_logits_graph_failed = True
            return None

    def _run_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        if end_seq_len > cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        key = (
            id(cache),
            initial_seq_len,
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            tuple(input_ids.shape),
        )
        captured = self._prefill_graphs.get(key)
        if (
            captured is None
            or captured.cache is not cache
            or captured.prompt_tokens != input_ids.size(1)
            or captured.initial_seq_len != initial_seq_len
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
        ):
            if not capture_on_miss:
                return None
            captured = self._capture_prefill_graph(input_ids, cache)
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if key not in self._prefill_graphs and len(self._prefill_graphs) >= max_graphs:
                self._prefill_graphs.clear()
            self._prefill_graphs[key] = captured
        else:
            captured.static_input_ids.copy_(input_ids)
            self._set_cache_seq_len(cache, captured.initial_seq_len)
            captured.graph.replay()
            self._set_cache_seq_len(cache, end_seq_len)
        return captured.output_token

    def _capture_prefill_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> _StaticPrefillGraphCall:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        static_input_ids = torch.empty_like(input_ids)
        static_input_ids.copy_(input_ids)
        captured = _StaticPrefillGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            output_token=torch.empty((input_ids.size(0),), device=self.device, dtype=torch.long),
            cache=cache,
            prompt_tokens=input_ids.size(1),
            initial_seq_len=initial_seq_len,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._set_cache_seq_len(cache, initial_seq_len)
            logits = self._forward_prefill_static(captured.static_input_ids, cache)
            self._sample_next_token(logits[:, -1, :], 0.0)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._set_cache_seq_len(cache, initial_seq_len)
        with torch.cuda.graph(captured.graph):
            logits = self._forward_prefill_static(captured.static_input_ids, cache)
            captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
        captured.graph.replay()
        self._set_cache_seq_len(cache, end_seq_len)
        return captured

    def _run_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        if end_seq_len > cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        key = (
            id(cache),
            initial_seq_len,
            input_ids.size(1),
            cache.layers[0].max_seq_len,
            tuple(input_ids.shape),
        )
        captured = self._prefill_logits_graphs.get(key)
        if (
            captured is None
            or captured.cache is not cache
            or captured.prompt_tokens != input_ids.size(1)
            or captured.initial_seq_len != initial_seq_len
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.static_input_ids.shape != input_ids.shape
        ):
            if not capture_on_miss:
                return None
            captured = self._capture_prefill_logits_graph(input_ids, cache)
            max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", 128, minimum=1)
            if key not in self._prefill_logits_graphs and len(self._prefill_logits_graphs) >= max_graphs:
                self._prefill_logits_graphs.clear()
            self._prefill_logits_graphs[key] = captured
        else:
            captured.static_input_ids.copy_(input_ids)
            self._set_cache_seq_len(cache, captured.initial_seq_len)
            captured.graph.replay()
            self._set_cache_seq_len(cache, end_seq_len)
        return captured.output_logits

    def _capture_prefill_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> _StaticPrefillLogitsGraphCall:
        initial_seq_len = cache.seq_len
        end_seq_len = initial_seq_len + input_ids.size(1)
        static_input_ids = torch.empty_like(input_ids)
        static_input_ids.copy_(input_ids)
        captured = _StaticPrefillLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            output_logits=torch.empty(
                (input_ids.size(0), 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            prompt_tokens=input_ids.size(1),
            initial_seq_len=initial_seq_len,
            max_seq_len=cache.layers[0].max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._set_cache_seq_len(cache, initial_seq_len)
            self._forward_prefill_static(captured.static_input_ids, cache)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._set_cache_seq_len(cache, initial_seq_len)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_prefill_static(captured.static_input_ids, cache)
        captured.graph.replay()
        self._set_cache_seq_len(cache, end_seq_len)
        return captured

    def _forward_prefill_static(self, input_ids: Tensor, cache: Llama3TensorParallelCache) -> Tensor:
        logits, _ = self.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        return logits

    def try_decode_one_token_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        *,
        temperature: float = 0.0,
    ) -> Tensor | None:
        if self._decode_graph_failed or not _should_use_decode_step_graph(input_ids, cache, temperature):
            return None
        try:
            return self._run_decode_step_graph(input_ids, cache)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_step_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} decode_step_graph_failed={exc!r}", flush=True)
            self._decode_graph_failed = True
            return None

    def try_decode_one_token_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> Tensor | None:
        if self._decode_logits_graph_failed or not _should_use_decode_step_logits_graph(input_ids, cache):
            return None
        try:
            return self._run_decode_step_logits_graph(input_ids, cache)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_step_logits_graph", exc)
            if _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
                print(f"rank={self.rank} decode_step_logits_graph_failed={exc!r}", flush=True)
            self._decode_logits_graph_failed = True
            return None

    def _run_decode_step_graph(self, input_ids: Tensor, cache: Llama3TensorParallelCache) -> Tensor:
        if cache.seq_len >= cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        attention_block_size = _decode_attention_block_size(cache.seq_len + 1, cache.layers[0].max_seq_len)
        key = (id(cache), input_ids.size(0), attention_block_size)
        captured = self._decode_graphs.get(key)
        if (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.attention_block_size != attention_block_size
            or captured.static_input_ids.shape != input_ids.shape
        ):
            captured = self._capture_decode_step_graph(input_ids, cache, attention_block_size)
        else:
            self._copy_decode_graph_inputs(captured, input_ids, cache)
            captured.graph.replay()
        self._advance_decode_graph_cache(cache)
        return captured.output_token

    def _capture_decode_step_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        attention_block_size: int,
    ) -> _StaticDecodeGraphCall:
        static_input_ids = torch.empty_like(input_ids)
        static_cache_position = torch.empty((), device=self.device, dtype=torch.int64)
        static_attention_length = torch.empty((), device=self.device, dtype=torch.int64)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_rotary_cos = torch.empty((1, rotary_cache_dim), device=self.device, dtype=self.dtype)
        static_rotary_sin = torch.empty((1, rotary_cache_dim), device=self.device, dtype=self.dtype)
        captured = _StaticDecodeGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_cache_position=static_cache_position,
            static_attention_length=static_attention_length,
            static_rotary_cos=static_rotary_cos,
            static_rotary_sin=static_rotary_sin,
            output_token=torch.empty((input_ids.size(0),), device=self.device, dtype=torch.long),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            attention_block_size=attention_block_size,
        )
        self._copy_decode_graph_inputs(captured, input_ids, cache)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
            self._sample_next_token(logits[:, -1, :], 0.0)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
            captured.output_token = self._sample_next_token(logits[:, -1, :], 0.0)
        captured.graph.replay()
        key = (id(cache), input_ids.size(0), attention_block_size)
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._decode_graphs and len(self._decode_graphs) >= max_graphs:
            self._decode_graphs.clear()
        self._decode_graphs[key] = captured
        return captured

    def _run_decode_step_logits_graph(self, input_ids: Tensor, cache: Llama3TensorParallelCache) -> Tensor:
        if cache.seq_len >= cache.layers[0].max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        attention_block_size = _decode_attention_block_size(cache.seq_len + 1, cache.layers[0].max_seq_len)
        key = (id(cache), input_ids.size(0), attention_block_size)
        captured = self._decode_logits_graphs.get(key)
        if (
            captured is None
            or captured.cache is not cache
            or captured.max_seq_len != cache.layers[0].max_seq_len
            or captured.attention_block_size != attention_block_size
            or captured.static_input_ids.shape != input_ids.shape
        ):
            captured = self._capture_decode_step_logits_graph(input_ids, cache, attention_block_size)
        else:
            self._copy_decode_logits_graph_inputs(captured, input_ids, cache)
            captured.graph.replay()
        self._advance_decode_graph_cache(cache)
        return captured.output_logits

    def _capture_decode_step_logits_graph(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        attention_block_size: int,
    ) -> _StaticDecodeLogitsGraphCall:
        static_input_ids = torch.empty_like(input_ids)
        static_cache_position = torch.empty((), device=self.device, dtype=torch.int64)
        static_attention_length = torch.empty((), device=self.device, dtype=torch.int64)
        rotary_cache_dim = self.rotary_cos_cache.size(1)
        static_rotary_cos = torch.empty((1, rotary_cache_dim), device=self.device, dtype=self.dtype)
        static_rotary_sin = torch.empty((1, rotary_cache_dim), device=self.device, dtype=self.dtype)
        captured = _StaticDecodeLogitsGraphCall(
            graph=torch.cuda.CUDAGraph(),
            static_input_ids=static_input_ids,
            static_cache_position=static_cache_position,
            static_attention_length=static_attention_length,
            static_rotary_cos=static_rotary_cos,
            static_rotary_sin=static_rotary_sin,
            output_logits=torch.empty(
                (input_ids.size(0), 1, self.local_vocab_size),
                device=self.device,
                dtype=self.dtype,
            ),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            attention_block_size=attention_block_size,
        )
        self._copy_decode_logits_graph_inputs(captured, input_ids, cache)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(captured.graph):
            captured.output_logits = self._forward_decode_static(
                captured.static_input_ids,
                cache,
                captured.static_cache_position,
                captured.static_attention_length,
                (captured.static_rotary_cos, captured.static_rotary_sin),
                attention_block_size,
            )
        captured.graph.replay()
        key = (id(cache), input_ids.size(0), attention_block_size)
        max_graphs = _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_GRAPHS", 4096, minimum=1)
        if key not in self._decode_logits_graphs and len(self._decode_logits_graphs) >= max_graphs:
            self._decode_logits_graphs.clear()
        self._decode_logits_graphs[key] = captured
        return captured

    def _copy_decode_graph_inputs(
        self,
        captured: _StaticDecodeGraphCall,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> None:
        position = cache.seq_len
        captured.static_input_ids.copy_(input_ids)
        captured.static_cache_position.fill_(position)
        captured.static_attention_length.fill_(position + 1)
        captured.static_rotary_cos.copy_(self.rotary_cos_cache[position : position + 1])
        captured.static_rotary_sin.copy_(self.rotary_sin_cache[position : position + 1])

    def _copy_decode_logits_graph_inputs(
        self,
        captured: _StaticDecodeLogitsGraphCall,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
    ) -> None:
        position = cache.seq_len
        captured.static_input_ids.copy_(input_ids)
        captured.static_cache_position.fill_(position)
        captured.static_attention_length.fill_(position + 1)
        captured.static_rotary_cos.copy_(self.rotary_cos_cache[position : position + 1])
        captured.static_rotary_sin.copy_(self.rotary_sin_cache[position : position + 1])

    def _advance_decode_graph_cache(self, cache: Llama3TensorParallelCache) -> None:
        next_seq_len = cache.seq_len + 1
        self._set_cache_seq_len(cache, next_seq_len)

    @staticmethod
    def _set_cache_seq_len(cache: Llama3TensorParallelCache, seq_len: int) -> None:
        cache.set_seq_len(seq_len)

    def _forward_decode_static(
        self,
        input_ids: Tensor,
        cache: Llama3TensorParallelCache,
        cache_position: Tensor,
        attention_length: Tensor,
        rotary: tuple[Tensor, Tensor],
        attention_block_size: int | None = None,
    ) -> Tensor:
        hidden = F.embedding(input_ids.to(self.device, non_blocking=True), self.embed_tokens_weight)
        attn_in: Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            next_norm_weight = (
                self.layers[layer_id + 1].input_layernorm_weight
                if layer_id + 1 < len(self.layers)
                else self.norm_weight
            )
            hidden, attn_in = layer.forward_decode_static(
                hidden,
                attn_in,
                rotary,
                cache.layers[layer_id],
                cache_position,
                attention_length,
                attention_block_size,
                next_norm_weight,
            )
        if attn_in is None:
            attn_in = _tp_decode_rms_norm(hidden, self.norm_weight, self.config.rms_norm_eps)
        return _decode_linear(attn_in, self.lm_head_weight, self.lm_head_weight_decode)

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
        if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER", False):
            try:
                return self._sample_next_token_temperature_gather(logits, temperature)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.temperature_sample_gather", exc)
                if _tp_flag("TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER_STRICT", False):
                    raise
        return self._sample_next_token_temperature(logits, temperature)

    def _sample_next_token_greedy(self, logits: Tensor) -> Tensor:
        if _tp_flag("TORCHINFERNO_GREEDY_SAMPLE_GATHER"):
            try:
                return self._sample_next_token_greedy_gather(logits)
            except Exception as exc:
                warn_optional_failure("llama3_tensor_parallel.greedy_sample_gather", exc)
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        global_values = local_values.clone()
        dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
        sentinel = torch.full_like(local_indices, self.config.vocab_size)
        local_tokens = local_indices + self.vocab_start
        next_token = torch.where(local_values == global_values, local_tokens, sentinel)
        dist.all_reduce(next_token, op=dist.ReduceOp.MIN)
        return next_token

    def _sample_next_token_greedy_gather(self, logits: Tensor) -> Tensor:
        local_values, local_indices = torch.max(logits.float(), dim=-1)
        local_tokens = (local_indices + self.vocab_start).to(torch.float32)
        local_pairs = torch.stack((local_values, local_tokens), dim=-1).contiguous()
        gathered = torch.empty(
            (self.world_size, *local_pairs.shape),
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered, local_pairs)
        values = gathered[..., 0]
        tokens = gathered[..., 1].to(torch.long)
        global_values = values.max(dim=0).values
        sentinel = torch.full_like(tokens, self.config.vocab_size)
        candidate_tokens = torch.where(values == global_values[None, :], tokens, sentinel)
        return candidate_tokens.min(dim=0).values

    def _sample_next_token_temperature_gather(self, logits: Tensor, temperature: float) -> Tensor:
        local_logits = logits.contiguous()
        gathered = torch.empty(
            (self.world_size, *local_logits.shape),
            dtype=local_logits.dtype,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered, local_logits)
        next_token = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        if self.is_primary:
            full_logits = gathered.permute(1, 0, 2).reshape(logits.size(0), self.world_size * logits.size(1))
            probs = torch.softmax(full_logits.float() / temperature, dim=-1)
            next_token.copy_(torch.multinomial(probs, num_samples=1).squeeze(-1))
        dist.broadcast(next_token, src=0)
        return next_token

    def _sample_next_token_temperature(self, logits: Tensor, temperature: float) -> Tensor:
        logits_float = logits.float() / temperature
        local_max = torch.max(logits_float, dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        weights = torch.exp(logits_float - global_max[:, None])
        local_sum = weights.sum(dim=-1)
        gathered_sums = torch.empty(
            (self.world_size, *local_sum.shape),
            dtype=local_sum.dtype,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered_sums, local_sum.contiguous())

        selected_rank = torch.empty(logits.size(0), dtype=torch.long, device=self.device)
        local_threshold = torch.empty(logits.size(0), dtype=torch.float32, device=self.device)
        if self.is_primary:
            cumulative = torch.cumsum(gathered_sums, dim=0)
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
        local_index = torch.searchsorted(cumulative_local.contiguous(), local_threshold[:, None]).squeeze(-1)
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

    def _rotary_cache(self, start: int, tokens: int) -> tuple[Tensor, Tensor]:
        end = start + tokens
        if end <= self.rotary_cos_cache.size(0):
            return self.rotary_cos_cache[start:end], self.rotary_sin_cache[start:end]
        positions = torch.arange(start, end, device=self.device)
        freqs = torch.outer(positions.float(), self.inv_freq)
        return freqs.cos().to(dtype=self.dtype), freqs.sin().to(dtype=self.dtype)


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


def _build_llama_rotary_cache(
    max_position_embeddings: int,
    inv_freq: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.float())
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def _apply_rotary_cached(q: Tensor, k: Tensor, rotary: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    cos, sin = rotary
    if q.is_cuda and k.is_cuda and _tp_flag("TORCHINFERNO_TRITON_ROTARY"):
        try:
            from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_inplace

            return triton_apply_rotary_llama_inplace(q, k, cos, sin)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rotary", exc)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return _rotate_llama(q, cos, sin), _rotate_llama(k, cos, sin)


def _rotate_llama(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if x.is_cuda and _tp_flag("TORCHINFERNO_COMPILE_ROTARY", False):
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
    if cos.size(-1) == half:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _decode_linear(x: Tensor, weight: Tensor, weight_t: Tensor | None = None) -> Tensor:
    if (
        x.is_cuda
        and x.ndim == 3
        and x.size(0) == 1
        and x.size(1) == 1
        and _tp_flag("TORCHINFERNO_DECODE_LINEAR_MV")
    ):
        if weight_t is not None:
            return torch.mm(x.reshape(1, -1), weight_t).view(1, 1, weight.size(0))
        return torch.mv(weight, x.reshape(-1)).view(1, 1, weight.size(0))
    return F.linear(x, weight)


def _maybe_decode_weight_t(weight: Tensor) -> Tensor | None:
    if (
        not weight.is_cuda
        or not _tp_flag("TORCHINFERNO_DECODE_TRANSPOSED_WEIGHTS")
        or weight.ndim != 2
    ):
        return None
    try:
        return weight.t().contiguous()
    except Exception as exc:
        warn_optional_failure("llama3_tensor_parallel.decode_weight_transpose", exc)
        return None


def _tp_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if x.is_cuda and weight.is_cuda and _tp_flag("TORCHINFERNO_TRITON_RMS_NORM", False):
        try:
            from torchinferno.kernels import rms_norm as kernel_rms_norm

            return kernel_rms_norm(x, weight, eps=eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.rms_norm", exc)
    return _torch_rms_norm(x, weight, eps)


def _tp_decode_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if x.is_cuda and weight.is_cuda and _tp_flag("TORCHINFERNO_TRITON_DECODE_RMS_NORM"):
        try:
            from torchinferno.kernels import rms_norm as kernel_rms_norm

            return kernel_rms_norm(x, weight, eps=eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_rms_norm", exc)
    return _torch_rms_norm(x, weight, eps)


def _tp_decode_add_rms_norm(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    if (
        x.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and _tp_flag("TORCHINFERNO_TRITON_DECODE_ADD_RMS_NORM")
    ):
        try:
            from torchinferno.kernels.triton_ops import triton_add_rms_norm

            return triton_add_rms_norm(x, residual, weight, eps)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_add_rms_norm", exc)
    hidden = residual + x
    return hidden, _torch_rms_norm(hidden, weight, eps)


def _tp_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_SWIGLU", False):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.swiglu", exc)
    return F.silu(gate) * up


def _tp_decode_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    if gate.is_cuda and up.is_cuda and _tp_flag("TORCHINFERNO_TRITON_DECODE_SWIGLU"):
        try:
            from torchinferno.kernels import swiglu_activation

            return swiglu_activation(gate, up)
        except Exception as exc:
            warn_optional_failure("llama3_tensor_parallel.decode_swiglu", exc)
    return F.silu(gate) * up


def _should_use_mlp_project_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) == 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_MLP")
    )


def _should_use_qkv_rotary_graph(hidden: Tensor) -> bool:
    prefill_tokens = hidden.size(1) > 1 and _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_QKV_ROTARY", False)
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and (hidden.size(1) == 1 or prefill_tokens)
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_QKV_ROTARY")
    )


def _should_use_prefill_gate_up_activation_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) > 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL_GATE_UP")
    )


def _should_use_attention_o_graph(hidden: Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) == 1
        and _tp_flag("TORCHINFERNO_CUDAGRAPH_ATTENTION_O")
    )


def _should_graph_all_reduce() -> bool:
    return _tp_flag("TORCHINFERNO_CUDAGRAPH_ALLREDUCE", False)


def _should_use_decode_step_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    temperature: float,
) -> bool:
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_STEP")
        and temperature <= 0.0
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and input_ids.size(1) == 1
        and bool(cache.layers)
        and cache.layers[0].keys.is_cuda
    )


def _should_use_decode_step_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
) -> bool:
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_STEP")
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _decode_step_max_batch()
        and input_ids.size(1) == 1
        and bool(cache.layers)
        and cache.layers[0].keys.is_cuda
    )


def _decode_step_max_batch() -> int:
    return _tp_int("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", _DEFAULT_DECODE_STEP_MAX_BATCH, minimum=1)


def _should_use_prefill_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
    temperature: float,
) -> bool:
    max_cache_tokens = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_CACHE_TOKENS", 1024, minimum=1)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL")
        and temperature <= 0.0
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_BATCH", 8, minimum=1)
        and input_ids.size(1) > 1
        and bool(cache.layers)
        and cache.layers[0].keys.is_cuda
        and cache.layers[0].max_seq_len <= max_cache_tokens
    )


def _should_use_prefill_logits_graph(
    input_ids: Tensor,
    cache: Llama3TensorParallelCache,
) -> bool:
    max_cache_tokens = _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_CACHE_TOKENS", 1024, minimum=1)
    return (
        _tp_flag("TORCHINFERNO_CUDAGRAPH_PREFILL")
        and input_ids.is_cuda
        and input_ids.ndim == 2
        and 1 <= input_ids.size(0) <= _tp_int("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_BATCH", 8, minimum=1)
        and input_ids.size(1) > 1
        and bool(cache.layers)
        and cache.layers[0].keys.is_cuda
        and cache.layers[0].max_seq_len <= max_cache_tokens
    )


def _decode_attention_block_size(attention_length: int, max_seq_len: int) -> int:
    if not _tp_flag("TORCHINFERNO_CUDAGRAPH_DECODE_ATTENTION_BLOCKS"):
        return max_seq_len
    if attention_length <= 1:
        return 1
    return min(max_seq_len, 1 << (attention_length - 1).bit_length())


def _should_use_symm_mem_all_reduce(hidden: Tensor, weight: Tensor, world_size: int) -> bool:
    return (
        world_size > 1
        and not _SYMM_REDUCE_DISABLED
        and _tp_flag("TORCHINFERNO_SYMM_MEM_ALLREDUCE")
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) == 1
    )


def _should_use_symm_mem_prefill_all_reduce(hidden: Tensor, weight: Tensor, world_size: int) -> bool:
    return (
        world_size > 1
        and not _SYMM_REDUCE_DISABLED
        and _tp_flag("TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE", False)
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.ndim == 3
        and hidden.size(0) == 1
        and hidden.size(1) > 1
    )


def _disable_symm_reduce() -> None:
    global _SYMM_REDUCE_DISABLED
    _SYMM_REDUCE_DISABLED = True


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
