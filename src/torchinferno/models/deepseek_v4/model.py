from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor, nn

from torchinferno.models.deepseek_v4.cache import DeepSeekV4Cache, DeepSeekV4CompressorCache, DeepSeekV4LayerCache
from torchinferno.models.deepseek_v4.config import DeepSeekV4Config
from torchinferno.models.deepseek_v4.ops import (
    apply_rotary_emb,
    attention_with_sink,
    fake_quant_fp4,
    fake_quant_fp8,
    hadamard_transform,
    hc_split_sinkhorn,
    precompute_freqs_cis,
)
from torchinferno.models.hf import HF_CONFIG_NAME, SAFETENSORS_NAME, load_config, load_state_dict, resolve_pretrained_path
from torchinferno.runtime.sampling import sample_next_token


class DeepSeekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(dtype=x.dtype)


class DeepSeekV4Linear(nn.Module):
    """Unquantized reference linear.

    Production block-FP8 and MXFP4 parameters are loaded by the CUDA TP model;
    keeping this class ordinary makes the v0 graph inspectable on CPU.
    """

    def __init__(self, in_features: int, out_features: int, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class DeepSeekV4Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        ratio: int,
        head_dim: int,
        *,
        rotate: bool,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = ratio == 4
        self.rotate = rotate
        factor = 2 if self.overlap else 1
        self.ape = nn.Parameter(torch.zeros(ratio, factor * head_dim, dtype=torch.float32))
        self.wkv = DeepSeekV4Linear(config.hidden_size, factor * head_dim, dtype=torch.float32)
        self.wgate = DeepSeekV4Linear(config.hidden_size, factor * head_dim, dtype=torch.float32)
        self.norm = DeepSeekV4RMSNorm(head_dim, config.rms_norm_eps)
        self.simulate_qat = config.simulate_qat

    def update(
        self,
        x: Tensor,
        position: int,
        cache: DeepSeekV4CompressorCache,
        rows: tuple[int, ...],
        freq: Tensor,
    ) -> Tensor | None:
        ratio = self.ratio
        slot = position % ratio
        raw_kv = self.wkv(x.float()).float()
        raw_score = self.wgate(x.float()).float() + self.ape[slot]
        row_list = list(rows)
        cache.raw_kv[row_list, position] = raw_kv
        cache.raw_score[row_list, position] = raw_score

        if self.overlap:
            cache.kv_state[row_list, ratio + slot] = raw_kv
            cache.score_state[row_list, ratio + slot] = raw_score
        else:
            cache.kv_state[row_list, slot] = raw_kv
            cache.score_state[row_list, slot] = raw_score
        if slot != ratio - 1:
            return None

        if self.overlap:
            kv_state = torch.cat(
                (cache.kv_state[row_list, :ratio, : self.head_dim], cache.kv_state[row_list, ratio:, self.head_dim :]),
                dim=1,
            )
            score_state = torch.cat(
                (
                    cache.score_state[row_list, :ratio, : self.head_dim],
                    cache.score_state[row_list, ratio:, self.head_dim :],
                ),
                dim=1,
            )
        else:
            kv_state = cache.kv_state[row_list]
            score_state = cache.score_state[row_list]
        compressed = (kv_state * score_state.softmax(dim=1)).sum(dim=1)
        compressed = self.norm(compressed.to(dtype=x.dtype))
        rope = apply_rotary_emb(compressed[..., -self.rope_head_dim :], freq)
        compressed = torch.cat((compressed[..., : -self.rope_head_dim], rope), dim=-1)
        if self.rotate:
            compressed = hadamard_transform(compressed)
            if self.simulate_qat:
                compressed = fake_quant_fp4(compressed)
        elif self.simulate_qat:
            nope = fake_quant_fp8(compressed[..., : -self.rope_head_dim])
            compressed = torch.cat((nope, compressed[..., -self.rope_head_dim :]), dim=-1)
        block = position // ratio
        cache.compressed[row_list, block] = compressed.float()
        if self.overlap:
            cache.kv_state[row_list, :ratio] = cache.kv_state[row_list, ratio:]
            cache.score_state[row_list, :ratio] = cache.score_state[row_list, ratio:]
        return compressed


class DeepSeekV4Indexer(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.wq_b = DeepSeekV4Linear(config.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = DeepSeekV4Linear(config.hidden_size, self.n_heads)
        self.compressor = DeepSeekV4Compressor(config, 4, self.head_dim, rotate=True)
        self.scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.simulate_qat = config.simulate_qat

    def update_and_select(
        self,
        x: Tensor,
        q_lora: Tensor,
        position: int,
        cache: DeepSeekV4CompressorCache,
        rows: tuple[int, ...],
        query_freq: Tensor,
        compress_freq: Tensor,
    ) -> Tensor:
        query = self.wq_b(q_lora).unflatten(-1, (self.n_heads, self.head_dim))
        rope = apply_rotary_emb(query[..., -self.rope_head_dim :], query_freq)
        query = torch.cat((query[..., : -self.rope_head_dim], rope), dim=-1)
        query = hadamard_transform(query)
        if self.simulate_qat:
            query = fake_quant_fp4(query)
        self.compressor.update(x, position, cache, rows, compress_freq)
        completed = (position + 1) // 4
        if completed == 0:
            return torch.empty(x.size(0), 0, dtype=torch.long, device=x.device)
        keys = cache.compressed[list(rows), :completed].to(dtype=query.dtype)
        per_head = torch.einsum("bhd,btd->bht", query, keys).relu_()
        head_weights = self.weights_proj(x) * self.scale
        scores = (per_head * head_weights.unsqueeze(-1)).sum(dim=1)
        return scores.topk(min(self.index_topk, completed), dim=-1).indices


class DeepSeekV4Attention(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim**-0.5
        self.attn_sink = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.wq_a = DeepSeekV4Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = DeepSeekV4RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = DeepSeekV4Linear(config.q_lora_rank, config.num_attention_heads * config.head_dim)
        self.wkv = DeepSeekV4Linear(config.hidden_size, config.head_dim)
        self.kv_norm = DeepSeekV4RMSNorm(config.head_dim, config.rms_norm_eps)
        group_input = config.num_attention_heads * config.head_dim // config.o_groups
        self.wo_a = DeepSeekV4Linear(group_input, config.o_groups * config.o_lora_rank)
        self.wo_b = DeepSeekV4Linear(config.o_groups * config.o_lora_rank, config.hidden_size)
        self.compressor: DeepSeekV4Compressor | None = None
        self.indexer: DeepSeekV4Indexer | None = None
        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(config, self.compress_ratio, config.head_dim, rotate=False)
        if self.compress_ratio == 4:
            self.indexer = DeepSeekV4Indexer(config)

    def _freqs(self, positions: Tensor, device: torch.device) -> Tensor:
        if self.compress_ratio:
            base = self.config.compress_rope_theta
            original = self.config.rope_original_max_position_embeddings
        else:
            base = self.config.rope_theta
            original = 0
        all_freqs = precompute_freqs_cis(
            self.rope_head_dim,
            int(positions.max().item()) + 1,
            original_seq_len=original,
            base=base,
            factor=self.config.rope_factor,
            beta_fast=self.config.rope_beta_fast,
            beta_slow=self.config.rope_beta_slow,
            device=device,
        )
        return all_freqs.index_select(0, positions)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: DeepSeekV4LayerCache,
        rows: tuple[int, ...],
    ) -> Tensor:
        batch, tokens, _ = x.shape
        if len(rows) != batch:
            raise ValueError("V4 cache row count must match attention batch")
        freqs = self._freqs(positions, x.device)
        q_lora = self.q_norm(self.wq_a(x))
        query = self.wq_b(q_lora).unflatten(-1, (self.n_heads, self.head_dim))
        query = query.float() * torch.rsqrt(query.float().square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps)
        query_rope = apply_rotary_emb(query[..., -self.rope_head_dim :], freqs)
        query = torch.cat((query[..., : -self.rope_head_dim], query_rope), dim=-1).to(dtype=x.dtype)

        kv = self.kv_norm(self.wkv(x))
        kv_rope = apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
        kv = torch.cat((kv[..., : -self.rope_head_dim], kv_rope), dim=-1)
        if self.config.simulate_qat:
            kv = torch.cat((fake_quant_fp8(kv[..., : -self.rope_head_dim]), kv[..., -self.rope_head_dim :]), dim=-1)

        outputs: list[Tensor] = []
        row_list = list(rows)
        for offset in range(tokens):
            position = int(positions[offset].item())
            cache.history_kv[row_list, position] = kv[:, offset]
            cache.swa_kv[row_list, position % self.window_size] = kv[:, offset]
            selected_compressed: Tensor | None = None
            if self.compressor is not None:
                assert cache.compressor is not None
                block_start = position + 1 - self.compress_ratio
                compress_freq = (
                    self._freqs(torch.tensor([block_start], device=x.device), x.device)[0]
                    if block_start >= 0
                    else freqs[offset]
                )
                self.compressor.update(x[:, offset], position, cache.compressor, rows, compress_freq)
                completed = (position + 1) // self.compress_ratio
                if self.indexer is not None:
                    assert cache.indexer_compressor is not None
                    selected_compressed = self.indexer.update_and_select(
                        x[:, offset],
                        q_lora[:, offset],
                        position,
                        cache.indexer_compressor,
                        rows,
                        freqs[offset],
                        compress_freq,
                    )
                elif completed:
                    selected_compressed = torch.arange(completed, device=x.device).expand(batch, -1)

            start = max(0, position - self.window_size + 1)
            window = cache.history_kv[row_list, start : position + 1]
            candidates = window
            if selected_compressed is not None and selected_compressed.numel():
                assert cache.compressor is not None
                compressed = cache.compressor.compressed[row_list]
                gather = selected_compressed.unsqueeze(-1).expand(-1, -1, self.head_dim)
                candidates = torch.cat((window, compressed.gather(1, gather).to(dtype=window.dtype)), dim=1)
            output = attention_with_sink(query[:, offset], candidates, self.attn_sink, self.softmax_scale)
            output_rope = apply_rotary_emb(output[..., -self.rope_head_dim :], freqs[offset], inverse=True)
            outputs.append(torch.cat((output[..., : -self.rope_head_dim], output_rope), dim=-1))
        attended = torch.stack(outputs, dim=1)
        heads_per_group = self.n_heads // self.n_groups
        grouped = attended.unflatten(2, (self.n_groups, heads_per_group)).flatten(3)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        low_rank = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        for row in rows:
            cache.seq_lens[row] = int(positions[-1].item()) + 1
        return self.wo_b(low_rank.flatten(2))


class DeepSeekV4Gate(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=config.hidden_size**-0.5)
        if self.hash:
            ids = torch.arange(config.vocab_size * self.topk, dtype=torch.long)
            self.tid2eid = nn.Parameter(ids.remainder(config.n_routed_experts).view(config.vocab_size, self.topk), requires_grad=False)
            self.register_parameter("bias", None)
        else:
            self.register_parameter("tid2eid", None)
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(self, x: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
        selection_scores = scores if self.bias is None else scores + self.bias
        indices = self.tid2eid[input_ids] if self.hash else selection_scores.topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale, indices


class DeepSeekV4Expert(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.w1 = DeepSeekV4Linear(config.hidden_size, config.moe_intermediate_size)
        self.w2 = DeepSeekV4Linear(config.moe_intermediate_size, config.hidden_size)
        self.w3 = DeepSeekV4Linear(config.hidden_size, config.moe_intermediate_size)
        self.limit = config.swiglu_limit

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.w2((F.silu(gate) * up).to(dtype=x.dtype))


class DeepSeekV4MoE(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = DeepSeekV4Gate(config, layer_idx)
        self.experts = nn.ModuleList([DeepSeekV4Expert(config) for _ in range(config.n_routed_experts)])
        self.shared_experts = DeepSeekV4Expert(config)

    def forward(self, x: Tensor, input_ids: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(flat, input_ids.reshape(-1))
        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_indices, slots = torch.where(indices == expert_idx)
            if token_indices.numel():
                contribution = expert(flat.index_select(0, token_indices))
                contribution = contribution * weights[token_indices, slots, None].to(dtype=contribution.dtype)
                output.index_add_(0, token_indices, contribution)
        output = output + self.shared_experts(flat)
        return output.view(shape)


class DeepSeekV4Block(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.attn = DeepSeekV4Attention(config, layer_idx)
        self.ffn = DeepSeekV4MoE(config, layer_idx)
        self.attn_norm = DeepSeekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepSeekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def hc_pre(self, x: Tensor, fn: Tensor, scale: Tensor, base: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shape = x.shape
        flat = x.flatten(2).float()
        mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            scale,
            base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        return (pre.unsqueeze(-1) * flat.view(shape)).sum(dim=2).to(dtype=x.dtype), post, comb

    @staticmethod
    def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        return (
            post.unsqueeze(-1) * x.unsqueeze(-2)
            + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        ).to(dtype=x.dtype)

    def forward(
        self,
        x: Tensor,
        input_ids: Tensor,
        positions: Tensor,
        cache: DeepSeekV4LayerCache,
        rows: tuple[int, ...],
    ) -> Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(x), positions, cache, rows)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self.hc_post(x, residual, post, comb)


class DeepSeekV4ForCausalLM(nn.Module):
    """Torch-native DeepSeek-V4 reference model.

    The formulas follow DeepSeek's public MIT-licensed V4-Flash inference
    reference. This class is the readable correctness contract; the production
    quantized TP implementation lives in ``tensor_parallel.py``.
    """

    provenance_variant = "deepseek-v4:v0"
    supports_padded_batch_prefill = False
    supports_prefix_cache = False

    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        if config.quantization_config is not None or config.expert_dtype == "fp4":
            raise ValueError(
                "the torch-native V4 reference accepts dequantized weights; "
                "use DeepSeekV4TensorParallelForCausalLM for the public FP8/MXFP4 checkpoint"
            )
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV4Block(config, index) for index in range(config.num_hidden_layers)])
        self.norm = DeepSeekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = DeepSeekV4Linear(config.hidden_size, config.vocab_size, dtype=torch.float32)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.zeros(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.zeros(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        if config.tie_word_embeddings:
            self.head.weight = self.embed.weight

    @property
    def model(self) -> "DeepSeekV4ForCausalLM":
        return self

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_backend: str = "dense",
        page_size: int = 16,
    ) -> DeepSeekV4Cache:
        del page_size
        if cache_backend not in {"dense", "v4-heterogeneous-dense"}:
            raise ValueError(f"unsupported DeepSeek V4 reference cache backend: {cache_backend}")
        max_seq_len = self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        device = self.embed.weight.device if device is None else device
        dtype = self.embed.weight.dtype if dtype is None else dtype
        return DeepSeekV4Cache.allocate(
            self.config,
            batch_size,
            max_seq_len,
            device=device,
            dtype=dtype,
        )

    def _hc_head(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.flatten(2).float()
        mixes = F.linear(flat, self.hc_head_fn) * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        )
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        return (pre.unsqueeze(-1) * flat.view(shape)).sum(dim=2).to(dtype=x.dtype)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Optional[DeepSeekV4Cache] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[DeepSeekV4Cache]]:
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, nonzero sequence]")
        batch, tokens = input_ids.shape
        active_cache = cache if use_cache else None
        if active_cache is None:
            active_cache = self.allocate_cache(batch, max_seq_len=tokens, device=input_ids.device)
        if len(active_cache.selected_rows) != batch:
            raise ValueError("V4 cache row selection must match input batch")
        start = active_cache.seq_len
        if start + tokens > active_cache.layers[0].max_seq_len:
            raise ValueError("input sequence exceeds V4 cache capacity")
        positions = torch.arange(start, start + tokens, device=input_ids.device)
        hidden = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        for layer, layer_cache in zip(self.layers, active_cache.layers):
            hidden = layer(hidden, input_ids, positions, layer_cache, active_cache.selected_rows)
        hidden = self.norm(self._hc_head(hidden))
        logits = F.linear(hidden.float(), self.head.weight.float())
        return logits, active_cache if use_cache else None

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        cache_backend: str = "dense",
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
                max_seq_len=input_ids.size(1) + max_new_tokens,
                device=input_ids.device,
                cache_backend=cache_backend,
            )
            generated = input_ids
            logits, cache = self(input_ids, cache=cache, use_cache=True)
            for step in range(max_new_tokens):
                token = sample_next_token(logits[:, -1], temperature=temperature)
                generated = torch.cat((generated, token[:, None]), dim=1)
                if eos_token_id is not None and bool(torch.all(token == eos_token_id)):
                    break
                if step + 1 < max_new_tokens:
                    logits, cache = self(token[:, None], cache=cache, use_cache=True)
            return generated
        finally:
            self.train(was_training)

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
    ) -> "DeepSeekV4ForCausalLM":
        path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
        config = DeepSeekV4Config.from_dict(load_config(path))
        if config.quantization_config is not None or config.expert_dtype == "fp4":
            raise ValueError(
                "public quantized DeepSeek V4 checkpoints require "
                "DeepSeekV4TensorParallelForCausalLM.from_pretrained"
            )
        model = cls(config)
        state_dict = load_state_dict(path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        return model

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / HF_CONFIG_NAME).write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n")
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        save_file(state, path / SAFETENSORS_NAME, metadata={"format": "pt"})
