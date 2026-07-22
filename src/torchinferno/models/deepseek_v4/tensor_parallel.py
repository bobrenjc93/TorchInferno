"""H100 tensor/expert-parallel DeepSeek-V4 execution path.

The model formulas are adapted from DeepSeek's public MIT-licensed V4-Flash
inference reference at commit 60d8d70770c6776ff598c94bb586a859a38244f1.
TorchInferno owns checkpoint streaming, process groups, serving integration,
and validation around this implementation. See ``THIRD_PARTY_NOTICES.md``.
"""

import math
import os
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchinferno.models.checkpoint_io import CheckpointTensorLoader
from torchinferno.models.deepseek_v4.config import DeepSeekV4Config
from torchinferno.models.deepseek_v4.ops import dequantize_block_fp8
from torchinferno.models.hf import load_config, resolve_pretrained_path


world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
_tensor_parallel_process_group: dist.ProcessGroup | None = None


def set_tensor_parallel_process_group(group: dist.ProcessGroup | None) -> None:
    """Select the collective group used by the V4 tensor-parallel replica."""

    global _tensor_parallel_process_group
    _tensor_parallel_process_group = group


def _all_reduce(tensor: torch.Tensor) -> None:
    dist.all_reduce(tensor, group=_tensor_parallel_process_group)


def _all_gather(output_tensors: list[torch.Tensor], tensor: torch.Tensor) -> None:
    dist.all_gather(output_tensors, tensor, group=_tensor_parallel_process_group)


def _v4_kernel(name: str):
    # TileLang records the current CUDA device while creating its packed ABI.
    # Import only after the worker has selected LOCAL_RANK.
    from torchinferno.kernels import deepseek_v4_tilelang

    return getattr(deepseek_v4_tilelang, name)


def act_quant(*args, **kwargs):
    return _v4_kernel("act_quant")(*args, **kwargs)


def fp4_act_quant(*args, **kwargs):
    return _v4_kernel("fp4_act_quant")(*args, **kwargs)


def fp8_gemm(*args, **kwargs):
    return _v4_kernel("fp8_gemm")(*args, **kwargs)


def fp4_gemm(*args, **kwargs):
    return _v4_kernel("fp4_gemm")(*args, **kwargs)


def sparse_attn(*args, **kwargs):
    return _v4_kernel("sparse_attn")(*args, **kwargs)


def hc_split_sinkhorn(*args, **kwargs):
    return _v4_kernel("hc_split_sinkhorn")(*args, **kwargs)


def q_norm_rope(*args, **kwargs):
    return _v4_kernel("q_norm_rope")(*args, **kwargs)


def rope_inplace(*args, **kwargs):
    return _v4_kernel("rope_inplace")(*args, **kwargs)


def fused_hc_post(*args, **kwargs):
    return _v4_kernel("hc_post")(*args, **kwargs)


def fused_hc_prenorm_gemm(*args, **kwargs):
    return _v4_kernel("hc_prenorm_gemm")(*args, **kwargs)


@lru_cache(1)
def _precompiled_rmsnorm_op():
    try:
        __import__("sgl_kernel")
        return torch.ops.sgl_kernel.rmsnorm.default
    except (ImportError, AttributeError):
        return None


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp32"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            _all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=scale_dtype)
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=scale_dtype)
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            _all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor):
        fused = os.environ.get(
            "TORCHINFERNO_V4_FUSED_RMSNORM",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if fused and x.is_cuda and x.dtype == self.weight.dtype and x.is_contiguous():
            op = _precompiled_rmsnorm_op()
            if op is not None:
                shape = x.shape
                x = x.view(-1, shape[-1])
                output = torch.empty_like(x)
                op(output, x, self.weight, self.eps, True)
                return output.view(shape)
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    assert x.dtype == torch.bfloat16
    try:
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=x.size(-1) ** -0.5)
    except ImportError:
        from torchinferno.models.deepseek_v4.ops import hadamard_transform as torch_hadamard

        return torch_hadamard(x)


@lru_cache(16)
def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    device: torch.device,
):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size, device=device),
                torch.arange(0, start_pos + 1, device=device),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(16)
def get_compress_topk_idxs(
    ratio: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: torch.device,
):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # When overlap, the first half of dims is for overlapping compression, second half for normal.
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int, row_indices: torch.Tensor):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # compression need fp32
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[row_indices, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[row_indices, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, remainder_kv = kv.split([cutoff, remainder], dim=1)
                self.kv_state[row_indices, offset : offset+remainder] = remainder_kv
                self.score_state[row_indices, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[row_indices, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[row_indices, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[row_indices, :ratio, :d], self.kv_state[row_indices, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[row_indices, :ratio, :d], self.score_state[row_indices, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[row_indices, :ratio] = self.kv_state[row_indices, ratio:]
                    self.score_state[row_indices, :ratio] = self.score_state[row_indices, ratio:]
            else:
                self.kv_state[row_indices, start_pos % ratio] = kv.squeeze(1)
                self.score_state[row_indices, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[row_indices] * self.score_state[row_indices].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[row_indices, :seqlen // ratio] = kv
        else:
            self.kv_cache[row_indices, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
        row_indices: torch.Tensor,
    ):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4 simulation for q and kv in indexer
        fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos, row_indices)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[row_indices, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            _all_reduce(index_score)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= torch.arange(
                1, seqlen + 1, device=x.device
            ).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer(
            "_freqs_cis_real",
            torch.view_as_real(freqs_cis).flatten(-2).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_q_norm_rope_output",
            torch.empty(
                args.max_batch_size,
                self.n_local_heads,
                self.head_dim,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        row_indices: torch.Tensor,
    ):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        fused_rope = os.environ.get(
            "TORCHINFERNO_V4_FUSED_ROPE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        fused_q_norm_rope = os.environ.get(
            "TORCHINFERNO_V4_FUSED_Q_NORM_ROPE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        if q.is_cuda and seqlen == 1 and fused_q_norm_rope:
            q = q_norm_rope(
                q[:, 0],
                self._freqs_cis_real[start_pos],
                self.eps,
                output=self._q_norm_rope_output[:bsz],
            ).unsqueeze(1)
        else:
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        if kv.is_cuda and seqlen == 1 and fused_rope:
            rope_inplace(
                kv[:, 0].unsqueeze(1),
                self._freqs_cis_real[start_pos],
            )
        else:
            apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset, row_indices)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset, x.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.to(device=x.device, dtype=torch.int32)

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[row_indices, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail, head = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                self.kv_cache[row_indices, cutoff: win] = tail
                self.kv_cache[row_indices, :cutoff] = head
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos, row_indices)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[row_indices, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos, row_indices)
            o = sparse_attn(q, self.kv_cache[row_indices], self.attn_sink, topk_idxs, self.softmax_scale)
        if o.is_cuda and seqlen == 1 and fused_rope:
            rope_inplace(
                o[:, 0],
                self._freqs_cis_real[start_pos],
                inverse=True,
            )
        else:
            apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
        # but using BF16 for simplicity.
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // world_size experts."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None
                                       for i in range(self.n_routed_experts)])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self._fused_experts = None

    def prepare_fused_experts(self) -> bool:
        """Replace native per-expert tensors with a grouped CUDA representation."""

        if self._fused_experts is not None:
            return True
        local = [
            self.experts[index]
            for index in range(self.experts_start_idx, self.experts_end_idx)
        ]
        if not local or local[0].w1.weight.dtype != torch.float4_e2m1fn_x2:
            return False
        from torchinferno.kernels.deepseek_v4_marlin import (
            load_mxfp4_moe_ops,
            prepare_mxfp4_experts,
        )

        if not load_mxfp4_moe_ops():
            return False
        self._fused_experts = prepare_mxfp4_experts(
            torch.stack([expert.w1.weight for expert in local]),
            torch.stack([expert.w3.weight for expert in local]),
            torch.stack([expert.w2.weight for expert in local]),
            torch.stack([expert.w1.scale for expert in local]),
            torch.stack([expert.w3.scale for expert in local]),
            torch.stack([expert.w2.scale for expert in local]),
            expert_start=self.experts_start_idx,
            global_num_experts=self.n_routed_experts,
        )
        # The checkpoint-native tensors and Marlin tensors encode the same
        # weights. Retaining both would nearly double expert memory.
        self.experts = nn.ModuleList([None for _ in range(self.n_routed_experts)])
        return True

    @torch.inference_mode()
    def warmup_fused_experts(self) -> None:
        if self._fused_experts is None:
            return
        from torchinferno.kernels.deepseek_v4_marlin import fused_mxfp4_moe

        selected = torch.arange(
            self.experts_start_idx,
            self.experts_start_idx + self.n_activated_experts,
            dtype=torch.int32,
            device=self.gate.weight.device,
        ).unsqueeze(0)
        weights = torch.full(
            (1, self.n_activated_experts),
            1.0 / self.n_activated_experts,
            dtype=torch.float32,
            device=self.gate.weight.device,
        )
        hidden = torch.zeros(
            (1, self.dim), dtype=torch.bfloat16, device=self.gate.weight.device
        )
        fused_mxfp4_moe(
            hidden,
            weights,
            selected,
            self._fused_experts,
            clamp_limit=self.shared_experts.swiglu_limit,
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        if self._fused_experts is not None:
            from torchinferno.kernels.deepseek_v4_marlin import fused_mxfp4_moe

            y = fused_mxfp4_moe(
                x.contiguous(),
                weights.float().contiguous(),
                indices.contiguous(),
                self._fused_experts,
                clamp_limit=self.shared_experts.swiglu_limit,
            )
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
        if world_size > 1:
            _all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        shape, dtype = x.size(), x.dtype
        fused = os.environ.get(
            "TORCHINFERNO_V4_FUSED_HC_PRE_GEMM",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if x.is_cuda and x.size(1) == 1 and fused:
            mixes, square_sum = fused_hc_prenorm_gemm(x, hc_fn)
            rsqrt = torch.rsqrt(
                square_sum / (self.hc_mult * shape[-1]) + self.norm_eps
            )
            mixes *= rsqrt
            mix_input = x
        else:
            flat = x.flatten(2).float()
            rsqrt = torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            mixes = F.linear(flat, hc_fn) * rsqrt
            mix_input = flat.view(shape)
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * mix_input, dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        fused = os.environ.get(
            "TORCHINFERNO_V4_FUSED_HC_POST",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if x.is_cuda and x.size(1) == 1 and fused:
            return fused_hc_post(x, residual, post, comb)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor],
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, row_indices)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // world_size)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def get_logits(self, x: torch.Tensor, *, return_last_logits_only: bool) -> torch.Tensor:
        if return_last_logits_only:
            x = x[:, -1:, :]
        return F.linear(x.float(), self.weight)

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: RMSNorm,
        *,
        return_last_logits_only: bool,
        return_sharded_logits: bool,
    ) -> torch.Tensor:
        # x: [b,s,hc,d]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(
            norm(x), return_last_logits_only=return_last_logits_only
        )
        if world_size > 1 and not return_sharded_logits:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            _all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids, row_indices)
        logits = self.head(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm,
            return_last_logits_only=True,
            return_sharded_logits=False,
        )
        return logits


@dataclass(frozen=True)
class DeepSeekV4TensorParallelLoadReport:
    checkpoint: str
    rank: int
    world_size: int
    device: str
    max_batch_size: int
    max_seq_len: int
    loaded_parameters: int


class DeepSeekV4TensorParallelCache:
    cache_backend = "v4-heterogeneous"

    def __init__(
        self,
        owner: "DeepSeekV4TensorParallelForCausalLM",
        rows: tuple[int, ...],
        seq_lens: list[int],
        max_seq_len: int,
        *,
        parent: "DeepSeekV4TensorParallelCache | None" = None,
    ) -> None:
        self._owner = owner
        self._rows = rows
        self._seq_lens = seq_lens
        self._max_seq_len = max_seq_len
        self._parent_cache = parent
        self._root_cache = self if parent is None else parent._root_cache
        self._released = False

    @property
    def selected_rows(self) -> tuple[int, ...]:
        return self._rows

    @property
    def seq_len(self) -> int:
        if not self._rows:
            return 0
        value = self._seq_lens[self._rows[0]]
        if any(self._seq_lens[row] != value for row in self._rows):
            raise ValueError("selected V4 TP cache rows must have equal sequence lengths")
        return value

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def reset(self) -> None:
        self.set_seq_len(0)

    def release(self) -> None:
        self._owner.release_cache(self)

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "DeepSeekV4TensorParallelCache":
        mapped = tuple(self._rows[row] for row in rows)
        return DeepSeekV4TensorParallelCache(
            self._owner,
            mapped,
            self._seq_lens,
            self._max_seq_len,
            parent=self,
        )

    def clear_row(self, row: int) -> None:
        physical = self._rows[row]
        self._owner._clear_cache_row(physical)
        self._seq_lens[physical] = 0

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len == 0:
            for local_row in range(len(self._rows)):
                self.clear_row(local_row)
            return
        if seq_len != self.seq_len:
            raise ValueError("V4 TP cache cannot be rewound without an explicit prefix copy")

    def copy_prefix_from(
        self,
        other: "DeepSeekV4TensorParallelCache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if other._owner is not self._owner:
            raise ValueError("in-process V4 prefix copy requires caches from the same model")
        source = other._rows[source_row]
        destination = self._rows[dest_row]
        if other._seq_lens[source] != tokens:
            raise ValueError("V4 TP prefix sources must end exactly at the copied token")
        self._owner._copy_cache_prefix(source, destination, tokens)
        self._seq_lens[destination] = tokens


class DeepSeekV4TensorParallelForCausalLM(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Sets global state (world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__."""
    provenance_variant = "deepseek-v4:tp-v0"
    supports_padded_batch_prefill = False
    supports_prefix_cache = False
    supports_runtime_graphs = False
    owns_cache_rows = True

    def __init__(self, args: ModelArgs):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype
        world_size = (
            dist.get_world_size(group=_tensor_parallel_process_group)
            if dist.is_initialized()
            else 1
        )
        rank = (
            dist.get_rank(group=_tensor_parallel_process_group)
            if dist.is_initialized()
            else 0
        )
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.args = args
        self.tensor_parallel_rank = rank
        self.tensor_parallel_size = world_size
        self.rank = rank
        self.world_size = world_size
        self.is_tensor_parallel = world_size > 1
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.logits_dtype = torch.float32
        self.max_seq_len = args.max_seq_len
        self.cache_row_capacity = args.max_batch_size
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

        self.load_report: DeepSeekV4TensorParallelLoadReport | None = None
        self._cache_seq_lens = [0 for _ in range(args.max_batch_size)]
        self._free_cache_rows = set(range(args.max_batch_size))
        self._cache_row_lock = threading.Lock()

    @torch.inference_mode()
    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        row_indices: torch.Tensor,
        *,
        return_last_logits_only: bool,
        return_sharded_logits: bool,
    ) -> torch.Tensor:
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids, row_indices)
        logits = self.head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm,
            return_last_logits_only=return_last_logits_only,
            return_sharded_logits=return_sharded_logits,
        )
        return logits

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: DeepSeekV4TensorParallelCache | None = None,
        use_cache: bool = True,
        return_last_logits_only: bool = True,
        return_sharded_logits: bool = False,
    ) -> tuple[torch.Tensor, DeepSeekV4TensorParallelCache | None]:
        if not use_cache:
            raise ValueError("the production V4 TP path requires an explicit cache")
        if cache is None:
            cache = self.allocate_cache(input_ids.size(0), max_seq_len=self.max_seq_len)
        if cache._owner is not self or len(cache.selected_rows) != input_ids.size(0):
            raise ValueError("V4 TP cache rows must match the input batch")
        start_pos = cache.seq_len
        end_pos = start_pos + input_ids.size(1)
        if end_pos > cache.max_seq_len:
            raise ValueError("input exceeds the allocated V4 TP sequence capacity")
        row_indices = torch.tensor(cache.selected_rows, dtype=torch.long, device=input_ids.device)
        if start_pos > 0 and input_ids.size(1) > 1:
            chunks = []
            for offset in range(input_ids.size(1)):
                chunk = self._forward_logits(
                    input_ids[:, offset : offset + 1],
                    start_pos + offset,
                    row_indices,
                    return_last_logits_only=return_last_logits_only,
                    return_sharded_logits=return_sharded_logits,
                )
                if not return_last_logits_only or offset + 1 == input_ids.size(1):
                    chunks.append(chunk)
            logits = chunks[-1] if return_last_logits_only else torch.cat(chunks, dim=1)
        else:
            logits = self._forward_logits(
                input_ids,
                start_pos,
                row_indices,
                return_last_logits_only=return_last_logits_only,
                return_sharded_logits=return_sharded_logits,
            )
        for row in cache.selected_rows:
            cache._seq_lens[row] = end_pos
        return logits, cache

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_backend: str = "dense",
        page_size: int = 16,
        stacked_storage: bool | None = None,
    ) -> DeepSeekV4TensorParallelCache:
        del dtype, page_size, stacked_storage
        if cache_backend not in {"dense", "v4-heterogeneous"}:
            raise ValueError(f"unsupported V4 TP cache backend: {cache_backend}")
        if device is not None and torch.device(device) != self.device:
            raise ValueError("V4 TP cache must live on the model device")
        requested_seq_len = self.max_seq_len if max_seq_len is None else max_seq_len
        if batch_size > self.args.max_batch_size or requested_seq_len > self.max_seq_len:
            raise ValueError("requested V4 TP cache exceeds the preallocated capacity")
        with self._cache_row_lock:
            rows = ()
            for start in range(self.args.max_batch_size - batch_size + 1):
                candidate = tuple(range(start, start + batch_size))
                if all(row in self._free_cache_rows for row in candidate):
                    rows = candidate
                    self._free_cache_rows.difference_update(rows)
                    break
            if not rows:
                raise RuntimeError("V4 TP cache row capacity is exhausted")
        cache = DeepSeekV4TensorParallelCache(
            self,
            rows,
            self._cache_seq_lens,
            requested_seq_len,
        )
        for local_row in range(len(rows)):
            cache.clear_row(local_row)
        return cache

    def release_cache(self, cache: DeepSeekV4TensorParallelCache) -> None:
        if cache._owner is not self:
            raise ValueError("V4 TP cache does not belong to this model")
        root = cache._root_cache
        with self._cache_row_lock:
            if root._released:
                return
            for row in root.selected_rows:
                self._clear_cache_row(row)
                self._cache_seq_lens[row] = 0
            self._free_cache_rows.update(root.selected_rows)
            root._released = True

    def release_decode_graphs_for_cache(self, cache: object) -> None:
        del cache

    def _sample_next_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        from torchinferno.runtime.sampling import sample_next_token

        if (
            temperature <= 0
            and logits.size(-1) == self.head.part_vocab_size
            and world_size > 1
        ):
            local_values, local_indices = logits.max(dim=-1)
            global_indices = local_indices + self.tensor_parallel_rank * self.head.part_vocab_size
            candidate = torch.stack((local_values.float(), global_indices.float()), dim=-1)
            candidates = [torch.empty_like(candidate) for _ in range(world_size)]
            _all_gather(candidates, candidate)
            candidates = torch.stack(candidates)
            winner_rank = candidates[..., 0].argmax(dim=0)
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            return candidates[winner_rank, batch_indices, 1].long()
        if logits.size(-1) == self.args.vocab_size:
            full_logits = logits
        elif logits.size(-1) == self.head.part_vocab_size and world_size > 1:
            shards = [torch.empty_like(logits) for _ in range(world_size)]
            _all_gather(shards, logits.contiguous())
            full_logits = torch.cat(shards, dim=-1)
        else:
            raise ValueError("V4 TP sampling received an unexpected logits shard")
        if world_size == 1 or not dist.is_initialized():
            return sample_next_token(full_logits, temperature)
        token = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        if self.tensor_parallel_rank == 0:
            token.copy_(sample_next_token(full_logits, temperature))
        source = (
            dist.get_global_rank(_tensor_parallel_process_group, 0)
            if _tensor_parallel_process_group is not None
            else 0
        )
        dist.broadcast(token, src=source, group=_tensor_parallel_process_group)
        return token

    def disaggregated_cache_tensors(
        self,
        cache: DeepSeekV4TensorParallelCache,
        *,
        batch_size: int,
        tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        """Return the live heterogeneous state required to continue decode."""

        if cache._owner is not self or batch_size > len(cache.selected_rows):
            raise ValueError("V4 disaggregated cache does not belong to this model")
        rows = cache.selected_rows[:batch_size]
        if not rows or rows != tuple(range(rows[0], rows[0] + batch_size)):
            raise ValueError("V4 disaggregated transfer requires contiguous cache rows")
        row_slice = slice(rows[0], rows[0] + batch_size)
        tensors: list[torch.Tensor] = []

        def append_compressor_state(compressor: Compressor) -> None:
            ratio = compressor.compress_ratio
            remainder = tokens % ratio
            if compressor.overlap and tokens >= ratio:
                tensors.extend(
                    (
                        compressor.kv_state[row_slice, :ratio],
                        compressor.score_state[row_slice, :ratio],
                    )
                )
            if remainder:
                offset = ratio if compressor.overlap else 0
                tensors.extend(
                    (
                        compressor.kv_state[
                            row_slice, offset : offset + remainder
                        ],
                        compressor.score_state[
                            row_slice, offset : offset + remainder
                        ],
                    )
                )

        for layer in self.layers:
            attention = layer.attn
            window_tokens = min(tokens, attention.window_size)
            if window_tokens:
                tensors.append(attention.kv_cache[row_slice, :window_tokens])
            ratio = attention.compress_ratio
            if not ratio:
                continue
            complete = tokens // ratio
            if complete:
                start = attention.window_size
                tensors.append(attention.kv_cache[row_slice, start : start + complete])
            append_compressor_state(attention.compressor)
            if attention.indexer is not None:
                if complete:
                    tensors.append(attention.indexer.kv_cache[row_slice, :complete])
                append_compressor_state(attention.indexer.compressor)
        return tuple(tensors)

    def finalize_disaggregated_cache_import(
        self,
        cache: DeepSeekV4TensorParallelCache,
        *,
        tokens: int,
    ) -> None:
        if cache._owner is not self:
            raise ValueError("V4 disaggregated cache does not belong to this model")
        for row in cache.selected_rows:
            cache._seq_lens[row] = tokens

    def _clear_cache_row(self, row: int) -> None:
        for layer in self.layers:
            attn = layer.attn
            if attn.compress_ratio:
                attn.compressor.kv_state[row].zero_()
                attn.compressor.score_state[row].fill_(float("-inf"))
                if attn.indexer is not None:
                    attn.indexer.compressor.kv_state[row].zero_()
                    attn.indexer.compressor.score_state[row].fill_(float("-inf"))

    def _copy_cache_prefix(self, source: int, destination: int, tokens: int) -> None:
        for layer in self.layers:
            attn = layer.attn
            win = attn.window_size
            attn.kv_cache[destination, :win].copy_(attn.kv_cache[source, :win])
            if not attn.compress_ratio:
                continue
            complete = tokens // attn.compress_ratio
            if complete:
                attn.kv_cache[destination, win : win + complete].copy_(
                    attn.kv_cache[source, win : win + complete]
                )
            attn.compressor.kv_state[destination].copy_(attn.compressor.kv_state[source])
            attn.compressor.score_state[destination].copy_(attn.compressor.score_state[source])
            if attn.indexer is not None:
                index_complete = tokens // attn.indexer.compress_ratio
                if index_complete:
                    attn.indexer.kv_cache[destination, :index_complete].copy_(
                        attn.indexer.kv_cache[source, :index_complete]
                    )
                attn.indexer.compressor.kv_state[destination].copy_(
                    attn.indexer.compressor.kv_state[source]
                )
                attn.indexer.compressor.score_state[destination].copy_(
                    attn.indexer.compressor.score_state[source]
                )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
        cache_backend: str = "dense",
    ) -> torch.Tensor:
        cache = self.allocate_cache(
            input_ids.size(0),
            max_seq_len=input_ids.size(1) + max_new_tokens,
            cache_backend=cache_backend,
        )
        try:
            generated = input_ids
            logits, cache = self(
                input_ids,
                cache=cache,
                return_sharded_logits=world_size > 1,
            )
            for step in range(max_new_tokens):
                token = self._sample_next_token(logits[:, -1], temperature)
                generated = torch.cat((generated, token[:, None]), dim=1)
                if eos_token_id is not None and bool(torch.all(token == eos_token_id)):
                    break
                if step + 1 < max_new_tokens:
                    logits, cache = self(
                        token[:, None],
                        cache=cache,
                        return_sharded_logits=world_size > 1,
                    )
            return generated
        finally:
            self.release_cache(cache)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        enable_mtp: bool = False,
    ) -> "DeepSeekV4TensorParallelForCausalLM":
        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeek V4 tensor parallelism requires CUDA")
        path = resolve_pretrained_path(
            checkpoint,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
        config = DeepSeekV4Config.from_dict(load_config(path))
        target = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        args = _model_args_from_config(
            config,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            enable_mtp=enable_mtp,
        )
        with torch.device(target), set_dtype(torch.bfloat16):
            model = cls(args)
        model.config = config
        model.dtype = torch.bfloat16
        model.checkpoint = str(path)
        loaded = model._load_checkpoint(path)
        model.load_report = DeepSeekV4TensorParallelLoadReport(
            checkpoint=str(path),
            rank=model.tensor_parallel_rank,
            world_size=model.tensor_parallel_size,
            device=str(target),
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            loaded_parameters=loaded,
        )
        return model

    def _load_checkpoint(self, path: str | Path) -> int:
        loaded = 0
        with CheckpointTensorLoader(path) as loader, torch.no_grad():
            for name, parameter in self.named_parameters():
                tensor = _load_parameter_tensor(
                    loader,
                    name,
                    parameter,
                    rank=self.tensor_parallel_rank,
                    world_size=self.tensor_parallel_size,
                    device=self.device,
                )
                parameter.copy_(tensor)
                loaded += 1
        fused_layers = 0
        for layer in (*self.layers, *self.mtp):
            fused_layers += int(layer.ffn.prepare_fused_experts())
        self.cuda_expert_backend = "marlin-mxfp4" if fused_layers else "tilelang-mxfp4"
        if fused_layers:
            self.layers[0].ffn.warmup_fused_experts()
            torch.cuda.synchronize(self.device)
        return loaded


def _model_args_from_config(
    config: DeepSeekV4Config,
    *,
    max_batch_size: int,
    max_seq_len: int,
    enable_mtp: bool,
) -> ModelArgs:
    if max_seq_len > config.max_position_embeddings:
        raise ValueError("max_seq_len exceeds the checkpoint context limit")
    return ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype="fp4",
        scale_dtype="fp32",
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_layers=config.num_hidden_layers,
        n_hash_layers=config.num_hash_layers,
        n_mtp_layers=config.num_nextn_predict_layers if enable_mtp else 0,
        n_heads=config.num_attention_heads,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        score_func=config.scoring_func,
        route_scale=config.routed_scaling_factor,
        swiglu_limit=config.swiglu_limit,
        q_lora_rank=config.q_lora_rank,
        head_dim=config.head_dim,
        rope_head_dim=config.qk_rope_head_dim,
        norm_eps=config.rms_norm_eps,
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        window_size=config.sliding_window,
        compress_ratios=config.compress_ratios,
        compress_rope_theta=config.compress_rope_theta,
        original_seq_len=config.rope_original_max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_factor=config.rope_factor,
        beta_fast=int(config.rope_beta_fast),
        beta_slow=int(config.rope_beta_slow),
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        hc_mult=config.hc_mult,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
    )


def _load_parameter_tensor(
    loader: CheckpointTensorLoader,
    name: str,
    parameter: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    if name.endswith(".attn.wo_a.weight"):
        weight = loader.shard(
            name,
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_scale = name.removesuffix("weight") + "scale"
        weight_scale = loader.shard(
            source_scale,
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=torch.float32,
        )
        return dequantize_block_fp8(weight, weight_scale).to(dtype=parameter.dtype)

    output_sharded = (
        name in {"embed.weight", "head.weight"}
        or name.endswith(".attn.attn_sink")
        or name.endswith(".attn.wq_b.weight")
        or name.endswith(".attn.wq_b.scale")
        or name.endswith(".attn.indexer.wq_b.weight")
        or name.endswith(".attn.indexer.wq_b.scale")
        or name.endswith(".attn.indexer.weights_proj.weight")
    )
    input_sharded = name.endswith(".attn.wo_b.weight") or name.endswith(".attn.wo_b.scale")
    requested_dtype = parameter.dtype if parameter.dtype != torch.float4_e2m1fn_x2 else None
    if output_sharded:
        tensor = loader.shard(
            name,
            dim=0,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=requested_dtype,
        )
    elif input_sharded:
        tensor = loader.shard(
            name,
            dim=1,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=requested_dtype,
        )
    else:
        tensor = loader.tensor(name, device=device, dtype=requested_dtype)
    if parameter.dtype == torch.float4_e2m1fn_x2:
        tensor = tensor.view(torch.float4_e2m1fn_x2)
    elif tensor.dtype != parameter.dtype:
        tensor = tensor.to(dtype=parameter.dtype)
    if tuple(tensor.shape) != tuple(parameter.shape):
        raise ValueError(
            f"checkpoint shape mismatch for {name}: source={tuple(tensor.shape)} target={tuple(parameter.shape)}"
        )
    return tensor
