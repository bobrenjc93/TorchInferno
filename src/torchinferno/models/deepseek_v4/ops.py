from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=16)
def _cached_freqs(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> Tensor:
    def correction_dim(rotations: float) -> float:
        return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
    angles = torch.outer(torch.arange(seqlen, dtype=torch.float32), inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    *,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device | None = None,
) -> Tensor:
    return _cached_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow).to(device=device)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, *, inverse: bool = False) -> Tensor:
    """Apply DeepSeek's interleaved complex RoPE without mutating ``x``."""

    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if freqs.ndim == 1:
        freqs = freqs.view(*([1] * (complex_x.ndim - 1)), freqs.size(0))
    elif freqs.ndim == 2:
        if complex_x.ndim < 3:
            raise ValueError("position-indexed RoPE requires a sequence dimension")
        freqs = freqs.view(1, freqs.size(0), *([1] * (complex_x.ndim - 3)), freqs.size(1))
    else:
        raise ValueError("freqs_cis must have one or two dimensions")
    rotated = torch.view_as_real(complex_x * freqs).flatten(-2)
    return rotated.to(dtype=x.dtype)


def hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure torch reference for the released mHC split/Sinkhorn operator."""

    pre_logits, post_logits, comb_logits = torch.split(
        mixes,
        (hc_mult, hc_mult, hc_mult * hc_mult),
        dim=-1,
    )
    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = comb.unflatten(-1, (hc_mult, hc_mult)).softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def attention_with_sink(q: Tensor, kv: Tensor, sink: Tensor, scale: float) -> Tensor:
    """MQA over selected shared KV vectors with a learned zero-value sink."""

    scores = torch.einsum("bhd,btd->bht", q.float(), kv.float()) * scale
    sink_scores = sink.float().view(1, -1, 1).expand(scores.size(0), -1, 1)
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bht,btd->bhd", probabilities, kv.float()).to(dtype=q.dtype)


def _power_of_two_scale(amax: Tensor, max_value: float) -> Tensor:
    minimum = max_value * (2.0**-126)
    return torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(minimum) / max_value)))


def fake_quant_fp8(x: Tensor, block_size: int = 64) -> Tensor:
    """Power-of-two block FP8 quantize/dequantize used by V4 QAT caches."""

    if x.size(-1) % block_size:
        raise ValueError("FP8 fake-quant dimension must be divisible by block_size")
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, block_size)
    scale = _power_of_two_scale(blocks.abs().amax(dim=-1, keepdim=True), 448.0)
    quantized = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(shape).to(dtype=x.dtype)


_FP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def fake_quant_fp4(x: Tensor, block_size: int = 32) -> Tensor:
    """Pure torch E2M1 block quantize/dequantize reference."""

    if x.size(-1) % block_size:
        raise ValueError("FP4 fake-quant dimension must be divisible by block_size")
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, block_size)
    scale = _power_of_two_scale(blocks.abs().amax(dim=-1, keepdim=True), 6.0)
    normalized = (blocks / scale).clamp(-6, 6)
    levels = _FP4_LEVELS.to(device=x.device)
    indices = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return (levels[indices] * scale).reshape(shape).to(dtype=x.dtype)


def hadamard_transform(x: Tensor) -> Tensor:
    """Normalized Walsh-Hadamard transform for power-of-two feature sizes."""

    width = x.size(-1)
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    shape = x.shape
    y = x.float().reshape(-1, width)
    stride = 1
    while stride < width:
        grouped = y.reshape(-1, width // (2 * stride), 2, stride)
        left, right = grouped.unbind(dim=2)
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, width)
        stride *= 2
    return (y / math.sqrt(width)).reshape(shape).to(dtype=x.dtype)


def unpack_mxfp4(packed: Tensor) -> Tensor:
    """Unpack public checkpoint low/high E2M1 nibbles along logical K."""

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    values = packed.view(torch.uint8)
    low = table[(values & 0x0F).long()]
    high = table[((values >> 4) & 0x0F).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def dequantize_mxfp4(packed: Tensor, scale: Tensor) -> Tensor:
    logical = unpack_mxfp4(packed)
    if logical.size(-1) != scale.size(-1) * 32:
        raise ValueError("MXFP4 scale must cover 32 logical K values")
    expanded = scale.float().repeat_interleave(32, dim=-1)
    return logical * expanded


def dequantize_block_fp8(weight: Tensor, scale: Tensor, block_size: int = 128) -> Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("block FP8 tensors must be matrices")
    rows, cols = weight.shape
    expected = (math.ceil(rows / block_size), math.ceil(cols / block_size))
    if tuple(scale.shape) != expected:
        raise ValueError(f"expected FP8 scale shape {expected}, got {tuple(scale.shape)}")
    expanded = scale.float().repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return weight.float() * expanded[:rows, :cols]
