"""FP8 (e4m3) W8A8 GEMM for COMPUTE-bound PREFILL, via torch._scaled_mm.

Prefill is compute-bound (large M = tokens) where FP8 tensor cores (~2x bf16 peak)
are the biggest TTFT lever. Two findings drive this module (scripts/bench_fp8_prefill.py):
  1. Scales MUST be tensorwise (scalar). Rowwise _scaled_mm hits a slow kernel (~0.3x);
     tensorwise hits the fast cublasLt path = ~2x.
  2. The activation quant must be FUSED. Eager amax+div+cast is ~96us (3 kernels + a
     16MB intermediate) and eats the GEMM win (net ~1.07x). The fused Triton scale+cast
     kernel below (dynamic amax via torch, then one pass to fp8) restores it: gate_up
     1.97x @ M=512, 1.4x @ M=1024-2048. Dynamic amax => SAFE (per-call, no clipping).

Correctness: FP8 prefill is greedy-EXACT vs bf16 (scripts/validate_fp8_prefill_correctness.py)
-- the KV + first logits it writes are accurate enough that bf16 decode reproduces the
sequence. FP8 DECODE is lossy (accumulates), so this is a PREFILL-ONLY lever; decode stays
bf16/marlin. Use for the big K-large GEMMs (gate_up, down); small qkv/o_proj barely benefit.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP8_E4M3 = torch.float8_e4m3fn
_FP8_MAX = 448.0

_triton_ok: bool | None = None
_scale_cast_kernel = None


def _load_triton() -> bool:
    # Lazily compile the fused scale+cast kernel. Returns False if Triton is absent
    # (caller falls back to bf16). Cached so the import/JIT cost is paid once.
    global _triton_ok, _scale_cast_kernel
    if _triton_ok is not None:
        return _triton_ok
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _scale_cast_fp8(x_ptr, out_ptr, inv_scale_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            off = pid * BLOCK + tl.arange(0, BLOCK)
            mask = off < n
            inv = tl.load(inv_scale_ptr)
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            y = (x * inv).to(tl.float8e4nv)
            tl.store(out_ptr + off, y, mask=mask)

        _scale_cast_kernel = _scale_cast_fp8
        _triton_ok = True
    except Exception:
        _triton_ok = False
    return _triton_ok


def quantize_weight_fp8(weight: Tensor) -> tuple[Tensor, Tensor]:
    # One-time tensorwise e4m3 quant of a [N,K] weight (the layout torch stores for
    # F.linear: out = x @ weight.t()). Returns (wq [N,K] fp8, scale scalar f32). At
    # GEMM time pass wq.t() -> [K,N] column-major (stride(0)==1), the layout _scaled_mm
    # needs for b. Done eagerly BEFORE any CUDA-graph capture (allocates).
    scale = (weight.abs().amax() / _FP8_MAX).clamp(min=1e-6).to(torch.float32)
    wq = (weight.to(torch.float32) / scale).to(FP8_E4M3)
    return wq, scale


def quantize_activation_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    # Dynamic tensorwise quant of activation x (any shape, contiguous last dim) via the
    # fused Triton scale+cast kernel: amax (torch, fast reduction) then ONE pass to fp8
    # (avoids the eager div's 16MB bf16 intermediate). Dynamic => no clipping on unseen
    # inputs. Returns (xq same-shape fp8, scale scalar f32).
    import triton

    scale = (x.abs().amax() / _FP8_MAX).clamp(min=1e-6).to(torch.float32)
    inv = (1.0 / scale).to(torch.float32)
    xq = torch.empty(x.shape, device=x.device, dtype=FP8_E4M3)
    n = x.numel()
    block = 4096
    _scale_cast_kernel[(triton.cdiv(n, block),)](x, xq, inv, n, block, num_warps=8)
    return xq, scale


def fp8_prefill_linear(x: Tensor, wq: Tensor, weight_scale: Tensor) -> Tensor:
    # out[.., N] = (x @ weight.t()) via fp8 e4m3 _scaled_mm. x is bf16 [.., K]; wq is the
    # pre-quantized [N,K] fp8 weight; weight_scale its scalar. Quantizes x dynamically
    # (fused), runs the tensorwise-scaled GEMM, returns bf16. 2D-reshapes x as needed.
    k = wq.size(1)
    x2d = x.reshape(-1, k)
    xq, sa = quantize_activation_fp8(x2d)
    out = torch._scaled_mm(
        xq, wq.t(), scale_a=sa, scale_b=weight_scale, out_dtype=torch.bfloat16
    )
    return out.reshape(*x.shape[:-1], wq.size(0))


def fp8_available() -> bool:
    return _load_triton()
