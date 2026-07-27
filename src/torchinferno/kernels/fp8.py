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
FP8_E4M3_MAX = 448.0
_FP8_MAX = FP8_E4M3_MAX

_triton_ok: bool | None = None
_scale_cast_kernel = None
_row_scale_cast_kernel = None
_sgl_per_token_ops: tuple[object, object] | bool | None = None


def _load_triton() -> bool:
    # Lazily compile the fused scale+cast kernel. Returns False if Triton is absent
    # (caller falls back to bf16). Cached so the import/JIT cost is paid once.
    global _triton_ok, _scale_cast_kernel, _row_scale_cast_kernel
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

        @triton.jit
        def _row_scale_cast_fp8(
            x_ptr,
            out_ptr,
            scale_ptr,
            cols: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            offsets = tl.arange(0, BLOCK)
            mask = offsets < cols
            row_offsets = row * cols + offsets
            values = tl.load(
                x_ptr + row_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 448.0, 1e-10)
            tl.store(out_ptr + row_offsets, values / scale, mask=mask)
            tl.store(scale_ptr + row, scale)

        _scale_cast_kernel = _scale_cast_fp8
        _row_scale_cast_kernel = _row_scale_cast_fp8
        _triton_ok = True
    except Exception:
        _triton_ok = False
    return _triton_ok


def _load_sgl_per_token_ops() -> tuple[object, object] | None:
    """Return compatible SGL FP8 ops, caching import/ABI failures."""

    global _sgl_per_token_ops
    if _sgl_per_token_ops is False:
        return None
    if isinstance(_sgl_per_token_ops, tuple):
        return _sgl_per_token_ops
    try:
        from sgl_kernel import fp8_scaled_mm, sgl_per_token_quant_fp8
    except Exception:
        _sgl_per_token_ops = False
        return None
    _sgl_per_token_ops = (sgl_per_token_quant_fp8, fp8_scaled_mm)
    return _sgl_per_token_ops


def _quantize_activation_fp8_per_token(
    x: Tensor,
    output: Tensor,
    scale: Tensor,
) -> None:
    if not _load_triton():
        raise RuntimeError("per-token FP8 activation quantization requires Triton")
    import triton

    cols = int(x.size(1))
    _row_scale_cast_kernel[(x.size(0),)](
        x,
        output,
        scale,
        cols,
        triton.next_power_of_2(cols),
        num_warps=8,
    )


def quantize_weight_fp8(weight: Tensor) -> tuple[Tensor, Tensor]:
    # One-time tensorwise e4m3 quant of a [N,K] weight (the layout torch stores for
    # F.linear: out = x @ weight.t()). Returns (wq [N,K] fp8, scale scalar f32). At
    # GEMM time pass wq.t() -> [K,N] column-major (stride(0)==1), the layout _scaled_mm
    # needs for b. Done eagerly BEFORE any CUDA-graph capture (allocates).
    scale = (weight.abs().amax() / _FP8_MAX).clamp(min=1e-6).to(torch.float32)
    wq = (weight.to(torch.float32) / scale).to(FP8_E4M3)
    return wq, scale


def quantize_weight_fp8_per_channel(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize an [N, K] weight with one FP8 scale per output channel."""

    if weight.ndim != 2:
        raise ValueError("FP8 linear weights must have shape [N, K]")
    scale = (
        weight.abs().amax(dim=1, keepdim=True).to(torch.float32) / _FP8_MAX
    ).clamp(min=1e-10)
    weight_q = (weight.to(torch.float32) / scale).to(FP8_E4M3)
    return weight_q, scale.transpose(0, 1).contiguous()


def quantize_activation_fp8(
    x: Tensor,
    *,
    scale: Tensor | None = None,
    inverse_scale: Tensor | None = None,
    out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    # Dynamic tensorwise quant of activation x (any shape, contiguous last dim) via the
    # fused Triton scale+cast kernel: amax (torch, fast reduction) then ONE pass to fp8
    # (avoids the eager div's 16MB bf16 intermediate). Dynamic => no clipping on unseen
    # inputs. Returns (xq same-shape fp8, scale scalar f32).
    if not _load_triton():
        raise RuntimeError("FP8 activation quantization requires Triton")
    import triton

    if scale is None:
        scale = (x.abs().amax() / _FP8_MAX).clamp(min=1e-6).to(torch.float32)
    elif scale.numel() != 1 or scale.device != x.device:
        raise ValueError("FP8 activation scale must be a scalar on the input device")
    else:
        scale = scale.to(torch.float32)
    if inverse_scale is None:
        inverse_scale = scale.reciprocal()
    elif inverse_scale.numel() != 1 or inverse_scale.device != x.device:
        raise ValueError("FP8 inverse activation scale must be a scalar on the input device")
    else:
        inverse_scale = inverse_scale.to(torch.float32)
    xq = out if out is not None else torch.empty(x.shape, device=x.device, dtype=FP8_E4M3)
    if xq.shape != x.shape or xq.device != x.device or xq.dtype != FP8_E4M3:
        raise ValueError("FP8 activation output must match the input shape and device")
    n = x.numel()
    block = 4096
    _scale_cast_kernel[(triton.cdiv(n, block),)](
        x,
        xq,
        inverse_scale,
        n,
        block,
        num_warps=8,
    )
    return xq, scale


def fp8_prefill_linear(
    x: Tensor,
    wq: Tensor,
    weight_scale: Tensor,
    *,
    activation_scale: Tensor | None = None,
    activation_inverse_scale: Tensor | None = None,
) -> Tensor:
    # out[.., N] = (x @ weight.t()) via fp8 e4m3 _scaled_mm. x is bf16 [.., K]; wq is the
    # pre-quantized [N,K] fp8 weight; weight_scale its scalar. Quantizes x dynamically
    # (fused), runs the tensorwise-scaled GEMM, returns bf16. 2D-reshapes x as needed.
    k = wq.size(1)
    x2d = x.reshape(-1, k)
    xq, sa = quantize_activation_fp8(
        x2d,
        scale=activation_scale,
        inverse_scale=activation_inverse_scale,
    )
    out = torch._scaled_mm(
        xq, wq.t(), scale_a=sa, scale_b=weight_scale, out_dtype=torch.bfloat16
    )
    return out.reshape(*x.shape[:-1], wq.size(0))


def fp8_per_token_linear(
    x: Tensor,
    weight_q: Tensor,
    weight_scale: Tensor,
    *,
    out: Tensor | None = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """Run Cutlass W8A8 with dynamic per-token and per-channel scales."""

    if x.ndim < 2 or weight_q.ndim != 2:
        raise ValueError("FP8 linear expects an activation matrix and [N, K] weight")
    if weight_q.dtype != FP8_E4M3 or weight_q.device != x.device:
        raise ValueError("FP8 weight must use e4m3 on the activation device")
    if weight_q.size(1) != x.size(-1):
        raise ValueError("FP8 activation and weight inner dimensions must match")
    if weight_scale.shape != (1, weight_q.size(0)) or weight_scale.device != x.device:
        raise ValueError("per-channel FP8 weight scale must have shape [1, N]")

    x_2d = x.reshape(-1, x.size(-1)).contiguous()
    x_q = torch.empty_like(x_2d, dtype=FP8_E4M3)
    x_scale = torch.empty(
        (x_2d.size(0), 1),
        dtype=torch.float32,
        device=x.device,
    )
    sgl_ops = _load_sgl_per_token_ops()
    if sgl_ops is None:
        _quantize_activation_fp8_per_token(x_2d, x_q, x_scale)
    else:
        sgl_ops[0](x_2d, x_q, x_scale)
    return fp8_per_token_linear_quantized(
        x_q.view(*x.shape[:-1], x.size(-1)),
        x_scale,
        weight_q,
        weight_scale,
        out_dtype=x.dtype,
        out=out,
        use_fast_accum=use_fast_accum,
    )


def fp8_per_token_linear_quantized(
    x_q: Tensor,
    x_scale: Tensor,
    weight_q: Tensor,
    weight_scale: Tensor,
    *,
    out_dtype: torch.dtype,
    out: Tensor | None = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """Run Cutlass W8A8 from an already per-token-quantized activation."""

    if x_q.ndim < 2 or x_q.dtype != FP8_E4M3:
        raise ValueError("quantized FP8 activation must be an e4m3 matrix")
    if weight_q.ndim != 2 or weight_q.dtype != FP8_E4M3:
        raise ValueError("FP8 weight must be an e4m3 matrix")
    if x_q.device != weight_q.device or x_q.size(-1) != weight_q.size(1):
        raise ValueError("FP8 activation and weight must have matching devices and inner dimensions")
    rows = x_q.numel() // x_q.size(-1)
    if x_scale.shape != (rows, 1) or x_scale.device != x_q.device:
        raise ValueError("per-token FP8 activation scale must have shape [rows, 1]")
    if weight_scale.shape != (1, weight_q.size(0)) or weight_scale.device != x_q.device:
        raise ValueError("per-channel FP8 weight scale must have shape [1, N]")
    x_2d = x_q.reshape(rows, x_q.size(-1))
    out_2d: Tensor | None = None
    if out is not None:
        expected_shape = (*x_q.shape[:-1], weight_q.size(0))
        if out.shape != expected_shape or out.device != x_q.device or out.dtype != out_dtype:
            raise ValueError("FP8 output buffer must match the projected output")
        out_2d = out.reshape(rows, weight_q.size(0))
    sgl_ops = (
        _load_sgl_per_token_ops()
        if use_fast_accum and out_2d is None
        else None
    )
    if out_2d is not None:
        projected = None
        if use_fast_accum:
            from torchinferno.kernels.sgl_fp8_out import scaled_mm_out

            projected = scaled_mm_out(
                out_2d,
                x_2d,
                weight_q.t(),
                x_scale,
                weight_scale,
            )
        if projected is None:
            torch.ops.aten._scaled_mm.out(
                x_2d,
                weight_q.t(),
                x_scale,
                weight_scale,
                out_dtype=out_dtype,
                use_fast_accum=use_fast_accum,
                out=out_2d,
            )
        projected = out_2d
    elif not use_fast_accum or sgl_ops is None:
        projected = torch._scaled_mm(
            x_2d,
            weight_q.t(),
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    else:
        projected = sgl_ops[1](
            x_2d,
            weight_q.t(),
            x_scale,
            weight_scale,
            out_dtype=out_dtype,
        )
    if out is not None:
        return out
    return projected.reshape(*x_q.shape[:-1], weight_q.size(0))


def fp8_available() -> bool:
    return _load_triton()
