"""GPU-gated correctness test for the self-contained Marlin int4 kernel module.

Skips unless CUDA is present AND vLLM's prebuilt _C (with marlin_gemm) can be
loaded via torch.ops.load_library (TORCHINFERNO_VLLM_C_SO). Validates that
marlin_int4_mm(quantize_to_marlin_int4(w)) matches a @ dequant(w) -- i.e. the
copied weight-prep + the kernel are self-consistent, with no vllm python import.
"""

import pytest
import torch

from torchinferno.kernels import marlin


def test_marlin_int4_mm_can_use_provided_output_buffer(monkeypatch):
    a = torch.ones((2, 4), dtype=torch.bfloat16)
    q_weight = torch.empty((1, 1), dtype=torch.int32)
    scales = torch.ones((1, 8), dtype=torch.bfloat16)
    workspace = torch.empty((1,), dtype=torch.int32)
    provided = torch.empty((2, 8), dtype=torch.bfloat16)
    expected = torch.arange(16, dtype=torch.float32).view(2, 8).to(torch.bfloat16)
    calls = []

    def fake_marlin_gemm(a_arg, out_arg, *args):
        assert a_arg is a
        assert out_arg is provided
        out_arg.copy_(expected)
        calls.append(out_arg)
        return out_arg

    monkeypatch.setattr(torch.ops._C, "marlin_gemm", fake_marlin_gemm, raising=False)

    out = marlin.marlin_int4_mm(a, q_weight, scales, workspace, 8, 4, out=provided)

    assert out is provided
    assert calls == [provided]
    torch.testing.assert_close(out, expected)


def _dequant_ref(a, w, group=128):
    # Reproduce the module's symmetric int4 (bias-8) quant as a bf16 reference.
    K, N = w.shape
    wg = w.reshape((-1, group, N)).permute(1, 0, 2).reshape((group, -1)).float()
    mx = wg.max(0, keepdim=True).values
    mn = wg.min(0, keepdim=True).values
    s = torch.max((mx / 7.0).abs(), (mn / -8.0).abs()).clamp(min=1e-8)
    q = torch.round(wg / s).clamp(-8, 7)
    deq = (q * s).reshape((group, -1, N)).permute(1, 0, 2).reshape((K, N))
    return a.float() @ deq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="marlin int4 needs CUDA")
def test_marlin_int4_mm_matches_dequant_reference():
    if not marlin.load_marlin_ops():
        pytest.skip("vLLM marlin_gemm op unavailable (set TORCHINFERNO_VLLM_C_SO)")
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    # gate_up and down shapes (N % 64 == 0, K % 16 == 0)
    for N, K in [(7168, 8192), (8192, 3584)]:
        assert marlin.marlin_supports_shape(N, K)
        w = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02
        a = torch.randn(48, K, device=dev, dtype=torch.bfloat16)
        q, s = marlin.quantize_to_marlin_int4(w, 128)
        ws = marlin.make_workspace(N, dev)
        out = marlin.marlin_int4_mm(a, q, s, ws, N, K)
        ref = _dequant_ref(a, w).to(out.dtype)
        assert out.shape == (48, N)
        rel = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
        # marlin output vs our own dequant ref: small (rounding-tie + fp16 accum)
        assert rel < 0.03, f"N={N} K={K} rel-err {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="marlin int4 needs CUDA")
def test_marlin_rejects_uneligible_shape():
    # lm_head vocab shard (N=16032) is not divisible by 64 -> not marlin-eligible.
    assert not marlin.marlin_supports_shape(16032, 8192)
