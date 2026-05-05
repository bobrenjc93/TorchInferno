import os
import subprocess
import sys

import pytest
import torch

from torchinferno.graph import PassRegistry
from torchinferno.kernels import (
    KernelBackend,
    KernelConfig,
    dequantize_nvfp4,
    nvfp4_linear_reference,
    paged_decode_attention,
    quantize_nvfp4,
)
from torchinferno.kernels.ops import triton_available
from torchinferno.kernels.passes import register_kernel_replacement_passes
from torchinferno.research.benchmarks import benchmark_callable
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import paged_causal_attention


def test_nvfp4_quantization_and_linear_reference() -> None:
    torch.manual_seed(40)
    weight = torch.randn(7, 9)
    x = torch.randn(3, 9)
    quantized = quantize_nvfp4(weight, block_size=8)
    dequantized = dequantize_nvfp4(quantized)
    actual = nvfp4_linear_reference(x, quantized)
    expected = x @ dequantized.t()

    assert quantized.packed.dtype == torch.uint8
    assert quantized.shape == tuple(weight.shape)
    assert dequantized.shape == weight.shape
    torch.testing.assert_close(actual, expected)


def test_kernel_replacement_pass_registry_includes_nvfp4_hook() -> None:
    registry = PassRegistry()

    register_kernel_replacement_passes(registry)

    assert "swiglu-reference-to-kernel" in registry.names()
    assert "nvfp4-linear-reference-marker" in registry.names()


def test_paged_decode_attention_torch_backend_matches_reference() -> None:
    torch.manual_seed(41)
    heads = 3
    seq_len = 7
    head_dim = 5
    value_dim = 4
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.randn(heads, seq_len, head_dim)
    values = torch.randn(heads, seq_len, value_dim)
    query = torch.randn(heads, 1, head_dim)
    cache.append("req", keys, values)

    actual = paged_decode_attention(query, cache, "req", seq_len - 1, config=KernelConfig(backend=KernelBackend.TORCH))
    expected = paged_causal_attention(query, cache, "req", torch.tensor([seq_len - 1]))

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton unavailable")
def test_triton_paged_decode_attention_matches_reference() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    heads = 4
    seq_len = 17
    head_dim = 16
    value_dim = 12
    cache = PagedKVCache(
        num_pages=5,
        page_size=4,
        num_key_value_heads=heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=device,
        dtype=torch.float32,
    )
    keys = torch.randn(heads, seq_len, head_dim, device=device)
    values = torch.randn(heads, seq_len, value_dim, device=device)
    query = torch.randn(heads, 1, head_dim, device=device)
    cache.append("req", keys, values)

    actual = paged_decode_attention(query, cache, "req", seq_len - 1, config=KernelConfig(backend=KernelBackend.TRITON))
    expected = paged_causal_attention(query, cache, "req", torch.tensor([seq_len - 1], device=device))

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_benchmark_callable_and_perf_cli() -> None:
    result = benchmark_callable("noop", lambda: 1 + 1, warmup=1, iters=2, device=torch.device("cpu"))
    env = {**os.environ, "PYTHONPATH": "src"}
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "perf-smoke",
            "--device",
            "cpu",
            "--heads",
            "2",
            "--seq-len",
            "8",
            "--head-dim",
            "8",
            "--value-dim",
            "8",
            "--iters",
            "2",
            "--warmup",
            "1",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.mean_ms >= 0
    assert "TorchInferno performance smoke" in cli.stdout
