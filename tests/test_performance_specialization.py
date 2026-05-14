import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from torchinferno.graph import PassRegistry, trace_with_make_fx
from torchinferno.kernels import (
    KernelBackend,
    KernelConfig,
    dequantize_nvfp4,
    fused_rmsnorm_swiglu,
    fused_rmsnorm_swiglu_reference,
    nvfp4_linear_reference,
    paged_decode_attention,
    quantize_nvfp4,
)
from torchinferno.kernels.ops import triton_available
from torchinferno.kernels.passes import register_kernel_replacement_passes
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelCache,
    Llama3TensorParallelLayerKVCache,
    _decode_attention_block_size,
    _decode_linear,
    _should_use_decode_step_graph,
    _should_use_decode_step_logits_graph,
)
from torchinferno.research.benchmarks import benchmark_callable
from torchinferno.research.helion import (
    HelionCandidateConfig,
    HelionDecisionStore,
    HelionRegionSearchConfig,
    run_helion_candidate,
    run_helion_fx_search,
    run_helion_region_search,
)
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import paged_causal_attention


def _reference_rmsnorm_swiglu_region(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    hidden = x + residual
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(variance + 1e-6).to(dtype=hidden.dtype) * norm_weight
    gate = normed * gate_weight
    up = normed * up_weight
    return F.silu(gate) * up


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

    assert "fused-rmsnorm-swiglu-symbolic-subgraph" in registry.names()
    assert "fused-rmsnorm-swiglu-aten-subgraph" in registry.names()
    assert "swiglu-reference-to-kernel" in registry.names()
    assert "nvfp4-linear-reference-marker" in registry.names()


def test_fused_rmsnorm_swiglu_matches_reference_on_cpu() -> None:
    torch.manual_seed(43)
    x = torch.randn(3, 16)
    residual = torch.randn(3, 16)
    norm_weight = torch.randn(16)
    gate_weight = torch.randn(16)
    up_weight = torch.randn(16)

    expected = fused_rmsnorm_swiglu_reference(
        x,
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        eps=1e-6,
    )
    actual = fused_rmsnorm_swiglu(
        x,
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        eps=1e-6,
        config=KernelConfig(backend=KernelBackend.TORCH),
    )
    custom_op_actual = fused_rmsnorm_swiglu(x, residual, norm_weight, gate_weight, up_weight, eps=1e-6)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(custom_op_actual, expected)


def test_symbolic_subgraph_pass_replaces_rmsnorm_swiglu_region() -> None:
    torch.manual_seed(44)
    args = (
        torch.randn(3, 16),
        torch.randn(3, 16),
        torch.randn(16),
        torch.randn(16),
        torch.randn(16),
    )
    graph_module = torch.fx.symbolic_trace(_reference_rmsnorm_swiglu_region)
    registry = PassRegistry()
    register_kernel_replacement_passes(registry)

    optimized = registry.run(graph_module)

    assert optimized.meta["fused_rmsnorm_swiglu_symbolic_matches"] == 1
    assert any(
        node.op == "call_function" and node.target == torch.ops.torchinferno.fused_rmsnorm_swiglu
        for node in optimized.graph.nodes
    )
    torch.testing.assert_close(optimized(*args), _reference_rmsnorm_swiglu_region(*args))


def test_make_fx_subgraph_pass_replaces_aten_rmsnorm_swiglu_region() -> None:
    torch.manual_seed(45)
    args = (
        torch.randn(4, 24),
        torch.randn(4, 24),
        torch.randn(24),
        torch.randn(24),
        torch.randn(24),
    )
    graph_module = trace_with_make_fx(_reference_rmsnorm_swiglu_region, *args, fake=False)
    registry = PassRegistry()
    register_kernel_replacement_passes(registry)

    optimized = registry.run(graph_module)

    assert optimized.meta["fused_rmsnorm_swiglu_aten_matches"] == 1
    assert any(
        node.op == "call_function" and node.target == torch.ops.torchinferno.fused_rmsnorm_swiglu.default
        for node in optimized.graph.nodes
    )
    torch.testing.assert_close(optimized(*args), _reference_rmsnorm_swiglu_region(*args))


def test_fused_rmsnorm_swiglu_custom_op_has_fake_tensor_trace() -> None:
    args = (
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(8),
        torch.randn(8),
        torch.randn(8),
    )

    graph_module = trace_with_make_fx(
        lambda x, residual, norm_weight, gate_weight, up_weight: fused_rmsnorm_swiglu(
            x,
            residual,
            norm_weight,
            gate_weight,
            up_weight,
            eps=1e-6,
        ),
        *args,
        fake=True,
    )

    assert any(
        node.op == "call_function" and node.target == torch.ops.torchinferno.fused_rmsnorm_swiglu.default
        for node in graph_module.graph.nodes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_decode_linear_uses_transposed_weight_layout() -> None:
    torch.manual_seed(47)
    device = torch.device("cuda")
    x = torch.randn(1, 1, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randn(48, 32, device=device, dtype=torch.bfloat16)
    weight_t = weight.t().contiguous()

    actual = _decode_linear(x, weight, weight_t)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected)


def test_tensor_parallel_kv_cache_row_views_support_mixed_lengths() -> None:
    layer = Llama3TensorParallelLayerKVCache(
        3,
        8,
        1,
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    row0_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    row0_values = row0_keys + 100
    row1_keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + 10
    row1_values = row1_keys + 100

    layer.for_rows((0,)).append(row0_keys, row0_values)
    layer.for_rows((1,)).append(row1_keys, row1_values)

    assert layer.for_rows((0,)).seq_len == 3
    assert layer.for_rows((1,)).seq_len == 4
    assert layer.for_rows((2,)).seq_len == 0
    with pytest.raises(ValueError, match="selected cache rows must have the same sequence length"):
        _ = layer.for_rows((0, 1)).seq_len

    final_key = torch.tensor([[[[99.0, 100.0]]]])
    final_value = final_key + 100
    keys, values = layer.for_rows((0,)).append(final_key, final_value)

    assert layer.for_rows((0, 1)).seq_len == 4
    torch.testing.assert_close(keys[:, :, :3, :], row0_keys)
    torch.testing.assert_close(values[:, :, :3, :], row0_values)
    torch.testing.assert_close(layer.keys[0:1, :, :4, :], keys)
    torch.testing.assert_close(layer.values[0:1, :, :4, :], values)


def test_tensor_parallel_kv_cache_row_views_copy_prefix_and_clear() -> None:
    cache = Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                2,
                8,
                1,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]
    )
    keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    values = keys + 100

    cache.for_rows((0,)).layers[0].append(keys, values)
    cache.for_rows((1,)).copy_prefix_from(cache.for_rows((0,)), 2)

    assert cache.for_rows((0,)).seq_len == 3
    assert cache.for_rows((1,)).seq_len == 2
    torch.testing.assert_close(cache.layers[0].keys[1:2, :, :2, :], keys[:, :, :2, :])
    torch.testing.assert_close(cache.layers[0].values[1:2, :, :2, :], values[:, :, :2, :])

    cache.for_rows((1,)).clear_row(0)

    assert cache.for_rows((1,)).seq_len == 0


def test_tensor_parallel_kv_cache_row_views_compose() -> None:
    cache = Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                4,
                8,
                1,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]
    )
    view = cache.for_rows((2, 3))
    nested = view.for_rows((1,))
    keys = torch.tensor([[[[7.0, 8.0]]]])
    values = keys + 100

    nested.layers[0].append(keys, values)

    assert cache.for_rows((3,)).seq_len == 1
    assert cache.for_rows((2,)).seq_len == 0
    torch.testing.assert_close(cache.layers[0].keys[3:4, :, :1, :], keys)
    with pytest.raises(ValueError, match="cache row out of range"):
        view.for_rows((2,))


def test_tensor_parallel_kv_cache_row_views_reject_invalid_rows() -> None:
    cache = Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                2,
                8,
                1,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]
    )

    with pytest.raises(ValueError, match="cache row out of range"):
        cache.for_rows((-1,))
    with pytest.raises(ValueError, match="cache row out of range"):
        cache.for_rows((2,))
    with pytest.raises(ValueError, match="cache row out of range"):
        cache.copy_prefix_from(cache, 0, source_row=0, dest_row=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_decode_linear_uses_transposed_weight_layout_for_decode_batches() -> None:
    torch.manual_seed(48)
    device = torch.device("cuda")
    x = torch.randn(8, 1, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randn(48, 32, device=device, dtype=torch.bfloat16)
    weight_t = weight.t().contiguous()

    actual = _decode_linear(x, weight, weight_t)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tensor_parallel_decode_graph_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_DECODE_STEP", raising=False)
    device = torch.device("cuda")
    input_ids = torch.zeros((1, 1), device=device, dtype=torch.long)
    cache = Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                1,
                8,
                1,
                8,
                device=device,
                dtype=torch.bfloat16,
            )
        ]
    )

    assert _should_use_decode_step_graph(input_ids, cache, temperature=0.0)
    assert _should_use_decode_step_logits_graph(input_ids, cache)

    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_DECODE_STEP", "0")

    assert not _should_use_decode_step_graph(input_ids, cache, temperature=0.0)
    assert not _should_use_decode_step_logits_graph(input_ids, cache)


def test_tensor_parallel_decode_attention_blocks_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_DECODE_ATTENTION_BLOCKS", raising=False)

    assert _decode_attention_block_size(33, 288) == 64
    assert _decode_attention_block_size(1, 288) == 1

    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_DECODE_ATTENTION_BLOCKS", "0")

    assert _decode_attention_block_size(33, 288) == 288


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton unavailable")
def test_triton_fused_rmsnorm_swiglu_matches_reference() -> None:
    torch.manual_seed(46)
    device = torch.device("cuda")
    x = torch.randn(5, 32, device=device)
    residual = torch.randn(5, 32, device=device)
    norm_weight = torch.randn(32, device=device)
    gate_weight = torch.randn(32, device=device)
    up_weight = torch.randn(32, device=device)

    actual = fused_rmsnorm_swiglu(
        x,
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        eps=1e-6,
        config=KernelConfig(backend=KernelBackend.TRITON),
    )
    expected = fused_rmsnorm_swiglu_reference(
        x,
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        eps=1e-6,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


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


def test_helion_candidate_reports_unavailable_on_cpu() -> None:
    report = run_helion_candidate(HelionCandidateConfig(device="cpu", batch_size=2, tokens=2, hidden_size=8))

    assert report.status == "unavailable"
    assert not report.promoted
    assert "CUDA" in report.reason


def test_helion_fx_search_finds_and_remembers_swiglu_window(tmp_path) -> None:
    search = run_helion_fx_search(
        HelionCandidateConfig(device="cpu", batch_size=2, tokens=2, hidden_size=8),
        min_nodes=2,
        max_nodes=2,
    )
    supported = [window for window in search.windows if window.supported_kernel == "swiglu"]
    store = HelionDecisionStore(tmp_path / "helion-decisions.jsonl")
    store.append_many(search.reports)

    assert supported
    assert len(search.reports) == 1
    assert search.reports[0].candidate_id == supported[0].candidate_id
    assert store.read()[0]["status"] == "unavailable"


def test_helion_region_search_reports_local_and_macro_opportunities() -> None:
    mlp = run_helion_region_search(
        HelionRegionSearchConfig(model_kind="deepseek", region="mlp", batch_size=2, tokens=2),
        min_nodes=2,
        max_nodes=2,
    )
    attention = run_helion_region_search(
        HelionRegionSearchConfig(model_kind="deepseek", region="attention", batch_size=1, tokens=2),
        min_nodes=1,
        max_nodes=2,
    )

    assert any(window.supported_kernel == "swiglu" for window in mlp.windows)
    assert any(candidate.name == "mlp-swiglu-activation" for candidate in mlp.macro_candidates)
    assert any(candidate.name == "attention-rope-cache-macro" for candidate in attention.macro_candidates)


def test_helion_candidate_cli_reports_decision() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "helion-candidate",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--tokens",
            "2",
            "--hidden-size",
            "8",
            "--iters",
            "1",
            "--warmup",
            "1",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "TorchInferno Helion candidate" in cli.stdout
    assert "status=unavailable" in cli.stdout
