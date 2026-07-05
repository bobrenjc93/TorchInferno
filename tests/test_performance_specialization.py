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
    batched_paged_decode_attention,
    nvfp4_linear_reference,
    paged_decode_attention,
    quantize_nvfp4,
)
from torchinferno.kernels.ops import triton_available
from torchinferno.kernels.passes import register_kernel_replacement_passes
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelCache,
    Llama3TensorParallelLayerKVCache,
    PagedLlama3TensorParallelLayerKVCache,
    _decode_attention_block_size,
    _decode_linear,
    _prefill_graph_cache_storage,
    _prepare_paged_ragged_decode_graph_state,
    _ragged_decode_cache_token_bucket,
    _should_use_decode_step_graph,
    _should_use_decode_step_logits_graph,
    _static_decode_cache_rows_are_contiguous,
    _static_decode_row_indices,
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
from torchinferno.runtime.paged_attention import (
    batched_paged_causal_attention,
    paged_causal_attention,
)
from torchinferno.models.llama3.tensor_parallel import (
    _ragged_scaled_dot_product_attention,
)


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


def test_paged_kv_cache_ragged_append_reuses_static_bucket_padding_rows() -> None:
    layer = PagedLlama3TensorParallelLayerKVCache(
        2,
        8,
        1,
        2,
        page_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    base_keys = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    layer.append(base_keys, base_keys + 100)

    first_query = torch.ones((2, 1, 1, 2))
    first_keys = torch.full((2, 1, 1, 2), 10.0)
    layer.append_and_attend_ragged(
        first_query,
        first_keys,
        first_keys + 100,
        torch.tensor([2, 2]),
        torch.tensor([0, 1]),
        enable_gqa=False,
    )

    second_query = torch.ones((2, 1, 1, 2))
    second_keys = torch.full((2, 1, 1, 2), 20.0)
    layer.append_and_attend_ragged(
        second_query,
        second_keys,
        second_keys + 100,
        torch.tensor([3, 2]),
        torch.tensor([0, 1]),
        enable_gqa=False,
    )

    assert layer.for_rows((0,)).seq_len == 4
    assert layer.for_rows((1,)).seq_len == 3
    row1_keys, _row1_values = layer.materialize_row(1)
    torch.testing.assert_close(row1_keys[:, -1:, :], first_keys[1, :, :, :])


def test_paged_kv_cache_seq_len_restore_keeps_graph_pages() -> None:
    layer = PagedLlama3TensorParallelLayerKVCache(
        1,
        8,
        1,
        2,
        page_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    values = keys + 100
    layer.append(keys, values)
    original_pages = tuple(layer.pages.sequence("batch-0").page_ids)

    layer.set_seq_len(2)
    layer.set_seq_len(4)

    assert tuple(layer.pages.sequence("batch-0").page_ids) == original_pages
    assert layer.seq_len == 4
    storage = _prefill_graph_cache_storage(Llama3TensorParallelCache([layer], cache_backend="paged"))
    assert storage is not None
    assert storage.data_ptr() == layer.pages.keys.data_ptr()


def test_paged_ragged_decode_graph_state_uses_cache_token_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKETS", "1")
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKET_VALUES", "4,8,16")
    dense_cache = Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                2,
                16,
                1,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]
    )
    cache = Llama3TensorParallelCache(
        [
            PagedLlama3TensorParallelLayerKVCache(
                2,
                16,
                1,
                2,
                page_size=2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(2)
        ],
        cache_backend="paged",
    )
    seq_lens = torch.tensor([2, 5], dtype=torch.long)

    assert _ragged_decode_cache_token_bucket(dense_cache, seq_lens, None, batch=2) == 8
    assert _ragged_decode_cache_token_bucket(cache, seq_lens, None, batch=2) == 8
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKETS", raising=False)
    setattr(dense_cache, "_torchinferno_ragged_decode_cache_token_limit", 6)
    assert _ragged_decode_cache_token_bucket(dense_cache, seq_lens, None, batch=2) == 6
    setattr(dense_cache, "_torchinferno_ragged_decode_cache_token_min_batch", 4)
    assert _ragged_decode_cache_token_bucket(dense_cache, seq_lens, None, batch=2) == 16
    assert _ragged_decode_cache_token_bucket(dense_cache, seq_lens, None, batch=4) == 6
    delattr(dense_cache, "_torchinferno_ragged_decode_cache_token_limit")
    delattr(dense_cache, "_torchinferno_ragged_decode_cache_token_min_batch")
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKETS", "1")
    page_tables, seq_lens_buffers = _prepare_paged_ragged_decode_graph_state(
        cache,
        batch=2,
        cache_positions=seq_lens,
        row_indices=None,
        device=torch.device("cpu"),
        cache_token_bucket=8,
    )
    assert len(page_tables) == 2
    assert len(seq_lens_buffers) == 2
    assert page_tables[0].shape == (2, 4)
    assert seq_lens_buffers[0].tolist() == [3, 6]
    assert cache.layers[0]._torchinferno_paged_decode_cache_tokens == 8
    first_page_table_ptr = page_tables[0].data_ptr()

    reused_page_tables, reused_seq_lens = _prepare_paged_ragged_decode_graph_state(
        cache,
        batch=2,
        cache_positions=seq_lens,
        row_indices=None,
        device=torch.device("cpu"),
        cache_token_bucket=8,
        page_tables=page_tables,
        seq_lens_buffers=seq_lens_buffers,
    )
    assert reused_page_tables[0].data_ptr() == first_page_table_ptr
    assert reused_seq_lens[0].data_ptr() == seq_lens_buffers[0].data_ptr()

    wider_page_tables, _wider_seq_lens = _prepare_paged_ragged_decode_graph_state(
        cache,
        batch=2,
        cache_positions=torch.tensor([9, 9], dtype=torch.long),
        row_indices=None,
        device=torch.device("cpu"),
        cache_token_bucket=16,
        page_tables=page_tables,
        seq_lens_buffers=seq_lens_buffers,
    )
    assert wider_page_tables[0].shape == (2, 8)
    assert wider_page_tables[0].data_ptr() != first_page_table_ptr
    assert cache.layers[0]._torchinferno_paged_decode_cache_tokens == 16


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


def test_tensor_parallel_kv_cache_static_storage_tracks_contiguous_row_views() -> None:
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
    layer = cache.layers[0]
    contiguous = layer.for_rows((1, 2))
    keys, values = contiguous._contiguous_key_value_storage(2)

    assert keys.shape == (2, 1, 8, 2)
    assert values.shape == (2, 1, 8, 2)
    keys[:, :, :1, :].fill_(3.0)
    torch.testing.assert_close(layer.keys[1:3, :, :1, :], torch.full((2, 1, 1, 2), 3.0))
    assert _static_decode_cache_rows_are_contiguous(cache.for_rows((1, 2)), 2)
    assert not _static_decode_cache_rows_are_contiguous(cache.for_rows((0, 2)), 2)
    assert layer.for_rows((0, 2))._contiguous_key_value_storage(2) is None
    sparse_indices = _static_decode_row_indices(cache.for_rows((0, 2)), 2)
    assert sparse_indices is not None
    torch.testing.assert_close(sparse_indices.cpu(), torch.tensor([0, 2]))


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


def test_paged_decode_attention_torch_backend_supports_grouped_query_attention() -> None:
    torch.manual_seed(43)
    kv_heads = 2
    query_heads = 4
    seq_len = 7
    head_dim = 5
    value_dim = 4
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.randn(kv_heads, seq_len, head_dim)
    values = torch.randn(kv_heads, seq_len, value_dim)
    query = torch.randn(query_heads, 1, head_dim)
    cache.append("req", keys, values)

    actual = paged_decode_attention(
        query,
        cache,
        "req",
        seq_len - 1,
        config=KernelConfig(backend=KernelBackend.TORCH),
        enable_gqa=True,
    )
    expected = paged_causal_attention(
        query,
        cache,
        "req",
        torch.tensor([seq_len - 1]),
        enable_gqa=True,
    )

    torch.testing.assert_close(actual, expected)


def test_batched_paged_decode_attention_torch_backend_matches_reference() -> None:
    torch.manual_seed(44)
    kv_heads = 2
    query_heads = 4
    head_dim = 5
    value_dim = 4
    lengths = [3, 6, 7]
    cache = PagedKVCache(
        num_pages=12,
        page_size=2,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    for row, length in enumerate(lengths):
        cache.append(
            f"req-{row}",
            torch.randn(kv_heads, length, head_dim),
            torch.randn(kv_heads, length, value_dim),
        )
    query = torch.randn(len(lengths), query_heads, 1, head_dim)
    request_ids = [f"req-{row}" for row in range(len(lengths))]
    positions = torch.tensor([length - 1 for length in lengths])

    actual = batched_paged_decode_attention(
        query,
        cache,
        request_ids,
        positions,
        config=KernelConfig(backend=KernelBackend.TORCH),
        enable_gqa=True,
    )
    expected = torch.stack(
        [
            paged_causal_attention(
                query[row],
                cache,
                request_id,
                torch.tensor([int(positions[row].item())]),
                enable_gqa=True,
            )
            for row, request_id in enumerate(request_ids)
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def _poisoned_ragged_cache(kv_heads, head_dim, value_dim, lengths):
    """A paged cache whose UNWRITTEN slots are NaN (simulating uninitialized
    torch.empty memory), with valid ragged KV appended for each request."""
    cache = PagedKVCache(
        num_pages=64,
        page_size=2,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache.keys.fill_(float("nan"))
    cache.values.fill_(float("nan"))
    for row, length in enumerate(lengths):
        cache.append(
            f"req-{row}",
            torch.randn(kv_heads, length, head_dim),
            torch.randn(kv_heads, length, value_dim),
        )
    return cache


def test_batched_paged_decode_attention_no_nan_from_unwritten_padding() -> None:
    # Regression: masked padding KV gathered from unwritten (NaN) cache slots used
    # to poison the output via NaN*0 == NaN. Output must be finite.
    torch.manual_seed(1)
    lengths = [3, 6, 7]
    cache = _poisoned_ragged_cache(2, 5, 4, lengths)
    query = torch.randn(len(lengths), 4, 1, 5)
    out = batched_paged_decode_attention(
        query,
        cache,
        [f"req-{r}" for r in range(len(lengths))],
        torch.tensor([length - 1 for length in lengths]),
        config=KernelConfig(backend=KernelBackend.TORCH),
        enable_gqa=True,
    )
    assert torch.isfinite(out).all()


def test_batched_paged_causal_attention_no_nan_from_unwritten_padding() -> None:
    # Regression: ragged length_mask over unwritten (NaN) padding poisoned the
    # output via NaN*0. Output must be finite for mixed-length requests.
    torch.manual_seed(2)
    lengths = [3, 6, 8]
    cache = _poisoned_ragged_cache(2, 5, 4, lengths)
    query = torch.randn(len(lengths), 4, 1, 5)
    out = batched_paged_causal_attention(
        query,
        cache,
        [f"req-{r}" for r in range(len(lengths))],
        torch.tensor([[length - 1] for length in lengths]),
        enable_gqa=True,
    )
    assert torch.isfinite(out).all()


def test_ragged_scaled_dot_product_attention_no_nan_from_unwritten_padding() -> None:
    # Regression: cache rows beyond attention_lengths are unwritten (NaN); SDPA
    # masks them but NaN*0 poisoned the output. Output must be finite.
    torch.manual_seed(3)
    batch, q_heads, kv_heads, head_dim, max_seq = 3, 4, 2, 8, 10
    lengths = torch.tensor([3, 6, 9])
    q = torch.randn(batch, q_heads, 1, head_dim)
    k = torch.randn(batch, kv_heads, max_seq, head_dim)
    v = torch.randn(batch, kv_heads, max_seq, head_dim)
    for row, length in enumerate(lengths.tolist()):
        k[row, :, length:, :] = float("nan")
        v[row, :, length:, :] = float("nan")
    out = _ragged_scaled_dot_product_attention(q, k, v, lengths, enable_gqa=True)
    assert torch.isfinite(out).all()


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


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton unavailable")
def test_triton_batched_paged_decode_attention_matches_reference() -> None:
    torch.manual_seed(45)
    device = torch.device("cuda")
    kv_heads = 2
    query_heads = 4
    head_dim = 16
    value_dim = 12
    lengths = [5, 9, 13]
    cache = PagedKVCache(
        num_pages=12,
        page_size=4,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=device,
        dtype=torch.float32,
    )
    for row, length in enumerate(lengths):
        cache.append(
            f"req-{row}",
            torch.randn(kv_heads, length, head_dim, device=device),
            torch.randn(kv_heads, length, value_dim, device=device),
        )
    query = torch.randn(len(lengths), query_heads, 1, head_dim, device=device)
    request_ids = [f"req-{row}" for row in range(len(lengths))]
    positions = torch.tensor([length - 1 for length in lengths], device=device)

    actual = batched_paged_decode_attention(
        query,
        cache,
        request_ids,
        positions,
        config=KernelConfig(backend=KernelBackend.TRITON),
        enable_gqa=True,
    )
    expected = torch.stack(
        [
            paged_causal_attention(
                query[row],
                cache,
                request_id,
                torch.tensor([int(positions[row].item())], device=device),
                enable_gqa=True,
            )
            for row, request_id in enumerate(request_ids)
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


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
