from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torchinferno.models.deepseek_v4.tensor_parallel as v4_tp
import torchinferno.runtime.disaggregated as disaggregated_runtime

from torchinferno.models.deepseek_v4 import ops as v4_ops
from torchinferno.models.deepseek_v4.tensor_parallel import (
    DeepSeekV4TensorParallelForCausalLM,
    ModelArgs,
)
from torchinferno.runtime.disaggregated import (
    DisaggregatedPrefillDecodeModel,
    _pack_live_cache,
    _pack_tensor_views,
    _unpack_live_cache,
    _unpack_tensor_views,
)


def test_disaggregated_logits_transport_uses_model_output_dtype(monkeypatch) -> None:
    model = object.__new__(DisaggregatedPrefillDecodeModel)
    model.config = SimpleNamespace(vocab_size=7)
    model.device = torch.device("cpu")
    model.logits_dtype = torch.float32
    model.topology = SimpleNamespace(
        global_rank=0,
        decode_root=1,
        coordinator=True,
        transfer_group=object(),
    )
    received_dtypes = []

    def receive(tensor, peer, group):
        del peer, group
        received_dtypes.append(tensor.dtype)
        tensor.fill_(3)

    monkeypatch.setattr(disaggregated_runtime, "_receive_tensor", receive)
    logits = model._return_decode_logits(None, batch_size=2, output_tokens=1)

    assert received_dtypes == [torch.float32]
    assert logits.dtype == torch.float32


def test_disaggregated_provenance_uses_wrapped_model_family(monkeypatch) -> None:
    monkeypatch.setattr(
        DisaggregatedPrefillDecodeModel,
        "_validate_replicas",
        lambda self: None,
    )
    role_model = SimpleNamespace(
        provenance_variant="deepseek-v4:tp-v0",
        config=SimpleNamespace(),
        dtype=torch.bfloat16,
    )
    topology = SimpleNamespace(
        device=torch.device("cpu"),
        global_rank=0,
        world_size=2,
        role="prefill",
        role_rank=0,
    )

    model = DisaggregatedPrefillDecodeModel(role_model, topology)

    assert model.provenance_variant == "deepseek-v4:tp-disaggregated-v1"


def _tiny_v4_tp_args(*, max_batch_size: int = 4) -> ModelArgs:
    return ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=16,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=32,
        dim=32,
        moe_inter_dim=16,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=2,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=16,
        head_dim=16,
        rope_head_dim=4,
        o_groups=1,
        o_lora_rank=8,
        window_size=8,
        compress_ratios=(0,),
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
    )


def test_deepseek_v4_tp_cache_allocator_maps_local_and_physical_rows() -> None:
    model = DeepSeekV4TensorParallelForCausalLM(_tiny_v4_tp_args())

    first = model.allocate_cache(2, 16)
    second = model.allocate_cache(2, 16)

    assert first.selected_rows == (0, 1)
    assert second.selected_rows == (2, 3)
    first.release()
    second.release()
    assert model._free_cache_rows == {0, 1, 2, 3}


def test_deepseek_v4_graph_release_does_not_release_owned_cache_rows() -> None:
    model = DeepSeekV4TensorParallelForCausalLM(_tiny_v4_tp_args(max_batch_size=2))
    cache = model.allocate_cache(1, 16)

    model.release_decode_graphs_for_cache(cache)

    assert len(model._free_cache_rows) == 1
    assert cache.seq_len == 0
    model.release_cache(cache)
    assert len(model._free_cache_rows) == 2


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_disaggregated_v4_role_cache_uses_active_batch_not_handle_capacity(role) -> None:
    role_model = DeepSeekV4TensorParallelForCausalLM(_tiny_v4_tp_args())
    model = object.__new__(DisaggregatedPrefillDecodeModel)
    model.role_model = role_model
    model.role = role
    state = disaggregated_runtime._LocalCacheState(
        batch_size=4,
        max_seq_len=16,
        cache_backend="v4-heterogeneous",
        page_size=16,
    )

    cache = model._ensure_role_cache(state, input_tokens=3, batch_size=2)

    assert cache.selected_rows == (0, 1)
    assert state.active_batch_size == 2

    state.active_batch_size = None
    replacement = model._ensure_role_cache(state, input_tokens=3, batch_size=1)

    assert len(replacement.selected_rows) == 1
    assert len(role_model._free_cache_rows) == 3


def test_deepseek_v4_tp_continuation_chunks_are_processed_incrementally(
    monkeypatch,
) -> None:
    model = DeepSeekV4TensorParallelForCausalLM(_tiny_v4_tp_args(max_batch_size=1))
    cache = model.allocate_cache(1, 16)
    cache._seq_lens[cache.selected_rows[0]] = 3
    calls = []

    def forward_logits(
        input_ids,
        start_pos,
        row_indices,
        *,
        return_last_logits_only,
        return_sharded_logits,
    ):
        calls.append(
            (
                input_ids.clone(),
                start_pos,
                row_indices.clone(),
                return_last_logits_only,
                return_sharded_logits,
            )
        )
        return torch.full(
            (input_ids.size(0), 1, model.args.vocab_size),
            float(start_pos),
        )

    monkeypatch.setattr(model, "_forward_logits", forward_logits)
    logits, cache = model.forward(
        torch.tensor([[7, 8]]),
        cache=cache,
        return_last_logits_only=False,
        return_sharded_logits=True,
    )

    assert logits.shape == (1, 2, 32)
    assert logits[0, :, 0].tolist() == [3.0, 4.0]
    assert [call[1] for call in calls] == [3, 4]
    assert all(not call[3] and call[4] for call in calls)
    assert cache.seq_len == 5


def test_deepseek_v4_generate_uses_collective_sampler(monkeypatch) -> None:
    model = DeepSeekV4TensorParallelForCausalLM(_tiny_v4_tp_args(max_batch_size=1))
    cache = object()
    sampled = []

    monkeypatch.setattr(model, "allocate_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(model, "release_cache", lambda candidate: None)

    def forward(input_ids, **kwargs):
        del kwargs
        return torch.zeros((1, 1, 32)), cache

    def sample(logits, temperature):
        sampled.append((logits.shape, temperature))
        return torch.tensor([5])

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(model, "_sample_next_token", sample)

    output = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, temperature=0.7)

    assert output.tolist() == [[1, 2, 5, 5]]
    assert sampled == [((1, 32), 0.7), ((1, 32), 0.7)]


def test_disaggregated_cache_transfer_copies_only_live_kv_region() -> None:
    source_layers = []
    target_layers = []
    for layer_id in range(3):
        keys = torch.full((2, 2, 7, 4), -100.0)
        values = torch.full_like(keys, -200.0)
        keys[:, :, :3, :] = layer_id + 1
        values[:, :, :3, :] = layer_id + 11
        source_layers.append(SimpleNamespace(keys=keys, values=values))
        target_layers.append(
            SimpleNamespace(
                keys=torch.full_like(keys, 901.0),
                values=torch.full_like(values, 902.0),
            )
        )

    source = SimpleNamespace(layers=source_layers)
    target = SimpleNamespace(layers=target_layers)
    live_elements = 3 * 2 * 2 * 2 * 3 * 4
    buffer = torch.empty(live_elements)

    _pack_live_cache(source, buffer, batch_size=2, tokens=3)
    _unpack_live_cache(target, buffer, batch_size=2, tokens=3)

    for source_layer, target_layer in zip(source.layers, target.layers):
        torch.testing.assert_close(target_layer.keys[:, :, :3, :], source_layer.keys[:, :, :3, :])
        torch.testing.assert_close(target_layer.values[:, :, :3, :], source_layer.values[:, :, :3, :])
        assert torch.all(target_layer.keys[:, :, 3:, :] == 901.0)
        assert torch.all(target_layer.values[:, :, 3:, :] == 902.0)


def test_disaggregated_cache_transfer_rejects_contract_size_mismatch() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.zeros(1, 1, 4, 2),
                values=torch.zeros(1, 1, 4, 2),
            )
        ]
    )

    with pytest.raises(ValueError, match="transfer contract"):
        _pack_live_cache(cache, torch.empty(3), batch_size=1, tokens=2)


def test_disaggregated_cache_transfer_rejects_capacity_overflow() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.zeros(1, 1, 2, 2),
                values=torch.zeros(1, 1, 2, 2),
            )
        ]
    )

    with pytest.raises(ValueError, match="exceeds cache storage"):
        _pack_live_cache(cache, torch.empty(16), batch_size=1, tokens=4)


def test_deepseek_v4_heterogeneous_handoff_includes_partial_compressor_state() -> None:
    args = ModelArgs(
        max_batch_size=2,
        max_seq_len=140,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=32,
        dim=32,
        moe_inter_dim=16,
        n_layers=4,
        n_hash_layers=1,
        n_mtp_layers=0,
        n_heads=2,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=16,
        head_dim=16,
        rope_head_dim=4,
        o_groups=1,
        o_lora_rank=8,
        window_size=8,
        compress_ratios=(0, 4, 128, 0),
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
    )
    source_model = DeepSeekV4TensorParallelForCausalLM(args)
    target_model = DeepSeekV4TensorParallelForCausalLM(args)
    source_cache = source_model.allocate_cache(2, 140)
    target_cache = target_model.allocate_cache(2, 140)
    source_views = source_model.disaggregated_cache_tensors(
        source_cache, batch_size=2, tokens=129
    )
    target_views = target_model.disaggregated_cache_tensors(
        target_cache, batch_size=2, tokens=129
    )

    for index, view in enumerate(source_views, 1):
        view.fill_(index)
    for view in target_views:
        view.fill_(-1)
    for dtype in (torch.bfloat16, torch.float32):
        source_group = [view for view in source_views if view.dtype == dtype]
        target_group = [view for view in target_views if view.dtype == dtype]
        buffer = torch.empty(sum(view.numel() for view in source_group), dtype=dtype)
        _pack_tensor_views(source_group, buffer)
        _unpack_tensor_views(target_group, buffer)

    for source, target in zip(source_views, target_views):
        torch.testing.assert_close(target, source)
    target_model.finalize_disaggregated_cache_import(target_cache, tokens=129)
    assert target_cache.seq_len == 129


def test_deepseek_v4_heterogeneous_handoff_preserves_next_logits(monkeypatch) -> None:
    def act_quant(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: object = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
    ) -> torch.Tensor:
        del scale_fmt, scale_dtype
        quantized = v4_ops.fake_quant_fp8(x, block_size)
        if not inplace:
            raise AssertionError("unexpected non-inplace act_quant")
        x.copy_(quantized)
        return x

    def fp4_act_quant(
        x: torch.Tensor,
        block_size: int = 32,
        inplace: bool = False,
    ) -> torch.Tensor:
        quantized = v4_ops.fake_quant_fp4(x, block_size)
        if not inplace:
            raise AssertionError("unexpected non-inplace fp4_act_quant")
        x.copy_(quantized)
        return x

    def sparse_attn(
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = q.shape
        output = torch.empty_like(q)
        for batch in range(batch_size):
            for token in range(seq_len):
                indices = topk_idxs[batch, token]
                selected = kv[batch, indices[indices >= 0].long()]
                output[batch, token] = v4_ops.attention_with_sink(
                    q[batch, token].unsqueeze(0),
                    selected.unsqueeze(0),
                    sink,
                    scale,
                )[0]
        return output

    monkeypatch.setattr(v4_tp, "hc_split_sinkhorn", v4_ops.hc_split_sinkhorn)
    monkeypatch.setattr(v4_tp, "act_quant", act_quant)
    monkeypatch.setattr(v4_tp, "fp4_act_quant", fp4_act_quant)
    monkeypatch.setattr(v4_tp, "sparse_attn", sparse_attn)
    args = ModelArgs(
        max_batch_size=2,
        max_seq_len=140,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=32,
        dim=32,
        moe_inter_dim=16,
        n_layers=4,
        n_hash_layers=1,
        n_mtp_layers=0,
        n_heads=2,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=16,
        head_dim=68,
        rope_head_dim=4,
        o_groups=1,
        o_lora_rank=8,
        window_size=8,
        compress_ratios=(0, 4, 128, 0),
        index_n_heads=2,
        index_head_dim=64,
        index_topk=8,
    )
    torch.manual_seed(123)
    with v4_tp.set_dtype(torch.bfloat16):
        source = DeepSeekV4TensorParallelForCausalLM(args).eval()
        target = DeepSeekV4TensorParallelForCausalLM(args).eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if parameter.is_floating_point():
                parameter.normal_(0, 0.02)
            elif name.endswith("tid2eid"):
                parameter.copy_(
                    torch.randint(
                        0,
                        args.n_routed_experts,
                        parameter.shape,
                        dtype=parameter.dtype,
                    )
                )
    target.load_state_dict(source.state_dict())

    for tokens in (1, 3, 4, 5, 7, 8, 9, 127, 128, 129):
        source_cache = source.allocate_cache(2, 140)
        target_cache = target.allocate_cache(2, 140)
        input_ids = torch.arange(tokens * 2).reshape(2, tokens).remainder(
            args.vocab_size
        )
        with torch.inference_mode():
            source(input_ids, cache=source_cache)
        source_views = source.disaggregated_cache_tensors(
            source_cache,
            batch_size=2,
            tokens=tokens,
        )
        target_views = target.disaggregated_cache_tensors(
            target_cache,
            batch_size=2,
            tokens=tokens,
        )
        for dtype in sorted({view.dtype for view in source_views}, key=str):
            source_group = [view for view in source_views if view.dtype == dtype]
            target_group = [view for view in target_views if view.dtype == dtype]
            buffer = torch.empty(
                sum(view.numel() for view in source_group),
                dtype=dtype,
            )
            _pack_tensor_views(source_group, buffer)
            _unpack_tensor_views(target_group, buffer)
        target.finalize_disaggregated_cache_import(target_cache, tokens=tokens)
        next_ids = torch.tensor([[3], [17]])
        with torch.inference_mode():
            expected, _ = source(next_ids, cache=source_cache)
            actual, _ = target(next_ids, cache=target_cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        source.release_cache(source_cache)
        target.release_cache(target_cache)
