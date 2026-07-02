import importlib
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

import torchinferno.models.deepseek as legacy_deepseek_mod
import torchinferno.models.deepseek_v32.model as canonical_deepseek_mod
import torchinferno.models.llama3.pipeline as llama3_pipeline
from torchinferno.models.deepseek_v32 import (
    DeepSeekV32V0ForCausalLM,
    DeepSeekV32V1ForCausalLM,
    tiny_deepseek_v32_v0_config,
)
from torchinferno.models.dsv4 import DSv4V0ForCausalLM, DSv4V1ForCausalLM, tiny_dsv4_v0_config
from torchinferno.models.llama3 import (
    Llama3PipelineForCausalLM,
    Llama3TensorParallelForCausalLM,
    Llama3V0ForCausalLM,
    Llama3V1ForCausalLM,
    raw_ops as llama3_raw_ops,
    llama3_70b_config,
    tiny_llama3_config,
)
from torchinferno.models.llama3.pipeline import (
    _CheckpointTensorLoader,
    _apply_rotary as _pipeline_apply_rotary,
)
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelLayerKVCache,
    _apply_rotary_cached as _tp_apply_rotary,
    _ragged_prefill_precision_graph_key,
)
from torchinferno.models.catalog import get_model_family, list_model_families
from torchinferno.models.variants import (
    get_model_variant,
    list_model_variants,
    model_variant_lineage,
)
from torchinferno.variant_validation import run_variant_logit_validation


def test_model_family_catalog_uses_canonical_packages() -> None:
    families = {family.name: family for family in list_model_families()}

    assert set(families) == {"dsv4", "deepseek-v3.2", "llama3"}
    assert families["dsv4"].package == "torchinferno.models.dsv4"
    assert families["deepseek-v3.2"].package == "torchinferno.models.deepseek_v32"
    assert families["llama3"].package == "torchinferno.models.llama3"
    assert families["llama3"].model_module == "torchinferno.models.llama3.model"
    assert get_model_family("deepseek_v32").name == "deepseek-v3.2"


def test_legacy_deepseek_import_aliases_canonical_module() -> None:
    assert legacy_deepseek_mod is canonical_deepseek_mod


def test_checkpoint_tensor_loader_returns_contiguous_column_shards(tmp_path) -> None:
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    save_file({"weight": weight}, tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": weight.numel() * weight.element_size()},
                "weight_map": {"weight": "model-00001-of-00001.safetensors"},
            }
        )
        + "\n"
    )

    loader = _CheckpointTensorLoader(tmp_path)
    shard = loader.get_tensor_shard(
        "weight",
        dim=1,
        rank=1,
        world_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert shard.is_contiguous()
    torch.testing.assert_close(shard, weight[:, 2:4].contiguous())


def test_checkpoint_tensor_loader_reuses_safetensor_handles(tmp_path, monkeypatch) -> None:
    from safetensors import safe_open as real_safe_open

    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    save_file({"weight": weight}, tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": weight.numel() * weight.element_size()},
                "weight_map": {"weight": "model-00001-of-00001.safetensors"},
            }
        )
        + "\n"
    )
    opened = []

    def counting_safe_open(*args, **kwargs):
        opened.append(args[0])
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(llama3_pipeline, "safe_open", counting_safe_open)

    with llama3_pipeline._CheckpointTensorLoader(tmp_path) as loader:
        assert loader.get_tensor_shape("weight") == (4, 6)
        torch.testing.assert_close(
            loader.get_tensor("weight", device=torch.device("cpu"), dtype=torch.float32),
            weight,
        )
        shard = loader.get_tensor_shard(
            "weight",
            dim=1,
            rank=1,
            world_size=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert len(opened) == 1
    torch.testing.assert_close(shard, weight[:, 2:4].contiguous())


def test_model_variant_registry_tracks_families_and_ops_modules() -> None:
    specs = list_model_variants()
    ids = {spec.id for spec in specs}

    assert {"dsv4:v0", "dsv4:v1", "deepseek-v3.2:v0", "deepseek-v3.2:v1", "llama3:v0", "llama3:v1"} <= ids
    assert [spec.variant for spec in model_variant_lineage("llama3", "v1")] == ["v0", "v1"]
    assert [spec.variant for spec in model_variant_lineage("dsv3.2", "v1")] == ["v0", "v1"]
    assert get_model_variant("dsv4", "v1").parents == ("v0",)
    for spec in specs:
        importlib.import_module(spec.ops_module)
        module = importlib.import_module(spec.module)
        assert hasattr(module, spec.class_name)


def test_v0_variants_are_make_fx_graph_backed() -> None:
    torch.manual_seed(49)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cases = (
        DSv4V0ForCausalLM(tiny_dsv4_v0_config(vocab_size=32, max_seq_len=16)).eval(),
        DeepSeekV32V0ForCausalLM(
            tiny_deepseek_v32_v0_config(vocab_size=32, max_position_embeddings=16)
        ).eval(),
        Llama3V0ForCausalLM(tiny_llama3_config(vocab_size=32, max_position_embeddings=16)).eval(),
    )

    for model in cases:
        graph = model.v0_graph(input_ids)
        readable = model.print_readable(input_ids, print_output=False)
        expected = model._traceable_forward(input_ids)
        if isinstance(model, Llama3V0ForCausalLM):
            actual = model(input_ids)
        else:
            actual, cache = model(input_ids, use_cache=False)
            assert cache is None

        assert isinstance(graph, torch.fx.GraphModule)
        assert model.v0_graph(input_ids) is graph
        assert "torch.ops.aten" in readable
        assert not any(key.startswith("_param_constant") for key in model.state_dict())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_llama3_tp_cache_partial_seq_len_updates_invalidate_uniform_state() -> None:
    cache = Llama3TensorParallelLayerKVCache(
        4,
        8,
        1,
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    cache._set_rows_seq_len((0, 1), 3)

    assert cache._seq_lens == [3, 3, 0, 0]
    assert cache._uniform_seq_len == [None]
    with pytest.raises(ValueError):
        _ = cache.seq_len

    cache._set_rows_seq_len((0, 1, 2, 3), 3)
    assert cache._seq_lens == [3, 3, 3, 3]
    assert cache._uniform_seq_len == [3]
    assert cache.seq_len == 3

    cache._set_rows_seq_len((0, 2), 3)
    assert cache._uniform_seq_len == [3]
    assert cache.seq_len_for_rows((1, 3)) == 3

    cache._set_rows_seq_len((0,), 4)
    assert cache._seq_lens == [4, 3, 3, 3]
    assert cache._uniform_seq_len == [None]
    with pytest.raises(ValueError):
        cache.seq_len_for_rows((0, 1))

    cache._set_rows_seq_len((0,), 3)
    assert cache._seq_lens == [3, 3, 3, 3]
    assert cache._uniform_seq_len == [None]
    assert cache.seq_len == 3

    cache._set_rows_seq_len((0, 1), 3)
    assert cache._uniform_seq_len == [3]
    assert cache.seq_len == 3


def test_llama3_v0_and_v1_are_weight_compatible() -> None:
    torch.manual_seed(50)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    v0 = Llama3V0ForCausalLM(config).eval()
    v1 = Llama3V1ForCausalLM(config).eval()
    v1.load_state_dict(v0.state_dict())
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    with torch.inference_mode():
        expected = v0(input_ids)
        actual = v1(input_ids)
        generated = v0.generate(input_ids[:1], max_new_tokens=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert generated.shape == (1, 5)


def test_llama3_70b_config_matches_public_architecture_shape() -> None:
    config = llama3_70b_config()

    assert config.hidden_size == 8192
    assert config.intermediate_size == 28672
    assert config.num_hidden_layers == 80
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.max_position_embeddings == 131072
    assert config.rope_scaling is not None
    assert config.rope_scaling["rope_type"] == "llama3"


def test_llama3_tensor_parallel_prefill_cache_only_matches_forward_cache(tmp_path) -> None:
    torch.manual_seed(54)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    _write_tiny_llama3_hf_checkpoint(reference, config, tmp_path)
    model = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    expected_cache = model.allocate_cache(1, 8)
    actual_cache = model.allocate_cache(1, 8)

    with torch.inference_mode():
        _, expected_cache = model.forward(
            input_ids,
            cache=expected_cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        returned_cache = model.prefill_cache_only(input_ids, actual_cache)

    assert returned_cache is actual_cache
    assert expected_cache.seq_len == actual_cache.seq_len == input_ids.size(1)
    for expected_layer, actual_layer in zip(expected_cache.layers, actual_cache.layers):
        assert expected_layer.seq_len == actual_layer.seq_len == input_ids.size(1)
        torch.testing.assert_close(
            actual_layer.keys[:, :, : input_ids.size(1), :],
            expected_layer.keys[:, :, : input_ids.size(1), :],
            atol=5e-5,
            rtol=5e-5,
        )
        torch.testing.assert_close(
            actual_layer.values[:, :, : input_ids.size(1), :],
            expected_layer.values[:, :, : input_ids.size(1), :],
            atol=5e-5,
            rtol=5e-5,
        )


def test_llama3_rotary_matches_huggingface_rotate_half_layout() -> None:
    torch.manual_seed(53)
    batch, heads, tokens, head_dim = 2, 3, 4, 8
    positions = torch.arange(tokens)
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    q = torch.randn(batch, heads, tokens, head_dim)
    k = torch.randn(batch, 1, tokens, head_dim)

    expected_q = _llama_rotate_half_reference(q, cos, sin)
    expected_k = _llama_rotate_half_reference(k, cos, sin)
    actual_raw = llama3_raw_ops.apply_rotary(q, cos, sin)
    actual_pipeline_q, actual_pipeline_k = _pipeline_apply_rotary(q, k, positions, inv_freq)
    actual_tp_q, actual_tp_k = _tp_apply_rotary(q, k, (cos, sin))

    torch.testing.assert_close(actual_raw, expected_q)
    torch.testing.assert_close(actual_pipeline_q, expected_q)
    torch.testing.assert_close(actual_pipeline_k, expected_k)
    torch.testing.assert_close(actual_tp_q, expected_q)
    torch.testing.assert_close(actual_tp_k, expected_k)


def test_llama3_pipeline_loads_hf_shaped_checkpoint_and_matches_v0(tmp_path) -> None:
    torch.manual_seed(52)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    state = reference.state_dict()
    hf_state = {
        "model.embed_tokens.weight": state["embed_tokens.weight"],
        "model.norm.weight": state["norm.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}."
        hf_prefix = f"model.layers.{layer_id}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            hf_state[hf_prefix + suffix] = state[prefix + suffix]

    save_file(hf_state, tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in hf_state.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_state},
            }
        )
        + "\n"
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()) + "\n")

    pipeline = Llama3PipelineForCausalLM.from_pretrained(tmp_path, devices=("cpu",), dtype="float32").eval()
    tensor_parallel = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.inference_mode():
        expected = reference(input_ids)
        actual, _ = pipeline.forward(input_ids, use_cache=False)
        tp_actual, _ = tensor_parallel.forward(input_ids, use_cache=False)
        tp_selected, _ = tensor_parallel.forward(
            input_ids,
            use_cache=False,
            logit_positions=torch.tensor([1], dtype=torch.long),
        )
        expected_generated = reference.generate(input_ids, max_new_tokens=2)
        actual_generated = pipeline.generate(input_ids, max_new_tokens=2)
        tp_generated = tensor_parallel.generate(input_ids, max_new_tokens=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tp_actual.cpu(), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tp_selected.cpu(), expected[:, 1:2, :], atol=1e-5, rtol=1e-5)
    assert torch.equal(actual_generated.cpu(), expected_generated)
    assert torch.equal(tp_generated.cpu(), expected_generated)


def test_llama3_tensor_parallel_ragged_decode_matches_independent_decode(tmp_path) -> None:
    torch.manual_seed(1234)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    _write_tiny_llama3_hf_checkpoint(reference, config, tmp_path)
    model = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    prompts = (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
    )
    decode_tokens = torch.tensor([[6], [7]], dtype=torch.long)
    combined_cache = model.allocate_cache(2, 8)
    expected_logits = []
    seq_lens = []

    with torch.inference_mode():
        for row, prompt in enumerate(prompts):
            row_cache = model.allocate_cache(1, 8)
            _, row_cache = model.forward(
                prompt,
                cache=row_cache,
                use_cache=True,
                return_last_logits_only=True,
                return_sharded_logits=True,
            )
            seq_len = prompt.size(1)
            seq_lens.append(seq_len)
            for source_layer, target_layer in zip(row_cache.layers, combined_cache.layers):
                target_layer.keys[row : row + 1, :, :seq_len, :].copy_(source_layer.keys[:, :, :seq_len, :])
                target_layer.values[row : row + 1, :, :seq_len, :].copy_(source_layer.values[:, :, :seq_len, :])
            logits, _ = model.forward(
                decode_tokens[row : row + 1],
                cache=row_cache,
                use_cache=True,
                return_last_logits_only=True,
                return_sharded_logits=True,
            )
            expected_logits.append(logits)
        for layer in combined_cache.layers:
            layer.seq_len = max(seq_lens)
        actual = model.decode_ragged_logits(
            decode_tokens,
            combined_cache,
            seq_lens=torch.tensor(seq_lens, dtype=torch.long),
        )

    torch.testing.assert_close(actual, torch.cat(expected_logits, dim=0), atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_llama3_tensor_parallel_ragged_decode_graph_matches_eager(tmp_path, monkeypatch) -> None:
    # This test verifies STRUCTURAL graph==eager decode equivalence in fp32; marlin
    # int4 (now default-on for decode) is orthogonal and would fail allclose (int4 !=
    # fp32 on this tiny model). Pin it off -- marlin's graph/eager + greedy correctness
    # is validated separately on the real 70B (decode-only M-gate, 4/4+5/5 greedy).
    monkeypatch.setenv("TORCHINFERNO_MARLIN_INT4_DECODE", "0")
    torch.manual_seed(1235)
    config = tiny_llama3_config(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=16,
    )
    reference = Llama3V0ForCausalLM(config).eval()
    _write_tiny_llama3_hf_checkpoint(reference, config, tmp_path)
    model = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    prompts = tuple(
        torch.tensor([[1, 2, *range(3, 3 + (row % 3))]], dtype=torch.long, device=model.device)
        for row in range(64)
    )
    decode_tokens = torch.tensor([[6 + (row % 16)] for row in range(64)], dtype=torch.long, device=model.device)
    seq_lens = torch.tensor([prompt.size(1) for prompt in prompts], dtype=torch.long, device=model.device)
    eager_cache = model.allocate_cache(64, 8)
    graph_cache = model.allocate_cache(64, 8)
    indexed_eager_cache = model.allocate_cache(64, 8)
    indexed_graph_cache = model.allocate_cache(64, 8)

    with torch.inference_mode():
        for row, prompt in enumerate(prompts):
            row_cache = model.allocate_cache(1, 8)
            _, row_cache = model.forward(
                prompt,
                cache=row_cache,
                use_cache=True,
                return_last_logits_only=True,
                return_sharded_logits=True,
            )
            seq_len = prompt.size(1)
            for target_cache in (eager_cache, graph_cache, indexed_eager_cache, indexed_graph_cache):
                for source_layer, target_layer in zip(row_cache.layers, target_cache.layers):
                    target_layer.keys[row : row + 1, :, :seq_len, :].copy_(source_layer.keys[:, :, :seq_len, :])
                    target_layer.values[row : row + 1, :, :seq_len, :].copy_(
                        source_layer.values[:, :, :seq_len, :]
                    )
        for cache in (eager_cache, graph_cache, indexed_eager_cache, indexed_graph_cache):
            for layer in cache.layers:
                layer.seq_len = int(seq_lens.max().item())

        expected = model.decode_ragged_logits(decode_tokens, eager_cache, seq_lens=seq_lens)
        actual = model.try_decode_ragged_logits_graph(decode_tokens, graph_cache, seq_lens=seq_lens)
        assert model._last_ragged_decode_logits_graph_captured is True
        row_indices = torch.tensor(
            [63, 1, 32, 0, 17, 9, 48, 2, 31, 4, 8, 16, 24, 40, 56, 7],
            dtype=torch.long,
            device=model.device,
        )
        expected_indexed = model.decode_ragged_logits(
            decode_tokens.index_select(0, row_indices),
            indexed_eager_cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
        )
        actual_indexed = model.try_decode_ragged_logits_graph(
            decode_tokens.index_select(0, row_indices),
            indexed_graph_cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
        )
        assert model._last_ragged_decode_logits_graph_captured is True

    assert actual is not None
    assert actual_indexed is not None
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_indexed, expected_indexed, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_llama3_tensor_parallel_ragged_decode_graph_replays_after_indexed_row_reuse(tmp_path, monkeypatch) -> None:
    # Pin marlin int4 off: this verifies fp32 structural graph==eager (int4 != fp32 on
    # the tiny model). See the sibling test above.
    monkeypatch.setenv("TORCHINFERNO_MARLIN_INT4_DECODE", "0")
    torch.manual_seed(1236)
    config = tiny_llama3_config(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=16,
    )
    reference = Llama3V0ForCausalLM(config).eval()
    _write_tiny_llama3_hf_checkpoint(reference, config, tmp_path)
    model = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    eager_cache = model.allocate_cache(64, 10)
    graph_cache = model.allocate_cache(64, 10)
    seq_lens = torch.zeros(64, dtype=torch.long, device=model.device)

    def load_prompt(caches, row: int, prompt: torch.Tensor) -> None:
        row_cache = model.allocate_cache(1, 10)
        _, row_cache = model.forward(
            prompt,
            cache=row_cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        for cache in caches:
            cache.copy_prefix_from(row_cache, prompt.size(1), dest_row=row)

    prompts = (
        torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device),
        torch.tensor([[4, 5, 6, 7]], dtype=torch.long, device=model.device),
        torch.tensor([[8, 9]], dtype=torch.long, device=model.device),
        torch.tensor([[10, 11, 12]], dtype=torch.long, device=model.device),
    )
    with torch.inference_mode():
        for row, prompt in enumerate(prompts):
            load_prompt((eager_cache, graph_cache), row, prompt)
            seq_lens[row] = prompt.size(1)
        for cache in (eager_cache, graph_cache):
            for layer in cache.layers:
                layer.seq_len = int(seq_lens.max().item())

        first_rows = torch.tensor([0, 2], dtype=torch.long, device=model.device)
        first_tokens = torch.tensor([[13], [14]], dtype=torch.long, device=model.device)
        expected_first = model.decode_ragged_logits(first_tokens, eager_cache, seq_lens=seq_lens, row_indices=first_rows)
        actual_first = model.try_decode_ragged_logits_graph(
            first_tokens,
            graph_cache,
            seq_lens=seq_lens,
            row_indices=first_rows,
        )
        assert actual_first is not None
        assert model._last_ragged_decode_logits_graph_captured is True
        # Graph replay returns a static output buffer; the second replay below reuses it.
        actual_first = actual_first.clone()
        seq_lens[first_rows] = seq_lens.index_select(0, first_rows) + 1

        replacement_prompt = torch.tensor([[15, 16, 17, 18]], dtype=torch.long, device=model.device)
        load_prompt((eager_cache, graph_cache), 2, replacement_prompt)
        seq_lens[2] = replacement_prompt.size(1)

        second_rows = torch.tensor([2, 1], dtype=torch.long, device=model.device)
        second_tokens = torch.tensor([[19], [20]], dtype=torch.long, device=model.device)
        expected_second = model.decode_ragged_logits(
            second_tokens,
            eager_cache,
            seq_lens=seq_lens,
            row_indices=second_rows,
        )
        actual_second = model.try_decode_ragged_logits_graph(
            second_tokens,
            graph_cache,
            seq_lens=seq_lens,
            row_indices=second_rows,
        )

    assert actual_second is not None
    assert model._last_ragged_decode_logits_graph_captured is False
    assert len(model._ragged_decode_logits_graphs) == 1
    torch.testing.assert_close(actual_first, expected_first, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_second, expected_second, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_llama3_tensor_parallel_prefill_logits_graph_replays_bucketed_real_last_logits(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_PREFILL", "1")
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_PREFILL_BUCKETING", "1")
    torch.manual_seed(1237)
    config = tiny_llama3_config(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=32,
    )
    reference = Llama3V0ForCausalLM(config).eval()
    _write_tiny_llama3_hf_checkpoint(reference, config, tmp_path)
    model = Llama3TensorParallelForCausalLM.from_pretrained(tmp_path, dtype="float32").eval()
    graph_cache = model.allocate_cache(1, 32)
    capture_prompt = torch.arange(1, 17, dtype=torch.long, device=model.device).unsqueeze(0)
    replay_prompt = torch.tensor([[5, 6, 7]], dtype=torch.long, device=model.device)

    with torch.inference_mode():
        captured = model.try_prefill_logits_graph(capture_prompt, graph_cache, capture_on_miss=True)
        assert captured is not None
        assert len(model._prefill_logits_graphs) == 1

        graph_cache.reset()
        actual = model.try_prefill_logits_graph(replay_prompt, graph_cache, capture_on_miss=True)
        assert actual is not None
        assert len(model._prefill_logits_graphs) == 1

        eager_cache = model.allocate_cache(1, 32)
        expected, _ = model.forward(
            replay_prompt,
            cache=eager_cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


class _RuntimeFp8PrefillLayer:
    def __init__(self, enabled: bool, min_m: int = 2048) -> None:
        self._runtime_fp8_prefill_enabled = enabled
        self._runtime_fp8_prefill_min_m = min_m


def test_ragged_prefill_precision_graph_key_tracks_runtime_fp8_policy(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_FP8_PREFILL", raising=False)
    monkeypatch.delenv("TORCHINFERNO_FP8_PREFILL_MIN_M", raising=False)
    layers = [_RuntimeFp8PrefillLayer(False), _RuntimeFp8PrefillLayer(False)]

    assert _ragged_prefill_precision_graph_key(1024, is_cuda=True, layers=layers) == (False,)

    for layer in layers:
        layer._runtime_fp8_prefill_enabled = True
        layer._runtime_fp8_prefill_min_m = 256
    assert _ragged_prefill_precision_graph_key(1024, is_cuda=True, layers=layers) == (True,)
    assert _ragged_prefill_precision_graph_key(128, is_cuda=True, layers=layers) == (False,)
    assert _ragged_prefill_precision_graph_key(1024, is_cuda=False, layers=layers) == (False,)

    layers[1]._runtime_fp8_prefill_enabled = False
    assert _ragged_prefill_precision_graph_key(1024, is_cuda=True, layers=layers) == (True, False)

    monkeypatch.setenv("TORCHINFERNO_FP8_PREFILL", "0")
    assert _ragged_prefill_precision_graph_key(1024, is_cuda=True, layers=layers) == (False,)


def test_ragged_prefill_precision_graph_key_tracks_global_fp8_env(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_FP8_PREFILL", "1")
    monkeypatch.delenv("TORCHINFERNO_FP8_PREFILL_MIN_M", raising=False)
    layers = [_RuntimeFp8PrefillLayer(False), _RuntimeFp8PrefillLayer(False)]

    assert _ragged_prefill_precision_graph_key(512, is_cuda=True, layers=layers) == (True,)

    monkeypatch.setenv("TORCHINFERNO_FP8_PREFILL_MIN_M", "1024")
    assert _ragged_prefill_precision_graph_key(512, is_cuda=True, layers=layers) == (False,)


def test_llama3_tensor_parallel_runtime_marlin_int4_decode_updates_layers() -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.layers = [type("Layer", (), {})(), type("Layer", (), {})()]

    model.set_runtime_marlin_int4_decode(False)

    assert [layer._runtime_marlin_int4_decode_enabled for layer in model.layers] == [False, False]

    model.set_runtime_marlin_int4_decode(True)

    assert [layer._runtime_marlin_int4_decode_enabled for layer in model.layers] == [True, True]


def test_llama3_tensor_parallel_temperature_sampling_uses_gumbel_max(monkeypatch) -> None:
    import torch.distributed as dist

    calls: list[object] = []

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        del tensor
        calls.append(op)

    def fail_collective(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("gumbel temperature sampling should not gather or broadcast")

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather_into_tensor", fail_collective)
    monkeypatch.setattr(dist, "broadcast", fail_collective)

    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.config = type("Config", (), {"vocab_size": 4})()
    model.rank = 0
    model.vocab_start = 0
    model.local_vocab_size = 4
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0, -1000.0]])

    sampled = model._sample_next_token(logits, temperature=0.7)

    assert sampled.tolist() == [0, 1]
    assert calls == [dist.ReduceOp.MAX, dist.ReduceOp.MIN]


def test_llama3_tensor_parallel_temperature_sampling_combines_broadcast(monkeypatch) -> None:
    import torch.distributed as dist

    calls: list[tuple[str, object]] = []

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        del tensor
        calls.append(("all_reduce", op))

    def all_gather_into_tensor(output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output[0].copy_(input_tensor)
        calls.append(("all_gather", tuple(output.shape)))

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        calls.append(("broadcast", tuple(tensor.shape)))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.world_size = 1
    model.rank = 0
    model.device = torch.device("cpu")
    model.vocab_start = 0
    model.local_vocab_size = 4
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0, -1000.0]])

    sampled = model._sample_next_token_temperature(logits, temperature=0.7)

    assert sampled.tolist() == [0, 1]
    assert calls == [
        ("all_reduce", dist.ReduceOp.MAX),
        ("all_gather", (1, 2)),
        ("broadcast", (2, 2)),
        ("all_reduce", dist.ReduceOp.SUM),
    ]


def test_llama3_tensor_parallel_repeated_temperature_sampling_combines_broadcast(monkeypatch) -> None:
    import torch.distributed as dist

    calls: list[tuple[str, object]] = []

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        del tensor
        calls.append(("all_reduce", op))

    def all_gather_into_tensor(output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output[0].copy_(input_tensor)
        calls.append(("all_gather", tuple(output.shape)))

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        calls.append(("broadcast", tuple(tensor.shape)))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.world_size = 1
    model.rank = 0
    model.device = torch.device("cpu")
    model.vocab_start = 0
    model.local_vocab_size = 4
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])

    sampled = model._sample_next_token_temperature_repeated(logits, batch_size=3, temperature=0.7)

    assert sampled.tolist() == [0, 0, 0]
    assert calls == [
        ("all_reduce", dist.ReduceOp.MAX),
        ("all_gather", (1, 1)),
        ("broadcast", (2, 3)),
        ("all_reduce", dist.ReduceOp.SUM),
    ]


def test_llama3_tensor_parallel_temperature_gumbel_generators_are_rank_local() -> None:
    first = object.__new__(Llama3TensorParallelForCausalLM)
    second = object.__new__(Llama3TensorParallelForCausalLM)
    first.rank = 0
    second.rank = 1

    first_generator = first._temperature_gumbel_generator(torch.device("cpu"))
    second_generator = second._temperature_gumbel_generator(torch.device("cpu"))
    first_values = torch.empty(8).exponential_(generator=first_generator)
    second_values = torch.empty(8).exponential_(generator=second_generator)

    assert not torch.equal(first_values, second_values)


def test_llama3_tensor_parallel_repeated_sample_state_samples_cached_cdf(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.delenv("TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_STATE_PREFETCH", raising=False)

    model = object.__new__(Llama3TensorParallelForCausalLM)
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])

    state = model.prepare_repeated_next_token_state(logits, temperature=0.7)
    sampled = model.sample_repeated_next_token_from_state(state, batch_size=3, temperature=0.7)

    assert sampled is not None
    assert sampled.tolist() == [0, 0, 0]
    assert model.sample_repeated_next_token_from_state(state, batch_size=3, temperature=0.8) is None


def test_llama3_tensor_parallel_repeated_sample_state_prefetches_tokens(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setenv("TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_STATE_PREFETCH", "5")

    model = object.__new__(Llama3TensorParallelForCausalLM)
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])
    state = model.prepare_repeated_next_token_state(logits, temperature=0.7)

    first = model.sample_repeated_next_token_from_state(state, batch_size=2, temperature=0.7)
    assert first is not None
    assert first.tolist() == [0, 0]
    assert state.cached_tokens is not None
    assert int(state.cached_offset) == 2

    second = model.sample_repeated_next_token_from_state(state, batch_size=3, temperature=0.7)
    assert second is not None
    assert second.tolist() == [0, 0, 0]
    assert int(state.cached_offset) == 5


def _write_tiny_llama3_hf_checkpoint(reference: Llama3V0ForCausalLM, config, path) -> None:
    state = reference.state_dict()
    hf_state = {
        "model.embed_tokens.weight": state["embed_tokens.weight"],
        "model.norm.weight": state["norm.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}."
        hf_prefix = f"model.layers.{layer_id}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            hf_state[hf_prefix + suffix] = state[prefix + suffix]

    save_file(hf_state, path / "model-00001-of-00001.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in hf_state.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_state},
            }
        )
        + "\n"
    )
    (path / "config.json").write_text(json.dumps(config.to_dict()) + "\n")


def _llama_rotate_half_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    cos = cos.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    sin = sin.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    return (x * cos) + (rotated * sin)


def test_dsv4_and_deepseek_v0_match_v1_greedy_generation() -> None:
    torch.manual_seed(51)
    dsv4_config = tiny_dsv4_v0_config(vocab_size=32, max_seq_len=16)
    dsv4_v0 = DSv4V0ForCausalLM(dsv4_config).eval()
    dsv4_v1 = DSv4V1ForCausalLM(dsv4_config).eval()
    dsv4_v1.load_state_dict(dsv4_v0.state_dict())
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    deepseek_config = tiny_deepseek_v32_v0_config(vocab_size=32, max_position_embeddings=16)
    deepseek_v0 = DeepSeekV32V0ForCausalLM(deepseek_config).eval()
    deepseek_v1 = DeepSeekV32V1ForCausalLM(deepseek_config).eval()
    deepseek_v1.load_state_dict(deepseek_v0.state_dict())

    with torch.inference_mode():
        assert torch.equal(dsv4_v0.generate(input_ids, max_new_tokens=2), dsv4_v1.generate(input_ids, max_new_tokens=2))
        assert torch.equal(
            deepseek_v0.generate(input_ids, max_new_tokens=2),
            deepseek_v1.generate(input_ids, max_new_tokens=2),
        )


def test_variant_logit_validation_harness_covers_eager_optimized_pairs() -> None:
    report = run_variant_logit_validation(
        device="cpu",
        batch_size=2,
        tokens=3,
        vocab_size=32,
        seed=54,
    )
    compared = {(comparison.family, comparison.optimized_variant) for comparison in report.comparisons}
    skipped = {(skipped.family, skipped.optimized_variant) for skipped in report.skipped}

    assert report.passed
    assert {
        ("dsv4", "v1"),
        ("deepseek-v3.2", "v1"),
        ("llama3", "v1"),
        ("llama3", "pipeline-v0"),
    } <= compared
    assert ("llama3", "tp-v0") in skipped
    for comparison in report.comparisons:
        assert comparison.max_abs_error <= 1e-4
        assert comparison.compared_logits == 2 * 3 * 32


def test_model_variants_cli_lists_lineage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "model-variants",
            "--family",
            "llama3",
            "--lineage",
            "v1",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "llama3:v0" in result.stdout
    assert "llama3:v1" in result.stdout
    assert "ops=torchinferno.models.llama3.fused_ops" in result.stdout


def test_validate_model_variants_cli_compares_eager_logits() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "validate-model-variants",
            "--family",
            "llama3",
            "--variant",
            "v1",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--tokens",
            "3",
            "--vocab-size",
            "32",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "TorchInferno variant logit validation" in result.stdout
    assert "passed=True" in result.stdout
    assert "llama3:v1 vs v0" in result.stdout
