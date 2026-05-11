import importlib
import json
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file

import torchinferno.models.deepseek as legacy_deepseek_mod
import torchinferno.models.deepseek_v32.model as canonical_deepseek_mod
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
from torchinferno.models.llama3.pipeline import _apply_rotary as _pipeline_apply_rotary
from torchinferno.models.llama3.tensor_parallel import _apply_rotary_cached as _tp_apply_rotary
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
        expected_generated = reference.generate(input_ids, max_new_tokens=2)
        actual_generated = pipeline.generate(input_ids, max_new_tokens=2)
        tp_generated = tensor_parallel.generate(input_ids, max_new_tokens=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tp_actual.cpu(), expected, atol=1e-5, rtol=1e-5)
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
    model.vocab_start = 0
    model.local_vocab_size = 4
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0, -1000.0]])

    sampled = model._sample_next_token(logits, temperature=0.7)

    assert sampled.tolist() == [0, 1]
    assert calls == [dist.ReduceOp.MAX, dist.ReduceOp.MIN]


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
