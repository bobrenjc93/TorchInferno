import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from torchinferno.graph import trace_with_make_fx
from torchinferno.models.conversion import audit_native_deepseek_checkpoint, convert_native_deepseek_checkpoint
from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config


def test_native_deepseek_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(20)
    config = tiny_deepseek_v32_config(vocab_size=64, max_position_embeddings=16, use_score_correction_bias=True)
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    with torch.inference_mode():
        full_logits, _ = model(input_ids, use_cache=False)
        cache = model.allocate_cache(input_ids.size(0), max_seq_len=16)
        _, cache = model(input_ids[:, :-1], cache=cache, use_cache=True)
        step_logits, _ = model(input_ids[:, -1:], cache=cache, use_cache=True)

    torch.testing.assert_close(step_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-5)


def test_native_deepseek_generate_and_direct_q_projection() -> None:
    torch.manual_seed(21)
    config = tiny_deepseek_v32_config(
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=None,
        n_shared_experts=0,
    )
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.inference_mode():
        output = model.generate(input_ids, max_new_tokens=3)

    assert output.shape == (1, 6)
    assert torch.equal(output[:, :3], input_ids)


def test_native_deepseek_generate_restores_training_mode_when_allocation_fails(monkeypatch) -> None:
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).train()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    def fail_allocate_cache(*args, **kwargs):
        raise RuntimeError("cache allocation failed")

    monkeypatch.setattr(model, "allocate_cache", fail_allocate_cache)

    with pytest.raises(RuntimeError, match="cache allocation failed"):
        model.generate(input_ids, max_new_tokens=1)

    assert model.training


def test_native_deepseek_save_load_round_trip(tmp_path) -> None:
    torch.manual_seed(22)
    model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    model.save_pretrained(tmp_path)
    loaded = DeepSeekV32ForCausalLM.from_pretrained(tmp_path).eval()

    with torch.inference_mode():
        expected, _ = model(input_ids, use_cache=False)
        actual, _ = loaded(input_ids, use_cache=False)

    torch.testing.assert_close(actual, expected)


def test_native_deepseek_conversion_accepts_routed_expert_keys(tmp_path) -> None:
    torch.manual_seed(23)
    model = DeepSeekV32ForCausalLM(
        tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16, use_score_correction_bias=True)
    ).eval()
    source = tmp_path / "source"
    output = tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n")
    source_state = {}
    for key, tensor in model.state_dict().items():
        source_key = key.replace(".mlp.experts.", ".mlp.routed_experts.")
        source_state[source_key] = tensor.detach().clone()
    save_file(source_state, source / "model.safetensors", metadata={"format": "pt"})

    audit = audit_native_deepseek_checkpoint(source)
    report = convert_native_deepseek_checkpoint(source, output, max_shard_size="32KB")
    loaded = DeepSeekV32ForCausalLM.from_pretrained(output).eval()

    assert audit.compatible
    assert report.compatible
    assert (output / "torchinferno_conversion_report.json").exists()
    with torch.inference_mode():
        expected, _ = model(torch.tensor([[1, 2, 3]]), use_cache=False)
        actual, _ = loaded(torch.tensor([[1, 2, 3]]), use_cache=False)
    torch.testing.assert_close(actual, expected)


def test_native_deepseek_attention_traces_with_fake_tensors() -> None:
    torch.manual_seed(24)
    config = tiny_deepseek_v32_config(max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    attention = model.model.layers[0].self_attn
    hidden = torch.randn(1, 3, config.hidden_size)
    positions = torch.arange(3)

    def forward_attention(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return attention(x, pos, None)

    graph_module = trace_with_make_fx(forward_attention, hidden, positions, fake=True)

    assert sum(1 for _ in graph_module.graph.nodes) > 20


def test_native_deepseek_cli_smokes(tmp_path) -> None:
    torch.manual_seed(25)
    model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)).eval()
    model.save_pretrained(tmp_path)
    env = {**os.environ, "PYTHONPATH": "src"}

    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "deepseek-smoke",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--prompt-tokens",
            "3",
            "--new-tokens",
            "2",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    hf_smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "deepseek-hf-smoke",
            str(tmp_path),
            "--device",
            "cpu",
            "--input-ids",
            "1",
            "2",
            "3",
            "--new-tokens",
            "2",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "TorchInferno native DeepSeek smoke" in smoke.stdout
    assert "shape=(1, 5)" in smoke.stdout
    assert "TorchInferno native DeepSeek HF smoke" in hf_smoke.stdout
    assert "shape=(1, 5)" in hf_smoke.stdout
