from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from torchinferno.models.llama3 import (
    Llama3TensorParallelForCausalLM,
    Llama3V0ForCausalLM,
    tiny_llama3_config,
)
from torchinferno.models.llama3 import tensor_parallel as tensor_parallel_module


def test_llama3_tensor_parallel_greedy_sampler_uses_gather_by_default(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 2
    model.vocab_start = 0
    model.config = type("Config", (), {"vocab_size": 8})()
    monkeypatch.delenv("TORCHINFERNO_GREEDY_SAMPLE_GATHER", raising=False)

    def gather(logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape == (1, 2)
        return torch.tensor([5])

    def all_reduce(*args, **kwargs) -> None:
        raise AssertionError("default greedy sampling should use the gather path")

    monkeypatch.setattr(model, "_sample_next_token_greedy_gather", gather)
    monkeypatch.setattr(tensor_parallel_module.dist, "all_reduce", all_reduce)

    sampled = model._sample_next_token_greedy(torch.zeros(1, 2))

    assert sampled.tolist() == [5]


def test_llama3_tensor_parallel_greedy_sampler_all_reduce_opt_out(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 2
    model.vocab_start = 4
    model.config = type("Config", (), {"vocab_size": 8})()
    monkeypatch.setenv("TORCHINFERNO_GREEDY_SAMPLE_GATHER", "0")

    def gather(logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("greedy gather should respect the env opt-out")

    monkeypatch.setattr(model, "_sample_next_token_greedy_gather", gather)
    monkeypatch.setattr(tensor_parallel_module.dist, "all_reduce", lambda *args, **kwargs: None)

    sampled = model._sample_next_token_greedy(torch.tensor([[0.0, 3.0, 1.0]]))

    assert sampled.tolist() == [5]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_llama3_tensor_parallel_matches_reference_under_torchrun(tmp_path) -> None:
    torch.manual_seed(9001)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = reference(input_ids)
        expected_generated = reference.generate(input_ids, max_new_tokens=2)
    torch.save(input_ids, tmp_path / "input_ids.pt")
    torch.save(expected_logits, tmp_path / "expected_logits.pt")
    torch.save(expected_generated, tmp_path / "expected_generated.pt")

    script = tmp_path / "check_tp.py"
    script.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys

            import torch
            import torch.distributed as dist

            from torchinferno.models.llama3 import Llama3TensorParallelForCausalLM


            checkpoint, artifact_dir = sys.argv[1], sys.argv[2]
            model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
            input_ids = torch.load(f"{artifact_dir}/input_ids.pt", weights_only=True).to(model.device)
            expected_logits = torch.load(f"{artifact_dir}/expected_logits.pt", weights_only=True)
            expected_generated = torch.load(f"{artifact_dir}/expected_generated.pt", weights_only=True)

            with torch.inference_mode():
                logits, _ = model.forward(input_ids, use_cache=False)
                generated = model.generate(input_ids, max_new_tokens=2)
                local_sample_logits = torch.arange(
                    model.local_vocab_size,
                    device=model.device,
                    dtype=torch.float32,
                )[None, :].expand(3, model.local_vocab_size)
                sampled = model._sample_next_token(local_sample_logits, temperature=0.7)

            torch.testing.assert_close(logits.cpu(), expected_logits, atol=2e-5, rtol=2e-5)
            if not torch.equal(generated.cpu(), expected_generated):
                raise AssertionError(f"generated={generated.cpu().tolist()} expected={expected_generated.tolist()}")
            gathered_samples = [torch.empty_like(sampled) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sampled)
            for rank_samples in gathered_samples:
                if not torch.equal(sampled, rank_samples):
                    raise AssertionError(f"temperature samples diverged across ranks: {gathered_samples}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            """
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(script),
            str(checkpoint),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")},
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _write_hf_checkpoint(reference: Llama3V0ForCausalLM, config, checkpoint) -> None:
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

    save_file(hf_state, checkpoint / "model-00001-of-00001.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in hf_state.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_state},
            }
        )
        + "\n"
    )
    (checkpoint / "config.json").write_text(json.dumps(config.to_dict()) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
