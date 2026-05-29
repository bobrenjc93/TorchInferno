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


def test_llama3_tensor_parallel_paged_cache_matches_dense_forward(tmp_path, monkeypatch) -> None:
    torch.manual_seed(9002)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model.device)

    with torch.inference_mode():
        dense_cache = model.allocate_cache(1, max_seq_len=8, cache_backend="dense")
        dense_logits, _ = model.forward(input_ids, cache=dense_cache, use_cache=True)

        paged_cache = model.allocate_cache(1, max_seq_len=8, cache_backend="paged", page_size=2)
        paged_logits, _ = model.forward(input_ids, cache=paged_cache, use_cache=True)

        decode_token = torch.tensor([[5]], dtype=torch.long, device=model.device)
        seq_lens = torch.tensor([input_ids.size(1)], dtype=torch.long, device=model.device)
        dense_decode_logits = model.decode_ragged_logits(decode_token, dense_cache, seq_lens=seq_lens)
        paged_decode_logits = model.decode_ragged_logits(decode_token, paged_cache, seq_lens=seq_lens)

    assert paged_cache.cache_backend == "paged"
    torch.testing.assert_close(paged_logits, dense_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(paged_decode_logits, dense_decode_logits, atol=2e-5, rtol=2e-5)


def test_llama3_tensor_parallel_ragged_prefill_matches_forward(tmp_path, monkeypatch) -> None:
    # Eager oracle for the row_indices ragged prefill (the continuous-batcher
    # graph body). Prefills a [batch, suffix_bucket] block into SCATTERED rows
    # with MIXED per-row prefix lengths (including a fresh start=0 row) and a
    # bucket-PADDED suffix, then asserts each row's next-token logits match a
    # plain forward() over that row's full prompt. Covers the three correctness
    # landmines: nonzero per-row rotary offset, pad-column KV not leaking, and
    # scattered-row gather.
    torch.manual_seed(9100)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=32)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device

    prompts = [
        [1, 5, 9, 13, 2],          # len 5, prefix 2 -> suffix 3
        [3, 7, 11, 4, 8, 6, 10],   # len 7, prefix 0 -> suffix 7 (fresh start)
        [2, 14, 1, 9],             # len 4, prefix 3 -> suffix 1
    ]
    rows = [3, 1, 2]               # scattered physical rows in a 4-row cache
    prefix_lens = [2, 0, 3]
    batch = len(prompts)

    with torch.inference_mode():
        # Reference: plain forward over each full prompt, last-token logits.
        ref_logits = []
        for prompt in prompts:
            ref_cache = model.allocate_cache(1, max_seq_len=32, cache_backend="dense")
            logits, _ = model.forward(
                torch.tensor([prompt], dtype=torch.long, device=device),
                cache=ref_cache,
                use_cache=True,
            )
            ref_logits.append(logits[0, -1, :])

        # Build one batch-4 dense cache; write each row's prefix via forward into
        # that row's view, then ragged-prefill all suffixes at once.
        cache = model.allocate_cache(4, max_seq_len=32, cache_backend="dense")
        for i, prompt in enumerate(prompts):
            if prefix_lens[i] > 0:
                view = cache.for_rows((rows[i],))
                model.forward(
                    torch.tensor([prompt[: prefix_lens[i]]], dtype=torch.long, device=device),
                    cache=view,
                    use_cache=True,
                )

        suffixes = [prompts[i][prefix_lens[i]:] for i in range(batch)]
        real_lens = [len(s) for s in suffixes]
        bucket = 8  # > every real suffix length, exercises pad columns
        padded = [s + [0] * (bucket - len(s)) for s in suffixes]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        seq_lens = torch.zeros(4, dtype=torch.long, device=device)
        for i in range(batch):
            seq_lens[rows[i]] = prefix_lens[i]
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        logit_positions = torch.tensor([rl - 1 for rl in real_lens], dtype=torch.long, device=device)

        out = model.prefill_ragged_logits(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
        )

    assert out.shape == (batch, 1, config.vocab_size)
    for i in range(batch):
        torch.testing.assert_close(out[i, 0, :], ref_logits[i], atol=2e-5, rtol=2e-5)


def test_llama3_tensor_parallel_ragged_prefill_folds_prefix_copy(tmp_path, monkeypatch) -> None:
    # The shared prefix is written to ONE prefix row; ragged prefill folds the
    # copy (src_prefix_row) into the forward -- broadcasting the prefix KV into
    # each active row before the suffix -- then attends via the flash context_len
    # path. Output must match a plain forward() over each full prompt.
    torch.manual_seed(9102)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=32)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device

    shared_prefix = [1, 5, 9, 13, 2, 6, 10]
    suffixes = [[3, 7], [11, 4, 8], [14]]
    prompts = [shared_prefix + s for s in suffixes]
    rows = [3, 1, 2]
    prefix_row = 0
    prefix_len = len(shared_prefix)
    bucket = 4
    batch = len(prompts)

    with torch.inference_mode():
        ref_logits = []
        for prompt in prompts:
            rc = model.allocate_cache(1, max_seq_len=32, cache_backend="dense")
            logits, _ = model.forward(torch.tensor([prompt], dtype=torch.long, device=device), cache=rc, use_cache=True)
            ref_logits.append(logits[0, -1, :])

        cache = model.allocate_cache(4, max_seq_len=32, cache_backend="dense")
        prefix_view = cache.for_rows((prefix_row,))
        model.forward(torch.tensor([shared_prefix], dtype=torch.long, device=device), cache=prefix_view, use_cache=True)

        padded = [s + [0] * (bucket - len(s)) for s in suffixes]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        seq_lens = torch.zeros(4, dtype=torch.long, device=device)
        for r in rows:
            seq_lens[r] = prefix_len
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        logit_positions = torch.tensor([len(s) - 1 for s in suffixes], dtype=torch.long, device=device)
        src_prefix_row = torch.tensor([prefix_row], dtype=torch.long, device=device)
        out = model.prefill_ragged_logits(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=prefix_len + bucket,
            src_prefix_row=src_prefix_row,
        )

    assert out.shape == (batch, 1, config.vocab_size)
    for i in range(batch):
        torch.testing.assert_close(out[i, 0, :], ref_logits[i], atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llama3_tensor_parallel_ragged_prefill_graph_replays_across_rows(tmp_path) -> None:
    # GPU capture/replay gate: capture the ragged-prefill graph on one scattered
    # row set, then REPLAY it on a DIFFERENT scattered row set with the same
    # (batch, suffix_bucket). Asserts (a) no 'Cannot copy CPU/CUDA during
    # capture', (b) exactly ONE captured graph after both calls (replay, not
    # recapture), and (c) replayed logits match the eager oracle for the new rows.
    torch.manual_seed(9101)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=64)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device
    assert device.type == "cuda"

    prompts = [[1, 5, 9, 13, 2, 6], [3, 7, 11, 4, 8, 6], [2, 14, 1, 9, 5, 7]]
    batch = len(prompts)
    bucket = 8
    padded = [p + [0] * (bucket - len(p)) for p in prompts]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    logit_positions = torch.tensor([len(p) - 1 for p in prompts], dtype=torch.long, device=device)
    seq_lens = torch.zeros(4, dtype=torch.long, device=device)  # fresh start=0 rows

    with torch.inference_mode():
        ref_logits = []
        for prompt in prompts:
            ref_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
            logits, _ = model.forward(
                torch.tensor([prompt], dtype=torch.long, device=device), cache=ref_cache, use_cache=True
            )
            ref_logits.append(logits[0, -1, :])

        cache = model.allocate_cache(4, max_seq_len=64, cache_backend="dense")

        # context_len = uniform prefix (0, fresh) + suffix bucket -> exercises the
        # flash causal_lower_right path used in serving (not the boolean mask).
        context_len = bucket

        def run(rows):
            row_indices = torch.tensor(rows, dtype=torch.long, device=device)
            return model.try_prefill_ragged_logits_graph(
                input_ids, cache, seq_lens=seq_lens, row_indices=row_indices,
                logit_positions=logit_positions, context_len=context_len, capture_on_miss=True,
            )

        out_a = run([3, 1, 2])   # captures
        assert out_a is not None, "ragged prefill graph returned None (capture failed)"
        graphs_after_a = len(model._ragged_prefill_logits_graphs)
        out_b = run([0, 2, 1])   # must REPLAY (same batch+bucket, different rows)
        graphs_after_b = len(model._ragged_prefill_logits_graphs)

    assert graphs_after_a == 1, f"expected 1 captured graph, got {graphs_after_a}"
    assert graphs_after_b == 1, f"replay recaptured a new graph: {graphs_after_b}"
    for i in range(batch):
        torch.testing.assert_close(out_a[i, 0, :], ref_logits[i], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out_b[i, 0, :], ref_logits[i], atol=1e-3, rtol=1e-3)


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
