import os
import subprocess
import sys

import torch

from torchinferno.models.dsv4 import DSv4Config, DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.runtime.offload import run_offloaded_generate_recompute, summarize_offload_events
from torchinferno.runtime.simulation import TimeSliceWorkload, TimeSlicedSimulator, VirtualGPU


def test_deepseek_v32_config_translation_uses_moe_expert_size() -> None:
    config = tiny_dsv4_config().to_dict()
    config.update(
        {
            "model_type": "deepseek_v32",
            "num_hidden_layers": 3,
            "kv_lora_rank": 16,
            "n_routed_experts": 6,
            "num_experts_per_tok": 2,
            "intermediate_size": 256,
            "moe_intermediate_size": 48,
        }
    )

    translated = DSv4Config.from_dict(config)

    assert translated.num_layers == 3
    assert translated.latent_kv_size == 16
    assert translated.num_experts == 6
    assert translated.top_k == 2
    assert translated.intermediate_size == 48


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(0)
    config = tiny_dsv4_config(vocab_size=64, max_seq_len=16)
    model = DSv4ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    with torch.inference_mode():
        full_logits, _ = model(input_ids, use_cache=False)
        cache = model.allocate_cache(input_ids.size(0), max_seq_len=16)
        _, cache = model(input_ids[:, :-1], cache=cache, use_cache=True)
        step_logits, _ = model(input_ids[:, -1:], cache=cache, use_cache=True)

    torch.testing.assert_close(step_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-5)


def test_generate_runs_end_to_end() -> None:
    torch.manual_seed(1)
    config = tiny_dsv4_config(vocab_size=32, max_seq_len=16)
    model = DSv4ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.inference_mode():
        output = model.generate(input_ids, max_new_tokens=4)

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], input_ids)


def test_save_and_load_pretrained_round_trip(tmp_path) -> None:
    torch.manual_seed(1)
    config = tiny_dsv4_config(vocab_size=32, max_seq_len=16)
    model = DSv4ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    model.save_pretrained(tmp_path)
    loaded = DSv4ForCausalLM.from_pretrained(tmp_path).eval()

    with torch.inference_mode():
        expected, _ = model(input_ids, use_cache=False)
        actual, _ = loaded(input_ids, use_cache=False)

    torch.testing.assert_close(actual, expected)


def test_time_sliced_simulator_runs_virtual_ranks() -> None:
    simulator = TimeSlicedSimulator([VirtualGPU(0, latency_us=10), VirtualGPU(1, latency_us=20)])
    events = simulator.run(lambda gpu: gpu.rank + 1)

    assert [event.rank for event in events] == [0, 1]
    assert [event.result for event in events] == [1, 2]
    assert all(event.elapsed_us > 0 for event in events)


def test_time_sliced_simulator_replays_profile_work_round_robin() -> None:
    simulator = TimeSlicedSimulator(
        [
            VirtualGPU(0, latency_us=1.0, time_slice_us=10.0),
            VirtualGPU(1, latency_us=1.0, time_slice_us=10.0),
        ]
    )

    replay = simulator.replay(
        [
            TimeSliceWorkload(rank=0, work_us=25.0, label="prefill"),
            TimeSliceWorkload(rank=1, work_us=10.0, label="decode"),
        ]
    )

    assert [event.rank for event in replay.events] == [0, 1, 0, 0]
    assert [event.work_us for event in replay.events] == [10.0, 10.0, 10.0, 5.0]
    assert replay.total_work_us == 35.0
    assert replay.total_overhead_us == 4.0
    assert 0 < replay.utilization <= 1


def test_cpu_offloaded_recompute_generate_matches_dsv4_generate() -> None:
    torch.manual_seed(4)
    model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=32, max_seq_len=16)).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.inference_mode():
        expected = model.generate(input_ids, max_new_tokens=2)
        actual = run_offloaded_generate_recompute(
            model,
            input_ids,
            max_new_tokens=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert torch.equal(actual.output, expected)
    summary = summarize_offload_events(actual.events)
    assert summary["event_count"] > 0
    assert summary["compute_ms"] > 0
    assert any(event.kind == "stage_to_device" for event in actual.events)


def test_cli_smoke_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "dsv4-smoke",
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
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "TorchInferno DSv4 smoke" in result.stdout
    assert "shape=(1, 5)" in result.stdout


def test_cli_hf_smoke_runs_from_local_checkpoint(tmp_path) -> None:
    torch.manual_seed(3)
    model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=32, max_seq_len=16)).eval()
    model.save_pretrained(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "dsv4-hf-smoke",
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
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "TorchInferno DSv4 HF smoke" in result.stdout
    assert "shape=(1, 5)" in result.stdout
