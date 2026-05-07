import json
import os
import subprocess
import sys

from torchinferno.benchmarks import (
    VLLMBenchmarkConfig,
    build_vllm_benchmark_commands,
    collect_vllm_benchmark_summary,
    plot_vllm_benchmark_results,
)


def test_vllm_benchmark_commands_match_vllm_entrypoints(tmp_path) -> None:
    config = VLLMBenchmarkConfig(
        output_dir=tmp_path,
        model="/models/llama70b",
        vllm_root=None,
        python="python3",
        tensor_parallel_size=8,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        engine_args=("--max-num-seqs", "256"),
    )

    commands = build_vllm_benchmark_commands(config)
    by_name = {command.name: command for command in commands}

    assert set(by_name) == {"latency", "throughput", "serve"}
    assert by_name["latency"].command[:5] == ("python3", "-m", "vllm.entrypoints.cli.main", "bench", "latency")
    assert "--tensor-parallel-size" in by_name["latency"].command
    assert "8" in by_name["latency"].command
    assert "--output-json" in by_name["throughput"].command
    assert "--random-input-len" in by_name["throughput"].command
    assert "--random-output-len" in by_name["throughput"].command
    assert "--save-result" in by_name["serve"].command
    assert "--result-filename" in by_name["serve"].command
    assert by_name["serve"].output_json.name == "serve.json"


def test_vllm_benchmark_summary_and_plot(tmp_path) -> None:
    (tmp_path / "latency.json").write_text(
        json.dumps(
            {
                "avg_latency": 1.25,
                "percentiles": {"50": 1.0, "90": 1.4, "99": 1.8},
            }
        )
        + "\n"
    )
    (tmp_path / "throughput.json").write_text(
        json.dumps(
            {
                "elapsed_time": 4.0,
                "num_requests": 16,
                "total_num_tokens": 2560,
                "requests_per_second": 4.0,
                "tokens_per_second": 640.0,
            }
        )
        + "\n"
    )
    (tmp_path / "serve.json").write_text(
        json.dumps(
            {
                "duration": 5.0,
                "completed": 10,
                "failed": 0,
                "request_throughput": 2.0,
                "output_throughput": 256.0,
                "total_token_throughput": 320.0,
                "mean_ttft_ms": 12.0,
                "p99_ttft_ms": 20.0,
                "mean_tpot_ms": 4.0,
                "p99_tpot_ms": 7.0,
                "mean_itl_ms": 4.1,
                "p99_itl_ms": 7.2,
            }
        )
        + "\n"
    )

    summary = collect_vllm_benchmark_summary(tmp_path)
    assert summary["results"]["latency"]["p99_latency_s"] == 1.8
    assert summary["results"]["throughput"]["tokens_per_second"] == 640.0
    assert summary["results"]["serve"]["request_throughput"] == 2.0

    outputs = plot_vllm_benchmark_results(tmp_path)
    assert outputs["html"].exists()
    assert outputs["csv"].exists()
    assert "vLLM Benchmark Performance" in outputs["html"].read_text()
    assert "tokens_per_second" in outputs["csv"].read_text()


def test_vllm_benchmark_cli_writes_plan_without_running(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "vllm-bench-suite",
            str(tmp_path),
            "--model",
            "/models/llama70b",
            "--vllm-root",
            "/tmp/no-vllm-needed-for-plan",
            "--python",
            "python3",
            "--benchmarks",
            "latency",
            "--no-plot",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "mode=planned" in result.stdout
    commands = json.loads((tmp_path / "commands.json").read_text())
    assert commands[0]["name"] == "latency"
    assert commands[0]["command"][:5] == ["python3", "-m", "vllm.entrypoints.cli.main", "bench", "latency"]


def test_llama_benchmark_cli_writes_plan_without_running(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "llama-bench-suite",
            str(tmp_path),
            "--model",
            "/models/llama70b",
            "--devices",
            "cuda:0",
            "cuda:1",
            "--benchmarks",
            "latency",
            "--no-plot",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "mode=planned" in result.stdout
    config = json.loads((tmp_path / "suite_config.json").read_text())
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert config["devices"] == ["cuda:0", "cuda:1"]
    assert status[0]["name"] == "latency"
