from __future__ import annotations

import json
import os
import subprocess
import sys

from torchinferno.benchmarks import OpenAIServerMicrobenchConfig, build_openai_server_microbench_command


def test_openai_server_microbench_command_uses_openai_server_entrypoint() -> None:
    config = OpenAIServerMicrobenchConfig(
        model="/models/tiny",
        model_kind="llama3",
        tokenizer="byte",
        python="python3",
        tensor_parallel_size=2,
        llama_parallelism="pipeline",
    )

    command = build_openai_server_microbench_command(config, port=8123)

    assert command[:3] == ("python3", "-m", "torchinferno.openai_server")
    assert command[command.index("--model") + 1] == "/models/tiny"
    assert command[command.index("--port") + 1] == "8123"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--llama-parallelism") + 1] == "pipeline"


def test_openai_server_microbench_cli_runs_tiny_server(tmp_path) -> None:
    output = tmp_path / "openai-server-microbench.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "openai-server-microbench",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--prompt-tokens",
            "3",
            "--max-tokens",
            "2",
            "--mode",
            "both",
            "--json-output",
            str(output),
        ],
        check=True,
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert "TorchInferno OpenAI server microbench" in result.stdout
    assert "mode=non-stream" in result.stdout
    assert "mode=stream" in result.stdout
    payload = json.loads(output.read_text())
    assert payload["started_server"] is True
    assert payload["results"]["non-stream"]["completed"] == 1
    assert payload["results"]["stream"]["completed"] == 1
    assert payload["results"]["non-stream"]["output_tokens"] == 2
    assert payload["results"]["stream"]["output_tokens"] == 2
