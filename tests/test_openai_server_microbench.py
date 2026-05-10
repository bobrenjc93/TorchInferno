from __future__ import annotations

import json
import os
import subprocess
import sys

from torchinferno.benchmarks import OpenAIServerMicrobenchConfig, build_openai_server_microbench_command
from torchinferno.benchmarks.openai_server import _request_payload


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


def test_openai_server_microbench_self_consistency_prompt_is_identical() -> None:
    config = OpenAIServerMicrobenchConfig(prompt_mode="self-consistency", temperature=0.7, max_tokens=256)

    first = _request_payload(config, stream=True, iteration=0, request_index=0)
    second = _request_payload(config, stream=True, iteration=0, request_index=15)

    assert first["messages"] == second["messages"]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["content"] == "17 * 23 ="
    assert first["temperature"] == 0.7
    assert first["max_tokens"] == 256


def test_openai_server_microbench_few_shot_prompt_has_shared_prefix() -> None:
    config = OpenAIServerMicrobenchConfig(prompt_mode="few-shot", temperature=0.0, max_tokens=256)

    first = _request_payload(config, stream=True, iteration=0, request_index=0)
    second = _request_payload(config, stream=True, iteration=0, request_index=15)

    assert first["messages"][0] == second["messages"][0]
    assert "Examples:" in first["messages"][0]["content"]
    assert first["messages"][1]["content"].startswith("Q: ")
    assert first["messages"][1] != second["messages"][1]
    assert first["max_tokens"] == 256


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
            "--prompt-mode",
            "self-consistency",
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
    assert payload["config"]["prompt_mode"] == "self-consistency"
    assert payload["results"]["non-stream"]["completed"] == 1
    assert payload["results"]["stream"]["completed"] == 1
    assert payload["results"]["non-stream"]["output_tokens"] == 2
    assert payload["results"]["stream"]["output_tokens"] == 2
