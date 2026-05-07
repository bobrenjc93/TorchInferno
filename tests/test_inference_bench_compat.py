from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def test_openai_server_matches_inference_bench_contract() -> None:
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = [
        sys.executable,
        "-m",
        "torchinferno.openai_server",
        "--model",
        "tiny",
        "--model-kind",
        "tiny-deepseek",
        "--tokenizer",
        "byte",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-model-len",
        "32",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_models(port, proc)
        models = _json_get(f"http://127.0.0.1:{port}/v1/models")
        assert models["object"] == "list"
        assert models["data"][0]["id"] == "tiny"

        body = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": False,
        }
        completion = _json_post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"]["role"] == "assistant"

        stream_body = {**body, "stream": True}
        lines = _stream_post(f"http://127.0.0.1:{port}/v1/chat/completions", stream_body)
        assert any(line.startswith("data: {") and "chat.completion.chunk" in line for line in lines)
        assert "data: [DONE]" in lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_inference_bench_provider_adapter_points_at_openai_server() -> None:
    provider_path = Path(__file__).resolve().parents[1] / "integrations" / "inference_bench" / "torchinferno.py"
    provider = provider_path.read_text()
    assert "@register(\"torchinferno\")" in provider
    assert ".[serve]" in provider
    assert "torchinferno.openai_server" in provider
    assert "--tensor-parallel-size" in provider


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_models(port: int, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"server exited early with code {proc.returncode}:\n{output}")
        try:
            _json_get(url)
            return
        except Exception:
            time.sleep(0.25)
    output = proc.stdout.read() if proc.stdout is not None else ""
    raise TimeoutError(f"server did not become ready:\n{output}")


def _json_get(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_post(url: str, payload: dict[str, object]) -> list[str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        lines = []
        while True:
            line = response.readline()
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue
            lines.append(decoded)
            if decoded == "data: [DONE]":
                break
        return lines
