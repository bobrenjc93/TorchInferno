from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch

from torchinferno.openai_server import (
    OpenAICompletionEngine,
    _ByteFallbackTokenizer,
    _TransformersChatTokenizer,
)


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
        assert completion["usage"]["prompt_tokens"] > 0
        assert completion["usage"]["completion_tokens"] == 2
        assert completion["usage"]["total_tokens"] == completion["usage"]["prompt_tokens"] + 2

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
    assert "--llama-parallelism" not in provider


def test_chat_template_batch_encoding_input_ids_are_extracted() -> None:
    tokenizer = _TransformersChatTokenizer(_BatchEncodingTokenizer())

    encoded = tokenizer.encode_messages([{"role": "user", "content": "hello"}])

    assert encoded == [7, 8, 9]


def test_openai_engine_microbatches_same_shape_requests() -> None:
    model = _BatchRecordingModel()
    tokenizer = _BarrierByteFallbackTokenizer(vocab_size=8, parties=2)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
    )
    barrier = threading.Barrier(3)
    results: list[list[int] | None] = [None, None]

    def run(index: int) -> None:
        barrier.wait()
        completion = engine.complete_chat(
            [{"role": "user", "content": "same"}],
            max_tokens=2,
            temperature=0.0,
        )
        results[index] = completion.tokens

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
    engine.close()

    assert results == [[2, 2], [2, 2]]
    assert model.calls[0][0] == 2


def test_openai_engine_single_request_skips_batch_wait() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    start = time.perf_counter()
    try:
        completion = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=1,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert completion.tokens == [2]
    assert time.perf_counter() - start < 0.5
    assert model.calls[0][0] == 1
    assert model.calls[0][2] != "torchinferno-openai-batcher"


def test_openai_engine_single_stream_request_uses_direct_path() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "single"}],
                max_tokens=2,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2, 2]
    assert model.calls[0][0] == 1
    assert model.calls[0][2] != "torchinferno-openai-batcher"


def test_openai_engine_single_request_does_not_use_batched_step_loop() -> None:
    model = _BatchRecordingModel()
    engine = _NoBatchStepEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        completion = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert completion.tokens == [2, 2]
    assert engine.batch_step_calls == 0


def test_openai_engine_reuses_resettable_generation_cache() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        first = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
        second = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert first.tokens == [2, 2]
    assert second.tokens == [2, 2]
    assert model.cache_allocations == 1


def test_openai_microbench_cli_runs_synthetic_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = [
        sys.executable,
        "-m",
        "torchinferno.cli",
        "openai-microbench",
        "--backend",
        "synthetic",
        "--device",
        "cpu",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--prompt-tokens",
        "3",
        "--max-tokens",
        "2",
        "--compare-batcher",
        "--concurrency",
        "2",
    ]

    result = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, check=True, timeout=30)

    assert "TorchInferno OpenAI microbench" in result.stdout
    assert "case=single-direct" in result.stdout
    assert "case=single-batcher" in result.stdout
    assert "case=concurrent-2" in result.stdout


class _BatchEncodingTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize
        assert add_generation_prompt
        assert messages[0]["role"] == "user"
        return {"input_ids": [[7, 8, 9]], "attention_mask": [[1, 1, 1]]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("apply_chat_template should be used")

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(str(token_id) for token_id in token_ids)


class _BarrierByteFallbackTokenizer(_ByteFallbackTokenizer):
    def __init__(self, *, vocab_size: int, parties: int) -> None:
        super().__init__(vocab_size)
        self.barrier = threading.Barrier(parties)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        self.barrier.wait(timeout=5)
        return super().encode_messages(messages)


class _BatchRecordingCache:
    def __init__(self) -> None:
        self.seq_len = 0


class _BatchRecordingModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 8})()
        self.calls: list[tuple[int, int, str]] = []
        self.cache_allocations = 0

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _BatchRecordingCache:
        self.cache_allocations += 1
        return _BatchRecordingCache()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _BatchRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        self.calls.append((input_ids.size(0), input_ids.size(1), threading.current_thread().name))
        cache.seq_len += input_ids.size(1)
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        logits = torch.zeros(input_ids.size(0), tokens, 8)
        logits[..., 2] = 1.0
        return logits, cache


class _NoBatchStepEngine(OpenAICompletionEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_step_calls = 0

    def _generate_batch_steps(self, *args, **kwargs):
        self.batch_step_calls += 1
        raise AssertionError("single-request fast path must not use batched step generation")


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
