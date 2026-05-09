from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class OpenAIServerMicrobenchConfig:
    model: str = "tiny"
    model_kind: str = "tiny-deepseek"
    tokenizer: str | None = "byte"
    host: str = "127.0.0.1"
    port: int = 0
    base_url: str | None = None
    python: str = sys.executable
    device: str | None = "cpu"
    dtype: str = "float32"
    max_model_len: int | None = 64
    tensor_parallel_size: int = 1
    cache_backend: str = "dense"
    page_size: int = 16
    max_batch_size: int = 32
    batch_wait_ms: float = 10.0
    llama_parallelism: str = "auto"
    trust_remote_code: bool = False
    token: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    modes: tuple[str, ...] = ("non-stream", "stream")
    prompt_tokens: int = 8
    max_tokens: int = 2
    concurrency: int = 1
    warmup: int = 1
    iters: int = 3
    temperature: float = 0.0
    ready_timeout_s: float = 30.0
    request_timeout_s: float = 30.0
    json_output: Path | None = None


@dataclass(frozen=True)
class _RequestMetric:
    latency_ms: float
    ttft_ms: float
    tpot_ms: float
    prompt_tokens: int
    output_tokens: int


def build_openai_server_microbench_command(config: OpenAIServerMicrobenchConfig, port: int) -> tuple[str, ...]:
    command: list[str] = [
        config.python,
        "-m",
        "torchinferno.openai_server",
        "--model",
        config.model,
        "--model-kind",
        config.model_kind,
        "--host",
        config.host,
        "--port",
        str(port),
        "--dtype",
        config.dtype,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--cache-backend",
        config.cache_backend,
        "--page-size",
        str(config.page_size),
        "--max-batch-size",
        str(config.max_batch_size),
        "--batch-wait-ms",
        str(config.batch_wait_ms),
        "--llama-parallelism",
        config.llama_parallelism,
    ]
    _append_optional(command, "--tokenizer", config.tokenizer)
    _append_optional(command, "--device", config.device)
    _append_optional(command, "--max-model-len", config.max_model_len)
    _append_optional(command, "--token", config.token)
    _append_optional(command, "--revision", config.revision)
    _append_optional(command, "--cache-dir", config.cache_dir)
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    return tuple(command)


def run_openai_server_microbench(config: OpenAIServerMicrobenchConfig) -> dict[str, Any]:
    modes = _normalize_modes(config.modes)
    port = config.port if config.port != 0 else _free_port()
    proc: subprocess.Popen[str] | None = None
    command: tuple[str, ...] | None = None
    startup_ms = 0.0
    base_url = _normalize_base_url(config.base_url) if config.base_url else f"http://{config.host}:{port}/v1"
    try:
        if config.base_url is None:
            command = build_openai_server_microbench_command(config, port)
            startup_start = time.perf_counter()
            proc = subprocess.Popen(
                command,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for_models(base_url, proc, config.ready_timeout_s)
            startup_ms = (time.perf_counter() - startup_start) * 1000.0
        else:
            startup_start = time.perf_counter()
            _wait_for_models(base_url, None, config.ready_timeout_s)
            startup_ms = (time.perf_counter() - startup_start) * 1000.0

        result: dict[str, Any] = {
            "config": _config_to_json(config),
            "base_url": base_url,
            "started_server": config.base_url is None,
            "startup_ms": startup_ms,
            "server": {
                "command": list(command) if command is not None else None,
                "port": port,
            },
            "results": {},
        }
        for mode in modes:
            result["results"][mode] = _run_mode(base_url, config, mode)
        if config.json_output is not None:
            config.json_output.parent.mkdir(parents=True, exist_ok=True)
            config.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        if proc is not None:
            _terminate_process(proc)


def format_openai_server_microbench_report(result: dict[str, Any]) -> str:
    lines = [
        "TorchInferno OpenAI server microbench",
        (
            f"base_url={result['base_url']} started_server={result['started_server']} "
            f"startup_ms={float(result['startup_ms']):.2f}"
        ),
    ]
    for mode, metrics in result["results"].items():
        lines.append(
            f"mode={mode} concurrency={metrics['concurrency']} completed={metrics['completed']} "
            f"request_throughput={metrics['request_throughput']:.2f}/s "
            f"output_throughput={metrics['output_token_throughput']:.2f} tok/s "
            f"latency_p50_ms={metrics['latency_p50_ms']:.2f} latency_p99_ms={metrics['latency_p99_ms']:.2f} "
            f"ttft_p50_ms={metrics['ttft_p50_ms']:.2f} tpot_p50_ms={metrics['tpot_p50_ms']:.2f}"
        )
    json_output = result.get("config", {}).get("json_output")
    if json_output:
        lines.append(f"json_output={json_output}")
    return "\n".join(lines)


def _run_mode(base_url: str, config: OpenAIServerMicrobenchConfig, mode: str) -> dict[str, float | int | str]:
    stream = mode == "stream"
    for iteration in range(config.warmup):
        _run_iteration(base_url, config, stream, iteration)

    start = time.perf_counter()
    metrics = [
        metric
        for iteration in range(config.iters)
        for metric in _run_iteration(base_url, config, stream, config.warmup + iteration)
    ]
    duration_s = time.perf_counter() - start
    return _summarize_metrics(metrics, duration_s, config.concurrency)


def _run_iteration(
    base_url: str,
    config: OpenAIServerMicrobenchConfig,
    stream: bool,
    iteration: int,
) -> list[_RequestMetric]:
    if config.concurrency <= 1:
        return [_run_request(base_url, config, stream, iteration, 0)]

    barrier = threading.Barrier(config.concurrency + 1)
    results: list[_RequestMetric | None] = [None for _ in range(config.concurrency)]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            results[index] = _run_request(base_url, config, stream, iteration, index)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(config.concurrency)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [metric for metric in results if metric is not None]


def _run_request(
    base_url: str,
    config: OpenAIServerMicrobenchConfig,
    stream: bool,
    iteration: int,
    request_index: int,
) -> _RequestMetric:
    payload = _request_payload(config, stream, iteration, request_index)
    start = time.perf_counter()
    if stream:
        output_tokens, token_times = _stream_chat_completion(base_url, payload, config.request_timeout_s)
        end = time.perf_counter()
        ttft_ms = ((token_times[0] - start) * 1000.0) if token_times else 0.0
        tpot_ms = ((token_times[-1] - token_times[0]) / (len(token_times) - 1) * 1000.0) if len(token_times) > 1 else 0.0
        return _RequestMetric(
            latency_ms=(end - start) * 1000.0,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            prompt_tokens=config.prompt_tokens,
            output_tokens=output_tokens,
        )

    response = _json_post(f"{base_url}/chat/completions", payload, config.request_timeout_s)
    end = time.perf_counter()
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", config.prompt_tokens)) if isinstance(usage, dict) else config.prompt_tokens
    output_tokens = int(usage.get("completion_tokens", config.max_tokens)) if isinstance(usage, dict) else config.max_tokens
    return _RequestMetric(
        latency_ms=(end - start) * 1000.0,
        ttft_ms=0.0,
        tpot_ms=0.0,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def _stream_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, list[float]]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token_times: list[float] = []
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        while True:
            line = response.readline()
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            if not decoded or not decoded.startswith("data: "):
                continue
            data = decoded.removeprefix("data: ")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict) and delta.get("content"):
                token_times.append(time.perf_counter())
    return len(token_times), token_times


def _summarize_metrics(metrics: Sequence[_RequestMetric], duration_s: float, concurrency: int) -> dict[str, float | int | str]:
    completed = len(metrics)
    prompt_tokens = sum(metric.prompt_tokens for metric in metrics)
    output_tokens = sum(metric.output_tokens for metric in metrics)
    total_tokens = prompt_tokens + output_tokens
    duration = max(duration_s, 1e-9)
    latencies = [metric.latency_ms for metric in metrics]
    ttfts = [metric.ttft_ms for metric in metrics if metric.ttft_ms > 0.0]
    tpots = [metric.tpot_ms for metric in metrics if metric.tpot_ms > 0.0]
    return {
        "completed": completed,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "request_throughput": completed / duration,
        "output_token_throughput": output_tokens / duration,
        "total_token_throughput": total_tokens / duration,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_mean_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p99_ms": _percentile(latencies, 99),
        "ttft_mean_ms": _mean(ttfts),
        "ttft_p50_ms": _percentile(ttfts, 50),
        "ttft_p99_ms": _percentile(ttfts, 99),
        "tpot_mean_ms": _mean(tpots),
        "tpot_p50_ms": _percentile(tpots, 50),
        "tpot_p99_ms": _percentile(tpots, 99),
    }


def _request_payload(
    config: OpenAIServerMicrobenchConfig,
    stream: bool,
    iteration: int,
    request_index: int,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": _prompt_text(config.prompt_tokens, iteration, request_index)}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": stream,
    }


def _prompt_text(prompt_tokens: int, iteration: int, request_index: int) -> str:
    offset = (iteration * 131 + request_index * 17) % 997
    return " ".join(f"tok{(offset + idx) % 997:03d}" for idx in range(max(1, prompt_tokens)))


def _wait_for_models(base_url: str, proc: subprocess.Popen[str] | None, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error: BaseException | None = None
    while time.perf_counter() < deadline:
        if proc is not None and proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"OpenAI server exited early with code {proc.returncode}:\n{output}")
        try:
            _json_get(f"{base_url}/models", timeout_s=2.0)
            return
        except BaseException as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"OpenAI server did not become ready at {base_url}: {last_error!r}")


def _json_get(url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_post(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _normalize_modes(modes: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for mode in modes:
        lowered = mode.lower()
        if lowered == "both":
            normalized.extend(("non-stream", "stream"))
        elif lowered in {"non-stream", "stream"}:
            normalized.append(lowered)
        else:
            raise ValueError(f"unsupported mode {mode!r}; expected non-stream, stream, or both")
    return tuple(dict.fromkeys(normalized)) or ("non-stream",)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _config_to_json(config: OpenAIServerMicrobenchConfig) -> dict[str, Any]:
    data = asdict(config)
    if config.json_output is not None:
        data["json_output"] = str(config.json_output)
    return data


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
