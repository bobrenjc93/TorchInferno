from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import threading
import time

import torch
import torch.distributed as dist

from torchinferno.openai_http import OpenAIHTTPServer
from torchinferno.openai_server import (
    OpenAIServerConfig,
    _is_tensor_parallel_worker_model,
    _tensor_parallel_worker_loop,
    build_engine,
)


MESSAGES = [
    {"role": "system", "content": "You are a calculator. Respond with only the numerical answer, nothing else."},
    {"role": "user", "content": "17 * 23 ="},
]


def _stream_request(port: int, model: str) -> dict[str, float | int | str]:
    chunks: list[str] = []
    body = json.dumps(
        {
            "model": model,
            "messages": MESSAGES,
            "temperature": 0.7,
            "max_tokens": 256,
            "stream": True,
        }
    )
    start = time.perf_counter()
    ttft = 0.0
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300.0)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
    while True:
        line = response.fp.readline() if response.fp is not None else b""
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: ") :].strip()
        if payload == b"[DONE]":
            break
        data = json.loads(payload)
        choices = data.get("choices") or []
        if not choices:
            continue
        content = ((choices[0].get("delta") or {}).get("content")) if isinstance(choices[0], dict) else None
        if content:
            if not ttft:
                ttft = (time.perf_counter() - start) * 1000.0
            chunks.append(str(content))
    conn.close()
    e2e = (time.perf_counter() - start) * 1000.0
    return {
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "chunks": len(chunks),
        "text": "".join(chunks),
    }


def _median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--requests", type=int, default=16)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    engine = build_engine(
        OpenAIServerConfig(
            model=args.model,
            host="127.0.0.1",
            port=args.port,
            tensor_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
            device=f"cuda:{local_rank}",
            dtype="auto",
            trust_remote_code=True,
            llama_parallelism="tensor",
            batch_wait_ms=float(os.environ.get("TORCHINFERNO_PROFILE_BATCH_WAIT_MS", "10")),
        )
    )
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return

    records: list[dict[str, object]] = []
    original = engine._generate_batch_steps
    original_identical = engine._generate_identical_prompt_batch_steps

    def timed_generate_batch_steps(*call_args, **call_kwargs):
        batch = int(call_args[0].size(0)) if call_args else -1
        prompt_tokens = int(call_args[0].size(1)) if call_args else -1
        temperature = float(call_kwargs.get("temperature", -1.0))
        started = time.perf_counter()
        steps = 0
        first_step_ms = None
        for step in original(*call_args, **call_kwargs):
            steps += 1
            if first_step_ms is None:
                first_step_ms = (time.perf_counter() - started) * 1000.0
            yield step
        records.append(
            {
                "batch": batch,
                "prompt_tokens": prompt_tokens,
                "temperature": temperature,
                "steps": steps,
                "first_step_ms": first_step_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    engine._generate_batch_steps = timed_generate_batch_steps  # type: ignore[method-assign]

    def timed_generate_identical_prompt_batch_steps(*call_args, **call_kwargs):
        batch = int(call_kwargs.get("batch_size", -1))
        prompt_tokens = int(call_args[0].size(1)) if call_args else -1
        started = time.perf_counter()
        steps = 0
        first_step_ms = None
        for step in original_identical(*call_args, **call_kwargs):
            steps += 1
            if first_step_ms is None:
                first_step_ms = (time.perf_counter() - started) * 1000.0
            yield step
        records.append(
            {
                "kind": "identical",
                "batch": batch,
                "prompt_tokens": prompt_tokens,
                "steps": steps,
                "first_step_ms": first_step_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    engine._generate_identical_prompt_batch_steps = timed_generate_identical_prompt_batch_steps  # type: ignore[method-assign]

    server = OpenAIHTTPServer(("127.0.0.1", args.port), engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as pool:
            futures = [pool.submit(_stream_request, args.port, args.model) for _ in range(args.requests)]
            client_metrics = [future.result() for future in concurrent.futures.as_completed(futures)]
        print(
            json.dumps(
                {
                    "client": {
                        "ttft_median_ms": _median([float(item["ttft_ms"]) for item in client_metrics]),
                        "e2e_median_ms": _median([float(item["e2e_ms"]) for item in client_metrics]),
                        "metrics": client_metrics,
                    },
                    "server_records": records,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=10)
        engine.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
