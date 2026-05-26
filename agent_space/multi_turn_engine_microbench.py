from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
from pathlib import Path

from torchinferno.cli import _build_openai_microbench_engine
from torchinferno.cli import _run_openai_microbench_messages
from torchinferno.cli import _sync_openai_engine
from torchinferno.openai_server import _is_tensor_parallel_worker_model
from torchinferno.openai_server import _tensor_parallel_worker_loop


def _generate_turns(count: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    turns = []
    for _ in range(count):
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/":
            b = rng.randint(2, 50)
            a = b * rng.randint(2, 50)
            answer = a // b
        elif op == "*":
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            answer = a * b
        elif op == "-":
            a = rng.randint(50, 2000)
            b = rng.randint(1, a)
            answer = a - b
        else:
            a = rng.randint(1, 2000)
            b = rng.randint(1, 2000)
            answer = a + b
        turns.append((f"{a} {op} {b} =", answer))
    return turns


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize(measurements: list[dict[str, float]]) -> dict[str, float]:
    return {
        "requests": float(len(measurements)),
        "ttft_p50_ms": _median([metric["ttft_ms"] for metric in measurements]),
        "tpot_p50_ms": _median([metric["tpot_ms"] for metric in measurements]),
        "e2e_p50_ms": _median([metric["e2e_ms"] for metric in measurements]),
        "throughput_p50_tps": _median([metric["throughput_tps"] for metric in measurements]),
        "output_tokens_p50": _median([metric["output_tokens"] for metric in measurements]),
    }


def _run_case(engine, conversations: int, turns_per_conversation: int) -> list[dict[str, float]]:
    _sync_openai_engine(engine)
    measurements: list[dict[str, float]] = []

    def run_conversation(index: int) -> list[dict[str, float]]:
        metrics: list[dict[str, float]] = []
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": "You are a calculator. Respond with only the numerical answer, nothing else.",
            },
        ]
        for equation, _expected in _generate_turns(turns_per_conversation, seed=index):
            messages.append({"role": "user", "content": equation})
            metric, response = _run_openai_microbench_messages(
                engine,
                messages,
                512,
                0.0,
            )
            metrics.append(metric)
            messages.append({"role": "assistant", "content": response})
        return metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(64, conversations)) as pool:
        futures = [pool.submit(run_conversation, index) for index in range(conversations)]
        for future in concurrent.futures.as_completed(futures):
            measurements.extend(future.result())
    _sync_openai_engine(engine)
    return measurements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--conversations", type=int, default=64)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--restore-cases", default="off,on")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    engine_args = argparse.Namespace(
        backend="model",
        model=args.model,
        model_kind="auto",
        tokenizer=None,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=None,
        device=None,
        dtype="auto",
        max_model_len=None,
        trust_remote_code=True,
        token=None,
        revision=None,
        cache_dir=None,
        cache_backend="dense",
        page_size=16,
        max_batch_size=64,
        batch_wait_ms=10.0,
        llama_parallelism="tensor",
    )
    engine = _build_openai_microbench_engine(engine_args)
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return 0

    results: dict[str, object] = {"cases": {}}
    try:
        selected_cases = tuple(part.strip() for part in args.restore_cases.split(",") if part.strip())
        case_values = {
            "off": ("restore_off", "0"),
            "on": ("restore_on", "1"),
        }
        for case_name in selected_cases:
            label, enabled = case_values[case_name]
            os.environ["TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE"] = enabled
            clear_prefix_cache = getattr(engine, "_clear_prefix_cache", None)
            if callable(clear_prefix_cache):
                clear_prefix_cache()
            measurements = _run_case(engine, args.conversations, args.turns)
            summary = _summarize(measurements)
            results["cases"][label] = {**summary, "raw_requests": measurements}
            print(
                f"case={label} requests={int(summary['requests'])} "
                f"ttft_p50_ms={summary['ttft_p50_ms']:.3f} "
                f"tpot_p50_ms={summary['tpot_p50_ms']:.3f} "
                f"e2e_p50_ms={summary['e2e_p50_ms']:.3f} "
                f"throughput_p50_tps={summary['throughput_p50_tps']:.2f} "
                f"output_tokens_p50={summary['output_tokens_p50']:.0f}"
            )
    finally:
        engine.close()
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
