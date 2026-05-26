from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path

from torchinferno.cli import _build_openai_microbench_engine
from torchinferno.cli import _run_openai_microbench_messages
from torchinferno.cli import _sync_openai_engine
from torchinferno.openai_server import _is_tensor_parallel_worker_model
from torchinferno.openai_server import _tensor_parallel_worker_loop


BRANCHES = 4
DEPTH = 3
SYSTEM_PROMPT = "You are a calculator. Respond with only the numerical answer, nothing else."


def _generate_equations(n: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    equations = []
    for _ in range(n):
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
        equations.append((f"{a} {op} {b} =", answer))
    return equations


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _run_tree(engine, tree_idx: int) -> list[dict[str, float]]:
    equations = _generate_equations(50, seed=tree_idx)
    metrics: list[dict[str, float]] = []
    eq_idx = 0
    for depth in range(DEPTH):
        num_candidates = max(1, BRANCHES // (depth + 1))
        for _cand_idx in range(num_candidates):
            barrier = threading.Barrier(BRANCHES + 1)
            results: list[dict[str, float] | None] = [None for _ in range(BRANCHES)]
            errors: list[BaseException] = []

            def branch_worker(branch: int) -> None:
                try:
                    eq, _expected = equations[(eq_idx + branch) % len(equations)]
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": eq},
                    ]
                    barrier.wait(timeout=60)
                    branch_metrics, _content = _run_openai_microbench_messages(
                        engine,
                        messages,
                        300,
                        0.7,
                    )
                    results[branch] = branch_metrics
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=branch_worker, args=(branch,)) for branch in range(BRANCHES)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=60)
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            metrics.extend(metric for metric in results if metric is not None)
            eq_idx += BRANCHES

        eq, _expected = equations[eq_idx % len(equations)]
        eval_metrics, _content = _run_openai_microbench_messages(
            engine,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": eq},
            ],
            400,
            0.0,
        )
        metrics.append(eval_metrics)
        eq_idx += 1
    return metrics


def _run_case(engine, trees: int, tree_workers: int) -> list[dict[str, float]]:
    _sync_openai_engine(engine)
    next_tree = 0
    lock = threading.Lock()
    all_metrics: list[dict[str, float]] = []
    errors: list[BaseException] = []

    def tree_worker() -> None:
        nonlocal next_tree
        while True:
            with lock:
                if next_tree >= trees:
                    return
                tree_idx = next_tree
                next_tree += 1
            try:
                metrics = _run_tree(engine, tree_idx)
            except BaseException as exc:
                errors.append(exc)
                return
            with lock:
                all_metrics.extend(metrics)

    threads = [threading.Thread(target=tree_worker) for _ in range(tree_workers)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    _sync_openai_engine(engine)
    if errors:
        raise errors[0]
    for metric in all_metrics:
        metric["case_elapsed_s"] = elapsed
    return all_metrics


def _summarize(measurements: list[dict[str, float]]) -> dict[str, float]:
    return {
        "requests": float(len(measurements)),
        "ttft_p50_ms": _median([metric["ttft_ms"] for metric in measurements]),
        "tpot_p50_ms": _median([metric["tpot_ms"] for metric in measurements if metric["tpot_ms"] > 0]),
        "e2e_p50_ms": _median([metric["e2e_ms"] for metric in measurements]),
        "throughput_p50_tps": _median([metric["throughput_tps"] for metric in measurements]),
        "output_tokens_p50": _median([metric["output_tokens"] for metric in measurements]),
        "elapsed_s": max((metric.get("case_elapsed_s", 0.0) for metric in measurements), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--trees", type=int, default=32)
    parser.add_argument("--tree-workers", type=int, default=16)
    parser.add_argument("--warmup-trees", type=int, default=2)
    parser.add_argument("--case-order", default="default,temp_wait512")
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
        max_batch_size=128,
        batch_wait_ms=10.0,
        llama_parallelism="tensor",
    )
    engine = _build_openai_microbench_engine(engine_args)
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return 0

    case_values = {
        "default": None,
        "temp_wait512": "512",
    }
    cases = {label: case_values[label] for label in args.case_order.split(",") if label in case_values}
    results: dict[str, object] = {"cases": {}}
    try:
        for label, max_tokens in cases.items():
            if max_tokens is None:
                os.environ.pop("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", None)
            else:
                os.environ["TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS"] = max_tokens
            clear_prefix_cache = getattr(engine, "_clear_prefix_cache", None)
            if callable(clear_prefix_cache):
                clear_prefix_cache()
            if args.warmup_trees > 0:
                _run_case(engine, args.warmup_trees, min(args.tree_workers, args.warmup_trees))
            measurements = _run_case(engine, args.trees, args.tree_workers)
            summary = _summarize(measurements)
            results["cases"][label] = {
                **summary,
                "raw_requests": measurements,
            }
            print(
                f"case={label} requests={int(summary['requests'])} "
                f"ttft_p50_ms={summary['ttft_p50_ms']:.3f} "
                f"tpot_p50_ms={summary['tpot_p50_ms']:.3f} "
                f"e2e_p50_ms={summary['e2e_p50_ms']:.3f} "
                f"throughput_p50_tps={summary['throughput_p50_tps']:.2f} "
                f"elapsed_s={summary['elapsed_s']:.2f}"
            )
    finally:
        engine.close()
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
