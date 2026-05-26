from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path

import torch

from torchinferno.cli import _build_openai_microbench_engine
from torchinferno.cli import _run_openai_microbench_messages
from torchinferno.cli import _summarize_openai_phase_records
from torchinferno.cli import _sync_openai_engine
from torchinferno.openai_server import _is_tensor_parallel_worker_model
from torchinferno.openai_server import _tensor_parallel_worker_loop


FEW_SHOT_EXAMPLES = (
    ("1 * 12345 =", "12345"),
    ("1 * 987654 =", "987654"),
    ("1 * 11223344556677 =", "11223344556677"),
)

SYSTEM_PROMPT = (
    "You are a calculator. Compute the answer to each math equation. "
    "Respond with only the numerical answer, nothing else.\n\n"
    "Examples:\n\n"
    + "\n\n".join(f"Q: {question}\nA: {answer}" for question, answer in FEW_SHOT_EXAMPLES)
)


def _make_big_number(length: int, seed: int = 0) -> str:
    out: list[str] = []
    index = 0
    while len(out) < length:
        digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
        for char in digest:
            if char.isdigit() and len(out) < length:
                out.append(char)
        index += 1
    if out[0] == "0":
        out[0] = "1"
    return "".join(out)


def _messages(request_index: int) -> tuple[list[dict[str, object]], int]:
    length = 25 + (request_index % 176)
    big_number = _make_big_number(length, seed=request_index)
    equation = f"1 * {big_number} ="
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Q: {equation}\nA:"},
    ]
    return messages, len(big_number) // 3 + 16


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _run_iteration(engine, concurrency: int, offset: int) -> list[dict[str, float]]:
    _sync_openai_engine(engine)
    barrier = threading.Barrier(concurrency + 1)
    results: list[dict[str, float] | None] = [None for _ in range(concurrency)]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            request_index = offset + index
            messages, max_tokens = _messages(request_index)
            barrier.wait(timeout=60)
            metrics, _content = _run_openai_microbench_messages(
                engine,
                messages,
                max_tokens,
                0.0,
            )
            results[index] = metrics
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(concurrency)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=60)
    for thread in threads:
        thread.join()
    _sync_openai_engine(engine)
    if errors:
        raise errors[0]
    return [metric for metric in results if metric is not None]


def _summarize(measurements: list[dict[str, float]]) -> dict[str, float]:
    return {
        "requests": float(len(measurements)),
        "ttft_p50_ms": _median([metric["ttft_ms"] for metric in measurements]),
        "tpot_p50_ms": _median([metric["tpot_ms"] for metric in measurements]),
        "e2e_p50_ms": _median([metric["e2e_ms"] for metric in measurements]),
        "throughput_p50_tps": _median([metric["throughput_tps"] for metric in measurements]),
        "output_tokens_p50": _median([metric["output_tokens"] for metric in measurements]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--phase-timings", action="store_true")
    parser.add_argument("--profile-breakdown", action="store_true")
    parser.add_argument(
        "--kernel-cases",
        default=None,
        help="Comma-separated streaming decode attention block sizes to test; use off for non-streaming.",
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    if args.phase_timings:
        os.environ["TORCHINFERNO_OPENAI_PHASE_TIMINGS"] = "1"
    if args.profile_breakdown:
        os.environ["TORCHINFERNO_PROFILE_FAST_PREFILL"] = "1"

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

    if args.kernel_cases:
        cases = {}
        for raw_case in args.kernel_cases.split(","):
            case = raw_case.strip()
            if not case:
                continue
            if case == "off":
                cases["attention_off"] = {"streaming": "0", "block_s": None, "cap": None}
            else:
                cases[f"attention_block{case}"] = {"streaming": "1", "block_s": case, "cap": None}
    else:
        cases = {
            "default": {"streaming": None, "block_s": None, "cap": None},
            "cap32": {"streaming": None, "block_s": None, "cap": "32"},
            "cap64": {"streaming": None, "block_s": None, "cap": "64"},
        }
    results: dict[str, object] = {"cases": {}}
    try:
        for label, case_env in cases.items():
            if args.profile_breakdown and hasattr(engine.model, "enable_profile"):
                engine.model.enable_profile()
            streaming_value = case_env["streaming"]
            block_s_value = case_env["block_s"]
            cap_value = case_env["cap"]
            if streaming_value is None:
                os.environ.pop("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION", None)
            else:
                os.environ["TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION"] = streaming_value
            if block_s_value is None:
                os.environ.pop("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S", None)
            else:
                os.environ["TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S"] = block_s_value
            if cap_value is None:
                os.environ.pop("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", None)
            else:
                os.environ["TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE"] = cap_value
            clear_prefix_cache = getattr(engine, "_clear_prefix_cache", None)
            if callable(clear_prefix_cache):
                clear_prefix_cache()
            for warmup_index in range(args.warmup):
                _run_iteration(engine, args.concurrency, warmup_index * args.concurrency)
            if args.phase_timings and hasattr(engine, "pop_phase_records"):
                engine.pop_phase_records()
            measurements: list[dict[str, float]] = []
            start = time.perf_counter()
            for iteration in range(args.iters):
                offset = (args.warmup + iteration) * args.concurrency
                measurements.extend(_run_iteration(engine, args.concurrency, offset))
            elapsed = time.perf_counter() - start
            summary = _summarize(measurements)
            summary["elapsed_s"] = elapsed
            results["cases"][label] = {
                **summary,
                "raw_requests": measurements,
            }
            if args.profile_breakdown and hasattr(engine.model, "profile_summary"):
                results["cases"][label]["profile_summary"] = engine.model.profile_summary()
            if args.phase_timings and hasattr(engine, "pop_phase_records"):
                phase_records = engine.pop_phase_records()
                if phase_records:
                    phase_summary = _summarize_openai_phase_records(phase_records)
                    results["cases"][label]["phase_timings_ms"] = phase_summary
                    print(
                        "  phase "
                        f"request_to_first_forward_p50_ms={phase_summary.get('request_to_first_forward_p50_ms', 0.0):.3f} "
                        f"broadcast_p50_ms={phase_summary.get('broadcast_p50_ms', 0.0):.3f} "
                        f"cache_p50_ms={phase_summary.get('cache_p50_ms', 0.0):.3f} "
                        f"prefix_cache_p50_ms={phase_summary.get('prefix_cache_p50_ms', 0.0):.3f} "
                        f"prefix_cache_tokens_p50={phase_summary.get('prefix_cache_tokens_p50', 0.0):.0f} "
                        f"prefill_tokens_p50={phase_summary.get('prefill_tokens_p50', 0.0):.0f} "
                        f"prefill_forward_p50_ms={phase_summary.get('prefill_forward_p50_ms', 0.0):.3f} "
                        f"sample_p50_ms={phase_summary.get('sample_p50_ms', 0.0):.3f} "
                        f"first_token_sync_p50_ms={phase_summary.get('first_token_sync_p50_ms', 0.0):.3f}"
                    )
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
