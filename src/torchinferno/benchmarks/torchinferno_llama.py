from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from torchinferno.benchmarks.vllm_compatible import (
    LLAMA3_70B_MODEL,
    SUPPORTED_BENCHMARKS,
    collect_vllm_benchmark_summary,
    plot_vllm_benchmark_results,
)
from torchinferno.models.llama3.pipeline import Llama3PipelineForCausalLM
from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.sampling import sample_next_token


@dataclass(frozen=True)
class TorchInfernoLlamaBenchmarkConfig:
    output_dir: str | Path
    model: str = LLAMA3_70B_MODEL
    devices: tuple[str, ...] = ()
    dtype: str = "auto"
    input_len: int = 32
    output_len: int = 128
    batch_size: int = 8
    num_iters_warmup: int = 1
    num_iters: int = 3
    num_prompts: int = 1000
    request_rate: str | float = "inf"
    max_concurrency: int = 256
    temperature: float = 1.0
    seed: int = 0
    benchmarks: tuple[str, ...] = ("latency", "throughput", "serve")
    token: str | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    parallelism: str = "pipeline"
    profile_breakdown: bool = False


@dataclass(frozen=True)
class TorchInfernoLlamaBenchmarkArtifacts:
    output_dir: Path
    config_path: Path
    status_path: Path
    summary_path: Path
    plot_path: Path | None
    csv_path: Path | None


def run_torchinferno_llama_benchmark_suite(
    config: TorchInfernoLlamaBenchmarkConfig,
    *,
    run: bool = False,
    plot: bool = True,
) -> TorchInfernoLlamaBenchmarkArtifacts:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "suite_config.json"
    status_path = output_dir / "run_status.json"
    summary_path = output_dir / "summary.json"
    benchmarks = _normalize_benchmarks(config.benchmarks)
    primary = _process_is_primary()
    if primary:
        config_path.write_text(json.dumps(_config_to_json(config), indent=2, sort_keys=True) + "\n")

    statuses: list[dict[str, object]]
    if run:
        torch.manual_seed(config.seed)
        if config.parallelism == "tensor":
            model = Llama3TensorParallelForCausalLM.from_pretrained(
                config.model,
                dtype=config.dtype,
                token=config.token,
                revision=config.revision,
                cache_dir=config.cache_dir,
            ).eval()
        elif config.parallelism == "pipeline":
            model = Llama3PipelineForCausalLM.from_pretrained(
                config.model,
                devices=config.devices or None,
                dtype=config.dtype,
                token=config.token,
                revision=config.revision,
                cache_dir=config.cache_dir,
            ).eval()
        else:
            raise ValueError("parallelism must be 'pipeline' or 'tensor'")
        if config.profile_breakdown and hasattr(model, "enable_profile"):
            model.enable_profile()
        primary = _model_is_primary(model)
        if primary:
            (output_dir / "load_report.json").write_text(
                json.dumps(model.load_report.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        statuses = []
        for benchmark in benchmarks:
            start = time.perf_counter()
            if benchmark == "latency":
                _run_latency(model, config, output_dir / "latency.json")
            elif benchmark == "throughput":
                _run_throughput(model, config, output_dir / "throughput.json")
            elif benchmark == "serve":
                _run_serve(model, config, output_dir / "serve.json")
            elapsed = time.perf_counter() - start
            if primary:
                statuses.append(
                    {
                        "name": benchmark,
                        "returncode": 0,
                        "status": "passed",
                        "elapsed_s": elapsed,
                        "output_json": str(output_dir / f"{benchmark}.json"),
                    }
                )
                status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            _barrier_model(model)
        if config.profile_breakdown and hasattr(model, "profile_summary"):
            rank = getattr(model, "rank", 0)
            (output_dir / f"profile_breakdown_rank{rank}.json").write_text(
                json.dumps(model.profile_summary(), indent=2, sort_keys=True) + "\n"
            )
    else:
        statuses = [
            {
                "name": benchmark,
                "returncode": None,
                "status": "planned",
                "output_json": str(output_dir / f"{benchmark}.json"),
            }
            for benchmark in benchmarks
        ]
        if primary:
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")

    summary: dict[str, Any] = {"results": {}}
    if primary:
        summary = collect_vllm_benchmark_summary(output_dir)
        summary["backend"] = "torchinferno"
        summary["model"] = config.model
        summary["benchmarks"] = list(benchmarks)
        summary["parallelism"] = config.parallelism
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    plot_path: Path | None = None
    csv_path: Path | None = None
    if primary and plot and summary.get("results"):
        plot_path = output_dir / "performance.html"
        csv_path = output_dir / "performance.csv"
        plot_vllm_benchmark_results(output_dir, output_html=plot_path, output_csv=csv_path)

    return TorchInfernoLlamaBenchmarkArtifacts(
        output_dir=output_dir,
        config_path=config_path,
        status_path=status_path,
        summary_path=summary_path,
        plot_path=plot_path,
        csv_path=csv_path,
    )


@torch.inference_mode()
def _run_latency(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    config: TorchInfernoLlamaBenchmarkConfig,
    output_path: Path,
) -> None:
    prompt = _random_prompts(config.batch_size, config.input_len, model.config.vocab_size)
    for _ in range(config.num_iters_warmup):
        model.generate(prompt, max_new_tokens=config.output_len, temperature=config.temperature)
    _sync_model_devices(model)

    latencies: list[float] = []
    for _ in range(config.num_iters):
        _sync_model_devices(model)
        start = time.perf_counter()
        model.generate(prompt, max_new_tokens=config.output_len, temperature=config.temperature)
        _sync_model_devices(model)
        latencies.append(time.perf_counter() - start)

    output = {
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "latencies": latencies,
        "percentiles": {str(p): _percentile(latencies, p) for p in (10, 25, 50, 75, 90, 99)},
    }
    if _model_is_primary(model):
        output_path.write_text(json.dumps(output, indent=2) + "\n")


@torch.inference_mode()
def _run_throughput(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    config: TorchInfernoLlamaBenchmarkConfig,
    output_path: Path,
) -> None:
    remaining = config.num_prompts
    engine_batch_size = max(1, config.max_concurrency)
    warmup_batch = min(engine_batch_size, config.num_prompts)
    for _ in range(config.num_iters_warmup):
        prompt = _random_prompts(warmup_batch, config.input_len, model.config.vocab_size)
        model.generate(prompt, max_new_tokens=config.output_len, temperature=config.temperature)
    _sync_model_devices(model)
    start = time.perf_counter()
    while remaining > 0:
        batch = min(engine_batch_size, remaining)
        prompt = _random_prompts(batch, config.input_len, model.config.vocab_size)
        model.generate(prompt, max_new_tokens=config.output_len, temperature=config.temperature)
        remaining -= batch
    _sync_model_devices(model)
    elapsed = time.perf_counter() - start
    total_tokens = config.num_prompts * (config.input_len + config.output_len)
    output = {
        "elapsed_time": elapsed,
        "num_requests": config.num_prompts,
        "total_num_tokens": total_tokens,
        "requests_per_second": config.num_prompts / elapsed if elapsed else 0.0,
        "tokens_per_second": total_tokens / elapsed if elapsed else 0.0,
    }
    if _model_is_primary(model):
        output_path.write_text(json.dumps(output, indent=2) + "\n")


@torch.inference_mode()
def _run_serve(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    config: TorchInfernoLlamaBenchmarkConfig,
    output_path: Path,
) -> None:
    if str(config.request_rate) != "inf":
        raise ValueError("native Llama serve benchmark currently supports request_rate=inf only")
    if config.max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    completed = 0
    failed = 0
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    itl_ms: list[float] = []
    e2el_ms: list[float] = []
    remaining = config.num_prompts

    _sync_model_devices(model)
    suite_start = time.perf_counter()
    while remaining > 0:
        batch = min(config.max_concurrency, remaining)
        prompt = _random_prompts(batch, config.input_len, model.config.vocab_size)
        batch_metrics = _generate_with_timing(model, prompt, config.output_len, config.temperature)
        ttft_ms.extend([batch_metrics["ttft_ms"]] * batch)
        tpot_ms.extend([batch_metrics["tpot_ms"]] * batch)
        e2el_ms.extend([batch_metrics["e2el_ms"]] * batch)
        for value in batch_metrics["itl_ms"]:
            itl_ms.extend([value] * batch)
        completed += batch
        remaining -= batch
    _sync_model_devices(model)
    duration = time.perf_counter() - suite_start

    total_input_tokens = completed * config.input_len
    total_output_tokens = completed * config.output_len
    output = {
        "date": time.strftime("%Y%m%d-%H%M%S"),
        "endpoint_type": "torchinferno",
        "backend": "torchinferno",
        "model_id": config.model,
        "tokenizer_id": config.model,
        "num_prompts": config.num_prompts,
        "request_rate": str(config.request_rate),
        "burstiness": 1.0,
        "max_concurrency": config.max_concurrency,
        "duration": duration,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": completed / duration if duration else 0.0,
        "request_goodput": None,
        "output_throughput": total_output_tokens / duration if duration else 0.0,
        "total_token_throughput": (total_input_tokens + total_output_tokens) / duration if duration else 0.0,
        "max_output_tokens_per_s": total_output_tokens / duration if duration else 0.0,
        "max_concurrent_requests": config.max_concurrency,
        "rtfx": 0.0,
        **_latency_stats("ttft", ttft_ms),
        **_latency_stats("tpot", tpot_ms),
        **_latency_stats("itl", itl_ms),
        **_latency_stats("e2el", e2el_ms),
    }
    if _model_is_primary(model):
        output_path.write_text(json.dumps(output, indent=2) + "\n")


def _generate_with_timing(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    input_ids: torch.Tensor,
    output_len: int,
    temperature: float,
) -> dict[str, Any]:
    cache = model.allocate_cache(input_ids.size(0), input_ids.size(1) + output_len)
    _sync_model_devices(model)
    start = time.perf_counter()
    logits, cache = _forward_last_logits(model, input_ids, cache)
    next_token = _sample_next(model, logits[:, -1, :], temperature)
    _sync_model_devices(model)
    ttft = time.perf_counter() - start

    decode_steps: list[float] = []
    for _ in range(1, output_len):
        _sync_model_devices(model)
        step_start = time.perf_counter()
        logits, cache = _forward_last_logits(model, next_token[:, None], cache)
        next_token = _sample_next(model, logits[:, -1, :], temperature)
        _sync_model_devices(model)
        decode_steps.append(time.perf_counter() - step_start)

    e2e = ttft + sum(decode_steps)
    denominator = max(1, output_len - 1)
    return {
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": (sum(decode_steps) / denominator) * 1000.0,
        "itl_ms": [value * 1000.0 for value in decode_steps],
        "e2el_ms": e2e * 1000.0,
    }


def _forward_last_logits(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    input_ids: torch.Tensor,
    cache: object,
) -> tuple[torch.Tensor, object]:
    if isinstance(model, Llama3TensorParallelForCausalLM):
        return model.forward(
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
    return model.forward(input_ids, cache=cache, use_cache=True, return_last_logits_only=True)


def _random_prompts(batch_size: int, input_len: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, input_len), dtype=torch.long)


def _sample_next(
    model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM,
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sampler = getattr(model, "_sample_next_token", None)
    if sampler is not None:
        return sampler(logits, temperature)
    if temperature > 0:
        return sample_next_token(logits, temperature).to(model.embed_device, non_blocking=True)
    return torch.argmax(logits, dim=-1).to(model.embed_device, non_blocking=True)


def _sync_model_devices(model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM) -> None:
    for device in model.devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _model_is_primary(model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM) -> bool:
    return bool(getattr(model, "is_primary", True))


def _barrier_model(model: Llama3PipelineForCausalLM | Llama3TensorParallelForCausalLM) -> None:
    if isinstance(model, Llama3TensorParallelForCausalLM):
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
            else:
                dist.barrier()


def _process_is_primary() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _latency_stats(name: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"mean_{name}_ms": sum(values) / len(values) if values else 0.0,
        f"median_{name}_ms": _percentile(values, 50),
        f"std_{name}_ms": _std(values),
        f"p50_{name}_ms": _percentile(values, 50),
        f"p90_{name}_ms": _percentile(values, 90),
        f"p99_{name}_ms": _percentile(values, 99),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _normalize_benchmarks(benchmarks: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for benchmark in benchmarks:
        lowered = benchmark.lower()
        if lowered not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"unsupported benchmark {benchmark!r}; expected one of {SUPPORTED_BENCHMARKS}")
        if lowered not in normalized:
            normalized.append(lowered)
    return tuple(normalized)


def _config_to_json(config: TorchInfernoLlamaBenchmarkConfig) -> dict[str, object]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["cache_dir"] = None if config.cache_dir is None else str(config.cache_dir)
    return data
