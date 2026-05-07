from __future__ import annotations

import csv
import html
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


LLAMA3_70B_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_VLLM_ROOT = Path("/home/bobren/local/d/vllm")
SUPPORTED_BENCHMARKS = ("latency", "throughput", "serve")


@dataclass(frozen=True)
class VLLMBenchmarkConfig:
    output_dir: str | Path
    model: str = LLAMA3_70B_MODEL
    vllm_root: str | Path | None = DEFAULT_VLLM_ROOT
    python: str = sys.executable
    tensor_parallel_size: int = 8
    dtype: str = "auto"
    input_len: int = 32
    output_len: int = 128
    batch_size: int = 8
    num_iters_warmup: int = 10
    num_iters: int = 30
    num_prompts: int = 1000
    dataset_name: str = "random"
    request_rate: str | float = "inf"
    throughput_backend: str = "vllm"
    serve_backend: str = "vllm"
    base_url: str | None = None
    temperature: float = 1.0
    benchmarks: tuple[str, ...] = SUPPORTED_BENCHMARKS
    disable_detokenize: bool = False
    trust_remote_code: bool = False
    enforce_eager: bool = False
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    engine_args: tuple[str, ...] = field(default_factory=tuple)
    throughput_args: tuple[str, ...] = field(default_factory=tuple)
    serve_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VLLMBenchmarkCommand:
    name: str
    command: tuple[str, ...]
    output_json: Path
    log_path: Path

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": list(self.command),
            "command_text": shlex.join(self.command),
            "output_json": str(self.output_json),
            "log_path": str(self.log_path),
        }


@dataclass(frozen=True)
class VLLMBenchmarkArtifacts:
    output_dir: Path
    config_path: Path
    commands_path: Path
    status_path: Path
    summary_path: Path
    plot_path: Path | None
    csv_path: Path | None


def build_vllm_benchmark_commands(config: VLLMBenchmarkConfig) -> list[VLLMBenchmarkCommand]:
    output_dir = Path(config.output_dir).resolve()
    benchmarks = _normalize_benchmarks(config.benchmarks)
    commands: list[VLLMBenchmarkCommand] = []
    entrypoint = (config.python, "-m", "vllm.entrypoints.cli.main", "bench")
    engine_args = _engine_args(config)

    if "latency" in benchmarks:
        output_json = output_dir / "latency.json"
        command = (
            *entrypoint,
            "latency",
            "--input-len",
            str(config.input_len),
            "--output-len",
            str(config.output_len),
            "--batch-size",
            str(config.batch_size),
            "--num-iters-warmup",
            str(config.num_iters_warmup),
            "--num-iters",
            str(config.num_iters),
            "--output-json",
            str(output_json),
            *engine_args,
        )
        if config.disable_detokenize:
            command = (*command, "--disable-detokenize")
        commands.append(VLLMBenchmarkCommand("latency", command, output_json, output_dir / "latency.log"))

    if "throughput" in benchmarks:
        output_json = output_dir / "throughput.json"
        dataset_args = _throughput_dataset_args(config)
        command = (
            *entrypoint,
            "throughput",
            "--backend",
            config.throughput_backend,
            "--dataset-name",
            config.dataset_name,
            "--num-prompts",
            str(config.num_prompts),
            "--output-json",
            str(output_json),
            *dataset_args,
            *engine_args,
            *config.throughput_args,
        )
        if config.disable_detokenize:
            command = (*command, "--disable-detokenize")
        commands.append(VLLMBenchmarkCommand("throughput", command, output_json, output_dir / "throughput.log"))

    if "serve" in benchmarks:
        output_json = output_dir / "serve.json"
        command = (
            *entrypoint,
            "serve",
            "--backend",
            config.serve_backend,
            "--model",
            config.model,
            "--dataset-name",
            config.dataset_name,
            "--input-len",
            str(config.input_len),
            "--output-len",
            str(config.output_len),
            "--num-prompts",
            str(config.num_prompts),
            "--request-rate",
            str(config.request_rate),
            "--ignore-eos",
            "--temperature",
            str(config.temperature),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,90,99",
            "--save-result",
            "--result-dir",
            str(output_dir),
            "--result-filename",
            output_json.name,
            "--disable-tqdm",
            *config.serve_args,
        )
        if config.base_url is not None:
            command = (*command, "--base-url", config.base_url)
        commands.append(VLLMBenchmarkCommand("serve", command, output_json, output_dir / "serve.log"))

    return commands


def run_vllm_benchmark_suite(
    config: VLLMBenchmarkConfig,
    *,
    run: bool = False,
    plot: bool = True,
) -> VLLMBenchmarkArtifacts:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "suite_config.json"
    commands_path = output_dir / "commands.json"
    status_path = output_dir / "run_status.json"
    summary_path = output_dir / "summary.json"

    commands = build_vllm_benchmark_commands(config)
    config_path.write_text(json.dumps(_config_to_json(config), indent=2, sort_keys=True) + "\n")
    commands_path.write_text(json.dumps([command.to_json() for command in commands], indent=2) + "\n")

    statuses: list[dict[str, object]] = []
    if run:
        for command in commands:
            statuses.append(_run_command(command, config))
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            if statuses[-1]["returncode"] != 0:
                raise subprocess.CalledProcessError(
                    int(statuses[-1]["returncode"]),
                    command.command,
                )
    else:
        statuses = [
            {
                "name": command.name,
                "returncode": None,
                "status": "planned",
                "output_json": str(command.output_json),
                "log_path": str(command.log_path),
            }
            for command in commands
        ]
        status_path.write_text(json.dumps(statuses, indent=2) + "\n")

    summary = collect_vllm_benchmark_summary(output_dir)
    summary["model"] = config.model
    summary["benchmarks"] = list(_normalize_benchmarks(config.benchmarks))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    plot_path: Path | None = None
    csv_path: Path | None = None
    if plot and _summary_rows(summary):
        plot_path = output_dir / "performance.html"
        csv_path = output_dir / "performance.csv"
        plot_vllm_benchmark_results(output_dir, output_html=plot_path, output_csv=csv_path)

    return VLLMBenchmarkArtifacts(
        output_dir=output_dir,
        config_path=config_path,
        commands_path=commands_path,
        status_path=status_path,
        summary_path=summary_path,
        plot_path=plot_path,
        csv_path=csv_path,
    )


def collect_vllm_benchmark_summary(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    summary: dict[str, Any] = {"results": {}}

    latency_path = output_path / "latency.json"
    if latency_path.exists():
        data = _load_json_result(latency_path)
        result: dict[str, Any] = {
            "source": str(latency_path),
            "avg_latency_s": _float_or_none(data.get("avg_latency")),
        }
        percentiles = data.get("percentiles", {})
        if isinstance(percentiles, dict):
            for percentile in ("50", "90", "99"):
                result[f"p{percentile}_latency_s"] = _float_or_none(
                    percentiles.get(percentile) or percentiles.get(int(percentile))
                )
        summary["results"]["latency"] = result

    throughput_path = output_path / "throughput.json"
    if throughput_path.exists():
        data = _load_json_result(throughput_path)
        summary["results"]["throughput"] = {
            "source": str(throughput_path),
            "elapsed_time_s": _float_or_none(data.get("elapsed_time")),
            "num_requests": _float_or_none(data.get("num_requests")),
            "total_num_tokens": _float_or_none(data.get("total_num_tokens")),
            "requests_per_second": _float_or_none(data.get("requests_per_second")),
            "tokens_per_second": _float_or_none(data.get("tokens_per_second")),
        }

    serve_path = output_path / "serve.json"
    if serve_path.exists():
        data = _load_json_result(serve_path)
        keep = (
            "duration",
            "completed",
            "failed",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p50_ttft_ms",
            "p90_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p50_tpot_ms",
            "p90_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p50_itl_ms",
            "p90_itl_ms",
            "p99_itl_ms",
            "mean_e2el_ms",
            "median_e2el_ms",
            "p50_e2el_ms",
            "p90_e2el_ms",
            "p99_e2el_ms",
        )
        result = {"source": str(serve_path)}
        for key in keep:
            if key in data:
                result[key] = _float_or_none(data[key])
        summary["results"]["serve"] = result

    return summary


def plot_vllm_benchmark_results(
    output_dir: str | Path,
    *,
    output_html: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    summary = collect_vllm_benchmark_summary(output_path)
    existing_summary_path = output_path / "summary.json"
    if existing_summary_path.exists():
        existing_summary = _load_json_result(existing_summary_path)
        for key in ("model", "benchmarks"):
            if key in existing_summary:
                summary[key] = existing_summary[key]
    rows = _summary_rows(summary)
    if not rows:
        raise FileNotFoundError(f"no vLLM benchmark JSON results found in {output_path}")

    html_path = Path(output_html) if output_html is not None else output_path / "performance.html"
    csv_path = Path(output_csv) if output_csv is not None else output_path / "performance.csv"
    html_path.write_text(_render_html(rows, summary), encoding="utf-8")
    _write_csv(rows, csv_path)
    return {"html": html_path, "csv": csv_path}


def _engine_args(config: VLLMBenchmarkConfig) -> tuple[str, ...]:
    args: list[str] = [
        "--model",
        config.model,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--dtype",
        config.dtype,
    ]
    if config.trust_remote_code:
        args.append("--trust-remote-code")
    if config.enforce_eager:
        args.append("--enforce-eager")
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])
    if config.gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    args.extend(config.engine_args)
    return tuple(args)


def _throughput_dataset_args(config: VLLMBenchmarkConfig) -> tuple[str, ...]:
    if config.dataset_name == "random":
        return (
            "--random-input-len",
            str(config.input_len),
            "--random-output-len",
            str(config.output_len),
        )
    return (
        "--input-len",
        str(config.input_len),
        "--output-len",
        str(config.output_len),
    )


def _run_command(command: VLLMBenchmarkCommand, config: VLLMBenchmarkConfig) -> dict[str, object]:
    env = os.environ.copy()
    if config.vllm_root is not None:
        vllm_root = str(Path(config.vllm_root))
        env["PYTHONPATH"] = vllm_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    start = time.perf_counter()
    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    with command.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shlex.join(command.command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command.command,
            env=env,
            cwd=str(Path(config.output_dir).resolve()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_s = time.perf_counter() - start
    return {
        "name": command.name,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "elapsed_s": elapsed_s,
        "output_json": str(command.output_json),
        "log_path": str(command.log_path),
    }


def _normalize_benchmarks(benchmarks: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for benchmark in benchmarks:
        lowered = benchmark.lower()
        if lowered not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"unsupported vLLM benchmark {benchmark!r}; expected one of {SUPPORTED_BENCHMARKS}")
        if lowered not in normalized:
            normalized.append(lowered)
    return tuple(normalized)


def _config_to_json(config: VLLMBenchmarkConfig) -> dict[str, object]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["vllm_root"] = None if config.vllm_root is None else str(config.vllm_root)
    data["benchmarks"] = list(config.benchmarks)
    data["engine_args"] = list(config.engine_args)
    data["throughput_args"] = list(config.throughput_args)
    data["serve_args"] = list(config.serve_args)
    return data


def _load_json_result(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = json.loads(text.splitlines()[-1])
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return loaded


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    results = summary.get("results", {})
    if not isinstance(results, dict):
        return []

    rows: list[dict[str, object]] = []
    metric_order = {
        "latency": (
            ("avg_latency_s", "avg latency", "s"),
            ("p50_latency_s", "p50 latency", "s"),
            ("p90_latency_s", "p90 latency", "s"),
            ("p99_latency_s", "p99 latency", "s"),
        ),
        "throughput": (
            ("requests_per_second", "requests", "req/s"),
            ("tokens_per_second", "tokens", "tok/s"),
        ),
        "serve": (
            ("request_throughput", "requests", "req/s"),
            ("output_throughput", "output tokens", "tok/s"),
            ("total_token_throughput", "total tokens", "tok/s"),
            ("mean_ttft_ms", "mean TTFT", "ms"),
            ("p99_ttft_ms", "p99 TTFT", "ms"),
            ("mean_tpot_ms", "mean TPOT", "ms"),
            ("p99_tpot_ms", "p99 TPOT", "ms"),
            ("mean_itl_ms", "mean ITL", "ms"),
            ("p99_itl_ms", "p99 ITL", "ms"),
        ),
    }
    for benchmark, metrics in metric_order.items():
        data = results.get(benchmark, {})
        if not isinstance(data, dict):
            continue
        for key, label, unit in metrics:
            value = data.get(key)
            if isinstance(value, int | float):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": key,
                        "label": label,
                        "value": float(value),
                        "unit": unit,
                    }
                )
    return rows


def _write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["benchmark", "metric", "label", "value", "unit"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_html(rows: list[dict[str, object]], summary: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["benchmark"]), []).append(row)

    sections = []
    for benchmark in SUPPORTED_BENCHMARKS:
        section_rows = grouped.get(benchmark, [])
        if not section_rows:
            continue
        max_value = max(float(row["value"]) for row in section_rows) or 1.0
        bars = []
        for row in section_rows:
            value = float(row["value"])
            width = max(1.0, value / max_value * 100.0)
            bars.append(
                "<tr>"
                f"<td>{html.escape(str(row['label']))}</td>"
                f"<td class=\"value\">{_format_float(value)} {html.escape(str(row['unit']))}</td>"
                "<td class=\"bar-cell\">"
                f"<div class=\"bar\" style=\"width:{width:.2f}%\"></div>"
                "</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{html.escape(benchmark.title())}</h2>"
            "<table><thead><tr><th>Metric</th><th>Value</th><th>Plot</th></tr></thead>"
            f"<tbody>{''.join(bars)}</tbody></table></section>"
        )

    model = html.escape(str(summary.get("model", "")))
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>vLLM Benchmark Performance</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:32px;color:#111827;background:#f8fafc}"
        "h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:24px 0 12px}"
        "p{margin:0 0 24px;color:#4b5563}"
        "section{max-width:980px;margin-bottom:20px}"
        "table{border-collapse:collapse;width:100%;background:white;border:1px solid #e5e7eb}"
        "th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:left;font-size:14px}"
        "th{background:#f3f4f6;color:#374151}.value{white-space:nowrap;font-variant-numeric:tabular-nums}"
        ".bar-cell{width:55%}.bar{height:14px;background:#2563eb;border-radius:3px}"
        "</style></head><body>"
        "<h1>vLLM Benchmark Performance</h1>"
        f"<p>{model}</p>"
        f"{''.join(sections)}"
        "</body></html>\n"
    )


def _format_float(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"
