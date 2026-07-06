from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


_LATENCY_METRICS = ("ttft_ms", "tpot_ms", "e2e_latency_ms")
_THROUGHPUT_METRICS = ("throughput_tps",)
_REQUEST_METRICS = (*_LATENCY_METRICS, *_THROUGHPUT_METRICS, "output_tokens")
_RAW_REQUEST_WAVE_SIZE = 64
_RAW_REQUEST_WAVES_PER_PROVIDER = 5
_PROVIDER_GAP_METRICS = (
    ("ttft_ms", "ttft_median_ms", "ttft_ms", False),
    ("tpot_ms", "tpot_median_ms", "tpot_ms", False),
    ("e2e_ms", "e2e_median_ms", "e2e_latency_ms", False),
    ("throughput_tps", "throughput_median_tps", "throughput_tps", True),
)
_BENCHMARK_QUEUE_PROFILE_KEYS = {
    "few_shot": (0.0, 256),
    "self_consistency": (0.7, 256),
    "multi_turn": (0.0, 512),
    "tree_of_thought": (0.7, 300),
    "long_output": (0.0, 96),
}
_VLLM_RUNTIME_RE = re.compile(
    r"Avg prompt throughput: (?P<prompt_tps>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation_tps>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, "
    r"Waiting: (?P<waiting>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<kv_cache_pct>[0-9.]+)%, "
    r"Prefix cache hit rate: (?P<prefix_hit_pct>[0-9.]+)%"
)
_SGLANG_PREFILL_RE = re.compile(
    r"Prefill batch, #new-seq: (?P<new_seq>[0-9]+), "
    r"#new-token: (?P<new_tokens>[0-9]+), "
    r"#cached-token: (?P<cached_tokens>[0-9]+).*?"
    r"cuda graph: (?P<cuda_graph>True|False), "
    r"input throughput \(token/s\): (?P<prompt_tps>[0-9.]+)"
)
_SGLANG_DECODE_RE = re.compile(
    r"Decode batch, #running-req: (?P<running>[0-9]+), "
    r"#token: (?P<tokens>[0-9]+).*?"
    r"cuda graph: (?P<cuda_graph>True|False), "
    r"gen throughput \(token/s\): (?P<generation_tps>[0-9.]+)"
)
_QUEUE_PROFILE_FIELDS = (
    "greedy_large_mixed_prefix_reuse",
    "fp8_prefill_enabled",
    "fp8_prefill_min_m",
    "marlin_int4_decode_enabled",
    "use_decode_many",
    "decode_many_async_readback",
    "decode_quantum",
    "drain_decode_quantum",
    "admit_per_step_cap",
    "admit_min_free_rows",
    "admit_min_ready_requests",
    "prefill_ready_before_decode",
    "prefill_ready_before_decode_active_cap",
    "request_queue_to_first_token_p50_ms",
    "request_queue_to_submit_p50_ms",
    "request_submit_to_first_token_p50_ms",
    "request_stream_prequeue_wait_count",
    "request_stream_prequeue_wait_p50_ms",
    "request_stream_prequeue_wait_configured_p50_ms",
    "request_stream_prequeue_wait_applied_count",
    "active_ready_wait_ms",
    "runtime_max_active_requests",
    "runtime_prefix_cache_capacity",
    "runtime_prefill_batches",
    "runtime_prefill_forward_ms",
    "runtime_prefill_wall_ms",
    "runtime_prefill_setup_ms",
    "runtime_prefill_copy_ms",
    "runtime_prefill_sample_ms",
    "runtime_prefill_state_ms",
    "runtime_prefill_packed_flashinfer_calls",
    "runtime_prefill_packed_flashinfer_ms",
    "runtime_prefill_packed_flashinfer_saved_tokens",
    "runtime_prefill_packed_flashinfer_tokens",
    "runtime_prefill_packed_flashinfer_model_tokens",
    "runtime_prefill_packed_eager_calls",
    "runtime_prefill_packed_eager_ms",
    "runtime_prefill_packed_eager_saved_tokens",
    "runtime_prefill_packed_eager_tokens",
    "runtime_prefill_packed_eager_model_tokens",
    "runtime_prefill_packed_eager_shape_counts",
    "runtime_prefill_packed_eager_shape_tokens",
    "runtime_prefill_packed_eager_shape_model_tokens",
    "runtime_prefill_packed_eager_shape_saved_tokens",
    "runtime_prefill_packed_eager_shape_ms",
    "runtime_prefill_packed_candidate_calls",
    "runtime_prefill_packed_candidate_tokens",
    "runtime_prefill_packed_candidate_model_tokens",
    "runtime_prefill_packed_candidate_saved_tokens",
    "runtime_prefill_packed_candidate_groups",
    "runtime_prefill_packed_candidate_shape_tokens",
    "runtime_prefill_packed_candidate_shape_model_tokens",
    "runtime_prefill_packed_candidate_shape_saved_tokens",
    "runtime_prefill_packed_candidate_shape_groups",
    "runtime_prefill_packed_candidate_signature_keys",
    "runtime_prefill_packed_candidate_signature_calls",
    "runtime_prefill_packed_candidate_signature_repeated_keys",
    "runtime_prefill_packed_candidate_signature_repeated_calls",
    "runtime_prefill_packed_candidate_signature_repeated_saved_tokens",
    "runtime_prefill_packed_candidate_signature_counts",
    "runtime_prefill_packed_candidate_signature_tokens",
    "runtime_prefill_packed_candidate_signature_model_tokens",
    "runtime_prefill_packed_candidate_signature_saved_tokens",
    "runtime_prefill_packed_candidate_signature_groups",
    "runtime_prefill_packed_candidate_pattern_keys",
    "runtime_prefill_packed_candidate_pattern_calls",
    "runtime_prefill_packed_candidate_pattern_repeated_keys",
    "runtime_prefill_packed_candidate_pattern_repeated_calls",
    "runtime_prefill_packed_candidate_pattern_repeated_saved_tokens",
    "runtime_prefill_packed_candidate_pattern_counts",
    "runtime_prefill_packed_candidate_pattern_tokens",
    "runtime_prefill_packed_candidate_pattern_model_tokens",
    "runtime_prefill_packed_candidate_pattern_saved_tokens",
    "runtime_prefill_packed_candidate_pattern_groups",
    "runtime_prefill_packed_candidate_pattern_slot_counts",
    "runtime_prefill_suffix_split_candidate_calls",
    "runtime_prefill_suffix_split_accepted_calls",
    "runtime_prefill_suffix_split_rejected_calls",
    "runtime_prefill_suffix_split_base_model_tokens",
    "runtime_prefill_suffix_split_candidate_model_tokens",
    "runtime_prefill_suffix_split_candidate_saved_tokens",
    "runtime_prefill_suffix_split_accepted_base_model_tokens",
    "runtime_prefill_suffix_split_accepted_model_tokens",
    "runtime_prefill_suffix_split_accepted_saved_tokens",
    "runtime_prefill_suffix_split_accepted_fragments",
    "runtime_prefill_suffix_split_reject_reason_counts",
    "runtime_prefill_suffix_split_candidate_shape_counts",
    "runtime_prefill_suffix_split_candidate_shape_saved_tokens",
    "runtime_prefill_suffix_split_accepted_shape_counts",
    "runtime_prefill_suffix_split_accepted_shape_saved_tokens",
    "runtime_prefill_suffix_split_accepted_fragment_counts",
    "runtime_prefill_graph_hits",
    "runtime_prefill_graph_captures",
    "runtime_prefill_graph_capture_ms",
    "runtime_prefill_graph_capture_gpu_ms",
    "runtime_prefill_graph_replays",
    "runtime_prefill_graph_replay_ms",
    "runtime_prefill_graph_replay_gpu_ms",
    "runtime_prefill_graph_misses",
    "runtime_prefill_graph_miss_shape_counts",
    "runtime_prefill_graph_cache_live_entries",
    "runtime_prefill_shape_counts",
    "runtime_prefill_shape_forward_ms",
    "runtime_prefill_shape_wall_ms",
    "runtime_prefill_shape_setup_ms",
    "runtime_prefill_shape_copy_ms",
    "runtime_prefill_shape_sample_ms",
    "runtime_prefill_shape_state_ms",
    "runtime_prefill_shape_active_requests",
    "runtime_prefill_shape_model_rows",
    "runtime_prefill_shape_active_tokens",
    "runtime_prefill_shape_model_tokens",
    "runtime_prefill_shape_graph_capture_counts",
    "runtime_prefill_shape_graph_capture_ms",
    "runtime_prefill_shape_graph_capture_gpu_ms",
    "runtime_prefill_shape_graph_replay_counts",
    "runtime_prefill_shape_graph_replay_ms",
    "runtime_prefill_shape_graph_replay_gpu_ms",
    "runtime_prefill_graph_capture_shape_gpu_ms",
    "runtime_prefill_graph_replay_shape_gpu_ms",
    "runtime_prefill_shape_padding_tokens",
    "runtime_prefill_shape_row_padding_tokens",
    "runtime_prefill_shape_suffix_padding_tokens",
    "runtime_prefill_shape_route_counts",
    "runtime_prefill_shape_route_reuse_tokens",
    "runtime_decode_ragged_cpu_tokens_ms",
    "runtime_decode_ragged_state_update_ms",
    "runtime_decode_ragged_model_gpu_ms",
    "runtime_decode_graph_hits",
    "runtime_decode_graph_captures",
    "runtime_decode_graph_capture_ms",
    "runtime_decode_graph_capture_shape_ms",
    "runtime_decode_graph_misses",
    "runtime_decode_graph_miss_shape_counts",
    "runtime_decode_graph_replays",
    "runtime_decode_graph_replay_ms",
    "runtime_decode_graph_replay_shape_ms",
    "runtime_decode_graph_cache_live_entries",
    "runtime_decode_graph_cache_live_shape_counts",
    "runtime_decode_graph_cache_live_cache_bucket_counts",
    "runtime_decode_graph_cache_live_symm_counts",
    "runtime_decode_shape_gpu_ms",
    "runtime_decode_shape_cpu_tokens_ms",
    "runtime_decode_many_graph_calls",
    "runtime_decode_many_graph_steps",
    "runtime_decode_many_graph_model_tokens",
    "runtime_decode_many_graph_ms",
    "runtime_decode_many_graph_shape_counts",
    "runtime_decode_many_graph_shape_steps",
    "runtime_decode_many_graph_shape_model_tokens",
    "runtime_decode_many_graph_shape_ms",
    "decode_many_stop_tail_max_steps",
    "decode_many_min_active_pct",
    "decode_many_sync_stops",
    "runtime_decode_many_calls",
    "runtime_decode_many_steps",
    "runtime_decode_many_model_tokens",
    "runtime_decode_many_padded_tokens",
    "runtime_decode_many_emitted_tokens",
    "runtime_decode_many_skipped_tokens",
    "runtime_decode_many_cpu_tokens_ms",
    "runtime_decode_many_model_gpu_ms",
    "runtime_decode_many_tail_limited_calls",
    "runtime_decode_many_tail_limited_steps",
    "runtime_decode_many_min_active_skips",
    "runtime_decode_many_padding_tokens",
    "runtime_decode_many_overgenerated_tokens",
    "runtime_decode_many_shape_model_tokens",
    "runtime_decode_many_shape_padded_tokens",
    "runtime_decode_many_shape_emitted_tokens",
    "runtime_decode_many_shape_skipped_tokens",
    "runtime_decode_many_shape_stop_finishes",
    "runtime_decode_many_shape_limit_finishes",
    "runtime_decode_many_shape_gpu_ms",
    "runtime_decode_many_shape_padding_tokens",
    "runtime_decode_many_shape_overgenerated_tokens",
    "runtime_decode_many_step_window_counts",
    "runtime_decode_many_step_window_model_tokens",
    "runtime_decode_many_step_window_padded_tokens",
    "runtime_decode_many_step_window_emitted_tokens",
    "runtime_decode_many_step_window_skipped_tokens",
    "runtime_decode_many_step_window_model_ms",
    "runtime_decode_many_step_window_cpu_tokens_ms",
    "runtime_generated_prefix_store_requests",
    "runtime_generated_prefix_reuse_requests",
    "runtime_generated_prefix_reuse_tokens",
)


@dataclass(frozen=True)
class PercentileSummary:
    p50: float
    p90: float
    p99: float
    minimum: float
    maximum: float
    count: int


@dataclass(frozen=True)
class ProviderBenchmarkSummary:
    provider: str
    benchmark: str
    metrics: dict[str, float]
    request_percentiles: dict[str, PercentileSummary]
    request_waves: tuple["RawRequestWaveSummary", ...] = ()


@dataclass(frozen=True)
class RawRequestWaveSummary:
    wave_index: int
    request_start: int
    request_end: int
    count: int
    request_percentiles: dict[str, PercentileSummary]


@dataclass(frozen=True)
class QueueProfileSummary:
    event: str
    temperature: float | None
    max_tokens: int | None
    submitted_requests: int | None
    finished_events: int | None
    fields: dict[str, Any] = field(default_factory=dict)
    segments: int = 1


@dataclass(frozen=True)
class ProviderServerLogSummary:
    provider: str
    prompt_events: int = 0
    prompt_tps_avg: float | None = None
    prompt_tps_max: float | None = None
    generation_events: int = 0
    generation_tps_avg: float | None = None
    generation_tps_max: float | None = None
    prefix_hit_pct_avg: float | None = None
    prefix_hit_pct_max: float | None = None
    prefill_batches: int = 0
    prefill_new_tokens: int = 0
    prefill_cached_tokens: int = 0
    prefill_cuda_graph_batches: int = 0
    prefill_new_seq_max: int | None = None
    decode_batches: int = 0
    decode_logged_tokens: int = 0
    decode_cuda_graph_batches: int = 0
    decode_running_max: int | None = None


@dataclass(frozen=True)
class InferenceBenchRunSummary:
    run_dir: Path
    model: str
    tensor_parallel_size: int
    hardware: str
    providers: tuple[str, ...]
    benchmarks: tuple[str, ...]
    provider_benchmarks: tuple[ProviderBenchmarkSummary, ...]
    torchinferno_queue_profiles: tuple[QueueProfileSummary, ...]
    provider_server_logs: tuple[ProviderServerLogSummary, ...]


def summarize_inference_bench_run(
    run_dir: str | Path,
    *,
    benchmarks: Sequence[str] | None = None,
    providers: Sequence[str] | None = None,
) -> InferenceBenchRunSummary:
    """Summarize an inference-bench run directory.

    The public inference-bench artifact stores comparable provider metrics in
    results.json and TorchInferno's internal phase counters in provider_logs.
    This helper joins those two views without requiring the benchmark harness.
    """

    root = Path(run_dir)
    data_path = root / "results.json"
    data = json.loads(data_path.read_text())
    provider_map = data.get("providers", {})
    selected_providers = tuple(providers or provider_map.keys())
    selected_benchmarks = _select_benchmarks(provider_map, benchmarks)

    provider_benchmarks: list[ProviderBenchmarkSummary] = []
    for provider in selected_providers:
        provider_data = provider_map.get(provider, {})
        benchmark_map = provider_data.get("benchmarks", {})
        for benchmark in selected_benchmarks:
            benchmark_data = benchmark_map.get(benchmark)
            if not isinstance(benchmark_data, dict):
                continue
            raw_requests = benchmark_data.get("raw_requests", [])
            provider_benchmarks.append(
                ProviderBenchmarkSummary(
                    provider=provider,
                    benchmark=benchmark,
                    metrics=_numeric_dict(benchmark_data.get("metrics", {})),
                    request_percentiles=_summarize_raw_requests(raw_requests),
                    request_waves=_summarize_raw_request_waves(raw_requests),
                )
            )

    return InferenceBenchRunSummary(
        run_dir=root,
        model=str(data.get("model", "")),
        tensor_parallel_size=int(data.get("tensor_parallel_size") or 0),
        hardware=str(data.get("hardware", "")),
        providers=selected_providers,
        benchmarks=selected_benchmarks,
        provider_benchmarks=tuple(provider_benchmarks),
        torchinferno_queue_profiles=_summarize_torchinferno_queue(root),
        provider_server_logs=_summarize_provider_server_logs(root),
    )


def format_inference_bench_summary(summary: InferenceBenchRunSummary) -> str:
    lines = [
        "TorchInferno inference-bench summary",
        f"run={summary.run_dir}",
        f"model={summary.model} tp={summary.tensor_parallel_size} hardware={summary.hardware}",
        "",
    ]
    for benchmark in summary.benchmarks:
        rows = [row for row in summary.provider_benchmarks if row.benchmark == benchmark]
        if not rows:
            continue
        lines.append(f"[{benchmark}]")
        header = (
            "provider",
            "score_ttft",
            "score_tpot",
            "score_e2e",
            "score_tps",
            "correct",
            "raw_ttft_p90",
            "out_p50",
        )
        body: list[tuple[str, ...]] = []
        for row in rows:
            body.append(
                (
                    row.provider,
                    _fmt_metric(row, "ttft_median_ms", fallback_metric="ttft_ms"),
                    _fmt_metric(row, "tpot_median_ms", fallback_metric="tpot_ms"),
                    _fmt_metric(row, "e2e_median_ms", fallback_metric="e2e_latency_ms"),
                    _fmt_metric(
                        row,
                        "throughput_median_tps",
                        fallback_metric="throughput_tps",
                    ),
                    _fmt_rate(row.metrics.get("correctness_rate")),
                    _fmt_percentile(row, "ttft_ms", "p90"),
                    _fmt_percentile(row, "output_tokens", "p50"),
                )
            )
        lines.extend(_format_table(header, body))
        gap_rows = _provider_gap_rows(rows)
        if gap_rows:
            lines.append("")
            lines.append(f"[{benchmark} provider gaps vs torchinferno]")
            lines.extend(
                _format_table(
                    (
                        "metric",
                        "torchinferno",
                        "best_other",
                        "best_provider",
                        "gap",
                        "ratio",
                    ),
                    gap_rows,
                )
            )
        wave_rows = _raw_request_wave_rows(rows)
        if wave_rows:
            lines.append("")
            lines.append(f"[{benchmark} raw request waves]")
            lines.extend(
                _format_table(
                    (
                        "provider",
                        "wave",
                        "requests",
                        "n",
                        "ttft_p50",
                        "ttft_p90",
                        "tpot_p50",
                        "e2e_p50",
                        "out_p50",
                    ),
                    wave_rows,
                )
            )
        lines.append("")

    queue_profiles = _queue_profiles_for_selected_benchmarks(summary)
    if queue_profiles:
        score_target_rows = _torchinferno_score_target_rows(summary)
        if score_target_rows:
            lines.append("[torchinferno score targets]")
            lines.extend(
                _format_table(
                    (
                        "benchmark",
                        "ttft_gap",
                        "tpot_gap",
                        "e2e_gap",
                        "phase_target",
                        "q2first",
                        "q2submit",
                        "submit2first",
                        "prefill_ms",
                        "prefill_sample_ms",
                        "prefill_pad",
                        "prefill_row_pad",
                        "prefill_sfx_pad",
                        "prefill_pad_pct",
                        "decode_ms",
                        "decode_many_ms",
                        "decode_cpu_ms",
                        "decode_many_cpu_ms",
                        "decode_many_calls",
                        "prefill_miss",
                        "prefill_miss_kind",
                        "decode_miss",
                        "decode_miss_kind",
                        "gen_store",
                        "gen_reuse",
                        "packed_saved",
                        "hot_prefill",
                        "hot_decode",
                    ),
                    score_target_rows,
                )
            )
            lines.append("")

        lines.append("[torchinferno queue profiles]")
        header = (
            "temp",
            "max_tokens",
            "max_active",
            "prefix_cap",
            "mixed_prefix",
            "fp8_prefill",
            "fp8_min_m",
            "marlin_decode",
            "decode_many",
            "many_async",
            "decode_q",
            "drain_q",
            "admit_cap",
            "min_free",
            "min_ready",
            "prefill_ready",
            "ready_cap",
            "submitted",
            "coverage",
            "q2first_p50",
            "q2submit_p50",
            "submit2first_p50",
            "preq_cfg_p50",
            "preq_p50",
            "preq_applied",
            "active_wait",
            "prefill_batches",
            "prefill_forward_ms",
            "prefill_wall_ms",
            "prefill_setup_ms",
            "prefill_copy_ms",
            "prefill_sample_ms",
            "prefill_state_ms",
            "prefill_pad",
            "prefill_row_pad",
            "prefill_sfx_pad",
            "prefill_pad_pct",
            "suffix_split_ok",
            "suffix_split_rej",
            "suffix_split_saved",
            "suffix_split_frags",
            "packed_fi_calls",
            "packed_fi_ms",
            "packed_fi_saved",
            "packed_eager_calls",
            "packed_eager_ms",
            "packed_eager_saved",
            "packed_cand_calls",
            "packed_cand_saved",
            "packed_cand_groups",
            "gen_store",
            "gen_reuse",
            "gen_tokens",
            "prefill_graph_miss",
            "prefill_miss_kind",
            "prefill_graph_cap_ms",
            "prefill_graph_cap_gpu_ms",
            "prefill_graph_replay_ms",
            "prefill_graph_replay_gpu_ms",
            "decode_gpu_ms",
            "decode_cpu_ms",
            "decode_state_ms",
            "decode_graph_miss",
            "decode_miss_kind",
            "decode_graph_cap_ms",
            "decode_graph_replay_ms",
            "decode_graph_cache",
            "decode_replay_cache_ms",
            "decode_graph_symm",
            "decode_replay_symm_ms",
            "decode_many_calls",
            "decode_many_gpu_ms",
            "decode_many_cpu_ms",
            "decode_many_steps",
            "decode_many_graph_calls",
            "decode_many_graph_steps",
            "tail_cap",
            "min_active_pct",
            "sync_stops",
            "tail_calls",
            "tail_steps",
            "min_active_skips",
            "overgen",
        )
        body = []
        for profile in queue_profiles:
            fields = profile.fields
            expected_requests = _expected_requests_for_queue_profile(summary, profile)
            _prefill_active_tokens, prefill_model_tokens, prefill_padding_tokens = (
                _prefill_token_totals(fields)
            )
            prefill_row_padding_tokens, prefill_suffix_padding_tokens = (
                _prefill_padding_split_totals(fields)
            )
            body.append(
                (
                    _fmt_value(profile.temperature),
                    _fmt_value(profile.max_tokens),
                    _fmt_value(fields.get("runtime_max_active_requests")),
                    _fmt_value(fields.get("runtime_prefix_cache_capacity")),
                    _fmt_value(fields.get("greedy_large_mixed_prefix_reuse")),
                    _fmt_value(fields.get("fp8_prefill_enabled")),
                    _fmt_value(fields.get("fp8_prefill_min_m")),
                    _fmt_value(fields.get("marlin_int4_decode_enabled")),
                    _fmt_value(fields.get("use_decode_many")),
                    _fmt_value(fields.get("decode_many_async_readback")),
                    _fmt_value(fields.get("decode_quantum")),
                    _fmt_value(fields.get("drain_decode_quantum")),
                    _fmt_value(fields.get("admit_per_step_cap")),
                    _fmt_value(fields.get("admit_min_free_rows")),
                    _fmt_value(fields.get("admit_min_ready_requests")),
                    _fmt_value(fields.get("prefill_ready_before_decode")),
                    _fmt_value(fields.get("prefill_ready_before_decode_active_cap")),
                    _fmt_value(profile.submitted_requests),
                    _fmt_queue_profile_coverage(profile, expected_requests),
                    _fmt_value(fields.get("request_queue_to_first_token_p50_ms")),
                    _fmt_value(fields.get("request_queue_to_submit_p50_ms")),
                    _fmt_value(fields.get("request_submit_to_first_token_p50_ms")),
                    _fmt_value(fields.get("request_stream_prequeue_wait_configured_p50_ms")),
                    _fmt_value(fields.get("request_stream_prequeue_wait_p50_ms")),
                    _fmt_value(fields.get("request_stream_prequeue_wait_applied_count")),
                    _fmt_value(fields.get("active_ready_wait_ms")),
                    _fmt_value(fields.get("runtime_prefill_batches")),
                    _fmt_value(fields.get("runtime_prefill_forward_ms")),
                    _fmt_value(fields.get("runtime_prefill_wall_ms")),
                    _fmt_value(fields.get("runtime_prefill_setup_ms")),
                    _fmt_value(fields.get("runtime_prefill_copy_ms")),
                    _fmt_value(fields.get("runtime_prefill_sample_ms")),
                    _fmt_value(fields.get("runtime_prefill_state_ms")),
                    _fmt_value(
                        None
                        if prefill_padding_tokens is None
                        else _int_if_whole(prefill_padding_tokens)
                    ),
                    _fmt_value(
                        None
                        if prefill_row_padding_tokens is None
                        else _int_if_whole(prefill_row_padding_tokens)
                    ),
                    _fmt_value(
                        None
                        if prefill_suffix_padding_tokens is None
                        else _int_if_whole(prefill_suffix_padding_tokens)
                    ),
                    _fmt_pct(prefill_padding_tokens or 0.0, prefill_model_tokens or 0.0),
                    _fmt_value(fields.get("runtime_prefill_suffix_split_accepted_calls")),
                    _fmt_value(fields.get("runtime_prefill_suffix_split_rejected_calls")),
                    _fmt_value(fields.get("runtime_prefill_suffix_split_accepted_saved_tokens")),
                    _fmt_value(fields.get("runtime_prefill_suffix_split_accepted_fragments")),
                    _fmt_value(fields.get("runtime_prefill_packed_flashinfer_calls")),
                    _fmt_value(fields.get("runtime_prefill_packed_flashinfer_ms")),
                    _fmt_value(fields.get("runtime_prefill_packed_flashinfer_saved_tokens")),
                    _fmt_value(fields.get("runtime_prefill_packed_eager_calls")),
                    _fmt_value(fields.get("runtime_prefill_packed_eager_ms")),
                    _fmt_value(fields.get("runtime_prefill_packed_eager_saved_tokens")),
                    _fmt_value(fields.get("runtime_prefill_packed_candidate_calls")),
                    _fmt_value(fields.get("runtime_prefill_packed_candidate_saved_tokens")),
                    _fmt_value(fields.get("runtime_prefill_packed_candidate_groups")),
                    _fmt_value(fields.get("runtime_generated_prefix_store_requests")),
                    _fmt_value(fields.get("runtime_generated_prefix_reuse_requests")),
                    _fmt_value(fields.get("runtime_generated_prefix_reuse_tokens")),
                    _fmt_value(fields.get("runtime_prefill_graph_misses")),
                    _fmt_mapping_summary(_prefill_graph_miss_kind_counts(fields)),
                    _fmt_value(fields.get("runtime_prefill_graph_capture_ms")),
                    _fmt_value(fields.get("runtime_prefill_graph_capture_gpu_ms")),
                    _fmt_value(fields.get("runtime_prefill_graph_replay_ms")),
                    _fmt_value(fields.get("runtime_prefill_graph_replay_gpu_ms")),
                    _fmt_value(fields.get("runtime_decode_ragged_model_gpu_ms")),
                    _fmt_value(fields.get("runtime_decode_ragged_cpu_tokens_ms")),
                    _fmt_value(fields.get("runtime_decode_ragged_state_update_ms")),
                    _fmt_value(fields.get("runtime_decode_graph_misses")),
                    _fmt_mapping_summary(_decode_graph_miss_kind_counts(fields)),
                    _fmt_value(fields.get("runtime_decode_graph_capture_ms")),
                    _fmt_value(fields.get("runtime_decode_graph_replay_ms")),
                    _fmt_mapping_summary(_decode_graph_cache_counts(fields)),
                    _fmt_mapping_summary(
                        _decode_graph_cache_value_totals(
                            fields.get("runtime_decode_graph_replay_shape_ms")
                        )
                    ),
                    _fmt_mapping_summary(_decode_graph_symm_counts(fields)),
                    _fmt_mapping_summary(
                        _decode_graph_symm_value_totals(
                            fields.get("runtime_decode_graph_replay_shape_ms")
                        )
                    ),
                    _fmt_value(fields.get("runtime_decode_many_calls")),
                    _fmt_value(fields.get("runtime_decode_many_model_gpu_ms")),
                    _fmt_value(fields.get("runtime_decode_many_cpu_tokens_ms")),
                    _fmt_value(fields.get("runtime_decode_many_steps")),
                    _fmt_value(fields.get("runtime_decode_many_graph_calls")),
                    _fmt_value(fields.get("runtime_decode_many_graph_steps")),
                    _fmt_value(fields.get("decode_many_stop_tail_max_steps")),
                    _fmt_value(fields.get("decode_many_min_active_pct")),
                    _fmt_value(fields.get("decode_many_sync_stops")),
                    _fmt_value(fields.get("runtime_decode_many_tail_limited_calls")),
                    _fmt_value(fields.get("runtime_decode_many_tail_limited_steps")),
                    _fmt_value(fields.get("runtime_decode_many_min_active_skips")),
                    _fmt_value(fields.get("runtime_decode_many_overgenerated_tokens")),
                )
            )
        lines.extend(_format_table(header, body))
        lines.append("")

        prefill_miss_shape_rows = _graph_miss_shape_rows(
            queue_profiles,
            "runtime_prefill_graph_miss_shape_counts",
            _prefill_graph_miss_kind,
        )
        if prefill_miss_shape_rows:
            lines.append("[torchinferno prefill graph miss shapes]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "kind",
                        "shape",
                        "misses",
                    ),
                    prefill_miss_shape_rows,
                )
            )
            lines.append("")

        decode_miss_shape_rows = _graph_miss_shape_rows(
            queue_profiles,
            "runtime_decode_graph_miss_shape_counts",
            _decode_graph_miss_kind,
        )
        if decode_miss_shape_rows:
            lines.append("[torchinferno decode graph miss shapes]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "kind",
                        "shape",
                        "misses",
                    ),
                    decode_miss_shape_rows,
                )
            )
            lines.append("")

        prefill_rows = _hot_prefill_shape_rows(queue_profiles)
        if prefill_rows:
            lines.append("[torchinferno hot prefill shapes]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "shape",
                        "calls",
                        "forward_ms",
                        "graph_gpu_ms",
                        "gpu_ms_call",
                        "gpu_us_tok",
                        "wall_ms",
                        "setup_ms",
                        "copy_ms",
                        "sample_ms",
                        "state_ms",
                        "active_tokens",
                        "model_tokens",
                        "padding_tokens",
                        "pad_pct",
                        "pad_call",
                        "row_pad",
                        "suffix_pad",
                        "graphs",
                    ),
                    prefill_rows,
                )
            )
            lines.append("")

        packed_candidate_rows = _hot_prefill_packed_candidate_rows(queue_profiles)
        if packed_candidate_rows:
            lines.append("[torchinferno packed prefill candidates]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "candidate_shape",
                        "real_tokens",
                        "model_tokens",
                        "saved_tokens",
                        "groups",
                    ),
                    packed_candidate_rows,
                )
            )
            lines.append("")

        packed_per_batch_rows = _prefill_packed_per_batch_target_rows(
            queue_profiles
        )
        if packed_per_batch_rows:
            lines.append("[torchinferno packed prefill per-batch targets]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "candidate_shape",
                        "calls",
                        "real_tokens",
                        "model_tokens",
                        "saved_tokens",
                        "row_saved",
                        "suffix_saved",
                        "saved_pct",
                        "est_saved_ms",
                        "obs_packed_ms",
                        "groups",
                    ),
                    packed_per_batch_rows,
                )
            )
            lines.append("")

        packed_signature_rows = _hot_prefill_packed_signature_rows(queue_profiles)
        if packed_signature_rows:
            lines.append("[torchinferno packed prefill signatures]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "signature",
                        "calls",
                        "real_tokens",
                        "model_tokens",
                        "saved_tokens",
                        "groups",
                    ),
                    packed_signature_rows,
                )
            )
            lines.append("")

        packed_pattern_rows = _hot_prefill_packed_pattern_rows(queue_profiles)
        if packed_pattern_rows:
            lines.append("[torchinferno packed prefill patterns]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "pattern",
                        "calls",
                        "real_tokens",
                        "model_tokens",
                        "saved_tokens",
                        "saved_pct",
                        "groups",
                    ),
                    packed_pattern_rows,
                )
            )
            lines.append("")

        packed_fixed_capacity_rows = _prefill_packed_fixed_capacity_plan_rows(
            queue_profiles
        )
        if packed_fixed_capacity_rows:
            lines.append("[torchinferno packed prefill fixed-capacity plans]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "pattern",
                        "calls",
                        "slot_src",
                        "sig_calls",
                        "sig_cov",
                        "slots",
                        "dense_tokens",
                        "fixed_tokens",
                        "fixed_saved",
                        "est_saved_ms",
                        "obs_packed_ms",
                        "fixed_saved_pct",
                    ),
                    packed_fixed_capacity_rows,
                )
            )
            lines.append("")

        packed_target_rows = _prefill_packed_implementation_target_rows(
            queue_profiles
        )
        if packed_target_rows:
            lines.append("[torchinferno packed prefill implementation targets]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "pattern",
                        "calls",
                        "repeat_saved",
                        "fixed_saved",
                        "est_saved_ms",
                        "obs_packed_ms",
                        "fixed_saved_pct",
                        "sig_cov",
                    ),
                    packed_target_rows,
                )
            )
            lines.append("")

        packed_signature_reuse_rows = _prefill_packed_signature_reuse_rows(
            queue_profiles
        )
        if packed_signature_reuse_rows:
            lines.append("[torchinferno packed prefill signature reuse]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "signature_keys",
                        "signature_calls",
                        "candidate_calls",
                        "repeat_calls",
                        "repeat_call_pct",
                        "repeat_saved",
                        "repeat_saved_pct",
                    ),
                    packed_signature_reuse_rows,
                )
            )
            lines.append("")

        packed_pattern_reuse_rows = _prefill_packed_key_reuse_rows(
            queue_profiles,
            kind="pattern",
        )
        if packed_pattern_reuse_rows:
            lines.append("[torchinferno packed prefill pattern reuse]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "pattern_keys",
                        "pattern_calls",
                        "candidate_calls",
                        "repeat_calls",
                        "repeat_call_pct",
                        "repeat_saved",
                        "repeat_saved_pct",
                    ),
                    packed_pattern_reuse_rows,
                )
            )
            lines.append("")

        prefill_graph_rows = _hot_prefill_graph_shape_rows(queue_profiles)
        if prefill_graph_rows:
            lines.append("[torchinferno hot prefill graph shapes]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "capture_shape",
                        "capture_ms",
                        "capture_gpu_ms",
                        "replay_shape",
                        "replay_ms",
                        "replay_gpu_ms",
                        "graphs",
                    ),
                    prefill_graph_rows,
                )
            )
            lines.append("")

        decode_rows = _hot_decode_shape_rows(queue_profiles)
        if decode_rows:
            lines.append("[torchinferno hot decode shapes]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "decode_gpu_shape",
                        "decode_gpu_ms",
                        "decode_cpu_ms",
                        "decode_many_shape",
                        "decode_many_gpu_ms",
                        "model_tokens",
                        "emitted",
                        "skipped",
                        "skip_pct",
                        "overgen_tokens",
                        "graphs",
                    ),
                    decode_rows,
                )
            )
            lines.append("")

        decode_window_rows = _decode_many_step_window_rows(queue_profiles)
        if decode_window_rows:
            lines.append("[torchinferno decode-many step windows]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "window",
                        "calls",
                        "model_tokens",
                        "padded_tokens",
                        "emitted",
                        "skipped",
                        "skip_pct",
                        "model_ms",
                        "cpu_ms",
                    ),
                    decode_window_rows,
                )
            )
            lines.append("")

        decode_window_target_rows = _decode_many_step_window_target_rows(
            queue_profiles
        )
        if decode_window_target_rows:
            lines.append("[torchinferno decode-many implementation targets]")
            lines.extend(
                _format_table(
                    (
                        "temp",
                        "max_tokens",
                        "window",
                        "calls",
                        "model_tokens",
                        "emitted",
                        "skipped",
                        "skip_pct",
                        "gpu_ms",
                        "gpu_src",
                        "cpu_ms",
                        "total_ms",
                        "us_tok",
                    ),
                    decode_window_target_rows,
                )
            )
            lines.append("")

    provider_phase_rows = _provider_server_log_rows(summary.provider_server_logs)
    if provider_phase_rows:
        lines.append("[provider server log phases]")
        lines.extend(
            _format_table(
                (
                    "provider",
                    "prompt_events",
                    "prompt_tps_avg",
                    "prompt_tps_max",
                    "gen_events",
                    "gen_tps_avg",
                    "gen_tps_max",
                    "prefix_hit_avg",
                    "prefix_hit_max",
                    "prefill_batches",
                    "prefill_tokens",
                    "cached_tokens",
                    "prefill_graph_pct",
                    "prefill_new_seq_max",
                    "decode_batches",
                    "decode_tokens",
                    "decode_graph_pct",
                    "decode_running_max",
                ),
                provider_phase_rows,
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _select_benchmarks(
    provider_map: dict[str, Any],
    benchmarks: Sequence[str] | None,
) -> tuple[str, ...]:
    if benchmarks is not None:
        return tuple(benchmarks)
    seen: list[str] = []
    for provider_data in provider_map.values():
        for benchmark in provider_data.get("benchmarks", {}).keys():
            if benchmark not in seen:
                seen.append(benchmark)
    return tuple(seen)


def _numeric_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def _summarize_raw_requests(raw_requests: Any) -> dict[str, PercentileSummary]:
    if not isinstance(raw_requests, list):
        return {}
    summaries: dict[str, PercentileSummary] = {}
    for metric in _REQUEST_METRICS:
        values = [
            float(request[metric])
            for request in raw_requests
            if isinstance(request, dict) and isinstance(request.get(metric), (int, float))
        ]
        if values:
            summaries[metric] = _percentiles(values)
    return summaries


def _summarize_raw_request_waves(raw_requests: Any) -> tuple[RawRequestWaveSummary, ...]:
    if not isinstance(raw_requests, list):
        return ()
    by_wave: dict[int, list[dict[str, Any]]] = {}
    for completion_index, request in enumerate(raw_requests):
        if not isinstance(request, dict):
            continue
        request_index = _raw_request_index(request, fallback=completion_index)
        wave_index = max(0, request_index) // _RAW_REQUEST_WAVE_SIZE
        by_wave.setdefault(wave_index, []).append(request)
    waves: list[RawRequestWaveSummary] = []
    for wave_index, requests in sorted(by_wave.items()):
        waves.append(
            RawRequestWaveSummary(
                wave_index=wave_index,
                request_start=wave_index * _RAW_REQUEST_WAVE_SIZE,
                request_end=(wave_index + 1) * _RAW_REQUEST_WAVE_SIZE - 1,
                count=len(requests),
                request_percentiles=_summarize_raw_requests(requests),
            )
        )
    return tuple(waves)


def _raw_request_index(request: dict[str, Any], *, fallback: int) -> int:
    for key in ("request_idx", "metadata_request_idx"):
        value = request.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    metadata = request.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("request_idx")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return int(fallback)


def _raw_request_wave_rows(rows: Sequence[ProviderBenchmarkSummary]) -> list[tuple[str, ...]]:
    body: list[tuple[str, ...]] = []
    for row in rows:
        waves = [
            wave
            for wave in row.request_waves
            if "ttft_ms" in wave.request_percentiles
        ]
        if not waves:
            continue
        selected = sorted(
            waves,
            key=lambda wave: wave.request_percentiles["ttft_ms"].p50,
            reverse=True,
        )[:_RAW_REQUEST_WAVES_PER_PROVIDER]
        for wave in sorted(selected, key=lambda item: item.wave_index):
            body.append(
                (
                    row.provider,
                    str(wave.wave_index),
                    f"{wave.request_start}-{wave.request_end}",
                    str(wave.count),
                    _fmt_wave_percentile(wave, "ttft_ms", "p50"),
                    _fmt_wave_percentile(wave, "ttft_ms", "p90"),
                    _fmt_wave_percentile(wave, "tpot_ms", "p50"),
                    _fmt_wave_percentile(wave, "e2e_latency_ms", "p50"),
                    _fmt_wave_percentile(wave, "output_tokens", "p50"),
                )
            )
    return body


def _expected_requests_for_queue_profile(
    summary: InferenceBenchRunSummary,
    profile: QueueProfileSummary,
) -> int | None:
    key = (profile.temperature, profile.max_tokens)
    benchmark = next(
        (
            name
            for name, expected_key in _BENCHMARK_QUEUE_PROFILE_KEYS.items()
            if expected_key == key
        ),
        None,
    )
    if benchmark is None:
        return None
    row = next(
        (
            item
            for item in summary.provider_benchmarks
            if item.provider == "torchinferno" and item.benchmark == benchmark
        ),
        None,
    )
    if row is None:
        return None
    metric_count = row.metrics.get("num_requests")
    if metric_count is not None and metric_count >= 0 and float(metric_count).is_integer():
        return int(metric_count)
    for percentile in row.request_percentiles.values():
        return percentile.count
    return None


def _queue_profiles_for_selected_benchmarks(
    summary: InferenceBenchRunSummary,
) -> tuple[QueueProfileSummary, ...]:
    profiles = summary.torchinferno_queue_profiles
    expected_keys = {
        _BENCHMARK_QUEUE_PROFILE_KEYS[benchmark]
        for benchmark in summary.benchmarks
        if benchmark in _BENCHMARK_QUEUE_PROFILE_KEYS
    }
    if not expected_keys:
        return profiles
    selected = tuple(
        profile
        for profile in profiles
        if (profile.temperature, profile.max_tokens) in expected_keys
    )
    return selected or profiles


def _fmt_queue_profile_coverage(
    profile: QueueProfileSummary,
    expected_requests: int | None,
) -> str:
    submitted = profile.submitted_requests
    if submitted is None:
        return "-"
    if expected_requests is None or expected_requests <= 0:
        text = str(submitted)
        if profile.segments > 1:
            return f"{text} {profile.segments}seg"
        return text
    text = f"{submitted}/{expected_requests}"
    if profile.segments > 1:
        text = f"{text} {profile.segments}seg"
    if submitted < expected_requests:
        return f"{text} partial"
    if submitted > expected_requests:
        return f"{text} over"
    return text


def _fmt_wave_percentile(wave: RawRequestWaveSummary, metric: str, field: str) -> str:
    summary = wave.request_percentiles.get(metric)
    if summary is None:
        return "-"
    return _fmt_value(getattr(summary, field))


def _percentiles(values: Iterable[float]) -> PercentileSummary:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty values")
    return PercentileSummary(
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p99=_percentile(ordered, 0.99),
        minimum=ordered[0],
        maximum=ordered[-1],
        count=len(ordered),
    )


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    index = int((len(ordered) - 1) * quantile)
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def _summarize_torchinferno_queue(root: Path) -> tuple[QueueProfileSummary, ...]:
    path = root / "provider_logs" / "torchinferno_queue_profile.jsonl"
    if not path.exists():
        return ()
    segments_by_key: dict[tuple[float | None, int | None], list[QueueProfileSummary]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event = str(record.get("event", ""))
        if event not in {"online_batcher", "online_batcher_quiescent"}:
            continue
        key = (_maybe_float(record.get("temperature")), _maybe_int(record.get("run_max_tokens")))
        profile = QueueProfileSummary(
            event=event,
            temperature=key[0],
            max_tokens=key[1],
            submitted_requests=_maybe_int(record.get("submitted_requests")),
            finished_events=_maybe_int(record.get("finished_events")),
            fields={name: record.get(name) for name in _QUEUE_PROFILE_FIELDS if name in record},
        )
        segments = segments_by_key.setdefault(key, [])
        if segments and _queue_profile_restarts(segments[-1], profile):
            segments.append(profile)
            continue
        if segments:
            segments[-1] = profile
        else:
            segments.append(profile)
    return tuple(
        _merge_queue_profile_segments(segments_by_key[key])
        for key in sorted(
            segments_by_key,
            key=lambda item: (
                -1.0 if item[0] is None else item[0],
                -1 if item[1] is None else item[1],
            ),
        )
    )


def _queue_profile_restarts(previous: QueueProfileSummary, current: QueueProfileSummary) -> bool:
    previous_submitted = previous.submitted_requests
    current_submitted = current.submitted_requests
    if previous_submitted is not None and current_submitted is not None:
        return current_submitted < previous_submitted
    previous_finished = previous.finished_events
    current_finished = current.finished_events
    return (
        previous_finished is not None
        and current_finished is not None
        and current_finished < previous_finished
    )


def _merge_queue_profile_segments(
    segments: Sequence[QueueProfileSummary],
) -> QueueProfileSummary:
    if len(segments) == 1:
        return segments[0]
    latest = segments[-1]
    field_names = sorted({name for segment in segments for name in segment.fields})
    fields: dict[str, Any] = {}
    for name in field_names:
        values = [segment.fields.get(name) for segment in segments if name in segment.fields]
        merged = _merge_queue_profile_field(name, values)
        if merged is not None:
            fields[name] = merged
    return QueueProfileSummary(
        event="online_batcher_merged",
        temperature=latest.temperature,
        max_tokens=latest.max_tokens,
        submitted_requests=_sum_optional_int(segment.submitted_requests for segment in segments),
        finished_events=_sum_optional_int(segment.finished_events for segment in segments),
        fields=fields,
        segments=len(segments),
    )


def _merge_queue_profile_field(name: str, values: Sequence[Any]) -> Any:
    if not values:
        return None
    if all(isinstance(value, dict) for value in values):
        if _queue_profile_field_is_additive(name):
            return _sum_numeric_mappings(values)
        return values[-1]
    if _queue_profile_field_is_additive(name):
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        if len(numeric_values) == len(values):
            return sum(numeric_values)
    first = values[0]
    if all(value == first for value in values):
        return first
    if name.startswith("request_") and name.endswith("_ms"):
        return None
    return values[-1]


def _queue_profile_field_is_additive(name: str) -> bool:
    if name in {
        "runtime_max_active_requests",
        "runtime_prefix_cache_capacity",
        "runtime_prefill_graph_cache_live_entries",
        "runtime_decode_graph_cache_live_entries",
    }:
        return False
    if "cache_live" in name:
        return False
    if name.startswith("request_"):
        return name.endswith("_count")
    if name.startswith("runtime_"):
        return True
    return name in {
        "active_ready_wait_ms",
        "request_stream_prequeue_wait_applied_count",
    }


def _sum_numeric_mappings(values: Sequence[Any]) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    integral: dict[str, bool] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            if not isinstance(item, (int, float)):
                continue
            text_key = str(key)
            totals[text_key] = totals.get(text_key, 0.0) + float(item)
            integral[text_key] = integral.get(text_key, True) and float(item).is_integer()
    return {
        key: int(total) if integral.get(key, False) and float(total).is_integer() else total
        for key, total in totals.items()
    }


def _sum_optional_int(values: Iterable[int | None]) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        total += int(value)
        seen = True
    return total if seen else None


def _summarize_provider_server_logs(root: Path) -> tuple[ProviderServerLogSummary, ...]:
    logs_dir = root / "provider_logs"
    summaries: list[ProviderServerLogSummary] = []
    vllm_path = _first_existing_provider_log(logs_dir, "vllm.log", "vllm_server.log")
    if vllm_path.exists():
        summaries.append(_summarize_vllm_server_log(vllm_path))
    sglang_path = _first_existing_provider_log(logs_dir, "sglang.log", "sglang_server.log")
    if sglang_path.exists():
        summaries.append(_summarize_sglang_server_log(sglang_path))
    return tuple(summaries)


def _first_existing_provider_log(logs_dir: Path, *names: str) -> Path:
    for name in names:
        path = logs_dir / name
        if path.exists():
            return path
    return logs_dir / names[0]


def _summarize_vllm_server_log(path: Path) -> ProviderServerLogSummary:
    prompt_tps: list[float] = []
    generation_tps: list[float] = []
    prefix_hit_pct: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        match = _VLLM_RUNTIME_RE.search(line)
        if match is None:
            continue
        prompt_tps.append(float(match.group("prompt_tps")))
        generation_tps.append(float(match.group("generation_tps")))
        prefix_hit_pct.append(float(match.group("prefix_hit_pct")))
    return ProviderServerLogSummary(
        provider="vllm",
        prompt_events=len(prompt_tps),
        prompt_tps_avg=_avg(prompt_tps),
        prompt_tps_max=max(prompt_tps) if prompt_tps else None,
        generation_events=len(generation_tps),
        generation_tps_avg=_avg(generation_tps),
        generation_tps_max=max(generation_tps) if generation_tps else None,
        prefix_hit_pct_avg=_avg(prefix_hit_pct),
        prefix_hit_pct_max=max(prefix_hit_pct) if prefix_hit_pct else None,
    )


def _summarize_sglang_server_log(path: Path) -> ProviderServerLogSummary:
    prompt_tps: list[float] = []
    generation_tps: list[float] = []
    prefill_batches = 0
    prefill_new_tokens = 0
    prefill_cached_tokens = 0
    prefill_cuda_graph_batches = 0
    prefill_new_seq_max: int | None = None
    decode_batches = 0
    decode_logged_tokens = 0
    decode_cuda_graph_batches = 0
    decode_running_max: int | None = None

    for line in path.read_text(errors="replace").splitlines():
        prefill_match = _SGLANG_PREFILL_RE.search(line)
        if prefill_match is not None:
            prefill_batches += 1
            new_seq = int(prefill_match.group("new_seq"))
            new_tokens = int(prefill_match.group("new_tokens"))
            cached_tokens = int(prefill_match.group("cached_tokens"))
            prefill_new_tokens += new_tokens
            prefill_cached_tokens += cached_tokens
            prefill_new_seq_max = (
                new_seq
                if prefill_new_seq_max is None
                else max(prefill_new_seq_max, new_seq)
            )
            if prefill_match.group("cuda_graph") == "True":
                prefill_cuda_graph_batches += 1
            prompt_tps.append(float(prefill_match.group("prompt_tps")))
            continue

        decode_match = _SGLANG_DECODE_RE.search(line)
        if decode_match is None:
            continue
        decode_batches += 1
        running = int(decode_match.group("running"))
        decode_logged_tokens += int(decode_match.group("tokens"))
        decode_running_max = (
            running
            if decode_running_max is None
            else max(decode_running_max, running)
        )
        if decode_match.group("cuda_graph") == "True":
            decode_cuda_graph_batches += 1
        generation_tps.append(float(decode_match.group("generation_tps")))

    return ProviderServerLogSummary(
        provider="sglang",
        prompt_events=len(prompt_tps),
        prompt_tps_avg=_avg(prompt_tps),
        prompt_tps_max=max(prompt_tps) if prompt_tps else None,
        generation_events=len(generation_tps),
        generation_tps_avg=_avg(generation_tps),
        generation_tps_max=max(generation_tps) if generation_tps else None,
        prefill_batches=prefill_batches,
        prefill_new_tokens=prefill_new_tokens,
        prefill_cached_tokens=prefill_cached_tokens,
        prefill_cuda_graph_batches=prefill_cuda_graph_batches,
        prefill_new_seq_max=prefill_new_seq_max,
        decode_batches=decode_batches,
        decode_logged_tokens=decode_logged_tokens,
        decode_cuda_graph_batches=decode_cuda_graph_batches,
        decode_running_max=decode_running_max,
    )


def _provider_server_log_rows(
    summaries: Sequence[ProviderServerLogSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for summary in summaries:
        rows.append(
            (
                summary.provider,
                _fmt_value(summary.prompt_events),
                _fmt_value(summary.prompt_tps_avg),
                _fmt_value(summary.prompt_tps_max),
                _fmt_value(summary.generation_events),
                _fmt_value(summary.generation_tps_avg),
                _fmt_value(summary.generation_tps_max),
                _fmt_pct_value(summary.prefix_hit_pct_avg),
                _fmt_pct_value(summary.prefix_hit_pct_max),
                _fmt_value(summary.prefill_batches),
                _fmt_value(summary.prefill_new_tokens),
                _fmt_value(summary.prefill_cached_tokens),
                _fmt_pct(summary.prefill_cuda_graph_batches, summary.prefill_batches),
                _fmt_value(summary.prefill_new_seq_max),
                _fmt_value(summary.decode_batches),
                _fmt_value(summary.decode_logged_tokens),
                _fmt_pct(summary.decode_cuda_graph_batches, summary.decode_batches),
                _fmt_value(summary.decode_running_max),
            )
        )
    return rows


def _avg(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _maybe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _maybe_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _fmt_percentile(row: ProviderBenchmarkSummary, metric: str, field: str) -> str:
    summary = row.request_percentiles.get(metric)
    if summary is None:
        return "-"
    return _fmt_value(getattr(summary, field))


def _fmt_metric(
    row: ProviderBenchmarkSummary,
    metric: str,
    *,
    fallback_metric: str | None = None,
) -> str:
    if metric in row.metrics:
        return _fmt_value(row.metrics[metric])
    if fallback_metric is not None:
        return _fmt_percentile(row, fallback_metric, "p50")
    return "-"


def _metric_number(
    row: ProviderBenchmarkSummary,
    metric: str,
    *,
    fallback_metric: str | None = None,
) -> float | None:
    if metric in row.metrics:
        return float(row.metrics[metric])
    if fallback_metric is not None:
        summary = row.request_percentiles.get(fallback_metric)
        if summary is not None:
            return float(summary.p50)
    return None


def _provider_gap_rows(
    rows: Sequence[ProviderBenchmarkSummary],
) -> list[tuple[str, ...]]:
    torchinferno_row = next((row for row in rows if row.provider == "torchinferno"), None)
    competitors = [row for row in rows if row.provider != "torchinferno"]
    if torchinferno_row is None or not competitors:
        return []

    gap_rows: list[tuple[str, ...]] = []
    for label, score_metric, fallback_metric, higher_is_better in _PROVIDER_GAP_METRICS:
        torchinferno_value = _metric_number(
            torchinferno_row,
            score_metric,
            fallback_metric=fallback_metric,
        )
        if torchinferno_value is None:
            continue
        competitor_values = [
            (
                row,
                _metric_number(row, score_metric, fallback_metric=fallback_metric),
            )
            for row in competitors
        ]
        competitor_values = [
            (row, value)
            for row, value in competitor_values
            if value is not None
        ]
        if not competitor_values:
            continue
        best_row, best_value = (
            max(competitor_values, key=lambda item: item[1])
            if higher_is_better
            else min(competitor_values, key=lambda item: item[1])
        )
        assert best_value is not None
        gap = (
            best_value - torchinferno_value
            if higher_is_better
            else torchinferno_value - best_value
        )
        ratio: float | None
        if higher_is_better:
            ratio = best_value / torchinferno_value if torchinferno_value > 0.0 else None
        else:
            ratio = torchinferno_value / best_value if best_value > 0.0 else None
        gap_rows.append(
            (
                label,
                _fmt_value(torchinferno_value),
                _fmt_value(best_value),
                best_row.provider,
                _fmt_signed_value(gap),
                _fmt_ratio(ratio),
            )
        )
    return gap_rows


def _torchinferno_score_target_rows(
    summary: InferenceBenchRunSummary,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    profiles_by_key = {
        (profile.temperature, profile.max_tokens): profile
        for profile in summary.torchinferno_queue_profiles
        if profile.temperature is not None and profile.max_tokens is not None
    }
    for benchmark in summary.benchmarks:
        provider_rows = [
            row for row in summary.provider_benchmarks if row.benchmark == benchmark
        ]
        torchinferno_row = next(
            (row for row in provider_rows if row.provider == "torchinferno"),
            None,
        )
        competitors = [row for row in provider_rows if row.provider != "torchinferno"]
        profile = profiles_by_key.get(_BENCHMARK_QUEUE_PROFILE_KEYS.get(benchmark))
        if torchinferno_row is None or not competitors or profile is None:
            continue
        ttft_gap = _provider_gap_value(
            torchinferno_row,
            competitors,
            "ttft_median_ms",
            "ttft_ms",
            higher_is_better=False,
        )
        tpot_gap = _provider_gap_value(
            torchinferno_row,
            competitors,
            "tpot_median_ms",
            "tpot_ms",
            higher_is_better=False,
        )
        e2e_gap = _provider_gap_value(
            torchinferno_row,
            competitors,
            "e2e_median_ms",
            "e2e_latency_ms",
            higher_is_better=False,
        )
        fields = profile.fields
        prefill_ms = _numeric_field(fields, "runtime_prefill_forward_ms")
        prefill_sample_ms = _numeric_field(fields, "runtime_prefill_sample_ms")
        _prefill_active_tokens, prefill_model_tokens, prefill_padding_tokens = (
            _prefill_token_totals(fields)
        )
        prefill_row_padding_tokens, prefill_suffix_padding_tokens = (
            _prefill_padding_split_totals(fields)
        )
        decode_ms = _numeric_field(fields, "runtime_decode_ragged_model_gpu_ms")
        decode_many_ms = _numeric_field(fields, "runtime_decode_many_model_gpu_ms")
        decode_cpu_ms = _numeric_field(fields, "runtime_decode_ragged_cpu_tokens_ms")
        decode_many_cpu_ms = _numeric_field(fields, "runtime_decode_many_cpu_tokens_ms")
        if decode_many_ms is None:
            decode_many_ms = _sum_numeric_mapping(
                fields.get("runtime_decode_many_shape_gpu_ms")
            )
        if (
            decode_many_ms is None
            and (_numeric_field(fields, "runtime_decode_many_calls") or 0.0) > 0.0
        ):
            decode_many_ms = _sum_numeric_mapping(fields.get("runtime_decode_shape_gpu_ms"))
        decode_target_ms = max(
            value for value in (decode_ms, decode_many_ms) if value is not None
        ) if decode_ms is not None or decode_many_ms is not None else None
        capture_ms = (_numeric_field(fields, "runtime_prefill_graph_capture_ms") or 0.0) + (
            _numeric_field(fields, "runtime_decode_graph_capture_ms") or 0.0
        )
        phase_target = _phase_target(
            prefill_ms=prefill_ms,
            decode_ms=decode_target_ms,
            sample_ms=prefill_sample_ms,
            capture_ms=capture_ms,
        )
        hot_prefill, _hot_prefill_ms = _top_mapping_entry(
            fields.get("runtime_prefill_shape_forward_ms")
        )
        hot_decode, _hot_decode_ms = _top_mapping_entry(
            fields.get("runtime_decode_many_shape_gpu_ms")
        )
        if hot_decode is None:
            hot_decode, _hot_decode_ms = _top_mapping_entry(
                fields.get("runtime_decode_shape_gpu_ms")
            )
        rows.append(
            (
                benchmark,
                _fmt_signed_or_dash(ttft_gap),
                _fmt_signed_or_dash(tpot_gap),
                _fmt_signed_or_dash(e2e_gap),
                phase_target,
                _fmt_value(fields.get("request_queue_to_first_token_p50_ms")),
                _fmt_value(fields.get("request_queue_to_submit_p50_ms")),
                _fmt_value(fields.get("request_submit_to_first_token_p50_ms")),
                _fmt_value(prefill_ms),
                _fmt_value(prefill_sample_ms),
                _fmt_value(
                    None
                    if prefill_padding_tokens is None
                    else _int_if_whole(prefill_padding_tokens)
                ),
                _fmt_value(
                    None
                    if prefill_row_padding_tokens is None
                    else _int_if_whole(prefill_row_padding_tokens)
                ),
                _fmt_value(
                    None
                    if prefill_suffix_padding_tokens is None
                    else _int_if_whole(prefill_suffix_padding_tokens)
                ),
                _fmt_pct(prefill_padding_tokens or 0.0, prefill_model_tokens or 0.0),
                _fmt_value(decode_ms),
                _fmt_value(decode_many_ms),
                _fmt_value(decode_cpu_ms),
                _fmt_value(decode_many_cpu_ms),
                _fmt_value(fields.get("runtime_decode_many_calls")),
                _fmt_value(fields.get("runtime_prefill_graph_misses")),
                _fmt_mapping_summary(_prefill_graph_miss_kind_counts(fields)),
                _fmt_value(fields.get("runtime_decode_graph_misses")),
                _fmt_mapping_summary(_decode_graph_miss_kind_counts(fields)),
                _fmt_value(fields.get("runtime_generated_prefix_store_requests")),
                _fmt_value(fields.get("runtime_generated_prefix_reuse_requests")),
                _fmt_value(fields.get("runtime_prefill_packed_candidate_saved_tokens")),
                _short_shape(hot_prefill),
                _short_shape(hot_decode),
            )
        )
    rows.sort(
        key=lambda row: (
            -_signed_row_value(row[3]),
            row[0],
        )
    )
    return rows


def _provider_gap_value(
    torchinferno_row: ProviderBenchmarkSummary,
    competitors: Sequence[ProviderBenchmarkSummary],
    score_metric: str,
    fallback_metric: str,
    *,
    higher_is_better: bool,
) -> float | None:
    torchinferno_value = _metric_number(
        torchinferno_row,
        score_metric,
        fallback_metric=fallback_metric,
    )
    if torchinferno_value is None:
        return None
    competitor_values = [
        _metric_number(row, score_metric, fallback_metric=fallback_metric)
        for row in competitors
    ]
    competitor_values = [value for value in competitor_values if value is not None]
    if not competitor_values:
        return None
    best_value = (
        max(competitor_values) if higher_is_better else min(competitor_values)
    )
    return (
        best_value - torchinferno_value
        if higher_is_better
        else torchinferno_value - best_value
    )


def _phase_target(
    *,
    prefill_ms: float | None,
    decode_ms: float | None,
    sample_ms: float | None,
    capture_ms: float,
) -> str:
    values = {
        name: value
        for name, value in (
            ("prefill", prefill_ms),
            ("decode", decode_ms),
            ("sample", sample_ms),
        )
        if value is not None
    }
    if capture_ms > 0.0 and capture_ms >= max(values.values(), default=0.0):
        return "capture"
    if not values:
        return "unknown"
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) >= 2:
        top_name, top_value = ranked[0]
        second_name, second_value = ranked[1]
        if second_value > 0.0 and second_value / top_value >= 0.75:
            labels = {top_name, second_name}
            return "+".join(
                label for label in ("prefill", "decode", "sample") if label in labels
            )
    return ranked[0][0]


def _sum_numeric_mapping(value: Any) -> float | None:
    mapping = _numeric_mapping(value)
    if not mapping:
        return None
    return sum(mapping.values())


def _prefill_token_totals(fields: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    active_tokens = _sum_numeric_mapping(fields.get("runtime_prefill_shape_active_tokens"))
    model_tokens = _sum_numeric_mapping(fields.get("runtime_prefill_shape_model_tokens"))
    padding_tokens = _sum_numeric_mapping(fields.get("runtime_prefill_shape_padding_tokens"))
    if padding_tokens is None and active_tokens is not None and model_tokens is not None:
        padding_tokens = max(0.0, model_tokens - active_tokens)
    return active_tokens, model_tokens, padding_tokens


def _prefill_padding_split_totals(fields: dict[str, Any]) -> tuple[float | None, float | None]:
    row_padding = _numeric_field(fields, "runtime_prefill_row_padding_tokens")
    suffix_padding = _numeric_field(fields, "runtime_prefill_suffix_padding_tokens")
    if row_padding is None:
        row_padding = _sum_numeric_mapping(
            fields.get("runtime_prefill_shape_row_padding_tokens")
        )
    if suffix_padding is None:
        suffix_padding = _sum_numeric_mapping(
            fields.get("runtime_prefill_shape_suffix_padding_tokens")
        )
    return row_padding, suffix_padding


def _prefill_graph_miss_kind_counts(fields: dict[str, Any]) -> dict[str, float | int]:
    shape_counts = _numeric_mapping_preserving_type(
        fields.get("runtime_prefill_graph_miss_shape_counts")
    )
    kind_counts: dict[str, float | int] = {}
    for shape, count in shape_counts.items():
        kind = _prefill_graph_miss_kind(shape)
        kind_counts[kind] = kind_counts.get(kind, 0) + count
    return kind_counts


def _prefill_graph_miss_kind(shape: str) -> str:
    parts = shape.split(":")
    if len(parts) >= 2 and parts[0] == "static_prefill":
        return f"static_{parts[1]}"
    if parts[:1] == ["ragged_prefill"]:
        return "ragged"
    if parts[:1] == ["prefix_graph"]:
        return "prefix_graph"
    return "other"


def _decode_graph_miss_kind_counts(fields: dict[str, Any]) -> dict[str, float | int]:
    shape_counts = _numeric_mapping_preserving_type(
        fields.get("runtime_decode_graph_miss_shape_counts")
    )
    kind_counts: dict[str, float | int] = {}
    for shape, count in shape_counts.items():
        kind = _decode_graph_miss_kind(shape)
        kind_counts[kind] = kind_counts.get(kind, 0) + count
    return kind_counts


def _decode_graph_miss_kind(shape: str) -> str:
    parts = shape.split(":")
    if len(parts) >= 2 and parts[0] == "static_decode":
        return f"static_{parts[1]}"
    if len(parts) >= 2 and parts[0] == "ragged_decode":
        return f"ragged_{parts[1]}"
    return "other"


def _graph_miss_shape_rows(
    profiles: Sequence[QueueProfileSummary],
    field_name: str,
    kind_func: Callable[[str], str],
    *,
    limit: int = 8,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        for shape, count in _top_mapping_entries(fields.get(field_name), limit=limit):
            rows.append(
                (
                    _fmt_value(profile.temperature),
                    _fmt_value(profile.max_tokens),
                    kind_func(shape),
                    shape,
                    _fmt_value(count),
                )
            )
    return rows


def _decode_graph_symm_counts(fields: dict[str, Any]) -> dict[str, float | int]:
    counts = {
        key: raw
        for key, raw in _numeric_mapping_preserving_type(
            fields.get("runtime_decode_graph_cache_live_symm_counts")
        ).items()
    }
    if counts:
        return counts
    shape_counts = _numeric_mapping_preserving_type(
        fields.get("runtime_decode_graph_cache_live_shape_counts")
    )
    derived_counts: dict[str, float | int] = {}
    for shape, count in shape_counts.items():
        symm_key = _decode_graph_shape_symm_key(shape)
        if symm_key is None:
            continue
        derived_counts[symm_key] = derived_counts.get(symm_key, 0) + count
    return derived_counts


def _decode_graph_cache_counts(fields: dict[str, Any]) -> dict[str, float | int]:
    counts = {
        key: raw
        for key, raw in _numeric_mapping_preserving_type(
            fields.get("runtime_decode_graph_cache_live_cache_bucket_counts")
        ).items()
    }
    if counts:
        return counts
    shape_counts = _numeric_mapping_preserving_type(
        fields.get("runtime_decode_graph_cache_live_shape_counts")
    )
    derived_counts: dict[str, float | int] = {}
    for shape, count in shape_counts.items():
        cache_key = _decode_graph_shape_cache_key(shape)
        if cache_key is None:
            continue
        derived_counts[cache_key] = derived_counts.get(cache_key, 0) + count
    return derived_counts


def _decode_graph_symm_value_totals(value: Any) -> dict[str, float | int]:
    totals: dict[str, float | int] = {}
    for shape, raw in _numeric_mapping_preserving_type(value).items():
        symm_key = _decode_graph_shape_symm_key(shape)
        if symm_key is None:
            continue
        totals[symm_key] = totals.get(symm_key, 0) + raw
    return totals


def _decode_graph_cache_value_totals(value: Any) -> dict[str, float | int]:
    totals: dict[str, float | int] = {}
    for shape, raw in _numeric_mapping_preserving_type(value).items():
        cache_key = _decode_graph_shape_cache_key(shape)
        if cache_key is None:
            continue
        totals[cache_key] = totals.get(cache_key, 0) + raw
    return totals


def _decode_graph_shape_symm_key(shape: str) -> str | None:
    for part in shape.split(":"):
        if part.startswith("symm"):
            return part
    return None


def _decode_graph_shape_cache_key(shape: str) -> str | None:
    for part in shape.split(":"):
        if part.startswith("cache"):
            return part
    return None


def _fmt_signed_or_dash(value: float | None) -> str:
    if value is None:
        return "-"
    return _fmt_signed_value(value)


def _signed_row_value(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _short_shape(shape: str | None) -> str:
    if not shape:
        return "-"
    if "|" in shape:
        shape = shape.split("|", 1)[0]
    if shape.startswith("prefix_graph:"):
        return shape.removeprefix("prefix_graph:")
    return shape


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _fmt_mapping_summary(value: Any, *, limit: int = 3) -> str:
    entries = _top_mapping_entries(value, limit=limit)
    if not entries:
        return "-"
    return ",".join(f"{key}={_fmt_value(raw)}" for key, raw in entries)


def _fmt_signed_value(value: float) -> str:
    return f"{value:+.1f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _fmt_pct(numerator: float | int, denominator: float | int) -> str:
    if float(denominator) <= 0.0:
        return "-"
    return f"{(float(numerator) / float(denominator)) * 100.0:.1f}%"


def _fmt_pct_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _ratio_or_none(
    numerator: Any,
    denominator: Any,
    *,
    scale: float = 1.0,
) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) * float(scale) / float(denominator)


def _int_if_whole(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _hot_prefill_shape_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        shape_entries = _top_mapping_entries(
            fields.get("runtime_prefill_shape_forward_ms"),
            limit=3,
        )
        if not shape_entries:
            shape_entries = _top_mapping_entries(
                fields.get("runtime_prefill_shape_padding_tokens"),
                limit=3,
            )
        if not shape_entries:
            continue
        for shape, forward_ms in shape_entries:
            active_tokens = _mapping_value(
                fields.get("runtime_prefill_shape_active_tokens"),
                shape,
            )
            model_tokens = _mapping_value(
                fields.get("runtime_prefill_shape_model_tokens"),
                shape,
            )
            padding_tokens = _mapping_value(
                fields.get("runtime_prefill_shape_padding_tokens"),
                shape,
            )
            if padding_tokens is None and isinstance(
                active_tokens,
                (int, float),
            ) and isinstance(model_tokens, (int, float)):
                padding_tokens = max(0, model_tokens - active_tokens)
            calls = _mapping_value(fields.get("runtime_prefill_shape_counts"), shape)
            graph_gpu_ms = _mapping_value(
                fields.get("runtime_prefill_shape_graph_replay_gpu_ms"),
                shape,
            )
            row_padding_tokens = _mapping_value(
                fields.get("runtime_prefill_shape_row_padding_tokens"),
                shape,
            )
            suffix_padding_tokens = _mapping_value(
                fields.get("runtime_prefill_shape_suffix_padding_tokens"),
                shape,
            )
            rows.append(
                (
                    _fmt_value(profile.temperature),
                    _fmt_value(profile.max_tokens),
                    shape,
                    _fmt_value(calls),
                    _fmt_value(forward_ms),
                    _fmt_value(graph_gpu_ms),
                    _fmt_value(_ratio_or_none(graph_gpu_ms, calls)),
                    _fmt_value(_ratio_or_none(graph_gpu_ms, model_tokens, scale=1000.0)),
                    _fmt_value(
                        _mapping_value(fields.get("runtime_prefill_shape_wall_ms"), shape)
                    ),
                    _fmt_value(
                        _mapping_value(fields.get("runtime_prefill_shape_setup_ms"), shape)
                    ),
                    _fmt_value(
                        _mapping_value(fields.get("runtime_prefill_shape_copy_ms"), shape)
                    ),
                    _fmt_value(
                        _mapping_value(fields.get("runtime_prefill_shape_sample_ms"), shape)
                    ),
                    _fmt_value(
                        _mapping_value(fields.get("runtime_prefill_shape_state_ms"), shape)
                    ),
                    _fmt_value(active_tokens),
                    _fmt_value(model_tokens),
                    _fmt_value(padding_tokens),
                    _fmt_pct(float(padding_tokens or 0), float(model_tokens or 0)),
                    _fmt_value(_ratio_or_none(padding_tokens, calls)),
                    _fmt_value(row_padding_tokens),
                    _fmt_value(suffix_padding_tokens),
                    _fmt_value(fields.get("runtime_prefill_graph_cache_live_entries")),
                )
            )
    return rows


def _hot_prefill_packed_candidate_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        candidate_shape, saved_tokens = _top_mapping_entry(
            fields.get("runtime_prefill_packed_candidate_shape_saved_tokens")
        )
        if candidate_shape is None:
            continue
        rows.append(
            (
                _fmt_value(profile.temperature),
                _fmt_value(profile.max_tokens),
                candidate_shape,
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_shape_tokens"),
                        candidate_shape,
                    )
                ),
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_shape_model_tokens"),
                        candidate_shape,
                    )
                ),
                _fmt_value(saved_tokens),
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_shape_groups"),
                        candidate_shape,
                    )
                ),
            )
        )
    return rows


def _prefill_packed_per_batch_target_rows(
    profiles: Sequence[QueueProfileSummary],
    *,
    limit: int = 8,
) -> list[tuple[str, ...]]:
    items: list[tuple[float, float, str, tuple[str, ...]]] = []
    for profile in profiles:
        fields = profile.fields
        shape_saved = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_shape_saved_tokens")
        )
        if not shape_saved:
            continue
        shape_model_tokens = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_shape_model_tokens")
        )
        shape_real_tokens = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_shape_tokens")
        )
        shape_row_saved = _numeric_mapping(
            fields.get("runtime_prefill_shape_row_padding_tokens")
        )
        shape_suffix_saved = _numeric_mapping(
            fields.get("runtime_prefill_shape_suffix_padding_tokens")
        )
        shape_counts = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_shape_counts")
        )
        if not shape_counts:
            shape_counts = _numeric_mapping(fields.get("runtime_prefill_shape_counts"))
        shape_groups = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_shape_groups")
        )
        for shape, saved_tokens in shape_saved.items():
            saved = max(0.0, float(saved_tokens))
            model_tokens = max(0.0, float(shape_model_tokens.get(shape, 0.0)))
            if saved <= 0.0 or model_tokens <= 0.0:
                continue
            forward_ms = _prefill_shape_dense_forward_ms(fields, shape)
            observed_packed_ms = _prefill_shape_observed_packed_ms(fields, shape)
            est_saved_ms: float | None = None
            score_ms = -1.0
            if forward_ms is not None and forward_ms > 0.0:
                est_saved_ms = forward_ms * saved / model_tokens
                score_ms = est_saved_ms
            has_split = shape in shape_row_saved or shape in shape_suffix_saved
            row_saved = shape_row_saved.get(shape, 0.0) if has_split else None
            suffix_saved = shape_suffix_saved.get(shape, 0.0) if has_split else None
            items.append(
                (
                    score_ms,
                    saved,
                    shape,
                    (
                        _fmt_value(profile.temperature),
                        _fmt_value(profile.max_tokens),
                        shape,
                        _fmt_value(_int_if_whole(shape_counts.get(shape, 0.0))),
                        _fmt_value(_int_if_whole(shape_real_tokens.get(shape, 0.0))),
                        _fmt_value(_int_if_whole(model_tokens)),
                        _fmt_value(_int_if_whole(saved)),
                        _fmt_value(
                            None
                            if row_saved is None
                            else _int_if_whole(max(0.0, row_saved))
                        ),
                        _fmt_value(
                            None
                            if suffix_saved is None
                            else _int_if_whole(max(0.0, suffix_saved))
                        ),
                        _fmt_pct(saved, model_tokens),
                        _fmt_value(
                            None
                            if est_saved_ms is None
                            else _int_if_whole(est_saved_ms)
                        ),
                        _fmt_value(
                            None
                            if observed_packed_ms is None
                            else _int_if_whole(observed_packed_ms)
                        ),
                        _fmt_value(_int_if_whole(shape_groups.get(shape, 0.0))),
                    ),
                )
            )
    items.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in items[:limit]]


def _hot_prefill_packed_signature_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        signature, saved_tokens = _top_mapping_entry(
            fields.get("runtime_prefill_packed_candidate_signature_saved_tokens")
        )
        if signature is None:
            continue
        rows.append(
            (
                _fmt_value(profile.temperature),
                _fmt_value(profile.max_tokens),
                signature,
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_signature_counts"),
                        signature,
                    )
                ),
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_signature_tokens"),
                        signature,
                    )
                ),
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_signature_model_tokens"),
                        signature,
                    )
                ),
                _fmt_value(saved_tokens),
                _fmt_value(
                    _mapping_value(
                        fields.get("runtime_prefill_packed_candidate_signature_groups"),
                        signature,
                    )
                ),
            )
        )
    return rows


def _hot_prefill_packed_pattern_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        total_saved_raw = fields.get("runtime_prefill_packed_candidate_saved_tokens")
        total_saved = (
            float(total_saved_raw)
            if isinstance(total_saved_raw, (int, float))
            else 0.0
        )
        for pattern, saved_tokens in _top_mapping_entries(
            fields.get("runtime_prefill_packed_candidate_pattern_saved_tokens"),
            limit=3,
        ):
            rows.append(
                (
                    _fmt_value(profile.temperature),
                    _fmt_value(profile.max_tokens),
                    pattern,
                    _fmt_value(
                        _mapping_value(
                            fields.get("runtime_prefill_packed_candidate_pattern_counts"),
                            pattern,
                        )
                    ),
                    _fmt_value(
                        _mapping_value(
                            fields.get("runtime_prefill_packed_candidate_pattern_tokens"),
                            pattern,
                        )
                    ),
                    _fmt_value(
                        _mapping_value(
                            fields.get("runtime_prefill_packed_candidate_pattern_model_tokens"),
                            pattern,
                        )
                    ),
                    _fmt_value(saved_tokens),
                    _fmt_pct(float(saved_tokens), total_saved),
                    _fmt_value(
                        _mapping_value(
                            fields.get("runtime_prefill_packed_candidate_pattern_groups"),
                            pattern,
                        )
                    ),
                )
            )
    return rows


def _prefill_packed_fixed_capacity_plan_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        pattern_calls = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_pattern_counts")
        )
        if not pattern_calls:
            continue
        pattern_max_slots: dict[str, dict[tuple[int, int], int]] = {}
        runtime_slot_patterns: set[str] = set()
        pattern_signature_calls: dict[str, float] = {}
        for slot_key, slot_count in _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_pattern_slot_counts")
        ).items():
            parsed_slot = _parse_packed_prefill_pattern_slot_key(slot_key)
            if parsed_slot is None:
                continue
            pattern, group = parsed_slot
            if pattern not in pattern_calls:
                continue
            max_slots = pattern_max_slots.setdefault(pattern, {})
            max_slots[group] = max(max_slots.get(group, 0), int(slot_count))
            runtime_slot_patterns.add(pattern)
        for signature, signature_calls in _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_signature_counts")
        ).items():
            parsed = _parse_packed_prefill_signature_key(signature)
            if parsed is None:
                continue
            pattern, group_counts = parsed
            if pattern not in pattern_calls:
                continue
            pattern_signature_calls[pattern] = (
                pattern_signature_calls.get(pattern, 0.0) + float(signature_calls)
            )
            max_slots = pattern_max_slots.setdefault(pattern, {})
            for group, count in group_counts.items():
                max_slots[group] = max(max_slots.get(group, 0), int(count))
        plan_items: list[tuple[float, tuple[str, ...]]] = []
        for pattern, calls in pattern_calls.items():
            max_slots = pattern_max_slots.get(pattern)
            if not max_slots:
                continue
            dense_per_call = _packed_prefill_dense_tokens_per_call(pattern)
            if dense_per_call is None or dense_per_call <= 0:
                continue
            fixed_per_call = sum(
                max(0, int(count)) * max(0, int(suffix_len))
                for (_start_len, suffix_len), count in max_slots.items()
            )
            if fixed_per_call <= 0:
                continue
            call_count = max(0.0, float(calls))
            dense_tokens = dense_per_call * call_count
            fixed_tokens = fixed_per_call * call_count
            fixed_saved = max(0.0, dense_tokens - fixed_tokens)
            if fixed_saved <= 0.0:
                continue
            sig_calls = pattern_signature_calls.get(pattern, 0.0)
            est_saved_ms = _prefill_packed_fixed_capacity_saved_ms(
                fields,
                pattern,
                fixed_saved,
            )
            observed_packed_ms = _prefill_shape_observed_packed_ms(
                fields,
                pattern.split("|", 1)[0],
            )
            plan_items.append(
                (
                    fixed_saved,
                    (
                        _fmt_value(profile.temperature),
                        _fmt_value(profile.max_tokens),
                        pattern,
                        _fmt_value(_int_if_whole(call_count)),
                        "runtime" if pattern in runtime_slot_patterns else "signature",
                        _fmt_value(_int_if_whole(sig_calls)),
                        _fmt_pct(sig_calls, call_count),
                        _fmt_value(sum(max_slots.values())),
                        _fmt_value(_int_if_whole(dense_tokens)),
                        _fmt_value(_int_if_whole(fixed_tokens)),
                        _fmt_value(_int_if_whole(fixed_saved)),
                        _fmt_value(None if est_saved_ms is None else _int_if_whole(est_saved_ms)),
                        _fmt_value(
                            None
                            if observed_packed_ms is None
                            else _int_if_whole(observed_packed_ms)
                        ),
                        _fmt_pct(fixed_saved, dense_tokens),
                    ),
                )
            )
        plan_items.sort(key=lambda item: (-item[0], item[1][2]))
        rows.extend(item[1] for item in plan_items[:3])
    return rows


def _prefill_packed_fixed_capacity_saved_ms(
    fields: dict[str, Any],
    pattern: str,
    fixed_saved_tokens: float,
) -> float | None:
    if fixed_saved_tokens <= 0:
        return None
    shape_key = pattern.split("|", 1)[0]
    shape_model_tokens = _mapping_value(
        fields.get("runtime_prefill_shape_model_tokens"),
        shape_key,
    )
    shape_ms = _prefill_shape_dense_forward_ms(fields, shape_key)
    if shape_ms is None or shape_model_tokens is None or shape_model_tokens <= 0:
        return None
    return float(fixed_saved_tokens) * float(shape_ms) / float(shape_model_tokens)


def _prefill_shape_observed_packed_ms(
    fields: dict[str, Any],
    shape_key: str,
) -> float | None:
    packed_ms = _mapping_value(
        fields.get("runtime_prefill_packed_eager_shape_ms"),
        shape_key,
    )
    if packed_ms is None:
        return None
    return max(0.0, float(packed_ms))


def _prefill_shape_dense_forward_ms(
    fields: dict[str, Any],
    shape_key: str,
) -> float | None:
    shape_ms = _mapping_value(fields.get("runtime_prefill_shape_forward_ms"), shape_key)
    if shape_ms is None:
        return None
    dense_ms = max(0.0, float(shape_ms))
    packed_ms = _prefill_shape_observed_packed_ms(fields, shape_key)
    if packed_ms is not None:
        dense_ms = max(0.0, dense_ms - packed_ms)
    return dense_ms


def _prefill_packed_implementation_target_rows(
    profiles: Sequence[QueueProfileSummary],
    *,
    limit: int = 8,
) -> list[tuple[str, ...]]:
    items: list[tuple[float, float, str, tuple[str, ...]]] = []
    for profile in profiles:
        fields = profile.fields
        pattern_calls = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_pattern_counts")
        )
        if not pattern_calls:
            continue
        pattern_saved = _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_pattern_saved_tokens")
        )
        pattern_max_slots: dict[str, dict[tuple[int, int], int]] = {}
        pattern_signature_calls: dict[str, float] = {}
        for slot_key, slot_count in _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_pattern_slot_counts")
        ).items():
            parsed_slot = _parse_packed_prefill_pattern_slot_key(slot_key)
            if parsed_slot is None:
                continue
            pattern, group = parsed_slot
            if pattern not in pattern_calls:
                continue
            max_slots = pattern_max_slots.setdefault(pattern, {})
            max_slots[group] = max(max_slots.get(group, 0), int(slot_count))
        for signature, signature_calls in _numeric_mapping(
            fields.get("runtime_prefill_packed_candidate_signature_counts")
        ).items():
            parsed = _parse_packed_prefill_signature_key(signature)
            if parsed is None:
                continue
            pattern, group_counts = parsed
            if pattern not in pattern_calls:
                continue
            pattern_signature_calls[pattern] = (
                pattern_signature_calls.get(pattern, 0.0) + float(signature_calls)
            )
            max_slots = pattern_max_slots.setdefault(pattern, {})
            for group, count in group_counts.items():
                max_slots[group] = max(max_slots.get(group, 0), int(count))
        for pattern, calls in pattern_calls.items():
            call_count = max(0.0, float(calls))
            if call_count < 2.0:
                continue
            dense_per_call = _packed_prefill_dense_tokens_per_call(pattern)
            max_slots = pattern_max_slots.get(pattern)
            if dense_per_call is None or dense_per_call <= 0 or not max_slots:
                continue
            fixed_per_call = sum(
                max(0, int(count)) * max(0, int(suffix_len))
                for (_start_len, suffix_len), count in max_slots.items()
            )
            if fixed_per_call <= 0:
                continue
            dense_tokens = float(dense_per_call) * call_count
            fixed_tokens = float(fixed_per_call) * call_count
            fixed_saved = max(0.0, dense_tokens - fixed_tokens)
            if fixed_saved <= 0.0:
                continue
            est_saved_ms = _prefill_packed_fixed_capacity_saved_ms(
                fields,
                pattern,
                fixed_saved,
            )
            observed_packed_ms = _prefill_shape_observed_packed_ms(
                fields,
                pattern.split("|", 1)[0],
            )
            repeat_saved = max(0.0, float(pattern_saved.get(pattern, 0.0)))
            sig_calls = pattern_signature_calls.get(pattern, 0.0)
            score_ms = float(est_saved_ms) if est_saved_ms is not None else -1.0
            items.append(
                (
                    score_ms,
                    fixed_saved,
                    pattern,
                    (
                        _fmt_value(profile.temperature),
                        _fmt_value(profile.max_tokens),
                        pattern,
                        _fmt_value(_int_if_whole(call_count)),
                        _fmt_value(_int_if_whole(repeat_saved)),
                        _fmt_value(_int_if_whole(fixed_saved)),
                        _fmt_value(None if est_saved_ms is None else _int_if_whole(est_saved_ms)),
                        _fmt_value(
                            None
                            if observed_packed_ms is None
                            else _int_if_whole(observed_packed_ms)
                        ),
                        _fmt_pct(fixed_saved, dense_tokens),
                        _fmt_pct(sig_calls, call_count),
                    ),
                )
            )
    items.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in items[:limit]]


def _parse_packed_prefill_signature_key(
    signature: str,
) -> tuple[str, dict[tuple[int, int], int]] | None:
    if "|" not in signature:
        return None
    shape_key, group_text = signature.split("|", 1)
    group_counts: dict[tuple[int, int], int] = {}
    pattern_groups: list[tuple[int, int]] = []
    for raw_group in group_text.split("/"):
        parts = raw_group.split(":")
        if len(parts) != 3:
            return None
        try:
            start_len = int(parts[0].removeprefix("p"))
            suffix_len = int(parts[1].removeprefix("s"))
            count = int(parts[2].removeprefix("n"))
        except ValueError:
            return None
        group = (max(0, start_len), max(0, suffix_len))
        group_counts[group] = max(group_counts.get(group, 0), max(0, count))
        pattern_groups.append(group)
    if not group_counts:
        return None
    pattern_suffix = "/".join(
        f"p{start_len}:s{suffix_len}" for start_len, suffix_len in sorted(set(pattern_groups))
    )
    return f"{shape_key}|{pattern_suffix}", group_counts


def _parse_packed_prefill_pattern_slot_key(
    slot_key: str,
) -> tuple[str, tuple[int, int]] | None:
    if "#" not in slot_key:
        return None
    pattern, slot = slot_key.rsplit("#", 1)
    parts = slot.split(":")
    if len(parts) != 2:
        return None
    try:
        start_len = int(parts[0].removeprefix("p"))
        suffix_len = int(parts[1].removeprefix("s"))
    except ValueError:
        return None
    if "|" not in pattern:
        return None
    return pattern, (max(0, start_len), max(0, suffix_len))


def _packed_prefill_dense_tokens_per_call(pattern: str) -> int | None:
    shape_key = pattern.split("|", 1)[0]
    batch: int | None = None
    suffix_bucket: int | None = None
    for part in shape_key.split(":"):
        if len(part) < 2:
            continue
        if part.startswith("b") and part[1:].isdigit():
            batch = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            suffix_bucket = int(part[1:])
    if batch is None or suffix_bucket is None:
        return None
    return batch * suffix_bucket


def _prefill_packed_signature_reuse_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    return _prefill_packed_key_reuse_rows(profiles, kind="signature")


def _prefill_packed_key_reuse_rows(
    profiles: Sequence[QueueProfileSummary],
    *,
    kind: str,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        counts = _numeric_mapping(
            fields.get(f"runtime_prefill_packed_candidate_{kind}_counts")
        )
        keys_scalar = _numeric_field(
            fields, f"runtime_prefill_packed_candidate_{kind}_keys"
        )
        if not counts and keys_scalar is None:
            continue
        calls_scalar = _numeric_field(
            fields, f"runtime_prefill_packed_candidate_{kind}_calls"
        )
        repeated_calls_scalar = _numeric_field(
            fields, f"runtime_prefill_packed_candidate_{kind}_repeated_calls"
        )
        repeated_saved_scalar = _numeric_field(
            fields, f"runtime_prefill_packed_candidate_{kind}_repeated_saved_tokens"
        )
        saved = _numeric_mapping(
            fields.get(f"runtime_prefill_packed_candidate_{kind}_saved_tokens")
        )
        total_calls = int(
            calls_scalar
            if calls_scalar is not None
            else sum(counts.values())
        )
        total_saved_raw = fields.get("runtime_prefill_packed_candidate_saved_tokens")
        total_saved = (
            float(total_saved_raw)
            if isinstance(total_saved_raw, (int, float))
            else sum(saved.values())
        )
        candidate_calls_raw = fields.get("runtime_prefill_packed_candidate_calls")
        candidate_calls = (
            float(candidate_calls_raw)
            if isinstance(candidate_calls_raw, (int, float)) and candidate_calls_raw > 0
            else float(total_calls)
        )
        repeated_calls = int(
            repeated_calls_scalar
            if repeated_calls_scalar is not None
            else sum(value for value in counts.values() if value > 1)
        )
        if repeated_saved_scalar is not None:
            repeated_saved: float | None = repeated_saved_scalar
        else:
            repeated_keys = [
                signature for signature, count in counts.items() if count > 1
            ]
            if not repeated_keys:
                repeated_saved = 0.0
            elif saved and all(signature in saved for signature in repeated_keys):
                repeated_saved = sum(saved[signature] for signature in repeated_keys)
            else:
                repeated_saved = None
        rows.append(
            (
                _fmt_value(profile.temperature),
                _fmt_value(profile.max_tokens),
                _fmt_value(
                    _int_if_whole(
                        keys_scalar
                        if keys_scalar is not None
                        else len(counts)
                    )
                ),
                _fmt_value(total_calls),
                _fmt_value(_int_if_whole(candidate_calls)),
                _fmt_value(repeated_calls),
                _fmt_pct(repeated_calls, candidate_calls),
                _fmt_value(
                    None if repeated_saved is None else _int_if_whole(repeated_saved)
                ),
                "-" if repeated_saved is None else _fmt_pct(repeated_saved, total_saved),
            )
        )
    return rows


def _hot_prefill_graph_shape_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        capture_shape, capture_ms = _top_mapping_entry(
            fields.get("runtime_prefill_shape_graph_capture_ms")
        )
        capture_gpu_ms = (
            _mapping_value(
                fields.get("runtime_prefill_shape_graph_capture_gpu_ms"),
                capture_shape,
            )
            if capture_shape is not None
            else None
        )
        replay_shape, replay_ms = _top_mapping_entry(
            fields.get("runtime_prefill_shape_graph_replay_ms")
        )
        replay_gpu_ms = (
            _mapping_value(
                fields.get("runtime_prefill_shape_graph_replay_gpu_ms"),
                replay_shape,
            )
            if replay_shape is not None
            else None
        )
        if capture_shape is None and replay_shape is None:
            continue
        rows.append(
            (
                _fmt_value(profile.temperature),
                _fmt_value(profile.max_tokens),
                capture_shape or "-",
                _fmt_value(capture_ms),
                _fmt_value(capture_gpu_ms),
                replay_shape or "-",
                _fmt_value(replay_ms),
                _fmt_value(replay_gpu_ms),
                _fmt_value(fields.get("runtime_prefill_graph_cache_live_entries")),
            )
        )
    return rows


def _hot_decode_shape_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        decode_shape, decode_gpu_ms = _top_mapping_entry(
            fields.get("runtime_decode_shape_gpu_ms")
        )
        decode_cpu_ms = (
            _mapping_value(fields.get("runtime_decode_shape_cpu_tokens_ms"), decode_shape)
            if decode_shape is not None
            else None
        )
        decode_many_shape, decode_many_gpu_ms = _top_mapping_entry(
            fields.get("runtime_decode_many_shape_gpu_ms")
        )
        overgen_shape, overgen_tokens = _top_mapping_entry(
            fields.get("runtime_decode_many_shape_overgenerated_tokens")
        )
        if decode_shape is None and decode_many_shape is None and overgen_shape is None:
            continue
        decode_many_key = decode_many_shape or overgen_shape
        if decode_many_gpu_ms is None and decode_many_key is not None and decode_many_key == decode_shape:
            decode_many_gpu_ms = decode_gpu_ms
        decode_many_model_tokens = (
            _mapping_value(
                fields.get("runtime_decode_many_shape_model_tokens"),
                decode_many_key,
            )
            if decode_many_key is not None
            else None
        )
        decode_many_emitted_tokens = (
            _mapping_value(
                fields.get("runtime_decode_many_shape_emitted_tokens"),
                decode_many_key,
            )
            if decode_many_key is not None
            else None
        )
        decode_many_skipped_tokens = (
            _mapping_value(
                fields.get("runtime_decode_many_shape_skipped_tokens"),
                decode_many_key,
            )
            if decode_many_key is not None
            else None
        )
        decode_many_overgen_tokens = (
            _mapping_value(
                fields.get("runtime_decode_many_shape_overgenerated_tokens"),
                decode_many_key,
            )
            if decode_many_key is not None
            else None
        )
        rows.append(
            (
                _fmt_value(profile.temperature),
                _fmt_value(profile.max_tokens),
                decode_shape or "-",
                _fmt_value(decode_gpu_ms),
                _fmt_value(decode_cpu_ms),
                decode_many_key or "-",
                _fmt_value(decode_many_gpu_ms),
                _fmt_value(decode_many_model_tokens),
                _fmt_value(decode_many_emitted_tokens),
                _fmt_value(decode_many_skipped_tokens),
                _fmt_pct(
                    float(decode_many_skipped_tokens or 0),
                    float(decode_many_model_tokens or 0),
                ),
                _fmt_value(decode_many_overgen_tokens),
                _fmt_value(fields.get("runtime_decode_graph_cache_live_entries")),
            )
        )
    return rows


def _decode_many_step_window_rows(
    profiles: Sequence[QueueProfileSummary],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for profile in profiles:
        fields = profile.fields
        timing_entries = _top_mapping_entries(
            fields.get("runtime_decode_many_step_window_model_ms"),
            limit=5,
        )
        has_timing_entries = bool(timing_entries)
        entries = timing_entries
        if not entries:
            entries = _top_mapping_entries(
                fields.get("runtime_decode_many_step_window_model_tokens"),
                limit=5,
            )
        for key, entry_value in entries:
            model_ms = entry_value if has_timing_entries else None
            model_tokens = _mapping_value(
                fields.get("runtime_decode_many_step_window_model_tokens"),
                key,
            )
            padded_tokens = _mapping_value(
                fields.get("runtime_decode_many_step_window_padded_tokens"),
                key,
            )
            emitted_tokens = _mapping_value(
                fields.get("runtime_decode_many_step_window_emitted_tokens"),
                key,
            )
            skipped_tokens = _mapping_value(
                fields.get("runtime_decode_many_step_window_skipped_tokens"),
                key,
            )
            cpu_ms = _mapping_value(
                fields.get("runtime_decode_many_step_window_cpu_tokens_ms"),
                key,
            )
            rows.append(
                (
                    _fmt_value(profile.temperature),
                    _fmt_value(profile.max_tokens),
                    key,
                    _fmt_value(
                        _mapping_value(
                            fields.get("runtime_decode_many_step_window_counts"),
                            key,
                        )
                    ),
                    _fmt_value(model_tokens),
                    _fmt_value(padded_tokens),
                    _fmt_value(emitted_tokens),
                    _fmt_value(skipped_tokens),
                    _fmt_pct(
                        float(skipped_tokens or 0),
                        float(model_tokens or 0),
                    ),
                    _fmt_value(model_ms),
                    _fmt_value(cpu_ms),
                )
            )
    return rows


def _decode_many_step_window_target_rows(
    profiles: Sequence[QueueProfileSummary],
    *,
    limit: int = 8,
) -> list[tuple[str, ...]]:
    items: list[tuple[float, float, str, tuple[str, ...]]] = []
    for profile in profiles:
        fields = profile.fields
        window_tokens = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_model_tokens")
        )
        if not window_tokens:
            continue
        shape_tokens = _numeric_mapping(
            fields.get("runtime_decode_many_shape_model_tokens")
        )
        shape_gpu_ms = _numeric_mapping(
            fields.get("runtime_decode_many_shape_gpu_ms")
        )
        window_model_ms = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_model_ms")
        )
        window_counts = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_counts")
        )
        window_emitted = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_emitted_tokens")
        )
        window_skipped = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_skipped_tokens")
        )
        window_cpu_ms = _numeric_mapping(
            fields.get("runtime_decode_many_step_window_cpu_tokens_ms")
        )
        for window, raw_tokens in window_tokens.items():
            model_tokens = max(0.0, float(raw_tokens))
            if model_tokens <= 0.0:
                continue
            shape = _decode_many_step_window_shape(window)
            shape_model_tokens = float(shape_tokens.get(shape, 0.0))
            shape_ms = float(shape_gpu_ms.get(shape, 0.0))
            gpu_ms: float | None = window_model_ms.get(window)
            gpu_src = "exact" if gpu_ms is not None else "est"
            if gpu_ms is None and shape_model_tokens > 0.0 and shape_ms > 0.0:
                gpu_ms = shape_ms * model_tokens / shape_model_tokens
            us_tok: float | None = None
            if gpu_ms is not None:
                us_tok = gpu_ms * 1000.0 / model_tokens
            cpu_ms = window_cpu_ms.get(window)
            total_ms: float | None = None
            if gpu_ms is not None or cpu_ms is not None:
                total_ms = (gpu_ms or 0.0) + (cpu_ms or 0.0)
            skipped = float(window_skipped.get(window, 0.0))
            score_ms = total_ms if total_ms is not None else -1.0
            items.append(
                (
                    score_ms,
                    model_tokens,
                    window,
                    (
                        _fmt_value(profile.temperature),
                        _fmt_value(profile.max_tokens),
                        window,
                        _fmt_value(_int_if_whole(window_counts.get(window, 0.0))),
                        _fmt_value(_int_if_whole(model_tokens)),
                        _fmt_value(_int_if_whole(window_emitted.get(window, 0.0))),
                        _fmt_value(_int_if_whole(skipped)),
                        _fmt_pct(skipped, model_tokens),
                        _fmt_value(None if gpu_ms is None else _int_if_whole(gpu_ms)),
                        gpu_src if gpu_ms is not None else "-",
                        _fmt_value(
                            None
                            if cpu_ms is None
                            else _int_if_whole(cpu_ms)
                        ),
                        _fmt_value(
                            None
                            if total_ms is None
                            else _int_if_whole(total_ms)
                        ),
                        _fmt_value(None if us_tok is None else _int_if_whole(us_tok)),
                    ),
                )
            )
    items.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in items[:limit]]


def _decode_many_step_window_shape(window: str) -> str:
    shape, sep, suffix = window.rpartition(":")
    if sep and suffix.startswith("g"):
        return shape
    return window


def _top_mapping_entry(value: Any) -> tuple[str | None, float | int | None]:
    entries = _top_mapping_entries(value, limit=1)
    if not entries:
        return None, None
    return entries[0]


def _top_mapping_entries(value: Any, *, limit: int) -> list[tuple[str, float | int]]:
    if not isinstance(value, dict) or not value or limit <= 0:
        return []
    entries: list[tuple[str, float | int]] = []
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            entries.append((str(key), raw))
    entries.sort(key=lambda item: (-float(item[1]), item[0]))
    return entries[:limit]


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(raw)
        for key, raw in value.items()
        if isinstance(raw, (int, float))
    }


def _numeric_mapping_preserving_type(value: Any) -> dict[str, float | int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): raw
        for key, raw in value.items()
        if isinstance(raw, (int, float))
    }


def _numeric_field(fields: dict[str, Any], name: str) -> float | None:
    raw = fields.get(name)
    return float(raw) if isinstance(raw, (int, float)) else None


def _mapping_value(value: Any, key: str) -> float | int | None:
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    return raw if isinstance(raw, (int, float)) else None


def _format_table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    formatted = ["  " + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(header))]
    formatted.append("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        formatted.append("  " + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return formatted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torchinferno.research.inference_bench",
        description="Summarize an inference-bench run directory.",
    )
    parser.add_argument("run_dir", help="Path to an inference-bench run directory containing results.json.")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Benchmark name to include. Repeat to include multiple; defaults to all.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="Provider name to include. Repeat to include multiple; defaults to all.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_inference_bench_run(
        args.run_dir,
        benchmarks=tuple(args.benchmark) if args.benchmark else None,
        providers=tuple(args.provider) if args.provider else None,
    )
    print(format_inference_bench_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
