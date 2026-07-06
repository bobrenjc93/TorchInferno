import json
import os
import subprocess
import sys

from torchinferno.research.inference_bench import (
    _decode_graph_cache_counts,
    _decode_graph_symm_counts,
    _phase_target,
    _prefill_shape_call_count,
    _prefill_shape_dense_forward_ms,
    format_inference_bench_summary,
    summarize_inference_bench_run,
)


def _write_inference_bench_run(tmp_path) -> None:
    results = {
        "model": "meta-llama/test",
        "tensor_parallel_size": 8,
        "hardware": "8xH100",
        "providers": {
            "torchinferno": {
                "benchmarks": {
                    "long_output": {
                        "metrics": {
                            "ttft_median_ms": 20.0,
                            "tpot_median_ms": 4.0,
                            "e2e_median_ms": 80.0,
                            "throughput_median_tps": 30.0,
                            "correctness_rate": 1.0,
                        },
                        "raw_requests": [
                            {
                                "request_idx": 0,
                                "ttft_ms": 10.0,
                                "tpot_ms": 3.0,
                                "e2e_latency_ms": 70.0,
                                "throughput_tps": 40.0,
                                "output_tokens": 4,
                            },
                            {
                                "request_idx": 65,
                                "ttft_ms": 30.0,
                                "tpot_ms": 5.0,
                                "e2e_latency_ms": 90.0,
                                "throughput_tps": 20.0,
                                "output_tokens": 6,
                            },
                        ],
                    }
                }
            },
            "vllm": {
                "benchmarks": {
                    "long_output": {
                        "metrics": {"ttft_median_ms": 12.0},
                        "raw_requests": [
                            {
                                "request_idx": 0,
                                "ttft_ms": 8.0,
                                "tpot_ms": 2.0,
                                "e2e_latency_ms": 50.0,
                                "throughput_tps": 60.0,
                                "output_tokens": 4,
                            },
                            {
                                "request_idx": 65,
                                "ttft_ms": 16.0,
                                "tpot_ms": 4.0,
                                "e2e_latency_ms": 70.0,
                                "throughput_tps": 30.0,
                                "output_tokens": 6,
                            },
                        ],
                    }
                }
            },
        },
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    logs = tmp_path / "provider_logs"
    logs.mkdir()
    queue_record = {
        "event": "online_batcher_quiescent",
        "temperature": 0.0,
        "run_max_tokens": 96,
        "submitted_requests": 2,
        "finished_events": 2,
        "runtime_max_active_requests": 96,
        "runtime_prefix_cache_capacity": 128,
        "greedy_large_mixed_prefix_reuse": True,
        "fp8_prefill_enabled": True,
        "fp8_prefill_min_m": 512,
        "marlin_int4_decode_enabled": True,
        "use_decode_many": True,
        "decode_quantum": 3,
        "drain_decode_quantum": 8,
        "admit_per_step_cap": 64,
        "admit_min_free_rows": 4,
        "admit_min_ready_requests": 12,
        "prefill_ready_before_decode": True,
        "prefill_ready_before_decode_active_cap": 8,
        "request_queue_to_first_token_p50_ms": 11.0,
        "request_queue_to_submit_p50_ms": 4.0,
        "request_submit_to_first_token_p50_ms": 7.0,
        "active_ready_wait_ms": 0.5,
        "runtime_prefill_batches": 1,
        "runtime_prefill_forward_ms": 12.0,
        "runtime_prefill_wall_ms": 13.0,
        "runtime_prefill_setup_ms": 1.5,
        "runtime_prefill_copy_ms": 2.5,
        "runtime_prefill_sample_ms": 3.5,
        "runtime_prefill_state_ms": 4.5,
        "runtime_prefill_packed_flashinfer_calls": 3,
        "runtime_prefill_packed_flashinfer_ms": 4.0,
        "runtime_prefill_packed_flashinfer_saved_tokens": 5,
        "runtime_prefill_packed_eager_calls": 6,
        "runtime_prefill_packed_eager_ms": 7.0,
        "runtime_prefill_packed_eager_saved_tokens": 8,
        "runtime_prefill_packed_candidate_calls": 4,
        "runtime_prefill_packed_candidate_saved_tokens": 20,
        "runtime_prefill_packed_candidate_groups": 11,
        "runtime_prefill_suffix_split_accepted_calls": 2,
        "runtime_prefill_suffix_split_rejected_calls": 1,
        "runtime_prefill_suffix_split_accepted_saved_tokens": 18,
        "runtime_prefill_suffix_split_accepted_fragments": 5,
        "runtime_prefill_graph_captures": 2,
        "runtime_prefill_graph_capture_ms": 21.0,
        "runtime_prefill_graph_capture_gpu_ms": 18.0,
        "runtime_prefill_graph_replays": 5,
        "runtime_prefill_graph_replay_ms": 6.0,
        "runtime_prefill_graph_replay_gpu_ms": 5.0,
        "runtime_prefill_graph_misses": 9,
        "runtime_prefill_graph_miss_shape_counts": {
            "ragged_prefill:b8:s32:rows1:ctx-128:src8": 6,
            "static_prefill:logits:b1:t56": 3,
        },
        "runtime_prefill_graph_cache_live_entries": 3,
        "runtime_prefill_shape_counts": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 2,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 1,
        },
        "runtime_prefill_shape_forward_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 9.5,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 2.0,
        },
        "runtime_prefill_shape_wall_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 12.5,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 3.0,
        },
        "runtime_prefill_shape_setup_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 1.1,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.2,
        },
        "runtime_prefill_shape_copy_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 1.2,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.3,
        },
        "runtime_prefill_shape_sample_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 1.3,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.4,
        },
        "runtime_prefill_shape_state_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 1.4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.5,
        },
        "runtime_prefill_shape_active_requests": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 3,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 1,
        },
        "runtime_prefill_shape_model_rows": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 2,
        },
        "runtime_prefill_shape_active_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 31,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 4,
        },
        "runtime_prefill_shape_model_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 48,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 8,
        },
        "runtime_prefill_shape_graph_capture_ms": {
            "chunk_graph:b8:s64:p111:logits0": 21.0,
        },
        "runtime_prefill_shape_graph_capture_gpu_ms": {
            "chunk_graph:b8:s64:p111:logits0": 18.0,
        },
        "runtime_prefill_shape_graph_replay_ms": {
            "chunk_graph:b8:s64:p111:logits0": 6.0,
        },
        "runtime_prefill_shape_graph_replay_gpu_ms": {
            "chunk_graph:b8:s64:p111:logits0": 5.0,
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 4.8,
        },
        "runtime_prefill_shape_padding_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 17,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 4,
        },
        "runtime_prefill_shape_row_padding_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 2,
        },
        "runtime_prefill_shape_suffix_padding_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 13,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 2,
        },
        "runtime_prefill_packed_candidate_shape_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 31,
        },
        "runtime_prefill_packed_candidate_shape_model_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 48,
        },
        "runtime_prefill_packed_candidate_shape_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 17,
        },
        "runtime_prefill_packed_candidate_shape_groups": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 2,
        },
        "runtime_prefill_packed_candidate_signature_keys": 2,
        "runtime_prefill_packed_candidate_signature_calls": 4,
        "runtime_prefill_packed_candidate_signature_repeated_keys": 1,
        "runtime_prefill_packed_candidate_signature_repeated_calls": 3,
        "runtime_prefill_packed_candidate_signature_repeated_saved_tokens": 16,
        "runtime_prefill_packed_candidate_signature_counts": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 3,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 1,
        },
        "runtime_prefill_packed_candidate_signature_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 31,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 4,
        },
        "runtime_prefill_packed_candidate_signature_model_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 48,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 8,
        },
        "runtime_prefill_packed_candidate_signature_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 16,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 4,
        },
        "runtime_prefill_packed_candidate_signature_groups": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 2,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 1,
        },
        "runtime_prefill_packed_candidate_pattern_keys": 1,
        "runtime_prefill_packed_candidate_pattern_calls": 4,
        "runtime_prefill_packed_candidate_pattern_repeated_keys": 1,
        "runtime_prefill_packed_candidate_pattern_repeated_calls": 4,
        "runtime_prefill_packed_candidate_pattern_repeated_saved_tokens": 20,
        "runtime_prefill_packed_candidate_pattern_counts": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 4,
            "prefix_graph:b2:s4:p0-0:src0:mixed0|p0:s4": 3,
        },
        "runtime_prefill_packed_candidate_pattern_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 35,
        },
        "runtime_prefill_packed_candidate_pattern_model_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 56,
        },
        "runtime_prefill_packed_candidate_pattern_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 20,
        },
        "runtime_prefill_packed_candidate_pattern_groups": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 2,
        },
        "runtime_prefill_packed_candidate_pattern_slot_counts": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11#p45:s10": 2,
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11#p45:s11": 1,
            "prefix_graph:b2:s4:p0-0:src0:mixed0|p0:s4#p0:s4": 3,
        },
        "runtime_decode_ragged_model_gpu_ms": 14.0,
        "runtime_decode_ragged_cpu_tokens_ms": 1.25,
        "runtime_decode_ragged_state_update_ms": 2.25,
        "runtime_decode_graph_misses": 10,
        "runtime_decode_graph_miss_shape_counts": {
            "static_decode:logits:b2:s55": 7,
            "ragged_decode:token:b8:rows1": 3,
        },
        "runtime_decode_graph_capture_ms": 7.0,
        "runtime_decode_graph_replay_ms": 8.0,
        "runtime_decode_graph_cache_live_cache_bucket_counts": {"cache1024": 3, "cache256": 1},
        "runtime_decode_graph_cache_live_symm_counts": {"symm128": 3, "symm0": 1},
        "runtime_decode_many_model_gpu_ms": 15.0,
        "runtime_decode_many_steps": 6,
        "decode_many_stop_tail_max_steps": 4,
        "decode_many_min_active_pct": 25,
        "decode_many_sync_stops": True,
        "runtime_decode_many_tail_limited_calls": 2,
        "runtime_decode_many_tail_limited_steps": 5,
        "runtime_decode_many_min_active_skips": 6,
        "runtime_decode_many_overgenerated_tokens": 3,
        "runtime_decode_graph_cache_live_entries": 4,
        "runtime_decode_shape_gpu_ms": {"ragged:b8/8": 14.0},
        "runtime_decode_shape_cpu_tokens_ms": {"ragged:b8/8": 1.25},
        "runtime_decode_many_shape_model_tokens": {"decode_many:b8/8": 19},
        "runtime_decode_many_cpu_tokens_ms": 1.5,
        "runtime_decode_many_shape_padded_tokens": {"decode_many:b8/8": 24},
        "runtime_decode_many_shape_emitted_tokens": {"decode_many:b8/8": 15},
        "runtime_decode_many_shape_skipped_tokens": {"decode_many:b8/8": 4},
        "runtime_decode_many_shape_stop_finishes": {"decode_many:b8/8": 3},
        "runtime_decode_many_shape_limit_finishes": {"decode_many:b8/8": 2},
        "runtime_decode_many_shape_gpu_ms": {"decode_many:b8/8": 15.0},
        "runtime_decode_many_shape_overgenerated_tokens": {"decode_many:b8/8": 3},
        "runtime_decode_many_step_window_counts": {
            "decode_many:b8/8:g1-16": 2,
        },
        "runtime_decode_graph_replay_shape_ms": {
            "ragged_decode:token:b8:rows1:cache256:symm128": 1.5,
        },
        "runtime_decode_many_step_window_model_tokens": {
            "decode_many:b8/8:g1-16": 19,
        },
        "runtime_decode_many_step_window_padded_tokens": {
            "decode_many:b8/8:g1-16": 24,
        },
        "runtime_decode_many_step_window_emitted_tokens": {
            "decode_many:b8/8:g1-16": 15,
        },
        "runtime_decode_many_step_window_skipped_tokens": {
            "decode_many:b8/8:g1-16": 4,
        },
        "runtime_decode_many_step_window_model_ms": {
            "decode_many:b8/8:g1-16": 11.5,
        },
        "runtime_decode_many_step_window_cpu_tokens_ms": {
            "decode_many:b8/8:g1-16": 1.2,
        },
        "runtime_generated_prefix_store_requests": 12,
        "runtime_generated_prefix_reuse_requests": 3,
        "runtime_generated_prefix_reuse_tokens": 33,
    }
    (logs / "torchinferno_queue_profile.jsonl").write_text(json.dumps(queue_record) + "\n")
    (logs / "vllm_server.log").write_text(
        "\n".join(
            [
                "INFO Engine 000: Avg prompt throughput: 100.0 tokens/s, "
                "Avg generation throughput: 20.0 tokens/s, Running: 2 reqs, "
                "Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 80.0%",
                "INFO Engine 000: Avg prompt throughput: 300.0 tokens/s, "
                "Avg generation throughput: 40.0 tokens/s, Running: 1 reqs, "
                "Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 90.0%",
            ]
        )
        + "\n"
    )
    (logs / "sglang_server.log").write_text(
        "\n".join(
            [
                "TP0] Prefill batch, #new-seq: 3, #new-token: 48, #cached-token: 12, "
                "token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, "
                "cuda graph: True, input throughput (token/s): 1200.0",
                "TP0] Prefill batch, #new-seq: 1, #new-token: 8, #cached-token: 4, "
                "token usage: 0.00, #running-req: 2, #queue-req: 0, #pending-token: 0, "
                "cuda graph: False, input throughput (token/s): 800.0",
                "TP0] Decode batch, #running-req: 4, #token: 64, token usage: 0.00, "
                "cuda graph: True, gen throughput (token/s): 1600.0, #queue-req: 0",
            ]
        )
        + "\n"
    )


def test_queue_profile_merges_restart_segments(tmp_path) -> None:
    results = {
        "model": "meta-llama/test",
        "tensor_parallel_size": 8,
        "hardware": "8xH100",
        "providers": {
            "torchinferno": {
                "benchmarks": {
                    "long_output": {
                        "metrics": {
                            "num_requests": 5,
                            "ttft_median_ms": 20.0,
                            "tpot_median_ms": 4.0,
                            "e2e_median_ms": 80.0,
                            "throughput_median_tps": 30.0,
                        }
                    }
                }
            }
        },
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    logs = tmp_path / "provider_logs"
    logs.mkdir()
    records = [
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 3,
            "finished_events": 3,
            "request_queue_to_first_token_p50_ms": 11.0,
            "runtime_prefill_batches": 2,
            "runtime_prefill_forward_ms": 10.0,
            "runtime_prefill_shape_counts": {"shape_a": 1},
            "runtime_decode_many_model_gpu_ms": 4.0,
        },
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 2,
            "finished_events": 2,
            "request_queue_to_first_token_p50_ms": 22.0,
            "runtime_prefill_batches": 5,
            "runtime_prefill_forward_ms": 7.0,
            "runtime_prefill_shape_counts": {"shape_a": 2, "shape_b": 1},
            "runtime_decode_many_model_gpu_ms": 6.0,
        },
    ]
    (logs / "torchinferno_queue_profile.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    summary = summarize_inference_bench_run(tmp_path)

    assert len(summary.torchinferno_queue_profiles) == 1
    profile = summary.torchinferno_queue_profiles[0]
    assert profile.segments == 2
    assert profile.submitted_requests == 5
    assert profile.finished_events == 5
    assert profile.fields["runtime_prefill_batches"] == 7
    assert profile.fields["runtime_prefill_forward_ms"] == 17.0
    assert profile.fields["runtime_decode_many_model_gpu_ms"] == 10.0
    assert profile.fields["runtime_prefill_shape_counts"] == {
        "shape_a": 3,
        "shape_b": 1,
    }
    assert "request_queue_to_first_token_p50_ms" not in profile.fields

    text = format_inference_bench_summary(summary)
    assert "5/5 2seg" in text


def test_benchmark_filter_limits_queue_profile_tables(tmp_path) -> None:
    results = {
        "model": "meta-llama/test",
        "tensor_parallel_size": 8,
        "hardware": "8xH100",
        "providers": {
            "torchinferno": {
                "benchmarks": {
                    "long_output": {
                        "metrics": {
                            "num_requests": 1,
                            "ttft_median_ms": 20.0,
                        }
                    },
                    "tree_of_thought": {
                        "metrics": {
                            "num_requests": 1,
                            "ttft_median_ms": 30.0,
                        }
                    },
                }
            }
        },
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    logs = tmp_path / "provider_logs"
    logs.mkdir()
    records = [
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 1,
            "runtime_prefill_shape_counts": {
                "long_output_shape": 1,
            },
            "runtime_prefill_shape_forward_ms": {
                "long_output_shape": 10.0,
            },
        },
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.7,
            "run_max_tokens": 300,
            "submitted_requests": 1,
            "runtime_prefill_shape_counts": {
                "tree_shape": 1,
            },
            "runtime_prefill_shape_forward_ms": {
                "tree_shape": 20.0,
            },
        },
    ]
    (logs / "torchinferno_queue_profile.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    summary = summarize_inference_bench_run(tmp_path, benchmarks=["tree_of_thought"])
    text = format_inference_bench_summary(summary)

    assert "[tree_of_thought]" in text
    assert "0.7" in text
    assert "300" in text
    assert "tree_shape" in text
    assert "long_output_shape" not in text
    assert "96" not in text


def test_prefill_shape_call_count_falls_back_to_graph_counts() -> None:
    fields = {
        "runtime_prefill_shape_graph_replay_counts": {"shape_a": 2},
        "runtime_prefill_shape_graph_capture_counts": {"shape_a": 1, "shape_b": 3},
    }

    assert _prefill_shape_call_count(fields, "shape_a") == 2
    assert _prefill_shape_call_count(fields, "shape_b") == 3

    fields["runtime_prefill_shape_counts"] = {"shape_a": 4}
    assert _prefill_shape_call_count(fields, "shape_a") == 4

    assert _prefill_shape_call_count(
        {
            "runtime_prefill_shape_model_tokens": {
                "prefix_graph:b8:s32:p45-56:src8:mixed1": 256,
            },
        },
        "prefix_graph:b8:s32:p45-56:src8:mixed1",
    ) == 1


def test_inference_bench_summary_parses_provider_and_queue_profiles(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)

    summary = summarize_inference_bench_run(tmp_path)

    assert summary.model == "meta-llama/test"
    assert summary.providers == ("torchinferno", "vllm")
    assert summary.benchmarks == ("long_output",)
    torch_row = next(row for row in summary.provider_benchmarks if row.provider == "torchinferno")
    assert torch_row.request_percentiles["ttft_ms"].p50 == 10.0
    assert torch_row.request_percentiles["ttft_ms"].p90 == 10.0
    assert torch_row.request_percentiles["output_tokens"].maximum == 6.0
    assert len(torch_row.request_waves) == 2
    assert torch_row.request_waves[0].request_start == 0
    assert torch_row.request_waves[0].request_end == 63
    assert torch_row.request_waves[1].request_start == 64
    assert torch_row.request_waves[1].request_end == 127
    assert torch_row.request_waves[1].request_percentiles["ttft_ms"].p90 == 30.0
    assert summary.torchinferno_queue_profiles[0].max_tokens == 96
    assert summary.torchinferno_queue_profiles[0].fields["runtime_max_active_requests"] == 96
    assert summary.torchinferno_queue_profiles[0].fields["runtime_prefix_cache_capacity"] == 128
    assert summary.torchinferno_queue_profiles[0].fields["runtime_prefill_batches"] == 1
    assert summary.torchinferno_queue_profiles[0].fields["runtime_prefill_graph_misses"] == 9
    assert summary.torchinferno_queue_profiles[0].fields[
        "runtime_prefill_graph_miss_shape_counts"
    ] == {
        "ragged_prefill:b8:s32:rows1:ctx-128:src8": 6,
        "static_prefill:logits:b1:t56": 3,
    }
    assert summary.torchinferno_queue_profiles[0].fields["runtime_decode_graph_misses"] == 10
    assert summary.torchinferno_queue_profiles[0].fields[
        "runtime_decode_graph_miss_shape_counts"
    ] == {
        "static_decode:logits:b2:s55": 7,
        "ragged_decode:token:b8:rows1": 3,
    }
    assert (
        summary.torchinferno_queue_profiles[0].fields[
            "runtime_generated_prefix_reuse_tokens"
        ]
        == 33
    )
    assert (
        summary.torchinferno_queue_profiles[0].fields["runtime_prefill_shape_forward_ms"][
            "prefix_graph:b8:s16:p45-45:src1:mixed0"
        ]
        == 9.5
    )
    assert len(summary.provider_server_logs) == 2
    vllm_log = next(row for row in summary.provider_server_logs if row.provider == "vllm")
    assert vllm_log.prompt_events == 2
    assert vllm_log.prompt_tps_avg == 200.0
    assert vllm_log.generation_tps_max == 40.0
    assert vllm_log.prefix_hit_pct_avg == 85.0
    sglang_log = next(row for row in summary.provider_server_logs if row.provider == "sglang")
    assert sglang_log.prefill_batches == 2
    assert sglang_log.prefill_new_tokens == 56
    assert sglang_log.prefill_cached_tokens == 16
    assert sglang_log.decode_batches == 1
    assert sglang_log.decode_logged_tokens == 64

    text = format_inference_bench_summary(summary)
    assert "[long_output]" in text
    assert "score_tpot" in text
    assert "4.0" in text
    assert "torchinferno" in text
    assert "[long_output raw request waves]" in text
    assert "ttft_p90" in text
    assert "0-63" in text
    assert "64-127" in text
    assert "[torchinferno queue profiles]" in text
    assert "max_active" in text
    assert "prefix_cap" in text
    assert "mixed_prefix" in text
    assert "fp8_prefill" in text
    assert "fp8_min_m" in text
    assert "marlin_decode" in text
    assert "decode_many" in text
    assert "decode_q" in text
    assert "drain_q" in text
    assert "admit_cap" in text
    assert "min_free" in text
    assert "min_ready" in text
    assert "prefill_ready" in text
    assert "ready_cap" in text
    assert "coverage" in text
    assert "2/2" in text
    assert "[torchinferno score targets]" in text
    assert "phase_target" in text
    assert "capture" in text
    assert "[torchinferno prefill graph miss shapes]" in text
    assert "static_prefill:logits:b1:t56" in text
    assert "[torchinferno decode graph miss shapes]" in text
    assert "static_decode:logits:b2:s55" in text
    assert "q2submit" in text
    assert "submit2first" in text
    assert "prefill_pad" in text
    assert "prefill_pad_pct" in text
    assert "suffix_split_ok" in text
    assert "suffix_split_saved" in text
    assert "suffix_split_frags" in text
    assert "active_wait" in text
    assert "hot_prefill" in text
    assert "b8:s16:p45-45:src1:mixed0" in text
    assert "hot_decode" in text
    assert "packed_fi_ms" in text
    assert "packed_fi_saved" in text
    assert "packed_eager_ms" in text
    assert "packed_eager_saved" in text
    assert "packed_cand_saved" in text
    assert "prefill_setup_ms" in text
    assert "prefill_copy_ms" in text
    assert "prefill_sample_ms" in text
    assert "gpu_ms_call" in text
    assert "gpu_us_tok" in text
    assert "pad_call" in text
    assert "2.4" in text
    assert "100.0" in text
    assert "8.5" in text
    assert "37.5%" in text
    assert "wall_ms" in text
    assert "sample_ms" in text
    assert "prefill_row_pad" in text
    assert "prefill_sfx_pad" in text
    assert "prefill_miss" in text
    assert "decode_miss" in text
    assert "decode_miss_kind" in text
    assert "static_logits=7,ragged_token=3" in text
    assert "decode_graph_cache" in text
    assert "cache1024=3,cache256=1" in text
    assert "decode_replay_cache_ms" in text
    assert "cache256=1.5" in text
    assert "decode_graph_symm" in text
    assert "symm128=3" in text
    assert "decode_replay_symm_ms" in text
    assert "symm128=1.5" in text
    assert "gen_store" in text
    assert "gen_reuse" in text
    assert "gen_tokens" in text
    assert "prefill_graph_miss" in text
    assert "prefill_miss_kind" in text
    assert "ragged=6,static_logits=3" in text
    assert "decode_graph_miss" in text
    assert "[long_output provider gaps vs torchinferno]" in text
    assert "best_provider" in text
    assert "+8.0" in text
    assert "1.67x" in text
    assert "+2.0" in text
    assert "2.00x" in text
    assert "+30.0" in text
    assert "1.60x" in text
    assert "[torchinferno packed prefill candidates]" in text
    assert "[torchinferno packed prefill per-batch targets]" in text
    assert "[torchinferno packed prefill signatures]" in text
    assert "[torchinferno packed prefill patterns]" in text
    assert "[torchinferno packed prefill fixed-capacity plans]" in text
    assert "[torchinferno packed prefill implementation targets]" in text
    assert "[torchinferno packed prefill signature reuse]" in text
    assert "[torchinferno packed prefill pattern reuse]" in text
    assert "p45:s10:n2/p45:s11:n1" in text
    assert "p45:s10/p45:s11" in text
    assert "slot_src" in text
    assert "runtime" in text
    assert "est_saved_ms" in text
    assert "fixed_saved_pct" in text
    assert "repeat_saved" in text
    assert "sig_cov" in text
    assert "75.8%" in text
    assert "prefix_graph:b2:s4:p0-0:src0:mixed0|p0:s4" not in text
    assert "saved_pct" in text
    assert "pattern_keys" in text
    assert "repeat_call_pct" in text
    assert "100.0%" in text
    assert "75.0%" in text
    assert "80.0%" in text
    assert "saved_tokens" in text
    assert "row_saved" in text
    assert "suffix_saved" in text
    assert "est_share" in text
    assert "28.0%" in text
    assert "obs_packed_ms" in text
    assert "prefill_graph_cap_ms" in text
    assert "prefill_graph_cap_gpu_ms" in text
    assert "prefill_graph_replay_gpu_ms" in text
    assert "decode_graph_cap_ms" in text
    assert "decode_cpu_ms" in text
    assert "decode_state_ms" in text
    assert "decode_many_calls" in text
    assert "decode_many_cpu_ms" in text
    assert "1.2" in text
    assert "1.5" in text
    assert "2.2" in text
    assert "tail_cap" in text
    assert "sync_stops" in text
    assert "True" in text
    assert "tail_calls" in text
    assert "tail_steps" in text
    assert "min_active_pct" in text
    assert "min_active_skips" in text
    assert "overgen" in text
    assert "[torchinferno hot prefill shapes]" in text
    assert "prefix_graph:b8:s16:p45-45:src1:mixed0" in text
    assert "prefix_graph:b4:s16:p45-45:src1:mixed0" in text
    assert "calls" in text
    assert "active_tokens" in text
    assert "pad_pct" in text
    assert "35.4%" in text
    assert "row_pad" in text
    assert "suffix_pad" in text
    assert "graph_gpu_ms" in text
    assert "[torchinferno hot prefill graph shapes]" in text
    assert "capture_gpu_ms" in text
    assert "replay_gpu_ms" in text
    assert "chunk_graph:b8:s64:p111:logits0" in text
    assert "[torchinferno hot decode shapes]" in text
    assert "decode_cpu_ms" in text
    assert "decode_many:b8/8" in text
    assert "skip_pct" in text
    assert "21.1%" in text
    assert "[torchinferno decode-many step windows]" in text
    assert "[torchinferno decode-many implementation targets]" in text
    assert "gpu_ms" in text
    assert "gpu_src" in text
    assert "tok_share" in text
    assert "total_share" in text
    assert "us_tok" in text
    assert "605.3" in text
    assert "77.0%" in text
    assert "decode_many:b8/8:g1-16" in text
    assert "exact" in text
    assert "model_ms" in text
    assert "cpu_ms" in text
    assert "total_ms" in text
    assert "12.7" in text
    assert "[provider server log phases]" in text
    assert "prefix_hit_avg" in text
    assert "prefill_graph_pct" in text
    assert "decode_graph_pct" in text
    assert "sglang" in text
    assert "85.0%" in text
    assert "50.0%" in text


def test_inference_bench_summary_reads_current_provider_log_names(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)
    logs = tmp_path / "provider_logs"
    (logs / "vllm_server.log").rename(logs / "vllm.log")
    (logs / "sglang_server.log").rename(logs / "sglang.log")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))

    assert "[provider server log phases]" in text
    assert "vllm" in text
    assert "sglang" in text
    assert "prefix_hit_avg" in text
    assert "prefill_graph_pct" in text
    assert "decode_graph_pct" in text


def test_inference_bench_summary_marks_partial_queue_profiles(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)
    queue_path = tmp_path / "provider_logs" / "torchinferno_queue_profile.jsonl"
    queue_record = json.loads(queue_path.read_text())
    queue_record["submitted_requests"] = 1
    queue_record["finished_events"] = 1
    queue_path.write_text(json.dumps(queue_record) + "\n")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))

    assert "coverage" in text
    assert "1/2 partial" in text


def test_phase_target_can_select_sampling() -> None:
    assert (
        _phase_target(
            prefill_ms=0.0,
            decode_ms=0.0,
            sample_ms=71.3,
            capture_ms=0.0,
        )
        == "sample"
    )
    assert (
        _phase_target(
            prefill_ms=100.0,
            decode_ms=80.0,
            sample_ms=5.0,
            capture_ms=0.0,
        )
        == "prefill+decode"
    )


def test_packed_prefill_estimate_excludes_observed_packed_cost() -> None:
    fields = {
        "runtime_prefill_shape_forward_ms": {"prefix_graph:b4:s16:p45-45:src1:mixed0": 10.0},
        "runtime_prefill_packed_eager_shape_ms": {
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 3.5,
        },
    }

    assert (
        _prefill_shape_dense_forward_ms(
            fields,
            "prefix_graph:b4:s16:p45-45:src1:mixed0",
        )
        == 6.5
    )
    assert (
        _prefill_shape_dense_forward_ms(
            {"runtime_prefill_shape_forward_ms": {"shape": 2.0}},
            "shape",
        )
        == 2.0
    )
    assert (
        _prefill_shape_dense_forward_ms(
            {
                "runtime_prefill_shape_forward_ms": {"shape": 2.0},
                "runtime_prefill_packed_eager_shape_ms": {"shape": 3.0},
            },
            "shape",
        )
        == 0.0
    )


def test_inference_bench_summary_derives_decode_graph_symm_counts_from_shapes() -> None:
    assert _decode_graph_symm_counts(
        {
            "runtime_decode_graph_cache_live_shape_counts": {
                "ragged_decode:b8:ctx4096:cache1024:rows1:symm128": 2,
                "ragged_decode:b4:ctx4096:cache1024:rows1:symm128": 1,
                "ragged_decode:b1:ctx4096:cache1024:rows1:symm0": 1,
            },
        }
    ) == {"symm128": 3, "symm0": 1}
    assert _decode_graph_symm_counts(
        {
            "runtime_decode_graph_cache_live_shape_counts": {
                "ragged_decode:b8:ctx4096:cache1024:rows1:symm0": 9,
            },
            "runtime_decode_graph_cache_live_symm_counts": {"symm128": 4},
        }
    ) == {"symm128": 4}


def test_inference_bench_summary_derives_decode_graph_cache_counts_from_shapes() -> None:
    assert _decode_graph_cache_counts(
        {
            "runtime_decode_graph_cache_live_shape_counts": {
                "ragged_decode:b8:ctx4096:cache1024:rows1:symm128": 2,
                "ragged_decode:b4:ctx4096:cache256:rows1:symm128": 1,
                "ragged_decode:b1:ctx4096:cache1024:rows1:symm0": 1,
            },
        }
    ) == {"cache1024": 3, "cache256": 1}
    assert _decode_graph_cache_counts(
        {
            "runtime_decode_graph_cache_live_shape_counts": {
                "ragged_decode:b8:ctx4096:cache1024:rows1:symm0": 9,
            },
            "runtime_decode_graph_cache_live_cache_bucket_counts": {"cache256": 4},
        }
    ) == {"cache256": 4}


def test_inference_bench_summary_leaves_decode_many_window_ms_blank_without_timing(
    tmp_path,
) -> None:
    _write_inference_bench_run(tmp_path)
    queue_path = tmp_path / "provider_logs" / "torchinferno_queue_profile.jsonl"
    queue_record = json.loads(queue_path.read_text())
    queue_record.pop("runtime_decode_many_step_window_model_ms")
    queue_record.pop("runtime_decode_many_step_window_cpu_tokens_ms")
    queue_path.write_text(json.dumps(queue_record) + "\n")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))
    row = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("0.0")
        and "decode_many:b8/8:g1-16" in line
    )

    assert row.split()[-2:] == ["-", "-"]


def test_inference_bench_summary_uses_runtime_slot_counts_without_signatures(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)
    queue_path = tmp_path / "provider_logs" / "torchinferno_queue_profile.jsonl"
    queue_record = json.loads(queue_path.read_text())
    for key in list(queue_record):
        if "packed_candidate_signature" in key:
            queue_record.pop(key)
    queue_path.write_text(json.dumps(queue_record) + "\n")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))

    assert "[torchinferno packed prefill fixed-capacity plans]" in text
    assert "fixed_saved_pct" in text
    assert "75.8%" in text
    assert "0.0%" in text


def test_inference_bench_summary_falls_back_to_signatures_for_fixed_capacity(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)
    queue_path = tmp_path / "provider_logs" / "torchinferno_queue_profile.jsonl"
    queue_record = json.loads(queue_path.read_text())
    queue_record.pop("runtime_prefill_packed_candidate_pattern_slot_counts", None)
    queue_path.write_text(json.dumps(queue_record) + "\n")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))

    assert "[torchinferno packed prefill fixed-capacity plans]" in text
    assert "slot_src" in text
    assert "signature" in text
    assert "75.8%" in text


def test_inference_bench_summary_cli_filters_benchmarks(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "inference-bench-summary",
            str(tmp_path),
            "--benchmark",
            "long_output",
            "--provider",
            "vllm",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "TorchInferno inference-bench summary" in result.stdout
    assert "vllm" in result.stdout
    assert "torchinferno" not in result.stdout.split("[long_output]", maxsplit=1)[1].splitlines()[2]


def test_inference_bench_summary_module_cli(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.research.inference_bench",
            str(tmp_path),
            "--benchmark",
            "long_output",
            "--provider",
            "torchinferno",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "TorchInferno inference-bench summary" in result.stdout
    assert "[long_output]" in result.stdout
    assert "torchinferno" in result.stdout
    assert "RuntimeWarning" not in result.stderr
