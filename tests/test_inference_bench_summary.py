import json
import os
import subprocess
import sys

from torchinferno.research.inference_bench import (
    QueueProfileSummary,
    _cache_integrity_rows,
    _decode_graph_cache_counts,
    _decode_graph_symm_counts,
    _decode_many_graph_policy_rows,
    _decode_many_phase_rows,
    _hot_prefill_shape_rows,
    _phase_target,
    _prefill_graph_phase_rows,
    _prefill_packed_dynamic_target_rows,
    _prefill_packed_flashinfer_gate_rows,
    _prefill_packed_fixed_capacity_runtime_rows,
    _prefill_packed_fixed_capacity_reject_rows,
    _prefill_non_fragmenting_target_rows,
    _prefill_shape_call_count,
    _prefill_shape_dense_forward_ms,
    _prefill_target_ms,
    _top_prefill_target_entry,
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
                "commit_hash": "abcdef1234567890",
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
                "commit_hash": "123456abcdef0000",
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
        "runtime_cache_backend": "dense",
        "runtime_max_active_requests": 96,
        "runtime_prefix_cache_capacity": 128,
        "greedy_large_mixed_prefix_reuse": True,
        "fp8_prefill_enabled": True,
        "fp8_prefill_min_m": 512,
        "marlin_int4_decode_enabled": True,
        "use_decode_many": True,
        "decode_many_graph": True,
        "decode_many_graph_min_steps": 2,
        "decode_quantum": 8,
        "drain_decode_quantum": 8,
        "admit_per_step_cap": 64,
        "admit_min_free_rows": 4,
        "admit_min_ready_requests": 8,
        "prefill_ready_before_decode": True,
        "prefill_ready_before_decode_active_cap": 6,
        "request_queue_to_first_token_p50_ms": 11.0,
        "request_queue_to_submit_p50_ms": 4.0,
        "request_submit_to_first_token_p50_ms": 7.0,
        "request_first_token_prefill_shape_counts": {"shape_a": 2},
        "request_first_token_prefill_shape_queue_to_submit_counts": {"shape_a": 2},
        "request_first_token_prefill_shape_queue_to_submit_p50_ms": {"shape_a": 4.0},
        "request_first_token_prefill_shape_queue_to_first_counts": {"shape_a": 2},
        "request_first_token_prefill_shape_queue_to_first_p50_ms": {"shape_a": 11.0},
        "request_first_token_prefill_shape_queue_to_first_p90_ms": {"shape_a": 13.0},
        "request_first_token_prefill_shape_queue_to_first_p99_ms": {"shape_a": 15.0},
        "request_first_token_prefill_shape_submit_to_first_counts": {"shape_a": 2},
        "request_first_token_prefill_shape_submit_to_first_p50_ms": {"shape_a": 7.0},
        "initial_wait_ms": 1.0,
        "idle_batch_wait_ms": 2.0,
        "active_ready_wait_ms": 0.5,
        "decode_capture_on_miss": False,
        "runtime_prefill_batches": 1,
        "runtime_prefill_forward_ms": 12.0,
        "runtime_prefill_wall_ms": 13.0,
        "runtime_prefill_setup_ms": 1.5,
        "runtime_prefill_copy_ms": 2.5,
        "runtime_prefill_sample_ms": 3.5,
        "runtime_prefill_sample_select_ms": 1.6,
        "runtime_prefill_sample_readback_ms": 1.7,
        "runtime_temperature_sample_calls": 2,
        "runtime_temperature_sample_rows": 16,
        "runtime_temperature_sample_total_ms": 1.4,
        "runtime_temperature_sample_max_ms": 0.2,
        "runtime_temperature_sample_weights_ms": 0.3,
        "runtime_temperature_sample_rank_ms": 0.4,
        "runtime_temperature_sample_cdf_ms": 0.35,
        "runtime_temperature_sample_reduce_ms": 0.15,
        "runtime_temperature_sample_gumbel_calls": 1,
        "runtime_temperature_sample_gumbel_rows": 8,
        "runtime_temperature_sample_gumbel_ms": 0.9,
        "runtime_temperature_sample_gumbel_noise_ms": 0.25,
        "runtime_temperature_sample_gumbel_max_ms": 0.45,
        "runtime_temperature_sample_gumbel_reduce_ms": 0.2,
        "runtime_prefill_state_ms": 4.5,
        "runtime_prefill_state_seq_ms": 1.8,
        "runtime_prefill_state_store_ms": 1.9,
        "runtime_prefill_state_create_ms": 2.0,
        "runtime_prefill_packed_flashinfer_calls": 3,
        "runtime_prefill_packed_flashinfer_ms": 4.0,
        "runtime_prefill_packed_flashinfer_saved_tokens": 5,
        "runtime_prefill_packed_eager_calls": 6,
        "runtime_prefill_packed_eager_ms": 7.0,
        "runtime_prefill_packed_eager_saved_tokens": 8,
        "runtime_prefill_packed_candidate_calls": 4,
        "runtime_prefill_packed_candidate_saved_tokens": 20,
        "runtime_prefill_packed_candidate_groups": 11,
        "runtime_prefill_packed_fixed_capacity_attempts": 4,
        "runtime_prefill_packed_fixed_capacity_accepts": 1,
        "runtime_prefill_packed_fixed_capacity_reject_reason_counts": {
            "capacity_grew": 3,
        },
        "runtime_prefix_reuse_requests": 5,
        "runtime_prefix_reuse_tokens": 64,
        "runtime_prefix_reuse_route_counts": {
            "common_prefix": 3,
            "request_prompt": 2,
        },
        "runtime_prefix_reuse_hit_token_counts": {"16": 3, "8": 2},
        "runtime_prefill_suffix_split_candidate_calls": 3,
        "runtime_prefill_suffix_split_candidate_saved_tokens": 42,
        "runtime_prefill_suffix_split_accepted_calls": 2,
        "runtime_prefill_suffix_split_rejected_calls": 1,
        "runtime_prefill_suffix_split_reject_reason_counts": {"disabled": 1},
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
        "runtime_prefill_graph_cache_max_entries": 64,
        "runtime_prefill_graph_cache_evictions": 2,
        "runtime_prefill_graph_cache_evicted_entries": 4,
        "runtime_prefill_graph_cache_live_suffix_counts": {"s12": 14, "s96": 2},
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
        "runtime_prefill_shape_sample_select_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 0.6,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.1,
        },
        "runtime_prefill_shape_sample_readback_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 0.7,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.2,
        },
        "runtime_prefill_shape_state_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 1.4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.5,
        },
        "runtime_prefill_shape_state_seq_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 0.8,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.2,
        },
        "runtime_prefill_shape_state_store_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 0.4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.2,
        },
        "runtime_prefill_shape_state_create_ms": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 0.2,
            "prefix_graph:b4:s16:p45-45:src1:mixed0": 0.1,
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
        "runtime_prefill_packed_candidate_shape_row_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 4,
        },
        "runtime_prefill_packed_candidate_shape_suffix_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 13,
        },
        "runtime_prefill_packed_candidate_shape_groups": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 2,
        },
        "runtime_prefill_packed_candidate_shape_max_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 10,
        },
        "runtime_prefill_packed_candidate_shape_max_model_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 16,
        },
        "runtime_prefill_packed_candidate_shape_max_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0": 6,
        },
        "runtime_prefill_packed_candidate_shape_max_groups": {
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
        "runtime_prefill_packed_candidate_signature_row_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 4,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 2,
        },
        "runtime_prefill_packed_candidate_signature_suffix_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10:n2/p45:s11:n1": 12,
            "prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s12:n1": 2,
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
        "runtime_prefill_packed_candidate_pattern_row_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 4,
        },
        "runtime_prefill_packed_candidate_pattern_suffix_saved_tokens": {
            "prefix_graph:b8:s16:p45-45:src1:mixed0|p45:s10/p45:s11": 16,
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
        "runtime_decode_many_calls": 2,
        "runtime_decode_many_model_gpu_ms": 15.0,
        "runtime_decode_many_steps": 6,
        "runtime_decode_many_model_tokens": 24,
        "runtime_decode_many_emitted_tokens": 15,
        "runtime_decode_many_graph_calls": 1,
        "runtime_decode_many_graph_steps": 3,
        "runtime_decode_many_graph_model_tokens": 12,
        "runtime_decode_many_graph_ms": 4.5,
        "runtime_decode_many_state_syncs": 1,
        "runtime_decode_many_state_sync_skips": 3,
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
        "runtime_decode_many_shape_steps": {"decode_many:b8/8": 6},
        "runtime_decode_many_shape_model_tokens": {"decode_many:b8/8": 19},
        "runtime_decode_many_cpu_tokens_ms": 1.5,
        "runtime_decode_many_token_wait_ms": 1.2,
        "runtime_decode_many_token_materialize_ms": 0.3,
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
        "runtime_decode_many_step_window_stop_finishes": {
            "decode_many:b8/8:g1-16": 3,
        },
        "runtime_decode_many_step_window_limit_finishes": {
            "decode_many:b8/8:g1-16": 2,
        },
        "runtime_decode_many_step_window_model_ms": {
            "decode_many:b8/8:g1-16": 11.5,
        },
        "runtime_decode_many_step_window_cpu_tokens_ms": {
            "decode_many:b8/8:g1-16": 1.2,
        },
        "runtime_decode_many_step_window_token_wait_ms": {
            "decode_many:b8/8:g1-16": 1.0,
        },
        "runtime_decode_many_step_window_token_materialize_ms": {
            "decode_many:b8/8:g1-16": 0.2,
        },
        "runtime_prompt_lookup_batches": 2,
        "runtime_prompt_lookup_requests": 3,
        "runtime_prompt_lookup_proposed_tokens": 10,
        "runtime_prompt_lookup_accepted_tokens": 4,
        "runtime_generated_prefix_cache_requested": True,
        "runtime_generated_prefix_cache_base_enabled": True,
        "runtime_generated_prefix_cache_effective_enabled": False,
        "runtime_prompt_lookup_decode_effective_enabled": True,
        "runtime_prefix_cache_store_logits_enabled": True,
        "runtime_pinned_prefix_cache_store_logits_enabled": False,
        "runtime_reusable_prefix_entries": 5,
        "runtime_reusable_prefix_logits_entries": 2,
        "runtime_reusable_prefix_logits_tokens": 34,
        "runtime_reusable_prefix_sample_state_entries": 1,
        "runtime_reusable_prefix_greedy_token_entries": 1,
        "runtime_generated_prefix_store_requests": 12,
        "runtime_generated_prefix_reuse_requests": 3,
        "runtime_generated_prefix_reuse_tokens": 33,
        "runtime_repeated_sample_state_prepares": 5,
        "runtime_repeated_sample_state_hits": 1,
        "runtime_repeated_sample_state_tokens": 7,
        "runtime_full_prompt_store_skipped_requests": 2,
        "runtime_full_prompt_store_skipped_tokens": 40,
        "runtime_full_prompt_store_skip_reason_counts": {
            "pinned_without_allowance": 2,
        },
        "runtime_full_prompt_store_skip_reason_tokens": {
            "pinned_without_allowance": 40,
        },
        "runtime_full_prompt_reuse_candidate_stored_requests": 3,
        "runtime_full_prompt_reuse_candidate_stored_tokens": 60,
        "runtime_full_prompt_reuse_candidate_requests": 2,
        "runtime_full_prompt_reuse_candidate_tokens": 44,
        "runtime_full_prompt_reuse_candidate_extra_tokens": 12,
        "runtime_full_prompt_reuse_candidate_suffix_tokens": 4,
        "runtime_full_prompt_reuse_candidate_token_counts": {"22": 2},
        "runtime_full_prompt_reuse_candidate_extra_token_counts": {"6": 2},
        "runtime_full_prompt_reuse_candidate_suffix_token_counts": {"2": 2},
        "runtime_persistent_full_prompt_reuse_candidate_stored_requests": 1,
        "runtime_persistent_full_prompt_reuse_candidate_stored_tokens": 24,
        "runtime_persistent_full_prompt_reuse_candidate_requests": 1,
        "runtime_persistent_full_prompt_reuse_candidate_tokens": 21,
        "runtime_persistent_full_prompt_reuse_candidate_extra_tokens": 7,
        "runtime_persistent_full_prompt_reuse_candidate_suffix_tokens": 1,
        "runtime_persistent_full_prompt_reuse_candidate_token_counts": {"21": 1},
        "runtime_persistent_full_prompt_reuse_candidate_extra_token_counts": {"7": 1},
        "runtime_persistent_full_prompt_reuse_candidate_suffix_token_counts": {"1": 1},
    }
    (logs / "torchinferno_queue_profile.jsonl").write_text(json.dumps(queue_record) + "\n")
    (logs / "torchinferno.log").write_text(
        "\n".join(
            [
                "[WARMUP] tensor-parallel startup warmup start cache_backend=dense "
                "prompt_counts=(32, 16, 64, 128, 256) new_tokens=2",
                "[WARMUP] online decode graph warmup start temperature=0 "
                "max_tokens=1 batches=(1, 2, 4, 8)",
                "[WARMUP] online decode graph warmup done temperature=0 "
                "max_tokens=1 in 35.8s",
                "[WARMUP] greedy common-prefix suffix warmup start temperature=0 "
                "max_tokens=128 row=48 prefixes=3 suffix_pairs=9 batches=7 "
                "shapes=63 token_graphs=True",
                "[WARMUP] greedy common-prefix suffix warmup done in 102.3s",
                "[WARMUP] tensor-parallel startup warmup done in 219.9s",
                "[RAGGED_PREFILL_REPLAY_PROF] batch=24 suffix=64 match=5 "
                "context_len=-256 src_rows=1 prefix_copy_len=none",
                "ncclDevKernel_AllReduce_Sum_bf16_RING_LL         0.00% "
                "0.000us 0.00% 0.000us 0.000us 24.592ms 29.49% "
                "24.592ms 153.699us 160",
                "nvjet_qqtst_112x128_128x7_2x1_v_bz_coopA_algo2_TNN "
                "0.00% 0.000us 0.00% 0.000us 0.000us 10.669ms 12.79% "
                "10.669ms 133.357us 80",
                "_add_rms_norm_kernel                            0.00% "
                "0.000us 0.00% 0.000us 0.000us 7.138ms 8.56% "
                "7.138ms 44.615us 160",
                "void softmax_warp_forward                       0.00% "
                "0.000us 0.00% 0.000us 0.000us 654.337us 0.78% "
                "654.337us 8.179us 80",
                "Self CUDA time total: 83.400ms",
                "[RAGGED_DECODE_MANY_REPLAY_PROF] batch=64 steps=8 match=3 "
                "cache_bucket=1024 rows=64",
                "ncclDevKernel_AllReduce_Sum_bf16_RING_LL         0.00% "
                "0.000us 0.00% 0.000us 0.000us 12.000ms 30.00% "
                "12.000ms 75.000us 160",
                "cutlass_gemm_kernel                             0.00% "
                "0.000us 0.00% 0.000us 0.000us 5.000ms 12.50% "
                "5.000ms 62.500us 80",
                "void marlin::Marlin<test>                       0.00% "
                "0.000us 0.00% 0.000us 0.000us 6.000ms 15.00% "
                "6.000ms 75.000us 80",
                "_grouped_gqa_decode_attention_streaming_kernel  0.00% "
                "0.000us 0.00% 0.000us 0.000us 1.500ms 3.75% "
                "1.500ms 18.750us 80",
                "_add_rms_norm_kernel                            0.00% "
                "0.000us 0.00% 0.000us 0.000us 2.000ms 5.00% "
                "2.000ms 12.500us 160",
                "Self CUDA time total: 40.000ms",
                "[RAGGED_DECODE_REPLAY_PROF] batch=48 match=2 "
                "cache_bucket=512 rows=32",
                "ncclDevKernel_AllReduce_Sum_bf16_RING_LL         0.00% "
                "0.000us 0.00% 0.000us 0.000us 3.000ms 15.00% "
                "3.000ms 18.750us 160",
                "aten::_scaled_mm                                0.00% "
                "0.000us 0.00% 0.000us 0.000us 4.000ms 20.00% "
                "4.000ms 50.000us 80",
                "void marlin::Marlin<test>                       0.00% "
                "0.000us 0.00% 0.000us 0.000us 2.000ms 10.00% "
                "2.000ms 25.000us 80",
                "_grouped_gqa_decode_attention_streaming_kernel  0.00% "
                "0.000us 0.00% 0.000us 0.000us 1.000ms 5.00% "
                "1.000ms 12.500us 80",
                "Self CUDA time total: 20.000ms",
                "[RAGGED_DECODE_MANY_EAGER_PROF] batch=64 steps=1 match=4 "
                "cache_bucket=1024 rows=64 active=64 padded=64",
                "nvjet_tst_64x64_64x13_2x1_v_bz_NNT            0.00% "
                "0.000us 0.00% 0.000us 0.000us 3.500ms 29.17% "
                "3.500ms 21.875us 160",
                "void marlin::Marlin<test>                       0.00% "
                "0.000us 0.00% 0.000us 0.000us 3.250ms 27.08% "
                "3.250ms 40.625us 80",
                "_grouped_gqa_decode_attention_streaming_kernel  0.00% "
                "0.000us 0.00% 0.000us 0.000us 1.250ms 10.42% "
                "1.250ms 15.625us 80",
                "Self CUDA time total: 12.000ms",
            ]
        )
        + "\n"
    )
    (logs / "vllm_server.log").write_text(
        "\n".join(
            [
                "INFO Chunked prefill is enabled with max_num_batched_tokens=8,192.",
                "INFO Asynchronous scheduling is enabled.",
                "INFO config enable_prefix_caching=True enable_chunked_prefill=True "
                "'max_cudagraph_capture_size': 512",
                "INFO GPU KV cache size: 1,451,760 tokens",
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
                "server_args=ServerArgs(disable_radix_cache=False, "
                "chunked_prefill_size=8192, disable_overlap_schedule=False, "
                "num_continuous_decode_steps=1, attention_backend='fa3', "
                "sampling_backend='flashinfer', disable_cuda_graph=False, "
                "disable_decode_cuda_graph=False, disable_prefill_cuda_graph=False, "
                "cuda_graph_config=CudaGraphConfig("
                "decode=PhaseConfig(backend='full', max_bs=512, bs=[1, 2]), "
                "prefill=PhaseConfig(backend='breakable', max_bs=8192, bs=[4, 8])))",
                "TP0] Prefill batch, #new-seq: 3, #new-token: 48, #cached-token: 12, "
                "token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, "
                "cuda graph: True, input throughput (token/s): 1200.0",
                "TP0] Prefill batch, #new-seq: 1, #new-token: 8, #cached-token: 4, "
                "token usage: 0.00, #running-req: 2, #queue-req: 0, #pending-token: 0, "
                "cuda graph: False, input throughput (token/s): 800.0",
                "TP0] Decode batch, #running-req: 4, #token: 64, token usage: 0.00, "
                "cuda graph: True, gen throughput (token/s): 1600.0, #queue-req: 0",
                "TP0] Decode batch, #running-req: 2, #token: 8, token usage: 0.00, "
                "cuda graph: False, gen throughput (token/s): 400.0, #queue-req: 0",
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


def test_queue_profile_merges_final_marker_segments_without_counter_drop(tmp_path) -> None:
    results = {
        "model": "meta-llama/test",
        "tensor_parallel_size": 8,
        "hardware": "8xH100",
        "providers": {
            "torchinferno": {
                "benchmarks": {
                    "long_output": {
                        "metrics": {
                            "num_requests": 1000,
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
            "submitted_requests": 120,
            "finished_events": 120,
            "runtime_prefill_batches": 1,
        },
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 499,
            "finished_events": 499,
            "runtime_prefill_batches": 10,
        },
        {
            "event": "online_batcher",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 499,
            "finished_events": 499,
            "runtime_prefill_batches": 10,
        },
        {
            "event": "online_batcher_quiescent",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 501,
            "finished_events": 501,
            "runtime_prefill_batches": 11,
        },
        {
            "event": "online_batcher",
            "temperature": 0.0,
            "run_max_tokens": 96,
            "submitted_requests": 501,
            "finished_events": 501,
            "runtime_prefill_batches": 11,
        },
    ]
    (logs / "torchinferno_queue_profile.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    summary = summarize_inference_bench_run(tmp_path)

    assert len(summary.torchinferno_queue_profiles) == 1
    profile = summary.torchinferno_queue_profiles[0]
    assert profile.segments == 2
    assert profile.submitted_requests == 1000
    assert profile.finished_events == 1000
    assert profile.fields["runtime_prefill_batches"] == 21

    text = format_inference_bench_summary(summary)
    assert "1000/1000 2seg" in text


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
    assert "0.0   96" not in text


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


def test_hot_prefill_shape_rows_keep_forward_blank_for_padding_ranked_rows() -> None:
    rows = _hot_prefill_shape_rows(
        [
            QueueProfileSummary(
                event="online_batcher",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=1,
                finished_events=1,
                fields={
                    "runtime_prefill_shape_counts": {"shape_a": 2},
                    "runtime_prefill_shape_active_tokens": {"shape_a": 40},
                    "runtime_prefill_shape_model_tokens": {"shape_a": 64},
                    "runtime_prefill_shape_padding_tokens": {"shape_a": 24},
                },
            )
        ]
    )

    assert rows[0][2] == "shape_a"
    assert rows[0][3] == "2"
    assert rows[0][4] == "-"
    assert rows[0][18] == "40"
    assert rows[0][19] == "64"
    assert rows[0][20] == "24"


def test_inference_bench_summary_parses_provider_and_queue_profiles(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)

    summary = summarize_inference_bench_run(tmp_path)

    assert summary.model == "meta-llama/test"
    assert summary.providers == ("torchinferno", "vllm")
    assert summary.benchmarks == ("long_output",)
    torch_row = next(row for row in summary.provider_benchmarks if row.provider == "torchinferno")
    assert torch_row.commit_hash == "abcdef1234567890"
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
    assert summary.torchinferno_queue_profiles[0].fields["runtime_cache_backend"] == "dense"
    assert summary.torchinferno_queue_profiles[0].fields["runtime_max_active_requests"] == 96
    assert summary.torchinferno_queue_profiles[0].fields[
        "request_first_token_prefill_shape_queue_to_submit_p50_ms"
    ] == {"shape_a": 4.0}
    assert summary.torchinferno_queue_profiles[0].fields[
        "request_first_token_prefill_shape_queue_to_first_p50_ms"
    ] == {"shape_a": 11.0}
    assert summary.torchinferno_queue_profiles[0].fields[
        "request_first_token_prefill_shape_submit_to_first_p50_ms"
    ] == {"shape_a": 7.0}
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
        summary.torchinferno_queue_profiles[0].fields[
            "runtime_prompt_lookup_accepted_tokens"
        ]
        == 4
    )
    assert (
        summary.torchinferno_queue_profiles[0].fields[
            "runtime_repeated_sample_state_tokens"
        ]
        == 7
    )
    assert (
        summary.torchinferno_queue_profiles[0].fields["runtime_prefill_shape_forward_ms"][
            "prefix_graph:b8:s16:p45-45:src1:mixed0"
        ]
        == 9.5
    )
    assert len(summary.provider_server_logs) == 2
    vllm_log = next(row for row in summary.provider_server_logs if row.provider == "vllm")
    assert vllm_log.prefix_cache is True
    assert vllm_log.chunked_prefill is True
    assert vllm_log.chunked_prefill_tokens == 8192
    assert vllm_log.async_scheduling is True
    assert vllm_log.decode_cuda_graph is True
    assert vllm_log.decode_cuda_graph_max_bs == 512
    assert vllm_log.kv_cache_tokens == 1451760
    assert vllm_log.prompt_events == 2
    assert vllm_log.prompt_tps_avg == 200.0
    assert vllm_log.generation_tps_max == 40.0
    assert vllm_log.running_max == 2
    assert vllm_log.waiting_max == 0
    assert vllm_log.kv_cache_pct_avg == 0.15000000000000002
    assert vllm_log.kv_cache_pct_max == 0.2
    assert vllm_log.prefix_hit_pct_avg == 85.0
    sglang_log = next(row for row in summary.provider_server_logs if row.provider == "sglang")
    assert sglang_log.prefix_cache is True
    assert sglang_log.chunked_prefill is True
    assert sglang_log.chunked_prefill_tokens == 8192
    assert sglang_log.async_scheduling is True
    assert sglang_log.decode_cuda_graph is True
    assert sglang_log.decode_cuda_graph_max_bs == 512
    assert sglang_log.prefill_cuda_graph is True
    assert sglang_log.prefill_cuda_graph_max_bs == 8192
    assert sglang_log.continuous_decode_steps == 1
    assert sglang_log.attention_backend == "fa3"
    assert sglang_log.sampling_backend == "flashinfer"
    assert sglang_log.prefill_batches == 2
    assert sglang_log.prefill_new_tokens == 56
    assert sglang_log.prefill_cached_tokens == 16
    assert sglang_log.prefill_new_seq_counts == {"3": 1, "1": 1}
    assert sglang_log.prefill_new_token_bucket_counts == {"<=64": 1, "<=8": 1}
    assert sglang_log.decode_batches == 2
    assert sglang_log.decode_logged_tokens == 72
    assert sglang_log.decode_running_counts == {"4": 1, "2": 1}
    assert len(summary.torchinferno_startup_warmups) == 3
    decode_warmup = summary.torchinferno_startup_warmups[0]
    assert decode_warmup.label == "online decode graph warmup"
    assert decode_warmup.seconds == 35.8
    assert decode_warmup.temperature == 0.0
    assert decode_warmup.max_tokens == 1
    greedy_warmup = summary.torchinferno_startup_warmups[1]
    assert greedy_warmup.label == "greedy common-prefix suffix warmup"
    assert greedy_warmup.seconds == 102.3
    assert greedy_warmup.max_tokens == 128
    assert greedy_warmup.shapes == 63
    assert greedy_warmup.token_graphs is True
    total_warmup = summary.torchinferno_startup_warmups[2]
    assert total_warmup.label == "tensor-parallel startup warmup"
    assert total_warmup.seconds == 219.9
    assert len(summary.torchinferno_profiler_events) == 4
    profiler_event = summary.torchinferno_profiler_events[0]
    assert profiler_event.kind == "RAGGED_PREFILL_REPLAY_PROF"
    assert profiler_event.batch == 24
    assert profiler_event.suffix == 64
    assert profiler_event.matches == 5
    assert profiler_event.context_len == "-256"
    assert profiler_event.src_rows == 1
    assert profiler_event.prefix_copy_len == "none"
    assert profiler_event.self_cuda_ms == 83.4
    assert round(profiler_event.allreduce_ms, 3) == 24.592
    assert round(profiler_event.gemm_ms, 3) == 10.669
    assert round(profiler_event.add_rms_ms, 3) == 7.138
    assert round(profiler_event.softmax_ms, 3) == 0.654
    decode_profiler_event = summary.torchinferno_profiler_events[1]
    assert decode_profiler_event.kind == "RAGGED_DECODE_MANY_REPLAY_PROF"
    assert decode_profiler_event.batch == 64
    assert decode_profiler_event.suffix is None
    assert decode_profiler_event.cache_bucket == "1024"
    assert decode_profiler_event.rows == 64
    assert decode_profiler_event.steps == 8
    assert decode_profiler_event.matches == 3
    assert decode_profiler_event.context_len is None
    assert decode_profiler_event.src_rows is None
    assert decode_profiler_event.prefix_copy_len is None
    assert decode_profiler_event.self_cuda_ms == 40.0
    assert decode_profiler_event.allreduce_ms == 12.0
    assert decode_profiler_event.gemm_ms == 5.0
    assert decode_profiler_event.marlin_ms == 6.0
    assert decode_profiler_event.attention_ms == 1.5
    assert decode_profiler_event.add_rms_ms == 2.0
    single_decode_profiler_event = summary.torchinferno_profiler_events[2]
    assert single_decode_profiler_event.kind == "RAGGED_DECODE_REPLAY_PROF"
    assert single_decode_profiler_event.batch == 48
    assert single_decode_profiler_event.suffix is None
    assert single_decode_profiler_event.cache_bucket == "512"
    assert single_decode_profiler_event.rows == 32
    assert single_decode_profiler_event.steps is None
    assert single_decode_profiler_event.matches == 2
    assert single_decode_profiler_event.self_cuda_ms == 20.0
    assert single_decode_profiler_event.allreduce_ms == 3.0
    assert single_decode_profiler_event.gemm_ms == 4.0
    assert single_decode_profiler_event.marlin_ms == 2.0
    assert single_decode_profiler_event.attention_ms == 1.0
    eager_decode_many_event = summary.torchinferno_profiler_events[3]
    assert eager_decode_many_event.kind == "RAGGED_DECODE_MANY_EAGER_PROF"
    assert eager_decode_many_event.batch == 64
    assert eager_decode_many_event.cache_bucket == "1024"
    assert eager_decode_many_event.rows == 64
    assert eager_decode_many_event.steps == 1
    assert eager_decode_many_event.matches == 4
    assert eager_decode_many_event.self_cuda_ms == 12.0
    assert eager_decode_many_event.gemm_ms == 3.5
    assert eager_decode_many_event.marlin_ms == 3.25
    assert eager_decode_many_event.attention_ms == 1.25

    text = format_inference_bench_summary(summary)
    assert "[long_output]" in text
    assert "[torchinferno ragged replay profiler]" in text
    assert "[torchinferno startup warmup]" in text
    assert "online decode graph warmup" in text
    assert "greedy common-prefix suffix warmup" in text
    assert "102.3" in text
    assert "63" in text
    assert "True" in text
    assert "commit" in text
    assert "abcdef1" in text
    assert "123456a" in text
    assert "score_tpot" in text
    assert "4.0" in text
    assert "torchinferno" in text
    assert "decode_many_replay" in text
    assert "decode_many_eager" in text
    assert "decode_replay" in text
    assert "cache" in text
    assert "1024" in text
    assert "512" in text
    assert "[long_output raw request waves]" in text
    assert "ttft_p90" in text
    assert "0-63" in text
    assert "64-127" in text
    assert "[torchinferno queue profiles]" in text
    assert "cache" in text
    assert "dense" in text
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
    assert "suffix_split_cand" in text
    assert "suffix_split_cand_saved" in text
    assert "suffix_split_ok" in text
    assert "suffix_split_reasons" in text
    assert "suffix_split_saved" in text
    assert "suffix_split_frags" in text
    assert "disabled=1" in text
    assert "init_wait" in text
    assert "idle_wait" in text
    assert "active_wait" in text
    assert "decode_capture" in text
    assert "False" in text
    assert "hot_prefill" in text
    assert "b8:s16:p45-45:src1:mixed0" in text
    assert "hot_decode" in text
    assert "[torchinferno prefill graph phase]" in text
    assert "active_tok_s" in text
    assert "sfx_ms_est" in text
    assert "[torchinferno decode-many phase]" in text
    assert "emit_tok_s" in text
    assert "overgen_pct" in text
    assert "[torchinferno decode-many graph policy]" in text
    assert "graph_step_pct" in text
    assert "graph_model_tok" in text
    assert "packed_fi_ms" in text
    assert "packed_fi_saved" in text
    assert "[torchinferno packed FlashInfer prefill gate]" in text
    assert "ran" in text
    assert "packed_eager_ms" in text
    assert "packed_eager_saved" in text
    assert "packed_cand_saved" in text
    assert "[torchinferno packed prefill fixed-capacity runtime]" in text
    assert "capacity_grew=3" in text
    assert "prefix_reuse" in text
    assert "prefix_reuse_tok" in text
    assert "prefix_routes" in text
    assert "common_prefix=3,request_prompt=2" in text
    assert "prefix_hits" in text
    assert "16=3,8=2" in text
    assert "prefill_setup_ms" in text
    assert "prefill_copy_ms" in text
    assert "prefill_sample_ms" in text
    assert "sample_select_ms" in text
    assert "sample_readback_ms" in text
    assert "tp_samp_ms" in text
    assert "tp_g_noise" in text
    assert "tp_g_reduce" in text
    assert "tp_samp_cdf" in text
    assert "tp_samp_reduce" in text
    assert "state_seq_ms" in text
    assert "state_store_ms" in text
    assert "state_create_ms" in text
    assert "gpu_ms_call" in text
    assert "gpu_us_tok" in text
    assert "decode_many_wait_ms" in text
    assert "decode_many_materialize_ms" in text
    assert "many_syncs" in text
    assert "many_sync_skips" in text
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
    assert "gen_req" in text
    assert "gen_base" in text
    assert "gen_on" in text
    assert "prompt_lookup_on" in text
    assert "store_logits" in text
    assert "pin_logits" in text
    assert "prefill_graph_miss" in text
    assert "prefill_miss_kind" in text
    assert "ragged=6,static_logits=3" in text
    assert "prefill_graph_cache" in text
    assert "prefill_graph_cache_cap" in text
    assert "prefill_graph_evictions" in text
    assert "prefill_graph_evicted" in text
    assert "prefill_graph_suffix" in text
    assert "s12=14,s96=2" in text
    assert "decode_graph_miss" in text
    assert "[torchinferno cache integrity]" in text
    assert "prompt_lookup_req" in text
    assert "prompt_lookup_accept" in text
    assert "prefix_logits" in text
    assert "prefix_logit_tok" in text
    assert "prefix_sample" in text
    assert "prefix_greedy" in text
    assert "repeat_hits" in text
    assert "review" in text
    assert "[torchinferno full-prompt store skips]" in text
    assert "pinned_without_allowance=2" in text
    assert "pinned_without_allowance=40" in text
    assert "[torchinferno full-prompt reuse candidates]" in text
    assert "session" in text
    assert "persistent" in text
    assert "22=2" in text
    assert "21=1" in text
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
    assert "[torchinferno packed prefill dynamic-count targets]" in text
    assert "[torchinferno non-fragmenting prefill targets]" in text
    assert "[torchinferno packed prefill implementation targets]" in text
    assert "[torchinferno packed prefill signature reuse]" in text
    assert "[torchinferno packed prefill pattern reuse]" in text
    assert "p45:s10:n2/p45:s11:n1" in text
    assert "p45:s10/p45:s11" in text
    assert "slot_src" in text
    assert "runtime" in text
    assert "est_saved_ms" in text
    assert "row_saved_ms" in text
    assert "suffix_saved_ms" in text
    assert "fixed_saved_pct" in text
    assert "dynamic_saved" in text
    assert "fixed_cover" in text
    assert "repeat_saved" in text
    assert "sig_cov" in text
    assert "suffix_cand_saved" in text
    assert "packed_body" in text
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
    assert "max_call_saved" in text
    assert "max_call_pct" in text
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
    assert "decode_many_model_tok" in text
    assert "decode_many_emit_tok" in text
    assert "decode_many_tok_call" in text
    assert "decode_many_steps_call" in text
    assert "decode_many_graph_ms" in text
    assert "50.0%" in text
    assert "12.0" in text
    assert "3.0" in text
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
    assert "[torchinferno first-token prefill shapes]" in text
    assert "q2submit_p50" in text
    assert "submit2first_p50" in text
    assert "q2first_p99" in text
    assert "shape_a" in text
    assert "[torchinferno hot prefill shapes]" in text
    assert "prefix_graph:b8:s16:p45-45:src1:mixed0" in text
    assert "prefix_graph:b4:s16:p45-45:src1:mixed0" in text
    assert "calls" in text
    assert "active_tokens" in text
    assert "pad_pct" in text
    assert "35.4%" in text
    assert "row_pad" in text
    assert "suffix_pad" in text
    assert "row_ms_est" in text
    assert "suffix_ms_est" in text
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
    assert "stop_fin" in text
    assert "limit_fin" in text
    assert "fin_src" in text
    assert "skip_ms_est" in text
    assert "skip_ms_share" in text
    assert "skip_stop" in text
    assert "window" in text
    assert "605.3" in text
    assert "77.0%" in text
    assert "decode_many:b8/8:g1-16" in text
    assert "exact" in text
    assert "model_ms" in text
    assert "cpu_ms" in text
    assert "wait_ms" in text
    assert "materialize_ms" in text
    assert "total_ms" in text
    assert "12.7" in text
    assert "[torchinferno ragged replay profiler]" in text
    assert "replay" in text
    assert "self_cuda_ms" in text
    assert "allreduce_pct" in text
    assert "marlin_ms" in text
    assert "marlin_pct" in text
    assert "attention_ms" in text
    assert "attention_pct" in text
    assert "gemm_pct" in text
    assert "83.4" in text
    assert "24.6" in text
    assert "29.5%" in text
    assert "10.7" in text
    assert "12.8%" in text
    assert "7.1" in text
    assert "0.7" in text
    assert "[provider serving config]" in text
    assert "prefix_cache" in text
    assert "chunk_tokens" in text
    assert "decode_max_bs" in text
    assert "prefill_max_bs" in text
    assert "kv_tokens" in text
    assert "1451760" in text
    assert "fa3" in text
    assert "flashinfer" in text
    assert "[provider server log phases]" in text
    assert "running_max" in text
    assert "kv_cache_avg" in text
    assert "prefix_hit_avg" in text
    assert "prefill_graph_pct" in text
    assert "prefill_tok_batch" in text
    assert "prefill_seq_top" in text
    assert "prefill_tok_top" in text
    assert "cached_pct" in text
    assert "decode_tok_batch" in text
    assert "decode_tok_top" in text
    assert "decode_graph_pct" in text
    assert "decode_running_top" in text
    assert "sglang" in text
    assert "0.2%" in text
    assert "85.0%" in text
    assert "50.0%" in text
    assert "28.0" in text
    assert "22.2%" in text
    assert "3=1" in text
    assert "<=64=1" in text
    assert "<=8=1" in text
    assert "4=1" in text
    assert "36.0" in text


def test_queue_profile_infers_cache_backend_from_torchinferno_log(tmp_path) -> None:
    for case_name, command_tail, expected_backend in (
        ("default", "--port 8001 --trust-remote-code", "dense"),
        ("explicit", "--cache-backend flashinfer --port 8001", "flashinfer"),
    ):
        case_root = tmp_path / case_name
        logs = case_root / "provider_logs"
        logs.mkdir(parents=True)
        (case_root / "results.json").write_text(
            json.dumps(
                {
                    "model": "meta-llama/test",
                    "tensor_parallel_size": 8,
                    "hardware": "8xH100",
                    "providers": {
                        "torchinferno": {
                            "benchmarks": {"long_output": {"metrics": {}}},
                        },
                    },
                }
            )
        )
        (logs / "torchinferno_queue_profile.jsonl").write_text(
            json.dumps(
                {
                    "event": "online_batcher",
                    "temperature": 0.0,
                    "run_max_tokens": 96,
                    "submitted_requests": 1,
                }
            )
            + "\n"
        )
        (logs / "torchinferno_server.log").write_text(
            "TorchInferno OpenAI server auto-launching tensor-parallel workers: "
            "python -m torchinferno.openai_server --model /models/llama "
            f"{command_tail}\n"
        )

        summary = summarize_inference_bench_run(case_root)

        assert (
            summary.torchinferno_queue_profiles[0].fields["runtime_cache_backend"]
            == expected_backend
        )


def test_packed_flashinfer_gate_rows_infers_dense_cache_blocker() -> None:
    profile = QueueProfileSummary(
        event="online_batcher",
        temperature=0.0,
        max_tokens=96,
        submitted_requests=64,
        finished_events=64,
        fields={
            "runtime_cache_backend": "dense",
            "runtime_prefill_packed_flashinfer_calls": 0,
            "runtime_prefill_packed_flashinfer_saved_tokens": 0,
            "runtime_prefill_packed_candidate_calls": 12,
            "runtime_prefill_packed_candidate_saved_tokens": 4096,
        },
    )

    assert _prefill_packed_flashinfer_gate_rows([profile]) == [
        (
            "0.0",
            "96",
            "dense",
            "-",
            "-",
            "-",
            "-",
            "cache_backend_dense",
            "0",
            "0",
            "12",
            "4096",
        )
    ]


def test_inference_bench_summary_reads_current_provider_log_names(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)
    logs = tmp_path / "provider_logs"
    (logs / "vllm_server.log").rename(logs / "vllm.log")
    (logs / "sglang_server.log").rename(logs / "sglang.log")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))

    assert "[provider serving config]" in text
    assert "chunk_tokens" in text
    assert "decode_max_bs" in text
    assert "[provider server log phases]" in text
    assert "vllm" in text
    assert "sglang" in text
    assert "prefix_hit_avg" in text
    assert "prefill_graph_pct" in text
    assert "prefill_tok_batch" in text
    assert "prefill_seq_top" in text
    assert "prefill_tok_top" in text
    assert "cached_pct" in text
    assert "decode_tok_batch" in text
    assert "decode_tok_top" in text
    assert "decode_graph_pct" in text
    assert "decode_running_top" in text


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


def test_prefill_target_uses_gpu_replay_timing_without_sync() -> None:
    fields = {
        "runtime_prefill_forward_ms": 0.0,
        "runtime_prefill_graph_replay_gpu_ms": 40.0,
        "runtime_prefill_shape_graph_replay_gpu_ms": {
            "prefix_graph:b2:s12:p45-45:src1:mixed0": 30.0,
            "prefix_graph:b4:s12:p45-45:src1:mixed0": 10.0,
        },
        "runtime_decode_many_model_gpu_ms": 0.0,
    }

    assert _prefill_target_ms(fields) == 40.0
    assert (
        _phase_target(
            prefill_ms=_prefill_target_ms(fields),
            decode_ms=0.0,
            sample_ms=0.0,
            capture_ms=0.0,
        )
        == "prefill"
    )
    assert _top_prefill_target_entry(fields) == (
        "prefix_graph:b2:s12:p45-45:src1:mixed0",
        30.0,
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
            {"runtime_prefill_shape_graph_replay_gpu_ms": {"shape": 5.0}},
            "shape",
        )
        == 5.0
    )
    assert (
        _prefill_shape_dense_forward_ms(
            {"runtime_prefill_graph_replay_shape_gpu_ms": {"shape": 7.0}},
            "shape",
        )
        == 7.0
    )
    assert (
        _prefill_shape_dense_forward_ms(
            {
                "runtime_prefill_graph_replay_gpu_ms": 50.0,
                "runtime_prefill_shape_model_tokens": {"shape": 64, "other": 36},
            },
            "shape",
        )
        == 32.0
    )
    assert (
        _prefill_shape_dense_forward_ms(
            {
                "runtime_prefill_shape_graph_replay_gpu_ms": {"shape": 5.0},
                "runtime_prefill_packed_eager_shape_ms": {"shape": 1.25},
            },
            "shape",
        )
        == 3.75
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


def test_dynamic_packed_prefill_targets_use_gpu_replay_timing_without_sync() -> None:
    shape = "prefix_graph:b4:s8:p10-10:src1:mixed0"
    pattern = f"{shape}|p10:s6/p10:s8"
    rows = _prefill_packed_dynamic_target_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=512,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "runtime_prefill_forward_ms": 0.0,
                    "runtime_prefill_graph_replay_gpu_ms": 40.0,
                    "runtime_prefill_shape_graph_replay_gpu_ms": {shape: 32.0},
                    "runtime_prefill_shape_model_tokens": {shape: 64},
                    "runtime_prefill_packed_candidate_pattern_counts": {pattern: 2},
                    "runtime_prefill_packed_candidate_pattern_model_tokens": {pattern: 64},
                    "runtime_prefill_packed_candidate_pattern_saved_tokens": {pattern: 16},
                    "runtime_prefill_packed_candidate_pattern_row_saved_tokens": {
                        pattern: 4,
                    },
                    "runtime_prefill_packed_candidate_pattern_suffix_saved_tokens": {
                        pattern: 12,
                    },
                },
            )
        ]
    )

    assert rows == [
        (
            "0.0",
            "512",
            pattern,
            "2",
            "16",
            "4",
            "12",
            "0",
            "0.0%",
            "8",
            "2",
            "6",
            "20.0%",
            "-",
        )
    ]


def test_dynamic_packed_prefill_targets_estimate_older_runs_from_total_replay() -> None:
    shape = "prefix_graph:b4:s8:p10-10:src1:mixed0"
    other_shape = "prefix_graph:b4:s8:p11-11:src1:mixed0"
    pattern = f"{shape}|p10:s6/p10:s8"
    rows = _prefill_packed_dynamic_target_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=512,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "runtime_prefill_forward_ms": 0.0,
                    "runtime_prefill_graph_replay_gpu_ms": 40.0,
                    "runtime_prefill_shape_model_tokens": {
                        shape: 64,
                        other_shape: 64,
                    },
                    "runtime_prefill_packed_candidate_pattern_counts": {pattern: 2},
                    "runtime_prefill_packed_candidate_pattern_model_tokens": {pattern: 64},
                    "runtime_prefill_packed_candidate_pattern_saved_tokens": {pattern: 16},
                },
            )
        ]
    )

    assert rows == [
        (
            "0.0",
            "512",
            pattern,
            "2",
            "16",
            "-",
            "-",
            "0",
            "0.0%",
            "5",
            "-",
            "-",
            "12.5%",
            "-",
        )
    ]


def test_dynamic_packed_prefill_targets_infer_saved_sources_from_signatures() -> None:
    shape = "prefix_graph:b4:s8:p10-10:src1:mixed0"
    pattern = f"{shape}|p10:s6/p10:s8"
    signature = f"{shape}|p10:s6:n2/p10:s8:n1"
    rows = _prefill_packed_dynamic_target_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=512,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "runtime_prefill_forward_ms": 0.0,
                    "runtime_prefill_graph_replay_gpu_ms": 40.0,
                    "runtime_prefill_shape_graph_replay_gpu_ms": {shape: 32.0},
                    "runtime_prefill_shape_model_tokens": {shape: 64},
                    "runtime_prefill_packed_candidate_pattern_counts": {pattern: 2},
                    "runtime_prefill_packed_candidate_pattern_model_tokens": {pattern: 64},
                    "runtime_prefill_packed_candidate_pattern_saved_tokens": {pattern: 24},
                    "runtime_prefill_packed_candidate_signature_counts": {signature: 2},
                    "runtime_prefill_packed_candidate_signature_model_tokens": {
                        signature: 64,
                    },
                    "runtime_prefill_packed_candidate_signature_saved_tokens": {
                        signature: 24,
                    },
                },
            )
        ]
    )

    assert rows == [
        (
            "0.0",
            "512",
            pattern,
            "2",
            "24",
            "16",
            "8",
            "24",
            "100.0%",
            "12",
            "8",
            "4",
            "30.0%",
            "-",
        )
    ]


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


def test_prefill_graph_phase_rows_summarize_padding_and_throughput() -> None:
    rows = _prefill_graph_phase_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "runtime_prefill_graph_replays": 2,
                    "runtime_prefill_graph_replay_gpu_ms": 10.0,
                    "runtime_prefill_shape_active_tokens": {
                        "prefix_graph:b8:s16:p45-45:src1:mixed0": 20,
                    },
                    "runtime_prefill_shape_model_tokens": {
                        "prefix_graph:b8:s16:p45-45:src1:mixed0": 30,
                    },
                    "runtime_prefill_shape_padding_tokens": {
                        "prefix_graph:b8:s16:p45-45:src1:mixed0": 10,
                    },
                    "runtime_prefill_row_padding_tokens": 3,
                    "runtime_prefill_suffix_padding_tokens": 7,
                },
            )
        ]
    )

    assert rows == [
        (
            "0.0",
            "96",
            "2",
            "10.0",
            "20",
            "30",
            "10",
            "3",
            "7",
            "1",
            "2.3",
            "33.3%",
            "2000.0",
            "3000.0",
            "500.0",
            "333.3",
        )
    ]


def test_decode_many_phase_rows_summarize_throughput_and_overgeneration() -> None:
    rows = _decode_many_phase_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "runtime_decode_many_calls": 2,
                    "runtime_decode_many_steps": 6,
                    "runtime_decode_many_model_gpu_ms": 15.0,
                    "runtime_decode_many_shape_model_tokens": {"decode_many:b8/8": 24},
                    "runtime_decode_many_shape_padded_tokens": {"decode_many:b8/8": 30},
                    "runtime_decode_many_shape_emitted_tokens": {"decode_many:b8/8": 15},
                    "runtime_decode_many_shape_skipped_tokens": {"decode_many:b8/8": 4},
                    "runtime_decode_many_shape_overgenerated_tokens": {
                        "decode_many:b8/8": 3,
                    },
                },
            )
        ]
    )

    assert rows == [
        (
            "0.0",
            "96",
            "2",
            "6",
            "15.0",
            "24",
            "30",
            "15",
            "4",
            "3",
            "20.0%",
            "16.7%",
            "12.5%",
            "1000.0",
            "1600.0",
            "1000.0",
            "625.0",
        )
    ]


def test_decode_many_graph_policy_rows_classify_runtime_state() -> None:
    rows = _decode_many_graph_policy_rows(
        [
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "use_decode_many": False,
                    "decode_many_graph": True,
                    "decode_capture_on_miss": True,
                    "runtime_decode_many_graph_calls": 0,
                    "runtime_decode_many_graph_steps": 0,
                    "runtime_decode_many_graph_model_tokens": 0,
                    "runtime_decode_many_steps": 6,
                    "runtime_decode_many_model_gpu_ms": 15.0,
                },
            ),
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "use_decode_many": True,
                    "decode_many_graph": True,
                    "decode_capture_on_miss": False,
                    "runtime_decode_many_graph_calls": 0,
                    "runtime_decode_many_graph_steps": 0,
                    "runtime_decode_many_graph_model_tokens": 0,
                    "runtime_decode_many_steps": 6,
                    "runtime_decode_many_model_gpu_ms": 15.0,
                },
            ),
            QueueProfileSummary(
                event="online_batcher_quiescent",
                temperature=0.0,
                max_tokens=96,
                submitted_requests=8,
                finished_events=8,
                fields={
                    "use_decode_many": True,
                    "decode_many_graph": True,
                    "decode_capture_on_miss": True,
                    "runtime_decode_many_graph_calls": 2,
                    "runtime_decode_many_graph_steps": 4,
                    "runtime_decode_many_graph_model_tokens": 32,
                    "runtime_decode_many_graph_ms": 7.5,
                    "runtime_decode_many_steps": 8,
                    "runtime_decode_many_model_gpu_ms": 20.0,
                },
            ),
        ]
    )

    assert rows == [
        (
            "0.0",
            "96",
            "True",
            "True",
            "0",
            "0",
            "0.0%",
            "0",
            "-",
            "15.0",
            "6",
            "decode_many_off",
        ),
        (
            "0.0",
            "96",
            "True",
            "False",
            "0",
            "0",
            "0.0%",
            "0",
            "-",
            "15.0",
            "6",
            "capture_off",
        ),
        (
            "0.0",
            "96",
            "True",
            "True",
            "2",
            "4",
            "50.0%",
            "32",
            "7.5",
            "20.0",
            "8",
            "ran",
        ),
    ]


def test_inference_bench_summary_leaves_decode_many_window_ms_blank_without_timing(
    tmp_path,
) -> None:
    _write_inference_bench_run(tmp_path)
    queue_path = tmp_path / "provider_logs" / "torchinferno_queue_profile.jsonl"
    queue_record = json.loads(queue_path.read_text())
    queue_record.pop("runtime_decode_many_step_window_model_ms")
    queue_record.pop("runtime_decode_many_step_window_cpu_tokens_ms")
    queue_record.pop("runtime_decode_many_step_window_token_wait_ms")
    queue_record.pop("runtime_decode_many_step_window_token_materialize_ms")
    queue_path.write_text(json.dumps(queue_record) + "\n")

    text = format_inference_bench_summary(summarize_inference_bench_run(tmp_path))
    row = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("0.0")
        and "decode_many:b8/8:g1-16" in line
    )

    assert row.split()[-4:] == ["-", "-", "-", "-"]


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


def test_prefill_fixed_capacity_reject_rows_show_over_dense_patterns() -> None:
    profile = QueueProfileSummary(
        event="online_batcher_quiescent",
        temperature=0.0,
        max_tokens=256,
        submitted_requests=1000,
        finished_events=1000,
        fields={
            "runtime_prefill_packed_candidate_pattern_counts": {
                "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14": 33,
            },
            "runtime_prefill_forward_ms": 120.0,
            "runtime_prefill_shape_forward_ms": {
                "prefix_graph:b32:s16:p122-122:src1:mixed0": 60.0,
            },
            "runtime_prefill_shape_model_tokens": {
                "prefix_graph:b32:s16:p122-122:src1:mixed0": 16896,
            },
            "runtime_prefill_packed_candidate_pattern_saved_tokens": {
                "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14": 5088,
            },
            "runtime_prefill_packed_candidate_pattern_slot_counts": {
                "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14#p122:s12": 24,
                "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14#p122:s13": 16,
                "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14#p122:s14": 7,
            },
        },
    )

    rows = _prefill_packed_fixed_capacity_reject_rows([profile])

    assert rows == [
        (
            "0.0",
            "256",
            "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14",
            "33",
            "16896",
            "19602",
            "2706",
            "5088",
            "16.0%",
        ),
    ]

    dynamic_rows = _prefill_packed_dynamic_target_rows([profile])

    assert dynamic_rows == [
        (
            "0.0",
            "256",
            "prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14",
            "33",
            "5088",
            "-",
            "-",
            "0",
            "0.0%",
            "18.1",
            "-",
            "-",
            "15.1%",
            "-",
        ),
    ]


def test_prefill_fixed_capacity_runtime_rows_show_acceptance_and_rejects() -> None:
    profile = QueueProfileSummary(
        event="online_batcher_quiescent",
        temperature=0.0,
        max_tokens=96,
        submitted_requests=1000,
        finished_events=1000,
        fields={
            "runtime_prefill_packed_fixed_capacity_attempts": 58,
            "runtime_prefill_packed_fixed_capacity_accepts": 1,
            "runtime_prefill_packed_fixed_capacity_dense_tokens": 4096,
            "runtime_prefill_packed_fixed_capacity_fixed_tokens": 2048,
            "runtime_prefill_packed_fixed_capacity_real_tokens": 1800,
            "runtime_prefill_packed_fixed_capacity_saved_tokens": 2048,
            "runtime_prefill_packed_fixed_capacity_padding_tokens": 248,
            "runtime_prefill_packed_fixed_capacity_reject_reason_counts": {
                "capacity_grew": 57,
            },
            "runtime_prefill_packed_eager_calls": 1,
            "runtime_prefill_packed_eager_saved_tokens": 248,
            "runtime_prefill_packed_eager_ms": 0.0,
            "runtime_prefill_packed_candidate_saved_tokens": 28117,
        },
    )

    assert _prefill_packed_fixed_capacity_runtime_rows([profile]) == [
        (
            "0.0",
            "96",
            "58",
            "1",
            "1.7%",
            "57",
            "capacity_grew=57",
            "1",
            "248",
            "4096",
            "2048",
            "1800",
            "2048",
            "248",
            "50.0%",
            "0.0",
            "28117",
        )
    ]


def test_prefill_non_fragmenting_target_rows_rank_packed_over_fragmented_split() -> None:
    profile = QueueProfileSummary(
        event="online_batcher_quiescent",
        temperature=0.0,
        max_tokens=96,
        submitted_requests=1000,
        finished_events=1000,
        fields={
            "runtime_prefill_forward_ms": 100.0,
            "runtime_prefill_shape_forward_ms": {
                "prefix_graph:b24:s96:p111-111:src1:mixed0": 40.0,
            },
            "runtime_prefill_shape_model_tokens": {
                "prefix_graph:b24:s96:p111-111:src1:mixed0": 23040,
            },
            "runtime_prefill_packed_candidate_calls": 15,
            "runtime_prefill_packed_candidate_saved_tokens": 5440,
            "runtime_prefill_packed_candidate_shape_saved_tokens": {
                "prefix_graph:b24:s96:p111-111:src1:mixed0": 1024,
            },
            "runtime_prefill_suffix_split_candidate_saved_tokens": 5440,
            "runtime_prefill_suffix_split_accepted_saved_tokens": 0,
            "runtime_prefill_suffix_split_rejected_calls": 15,
            "runtime_prefill_suffix_split_reject_reason_counts": {
                "disabled": 4,
                "min_fill": 5,
                "no_savings": 6,
            },
        },
    )

    assert _prefill_non_fragmenting_target_rows([profile]) == [
        (
            "0.0",
            "96",
            "5440",
            "15",
            "5440",
            "0",
            "15",
            "no_savings=6,min_fill=5,disabled=4",
            "prefix_graph:b24:s96:p111-111:src1:mixed0",
            "1024",
            "1.8",
            "1.8%",
            "packed_body",
        )
    ]


def test_cache_integrity_rows_flag_enabled_shortcut_config() -> None:
    profile = QueueProfileSummary(
        event="online_batcher_quiescent",
        temperature=0.7,
        max_tokens=256,
        submitted_requests=64,
        finished_events=64,
        fields={
            "runtime_generated_prefix_cache_requested": True,
            "runtime_generated_prefix_cache_base_enabled": True,
            "runtime_generated_prefix_cache_effective_enabled": False,
            "runtime_prompt_lookup_decode_effective_enabled": True,
            "runtime_prefix_cache_store_logits_enabled": False,
            "runtime_pinned_prefix_cache_store_logits_enabled": True,
            "runtime_generated_prefix_store_requests": 0,
            "runtime_generated_prefix_reuse_requests": 0,
            "runtime_generated_prefix_reuse_tokens": 0,
            "runtime_prompt_lookup_requests": 0,
            "runtime_prompt_lookup_proposed_tokens": 0,
            "runtime_prompt_lookup_accepted_tokens": 0,
            "runtime_reusable_prefix_logits_entries": 0,
            "runtime_reusable_prefix_logits_tokens": 0,
            "runtime_reusable_prefix_sample_state_entries": 0,
            "runtime_reusable_prefix_greedy_token_entries": 0,
            "runtime_repeated_sample_state_hits": 0,
            "runtime_repeated_sample_state_tokens": 0,
        },
    )

    assert _cache_integrity_rows([profile]) == [
        (
            "0.7",
            "256",
            "64",
            "0",
            "0",
            "0",
            "on",
            "on",
            "off",
            "on",
            "off",
            "on",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "review",
        )
    ]


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


def test_inference_bench_summary_cli_accepts_results_json_path(tmp_path) -> None:
    _write_inference_bench_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "inference-bench-summary",
            str(tmp_path / "results.json"),
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
