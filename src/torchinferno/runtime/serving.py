from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from inspect import signature
from typing import Callable, Hashable, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import Tensor

from torchinferno.runtime.options import env_flag, env_int, warn_optional_failure
from torchinferno.runtime.prefix import PrefixMatch
from torchinferno.runtime.prefix_cache import PrefixCacheEntry, PrefixCacheIndex
from torchinferno.runtime.sampling import sample_next_token


def _fi_decode_graph_mode() -> str:
    raw = os.environ.get("TORCHINFERNO_FI_DECODE_GRAPH", "sampled").strip().lower()
    if raw in {"1", "true", "yes", "on", "always"}:
        return "always"
    if raw in {"0", "false", "no", "off", "never", ""}:
        return "off"
    if raw in {"sample", "sampled", "auto"}:
        return "sampled"
    return "off"


def _ragged_decode_graph_key_symm_suffix(graph_key: object) -> str | None:
    if not isinstance(graph_key, tuple) or len(graph_key) < 6:
        return None
    raw = graph_key[5]
    return f"symm{str(raw).replace(' ', '')}"


def _ragged_decode_graph_key_cache_suffix(graph_key: object) -> str | None:
    if not isinstance(graph_key, tuple) or len(graph_key) < 4:
        return None
    raw = graph_key[3]
    return f"cache{str(raw).replace(' ', '')}"


def _preferred_prefix_rows() -> tuple[int, ...]:
    raw = os.environ.get("TORCHINFERNO_CONTINUOUS_PREFERRED_PREFIX_ROWS", "48,53,68,69,128")
    return _parse_positive_int_csv(raw, minimum=0)


def _parse_positive_int_csv(raw: str | None, *, minimum: int = 1) -> tuple[int, ...]:
    rows: list[int] = []
    seen: set[int] = set()
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            row = int(token)
        except ValueError:
            continue
        if row < minimum or row in seen:
            continue
        rows.append(row)
        seen.add(row)
    return tuple(rows)


def _bucket_from_values(length: int, buckets: tuple[int, ...]) -> int | None:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return None


def _packed_prefill_candidate_signature(
    shape_key: str,
    *,
    suffix_lengths: Sequence[int],
    start_lens: Sequence[int],
) -> str:
    group_counts = _packed_prefill_group_counts(
        suffix_lengths=suffix_lengths,
        start_lens=start_lens,
    )
    return _packed_prefill_candidate_signature_from_counts(shape_key, group_counts)


def _packed_prefill_candidate_signature_from_counts(
    shape_key: str,
    group_counts: Mapping[tuple[int, int], int],
) -> str:
    groups = "/".join(
        f"p{start}:s{suffix}:n{count}"
        for (start, suffix), count in sorted(group_counts.items())
    )
    return f"{shape_key}|{groups}" if groups else f"{shape_key}|empty"


def _packed_prefill_candidate_pattern(
    shape_key: str,
    *,
    suffix_lengths: Sequence[int],
    start_lens: Sequence[int],
) -> str:
    group_counts = _packed_prefill_group_counts(
        suffix_lengths=suffix_lengths,
        start_lens=start_lens,
    )
    return _packed_prefill_candidate_pattern_from_counts(shape_key, group_counts)


def _packed_prefill_candidate_pattern_from_counts(
    shape_key: str,
    group_counts: Mapping[tuple[int, int], int],
) -> str:
    groups = "/".join(
        f"p{start}:s{suffix}"
        for start, suffix in sorted(group_counts)
    )
    return f"{shape_key}|{groups}" if groups else f"{shape_key}|empty"


def _packed_prefill_group_counts(
    *,
    suffix_lengths: Sequence[int],
    start_lens: Sequence[int],
) -> dict[tuple[int, int], int]:
    group_counts: dict[tuple[int, int], int] = defaultdict(int)
    for suffix_len, start_len in zip(suffix_lengths, start_lens):
        group_counts[(max(0, int(start_len)), max(0, int(suffix_len)))] += 1
    return dict(group_counts)


def _packed_prefill_pattern_slot_key(pattern_key: str, *, start_len: int, suffix_len: int) -> str:
    return f"{pattern_key}#p{max(0, int(start_len))}:s{max(0, int(suffix_len))}"


def _parse_string_csv(raw: str | None) -> tuple[str, ...]:
    return tuple(token for token in (part.strip() for part in (raw or "").split(",")) if token)


def _queue_profile_counts_enabled() -> bool:
    return bool(
        os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL")
        or os.environ.get("TORCHINFERNO_OPENAI_QUEUE_PROFILE")
    )


def _packed_prefill_eager_pattern_matches(
    *,
    profile_shape_key: str | None,
    packed_prefill_pattern_key: str | None,
) -> bool:
    targets = _parse_string_csv(
        os.environ.get("TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_PATTERN")
    )
    if not targets:
        return False
    target_set = set(targets)
    return (
        (profile_shape_key is not None and profile_shape_key in target_set)
        or (
            packed_prefill_pattern_key is not None
            and packed_prefill_pattern_key in target_set
        )
    )


def _packed_prefill_fixed_capacity_enabled(
    *,
    profile_shape_key: str | None,
    packed_prefill_pattern_key: str | None,
) -> bool:
    if not env_flag(
        "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_FIXED_CAPACITY_GRAPH",
        False,
    ):
        return False
    targets = _parse_string_csv(
        os.environ.get("TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_PATTERN")
    )
    if not targets:
        return True
    return _packed_prefill_eager_pattern_matches(
        profile_shape_key=profile_shape_key,
        packed_prefill_pattern_key=packed_prefill_pattern_key,
    )


_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CACHE = PrefixCacheIndex()
_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER: list[Hashable] = []
_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SEQUENCE = 0
_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SESSION_SEQUENCE = 0


def _next_persistent_full_prompt_reuse_candidate_sequence() -> int:
    global _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SEQUENCE
    _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SEQUENCE += 1
    return _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SEQUENCE


def _next_persistent_full_prompt_reuse_candidate_session() -> int:
    global _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SESSION_SEQUENCE
    _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SESSION_SEQUENCE += 1
    return _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_SESSION_SEQUENCE


def _persistent_full_prompt_reuse_candidate_session(route_id: Hashable) -> int | None:
    if (
        isinstance(route_id, tuple)
        and len(route_id) >= 3
        and route_id[0] == "persistent_full_prompt_reuse_candidate"
        and isinstance(route_id[1], int)
    ):
        return route_id[1]
    return None


def _default_prefix_prefill_suffix_buckets(
    temperature: float,
    max_generation_tokens: int | None,
) -> tuple[int, ...]:
    if max_generation_tokens is None:
        return ()
    if temperature > 0.0:
        sampled_medium_min_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_SAMPLED_MEDIUM_MIN_TOKENS",
            256,
            minimum=0,
        )
        sampled_medium_max_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_SAMPLED_MEDIUM_MAX_TOKENS",
            384,
            minimum=sampled_medium_min_tokens,
        )
        if not (sampled_medium_min_tokens < int(max_generation_tokens) <= sampled_medium_max_tokens):
            return ()
        return _parse_positive_int_csv(
            os.environ.get(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_SAMPLED_MEDIUM",
                "12,16",
            )
        )
    short_max_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT_MAX_TOKENS",
        128,
        minimum=1,
    )
    if 0 < max_generation_tokens <= short_max_tokens:
        return _parse_positive_int_csv(
            os.environ.get(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT",
                "16,32,64,96,128,256",
            )
        )
    min_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_LARGE_MIN_TOKENS",
        400,
        minimum=0,
    )
    max_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_LARGE_MAX_TOKENS",
        512,
        minimum=1,
    )
    if not (min_tokens < max_generation_tokens <= max_tokens):
        return ()
    return _parse_positive_int_csv(
        os.environ.get(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_LARGE",
            "16,32,64,80,96,112,128,144,160,192,224,256",
        )
    )


def _default_prefix_prefill_batch_buckets(
    temperature: float,
    max_generation_tokens: int | None,
    max_active_requests: int,
) -> tuple[int, ...]:
    if max_generation_tokens is None or max_active_requests <= 0:
        return ()
    if temperature <= 0.0:
        greedy_short_min_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_GREEDY_SHORT_MIN_TOKENS",
            1,
            minimum=0,
        )
        greedy_short_max_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_GREEDY_SHORT_MAX_TOKENS",
            128,
            minimum=greedy_short_min_tokens,
        )
        if not (greedy_short_min_tokens <= int(max_generation_tokens) <= greedy_short_max_tokens):
            return ()
        buckets = _parse_positive_int_csv(
            os.environ.get(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_GREEDY_SHORT",
                "1,2,4,8,16,24,32",
            )
        )
        return tuple(bucket for bucket in buckets if bucket <= int(max_active_requests))
    sampled_medium_min_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM_MIN_TOKENS",
        256,
        minimum=0,
    )
    sampled_medium_max_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM_MAX_TOKENS",
        384,
        minimum=sampled_medium_min_tokens,
    )
    if not (sampled_medium_min_tokens < int(max_generation_tokens) <= sampled_medium_max_tokens):
        return ()
    buckets = _parse_positive_int_csv(
        os.environ.get(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM",
            "1,2,4,8,16,24,32",
        )
    )
    return tuple(bucket for bucket in buckets if bucket <= int(max_active_requests))


def _enable_runtime_cache_capture_sync(cache: object) -> None:
    try:
        delattr(cache, "_skip_capture_sync")
    except AttributeError:
        pass
    except Exception:
        pass


def _dynamic_prefix_prefill_context_len(
    prefix_len: int,
    suffix_bucket: int,
    *,
    max_seq_len: int | None = None,
    max_dynamic_suffix: int | None = None,
) -> int:
    exact_len = max(1, int(prefix_len) + int(suffix_bucket))
    explicit_enabled = "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GRAPH" in os.environ
    if explicit_enabled:
        dynamic_enabled = env_flag("TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GRAPH", False)
    else:
        if max_dynamic_suffix is None:
            max_dynamic_suffix = env_int(
                "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MAX_SUFFIX",
                16,
                minimum=1,
            )
        dynamic_enabled = int(suffix_bucket) <= int(max_dynamic_suffix)
    if not dynamic_enabled:
        return exact_len
    bucket = exact_len
    min_bucket = env_int(
        "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MIN_CONTEXT",
        64,
        minimum=0,
    )
    if min_bucket > 0:
        bucket = max(bucket, min_bucket)
    bucket = 1 << (bucket - 1).bit_length()
    if max_seq_len is not None:
        bucket = min(bucket, max(1, int(max_seq_len)))
    if bucket < exact_len:
        return exact_len
    return -bucket


def _dynamic_prefix_prefill_max_suffix_for_policy(
    temperature: float,
    max_tokens: int | None,
) -> int | None:
    if "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MAX_SUFFIX" in os.environ:
        return None
    if temperature > 0.0:
        return 0
    if max_tokens is None or int(max_tokens) <= 0:
        return None
    short_max_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GREEDY_SHORT_MAX_TOKENS",
        128,
        minimum=1,
    )
    if int(max_tokens) <= short_max_tokens:
        return env_int(
            "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GREEDY_SHORT_MAX_SUFFIX",
            128,
            minimum=1,
        )
    large_min_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GREEDY_LARGE_MIN_TOKENS",
        400,
        minimum=0,
    )
    large_max_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GREEDY_LARGE_MAX_TOKENS",
        512,
        minimum=1,
    )
    if large_min_tokens < int(max_tokens) <= large_max_tokens:
        return env_int(
            "TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GREEDY_LARGE_MAX_SUFFIX",
            32,
            minimum=0,
        )
    return None


def _greedy_large_mixed_prefix_reuse_policy_enabled(
    temperature: float,
    max_tokens: int | None,
) -> bool:
    env_name = "TORCHINFERNO_CONTINUOUS_GREEDY_LARGE_MIXED_PREFIX_REUSE"
    if not env_flag(env_name, False):
        return False
    if temperature > 0.0 or max_tokens is None:
        return False
    target_tokens = env_int(
        "TORCHINFERNO_CONTINUOUS_GREEDY_LARGE_MIXED_PREFIX_REUSE_MAX_TOKENS",
        512,
        minimum=1,
    )
    return int(max_tokens) == target_tokens


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    arrival_step: int = 0
    eos_token_id: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()
    temperature: Optional[float] = None

    def __post_init__(self) -> None:
        stop_ids = {int(token_id) for token_id in self.stop_token_ids if int(token_id) >= 0}
        if self.eos_token_id is not None:
            eos_token_id = int(self.eos_token_id)
            object.__setattr__(self, "eos_token_id", eos_token_id)
            if eos_token_id >= 0:
                stop_ids.add(eos_token_id)
        object.__setattr__(self, "stop_token_ids", tuple(sorted(stop_ids)))
        if self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))

    def is_stop_token(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids


@dataclass(frozen=True)
class ServingResult:
    request_id: str
    tokens: tuple[int, ...]
    prefix_hit_tokens: int
    arrival_step: int
    started_step: int
    finished_step: int


@dataclass(frozen=True)
class ServingTokenEvent:
    request_id: str
    token: int
    step: int
    generated: int
    finished: bool


@dataclass
class ServingStats:
    prefill_model_calls: int = 0
    prefill_batches: int = 0
    prefill_tokens: int = 0
    decode_model_calls: int = 0
    decode_batches: int = 0
    decode_tokens: int = 0
    decode_active_tokens: int = 0
    ragged_decode_batches: int = 0
    ragged_decode_tokens: int = 0
    ragged_decode_active_tokens: int = 0
    ragged_decode_padding_tokens: int = 0
    decode_graph_hits: int = 0
    decode_graph_misses: int = 0
    decode_graph_captures: int = 0
    decode_graph_replays: int = 0
    decode_graph_capture_ms: float = 0.0
    decode_graph_replay_ms: float = 0.0
    decode_many_graph_calls: int = 0
    decode_many_graph_steps: int = 0
    decode_many_graph_model_tokens: int = 0
    decode_many_graph_ms: float = 0.0
    decode_many_calls: int = 0
    decode_many_steps: int = 0
    decode_many_model_tokens: int = 0
    decode_many_padded_tokens: int = 0
    decode_many_emitted_tokens: int = 0
    decode_many_skipped_tokens: int = 0
    decode_many_stop_finishes: int = 0
    decode_many_limit_finishes: int = 0
    decode_many_cpu_tokens_ms: float = 0.0
    decode_many_token_wait_ms: float = 0.0
    decode_many_token_materialize_ms: float = 0.0
    decode_many_model_ms: float = 0.0
    decode_many_model_gpu_ms: float = 0.0
    decode_many_state_syncs: int = 0
    decode_many_state_sync_skips: int = 0
    decode_many_tail_limited_calls: int = 0
    decode_many_tail_limited_steps: int = 0
    decode_many_min_active_skips: int = 0
    decode_many_shape_steps: dict[str, int] = field(default_factory=dict)
    decode_many_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_shape_padded_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_shape_emitted_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_shape_skipped_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_shape_stop_finishes: dict[str, int] = field(default_factory=dict)
    decode_many_shape_limit_finishes: dict[str, int] = field(default_factory=dict)
    decode_many_shape_model_ms: dict[str, float] = field(default_factory=dict)
    decode_many_shape_gpu_ms: dict[str, float] = field(default_factory=dict)
    decode_many_graph_shape_counts: dict[str, int] = field(default_factory=dict)
    decode_many_graph_shape_steps: dict[str, int] = field(default_factory=dict)
    decode_many_graph_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_graph_shape_ms: dict[str, float] = field(default_factory=dict)
    decode_many_step_window_counts: dict[str, int] = field(default_factory=dict)
    decode_many_step_window_model_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_step_window_padded_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_step_window_emitted_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_step_window_skipped_tokens: dict[str, int] = field(default_factory=dict)
    decode_many_step_window_model_ms: dict[str, float] = field(default_factory=dict)
    decode_many_step_window_cpu_tokens_ms: dict[str, float] = field(default_factory=dict)
    decode_many_step_window_token_wait_ms: dict[str, float] = field(default_factory=dict)
    decode_many_step_window_token_materialize_ms: dict[str, float] = field(default_factory=dict)
    prefix_reuse_requests: int = 0
    prefix_reuse_tokens: int = 0
    queued_requests: int = 0
    scheduler_steps: int = 0
    max_model_batch_size: int = 0
    persistent_cache_rows: int = 0
    prefill_admitted_requests: int = 0
    prefill_single_batches: int = 0
    prefill_plain_batches: int = 0
    prefill_prefix_reuse_batches: int = 0
    prefill_common_prefix_batches: int = 0
    prefill_padded_suffix_batches: int = 0
    prefill_packed_eager_calls: int = 0
    prefill_packed_eager_tokens: int = 0
    prefill_packed_eager_model_tokens: int = 0
    prefill_packed_eager_saved_tokens: int = 0
    prefill_packed_eager_ms: float = 0.0
    prefill_packed_flashinfer_calls: int = 0
    prefill_packed_flashinfer_tokens: int = 0
    prefill_packed_flashinfer_model_tokens: int = 0
    prefill_packed_flashinfer_saved_tokens: int = 0
    prefill_packed_flashinfer_ms: float = 0.0
    prefill_packed_candidate_calls: int = 0
    prefill_packed_candidate_tokens: int = 0
    prefill_packed_candidate_model_tokens: int = 0
    prefill_packed_candidate_saved_tokens: int = 0
    prefill_packed_candidate_groups: int = 0
    prefill_packed_fixed_capacity_attempts: int = 0
    prefill_packed_fixed_capacity_accepts: int = 0
    prefill_prefix_copy_skipped_batches: int = 0
    prefill_prefix_copy_skipped_tokens: int = 0
    prefill_row_indices_omitted_batches: int = 0
    prefill_row_indices_omitted_rows: int = 0
    prefill_row_indices_indexed_batches: int = 0
    prefill_row_indices_indexed_rows: int = 0
    prefill_graph_hits: int = 0
    prefill_graph_misses: int = 0
    prefill_graph_captures: int = 0
    prefill_graph_replays: int = 0
    prefill_wall_ms: float = 0.0
    prefill_copy_ms: float = 0.0
    prefill_forward_ms: float = 0.0
    prefill_graph_capture_ms: float = 0.0
    prefill_graph_replay_ms: float = 0.0
    prefill_graph_capture_gpu_ms: float = 0.0
    prefill_graph_replay_gpu_ms: float = 0.0
    prefill_setup_ms: float = 0.0
    prefill_sample_ms: float = 0.0
    prefill_sample_select_ms: float = 0.0
    prefill_sample_readback_ms: float = 0.0
    prefill_state_ms: float = 0.0
    prefill_state_seq_ms: float = 0.0
    prefill_state_store_ms: float = 0.0
    prefill_state_create_ms: float = 0.0
    prefill_shape_wall_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_copy_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_setup_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_forward_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_sample_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_sample_select_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_sample_readback_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_state_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_state_seq_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_state_store_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_state_create_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_active_requests: dict[str, int] = field(default_factory=dict)
    prefill_shape_model_rows: dict[str, int] = field(default_factory=dict)
    prefill_shape_active_tokens: dict[str, int] = field(default_factory=dict)
    prefill_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_shape_real_batch_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_suffix_length_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_route_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_route_active_tokens: dict[str, int] = field(default_factory=dict)
    prefill_shape_route_reuse_tokens: dict[str, int] = field(default_factory=dict)
    prefill_shape_graph_capture_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_graph_replay_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_graph_miss_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_graph_capture_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_graph_replay_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_graph_capture_gpu_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_graph_replay_gpu_ms: dict[str, float] = field(default_factory=dict)
    prefill_graph_capture_shape_gpu_ms: dict[str, float] = field(default_factory=dict)
    prefill_graph_replay_shape_gpu_ms: dict[str, float] = field(default_factory=dict)
    prefill_shape_row_indices_omitted_batches: dict[str, int] = field(default_factory=dict)
    prefill_shape_row_indices_omitted_rows: dict[str, int] = field(default_factory=dict)
    prefill_shape_row_indices_indexed_batches: dict[str, int] = field(default_factory=dict)
    prefill_shape_row_indices_indexed_rows: dict[str, int] = field(default_factory=dict)
    prefill_packed_eager_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_packed_eager_shape_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_eager_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_eager_shape_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_eager_shape_ms: dict[str, float] = field(default_factory=dict)
    prefill_packed_candidate_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_groups: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_max_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_max_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_max_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_shape_max_groups: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_signature_counts: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_signature_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_signature_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_signature_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_signature_groups: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_counts: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_model_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_groups: dict[str, int] = field(default_factory=dict)
    prefill_packed_candidate_pattern_slot_counts: dict[str, int] = field(default_factory=dict)
    prefill_packed_fixed_capacity_reject_reason_counts: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_candidate_calls: int = 0
    prefill_suffix_split_accepted_calls: int = 0
    prefill_suffix_split_rejected_calls: int = 0
    prefill_suffix_split_base_model_tokens: int = 0
    prefill_suffix_split_candidate_model_tokens: int = 0
    prefill_suffix_split_candidate_saved_tokens: int = 0
    prefill_suffix_split_accepted_base_model_tokens: int = 0
    prefill_suffix_split_accepted_model_tokens: int = 0
    prefill_suffix_split_accepted_saved_tokens: int = 0
    prefill_suffix_split_accepted_fragments: int = 0
    prefill_suffix_split_reject_reason_counts: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_candidate_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_candidate_shape_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_accepted_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_accepted_shape_saved_tokens: dict[str, int] = field(default_factory=dict)
    prefill_suffix_split_accepted_fragment_counts: dict[str, int] = field(default_factory=dict)
    decode_ragged_prepare_ms: float = 0.0
    decode_ragged_model_ms: float = 0.0
    decode_ragged_model_gpu_ms: float = 0.0
    decode_ragged_cpu_tokens_ms: float = 0.0
    decode_ragged_state_update_ms: float = 0.0
    decode_shape_model_ms: dict[str, float] = field(default_factory=dict)
    decode_shape_gpu_ms: dict[str, float] = field(default_factory=dict)
    decode_shape_cpu_tokens_ms: dict[str, float] = field(default_factory=dict)
    prompt_lookup_batches: int = 0
    prompt_lookup_requests: int = 0
    prompt_lookup_proposed_tokens: int = 0
    prompt_lookup_accepted_tokens: int = 0
    generated_prefix_store_requests: int = 0
    generated_prefix_reuse_requests: int = 0
    generated_prefix_reuse_tokens: int = 0
    full_prompt_store_requests: int = 0
    full_prompt_store_stored_requests: int = 0
    full_prompt_store_deferred_requests: int = 0
    full_prompt_store_deferred_tokens: int = 0
    full_prompt_store_adopted_requests: int = 0
    full_prompt_store_adopted_tokens: int = 0
    full_prompt_store_skipped_requests: int = 0
    full_prompt_store_skipped_tokens: int = 0
    full_prompt_reuse_candidate_stored_requests: int = 0
    full_prompt_reuse_candidate_stored_tokens: int = 0
    full_prompt_reuse_candidate_requests: int = 0
    full_prompt_reuse_candidate_tokens: int = 0
    full_prompt_reuse_candidate_extra_tokens: int = 0
    full_prompt_reuse_candidate_suffix_tokens: int = 0
    persistent_full_prompt_reuse_candidate_stored_requests: int = 0
    persistent_full_prompt_reuse_candidate_stored_tokens: int = 0
    persistent_full_prompt_reuse_candidate_requests: int = 0
    persistent_full_prompt_reuse_candidate_tokens: int = 0
    persistent_full_prompt_reuse_candidate_extra_tokens: int = 0
    persistent_full_prompt_reuse_candidate_suffix_tokens: int = 0
    repeated_sample_state_prepares: int = 0
    repeated_sample_state_hits: int = 0
    repeated_sample_state_tokens: int = 0
    prefix_reuse_route_counts: dict[str, int] = field(default_factory=dict)
    prefix_reuse_hit_token_counts: dict[str, int] = field(default_factory=dict)
    full_prompt_store_skip_reason_counts: dict[str, int] = field(default_factory=dict)
    full_prompt_store_skip_reason_tokens: dict[str, int] = field(default_factory=dict)
    full_prompt_reuse_candidate_token_counts: dict[str, int] = field(default_factory=dict)
    full_prompt_reuse_candidate_extra_token_counts: dict[str, int] = field(default_factory=dict)
    full_prompt_reuse_candidate_suffix_token_counts: dict[str, int] = field(default_factory=dict)
    persistent_full_prompt_reuse_candidate_token_counts: dict[str, int] = field(default_factory=dict)
    persistent_full_prompt_reuse_candidate_extra_token_counts: dict[str, int] = field(default_factory=dict)
    persistent_full_prompt_reuse_candidate_suffix_token_counts: dict[str, int] = field(default_factory=dict)
    prefill_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_graph_capture_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_graph_miss_shape_counts: dict[str, int] = field(default_factory=dict)
    prefill_graph_capture_shape_ms: dict[str, float] = field(default_factory=dict)
    prefill_graph_replay_shape_ms: dict[str, float] = field(default_factory=dict)
    decode_graph_capture_shape_counts: dict[str, int] = field(default_factory=dict)
    decode_graph_miss_shape_counts: dict[str, int] = field(default_factory=dict)
    decode_graph_capture_shape_ms: dict[str, float] = field(default_factory=dict)
    decode_graph_replay_shape_ms: dict[str, float] = field(default_factory=dict)
    decode_shape_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _QueuedRequest:
    original_index: int
    request: ServingRequest
    sequence: int


@dataclass
class _ReusablePrefix:
    route_id: Hashable
    tokens: tuple[int, ...]
    row: int
    logits: Tensor | None
    sample_state: object | None = None
    sample_temperature: float | None = None
    greedy_token: int | None = None


@dataclass
class _ActiveRequest:
    original_index: int
    request: ServingRequest
    tokens: list[int]
    generated: int
    row: int
    last_token: int
    seq_len: int
    prefix_hit_tokens: int
    started_step: int
    # Chunked prefill: a request stays in the 'prefilling' phase, advancing
    # prompt_cursor by a bounded chunk each step (so a long prompt does not stall
    # decode in one shot), until prompt_cursor == len(prompt), then it samples its
    # first token and flips to 'decoding'. Default 'decoding' preserves the
    # one-shot-prefill path when chunking is off.
    phase: str = "decoding"
    prompt_cursor: int = 0
    prefix_source_row: int = -1  # reusable-prefix source row (first chunk folds its copy)


def _contiguous_int_span(values: tuple[int, ...]) -> tuple[int, int] | None:
    if not values:
        return None
    start = values[0]
    for offset, value in enumerate(values):
        if value != start + offset:
            return None
    return start, start + len(values)


class ServingQueue:
    """Arrival-ordered queue with prefix-aware admission hooks."""

    def __init__(self, requests: list[tuple[int, ServingRequest]] | None = None) -> None:
        self._items: list[_QueuedRequest] = []
        self._next_sequence = 0
        if requests is not None:
            for original_index, request in requests:
                self.push(original_index, request)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, original_index: int, request: ServingRequest) -> None:
        self._items.append(_QueuedRequest(original_index, request, self._next_sequence))
        self._next_sequence += 1
        self._items.sort(key=self._arrival_key)

    def pop_admissible(
        self,
        *,
        step: int,
        capacity: int,
        token_budget: int | None = None,
        token_cost: Callable[[ServingRequest], int] | None = None,
        priority_key: Callable[[_QueuedRequest], tuple[object, ...]] | None = None,
    ) -> list[tuple[int, ServingRequest]]:
        if capacity <= 0:
            return []
        ready: list[_QueuedRequest] = []
        waiting: list[_QueuedRequest] = []
        for item in self._items:
            if item.request.arrival_step <= step:
                ready.append(item)
            else:
                waiting.append(item)
        if not ready:
            self._items = waiting
            return []
        if priority_key is None:
            ready.sort(key=self._arrival_key)
        else:
            ready.sort(key=priority_key)

        selected: list[_QueuedRequest] = []
        deferred: list[_QueuedRequest] = []
        remaining_budget = token_budget
        for item in ready:
            if len(selected) >= capacity:
                deferred.append(item)
                continue
            cost = max(1, token_cost(item.request) if token_cost is not None else len(item.request.prompt))
            if remaining_budget is not None and selected and cost > remaining_budget:
                deferred.append(item)
                continue
            selected.append(item)
            if remaining_budget is not None:
                remaining_budget -= cost
        self._items = [*deferred, *waiting]
        self._items.sort(key=self._arrival_key)
        return [(item.original_index, item.request) for item in selected]

    def ready_count(self, *, step: int) -> int:
        return sum(1 for item in self._items if item.request.arrival_step <= step)

    def next_arrival_step(self) -> int | None:
        if not self._items:
            return None
        return min(item.request.arrival_step for item in self._items)

    @staticmethod
    def _arrival_key(item: _QueuedRequest) -> tuple[int, int]:
        return (item.request.arrival_step, item.sequence)


class ContinuousBatchEngine:
    """Token-step continuous serving harness with persistent row-assigned cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        cache_backend: str = "dense",
        page_size: int = 16,
        temperature: float = 0.0,
        max_active_requests: int = 16,
        prefix_cache_capacity: int | None = None,
        prefill_token_budget: int | None = None,
        prefill_chunk_size: int | None = None,
        decode_first: bool = True,
        prefill_ready_before_decode: bool = False,
        prefill_ready_before_decode_active_cap: int | None = None,
        enable_ragged_decode: bool = True,
        store_reusable_prefixes: bool = True,
        store_full_prompt_prefixes: bool = True,
        pin_shared_prefix: bool = False,
        graph_prefill: bool = False,
        profile_timings: bool = False,
        admit_min_free_rows: int | None = None,
        admit_min_ready_requests: int | None = None,
        admit_per_step_cap: int | None = None,
        enable_decode_many: bool | None = None,
        decode_many_allow_stop: bool | None = None,
        decode_many_with_waiting: bool | None = None,
        decode_many_stop_tail_max_steps: int | None = None,
        decode_many_with_waiting_min_active: int | None = None,
        decode_many_min_active_pct: int | None = None,
        decode_many_sync_stops: bool | None = None,
        generated_prefix_cache: bool | None = None,
        greedy_large_mixed_prefix_reuse: bool | None = None,
        max_generation_tokens: int | None = None,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        if prefix_cache_capacity is not None and prefix_cache_capacity < 0:
            raise ValueError("prefix_cache_capacity must be non-negative")
        if prefill_token_budget is not None and prefill_token_budget < 1:
            raise ValueError("prefill_token_budget must be positive")
        model_to = getattr(model, "to", None)
        if callable(model_to):
            model = model_to(device)
        model_eval = getattr(model, "eval", None)
        if callable(model_eval):
            model = model_eval()
        self.model = model
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.temperature = temperature
        self.max_generation_tokens = (
            None if max_generation_tokens is None else max(0, int(max_generation_tokens))
        )
        self.max_active_requests = max_active_requests
        self.prefix_cache_capacity = max_active_requests if prefix_cache_capacity is None else prefix_cache_capacity
        self.prefill_token_budget = prefill_token_budget
        # Chunked prefill: when set, an admitted request prefills its suffix in
        # bounded chunks of this many tokens across steps (interleaved with
        # decode) instead of in one shot, bounding how long a long prompt's
        # prefill stalls the active decode batch. None preserves one-shot prefill.
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_first = decode_first
        self.prefill_ready_before_decode = prefill_ready_before_decode
        self.prefill_ready_before_decode_active_cap = (
            None
            if prefill_ready_before_decode_active_cap is None
            else max(0, int(prefill_ready_before_decode_active_cap))
        )
        self.enable_ragged_decode = enable_ragged_decode
        self._ragged_decode_buckets_enabled = env_flag(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKETS",
            True,
        )
        self._ragged_decode_bucket_capacity = env_int(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKET_CAPACITY",
            max_active_requests,
            minimum=1,
        )
        self.store_reusable_prefixes = store_reusable_prefixes
        self.store_full_prompt_prefixes = store_full_prompt_prefixes
        self._cached_repeated_sample_state_enabled = env_flag(
            "TORCHINFERNO_CONTINUOUS_CACHED_REPEATED_SAMPLE_STATE",
            True,
        )
        prepare_repeated_sample_state = getattr(model, "prepare_repeated_next_token_state", None)
        self._prepare_repeated_sample_state = (
            prepare_repeated_sample_state if callable(prepare_repeated_sample_state) else None
        )
        sample_repeated_from_state = getattr(model, "sample_repeated_next_token_from_state", None)
        self._sample_repeated_from_state = (
            sample_repeated_from_state if callable(sample_repeated_from_state) else None
        )
        prefill_token_logits_graph = getattr(model, "try_prefill_ragged_token_logits_graph", None)
        self._prefill_ragged_token_logits_graph = (
            prefill_token_logits_graph if callable(prefill_token_logits_graph) else None
        )
        prefill_token_graph = getattr(model, "try_prefill_ragged_token_graph", None)
        self._prefill_ragged_token_graph = (
            prefill_token_graph if callable(prefill_token_graph) else None
        )
        # When pinning is on, the engine caches ONLY shared common prefixes and
        # pins them against eviction, skipping per-request full-prompt stores
        # that would otherwise starve the prefix-row pool and shadow the shared
        # prefix in the radix tree (preventing cross-batch reuse).
        self.pin_shared_prefix = pin_shared_prefix
        # When graph_prefill is on, suffix prefills route through the model's
        # row_indices ragged-prefill LOGITS graph (try_prefill_ragged_logits_graph):
        # the suffix KV is scatter-written into the (scattered) active rows and
        # one logit row per request is gathered, replaying the graph across
        # changing row sets. Batch and suffix are padded to stable buckets so
        # graph shapes repeat. Per-row start positions handle mixed prefixes.
        self.graph_prefill = graph_prefill
        self.profile_timings = profile_timings
        self.admit_min_free_rows = (
            None if admit_min_free_rows is None else max(1, int(admit_min_free_rows))
        )
        self.admit_min_ready_requests = admit_min_ready_requests
        self.admit_per_step_cap = admit_per_step_cap
        self.generated_prefix_cache = generated_prefix_cache
        self.greedy_large_mixed_prefix_reuse = (
            None if greedy_large_mixed_prefix_reuse is None else bool(greedy_large_mixed_prefix_reuse)
        )
        self.enable_decode_many = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY", False)
            if enable_decode_many is None
            else bool(enable_decode_many)
        )
        self.decode_many_allow_stop = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP", False)
            if decode_many_allow_stop is None
            else bool(decode_many_allow_stop)
        )
        self.decode_many_with_waiting = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WITH_WAITING", False)
            if decode_many_with_waiting is None
            else bool(decode_many_with_waiting)
        )
        if decode_many_stop_tail_max_steps is None:
            decode_many_stop_tail_max_steps = env_int(
                "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_STOP_TAIL_MAX_STEPS",
                0,
                minimum=0,
            )
        self.decode_many_stop_tail_max_steps = max(0, int(decode_many_stop_tail_max_steps))
        if decode_many_with_waiting_min_active is None:
            decode_many_with_waiting_min_active = env_int(
                "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WITH_WAITING_MIN_ACTIVE",
                0,
                minimum=0,
            )
        self.decode_many_with_waiting_min_active = max(0, int(decode_many_with_waiting_min_active))
        if decode_many_min_active_pct is None:
            decode_many_min_active_pct = env_int(
                "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_MIN_ACTIVE_PCT",
                0,
                minimum=0,
            )
        self.decode_many_min_active_pct = min(100, max(0, int(decode_many_min_active_pct)))
        self.decode_many_sync_stops = (
            env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_SYNC_STOPS", False)
            if decode_many_sync_stops is None
            else bool(decode_many_sync_stops)
        )
        self.decode_many_graph = env_flag(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH",
            False,
        )
        self.decode_many_graph_min_steps = env_int(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH_MIN_STEPS",
            2,
            minimum=2,
        )
        self._decode_many_profile_step_window = env_int(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_PROFILE_STEP_WINDOW",
            16,
            minimum=1,
        )
        self.decode_many_sync_model_timings = env_flag(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_SYNC_MODEL_TIMINGS",
            False,
        )
        self.decode_many_async_readback = env_flag(
            "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ASYNC_READBACK",
            False,
        )
        self._fi_decode_graph_mode = _fi_decode_graph_mode()
        self._sampled_fi_decode_max_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_FI_DECODE_SAMPLED_MAX_TOKENS",
            256,
            minimum=0,
        )
        self._decode_capture_on_miss_override = (
            env_flag("TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE", False)
            if "TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE" in os.environ
            else None
        )
        self.unified_forward = bool(
            env_flag("TORCHINFERNO_CONTINUOUS_UNIFIED_FORWARD", False)
            and hasattr(model, "forward_step_flashinfer")
        )
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes: dict[Hashable, _ReusablePrefix] = {}
        self._full_prompt_reuse_candidate_cache = PrefixCacheIndex()
        self._full_prompt_reuse_candidate_order: list[Hashable] = []
        self._full_prompt_reuse_candidate_session_id = (
            _next_persistent_full_prompt_reuse_candidate_session()
        )
        self._pinned_prefix_routes: set[Hashable] = set()
        self.stats = ServingStats()
        self._cache: object | None = None
        self._cache_views: dict[tuple[int, ...], object] = {}
        self._reported_static_graph_miss = False
        self._packed_prefill_fixed_capacity_counts: dict[
            str,
            dict[tuple[int, int], int],
        ] = {}
        self._packed_prefill_fixed_capacity_seen: dict[str, int] = {}
        self._packed_prefill_fixed_capacity_stable_seen: dict[str, int] = {}
        self._packed_prefill_fixed_capacity_graph_none_keys: set[
            tuple[str, tuple[tuple[int, int], ...]]
        ] = set()
        self._prefill_token_graph_miss_keys: set[str] = set()
        self._free_active_rows: list[int] = []
        self._free_prefix_rows: list[int] = []
        self._row_seq_lens: list[int] = []
        self._row_cached_prefixes: list[tuple[int, ...] | None] = []
        self._device_index_tensors: dict[tuple[int, ...], Tensor] = {}
        self._graph_capture_on_miss_support: dict[object, bool] = {}
        self._prefix_order: list[Hashable] = []
        self._online_waiting: ServingQueue | None = None
        self._online_active: list[_ActiveRequest] = []
        self._online_prefilling: list[_ActiveRequest] = []
        self._online_step = 0
        self._online_next_index = 0
        self._pending_prefill_graph_events: list[tuple[object, ...]] = []
        self._pending_decode_ragged_model_events: list[tuple[object, ...]] = []

    def _dynamic_prefix_prefill_context_len(
        self,
        prefix_len: int,
        suffix_bucket: int,
        *,
        max_seq_len: int | None = None,
    ) -> int:
        return _dynamic_prefix_prefill_context_len(
            prefix_len,
            suffix_bucket,
            max_seq_len=max_seq_len,
            max_dynamic_suffix=_dynamic_prefix_prefill_max_suffix_for_policy(
                self.temperature,
                self.max_generation_tokens,
            ),
        )

    def _greedy_large_mixed_prefix_reuse_enabled(self) -> bool:
        if self.greedy_large_mixed_prefix_reuse is not None:
            return self.greedy_large_mixed_prefix_reuse
        return _greedy_large_mixed_prefix_reuse_policy_enabled(
            self.temperature,
            self.max_generation_tokens,
        )

    def _policy_or_env_flag(self, env_name: str) -> bool:
        if env_name in os.environ:
            return env_flag(env_name, False)
        return self._greedy_large_mixed_prefix_reuse_enabled()

    def _mixed_prefix_prefill_enabled(self) -> bool:
        return self._policy_or_env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL")

    def _mixed_prefix_dynamic_context_enabled(self) -> bool:
        return self._policy_or_env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT")

    def _mixed_prefix_dynamic_context_max_suffix(self) -> int:
        return env_int(
            "TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT_MAX_SUFFIX",
            32,
            minimum=1,
        )

    def _mixed_prefix_long_suffix_common_fallback_enabled(self) -> bool:
        return self._policy_or_env_flag(
            "TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_LONG_SUFFIX_COMMON_FALLBACK"
        )

    def _mixed_prefix_min_extra_tokens(self) -> int:
        return env_int(
            "TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_MIN_EXTRA_TOKENS",
            0,
            minimum=0,
        )

    def _mixed_prefix_max_extra_tokens(self) -> int | None:
        env_name = "TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_MAX_EXTRA_TOKENS"
        if env_name not in os.environ:
            return None
        return env_int(env_name, 0, minimum=0)

    def _non_common_prefix_graph_prefill_enabled(self) -> bool:
        return self._policy_or_env_flag("TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL")

    def _mixed_prefix_prefill_graph_enabled(self) -> bool:
        return self._policy_or_env_flag("TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL_GRAPH")

    @torch.inference_mode()
    def run(self, requests: list[ServingRequest]) -> list[ServingResult]:
        results, _events = self.run_with_events(requests, collect_events=False)
        return results

    @torch.inference_mode()
    def run_with_events(
        self,
        requests: list[ServingRequest],
        *,
        collect_events: bool = True,
    ) -> tuple[list[ServingResult], list[ServingTokenEvent]]:
        self._reset_run_state(requests)
        waiting = ServingQueue(list(enumerate(requests)))
        active: list[_ActiveRequest] = []
        indexed_results: list[tuple[int, ServingResult]] = []
        events: list[ServingTokenEvent] | None = [] if collect_events else None
        step = 0

        while waiting or active:
            self.stats.scheduler_steps += 1
            if self.decode_first and active:
                decoded_results, active = self._decode_active(active, step, events=events)
                indexed_results.extend(decoded_results)

            admitted = self._admit_ready_requests(waiting, step, len(active))
            if admitted:
                admitted_results, admitted_active = self._prefill_many(admitted, step, events=events)
                indexed_results.extend(admitted_results)
                active.extend(admitted_active)

            if not self.decode_first and active:
                decoded_results, active = self._decode_active(active, step + 1, events=events)
                indexed_results.extend(decoded_results)
            step += 1

            next_arrival_step = waiting.next_arrival_step()
            if next_arrival_step is not None and not active and next_arrival_step > step:
                step = next_arrival_step

        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])], events or []

    def iter_events(self, requests: list[ServingRequest]) -> Iterator[ServingTokenEvent]:
        with torch.inference_mode():
            self._reset_run_state(requests)
            waiting = ServingQueue(list(enumerate(requests)))
            active: list[_ActiveRequest] = []
            step = 0

            while waiting or active:
                self.stats.scheduler_steps += 1
                step_events: list[ServingTokenEvent] = []
                if self.decode_first and active:
                    _decoded_results, active = self._decode_active(active, step, events=step_events)

                admitted = self._admit_ready_requests(waiting, step, len(active))
                if admitted:
                    _admitted_results, admitted_active = self._prefill_many(admitted, step, events=step_events)
                    active.extend(admitted_active)

                if not self.decode_first and active:
                    _decoded_results, active = self._decode_active(active, step + 1, events=step_events)
                for event in step_events:
                    yield event
                step += 1

                next_arrival_step = waiting.next_arrival_step()
                if next_arrival_step is not None and not active and next_arrival_step > step:
                    step = next_arrival_step

    def start_online(
        self,
        *,
        max_seq_len: int,
        external_cache: object | None = None,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self._reset_capacity(max_seq_len=max_seq_len, queued_requests=0, external_cache=external_cache)
        self._online_waiting = ServingQueue()
        self._online_active = []
        self._online_step = 0
        self._online_next_index = 0

    def submit_online(self, request: ServingRequest) -> None:
        waiting = self._require_online_waiting()
        waiting.push(self._online_next_index, request)
        self._online_next_index += 1
        self.stats.queued_requests += 1

    @torch.inference_mode()
    def step_decode_only(self) -> list[ServingTokenEvent]:
        if not self._online_active:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        active = self._online_active
        self.stats.scheduler_steps += 1
        _decoded_results, active = self._decode_active(active, step, events=events)
        self._online_active = active
        self._online_step = step + 1
        return events

    @torch.inference_mode()
    def step_online_many(self, max_steps: int) -> tuple[list[ServingTokenEvent], int]:
        steps_left = max(1, int(max_steps))
        events: list[ServingTokenEvent] = []
        steps_run = 0
        while steps_left > 0 and self.has_online_work():
            if self._can_step_decode_many(steps_left):
                many_max_steps = self._decode_many_step_limit(steps_left)
                tail_limited = many_max_steps < steps_left
                if tail_limited:
                    self.stats.decode_many_tail_limited_calls += 1
                    self.stats.decode_many_tail_limited_steps += steps_left - many_max_steps
                many_events, many_steps = self._step_decode_only_many(many_max_steps)
                if many_steps <= 0:
                    break
                events.extend(many_events)
                steps_run += many_steps
                steps_left -= many_steps
                if tail_limited:
                    break
                continue
            step_events = self.step_online()
            events.extend(step_events)
            steps_run += 1
            steps_left -= 1
            break
        return events, steps_run

    def _can_step_decode_many(self, max_steps: int) -> bool:
        if max_steps <= 1:
            return False
        if not self.enable_decode_many:
            return False
        waiting = self._online_waiting
        if waiting:
            if not self.decode_many_with_waiting:
                return False
            if (
                self.decode_many_with_waiting_min_active > 0
                and len(self._online_active) < self.decode_many_with_waiting_min_active
            ):
                return False
        min_active_pct = int(getattr(self, "decode_many_min_active_pct", 0))
        if min_active_pct > 0 and len(self._online_active) * 100 < self.max_active_requests * min_active_pct:
            self.stats.decode_many_min_active_skips += 1
            return False
        if self._online_prefilling or not self._online_active:
            return False
        if self.unified_forward or not self.decode_first:
            return False
        if (
            any(self._state_temperature(state) > 0.0 for state in self._online_active)
            and not self.decode_many_allow_stop
        ):
            return False
        if (
            any(state.request.stop_token_ids for state in self._online_active)
            and not self.decode_many_allow_stop
        ):
            return False
        if self._generated_prefix_cache_enabled() or self._prompt_lookup_decode_enabled():
            return False
        return self._can_decode_ragged(self._online_active)

    def _decode_many_step_limit(self, max_steps: int) -> int:
        requested_steps = max(1, int(max_steps))
        tail_max_steps = max(0, int(self.decode_many_stop_tail_max_steps))
        if tail_max_steps <= 0 or requested_steps <= tail_max_steps:
            return requested_steps
        active = self._online_active
        if not active or len(active) >= self.max_active_requests:
            return requested_steps
        if not self.decode_many_allow_stop:
            return requested_steps
        if not any(state.request.stop_token_ids for state in active):
            return requested_steps
        return max(1, min(requested_steps, tail_max_steps))

    def _decode_many_graph_step_count(
        self,
        active: list[_ActiveRequest],
        max_steps: int,
    ) -> int:
        if (
            not self.decode_many_graph
            or max_steps < self.decode_many_graph_min_steps
            or not active
            or any(self._state_temperature(state) > 0.0 for state in active)
        ):
            return 0
        remaining = [
            int(state.request.max_new_tokens) - int(state.generated)
            for state in active
        ]
        if not remaining:
            return 0
        step_count = min(int(max_steps), min(remaining))
        if step_count < self.decode_many_graph_min_steps:
            return 0
        return step_count

    def _try_decode_many_graph_tokens(
        self,
        states: list[_ActiveRequest],
        max_steps: int,
        *,
        profile_source: str | None,
    ) -> tuple[Tensor, int, str | None, float] | None:
        step_count = self._decode_many_graph_step_count(states, max_steps)
        if step_count <= 0:
            return None
        graph = getattr(self.model, "try_decode_ragged_token_graph_many", None)
        if graph is None:
            return None
        rows = [state.row for state in states]
        n_active = len(states)
        decode_rows = self._ragged_decode_bucket_rows(rows)
        n_padded = len(decode_rows)
        contiguous_row_set = n_active == n_padded and sorted(decode_rows) == list(range(n_padded))
        state_order_indices: Tensor | None = None
        if contiguous_row_set:
            row_indices = None
            input_ids = self._ensure_gpu_token_buf()[:n_padded].view(n_padded, 1)
            if any(row != index for index, row in enumerate(rows)):
                state_order_indices = self._device_index_tensor(tuple(rows))
        else:
            if not env_flag(
                "TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH_ALLOW_ROW_INDICES",
                False,
            ):
                return None
            row_indices = self._device_index_tensor(tuple(decode_rows))
            input_ids = self._ensure_gpu_token_buf().index_select(0, row_indices).view(n_padded, 1)
        seq_lens = self._decode_many_seq_lens_tensor(states, decode_rows)
        shape_key = f"decode_many:b{n_active}/{n_padded}" if self.profile_timings else None

        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        try:
            kwargs: dict[str, object] = {
                "seq_lens": seq_lens,
                "row_indices": row_indices,
                "steps": step_count,
                "temperature": 0.0,
            }
            if self._graph_accepts_capture_on_miss(graph):
                kwargs["capture_on_miss"] = self._decode_capture_on_miss()
            token_matrix = graph(input_ids, self._require_cache(), **kwargs)
        except Exception as exc:
            warn_optional_failure("continuous.decode_many_graph", exc)
            return None
        finally:
            self._stop_decode_ragged_model_gpu_timer(
                gpu_model_events,
                shape_key=shape_key,
                profile_source=profile_source,
            )
        if token_matrix is None:
            return None
        if token_matrix.ndim != 2 or token_matrix.shape != (step_count, n_padded):
            return None
        token_matrix = token_matrix.to(self.device)
        last_tokens = token_matrix[step_count - 1, :n_active]
        if row_indices is None:
            self._ensure_gpu_token_buf()[:n_active].copy_(last_tokens)
            seq_lens_buf = getattr(self, "_gpu_seq_lens", None)
            if seq_lens_buf is not None:
                seq_lens_buf[:n_active].add_(step_count)
            if state_order_indices is not None:
                token_matrix = token_matrix.index_select(1, state_order_indices)
        else:
            active_row_indices = row_indices[:n_active]
            self._ensure_gpu_token_buf().index_copy_(0, active_row_indices, last_tokens)
            self._advance_gpu_seq_lens(active_row_indices, amount=step_count)
        self.stats.decode_graph_hits += step_count
        self.stats.decode_many_graph_calls += 1
        self.stats.decode_many_graph_steps += step_count
        self.stats.decode_many_graph_model_tokens += n_padded * step_count
        model_elapsed_ms = 0.0
        if self.profile_timings:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_elapsed_ms = (time.perf_counter() - model_start_s) * 1000.0
            self.stats.decode_ragged_model_ms += model_elapsed_ms
            self.stats.decode_many_graph_ms += model_elapsed_ms
            if shape_key is not None:
                self._record_shape_count(
                    self.stats.decode_many_graph_shape_counts,
                    shape_key,
                )
                self._record_shape_total(
                    self.stats.decode_many_graph_shape_steps,
                    shape_key,
                    step_count,
                )
                self._record_shape_total(
                    self.stats.decode_many_graph_shape_model_tokens,
                    shape_key,
                    n_padded * step_count,
                )
                self._record_shape_time(
                    self.stats.decode_many_graph_shape_ms,
                    shape_key,
                    model_elapsed_ms,
                )
                self._record_shape_time(
                    self.stats.decode_shape_model_ms,
                    shape_key,
                    model_elapsed_ms,
                )
        return token_matrix, step_count, shape_key, model_elapsed_ms

    def _step_decode_only_many(self, max_steps: int) -> tuple[list[ServingTokenEvent], int]:
        active = list(self._online_active)
        if not active:
            return [], 0

        state_signature = self._make_decode_many_gpu_state_signature(active)
        if self._decode_many_gpu_state_is_current(state_signature):
            self.stats.decode_many_state_sync_skips += 1
        else:
            self._sync_gpu_last_tokens_from_states(active)
            self._sync_gpu_seq_lens_from_states(active)
            self._decode_many_gpu_state_signature = state_signature
            self.stats.decode_many_state_syncs += 1
        if self._decode_many_sync_stops_enabled(active):
            return self._step_decode_only_many_sync_stops(active, max_steps)
        records: list[
            tuple[list[_ActiveRequest], int, list[int], list[bool], str | None, int, str | None]
        ] = []
        token_scratch = self._ensure_decode_many_token_scratch(max_steps * max(1, self.max_active_requests))
        if self._decode_many_async_readback_enabled():
            self._ensure_decode_many_cpu_token_scratch(token_scratch.numel())
        token_offset = 0
        shape_parts: list[tuple[str, int, int]] = []
        steps_run = 0
        record_sync_model_timing = self._decode_many_records_sync_model_timing()
        record_shape_counts = self.profile_timings or _queue_profile_counts_enabled()

        def record_count(counts: dict[str, int], key: str) -> None:
            counts[key] = counts.get(key, 0) + 1

        def record_total(counts: dict[str, int], key: str, amount: int) -> None:
            counts[key] = counts.get(key, 0) + int(amount)

        while steps_run < max_steps and active and self._can_decode_ragged(active):
            step = self._online_step + steps_run
            states = list(active)
            self.stats.scheduler_steps += 1
            many_graph = self._try_decode_many_graph_tokens(
                states,
                max_steps - steps_run,
                profile_source="decode_many",
            )
            if many_graph is None:
                model_ms_before = (
                    self.stats.decode_ragged_model_ms
                    if self.profile_timings and record_sync_model_timing
                    else 0.0
                )
                next_token_tensor = self._decode_ragged_batch_token_tensor(
                    states,
                    profile_source="decode_many",
                )
                token_matrix = next_token_tensor[: len(states)].view(1, len(states))
                graph_steps = 1
                graph_model_elapsed_ms = (
                    max(0.0, self.stats.decode_ragged_model_ms - model_ms_before)
                    if self.profile_timings and record_sync_model_timing
                    else 0.0
                )
                graph_shape_key: str | None = None
                graph_shape_model_tokens = int(next_token_tensor.numel())
                record_model_call_for_steps = False
                record_model_timing_for_steps = record_sync_model_timing
            else:
                token_matrix, graph_steps, graph_shape_key, graph_model_elapsed_ms = many_graph
                graph_shape_model_tokens = int(token_matrix.size(1))
                record_model_call_for_steps = True
                record_model_timing_for_steps = True

            for graph_step in range(graph_steps):
                if graph_step > 0:
                    self.stats.scheduler_steps += 1
                step_states = list(active)
                active_tokens = len(step_states)
                token_start = token_offset
                token_scratch[token_start : token_start + active_tokens].copy_(
                    token_matrix[graph_step, :active_tokens]
                )
                token_offset += active_tokens
                self._maybe_schedule_decode_many_readback(
                    token_scratch[token_start:token_offset],
                    token_start,
                    token_offset,
                )
                shape_key: str | None = None
                step_window_key: str | None = None
                shape_model_tokens = graph_shape_model_tokens
                model_elapsed_ms = 0.0
                if record_shape_counts:
                    shape_key = graph_shape_key or f"decode_many:b{active_tokens}/{shape_model_tokens}"
                    if self.profile_timings and record_model_call_for_steps:
                        self._record_shape_count(self.stats.decode_shape_counts, shape_key)
                    record_count(self.stats.decode_many_shape_steps, shape_key)
                    if self.profile_timings:
                        shape_parts.append((shape_key, active_tokens, shape_model_tokens))
                    model_elapsed_ms = graph_model_elapsed_ms / max(1, graph_steps)
                    if self.profile_timings and record_model_timing_for_steps:
                        self.stats.decode_many_model_ms += model_elapsed_ms
                        self._record_shape_time(
                            self.stats.decode_many_shape_model_ms,
                            shape_key,
                            model_elapsed_ms,
                        )

                state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
                generated_after: list[int] = []
                finished_by_limit: list[bool] = []
                next_active: list[_ActiveRequest] = []
                for state in step_states:
                    state.generated += 1
                    next_seq_len = state.seq_len + 1
                    self._remember_row_seq_len(state.row, next_seq_len)
                    state.seq_len = next_seq_len
                    generated_after.append(state.generated)
                    done = state.generated >= state.request.max_new_tokens
                    finished_by_limit.append(done)
                    if not done:
                        next_active.append(state)
                if self.profile_timings:
                    self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
                if shape_key is not None:
                    step_window_key = self._decode_many_step_window_key(generated_after, shape_key)
                    record_count(
                        self.stats.decode_many_step_window_counts,
                        step_window_key,
                    )
                    record_total(
                        self.stats.decode_many_step_window_model_tokens,
                        step_window_key,
                        active_tokens,
                    )
                    record_total(
                        self.stats.decode_many_step_window_padded_tokens,
                        step_window_key,
                        shape_model_tokens,
                    )
                    if self.profile_timings:
                        if record_model_timing_for_steps:
                            self._record_shape_time(
                                self.stats.decode_many_step_window_model_ms,
                                step_window_key,
                                model_elapsed_ms,
                            )
                        else:
                            self._attach_latest_decode_many_gpu_window(
                                step_window_key,
                                shape_model_tokens,
                                shape_key=shape_key,
                            )
                records.append(
                    (
                        step_states,
                        step + graph_step,
                        generated_after,
                        finished_by_limit,
                        shape_key,
                        shape_model_tokens,
                        step_window_key,
                    )
                )
                if record_model_call_for_steps:
                    self._record_model_call(
                        "decode",
                        shape_model_tokens,
                        tokens=shape_model_tokens,
                        ragged=True,
                        active_tokens=active_tokens,
                    )
                active = next_active
                steps_run += 1
                if steps_run >= max_steps or not active:
                    break

        if steps_run <= 0 or token_offset <= 0:
            return [], 0

        model_tokens = token_offset
        padded_model_tokens = sum(record[5] for record in records)
        if self.profile_timings:
            cpu_tokens_start_s = time.perf_counter()
            (
                flat_tokens,
                token_wait_ms,
                token_materialize_ms,
            ) = self._decode_many_tokens_to_list_profiled(token_scratch, token_offset)
            cpu_elapsed_ms = (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self.stats.decode_ragged_cpu_tokens_ms += cpu_elapsed_ms
            self.stats.decode_many_cpu_tokens_ms += cpu_elapsed_ms
            self.stats.decode_many_token_wait_ms += token_wait_ms
            self.stats.decode_many_token_materialize_ms += token_materialize_ms
            shape_token_count = sum(active_count for _shape_key, active_count, _model_count in shape_parts)
            if shape_token_count > 0:
                for shape_key, active_count, _model_count in shape_parts:
                    self._record_shape_time(
                        self.stats.decode_shape_cpu_tokens_ms,
                        shape_key,
                        cpu_elapsed_ms * (active_count / shape_token_count),
                    )
                for (
                    states,
                    _step,
                    _generated_after,
                    _finished_by_limit,
                    _shape_key,
                    _shape_model_tokens,
                    step_window_key,
                ) in records:
                    if step_window_key is None:
                        continue
                    self._record_shape_time(
                        self.stats.decode_many_step_window_cpu_tokens_ms,
                        step_window_key,
                        cpu_elapsed_ms * (len(states) / shape_token_count),
                    )
                    self._record_shape_time(
                        self.stats.decode_many_step_window_token_wait_ms,
                        step_window_key,
                        token_wait_ms * (len(states) / shape_token_count),
                    )
                    self._record_shape_time(
                        self.stats.decode_many_step_window_token_materialize_ms,
                        step_window_key,
                        token_materialize_ms * (len(states) / shape_token_count),
                    )
            self._flush_decode_ragged_model_gpu_timers()
        else:
            flat_tokens = self._decode_many_tokens_to_list(token_scratch, token_offset)

        events: list[ServingTokenEvent] = []
        terminated: set[int] = set()
        skipped_tokens = 0
        stop_finishes = 0
        limit_finishes = 0
        offset = 0
        for (
            states,
            step,
            generated_after,
            finished_by_limit,
            shape_key,
            shape_model_tokens,
            step_window_key,
        ) in records:
            row_tokens = flat_tokens[offset : offset + len(states)]
            offset += len(states)
            record_emitted_tokens = 0
            record_skipped_tokens = 0
            record_stop_finishes = 0
            record_limit_finishes = 0
            for state, token_value, generated, limit_finished in zip(
                states,
                row_tokens,
                generated_after,
                finished_by_limit,
            ):
                state_id = id(state)
                if state_id in terminated:
                    skipped_tokens += 1
                    record_skipped_tokens += 1
                    continue
                token = int(token_value)
                state.tokens.append(token)
                state.last_token = token
                stop_finished = state.request.is_stop_token(token)
                finished = bool(limit_finished or stop_finished)
                events.append(
                    ServingTokenEvent(
                        request_id=state.request.request_id,
                        token=token,
                        step=step,
                        generated=generated,
                        finished=finished,
                    )
                )
                record_emitted_tokens += 1
                if finished:
                    if stop_finished:
                        stop_finishes += 1
                        record_stop_finishes += 1
                    if limit_finished:
                        limit_finishes += 1
                        record_limit_finishes += 1
                    terminated.add(state_id)
                    self._finish_and_release(state, step)
            if shape_key is not None:
                record_total(
                    self.stats.decode_many_shape_model_tokens,
                    shape_key,
                    len(states),
                )
                record_total(
                    self.stats.decode_many_shape_padded_tokens,
                    shape_key,
                    shape_model_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_emitted_tokens,
                    shape_key,
                    record_emitted_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_skipped_tokens,
                    shape_key,
                    record_skipped_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_stop_finishes,
                    shape_key,
                    record_stop_finishes,
                )
                record_total(
                    self.stats.decode_many_shape_limit_finishes,
                    shape_key,
                    record_limit_finishes,
                )
            if step_window_key is not None:
                record_total(
                    self.stats.decode_many_step_window_emitted_tokens,
                    step_window_key,
                    record_emitted_tokens,
                )
                record_total(
                    self.stats.decode_many_step_window_skipped_tokens,
                    step_window_key,
                    record_skipped_tokens,
                )

        self.stats.decode_many_calls += 1
        self.stats.decode_many_steps += steps_run
        self.stats.decode_many_model_tokens += int(model_tokens)
        self.stats.decode_many_padded_tokens += int(padded_model_tokens)
        self.stats.decode_many_emitted_tokens += len(events)
        self.stats.decode_many_skipped_tokens += skipped_tokens
        self.stats.decode_many_stop_finishes += stop_finishes
        self.stats.decode_many_limit_finishes += limit_finishes
        self._online_active = [state for state in active if id(state) not in terminated]
        self._decode_many_gpu_state_signature = self._make_decode_many_gpu_state_signature(
            self._online_active
        )
        self._online_step += steps_run
        return events, steps_run

    def _decode_many_sync_stops_enabled(self, active: list[_ActiveRequest]) -> bool:
        return bool(
            self.decode_many_sync_stops
            and self.decode_many_allow_stop
            and active
            and any(state.request.stop_token_ids for state in active)
        )

    def _step_decode_only_many_sync_stops(
        self,
        active: list[_ActiveRequest],
        max_steps: int,
    ) -> tuple[list[ServingTokenEvent], int]:
        events: list[ServingTokenEvent] = []
        steps_run = 0
        model_tokens = 0
        padded_model_tokens = 0
        stop_finishes = 0
        limit_finishes = 0
        record_sync_model_timing = self._decode_many_records_sync_model_timing()
        record_shape_counts = self.profile_timings or _queue_profile_counts_enabled()

        def record_count(counts: dict[str, int], key: str) -> None:
            counts[key] = counts.get(key, 0) + 1

        def record_total(counts: dict[str, int], key: str, amount: int) -> None:
            counts[key] = counts.get(key, 0) + int(amount)

        while steps_run < max_steps and active and self._can_decode_ragged(active):
            step = self._online_step + steps_run
            states = list(active)
            self.stats.scheduler_steps += 1
            model_ms_before = (
                self.stats.decode_ragged_model_ms
                if self.profile_timings and record_sync_model_timing
                else 0.0
            )
            next_token_tensor = self._decode_ragged_batch_token_tensor(
                states,
                profile_source="decode_many",
            )
            active_tokens = len(states)
            shape_model_tokens = int(next_token_tensor.numel())
            shape_key: str | None = None
            step_window_key: str | None = None
            model_elapsed_ms = 0.0
            if record_shape_counts:
                shape_key = f"decode_many:b{active_tokens}/{shape_model_tokens}"
                if self.profile_timings and record_sync_model_timing:
                    model_elapsed_ms = max(0.0, self.stats.decode_ragged_model_ms - model_ms_before)
                    self.stats.decode_many_model_ms += model_elapsed_ms
                    self._record_shape_time(
                        self.stats.decode_many_shape_model_ms,
                        shape_key,
                        model_elapsed_ms,
                    )

            cpu_elapsed_ms = 0.0
            if self.profile_timings:
                cpu_tokens_start_s = time.perf_counter()
                (
                    row_tokens,
                    token_wait_ms,
                    token_materialize_ms,
                ) = self._token_tensor_to_list_profiled(next_token_tensor[:active_tokens])
                cpu_elapsed_ms = (time.perf_counter() - cpu_tokens_start_s) * 1000.0
                self.stats.decode_ragged_cpu_tokens_ms += cpu_elapsed_ms
                self.stats.decode_many_cpu_tokens_ms += cpu_elapsed_ms
                self.stats.decode_many_token_wait_ms += token_wait_ms
                self.stats.decode_many_token_materialize_ms += token_materialize_ms
                if shape_key is not None:
                    self._record_shape_time(
                        self.stats.decode_shape_cpu_tokens_ms,
                        shape_key,
                        cpu_elapsed_ms,
                    )
            else:
                row_tokens = next_token_tensor[:active_tokens].detach().cpu().tolist()

            state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
            generated_after: list[int] = []
            next_active: list[_ActiveRequest] = []
            record_stop_finishes = 0
            record_limit_finishes = 0
            for state, token_value in zip(states, row_tokens):
                token = int(token_value)
                state.tokens.append(token)
                state.last_token = token
                state.generated += 1
                next_seq_len = state.seq_len + 1
                self._remember_row_seq_len(state.row, next_seq_len)
                state.seq_len = next_seq_len
                generated_after.append(state.generated)
                limit_finished = state.generated >= state.request.max_new_tokens
                stop_finished = state.request.is_stop_token(token)
                finished = bool(limit_finished or stop_finished)
                events.append(
                    ServingTokenEvent(
                        request_id=state.request.request_id,
                        token=token,
                        step=step,
                        generated=state.generated,
                        finished=finished,
                    )
                )
                if finished:
                    if stop_finished:
                        stop_finishes += 1
                        record_stop_finishes += 1
                    if limit_finished:
                        limit_finishes += 1
                        record_limit_finishes += 1
                    self._finish_and_release(state, step)
                else:
                    next_active.append(state)
            if self.profile_timings:
                self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
                self._flush_decode_ragged_model_gpu_timers()
            if shape_key is not None:
                step_window_key = self._decode_many_step_window_key(generated_after, shape_key)
                record_count(
                    self.stats.decode_many_step_window_counts,
                    step_window_key,
                )
                record_total(
                    self.stats.decode_many_step_window_model_tokens,
                    step_window_key,
                    active_tokens,
                )
                record_total(
                    self.stats.decode_many_step_window_padded_tokens,
                    step_window_key,
                    shape_model_tokens,
                )
                record_total(
                    self.stats.decode_many_step_window_emitted_tokens,
                    step_window_key,
                    active_tokens,
                )
                record_total(
                    self.stats.decode_many_step_window_skipped_tokens,
                    step_window_key,
                    0,
                )
                if self.profile_timings:
                    if record_sync_model_timing:
                        self._record_shape_time(
                            self.stats.decode_many_step_window_model_ms,
                            step_window_key,
                            model_elapsed_ms,
                        )
                    else:
                        self._attach_latest_decode_many_gpu_window(
                            step_window_key,
                            shape_model_tokens,
                            shape_key=shape_key,
                        )
                    self._record_shape_time(
                        self.stats.decode_many_step_window_cpu_tokens_ms,
                        step_window_key,
                        cpu_elapsed_ms,
                    )
                    self._record_shape_time(
                        self.stats.decode_many_step_window_token_wait_ms,
                        step_window_key,
                        token_wait_ms,
                    )
                    self._record_shape_time(
                        self.stats.decode_many_step_window_token_materialize_ms,
                        step_window_key,
                        token_materialize_ms,
                    )
            if shape_key is not None:
                record_count(
                    self.stats.decode_many_shape_steps,
                    shape_key,
                )
                record_total(
                    self.stats.decode_many_shape_model_tokens,
                    shape_key,
                    active_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_padded_tokens,
                    shape_key,
                    shape_model_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_emitted_tokens,
                    shape_key,
                    active_tokens,
                )
                record_total(
                    self.stats.decode_many_shape_skipped_tokens,
                    shape_key,
                    0,
                )
                record_total(
                    self.stats.decode_many_shape_stop_finishes,
                    shape_key,
                    record_stop_finishes,
                )
                record_total(
                    self.stats.decode_many_shape_limit_finishes,
                    shape_key,
                    record_limit_finishes,
                )

            model_tokens += active_tokens
            padded_model_tokens += shape_model_tokens
            active = next_active
            steps_run += 1

        if steps_run <= 0:
            return [], 0

        self.stats.decode_many_calls += 1
        self.stats.decode_many_steps += steps_run
        self.stats.decode_many_model_tokens += int(model_tokens)
        self.stats.decode_many_padded_tokens += int(padded_model_tokens)
        self.stats.decode_many_emitted_tokens += len(events)
        self.stats.decode_many_stop_finishes += stop_finishes
        self.stats.decode_many_limit_finishes += limit_finishes
        self._online_active = active
        self._decode_many_gpu_state_signature = self._make_decode_many_gpu_state_signature(
            self._online_active
        )
        self._online_step += steps_run
        return events, steps_run

    def has_online_work(self) -> bool:
        waiting = self._online_waiting
        return bool(waiting) or bool(self._online_active) or bool(self._online_prefilling)

    def has_online_waiting_requests(self) -> bool:
        return bool(self._online_waiting)

    def online_active_min_generated(self) -> int | None:
        if not self._online_active:
            return None
        return min(state.generated for state in self._online_active)

    @torch.inference_mode()
    def step_online(self) -> list[ServingTokenEvent]:
        if self.unified_forward:
            return self._step_online_unified()
        waiting = self._require_online_waiting()
        if not waiting and not self._online_active and not self._online_prefilling:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        active = self._online_active
        self.stats.scheduler_steps += 1
        decode_before_admit = bool(self.decode_first and active)
        decode_after_admit = not self.decode_first
        if decode_before_admit and self.prefill_ready_before_decode and waiting:
            active_cap = self.prefill_ready_before_decode_active_cap
            prefill_before_decode = active_cap is None or len(active) <= active_cap
        else:
            prefill_before_decode = False
        if prefill_before_decode:
            decode_before_admit = False
            decode_after_admit = True
        if decode_before_admit:
            _da_start = time.perf_counter() if self.profile_timings else 0.0
            _decoded_results, active = self._decode_active(active, step, events=events)
            if self.profile_timings:
                self.stats._decode_active_ms = getattr(self.stats, '_decode_active_ms', 0.0) + (time.perf_counter() - _da_start) * 1000.0

        if self.prefill_chunk_size and self._online_prefilling:
            active.extend(self._advance_prefilling(step, events=events))

        _admit_start = time.perf_counter() if self.profile_timings else 0.0
        occupied = len(active) + len(self._online_prefilling)
        admitted = self._admit_ready_requests(waiting, step, occupied)
        if self.profile_timings:
            self.stats._admit_ms = getattr(self.stats, '_admit_ms', 0.0) + (time.perf_counter() - _admit_start) * 1000.0
        if admitted:
            if self.prefill_chunk_size:
                active.extend(self._admit_to_prefilling(admitted, step, events=events))
            else:
                _admitted_results, admitted_active = self._prefill_many(admitted, step, events=events)
                active.extend(admitted_active)

        if active and decode_after_admit:
            _decoded_results, active = self._decode_active(active, step + 1, events=events)
        if self.decode_first and active:
            active = self._release_online_prefill_finished(active, step)
        self._online_active = active
        self._online_step = step + 1

        next_arrival_step = waiting.next_arrival_step()
        idle = not active and not self._online_prefilling
        if next_arrival_step is not None and idle and next_arrival_step > self._online_step:
            self._online_step = next_arrival_step
        return events

    @torch.inference_mode()
    def _step_online_unified(self) -> list[ServingTokenEvent]:
        waiting = self._require_online_waiting()
        if not waiting and not self._online_active and not self._online_prefilling:
            return []
        events: list[ServingTokenEvent] = []
        step = self._online_step
        self.stats.scheduler_steps += 1
        cache = self._require_cache()

        decode_states: list[_ActiveRequest] = []
        for state in self._online_active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                self._finish_and_release(state, step)
                self._record_token_event(events, state, state.last_token, step, finished=True)
            else:
                decode_states.append(state)

        occupied = len(decode_states) + len(self._online_prefilling)
        admitted = self._admit_ready_requests(waiting, step, occupied)

        prefill_states: list[_ActiveRequest] = []
        exact_reuse_group: list[tuple[int, ServingRequest, int, _ReusablePrefix]] = []
        for original_index, request in admitted:
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            prefix_hit = match.depth if (reusable is not None and match.depth > 0) else 0
            if reusable is not None and prefix_hit >= len(request.prompt) and reusable.logits is not None:
                exact_reuse_group.append((original_index, request, prefix_hit, reusable))
                continue
            if reusable is not None and prefix_hit >= len(request.prompt):
                reusable = None
                prefix_hit = 0
            self._record_full_prompt_reuse_candidate_lookup(request.prompt, prefix_hit)
            row = self._acquire_active_row()
            if reusable is not None and prefix_hit > 0:
                self._copy_prefix(reusable.row, row, prefix_hit)
                self._record_prefix_reuse(prefix_hit, reusable)
            suffix = request.prompt[prefix_hit:]
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=list(request.prompt),
                generated=0,
                row=row,
                last_token=request.prompt[-1] if request.prompt else 0,
                seq_len=prefix_hit,
                prefix_hit_tokens=prefix_hit,
                started_step=step,
            )
            state._prefill_suffix = suffix
            prefill_states.append(state)
        self.stats.prefill_admitted_requests += len(prefill_states) + len(exact_reuse_group)

        exact_reuse_active: list[_ActiveRequest] = []
        exact_reuse_processed = False

        def finish_exact_reuse() -> list[_ActiveRequest]:
            nonlocal exact_reuse_active, exact_reuse_processed
            if exact_reuse_group and not exact_reuse_processed:
                exact_reuse_active = self._prefill_exact_prefix_batch(
                    exact_reuse_group,
                    step,
                    events=events,
                )
                exact_reuse_processed = True
            return exact_reuse_active

        if not decode_states and not prefill_states and not self._online_prefilling:
            self._online_active = finish_exact_reuse()
            self._online_step = step + 1
            return events

        if prefill_states or self._online_prefilling:
            all_prefilling = list(self._online_prefilling) + prefill_states
            batch_rows = []
            batch_q_lens = []
            batch_input_ids: list[list[int]] = []
            batch_write_pos: list[list[int]] = []
            batch_logit_pos = []
            batch_seq_lens = []
            max_q = 1
            batch_is_decode: list[bool] = []
            batch_states: list[_ActiveRequest] = []

            for state in decode_states:
                batch_rows.append(state.row)
                batch_q_lens.append(1)
                batch_input_ids.append([state.last_token])
                batch_write_pos.append([state.seq_len])
                batch_logit_pos.append(0)
                batch_seq_lens.append(state.seq_len)
                batch_is_decode.append(True)
                batch_states.append(state)

            chunk = self.prefill_chunk_size
            still_prefilling: list[_ActiveRequest] = []
            for state in all_prefilling:
                suffix = getattr(state, '_prefill_suffix', None)
                if suffix is None:
                    suffix = state.request.prompt[state.seq_len:]
                if chunk and len(suffix) > chunk:
                    cur_suffix = suffix[:chunk]
                    state._prefill_suffix = suffix[chunk:]
                    still_prefilling.append(state)
                else:
                    cur_suffix = suffix
                    state._prefill_suffix = None
                cursor = state.seq_len
                q_len = len(cur_suffix)
                if q_len == 0:
                    continue
                batch_rows.append(state.row)
                batch_q_lens.append(q_len)
                batch_input_ids.append(list(cur_suffix))
                batch_write_pos.append(list(range(cursor, cursor + q_len)))
                batch_logit_pos.append(q_len - 1)
                batch_seq_lens.append(cursor)
                batch_is_decode.append(False)
                batch_states.append(state)
                max_q = max(max_q, q_len)

            if not batch_rows:
                self._online_active = decode_states + finish_exact_reuse()
                self._online_prefilling = still_prefilling
                self._online_step = step + 1
                return events

            n = len(batch_rows)
            for i in range(n):
                last_wp = batch_write_pos[i][-1] if batch_write_pos[i] else 0
                while len(batch_input_ids[i]) < max_q:
                    batch_input_ids[i].append(0)
                while len(batch_write_pos[i]) < max_q:
                    batch_write_pos[i].append(last_wp)

            input_ids = torch.tensor(batch_input_ids, device=self.device, dtype=torch.long)
            q_lens_t = torch.tensor(batch_q_lens, device=self.device, dtype=torch.long)
            write_positions = torch.tensor(batch_write_pos, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(batch_logit_pos, device=self.device, dtype=torch.long)
            seq_lens_t = torch.tensor(batch_seq_lens, device=self.device, dtype=torch.long)
            row_indices = torch.tensor(batch_rows, device=self.device, dtype=torch.long)

            logits = self.model.forward_step_flashinfer(
                input_ids, cache,
                seq_lens=seq_lens_t, q_lens=q_lens_t,
                write_positions=write_positions,
                logit_positions=logit_positions,
                row_indices=row_indices,
            )
            self._record_model_call("unified", n, tokens=int(q_lens_t.sum().item()))
            next_tokens_cpu = self._sample_logits_for_states(
                logits[:, -1, :],
                batch_states,
            ).detach().cpu().tolist()

            next_active: list[_ActiveRequest] = []
            for i, state in enumerate(batch_states):
                tok = int(next_tokens_cpu[i])
                if batch_is_decode[i]:
                    state.tokens.append(tok)
                    state.generated += 1
                    state.last_token = tok
                    state.seq_len += 1
                    self._remember_row_seq_len(state.row, state.seq_len)
                    finished = self._should_finish_after_decode(state)
                    self._record_token_event(events, state, tok, step, finished=finished)
                    if finished:
                        self._finish_and_release(state, step)
                    else:
                        next_active.append(state)
                else:
                    q_len = batch_q_lens[i]
                    new_seq_len = batch_seq_lens[i] + q_len
                    self._set_cache_row_seq_len(state.row, new_seq_len)
                    self._remember_row_seq_len(state.row, new_seq_len)
                    state.seq_len = new_seq_len
                    if state._prefill_suffix is None or len(state._prefill_suffix) == 0:
                        state.tokens.append(tok)
                        state.generated = 1
                        state.last_token = tok
                        state._prefill_suffix = None
                        self._store_reusable_prefix(
                            state.request.request_id, state.request.prompt,
                            state.row, logits[i:i+1],
                            allow_pinned=self._allow_pinned_full_prompt_store(state.request),
                        )
                        finished = self._should_finish_after_decode(state)
                        self._record_token_event(events, state, tok, step, finished=finished)
                        if finished:
                            self._finish_and_release(state, step)
                        else:
                            next_active.append(state)
                    else:
                        still_prefilling.append(state)

            next_active.extend(finish_exact_reuse())
            self._online_active = next_active
            self._online_prefilling = still_prefilling
        else:
            decoded = (
                self._decode_ragged_batch(decode_states, step, events=events)
                if self._can_decode_ragged(decode_states)
                else (
                    self._decode_batch(decode_states, step, events=events)
                    if len(decode_states) > 1
                    else ([self._decode_one(decode_states[0], step, events=events)] if decode_states else [])
                )
            )
            next_active = []
            for item, state in zip(decoded, decode_states):
                if isinstance(item, ServingResult):
                    pass
                else:
                    next_active.append(item)
            next_active.extend(finish_exact_reuse())
            self._online_active = next_active

        self._online_step = step + 1
        next_arrival_step = waiting.next_arrival_step()
        idle = not self._online_active and not self._online_prefilling
        if next_arrival_step is not None and idle and next_arrival_step > self._online_step:
            self._online_step = next_arrival_step
        return events


    def _admit_to_prefilling(
        self,
        admitted: list[tuple[int, ServingRequest]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        # Create a 'prefilling' state per admitted request. A newly admitted wave
        # may first materialize one shared common prefix; suffix prefill still
        # advances incrementally, and prefix KV copies are folded into the first
        # suffix chunk.
        self._prepare_chunked_common_prefix_for_admission(admitted)
        exact_reuse_group: list[tuple[int, ServingRequest, int, _ReusablePrefix]] = []
        for original_index, request in admitted:
            if request.max_new_tokens == 0:
                continue
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            prefix_hit = match.depth if (reusable is not None and match.depth > 0) else 0
            source_row = reusable.row if (reusable is not None and prefix_hit > 0) else -1
            if reusable is not None and prefix_hit >= len(request.prompt) and reusable.logits is not None:
                exact_reuse_group.append((original_index, request, prefix_hit, reusable))
                continue
            if reusable is not None and prefix_hit >= len(request.prompt):
                reusable = None
                prefix_hit = 0
            row = self._acquire_active_row()
            if prefix_hit > 0:
                self._record_prefix_reuse(prefix_hit, reusable)
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=list(request.prompt),
                generated=0,
                row=row,
                last_token=request.prompt[-1],
                seq_len=prefix_hit,
                prefix_hit_tokens=prefix_hit,
                started_step=step,
                phase="prefilling",
                prompt_cursor=prefix_hit,
                prefix_source_row=source_row,
            )
            self._online_prefilling.append(state)
        return self._prefill_exact_prefix_batch(
            exact_reuse_group,
            step,
            events=events,
        )

    def _prepare_chunked_common_prefix_for_admission(
        self,
        admitted: list[tuple[int, ServingRequest]],
    ) -> None:
        if len(admitted) <= 1 or self.prefix_cache_capacity <= 0:
            return
        requests = [request for _index, request in admitted if request.max_new_tokens > 0]
        if len(requests) <= 1:
            return
        prefix_tokens = _common_prefix_token_count([request.prompt for request in requests])
        if prefix_tokens < 16 or not all(len(request.prompt) > prefix_tokens for request in requests):
            return
        prefix_tuple = tuple(requests[0].prompt[:prefix_tokens])
        common_route = ("common_prefix", prefix_tuple)
        if common_route in self.reusable_prefixes:
            if self.pin_shared_prefix:
                self._pinned_prefix_routes.add(common_route)
            return
        prefix_row = self._acquire_prefix_row()
        if prefix_row is None:
            return
        prefix_row_adopted = False
        try:
            prefix_ids = torch.tensor(
                [prefix_tuple],
                device=self.device,
                dtype=torch.long,
            )
            cache_view = self._cache_view([prefix_row])
            if not self._prefill_cache_only(prefix_ids, rows=[prefix_row]):
                prefix_logits, _ = self._prefill_logits(prefix_ids, cache=cache_view)
                del prefix_logits
            self._refresh_row_seq_len_from_cache(prefix_row, prefix_tokens)
            self._record_model_call("prefill", 1, tokens=prefix_ids.numel())
            self._record_shape_count(self.stats.prefill_shape_counts, f"common_prefix:b1:t{prefix_tokens}")
            self.stats.prefill_common_prefix_batches += 1
            prefix_row_adopted = self._store_reusable_prefix_tokens_in_row(
                common_route,
                "__common_prefix__",
                prefix_tuple,
                prefix_row,
                None,
                store_logits=False,
            )
            if self.pin_shared_prefix and common_route in self.reusable_prefixes:
                self._pinned_prefix_routes.add(common_route)
        finally:
            if not prefix_row_adopted:
                self._release_prefix_row(prefix_row)

    def _advance_prefilling(self, step: int, events: list[ServingTokenEvent] | None) -> list[_ActiveRequest]:
        chunk = int(self.prefill_chunk_size or 0)
        if chunk <= 0 or not self._online_prefilling:
            return []
        # Group by (prefix length, cursor, prefix source) so every row in a group
        # shares one absolute start -> the flash context_len path applies, and the
        # first chunk of a group folds the shared-prefix copy from one source row.
        groups: dict[tuple[int, int, int], list[_ActiveRequest]] = defaultdict(list)
        for state in self._online_prefilling:
            groups[(state.prefix_hit_tokens, state.prompt_cursor, state.prefix_source_row)].append(state)
        newly_decoding: list[_ActiveRequest] = []
        still_prefilling: list[_ActiveRequest] = []
        for (prefix_hit, cursor, source_row), states in groups.items():
            finished, pending = self._prefill_chunk_group(states, cursor, source_row, chunk, step, events)
            newly_decoding.extend(finished)
            still_prefilling.extend(pending)
        self._online_prefilling = still_prefilling
        return newly_decoding

    def _acquire_chunk_prefill_padding_rows(
        self,
        count: int,
    ) -> tuple[list[int], list[int]]:
        if count <= 0:
            return [], []
        active_rows: list[int] = []
        prefix_rows: list[int] = []
        for _ in range(count):
            row = self._acquire_free_prefix_row_or_none()
            if row is not None:
                prefix_rows.append(row)
                continue
            row = self._acquire_active_row_or_none(clear_cache=False)
            if row is not None:
                active_rows.append(row)
                continue
            break
        if len(active_rows) + len(prefix_rows) == count:
            return active_rows, prefix_rows
        for row in active_rows:
            self._release_active_row(row)
        for row in prefix_rows:
            self._release_chunk_prefill_padding_prefix_row(row)
        return [], []

    def _release_chunk_prefill_padding_prefix_row(self, row: int) -> None:
        self._remember_row_seq_len(row, 0)
        if row not in self._free_prefix_rows:
            self._free_prefix_rows.append(row)
            self._free_prefix_rows.sort()

    def _prefill_chunk_group(
        self,
        states: list[_ActiveRequest],
        cursor: int,
        source_row: int,
        chunk: int,
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> tuple[list[_ActiveRequest], list[_ActiveRequest]]:
        chunk_lens = [min(chunk, len(s.request.prompt) - cursor) for s in states]
        chunk_bucket = self._suffix_bucket(max(chunk_lens))
        cache_max_seq = self._cache_max_seq_len()
        if cache_max_seq is not None:
            chunk_bucket = min(chunk_bucket, max(1, cache_max_seq - cursor))
        context_len = self._dynamic_prefix_prefill_context_len(
            cursor,
            chunk_bucket,
            max_seq_len=cache_max_seq,
        )
        chunks = [s.request.prompt[cursor : cursor + n] for s, n in zip(states, chunk_lens)]
        padded = [[*c, *([0] * (chunk_bucket - len(c)))] for c in chunks]
        rows = [s.row for s in states]
        batch_bucket = self._prefill_batch_bucket(len(states))
        pad_active_rows: list[int] = []
        pad_prefix_rows: list[int] = []
        if batch_bucket > len(states):
            pad_active_rows, pad_prefix_rows = self._acquire_chunk_prefill_padding_rows(
                batch_bucket - len(states)
            )
            if len(pad_active_rows) + len(pad_prefix_rows) != batch_bucket - len(states):
                batch_bucket = len(states)
        pad_count = max(0, batch_bucket - len(states))
        if pad_count > 0:
            dummy_suffix = padded[0] if padded else [0] * chunk_bucket
            padded.extend([list(dummy_suffix) for _ in range(pad_count)])
        graph_rows = rows + pad_active_rows + pad_prefix_rows
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        row_indices = torch.tensor(graph_rows, device=self.device, dtype=torch.long)
        required = max(graph_rows + ([source_row] if source_row >= 0 else [0])) + 1
        seq_lens_list = [0] * required
        for row in graph_rows:
            seq_lens_list[row] = cursor
        seq_lens = torch.tensor(seq_lens_list, device=self.device, dtype=torch.long)
        graph_chunk_lens = [*chunk_lens, *([chunk_lens[0]] * pad_count)]
        logit_positions = torch.tensor(
            [max(0, n - 1) for n in graph_chunk_lens],
            device=self.device,
            dtype=torch.long,
        )
        # Fold the prefix copy only on the FIRST chunk of a reused prefix.
        src_prefix_row = None
        if source_row >= 0 and cursor == states[0].prefix_hit_tokens and cursor > 0:
            src_prefix_row = torch.tensor([source_row], device=self.device, dtype=torch.long)
        chunk_completes_prompt = [
            cursor + chunk_len >= len(state.request.prompt)
            for state, chunk_len in zip(states, chunk_lens)
        ]
        shape_key = (
            "chunk_graph:"
            f"b{batch_bucket}:s{chunk_bucket}:p{cursor}:"
            f"logits{int(any(chunk_completes_prompt))}"
        )
        self._record_shape_count(self.stats.prefill_shape_counts, shape_key)
        try:
            if not any(chunk_completes_prompt):
                cache_filled = self._try_ragged_prefill_cache(
                    input_ids,
                    seq_lens,
                    row_indices,
                    context_len,
                    src_prefix_row,
                    profile_shape_key=shape_key,
                )
                if not cache_filled:
                    logits = self._ragged_prefill_logits_eager(
                        input_ids,
                        seq_lens,
                        row_indices,
                        logit_positions,
                        context_len,
                        src_prefix_row,
                    )
                    if logits is None:
                        raise RuntimeError("chunked prefill requires ragged prefill logits or cache support")
                self._record_model_call("prefill", len(states), tokens=sum(chunk_lens))
                self.stats.prefill_prefix_reuse_batches += 1
                pending: list[_ActiveRequest] = []
                for index, state in enumerate(states):
                    new_cursor = cursor + chunk_lens[index]
                    state.prompt_cursor = new_cursor
                    self._set_cache_row_seq_len(state.row, new_cursor)
                    state.seq_len = new_cursor
                    pending.append(state)
                return [], pending
            logits = self._try_ragged_prefill_logits(
                input_ids,
                seq_lens,
                row_indices,
                logit_positions,
                context_len,
                src_prefix_row,
                profile_shape_key=shape_key,
            )
            if logits is None:
                logits = self._ragged_prefill_logits_eager(
                    input_ids, seq_lens, row_indices, logit_positions, context_len, src_prefix_row
                )
            logits = logits[: len(states)]
            self._record_model_call("prefill", len(states), tokens=sum(chunk_lens))
            self.stats.prefill_prefix_reuse_batches += 1
            next_tokens = self._sample_logits_for_states(
                logits[:, -1, :],
                states,
            ).detach().cpu().tolist()
            finished: list[_ActiveRequest] = []
            pending: list[_ActiveRequest] = []
            for index, state in enumerate(states):
                new_cursor = cursor + chunk_lens[index]
                state.prompt_cursor = new_cursor
                self._set_cache_row_seq_len(state.row, new_cursor)
                state.seq_len = new_cursor
                if new_cursor >= len(state.request.prompt):
                    next_token = int(next_tokens[index])
                    state.tokens.append(next_token)
                    state.generated = 1
                    state.last_token = next_token
                    state.phase = "decoding"
                    request_finished = self._should_finish_before_decode(state)
                    self._record_token_event(events, state, next_token, step, finished=request_finished)
                    if request_finished:
                        self._finish_and_release(state, step)
                    else:
                        finished.append(state)
                else:
                    pending.append(state)
            return finished, pending
        finally:
            for row in pad_active_rows:
                self._release_active_row(row)
            for row in pad_prefix_rows:
                self._release_chunk_prefill_padding_prefix_row(row)

    def _reset_run_state(self, requests: list[ServingRequest]) -> None:
        max_seq_len = max((len(request.prompt) + request.max_new_tokens for request in requests), default=1)
        self._reset_capacity(max_seq_len=max(1, max_seq_len), queued_requests=len(requests))

    def _reset_capacity(
        self,
        *,
        max_seq_len: int,
        queued_requests: int,
        external_cache: object | None = None,
    ) -> None:
        self.stats = ServingStats()
        self.prefix_cache = PrefixCacheIndex()
        self.reusable_prefixes = {}
        self._full_prompt_reuse_candidate_cache = PrefixCacheIndex()
        self._full_prompt_reuse_candidate_order = []
        self._full_prompt_reuse_candidate_session_id = (
            _next_persistent_full_prompt_reuse_candidate_session()
        )
        self._prefix_order = []
        self._pinned_prefix_routes = set()
        self._cache_views = {}
        self._reported_static_graph_miss = False
        self._packed_prefill_fixed_capacity_counts = {}
        self._packed_prefill_fixed_capacity_seen = {}
        self._packed_prefill_fixed_capacity_stable_seen = {}
        self._packed_prefill_fixed_capacity_graph_none_keys = set()
        total_rows = self.max_active_requests + self.prefix_cache_capacity
        if external_cache is not None:
            self._cache = external_cache
            _enable_runtime_cache_capture_sync(self._cache)
        else:
            self._cache = self._allocate_cache(max(1, total_rows), max_seq_len)
        if not hasattr(self._cache, "for_rows"):
            raise ValueError("model cache must support row views for persistent serving")
        self._row_seq_lens = [0 for _ in range(total_rows)]
        self._row_cached_prefixes = [None for _ in range(total_rows)]
        self._gpu_seq_lens = None
        self._decode_many_gpu_state_signature = None
        self._device_index_tensors = {}
        self._free_active_rows = list(reversed(range(self.max_active_requests)))
        self._free_prefix_rows = list(reversed(range(self.max_active_requests, total_rows)))
        self.stats.persistent_cache_rows = total_rows
        self.stats.queued_requests = queued_requests
        self._online_waiting = None
        self._online_active = []
        self._online_prefilling = []
        self._online_step = 0
        self._online_next_index = 0
        self._pending_prefill_graph_events = []
        self._pending_decode_ragged_model_events = []

    def _admit_ready_requests(
        self,
        waiting: ServingQueue,
        step: int,
        active_count: int,
    ) -> list[tuple[int, ServingRequest]]:
        capacity = self.max_active_requests - active_count
        default_min_free_rows = (
            1 if self.admit_min_free_rows is None else int(self.admit_min_free_rows)
        )
        min_free_rows = env_int(
            "TORCHINFERNO_CONTINUOUS_ADMIT_MIN_FREE_ROWS",
            default_min_free_rows,
            minimum=1,
        )
        if active_count > 0 and capacity < min(min_free_rows, self.max_active_requests):
            return []
        # Always cap NEW admissions per step at per_step_cap. This decouples the
        # prefill batch size (<= per_step_cap, where the prefill CUDA graphs live)
        # from the decode batch size (active rows can grow to max_active across
        # several steps). A larger decode batch lifts memory-bound decode
        # throughput without forcing a giant single-step prefill.
        per_step_cap = self.admit_per_step_cap
        if per_step_cap is None:
            per_step_cap = env_int("TORCHINFERNO_CONTINUOUS_ADMIT_PER_STEP_CAP", 48, minimum=0)
        if per_step_cap > 0:
            capacity = min(capacity, per_step_cap)
        default_min_ready_requests = self.admit_min_ready_requests
        if default_min_ready_requests is None:
            default_min_ready_requests = 1
        min_ready_requests = env_int(
            "TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS",
            int(default_min_ready_requests),
            minimum=1,
        )
        if active_count > 0 and min_ready_requests > 1:
            if waiting.ready_count(step=step) < min_ready_requests:
                return []
        return waiting.pop_admissible(
            step=step,
            capacity=capacity,
            token_budget=self.prefill_token_budget,
            token_cost=self._prefill_token_cost,
            priority_key=lambda item: self._admission_priority(
                item,
                active_count=active_count,
            ),
        )

    def _prefill_token_cost(self, request: ServingRequest) -> int:
        prefix_hit_tokens = self._reusable_prefix_hit_tokens(request.prompt)
        return max(1, len(request.prompt) - prefix_hit_tokens)

    def _admission_priority(
        self,
        item: _QueuedRequest,
        *,
        active_count: int = 0,
    ) -> tuple[object, ...]:
        prefix_hit_tokens = self._reusable_prefix_hit_tokens(item.request.prompt)
        prefix_priority = -prefix_hit_tokens if self._admit_prefix_hit_priority_enabled() else 0
        if self._admit_prefill_cost_priority_enabled(active_count=active_count):
            prefill_cost = max(1, len(item.request.prompt) - prefix_hit_tokens)
            return (
                prefix_priority,
                prefill_cost,
                item.request.max_new_tokens,
                item.request.arrival_step,
                item.sequence,
            )
        return (prefix_priority, item.request.arrival_step, item.sequence)

    def _admit_prefix_hit_priority_enabled(self) -> bool:
        return env_flag("TORCHINFERNO_CONTINUOUS_ADMIT_PREFIX_HIT_PRIORITY", True)

    def _admit_prefill_cost_priority_enabled(self, *, active_count: int = 0) -> bool:
        if "TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY" in os.environ:
            return env_flag("TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY", False)
        if self.temperature > 0.0 or self.max_generation_tokens is None:
            return False
        short_max = env_int(
            "TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY_GREEDY_SHORT_MAX_TOKENS",
            128,
            minimum=1,
        )
        if 0 < int(self.max_generation_tokens) <= short_max:
            return True
        if active_count <= 0:
            return False
        large_refill_env = "TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY_GREEDY_LARGE_REFILL"
        if large_refill_env in os.environ:
            return env_flag(large_refill_env, False)
        return False

    def _prefill_many(
        self,
        indexed_requests: list[tuple[int, ServingRequest]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        timing_start_s = time.perf_counter() if self.profile_timings else 0.0
        self.stats.prefill_admitted_requests += len(indexed_requests)
        indexed_results: list[tuple[int, ServingResult]] = []
        active: list[_ActiveRequest] = []
        batchable: dict[int, list[tuple[int, ServingRequest, int]]] = defaultdict(list)
        prefix_batchable: dict[tuple[int, int], list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = defaultdict(list)
        # graph_prefill pads suffixes to a common length and buckets the batch,
        # so reuse requests must be grouped by prefix alone (suffix key -1)
        # rather than split per suffix length -- otherwise each suffix length
        # reaches the graph path as its own tiny batch and never amortizes.
        pad_prefix_suffixes = env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False) or self.graph_prefill

        for original_index, request in indexed_requests:
            if not request.prompt:
                raise ValueError("request prompt must contain at least one token")
            match, entry = self.prefix_cache.lookup(request.prompt)
            reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
            match, entry, reusable = self._maybe_demote_mixed_prefix_long_suffix(
                request.prompt,
                match,
                entry,
                reusable,
                pad_prefix_suffixes=pad_prefix_suffixes,
            )
            match, entry, reusable = self._maybe_demote_mixed_prefix_extra_tokens(
                request.prompt,
                match,
                entry,
                reusable,
                pad_prefix_suffixes=pad_prefix_suffixes,
            )
            reusable_prefix_tokens = match.depth if reusable is not None else 0
            if request.max_new_tokens == 0:
                indexed_results.append(
                    (
                        original_index,
                        ServingResult(
                            request.request_id,
                            request.prompt,
                            0,
                            request.arrival_step,
                            step,
                            step,
                        ),
                    )
                )
                continue
            if (
                reusable is not None
                and reusable_prefix_tokens >= len(request.prompt)
                and reusable.logits is None
            ):
                reusable = None
                reusable_prefix_tokens = 0
            self._record_full_prompt_reuse_candidate_lookup(
                request.prompt,
                reusable_prefix_tokens if reusable is not None else 0,
            )
            if reusable is not None and reusable_prefix_tokens > 0:
                suffix_len = len(request.prompt) - reusable_prefix_tokens
                batch_suffix_len = -1 if pad_prefix_suffixes else suffix_len
                if pad_prefix_suffixes and self._mixed_prefix_prefill_enabled():
                    suffix_bucket = self._suffix_bucket(suffix_len)
                    cache_max_seq = self._cache_max_seq_len()
                    if cache_max_seq is not None:
                        suffix_bucket = min(
                            suffix_bucket,
                            max(1, cache_max_seq - reusable_prefix_tokens),
                        )
                    if (
                        not self._mixed_prefix_dynamic_context_enabled()
                        or suffix_bucket <= self._mixed_prefix_dynamic_context_max_suffix()
                    ):
                        reusable_prefix_tokens = -1
                prefix_batchable[(reusable_prefix_tokens, batch_suffix_len)].append(
                    (original_index, request, match.depth, reusable)
                )
            else:
                batchable[len(request.prompt)].append((original_index, request, 0))

        plain_group = [item for group in batchable.values() for item in group]

        # FlashInfer-native prefix reuse for cached-prefix hits; on failure the
        # group falls back into the full-prompt path below (returns None).
        # OFF by default. The reuse LOGIC is verified correct in isolation --
        # scripts/debug_reuse_engine.py reproduces the full engine path (prefill
        # graphs + online stepping + reuse) single-GPU and PASSES. Enabling it on
        # the 8-rank server hangs/CUDA-asserts. Narrowed via env-gated diagnostics
        # (TORCHINFERNO_REUSE_DEBUG): it is reuse-vs-TP, NOT the graphs. Ruled out:
        # reuse-OFF + graphs-OFF + a single large request works fine on 8 ranks
        # (so the _prefill_one -> _prefill_logits FI-eager fallback is sound). The
        # hang appears ONLY with the reuse config (FI_REUSE=1 + pin_shared_prefix
        # =False + prefix_rows>2) and resists single-GPU repro -- the standalone
        # engine harness (scripts/debug_reuse_engine.py) PASSES. This is a subtle
        # multi-rank collective-divergence bug that needs a dedicated instrumented
        # 8-GPU session, not incremental loop runs. Enable via the env flag +
        # pin_shared_prefix=False once that divergence is found.
        #
        # IMPORTANT value caveat (8-GPU REUSE_DEBUG trace): with persistent=False
        # every request burst starts a fresh online session and start_online ->
        # _reset_capacity WIPES reusable_prefixes/prefix_cache. So cross-burst
        # reuse never triggers (each _prefill_many logs cached_prefixes=0 at
        # step=0) -- e.g. multi_turn's per-turn requests are separate bursts and
        # get no reuse. Only WITHIN-session reuse fires (and that is the case that
        # hangs on TP). Making reuse actually pay off therefore ALSO needs a
        # persistent engine whose prefix cache survives across bursts -- a larger
        # change than the reuse path itself.
        _reuse_dbg = env_flag("TORCHINFERNO_REUSE_DEBUG", False)
        _reuse_handled = 0
        if env_flag("TORCHINFERNO_CONTINUOUS_FI_REUSE", False) and prefix_batchable:
            unhandled: dict[tuple[int, int], list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = {}
            for key, group in prefix_batchable.items():
                reuse_active = self._prefill_flashinfer_reuse(group, step, events=events)
                if reuse_active is not None:
                    active.extend(reuse_active)
                    _reuse_handled += len(reuse_active)
                else:
                    unhandled[key] = group
            prefix_batchable = unhandled
        if _reuse_dbg:
            import sys as _rdbg
            _rk = getattr(self.model, "rank", 0)
            _n_pref = sum(len(g) for g in prefix_batchable.values())
            print(
                f"[REUSE_DBG] rank={_rk} step={step} reuse_handled={_reuse_handled} "
                f"unhandled_prefix={_n_pref} plain={len(plain_group)} "
                f"cached_prefixes={len(self.reusable_prefixes)}",
                file=_rdbg.stderr, flush=True,
            )

        # Common-prefix fast path: prefill a shared prefix ONCE then per-request
        # suffixes. DISABLED by default -- live A/B on the real 70B (TP8) showed it
        # is a regression in EVERY regime, including the identical-prompt
        # self_consistency case it was built for. Cause: it routes the burst through
        # the EAGER _prefill_logits path (launch-overhead bound, ~245ms/call on TP8
        # because per-layer allreduces are not graph-amortized), bypassing the
        # graph-backed _try_flashinfer_prefill below (try_prefill_flashinfer_graph,
        # the warmup-captured _fi_prefill_graphs). Worse, the online batcher admits
        # in waves (initial_batch_size=1), so each wave pays its own ~245ms eager
        # prefix prefill. Measured TTFT (identical / distinct), ON vs OFF: N=8
        # 603/737 -> 122/124; N=64 1549/2378 -> 999/949 (3-7x at low N, ~2x at high
        # N, ~2x throughput). The graph-FI path handles identical and distinct
        # equally fast, so this path only wins if eager prefill is ever fixed.
        if (
            env_flag("TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_PREFILL", False)
            and len(plain_group) > 1
            and self.prefix_cache_capacity > 0
        ):
            cp_tokens = _common_prefix_token_count([req.prompt for _i, req, _h in plain_group])
            if cp_tokens >= 16:
                shared_active = self._prefill_common_prefix_batch(
                    [(idx, req, 0) for idx, req, _h in plain_group], step, events=events
                )
                if shared_active is not None:
                    active.extend(shared_active)
                    plain_group = []

        all_fi_requests: list[tuple[int, ServingRequest, int, _ReusablePrefix | None]] = []
        # FlashInfer prefill defaults OFF. MEASURED 2026-06-10 on the real 70B TP8
        # local full bench at 64-conc: FI-prefill ON regressed few_shot ttft
        # 216->759ms (3.5x), tpot 73->247ms, tput 4.1->1.1 vs identical config with
        # FI-prefill OFF. The bench's prefills are small, so FlashInfer's varlen
        # advantage does not apply and its per-wave plan/launch overhead dominates;
        # TI's graphed ragged prefill is already GEMM-bound and faster here. The
        # gate was default-ON, a footgun: it auto-enables whenever flashinfer is
        # importable, so installing flashinfer (e.g. for paged decode) would
        # silently regress serving 3.5x. Set the flag to 0 to opt back in.
        paged_kv_cache = self._cache_uses_paged_kv()
        if paged_kv_cache or not env_flag("TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE", True):
            # Do not let full-prompt FI prefill steal cached-prefix groups. The
            # 20260704 tree packed-FI probe saved only 1.4K padded tokens but
            # disabled all common-prefix reuse, pushing prefill wall to 17s.
            # Prefix-hit FI handling stays behind the separate FI_REUSE gate on
            # dense caches. With a paged FlashInfer cache, dense fallback is
            # unsafe, so route prefix-hit requests as full prompts instead of
            # falling through to _prefill_logits on paged storage.
            if env_flag("TORCHINFERNO_CONTINUOUS_FI_REUSE", False) and not paged_kv_cache:
                for group in prefix_batchable.values():
                    for idx, req, hit, reusable in group:
                        all_fi_requests.append((idx, req, hit, reusable))
            elif paged_kv_cache:
                for group in prefix_batchable.values():
                    for idx, req, _hit, _reusable in group:
                        all_fi_requests.append((idx, req, 0, None))
            for idx, req, hit in plain_group:
                all_fi_requests.append((idx, req, hit, None))

        if all_fi_requests:
            _fi_debug = env_flag("TORCHINFERNO_FI_PREFILL_PROFILE", False)
            _fi_t0 = time.perf_counter() if _fi_debug else 0.0
            # For a single request, eager FlashInfer is launch-overhead bound
            # (~245ms regardless of prompt length), worse than the SDPA single
            # path. So batch=1 uses the prefill CUDA graph ONLY (a captured
            # batch=1 graph replays in ~25ms even padded to the q bucket, since
            # batch=1 compute is tiny); a graph miss falls through to _prefill_one.
            graph_only = len(all_fi_requests) == 1 and not paged_kv_cache
            fi_active = self._try_flashinfer_prefill(
                all_fi_requests, step, events=events, graph_only=graph_only
            )
            if fi_active is not None:
                if _fi_debug:
                    import sys as _fpm
                    print(
                        f"[FI_PREFILL] OK batch={len(all_fi_requests)} active={len(fi_active)} "
                        f"time={(time.perf_counter()-_fi_t0)*1000:.1f}ms",
                        file=_fpm.stderr, flush=True,
                    )
                active.extend(fi_active)
                if self.profile_timings:
                    self.stats.prefill_wall_ms += (time.perf_counter() - timing_start_s) * 1000.0
                return indexed_results, active
            if paged_kv_cache:
                raise RuntimeError(
                    "paged-cache prefill requires FlashInfer prefill; "
                    "FlashInfer prefill failed before dense fallback"
                )
            elif _fi_debug:
                import sys as _fpm
                print(
                    f"[FI_PREFILL] FALLBACK batch={len(all_fi_requests)} "
                    f"prefix_batchable={len(prefix_batchable)} plain={len(plain_group)}",
                    file=_fpm.stderr, flush=True,
                )

        for group in prefix_batchable.values():
            active.extend(self._prefill_prefix_batch(group, step, events=events))

        if env_flag("TORCHINFERNO_CONTINUOUS_RAGGED_GRAPH_PREFILL", False) and plain_group and self.graph_prefill and len(plain_group) > 1:
            ragged_active = self._prefill_ragged_graph_batch(plain_group, step, events=events)
            if ragged_active is not None:
                active.extend(ragged_active)
                plain_group = []
        if plain_group:
            shared_prefix_active = self._prefill_common_prefix_batch(plain_group, step, events=events)
            if shared_prefix_active is not None:
                active.extend(shared_prefix_active)
            elif len(plain_group) > 1 and self._can_padded_batch_prefill(plain_group):
                active.extend(self._prefill_padded_batch(plain_group, step, events=events))
            else:
                for group in batchable.values():
                    if len(group) == 1:
                        original_index, request, prefix_hit_tokens = group[0]
                        active.append(
                            self._prefill_one(
                                original_index,
                                request,
                                step,
                                prefix_hit_tokens,
                                None,
                                events=events,
                            )
                        )
                    else:
                        active.extend(self._prefill_batch(group, step, events=events))
        if self.profile_timings:
            self.stats.prefill_wall_ms += (time.perf_counter() - timing_start_s) * 1000.0
        return indexed_results, active

    def _maybe_demote_mixed_prefix_long_suffix(
        self,
        prompt: tuple[int, ...],
        match: PrefixMatch,
        entry: PrefixCacheEntry | None,
        reusable: _ReusablePrefix | None,
        *,
        pad_prefix_suffixes: bool,
    ) -> tuple[PrefixMatch, PrefixCacheEntry | None, _ReusablePrefix | None]:
        if (
            entry is None
            or reusable is None
            or not pad_prefix_suffixes
            or not self._mixed_prefix_prefill_enabled()
            or not self._mixed_prefix_dynamic_context_enabled()
            or not self._mixed_prefix_long_suffix_common_fallback_enabled()
        ):
            return match, entry, reusable
        if self._prefix_reuse_route_kind(reusable.route_id) == "common_prefix":
            return match, entry, reusable

        suffix_len = len(prompt) - match.depth
        if suffix_len <= 0:
            return match, entry, reusable
        suffix_bucket = self._suffix_bucket(suffix_len)
        cache_max_seq = self._cache_max_seq_len()
        if cache_max_seq is not None:
            suffix_bucket = min(suffix_bucket, max(1, cache_max_seq - match.depth))
        mixed_max_suffix = self._mixed_prefix_dynamic_context_max_suffix()
        if suffix_bucket <= mixed_max_suffix:
            return match, entry, reusable

        fallback_match, fallback_entry, fallback_reusable = self._lookup_live_common_prefix(prompt)
        if (
            fallback_reusable is None
            or fallback_match.depth <= 0
            or fallback_match.depth >= match.depth
        ):
            return match, entry, reusable
        return fallback_match, fallback_entry, fallback_reusable

    def _maybe_demote_mixed_prefix_extra_tokens(
        self,
        prompt: tuple[int, ...],
        match: PrefixMatch,
        entry: PrefixCacheEntry | None,
        reusable: _ReusablePrefix | None,
        *,
        pad_prefix_suffixes: bool,
    ) -> tuple[PrefixMatch, PrefixCacheEntry | None, _ReusablePrefix | None]:
        if (
            entry is None
            or reusable is None
            or not pad_prefix_suffixes
            or not self._mixed_prefix_prefill_enabled()
        ):
            return match, entry, reusable
        if self._prefix_reuse_route_kind(reusable.route_id) == "common_prefix":
            return match, entry, reusable

        min_extra = self._mixed_prefix_min_extra_tokens()
        max_extra = self._mixed_prefix_max_extra_tokens()
        if min_extra <= 0 and max_extra is None:
            return match, entry, reusable

        fallback_match, fallback_entry, fallback_reusable = self._lookup_live_common_prefix(prompt)
        if (
            fallback_reusable is None
            or fallback_match.depth <= 0
            or fallback_match.depth >= match.depth
        ):
            return match, entry, reusable

        extra_tokens = int(match.depth) - int(fallback_match.depth)
        if extra_tokens < min_extra or (max_extra is not None and extra_tokens > max_extra):
            return fallback_match, fallback_entry, fallback_reusable
        return match, entry, reusable

    def _lookup_live_common_prefix(
        self,
        prompt: tuple[int, ...],
    ) -> tuple[PrefixMatch, PrefixCacheEntry | None, _ReusablePrefix | None]:
        def _live_common_prefix(candidate: PrefixCacheEntry) -> bool:
            return (
                isinstance(candidate.route_id, tuple)
                and candidate.route_id[:1] == ("common_prefix",)
                and candidate.route_id in self.reusable_prefixes
            )

        match, entry = self.prefix_cache.lookup_filtered(prompt, _live_common_prefix)
        reusable = self.reusable_prefixes.get(entry.route_id) if entry is not None else None
        return match, entry, reusable

    def _prefill_common_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if len(group) <= 1 or self.prefix_cache_capacity <= 0:
            return None
        prefix_tokens = _common_prefix_token_count([request.prompt for _index, request, _hit in group])
        min_prefix_tokens = 16
        if prefix_tokens < min_prefix_tokens:
            return None
        prefix_row = self._acquire_prefix_row()
        if prefix_row is None:
            return None
        prefix_row_adopted = False
        try:
            prefix_ids = torch.tensor(
                [group[0][1].prompt[:prefix_tokens]],
                device=self.device,
                dtype=torch.long,
            )
            store_common_prefix_logits = any(
                len(request.prompt) <= prefix_tokens
                for _index, request, _hit in group
            )
            prefix_logits: Tensor | None = None
            prefix_filled = (
                not store_common_prefix_logits
                and not self._cache_uses_paged_kv()
                and self._cache_supports_tensor_ragged_prefill()
                and self._prefill_cache_only(prefix_ids, rows=[prefix_row])
            )
            if not prefix_filled:
                prefix_logits, _ = self._prefill_logits(prefix_ids, cache=self._cache_view([prefix_row]))
            self._refresh_row_seq_len_from_cache(prefix_row, prefix_tokens)
            self._record_model_call("prefill", 1, tokens=prefix_ids.numel())
            self._record_shape_count(self.stats.prefill_shape_counts, f"common_prefix:b1:t{prefix_tokens}")
            self.stats.prefill_common_prefix_batches += 1
            prefix_tuple = tuple(group[0][1].prompt[:prefix_tokens])
            common_route = ("common_prefix", prefix_tuple)
            prefix_row_adopted = self._store_reusable_prefix_tokens_in_row(
                common_route,
                "__common_prefix__",
                prefix_tuple,
                prefix_row,
                prefix_logits if store_common_prefix_logits else None,
                store_logits=store_common_prefix_logits,
            )
            reusable = self.reusable_prefixes.get(common_route)
            if self.pin_shared_prefix and common_route in self.reusable_prefixes:
                self._pinned_prefix_routes.add(common_route)

            # Folding the common-prefix KV copy into the ragged suffix graph is
            # profitable for greedy streams when the shared prefix fits in the
            # startup-warmed prefix-suffix buckets, where it avoids per-request
            # eager suffix prefill without promoting large-prefix graph misses.
            max_ragged_prefix_tokens = self._common_prefix_ragged_suffix_max_prefix_tokens(group)
            if (
                self.graph_prefill
                and reusable is not None
                and max_ragged_prefix_tokens > 0
                and prefix_tokens <= max_ragged_prefix_tokens
                and all(len(request.prompt) > prefix_tokens for _index, request, _hit in group)
            ):
                graph_active = self._prefill_prefix_graph_batch(
                    [
                        (original_index, request, prefix_tokens, reusable)
                        for original_index, request, _prefix_hit_tokens in group
                    ],
                    step,
                    events=events,
                )
                if graph_active is not None:
                    return graph_active

            padded_active = self._prefill_common_prefix_padded_suffix_batch(
                group,
                prefix_row=prefix_row,
                prefix_tokens=prefix_tokens,
                step=step,
                events=events,
            )
            if padded_active is not None:
                return padded_active

            active: list[_ActiveRequest] = []
            suffix_groups: dict[int, list[tuple[int, ServingRequest, int]]] = defaultdict(list)
            for original_index, request, prefix_hit_tokens in group:
                del prefix_hit_tokens
                suffix_groups[len(request.prompt) - prefix_tokens].append((original_index, request, 0))
            for suffix_group in suffix_groups.values():
                rows = [self._acquire_active_row() for _ in suffix_group]
                self._copy_prefix_to_rows(prefix_row, rows, prefix_tokens)
                suffixes = [
                    request.prompt[prefix_tokens:]
                    for _original_index, request, _prefix_hit_tokens in suffix_group
                ]

                if not suffixes or not suffixes[0]:
                    if prefix_logits is None:
                        prefix_logits, _ = self._prefill_logits(
                            prefix_ids,
                            cache=self._cache_view([prefix_row]),
                        )
                    logits = prefix_logits.expand(len(suffix_group), -1, -1)
                else:
                    input_ids = torch.tensor(suffixes, device=self.device, dtype=torch.long)
                    logits, _ = self._prefill_logits(input_ids, cache=self._cache_view(rows))
                    self._record_model_call("prefill", len(suffix_group), tokens=input_ids.numel())
                next_tokens = self._sample_logits_for_requests(
                    logits[: len(suffix_group), -1, :],
                    [request for _original_index, request, _prefix_hit_tokens in suffix_group],
                ).detach().cpu().tolist()

                for row_index, (original_index, request, prefix_hit_tokens) in enumerate(suffix_group):
                    row = rows[row_index]
                    seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
                    self._store_reusable_prefix(
                        request.request_id,
                        request.prompt,
                        row,
                        logits[row_index : row_index + 1],
                        allow_pinned=self._allow_pinned_full_prompt_store(request),
                    )
                    next_token = int(next_tokens[row_index])
                    state = _ActiveRequest(
                        original_index=original_index,
                        request=request,
                        tokens=[*request.prompt, next_token],
                        generated=1,
                        row=row,
                        last_token=next_token,
                        seq_len=seq_len,
                        prefix_hit_tokens=prefix_hit_tokens,
                        started_step=step,
                    )
                    self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                    active.append(state)
            return active
        finally:
            if not prefix_row_adopted:
                self._release_prefix_row(prefix_row)

    def _common_prefix_ragged_suffix_max_prefix_tokens(
        self,
        group: list[tuple[int, ServingRequest, int]],
    ) -> int:
        configured = os.environ.get(
            "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_MAX_PREFIX_TOKENS"
        )
        if configured is not None:
            return env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_MAX_PREFIX_TOKENS",
                64,
                minimum=0,
            )
        if self.temperature <= 0.0 and group:
            max_new_tokens = max(request.max_new_tokens for _index, request, _hit in group)
            greedy_short_max = env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_SHORT_MAX_TOKENS",
                128,
                minimum=1,
            )
            if 0 < max_new_tokens <= greedy_short_max:
                return env_int(
                    "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_SHORT_PREFIX_TOKENS",
                    128,
                    minimum=0,
                )
            greedy_mid_min = env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_MID_MIN_TOKENS",
                greedy_short_max + 1,
                minimum=1,
            )
            greedy_mid_max = env_int(
                "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_MID_MAX_TOKENS",
                300,
                minimum=greedy_mid_min,
            )
            if greedy_mid_min <= max_new_tokens <= greedy_mid_max:
                return env_int(
                    "TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_GREEDY_MID_PREFIX_TOKENS",
                    128,
                    minimum=0,
                )
        return 64

    def _prefill_common_prefix_padded_suffix_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        *,
        prefix_row: int,
        prefix_tokens: int,
        step: int,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False):
            return None
        suffixes = [request.prompt[prefix_tokens:] for _original_index, request, _hit in group]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0:
            return None
        max_suffix_len = max(suffix_lengths)
        static_batch = self._prefill_static_batch_size(len(group))
        padded_batch_size = max(len(group), static_batch)
        padding_tokens = padded_batch_size * max_suffix_len - sum(suffix_lengths)
        max_padding_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_MAX_PADDING_TOKENS",
            4096,
            minimum=0,
        )
        if padding_tokens > max_padding_tokens:
            return None

        rows = [self._acquire_active_row() for _ in group]
        try:
            self._copy_prefix_to_rows(prefix_row, rows, prefix_tokens)
            padded_suffixes = [
                [*suffix, *([0] * (max_suffix_len - len(suffix)))]
                for suffix in suffixes
            ]
            pad_rows: list[int] = []
            if padded_batch_size > len(group):
                dummy_suffix = padded_suffixes[0] if padded_suffixes else [0] * max_suffix_len
                for _ in range(padded_batch_size - len(group)):
                    pad_row = self._acquire_active_row_or_none()
                    if pad_row is None:
                        break
                    pad_rows.append(pad_row)
                    self._copy_prefix(prefix_row, pad_row, prefix_tokens)
                    padded_suffixes.append(list(dummy_suffix))
            all_rows = rows + pad_rows
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(
                [length - 1 for length in suffix_lengths]
                + [max_suffix_len - 1] * len(pad_rows),
                device=self.device,
                dtype=torch.long,
            )
            logits = self._forward_selected_logits(
                input_ids,
                cache=self._cache_view(all_rows),
                logit_positions=logit_positions,
            )
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            if logits is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
            self.stats.prefill_padded_suffix_batches += 1
            next_tokens = self._sample_logits_for_requests(
                logits[: len(group), -1, :],
                [request for _original_index, request, _prefix_hit_tokens in group],
            ).detach().cpu().tolist()

            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
                row = rows[row_index]
                self._set_cache_row_seq_len(row, len(request.prompt))
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    logits[row_index : row_index + 1],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
                next_token = int(next_tokens[row_index])
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=1,
                    row=row,
                    last_token=next_token,
                    seq_len=self._cache_row_seq_len(row, len(request.prompt)),
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
                self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                active.append(state)
            return active
        except Exception:
            for row in rows:
                self._release_active_row(row)
            raise

    def _prefill_batch_bucket(self, count: int) -> int:
        configured = os.environ.get("TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS")
        if configured is not None:
            configured_buckets = tuple(
                bucket
                for bucket in _parse_positive_int_csv(configured)
                if bucket <= self.max_active_requests
            )
            configured_bucket = _bucket_from_values(count, configured_buckets)
            if configured_bucket is not None:
                return configured_bucket
        default_bucket = _bucket_from_values(
            count,
            _default_prefix_prefill_batch_buckets(
                self.temperature,
                self.max_generation_tokens,
                self.max_active_requests,
            ),
        )
        if default_bucket is not None:
            return default_bucket
        # Pad the prefill batch to a power of two so the model's prefill graph
        # key -- (batch, suffix_bucket, prefix_len) -- repeats across batches and
        # replays instead of recapturing on every differently-sized batch. Cap
        # at the active-row capacity so dummy padding rows never exceed the cache.
        if count <= 1:
            return 1
        bucket = 1 << (count - 1).bit_length()
        return min(bucket, self.max_active_requests)

    def _prefix_prefill_capture_on_miss(self, batch_bucket: int) -> bool:
        env_name = "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_ON_MISS"
        if env_name in os.environ:
            return env_flag(env_name, True)
        if self._greedy_large_mixed_prefix_reuse_enabled():
            return False
        max_tokens = self.max_generation_tokens
        if self.temperature <= 0.0 and max_tokens is not None:
            greedy_short_max_tokens = env_int(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_GREEDY_SHORT_MAX_TOKENS",
                128,
                minimum=1,
            )
            if 0 < int(max_tokens) <= greedy_short_max_tokens:
                max_capture_batch = env_int(
                    "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_GREEDY_SHORT_MAX_BATCH",
                    32,
                    minimum=0,
                )
                return int(batch_bucket) <= max_capture_batch
        return True

    def _prefix_prefill_token_graph_capture_on_miss(self) -> bool:
        return env_flag(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_TOKEN_GRAPH_CAPTURE_ON_MISS",
            False,
        )

    def _prefix_prefill_token_graph_enabled(self) -> bool:
        warmup_env = "TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_TOKEN_SUFFIX_PREFILL"
        warmup_default_enabled = False
        max_tokens = self.max_generation_tokens
        if self.temperature <= 0.0 and max_tokens is not None:
            greedy_short_max_tokens = env_int(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT_MAX_TOKENS",
                128,
                minimum=1,
            )
            warmup_default_enabled = 0 < int(max_tokens) <= greedy_short_max_tokens
        return bool(
            self._prefix_prefill_token_graph_capture_on_miss()
            or env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_TOKEN_GRAPH", False)
            or env_flag(warmup_env, warmup_default_enabled)
        )

    def _prefix_prefill_token_only_graph_enabled(self) -> bool:
        return env_flag(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_TOKEN_ONLY_GRAPH",
            False,
        )

    def _prefix_prefill_split_on_capture_skip_batch(self, batch_bucket: int) -> int:
        env_name = "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_ON_CAPTURE_SKIP_BATCH"
        if env_name in os.environ:
            return env_int(env_name, 0, minimum=0)
        if self._prefix_prefill_capture_on_miss(batch_bucket):
            return 0
        max_tokens = self.max_generation_tokens
        if self.temperature <= 0.0 and max_tokens is not None:
            greedy_short_max_tokens = env_int(
                "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_GREEDY_SHORT_MAX_TOKENS",
                128,
                minimum=1,
            )
            if 0 < int(max_tokens) <= greedy_short_max_tokens:
                return env_int(
                    "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_GREEDY_SHORT_MAX_BATCH",
                    32,
                    minimum=0,
                )
        return 0

    def _suffix_bucket(self, length: int) -> int:
        configured_buckets = _parse_positive_int_csv(
            os.environ.get("TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS")
        )
        configured_bucket = _bucket_from_values(length, configured_buckets)
        if configured_bucket is not None:
            return configured_bucket
        default_bucket = _bucket_from_values(
            length,
            _default_prefix_prefill_suffix_buckets(
                self.temperature,
                self.max_generation_tokens,
            ),
        )
        if default_bucket is not None:
            return default_bucket
        # Pad the suffix length to a power of two so the ragged-prefill graph key
        # (batch, suffix_bucket, ...) repeats across batches and replays.
        if length <= 1:
            return 1
        return 1 << (length - 1).bit_length()

    def _prefill_prefix_graph_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        # Route suffix prefill through the model's row_indices ragged-prefill
        # LOGITS graph. The shared prefix KV is copied into each (scattered) row,
        # then the suffix is scatter-written and one logit row per request is
        # gathered. Per-row start positions handle MIXED prefix lengths, and the
        # graph replays across changing scattered row sets (unlike the old
        # contiguous for_rows selected-logits path). Batch and suffix are padded
        # to stable buckets so graph shapes repeat.
        prefix_hits = [prefix_hit_tokens for _i, _req, prefix_hit_tokens, _r in group]
        # _prefill_many groups reuse requests by prefix length, so the prefix is
        # uniform here; that lets the suffix attention use a flash causal_lower_right
        # over a static context_len (prefix + suffix_bucket) instead of a boolean
        # mask (which OOMs at large suffix x context). A non-uniform group (rare)
        # falls back to the eager per-suffix-length path.
        mixed_prefixes = len(set(prefix_hits)) != 1
        if mixed_prefixes and not self._mixed_prefix_prefill_enabled():
            return None
        suffixes = [request.prompt[prefix_hits[i]:] for i, (_idx, request, _h, _r) in enumerate(group)]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0:
            return None
        non_common_graph_prefill = self._non_common_prefix_graph_prefill_enabled()
        if not non_common_graph_prefill:
            for _index, _request, _prefix_hit_tokens, reusable in group:
                route_id = reusable.route_id
                if not (isinstance(route_id, tuple) and route_id[:1] == ("common_prefix",)):
                    return None
        elif not env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_GRAPH_PREFILL", False):
            for _index, _request, _prefix_hit_tokens, reusable in group:
                route_id = reusable.route_id
                if isinstance(route_id, tuple) and route_id[:1] == ("finished_prefix",):
                    return None
        cache_max_seq = self._cache_max_seq_len()
        suffix_bucket = self._suffix_bucket(max(suffix_lengths))
        if cache_max_seq is not None:
            suffix_bucket = min(suffix_bucket, max(1, cache_max_seq - max(prefix_hits)))
        if mixed_prefixes:
            context_len = None
            if self._mixed_prefix_dynamic_context_enabled():
                mixed_max_suffix = self._mixed_prefix_dynamic_context_max_suffix()
                dynamic_context_len = _dynamic_prefix_prefill_context_len(
                    max(prefix_hits),
                    suffix_bucket,
                    max_seq_len=cache_max_seq,
                    max_dynamic_suffix=mixed_max_suffix,
                )
                if dynamic_context_len < 0:
                    context_len = dynamic_context_len
        else:
            context_len = self._dynamic_prefix_prefill_context_len(
                prefix_hits[0],
                suffix_bucket,
                max_seq_len=cache_max_seq,
            )
        count = len(group)
        batch_bucket = self._prefill_batch_bucket(count)
        skip_active_row_clear = self._can_skip_prefix_graph_active_row_clear()
        rows = [self._acquire_active_row(clear_cache=not skip_active_row_clear) for _ in group]
        pad_rows: list[int] = []
        pad_prefix_rows: list[int] = []
        try:
            # The prefix KV broadcast is FOLDED INTO the prefill graph (copy from
            # each reusable source row to its active row in one captured pass), so
            # the engine no longer issues ~80 per-layer index_copy launches per
            # batch here -- it only records reuse accounting and source rows.
            copy_start_s = time.perf_counter() if self.profile_timings else 0.0
            source_prefix_rows = [
                reusable.row
                for _index, _request, _prefix_hit_tokens, reusable in group
            ]
            self._record_prefix_reuse_batch(
                (prefix_hit_tokens, reusable)
                for _index, _request, prefix_hit_tokens, reusable in group
            )
            skip_prefix_copy = (
                not mixed_prefixes
                and self._warm_row_prefix_copy_skip_enabled()
                and self._rows_have_cached_prefix(
                    rows,
                    group[0][1].prompt,
                    prefix_hits[0],
                )
            )
            if skip_prefix_copy:
                source_prefix_rows = []
                self.stats.prefill_prefix_copy_skipped_batches += 1
                self.stats.prefill_prefix_copy_skipped_tokens += prefix_hits[0] * len(rows)
            padded_suffixes = [
                [*suffix, *([0] * (suffix_bucket - len(suffix)))]
                for suffix in suffixes
            ]
            start_lens = list(prefix_hits)
            if batch_bucket > count:
                dummy_suffix = padded_suffixes[0]
                for _ in range(batch_bucket - count):
                    pad_row = self._acquire_active_row_or_none(clear_cache=not skip_active_row_clear)
                    if pad_row is None:
                        prefix_pad_row = self._acquire_free_prefix_row_or_none()
                        if prefix_pad_row is None:
                            break
                        pad_prefix_rows.append(prefix_pad_row)
                        pad_row = prefix_pad_row
                    else:
                        pad_rows.append(pad_row)
                    padded_suffixes.append(list(dummy_suffix))
                    start_lens.append(prefix_hits[0])
                    if source_prefix_rows:
                        source_prefix_rows.append(source_prefix_rows[0])
            if not mixed_prefixes and len(set(source_prefix_rows)) == 1:
                source_prefix_rows = [source_prefix_rows[0]]
            shape_key = (
                "prefix_graph:"
                f"b{batch_bucket}:"
                f"s{suffix_bucket}:"
                f"p{min(prefix_hits)}-{max(prefix_hits)}:"
                f"src{len(source_prefix_rows)}:"
                f"mixed{int(mixed_prefixes)}"
            )
            self._record_shape_count(self.stats.prefill_shape_counts, shape_key)
            self._record_prefill_shape_batch_details(
                shape_key,
                real_batch=count,
                suffix_lengths=suffix_lengths,
            )
            self._record_prefix_graph_route_totals(shape_key, group, suffix_lengths)
            shape_wall_start_s = copy_start_s
            if self.profile_timings:
                copy_elapsed_ms = (time.perf_counter() - copy_start_s) * 1000.0
                self.stats.prefill_copy_ms += copy_elapsed_ms
                self._record_shape_time(
                    self.stats.prefill_shape_copy_ms,
                    shape_key,
                    copy_elapsed_ms,
                )
            setup_start_s = time.perf_counter() if self.profile_timings else 0.0
            all_rows = rows + pad_rows + pad_prefix_rows
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            row_indices = self._device_index_tensor(tuple(all_rows))
            model_row_indices = (
                None
                if all_rows == list(range(len(all_rows)))
                else row_indices
            )
            src_prefix_row = (
                self._device_index_tensor(tuple(source_prefix_rows))
                if source_prefix_rows
                else None
            )
            required = max(all_rows + source_prefix_rows) + 1
            seq_lens = self._prefix_prefill_seq_lens_tensor(
                all_rows,
                start_lens,
                row_indices=row_indices,
                required=required,
            )
            logit_positions = self._device_index_tensor(
                tuple(
                    [length - 1 for length in suffix_lengths]
                    + [0] * (len(pad_rows) + len(pad_prefix_rows))
                )
            )
            packed_prefill_pattern_key = _packed_prefill_candidate_pattern(
                shape_key,
                suffix_lengths=suffix_lengths,
                start_lens=start_lens[:count],
            )
            self._record_packed_prefill_candidate(
                shape_key,
                suffix_lengths=suffix_lengths,
                start_lens=start_lens[:count],
                model_tokens=int(input_ids.numel()),
                packed_prefill_pattern_key=packed_prefill_pattern_key,
            )
            if self.profile_timings:
                setup_elapsed_ms = (time.perf_counter() - setup_start_s) * 1000.0
                self.stats.prefill_setup_ms += setup_elapsed_ms
                self._record_shape_time(
                    self.stats.prefill_shape_setup_ms,
                    shape_key,
                    setup_elapsed_ms,
                )
            forward_start_s = time.perf_counter() if self.profile_timings else 0.0
            logits = None
            next_token_tensor: Tensor | None = None
            prefix_copy_len = max(prefix_hits) if mixed_prefixes and context_len is None else None
            output_group = group
            output_rows = rows
            output_suffix_lengths = suffix_lengths
            model_rows_for_stats = int(input_ids.size(0))
            model_tokens_for_stats = int(input_ids.numel())
            fixed_packed = self._try_fixed_capacity_packed_prefill_logits(
                input_ids=input_ids,
                group=group,
                rows=rows,
                suffixes=suffixes,
                suffix_lengths=suffix_lengths,
                suffix_bucket=suffix_bucket,
                source_prefix_rows=source_prefix_rows,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                pad_rows=pad_rows,
                pad_prefix_rows=pad_prefix_rows,
                capture_on_miss=self._prefix_prefill_capture_on_miss(batch_bucket),
                profile_shape_key=shape_key,
                packed_prefill_pattern_key=packed_prefill_pattern_key,
                skip_active_row_clear=skip_active_row_clear,
            )
            if fixed_packed is not None:
                (
                    logits,
                    output_group,
                    output_rows,
                    output_suffix_lengths,
                    model_rows_for_stats,
                    model_tokens_for_stats,
                ) = fixed_packed
            else:
                self._record_prefill_row_index_mode(
                    shape_key,
                    omitted=model_row_indices is None,
                    model_rows=int(input_ids.size(0)),
                )
                if not mixed_prefixes or self._mixed_prefix_prefill_graph_enabled():
                    if (
                        self._prefix_prefill_token_only_graph_enabled()
                        and self._prefix_prefill_group_can_omit_logits(group)
                    ):
                        next_token_tensor = self._try_ragged_prefill_greedy_tokens(
                            input_ids,
                            seq_lens,
                            model_row_indices,
                            logit_positions,
                            [request for _index, request, _prefix_hit_tokens, _reusable in group],
                            context_len,
                            src_prefix_row,
                            prefix_copy_len,
                            capture_on_miss=self._prefix_prefill_token_graph_capture_on_miss(),
                            profile_shape_key=shape_key,
                            packed_prefill_pattern_key=packed_prefill_pattern_key,
                        )
                    token_graph_output = (
                        self._try_ragged_prefill_logits_with_greedy_tokens(
                            input_ids,
                            seq_lens,
                            model_row_indices,
                            logit_positions,
                            [request for _index, request, _prefix_hit_tokens, _reusable in group],
                            context_len,
                            src_prefix_row,
                            prefix_copy_len,
                            capture_on_miss=self._prefix_prefill_token_graph_capture_on_miss(),
                            profile_shape_key=shape_key,
                            packed_prefill_pattern_key=packed_prefill_pattern_key,
                        )
                        if next_token_tensor is None and self._prefix_prefill_token_graph_enabled()
                        else None
                    )
                    if token_graph_output is not None:
                        logits, next_token_tensor = token_graph_output
                    elif next_token_tensor is None:
                        logits = self._try_ragged_prefill_logits(
                            input_ids,
                            seq_lens,
                            model_row_indices,
                            logit_positions,
                            context_len,
                            src_prefix_row,
                            prefix_copy_len,
                            capture_on_miss=self._prefix_prefill_capture_on_miss(batch_bucket),
                            profile_shape_key=shape_key,
                            packed_prefill_pattern_key=packed_prefill_pattern_key,
                        )
            if logits is None and next_token_tensor is None:
                logits = self._ragged_prefill_logits_eager(
                    input_ids,
                    seq_lens,
                    model_row_indices,
                    logit_positions,
                    context_len,
                    src_prefix_row,
                    prefix_copy_len,
                )
            if self.profile_timings and (logits is not None or next_token_tensor is not None):
                # force the prefill graph/forward to complete for honest timing
                torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
                self._flush_prefill_graph_gpu_timers()
                forward_elapsed_ms = (time.perf_counter() - forward_start_s) * 1000.0
                self.stats.prefill_forward_ms += forward_elapsed_ms
                self._record_shape_time(
                    self.stats.prefill_shape_forward_ms,
                    shape_key,
                    forward_elapsed_ms,
                )
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            pad_rows = []
            for pad_row in pad_prefix_rows:
                self._release_prefix_row(pad_row)
            pad_prefix_rows = []
            if logits is None and next_token_tensor is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", count, tokens=count * max(suffix_lengths))
            self._record_shape_total(
                self.stats.prefill_shape_active_requests,
                shape_key,
                count,
            )
            self._record_shape_total(
                self.stats.prefill_shape_model_rows,
                shape_key,
                model_rows_for_stats,
            )
            self._record_shape_total(
                self.stats.prefill_shape_active_tokens,
                shape_key,
                sum(output_suffix_lengths),
            )
            self._record_shape_total(
                self.stats.prefill_shape_model_tokens,
                shape_key,
                model_tokens_for_stats,
            )
            self.stats.prefill_prefix_reuse_batches += 1
            self.stats.prefill_padded_suffix_batches += 1
            sample_start_s = time.perf_counter() if self.profile_timings else 0.0
            sample_select_start_s = time.perf_counter() if self.profile_timings else 0.0
            if next_token_tensor is None:
                next_token_tensor = self._sample_logits_for_requests(
                    logits[: len(output_group), -1, :],
                    [request for _original_index, request, _prefix_hit_tokens, _reusable in output_group],
                ).detach()
            else:
                next_token_tensor = next_token_tensor[: len(output_group)].detach()
            sample_select_ms = (
                (time.perf_counter() - sample_select_start_s) * 1000.0
                if self.profile_timings
                else 0.0
            )
            sample_readback_start_s = time.perf_counter() if self.profile_timings else 0.0
            next_tokens = next_token_tensor.cpu().tolist()
            sample_readback_ms = (
                (time.perf_counter() - sample_readback_start_s) * 1000.0
                if self.profile_timings
                else 0.0
            )
            if self.profile_timings:
                sample_elapsed_ms = (time.perf_counter() - sample_start_s) * 1000.0
                self.stats.prefill_sample_ms += sample_elapsed_ms
                self.stats.prefill_sample_select_ms += sample_select_ms
                self.stats.prefill_sample_readback_ms += sample_readback_ms
                self._record_shape_time(
                    self.stats.prefill_shape_sample_ms,
                    shape_key,
                    sample_elapsed_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_sample_select_ms,
                    shape_key,
                    sample_select_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_sample_readback_ms,
                    shape_key,
                    sample_readback_ms,
                )
            active: list[_ActiveRequest] = []
            output_prompt_lens = [
                len(request.prompt)
                for _original_index, request, _prefix_hit_tokens, _reusable in output_group
            ]
            if self.profile_timings:
                state_start_s = time.perf_counter()
                state_seq_start_s = time.perf_counter()
                self._set_cache_row_seq_lens(
                    output_rows[: len(output_group)],
                    output_prompt_lens,
                )
                state_seq_ms = (time.perf_counter() - state_seq_start_s) * 1000.0

                state_store_start_s = time.perf_counter()
                for row_index, (_original_index, request, _prefix_hit_tokens, _reusable) in enumerate(output_group):
                    row = output_rows[row_index]
                    self._store_reusable_prefix(
                        request.request_id,
                        request.prompt,
                        row,
                        None if logits is None else logits[row_index : row_index + 1],
                        allow_pinned=self._allow_pinned_full_prompt_store(request),
                    )
                state_store_ms = (time.perf_counter() - state_store_start_s) * 1000.0

                state_create_start_s = time.perf_counter()
                for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(output_group):
                    row = output_rows[row_index]
                    prompt_len = output_prompt_lens[row_index]
                    next_token = int(next_tokens[row_index])
                    state = _ActiveRequest(
                        original_index=original_index,
                        request=request,
                        tokens=[*request.prompt, next_token],
                        generated=1,
                        row=row,
                        last_token=next_token,
                        seq_len=prompt_len,
                        prefix_hit_tokens=prefix_hit_tokens,
                        started_step=step,
                    )
                    self._record_token_event(
                        events,
                        state,
                        next_token,
                        step,
                        finished=self._should_finish_before_decode(state),
                    )
                    active.append(state)
                state_create_ms = (time.perf_counter() - state_create_start_s) * 1000.0
                state_elapsed_ms = (time.perf_counter() - state_start_s) * 1000.0
                self.stats.prefill_state_ms += state_elapsed_ms
                self.stats.prefill_state_seq_ms += state_seq_ms
                self.stats.prefill_state_store_ms += state_store_ms
                self.stats.prefill_state_create_ms += state_create_ms
                self._record_shape_time(
                    self.stats.prefill_shape_state_ms,
                    shape_key,
                    state_elapsed_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_state_seq_ms,
                    shape_key,
                    state_seq_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_state_store_ms,
                    shape_key,
                    state_store_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_state_create_ms,
                    shape_key,
                    state_create_ms,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_wall_ms,
                    shape_key,
                    (time.perf_counter() - shape_wall_start_s) * 1000.0,
                )
            else:
                self._set_cache_row_seq_lens(
                    output_rows[: len(output_group)],
                    output_prompt_lens,
                )
                for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(output_group):
                    row = output_rows[row_index]
                    prompt_len = output_prompt_lens[row_index]
                    self._store_reusable_prefix(
                        request.request_id,
                        request.prompt,
                        row,
                        None if logits is None else logits[row_index : row_index + 1],
                        allow_pinned=self._allow_pinned_full_prompt_store(request),
                    )
                    next_token = int(next_tokens[row_index])
                    state = _ActiveRequest(
                        original_index=original_index,
                        request=request,
                        tokens=[*request.prompt, next_token],
                        generated=1,
                        row=row,
                        last_token=next_token,
                        seq_len=prompt_len,
                        prefix_hit_tokens=prefix_hit_tokens,
                        started_step=step,
                    )
                    self._record_token_event(
                        events,
                        state,
                        next_token,
                        step,
                        finished=self._should_finish_before_decode(state),
                    )
                    active.append(state)
            return active
        except Exception:
            for pad_row in pad_rows:
                self._release_active_row(pad_row)
            for pad_row in pad_prefix_rows:
                self._release_prefix_row(pad_row)
            for row in rows:
                self._release_active_row(row)
            raise

    def _cache_max_seq_len(self) -> int | None:
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if not layers:
            return None
        return getattr(layers[0], "max_seq_len", None)

    def _record_packed_prefill_eager_result(
        self,
        *,
        profile_shape_key: str | None,
        real_tokens: int,
        model_tokens: int,
        elapsed_ms: float | None,
    ) -> None:
        saved_tokens = max(0, int(model_tokens) - int(real_tokens))
        self.stats.prefill_packed_eager_calls += 1
        self.stats.prefill_packed_eager_tokens += int(real_tokens)
        self.stats.prefill_packed_eager_model_tokens += int(model_tokens)
        self.stats.prefill_packed_eager_saved_tokens += saved_tokens
        if elapsed_ms is None:
            return
        self.stats.prefill_packed_eager_ms += elapsed_ms
        if profile_shape_key is None:
            return
        self._record_shape_count(
            self.stats.prefill_packed_eager_shape_counts,
            profile_shape_key,
        )
        self._record_shape_total(
            self.stats.prefill_packed_eager_shape_tokens,
            profile_shape_key,
            int(real_tokens),
        )
        self._record_shape_total(
            self.stats.prefill_packed_eager_shape_model_tokens,
            profile_shape_key,
            int(model_tokens),
        )
        self._record_shape_total(
            self.stats.prefill_packed_eager_shape_saved_tokens,
            profile_shape_key,
            saved_tokens,
        )
        self._record_shape_time(
            self.stats.prefill_packed_eager_shape_ms,
            profile_shape_key,
            elapsed_ms,
        )

    def _record_fixed_capacity_packed_prefill_reject(self, reason: str) -> None:
        counts = self.stats.prefill_packed_fixed_capacity_reject_reason_counts
        counts[reason] = int(counts.get(reason, 0)) + 1

    def _try_fixed_capacity_packed_prefill_logits(
        self,
        *,
        input_ids: Tensor,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        rows: list[int],
        suffixes: list[Sequence[int]],
        suffix_lengths: list[int],
        suffix_bucket: int,
        source_prefix_rows: list[int],
        src_prefix_row: Tensor | None,
        prefix_copy_len: int | None,
        pad_rows: list[int],
        pad_prefix_rows: list[int],
        capture_on_miss: bool,
        profile_shape_key: str,
        packed_prefill_pattern_key: str,
        skip_active_row_clear: bool,
    ) -> tuple[
        Tensor,
        list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        list[int],
        list[int],
        int,
        int,
    ] | None:
        if not env_flag(
            "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_FIXED_CAPACITY_GRAPH",
            False,
        ):
            return None
        if not _packed_prefill_fixed_capacity_enabled(
            profile_shape_key=profile_shape_key,
            packed_prefill_pattern_key=packed_prefill_pattern_key,
        ):
            self.stats.prefill_packed_fixed_capacity_attempts += 1
            self._record_fixed_capacity_packed_prefill_reject("pattern_disabled")
            return None
        self.stats.prefill_packed_fixed_capacity_attempts += 1
        packed_graph = getattr(
            self.model,
            "try_prefill_ragged_logits_packed_eager_graph",
            None,
        )
        if packed_graph is None:
            self._record_fixed_capacity_packed_prefill_reject("no_graph")
            return None
        if not group:
            self._record_fixed_capacity_packed_prefill_reject("empty_group")
            return None
        if len(rows) != len(group) or len(suffixes) != len(group):
            self._record_fixed_capacity_packed_prefill_reject("shape_mismatch")
            return None
        if src_prefix_row is None and any(prefix_hit > 0 for _i, _req, prefix_hit, _r in group):
            self._record_fixed_capacity_packed_prefill_reject("missing_src_prefix")
            return None
        if src_prefix_row is not None and int(src_prefix_row.numel()) != 1:
            self._record_fixed_capacity_packed_prefill_reject("multi_src_prefix")
            return None
        if suffix_bucket <= 0 or any(length <= 0 or length > suffix_bucket for length in suffix_lengths):
            self._record_fixed_capacity_packed_prefill_reject("invalid_suffix")
            return None

        start_lens = [int(prefix_hit) for _i, _request, prefix_hit, _reusable in group]
        current_counts = _packed_prefill_group_counts(
            suffix_lengths=suffix_lengths,
            start_lens=start_lens,
        )
        capacities = self._packed_prefill_fixed_capacity_counts.setdefault(
            packed_prefill_pattern_key,
            {},
        )
        grew_capacity = False
        for key, count in current_counts.items():
            if count > capacities.get(key, 0):
                capacities[key] = int(count)
                grew_capacity = True
        self._packed_prefill_fixed_capacity_seen[packed_prefill_pattern_key] = (
            self._packed_prefill_fixed_capacity_seen.get(packed_prefill_pattern_key, 0) + 1
        )
        if grew_capacity:
            self._packed_prefill_fixed_capacity_stable_seen[packed_prefill_pattern_key] = 0
            self._record_fixed_capacity_packed_prefill_reject("capacity_grew")
            return None
        stable_seen = (
            self._packed_prefill_fixed_capacity_stable_seen.get(
                packed_prefill_pattern_key,
                0,
            )
            + 1
        )
        self._packed_prefill_fixed_capacity_stable_seen[packed_prefill_pattern_key] = stable_seen
        min_calls = env_int(
            "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_FIXED_CAPACITY_MIN_CALLS",
            2,
            minimum=1,
        )
        if stable_seen < min_calls:
            self._record_fixed_capacity_packed_prefill_reject("warming")
            return None

        real_by_key: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (start_len, suffix_len) in enumerate(zip(start_lens, suffix_lengths)):
            real_by_key[(int(start_len), int(suffix_len))].append(index)

        slot_specs: list[tuple[int, int, int | None]] = []
        for key in sorted(capacities):
            start_len, suffix_len = key
            if suffix_len <= 0 or suffix_len > suffix_bucket:
                self._record_fixed_capacity_packed_prefill_reject("invalid_capacity")
                return None
            entries = real_by_key.get(key, [])
            for slot in range(capacities[key]):
                real_index = entries[slot] if slot < len(entries) else None
                slot_specs.append((start_len, suffix_len, real_index))
        if not slot_specs:
            self._record_fixed_capacity_packed_prefill_reject("no_slots")
            return None

        fixed_tokens = sum(suffix_len for _start_len, suffix_len, _real_index in slot_specs)
        dense_tokens = int(input_ids.numel())
        if fixed_tokens <= 0 or fixed_tokens >= dense_tokens:
            self._record_fixed_capacity_packed_prefill_reject("no_savings")
            return None
        graph_none_key = (
            packed_prefill_pattern_key,
            tuple((start_len, suffix_len) for start_len, suffix_len, _real_index in slot_specs),
        )
        cache_graph_none = not bool(capture_on_miss)
        if (
            cache_graph_none
            and graph_none_key in self._packed_prefill_fixed_capacity_graph_none_keys
        ):
            self._record_fixed_capacity_packed_prefill_reject("graph_returned_none_cached")
            return None

        dummy_needed = sum(
            1 for _start_len, _suffix_len, real_index in slot_specs if real_index is None
        )
        dummy_pool = list(pad_rows) + list(pad_prefix_rows)
        extra_active_rows: list[int] = []
        extra_prefix_rows: list[int] = []
        extras_committed = False
        try:
            while len(dummy_pool) < dummy_needed:
                pad_row = self._acquire_active_row_or_none(
                    clear_cache=not skip_active_row_clear,
                )
                if pad_row is not None:
                    extra_active_rows.append(pad_row)
                    dummy_pool.append(pad_row)
                    continue
                prefix_pad_row = self._acquire_free_prefix_row_or_none()
                if prefix_pad_row is None:
                    self._record_fixed_capacity_packed_prefill_reject("no_dummy_rows")
                    return None
                extra_prefix_rows.append(prefix_pad_row)
                dummy_pool.append(prefix_pad_row)

            dummy_iter = iter(dummy_pool)
            fixed_suffixes: list[list[int]] = []
            fixed_rows: list[int] = []
            fixed_start_lens: list[int] = []
            fixed_q_lens_values: list[int] = []
            fixed_logit_positions_values: list[int] = []
            real_slot_indices: list[int] = []
            fixed_group: list[tuple[int, ServingRequest, int, _ReusablePrefix]] = []
            fixed_real_rows: list[int] = []
            fixed_real_suffix_lengths: list[int] = []
            for slot_index, (start_len, suffix_len, real_index) in enumerate(slot_specs):
                fixed_start_lens.append(start_len)
                fixed_q_lens_values.append(suffix_len)
                fixed_logit_positions_values.append(suffix_len - 1)
                if real_index is None:
                    fixed_rows.append(next(dummy_iter))
                    fixed_suffixes.append([0] * suffix_bucket)
                    continue
                suffix = list(suffixes[real_index])
                fixed_rows.append(rows[real_index])
                fixed_suffixes.append([*suffix, *([0] * (suffix_bucket - len(suffix)))])
                real_slot_indices.append(slot_index)
                fixed_group.append(group[real_index])
                fixed_real_rows.append(rows[real_index])
                fixed_real_suffix_lengths.append(suffix_lengths[real_index])
            if len(real_slot_indices) != len(group):
                self._record_fixed_capacity_packed_prefill_reject("real_slot_mismatch")
                return None

            required = max(fixed_rows + source_prefix_rows) + 1
            seq_lens_list = [0] * required
            for physical_row, start_len in zip(fixed_rows, fixed_start_lens):
                seq_lens_list[physical_row] = start_len
            fixed_input_ids = torch.tensor(
                fixed_suffixes,
                device=self.device,
                dtype=torch.long,
            )
            fixed_seq_lens = self._device_index_tensor(tuple(seq_lens_list))
            fixed_row_indices = self._device_index_tensor(tuple(fixed_rows))
            fixed_q_lens = self._device_index_tensor(tuple(fixed_q_lens_values))
            fixed_logit_positions = self._device_index_tensor(
                tuple(fixed_logit_positions_values)
            )
            packed_start_s = time.perf_counter() if self.profile_timings else 0.0
            logits = packed_graph(
                fixed_input_ids,
                self._require_cache(),
                seq_lens=fixed_seq_lens,
                q_lens=fixed_q_lens,
                row_indices=fixed_row_indices,
                logit_positions=fixed_logit_positions,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
                capture_on_miss=capture_on_miss,
            )
            if logits is None:
                self._record_fixed_capacity_packed_prefill_reject("graph_returned_none")
                if cache_graph_none:
                    self._packed_prefill_fixed_capacity_graph_none_keys.add(graph_none_key)
                return None
            real_index_tensor = self._device_index_tensor(tuple(real_slot_indices)).to(
                device=logits.device
            )
            real_logits = logits.index_select(0, real_index_tensor)
            elapsed_ms = (
                (time.perf_counter() - packed_start_s) * 1000.0
                if self.profile_timings
                else None
            )
            self.stats.prefill_packed_fixed_capacity_accepts += 1
            self._record_packed_prefill_eager_result(
                profile_shape_key=profile_shape_key,
                real_tokens=sum(suffix_lengths),
                model_tokens=fixed_tokens,
                elapsed_ms=elapsed_ms,
            )
            pad_rows.extend(extra_active_rows)
            pad_prefix_rows.extend(extra_prefix_rows)
            extras_committed = True
            return (
                real_logits,
                fixed_group,
                fixed_real_rows,
                fixed_real_suffix_lengths,
                len(slot_specs),
                fixed_tokens,
            )
        except Exception as exc:
            self._record_fixed_capacity_packed_prefill_reject("exception")
            warn_optional_failure("continuous.fixed_capacity_packed_prefill", exc)
            return None
        finally:
            if not extras_committed:
                for row in extra_active_rows:
                    self._release_active_row(row)
                for row in extra_prefix_rows:
                    self._release_prefix_row(row)

    def _try_ragged_prefill_logits(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        *,
        capture_on_miss: bool = True,
        profile_shape_key: str | None = None,
        packed_prefill_pattern_key: str | None = None,
    ) -> Tensor | None:
        if not self._cache_supports_tensor_ragged_prefill():
            return None
        packed_eager_enabled = env_flag(
            "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER",
            False,
        )
        if not packed_eager_enabled:
            packed_eager_enabled = _packed_prefill_eager_pattern_matches(
                profile_shape_key=profile_shape_key,
                packed_prefill_pattern_key=packed_prefill_pattern_key,
            )
        if packed_eager_enabled:
            q_lens = logit_positions + 1
            if bool(torch.any(q_lens < input_ids.size(1))):
                packed_start_s = time.perf_counter() if self.profile_timings else 0.0
                logits = None
                packed_graph_enabled = env_flag(
                    "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH",
                    False,
                )
                if packed_graph_enabled:
                    packed_graph = getattr(
                        self.model,
                        "try_prefill_ragged_logits_packed_eager_graph",
                        None,
                    )
                    if packed_graph is not None:
                        logits = packed_graph(
                            input_ids,
                            self._require_cache(),
                            seq_lens=seq_lens,
                            q_lens=q_lens,
                            row_indices=row_indices,
                            logit_positions=logit_positions,
                            src_prefix_row=src_prefix_row,
                            prefix_copy_len=prefix_copy_len,
                            capture_on_miss=capture_on_miss,
                        )
                packed_graph_only = packed_graph_enabled and env_flag(
                    "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH_ONLY",
                    False,
                )
                if logits is None and not packed_graph_only:
                    packed_eager = getattr(self.model, "prefill_ragged_logits_packed_eager", None)
                    if packed_eager is not None:
                        logits = packed_eager(
                            input_ids,
                            self._require_cache(),
                            seq_lens=seq_lens,
                            q_lens=q_lens,
                            row_indices=row_indices,
                            logit_positions=logit_positions,
                            src_prefix_row=src_prefix_row,
                            prefix_copy_len=prefix_copy_len,
                        )
                if logits is not None:
                    real_tokens = int(q_lens.sum().item())
                    model_tokens = int(input_ids.size(0) * input_ids.size(1))
                    elapsed_ms = (
                        (time.perf_counter() - packed_start_s) * 1000.0
                        if self.profile_timings
                        else None
                    )
                    self._record_packed_prefill_eager_result(
                        profile_shape_key=profile_shape_key,
                        real_tokens=real_tokens,
                        model_tokens=model_tokens,
                        elapsed_ms=elapsed_ms,
                    )
                    return logits
        graph = getattr(self.model, "try_prefill_ragged_logits_graph", None)
        if graph is None:
            return None
        graph_start_s = time.perf_counter() if self.profile_timings else 0.0
        graph_shape_key = self._ragged_prefill_graph_shape_key(
            input_ids,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
        )
        gpu_events = self._start_prefill_graph_gpu_timer()
        logits = graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            capture_on_miss=capture_on_miss,
        )
        if logits is None:
            self._record_ragged_prefill_graph_miss(
                input_ids,
                row_indices=row_indices,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                profile_shape_key=profile_shape_key,
            )
            return None
        self._record_ragged_prefill_graph_hit(
            graph_start_s=graph_start_s,
            graph_shape_key=graph_shape_key,
            gpu_events=gpu_events,
            profile_shape_key=profile_shape_key,
        )
        return logits

    def _try_ragged_prefill_greedy_tokens(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        requests: Sequence[ServingRequest],
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        *,
        capture_on_miss: bool = True,
        profile_shape_key: str | None = None,
        packed_prefill_pattern_key: str | None = None,
    ) -> Tensor | None:
        if not self._cache_supports_tensor_ragged_prefill():
            return None
        graph = self._prefill_ragged_token_graph
        if graph is None:
            return None
        packed_eager_enabled = env_flag(
            "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER",
            False,
        )
        if not packed_eager_enabled:
            packed_eager_enabled = _packed_prefill_eager_pattern_matches(
                profile_shape_key=profile_shape_key,
                packed_prefill_pattern_key=packed_prefill_pattern_key,
            )
        if packed_eager_enabled and bool(torch.any(logit_positions + 1 < input_ids.size(1))):
            return None
        shared_temperature = self._shared_temperature_for_requests(
            requests,
            limit=int(input_ids.size(0)),
        )
        if shared_temperature is None or shared_temperature > 0.0:
            return None
        graph_start_s = time.perf_counter() if self.profile_timings else 0.0
        graph_shape_key = self._ragged_prefill_graph_shape_key(
            input_ids,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
        )
        if not capture_on_miss and graph_shape_key in self._prefill_token_graph_miss_keys:
            return None
        gpu_events = self._start_prefill_graph_gpu_timer() if capture_on_miss else None
        tokens = graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            temperature=shared_temperature,
            capture_on_miss=capture_on_miss,
        )
        if tokens is None:
            if not capture_on_miss:
                self._prefill_token_graph_miss_keys.add(graph_shape_key)
            return None
        self._record_ragged_prefill_graph_hit(
            graph_start_s=graph_start_s,
            graph_shape_key=graph_shape_key,
            gpu_events=gpu_events,
            profile_shape_key=profile_shape_key,
        )
        return tokens

    def _try_ragged_prefill_logits_with_greedy_tokens(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        requests: Sequence[ServingRequest],
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        *,
        capture_on_miss: bool = True,
        profile_shape_key: str | None = None,
        packed_prefill_pattern_key: str | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        if not self._cache_supports_tensor_ragged_prefill():
            return None
        graph = self._prefill_ragged_token_logits_graph
        if graph is None:
            return None
        packed_eager_enabled = env_flag(
            "TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER",
            False,
        )
        if not packed_eager_enabled:
            packed_eager_enabled = _packed_prefill_eager_pattern_matches(
                profile_shape_key=profile_shape_key,
                packed_prefill_pattern_key=packed_prefill_pattern_key,
            )
        if packed_eager_enabled and bool(torch.any(logit_positions + 1 < input_ids.size(1))):
            return None
        shared_temperature = self._shared_temperature_for_requests(
            requests,
            limit=int(input_ids.size(0)),
        )
        if shared_temperature is None or shared_temperature > 0.0:
            return None
        graph_start_s = time.perf_counter() if self.profile_timings else 0.0
        graph_shape_key = self._ragged_prefill_graph_shape_key(
            input_ids,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
        )
        if not capture_on_miss and graph_shape_key in self._prefill_token_graph_miss_keys:
            return None
        gpu_events = self._start_prefill_graph_gpu_timer() if capture_on_miss else None
        output = graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            temperature=shared_temperature,
            capture_on_miss=capture_on_miss,
        )
        if output is None:
            if not capture_on_miss:
                self._prefill_token_graph_miss_keys.add(graph_shape_key)
            return None
        logits, tokens = output
        self._record_ragged_prefill_graph_hit(
            graph_start_s=graph_start_s,
            graph_shape_key=graph_shape_key,
            gpu_events=gpu_events,
            profile_shape_key=profile_shape_key,
        )
        return logits, tokens

    def _record_ragged_prefill_graph_hit(
        self,
        *,
        graph_start_s: float,
        graph_shape_key: str,
        gpu_events: tuple[object, object] | None,
        profile_shape_key: str | None = None,
    ) -> None:
        self.stats.prefill_graph_hits += 1
        captured = getattr(self.model, "_last_ragged_prefill_graph_captured", None)
        if not isinstance(captured, bool):
            return
        self._stop_prefill_graph_gpu_timer(
            gpu_events,
            captured=captured,
            graph_shape_key=graph_shape_key,
            profile_shape_key=profile_shape_key,
        )
        elapsed_ms = (time.perf_counter() - graph_start_s) * 1000.0 if self.profile_timings else 0.0
        if captured:
            self.stats.prefill_graph_captures += 1
            self.stats.prefill_graph_capture_ms += elapsed_ms
            self._record_shape_count(
                self.stats.prefill_graph_capture_shape_counts,
                graph_shape_key,
            )
            self._record_shape_time(
                self.stats.prefill_graph_capture_shape_ms,
                graph_shape_key,
                elapsed_ms,
            )
            if profile_shape_key is not None:
                self._record_shape_count(
                    self.stats.prefill_shape_graph_capture_counts,
                    profile_shape_key,
                )
                self._record_shape_time(
                    self.stats.prefill_shape_graph_capture_ms,
                    profile_shape_key,
                    elapsed_ms,
                )
            return

        self.stats.prefill_graph_replays += 1
        self.stats.prefill_graph_replay_ms += elapsed_ms
        self._record_shape_time(
            self.stats.prefill_graph_replay_shape_ms,
            graph_shape_key,
            elapsed_ms,
        )
        if profile_shape_key is not None:
            self._record_shape_count(
                self.stats.prefill_shape_graph_replay_counts,
                profile_shape_key,
            )
            self._record_shape_time(
                self.stats.prefill_shape_graph_replay_ms,
                profile_shape_key,
                elapsed_ms,
            )

    def _try_ragged_prefill_cache(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
        *,
        capture_on_miss: bool = True,
        profile_shape_key: str | None = None,
    ) -> bool:
        if not self._cache_supports_tensor_ragged_prefill():
            return False
        graph = getattr(self.model, "try_prefill_ragged_cache_graph", None)
        if graph is None:
            return self._ragged_prefill_cache_eager(
                input_ids,
                seq_lens,
                row_indices,
                context_len,
                src_prefix_row,
                prefix_copy_len,
            )
        graph_start_s = time.perf_counter() if self.profile_timings else 0.0
        graph_shape_key = self._ragged_prefill_graph_shape_key(
            input_ids,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
        )
        gpu_events = self._start_prefill_graph_gpu_timer()
        filled = graph(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
            capture_on_miss=capture_on_miss,
        )
        if not filled:
            self._record_ragged_prefill_graph_miss(
                input_ids,
                row_indices=row_indices,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                profile_shape_key=profile_shape_key,
            )
            return self._ragged_prefill_cache_eager(
                input_ids,
                seq_lens,
                row_indices,
                context_len,
                src_prefix_row,
                prefix_copy_len,
            )
        self.stats.prefill_graph_hits += 1
        captured = getattr(self.model, "_last_ragged_prefill_graph_captured", None)
        if isinstance(captured, bool):
            self._stop_prefill_graph_gpu_timer(
                gpu_events,
                captured=captured,
                graph_shape_key=graph_shape_key,
                profile_shape_key=profile_shape_key,
            )
            elapsed_ms = (time.perf_counter() - graph_start_s) * 1000.0 if self.profile_timings else 0.0
            if captured:
                self.stats.prefill_graph_captures += 1
                self.stats.prefill_graph_capture_ms += elapsed_ms
                self._record_shape_count(
                    self.stats.prefill_graph_capture_shape_counts,
                    graph_shape_key,
                )
                self._record_shape_time(
                    self.stats.prefill_graph_capture_shape_ms,
                    graph_shape_key,
                    elapsed_ms,
                )
                if profile_shape_key is not None:
                    self._record_shape_count(
                        self.stats.prefill_shape_graph_capture_counts,
                        profile_shape_key,
                    )
                    self._record_shape_time(
                        self.stats.prefill_shape_graph_capture_ms,
                        profile_shape_key,
                        elapsed_ms,
                    )
            else:
                self.stats.prefill_graph_replays += 1
                self.stats.prefill_graph_replay_ms += elapsed_ms
                self._record_shape_time(
                    self.stats.prefill_graph_replay_shape_ms,
                    graph_shape_key,
                    elapsed_ms,
                )
                if profile_shape_key is not None:
                    self._record_shape_count(
                        self.stats.prefill_shape_graph_replay_counts,
                        profile_shape_key,
                    )
                    self._record_shape_time(
                        self.stats.prefill_shape_graph_replay_ms,
                        profile_shape_key,
                        elapsed_ms,
                    )
        return True

    def _record_ragged_prefill_graph_miss(
        self,
        input_ids: Tensor,
        *,
        row_indices: Tensor | None,
        context_len: int | None,
        src_prefix_row: Tensor | None,
        profile_shape_key: str | None = None,
    ) -> None:
        self.stats.prefill_graph_misses += 1
        self._record_shape_count(
            self.stats.prefill_graph_miss_shape_counts,
            self._ragged_prefill_graph_shape_key(
                input_ids,
                row_indices=row_indices,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
            ),
        )
        if profile_shape_key is not None:
            self._record_shape_count(
                self.stats.prefill_shape_graph_miss_counts,
                profile_shape_key,
            )

    def _ragged_prefill_cache_eager(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
    ) -> bool:
        if not self._cache_supports_tensor_ragged_prefill():
            return False
        eager = getattr(self.model, "prefill_ragged_cache", None)
        if eager is None:
            return False
        return bool(
            eager(
                input_ids,
                self._require_cache(),
                seq_lens=seq_lens,
                row_indices=row_indices,
                context_len=context_len,
                src_prefix_row=src_prefix_row,
                prefix_copy_len=prefix_copy_len,
            )
        )

    @staticmethod
    def _ragged_prefill_graph_shape_key(
        input_ids: Tensor,
        *,
        row_indices: Tensor | None,
        context_len: int | None,
        src_prefix_row: Tensor | None,
    ) -> str:
        src_rows = int(src_prefix_row.numel()) if src_prefix_row is not None else 0
        return (
            "ragged_prefill:"
            f"b{input_ids.size(0)}:"
            f"s{input_ids.size(1)}:"
            f"rows{int(row_indices is not None)}:"
            f"ctx{context_len if context_len is not None else -1}:"
            f"src{src_rows}"
        )

    def _ragged_prefill_logits_eager(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor,
        logit_positions: Tensor,
        context_len: int | None = None,
        src_prefix_row: Tensor | None = None,
        prefix_copy_len: int | None = None,
    ) -> Tensor | None:
        if not self._cache_supports_tensor_ragged_prefill():
            return None
        eager = getattr(self.model, "prefill_ragged_logits", None)
        if eager is None:
            return None
        return eager(
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=context_len,
            src_prefix_row=src_prefix_row,
            prefix_copy_len=prefix_copy_len,
        )

    def _prefill_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        suffix_lengths = [
            len(request.prompt) - prefix_hit_tokens
            for _index, request, prefix_hit_tokens, _reusable in group
        ]
        split_suffix_buckets = self._prefix_prefill_split_suffix_buckets_enabled()
        profile_suffix_split_candidates = (
            not split_suffix_buckets
            and self._prefix_prefill_split_suffix_buckets_profile_candidates_enabled()
        )
        if self.graph_prefill and suffix_lengths and (
            split_suffix_buckets or profile_suffix_split_candidates
        ):
            split_groups = self._prefix_prefill_suffix_bucket_split_groups(
                group,
                suffix_lengths,
                accept_enabled=split_suffix_buckets,
                policy_enabled=split_suffix_buckets or profile_suffix_split_candidates,
            )
            if split_groups is not None:
                active: list[_ActiveRequest] = []
                for suffix_group in split_groups:
                    active.extend(self._prefill_prefix_batch(suffix_group, step, events=events))
                return active
        if self.graph_prefill and suffix_lengths and max(suffix_lengths) > 0:
            batch_bucket = self._prefill_batch_bucket(len(group))
            split_batch = self._prefix_prefill_split_on_capture_skip_batch(batch_bucket)
            if 0 < split_batch < len(group):
                active: list[_ActiveRequest] = []
                for start in range(0, len(group), split_batch):
                    active.extend(
                        self._prefill_prefix_batch(
                            group[start : start + split_batch],
                            step,
                            events=events,
                        )
                    )
                return active
        if self.graph_prefill:
            graph_active = self._prefill_prefix_graph_batch(group, step, events=events)
            if graph_active is not None:
                return graph_active
        if events is not None and suffix_lengths and max(suffix_lengths) == 0:
            return self._prefill_exact_prefix_batch(group, step, events=events)
        if len(set(suffix_lengths)) > 1:
            padded_active = self._prefill_prefix_padded_suffix_batch(group, step, events=events)
            if padded_active is not None:
                return padded_active
            active: list[_ActiveRequest] = []
            by_suffix_len: dict[int, list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = defaultdict(list)
            for item, suffix_len in zip(group, suffix_lengths):
                by_suffix_len[suffix_len].append(item)
            for suffix_group in by_suffix_len.values():
                active.extend(self._prefill_prefix_batch(suffix_group, step, events=events))
            return active

        self.stats.prefill_prefix_reuse_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        suffixes = []
        reusable_logits = []
        self._copy_reusable_prefixes_to_rows(rows, group)
        for _row, (_original_index, request, prefix_hit_tokens, reusable) in zip(rows, group):
            suffixes.append(request.prompt[prefix_hit_tokens:])
            reusable_logits.append(reusable.logits)

        if suffixes and suffixes[0]:
            input_ids = torch.tensor(suffixes, device=self.device, dtype=torch.long)
            logits, _ = self._prefill_logits(input_ids, cache=self._cache_view(rows))
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
        else:
            if any(item is None for item in reusable_logits):
                raise RuntimeError("exact-prefix reuse requires cached logits")
            logits = torch.cat([item.to(self.device) for item in reusable_logits], dim=0)

        next_tokens = self._sample_logits_for_requests(
            logits[:, -1, :],
            [request for _original_index, request, _prefix_hit_tokens, _reusable in group],
        ).detach().cpu().tolist()
        active = []
        for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            next_token = int(next_tokens[row_index])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefix_prefill_split_suffix_buckets_enabled(self) -> bool:
        env_name = "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS"
        if env_name in os.environ:
            return env_flag(env_name, False)
        if not self._prefix_prefill_split_suffix_buckets_greedy_short_scope():
            return False
        return env_flag(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT",
            False,
        )

    def _prefix_prefill_split_suffix_buckets_profile_candidates_enabled(self) -> bool:
        env_name = "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_PROFILE_CANDIDATES"
        if env_name in os.environ:
            return env_flag(env_name, False)
        return bool(
            self.profile_timings
            and self._prefix_prefill_split_suffix_buckets_greedy_short_scope()
        )

    def _prefix_prefill_split_suffix_buckets_greedy_short_scope(self) -> bool:
        if self.temperature > 0.0 or self.max_generation_tokens is None:
            return False
        greedy_short_max_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT_MAX_TOKENS",
            128,
            minimum=1,
        )
        return 0 < int(self.max_generation_tokens) <= greedy_short_max_tokens

    def _record_prefix_prefill_suffix_split_candidate(
        self,
        *,
        group_size: int,
        base_bucket: int,
        base_model_tokens: int,
        split_model_tokens: int,
        by_suffix_bucket: Mapping[int, Sequence[object]],
        accepted: bool,
        reject_reason: str | None = None,
    ) -> None:
        saved_tokens = max(0, int(base_model_tokens) - int(split_model_tokens))
        self.stats.prefill_suffix_split_candidate_calls += 1
        self.stats.prefill_suffix_split_base_model_tokens += int(base_model_tokens)
        self.stats.prefill_suffix_split_candidate_model_tokens += int(split_model_tokens)
        self.stats.prefill_suffix_split_candidate_saved_tokens += saved_tokens
        shape_key = self._prefix_prefill_suffix_split_shape_key(
            group_size=group_size,
            base_bucket=base_bucket,
            by_suffix_bucket=by_suffix_bucket,
        )
        self._record_shape_count(
            self.stats.prefill_suffix_split_candidate_shape_counts,
            shape_key,
        )
        self._record_shape_total(
            self.stats.prefill_suffix_split_candidate_shape_saved_tokens,
            shape_key,
            saved_tokens,
        )
        if accepted:
            self.stats.prefill_suffix_split_accepted_calls += 1
            self.stats.prefill_suffix_split_accepted_base_model_tokens += int(base_model_tokens)
            self.stats.prefill_suffix_split_accepted_model_tokens += int(split_model_tokens)
            self.stats.prefill_suffix_split_accepted_saved_tokens += saved_tokens
            self.stats.prefill_suffix_split_accepted_fragments += len(by_suffix_bucket)
            self._record_shape_count(
                self.stats.prefill_suffix_split_accepted_shape_counts,
                shape_key,
            )
            self._record_shape_total(
                self.stats.prefill_suffix_split_accepted_shape_saved_tokens,
                shape_key,
                saved_tokens,
            )
            for suffix_bucket, items in by_suffix_bucket.items():
                self._record_shape_count(
                    self.stats.prefill_suffix_split_accepted_fragment_counts,
                    f"b{len(items)}:s{int(suffix_bucket)}",
                )
            return

        self.stats.prefill_suffix_split_rejected_calls += 1
        reason = str(reject_reason or "unknown")
        self.stats.prefill_suffix_split_reject_reason_counts[reason] = (
            self.stats.prefill_suffix_split_reject_reason_counts.get(reason, 0) + 1
        )

    @staticmethod
    def _prefix_prefill_suffix_split_shape_key(
        *,
        group_size: int,
        base_bucket: int,
        by_suffix_bucket: Mapping[int, Sequence[object]],
    ) -> str:
        parts = [
            f"b{len(items)}:s{int(suffix_bucket)}"
            for suffix_bucket, items in sorted(by_suffix_bucket.items())
        ]
        return f"base_b{int(group_size)}:s{int(base_bucket)}->" + "+".join(parts)

    def _prefix_prefill_suffix_bucket_split_groups(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        suffix_lengths: Sequence[int],
        *,
        accept_enabled: bool = True,
        policy_enabled: bool | None = None,
    ) -> list[list[tuple[int, ServingRequest, int, _ReusablePrefix]]] | None:
        if policy_enabled is None:
            policy_enabled = self._prefix_prefill_split_suffix_buckets_enabled()
        by_suffix_bucket: dict[int, list[tuple[int, ServingRequest, int, _ReusablePrefix]]] = defaultdict(list)
        for item, suffix_len in zip(group, suffix_lengths):
            by_suffix_bucket[self._suffix_bucket(max(1, suffix_len))].append(item)
        if len(by_suffix_bucket) <= 1:
            return None

        base_bucket = self._suffix_bucket(max(suffix_lengths))
        base_model_tokens = self._prefill_batch_bucket(len(group)) * base_bucket
        split_model_tokens = sum(
            self._prefill_batch_bucket(len(items)) * suffix_bucket
            for suffix_bucket, items in by_suffix_bucket.items()
        )
        if split_model_tokens >= base_model_tokens:
            self._record_prefix_prefill_suffix_split_candidate(
                group_size=len(group),
                base_bucket=base_bucket,
                base_model_tokens=base_model_tokens,
                split_model_tokens=split_model_tokens,
                by_suffix_bucket=by_suffix_bucket,
                accepted=False,
                reject_reason="no_savings",
            )
            return None
        min_savings_pct = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_MIN_SAVINGS_PCT",
            0,
            minimum=0,
        )
        if min_savings_pct > 0 and (
            (base_model_tokens - split_model_tokens) * 100 < base_model_tokens * min_savings_pct
        ):
            self._record_prefix_prefill_suffix_split_candidate(
                group_size=len(group),
                base_bucket=base_bucket,
                base_model_tokens=base_model_tokens,
                split_model_tokens=split_model_tokens,
                by_suffix_bucket=by_suffix_bucket,
                accepted=False,
                reject_reason="min_savings",
            )
            return None
        default_min_group_size = 1
        if (
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS" not in os.environ
            and policy_enabled
        ):
            default_min_group_size = 2
        min_group_size = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_MIN_GROUP",
            default_min_group_size,
            minimum=1,
        )
        default_min_fill_pct = 75 if policy_enabled else 0
        min_fill_pct = env_int(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_MIN_FILL_PCT",
            default_min_fill_pct,
            minimum=0,
        )
        for items in by_suffix_bucket.values():
            if len(items) < min_group_size:
                self._record_prefix_prefill_suffix_split_candidate(
                    group_size=len(group),
                    base_bucket=base_bucket,
                    base_model_tokens=base_model_tokens,
                    split_model_tokens=split_model_tokens,
                    by_suffix_bucket=by_suffix_bucket,
                    accepted=False,
                    reject_reason="min_group",
                )
                return None
            if min_fill_pct > 0 and (
                len(items) * 100 < self._prefill_batch_bucket(len(items)) * min_fill_pct
            ):
                self._record_prefix_prefill_suffix_split_candidate(
                    group_size=len(group),
                    base_bucket=base_bucket,
                    base_model_tokens=base_model_tokens,
                    split_model_tokens=split_model_tokens,
                    by_suffix_bucket=by_suffix_bucket,
                    accepted=False,
                    reject_reason="min_fill",
                )
                return None
        if not accept_enabled:
            self._record_prefix_prefill_suffix_split_candidate(
                group_size=len(group),
                base_bucket=base_bucket,
                base_model_tokens=base_model_tokens,
                split_model_tokens=split_model_tokens,
                by_suffix_bucket=by_suffix_bucket,
                accepted=False,
                reject_reason="disabled",
            )
            return None
        self._record_prefix_prefill_suffix_split_candidate(
            group_size=len(group),
            base_bucket=base_bucket,
            base_model_tokens=base_model_tokens,
            split_model_tokens=split_model_tokens,
            by_suffix_bucket=by_suffix_bucket,
            accepted=True,
        )
        return list(by_suffix_bucket.values())

    def _prefill_exact_prefix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        if not group:
            return []
        self.stats.prefill_prefix_reuse_batches += 1
        acquired_rows: list[int] = []
        try:
            next_tokens, logits_by_index = self._sample_exact_prefix_group(group)
            self._record_prefix_reuse_batch(
                (prefix_hit_tokens, reusable)
                for _original_index, _request, prefix_hit_tokens, reusable in group
            )

            active_items: list[tuple[_ActiveRequest, int, int, bool]] = []
            copy_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            active: list[_ActiveRequest] = []
            tokens_by_row = [list(request.prompt) for _original_index, request, _hit, _reusable in group]
            generated_by_row = [0 for _ in group]
            continuation_reuse: dict[int, _ReusablePrefix] = {}
            final_prefix_by_row: dict[int, tuple[_ReusablePrefix, int, bool]] = {}
            finished_rows: set[int] = set()
            last_event_index_by_row: dict[int, int] = {}

            def append_event(
                row_index: int,
                token: int,
                generated: int,
                *,
                finished: bool,
            ) -> None:
                if events is None:
                    return
                request = group[row_index][1]
                events.append(
                    ServingTokenEvent(
                        request_id=request.request_id,
                        token=int(token),
                        step=step,
                        generated=int(generated),
                        finished=bool(finished),
                    )
                )
                last_event_index_by_row[row_index] = len(events) - 1

            def finish_last_event(row_index: int) -> None:
                if events is None:
                    return
                event_index = last_event_index_by_row.get(row_index)
                if event_index is None:
                    return
                event = events[event_index]
                events[event_index] = ServingTokenEvent(
                    request_id=event.request_id,
                    token=event.token,
                    step=event.step,
                    generated=event.generated,
                    finished=True,
                )

            for row_index, (_original_index, request, _prefix_hit_tokens, _reusable) in enumerate(group):
                next_token = int(next_tokens[row_index])
                tokens_by_row[row_index].append(next_token)
                generated = 1
                generated_by_row[row_index] = generated
                finished = request.is_stop_token(next_token) or generated >= request.max_new_tokens
                append_event(row_index, next_token, generated, finished=finished)
                if finished:
                    finished_rows.add(row_index)
                    continue

                generated_prefix = tuple(tokens_by_row[row_index])
                continuation = self._lookup_exact_reusable_prefix(generated_prefix)
                if continuation is not None and continuation.logits is not None:
                    continuation_reuse[row_index] = continuation
                    continue
                final_prefix_by_row[row_index] = (_reusable, _prefix_hit_tokens, True)

            generated_reuse_entries: list[tuple[int, _ReusablePrefix | None]] = []

            max_generated_hops = env_int(
                "TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_REUSE_MAX_HOPS",
                8,
                minimum=1,
            )
            generated_hops = 0
            while continuation_reuse and generated_hops < max_generated_hops:
                continuation_reuse_groups: dict[tuple[Hashable, float], list[int]] = defaultdict(list)
                for row_index in sorted(continuation_reuse):
                    request = group[row_index][1]
                    continuation = continuation_reuse[row_index]
                    continuation_reuse_groups[
                        (continuation.route_id, self._request_temperature(request))
                    ].append(row_index)

                sampled_by_row: dict[int, tuple[int, _ReusablePrefix]] = {}
                for indices in continuation_reuse_groups.values():
                    continuation = continuation_reuse[indices[0]]
                    temperature = self._request_temperature(group[indices[0]][1])
                    sampled_tokens = self._sample_reusable_prefix_next_token_list(
                        continuation,
                        len(indices),
                        temperature=temperature,
                    )
                    for token_index, row_index in enumerate(indices):
                        sampled_by_row[row_index] = (
                            int(sampled_tokens[token_index]),
                            continuation,
                        )

                next_continuation_reuse: dict[int, _ReusablePrefix] = {}
                for row_index in range(len(group)):
                    sampled = sampled_by_row.get(row_index)
                    if sampled is None:
                        continue
                    next_token, continuation = sampled
                    original_index, request, _prefix_hit_tokens, _reusable = group[row_index]
                    del original_index, _prefix_hit_tokens, _reusable
                    generated_prefix_len = len(continuation.tokens)
                    generated_reuse_entries.append((generated_prefix_len, continuation))
                    self.stats.generated_prefix_reuse_requests += 1
                    self.stats.generated_prefix_reuse_tokens += generated_prefix_len

                    generated = generated_by_row[row_index] + 1
                    generated_by_row[row_index] = generated
                    if request.is_stop_token(next_token):
                        finish_last_event(row_index)
                        finished_rows.add(row_index)
                        continue

                    tokens_by_row[row_index].append(next_token)
                    reached_limit = generated >= request.max_new_tokens
                    append_event(row_index, next_token, generated, finished=reached_limit)
                    if reached_limit:
                        finished_rows.add(row_index)
                        continue

                    generated_prefix = tuple(tokens_by_row[row_index])
                    next_continuation = self._lookup_exact_reusable_prefix(generated_prefix)
                    if (
                        next_continuation is not None
                        and next_continuation.logits is not None
                        and generated_hops + 1 < max_generated_hops
                    ):
                        next_continuation_reuse[row_index] = next_continuation
                        continue
                    final_prefix_by_row[row_index] = (
                        continuation,
                        generated_prefix_len,
                        False,
                    )
                continuation_reuse = next_continuation_reuse
                generated_hops += 1

            for row_index, continuation in continuation_reuse.items():
                final_prefix_by_row[row_index] = (continuation, len(continuation.tokens), False)

            self._record_prefix_reuse_batch(generated_reuse_entries)
            for row_index in range(len(group)):
                if row_index in finished_rows:
                    continue
                source = final_prefix_by_row.get(row_index)
                if source is None:
                    continue
                source_reusable, source_tokens, should_store_prompt_logits = source
                original_index, request, _prefix_hit_tokens, _reusable = group[row_index]
                row = self._acquire_active_row()
                acquired_rows.append(row)
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=tokens_by_row[row_index],
                    generated=generated_by_row[row_index],
                    row=row,
                    last_token=tokens_by_row[row_index][-1],
                    seq_len=source_tokens,
                    prefix_hit_tokens=source_tokens,
                    started_step=step,
                )
                copy_groups[(source_reusable.row, source_tokens)].append(row)
                active_items.append((state, row_index, row, should_store_prompt_logits))

            for (source_row, prefix_hit_tokens), dest_rows in copy_groups.items():
                self._copy_prefix_to_rows(source_row, dest_rows, prefix_hit_tokens)
            for state, row_index, row, should_store_prompt_logits in active_items:
                state.seq_len = self._cache_row_seq_len(row, state.seq_len)
                if should_store_prompt_logits:
                    self._store_reusable_prefix(
                        state.request.request_id,
                        state.request.prompt,
                        row,
                        logits_by_index[row_index],
                        allow_pinned=self._allow_pinned_full_prompt_store(state.request),
                    )
                active.append(state)
            return active
        except Exception:
            for row in acquired_rows:
                self._release_active_row(row)
            raise

    def _sample_exact_prefix_group(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
    ) -> tuple[list[int], list[Tensor]]:
        by_route: dict[tuple[Hashable, float], list[int]] = defaultdict(list)
        for index, (_original_index, _request, _prefix_hit_tokens, reusable) in enumerate(group):
            by_route[(reusable.route_id, self._request_temperature(_request))].append(index)

        next_tokens = [0 for _ in group]
        logits_by_index: list[Tensor | None] = [None for _ in group]
        for indices in by_route.values():
            reusable = group[indices[0]][3]
            if reusable.logits is None:
                raise RuntimeError("exact-prefix sampling requires cached logits")
            sampled_tokens = self._sample_reusable_prefix_next_token_list(
                reusable,
                len(indices),
                temperature=self._request_temperature(group[indices[0]][1]),
            )
            for token_index, group_index in enumerate(indices):
                next_tokens[group_index] = int(sampled_tokens[token_index])
                logits_by_index[group_index] = reusable.logits

        if any(logits is None for logits in logits_by_index):
            raise RuntimeError("exact-prefix sampling did not produce logits for every request")
        return next_tokens, [logits for logits in logits_by_index if logits is not None]

    def _sample_reusable_prefix_next_token_list(
        self,
        reusable: _ReusablePrefix,
        batch_size: int,
        *,
        temperature: float | None = None,
    ) -> list[int]:
        if reusable.logits is None:
            raise RuntimeError("exact-prefix sampling requires cached logits")
        sampling_temperature = float(self.temperature if temperature is None else temperature)
        if sampling_temperature <= 0.0 and reusable.greedy_token is not None:
            sample_start_s = time.perf_counter() if self.profile_timings else 0.0
            tokens = [int(reusable.greedy_token)] * max(0, int(batch_size))
            if self.profile_timings:
                sample_elapsed_ms = (time.perf_counter() - sample_start_s) * 1000.0
                self.stats.prefill_sample_ms += sample_elapsed_ms
                self.stats.prefill_sample_select_ms += sample_elapsed_ms
            return tokens
        sampled = self._sample_reusable_prefix_next_tokens(
            reusable,
            batch_size,
            temperature=sampling_temperature,
        )
        return [int(token) for token in sampled.detach().cpu().tolist()]

    def _sample_reusable_prefix_next_tokens(
        self,
        reusable: _ReusablePrefix,
        batch_size: int,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        if reusable.logits is None:
            raise RuntimeError("exact-prefix sampling requires cached logits")
        sample_start_s = time.perf_counter() if self.profile_timings else 0.0
        sample_select_ms = 0.0
        try:
            logits = reusable.logits
            sampling_temperature = float(self.temperature if temperature is None else temperature)
            if sampling_temperature <= 0.0 and reusable.greedy_token is not None:
                result = torch.full(
                    (max(0, int(batch_size)),),
                    int(reusable.greedy_token),
                    dtype=torch.long,
                    device=self.device,
                )
                if self.profile_timings:
                    sample_select_ms = (time.perf_counter() - sample_start_s) * 1000.0
                return result
            if (
                sampling_temperature > 0.0
                and self._cached_repeated_sample_state_enabled
            ):
                prepare_state = self._prepare_repeated_sample_state
                sample_from_state = self._sample_repeated_from_state
                if callable(prepare_state) and callable(sample_from_state):
                    if reusable.sample_state is None or reusable.sample_temperature != sampling_temperature:
                        state_logits = logits[:, -1, :].to(self.device)
                        reusable.sample_state = prepare_state(state_logits, sampling_temperature)
                        reusable.sample_temperature = sampling_temperature
                        if reusable.sample_state is not None:
                            self.stats.repeated_sample_state_prepares += 1
                    if reusable.sample_state is not None:
                        sampled = sample_from_state(reusable.sample_state, batch_size, sampling_temperature)
                        if sampled is not None:
                            self.stats.repeated_sample_state_hits += 1
                            self.stats.repeated_sample_state_tokens += int(batch_size)
                            result = sampled.to(self.device)
                            if self.profile_timings:
                                sample_select_ms = (time.perf_counter() - sample_start_s) * 1000.0
                            return result
            result = self._sample_repeated_logits(
                logits[:, -1, :].to(self.device),
                batch_size,
                temperature=sampling_temperature,
            )
            if sampling_temperature <= 0.0 and result.numel() > 0:
                reusable.greedy_token = int(result.reshape(-1)[0].detach().cpu().item())
            if self.profile_timings:
                sample_select_ms = (time.perf_counter() - sample_start_s) * 1000.0
            return result
        finally:
            if self.profile_timings:
                sample_elapsed_ms = (time.perf_counter() - sample_start_s) * 1000.0
                self.stats.prefill_sample_ms += sample_elapsed_ms
                self.stats.prefill_sample_select_ms += sample_select_ms

    def _sample_repeated_logits(
        self,
        logits: Tensor,
        batch_size: int,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        sampling_temperature = float(self.temperature if temperature is None else temperature)
        if batch_size <= 1:
            return self._sample_logits_with_temperature(logits, sampling_temperature)
        sample_repeated = getattr(self.model, "sample_repeated_next_token", None)
        if callable(sample_repeated):
            return sample_repeated(logits, batch_size, sampling_temperature).to(self.device)
        expanded = logits.expand(batch_size, logits.size(-1)).contiguous()
        return self._sample_logits_with_temperature(expanded, sampling_temperature)

    def _prefill_prefix_padded_suffix_batch(
        self,
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", False):
            return None
        suffixes = [request.prompt[prefix_hit_tokens:] for _index, request, prefix_hit_tokens, _reusable in group]
        suffix_lengths = [len(suffix) for suffix in suffixes]
        if not suffix_lengths or min(suffix_lengths) <= 0 or len(set(suffix_lengths)) <= 1:
            return None
        max_suffix_len = max(suffix_lengths)
        padding_tokens = len(suffix_lengths) * max_suffix_len - sum(suffix_lengths)
        max_padding_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_MAX_PADDING_TOKENS",
            1024,
            minimum=0,
        )
        if padding_tokens > max_padding_tokens:
            return None

        rows = [self._acquire_active_row() for _ in group]
        try:
            self._copy_reusable_prefixes_to_rows(rows, group)

            padded_suffixes = [
                [*suffix, *([0] * (max_suffix_len - len(suffix)))]
                for suffix in suffixes
            ]
            input_ids = torch.tensor(padded_suffixes, device=self.device, dtype=torch.long)
            logit_positions = torch.tensor(
                [length - 1 for length in suffix_lengths],
                device=self.device,
                dtype=torch.long,
            )
            logits = self._forward_selected_logits(
                input_ids,
                cache=self._cache_view(rows),
                logit_positions=logit_positions,
            )
            if logits is None:
                for row in rows:
                    self._release_active_row(row)
                return None
            self._record_model_call("prefill", len(group), tokens=input_ids.numel())
            self.stats.prefill_prefix_reuse_batches += 1
            self.stats.prefill_padded_suffix_batches += 1
            next_tokens = self._sample_logits_for_requests(
                logits[:, -1, :],
                [request for _original_index, request, _prefix_hit_tokens, _reusable in group],
            ).detach().cpu().tolist()

            active: list[_ActiveRequest] = []
            for row_index, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
                row = rows[row_index]
                self._set_cache_row_seq_len(row, len(request.prompt))
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    logits[row_index : row_index + 1],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
                next_token = int(next_tokens[row_index])
                state = _ActiveRequest(
                    original_index=original_index,
                    request=request,
                    tokens=[*request.prompt, next_token],
                    generated=1,
                    row=row,
                    last_token=next_token,
                    seq_len=self._cache_row_seq_len(row, len(request.prompt)),
                    prefix_hit_tokens=prefix_hit_tokens,
                    started_step=step,
                )
                self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
                active.append(state)
            return active
        except Exception:
            for row in rows:
                self._release_active_row(row)
            raise

    def _copy_reusable_prefixes_to_rows(
        self,
        rows: list[int],
        group: list[tuple[int, ServingRequest, int, _ReusablePrefix]],
    ) -> None:
        copy_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        reuse_entries: list[tuple[int, _ReusablePrefix | None]] = []
        for row, (_original_index, _request, prefix_hit_tokens, reusable) in zip(rows, group):
            copy_groups[(reusable.row, prefix_hit_tokens)].append(row)
            reuse_entries.append((prefix_hit_tokens, reusable))
        self._record_prefix_reuse_batch(reuse_entries)
        for (source_row, prefix_hit_tokens), dest_rows in copy_groups.items():
            self._copy_prefix_to_rows(source_row, dest_rows, prefix_hit_tokens)

    def _prefill_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        self.stats.prefill_plain_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        prompts = torch.tensor([request.prompt for _, request, _ in group], device=self.device, dtype=torch.long)
        self._record_shape_count(self.stats.prefill_shape_counts, f"plain:b{len(group)}:t{prompts.size(1)}")
        cache_view = self._cache_view(rows)
        logits, _ = self._prefill_logits(prompts, cache=cache_view)
        self._record_model_call("prefill", len(group), tokens=prompts.numel())
        next_tokens = self._sample_logits_for_requests(
            logits[:, -1, :],
            [request for _original_index, request, _prefix_hit_tokens in group],
        ).detach().cpu().tolist()

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            next_token = int(next_tokens[row_index])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefill_ragged_graph_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest] | None:
        ragged_graph = getattr(self.model, "try_prefill_ragged_logits_graph", None)
        if ragged_graph is None:
            return None
        if not self._cache_supports_tensor_ragged_prefill():
            return None
        rows = [self._acquire_active_row() for _ in group]
        cache = self._require_cache()
        for row in rows:
            cache.clear_row(row)
            for layer_cache in cache.layers:
                keys = getattr(layer_cache, "keys", None)
                values = getattr(layer_cache, "values", None)
                if isinstance(keys, Tensor) and isinstance(values, Tensor):
                    keys[row].zero_()
                    values[row].zero_()
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        suffix_bucket = self._suffix_bucket(max_len)
        batch_bucket = self._prefill_batch_bucket(len(group))
        self._record_shape_count(
            self.stats.prefill_shape_counts,
            f"ragged_graph:b{batch_bucket}:t{suffix_bucket}",
        )
        padded = []
        for _, request, _ in group:
            prompt = list(request.prompt)
            prompt.extend([0] * (suffix_bucket - len(prompt)))
            padded.append(prompt)
        while len(padded) < batch_bucket:
            padded.append([0] * suffix_bucket)
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        row_indices_list = list(rows)
        pad_row = self._free_active_rows[-1] if self._free_active_rows else rows[0]
        while len(row_indices_list) < batch_bucket:
            row_indices_list.append(pad_row)
        row_indices = torch.tensor(row_indices_list, device=self.device, dtype=torch.long)
        seq_lens_list = [0] * (max(row_indices_list) + 1)
        seq_lens = torch.tensor(seq_lens_list, device=self.device, dtype=torch.long)
        logit_pos = [l - 1 for l in lengths]
        while len(logit_pos) < batch_bucket:
            logit_pos.append(0)
        logit_positions = torch.tensor(logit_pos, device=self.device, dtype=torch.long)
        try:
            logits = ragged_graph(
                input_ids, self._require_cache(),
                seq_lens=seq_lens, row_indices=row_indices,
                logit_positions=logit_positions,
                context_len=suffix_bucket,
            )
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None
        n = len(group)
        logits = logits[:n]
        self._record_model_call("prefill", n, tokens=sum(lengths))
        next_tokens = self._sample_logits_for_requests(
            logits[:, -1, :],
            [request for _original_index, request, _prefix_hit_tokens in group],
        ).detach().cpu().tolist()
        for i, (_, request, _) in enumerate(group):
            self._set_cache_row_seq_len(rows[i], len(request.prompt))
        active = []
        for i, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[i]
            seq_len = len(request.prompt)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[i:i+1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            next_token = int(next_tokens[i])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _can_padded_batch_prefill(self, group: list[tuple[int, ServingRequest, int]]) -> bool:
        # Off by default: it forces seq_len=len(prompt) and changes prefill
        # grouping, which differs from _prefill_batch for skewed-seq-len models.
        # The shared-prefix workloads we care about use the common-prefix path
        # (and cross-batch prefix pinning) instead. Available via env when a
        # workload has no shared prefix but similar suffix lengths.
        if not env_flag("TORCHINFERNO_CONTINUOUS_PADDED_BATCH_PREFILL", True):
            return False
        if not callable(getattr(self.model, "forward", None)):
            return False
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        min_len = min(lengths)
        return max_len > 0 and min_len >= max_len * 0.5

    def _prefill_padded_batch(
        self,
        group: list[tuple[int, ServingRequest, int]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest]:
        self.stats.prefill_plain_batches += 1
        rows = [self._acquire_active_row() for _ in group]
        lengths = [len(request.prompt) for _, request, _ in group]
        max_len = max(lengths)
        self._record_shape_count(
            self.stats.prefill_shape_counts,
            f"padded_plain:b{len(group)}:t{min(lengths)}-{max_len}",
        )
        padded = []
        logit_positions = []
        for _i, (_, request, _) in enumerate(group):
            prompt = list(request.prompt)
            logit_positions.append(len(prompt) - 1)
            prompt.extend([0] * (max_len - len(prompt)))
            padded.append(prompt)
        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        logit_pos_tensor = torch.tensor(logit_positions, device=self.device, dtype=torch.long)
        cache_view = self._cache_view(rows)
        selected_logits = self._forward_selected_logits(
            input_ids, cache=cache_view, logit_positions=logit_pos_tensor,
        )
        if selected_logits is not None:
            logits = selected_logits
        else:
            full_logits, _ = self._prefill_logits(input_ids, cache=cache_view)
            if full_logits.size(1) == 1:
                logits = full_logits
            else:
                logits = full_logits[
                    torch.arange(len(group), device=self.device),
                    torch.tensor([length - 1 for length in lengths], device=self.device),
                ].unsqueeze(1)
        self._record_model_call("prefill", len(group), tokens=input_ids.numel())
        next_tokens = self._sample_logits_for_requests(
            logits[:, -1, :],
            [request for _original_index, request, _prefix_hit_tokens in group],
        ).detach().cpu().tolist()

        for row_index, (_, request, _) in enumerate(group):
            self._set_cache_row_seq_len(rows[row_index], len(request.prompt))

        active = []
        for row_index, (original_index, request, prefix_hit_tokens) in enumerate(group):
            row = rows[row_index]
            seq_len = len(request.prompt)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[row_index : row_index + 1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            next_token = int(next_tokens[row_index])
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefill_one(
        self,
        original_index: int,
        request: ServingRequest,
        step: int,
        prefix_hit_tokens: int,
        reusable: _ReusablePrefix | None,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> _ActiveRequest:
        self.stats.prefill_single_batches += 1
        row = self._acquire_active_row()
        suffix = request.prompt
        logits: Tensor
        if reusable is not None and prefix_hit_tokens > 0:
            self._copy_prefix(reusable.row, row, prefix_hit_tokens)
            suffix = request.prompt[prefix_hit_tokens:]
            self._record_prefix_reuse(prefix_hit_tokens, reusable)

        if suffix:
            input_ids = torch.tensor([suffix], device=self.device, dtype=torch.long)
            self._record_shape_count(self.stats.prefill_shape_counts, f"single:b1:t{input_ids.size(1)}")
            logits, _ = self._prefill_logits(input_ids, cache=self._cache_view([row]))
            self._record_model_call("prefill", 1, tokens=input_ids.numel())
        elif reusable is not None and reusable.logits is not None:
            logits = reusable.logits.to(self.device)
        else:
            raise RuntimeError("empty prompt suffix without a reusable prefix")

        next_token_t = self._sample_logits_for_requests(logits[:, -1, :], [request])
        next_token = int(next_token_t.item())
        seq_len = self._refresh_row_seq_len_from_cache(row, len(request.prompt))
        self._store_reusable_prefix(
            request.request_id,
            request.prompt,
            row,
            logits,
            allow_pinned=self._allow_pinned_full_prompt_store(request),
        )
        state = _ActiveRequest(
            original_index=original_index,
            request=request,
            tokens=[*request.prompt, next_token],
            generated=1,
            row=row,
            last_token=next_token,
            seq_len=seq_len,
            prefix_hit_tokens=prefix_hit_tokens,
            started_step=step,
        )
        self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
        return state

    def _flush_pending_decode(self, events: list[ServingTokenEvent] | None = None) -> None:
        pending = getattr(self, "_pending_decode", None)
        if pending is not None:
            cpu_buf, event, states, step, pending_events = pending
            event.synchronize()
            tokens = cpu_buf[:len(states)].tolist()
            target_events = events if events is not None else pending_events
            self._finalize_decode(tokens, states, step, target_events)
            self._pending_decode = None

    def _decode_active(
        self,
        active: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        # GPU-resident decode runner (token stash + async D2H). Gated by
        # TORCHINFERNO_DECODE_GRAPH_RUNNER (default off; the runner is only built when on),
        # handed to the engine via _decode_runner. Correctness is validated (greedy output
        # is bit-identical to the baseline path on 8-rank TP -- sampling collectives already
        # sync the token across ranks, so the GPU->GPU feed is TP-safe). The path is still
        # SYNCHRONOUS (get_cpu_tokens harvests immediately after step), so it does NOT yet
        # win: shape-dependent in A/B -- helps prefill-heavy few_shot (TPOT 68->29) but
        # hurts decode-bound long_output (24->30) and tree (57->68). The actual win needs
        # PIPELINING (double-buffer the readback + lagged harvest so decode replays run
        # back-to-back and the .cpu() sync overlaps GPU compute); see docs/PERF_GAP_ANALYSIS.
        runner = getattr(self, "_decode_runner", None)
        if runner is not None and active:
            return self._decode_active_with_runner(runner, active, step, events=events)
        _p = self.profile_timings
        _t0 = time.perf_counter() if _p else 0.0
        indexed_results: list[tuple[int, ServingResult]] = []
        live: list[_ActiveRequest] = []
        for state in active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                indexed_results.append((state.original_index, self._finish_and_release(state, step)))
            else:
                live.append(state)
        if _p:
            self.stats._da_filter_ms = getattr(self.stats, '_da_filter_ms', 0.0) + (time.perf_counter() - _t0) * 1000.0

        _t1 = time.perf_counter() if _p else 0.0
        next_active: list[_ActiveRequest] = []
        groups = [live] if self._can_decode_ragged(live) else self._decode_groups(live)
        if _p:
            self.stats._da_group_ms = getattr(self.stats, '_da_group_ms', 0.0) + (time.perf_counter() - _t1) * 1000.0
        for group in groups:
            if self._can_decode_ragged(group):
                decoded = self._decode_ragged_batch(group, step, events=events)
            else:
                decoded = (
                    self._decode_batch(group, step, events=events)
                    if len(group) > 1
                    else [self._decode_one(group[0], step, events=events)]
                )
            _t2 = time.perf_counter() if _p else 0.0
            for item, state in zip(decoded, group):
                if isinstance(item, ServingResult):
                    indexed_results.append((state.original_index, item))
                else:
                    next_active.append(item)
            if _p:
                self.stats._da_collect_ms = getattr(self.stats, '_da_collect_ms', 0.0) + (time.perf_counter() - _t2) * 1000.0
        return indexed_results, next_active

    def _decode_active_with_runner(
        self,
        runner: object,
        active: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> tuple[list[tuple[int, ServingResult]], list[_ActiveRequest]]:
        indexed_results: list[tuple[int, ServingResult]] = []
        live: list[_ActiveRequest] = []
        for state in active:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            if self._should_finish_before_decode(state):
                indexed_results.append((state.original_index, self._finish_and_release(state, step)))
            else:
                live.append(state)
        if not live:
            return indexed_results, []
        rows = [s.row for s in live]
        if runner.n_active != len(live) or runner.active_rows != rows:
            runner.set_active(rows, [s.last_token for s in live], [s.seq_len for s in live])
        runner.step()
        tokens = runner.get_cpu_tokens()
        next_active: list[_ActiveRequest] = []
        any_finished = False
        for i, state in enumerate(live):
            tok = int(tokens[i]) if i < len(tokens) else 0
            state.tokens.append(tok)
            state.generated += 1
            state.last_token = tok
            state.seq_len += 1
            self._remember_row_seq_len(state.row, state.seq_len)
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, tok, step, finished=finished)
            if finished:
                self._finish_and_release(state, step)
                any_finished = True
            else:
                next_active.append(state)
        if any_finished:
            runner.set_active(
                [s.row for s in next_active],
                [s.last_token for s in next_active],
                [s.seq_len for s in next_active],
            )
        return indexed_results, next_active

    def _decode_groups(self, states: list[_ActiveRequest]) -> list[list[_ActiveRequest]]:
        grouped: dict[int, list[_ActiveRequest]] = defaultdict(list)
        for state in states:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            grouped[state.seq_len].append(state)
        return list(grouped.values())

    def _decode_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest | ServingResult]:
        rows = [state.row for state in states]
        if states:
            seq_len = states[0].seq_len
            if all(state.seq_len == seq_len for state in states):
                self._set_cache_rows_seq_len(rows, seq_len)
            else:
                for state in states:
                    self._set_cache_row_seq_len(state.row, state.seq_len)
        input_ids = torch.tensor([[state.last_token] for state in states], device=self.device, dtype=torch.long)
        reuse_logits: Tensor | None = None
        need_generated_prefix_logits = self._needs_generated_prefix_logits(states)
        if not hasattr(self, "_has_fi_decode"):
            self._has_fi_decode = bool(getattr(self.model, "_fi_decode_graphs", None))
        shared_temperature = self._shared_temperature_for_states(states)
        use_ragged_token_graph = shared_temperature is not None and not need_generated_prefix_logits
        use_static_token_graph = (
            shared_temperature is not None
            and shared_temperature <= 0.0
            and not need_generated_prefix_logits
        )
        if self._has_fi_decode:
            row_indices_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            seq_lens_t = self._seq_lens_tensor(states, rows=rows)
            if need_generated_prefix_logits:
                logits = self._ragged_decode_logits(input_ids, seq_lens_t, row_indices_t)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits_for_states(reuse_logits, states)
            else:
                self._last_ragged_decode_logits = None
                fi_token = (
                    self._try_ragged_token_graph(
                        input_ids,
                        seq_lens_t,
                        row_indices_t,
                        temperature=shared_temperature,
                    )
                    if use_ragged_token_graph
                    else None
                )
                if fi_token is not None:
                    next_token_tensor = fi_token.to(self.device)
                    reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
                    self.stats.decode_graph_hits += 1
                else:
                    cache_view = self._cache_view(rows)
                    logits = self._static_decode_logits(input_ids, cache_view)
                    reuse_logits = logits[:, -1, :]
                    next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], states)
        else:
            cache_view = self._cache_view(rows)
            graph_token = (
                self._try_static_token_graph(
                    input_ids,
                    cache_view,
                    temperature=shared_temperature,
                )
                if use_static_token_graph
                else None
            )
            if graph_token is None:
                logits = self._static_decode_logits(input_ids, cache_view)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], states)
            else:
                next_token_tensor = graph_token.to(self.device)
                self.stats.decode_graph_hits += 1
        self._record_model_call("decode", len(states), tokens=len(states))
        next_tokens = next_token_tensor.detach().cpu().tolist()
        self._store_decoded_reusable_prefixes(states, reuse_logits)

        decoded: list[_ActiveRequest | ServingResult] = []
        batched_next_seq_len: int | None = None
        if states:
            candidate_next_seq_len = states[0].seq_len + 1
            if all(state.seq_len + 1 == candidate_next_seq_len for state in states):
                self._set_cache_rows_seq_len(rows, candidate_next_seq_len)
                batched_next_seq_len = candidate_next_seq_len
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            if batched_next_seq_len is None:
                next_seq_len = state.seq_len + 1
                self._set_cache_row_seq_len(state.row, next_seq_len)
                state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
            else:
                state.seq_len = batched_next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        return decoded

    def _finalize_decode(
        self,
        next_tokens: list[int],
        states: list[_ActiveRequest],
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> None:
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
            self._remember_row_seq_len(state.row, next_seq_len)
            state.seq_len = next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                self._finish_and_release(state, step)

    def _decode_one(
        self,
        state: _ActiveRequest,
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> _ActiveRequest | ServingResult:
        self._set_cache_row_seq_len(state.row, state.seq_len)
        input_ids = torch.tensor([[state.last_token]], device=self.device, dtype=torch.long)
        reuse_logits: Tensor | None = None
        need_generated_prefix_logits = self._needs_generated_prefix_logits([state])
        if not hasattr(self, "_has_fi_decode"):
            self._has_fi_decode = bool(getattr(self.model, "_fi_decode_graphs", None))
        state_temperature = self._state_temperature(state)
        if self._has_fi_decode:
            row_indices_t = torch.tensor([state.row], dtype=torch.long, device=self.device)
            seq_lens_t = self._seq_lens_tensor([state], rows=[state.row])
            if need_generated_prefix_logits:
                logits = self._ragged_decode_logits(input_ids, seq_lens_t, row_indices_t)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits_for_states(reuse_logits, [state])
            else:
                self._last_ragged_decode_logits = None
                fi_token = self._try_ragged_token_graph(
                    input_ids,
                    seq_lens_t,
                    row_indices_t,
                    temperature=state_temperature,
                )
                if fi_token is not None:
                    next_token_tensor = fi_token.to(self.device)
                    reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
                    self.stats.decode_graph_hits += 1
                else:
                    cache_view = self._cache_view([state.row])
                    logits = self._static_decode_logits(input_ids, cache_view)
                    reuse_logits = logits[:, -1, :]
                    next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], [state])
        else:
            cache_view = self._cache_view([state.row])
            graph_token = (
                self._try_static_token_graph(
                    input_ids,
                    cache_view,
                    temperature=state_temperature,
                )
                if state_temperature <= 0.0 and not need_generated_prefix_logits
                else None
            )
            if graph_token is None:
                logits = self._static_decode_logits(input_ids, cache_view)
                reuse_logits = logits[:, -1, :]
                next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], [state])
            else:
                next_token_tensor = graph_token.to(self.device)
                self.stats.decode_graph_hits += 1
        self._record_model_call("decode", 1, tokens=1)
        next_token = int(next_token_tensor.item())
        self._store_decoded_reusable_prefixes([state], reuse_logits)
        state.tokens.append(next_token)
        state.generated += 1
        state.last_token = next_token
        next_seq_len = state.seq_len + 1
        self._set_cache_row_seq_len(state.row, next_seq_len)
        state.seq_len = self._cache_row_seq_len(state.row, next_seq_len)
        finished = self._should_finish_after_decode(state)
        self._record_token_event(events, state, next_token, step, finished=finished)
        if finished:
            return self._finish_and_release(state, step)
        return state

    def _start_prefill_graph_gpu_timer(self) -> tuple[object, object] | None:
        if not self.profile_timings or self.device.type != "cuda":
            return None
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
            return start, end
        except Exception:
            return None

    def _stop_prefill_graph_gpu_timer(
        self,
        events: tuple[object, object] | None,
        *,
        captured: bool,
        graph_shape_key: str,
        profile_shape_key: str | None = None,
    ) -> None:
        if events is None:
            return
        try:
            start, end = events
            end.record(torch.cuda.current_stream(self.device))
        except Exception:
            return
        self._pending_prefill_graph_events.append(
            (start, end, bool(captured), str(graph_shape_key), profile_shape_key)
        )

    def _flush_prefill_graph_gpu_timers(self) -> None:
        pending = getattr(self, "_pending_prefill_graph_events", [])
        if not pending:
            return
        remaining: list[tuple[object, ...]] = []
        for item in pending:
            start, end, captured, graph_shape_key, profile_shape_key = item
            try:
                elapsed_ms = float(start.elapsed_time(end))
            except RuntimeError:
                remaining.append(item)
                continue
            except Exception:
                continue
            if bool(captured):
                self.stats.prefill_graph_capture_gpu_ms += elapsed_ms
                self._record_shape_time(
                    self.stats.prefill_graph_capture_shape_gpu_ms,
                    str(graph_shape_key),
                    elapsed_ms,
                )
                if profile_shape_key is not None:
                    self._record_shape_time(
                        self.stats.prefill_shape_graph_capture_gpu_ms,
                        str(profile_shape_key),
                        elapsed_ms,
                    )
                continue
            self.stats.prefill_graph_replay_gpu_ms += elapsed_ms
            self._record_shape_time(
                self.stats.prefill_graph_replay_shape_gpu_ms,
                str(graph_shape_key),
                elapsed_ms,
            )
            if profile_shape_key is not None:
                self._record_shape_time(
                    self.stats.prefill_shape_graph_replay_gpu_ms,
                    str(profile_shape_key),
                    elapsed_ms,
                )
        self._pending_prefill_graph_events = remaining

    def _start_decode_ragged_model_gpu_timer(self) -> tuple[object, object] | None:
        if not self.profile_timings or self.device.type != "cuda":
            return None
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
            return start, end
        except Exception:
            return None

    def _decode_many_records_sync_model_timing(self) -> bool:
        return self.device.type != "cuda" or bool(self.decode_many_sync_model_timings)

    def _stop_decode_ragged_model_gpu_timer(
        self,
        events: tuple[object, object] | None,
        *,
        shape_key: str | None = None,
        profile_source: str | None = None,
    ) -> None:
        if events is None:
            return
        try:
            start, end = events
            end.record(torch.cuda.current_stream(self.device))
        except Exception:
            return
        self._pending_decode_ragged_model_events.append(
            (start, end, shape_key, profile_source, None)
        )

    def _decode_gpu_timer_event_fields(
        self,
        item: tuple[object, ...],
    ) -> tuple[object, object, str | None, str | None, dict[str, float] | None]:
        start, end = item[0], item[1]
        shape_key = item[2] if len(item) >= 3 and isinstance(item[2], str) else None
        profile_source = item[3] if len(item) >= 4 and isinstance(item[3], str) else None
        window_weights = item[4] if len(item) >= 5 and isinstance(item[4], dict) else None
        return start, end, shape_key, profile_source, window_weights

    def _attach_latest_decode_many_gpu_window(
        self,
        step_window_key: str | None,
        model_tokens: int,
        *,
        shape_key: str | None = None,
    ) -> None:
        if not step_window_key or model_tokens <= 0:
            return
        pending = getattr(self, "_pending_decode_ragged_model_events", [])
        if not pending:
            return
        start, end, pending_shape_key, profile_source, window_weights = (
            self._decode_gpu_timer_event_fields(pending[-1])
        )
        if profile_source != "decode_many":
            return
        if shape_key is not None and shape_key != pending_shape_key:
            return
        weights = dict(window_weights or {})
        weights[step_window_key] = weights.get(step_window_key, 0.0) + float(model_tokens)
        pending[-1] = (start, end, pending_shape_key, profile_source, weights)

    def _flush_decode_ragged_model_gpu_timers(self) -> None:
        pending = getattr(self, "_pending_decode_ragged_model_events", [])
        if not pending:
            return
        remaining: list[tuple[object, ...]] = []
        for item in pending:
            start, end, shape_key, profile_source, window_weights = (
                self._decode_gpu_timer_event_fields(item)
            )
            try:
                elapsed_ms = float(start.elapsed_time(end))
                self.stats.decode_ragged_model_gpu_ms += elapsed_ms
                if shape_key is not None:
                    self._record_shape_time(
                        self.stats.decode_shape_gpu_ms,
                        shape_key,
                        elapsed_ms,
                    )
                if profile_source == "decode_many":
                    self.stats.decode_many_model_gpu_ms += elapsed_ms
                    if shape_key is not None:
                        self._record_shape_time(
                            self.stats.decode_many_shape_gpu_ms,
                            shape_key,
                            elapsed_ms,
                        )
                    if window_weights:
                        total_weight = sum(
                            max(0.0, float(weight)) for weight in window_weights.values()
                        )
                        if total_weight > 0.0:
                            for step_window_key, weight in window_weights.items():
                                share = max(0.0, float(weight)) / total_weight
                                if share <= 0.0:
                                    continue
                                self._record_shape_time(
                                    self.stats.decode_many_step_window_model_ms,
                                    str(step_window_key),
                                    elapsed_ms * share,
                                )
            except RuntimeError:
                remaining.append((start, end, shape_key, profile_source, window_weights))
            except Exception:
                continue
        self._pending_decode_ragged_model_events = remaining

    def _ensure_gpu_token_buf(self) -> Tensor:
        buf = getattr(self, "_gpu_last_tokens", None)
        total = self.max_active_requests + (getattr(self, "prefix_cache_capacity", 0) or 0) + 2
        if buf is None or buf.size(0) < total:
            buf = torch.zeros(total, dtype=torch.long, device=self.device)
            self._gpu_last_tokens = buf
        return buf

    def _ensure_decode_many_token_scratch(self, tokens: int) -> Tensor:
        needed = max(1, int(tokens))
        scratch = getattr(self, "_decode_many_token_scratch", None)
        if scratch is None or scratch.numel() < needed or scratch.device != self.device:
            scratch = torch.empty(needed, dtype=torch.long, device=self.device)
            self._decode_many_token_scratch = scratch
        return scratch

    def _decode_many_async_readback_enabled(self) -> bool:
        return bool(
            self.decode_many_async_readback
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def _ensure_decode_many_cpu_token_scratch(self, tokens: int) -> Tensor | None:
        needed = max(1, int(tokens))
        scratch = getattr(self, "_decode_many_cpu_token_scratch", None)
        if scratch is None or scratch.numel() < needed:
            try:
                scratch = torch.empty(needed, dtype=torch.long, device="cpu", pin_memory=True)
            except Exception as exc:
                warn_optional_failure("continuous.decode_many_async_readback", exc)
                self.decode_many_async_readback = False
                return None
            self._decode_many_cpu_token_scratch = scratch
        return scratch

    def _decode_many_readback_stream(self) -> torch.cuda.Stream | None:
        if not self._decode_many_async_readback_enabled():
            return None
        stream = getattr(self, "_decode_many_readback_cuda_stream", None)
        if stream is None:
            try:
                stream = torch.cuda.Stream(device=self.device)
            except Exception as exc:
                warn_optional_failure("continuous.decode_many_async_readback_stream", exc)
                self.decode_many_async_readback = False
                return None
            self._decode_many_readback_cuda_stream = stream
        return stream

    def _maybe_schedule_decode_many_readback(
        self,
        src: Tensor,
        start: int,
        end: int,
    ) -> None:
        if not self._decode_many_async_readback_enabled():
            return
        stream = self._decode_many_readback_stream()
        if stream is None:
            return
        cpu_scratch = self._ensure_decode_many_cpu_token_scratch(end)
        if cpu_scratch is None:
            return
        try:
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                cpu_scratch[start:end].copy_(src.detach(), non_blocking=True)
        except Exception as exc:
            warn_optional_failure("continuous.decode_many_async_readback_copy", exc)
            try:
                stream.synchronize()
            except Exception:
                pass
            self.decode_many_async_readback = False

    def _decode_many_tokens_to_list(self, token_scratch: Tensor, token_count: int) -> list[int]:
        stream = (
            self._decode_many_readback_stream()
            if self._decode_many_async_readback_enabled()
            else None
        )
        cpu_scratch = getattr(self, "_decode_many_cpu_token_scratch", None)
        if stream is not None and cpu_scratch is not None and cpu_scratch.numel() >= token_count:
            stream.synchronize()
            return cpu_scratch[:token_count].tolist()
        stale_stream = getattr(self, "_decode_many_readback_cuda_stream", None)
        if stale_stream is not None:
            try:
                stale_stream.synchronize()
            except Exception:
                pass
        return token_scratch[:token_count].detach().cpu().tolist()

    def _decode_many_tokens_to_list_profiled(
        self,
        token_scratch: Tensor,
        token_count: int,
    ) -> tuple[list[int], float, float]:
        stream = (
            self._decode_many_readback_stream()
            if self._decode_many_async_readback_enabled()
            else None
        )
        cpu_scratch = getattr(self, "_decode_many_cpu_token_scratch", None)
        if stream is not None and cpu_scratch is not None and cpu_scratch.numel() >= token_count:
            wait_start_s = time.perf_counter()
            stream.synchronize()
            wait_ms = (time.perf_counter() - wait_start_s) * 1000.0
            materialize_start_s = time.perf_counter()
            tokens = cpu_scratch[:token_count].tolist()
            materialize_ms = (time.perf_counter() - materialize_start_s) * 1000.0
            return tokens, wait_ms, materialize_ms

        stale_stream = getattr(self, "_decode_many_readback_cuda_stream", None)
        wait_ms = 0.0
        if stale_stream is not None:
            try:
                wait_start_s = time.perf_counter()
                stale_stream.synchronize()
                wait_ms += (time.perf_counter() - wait_start_s) * 1000.0
            except Exception:
                pass
        tokens, tensor_wait_ms, materialize_ms = self._token_tensor_to_list_profiled(
            token_scratch[:token_count]
        )
        return tokens, wait_ms + tensor_wait_ms, materialize_ms

    def _token_tensor_to_list_profiled(self, token_tensor: Tensor) -> tuple[list[int], float, float]:
        wait_ms = 0.0
        if token_tensor.device.type == "cuda" and torch.cuda.is_available():
            try:
                wait_start_s = time.perf_counter()
                torch.cuda.current_stream(token_tensor.device).synchronize()
                wait_ms = (time.perf_counter() - wait_start_s) * 1000.0
            except Exception:
                wait_ms = 0.0
        materialize_start_s = time.perf_counter()
        tokens = token_tensor.detach().cpu().tolist()
        materialize_ms = (time.perf_counter() - materialize_start_s) * 1000.0
        return tokens, wait_ms, materialize_ms

    def _device_index_tensor(self, values: tuple[int, ...]) -> Tensor:
        cached = self._device_index_tensors.get(values)
        if cached is None:
            cached = torch.tensor(values, device=self.device, dtype=torch.long)
            self._device_index_tensors[values] = cached
        return cached

    def _sync_gpu_last_tokens_from_states(self, states: list[_ActiveRequest]) -> None:
        if not states:
            return
        rows = self._device_index_tensor(tuple(state.row for state in states))
        tokens = torch.tensor([state.last_token for state in states], device=self.device, dtype=torch.long)
        self._ensure_gpu_token_buf().index_copy_(0, rows, tokens)

    def _ensure_gpu_seq_lens_buf(self) -> Tensor:
        total = max(1, len(self._row_seq_lens))
        buf = getattr(self, "_gpu_seq_lens", None)
        if buf is None or buf.numel() < total:
            buf = torch.zeros(total, dtype=torch.long, device=self.device)
            self._gpu_seq_lens = buf
        return buf

    def _sync_gpu_seq_lens_from_states(self, states: list[_ActiveRequest]) -> None:
        if not states:
            return
        buf = self._ensure_gpu_seq_lens_buf()
        rows = self._device_index_tensor(tuple(state.row for state in states))
        seq_lens = torch.tensor([state.seq_len for state in states], device=self.device, dtype=torch.long)
        buf.index_copy_(0, rows, seq_lens)

    @staticmethod
    def _make_decode_many_gpu_state_signature(
        states: list[_ActiveRequest],
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple((state.row, int(state.last_token), int(state.seq_len)) for state in states)

    def _decode_many_gpu_state_is_current(
        self,
        signature: tuple[tuple[int, int, int], ...],
    ) -> bool:
        if getattr(self, "_decode_many_gpu_state_signature", None) != signature:
            return False
        if not signature:
            return False
        max_row = max(row for row, _last_token, _seq_len in signature)
        last_tokens = getattr(self, "_gpu_last_tokens", None)
        seq_lens = getattr(self, "_gpu_seq_lens", None)
        return (
            last_tokens is not None
            and seq_lens is not None
            and last_tokens.device == self.device
            and seq_lens.device == self.device
            and last_tokens.numel() > max_row
            and seq_lens.numel() > max_row
        )

    def _decode_many_seq_lens_tensor(self, states: list[_ActiveRequest], rows: list[int]) -> Tensor:
        buf = self._ensure_gpu_seq_lens_buf()
        if not states:
            return buf
        if len(rows) == len(states) and all(row == state.row for row, state in zip(rows, states)):
            return buf
        active_rows = {state.row for state in states}
        pad_seq_len = max(state.seq_len for state in states)
        pad_rows = tuple(
            row
            for row in rows
            if row not in active_rows and 0 <= row < len(self._row_seq_lens) and self._row_seq_lens[row] <= 0
        )
        if pad_rows:
            buf.index_fill_(0, self._device_index_tensor(pad_rows), int(pad_seq_len))
        return buf

    def _decode_many_step_window_key(self, generated_after: list[int], shape_key: str) -> str:
        window = max(1, int(self._decode_many_profile_step_window))
        if not generated_after:
            return f"{shape_key}:g?"
        first_generated = max(1, min(int(value) for value in generated_after))
        start = ((first_generated - 1) // window) * window + 1
        end = start + window - 1
        return f"{shape_key}:g{start}-{end}"

    def _advance_gpu_seq_lens(self, rows: Tensor, *, amount: int = 1) -> None:
        buf = getattr(self, "_gpu_seq_lens", None)
        if buf is None or rows.numel() == 0:
            return
        increment = torch.full_like(rows, max(1, int(amount)), dtype=torch.long)
        buf.index_add_(0, rows, increment)

    def _decode_ragged_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None = None,
    ) -> list[_ActiveRequest | ServingResult]:
        prompt_lookup_decoded = self._try_prompt_lookup_decode_batch(
            states,
            step,
            events=events,
        )
        if prompt_lookup_decoded is not None:
            return prompt_lookup_decoded
        return self._decode_ragged_batch_baseline(states, step, events=events)

    def _try_prompt_lookup_decode_batch(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult] | None:
        if not self._prompt_lookup_decode_enabled() or not states:
            return None
        if any(self._state_temperature(state) > 0.0 for state in states):
            return None

        proposals: list[tuple[int, ...]] = [
            self._prompt_lookup_proposal(state)
            for state in states
        ]
        candidate_indices = [
            index for index, proposal in enumerate(proposals)
            if proposal
        ]
        if not candidate_indices:
            return None

        decoded: list[_ActiveRequest | ServingResult | None] = [None for _ in states]
        for group_indices in self._prompt_lookup_groups(states, proposals, candidate_indices).values():
            if not group_indices:
                continue
            group_states = [states[index] for index in group_indices]
            group_proposals = [proposals[index] for index in group_indices]
            group_decoded = self._decode_prompt_lookup_group(
                group_states,
                group_proposals,
                step,
                events=events,
            )
            for index, item in zip(group_indices, group_decoded):
                decoded[index] = item

        remaining = [
            states[index]
            for index, item in enumerate(decoded)
            if item is None
        ]
        if remaining:
            fallback = self._decode_ragged_batch_baseline(remaining, step, events=events)
            fallback_iter = iter(fallback)
            for index, item in enumerate(decoded):
                if item is None:
                    decoded[index] = next(fallback_iter)

        return [item for item in decoded if item is not None]

    def _prompt_lookup_decode_enabled(self) -> bool:
        if self.temperature > 0.0:
            return False
        if not env_flag("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_DECODE", False):
            return False
        cache = self._require_cache()
        if getattr(cache, "cache_backend", self.cache_backend) != "dense":
            return False
        return True

    def _prompt_lookup_proposal(self, state: _ActiveRequest) -> tuple[int, ...]:
        min_max_new_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MIN_MAX_TOKENS",
            16,
            minimum=1,
        )
        if state.request.max_new_tokens < min_max_new_tokens:
            return ()
        remaining = state.request.max_new_tokens - state.generated
        if remaining <= 1:
            return ()
        ngram = env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_NGRAM", 3, minimum=1)
        history = state.tokens
        if len(history) <= ngram:
            return ()
        max_proposal = min(
            env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MAX_PROPOSAL_TOKENS", 8, minimum=1),
            remaining - 1,
        )
        if max_proposal <= 0:
            return ()
        min_proposal = min(
            env_int("TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_MIN_PROPOSAL_TOKENS", 4, minimum=1),
            max_proposal,
        )
        needle = tuple(history[-ngram:])
        for start in range(len(history) - ngram - 1, -1, -1):
            if tuple(history[start : start + ngram]) != needle:
                continue
            proposal = tuple(history[start + ngram : start + ngram + max_proposal])
            if len(proposal) >= min_proposal:
                return proposal
        return ()

    @staticmethod
    def _prompt_lookup_groups(
        states: list[_ActiveRequest],
        proposals: list[tuple[int, ...]],
        candidate_indices: list[int],
    ) -> dict[tuple[int, int], list[int]]:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in candidate_indices:
            groups[(states[index].seq_len, len(proposals[index]))].append(index)
        return groups

    def _decode_prompt_lookup_group(
        self,
        states: list[_ActiveRequest],
        proposals: list[tuple[int, ...]],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        if not states:
            return []
        proposal_len = len(proposals[0])
        shape_key = (
            f"prompt_lookup:b{len(states)}:proposal{proposal_len}"
            if self.profile_timings
            else None
        )
        input_ids = torch.tensor(
            [
                [state.last_token, *proposal]
                for state, proposal in zip(states, proposals)
            ],
            device=self.device,
            dtype=torch.long,
        )
        rows = [state.row for state in states]
        for state in states:
            self._set_cache_row_seq_len(state.row, state.seq_len)

        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        logits, _ = self._prefill_full_logits(input_ids, cache=self._cache_view(rows))
        self._stop_decode_ragged_model_gpu_timer(gpu_model_events, shape_key=shape_key)
        if self.profile_timings:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_elapsed_ms = (time.perf_counter() - model_start_s) * 1000.0
            self.stats.decode_ragged_model_ms += model_elapsed_ms
            if shape_key is not None:
                self._record_shape_time(
                    self.stats.decode_shape_model_ms,
                    shape_key,
                    model_elapsed_ms,
                )

        if logits.ndim != 3 or logits.size(1) < proposal_len + 1:
            for state in states:
                self._set_cache_row_seq_len(state.row, state.seq_len)
            return self._decode_ragged_batch_baseline(states, step, events=events)

        self._record_model_call(
            "decode",
            len(states),
            tokens=len(states) * (proposal_len + 1),
            ragged=True,
            active_tokens=len(states) * (proposal_len + 1),
        )
        if shape_key is not None:
            self._record_shape_count(self.stats.decode_shape_counts, shape_key)
        self.stats.prompt_lookup_batches += 1
        self.stats.prompt_lookup_requests += len(states)
        self.stats.prompt_lookup_proposed_tokens += sum(len(proposal) for proposal in proposals)

        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        flat_logits = logits[:, : proposal_len + 1, :].reshape(-1, logits.size(-1))
        predicted = self._sample_logits_for_temperatures(
            flat_logits,
            [
                self._state_temperature(state)
                for state in states
                for _ in range(proposal_len + 1)
            ],
        ).view(len(states), proposal_len + 1)
        predicted_tokens = predicted.detach().cpu().tolist()
        if self.profile_timings:
            cpu_elapsed_ms = (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self.stats.decode_ragged_cpu_tokens_ms += cpu_elapsed_ms
            if shape_key is not None:
                self._record_shape_time(
                    self.stats.decode_shape_cpu_tokens_ms,
                    shape_key,
                    cpu_elapsed_ms,
                )
            self._flush_decode_ragged_model_gpu_timers()

        decoded: list[_ActiveRequest | ServingResult] = []
        state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
        for state, proposal, row_predictions in zip(states, proposals, predicted_tokens):
            old_seq_len = state.seq_len
            emitted = [int(row_predictions[0])]
            for index, proposed_token in enumerate(proposal):
                if int(proposed_token) != int(row_predictions[index]):
                    break
                emitted.append(int(row_predictions[index + 1]))
            remaining = state.request.max_new_tokens - state.generated
            emitted = emitted[:remaining]

            finished_result: ServingResult | None = None
            emitted_count = 0
            for token in emitted:
                state.tokens.append(int(token))
                state.generated += 1
                state.last_token = int(token)
                emitted_count += 1
                finished = self._should_finish_after_decode(state)
                self._record_token_event(events, state, int(token), step, finished=finished)
                if finished:
                    self._set_cache_row_seq_len(state.row, old_seq_len + emitted_count)
                    state.seq_len = old_seq_len + emitted_count
                    finished_result = self._finish_and_release(state, step)
                    break

            state.seq_len = old_seq_len + emitted_count
            self.stats.prompt_lookup_accepted_tokens += max(0, emitted_count - 1)
            if finished_result is not None:
                decoded.append(finished_result)
                continue
            self._set_cache_row_seq_len(state.row, state.seq_len)
            decoded.append(state)
        if self.profile_timings:
            self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
        return decoded

    def _decode_ragged_batch_baseline(
        self,
        states: list[_ActiveRequest],
        step: int,
        *,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        prepare_start_s = time.perf_counter() if self.profile_timings else 0.0
        rows = [state.row for state in states]
        decode_rows = self._ragged_decode_bucket_rows(rows)
        n_active = len(states)
        n_padded = len(decode_rows)
        shape_key = f"ragged:b{n_active}/{n_padded}" if self.profile_timings else None
        if shape_key is not None:
            self._record_shape_count(self.stats.decode_shape_counts, shape_key)
        need_generated_prefix_logits = self._needs_generated_prefix_logits(states)
        shared_temperature = self._shared_temperature_for_states(states)
        # Same-temperature sampled batches can use the contiguous graph too;
        # token/logit outputs are reordered back to active-state order below.
        contiguous_row_set = (
            shared_temperature is not None
            and n_active == n_padded
            and sorted(decode_rows) == list(range(n_padded))
        )
        state_order_indices: Tensor | None = None
        row_indices: Tensor | None
        if contiguous_row_set:
            input_tokens = [0 for _ in range(n_padded)]
            for state in states:
                input_tokens[state.row] = state.last_token
            input_ids = torch.tensor([[token] for token in input_tokens], device=self.device, dtype=torch.long)
            row_indices = None
            if any(row != index for index, row in enumerate(rows)):
                state_order_indices = self._device_index_tensor(tuple(rows))
        else:
            pad_token = states[0].last_token
            input_tokens = [
                states[index].last_token if index < n_active else pad_token
                for index, _row in enumerate(decode_rows)
            ]
            input_ids = torch.tensor([[token] for token in input_tokens], device=self.device, dtype=torch.long)
            row_indices = torch.tensor(decode_rows, dtype=torch.long, device=self.device)
        seq_lens = self._seq_lens_tensor(states, rows=decode_rows)
        if self.profile_timings:
            self.stats.decode_ragged_prepare_ms += (time.perf_counter() - prepare_start_s) * 1000.0
        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        self._last_ragged_decode_logits = None
        reuse_logits: Tensor | None = None
        graph_token = (
            self._try_ragged_token_graph(
                input_ids,
                seq_lens,
                row_indices,
                temperature=shared_temperature,
            )
            if not need_generated_prefix_logits and shared_temperature is not None
            else None
        )
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            reuse_logits = getattr(self, "_last_ragged_decode_logits", None)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._ragged_decode_logits(input_ids, seq_lens, row_indices)
            reuse_logits = logits[:, -1, :]
            next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], states)
        if row_indices is None and state_order_indices is not None:
            next_token_tensor = next_token_tensor.index_select(0, state_order_indices)
            if reuse_logits is not None:
                reuse_logits = reuse_logits.index_select(0, state_order_indices)
        self._stop_decode_ragged_model_gpu_timer(gpu_model_events, shape_key=shape_key)
        self._record_model_call("decode", n_padded, tokens=n_padded, ragged=True, active_tokens=n_active)
        if self.profile_timings:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_elapsed_ms = (time.perf_counter() - model_start_s) * 1000.0
            self.stats.decode_ragged_model_ms += model_elapsed_ms
            if shape_key is not None:
                self._record_shape_time(
                    self.stats.decode_shape_model_ms,
                    shape_key,
                    model_elapsed_ms,
                )
        cpu_tokens_start_s = time.perf_counter() if self.profile_timings else 0.0
        next_tokens = next_token_tensor[:n_active].detach().cpu().tolist()
        if self.profile_timings:
            cpu_elapsed_ms = (time.perf_counter() - cpu_tokens_start_s) * 1000.0
            self.stats.decode_ragged_cpu_tokens_ms += cpu_elapsed_ms
            if shape_key is not None:
                self._record_shape_time(
                    self.stats.decode_shape_cpu_tokens_ms,
                    shape_key,
                    cpu_elapsed_ms,
                )
            self._flush_decode_ragged_model_gpu_timers()
        self._store_decoded_reusable_prefixes(
            states,
            None if reuse_logits is None else reuse_logits[:n_active],
        )
        return self._apply_decoded_tokens(states, next_tokens, step, events)

    def _decode_ragged_batch_token_tensor(
        self,
        states: list[_ActiveRequest],
        *,
        profile_source: str | None = None,
    ) -> Tensor:
        prepare_start_s = time.perf_counter() if self.profile_timings else 0.0
        record_sync_model_timing = (
            profile_source != "decode_many"
            or self._decode_many_records_sync_model_timing()
        )
        rows = [state.row for state in states]
        decode_rows = self._ragged_decode_bucket_rows(rows)
        n_active = len(states)
        n_padded = len(decode_rows)
        shape_key = f"decode_many:b{n_active}/{n_padded}" if self.profile_timings else None
        if shape_key is not None:
            self._record_shape_count(self.stats.decode_shape_counts, shape_key)
        shared_temperature = self._shared_temperature_for_states(states)
        # Same-temperature sampled batches can use the contiguous graph too;
        # token outputs are reordered back to active-state order below.
        contiguous_row_set = (
            shared_temperature is not None
            and n_active == n_padded
            and sorted(decode_rows) == list(range(n_padded))
        )
        state_order_indices: Tensor | None = None
        row_indices: Tensor | None
        if contiguous_row_set:
            row_indices = None
            input_ids = self._ensure_gpu_token_buf()[:n_padded].view(n_padded, 1)
            if any(row != index for index, row in enumerate(rows)):
                state_order_indices = self._device_index_tensor(tuple(rows))
        else:
            row_indices = self._device_index_tensor(tuple(decode_rows))
            input_ids = self._ensure_gpu_token_buf().index_select(0, row_indices).view(n_padded, 1)
        seq_lens = self._decode_many_seq_lens_tensor(states, decode_rows)
        if self.profile_timings:
            self.stats.decode_ragged_prepare_ms += (time.perf_counter() - prepare_start_s) * 1000.0

        model_start_s = time.perf_counter() if self.profile_timings else 0.0
        gpu_model_events = self._start_decode_ragged_model_gpu_timer()
        self._last_ragged_decode_logits = None
        graph_token = (
            self._try_ragged_token_graph(
                input_ids,
                seq_lens,
                row_indices,
                temperature=shared_temperature,
            )
            if shared_temperature is not None
            else None
        )
        if graph_token is not None:
            next_token_tensor = graph_token.to(self.device)
            self.stats.decode_graph_hits += 1
        else:
            logits = self._ragged_decode_logits(input_ids, seq_lens, row_indices)
            next_token_tensor = self._sample_logits_for_states(logits[:, -1, :], states)
        self._stop_decode_ragged_model_gpu_timer(
            gpu_model_events,
            shape_key=shape_key,
            profile_source=profile_source,
        )
        self._record_model_call("decode", n_padded, tokens=n_padded, ragged=True, active_tokens=n_active)
        if row_indices is None:
            self._ensure_gpu_token_buf()[:n_active].copy_(next_token_tensor[:n_active])
            seq_lens_buf = getattr(self, "_gpu_seq_lens", None)
            if seq_lens_buf is not None:
                seq_lens_buf[:n_active].add_(1)
            if state_order_indices is not None:
                next_token_tensor = next_token_tensor.index_select(0, state_order_indices)
        else:
            active_row_indices = row_indices[:n_active]
            self._ensure_gpu_token_buf().index_copy_(0, active_row_indices, next_token_tensor[:n_active])
            self._advance_gpu_seq_lens(active_row_indices)
        if self.profile_timings and record_sync_model_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_elapsed_ms = (time.perf_counter() - model_start_s) * 1000.0
            self.stats.decode_ragged_model_ms += model_elapsed_ms
            if shape_key is not None:
                self._record_shape_time(
                    self.stats.decode_shape_model_ms,
                    shape_key,
                    model_elapsed_ms,
                )
        return next_token_tensor

    def _apply_decoded_tokens(
        self,
        states: list[_ActiveRequest],
        next_tokens: list[int],
        step: int,
        events: list[ServingTokenEvent] | None,
    ) -> list[_ActiveRequest | ServingResult]:
        state_update_start_s = time.perf_counter() if self.profile_timings else 0.0
        decoded: list[_ActiveRequest | ServingResult] = []
        for row_index, state in enumerate(states):
            next_token = int(next_tokens[row_index])
            state.tokens.append(next_token)
            state.generated += 1
            state.last_token = next_token
            next_seq_len = state.seq_len + 1
            self._remember_row_seq_len(state.row, next_seq_len)
            state.seq_len = next_seq_len
            finished = self._should_finish_after_decode(state)
            self._record_token_event(events, state, next_token, step, finished=finished)
            if finished:
                decoded.append(self._finish_and_release(state, step))
            else:
                decoded.append(state)
        if self.profile_timings:
            self.stats.decode_ragged_state_update_ms += (time.perf_counter() - state_update_start_s) * 1000.0
        return decoded

    def _ragged_decode_bucket_rows(self, rows: list[int]) -> list[int]:
        if not self._ragged_decode_buckets_enabled:
            return rows
        active_count = len(rows)
        if active_count <= 1:
            return rows
        capacity = min(self.max_active_requests, max(1, int(self._ragged_decode_bucket_capacity)))
        if active_count >= capacity:
            return rows
        bucket_size = min(capacity, 1 << (active_count - 1).bit_length())
        if bucket_size <= active_count:
            return rows
        row_set = set(rows)
        bucketed = list(rows)
        for row in reversed(self._free_active_rows):
            if row in row_set:
                continue
            bucketed.append(row)
            if len(bucketed) >= bucket_size:
                return bucketed
        return rows

    def _finish_and_release(self, state: _ActiveRequest, step: int) -> ServingResult:
        row_adopted = self._store_finished_reusable_prefix(state)
        if not row_adopted:
            row_adopted = self._store_delayed_pinned_full_prompt_prefix(state)
        result = ServingResult(
            state.request.request_id,
            tuple(state.tokens),
            state.prefix_hit_tokens,
            state.request.arrival_step,
            state.started_step,
            step,
        )
        if not row_adopted:
            if self._warm_row_prefix_copy_skip_enabled():
                self._remember_row_cached_prefix(state.row, state.tokens)
            self._release_active_row(state.row)
        return result

    def _allow_pinned_full_prompt_store(self, request: ServingRequest) -> bool:
        env_name = "TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_MIN_MAX_TOKENS"
        if env_name not in os.environ and self._greedy_large_mixed_prefix_reuse_enabled():
            return request.max_new_tokens >= int(self.max_generation_tokens or 0)
        threshold = env_int(
            env_name,
            0,
            minimum=0,
        )
        return threshold > 0 and request.max_new_tokens >= threshold

    def _delayed_pinned_full_prompt_store_enabled(self, request: ServingRequest) -> bool:
        if not self._delayed_pinned_full_prompt_store_allowed(
            allow_pinned=self._allow_pinned_full_prompt_store(request)
        ):
            return False
        return True

    def _delayed_pinned_full_prompt_store_allowed(self, *, allow_pinned: bool) -> bool:
        if not self.pin_shared_prefix or not allow_pinned:
            return False
        if not env_flag("TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_ADOPT_ON_FINISH", True):
            return False
        if env_flag("TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_LOGITS", False):
            return False
        if env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS", False):
            return False
        return True

    def _store_reusable_prefix(
        self,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor | None,
        *,
        allow_pinned: bool = False,
    ) -> None:
        self.stats.full_prompt_store_requests += 1
        if not self.store_full_prompt_prefixes:
            self._record_full_prompt_store_skip("disabled", tokens)
            return
        if self.pin_shared_prefix and not allow_pinned:
            # Per-request full-prompt stores would starve the prefix-row pool.
            # Keep only pinned shared prefixes in this mode; unpinned modes
            # remove evicted routes from the radix index so shorter live prefixes
            # can still match. A long-output opt-in below enables this for
            # multi-turn style workloads where one full-prompt row can save a
            # much larger next-turn suffix prefill.
            self._record_full_prompt_store_skip("pinned_without_allowance", tokens)
            self._record_full_prompt_reuse_candidate_store(request_id, tokens)
            return
        if self._delayed_pinned_full_prompt_store_allowed(allow_pinned=allow_pinned):
            self.stats.full_prompt_store_deferred_requests += 1
            self.stats.full_prompt_store_deferred_tokens += len(tokens)
            return
        store_logits = True
        if allow_pinned:
            store_logits = env_flag(
                "TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_LOGITS",
                False,
            )
        if "TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS" in os.environ:
            store_logits = env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS", store_logits)
        self._store_reusable_prefix_tokens(
            None,
            request_id,
            tokens,
            source_row,
            logits,
            store_logits=store_logits,
        )
        if request_id in self.reusable_prefixes:
            self.stats.full_prompt_store_stored_requests += 1
        else:
            self._record_full_prompt_store_skip("prefix_row_unavailable_or_store_disabled", tokens)

    def _full_prompt_store_needs_logits(self, request: ServingRequest) -> bool:
        if not self.store_full_prompt_prefixes:
            return False
        allow_pinned = self._allow_pinned_full_prompt_store(request)
        if self.pin_shared_prefix and not allow_pinned:
            return False
        if self._delayed_pinned_full_prompt_store_allowed(allow_pinned=allow_pinned):
            return False
        store_logits = True
        if allow_pinned:
            store_logits = env_flag(
                "TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_LOGITS",
                False,
            )
        if "TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS" in os.environ:
            store_logits = env_flag(
                "TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS",
                store_logits,
            )
        return bool(store_logits)

    def _prefix_prefill_group_can_omit_logits(
        self,
        group: Sequence[tuple[int, ServingRequest, int, _ReusablePrefix]],
    ) -> bool:
        return all(
            not self._full_prompt_store_needs_logits(request)
            for _original_index, request, _prefix_hit_tokens, _reusable in group
        )

    def _record_full_prompt_store_skip(self, reason: str, tokens: tuple[int, ...]) -> None:
        self.stats.full_prompt_store_skipped_requests += 1
        self.stats.full_prompt_store_skipped_tokens += len(tokens)
        if not self.profile_timings:
            return
        self._record_shape_count(
            self.stats.full_prompt_store_skip_reason_counts,
            reason,
        )
        self._record_shape_total(
            self.stats.full_prompt_store_skip_reason_tokens,
            reason,
            len(tokens),
        )

    def _record_full_prompt_reuse_candidate_store(
        self,
        request_id: str,
        tokens: tuple[int, ...],
    ) -> None:
        if not self.profile_timings or not tokens:
            return
        if self._full_prompt_reuse_candidate_enabled() and self.prefix_cache_capacity > 0:
            route_id: Hashable = ("full_prompt_reuse_candidate", request_id)
            if route_id in self._full_prompt_reuse_candidate_order:
                self._full_prompt_reuse_candidate_order.remove(route_id)
            self._full_prompt_reuse_candidate_cache.remove(route_id)
            self._full_prompt_reuse_candidate_cache.add(request_id, tokens, route_id=route_id)
            self._full_prompt_reuse_candidate_order.append(route_id)
            self.stats.full_prompt_reuse_candidate_stored_requests += 1
            self.stats.full_prompt_reuse_candidate_stored_tokens += len(tokens)

            limit = env_int(
                "TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_CAPACITY",
                self.prefix_cache_capacity,
                minimum=0,
            )
            while limit > 0 and len(self._full_prompt_reuse_candidate_order) > limit:
                oldest = self._full_prompt_reuse_candidate_order.pop(0)
                self._full_prompt_reuse_candidate_cache.remove(oldest)
            if limit == 0:
                for route in self._full_prompt_reuse_candidate_order:
                    self._full_prompt_reuse_candidate_cache.remove(route)
                self._full_prompt_reuse_candidate_order = []

        if self._persistent_full_prompt_reuse_candidate_enabled():
            self._record_persistent_full_prompt_reuse_candidate_store(request_id, tokens)

    def _record_full_prompt_reuse_candidate_lookup(
        self,
        prompt: tuple[int, ...],
        actual_prefix_tokens: int,
    ) -> None:
        if not self.profile_timings:
            return
        if self._full_prompt_reuse_candidate_enabled() and self._full_prompt_reuse_candidate_order:
            match, entry = self._full_prompt_reuse_candidate_cache.lookup(prompt)
            if entry is not None:
                self._record_full_prompt_reuse_candidate_match(
                    prompt,
                    actual_prefix_tokens,
                    int(match.depth),
                    persistent=False,
                )
        if self._persistent_full_prompt_reuse_candidate_enabled():
            current_session = int(getattr(self, "_full_prompt_reuse_candidate_session_id", 0))
            match, entry = _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CACHE.lookup_filtered(
                prompt,
                lambda candidate: (
                    _persistent_full_prompt_reuse_candidate_session(candidate.route_id)
                    != current_session
                ),
            )
            if entry is not None:
                self._record_full_prompt_reuse_candidate_match(
                    prompt,
                    actual_prefix_tokens,
                    int(match.depth),
                    persistent=True,
                )

    def _full_prompt_reuse_candidate_enabled(self) -> bool:
        env_name = "TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_PROFILE"
        if env_name in os.environ:
            return env_flag(env_name, False)
        if "TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_CAPACITY" in os.environ:
            return env_int(
                "TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_CAPACITY",
                self.prefix_cache_capacity,
                minimum=0,
            ) > 0
        return False

    def _persistent_full_prompt_reuse_candidate_enabled(self) -> bool:
        return env_flag(
            "TORCHINFERNO_CONTINUOUS_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE",
            False,
        )

    def _record_persistent_full_prompt_reuse_candidate_store(
        self,
        request_id: str,
        tokens: tuple[int, ...],
    ) -> None:
        sequence = _next_persistent_full_prompt_reuse_candidate_sequence()
        session_id = int(getattr(self, "_full_prompt_reuse_candidate_session_id", 0))
        route_id: Hashable = (
            "persistent_full_prompt_reuse_candidate",
            session_id,
            sequence,
        )
        _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CACHE.add(
            request_id,
            tokens,
            route_id=route_id,
        )
        _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER.append(route_id)
        self.stats.persistent_full_prompt_reuse_candidate_stored_requests += 1
        self.stats.persistent_full_prompt_reuse_candidate_stored_tokens += len(tokens)

        limit = env_int(
            "TORCHINFERNO_CONTINUOUS_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CAPACITY",
            max(1, self.prefix_cache_capacity),
            minimum=0,
        )
        while limit > 0 and len(_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER) > limit:
            oldest = _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER.pop(0)
            _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CACHE.remove(oldest)
        if limit == 0:
            for route in _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER:
                _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_CACHE.remove(route)
            _PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE_ORDER.clear()

    def _record_full_prompt_reuse_candidate_match(
        self,
        prompt: tuple[int, ...],
        actual_prefix_tokens: int,
        match_depth: int,
        *,
        persistent: bool,
    ) -> None:
        if match_depth <= 0:
            return
        # Pinned full-prompt adoption stores KV without logits by default, so an
        # exact prompt hit still needs at least one prompt token replay to sample.
        candidate_tokens = min(int(match_depth), max(0, len(prompt) - 1))
        actual_tokens = max(0, int(actual_prefix_tokens))
        if candidate_tokens <= actual_tokens:
            return
        suffix_tokens = max(0, len(prompt) - candidate_tokens)
        extra_tokens = candidate_tokens - actual_tokens
        prefix = (
            "persistent_full_prompt_reuse_candidate"
            if persistent
            else "full_prompt_reuse_candidate"
        )
        setattr(
            self.stats,
            f"{prefix}_requests",
            getattr(self.stats, f"{prefix}_requests") + 1,
        )
        setattr(
            self.stats,
            f"{prefix}_tokens",
            getattr(self.stats, f"{prefix}_tokens") + candidate_tokens,
        )
        setattr(
            self.stats,
            f"{prefix}_extra_tokens",
            getattr(self.stats, f"{prefix}_extra_tokens") + extra_tokens,
        )
        setattr(
            self.stats,
            f"{prefix}_suffix_tokens",
            getattr(self.stats, f"{prefix}_suffix_tokens") + suffix_tokens,
        )
        self._record_shape_count(
            getattr(self.stats, f"{prefix}_token_counts"),
            str(candidate_tokens),
        )
        self._record_shape_count(
            getattr(self.stats, f"{prefix}_extra_token_counts"),
            str(extra_tokens),
        )
        self._record_shape_count(
            getattr(self.stats, f"{prefix}_suffix_token_counts"),
            str(suffix_tokens),
        )

    def _store_delayed_pinned_full_prompt_prefix(self, state: _ActiveRequest) -> bool:
        if not self._delayed_pinned_full_prompt_store_enabled(state.request):
            return False
        prompt = state.request.prompt
        if not prompt or not self._state_has_full_prompt_kv(state):
            return False
        if self._cache_row_seq_len(state.row, state.seq_len) < len(prompt):
            return False
        route_id = state.request.request_id
        try:
            adopted = self._adopt_reusable_prefix_tokens(
                route_id,
                state.request.request_id,
                prompt,
                state.row,
            )
            if adopted:
                self.stats.full_prompt_store_adopted_requests += 1
                self.stats.full_prompt_store_adopted_tokens += len(prompt)
            return adopted
        except Exception:
            if env_flag("TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_STRICT", False):
                raise
            return False

    def _generated_prefix_cache_base_enabled(self) -> bool:
        return (
            self.prefix_cache_capacity > 0
            and self.store_reusable_prefixes
            and self.store_full_prompt_prefixes
        )

    def _generated_prefix_cache_enabled(self) -> bool:
        if not self._generated_prefix_cache_base_enabled():
            return False
        configured = self.generated_prefix_cache
        if configured is not None:
            return bool(configured)
        if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE", False):
            return True
        if not env_flag("TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE", False):
            return False
        return any(
            isinstance(route_id, tuple) and route_id[:1] == ("generated_prefix",)
            for route_id in self.reusable_prefixes
        )

    def _should_collect_generated_prefix_logits(self, states: list[_ActiveRequest]) -> bool:
        if not states or not self._generated_prefix_cache_base_enabled():
            return False
        configured = self.generated_prefix_cache
        if configured is not None:
            return bool(configured)
        if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE", False):
            return True
        if not env_flag("TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE", False):
            return False
        waiting = self._online_waiting
        if waiting is None or not bool(waiting):
            return False
        candidate_prompts = {
            state.request.prompt
            for state in states
            if state.generated > 0 and self._state_has_full_prompt_kv(state)
        }
        if not candidate_prompts:
            return False
        pending_exact = sum(
            1
            for item in waiting._items
            if item.request.prompt in candidate_prompts
        )
        min_pending = env_int(
            "TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_MIN_PENDING",
            16,
            minimum=1,
        )
        return pending_exact >= min_pending

    def _finished_prefix_cache_enabled(self) -> bool:
        return (
            self.prefix_cache_capacity > 0
            and self.store_reusable_prefixes
            and self.store_full_prompt_prefixes
            and env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE", False)
        )

    @staticmethod
    def _generated_prefix_route_id(tokens: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        return ("generated_prefix", tokens)

    @staticmethod
    def _finished_prefix_route_id(tokens: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        return ("finished_prefix", tokens)

    def _lookup_exact_reusable_prefix(self, tokens: tuple[int, ...]) -> _ReusablePrefix | None:
        if not self._generated_prefix_cache_enabled():
            return None
        route_id = self._generated_prefix_route_id(tokens)
        reusable = self.reusable_prefixes.get(route_id)
        if reusable is not None and reusable.tokens == tokens:
            return reusable
        match, entry = self.prefix_cache.lookup(tokens)
        if entry is None or match.depth != len(tokens):
            return None
        reusable = self.reusable_prefixes.get(entry.route_id)
        if reusable is None or reusable.tokens != tokens:
            return None
        return reusable

    def _store_generated_reusable_prefix(
        self,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor,
    ) -> None:
        if not self._generated_prefix_cache_base_enabled():
            return
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) <= 0 or len(tokens) > max_tokens:
            return
        route_id = self._generated_prefix_route_id(tokens)
        if route_id in self.reusable_prefixes:
            return
        store_logits = logits[:, None, :] if logits.ndim == 2 else logits
        try:
            self._store_reusable_prefix_tokens(route_id, request_id, tokens, source_row, store_logits)
        except Exception:
            if env_flag("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_STRICT", False):
                raise
            return
        if route_id in self.reusable_prefixes:
            self.stats.generated_prefix_store_requests += 1

    def _store_finished_reusable_prefix(self, state: _ActiveRequest) -> bool:
        if not self._finished_prefix_cache_enabled():
            return False
        if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
            return False
        kv_token_count = min(
            self._cache_row_seq_len(state.row, state.seq_len),
            len(state.tokens),
        )
        tokens = tuple(state.tokens[:kv_token_count])
        if len(tokens) <= len(state.request.prompt):
            return False
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        if len(tokens) > max_tokens:
            return False
        route_id = self._finished_prefix_route_id(tokens)
        if route_id in self.reusable_prefixes:
            return False
        if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_ADOPT_ROWS", True):
            try:
                if self._adopt_reusable_prefix_tokens(
                    route_id,
                    state.request.request_id,
                    tokens,
                    state.row,
                ):
                    self.stats.generated_prefix_store_requests += 1
                    return True
            except Exception:
                if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_STRICT", False):
                    raise
        try:
            self._set_cache_row_seq_len(state.row, len(tokens))
            self._store_reusable_prefix_tokens(
                route_id,
                state.request.request_id,
                tokens,
                state.row,
                None,
                store_logits=False,
            )
        except Exception:
            if env_flag("TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_STRICT", False):
                raise
            return False
        if route_id in self.reusable_prefixes:
            self.stats.generated_prefix_store_requests += 1
        return False

    def _store_decoded_reusable_prefixes(
        self,
        states: list[_ActiveRequest],
        logits: Tensor | None,
    ) -> None:
        if logits is None or not states or not self._should_collect_generated_prefix_logits(states):
            return
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for row_index, state in enumerate(states):
            if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
                continue
            tokens = tuple(state.tokens)
            if len(tokens) <= len(state.request.prompt):
                continue
            route_id = self._generated_prefix_route_id(tokens)
            if route_id in seen or route_id in self.reusable_prefixes:
                continue
            seen.add(route_id)
            self._set_cache_row_seq_len(state.row, len(tokens))
            self._store_generated_reusable_prefix(
                state.request.request_id,
                tokens,
                state.row,
                logits[row_index : row_index + 1],
            )

    @staticmethod
    def _state_has_full_prompt_kv(state: _ActiveRequest) -> bool:
        prompt_len = len(state.request.prompt)
        return state.prefix_hit_tokens >= prompt_len or state.seq_len >= prompt_len

    def _needs_generated_prefix_logits(self, states: list[_ActiveRequest]) -> bool:
        if not states or not self._should_collect_generated_prefix_logits(states):
            return False
        max_tokens = env_int("TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE_MAX_TOKENS", 1024, minimum=1)
        for state in states:
            if state.generated <= 0 or not self._state_has_full_prompt_kv(state):
                continue
            tokens = tuple(state.tokens)
            if len(tokens) <= len(state.request.prompt) or len(tokens) > max_tokens:
                continue
            if self._generated_prefix_route_id(tokens) not in self.reusable_prefixes:
                return True
        return False

    def _store_reusable_prefix_tokens(
        self,
        route_id: Hashable | None,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
        logits: Tensor | None,
        *,
        store_logits: bool = True,
    ) -> None:
        if not self.store_reusable_prefixes or not env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE", True):
            return
        actual_route = request_id if route_id is None else route_id
        old_prefix = self.reusable_prefixes.pop(actual_route, None)
        if old_prefix is not None:
            self._clear_physical_row(old_prefix.row)
            if actual_route in self._prefix_order:
                self._prefix_order.remove(actual_route)
            self._free_prefix_rows.append(old_prefix.row)
            self._free_prefix_rows.sort()
        # Drop any stale pin; the caller re-pins after a successful re-store.
        self._pinned_prefix_routes.discard(actual_route)
        self.prefix_cache.remove(actual_route)
        prefix_row = self._acquire_prefix_row()
        if env_flag("TORCHINFERNO_REUSE_DEBUG", False):
            import sys as _sd
            print(
                f"[STORE_DBG] rank={getattr(self.model, 'rank', 0)} "
                f"store={'SKIP(no_row)' if prefix_row is None else prefix_row} "
                f"ntoks={len(tokens)} free_prefix_rows={len(self._free_prefix_rows)} "
                f"cached={len(self.reusable_prefixes)}",
                file=_sd.stderr, flush=True,
            )
        if prefix_row is None:
            return
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        self._copy_prefix(source_row, prefix_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            prefix_row,
            (
                logits[:, -1:, :].detach().clone().cpu()
                if store_logits and logits is not None
                else None
            ),
        )
        self._prefix_order.append(entry.route_id)

    def _store_reusable_prefix_tokens_in_row(
        self,
        route_id: Hashable | None,
        request_id: str,
        tokens: tuple[int, ...],
        prefix_row: int,
        logits: Tensor | None,
        *,
        store_logits: bool = True,
    ) -> bool:
        if not self.store_reusable_prefixes or not env_flag(
            "TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE",
            True,
        ):
            return False
        actual_route = request_id if route_id is None else route_id
        old_prefix = self.reusable_prefixes.pop(actual_route, None)
        if old_prefix is not None:
            if actual_route in self._prefix_order:
                self._prefix_order.remove(actual_route)
            if old_prefix.row != prefix_row:
                self._clear_physical_row(old_prefix.row)
                if old_prefix.row not in self._free_prefix_rows:
                    self._free_prefix_rows.append(old_prefix.row)
                    self._free_prefix_rows.sort()
        self._pinned_prefix_routes.discard(actual_route)
        self.prefix_cache.remove(actual_route)
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        self._set_cache_row_seq_len(prefix_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            prefix_row,
            (
                logits[:, -1:, :].detach().clone().cpu()
                if store_logits and logits is not None
                else None
            ),
        )
        self._prefix_order.append(entry.route_id)
        return True

    def _adopt_reusable_prefix_tokens(
        self,
        route_id: Hashable,
        request_id: str,
        tokens: tuple[int, ...],
        source_row: int,
    ) -> bool:
        if not self.store_reusable_prefixes or not env_flag("TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE", True):
            return False
        if route_id in self.reusable_prefixes:
            return False
        replacement_active_row = self._acquire_prefix_row()
        if replacement_active_row is None:
            return False
        self.prefix_cache.remove(route_id)
        entry = self.prefix_cache.add(request_id, tokens, route_id=route_id)
        self._set_cache_row_seq_len(source_row, len(tokens))
        self.reusable_prefixes[entry.route_id] = _ReusablePrefix(
            entry.route_id,
            tokens,
            source_row,
            None,
        )
        self._prefix_order.append(entry.route_id)
        self._remember_row_seq_len(replacement_active_row, 0)
        self._mark_active_row_free(replacement_active_row)
        return True

    def _reusable_prefix_hit_tokens(self, prompt: tuple[int, ...]) -> int:
        match, entry = self.prefix_cache.lookup(prompt)
        if entry is None or entry.route_id not in self.reusable_prefixes:
            return 0
        return match.depth

    def _copy_prefix(self, source_row: int, dest_row: int, tokens: int) -> None:
        cache = self._require_cache()
        cache.copy_prefix_from(cache, tokens, source_row=source_row, dest_row=dest_row)  # type: ignore[attr-defined]
        self._remember_row_seq_len(dest_row, tokens)

    def _copy_prefix_to_rows(self, source_row: int, dest_rows: list[int], tokens: int) -> None:
        if not dest_rows:
            return
        if self._copy_prefix_to_rows_dense(source_row, dest_rows, tokens):
            return
        for row in dest_rows:
            self._copy_prefix(source_row, row, tokens)

    def _copy_prefix_to_rows_dense(self, source_row: int, dest_rows: list[int], tokens: int) -> bool:
        if tokens <= 0:
            for row in dest_rows:
                self._remember_row_seq_len(row, 0)
            return True
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if not layers:
            return False
        try:
            for layer in layers:
                keys = getattr(layer, "keys")
                values = getattr(layer, "values")
                physical = getattr(layer, "_physical_row", None)
                src = int(physical(source_row) if callable(physical) else source_row)
                dst = tuple(int(physical(row) if callable(physical) else row) for row in dest_rows)
                if tokens > keys.size(2) or tokens > values.size(2):
                    return False
                seq_len_for_rows = getattr(layer, "seq_len_for_rows", None)
                if callable(seq_len_for_rows):
                    if int(seq_len_for_rows((src,))) < tokens:
                        return False
                else:
                    seq_lens = getattr(layer, "_seq_lens", None)
                    if isinstance(seq_lens, list) and (src >= len(seq_lens) or int(seq_lens[src]) < tokens):
                        return False
                source_keys = keys[src : src + 1, :, :tokens, :].expand(len(dst), -1, -1, -1)
                source_values = values[src : src + 1, :, :tokens, :].expand(len(dst), -1, -1, -1)
                span = _contiguous_int_span(dst)
                if span is not None:
                    start, end = span
                    keys[start:end, :, :tokens, :].copy_(source_keys)
                    values[start:end, :, :tokens, :].copy_(source_values)
                else:
                    index = torch.tensor(dst, dtype=torch.long, device=keys.device)
                    keys[:, :, :tokens, :].index_copy_(0, index, source_keys)
                    values[:, :, :tokens, :].index_copy_(0, index, source_values)
                setter = getattr(layer, "_set_rows_seq_len", None)
                if callable(setter):
                    setter(dst, tokens)
                else:
                    seq_lens = getattr(layer, "_seq_lens", None)
                    if isinstance(seq_lens, list):
                        for row in dst:
                            seq_lens[row] = int(tokens)
            for row in dest_rows:
                self._remember_row_seq_len(row, tokens)
            return True
        except Exception:
            return False

    def _acquire_active_row(self, *, clear_cache: bool = True) -> int:
        if not self._free_active_rows:
            raise RuntimeError("no active serving rows available")
        row = self._free_active_rows.pop()
        self._reset_active_row_for_acquire(row, clear_cache=clear_cache)
        return row

    def _acquire_active_row_or_none(self, *, clear_cache: bool = True) -> int | None:
        if not self._free_active_rows:
            return None
        row = self._free_active_rows.pop()
        self._reset_active_row_for_acquire(row, clear_cache=clear_cache)
        return row

    def _reset_active_row_for_acquire(self, row: int, *, clear_cache: bool = True) -> None:
        if not clear_cache or env_flag("TORCHINFERNO_CONTINUOUS_SKIP_ACTIVE_ROW_CLEAR", False):
            self._set_cache_row_seq_len(row, 0)
            return
        self._clear_physical_row(row)

    def _can_skip_prefix_graph_active_row_clear(self) -> bool:
        cache = self._require_cache()
        return getattr(cache, "cache_backend", self.cache_backend) == "dense"

    def _acquire_free_prefix_row_or_none(self) -> int | None:
        if self.prefix_cache_capacity == 0 or not self._free_prefix_rows:
            return None
        for row in _preferred_prefix_rows():
            if row in self._free_prefix_rows:
                self._free_prefix_rows.remove(row)
                return row
        return self._free_prefix_rows.pop()

    def _release_active_row(self, row: int) -> None:
        # Skip the GPU KV zero on release: _acquire_active_row already clears a
        # row before any reuse, so clearing here too is a redundant second pass
        # that runs inside the hot decode state-update loop (once per finishing
        # request). Only the seq_len reset is needed for correctness -- a free
        # row reused as decode-bucket padding has seq_len 0 so attention never
        # reads its stale KV, and its bucket output is discarded regardless.
        self._remember_row_seq_len(row, 0)
        self._mark_active_row_free(row)

    def _mark_active_row_free(self, row: int) -> None:
        if row in self._free_active_rows:
            return
        # Keep descending order so pop() returns the lowest available row. This
        # preserves dense low-row active batches after prefix-row adoption.
        insert_at = 0
        while (
            insert_at < len(self._free_active_rows)
            and self._free_active_rows[insert_at] > row
        ):
            insert_at += 1
        self._free_active_rows.insert(insert_at, row)

    def _prefill_static_batch_size(self, request_count: int) -> int:
        bucket = env_int("TORCHINFERNO_CONTINUOUS_PREFILL_STATIC_BATCH", self.max_active_requests, minimum=1)
        available = request_count + len(self._free_active_rows)
        return min(bucket, self.max_active_requests, available)

    def _acquire_prefix_row(self) -> int | None:
        if self.prefix_cache_capacity == 0:
            return None
        if self._free_prefix_rows:
            for row in _preferred_prefix_rows():
                if row in self._free_prefix_rows:
                    self._free_prefix_rows.remove(row)
                    return row
            return self._free_prefix_rows.pop()
        # Evict the oldest UNPINNED reusable prefix. Pinned routes (the active
        # shared prefix) are skipped so their KV stays a valid copy source.
        for index, route_id in enumerate(self._prefix_order):
            if route_id in self._pinned_prefix_routes:
                continue
            self._prefix_order.pop(index)
            prefix = self.reusable_prefixes.pop(route_id, None)
            self.prefix_cache.remove(route_id)
            if prefix is not None:
                self._remember_row_seq_len(prefix.row, 0)
                self._forget_row_cached_prefix(prefix.row)
                return prefix.row
            return self._acquire_prefix_row()
        return None

    def _release_prefix_row(self, row: int) -> None:
        self._clear_physical_row(row)
        if row not in self._free_prefix_rows:
            self._free_prefix_rows.append(row)
            self._free_prefix_rows.sort()

    def _cache_view(self, rows: list[int]) -> object:
        row_key = tuple(rows)
        view = self._cache_views.get(row_key)
        if view is None:
            view = self._require_cache().for_rows(row_key)  # type: ignore[attr-defined]
            self._cache_views[row_key] = view
        return view

    def _cache_uses_paged_kv(self) -> bool:
        if self._cache is None:
            return False
        layers = getattr(self._cache, "layers", None)
        return bool(layers and hasattr(layers[0], "paged_kv"))

    def _cache_supports_tensor_ragged_prefill(self) -> bool:
        cache = self._cache
        if cache is None:
            return True
        if str(getattr(cache, "cache_backend", "dense")).lower() == "dense":
            return True
        layers = tuple(getattr(cache, "layers", ()) or ())
        if not layers:
            return True
        for layer in layers:
            try:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
            except Exception:
                return False
            if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
                return False
        return True

    def _require_cache(self) -> object:
        if self._cache is None:
            raise RuntimeError("serving cache has not been initialized")
        return self._cache

    def _require_online_waiting(self) -> ServingQueue:
        if self._online_waiting is None:
            raise RuntimeError("online serving has not been initialized")
        return self._online_waiting

    def _clear_physical_row(self, row: int) -> None:
        self._cache_view([row]).clear_row(0)  # type: ignore[attr-defined]
        self._remember_row_seq_len(row, 0)
        self._forget_row_cached_prefix(row)

    def _allocate_cache(self, batch_size: int, max_seq_len: int) -> object:
        allocate_cache = getattr(self.model, "allocate_cache")
        if self.cache_backend != "dense":
            try:
                return allocate_cache(
                    batch_size,
                    max_seq_len=max_seq_len,
                    device=self.device,
                    cache_backend=self.cache_backend,
                    page_size=self.page_size,
                )
            except TypeError:
                try:
                    return allocate_cache(
                        batch_size,
                        max_seq_len=max_seq_len,
                        cache_backend=self.cache_backend,
                        page_size=self.page_size,
                    )
                except TypeError:
                    raise ValueError(f"model does not support cache_backend={self.cache_backend}") from None
        try:
            return allocate_cache(
                batch_size,
                max_seq_len=max_seq_len,
                device=self.device,
            )
        except TypeError:
            return allocate_cache(batch_size, max_seq_len=max_seq_len)

    def _forward_model(self, input_ids: Tensor, *, cache: object, use_cache: bool) -> tuple[Tensor, object | None]:
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if callable(forward):
            if self._prefer_sharded_logits():
                try:
                    return forward(
                        input_ids,
                        cache=cache,
                        use_cache=use_cache,
                        return_last_logits_only=True,
                        return_sharded_logits=True,
                    )
                except TypeError:
                    pass
            return forward(input_ids, cache=cache, use_cache=use_cache)
        raise TypeError("serving model must be callable or expose forward()")

    def _prefill_full_logits(self, input_ids: Tensor, *, cache: object) -> tuple[Tensor, object | None]:
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if not callable(forward):
            raise TypeError("serving model must be callable or expose forward()")
        kwargs: dict[str, object] = {
            "cache": cache,
            "use_cache": True,
        }
        if self._prefer_sharded_logits():
            kwargs["return_last_logits_only"] = False
            kwargs["return_sharded_logits"] = True
        try:
            return forward(input_ids, **kwargs)
        except TypeError:
            return forward(input_ids, cache=cache, use_cache=True)

    def _prefill_logits(self, input_ids: Tensor, *, cache: object) -> tuple[Tensor, object | None]:
        graph_logits = self._try_prefill_logits_graph(input_ids, cache)
        if graph_logits is not None:
            return graph_logits, cache
        fi_fwd = getattr(self.model, "forward_step_flashinfer", None)
        fi_ready = getattr(self.model, "_flashinfer_jit_warmed", False)
        has_paged = self._cache_uses_paged_kv()
        if fi_fwd is not None and fi_ready and has_paged and input_ids.device.type == "cuda":
            try:
                batch, seq_len = input_ids.shape
                full_cache = self._require_cache()
                parent = getattr(cache, "_parent_cache", None)
                rows_attr = getattr(cache, "_rows", None) or getattr(cache, "_row_list", None)
                if rows_attr is not None and parent is not None:
                    row_list = list(rows_attr) if not isinstance(rows_attr, list) else rows_attr
                else:
                    row_list = list(range(batch))
                    full_cache = cache
                seq_lens_val = 0
                try:
                    seq_lens_val = int(cache.seq_len)
                except Exception:
                    pass
                # Reject writes past the row capacity BEFORE launching the kernel:
                # a KV scatter at columns [seq_lens_val, seq_lens_val+seq_len) that
                # exceeds max_seq is a device-side index assert that kills every TP
                # rank. Raise a catchable error so one request fails instead.
                _ms = self._cache_max_seq_len()
                if _ms is not None and seq_lens_val + seq_len > _ms:
                    raise RuntimeError(
                        f"prefill write past cache: seq_lens={seq_lens_val} + "
                        f"q={seq_len} > max_seq={_ms}"
                    )
                row_indices = torch.tensor(row_list[:batch], device=input_ids.device, dtype=torch.long)
                seq_lens = torch.full((batch,), seq_lens_val, device=input_ids.device, dtype=torch.long)
                q_lens = torch.full((batch,), seq_len, device=input_ids.device, dtype=torch.long)
                write_pos = torch.arange(seq_lens_val, seq_lens_val + seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
                logit_pos = torch.full((batch,), seq_len - 1, device=input_ids.device, dtype=torch.long)
                logits = fi_fwd(
                    input_ids, full_cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_pos, logit_positions=logit_pos,
                    row_indices=row_indices,
                )
                for lc in full_cache.layers:
                    if hasattr(lc, '_seq_lens'):
                        for r in row_list[:batch]:
                            lc._seq_lens[r] = seq_lens_val + seq_len
                    if hasattr(lc, '_uniform_seq_len'):
                        lc._uniform_seq_len[0] = None
                return logits, cache
            except Exception as _fi_exc:
                import sys as _fis
                print(
                    f"[PREFILL] FlashInfer eager prefill failed: {_fi_exc!r}",
                    file=_fis.stderr, flush=True,
                )
        if has_paged:
            # The SDPA dense _forward_model below indexes the cache as a contiguous
            # [batch, ...] tensor, which CUDA index-asserts against the paged
            # FlashInfer cache and takes down EVERY tensor-parallel rank (an
            # unrecoverable device-side assert, not a Python exception). When the
            # cache is paged, fail THIS request with a catchable error instead so
            # the engine degrades one request rather than crashing the server.
            raise RuntimeError(
                "paged-cache prefill requires a FlashInfer graph/eager path; "
                "refusing to fall back to the SDPA dense forward"
            )
        return self._forward_model(input_ids, cache=cache, use_cache=True)

    def _prefill_cache_only(self, input_ids: Tensor, *, rows: list[int]) -> bool:
        batch = int(input_ids.size(0))
        if len(rows) != batch:
            raise ValueError("cache-only prefill row count must match input batch")
        cache = self._require_cache()
        required_rows = max(rows) + 1 if rows else 0
        seq_lens_list = [0] * required_rows
        for row in rows:
            seq_lens_list[row] = self._cache_row_seq_len(row, 0)
        seq_lens = torch.tensor(seq_lens_list, device=input_ids.device, dtype=torch.long)
        row_indices = torch.tensor(rows, device=input_ids.device, dtype=torch.long)
        graph = getattr(self.model, "try_prefill_ragged_cache_graph", None)
        if callable(graph):
            try:
                filled = graph(
                    input_ids,
                    cache,
                    seq_lens=seq_lens,
                    row_indices=row_indices,
                    capture_on_miss=False,
                )
                if filled:
                    self.stats.prefill_graph_hits += 1
                    return True
            except Exception as exc:
                warn_optional_failure("serving.prefill_cache_only_graph", exc)
        prefill_cache = getattr(self.model, "prefill_ragged_cache", None)
        if not callable(prefill_cache):
            return False
        try:
            return bool(
                prefill_cache(
                    input_ids,
                    cache,
                    seq_lens=seq_lens,
                    row_indices=row_indices,
                )
            )
        except Exception as exc:
            warn_optional_failure("serving.prefill_cache_only", exc)
            return False

    def _try_flashinfer_prefill(
        self,
        requests: list[tuple[int, "ServingRequest", int, "_ReusablePrefix | None"]],
        step: int,
        *,
        events: list["ServingTokenEvent"] | None = None,
        graph_only: bool = False,
    ) -> list["_ActiveRequest"] | None:
        forward_fi = getattr(self.model, "forward_step_flashinfer", None)
        if forward_fi is None:
            return None
        try:
            __import__("flashinfer")
        except ImportError:
            return None
        cache = self._require_cache()
        active: list[_ActiveRequest] = []
        rows = []
        prompts: list[tuple[int, ...]] = []
        for original_index, request, prefix_hit_tokens, reusable in requests:
            row = self._acquire_active_row()
            rows.append(row)
            prompts.append(request.prompt)

        if not prompts:
            return None

        prompt_lens = [len(p) for p in prompts]
        max_prompt_len = max(prompt_lens)
        batch = len(prompts)
        padded = [list(p) + [0] * (max_prompt_len - len(p)) for p in prompts]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        q_lens = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        seq_lens = torch.zeros(batch, dtype=torch.long, device=self.device)
        row_indices = torch.tensor(rows, dtype=torch.long, device=self.device)
        positions = []
        for i in range(batch):
            positions.append(list(range(max_prompt_len)))
        write_positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        logit_positions = torch.tensor(
            [plen - 1 for plen in prompt_lens], dtype=torch.long, device=self.device,
        )
        try:
            logits = None
            packed_fi = getattr(self.model, "prefill_ragged_logits_packed_flashinfer", None)
            if (
                env_flag("TORCHINFERNO_CONTINUOUS_PACKED_FLASHINFER_PREFILL", False)
                and packed_fi is not None
                and getattr(cache, "cache_backend", self.cache_backend) == "flashinfer"
                and bool(torch.any(q_lens < max_prompt_len))
            ):
                seq_lens_full = torch.zeros(
                    max(rows) + 1,
                    dtype=torch.long,
                    device=self.device,
                )
                packed_start_s = time.perf_counter() if self.profile_timings else 0.0
                try:
                    logits = packed_fi(
                        input_ids,
                        cache,
                        seq_lens=seq_lens_full,
                        q_lens=q_lens,
                        row_indices=row_indices,
                        logit_positions=logit_positions,
                    )
                    self._record_packed_flashinfer_prefill(
                        q_lens,
                        model_tokens=int(input_ids.size(0) * input_ids.size(1)),
                        start_s=packed_start_s,
                    )
                except Exception as exc:
                    warn_optional_failure("serving.packed_flashinfer_prefill", exc)
                    logits = None
            graph_fn = getattr(self.model, "try_prefill_flashinfer_graph", None)
            if logits is None and graph_fn is not None:
                logits = graph_fn(
                    input_ids, cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_positions,
                    logit_positions=logit_positions,
                    row_indices=row_indices,
                )
            if logits is None and graph_only:
                # Graph missed and the caller (single-request path) does not want
                # the launch-bound eager FlashInfer fallback; release rows so the
                # normal _prefill_one/SDPA path can handle this request instead.
                for row in rows:
                    self._release_active_row(row)
                return None
            if logits is None:
                logits = forward_fi(
                    input_ids, cache,
                    seq_lens=seq_lens, q_lens=q_lens,
                    write_positions=write_positions,
                    logit_positions=logit_positions,
                    row_indices=row_indices,
                )
        except Exception as _fi_prefill_exc:
            import sys as _fps
            import traceback as _fptb
            print(
                f"[FI_PREFILL] FlashInfer prefill failed batch={batch}: {_fi_prefill_exc}",
                file=_fps.stderr, flush=True,
            )
            _fptb.print_exc(file=_fps.stderr)
            for row in rows:
                self._release_active_row(row)
            return None
        self._record_model_call("prefill", batch, tokens=int(q_lens.sum().item()))
        next_tokens = self._sample_logits_for_requests(
            logits[:, -1, :],
            [request for _original_index, request, _prefix_hit_tokens, _reusable in requests],
        ).detach().cpu().tolist()
        for i in range(batch):
            self._set_cache_row_seq_len(rows[i], len(requests[i][1].prompt))
        for i, (original_index, request, prefix_hit_tokens, reusable) in enumerate(requests):
            row = rows[i]
            next_token = int(next_tokens[i])
            prompt_len = len(request.prompt)
            seq_len = prompt_len
            self._remember_row_seq_len(row, seq_len)
            self._store_reusable_prefix(
                request.request_id,
                request.prompt,
                row,
                logits[i:i+1],
                allow_pinned=self._allow_pinned_full_prompt_store(request),
            )
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=seq_len,
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _prefill_flashinfer_reuse(
        self,
        group: list[tuple[int, "ServingRequest", int, "_ReusablePrefix"]],
        step: int,
        *,
        events: list["ServingTokenEvent"] | None = None,
    ) -> list["_ActiveRequest"] | None:
        # FlashInfer-native prefix reuse: copy cached prefix KV into each row,
        # then prefill only the suffix (seq_lens=prefix_len). Empty suffix (whole
        # prompt cached) just samples the cached logits. FlashInfer-cache-safe.
        forward_fi = getattr(self.model, "forward_step_flashinfer", None)
        if forward_fi is None:
            return None
        try:
            __import__("flashinfer")
        except ImportError:
            return None
        max_seq = self._cache_max_seq_len()
        for _idx, request, hit, reusable in group:
            if reusable is None or hit <= 0 or hit > len(request.prompt):
                return None
            if max_seq is not None and len(request.prompt) > max_seq:
                return None
            if reusable.row < 0:
                return None
        cache = self._require_cache()
        rows: list[int] = []
        reuse_entries: list[tuple[int, _ReusablePrefix | None]] = []
        try:
            for _idx, request, hit, reusable in group:
                row = self._acquire_active_row()
                self._copy_prefix(reusable.row, row, hit)
                rows.append(row)
                reuse_entries.append((hit, reusable))
            self._record_prefix_reuse_batch(reuse_entries)
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None

        n = len(group)
        next_tokens: list[int | None] = [None] * n
        out_logits: list[Tensor | None] = [None] * n
        suffix_idx = [i for i, (_x, req, hit, _r) in enumerate(group) if len(req.prompt) > hit]
        full_idx = [i for i, (_x, req, hit, _r) in enumerate(group) if len(req.prompt) <= hit]
        try:
            if suffix_idx:
                suffixes = [list(group[i][1].prompt[group[i][2]:]) for i in suffix_idx]
                msl = max(len(s) for s in suffixes)
                padded = [s + [0] * (msl - len(s)) for s in suffixes]
                b = len(suffix_idx)
                input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
                q_lens = torch.tensor([len(s) for s in suffixes], dtype=torch.long, device=self.device)
                seq_lens = torch.tensor([group[i][2] for i in suffix_idx], dtype=torch.long, device=self.device)
                ris = torch.tensor([rows[i] for i in suffix_idx], dtype=torch.long, device=self.device)
                wpos = torch.tensor(
                    [[group[i][2] + j for j in range(msl)] for i in suffix_idx],
                    dtype=torch.long, device=self.device,
                )
                lpos = torch.tensor([len(s) - 1 for s in suffixes], dtype=torch.long, device=self.device)
                logits = None
                packed_fi = getattr(self.model, "prefill_ragged_logits_packed_flashinfer", None)
                if (
                    env_flag("TORCHINFERNO_CONTINUOUS_PACKED_FLASHINFER_PREFILL", False)
                    and packed_fi is not None
                    and getattr(cache, "cache_backend", self.cache_backend) == "flashinfer"
                    and bool(torch.any(q_lens < msl))
                ):
                    seq_lens_full = torch.zeros(
                        max(rows) + 1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for suffix_row, group_index in enumerate(suffix_idx):
                        seq_lens_full[rows[group_index]] = int(seq_lens[suffix_row].item())
                    packed_start_s = time.perf_counter() if self.profile_timings else 0.0
                    try:
                        logits = packed_fi(
                            input_ids,
                            cache,
                            seq_lens=seq_lens_full,
                            q_lens=q_lens,
                            row_indices=ris,
                            logit_positions=lpos,
                        )
                        self._record_packed_flashinfer_prefill(
                            q_lens,
                            model_tokens=int(input_ids.size(0) * input_ids.size(1)),
                            start_s=packed_start_s,
                        )
                    except Exception as exc:
                        warn_optional_failure("serving.packed_flashinfer_reuse", exc)
                        logits = None
                if logits is None:
                    logits = forward_fi(
                        input_ids, cache, seq_lens=seq_lens, q_lens=q_lens,
                        write_positions=wpos, logit_positions=lpos, row_indices=ris,
                    )
                self._record_model_call("prefill", b, tokens=int(q_lens.sum().item()))
                toks = self._sample_logits_for_requests(
                    logits[:, -1, :],
                    [group[i][1] for i in suffix_idx],
                ).detach().cpu().tolist()
                for k, i in enumerate(suffix_idx):
                    next_tokens[i] = int(toks[k])
                    out_logits[i] = logits[k:k + 1]
            for i in full_idx:
                if group[i][3].logits is None:
                    return None
                cached = group[i][3].logits.to(self.device)
                next_tokens[i] = int(
                    self._sample_logits_for_requests(cached[:, -1, :], [group[i][1]]).item()
                )
                out_logits[i] = cached
        except Exception:
            for row in rows:
                self._release_active_row(row)
            return None

        self.stats.prefill_prefix_reuse_batches += 1
        active: list[_ActiveRequest] = []
        for i, (original_index, request, prefix_hit_tokens, _reusable) in enumerate(group):
            row = rows[i]
            self._set_cache_row_seq_len(row, len(request.prompt))
            self._remember_row_seq_len(row, len(request.prompt))
            next_token = int(next_tokens[i])
            if out_logits[i] is not None:
                self._store_reusable_prefix(
                    request.request_id,
                    request.prompt,
                    row,
                    out_logits[i],
                    allow_pinned=self._allow_pinned_full_prompt_store(request),
                )
            state = _ActiveRequest(
                original_index=original_index,
                request=request,
                tokens=[*request.prompt, next_token],
                generated=1,
                row=row,
                last_token=next_token,
                seq_len=len(request.prompt),
                prefix_hit_tokens=prefix_hit_tokens,
                started_step=step,
            )
            self._record_token_event(events, state, next_token, step, finished=self._should_finish_before_decode(state))
            active.append(state)
        return active

    def _try_prefill_logits_graph(self, input_ids: Tensor, cache: object) -> Tensor | None:
        graph = getattr(self.model, "try_prefill_logits_graph", None)
        if not callable(graph):
            return None
        capture_on_miss = env_flag("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", False)
        logits = self._call_prefill_graph(graph, input_ids, cache, capture_on_miss=capture_on_miss)
        if logits is None:
            self._record_static_prefill_graph_miss(input_ids, graph_kind="logits")
            return None
        self.stats.prefill_graph_hits += 1
        return logits

    def _forward_selected_logits(
        self,
        input_ids: Tensor,
        *,
        cache: object,
        logit_positions: Tensor,
    ) -> Tensor | None:
        graph_logits = self._try_prefill_selected_logits_graph(
            input_ids,
            cache,
            logit_positions=logit_positions,
        )
        if graph_logits is not None:
            return graph_logits
        forward = self.model if callable(self.model) else getattr(self.model, "forward", None)
        if not callable(forward):
            raise TypeError("serving model must be callable or expose forward()")
        kwargs: dict[str, object] = {
            "cache": cache,
            "use_cache": True,
            "logit_positions": logit_positions,
        }
        if self._prefer_sharded_logits():
            kwargs["return_last_logits_only"] = False
            kwargs["return_sharded_logits"] = True
        try:
            logits, _cache = forward(input_ids, **kwargs)
        except TypeError:
            return None
        return logits

    def _try_prefill_selected_logits_graph(
        self,
        input_ids: Tensor,
        cache: object,
        *,
        logit_positions: Tensor,
    ) -> Tensor | None:
        graph = getattr(self.model, "try_prefill_selected_logits_graph", None)
        if not callable(graph):
            return None
        capture_on_miss = env_flag(
            "TORCHINFERNO_CONTINUOUS_SELECTED_PREFILL_CAPTURE",
            env_flag("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", self.graph_prefill),
        )
        logits = self._call_prefill_graph(
            graph,
            input_ids,
            cache,
            capture_on_miss=capture_on_miss,
            logit_positions=logit_positions,
        )
        if logits is None:
            self._record_static_prefill_graph_miss(input_ids, graph_kind="selected")
            return None
        self.stats.prefill_graph_hits += 1
        return logits

    def _call_prefill_graph(
        self,
        graph: Callable[..., Tensor | None],
        input_ids: Tensor,
        cache: object,
        *,
        capture_on_miss: bool,
        logit_positions: Tensor | None = None,
    ) -> Tensor | None:
        if self._graph_accepts_capture_on_miss(graph):
            if logit_positions is None:
                return graph(input_ids, cache, capture_on_miss=capture_on_miss)
            return graph(
                input_ids,
                cache,
                logit_positions=logit_positions,
                capture_on_miss=capture_on_miss,
            )
        if not capture_on_miss:
            return None
        if logit_positions is None:
            return graph(input_ids, cache)
        return graph(input_ids, cache, logit_positions=logit_positions)

    def _record_static_prefill_graph_miss(self, input_ids: Tensor, *, graph_kind: str) -> None:
        self.stats.prefill_graph_misses += 1
        self._record_shape_count(
            self.stats.prefill_graph_miss_shape_counts,
            self._static_prefill_graph_shape_key(input_ids, graph_kind=graph_kind),
        )

    @staticmethod
    def _static_prefill_graph_shape_key(input_ids: Tensor, *, graph_kind: str) -> str:
        return f"static_prefill:{graph_kind}:b{int(input_ids.size(0))}:t{int(input_ids.size(1))}"

    def _prefer_sharded_logits(self) -> bool:
        return int(getattr(self.model, "world_size", 1)) > 1 and callable(
            getattr(self.model, "_sample_next_token", None)
        )

    def _record_model_call(
        self,
        kind: str,
        batch_size: int,
        *,
        tokens: int,
        ragged: bool = False,
        active_tokens: int | None = None,
    ) -> None:
        if kind == "prefill":
            self.stats.prefill_model_calls += 1
            self.stats.prefill_batches += 1
            self.stats.prefill_tokens += tokens
        elif kind == "decode":
            active = tokens if active_tokens is None else active_tokens
            self.stats.decode_model_calls += 1
            self.stats.decode_batches += 1
            self.stats.decode_tokens += tokens
            self.stats.decode_active_tokens += active
            if ragged:
                self.stats.ragged_decode_batches += 1
                self.stats.ragged_decode_tokens += tokens
                self.stats.ragged_decode_active_tokens += active
                self.stats.ragged_decode_padding_tokens += max(0, tokens - active)
        elif kind == "unified":
            self.stats.prefill_model_calls += 1
            self.stats.decode_model_calls += 1
            self.stats.prefill_tokens += tokens
            self.stats.decode_tokens += tokens
            self.stats.decode_active_tokens += tokens
        self.stats.max_model_batch_size = max(self.stats.max_model_batch_size, batch_size)

    def _record_shape_count(self, counts: dict[str, int], key: str) -> None:
        if not self.profile_timings:
            return
        counts[key] = counts.get(key, 0) + 1

    def _record_shape_total(self, counts: dict[str, int], key: str, amount: int) -> None:
        if not self.profile_timings:
            return
        counts[key] = counts.get(key, 0) + int(amount)

    def _record_prefill_row_index_mode(
        self,
        shape_key: str,
        *,
        omitted: bool,
        model_rows: int,
    ) -> None:
        rows = max(0, int(model_rows))
        if omitted:
            self.stats.prefill_row_indices_omitted_batches += 1
            self.stats.prefill_row_indices_omitted_rows += rows
            self._record_shape_count(
                self.stats.prefill_shape_row_indices_omitted_batches,
                shape_key,
            )
            self._record_shape_total(
                self.stats.prefill_shape_row_indices_omitted_rows,
                shape_key,
                rows,
            )
            return
        self.stats.prefill_row_indices_indexed_batches += 1
        self.stats.prefill_row_indices_indexed_rows += rows
        self._record_shape_count(
            self.stats.prefill_shape_row_indices_indexed_batches,
            shape_key,
        )
        self._record_shape_total(
            self.stats.prefill_shape_row_indices_indexed_rows,
            shape_key,
            rows,
        )

    def _record_shape_time(
        self,
        timings: dict[str, float],
        key: str,
        elapsed_ms: float,
    ) -> None:
        if not self.profile_timings:
            return
        timings[key] = timings.get(key, 0.0) + float(elapsed_ms)

    def _record_prefill_shape_batch_details(
        self,
        shape_key: str,
        *,
        real_batch: int,
        suffix_lengths: Sequence[int],
    ) -> None:
        if not self.profile_timings:
            return
        self._record_shape_count(
            self.stats.prefill_shape_real_batch_counts,
            f"{shape_key}|real_b{max(0, int(real_batch))}",
        )
        suffix_counts: dict[int, int] = defaultdict(int)
        for suffix_len in suffix_lengths:
            suffix_counts[max(0, int(suffix_len))] += 1
        for suffix_len, count in suffix_counts.items():
            self._record_shape_total(
                self.stats.prefill_shape_suffix_length_counts,
                f"{shape_key}|suffix{suffix_len}",
                count,
            )

    def _record_packed_prefill_candidate(
        self,
        shape_key: str,
        *,
        suffix_lengths: Sequence[int],
        start_lens: Sequence[int],
        model_tokens: int,
        packed_prefill_pattern_key: str | None = None,
    ) -> None:
        group_counts: dict[tuple[int, int], int] = defaultdict(int)
        real_tokens = 0
        for suffix_len, start_len in zip(suffix_lengths, start_lens):
            suffix = max(0, int(suffix_len))
            start = max(0, int(start_len))
            real_tokens += suffix
            group_counts[(start, suffix)] += 1
        saved_tokens = max(0, int(model_tokens) - real_tokens)
        if saved_tokens <= 0:
            return
        groups = len(group_counts)
        self.stats.prefill_packed_candidate_calls += 1
        self.stats.prefill_packed_candidate_tokens += real_tokens
        self.stats.prefill_packed_candidate_model_tokens += int(model_tokens)
        self.stats.prefill_packed_candidate_saved_tokens += saved_tokens
        self.stats.prefill_packed_candidate_groups += groups
        if not (self.profile_timings or _queue_profile_counts_enabled()):
            return
        def record_count(counts: dict[str, int], key: str) -> None:
            counts[key] = counts.get(key, 0) + 1

        def record_total(counts: dict[str, int], key: str, amount: int) -> None:
            counts[key] = counts.get(key, 0) + int(amount)

        record_count(
            self.stats.prefill_packed_candidate_shape_counts,
            shape_key,
        )
        record_total(
            self.stats.prefill_packed_candidate_shape_tokens,
            shape_key,
            real_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_shape_model_tokens,
            shape_key,
            int(model_tokens),
        )
        record_total(
            self.stats.prefill_packed_candidate_shape_saved_tokens,
            shape_key,
            saved_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_shape_groups,
            shape_key,
            groups,
        )
        prior_max_saved = self.stats.prefill_packed_candidate_shape_max_saved_tokens.get(
            shape_key,
            -1,
        )
        prior_max_groups = self.stats.prefill_packed_candidate_shape_max_groups.get(
            shape_key,
            -1,
        )
        if saved_tokens > prior_max_saved or (
            saved_tokens == prior_max_saved and groups > prior_max_groups
        ):
            self.stats.prefill_packed_candidate_shape_max_tokens[shape_key] = real_tokens
            self.stats.prefill_packed_candidate_shape_max_model_tokens[shape_key] = int(
                model_tokens
            )
            self.stats.prefill_packed_candidate_shape_max_saved_tokens[shape_key] = (
                saved_tokens
            )
            self.stats.prefill_packed_candidate_shape_max_groups[shape_key] = groups
        signature_key = _packed_prefill_candidate_signature_from_counts(shape_key, group_counts)
        record_count(
            self.stats.prefill_packed_candidate_signature_counts,
            signature_key,
        )
        record_total(
            self.stats.prefill_packed_candidate_signature_tokens,
            signature_key,
            real_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_signature_model_tokens,
            signature_key,
            int(model_tokens),
        )
        record_total(
            self.stats.prefill_packed_candidate_signature_saved_tokens,
            signature_key,
            saved_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_signature_groups,
            signature_key,
            groups,
        )
        pattern_key = packed_prefill_pattern_key or _packed_prefill_candidate_pattern_from_counts(
            shape_key,
            group_counts,
        )
        record_count(
            self.stats.prefill_packed_candidate_pattern_counts,
            pattern_key,
        )
        record_total(
            self.stats.prefill_packed_candidate_pattern_tokens,
            pattern_key,
            real_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_pattern_model_tokens,
            pattern_key,
            int(model_tokens),
        )
        record_total(
            self.stats.prefill_packed_candidate_pattern_saved_tokens,
            pattern_key,
            saved_tokens,
        )
        record_total(
            self.stats.prefill_packed_candidate_pattern_groups,
            pattern_key,
            groups,
        )
        for (start_len, suffix_len), count in group_counts.items():
            slot_key = _packed_prefill_pattern_slot_key(
                pattern_key,
                start_len=start_len,
                suffix_len=suffix_len,
            )
            self.stats.prefill_packed_candidate_pattern_slot_counts[slot_key] = max(
                self.stats.prefill_packed_candidate_pattern_slot_counts.get(slot_key, 0),
                int(count),
            )

    def _record_packed_flashinfer_prefill(
        self,
        q_lens: Tensor,
        *,
        model_tokens: int,
        start_s: float,
    ) -> None:
        real_tokens = int(q_lens.sum().item())
        saved_tokens = max(0, int(model_tokens) - real_tokens)
        self.stats.prefill_packed_flashinfer_calls += 1
        self.stats.prefill_packed_flashinfer_tokens += real_tokens
        self.stats.prefill_packed_flashinfer_model_tokens += int(model_tokens)
        self.stats.prefill_packed_flashinfer_saved_tokens += saved_tokens
        if self.profile_timings:
            self.stats.prefill_packed_flashinfer_ms += (time.perf_counter() - start_s) * 1000.0

    def _record_prefix_reuse(
        self,
        prefix_hit_tokens: int,
        reusable: _ReusablePrefix | None,
    ) -> None:
        route_kind = (
            self._prefix_reuse_route_kind(reusable.route_id if reusable is not None else None)
            if self.profile_timings
            else None
        )
        self._record_prefix_reuse_total(prefix_hit_tokens, route_kind=route_kind, count=1)

    def _record_prefix_reuse_batch(
        self,
        entries: Iterable[tuple[int, _ReusablePrefix | None]],
    ) -> None:
        request_count = 0
        token_count = 0
        grouped_counts: dict[tuple[int, str], int] | None = defaultdict(int) if self.profile_timings else None
        for prefix_hit_tokens, reusable in entries:
            hit_tokens = int(prefix_hit_tokens)
            request_count += 1
            token_count += hit_tokens
            if grouped_counts is not None:
                route_kind = self._prefix_reuse_route_kind(
                    reusable.route_id if reusable is not None else None
                )
                grouped_counts[(hit_tokens, route_kind)] += 1
        if request_count <= 0:
            return
        self.stats.prefix_reuse_requests += request_count
        self.stats.prefix_reuse_tokens += token_count
        if grouped_counts is None:
            return
        for (hit_tokens, route_kind), count in grouped_counts.items():
            self._record_prefix_reuse_profile_counts(hit_tokens, route_kind=route_kind, count=count)

    def _record_prefix_reuse_total(
        self,
        prefix_hit_tokens: int,
        *,
        route_kind: str | None,
        count: int,
    ) -> None:
        count = int(count)
        if count <= 0:
            return
        hit_tokens = int(prefix_hit_tokens)
        self.stats.prefix_reuse_requests += count
        self.stats.prefix_reuse_tokens += hit_tokens * count
        if not self.profile_timings:
            return
        self._record_prefix_reuse_profile_counts(
            hit_tokens,
            route_kind=route_kind or "none",
            count=count,
        )

    def _record_prefix_reuse_profile_counts(
        self,
        prefix_hit_tokens: int,
        *,
        route_kind: str,
        count: int,
    ) -> None:
        self._record_shape_total(
            self.stats.prefix_reuse_route_counts,
            route_kind,
            count,
        )
        self._record_shape_total(
            self.stats.prefix_reuse_hit_token_counts,
            str(int(prefix_hit_tokens)),
            count,
        )

    def _record_prefix_graph_route_totals(
        self,
        shape_key: str,
        group: Sequence[tuple[int, ServingRequest, int, _ReusablePrefix]],
        suffix_lengths: Sequence[int],
    ) -> None:
        if not self.profile_timings:
            return
        route_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for (_index, _request, prefix_hit_tokens, reusable), suffix_len in zip(
            group,
            suffix_lengths,
        ):
            route_kind = self._prefix_reuse_route_kind(reusable.route_id)
            totals = route_totals[route_kind]
            totals[0] += 1
            totals[1] += max(0, int(suffix_len))
            totals[2] += max(0, int(prefix_hit_tokens))
        for route_kind, (count, active_tokens, reuse_tokens) in route_totals.items():
            route_key = f"{shape_key}|route={route_kind}"
            self._record_shape_total(self.stats.prefill_shape_route_counts, route_key, count)
            self._record_shape_total(
                self.stats.prefill_shape_route_active_tokens,
                route_key,
                active_tokens,
            )
            self._record_shape_total(
                self.stats.prefill_shape_route_reuse_tokens,
                route_key,
                reuse_tokens,
            )

    @staticmethod
    def _prefix_reuse_route_kind(route_id: Hashable | None) -> str:
        if isinstance(route_id, tuple) and route_id and isinstance(route_id[0], str):
            return route_id[0]
        if isinstance(route_id, str):
            return "request_prompt"
        if route_id is None:
            return "unknown"
        return type(route_id).__name__

    def _can_decode_ragged(self, states: list[_ActiveRequest]) -> bool:
        if not self.enable_ragged_decode:
            return False
        if len(states) <= 1:
            return False
        if not hasattr(self, "_uniform_ragged"):
            self._uniform_ragged = env_flag("TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE", False)
        if len({state.seq_len for state in states}) <= 1 and not self._uniform_ragged:
            return False
        return (
            hasattr(self.model, "decode_ragged_logits")
            or hasattr(self.model, "try_decode_ragged_logits_graph")
            or hasattr(self.model, "try_decode_ragged_token_graph")
        )

    def _seq_lens_tensor(self, states: list[_ActiveRequest], *, rows: list[int] | None = None) -> Tensor:
        required = max([state.row for state in states] + list(rows or [0])) + 1
        if len(self._row_seq_lens) >= required:
            seq_lens = list(self._row_seq_lens[:required])
        else:
            seq_lens = [0 for _ in range(required)]
            for row, seq_len in enumerate(self._row_seq_lens):
                if row < required:
                    seq_lens[row] = int(seq_len)
        pad_seq_len = 0
        active_rows: set[int] = set()
        for state in states:
            state.seq_len = self._cache_row_seq_len(state.row, state.seq_len)
            seq_lens[state.row] = state.seq_len
            active_rows.add(state.row)
            pad_seq_len = max(pad_seq_len, state.seq_len)
        for row in rows or ():
            if row not in active_rows and 0 <= row < len(seq_lens) and seq_lens[row] <= 0:
                seq_lens[row] = pad_seq_len
        return torch.tensor(seq_lens, device=self.device, dtype=torch.long)

    def _prefix_prefill_seq_lens_tensor(
        self,
        rows: Sequence[int],
        start_lens: Sequence[int],
        *,
        row_indices: Tensor,
        required: int,
    ) -> Tensor:
        if len(rows) != len(start_lens):
            raise ValueError("rows and start_lens must have the same length")
        total = max(1, int(required))
        scratch = getattr(self, "_prefix_prefill_seq_lens_scratch", None)
        if scratch is None or scratch.device != self.device or scratch.numel() < total:
            scratch = torch.empty(total, dtype=torch.long, device=self.device)
            self._prefix_prefill_seq_lens_scratch = scratch
        scratch[:total].zero_()
        values = self._device_index_tensor(tuple(int(value) for value in start_lens))
        scratch[:total].index_copy_(0, row_indices.to(dtype=torch.long), values)
        return scratch[:total]

    def _cache_row_seq_len(self, row: int, fallback: int) -> int:
        if 0 <= row < len(self._row_seq_lens):
            seq_len = int(self._row_seq_lens[row])
            if seq_len > 0 or fallback <= 0:
                return seq_len
        seq_len = self._cache_row_seq_len_from_cache(row, fallback)
        self._remember_row_seq_len(row, seq_len)
        return seq_len

    def _refresh_row_seq_len_from_cache(self, row: int, fallback: int) -> int:
        seq_len = self._cache_row_seq_len_from_cache(row, fallback)
        self._remember_row_seq_len(row, seq_len)
        return seq_len

    def _cache_row_seq_len_from_cache(self, row: int, fallback: int) -> int:
        cache = self._require_cache()
        layers = tuple(getattr(cache, "layers", ()) or ())
        if layers:
            layer = layers[0]
            seq_len_for_rows = getattr(layer, "seq_len_for_rows", None)
            if callable(seq_len_for_rows):
                try:
                    return int(seq_len_for_rows((row,)))
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
                try:
                    return int(seq_lens[row])
                except Exception:
                    pass
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                return int(getattr(cache_view((row,)), "seq_len"))
            except Exception:
                pass
        return int(fallback)

    def _remember_row_seq_len(self, row: int, seq_len: int) -> None:
        if 0 <= row < len(self._row_seq_lens):
            self._row_seq_lens[row] = int(seq_len)

    def _forget_row_cached_prefix(self, row: int) -> None:
        if 0 <= row < len(self._row_cached_prefixes):
            self._row_cached_prefixes[row] = None

    @staticmethod
    def _warm_row_prefix_copy_skip_enabled() -> bool:
        return env_flag(
            "TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SKIP_WARM_PREFIX_COPY",
            False,
        )

    def _remember_row_cached_prefix(
        self,
        row: int,
        tokens: Sequence[int],
    ) -> None:
        if not (0 <= row < len(self._row_cached_prefixes)):
            return
        max_tokens = env_int(
            "TORCHINFERNO_CONTINUOUS_ROW_CACHED_PREFIX_MAX_TOKENS",
            256,
            minimum=1,
        )
        cached = tuple(int(token) for token in tokens[:max_tokens])
        self._row_cached_prefixes[row] = cached if cached else None

    def _rows_have_cached_prefix(
        self,
        rows: Sequence[int],
        tokens: Sequence[int],
        prefix_len: int,
    ) -> bool:
        if prefix_len <= 0:
            return False
        prefix = tuple(int(token) for token in tokens[:prefix_len])
        if len(prefix) != prefix_len:
            return False
        for row in rows:
            if not (0 <= row < len(self._row_cached_prefixes)):
                return False
            cached = self._row_cached_prefixes[row]
            if cached is None or len(cached) < prefix_len or cached[:prefix_len] != prefix:
                return False
        return True

    def _set_cache_row_seq_len(self, row: int, seq_len: int) -> None:
        self._remember_row_seq_len(row, seq_len)
        cache = self._require_cache()
        # Cheap path FIRST: set the per-layer seq_len list (and per-layer setter)
        # directly. The for_rows-view path below builds an UNCACHED view with a
        # fresh GPU index tensor on every call -- when run once per row in the
        # prefill/decode state loops that allocation dominated wall time (~49% of
        # prefill). The direct list write is pure Python and equivalent for the
        # dense cache; the view path stays as a fallback for backends (e.g.
        # paged) that manage seq_len behind a view.
        layers = tuple(getattr(cache, "layers", ()) or ())
        changed = False
        for layer in layers:
            setter = getattr(layer, "_set_rows_seq_len", None)
            if callable(setter):
                try:
                    setter((row,), int(seq_len))
                    changed = True
                    continue
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
                seq_lens[row] = int(seq_len)
                uniform = getattr(layer, "_uniform_seq_len", None)
                if isinstance(uniform, list) and uniform:
                    uniform[0] = int(seq_len) if all(value == int(seq_len) for value in seq_lens) else None
                changed = True
        if changed:
            return
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                view = self._cache_view([row])
                setter = getattr(view, "set_seq_len", None)
                if callable(setter):
                    setter(int(seq_len))
                    return
            except Exception:
                pass
        seq_lens = getattr(cache, "_seq_lens", None)
        if isinstance(seq_lens, list) and 0 <= row < len(seq_lens):
            seq_lens[row] = int(seq_len)

    def _set_cache_rows_seq_len(self, rows: list[int], seq_len: int) -> None:
        if not rows:
            return
        seq_len = int(seq_len)
        for row in rows:
            self._remember_row_seq_len(row, seq_len)
        cache = self._require_cache()
        row_tuple = tuple(int(row) for row in rows)
        layers = tuple(getattr(cache, "layers", ()) or ())
        changed = False
        for layer in layers:
            setter = getattr(layer, "_set_rows_seq_len", None)
            if callable(setter):
                try:
                    setter(row_tuple, seq_len)
                    changed = True
                    continue
                except Exception:
                    pass
            seq_lens = getattr(layer, "_seq_lens", None)
            if isinstance(seq_lens, list):
                for row in row_tuple:
                    if 0 <= row < len(seq_lens):
                        seq_lens[row] = seq_len
                        changed = True
                uniform = getattr(layer, "_uniform_seq_len", None)
                if isinstance(uniform, list) and uniform:
                    uniform[0] = seq_len if all(value == seq_len for value in seq_lens) else None
        if changed:
            return
        cache_view = getattr(cache, "for_rows", None)
        if callable(cache_view):
            try:
                view = self._cache_view(row_tuple)
                setter = getattr(view, "set_seq_len", None)
                if callable(setter):
                    setter(seq_len)
                    return
            except Exception:
                pass
        seq_lens = getattr(cache, "_seq_lens", None)
        if isinstance(seq_lens, list):
            for row in row_tuple:
                if 0 <= row < len(seq_lens):
                    seq_lens[row] = seq_len

    def _set_cache_row_seq_lens(
        self,
        rows: Sequence[int],
        seq_lens: Sequence[int],
    ) -> None:
        if len(rows) != len(seq_lens):
            raise ValueError("rows and seq_lens must have the same length")
        if not rows:
            return
        grouped_rows: dict[int, list[int]] = defaultdict(list)
        for row, seq_len in zip(rows, seq_lens):
            grouped_rows[int(seq_len)].append(int(row))
        for seq_len, group_rows in grouped_rows.items():
            self._set_cache_rows_seq_len(group_rows, seq_len)

    def _try_ragged_token_graph(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor | None,
        *,
        temperature: float | None = None,
    ) -> Tensor | None:
        sampling_temperature = float(self.temperature if temperature is None else temperature)
        fi_decode_mode = self._fi_decode_graph_mode
        use_fi_decode = fi_decode_mode == "always" or (
            fi_decode_mode == "sampled"
            and sampling_temperature > 0.0
            and self._sampled_fi_decode_enabled_for_request()
        )
        fi_graphs = (
            getattr(self.model, "_fi_decode_graphs", None)
            if use_fi_decode
            else None
        )
        if fi_graphs:
            batch = input_ids.size(0)
            if row_indices is None:
                row_indices = self._device_index_tensor(tuple(range(batch)))
            bucket = 1 << (batch - 1).bit_length() if batch > 1 else 1
            entry = fi_graphs.get(bucket)
            if entry is not None:
                graph, dw, s_ids, s_wp, s_ri, s_logits, nqo, nkv, hd, ms, qd = entry
                s_ids[:batch].copy_(input_ids)
                s_ri[:batch].copy_(row_indices)
                if batch < bucket:
                    s_ids[batch:] = 0
                    s_ri[batch:] = 0
                fi_bufs = getattr(self, "_fi_bufs", {})
                if bucket not in fi_bufs:
                    fi_bufs[bucket] = (
                        torch.arange(bucket + 1, dtype=torch.int32, device=self.device),
                        torch.ones(bucket, dtype=torch.int32, device=self.device),
                    )
                    self._fi_bufs = fi_bufs
                indptr, lpl = fi_bufs[bucket]
                lpl.fill_(1)
                indices = s_ri.to(dtype=torch.int32)
                row_sl = seq_lens[row_indices[:batch].long()]
                s_wp[:batch, 0].copy_(row_sl)
                lpl[:batch] = (row_sl + 1).to(torch.int32)
                if batch < bucket:
                    s_wp[batch:] = 0
                dw.plan(indptr=indptr, indices=indices, last_page_len=lpl,
                        num_qo_heads=nqo, num_kv_heads=nkv, head_dim=hd, page_size=ms, q_data_type=qd)
                graph.replay()
                last_logits = s_logits[:batch, -1, :]
                self._last_ragged_decode_logits = last_logits
                return self._sample_logits_with_temperature(last_logits, sampling_temperature)
        if sampling_temperature > 0.0:
            return None
        decode_graph = getattr(self.model, "try_decode_ragged_token_graph", None)
        if decode_graph is None:
            return None
        graph_start_s = time.perf_counter() if self.profile_timings else 0.0
        token = self._call_decode_graph(
            decode_graph,
            input_ids,
            self._require_cache(),
            seq_lens=seq_lens,
            row_indices=row_indices,
            temperature=sampling_temperature,
            capture_on_miss=self._decode_capture_on_miss(),
        )
        if token is None:
            self._record_decode_graph_miss(
                input_ids,
                row_indices,
                graph_kind="token",
            )
        else:
            self._record_decode_graph_capture_state(
                getattr(self.model, "_last_ragged_decode_graph_captured", None),
                getattr(self.model, "_last_ragged_decode_graph_key", None),
                input_ids,
                row_indices,
                graph_start_s,
                graph_kind="token",
            )
        return token

    def _record_decode_graph_capture_state(
        self,
        captured: object,
        graph_key: object,
        input_ids: Tensor,
        row_indices: Tensor | None,
        graph_start_s: float,
        *,
        graph_kind: str,
    ) -> None:
        if not isinstance(captured, bool):
            return
        elapsed_ms = (time.perf_counter() - graph_start_s) * 1000.0 if self.profile_timings else 0.0
        shape_key = self._ragged_decode_graph_shape_key(
            input_ids,
            row_indices,
            graph_kind=graph_kind,
            graph_key=graph_key,
        )
        if captured:
            self.stats.decode_graph_captures += 1
            self.stats.decode_graph_capture_ms += elapsed_ms
            self._record_shape_count(self.stats.decode_graph_capture_shape_counts, shape_key)
            if self.profile_timings:
                self._record_shape_time(
                    self.stats.decode_graph_capture_shape_ms,
                    shape_key,
                    elapsed_ms,
                )
        else:
            self.stats.decode_graph_replays += 1
            self.stats.decode_graph_replay_ms += elapsed_ms
            if self.profile_timings:
                self._record_shape_time(
                    self.stats.decode_graph_replay_shape_ms,
                    shape_key,
                    elapsed_ms,
                )

    def _record_decode_graph_miss(
        self,
        input_ids: Tensor,
        row_indices: Tensor | None,
        *,
        graph_kind: str,
    ) -> None:
        self.stats.decode_graph_misses += 1
        self._record_shape_count(
            self.stats.decode_graph_miss_shape_counts,
            self._ragged_decode_graph_shape_key(input_ids, row_indices, graph_kind=graph_kind),
        )

    @staticmethod
    def _ragged_decode_graph_shape_key(
        input_ids: Tensor,
        row_indices: Tensor | None,
        *,
        graph_kind: str,
        graph_key: object | None = None,
    ) -> str:
        key = (
            f"ragged_decode:{graph_kind}:b{int(input_ids.size(0))}:"
            f"rows{1 if row_indices is not None else 0}"
        )
        cache_suffix = _ragged_decode_graph_key_cache_suffix(graph_key)
        if cache_suffix is not None:
            key = f"{key}:{cache_suffix}"
        symm_suffix = _ragged_decode_graph_key_symm_suffix(graph_key)
        if symm_suffix is not None:
            key = f"{key}:{symm_suffix}"
        return key

    def _sampled_fi_decode_enabled_for_request(self) -> bool:
        max_tokens = self.max_generation_tokens
        if max_tokens is None:
            return True
        limit = max(0, int(self._sampled_fi_decode_max_tokens))
        return limit > 0 and int(max_tokens) <= limit

    def _try_static_token_graph(
        self,
        input_ids: Tensor,
        cache: object,
        *,
        temperature: float | None = None,
    ) -> Tensor | None:
        decode_graph = getattr(self.model, "try_decode_one_token_graph", None)
        if decode_graph is None:
            self._report_static_graph_miss(input_ids, cache, "no_token_graph")
            return None
        sampling_temperature = float(self.temperature if temperature is None else temperature)
        token = self._call_decode_graph(
            decode_graph,
            input_ids,
            cache,
            temperature=sampling_temperature,
            capture_on_miss=self._decode_capture_on_miss(),
        )
        if token is None:
            self._record_static_decode_graph_miss(input_ids, cache, graph_kind="token")
            self._report_static_graph_miss(input_ids, cache, "token_graph_returned_none")
        return token

    def _static_decode_logits(self, input_ids: Tensor, cache: object) -> Tensor:
        decode_graph = getattr(self.model, "try_decode_one_token_logits_graph", None)
        if decode_graph is not None:
            logits = self._call_decode_graph(
                decode_graph,
                input_ids,
                cache,
                capture_on_miss=self._decode_capture_on_miss(),
            )
            if logits is not None:
                self.stats.decode_graph_hits += 1
                return logits
            self._record_static_decode_graph_miss(input_ids, cache, graph_kind="logits")
            self._report_static_graph_miss(input_ids, cache, "logits_graph_returned_none")
        else:
            self._report_static_graph_miss(input_ids, cache, "no_logits_graph")
        logits, _ = self._forward_model(input_ids, cache=cache, use_cache=True)
        return logits

    def _report_static_graph_miss(self, input_ids: Tensor, cache: object, reason: str) -> None:
        if self._reported_static_graph_miss or not env_flag("TORCHINFERNO_CUDAGRAPH_DECODE_DEBUG", False):
            return
        self._reported_static_graph_miss = True
        cache_seq_len: object
        try:
            cache_seq_len = getattr(cache, "seq_len", None)
        except Exception as exc:
            cache_seq_len = f"error:{exc!r}"
        print(
            "continuous_decode_graph_miss "
            f"reason={reason} "
            f"batch={int(input_ids.size(0))} "
            f"input_cuda={bool(input_ids.is_cuda)} "
            f"temperature={float(self.temperature)} "
            f"cache_seq_len={cache_seq_len} "
            f"token_failed={getattr(self.model, '_decode_graph_failed', None)} "
            f"logits_failed={getattr(self.model, '_decode_logits_graph_failed', None)}",
            flush=True,
        )

    def _record_static_decode_graph_miss(self, input_ids: Tensor, cache: object, *, graph_kind: str) -> None:
        self.stats.decode_graph_misses += 1
        self._record_shape_count(
            self.stats.decode_graph_miss_shape_counts,
            self._static_decode_graph_shape_key(input_ids, cache, graph_kind=graph_kind),
        )

    @staticmethod
    def _static_decode_graph_shape_key(input_ids: Tensor, cache: object, *, graph_kind: str) -> str:
        key = f"static_decode:{graph_kind}:b{int(input_ids.size(0))}"
        try:
            seq_len = getattr(cache, "seq_len", None)
        except Exception:
            seq_len = None
        if isinstance(seq_len, int):
            return f"{key}:s{seq_len}"
        return key

    def _ragged_decode_logits(
        self,
        input_ids: Tensor,
        seq_lens: Tensor,
        row_indices: Tensor | None,
    ) -> Tensor:
        decode_graph = getattr(self.model, "try_decode_ragged_logits_graph", None)
        if decode_graph is not None:
            graph_start_s = time.perf_counter() if self.profile_timings else 0.0
            logits = self._call_decode_graph(
                decode_graph,
                input_ids,
                self._require_cache(),
                seq_lens=seq_lens,
                row_indices=row_indices,
                capture_on_miss=self._decode_capture_on_miss(),
            )
            if logits is not None:
                self.stats.decode_graph_hits += 1
                self._record_decode_graph_capture_state(
                    getattr(self.model, "_last_ragged_decode_logits_graph_captured", None),
                    getattr(self.model, "_last_ragged_decode_logits_graph_key", None),
                    input_ids,
                    row_indices,
                    graph_start_s,
                    graph_kind="logits",
                )
                return logits
            self._record_decode_graph_miss(
                input_ids,
                row_indices,
                graph_kind="logits",
            )
        decode = getattr(self.model, "decode_ragged_logits", None)
        if decode is None:
            raise RuntimeError("model does not support ragged decode")
        return decode(input_ids, self._require_cache(), seq_lens=seq_lens, row_indices=row_indices)

    def _request_temperature(self, request: ServingRequest) -> float:
        if request.temperature is None:
            return float(self.temperature)
        return float(request.temperature)

    def _state_temperature(self, state: _ActiveRequest) -> float:
        return self._request_temperature(state.request)

    def _shared_temperature(self, temperatures: Sequence[float]) -> float | None:
        if not temperatures:
            return float(self.temperature)
        first = float(temperatures[0])
        if all(float(temperature) == first for temperature in temperatures[1:]):
            return first
        return None

    def _shared_temperature_for_requests(
        self,
        requests: Sequence[ServingRequest],
        *,
        limit: int | None = None,
    ) -> float | None:
        count = len(requests) if limit is None else min(len(requests), max(0, int(limit)))
        if count <= 0:
            return float(self.temperature)
        first = self._request_temperature(requests[0])
        for index in range(1, count):
            if self._request_temperature(requests[index]) != first:
                return None
        return first

    def _shared_temperature_for_states(
        self,
        states: Sequence[_ActiveRequest],
        *,
        limit: int | None = None,
    ) -> float | None:
        count = len(states) if limit is None else min(len(states), max(0, int(limit)))
        if count <= 0:
            return float(self.temperature)
        first = self._state_temperature(states[0])
        for index in range(1, count):
            if self._state_temperature(states[index]) != first:
                return None
        return first

    def _sample_logits_with_temperature(self, logits: Tensor, temperature: float) -> Tensor:
        sampler = getattr(self.model, "_sample_next_token", None)
        if callable(sampler):
            return sampler(logits, float(temperature)).to(self.device)
        return sample_next_token(logits, float(temperature)).to(self.device)

    def _sample_logits(self, logits: Tensor) -> Tensor:
        return self._sample_logits_with_temperature(logits, float(self.temperature))

    def _sample_logits_for_temperatures(
        self,
        logits: Tensor,
        temperatures: Sequence[float],
    ) -> Tensor:
        row_count = int(logits.size(0))
        if row_count <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        active_temperatures = [float(temperature) for temperature in temperatures[:row_count]]
        active_count = len(active_temperatures)
        if active_count == row_count:
            shared = self._shared_temperature(active_temperatures)
            if shared is not None:
                return self._sample_logits_with_temperature(logits, shared)

        sampled = torch.empty((row_count,), dtype=torch.long, device=self.device)
        by_temperature: dict[float, list[int]] = defaultdict(list)
        for row_index, temperature in enumerate(active_temperatures):
            by_temperature[temperature].append(row_index)
        for temperature, row_indices in by_temperature.items():
            index = torch.tensor(row_indices, dtype=torch.long, device=self.device)
            sampled.index_copy_(
                0,
                index,
                self._sample_logits_with_temperature(
                    logits.index_select(0, index.to(logits.device)),
                    temperature,
                ),
            )
        if active_count < row_count:
            tail = self._sample_logits_with_temperature(
                logits[active_count:],
                float(self.temperature),
            )
            sampled[active_count:].copy_(tail)
        return sampled

    def _sample_logits_for_requests(
        self,
        logits: Tensor,
        requests: Sequence[ServingRequest],
    ) -> Tensor:
        row_count = int(logits.size(0))
        if row_count <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if len(requests) >= row_count:
            shared = self._shared_temperature_for_requests(requests, limit=row_count)
            if shared is not None:
                return self._sample_logits_with_temperature(logits, shared)
        return self._sample_logits_for_temperatures(
            logits,
            [self._request_temperature(request) for request in requests],
        )

    def _sample_logits_for_states(
        self,
        logits: Tensor,
        states: Sequence[_ActiveRequest],
    ) -> Tensor:
        row_count = int(logits.size(0))
        if row_count <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if len(states) >= row_count:
            shared = self._shared_temperature_for_states(states, limit=row_count)
            if shared is not None:
                return self._sample_logits_with_temperature(logits, shared)
        return self._sample_logits_for_temperatures(
            logits,
            [self._state_temperature(state) for state in states],
        )

    def _decode_capture_on_miss(self) -> bool:
        if self._decode_capture_on_miss_override is not None:
            return bool(self._decode_capture_on_miss_override)
        # TP online workers are command-driven; a rank-local decode graph miss
        # must not insert an unscheduled capture-coordination collective.
        if self._model_world_size() > 1:
            return False
        if self._generated_prefix_cache_enabled():
            return False
        return True

    def _model_world_size(self) -> int:
        try:
            return max(1, int(getattr(self.model, "world_size", 1)))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _graph_signature_cache_key(graph: Callable[..., object]) -> object:
        key = getattr(graph, "__func__", graph)
        try:
            hash(key)
        except TypeError:
            return id(key)
        return key

    def _graph_accepts_capture_on_miss(self, graph: Callable[..., object]) -> bool:
        key = self._graph_signature_cache_key(graph)
        cached = self._graph_capture_on_miss_support.get(key)
        if cached is not None:
            return cached
        try:
            accepts = "capture_on_miss" in signature(graph).parameters
        except (TypeError, ValueError):
            accepts = False
        self._graph_capture_on_miss_support[key] = accepts
        return accepts

    def _call_decode_graph(
        self,
        graph: Callable[..., Tensor | None],
        input_ids: Tensor,
        cache: object,
        *,
        capture_on_miss: bool,
        temperature: float | None = None,
        seq_lens: Tensor | None = None,
        row_indices: Tensor | None = None,
    ) -> Tensor | None:
        kwargs: dict[str, object] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seq_lens is not None:
            kwargs["seq_lens"] = seq_lens
            kwargs["row_indices"] = row_indices
        if self._graph_accepts_capture_on_miss(graph):
            kwargs["capture_on_miss"] = capture_on_miss
        return graph(input_ids, cache, **kwargs)

    @staticmethod
    def _record_token_event(
        events: list[ServingTokenEvent] | None,
        state: _ActiveRequest,
        token: int,
        step: int,
        *,
        finished: bool,
    ) -> None:
        if events is None:
            return
        events.append(
            ServingTokenEvent(
                request_id=state.request.request_id,
                token=int(token),
                step=step,
                generated=state.generated,
                finished=finished,
            )
        )

    def _release_online_prefill_finished(
        self,
        active: list[_ActiveRequest],
        step: int,
    ) -> list[_ActiveRequest]:
        live: list[_ActiveRequest] = []
        for state in active:
            if self._should_finish_before_decode(state):
                self._finish_and_release(state, step)
            else:
                live.append(state)
        return live

    @staticmethod
    def _should_finish_before_decode(state: _ActiveRequest) -> bool:
        if state.request.is_stop_token(state.last_token):
            return True
        return state.generated >= state.request.max_new_tokens

    @staticmethod
    def _should_finish_after_decode(state: _ActiveRequest) -> bool:
        if state.request.is_stop_token(state.last_token):
            return True
        return state.generated >= state.request.max_new_tokens


def _common_prefix_token_count(prompts: list[tuple[int, ...]]) -> int:
    if len(prompts) <= 1:
        return 0
    min_len = min((len(prompt) for prompt in prompts), default=0)
    if min_len <= 1:
        return 0
    prefix_tokens = 0
    for offset in range(min_len):
        token = prompts[0][offset]
        if any(prompt[offset] != token for prompt in prompts[1:]):
            break
        prefix_tokens += 1
    return prefix_tokens
