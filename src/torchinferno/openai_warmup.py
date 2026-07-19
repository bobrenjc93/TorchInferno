from __future__ import annotations

import os
from collections.abc import Iterable


def warmup_prompt_token_counts(default_prompt_tokens: int) -> tuple[int, ...]:
    raw = os.environ.get("TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKEN_BUCKETS")
    if raw is not None:
        return parse_positive_int_csv(raw) or (max(1, int(default_prompt_tokens)),)
    return _dedupe_positive((default_prompt_tokens, 16, 32, 64, 128, 256))


def warmup_temperature_prompt_token_counts() -> tuple[int, ...]:
    configured = os.environ.get("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS")
    return parse_positive_int_csv(configured) if configured is not None else _power_of_two_buckets(64, minimum=16)


def warmup_temperature_batch_sizes() -> tuple[int, ...]:
    configured = os.environ.get("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES")
    return parse_positive_int_csv(configured) if configured is not None else _power_of_two_buckets(64)


def warmup_ragged_decode_batch_sizes() -> tuple[int, ...]:
    configured = os.environ.get("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_BATCH_SIZES")
    return parse_positive_int_csv(configured) if configured is not None else _power_of_two_buckets(64)


def warmup_ragged_decode_row_counts() -> tuple[int, ...]:
    configured = os.environ.get("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_ROW_COUNTS")
    if configured is not None:
        return parse_positive_int_csv(configured)
    return tuple(range(1, 9)) + tuple(range(16, 65, 8))


def warmup_ragged_decode_cache_token_counts() -> tuple[int, ...]:
    return parse_positive_int_csv(
        os.environ.get("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_CACHE_TOKENS", "256,512")
    )


def warmup_ragged_decode_extra_cache_specs() -> tuple[tuple[int, int], ...]:
    specs = parse_nonnegative_positive_int_pair_csv(
        os.environ.get("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_EXTRA_CACHE_SPECS", "64:1024")
    )
    return tuple((batch, cache_tokens) for batch, cache_tokens in specs if batch > 0)


def warmup_ragged_decode_prompt_tokens(default_prompt_tokens: int) -> int:
    return max(
        1,
        int(os.environ.get("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_PROMPT_TOKENS", default_prompt_tokens)),
    )


def warmup_prefill_cache_token_counts() -> tuple[int, ...]:
    return parse_positive_int_csv(
        os.environ.get("TORCHINFERNO_OPENAI_WARMUP_PREFILL_CACHE_TOKENS", "128,256,512,1024")
    )


def warmup_prefix_suffix_token_counts() -> tuple[tuple[int, int], ...]:
    raw = os.environ.get(
        "TORCHINFERNO_OPENAI_WARMUP_PREFIX_SUFFIX_TOKENS",
        "32:16,64:16,128:32,256:32",
    )
    return parse_nonnegative_positive_int_pair_csv(raw)


def warmup_prefix_suffix_cache_token_counts() -> tuple[int, ...]:
    return parse_positive_int_csv(
        os.environ.get("TORCHINFERNO_OPENAI_WARMUP_PREFIX_SUFFIX_CACHE_TOKENS", "128,256,512,1024")
    )


def parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    return _dedupe_positive(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_nonnegative_positive_int_pair_csv(raw: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        prefix, separator, suffix = stripped.partition(":")
        if not separator:
            raise ValueError(f"expected prefix:suffix pair, got {stripped!r}")
        pairs.append((max(0, int(prefix.strip())), max(1, int(suffix.strip()))))
    return tuple(dict.fromkeys(pairs))


def _dedupe_positive(counts: Iterable[int]) -> tuple[int, ...]:
    values = [max(1, int(count)) for count in counts]
    return tuple(dict.fromkeys(values))


def _power_of_two_buckets(maximum: int, *, minimum: int = 1) -> tuple[int, ...]:
    maximum = max(1, int(maximum))
    minimum = max(1, int(minimum))
    values: list[int] = []
    bucket = 1
    while bucket < minimum:
        bucket *= 2
    while bucket <= maximum:
        values.append(bucket)
        bucket *= 2
    if not values or values[-1] != maximum:
        values.append(maximum)
    return tuple(values)
