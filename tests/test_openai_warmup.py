from __future__ import annotations

from torchinferno.openai_warmup import (
    warmup_ragged_decode_batch_sizes,
    warmup_ragged_decode_cache_token_counts,
    warmup_ragged_decode_extra_cache_specs,
    warmup_ragged_decode_row_counts,
    warmup_temperature_batch_sizes,
    warmup_temperature_prompt_token_counts,
)


def test_temperature_warmup_uses_power_of_two_buckets(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS", raising=False)

    assert warmup_temperature_batch_sizes() == (1, 2, 4, 8, 16, 32, 64)
    assert warmup_temperature_prompt_token_counts() == (16, 32, 64)


def test_ragged_decode_warmup_covers_high_concurrency_shapes(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_ROW_COUNTS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_CACHE_TOKENS", raising=False)

    assert warmup_ragged_decode_batch_sizes() == (1, 2, 4, 8, 16, 32, 64)
    assert warmup_ragged_decode_row_counts() == (
        1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64
    )
    assert {256, 512}.issubset(set(warmup_ragged_decode_cache_token_counts()))
    assert (64, 1024) in set(warmup_ragged_decode_extra_cache_specs())
