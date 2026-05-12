from __future__ import annotations

from torchinferno.openai_warmup import (
    warmup_ragged_decode_batch_sizes,
    warmup_ragged_decode_cache_token_counts,
    warmup_ragged_decode_row_counts,
    warmup_temperature_batch_sizes,
    warmup_temperature_prompt_token_counts,
)


def test_temperature_warmup_covers_self_consistency_batch(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS", raising=False)

    assert {15, 16, 64}.issubset(set(warmup_temperature_batch_sizes()))
    assert 55 in set(warmup_temperature_prompt_token_counts())


def test_ragged_decode_warmup_covers_high_concurrency_shapes(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_ROW_COUNTS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_CACHE_TOKENS", raising=False)

    assert 64 in set(warmup_ragged_decode_batch_sizes())
    assert {16, 32, 64}.issubset(set(warmup_ragged_decode_row_counts()))
    assert {256, 512}.issubset(set(warmup_ragged_decode_cache_token_counts()))
