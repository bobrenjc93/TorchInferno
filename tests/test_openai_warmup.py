from __future__ import annotations

from torchinferno.openai_warmup import warmup_temperature_batch_sizes, warmup_temperature_prompt_token_counts


def test_temperature_warmup_covers_self_consistency_batch(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS", raising=False)

    assert 64 in set(warmup_temperature_batch_sizes())
    assert 55 in set(warmup_temperature_prompt_token_counts())
