from __future__ import annotations

from torchinferno.openai_warmup import warmup_temperature_batch_sizes


def test_temperature_warmup_covers_self_consistency_batch(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", raising=False)

    assert 16 in set(warmup_temperature_batch_sizes())
