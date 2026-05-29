import os
import subprocess
import sys

import pytest
import torch

import torchinferno.models.deepseek_v32.model as deepseek_mod
from torchinferno.models.deepseek_v32 import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.runtime.serving import ContinuousBatchEngine, ServingRequest


class _ToyCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        rows: tuple[int, ...] | None = None,
        seq_lens: list[int] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self._rows = tuple(range(batch_size)) if rows is None else rows
        self._seq_lens = [0 for _ in range(batch_size)] if seq_lens is None else seq_lens

    @property
    def seq_len(self) -> int:
        if not self._rows:
            return 0
        seq_len = self._seq_lens[self._rows[0]]
        if any(self._seq_lens[row] != seq_len for row in self._rows):
            raise ValueError("selected cache rows must have the same sequence length")
        return seq_len

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "_ToyCache":
        physical = tuple(self._rows[int(row)] for row in rows)
        return _ToyCache(len(physical), self.max_seq_len, rows=physical, seq_lens=self._seq_lens)

    def clear_row(self, row: int) -> None:
        self._seq_lens[self._rows[row]] = 0

    def copy_prefix_from(self, source: "_ToyCache", tokens: int, *, source_row: int = 0, dest_row: int = 0) -> None:
        source_physical = source._rows[source_row]
        dest_physical = self._rows[dest_row]
        if tokens > source._seq_lens[source_physical]:
            raise ValueError("prefix length exceeds source row")
        self._seq_lens[dest_physical] = tokens

    def advance_rows(self, rows: list[int], tokens: int) -> None:
        for row in rows:
            self._seq_lens[row] += tokens


class _RaggedGraphToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ragged_logits_graph_calls = 0
        self.ragged_eager_calls = 0

    def allocate_cache(self, batch_size: int, max_seq_len: int | None = None, **kwargs) -> _ToyCache:
        return _ToyCache(batch_size, max_seq_len or 1)

    def forward(self, input_ids, *, cache, use_cache=False):
        del use_cache
        rows = list(cache._rows[: input_ids.size(0)])
        cache.advance_rows(rows, input_ids.size(1))
        return self._logits(input_ids[:, -1] + 1), cache

    def try_decode_ragged_logits_graph(self, input_ids, cache, *, seq_lens, row_indices):
        self.ragged_logits_graph_calls += 1
        cache.advance_rows(row_indices.detach().cpu().tolist(), 1)
        return self._logits(input_ids[:, -1] + 1)

    def decode_ragged_logits(self, input_ids, cache, *, seq_lens, row_indices):
        del seq_lens
        self.ragged_eager_calls += 1
        cache.advance_rows(row_indices.detach().cpu().tolist(), 1)
        return self._logits(input_ids[:, -1] + 1)

    def _sample_next_token(self, logits, temperature):
        del temperature
        return torch.argmax(logits, dim=-1)

    def _logits(self, next_ids):
        next_ids = next_ids.remainder(self.vocab_size).to(torch.long)
        logits = torch.zeros((next_ids.numel(), 1, self.vocab_size), device=next_ids.device)
        logits[torch.arange(next_ids.numel(), device=next_ids.device), 0, next_ids] = 1.0
        return logits


class _StaticDecodeGraphToyModel(_RaggedGraphToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.static_token_graph_calls = 0

    def try_decode_one_token_graph(self, input_ids, cache, *, temperature=0.0):
        del temperature
        self.static_token_graph_calls += 1
        rows = list(cache._rows[: input_ids.size(0)])
        cache.advance_rows(rows, 1)
        return torch.argmax(self._logits(input_ids[:, -1] + 1)[:, -1, :], dim=-1)


class _SeqLenCheckingSkewedToyModel(_RaggedGraphToyModel):
    def forward(self, input_ids, *, cache, use_cache=False):
        del use_cache
        _ = cache.seq_len
        rows = list(cache._rows[: input_ids.size(0)])
        for offset, row in enumerate(rows):
            cache._seq_lens[row] += input_ids.size(1)
            if input_ids.size(1) > 1 and offset == 1:
                cache._seq_lens[row] += 1
        return self._logits(input_ids[:, -1] + 1), cache


class _RaggedNoSeqLenUpdateToyModel(_RaggedGraphToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.decode_positions: list[list[int]] = []

    def try_decode_ragged_logits_graph(self, input_ids, cache, *, seq_lens, row_indices):
        del cache
        self.decode_positions.append(seq_lens.index_select(0, row_indices).detach().cpu().tolist())
        return self._logits(input_ids[:, -1] + 1)

    def decode_ragged_logits(self, input_ids, cache, *, seq_lens, row_indices):
        del cache
        self.decode_positions.append(seq_lens.index_select(0, row_indices).detach().cpu().tolist())
        return self._logits(input_ids[:, -1] + 1)


class _RaggedDecodeShapeRecordingToyModel(_RaggedGraphToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.decode_shapes: list[tuple[int, list[int]]] = []

    def try_decode_ragged_logits_graph(self, input_ids, cache, *, seq_lens, row_indices):
        del seq_lens
        rows = row_indices.detach().cpu().tolist()
        self.decode_shapes.append((int(input_ids.size(0)), [int(row) for row in rows]))
        cache.advance_rows(rows, 1)
        return self._logits(input_ids[:, -1] + 1)


class _SelectedLogitsToyModel(_RaggedGraphToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.selected_positions: list[list[int]] = []

    def forward(
        self,
        input_ids,
        *,
        cache,
        use_cache=False,
        logit_positions=None,
        return_last_logits_only=False,
        return_sharded_logits=False,
    ):
        del use_cache, return_last_logits_only, return_sharded_logits
        rows = list(cache._rows[: input_ids.size(0)])
        cache.advance_rows(rows, input_ids.size(1))
        if logit_positions is None:
            return self._logits(input_ids[:, -1] + 1), cache
        positions = logit_positions.to(input_ids.device)
        self.selected_positions.append(positions.detach().cpu().tolist())
        row_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return self._logits(input_ids[row_indices, positions] + 1), cache

    def _ragged_prefill_compute(self, input_ids, logit_positions):
        positions = logit_positions.to(input_ids.device)
        self.selected_positions.append(positions.detach().cpu().tolist())
        idx = torch.arange(input_ids.size(0), device=input_ids.device)
        return self._logits(input_ids[idx, positions] + 1)

    def try_prefill_ragged_logits_graph(
        self, input_ids, cache, *, seq_lens, row_indices, logit_positions,
        context_len=None, src_prefix_row=None, capture_on_miss=True,
    ):
        del seq_lens, context_len, src_prefix_row, capture_on_miss
        cache.advance_rows(row_indices.detach().cpu().tolist(), input_ids.size(1))
        return self._ragged_prefill_compute(input_ids, logit_positions)

    def prefill_ragged_logits(
        self, input_ids, cache, *, seq_lens, row_indices, logit_positions, context_len=None, src_prefix_row=None
    ):
        del seq_lens, context_len, src_prefix_row
        cache.advance_rows(row_indices.detach().cpu().tolist(), input_ids.size(1))
        return self._ragged_prefill_compute(input_ids, logit_positions)


class _PrefillLogitsGraphToyModel(_RaggedGraphToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.prefill_capture_flags: list[bool] = []
        self.forward_calls = 0

    def forward(self, input_ids, *, cache, use_cache=False):
        self.forward_calls += 1
        return super().forward(input_ids, cache=cache, use_cache=use_cache)

    def try_prefill_logits_graph(self, input_ids, cache, *, capture_on_miss=True):
        self.prefill_capture_flags.append(bool(capture_on_miss))
        if not capture_on_miss:
            return None
        return super().forward(input_ids, cache=cache, use_cache=True)[0]


class _SelectedLogitsGraphToyModel(_SelectedLogitsToyModel):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__(vocab_size)
        self.selected_capture_flags: list[bool] = []

    def try_prefill_selected_logits_graph(
        self,
        input_ids,
        cache,
        *,
        logit_positions,
        capture_on_miss=True,
    ):
        self.selected_capture_flags.append(bool(capture_on_miss))
        if not capture_on_miss:
            return None
        return super().forward(
            input_ids,
            cache=cache,
            use_cache=True,
            logit_positions=logit_positions,
        )[0]


class _DeviceResidentToyWrapper:
    def __init__(self, model: _RaggedGraphToyModel) -> None:
        self.model = model
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        self.model.eval()
        return self

    def allocate_cache(self, batch_size: int, max_seq_len: int):
        return self.model.allocate_cache(batch_size, max_seq_len=max_seq_len)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class _ShardedForwardToyWrapper(_DeviceResidentToyWrapper):
    world_size = 2

    def __init__(self, model: _RaggedGraphToyModel) -> None:
        super().__init__(model)
        self.return_sharded_logits_values: list[bool] = []
        self.return_last_logits_only_values: list[bool] = []

    def forward(
        self,
        input_ids,
        *,
        cache,
        use_cache=False,
        return_last_logits_only=False,
        return_sharded_logits=False,
    ):
        self.return_last_logits_only_values.append(bool(return_last_logits_only))
        self.return_sharded_logits_values.append(bool(return_sharded_logits))
        return self.model(input_ids, cache=cache, use_cache=use_cache)

    def _sample_next_token(self, logits, temperature):
        return self.model._sample_next_token(logits, temperature)


def test_native_deepseek_paged_cache_matches_dense_cache_decode() -> None:
    torch.manual_seed(50)
    config = tiny_deepseek_v32_config(vocab_size=64, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.inference_mode():
        dense_cache = model.allocate_cache(1, max_seq_len=16, cache_backend="dense")
        _, dense_cache = model(input_ids[:, :-1], cache=dense_cache, use_cache=True)
        dense_logits, _ = model(input_ids[:, -1:], cache=dense_cache, use_cache=True)

        paged_cache = model.allocate_cache(1, max_seq_len=16, cache_backend="paged", page_size=2)
        _, paged_cache = model(input_ids[:, :-1], cache=paged_cache, use_cache=True)
        paged_logits, _ = model(input_ids[:, -1:], cache=paged_cache, use_cache=True)

    assert paged_cache.cache_backend == "paged"
    torch.testing.assert_close(paged_logits, dense_logits, atol=1e-5, rtol=1e-5)


def test_native_deepseek_generate_supports_paged_cache() -> None:
    torch.manual_seed(51)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.inference_mode():
        dense = model.generate(input_ids, max_new_tokens=3, cache_backend="dense")
        paged = model.generate(input_ids, max_new_tokens=3, cache_backend="paged", page_size=2)

    torch.testing.assert_close(paged, dense)


def test_native_deepseek_paged_decode_does_not_materialize_cache(monkeypatch) -> None:
    torch.manual_seed(53)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    decode_token = torch.tensor([[4]], dtype=torch.long)
    decode_calls: list[str] = []
    original_decode = deepseek_mod.batched_paged_decode_attention

    def record_decode(query, cache, request_ids, positions, **kwargs):
        decode_calls.extend(request_ids)
        return original_decode(query, cache, request_ids, positions, **kwargs)

    def fail_materialize(self, batch):
        raise AssertionError("single-token paged decode should not materialize dense KV")

    with torch.inference_mode():
        paged_cache = model.allocate_cache(1, max_seq_len=16, cache_backend="paged", page_size=2)
        _, paged_cache = model(input_ids, cache=paged_cache, use_cache=True)
        monkeypatch.setattr(deepseek_mod, "batched_paged_decode_attention", record_decode)
        monkeypatch.setattr(deepseek_mod.PagedDeepSeekLayerKVCache, "materialize", fail_materialize)
        model(decode_token, cache=paged_cache, use_cache=True)

    assert decode_calls == ["batch-0"] * config.num_hidden_layers


def test_native_deepseek_paged_prefill_does_not_materialize_cache(monkeypatch) -> None:
    torch.manual_seed(58)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    def fail_materialize(self, batch):
        raise AssertionError("paged prefill should attend over pages without materializing dense KV")

    with torch.inference_mode():
        paged_cache = model.allocate_cache(1, max_seq_len=16, cache_backend="paged", page_size=2)
        monkeypatch.setattr(deepseek_mod.PagedDeepSeekLayerKVCache, "materialize", fail_materialize)
        model(input_ids, cache=paged_cache, use_cache=True)


def test_native_deepseek_cache_row_views_support_mixed_lengths() -> None:
    torch.manual_seed(55)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    cache = model.allocate_cache(3, max_seq_len=16, cache_backend="paged", page_size=2)

    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3]], dtype=torch.long), cache=cache.for_rows((0,)), use_cache=True)
        model(torch.tensor([[4, 5, 6, 7]], dtype=torch.long), cache=cache.for_rows((1,)), use_cache=True)
        model(torch.tensor([[8]], dtype=torch.long), cache=cache.for_rows((0,)), use_cache=True)

    assert cache.for_rows((0,)).seq_len == 4
    assert cache.for_rows((1,)).seq_len == 4
    assert cache.for_rows((2,)).seq_len == 0


def test_native_deepseek_cache_row_views_reject_invalid_rows() -> None:
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    cache = model.allocate_cache(2, max_seq_len=16, cache_backend="paged", page_size=2)

    with pytest.raises(ValueError, match="cache row out of range"):
        cache.for_rows((-1,))
    with pytest.raises(ValueError, match="cache row out of range"):
        cache.for_rows((2,))
    with pytest.raises(ValueError, match="cache row out of range"):
        cache.copy_prefix_from(cache, 0, source_row=0, dest_row=2)


def test_native_deepseek_paged_cache_aliases_prefix_pages() -> None:
    torch.manual_seed(56)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    cache = model.allocate_cache(3, max_seq_len=16, cache_backend="paged", page_size=2)

    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3]], dtype=torch.long), cache=cache.for_rows((0,)), use_cache=True)
        cache.copy_prefix_from(cache, 3, source_row=0, dest_row=1)
        layer = cache.layers[0]
        source_pages = tuple(layer.pages.sequence(layer.request_ids[0]).page_ids)
        aliased_pages = tuple(layer.pages.sequence(layer.request_ids[1]).page_ids)
        model(torch.tensor([[4]], dtype=torch.long), cache=cache.for_rows((1,)), use_cache=True)
        extended_pages = tuple(layer.pages.sequence(layer.request_ids[1]).page_ids)

    assert aliased_pages == source_pages
    assert extended_pages[0] == source_pages[0]
    assert extended_pages[1] != source_pages[1]
    assert layer.pages.sequence(layer.request_ids[0]).length == 3
    assert layer.pages.sequence(layer.request_ids[1]).length == 4


def test_continuous_batch_engine_runs_requests_with_prefix_hits() -> None:
    torch.manual_seed(52)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=2,
    )
    requests = [
        ServingRequest("req-a", (1, 2, 3), 2, arrival_step=0),
        ServingRequest("req-b", (1, 2, 3, 4), 2, arrival_step=1),
        ServingRequest("req-c", (5, 6, 7, 8), 2, arrival_step=1),
    ]

    results = engine.run(requests)

    assert [result.request_id for result in results] == ["req-a", "req-b", "req-c"]
    assert [len(result.tokens) for result in results] == [5, 6, 6]
    assert results[1].prefix_hit_tokens == 3
    assert results[2].prefix_hit_tokens == 0
    assert engine.stats.prefix_reuse_tokens == 3
    assert engine.stats.prefix_reuse_requests == 1
    assert engine.stats.max_model_batch_size == 2
    assert engine.stats.decode_model_calls < 3
    assert engine.stats.persistent_cache_rows == 4


def test_continuous_batch_engine_prefix_reuse_matches_full_prefill() -> None:
    torch.manual_seed(54)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    reuse_engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=2,
    )
    baseline_engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=1,
    )

    reuse_results = reuse_engine.run(
        [
            ServingRequest("warm", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("reuse", (1, 2, 3, 4), 2, arrival_step=1),
        ]
    )
    baseline_results = baseline_engine.run([ServingRequest("reuse", (1, 2, 3, 4), 2, arrival_step=0)])

    assert reuse_results[1].tokens == baseline_results[0].tokens
    assert reuse_results[1].prefix_hit_tokens == 3
    assert reuse_engine.stats.prefix_reuse_tokens == 3


def test_continuous_batch_engine_pins_shared_prefix_across_batches() -> None:
    # With pin_shared_prefix, a recurring shared prompt prefix is prefilled once
    # and reused across separate scheduler batches (the online-batcher case),
    # while producing the same tokens as a full per-request prefill.
    torch.manual_seed(56)
    config = tiny_deepseek_v32_config(vocab_size=64, max_position_embeddings=64)
    model = DeepSeekV32ForCausalLM(config).eval()
    shared = tuple(range(1, 17))  # 16-token shared prefix (>= min_prefix_tokens)
    pinned = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=4,
        max_active_requests=4,
        prefix_cache_capacity=8,
        pin_shared_prefix=True,
    )
    # Batch 0 (arrival_step 0) establishes the shared prefix; batch 1
    # (arrival_step 1) must reuse it rather than re-prefill it.
    results = pinned.run(
        [
            ServingRequest("b0-a", (*shared, 20), 1, arrival_step=0),
            ServingRequest("b0-b", (*shared, 21), 1, arrival_step=0),
            ServingRequest("b1-a", (*shared, 22), 1, arrival_step=1),
            ServingRequest("b1-b", (*shared, 23), 1, arrival_step=1),
        ]
    )
    by_id = {r.request_id: r for r in results}
    # The second-batch requests reuse the pinned 16-token prefix.
    assert by_id["b1-a"].prefix_hit_tokens == len(shared)
    assert by_id["b1-b"].prefix_hit_tokens == len(shared)
    assert pinned.stats.prefix_reuse_requests >= 2
    # The shared prefix is prefilled exactly once across both batches.
    assert pinned.stats.prefill_common_prefix_batches == 1

    # Outputs match a full per-request prefill (no reuse) baseline.
    baseline = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=4,
        max_active_requests=1,
    )
    base = baseline.run([ServingRequest("b1-a", (*shared, 22), 1, arrival_step=0)])
    assert by_id["b1-a"].tokens == base[0].tokens


def test_continuous_batch_engine_does_not_report_hit_without_prefix_storage() -> None:
    torch.manual_seed(55)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=1,
        prefix_cache_capacity=0,
    )

    results = engine.run(
        [
            ServingRequest("warm", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("cold", (1, 2, 3, 4), 1, arrival_step=1),
        ]
    )

    assert results[1].prefix_hit_tokens == 0
    assert engine.stats.prefix_reuse_requests == 0
    assert engine.stats.prefix_reuse_tokens == 0


def test_continuous_batch_engine_does_not_report_evicted_prefix_hit() -> None:
    torch.manual_seed(56)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=1,
        prefix_cache_capacity=1,
    )

    results = engine.run(
        [
            ServingRequest("evicted", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("replacement", (5, 6, 7), 1, arrival_step=1),
            ServingRequest("stale-match", (1, 2, 3, 4), 1, arrival_step=2),
        ]
    )

    assert results[2].prefix_hit_tokens == 0
    assert engine.stats.prefix_reuse_requests == 0
    assert engine.stats.prefix_reuse_tokens == 0


def test_continuous_batch_engine_uses_one_persistent_cache(monkeypatch) -> None:
    torch.manual_seed(57)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    original_allocate_cache = model.allocate_cache
    calls: list[tuple[int, int | None, str | None]] = []

    def record_allocate_cache(batch_size, max_seq_len=None, **kwargs):
        calls.append((batch_size, max_seq_len, kwargs.get("cache_backend")))
        return original_allocate_cache(batch_size, max_seq_len=max_seq_len, **kwargs)

    monkeypatch.setattr(model, "allocate_cache", record_allocate_cache)
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=2,
        prefix_cache_capacity=2,
    )

    engine.run(
        [
            ServingRequest("req-a", (1, 2, 3), 2, arrival_step=0),
            ServingRequest("req-b", (1, 2, 3, 4), 2, arrival_step=1),
            ServingRequest("req-c", (5, 6, 7, 8), 2, arrival_step=1),
        ]
    )

    assert calls == [(4, 6, "paged")]
    assert engine.stats.max_model_batch_size == 2


def test_continuous_batch_engine_prefers_ready_prefix_hits() -> None:
    torch.manual_seed(59)
    config = tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)
    model = DeepSeekV32ForCausalLM(config).eval()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        cache_backend="paged",
        page_size=2,
        max_active_requests=1,
        prefix_cache_capacity=1,
    )

    results = engine.run(
        [
            ServingRequest("warm", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("cold", (8, 9, 10, 11), 1, arrival_step=1),
            ServingRequest("hit", (1, 2, 3, 4), 1, arrival_step=1),
        ]
    )
    by_id = {result.request_id: result for result in results}

    assert by_id["hit"].prefix_hit_tokens == 3
    assert by_id["hit"].started_step == 1
    assert by_id["cold"].started_step == 2
    assert engine.stats.prefix_reuse_tokens == 3


def test_continuous_batch_engine_batches_prefix_hit_suffix_prefill() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=3,
    )

    results = engine.run(
        [
            ServingRequest("warm", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("hit-a", (1, 2, 3, 4), 1, arrival_step=1),
            ServingRequest("hit-b", (1, 2, 3, 5), 1, arrival_step=1),
        ]
    )

    assert [result.prefix_hit_tokens for result in results] == [0, 3, 3]
    assert engine.stats.prefix_reuse_requests == 2
    assert engine.stats.prefix_reuse_tokens == 6
    assert engine.stats.prefill_model_calls == 2
    assert engine.stats.max_model_batch_size == 2


def test_continuous_batch_engine_splits_prefix_hit_suffixes_by_prefix_length() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=4,
    )

    results = engine.run(
        [
            ServingRequest("warm-a", (1, 2, 3), 1, arrival_step=0),
            ServingRequest("warm-b", (10, 11, 12, 13, 14), 1, arrival_step=0),
            ServingRequest("hit-a", (1, 2, 3, 21), 1, arrival_step=1),
            ServingRequest("hit-b", (10, 11, 12, 13, 14, 21), 1, arrival_step=1),
        ]
    )

    assert [result.prefix_hit_tokens for result in results] == [0, 0, 3, 5]
    assert engine.stats.prefill_model_calls == 4
    assert engine.stats.prefix_reuse_tokens == 8


def test_continuous_batch_engine_batches_common_prefix_prefill() -> None:
    shared = tuple(range(16))
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=4,
    )
    baseline_model = _RaggedGraphToyModel()
    baseline = ContinuousBatchEngine(
        baseline_model,
        device=torch.device("cpu"),
        max_active_requests=1,
        prefix_cache_capacity=0,
    )
    requests = [
        ServingRequest("a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("b", (*shared, 22), 1, arrival_step=0),
        ServingRequest("c", (*shared, 23), 1, arrival_step=0),
    ]

    results = engine.run(requests)
    baseline_results = baseline.run(requests)

    assert [result.tokens for result in results] == [result.tokens for result in baseline_results]
    assert [result.prefix_hit_tokens for result in results] == [0, 0, 0]
    assert engine.stats.prefill_model_calls == 2
    assert engine.stats.prefill_tokens == 19
    assert engine.stats.max_model_batch_size == 3


def test_continuous_batch_engine_can_pad_common_prefix_suffix_prefill(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", "1")
    shared = tuple(range(16))
    model = _SelectedLogitsToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=4,
    )
    requests = [
        ServingRequest("a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("b", (*shared, 22, 23, 24), 1, arrival_step=0),
        ServingRequest("c", (*shared, 25, 26), 1, arrival_step=0),
    ]

    results = engine.run(requests)

    assert [result.tokens[-1] for result in results] == [22, 25, 27]
    assert model.selected_positions == [[0, 2, 1]]
    assert engine.stats.prefill_model_calls == 2
    assert engine.stats.prefill_tokens == 25
    assert engine.stats.max_model_batch_size == 3


def test_continuous_batch_engine_reuses_common_prefix_for_padded_refill(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", "1")
    shared = tuple(range(16))
    model = _SelectedLogitsToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=4,
    )
    requests = [
        ServingRequest("warm-a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("warm-b", (*shared, 22, 23, 24), 1, arrival_step=0),
        ServingRequest("late-a", (*shared, 31), 1, arrival_step=1),
        ServingRequest("late-b", (*shared, 32, 33), 1, arrival_step=1),
    ]

    results = engine.run(requests)

    assert [result.tokens[-1] for result in results] == [22, 25, 32, 34]
    assert [result.prefix_hit_tokens for result in results] == [0, 0, 16, 16]
    assert model.selected_positions == [[0, 2], [0, 1]]
    assert engine.stats.prefill_common_prefix_batches == 1
    assert engine.stats.prefill_prefix_reuse_batches == 1
    assert engine.stats.prefill_padded_suffix_batches == 2
    assert engine.stats.prefill_model_calls == 3
    assert engine.stats.prefix_reuse_requests == 2
    assert engine.stats.prefix_reuse_tokens == 32


def test_continuous_batch_engine_graph_prefill_buckets_batch_and_matches() -> None:
    # graph_prefill routes suffix prefill through the model's row_indices
    # ragged-prefill LOGITS graph with the batch padded to a power of two so
    # graph shapes repeat across batches. A reuse batch of three differently-sized
    # suffixes is bucketed up to four rows (one dummy padding row); outputs must
    # equal the toy model's selected-token contract and the ragged graph path
    # must be taken (prefill_graph_hits incremented).
    shared = tuple(range(16))
    model = _SelectedLogitsToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=8,
        prefix_cache_capacity=8,
        pin_shared_prefix=True,
        graph_prefill=True,
    )
    requests = [
        ServingRequest("warm-a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("warm-b", (*shared, 22), 1, arrival_step=0),
        ServingRequest("late-a", (*shared, 31), 1, arrival_step=1),
        ServingRequest("late-b", (*shared, 32, 33), 1, arrival_step=1),
        ServingRequest("late-c", (*shared, 34, 35, 36), 1, arrival_step=1),
    ]

    results = engine.run(requests)
    by_id = {result.request_id: result for result in results}

    # Toy contract: next token = (selected real last prompt token) + 1.
    assert by_id["late-a"].tokens[-1] == 32
    assert by_id["late-b"].tokens[-1] == 34
    assert by_id["late-c"].tokens[-1] == 37
    assert by_id["late-a"].prefix_hit_tokens == 16
    # The reuse batch of three was bucketed up to four rows in the ragged graph
    # call (three real logit positions 0/1/2 plus one dummy padding row at 0).
    assert len(model.selected_positions[-1]) == 4
    assert sorted(model.selected_positions[-1]) == [0, 0, 1, 2]
    # The ragged-prefill graph path was taken (not the eager fallback).
    assert engine.stats.prefill_graph_hits >= 1
    assert engine.stats.prefill_prefix_reuse_batches >= 1
    assert engine.stats.prefill_padded_suffix_batches >= 1
    assert engine.stats.prefix_reuse_requests >= 3


def test_continuous_batch_engine_can_keep_common_prefix_without_full_prompt_entries(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", "1")
    shared = tuple(range(16))
    model = _SelectedLogitsToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=2,
        store_full_prompt_prefixes=False,
    )
    requests = [
        ServingRequest("warm-a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("warm-b", (*shared, 22, 23), 1, arrival_step=0),
        ServingRequest("late-a", (*shared, 31), 1, arrival_step=1),
        ServingRequest("late-b", (*shared, 32, 33), 1, arrival_step=1),
    ]

    results = engine.run(requests)

    assert [result.prefix_hit_tokens for result in results] == [0, 0, 16, 16]
    assert engine.stats.prefill_common_prefix_batches == 1
    assert engine.stats.prefill_prefix_reuse_batches == 1
    assert engine.stats.prefix_reuse_requests == 2
    assert engine.stats.prefix_reuse_tokens == 32


def test_continuous_batch_engine_can_use_prefill_logits_graph(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PREFILL_CAPTURE", "1")
    model = _PrefillLogitsGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
    )
    requests = [
        ServingRequest("a", (1, 2, 3), 1, arrival_step=0),
        ServingRequest("b", (4, 5, 6), 1, arrival_step=0),
    ]

    results = engine.run(requests)

    assert [result.tokens[-1] for result in results] == [4, 7]
    assert model.prefill_capture_flags == [True]
    assert model.forward_calls == 0
    assert engine.stats.prefill_graph_hits == 1
    assert engine.stats.prefill_graph_misses == 0


def test_continuous_batch_engine_can_use_selected_prefill_logits_graph(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PADDED_SUFFIX_PREFILL", "1")
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_SELECTED_PREFILL_CAPTURE", "1")
    shared = tuple(range(16))
    model = _SelectedLogitsGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=4,
    )
    requests = [
        ServingRequest("a", (*shared, 21), 1, arrival_step=0),
        ServingRequest("b", (*shared, 22, 23, 24), 1, arrival_step=0),
        ServingRequest("c", (*shared, 25, 26), 1, arrival_step=0),
    ]

    results = engine.run(requests)

    assert [result.tokens[-1] for result in results] == [22, 25, 27]
    assert model.selected_capture_flags == [True]
    assert model.selected_positions == [[0, 2, 1]]
    assert engine.stats.prefill_graph_hits == 1
    assert engine.stats.prefill_graph_misses == 0


def test_continuous_batch_engine_common_prefix_prefill_respects_min_length() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=2,
    )

    engine.run(
        [
            ServingRequest("a", (1, 2, 3, 4), 1, arrival_step=0),
            ServingRequest("b", (1, 2, 3, 5), 1, arrival_step=0),
        ]
    )

    assert engine.stats.prefill_model_calls == 1
    assert engine.stats.prefill_tokens == 8
    assert engine.stats.max_model_batch_size == 2


def test_continuous_batch_engine_uses_ragged_graph_decode_for_mixed_lengths() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
    )

    results = engine.run(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("medium", (3, 4, 5), 3, arrival_step=0),
            ServingRequest("long", (6, 7, 8, 9), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 6, 7]
    assert model.ragged_logits_graph_calls == 2
    assert model.ragged_eager_calls == 0
    assert engine.stats.decode_model_calls == 2
    assert engine.stats.ragged_decode_batches == 2
    assert engine.stats.ragged_decode_tokens == 6
    assert engine.stats.decode_graph_hits == 2
    assert engine.stats.max_model_batch_size == 3


def test_continuous_batch_engine_uses_ragged_graph_decode_for_uniform_batches(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE", "1")
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
    )

    results = engine.run(
        [
            ServingRequest("a", (1, 2), 3, arrival_step=0),
            ServingRequest("b", (3, 4), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 5]
    assert model.ragged_logits_graph_calls == 2
    assert model.ragged_eager_calls == 0
    assert engine.stats.decode_model_calls == 2
    assert engine.stats.ragged_decode_batches == 2
    assert engine.stats.ragged_decode_tokens == 4


def test_continuous_batch_engine_can_bucket_mixed_length_decode_without_ragged() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
        enable_ragged_decode=False,
    )

    results = engine.run(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("medium", (3, 4, 5), 3, arrival_step=0),
            ServingRequest("long", (6, 7, 8, 9), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 6, 7]
    assert model.ragged_logits_graph_calls == 0
    assert model.ragged_eager_calls == 0
    assert engine.stats.ragged_decode_batches == 0
    assert engine.stats.decode_model_calls == 6


def test_continuous_batch_engine_buckets_by_actual_cache_row_seq_len() -> None:
    model = _SeqLenCheckingSkewedToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
        enable_ragged_decode=False,
    )

    results = engine.run(
        [
            ServingRequest("a", (1, 2), 2, arrival_step=0),
            ServingRequest("b", (3, 4), 2, arrival_step=0),
        ]
    )

    assert [result.request_id for result in results] == ["a", "b"]
    assert engine.stats.decode_model_calls == 2


def test_continuous_batch_engine_advances_cache_row_seq_len_after_ragged_decode() -> None:
    model = _RaggedNoSeqLenUpdateToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
    )

    engine.run(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("long", (3, 4, 5), 3, arrival_step=0),
        ]
    )

    assert model.decode_positions == [[2, 3], [3, 4]]


def test_continuous_batch_engine_buckets_ragged_decode_rows(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKETS", "1")
    model = _RaggedDecodeShapeRecordingToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=4,
        prefix_cache_capacity=0,
    )

    results = engine.run(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("medium", (3, 4, 5), 3, arrival_step=0),
            ServingRequest("long", (6, 7, 8, 9), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 6, 7]
    assert model.decode_shapes[0] == (4, [0, 1, 2, 3])
    assert engine.stats.max_model_batch_size == 4


def test_continuous_batch_engine_can_disable_ragged_decode_row_buckets(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKETS", "0")
    model = _RaggedDecodeShapeRecordingToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=4,
        prefix_cache_capacity=0,
    )

    engine.run(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("medium", (3, 4, 5), 3, arrival_step=0),
            ServingRequest("long", (6, 7, 8, 9), 3, arrival_step=0),
        ]
    )

    assert model.decode_shapes[0] == (3, [0, 1, 2])


def test_continuous_batch_engine_dispatches_bucketed_decode_graphs() -> None:
    model = _StaticDecodeGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
        enable_ragged_decode=False,
    )

    results = engine.run(
        [
            ServingRequest("a", (1, 2), 3, arrival_step=0),
            ServingRequest("b", (3, 4), 3, arrival_step=0),
            ServingRequest("c", (5, 6), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 5, 5]
    assert model.static_token_graph_calls == 2
    assert engine.stats.decode_graph_hits == 2
    assert engine.stats.decode_model_calls == 2


def test_continuous_batch_engine_accepts_device_resident_model_wrapper() -> None:
    wrapper = _DeviceResidentToyWrapper(_RaggedGraphToyModel())
    engine = ContinuousBatchEngine(
        wrapper,
        device=torch.device("cpu"),
        max_active_requests=1,
        prefix_cache_capacity=0,
    )

    results = engine.run([ServingRequest("req", (1, 2), 2, arrival_step=0)])

    assert len(results[0].tokens) == 4
    assert wrapper.eval_calls == 1


def test_continuous_batch_engine_requests_sharded_logits_for_tp_sampler() -> None:
    wrapper = _ShardedForwardToyWrapper(_RaggedGraphToyModel())
    engine = ContinuousBatchEngine(
        wrapper,
        device=torch.device("cpu"),
        max_active_requests=1,
        prefix_cache_capacity=0,
    )

    results = engine.run([ServingRequest("req", (1, 2), 2, arrival_step=0)])

    assert len(results[0].tokens) == 4
    assert wrapper.return_sharded_logits_values == [True, True]
    assert wrapper.return_last_logits_only_values == [True, True]


def test_continuous_batch_engine_records_stream_token_events() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
    )

    results, events = engine.run_with_events(
        [
            ServingRequest("short", (1, 2), 3, arrival_step=0),
            ServingRequest("long", (6, 7, 8), 3, arrival_step=0),
        ]
    )

    assert [len(result.tokens) for result in results] == [5, 6]
    assert [event.request_id for event in events] == [
        "short",
        "long",
        "short",
        "long",
        "short",
        "long",
    ]
    assert [event.step for event in events] == [0, 0, 1, 1, 2, 2]
    assert [event.generated for event in events] == [1, 1, 2, 2, 3, 3]
    assert [event.finished for event in events] == [False, False, False, False, True, True]

    stream_engine = ContinuousBatchEngine(
        _RaggedGraphToyModel(),
        device=torch.device("cpu"),
        max_active_requests=2,
        prefix_cache_capacity=0,
    )
    stream_events = list(
        stream_engine.iter_events(
            [
                ServingRequest("short", (1, 2), 3, arrival_step=0),
                ServingRequest("long", (6, 7, 8), 3, arrival_step=0),
            ]
        )
    )
    assert stream_events == events


def test_continuous_batch_engine_online_step_accepts_new_requests_between_decodes() -> None:
    engine = ContinuousBatchEngine(
        _RaggedGraphToyModel(),
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
    )
    engine.start_online(max_seq_len=8)
    engine.submit_online(ServingRequest("short", (1, 2), 2, arrival_step=0))
    engine.submit_online(ServingRequest("long", (6, 7, 8), 2, arrival_step=0))

    first_step = engine.step_online()
    engine.submit_online(ServingRequest("late", (10, 11), 1, arrival_step=1))
    second_step = engine.step_online()
    final_step = engine.step_online()

    assert [event.request_id for event in first_step] == ["short", "long"]
    assert [event.step for event in first_step] == [0, 0]
    assert [event.request_id for event in second_step] == ["short", "long", "late"]
    assert [event.step for event in second_step] == [1, 1, 1]
    assert [event.generated for event in second_step] == [2, 2, 1]
    assert [event.finished for event in second_step] == [True, True, True]
    assert final_step == []
    assert not engine.has_online_work()
    assert engine.stats.queued_requests == 3


def test_continuous_batch_engine_online_refill_can_wait_for_free_rows(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_FREE_ROWS", "2")
    engine = ContinuousBatchEngine(
        _RaggedGraphToyModel(),
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
    )
    engine.start_online(max_seq_len=8)
    engine.submit_online(ServingRequest("a", (1, 2), 3, arrival_step=0))
    engine.submit_online(ServingRequest("b", (6, 7), 3, arrival_step=0))

    first_step = engine.step_online()
    engine.submit_online(ServingRequest("late", (10, 11), 1, arrival_step=1))
    second_step = engine.step_online()
    third_step = engine.step_online()

    assert [event.request_id for event in first_step] == ["a", "b"]
    assert [event.request_id for event in second_step] == ["a", "b"]
    assert [event.finished for event in second_step] == [False, False]
    assert [event.request_id for event in third_step] == ["a", "b", "late"]
    assert [event.finished for event in third_step] == [True, True, True]


def test_continuous_batch_engine_online_refill_can_wait_for_ready_requests(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS", "2")
    engine = ContinuousBatchEngine(
        _RaggedGraphToyModel(),
        device=torch.device("cpu"),
        max_active_requests=4,
        prefix_cache_capacity=0,
    )
    engine.start_online(max_seq_len=8)
    engine.submit_online(ServingRequest("a", (1, 2), 4, arrival_step=0))
    engine.submit_online(ServingRequest("b", (6, 7), 4, arrival_step=0))

    first_step = engine.step_online()
    engine.submit_online(ServingRequest("late-1", (10, 11), 1, arrival_step=1))
    second_step = engine.step_online()
    engine.submit_online(ServingRequest("late-2", (12, 13), 1, arrival_step=2))
    third_step = engine.step_online()

    assert [event.request_id for event in first_step] == ["a", "b"]
    assert [event.request_id for event in second_step] == ["a", "b"]
    assert [event.finished for event in second_step] == [False, False]
    assert [event.request_id for event in third_step] == ["a", "b", "late-1", "late-2"]
    assert [event.finished for event in third_step] == [False, False, True, True]


def test_continuous_batch_engine_prefill_token_budget_limits_admission() -> None:
    model = _RaggedGraphToyModel()
    engine = ContinuousBatchEngine(
        model,
        device=torch.device("cpu"),
        max_active_requests=3,
        prefix_cache_capacity=0,
        prefill_token_budget=4,
    )

    results = engine.run(
        [
            ServingRequest("req-a", (1, 2, 3, 4), 1, arrival_step=0),
            ServingRequest("req-b", (5, 6, 7, 8), 1, arrival_step=0),
            ServingRequest("req-c", (9, 10, 11, 12), 1, arrival_step=0),
        ]
    )

    assert [result.started_step for result in results] == [0, 1, 2]
    assert engine.stats.scheduler_steps == 4
    assert engine.stats.prefill_model_calls == 3


def test_serve_smoke_cli_runs() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "serve-smoke",
            "--device",
            "cpu",
            "--cache-backend",
            "paged",
            "--page-size",
            "2",
            "--new-tokens",
            "2",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "TorchInferno serving smoke" in result.stdout
    assert "prefix_hit_tokens=3" in result.stdout
    assert "prefix_reuse_tokens=3" in result.stdout
