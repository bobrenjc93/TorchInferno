import os
import subprocess
import sys

import pytest
import torch

import torchinferno.models.deepseek as deepseek_mod
from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.runtime.serving import ContinuousBatchEngine, ServingRequest


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
