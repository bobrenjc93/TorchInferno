import os
import subprocess
import sys

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
    original_decode = deepseek_mod.paged_decode_attention

    def record_decode(query, cache, request_id, position, **kwargs):
        decode_calls.append(request_id)
        return original_decode(query, cache, request_id, position, **kwargs)

    def fail_materialize(self, batch):
        raise AssertionError("single-token paged decode should not materialize dense KV")

    with torch.inference_mode():
        paged_cache = model.allocate_cache(1, max_seq_len=16, cache_backend="paged", page_size=2)
        _, paged_cache = model(input_ids, cache=paged_cache, use_cache=True)
        monkeypatch.setattr(deepseek_mod, "paged_decode_attention", record_decode)
        monkeypatch.setattr(deepseek_mod.PagedDeepSeekLayerKVCache, "materialize", fail_materialize)
        model(decode_token, cache=paged_cache, use_cache=True)

    assert decode_calls == ["batch-0"] * config.num_hidden_layers


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
        ServingRequest("req-c", (5, 6), 1, arrival_step=1),
    ]

    results = engine.run(requests)

    assert [result.request_id for result in results] == ["req-a", "req-b", "req-c"]
    assert [len(result.tokens) for result in results] == [5, 6, 3]
    assert results[1].prefix_hit_tokens == 3
    assert results[2].prefix_hit_tokens == 0


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
