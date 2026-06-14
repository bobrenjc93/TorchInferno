import os
import subprocess
import sys

import torch

from torchinferno.models.auto import load_model_auto
from torchinferno.models.deepseek_v32 import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import paged_causal_attention
from torchinferno.runtime.prefix_cache import PrefixCacheIndex
from torchinferno.runtime.scheduler import DisaggregatedPrefillDecodeSimulator
from torchinferno.runtime.traffic import TrafficPattern, simulate_traffic
from torchinferno.tokenization import load_text_tokenizer
from torchinferno.validation import capture_logit_reference, validate_logit_reference


def _write_wordlevel_tokenizer(path, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0}
    vocab.update({f"t{i}": i for i in range(1, vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))


def test_auto_loader_tokenizer_and_logit_validation(tmp_path) -> None:
    torch.manual_seed(30)
    model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)).eval()
    model.save_pretrained(tmp_path)
    _write_wordlevel_tokenizer(tmp_path, 32)

    loaded = load_model_auto(tmp_path).eval()
    tokenizer = load_text_tokenizer(tmp_path)
    input_ids = tokenizer.encode("t1 t2 t3")
    reference = capture_logit_reference(loaded, input_ids)
    result = validate_logit_reference(loaded, reference)

    assert input_ids == [1, 2, 3]
    assert tokenizer.decode(input_ids)
    assert result.passed


def test_paged_attention_matches_dense_attention_with_value_dim() -> None:
    torch.manual_seed(31)
    heads = 2
    tokens = 5
    head_dim = 4
    value_dim = 3
    keys = torch.randn(heads, tokens, head_dim)
    values = torch.randn(heads, tokens, value_dim)
    query = torch.randn(heads, tokens, head_dim)
    positions = torch.arange(tokens)
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache.append("req", keys, values)

    actual = paged_causal_attention(query, cache, "req", positions)
    scores = torch.matmul(query, keys.transpose(-1, -2)) / (head_dim**0.5)
    allowed = torch.arange(tokens)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
    expected = torch.matmul(torch.softmax(scores, dim=-1), values)

    torch.testing.assert_close(actual, expected)


def test_paged_attention_supports_grouped_query_attention() -> None:
    torch.manual_seed(32)
    kv_heads = 2
    query_heads = 4
    tokens = 5
    head_dim = 4
    value_dim = 3
    keys = torch.randn(kv_heads, tokens, head_dim)
    values = torch.randn(kv_heads, tokens, value_dim)
    query = torch.randn(query_heads, tokens, head_dim)
    positions = torch.arange(tokens)
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache.append("req", keys, values)

    actual = paged_causal_attention(query, cache, "req", positions, enable_gqa=True)
    expanded_keys = keys.repeat_interleave(query_heads // kv_heads, dim=0)
    expanded_values = values.repeat_interleave(query_heads // kv_heads, dim=0)
    scores = torch.matmul(query, expanded_keys.transpose(-1, -2)) / (head_dim**0.5)
    allowed = torch.arange(tokens)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
    expected = torch.matmul(torch.softmax(scores, dim=-1), expanded_values)

    torch.testing.assert_close(actual, expected)


def test_prefix_cache_and_traffic_simulation() -> None:
    prefix_cache = PrefixCacheIndex()
    prefix_cache.add("prefill-a", (1, 2, 3), route_id="route-a")
    match, entry = prefix_cache.lookup((1, 2, 3, 4, 5))

    scheduler = DisaggregatedPrefillDecodeSimulator(
        prefill_ranks=(0,),
        decode_ranks=(1,),
        prefill_us_per_token=2.0,
        decode_us_per_token=4.0,
        network_latency_us=10.0,
    )
    traffic = simulate_traffic(TrafficPattern(requests=6, burst_size=3, seed=4), scheduler)

    assert match.matched_tokens == (1, 2, 3)
    assert entry is not None and entry.request_id == "prefill-a"
    assert len(traffic.jobs) == 6
    assert len(traffic.stages) == 12
    assert traffic.requests_per_second > 0


def test_prefix_cache_remove_falls_back_to_shorter_prefix() -> None:
    prefix_cache = PrefixCacheIndex()
    prefix_cache.add("shared", (1, 2), route_id="shared")
    prefix_cache.add("conversation", (1, 2, 3, 4), route_id="conversation")

    match, entry = prefix_cache.lookup((1, 2, 3, 4, 5))
    assert match.matched_tokens == (1, 2, 3, 4)
    assert entry is not None and entry.route_id == "conversation"

    assert prefix_cache.remove("conversation")
    assert not prefix_cache.remove("missing")

    match, entry = prefix_cache.lookup((1, 2, 3, 4, 5))
    assert match.matched_tokens == (1, 2)
    assert entry is not None and entry.route_id == "shared"


def test_production_cli_workflows(tmp_path) -> None:
    torch.manual_seed(32)
    model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(vocab_size=32, max_position_embeddings=16)).eval()
    model.save_pretrained(tmp_path)
    _write_wordlevel_tokenizer(tmp_path, 32)
    reference = tmp_path / "reference.json"
    env = {**os.environ, "PYTHONPATH": "src"}

    text = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "text-generate",
            str(tmp_path),
            "t1 t2 t3",
            "--device",
            "cpu",
            "--new-tokens",
            "2",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    capture = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "capture-logits",
            str(tmp_path),
            str(reference),
            "--device",
            "cpu",
            "--input-ids",
            "1",
            "2",
            "3",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "validate-logits",
            str(tmp_path),
            str(reference),
            "--device",
            "cpu",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    traffic = subprocess.run(
        [sys.executable, "-m", "torchinferno.cli", "traffic-smoke", "--requests", "4"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "TorchInferno text generation" in text.stdout
    assert "TorchInferno logit reference captured" in capture.stdout
    assert "passed=True" in validate.stdout
    assert "TorchInferno traffic simulation" in traffic.stdout
