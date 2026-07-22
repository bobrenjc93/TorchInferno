import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
import torchinferno.models.deepseek_v4.tensor_parallel as v4_tp

from torchinferno.kernels import deepseek_v4_tilelang as v4_kernels
from torchinferno.models.auto import load_model_auto
from torchinferno.models.checkpoint_io import CheckpointTensorLoader
from torchinferno.models.deepseek_v32 import DeepSeekV32Config
from torchinferno.models.deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM, tiny_deepseek_v4_config
from torchinferno.models.deepseek_v4.checkpoint import (
    audit_deepseek_v4_checkpoint,
    expected_deepseek_v4_tensor_names,
)
from torchinferno.models.deepseek_v4.ops import (
    dequantize_block_fp8,
    dequantize_mxfp4,
    hc_split_sinkhorn,
)
from torchinferno.models.identity import detect_model_identity
from torchinferno.kernels.deepseek_v4_marlin import _align_tokens


def _tiny_model(*, max_position_embeddings: int = 140) -> DeepSeekV4ForCausalLM:
    torch.manual_seed(401)
    return DeepSeekV4ForCausalLM(
        tiny_deepseek_v4_config(max_position_embeddings=max_position_embeddings)
    ).eval()


def test_v4_tp_collectives_use_the_configured_replica_group(monkeypatch) -> None:
    group = object()
    calls = []

    monkeypatch.setattr(v4_tp.dist, "all_reduce", lambda tensor, *, group: calls.append(("reduce", group)))
    monkeypatch.setattr(
        v4_tp.dist,
        "all_gather",
        lambda outputs, tensor, *, group: calls.append(("gather", group)),
    )
    v4_tp.set_tensor_parallel_process_group(group)
    try:
        tensor = torch.zeros(1)
        v4_tp._all_reduce(tensor)
        v4_tp._all_gather([torch.zeros_like(tensor)], tensor)
    finally:
        v4_tp.set_tensor_parallel_process_group(None)

    assert calls == [("reduce", group), ("gather", group)]


def test_v4_tp_rmsnorm_keeps_checkpoint_precision_on_cpu() -> None:
    norm = v4_tp.RMSNorm(16)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)

    x_float = x.float()
    expected = (
        weight * x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + norm.eps)
    ).to(x.dtype)

    assert norm.weight.dtype == torch.bfloat16
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


def test_v4_tp_greedy_sampling_gathers_only_shard_candidates(monkeypatch) -> None:
    monkeypatch.setattr(v4_tp, "world_size", 2)
    monkeypatch.setattr(v4_tp, "rank", 0)
    gathered_shapes = []

    def all_gather(outputs, candidate):
        gathered_shapes.append(tuple(candidate.shape))
        outputs[0].copy_(candidate)
        outputs[1].copy_(torch.tensor([[4.0, 4.0], [5.0, 3.0]]))

    monkeypatch.setattr(v4_tp, "_all_gather", all_gather)
    model = type(
        "FakeV4Sampler",
        (),
        {
            "args": type("Args", (), {"vocab_size": 6})(),
            "head": type("Head", (), {"part_vocab_size": 3})(),
            "tensor_parallel_rank": 0,
        },
    )()
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 2.0, 1.0]])

    token = v4_tp.DeepSeekV4TensorParallelForCausalLM._sample_next_token(
        model,
        logits,
        0.0,
    )

    assert gathered_shapes == [(2, 2)]
    assert token.tolist() == [4, 0]


@pytest.mark.parametrize(
    "filename",
    ("../outside.safetensors", "/tmp/outside.safetensors", "nested/model.safetensors", "model.bin"),
)
def test_checkpoint_tensor_loader_rejects_unsafe_shard_paths(tmp_path, filename) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": filename}})
    )

    with pytest.raises(ValueError, match="unsafe checkpoint shard"):
        CheckpointTensorLoader(checkpoint)


def test_checkpoint_tensor_loader_rejects_shard_symlink_escape(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    (checkpoint / "escape.safetensors").symlink_to(outside)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "escape.safetensors"}})
    )

    with pytest.raises(ValueError, match="escapes checkpoint root"):
        CheckpointTensorLoader(checkpoint)


def test_v4_runtime_tilelang_loader_does_not_import_compiler_modules() -> None:
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__
        blocked = {
            "tilelang",
            "torchinferno.kernels.deepseek_v4_tilelang_builder",
            "torchinferno.kernels.deepseek_v4_tilelang_definitions",
        }

        def guarded_import(name, *args, **kwargs):
            if name in blocked or name.startswith("tilelang."):
                raise AssertionError(f"runtime imported offline compiler module: {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        import torchinferno.kernels.deepseek_v4_tilelang
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_v4_fused_decode_kernel_wrappers_dispatch_general_shapes(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def kernel(name: str, *specialization: object):
        calls.append((name, specialization))

        def run(*args: torch.Tensor) -> None:
            if name == "q_norm_rope":
                args[2].copy_(args[0])
            elif name == "rope_inplace":
                args[0].add_(1)
            elif name == "hc_post":
                value, residual, post, comb, output = args
                expected = post[..., None] * value[:, None, :]
                expected += torch.einsum("tsc,tsh->tch", comb, residual.float())
                output.copy_(expected)
            elif name == "hc_prenorm_gemm":
                value, weight, output, square_sum = args
                output.copy_(value.float() @ weight.T)
                square_sum.copy_(value.float().square().sum(-1))

        return run

    monkeypatch.setattr(v4_kernels, "_kernel", kernel)
    q = torch.randn(3, 2, 512, dtype=torch.bfloat16)
    freqs = torch.randn(64)
    q_output = torch.empty_like(q)
    assert v4_kernels.q_norm_rope(q, freqs, 1e-6, output=q_output) is q_output
    assert torch.equal(q_output, q)

    rope_value = q.clone()
    assert v4_kernels.rope_inplace(rope_value, freqs, inverse=True) is rope_value
    torch.testing.assert_close(rope_value, q + 1)

    value = torch.randn(3, 1024, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 1024, dtype=torch.bfloat16)
    post = torch.randn(3, 4)
    comb = torch.randn(3, 4, 4)
    post_output = v4_kernels.hc_post(value, residual, post, comb)
    expected_post = post[..., None] * value[:, None, :]
    expected_post += torch.einsum("tsc,tsh->tch", comb, residual.float())
    torch.testing.assert_close(post_output, expected_post.to(torch.bfloat16))

    hc_value = torch.randn(3, 4, 1024, dtype=torch.bfloat16)
    weight = torch.randn(24, 4096)
    projection, square_sum = v4_kernels.hc_prenorm_gemm(hc_value, weight)
    torch.testing.assert_close(projection, hc_value.flatten(1).float() @ weight.T)
    torch.testing.assert_close(
        square_sum,
        hc_value.flatten(1).float().square().sum(-1, keepdim=True),
    )
    assert calls == [
        ("q_norm_rope", (2, 512, 64, 1e-6)),
        ("rope_inplace", (2, 512, 64, True)),
        ("hc_post", (4, 1024)),
        ("hc_prenorm_gemm", (1024, 4, 24)),
    ]


def test_v4_marlin_alignment_keeps_nonlocal_routes_filtered(monkeypatch) -> None:
    call = {}

    def align(
        local_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens,
        cumsum,
        sort,
    ):
        del sorted_ids, expert_ids
        call.update(
            local_ids=local_ids.clone(),
            num_experts=num_experts,
            block_size=block_size,
            cumsum_size=cumsum.numel(),
            sort=sort,
        )
        num_tokens.zero_()

    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel",
        types.SimpleNamespace(moe_align_block_size=align),
    )
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    _align_tokens(
        topk_ids,
        block_size=8,
        global_num_experts=4,
        expert_map=expert_map,
    )

    assert torch.equal(
        call["local_ids"],
        torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
    )
    assert call["num_experts"] == 5
    assert call["block_size"] == 8
    assert call["cumsum_size"] == 6
    assert call["sort"] is True


def test_public_v4_config_contract_and_aliases() -> None:
    config = DeepSeekV4Config.from_dict(
        {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_hash_layers": 3,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "compress_ratios": [0, 0, *([4, 128] * 20), 4, 0],
            "rope_scaling": {
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        }
    )

    assert config.num_hash_layers == 3
    assert config.qk_nope_head_dim == 448
    assert config.compress_ratios[2:6] == (4, 128, 4, 128)
    assert config.to_dict()["model_type"] == "deepseek_v4"


def test_v4_mhc_matches_literal_released_formula() -> None:
    torch.manual_seed(402)
    mixes = torch.randn(2, 3, 24)
    scale = torch.randn(3)
    base = torch.randn(24)

    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, 5, 1e-6)

    expected_pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    expected_comb = (mixes[..., 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4))
    expected_comb = expected_comb.softmax(-1) + 1e-6
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(4):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + 1e-6)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)

    torch.testing.assert_close(pre, expected_pre)
    torch.testing.assert_close(post, expected_post)
    torch.testing.assert_close(comb, expected_comb)


@pytest.mark.parametrize(
    "cuts",
    [
        (3,),
        (4,),
        (7, 11),
        (127,),
        (1, 2, 3, 4, 5, 127, 128),
    ],
)
def test_v4_chunked_cache_matches_full_across_compression_boundaries(cuts: tuple[int, ...]) -> None:
    model = _tiny_model()
    input_ids = torch.arange(130).remainder(model.config.vocab_size)[None]

    with torch.inference_mode():
        expected, _ = model(input_ids, use_cache=False)
        cache = model.allocate_cache(1, 140)
        outputs = []
        start = 0
        for end in (*cuts, input_ids.size(1)):
            if end <= start:
                continue
            logits, cache = model(input_ids[:, start:end], cache=cache)
            outputs.append(logits)
            start = end

    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=2e-6, rtol=2e-6)
    assert cache.seq_len == 130


def test_v4_c4_indexer_accumulates_every_token_before_first_boundary(monkeypatch) -> None:
    model = _tiny_model(max_position_embeddings=16)
    attention = model.layers[1].attn
    assert attention.indexer is not None
    cache = model.allocate_cache(1, 16).layers[1]
    recorded_positions = []
    original_update = attention.indexer.compressor.update

    def record_update(x, position, *args, **kwargs):
        recorded_positions.append(position)
        return original_update(x, position, *args, **kwargs)

    monkeypatch.setattr(attention.indexer.compressor, "update", record_update)
    hidden = torch.randn(1, 4, model.config.hidden_size)

    with torch.inference_mode():
        attention(hidden, torch.arange(4), cache, (0,))

    assert recorded_positions == [0, 1, 2, 3]
    assert torch.count_nonzero(cache.indexer_compressor.raw_kv[0, :4]) > 0


def test_v4_prefix_copy_restores_partial_c4_and_c128_state() -> None:
    model = _tiny_model()
    input_ids = torch.arange(130).remainder(model.config.vocab_size)[None]
    source = model.allocate_cache(1, 140)
    destination = model.allocate_cache(1, 140)

    with torch.inference_mode():
        expected, source = model(input_ids, cache=source)
        destination.copy_prefix_from(source, 127)
        actual, destination = model(input_ids[:, 127:], cache=destination)

    torch.testing.assert_close(actual, expected[:, 127:], atol=2e-6, rtol=2e-6)
    assert destination.seq_len == 130


def test_v4_row_views_keep_request_state_independent() -> None:
    model = _tiny_model(max_position_embeddings=32)
    cache = model.allocate_cache(3, 32)
    rows_02 = cache.for_rows((0, 2))

    with torch.inference_mode():
        _, rows_02 = model(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), cache=rows_02)
        row_1 = cache.for_rows((1,))
        _, row_1 = model(torch.tensor([[9, 10]]), cache=row_1)

    assert rows_02.seq_len == 4
    assert row_1.seq_len == 2
    assert cache.layers[0].seq_lens == [4, 2, 4]


def test_v4_hash_router_and_score_router_contract() -> None:
    model = _tiny_model(max_position_embeddings=16)
    hash_gate = model.layers[0].ffn.gate
    score_gate = model.layers[1].ffn.gate
    hidden = torch.randn(3, model.config.hidden_size)
    token_ids = torch.tensor([1, 7, 13])

    hash_weights, hash_indices = hash_gate(hidden, token_ids)
    score_weights, score_indices = score_gate(hidden, token_ids)

    assert torch.equal(hash_indices, hash_gate.tid2eid[token_ids])
    assert hash_weights.shape == score_weights.shape == (3, model.config.num_experts_per_tok)
    assert score_indices.shape == hash_indices.shape
    torch.testing.assert_close(hash_weights.sum(-1), torch.full((3,), model.config.routed_scaling_factor))
    torch.testing.assert_close(score_weights.sum(-1), torch.full((3,), model.config.routed_scaling_factor))


def test_v4_native_save_load_and_auto_detection(tmp_path) -> None:
    model = _tiny_model(max_position_embeddings=16)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model.save_pretrained(tmp_path)

    loaded = DeepSeekV4ForCausalLM.from_pretrained(tmp_path).eval()
    automatic = load_model_auto(tmp_path).eval()
    with torch.inference_mode():
        expected, _ = model(input_ids, use_cache=False)
        actual, _ = loaded(input_ids, use_cache=False)
        auto_logits, _ = automatic(input_ids, use_cache=False)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(auto_logits, expected)


def test_model_identity_rejects_cross_family_relabel(tmp_path) -> None:
    config = tiny_deepseek_v4_config(max_position_embeddings=16).to_dict()
    config["architectures"] = ["DeepSeekV32ForCausalLM"]
    assert detect_model_identity({"model_type": "deepseek_v4"}) == "deepseek-v4"
    with pytest.raises(ValueError, match="conflicting model identity"):
        detect_model_identity(config)

    v32 = DeepSeekV32Config().to_dict()
    v32["model_type"] = "deepseek_v4"
    with pytest.raises(ValueError, match="expected model_type='deepseek_v32'"):
        DeepSeekV32Config.from_dict(v32)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting model identity"):
        load_model_auto(tmp_path)


def test_v4_state_names_follow_public_checkpoint_contract() -> None:
    state = _tiny_model(max_position_embeddings=16).state_dict()

    assert "embed.weight" in state
    assert "layers.0.hc_attn_fn" in state
    assert "layers.0.attn.wq_a.weight" in state
    assert "layers.1.attn.compressor.ape" in state
    assert "layers.1.attn.indexer.compressor.wgate.weight" in state
    assert "layers.0.ffn.gate.tid2eid" in state
    assert "layers.1.ffn.gate.bias" in state
    assert "hc_head_fn" in state
    assert "head.weight" in state


def test_v4_public_quantization_reference_dequantizers() -> None:
    fp8 = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    fp8_scale = torch.tensor([[2.0]], dtype=torch.float8_e8m0fnu)
    torch.testing.assert_close(
        dequantize_block_fp8(fp8, fp8_scale),
        torch.tensor([[2.0, -4.0], [6.0, 8.0]]),
    )

    packed = torch.zeros(1, 16, dtype=torch.int8)
    packed[0, 0] = 0x21
    scale = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)
    unpacked = dequantize_mxfp4(packed, scale)
    assert tuple(unpacked.shape) == (1, 32)
    torch.testing.assert_close(unpacked[0, :2], torch.tensor([0.5, 1.0]))


def test_v4_manifest_audit_is_metadata_only_and_strict(tmp_path) -> None:
    config = tiny_deepseek_v4_config(
        num_nextn_predict_layers=0,
        expert_dtype="fp4",
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    names = expected_deepseek_v4_tensor_names(config)
    weight_map = {name: "model-00001-of-00001.safetensors" for name in names}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map})
    )

    report = audit_deepseek_v4_checkpoint(tmp_path)
    assert report.compatible
    assert not report.weights_available
    assert report.tensor_count == len(names)
    assert report.total_size == 123

    weight_map.pop("layers.0.attn.attn_sink")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    invalid = audit_deepseek_v4_checkpoint(tmp_path)
    assert not invalid.compatible
    assert any(issue.code == "missing_tensor" and issue.tensor == "layers.0.attn.attn_sink" for issue in invalid.issues)
