import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from torchinferno.kernels import KernelBackend, KernelConfig, rms_norm, swiglu_activation
from torchinferno.kernels.ops import triton_available
from torchinferno.models.conversion import (
    audit_deepseek_checkpoint,
    convert_deepseek_checkpoint,
    deepseek_to_dsv4_key_map,
)
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config


def _write_native_deepseek_style_checkpoint(path, model: DSv4ForCausalLM) -> None:
    path.mkdir(parents=True)
    config = model.config.to_dict()
    config.update(
        {
            "model_type": "deepseek_v32",
            "num_hidden_layers": model.config.num_layers,
            "num_experts_per_tok": model.config.top_k,
            "n_routed_experts": model.config.num_experts,
            "moe_intermediate_size": model.config.intermediate_size,
            "kv_lora_rank": model.config.latent_kv_size,
        }
    )
    (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    state_dict = model.state_dict()
    key_map = deepseek_to_dsv4_key_map(model.config)
    native_state = {}
    for target_name, candidates in key_map.items():
        if target_name in state_dict:
            native_state[candidates[0]] = state_dict[target_name].detach().clone()
    save_file(native_state, path / "model.safetensors", metadata={"format": "pt"})


def test_streaming_decode_attention_block_size_prefers_larger_single_batch(monkeypatch) -> None:
    from torchinferno.kernels.triton_ops import _streaming_decode_attention_block_s

    monkeypatch.delenv("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S", raising=False)

    assert _streaming_decode_attention_block_s(1) == 64
    assert _streaming_decode_attention_block_s(2) == 64

    monkeypatch.setenv("TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S", "32")
    assert _streaming_decode_attention_block_s(1) == 32
    assert _streaming_decode_attention_block_s(4) == 32


def test_deepseek_style_checkpoint_conversion_round_trip(tmp_path) -> None:
    torch.manual_seed(10)
    config = tiny_dsv4_config(vocab_size=32, max_seq_len=16)
    model = DSv4ForCausalLM(config).eval()
    source = tmp_path / "native"
    output = tmp_path / "converted"
    _write_native_deepseek_style_checkpoint(source, model)

    report = convert_deepseek_checkpoint(source, output, max_shard_size="32KB")
    loaded = DSv4ForCausalLM.from_pretrained(output).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert report.compatible
    assert (output / "torchinferno_conversion_report.json").exists()
    with torch.inference_mode():
        expected, _ = model(input_ids, use_cache=False)
        actual, _ = loaded(input_ids, use_cache=False)
    torch.testing.assert_close(actual, expected)


def test_deepseek_audit_reports_unsupported_native_contract(tmp_path) -> None:
    torch.manual_seed(11)
    model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=32, max_seq_len=16)).eval()
    source = tmp_path / "native"
    _write_native_deepseek_style_checkpoint(source, model)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["q_lora_rank"] = 8
    config_path.write_text(json.dumps(config) + "\n")

    report = audit_deepseek_checkpoint(source)

    assert not report.compatible
    assert any(issue.code == "unsupported_config" for issue in report.issues)


def test_conversion_cli_audit_and_convert(tmp_path) -> None:
    torch.manual_seed(12)
    model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=32, max_seq_len=16)).eval()
    source = tmp_path / "native"
    output = tmp_path / "converted"
    _write_native_deepseek_style_checkpoint(source, model)
    env = {**os.environ, "PYTHONPATH": "src"}

    audit = subprocess.run(
        [sys.executable, "-m", "torchinferno.cli", "dsv4-audit", str(source)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    convert = subprocess.run(
        [sys.executable, "-m", "torchinferno.cli", "dsv4-convert", str(source), str(output), "--max-shard-size", "32KB"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "compatible" in audit.stdout
    assert "wrote=" in convert.stdout
    assert (output / "config.json").exists()


def test_kernel_fallbacks_match_torch_reference() -> None:
    torch.manual_seed(13)
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    x = torch.randn(2, 3, 8)
    weight = torch.randn(8)

    torch.testing.assert_close(swiglu_activation(gate, up), torch.nn.functional.silu(gate) * up)
    provided = torch.empty_like(gate)
    actual = swiglu_activation(gate, up, out=provided)
    assert actual is provided
    torch.testing.assert_close(provided, torch.nn.functional.silu(gate) * up)
    expected_norm = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype) * weight
    torch.testing.assert_close(rms_norm(x, weight, eps=1e-6), expected_norm)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_cuda_kernels_match_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_add_rms_norm

    torch.manual_seed(14)
    gate = torch.randn(16, 32, device="cuda")
    up = torch.randn(16, 32, device="cuda")
    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16 = up.to(torch.bfloat16)
    x = torch.randn(4, 8, 32, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(32, device="cuda")
    config = KernelConfig(backend=KernelBackend.TRITON)

    torch.testing.assert_close(
        swiglu_activation(gate, up, config=config),
        torch.nn.functional.silu(gate) * up,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        swiglu_activation(gate_bf16, up_bf16, config=config),
        torch.nn.functional.silu(gate_bf16) * up_bf16,
        atol=2e-2,
        rtol=2e-2,
    )
    provided = torch.empty_like(gate)
    actual = swiglu_activation(gate, up, out=provided, config=config)
    assert actual is provided
    torch.testing.assert_close(
        provided,
        torch.nn.functional.silu(gate) * up,
        atol=1e-5,
        rtol=1e-5,
    )
    expected_norm = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype) * weight
    torch.testing.assert_close(rms_norm(x, weight, eps=1e-6, config=config), expected_norm, atol=1e-5, rtol=1e-5)
    expected_hidden = x + residual
    expected_add_norm = (
        expected_hidden
        * torch.rsqrt(expected_hidden.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)
        * weight
    )
    actual_hidden, actual_add_norm = triton_add_rms_norm(x, residual, weight, eps=1e-6)
    torch.testing.assert_close(actual_hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_add_norm, expected_add_norm, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_rotary_interleaved_inplace_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_apply_rotary_interleaved_inplace
    from torchinferno.models.llama3.tensor_parallel import _rotate_interleaved_eager

    torch.manual_seed(15)
    batch, tokens, q_heads, kv_heads, head_dim = 3, 5, 4, 1, 16
    packed = torch.randn(
        batch,
        tokens,
        (q_heads + kv_heads + kv_heads) * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, _ = packed.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1)
    q = q.view(batch, tokens, q_heads, head_dim).transpose(1, 2)
    k = k.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    freqs = torch.randn(tokens, head_dim // 2, device="cuda")
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    expected_q = _rotate_interleaved_eager(q, cos[None, None, :, :], sin[None, None, :, :])
    expected_k = _rotate_interleaved_eager(k, cos[None, None, :, :], sin[None, None, :, :])

    actual_q, actual_k = triton_apply_rotary_interleaved_inplace(q.clone(), k.clone(), cos, sin)

    torch.testing.assert_close(actual_q, expected_q, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_rotary_llama_inplace_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_inplace
    from torchinferno.models.llama3.tensor_parallel import _rotate_llama_eager

    torch.manual_seed(16)
    batch, tokens, q_heads, kv_heads, head_dim = 3, 5, 4, 1, 16
    packed = torch.randn(
        batch,
        tokens,
        (q_heads + kv_heads + kv_heads) * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, _ = packed.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1)
    q = q.view(batch, tokens, q_heads, head_dim).transpose(1, 2)
    k = k.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    freqs = torch.randn(tokens, head_dim // 2, device="cuda")
    cos_half = freqs.cos().to(torch.bfloat16)
    sin_half = freqs.sin().to(torch.bfloat16)
    cos_full = torch.cat((cos_half, cos_half), dim=-1)
    sin_full = torch.cat((sin_half, sin_half), dim=-1)
    expected_q = _rotate_llama_eager(q, cos_full[None, None, :, :], sin_full[None, None, :, :])
    expected_k = _rotate_llama_eager(k, cos_full[None, None, :, :], sin_full[None, None, :, :])

    actual_q, actual_k = triton_apply_rotary_llama_inplace(q.clone(), k.clone(), cos_half, sin_half)

    torch.testing.assert_close(actual_q, expected_q, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_rotary_llama_batched_inplace_matches_torch_reference() -> None:
    # Per-(batch,token) variant used by the decode / ragged-suffix rope: each row
    # carries its OWN positions (cos/sin are [batch, tokens, dim]), unlike the
    # shared-across-batch kernel above. Validates it matches _rotate_llama_eager.
    from torchinferno.kernels.triton_ops import triton_apply_rotary_llama_batched_inplace
    from torchinferno.models.llama3.tensor_parallel import _rotate_llama_eager

    torch.manual_seed(17)
    batch, tokens, q_heads, kv_heads, head_dim = 4, 1, 8, 1, 128
    packed = torch.randn(
        batch, tokens, (q_heads + kv_heads + kv_heads) * head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    q, k, _ = packed.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1)
    q = q.view(batch, tokens, q_heads, head_dim).transpose(1, 2)
    k = k.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    freqs = torch.randn(batch, tokens, head_dim // 2, device="cuda")
    cos_half = freqs.cos().to(torch.bfloat16)
    sin_half = freqs.sin().to(torch.bfloat16)
    cos_full = torch.cat((cos_half, cos_half), dim=-1)
    sin_full = torch.cat((sin_half, sin_half), dim=-1)
    expected_q = _rotate_llama_eager(q, cos_full[:, None, :, :], sin_full[:, None, :, :])
    expected_k = _rotate_llama_eager(k, cos_full[:, None, :, :], sin_full[:, None, :, :])

    actual_q, actual_k = triton_apply_rotary_llama_batched_inplace(q.clone(), k.clone(), cos_full, sin_full)

    torch.testing.assert_close(actual_q, expected_q, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_kv_cache_append_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_append_kv_cache

    torch.manual_seed(17)
    batch, heads, tokens, head_dim, max_seq_len = 2, 3, 4, 16, 12
    packed = torch.randn(batch, tokens, heads * head_dim * 2, device="cuda", dtype=torch.bfloat16)
    keys, values = packed.split(heads * head_dim, dim=-1)
    keys = keys.view(batch, tokens, heads, head_dim).transpose(1, 2)
    values = values.view(batch, tokens, heads, head_dim).transpose(1, 2)
    cache_keys = torch.zeros(batch, heads, max_seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_values = torch.zeros_like(cache_keys)
    expected_keys = cache_keys.clone()
    expected_values = cache_values.clone()
    seq_start = 3

    expected_keys[:, :, seq_start : seq_start + tokens, :].copy_(keys)
    expected_values[:, :, seq_start : seq_start + tokens, :].copy_(values)
    triton_append_kv_cache(keys, values, cache_keys, cache_values, seq_start)

    torch.testing.assert_close(cache_keys, expected_keys)
    torch.testing.assert_close(cache_values, expected_values)

    cache_keys.zero_()
    cache_values.zero_()
    dynamic_seq_start = torch.tensor(seq_start, device="cuda", dtype=torch.int64)
    triton_append_kv_cache(keys, values, cache_keys, cache_values, dynamic_seq_start)
    torch.testing.assert_close(cache_keys, expected_keys)
    torch.testing.assert_close(cache_values, expected_values)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_decode_rotary_append_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_apply_rotary_append_kv_decode
    from torchinferno.models.llama3.tensor_parallel import _rotate_llama_eager

    torch.manual_seed(19)
    batch, q_heads, kv_heads, head_dim, max_seq_len = 2, 4, 1, 16, 12
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_keys = torch.zeros(batch, kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_values = torch.zeros_like(cache_keys)
    freqs = torch.randn(1, head_dim // 2, device="cuda")
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    seq_start = torch.tensor(3, device="cuda", dtype=torch.int64)

    expected_q = _rotate_llama_eager(q, cos[None, None, :, :], sin[None, None, :, :])
    expected_k = _rotate_llama_eager(k, cos[None, None, :, :], sin[None, None, :, :])
    expected_cache_keys = cache_keys.clone()
    expected_cache_values = cache_values.clone()
    expected_cache_keys[:, :, int(seq_start.item()) : int(seq_start.item()) + 1, :].copy_(expected_k)
    expected_cache_values[:, :, int(seq_start.item()) : int(seq_start.item()) + 1, :].copy_(v)

    actual_q = triton_apply_rotary_append_kv_decode(q.clone(), k, v, cache_keys, cache_values, seq_start, cos, sin)

    torch.testing.assert_close(actual_q, expected_q, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(cache_keys, expected_cache_keys, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(cache_values, expected_cache_values)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_ragged_decode_rotary_append_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_apply_rotary_append_kv_ragged_decode
    from torchinferno.models.llama3.tensor_parallel import _rotate_llama_eager

    torch.manual_seed(20)
    batch, q_heads, kv_heads, head_dim, max_seq_len = 3, 4, 1, 16, 12
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_keys = torch.zeros(batch + 1, kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_values = torch.zeros_like(cache_keys)
    freqs = torch.randn(batch, head_dim // 2, device="cuda")
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    positions = torch.tensor([3, 5, 4], device="cuda", dtype=torch.int64)
    row_indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)

    expected_q = _rotate_llama_eager(q, cos[:, None, None, :], sin[:, None, None, :])
    expected_k = _rotate_llama_eager(k, cos[:, None, None, :], sin[:, None, None, :])
    expected_cache_keys = cache_keys.clone()
    expected_cache_values = cache_values.clone()
    for source_row, target_row in enumerate(row_indices.detach().cpu().tolist()):
        pos = int(positions[source_row].item())
        expected_cache_keys[target_row : target_row + 1, :, pos : pos + 1, :].copy_(
            expected_k[source_row : source_row + 1]
        )
        expected_cache_values[target_row : target_row + 1, :, pos : pos + 1, :].copy_(
            v[source_row : source_row + 1]
        )

    actual_q = triton_apply_rotary_append_kv_ragged_decode(
        q.clone(),
        k,
        v,
        cache_keys,
        cache_values,
        positions,
        cos,
        sin,
        row_indices,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(cache_keys, expected_cache_keys, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(cache_values, expected_cache_values)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_dense_gqa_decode_attention_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_dense_gqa_decode_attention, triton_grouped_gqa_decode_attention

    torch.manual_seed(17)
    batch, q_heads, kv_heads, seq_len, head_dim = 2, 4, 2, 13, 16
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )

    actual = triton_dense_gqa_decode_attention(q, k, v)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)

    padded_k = torch.zeros(batch, kv_heads, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    padded_v = torch.zeros_like(padded_k)
    padded_k[:, :, :seq_len, :].copy_(k)
    padded_v[:, :, :seq_len, :].copy_(v)
    dynamic_seq_len = torch.tensor(seq_len, device="cuda", dtype=torch.int64)
    dynamic_actual = triton_dense_gqa_decode_attention(q, padded_k, padded_v, seq_len=dynamic_seq_len)

    torch.testing.assert_close(dynamic_actual, expected, atol=4e-2, rtol=4e-2)

    grouped_actual = triton_grouped_gqa_decode_attention(q, padded_k, padded_v, dynamic_seq_len)

    torch.testing.assert_close(grouped_actual, expected, atol=4e-2, rtol=4e-2)

    row_seq_lens = torch.tensor([seq_len, seq_len - 3], device="cuda", dtype=torch.int64)
    key_positions = torch.arange(padded_k.size(2), device="cuda")
    row_mask = key_positions[None, :] < row_seq_lens[:, None]
    row_expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        padded_k,
        padded_v,
        attn_mask=row_mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )
    row_actual = triton_grouped_gqa_decode_attention(q, padded_k, padded_v, row_seq_lens)

    torch.testing.assert_close(row_actual, row_expected, atol=4e-2, rtol=4e-2)

    cache_k = torch.randn(batch + 2, kv_heads, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_v = torch.randn_like(cache_k)
    row_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    indexed_k = cache_k.index_select(0, row_indices)
    indexed_v = cache_v.index_select(0, row_indices)
    indexed_expected = triton_grouped_gqa_decode_attention(q, indexed_k, indexed_v, row_seq_lens)
    indexed_actual = triton_grouped_gqa_decode_attention(
        q,
        cache_k,
        cache_v,
        row_seq_lens,
        row_indices=row_indices,
    )

    torch.testing.assert_close(indexed_actual, indexed_expected, atol=4e-2, rtol=4e-2)
