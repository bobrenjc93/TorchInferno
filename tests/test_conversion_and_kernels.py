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
    expected_norm = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype) * weight
    torch.testing.assert_close(rms_norm(x, weight, eps=1e-6), expected_norm)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_cuda_kernels_match_torch_reference() -> None:
    torch.manual_seed(14)
    gate = torch.randn(16, 32, device="cuda")
    up = torch.randn(16, 32, device="cuda")
    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16 = up.to(torch.bfloat16)
    x = torch.randn(4, 8, 32, device="cuda")
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
    expected_norm = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype) * weight
    torch.testing.assert_close(rms_norm(x, weight, eps=1e-6, config=config), expected_norm, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="CUDA Triton kernels unavailable")
def test_triton_rotary_interleaved_inplace_matches_torch_reference() -> None:
    from torchinferno.kernels.triton_ops import triton_apply_rotary_interleaved_inplace
    from torchinferno.models.llama3_family.tensor_parallel import _rotate_interleaved_eager

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
