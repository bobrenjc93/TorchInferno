import importlib
import os
import subprocess
import sys

import torch

from torchinferno.models.deepseek_v32_family import (
    DeepSeekV32V0ForCausalLM,
    DeepSeekV32V1ForCausalLM,
    tiny_deepseek_v32_v0_config,
)
from torchinferno.models.dsv4_family import DSv4V0ForCausalLM, DSv4V1ForCausalLM, tiny_dsv4_v0_config
from torchinferno.models.llama3_family import (
    Llama3V0ForCausalLM,
    Llama3V1ForCausalLM,
    llama3_70b_config,
    tiny_llama3_config,
)
from torchinferno.models.variants import get_model_variant, list_model_variants, model_variant_lineage


def test_model_variant_registry_tracks_families_and_ops_modules() -> None:
    specs = list_model_variants()
    ids = {spec.id for spec in specs}

    assert {"dsv4:v0", "dsv4:v1", "deepseek-v3.2:v0", "deepseek-v3.2:v1", "llama3:v0", "llama3:v1"} <= ids
    assert [spec.variant for spec in model_variant_lineage("llama3", "v1")] == ["v0", "v1"]
    assert [spec.variant for spec in model_variant_lineage("dsv3.2", "v1")] == ["v0", "v1"]
    assert get_model_variant("dsv4", "v1").parents == ("v0",)
    for spec in specs:
        importlib.import_module(spec.ops_module)
        module = importlib.import_module(spec.module)
        assert hasattr(module, spec.class_name)


def test_llama3_v0_and_v1_are_weight_compatible() -> None:
    torch.manual_seed(50)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    v0 = Llama3V0ForCausalLM(config).eval()
    v1 = Llama3V1ForCausalLM(config).eval()
    v1.load_state_dict(v0.state_dict())
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    with torch.inference_mode():
        expected = v0(input_ids)
        actual = v1(input_ids)
        generated = v0.generate(input_ids[:1], max_new_tokens=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert generated.shape == (1, 5)


def test_llama3_70b_config_matches_public_architecture_shape() -> None:
    config = llama3_70b_config()

    assert config.hidden_size == 8192
    assert config.intermediate_size == 28672
    assert config.num_hidden_layers == 80
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.max_position_embeddings == 131072
    assert config.rope_scaling is not None
    assert config.rope_scaling["rope_type"] == "llama3"


def test_dsv4_and_deepseek_v0_match_v1_greedy_generation() -> None:
    torch.manual_seed(51)
    dsv4_config = tiny_dsv4_v0_config(vocab_size=32, max_seq_len=16)
    dsv4_v0 = DSv4V0ForCausalLM(dsv4_config).eval()
    dsv4_v1 = DSv4V1ForCausalLM(dsv4_config).eval()
    dsv4_v1.load_state_dict(dsv4_v0.state_dict())
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    deepseek_config = tiny_deepseek_v32_v0_config(vocab_size=32, max_position_embeddings=16)
    deepseek_v0 = DeepSeekV32V0ForCausalLM(deepseek_config).eval()
    deepseek_v1 = DeepSeekV32V1ForCausalLM(deepseek_config).eval()
    deepseek_v1.load_state_dict(deepseek_v0.state_dict())

    with torch.inference_mode():
        assert torch.equal(dsv4_v0.generate(input_ids, max_new_tokens=2), dsv4_v1.generate(input_ids, max_new_tokens=2))
        assert torch.equal(
            deepseek_v0.generate(input_ids, max_new_tokens=2),
            deepseek_v1.generate(input_ids, max_new_tokens=2),
        )


def test_model_variants_cli_lists_lineage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "model-variants",
            "--family",
            "llama3",
            "--lineage",
            "v1",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert "llama3:v0" in result.stdout
    assert "llama3:v1" in result.stdout
    assert "ops=torchinferno.models.llama3_family.fused_ops" in result.stdout
