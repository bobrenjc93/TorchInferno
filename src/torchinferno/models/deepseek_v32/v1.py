from __future__ import annotations

from torchinferno.models.deepseek_v32 import DeepSeekV32Config, DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.deepseek_v32 import fused_ops


class DeepSeekV32V1ForCausalLM(DeepSeekV32ForCausalLM):
    """DeepSeek-V3.2 v1 fused/cached native variant."""

    provenance_variant = "deepseek-v3.2:v1"
    ops = fused_ops

    def __init__(self, config: DeepSeekV32Config) -> None:
        super().__init__(config, fused_ops)


def tiny_deepseek_v32_v1_config(**overrides: int | float | bool | str | None) -> DeepSeekV32Config:
    return tiny_deepseek_v32_config(**overrides)
