from __future__ import annotations

from torchinferno.models.deepseek_v32 import (
    DeepSeekV32Config,
    tiny_deepseek_v32_config,
)
from torchinferno.models.deepseek_v32.traceable_model import TraceableDeepSeekV32ForCausalLM


class DeepSeekV32V0ForCausalLM(TraceableDeepSeekV32ForCausalLM):
    """DeepSeek-V3.2 v0 make_fx provenance baseline.

    `model.py` owns the pure eager implementation. This v0 wrapper traces the
    no-cache full-prefix forward through `traceable_model.py`, caches the
    resulting make_fx graph per input shape, and exposes `print_readable()`.
    """

    provenance_variant = "deepseek-v3.2:v0"

    def __init__(self, config: DeepSeekV32Config) -> None:
        super().__init__(config)


def tiny_deepseek_v32_v0_config(**overrides: int | float | bool | str | None) -> DeepSeekV32Config:
    return tiny_deepseek_v32_config(**overrides)
