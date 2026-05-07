from __future__ import annotations

from torchinferno.models.llama3_family import fused_ops
from torchinferno.models.llama3_family.config import Llama3Config, tiny_llama3_config
from torchinferno.models.llama3_family.v0 import _Llama3ForCausalLMBase


class Llama3V1ForCausalLM(_Llama3ForCausalLMBase):
    """Llama3 v1 fused-op variant.

    It preserves v0 module boundaries and parameter names while swapping the
    operation module from `raw_ops.py` to `fused_ops.py`.
    """

    provenance_variant = "llama3:v1"

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config, fused_ops)


def tiny_llama3_v1_config(**overrides: int | float | bool) -> Llama3Config:
    return tiny_llama3_config(**overrides)
