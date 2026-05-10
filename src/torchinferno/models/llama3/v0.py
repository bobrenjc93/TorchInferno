from __future__ import annotations

from torchinferno.models.llama3.config import Llama3Config, tiny_llama3_config
from torchinferno.models.llama3.traceable_model import TraceableLlama3ForCausalLM


class Llama3V0ForCausalLM(TraceableLlama3ForCausalLM):
    """Llama3 v0 make_fx provenance baseline.

    `model.py` owns the pure eager implementation. This v0 wrapper traces the
    full-prefix forward through `traceable_model.py`, caches the resulting
    make_fx graph per input shape, and exposes `print_readable()`.
    """

    provenance_variant = "llama3:v0"

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config)


def tiny_llama3_v0_config(**overrides: int | float | bool) -> Llama3Config:
    return tiny_llama3_config(**overrides)
