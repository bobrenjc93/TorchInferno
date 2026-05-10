from __future__ import annotations

from torchinferno.models.dsv4 import (
    DSv4Config,
    tiny_dsv4_config,
)
from torchinferno.models.dsv4.traceable_model import TraceableDSv4ForCausalLM


class DSv4V0ForCausalLM(TraceableDSv4ForCausalLM):
    """DSv4 v0 make_fx provenance baseline.

    `model.py` owns the pure eager implementation. This v0 wrapper traces the
    no-cache full-prefix forward through `traceable_model.py`, caches the
    resulting make_fx graph per input shape, and exposes `print_readable()`.
    """

    provenance_variant = "dsv4:v0"

    def __init__(self, config: DSv4Config) -> None:
        super().__init__(config)


def tiny_dsv4_v0_config(**overrides: int | float | bool) -> DSv4Config:
    return tiny_dsv4_config(**overrides)
