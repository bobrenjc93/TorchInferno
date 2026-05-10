from __future__ import annotations

from torchinferno.models.dsv4 import DSv4Config, DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.models.dsv4_family import fused_ops


class DSv4V1ForCausalLM(DSv4ForCausalLM):
    """DSv4 v1 fused/cached variant.

    v1 is the current stable DSv4 implementation with kernel-backed RMSNorm and
    SwiGLU plus incremental cached decode. It is intentionally a child of v0 in
    the registry, so alternative v1 branches can still start from v0 later.
    """

    provenance_variant = "dsv4:v1"
    ops = fused_ops

    def __init__(self, config: DSv4Config) -> None:
        super().__init__(config, fused_ops)


def tiny_dsv4_v1_config(**overrides: int | float | bool) -> DSv4Config:
    return tiny_dsv4_config(**overrides)
