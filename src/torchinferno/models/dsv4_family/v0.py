from __future__ import annotations

import torch
from torch import Tensor, nn

from torchinferno.models.dsv4 import (
    DSv4Config,
    DSv4ForCausalLM,
    RMSNorm,
    SwiGLUExpert,
    sample_next_token,
    tiny_dsv4_config,
)
from torchinferno.models.dsv4_family import raw_ops


class RawRMSNorm(nn.Module):
    def __init__(self, source: RMSNorm) -> None:
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.eps = source.eps

    def forward(self, x: Tensor) -> Tensor:
        return raw_ops.rms_norm(x, self.weight, self.eps)


class RawSwiGLUExpert(nn.Module):
    def __init__(self, source: SwiGLUExpert) -> None:
        super().__init__()
        self.w1 = source.w1
        self.w3 = source.w3
        self.w2 = source.w2

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(raw_ops.swiglu(self.w1(x), self.w3(x)))


class DSv4V0ForCausalLM(DSv4ForCausalLM):
    """DSv4 v0 provenance baseline.

    v0 keeps the existing torch-native DSv4 tensor contracts but deliberately
    disables cached decode in `generate` and routes provenance-visible scalar
    operations through `raw_ops.py`. The next step for this family is to move
    every module body into this package so v0 is entirely independent of the
    stable DSv4 facade.
    """

    provenance_variant = "dsv4:v0"
    ops = raw_ops

    def __init__(self, config: DSv4Config) -> None:
        super().__init__(config)
        _replace_raw_modules(self)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        output = input_ids
        for _ in range(max_new_tokens):
            if output.size(1) > self.config.max_seq_len:
                raise ValueError("input sequence exceeds configured max_seq_len")
            logits, _ = self(output, use_cache=False)
            next_token = sample_next_token(logits[:, -1, :], temperature)
            output = torch.cat([output, next_token[:, None]], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return output


def tiny_dsv4_v0_config(**overrides: int | float | bool) -> DSv4Config:
    return tiny_dsv4_config(**overrides)


def _replace_raw_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, RMSNorm):
            setattr(module, name, RawRMSNorm(child))
        elif isinstance(child, SwiGLUExpert):
            setattr(module, name, RawSwiGLUExpert(child))
        else:
            _replace_raw_modules(child)
