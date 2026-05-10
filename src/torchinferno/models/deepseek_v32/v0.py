from __future__ import annotations

import torch
from torch import Tensor

from torchinferno.models.deepseek_v32 import (
    DeepSeekV32Config,
    DeepSeekV32ForCausalLM,
    tiny_deepseek_v32_config,
)
from torchinferno.models.deepseek_v32 import raw_ops
from torchinferno.runtime.sampling import sample_next_token


class DeepSeekV32V0ForCausalLM(DeepSeekV32ForCausalLM):
    """DeepSeek-V3.2 v0 provenance baseline.

    v0 uses full-prefix recompute generation and keeps raw-op provenance beside
    the family. The stable native class still supplies the production tensor
    contract while this package gives us a place to branch raw/fused variants.
    """

    provenance_variant = "deepseek-v3.2:v0"
    ops = raw_ops

    def __init__(self, config: DeepSeekV32Config) -> None:
        super().__init__(config, raw_ops)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
        cache_backend: str = "dense",
        page_size: int = 16,
    ) -> Tensor:
        del cache_backend, page_size
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        output = input_ids
        for _ in range(max_new_tokens):
            if output.size(1) > self.config.max_position_embeddings:
                raise ValueError("input sequence exceeds configured max_position_embeddings")
            logits, _ = self(output, use_cache=False)
            next_token = sample_next_token(logits[:, -1, :], temperature)
            output = torch.cat([output, next_token[:, None]], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return output


def tiny_deepseek_v32_v0_config(**overrides: int | float | bool | str | None) -> DeepSeekV32Config:
    return tiny_deepseek_v32_config(**overrides)
