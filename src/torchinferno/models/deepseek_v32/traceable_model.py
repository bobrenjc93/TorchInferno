from __future__ import annotations

import torch
from torch import Tensor

from torchinferno.models.deepseek_v32 import raw_ops
from torchinferno.models.deepseek_v32.model import DeepSeekV32Config, DeepSeekV32ForCausalLM
from torchinferno.models.fx import FXGraphBackedMixin, default_input_ids
from torchinferno.runtime.sampling import sample_next_token


class TraceableDeepSeekV32ForCausalLM(FXGraphBackedMixin, DeepSeekV32ForCausalLM):
    """DeepSeek-V3.2 graph-capture form used by the v0 provenance variant."""

    ops = raw_ops

    def __init__(self, config: DeepSeekV32Config) -> None:
        super().__init__(config, raw_ops)

    def _default_v0_input_ids(self) -> Tensor:
        return default_input_ids(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_position_embeddings,
            device=self.model.embed_tokens.weight.device,
        )

    def _traceable_forward(self, input_ids: Tensor) -> Tensor:
        logits, _ = DeepSeekV32ForCausalLM.forward(self, input_ids, cache=None, use_cache=False)
        return logits

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, None]:
        if cache is not None or use_cache:
            raise ValueError("DeepSeek-V3.2 v0 is a full-prefix make_fx graph and does not accept cache state")
        return self._run_v0_graph(input_ids), None

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
