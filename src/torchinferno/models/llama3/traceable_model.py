from __future__ import annotations

from torch import Tensor

from torchinferno.models.fx import FXGraphBackedMixin, default_input_ids
from torchinferno.models.llama3 import raw_ops
from torchinferno.models.llama3.config import Llama3Config
from torchinferno.models.llama3.model import _Llama3ForCausalLMBase


class TraceableLlama3ForCausalLM(FXGraphBackedMixin, _Llama3ForCausalLMBase):
    """Llama3 graph-capture form used by the v0 provenance variant."""

    ops = raw_ops

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config, raw_ops)

    def _default_v0_input_ids(self) -> Tensor:
        return default_input_ids(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_position_embeddings,
            device=self.embed_tokens.weight.device,
        )

    def _traceable_forward(self, input_ids: Tensor) -> Tensor:
        return _Llama3ForCausalLMBase.forward(self, input_ids)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self._run_v0_graph(input_ids)
