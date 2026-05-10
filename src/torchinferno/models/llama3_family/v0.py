from __future__ import annotations

from types import ModuleType

import torch
from torch import Tensor, nn

from torchinferno.models.llama3_family import raw_ops
from torchinferno.models.llama3_family.config import Llama3Config, tiny_llama3_config
from torchinferno.runtime.sampling import sample_next_token


class Llama3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, ops: ModuleType = raw_ops) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        return self.ops.rms_norm(x, self.weight, self.eps)


class Llama3Attention(nn.Module):
    def __init__(self, config: Llama3Config, ops: ModuleType = raw_ops) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        q = self.q_proj(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        cos, sin = self.ops.rotary_cache(head_dim, positions, self.config.rope_theta)
        q = self.ops.apply_rotary(q, cos, sin)
        k = self.ops.apply_rotary(k, cos, sin)
        repeats = heads // kv_heads
        k = self.ops.repeat_kv(k, repeats)
        v = self.ops.repeat_kv(v, repeats)
        out = self.ops.causal_attention(q, k, v, positions)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, heads * head_dim)
        return self.o_proj(out)


class Llama3MLP(nn.Module):
    def __init__(self, config: Llama3Config, ops: ModuleType = raw_ops) -> None:
        super().__init__()
        self.ops = ops
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.ops.swiglu(self.gate_proj(x), self.up_proj(x)))


class Llama3DecoderLayer(nn.Module):
    def __init__(self, config: Llama3Config, ops: ModuleType = raw_ops) -> None:
        super().__init__()
        self.input_layernorm = Llama3RMSNorm(config.hidden_size, config.rms_norm_eps, ops)
        self.self_attn = Llama3Attention(config, ops)
        self.post_attention_layernorm = Llama3RMSNorm(config.hidden_size, config.rms_norm_eps, ops)
        self.mlp = Llama3MLP(config, ops)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _Llama3ForCausalLMBase(nn.Module):
    provenance_variant = "llama3:base"

    def __init__(self, config: Llama3Config, ops: ModuleType = raw_ops) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Llama3DecoderLayer(config, ops) for _ in range(config.num_hidden_layers)])
        self.norm = Llama3RMSNorm(config.hidden_size, config.rms_norm_eps, ops)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) < 1:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds configured max_position_embeddings")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.norm(hidden))

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
            logits = self(output)
            next_token = sample_next_token(logits[:, -1, :], temperature)
            output = torch.cat([output, next_token[:, None]], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return output


class Llama3V0ForCausalLM(_Llama3ForCausalLMBase):
    provenance_variant = "llama3:v0"

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config, raw_ops)

def tiny_llama3_v0_config(**overrides: int | float | bool) -> Llama3Config:
    return tiny_llama3_config(**overrides)
