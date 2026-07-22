from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class DeepSeekV4Config:
    """DeepSeek-V4 model contract.

    Field names follow the public ``deepseek-ai/DeepSeek-V4-Flash`` config.
    V4 is intentionally a separate family from TorchInferno's compact DSv4
    workbench and the native DeepSeek-V3.2 implementation.
    """

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    sliding_window: int = 128
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    rope_original_max_position_embeddings: int = 65536
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    compress_ratios: tuple[int, ...] = (
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    )
    torch_dtype: str = "bfloat16"
    expert_dtype: str | None = "fp4"
    quantization_config: dict[str, Any] | None = None
    simulate_qat: bool = True

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "q_lora_rank": self.q_lora_rank,
            "head_dim": self.head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "o_groups": self.o_groups,
            "o_lora_rank": self.o_lora_rank,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "hc_mult": self.hc_mult,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek V4 requires exactly one key/value head")
        if self.qk_rope_head_dim > self.head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even and no larger than head_dim")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if not 0 <= self.num_hash_layers <= self.num_hidden_layers:
            raise ValueError("num_hash_layers must be in [0, num_hidden_layers]")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek V4 currently requires one shared expert")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek V4 requires sqrtsoftplus expert scoring")
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError("compress_ratios must cover every decoder layer")
        if any(ratio not in {0, 4, 128} for ratio in self.compress_ratios):
            raise ValueError("DeepSeek V4 compression ratios must be 0, 4, or 128")
        if self.hc_sinkhorn_iters < 1:
            raise ValueError("hc_sinkhorn_iters must be positive")
        if self.hc_eps <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("normalization epsilons must be positive")
        if self.expert_dtype not in {None, "bf16", "fp4"}:
            raise ValueError("expert_dtype must be None, 'bf16', or 'fp4'")

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compress_ratios"] = list(self.compress_ratios)
        data["model_type"] = "deepseek_v4"
        data["architectures"] = ["DeepseekV4ForCausalLM"]
        data["rope_scaling"] = {
            "type": "yarn",
            "factor": self.rope_factor,
            "original_max_position_embeddings": self.rope_original_max_position_embeddings,
            "beta_fast": self.rope_beta_fast,
            "beta_slow": self.rope_beta_slow,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepSeekV4Config":
        model_type = str(data.get("model_type", "")).lower()
        architectures = {str(value).lower() for value in data.get("architectures", ())}
        if model_type and model_type != "deepseek_v4":
            raise ValueError(f"expected model_type='deepseek_v4', got {model_type!r}")
        if architectures and not architectures.intersection(
            {"deepseekv4forcausallm", "deepseek_v4forcausallm"}
        ):
            raise ValueError(f"checkpoint architectures do not identify DeepSeek V4: {sorted(architectures)}")

        normalized = dict(data)
        aliases = {
            "n_hash_layers": "num_hash_layers",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_activated_experts": "num_experts_per_tok",
            "score_func": "scoring_func",
            "route_scale": "routed_scaling_factor",
            "dim": "hidden_size",
            "moe_inter_dim": "moe_intermediate_size",
            "rope_head_dim": "qk_rope_head_dim",
            "window_size": "sliding_window",
            "norm_eps": "rms_norm_eps",
            "original_seq_len": "rope_original_max_position_embeddings",
            "beta_fast": "rope_beta_fast",
            "beta_slow": "rope_beta_slow",
        }
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        rope_scaling = normalized.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_aliases = {
                "factor": "rope_factor",
                "original_max_position_embeddings": "rope_original_max_position_embeddings",
                "beta_fast": "rope_beta_fast",
                "beta_slow": "rope_beta_slow",
            }
            for source, target in rope_aliases.items():
                if source in rope_scaling:
                    normalized[target] = rope_scaling[source]
        if "compress_ratios" in normalized:
            normalized["compress_ratios"] = tuple(int(value) for value in normalized["compress_ratios"])
        if "quantization_config" in normalized and normalized["quantization_config"] is not None:
            normalized["quantization_config"] = dict(normalized["quantization_config"])
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in allowed})


def tiny_deepseek_v4_config(**overrides: Any) -> DeepSeekV4Config:
    """Small, unquantized V4 with the released layer kinds intact."""

    config = DeepSeekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=24,
        num_hidden_layers=4,
        num_hash_layers=1,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_experts_per_tok=2,
        n_routed_experts=4,
        q_lora_rank=16,
        head_dim=16,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=16,
        sliding_window=8,
        max_position_embeddings=256,
        rope_original_max_position_embeddings=32,
        rope_factor=4.0,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        compress_ratios=(0, 4, 128, 0),
        torch_dtype="float32",
        expert_dtype=None,
        quantization_config=None,
        simulate_qat=False,
    )
    return replace(config, **overrides)
