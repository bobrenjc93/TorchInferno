from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace


@dataclass
class Llama3Config:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rope_scaling: dict[str, object] | None = None
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_type"] = "llama3"
        data["architectures"] = ["Llama3V0ForCausalLM"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Llama3Config":
        aliases = {
            "num_layers": "num_hidden_layers",
            "n_layers": "num_hidden_layers",
            "num_heads": "num_attention_heads",
            "n_heads": "num_attention_heads",
            "num_kv_heads": "num_key_value_heads",
            "max_seq_len": "max_position_embeddings",
        }
        normalized = dict(data)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in allowed})


def tiny_llama3_config(**overrides: int | float | bool | dict[str, object] | None) -> Llama3Config:
    config = Llama3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return replace(config, **overrides)


def llama3_70b_config(**overrides: int | float | bool | dict[str, object] | None) -> Llama3Config:
    """Return the Llama 3 70B architecture config without allocating weights."""

    config = Llama3Config(
        vocab_size=128256,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    return replace(config, **overrides)
