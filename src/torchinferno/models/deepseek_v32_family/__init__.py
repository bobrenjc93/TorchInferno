from torchinferno.models.deepseek_v32_family.registry import VARIANTS
from torchinferno.models.deepseek_v32_family.v0 import DeepSeekV32V0ForCausalLM, tiny_deepseek_v32_v0_config
from torchinferno.models.deepseek_v32_family.v1 import DeepSeekV32V1ForCausalLM, tiny_deepseek_v32_v1_config

__all__ = [
    "DeepSeekV32V0ForCausalLM",
    "DeepSeekV32V1ForCausalLM",
    "VARIANTS",
    "tiny_deepseek_v32_v0_config",
    "tiny_deepseek_v32_v1_config",
]
