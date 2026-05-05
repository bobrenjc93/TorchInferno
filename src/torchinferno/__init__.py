"""TorchInferno: torch-native inference building blocks."""

from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.kernels import (
    NVFP4Tensor,
    dequantize_nvfp4,
    nvfp4_linear_reference,
    paged_decode_attention,
    quantize_nvfp4,
)
from torchinferno.models.auto import load_model_auto
from torchinferno.models.conversion import (
    audit_deepseek_checkpoint,
    audit_native_deepseek_checkpoint,
    convert_deepseek_checkpoint,
    convert_native_deepseek_checkpoint,
)
from torchinferno.models.deepseek import DeepSeekV32Config, DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4Config, DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.tokenization import load_text_tokenizer
from torchinferno.validation import (
    LogitReference,
    ValidationResult,
    capture_logit_reference,
    load_logit_reference,
    save_logit_reference,
    validate_logit_reference,
)

__all__ = [
    "CompileConfig",
    "DSv4Config",
    "DSv4ForCausalLM",
    "DeepSeekV32Config",
    "DeepSeekV32ForCausalLM",
    "LogitReference",
    "NVFP4Tensor",
    "ValidationResult",
    "audit_deepseek_checkpoint",
    "audit_native_deepseek_checkpoint",
    "capture_logit_reference",
    "compile_forward",
    "convert_deepseek_checkpoint",
    "convert_native_deepseek_checkpoint",
    "dequantize_nvfp4",
    "load_logit_reference",
    "load_model_auto",
    "load_text_tokenizer",
    "nvfp4_linear_reference",
    "paged_decode_attention",
    "quantize_nvfp4",
    "save_logit_reference",
    "tiny_deepseek_v32_config",
    "tiny_dsv4_config",
    "validate_logit_reference",
]
