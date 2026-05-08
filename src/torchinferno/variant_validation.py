from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file
from torch import Tensor

from torchinferno.models.deepseek_v32_family import (
    DeepSeekV32V0ForCausalLM,
    DeepSeekV32V1ForCausalLM,
    tiny_deepseek_v32_v0_config,
)
from torchinferno.models.dsv4_family import DSv4V0ForCausalLM, DSv4V1ForCausalLM, tiny_dsv4_v0_config
from torchinferno.models.hf import HF_CONFIG_NAME, SAFETENSORS_INDEX_NAME
from torchinferno.models.llama3_family import (
    Llama3PipelineForCausalLM,
    Llama3V0ForCausalLM,
    Llama3V1ForCausalLM,
    tiny_llama3_v0_config,
)
from torchinferno.models.variants import list_model_variants


@dataclass(frozen=True)
class VariantLogitComparison:
    family: str
    eager_variant: str
    optimized_variant: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    compared_logits: int
    input_shape: tuple[int, int]
    logits_shape: tuple[int, ...]
    device: str
    dtype: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        data["logits_shape"] = list(self.logits_shape)
        return data


@dataclass(frozen=True)
class SkippedVariantComparison:
    family: str
    eager_variant: str
    optimized_variant: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VariantLogitValidationReport:
    comparisons: tuple[VariantLogitComparison, ...]
    skipped: tuple[SkippedVariantComparison, ...]
    atol: float
    rtol: float
    seed: int

    @property
    def passed(self) -> bool:
        return all(comparison.passed for comparison in self.comparisons)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "seed": self.seed,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "skipped": [skipped.to_dict() for skipped in self.skipped],
        }


@dataclass(frozen=True)
class _VariantCase:
    family: str
    eager_variant: str
    optimized_variant: str
    factory: Callable[[torch.device, torch.dtype, int, int], tuple[object, object]]
    requires_tensor_parallel: bool = False


def run_variant_logit_validation(
    *,
    family: str | None = None,
    variant: str | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_size: int = 2,
    tokens: int = 4,
    vocab_size: int = 32,
    seed: int = 0,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    include_tensor_parallel: bool = False,
) -> VariantLogitValidationReport:
    """Compare optimized model variants against their eager v0 logits.

    The default cases are intentionally tiny and CPU-friendly so this can be run
    as part of normal optimization loops. `rtol=1e-2` encodes the "within 1% of
    eager" contract while `atol` keeps near-zero eager logits from dominating the
    pass/fail decision.
    """

    resolved_device = torch.device(device)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")

    cases, unsupported = _selected_cases(family=family, variant=variant)
    comparisons: list[VariantLogitComparison] = []
    skipped: list[SkippedVariantComparison] = list(unsupported)
    for index, case in enumerate(cases):
        if case.requires_tensor_parallel and not include_tensor_parallel:
            skipped.append(
                SkippedVariantComparison(
                    family=case.family,
                    eager_variant=case.eager_variant,
                    optimized_variant=case.optimized_variant,
                    reason="tensor-parallel validation is opt-in because it may initialize distributed/CUDA state",
                )
            )
            continue
        torch.manual_seed(seed + index)
        eager, optimized = case.factory(resolved_device, dtype, tokens, vocab_size)
        input_ids = _make_input_ids(
            batch_size=batch_size,
            tokens=tokens,
            vocab_size=vocab_size,
            device=_model_input_device(eager, resolved_device),
        )
        comparisons.append(
            compare_model_logits(
                eager,
                optimized,
                input_ids,
                family=case.family,
                eager_variant=case.eager_variant,
                optimized_variant=case.optimized_variant,
                atol=atol,
                rtol=rtol,
                dtype=dtype,
            )
        )
    return VariantLogitValidationReport(
        comparisons=tuple(comparisons),
        skipped=tuple(skipped),
        atol=atol,
        rtol=rtol,
        seed=seed,
    )


def compare_model_logits(
    eager: object,
    optimized: object,
    input_ids: Tensor,
    *,
    family: str,
    eager_variant: str,
    optimized_variant: str,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    dtype: torch.dtype = torch.float32,
) -> VariantLogitComparison:
    eager_logits = _forward_logits(eager, input_ids.to(_model_input_device(eager, input_ids.device)))
    optimized_logits = _forward_logits(optimized, input_ids.to(_model_input_device(optimized, input_ids.device)))
    expected = eager_logits.detach().float().cpu()
    actual = optimized_logits.detach().float().cpu()
    if actual.shape != expected.shape:
        raise ValueError(
            f"logit shape mismatch for {family}:{optimized_variant}: "
            f"eager={tuple(expected.shape)} optimized={tuple(actual.shape)}"
        )
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-12)
    return VariantLogitComparison(
        family=family,
        eager_variant=eager_variant,
        optimized_variant=optimized_variant,
        passed=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        max_abs_error=float(diff.max().item()),
        max_rel_error=float(rel.max().item()),
        mean_abs_error=float(diff.mean().item()),
        compared_logits=actual.numel(),
        input_shape=tuple(int(dim) for dim in input_ids.shape),
        logits_shape=tuple(int(dim) for dim in actual.shape),
        device=str(input_ids.device),
        dtype=str(dtype).replace("torch.", ""),
    )


def save_variant_logit_report(report: VariantLogitValidationReport, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def _selected_cases(
    *,
    family: str | None,
    variant: str | None,
) -> tuple[tuple[_VariantCase, ...], tuple[SkippedVariantComparison, ...]]:
    available = {(case.family, case.optimized_variant): case for case in _variant_cases()}
    registered = [
        spec
        for spec in list_model_variants(family)
        if spec.parents and (variant is None or spec.variant == variant)
    ]
    cases: list[_VariantCase] = []
    skipped: list[SkippedVariantComparison] = []
    for spec in registered:
        key = (spec.family, spec.variant)
        case = available.get(key)
        if case is not None:
            cases.append(case)
        else:
            skipped.append(
                SkippedVariantComparison(
                    family=spec.family,
                    eager_variant=spec.parents[0],
                    optimized_variant=spec.variant,
                    reason="no tiny eager validation factory registered for this variant",
                )
            )
    if variant is not None and not cases and not skipped:
        family_text = family or "any family"
        raise ValueError(f"no eager logit validation case for variant={variant!r} family={family_text!r}")
    return tuple(cases), tuple(skipped)


def _variant_cases() -> tuple[_VariantCase, ...]:
    return (
        _VariantCase("dsv4", "v0", "v1", _build_dsv4_v1_case),
        _VariantCase("deepseek-v3.2", "v0", "v1", _build_deepseek_v1_case),
        _VariantCase("llama3", "v0", "v1", _build_llama3_v1_case),
        _VariantCase("llama3", "v0", "pipeline-v0", _build_llama3_pipeline_case),
        _VariantCase("llama3", "v0", "tp-v0", _build_llama3_tensor_parallel_case, requires_tensor_parallel=True),
    )


def _build_dsv4_v1_case(
    device: torch.device,
    dtype: torch.dtype,
    tokens: int,
    vocab_size: int,
) -> tuple[object, object]:
    config = tiny_dsv4_v0_config(vocab_size=vocab_size, max_seq_len=tokens + 8)
    eager = DSv4V0ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized = DSv4V1ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized.load_state_dict(eager.state_dict())
    return eager, optimized


def _build_deepseek_v1_case(
    device: torch.device,
    dtype: torch.dtype,
    tokens: int,
    vocab_size: int,
) -> tuple[object, object]:
    config = tiny_deepseek_v32_v0_config(vocab_size=vocab_size, max_position_embeddings=tokens + 8)
    eager = DeepSeekV32V0ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized = DeepSeekV32V1ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized.load_state_dict(eager.state_dict())
    return eager, optimized


def _build_llama3_v1_case(
    device: torch.device,
    dtype: torch.dtype,
    tokens: int,
    vocab_size: int,
) -> tuple[object, object]:
    config = tiny_llama3_v0_config(vocab_size=vocab_size, max_position_embeddings=tokens + 8)
    eager = Llama3V0ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized = Llama3V1ForCausalLM(config).to(device=device, dtype=dtype).eval()
    optimized.load_state_dict(eager.state_dict())
    return eager, optimized


def _build_llama3_pipeline_case(
    device: torch.device,
    dtype: torch.dtype,
    tokens: int,
    vocab_size: int,
) -> tuple[object, object]:
    config = tiny_llama3_v0_config(vocab_size=vocab_size, max_position_embeddings=tokens + 8)
    eager = Llama3V0ForCausalLM(config).to(device=device, dtype=dtype).eval()
    with tempfile.TemporaryDirectory(prefix="torchinferno-llama3-pipeline-") as tmp:
        checkpoint = Path(tmp)
        _write_llama3_hf_checkpoint(eager, checkpoint)
        optimized = Llama3PipelineForCausalLM.from_pretrained(
            checkpoint,
            devices=(str(device),),
            dtype=str(dtype).replace("torch.", ""),
        ).eval()
        return eager, optimized


def _build_llama3_tensor_parallel_case(
    device: torch.device,
    dtype: torch.dtype,
    tokens: int,
    vocab_size: int,
) -> tuple[object, object]:
    from torchinferno.models.llama3_family import Llama3TensorParallelForCausalLM

    config = tiny_llama3_v0_config(vocab_size=vocab_size, max_position_embeddings=tokens + 8)
    eager = Llama3V0ForCausalLM(config).to(device=device, dtype=dtype).eval()
    with tempfile.TemporaryDirectory(prefix="torchinferno-llama3-tp-") as tmp:
        checkpoint = Path(tmp)
        _write_llama3_hf_checkpoint(eager, checkpoint)
        optimized = Llama3TensorParallelForCausalLM.from_pretrained(
            checkpoint,
            dtype=str(dtype).replace("torch.", ""),
        ).eval()
        return eager, optimized


def _write_llama3_hf_checkpoint(model: Llama3V0ForCausalLM, path: Path) -> None:
    state = model.state_dict()
    hf_state: dict[str, Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].detach().cpu(),
        "model.norm.weight": state["norm.weight"].detach().cpu(),
        "lm_head.weight": state["lm_head.weight"].detach().cpu(),
    }
    for layer_id in range(model.config.num_hidden_layers):
        prefix = f"layers.{layer_id}."
        hf_prefix = f"model.layers.{layer_id}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            hf_state[hf_prefix + suffix] = state[prefix + suffix].detach().cpu()
    weights_name = "model-00001-of-00001.safetensors"
    save_file(hf_state, path / weights_name)
    (path / SAFETENSORS_INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in hf_state.values())},
                "weight_map": {name: weights_name for name in hf_state},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (path / HF_CONFIG_NAME).write_text(json.dumps(model.config.to_dict(), sort_keys=True) + "\n")


def _make_input_ids(*, batch_size: int, tokens: int, vocab_size: int, device: torch.device) -> Tensor:
    return torch.randint(0, vocab_size, (batch_size, tokens), device=device, dtype=torch.long)


def _forward_logits(model: object, input_ids: Tensor) -> Tensor:
    with torch.inference_mode():
        forward = getattr(model, "forward", None)
        if callable(forward):
            try:
                output = forward(input_ids, use_cache=False)
            except TypeError:
                output = forward(input_ids)
        else:
            try:
                output = model(input_ids, use_cache=False)  # type: ignore[misc,operator]
            except TypeError:
                output = model(input_ids)  # type: ignore[misc,operator]
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError(f"model forward returned unsupported output type: {type(output)!r}")
    return output


def _model_input_device(model: object, fallback: torch.device) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    embed_device = getattr(model, "embed_device", None)
    if embed_device is not None:
        return torch.device(embed_device)
    return fallback
