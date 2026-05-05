from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LogitReference:
    input_ids: list[int]
    logits: list[float]
    atol: float = 1e-4
    rtol: float = 1e-4
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LogitReference":
        return cls(
            input_ids=[int(value) for value in data["input_ids"]],  # type: ignore[index]
            logits=[float(value) for value in data["logits"]],  # type: ignore[index]
            atol=float(data.get("atol", 1e-4)),
            rtol=float(data.get("rtol", 1e-4)),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float


def capture_logit_reference(
    model: torch.nn.Module,
    input_ids: list[int],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    description: str = "",
) -> LogitReference:
    device = next(model.parameters()).device
    tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(tensor, use_cache=False)  # type: ignore[misc]
    return LogitReference(
        input_ids=list(input_ids),
        logits=[float(value) for value in logits[0, -1].detach().cpu().float().tolist()],
        atol=atol,
        rtol=rtol,
        description=description,
    )


def validate_logit_reference(model: torch.nn.Module, reference: LogitReference) -> ValidationResult:
    device = next(model.parameters()).device
    tensor = torch.tensor([reference.input_ids], device=device, dtype=torch.long)
    expected = torch.tensor(reference.logits, device=device, dtype=torch.float32)
    with torch.inference_mode():
        actual_logits, _ = model(tensor, use_cache=False)  # type: ignore[misc]
    actual = actual_logits[0, -1].float()
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-12)
    return ValidationResult(
        passed=bool(torch.allclose(actual, expected, atol=reference.atol, rtol=reference.rtol)),
        max_abs_error=float(diff.max().detach().cpu()),
        max_rel_error=float(rel.max().detach().cpu()),
    )


def save_logit_reference(reference: LogitReference, path: str | Path) -> None:
    Path(path).write_text(json.dumps(reference.to_dict(), indent=2, sort_keys=True) + "\n")


def load_logit_reference(path: str | Path) -> LogitReference:
    return LogitReference.from_dict(json.loads(Path(path).read_text()))
