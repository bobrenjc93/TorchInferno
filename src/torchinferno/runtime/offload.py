from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import torch
from torch import Tensor, nn

from torchinferno.models.deepseek import DeepSeekV32ForCausalLM
from torchinferno.models.dsv4 import DSv4ForCausalLM
from torchinferno.runtime.sampling import sample_next_token


@dataclass(frozen=True)
class OffloadEvent:
    name: str
    kind: str
    elapsed_ms: float
    bytes_moved: int
    module_bytes: int
    device: str


@dataclass(frozen=True)
class OffloadRunResult:
    output: Tensor
    events: tuple[OffloadEvent, ...]

    @property
    def total_elapsed_ms(self) -> float:
        return sum(event.elapsed_ms for event in self.events)

    @property
    def compute_ms(self) -> float:
        return sum(event.elapsed_ms for event in self.events if event.kind == "compute")

    @property
    def movement_ms(self) -> float:
        return sum(event.elapsed_ms for event in self.events if event.kind != "compute")


@torch.inference_mode()
def run_offloaded_forward(
    model: nn.Module,
    input_ids: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    activation_offload: bool = False,
    label: str = "forward",
) -> OffloadRunResult:
    """Run one no-cache forward pass by staging modules from CPU to device.

    This is a correctness and profiling path for single-GPU offload studies. It
    keeps model weights resident on CPU and explicitly stages each executable
    block to the target device, recording movement and compute timing separately.
    """

    model.to("cpu").eval()
    events: list[OffloadEvent] = []
    if isinstance(model, DSv4ForCausalLM):
        output = _run_dsv4_offloaded_forward(
            model,
            input_ids.detach().cpu(),
            device=device,
            dtype=dtype,
            activation_offload=activation_offload,
            label=label,
            events=events,
        )
    elif isinstance(model, DeepSeekV32ForCausalLM):
        output = _run_deepseek_offloaded_forward(
            model,
            input_ids.detach().cpu(),
            device=device,
            dtype=dtype,
            activation_offload=activation_offload,
            label=label,
            events=events,
        )
    else:
        raise TypeError(f"unsupported offload model type: {type(model).__name__}")
    output = _move_tensor(output, torch.device("cpu"), f"{label}.output", "output_to_cpu", events)
    return OffloadRunResult(output, tuple(events))


@torch.inference_mode()
def run_offloaded_generate_recompute(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 0.0,
    activation_offload: bool = False,
) -> OffloadRunResult:
    """Generate tokens with CPU-offloaded full-prefix recompute steps.

    This deliberately favors full-model correctness over speed. Decode cache
    offload is a separate production milestone; this runner records the transfer
    overhead that should be subtracted when estimating a resident/sharded run.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    tokens = input_ids.detach().cpu()
    events: list[OffloadEvent] = []
    if max_new_tokens == 0:
        result = run_offloaded_forward(
            model,
            tokens,
            device=device,
            dtype=dtype,
            activation_offload=activation_offload,
            label="step0",
        )
        return result
    for step in range(max_new_tokens):
        result = run_offloaded_forward(
            model,
            tokens,
            device=device,
            dtype=dtype,
            activation_offload=activation_offload,
            label=f"step{step}",
        )
        events.extend(result.events)
        next_token = sample_next_token(result.output[:, -1, :], temperature)
        tokens = torch.cat([tokens, next_token[:, None].cpu()], dim=1)
    return OffloadRunResult(tokens, tuple(events))


def summarize_offload_events(events: tuple[OffloadEvent, ...]) -> dict[str, float | int]:
    total_ms = sum(event.elapsed_ms for event in events)
    compute_ms = sum(event.elapsed_ms for event in events if event.kind == "compute")
    movement_ms = total_ms - compute_ms
    stage_ms = sum(event.elapsed_ms for event in events if event.kind == "stage_to_device")
    evict_ms = sum(event.elapsed_ms for event in events if event.kind == "evict_to_cpu")
    activation_ms = sum(event.elapsed_ms for event in events if event.kind.startswith("activation_"))
    output_ms = sum(event.elapsed_ms for event in events if event.kind == "output_to_cpu")
    bytes_moved = sum(event.bytes_moved for event in events)
    weight_bytes_moved = sum(
        event.bytes_moved for event in events if event.kind in {"stage_to_device", "evict_to_cpu"}
    )
    peak_module_bytes = max((event.module_bytes for event in events), default=0)
    return {
        "event_count": len(events),
        "observed_elapsed_ms": total_ms,
        "compute_ms": compute_ms,
        "movement_ms": movement_ms,
        "stage_to_device_ms": stage_ms,
        "evict_to_cpu_ms": evict_ms,
        "activation_transfer_ms": activation_ms,
        "output_transfer_ms": output_ms,
        "compute_only_estimate_ms": compute_ms,
        "bytes_moved": bytes_moved,
        "weight_bytes_moved": weight_bytes_moved,
        "peak_module_bytes": peak_module_bytes,
    }


def _run_dsv4_offloaded_forward(
    model: DSv4ForCausalLM,
    input_ids: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    activation_offload: bool,
    label: str,
    events: list[OffloadEvent],
) -> Tensor:
    tokens = input_ids.size(1)
    positions = torch.arange(tokens, device=device)
    _stage_module(model.embed_tokens, device, dtype, f"{label}.embed_tokens", events)
    ids = _move_tensor(input_ids, device, f"{label}.input_ids", "input_to_device", events)
    hidden = _time_compute(lambda: model.embed_tokens(ids), f"{label}.embed_tokens", device, events)
    _evict_module(model.embed_tokens, f"{label}.embed_tokens", events)
    hidden = _maybe_offload_activation(hidden, activation_offload, f"{label}.embed_tokens", events)

    for layer_idx, layer in enumerate(model.layers):
        hidden = _ensure_activation_on_device(hidden, device, f"{label}.layers.{layer_idx}", events)
        _stage_module(layer, device, dtype, f"{label}.layers.{layer_idx}", events)
        hidden = _time_compute(lambda layer=layer, hidden=hidden: layer(hidden, positions, None), f"{label}.layers.{layer_idx}", device, events)
        _evict_module(layer, f"{label}.layers.{layer_idx}", events)
        hidden = _maybe_offload_activation(hidden, activation_offload, f"{label}.layers.{layer_idx}", events)

    hidden = _ensure_activation_on_device(hidden, device, f"{label}.norm", events)
    _stage_module(model.norm, device, dtype, f"{label}.norm", events)
    hidden = _time_compute(lambda: model.norm(hidden), f"{label}.norm", device, events)
    _evict_module(model.norm, f"{label}.norm", events)

    _stage_module(model.lm_head, device, dtype, f"{label}.lm_head", events)
    logits = _time_compute(lambda: model.lm_head(hidden), f"{label}.lm_head", device, events)
    _evict_module(model.lm_head, f"{label}.lm_head", events)
    return logits


def _run_deepseek_offloaded_forward(
    model: DeepSeekV32ForCausalLM,
    input_ids: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    activation_offload: bool,
    label: str,
    events: list[OffloadEvent],
) -> Tensor:
    tokens = input_ids.size(1)
    positions = torch.arange(tokens, device=device)
    body = model.model
    _stage_module(body.embed_tokens, device, dtype, f"{label}.model.embed_tokens", events)
    ids = _move_tensor(input_ids, device, f"{label}.input_ids", "input_to_device", events)
    hidden = _time_compute(lambda: body.embed_tokens(ids), f"{label}.model.embed_tokens", device, events)
    _evict_module(body.embed_tokens, f"{label}.model.embed_tokens", events)
    hidden = _maybe_offload_activation(hidden, activation_offload, f"{label}.model.embed_tokens", events)

    for layer_idx, layer in enumerate(body.layers):
        hidden = _ensure_activation_on_device(hidden, device, f"{label}.model.layers.{layer_idx}", events)
        _stage_module(layer, device, dtype, f"{label}.model.layers.{layer_idx}", events)
        hidden = _time_compute(
            lambda layer=layer, hidden=hidden: layer(hidden, positions, None, None),
            f"{label}.model.layers.{layer_idx}",
            device,
            events,
        )
        _evict_module(layer, f"{label}.model.layers.{layer_idx}", events)
        hidden = _maybe_offload_activation(hidden, activation_offload, f"{label}.model.layers.{layer_idx}", events)

    hidden = _ensure_activation_on_device(hidden, device, f"{label}.model.norm", events)
    _stage_module(body.norm, device, dtype, f"{label}.model.norm", events)
    hidden = _time_compute(lambda: body.norm(hidden), f"{label}.model.norm", device, events)
    _evict_module(body.norm, f"{label}.model.norm", events)

    _stage_module(model.lm_head, device, dtype, f"{label}.lm_head", events)
    logits = _time_compute(lambda: model.lm_head(hidden), f"{label}.lm_head", device, events)
    _evict_module(model.lm_head, f"{label}.lm_head", events)
    return logits


def _stage_module(module: nn.Module, device: torch.device, dtype: torch.dtype, name: str, events: list[OffloadEvent]) -> None:
    module_bytes = _module_nbytes(module)

    def move() -> None:
        module.to(device=device, dtype=dtype)

    elapsed_ms = _time_call(move, device)
    events.append(
        OffloadEvent(
            name,
            "stage_to_device",
            elapsed_ms,
            0 if device.type == "cpu" else module_bytes,
            module_bytes,
            str(device),
        )
    )


def _evict_module(module: nn.Module, name: str, events: list[OffloadEvent]) -> None:
    module_bytes = _module_nbytes(module)
    source_device = _module_device(module)

    def move() -> None:
        module.to("cpu")

    elapsed_ms = _time_call(move, source_device)
    events.append(
        OffloadEvent(
            name,
            "evict_to_cpu",
            elapsed_ms,
            0 if source_device.type == "cpu" else module_bytes,
            module_bytes,
            "cpu",
        )
    )


def _time_compute(fn, name: str, device: torch.device, events: list[OffloadEvent]) -> Tensor:
    result: Optional[Tensor] = None

    def run() -> None:
        nonlocal result
        result = fn()

    elapsed_ms = _time_call(run, device)
    events.append(OffloadEvent(name, "compute", elapsed_ms, 0, 0, str(device)))
    assert result is not None
    return result


def _move_tensor(tensor: Tensor, device: torch.device, name: str, kind: str, events: list[OffloadEvent]) -> Tensor:
    source = tensor.device
    result: Optional[Tensor] = None

    def move() -> None:
        nonlocal result
        result = tensor.to(device)

    elapsed_ms = _time_call(move, device if device.type != "cpu" else source)
    moved = 0 if source == device else tensor.numel() * tensor.element_size()
    events.append(OffloadEvent(name, kind, elapsed_ms, moved, 0, str(device)))
    assert result is not None
    return result


def _maybe_offload_activation(
    hidden: Tensor,
    enabled: bool,
    name: str,
    events: list[OffloadEvent],
) -> Tensor:
    if not enabled:
        return hidden
    return _move_tensor(hidden, torch.device("cpu"), name, "activation_to_cpu", events)


def _ensure_activation_on_device(
    hidden: Tensor,
    device: torch.device,
    name: str,
    events: list[OffloadEvent],
) -> Tensor:
    if hidden.device == device:
        return hidden
    return _move_tensor(hidden, device, name, "activation_to_device", events)


def _time_call(fn, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _module_nbytes(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.numel() * tensor.element_size()
    return int(total)


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        return tensor.device
    return torch.device("cpu")
