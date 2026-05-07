from __future__ import annotations

import torch
from torch import Tensor

try:
    import helion
    import helion.language as hl
except Exception as exc:  # pragma: no cover - import guarded by caller
    raise RuntimeError("Helion is required for torchinferno.kernels.helion_ops") from exc


HELION_KERNEL_PROVENANCE = {
    "swiglu": {
        "status": "candidate",
        "origin": "Helion DSL candidate kernel",
        "fx_pattern": "aten.silu.default -> aten.mul.Tensor",
        "promotion_policy": "Only route production through this kernel after helion-search-fx beats the best baseline.",
    }
}


@helion.kernel(config=helion.Config(block_sizes=[1024], num_warps=4, indexing="pointer"))
def _helion_swiglu_kernel(gate: Tensor, up: Tensor) -> Tensor:
    """Candidate Helion SwiGLU kernel used by research trials."""

    out = torch.empty_like(gate)
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    out_flat = out.view(-1)
    for tile in hl.tile(gate_flat.size(0)):
        out_flat[tile] = (
            gate_flat[tile].to(torch.float32)
            / (1.0 + torch.exp(-gate_flat[tile].to(torch.float32)))
        ) * up_flat[tile]
    return out


def helion_swiglu_activation(gate: Tensor, up: Tensor) -> Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    return _helion_swiglu_kernel(gate.contiguous(), up.contiguous())
