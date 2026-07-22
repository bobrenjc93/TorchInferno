"""Fused MXFP4 MoE kernels for the DeepSeek V4 CUDA path.

The runtime primitive is loaded from an offline-built SGLang Marlin artifact
matched to the active PyTorch runtime. Weight repacking and scale permutation
below are small, Torch-native adaptations of Apache-2.0 Marlin utilities.
Routing and expert-parallel collectives remain owned by TorchInferno.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


_MARLIN_TILE = 16
_loaded = False
_runtime_module: object | None = None
_MARLIN_PARAMETERS = (
    "a",
    "c_or_none",
    "b_q_weight",
    "b_bias_or_none",
    "b_scales",
    "global_scale_or_none",
    "b_zeros_or_none",
    "g_idx_or_none",
    "perm_or_none",
    "workspace",
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
    "topk_weights",
    "moe_block_size",
    "top_k",
    "mul_topk_weights",
    "is_ep",
    "b_q_type",
    "size_m",
    "size_n",
    "size_k",
    "is_k_full",
    "use_atomic_add",
    "use_fp32_reduce",
    "is_zp_float",
)


def _offline_preparation_enabled() -> bool:
    return os.environ.get("TORCHINFERNO_V4_KERNEL_PREPARE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prebuilt_marlin_library() -> Path:
    from sglang.jit_kernel import utils

    args = utils.make_cpp_args(torch.bfloat16)
    source = (utils.KERNEL_PATH / "csrc" / "gemm/marlin_moe/moe_wna16_marlin.cuh").resolve()
    module_name = "sgl_kernel_jit_moe_wna16_marlin_" + "_".join(str(arg) for arg in args)
    module_name += "_" + utils._local_jit_source_hash([str(source)])
    cache_root = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
    build_dir = cache_root / utils._jit_build_dir_name(module_name)
    return build_dir / f"{module_name}.so"


def load_mxfp4_moe_ops() -> bool:
    """Load the optional stable-ABI Marlin MoE provider."""

    global _loaded, _runtime_module
    if _loaded:
        return True
    try:
        from sgl_kernel.scalar_type import scalar_types
        from sglang.jit_kernel.moe_wna16_marlin import (  # noqa: F401
            moe_wna16_marlin_gemm,
        )

        parameters = tuple(inspect.signature(moe_wna16_marlin_gemm).parameters)
        if not hasattr(scalar_types, "float4_e2m1f"):
            raise RuntimeError("the installed SGLang provider lacks MXFP4 scalar types")
        if parameters != _MARLIN_PARAMETERS:
            raise RuntimeError(
                "the installed SGLang Marlin ABI does not match TorchInferno's pinned provider"
            )
        if not _offline_preparation_enabled():
            library = _prebuilt_marlin_library()
            if not library.is_file():
                raise RuntimeError(
                    "the DeepSeek V4 Marlin provider is not prepared; run the offline kernel prepare step"
                )
            from tvm_ffi import load_module
            import sglang.jit_kernel.moe_wna16_marlin as marlin_provider

            _runtime_module = load_module(str(library))
            marlin_provider._jit_moe_wna16_marlin_module = (  # type: ignore[attr-defined]
                lambda dtype: _runtime_module
            )

        _loaded = True
    except (ImportError, OSError, RuntimeError):
        _loaded = False
    return _loaded


def _weight_permutation(device: torch.device) -> Tensor:
    values: list[int] = []
    for index in range(32):
        column = index // 4
        tile: list[int] = []
        for block in (0, 1):
            for row in (
                2 * (index % 4),
                2 * (index % 4) + 1,
                2 * (index % 4 + 4),
                2 * (index % 4 + 4) + 1,
            ):
                tile.append(16 * row + column + 8 * block)
        for group in range(4):
            values.extend(value + 256 * group for value in tile)
    permutation = torch.tensor(values, dtype=torch.long, device=device)
    interleave = torch.tensor((0, 2, 4, 6, 1, 3, 5, 7), device=device)
    return permutation.view(-1, 8)[:, interleave].reshape(-1)


def _repack_one_mxfp4(weight: Tensor, *, size_k: int, size_n: int) -> Tensor:
    """Convert checkpoint nibble order to the Marlin MMA tile layout."""

    if tuple(weight.shape) != (size_n, size_k // 2):
        raise ValueError(
            f"expected packed MXFP4 [{size_n}, {size_k // 2}], got {tuple(weight.shape)}"
        )
    packed = weight.view(torch.uint8)
    logical = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(size_n, size_k)
    logical = logical.transpose(0, 1).contiguous()
    if size_k % _MARLIN_TILE or size_n % _MARLIN_TILE:
        raise ValueError("Marlin MXFP4 dimensions must be divisible by 16")
    tiled = logical.reshape(
        size_k // _MARLIN_TILE,
        _MARLIN_TILE,
        size_n // _MARLIN_TILE,
        _MARLIN_TILE,
    )
    tiled = tiled.permute(0, 2, 1, 3).reshape(size_k // _MARLIN_TILE, size_n * _MARLIN_TILE)
    permutation = _weight_permutation(weight.device)
    tiled = tiled.reshape(-1, permutation.numel())[:, permutation].reshape(tiled.shape)
    shifts = torch.arange(8, dtype=torch.int32, device=weight.device) * 4
    return torch.sum(tiled.reshape(*tiled.shape[:-1], -1, 8).int() << shifts, dim=-1).int()


def _scale_permutation(device: torch.device) -> Tensor:
    return torch.tensor(
        [index + 8 * group for index in range(8) for group in range(8)],
        dtype=torch.long,
        device=device,
    )


def _prepare_one_scale(scale: Tensor, *, size_k: int, size_n: int) -> Tensor:
    if tuple(scale.shape) != (size_n, size_k // 32):
        raise ValueError(
            f"expected MXFP4 scale [{size_n}, {size_k // 32}], got {tuple(scale.shape)}"
        )
    value = scale.transpose(0, 1).contiguous()
    permutation = _scale_permutation(scale.device)
    value = value.reshape(-1, permutation.numel())[:, permutation]
    value = value.reshape(-1, size_n).contiguous()
    # Marlin's BF16 dequantizer consumes pairs in this lane order.
    value = value.view(-1, 4)[:, (0, 2, 1, 3)].reshape(value.shape)
    return value.to(torch.float8_e8m0fnu)


@dataclass
class PreparedMxfp4Experts:
    w13: Tensor
    w2: Tensor
    w13_scale: Tensor
    w2_scale: Tensor
    workspace: Tensor
    expert_map: Tensor
    global_num_experts: int
    hidden_size: int
    intermediate_size: int


def prepare_mxfp4_experts(
    w1: Tensor,
    w3: Tensor,
    w2: Tensor,
    w1_scale: Tensor,
    w3_scale: Tensor,
    w2_scale: Tensor,
    *,
    expert_start: int,
    global_num_experts: int,
) -> PreparedMxfp4Experts:
    """Repack one rank's native MXFP4 experts for grouped Marlin GEMMs."""

    if not load_mxfp4_moe_ops():
        raise RuntimeError("the optional Marlin MoE CUDA provider is unavailable")
    if w1.ndim != 3 or w2.ndim != 3 or w3.ndim != 3:
        raise ValueError("expert weights must be stacked [experts, output, packed-K]")
    local_experts = w1.size(0)
    hidden_size = w1.size(2) * 2
    intermediate_size = w1.size(1)
    if tuple(w3.shape) != tuple(w1.shape):
        raise ValueError("w1 and w3 expert shapes must match")
    if tuple(w2.shape) != (local_experts, hidden_size, intermediate_size // 2):
        raise ValueError("w2 expert shape does not match w1/w3")

    native_w13 = torch.cat((w1, w3), dim=1).contiguous()
    native_s13 = torch.cat((w1_scale, w3_scale), dim=1).contiguous()
    repacked_w13 = torch.stack(
        [
            _repack_one_mxfp4(weight, size_k=hidden_size, size_n=2 * intermediate_size)
            for weight in native_w13
        ]
    )
    repacked_w2 = torch.stack(
        [
            _repack_one_mxfp4(weight, size_k=intermediate_size, size_n=hidden_size)
            for weight in w2
        ]
    )
    repacked_s13 = torch.stack(
        [
            _prepare_one_scale(scale, size_k=hidden_size, size_n=2 * intermediate_size)
            for scale in native_s13
        ]
    )
    repacked_s2 = torch.stack(
        [
            _prepare_one_scale(scale, size_k=intermediate_size, size_n=hidden_size)
            for scale in w2_scale
        ]
    )

    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.int32, device=w1.device
    )
    expert_map[expert_start : expert_start + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=w1.device
    )
    multiprocessors = torch.cuda.get_device_properties(w1.device).multi_processor_count
    workspace = torch.zeros(multiprocessors * 4, dtype=torch.int32, device=w1.device)
    return PreparedMxfp4Experts(
        w13=repacked_w13,
        w2=repacked_w2,
        w13_scale=repacked_s13,
        w2_scale=repacked_s2,
        workspace=workspace,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def _align_tokens(
    topk_ids: Tensor,
    *,
    block_size: int,
    global_num_experts: int,
    expert_map: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        from sgl_kernel import moe_align_block_size
    except ImportError as exc:
        raise RuntimeError("fused MXFP4 MoE requires the sglang-kernel package") from exc

    local_ids = expert_map[topk_ids.long()]
    # The alignment kernel reserves -1 for routes owned by another EP rank.
    # Mapping those routes to a synthetic local expert makes Marlin index one
    # row past the packed weights and corrupts batched outputs.
    local_ids = local_ids.int().contiguous()
    aligned_num_experts = global_num_experts + 1
    maximum = local_ids.numel() + aligned_num_experts * (block_size - 1)
    if local_ids.numel() < aligned_num_experts:
        maximum = min(topk_ids.numel() * block_size, maximum)
    sorted_ids = torch.empty(maximum, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(
        (maximum + block_size - 1) // block_size,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.empty(
        aligned_num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    moe_align_block_size(
        local_ids,
        aligned_num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens,
        cumsum,
        True,
    )
    return sorted_ids, expert_ids, num_tokens


def fused_mxfp4_moe(
    hidden_states: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    experts: PreparedMxfp4Experts,
    *,
    clamp_limit: float = 0.0,
) -> Tensor:
    """Run grouped routed experts for one expert-parallel rank."""

    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be a contiguous matrix")
    tokens, hidden_size = hidden_states.shape
    if hidden_size != experts.hidden_size:
        raise ValueError("hidden size does not match prepared experts")
    topk = topk_ids.size(1)
    local_experts = experts.w13.size(0)
    for block_size in (8, 16, 32, 48, 64):
        if tokens * topk / local_experts / block_size < 0.9:
            break
    sorted_ids, expert_ids, num_tokens = _align_tokens(
        topk_ids,
        block_size=block_size,
        global_num_experts=experts.global_num_experts,
        expert_map=experts.expert_map,
    )
    intermediate = torch.empty(
        (tokens * topk, 2 * experts.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    from sgl_kernel.scalar_type import scalar_types
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

    moe_wna16_marlin_gemm(
        hidden_states,
        intermediate,
        experts.w13,
        None,
        experts.w13_scale,
        None,
        None,
        None,
        None,
        experts.workspace,
        sorted_ids,
        expert_ids,
        num_tokens,
        topk_weights,
        block_size,
        topk,
        False,
        True,
        scalar_types.float4_e2m1f,
        tokens,
        2 * experts.intermediate_size,
        hidden_size,
        True,
        False,
        True,
        False,
    )
    gate, up = intermediate.chunk(2, dim=-1)
    if clamp_limit > 0:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    activated = F.silu(gate) * up
    output = torch.zeros(
        (tokens * topk, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    moe_wna16_marlin_gemm(
        activated,
        output,
        experts.w2,
        None,
        experts.w2_scale,
        None,
        None,
        None,
        None,
        experts.workspace,
        sorted_ids,
        expert_ids,
        num_tokens,
        topk_weights,
        block_size,
        1,
        True,
        True,
        scalar_types.float4_e2m1f,
        tokens * topk,
        hidden_size,
        experts.intermediate_size,
        True,
        False,
        True,
        False,
    )
    return output.view(tokens, topk, hidden_size).sum(dim=1)
