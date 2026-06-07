"""Self-contained Marlin int4 (W4A16) GEMM for batched decode.

The runtime op is vLLM's prebuilt `torch.ops._C.marlin_gemm`, registered via
`torch.ops.load_library` on vLLM's `_C.abi3.so` (built against torch's STABLE
libtorch ABI, so it loads against our custom torch WITHOUT importing the vllm
python package -- no soxr/transformers chain). The weight-prep helpers below are
copied from vLLM's marlin_utils (Apache-2.0) so this module has NO vllm python
dependency; correctness is validated against a bf16 reference in tests.

Measured (CUDA-graph floor, M=48, see scripts/bench_marlin_int4.py): gate_up
1.52x, down 1.44x vs fp16. Use for the big K-large decode GEMMs (gate_up, down);
small ones (qkv, o_proj) and lm_head (N not %64) stay bf16. NOT yet wired into
serving -- accuracy (RTN int4) must be validated end-to-end against the 98% bar
before enabling.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MAX_PARALLEL = 16
_NUM_BITS = 4
_PACK_FACTOR = 32 // _NUM_BITS  # 8
_UINT4B8_ID = 1125899907892224  # vllm scalar_types.uint4b8.id (signed int4, bias 8)

_DEFAULT_VLLM_SO = "/data/users/bobren/d/vllm/vllm/_C.abi3.so"
_loaded = False


def load_marlin_ops() -> bool:
    """Register torch.ops._C.marlin_gemm from vLLM's prebuilt _C (idempotent).

    Returns True if marlin_gemm is available. Path overridable via
    TORCHINFERNO_VLLM_C_SO.
    """
    global _loaded
    if hasattr(torch.ops._C, "marlin_gemm"):
        _loaded = True
        return True
    so = os.environ.get("TORCHINFERNO_VLLM_C_SO", _DEFAULT_VLLM_SO)
    try:
        torch.ops.load_library(so)
        _loaded = hasattr(torch.ops._C, "marlin_gemm")
    except Exception:
        _loaded = False
    return _loaded


# ---- weight prep (copied from vllm marlin_utils / quant_utils, Apache-2.0) ----

def _get_weight_perm() -> Tensor:
    perm_list: list[int] = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


def _get_scale_perms():
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scale_perm_single = [2 * i + j for i in range(4) for j in [0, 1, 8, 9, 16, 17, 24, 25]]
    return scale_perm, scale_perm_single


def _marlin_permute_weights(q_w: Tensor, size_k: int, size_n: int, perm: Tensor) -> Tensor:
    tile = GPTQ_MARLIN_TILE
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0 and size_n % tile == 0
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3)).reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


def _marlin_weights(q_w: Tensor, size_k: int, size_n: int, perm: Tensor) -> Tensor:
    q_w = _marlin_permute_weights(q_w, size_k, size_n, perm)
    dev = q_w.device
    q_w = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // _PACK_FACTOR), dtype=np.uint32)
    for i in range(_PACK_FACTOR):
        q_packed |= q_w[:, i::_PACK_FACTOR] << (_NUM_BITS * i)
    return torch.from_numpy(q_packed.astype(np.int32)).to(dev)


def _marlin_permute_scales(s: Tensor, size_k: int, size_n: int, group_size: int) -> Tensor:
    scale_perm, scale_perm_single = _get_scale_perms()
    if 0 < group_size < size_k:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def quantize_to_marlin_int4(weight_kn: Tensor, group_size: int = 128):
    """bf16/fp16 weight [K, N] -> (marlin-packed q_weight, marlin-permuted scales).

    Symmetric int4 (uint4b8: stored 0..15, dequant = (q-8)*scale), groupwise.
    """
    w = weight_kn.to(torch.float16)
    size_k, size_n = w.shape
    gs = size_k if group_size == -1 else group_size
    # per-group symmetric scale (max_q=7, min_q=-8)
    wg = w.reshape((-1, gs, size_n)).permute(1, 0, 2).reshape((gs, -1)).float()
    max_val = wg.max(0, keepdim=True).values
    min_val = wg.min(0, keepdim=True).values
    w_s = torch.max((max_val / 7.0).abs(), (min_val / -8.0).abs()).clamp(min=1e-8)
    w_q = torch.round(wg / w_s).clamp(-8, 7).int() + 8  # 0..15
    # restore [K, N]
    w_q = w_q.reshape((gs, -1, size_n)).permute(1, 0, 2).reshape((size_k, size_n)).contiguous()
    w_s = w_s.reshape((-1, size_n)).contiguous().to(torch.float16)
    q_marlin = _marlin_weights(w_q, size_k, size_n, _get_weight_perm().to(w.device))
    s_marlin = _marlin_permute_scales(w_s, size_k, size_n, gs)
    return q_marlin, s_marlin


def make_workspace(size_n: int, device) -> Tensor:
    n_tiles = size_n // GPTQ_MARLIN_MIN_THREAD_N
    return torch.zeros(n_tiles * GPTQ_MARLIN_MAX_PARALLEL * 2, dtype=torch.int, device=device)


_EMPTY = None


def marlin_int4_mm(a: Tensor, q_weight: Tensor, scales: Tensor, workspace: Tensor,
                   size_n: int, size_k: int) -> Tensor:
    """a [M, K] (fp16) @ int4 weight -> [M, N] (fp16). a must be 2D contiguous."""
    global _EMPTY
    if _EMPTY is None or _EMPTY.device != a.device:
        _EMPTY = torch.empty(0, dtype=torch.int, device=a.device)
    m = a.shape[0]
    out = torch.empty((m, size_n), dtype=a.dtype, device=a.device)
    return torch.ops._C.marlin_gemm(
        a, out, q_weight, None, scales, None, None, None, _EMPTY, _EMPTY, workspace,
        _UINT4B8_ID, m, size_n, size_k, True, False, False, False,
    )


def marlin_supports_shape(size_n: int, size_k: int) -> bool:
    return size_n % GPTQ_MARLIN_MIN_THREAD_N == 0 and size_k % GPTQ_MARLIN_TILE == 0
