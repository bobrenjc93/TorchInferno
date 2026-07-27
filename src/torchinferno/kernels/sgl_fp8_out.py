"""Prepared SM90 FP8 scaled-mm provider with caller-owned output.

The shared library is compiled by ``scripts/prepare_sgl_fp8_out.py``. Runtime
code only loads the concrete artifact and falls back when it is unavailable or
incompatible; it never invokes a compiler from a model or serving hot path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch import Tensor

EXPECTED_SGLANG_KERNEL_VERSION = "0.4.4"
_ARTIFACT_SCHEMA = 2
_loaded_op: Callable[..., Tensor] | bool | None = None


def source_path() -> Path:
    return Path(__file__).with_name("csrc") / "sgl_fp8_out.cpp"


def provider_library_path() -> Path | None:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None or spec.origin is None:
        return None
    library = Path(spec.origin).resolve().parent / "sm90" / "common_ops.abi3.so"
    return library if library.is_file() else None


def compatible_provider_version(installed: str) -> bool:
    """Accept the pinned release and official CUDA-local wheel variants."""

    return installed.split("+", 1)[0] == EXPECTED_SGLANG_KERNEL_VERSION


def compatibility() -> dict[str, object]:
    try:
        provider_version = version("sglang-kernel")
    except PackageNotFoundError:
        provider_version = "missing"
    source = source_path()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provider = provider_library_path()
    return {
        "schema": _ARTIFACT_SCHEMA,
        "source_sha256": source_digest,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "sglang_kernel": provider_version,
        # The adapter has an RPATH to this exact shared object. Include both its
        # location and contents so another virtualenv can never reuse an
        # adapter that would load a second SGL operator library into one process.
        "sglang_provider_library": (
            str(provider) if provider is not None else "missing"
        ),
        "sglang_provider_sha256": (
            hashlib.sha256(provider.read_bytes()).hexdigest()
            if provider is not None
            else "missing"
        ),
    }


def artifact_key() -> str:
    payload = json.dumps(compatibility(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def default_artifact_root() -> Path:
    configured = os.environ.get("TORCHINFERNO_KERNEL_ARTIFACT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.cache/torchinferno/kernels").expanduser().resolve()


def prepared_library_path(root: str | Path | None = None) -> Path:
    artifact_root = (
        default_artifact_root()
        if root is None
        else Path(root).expanduser().resolve()
    )
    key = artifact_key()
    return artifact_root / "sgl-fp8-out" / key / f"torchinferno_sgl_fp8_out_{key}.so"


def _compatible_sm90_device(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) == (9, 0)


def load_prepared_op() -> Callable[..., Tensor] | None:
    """Load the pinned provider once, returning ``None`` on a clean fallback."""

    global _loaded_op
    if _loaded_op is False:
        return None
    if callable(_loaded_op):
        return _loaded_op
    try:
        if not compatible_provider_version(version("sglang-kernel")):
            raise RuntimeError("incompatible sglang-kernel provider version")
        library = prepared_library_path()
        if not library.is_file():
            raise RuntimeError("the SGL FP8 output provider is not prepared")
        torch.ops.load_library(str(library))
        op = torch.ops.torchinferno_sgl_fp8.scaled_mm_out
        if not callable(op):
            raise TypeError("the prepared SGL FP8 output provider has no operator")
        _loaded_op = op
    except (
        AttributeError,
        ImportError,
        OSError,
        PackageNotFoundError,
        RuntimeError,
        TypeError,
    ):
        _loaded_op = False
        return None
    return _loaded_op


def scaled_mm_out(
    out: Tensor,
    a: Tensor,
    b: Tensor,
    a_scales: Tensor,
    b_scales: Tensor,
) -> Tensor | None:
    """Run the prepared provider for its validated small-row SM90 region."""

    rows = a.size(0) if a.ndim == 2 else 0
    inner = a.size(1) if a.ndim == 2 else 0
    cols = b.size(1) if b.ndim == 2 else 0
    if (
        a.ndim != 2
        or b.ndim != 2
        or out.ndim != 2
        or rows < 1
        or rows > 64
        or inner % 16 != 0
        or cols % 16 != 0
        or b.size(0) != inner
        or out.shape != (rows, cols)
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != torch.float8_e4m3fn
        or out.dtype not in (torch.bfloat16, torch.float16)
        or a.device != b.device
        or a.device != out.device
        or a.device != a_scales.device
        or a.device != b_scales.device
        or a_scales.dtype != torch.float32
        or b_scales.dtype != torch.float32
        or a_scales.shape != (rows, 1)
        or b_scales.shape != (1, cols)
        or not a.is_contiguous()
        or b.stride() != (1, inner)
        or not out.is_contiguous()
        or not a_scales.is_contiguous()
        or not b_scales.is_contiguous()
        or not _compatible_sm90_device(a.device)
    ):
        return None
    if _loaded_op is None and torch.cuda.is_current_stream_capturing():
        return None
    op = load_prepared_op()
    if op is None:
        return None
    try:
        return op(out, a, b, a_scales, b_scales)
    except RuntimeError:
        return None
