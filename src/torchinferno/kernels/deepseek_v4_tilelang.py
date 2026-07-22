"""Runtime loader for offline-built DeepSeek V4 TileLang kernels.

The TileLang programs live in ``deepseek_v4_tilelang_definitions`` and are
imported only by the explicit offline builder. Normal model execution loads
concrete shared-library artifacts and never invokes TileLang code generation
or a CUDA compiler.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import cloudpickle
import torch


_ARTIFACT_ENV = "TORCHINFERNO_V4_KERNEL_ARTIFACT_DIR"
_ARTIFACT_SCHEMA = 2
_kernels: dict[tuple[object, ...], object] = {}


def _artifact_root() -> Path:
    configured = os.environ.get(_ARTIFACT_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            "DeepSeek V4 CUDA kernels require offline-built artifacts; set "
            f"{_ARTIFACT_ENV} to a prepared directory"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"DeepSeek V4 kernel artifact directory does not exist: {root}")
    return root


def _canonical(value: object) -> object:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    raise TypeError(f"unsupported V4 kernel specialization value: {value!r}")


def _artifact_key(name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    definitions_hash = _definitions_hash()
    payload = json.dumps(
        {
            "schema": _ARTIFACT_SCHEMA,
            "definitions": definitions_hash,
            "name": name,
            "args": _canonical(args),
            "kwargs": _canonical(kwargs),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{name}-{hashlib.sha256(payload.encode('ascii')).hexdigest()[:20]}"


@lru_cache(maxsize=1)
def _definitions_hash() -> str:
    source = Path(__file__).with_name("deepseek_v4_tilelang_definitions.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _compatibility() -> dict[str, str]:
    import tilelang

    capability = torch.cuda.get_device_capability()
    return {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "tilelang": str(getattr(tilelang, "__version__", "unknown")),
        "cuda_arch": f"sm_{capability[0]}{capability[1]}",
        "definitions": _definitions_hash(),
        "schema": str(_ARTIFACT_SCHEMA),
    }


def _load_artifact(path: Path, key: str) -> object:
    from tilelang import tvm
    from tilelang.jit import JITKernel

    metadata_path = path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid DeepSeek V4 kernel artifact {key}: {exc}") from exc
    expected = _compatibility()
    actual = metadata.get("compatibility")
    if actual != expected:
        raise RuntimeError(
            f"DeepSeek V4 kernel artifact {key} has incompatible ABI: "
            f"expected {expected}, got {actual}"
        )
    try:
        with (path / "params.pkl").open("rb") as handle:
            params = cloudpickle.load(handle)
        host_source = (path / "host_kernel.cu").read_text(encoding="utf-8")
        device_source = (path / "device_kernel.cu").read_text(encoding="utf-8")
        prim_func = tvm.ir.load_json((path / "prim_func.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"incomplete DeepSeek V4 kernel artifact {key}: {exc}") from exc
    library = path / "kernel_lib.so"
    if not library.is_file():
        raise RuntimeError(f"incomplete DeepSeek V4 kernel artifact {key}: missing kernel_lib.so")
    return JITKernel.from_database(
        func=prim_func,
        host_kernel_source=host_source,
        device_kernel_source=device_source,
        kernel_lib_path=str(library),
        params=params,
        target=str(metadata["target"]),
        target_host=None,
        out_idx=metadata.get("out_idx"),
        execution_backend=str(metadata.get("execution_backend", "tvm_ffi")),
        pass_configs=None,
    )


def _kernel(name: str, *args: object, **kwargs: object) -> object:
    specialization = (name, args, tuple(sorted(kwargs.items())))
    cached = _kernels.get(specialization)
    if cached is not None:
        return cached
    key = _artifact_key(name, args, kwargs)
    path = _artifact_root() / key
    if path.is_dir():
        kernel = _load_artifact(path, key)
    else:
        raise RuntimeError(
            f"DeepSeek V4 kernel specialization {key} is not prepared in {path.parent}; "
            "run scripts/prepare_deepseek_v4_kernels.py offline"
        )
    _kernels[specialization] = kernel
    return kernel


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    n = x.size(-1)
    if n % block_size:
        raise ValueError("FP8 quantization width must be divisible by block size")
    scale_type = "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"
    value = x.contiguous()
    output = torch.empty_like(value) if inplace else torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale = value.new_empty(*value.size()[:-1], n // block_size, dtype=scale_dtype)
    kernel = _kernel(
        "act_quant",
        n,
        block_size,
        scale_dtype=scale_type,
        round_scale=scale_fmt is not None,
        inplace=inplace,
    )
    kernel(value.view(-1, n), output.view(-1, n), scale.view(-1, n // block_size))
    if inplace:
        x.copy_(output)
        return x
    return output, scale


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    n = x.size(-1)
    if n % block_size:
        raise ValueError("FP4 quantization width must be divisible by block size")
    value = x.contiguous()
    output = (
        torch.empty_like(value)
        if inplace
        else value.new_empty(*value.shape[:-1], n // 2, dtype=torch.float4_e2m1fn_x2)
    )
    scale = value.new_empty(
        *value.size()[:-1],
        n // block_size,
        dtype=torch.float8_e8m0fnu,
    )
    kernel = _kernel("fp4_quant", n, block_size, inplace=inplace)
    kernel(
        value.view(-1, n),
        output.view(-1, output.size(-1)),
        scale.view(-1, n // block_size),
    )
    if inplace:
        x.copy_(output)
        return x
    return output, scale


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not all(tensor.is_contiguous() for tensor in (a, a_s, b, b_s)):
        raise ValueError("FP8 GEMM inputs and scales must be contiguous")
    scale_type = "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    output = a.new_empty(*a.size()[:-1], n, dtype=torch.bfloat16)
    kernel = _kernel("fp8_gemm", n, k, scale_dtype=scale_type)
    kernel(a.view(m, k), b, output.view(m, n), a_s.view(m, -1), b_s)
    return output


def q_norm_rope(
    q: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.ndim != 3 or q.dtype != torch.bfloat16:
        raise ValueError("V4 Q norm/RoPE expects [tokens, heads, head_dim] BF16 input")
    if freqs.shape != (64,) or freqs.dtype != torch.float32 or not freqs.is_contiguous():
        raise ValueError("V4 Q norm/RoPE expects one contiguous 64-wide FP32 frequency row")
    if q.size(2) < freqs.size(0):
        raise ValueError("V4 Q norm/RoPE requires a 64-wide rotary tail")
    if freqs.device != q.device:
        raise ValueError("V4 Q norm/RoPE inputs must be on the same device")
    value = q.contiguous()
    if output is None:
        output = torch.empty_like(value)
    elif (
        output.shape != value.shape
        or output.dtype != value.dtype
        or output.device != value.device
        or not output.is_contiguous()
    ):
        raise ValueError("V4 Q norm/RoPE output must be contiguous and match the input")
    kernel = _kernel(
        "q_norm_rope",
        value.size(1),
        value.size(2),
        freqs.size(0),
        eps,
    )
    kernel(value, freqs, output)
    return output


def rope_inplace(
    value: torch.Tensor,
    freqs: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if value.ndim != 3 or value.dtype != torch.bfloat16 or not value.is_contiguous():
        raise ValueError("V4 RoPE expects contiguous [tokens, heads, head_dim] BF16 input")
    if freqs.shape != (64,) or freqs.dtype != torch.float32 or not freqs.is_contiguous():
        raise ValueError("V4 RoPE expects one contiguous 64-wide FP32 frequency row")
    if value.size(2) < freqs.size(0):
        raise ValueError("V4 RoPE requires a 64-wide rotary tail")
    if freqs.device != value.device:
        raise ValueError("V4 RoPE inputs must be on the same device")
    kernel = _kernel(
        "rope_inplace",
        value.size(1),
        value.size(2),
        freqs.size(0),
        inverse,
    )
    kernel(value, freqs)
    return value


def hc_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if value.ndim < 2 or residual.ndim != value.ndim + 1:
        raise ValueError("V4 mHC post inputs have incompatible ranks")
    if value.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("V4 mHC post values and residual must be BF16")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise ValueError("V4 mHC post mixing tensors must be FP32")
    if any(tensor.device != value.device for tensor in (residual, post, comb)):
        raise ValueError("V4 mHC post inputs must be on the same device")
    hidden = value.size(-1)
    hc = residual.size(-2)
    if residual.shape[:-2] != value.shape[:-1] or residual.size(-1) != hidden:
        raise ValueError("V4 mHC residual shape does not match the layer output")
    if post.shape != (*value.shape[:-1], hc) or comb.shape != (
        *value.shape[:-1],
        hc,
        hc,
    ):
        raise ValueError("V4 mHC mixing tensors do not match the residual shape")
    tokens = value.numel() // hidden
    output = torch.empty_like(residual)
    kernel = _kernel("hc_post", hc, hidden)
    kernel(
        value.contiguous().view(tokens, hidden),
        residual.contiguous().view(tokens, hc, hidden),
        post.contiguous().view(tokens, hc),
        comb.contiguous().view(tokens, hc, hc),
        output.view(tokens, hc, hidden),
    )
    return output


def hc_prenorm_gemm(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim < 3 or value.dtype != torch.bfloat16:
        raise ValueError("V4 mHC pre-norm expects [..., hc, hidden] BF16 input")
    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise ValueError("V4 mHC pre-norm projection weight must be FP32")
    if weight.device != value.device:
        raise ValueError("V4 mHC pre-norm inputs must be on the same device")
    hc = value.size(-2)
    hidden = value.size(-1)
    width = hc * hidden
    if weight.size(1) != width:
        raise ValueError("V4 mHC pre-norm projection width does not match the input")
    if width % 1024:
        raise ValueError("V4 mHC pre-norm projection width must be divisible by 1024")
    tokens = value.numel() // width
    output_shape = (*value.shape[:-2], weight.size(0))
    output = torch.empty(output_shape, dtype=torch.float32, device=value.device)
    square_sum = torch.empty(
        (*value.shape[:-2], 1),
        dtype=torch.float32,
        device=value.device,
    )
    kernel = _kernel("hc_prenorm_gemm", hidden, hc, weight.size(0))
    kernel(
        value.contiguous().view(tokens, width),
        weight.contiguous(),
        output.view(tokens, weight.size(0)),
        square_sum.view(tokens),
    )
    return output, square_sum


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch, tokens, heads, head_dim = q.size()
    if heads < 16:
        q = torch.cat((q, q.new_zeros(batch, tokens, 16 - heads, head_dim)), dim=2)
        attn_sink = torch.cat((attn_sink, attn_sink.new_zeros(16 - heads)))
    output = torch.empty_like(q)
    kernel = _kernel("sparse_attn", q.size(2), head_dim, softmax_scale)
    kernel(q, kv, output, attn_sink, topk_idxs)
    return output.narrow(2, 0, heads).contiguous() if heads < 16 else output


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, _ = mixes.size()
    pre = mixes.new_empty(batch, tokens, hc_mult)
    post = mixes.new_empty(batch, tokens, hc_mult)
    comb = mixes.new_empty(batch, tokens, hc_mult, hc_mult)
    kernel = _kernel("hc_split_sinkhorn", hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not all(tensor.is_contiguous() for tensor in (a, a_s, b, b_s)):
        raise ValueError("FP4 GEMM inputs and scales must be contiguous")
    scale_type = "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    output = a.new_empty(*a.size()[:-1], n, dtype=torch.bfloat16)
    kernel = _kernel("fp4_gemm", n, k, scale_dtype=scale_type)
    kernel(a.view(m, k), b, output.view(m, n), a_s.view(m, -1), b_s)
    return output


def prepared_artifact_keys() -> tuple[str, ...]:
    root = _artifact_root()
    return tuple(
        sorted(path.name for path in root.iterdir() if (path / "metadata.json").is_file())
    )
