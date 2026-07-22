"""Offline-only builder for concrete DeepSeek V4 TileLang artifacts."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

import cloudpickle

from torchinferno.kernels import deepseek_v4_tilelang as runtime


def build_kernel(
    artifact_root: str | Path,
    name: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Build or load one specialization under an inter-process lock."""

    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = runtime._artifact_key(name, args, kwargs)
    path = root / key
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (path / "metadata.json").is_file():
            return runtime._load_artifact(path, key)
        if path.exists():
            shutil.rmtree(path)

        # Importing this module invokes TileLang code generation. Keep it in
        # this explicit offline builder, never in model or serving runtime.
        from torchinferno.kernels import deepseek_v4_tilelang_definitions as definitions

        factory = getattr(definitions, f"{name}_kernel")
        kernel = factory(*args, **kwargs)
        adapter = kernel.adapter
        from tilelang import tvm

        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
        try:
            (temporary / "prim_func.json").write_text(
                tvm.ir.save_json(kernel.prim_func),
                encoding="utf-8",
            )
            (temporary / "device_kernel.cu").write_text(
                str(adapter.get_device_source()),
                encoding="utf-8",
            )
            (temporary / "host_kernel.cu").write_text(
                str(adapter.get_host_source()),
                encoding="utf-8",
            )
            with (temporary / "params.pkl").open("wb") as handle:
                cloudpickle.dump(adapter.params, handle)
            artifact_library = temporary / "kernel_lib.so"
            compiled_library_value = getattr(adapter, "libpath", None)
            compiled_library = (
                Path(compiled_library_value) if compiled_library_value else None
            )
            if compiled_library is not None and compiled_library.is_file():
                shutil.copy2(compiled_library, artifact_library)
            else:
                executable = getattr(adapter, "executable", None)
                export_library = getattr(executable, "export_library", None)
                if not callable(export_library):
                    raise RuntimeError(
                        "TileLang did not expose a loadable DeepSeek V4 kernel library"
                    )
                export_library(str(artifact_library))
            metadata = {
                "key": key,
                "name": name,
                "args": runtime._canonical(args),
                "kwargs": runtime._canonical(kwargs),
                "compatibility": runtime._compatibility(),
                "target": str(kernel.target),
                "execution_backend": str(kernel.execution_backend),
                "out_idx": runtime._canonical(adapter.result_idx),
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.rename(path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return kernel


def install_offline_builder(artifact_root: str | Path) -> None:
    """Route kernel requests to the builder in an explicit preparation process."""

    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ[runtime._ARTIFACT_ENV] = str(root)

    def build(name: str, *args: object, **kwargs: object) -> object:
        return build_kernel(root, name, *args, **kwargs)

    runtime._kernels.clear()
    runtime._kernel = build


def prepare_mxfp4_fallback(artifact_root: str | Path) -> tuple[str, ...]:
    """Build the general TileLang expert path even when Marlin is available."""

    specs = (
        (
            "act_quant",
            (4096, 128),
            {"scale_dtype": "float32", "round_scale": True, "inplace": False},
        ),
        (
            "act_quant",
            (2048, 128),
            {"scale_dtype": "float32", "round_scale": True, "inplace": False},
        ),
        ("fp4_gemm", (2048, 4096), {"scale_dtype": "float32"}),
        ("fp4_gemm", (4096, 2048), {"scale_dtype": "float32"}),
    )
    keys = []
    for name, args, kwargs in specs:
        build_kernel(artifact_root, name, *args, **kwargs)
        keys.append(runtime._artifact_key(name, args, kwargs))
    return tuple(keys)
