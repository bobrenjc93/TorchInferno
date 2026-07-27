"""Offline builder for the pinned SM90 FP8 output adapter."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from torch.utils.cpp_extension import load

from torchinferno.kernels.sgl_fp8_out import (
    EXPECTED_SGLANG_KERNEL_VERSION,
    compatibility,
    compatible_provider_version,
    prepared_library_path,
    provider_library_path,
    source_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()

    provider_version = version("sglang-kernel")
    if not compatible_provider_version(provider_version):
        raise RuntimeError(
            "SGL FP8 output preparation requires sglang-kernel "
            f"{EXPECTED_SGLANG_KERNEL_VERSION}, found {provider_version}"
        )
    common = provider_library_path()
    if common is None:
        raise RuntimeError("SGLang SM90 provider library is missing")

    output = prepared_library_path(args.artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    load(
        name=output.stem,
        sources=[str(source_path())],
        build_directory=str(output.parent),
        extra_cflags=["-O3"],
        extra_ldflags=[
            f"-L{common.parent}",
            "-l:common_ops.abi3.so",
            f"-Wl,-rpath,{common.parent}",
        ],
        is_python_module=False,
        verbose=True,
    )
    if not output.is_file():
        raise RuntimeError(f"kernel preparation did not produce {output}")
    metadata = {
        "compatibility": compatibility(),
        "library": output.name,
        "provider_library": str(common),
    }
    (output.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
