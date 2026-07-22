#!/usr/bin/env python3
"""Build DeepSeek V4 CUDA artifacts outside model and serving runtime paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("tensor-parallel sizes must be positive integers")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("artifact_dir")
    parser.add_argument("--tensor-parallel-sizes", type=_parse_sizes, default=(4, 8))
    parser.add_argument("--prompt-tokens", type=int, default=129)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    root = Path(args.artifact_dir).expanduser().resolve()
    tilelang_root = root / "tilelang"
    marlin_root = root / "marlin"
    tilelang_root.mkdir(parents=True, exist_ok=True)
    marlin_root.mkdir(parents=True, exist_ok=True)
    smoke = Path(__file__).with_name("smoke_deepseek_v4_tp.py")
    environment = os.environ.copy()
    environment.update(
        {
            "TORCHINFERNO_V4_KERNEL_PREPARE": "1",
            "TORCHINFERNO_V4_KERNEL_ARTIFACT_DIR": str(tilelang_root),
            "TVM_FFI_CACHE_DIR": str(marlin_root),
        }
    )
    for size in args.tensor_parallel_sizes:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(size),
            str(smoke),
            args.checkpoint,
            "--batch-size",
            "1",
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--new-tokens",
            "2",
            "--max-seq-len",
            str(args.max_seq_len),
            "--iterations",
            "1",
            "--prepare-kernels",
            str(tilelang_root),
        ]
        subprocess.run(command, env=environment, check=True)
    print(
        {
            "tilelang_artifacts": str(tilelang_root),
            "marlin_artifacts": str(marlin_root),
            "tensor_parallel_sizes": args.tensor_parallel_sizes,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
