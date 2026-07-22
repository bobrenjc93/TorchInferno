from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torchinferno.models.deepseek_v4.tensor_parallel as v4_tp

from torchinferno.kernels import deepseek_v4_tilelang as v4_kernels
from torchinferno.models.deepseek_v4.tensor_parallel import (
    DeepSeekV4TensorParallelForCausalLM,
)


_FUSED_ENV_NAMES = (
    "TORCHINFERNO_V4_FUSED_RMSNORM",
    "TORCHINFERNO_V4_FUSED_Q_NORM_ROPE",
    "TORCHINFERNO_V4_FUSED_ROPE",
    "TORCHINFERNO_V4_FUSED_HC_PRE_GEMM",
    "TORCHINFERNO_V4_FUSED_HC_POST",
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no", "off"}


def _git_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = git("status", "--short")
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _loaded_artifacts() -> list[dict[str, object]]:
    artifacts = []
    for name, args, keyword_items in v4_kernels._kernels:
        kwargs = dict(keyword_items)
        artifacts.append(
            {
                "name": name,
                "args": v4_kernels._canonical(args),
                "kwargs": v4_kernels._canonical(kwargs),
                "key": v4_kernels._artifact_key(name, args, kwargs),
            }
        )
    return sorted(artifacts, key=lambda value: str(value["key"]))


def _artifact_directory_entries(environment_name: str) -> list[str]:
    configured = os.environ.get(environment_name, "").strip()
    if not configured:
        return []
    root = Path(configured).expanduser()
    try:
        return sorted(path.name for path in root.iterdir() if not path.name.startswith("."))
    except OSError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--prompt-stride", type=int, default=17)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    model = DeepSeekV4TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    ).eval()
    input_ids = torch.arange(args.prompt_tokens, device=device).unsqueeze(0)
    offsets = torch.arange(args.batch_size, device=device).unsqueeze(1) * args.prompt_stride
    input_ids = (input_ids + offsets).remainder(model.args.vocab_size).contiguous()

    # Warm every production kernel, then rebuild an equivalent cache for the
    # measured one-token decode.
    warm_cache = model.allocate_cache(args.batch_size, args.max_seq_len)
    warm_logits, warm_cache = model(
        input_ids,
        cache=warm_cache,
        return_sharded_logits=dist.get_world_size() > 1,
    )
    warm_token = model._sample_next_token(warm_logits[:, -1], 0.0)
    for _ in range(args.decode_steps):
        warm_logits, warm_cache = model(
            warm_token[:, None],
            cache=warm_cache,
            return_sharded_logits=dist.get_world_size() > 1,
        )
        warm_token = model._sample_next_token(warm_logits[:, -1], 0.0)
    torch.cuda.synchronize(device)
    model.release_cache(warm_cache)

    cache = model.allocate_cache(args.batch_size, args.max_seq_len)
    logits, cache = model(
        input_ids,
        cache=cache,
        return_sharded_logits=dist.get_world_size() > 1,
    )
    token = model._sample_next_token(logits[:, -1], 0.0)
    torch.cuda.synchronize(device)

    rank = dist.get_rank()
    output = Path(args.output)
    profile_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        if rank == 0
        else nullcontext()
    )
    with profile_context as profile:
        for _ in range(args.decode_steps):
            with torch.profiler.record_function("deepseek_v4_decode_step"):
                logits, cache = model(
                    token[:, None],
                    cache=cache,
                    return_sharded_logits=dist.get_world_size() > 1,
                )
                token = model._sample_next_token(logits[:, -1], 0.0)
        torch.cuda.synchronize(device)

    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        table = profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=50
        )
        (output / "decode_cuda_table.txt").write_text(table)
        profile.export_chrome_trace(str(output / "decode_trace.json"))
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "batch_size": args.batch_size,
                    "prompt_tokens": args.prompt_tokens,
                    "prompt_stride": args.prompt_stride,
                    "decode_steps": args.decode_steps,
                    "max_seq_len": args.max_seq_len,
                    "tensor_parallel_size": dist.get_world_size(),
                    "expert_backend": model.cuda_expert_backend,
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cuda_device": torch.cuda.get_device_name(device),
                    "cuda_capability": list(torch.cuda.get_device_capability(device)),
                    "source": _git_metadata(),
                    "tilelang_definitions_hash": v4_kernels._definitions_hash(),
                    "tilelang_artifact_root": os.environ.get(
                        "TORCHINFERNO_V4_KERNEL_ARTIFACT_DIR"
                    ),
                    "tilelang_artifacts": _loaded_artifacts(),
                    "marlin_artifact_root": os.environ.get("TVM_FFI_CACHE_DIR"),
                    "marlin_artifacts": _artifact_directory_entries("TVM_FFI_CACHE_DIR"),
                    "fused_env": {
                        name: {
                            "value": os.environ.get(name),
                            "enabled": _enabled(name),
                        }
                        for name in _FUSED_ENV_NAMES
                    },
                    "precompiled_rmsnorm_available": (
                        v4_tp._precompiled_rmsnorm_op() is not None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(table, flush=True)
    model.release_cache(cache)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
