from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from torchinferno.models.deepseek_v4.tensor_parallel import (
    DeepSeekV4TensorParallelForCausalLM,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    model = DeepSeekV4TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        device=device,
    ).eval()
    input_ids = torch.arange(args.prompt_tokens, device=device)[None]

    # Warm every production kernel, then rebuild an equivalent cache for the
    # measured one-token decode.
    warm_cache = model.allocate_cache(1, args.max_seq_len)
    warm_logits, warm_cache = model(input_ids, cache=warm_cache)
    warm_token = warm_logits[:, -1].argmax(-1)
    model(warm_token[:, None], cache=warm_cache)
    torch.cuda.synchronize(device)
    model.release_cache(warm_cache)

    cache = model.allocate_cache(1, args.max_seq_len)
    logits, cache = model(input_ids, cache=cache)
    token = logits[:, -1].argmax(-1)
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
        model(token[:, None], cache=cache)
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
                    "prompt_tokens": args.prompt_tokens,
                    "tensor_parallel_size": dist.get_world_size(),
                    "expert_backend": model.cuda_expert_backend,
                    "torch_version": torch.__version__,
                },
                indent=2,
            )
            + "\n"
        )
        print(table, flush=True)
    model.release_cache(cache)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
