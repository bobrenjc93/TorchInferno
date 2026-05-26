from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from torchinferno.openai_server import OpenAIServerConfig, build_engine, _forward, _reset_generation_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="agent_space/profile_tp_prefill_torchprof.json")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    engine = build_engine(
        OpenAIServerConfig(
            model=args.model,
            tensor_parallel_size=world_size,
            device=f"cuda:{local_rank}",
            dtype="auto",
            trust_remote_code=True,
            llama_parallelism="tensor",
        )
    )
    device = engine.device
    vocab_size = int(getattr(getattr(engine.model, "config", object()), "vocab_size", 32000))
    input_ids = (torch.arange(args.prompt_tokens, device=device, dtype=torch.long) % vocab_size)[None, :]
    cache = engine._generation_cache(1, args.prompt_tokens + args.max_tokens, model=engine.model)

    for _ in range(args.warmup):
        _reset_generation_cache(cache)
        _forward(engine.model, input_ids, cache)
    torch.cuda.synchronize(device)
    _reset_generation_cache(cache)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        _forward(engine.model, input_ids, cache)
        torch.cuda.synchronize(device)

    if int(getattr(engine.model, "rank", 0)) == 0:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        print(table, flush=True)
        rows = []
        for event in prof.key_averages():
            rows.append(
                {
                    "key": event.key,
                    "cpu_time_total_us": event.cpu_time_total,
                    "cuda_time_total_us": event.cuda_time_total,
                    "count": event.count,
                }
            )
        with open(args.output, "w") as f:
            json.dump(sorted(rows, key=lambda row: row["cuda_time_total_us"], reverse=True), f, indent=2)
        print(f"json_output={args.output}", flush=True)

    engine.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
