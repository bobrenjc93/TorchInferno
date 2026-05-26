from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from torchinferno.openai_server import OpenAIServerConfig, build_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    engine = build_engine(
        OpenAIServerConfig(
            model=args.model,
            tensor_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
            device=f"cuda:{local_rank}",
            dtype="auto",
            trust_remote_code=True,
            llama_parallelism="tensor",
        )
    )
    device = engine.device
    messages = [
        {"role": "system", "content": "You are a calculator. Respond with only the numerical answer, nothing else."},
        {"role": "user", "content": "17 * 23 ="},
    ]
    prompt = engine.tokenizer.encode_messages(messages)
    input_ids = torch.tensor([prompt for _ in range(args.batch_size)], dtype=torch.long, device=device)

    def run_once() -> tuple[list[list[int | None]], float]:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        steps = list(
            engine._generate_batch_steps(
                input_ids,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                broadcast_tensor_parallel=False,
                row_max_tokens=[args.max_tokens for _ in range(args.batch_size)],
            )
        )
        torch.cuda.synchronize(device)
        return steps, (time.perf_counter() - start) * 1000.0

    for _ in range(args.warmup):
        run_once()
    results = []
    for _ in range(args.iters):
        steps, elapsed_ms = run_once()
        results.append(
            {
                "elapsed_ms": elapsed_ms,
                "steps": len(steps),
                "first_step": steps[0] if steps else [],
                "last_step": steps[-1] if steps else [],
                "non_empty_steps": sum(any(token is not None for token in step) for step in steps),
            }
        )
    if int(getattr(engine.model, "rank", 0)) == 0:
        print(json.dumps({"prompt_tokens": len(prompt), "results": results}, indent=2), flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
