from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Callable

import torch
import torch.distributed as dist

from torchinferno.openai_server import (
    OpenAIServerConfig,
    build_engine,
    _prefill_repeated_prefix_next_token,
    _repeat_generation_cache_first_batch,
    _reset_generation_cache,
    _sample,
    _try_decode_one_token_logits_graph,
)


def _event_ms(fn: Callable[[], object], device: torch.device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
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
    model = engine.model
    device = engine.device
    messages = [
        {"role": "system", "content": "You are a calculator. Respond with only the numerical answer, nothing else."},
        {"role": "user", "content": "17 * 23 ="},
    ]
    prompt = engine.tokenizer.encode_messages(messages)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    cache = engine._generation_cache(
        args.batch_size,
        input_ids.size(1) + args.max_tokens,
        model=model,
    )

    decode_ms: list[float] = []
    sample_ms: list[float] = []
    total_ms: list[float] = []
    logits_holder: dict[str, torch.Tensor] = {}

    def prepare() -> torch.Tensor:
        _reset_generation_cache(cache)
        next_token, _ = _prefill_repeated_prefix_next_token(
            model,
            input_ids,
            cache,
            args.batch_size,
            args.temperature,
            allow_capture=False,
        )
        next_token = next_token.to(device)
        _repeat_generation_cache_first_batch(cache, args.batch_size)
        return next_token

    for _ in range(args.warmup):
        with torch.inference_mode():
            next_token = prepare()
            logits = _try_decode_one_token_logits_graph(model, next_token[:, None], cache)
            if logits is None:
                raise RuntimeError("decode logits graph unavailable")
            _sample(model, logits[:, -1, :], args.temperature)
        torch.cuda.synchronize(device)

    for _ in range(args.iters):
        with torch.inference_mode():
            next_token = prepare()

        def decode() -> None:
            with torch.inference_mode():
                logits = _try_decode_one_token_logits_graph(model, next_token[:, None], cache)
                if logits is None:
                    raise RuntimeError("decode logits graph unavailable")
                logits_holder["logits"] = logits

        decode_time = _event_ms(decode, device)

        def sample() -> None:
            with torch.inference_mode():
                _sample(model, logits_holder["logits"][:, -1, :], args.temperature)

        sample_time = _event_ms(sample, device)

        def decode_and_sample() -> None:
            with torch.inference_mode():
                _reset_generation_cache(cache)
                token, _ = _prefill_repeated_prefix_next_token(
                    model,
                    input_ids,
                    cache,
                    args.batch_size,
                    args.temperature,
                    allow_capture=False,
                )
                token = token.to(device)
                _repeat_generation_cache_first_batch(cache, args.batch_size)
                logits = _try_decode_one_token_logits_graph(model, token[:, None], cache)
                if logits is None:
                    raise RuntimeError("decode logits graph unavailable")
                _sample(model, logits[:, -1, :], args.temperature)

        total_time = _event_ms(decode_and_sample, device)
        decode_ms.append(decode_time)
        sample_ms.append(sample_time)
        total_ms.append(total_time)

    summary = {
        "rank": int(getattr(model, "rank", 0)),
        "world_size": int(getattr(model, "world_size", 1)),
        "prompt_tokens": len(prompt),
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "decode_graph_p50_ms": _median(decode_ms),
        "sample_p50_ms": _median(sample_ms),
        "prefill_decode_sample_p50_ms": _median(total_ms),
        "decode_graph_ms": decode_ms,
        "sample_ms": sample_ms,
        "prefill_decode_sample_ms": total_ms,
    }
    if int(getattr(model, "rank", 0)) == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
