#!/usr/bin/env python3
"""Profile one representative Llama TP mixed-prefill step by model region."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelForCausalLM,
    validate_symm_mem_allreduce_collective,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-capacity", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--cache-tokens", type=int, default=0)
    parser.add_argument("--query-tokens", type=int, default=16)
    parser.add_argument("--token-bucket", action="store_true")
    parser.add_argument("--ragged-decode", action="store_true")
    parser.add_argument("--ragged-decode-sweep", action="store_true")
    parser.add_argument("--indexed-rows", action="store_true")
    parser.add_argument("--online-engine", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-sampling", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    return parser.parse_args()


def _profile_token_bucket(
    model: Llama3TensorParallelForCausalLM,
    args: argparse.Namespace,
) -> None:
    device = model.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    if args.rows > args.batch_capacity:
        raise ValueError("rows cannot exceed batch capacity")
    token_capacity = args.rows * args.query_tokens
    max_seq_len = args.prefix_tokens + args.query_tokens
    cache = model.allocate_cache(args.batch_capacity, max_seq_len)
    for layer in cache.layers:
        layer.keys.zero_()
        layer.values.zero_()

    generator = torch.Generator(device=device).manual_seed(1701)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, token_capacity),
        generator=generator,
        device=device,
    )
    q_lens = torch.tensor(
        [*([args.query_tokens] * args.rows), *([0] * (args.batch_capacity - args.rows))],
        dtype=torch.long,
        device=device,
    )
    start_positions = torch.tensor(
        [*([args.prefix_tokens] * args.rows), *([0] * (args.batch_capacity - args.rows))],
        dtype=torch.long,
        device=device,
    )
    row_indices = torch.arange(args.batch_capacity, dtype=torch.long, device=device)
    write_positions = torch.cat(
        [
            torch.arange(
                args.prefix_tokens,
                args.prefix_tokens + args.query_tokens,
                dtype=torch.long,
                device=device,
            )
            for _ in range(args.rows)
        ]
    )
    flat_rows = torch.arange(args.rows, device=device, dtype=torch.long).repeat_interleave(
        args.query_tokens
    )
    logit_positions = torch.tensor(
        [
            *[(index + 1) * args.query_tokens - 1 for index in range(args.rows)],
            *([0] * (args.batch_capacity - args.rows)),
        ],
        dtype=torch.long,
        device=device,
    )

    capture_ready = False

    def replay() -> torch.Tensor:
        nonlocal capture_ready
        output = model.try_prefill_token_bucket_fa3_graph(
            input_ids,
            cache,
            start_positions=start_positions,
            q_lens=q_lens,
            row_indices=row_indices,
            write_positions=write_positions,
            flat_rows=flat_rows,
            logit_positions=logit_positions,
            capture_on_miss=not args.replay_only or not capture_ready,
        )
        if output is None:
            raise RuntimeError("token-bucket FA3 graph is unavailable")
        capture_ready = True
        return output

    with torch.inference_mode():
        output = replay()
        torch.cuda.synchronize(device)
        output = replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start_s = time.perf_counter()
        start.record()
        for _ in range(20):
            replay()
        end.record()
        torch.cuda.synchronize(device)
        replay_wall_ms = (time.perf_counter() - wall_start_s) * 1000.0 / 20.0
    if rank == 0:
        print(
            json.dumps(
                {
                    "batch_capacity": args.batch_capacity,
                    "fp8_fused_activation_eligible_calls": sum(
                        int(getattr(layer, "_fp8_fused_activation_eligible_calls", 0))
                        for layer in model.layers
                    ),
                    "fp8_fused_activation_success_calls": sum(
                        int(getattr(layer, "_fp8_fused_activation_success_calls", 0))
                        for layer in model.layers
                    ),
                    "fp8_qkv_cached_layers": sum(
                        getattr(layer, "_fp8_per_token_qkv_wq", None) is not None
                        for layer in model.layers
                    ),
                    "rows": args.rows,
                    "prefix_tokens": args.prefix_tokens,
                    "query_tokens": args.query_tokens,
                    "token_capacity": token_capacity,
                    "replay_ms": start.elapsed_time(end) / 20.0,
                    "replay_wall_ms": replay_wall_ms,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    if args.profile_sampling:
        sample_iterations = 100
        with torch.inference_mode():
            for _ in range(10):
                tokens = model._sample_next_token(output[:, -1, :], 0.0)
                tokens.detach().cpu().tolist()
            torch.cuda.synchronize(device)
            sample_start = time.perf_counter()
            for _ in range(sample_iterations):
                tokens = model._sample_next_token(output[:, -1, :], 0.0)
                tokens.detach().cpu().tolist()
            torch.cuda.synchronize(device)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "sample_and_readback_ms": (
                            (time.perf_counter() - sample_start)
                            * 1000.0
                            / sample_iterations
                        ),
                        "sample_rows": int(output.size(0)),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
    if args.profile:
        profile_context = (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            if rank == 0
            else nullcontext()
        )
        with profile_context as profiler:
            if dist.is_initialized():
                dist.barrier()
            replay()
            torch.cuda.synchronize(device)
        if rank == 0:
            print(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=40,
                ),
                flush=True,
            )


def _profile_ragged_decode(
    model: Llama3TensorParallelForCausalLM,
    args: argparse.Namespace,
) -> None:
    device = model.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    cache = model.allocate_cache(
        args.batch_capacity,
        max(args.prefix_tokens + 16, args.cache_tokens),
    )
    for layer in cache.layers:
        layer.keys.zero_()
        layer.values.zero_()
    input_ids = torch.ones((args.rows, 1), dtype=torch.long, device=device)
    seq_lens = torch.full(
        (args.batch_capacity,),
        args.prefix_tokens,
        dtype=torch.long,
        device=device,
    )
    row_indices = (
        (torch.arange(args.rows, device=device, dtype=torch.long) * 5)
        % args.batch_capacity
        if args.indexed_rows
        else None
    )
    capture_ready = False

    def replay() -> torch.Tensor:
        nonlocal capture_ready
        output = model.try_decode_ragged_token_graph(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            temperature=0.0,
            capture_on_miss=not args.replay_only or not capture_ready,
        )
        if output is None:
            raise RuntimeError("ragged decode token graph is unavailable")
        capture_ready = True
        return output

    with torch.inference_mode():
        replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start_s = time.perf_counter()
        start.record()
        for _ in range(20):
            replay()
        end.record()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - wall_start_s) * 1000.0 / 20.0
    if rank == 0:
        print(
            json.dumps(
                {
                    "batch_capacity": args.batch_capacity,
                    "fp8_fused_activation_eligible_calls": sum(
                        int(getattr(layer, "_fp8_fused_activation_eligible_calls", 0))
                        for layer in model.layers
                    ),
                    "fp8_fused_activation_success_calls": sum(
                        int(getattr(layer, "_fp8_fused_activation_success_calls", 0))
                        for layer in model.layers
                    ),
                    "fp8_qkv_cached_layers": sum(
                        getattr(layer, "_fp8_per_token_qkv_wq", None) is not None
                        for layer in model.layers
                    ),
                    "rows": args.rows,
                    "prefix_tokens": args.prefix_tokens,
                    "replay_ms": start.elapsed_time(end) / 20.0,
                    "replay_wall_ms": wall_ms,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    if args.profile:
        captured_graphs = getattr(model, "_ragged_decode_graphs", {})
        if not captured_graphs:
            raise RuntimeError("ragged decode profiling requires a captured graph")
        captured = next(reversed(captured_graphs.values()))
        profile_context = (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            if rank == 0
            else nullcontext()
        )
        with profile_context as profiler:
            if dist.is_initialized():
                dist.barrier()
            captured.graph.replay()
            torch.cuda.synchronize(device)
        if rank == 0:
            print(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=40,
                ),
                flush=True,
            )


def _profile_ragged_decode_sweep(
    model: Llama3TensorParallelForCausalLM,
    args: argparse.Namespace,
) -> None:
    device = model.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    cache = model.allocate_cache(
        args.batch_capacity,
        max(args.prefix_tokens + 16, args.cache_tokens),
    )
    for layer in cache.layers:
        layer.keys.zero_()
        layer.values.zero_()
    seq_lens = torch.full(
        (args.batch_capacity,),
        args.prefix_tokens,
        dtype=torch.long,
        device=device,
    )
    rows_to_profile = tuple(
        rows
        for rows in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
        if rows <= args.batch_capacity
    )
    results: list[dict[str, float | int]] = []
    with torch.inference_mode():
        for rows in rows_to_profile:
            input_ids = torch.ones((rows, 1), dtype=torch.long, device=device)

            def replay(*, capture_on_miss: bool) -> torch.Tensor:
                output = model.try_decode_ragged_token_graph(
                    input_ids,
                    cache,
                    seq_lens=seq_lens,
                    row_indices=None,
                    temperature=0.0,
                    capture_on_miss=capture_on_miss,
                )
                if output is None:
                    raise RuntimeError(f"ragged decode graph is unavailable for batch {rows}")
                return output

            replay(capture_on_miss=True)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start_s = time.perf_counter()
            start.record()
            for _ in range(40):
                replay(capture_on_miss=False)
            end.record()
            torch.cuda.synchronize(device)
            results.append(
                {
                    "rows": rows,
                    "replay_ms": start.elapsed_time(end) / 40.0,
                    "replay_wall_ms": (time.perf_counter() - wall_start_s) * 1000.0 / 40.0,
                }
            )
    if rank == 0:
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)


def _profile_online_engine(
    model: Llama3TensorParallelForCausalLM,
    args: argparse.Namespace,
) -> None:
    from torchinferno.runtime.serving import ContinuousBatchEngine, ServingRequest

    rank = dist.get_rank() if dist.is_initialized() else 0
    engine = ContinuousBatchEngine(
        model,
        device=model.device,
        max_active_requests=args.rows,
        prefix_cache_capacity=0,
        enable_ragged_decode=True,
        store_reusable_prefixes=False,
        store_full_prompt_prefixes=False,
        profile_timings=args.profile,
    )
    engine.start_online(max_seq_len=max(args.cache_tokens, args.prefix_tokens + 16))
    for row in range(args.rows):
        prompt = tuple(
            1 + ((row * 97 + position * 13) % 4096)
            for position in range(args.prefix_tokens)
        )
        engine.submit_online(
            ServingRequest(
                f"synthetic-{row}",
                prompt,
                10,
                arrival_step=0,
            )
        )

    with torch.inference_mode():
        engine.step_online()
        torch.cuda.synchronize(model.device)
        before = {
            "prepare": engine.stats.decode_ragged_prepare_ms,
            "model": engine.stats.decode_ragged_model_ms,
            "readback": engine.stats.decode_ragged_cpu_tokens_ms,
            "state": engine.stats.decode_ragged_state_update_ms,
        }
        step_ms: list[float] = []
        for _ in range(8):
            start_s = time.perf_counter()
            engine.step_online()
            torch.cuda.synchronize(model.device)
            step_ms.append((time.perf_counter() - start_s) * 1000.0)
        after = {
            "prepare": engine.stats.decode_ragged_prepare_ms,
            "model": engine.stats.decode_ragged_model_ms,
            "readback": engine.stats.decode_ragged_cpu_tokens_ms,
            "state": engine.stats.decode_ragged_state_update_ms,
        }
    if rank == 0:
        print(
            json.dumps(
                {
                    "rows": args.rows,
                    "prefix_tokens": args.prefix_tokens,
                    "cache_tokens": max(args.cache_tokens, args.prefix_tokens + 16),
                    "step_ms": step_ms,
                    "step_mean_ms": sum(step_ms) / len(step_ms),
                    "component_mean_ms": {
                        name: (after[name] - before[name]) / len(step_ms)
                        for name in after
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    args = _args()
    if args.rows < 1 or args.prefix_tokens < 1 or args.query_tokens < 1:
        raise ValueError("rows and token counts must be positive")
    model = Llama3TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype="bfloat16",
    ).eval()
    device = model.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    validate_symm_mem_allreduce_collective(model, device)
    if args.ragged_decode_sweep:
        _profile_ragged_decode_sweep(model, args)
        return
    if args.token_bucket:
        _profile_token_bucket(model, args)
        return
    if args.ragged_decode:
        _profile_ragged_decode(model, args)
        return
    if args.online_engine:
        _profile_online_engine(model, args)
        return
    generator = torch.Generator(device=device).manual_seed(1701)
    max_seq_len = args.prefix_tokens + 2 * args.query_tokens
    cache = model.allocate_cache(args.rows, max_seq_len)
    row_indices = torch.arange(args.rows, device=device, dtype=torch.long)

    def step(start: int) -> torch.Tensor:
        input_ids = torch.randint(
            0,
            model.config.vocab_size,
            (args.rows, args.query_tokens),
            generator=generator,
            device=device,
        )
        seq_lens = torch.full(
            (args.rows,),
            start,
            device=device,
            dtype=torch.long,
        )
        q_lens = torch.full(
            (args.rows,),
            args.query_tokens,
            device=device,
            dtype=torch.long,
        )
        write_positions = torch.arange(
            start,
            start + args.query_tokens,
            device=device,
            dtype=torch.long,
        ).expand(args.rows, -1)
        logit_positions = torch.full(
            (args.rows,),
            args.query_tokens - 1,
            device=device,
            dtype=torch.long,
        )
        return model.forward_step_flashinfer(
            input_ids,
            cache,
            seq_lens=seq_lens,
            q_lens=q_lens,
            write_positions=write_positions,
            logit_positions=logit_positions,
            row_indices=row_indices,
        )

    prefix = torch.randint(
        0,
        model.config.vocab_size,
        (args.rows, args.prefix_tokens),
        generator=generator,
        device=device,
    )
    with torch.inference_mode():
        model.forward(
            prefix,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        step(args.prefix_tokens)
        torch.cuda.synchronize(device)
        for layer in model.layers:
            layer.profile_seconds = {}
            layer.profile_counts = {}
        start_s = time.perf_counter()
        step(args.prefix_tokens + args.query_tokens)
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - start_s) * 1000.0

    regions: dict[str, float] = {}
    counts: dict[str, int] = {}
    for layer in model.layers:
        for name, seconds in (layer.profile_seconds or {}).items():
            regions[name] = regions.get(name, 0.0) + seconds * 1000.0
        for name, count in (layer.profile_counts or {}).items():
            counts[name] = counts.get(name, 0) + count

    if rank == 0:
        ordered = dict(sorted(regions.items(), key=lambda item: item[1], reverse=True))
        print(
            json.dumps(
                {
                    "rows": args.rows,
                    "prefix_tokens": args.prefix_tokens,
                    "query_tokens": args.query_tokens,
                    "model_tokens": args.rows * args.query_tokens,
                    "wall_ms_with_region_sync": wall_ms,
                    "region_ms": ordered,
                    "region_calls": counts,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
