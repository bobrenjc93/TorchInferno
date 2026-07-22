from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.deepseek_v4.tensor_parallel import (
    DeepSeekV4TensorParallelForCausalLM,
    set_tensor_parallel_process_group,
)
from torchinferno.runtime.disaggregated import (
    DisaggregatedPrefillDecodeModel,
    initialize_disaggregated_topology,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--prompt-stride", type=int, default=0)
    parser.add_argument("--new-tokens", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=140)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--profile-transfer", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    topology = initialize_disaggregated_topology(args.tensor_parallel_size)
    set_tensor_parallel_process_group(topology.role_group)
    started = time.perf_counter()
    role_model = DeepSeekV4TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=topology.device,
    ).eval()
    model = DisaggregatedPrefillDecodeModel(
        role_model,
        topology,
        cache_backend="v4-heterogeneous",
        profile_transfer=args.profile_transfer,
    ).eval()
    torch.cuda.synchronize(topology.device)
    load_seconds = time.perf_counter() - started

    if not model.is_coordinator:
        try:
            model.run_worker_loop()
        finally:
            dist.destroy_process_group()
        return
    try:
        input_ids = torch.arange(
            args.prompt_tokens, dtype=torch.long, device=topology.device
        )[None]
        offsets = (
            torch.arange(args.batch_size, device=topology.device)[:, None]
            * args.prompt_stride
        )
        input_ids = (input_ids + offsets).remainder(role_model.args.vocab_size).contiguous()
        for _ in range(args.warmup):
            model.generate(
                input_ids,
                max_new_tokens=args.new_tokens,
                temperature=0.0,
            )
        torch.cuda.synchronize(topology.device)
        model.reset_disaggregation_stats()
        generation_seconds = []
        output = None
        for _ in range(args.iterations):
            started = time.perf_counter()
            output = model.generate(
                input_ids,
                max_new_tokens=args.new_tokens,
                temperature=0.0,
            )
            torch.cuda.synchronize(topology.device)
            generation_seconds.append(time.perf_counter() - started)
        assert output is not None
        print(
            {
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "output_tail": output[:, -min(8, output.size(1)) :].tolist(),
                "stats": model.disaggregation_stats(),
                "memory_gib": torch.cuda.memory_allocated() / (1024**3),
                "expert_backend": role_model.cuda_expert_backend,
            },
            flush=True,
        )
    finally:
        model.shutdown_workers()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
