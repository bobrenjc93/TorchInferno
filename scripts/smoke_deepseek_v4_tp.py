from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.deepseek_v4.tensor_parallel import DeepSeekV4TensorParallelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--prompt-stride", type=int, default=0)
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--prepare-kernels",
        help="offline-only TileLang artifact directory; never use in a serving process",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    if args.prepare_kernels:
        from torchinferno.kernels.deepseek_v4_tilelang_builder import (
            install_offline_builder,
            prepare_mxfp4_fallback,
        )

        install_offline_builder(args.prepare_kernels)
        if dist.get_rank() == 0:
            prepare_mxfp4_fallback(args.prepare_kernels)
        dist.barrier()
    started = time.perf_counter()
    model = DeepSeekV4TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    if dist.get_rank() == 0:
        print(
            {
                "stage": "loaded",
                "load_seconds": load_seconds,
                "memory_gib": torch.cuda.memory_allocated() / (1024**3),
                "expert_backend": model.cuda_expert_backend,
            },
            flush=True,
        )
    input_ids = torch.arange(args.prompt_tokens, device=device).unsqueeze(0)
    offsets = torch.arange(args.batch_size, device=device).unsqueeze(1) * args.prompt_stride
    input_ids = (input_ids + offsets).remainder(model.args.vocab_size).contiguous()
    for _ in range(args.warmup):
        model.generate(input_ids, max_new_tokens=args.new_tokens)
    torch.cuda.synchronize()
    generation_seconds = []
    output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = model.generate(input_ids, max_new_tokens=args.new_tokens)
        torch.cuda.synchronize()
        generation_seconds.append(time.perf_counter() - started)
    assert output is not None
    if dist.get_rank() == 0:
        print(
            {
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "output_shape": tuple(output.shape),
                "output_tail": output[0, -min(8, output.size(1)) :].tolist(),
                "generated_ids": output[:, -args.new_tokens :].tolist(),
                "max_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "expert_backend": model.cuda_expert_backend,
                "load_report": model.load_report,
            },
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
