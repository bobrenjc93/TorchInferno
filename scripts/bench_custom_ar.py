import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _benchmark(fn: Callable[[], object], *, iterations: int = 300, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1e6


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    hidden_size = 8192
    group_name = dist.group.WORLD.group_name

    custom_all_reduce = None
    custom_all_reduce_error = None
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        gloo_group = dist.new_group(backend="gloo")
        custom_all_reduce = CustomAllreduce(group=gloo_group, device=device)
        if custom_all_reduce.disabled:
            custom_all_reduce_error = "CustomAllreduce.disabled=True"
    except Exception as exc:
        custom_all_reduce_error = repr(exc)[:160]

    for rows in (16, 32, 48, 64, 128, 256, 512, 1024):
        buffer = symm_mem.empty(
            rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        symm_mem.rendezvous(buffer, group_name)
        input_tensor = torch.randn(
            rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        buffer.copy_(input_tensor)
        multimem_us = _benchmark(
            lambda: torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group_name)
        )
        custom_all_reduce_us = None
        if custom_all_reduce is not None and not custom_all_reduce.disabled:
            output = custom_all_reduce.custom_all_reduce(input_tensor)
            if output is not None:
                custom_all_reduce_us = _benchmark(
                    lambda: custom_all_reduce.custom_all_reduce(input_tensor)
                )
        if rank == 0:
            line = f"[{rows},8192] multimem={multimem_us:.1f}us"
            if custom_all_reduce_us is not None:
                line += (
                    f" custom_ar={custom_all_reduce_us:.1f}us"
                    f" speedup={multimem_us / custom_all_reduce_us:.2f}x"
                )
            else:
                reason = custom_all_reduce_error or "size not supported"
                line += f" custom_ar unavailable: {reason}"
            print(line, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
