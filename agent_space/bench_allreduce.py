from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


def _time_cuda(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 64])
    parser.add_argument("--hidden", type=int, default=8192)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    group_name = dist.group.WORLD.group_name

    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:
        if dist.get_rank() == 0:
            print(json.dumps({"error": repr(exc)}), flush=True)
        return

    results: dict[str, dict[str, float]] = {}
    dtype = torch.bfloat16
    for batch in args.batches:
        shape = (batch, 1, args.hidden)
        nccl_tensor = torch.randn(shape, device=device, dtype=dtype)
        symm_tensor = symm_mem.empty(shape, device=device, dtype=dtype)
        symm_tensor.copy_(nccl_tensor)
        symm_mem.rendezvous(symm_tensor, group_name)

        for _ in range(args.warmup):
            x = nccl_tensor.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            funcol.all_reduce_inplace(x, "sum", dist.group.WORLD)
            symm_tensor.copy_(nccl_tensor)
            torch.ops.symm_mem.multimem_all_reduce_(symm_tensor, "sum", group_name)
            symm_tensor.copy_(nccl_tensor)
            torch.ops.symm_mem.two_shot_all_reduce_(symm_tensor, "sum", group_name)

        nccl_ms = _time_cuda(
            lambda: dist.all_reduce(nccl_tensor, op=dist.ReduceOp.SUM),
            args.iters,
        )
        funcol_ms = _time_cuda(
            lambda: funcol.all_reduce_inplace(nccl_tensor, "sum", dist.group.WORLD),
            args.iters,
        )
        multimem_ms = _time_cuda(
            lambda: torch.ops.symm_mem.multimem_all_reduce_(symm_tensor, "sum", group_name),
            args.iters,
        )
        two_shot_ms = _time_cuda(
            lambda: torch.ops.symm_mem.two_shot_all_reduce_(symm_tensor, "sum", group_name),
            args.iters,
        )
        results[str(batch)] = {
            "nccl_ms": nccl_ms,
            "funcol_ms": funcol_ms,
            "multimem_ms": multimem_ms,
            "two_shot_ms": two_shot_ms,
        }

    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
