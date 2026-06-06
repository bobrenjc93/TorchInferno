#!/usr/bin/env python3
"""Microbenchmark: NCCL ring all_reduce vs symm-mem multimem_all_reduce_ for
prefill-sized [tokens, hidden] bf16 tensors on 8xH100 NVLink.

Decides whether routing the FlashInfer prefill allreduce through symm-mem is
worth a (TP-risky) model change. Run:

  torchrun --standalone --nproc-per-node 8 scripts/bench_allreduce.py
"""

import os
import time

import torch
import torch.distributed as dist


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    group_name = dist.group.WORLD.group_name
    import torch.distributed._symmetric_memory as symm_mem

    hidden = 8192
    # Representative prefill token counts (batch*q) seen in the online batcher.
    for tokens in (1024, 3200, 6144, 12288, 30720):
        x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)

        def nccl():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

        nccl_us = bench(nccl)

        sm_us = None
        try:
            buf = symm_mem.empty(tokens, hidden, device=dev, dtype=torch.bfloat16)
            symm_mem.rendezvous(buf, group_name)
            buf.copy_(x)

            def sm():
                torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", group_name)

            sm_us = bench(sm)
        except Exception as e:
            if rank == 0:
                print(f"  symm-mem failed at tokens={tokens}: {e!r}", flush=True)

        if rank == 0:
            mb = tokens * hidden * 2 / 1e6
            if sm_us:
                print(f"tokens={tokens:6d} ({mb:6.1f}MB)  nccl_ring={nccl_us:7.1f}us  "
                      f"symm_mem={sm_us:7.1f}us  speedup={nccl_us/sm_us:.2f}x", flush=True)
            else:
                print(f"tokens={tokens:6d} ({mb:6.1f}MB)  nccl_ring={nccl_us:7.1f}us  symm_mem=FAILED", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
