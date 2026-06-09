#!/usr/bin/env python3
"""Compare NCCL ring vs symm-mem multimem (multicast) vs one_shot/two_shot
(P2P, NO multicast) all_reduce on 8xH100, at DECODE sizes (small, latency-bound
-- where the TPOT win lives) and prefill sizes. Verifies correctness vs NCCL.

The multicast path (multimem_all_reduce_) is what deadlocked the bench (per-rank
fallback divergence on init_multicast_for_block). one_shot/two_shot use the same
rendezvous'd buffer but reduce via P2P -> should be robust where multimem fails.

  torchrun --standalone --nproc-per-node 8 scripts/bench_allreduce_oneshot.py
"""

import os
import time

import torch
import torch.distributed as dist


def bench(fn, iters=100, warmup=20):
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
    # decode sizes (rows = concurrent decode batch) then prefill sizes (tokens).
    for tokens in (16, 32, 64, 128, 256, 1024, 6144):
        # reference input (same on all ranks via broadcast of a per-rank-seeded
        # tensor sum -> deterministic expected = sum over ranks).
        x = torch.full((tokens, hidden), float(rank + 1), device=dev, dtype=torch.bfloat16)
        expected = float(sum(r + 1 for r in range(dist.get_world_size())))

        def nccl():
            t = x.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t

        nccl_t = nccl()
        nccl_us = bench(lambda: dist.all_reduce(x.clone(), op=dist.ReduceOp.SUM))

        results = {}
        for label, opname, inplace in [
            ("multimem", "multimem_all_reduce_", True),
            ("one_shot", "one_shot_all_reduce", False),
            ("two_shot", "two_shot_all_reduce_", True),
        ]:
            try:
                buf = symm_mem.empty(tokens, hidden, device=dev, dtype=torch.bfloat16)
                symm_mem.rendezvous(buf, group_name)
                op = getattr(torch.ops.symm_mem, opname)
                buf.copy_(x)
                # correctness: run once, compare to expected sum
                if inplace:
                    op(buf, "sum", group_name)
                    out = buf
                else:
                    out = op(buf, "sum", group_name)
                ok = torch.allclose(out.float(), torch.full_like(out, expected).float(), rtol=2e-2, atol=2e-2)

                def run(op=op, buf=buf, x=x, inplace=inplace):
                    if inplace:
                        buf.copy_(x)
                        op(buf, "sum", group_name)
                    else:
                        op(buf, "sum", group_name)

                us = bench(run)
                results[label] = (us, ok)
            except Exception as e:
                results[label] = (None, f"FAIL: {str(e)[:60]}")

        if rank == 0:
            mb = tokens * hidden * 2 / 1e6
            line = f"tokens={tokens:6d} ({mb:6.1f}MB)  nccl={nccl_us:7.1f}us"
            for label in ("multimem", "one_shot", "two_shot"):
                us, ok = results[label]
                if us is not None:
                    line += f"  {label}={us:6.1f}us(x{nccl_us/us:.2f},ok={ok})"
                else:
                    line += f"  {label}={ok}"
            print(line, flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
