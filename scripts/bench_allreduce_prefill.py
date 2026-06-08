#!/usr/bin/env python3
"""Is the prefill allreduce a lever? symm_mem multimem vs NCCL at TP8 prefill shapes.

prefill = GEMMs 61% + allreduce 24% + norms 11% (prefill-is-gemm-bound). fp8 attacks
the GEMMs; this checks whether the 24% allreduce has headroom worth a custom kernel.

RESULT (real 8xH100): symm_mem multimem is ALREADY 1.64-2.37x faster than NCCL --
it IS the optimized custom allreduce (NVLink hardware multicast). So the allreduce is
NOT a lever: there is no easy 2x left there (we already beat the NCCL baseline by ~2x).
=> fp8 prefill (~1.4x on the GEMMs) is essentially the WHOLE available prefill-compute
lever; the residual TTFT gap vs vllm after fp8 is scheduling / kernel-quality, not
allreduce. (M=512: nccl 98us vs symm 46us = 2.14x; M=2048: 263 vs 160 = 1.64x.)

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29802 \
           --nproc-per-node 8 scripts/bench_allreduce_prefill.py
"""
import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    import torch.distributed._symmetric_memory as symm

    def gbench(fn, it=200):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / it * 1000

    gname = dist.group.WORLD.group_name
    for M in (256, 512, 1024, 2048):
        x = torch.randn(M, 8192, device=dev, dtype=torch.bfloat16)
        buf = symm.empty((M, 8192), device=dev, dtype=torch.bfloat16)
        symm.rendezvous(buf, gname)

        def nccl(x=x):
            y = x.clone()
            dist.all_reduce(y)
            return y

        def mm(buf=buf, x=x):
            buf.copy_(x)
            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", gname)
            return buf

        tn, tm = gbench(nccl), gbench(mm)
        if rank == 0:
            print(f"M={M:5d} [{M * 8192 * 2 / 1e6:.1f}MB]: nccl={tn:.1f}us  "
                  f"symm_multimem={tm:.1f}us  ratio={tn / tm:.2f}x")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
