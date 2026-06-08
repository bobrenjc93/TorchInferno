#!/usr/bin/env python3
"""Capture a CUDA graph around forward_prefill_paged and prove it is FAST + CORRECT.

The paged prefill ran EAGER -- ~245ms/call launch-overhead bound on TP8 (80 layers x
per-layer allreduce, not graph-amortized) = the multi_turn TTFT killer (a single
1500-tok paged prefill = 850ms despite only ~27ms of compute). This captures
forward_prefill_paged (block_table path -- on-device slots, no host sync) with a
CUDAGraph BatchPrefillWithPagedKVCacheWrapper at a fixed (batch, T) bucket and checks:
  (1) graphed replay last-token logits == eager forward_prefill_paged (correctness),
  (2) graphed step time << eager step time (the whole point).

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29761 \
           --nproc-per-node 8 scripts/bench_paged_prefill_graph.py
"""
import math
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import PagedPrefillGraphRunner, _plan_prefill

PAGE = 16
BATCH = int(os.environ.get("PG_BATCH", "8"))
T = int(os.environ.get("PG_T", "512"))


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    import flashinfer

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")
    nqo = model.layers[0].local_attention_heads
    nkv = model.layers[0].local_key_value_heads
    hd = model.config.head_dim

    pages_per = math.ceil(T / PAGE)
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=BATCH * pages_per + 16, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    rids = [str(i) for i in range(BATCH)]
    for rid in rids:
        cache.reserve(rid, T)
        cache._sequences[rid].length = T

    # identical random prompt on every rank (broadcast from rank 0 so the TP shards
    # see the same tokens -- a fresh [BATCH, T] prefill block).
    g = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, T), generator=g).to(dev)

    with torch.inference_mode():
        # eager reference (request_ids / slot_mapping path)
        pw = _plan_prefill(flashinfer, cache, rids, [T] * BATCH, nqo, nkv, hd, PAGE)
        eager = model.forward_prefill_paged(
            input_ids, cache, request_ids=rids, prefill_wrapper=pw,
        ).clone()

        # graphed (block_table / on-device-slots path)
        runner = PagedPrefillGraphRunner(model, cache, batch=BATCH, T=T)
        graphed = runner.step(input_ids, rids).clone()

        # correctness: compare LAST-token logits per request (this rank's vocab shard)
        e = eager[:, -1, :].float()
        gv = graphed[:, -1, :].float()
        maxd = (e - gv).abs().max().item()
        rel = maxd / max(1e-9, e.abs().max().item())
        # also check every position (full prefill block), not just the last token
        ef = eager.reshape(-1, eager.shape[-1]).float()
        gf = graphed.reshape(-1, graphed.shape[-1]).float()
        rel_all = (ef - gf).abs().max().item() / max(1e-9, ef.abs().max().item())
        log(rank, f"[correctness] graphed vs eager paged prefill: "
                  f"last-tok rel={rel:.6f}  all-pos rel={rel_all:.6f}  (BATCH={BATCH}, T={T})")

        # timing: eager vs graphed
        N = 20
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(N):
            pw = _plan_prefill(flashinfer, cache, rids, [T] * BATCH, nqo, nkv, hd, PAGE)
            model.forward_prefill_paged(input_ids, cache, request_ids=rids, prefill_wrapper=pw)
        torch.cuda.synchronize(); dist.barrier()
        eager_ms = 1000 * (time.time() - t0) / N
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(N):
            runner.step(input_ids, rids)
        torch.cuda.synchronize(); dist.barrier()
        graph_ms = 1000 * (time.time() - t0) / N
        log(rank, f"[timing] eager={eager_ms:.1f} ms  graphed={graph_ms:.1f} ms  "
                  f"speedup={eager_ms/graph_ms:.1f}x  (BATCH={BATCH}, T={T})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
