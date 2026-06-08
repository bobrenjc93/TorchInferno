#!/usr/bin/env python3
"""Validate the PagedEngine prefill-graph integration: graph ON == graph OFF, token
for token, across VARIED prompt lengths (exercises the per-(batch,T) runner cache +
LRU), on the real 70B TP8. Also reports prefill wall time graph-off vs graph-on.

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29781 \
           --nproc-per-node 8 scripts/validate_paged_prefill_graph_engine.py
"""
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def run_engine(model, reqs, *, use_prefill_graph):
    eng = PagedEngine(model, page_size=16, max_active=16, max_seq=2048, use_graph=True)
    eng._use_prefill_graph = use_prefill_graph
    for i, (prompt, mx) in enumerate(reqs):
        eng.submit(f"r{i}", prompt, mx, eos_token_id=None)
    out: dict[str, list[int]] = {}
    t0 = time.time()
    while eng.has_work():
        for ext, tok, fin in eng.step():
            out.setdefault(ext, []).append(tok)
    torch.cuda.synchronize()
    wall = time.time() - t0
    return out, wall


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")

    # varied prompt lengths (distinct (batch,T) shapes) + modest outputs.
    g = torch.Generator(device="cpu").manual_seed(7)
    lengths = [128, 256, 256, 384, 512, 128, 256, 384]
    reqs = []
    for L in lengths:
        prompt = torch.randint(0, model.config.vocab_size, (L,), generator=g).tolist()
        reqs.append((prompt, 24))

    with torch.inference_mode():
        eager_out, eager_wall = run_engine(model, reqs, use_prefill_graph=False)
        graph_out, graph_wall = run_engine(model, reqs, use_prefill_graph=True)

    # token-identical check
    mismatches = 0
    for k in eager_out:
        if eager_out[k] != graph_out.get(k):
            mismatches += 1
    log(rank, f"[correctness] requests={len(eager_out)} mismatches={mismatches} "
              f"(token-identical graph-on vs graph-off)")
    if rank == 0 and mismatches:
        for k in list(eager_out)[:3]:
            log(rank, f"  {k}: eager={eager_out[k][:8]} graph={graph_out.get(k, [])[:8]}")
    log(rank, f"[timing] full-run wall: graph-off={eager_wall*1000:.0f}ms  "
              f"graph-on={graph_wall*1000:.0f}ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
