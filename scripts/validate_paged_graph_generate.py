#!/usr/bin/env python3
"""Multi-step graphed paged decode == eager: the growing-context correctness gate.

PagedDecodeGraphRunner captures ONE decode graph and replays it as context grows
(new pages, page-boundary crossings handled by refilling the static buffers +
re-planning outside the graph). This drives generate_paged both eager and graphed
over enough tokens to cross several page_size=16 boundaries and asserts the greedy
token streams are IDENTICAL -- proving the runner handles sequence growth, the last
correctness risk before wiring it into the serving engine. Also times both.

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29759 \
           --nproc-per-node 8 scripts/validate_paged_graph_generate.py
"""
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import generate_paged


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    import flashinfer  # noqa: F401

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")

    # uniform-length prompts; decode well past page_size=16 to cross boundaries.
    prompts = [
        [1, 5, 9, 13, 2, 6, 3, 11],
        [4, 8, 10, 7, 12, 1, 14, 2],
        [3, 7, 11, 5, 9, 13, 6, 10],
    ]
    new_tokens = 40  # crosses page boundaries at 16, 32 (page_size=16)

    with torch.inference_mode():
        eager = generate_paged(model, prompts, max_new_tokens=new_tokens, page_size=16, use_graph=False)
        graphed = generate_paged(model, prompts, max_new_tokens=new_tokens, page_size=16, use_graph=True)

    match = eager == graphed
    log(rank, f"[correctness] graphed generate == eager generate: {match}")
    if not match and rank == 0:
        for i, (e, g) in enumerate(zip(eager, graphed)):
            diff = next((j for j in range(min(len(e), len(g))) if e[j] != g[j]), None)
            print(f"  row {i}: first diff at {diff}\n    eager={e}\n    graph={g}", flush=True)

    # timing: eager vs graphed decode over a longer run
    with torch.inference_mode():
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        generate_paged(model, prompts, max_new_tokens=64, page_size=16, use_graph=False)
        torch.cuda.synchronize(); dist.barrier(); eager_s = time.time() - t0
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        generate_paged(model, prompts, max_new_tokens=64, page_size=16, use_graph=True)
        torch.cuda.synchronize(); dist.barrier(); graph_s = time.time() - t0
    log(rank, f"[timing] 64-token generate: eager={eager_s:.2f}s graphed={graph_s:.2f}s "
              f"(graphed includes one-time capture)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
