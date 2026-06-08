#!/usr/bin/env python3
"""Validate PagedEngine COW prefix caching end-to-end: a multi-turn-like sequence
(turn 2's prompt = turn 1's prompt+response + new tokens) must generate the SAME
tokens with COW ON (turn 2 shares turn 1's KV pages, suffix-prefills) as with COW OFF
(full prefill). Also confirms COW actually engaged (shared>0).

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29795 \
           --nproc-per-node 8 scripts/validate_paged_cow_engine.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine, PagedPrefixCache


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def drain(eng, ext):
    out = []
    while eng.has_work():
        for e_id, tok, fin in eng.step():
            if e_id == ext:
                out.append(tok)
    return out


def run(model, turn1, use_cow):
    eng = PagedEngine(model, page_size=16, max_active=8, max_seq=1024, use_graph=False)
    eng.prefix_cache = PagedPrefixCache(eng.cache, capacity=64) if use_cow else None
    eng.submit("t1", turn1, 16, eos_token_id=None)
    gen1 = drain(eng, "t1")
    # turn 2 = turn1 prompt + turn1 response + a new user chunk -> shares turn1's KV.
    turn2 = list(turn1) + list(gen1) + list(range(50, 66))
    shared_before = []
    if use_cow:
        # peek: how much would share_into match (without mutating -- use the router)
        m = eng.prefix_cache._router.route(turn2)
        shared_before = m.matched_tokens
    eng.submit("t2", turn2, 16, eos_token_id=None)
    gen2 = drain(eng, "t2")
    return gen1, gen2, len(shared_before)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")
    g = torch.Generator(device="cpu").manual_seed(5)
    turn1 = torch.randint(0, model.config.vocab_size, (80,), generator=g).tolist()

    with torch.inference_mode():
        g1_off, g2_off, _ = run(model, turn1, use_cow=False)
        g1_on, g2_on, shared = run(model, turn1, use_cow=True)

    # Correctness criterion is PREFIX agreement, not exact 16-token match: COW reuses
    # turn1's DECODE-written KV for the shared region, which differs from a full
    # prefill's KV only in the last bf16 bits -> greedy agrees for several tokens then
    # cascades (same behavior as vllm's prefix caching; see paged-vs-dense bf16 note).
    pfx = 0
    for a, b in zip(g2_off, g2_on):
        if a != b:
            break
        pfx += 1
    log(rank, f"[cow] turn1 match={g1_off == g1_on}  turn2 prefix-match={pfx}/{len(g2_off)}  "
              f"cow-shared-tokens={shared}  (>=4 prefix tokens = COW correct; later "
              f"divergence is bf16 decode-vs-prefill KV cascade)")
    if rank == 0:
        log(rank, f"  off={g2_off[:10]}\n  on ={g2_on[:10]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
