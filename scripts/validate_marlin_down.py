#!/usr/bin/env python3
"""Validate marlin int4 down_proj decode: greedy generation with MARLIN_INT4_DOWN
must agree with the down-OFF baseline (gate_up marlin ON in both) for enough tokens
to confirm int4 down_proj does not corrupt decode. down_proj feeds the residual
directly, so its int4 error accumulates across 80 layers -- this is the gate that
decides whether down marlin is safe to ship (like the gate_up 5/5 greedy check).

Run BOTH (the harness prints the token sequence; diff the two):
  TORCHINFERNO_MARLIN_INT4_DOWN=0 torchrun --nnodes 1 --node-rank 0 \
    --master-addr 127.0.0.1 --master-port 29796 --nproc-per-node 8 \
    scripts/validate_marlin_down.py
  TORCHINFERNO_MARLIN_INT4_DOWN=1 torchrun --nnodes 1 --node-rank 0 \
    --master-addr 127.0.0.1 --master-port 29796 --nproc-per-node 8 \
    scripts/validate_marlin_down.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))

    down = os.environ.get("TORCHINFERNO_MARLIN_INT4_DOWN", "1")
    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, f"model loaded (MARLIN_INT4_DOWN={down})")

    g = torch.Generator(device="cpu").manual_seed(7)
    prompt = torch.randint(0, model.config.vocab_size, (96,), generator=g).tolist()

    with torch.inference_mode():
        eng = PagedEngine(model, page_size=16, max_active=4, max_seq=512, use_graph=True)
        eng.submit("t", prompt, 48, eos_token_id=None)
        toks = []
        while eng.has_work():
            for e_id, tok, fin in eng.step():
                if e_id == "t":
                    toks.append(tok)
    log(rank, f"[down={down}] generated {len(toks)} tokens:")
    log(rank, " ".join(str(t) for t in toks))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
