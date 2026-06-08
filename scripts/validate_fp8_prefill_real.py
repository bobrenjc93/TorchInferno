#!/usr/bin/env python3
"""End-to-end validation of the REAL fp8 prefill wiring (TORCHINFERNO_FP8_PREFILL):
greedy generation must match bf16. Uses a >256-token prompt so the M-gate engages on
the paged prefill (M=prompt>256 -> fp8; decode M=1 -> bf16/marlin). This exercises the
actual _scaled_mm path + the fused-quant kernel + the symm-buffer copy, unlike the
fake-quant probe.

  FP8_PREFILL=0/1 toggles. Run both, diff token lines:
  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29799 \
           --nproc-per-node 8 scripts/validate_fp8_prefill_real.py
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
    fp8 = os.environ.get("FP8_PREFILL", "0")
    os.environ["TORCHINFERNO_FP8_PREFILL"] = fp8

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, f"model loaded (TORCHINFERNO_FP8_PREFILL={fp8})")

    g = torch.Generator(device="cpu").manual_seed(13)
    prompt = torch.randint(0, model.config.vocab_size, (384,), generator=g).tolist()  # >256 -> fp8 prefill
    with torch.inference_mode():
        eng = PagedEngine(model, page_size=16, max_active=4, max_seq=768, use_graph=True)
        eng.submit("t", prompt, 48, eos_token_id=None)
        toks = []
        while eng.has_work():
            for e_id, tok, fin in eng.step():
                if e_id == "t":
                    toks.append(tok)
    log(rank, f"[fp8={fp8}] {len(toks)} tokens:")
    log(rank, " ".join(str(t) for t in toks))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
