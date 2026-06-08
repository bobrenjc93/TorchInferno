#!/usr/bin/env python3
"""Validate COW suffix-prefill: prefilling ONLY the suffix over a shared/cached prefix
(forward_prefill_paged start_position>0) gives the SAME logits as prefilling the whole
prompt. This is the model-side correctness gate for zero-copy prefix caching (the
multi_turn TTFT lever -- turn k shares turns 1..k-1's pages, prefills only the new turn).

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29791 \
           --nproc-per-node 8 scripts/validate_paged_suffix_prefill.py
"""
import math
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import _greedy_tokens, _plan_prefill

PAGE = 16
L = 96      # full prompt length (page-aligned)
P = 48      # shared prefix length (page-aligned) -> suffix = 48


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

    pages_per = math.ceil(L / PAGE)
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=8 * pages_per + 16, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    g = torch.Generator(device="cpu").manual_seed(99)
    prompt = torch.randint(0, model.config.vocab_size, (1, L), generator=g).to(dev)

    with torch.inference_mode():
        # (A) FULL prefill of the whole prompt -> reference last-token logits.
        cache.reserve("ref", L); cache._sequences["ref"].length = L
        pw = _plan_prefill(flashinfer, cache, ["ref"], [L], nqo, nkv, hd, PAGE)
        ref = model.forward_prefill_paged(prompt, cache, request_ids=["ref"], prefill_wrapper=pw)
        ref_last = ref[:, -1, :].float().clone()

        # (B) SUFFIX path: prefill the prefix into "src", SHARE its pages with "dst",
        # then prefill ONLY the suffix at start_position=P over the shared prefix.
        cache.reserve("src", P); cache._sequences["src"].length = P
        pw2 = _plan_prefill(flashinfer, cache, ["src"], [P], nqo, nkv, hd, PAGE)
        model.forward_prefill_paged(prompt[:, :P], cache, request_ids=["src"], prefill_wrapper=pw2)

        shared = cache.share_prefix("src", "dst", P)
        assert shared == P, f"shared {shared} != {P}"
        cache.reserve("dst", L); cache._sequences["dst"].length = L  # add suffix pages
        suffix_ids = prompt[:, P:]
        pw3 = _plan_prefill(flashinfer, cache, ["dst"], [L - P], nqo, nkv, hd, PAGE)
        suf = model.forward_prefill_paged(
            suffix_ids, cache, request_ids=["dst"], prefill_wrapper=pw3, start_position=P,
        )
        suf_last = suf[:, -1, :].float()

        rel = (ref_last - suf_last).abs().max().item() / max(1e-9, ref_last.abs().max().item())
        tok_ref = int(_greedy_tokens(model, ref_last)[0])
        tok_suf = int(_greedy_tokens(model, suf_last)[0])
        log(rank, f"[suffix-prefill] last-tok rel={rel:.6f}  greedy ref={tok_ref} suffix={tok_suf} "
                  f"match={tok_ref == tok_suf}  (L={L}, P={P})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
