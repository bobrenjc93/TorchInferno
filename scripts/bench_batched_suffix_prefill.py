#!/usr/bin/env python3
"""DECISIVE experiment: is BATCHED suffix-prefill over a shared cached prefix faster
than full prefill, at few_shot's shape? few_shot = a FIXED ~P-token shared system
prompt (the 5 examples) + a tiny ~S-token varied question, x N concurrent. vllm
prefix-caches the shared part (prefills ~S/req); we re-prefill the full P+S.

Prior reuse attempts (common-prefix, COW, FI_REUSE) all REGRESSED -- but via OVERHEAD
(eager path / INDIVIDUAL per-request suffix prefills), never BATCHED. This times the
batched version against full prefill to decide if the (big, TP-collective) batched-reuse
serving build is worth it:
  FULL  : forward_prefill_paged of N requests x (P+S) tokens (what we do today)
  REUSE : prefill the shared P-prefix ONCE, share_prefix into N reqs, then ONE batched
          forward of N x S suffix tokens (each attending to the shared prefix)

RESULT (real 70B TP8, P=96 S=16 N=64): REUSE only 1.20x faster (FULL 302ms vs REUSE
253ms) -- NOT the ~7x the token-count (7168 vs 1024) predicts. WHY: the paged
forward_prefill_paged path is LAUNCH-BOUND (~250ms floor, 80 layers of un-amortized eager
kernels), so cutting tokens barely helps. This is the SAME reason common-prefix/COW/FI_REUSE
all regressed: reuse runs on the slow paged/eager path while our FULL prefill runs on the
FAST dense graph-FI path (compute-bound, ~1% launch, [[prefill-is-gemm-bound]]). So reuse
can't beat our own fast full-prefill unless done ON the graph-FI path -- which has no paged
shared-prefix support (a deep build). CONCLUSION: batched reuse via the available (paged)
path is NOT the few_shot lever; the launch floor erases the token savings.

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29803 \
           --nproc-per-node 8 scripts/bench_batched_suffix_prefill.py
"""
import math
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import _plan_prefill

PAGE = 16
P = 96   # shared prefix length (page-aligned) -- few_shot system prompt ~5 examples
S = 16   # per-request suffix length -- the short varied question
N = 64   # concurrent requests


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def gbench(fn, it=20, wu=5):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def main():
    import flashinfer  # noqa: F401

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, f"model loaded (P={P} shared, S={S} suffix, N={N} reqs)")
    nqo = model.layers[0].local_attention_heads
    nkv = model.layers[0].local_key_value_heads
    hd = model.config.head_dim
    pages_per_full = math.ceil((P + S) / PAGE)
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=N * pages_per_full + 2 * N + 32,
        page_size=PAGE, num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    g = torch.Generator(device="cpu").manual_seed(7)
    full_ids = torch.randint(0, model.config.vocab_size, (N, P + S), generator=g).to(dev)

    with torch.inference_mode():
        # ---- FULL: prefill N requests x (P+S) tokens (today's path) ----
        def full():
            for i in range(N):
                cache.free(f"f{i}")
            for i in range(N):
                cache.reserve(f"f{i}", P + S)
                cache._sequences[f"f{i}"].length = P + S
            pw = _plan_prefill(flashinfer, cache, [f"f{i}" for i in range(N)],
                               [P + S] * N, nqo, nkv, hd, PAGE)
            model.forward_prefill_paged(
                full_ids, cache,
                request_ids=[f"f{i}" for i in range(N)], prefill_wrapper=pw)

        # ---- REUSE: shared P-prefix prefilled ONCE, then batched N x S suffix ----
        cache.reserve("shared", P)
        cache._sequences["shared"].length = P
        pw0 = _plan_prefill(flashinfer, cache, ["shared"], [P], nqo, nkv, hd, PAGE)
        model.forward_prefill_paged(full_ids[:1, :P], cache, request_ids=["shared"], prefill_wrapper=pw0)
        suffix_ids = full_ids[:, P:].contiguous()

        def reuse():
            for i in range(N):
                cache.free(f"r{i}")
            for i in range(N):
                shared = cache.share_prefix("shared", f"r{i}", P)
                assert shared == P
                cache.reserve(f"r{i}", P + S)
                cache._sequences[f"r{i}"].length = P + S
            pw = _plan_prefill(flashinfer, cache, [f"r{i}" for i in range(N)],
                               [S] * N, nqo, nkv, hd, PAGE)
            model.forward_prefill_paged(
                suffix_ids, cache,
                request_ids=[f"r{i}" for i in range(N)], prefill_wrapper=pw, start_position=P)

        t_full = gbench(full)
        t_reuse = gbench(reuse)
        log(rank, f"[few_shot-shape] FULL prefill ({N}x{P+S}={N*(P+S)} tok) = {t_full:.2f} ms")
        log(rank, f"[few_shot-shape] REUSE batched-suffix ({N}x{S}={N*S} tok over shared {P}) = {t_reuse:.2f} ms")
        log(rank, f"   speedup = {t_full / t_reuse:.2f}x  (>1 => batched reuse is the few_shot TTFT lever)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
