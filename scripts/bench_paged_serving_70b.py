#!/usr/bin/env python3
"""Real-70B validation of the FlashInfer-paged model forward at scale.

Loads the actual Llama-3.1-70B TP8 model and exercises forward_prefill_paged +
forward_decode_paged through a LayeredPagedKVCache: (1) a correctness sanity vs a
dense forward for one sequence, and (2) decode throughput at increasing
concurrency (the concurrency win that paging unlocks vs the dense 48-row cap).

  torchrun --standalone --nproc-per-node 8 scripts/bench_paged_serving_70b.py
"""
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache

PAGE = 16
PROMPT = 24
DECODE_STEPS = 32


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def build_cache(model, num_seqs, max_seq, dev):
    layer0 = model.layers[0]
    pages_per = (max_seq + PAGE - 1) // PAGE
    return LayeredPagedKVCache(
        num_layers=len(model.layers),
        num_pages=num_seqs * pages_per + 16,
        page_size=PAGE,
        num_key_value_heads=layer0.local_key_value_heads,
        head_dim=model.config.head_dim,
        device=dev,
        dtype=model.dtype,
    )


def plan_decode(flashinfer, cache, rids, nqo, nkv, hd, dev):
    indptr, indices, lpl = cache.flashinfer_page_table(rids)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    dw.plan(indptr=indptr, indices=indices, last_page_len=lpl, num_qo_heads=nqo,
            num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
    return dw


def main():
    import flashinfer

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    t0 = time.time()
    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, f"model loaded in {time.time()-t0:.0f}s")
    nqo = model.layers[0].local_attention_heads
    nkv = model.layers[0].local_key_value_heads
    hd = model.config.head_dim
    vocab = model.config.vocab_size

    with torch.inference_mode():
        # (1) correctness sanity vs dense for one sequence (decode of the last token)
        seq = [1, 5, 9, 13, 2, 6, 3, 11, 8, 4]
        ref_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        ref_logits, _ = model.forward(
            torch.tensor([seq], dtype=torch.long, device=dev), cache=ref_cache, use_cache=True
        )
        ref = ref_logits[0, -1, :].float()
        P = len(seq) - 1
        pc = build_cache(model, 1, 64, dev)
        pc.reserve("r", P + 1)
        pc._sequences["r"].length = P + 1
        # copy prefix KV from the dense prefill of seq[:-1]
        pre = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        model.forward(torch.tensor([seq[:-1]], dtype=torch.long, device=dev), cache=pre, use_cache=True)
        pre_slots = pc.slot_mapping(["r"] * P, list(range(P)))
        for L in range(len(model.layers)):
            dk = pre.layers[L].keys[0, :, :P, :].permute(1, 0, 2).contiguous()
            dv = pre.layers[L].values[0, :, :P, :].permute(1, 0, 2).contiguous()
            pc.scatter_write(L, pre_slots, dk, dv)
        dw = plan_decode(flashinfer, pc, ["r"], nqo, nkv, hd, dev)
        out = model.forward_decode_paged(
            torch.tensor([[seq[-1]]], dtype=torch.long, device=dev), pc,
            request_ids=["r"], positions=torch.tensor([P], device=dev), decode_wrapper=dw,
        )
        # dense forward() returns GATHERED full-vocab logits; forward_decode_paged
        # returns this rank's vocab SHARD (lm_head is column-parallel). Compare the
        # paged shard to the matching slice of the dense full-vocab logits.
        got = out.reshape(-1, out.shape[-1])[0].float()
        lv = got.shape[-1]
        ref_shard = ref[rank * lv:(rank + 1) * lv]
        maxdiff = (got - ref_shard).abs().max().item()
        rel = maxdiff / ref_shard.abs().max().item()
        log(rank, f"[correctness] paged vs dense decode logits: max|d|={maxdiff:.3f} rel={rel:.4f}")

    # (2) decode throughput at increasing concurrency
    log(rank, "[throughput] decode tokens/sec at concurrency (paged):")
    with torch.inference_mode():
        for N in (48, 128, 256, 512):
            max_seq = PROMPT + DECODE_STEPS + 4
            cache = build_cache(model, N, max_seq, dev)
            rids = [str(i) for i in range(N)]
            # prefill all N (uniform short prompt)
            for rid in rids:
                cache.reserve(rid, max_seq)  # reserve all decode positions upfront
                cache._sequences[rid].length = PROMPT
            # seed prefix KV cheaply (random) so decode has something to attend; we
            # measure decode STEP throughput, not prefill, so contents are irrelevant.
            seed_slots = cache.slot_mapping([r for r in rids for _ in range(PROMPT)],
                                            [p for _ in rids for p in range(PROMPT)])
            for L in range(len(model.layers)):
                k = torch.randn(N * PROMPT, nkv, hd, device=dev, dtype=model.dtype)
                cache.scatter_write(L, seed_slots, k, k.clone())
            tok = torch.randint(0, vocab, (N, 1), device=dev)
            # warmup + timed decode loop
            for step in range(DECODE_STEPS):
                pos = torch.full((N,), PROMPT + step, device=dev)
                for rid in rids:
                    cache._sequences[rid].length = PROMPT + step + 1
                dw = plan_decode(flashinfer, cache, rids, nqo, nkv, hd, dev)
                if step == 3:
                    torch.cuda.synchronize(); dist.barrier(); t_start = time.time(); steps_timed = 0
                _ = model.forward_decode_paged(tok, cache, request_ids=rids, positions=pos, decode_wrapper=dw)
                if step >= 3:
                    steps_timed += 1
            torch.cuda.synchronize(); dist.barrier()
            dt = time.time() - t_start
            tps = N * steps_timed / dt
            log(rank, f"  N={N:4d}: {tps:8.0f} tok/s  ({1000*dt/steps_timed:.1f} ms/step, {N} rows)")
            del cache
            torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
