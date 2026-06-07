#!/usr/bin/env python3
"""Does PAGED decode step time stay flat as context grows?

The benchmark few_shot/tree TPOT gap was traced to the dense serving decode cost
scaling with context length (real-70B: ~35ms @350ctx -> ~94ms @950ctx, all on the
graphed path). That is a dense-KV-layout inefficiency (the FlashInfer decode plan
uses page_size = max_seq, one giant page per row). This measures forward_decode_paged
(page_size=16) step time at FIXED rows over growing context: if it stays flat, true
paging fixes the context-scaling gap and justifies the serving-engine integration.

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29753 \
           --nproc-per-node 8 scripts/bench_decode_context_scaling.py
"""
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache

PAGE = 16
ROWS = 48
DECODE_STEPS = 24
CONTEXTS = (256, 512, 1024, 2048, 4096)


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


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

    log(rank, f"[paged decode] {ROWS} rows, page_size={PAGE}; step time vs context:")
    with torch.inference_mode():
        for ctx in CONTEXTS:
            max_seq = ctx + DECODE_STEPS + 4
            pages_per = (max_seq + PAGE - 1) // PAGE
            cache = LayeredPagedKVCache(
                num_layers=len(model.layers), num_pages=ROWS * pages_per + 16,
                page_size=PAGE, num_key_value_heads=nkv, head_dim=hd,
                device=dev, dtype=model.dtype,
            )
            rids = [str(i) for i in range(ROWS)]
            for rid in rids:
                cache.reserve(rid, max_seq)
                cache._sequences[rid].length = ctx
            # seed ctx tokens of (random) KV per row so decode attends a full context
            seed_slots = cache.slot_mapping([r for r in rids for _ in range(ctx)],
                                            [p for _ in rids for p in range(ctx)])
            for L in range(len(model.layers)):
                k = torch.randn(ROWS * ctx, nkv, hd, device=dev, dtype=model.dtype)
                cache.scatter_write(L, seed_slots, k, k.clone())
            tok = torch.randint(0, vocab, (ROWS, 1), device=dev)
            for step in range(DECODE_STEPS):
                pos = torch.full((ROWS,), ctx + step, device=dev)
                for rid in rids:
                    cache._sequences[rid].length = ctx + step + 1
                dw = plan_decode(flashinfer, cache, rids, nqo, nkv, hd, dev)
                if step == 3:
                    torch.cuda.synchronize(); dist.barrier(); t_start = time.time(); timed = 0
                _ = model.forward_decode_paged(tok, cache, request_ids=rids, positions=pos, decode_wrapper=dw)
                if step >= 3:
                    timed += 1
            torch.cuda.synchronize(); dist.barrier()
            ms = 1000 * (time.time() - t_start) / timed
            log(rank, f"  ctx={ctx:5d}: {ms:6.1f} ms/step  ({ROWS} rows)")
            del cache
            torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
