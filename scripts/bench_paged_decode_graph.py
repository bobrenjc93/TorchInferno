#!/usr/bin/env python3
"""Capture a CUDA graph around forward_decode_paged and prove it is FAST + CORRECT.

The paged decode is context-flat (scripts/bench_decode_context_scaling.py) but ran
EAGER at ~146ms/step vs the dense GRAPHED ~21ms. The serving win needs paged decode
GRAPHED. This captures forward_decode_paged (block_table path -- fully on-device, no
host sync) with CUDAGraphBatchDecodeWithPagedKVCacheWrapper and checks:
  (1) graphed replay output == eager forward_decode_paged output (correctness),
  (2) graphed step time << eager step time (the whole point).

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29757 \
           --nproc-per-node 8 scripts/bench_paged_decode_graph.py
"""
import os
import time

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache

PAGE = 16
ROWS = 48
CTX = 512


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
    vocab = model.config.vocab_size

    max_seq = CTX + 8
    pages_per = (max_seq + PAGE - 1) // PAGE
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=ROWS * pages_per + 16, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    rids = [str(i) for i in range(ROWS)]
    for rid in rids:
        cache.reserve(rid, max_seq)
        cache._sequences[rid].length = CTX
    seed_slots = cache.slot_mapping([r for r in rids for _ in range(CTX)],
                                    [p for _ in rids for p in range(CTX)])
    for L in range(len(model.layers)):
        k = torch.randn(ROWS * CTX, nkv, hd, device=dev, dtype=model.dtype)
        cache.scatter_write(L, seed_slots, k, k.clone())

    with torch.inference_mode():
        # decode the token at position CTX for every row.
        pos_val = CTX
        for rid in rids:
            cache._sequences[rid].length = pos_val + 1
        s_ids = torch.randint(0, vocab, (ROWS, 1), device=dev)
        s_pos = torch.full((ROWS,), pos_val, dtype=torch.long, device=dev)
        s_bt = cache.block_table(rids, max_pages=pages_per)  # [ROWS, pages_per] static
        # CUDAGraph paged decode wrapper with static page-table buffers.
        npages_row = (pos_val + 1 + PAGE - 1) // PAGE
        total_pages = ROWS * npages_row
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        ind = torch.empty(ROWS + 1, dtype=torch.int32, device=dev)
        idx = torch.empty(total_pages, dtype=torch.int32, device=dev)
        lp = torch.empty(ROWS, dtype=torch.int32, device=dev)
        dw = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(ws, ind, idx, lp, kv_layout="NHD")

        def do_plan():
            indptr, indices, last = cache.flashinfer_page_table(rids)
            ind[: indptr.numel()].copy_(indptr)
            idx[: indices.numel()].copy_(indices)
            lp.copy_(last)
            dw.plan(indptr=ind[: indptr.numel()], indices=idx[: indices.numel()],
                    last_page_len=lp, num_qo_heads=nqo, num_kv_heads=nkv,
                    head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)

        # eager reference (request_ids path)
        do_plan()
        eager_out = model.forward_decode_paged(
            s_ids, cache, request_ids=rids, positions=s_pos, decode_wrapper=dw,
        ).clone()

        # warmup on a side stream, then capture (block_table path -- graphable)
        do_plan()
        model.forward_decode_paged(s_ids, cache, positions=s_pos, decode_wrapper=dw, block_table=s_bt)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        do_plan()
        with torch.cuda.stream(stream):
            model.forward_decode_paged(s_ids, cache, positions=s_pos, decode_wrapper=dw, block_table=s_bt)
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        do_plan()
        with torch.cuda.graph(graph, stream=stream):
            g_out = model.forward_decode_paged(s_ids, cache, positions=s_pos, decode_wrapper=dw, block_table=s_bt)

        # correctness: replay vs eager (same inputs) -- compare this rank's vocab shard
        do_plan()
        graph.replay()
        torch.cuda.synchronize()
        e = eager_out.reshape(-1, eager_out.shape[-1]).float()
        g = g_out.reshape(-1, g_out.shape[-1]).float()
        maxd = (e - g).abs().max().item()
        rel = maxd / e.abs().max().item()
        log(rank, f"[correctness] graphed vs eager paged decode: max|d|={maxd:.4f} rel={rel:.5f}")

        # timing: eager vs graphed
        N = 30
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(N):
            do_plan()
            model.forward_decode_paged(s_ids, cache, positions=s_pos, decode_wrapper=dw, block_table=s_bt)
        torch.cuda.synchronize(); dist.barrier()
        eager_ms = 1000 * (time.time() - t0) / N
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(N):
            do_plan()
            graph.replay()
        torch.cuda.synchronize(); dist.barrier()
        graph_ms = 1000 * (time.time() - t0) / N
        log(rank, f"[timing] eager={eager_ms:.1f} ms/step  graphed={graph_ms:.1f} ms/step  "
                  f"speedup={eager_ms/graph_ms:.1f}x  (ROWS={ROWS}, CTX={CTX})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
