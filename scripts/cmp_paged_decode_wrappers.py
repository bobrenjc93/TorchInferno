#!/usr/bin/env python3
"""Isolate: does CUDAGraph paged decode wrapper == regular paged decode wrapper?

Prefill real KV into a paged cache, then run ONE forward_decode_paged step with the
regular BatchDecodeWithPagedKVCacheWrapper vs the CUDAGraphBatchDecodeWithPagedKV...
wrapper (NO graph capture -- just .run()), comparing logits. The regular path is
validated == dense (test_generate_paged_matches_dense_greedy). If the CUDAGraph
wrapper diverges here, the bug is the wrapper usage (e.g. needs fixed-size padded
page tables), not graph capture.

  torchrun ... --nproc-per-node 8 scripts/cmp_paged_decode_wrappers.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache

PAGE = 16


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

    prompts = [[1, 5, 9, 13, 2, 6, 3, 11, 8, 4], [3, 7, 11, 4, 8, 6, 10, 2, 14, 1]]
    T = len(prompts[0])
    batch = len(prompts)
    rids = [str(i) for i in range(batch)]
    max_seq = T + 8
    pages_per = (max_seq + PAGE - 1) // PAGE
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=batch * pages_per + 8, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    for rid in rids:
        cache.reserve(rid, max_seq)
        cache._sequences[rid].length = T

    with torch.inference_mode():
        # real prefill so KV is meaningful
        indptr, indices, lpl = cache.flashinfer_page_table(rids)
        qo = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
        qo[1:] = torch.tensor([T] * batch, dtype=torch.int32, device=dev).cumsum(0)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        pw.plan(qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=indices,
                paged_kv_last_page_len=lpl, num_qo_heads=nqo, num_kv_heads=nkv,
                head_dim_qk=hd, page_size=PAGE, causal=True, q_data_type=cache.kv.dtype)
        logits = model.forward_prefill_paged(
            torch.tensor(prompts, dtype=torch.long, device=dev), cache,
            request_ids=rids, prefill_wrapper=pw)
        tok = logits[:, -1, :].argmax(-1)

        # one decode step at position T
        for rid in rids:
            cache._sequences[rid].length = T + 1
        positions = torch.full((batch,), T, dtype=torch.long, device=dev)

        # (A) regular wrapper
        ip, ix, lp = cache.flashinfer_page_table(rids)
        ws_a = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        dwa = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws_a, kv_layout="NHD")
        dwa.plan(indptr=ip, indices=ix, last_page_len=lp, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        out_a = model.forward_decode_paged(
            tok.view(batch, 1), cache, request_ids=rids, positions=positions, decode_wrapper=dwa).clone()

        # (B) CUDAGraph wrapper -- plain .run(), NO capture
        ws_b = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        ind = torch.empty(batch + 1, dtype=torch.int32, device=dev)
        idx = torch.empty(batch * pages_per, dtype=torch.int32, device=dev)
        lpb = torch.empty(batch, dtype=torch.int32, device=dev)
        dwb = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(ws_b, ind, idx, lpb, kv_layout="NHD")
        ip2, ix2, lp2 = cache.flashinfer_page_table(rids)
        dwb.plan(indptr=ip2, indices=ix2, last_page_len=lp2, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        out_b = model.forward_decode_paged(
            tok.view(batch, 1), cache, request_ids=rids, positions=positions, decode_wrapper=dwb).clone()

        # (C) CUDAGraph wrapper + BLOCK_TABLE path, no capture
        s_bt = cache.block_table(rids, max_pages=pages_per)
        ip3, ix3, lp3 = cache.flashinfer_page_table(rids)
        dwb.plan(indptr=ip3, indices=ix3, last_page_len=lp3, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        out_c = model.forward_decode_paged(
            tok.view(batch, 1), cache, positions=positions, decode_wrapper=dwb, block_table=s_bt).clone()

        # (D) CUDAGraph wrapper + block_table + CAPTURE+REPLAY
        s_ids = tok.view(batch, 1).clone()
        ip4, ix4, lp4 = cache.flashinfer_page_table(rids)
        dwb.plan(indptr=ip4, indices=ix4, last_page_len=lp4, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        model.forward_decode_paged(s_ids, cache, positions=positions, decode_wrapper=dwb, block_table=s_bt)
        torch.cuda.synchronize()
        st = torch.cuda.Stream(device=dev); st.wait_stream(torch.cuda.current_stream(dev))
        dwb.plan(indptr=ip4, indices=ix4, last_page_len=lp4, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        with torch.cuda.stream(st):
            model.forward_decode_paged(s_ids, cache, positions=positions, decode_wrapper=dwb, block_table=s_bt)
        torch.cuda.current_stream(dev).wait_stream(st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        dwb.plan(indptr=ip4, indices=ix4, last_page_len=lp4, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        with torch.cuda.graph(g, stream=st):
            out_d_buf = model.forward_decode_paged(s_ids, cache, positions=positions, decode_wrapper=dwb, block_table=s_bt)
        dwb.plan(indptr=ip4, indices=ix4, last_page_len=lp4, num_qo_heads=nqo,
                 num_kv_heads=nkv, head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        g.replay(); torch.cuda.synchronize()
        out_d = out_d_buf.clone()

    def cmp(name, out):
        a = out_a.reshape(-1, out_a.shape[-1]).float()
        x = out.reshape(-1, out.shape[-1]).float()
        maxd = (a - x).abs().max().item()
        rel = maxd / a.abs().max().item()
        log(rank, f"[{name}] vs A(regular,reqids): max|d|={maxd:.4f} rel={rel:.5f} "
                  f"argmax={x.argmax(-1).tolist()} (A={a.argmax(-1).tolist()})")

    cmp("B cudagraph+reqids,nocap", out_b)
    cmp("C cudagraph+blocktable,nocap", out_c)
    cmp("D cudagraph+blocktable,CAPTURE", out_d)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
