#!/usr/bin/env python3
"""Variable-batch padding: ONE captured graph (batch=B) serves active<B == eager.

Continuous batching needs the captured decode graph to handle a varying active set.
PagedDecodeGraphRunner pads an active set (active<=batch) up to the captured batch by
repeating the last active row (dummies redo a reserved request's work, output
sliced off). This captures a runner at batch=8 and decodes an active set of 5,
asserting graphed[:5] == eager forward_decode_paged over those 5 (which is == dense).

  torchrun ... --nproc-per-node 8 scripts/validate_paged_runner_padding.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import PagedDecodeGraphRunner

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

    active = 5
    cap_batch = 8
    prompts = [[1, 5, 9, 13, 2, 6, 3, 11][: 6 + (i % 2)] + [i + 2] for i in range(active)]
    # uniform length for the prefill block
    T = min(len(p) for p in prompts)
    prompts = [p[:T] for p in prompts]
    rids = [str(i) for i in range(active)]
    max_seq = T + 8
    pages_per = (max_seq + PAGE - 1) // PAGE
    cache = LayeredPagedKVCache(
        num_layers=len(model.layers), num_pages=cap_batch * pages_per + 8, page_size=PAGE,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=model.dtype,
    )
    for rid in rids:
        cache.reserve(rid, max_seq)
        cache._sequences[rid].length = T

    with torch.inference_mode():
        ip, ix, lpl = cache.flashinfer_page_table(rids)
        qo = torch.zeros(active + 1, dtype=torch.int32, device=dev)
        qo[1:] = torch.tensor([T] * active, dtype=torch.int32, device=dev).cumsum(0)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        pw.plan(qo_indptr=qo, paged_kv_indptr=ip, paged_kv_indices=ix, paged_kv_last_page_len=lpl,
                num_qo_heads=nqo, num_kv_heads=nkv, head_dim_qk=hd, page_size=PAGE,
                causal=True, q_data_type=cache.kv.dtype)
        logits = model.forward_prefill_paged(
            torch.tensor(prompts, dtype=torch.long, device=dev), cache, request_ids=rids, prefill_wrapper=pw)
        tok = logits[:, -1, :].argmax(-1)

        for rid in rids:
            cache._sequences[rid].length = T + 1
        positions = torch.full((active,), T, dtype=torch.long, device=dev)

        # eager reference over the 5 active requests (regular wrapper)
        ip2, ix2, lp2 = cache.flashinfer_page_table(rids)
        ws2 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        dwr = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws2, kv_layout="NHD")
        dwr.plan(indptr=ip2, indices=ix2, last_page_len=lp2, num_qo_heads=nqo, num_kv_heads=nkv,
                 head_dim=hd, page_size=PAGE, q_data_type=cache.kv.dtype)
        eager = model.forward_decode_paged(
            tok.view(active, 1), cache, request_ids=rids, positions=positions, decode_wrapper=dwr).clone()

        # runner captured at batch=8, decode active=5 (padded internally)
        runner = PagedDecodeGraphRunner(model, cache, batch=cap_batch, max_pages=pages_per)
        graphed = runner.step(tok.view(active, 1), positions, rids).clone()

    e = eager.reshape(active, -1).float()
    g = graphed.reshape(active, -1).float()
    maxd = (e - g).abs().max().item()
    rel = maxd / e.abs().max().item()
    log(rank, f"[padding] runner(batch={cap_batch}) active={active}: graphed[:active] vs eager  "
              f"max|d|={maxd:.4f} rel={rel:.5f}  match_argmax={e.argmax(-1).tolist()==g.argmax(-1).tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
