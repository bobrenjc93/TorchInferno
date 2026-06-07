#!/usr/bin/env python3
"""Is small-page (paged) decode attention more efficient than giant-page (dense) at
the multi_turn regime (high batch x long context)?

The dense serving cache uses page_size = max_seq (ONE giant page per row); true
paging uses page_size=16. The benchmark regression showed dense decode TPOT blows up
with batch x context (multi_turn 68ms@48rows -> 134ms@73rows). This isolates JUST the
FlashInfer decode-attention kernel (dw.run) at per-rank 70B dims (nqo=8, nkv=1,
hd=128) -- no model load -- comparing page_size=16 vs page_size=max_seq across the
(batch, context) grid. If paged-16 is faster at high batch x long ctx, paging fixes
multi_turn TPOT via kernel EFFICIENCY (not just concurrency). If equal, paging only
helps throughput/TTFT, not TPOT.

  python scripts/bench_decode_attn_pagesize.py   # single GPU
"""
import time

import torch

NQO, NKV, HD = 8, 1, 128
DT = torch.bfloat16


def bench_run(dw, q, kv, iters=50):
    for _ in range(10):
        dw.run(q, kv)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        dw.run(q, kv)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # us


def main():
    import flashinfer

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"per-rank dims nqo={NQO} nkv={NKV} hd={HD}; decode-attention us/call (1 layer):", flush=True)
    print(f"{'batch':>6} {'ctx':>6} {'paged16':>9} {'dense':>9} {'ratio':>7}", flush=True)

    for batch in (48, 64, 96, 128):
        for ctx in (1024, 2048, 4096):
            q = torch.randn(batch, NQO, HD, device=dev, dtype=DT)

            # paged-16: kv pool [num_pages, 2, 16, nkv, hd]
            PAGE = 16
            ppr = (ctx + PAGE - 1) // PAGE
            npages = batch * ppr
            kv16 = torch.randn(npages, 2, PAGE, NKV, HD, device=dev, dtype=DT)
            indptr = torch.arange(0, (batch + 1) * ppr, ppr, dtype=torch.int32, device=dev)
            indices = torch.arange(npages, dtype=torch.int32, device=dev)
            last = torch.full((batch,), ctx - (ppr - 1) * PAGE, dtype=torch.int32, device=dev)
            ws1 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
            dw16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws1, kv_layout="NHD")
            dw16.plan(indptr=indptr, indices=indices, last_page_len=last, num_qo_heads=NQO,
                      num_kv_heads=NKV, head_dim=HD, page_size=PAGE, q_data_type=DT)
            t16 = bench_run(dw16, q, kv16)

            # dense: 1 giant page per row, page_size=max_seq
            mx = ctx
            kvd = torch.randn(batch, 2, mx, NKV, HD, device=dev, dtype=DT)
            indptrd = torch.arange(batch + 1, dtype=torch.int32, device=dev)
            indicesd = torch.arange(batch, dtype=torch.int32, device=dev)
            lastd = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
            ws2 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
            dwd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws2, kv_layout="NHD")
            dwd.plan(indptr=indptrd, indices=indicesd, last_page_len=lastd, num_qo_heads=NQO,
                     num_kv_heads=NKV, head_dim=HD, page_size=mx, q_data_type=DT)
            td = bench_run(dwd, q, kvd)

            print(f"{batch:>6} {ctx:>6} {t16:>8.1f} {td:>8.1f} {t16/td:>7.2f}", flush=True)
            del kv16, kvd, ws1, ws2
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
