#!/usr/bin/env python3
"""Does FlashInfer paged decode ATTENTION stay cheap at high concurrency + long
context? This is the last viability unknown for the paged-KV lever: decode is
GEMM-bound (weight-bound, flat to 256 rows -- see bench_decode_batch_scaling) and
GEMMs are independent of context length, so if paged attention also stays a small,
sub-linear fraction as we pack many long-context rows (which paging enables and the
dense [batch, kv_heads, max_seq, head_dim] cache cannot), then high-concurrency
long-context serving is viable -> the queueing-bound multi_turn/long_output TPOT +
TTFT/throughput gaps.

Uses the validated LayeredPagedKVCache (NHD pages) + FlashInfer paged decode.

  PYTHONPATH=src python scripts/bench_paged_decode_concurrency.py
"""
import torch

from torchinferno.runtime.paged import LayeredPagedKVCache

NQO, NKV, HD = 8, 1, 128  # Llama3-70B TP8 per-GPU: 64/8 q heads, 8/8 kv heads
PAGE = 16
CTX = 2048  # long context (multi_turn-like)


def bench(fn, it=100, wu=20):
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
    return s.elapsed_time(e) / it * 1000  # us


def main():
    import flashinfer

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    pages_per_seq = (CTX + PAGE - 1) // PAGE
    print(f"ctx={CTX} page_size={PAGE} pages/seq={pages_per_seq}  (NQO={NQO} NKV={NKV} HD={HD})")
    print(f"{'rows':>5} {'attn_us':>9} {'us/row':>8} {'KV_GB':>7}  (per-GPU paged KV at this concurrency)")
    base = None
    for rows in (48, 128, 256, 512):
        num_pages = rows * pages_per_seq + 16
        cache = LayeredPagedKVCache(
            num_layers=1, num_pages=num_pages, page_size=PAGE,
            num_key_value_heads=NKV, head_dim=HD, device=dev, dtype=torch.bfloat16,
        )
        rids = [str(i) for i in range(rows)]
        for rid in rids:
            cache.extend(rid, CTX)
            cache.write_layer(
                0, rid,
                torch.randn(CTX, NKV, HD, device=dev, dtype=torch.bfloat16),
                torch.randn(CTX, NKV, HD, device=dev, dtype=torch.bfloat16),
                start=0,
            )
        indptr, indices, lpl = cache.flashinfer_page_table(rids)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        dw.plan(
            indptr=indptr, indices=indices, last_page_len=lpl,
            num_qo_heads=NQO, num_kv_heads=NKV, head_dim=HD, page_size=PAGE,
            q_data_type=torch.bfloat16,
        )
        q = torch.randn(rows, NQO, HD, device=dev, dtype=torch.bfloat16)
        paged = cache.layer_kv(0)
        t = bench(lambda: dw.run(q, paged))
        # KV bytes for 80 layers at this row count (NHD, k+v, bf16)
        kv_gb = num_pages * 2 * PAGE * NKV * HD * 2 * 80 / 1e9
        if base is None:
            base = t
        print(f"{rows:>5} {t:9.1f} {t/rows:8.3f} {kv_gb:7.2f}  ({t/base:.2f}x vs 48 rows)")


if __name__ == "__main__":
    main()
