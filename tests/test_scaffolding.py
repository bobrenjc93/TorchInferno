import operator
import os
import subprocess
import sys

import pytest
import torch

from torchinferno.graph import PassRegistry, replace_call_function_targets, trace_with_make_fx
from torchinferno.research import ExperimentResult, ResearchHarness
from torchinferno.runtime.cudagraphs import CUDAGraphPiece, PiecewiseCUDAGraphRunner
from torchinferno.runtime.fake_dist import FakeProcessWorld
from torchinferno.runtime.flex import causal_mask_mod, flex_attention_or_fallback
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.prefix import PrefixAwareRouter
from torchinferno.runtime.scheduler import (
    DisaggregatedPrefillDecodeSimulator,
    InferenceJob,
    PersistentBatchRequest,
    PersistentBatchScheduler,
    TokenBudgetModelStepState,
    TokenBudgetPlan,
    TokenBudgetRequest,
    TokenBudgetScheduler,
    TokenBudgetScheduledChunk,
    apply_token_budget_model_step_command,
    token_budget_model_step_command,
)


def test_fake_process_world_collectives() -> None:
    world = FakeProcessWorld(2, mesh_shape=(1, 2))
    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

    reduced = world.all_reduce(tensors, op="sum")
    gathered = world.all_gather(tensors)
    ranks = world.run(lambda group: (group.rank, group.coordinate()))

    torch.testing.assert_close(reduced[0], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(reduced[1], torch.tensor([4.0, 6.0]))
    assert len(gathered) == 2
    assert ranks[1].result == (1, (0, 1))


def test_paged_kv_cache_materializes_request_tokens() -> None:
    cache = PagedKVCache(
        num_pages=3,
        page_size=2,
        num_key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    values = keys + 100

    seq = cache.append("req", keys, values)
    actual_keys, actual_values = cache.materialize("req")

    assert seq.page_ids == [0, 1, 2]
    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)
    cache.free("req")
    assert cache.free_pages == (0, 1, 2)


def test_paged_kv_cache_flashinfer_page_table() -> None:
    cache = PagedKVCache(
        num_pages=8,
        page_size=4,
        num_key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # req_a: 5 tokens -> 2 pages (last page holds 1 token);
    # req_b: 8 tokens -> 2 pages (last page full = page_size);
    # req_c: 3 tokens -> 1 page (last page holds 3).
    cache.append("a", torch.zeros(1, 5, 2), torch.zeros(1, 5, 2))
    cache.append("b", torch.zeros(1, 8, 2), torch.zeros(1, 8, 2))
    cache.append("c", torch.zeros(1, 3, 2), torch.zeros(1, 3, 2))

    indptr, indices, last_page_len = cache.flashinfer_page_table(["a", "b", "c"])

    # CSR page counts: a=2, b=2, c=1 -> indptr cumulative.
    torch.testing.assert_close(indptr, torch.tensor([0, 2, 4, 5], dtype=torch.int32))
    # indices concatenate each request's actual page ids in order.
    expected_pages = cache.sequence("a").page_ids[:2] + cache.sequence("b").page_ids[:2] + cache.sequence("c").page_ids[:1]
    torch.testing.assert_close(indices, torch.tensor(expected_pages, dtype=torch.int32))
    # last-page valid token counts: a=1, b=4 (full), c=3.
    torch.testing.assert_close(last_page_len, torch.tensor([1, 4, 3], dtype=torch.int32))


def test_layered_paged_kv_cache_shared_block_table_and_writes() -> None:
    from torchinferno.runtime.paged import LayeredPagedKVCache

    cache = LayeredPagedKVCache(
        num_layers=3,
        num_pages=8,
        page_size=4,
        num_key_value_heads=2,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Reserve 6 tokens for req "a" -> 2 pages (shared by all 3 layers).
    start = cache.extend("a", 6)
    assert start == 0
    assert cache.sequence_length("a") == 6
    seq_pages = cache._sequences["a"].page_ids
    assert len(seq_pages) == 2  # ceil(6/4)

    # Distinct K/V per layer; verify per-layer round-trip across the page boundary.
    for layer in range(3):
        keys = torch.arange(6 * 2 * 2, dtype=torch.float32).view(6, 2, 2) + layer * 1000
        values = keys + 500
        cache.write_layer(layer, "a", keys, values, start=0)
        mk, mv = cache.materialize_layer(layer, "a")
        torch.testing.assert_close(mk, keys)
        torch.testing.assert_close(mv, values)

    # Second request shares the pool; page table is layer-independent.
    cache.extend("b", 3)  # 1 page
    indptr, indices, last_page_len = cache.flashinfer_page_table(["a", "b"])
    torch.testing.assert_close(indptr, torch.tensor([0, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(last_page_len, torch.tensor([2, 3], dtype=torch.int32))
    assert indices.tolist() == cache._sequences["a"].page_ids + cache._sequences["b"].page_ids

    # Free returns pages to the pool for reuse.
    free_before = len(cache.free_pages)
    cache.free("a")
    assert len(cache.free_pages) == free_before + 2


def test_layered_paged_kv_cache_share_prefix_zero_copy_refcount() -> None:
    from torchinferno.runtime.paged import LayeredPagedKVCache

    cache = LayeredPagedKVCache(
        num_layers=2, num_pages=16, page_size=4,
        num_key_value_heads=2, head_dim=2,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    # Source "a": 10 tokens -> 3 pages; write distinct KV so we can verify the shared
    # prefix is the SAME storage (zero-copy), not a copy.
    cache.extend("a", 10)
    a_pages = list(cache._sequences["a"].page_ids)
    assert len(a_pages) == 3
    for layer in range(2):
        keys = torch.arange(10 * 2 * 2, dtype=torch.float32).view(10, 2, 2) + layer * 100
        cache.write_layer(layer, "a", keys, keys + 7, start=0)

    free_after_a = len(cache.free_pages)
    # Share the first 8 tokens (2 full pages) with "b"; the partial 3rd page is NOT shared.
    shared = cache.share_prefix("a", "b", 8)
    assert shared == 8  # 2 pages * page_size 4
    b_pages = cache._sequences["b"].page_ids
    assert b_pages == a_pages[:2]                 # ZERO-COPY: same page ids
    assert len(cache.free_pages) == free_after_a  # sharing allocated NO new pages
    assert cache.page_refcount(a_pages[0]) == 2   # refcounted
    assert cache.page_refcount(a_pages[2]) == 1   # unshared page unchanged

    # The shared pages carry a's KV (b reads a's prefix for free).
    for layer in range(2):
        mk, _ = cache.materialize_layer(layer, "a")
        bk, _ = cache.materialize_layer(layer, "b")
        torch.testing.assert_close(bk, mk[:8])  # b's 8 shared tokens == a's first 8

    # Freeing "b" must NOT release the shared pages (still referenced by "a").
    free_before = len(cache.free_pages)
    cache.free("b")
    assert len(cache.free_pages) == free_before   # 0 pages returned (all shared)
    assert cache.page_refcount(a_pages[0]) == 1   # decremented back

    # Freeing "a" now releases all 3 of its pages.
    cache.free("a")
    assert len(cache.free_pages) == free_before + 3


def test_paged_prefix_cache_zero_copy_share_and_evict() -> None:
    from torchinferno.runtime.paged import LayeredPagedKVCache
    from torchinferno.runtime.paged_serving import PagedPrefixCache

    cache = LayeredPagedKVCache(
        num_layers=1, num_pages=32, page_size=4,
        num_key_value_heads=1, head_dim=2,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    pc = PagedPrefixCache(cache, capacity=2)

    # "a": 12-token prompt (3 pages). Remember it, then FREE the request -- the cache
    # must RETAIN a's pages (the prefix cache holds refs).
    toks_a = list(range(12))
    cache.reserve("a", 12); cache._sequences["a"].length = 12
    a_pages = list(cache._sequences["a"].page_ids)
    pc.remember("a", toks_a)
    free_pages_before = len(cache.free_pages)
    cache.free("a")
    assert len(cache.free_pages) == free_pages_before  # retained -> none released

    # "b" shares a's first 8 tokens (2 pages) + a divergent suffix -> zero-copy share.
    toks_b = toks_a[:8] + [99, 99, 99, 99]
    shared = pc.share_into("b", toks_b)
    assert shared == 8                                  # 2 full shared pages
    assert cache._sequences["b"].page_ids == a_pages[:2]  # ZERO-COPY same pages
    assert cache.page_refcount(a_pages[0]) == 2         # retained(1) + b(1)

    # No prefix match -> no share.
    assert pc.share_into("c", [7, 7, 7, 7, 7, 7, 7, 7]) == 0

    # LRU eviction: remember 2 more (capacity=2) -> "a" evicted -> its now-unreferenced
    # page (a_pages[2], the unshared one) returns to the pool; a_pages[:2] still held by b.
    for name, base in (("d", 200), ("e", 300)):
        t = list(range(base, base + 8))
        cache.reserve(name, 8); cache._sequences[name].length = 8
        pc.remember(name, t)
    assert cache.page_refcount(a_pages[2]) == 0
    assert cache.page_refcount(a_pages[0]) == 1         # b still holds it


def test_paged_engine_resizes_decode_runner_for_wider_page_tables(monkeypatch) -> None:
    import torch as _t
    import torchinferno.runtime.paged_serving as paged_serving

    created_max_pages: list[int] = []

    class _FakeRunner:
        def __init__(self, model, cache, *, batch, max_pages):
            del model, cache, batch
            self.max_pages = max_pages
            created_max_pages.append(max_pages)

    class _FakeLayer:
        local_attention_heads = 1
        local_key_value_heads = 1

    class _FakeConfig:
        head_dim = 1

    class _FakeModel:
        device = _t.device("cpu")
        dtype = _t.float32
        layers = [_FakeLayer()]
        config = _FakeConfig()

    monkeypatch.setattr(paged_serving, "PagedDecodeGraphRunner", _FakeRunner)
    engine = paged_serving.PagedEngine(
        _FakeModel(),
        page_size=4,
        max_active=2,
        max_seq=8,
        use_graph=True,
    )

    engine.cache.reserve("short", 8)
    engine._ensure_decode_runner_capacity(["short"])
    engine.cache.reserve("wide", 12)
    engine._ensure_decode_runner_capacity(["wide"])

    assert created_max_pages == [2, 4]
    assert engine.runner.max_pages == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer paged decode needs CUDA")
def test_layered_paged_kv_cache_flashinfer_decode_matches_dense() -> None:
    # End-to-end de-risk of the paged-KV foundation: prove LayeredPagedKVCache's
    # NHD storage (layer_kv) + flashinfer_page_table feed FlashInfer paged decode
    # and match a dense per-sequence SDPA reference, across varied lengths that
    # cross page boundaries. This validates the layout the eventual llama3-TP
    # migration will rely on -- before any serving-path wiring.
    import math as _math

    flashinfer = pytest.importorskip("flashinfer")
    from torchinferno.runtime.paged import LayeredPagedKVCache

    dev = torch.device("cuda")
    torch.manual_seed(0)
    nqo, nkv, hd, page_size = 8, 2, 64, 16  # GQA 8/2, head_dim 64
    cache = LayeredPagedKVCache(
        num_layers=1, num_pages=64, page_size=page_size,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=torch.bfloat16,
    )
    lengths = {"a": 20, "b": 33, "c": 7}  # pages: 2, 3, 1 (last pages partial)
    refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for rid, n in lengths.items():
        cache.extend(rid, n)
        k = torch.randn(n, nkv, hd, device=dev, dtype=torch.bfloat16)
        v = torch.randn(n, nkv, hd, device=dev, dtype=torch.bfloat16)
        cache.write_layer(0, rid, k, v, start=0)
        refs[rid] = (k, v)

    rids = ["a", "b", "c"]
    indptr, indices, last_page_len = cache.flashinfer_page_table(rids)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    dw.plan(
        indptr=indptr, indices=indices, last_page_len=last_page_len,
        num_qo_heads=nqo, num_kv_heads=nkv, head_dim=hd, page_size=page_size,
        q_data_type=torch.bfloat16,
    )
    q = torch.randn(len(rids), nqo, hd, device=dev, dtype=torch.bfloat16)
    out = dw.run(q, cache.layer_kv(0))  # [batch, nqo, hd]

    scale = 1.0 / _math.sqrt(hd)
    rep = nqo // nkv
    for i, rid in enumerate(rids):
        k, v = refs[rid]
        kx = k.repeat_interleave(rep, dim=1).float()  # [n, nqo, hd]
        vx = v.repeat_interleave(rep, dim=1).float()
        qi = q[i].float()  # [nqo, hd]
        scores = torch.einsum("hd,nhd->hn", qi, kx) * scale
        ref = torch.einsum("hn,nhd->hd", scores.softmax(-1), vx)
        torch.testing.assert_close(out[i].float(), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer paged decode needs CUDA")
def test_layered_paged_kv_cache_scatter_write_flashinfer_matches_dense() -> None:
    # Same end-to-end check as above but the KV is written via the PRODUCTION
    # hot-path primitives slot_mapping() + scatter_write() (single batched scatter
    # over all (request, token) pairs) rather than the per-request write_layer()
    # loop. Confirms the exact write path the paged decode serving will use is
    # FlashInfer-correct vs a dense SDPA reference.
    import math as _math

    flashinfer = pytest.importorskip("flashinfer")
    from torchinferno.runtime.paged import LayeredPagedKVCache

    dev = torch.device("cuda")
    torch.manual_seed(1)
    nqo, nkv, hd, page_size = 8, 2, 64, 16
    cache = LayeredPagedKVCache(
        num_layers=1, num_pages=64, page_size=page_size,
        num_key_value_heads=nkv, head_dim=hd, device=dev, dtype=torch.bfloat16,
    )
    lengths = {"a": 20, "b": 33, "c": 7}
    refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    # Reserve pages first, then build one flat batched scatter over every token.
    all_ids: list[str] = []
    all_pos: list[int] = []
    all_k: list[torch.Tensor] = []
    all_v: list[torch.Tensor] = []
    for rid, n in lengths.items():
        cache.reserve(rid, n)
        cache._sequences[rid].length = n  # mark filled (scatter_write sets storage)
        k = torch.randn(n, nkv, hd, device=dev, dtype=torch.bfloat16)
        v = torch.randn(n, nkv, hd, device=dev, dtype=torch.bfloat16)
        refs[rid] = (k, v)
        all_ids += [rid] * n
        all_pos += list(range(n))
        all_k.append(k)
        all_v.append(v)
    slots = cache.slot_mapping(all_ids, all_pos)
    cache.scatter_write(0, slots, torch.cat(all_k, 0), torch.cat(all_v, 0))

    rids = ["a", "b", "c"]
    indptr, indices, last_page_len = cache.flashinfer_page_table(rids)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    dw.plan(
        indptr=indptr, indices=indices, last_page_len=last_page_len,
        num_qo_heads=nqo, num_kv_heads=nkv, head_dim=hd, page_size=page_size,
        q_data_type=torch.bfloat16,
    )
    q = torch.randn(len(rids), nqo, hd, device=dev, dtype=torch.bfloat16)
    out = dw.run(q, cache.layer_kv(0))

    scale = 1.0 / _math.sqrt(hd)
    rep = nqo // nkv
    for i, rid in enumerate(rids):
        k, v = refs[rid]
        kx = k.repeat_interleave(rep, dim=1).float()
        vx = v.repeat_interleave(rep, dim=1).float()
        qi = q[i].float()
        scores = torch.einsum("hd,nhd->hn", qi, kx) * scale
        ref = torch.einsum("hn,nhd->hd", scores.softmax(-1), vx)
        torch.testing.assert_close(out[i].float(), ref, atol=3e-2, rtol=3e-2)


def test_layered_paged_kv_cache_slot_mapping_scatter_write() -> None:
    from torchinferno.runtime.paged import LayeredPagedKVCache

    cache = LayeredPagedKVCache(
        num_layers=2, num_pages=16, page_size=4, num_key_value_heads=2,
        head_dim=3, device=torch.device("cpu"), dtype=torch.float32,
    )
    # Two requests; reserve enough pages for the positions we will write.
    cache.reserve("a", 8)  # 2 pages
    cache.reserve("b", 8)  # 2 pages
    # Write one token per request at a batch of positions spanning page boundaries
    # (pos 5 is page 1 offset 1 for "a"; pos 3 is page 0 offset 3 for "b"), via the
    # batched slot-mapping scatter, and confirm it lands where write_layer would.
    rids = ["a", "b", "a", "b"]
    positions = [0, 0, 5, 3]
    slots = cache.slot_mapping(rids, positions)
    # Expected flat slots: a.page0*4+0, b.page0*4+0, a.page1*4+1, b.page0*4+3.
    exp = [
        cache._sequences["a"].page_ids[0] * 4 + 0,
        cache._sequences["b"].page_ids[0] * 4 + 0,
        cache._sequences["a"].page_ids[1] * 4 + 1,
        cache._sequences["b"].page_ids[0] * 4 + 3,
    ]
    assert slots.tolist() == exp

    for layer in range(2):
        keys = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3) + layer * 100
        values = keys + 50
        cache.scatter_write(layer, slots, keys, values)
        # Read back each written slot directly from storage and compare.
        for i, (rid, pos) in enumerate(zip(rids, positions)):
            page = cache._sequences[rid].page_ids[pos // 4]
            off = pos % 4
            torch.testing.assert_close(cache.kv[layer, page, 0, off], keys[i])
            torch.testing.assert_close(cache.kv[layer, page, 1, off], values[i])


def test_layered_paged_kv_cache_slot_mapping_device_matches_host() -> None:
    # The on-device, CUDA-graph-capturable slot computation (block_table +
    # slots_from_block_table / slot_mapping_device) must produce EXACTLY the same
    # slots as the host-loop slot_mapping() -- it is the graph-friendly replacement
    # that removes the positions.tolist() host sync from the paged decode step.
    import torch as _t
    from torchinferno.runtime.paged import LayeredPagedKVCache

    cache = LayeredPagedKVCache(
        num_layers=1, num_pages=32, page_size=4, num_key_value_heads=1,
        head_dim=1, device=_t.device("cpu"), dtype=_t.float32,
    )
    cache.reserve("a", 12)  # 3 pages
    cache.reserve("b", 8)   # 2 pages
    cache.reserve("c", 16)  # 4 pages
    rids = ["a", "b", "c", "a", "c"]
    positions = [0, 5, 9, 11, 3]  # span page boundaries within each request
    host = cache.slot_mapping(rids, positions)
    dev = cache.slot_mapping_device(rids, _t.tensor(positions, dtype=_t.long))
    assert dev.tolist() == host.tolist()

    # block_table rows must be the zero-padded page-id lists.
    table = cache.block_table(rids)
    assert table.shape[0] == len(rids)
    assert table[0, :3].tolist() == cache._sequences["a"].page_ids[:3]
    # explicit slots_from_block_table call (the in-graph primitive).
    direct = LayeredPagedKVCache.slots_from_block_table(
        table, _t.tensor(positions, dtype=_t.long), cache.page_size
    )
    assert direct.tolist() == host.tolist()


def test_layered_paged_kv_cache_out_of_pages_raises() -> None:
    from torchinferno.runtime.paged import LayeredPagedKVCache

    cache = LayeredPagedKVCache(
        num_layers=1, num_pages=2, page_size=4, num_key_value_heads=1,
        head_dim=1, device=torch.device("cpu"), dtype=torch.float32,
    )
    cache.extend("a", 8)  # exactly 2 pages
    with pytest.raises(RuntimeError, match="out of pages"):
        cache.extend("b", 1)


def test_paged_kv_cache_alias_prefix_is_copy_on_write() -> None:
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    values = keys + 100
    cache.append("source", keys, values)

    target = cache.alias_prefix("source", "target", tokens=3)
    cache.append("target", torch.tensor([[[6.0, 7.0]]]), torch.tensor([[[106.0, 107.0]]]))
    source_keys, _ = cache.materialize("source")
    target_keys, _ = cache.materialize("target")

    assert target.page_ids[0] == 0
    assert target.page_ids[1] != 1
    torch.testing.assert_close(source_keys, keys)
    torch.testing.assert_close(target_keys, torch.arange(8, dtype=torch.float32).view(1, 4, 2))
    cache.free("target")
    torch.testing.assert_close(cache.materialize("source")[0], keys)
    cache.free("source")
    assert cache.free_pages == (0, 1, 2, 3)


def test_paged_kv_cache_truncate_releases_tail_pages() -> None:
    cache = PagedKVCache(
        num_pages=4,
        page_size=2,
        num_key_value_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    values = keys + 100
    cache.append("req", keys, values)

    seq = cache.truncate("req", 3)
    actual_keys, actual_values = cache.materialize("req")

    assert seq.length == 3
    assert seq.page_ids == [0, 1]
    assert cache.free_pages == (2, 3)
    torch.testing.assert_close(actual_keys, keys[:, :3, :])
    torch.testing.assert_close(actual_values, values[:, :3, :])


def test_prefix_router_uses_longest_registered_prefix() -> None:
    router = PrefixAwareRouter(default_route="cold")
    router.add_prefix((1, 2), "warm")
    router.add_prefix((1, 2, 3, 4), "hot")

    match = router.route((1, 2, 3, 9))
    cold = router.route((8, 9))

    assert match.route_id == "warm"
    assert match.matched_tokens == (1, 2)
    assert cold.route_id == "cold"
    assert cold.matched_tokens == ()


def test_disaggregated_prefill_decode_scheduler_orders_stages() -> None:
    simulator = DisaggregatedPrefillDecodeSimulator(
        prefill_ranks=(0,),
        decode_ranks=(1,),
        prefill_us_per_token=2.0,
        decode_us_per_token=3.0,
        network_latency_us=5.0,
    )

    stages = simulator.plan([InferenceJob("req", prompt_tokens=4, decode_tokens=2)])

    assert [(stage.stage, stage.rank) for stage in stages] == [("prefill", 0), ("decode", 1)]
    assert stages[1].start_us == stages[0].end_us + 5.0
    assert stages[1].elapsed_us == 6.0


def test_persistent_batch_scheduler_refills_freed_rows_with_prefix_priority() -> None:
    scheduler = PersistentBatchScheduler(max_rows=2, prefill_token_budget=6)
    scheduler.submit(
        PersistentBatchRequest(
            "cold-a",
            prompt_tokens=4,
            max_new_tokens=4,
            prefix_key="chat",
        )
    )
    scheduler.submit(
        PersistentBatchRequest(
            "hot",
            prompt_tokens=10,
            max_new_tokens=4,
            prefix_hit_tokens=8,
            prefix_key="chat",
        )
    )
    scheduler.submit(
        PersistentBatchRequest(
            "cold-b",
            prompt_tokens=4,
            max_new_tokens=4,
            prefix_key="chat",
        )
    )

    first = scheduler.step()
    second = scheduler.step()
    refill = scheduler.step(finished_request_ids=("hot",))

    assert [(item.request_id, item.row, item.prefill_tokens) for item in first.prefill_admissions] == [
        ("hot", 0, 2),
        ("cold-a", 1, 4),
    ]
    assert second.decode_request_ids == ("hot", "cold-a")
    assert second.decode_rows == (0, 1)
    assert refill.decode_request_ids == ("cold-a",)
    assert refill.decode_rows == (1,)
    assert [(item.request_id, item.row) for item in refill.prefill_admissions] == [("cold-b", 0)]


def test_persistent_batch_scheduler_groups_prefill_admissions_by_prefix_key() -> None:
    scheduler = PersistentBatchScheduler(max_rows=3)
    scheduler.submit(
        PersistentBatchRequest(
            "chat-a",
            prompt_tokens=14,
            max_new_tokens=3,
            prefix_hit_tokens=10,
            prefix_key=("chat", 10),
        )
    )
    scheduler.submit(
        PersistentBatchRequest(
            "chat-b",
            prompt_tokens=16,
            max_new_tokens=3,
            prefix_hit_tokens=10,
            prefix_key=("chat", 10),
        )
    )
    scheduler.submit(
        PersistentBatchRequest(
            "tools",
            prompt_tokens=12,
            max_new_tokens=3,
            prefix_hit_tokens=8,
            prefix_key=("tools", 8),
        )
    )

    plan = scheduler.step()

    assert [(group.prefix_key, group.request_ids, group.rows, group.suffix_tokens) for group in plan.prefill_groups] == [
        (("chat", 10), ("chat-a", "chat-b"), (0, 1), (4, 6)),
        (("tools", 8), ("tools",), (2,), (4,)),
    ]


def test_persistent_batch_scheduler_admits_one_request_over_budget_to_avoid_deadlock() -> None:
    scheduler = PersistentBatchScheduler(max_rows=1, prefill_token_budget=4)
    scheduler.submit(PersistentBatchRequest("large", prompt_tokens=32, max_new_tokens=1))

    plan = scheduler.step()

    assert [(item.request_id, item.prefill_tokens) for item in plan.prefill_admissions] == [("large", 32)]
    assert plan.finished_after_prefill == ("large",)
    assert not scheduler.has_work()


def test_token_budget_scheduler_chunks_prefill_before_first_emit() -> None:
    scheduler = TokenBudgetScheduler(
        max_rows=1,
        max_scheduled_tokens=4,
        prefill_chunk_size=4,
    )
    scheduler.submit(TokenBudgetRequest("long", prompt_tokens=10, max_new_tokens=2))

    first = scheduler.step()
    second = scheduler.step()
    third = scheduler.step()
    fourth = scheduler.step()

    assert [(chunk.kind, chunk.start_token, chunk.token_count, chunk.emits_token) for chunk in first.chunks] == [
        ("prefill", 0, 4, False),
    ]
    assert [(chunk.kind, chunk.start_token, chunk.token_count, chunk.emits_token) for chunk in second.chunks] == [
        ("prefill", 4, 4, False),
    ]
    assert [(chunk.kind, chunk.start_token, chunk.token_count, chunk.emits_token) for chunk in third.chunks] == [
        ("prefill", 8, 2, True),
    ]
    assert [(chunk.kind, chunk.start_token, chunk.token_count, chunk.emits_token) for chunk in fourth.chunks] == [
        ("decode", 10, 1, True),
    ]
    assert fourth.finished_request_ids == ("long",)
    assert not scheduler.has_work()


def test_token_budget_scheduler_schedules_running_decode_before_waiting_prefill() -> None:
    scheduler = TokenBudgetScheduler(max_rows=2, max_scheduled_tokens=2)
    scheduler.submit(TokenBudgetRequest("running", prompt_tokens=2, max_new_tokens=3))
    scheduler.submit(TokenBudgetRequest("waiting", prompt_tokens=2, max_new_tokens=1))

    first = scheduler.step()
    second = scheduler.step()

    assert [(chunk.request_id, chunk.kind, chunk.token_count) for chunk in first.chunks] == [
        ("running", "prefill", 2),
    ]
    assert [(chunk.request_id, chunk.kind, chunk.token_count) for chunk in second.chunks] == [
        ("running", "decode", 1),
        ("waiting", "prefill", 1),
    ]
    assert scheduler.active_rows == (0, 1)


def test_token_budget_scheduler_prioritizes_prefix_hits_and_reuses_rows() -> None:
    scheduler = TokenBudgetScheduler(max_rows=2, max_scheduled_tokens=8)
    scheduler.submit(
        TokenBudgetRequest(
            "cold",
            prompt_tokens=8,
            max_new_tokens=1,
            prefix_key="chat",
        )
    )
    scheduler.submit(
        TokenBudgetRequest(
            "hot",
            prompt_tokens=20,
            max_new_tokens=1,
            prefix_hit_tokens=18,
            prefix_key="chat",
        )
    )
    scheduler.submit(TokenBudgetRequest("later", prompt_tokens=2, max_new_tokens=1))

    first = scheduler.step()
    refill = scheduler.step(finished_request_ids=("hot", "cold"))

    assert [(chunk.request_id, chunk.row, chunk.start_token, chunk.token_count) for chunk in first.chunks] == [
        ("hot", 0, 18, 2),
        ("cold", 1, 0, 6),
    ]
    assert first.finished_request_ids == ("hot",)
    assert [(chunk.request_id, chunk.row, chunk.kind) for chunk in refill.chunks] == [
        ("later", 0, "prefill"),
    ]


def test_token_budget_scheduler_releases_finished_rows_after_the_plan() -> None:
    scheduler = TokenBudgetScheduler(max_rows=1, max_scheduled_tokens=4)
    scheduler.submit(TokenBudgetRequest("first", prompt_tokens=1, max_new_tokens=1))
    scheduler.submit(TokenBudgetRequest("second", prompt_tokens=1, max_new_tokens=1))

    first = scheduler.step()
    second = scheduler.step()

    assert [(chunk.request_id, chunk.row) for chunk in first.chunks] == [("first", 0)]
    assert first.finished_request_ids == ("first",)
    assert [(chunk.request_id, chunk.row) for chunk in second.chunks] == [("second", 0)]


def test_token_budget_model_step_command_preserves_scheduler_transcript() -> None:
    plan = TokenBudgetPlan(
        step=7,
        chunks=(
            TokenBudgetScheduledChunk(
                request_id="running",
                row=0,
                kind="decode",
                start_token=12,
                token_count=1,
                prompt_complete=True,
                emits_token=True,
                prefix_key="chat",
            ),
            TokenBudgetScheduledChunk(
                request_id="chunked",
                row=1,
                kind="prefill",
                start_token=16,
                token_count=8,
                prompt_complete=False,
                emits_token=False,
                prefix_key="chat",
            ),
            TokenBudgetScheduledChunk(
                request_id="finished",
                row=2,
                kind="prefill",
                start_token=30,
                token_count=2,
                prompt_complete=True,
                emits_token=True,
            ),
        ),
        finished_request_ids=("finished",),
    )

    command = token_budget_model_step_command(plan)

    assert command.step == 7
    assert command.chunks == plan.chunks
    assert command.decode_rows == (0,)
    assert command.prefill_rows == (1, 2)
    assert command.emit_request_ids == ("running", "finished")
    assert command.emit_rows == (0, 2)
    assert command.finished_request_ids == ("finished",)
    assert command.scheduled_tokens == 11
    assert not command.is_empty


def test_token_budget_model_step_command_rejects_same_step_row_reuse() -> None:
    plan = TokenBudgetPlan(
        step=0,
        chunks=(
            TokenBudgetScheduledChunk("a", row=0, kind="prefill", start_token=0, token_count=1),
            TokenBudgetScheduledChunk("b", row=0, kind="prefill", start_token=0, token_count=1),
        ),
        finished_request_ids=(),
    )

    with pytest.raises(ValueError, match="reuse a row"):
        token_budget_model_step_command(plan)


def test_token_budget_model_step_state_applies_chunked_prefill_decode_and_refill() -> None:
    rank0 = TokenBudgetModelStepState.empty(max_rows=1)
    worker = TokenBudgetModelStepState.empty(max_rows=1)
    plans = [
        TokenBudgetPlan(
            step=0,
            chunks=(TokenBudgetScheduledChunk("long", row=0, kind="prefill", start_token=0, token_count=4),),
            finished_request_ids=(),
        ),
        TokenBudgetPlan(
            step=1,
            chunks=(
                TokenBudgetScheduledChunk(
                    "long",
                    row=0,
                    kind="prefill",
                    start_token=4,
                    token_count=2,
                    prompt_complete=True,
                    emits_token=True,
                ),
            ),
            finished_request_ids=(),
        ),
        TokenBudgetPlan(
            step=2,
            chunks=(
                TokenBudgetScheduledChunk(
                    "long",
                    row=0,
                    kind="decode",
                    start_token=6,
                    token_count=1,
                    prompt_complete=True,
                    emits_token=True,
                ),
            ),
            finished_request_ids=("long",),
        ),
        TokenBudgetPlan(
            step=3,
            chunks=(
                TokenBudgetScheduledChunk(
                    "next",
                    row=0,
                    kind="prefill",
                    start_token=5,
                    token_count=1,
                    prompt_complete=True,
                    emits_token=True,
                ),
            ),
            finished_request_ids=("next",),
        ),
    ]

    results = []
    for plan in plans:
        command = token_budget_model_step_command(plan)
        results.append(apply_token_budget_model_step_command(rank0, command))
        worker_result = apply_token_budget_model_step_command(worker, command)
        assert worker_result == results[-1]
        assert worker == rank0

    assert [result.emitted_request_ids for result in results] == [(), ("long",), ("long",), ("next",)]
    assert [result.finished_request_ids for result in results] == [(), (), ("long",), ("next",)]
    assert rank0 == TokenBudgetModelStepState.empty(max_rows=1)


def test_token_budget_model_step_state_rejects_divergent_decode_transcript() -> None:
    state = TokenBudgetModelStepState.empty(max_rows=1)
    command = token_budget_model_step_command(
        TokenBudgetPlan(
            step=0,
            chunks=(
                TokenBudgetScheduledChunk(
                    "missing",
                    row=0,
                    kind="decode",
                    start_token=2,
                    token_count=1,
                    prompt_complete=True,
                    emits_token=True,
                ),
            ),
            finished_request_ids=(),
        )
    )

    with pytest.raises(ValueError, match="occupied row"):
        apply_token_budget_model_step_command(state, command)


def test_graph_pass_registry_replaces_call_function_targets() -> None:
    graph_module = torch.fx.symbolic_trace(lambda x: operator.add(x, x))
    registry = PassRegistry()
    registry.register(
        "add-to-mul",
        replace_call_function_targets({operator.add: operator.mul}),
        "Example target replacement for custom-kernel pass scaffolding.",
    )

    optimized = registry.run(graph_module)

    assert registry.names() == ["add-to-mul"]
    torch.testing.assert_close(optimized(torch.tensor([2.0])), torch.tensor([4.0]))


def test_fake_tensor_make_fx_trace_runs() -> None:
    graph_module = trace_with_make_fx(lambda x: torch.sin(x) + 1, torch.ones(2), fake=True)

    assert any(node.op == "call_function" for node in graph_module.graph.nodes)


def test_flex_attention_fallback_and_piecewise_cudagraph_runner() -> None:
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    attention = flex_attention_or_fallback(q, k, v, mask_mod=causal_mask_mod)

    runner = PiecewiseCUDAGraphRunner(enabled=False)
    runner.register(CUDAGraphPiece("decode", lambda x: x + 1))

    assert attention.shape == q.shape
    assert runner.names() == ("decode",)
    assert runner.run("decode", 4) == 5


def test_research_harness_selects_best_metric() -> None:
    harness = ResearchHarness()
    harness.register("baseline", lambda: ExperimentResult("baseline", {"latency": 10.0}))
    harness.register("candidate", lambda: ExperimentResult("candidate", {"latency": 6.0}))

    results = harness.run()
    best = harness.best(results, "latency")

    assert [result.name for result in results] == ["baseline", "candidate"]
    assert best.name == "candidate"


def test_cli_scaffold_smokes_run() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    commands = [
        [sys.executable, "-m", "torchinferno.cli", "audit"],
        [sys.executable, "-m", "torchinferno.cli", "trace-smoke", "--device", "cpu", "--tokens", "2"],
        [sys.executable, "-m", "torchinferno.cli", "sim-smoke"],
        [sys.executable, "-m", "torchinferno.cli", "research-smoke"],
    ]

    outputs = [
        subprocess.run(command, check=True, env=env, text=True, capture_output=True).stdout
        for command in commands
    ]

    assert "TorchInferno audit" in outputs[0]
    assert "TorchInferno trace smoke" in outputs[1]
    assert "TorchInferno disaggregated simulation smoke" in outputs[2]
    assert "TorchInferno research smoke" in outputs[3]
