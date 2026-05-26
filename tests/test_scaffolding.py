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
