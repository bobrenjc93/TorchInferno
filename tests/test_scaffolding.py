import operator
import os
import subprocess
import sys

import torch

from torchinferno.graph import PassRegistry, replace_call_function_targets, trace_with_make_fx
from torchinferno.research import ExperimentResult, ResearchHarness
from torchinferno.runtime.cudagraphs import CUDAGraphPiece, PiecewiseCUDAGraphRunner
from torchinferno.runtime.fake_dist import FakeProcessWorld
from torchinferno.runtime.flex import causal_mask_mod, flex_attention_or_fallback
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.prefix import PrefixAwareRouter
from torchinferno.runtime.scheduler import DisaggregatedPrefillDecodeSimulator, InferenceJob


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
        [sys.executable, "-m", "torchinferno.cli", "trace-smoke", "--device", "cpu", "--tokens", "2"],
        [sys.executable, "-m", "torchinferno.cli", "sim-smoke"],
        [sys.executable, "-m", "torchinferno.cli", "research-smoke"],
    ]

    outputs = [
        subprocess.run(command, check=True, env=env, text=True, capture_output=True).stdout
        for command in commands
    ]

    assert "TorchInferno trace smoke" in outputs[0]
    assert "TorchInferno disaggregated simulation smoke" in outputs[1]
    assert "TorchInferno research smoke" in outputs[2]
