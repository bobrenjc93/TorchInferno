from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from torchinferno.models.llama3 import (
    Llama3TensorParallelForCausalLM,
    Llama3V0ForCausalLM,
    tiny_llama3_config,
)
from torchinferno.models.llama3 import tensor_parallel as tensor_parallel_module


def test_llama3_tensor_parallel_greedy_sampler_uses_all_reduce_by_default(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 2
    model.vocab_start = 4
    model.config = type("Config", (), {"vocab_size": 8})()
    monkeypatch.delenv("TORCHINFERNO_GREEDY_SAMPLE_GATHER", raising=False)

    def gather(logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("default greedy sampling should use the all-reduce path")

    monkeypatch.setattr(model, "_sample_next_token_greedy_gather", gather)
    monkeypatch.setattr(tensor_parallel_module.dist, "all_reduce", lambda *args, **kwargs: None)

    sampled = model._sample_next_token_greedy(torch.tensor([[0.0, 3.0, 1.0]]))

    assert sampled.tolist() == [5]


def test_llama3_tensor_parallel_greedy_sampler_gather_opt_in(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 2
    model.vocab_start = 0
    model.config = type("Config", (), {"vocab_size": 8})()
    monkeypatch.setenv("TORCHINFERNO_GREEDY_SAMPLE_GATHER", "1")

    def gather(logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape == (1, 2)
        return torch.tensor([5])

    def all_reduce(*args, **kwargs) -> None:
        raise AssertionError("greedy gather opt-in should not use the all-reduce path")

    monkeypatch.setattr(model, "_sample_next_token_greedy_gather", gather)
    monkeypatch.setattr(tensor_parallel_module.dist, "all_reduce", all_reduce)

    sampled = model._sample_next_token_greedy(torch.zeros(1, 2))

    assert sampled.tolist() == [5]


def test_llama3_tensor_parallel_mixed_prefill_capture_failure_keeps_uniform_replays(
    monkeypatch,
) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.layers = []
    model._ragged_prefill_logits_graphs = {}
    model._ragged_prefill_logits_graph_failed = False
    model._ragged_prefill_mixed_logits_graph_failed = False
    model._ragged_prefill_capture_on_miss_failed = False
    model._last_ragged_prefill_graph_captured = False
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(max_seq_len=16)])
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    seq_lens = torch.zeros(4, dtype=torch.long)
    row_indices = torch.tensor([0, 1], dtype=torch.long)
    logit_positions = torch.tensor([3, 3], dtype=torch.long)
    mixed_src_rows = torch.tensor([2, 3], dtype=torch.long)
    uniform_src_row = torch.tensor([2], dtype=torch.long)
    monkeypatch.setattr(
        tensor_parallel_module,
        "_should_use_ragged_prefill_logits_graph",
        lambda *args, **kwargs: True,
    )

    def fail_capture(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("capture failed")

    monkeypatch.setattr(model, "_capture_ragged_prefill_logits_graph", fail_capture)

    mixed = model.try_prefill_ragged_logits_graph(
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        logit_positions=logit_positions,
        context_len=None,
        src_prefix_row=mixed_src_rows,
        prefix_copy_len=6,
        capture_on_miss=True,
    )

    assert mixed is None
    assert model._ragged_prefill_mixed_logits_graph_failed
    assert model._ragged_prefill_capture_on_miss_failed
    assert not model._ragged_prefill_logits_graph_failed

    captured_logits = torch.ones((2, 1, 8), dtype=torch.float32)
    capture_calls = []

    def capture_uniform(
        input_ids,
        cache,
        seq_lens,
        row_indices,
        logit_positions,
        context_len=None,
        src_prefix_row=None,
        prefix_copy_len=None,
    ):
        del seq_lens, row_indices, logit_positions, context_len, src_prefix_row, prefix_copy_len
        capture_calls.append(tuple(input_ids.shape))
        return None

    monkeypatch.setattr(model, "_capture_ragged_prefill_logits_graph", capture_uniform)

    uniform_miss = model.try_prefill_ragged_logits_graph(
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        logit_positions=logit_positions,
        context_len=8,
        src_prefix_row=uniform_src_row,
        prefix_copy_len=None,
        capture_on_miss=True,
    )

    assert uniform_miss is None
    assert capture_calls == []
    assert not model._ragged_prefill_logits_graph_failed

    class FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    fake_graph = FakeGraph()
    copied = []
    monkeypatch.setattr(
        model,
        "_copy_ragged_prefill_graph_inputs",
        lambda *args, **kwargs: copied.append((args, kwargs)),
    )
    key = (
        id(cache),
        input_ids.size(0),
        input_ids.size(1),
        cache.layers[0].max_seq_len,
        True,
        8,
        -1,
        1,
        (False,),
        0,
    )
    model._ragged_prefill_logits_graphs[key] = types.SimpleNamespace(
        graph=fake_graph,
        static_input_ids=torch.empty_like(input_ids),
        static_row_indices=torch.empty_like(row_indices),
        static_src_prefix_row=torch.empty_like(uniform_src_row),
        output_logits=captured_logits,
        cache=cache,
        max_seq_len=cache.layers[0].max_seq_len,
        context_len=8,
        prefix_copy_len=None,
    )

    uniform_replay = model.try_prefill_ragged_logits_graph(
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        logit_positions=logit_positions,
        context_len=8,
        src_prefix_row=uniform_src_row,
        prefix_copy_len=None,
        capture_on_miss=True,
    )

    assert uniform_replay is captured_logits
    assert fake_graph.replays == 1
    assert len(copied) == 1
    assert not model._last_ragged_prefill_graph_captured
    assert not model._ragged_prefill_logits_graph_failed


def test_llama3_tensor_parallel_ragged_prefill_graph_counts_evictions(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 1
    model.layers = []
    model._ragged_prefill_logits_graphs = {("old",): object()}
    model._ragged_prefill_logits_graph_evictions = 0
    model._ragged_prefill_logits_graph_evicted_entries = 0
    model._ragged_prefill_logits_graph_max_entries = 0
    model._ragged_prefill_logits_graph_failed = False
    model._ragged_prefill_mixed_logits_graph_failed = False
    model._ragged_prefill_capture_on_miss_failed = False
    model._last_ragged_prefill_graph_captured = False
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(max_seq_len=16)])
    cache._skip_capture_sync = True
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    seq_lens = torch.zeros(4, dtype=torch.long)
    row_indices = torch.tensor([0, 1], dtype=torch.long)
    logit_positions = torch.tensor([3, 3], dtype=torch.long)
    src_prefix_row = torch.tensor([2], dtype=torch.long)
    captured_logits = torch.ones((2, 1, 8), dtype=torch.float32)
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", "1")
    monkeypatch.setattr(
        tensor_parallel_module,
        "_should_use_ragged_prefill_logits_graph",
        lambda *args, **kwargs: True,
    )

    def capture_graph(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        return types.SimpleNamespace(
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            static_input_ids=torch.empty_like(input_ids),
            static_row_indices=torch.empty_like(row_indices),
            static_src_prefix_row=torch.empty_like(src_prefix_row),
            context_len=8,
            prefix_copy_len=None,
            output_logits=captured_logits,
        )

    monkeypatch.setattr(model, "_capture_ragged_prefill_logits_graph", capture_graph)

    logits = model.try_prefill_ragged_logits_graph(
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        logit_positions=logit_positions,
        context_len=8,
        src_prefix_row=src_prefix_row,
        capture_on_miss=True,
    )

    assert logits is captured_logits
    assert model._ragged_prefill_logits_graph_evictions == 1
    assert model._ragged_prefill_logits_graph_evicted_entries == 1
    assert model._ragged_prefill_logits_graph_max_entries == 1
    assert len(model._ragged_prefill_logits_graphs) == 1


def test_llama3_tensor_parallel_ragged_prefill_graph_default_cap(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", raising=False)
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 1
    model.layers = []
    model._ragged_prefill_logits_graphs = {}
    model._ragged_prefill_logits_graph_evictions = 0
    model._ragged_prefill_logits_graph_evicted_entries = 0
    model._ragged_prefill_logits_graph_max_entries = 0
    model._ragged_prefill_logits_graph_failed = False
    model._ragged_prefill_mixed_logits_graph_failed = False
    model._ragged_prefill_capture_on_miss_failed = False
    model._last_ragged_prefill_graph_captured = False
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(max_seq_len=16)])
    cache._skip_capture_sync = True
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    seq_lens = torch.zeros(4, dtype=torch.long)
    row_indices = torch.tensor([0, 1], dtype=torch.long)
    logit_positions = torch.tensor([3, 3], dtype=torch.long)
    captured_logits = torch.ones((2, 1, 8), dtype=torch.float32)
    monkeypatch.setattr(
        tensor_parallel_module,
        "_should_use_ragged_prefill_logits_graph",
        lambda *args, **kwargs: True,
    )

    def capture_graph(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        return types.SimpleNamespace(
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            static_input_ids=torch.empty_like(input_ids),
            static_row_indices=torch.empty_like(row_indices),
            static_src_prefix_row=None,
            context_len=None,
            prefix_copy_len=None,
            output_logits=captured_logits,
        )

    monkeypatch.setattr(model, "_capture_ragged_prefill_logits_graph", capture_graph)

    logits = model.try_prefill_ragged_logits_graph(
        input_ids,
        cache,
        seq_lens=seq_lens,
        row_indices=row_indices,
        logit_positions=logit_positions,
        capture_on_miss=True,
    )

    assert logits is captured_logits
    assert model._ragged_prefill_logits_graph_max_entries == 192
    assert model._ragged_prefill_logits_graph_evictions == 0


def test_llama3_tensor_parallel_ragged_prefill_graph_replay_refreshes_eviction_order(monkeypatch) -> None:
    model = object.__new__(Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.rank = 0
    model.world_size = 1
    model.layers = []
    model._ragged_prefill_logits_graphs = {}
    model._ragged_prefill_logits_graph_evictions = 0
    model._ragged_prefill_logits_graph_evicted_entries = 0
    model._ragged_prefill_logits_graph_max_entries = 0
    model._ragged_prefill_logits_graph_failed = False
    model._ragged_prefill_mixed_logits_graph_failed = False
    model._ragged_prefill_capture_on_miss_failed = False
    model._last_ragged_prefill_graph_captured = False
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(max_seq_len=16)])
    cache._skip_capture_sync = True
    seq_lens = torch.zeros(4, dtype=torch.long)
    src_prefix_row = torch.tensor([2], dtype=torch.long)
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS", "2")
    monkeypatch.setattr(
        tensor_parallel_module,
        "_should_use_ragged_prefill_logits_graph",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(model, "_copy_ragged_prefill_graph_inputs", lambda *args, **kwargs: None)

    class FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    def graph_key(batch: int) -> tuple[object, ...]:
        return (
            id(cache),
            batch,
            4,
            cache.layers[0].max_seq_len,
            True,
            8,
            -1,
            1,
            (False,),
            0,
        )

    def captured_call(batch: int, output_value: float) -> types.SimpleNamespace:
        row_indices = torch.arange(batch, dtype=torch.long)
        return types.SimpleNamespace(
            graph=FakeGraph(),
            static_input_ids=torch.empty((batch, 4), dtype=torch.long),
            static_row_indices=torch.empty_like(row_indices),
            static_src_prefix_row=torch.empty_like(src_prefix_row),
            output_logits=torch.full((batch, 1, 8), output_value, dtype=torch.float32),
            cache=cache,
            max_seq_len=cache.layers[0].max_seq_len,
            context_len=8,
            prefix_copy_len=None,
        )

    key_a = graph_key(2)
    key_b = graph_key(3)
    captured_a = captured_call(2, 1.0)
    captured_b = captured_call(3, 2.0)
    model._ragged_prefill_logits_graphs[key_a] = captured_a
    model._ragged_prefill_logits_graphs[key_b] = captured_b

    replay_logits = model.try_prefill_ragged_logits_graph(
        torch.zeros((2, 4), dtype=torch.long),
        cache,
        seq_lens=seq_lens,
        row_indices=torch.tensor([0, 1], dtype=torch.long),
        logit_positions=torch.tensor([3, 3], dtype=torch.long),
        context_len=8,
        src_prefix_row=src_prefix_row,
        capture_on_miss=True,
    )

    assert replay_logits is captured_a.output_logits
    assert captured_a.graph.replays == 1
    assert list(model._ragged_prefill_logits_graphs) == [key_b, key_a]

    captured_c = captured_call(4, 3.0)
    monkeypatch.setattr(
        model,
        "_capture_ragged_prefill_logits_graph",
        lambda *args, **kwargs: captured_c,
    )
    capture_logits = model.try_prefill_ragged_logits_graph(
        torch.zeros((4, 4), dtype=torch.long),
        cache,
        seq_lens=seq_lens,
        row_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        logit_positions=torch.tensor([3, 3, 3, 3], dtype=torch.long),
        context_len=8,
        src_prefix_row=src_prefix_row,
        capture_on_miss=True,
    )

    assert capture_logits is captured_c.output_logits
    assert model._ragged_prefill_logits_graph_evictions == 1
    assert key_b not in model._ragged_prefill_logits_graphs
    assert key_a in model._ragged_prefill_logits_graphs
    assert graph_key(4) in model._ragged_prefill_logits_graphs


def test_tensor_parallel_remote_checkpoint_resolves_on_rank_zero(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    cache_dir = tmp_path / "cache"
    calls: list[tuple[object, object, object, object]] = []

    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_rank", lambda: 0)

    def resolve(checkpoint: object, *, token: object, revision: object, cache_dir: object) -> Path:
        calls.append((checkpoint, token, revision, cache_dir))
        return snapshot

    def broadcast(objects: list[object], *, src: int, device: torch.device) -> None:
        assert src == 0
        assert device == torch.device("cpu")
        assert objects == [str(snapshot)]

    monkeypatch.setattr(tensor_parallel_module, "resolve_llama3_checkpoint", resolve)
    monkeypatch.setattr(tensor_parallel_module, "_broadcast_object_list", broadcast)

    root = tensor_parallel_module._resolve_tensor_parallel_checkpoint(
        "org/model",
        token="hf-token",
        revision="main",
        cache_dir=cache_dir,
        device=torch.device("cpu"),
    )

    assert root == snapshot
    assert calls == [("org/model", "hf-token", "main", cache_dir)]


def test_tensor_parallel_remote_checkpoint_nonzero_rank_waits_for_broadcast(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"

    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_rank", lambda: 1)

    def resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("nonzero ranks should not resolve remote checkpoints")

    def broadcast(objects: list[object], *, src: int, device: torch.device) -> None:
        assert src == 0
        objects[0] = str(snapshot)

    monkeypatch.setattr(tensor_parallel_module, "resolve_llama3_checkpoint", resolve)
    monkeypatch.setattr(tensor_parallel_module, "_broadcast_object_list", broadcast)

    root = tensor_parallel_module._resolve_tensor_parallel_checkpoint(
        "org/model",
        token=None,
        revision=None,
        cache_dir=None,
        device=torch.device("cpu"),
    )

    assert root == snapshot


def test_tensor_parallel_local_checkpoint_bypasses_distributed_broadcast(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(tensor_parallel_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        tensor_parallel_module,
        "resolve_llama3_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not resolve")),
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_broadcast_object_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not broadcast")),
    )

    root = tensor_parallel_module._resolve_tensor_parallel_checkpoint(
        checkpoint,
        token=None,
        revision=None,
        cache_dir=None,
        device=torch.device("cpu"),
    )

    assert root == checkpoint


def test_tensor_parallel_checkpoint_tensor_broadcast_defaults_off_for_cuda_tp(monkeypatch) -> None:
    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", raising=False)

    assert not tensor_parallel_module._rank0_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "0")
    assert not tensor_parallel_module._rank0_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "1")
    assert tensor_parallel_module._rank0_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )
    assert not tensor_parallel_module._rank0_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=1,
        dtype=torch.bfloat16,
    )
    assert not tensor_parallel_module._rank0_checkpoint_broadcast_enabled(
        device=torch.device("cpu"),
        world_size=2,
        dtype=torch.bfloat16,
    )


def test_tensor_parallel_replicated_checkpoint_broadcast_defaults_off_for_cuda_tp(monkeypatch) -> None:
    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", raising=False)
    monkeypatch.delenv(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST",
        raising=False,
    )

    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST", "0")
    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST", "1")
    assert tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.delenv(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST",
        raising=False,
    )
    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "0")
    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST", "1")
    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.delenv(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST",
        raising=False,
    )
    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "1")
    assert tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )
    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cuda"),
        world_size=1,
        dtype=torch.bfloat16,
    )
    assert not tensor_parallel_module._rank0_replicated_checkpoint_broadcast_enabled(
        device=torch.device("cpu"),
        world_size=2,
        dtype=torch.bfloat16,
    )


def test_tensor_parallel_replicated_checkpoint_page_cache_warm_defaults_off_for_cuda_tp(monkeypatch) -> None:
    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", raising=False)
    monkeypatch.delenv(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST",
        raising=False,
    )
    monkeypatch.delenv(
        "TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM",
        raising=False,
    )

    assert not tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "0")
    assert not tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM", "1")
    assert tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM", "0")
    assert not tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    assert not tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cuda"),
        world_size=1,
        dtype=torch.bfloat16,
    )
    assert not tensor_parallel_module._rank0_replicated_checkpoint_page_cache_warm_enabled(
        device=torch.device("cpu"),
        world_size=2,
        dtype=torch.bfloat16,
    )


def test_tensor_parallel_checkpoint_shard_scatter_defaults_off_when_broadcast_unset(monkeypatch) -> None:
    monkeypatch.setattr(tensor_parallel_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_direct_scatter_enabled", lambda: True)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", raising=False)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER", raising=False)

    assert not tensor_parallel_module._rank0_checkpoint_shard_scatter_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER", "0")
    assert not tensor_parallel_module._rank0_checkpoint_shard_scatter_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER", "1")
    assert tensor_parallel_module._rank0_checkpoint_shard_scatter_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "0")
    assert not tensor_parallel_module._rank0_checkpoint_shard_scatter_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST", "1")
    assert tensor_parallel_module._rank0_checkpoint_shard_scatter_enabled(
        device=torch.device("cuda"),
        world_size=2,
        dtype=torch.bfloat16,
    )


def test_tensor_parallel_replicated_checkpoint_broadcast_is_chunked_on_rank_zero(monkeypatch) -> None:
    full = torch.arange(10, dtype=torch.float32)
    reads: list[str] = []
    broadcasts: list[torch.Tensor] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return tuple(full.shape)

        def get_tensor(
            self,
            name: str,
            *,
            device: torch.device,
            dtype: torch.dtype | None,
        ) -> torch.Tensor:
            reads.append(name)
            return full.to(device=device, dtype=dtype)

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0
        broadcasts.append(tensor.clone())

    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_broadcast_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_checkpoint_broadcast_chunk_bytes",
        lambda: 4 * full.element_size(),
    )
    monkeypatch.setattr(tensor_parallel_module.dist, "broadcast", broadcast)

    tensor = tensor_parallel_module._load_checkpoint_tensor(
        Loader(),
        "weight",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert reads == ["weight"]
    torch.testing.assert_close(tensor, full)
    assert [chunk.tolist() for chunk in broadcasts] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0],
    ]


def test_tensor_parallel_replicated_checkpoint_broadcast_nonzero_rank_avoids_checkpoint_read(
    monkeypatch,
) -> None:
    full = torch.arange(10, dtype=torch.float32)
    offset = 0

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return tuple(full.shape)

        def get_tensor(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("nonzero ranks should receive replicated tensors from rank 0")

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        nonlocal offset
        assert src == 0
        tensor.copy_(full.narrow(0, offset, tensor.numel()))
        offset += tensor.numel()

    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_broadcast_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_checkpoint_broadcast_chunk_bytes",
        lambda: 4 * full.element_size(),
    )
    monkeypatch.setattr(tensor_parallel_module.dist, "broadcast", broadcast)

    tensor = tensor_parallel_module._load_checkpoint_tensor(
        Loader(),
        "weight",
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert offset == full.numel()
    torch.testing.assert_close(tensor, full)


def test_tensor_parallel_replicated_checkpoint_page_cache_warm_orders_rank0_read(monkeypatch) -> None:
    full = torch.arange(10, dtype=torch.float32)
    calls: list[str] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            calls.append("shape")
            return tuple(full.shape)

        def get_tensor(
            self,
            name: str,
            *,
            device: torch.device,
            dtype: torch.dtype | None,
        ) -> torch.Tensor:
            assert name == "weight"
            calls.append("read")
            return full.to(device=device, dtype=dtype)

    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_broadcast_enabled",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_page_cache_warm_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_replicated_checkpoint_page_cache_warm_min_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(tensor_parallel_module, "_barrier", lambda: calls.append("barrier"))

    tensor = tensor_parallel_module._load_checkpoint_tensor(
        Loader(),
        "weight",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == ["shape", "read", "barrier", "barrier"]
    torch.testing.assert_close(tensor, full)


def test_tensor_parallel_replicated_checkpoint_page_cache_warm_nonzero_waits_before_read(monkeypatch) -> None:
    full = torch.arange(10, dtype=torch.float32)
    calls: list[str] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            calls.append("shape")
            return tuple(full.shape)

        def get_tensor(
            self,
            name: str,
            *,
            device: torch.device,
            dtype: torch.dtype | None,
        ) -> torch.Tensor:
            assert name == "weight"
            calls.append("read")
            return full.to(device=device, dtype=dtype)

    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_broadcast_enabled",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_page_cache_warm_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_replicated_checkpoint_page_cache_warm_min_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(tensor_parallel_module, "_barrier", lambda: calls.append("barrier"))

    tensor = tensor_parallel_module._load_checkpoint_tensor(
        Loader(),
        "weight",
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == ["shape", "barrier", "read", "barrier"]
    torch.testing.assert_close(tensor, full)


def test_tensor_parallel_replicated_checkpoint_page_cache_warm_skips_small_tensors(monkeypatch) -> None:
    full = torch.arange(10, dtype=torch.float32)
    calls: list[str] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            calls.append("shape")
            return tuple(full.shape)

        def get_tensor(
            self,
            name: str,
            *,
            device: torch.device,
            dtype: torch.dtype | None,
        ) -> torch.Tensor:
            assert name == "weight"
            calls.append("read")
            return full.to(device=device, dtype=dtype)

    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_broadcast_enabled",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_rank0_replicated_checkpoint_page_cache_warm_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        tensor_parallel_module,
        "_replicated_checkpoint_page_cache_warm_min_bytes",
        lambda: 1024,
    )
    monkeypatch.setattr(tensor_parallel_module, "_barrier", lambda: calls.append("barrier"))

    tensor = tensor_parallel_module._load_checkpoint_tensor(
        Loader(),
        "weight",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == ["shape", "read"]
    torch.testing.assert_close(tensor, full)


def test_tensor_parallel_rank0_checkpoint_scatter_packs_dim1_shards(monkeypatch) -> None:
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    calls: list[str] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return tuple(full.shape)

        def get_tensor(self, name: str, *, device: torch.device, dtype: torch.dtype | None) -> torch.Tensor:
            calls.append(name)
            return full.to(device=device, dtype=dtype)

        def get_tensor_shard(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("scatter path should not read per-rank checkpoint shards")

    def reduce_scatter(output: torch.Tensor, input_tensor: torch.Tensor, **kwargs: object) -> None:
        assert kwargs["op"] == tensor_parallel_module.dist.ReduceOp.SUM
        torch.testing.assert_close(
            input_tensor,
            torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]),
        )
        output.copy_(input_tensor[: output.size(0)])

    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_broadcast_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_shard_scatter_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_direct_scatter_enabled", lambda: False)
    monkeypatch.setattr(tensor_parallel_module.dist, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SCATTER", raising=False)

    shard = tensor_parallel_module._load_checkpoint_tensor_shard(
        Loader(),
        "weight",
        dim=1,
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == ["weight"]
    torch.testing.assert_close(shard, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_tensor_parallel_rank0_checkpoint_direct_scatter_packs_dim1_shards(monkeypatch) -> None:
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    calls: list[str] = []

    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return tuple(full.shape)

        def get_tensor(self, name: str, *, device: torch.device, dtype: torch.dtype | None) -> torch.Tensor:
            calls.append(name)
            return full.to(device=device, dtype=dtype)

        def get_tensor_shard(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("scatter path should not read per-rank checkpoint shards")

    def scatter(output: torch.Tensor, scatter_list: list[torch.Tensor] | None, **kwargs: object) -> None:
        assert kwargs["src"] == 0
        assert scatter_list is not None
        assert len(scatter_list) == 2
        torch.testing.assert_close(scatter_list[0], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
        torch.testing.assert_close(scatter_list[1], torch.tensor([[3.0, 4.0], [7.0, 8.0]]))
        output.copy_(scatter_list[0])

    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_broadcast_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_shard_scatter_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_direct_scatter_enabled", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "scatter", scatter)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SCATTER", raising=False)

    shard = tensor_parallel_module._load_checkpoint_tensor_shard(
        Loader(),
        "weight",
        dim=1,
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == ["weight"]
    torch.testing.assert_close(shard, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_tensor_parallel_checkpoint_scatter_nonzero_rank_avoids_checkpoint_read(monkeypatch) -> None:
    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return (2, 4)

        def get_tensor(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("nonzero ranks should receive their shard from the collective")

        def get_tensor_shard(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("rank-0 broadcast path should not read per-rank checkpoint shards")

    def reduce_scatter(output: torch.Tensor, input_tensor: torch.Tensor, **kwargs: object) -> None:
        assert input_tensor.shape == (4, 2)
        torch.testing.assert_close(input_tensor, torch.zeros(4, 2))
        output.copy_(torch.tensor([[3.0, 4.0], [7.0, 8.0]]))

    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_broadcast_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_shard_scatter_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_direct_scatter_enabled", lambda: False)
    monkeypatch.setattr(tensor_parallel_module.dist, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SCATTER", raising=False)

    shard = tensor_parallel_module._load_checkpoint_tensor_shard(
        Loader(),
        "weight",
        dim=1,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(shard, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_tensor_parallel_direct_scatter_nonzero_rank_avoids_checkpoint_read(monkeypatch) -> None:
    class Loader:
        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            assert name == "weight"
            return (2, 4)

        def get_tensor(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("nonzero ranks should receive their shard from rank 0")

        def get_tensor_shard(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("rank-0 scatter path should not read per-rank checkpoint shards")

    def scatter(output: torch.Tensor, scatter_list: list[torch.Tensor] | None, **kwargs: object) -> None:
        assert kwargs["src"] == 0
        assert scatter_list is None
        output.copy_(torch.tensor([[3.0, 4.0], [7.0, 8.0]]))

    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_broadcast_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_shard_scatter_enabled", lambda **kwargs: True)
    monkeypatch.setattr(tensor_parallel_module, "_rank0_checkpoint_direct_scatter_enabled", lambda: True)
    monkeypatch.setattr(tensor_parallel_module.dist, "scatter", scatter)
    monkeypatch.delenv("TORCHINFERNO_TP_RANK0_CHECKPOINT_SCATTER", raising=False)

    shard = tensor_parallel_module._load_checkpoint_tensor_shard(
        Loader(),
        "weight",
        dim=1,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(shard, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_tensor_parallel_process_group_timeout_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_TP_PROCESS_GROUP_TIMEOUT_S", raising=False)
    assert tensor_parallel_module._tensor_parallel_process_group_timeout() == timedelta(seconds=1800)

    monkeypatch.setenv("TORCHINFERNO_TP_PROCESS_GROUP_TIMEOUT_S", "2400")
    assert tensor_parallel_module._tensor_parallel_process_group_timeout() == timedelta(seconds=2400)


def test_llama3_tensor_parallel_paged_cache_matches_dense_forward(tmp_path, monkeypatch) -> None:
    torch.manual_seed(9002)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model.device)

    with torch.inference_mode():
        dense_cache = model.allocate_cache(1, max_seq_len=8, cache_backend="dense")
        dense_logits, _ = model.forward(input_ids, cache=dense_cache, use_cache=True)

        paged_cache = model.allocate_cache(1, max_seq_len=8, cache_backend="paged", page_size=2)
        paged_logits, _ = model.forward(input_ids, cache=paged_cache, use_cache=True)

        decode_token = torch.tensor([[5]], dtype=torch.long, device=model.device)
        seq_lens = torch.tensor([input_ids.size(1)], dtype=torch.long, device=model.device)
        dense_decode_logits = model.decode_ragged_logits(decode_token, dense_cache, seq_lens=seq_lens)
        paged_decode_logits = model.decode_ragged_logits(decode_token, paged_cache, seq_lens=seq_lens)

    assert paged_cache.cache_backend == "paged"
    torch.testing.assert_close(paged_logits, dense_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(paged_decode_logits, dense_decode_logits, atol=2e-5, rtol=2e-5)


def test_llama3_tensor_parallel_ragged_prefill_matches_forward(tmp_path, monkeypatch) -> None:
    # Eager oracle for the row_indices ragged prefill (the continuous-batcher
    # graph body). Prefills a [batch, suffix_bucket] block into SCATTERED rows
    # with MIXED per-row prefix lengths (including a fresh start=0 row) and a
    # bucket-PADDED suffix, then asserts each row's next-token logits match a
    # plain forward() over that row's full prompt. Covers the three correctness
    # landmines: nonzero per-row rotary offset, pad-column KV not leaking, and
    # scattered-row gather.
    torch.manual_seed(9100)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=32)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device

    prompts = [
        [1, 5, 9, 13, 2],          # len 5, prefix 2 -> suffix 3
        [3, 7, 11, 4, 8, 6, 10],   # len 7, prefix 0 -> suffix 7 (fresh start)
        [2, 14, 1, 9],             # len 4, prefix 3 -> suffix 1
    ]
    rows = [3, 1, 2]               # scattered physical rows in a 4-row cache
    prefix_lens = [2, 0, 3]
    batch = len(prompts)

    with torch.inference_mode():
        # Reference: plain forward over each full prompt, last-token logits.
        ref_logits = []
        for prompt in prompts:
            ref_cache = model.allocate_cache(1, max_seq_len=32, cache_backend="dense")
            logits, _ = model.forward(
                torch.tensor([prompt], dtype=torch.long, device=device),
                cache=ref_cache,
                use_cache=True,
            )
            ref_logits.append(logits[0, -1, :])

        # Build one batch-4 dense cache; write each row's prefix via forward into
        # that row's view, then ragged-prefill all suffixes at once.
        cache = model.allocate_cache(4, max_seq_len=32, cache_backend="dense")
        for i, prompt in enumerate(prompts):
            if prefix_lens[i] > 0:
                view = cache.for_rows((rows[i],))
                model.forward(
                    torch.tensor([prompt[: prefix_lens[i]]], dtype=torch.long, device=device),
                    cache=view,
                    use_cache=True,
                )

        suffixes = [prompts[i][prefix_lens[i]:] for i in range(batch)]
        real_lens = [len(s) for s in suffixes]
        bucket = 8  # > every real suffix length, exercises pad columns
        padded = [s + [0] * (bucket - len(s)) for s in suffixes]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        seq_lens = torch.zeros(4, dtype=torch.long, device=device)
        for i in range(batch):
            seq_lens[rows[i]] = prefix_lens[i]
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        logit_positions = torch.tensor([rl - 1 for rl in real_lens], dtype=torch.long, device=device)

        out = model.prefill_ragged_logits(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
        )

    assert out.shape == (batch, 1, config.vocab_size)
    for i in range(batch):
        torch.testing.assert_close(out[i, 0, :], ref_logits[i], atol=2e-5, rtol=2e-5)


def test_llama3_tensor_parallel_ragged_prefill_folds_prefix_copy(tmp_path, monkeypatch) -> None:
    # The shared prefix is written to ONE prefix row; ragged prefill folds the
    # copy (src_prefix_row) into the forward -- broadcasting the prefix KV into
    # each active row before the suffix -- then attends via the flash context_len
    # path. Output must match a plain forward() over each full prompt.
    torch.manual_seed(9102)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=32)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device

    shared_prefix = [1, 5, 9, 13, 2, 6, 10]
    suffixes = [[3, 7], [11, 4, 8], [14]]
    prompts = [shared_prefix + s for s in suffixes]
    rows = [3, 1, 2]
    prefix_row = 0
    prefix_len = len(shared_prefix)
    bucket = 4
    batch = len(prompts)

    with torch.inference_mode():
        ref_logits = []
        for prompt in prompts:
            rc = model.allocate_cache(1, max_seq_len=32, cache_backend="dense")
            logits, _ = model.forward(torch.tensor([prompt], dtype=torch.long, device=device), cache=rc, use_cache=True)
            ref_logits.append(logits[0, -1, :])

        cache = model.allocate_cache(4, max_seq_len=32, cache_backend="dense")
        prefix_view = cache.for_rows((prefix_row,))
        model.forward(torch.tensor([shared_prefix], dtype=torch.long, device=device), cache=prefix_view, use_cache=True)

        padded = [s + [0] * (bucket - len(s)) for s in suffixes]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        seq_lens = torch.zeros(4, dtype=torch.long, device=device)
        for r in rows:
            seq_lens[r] = prefix_len
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        logit_positions = torch.tensor([len(s) - 1 for s in suffixes], dtype=torch.long, device=device)
        src_prefix_row = torch.tensor([prefix_row], dtype=torch.long, device=device)
        out = model.prefill_ragged_logits(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=prefix_len + bucket,
            src_prefix_row=src_prefix_row,
        )

    assert out.shape == (batch, 1, config.vocab_size)
    for i in range(batch):
        torch.testing.assert_close(out[i, 0, :], ref_logits[i], atol=2e-5, rtol=2e-5)


def test_llama3_tensor_parallel_ragged_prefill_dynamic_context_bucket(tmp_path, monkeypatch) -> None:
    # A negative context_len selects a fixed context bucket while start_positions
    # remain dynamic. The bucket can be larger than prefix+suffix; padded and
    # uninitialized KV columns must not affect the selected logits.
    torch.manual_seed(9103)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=32)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device

    shared_prefix = [1, 5, 9, 13, 2, 6, 10]
    suffixes = [[3, 7], [11, 4, 8], [14]]
    prompts = [shared_prefix + s for s in suffixes]
    rows = [3, 1, 2]
    prefix_row = 0
    prefix_len = len(shared_prefix)
    suffix_bucket = 4
    context_bucket = 16
    batch = len(prompts)

    with torch.inference_mode():
        ref_logits = []
        for prompt in prompts:
            rc = model.allocate_cache(1, max_seq_len=32, cache_backend="dense")
            logits, _ = model.forward(torch.tensor([prompt], dtype=torch.long, device=device), cache=rc, use_cache=True)
            ref_logits.append(logits[0, -1, :])

        cache = model.allocate_cache(4, max_seq_len=32, cache_backend="dense")
        prefix_view = cache.for_rows((prefix_row,))
        model.forward(torch.tensor([shared_prefix], dtype=torch.long, device=device), cache=prefix_view, use_cache=True)

        padded = [s + [0] * (suffix_bucket - len(s)) for s in suffixes]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        seq_lens = torch.zeros(4, dtype=torch.long, device=device)
        for r in rows:
            seq_lens[r] = prefix_len
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        logit_positions = torch.tensor([len(s) - 1 for s in suffixes], dtype=torch.long, device=device)
        src_prefix_row = torch.tensor([prefix_row], dtype=torch.long, device=device)
        out = model.prefill_ragged_logits(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
            logit_positions=logit_positions,
            context_len=-context_bucket,
            src_prefix_row=src_prefix_row,
        )

    assert out.shape == (batch, 1, config.vocab_size)
    for i in range(batch):
        torch.testing.assert_close(out[i, 0, :], ref_logits[i], atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llama3_tensor_parallel_ragged_prefill_graph_replays_across_rows(tmp_path) -> None:
    # GPU capture/replay gate: capture the ragged-prefill graph on one scattered
    # row set, then REPLAY it on a DIFFERENT scattered row set with the same
    # (batch, suffix_bucket). Asserts (a) no 'Cannot copy CPU/CUDA during
    # capture', (b) exactly ONE captured graph after both calls (replay, not
    # recapture), and (c) replayed logits match the eager oracle for the new rows.
    torch.manual_seed(9101)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=64)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
    device = model.device
    assert device.type == "cuda"

    prompts = [[1, 5, 9, 13, 2, 6], [3, 7, 11, 4, 8, 6], [2, 14, 1, 9, 5, 7]]
    batch = len(prompts)
    bucket = 8
    padded = [p + [0] * (bucket - len(p)) for p in prompts]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    logit_positions = torch.tensor([len(p) - 1 for p in prompts], dtype=torch.long, device=device)
    seq_lens = torch.zeros(4, dtype=torch.long, device=device)  # fresh start=0 rows

    with torch.inference_mode():
        ref_logits = []
        for prompt in prompts:
            ref_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
            logits, _ = model.forward(
                torch.tensor([prompt], dtype=torch.long, device=device), cache=ref_cache, use_cache=True
            )
            ref_logits.append(logits[0, -1, :])

        cache = model.allocate_cache(4, max_seq_len=64, cache_backend="dense")

        # context_len = uniform prefix (0, fresh) + suffix bucket -> exercises the
        # flash causal_lower_right path used in serving (not the boolean mask).
        context_len = bucket

        def run(rows):
            row_indices = torch.tensor(rows, dtype=torch.long, device=device)
            return model.try_prefill_ragged_logits_graph(
                input_ids, cache, seq_lens=seq_lens, row_indices=row_indices,
                logit_positions=logit_positions, context_len=context_len, capture_on_miss=True,
            )

        out_a = run([3, 1, 2])   # captures
        assert out_a is not None, "ragged prefill graph returned None (capture failed)"
        graphs_after_a = len(model._ragged_prefill_logits_graphs)
        out_b = run([0, 2, 1])   # must REPLAY (same batch+bucket, different rows)
        graphs_after_b = len(model._ragged_prefill_logits_graphs)

    assert graphs_after_a == 1, f"expected 1 captured graph, got {graphs_after_a}"
    assert graphs_after_b == 1, f"replay recaptured a new graph: {graphs_after_b}"
    for i in range(batch):
        torch.testing.assert_close(out_a[i, 0, :], ref_logits[i], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out_b[i, 0, :], ref_logits[i], atol=1e-3, rtol=1e-3)


def test_llama3_tensor_parallel_forward_decode_paged_matches_dense(tmp_path) -> None:
    # Validate the WIP true-paged-KV decode (forward_decode_paged) end-to-end on a
    # tiny model: build correct prefix KV (from a dense prefill), copy it into a
    # LayeredPagedKVCache, then paged-decode the next token and confirm its logits
    # match the dense full-forward reference (forward(seq)[-1] = decode of seq[-1]).
    flashinfer = pytest.importorskip("flashinfer")
    from torchinferno.runtime.paged import LayeredPagedKVCache

    torch.manual_seed(4242)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=64)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16").eval()
    dev = model.device
    if dev.type != "cuda":
        pytest.skip("paged decode needs CUDA")

    seq = [1, 5, 9, 13, 2, 6, 3]   # decode seq[-1] (=3) at position P
    prefix, last = seq[:-1], seq[-1]
    P = len(prefix)

    with torch.inference_mode():
        # Reference: full-forward last-position logits == decode of `last`.
        ref_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        ref_logits, _ = model.forward(
            torch.tensor([seq], dtype=torch.long, device=dev), cache=ref_cache, use_cache=True
        )
        ref = ref_logits[0, -1, :].float()

        # Build correct prefix KV via a dense prefill, then copy it into the pool.
        pre_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        model.forward(
            torch.tensor([prefix], dtype=torch.long, device=dev), cache=pre_cache, use_cache=True
        )
        nkv = model.layers[0].local_key_value_heads
        nqo = model.layers[0].local_attention_heads
        hd = config.head_dim
        nl = len(model.layers)
        paged = LayeredPagedKVCache(
            num_layers=nl, num_pages=64, page_size=4, num_key_value_heads=nkv,
            head_dim=hd, device=dev, dtype=torch.bfloat16,
        )
        paged.reserve("r", P + 1)
        paged._sequences["r"].length = P + 1  # tokens 0..P present after the decode write
        pre_slots = paged.slot_mapping(["r"] * P, list(range(P)))
        for layer_id in range(nl):
            dk = pre_cache.layers[layer_id].keys[0, :, :P, :].permute(1, 0, 2).contiguous()  # [P, nkv, hd]
            dv = pre_cache.layers[layer_id].values[0, :, :P, :].permute(1, 0, 2).contiguous()
            paged.scatter_write(layer_id, pre_slots, dk, dv)

        indptr, indices, lpl = paged.flashinfer_page_table(["r"])
        ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=dev)
        dw = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        dw.plan(
            indptr=indptr, indices=indices, last_page_len=lpl,
            num_qo_heads=nqo, num_kv_heads=nkv, head_dim=hd, page_size=4,
            q_data_type=torch.bfloat16,
        )
        out = model.forward_decode_paged(
            torch.tensor([[last]], dtype=torch.long, device=dev),
            paged, request_ids=["r"], positions=torch.tensor([P], device=dev),
            decode_wrapper=dw,
        )
    got = out.reshape(-1, out.shape[-1])[0].float()
    torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)


def test_llama3_tensor_parallel_forward_prefill_paged_matches_dense(tmp_path) -> None:
    # Validate the WIP FlashInfer-paged prefill (forward_prefill_paged): prefill a
    # fresh prompt into the small-page pool and confirm per-position logits match a
    # dense forward. Together with the decode test this proves the model-side paged
    # forward (prefill + decode) is correct end-to-end on a real model.
    flashinfer = pytest.importorskip("flashinfer")
    from torchinferno.runtime.paged import LayeredPagedKVCache

    torch.manual_seed(7777)
    # FlashInfer's prefill (hopper) kernel only supports head_dim in {64,128,256},
    # so use head_dim=64 (hidden=128, 2 q heads, 1 kv head) rather than the default
    # tiny head_dim=16 (which the decode wrapper accepts but prefill does not).
    config = tiny_llama3_config(
        vocab_size=32, max_position_embeddings=64,
        hidden_size=128, num_attention_heads=2, num_key_value_heads=1,  # head_dim = 128/2 = 64
    )
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16").eval()
    dev = model.device
    if dev.type != "cuda":
        pytest.skip("paged prefill needs CUDA")

    prompt = [1, 5, 9, 13, 2, 6]
    T = len(prompt)
    with torch.inference_mode():
        ref_cache = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
        ref_logits, _ = model.forward(
            torch.tensor([prompt], dtype=torch.long, device=dev), cache=ref_cache, use_cache=True
        )
        ref = ref_logits[0].float()  # [T, vocab]

        nkv = model.layers[0].local_key_value_heads
        nqo = model.layers[0].local_attention_heads
        hd = config.head_dim
        nl = len(model.layers)
        paged = LayeredPagedKVCache(
            num_layers=nl, num_pages=64, page_size=4, num_key_value_heads=nkv,
            head_dim=hd, device=dev, dtype=torch.bfloat16,
        )
        paged.reserve("r", T)
        paged._sequences["r"].length = T
        indptr, indices, lpl = paged.flashinfer_page_table(["r"])
        ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
        pw = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=dev)
        pw.plan(
            qo_indptr=qo_indptr, paged_kv_indptr=indptr, paged_kv_indices=indices,
            paged_kv_last_page_len=lpl, num_qo_heads=nqo, num_kv_heads=nkv,
            head_dim_qk=hd, page_size=4, causal=True, q_data_type=torch.bfloat16,
        )
        out = model.forward_prefill_paged(
            torch.tensor([prompt], dtype=torch.long, device=dev),
            paged, request_ids=["r"], prefill_wrapper=pw,
        )
    got = out[0].float()  # [T, vocab]
    torch.testing.assert_close(got, ref, atol=6e-2, rtol=6e-2)


def test_generate_paged_matches_dense_greedy(tmp_path) -> None:
    # End-to-end paged SERVING-logic check: greedy generate_paged (chained paged
    # prefill -> decode loop over a page pool) must produce the same token sequence
    # as a dense forward-greedy reference. Validates the prefill->decode chaining +
    # page management the serving engine will use.
    pytest.importorskip("flashinfer")
    from torchinferno.runtime.paged_serving import generate_paged

    torch.manual_seed(31337)
    config = tiny_llama3_config(
        vocab_size=32, max_position_embeddings=64,
        hidden_size=128, num_attention_heads=2, num_key_value_heads=1,  # head_dim 64 for FI prefill
    )
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)
    model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16").eval()
    dev = model.device
    if dev.type != "cuda":
        pytest.skip("paged serving needs CUDA")

    prompts = [[1, 5, 9, 13, 2, 6], [3, 7, 11, 4, 8, 10]]
    new_tokens = 6

    # dense forward-greedy reference (re-prefill the growing sequence each step).
    ref_out: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            seq = list(prompt)
            gen: list[int] = []
            for _ in range(new_tokens):
                c = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
                logits, _ = model.forward(torch.tensor([seq], dtype=torch.long, device=dev), cache=c, use_cache=True)
                nxt = int(logits[0, -1, :].argmax())
                gen.append(nxt)
                seq.append(nxt)
            ref_out.append(gen)

    got = generate_paged(model, prompts, max_new_tokens=new_tokens, page_size=4)
    assert got == ref_out, f"paged greedy {got} != dense greedy {ref_out}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_llama3_tensor_parallel_matches_reference_under_torchrun(tmp_path) -> None:
    torch.manual_seed(9001)
    config = tiny_llama3_config(vocab_size=32, max_position_embeddings=16)
    reference = Llama3V0ForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_hf_checkpoint(reference, config, checkpoint)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = reference(input_ids)
        expected_generated = reference.generate(input_ids, max_new_tokens=2)
    torch.save(input_ids, tmp_path / "input_ids.pt")
    torch.save(expected_logits, tmp_path / "expected_logits.pt")
    torch.save(expected_generated, tmp_path / "expected_generated.pt")

    script = tmp_path / "check_tp.py"
    script.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys

            import torch
            import torch.distributed as dist

            from torchinferno.models.llama3 import Llama3TensorParallelForCausalLM


            checkpoint, artifact_dir = sys.argv[1], sys.argv[2]
            model = Llama3TensorParallelForCausalLM.from_pretrained(checkpoint, dtype="float32").eval()
            input_ids = torch.load(f"{artifact_dir}/input_ids.pt", weights_only=True).to(model.device)
            expected_logits = torch.load(f"{artifact_dir}/expected_logits.pt", weights_only=True)
            expected_generated = torch.load(f"{artifact_dir}/expected_generated.pt", weights_only=True)

            with torch.inference_mode():
                logits, _ = model.forward(input_ids, use_cache=False)
                generated = model.generate(input_ids, max_new_tokens=2)
                local_sample_logits = torch.arange(
                    model.local_vocab_size,
                    device=model.device,
                    dtype=torch.float32,
                )[None, :].expand(3, model.local_vocab_size)
                sampled = model._sample_next_token(local_sample_logits, temperature=0.7)

            torch.testing.assert_close(logits.cpu(), expected_logits, atol=2e-5, rtol=2e-5)
            if not torch.equal(generated.cpu(), expected_generated):
                raise AssertionError(f"generated={generated.cpu().tolist()} expected={expected_generated.tolist()}")
            gathered_samples = [torch.empty_like(sampled) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sampled)
            for rank_samples in gathered_samples:
                if not torch.equal(sampled, rank_samples):
                    raise AssertionError(f"temperature samples diverged across ranks: {gathered_samples}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            """
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(script),
            str(checkpoint),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")},
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _write_hf_checkpoint(reference: Llama3V0ForCausalLM, config, checkpoint) -> None:
    state = reference.state_dict()
    hf_state = {
        "model.embed_tokens.weight": state["embed_tokens.weight"],
        "model.norm.weight": state["norm.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}."
        hf_prefix = f"model.layers.{layer_id}."
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            hf_state[hf_prefix + suffix] = state[prefix + suffix]

    save_file(hf_state, checkpoint / "model-00001-of-00001.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in hf_state.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_state},
            }
        )
        + "\n"
    )
    (checkpoint / "config.json").write_text(json.dumps(config.to_dict()) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
