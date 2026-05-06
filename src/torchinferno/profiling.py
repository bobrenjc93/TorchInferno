from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Literal

import torch

from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.graph import trace_with_make_fx
from torchinferno.graph.passes import PassRegistry
from torchinferno.kernels.ops import fused_rmsnorm_swiglu_reference
from torchinferno.kernels.ops import triton_available
from torchinferno.kernels.passes import register_kernel_replacement_passes
from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config


ModelKind = Literal["dsv4", "deepseek"]


@dataclass(frozen=True)
class ProfileRunConfig:
    output_dir: Path
    model_kind: ModelKind = "dsv4"
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    batch_size: int = 1
    prompt_tokens: int = 8
    new_tokens: int = 8
    vocab_size: int = 128
    temperature: float = 0.0
    compile: bool = False
    cache_backend: str = "dense"
    page_size: int = 16
    warmup: int = 1
    capture_graph: bool = True
    fake_graph: bool = False
    require_graph: bool = False
    capture_profiler: bool = True
    export_chrome_trace: bool = True
    with_stack: bool = False
    with_flops: bool = True
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileRunArtifacts:
    output_dir: Path
    manifest: Path
    run_config: Path
    environment: Path
    output: Path
    memory_profile: Path
    operator_profile: Path | None
    chrome_trace: Path | None
    graph_json: Path | None
    graph_text: Path | None
    graph_code: Path | None
    repro: Path


@dataclass(frozen=True)
class RegionProfileConfig:
    output_dir: Path
    region: str
    model_kind: ModelKind = "dsv4"
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    batch_size: int = 1
    tokens: int = 8
    vocab_size: int = 128
    warmup: int = 3
    iters: int = 10
    fake_graph: bool = False
    require_graph: bool = False
    capture_profiler: bool = True
    export_chrome_trace: bool = True
    with_stack: bool = False
    with_flops: bool = True
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatternProfileConfig:
    output_dir: Path
    pattern: str = "fused-rmsnorm-swiglu"
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    batch_size: int = 1
    tokens: int = 8
    hidden_size: int = 128
    warmup: int = 3
    iters: int = 10
    apply_passes: bool = True
    fake_graph: bool = False
    require_graph: bool = False
    capture_profiler: bool = True
    export_chrome_trace: bool = True
    with_stack: bool = False
    with_flops: bool = True
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class FocusProfileArtifacts:
    output_dir: Path
    manifest: Path
    run_config: Path
    environment: Path
    output: Path
    memory_profile: Path
    operator_profile: Path | None
    chrome_trace: Path | None
    graph_json: Path | None
    graph_text: Path | None
    graph_code: Path | None
    repro: Path


@dataclass(frozen=True)
class PatternProfileArtifacts:
    output_dir: Path
    manifest: Path
    run_config: Path
    environment: Path
    comparison: Path
    pass_report: Path
    reference_profile: Path | None
    optimized_profile: Path | None
    reference_trace: Path | None
    optimized_trace: Path | None
    reference_graph: Path | None
    optimized_graph: Path | None
    repro: Path


@dataclass(frozen=True)
class SubgraphProfileConfig:
    output_dir: Path
    source_run_dir: Path
    node_ids: tuple[int, ...]
    device: str | None = None
    warmup: int = 3
    iters: int = 10
    capture_profiler: bool = True
    export_chrome_trace: bool = True
    with_stack: bool = False
    with_flops: bool = True
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubgraphProfileArtifacts:
    output_dir: Path
    manifest: Path
    run_config: Path
    environment: Path
    source_graph: Path
    subgraph_spec: Path
    subgraph_graph: Path
    subgraph_text: Path
    subgraph_code: Path
    output: Path
    memory_profile: Path
    operator_profile: Path | None
    chrome_trace: Path | None
    repro: Path


@dataclass(frozen=True)
class _RegionWorkload:
    module_path: str
    module_class: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    input_names: tuple[str, ...]


@dataclass(frozen=True)
class _SubgraphBoundary:
    arg_name: str
    source_id: int
    source_name: str
    source_op: str
    source_target: str


@dataclass(frozen=True)
class _ExtractedSubgraph:
    graph_module: torch.fx.GraphModule
    boundaries: tuple[_SubgraphBoundary, ...]
    output_nodes: tuple[torch.fx.Node, ...]
    selected_nodes: tuple[torch.fx.Node, ...]


def run_profile_capture(config: ProfileRunConfig) -> ProfileRunArtifacts:
    """Run one generation workload and write graph/profile artifacts."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    dtype = _dtype_from_name(config.dtype)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        cuda_index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
        torch.cuda.manual_seed_all(config.seed)

    model = _build_model(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.prompt_tokens),
        device=device,
        dtype=torch.long,
    )

    run_config_path = output_dir / "run_config.json"
    environment_path = output_dir / "environment.json"
    input_path = output_dir / "input_ids.json"
    output_path = output_dir / "output.json"
    memory_path = output_dir / "memory_profile.json"
    operator_path = output_dir / "operator_profile.json" if config.capture_profiler else None
    chrome_path = output_dir / "chrome_trace.json" if config.capture_profiler and config.export_chrome_trace else None
    graph_json_path = output_dir / "graph.json" if config.capture_graph else None
    graph_text_path = output_dir / "graph.txt" if config.capture_graph else None
    graph_code_path = output_dir / "graph_module.py" if config.capture_graph else None
    manifest_path = output_dir / "manifest.json"
    repro_path = output_dir / "repro.py"

    _write_json(run_config_path, _config_to_json(config))
    _write_json(environment_path, _environment_payload(device))
    _write_json(input_path, {"input_ids": input_ids.detach().cpu().tolist()})

    graph_error: dict[str, Any] | None = None
    if config.capture_graph:
        try:
            graph_module = _trace_forward_graph(model, input_ids, fake=config.fake_graph)
            assert graph_json_path is not None
            assert graph_text_path is not None
            assert graph_code_path is not None
            _write_json(graph_json_path, _graph_to_json(graph_module))
            graph_text_path.write_text(str(graph_module.graph) + "\n")
            graph_code_path.write_text(graph_module.code)
        except Exception as exc:
            graph_error = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(output_dir / "graph_error.json", graph_error)
            if config.require_graph:
                raise

    if config.compile:
        compile_forward(model, CompileConfig(mode="reduce-overhead"))

    for _ in range(config.warmup):
        _run_generate(model, input_ids, config)
    _sync_if_needed(device)

    memory_before = _memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    profiler = None
    if config.capture_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=config.with_stack,
            with_flops=config.with_flops,
            acc_events=True,
        ) as prof:
            with torch.profiler.record_function("torchinferno.generate"):
                generated = _run_generate(model, input_ids, config)
            prof.step()
        profiler = prof
    else:
        generated = _run_generate(model, input_ids, config)
    _sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

    memory_after = _memory_snapshot(device)
    _write_json(
        output_path,
        {
            "shape": list(generated.shape),
            "tokens": generated.detach().cpu().tolist(),
            "elapsed_ms": elapsed_ms,
            "tokens_per_second": _tokens_per_second(config.batch_size * config.new_tokens, elapsed_ms),
        },
    )
    _write_json(memory_path, {"before": memory_before, "after": memory_after})

    if profiler is not None:
        assert operator_path is not None
        _write_json(operator_path, _profiler_key_averages(profiler))
        if chrome_path is not None:
            profiler.export_chrome_trace(str(chrome_path))

    _write_repro(repro_path, config, input_ids.detach().cpu().tolist())
    artifacts = ProfileRunArtifacts(
        output_dir=output_dir,
        manifest=manifest_path,
        run_config=run_config_path,
        environment=environment_path,
        output=output_path,
        memory_profile=memory_path,
        operator_profile=operator_path,
        chrome_trace=chrome_path,
        graph_json=graph_json_path if graph_json_path is not None and graph_json_path.exists() else None,
        graph_text=graph_text_path if graph_text_path is not None and graph_text_path.exists() else None,
        graph_code=graph_code_path if graph_code_path is not None and graph_code_path.exists() else None,
        repro=repro_path,
    )
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "artifacts": _artifact_manifest(artifacts),
            "graph_error": graph_error,
        },
    )
    return artifacts


def run_region_profile_capture(config: RegionProfileConfig) -> FocusProfileArtifacts:
    """Profile one named model region and write focused artifacts."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _normalize_device(torch.device(config.device))
    dtype = _dtype_from_name(config.dtype)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model_config = ProfileRunConfig(
        output_dir=output_dir,
        model_kind=config.model_kind,
        device=str(device),
        dtype=config.dtype,
        seed=config.seed,
        batch_size=config.batch_size,
        prompt_tokens=config.tokens,
        new_tokens=1,
        vocab_size=config.vocab_size,
        warmup=0,
    )
    model = _build_model(model_config).to(device=device, dtype=dtype).eval()
    workload = _build_region_workload(model, config.region, config.batch_size, config.tokens, device, dtype)

    run_config_path = output_dir / "run_config.json"
    environment_path = output_dir / "environment.json"
    output_path = output_dir / "output.json"
    memory_path = output_dir / "memory_profile.json"
    operator_path = output_dir / "operator_profile.json" if config.capture_profiler else None
    chrome_path = output_dir / "chrome_trace.json" if config.capture_profiler and config.export_chrome_trace else None
    graph_json_path = output_dir / "region_graph.json"
    graph_text_path = output_dir / "region_graph.txt"
    graph_code_path = output_dir / "region_graph_module.py"
    manifest_path = output_dir / "manifest.json"
    repro_path = output_dir / "region_repro.py"

    _write_json(run_config_path, _region_config_to_json(config))
    _write_json(environment_path, _environment_payload(device))
    _write_json(
        output_dir / "region_spec.json",
        {
            "region": config.region,
            "resolved_module_path": workload.module_path,
            "module_class": workload.module_class,
            "input_names": list(workload.input_names),
            "inputs": [_value_summary(arg) for arg in workload.args],
        },
    )

    graph_error: dict[str, Any] | None = None
    try:
        graph_module = trace_with_make_fx(workload.fn, *workload.args, fake=config.fake_graph)
        _write_json(graph_json_path, _graph_to_json(graph_module))
        graph_text_path.write_text(str(graph_module.graph) + "\n")
        graph_code_path.write_text(graph_module.code)
    except Exception as exc:
        graph_error = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output_dir / "graph_error.json", graph_error)
        if config.require_graph:
            raise

    for _ in range(config.warmup):
        workload.fn(*workload.args)
    _sync_if_needed(device)
    memory_before = _memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result, elapsed_ms, profiler = _profile_callable(
        workload.fn,
        workload.args,
        label=f"torchinferno.region.{workload.module_path}",
        device=device,
        iters=config.iters,
        capture_profiler=config.capture_profiler,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
    )
    memory_after = _memory_snapshot(device)

    _write_json(
        output_path,
        {
            "elapsed_ms": elapsed_ms,
            "iters": config.iters,
            "per_iter_ms": elapsed_ms / max(1, config.iters),
            "output": _value_summary(result),
        },
    )
    _write_json(memory_path, {"before": memory_before, "after": memory_after})
    if profiler is not None:
        assert operator_path is not None
        _write_json(operator_path, _profiler_key_averages(profiler))
        if chrome_path is not None:
            profiler.export_chrome_trace(str(chrome_path))
    _write_region_repro(repro_path, config)

    artifacts = FocusProfileArtifacts(
        output_dir=output_dir,
        manifest=manifest_path,
        run_config=run_config_path,
        environment=environment_path,
        output=output_path,
        memory_profile=memory_path,
        operator_profile=operator_path,
        chrome_trace=chrome_path,
        graph_json=graph_json_path if graph_json_path.exists() else None,
        graph_text=graph_text_path if graph_text_path.exists() else None,
        graph_code=graph_code_path if graph_code_path.exists() else None,
        repro=repro_path,
    )
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "artifacts": _focus_artifact_manifest(artifacts),
            "graph_error": graph_error,
        },
    )
    _write_focus_readme(output_dir / "README.md", "region", artifacts)
    return artifacts


def run_pattern_profile_capture(config: PatternProfileConfig) -> PatternProfileArtifacts:
    """Profile a known reference pattern and its optimized graph replacement."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _normalize_device(torch.device(config.device))
    dtype = _dtype_from_name(config.dtype)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    fn, args = _build_pattern_workload(config, device, dtype)

    run_config_path = output_dir / "run_config.json"
    environment_path = output_dir / "environment.json"
    comparison_path = output_dir / "comparison.json"
    pass_report_path = output_dir / "pass_report.json"
    manifest_path = output_dir / "manifest.json"
    repro_path = output_dir / "pattern_repro.py"
    _write_json(run_config_path, _pattern_config_to_json(config))
    _write_json(environment_path, _environment_payload(device))
    _write_json(output_dir / "pattern_spec.json", {"pattern": config.pattern, "inputs": [_value_summary(arg) for arg in args]})

    reference_graph_path: Path | None = output_dir / "reference_graph.json"
    optimized_graph_path: Path | None = output_dir / "optimized_graph.json"
    reference_profile_path = output_dir / "reference_profile.json" if config.capture_profiler else None
    optimized_profile_path = output_dir / "optimized_profile.json" if config.capture_profiler else None
    reference_trace_path = output_dir / "reference_chrome_trace.json" if config.capture_profiler and config.export_chrome_trace else None
    optimized_trace_path = output_dir / "optimized_chrome_trace.json" if config.capture_profiler and config.export_chrome_trace else None

    graph_error: dict[str, Any] | None = None
    optimized_graph_module: torch.fx.GraphModule | None = None
    pass_report: dict[str, Any] = {"applied": config.apply_passes, "passes": [], "graph_meta": {}}
    try:
        reference_graph_module = trace_with_make_fx(fn, *args, fake=config.fake_graph)
        _write_graph_artifacts(output_dir, "reference", reference_graph_module)
        if config.apply_passes:
            registry = PassRegistry()
            register_kernel_replacement_passes(registry)
            pass_report["passes"] = [
                {"name": registered.name, "description": registered.description}
                for registered in registry.describe()
            ]
            optimized_graph_module = registry.run(reference_graph_module)
        else:
            optimized_graph_module = reference_graph_module
        pass_report["graph_meta"] = _plain_json(optimized_graph_module.meta)
        _write_graph_artifacts(output_dir, "optimized", optimized_graph_module)
    except Exception as exc:
        graph_error = {"type": type(exc).__name__, "message": str(exc)}
        pass_report["error"] = graph_error
        _write_json(output_dir / "graph_error.json", graph_error)
        reference_graph_path = None
        optimized_graph_path = None
        if config.require_graph:
            raise
    _write_json(pass_report_path, pass_report)

    reference_result, reference_elapsed_ms, reference_profiler = _profile_callable(
        fn,
        args,
        label=f"torchinferno.pattern.{config.pattern}.reference",
        device=device,
        iters=config.iters,
        capture_profiler=config.capture_profiler,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        warmup=config.warmup,
    )
    optimized_callable: Callable[..., Any] = optimized_graph_module if optimized_graph_module is not None else fn
    optimized_result, optimized_elapsed_ms, optimized_profiler = _profile_callable(
        optimized_callable,
        args,
        label=f"torchinferno.pattern.{config.pattern}.optimized",
        device=device,
        iters=config.iters,
        capture_profiler=config.capture_profiler,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        warmup=config.warmup,
    )
    if reference_profiler is not None:
        assert reference_profile_path is not None
        _write_json(reference_profile_path, _profiler_key_averages(reference_profiler))
        if reference_trace_path is not None:
            reference_profiler.export_chrome_trace(str(reference_trace_path))
    if optimized_profiler is not None:
        assert optimized_profile_path is not None
        _write_json(optimized_profile_path, _profiler_key_averages(optimized_profiler))
        if optimized_trace_path is not None:
            optimized_profiler.export_chrome_trace(str(optimized_trace_path))

    comparison = {
        "pattern": config.pattern,
        "apply_passes": config.apply_passes,
        "reference_elapsed_ms": reference_elapsed_ms,
        "optimized_elapsed_ms": optimized_elapsed_ms,
        "reference_per_iter_ms": reference_elapsed_ms / max(1, config.iters),
        "optimized_per_iter_ms": optimized_elapsed_ms / max(1, config.iters),
        "speedup": reference_elapsed_ms / optimized_elapsed_ms if optimized_elapsed_ms > 0 else None,
        "max_abs_diff": _max_abs_diff(reference_result, optimized_result),
        "reference_output": _value_summary(reference_result),
        "optimized_output": _value_summary(optimized_result),
    }
    _write_json(comparison_path, comparison)
    _write_pattern_repro(repro_path, config)
    artifacts = PatternProfileArtifacts(
        output_dir=output_dir,
        manifest=manifest_path,
        run_config=run_config_path,
        environment=environment_path,
        comparison=comparison_path,
        pass_report=pass_report_path,
        reference_profile=reference_profile_path,
        optimized_profile=optimized_profile_path,
        reference_trace=reference_trace_path,
        optimized_trace=optimized_trace_path,
        reference_graph=reference_graph_path if reference_graph_path is not None and reference_graph_path.exists() else None,
        optimized_graph=optimized_graph_path if optimized_graph_path is not None and optimized_graph_path.exists() else None,
        repro=repro_path,
    )
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "artifacts": _pattern_artifact_manifest(artifacts),
            "graph_error": graph_error,
        },
    )
    _write_pattern_readme(output_dir / "README.md", artifacts)
    return artifacts


def run_subgraph_profile_capture(config: SubgraphProfileConfig) -> SubgraphProfileArtifacts:
    """Extract and profile an arbitrary FX subgraph from a prior profile-run."""

    if not config.node_ids:
        raise ValueError("node_ids must contain at least one graph node id")
    config = replace(config, source_run_dir=config.source_run_dir.resolve())
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_run_dir = config.source_run_dir
    source_config = _load_profile_run_config(source_run_dir / "run_config.json")
    device = _normalize_device(torch.device(config.device or source_config.device))
    dtype = _dtype_from_name(source_config.dtype)
    torch.manual_seed(source_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(source_config.seed)

    effective_source_config = replace(source_config, output_dir=output_dir, device=str(device))
    model = _build_model(effective_source_config).to(device=device, dtype=dtype).eval()
    input_ids = _load_profile_input_ids(source_run_dir, effective_source_config, device)

    def forward_only(ids: torch.Tensor) -> torch.Tensor:
        logits, _ = model(ids, use_cache=False)
        return logits

    run_config_path = output_dir / "run_config.json"
    environment_path = output_dir / "environment.json"
    source_graph_path = output_dir / "source_graph.json"
    subgraph_spec_path = output_dir / "subgraph_spec.json"
    subgraph_graph_path = output_dir / "subgraph_graph.json"
    subgraph_text_path = output_dir / "subgraph_graph.txt"
    subgraph_code_path = output_dir / "subgraph_graph_module.py"
    output_path = output_dir / "output.json"
    memory_path = output_dir / "memory_profile.json"
    operator_path = output_dir / "operator_profile.json" if config.capture_profiler else None
    chrome_path = output_dir / "chrome_trace.json" if config.capture_profiler and config.export_chrome_trace else None
    manifest_path = output_dir / "manifest.json"
    repro_path = output_dir / "subgraph_repro.py"

    _write_json(run_config_path, _subgraph_config_to_json(config))
    _write_json(environment_path, _environment_payload(device))

    graph_module = trace_with_make_fx(forward_only, input_ids, fake=False)
    _write_json(source_graph_path, _graph_to_json(graph_module))
    (output_dir / "source_graph.txt").write_text(str(graph_module.graph) + "\n")
    (output_dir / "source_graph_module.py").write_text(graph_module.code)

    extracted = _extract_fx_subgraph(graph_module, config.node_ids)
    needed_value_names = {boundary.source_name for boundary in extracted.boundaries}
    needed_value_names.update(node.name for node in extracted.output_nodes)
    with torch.inference_mode():
        node_values = _capture_graph_node_values(graph_module, (input_ids,), needed_value_names)
    boundary_args = tuple(node_values[boundary.source_name] for boundary in extracted.boundaries)
    expected_outputs = tuple(node_values[node.name] for node in extracted.output_nodes)
    expected = expected_outputs[0] if len(expected_outputs) == 1 else expected_outputs

    _write_json(subgraph_graph_path, _graph_to_json(extracted.graph_module))
    subgraph_text_path.write_text(str(extracted.graph_module.graph) + "\n")
    subgraph_code_path.write_text(extracted.graph_module.code)
    source_ids = _node_id_map(graph_module)
    _write_json(
        subgraph_spec_path,
        {
            "source_run_dir": str(source_run_dir),
            "requested_node_ids": list(config.node_ids),
            "selected_nodes": [_node_summary(node, source_ids[node]) for node in extracted.selected_nodes],
            "boundary_inputs": [asdict(boundary) for boundary in extracted.boundaries],
            "output_nodes": [_node_summary(node, source_ids[node]) for node in extracted.output_nodes],
            "boundary_values": [_value_summary(arg) for arg in boundary_args],
        },
    )

    for _ in range(config.warmup):
        with torch.inference_mode():
            extracted.graph_module(*boundary_args)
    _sync_if_needed(device)
    memory_before = _memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result, elapsed_ms, profiler = _profile_callable(
        extracted.graph_module,
        boundary_args,
        label="torchinferno.subgraph",
        device=device,
        iters=config.iters,
        capture_profiler=config.capture_profiler,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
    )
    memory_after = _memory_snapshot(device)

    _write_json(
        output_path,
        {
            "elapsed_ms": elapsed_ms,
            "iters": config.iters,
            "per_iter_ms": elapsed_ms / max(1, config.iters),
            "max_abs_diff_vs_source": _max_abs_diff(expected, result),
            "expected_output": _value_summary(expected),
            "output": _value_summary(result),
        },
    )
    _write_json(memory_path, {"before": memory_before, "after": memory_after})
    if profiler is not None:
        assert operator_path is not None
        _write_json(operator_path, _profiler_key_averages(profiler))
        if chrome_path is not None:
            profiler.export_chrome_trace(str(chrome_path))
    _write_subgraph_repro(repro_path, config)

    artifacts = SubgraphProfileArtifacts(
        output_dir=output_dir,
        manifest=manifest_path,
        run_config=run_config_path,
        environment=environment_path,
        source_graph=source_graph_path,
        subgraph_spec=subgraph_spec_path,
        subgraph_graph=subgraph_graph_path,
        subgraph_text=subgraph_text_path,
        subgraph_code=subgraph_code_path,
        output=output_path,
        memory_profile=memory_path,
        operator_profile=operator_path,
        chrome_trace=chrome_path,
        repro=repro_path,
    )
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "artifacts": _subgraph_artifact_manifest(artifacts),
        },
    )
    _write_subgraph_readme(output_dir / "README.md", artifacts)
    return artifacts


def _build_model(config: ProfileRunConfig) -> DSv4ForCausalLM | DeepSeekV32ForCausalLM:
    max_seq_len = config.prompt_tokens + config.new_tokens + 8
    if config.model_kind == "dsv4":
        return DSv4ForCausalLM(tiny_dsv4_config(vocab_size=config.vocab_size, max_seq_len=max_seq_len))
    if config.model_kind == "deepseek":
        return DeepSeekV32ForCausalLM(
            tiny_deepseek_v32_config(vocab_size=config.vocab_size, max_position_embeddings=max_seq_len)
        )
    raise ValueError(f"unknown model kind: {config.model_kind}")


def _run_generate(
    model: DSv4ForCausalLM | DeepSeekV32ForCausalLM,
    input_ids: torch.Tensor,
    config: ProfileRunConfig,
) -> torch.Tensor:
    with torch.inference_mode():
        if isinstance(model, DeepSeekV32ForCausalLM):
            return model.generate(
                input_ids,
                max_new_tokens=config.new_tokens,
                temperature=config.temperature,
                cache_backend=config.cache_backend,
                page_size=config.page_size,
            )
        return model.generate(input_ids, max_new_tokens=config.new_tokens, temperature=config.temperature)


def _trace_forward_graph(
    model: DSv4ForCausalLM | DeepSeekV32ForCausalLM,
    input_ids: torch.Tensor,
    *,
    fake: bool,
) -> torch.fx.GraphModule:
    def forward_only(ids: torch.Tensor) -> torch.Tensor:
        logits, _ = model(ids, use_cache=False)
        return logits

    return trace_with_make_fx(forward_only, input_ids, fake=fake)


def _graph_to_json(graph_module: torch.fx.GraphModule) -> dict[str, Any]:
    nodes = []
    for node_id, node in enumerate(graph_module.graph.nodes):
        nodes.append(
            {
                "id": node_id,
                "label": f"{node_id}:{node.name}",
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": _serialize_node_arg(node.args),
                "kwargs": _serialize_node_arg(node.kwargs),
                "users": sorted(user.name for user in node.users),
                "meta": _serialize_node_meta(node.meta),
            }
        )
    return {"node_count": len(nodes), "nodes": nodes}


def _serialize_node_arg(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if isinstance(value, tuple):
        return [_serialize_node_arg(item) for item in value]
    if isinstance(value, list):
        return [_serialize_node_arg(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_node_arg(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _serialize_node_meta(meta: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tensor = meta.get("val")
    if isinstance(tensor, torch.Tensor):
        result["tensor"] = _tensor_summary(tensor)
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is not None:
        result["tensor_meta"] = repr(tensor_meta)
    return result


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _profiler_key_averages(profiler: torch.profiler.profile) -> dict[str, Any]:
    events = []
    for event in profiler.key_averages():
        events.append(
            {
                "key": event.key,
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0.0)),
                "device_time_total_us": float(getattr(event, "device_time_total", 0.0)),
                "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0)),
                "flops": int(getattr(event, "flops", 0) or 0),
            }
        )
    events.sort(
        key=lambda event: event["self_device_time_total_us"] or event["self_cpu_time_total_us"],
        reverse=True,
    )
    return {"events": events}


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type != "cuda":
        return payload
    payload.update(
        {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "stats": {key: int(value) for key, value in torch.cuda.memory_stats(device).items()},
        }
    )
    return payload


def _environment_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "triton_available": triton_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["cuda_device_name"] = torch.cuda.get_device_name(device)
        payload["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return payload


def _config_to_json(config: ProfileRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["command"] = list(config.command)
    return payload


def _artifact_manifest(artifacts: ProfileRunArtifacts) -> dict[str, str | None]:
    output_dir = artifacts.output_dir

    def rel(path: Path | None) -> str | None:
        return str(path.relative_to(output_dir)) if path is not None else None

    return {
        "run_config": rel(artifacts.run_config),
        "environment": rel(artifacts.environment),
        "output": rel(artifacts.output),
        "memory_profile": rel(artifacts.memory_profile),
        "operator_profile": rel(artifacts.operator_profile),
        "chrome_trace": rel(artifacts.chrome_trace),
        "graph_json": rel(artifacts.graph_json),
        "graph_text": rel(artifacts.graph_text),
        "graph_code": rel(artifacts.graph_code),
        "repro": rel(artifacts.repro),
    }


def _focus_artifact_manifest(artifacts: FocusProfileArtifacts) -> dict[str, str | None]:
    output_dir = artifacts.output_dir

    def rel(path: Path | None) -> str | None:
        return str(path.relative_to(output_dir)) if path is not None else None

    return {
        "run_config": rel(artifacts.run_config),
        "environment": rel(artifacts.environment),
        "output": rel(artifacts.output),
        "memory_profile": rel(artifacts.memory_profile),
        "operator_profile": rel(artifacts.operator_profile),
        "chrome_trace": rel(artifacts.chrome_trace),
        "graph_json": rel(artifacts.graph_json),
        "graph_text": rel(artifacts.graph_text),
        "graph_code": rel(artifacts.graph_code),
        "repro": rel(artifacts.repro),
    }


def _pattern_artifact_manifest(artifacts: PatternProfileArtifacts) -> dict[str, str | None]:
    output_dir = artifacts.output_dir

    def rel(path: Path | None) -> str | None:
        return str(path.relative_to(output_dir)) if path is not None else None

    return {
        "run_config": rel(artifacts.run_config),
        "environment": rel(artifacts.environment),
        "comparison": rel(artifacts.comparison),
        "pass_report": rel(artifacts.pass_report),
        "reference_profile": rel(artifacts.reference_profile),
        "optimized_profile": rel(artifacts.optimized_profile),
        "reference_trace": rel(artifacts.reference_trace),
        "optimized_trace": rel(artifacts.optimized_trace),
        "reference_graph": rel(artifacts.reference_graph),
        "optimized_graph": rel(artifacts.optimized_graph),
        "repro": rel(artifacts.repro),
    }


def _subgraph_artifact_manifest(artifacts: SubgraphProfileArtifacts) -> dict[str, str | None]:
    output_dir = artifacts.output_dir

    def rel(path: Path | None) -> str | None:
        return str(path.relative_to(output_dir)) if path is not None else None

    return {
        "run_config": rel(artifacts.run_config),
        "environment": rel(artifacts.environment),
        "source_graph": rel(artifacts.source_graph),
        "subgraph_spec": rel(artifacts.subgraph_spec),
        "subgraph_graph": rel(artifacts.subgraph_graph),
        "subgraph_text": rel(artifacts.subgraph_text),
        "subgraph_code": rel(artifacts.subgraph_code),
        "output": rel(artifacts.output),
        "memory_profile": rel(artifacts.memory_profile),
        "operator_profile": rel(artifacts.operator_profile),
        "chrome_trace": rel(artifacts.chrome_trace),
        "repro": rel(artifacts.repro),
    }


def _build_region_workload(
    model: DSv4ForCausalLM | DeepSeekV32ForCausalLM,
    region: str,
    batch_size: int,
    tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _RegionWorkload:
    module_path, module = _resolve_region_module(model, region)
    module_class = module.__class__.__name__
    hidden_size = int(getattr(model.config, "hidden_size"))
    hidden = torch.randn(batch_size, tokens, hidden_size, device=device, dtype=dtype)
    positions = torch.arange(tokens, device=device)
    lower_class = module_class.lower()

    if module is model:
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, tokens), device=device, dtype=torch.long)

        def fn(ids: torch.Tensor) -> torch.Tensor:
            logits, _ = model(ids, use_cache=False)
            return logits

        return _RegionWorkload(module_path, module_class, fn, (input_ids,), ("input_ids",))

    if isinstance(module, torch.nn.Embedding):
        input_ids = torch.randint(0, module.num_embeddings, (batch_size, tokens), device=device, dtype=torch.long)
        return _RegionWorkload(module_path, module_class, module, (input_ids,), ("input_ids",))

    if isinstance(module, torch.nn.Linear):
        x = torch.randn(batch_size, tokens, module.in_features, device=device, dtype=dtype)
        return _RegionWorkload(module_path, module_class, module, (x,), ("x",))

    if "attention" in lower_class or "decoderlayer" in lower_class:

        def fn(x: torch.Tensor, pos: torch.Tensor) -> Any:
            return module(x, pos, None)

        return _RegionWorkload(module_path, module_class, fn, (hidden, positions), ("hidden_states", "positions"))

    if any(name in lower_class for name in ("rmsnorm", "layernorm", "norm", "moe", "mlp", "expert", "gate")):
        return _RegionWorkload(module_path, module_class, module, (hidden,), ("hidden_states",))

    try:
        module(hidden)
    except Exception as exc:
        raise ValueError(
            f"Do not know how to build inputs for region {region!r} ({module_class}). "
            "Supported regions include attention, decoder layer, norm, MLP/MoE, embedding, linear, and full model."
        ) from exc
    return _RegionWorkload(module_path, module_class, module, (hidden,), ("hidden_states",))


def _resolve_region_module(
    model: DSv4ForCausalLM | DeepSeekV32ForCausalLM,
    region: str,
) -> tuple[str, torch.nn.Module]:
    if region in {"", "forward", "model"}:
        return "forward", model
    candidates = [region]
    if isinstance(model, DeepSeekV32ForCausalLM) and not region.startswith("model."):
        candidates.append(f"model.{region}")
    errors = []
    for candidate in candidates:
        try:
            return candidate, model.get_submodule(candidate)
        except AttributeError as exc:
            errors.append(str(exc))
    raise ValueError(f"unknown region {region!r}; tried {candidates}. Last errors: {errors}")


def _build_pattern_workload(
    config: PatternProfileConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
    if config.pattern != "fused-rmsnorm-swiglu":
        raise ValueError("only pattern 'fused-rmsnorm-swiglu' is currently registered")
    shape = (config.batch_size, config.tokens, config.hidden_size)
    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    norm_weight = torch.randn(config.hidden_size, device=device, dtype=dtype)
    gate_weight = torch.randn(config.hidden_size, device=device, dtype=dtype)
    up_weight = torch.randn(config.hidden_size, device=device, dtype=dtype)

    def fn(
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        return fused_rmsnorm_swiglu_reference(
            x,
            residual,
            norm_weight,
            gate_weight,
            up_weight,
            eps=1e-6,
        )

    return fn, (x, residual, norm_weight, gate_weight, up_weight)


def _profile_callable(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    label: str,
    device: torch.device,
    iters: int,
    capture_profiler: bool,
    with_stack: bool,
    with_flops: bool,
    warmup: int = 0,
) -> tuple[Any, float, torch.profiler.profile | None]:
    if iters < 1:
        raise ValueError("iters must be positive")
    for _ in range(warmup):
        with torch.inference_mode():
            fn(*args)
    _sync_if_needed(device)
    start = time.perf_counter()
    profiler = None
    result: Any = None
    if capture_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            with_flops=with_flops,
            acc_events=True,
        ) as prof:
            for _ in range(iters):
                with torch.profiler.record_function(label), torch.inference_mode():
                    result = fn(*args)
                prof.step()
        profiler = prof
    else:
        for _ in range(iters):
            with torch.inference_mode():
                result = fn(*args)
    _sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms, profiler


def _write_graph_artifacts(output_dir: Path, prefix: str, graph_module: torch.fx.GraphModule) -> None:
    _write_json(output_dir / f"{prefix}_graph.json", _graph_to_json(graph_module))
    (output_dir / f"{prefix}_graph.txt").write_text(str(graph_module.graph) + "\n")
    (output_dir / f"{prefix}_graph_module.py").write_text(graph_module.code)


def _load_profile_run_config(path: Path) -> ProfileRunConfig:
    payload = json.loads(path.read_text())
    field_names = {field.name for field in fields(ProfileRunConfig)}
    kwargs = {key: value for key, value in payload.items() if key in field_names}
    kwargs["output_dir"] = Path(kwargs.get("output_dir", path.parent))
    kwargs["command"] = tuple(kwargs.get("command", ()))
    return ProfileRunConfig(**kwargs)


def _load_profile_input_ids(
    source_run_dir: Path,
    source_config: ProfileRunConfig,
    device: torch.device,
) -> torch.Tensor:
    input_path = source_run_dir / "input_ids.json"
    if input_path.exists():
        payload = json.loads(input_path.read_text())
        return torch.tensor(payload["input_ids"], device=device, dtype=torch.long)
    return torch.randint(
        0,
        source_config.vocab_size,
        (source_config.batch_size, source_config.prompt_tokens),
        device=device,
        dtype=torch.long,
    )


def _capture_graph_node_values(
    graph_module: torch.fx.GraphModule,
    args: tuple[Any, ...],
    stop_after_names: set[str] | None = None,
) -> dict[str, Any]:
    class CaptureComplete(Exception):
        pass

    class CaptureInterpreter(torch.fx.Interpreter):
        def __init__(self, module: torch.fx.GraphModule) -> None:
            super().__init__(module)
            self.values: dict[str, Any] = {}

        def run_node(self, node: torch.fx.Node) -> Any:
            result = super().run_node(node)
            self.values[node.name] = result
            if stop_after_names is not None and stop_after_names.issubset(self.values):
                raise CaptureComplete
            return result

    interpreter = CaptureInterpreter(graph_module)
    try:
        interpreter.run(*args)
    except CaptureComplete:
        pass
    if stop_after_names is not None:
        missing = sorted(stop_after_names - interpreter.values.keys())
        if missing:
            raise RuntimeError(f"failed to capture requested graph node values: {missing}")
    return interpreter.values


def _extract_fx_subgraph(
    graph_module: torch.fx.GraphModule,
    requested_node_ids: tuple[int, ...],
) -> _ExtractedSubgraph:
    source_ids = _node_id_map(graph_module)
    id_to_node = {node_id: node for node, node_id in source_ids.items()}
    unique_ids = tuple(dict.fromkeys(requested_node_ids))
    missing = [node_id for node_id in unique_ids if node_id not in id_to_node]
    if missing:
        raise ValueError(f"unknown graph node ids: {missing}")

    requested_nodes = tuple(id_to_node[node_id] for node_id in unique_ids)
    selected_set = {node for node in requested_nodes if node.op not in {"placeholder", "output"}}
    compute_nodes = tuple(node for node in requested_nodes if node.op not in {"placeholder", "output", "get_attr"})
    if not compute_nodes:
        raise ValueError("subgraph selection must include at least one compute node")

    new_graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    boundary_by_source: dict[torch.fx.Node, _SubgraphBoundary] = {}

    def copy_get_attr(source: torch.fx.Node) -> torch.fx.Node:
        if source not in env:
            copied = new_graph.get_attr(source.target)
            copied.meta = dict(source.meta)
            env[source] = copied
        return env[source]

    def boundary_for(source: torch.fx.Node) -> torch.fx.Node:
        if source in env:
            return env[source]
        arg_name = f"in_{source.name}"
        placeholder = new_graph.placeholder(arg_name)
        placeholder.meta = dict(source.meta)
        env[source] = placeholder
        boundary_by_source[source] = _SubgraphBoundary(
            arg_name=arg_name,
            source_id=source_ids[source],
            source_name=source.name,
            source_op=source.op,
            source_target=str(source.target),
        )
        return placeholder

    def map_input(value: torch.fx.Node) -> torch.fx.Node:
        if value in env:
            return env[value]
        if value in selected_set:
            raise ValueError(f"selected node {value.name!r} is used before it is copied")
        if value.op == "get_attr":
            return copy_get_attr(value)
        return boundary_for(value)

    for source in graph_module.graph.nodes:
        if source not in selected_set:
            continue
        if source.op == "get_attr":
            copy_get_attr(source)
            continue
        copied = new_graph.node_copy(source, map_input)
        copied.meta = dict(source.meta)
        env[source] = copied

    output_nodes = tuple(
        node
        for node in requested_nodes
        if node.op not in {"placeholder", "output", "get_attr"}
        and not any(user in selected_set and user.op != "output" for user in node.users)
    )
    if not output_nodes:
        output_nodes = (compute_nodes[-1],)
    if len(output_nodes) == 1:
        new_graph.output(env[output_nodes[0]])
    else:
        new_graph.output(tuple(env[node] for node in output_nodes))
    new_graph.lint()
    subgraph_module = torch.fx.GraphModule(graph_module, new_graph)
    subgraph_module.recompile()
    return _ExtractedSubgraph(
        graph_module=subgraph_module,
        boundaries=tuple(boundary_by_source.values()),
        output_nodes=output_nodes,
        selected_nodes=requested_nodes,
    )


def _node_id_map(graph_module: torch.fx.GraphModule) -> dict[torch.fx.Node, int]:
    return {node: node_id for node_id, node in enumerate(graph_module.graph.nodes)}


def _node_summary(node: torch.fx.Node, node_id: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "name": node.name,
        "op": node.op,
        "target": str(node.target),
    }


def _max_abs_diff(left: Any, right: Any) -> float | None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.numel() == 0 and right.numel() == 0:
            return 0.0
        return float((left.detach().float() - right.detach().float()).abs().max().item())
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) and len(left) == len(right):
        values = [_max_abs_diff(l_item, r_item) for l_item, r_item in zip(left, right)]
        values = [value for value in values if value is not None]
        return max(values) if values else None
    if isinstance(left, dict) and isinstance(right, dict):
        values = [_max_abs_diff(left[key], right[key]) for key in left.keys() & right.keys()]
        values = [value for value in values if value is not None]
        return max(values) if values else None
    return None


def _plain_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    return repr(value)


def _value_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        flat = tensor.flatten()
        return {
            **_tensor_summary(tensor),
            "numel": int(tensor.numel()),
            "sample": flat[: min(8, flat.numel())].detach().cpu().tolist(),
        }
    if isinstance(value, tuple):
        return [_value_summary(item) for item in value]
    if isinstance(value, list):
        return [_value_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value_summary(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _region_config_to_json(config: RegionProfileConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["command"] = list(config.command)
    return payload


def _pattern_config_to_json(config: PatternProfileConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["command"] = list(config.command)
    return payload


def _subgraph_config_to_json(config: SubgraphProfileConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["source_run_dir"] = str(config.source_run_dir)
    payload["node_ids"] = list(config.node_ids)
    payload["command"] = list(config.command)
    return payload


def _write_region_repro(path: Path, config: RegionProfileConfig) -> None:
    source = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path({str(Path.cwd())!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.profiling import RegionProfileConfig, run_region_profile_capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="region_repro_artifacts")
    parser.add_argument("--device", default={config.device!r})
    args = parser.parse_args()
    artifacts = run_region_profile_capture(
        RegionProfileConfig(
            output_dir=Path(args.output_dir),
            region={config.region!r},
            model_kind={config.model_kind!r},
            device=args.device,
            dtype={config.dtype!r},
            seed={config.seed!r},
            batch_size={config.batch_size!r},
            tokens={config.tokens!r},
            vocab_size={config.vocab_size!r},
            warmup={config.warmup!r},
            iters={config.iters!r},
            fake_graph={config.fake_graph!r},
            require_graph={config.require_graph!r},
            capture_profiler={config.capture_profiler!r},
            export_chrome_trace={config.export_chrome_trace!r},
            with_stack={config.with_stack!r},
            with_flops={config.with_flops!r},
        )
    )
    print(artifacts.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(source)
    path.chmod(0o755)


def _write_pattern_repro(path: Path, config: PatternProfileConfig) -> None:
    source = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path({str(Path.cwd())!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.profiling import PatternProfileConfig, run_pattern_profile_capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="pattern_repro_artifacts")
    parser.add_argument("--device", default={config.device!r})
    args = parser.parse_args()
    artifacts = run_pattern_profile_capture(
        PatternProfileConfig(
            output_dir=Path(args.output_dir),
            pattern={config.pattern!r},
            device=args.device,
            dtype={config.dtype!r},
            seed={config.seed!r},
            batch_size={config.batch_size!r},
            tokens={config.tokens!r},
            hidden_size={config.hidden_size!r},
            warmup={config.warmup!r},
            iters={config.iters!r},
            apply_passes={config.apply_passes!r},
            fake_graph={config.fake_graph!r},
            require_graph={config.require_graph!r},
            capture_profiler={config.capture_profiler!r},
            export_chrome_trace={config.export_chrome_trace!r},
            with_stack={config.with_stack!r},
            with_flops={config.with_flops!r},
        )
    )
    print(artifacts.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(source)
    path.chmod(0o755)


def _write_subgraph_repro(path: Path, config: SubgraphProfileConfig) -> None:
    source = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path({str(Path.cwd())!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.profiling import SubgraphProfileConfig, run_subgraph_profile_capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="subgraph_repro_artifacts")
    parser.add_argument("--device", default={config.device!r})
    parser.add_argument("--source-run", default={str(config.source_run_dir)!r})
    args = parser.parse_args()
    artifacts = run_subgraph_profile_capture(
        SubgraphProfileConfig(
            output_dir=Path(args.output_dir),
            source_run_dir=Path(args.source_run),
            node_ids={config.node_ids!r},
            device=args.device,
            warmup={config.warmup!r},
            iters={config.iters!r},
            capture_profiler={config.capture_profiler!r},
            export_chrome_trace={config.export_chrome_trace!r},
            with_stack={config.with_stack!r},
            with_flops={config.with_flops!r},
        )
    )
    print(artifacts.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(source)
    path.chmod(0o755)


def _write_focus_readme(path: Path, kind: str, artifacts: FocusProfileArtifacts) -> None:
    path.write_text(
        f"# TorchInferno {kind.title()} Profile\n\n"
        "Key files:\n\n"
        "- `region_spec.json`: resolved module and generated inputs.\n"
        "- `region_graph.json`: focused make_fx graph.\n"
        "- `operator_profile.json`: profiler key averages for only this region.\n"
        "- `chrome_trace.json`: trace viewer artifact when profiler export is enabled.\n"
        "- `memory_profile.json`: allocator snapshot before and after region profiling.\n"
        "- `region_repro.py`: rerun this profile in a fresh artifact directory.\n"
    )


def _write_pattern_readme(path: Path, artifacts: PatternProfileArtifacts) -> None:
    path.write_text(
        "# TorchInferno Pattern Profile\n\n"
        "Key files:\n\n"
        "- `pattern_spec.json`: pattern name and generated inputs.\n"
        "- `reference_graph.json`: make_fx graph before replacement.\n"
        "- `optimized_graph.json`: graph after registered passes.\n"
        "- `pass_report.json`: registered pass names and replacement match counts.\n"
        "- `reference_profile.json` and `optimized_profile.json`: profiler key averages.\n"
        "- `comparison.json`: timing, speedup, and max_abs_diff.\n"
        "- `pattern_repro.py`: rerun this profile in a fresh artifact directory.\n"
    )


def _write_subgraph_readme(path: Path, artifacts: SubgraphProfileArtifacts) -> None:
    path.write_text(
        "# TorchInferno Subgraph Profile\n\n"
        "Key files:\n\n"
        "- `source_graph.json`: re-traced full graph with stable integer node ids.\n"
        "- `subgraph_spec.json`: selected ids, boundary inputs, and output nodes.\n"
        "- `subgraph_graph.json`: extracted callable FX graph.\n"
        "- `operator_profile.json`: profiler key averages for only this subgraph.\n"
        "- `chrome_trace.json`: trace viewer artifact when profiler export is enabled.\n"
        "- `memory_profile.json`: allocator snapshot before and after subgraph profiling.\n"
        "- `subgraph_repro.py`: rerun this extraction/profile in a fresh artifact directory.\n"
    )


def _write_repro(path: Path, config: ProfileRunConfig, input_ids: list[list[int]]) -> None:
    source = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import torch

REPO_ROOT = Path({str(Path.cwd())!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config

MODEL_KIND = {config.model_kind!r}
DTYPE = {config.dtype!r}
SEED = {config.seed!r}
BATCH_SIZE = {config.batch_size!r}
PROMPT_TOKENS = {config.prompt_tokens!r}
NEW_TOKENS = {config.new_tokens!r}
VOCAB_SIZE = {config.vocab_size!r}
TEMPERATURE = {config.temperature!r}
COMPILE = {config.compile!r}
CACHE_BACKEND = {config.cache_backend!r}
PAGE_SIZE = {config.page_size!r}
INPUT_IDS = {input_ids!r}


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {{name}}")


def build_model():
    max_seq_len = PROMPT_TOKENS + NEW_TOKENS + 8
    if MODEL_KIND == "dsv4":
        return DSv4ForCausalLM(tiny_dsv4_config(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len))
    if MODEL_KIND == "deepseek":
        return DeepSeekV32ForCausalLM(
            tiny_deepseek_v32_config(vocab_size=VOCAB_SIZE, max_position_embeddings=max_seq_len)
        )
    raise ValueError(f"unknown model kind: {{MODEL_KIND}}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default={config.device!r})
    parser.add_argument("--output", default="repro_output.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        cuda_index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
        torch.cuda.manual_seed_all(SEED)
    model = build_model().to(device=device, dtype=dtype_from_name(DTYPE)).eval()
    if COMPILE:
        compile_forward(model, CompileConfig(mode="reduce-overhead"))
    input_ids = torch.tensor(INPUT_IDS, device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        if MODEL_KIND == "deepseek":
            output = model.generate(
                input_ids,
                max_new_tokens=NEW_TOKENS,
                temperature=TEMPERATURE,
                cache_backend=CACHE_BACKEND,
                page_size=PAGE_SIZE,
            )
        else:
            output = model.generate(input_ids, max_new_tokens=NEW_TOKENS, temperature=TEMPERATURE)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {{
        "shape": list(output.shape),
        "tokens": output.detach().cpu().tolist(),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }}
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(source)
    path.chmod(0o755)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {name}")


def _normalize_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    cuda_index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(cuda_index)
    return torch.device("cuda", cuda_index)


def _tokens_per_second(tokens: int, elapsed_ms: float) -> float:
    return 0.0 if elapsed_ms <= 0 else tokens / (elapsed_ms / 1000)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
