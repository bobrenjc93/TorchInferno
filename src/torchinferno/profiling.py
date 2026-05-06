from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Literal

import torch

from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.graph import trace_with_make_fx
from torchinferno.kernels.ops import triton_available
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
    for node in graph_module.graph.nodes:
        nodes.append(
            {
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


def _tokens_per_second(tokens: int, elapsed_ms: float) -> float:
    return 0.0 if elapsed_ms <= 0 else tokens / (elapsed_ms / 1000)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
