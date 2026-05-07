from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from torchinferno.graph import trace_with_make_fx
from torchinferno.kernels import KernelBackend, KernelConfig, swiglu_activation
from torchinferno.kernels.ops import helion_available
from torchinferno.research.benchmarks import BenchmarkResult, benchmark_callable


@dataclass(frozen=True)
class HelionCandidateConfig:
    candidate: str = "swiglu"
    batch_size: int = 1000
    tokens: int = 32
    hidden_size: int = 3584
    dtype: str = "bfloat16"
    device: str | None = None
    warmup: int = 10
    iters: int = 50
    seed: int = 0
    min_speedup: float = 1.02
    atol: float = 2e-2
    rtol: float = 2e-2


@dataclass(frozen=True)
class HelionRegionSearchConfig:
    model_kind: str = "deepseek"
    region: str = "mlp"
    batch_size: int = 2
    tokens: int = 8
    hidden_size: int = 64
    intermediate_size: int = 128
    dtype: str = "float32"
    device: str | None = None
    trace_device: str = "cpu"
    warmup: int = 5
    iters: int = 20
    seed: int = 0
    min_speedup: float = 1.02
    atol: float = 2e-2
    rtol: float = 2e-2


@dataclass(frozen=True)
class FXNodeRef:
    name: str
    op: str
    target: str


@dataclass(frozen=True)
class FXWindowCandidate:
    candidate_id: str
    nodes: tuple[FXNodeRef, ...]
    supported_kernel: str | None
    reason: str


@dataclass(frozen=True)
class MacroRegionCandidate:
    name: str
    status: str
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class HelionCandidateReport:
    candidate: str
    candidate_id: str
    status: str
    reason: str
    promoted: bool
    baseline: BenchmarkResult | None
    torch_compile: BenchmarkResult | None
    helion: BenchmarkResult | None
    speedup: float
    speedup_vs_compile: float
    correct: bool
    max_abs_error: float
    shape: tuple[int, int, int]
    dtype: str
    device: str
    min_speedup: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["baseline"] = asdict(self.baseline) if self.baseline is not None else None
        data["torch_compile"] = asdict(self.torch_compile) if self.torch_compile is not None else None
        data["helion"] = asdict(self.helion) if self.helion is not None else None
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class HelionFXSearchReport:
    candidate: str
    windows: tuple[FXWindowCandidate, ...]
    reports: tuple[HelionCandidateReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "windows": [asdict(window) for window in self.windows],
            "reports": [report.to_dict() for report in self.reports],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class HelionRegionSearchReport:
    model_kind: str
    region: str
    candidate_activation_shape: tuple[int, int, int]
    windows: tuple[FXWindowCandidate, ...]
    macro_candidates: tuple[MacroRegionCandidate, ...]
    reports: tuple[HelionCandidateReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_kind": self.model_kind,
            "region": self.region,
            "candidate_activation_shape": self.candidate_activation_shape,
            "windows": [asdict(window) for window in self.windows],
            "macro_candidates": [asdict(candidate) for candidate in self.macro_candidates],
            "reports": [report.to_dict() for report in self.reports],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class HelionDecisionStore:
    """Append-only memory for Helion search decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, report: HelionCandidateReport) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")

    def append_many(self, reports: Iterable[HelionCandidateReport]) -> None:
        for report in reports:
            self.append(report)

    def read(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def run_helion_candidate(config: HelionCandidateConfig) -> HelionCandidateReport:
    if config.candidate != "swiglu":
        raise ValueError("only the 'swiglu' Helion candidate is currently registered")
    return run_helion_swiglu_candidate(config)


def run_helion_fx_search(
    config: HelionCandidateConfig,
    *,
    min_nodes: int = 1,
    max_nodes: int = 5,
) -> HelionFXSearchReport:
    graph_module = trace_helion_candidate(config)
    windows = discover_fx_windows(graph_module, min_nodes=min_nodes, max_nodes=max_nodes)
    reports = []
    for window in windows:
        if window.supported_kernel == config.candidate:
            reports.append(run_helion_swiglu_candidate(config, candidate_id=window.candidate_id))
    return HelionFXSearchReport(config.candidate, tuple(windows), tuple(reports))


def run_helion_region_search(
    config: HelionRegionSearchConfig,
    *,
    min_nodes: int = 1,
    max_nodes: int = 5,
) -> HelionRegionSearchReport:
    workload = _build_region_workload(config)
    graph_module = trace_with_make_fx(
        workload["fn"],
        *workload["args"],
        fake=workload["device"].type == "cpu",
    )
    windows = discover_fx_windows(graph_module, min_nodes=min_nodes, max_nodes=max_nodes)
    macro_candidates = _discover_macro_candidates(graph_module, config)
    reports = []
    for window in windows:
        if window.supported_kernel == "swiglu":
            candidate_config = HelionCandidateConfig(
                candidate="swiglu",
                batch_size=config.batch_size,
                tokens=config.tokens,
                hidden_size=workload["candidate_activation_size"],
                dtype=config.dtype,
                device=config.device,
                warmup=config.warmup,
                iters=config.iters,
                seed=config.seed,
                min_speedup=config.min_speedup,
                atol=config.atol,
                rtol=config.rtol,
            )
            reports.append(run_helion_swiglu_candidate(candidate_config, candidate_id=window.candidate_id))
    return HelionRegionSearchReport(
        model_kind=config.model_kind,
        region=config.region,
        candidate_activation_shape=(config.batch_size, config.tokens, workload["candidate_activation_size"]),
        windows=windows,
        macro_candidates=macro_candidates,
        reports=tuple(reports),
    )


def trace_helion_candidate(config: HelionCandidateConfig) -> torch.fx.GraphModule:
    if config.candidate != "swiglu":
        raise ValueError("only the 'swiglu' Helion candidate is currently registered")
    device = torch.device(config.device or "cpu")
    dtype = _resolve_dtype(config.dtype)
    gate = torch.randn(
        config.batch_size,
        config.tokens,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )
    up = torch.randn_like(gate)

    def fn(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up

    return trace_with_make_fx(fn, gate, up, fake=device.type == "cpu")


def discover_fx_windows(
    graph_module: torch.fx.GraphModule,
    *,
    min_nodes: int = 1,
    max_nodes: int = 5,
) -> tuple[FXWindowCandidate, ...]:
    if min_nodes < 1 or max_nodes < min_nodes:
        raise ValueError("expected 1 <= min_nodes <= max_nodes")
    nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op in {"call_function", "call_method", "call_module"}
    ]
    windows = []
    for start in range(len(nodes)):
        for end in range(start + min_nodes, min(len(nodes), start + max_nodes) + 1):
            window_nodes = tuple(_node_ref(node) for node in nodes[start:end])
            supported_kernel, reason = _classify_window(nodes[start:end])
            windows.append(
                FXWindowCandidate(
                    candidate_id=_candidate_id(window_nodes),
                    nodes=window_nodes,
                    supported_kernel=supported_kernel,
                    reason=reason,
                )
            )
    return tuple(windows)


def run_helion_swiglu_candidate(
    config: HelionCandidateConfig,
    *,
    candidate_id: str | None = None,
) -> HelionCandidateReport:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shape = (config.batch_size, config.tokens, config.hidden_size)
    candidate_id = candidate_id or "manual:swiglu"
    if device.type != "cuda":
        return _unavailable(config, candidate_id, shape, str(device), "Helion candidate kernels require CUDA")
    if not helion_available():
        return _unavailable(config, candidate_id, shape, str(device), "Helion is not installed")

    from torchinferno.kernels.helion_ops import helion_swiglu_activation

    dtype = _resolve_dtype(config.dtype)
    torch.manual_seed(config.seed)
    gate = torch.randn(shape, device=device, dtype=dtype)
    up = torch.randn_like(gate)
    reference = F.silu(gate) * up
    candidate_out = helion_swiglu_activation(gate, up)
    diff = (candidate_out - reference).abs()
    max_abs_error = float(diff.max().item()) if diff.numel() else 0.0
    correct = bool(torch.allclose(candidate_out, reference, atol=config.atol, rtol=config.rtol))

    baseline = benchmark_callable(
        "status-quo-swiglu",
        lambda: swiglu_activation(gate, up, config=KernelConfig(backend=KernelBackend.AUTO)),
        warmup=config.warmup,
        iters=config.iters,
        device=device,
    )
    compiled = _benchmark_compiled_swiglu(gate, up, config, device)
    helion = benchmark_callable(
        "helion-swiglu-candidate",
        lambda: helion_swiglu_activation(gate, up),
        warmup=config.warmup,
        iters=config.iters,
        device=device,
    )
    comparison_baselines = [baseline, *(benchmark for benchmark in (compiled,) if benchmark is not None)]
    best_baseline = min(comparison_baselines, key=lambda result: result.mean_ms)
    speedup = best_baseline.mean_ms / helion.mean_ms if helion.mean_ms > 0 else 0.0
    speedup_vs_compile = compiled.mean_ms / helion.mean_ms if compiled is not None and helion.mean_ms > 0 else 0.0
    promoted = correct and speedup >= config.min_speedup
    if not correct:
        status = "reject"
        reason = "Helion candidate failed correctness tolerance against torch reference"
    elif promoted:
        status = "promote"
        reason = f"Helion candidate beat best baseline ({best_baseline.name}) by the configured speedup threshold"
    else:
        status = "reject"
        reason = f"Helion candidate did not beat best baseline ({best_baseline.name}) by the configured speedup threshold"
    return HelionCandidateReport(
        candidate=config.candidate,
        candidate_id=candidate_id,
        status=status,
        reason=reason,
        promoted=promoted,
        baseline=baseline,
        torch_compile=compiled,
        helion=helion,
        speedup=speedup,
        speedup_vs_compile=speedup_vs_compile,
        correct=correct,
        max_abs_error=max_abs_error,
        shape=shape,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        min_speedup=config.min_speedup,
    )


def _benchmark_compiled_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    config: HelionCandidateConfig,
    device: torch.device,
) -> BenchmarkResult | None:
    def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up

    try:
        compiled = torch.compile(reference, fullgraph=True, options={"triton.cudagraphs": False})
        return benchmark_callable(
            "torch-compile-swiglu",
            lambda: compiled(gate, up),
            warmup=config.warmup,
            iters=config.iters,
            device=device,
        )
    except Exception:
        return None


def _unavailable(
    config: HelionCandidateConfig,
    candidate_id: str,
    shape: tuple[int, int, int],
    device: str,
    reason: str,
) -> HelionCandidateReport:
    return HelionCandidateReport(
        candidate=config.candidate,
        candidate_id=candidate_id,
        status="unavailable",
        reason=reason,
        promoted=False,
        baseline=None,
        torch_compile=None,
        helion=None,
        speedup=0.0,
        speedup_vs_compile=0.0,
        correct=False,
        max_abs_error=0.0,
        shape=shape,
        dtype=config.dtype,
        device=device,
        min_speedup=config.min_speedup,
    )


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _build_region_workload(config: HelionRegionSearchConfig) -> dict[str, object]:
    device = torch.device(config.trace_device)
    dtype = _resolve_dtype(config.dtype)
    torch.manual_seed(config.seed)
    model_kind = config.model_kind.lower()
    region = config.region.lower()
    if model_kind == "dsv4":
        from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config

        model = DSv4ForCausalLM(
            tiny_dsv4_config(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                max_seq_len=max(16, config.tokens + 4),
            )
        ).to(device=device, dtype=dtype).eval()
        if region in {"mlp", "expert", "swiglu"}:
            module = model.layers[0].moe.experts[0]
            x = torch.randn(config.batch_size, config.tokens, config.hidden_size, device=device, dtype=dtype)
            return {"fn": module, "args": (x,), "device": device, "candidate_activation_size": config.intermediate_size}
        if region in {"attention", "attn"}:
            module = model.layers[0].attn
            x = torch.randn(config.batch_size, config.tokens, config.hidden_size, device=device, dtype=dtype)
            positions = torch.arange(config.tokens, device=device)
            return {"fn": lambda x, positions: module(x, positions, None), "args": (x, positions), "device": device, "candidate_activation_size": config.hidden_size}
    elif model_kind in {"deepseek", "deepseek-v3.2", "deepseek_v32"}:
        from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config

        model = DeepSeekV32ForCausalLM(
            tiny_deepseek_v32_config(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                moe_intermediate_size=max(1, config.intermediate_size // 4),
                max_position_embeddings=max(16, config.tokens + 4),
            )
        ).to(device=device, dtype=dtype).eval()
        if region in {"mlp", "expert", "swiglu"}:
            module = model.model.layers[0].mlp
            x = torch.randn(config.batch_size, config.tokens, config.hidden_size, device=device, dtype=dtype)
            return {"fn": module, "args": (x,), "device": device, "candidate_activation_size": config.intermediate_size}
        if region in {"attention", "attn"}:
            module = model.model.layers[0].self_attn
            x = torch.randn(config.batch_size, config.tokens, config.hidden_size, device=device, dtype=dtype)
            positions = torch.arange(config.tokens, device=device)
            fn = lambda x, positions: module(x, positions, None, None)
            q_out = model.config.num_attention_heads * model.config.qk_head_dim
            return {"fn": fn, "args": (x, positions), "device": device, "candidate_activation_size": q_out}
    raise ValueError(
        "unsupported Helion region search target; expected model_kind dsv4/deepseek "
        "and region mlp/expert/attention"
    )


def _discover_macro_candidates(
    graph_module: torch.fx.GraphModule,
    config: HelionRegionSearchConfig,
) -> tuple[MacroRegionCandidate, ...]:
    targets = tuple(_target_name(node.target) for node in graph_module.graph.nodes)
    lower_targets = tuple(target.lower() for target in targets)
    candidates: list[MacroRegionCandidate] = []
    if config.region.lower() in {"attention", "attn"}:
        has_norm = any("rms_norm" in target or "native_rms_norm" in target for target in lower_targets)
        has_rope = any("cos" in target or "sin" in target or "cat" in target for target in lower_targets)
        has_attention = any("matmul" in target or "bmm" in target or "scaled_dot_product" in target for target in lower_targets)
        if config.model_kind.lower().startswith("deepseek"):
            status = "candidate"
            reason = (
                "DeepSeek attention trace contains MLA-style projection/rope/attention structure; "
                "a vLLM-style fused norm/rope/cache/indexer generator should be tried here."
            )
        else:
            status = "candidate" if has_rope and has_attention else "inspect"
            reason = "Attention trace contains RoPE/attention ops that may benefit from macro fusion."
        evidence = tuple(target for target in targets if any(key in target.lower() for key in ("rms", "split", "cat", "matmul", "softmax", "sin", "cos")))[:20]
        candidates.append(MacroRegionCandidate("attention-rope-cache-macro", status, reason, evidence))
        if has_norm and has_rope:
            candidates.append(
                MacroRegionCandidate(
                    "fused-norm-rope",
                    "candidate",
                    "Norm and RoPE-like ops appear in the same attention region.",
                    tuple(target for target in targets if "rms" in target.lower() or "cos" in target.lower() or "sin" in target.lower())[:20],
                )
            )
    if config.region.lower() in {"mlp", "expert", "swiglu"}:
        supported_windows = discover_fx_windows(graph_module, min_nodes=2, max_nodes=2)
        if any(window.supported_kernel == "swiglu" for window in supported_windows):
            candidates.append(
                MacroRegionCandidate(
                    "mlp-swiglu-activation",
                    "candidate",
                    "MLP trace contains a SwiGLU activation window; benchmark Helion against the current activation kernel.",
                    tuple(target for target in targets if "silu" in target.lower() or "mul" in target.lower())[:20],
                )
            )
    return tuple(candidates)


def _node_ref(node: torch.fx.Node) -> FXNodeRef:
    return FXNodeRef(name=node.name, op=node.op, target=_target_name(node.target))


def _candidate_id(nodes: tuple[FXNodeRef, ...]) -> str:
    payload = json.dumps([asdict(node) for node in nodes], sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"fx:{digest}"


def _classify_window(nodes: list[torch.fx.Node]) -> tuple[str | None, str]:
    targets = [_target_name(node.target) for node in nodes]
    if len(nodes) == 2 and _is_silu_target(targets[0]) and _is_mul_target(targets[1]):
        if nodes[0] in _iter_node_args(nodes[1]):
            return "swiglu", "matches silu -> mul SwiGLU window"
    return None, "no registered Helion candidate generator for this FX window"


def _iter_node_args(node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
    found: list[torch.fx.Node] = []

    def visit(value: object) -> None:
        if isinstance(value, torch.fx.Node):
            found.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(node.args)
    visit(node.kwargs)
    return tuple(found)


def _target_name(target: object) -> str:
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    return str(target)


def _is_silu_target(target: str) -> bool:
    return "silu" in target.lower()


def _is_mul_target(target: str) -> bool:
    return "mul" in target.lower()
