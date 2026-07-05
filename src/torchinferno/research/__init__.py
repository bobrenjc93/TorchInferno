from torchinferno.research.benchmarks import BenchmarkResult, benchmark_callable
from torchinferno.research.harness import ExperimentResult, ResearchHarness
from torchinferno.research.helion import (
    FXNodeRef,
    FXWindowCandidate,
    HelionCandidateConfig,
    HelionCandidateReport,
    HelionDecisionStore,
    HelionFXSearchReport,
    HelionRegionSearchConfig,
    HelionRegionSearchReport,
    MacroRegionCandidate,
    discover_fx_windows,
    run_helion_candidate,
    run_helion_fx_search,
    run_helion_region_search,
    trace_helion_candidate,
)

_INFERENCE_BENCH_EXPORTS = {
    "InferenceBenchRunSummary",
    "ProviderBenchmarkSummary",
    "QueueProfileSummary",
    "format_inference_bench_summary",
    "summarize_inference_bench_run",
}


def __getattr__(name: str) -> object:
    if name in _INFERENCE_BENCH_EXPORTS:
        from torchinferno.research import inference_bench

        value = getattr(inference_bench, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BenchmarkResult",
    "ExperimentResult",
    "FXNodeRef",
    "FXWindowCandidate",
    "HelionCandidateConfig",
    "HelionCandidateReport",
    "HelionDecisionStore",
    "HelionFXSearchReport",
    "HelionRegionSearchConfig",
    "HelionRegionSearchReport",
    "InferenceBenchRunSummary",
    "MacroRegionCandidate",
    "ProviderBenchmarkSummary",
    "QueueProfileSummary",
    "ResearchHarness",
    "benchmark_callable",
    "discover_fx_windows",
    "format_inference_bench_summary",
    "run_helion_candidate",
    "run_helion_fx_search",
    "run_helion_region_search",
    "summarize_inference_bench_run",
    "trace_helion_candidate",
]
