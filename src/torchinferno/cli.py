from __future__ import annotations

import argparse
import time

import torch

from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.graph import trace_with_make_fx
from torchinferno.models.conversion import (
    IncompatibleCheckpointError,
    audit_deepseek_checkpoint,
    convert_deepseek_checkpoint,
    dtype_from_name,
)
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.research import ExperimentResult, ResearchHarness
from torchinferno.runtime.batching import InferenceRequest, run_continuous_batch
from torchinferno.runtime.scheduler import DisaggregatedPrefillDecodeSimulator, InferenceJob


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dsv4_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_dsv4_config(
        vocab_size=args.vocab_size,
        max_seq_len=args.prompt_tokens + args.new_tokens + 8,
    )
    model = DSv4ForCausalLM(config).to(device).eval()
    if args.compile:
        compile_forward(model, CompileConfig(mode="reduce-overhead"))

    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_tokens), device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(prompt, max_new_tokens=args.new_tokens, temperature=args.temperature)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("TorchInferno DSv4 smoke")
    print(f"device={device} batch={args.batch_size} prompt={args.prompt_tokens} new={args.new_tokens}")
    print(f"shape={tuple(output.shape)} elapsed_ms={elapsed_ms:.2f}")
    print(f"tokens[0]={output[0].tolist()}")
    return 0


def run_batch_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_dsv4_config(max_seq_len=64)
    model = DSv4ForCausalLM(config).to(device).eval()
    requests = [
        InferenceRequest("req-a", (1, 2, 3), 2),
        InferenceRequest("req-b", (4, 5), 3),
        InferenceRequest("req-c", (6, 7, 8), 1),
    ]
    with torch.inference_mode():
        results = run_continuous_batch(model, requests, device=device)
    for result in results:
        print(f"{result.request_id}: {list(result.tokens)}")
    return 0


def run_hf_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    model = DSv4ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        map_location="cpu",
        strict=not args.non_strict,
    ).to(device)
    model.eval()
    prompt = torch.tensor([args.input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        output = model.generate(prompt, max_new_tokens=args.new_tokens, temperature=args.temperature)
    print("TorchInferno DSv4 HF smoke")
    print(f"model={args.model} device={device} new={args.new_tokens}")
    print(f"shape={tuple(output.shape)}")
    print(f"tokens[0]={output[0].tolist()}")
    return 0


def run_trace_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else torch.device("cpu")
    torch.manual_seed(args.seed)
    config = tiny_dsv4_config(max_seq_len=args.tokens + 4)
    model = DSv4ForCausalLM(config).to(device).eval()
    attention = model.layers[0].attn
    x = torch.randn(1, args.tokens, config.hidden_size, device=device)
    positions = torch.arange(args.tokens, device=device)

    def forward_attention(hidden: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return attention(hidden, pos, None)

    graph_module = trace_with_make_fx(forward_attention, x, positions, fake=args.fake)
    node_count = sum(1 for _ in graph_module.graph.nodes)
    print("TorchInferno trace smoke")
    print(f"device={device} fake={args.fake} nodes={node_count}")
    if args.print_graph:
        print(graph_module.graph)
    return 0


def run_sim_smoke(args: argparse.Namespace) -> int:
    simulator = DisaggregatedPrefillDecodeSimulator(
        prefill_ranks=(0,),
        decode_ranks=(1,),
        prefill_us_per_token=args.prefill_us_per_token,
        decode_us_per_token=args.decode_us_per_token,
        network_latency_us=args.network_latency_us,
    )
    jobs = [
        InferenceJob("req-a", prompt_tokens=8, decode_tokens=2, arrival_us=0),
        InferenceJob("req-b", prompt_tokens=3, decode_tokens=4, arrival_us=args.arrival_gap_us),
    ]
    stages = simulator.plan(jobs)
    print("TorchInferno disaggregated simulation smoke")
    for stage in stages:
        print(
            f"{stage.request_id} {stage.stage} rank={stage.rank} "
            f"start_us={stage.start_us:.1f} end_us={stage.end_us:.1f}"
        )
    return 0


def run_research_smoke(args: argparse.Namespace) -> int:
    harness = ResearchHarness()

    def baseline() -> ExperimentResult:
        simulator = DisaggregatedPrefillDecodeSimulator(
            prefill_ranks=(0,),
            decode_ranks=(1,),
            prefill_us_per_token=2.0,
            decode_us_per_token=4.0,
            network_latency_us=10.0,
        )
        stages = simulator.plan([InferenceJob("req", prompt_tokens=8, decode_tokens=4)])
        return ExperimentResult("baseline", {"total_us": max(stage.end_us for stage in stages)})

    def faster_decode() -> ExperimentResult:
        simulator = DisaggregatedPrefillDecodeSimulator(
            prefill_ranks=(0,),
            decode_ranks=(1,),
            prefill_us_per_token=2.0,
            decode_us_per_token=2.0,
            network_latency_us=10.0,
        )
        stages = simulator.plan([InferenceJob("req", prompt_tokens=8, decode_tokens=4)])
        return ExperimentResult("faster-decode", {"total_us": max(stage.end_us for stage in stages)})

    harness.register("baseline", baseline)
    harness.register("faster-decode", faster_decode)
    results = harness.run()
    best = harness.best(results, "total_us")
    print("TorchInferno research smoke")
    for result in results:
        print(f"{result.name}: total_us={result.metrics['total_us']:.1f}")
    print(f"best={best.name}")
    return 0


def run_dsv4_audit(args: argparse.Namespace) -> int:
    report = audit_deepseek_checkpoint(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(report.summary())
    return 0 if report.compatible else 2


def run_dsv4_convert(args: argparse.Namespace) -> int:
    try:
        report = convert_deepseek_checkpoint(
            args.model,
            args.output_dir,
            revision=args.revision,
            cache_dir=args.cache_dir,
            dtype=dtype_from_name(args.dtype),
            max_shard_size=args.max_shard_size,
            allow_partial=args.allow_partial,
        )
    except IncompatibleCheckpointError as exc:
        print(exc.report.summary())
        print("Refusing to convert incompatible checkpoint. Use --allow-partial only for debugging.")
        return 2
    print(report.summary())
    print(f"wrote={args.output_dir}")
    return 0 if report.compatible else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchinferno")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("dsv4-smoke", help="Run a DSv4 end-to-end generation smoke test.")
    smoke.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--prompt-tokens", type=int, default=8)
    smoke.add_argument("--new-tokens", type=int, default=4)
    smoke.add_argument("--vocab-size", type=int, default=128)
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--compile", action="store_true", help="Compile the DSv4 forward path with torch.compile.")
    smoke.set_defaults(func=run_dsv4_smoke)

    batch = subparsers.add_parser("batch-smoke", help="Run the ragged request batching harness.")
    batch.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    batch.add_argument("--seed", type=int, default=0)
    batch.set_defaults(func=run_batch_smoke)

    hf_smoke = subparsers.add_parser(
        "dsv4-hf-smoke",
        help="Load a compatible local or Hugging Face DSv4 checkpoint and generate tokens.",
    )
    hf_smoke.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    hf_smoke.add_argument("--revision", default=None)
    hf_smoke.add_argument("--cache-dir", default=None)
    hf_smoke.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    hf_smoke.add_argument("--input-ids", type=int, nargs="+", default=[1, 2, 3])
    hf_smoke.add_argument("--new-tokens", type=int, default=2)
    hf_smoke.add_argument("--temperature", type=float, default=0.0)
    hf_smoke.add_argument("--non-strict", action="store_true", help="Allow missing or unexpected weight keys.")
    hf_smoke.set_defaults(func=run_hf_smoke)

    trace = subparsers.add_parser("trace-smoke", help="Trace a DSv4 attention slice with make_fx.")
    trace.add_argument("--device", default="cpu")
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--tokens", type=int, default=3)
    trace.add_argument("--fake", action="store_true", help="Trace with FakeTensorMode.")
    trace.add_argument("--print-graph", action="store_true", help="Print the full FX graph.")
    trace.set_defaults(func=run_trace_smoke)

    sim = subparsers.add_parser("sim-smoke", help="Plan disaggregated prefill/decode on virtual ranks.")
    sim.add_argument("--prefill-us-per-token", type=float, default=2.0)
    sim.add_argument("--decode-us-per-token", type=float, default=4.0)
    sim.add_argument("--network-latency-us", type=float, default=10.0)
    sim.add_argument("--arrival-gap-us", type=float, default=5.0)
    sim.set_defaults(func=run_sim_smoke)

    research = subparsers.add_parser("research-smoke", help="Run a tiny auto research harness.")
    research.set_defaults(func=run_research_smoke)

    audit = subparsers.add_parser("dsv4-audit", help="Audit a DeepSeek-style checkpoint for DSv4 conversion.")
    audit.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    audit.add_argument("--revision", default=None)
    audit.add_argument("--cache-dir", default=None)
    audit.set_defaults(func=run_dsv4_audit)

    convert = subparsers.add_parser(
        "dsv4-convert",
        help="Convert a compatible DeepSeek-style checkpoint into TorchInferno DSv4 format.",
    )
    convert.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    convert.add_argument("output_dir")
    convert.add_argument("--revision", default=None)
    convert.add_argument("--cache-dir", default=None)
    convert.add_argument("--dtype", default=None, help="Optional output dtype: float32, float16, or bfloat16.")
    convert.add_argument("--max-shard-size", default="5GB")
    convert.add_argument("--allow-partial", action="store_true", help="Write only convertible tensors for debugging.")
    convert.set_defaults(func=run_dsv4_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
