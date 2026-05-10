from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch

from torchinferno.audit import build_audit_report
from torchinferno.benchmarks import (
    LLAMA3_70B_MODEL,
    OpenAIServerMicrobenchConfig,
    TorchInfernoLlamaBenchmarkConfig,
    VLLMBenchmarkConfig,
    format_openai_server_microbench_report,
    plot_vllm_benchmark_results,
    run_openai_server_microbench,
    run_torchinferno_llama_benchmark_suite,
    run_vllm_benchmark_suite,
)
from torchinferno.compiler import CompileConfig, compile_forward
from torchinferno.graph import trace_with_make_fx
from torchinferno.models.auto import load_model_auto
from torchinferno.models.conversion import (
    IncompatibleCheckpointError,
    audit_deepseek_checkpoint,
    audit_native_deepseek_checkpoint,
    convert_deepseek_checkpoint,
    convert_native_deepseek_checkpoint,
    dtype_from_name,
)
from torchinferno.models.deepseek_v32 import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.models.variants import list_model_variants, model_variant_lineage
from torchinferno.openai_server import config_from_args as openai_config_from_args
from torchinferno.openai_server import (
    OpenAICompletionEngine,
    OpenAIServerConfig,
    _ByteFallbackTokenizer,
    _is_tensor_parallel_worker_model,
    _tensor_parallel_worker_loop,
    build_engine as build_openai_engine,
    serve as serve_openai_api,
)
from torchinferno.kernels import KernelBackend, KernelConfig, paged_decode_attention
from torchinferno.profiling import (
    PatternProfileConfig,
    ProfileRunConfig,
    RegionProfileConfig,
    SubgraphProfileConfig,
    OffloadProfileConfig,
    TimeSliceProfileConfig,
    run_offload_profile_capture,
    run_pattern_profile_capture,
    run_profile_capture,
    run_region_profile_capture,
    run_subgraph_profile_capture,
    run_timeslice_profile_capture,
)
from torchinferno.research import (
    ExperimentResult,
    HelionCandidateConfig,
    HelionDecisionStore,
    HelionRegionSearchConfig,
    ResearchHarness,
    run_helion_candidate as run_helion_candidate_trial,
    run_helion_fx_search,
    run_helion_region_search,
)
from torchinferno.research.benchmarks import benchmark_callable
from torchinferno.runtime.disagg import run_disagg_request, write_rank_files
from torchinferno.runtime.paged import PagedKVCache
from torchinferno.runtime.paged_attention import paged_causal_attention
from torchinferno.runtime.scheduler import DisaggregatedPrefillDecodeSimulator, InferenceJob
from torchinferno.runtime.serving import ContinuousBatchEngine, ServingRequest
from torchinferno.runtime.traffic import TrafficPattern, simulate_traffic
from torchinferno.tokenization import load_text_tokenizer
from torchinferno.validation import (
    capture_logit_reference,
    load_logit_reference,
    save_logit_reference,
    validate_logit_reference,
)
from torchinferno.variant_validation import run_variant_logit_validation, save_variant_logit_report


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


def run_audit(args: argparse.Namespace) -> int:
    print(build_audit_report().format())
    return 0


def run_model_variants(args: argparse.Namespace) -> int:
    specs = (
        model_variant_lineage(args.family, args.lineage)
        if args.lineage is not None
        else list_model_variants(args.family)
    )
    print("TorchInferno model variants")
    for spec in specs:
        parents = ",".join(spec.parents) if spec.parents else "-"
        print(
            f"{spec.family}:{spec.variant} stage={spec.stage} parents={parents} "
            f"status={spec.status} class={spec.class_path} ops={spec.ops_module}"
        )
    return 0


def run_validate_model_variants(args: argparse.Namespace) -> int:
    device = torch.device(args.device or "cpu")
    dtype = dtype_from_name(args.dtype) or torch.float32
    report = run_variant_logit_validation(
        family=args.family,
        variant=args.variant,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        tokens=args.tokens,
        vocab_size=args.vocab_size,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        include_tensor_parallel=args.include_tensor_parallel,
    )
    if args.json_output:
        save_variant_logit_report(report, args.json_output)

    print("TorchInferno variant logit validation")
    print(
        f"passed={report.passed} comparisons={len(report.comparisons)} skipped={len(report.skipped)} "
        f"device={device} dtype={str(dtype).replace('torch.', '')} "
        f"batch={args.batch_size} tokens={args.tokens} atol={args.atol:g} rtol={args.rtol:g}"
    )
    for comparison in report.comparisons:
        print(
            f"{comparison.family}:{comparison.optimized_variant} vs {comparison.eager_variant} "
            f"passed={comparison.passed} max_abs_error={comparison.max_abs_error:.6g} "
            f"max_rel_error={comparison.max_rel_error:.6g} mean_abs_error={comparison.mean_abs_error:.6g} "
            f"logits={comparison.compared_logits}"
        )
    for skipped in report.skipped:
        print(f"skipped {skipped.family}:{skipped.optimized_variant} vs {skipped.eager_variant}: {skipped.reason}")
    if args.json_output:
        print(f"json_output={args.json_output}")
    return 0 if report.passed and report.comparisons else 2


def run_vllm_bench_suite(args: argparse.Namespace) -> int:
    config = VLLMBenchmarkConfig(
        output_dir=args.output_dir,
        model=args.model,
        vllm_root=args.vllm_root,
        python=args.python,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_size=args.batch_size,
        num_iters_warmup=args.num_iters_warmup,
        num_iters=args.num_iters,
        num_prompts=args.num_prompts,
        dataset_name=args.dataset_name,
        request_rate=args.request_rate,
        throughput_backend=args.throughput_backend,
        serve_backend=args.serve_backend,
        base_url=args.base_url,
        temperature=args.temperature,
        benchmarks=tuple(args.benchmarks),
        disable_detokenize=args.disable_detokenize,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        engine_args=tuple(args.engine_arg or ()),
        throughput_args=tuple(args.throughput_arg or ()),
        serve_args=tuple(args.serve_arg or ()),
    )
    try:
        artifacts = run_vllm_benchmark_suite(config, run=args.run, plot=not args.no_plot)
    except subprocess.CalledProcessError as error:
        print("TorchInferno vLLM benchmark suite failed")
        print(f"benchmark={args.benchmarks} returncode={error.returncode}")
        print(f"output_dir={Path(args.output_dir).resolve()}")
        print("See run_status.json and *.log in the output directory.")
        return int(error.returncode)

    mode = "ran" if args.run else "planned"
    print("TorchInferno vLLM benchmark suite")
    print(f"mode={mode} model={args.model} tp={args.tensor_parallel_size}")
    print(f"output_dir={artifacts.output_dir}")
    print(f"commands={artifacts.commands_path}")
    print(f"summary={artifacts.summary_path}")
    if artifacts.plot_path is not None:
        print(f"plot={artifacts.plot_path}")
    return 0


def run_vllm_bench_plot(args: argparse.Namespace) -> int:
    outputs = plot_vllm_benchmark_results(args.output_dir, output_html=args.output_html, output_csv=args.output_csv)
    print("TorchInferno vLLM benchmark plot")
    print(f"html={outputs['html']}")
    print(f"csv={outputs['csv']}")
    return 0


def run_llama_bench_suite(args: argparse.Namespace) -> int:
    config = TorchInfernoLlamaBenchmarkConfig(
        output_dir=args.output_dir,
        model=args.model,
        devices=tuple(args.devices or ()),
        dtype=args.dtype,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_size=args.batch_size,
        num_iters_warmup=args.num_iters_warmup,
        num_iters=args.num_iters,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
        seed=args.seed,
        benchmarks=tuple(args.benchmarks),
        token=args.token,
        revision=args.revision,
        cache_dir=args.cache_dir,
        parallelism=args.parallelism,
        profile_breakdown=args.profile_breakdown,
    )
    artifacts = run_torchinferno_llama_benchmark_suite(config, run=args.run, plot=not args.no_plot)
    if int(os.environ.get("RANK", "0")) != 0:
        return 0
    mode = "ran" if args.run else "planned"
    print("TorchInferno native Llama benchmark suite")
    print(
        f"mode={mode} model={args.model} parallelism={args.parallelism} "
        f"devices={','.join(args.devices or ['auto'])}"
    )
    print(f"output_dir={artifacts.output_dir}")
    print(f"summary={artifacts.summary_path}")
    if artifacts.plot_path is not None:
        print(f"plot={artifacts.plot_path}")
    return 0


def run_openai_server_microbench_cli(args: argparse.Namespace) -> int:
    config = OpenAIServerMicrobenchConfig(
        model=args.model,
        model_kind=args.model_kind,
        tokenizer=args.tokenizer,
        host=args.host,
        port=args.port,
        base_url=args.base_url,
        python=args.python,
        device=args.device,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        cache_backend=args.cache_backend,
        page_size=args.page_size,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        llama_parallelism=args.llama_parallelism,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        revision=args.revision,
        cache_dir=args.cache_dir,
        modes=(args.mode,),
        prompt_mode=args.prompt_mode,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        warmup=args.warmup,
        iters=args.iters,
        temperature=args.temperature,
        ready_timeout_s=args.ready_timeout_s,
        request_timeout_s=args.request_timeout_s,
        json_output=Path(args.json_output) if args.json_output else None,
    )
    result = run_openai_server_microbench(config)
    print(format_openai_server_microbench_report(result))
    return 0


def run_deepseek_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_deepseek_v32_config(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.prompt_tokens + args.new_tokens + 8,
        q_lora_rank=None if args.no_q_lora else 16,
        use_score_correction_bias=args.score_bias,
    )
    model = DeepSeekV32ForCausalLM(config).to(device).eval()
    if args.compile:
        compile_forward(model, CompileConfig(mode="reduce-overhead"))

    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_tokens), device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            prompt,
            max_new_tokens=args.new_tokens,
            temperature=args.temperature,
            cache_backend=args.cache_backend,
            page_size=args.page_size,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("TorchInferno native DeepSeek smoke")
    print(f"device={device} batch={args.batch_size} prompt={args.prompt_tokens} new={args.new_tokens}")
    print(f"shape={tuple(output.shape)} elapsed_ms={elapsed_ms:.2f}")
    print(f"tokens[0]={output[0].tolist()}")
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


def run_deepseek_hf_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    model = DeepSeekV32ForCausalLM.from_pretrained(
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
    print("TorchInferno native DeepSeek HF smoke")
    print(f"model={args.model} device={device} new={args.new_tokens}")
    print(f"shape={tuple(output.shape)}")
    print(f"tokens[0]={output[0].tolist()}")
    return 0


def run_text_generate(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    model = load_model_auto(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        map_location="cpu",
        strict=not args.non_strict,
    ).to(device)
    model.eval()
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = load_text_tokenizer(tokenizer_path, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    if not input_ids:
        raise ValueError("tokenizer produced no input ids")
    prompt = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        output = model.generate(prompt, max_new_tokens=args.new_tokens, temperature=args.temperature)  # type: ignore[attr-defined]
    generated = output[0].detach().cpu().tolist()
    print("TorchInferno text generation")
    print(f"model={args.model} tokenizer={tokenizer_path} device={device}")
    print(f"input_ids={input_ids}")
    print(f"output_ids={generated}")
    print(tokenizer.decode(generated))
    return 0


class _OpenAIMicrobenchCache:
    def __init__(self) -> None:
        self.seq_len = 0


class _OpenAIMicrobenchModel:
    def __init__(self, *, vocab_size: int, device: torch.device, dtype: torch.dtype, sleep_us: float = 0.0) -> None:
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.device = device
        self.dtype = dtype
        self.sleep_s = max(0.0, sleep_us / 1_000_000.0)
        self.calls: list[tuple[int, int]] = []
        self.devices = (device,)

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs: object) -> _OpenAIMicrobenchCache:
        return _OpenAIMicrobenchCache()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _OpenAIMicrobenchCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ) -> tuple[torch.Tensor, _OpenAIMicrobenchCache]:
        del use_cache
        if self.sleep_s:
            time.sleep(self.sleep_s)
        batch_size = input_ids.size(0)
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        self.calls.append((batch_size, input_ids.size(1)))
        cache.seq_len += input_ids.size(1)
        logits = torch.zeros(batch_size, tokens, self.config.vocab_size, device=self.device, dtype=self.dtype)
        logits[..., min(2, self.config.vocab_size - 1)] = 1.0
        return logits, cache


def run_openai_microbench(args: argparse.Namespace) -> int:
    if args.phase_timings:
        os.environ["TORCHINFERNO_OPENAI_PHASE_TIMINGS"] = "1"
    if args.profile_breakdown:
        os.environ["TORCHINFERNO_PROFILE_FAST_PREFILL"] = "1"
    engine = _build_openai_microbench_engine(args)
    if args.profile_breakdown and hasattr(engine.model, "enable_profile"):
        engine.model.enable_profile()
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return 0

    try:
        cases, skipped_batcher_compare = _openai_microbench_cases(
            engine,
            compare_batcher=args.compare_batcher,
            concurrency=args.concurrency,
        )

        print("TorchInferno OpenAI microbench")
        print(
            f"backend={args.backend} device={engine.device} prompt_tokens={args.prompt_tokens} "
            f"max_tokens={args.max_tokens} warmup={args.warmup} iters={args.iters} "
            f"batch_wait_ms={args.batch_wait_ms:g} prompt_mode={args.prompt_mode}"
        )
        if skipped_batcher_compare:
            print("skip=single-batcher reason=tensor_parallel_worker_protocol")
        results: dict[str, object] = {
            "backend": args.backend,
            "device": str(engine.device),
            "prompt_tokens": args.prompt_tokens,
            "prompt_mode": args.prompt_mode,
            "max_tokens": args.max_tokens,
            "batch_wait_ms": args.batch_wait_ms,
            "skipped_single_batcher": skipped_batcher_compare,
            "cases": {},
        }
        for label, fast_path, concurrency in cases:
            engine.single_request_fast_path = fast_path
            case = _run_openai_microbench_case(
                engine,
                label=label,
                concurrency=concurrency,
                warmup=args.warmup,
                iters=args.iters,
                prompt_tokens=args.prompt_tokens,
                prompt_mode=args.prompt_mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                phase_timings=args.phase_timings,
                profile_breakdown=args.profile_breakdown,
            )
            results["cases"][label] = case
            print(
                f"case={label} concurrency={concurrency} fast_path={fast_path} "
                f"ttft_p50_ms={case['ttft_p50_ms']:.3f} tpot_p50_ms={case['tpot_p50_ms']:.3f} "
                f"e2e_p50_ms={case['e2e_p50_ms']:.3f} throughput_p50_tps={case['throughput_p50_tps']:.2f} "
                f"max_model_batch={case.get('max_model_batch', 0)}"
            )
            phase_summary = case.get("phase_timings_ms")
            if isinstance(phase_summary, dict):
                print(
                    "  phase "
                    f"request_to_first_forward_p50_ms={phase_summary.get('request_to_first_forward_p50_ms', 0.0):.3f} "
                    f"broadcast_p50_ms={phase_summary.get('broadcast_p50_ms', 0.0):.3f} "
                    f"cache_p50_ms={phase_summary.get('cache_p50_ms', 0.0):.3f} "
                    f"prefix_cache_p50_ms={phase_summary.get('prefix_cache_p50_ms', 0.0):.3f} "
                    f"prefix_cache_tokens_p50={phase_summary.get('prefix_cache_tokens_p50', 0.0):.0f} "
                    f"prefill_tokens_p50={phase_summary.get('prefill_tokens_p50', 0.0):.0f} "
                    f"prefill_forward_p50_ms={phase_summary.get('prefill_forward_p50_ms', 0.0):.3f} "
                    f"sample_p50_ms={phase_summary.get('sample_p50_ms', 0.0):.3f} "
                    f"first_token_sync_p50_ms={phase_summary.get('first_token_sync_p50_ms', 0.0):.3f}"
                )
        if args.json_output:
            if args.profile_breakdown and hasattr(engine.model, "profile_summary"):
                results["profile_summary"] = engine.model.profile_summary()
            Path(args.json_output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
            print(f"json_output={args.json_output}")
        if args.profile_breakdown and hasattr(engine.model, "profile_summary"):
            profile = engine.model.profile_summary()
            seconds = profile.get("seconds", {})
            if isinstance(seconds, dict):
                top = sorted(((float(value), str(key)) for key, value in seconds.items()), reverse=True)[:8]
                for value, key in top:
                    print(f"  profile {key} total_ms={value * 1000.0:.3f}")
    finally:
        engine.close()
    return 0


def _openai_microbench_cases(
    engine: OpenAICompletionEngine,
    *,
    compare_batcher: bool,
    concurrency: int,
) -> tuple[list[tuple[str, bool, int]], bool]:
    default_fast_path = bool(engine.single_request_fast_path)
    cases: list[tuple[str, bool, int]] = []
    skipped_batcher_compare = False
    if compare_batcher:
        cases.append(("single-direct", True, 1))
        if int(getattr(engine.model, "world_size", 1)) > 1:
            skipped_batcher_compare = True
        else:
            cases.append(("single-batcher", False, 1))
    else:
        cases.append(("single", default_fast_path, 1))
    if concurrency > 1:
        cases.append((f"concurrent-{concurrency}", default_fast_path, concurrency))
    return cases, skipped_batcher_compare


def _build_openai_microbench_engine(args: argparse.Namespace) -> OpenAICompletionEngine:
    device = torch.device(args.device) if args.device else _default_device()
    synthetic_dtype_name = None if args.dtype == "auto" else args.dtype
    dtype = dtype_from_name(synthetic_dtype_name) or torch.float32
    if args.backend == "synthetic":
        model = _OpenAIMicrobenchModel(
            vocab_size=args.vocab_size,
            device=device,
            dtype=dtype,
            sleep_us=args.synthetic_forward_sleep_us,
        )
        return OpenAICompletionEngine(
            model,
            _ByteFallbackTokenizer(vocab_size=args.vocab_size),
            model_id="synthetic",
            device=device,
            max_batch_size=args.max_batch_size,
            batch_wait_ms=args.batch_wait_ms,
        )

    if not args.model:
        raise ValueError("--model is required when --backend=model")
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip()) if args.devices else ()
    return build_openai_engine(
        OpenAIServerConfig(
            model=args.model,
            model_kind=args.model_kind,
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            devices=devices,
            device=args.device,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            token=args.token,
            revision=args.revision,
            cache_dir=args.cache_dir,
            cache_backend=args.cache_backend,
            page_size=args.page_size,
            max_batch_size=args.max_batch_size,
            batch_wait_ms=args.batch_wait_ms,
            llama_parallelism=args.llama_parallelism,
        )
    )


def _run_openai_microbench_case(
    engine: OpenAICompletionEngine,
    *,
    label: str,
    concurrency: int,
    warmup: int,
    iters: int,
    prompt_tokens: int,
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
    phase_timings: bool = False,
    profile_breakdown: bool = False,
) -> dict[str, object]:
    del label
    for _ in range(warmup):
        _run_openai_microbench_iteration(
            engine,
            concurrency,
            prompt_tokens,
            prompt_mode,
            max_tokens,
            temperature,
        )
    if phase_timings and hasattr(engine, "pop_phase_records"):
        engine.pop_phase_records()
    if profile_breakdown and hasattr(engine.model, "enable_profile"):
        engine.model.enable_profile()

    start_calls = len(getattr(engine.model, "calls", ()))
    measurements = [
        metric
        for _ in range(iters)
        for metric in _run_openai_microbench_iteration(
            engine,
            concurrency,
            prompt_tokens,
            prompt_mode,
            max_tokens,
            temperature,
        )
    ]
    call_slice = getattr(engine.model, "calls", ())[start_calls:]
    max_model_batch = max((batch for batch, _tokens in call_slice), default=0)
    result: dict[str, object] = {
        "requests": len(measurements),
        "ttft_p50_ms": _median([metric["ttft_ms"] for metric in measurements]),
        "ttft_p99_ms": _p99([metric["ttft_ms"] for metric in measurements]),
        "tpot_p50_ms": _median([metric["tpot_ms"] for metric in measurements]),
        "tpot_p99_ms": _p99([metric["tpot_ms"] for metric in measurements]),
        "e2e_p50_ms": _median([metric["e2e_ms"] for metric in measurements]),
        "e2e_p99_ms": _p99([metric["e2e_ms"] for metric in measurements]),
        "throughput_p50_tps": _median([metric["throughput_tps"] for metric in measurements]),
        "output_tokens_p50": _median([metric["output_tokens"] for metric in measurements]),
        "model_forward_calls": len(call_slice),
        "max_model_batch": max_model_batch,
        "raw_requests": measurements,
    }
    if phase_timings and hasattr(engine, "pop_phase_records"):
        phase_records = engine.pop_phase_records()
        if phase_records:
            result["phase_timings_ms"] = _summarize_openai_phase_records(phase_records)
    return result


def _summarize_openai_phase_records(records: list[dict[str, float]]) -> dict[str, float]:
    def duration(start: str, end: str) -> float:
        values = [(record[end] - record[start]) * 1000.0 for record in records if start in record and end in record]
        return _median(values) if values else 0.0

    def value(name: str) -> float:
        values = [record[name] for record in records if name in record]
        return _median(values) if values else 0.0

    return {
        "request_to_first_forward_p50_ms": duration("request_start", "first_forward_start"),
        "encode_p50_ms": duration("request_start", "encoded_prompt"),
        "input_tensor_p50_ms": duration("encoded_prompt", "built_input_tensor"),
        "broadcast_p50_ms": duration("broadcast_start", "broadcast_done"),
        "cache_p50_ms": duration("cache_start", "cache_done"),
        "prefix_cache_p50_ms": duration("cache_done", "prefix_cache_done"),
        "prefix_cache_tokens_p50": value("prefix_cache_tokens"),
        "prefill_tokens_p50": value("prefill_tokens"),
        "prefill_forward_p50_ms": duration("first_forward_start", "first_forward_done"),
        "sample_p50_ms": duration("first_forward_done", "prefill_sample_done"),
        "first_token_sync_p50_ms": duration("first_token_sync_start", "first_token_ready"),
        "internal_ttft_p50_ms": duration("request_start", "first_token_ready"),
    }


def _run_openai_microbench_iteration(
    engine: OpenAICompletionEngine,
    concurrency: int,
    prompt_tokens: int,
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
) -> list[dict[str, float]]:
    _sync_openai_engine(engine)
    if concurrency == 1:
        metrics = [_run_openai_microbench_request(engine, 0, prompt_tokens, prompt_mode, max_tokens, temperature)]
        _sync_openai_engine(engine)
        return metrics

    barrier = threading.Barrier(concurrency + 1)
    results: list[dict[str, float] | None] = [None for _ in range(concurrency)]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            results[index] = _run_openai_microbench_request(
                engine,
                index,
                prompt_tokens,
                prompt_mode,
                max_tokens,
                temperature,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(concurrency)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    _sync_openai_engine(engine)
    if errors:
        raise errors[0]
    return [metric for metric in results if metric is not None]


def _run_openai_microbench_request(
    engine: OpenAICompletionEngine,
    request_index: int,
    prompt_tokens: int,
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, float]:
    messages = _microbench_messages(prompt_tokens, request_index, prompt_mode)
    metrics, _content = _run_openai_microbench_messages(engine, messages, max_tokens, temperature)
    return metrics


def _run_openai_microbench_messages(
    engine: OpenAICompletionEngine,
    messages: list[dict[str, object]],
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, float], str]:
    start = time.perf_counter()
    token_times: list[float] = []
    chunks: list[str] = []
    for token_id in engine.generate_chat_tokens(messages, max_tokens=max_tokens, temperature=temperature):
        content = engine.tokenizer.decode_token(token_id)
        if not content:
            continue
        token_times.append(time.perf_counter())
        chunks.append(content)
    end = time.perf_counter()
    output_tokens = len(token_times)
    ttft_ms = ((token_times[0] - start) * 1000.0) if token_times else 0.0
    e2e_ms = (end - start) * 1000.0
    if output_tokens > 1:
        tpot_ms = ((token_times[-1] - token_times[0]) / (output_tokens - 1)) * 1000.0
    else:
        tpot_ms = 0.0
    throughput_tps = output_tokens / (e2e_ms / 1000.0) if e2e_ms > 0.0 else 0.0
    return (
        {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "output_tokens": float(output_tokens),
            "throughput_tps": throughput_tps,
        },
        "".join(chunks),
    )


def _microbench_messages(
    prompt_tokens: int,
    request_index: int,
    prompt_mode: str = "synthetic",
) -> list[dict[str, object]]:
    if prompt_mode == "self-consistency":
        return [
            {
                "role": "system",
                "content": "You are a calculator. Respond with only the numerical answer, nothing else.",
            },
            {"role": "user", "content": "17 * 23 ="},
        ]
    if prompt_mode == "few-shot":
        examples = (
            ("15 + 27 =", 42),
            ("198 - 53 =", 145),
            ("12 * 14 =", 168),
            ("225 / 9 =", 25),
            ("347 + 258 =", 605),
        )
        example_text = "\n\n".join(f"Q: {question}\nA: {answer}" for question, answer in examples)
        system_prompt = (
            "You are a calculator. Compute the answer to each math equation. "
            "Respond with only the numerical answer, nothing else.\n\n"
            "Examples:\n\n" + example_text
        )
        left = 100 + (request_index * 37) % 1900
        right = 1 + (request_index * 17) % 1900
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Q: {left} + {right} =\nA:"},
        ]
    if prompt_mode != "synthetic":
        raise ValueError(f"unsupported prompt_mode {prompt_mode!r}")
    return [{"role": "user", "content": _microbench_prompt_text(prompt_tokens, request_index)}]


def _microbench_prompt_text(prompt_tokens: int, request_index: int) -> str:
    return " ".join(f"tok{(request_index + idx) % 97:02d}" for idx in range(prompt_tokens))


def _sync_openai_engine(engine: OpenAICompletionEngine) -> None:
    devices = getattr(engine.model, "devices", (engine.device,))
    for device in devices:
        resolved = torch.device(device)
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * 0.99))
    return ordered[index]


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


def run_disagg_init(args: argparse.Namespace) -> int:
    plan = write_rank_files(
        Path(args.output_dir),
        prefill_ranks=args.prefill_ranks,
        decode_ranks=args.decode_ranks,
        host=args.host,
        base_port=args.base_port,
        device=args.device,
        seed=args.seed,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
    )
    print("TorchInferno disaggregated rank files")
    print(f"output_dir={plan.output_dir}")
    print(f"manifest={plan.manifest}")
    print(f"client_smoke={plan.client_smoke}")
    for endpoint in plan.endpoints:
        print(f"rank={endpoint.rank_id} role={endpoint.role} url={endpoint.url} file={endpoint.file}")
    return 0


def run_disagg_smoke(args: argparse.Namespace) -> int:
    result = run_disagg_request(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        request_id=args.request_id,
        prompt=args.prompt,
        max_new_tokens=args.new_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )
    print("TorchInferno disaggregated RPC smoke")
    print(f"prefill_tokens={result['prefill']['tokens']}")
    print(f"decode_tokens={result['decode']['tokens']}")
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


def run_helion_candidate_cli(args: argparse.Namespace) -> int:
    report = run_helion_candidate_trial(
        HelionCandidateConfig(
            candidate=args.candidate,
            batch_size=args.batch_size,
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            dtype=args.dtype,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            min_speedup=args.min_speedup,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    if args.output:
        report.write_json(args.output)
    print("TorchInferno Helion candidate")
    print(
        f"candidate={report.candidate} candidate_id={report.candidate_id} "
        f"status={report.status} promoted={report.promoted}"
    )
    print(f"shape={report.shape} dtype={report.dtype} device={report.device}")
    if report.baseline is not None and report.helion is not None:
        print(f"baseline={report.baseline.mean_ms:.4f}ms name={report.baseline.name}")
        if report.torch_compile is not None:
            print(f"torch_compile={report.torch_compile.mean_ms:.4f}ms name={report.torch_compile.name}")
        print(f"helion={report.helion.mean_ms:.4f}ms name={report.helion.name}")
        print(f"speedup={report.speedup:.3f} min_speedup={report.min_speedup:.3f}")
        if report.speedup_vs_compile:
            print(f"speedup_vs_compile={report.speedup_vs_compile:.3f}")
        print(f"correct={report.correct} max_abs_error={report.max_abs_error:.6g}")
    print(f"reason={report.reason}")
    if args.output:
        print(f"output={Path(args.output).resolve()}")
    return 0


def run_helion_search_fx_cli(args: argparse.Namespace) -> int:
    config = HelionCandidateConfig(
        candidate=args.candidate,
        batch_size=args.batch_size,
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        min_speedup=args.min_speedup,
        atol=args.atol,
        rtol=args.rtol,
    )
    search = run_helion_fx_search(config, min_nodes=args.min_nodes, max_nodes=args.max_nodes)
    if args.output:
        search.write_json(args.output)
    if args.remember:
        HelionDecisionStore(args.remember).append_many(search.reports)
    supported = [window for window in search.windows if window.supported_kernel]
    promoted = [report for report in search.reports if report.promoted]
    print("TorchInferno Helion FX search")
    print(
        f"candidate={search.candidate} windows={len(search.windows)} "
        f"supported={len(supported)} reports={len(search.reports)} promoted={len(promoted)}"
    )
    for window in supported:
        node_targets = " -> ".join(node.target for node in window.nodes)
        print(f"window={window.candidate_id} kernel={window.supported_kernel} nodes={node_targets}")
    for report in search.reports:
        print(
            f"decision={report.status} candidate_id={report.candidate_id} "
            f"promoted={report.promoted} speedup={report.speedup:.3f} reason={report.reason}"
        )
    if args.output:
        print(f"output={Path(args.output).resolve()}")
    if args.remember:
        print(f"remembered={Path(args.remember).resolve()}")
    return 0


def run_helion_search_region_cli(args: argparse.Namespace) -> int:
    config = HelionRegionSearchConfig(
        model_kind=args.model_kind,
        region=args.region,
        batch_size=args.batch_size,
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype=args.dtype,
        device=args.device,
        trace_device=args.trace_device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        min_speedup=args.min_speedup,
        atol=args.atol,
        rtol=args.rtol,
    )
    search = run_helion_region_search(config, min_nodes=args.min_nodes, max_nodes=args.max_nodes)
    if args.output:
        search.write_json(args.output)
    if args.remember:
        HelionDecisionStore(args.remember).append_many(search.reports)
    supported = [window for window in search.windows if window.supported_kernel]
    promoted = [report for report in search.reports if report.promoted]
    print("TorchInferno Helion region search")
    print(
        f"model={search.model_kind} region={search.region} windows={len(search.windows)} "
        f"supported={len(supported)} macro={len(search.macro_candidates)} reports={len(search.reports)} "
        f"promoted={len(promoted)}"
    )
    print(f"candidate_activation_shape={search.candidate_activation_shape}")
    for candidate in search.macro_candidates:
        print(f"macro={candidate.name} status={candidate.status} reason={candidate.reason}")
        if candidate.evidence:
            print(f"  evidence={'; '.join(candidate.evidence[:5])}")
    for window in supported:
        node_targets = " -> ".join(node.target for node in window.nodes)
        print(f"window={window.candidate_id} kernel={window.supported_kernel} nodes={node_targets}")
    for report in search.reports:
        print(
            f"decision={report.status} candidate_id={report.candidate_id} "
            f"promoted={report.promoted} speedup={report.speedup:.3f} reason={report.reason}"
        )
    if args.output:
        print(f"output={Path(args.output).resolve()}")
    if args.remember:
        print(f"remembered={Path(args.remember).resolve()}")
    return 0


def run_traffic_smoke(args: argparse.Namespace) -> int:
    scheduler = DisaggregatedPrefillDecodeSimulator(
        prefill_ranks=tuple(range(args.prefill_ranks)),
        decode_ranks=tuple(range(args.prefill_ranks, args.prefill_ranks + args.decode_ranks)),
        prefill_us_per_token=args.prefill_us_per_token,
        decode_us_per_token=args.decode_us_per_token,
        network_latency_us=args.network_latency_us,
    )
    pattern = TrafficPattern(
        requests=args.requests,
        prompt_min=args.prompt_min,
        prompt_max=args.prompt_max,
        decode_min=args.decode_min,
        decode_max=args.decode_max,
        burst_size=args.burst_size,
        burst_gap_us=args.burst_gap_us,
        in_burst_gap_us=args.in_burst_gap_us,
        seed=args.seed,
    )
    result = simulate_traffic(pattern, scheduler)
    print("TorchInferno traffic simulation")
    print(
        f"requests={len(result.jobs)} stages={len(result.stages)} "
        f"makespan_us={result.makespan_us:.1f} rps={result.requests_per_second:.2f}"
    )
    for stage in result.stages[: args.print_stages]:
        print(f"{stage.request_id} {stage.stage} rank={stage.rank} start={stage.start_us:.1f} end={stage.end_us:.1f}")
    return 0


def run_serve_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_deepseek_v32_config(vocab_size=args.vocab_size, max_position_embeddings=64)
    model = DeepSeekV32ForCausalLM(config).to(device).eval()
    engine = ContinuousBatchEngine(
        model,
        device=device,
        cache_backend=args.cache_backend,
        page_size=args.page_size,
        temperature=args.temperature,
        max_active_requests=args.max_active_requests,
    )
    requests = [
        ServingRequest("req-a", (1, 2, 3), args.new_tokens, arrival_step=0),
        ServingRequest("req-b", (1, 2, 3, 4), args.new_tokens, arrival_step=1),
        ServingRequest("req-c", (5, 6, 7, 8), args.new_tokens, arrival_step=1),
    ]
    start = time.perf_counter()
    results = engine.run(requests)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print("TorchInferno serving smoke")
    print(f"device={device} cache_backend={args.cache_backend} requests={len(results)} elapsed_ms={elapsed_ms:.2f}")
    print(
        f"prefill_model_calls={engine.stats.prefill_model_calls} "
        f"decode_model_calls={engine.stats.decode_model_calls} "
        f"prefix_reuse_tokens={engine.stats.prefix_reuse_tokens} "
        f"persistent_cache_rows={engine.stats.persistent_cache_rows} "
        f"max_model_batch_size={engine.stats.max_model_batch_size}"
    )
    for result in results:
        print(
            f"{result.request_id}: tokens={list(result.tokens)} "
            f"prefix_hit_tokens={result.prefix_hit_tokens} finished_step={result.finished_step}"
        )
    return 0


def run_openai_server(args: argparse.Namespace) -> int:
    serve_openai_api(openai_config_from_args(args))
    return 0


def run_perf_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    cache = PagedKVCache(
        num_pages=(args.seq_len + args.page_size - 1) // args.page_size,
        page_size=args.page_size,
        num_key_value_heads=args.heads,
        head_dim=args.head_dim,
        value_head_dim=args.value_dim,
        device=device,
        dtype=torch.float32,
    )
    keys = torch.randn(args.heads, args.seq_len, args.head_dim, device=device)
    values = torch.randn(args.heads, args.seq_len, args.value_dim, device=device)
    query = torch.randn(args.heads, 1, args.head_dim, device=device)
    cache.append("req", keys, values)
    position = args.seq_len - 1
    backend = KernelBackend(args.backend)

    reference = benchmark_callable(
        "paged-reference",
        lambda: paged_causal_attention(query, cache, "req", torch.tensor([position], device=device)),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    specialized = benchmark_callable(
        f"paged-decode-{backend.value}",
        lambda: paged_decode_attention(query, cache, "req", position, config=KernelConfig(backend=backend)),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    print("TorchInferno performance smoke")
    for result in (reference, specialized):
        print(f"{result.name}: mean_ms={result.mean_ms:.4f} iters={result.iters} device={result.device}")
    if specialized.mean_ms > 0:
        print(f"speedup_vs_reference={reference.mean_ms / specialized.mean_ms:.3f}")
    return 0


def run_profile_run(args: argparse.Namespace) -> int:
    device = args.device if args.device is not None else str(_default_device())
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(".torchinferno_runs") / f"{args.model_kind}-{stamp}"
    else:
        output_dir = Path(args.output_dir)
    artifacts = run_profile_capture(
        ProfileRunConfig(
            output_dir=output_dir,
            model_kind=args.model_kind,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            new_tokens=args.new_tokens,
            vocab_size=args.vocab_size,
            temperature=args.temperature,
            compile=args.compile,
            cache_backend=args.cache_backend,
            page_size=args.page_size,
            warmup=args.warmup,
            capture_graph=not args.no_graph,
            fake_graph=args.fake_graph,
            require_graph=args.require_graph,
            capture_profiler=not args.no_profiler,
            export_chrome_trace=not args.no_chrome_trace,
            with_stack=args.with_stack,
            with_flops=not args.no_flops,
            command=tuple(sys.argv),
        )
    )
    print("TorchInferno profile run")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(f"repro={artifacts.repro}")
    if artifacts.chrome_trace is not None:
        print(f"chrome_trace={artifacts.chrome_trace}")
    if artifacts.graph_json is not None:
        print(f"graph_json={artifacts.graph_json}")
    return 0


def run_profile_region(args: argparse.Namespace) -> int:
    device = args.device if args.device is not None else str(_default_device())
    artifacts = run_region_profile_capture(
        RegionProfileConfig(
            output_dir=Path(args.output_dir),
            region=args.region,
            model_kind=args.model_kind,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            tokens=args.tokens,
            vocab_size=args.vocab_size,
            warmup=args.warmup,
            iters=args.iters,
            fake_graph=args.fake_graph,
            require_graph=args.require_graph,
            capture_profiler=not args.no_profiler,
            export_chrome_trace=not args.no_chrome_trace,
            with_stack=args.with_stack,
            with_flops=not args.no_flops,
            command=tuple(sys.argv),
        )
    )
    print("TorchInferno region profile")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(f"repro={artifacts.repro}")
    if artifacts.graph_json is not None:
        print(f"graph_json={artifacts.graph_json}")
    if artifacts.chrome_trace is not None:
        print(f"chrome_trace={artifacts.chrome_trace}")
    return 0


def run_profile_pattern(args: argparse.Namespace) -> int:
    device = args.device if args.device is not None else str(_default_device())
    artifacts = run_pattern_profile_capture(
        PatternProfileConfig(
            output_dir=Path(args.output_dir),
            pattern=args.pattern,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            warmup=args.warmup,
            iters=args.iters,
            apply_passes=not args.no_apply_passes,
            fake_graph=args.fake_graph,
            require_graph=args.require_graph,
            capture_profiler=not args.no_profiler,
            export_chrome_trace=not args.no_chrome_trace,
            with_stack=args.with_stack,
            with_flops=not args.no_flops,
            command=tuple(sys.argv),
        )
    )
    print("TorchInferno pattern profile")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(f"comparison={artifacts.comparison}")
    print(f"repro={artifacts.repro}")
    return 0


def run_profile_subgraph(args: argparse.Namespace) -> int:
    artifacts = run_subgraph_profile_capture(
        SubgraphProfileConfig(
            output_dir=Path(args.output_dir),
            source_run_dir=Path(args.source_run),
            node_ids=_parse_node_ids(args.nodes),
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            capture_profiler=not args.no_profiler,
            export_chrome_trace=not args.no_chrome_trace,
            with_stack=args.with_stack,
            with_flops=not args.no_flops,
            command=tuple(sys.argv),
        )
    )
    print("TorchInferno subgraph profile")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(f"subgraph_spec={artifacts.subgraph_spec}")
    print(f"subgraph_graph={artifacts.subgraph_graph}")
    print(f"repro={artifacts.repro}")
    if artifacts.chrome_trace is not None:
        print(f"chrome_trace={artifacts.chrome_trace}")
    return 0


def run_profile_nodes(args: argparse.Namespace) -> int:
    graph_path = Path(args.graph_or_run)
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    graph = json.loads(graph_path.read_text())
    needle = args.grep.lower() if args.grep is not None else None
    for fallback_id, node in enumerate(graph["nodes"]):
        node_id = node.get("id", fallback_id)
        name = str(node["name"])
        op = str(node["op"])
        target = str(node["target"])
        haystack = f"{node_id} {name} {op} {target}".lower()
        if needle is not None and needle not in haystack:
            continue
        print(f"{node_id:04d} {op:14s} {name:32s} {target}")
    return 0


def run_profile_timeslice(args: argparse.Namespace) -> int:
    device = args.device if args.device is not None else str(_default_device())
    artifacts = run_timeslice_profile_capture(
        TimeSliceProfileConfig(
            output_dir=Path(args.output_dir),
            model_kind=args.model_kind,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            new_tokens=args.new_tokens,
            vocab_size=args.vocab_size,
            temperature=args.temperature,
            compile=args.compile,
            cache_backend=args.cache_backend,
            page_size=args.page_size,
            warmup=args.warmup,
            iters=args.iters,
            virtual_gpus=args.virtual_gpus,
            time_slice_us=args.time_slice_us,
            context_switch_us=args.context_switch_us,
            arrival_gap_us=args.arrival_gap_us,
            profile_scale=args.profile_scale,
            capture_profiler=not args.no_profiler,
            export_chrome_trace=not args.no_chrome_trace,
            with_stack=args.with_stack,
            with_flops=not args.no_flops,
            command=tuple(sys.argv),
        )
    )
    summary = json.loads(artifacts.summary.read_text())
    representative = json.loads(artifacts.output.read_text())
    print("TorchInferno time-sliced profile")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(f"representative_per_iter_ms={representative['measured_per_iter_ms']:.4f}")
    print(
        f"virtual_gpus={args.virtual_gpus} slices={summary['slice_count']} "
        f"simulated_elapsed_ms={summary['total_elapsed_us'] / 1000.0:.4f} "
        f"utilization={summary['utilization']:.3f}"
    )
    print(f"timeline={artifacts.timeline}")
    print(f"repro={artifacts.repro}")
    if artifacts.chrome_trace is not None:
        print(f"chrome_trace={artifacts.chrome_trace}")
    return 0


def run_profile_offload(args: argparse.Namespace) -> int:
    device = args.device if args.device is not None else str(_default_device())
    artifacts = run_offload_profile_capture(
        OffloadProfileConfig(
            output_dir=Path(args.output_dir),
            checkpoint=args.checkpoint,
            model_kind=args.model_kind,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            new_tokens=args.new_tokens,
            vocab_size=args.vocab_size,
            temperature=args.temperature,
            activation_offload=args.activation_offload,
            warmup=args.warmup,
            iters=args.iters,
            revision=args.revision,
            cache_dir=args.cache_dir,
            strict=not args.non_strict,
            command=tuple(sys.argv),
        )
    )
    summary = json.loads(artifacts.summary.read_text())
    print("TorchInferno offload profile")
    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest={artifacts.manifest}")
    print(
        f"observed_per_iter_ms={summary['observed_per_iter_ms']:.4f} "
        f"movement_per_iter_ms={summary['movement_per_iter_ms']:.4f} "
        f"compute_only_per_iter_ms={summary['compute_only_per_iter_ms']:.4f}"
    )
    print(
        f"bytes_moved={summary['bytes_moved']} "
        f"peak_module_bytes={summary['peak_module_bytes']} "
        f"events={summary['event_count']}"
    )
    print(f"events={artifacts.events}")
    print(f"repro={artifacts.repro}")
    return 0


def run_capture_logits(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    model = load_model_auto(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        map_location="cpu",
        strict=not args.non_strict,
    ).to(device)
    model.eval()
    reference = capture_logit_reference(
        model,
        args.input_ids,
        atol=args.atol,
        rtol=args.rtol,
        description=args.description,
    )
    save_logit_reference(reference, args.output)
    print("TorchInferno logit reference captured")
    print(f"model={args.model} output={args.output} vocab={len(reference.logits)}")
    return 0


def run_validate_logits(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    model = load_model_auto(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        map_location="cpu",
        strict=not args.non_strict,
    ).to(device)
    model.eval()
    reference = load_logit_reference(args.reference)
    result = validate_logit_reference(model, reference)
    print("TorchInferno logit validation")
    print(f"passed={result.passed} max_abs_error={result.max_abs_error:.6g} max_rel_error={result.max_rel_error:.6g}")
    return 0 if result.passed else 2


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


def run_deepseek_audit(args: argparse.Namespace) -> int:
    report = audit_native_deepseek_checkpoint(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(report.summary())
    return 0 if report.compatible else 2


def run_deepseek_convert(args: argparse.Namespace) -> int:
    try:
        report = convert_native_deepseek_checkpoint(
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


def _parse_node_ids(values: list[str]) -> tuple[int, ...]:
    node_ids: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start_text, end_text = part.split(":", 1)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                node_ids.extend(range(start, end + step, step))
            else:
                node_ids.append(int(part))
    return tuple(dict.fromkeys(node_ids))


_FLOAT_DTYPE_CHOICES = ["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]
_AUTO_DTYPE_CHOICES = ["auto", *_FLOAT_DTYPE_CHOICES]
_PROFILE_DTYPE_CHOICES = ["float32", "float16", "bfloat16"]


def _add_device_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    parser.add_argument("--device", default=default, help="Torch device, defaults to cuda when available.")


def _add_dtype_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    choices: list[str],
    help_text: str | None = None,
) -> None:
    parser.add_argument("--dtype", default=default, choices=choices, help=help_text)


def _add_checkpoint_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)


def _add_non_strict_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--non-strict", action="store_true", help="Allow missing or unexpected weight keys.")


def _add_profile_capture_arguments(
    parser: argparse.ArgumentParser,
    *,
    graph_options: bool = False,
    chrome_trace_help: str = "Do not export chrome_trace.json.",
) -> None:
    if graph_options:
        parser.add_argument("--fake-graph", action="store_true", help="Try graph capture under FakeTensorMode.")
        parser.add_argument("--require-graph", action="store_true", help="Fail if graph capture fails.")
    parser.add_argument("--no-profiler", action="store_true", help="Skip torch.profiler capture.")
    parser.add_argument("--no-chrome-trace", action="store_true", help=chrome_trace_help)
    parser.add_argument("--with-stack", action="store_true", help="Capture Python stack traces in the profiler.")
    parser.add_argument("--no-flops", action="store_true", help="Disable profiler FLOP estimation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchinferno")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_status = subparsers.add_parser("audit", help="Print environment and feature readiness status.")
    audit_status.set_defaults(func=run_audit)

    variants = subparsers.add_parser("model-variants", help="List provenance-tracked model variants.")
    variants.add_argument("--family", default=None, help="Filter to a model family such as dsv4, dsv3.2, deepseek-v3.2, or llama3.")
    variants.add_argument("--lineage", default=None, help="Print lineage ending at this variant, e.g. --family llama3 --lineage v1.")
    variants.set_defaults(func=run_model_variants)

    validate_variants = subparsers.add_parser(
        "validate-model-variants",
        help="Compare optimized model variant logits against make_fx v0 references.",
    )
    validate_variants.add_argument("--family", default=None, help="Filter to dsv4, dsv3.2/deepseek-v3.2, or llama3.")
    validate_variants.add_argument("--variant", default=None, help="Only validate one optimized variant, such as v1.")
    validate_variants.add_argument("--device", default="cpu", help="Torch device for tiny validation models.")
    _add_dtype_argument(
        validate_variants,
        default="float32",
        choices=_FLOAT_DTYPE_CHOICES,
        help_text="Model dtype for the comparison.",
    )
    validate_variants.add_argument("--batch-size", type=int, default=2)
    validate_variants.add_argument("--tokens", type=int, default=4)
    validate_variants.add_argument("--vocab-size", type=int, default=32)
    validate_variants.add_argument("--seed", type=int, default=0)
    validate_variants.add_argument("--atol", type=float, default=1e-4)
    validate_variants.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance; default is the 1% eager contract.")
    validate_variants.add_argument("--json-output", default=None, help="Optional JSON report path for agent/research loops.")
    validate_variants.add_argument(
        "--include-tensor-parallel",
        action="store_true",
        help="Also validate the torchrun/NCCL tensor-parallel Llama path in the current process.",
    )
    validate_variants.set_defaults(func=run_validate_model_variants)

    vllm_bench = subparsers.add_parser(
        "vllm-bench-suite",
        help="Plan or run the vLLM latency, throughput, and serve benchmarks and plot their JSON results.",
    )
    vllm_bench.add_argument("output_dir", nargs="?", default=".torchinferno_runs/vllm-llama70b")
    vllm_bench.add_argument("--model", default=LLAMA3_70B_MODEL, help="Model path or Hugging Face repo ID.")
    vllm_bench.add_argument("--vllm-root", default="/home/bobren/local/d/vllm", help="Local vLLM checkout to prepend to PYTHONPATH.")
    vllm_bench.add_argument("--python", default=sys.executable, help="Python executable used to launch vLLM.")
    vllm_bench.add_argument("--tensor-parallel-size", type=int, default=8)
    vllm_bench.add_argument("--dtype", default="auto")
    vllm_bench.add_argument("--input-len", type=int, default=32)
    vllm_bench.add_argument("--output-len", type=int, default=128)
    vllm_bench.add_argument("--batch-size", type=int, default=8)
    vllm_bench.add_argument("--num-iters-warmup", type=int, default=10)
    vllm_bench.add_argument("--num-iters", type=int, default=30)
    vllm_bench.add_argument("--num-prompts", type=int, default=1000)
    vllm_bench.add_argument("--dataset-name", default="random")
    vllm_bench.add_argument("--request-rate", default="inf")
    vllm_bench.add_argument("--throughput-backend", default="vllm")
    vllm_bench.add_argument("--serve-backend", default="vllm")
    vllm_bench.add_argument("--base-url", default=None, help="Base URL for vLLM serve benchmark; defaults to vLLM's local endpoint.")
    vllm_bench.add_argument("--temperature", type=float, default=1.0)
    vllm_bench.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["latency", "throughput", "serve"],
        default=["latency", "throughput", "serve"],
    )
    vllm_bench.add_argument("--run", action="store_true", help="Execute the vLLM commands instead of only writing the plan.")
    vllm_bench.add_argument("--no-plot", action="store_true", help="Skip performance.html/performance.csv generation.")
    vllm_bench.add_argument("--disable-detokenize", action="store_true")
    vllm_bench.add_argument("--trust-remote-code", action="store_true")
    vllm_bench.add_argument("--enforce-eager", action="store_true")
    vllm_bench.add_argument("--max-model-len", type=int, default=None)
    vllm_bench.add_argument("--gpu-memory-utilization", type=float, default=None)
    vllm_bench.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        help="Extra EngineArgs flag/value passed to latency and throughput. Repeat as --engine-arg=--flag --engine-arg=value.",
    )
    vllm_bench.add_argument(
        "--throughput-arg",
        action="append",
        default=[],
        help="Extra argument passed only to vllm bench throughput. Repeat once per token.",
    )
    vllm_bench.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="Extra argument passed only to vllm bench serve. Repeat once per token.",
    )
    vllm_bench.set_defaults(func=run_vllm_bench_suite)

    vllm_plot = subparsers.add_parser("vllm-bench-plot", help="Plot existing vLLM benchmark JSON results.")
    vllm_plot.add_argument("output_dir")
    vllm_plot.add_argument("--output-html", default=None)
    vllm_plot.add_argument("--output-csv", default=None)
    vllm_plot.set_defaults(func=run_vllm_bench_plot)

    llama_bench = subparsers.add_parser(
        "llama-bench-suite",
        help="Plan or run the native TorchInferno Llama latency, throughput, and serve-shaped benchmarks.",
    )
    llama_bench.add_argument("output_dir", nargs="?", default=".torchinferno_runs/torchinferno-llama70b")
    llama_bench.add_argument("--model", default=LLAMA3_70B_MODEL, help="Model path or Hugging Face repo ID.")
    llama_bench.add_argument(
        "--devices",
        nargs="+",
        default=[],
        help="Devices for pipeline sharding, e.g. --devices cuda:0 cuda:1 ...; defaults to all visible CUDA devices.",
    )
    llama_bench.add_argument("--dtype", default="auto")
    llama_bench.add_argument("--parallelism", choices=["pipeline", "tensor"], default="pipeline")
    llama_bench.add_argument("--input-len", type=int, default=32)
    llama_bench.add_argument("--output-len", type=int, default=128)
    llama_bench.add_argument("--batch-size", type=int, default=8)
    llama_bench.add_argument("--num-iters-warmup", type=int, default=1)
    llama_bench.add_argument("--num-iters", type=int, default=3)
    llama_bench.add_argument("--num-prompts", type=int, default=1000)
    llama_bench.add_argument("--request-rate", default="inf")
    llama_bench.add_argument("--max-concurrency", type=int, default=256)
    llama_bench.add_argument("--temperature", type=float, default=1.0)
    llama_bench.add_argument("--seed", type=int, default=0)
    llama_bench.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["latency", "throughput", "serve"],
        default=["latency", "throughput", "serve"],
    )
    llama_bench.add_argument("--run", action="store_true", help="Execute the benchmark instead of only writing config.")
    llama_bench.add_argument("--no-plot", action="store_true")
    llama_bench.add_argument("--token", default=None)
    llama_bench.add_argument("--revision", default=None)
    llama_bench.add_argument("--cache-dir", default=None)
    llama_bench.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Write per-rank timing breakdown JSON for native tensor-parallel runs.",
    )
    llama_bench.set_defaults(func=run_llama_bench_suite)

    openai_microbench = subparsers.add_parser(
        "openai-microbench",
        help="Microbenchmark OpenAI engine request dispatch, streaming decode, and batching overhead.",
    )
    openai_microbench.add_argument("--backend", choices=["synthetic", "model"], default="synthetic")
    openai_microbench.add_argument("--model", default=None, help="Model id/path for --backend=model.")
    openai_microbench.add_argument("--model-kind", default="auto")
    openai_microbench.add_argument("--tokenizer", default=None)
    openai_microbench.add_argument("--tensor-parallel-size", type=int, default=1)
    openai_microbench.add_argument("--devices", default=None, help="Comma-separated device list for model backend.")
    openai_microbench.add_argument("--device", default=None)
    _add_dtype_argument(openai_microbench, default="auto", choices=_AUTO_DTYPE_CHOICES)
    openai_microbench.add_argument("--max-model-len", type=int, default=None)
    openai_microbench.add_argument("--trust-remote-code", action="store_true")
    openai_microbench.add_argument("--token", default=None)
    openai_microbench.add_argument("--revision", default=None)
    openai_microbench.add_argument("--cache-dir", default=None)
    openai_microbench.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    openai_microbench.add_argument("--page-size", type=int, default=16)
    openai_microbench.add_argument("--llama-parallelism", choices=["auto", "pipeline", "tensor"], default="auto")
    openai_microbench.add_argument("--max-batch-size", type=int, default=64)
    openai_microbench.add_argument("--batch-wait-ms", type=float, default=10.0)
    openai_microbench.add_argument("--prompt-tokens", type=int, default=32)
    openai_microbench.add_argument(
        "--prompt-mode",
        choices=["synthetic", "self-consistency", "few-shot"],
        default="synthetic",
        help="Prompt shape to benchmark; few-shot and self-consistency match inference-bench calculator workloads.",
    )
    openai_microbench.add_argument("--max-tokens", type=int, default=64)
    openai_microbench.add_argument("--concurrency", type=int, default=1)
    openai_microbench.add_argument("--warmup", type=int, default=2)
    openai_microbench.add_argument("--iters", type=int, default=5)
    openai_microbench.add_argument("--temperature", type=float, default=0.0)
    openai_microbench.add_argument("--vocab-size", type=int, default=256)
    openai_microbench.add_argument("--synthetic-forward-sleep-us", type=float, default=0.0)
    openai_microbench.add_argument("--compare-batcher", action="store_true")
    openai_microbench.add_argument("--phase-timings", action="store_true")
    openai_microbench.add_argument("--profile-breakdown", action="store_true")
    openai_microbench.add_argument("--json-output", default=None)
    openai_microbench.set_defaults(func=run_openai_microbench)

    openai_server_microbench = subparsers.add_parser(
        "openai-server-microbench",
        help="Launch or target the OpenAI-compatible server and benchmark chat completions over HTTP.",
    )
    openai_server_microbench.add_argument("--base-url", default=None, help="Existing /v1 base URL; when omitted a local server is launched.")
    openai_server_microbench.add_argument("--python", default=sys.executable, help="Python executable used to launch the local server.")
    openai_server_microbench.add_argument("--model", default="tiny")
    openai_server_microbench.add_argument("--model-kind", default="tiny-deepseek")
    openai_server_microbench.add_argument("--tokenizer", default="byte")
    openai_server_microbench.add_argument("--host", default="127.0.0.1")
    openai_server_microbench.add_argument("--port", type=int, default=0, help="Local server port; 0 chooses a free port.")
    openai_server_microbench.add_argument("--device", default="cpu")
    _add_dtype_argument(openai_server_microbench, default="float32", choices=_AUTO_DTYPE_CHOICES)
    openai_server_microbench.add_argument("--max-model-len", type=int, default=64)
    openai_server_microbench.add_argument("--tensor-parallel-size", type=int, default=1)
    openai_server_microbench.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    openai_server_microbench.add_argument("--page-size", type=int, default=16)
    openai_server_microbench.add_argument("--max-batch-size", type=int, default=64)
    openai_server_microbench.add_argument("--batch-wait-ms", type=float, default=10.0)
    openai_server_microbench.add_argument("--llama-parallelism", choices=["auto", "pipeline", "tensor"], default="auto")
    openai_server_microbench.add_argument("--trust-remote-code", action="store_true")
    openai_server_microbench.add_argument("--token", default=None)
    openai_server_microbench.add_argument("--revision", default=None)
    openai_server_microbench.add_argument("--cache-dir", default=None)
    openai_server_microbench.add_argument("--mode", choices=["non-stream", "stream", "both"], default="both")
    openai_server_microbench.add_argument(
        "--prompt-mode",
        choices=["synthetic", "self-consistency", "few-shot"],
        default="synthetic",
        help="Prompt shape to benchmark; few-shot and self-consistency match inference-bench calculator workloads.",
    )
    openai_server_microbench.add_argument("--prompt-tokens", type=int, default=8)
    openai_server_microbench.add_argument("--max-tokens", type=int, default=2)
    openai_server_microbench.add_argument("--concurrency", type=int, default=1)
    openai_server_microbench.add_argument("--warmup", type=int, default=1)
    openai_server_microbench.add_argument("--iters", type=int, default=3)
    openai_server_microbench.add_argument("--temperature", type=float, default=0.0)
    openai_server_microbench.add_argument("--ready-timeout-s", type=float, default=30.0)
    openai_server_microbench.add_argument("--request-timeout-s", type=float, default=30.0)
    openai_server_microbench.add_argument("--json-output", default=None)
    openai_server_microbench.set_defaults(func=run_openai_server_microbench_cli)

    smoke = subparsers.add_parser("dsv4-smoke", help="Run a DSv4 end-to-end generation smoke test.")
    _add_device_argument(smoke)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--prompt-tokens", type=int, default=8)
    smoke.add_argument("--new-tokens", type=int, default=4)
    smoke.add_argument("--vocab-size", type=int, default=128)
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--compile", action="store_true", help="Compile the DSv4 forward path with torch.compile.")
    smoke.set_defaults(func=run_dsv4_smoke)

    native = subparsers.add_parser("deepseek-smoke", help="Run a native DeepSeek-V3.2-style generation smoke test.")
    _add_device_argument(native)
    native.add_argument("--seed", type=int, default=0)
    native.add_argument("--batch-size", type=int, default=1)
    native.add_argument("--prompt-tokens", type=int, default=3)
    native.add_argument("--new-tokens", type=int, default=2)
    native.add_argument("--vocab-size", type=int, default=128)
    native.add_argument("--temperature", type=float, default=0.0)
    native.add_argument("--compile", action="store_true", help="Compile the native DeepSeek forward path.")
    native.add_argument("--no-q-lora", action="store_true", help="Use direct q_proj instead of q LoRA.")
    native.add_argument("--score-bias", action="store_true", help="Enable routed score correction bias.")
    native.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    native.add_argument("--page-size", type=int, default=16)
    native.set_defaults(func=run_deepseek_smoke)

    hf_smoke = subparsers.add_parser(
        "dsv4-hf-smoke",
        help="Load a compatible local or Hugging Face DSv4 checkpoint and generate tokens.",
    )
    _add_checkpoint_source_arguments(hf_smoke)
    _add_device_argument(hf_smoke)
    hf_smoke.add_argument("--input-ids", type=int, nargs="+", default=[1, 2, 3])
    hf_smoke.add_argument("--new-tokens", type=int, default=2)
    hf_smoke.add_argument("--temperature", type=float, default=0.0)
    _add_non_strict_argument(hf_smoke)
    hf_smoke.set_defaults(func=run_hf_smoke)

    native_hf = subparsers.add_parser(
        "deepseek-hf-smoke",
        help="Load a native DeepSeek-style checkpoint and generate tokens.",
    )
    _add_checkpoint_source_arguments(native_hf)
    _add_device_argument(native_hf)
    native_hf.add_argument("--input-ids", type=int, nargs="+", default=[1, 2, 3])
    native_hf.add_argument("--new-tokens", type=int, default=2)
    native_hf.add_argument("--temperature", type=float, default=0.0)
    _add_non_strict_argument(native_hf)
    native_hf.set_defaults(func=run_deepseek_hf_smoke)

    text = subparsers.add_parser("text-generate", help="Load a TorchInferno checkpoint, tokenize text, and generate.")
    text.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    text.add_argument("prompt")
    text.add_argument("--tokenizer", default=None, help="Tokenizer path or repo ID, defaults to model.")
    text.add_argument("--revision", default=None)
    text.add_argument("--cache-dir", default=None)
    _add_device_argument(text)
    text.add_argument("--new-tokens", type=int, default=16)
    text.add_argument("--temperature", type=float, default=0.0)
    text.add_argument("--trust-remote-code", action="store_true")
    _add_non_strict_argument(text)
    text.set_defaults(func=run_text_generate)

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

    disagg_init = subparsers.add_parser(
        "disagg-init",
        help="Generate one standalone Python file per prefill/decode rank.",
    )
    disagg_init.add_argument("output_dir")
    disagg_init.add_argument("--prefill-ranks", type=int, default=1)
    disagg_init.add_argument("--decode-ranks", type=int, default=1)
    disagg_init.add_argument("--host", default="127.0.0.1")
    disagg_init.add_argument("--base-port", type=int, default=8800)
    disagg_init.add_argument("--device", default="cpu")
    disagg_init.add_argument("--seed", type=int, default=0)
    disagg_init.add_argument("--vocab-size", type=int, default=128)
    disagg_init.add_argument("--max-seq-len", type=int, default=64)
    disagg_init.set_defaults(func=run_disagg_init)

    disagg_smoke = subparsers.add_parser(
        "disagg-smoke",
        help="Send one request through live disaggregated rank RPC endpoints.",
    )
    disagg_smoke.add_argument("--prefill-url", default="http://127.0.0.1:8800")
    disagg_smoke.add_argument("--decode-url", default="http://127.0.0.1:8801")
    disagg_smoke.add_argument("--request-id", default="req-smoke")
    disagg_smoke.add_argument("--prompt", type=int, nargs="+", default=[1, 2, 3])
    disagg_smoke.add_argument("--new-tokens", type=int, default=2)
    disagg_smoke.add_argument("--temperature", type=float, default=0.0)
    disagg_smoke.add_argument("--timeout-s", type=float, default=30.0)
    disagg_smoke.set_defaults(func=run_disagg_smoke)

    research = subparsers.add_parser("research-smoke", help="Run a tiny auto research harness.")
    research.set_defaults(func=run_research_smoke)

    helion = subparsers.add_parser(
        "helion-candidate",
        help="Benchmark a Helion-generated kernel candidate against the current TorchInferno kernel.",
    )
    helion.add_argument("--candidate", choices=["swiglu"], default="swiglu")
    helion.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    helion.add_argument("--batch-size", type=int, default=1000)
    helion.add_argument("--tokens", type=int, default=32)
    helion.add_argument("--hidden-size", type=int, default=3584)
    helion.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    helion.add_argument("--warmup", type=int, default=10)
    helion.add_argument("--iters", type=int, default=50)
    helion.add_argument("--seed", type=int, default=0)
    helion.add_argument("--min-speedup", type=float, default=1.02)
    helion.add_argument("--atol", type=float, default=2e-2)
    helion.add_argument("--rtol", type=float, default=2e-2)
    helion.add_argument("--output", default=None, help="Optional JSON report path.")
    helion.set_defaults(func=run_helion_candidate_cli)

    helion_search = subparsers.add_parser(
        "helion-search-fx",
        help="Enumerate FX node windows and benchmark supported Helion candidates against current baselines.",
    )
    helion_search.add_argument("--candidate", choices=["swiglu"], default="swiglu")
    helion_search.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    helion_search.add_argument("--batch-size", type=int, default=1000)
    helion_search.add_argument("--tokens", type=int, default=32)
    helion_search.add_argument("--hidden-size", type=int, default=3584)
    helion_search.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    helion_search.add_argument("--warmup", type=int, default=10)
    helion_search.add_argument("--iters", type=int, default=50)
    helion_search.add_argument("--seed", type=int, default=0)
    helion_search.add_argument("--min-speedup", type=float, default=1.02)
    helion_search.add_argument("--atol", type=float, default=2e-2)
    helion_search.add_argument("--rtol", type=float, default=2e-2)
    helion_search.add_argument("--min-nodes", type=int, default=1)
    helion_search.add_argument("--max-nodes", type=int, default=5)
    helion_search.add_argument("--output", default=None, help="Optional JSON report path.")
    helion_search.add_argument("--remember", default=None, help="Optional JSONL decision-store path.")
    helion_search.set_defaults(func=run_helion_search_fx_cli)

    helion_region = subparsers.add_parser(
        "helion-search-region",
        help="Trace a model region, enumerate FX windows, and report Helion macro/local opportunities.",
    )
    helion_region.add_argument("--model-kind", choices=["dsv4", "deepseek"], default="deepseek")
    helion_region.add_argument("--region", choices=["mlp", "expert", "swiglu", "attention", "attn"], default="mlp")
    helion_region.add_argument("--device", default=None, help="Benchmark device for generated candidates.")
    helion_region.add_argument("--trace-device", default="cpu", help="Device used for torch-native FX tracing.")
    helion_region.add_argument("--batch-size", type=int, default=2)
    helion_region.add_argument("--tokens", type=int, default=8)
    helion_region.add_argument("--hidden-size", type=int, default=64)
    helion_region.add_argument("--intermediate-size", type=int, default=128)
    helion_region.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    helion_region.add_argument("--warmup", type=int, default=5)
    helion_region.add_argument("--iters", type=int, default=20)
    helion_region.add_argument("--seed", type=int, default=0)
    helion_region.add_argument("--min-speedup", type=float, default=1.02)
    helion_region.add_argument("--atol", type=float, default=2e-2)
    helion_region.add_argument("--rtol", type=float, default=2e-2)
    helion_region.add_argument("--min-nodes", type=int, default=1)
    helion_region.add_argument("--max-nodes", type=int, default=5)
    helion_region.add_argument("--output", default=None, help="Optional JSON report path.")
    helion_region.add_argument("--remember", default=None, help="Optional JSONL decision-store path.")
    helion_region.set_defaults(func=run_helion_search_region_cli)

    traffic = subparsers.add_parser("traffic-smoke", help="Simulate bursty request traffic through prefill/decode ranks.")
    traffic.add_argument("--requests", type=int, default=8)
    traffic.add_argument("--prompt-min", type=int, default=2)
    traffic.add_argument("--prompt-max", type=int, default=32)
    traffic.add_argument("--decode-min", type=int, default=1)
    traffic.add_argument("--decode-max", type=int, default=8)
    traffic.add_argument("--burst-size", type=int, default=4)
    traffic.add_argument("--burst-gap-us", type=float, default=1000.0)
    traffic.add_argument("--in-burst-gap-us", type=float, default=10.0)
    traffic.add_argument("--prefill-ranks", type=int, default=1)
    traffic.add_argument("--decode-ranks", type=int, default=1)
    traffic.add_argument("--prefill-us-per-token", type=float, default=2.0)
    traffic.add_argument("--decode-us-per-token", type=float, default=4.0)
    traffic.add_argument("--network-latency-us", type=float, default=10.0)
    traffic.add_argument("--print-stages", type=int, default=8)
    traffic.add_argument("--seed", type=int, default=0)
    traffic.set_defaults(func=run_traffic_smoke)

    serve = subparsers.add_parser("serve-smoke", help="Run the token-level continuous serving engine.")
    _add_device_argument(serve)
    serve.add_argument("--cache-backend", choices=["dense", "paged"], default="paged")
    serve.add_argument("--page-size", type=int, default=16)
    serve.add_argument("--max-active-requests", type=int, default=2)
    serve.add_argument("--new-tokens", type=int, default=2)
    serve.add_argument("--vocab-size", type=int, default=128)
    serve.add_argument("--temperature", type=float, default=0.0)
    serve.add_argument("--seed", type=int, default=0)
    serve.set_defaults(func=run_serve_smoke)

    openai_serve = subparsers.add_parser(
        "openai-server",
        help="Serve TorchInferno through the OpenAI-compatible chat completions API.",
    )
    openai_serve.add_argument("--model", required=True, help="Model id or local checkpoint path.")
    openai_serve.add_argument("--host", default="0.0.0.0")
    openai_serve.add_argument("--port", type=int, default=8000)
    openai_serve.add_argument("--model-kind", default="auto")
    openai_serve.add_argument("--tokenizer", default=None)
    openai_serve.add_argument("--tensor-parallel-size", type=int, default=1)
    openai_serve.add_argument("--devices", default=None, help="Comma-separated device list.")
    openai_serve.add_argument("--device", default=None)
    _add_dtype_argument(openai_serve, default="auto", choices=_AUTO_DTYPE_CHOICES)
    openai_serve.add_argument("--max-model-len", type=int, default=None)
    openai_serve.add_argument("--trust-remote-code", action="store_true")
    openai_serve.add_argument("--token", default=None)
    openai_serve.add_argument("--revision", default=None)
    openai_serve.add_argument("--cache-dir", default=None)
    openai_serve.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    openai_serve.add_argument("--page-size", type=int, default=16)
    openai_serve.add_argument("--max-batch-size", type=int, default=64)
    openai_serve.add_argument("--batch-wait-ms", type=float, default=10.0)
    openai_serve.add_argument(
        "--single-request-admission-wait-ms",
        type=float,
        default=None,
        help=(
            "Optional wait before a lone request takes the direct model path. "
            "Lower values minimize TTFT; higher values preserve a short batching window."
        ),
    )
    openai_serve.add_argument(
        "--llama-parallelism",
        choices=["auto", "pipeline", "tensor"],
        default="auto",
    )
    openai_serve.set_defaults(func=run_openai_server)

    perf = subparsers.add_parser("perf-smoke", help="Benchmark paged attention reference and specialized decode paths.")
    _add_device_argument(perf)
    perf.add_argument("--backend", choices=[backend.value for backend in KernelBackend], default=KernelBackend.AUTO.value)
    perf.add_argument("--heads", type=int, default=8)
    perf.add_argument("--seq-len", type=int, default=128)
    perf.add_argument("--head-dim", type=int, default=64)
    perf.add_argument("--value-dim", type=int, default=64)
    perf.add_argument("--page-size", type=int, default=16)
    perf.add_argument("--warmup", type=int, default=3)
    perf.add_argument("--iters", type=int, default=10)
    perf.add_argument("--seed", type=int, default=0)
    perf.set_defaults(func=run_perf_smoke)

    profile = subparsers.add_parser(
        "profile-run",
        help="Run one model generation and write graph/profile/memory artifacts.",
    )
    profile.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Artifact directory; defaults to .torchinferno_runs/<model>-<timestamp>.",
    )
    profile.add_argument("--model-kind", choices=["dsv4", "deepseek"], default="dsv4")
    _add_device_argument(profile)
    _add_dtype_argument(profile, default="float32", choices=_PROFILE_DTYPE_CHOICES)
    profile.add_argument("--seed", type=int, default=0)
    profile.add_argument("--batch-size", type=int, default=1)
    profile.add_argument("--prompt-tokens", type=int, default=8)
    profile.add_argument("--new-tokens", type=int, default=8)
    profile.add_argument("--vocab-size", type=int, default=128)
    profile.add_argument("--temperature", type=float, default=0.0)
    profile.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model before the profiled generation run.",
    )
    profile.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    profile.add_argument("--page-size", type=int, default=16)
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--no-graph", action="store_true", help="Skip make_fx graph capture.")
    profile.add_argument("--fake-graph", action="store_true", help="Try graph capture under FakeTensorMode.")
    profile.add_argument("--require-graph", action="store_true", help="Fail the run if graph capture fails.")
    _add_profile_capture_arguments(profile)
    profile.set_defaults(func=run_profile_run)

    profile_timeslice = subparsers.add_parser(
        "profile-timeslice",
        help="Measure one representative generation and replay it across time-sliced virtual GPUs.",
    )
    profile_timeslice.add_argument("output_dir")
    profile_timeslice.add_argument("--model-kind", choices=["dsv4", "deepseek"], default="dsv4")
    _add_device_argument(profile_timeslice)
    _add_dtype_argument(profile_timeslice, default="float32", choices=_PROFILE_DTYPE_CHOICES)
    profile_timeslice.add_argument("--seed", type=int, default=0)
    profile_timeslice.add_argument("--batch-size", type=int, default=1)
    profile_timeslice.add_argument("--prompt-tokens", type=int, default=8)
    profile_timeslice.add_argument("--new-tokens", type=int, default=8)
    profile_timeslice.add_argument("--vocab-size", type=int, default=128)
    profile_timeslice.add_argument("--temperature", type=float, default=0.0)
    profile_timeslice.add_argument(
        "--compile",
        action="store_true",
        help="Compile the representative model before profiling.",
    )
    profile_timeslice.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    profile_timeslice.add_argument("--page-size", type=int, default=16)
    profile_timeslice.add_argument("--warmup", type=int, default=1)
    profile_timeslice.add_argument("--iters", type=int, default=3)
    profile_timeslice.add_argument("--virtual-gpus", type=int, default=4)
    profile_timeslice.add_argument("--time-slice-us", type=float, default=1000.0)
    profile_timeslice.add_argument("--context-switch-us", type=float, default=0.0)
    profile_timeslice.add_argument("--arrival-gap-us", type=float, default=0.0)
    profile_timeslice.add_argument(
        "--profile-scale",
        type=float,
        default=1.0,
        help="Multiply measured representative latency before replaying virtual ranks.",
    )
    _add_profile_capture_arguments(profile_timeslice)
    profile_timeslice.set_defaults(func=run_profile_timeslice)

    profile_offload = subparsers.add_parser(
        "profile-offload",
        help="Run explicit CPU-offloaded model replay and subtract movement overhead from timing.",
    )
    profile_offload.add_argument("output_dir")
    profile_offload.add_argument("--checkpoint", default=None, help="Optional local/Hugging Face TorchInferno checkpoint.")
    profile_offload.add_argument("--model-kind", choices=["dsv4", "deepseek"], default="dsv4")
    _add_device_argument(profile_offload)
    _add_dtype_argument(profile_offload, default="float32", choices=_PROFILE_DTYPE_CHOICES)
    profile_offload.add_argument("--seed", type=int, default=0)
    profile_offload.add_argument("--batch-size", type=int, default=1)
    profile_offload.add_argument("--prompt-tokens", type=int, default=8)
    profile_offload.add_argument("--new-tokens", type=int, default=1)
    profile_offload.add_argument("--vocab-size", type=int, default=128)
    profile_offload.add_argument("--temperature", type=float, default=0.0)
    profile_offload.add_argument("--warmup", type=int, default=0)
    profile_offload.add_argument("--iters", type=int, default=1)
    profile_offload.add_argument(
        "--activation-offload",
        action="store_true",
        help="Move activations back to CPU between staged modules.",
    )
    profile_offload.add_argument("--revision", default=None)
    profile_offload.add_argument("--cache-dir", default=None)
    _add_non_strict_argument(profile_offload)
    profile_offload.set_defaults(func=run_profile_offload)

    profile_region = subparsers.add_parser(
        "profile-region",
        help="Profile one named model region and emit focused graph/profile artifacts.",
    )
    profile_region.add_argument("output_dir")
    profile_region.add_argument("--region", required=True, help="Module path such as layers.0.attn or model.layers.0.self_attn.")
    profile_region.add_argument("--model-kind", choices=["dsv4", "deepseek"], default="dsv4")
    _add_device_argument(profile_region)
    _add_dtype_argument(profile_region, default="float32", choices=_PROFILE_DTYPE_CHOICES)
    profile_region.add_argument("--seed", type=int, default=0)
    profile_region.add_argument("--batch-size", type=int, default=1)
    profile_region.add_argument("--tokens", type=int, default=8)
    profile_region.add_argument("--vocab-size", type=int, default=128)
    profile_region.add_argument("--warmup", type=int, default=3)
    profile_region.add_argument("--iters", type=int, default=10)
    _add_profile_capture_arguments(profile_region, graph_options=True)
    profile_region.set_defaults(func=run_profile_region)

    profile_pattern = subparsers.add_parser(
        "profile-pattern",
        help="Profile a reference graph pattern before and after registered replacement passes.",
    )
    profile_pattern.add_argument("output_dir")
    profile_pattern.add_argument("--pattern", choices=["fused-rmsnorm-swiglu"], default="fused-rmsnorm-swiglu")
    _add_device_argument(profile_pattern)
    _add_dtype_argument(profile_pattern, default="float32", choices=_PROFILE_DTYPE_CHOICES)
    profile_pattern.add_argument("--seed", type=int, default=0)
    profile_pattern.add_argument("--batch-size", type=int, default=1)
    profile_pattern.add_argument("--tokens", type=int, default=8)
    profile_pattern.add_argument("--hidden-size", type=int, default=128)
    profile_pattern.add_argument("--warmup", type=int, default=3)
    profile_pattern.add_argument("--iters", type=int, default=10)
    profile_pattern.add_argument("--no-apply-passes", action="store_true", help="Profile only the reference graph.")
    _add_profile_capture_arguments(
        profile_pattern,
        graph_options=True,
        chrome_trace_help="Do not export chrome_trace JSON files.",
    )
    profile_pattern.set_defaults(func=run_profile_pattern)

    profile_subgraph = subparsers.add_parser(
        "profile-subgraph",
        help="Extract node ids from a prior profile-run graph and profile only that FX subgraph.",
    )
    profile_subgraph.add_argument("output_dir")
    profile_subgraph.add_argument("--source-run", required=True, help="Artifact directory produced by profile-run.")
    profile_subgraph.add_argument(
        "--nodes",
        nargs="+",
        required=True,
        help="Node ids, comma lists, or inclusive ranges such as 42 43 44 or 42:50.",
    )
    profile_subgraph.add_argument("--device", default=None, help="Override source run device.")
    profile_subgraph.add_argument("--warmup", type=int, default=3)
    profile_subgraph.add_argument("--iters", type=int, default=10)
    _add_profile_capture_arguments(profile_subgraph)
    profile_subgraph.set_defaults(func=run_profile_subgraph)

    profile_nodes = subparsers.add_parser(
        "profile-nodes",
        help="Print labeled node ids from a profile-run graph.json or artifact directory.",
    )
    profile_nodes.add_argument("graph_or_run", help="graph.json path or a profile-run artifact directory.")
    profile_nodes.add_argument("--grep", default=None, help="Filter by id, name, op, or target substring.")
    profile_nodes.set_defaults(func=run_profile_nodes)

    capture = subparsers.add_parser("capture-logits", help="Capture a known-logit reference for a checkpoint.")
    capture.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    capture.add_argument("output")
    capture.add_argument("--input-ids", type=int, nargs="+", required=True)
    capture.add_argument("--revision", default=None)
    capture.add_argument("--cache-dir", default=None)
    capture.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    capture.add_argument("--atol", type=float, default=1e-4)
    capture.add_argument("--rtol", type=float, default=1e-4)
    capture.add_argument("--description", default="")
    capture.add_argument("--non-strict", action="store_true", help="Allow missing or unexpected weight keys.")
    capture.set_defaults(func=run_capture_logits)

    validate = subparsers.add_parser("validate-logits", help="Validate a checkpoint against a known-logit reference.")
    validate.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    validate.add_argument("reference")
    validate.add_argument("--revision", default=None)
    validate.add_argument("--cache-dir", default=None)
    validate.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    validate.add_argument("--non-strict", action="store_true", help="Allow missing or unexpected weight keys.")
    validate.set_defaults(func=run_validate_logits)

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

    native_audit = subparsers.add_parser(
        "deepseek-audit",
        help="Audit a native DeepSeek-style checkpoint for TorchInferno native loading.",
    )
    native_audit.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    native_audit.add_argument("--revision", default=None)
    native_audit.add_argument("--cache-dir", default=None)
    native_audit.set_defaults(func=run_deepseek_audit)

    native_convert = subparsers.add_parser(
        "deepseek-convert",
        help="Convert a native DeepSeek-style checkpoint into TorchInferno native format.",
    )
    native_convert.add_argument("model", help="Local checkpoint directory or Hugging Face repo ID.")
    native_convert.add_argument("output_dir")
    native_convert.add_argument("--revision", default=None)
    native_convert.add_argument("--cache-dir", default=None)
    native_convert.add_argument("--dtype", default=None, help="Optional output dtype: float32, float16, or bfloat16.")
    native_convert.add_argument("--max-shard-size", default="5GB")
    native_convert.add_argument("--allow-partial", action="store_true", help="Write only convertible tensors for debugging.")
    native_convert.set_defaults(func=run_deepseek_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
