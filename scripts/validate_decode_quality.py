#!/usr/bin/env python3
"""Measure held-out next-token quality through the online ragged decode path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--scored-tokens", type=int, default=64)
    parser.add_argument(
        "--runtime-profile",
        choices=("reference", "serving"),
        default="reference",
        help="Evaluate the BF16 reference or the production serving defaults.",
    )
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--max-relative-nll-increase", type=float, default=0.01)
    parser.add_argument("--max-top1-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure_runtime_profile(args: argparse.Namespace) -> None:
    if args.runtime_profile == "serving":
        from torchinferno.openai_server import (
            OpenAIServerConfig,
            _apply_tensor_parallel_serving_defaults,
        )

        _apply_tensor_parallel_serving_defaults(
            OpenAIServerConfig(
                model=args.checkpoint,
                model_kind="llama3",
                tensor_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
            )
        )
        return

    for name in (
        "TORCHINFERNO_FP8_DECODE",
        "TORCHINFERNO_FP8_PREFILL",
        "TORCHINFERNO_FP8_PER_TOKEN_SCALE",
        "TORCHINFERNO_FP8_QKV",
        "TORCHINFERNO_FP8_LM_HEAD",
        "TORCHINFERNO_MARLIN_INT4_DECODE",
        "TORCHINFERNO_MARLIN_INT4_DOWN",
    ):
        os.environ[name] = "0"


def _source_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compare_with_reference(
    reference_path: Path,
    result: dict[str, object],
) -> dict[str, float | int]:
    reference = json.loads(reference_path.read_text())
    if not isinstance(reference, dict):
        raise ValueError("reference quality artifact must contain a JSON object")
    for field in (
        "checkpoint",
        "dataset",
        "dataset_config",
        "split",
        "sequences",
        "context_tokens",
        "scored_tokens",
        "num_scored_tokens",
        "offsets",
        "targets",
    ):
        if reference.get(field) != result.get(field):
            raise ValueError(f"reference quality artifact has mismatched {field}")
    if reference.get("runtime_profile") != "reference":
        raise ValueError("reference quality artifact was not produced by the reference profile")
    if result.get("runtime_profile") != "serving":
        raise ValueError("quality comparisons require the serving runtime profile")
    serving_validation = result.get("serving_validation")
    if not isinstance(serving_validation, dict):
        raise ValueError("serving quality artifact is missing validation metadata")
    if serving_validation.get("fp8_o_projection_scope") is not True:
        raise ValueError("serving quality artifact did not exercise FP8 O projection")
    if int(serving_validation.get("prepared_fp8_o_layers", 0)) < 1:
        raise ValueError("serving quality artifact prepared no FP8 O weights")
    if serving_validation.get("failed_fp8_o_layers") != 0:
        raise ValueError("serving quality artifact observed FP8 O failures")
    if serving_validation.get("successful_fp8_o_layers") != serving_validation.get(
        "prepared_fp8_o_layers"
    ):
        raise ValueError("serving quality artifact did not exercise every FP8 O layer")
    if int(serving_validation.get("successful_fp8_o_calls", 0)) < int(
        serving_validation["prepared_fp8_o_layers"]
    ):
        raise ValueError("serving quality artifact observed no successful FP8 O pass")
    environment = result.get("torchinferno_environment")
    if not isinstance(environment, dict):
        raise ValueError("serving quality artifact is missing runtime environment")
    for name in (
        "TORCHINFERNO_FP8_O_PROJ",
        "TORCHINFERNO_FP8_O_PROJ_GREEDY_ONLY",
    ):
        if environment.get(name) != "1":
            raise ValueError(f"serving quality artifact did not enable {name}")
    reference_nll = float(reference["mean_nll"])
    serving_nll = float(result["mean_nll"])
    if not math.isfinite(reference_nll) or reference_nll <= 0.0:
        raise ValueError("reference mean NLL must be finite and positive")
    if not math.isfinite(serving_nll):
        raise ValueError("serving mean NLL must be finite")
    reference_accuracy = float(reference["top1_accuracy"])
    serving_accuracy = float(result["top1_accuracy"])
    reference_predictions = [
        int(token)
        for sequence in reference["predictions"]
        for token in sequence
    ]
    serving_predictions = [
        int(token)
        for sequence in result["predictions"]
        for token in sequence
    ]
    if len(reference_predictions) != len(serving_predictions):
        raise ValueError("reference and serving artifacts score different token counts")
    return {
        "relative_nll_increase": (serving_nll / reference_nll) - 1.0,
        "top1_accuracy_drop": reference_accuracy - serving_accuracy,
        "prediction_changes": sum(
            left != right
            for left, right in zip(reference_predictions, serving_predictions)
        ),
    }


def _validate_reference_args(reference_path: Path, args: argparse.Namespace) -> None:
    reference = json.loads(reference_path.read_text())
    if not isinstance(reference, dict):
        raise ValueError("reference quality artifact must contain a JSON object")
    expected = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sequences": args.sequences,
        "context_tokens": args.context_tokens,
        "scored_tokens": args.scored_tokens,
    }
    for field, value in expected.items():
        if reference.get(field) != value:
            raise ValueError(f"reference quality artifact has mismatched {field}")


def _held_out_tokens(args: argparse.Namespace) -> list[int]:
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.model_max_length = max(tokenizer.model_max_length, len(text))
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _sequence_offsets(total_tokens: int, *, count: int, window: int) -> list[int]:
    if count < 1 or window < 2 or total_tokens < window:
        raise ValueError("the held-out corpus is too small for the requested windows")
    if count == 1:
        return [0]
    stride = (total_tokens - window) // (count - 1)
    return [index * stride for index in range(count)]


def main() -> None:
    args = _parse_args()
    if args.context_tokens < 1 or args.scored_tokens < 1 or args.sequences < 1:
        raise ValueError("sequence, context, and score sizes must be positive")
    if args.max_relative_nll_increase < 0.0 or args.max_top1_accuracy_drop < 0.0:
        raise ValueError("quality regression limits must be nonnegative")
    if args.reference_output is not None:
        _validate_reference_args(args.reference_output, args)

    _configure_runtime_profile(args)

    model = Llama3TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype="bfloat16",
    ).eval()
    prepared_fp8_o_layers = 0
    fp8_o_projection_scope = False
    fp8_o_scope = nullcontext
    fp8_o_observer_scope = nullcontext({})
    if args.runtime_profile == "serving":
        from torchinferno.models.llama3.tensor_parallel import (
            _fp8_o_projection,
            _observe_fp8_o_projection_successes,
            validate_symm_mem_allreduce_collective,
        )

        validate_symm_mem_allreduce_collective(model, model.device)
        if os.environ.get("TORCHINFERNO_FP8_O_PROJ") != "1":
            raise RuntimeError("serving quality validation requires FP8 O projection")
        if os.environ.get("TORCHINFERNO_FP8_O_PROJ_GREEDY_ONLY") != "1":
            raise RuntimeError(
                "serving quality validation requires the greedy-only FP8 O policy"
            )
        prepared_fp8_o_layers = model.prepare_fp8_o_projection_weights()
        if prepared_fp8_o_layers < 1:
            raise RuntimeError("serving quality validation prepared no FP8 O weights")
        fp8_o_projection_scope = True

        fp8_o_scope = partial(_fp8_o_projection, True)
        fp8_o_observer_scope = _observe_fp8_o_projection_successes()
    rank = dist.get_rank() if dist.is_initialized() else 0
    tokens = _held_out_tokens(args)
    window = args.context_tokens + args.scored_tokens
    offsets = _sequence_offsets(len(tokens), count=args.sequences, window=window)

    total_nll = 0.0
    total_correct = 0
    total_scored = 0
    predictions: list[list[int]] = []
    targets: list[list[int]] = []

    with fp8_o_observer_scope as fp8_o_successes, torch.inference_mode():
        for sequence_index, offset in enumerate(offsets):
            sample = tokens[offset : offset + window]
            cache = model.allocate_cache(1, window + 1)
            prompt = torch.tensor(
                [sample[: args.context_tokens]],
                dtype=torch.long,
                device=model.device,
            )
            with fp8_o_scope():
                local_logits, _ = model.forward(
                    prompt,
                    cache=cache,
                    use_cache=True,
                    return_last_logits_only=True,
                    return_sharded_logits=True,
                )

            sequence_predictions: list[int] = []
            sequence_targets = sample[args.context_tokens :]
            for score_index, target in enumerate(sequence_targets):
                logits = model._gather_logits(local_logits)[:, -1, :].float()
                target_tensor = torch.tensor([target], dtype=torch.long, device=model.device)
                nll = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="sum")
                predicted = int(torch.argmax(logits, dim=-1).item())
                total_nll += float(nll.item())
                total_correct += int(predicted == target)
                total_scored += 1
                sequence_predictions.append(predicted)

                if score_index + 1 < len(sequence_targets):
                    seq_len = args.context_tokens + score_index
                    with fp8_o_scope():
                        local_logits = model.decode_ragged_logits(
                            torch.tensor([[target]], dtype=torch.long, device=model.device),
                            cache,
                            seq_lens=torch.tensor(
                                [seq_len],
                                dtype=torch.long,
                                device=model.device,
                            ),
                        )

            predictions.append(sequence_predictions)
            targets.append(sequence_targets)
            if rank == 0:
                print(
                    f"quality sequence {sequence_index + 1}/{len(offsets)} complete",
                    flush=True,
                )

    eligible_fp8_o_layer_ids = {
        id(layer)
        for layer in model.layers
        if getattr(layer, "_fp8_per_token_o_wq", None) is not None
        and getattr(layer, "_fp8_per_token_o_ws", None) is not None
    }
    failed_fp8_o_layers = sum(
        bool(getattr(layer, "_fp8_per_token_o_failed", False))
        for layer in model.layers
    )
    successful_fp8_o_layer_ids = set(fp8_o_successes)
    if fp8_o_projection_scope:
        if failed_fp8_o_layers:
            raise RuntimeError("serving quality validation observed FP8 O failures")
        if successful_fp8_o_layer_ids != eligible_fp8_o_layer_ids:
            raise RuntimeError(
                "serving quality validation did not execute every prepared FP8 O layer"
            )

    if rank == 0:
        mean_nll = total_nll / total_scored
        result = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "runtime_profile": args.runtime_profile,
            "source_commit": _source_commit(),
            "source_dirty": _source_dirty(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "sglang_kernel_version": _package_version("sglang-kernel"),
            "torchinferno_environment": {
                name: value
                for name, value in sorted(os.environ.items())
                if name.startswith("TORCHINFERNO_")
            },
            "serving_validation": {
                "failed_fp8_o_layers": failed_fp8_o_layers,
                "fp8_o_projection_scope": fp8_o_projection_scope,
                "prepared_fp8_o_layers": prepared_fp8_o_layers,
                "successful_fp8_o_calls": sum(fp8_o_successes.values()),
                "successful_fp8_o_layers": len(successful_fp8_o_layer_ids),
            },
            "sequences": args.sequences,
            "context_tokens": args.context_tokens,
            "scored_tokens": args.scored_tokens,
            "num_scored_tokens": total_scored,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "top1_accuracy": total_correct / total_scored,
            "offsets": offsets,
            "predictions": predictions,
            "targets": targets,
        }
        comparison = None
        if args.reference_output is not None:
            comparison = _compare_with_reference(args.reference_output, result)
            result["reference_comparison"] = comparison
            result["quality_gate"] = {
                "max_relative_nll_increase": args.max_relative_nll_increase,
                "max_top1_accuracy_drop": args.max_top1_accuracy_drop,
                "passed": (
                    comparison["relative_nll_increase"]
                    <= args.max_relative_nll_increase
                    and comparison["top1_accuracy_drop"]
                    <= args.max_top1_accuracy_drop
                ),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: result[key] for key in ("mean_nll", "perplexity", "top1_accuracy")}))
        if comparison is not None and not result["quality_gate"]["passed"]:
            raise RuntimeError(
                "serving quality exceeds the configured reference regression limits"
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
