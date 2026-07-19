#!/usr/bin/env python3
"""Measure held-out next-token quality through the online ragged decode path."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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

    model = Llama3TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype="bfloat16",
    ).eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    tokens = _held_out_tokens(args)
    window = args.context_tokens + args.scored_tokens
    offsets = _sequence_offsets(len(tokens), count=args.sequences, window=window)

    total_nll = 0.0
    total_correct = 0
    total_scored = 0
    predictions: list[list[int]] = []
    targets: list[list[int]] = []

    with torch.inference_mode():
        for sequence_index, offset in enumerate(offsets):
            sample = tokens[offset : offset + window]
            cache = model.allocate_cache(1, window + 1)
            prompt = torch.tensor(
                [sample[: args.context_tokens]],
                dtype=torch.long,
                device=model.device,
            )
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
                    local_logits = model.decode_ragged_logits(
                        torch.tensor([[target]], dtype=torch.long, device=model.device),
                        cache,
                        seq_lens=torch.tensor([seq_len], dtype=torch.long, device=model.device),
                    )

            predictions.append(sequence_predictions)
            targets.append(sequence_targets)
            if rank == 0:
                print(
                    f"quality sequence {sequence_index + 1}/{len(offsets)} complete",
                    flush=True,
                )

    if rank == 0:
        mean_nll = total_nll / total_scored
        result = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: result[key] for key in ("mean_nll", "perplexity", "top1_accuracy")}))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
