from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

import torch

from torchinferno.models.dsv4 import DSv4ForCausalLM


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    tokens: tuple[int, ...]


def run_continuous_batch(
    model: DSv4ForCausalLM,
    requests: Iterable[InferenceRequest],
    *,
    device: torch.device,
    temperature: float = 0.0,
) -> list[InferenceResult]:
    """Run a simple continuous-batching harness.

    Requests are bucketed by prompt length so the current DSv4 model can execute
    dense tensors while still accepting ragged traffic at the API boundary.
    """

    request_list = list(requests)
    indexed_requests = list(enumerate(request_list))
    sorted_requests = sorted(indexed_requests, key=lambda item: len(item[1].prompt))
    indexed_results: list[tuple[int, InferenceResult]] = []
    for _, bucket_iter in groupby(sorted_requests, key=lambda item: len(item[1].prompt)):
        indexed_bucket = list(bucket_iter)
        bucket = [request for _, request in indexed_bucket]
        prompts = torch.tensor([request.prompt for request in bucket], device=device, dtype=torch.long)
        max_new_tokens = max(request.max_new_tokens for request in bucket)
        generated = model.generate(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
        for row, (original_index, request) in zip(generated.tolist(), indexed_bucket):
            total = len(request.prompt) + request.max_new_tokens
            indexed_results.append((original_index, InferenceResult(request.request_id, tuple(row[:total]))))
    return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]
