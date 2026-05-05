from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from torchinferno.runtime.prefix_cache import PrefixCacheIndex


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    arrival_step: int = 0
    eos_token_id: Optional[int] = None


@dataclass(frozen=True)
class ServingResult:
    request_id: str
    tokens: tuple[int, ...]
    prefix_hit_tokens: int
    arrival_step: int
    started_step: int
    finished_step: int


@dataclass
class _ActiveRequest:
    original_index: int
    request: ServingRequest
    tokens: list[int]
    generated: int
    cache: object
    last_token: int
    prefix_hit_tokens: int
    started_step: int


class ContinuousBatchEngine:
    """Token-step continuous serving harness.

    This intentionally stays single-process and deterministic. It is the first
    integration point for cache backend policy, prefix-aware routing, admission,
    and decode stepping without requiring a production server.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        cache_backend: str = "dense",
        page_size: int = 16,
        temperature: float = 0.0,
        max_active_requests: int = 16,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        self.model = model.to(device).eval()
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.temperature = temperature
        self.max_active_requests = max_active_requests
        self.prefix_cache = PrefixCacheIndex()

    @torch.inference_mode()
    def run(self, requests: list[ServingRequest]) -> list[ServingResult]:
        waiting = sorted(enumerate(requests), key=lambda item: (item[1].arrival_step, item[0]))
        active: list[_ActiveRequest] = []
        indexed_results: list[tuple[int, ServingResult]] = []
        step = 0
        cursor = 0

        while cursor < len(waiting) or active:
            while (
                cursor < len(waiting)
                and waiting[cursor][1].arrival_step <= step
                and len(active) < self.max_active_requests
            ):
                original_index, request = waiting[cursor]
                cursor += 1
                state_or_result = self._prefill(original_index, request, step)
                if isinstance(state_or_result, ServingResult):
                    indexed_results.append((original_index, state_or_result))
                else:
                    active.append(state_or_result)

            next_active: list[_ActiveRequest] = []
            for state in active:
                decoded = self._decode_one(state, step + 1)
                if isinstance(decoded, ServingResult):
                    indexed_results.append((state.original_index, decoded))
                else:
                    next_active.append(decoded)
            active = next_active
            step += 1

            if cursor < len(waiting) and not active and waiting[cursor][1].arrival_step > step:
                step = waiting[cursor][1].arrival_step

        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    def _prefill(self, original_index: int, request: ServingRequest, step: int) -> _ActiveRequest | ServingResult:
        if not request.prompt:
            raise ValueError("request prompt must contain at least one token")
        match, _ = self.prefix_cache.lookup(request.prompt)
        prefix_hit_tokens = match.depth
        self.prefix_cache.add(request.request_id, request.prompt)
        if request.max_new_tokens == 0:
            return ServingResult(
                request.request_id,
                request.prompt,
                prefix_hit_tokens,
                request.arrival_step,
                step,
                step,
            )

        cache = self._allocate_cache(1, len(request.prompt) + request.max_new_tokens)
        input_ids = torch.tensor([request.prompt], device=self.device, dtype=torch.long)
        logits, cache = self.model(input_ids, cache=cache, use_cache=True)  # type: ignore[misc]
        next_token = int(sample_next_token(logits[:, -1, :], self.temperature).item())
        tokens = [*request.prompt, next_token]
        return _ActiveRequest(
            original_index=original_index,
            request=request,
            tokens=tokens,
            generated=1,
            cache=cache,
            last_token=next_token,
            prefix_hit_tokens=prefix_hit_tokens,
            started_step=step,
        )

    def _decode_one(self, state: _ActiveRequest, step: int) -> _ActiveRequest | ServingResult:
        if state.request.eos_token_id is not None and state.last_token == state.request.eos_token_id:
            return self._finish(state, step)
        if state.generated >= state.request.max_new_tokens:
            return self._finish(state, step)

        input_ids = torch.tensor([[state.last_token]], device=self.device, dtype=torch.long)
        logits, cache = self.model(input_ids, cache=state.cache, use_cache=True)  # type: ignore[misc]
        next_token = int(sample_next_token(logits[:, -1, :], self.temperature).item())
        state.cache = cache
        state.tokens.append(next_token)
        state.generated += 1
        state.last_token = next_token
        if state.request.eos_token_id is not None and next_token == state.request.eos_token_id:
            return self._finish(state, step)
        if state.generated >= state.request.max_new_tokens:
            return self._finish(state, step)
        return state

    def _finish(self, state: _ActiveRequest, step: int) -> ServingResult:
        return ServingResult(
            state.request.request_id,
            tuple(state.tokens),
            state.prefix_hit_tokens,
            state.request.arrival_step,
            state.started_step,
            step,
        )

    def _allocate_cache(self, batch_size: int, max_seq_len: int) -> object:
        allocate_cache = getattr(self.model, "allocate_cache")
        try:
            return allocate_cache(
                batch_size,
                max_seq_len=max_seq_len,
                device=self.device,
                cache_backend=self.cache_backend,
                page_size=self.page_size,
            )
        except TypeError:
            if self.cache_backend != "dense":
                raise ValueError(f"model does not support cache_backend={self.cache_backend}") from None
            return allocate_cache(batch_size, max_seq_len=max_seq_len, device=self.device)


def sample_next_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
