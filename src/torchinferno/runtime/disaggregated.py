from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
import threading
import time
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor

from torchinferno.runtime.sampling import sample_next_token


DisaggregatedRole = Literal["prefill", "decode"]


@dataclass(frozen=True)
class DisaggregatedTopology:
    global_rank: int
    world_size: int
    tensor_parallel_size: int
    role: DisaggregatedRole
    role_rank: int
    device: torch.device
    control_group: dist.ProcessGroup
    prefill_group: object
    decode_group: object
    role_group: dist.ProcessGroup
    transfer_group: dist.ProcessGroup

    @property
    def coordinator(self) -> bool:
        return self.global_rank == 0

    @property
    def decode_root(self) -> int:
        return self.tensor_parallel_size

    @property
    def peer_rank(self) -> int:
        if self.role == "prefill":
            return self.global_rank + self.tensor_parallel_size
        return self.global_rank - self.tensor_parallel_size


def initialize_disaggregated_topology(
    tensor_parallel_size: int,
    *,
    timeout_s: int = 1800,
) -> DisaggregatedTopology:
    """Split one single-node CUDA world into equal prefill and decode groups."""

    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("disaggregated prefill/decode requires CUDA")
    if not dist.is_available():
        raise RuntimeError("disaggregated prefill/decode requires torch.distributed")
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=max(1, int(timeout_s))))
    if dist.get_backend() != "nccl":
        raise RuntimeError("disaggregated prefill/decode requires an NCCL world process group")

    world_size = dist.get_world_size()
    expected_world_size = 2 * tensor_parallel_size
    if world_size != expected_world_size:
        raise RuntimeError(
            "disaggregated prefill/decode requires exactly two equal tensor-parallel groups: "
            f"WORLD_SIZE={world_size}, expected {expected_world_size}"
        )
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if local_world_size != world_size:
        raise RuntimeError("disaggregated prefill/decode currently supports one node only")

    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} does not address a visible CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    all_ranks = list(range(expected_world_size))
    prefill_ranks = list(range(tensor_parallel_size))
    decode_ranks = list(range(tensor_parallel_size, expected_world_size))
    # Every global rank must create every group in the same order.
    control_group = dist.new_group(all_ranks, backend="gloo")
    prefill_group = dist.new_group(prefill_ranks, backend="nccl")
    decode_group = dist.new_group(decode_ranks, backend="nccl")
    transfer_groups = tuple(
        dist.new_group([rank, rank + tensor_parallel_size], backend="nccl")
        for rank in range(tensor_parallel_size)
    )
    if global_rank < tensor_parallel_size:
        role: DisaggregatedRole = "prefill"
        role_rank = global_rank
        role_group = prefill_group
    else:
        role = "decode"
        role_rank = global_rank - tensor_parallel_size
        role_group = decode_group
    transfer_group = transfer_groups[role_rank]
    if not isinstance(role_group, dist.ProcessGroup):
        raise RuntimeError("current rank did not join its disaggregated role group")
    if not isinstance(control_group, dist.ProcessGroup):
        raise RuntimeError("current rank did not join the disaggregated control group")
    if not isinstance(transfer_group, dist.ProcessGroup):
        raise RuntimeError("current rank did not join its KV transfer group")
    if dist.get_rank(group=role_group) != role_rank:
        raise RuntimeError("role process-group rank mapping is inconsistent")

    return DisaggregatedTopology(
        global_rank=global_rank,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        role=role,
        role_rank=role_rank,
        device=device,
        control_group=control_group,
        prefill_group=prefill_group,
        decode_group=decode_group,
        role_group=role_group,
        transfer_group=transfer_group,
    )


@dataclass
class _LocalCacheState:
    batch_size: int
    max_seq_len: int
    cache_backend: str
    page_size: int
    phase: DisaggregatedRole = "prefill"
    seq_len: int = 0
    prefill_cache: object | None = None
    decode_cache: object | None = None
    transfer_buffer: Tensor | None = None


class DisaggregatedCacheHandle:
    """Coordinator-side handle for role-local KV caches on every rank."""

    def __init__(
        self,
        owner: "DisaggregatedPrefillDecodeModel",
        cache_id: int,
        batch_size: int,
        max_seq_len: int,
        cache_backend: str,
    ) -> None:
        self._owner = owner
        self.cache_id = cache_id
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.cache_backend = cache_backend
        self.phase: DisaggregatedRole = "prefill"
        self.seq_len = 0
        self.active_batch_size: int | None = None
        self.transfer_bytes = 0
        self.transfer_ms: float | None = None

    def reset(self) -> None:
        self._owner._reset_cache(self)

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len != 0:
            raise ValueError("disaggregated caches do not support external nonzero sequence restoration")
        self.reset()


class DisaggregatedPrefillDecodeModel:
    """Incremental model backed by independent TP prefill and decode replicas.

    Rank 0 owns the serving API. All other ranks execute a single ordered
    command stream. The first cache-populating forward runs on the prefill
    group, transfers live KV shards with NCCL, and returns its logits. Later
    forwards run only on the decode group and return full logits to rank 0.
    """

    provenance_variant = "llama3:tp-disaggregated-v1"
    is_disaggregated_prefill_decode = True

    def __init__(
        self,
        role_model: object,
        topology: DisaggregatedTopology,
        *,
        cache_backend: str = "dense",
        page_size: int = 16,
        profile_transfer: bool = False,
    ) -> None:
        if cache_backend not in {"dense", "flashinfer"}:
            raise ValueError("disaggregated prefill/decode supports dense or flashinfer KV cache")
        self.role_model = role_model
        self.topology = topology
        self.config = getattr(role_model, "config")
        self.device = topology.device
        self.devices = (topology.device,)
        self.dtype = getattr(role_model, "dtype")
        self.rank = topology.global_rank
        self.world_size = topology.world_size
        self.role = topology.role
        self.role_rank = topology.role_rank
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.profile_transfer = profile_transfer
        self.training = False
        self._cache_states: dict[int, _LocalCacheState] = {}
        self._next_cache_id = 1
        self._command_lock = threading.RLock()
        self._workers_stopped = False
        self._transfer_count = 0
        self._transfer_bytes = 0
        self._transfer_ms = 0.0
        self._prefill_forward_count = 0
        self._prefill_forward_ms = 0.0
        self._decode_forward_count = 0
        self._decode_forward_ms = 0.0
        self._validate_replicas()

    @property
    def is_coordinator(self) -> bool:
        return self.topology.coordinator

    def eval(self) -> "DisaggregatedPrefillDecodeModel":
        self.training = False
        eval_model = getattr(self.role_model, "eval", None)
        if callable(eval_model):
            eval_model()
        return self

    def train(self, mode: bool = True) -> "DisaggregatedPrefillDecodeModel":
        self.training = bool(mode)
        train_model = getattr(self.role_model, "train", None)
        if callable(train_model):
            train_model(mode)
        return self

    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        cache_backend: str | None = None,
        page_size: int | None = None,
        device: torch.device | None = None,
    ) -> DisaggregatedCacheHandle:
        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may allocate a disaggregated cache")
        if batch_size < 1 or max_seq_len < 1:
            raise ValueError("cache dimensions must be positive")
        if device is not None and torch.device(device) != self.device:
            raise ValueError("disaggregated coordinator cache must use the coordinator CUDA device")
        backend = self.cache_backend if cache_backend is None else str(cache_backend)
        configured_page_size = self.page_size if page_size is None else int(page_size)
        if backend not in {"dense", "flashinfer"}:
            raise ValueError("disaggregated prefill/decode supports dense or flashinfer KV cache")
        with self._command_lock:
            cache_id = self._next_cache_id
            self._next_cache_id += 1
            command = {
                "op": "allocate",
                "cache_id": cache_id,
                "batch_size": int(batch_size),
                "max_seq_len": int(max_seq_len),
                "cache_backend": backend,
                "page_size": configured_page_size,
            }
            self._broadcast_command(command)
            self._allocate_local_state(command)
        return DisaggregatedCacheHandle(
            self,
            cache_id,
            batch_size,
            max_seq_len,
            backend,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: DisaggregatedCacheHandle,
        use_cache: bool = True,
        return_last_logits_only: bool = False,
    ) -> tuple[Tensor, DisaggregatedCacheHandle]:
        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may submit a disaggregated forward")
        if not use_cache:
            raise ValueError("disaggregated forward requires use_cache=True")
        self._validate_handle(cache)
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        if input_ids.size(0) > cache.batch_size:
            raise ValueError("input batch exceeds cache capacity")
        if cache.active_batch_size is not None and input_ids.size(0) != cache.active_batch_size:
            raise ValueError("decode batch size must match the batch used for prefill")
        if cache.seq_len + input_ids.size(1) > cache.max_seq_len:
            raise ValueError("input exceeds cache sequence capacity")
        with self._command_lock:
            phase = cache.phase
            started = time.perf_counter() if self.profile_transfer else 0.0
            command = {
                "op": "forward",
                "cache_id": cache.cache_id,
                "phase": phase,
                "shape": tuple(int(size) for size in input_ids.shape),
                "return_last_logits_only": bool(return_last_logits_only),
            }
            self._broadcast_command(command)
            distributed_input = input_ids.to(self.device, dtype=torch.long, non_blocking=True).contiguous()
            self._distribute_input(command, distributed_input)
            logits, transfer_bytes, transfer_ms = self._run_local_forward(
                command,
                distributed_input,
            )
            if self.profile_transfer:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if phase == "prefill":
                    self._prefill_forward_count += 1
                    self._prefill_forward_ms += elapsed_ms
                else:
                    self._decode_forward_count += 1
                    self._decode_forward_ms += elapsed_ms
        cache.seq_len += int(input_ids.size(1))
        cache.phase = "decode"
        if cache.active_batch_size is None:
            cache.active_batch_size = int(input_ids.size(0))
        if transfer_bytes:
            cache.transfer_bytes = transfer_bytes
            cache.transfer_ms = transfer_ms
        if logits is None:
            raise RuntimeError("coordinator forward did not produce logits")
        return logits, cache

    __call__ = forward

    @torch.inference_mode()
    def try_decode_one_token_graph(
        self,
        input_ids: Tensor,
        cache: DisaggregatedCacheHandle,
        *,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> Tensor | None:
        """Replay the decode replica's sampled-token graph when available."""

        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may submit a disaggregated decode")
        self._validate_handle(cache)
        if cache.phase != "decode":
            return None
        if cache.cache_backend != "dense":
            return None
        if temperature > 0.0:
            return None
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            return None
        if cache.active_batch_size is None or input_ids.size(0) != cache.active_batch_size:
            raise ValueError("decode batch size must match the batch used for prefill")
        if cache.seq_len + 1 > cache.max_seq_len:
            raise ValueError("input exceeds cache sequence capacity")
        with self._command_lock:
            started = time.perf_counter() if self.profile_transfer else 0.0
            command = {
                "op": "decode_token_graph",
                "cache_id": cache.cache_id,
                "shape": tuple(int(size) for size in input_ids.shape),
                "temperature": float(temperature),
                "capture_on_miss": bool(capture_on_miss),
            }
            self._broadcast_command(command)
            distributed_input = input_ids.to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            ).contiguous()
            self._distribute_input(command, distributed_input)
            token = self._run_local_decode_token_graph(command, distributed_input)
            if token is not None and self.profile_transfer:
                self._decode_forward_count += 1
                self._decode_forward_ms += (time.perf_counter() - started) * 1000.0
        if token is None:
            return None
        cache.seq_len += 1
        return token

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids.to(self.device, non_blocking=True)
        cache = self.allocate_cache(input_ids.size(0), input_ids.size(1) + max_new_tokens)
        try:
            logits, cache = self.forward(
                input_ids,
                cache=cache,
                use_cache=True,
                return_last_logits_only=True,
            )
            next_token = sample_next_token(logits[:, -1, :], temperature).to(self.device)
            output = [input_ids.to(self.device, non_blocking=True), next_token[:, None]]
            for _ in range(1, max_new_tokens):
                if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                    break
                logits, cache = self.forward(
                    next_token[:, None],
                    cache=cache,
                    use_cache=True,
                    return_last_logits_only=True,
                )
                next_token = sample_next_token(logits[:, -1, :], temperature).to(self.device)
                output.append(next_token[:, None])
            return torch.cat(output, dim=1)
        finally:
            self.release_decode_graphs_for_cache(cache)

    def run_worker_loop(self) -> None:
        if self.is_coordinator:
            raise RuntimeError("the coordinator does not run the disaggregated worker loop")
        while True:
            command = self._receive_command()
            if self._run_worker_command(command) == "stop":
                self._workers_stopped = True
                return

    def startup_warmup(self, *, prompt_tokens: int = 32, new_tokens: int = 2) -> None:
        if prompt_tokens < 1 or new_tokens < 2:
            raise ValueError("disaggregated startup warmup requires prompt_tokens >= 1 and new_tokens >= 2")
        if self.is_coordinator:
            try:
                vocab_size = int(getattr(self.config, "vocab_size"))
                input_ids = (
                    torch.arange(prompt_tokens, device=self.device, dtype=torch.long) % vocab_size
                )[None, :]
                self.generate(input_ids, max_new_tokens=new_tokens, temperature=0.0)
            finally:
                self._broadcast_command({"op": "warmup_done"})
            self._reset_profile_stats()
            return
        while True:
            command = self._receive_command()
            op = self._run_worker_command(command)
            if op == "warmup_done":
                return
            if op == "stop":
                raise RuntimeError("disaggregated workers stopped during startup warmup")

    def shutdown_workers(self) -> None:
        if not self.is_coordinator or self._workers_stopped:
            return
        with self._command_lock:
            self._broadcast_command({"op": "stop"})
            self._workers_stopped = True

    def release_decode_graphs_for_cache(self, cache: object) -> None:
        if not isinstance(cache, DisaggregatedCacheHandle) or cache._owner is not self:
            return
        if cache.cache_id not in self._cache_states:
            return
        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may release a disaggregated cache")
        with self._command_lock:
            self._broadcast_command({"op": "release", "cache_id": cache.cache_id})
            self._release_local_state(cache.cache_id)

    def disaggregation_stats(self) -> dict[str, object]:
        average_ms = self._transfer_ms / self._transfer_count if self._transfer_count else None
        average_prefill_ms = (
            self._prefill_forward_ms / self._prefill_forward_count
            if self._prefill_forward_count
            else None
        )
        average_decode_ms = (
            self._decode_forward_ms / self._decode_forward_count
            if self._decode_forward_count
            else None
        )
        return {
            "mode": "prefill-decode",
            "transport": "nccl-p2p",
            "tensor_parallel_size_per_role": self.topology.tensor_parallel_size,
            "world_size": self.topology.world_size,
            "transfer_count": self._transfer_count,
            "transfer_bytes": self._transfer_bytes,
            "profiled_transfer_ms": self._transfer_ms if self.profile_transfer else None,
            "profiled_average_transfer_ms": average_ms if self.profile_transfer else None,
            "profiled_average_prefill_forward_ms": average_prefill_ms if self.profile_transfer else None,
            "profiled_average_decode_forward_ms": average_decode_ms if self.profile_transfer else None,
        }

    def reset_disaggregation_stats(self) -> None:
        self._reset_profile_stats()

    def _validate_handle(self, cache: DisaggregatedCacheHandle) -> None:
        if not isinstance(cache, DisaggregatedCacheHandle) or cache._owner is not self:
            raise ValueError("cache was not allocated by this disaggregated model")
        if cache.cache_id not in self._cache_states:
            raise ValueError("unknown disaggregated cache")

    def _run_worker_command(self, command: dict[str, Any]) -> str:
        op = str(command.get("op"))
        if op in {"stop", "warmup_done"}:
            return op
        if op == "allocate":
            self._allocate_local_state(command)
            return op
        if op == "reset":
            self._reset_local_state(int(command["cache_id"]))
            return op
        if op == "release":
            self._release_local_state(int(command["cache_id"]))
            return op
        if op not in {"forward", "decode_token_graph"}:
            raise RuntimeError(f"unknown disaggregated command: {op}")
        shape = tuple(int(size) for size in command["shape"])
        input_ids = torch.empty(shape, device=self.device, dtype=torch.long)
        self._distribute_input(command, input_ids)
        if op == "forward":
            self._run_local_forward(command, input_ids)
        else:
            self._run_local_decode_token_graph(command, input_ids)
        return op

    def _distribute_input(self, command: dict[str, Any], input_ids: Tensor) -> None:
        phase = str(command.get("phase", "decode"))
        if phase == "prefill":
            if self.role == "prefill":
                dist.broadcast(input_ids, src=0, group=self.topology.prefill_group)
            return
        if phase != "decode":
            raise RuntimeError(f"unknown disaggregated input phase: {phase}")

        if self.is_coordinator:
            _send_tensor(input_ids, self.topology.decode_root, self.topology.transfer_group)
        elif self.topology.global_rank == self.topology.decode_root:
            _receive_tensor(input_ids, 0, self.topology.transfer_group)
        if self.role == "decode":
            dist.broadcast(
                input_ids,
                src=self.topology.decode_root,
                group=self.topology.decode_group,
            )

    def _reset_profile_stats(self) -> None:
        self._transfer_count = 0
        self._transfer_bytes = 0
        self._transfer_ms = 0.0
        self._prefill_forward_count = 0
        self._prefill_forward_ms = 0.0
        self._decode_forward_count = 0
        self._decode_forward_ms = 0.0

    def _reset_cache(self, cache: DisaggregatedCacheHandle) -> None:
        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may reset a disaggregated cache")
        self._validate_handle(cache)
        with self._command_lock:
            self._broadcast_command({"op": "reset", "cache_id": cache.cache_id})
            self._reset_local_state(cache.cache_id)
        cache.phase = "prefill"
        cache.seq_len = 0
        cache.active_batch_size = None
        cache.transfer_bytes = 0
        cache.transfer_ms = None

    def _allocate_local_state(self, command: dict[str, Any]) -> None:
        cache_id = int(command["cache_id"])
        self._cache_states[cache_id] = _LocalCacheState(
            batch_size=int(command["batch_size"]),
            max_seq_len=int(command["max_seq_len"]),
            cache_backend=str(command["cache_backend"]),
            page_size=int(command["page_size"]),
        )

    def _reset_local_state(self, cache_id: int) -> None:
        state = self._cache_states.get(cache_id)
        if state is None:
            raise ValueError("unknown disaggregated cache")
        state.phase = "prefill"
        state.seq_len = 0
        if state.prefill_cache is not None:
            state.prefill_cache.reset()
        if state.decode_cache is not None:
            state.decode_cache.reset()

    def _release_local_state(self, cache_id: int) -> None:
        state = self._cache_states.pop(cache_id, None)
        if state is None:
            return
        release_graphs = getattr(self.role_model, "release_decode_graphs_for_cache", None)
        if not callable(release_graphs):
            return
        role_cache = state.prefill_cache if self.role == "prefill" else state.decode_cache
        if role_cache is not None:
            release_graphs(role_cache)

    def _ensure_role_cache(self, state: _LocalCacheState, input_tokens: int) -> object:
        allocate_cache = getattr(self.role_model, "allocate_cache")
        if self.role == "prefill":
            layers = tuple(getattr(state.prefill_cache, "layers", ()) or ())
            capacity = int(getattr(layers[0], "max_seq_len", 0)) if layers else 0
            if state.prefill_cache is None or capacity != input_tokens:
                state.prefill_cache = allocate_cache(
                    state.batch_size,
                    input_tokens,
                    cache_backend=state.cache_backend,
                    page_size=state.page_size,
                    stacked_storage=state.cache_backend == "dense",
                )
            else:
                state.prefill_cache.reset()
            return state.prefill_cache
        if state.decode_cache is None:
            state.decode_cache = allocate_cache(
                state.batch_size,
                state.max_seq_len,
                cache_backend=state.cache_backend,
                page_size=state.page_size,
                stacked_storage=state.cache_backend == "dense",
            )
        return state.decode_cache

    def _run_local_forward(
        self,
        command: dict[str, Any],
        input_ids: Tensor,
    ) -> tuple[Tensor | None, int, float | None]:
        cache_id = int(command["cache_id"])
        state = self._cache_states.get(cache_id)
        if state is None:
            raise ValueError("unknown disaggregated cache")
        return_last = bool(command.get("return_last_logits_only", False))
        if state.phase == "prefill":
            logits: Tensor | None = None
            if self.role == "prefill":
                role_cache = self._ensure_role_cache(state, int(input_ids.size(1)))
                logits, role_cache = self.role_model.forward(
                    input_ids,
                    cache=role_cache,
                    use_cache=True,
                    return_last_logits_only=return_last,
                    return_sharded_logits=False,
                )
                state.prefill_cache = role_cache
            else:
                self._ensure_role_cache(state, int(input_ids.size(1)))
            transfer_bytes, transfer_ms = self._transfer_cache(
                state,
                batch_size=int(input_ids.size(0)),
                tokens=int(input_ids.size(1)),
            )
            state.phase = "decode"
            state.seq_len = int(input_ids.size(1))
            if self.is_coordinator:
                self._transfer_count += 1
                self._transfer_bytes += transfer_bytes
                if transfer_ms is not None:
                    self._transfer_ms += transfer_ms
            return logits if self.is_coordinator else None, transfer_bytes, transfer_ms

        logits = None
        if self.role == "decode":
            role_cache = self._ensure_role_cache(state, int(input_ids.size(1)))
            logits, role_cache = self.role_model.forward(
                input_ids,
                cache=role_cache,
                use_cache=True,
                return_last_logits_only=return_last,
                return_sharded_logits=False,
            )
            state.decode_cache = role_cache
        state.seq_len += int(input_ids.size(1))
        coordinator_logits = self._return_decode_logits(
            logits,
            batch_size=int(input_ids.size(0)),
            output_tokens=1 if return_last else int(input_ids.size(1)),
        )
        return coordinator_logits, 0, None

    def _run_local_decode_token_graph(
        self,
        command: dict[str, Any],
        input_ids: Tensor,
    ) -> Tensor | None:
        state = self._cache_states.get(int(command["cache_id"]))
        if state is None:
            raise ValueError("unknown disaggregated cache")
        if state.phase != "decode":
            raise ValueError("decode token graph requires a completed KV handoff")

        token: Tensor | None = None
        if self.role == "decode":
            if state.decode_cache is None:
                raise RuntimeError("decode cache is missing after handoff")
            run_graph = getattr(self.role_model, "try_decode_one_token_graph", None)
            if callable(run_graph):
                token = run_graph(
                    input_ids,
                    state.decode_cache,
                    temperature=float(command.get("temperature", 0.0)),
                    capture_on_miss=bool(command.get("capture_on_miss", True)),
                )
            available = torch.tensor(
                [token is not None],
                device=self.device,
                dtype=torch.int32,
            )
            dist.all_reduce(available, op=dist.ReduceOp.MIN, group=self.topology.role_group)
            graph_available = bool(available.item())
            if graph_available != (token is not None):
                raise RuntimeError("decode graph availability differs across tensor-parallel ranks")
            if graph_available:
                state.seq_len += 1
        return self._return_decode_token(token, batch_size=int(input_ids.size(0)))

    def _transfer_cache(
        self,
        state: _LocalCacheState,
        *,
        batch_size: int,
        tokens: int,
    ) -> tuple[int, float | None]:
        layer_count = int(getattr(self.config, "num_hidden_layers"))
        local_heads = int(getattr(self.config, "num_key_value_heads")) // self.topology.tensor_parallel_size
        head_dim = int(getattr(self.config, "head_dim"))
        elements = layer_count * 2 * batch_size * local_heads * tokens * head_dim
        transfer_bytes = elements * torch.empty((), dtype=self.dtype).element_size()
        if state.transfer_buffer is None or state.transfer_buffer.numel() != elements:
            state.transfer_buffer = torch.empty(elements, device=self.device, dtype=self.dtype)
        buffer = state.transfer_buffer

        if self.profile_transfer:
            dist.barrier(
                group=self.topology.transfer_group,
                device_ids=[self.device.index if self.device.index is not None else 0],
            )
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
        pack_ms = 0.0
        p2p_ms = 0.0
        unpack_ms = 0.0
        if self.role == "prefill":
            if state.prefill_cache is None:
                raise RuntimeError("prefill cache is missing before handoff")
            _pack_live_cache(state.prefill_cache, buffer, batch_size=batch_size, tokens=tokens)
            if self.profile_transfer:
                torch.cuda.synchronize(self.device)
                packed = time.perf_counter()
                pack_ms = (packed - started) * 1000.0
            _send_tensor(buffer, self.topology.peer_rank, self.topology.transfer_group)
            if self.profile_transfer:
                torch.cuda.synchronize(self.device)
                p2p_ms = (time.perf_counter() - packed) * 1000.0
        else:
            if state.decode_cache is None:
                raise RuntimeError("decode cache is missing before handoff")
            _receive_tensor(buffer, self.topology.peer_rank, self.topology.transfer_group)
            if self.profile_transfer:
                torch.cuda.synchronize(self.device)
                received = time.perf_counter()
                p2p_ms = (received - started) * 1000.0
            _unpack_live_cache(state.decode_cache, buffer, batch_size=batch_size, tokens=tokens)
            state.decode_cache.set_seq_len(tokens)
            if self.profile_transfer:
                torch.cuda.synchronize(self.device)
                unpack_ms = (time.perf_counter() - received) * 1000.0

        transfer_ms: float | None = None
        if self.profile_transfer:
            torch.cuda.synchronize(self.device)
            local_ms = (time.perf_counter() - started) * 1000.0
            elapsed = torch.tensor(
                [local_ms, pack_ms, p2p_ms, unpack_ms],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
            transfer_ms = float(elapsed[0].item())
            if self.is_coordinator:
                aggregate_bytes = transfer_bytes * self.topology.tensor_parallel_size
                bandwidth_gbps = aggregate_bytes / (transfer_ms * 1_000_000.0)
                print(
                    "[TorchInferno disaggregated] "
                    f"kv_handoff_bytes={aggregate_bytes} "
                    f"kv_handoff_ms={transfer_ms:.3f} "
                    f"pack_ms={float(elapsed[1].item()):.3f} "
                    f"p2p_ms={float(elapsed[2].item()):.3f} "
                    f"unpack_ms={float(elapsed[3].item()):.3f} "
                    f"aggregate_bandwidth_GBps={bandwidth_gbps:.3f}",
                    flush=True,
                )
        return transfer_bytes * self.topology.tensor_parallel_size, transfer_ms

    def _return_decode_logits(
        self,
        logits: Tensor | None,
        *,
        batch_size: int,
        output_tokens: int,
    ) -> Tensor | None:
        shape = (batch_size, output_tokens, int(getattr(self.config, "vocab_size")))
        if self.topology.global_rank == self.topology.decode_root:
            if logits is None or tuple(logits.shape) != shape:
                raise RuntimeError("decode root produced an unexpected logits shape")
            _send_tensor(logits.contiguous(), 0, self.topology.transfer_group)
            return None
        if self.is_coordinator:
            received = torch.empty(shape, device=self.device, dtype=self.dtype)
            _receive_tensor(received, self.topology.decode_root, self.topology.transfer_group)
            return received
        return None

    def _return_decode_token(self, token: Tensor | None, *, batch_size: int) -> Tensor | None:
        if self.topology.global_rank == self.topology.decode_root:
            status = torch.tensor([token is not None], device=self.device, dtype=torch.int32)
            _send_tensor(status, 0, self.topology.transfer_group)
            if token is not None:
                if tuple(token.shape) != (batch_size,):
                    raise RuntimeError("decode root produced an unexpected token shape")
                _send_tensor(token.to(dtype=torch.long).contiguous(), 0, self.topology.transfer_group)
            return None
        if self.is_coordinator:
            status = torch.empty((1,), device=self.device, dtype=torch.int32)
            _receive_tensor(status, self.topology.decode_root, self.topology.transfer_group)
            if not bool(status.item()):
                return None
            received = torch.empty((batch_size,), device=self.device, dtype=torch.long)
            _receive_tensor(received, self.topology.decode_root, self.topology.transfer_group)
            return received
        return None

    def _broadcast_command(self, command: dict[str, Any]) -> None:
        if not self.is_coordinator:
            raise RuntimeError("only global rank 0 may broadcast disaggregated commands")
        payload: list[object] = [command]
        dist.broadcast_object_list(
            payload,
            src=0,
            group=self.topology.control_group,
        )

    def _receive_command(self) -> dict[str, Any]:
        payload: list[object] = [None]
        dist.broadcast_object_list(
            payload,
            src=0,
            group=self.topology.control_group,
        )
        command = payload[0]
        if not isinstance(command, dict):
            raise RuntimeError("invalid disaggregated command payload")
        return command

    def _validate_replicas(self) -> None:
        model_world_size = int(getattr(self.role_model, "world_size", 1))
        model_rank = int(getattr(self.role_model, "rank", -1))
        if model_world_size != self.topology.tensor_parallel_size or model_rank != self.role_rank:
            raise RuntimeError("role model tensor-parallel metadata does not match the disaggregated topology")
        fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        signature = (
            tuple((name, int(getattr(self.config, name))) for name in fields),
            str(self.dtype),
            str(getattr(self.role_model, "checkpoint", "")),
        )
        signatures: list[object] = [None for _ in range(self.topology.world_size)]
        dist.all_gather_object(signatures, signature, group=self.topology.control_group)
        if any(item != signature for item in signatures):
            raise RuntimeError("prefill and decode replicas do not have identical model metadata")


def _pack_live_cache(cache: object, buffer: Tensor, *, batch_size: int, tokens: int) -> None:
    offset = 0
    stacked = _stacked_cache_storage(cache)
    if stacked is not None:
        for storage in stacked:
            if storage.device != buffer.device or storage.dtype != buffer.dtype:
                raise ValueError("KV cache device or dtype does not match the transfer buffer")
            if batch_size > storage.size(1) or tokens > storage.size(3):
                raise ValueError("live KV region exceeds cache storage")
            live = storage[:, :batch_size, :, :tokens, :]
            count = live.numel()
            if offset + count > buffer.numel():
                raise ValueError("KV cache shape does not match the transfer contract")
            buffer[offset : offset + count].view(live.shape).copy_(live)
            offset += count
    else:
        layers = tuple(getattr(cache, "layers", ()))
        for name in ("keys", "values"):
            for layer in layers:
                storage = getattr(layer, name)
                if storage.device != buffer.device or storage.dtype != buffer.dtype:
                    raise ValueError("KV cache device or dtype does not match the transfer buffer")
                if batch_size > storage.size(0) or tokens > storage.size(2):
                    raise ValueError("live KV region exceeds cache storage")
                live = storage[:batch_size, :, :tokens, :]
                count = live.numel()
                if offset + count > buffer.numel():
                    raise ValueError("KV cache shape does not match the transfer contract")
                buffer[offset : offset + count].view(live.shape).copy_(live)
                offset += count
    if offset != buffer.numel():
        raise ValueError("KV cache shape does not match the transfer contract")


def _unpack_live_cache(cache: object, buffer: Tensor, *, batch_size: int, tokens: int) -> None:
    offset = 0
    stacked = _stacked_cache_storage(cache)
    if stacked is not None:
        for storage in stacked:
            if storage.device != buffer.device or storage.dtype != buffer.dtype:
                raise ValueError("KV cache device or dtype does not match the transfer buffer")
            if batch_size > storage.size(1) or tokens > storage.size(3):
                raise ValueError("live KV region exceeds cache storage")
            live = storage[:, :batch_size, :, :tokens, :]
            count = live.numel()
            if offset + count > buffer.numel():
                raise ValueError("KV cache shape does not match the transfer contract")
            live.copy_(buffer[offset : offset + count].view(live.shape))
            offset += count
    else:
        layers = tuple(getattr(cache, "layers", ()))
        for name in ("keys", "values"):
            for layer in layers:
                storage = getattr(layer, name)
                if storage.device != buffer.device or storage.dtype != buffer.dtype:
                    raise ValueError("KV cache device or dtype does not match the transfer buffer")
                if batch_size > storage.size(0) or tokens > storage.size(2):
                    raise ValueError("live KV region exceeds cache storage")
                live = storage[:batch_size, :, :tokens, :]
                count = live.numel()
                if offset + count > buffer.numel():
                    raise ValueError("KV cache shape does not match the transfer contract")
                live.copy_(buffer[offset : offset + count].view(live.shape))
                offset += count
    if offset != buffer.numel():
        raise ValueError("KV cache shape does not match the transfer contract")


def _stacked_cache_storage(cache: object) -> tuple[Tensor, Tensor] | None:
    keys = getattr(cache, "_stacked_keys", None)
    values = getattr(cache, "_stacked_values", None)
    if isinstance(keys, Tensor) and isinstance(values, Tensor):
        return keys, values
    return None


def _send_tensor(tensor: Tensor, peer: int, group: dist.ProcessGroup) -> None:
    operations = [dist.P2POp(dist.isend, tensor, peer, group=group)]
    for work in dist.batch_isend_irecv(operations):
        work.wait()


def _receive_tensor(tensor: Tensor, peer: int, group: dist.ProcessGroup) -> None:
    operations = [dist.P2POp(dist.irecv, tensor, peer, group=group)]
    for work in dist.batch_isend_irecv(operations):
        work.wait()
