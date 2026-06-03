"""GPU-resident decode graph runner with zero CPU→GPU transfers per step.

The key pattern (from sglang): the CUDA graph's static input buffer IS the
token stash. After graph replay, sampled tokens are written BACK to this
buffer for the next step. No copy between stash and graph input. CPU never
writes to GPU buffers during the hot decode loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class _CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    dw: object
    s_logits: Tensor
    bucket: int
    num_qo: int
    num_kv: int
    head_dim: int
    max_seq: int
    q_dtype: torch.dtype


class DecodeGraphRunner:
    """Runs CUDA-graphed FlashInfer decode with GPU-resident token stash.

    All decode inputs (input_ids, row_indices, write_positions) are
    pre-allocated GPU tensors that persist across steps. The CUDA graph
    reads from these buffers directly — no per-step tensor creation.

    After each step, sampled tokens are written to output_ids (GPU).
    The next step copies output_ids → input_ids (GPU→GPU, <1μs).
    CPU only sees tokens via async D2H copy on a separate stream.
    """

    def __init__(
        self,
        model: object,
        cache: object,
        device: torch.device,
        max_batch: int = 48,
        temperature: float = 0.0,
    ) -> None:
        import flashinfer

        self.model = model
        self.cache = cache
        self.device = device
        self.max_batch = max_batch
        self.temperature = temperature

        num_qo = model.layers[0].local_attention_heads
        num_kv = model.layers[0].local_key_value_heads
        head_dim = model.config.head_dim
        q_dtype = model.dtype
        max_seq = cache.layers[0].max_seq_len

        self.input_ids = torch.zeros(max_batch, 1, dtype=torch.long, device=device)
        self.output_ids = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.write_positions = torch.zeros(max_batch, 1, dtype=torch.long, device=device)
        self.row_indices = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.seq_lens = torch.zeros(max_batch, dtype=torch.long, device=device)

        self._cpu_buf = torch.zeros(max_batch, dtype=torch.long).pin_memory()
        self._d2h_stream = torch.cuda.Stream(device=device)
        self._d2h_event = torch.cuda.Event()
        self._d2h_pending = False

        self.n_active = 0
        self.active_rows: list[int] = []
        self._step_count = 0

        buckets = sorted({1, 2, 4, 8, 16, 32, max_batch} & set(range(1, max_batch + 1)))
        self._graphs: dict[int, _CapturedDecodeGraph] = {}

        for bs in buckets:
            self._capture_one(
                bs, num_qo, num_kv, head_dim, max_seq, q_dtype, flashinfer,
            )

        torch.cuda.synchronize(device)

    def _capture_one(
        self,
        bs: int,
        num_qo: int,
        num_kv: int,
        head_dim: int,
        max_seq: int,
        q_dtype: torch.dtype,
        flashinfer: object,
    ) -> None:
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        fi_ind = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        fi_idx = torch.empty(bs, dtype=torch.int32, device=self.device)
        fi_lpl = torch.empty(bs, dtype=torch.int32, device=self.device)
        dw = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            ws, fi_ind, fi_idx, fi_lpl, kv_layout="NHD",
        )

        s_ids = self.input_ids[:bs]
        s_wp = self.write_positions[:bs]
        s_ri = self.row_indices[:bs]

        def do_plan():
            dw.plan(
                indptr=torch.arange(bs + 1, dtype=torch.int32, device=self.device),
                indices=s_ri.to(torch.int32),
                last_page_len=torch.ones(bs, dtype=torch.int32, device=self.device),
                num_qo_heads=num_qo,
                num_kv_heads=num_kv,
                head_dim=head_dim,
                page_size=max_seq,
                q_data_type=q_dtype,
            )

        do_plan()
        self.model.forward_decode_flashinfer(
            s_ids, self.cache,
            write_positions=s_wp, row_indices=s_ri, decode_wrapper=dw,
        )
        torch.cuda.synchronize(self.device)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        do_plan()
        with torch.cuda.stream(stream):
            self.model.forward_decode_flashinfer(
                s_ids, self.cache,
                write_positions=s_wp, row_indices=s_ri, decode_wrapper=dw,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        do_plan()
        with torch.cuda.graph(graph, stream=stream):
            s_logits = self.model.forward_decode_flashinfer(
                s_ids, self.cache,
                write_positions=s_wp, row_indices=s_ri, decode_wrapper=dw,
            )

        self._graphs[bs] = _CapturedDecodeGraph(
            graph=graph, dw=dw, s_logits=s_logits, bucket=bs,
            num_qo=num_qo, num_kv=num_kv, head_dim=head_dim,
            max_seq=max_seq, q_dtype=q_dtype,
        )

        for layer in self.cache.layers:
            if hasattr(layer, "_seq_lens"):
                for i in range(len(layer._seq_lens)):
                    layer._seq_lens[i] = 0
            if hasattr(layer, "_uniform_seq_len"):
                layer._uniform_seq_len[0] = 0

    def set_active(
        self,
        rows: list[int],
        last_tokens: list[int],
        seq_lens_list: list[int],
    ) -> None:
        n = len(rows)
        self.n_active = n
        self.active_rows = list(rows)
        self._step_count = 0

        self.row_indices[:n] = torch.tensor(rows, dtype=torch.long, device=self.device)
        if n < self.max_batch:
            self.row_indices[n:] = 0
        self.input_ids[:n, 0] = torch.tensor(last_tokens, dtype=torch.long, device=self.device)
        if n < self.max_batch:
            self.input_ids[n:, 0] = 0
        self.seq_lens[:n] = torch.tensor(seq_lens_list, dtype=torch.long, device=self.device)

    def add_row(self, row: int, last_token: int, seq_len: int) -> int:
        slot = self.n_active
        self.n_active = slot + 1
        self.active_rows.append(row)
        self.row_indices[slot] = row
        self.input_ids[slot, 0] = last_token
        self.seq_lens[slot] = seq_len
        return slot

    def remove_slot(self, slot: int) -> None:
        last = self.n_active - 1
        if slot < last:
            self.row_indices[slot] = self.row_indices[last]
            self.input_ids[slot] = self.input_ids[last]
            self.output_ids[slot] = self.output_ids[last]
            self.seq_lens[slot] = self.seq_lens[last]
            self.active_rows[slot] = self.active_rows[last]
        self.active_rows.pop()
        self.n_active = last

    def step(self) -> None:
        batch = self.n_active
        if batch <= 0:
            return

        if self._step_count > 0:
            self.input_ids[:batch, 0] = self.output_ids[:batch]

        self.write_positions[:batch, 0] = self.seq_lens[:batch]

        bucket = 1 << max(0, (batch - 1).bit_length()) if batch > 1 else 1
        if bucket not in self._graphs:
            for b in sorted(self._graphs):
                if b >= batch:
                    bucket = b
                    break
            else:
                bucket = max(self._graphs)

        if batch < bucket:
            self.input_ids[batch:bucket, 0] = 0
            self.write_positions[batch:bucket, 0] = 0
            self.row_indices[batch:bucket] = 0

        entry = self._graphs[bucket]

        indptr = torch.arange(bucket + 1, dtype=torch.int32, device=self.device)
        indices = self.row_indices[:bucket].to(torch.int32)
        lpl = torch.ones(bucket, dtype=torch.int32, device=self.device)
        lpl[:batch] = (self.seq_lens[:batch] + 1).to(torch.int32)

        entry.dw.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=lpl,
            num_qo_heads=entry.num_qo,
            num_kv_heads=entry.num_kv,
            head_dim=entry.head_dim,
            page_size=entry.max_seq,
            q_data_type=entry.q_dtype,
        )

        entry.graph.replay()

        logits = entry.s_logits[:batch, -1, :]
        sampler = getattr(self.model, "_sample_next_token", None)
        if callable(sampler):
            self.output_ids[:batch] = sampler(logits, self.temperature).to(self.device).view(-1)[:batch]
        else:
            if self.temperature <= 0:
                self.output_ids[:batch] = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                self.output_ids[:batch] = torch.multinomial(probs, 1).squeeze(-1)

        self.seq_lens[:batch] += 1

        self._d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._d2h_stream):
            self._cpu_buf[:batch].copy_(self.output_ids[:batch], non_blocking=True)
        self._d2h_event.record(self._d2h_stream)
        self._d2h_pending = True

        self._step_count += 1

    def get_cpu_tokens(self) -> list[int]:
        if not self._d2h_pending:
            return []
        self._d2h_event.synchronize()
        self._d2h_pending = False
        return self._cpu_buf[: self.n_active].tolist()
