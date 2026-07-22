from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchinferno.models.deepseek_v4.config import DeepSeekV4Config


@dataclass
class DeepSeekV4CompressorCache:
    ratio: int
    head_dim: int
    overlap: bool
    raw_kv: Tensor
    raw_score: Tensor
    compressed: Tensor
    kv_state: Tensor
    score_state: Tensor

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        max_seq_len: int,
        ratio: int,
        head_dim: int,
        *,
        overlap: bool,
        device: torch.device,
    ) -> "DeepSeekV4CompressorCache":
        factor = 2 if overlap else 1
        feature_dim = factor * head_dim
        state_tokens = factor * ratio
        return cls(
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            raw_kv=torch.empty(batch_size, max_seq_len, feature_dim, dtype=torch.float32, device=device),
            raw_score=torch.empty(batch_size, max_seq_len, feature_dim, dtype=torch.float32, device=device),
            compressed=torch.empty(
                batch_size,
                math.ceil(max_seq_len / ratio),
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            kv_state=torch.zeros(batch_size, state_tokens, feature_dim, dtype=torch.float32, device=device),
            score_state=torch.full(
                (batch_size, state_tokens, feature_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            ),
        )

    def clear_rows(self, rows: tuple[int, ...]) -> None:
        if not rows:
            return
        self.kv_state[list(rows)] = 0
        self.score_state[list(rows)] = float("-inf")

    def restore_rows(self, rows: tuple[int, ...], seq_len: int) -> None:
        self.clear_rows(rows)
        if seq_len <= 0:
            return
        ratio = self.ratio
        completed_start = max(0, (seq_len // ratio - 1) * ratio)
        completed_end = min(completed_start + ratio, seq_len)
        current_start = (seq_len // ratio) * ratio
        current_end = seq_len
        row_list = list(rows)
        if self.overlap:
            if completed_end - completed_start == ratio:
                self.kv_state[row_list, :ratio] = self.raw_kv[row_list, completed_start:completed_end]
                self.score_state[row_list, :ratio] = self.raw_score[row_list, completed_start:completed_end]
            if current_end > current_start:
                count = current_end - current_start
                self.kv_state[row_list, ratio : ratio + count] = self.raw_kv[row_list, current_start:current_end]
                self.score_state[row_list, ratio : ratio + count] = self.raw_score[row_list, current_start:current_end]
        else:
            block_start = current_start if current_end > current_start else completed_start
            block_end = current_end if current_end > current_start else completed_end
            count = block_end - block_start
            self.kv_state[row_list, :count] = self.raw_kv[row_list, block_start:block_end]
            self.score_state[row_list, :count] = self.raw_score[row_list, block_start:block_end]


class DeepSeekV4LayerCache:
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_idx: int,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.ratio = config.compress_ratios[layer_idx]
        self.window_size = config.sliding_window
        self.max_seq_len = max_seq_len
        self.history_kv = torch.empty(
            batch_size,
            max_seq_len,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        self.swa_kv = torch.empty(
            batch_size,
            config.sliding_window,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        self.seq_lens = [0 for _ in range(batch_size)]
        self.compressor: DeepSeekV4CompressorCache | None = None
        self.indexer_compressor: DeepSeekV4CompressorCache | None = None
        if self.ratio:
            self.compressor = DeepSeekV4CompressorCache.allocate(
                batch_size,
                max_seq_len,
                self.ratio,
                config.head_dim,
                overlap=self.ratio == 4,
                device=device,
            )
        if self.ratio == 4:
            self.indexer_compressor = DeepSeekV4CompressorCache.allocate(
                batch_size,
                max_seq_len,
                4,
                config.index_head_dim,
                overlap=True,
                device=device,
            )

    def seq_len_for_rows(self, rows: tuple[int, ...]) -> int:
        if not rows:
            return 0
        value = self.seq_lens[rows[0]]
        if any(self.seq_lens[row] != value for row in rows):
            raise ValueError("selected V4 cache rows must have the same sequence length")
        return value

    def set_rows_seq_len(self, rows: tuple[int, ...], seq_len: int) -> None:
        if seq_len < 0 or seq_len > self.max_seq_len:
            raise ValueError("V4 cache sequence length is out of range")
        for row in rows:
            self.seq_lens[row] = seq_len
            self.swa_kv[row].zero_()
            start = max(0, seq_len - self.window_size)
            if seq_len > start:
                positions = torch.arange(start, seq_len, device=self.swa_kv.device)
                self.swa_kv[row, positions % self.window_size] = self.history_kv[row, start:seq_len]
        if self.compressor is not None:
            self.compressor.restore_rows(rows, seq_len)
        if self.indexer_compressor is not None:
            self.indexer_compressor.restore_rows(rows, seq_len)

    def copy_prefix(
        self,
        source: "DeepSeekV4LayerCache",
        tokens: int,
        *,
        source_row: int,
        dest_row: int,
    ) -> None:
        if tokens < 0 or tokens > source.seq_lens[source_row] or tokens > self.max_seq_len:
            raise ValueError("invalid V4 cache prefix length")
        if tokens:
            self.history_kv[dest_row, :tokens].copy_(source.history_kv[source_row, :tokens])
        for dest, src in (
            (self.compressor, source.compressor),
            (self.indexer_compressor, source.indexer_compressor),
        ):
            if dest is None or src is None:
                if dest is not src:
                    raise ValueError("V4 cache layer kinds do not match")
                continue
            if tokens:
                dest.raw_kv[dest_row, :tokens].copy_(src.raw_kv[source_row, :tokens])
                dest.raw_score[dest_row, :tokens].copy_(src.raw_score[source_row, :tokens])
            complete = tokens // dest.ratio
            if complete:
                dest.compressed[dest_row, :complete].copy_(src.compressed[source_row, :complete])
        self.set_rows_seq_len((dest_row,), tokens)


class DeepSeekV4Cache:
    cache_backend = "v4-heterogeneous-dense"

    def __init__(
        self,
        layers: list[DeepSeekV4LayerCache],
        *,
        rows: tuple[int, ...] | None = None,
        parent: "DeepSeekV4Cache | None" = None,
    ) -> None:
        self.layers = layers
        self._rows = rows if rows is not None else tuple(range(len(layers[0].seq_lens) if layers else 0))
        self._parent_cache = parent

    @classmethod
    def allocate(
        cls,
        config: DeepSeekV4Config,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DeepSeekV4Cache":
        if max_seq_len > config.max_position_embeddings:
            raise ValueError("V4 cache exceeds max_position_embeddings")
        layers = [
            DeepSeekV4LayerCache(config, index, batch_size, max_seq_len, device=device, dtype=dtype)
            for index in range(config.num_hidden_layers)
        ]
        return cls(layers)

    @property
    def selected_rows(self) -> tuple[int, ...]:
        return self._rows

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len_for_rows(self._rows) if self.layers else 0

    def for_rows(self, rows: tuple[int, ...] | list[int]) -> "DeepSeekV4Cache":
        mapped = tuple(self._rows[row] for row in rows)
        return DeepSeekV4Cache(self.layers, rows=mapped, parent=self)

    def set_seq_len(self, seq_len: int) -> None:
        for layer in self.layers:
            layer.set_rows_seq_len(self._rows, seq_len)

    def reset(self) -> None:
        self.set_seq_len(0)

    def clear_row(self, row: int) -> None:
        physical = self._rows[row]
        for layer in self.layers:
            layer.set_rows_seq_len((physical,), 0)

    def copy_prefix_from(
        self,
        other: "DeepSeekV4Cache",
        tokens: int,
        *,
        source_row: int = 0,
        dest_row: int = 0,
    ) -> None:
        if len(self.layers) != len(other.layers):
            raise ValueError("V4 cache layer counts do not match")
        source_physical = other._rows[source_row]
        dest_physical = self._rows[dest_row]
        for dest_layer, source_layer in zip(self.layers, other.layers):
            dest_layer.copy_prefix(
                source_layer,
                tokens,
                source_row=source_physical,
                dest_row=dest_physical,
            )
