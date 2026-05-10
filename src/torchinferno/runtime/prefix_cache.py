from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Hashable, Iterable

from torch import Tensor

from torchinferno.runtime.prefix import PrefixAwareRouter, PrefixMatch


@dataclass(frozen=True)
class PrefixCacheEntry:
    route_id: Hashable
    request_id: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class TensorPrefixCacheEntry:
    tokens: tuple[int, ...]
    layers: tuple[tuple[Tensor, Tensor], ...]
    device: str
    backend: str
    page_size: int


class PrefixCacheIndex:
    """Prefix-aware lookup table for routing requests to reusable KV entries."""

    def __init__(self) -> None:
        self._router = PrefixAwareRouter(default_route=None)
        self._entries: dict[Hashable, PrefixCacheEntry] = {}

    def add(self, request_id: str, tokens: Iterable[int], *, route_id: Hashable | None = None) -> PrefixCacheEntry:
        token_tuple = tuple(int(token) for token in tokens)
        actual_route = request_id if route_id is None else route_id
        entry = PrefixCacheEntry(actual_route, request_id, token_tuple)
        self._entries[actual_route] = entry
        self._router.add_prefix(token_tuple, actual_route)
        return entry

    def lookup(self, tokens: Iterable[int]) -> tuple[PrefixMatch, PrefixCacheEntry | None]:
        match = self._router.route(tokens)
        if match.route_id is None:
            return match, None
        return match, self._entries[match.route_id]


def restore_tensor_prefix_cache(
    entry: TensorPrefixCacheEntry,
    input_tokens: Iterable[int],
    cache: object,
    *,
    min_prefix_tokens: int,
    device: str,
    backend: str,
    page_size: int,
    on_seq_len_restore_error: Callable[[BaseException], None] | None = None,
) -> int:
    if entry.device != device or entry.backend != backend or entry.page_size != page_size:
        return 0
    cache_layers = tuple(getattr(cache, "layers", ()) or ())
    if not cache_layers or len(cache_layers) != len(entry.layers):
        return 0

    token_tuple = tuple(int(token) for token in input_tokens)
    if len(token_tuple) <= len(entry.tokens):
        return 0

    max_prefix = min(len(token_tuple) - 1, len(entry.tokens))
    while max_prefix >= min_prefix_tokens:
        if token_tuple[:max_prefix] == entry.tokens[:max_prefix]:
            break
        max_prefix -= 1
    if max_prefix < min_prefix_tokens:
        return 0

    layer_pairs: list[tuple[object, Tensor, Tensor, Tensor, Tensor]] = []
    for layer, (keys, values) in zip(cache_layers, entry.layers):
        layer_keys = getattr(layer, "keys", None)
        layer_values = getattr(layer, "values", None)
        if not isinstance(layer_keys, Tensor) or not isinstance(layer_values, Tensor):
            return 0
        if layer_keys.size(0) < 1 or layer_keys.size(2) < max_prefix:
            return 0
        layer_pairs.append((layer, layer_keys, layer_values, keys, values))

    for layer, layer_keys, layer_values, keys, values in layer_pairs:
        layer_keys[:1, :, :max_prefix, :].copy_(keys[:, :, :max_prefix, :])
        layer_values[:1, :, :max_prefix, :].copy_(values[:, :, :max_prefix, :])
        if hasattr(layer, "seq_len"):
            try:
                setattr(layer, "seq_len", max_prefix)
            except Exception as exc:
                seq_lens = getattr(layer, "seq_lens", None)
                if isinstance(seq_lens, list) and seq_lens:
                    seq_lens[0] = max_prefix
                elif on_seq_len_restore_error is not None:
                    on_seq_len_restore_error(exc)
    if hasattr(cache, "seq_len"):
        try:
            setattr(cache, "seq_len", max_prefix)
        except Exception as exc:
            if on_seq_len_restore_error is not None:
                on_seq_len_restore_error(exc)
    return max_prefix


def snapshot_tensor_prefix_cache(
    cache: object,
    tokens: Iterable[int],
    *,
    seq_len: int,
    device: str,
    backend: str,
    page_size: int,
) -> TensorPrefixCacheEntry | None:
    token_tuple = tuple(int(token) for token in tokens)
    if seq_len < len(token_tuple):
        return None
    cache_layers = tuple(getattr(cache, "layers", ()) or ())
    if not cache_layers:
        return None
    layers: list[tuple[Tensor, Tensor]] = []
    for layer in cache_layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            return None
        if keys.size(0) < 1 or keys.size(2) < seq_len:
            return None
        layers.append(
            (
                keys[:1, :, :seq_len, :].detach().clone(),
                values[:1, :, :seq_len, :].detach().clone(),
            )
        )
    return TensorPrefixCacheEntry(
        tokens=token_tuple[:seq_len],
        layers=tuple(layers),
        device=device,
        backend=backend,
        page_size=page_size,
    )
