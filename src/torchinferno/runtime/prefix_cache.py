from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol

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


class SequenceCache(Protocol):
    @property
    def seq_len(self) -> int: ...

    def set_seq_len(self, seq_len: int) -> None: ...

    def reset(self) -> None: ...


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

    def remove(self, route_id: Hashable) -> bool:
        entry = self._entries.get(route_id)
        if entry is None:
            return False
        removed = self._router.remove_prefix(entry.tokens, route_id)
        if removed:
            self._entries.pop(route_id, None)
        return removed

    def lookup(self, tokens: Iterable[int]) -> tuple[PrefixMatch, PrefixCacheEntry | None]:
        token_tuple = tuple(int(token) for token in tokens)
        match = self._router.route(token_tuple)
        while match.route_id is not None:
            entry = self._entries.get(match.route_id)
            if entry is not None:
                return match, entry
            self._router.remove_prefix(match.matched_tokens, match.route_id)
            match = self._router.route(token_tuple)
        return match, None

    def lookup_filtered(
        self,
        tokens: Iterable[int],
        predicate: Callable[[PrefixCacheEntry], bool],
    ) -> tuple[PrefixMatch, PrefixCacheEntry | None]:
        token_tuple = tuple(int(token) for token in tokens)
        best_entry: PrefixCacheEntry | None = None
        best_depth = 0
        for entry in self._entries.values():
            depth = len(entry.tokens)
            if depth < best_depth or depth > len(token_tuple):
                continue
            if token_tuple[:depth] != entry.tokens:
                continue
            if not predicate(entry):
                continue
            best_entry = entry
            best_depth = depth
        if best_entry is None:
            return PrefixMatch(None, ()), None
        return PrefixMatch(best_entry.route_id, best_entry.tokens), best_entry


def cache_sequence_length(cache: object) -> int:
    seq_len = getattr(cache, "seq_len", None)
    if seq_len is not None:
        return int(seq_len)
    for layer in _cache_layers(cache):
        seq_len = getattr(layer, "seq_len", None)
        if seq_len is not None:
            return int(seq_len)
    return 0


def set_cache_sequence_length(
    cache: object,
    seq_len: int,
    *,
    on_error: Callable[[BaseException], None] | None = None,
) -> bool:
    setter = getattr(cache, "set_seq_len", None)
    if callable(setter):
        try:
            setter(seq_len)
            return True
        except Exception as exc:
            if on_error is not None:
                on_error(exc)
            return False

    changed = False
    if hasattr(cache, "seq_len"):
        try:
            setattr(cache, "seq_len", seq_len)
            changed = True
        except Exception as exc:
            if on_error is not None:
                on_error(exc)
    for layer in _cache_layers(cache):
        if not hasattr(layer, "seq_len"):
            continue
        try:
            setattr(layer, "seq_len", seq_len)
            changed = True
        except Exception as exc:
            seq_lens = getattr(layer, "seq_lens", None)
            if isinstance(seq_lens, list) and seq_lens:
                for row in range(len(seq_lens)):
                    seq_lens[row] = seq_len
                changed = True
            elif on_error is not None:
                on_error(exc)
    return changed


def reset_cache_sequence(
    cache: object,
    *,
    on_error: Callable[[BaseException], None] | None = None,
) -> bool:
    reset = getattr(cache, "reset", None)
    if callable(reset):
        try:
            reset()
            return True
        except Exception as exc:
            if on_error is not None:
                on_error(exc)
            return False
    return set_cache_sequence_length(cache, 0, on_error=on_error)


def restore_tensor_prefix_cache(
    entry: TensorPrefixCacheEntry,
    input_tokens: Iterable[int],
    cache: object,
    *,
    min_prefix_tokens: int,
    device: str,
    backend: str,
    page_size: int,
    row: int = 0,
    restore_seq_len: bool = True,
    on_seq_len_restore_error: Callable[[BaseException], None] | None = None,
) -> int:
    if entry.device != device or entry.backend != backend or entry.page_size != page_size:
        return 0
    cache_layers = _cache_layers(cache)
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
        if (
            layer_keys.size(0) <= row
            or layer_values.size(0) <= row
            or layer_keys.size(2) < max_prefix
            or layer_values.size(2) < max_prefix
        ):
            return 0
        layer_pairs.append((layer, layer_keys, layer_values, keys, values))

    for layer, layer_keys, layer_values, keys, values in layer_pairs:
        layer_keys[row : row + 1, :, :max_prefix, :].copy_(keys[:, :, :max_prefix, :])
        layer_values[row : row + 1, :, :max_prefix, :].copy_(values[:, :, :max_prefix, :])
    if restore_seq_len:
        set_cache_sequence_length(cache, max_prefix, on_error=on_seq_len_restore_error)
    return max_prefix


def snapshot_tensor_prefix_cache(
    cache: object,
    tokens: Iterable[int],
    *,
    seq_len: int,
    device: str,
    backend: str,
    page_size: int,
    row: int = 0,
) -> TensorPrefixCacheEntry | None:
    token_tuple = tuple(int(token) for token in tokens)
    if seq_len < len(token_tuple):
        return None
    cache_layers = _cache_layers(cache)
    if not cache_layers:
        return None
    layers: list[tuple[Tensor, Tensor]] = []
    for layer in cache_layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            return None
        if keys.size(0) <= row or values.size(0) <= row or keys.size(2) < seq_len or values.size(2) < seq_len:
            return None
        layers.append(
            (
                keys[row : row + 1, :, :seq_len, :].detach().clone(),
                values[row : row + 1, :, :seq_len, :].detach().clone(),
            )
        )
    return TensorPrefixCacheEntry(
        tokens=token_tuple[:seq_len],
        layers=tuple(layers),
        device=device,
        backend=backend,
        page_size=page_size,
    )


def _cache_layers(cache: object) -> tuple[object, ...]:
    return tuple(getattr(cache, "layers", ()) or ())
