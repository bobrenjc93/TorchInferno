from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

from torchinferno.runtime.prefix import PrefixAwareRouter, PrefixMatch


@dataclass(frozen=True)
class PrefixCacheEntry:
    route_id: Hashable
    request_id: str
    tokens: tuple[int, ...]


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
