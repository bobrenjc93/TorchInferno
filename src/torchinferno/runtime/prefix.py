from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Optional


@dataclass(frozen=True)
class PrefixMatch:
    route_id: Optional[Hashable]
    matched_tokens: tuple[int, ...]

    @property
    def depth(self) -> int:
        return len(self.matched_tokens)


@dataclass
class _RadixNode:
    children: dict[int, "_RadixNode"] = field(default_factory=dict)
    route_id: Optional[Hashable] = None


class RadixPrefixTree:
    """Token-prefix index for radix attention and prefix-aware routing."""

    def __init__(self) -> None:
        self._root = _RadixNode()

    def insert(self, tokens: Iterable[int], route_id: Hashable) -> None:
        node = self._root
        for token in tokens:
            node = node.children.setdefault(int(token), _RadixNode())
        node.route_id = route_id

    def longest_prefix(self, tokens: Iterable[int]) -> PrefixMatch:
        node = self._root
        best_route = node.route_id
        best_depth = 0
        consumed: list[int] = []
        for token in tokens:
            token = int(token)
            if token not in node.children:
                break
            node = node.children[token]
            consumed.append(token)
            if node.route_id is not None:
                best_route = node.route_id
                best_depth = len(consumed)
        return PrefixMatch(best_route, tuple(consumed[:best_depth]))


class PrefixAwareRouter:
    def __init__(self, *, default_route: Hashable = "cold") -> None:
        self.default_route = default_route
        self._tree = RadixPrefixTree()

    def add_prefix(self, tokens: Iterable[int], route_id: Hashable) -> None:
        self._tree.insert(tokens, route_id)

    def route(self, prompt: Iterable[int]) -> PrefixMatch:
        match = self._tree.longest_prefix(prompt)
        if match.route_id is not None:
            return match
        return PrefixMatch(self.default_route, ())
