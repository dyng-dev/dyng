# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python oracles: small, obviously correct reference implementations for tests.

They share no code with the library: :class:`GraphModel` models a graph under the default
properties (sorted rows, no parallel edges) and the default batch semantics (deletions first,
upsert, missing deletions ignored, self-loops kept, vertex growth), :func:`dijkstra` computes the
canonical shortest-path tree (lowest-id parent among the tight in-neighbours) and
:func:`simple_cycle_histogram` counts directed simple cycles by enumeration.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence

__all__ = ["GraphModel", "dijkstra", "simple_cycle_histogram", "INFINITE_DISTANCE"]

INFINITE_DISTANCE = (2**63 - 1) // 4
"""The library's "unreachable" distance (max(int64) // 4)."""


class GraphModel:
    """A directed simple graph with one integer weight per edge (or none) and a vertex count.

    Built like ``Graph.from_edges`` with the default properties: repeated (u, v) keep one edge
    with the last weight.
    """

    def __init__(
        self,
        num_vertices: int,
        src: Iterable[int],
        dst: Iterable[int],
        weights: Iterable[int] | None = None,
    ) -> None:
        self.num_vertices = int(num_vertices)
        self.edges: dict[tuple[int, int], int] = {}
        s, d = list(src), list(dst)
        w = list(weights) if weights is not None else [1] * len(s)
        for u, v, x in zip(s, d, w, strict=True):
            self.edges[(int(u), int(v))] = int(x)

    def apply(
        self,
        insert_src: Sequence[int] = (),
        insert_dst: Sequence[int] = (),
        insert_weights: Sequence[int] | None = None,
        delete_src: Sequence[int] = (),
        delete_dst: Sequence[int] = (),
    ) -> None:
        """Apply a batch under upsert_last_wins (deletions first)."""
        for u, v in zip(delete_src, delete_dst, strict=True):
            self.edges.pop((int(u), int(v)), None)
        w = list(insert_weights) if insert_weights is not None else [1] * len(insert_src)
        for u, v, x in zip(insert_src, insert_dst, w, strict=True):
            self.edges[(int(u), int(v))] = int(x)
            self.num_vertices = max(self.num_vertices, int(u) + 1, int(v) + 1)

    def in_edges(self) -> list[list[tuple[int, int]]]:
        """For every vertex, its (in-neighbour, weight) pairs."""
        out: list[list[tuple[int, int]]] = [[] for _ in range(self.num_vertices)]
        for (u, v), w in self.edges.items():
            out[v].append((u, w))
        return out


def dijkstra(model: GraphModel, source: int) -> tuple[list[int], list[int]]:
    """Distances and canonical parents (-1: none) from ``source``."""
    n = model.num_vertices
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (u, v), w in model.edges.items():
        adj[u].append((v, w))
    dist = [INFINITE_DISTANCE] * n
    dist[source] = 0
    heap = [(0, source)]
    done = [False] * n
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    parent = [-1] * n
    for v, ins in enumerate(model.in_edges()):
        if v == source or dist[v] == INFINITE_DISTANCE:
            continue
        tight = [u for u, w in ins if dist[u] != INFINITE_DISTANCE and dist[u] + w == dist[v]]
        parent[v] = min(tight)
    return dist, parent


def simple_cycle_histogram(model: GraphModel, max_length: int = -1) -> list[int]:
    """counts[len] of the directed simple cycles of length 2..bound, where bound is
    min(max_length, max(n, 2)), or max(n, 2) without a bound (the shape of the library's
    histogram)."""
    n = model.num_vertices
    bound = max(n, 2) if max_length < 0 else min(max_length, max(n, 2))
    counts = [0] * (bound + 1)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in model.edges:
        if u != v:
            adj[u].append(v)
    for root in range(n):
        # Every cycle is counted once, from its smallest vertex: paths through larger vertices.
        stack = [(root, [root])]
        while stack:
            u, path = stack.pop()
            for v in adj[u]:
                if v == root and len(path) >= 2 and len(path) <= bound:
                    counts[len(path)] += 1
                elif v > root and v not in path and len(path) < bound:
                    stack.append((v, [*path, v]))
    return counts
