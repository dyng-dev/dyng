# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python oracles: small, obviously correct reference implementations for tests.

They share no code with the library: :class:`GraphModel` models a graph under the default
properties (sorted rows, no parallel edges) and the default batch semantics (deletions first,
upsert, missing deletions ignored, self-loops kept, vertex growth), :func:`dijkstra` computes the
canonical shortest-path tree (lowest-id parent among the tight in-neighbours),
:func:`simple_cycle_histogram` counts directed simple cycles by enumeration, and
:class:`HypergraphModel` models a hypergraph's live hyperedges by id and checks the ids a batch
gave (the rules of best-fit reuse are the store's; the model checks what any valid choice must
satisfy).
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence

__all__ = [
    "GraphModel",
    "HypergraphModel",
    "dijkstra",
    "simple_cycle_histogram",
    "INFINITE_DISTANCE",
]

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


class HypergraphModel:
    """The live hyperedges of a hypergraph by id (member sets, weights), batch by batch.

    Built like ``Hypergraph.from_lists``: hyperedge i gets id i, members sorted and deduplicated.
    :meth:`apply` takes a batch and the ids the hypergraph gave its insertions
    (``HyperApplySummary.inserted_ids``) and checks that they are a valid choice: distinct, not
    live when the insertions run (after the deletions under ``"erase_first"``, before them under
    ``"insert_first"``), and non-negative. It returns the counts a summary must report.
    """

    def __init__(
        self,
        num_vertices: int,
        hyperedges: Iterable[Iterable[int]],
        weights: Iterable[int] | None = None,
        reuse: str = "erase_first",
    ) -> None:
        self.num_vertices = int(num_vertices)
        self.reuse = reuse
        rows = [sorted(set(int(v) for v in h)) for h in hyperedges]
        w = list(weights) if weights is not None else None
        self.rows: dict[int, list[int]] = dict(enumerate(rows))
        self.weights: dict[int, int] | None = (
            None if w is None else {i: int(x) for i, x in enumerate(w)}
        )

    @property
    def live_ids(self) -> list[int]:
        """The live ids, increasing."""
        return sorted(self.rows)

    def rows_by_id(self, bound: int) -> list[list[int]]:
        """The member list of every id below ``bound`` (empty for a dead id)."""
        return [self.rows.get(i, []) for i in range(bound)]

    def member_sets(self) -> list[tuple[int, ...]]:
        """The live member sets, sorted (the state by content)."""
        return sorted(tuple(r) for r in self.rows.values())

    def apply(
        self,
        delete_hyperedges: Sequence[int],
        insert_hyperedges: Sequence[Sequence[int]],
        insert_weights: Sequence[int] | None,
        inserted_ids: Sequence[int],
    ) -> dict[str, int]:
        """Apply a batch with the ids the hypergraph gave; returns the summary's counts.

        Raises:
            AssertionError: the ids are not a valid choice.
        """
        before = set(self.rows)
        deleted: set[int] = set()
        ignored = 0
        for h in delete_hyperedges:
            if int(h) in before and int(h) not in deleted:
                deleted.add(int(h))
            else:
                ignored += 1
        rows = [sorted(set(int(v) for v in h)) for h in insert_hyperedges]
        dropped = sum(len(h) for h in insert_hyperedges) - sum(len(r) for r in rows)
        ids = [int(i) for i in inserted_ids]
        assert len(ids) == len(rows), f"{len(ids)} ids for {len(rows)} insertions"
        assert len(set(ids)) == len(ids), f"repeated inserted ids {ids}"
        live_then = before if self.reuse == "insert_first" else before - deleted
        for i in ids:
            assert i >= 0, f"negative id {i}"
            assert i not in live_then, f"id {i} was live when the insertions ran ({self.reuse})"
        for h in deleted:
            del self.rows[h]
            if self.weights is not None:
                self.weights.pop(h, None)
        for k, i in enumerate(ids):
            self.rows[i] = rows[k]
            if self.weights is not None and insert_weights is not None:
                self.weights[i] = int(insert_weights[k])
        return {
            "inserted_hyperedges": len(rows),
            "deleted_hyperedges": len(deleted),
            "ignored_deletions": ignored,
            "dropped_members": dropped,
            "skipped_ops": ignored + dropped,  # 0.3: no incidence edits to skip
            "num_hyperedges_after": len(self.rows),
        }

    def line_graph(self, bound: int) -> list[list[int]]:
        """The line graph by brute force: per id below ``bound``, the live ids sharing a vertex."""
        out: list[list[int]] = []
        for i in range(bound):
            a = set(self.rows.get(i, []))
            out.append(
                sorted(j for j, r in self.rows.items() if j != i and a and a.intersection(r))
            )
        return out
