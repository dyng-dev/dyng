# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis strategies for random graphs and batches (needs the ``hypothesis`` package).

Example:
    >>> from hypothesis import given
    >>> from dyng.testing import strategies as dst
    >>> @given(dst.graphs())
    ... def test_builds(g):
    ...     assert g.to_graph().num_vertices == g.num_vertices
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from ..batch import EdgeBatch
from ..graph import Graph
from .oracles import GraphModel

__all__ = ["RandomGraph", "RandomBatch", "graphs", "batches"]


@dataclass(frozen=True)
class RandomGraph:
    """A drawn edge list (with repeated pairs and self-loops unless excluded)."""

    num_vertices: int
    src: tuple[int, ...]
    dst: tuple[int, ...]
    weights: tuple[int, ...] | None

    def to_graph(self, **kwargs: object) -> Graph:
        """The dynG graph (``Graph.from_edges`` with ``kwargs``)."""
        return Graph.from_edges(
            list(self.src),
            list(self.dst),
            None if self.weights is None else list(self.weights),
            num_vertices=self.num_vertices,
            **kwargs,  # type: ignore[arg-type]
        )

    def model(self) -> GraphModel:
        """The pure-Python model of the graph."""
        return GraphModel(self.num_vertices, self.src, self.dst, self.weights)


@dataclass(frozen=True)
class RandomBatch:
    """A drawn batch: deletions (existing or missing edges) and insertions (new edges, existing
    ones, and edges naming new vertices when growth is allowed)."""

    insert_src: tuple[int, ...]
    insert_dst: tuple[int, ...]
    insert_weights: tuple[int, ...] | None
    delete_src: tuple[int, ...]
    delete_dst: tuple[int, ...]

    def to_batch(self) -> EdgeBatch:
        """The dynG batch."""
        insert: tuple[list[int], ...] = (list(self.insert_src), list(self.insert_dst))
        if self.insert_weights is not None:
            insert = (*insert, list(self.insert_weights))
        return EdgeBatch(insert=insert, delete=(list(self.delete_src), list(self.delete_dst)))

    def apply_to(self, model: GraphModel) -> None:
        """Apply the batch to a model (default semantics)."""
        model.apply(
            self.insert_src, self.insert_dst, self.insert_weights, self.delete_src, self.delete_dst
        )


@st.composite
def graphs(
    draw: st.DrawFn,
    *,
    min_vertices: int = 1,
    max_vertices: int = 10,
    max_edges: int = 30,
    weighted: bool = True,
    max_weight: int = 20,
    self_loops: bool = True,
) -> RandomGraph:
    """Random directed graphs with integer weights in [1, max_weight] (or none)."""
    n = draw(st.integers(min_vertices, max_vertices))
    pairs = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1))
    if not self_loops:
        pairs = pairs.filter(lambda p: p[0] != p[1])
    edges = draw(st.lists(pairs, max_size=max_edges))
    weights = (
        tuple(draw(st.lists(st.integers(1, max_weight), min_size=len(edges), max_size=len(edges))))
        if weighted
        else None
    )
    return RandomGraph(n, tuple(e[0] for e in edges), tuple(e[1] for e in edges), weights)


@st.composite
def batches(
    draw: st.DrawFn,
    graph: RandomGraph,
    *,
    max_insertions: int = 8,
    max_deletions: int = 8,
    weighted: bool = True,
    max_weight: int = 20,
    grow: bool = True,
    self_loops: bool = True,
) -> RandomBatch:
    """Random batches for ``graph``: deletions of existing or missing edges, insertions of new
    or existing edges and (with ``grow``) of edges naming up to two new vertices."""
    n = graph.num_vertices
    hi = n + 1 if grow else n - 1
    existing = sorted(set(zip(graph.src, graph.dst, strict=True)))
    pair = st.tuples(st.integers(0, max(n - 1, 0)), st.integers(0, max(n - 1, 0)))
    deletions = draw(
        st.lists(st.sampled_from(existing) | pair if existing else pair, max_size=max_deletions)
    )
    new_pair = st.tuples(st.integers(0, hi), st.integers(0, hi))
    if not self_loops:
        new_pair = new_pair.filter(lambda p: p[0] != p[1])
        deletions = [d for d in deletions if d[0] != d[1]]
    insertions = draw(st.lists(new_pair, max_size=max_insertions))
    weights = (
        tuple(
            draw(
                st.lists(
                    st.integers(1, max_weight), min_size=len(insertions), max_size=len(insertions)
                )
            )
        )
        if weighted
        else None
    )
    return RandomBatch(
        tuple(p[0] for p in insertions),
        tuple(p[1] for p in insertions),
        weights,
        tuple(p[0] for p in deletions),
        tuple(p[1] for p in deletions),
    )
