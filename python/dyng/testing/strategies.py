# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis strategies for random graphs, hypergraphs and their batches (needs the
``hypothesis`` package). Every strategy shrinks: a failing case is reduced to a small graph or
hypergraph and short batches.

Example:
    >>> from hypothesis import given
    >>> from dyng.testing import strategies as dst
    >>> @given(dst.graphs())
    ... def test_builds(g):
    ...     assert g.to_graph().num_vertices == g.num_vertices

A hypergraph's batches name its current ids, which depend on the ids earlier batches got, so
:func:`hyperedge_batches` draws from the state of a :class:`~dyng.testing.oracles.HypergraphModel`
(use ``st.data()`` and draw one batch per step)::

    @given(dst.hypergraphs(), st.data())
    def test_ids(h, data):
        hg, model = h.to_hypergraph(), h.model()
        for _ in range(3):
            b = data.draw(dst.hyperedge_batches(model))
            s = hg.apply(b.to_batch())
            model.apply(b.delete_hyperedges, b.insert_hyperedges, b.insert_weights,
                        s.inserted_ids)
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from ..batch import EdgeBatch
from ..graph import Graph
from ..hypergraph import HyperedgeBatch, Hypergraph
from .oracles import GraphModel, HypergraphModel

__all__ = [
    "RandomGraph",
    "RandomBatch",
    "graphs",
    "batches",
    "RandomHypergraph",
    "RandomHyperedgeBatch",
    "hypergraphs",
    "hyperedge_batches",
]


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


@dataclass(frozen=True)
class RandomHypergraph:
    """A drawn hypergraph: member lists (with repeated members unless excluded) and weights."""

    num_vertices: int
    hyperedges: tuple[tuple[int, ...], ...]
    weights: tuple[int, ...] | None

    def to_hypergraph(self, **kwargs: object) -> Hypergraph:
        """The dynG hypergraph (``Hypergraph.from_lists`` with ``kwargs``)."""
        return Hypergraph.from_lists(
            [list(h) for h in self.hyperedges],
            None if self.weights is None else list(self.weights),
            num_vertices=self.num_vertices,
            **kwargs,  # type: ignore[arg-type]
        )

    def model(self, reuse: str = "erase_first") -> HypergraphModel:
        """The pure-Python model of the hypergraph (under the ``id_reuse`` order ``reuse``)."""
        return HypergraphModel(self.num_vertices, self.hyperedges, self.weights, reuse)


@dataclass(frozen=True)
class RandomHyperedgeBatch:
    """A drawn hyperedge batch: deletions (live, dead, repeated or out-of-range ids) and
    insertions (member lists, with repeated members unless excluded)."""

    delete_hyperedges: tuple[int, ...]
    insert_hyperedges: tuple[tuple[int, ...], ...]
    insert_weights: tuple[int, ...] | None

    def to_batch(self) -> HyperedgeBatch:
        """The dynG batch."""
        return HyperedgeBatch(
            delete_hyperedges=list(self.delete_hyperedges),
            insert_hyperedges=[list(h) for h in self.insert_hyperedges],
            insert_weights=None if self.insert_weights is None else list(self.insert_weights),
        )


def _member_lists(
    n: int, max_count: int, max_cardinality: int, repeats: bool
) -> st.SearchStrategy[list[tuple[int, ...]]]:
    member = st.integers(0, n - 1)
    if repeats:
        hyperedge = st.lists(member, min_size=1, max_size=max_cardinality).map(tuple)
    else:
        hyperedge = st.lists(member, min_size=1, max_size=max_cardinality, unique=True).map(
            lambda m: tuple(sorted(m))
        )
    return st.lists(hyperedge, max_size=max_count)


@st.composite
def hypergraphs(
    draw: st.DrawFn,
    *,
    min_vertices: int = 1,
    max_vertices: int = 12,
    max_hyperedges: int = 20,
    max_cardinality: int = 5,
    weighted: bool = False,
    max_weight: int = 20,
    repeated_members: bool = True,
) -> RandomHypergraph:
    """Random hypergraphs: up to ``max_hyperedges`` non-empty member lists over ``n`` vertices
    (parallel hyperedges allowed; repeated members unless ``repeated_members=False``), with
    integer weights in [1, max_weight] when ``weighted``."""
    n = draw(st.integers(min_vertices, max_vertices))
    edges = draw(_member_lists(n, max_hyperedges, max_cardinality, repeated_members))
    weights = (
        tuple(draw(st.lists(st.integers(1, max_weight), min_size=len(edges), max_size=len(edges))))
        if weighted
        else None
    )
    return RandomHypergraph(n, tuple(edges), weights)


@st.composite
def hyperedge_batches(
    draw: st.DrawFn,
    model: HypergraphModel,
    *,
    max_insertions: int = 6,
    max_deletions: int = 6,
    max_cardinality: int = 5,
    weighted: bool | None = None,
    max_weight: int = 20,
    repeated_members: bool = True,
    skipped_deletions: bool = True,
) -> RandomHyperedgeBatch:
    """Random batches for the current state of ``model``: deletions of live ids (and, with
    ``skipped_deletions``, of dead, repeated and out-of-range ids, which ``apply()`` skips and
    counts) and insertions of member lists over the model's vertices. ``weighted`` defaults to
    whether the model has weights."""
    n = model.num_vertices
    live = model.live_ids
    bound = max(live, default=-1) + 1
    choices: list[st.SearchStrategy[int]] = []
    if live:
        choices.append(st.sampled_from(live))
    if skipped_deletions:
        choices.append(st.integers(-2, bound + 2))
    deletions = draw(st.lists(st.one_of(choices), max_size=max_deletions)) if choices else []
    insertions = (
        draw(_member_lists(n, max_insertions, max_cardinality, repeated_members)) if n > 0 else []
    )
    if weighted is None:
        weighted = model.weights is not None
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
    return RandomHyperedgeBatch(tuple(deletions), tuple(insertions), weights)
