# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests (Hypothesis): random graphs and batches against pure-Python oracles.

sssp is compared with Dijkstra (lowest-id ties) and cycle_count with a brute-force enumeration,
both on a model of the graph that applies the batch under the default semantics
(dyng.testing.oracles), and every update is compared with a fresh compute; the tutorial
algorithms dynamic_bfs and triangle_delta are compared with a queue BFS and a triangle
enumeration over a chain of batches. The profile is set in conftest.py
(DYNG_HYPOTHESIS_PROFILE: dyng, ci, dev, or full for the long run of property.yml).
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

import dyng  # noqa: E402
from conftest import host_backends  # noqa: E402
from dyng.testing import oracles  # noqa: E402
from dyng.testing import strategies as dst  # noqa: E402
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

BACKENDS = st.sampled_from(host_backends())


@st.composite
def weighted_case(draw: st.DrawFn) -> tuple[dst.RandomGraph, dst.RandomBatch, int]:
    g = draw(dst.graphs(max_vertices=12, max_edges=40))
    b = draw(dst.batches(g, max_insertions=10, max_deletions=10))
    source = draw(st.integers(0, g.num_vertices - 1))
    return g, b, source


@st.composite
def unweighted_case(draw: st.DrawFn) -> tuple[dst.RandomGraph, dst.RandomBatch, int]:
    g = draw(dst.graphs(max_vertices=7, max_edges=25, weighted=False))
    b = draw(dst.batches(g, max_insertions=8, max_deletions=8, weighted=False))
    k = draw(st.sampled_from([2, 3, 4, 5, -1]))
    return g, b, k


@given(weighted_case(), BACKENDS)
def test_sssp_compute_and_update_match_dijkstra(case, backend) -> None:  # type: ignore[no-untyped-def]
    rg, rb, source = case
    res = dyng.Resources(backend)
    g = rg.to_graph(resources=res)
    model = rg.model()
    tree = dyng.sssp.compute(g, source)
    dist, parent = oracles.dijkstra(model, source)
    assert tree.distances.tolist() == dist
    assert tree.parents.tolist() == parent
    dyng.sssp.update(g, rb.to_batch(), tree)
    rb.apply_to(model)
    dist, parent = oracles.dijkstra(model, source)
    assert g.num_vertices == model.num_vertices
    assert tree.distances.tolist() == dist
    assert tree.parents.tolist() == parent
    fresh = dyng.sssp.compute(g, source)
    assert tree.parents.tolist() == fresh.parents.tolist()


@given(unweighted_case(), BACKENDS)
def test_cycle_count_compute_and_update_match_brute_force(case, backend) -> None:  # type: ignore[no-untyped-def]
    rg, rb, k = case
    res = dyng.Resources(backend)
    g = rg.to_graph(resources=res)
    model = rg.model()
    hist = dyng.cycle_count.compute(g, max_length=k)
    assert hist.counts.tolist() == oracles.simple_cycle_histogram(model, k)
    st_ = dyng.cycle_count.update(g, rb.to_batch(), hist)
    rb.apply_to(model)
    expected = oracles.simple_cycle_histogram(model, k)
    assert hist.counts.tolist() == expected
    assert hist.total == sum(expected)
    assert st_.cycles_added - st_.cycles_removed == hist.total - sum(
        oracles.simple_cycle_histogram(rg.model(), k)
    )


@given(weighted_case(), st.sampled_from([2, 3, -1]))
def test_one_batch_updates_both_results(case, k) -> None:  # type: ignore[no-untyped-def]
    rg, rb, source = case
    g = rg.to_graph()
    model = rg.model()
    tree, hist = dyng.sssp.compute(g, source), dyng.cycle_count.compute(g, max_length=k)
    dyng.update(g, rb.to_batch(), tree, hist)
    rb.apply_to(model)
    assert tree.distances.tolist() == oracles.dijkstra(model, source)[0]
    assert hist.counts.tolist() == oracles.simple_cycle_histogram(model, k)


@given(
    st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50)), max_size=20),
    st.sampled_from(["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"]),
)
def test_dtype_dispatch_never_narrows(edges, dtype) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    s = np.array([e[0] for e in edges], dtype=dtype)
    d = np.array([e[1] for e in edges], dtype=dtype)
    g = dyng.Graph.from_edges(s, d, np.ones(len(edges), np.int32))
    width = np.dtype(dtype).itemsize
    wide = width > 4 or (np.dtype(dtype).kind == "u" and width == 4)
    assert g.vertex_dtype == (np.int64 if wide else np.int32)


def bfs_levels(model: oracles.GraphModel, source: int) -> list[int]:
    """Hop levels from `source` (-1: unreachable) by a queue BFS over the model."""
    out: list[list[int]] = [[] for _ in range(model.num_vertices)]
    for u, v in model.edges:
        out[u].append(v)
    levels = [-1] * model.num_vertices
    levels[source] = 0
    queue = [source]
    for u in queue:
        for v in out[u]:
            if levels[v] < 0:
                levels[v] = levels[u] + 1
                queue.append(v)
    return levels


@st.composite
def bfs_case(draw: st.DrawFn) -> tuple[dst.RandomGraph, list[dst.RandomBatch], int]:
    g = draw(dst.graphs(max_vertices=12, max_edges=40, weighted=False))
    chain = []
    model = g.model()
    for _ in range(draw(st.integers(1, 3))):
        grown = dst.RandomGraph(
            model.num_vertices,
            tuple(u for u, _ in model.edges),
            tuple(v for _, v in model.edges),
            None,
        )
        b = draw(dst.batches(grown, max_insertions=8, max_deletions=8, weighted=False))
        b.apply_to(model)
        chain.append(b)
    return g, chain, draw(st.integers(0, g.num_vertices - 1))


@given(bfs_case(), BACKENDS)
def test_dynamic_bfs_chain_matches_a_queue_bfs(case, backend) -> None:  # type: ignore[no-untyped-def]
    rg, chain, source = case
    g = rg.to_graph(resources=dyng.Resources(backend))
    model = rg.model()
    bfs = dyng.dynamic_bfs.compute(g, source)
    assert bfs.levels.to_numpy().tolist() == bfs_levels(model, source)
    for rb in chain:
        dyng.dynamic_bfs.update(g, rb.to_batch(), bfs)
        rb.apply_to(model)
        assert bfs.levels.to_numpy().tolist() == bfs_levels(model, source)
    assert (
        bfs.levels.to_numpy().tolist()
        == dyng.dynamic_bfs.compute(g, source).levels.to_numpy().tolist()
    )


def triangle_count(edges: set[tuple[int, int]]) -> int:
    """Triangles of an undirected simple graph given as pairs (a, b) with a < b."""
    adjacent: dict[int, set[int]] = {}
    for a, b in edges:
        adjacent.setdefault(a, set()).add(b)
        adjacent.setdefault(b, set()).add(a)
    return sum(len(adjacent[a] & adjacent[b]) for a, b in edges) // 3


@st.composite
def triangle_case(draw: st.DrawFn) -> tuple[int, list[tuple[int, int]], list[tuple[list, list]]]:
    n = draw(st.integers(3, 10))
    pair = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda p: p[0] != p[1])
    edges = draw(st.lists(pair, max_size=30))
    chain = [
        (draw(st.lists(pair, max_size=8)), draw(st.lists(pair, max_size=8)))
        for _ in range(draw(st.integers(1, 3)))
    ]
    return n, edges, chain


@given(triangle_case(), BACKENDS)
def test_triangle_delta_chain_matches_an_enumeration(case, backend) -> None:  # type: ignore[no-untyped-def]
    n, edges, chain = case
    res = dyng.Resources(backend)
    simple = {(min(u, v), max(u, v)) for u, v in edges}
    src, dst_ = ([a for a, _ in sorted(simple)], [b for _, b in sorted(simple)])
    g = dyng.Graph.from_edges(src, dst_, num_vertices=n, directed=False, resources=res)
    tri = dyng.triangle_delta.compute(g)
    assert tri.count == triangle_count(simple)
    for insert, delete in chain:
        batch = dyng.EdgeBatch(
            insert=([u for u, _ in insert], [v for _, v in insert]),
            delete=([u for u, _ in delete], [v for _, v in delete]),
        )
        dyng.triangle_delta.update(g, batch, tri)
        simple -= {(min(u, v), max(u, v)) for u, v in delete}  # deletions first
        simple |= {(min(u, v), max(u, v)) for u, v in insert}
        assert tri.count == triangle_count(simple)
    assert tri.count == dyng.triangle_delta.compute(g).count
