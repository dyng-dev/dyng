# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.dynamic_bfs and dyng.triangle_delta (the tutorial algorithms): compute, update, the
docstring examples, and random chains against brute force."""

from __future__ import annotations

import doctest
import itertools
import random
from collections import deque

import dyng
import pytest


def bfs_levels(n: int, edges: set[tuple[int, int]], source: int = 0) -> list[int]:
    out: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        out[u].append(v)
    levels = [-1] * n
    levels[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in out[u]:
            if levels[v] < 0:
                levels[v] = levels[u] + 1
                queue.append(v)
    return levels


def triangles(n: int, edges: set[tuple[int, int]]) -> int:
    adjacent = {(min(u, v), max(u, v)) for u, v in edges if u != v}
    return sum(
        1
        for a, b, c in itertools.combinations(range(n), 3)
        if (a, b) in adjacent and (a, c) in adjacent and (b, c) in adjacent
    )


@pytest.mark.parametrize("module", [dyng.dynamic_bfs, dyng.triangle_delta])
def test_docstring_examples(module: object) -> None:
    failures, tests = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tests > 0 and failures == 0


def test_dynamic_bfs(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0, 0, 1, 2, 3], [1, 2, 3, 3, 4], resources=res)
    bfs = dyng.dynamic_bfs.compute(g, 0)
    assert bfs.levels.to_numpy().tolist() == [0, 1, 1, 2, 3]
    assert bfs.options.source == 0 and bfs.graph_version == g.version
    st = dyng.dynamic_bfs.update(g, dyng.EdgeBatch(delete=([1], [3])), bfs)
    assert isinstance(st, dyng.dynamic_bfs.Stats)
    assert (st.invalidated, st.invalidation_rounds, st.affected) == (2, 3, 0)
    assert st.engine_used == "operators" and not st.fallback_used
    with pytest.raises(dyng.InvalidArgumentError):
        bfs.set_options(source=1)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.dynamic_bfs.compute(g, 9)
    from_options = dyng.dynamic_bfs.compute(g, options=dyng.dynamic_bfs.Options(source=1))
    assert from_options.options.source == 1
    assert from_options.levels.to_numpy().tolist()[:2] == [-1, 0]


def test_triangle_delta(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0, 0, 0, 1, 1], [1, 2, 3, 2, 3], directed=False, resources=res)
    tri = dyng.triangle_delta.compute(g)
    assert tri.count == 2 and tri.space == "host"
    st = dyng.triangle_delta.update(g, dyng.EdgeBatch(delete=([0, 0], [1, 2])), tri)
    assert (st.deletions, st.triangles_removed, tri.count) == (2, 2, 0)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.triangle_delta.compute(dyng.Graph.from_edges([0], [1]))  # directed


@pytest.mark.parametrize("seed", range(3))
def test_random_chains_match_brute_force(seed: int) -> None:
    rnd = random.Random(seed)
    n = 24
    directed = {(rnd.randrange(n), rnd.randrange(n)) for _ in range(60)}
    directed = {(u, v) for u, v in directed if u != v}
    undirected = {(min(u, v), max(u, v)) for u, v in directed}
    gd = dyng.Graph.from_edges(*zip(*sorted(directed), strict=True), num_vertices=n)
    gu = dyng.Graph.from_edges(
        *zip(*sorted(undirected), strict=True), num_vertices=n, directed=False
    )
    bfs = dyng.dynamic_bfs.compute(gd, 0)
    tri = dyng.triangle_delta.compute(gu)
    for _ in range(5):
        delete = rnd.sample(sorted(directed), 6)
        insert = {(rnd.randrange(n), rnd.randrange(n)) for _ in range(6)}
        insert = {(u, v) for u, v in insert if u != v and (u, v) not in directed}
        batch = dyng.EdgeBatch(
            insert=tuple(map(list, zip(*sorted(insert), strict=True))) if insert else None,
            delete=tuple(map(list, zip(*delete, strict=True))),
        )
        dyng.dynamic_bfs.update(gd, batch, bfs)
        directed = (directed - set(delete)) | insert
        assert bfs.levels.to_numpy().tolist() == bfs_levels(n, directed)
        dyng.triangle_delta.update(gu, batch, tri)
        undirected = (undirected - {(min(u, v), max(u, v)) for u, v in delete}) | {
            (min(u, v), max(u, v)) for u, v in insert
        }
        assert tri.count == triangles(n, undirected)
