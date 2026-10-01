# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.sssp: compute, update, options, results imported from arrays."""

from __future__ import annotations

import dyng
import numpy as np
import pytest
from dyng.testing import oracles

INF = dyng.sssp.INFINITE_DISTANCE


def small(res: dyng.Resources | None = None, **kw: object) -> dyng.Graph:
    return dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], [4, 1, 1, 5], resources=res, **kw)


def test_compute(res: dyng.Resources) -> None:
    tree = dyng.sssp.compute(small(res), source=0)
    assert tree.distances.tolist() == [0, 4, 1, 6]
    assert tree.parents.tolist() == [-1, 0, 0, 2]
    assert tree.source == 0 and tree.graph_version == 0 and tree.space == "host"
    assert tree.vertex_dtype == np.int32
    assert "source=0" in repr(tree)


def test_unreachable(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0], [1], [3], num_vertices=3, resources=res)
    tree = dyng.sssp.compute(g, 0)
    assert tree.distances.tolist() == [0, 3, INF]
    assert tree.parents.tolist() == [-1, 0, -1]


def test_update_matches_compute(res: dyng.Resources) -> None:
    g = small(res)
    tree = dyng.sssp.compute(g, 0)
    b = dyng.EdgeBatch(insert=([1, 3], [3, 4], [1, 2]), delete=([0], [2]))
    st = dyng.sssp.update(g, b, tree)
    assert isinstance(st, dyng.sssp.Stats)
    assert st.batch.inserted_edges == 2 and st.batch.deleted_edges == 1
    assert st.invalidated == 2 and isinstance(st.engine_used, str)
    fresh = dyng.sssp.compute(g, 0)
    assert tree.distances.tolist() == fresh.distances.tolist() == [0, 4, 5, 5, 7]
    assert tree.parents.tolist() == fresh.parents.tolist()
    assert tree.graph_version == g.version == 1
    assert dyng.testing.check_sssp_tree(g, tree)


def test_backends_agree_bitwise() -> None:
    rng = np.random.default_rng(7)
    n, m = 60, 400
    src, dst, w = rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(1, 20, m)
    ins = (rng.integers(0, n, 30), rng.integers(0, n, 30), rng.integers(1, 20, 30))
    batch = dyng.EdgeBatch(insert=ins, delete=(src[:40], dst[:40]))
    out = []
    for backend in ("sequential", "openmp"):
        if not dyng.config()["backends"][backend]:
            continue
        g = dyng.Graph.from_edges(
            src, dst, w, vertex_dtype="int32", resources=dyng.Resources(backend)
        )
        tree = dyng.sssp.compute(g, 3)
        dyng.sssp.update(g, batch, tree)
        out.append((tree.distances.to_numpy(), tree.parents.to_numpy()))
    for d, p in out[1:]:
        assert np.array_equal(d, out[0][0]) and np.array_equal(p, out[0][1])


def test_options() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [[1, 10], [1, 10]])
    tree = dyng.sssp.compute(g, 0, objective=1, delta=5)
    assert tree.distances.tolist() == [0, 10, 20]
    assert tree.options.objective == 1 and tree.options.delta == 5
    tree.set_options(delta=7)
    assert tree.options.delta == 7
    with pytest.raises(dyng.InvalidArgumentError):
        tree.set_options(objective=0)
    opts = dyng.sssp.Options(objective=1)
    assert dyng.sssp.compute(g, 0, options=opts).distances.tolist() == [0, 10, 20]
    with pytest.raises(TypeError, match="unexpected keyword"):
        dyng.sssp.compute(g, 0, tolerance=1e-3)
    with pytest.raises(dyng.InvalidArgumentError, match="cuda_engine"):
        dyng.sssp.compute(g, 0, cuda_engine="warp")
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.sssp.compute(g, 0, objective=2)


def test_invalid_inputs() -> None:
    g = small()
    with pytest.raises(dyng.InvalidArgumentError, match="out of range"):
        dyng.sssp.compute(g, 9)
    with pytest.raises(dyng.NotSupportedError, match="weights"):
        dyng.sssp.compute(dyng.Graph.from_edges([0], [1]), 0)
    zero = dyng.Graph.from_edges([0], [1], [0])
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.sssp.compute(zero, 0)
    no_in = dyng.Graph.from_edges([0], [1], [1], store_transposed=False)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.sssp.compute(no_in, 0)
    with pytest.raises(TypeError):
        dyng.sssp.update(g, dyng.EdgeBatch(), "tree")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dyng.sssp.Result()


def test_int64_graph() -> None:
    s = np.array([0, 0, 1, 2], np.int64)
    d = np.array([1, 2, 2, 3], np.int64)
    g = dyng.Graph.from_edges(s, d, [4, 1, 1, 5])
    tree = dyng.sssp.compute(g, 0)
    assert tree.parents.dtype == np.int64 and tree.distances.tolist() == [0, 4, 1, 6]
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([3], [0], [1])), tree)
    assert tree.graph_version == 1


def test_from_arrays(res: dyng.Resources) -> None:
    g = small(res)
    ref = dyng.sssp.compute(g, 0)
    imported = dyng.sssp.Result.from_arrays(g, 0, ref.distances, ref.parents)
    assert imported.distances.tolist() == ref.distances.tolist()
    b = dyng.EdgeBatch(delete=([0], [2]))
    dyng.sssp.update(g, b, imported)
    assert imported.distances.tolist() == dyng.sssp.compute(g, 0).distances.tolist()
    with pytest.raises(dyng.InvalidArgumentError):  # a malformed tree is rejected
        dyng.sssp.Result.from_arrays(g, 0, [3, 4, 1, 6], [-1, 0, 0, 2])  # source not at 0
    with pytest.raises(dyng.InvalidArgumentError):  # a parent cycle
        dyng.sssp.Result.from_arrays(g, 0, [0, 4, 1, 6], [-1, 3, 0, 1], canonicalize=False)
    with pytest.raises(dyng.InvalidArgumentError):  # arrays of the wrong length
        dyng.sssp.Result.from_arrays(g, 0, [0], [-1])


def test_clone(res: dyng.Resources) -> None:
    g = small(res)
    tree = dyng.sssp.compute(g, 0)
    copy = tree.clone()
    dyng.sssp.update(g, dyng.EdgeBatch(delete=([0], [2])), tree)
    assert copy.distances.tolist() == [0, 4, 1, 6]
    assert copy.graph_version == 0


def test_matches_the_pure_python_oracle() -> None:
    rng = np.random.default_rng(11)
    for _ in range(5):
        n = int(rng.integers(2, 30))
        m = int(rng.integers(0, 80))
        src, dst, w = rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(1, 9, m)
        g = dyng.Graph.from_edges(src, dst, w, num_vertices=n, vertex_dtype="int32")
        model = oracles.GraphModel(n, src, dst, w)
        dist, parent = oracles.dijkstra(model, 0)
        tree = dyng.sssp.compute(g, 0)
        assert tree.distances.tolist() == dist and tree.parents.tolist() == parent
