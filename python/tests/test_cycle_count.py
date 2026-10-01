# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.cycle_count: compute, update, options, bounds."""

from __future__ import annotations

import dyng
import numpy as np
import pytest
from dyng.testing import oracles


def triangle_plus(res: dyng.Resources | None = None, **kw: object) -> dyng.Graph:
    return dyng.Graph.from_edges([0, 1, 2, 1], [1, 2, 0, 0], resources=res, **kw)


def test_compute(res: dyng.Resources) -> None:
    h = dyng.cycle_count.compute(triangle_plus(res), max_length=3)
    assert h.counts.tolist() == [0, 0, 1, 1]
    assert h.total == 2 and h.bound == 3
    assert h.count(2) == 1 and h.count(3) == 1 and h.count(9) == 0
    assert h.to_dict() == {2: 1, 3: 1}
    assert h.space == "host" and h.graph_version == 0
    assert "graph_version=0" in repr(h)


def test_unbounded(res: dyng.Resources) -> None:
    # the complete digraph on 4 vertices: 6 two-cycles, 8 three-cycles, 6 four-cycles
    src, dst = zip(*[(u, v) for u in range(4) for v in range(4) if u != v], strict=True)
    g = dyng.Graph.from_edges(list(src), list(dst), resources=res)
    h = dyng.cycle_count.compute(g)
    assert h.counts.tolist() == [0, 0, 6, 8, 6]
    assert h.options.max_length == -1


def test_update_matches_compute(res: dyng.Resources) -> None:
    g = triangle_plus(res, properties="cycle_enum_compatible")
    h = dyng.cycle_count.compute(g, max_length=4)
    b = dyng.EdgeBatch(insert=([2, 3], [3, 0]), delete=([2], [0]))
    st = dyng.cycle_count.update(g, b, h)
    assert isinstance(st, dyng.cycle_count.Stats)
    assert (st.deletions, st.insertions) == (1, 2)
    assert st.cycles_removed == 1 and st.cycles_added == 1
    assert h.counts.tolist() == dyng.cycle_count.compute(g, max_length=4).counts.tolist()
    assert h.to_dict() == {2: 1, 4: 1}


def test_weighted_graphs_are_accepted() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 0], [5, 7])
    assert dyng.cycle_count.compute(g).total == 1


def test_requirements() -> None:
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.cycle_count.compute(triangle_plus(properties="mosp_compatible"))
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.cycle_count.compute(triangle_plus(), max_length=1)
    s = np.array([0, 1], np.int64)
    g64 = dyng.Graph.from_edges(s, s[::-1].copy(), [1, 1])
    with pytest.raises(dyng.NotSupportedError, match="int32 vertex ids"):
        dyng.cycle_count.compute(g64)
    with pytest.raises(TypeError, match="unexpected keyword"):
        dyng.cycle_count.compute(triangle_plus(), source=0)
    with pytest.raises(dyng.InvalidArgumentError, match="scheduler"):
        dyng.cycle_count.compute(triangle_plus(), scheduler="lazy")


def test_options() -> None:
    h = dyng.cycle_count.compute(triangle_plus(), max_length=3, work_items="edges")
    assert h.options.work_items == "edges"
    h.set_options(scheduler="naive")
    assert h.options.scheduler == "naive"
    with pytest.raises(dyng.InvalidArgumentError):
        h.set_options(max_length=5)
    c = h.clone()
    assert c.counts.tolist() == h.counts.tolist()


def test_matches_the_brute_force_oracle() -> None:
    rng = np.random.default_rng(3)
    for _ in range(8):
        n = int(rng.integers(1, 9))
        m = int(rng.integers(0, 30))
        src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
        g = dyng.Graph.from_edges(src, dst, num_vertices=n, vertex_dtype="int32")
        model = oracles.GraphModel(n, src, dst)
        for k in (2, 3, 5, -1):
            assert dyng.cycle_count.compute(
                g, max_length=k
            ).counts.tolist() == oracles.simple_cycle_histogram(model, k)
            ref = dyng.testing.simple_cycles(g, k)  # the library's own oracle (C++)
            mine = oracles.simple_cycle_histogram(model, k)
            assert {i: v for i, v in enumerate(ref.tolist()) if v} == {
                i: v for i, v in enumerate(mine) if v
            }
