# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python examples of PLAN Section 5.5 (the 0.1 parts), on the small fixture files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dyng


def test_sssp_example(data: Path, res: dyng.Resources) -> None:
    g = dyng.io.read_matrix_market(
        data / "mosp_graph_io" / "mtx" / "m1_general.mtx",
        random_weights=(1, 100, 12345),
        resources=res,
    )
    tree = dyng.sssp.compute(g, source=0)
    dist = tree.distances.to_numpy()  # copy to host (int64, length n)
    assert dist.dtype == np.int64 and dist.size == g.num_vertices
    assert dyng.testing.check_sssp_tree(g, tree)  # oracle: Dijkstra with lowest-id ties
    b = dyng.EdgeBatch(
        insert=(np.array([0, 5]), np.array([7, 9]), np.array([3, 2])),
        delete=(np.array([1]), np.array([2])),
    )
    st = dyng.sssp.update(g, b, tree)
    assert st.engine_used in ("fused", "operators") and st.invalidated >= 0
    assert dyng.testing.check_sssp_tree(g, tree)


def test_cycle_count_example(data: Path, res: dyng.Resources) -> None:
    cg = dyng.io.read_edge_list(
        data / "cycle_enum" / "parser" / "tudataset_A.txt",
        weighted=False,
        properties="cycle_enum_compatible",
        resources=res,
    )
    hist = dyng.cycle_count.compute(cg, max_length=4)
    ins = np.array([[3, 8], [8, 3]])
    dele = np.array([[1, 2]])
    st = dyng.cycle_count.update(
        cg, dyng.EdgeBatch(insert=(ins[:, 0], ins[:, 1]), delete=(dele[:, 0], dele[:, 1])), hist
    )
    assert st.batch.inserted_edges >= 0 and hist.total == sum(hist.counts.tolist())
    assert list(hist.counts) == list(dyng.cycle_count.compute(cg, max_length=4).counts)


def test_several_results_example(data: Path) -> None:
    g = dyng.io.read_matrix_market(
        data / "mosp_graph_io" / "mtx" / "m1_general.mtx", random_weights=(1, 100, 12345)
    )
    b = dyng.EdgeBatch(insert=([0, 5], [7, 9], [3, 2]), delete=([1], [2]))
    # g has the default properties (sorted rows, no parallel edges), so cycle_count accepts it
    tree, hist = dyng.sssp.compute(g, source=0), dyng.cycle_count.compute(g, max_length=3)
    st_tree, st_hist = dyng.update(g, b, tree, hist)  # one apply, both results
    assert st_tree.batch == st_hist.batch
    assert hist.counts.tolist() == dyng.cycle_count.compute(g, max_length=3).counts.tolist()
    assert tree.distances.tolist() == dyng.sssp.compute(g, 0).distances.tolist()
    print(dyng.citation("sssp")[:10])
