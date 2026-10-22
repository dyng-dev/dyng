# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python examples of PLAN Section 5.5 (the 0.1 parts and, from 0.3, triad_count), on small
fixture files."""

from __future__ import annotations

from pathlib import Path

import dyng
import numpy as np


def test_sssp_example(data: Path, res: dyng.Resources, tmp_path: Path) -> None:
    g = dyng.io.read_matrix_market(
        data / "mosp_graph_io" / "mtx" / "m1_general.mtx",
        random_weights=(1, 100, 12345),
        resources=res,
    )
    batches = tmp_path / "m1.safe.dgt"  # the stand-in for roadNet-CA.safe50k.dgt
    (batches).write_text("%dgt 1\n%batch 0\n+e 0 3 7\n-e 0 1\n%batch 1\n+e 2 0 1\n")
    tree = dyng.sssp.compute(g, source=0)
    for batch in dyng.io.read_batches(batches, resources=res):
        st = dyng.sssp.update(g, batch, tree)
        print(st.invalidated, st.engine_used)
    assert g.version == 2 and dyng.testing.check_sssp_tree(g, tree)
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


def test_triad_count_example(tmp_path: Path) -> None:
    # PLAN 5.5's triad_count lines, adapted as ADR 0041 (section 1) and ADR 0046 (item 2) record:
    # HyperApplySummary.inserted_ids is already an int64 NumPy array, so the plan's
    # `.to_numpy()` is dropped. A 60-hyperedge stand-in for coauth-DBLP.hg (ids 1-based, as
    # ESCHER-GPU's files; the batch deletes ids 17 and 42 and inserts over vertices up to 41).
    lines = [f"{1 + i % 45} {1 + (3 * i) % 47} {1 + (7 * i) % 43}" for i in range(60)]
    lines += lines[:5]  # repeated hyperedges, which dedupe_hyperedges=True drops
    (tmp_path / "coauth-DBLP.hg").write_text("\n".join(lines) + "\n")
    hg = dyng.io.read_hypergraph(
        tmp_path / "coauth-DBLP.hg", dedupe_hyperedges=True, line_graph="sorted_merge"
    )
    assert hg.num_hyperedges == 60
    tri = dyng.triad_count.compute(hg)  # kind="h_motif": 30 bins
    hb = dyng.HyperedgeBatch(
        delete_hyperedges=[17, 42], insert_hyperedges=[[3, 9, 12], [9, 12, 40, 41]]
    )
    st = dyng.triad_count.update(hg, hb, tri)
    print(st.batch.inserted_ids, st.delta)
    print(dyng.citation("triad_count"))
    # the store's best-fit choice (read the ids there, never predict them)
    assert st.batch.inserted_ids.dtype == np.int64 and st.batch.inserted_ids.size == 2
    assert st.delta.size == 30 and tri.num_bins == 30
    assert tri.counts.tolist() == dyng.triad_count.compute(hg).counts.tolist()
    assert "escher2026" in dyng.citation("triad_count")
