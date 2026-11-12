# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python code of the "Shortest hyperpaths" tutorial (docs/tutorials/shortest_hyperpaths.md)
beyond its example program: a batch built in code with hyperedge operations and incidence edits,
the vertex distances, a shortest hyperpath walked back through the parents, the work budget read
from Stats.relaxations, and the check against the oracle dyng.testing.hyper_sssp_reference. The
page quotes the marked ranges (ci/tutorial_quotes.py pins them); the expected values are the
page's, and the C++ twin is cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp."""

from __future__ import annotations

from pathlib import Path

import dyng
import pytest

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"

if not DATA.is_dir():
    pytest.skip("examples/data is not present", allow_module_level=True)


def test_the_tutorial() -> None:
    res = dyng.Resources("sequential")  # hyper_sssp: sequential or cuda (no openmp in 0.4)
    hg = dyng.io.read_hypergraph(
        DATA / "hyper_sssp.hg", weighted=True, storage="slack_csr", resources=res
    )
    sp = dyng.hyper_sssp.compute(hg, 0)
    # [tutorial: batch]
    batch = dyng.HyperedgeBatch(
        delete_hyperedges=[3],  # -h 3
        delete_incidences=([5], [7]),  # -i 5 7: vertex 7 leaves hyperedge 5
        insert_incidences=([1], [6]),  # +i 1 6: vertex 6 joins hyperedge 1
        insert_hyperedges=[[0, 11]],  # +h 0 11 ; 6
        insert_weights=[6],
    )
    st = dyng.hyper_sssp.update(hg, batch, sp)
    # [tutorial: batch end]
    assert (st.batch.deleted_incidences, st.batch.inserted_incidences) == (1, 1)
    assert st.batch.inserted_ids.tolist() == [3]
    # [tutorial: vertex distances]
    vd = sp.vertex_distances()  # a dyng.Array of host memory, one distance per vertex
    assert vd[0] == 0  # the source
    assert vd[11] == sp.distance_to(11) == 6  # the cheapest hyperedge holding vertex 11: id 3
    # [tutorial: vertex distances end]
    # [tutorial: path]
    d, p = sp.hyperedge_distances.to_numpy(), sp.hyperedge_parents.to_numpy()
    path = [6]  # the hyperedges from the source to hyperedge 6, reversed
    while p[path[-1]] != -1:
        path.append(int(p[path[-1]]))
    assert path == [6, 7, 3]  # 3 holds vertex 0, then 7, then 6
    assert d[6] == 6 + 2 + 5  # w(3) + w(7) + w(6): every hyperedge entered costs its weight
    # [tutorial: path end]
    # [tutorial: budget]
    sp.set_options(work_budget=0.0)  # recompute at the first push round or expansion
    shortcut = dyng.HyperedgeBatch(insert_hyperedges=[[0, 9]], insert_weights=[1])
    st = dyng.hyper_sssp.update(hg, shortcut, sp)
    assert st.fallback_used and st.fallback_reason == "work_budget"
    assert st.relaxations > 0  # the work the budget counted: tune work_budget from it
    # [tutorial: budget end]
    # [tutorial: oracle]
    d, p = dyng.testing.hyper_sssp_reference(hg, sp.source)  # MOSP_ESCHER's test oracle
    assert sp.hyperedge_distances.tolist() == d.tolist()
    assert sp.hyperedge_parents.tolist() == p.tolist()
    assert dyng.testing.check_hyper_sssp_tree(hg, sp) == (0, 0)
    # [tutorial: oracle end]
