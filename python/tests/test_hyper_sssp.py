# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.hyper_sssp on the sequential backend: compute() and update() against the oracle
(dyng.testing.hyper_sssp_reference, MOSP_ESCHER's test oracle), update chains with hyperedge
batches and incidence edits equal to compute() (Hypothesis), the vertex-distance rule (HD11),
the options and the budget (HD8), the stats, the arrays, dyng.update() with several results,
the requirement errors and the backends (HD14: no OpenMP backend in 0.4)."""

from __future__ import annotations

import copy
import pickle

import dyng
import numpy as np
import pytest
from dyng.testing import strategies as dst
from hypothesis import given
from hypothesis import strategies as st

SEQ = dyng.Resources("sequential")
INF = dyng.hyper_sssp.INFINITE_DISTANCE

LISTS = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5]]
WEIGHTS = [2, 3, 1, 9, 4]


def slack(lists: list[list[int]], weights: list[int], **kw: object) -> dyng.Hypergraph:
    """A weighted slack_csr hypergraph on the sequential backend."""
    kw.setdefault("resources", SEQ)
    return dyng.Hypergraph.from_lists(
        lists,
        weights,
        storage="slack_csr",
        line_graph="slack_csr",
        **kw,  # type: ignore[arg-type]
    )


def check(hg: dyng.Hypergraph, sp: dyng.hyper_sssp.Result) -> None:
    """The result equals the oracle and compute() on the hypergraph, bit for bit."""
    d, p = dyng.testing.hyper_sssp_reference(hg, sp.source)
    assert sp.hyperedge_distances.tolist() == d.tolist()
    assert sp.hyperedge_parents.tolist() == p.tolist()
    assert dyng.testing.check_hyper_sssp_tree(hg, sp) == (0, 0)
    again = dyng.hyper_sssp.compute(hg, sp.source, options=sp.options, resources=SEQ)
    assert again.hyperedge_distances.tolist() == sp.hyperedge_distances.tolist()
    assert again.hyperedge_parents.tolist() == sp.hyperedge_parents.tolist()
    assert sp.graph_version == hg.version


def vertex_rule(hg: dyng.Hypergraph, sp: dyng.hyper_sssp.Result) -> list[int]:
    """HD11: min over the live hyperedges of v of D(h); 0 for the source."""
    d = sp.hyperedge_distances.tolist()
    rows = hg.v2h()
    out = []
    for v in range(hg.num_vertices):
        hs = rows.col_ind[rows.row_ptr[v] : rows.row_ptr[v + 1]].tolist()
        out.append(0 if v == sp.source else min((d[h] for h in hs), default=INF))
    return out


# -- compute ----------------------------------------------------------------------------------


def test_compute_example() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, source=0)
    assert sp.hyperedge_distances.tolist() == [2, 5, 6, 9, INF]
    assert sp.hyperedge_parents.tolist() == [-1, 0, 1, -1, -1]
    assert sp.source == 0 and sp.graph_version == 0 and sp.space == "host"
    assert sp.hyperedge_distances.dtype == np.int64
    assert sp.hyperedge_parents.dtype == np.int32
    # HD11: D(h) already holds w(h); nothing is added for the vertex, and the source is 0
    assert [sp.distance_to(v) for v in range(6)] == [0, 2, 5, 6, INF, INF]
    assert sp.vertex_distances().tolist() == vertex_rule(hg, sp)
    check(hg, sp)


def test_line_graph_automatic_builds_the_slack_line_graph() -> None:
    hg = dyng.Hypergraph.from_lists(LISTS, WEIGHTS, storage="slack_csr", resources=SEQ)
    assert hg.properties.line_graph == "automatic"
    check(hg, dyng.hyper_sssp.compute(hg, 3))


def test_arrays_are_read_only_and_stale_after_an_update() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, 0)
    d = sp.hyperedge_distances
    v = sp.vertex_distances()
    for a in (d, v):
        assert not a.to_numpy(copy=None).flags.writeable
        with pytest.raises(ValueError):
            a.to_numpy(copy=None)[0] = 7
    dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[1]), sp)
    with pytest.raises(dyng.StaleResultError):
        d.to_numpy()
    with pytest.raises(dyng.StaleResultError):
        v.to_numpy()
    assert sp.hyperedge_distances.tolist()[1] == INF


# -- update -----------------------------------------------------------------------------------


def test_update_with_hyperedges_and_incidence_edits() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, 0)
    st = dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(insert_incidences=([0], [3])), sp)
    assert sp.hyperedge_distances.tolist()[:4] == [2, 5, 3, 9]
    assert (st.line_graph_insertions, st.line_graph_deletions) == (1, 0)
    assert st.batch.inserted_incidences == 1 and st.batch.skipped_ops == 0
    assert st.engine_used == "operators" and st.converged and not st.fallback_used
    assert st.fallback_reason == dyng.hyper_sssp.FallbackReason.none == "none"
    check(hg, sp)
    # removals (one skipped: 3 is no longer in 1), a deletion and an insertion
    st = dyng.hyper_sssp.update(
        hg,
        dyng.HyperedgeBatch(
            delete_hyperedges=[3],
            insert_hyperedges=[[3, 4]],
            insert_weights=[1],
            delete_incidences=([0, 1], [1, 3]),
        ),
        sp,
    )
    assert st.batch.deleted_incidences == 1 and st.batch.skipped_ops == 1
    assert st.batch.inserted_ids.size == 1
    assert st.invalidated == st.affected
    assert sp.vertex_distances().tolist() == vertex_rule(hg, sp)
    check(hg, sp)


@pytest.mark.parametrize("reuse", ["erase_first", "insert_first"])
@given(h=dst.hypergraphs(weighted=True, min_vertices=2, max_hyperedges=14), data=st.data())
def test_update_chain_equals_compute(
    reuse: str, h: dst.RandomHypergraph, data: st.DataObject
) -> None:
    hg = h.to_hypergraph(storage="slack_csr", line_graph="slack_csr", id_reuse=reuse, resources=SEQ)
    model = h.model(reuse)
    source = data.draw(st.integers(0, h.num_vertices - 1))
    sp = dyng.hyper_sssp.compute(hg, source)
    other = dyng.hyper_sssp.compute(hg, (source + 1) % h.num_vertices, work_budget=0.0)
    for _ in range(3):
        b = data.draw(dst.hyperedge_batches(model, max_incidence_edits=4))
        st_sp, st_other = dyng.update(hg, b.to_batch(), sp, other)
        counts = model.apply(
            b.delete_hyperedges,
            b.insert_hyperedges,
            b.insert_weights,
            st_sp.batch.inserted_ids,
            delete_incidences=b.delete_incidences,
            insert_incidences=b.insert_incidences,
        )
        for name, value in counts.items():
            assert getattr(st_sp.batch, name) == value, name
        assert st_sp.batch.inserted_ids.tolist() == st_other.batch.inserted_ids.tolist()
        assert sorted(tuple(r) for r in hg.to_hyperedge_list().to_lists()) == model.member_sets()
        check(hg, sp)
        check(hg, other)
        assert st_other.fallback_used == (st_other.fallback_reason == "work_budget")
        assert sp.vertex_distances().tolist() == vertex_rule(hg, sp)


def test_dyng_update_two_sources_and_triad_count_refusal() -> None:
    hg = slack(LISTS, WEIGHTS)
    a = dyng.hyper_sssp.compute(hg, 0)
    b = dyng.hyper_sssp.compute(hg, 5)
    batch = dyng.HyperedgeBatch(insert_incidences=([4], [2]))
    sa, sb = dyng.update(hg, batch, a, b)
    assert isinstance(sa, dyng.hyper_sssp.Stats) and isinstance(sb, dyng.hyper_sssp.Stats)
    assert sa.batch.inserted_incidences == sb.batch.inserted_incidences == 1
    check(hg, a)
    check(hg, b)
    assert a.distance_to(5) == 2 + 3 + 4  # 0 -> {0,1} -> {1,2} -> {4,5,2}
    # triad_count refuses slack_csr storage (HD9): no result of it exists on this hypergraph
    with pytest.raises(dyng.NotSupportedError, match="slack_csr"):
        dyng.triad_count.compute(hg)
    esc = dyng.Hypergraph.from_lists(LISTS, WEIGHTS, resources=SEQ)
    tri = dyng.triad_count.compute(esc)
    with pytest.raises(dyng.NotSupportedError, match="slack_csr"):
        dyng.hyper_sssp.compute(esc, 0)
    with pytest.raises(dyng.StaleResultError):  # a result of another hypergraph: nothing changes
        dyng.update(esc, dyng.HyperedgeBatch(delete_hyperedges=[0]), tri, a)
    # A triad_count result with a slack_csr hypergraph: the storage requirement's error (checked
    # before the result's identity), alone and with a hyper_sssp result; nothing changes.
    with pytest.raises(dyng.NotSupportedError, match="slack_csr"):
        dyng.triad_count.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[1]), tri)
    with pytest.raises(dyng.NotSupportedError, match="slack_csr"):
        dyng.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[1]), tri, a)
    assert esc.version == 0 and hg.version == 1
    with pytest.raises(TypeError, match="hypergraph algorithm"):
        dyng.update(dyng.Graph.from_edges([0], [1], [1]), dyng.EdgeBatch(), a)  # type: ignore[call-overload]


# -- options, budget, stats -------------------------------------------------------------------


def test_options_and_set_options() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, 0, work_budget=2.5, on_budget="error")
    o = sp.options
    assert (o.work_budget, o.max_iterations, o.on_budget, o.cuda_engine, o.validate_inputs) == (
        2.5,
        4096,
        dyng.hyper_sssp.BudgetAction.error,
        "automatic",
        True,
    )
    sp.set_options(on_budget=dyng.hyper_sssp.BudgetAction.recompute, max_iterations=7)
    assert (sp.options.on_budget, sp.options.max_iterations) == ("recompute", 7)
    for bad in ({"work_budget": -1.0}, {"work_budget": float("nan")}, {"max_iterations": 0}):
        with pytest.raises(dyng.InvalidArgumentError):
            sp.set_options(**bad)  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.hyper_sssp.compute(hg, 0, on_budget="sometimes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="delta"):
        dyng.hyper_sssp.compute(hg, 0, delta=3)  # type: ignore[call-arg]
    assert sp.options.max_iterations == 7  # nothing changed by the failed calls


def test_the_work_budget_falls_back_or_raises() -> None:
    # a path 0-1-2-3-4-5 of unit hyperedges; a shortcut {0, 4} makes the update expand
    hg = slack([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], [1, 1, 1, 1, 1])
    shortcut = dyng.HyperedgeBatch(insert_hyperedges=[[0, 4]], insert_weights=[1])
    sp = dyng.hyper_sssp.compute(hg, 0, work_budget=0.0)
    st = dyng.hyper_sssp.update(hg, shortcut, sp)
    assert st.fallback_used and st.fallback_reason == "work_budget"
    check(hg, sp)
    sp.set_options(work_budget=float("inf"))
    st = dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[5]), sp)
    assert not st.fallback_used and st.fallback_reason == "none"
    st = dyng.hyper_sssp.update(hg, shortcut, sp)
    assert not st.fallback_used and st.relaxations > 0 and st.iterations > 0
    check(hg, sp)
    sp.set_options(work_budget=0.0, on_budget="error")
    with pytest.raises(dyng.ConvergenceError):
        dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[5]), sp)
    assert hg.version == 4  # the budget is counted after the commit (the basic guarantee)
    with pytest.raises(dyng.StaleResultError, match="recompute"):
        _ = sp.hyperedge_distances
    with pytest.raises(dyng.StaleResultError, match="recompute"):
        dyng.hyper_sssp.update(hg, shortcut, sp)
    check(hg, dyng.hyper_sssp.compute(hg, 0))


# -- requirements and errors ------------------------------------------------------------------


def test_requirements_and_errors() -> None:
    hg = slack(LISTS, WEIGHTS)
    with pytest.raises(dyng.NotSupportedError, match="weighted"):
        dyng.hyper_sssp.compute(
            dyng.Hypergraph.from_lists(LISTS, storage="slack_csr", resources=SEQ), 0
        )
    with pytest.raises(dyng.InvalidArgumentError, match="source"):
        dyng.hyper_sssp.compute(hg, 6)
    with pytest.raises(dyng.InvalidArgumentError, match="2\\^28"):
        dyng.hyper_sssp.compute(slack([[0, 1]], [2**28 + 1]), 0)
    big = dyng.hyper_sssp.compute(slack([[0, 1]], [2**28 + 1]), 0, validate_inputs=False)
    assert big.hyperedge_distances.tolist() == [2**28 + 1]
    sp = dyng.hyper_sssp.compute(hg, 0)
    with pytest.raises(dyng.InvalidArgumentError, match="2\\^28"):  # before the commit
        dyng.hyper_sssp.update(
            hg, dyng.HyperedgeBatch(insert_hyperedges=[[0]], insert_weights=[2**28 + 1]), sp
        )
    assert hg.version == 0
    with pytest.raises(dyng.InvalidArgumentError):
        sp.distance_to(6)
    with pytest.raises(dyng.InvalidArgumentError):
        sp.distance_to(-1)
    hg.apply(dyng.HyperedgeBatch(delete_hyperedges=[4]))  # the result is now stale
    with pytest.raises(dyng.StaleResultError):
        sp.distance_to(0)
    with pytest.raises(dyng.StaleResultError):
        sp.vertex_distances()
    with pytest.raises(dyng.StaleResultError):
        dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(), sp)
    with pytest.raises(TypeError, match="Hypergraph"):
        dyng.hyper_sssp.compute(dyng.Graph.from_edges([0], [1], [1]), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HyperedgeBatch"):
        dyng.hyper_sssp.update(hg, dyng.EdgeBatch(), sp)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use dyng.hyper_sssp.compute"):
        dyng.hyper_sssp.Result()


def test_a_gone_hypergraph_makes_the_vertex_accessors_stale() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, 0)
    d = sp.hyperedge_distances.tolist()
    del hg
    import gc

    gc.collect()
    assert sp.hyperedge_distances.tolist() == d
    with pytest.raises(dyng.StaleResultError, match="no longer exists"):
        sp.distance_to(1)


def test_openmp_resources_are_refused_with_the_fix() -> None:
    if not dyng.config()["backends"]["openmp"]:
        pytest.skip("no OpenMP backend in this build")
    omp = dyng.Resources.openmp(2)
    hg = slack(LISTS, WEIGHTS, resources=omp)
    with pytest.raises(dyng.NotSupportedError) as e:
        dyng.hyper_sssp.compute(hg, 0)
    text = str(e.value)
    assert "sequential" in text and "cuda" in text and 'dyng.Resources("sequential")' in text
    # the host store serves both host backends: the fix works on the same hypergraph
    sp = dyng.hyper_sssp.compute(hg, 0, resources=SEQ)
    with pytest.raises(dyng.NotSupportedError, match="sequential"):
        dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[0]), sp)
    assert hg.version == 0
    dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[0]), sp, resources=SEQ)
    check(hg, sp)


def test_clone_copy_pickle_repr_and_citation() -> None:
    hg = slack(LISTS, WEIGHTS)
    sp = dyng.hyper_sssp.compute(hg, 2)
    for c in (sp.clone(), copy.copy(sp), copy.deepcopy(sp), sp.clone(SEQ)):
        assert c.hyperedge_distances.tolist() == sp.hyperedge_distances.tolist()
        assert c.distance_to(0) == sp.distance_to(0)
    c = sp.clone()
    dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[2]), sp)
    check(hg, sp)
    with pytest.raises(dyng.StaleResultError):  # the copy still matches the old version
        c.distance_to(0)
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(sp)
    assert "dyng.hyper_sssp.Result(source=2" in repr(sp)
    # A clone moved with its hypergraph: the readers read the original until the copy's first
    # update on the moved hypergraph (an empty batch is enough) links it there.
    moved = hg.to_backend(SEQ)
    m = sp.clone(SEQ)
    assert m.distance_to(0) == sp.distance_to(0)
    want = sp.vertex_distances().tolist()
    del hg, sp, c
    import gc

    gc.collect()
    with pytest.raises(dyng.StaleResultError, match="update"):
        m.distance_to(0)
    dyng.hyper_sssp.update(moved, dyng.HyperedgeBatch(), m)
    assert m.vertex_distances().tolist() == want
    check(moved, m)
    assert "hsosp2026" in dyng.citation("hyper_sssp")
    info = {a.name: a for a in dyng.algorithms()}["hyper_sssp"]
    assert info.container == "hypergraph" and info.backends == ("sequential", "cuda")


def test_the_stats_fields() -> None:
    assert [f for f in dyng.hyper_sssp.Stats.__dataclass_fields__] == [
        "affected",
        "iterations",
        "frontier_visits",
        "fallback_used",
        "converged",
        "engine_used",
        "batch",
        "line_graph_insertions",
        "line_graph_deletions",
        "invalidated",
        "relaxations",
        "fallback_reason",
    ]
    assert [m.value for m in dyng.hyper_sssp.FallbackReason] == [
        "none",
        "work_budget",
        "iterations",
        "distance_width",
        "invalidation_depth",
    ]
    assert [m.value for m in dyng.hyper_sssp.BudgetAction] == ["recompute", "error"]
