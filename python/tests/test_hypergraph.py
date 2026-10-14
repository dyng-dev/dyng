# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Hypergraph: construction, properties, batches, ids, views, copies and errors."""

from __future__ import annotations

import copy
import pickle
import threading
import time

import dyng
import numpy as np
import pytest
from conftest import host_backends  # noqa: E402
from dyng.testing.oracles import HypergraphModel

BACKENDS = host_backends()


@pytest.fixture(params=BACKENDS)
def hres(request: pytest.FixtureRequest) -> dyng.Resources:
    return dyng.Resources(request.param)


def rows(h: dyng.Hypergraph) -> list[list[int]]:
    v = h.h2v()
    return [v.row(i).tolist() for i in range(v.num_rows)]


# -- construction -------------------------------------------------------------------------------


def test_from_lists_and_from_hyperedges_agree(hres: dyng.Resources) -> None:
    a = dyng.Hypergraph.from_lists([[2, 0, 1, 1], [3, 1], [4]], resources=hres)
    b = dyng.Hypergraph.from_hyperedges(
        np.array([0, 4, 6, 7]),
        np.array([2, 0, 1, 1, 3, 1, 4], dtype=np.int32),
        resources=hres,
    )
    for h in (a, b):
        assert (h.num_vertices, h.num_hyperedges, h.hyperedge_id_bound, h.version) == (5, 3, 3, 0)
        assert rows(h) == [[0, 1, 2], [1, 3], [4]]
        assert not h.weighted and h.weights is None
        assert h.space == "host"
        assert h.vertex_dtype == np.dtype(np.int32)
        h.check_integrity()


def test_an_empty_hypergraph(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([], num_vertices=4, resources=hres)
    assert (h.num_vertices, h.num_hyperedges, h.hyperedge_id_bound) == (4, 0, 0)
    assert h.h2v().num_rows == 0
    assert h.v2h().num_rows == 4
    s = h.apply(dyng.HyperedgeBatch(insert_hyperedges=[[0, 3], [1]]))
    assert s.inserted_ids.tolist() == [0, 1]
    assert h.version == 1


def test_weighted_hypergraph(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2]], [5, 7], resources=hres)
    assert h.weighted
    assert h.weights.tolist() == [5, 7]  # type: ignore[union-attr]
    s = h.apply(dyng.HyperedgeBatch([0], [[0, 2]], [9]))
    w = h.weights
    assert w is not None and w[s.inserted_ids[0]] == 9
    with pytest.raises(dyng.InvalidArgumentError, match="weight"):
        h.apply(dyng.HyperedgeBatch(insert_hyperedges=[[0]]))  # no weight for a weighted graph
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.Hypergraph.from_lists([[0]], [0], resources=hres)  # weights are >= 1


def test_int64_members_are_never_narrowed_silently() -> None:
    offsets, members = np.array([0, 2]), np.array([0, 1], dtype=np.int64)
    with pytest.raises(dyng.NotSupportedError, match="vertex_dtype"):
        dyng.Hypergraph.from_hyperedges(offsets, members)
    h = dyng.Hypergraph.from_hyperedges(offsets, members, vertex_dtype="int32")
    assert h.num_vertices == 2
    with pytest.raises(dyng.InvalidArgumentError, match="does not fit"):
        dyng.Hypergraph.from_hyperedges(
            offsets, np.array([0, 2**40], dtype=np.int64), vertex_dtype="int32"
        )


def test_int64_ids_are_kept() -> None:
    h = dyng.Hypergraph.from_lists([[0, 1]])
    s = h.apply(dyng.HyperedgeBatch(delete_hyperedges=[2**40, -1]))
    assert s.ignored_deletions == 2 and s.skipped_ops == 2
    assert h.num_hyperedges == 1
    assert dyng.HyperedgeBatch(delete_hyperedges=[2**40]).delete_hyperedges.tolist() == [2**40]
    assert s.inserted_ids.dtype == np.int64


# -- properties and enums -----------------------------------------------------------------------


def test_properties_defaults_and_strings(hres: dyng.Resources) -> None:
    p = dyng.HypergraphProperties()
    assert (p.storage, p.line_graph, p.reuse, p.store_v2h) == (
        "escher",
        "automatic",
        "erase_first",
        True,
    )
    h = dyng.Hypergraph.from_lists(
        [[0, 1]],
        line_graph="sorted_merge",
        id_reuse="insert_first",
        capacity="mosp_escher",
        resources=hres,
    )
    got = h.properties
    assert (got.line_graph, got.reuse) == ("sorted_merge", "insert_first")
    assert got.capacity == dyng.CapacityPolicy.mosp_escher()
    with pytest.raises(dyng.InvalidArgumentError, match="id_reuse"):
        dyng.Hypergraph.from_lists([[0]], id_reuse="later")  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError, match="capacity"):
        dyng.Hypergraph.from_lists([[0]], capacity="huge")  # type: ignore[arg-type]


def test_capacity_presets_equal_the_cpp_presets() -> None:
    from dyng._backend import native

    assert dyng.CapacityPolicy.escher_gpu(4096) == dyng.CapacityPolicy._from_native(
        native.CapacityPolicy.escher_gpu(4096)
    )
    e = dyng.CapacityPolicy.escher_gpu()
    assert (e.initial_headroom, e.initial_slack, e.grow_factor, e.compact_before_grow) == (
        0.125,
        1024,
        1.5,
        False,
    )
    m = dyng.CapacityPolicy.mosp_escher(17)
    assert (m.initial_headroom, m.initial_slack, m.grow_factor, m.max_payload) == (
        0.3,
        17,
        0.0,
        2_000_000_000,
    )
    assert dyng.CapacityPolicy.preset("default") == dyng.CapacityPolicy()


@pytest.mark.parametrize("value", ["slack_csr"])
def test_slack_csr_is_planned_for_0_4(value: str) -> None:
    with pytest.raises(dyng.NotSupportedError, match="0.4"):
        dyng.Hypergraph.from_lists([[0]], storage=value)  # type: ignore[arg-type]
    with pytest.raises(dyng.NotSupportedError, match="0.4"):
        dyng.Hypergraph.from_lists([[0]], line_graph=value)  # type: ignore[arg-type]


# -- batches and ids ----------------------------------------------------------------------------


@pytest.mark.parametrize(("reuse", "want"), [("erase_first", 1), ("insert_first", 3)])
def test_the_id_reuse_example(hres: dyng.Resources, reuse: str, want: int) -> None:
    # ADR 0041, section 9 (and the concepts page): {0,1,2,3}, {4}, {5,6} under escher_gpu();
    # delete id 1 and insert {7,8}.
    h = dyng.Hypergraph.from_lists(
        [[0, 1, 2, 3], [4], [5, 6]],
        num_vertices=9,
        id_reuse=reuse,  # type: ignore[arg-type]
        capacity="escher_gpu",
        resources=hres,
    )
    s = h.apply(dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[7, 8]]))
    assert s.inserted_ids.tolist() == [want]
    assert s.deleted_hyperedges == 1


def test_skipped_ops_are_counted(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2], [2, 3]], resources=hres)
    h.apply(dyng.HyperedgeBatch(delete_hyperedges=[2]))
    s = h.apply(
        dyng.HyperedgeBatch(
            delete_hyperedges=[0, 0, 2, 7, -3],  # live, repeated, dead, out of range (twice)
            insert_hyperedges=[[3, 3, 1, 1, 0]],  # three repeated members
        )
    )
    assert (s.deleted_hyperedges, s.ignored_deletions, s.dropped_members) == (1, 4, 2)
    assert s.skipped_ops == 6
    assert s.num_hyperedges_after == h.num_hyperedges == 2
    assert s.id_bound_after == h.hyperedge_id_bound
    assert rows(h)[s.inserted_ids[0]] == [0, 1, 3]


def test_validation_errors_change_nothing(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2]], resources=hres)
    before = (h.version, rows(h), h.to_hyperedge_list().to_lists())
    for bad in (
        dyng.HyperedgeBatch(delete_hyperedges=[0], insert_hyperedges=[[0, 9]]),  # out of range
        dyng.HyperedgeBatch(insert_offsets=[0, 0], insert_members=[]),  # an empty hyperedge
    ):
        with pytest.raises(dyng.InvalidArgumentError):
            h.apply(bad)
        assert (h.version, rows(h), h.to_hyperedge_list().to_lists()) == before
    with pytest.raises(dyng.NotSupportedError, match="0.4"):
        h.apply(dyng.HyperedgeBatch(delete_incidences=([0], [1])))
    assert (h.version, rows(h)) == before[:2]


def test_a_batch_against_a_model(hres: dyng.Resources) -> None:
    lists = [[0, 1, 2], [2, 3], [4, 5, 6, 7], [1, 7], [3]]
    h = dyng.Hypergraph.from_lists(lists, line_graph="sorted_merge", resources=hres)
    model = HypergraphModel(8, lists)
    for deletions, insertions in (
        ([1, 3], [[0, 7], [5, 6, 6]]),
        ([0, 0, 9], [[2, 3, 4, 5, 6], [1]]),
        ([], []),
        (model.live_ids, [[0]]),  # delete all, insert one
    ):
        s = h.apply(dyng.HyperedgeBatch(deletions, insertions))
        counts = model.apply(deletions, insertions, None, s.inserted_ids)
        for name, value in counts.items():
            assert getattr(s, name) == value, name
        assert rows(h) == model.rows_by_id(h.hyperedge_id_bound)
        lg = h.h2h()
        assert [lg.row(i).tolist() for i in range(lg.num_rows)] == model.line_graph(
            h.hyperedge_id_bound
        )
        oracle = dyng.testing.rebuild_line_graph(h)
        assert np.array_equal(lg.row_ptr, oracle.row_ptr)
        assert np.array_equal(lg.col_ind, oracle.col_ind)
        assert np.array_equal(lg.out, oracle.out)
        h.check_integrity()


def test_views(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2], [3]], resources=hres)
    v2h = h.v2h()
    assert [v2h.row(v).tolist() for v in range(4)] == [[0], [0, 1], [1], [2]]
    assert v2h.col_ind.dtype == np.int64 and v2h.version == 0
    lg = h.h2h()  # automatic: built on first use
    assert [lg.row(i).tolist() for i in range(3)] == [[1], [0], []]
    assert lg.out.tolist() == [lg.row_ptr[0] + 0, lg.row_ptr[1] + 1, lg.row_ptr[2]]
    none = dyng.Hypergraph.from_lists([[0]], line_graph="none", resources=hres)
    with pytest.raises(dyng.InvalidArgumentError):
        none.h2h()
    no_v2h = dyng.Hypergraph.from_lists([[0]], store_v2h=False, line_graph="none", resources=hres)
    with pytest.raises(dyng.InvalidArgumentError):
        no_v2h.v2h()


def test_clone_and_to_backend_keep_the_ids(hres: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(
        [[0, 1, 2, 3], [4], [5, 6], [1, 2]], num_vertices=9, resources=hres
    )
    h.apply(dyng.HyperedgeBatch(delete_hyperedges=[0, 2]))
    copies = [h.clone(), copy.deepcopy(h), h.to_backend(dyng.Resources("sequential"))]
    batch = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[7, 8], [0, 1, 2]])
    want = h.apply(batch).inserted_ids.tolist()
    for c in copies:
        assert c.version == 1
        assert c.apply(batch).inserted_ids.tolist() == want
        assert rows(c) == rows(h)


def test_backends_agree() -> None:
    rng = np.random.default_rng(3)
    results = []
    for backend in BACKENDS:
        h = dyng.generators.uniform_hypergraph(
            num_vertices=40,
            num_hyperedges=60,
            max_cardinality=6,
            seed=5,
            resources=dyng.Resources(backend),
        )
        ids = []
        for i in range(5):
            b = dyng.generators.random_hyperedge_batch(h, mix="mixed", size=8, seed=i)
            ids.append(h.apply(b).inserted_ids.tolist())
        results.append((ids, rows(h)))
    del rng
    assert all(r == results[0] for r in results)


def test_store_stats(hres: dyng.Resources) -> None:
    st = dyng.Hypergraph.from_lists([[0, 1], [1, 2]], resources=hres).store_stats()
    assert st.payload_capacity >= st.payload_used > 0
    assert st.host_bytes > 0 and st.device_bytes == 0
    assert isinstance(st.fill, dyng.StoragePathCounters)


def test_not_constructible_or_picklable() -> None:
    with pytest.raises(TypeError):
        dyng.Hypergraph()
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(dyng.Hypergraph.from_lists([[0]]))


def test_batch_arguments() -> None:
    b = dyng.HyperedgeBatch(insert_offsets=[0, 2, 3], insert_members=[1, 2, 0])
    assert b.insertions() == [[1, 2], [0]]
    assert not b.empty and dyng.HyperedgeBatch().empty
    with pytest.raises(dyng.InvalidArgumentError, match="not both"):
        dyng.HyperedgeBatch(insert_hyperedges=[[0]], insert_offsets=[0, 1], insert_members=[0])
    with pytest.raises(dyng.InvalidArgumentError, match="weights"):
        dyng.HyperedgeBatch(insert_hyperedges=[[0], [1]], insert_weights=[1])
    with pytest.raises(dyng.InvalidArgumentError, match="integers"):
        dyng.HyperedgeBatch(delete_hyperedges=[0.5])
    h = dyng.Hypergraph.from_lists([[0, 1]])
    with pytest.raises(dyng.InvalidArgumentError, match="unweighted"):
        h.apply(dyng.HyperedgeBatch(insert_hyperedges=[[0]], insert_weights=[3]))
    with pytest.raises(TypeError):
        h.apply(dyng.EdgeBatch(insert=([0], [1])))  # type: ignore[arg-type]


def test_gil_is_released_during_an_apply() -> None:
    h = dyng.generators.uniform_hypergraph(
        num_vertices=20_000,
        num_hyperedges=200_000,
        max_cardinality=8,
        seed=1,
        line_graph="sorted_merge",
        resources=dyng.Resources("sequential"),
    )
    batch = dyng.generators.random_hyperedge_batch(h, mix="mixed", size=50_000, seed=2)
    ticks = 0
    done = threading.Event()

    def work() -> None:
        h.apply(batch)
        done.set()

    worker = threading.Thread(target=work)
    started = time.perf_counter()
    worker.start()
    while not done.is_set():
        ticks += 1
        time.sleep(0.0005)
    worker.join()
    assert time.perf_counter() - started > 0.02, "the apply was too fast to observe the GIL"
    assert ticks > 5  # the main thread ran while the native apply was running
