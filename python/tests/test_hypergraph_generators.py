# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The hypergraph generators (library-owned and legacy) and the Hypothesis hypergraph strategies."""

from __future__ import annotations

import typing

import dyng
import numpy as np
import pytest
from conftest import host_backends  # noqa: E402
from dyng.generators.hypergraphs import HyperedgeBatchMixName

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies
from dyng.testing import strategies as dst  # noqa: E402


def test_uniform_hypergraph_is_seeded() -> None:
    a = dyng.generators.uniform_hypergraph_arrays(
        num_vertices=50, num_hyperedges=40, min_cardinality=2, max_cardinality=5, seed=9
    )
    b = dyng.generators.uniform_hypergraph_arrays(
        num_vertices=50, num_hyperedges=40, min_cardinality=2, max_cardinality=5, seed=9
    )
    c = dyng.generators.uniform_hypergraph_arrays(
        num_vertices=50, num_hyperedges=40, min_cardinality=2, max_cardinality=5, seed=10
    )
    assert a.to_lists() == b.to_lists() != c.to_lists()
    assert a.num_vertices == 50 and a.num_hyperedges == 40 and a.weights is None
    for r in a.to_lists():
        assert 2 <= len(r) <= 5 and r == sorted(set(r))
    w = dyng.generators.uniform_hypergraph_arrays(
        num_vertices=5, num_hyperedges=30, weighted=True, min_weight=3, max_weight=4, seed=1
    )
    assert w.weights is not None and set(w.weights.tolist()) <= {3, 4}


@pytest.mark.parametrize("mix", list(typing.get_args(HyperedgeBatchMixName)))
def test_every_mix_applies(mix: str) -> None:
    h = dyng.generators.uniform_hypergraph(
        num_vertices=1200, num_hyperedges=30, max_cardinality=4, seed=3, line_graph="sorted_merge"
    )
    b = dyng.generators.random_hyperedge_batch(h, mix=mix, size=5, seed=4)  # type: ignore[arg-type]
    s = h.apply(b)
    if mix == "skewed":
        assert max(len(r) for r in b.insertions()) >= 1024
    if mix == "delete_all":
        assert h.num_hyperedges == 0
    if mix == "empty":
        assert b.empty
    if mix == "dead_ids":
        assert s.ignored_deletions >= 2
    h.check_integrity()


def test_random_batches_agree_across_backends() -> None:
    out = []
    for backend in host_backends():
        h = dyng.generators.uniform_hypergraph(
            num_vertices=30, num_hyperedges=25, seed=2, resources=dyng.Resources(backend)
        )
        b = dyng.generators.random_hyperedge_batch(h, mix="reuse", size=4, seed=7)
        out.append((b.delete_hyperedges.tolist(), b.insertions(), h.apply(b).inserted_ids.tolist()))
    assert all(o == out[0] for o in out)


def test_legacy_escher_generators() -> None:
    lg = dyng.generators.legacy
    assert lg.standard_library_matches_legacy_streams()
    one = lg.escher_hypergraph_arrays(
        lg.EscherEngine(42), num_hyperedges=8, max_cardinality=5, min_vertex=0, max_vertex=99
    )
    two = lg.escher_hypergraph_arrays(
        lg.EscherEngine(42), num_hyperedges=8, max_cardinality=5, min_vertex=0, max_vertex=99
    )
    assert one.to_lists() == two.to_lists() and one.num_vertices == 100
    engine = lg.EscherEngine(42)
    h = lg.escher_hypergraph(
        engine, num_hyperedges=8, max_cardinality=5, min_vertex=0, max_vertex=99
    )
    p = h.properties
    assert p.reuse == "erase_first" and p.capacity == dyng.CapacityPolicy.escher_gpu()
    gen = lg.EscherBatchGenerator(engine, max_cardinality=5, min_vertex=0, max_vertex=99)
    b = gen.next(h)  # ESCHER-GPU's demo batch: min(2, live) deletions and one more insertion
    assert (b.num_deletions, b.num_insertions) == (2, 3)
    assert b.delete_hyperedges.tolist() == sorted(b.delete_hyperedges.tolist())
    h.apply(b)
    gen = lg.EscherBatchGenerator(engine, batch_size=4, insert_mode="sample")
    b = gen.next(h)
    assert (b.num_deletions, b.num_insertions) == (2, 2)
    with pytest.raises(TypeError):
        lg.EscherBatchGenerator(42)  # type: ignore[arg-type]


@hypothesis.settings(max_examples=40)
@hypothesis.given(
    dst.hypergraphs(max_hyperedges=12, weighted=False),
    st.sampled_from(["erase_first", "insert_first"]),
    st.sampled_from(host_backends()),
    st.data(),
)
def test_batches_against_the_model_and_the_oracle(
    hg: dst.RandomHypergraph, reuse: str, backend: str, data: typing.Any
) -> None:
    h = hg.to_hypergraph(
        id_reuse=reuse, line_graph="sorted_merge", resources=dyng.Resources(backend)
    )
    model = hg.model(reuse)
    for _ in range(3):
        b = data.draw(dst.hyperedge_batches(model))
        s = h.apply(b.to_batch())
        counts = model.apply(b.delete_hyperedges, b.insert_hyperedges, None, s.inserted_ids)
        for name, value in counts.items():
            assert getattr(s, name) == value, name
        v = h.h2v()
        assert [v.row(i).tolist() for i in range(v.num_rows)] == model.rows_by_id(
            h.hyperedge_id_bound
        )
        lg = h.h2h()
        oracle = dyng.testing.rebuild_line_graph(h)
        assert np.array_equal(lg.col_ind, oracle.col_ind)
        assert np.array_equal(lg.row_ptr, oracle.row_ptr)
        assert np.array_equal(lg.out, oracle.out)
        h.check_integrity()


@hypothesis.settings(max_examples=20)
@hypothesis.given(dst.hypergraphs(weighted=True, max_hyperedges=8), st.data())
def test_weighted_batches(hg: dst.RandomHypergraph, data: typing.Any) -> None:
    h = hg.to_hypergraph()
    model = hg.model()
    b = data.draw(dst.hyperedge_batches(model))
    s = h.apply(b.to_batch())
    model.apply(b.delete_hyperedges, b.insert_hyperedges, b.insert_weights, s.inserted_ids)
    w = h.weights
    assert w is not None and model.weights is not None
    assert {i: int(w[i]) for i in model.live_ids} == model.weights
