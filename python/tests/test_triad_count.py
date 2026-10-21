# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.triad_count (PLAN Sections 5.4 and 5.5; ADR 0059): compute and update against two
oracles, the make-test counts of ESCHER-GPU@abcf9b2, "update chain == compute" under Hypothesis,
the result and stats objects, the errors, stale results and dyng.update over hypergraphs.

The oracles: dyng.testing.triad_reference (ESCHER-GPU's CPU reference, C++, sharing no code
with the algorithm) and brute_force() below (pure Python: every triple of live hyperedges, its
Venn regions, and the h-motif classes from the documented table of canonical Venn ids).
"""

from __future__ import annotations

import copy
import doctest
import itertools
import pickle
import typing
from math import comb
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.triad_count import Pattern

hypothesis = pytest.importorskip("hypothesis")
from dyng.testing import strategies as dst  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from triad_fixture import EXES, KINDS, read_fixture, regenerate  # noqa: E402

# The canonical Venn id of every h-motif class 1..30 (bin k - 1), the table of
# <dyng/testing/triad_reference.hpp>: bit 0, 1, 2 "A only", "B only", "C only"; bit 3, 4, 5
# "A and B only", "B and C only", "C and A only"; bit 6 "A, B and C".
CANONICAL = [64, 65, 67, 71, 72, 73, 76, 75, 77, 79, 88, 90, 89, 91, 93, 95, 120, 121, 123, 127,
             24, 26, 25, 27, 29, 31, 56, 57, 59, 63]  # fmt: skip


def _venn(a: set[int], b: set[int], c: set[int]) -> int:
    regions = [a - b - c, b - a - c, c - a - b, (a & b) - c, (b & c) - a, (c & a) - b, a & b & c]
    return sum(1 << i for i, r in enumerate(regions) if r)


def brute_force(rows: list[list[int]], kind: str) -> list[int]:
    """The counts of the live hyperedges ``rows`` (member lists) by definition."""
    sets = [set(r) for r in rows if r]
    if kind == "stathyper_type1":
        return [sum(comb(len(e), 3) for e in sets)]
    if kind == "stathyper_type2":
        total = 0
        for e, f in itertools.combinations(sets, 2):
            a, b, c = len(e & f), len(e - f), len(f - e)
            if a >= 2 and b > 0 and c > 0:
                total += comb(a, 2) * b * c
        return [total]
    bins = [0] * 30
    total = 0
    for a, b, c in itertools.combinations(sets, 3):
        if not (a & b and b & c and c & a):
            continue  # not a triangle of the line graph (an open or no triad)
        if kind == "stathyper_type3":
            x, y, z = len((a & b) - c), len((b & c) - a), len((c & a) - b)
            total += x * y * z
            continue
        canonical = min(_venn(*p) for p in itertools.permutations((a, b, c)))
        bins[CANONICAL.index(canonical)] += 1
    return [total] if kind == "stathyper_type3" else bins


def live_rows(h: dyng.Hypergraph) -> list[list[int]]:
    return h.to_hyperedge_list().to_lists()


def check(h: dyng.Hypergraph, r: dyng.triad_count.Result) -> None:
    """The result equals both oracles (the brute force on small hypergraphs) and compute()."""
    kind = str(r.kind)
    got = r.counts.tolist()
    if h.num_hyperedges <= 60:
        assert got == brute_force(live_rows(h), kind), kind
    assert got == dyng.testing.triad_reference(h, kind).tolist(), kind
    assert got == dyng.triad_count.compute(h, kind=kind).counts.tolist(), kind
    assert r.graph_version == h.version


SMALL = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4], [0, 1, 2, 3, 4, 5], [5, 6], [1, 2]]


def test_the_module_example_runs() -> None:
    failures, tests = doctest.testmod(dyng.triad_count, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tests > 0 and failures == 0


def test_brute_force_classes() -> None:
    # One closed triad per region pattern: ABC only -> class 1; AB, BC, CA only -> class 27.
    assert brute_force([[0], [0], [0]], "h_motif")[0] == 1
    assert brute_force([[0, 2], [0, 1], [1, 2]], "h_motif")[26] == 1
    assert brute_force([[0, 1], [1, 2], [3]], "h_motif") == [0] * 30  # no triangle


# -------------------------------------------------------------------------------------------------
# compute, update and the oracles
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_compute_and_update_equal_the_oracles(
    res: dyng.Resources, kind: str, weighted: bool
) -> None:
    w = list(range(1, len(SMALL) + 1)) if weighted else None
    h = dyng.Hypergraph.from_lists(SMALL, w, resources=res)
    r = dyng.triad_count.compute(h, kind=kind)
    assert r.num_bins == (30 if kind == "h_motif" else 1)
    check(h, r)
    batches = [
        dyng.HyperedgeBatch(delete_hyperedges=[1, 4], insert_hyperedges=[[0, 1, 3, 4], [2, 5]]),
        dyng.HyperedgeBatch(insert_hyperedges=[[1, 2, 3], [0, 4, 6]]),
        dyng.HyperedgeBatch(delete_hyperedges=[0, 2, 3, 99]),
    ]
    for b in batches:
        if weighted:
            b = dyng.HyperedgeBatch(
                delete_hyperedges=b.delete_hyperedges,
                insert_hyperedges=b.insertions(),
                insert_weights=[7] * b.num_insertions,
            )
        before = r.counts.astype(np.int64)
        st_ = dyng.triad_count.update(h, b, r)
        check(h, r)
        assert st_.delta.dtype == np.int64 and st_.delta.size == r.num_bins
        assert not st_.delta.flags.writeable
        assert st_.delta.tolist() == (r.counts.astype(np.int64) - before).tolist()
        assert int(st_.delta.sum()) == st_.triads_added - st_.triads_removed
        assert st_.deleted_frontier == st_.batch.deleted_hyperedges
        assert st_.inserted_frontier == st_.batch.inserted_hyperedges
        assert st_.frontier_visits == st_.deleted_frontier + st_.inserted_frontier
        assert st_.batch.inserted_ids.dtype == np.int64
        assert st_.fallback_used is False and st_.converged is True
        assert st_.engine_used == "operators"
        assert st_.affected == int(np.count_nonzero(st_.delta))


@pytest.mark.parametrize("kind", ["h_motif", "stathyper_type3"])
def test_backends_agree(kind: str) -> None:
    rows = [[i % 9, (i * 7) % 11, (i * 5) % 13 + 2] for i in range(40)]
    out = []
    for backend in ("sequential", "openmp") if dyng.config()["backends"]["openmp"] else ():
        h = dyng.Hypergraph.from_lists(rows, resources=dyng.Resources(backend))
        r = dyng.triad_count.compute(h, kind=kind)
        st_ = dyng.triad_count.update(
            h, dyng.HyperedgeBatch(delete_hyperedges=[0, 5], insert_hyperedges=[[1, 2, 3]]), r
        )
        out.append((r.counts.tolist(), st_.delta.tolist(), st_.triads_removed))
    assert len(set(map(repr, out))) <= 1


# -------------------------------------------------------------------------------------------------
# Hypothesis: update chain == compute (and the oracle)
# -------------------------------------------------------------------------------------------------


@hypothesis.settings(max_examples=40)
@hypothesis.given(
    dst.hypergraphs(max_hyperedges=10, max_vertices=8),
    st.sampled_from(["erase_first", "insert_first"]),
    st.lists(st.sampled_from(KINDS), min_size=1, max_size=4, unique=True),
    st.data(),
)
def test_update_chain_equals_compute(
    hg: dst.RandomHypergraph, reuse: str, kinds: list[str], data: typing.Any
) -> None:
    backend = data.draw(st.sampled_from(["sequential", "openmp"]))
    if backend == "openmp" and not dyng.config()["backends"]["openmp"]:
        backend = "sequential"
    h = hg.to_hypergraph(id_reuse=reuse, resources=dyng.Resources(backend))
    model = hg.model(reuse)
    results = [dyng.triad_count.compute(h, kind=k) for k in kinds]
    for _ in range(3):
        b = data.draw(dst.hyperedge_batches(model))
        before = [r.counts.astype(np.int64) for r in results]
        stats = dyng.update(h, b.to_batch(), *results)
        model.apply(b.delete_hyperedges, b.insert_hyperedges, None, stats[0].batch.inserted_ids)
        rows = [list(m) for m in model.member_sets()]
        for r, s, old in zip(results, stats, before, strict=True):
            assert np.array_equal(s.batch.inserted_ids, stats[0].batch.inserted_ids)
            assert r.counts.tolist() == dyng.triad_count.compute(h, kind=r.kind).counts.tolist()
            assert r.counts.tolist() == brute_force(rows, str(r.kind))
            assert s.delta.tolist() == (r.counts.astype(np.int64) - old).tolist()


# -------------------------------------------------------------------------------------------------
# ESCHER-GPU's make-test counts (the committed fixture of decision D5)
# -------------------------------------------------------------------------------------------------


def test_make_test_counts_equal_escher_gpus(data: Path, res: dyng.Resources) -> None:
    if not dyng.generators.legacy.standard_library_matches_legacy_streams():
        pytest.skip("this standard library does not reproduce the legacy generator streams")
    groups, cases = read_fixture(data)
    assert [g["name"] for g in groups] == [
        "01-smoke",
        "03-reference_syn100k",
        "07-syn100k_b5000",
        "12-sampled_inserts_synthetic",
    ]
    for g in groups:
        heavy = g["n"] > 3000  # the syn100k groups: the patterns without triangles
        exes = ["type1", "type2"] if heavy else list(EXES)
        h, gen = regenerate(g, res)
        results = [dyng.triad_count.compute(h, kind=EXES[e]) for e in exes]
        for e, r in zip(exes, results, strict=True):
            assert r.counts.tolist() == cases[(g["name"], e)]["full"], (g["name"], e)
        for i in range(g["batches"]):
            b = gen.next(h)
            stats = dyng.update(h, b, *results)
            for e, s in zip(exes, stats, strict=True):
                want = cases[(g["name"], e)]["batches"][i]
                got = (b.num_deletions, b.num_insertions, s.delta.tolist())
                assert got == want, (g["name"], e, i + 1)
        if not heavy:
            for r in results:
                check(h, r)


# -------------------------------------------------------------------------------------------------
# Options, results and stats
# -------------------------------------------------------------------------------------------------


def test_kind_accepts_names_and_the_enum(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    a = dyng.triad_count.compute(h, kind="stathyper_type2")
    b = dyng.triad_count.compute(h, kind=Pattern.stathyper_type2)
    c = dyng.triad_count.compute(h, options=dyng.triad_count.Options(kind="stathyper_type2"))
    assert a.kind is b.kind is c.kind is Pattern.stathyper_type2
    assert a.kind == "stathyper_type2" and str(a.kind) == "stathyper_type2"
    assert a.counts.tolist() == b.counts.tolist() == c.counts.tolist()
    assert a.options == dyng.triad_count.Options(kind="stathyper_type2")
    assert dyng.triad_count.compute(h).kind is Pattern.h_motif  # the default
    with pytest.raises(dyng.InvalidArgumentError, match="'h_motif', 'stathyper_type1'"):
        dyng.triad_count.compute(h, kind="open_triads")  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError, match="cuda_engine"):
        dyng.triad_count.compute(h, cuda_engine="fast")
    with pytest.raises(TypeError, match="unexpected keyword"):
        dyng.triad_count.compute(h, max_length=3)
    assert [p.value for p in Pattern] == KINDS
    assert dyng.triad_count.MAX_BINS == 30


def test_result_object(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    r = dyng.triad_count.compute(h)
    c = r.counts
    assert c.dtype == np.uint64 and c.shape == (30,) and not c.flags.writeable
    with pytest.raises(ValueError):
        c[0] = 1
    assert r.total == int(c.sum()) and r.total > 0
    assert [r.count(i) for i in range(30)] == c.tolist()
    for bad in (-1, 30):
        with pytest.raises(dyng.InvalidArgumentError, match="bin"):
            r.count(bad)
    with pytest.raises(dyng.InvalidArgumentError):
        r.count(1.5)  # type: ignore[arg-type]
    assert r.space == "host" and r.graph_version == 0
    one = dyng.triad_count.compute(h, kind="stathyper_type1")
    assert one.counts.shape == (1,) and one.count(0) == one.total
    with pytest.raises(dyng.InvalidArgumentError):
        one.count(1)
    # clone, copy, deepcopy: independent results of the same state.
    for twin in (r.clone(), copy.copy(r), copy.deepcopy(r)):
        assert twin.counts.tolist() == c.tolist() and twin.kind == r.kind
    twin = r.clone()
    dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[0]), twin)
    assert r.counts.tolist() == c.tolist()  # the original is unchanged (and now stale)
    assert c.tolist() == r.counts.tolist()  # an exported copy never changes
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(r)
    with pytest.raises(TypeError):
        dyng.triad_count.Result()
    assert "h_motif" in repr(r) and "graph_version=0" in repr(r)
    # set_options: the tunables only.
    r.set_options(cuda_engine="fused")
    assert r.options.cuda_engine == "fused"
    with pytest.raises(dyng.InvalidArgumentError, match="kind"):
        r.set_options(kind="stathyper_type1")


def test_stats_of_a_deletion(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]], resources=res)
    r = dyng.triad_count.compute(h)
    st_ = dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[3]), r)
    assert isinstance(st_, dyng.triad_count.Stats)
    assert (st_.triads_removed, st_.triads_added) == (3, 0)
    assert (st_.deleted_frontier, st_.inserted_frontier) == (1, 0)
    assert st_.batch.deleted_hyperedges == 1 and st_.batch.inserted_ids.size == 0
    assert r.total == 1


def test_requirements_name_the_property(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, line_graph="none", resources=res)
    for kind in ("h_motif", "stathyper_type2", "stathyper_type3"):
        with pytest.raises(dyng.InvalidArgumentError, match="line_graph"):
            dyng.triad_count.compute(h, kind=kind)
    one = dyng.triad_count.compute(h, kind="stathyper_type1")  # needs neither
    dyng.triad_count.update(h, dyng.HyperedgeBatch(insert_hyperedges=[[1, 2, 3, 4]]), one)
    check(h, one)
    h = dyng.Hypergraph.from_lists(SMALL, store_v2h=False, resources=res)
    with pytest.raises(dyng.InvalidArgumentError, match="store_v2h"):
        dyng.triad_count.compute(h)
    assert dyng.triad_count.compute(h, kind="stathyper_type1").total > 0


def test_errors(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    r = dyng.triad_count.compute(h)
    with pytest.raises(TypeError, match="Hypergraph"):
        dyng.triad_count.compute(dyng.Graph.from_edges([0], [1]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HyperedgeBatch"):
        dyng.triad_count.update(h, dyng.EdgeBatch(), r)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Result"):
        dyng.triad_count.update(h, dyng.HyperedgeBatch(), object())  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError):  # a member out of range: nothing changes
        dyng.triad_count.update(h, dyng.HyperedgeBatch(insert_hyperedges=[[0, 99]]), r)
    assert h.version == 0 and r.graph_version == 0
    with pytest.raises(dyng.NotSupportedError):  # incidence edits (0.4): nothing changes
        dyng.triad_count.update(h, dyng.HyperedgeBatch(insert_incidences=([0], [5])), r)
    check(h, r)


def test_stale_results(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    a = dyng.triad_count.compute(h)
    b = dyng.triad_count.compute(h, kind="stathyper_type1")
    dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[2]), a)
    with pytest.raises(dyng.StaleResultError):  # b missed the batch
        dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[3]), b)
    with pytest.raises(dyng.StaleResultError):
        dyng.update(h, dyng.HyperedgeBatch(delete_hyperedges=[3]), a, b)
    assert h.version == 1  # nothing applied
    h.apply(dyng.HyperedgeBatch(delete_hyperedges=[3]))
    with pytest.raises(dyng.StaleResultError):
        dyng.triad_count.update(h, dyng.HyperedgeBatch(), a)
    other = dyng.Hypergraph.from_lists(SMALL, resources=res)
    fresh = dyng.triad_count.compute(other)
    with pytest.raises(dyng.StaleResultError):  # a result of another hypergraph
        dyng.triad_count.update(h, dyng.HyperedgeBatch(), fresh)


# -------------------------------------------------------------------------------------------------
# dyng.update over hypergraphs
# -------------------------------------------------------------------------------------------------


def test_update_two_results_on_one_hypergraph(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    motifs = dyng.triad_count.compute(h)
    type3 = dyng.triad_count.compute(h, kind="stathyper_type3")
    b = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 1, 3], [2, 3, 5]])
    st_m, st_3 = dyng.update(h, b, motifs, type3)
    assert h.version == 1
    assert isinstance(st_m, dyng.triad_count.Stats) and isinstance(st_3, dyng.triad_count.Stats)
    assert st_m.delta.size == 30 and st_3.delta.size == 1
    assert np.array_equal(st_m.batch.inserted_ids, st_3.batch.inserted_ids)
    assert st_m.batch.inserted_ids.dtype == np.int64
    check(h, motifs)
    check(h, type3)
    # all four together, twice
    rs = [dyng.triad_count.compute(h, kind=k) for k in KINDS]
    for b in (
        dyng.HyperedgeBatch(delete_hyperedges=[0, 2]),
        dyng.HyperedgeBatch(insert_hyperedges=[[4, 5, 6], [0, 6]]),
    ):
        stats = dyng.update(h, b, *rs)
        assert len(stats) == 4
        for r in rs:
            check(h, r)


def test_update_errors_over_hypergraphs(res: dyng.Resources) -> None:
    h = dyng.Hypergraph.from_lists(SMALL, resources=res)
    r = dyng.triad_count.compute(h)
    with pytest.raises(dyng.InvalidArgumentError, match="no results"):
        dyng.update(h, dyng.HyperedgeBatch())
    with pytest.raises(dyng.InvalidArgumentError, match="more than once"):
        dyng.update(h, dyng.HyperedgeBatch(), r, r)
    with pytest.raises(TypeError, match="HyperedgeBatch"):
        dyng.update(h, dyng.EdgeBatch(), r)  # type: ignore[call-overload]
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0], [1, 1, 1])
    with pytest.raises(TypeError, match="hypergraph algorithm"):
        dyng.update(h, dyng.HyperedgeBatch(), dyng.sssp.compute(g, 0))  # type: ignore[call-overload]
    with pytest.raises(TypeError):  # a triad result with a graph
        dyng.update(g, dyng.EdgeBatch(), r)  # type: ignore[call-overload]
    assert h.version == 0
    check(h, r)


def test_weighted_hypergraph_with_dyng_update() -> None:
    h = dyng.Hypergraph.from_lists(SMALL, list(range(1, len(SMALL) + 1)))
    rs = [dyng.triad_count.compute(h, kind=k) for k in KINDS]
    b = dyng.HyperedgeBatch(insert_hyperedges=[[1, 2, 6]], insert_weights=[3])
    dyng.update(h, b, *rs)
    for r in rs:
        check(h, r)


def test_citation() -> None:
    text = dyng.citation("triad_count")
    assert "escher2026" in text and "escherplus2026" in text
    assert {"escher2026", "escherplus2026"} <= set(dyng.citation_keys("triad_count"))


# -------------------------------------------------------------------------------------------------
# Triad bins (JSON)
# -------------------------------------------------------------------------------------------------


def test_triad_bins_round_trip(tmp_path: Path, res: dyng.Resources) -> None:
    for kind in KINDS:
        h = dyng.Hypergraph.from_lists(SMALL, resources=res)
        r = dyng.triad_count.compute(h, kind=kind)
        st_ = dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[0]), r)
        path = tmp_path / kind / "bins.json"
        dyng.io.write_triad_bins(path, r, deltas=[st_.delta, st_.delta])
        back = dyng.io.read_triad_bins(path)
        assert back == dyng.io.TriadBins.of(r, [st_.delta, st_.delta])
        assert back.kind == kind and back.graph_version == 1
        assert back.counts.dtype == np.uint64 and back.counts.tolist() == r.counts.tolist()
        assert back.deltas.shape == (2, r.num_bins) and back.deltas.dtype == np.int64
        assert path.read_text() == dyng.io.triad_bins_json(back)
        dyng.io.write_triad_bins(tmp_path / "again.json", back)
        assert dyng.io.read_triad_bins(tmp_path / "again.json") == back
    # no deltas: the key is left out
    r = dyng.triad_count.compute(h, kind="stathyper_type1")
    text = dyng.io.triad_bins_json(r)
    assert '"deltas"' not in text and text.endswith("}\n")
    assert dyng.io.read_triad_bins(_write(tmp_path / "n.json", text)).deltas.shape == (0, 1)
    # extremes of the value ranges
    big = dyng.io.TriadBins(
        "stathyper_type3",
        np.array([2**64 - 1], dtype=np.uint64),
        2**64 - 1,
        [[-(2**63)], [2**63 - 1]],
    )
    dyng.io.write_triad_bins(tmp_path / "big.json", big)
    assert dyng.io.read_triad_bins(tmp_path / "big.json") == dyng.io.TriadBins(
        "stathyper_type3",
        np.array([2**64 - 1], dtype=np.uint64),
        2**64 - 1,
        np.array([[-(2**63)], [2**63 - 1]], dtype=np.int64),
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_triad_bins_errors(tmp_path: Path) -> None:
    with pytest.raises(dyng.InvalidArgumentError, match="kind"):
        dyng.io.triad_bins_json(dyng.io.TriadBins("motifs", np.zeros(30, dtype=np.uint64)))
    with pytest.raises(dyng.InvalidArgumentError, match="30 bins|has 29"):
        dyng.io.triad_bins_json(dyng.io.TriadBins("h_motif", np.zeros(29, dtype=np.uint64)))
    with pytest.raises(dyng.InvalidArgumentError, match="delta row 0"):
        dyng.io.triad_bins_json(
            dyng.io.TriadBins("stathyper_type1", np.zeros(1, dtype=np.uint64), 0, [[1, 2]])
        )
    with pytest.raises(dyng.InvalidArgumentError, match="does not fit"):
        dyng.io.triad_bins_json(dyng.io.TriadBins("stathyper_type1", np.array([-1])))
    with pytest.raises(TypeError, match="TriadBins"):
        dyng.io.write_triad_bins(tmp_path / "x.json", [1, 2])
    bad = _write(tmp_path / "bad.json", '{"format": "dyng.triad_bins", "version": 2}\n')
    with pytest.raises(dyng.FileFormatError, match="version 2") as e:
        dyng.io.read_triad_bins(bad)
    assert e.value.line == 1 and e.value.path == str(bad)
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_triad_bins(tmp_path / "missing.json")
