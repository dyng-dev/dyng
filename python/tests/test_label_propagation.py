# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.label_propagation and the 0.4 graph surface it needs (M10b step 3; ADR 0108).

- AD14's dtype rule: float32 weights select the float graphs, float64 only with
  weight_dtype="float32" (checked), never narrowed silently; Graph.weighted keeps its meaning.
- Graph.with_capacity (slotted and slack rows, net_effect), the vertex arrays of EdgeBatch,
  ApplySummary's vertex fields, num_active_vertices and vertex_status.
- The published setup on LabelPropagation-CUDA's 5k5 stream gives dyng-compat-dynlp's sequential
  labels byte for byte; the harmonic oracle; the keywords mirror the C++ fields.
- Result.labels: a read-only float32 dyng.Array, zero-copy (NumPy, PyTorch), copy-on-write.
- AD19: OpenMP resources raise NotSupportedError naming the fix.
Every test passes explicit resources (AD19).
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
from pathlib import Path

import dyng
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
DIGESTS = REPO / "examples" / "data" / "label_propagation_5k5_damped.sha256"
GENERATOR_DIGESTS = (
    REPO / "cpp" / "tests" / "data" / "goldens" / "label_propagation"
    / "generator_5k5_d1e-4_s42.sha256"
)  # fmt: skip

SEQ = dyng.Resources("sequential")
F32 = np.float32


def digests(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            sha, name = line.split()
            out[name] = sha
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def stream5k5() -> dyng.generators.VertexArrivalStream:
    return dyng.generators.legacy.dynlp_vertex_arrival_stream(
        total_vertices=5000, num_batches=5, seed=42, resources=SEQ
    )


def published(
    stream: dyng.generators.VertexArrivalStream, res: dyng.Resources = SEQ, **options: object
) -> tuple[dyng.Graph, dyng.label_propagation.Result, list[dyng.label_propagation.Stats]]:
    """The published setup over the whole stream."""
    g = dyng.Graph.with_capacity(stream.row_capacity, resources=res)
    r = dyng.label_propagation.compute(
        g, similarity_threshold=stream.similarity_threshold, **options
    )
    r.reserve(stream.max_batch_vertices, stream.max_batch_edges)
    stats = [dyng.label_propagation.update(g, b, r) for b in stream.batches]
    return g, r, stats


# -------------------------------------------------------------------------------------------------
# The dtype rule (AD14)
# -------------------------------------------------------------------------------------------------


def test_float32_weights_select_the_float_graph() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], np.array([0.5, 0.25], F32), resources=SEQ)
    assert g.weight_dtype == np.dtype(F32) and not g.weighted  # weighted: int32 only (0.1)
    assert g.edges()[2] is not None and g.edges()[2].dtype == F32
    assert g.to_csr().weights.tolist() == [[0.5], [0.25]]
    h = dyng.Graph.from_edges([0, 1], [1, 2], [3, 4], resources=SEQ)
    assert h.weight_dtype == np.dtype(np.int32) and h.weighted
    u = dyng.Graph.from_edges([0, 1], [1, 2], resources=SEQ)
    assert u.weight_dtype is None and not u.weighted
    assert "float32" in repr(g)


def test_float64_is_never_narrowed_silently() -> None:
    w64 = np.array([0.1, 0.2])
    with pytest.raises(dyng.InvalidArgumentError, match="float64 weights"):
        dyng.Graph.from_edges([0, 1], [1, 2], w64, resources=SEQ)
    g = dyng.Graph.from_edges([0, 1], [1, 2], w64, weight_dtype="float32", resources=SEQ)
    assert g.to_csr().weights.reshape(-1).tolist() == w64.astype(F32).tolist()
    # checked: finite and within float32's range, also with the explicit request
    for bad, match in (([np.nan, 0.5], "NaN"), ([np.inf, 0.5], "NaN"), ([1e39, 0.5], "range")):
        with pytest.raises(dyng.InvalidArgumentError, match=match):
            dyng.Graph.from_edges([0, 1], [1, 2], np.array(bad), weight_dtype="float32")
    with pytest.raises(dyng.InvalidArgumentError, match="NaN"):
        dyng.Graph.from_edges([0, 1], [1, 2], np.array([np.nan, 1], F32))
    # integers become a float graph only on request; then exactly or rounded (asked for)
    gi = dyng.Graph.from_edges([0], [1], [3], weight_dtype="float32", resources=SEQ)
    assert gi.weight_dtype == np.dtype(F32)
    with pytest.raises(dyng.NotSupportedError, match="int32 or float32"):
        dyng.Graph.from_edges([0], [1], [3], weight_dtype="float64")
    with pytest.raises(dyng.InvalidArgumentError, match="weight_dtype without weights"):
        dyng.Graph.from_edges([0], [1], weight_dtype="float32")


def test_batches_of_a_float_graph() -> None:
    g = dyng.Graph.from_edges(
        [0], [1], np.array([0.5], F32), directed=False, num_vertices=3, resources=SEQ
    )
    g.apply(dyng.EdgeBatch(insert=([1], [2], np.array([0.75], F32))))
    # float64 batch weights: only when exactly float32 values ...
    g.apply(dyng.EdgeBatch(insert=([0], [2], np.array([0.25]))))
    with pytest.raises(dyng.InvalidArgumentError, match="not exactly float32"):
        g.apply(dyng.EdgeBatch(insert=([0], [2], np.array([0.1]))))
    # ... or converted on request (checked)
    b = dyng.EdgeBatch(insert=([0], [2], np.array([0.1])), weight_dtype="float32")
    assert b.insert_weights is not None and b.insert_weights.dtype == F32
    g.apply(b)
    with pytest.raises(dyng.InvalidArgumentError, match="NaN"):
        dyng.EdgeBatch(insert=([0], [2], np.array([np.inf])), weight_dtype="float32")
    # an int32 graph refuses float weights
    with pytest.raises(dyng.InvalidArgumentError, match="int32"):
        dyng.Graph.from_edges([0], [1], [1], resources=SEQ).apply(b)


# -------------------------------------------------------------------------------------------------
# Graph.with_capacity, vertex batches, net_effect
# -------------------------------------------------------------------------------------------------


def test_with_capacity_and_vertex_batches() -> None:
    g = dyng.Graph.with_capacity([2, 2, 2, 0], resources=SEQ)
    assert g.num_vertices == 4 and g.num_active_vertices == 0 and g.num_edges == 0
    p = g.properties
    assert (
        p.layout == "slotted" and not p.directed and p.semantics == dyng.BatchSemantics.net_effect()
    )
    assert p.semantics.as_net_effect and g.edge_dtype == np.dtype(np.int32)
    assert g.vertex_status().tolist() == [dyng.VertexStatus.inactive] * 4
    b = dyng.EdgeBatch(
        insert=([0, 1], [1, 2], np.array([0.5, 0.25], F32)),
        insert_vertices=[0, 1, 2],
        vertex_labels=[1, -1, 0],
    )
    s = g.apply(b)
    # an undirected edge counts twice (both rows), as num_edges
    assert (s.inserted_vertices, s.deleted_vertices, s.inserted_edges) == (3, 0, 4)
    assert g.num_active_vertices == 3 and g.num_edges == 4
    s = g.apply(dyng.EdgeBatch(delete_vertices=[1]))
    assert s.deleted_vertices == 1 and g.num_active_vertices == 2
    assert g.vertex_status().tolist() == [
        dyng.VertexStatus.active,
        dyng.VertexStatus.deleted,
        dyng.VertexStatus.active,
        dyng.VertexStatus.inactive,
    ]
    # an edge to an inactive vertex is dropped (net_effect)
    s = g.apply(dyng.EdgeBatch(insert=([0], [3], np.array([0.5], F32))))
    assert s.dropped_insertions == 2  # both directions, as inserted_edges
    # a slotted row's overflow is refused before anything changes
    v = g.version
    with pytest.raises(dyng.CapacityError):
        g.apply(dyng.EdgeBatch(insert=([0, 0], [2, 2], np.array([0.5, 0.5], F32))))
    assert g.version == v
    # compact rows refuse vertex batches
    c = dyng.Graph.from_edges([0], [1], np.array([0.5], F32), directed=False, resources=SEQ)
    with pytest.raises(dyng.NotSupportedError):
        c.apply(dyng.EdgeBatch(insert_vertices=[0]))
    assert c.vertex_status().tolist() == [dyng.VertexStatus.active] * 2
    assert c.num_active_vertices == 2


def test_slack_rows_move_instead_of_overflowing() -> None:
    g = dyng.Graph.with_capacity([1, 1, 1, 1], layout="slack", headroom=0.0, resources=SEQ)
    assert g.properties.layout == "slack" and g.properties.headroom == 0.0
    g.apply(dyng.EdgeBatch(insert_vertices=[0, 1, 2, 3]))
    s = g.apply(dyng.EdgeBatch(insert=([0, 0, 0], [1, 2, 3], np.full(3, 0.5, F32))))
    assert s.relocated_rows >= 1 and g.num_edges == 6
    g.check_integrity()
    with pytest.raises(dyng.NotSupportedError):
        dyng.Graph.with_capacity([1], semantics="upsert_last_wins", resources=SEQ)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.Graph.with_capacity([1, -1], resources=SEQ)


def test_vertex_batches_round_trip_through_batch_files(
    tmp_path: Path, stream5k5: dyng.generators.VertexArrivalStream
) -> None:
    """The M10a generator batches round-trip through .dgt and .dgb (AD13)."""
    dyng.io.write_batches(tmp_path / "s.dgt", stream5k5.batches)
    dyng.io.write_binary_batches(tmp_path / "s.dgb", stream5k5.batches)
    text = dyng.io.read_batches(tmp_path / "s.dgt", weight_dtype="float32")
    binary = dyng.io.read_binary_batches(tmp_path / "s.dgb")
    with dyng.io.BatchReader(tmp_path / "s.dgb") as r:
        assert r.header.weight_type == "float32" and r.header.flags & 16
    for a, b, c in zip(stream5k5.batches, text, binary, strict=True):
        for x in (b, c):
            assert isinstance(x, dyng.EdgeBatch)
            for name in ("insert_src", "insert_dst", "insert_vertices", "delete_vertices"):
                assert np.array_equal(getattr(a, name), getattr(x, name)), name
            assert np.array_equal(a.vertex_labels, x.vertex_labels)
            wa, wx = a.insert_weights, x.insert_weights
            assert wa is not None and wx is not None and wx.dtype == F32
            assert wa.tobytes() == wx.tobytes()
    # an edge-only file keeps 0.3's bytes: no vertex records, int32 weights
    dyng.io.write_binary_batches(tmp_path / "e.dgb", [dyng.EdgeBatch(insert=([0], [1], [2]))])
    with dyng.io.BatchReader(tmp_path / "e.dgb") as r:
        assert r.header.flags == 0 and r.header.weight_type == "int32"


def test_float_edge_lists_and_label_files(tmp_path: Path) -> None:
    path = tmp_path / "g.txt"
    path.write_text("0 1 0.5\n1 2 0.125\n")
    g = dyng.io.read_edge_list(
        path, weight_dtype="float32", ids="as_is", directed=False, resources=SEQ
    )
    assert g.weight_dtype == np.dtype(F32) and g.num_edges == 4
    out = tmp_path / "w.txt"
    dyng.io.write_edge_list(out, dyng.io.read_edge_list_arrays(path, weight_dtype="float32"))
    assert out.read_text() == "0 1 0.5\n1 2 0.125\n"
    bad = tmp_path / "bad.txt"
    bad.write_text("0 1 nan\n")
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_edge_list(bad, weight_dtype="float32")
    with pytest.raises(dyng.FileFormatError):  # the integer path is unchanged: no float tokens
        dyng.io.read_edge_list(path, num_weights=1)
    labels = np.array([0.0, 1.0, 0.5, 0.1234565, 1 / 3], F32)
    dyng.io.write_labels(tmp_path / "l.txt", labels)
    assert (tmp_path / "l.txt").read_text() == (
        f"0 0.000000\n1 1.000000\n2 0.500000\n3 {float(labels[3]):.6f}\n4 0.333333\n"
    )
    back = dyng.io.read_labels(tmp_path / "l.txt", 5)
    assert back.dtype == F32 and np.allclose(back, labels, atol=6e-7)


# -------------------------------------------------------------------------------------------------
# label_propagation: the published setup, parity, the oracle, the options
# -------------------------------------------------------------------------------------------------


def test_published_setup_equals_the_compat_tool(
    tmp_path: Path, stream5k5: dyng.generators.VertexArrivalStream
) -> None:
    """Damped on the original's 5k5 stream: the labels are dyng-compat-dynlp's sequential output
    byte for byte (the committed digest; the CLI test runs the tool itself when it is built)."""
    g, r, stats = published(stream5k5, update_schedule="damped")
    assert r.labels.dtype == F32 and len(r.labels) == 5000
    dyng.io.write_labels(tmp_path / "dynlp_labels.txt", r.labels)
    assert sha256(tmp_path / "dynlp_labels.txt") == digests(DIGESTS)["sequential"]
    assert [s.iterations for s in stats] == [0, 1217, 548, 327, 65]  # results.csv of the tool
    assert [s.components for s in stats] == [50, 829, 850, 836, 858]
    assert all(s.converged and not s.fallback_used for s in stats)
    assert stats[1].batch.inserted_vertices == 1127 and stats[0].initial_frontier == 0
    assert r.graph_version == g.version == 5
    seeds = r.seed_labels.to_numpy()
    assert seeds.dtype == np.int8 and set(np.unique(seeds).tolist()) <= {-1, 0, 1}
    lab = r.labels.to_numpy()
    assert np.all(lab[seeds == 1] == 1) and np.all(lab[seeds == 0] == 0)


def test_graph_files_are_the_originals(
    tmp_path: Path, stream5k5: dyng.generators.VertexArrivalStream
) -> None:
    stream5k5.write_graph_files(tmp_path / "graph")
    want = digests(GENERATOR_DIGESTS)
    for name in ("csrRowPtr.txt", "csrColInd.txt", "csrValues.txt", "csrTrueLabels.txt"):
        assert sha256(tmp_path / "graph" / name) == want[name], name


def test_the_harmonic_oracle_at_delta_1e6() -> None:
    """AD7 on 5k5 at delta 1e-6, damped: mean |F - F*| <= 2000 delta on every batch."""
    s = dyng.generators.legacy.dynlp_vertex_arrival_stream(
        total_vertices=5000, num_batches=5, seed=42, resources=SEQ
    )
    g = dyng.Graph.with_capacity(s.row_capacity, resources=SEQ)
    r = dyng.label_propagation.compute(
        g, similarity_threshold=s.similarity_threshold, tolerance=1e-6, update_schedule="damped"
    )
    for b in s.batches:
        dyng.label_propagation.update(g, b, r)
        f = dyng.testing.harmonic_solution(g, r.seed_labels)
        assert f.converged and f.num_seeds > 0
        mean, worst = f.error(r.labels)
        assert mean <= 2000 * 1e-6 and worst <= 1.0, (mean, worst)


def test_the_library_generator(stream5k5: dyng.generators.VertexArrivalStream) -> None:
    kw = {"total_vertices": 3001, "num_batches": 4, "seed": 7, "average_degree": 7}
    a = dyng.generators.vertex_arrival_stream(**kw, resources=SEQ)
    b = dyng.generators.vertex_arrival_stream(**kw, resources=dyng.Resources.openmp(3))
    c = dyng.generators.vertex_arrival_stream(**kw, edge_dtype="int64", resources=SEQ)
    assert c.row_ptr.dtype == np.int64 and c.row_capacity.dtype == np.int64
    for x in (b, c):
        assert np.array_equal(a.col_ind, x.col_ind) and a.weights.tobytes() == x.weights.tobytes()
        assert a.similarity_threshold == x.similarity_threshold
        assert np.array_equal(a.row_capacity, x.row_capacity)
    assert a.num_vertices == 3001 and a.num_batches == 4
    assert a.max_batch_vertices == max(bb.insert_vertices.size for bb in a.batches)
    assert a.max_batch_edges == max(bb.num_insertions for bb in a.batches)
    assert set(np.unique(a.true_labels).tolist()) == {0, 1}
    legacy = dyng.generators.legacy.dynlp_vertex_arrival_stream(**kw, resources=SEQ)
    assert not np.array_equal(a.col_ind, legacy.col_ind)  # the library's own draws
    assert "VertexArrivalStream" in repr(stream5k5) and stream5k5.num_vertices == 5000


def test_options_mirror_the_cpp_fields() -> None:
    names = [f.name for f in dataclasses.fields(dyng.label_propagation.Options)]
    assert names == [
        "similarity_threshold",
        "tolerance",
        "max_iterations",
        "on_iteration_limit",
        "update_schedule",
        "cuda_engine",
    ]
    stats = [f.name for f in dataclasses.fields(dyng.label_propagation.Stats)]
    assert stats == [
        "affected",
        "iterations",
        "frontier_visits",
        "fallback_used",
        "converged",
        "engine_used",
        "batch",
        "components",
        "sparsified_edges",
        "initial_frontier",
    ]
    g = dyng.Graph.with_capacity([1, 1], resources=SEQ)
    r = dyng.label_propagation.compute(g)
    assert r.options == dyng.label_propagation.Options()
    r.set_options(similarity_threshold=0.25, update_schedule="damped", max_iterations=7)
    o = r.options
    assert (o.similarity_threshold, o.update_schedule, o.max_iterations) == (0.25, "damped", 7)
    r.set_options(similarity_threshold=None)
    assert r.options.similarity_threshold is None
    with pytest.raises(TypeError, match="unexpected keyword"):
        dyng.label_propagation.compute(g, delta=1e-4)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.label_propagation.compute(g, tolerance=0.0)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.label_propagation.compute(g, max_iterations=0)


def test_compute_with_seeds_on_a_compact_graph() -> None:
    g = dyng.Graph.from_edges(
        [0, 1, 3], [1, 2, 4], np.array([1, 1, 1], F32), directed=False, resources=SEQ
    )
    r = dyng.label_propagation.compute(g, [0, 2], [1, 0], tolerance=1e-7)
    lab = r.labels.to_numpy()
    assert lab[0] == 1.0 and lab[2] == 0.0 and abs(float(lab[1]) - 0.5) < 1e-3
    assert lab[3] == 0.5 and lab[4] == 0.5  # a seed-free component
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.label_propagation.compute(g, [0], [2])
    with pytest.raises(TypeError, match="both or neither"):
        dyng.label_propagation.compute(g, [0])
    with pytest.raises(dyng.NotSupportedError, match="float32 graphs"):
        dyng.label_propagation.compute(dyng.Graph.from_edges([0], [1], [1], resources=SEQ))
    # update needs slot rows
    with pytest.raises(dyng.NotSupportedError):
        dyng.label_propagation.update(g, dyng.EdgeBatch(insert_vertices=[0]), r)


def test_openmp_resources_name_the_fix() -> None:
    """AD19: no OpenMP backend in 0.4; the error names 'sequential', 'cuda' and the fix."""
    if not dyng.config()["backends"]["openmp"]:
        pytest.skip("a build without OpenMP")
    omp = dyng.Resources.openmp(2)
    g = dyng.Graph.with_capacity([1, 1], resources=omp)
    for call in (
        lambda: dyng.label_propagation.compute(g),
        lambda: dyng.label_propagation.compute(g, resources=omp),
    ):
        with pytest.raises(dyng.NotSupportedError) as e:
            call()
        text = str(e.value)
        assert "'sequential'" in text and "'cuda'" in text
        assert 'dyng.Resources("sequential")' in text
    r = dyng.label_propagation.compute(g, resources=SEQ)  # the graph's memory is the host's
    with pytest.raises(dyng.NotSupportedError, match="sequential"):
        dyng.label_propagation.update(g, dyng.EdgeBatch(insert_vertices=[0]), r, resources=omp)
    with pytest.raises(dyng.NotSupportedError, match="sequential"):
        dyng.update(g, dyng.EdgeBatch(insert_vertices=[0]), r, resources=omp)


def test_dyng_update_composes(stream5k5: dyng.generators.VertexArrivalStream) -> None:
    g1, r1, s1 = published(stream5k5, update_schedule="damped")
    g2 = dyng.Graph.with_capacity(stream5k5.row_capacity, resources=SEQ)
    r2 = dyng.label_propagation.compute(
        g2, similarity_threshold=stream5k5.similarity_threshold, update_schedule="damped"
    )
    for b, want in zip(stream5k5.batches, s1, strict=True):
        (st,) = dyng.update(g2, b, r2)
        assert isinstance(st, dyng.label_propagation.Stats) and st == want
    assert r1.labels.to_numpy().tobytes() == r2.labels.to_numpy().tobytes()


# -------------------------------------------------------------------------------------------------
# Result arrays: read-only, zero-copy, copy-on-write
# -------------------------------------------------------------------------------------------------


def _ptr(a: dyng.Array) -> int:
    return int(a.__array_interface__["data"][0])


def test_labels_are_read_only_zero_copy_and_copy_on_write(
    stream5k5: dyng.generators.VertexArrivalStream,
) -> None:
    g = dyng.Graph.with_capacity(stream5k5.row_capacity, resources=SEQ)
    r = dyng.label_propagation.compute(
        g, similarity_threshold=stream5k5.similarity_threshold, update_schedule="damped"
    )
    batches = stream5k5.batches
    dyng.label_propagation.update(g, batches[0], r)
    dyng.label_propagation.update(g, batches[1], r)
    a = r.labels
    assert a.dtype == F32 and a.shape == (5000,) and a.device == "cpu"
    view = np.from_dlpack(a)
    assert not view.flags.writeable and _ptr(a) == int(view.__array_interface__["data"][0])
    with pytest.raises(ValueError):
        view[0] = 1.0
    before = view.copy()
    old_ptr = _ptr(a)
    # a live export keeps its labels: the update copies the state once (copy-on-write)
    dyng.label_propagation.update(g, batches[2], r)
    assert np.array_equal(view, before)
    with pytest.raises(dyng.StaleResultError):
        a.to_numpy()
    fresh = r.labels
    assert _ptr(fresh) != old_ptr and not np.array_equal(fresh.to_numpy(), before)
    # without an export the next update writes in place: no copy
    ptr = _ptr(fresh)
    del a, view, fresh
    dyng.label_propagation.update(g, batches[3], r)
    assert _ptr(r.labels) == ptr


def test_torch_from_dlpack_is_zero_copy(stream5k5: dyng.generators.VertexArrivalStream) -> None:
    torch = pytest.importorskip("torch")
    _, r, _ = published(stream5k5, update_schedule="damped")
    labels = r.labels
    t = torch.from_dlpack(labels)
    assert t.dtype == torch.float32 and t.device.type == "cpu"
    assert t.data_ptr() == _ptr(labels)  # the same memory: no copy
    assert np.array_equal(t.numpy(), labels.to_numpy())


def test_clone_reserve_repr_and_pickle(stream5k5: dyng.generators.VertexArrivalStream) -> None:
    g, r, _ = published(stream5k5, update_schedule="damped")
    c = r.clone()
    assert c.labels.to_numpy().tobytes() == r.labels.to_numpy().tobytes()
    assert c.graph_version == r.graph_version and c.space == "host"
    r.reserve(10, 10)
    assert "graph_version=5" in repr(r)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(r)
    with pytest.raises(TypeError):
        dyng.label_propagation.Result()


def test_the_docstring_examples() -> None:
    """The examples of the module, Graph.with_capacity and testing.harmonic_solution run."""
    import doctest

    for module in (dyng.label_propagation, dyng.graph, dyng.testing):
        failures, tests = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
        assert tests > 0 and failures == 0, module.__name__
