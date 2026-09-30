# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Graph: construction, dtype dispatch (never narrowing silently), properties, CSR."""

from __future__ import annotations

import dyng
import numpy as np
import pytest


def test_from_edges_basic(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0, 0, 1], [2, 1, 2], [3, 4, 5], resources=res)
    assert (g.num_vertices, g.num_edges, g.num_weights) == (3, 3, 1)
    assert g.version == 0 and g.directed and g.has_transposed and g.weighted
    assert g.space == "host"
    assert g.resources is res
    csr = g.to_csr()
    assert csr.row_ptr.tolist() == [0, 2, 3, 3]
    assert csr.col_ind.tolist() == [1, 2, 2]  # sorted rows by default
    assert csr.weights is not None and csr.weights[:, 0].tolist() == [4, 3, 5]
    src, dst, w = g.edges()
    assert src.tolist() == [0, 0, 1] and dst.tolist() == [1, 2, 2]
    assert w is not None and w.shape == (3, 1)
    assert "num_vertices=3" in repr(g)
    g.check_integrity()


def test_graph_is_not_constructible_directly() -> None:
    with pytest.raises(TypeError):
        dyng.Graph()


def test_num_vertices_and_isolated_vertices() -> None:
    g = dyng.Graph.from_edges([0], [1], num_vertices=5)
    assert g.num_vertices == 5 and not g.weighted
    empty = dyng.Graph.from_edges([], [])
    assert empty.num_vertices == 0 and empty.num_edges == 0


def test_multi_weights() -> None:
    w = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int32)
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0], w)
    assert g.num_weights == 2
    assert g.to_csr().weights.tolist() == w.tolist()  # type: ignore[union-attr]


def test_dtype_dispatch_from_declared_dtypes() -> None:
    s32, d32 = np.array([0, 1], np.int32), np.array([1, 2], np.int32)
    assert dyng.Graph.from_edges(s32, d32, [1, 1]).vertex_dtype == np.int32
    g64 = dyng.Graph.from_edges(s32.astype(np.int64), d32.astype(np.int64), [1, 1])
    assert g64.vertex_dtype == np.int64 and g64.edge_dtype == np.int64
    # narrower types are widened
    g8 = dyng.Graph.from_edges(s32.astype(np.int8), d32.astype(np.uint16))
    assert g8.vertex_dtype == np.int32
    # lists have no dtype: the smallest width that holds the values
    assert dyng.Graph.from_edges([0, 1], [1, 2]).vertex_dtype == np.int32
    # (a graph with an id >= 2**31 would have 2**31 vertices: the rule is checked on its own)
    from dyng._dtypes import infer_id_dtype

    assert infer_id_dtype([[0, 2**31], [1, 2]], None, "test") == np.int64
    assert infer_id_dtype([[0, 5], range(3)], None, "test") == np.int32
    # uint32 needs 64 bits
    assert dyng.Graph.from_edges(s32.astype(np.uint32), d32, [1, 1]).vertex_dtype == np.int64


def test_int64_ids_are_never_narrowed_silently() -> None:
    s = np.array([0, 1], np.int64)
    d = np.array([1, 2], np.int64)
    g = dyng.Graph.from_edges(s, d, [1, 1])
    assert g.vertex_dtype == np.int64  # not int32, although the values would fit
    g32 = dyng.Graph.from_edges(s, d, [1, 1], vertex_dtype="int32")  # asked for: checked
    assert g32.vertex_dtype == np.int32
    with pytest.raises(dyng.InvalidArgumentError, match="does not fit int32"):
        dyng.Graph.from_edges(np.array([0, 2**40]), np.array([1, 2]), [1, 1], vertex_dtype="int32")


def test_unsupported_combinations_list_the_supported_ones() -> None:
    s = np.array([0, 1], np.int64)
    with pytest.raises(dyng.NotSupportedError, match="supported combinations") as e:
        dyng.Graph.from_edges(s, s)  # int64 ids without weights: not instantiated
    assert "vertex int32, edge int64, weight unweighted" in str(e.value)
    with pytest.raises(dyng.NotSupportedError):
        dyng.Graph.from_edges([0], [1], [1], vertex_dtype="int64", edge_dtype="int32")
    with pytest.raises(dyng.NotSupportedError):
        dyng.Graph.from_edges([0], [1], vertex_dtype="int16")
    with pytest.raises(dyng.NotSupportedError, match="uint64"):
        dyng.Graph.from_edges(np.array([0], np.uint64), np.array([1], np.uint64), [1])


def test_bad_inputs() -> None:
    with pytest.raises(dyng.InvalidArgumentError, match="integers"):
        dyng.Graph.from_edges(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    with pytest.raises(dyng.InvalidArgumentError, match="differ in length"):
        dyng.Graph.from_edges([0, 1], [1])
    with pytest.raises(dyng.InvalidArgumentError, match="weights must be integers"):
        dyng.Graph.from_edges([0], [1], [1.5])
    with pytest.raises(dyng.InvalidArgumentError, match="expected 1 weights"):
        dyng.Graph.from_edges([0], [1], [1, 2])
    with pytest.raises(dyng.InvalidArgumentError, match="1-D"):
        dyng.Graph.from_edges([[0, 1]], [[1, 2]])
    with pytest.raises(dyng.InvalidArgumentError):  # an id >= num_vertices (the library's check)
        dyng.Graph.from_edges([0, 5], [1, 2], num_vertices=3)
    with pytest.raises(dyng.InvalidArgumentError):  # weights beyond int32
        dyng.Graph.from_edges([0], [1], np.array([2**40]))


def test_edge_dtype() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 0], edge_dtype="int64")
    assert g.edge_dtype == np.int64 and g.vertex_dtype == np.int32
    assert g.to_csr().row_ptr.dtype == np.int64


def test_presets_and_overrides() -> None:
    mosp = dyng.Graph.from_edges([1, 0, 0], [0, 2, 1], [1, 1, 1], properties="mosp_compatible")
    p = mosp.properties
    assert p.order == "append" and p.parallel_edges == "allow"
    assert mosp.to_csr().col_ind.tolist() == [2, 1, 0]  # append keeps the input order
    ce = dyng.Graph.from_edges([0, 1], [1, 0], properties="cycle_enum_compatible")
    assert ce.properties.semantics.as_sets and ce.properties.semantics.on_self_loop == "drop"
    und = dyng.Graph.from_edges([0], [1], [1], directed=False)
    assert not und.directed and und.num_edges == 2
    s = dyng.Graph.from_edges([0], [1], semantics="set")
    assert s.properties.semantics == dyng.BatchSemantics.set()
    props = dyng.GraphProperties(order="append", parallel_edges="allow")
    assert dyng.Graph.from_edges([0, 0], [1, 1], [1, 2], properties=props).num_edges == 2
    assert dyng.Graph.from_edges([0, 0], [1, 1], [1, 2]).num_edges == 1  # merged, last wins
    with pytest.raises(dyng.InvalidArgumentError, match="preset"):
        dyng.Graph.from_edges([0], [1], properties="fast")  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError, match="order"):
        dyng.Graph.from_edges([0], [1], row_order="random")
    with pytest.raises(dyng.NotSupportedError):
        dyng.Graph.from_edges([0], [1], layout="slotted")
    with pytest.raises(dyng.InvalidArgumentError, match="semantics"):
        dyng.Graph.from_edges([0], [1], semantics="sometimes")  # type: ignore[arg-type]


def test_from_csr() -> None:
    row_ptr = np.array([0, 2, 3, 3], np.int32)
    col_ind = np.array([2, 1, 0], np.int32)
    g = dyng.Graph.from_csr(row_ptr, col_ind, [5, 6, 7])
    assert (g.num_vertices, g.num_edges) == (3, 3)
    csr = g.to_csr()
    assert csr.col_ind.tolist() == [1, 2, 0]  # sorted
    assert csr.weights[:, 0].tolist() == [6, 5, 7]  # type: ignore[index]
    g64 = dyng.Graph.from_csr(row_ptr.astype(np.int64), col_ind, [5, 6, 7])
    assert g64.edge_dtype == np.int64
    wk = np.array([[1, 2], [3, 4], [5, 6]])
    g2 = dyng.Graph.from_csr(row_ptr, col_ind, wk, properties="mosp_compatible")
    assert g2.to_csr().weights.tolist() == wk.tolist()  # type: ignore[union-attr]
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.Graph.from_csr([0, 5], [0], [1])


def test_apply_clone_reserve(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1], resources=res)
    c = g.clone()
    s = g.apply(dyng.EdgeBatch(insert=([2], [3], [4]), delete=([0], [1])))
    assert isinstance(s, dyng.ApplySummary)
    assert (s.inserted_edges, s.deleted_edges, s.inserted_vertices, s.num_vertices_after) == (
        1,
        1,
        1,
        4,
    )
    assert g.version == 1 and g.num_vertices == 4
    assert c.version == 0 and c.num_vertices == 3  # the clone is independent
    g.reserve(100)
    with pytest.raises(dyng.InvalidArgumentError):
        g.reserve(-1)
    other = dyng.Graph.from_edges([0], [1], [1], resources=dyng.Resources.sequential())
    moved = other.to_backend(res)
    assert moved.resources is res and moved.num_edges == 1


def test_apply_vertex_operations_are_0_3() -> None:
    g = dyng.Graph.from_edges([0], [1], [1])
    with pytest.raises(dyng.NotSupportedError):
        g.apply(dyng.EdgeBatch(insert_vertices=[5]))


def test_semantics_errors() -> None:
    sem = dyng.BatchSemantics(on_missing_delete="error")
    g = dyng.Graph.from_edges([0], [1], [1], semantics=sem)
    with pytest.raises(dyng.InvalidArgumentError):
        g.apply(dyng.EdgeBatch(delete=([1], [0])))
    assert g.version == 0  # the strong guarantee: nothing changed


def test_capacity_error_is_memory_error() -> None:
    assert issubclass(dyng.CapacityError, MemoryError)
