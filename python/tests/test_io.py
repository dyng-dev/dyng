# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.io: the readers and writers of the 0.1 formats, byte for byte with the originals."""

from __future__ import annotations

from pathlib import Path

import dyng
import numpy as np
import pytest


def same_bytes(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def csr_files_equal(prefix_a: Path, prefix_b: Path) -> bool:
    return all(
        same_bytes(Path(f"{prefix_a}{part}.txt"), Path(f"{prefix_b}{part}.txt"))
        for part in ("RowPtr", "ColInd", "Values")
    )


def test_csr_triplet_round_trip(data: Path, tmp_path: Path, res: dyng.Resources) -> None:
    src = data / "mosp_graph_io" / "testCase0" / "graphCsr"
    g = dyng.io.read_csr_triplet(src, properties="mosp_compatible", resources=res)
    assert g.num_weights >= 1 and g.properties.order == "append"
    out = tmp_path / "out" / "graphCsr"
    dyng.io.write_csr_triplet(out, g)
    assert csr_files_equal(src, out)


def test_mosp_apply_equals_the_original(data: Path, tmp_path: Path, res: dyng.Resources) -> None:
    case = data / "mosp_graph_io" / "testCase0"
    g = dyng.io.read_csr_triplet(case / "graphCsr", properties="mosp_compatible", resources=res)
    b = dyng.io.read_legacy_batch(
        case / "insert.txt",
        case / "delete.txt",
        num_weights=g.num_weights,
        num_vertices=g.num_vertices,
    )
    g.apply(b)
    dyng.io.write_csr_triplet(tmp_path / "graphCsr", g)
    assert csr_files_equal(case / "applied" / "graphCsr", tmp_path / "graphCsr")


def test_legacy_batch_round_trip(data: Path, tmp_path: Path) -> None:
    case = data / "mosp_graph_io" / "testCase0"
    k = dyng.io.read_csr_triplet(case / "graphCsr").num_weights
    b = dyng.io.read_legacy_batch(case / "insert.txt", case / "delete.txt", num_weights=k)
    dyng.io.write_legacy_batch(tmp_path / "i.txt", tmp_path / "d.txt", b)
    assert same_bytes(case / "insert.txt", tmp_path / "i.txt")
    assert same_bytes(case / "delete.txt", tmp_path / "d.txt")
    b64 = dyng.io.read_legacy_batch(
        case / "insert.txt", case / "delete.txt", num_weights=k, vertex_dtype="int64"
    )
    assert b64.insert_src.dtype == np.int64


@pytest.mark.parametrize(
    ("mtx", "k", "wmax", "seed", "prefix"),
    [
        ("m1_general.mtx", 1, 100, 12345, "m1_k1_seed12345_"),
        ("m1_general.mtx", 4, 2**31 - 1, 7, "m1_k4_seed7_"),
    ],
)
def test_matrix_market_random_weights_equal_mosp_prep(
    data: Path, tmp_path: Path, mtx: str, k: int, wmax: int, seed: int, prefix: str
) -> None:
    d = data / "mosp_graph_io" / "mtx"
    g = dyng.io.read_matrix_market(
        d / mtx, num_weights=k, random_weights=(1, wmax, seed), properties="mosp_compatible"
    )
    assert g.num_weights == k
    dyng.io.write_csr_triplet(tmp_path / "g", g)
    assert csr_files_equal(d / prefix, tmp_path / "g")


def test_edge_list_equals_cycle_enum_parser(data: Path) -> None:
    f = data / "cycle_enum" / "parser" / "reference_sample.txt"
    g = dyng.io.read_edge_list(f, weighted=False, properties="cycle_enum_compatible")
    expected = dict(
        line.split(" ", 1) for line in (f.parent / (f.name + ".csr")).read_text().splitlines()
    )
    csr = g.to_csr()
    assert g.num_vertices == int(expected["n"]) and g.num_edges == int(expected["m"])
    assert " ".join(map(str, csr.row_ptr.tolist())) == expected["row_ptr"]
    assert " ".join(map(str, csr.col_ind.tolist())) == expected["col_ind"]
    arrays = dyng.io.read_edge_list_arrays(f)
    assert arrays.info is not None and arrays.info.external_ids.size == g.num_vertices
    assert arrays.weights is None and arrays.num_edges == g.num_edges


def test_edge_list_round_trip(tmp_path: Path) -> None:
    g = dyng.Graph.from_edges([0, 2, 1], [1, 0, 2], [[5, 6], [7, 8], [9, 1]])
    dyng.io.write_edge_list(tmp_path / "e.txt", g)
    back = dyng.io.read_edge_list(
        tmp_path / "e.txt", num_weights=2, ids="as_is", drop_self_loops=False
    )
    assert back.to_csr().col_ind.tolist() == g.to_csr().col_ind.tolist()
    assert back.to_csr().weights.tolist() == g.to_csr().weights.tolist()  # type: ignore[union-attr]
    dyng.io.write_edge_list(tmp_path / "t.txt", ([0, 1], [1, 0]))
    assert (tmp_path / "t.txt").read_text() == "0 1\n1 0\n"
    wide = dyng.io.read_edge_list_arrays(tmp_path / "t.txt", vertex_dtype="int64")
    assert wide.src.dtype == np.int64
    with pytest.raises(dyng.NotSupportedError):  # no unweighted graph with int64 ids
        dyng.io.read_edge_list(tmp_path / "t.txt", vertex_dtype="int64")


def test_matrix_market_round_trip(tmp_path: Path) -> None:
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0], [3, 4, 5])
    dyng.io.write_matrix_market(tmp_path / "g.mtx", g)
    text = (tmp_path / "g.mtx").read_text()
    assert text.startswith("%%MatrixMarket matrix coordinate integer general")
    back = dyng.io.read_matrix_market(tmp_path / "g.mtx")
    assert back.to_csr().weights[:, 0].tolist() == [3, 4, 5]  # type: ignore[index]
    pattern = dyng.io.read_matrix_market(tmp_path / "g.mtx", weights="none")
    assert pattern.num_weights == 0 and not pattern.weighted


def test_distances_and_parents(tmp_path: Path) -> None:
    g = dyng.Graph.from_edges([0], [1], [3], num_vertices=3)
    tree = dyng.sssp.compute(g, 0)
    dyng.io.write_distances(tmp_path / "d.txt", tree.distances)
    dyng.io.write_parents(tmp_path / "p.txt", tree.parents)
    assert (tmp_path / "d.txt").read_text() == "0 0\n1 3\n2 INF\n"
    assert (tmp_path / "p.txt").read_text() == "0 -1\n1 0\n2 -1\n"
    d = dyng.io.read_distances(tmp_path / "d.txt", 3)
    p = dyng.io.read_parents(tmp_path / "p.txt", 3)
    assert d.tolist() == tree.distances.tolist() and p.tolist() == [-1, 0, -1]
    assert dyng.io.read_parents(tmp_path / "p.txt", 3, vertex_dtype="int64").dtype == np.int64
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_distances(tmp_path / "d.txt", 2)


def test_histogram_csv(tmp_path: Path) -> None:
    g = dyng.Graph.from_edges([0, 1, 2, 1], [1, 2, 0, 0])
    h = dyng.cycle_count.compute(g)
    text = dyng.io.histogram_csv(h)
    assert text == "# cycle_size, num_of_cycles\n2, 1\n3, 1\nTotal, 2\n"
    assert dyng.io.histogram_csv(h.counts, include_total=False).endswith("3, 1\n")
    dyng.io.write_histogram_csv(tmp_path / "sub" / "h.csv", h)
    assert (tmp_path / "sub" / "h.csv").read_text() == text


def test_write_csr_triplet_needs_weights(tmp_path: Path) -> None:
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.io.write_csr_triplet(tmp_path / "x", dyng.Graph.from_edges([0], [1]))
