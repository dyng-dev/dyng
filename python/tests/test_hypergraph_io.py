# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.io for hypergraphs: .hg files, the hyperedge batches of .dgt, and .dgb binary batches."""

from __future__ import annotations

from pathlib import Path

import dyng
import numpy as np
import pytest


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -- .hg ----------------------------------------------------------------------------------------


def test_read_hypergraph_option_sets(tmp_path: Path) -> None:
    f = _write(tmp_path / "h.hg", "# a comment\n3 1 2 2\n\n5,4\n1 2 3\n9 8 7 6 5 4\n")
    h = dyng.io.read_hypergraph(f)  # escher_gpu()'s settings: ids kept (minus 1), every line
    assert (h.num_vertices, h.num_hyperedges) == (9, 4)
    lst, info = dyng.io.read_hypergraph_arrays(f, options="escher_gpu")
    assert lst.to_lists() == [[0, 1, 2], [3, 4], [0, 1, 2], [3, 4, 5, 6, 7, 8]]
    assert (info.lines, info.repeated_members) == (4, 1)
    lst, info = dyng.io.read_hypergraph_arrays(
        f, dedupe_hyperedges=True, max_cardinality=3, relabel="first_appearance"
    )
    assert lst.to_lists() == [[0, 1, 2], [3, 4]]
    assert info.external_ids.tolist() == [1, 2, 3, 4, 5]
    assert (info.over_cardinality, info.repeated_hyperedges) == (1, 1)
    w, _ = dyng.io.read_hypergraph_arrays(f, options="hsosp")
    assert w.weights is not None and w.weights.dtype == np.int32
    assert ((w.weights >= 1) & (w.weights <= 100)).all()
    again, _ = dyng.io.read_hypergraph_arrays(
        f, options=dyng.io.HypergraphReadOptions.hsosp(seed=20260725, max_cardinality=25)
    )
    assert np.array_equal(again.weights, w.weights)  # bit for bit for a seed
    r, _ = dyng.io.read_hypergraph_arrays(f, random_weights=(5, 6, 1))
    assert set(r.weights.tolist()) <= {5, 6}  # type: ignore[union-attr]


def test_read_hypergraph_properties(tmp_path: Path) -> None:
    f = _write(tmp_path / "h.hg", "1 2 3\n2 4\n")
    h = dyng.io.read_hypergraph(
        f,
        line_graph="sorted_merge",
        id_reuse="insert_first",
        capacity="mosp_escher",
        max_cardinality=25,
        random_weights=(1, 100, 7),
    )
    p = h.properties
    assert (p.line_graph, p.reuse) == ("sorted_merge", "insert_first")
    assert h.weighted and h.h2h().num_rows == 2
    with pytest.raises(dyng.NotSupportedError, match="0.4"):
        dyng.io.read_hypergraph(f, storage="slack_csr")


def test_hg_round_trip(tmp_path: Path) -> None:
    h = dyng.Hypergraph.from_lists([[0, 2], [1], [2, 3, 4]], [3, 4, 5])
    h.apply(dyng.HyperedgeBatch([1], [[4, 0]], [6]))
    dyng.io.write_hypergraph(tmp_path / "w.hg", h)
    back = dyng.io.read_hypergraph(tmp_path / "w.hg", weighted=True)
    assert back.to_hyperedge_list().to_lists() == h.to_hyperedge_list().to_lists()
    assert back.weights.tolist() == [3, 6, 5]  # type: ignore[union-attr]
    dyng.io.write_hypergraph(tmp_path / "z.hg", h.to_hyperedge_list(), index_base=0)
    assert (tmp_path / "z.hg").read_text().splitlines()[0] == "0 2 ; 3"


@pytest.mark.parametrize(
    ("text", "line", "column"),
    [("1 2\n3 x\n", 2, 3), ("1 2\n\n-4\n", 3, 1), ("1 2 ; 4\n", 1, 5)],
)
def test_malformed_hg_names_path_line_and_column(
    tmp_path: Path, text: str, line: int, column: int, capfd: pytest.CaptureFixture[str]
) -> None:
    f = _write(tmp_path / "bad.hg", text)
    with pytest.raises(dyng.FileFormatError) as e:
        dyng.io.read_hypergraph(f)
    assert e.value.path == str(f) and e.value.line == line and e.value.column == column
    out, err = capfd.readouterr()
    assert out == "" and err == ""  # readers never print
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_hypergraph(tmp_path / "missing.hg")


# -- .dgt ---------------------------------------------------------------------------------------


def test_dgt_hyperedge_batches_round_trip(tmp_path: Path) -> None:
    batches = [
        dyng.HyperedgeBatch([0, 5], [[1, 2], [3, 3, 0]]),
        dyng.HyperedgeBatch(),
        dyng.HyperedgeBatch(delete_hyperedges=[2**40]),
    ]
    dyng.io.write_hyperedge_batches(tmp_path / "b.dgt", batches)
    text = (tmp_path / "b.dgt").read_text()
    assert text.startswith("%dgt 1 hypergraph\n") and "+h 3 3 0" in text and "-h 5" in text
    back = dyng.io.read_hyperedge_batches(tmp_path / "b.dgt")
    assert [b.delete_hyperedges.tolist() for b in back] == [[0, 5], [], [2**40]]
    assert [b.insertions() for b in back] == [[[1, 2], [3, 3, 0]], [], []]
    weighted = [dyng.HyperedgeBatch([], [[0, 1]], [9])]
    dyng.io.write_hyperedge_batches(tmp_path / "w.dgt", weighted)
    back = dyng.io.read_hyperedge_batches(tmp_path / "w.dgt", weighted=True)
    assert back[0].insert_weights.tolist() == [9]  # type: ignore[union-attr]


def test_dgt_kinds_are_not_mixed(tmp_path: Path) -> None:
    f = _write(tmp_path / "m.dgt", "%batch 0\n+h 0 1\n+e 0 1\n")
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_hyperedge_batches(f)
    g = _write(tmp_path / "g.dgt", "+e 0 1 1\n")
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_hyperedge_batches(g)
    with pytest.raises(dyng.FileFormatError):
        dyng.io.read_batches(f)
    i = _write(tmp_path / "i.dgt", "+i 0 1\n")
    with pytest.raises(dyng.FileFormatError, match="0.4"):
        dyng.io.read_hyperedge_batches(i)


# -- .dgb ---------------------------------------------------------------------------------------


def test_dgb_hyperedge_batches_with_annotations(tmp_path: Path) -> None:
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2], [2, 3]], id_reuse="insert_first")
    batches = [dyng.HyperedgeBatch([1], [[0, 3], [1]]), dyng.HyperedgeBatch([0], [])]
    path = tmp_path / "b.dgb"
    clone = h.clone()
    ids = [clone.apply(b).inserted_ids for b in batches]
    dyng.io.write_binary_batches(
        path, batches, deleted_members=[[[1, 2]], [[0, 1]]], expected_ids=ids, insert_first=True
    )
    assert path.read_bytes()[:8] == b"DYNGBAT1"
    reader = dyng.io.BatchReader(path, num_vertices=4)
    hd = reader.header
    assert (hd.format_version, hd.kind, hd.vertex_type, hd.weight_type, hd.num_batches) == (
        1,
        "hypergraph",
        "int32",
        "none",
        2,
    )
    assert hd.deleted_members and hd.expected_ids and hd.insert_first
    records = list(reader)
    assert [r.batch_id for r in records] == [0, 1]
    for r in records:
        assert isinstance(r.batch, dyng.HyperedgeBatch)
        s = h.apply(r.batch)
        assert r.expected_ids is not None and np.array_equal(s.inserted_ids, r.expected_ids)
    o, m = records[0].deleted_members  # type: ignore[misc]
    assert o.tolist() == [0, 2] and m.tolist() == [1, 2]
    assert [b.num_insertions for b in dyng.io.read_binary_batches(path)] == [2, 0]


def test_dgb_edge_batches_round_trip(tmp_path: Path) -> None:
    batches = [
        dyng.EdgeBatch(insert=([0, 1], [1, 2], [[3, 4], [5, 6]]), delete=([2], [0])),
        dyng.EdgeBatch(delete=([1], [2])),
    ]
    dyng.io.write_binary_batches(tmp_path / "e.dgb", batches)
    reader = dyng.io.BatchReader(tmp_path / "e.dgb")
    assert (reader.header.kind, reader.header.num_weights) == ("graph", 2)
    back = [r.batch for r in reader]
    assert all(isinstance(b, dyng.EdgeBatch) for b in back)
    assert back[0].insert_weights.tolist() == [[3, 4], [5, 6]]  # type: ignore[union-attr]
    assert back[1].delete_src.tolist() == [1]
    dyng.io.write_binary_batches(tmp_path / "u.dgb", [dyng.EdgeBatch(insert=([0], [3]))])
    assert dyng.io.BatchReader(tmp_path / "u.dgb").header.weight_type == "none"
    with pytest.raises(TypeError, match="one kind"):
        dyng.io.write_binary_batches(tmp_path / "x.dgb", [batches[0], dyng.HyperedgeBatch()])


def test_malformed_dgb_raises_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "b.dgb"
    dyng.io.write_binary_batches(path, [dyng.HyperedgeBatch([0], [[1, 2]])])
    data = path.read_bytes()
    for name, bad in (
        ("truncated", data[: len(data) - 3]),
        ("crc", data[:-1] + bytes([data[-1] ^ 0xFF])),
        ("magic", b"DYNGBATX" + data[8:]),
        ("short", data[:20]),
    ):
        f = tmp_path / f"{name}.dgb"
        f.write_bytes(bad)
        with pytest.raises(dyng.FileFormatError):
            list(dyng.io.BatchReader(f))


# -- the M8b review ------------------------------------------------------------------------------


def test_dgb_reader_stops_at_its_first_error(tmp_path: Path) -> None:
    # A damaged first record of three: a retry raises again instead of returning record 1.
    path = tmp_path / "b.dgb"
    dyng.io.write_binary_batches(
        path, [dyng.HyperedgeBatch([0], [[1, 2]]), dyng.HyperedgeBatch([1], [[3]])] * 2
    )
    data = bytearray(path.read_bytes())
    data[64 + 16 + 4] ^= 0x10  # a payload byte of record 0
    path.write_bytes(bytes(data))
    reader = dyng.io.BatchReader(path)
    with pytest.raises(dyng.FileFormatError, match="CRC32"):
        next(reader)
    for _ in range(2):
        with pytest.raises(dyng.FileFormatError, match="failed at batch record 0"):
            next(reader)


def test_a_closed_dgb_reader_raises(tmp_path: Path) -> None:
    path = tmp_path / "b.dgb"
    dyng.io.write_binary_batches(path, [dyng.HyperedgeBatch([0], [[1, 2]])])
    reader = dyng.io.BatchReader(path)
    with reader:
        pass
    with pytest.raises(dyng.InvalidArgumentError, match="closed"):
        next(reader)


def test_one_dgb_reader_shared_by_two_threads(tmp_path: Path) -> None:
    # The native reader serializes next() and the copy of its batch (the GIL is released while a
    # record is read): every batch is read once, intact.
    import threading

    path = tmp_path / "many.dgb"
    batches = [dyng.HyperedgeBatch([i], [[i % 50, i % 50 + 1]] * 20) for i in range(300)]
    dyng.io.write_binary_batches(path, batches)
    for _ in range(5):
        reader = dyng.io.BatchReader(path)
        got: list[int] = []
        errors: list[BaseException] = []

        def drain(r: dyng.io.BatchReader = reader) -> None:
            try:
                for rec in r:
                    assert rec.batch.num_insertions == 20
                    got.append(rec.batch_id)
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=drain) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert sorted(got) == list(range(300))


def test_a_directory_is_a_file_format_error(tmp_path: Path) -> None:
    with pytest.raises(dyng.FileFormatError, match="directory"):
        dyng.io.read_hypergraph(tmp_path)
    with pytest.raises(dyng.FileFormatError):
        dyng.io.BatchReader(tmp_path)


@pytest.mark.parametrize("value", [2**63, 2**64, 2**70, -(2**63) - 1])
def test_reader_options_beyond_int64_are_invalid_arguments(tmp_path: Path, value: int) -> None:
    path = tmp_path / "h.hg"
    path.write_text("1 2\n")
    with pytest.raises(dyng.InvalidArgumentError, match="outside"):
        dyng.io.read_hypergraph(path, index_base=value)
    with pytest.raises(dyng.InvalidArgumentError, match="outside"):
        dyng.io.read_hypergraph(path, num_vertices=value)
    with pytest.raises(dyng.InvalidArgumentError, match="outside"):
        dyng.io.BatchReader(path, num_vertices=value)
