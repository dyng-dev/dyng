# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/dynlp_trace.py, the reader of the DynLP recording twins' traces (M10a step 1,
ADR 0065), on traces written here in the format of parity/exporters/dynlp/dynlp_trace.hpp: the
fields round-trip (floats as bits), the digest mode gives the SHA-256 of the raw bytes, and a
trace without its end record, with a wrong record count or truncated is refused."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace  # noqa: E402

FMT = {1: "i", 2: "q", 3: "Q", 4: "f", 5: "B"}


def field(name: str, dtype: int, values: list, digest: bool = False) -> bytes:
    raw = struct.pack(f"<{len(values)}{FMT[dtype]}", *values)
    head = struct.pack("<H", len(name)) + name.encode()
    if digest:
        return head + struct.pack("<BBQ", dtype, 1, len(values)) + hashlib.sha256(raw).digest()
    return head + struct.pack("<BBQ", dtype, 0, len(values)) + raw


def record(tag: str, batch: int, seq: int, fields: list[bytes]) -> bytes:
    return (struct.pack("<IH", 0x5250444C, len(tag)) + tag.encode()
            + struct.pack("<iiI", batch, seq, len(fields)) + b"".join(fields))  # fmt: skip


def trace(records: list[bytes], digest: bool = False, end: int | None = None) -> bytes:
    out = b"DYNGDLT1" + struct.pack("<II", 1, 1 if digest else 0) + b"".join(records)
    if end is not None:
        out += record("end", -1, 0, [field("records", 3, [end])])
    return out


def test_fields_round_trip(tmp_path: Path) -> None:
    recs = [
        record("apply_vertex_changes", 0, 0, [field("status", 1, [3, 0, 2]),
                                              field("gt_labels", 4, [1.0, 0.0])]),
        record("propagation_end", 0, 0, [field("iterations", 1, [7]),
                                         field("frontier_visits", 2, [12345678901]),
                                         field("labels", 4, [0.5, 0.25, 1.0])]),
        record("apply_vertex_changes", 1, 1, [field("status", 1, [1, 0, 2])]),
    ]  # fmt: skip
    p = tmp_path / "dynLP.dlt"
    p.write_bytes(trace(recs, end=3))
    t = dynlp_trace.read(p)
    assert [r.tag for r in t.all] == ["apply_vertex_changes", "propagation_end",
                                      "apply_vertex_changes"]  # fmt: skip
    b = t.batches()
    assert sorted(b) == [0, 1] and set(b[0]) == {"apply_vertex_changes", "propagation_end"}
    assert list(b[0]["apply_vertex_changes"]["status"].values()) == [3, 0, 2]
    assert b[0]["propagation_end"].scalar("frontier_visits") == 12345678901
    labels = b[0]["propagation_end"]["labels"]
    assert list(labels.values()) == [0.5, 0.25, 1.0]
    assert list(labels.bits()) == [0x3F000000, 0x3E800000, 0x3F800000]
    assert labels.sha256 == hashlib.sha256(struct.pack("<3f", 0.5, 0.25, 1.0)).hexdigest()


def test_digest_mode_gives_the_raw_digest(tmp_path: Path) -> None:
    p = tmp_path / "d.dlt"
    p.write_bytes(trace([record("append_edges", 0, 0, [field("row_end", 1, [1, 2, 3], True)])],
                        digest=True, end=1))  # fmt: skip
    t = dynlp_trace.read(p)
    f = t.one("append_edges")["row_end"]
    assert t.digest_mode and f.digest and f.count == 3
    assert f.sha256 == hashlib.sha256(struct.pack("<3i", 1, 2, 3)).hexdigest()
    with pytest.raises(ValueError, match="digest"):
        f.values()


def test_two_markers_in_one_batch_are_refused(tmp_path: Path) -> None:
    rec = record("apply_vertex_changes", 0, 0, [field("status", 1, [0])])
    p = tmp_path / "t.dlt"
    p.write_bytes(trace([rec, rec], end=2))
    with pytest.raises(ValueError, match="two 'apply_vertex_changes'"):
        dynlp_trace.read(p).batches()


@pytest.mark.parametrize("damage", ["no_end", "count", "truncated", "magic"])
def test_incomplete_traces_are_refused(tmp_path: Path, damage: str) -> None:
    recs = [record("run_end", -1, 0, [field("iterations", 1, [1, 2])])]
    data = {
        "no_end": trace(recs),
        "count": trace(recs, end=5),
        "truncated": trace(recs, end=1)[:-40],
        "magic": b"NOTATRCE" + trace(recs, end=1)[8:],
    }[damage]
    p = tmp_path / "bad.dlt"
    p.write_bytes(data)
    with pytest.raises(ValueError):
        dynlp_trace.read(p)
