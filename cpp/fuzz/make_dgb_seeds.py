#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Write the seed corpus of the `.dgb` fuzzer (cpp/fuzz/corpus/batch_binary).

The files are written from the format's specification (docs/api/file_formats.md, ".dgb"), not
with the library's writer, so the seeds are also an independent check of the reader (the replay
of the corpus must accept the valid ones):

    python3 cpp/fuzz/make_dgb_seeds.py [--out cpp/fuzz/corpus/batch_binary]

Each seed is `<option line>\\n<file bytes>` (cpp/fuzz/fuzz_input.hpp): the option line's byte 0
selects the vertex count of the reader (0: unchecked, 1: 8, 2: 1000).
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

MAGIC = b"DYNGBAT1"
GRAPH, HYPERGRAPH = 0, 1
NONE, INT32, INT64 = 0, 1, 2
DELETED_MEMBERS, EXPECTED_IDS, INSERT_FIRST = 1, 2, 4


def header(kind: int, vtype: int, wtype: int, k: int, flags: int, count: int) -> bytes:
    id_type = INT64 if kind == HYPERGRAPH else NONE
    body = MAGIC + struct.pack("<II", 1, 64) + bytes([kind, vtype, wtype, id_type])
    body += struct.pack("<IQQ", k, flags, count) + bytes(20)
    return body + struct.pack("<I", zlib.crc32(body))


def array(values: list[int], dtype: int) -> bytes:
    fmt = "<" + ("i" if dtype == INT32 else "q") * len(values)
    data = struct.pack(fmt, *values)
    return data + bytes((-len(data)) % 8)


def record(payload: bytes) -> bytes:
    assert len(payload) % 8 == 0
    return struct.pack("<QII", len(payload), zlib.crc32(payload), 0) + payload


def graph_record(batch_id: int, dels, ins, weights, vtype=INT32, wtype=INT32) -> bytes:
    p = struct.pack("<QQQ", batch_id, len(dels), len(ins))
    p += array([u for u, _ in dels], vtype) + array([v for _, v in dels], vtype)
    p += array([u for u, _ in ins], vtype) + array([v for _, v in ins], vtype)
    if wtype != NONE:
        p += array(weights, wtype)
    return record(p)


def hyper_record(batch_id, dels, rows, weights=None, deleted_rows=None, expected=None) -> bytes:
    members = [v for r in rows for v in r]
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + len(r))
    deleted_members = [v for r in (deleted_rows or []) for v in r]
    p = struct.pack("<QQQQQ", batch_id, len(dels), len(rows), len(members), len(deleted_members))
    p += array(dels, INT64) + array(offsets, INT64) + array(members, INT32)
    if weights is not None:
        p += array(weights, INT32)
    if deleted_rows is not None:
        doff = [0]
        for r in deleted_rows:
            doff.append(doff[-1] + len(r))
        p += array(doff, INT64) + array(deleted_members, INT32)
    if expected is not None:
        p += array(expected, INT64)
    return record(p)


def seeds() -> dict[str, bytes]:
    out = {}
    out["graph_k1"] = (
        header(GRAPH, INT32, INT32, 1, 0, 2)
        + graph_record(0, [(2, 3)], [(0, 1), (1, 2)], [5, 6])
        + graph_record(1, [(4, 5)], [], [])
    )
    out["graph_unweighted_int64"] = header(GRAPH, INT64, NONE, 0, 0, 1) + graph_record(
        0, [], [(5000000000, 7)], [], vtype=INT64, wtype=NONE
    )
    out["hyper_weighted"] = header(HYPERGRAPH, INT32, INT32, 1, 0, 3) + (
        hyper_record(0, [3], [[4, 1, 1], [7]], weights=[6, 2147483647])
        + hyper_record(1, [0, 900000000000], [], weights=[])
        + hyper_record(2, [], [], weights=[])
    )
    out["hyper_annotated"] = header(
        HYPERGRAPH, INT32, NONE, 0, DELETED_MEMBERS | EXPECTED_IDS, 2
    ) + (
        hyper_record(0, [1, 5], [[2, 3], [0]], deleted_rows=[[1, 2], [6, 7, 7]], expected=[1, 5])
        + hyper_record(1, [], [[9]], deleted_rows=[], expected=[8])
    )
    out["hyper_insert_first"] = header(HYPERGRAPH, INT32, NONE, 0, INSERT_FIRST, 1) + hyper_record(
        0, [2], [[1, 2, 3]]
    )
    out["empty_file"] = header(HYPERGRAPH, INT32, NONE, 0, 0, 0)
    # Rejected inputs: a damaged CRC, a future version (the size-lying files are reproducers in
    # cpp/fuzz/regressions/batch_binary, written by --regressions).
    damaged = bytearray(out["graph_k1"])
    damaged[90] ^= 0x40
    out["crc_mismatch"] = bytes(damaged)
    future = bytearray(header(GRAPH, INT32, INT32, 1, 0, 0))
    struct.pack_into("<I", future, 8, 2)
    struct.pack_into("<I", future, 60, zlib.crc32(bytes(future[:60])))
    out["future_version"] = bytes(future)
    return out


def regressions() -> dict[str, bytes]:
    """Inputs whose size fields lie (PLAN 5.7: checked against the rest of the file before any
    allocation; M6b's readers allocated from such fields): a record length of 2^40 bytes, a batch
    count of 2^50, and an insertion count of 2^60 inside a record whose CRC matches."""
    out = {}
    lying = bytearray(seeds()["hyper_weighted"])
    struct.pack_into("<Q", lying, 64, 1 << 40)
    out["size_lying_record"] = bytes(lying)
    out["size_lying_count"] = header(GRAPH, INT32, INT32, 1, 0, 1 << 50)
    payload = bytearray(struct.pack("<QQQQQ", 0, 0, 1 << 60, 0, 0) + bytes(16))
    out["size_lying_insertions"] = header(HYPERGRAPH, INT32, NONE, 0, 0, 1) + record(bytes(payload))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    p.add_argument("--out", type=Path, default=here / "corpus" / "batch_binary")
    p.add_argument("--regressions", type=Path, default=here / "regressions" / "batch_binary")
    args = p.parse_args()
    options = {"hyper_annotated": b"2", "graph_k1": b"1"}
    for directory, files in ((args.out, seeds()), (args.regressions, regressions())):
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (directory / name).write_bytes(options.get(name, b"0") + b"\n" + data)
        print(f"{len(files)} inputs in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
