#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Reader of the H-SOSP dumps of MOSP_ESCHER@4b86159 (format 1, ADR 0081).

The dumps are written by the recording twins of parity/exporters/hsosp (the original's unchanged
objects relinked with -Xlinker --wrap; DYNG_HSOSP_DUMP_DIR). This module reads them (plain or
zstd-compressed), groups the records into sessions (one DynamicHypergraph each) and writes the
canonical encodings of ADR 0080 section 2:

  distances  int64 little-endian by dynG id (= B id - 1, = B's 0-based node index), dead or
             unreachable = INT64_MAX / 4 (B's HostHypergraph::INF);
  parents    int32 little-endian by dynG id, 0-based, -1 = none;
  pairs      the sorted (min, max) pairs of dynG ids, the deleted list then the inserted list, as
             int32 little-endian (a, b) values.

    parity/hsosp_dump.py summary <dump.hsd[.zst]>... [--json OUT]
    parity/hsosp_dump.py dump <dump.hsd[.zst]> [--arrays]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import escher_trace  # noqa: E402  (the field encoding is the CBST trace format's)

MAGIC = b"DYNGHSD1"
FORMAT_VERSION = 1
TAGS = {
    1: "header",
    2: "session",
    3: "load",
    4: "device_build",
    5: "begin_batch",
    6: "apply_batch",
    7: "sssp",
    8: "compare",
    9: "end",
}
TAG_IDS = {v: k for k, v in TAGS.items()}
INF = (2**63 - 1) // 4  # B's HostHypergraph::INF
# The record fields that vary from run to run (logged only, never hashed; HD7 and ADR 0081): the
# update's and recompute's iteration counts, work, frontier sizes and jump rounds.
LOGGED_FIELDS = (
    "stats.iterations",
    "stats.fallback_iterations",
    "stats.max_frontier",
    "stats.jump_rounds",
    "stats.work",
)
# The columns of hsospBench's CSV that do not vary from run to run (hsospBench.cu:155-164): every
# column but the timestamp, the times, the speedup, the device memory in use and the logged-only
# counters (iterations, frontier, work).
CSV_RUN_VARYING = (
    "timestamp",
    "t_escher_ms",
    "t_delta_ms",
    "t_csr_apply_ms",
    "t_sosp_update_ms",
    "t_dynamic_total_ms",
    "t_static_ms",
    "speedup",
    "iters",
    "max_frontier",
    "dev_mem_mb",
    "fallback_iters",
    "update_work",
    "static_iters",
)


def csv_deterministic(text: str) -> list[dict]:
    """The rows of an hsospBench CSV with the run-varying columns left out."""
    import csv

    return [
        {k: v for k, v in row.items() if k not in CSV_RUN_VARYING}
        for row in csv.DictReader(text.splitlines())
    ]


class DumpError(Exception):
    pass


def values(rec: dict, name: str) -> array:
    a = rec[name]
    if not isinstance(a, escher_trace.Array):
        raise DumpError(f"{name} is not an array")
    return a.values()


def rows_of(rec: dict, name: str) -> list[list[int]]:
    """The rows `<name>.offsets` / `<name>.values` of a record."""
    off, val = values(rec, name + ".offsets"), values(rec, name + ".values")
    return [list(val[off[i] : off[i + 1]]) for i in range(len(off) - 1)]


class Dump:
    """One dump file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = memoryview(escher_trace._open_bytes(self.path))
        if bytes(self.data[:8]) != MAGIC:
            raise DumpError(f"{self.path}: not an H-SOSP dump")
        (version,) = struct.unpack_from("<I", self.data, 8)
        if version != FORMAT_VERSION:
            raise DumpError(f"{self.path}: format {version}, this reader reads {FORMAT_VERSION}")

    def records(self):
        """(tag name, fields) of every record, in file order."""
        buf, pos, n = self.data, 16, len(self.data)
        while pos < n:
            if pos + 12 > n:
                raise DumpError(f"{self.path}: truncated record header at {pos}")
            tag, length = struct.unpack_from("<IQ", buf, pos)
            pos += 12
            if pos + length > n:
                raise DumpError(f"{self.path}: truncated record at {pos}")
            yield TAGS.get(tag, f"tag{tag}"), escher_trace.parse_fields(buf, pos, pos + length)
            pos += length


@dataclass
class Session:
    """The records of one DynamicHypergraph, in order."""

    id: int
    records: list[tuple[str, dict]] = field(default_factory=list)

    def of(self, kind: str) -> list[dict]:
        return [r for k, r in self.records if k == kind]


def sessions(d: Dump) -> tuple[dict, list[Session], dict | None, list[tuple[str, dict]]]:
    """(header, sessions in order, end record or None, records of no session)."""
    header, end, out, loose = {}, None, {}, []
    for kind, rec in d.records():
        if kind == "header":
            header = rec
        elif kind == "end":
            end = rec
        elif "session" in rec and rec["session"] > 0:
            out.setdefault(rec["session"], Session(rec["session"])).records.append((kind, rec))
        else:
            loose.append((kind, rec))
    return header, [out[k] for k in sorted(out)], end, loose


# --- Writing (the golden set stores rewritten dumps) ---------------------------------------------


def _field_bytes(name: str, value) -> bytes:
    nb = name.encode()
    head = lambda t: bytes([t]) + struct.pack("<H", len(nb)) + nb  # noqa: E731
    if isinstance(value, bool) or isinstance(value, int):
        return head(1) + struct.pack("<q", int(value))
    if isinstance(value, str):
        b = value.encode()
        return head(2) + struct.pack("<I", len(b)) + b
    if isinstance(value, list):
        out = head(4) + struct.pack("<I", len(value))
        for s in value:
            b = str(s).encode()
            out += struct.pack("<I", len(b)) + b
        return out
    if isinstance(value, escher_trace.Array):
        if value.kind != "inline":
            raise DumpError(f"{name}: only inline arrays are written")
        data = value.data
        if sys.byteorder != "little":
            data = array(data.typecode, data)
            data.byteswap()
        return head(3) + bytes([0, value.elem]) + struct.pack("<Q", value.count) + data.tobytes()
    raise DumpError(f"{name}: cannot write a {type(value).__name__}")


def encode_record(kind: str, rec: dict) -> bytes:
    """One record (tag, payload length, field list), the fields in `rec`'s order."""
    body = struct.pack("<I", len(rec)) + b"".join(_field_bytes(k, v) for k, v in rec.items())
    return struct.pack("<IQ", TAG_IDS[kind], len(body)) + body


def file_head() -> bytes:
    return MAGIC + struct.pack("<II", FORMAT_VERSION, 0)


def ints_array(values, elem: int = 4) -> escher_trace.Array:
    a = escher_trace.Array("inline", elem, 0)
    a.data = array("i" if elem == 4 else "q", values)
    a.count = len(a.data)
    return a


# --- Canonical encodings and digests --------------------------------------------------------------


def _canonical(values, code: str, fill: int, bound: int | None) -> bytes:
    a = values if isinstance(values, array) and values.typecode == code else array(code, values)
    n = len(a) if bound is None else bound
    if n != len(a):
        a = array(code, a[:n]) + array(code, [fill]) * max(0, n - len(a))
    if sys.byteorder != "little":
        a = array(code, a)
        a.byteswap()
    return a.tobytes()


def canonical_distances(dist: array | list[int], bound: int | None = None) -> bytes:
    """int64 little-endian by id over 0..bound - 1 (padded with INF; ADR 0080 section 2)."""
    return _canonical(dist, "q", INF, bound)


def canonical_parents(parent: array | list[int], bound: int | None = None) -> bytes:
    """int32 little-endian by id over 0..bound - 1, 0-based, -1 = none (padded with -1)."""
    return _canonical(parent, "i", -1, bound)


def normalize(arg: str) -> str:
    """An absolute path as the dumps record it: "<abs>/<last component>"."""
    return "<abs>/" + arg.rsplit("/", 1)[-1] if arg.startswith("/") else arg


def canonical_pairs(rec: dict) -> tuple[bytes, int, int]:
    """The pairs of an apply_batch record: (bytes, deleted, inserted); the deleted list first."""
    out = array("i")
    counts = []
    for kind in ("deleted", "inserted"):
        a, b = values(rec, f"pairs.{kind}.a"), values(rec, f"pairs.{kind}.b")
        pairs = sorted(zip(a, b, strict=True))
        if any(x >= y for x, y in pairs):
            raise DumpError(f"pairs.{kind}: a pair is not (min, max)")
        for x, y in pairs:
            out.append(x)
            out.append(y)
        counts.append(len(pairs))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes(), counts[0], counts[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- CLI ------------------------------------------------------------------------------------------


def summarize(path: Path) -> dict:
    d = Dump(path)
    header, ss, end, loose = sessions(d)
    out = {
        "file": path.name,
        "argv": header.get("argv"),
        "rows": header.get("rows"),
        "end": end.get("exit") if end else None,
        "loose_records": len(loose),
        "sessions": [],
    }
    for s in ss:
        kinds: dict[str, int] = {}
        for k, _ in s.records:
            kinds[k] = kinds.get(k, 0) + 1
        fallbacks = sum(r.get("stats.fallback_recompute", 0) for r in s.of("sssp"))
        out["sessions"].append({"session": s.id, "records": kinds, "fallbacks": fallbacks})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("dumps", nargs="+", type=Path)
    s.add_argument("--json", type=Path)
    dmp = sub.add_parser("dump")
    dmp.add_argument("dump", type=Path)
    dmp.add_argument("--arrays", action="store_true", help="print array values (first 16)")
    args = p.parse_args(argv)
    if args.cmd == "summary":
        res = [summarize(x) for x in args.dumps]
        text = json.dumps(res, indent=1)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        return 0
    for kind, rec in Dump(args.dump).records():
        items = []
        for k, v in rec.items():
            if isinstance(v, escher_trace.Array):
                items.append(
                    f"{k}=[{v.count}]"
                    + (f"{list(v.values()[:16])}" if args.arrays and v.kind == "inline" else "")
                )
            else:
                items.append(f"{k}={v}")
        print(kind, " ".join(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
