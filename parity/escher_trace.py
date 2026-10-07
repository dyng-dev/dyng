#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Reader of the CBST traces of ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159 (format 1, ADR 0036).

The traces are written by the recording twins of parity/exporters/escher (the originals'
unchanged objects relinked with -Xlinker --wrap, PLAN 6.4.5 and 8.3). This module reads them
(plain or zstd-compressed, with their side file of large input arrays) and rebuilds the state of
every tree after every operation from the state digests.

    parity/escher_trace.py summary <trace.cbt[.zst]>... [--json OUT]
    parity/escher_trace.py dump <trace.cbt[.zst]> [--records N] [--arrays]
    parity/escher_trace.py verify <trace.cbt[.zst]>...

summary reports per trace: the operations per tree and type; inserts with best-fit reuse, with
surplus items and with a surplus rebuild (a new key list), K = 0 and D = 0; fills with overflow
segments and with payload growth (A) or compaction (inferred from the state: the bump pointer
fell, or a row the op did not name moved); explicit compactions; the kernel levels (thread /
warp / block) the inputs select for fill, reuse and unfill (and A's recorded launch counts);
unfills of chained rows; deleted keys reissued by the surplus path; exceptions; the longest
chain; the original's integrity verdicts; unrecorded changes between operations.

verify rebuilds every digest (delta digests from the previous state of the tree) and checks
every recorded hash (rows, keys, per-key records, the whole state): the reader and the recorder
agree on the state after every operation.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"DYNGCBT1"
BLOB_MAGIC = b"DYNGCBB1"
FORMAT_VERSION = 1
TAGS = {1: "header", 2: "tree", 3: "op", 4: "event", 5: "batch", 6: "note", 7: "end"}
ARRAY_KINDS = {0: "inline", 1: "blob", 2: "hash", 3: "null"}
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = (1 << 64) - 1
# Per-key record columns of a digest, in the order of the record hash.
KEY_COLUMNS = (
    "key",
    "avail",
    "length",
    "size",
    "occupancy",
    "tail_capacity",
    "value",
    "tail_base",
    "segments",
    "row_len",
    "terminated",
)
COL_VALUE = KEY_COLUMNS.index("value")
# The columns a compaction rewrites in rows an op does not name.
COMPACTION_COLUMNS = tuple(
    KEY_COLUMNS.index(c) for c in ("value", "tail_base", "tail_capacity", "segments")
)


class TraceError(Exception):
    pass


@dataclass
class Array:
    """An array field: inline data, a reference into the side file, a hash only, or null."""

    kind: str
    elem: int
    count: int
    data: array | None = None
    offset: int = 0
    hash: int | None = None

    def values(self, trace: Trace | None = None) -> array:
        if self.kind == "inline":
            return self.data
        if self.kind == "blob":
            if trace is None:
                raise TraceError("a side-file array needs its trace")
            return trace.blob(self.offset, self.elem, self.count)
        if self.kind == "null":
            return array("i")
        raise TraceError("hash-only array has no values")


def fnv1a(data: bytes, h: int = FNV_OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK
    return h


def fnv_u64(h: int, v: int) -> int:
    return fnv1a(struct.pack("<Q", v & MASK), h)


def _open_bytes(path: Path) -> bytes:
    if path.suffix == ".zst":
        return subprocess.run(
            ["zstd", "-dc", "--long=31", str(path)], check=True, stdout=subprocess.PIPE
        ).stdout
    return path.read_bytes()


def parse_fields(buf: memoryview, pos: int, end: int) -> dict:
    (count,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    out: dict = {}
    for _ in range(count):
        ftype = buf[pos]
        (nlen,) = struct.unpack_from("<H", buf, pos + 1)
        name = bytes(buf[pos + 3 : pos + 3 + nlen]).decode()
        pos += 3 + nlen
        if ftype == 1:
            (v,) = struct.unpack_from("<q", buf, pos)
            pos += 8
            out[name] = v
        elif ftype == 2:
            (n,) = struct.unpack_from("<I", buf, pos)
            out[name] = bytes(buf[pos + 4 : pos + 4 + n]).decode(errors="replace")
            pos += 4 + n
        elif ftype == 3:
            kind, elem = buf[pos], buf[pos + 1]
            (n,) = struct.unpack_from("<Q", buf, pos + 2)
            pos += 10
            a = Array(ARRAY_KINDS.get(kind, "?"), elem, n)
            if kind == 0:
                data = array("i" if elem == 4 else "q")
                data.frombytes(bytes(buf[pos : pos + n * elem]))
                if sys.byteorder != "little":
                    data.byteswap()
                a.data = data
                pos += n * elem
            elif kind == 1:
                a.offset, a.hash = struct.unpack_from("<QQ", buf, pos)
                pos += 16
            elif kind == 2:
                (a.hash,) = struct.unpack_from("<Q", buf, pos)
                pos += 8
            out[name] = a
        elif ftype == 4:
            (n,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            items = []
            for _ in range(n):
                (sl,) = struct.unpack_from("<I", buf, pos)
                items.append(bytes(buf[pos + 4 : pos + 4 + sl]).decode(errors="replace"))
                pos += 4 + sl
            out[name] = items
        else:
            raise TraceError(f"unknown field type {ftype} of {name}")
        if pos > end:
            raise TraceError("field list overruns its record")
    return out


class Trace:
    """One trace file (and its side file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = memoryview(_open_bytes(self.path))
        if bytes(self.data[:8]) != MAGIC:
            raise TraceError(f"{self.path}: not a CBST trace")
        (version,) = struct.unpack_from("<I", self.data, 8)
        if version != FORMAT_VERSION:
            raise TraceError(f"{self.path}: format {version}, this reader reads {FORMAT_VERSION}")
        self._blob: bytes | None = None

    def records(self):
        """(tag name, fields) of every record, in file order."""
        buf, pos, n = self.data, 16, len(self.data)
        while pos < n:
            if pos + 12 > n:
                raise TraceError(f"{self.path}: truncated record header at {pos}")
            tag, length = struct.unpack_from("<IQ", buf, pos)
            pos += 12
            if pos + length > n:
                raise TraceError(f"{self.path}: truncated record at {pos}")
            yield TAGS.get(tag, f"tag{tag}"), parse_fields(buf, pos, pos + length)
            pos += length

    def blob(self, offset: int, elem: int, count: int) -> array:
        if self._blob is None:
            stem = str(self.path)
            stem = stem[: -len(".cbt.zst")] if stem.endswith(".cbt.zst") else stem[: -len(".cbt")]
            for cand in (Path(stem + ".blob"), Path(stem + ".blob.zst")):
                if cand.is_file():
                    self._blob = _open_bytes(cand)
                    break
            else:
                raise TraceError(f"{self.path}: side file missing")
            if self._blob[:8] != BLOB_MAGIC:
                raise TraceError(f"{self.path}: bad side file")
        out = array("i" if elem == 4 else "q")
        out.frombytes(self._blob[offset : offset + elem * count])
        return out


# --- State of the trees -------------------------------------------------------------------------


@dataclass
class TreeState:
    """The state of one tree as the digests describe it (rows only where recorded)."""

    name: str = ""
    keys: list[int] = field(default_factory=list)
    subtree_avail: list[int] = field(default_factory=list)
    records: dict[int, tuple] = field(default_factory=dict)  # key -> KEY_COLUMNS values
    rows: dict[int, list[int] | None] = field(default_factory=dict)  # None: hash-only
    row_hash: dict[int, int] = field(default_factory=dict)
    scalars: dict[str, int] = field(default_factory=dict)


def record_hash(rec: tuple, row_hash: int) -> int:
    return fnv_u64(fnv1a(struct.pack("<11i", *rec)), row_hash)


def apply_digest(state: TreeState, op: dict, keep_rows: bool) -> list[int]:
    """Updates `state` from the digest of `op`; returns the keys written in full."""
    if "digest.mode" not in op:
        return []
    full = op["digest.mode"] == 0
    if "state.keys" in op:
        state.keys = list(op["state.keys"].values())
    if "state.subtree_avail" in op:
        state.subtree_avail = list(op["state.subtree_avail"].values())
    cols = [op[f"key.{c}"].values() for c in KEY_COLUMNS]
    touched = list(cols[0])
    hashes = op["key.row_hash"].values()
    rows_field = op["key.rows"]
    rows = rows_field.values() if keep_rows and rows_field.kind == "inline" else None
    if full:
        state.records, state.rows, state.row_hash = {}, {}, {}
    at = 0
    for i, key in enumerate(touched):
        rec = tuple(c[i] for c in cols)
        state.records[key] = rec
        state.row_hash[key] = hashes[i] & MASK
        n = rec[9]
        if keep_rows:
            state.rows[key] = list(rows[at : at + n]) if rows is not None else None
        at += n
    live = set(state.keys)
    for key in [k for k in state.records if k not in live]:
        del state.records[key]
        state.rows.pop(key, None)
        state.row_hash.pop(key, None)
    for name in ("fixed_size", "num_records", "initial_payload_size", "alignment"):
        if f"after.{name}" in op:
            state.scalars[name] = op[f"after.{name}"]
    return touched


def verify_op(state: TreeState, op: dict) -> list[str]:
    """Recomputes the hashes of a digest from the rebuilt state."""
    problems = []
    if "digest.mode" not in op:
        return problems
    keys = array("i", state.keys)
    if fnv1a(keys.tobytes()) != op["state.keys_hash"] & MASK:
        problems.append("keys hash")
    if fnv1a(array("i", state.subtree_avail).tobytes()) != op["state.subtree_avail_hash"] & MASK:
        problems.append("subtree_avail hash")
    for key in state.keys:
        if key not in state.records:
            problems.append(f"key {key}: no record")
            return problems
    for key, row in state.rows.items():
        if row is not None and fnv1a(array("i", row).tobytes()) != state.row_hash[key]:
            problems.append(f"key {key}: row hash")
    if "state.key_hashes" in op:
        want = op["state.key_hashes"].values()
        for i, key in enumerate(state.keys):
            if record_hash(state.records[key], state.row_hash[key]) != want[i] & MASK:
                problems.append(f"key {key}: record hash")
                break
    h = FNV_OFFSET
    for name in ("num_records", "fixed_size", "initial_payload_size", "alignment"):
        h = fnv_u64(h, state.scalars.get(name, 0))
    h = fnv_u64(h, op["state.keys_hash"])
    h = fnv_u64(h, op["state.subtree_avail_hash"])
    for key in state.keys:
        h = fnv_u64(h, record_hash(state.records[key], state.row_hash[key]))
    if h != op["state.hash"] & MASK:
        problems.append("state hash")
    return problems


# --- summary ------------------------------------------------------------------------------------


def level(size: int) -> int:
    """The degree bin of an item (both originals' DegreeBins: < 32 thread, < 1024 warp)."""
    return 0 if size < 32 else (1 if size < 1024 else 2)


def known(rec: dict, name: str):
    """The values of an array field, or None when it is absent or recorded as a hash only (the
    ops of a batches-mode trace)."""
    a = rec.get(name)
    return a.values() if isinstance(a, Array) and a.kind in ("inline", "null") else None


def sizes_of(prefix) -> list[int]:
    out, prev = [], 0
    for p in prefix:
        out.append(p - prev)
        prev = p
    return out


def summarize(path: Path) -> dict:
    trace = Trace(path)
    trees: dict[int, TreeState] = {}
    s: dict = {
        "trace": path.name,
        "ops": {},
        "insert": {
            "total": 0,
            "reuse": 0,
            "surplus": 0,
            "rebuild": 0,
            "k0": 0,
            "d0": 0,
            "reuse_only": 0,
            "reuse_surplus_rebuild": 0,
            "reissued_dead_key": 0,
        },
        "fill": {"total": 0, "overflow": 0, "growth": 0, "overflow_growth": 0, "compaction": 0},
        "capacity_throws": 0,
        "insert_compaction": 0,
        "compact_calls": 0,
        "compaction_ops": [],
        "unfill": {"total": 0, "chained": 0, "chained_fallback": 0},
        "levels": {"fill": [0, 0, 0], "reuse": [0, 0, 0], "unfill": [0, 0, 0]},
        "recorded_launches": {"fill": [0, 0, 0], "reuse": [0, 0, 0], "unfill": [0, 0, 0]},
        "exceptions": [],
        "max_segments": 1,
        "integrity_failures": 0,
        "unrecorded_changes": 0,
        "batches": 0,
        "hash_only": [],
        "exit": None,
        "argv": [],
    }
    for kind, rec in trace.records():
        if kind == "header":
            s["argv"] = rec.get("argv", [])
            s["repo"] = rec.get("repo")
            s["mode"] = rec.get("mode")
        elif kind == "tree":
            trees[rec["tree"]] = TreeState(name=rec["dataset_name"])
        elif kind == "batch" and rec.get("phase") == "exit":
            s["batches"] += rec["hook"].endswith(("update", "begin_batch"))
        elif kind == "end":
            s["exit"] = rec.get("exit")
        elif kind == "op":
            tree = trees.setdefault(rec["tree"], TreeState(name="?"))
            op = rec["op"]
            key = f"{tree.name}:{op}"
            s["ops"][key] = s["ops"].get(key, 0) + 1
            before = dict(tree.records)
            for name, v in rec.items():
                if isinstance(v, Array) and v.kind == "hash" and name not in s["hash_only"]:
                    s["hash_only"].append(name)
            if rec.get("pre.matches") == 0:
                s["unrecorded_changes"] += 1
            if rec.get("status") == 1 and re.search(
                r"payload overflow|exceed|capacity", rec.get("exception.what", "")
            ):
                s["capacity_throws"] += 1
            if rec.get("status") == 1:
                s["exceptions"].append(
                    {
                        "seq": rec["seq"],
                        "op": key,
                        "type": rec.get("exception.type"),
                        "what": rec.get("exception.what", "")[:200],
                    }
                )
            for which in ("fill", "reuse", "unfill"):
                if f"bins.{which}" in rec:
                    for i, x in enumerate(rec[f"bins.{which}"].values()):
                        s["recorded_launches"][which][i] += x
            touched = apply_digest(tree, rec, keep_rows=False)
            s["max_segments"] = max(s["max_segments"], rec.get("rec.max_segments", 1))
            if rec.get("check.rows", 0) > 0 or rec.get("check.subtree_avail", 0) > 0:
                s["integrity_failures"] += 1
            named = set()
            if "in.keys" in rec and op in ("fill", "unfill", "erase"):
                named = set(known(rec, "in.keys") or [])
            # A compaction (B) rewrites the rows the op does not name (offsets, tail, chain) and
            # resets the bump pointer to the compacted size.
            if op == "insert" and "out.mapping" in rec:
                named = set(rec["out.mapping"].values())
            moved = any(
                k in before
                and k not in named
                and k in tree.records
                and any(before[k][c] != tree.records[k][c] for c in COMPACTION_COLUMNS)
                for k in touched
            )
            fell = rec.get("after.initial_payload_size", 0) < rec.get(
                "before.initial_payload_size", 0
            )
            if op == "insert":
                s["insert"]["total"] += 1
                mapping = list(rec["out.mapping"].values()) if "out.mapping" in rec else []
                d = rec.get("before.d", -1)
                s["insert"]["k0"] += len(mapping) == 0
                s["insert"]["d0"] += d == 0 and len(mapping) > 0
                dead_before = {k for k, r in before.items() if r[1] == 1}
                # Reuse writes the item into the dead slot (same offset); a surplus item is
                # appended behind the bump pointer, also when its key is a dead key that the
                # surplus rebuild dropped and issued again.
                reused = [
                    k
                    for k in mapping
                    if k in dead_before
                    and k in tree.records
                    and tree.records[k][COL_VALUE] == before[k][COL_VALUE]
                ]
                fresh = [k for k in mapping if k not in before]
                if rec.get("status", 0) == 0:
                    s["insert"]["reuse"] += bool(reused)
                    s["insert"]["surplus"] += bool(fresh) or len(reused) < len(mapping)
                    s["insert"]["reuse_only"] += bool(reused) and len(reused) == len(mapping)
                    # Both originals rebuild the tree from its live nodes whenever an item is
                    # surplus (the key list can stay the same: a reissued dead key).
                    rebuilt = len(reused) < len(mapping)
                    s["insert"]["rebuild"] += rebuilt
                    s["insert"]["reuse_surplus_rebuild"] += (
                        bool(reused) and len(reused) < len(mapping) and rebuilt
                    )
                    # a dead key the surplus path issued again (it was dropped by the rebuild)
                    surplus_keys = [k for k in mapping if k not in reused]
                    reissued = [k for k in surplus_keys if k in dead_before]
                    s["insert"]["reissued_dead_key"] += bool(reissued)
                    prefix = known(rec, "in.prefix")
                    sizes = sizes_of(prefix) if prefix is not None else []
                    for i, k in enumerate(mapping):
                        if k in reused and i < len(sizes):
                            s["levels"]["reuse"][level(sizes[i])] += 1
                if moved or fell:
                    s["insert_compaction"] += 1
                    s["compaction_ops"].append({"seq": rec["seq"], "op": key, "inside": "insert"})
            elif op == "fill":
                s["fill"]["total"] += 1
                for sz in sizes_of(known(rec, "in.prefix") or []):
                    s["levels"]["fill"][level(sz)] += 1
                grew = rec.get("after.fixed_size", 0) > rec.get("before.fixed_size", 0)
                s["fill"]["growth"] += grew
                overflow = any(
                    k in before and tree.records.get(k, before[k])[8] > before[k][8]
                    for k in touched
                ) or rec.get("after.initial_payload_size", 0) > rec.get(
                    "before.initial_payload_size", 0
                )
                s["fill"]["overflow"] += overflow
                s["fill"]["overflow_growth"] += overflow and grew
                if moved or fell:
                    s["fill"]["compaction"] += 1
                    s["compaction_ops"].append({"seq": rec["seq"], "op": key, "inside": "fill"})
            elif op == "unfill":
                s["unfill"]["total"] += 1
                for sz in sizes_of(known(rec, "in.prefix") or []):
                    s["levels"]["unfill"][level(sz)] += 1
                keys = known(rec, "in.keys") or []
                sizes = sizes_of(known(rec, "in.prefix") or [])
                if any(before.get(k, (0,) * 11)[8] > 1 for k in keys):
                    s["unfill"]["chained"] += 1
                # warp / block items on chained rows run the serial fallback (unfillRowSerial)
                if any(
                    before.get(k, (0,) * 11)[8] > 1 and level(sizes[i]) > 0
                    for i, k in enumerate(keys)
                ):
                    s["unfill"]["chained_fallback"] += 1
            elif op == "compact":
                s["compact_calls"] += 1
                s["compaction_ops"].append({"seq": rec["seq"], "op": key, "inside": "compact"})
    s["trees"] = {str(i): t.name for i, t in trees.items()}
    s["max_chain_hops"] = s["max_segments"] - 1
    return s


def cmd_summary(args) -> int:
    out = []
    for p in args.traces:
        s = summarize(Path(p))
        out.append(s)
        if not args.json:
            print(f"== {s['trace']} ({s.get('repo')}, {s.get('mode')}): {' '.join(s['argv'])}")
            print("  ops: " + ", ".join(f"{k}={v}" for k, v in sorted(s["ops"].items())))
            print(f"  insert: {s['insert']}")
            print(
                f"  fill: {s['fill']}  insert compactions: {s['insert_compaction']}"
                f"  compact(): {s['compact_calls']}"
            )
            print(f"  unfill: {s['unfill']}")
            print(f"  levels (from the inputs, thread/warp/block): {s['levels']}")
            print(f"  recorded launches (A): {s['recorded_launches']}")
            print(
                f"  max chain hops: {s['max_chain_hops']}  integrity failures: "
                f"{s['integrity_failures']}  unrecorded changes: {s['unrecorded_changes']}"
                f"  exit: {s['exit']}"
            )
            for e in s["exceptions"]:
                print(f"  exception at seq {e['seq']} ({e['op']}): {e['type']}: {e['what']}")
            if s["hash_only"]:
                print(f"  hash-only arrays: {', '.join(s['hash_only'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def cmd_dump(args) -> int:
    trace = Trace(args.trace)
    for i, (kind, rec) in enumerate(trace.records()):
        if args.records and i >= args.records:
            break
        shown = {}
        for k, v in rec.items():
            if isinstance(v, Array):
                if args.arrays and v.kind == "inline":
                    shown[k] = list(v.data)
                else:
                    shown[k] = f"<{v.kind} {v.count}x{v.elem}B>"
            else:
                shown[k] = v
        print(kind, json.dumps(shown))
    return 0


def cmd_verify(args) -> int:
    bad = 0
    for p in args.traces:
        trace = Trace(p)
        trees: dict[int, TreeState] = {}
        ops = 0
        ended = failed = False
        for kind, rec in trace.records():
            if kind == "end":
                ended = True
            if kind != "op":
                continue
            ops += 1
            tree = trees.setdefault(rec["tree"], TreeState())
            apply_digest(tree, rec, keep_rows=True)
            problems = verify_op(tree, rec)
            if problems and not failed:
                failed = True
                print(f"{p}: op seq {rec['seq']} ({rec['op']}): {', '.join(problems[:5])}")
        if not ended:
            failed = True
            print(f"{p}: no END record (the process did not finish)")
        bad += failed
        print(f"{p}: {ops} ops verified" if not failed else f"{p}: FAILED")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("summary", help="what a trace covers")
    a.add_argument("traces", nargs="+")
    a.add_argument("--json", help="write the summaries as JSON")
    a.set_defaults(func=cmd_summary)
    a = sub.add_parser("dump", help="print the records")
    a.add_argument("trace")
    a.add_argument("--records", type=int, default=0)
    a.add_argument("--arrays", action="store_true", help="print inline arrays")
    a.set_defaults(func=cmd_dump)
    a = sub.add_parser("verify", help="rebuild every state and check every recorded hash")
    a.add_argument("traces", nargs="+")
    a.set_defaults(func=cmd_verify)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
