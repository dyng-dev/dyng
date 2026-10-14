#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-check of the two ESCHER cores (PLAN 6.3 step 2; M8a criterion 6; ADR 0036).

    parity/escher_crosscheck.py run --set <golden set> --store ESCHER-GPU|MOSP_ESCHER
                                    [--trees H2V,V2H] [--capacity-floor N] [--counts scan|null]
                                    [--cases c1,c2] [--json OUT]
    parity/escher_crosscheck.py compare <source.cbt> <replay.cbt>

`run` replays every trace of a CBST trace golden set (parity/cbst_trace_goldens.py) through the
store of `--store` (parity_export/bin/cbst_crossreplay of its PATCHED copy: the original's
unchanged objects; its replay is itself recorded) and compares the replay with the source op by
op. With the source's own store the replay must be identical (the harness checks itself); with
the other original's store, the first divergence of every tree is reported with its op, the
field that differs and a classification by the documented behaviour differences of the two cores
(survey_hyper.md Section 4; PLAN 6.4.5):

  grows_vs_compacts   ESCHER-GPU grows the payload (realloc, offsets kept) where MOSP_ESCHER
                      compacts it (offsets rewritten) or throws at its fixed capacity
  b_validates         MOSP_ESCHER's construct validation throws where ESCHER-GPU continues
  caller_counts       MOSP_ESCHER's construct takes the caller's row counts (or occupancy 0 with
                      nullptr) where ESCHER-GPU scans the payload (initTailMetadata)
  compact_only_in_b   an explicit compact() of a MOSP_ESCHER trace (ESCHER-GPU has none)
  rebuild_only_in_a   ESCHER-GPU's rebuildFromDevice (MOSP_ESCHER has none): the tree stops
  fill_clamp          MOSP_ESCHER's fill clamps max(0, tailCapacity - occupancy)
  unexplained         none of these: must be investigated

Compared per op: exception type, InsertMapping, the key list, subtree_avail, and per key avail,
length, occupancy, tail capacity, offsets (value, tail base), segment count and the row in read
order; then the bump pointer (not the payload capacity, which the replay sets).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import escher_trace as et  # noqa: E402

NOTE = re.compile(r"^replay source_seq=(\d+) source_tree=(\d+) op=(\w+)(.*)$")
FIELDS = ("avail", "length", "occupancy", "tail_capacity", "value", "tail_base", "segments")
COL = {c: i for i, c in enumerate(et.KEY_COLUMNS)}


def scratch_default() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))


COMPACTION_COLUMNS = tuple(COL[c] for c in ("value", "tail_base", "tail_capacity", "segments"))


def apply_op(state: et.TreeState, op: dict) -> bool:
    """Applies an op's digest; True if the op compacted the payload (as escher_trace.py summary
    infers it: rows the op does not name changed their offsets, tails or chains, or the bump
    pointer fell)."""
    touched = list(op["key.key"].values()) if "key.key" in op else []
    old = {k: state.records[k] for k in touched if k in state.records}
    et.apply_digest(state, op, keep_rows=True)
    named: set = set()
    if op.get("op") in ("fill", "unfill", "erase") and et.known(op, "in.keys") is not None:
        named = set(et.known(op, "in.keys"))
    elif "out.mapping" in op:
        named = set(op["out.mapping"].values())
    moved = any(
        k not in named
        and k in state.records
        and any(old[k][c] != state.records[k][c] for c in COMPACTION_COLUMNS)
        for k in old
    )
    fell = op.get("after.initial_payload_size", 0) < op.get("before.initial_payload_size", 0)
    return moved or fell


def snapshot(state: et.TreeState, op: dict, compacted: bool = False) -> dict:
    return {
        "compacted": compacted,
        "what": op.get("exception.what", ""),
        "status": op.get("status", 0),
        "exception": op.get("exception.type"),
        "mapping": list(op["out.mapping"].values()) if "out.mapping" in op else None,
        "keys": state.keys,
        "subtree_avail": state.subtree_avail,
        "records": state.records,
        "rows": state.rows,
        "fixed_size": op.get("after.fixed_size"),
        "bump": op.get("after.initial_payload_size"),
        "bump_before": op.get("before.initial_payload_size"),
        "fixed_before": op.get("before.fixed_size"),
    }


def first_difference(a: dict, b: dict) -> tuple[str, str] | None:
    """(field, detail) of the first difference of two snapshots (a: source, b: replay)."""
    if a["status"] != b["status"] or a["exception"] != b["exception"]:
        return "exception", f"source {a['exception'] or 'none'}, replay {b['exception'] or 'none'}"
    if a["mapping"] != b["mapping"]:
        i = next(
            (
                i
                for i, (x, y) in enumerate(
                    zip(a["mapping"] or [], b["mapping"] or [], strict=False)
                )
                if x != y
            ),
            None,
        )
        return (
            "mapping",
            f"item {i}: source {a['mapping'][i] if i is not None else a['mapping']}"
            f", replay {b['mapping'][i] if i is not None else b['mapping']}",
        )
    if a["keys"] != b["keys"]:
        sa, sb = set(a["keys"]), set(b["keys"])
        return "keys", f"only in source {sorted(sa - sb)[:5]}, only in replay {sorted(sb - sa)[:5]}"
    if a["subtree_avail"] != b["subtree_avail"]:
        i = next(
            i
            for i, (x, y) in enumerate(zip(a["subtree_avail"], b["subtree_avail"], strict=False))
            if x != y
        )
        return (
            "subtree_avail",
            f"position {i}: source {a['subtree_avail'][i]}, replay {b['subtree_avail'][i]}",
        )
    for key in a["keys"]:
        ra, rb = a["records"][key], b["records"][key]
        for f in ("avail", "length", "occupancy", "tail_capacity", "segments"):
            if ra[COL[f]] != rb[COL[f]]:
                return f, f"key {key}: source {ra[COL[f]]}, replay {rb[COL[f]]}"
        rowa, rowb = a["rows"].get(key), b["rows"].get(key)
        if rowa is not None and rowb is not None and rowa != rowb:
            kind = "row order" if sorted(rowa) == sorted(rowb) else "row"
            return kind, f"key {key}: source {rowa[:8]}, replay {rowb[:8]}"
        for f in ("value", "tail_base"):
            if ra[COL[f]] != rb[COL[f]]:
                return f, f"key {key}: source {ra[COL[f]]}, replay {rb[COL[f]]}"
    if a["bump"] != b["bump"]:
        return "bump", f"source {a['bump']}, replay {b['bump']}"
    # The payload capacity is not compared: the replay sets it (--capacity-floor), and growth or
    # compaction shows in the offsets and the bump pointer.
    return None


def classify(field_: str, op: str, a: dict, b: dict, source_repo: str, note: str) -> str:
    grew_a = (a.get("fixed_size") or 0) > (a.get("fixed_before") or 0)
    grew_b = (b.get("fixed_size") or 0) > (b.get("fixed_before") or 0)
    fell_a = (a.get("bump") or 0) < (a.get("bump_before") or 0)
    fell_b = (b.get("bump") or 0) < (b.get("bump_before") or 0)
    if "rebuildFromDevice" in note or op == "rebuild":
        return "rebuild_only_in_a"
    if op == "compact":
        return "compact_only_in_b"
    if op == "construct":
        if field_ == "exception":
            return "b_validates"
        if field_ == "occupancy":
            return "caller_counts"
    capacity = ("payload overflow", "compacted rows need", "Increase payloadCapacity")
    if field_ == "exception" and any(
        m in a.get("what", "") or m in b.get("what", "") for m in capacity
    ):
        return "grows_vs_compacts"  # B throws at its fixed capacity where A grows
    if a.get("compacted") or b.get("compacted"):
        return "grows_vs_compacts"  # B compacted (offsets, tails, chains rewritten); A did not
    if (grew_a or grew_b) and (fell_a or fell_b):
        return "grows_vs_compacts"
    if field_ in ("value", "tail_base", "bump", "fixed_size") and (
        grew_a or grew_b or fell_a or fell_b
    ):
        return "grows_vs_compacts"
    if field_ in ("occupancy", "row", "row order") and op == "fill":
        return "fill_clamp"
    return "unexplained"


def replay_steps(replay: Path):
    """(source seq, note, replay state, replay op) of every replayed op, in order; a note that
    skips or stops an op yields (seq, note, None, None)."""
    rtrees: dict[int, et.TreeState] = {}
    pending = None
    for kind, rec in et.Trace(replay).records():
        if kind == "note":
            m = NOTE.match(rec.get("text", ""))
            if m:
                note = m.group(4).strip()
                if note.startswith(("skipped", "stop")):
                    yield int(m.group(1)), note, None, None, False
                else:
                    pending = (int(m.group(1)), note)
            continue
        if kind != "op":
            continue
        st = rtrees.setdefault(rec["tree"], et.TreeState())
        compacted = apply_op(st, rec)
        if pending is not None:
            yield pending[0], pending[1], st, rec, compacted
            pending = None


def compare(source: Path, replay: Path, source_repo: str, trees_filter: set | None = None) -> dict:
    """First divergence per source tree of a replay (both traces streamed in step)."""
    steps = replay_steps(replay)
    step = next(steps, None)
    trees: dict[int, et.TreeState] = {}
    names: dict[int, str] = {}
    result: dict = {}
    for kind, rec in et.Trace(source).records():
        if kind == "tree":
            names[rec["tree"]] = rec["dataset_name"]
            continue
        if kind != "op":
            continue
        tid = rec["tree"]
        if trees_filter and names.get(tid) not in trees_filter:
            continue
        st = trees.setdefault(tid, et.TreeState())
        compacted = apply_op(st, rec)
        entry = result.setdefault(
            tid, {"tree": names.get(tid, "?"), "ops": 0, "compared": 0, "first_divergence": None}
        )
        entry["ops"] += 1
        seq = rec["seq"]
        while step is not None and step[0] < seq:
            step = next(steps, None)
        here = step if step is not None and step[0] == seq else None
        # Compare before the replay advances (its next op updates the same tree state).
        if entry["first_divergence"] is None:
            if here is None or here[2] is None:
                note = here[1] if here else "no replay of this op"
                entry["first_divergence"] = {
                    "op_index": entry["ops"] - 1,
                    "seq": seq,
                    "op": rec["op"],
                    "field": "not replayed",
                    "detail": note,
                    "class": classify("not replayed", rec["op"], {}, {}, source_repo, note),
                }
            else:
                a, b = snapshot(st, rec, compacted), snapshot(here[2], here[3], here[4])
                diff = first_difference(a, b)
                entry["compared"] += 1
                if diff:
                    entry["first_divergence"] = {
                        "op_index": entry["ops"] - 1,
                        "seq": seq,
                        "op": rec["op"],
                        "field": diff[0],
                        "detail": diff[1],
                        "class": classify(diff[0], rec["op"], a, b, source_repo, here[1]),
                    }
        if here is not None:
            step = next(steps, None)
    return {str(k): v for k, v in result.items()}


def driver(store: str, scratch: Path) -> Path:
    out = subprocess.check_output(
        [
            str(REPO / "parity" / "build_reference.sh"),
            "--variant",
            "patched",
            "--scratch",
            str(scratch),
            "--print-dir",
            store,
        ],
        text=True,
    )
    exe = Path(out.strip().splitlines()[-1]) / "parity_export" / "bin" / "cbst_crossreplay"
    if not exe.is_file():
        raise SystemExit(f"{exe} is missing (parity/build_reference.sh --variant patched {store})")
    return exe


def run_main(args) -> int:
    sys.path.insert(0, str(REPO / "parity"))
    import cbst_trace_goldens as g

    src_dir = g.set_dir(args.scratch, args.set)
    source_repo = g.SETS[args.set]["reference"]
    exe = driver(args.store, args.scratch)
    cases = sorted(p.parent for p in src_dir.glob("*/trace.cbt.zst"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.name in wanted]
    record = {
        "set": args.set,
        "source": source_repo,
        "store": args.store,
        "trees": args.trees,
        "capacity_floor": args.capacity_floor,
        "counts": args.counts,
        "cases": {},
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYNG_CBST_")}
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env["DYNG_CBST_TRACE_MODE"] = "full" if args.full else "auto"
    unexplained = 0
    for case in cases:
        with tempfile.TemporaryDirectory(dir=args.scratch / "runs") as t:
            t = Path(t)
            src = t / "source.cbt"
            subprocess.run(
                [
                    "zstd",
                    "-q",
                    "-d",
                    "--long=31",
                    "-f",
                    str(case / "trace.cbt.zst"),
                    "-o",
                    str(src),
                ],
                check=True,
            )
            if (case / "trace.blob.zst").is_file():
                subprocess.run(
                    [
                        "zstd",
                        "-q",
                        "-d",
                        "--long=31",
                        "-f",
                        str(case / "trace.blob.zst"),
                        "-o",
                        str(t / "source.blob"),
                    ],
                    check=True,
                )
            env["DYNG_CBST_TRACE_DIR"] = str(t / "replay")
            cmd = [
                str(exe),
                str(src),
                "--capacity-floor",
                str(args.capacity_floor),
                "--counts",
                args.counts,
            ]
            if args.trees:
                cmd += ["--trees", args.trees]
            p = subprocess.run(cmd, cwd=t, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                record["cases"][case.name] = {"error": (p.stdout + p.stderr)[-2000:]}
                unexplained += 1
                print(f"{case.name}: replay failed: {(p.stdout + p.stderr)[-300:]}")
                continue
            replay = t / "replay" / "000001.cbt"
            res = compare(
                src, replay, source_repo, set(args.trees.split(",")) if args.trees else None
            )
            record["cases"][case.name] = res
            for tree in res.values():
                d = tree["first_divergence"]
                if d and d["class"] == "unexplained":
                    unexplained += 1
                print(
                    f"{case.name} {tree['tree']}: {tree['compared']}/{tree['ops']} ops compared;"
                    + (
                        f" first divergence at op {d['op_index']} ({d['op']}, seq {d['seq']}):"
                        f" {d['field']}: {d['detail']} [{d['class']}]"
                        if d
                        else " identical"
                    ),
                    flush=True,
                )
    record["unexplained"] = unexplained
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(condense(record), indent=1) + "\n")
    return 1 if unexplained else 0


def condense(record: dict) -> dict:
    """The record of a run without the identical trees: per tree name the counts by class, and
    every first divergence (case, tree, op, field, detail, class)."""
    by_name: dict = {}
    divergences = []
    for case, trees in sorted(record["cases"].items()):
        if "error" in trees:
            divergences.append({"case": case, "error": trees["error"][-300:]})
            continue
        for t in trees.values():
            d = t["first_divergence"]
            cls = d["class"] if d else "identical"
            counts = by_name.setdefault(t["tree"], {})
            counts[cls] = counts.get(cls, 0) + 1
            if d:
                divergences.append(
                    {
                        "case": case,
                        "tree": t["tree"],
                        "ops": t["ops"],
                        "compared": t["compared"],
                        **d,
                    }
                )
    out = {k: v for k, v in record.items() if k != "cases"}
    out["cases"] = len(record["cases"])
    out["trees_by_name"] = by_name
    out["divergences"] = divergences
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--set", required=True)
    a.add_argument("--store", required=True, choices=["ESCHER-GPU", "MOSP_ESCHER"])
    a.add_argument("--trees", default="")
    a.add_argument("--capacity-floor", type=int, default=0)
    a.add_argument("--counts", default="scan", choices=["scan", "null"])
    a.add_argument("--cases", default="")
    a.add_argument("--full", action="store_true", help="full digests in the replay")
    a.add_argument("--scratch", type=Path, default=scratch_default())
    a.add_argument("--json")
    a.add_argument("--summary-json", help="the condensed record (parity/results)")
    c = sub.add_parser("compare")
    c.add_argument("source")
    c.add_argument("replay")
    c.add_argument("--source-repo", default="")
    d = sub.add_parser("condense", help="condense the JSON record of a run")
    d.add_argument("record")
    d.add_argument("out")
    args = p.parse_args(argv)
    if args.cmd == "condense":
        rec = json.loads(Path(args.record).read_text())
        Path(args.out).write_text(json.dumps(condense(rec), indent=1) + "\n")
        return 0
    if args.cmd == "compare":
        print(json.dumps(compare(Path(args.source), Path(args.replay), args.source_repo), indent=1))
        return 0
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
