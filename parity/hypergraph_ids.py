#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Container-level id parity of dyng::hypergraph (M8b, criterion 4; ADR 0041, section 11).

    parity/compare.py hypergraph_ids --set <name> --backend <sequential|openmp|cuda>
                                     [--threads N] [--mode by_id|by_content|both]
                                     [--record DIR] [--jobs N] [--gpu N] [--exe PATH]

replays the batch records of M8a's CBST trace goldens ($DYNG_SCRATCH/goldens/cbst_trace/<set>,
parity/goldens.toml) through the hypergraph container with the tool dyng_hypergraph_ids
(cpp/tests/parity/escher/hypergraph_ids_main.cpp): ESCHER-GPU@abcf9b2's sets under {erase_first,
escher_gpu()} by id and by content, MOSP_ESCHER@4b86159's hyperedge-kind sessions under
{insert_first, mosp_escher(extra)}; every inserted id, h2v row, v2h row and skipped count is
compared with the original's, with check_integrity() and the CBST host model in lockstep after
every batch. The set `committed` is the committed traces and batch extracts of
cpp/tests/data/goldens (no $DYNG_SCRATCH needed).

--line-graph (M8b step 2, criteria 7 and 8; ESCHER-GPU's sets): the by-id container keeps the
line graph (line_graph_policy::sorted_merge); after the build and every batch its H2H layout must
equal the input of ESCHER-GPU's H2H construct or rebuild (offsets and payload, by their recorded
hashes), every key's row hash must equal the state digest's where the trace has one (the full rows
too where the digest records them), and its device bytes must be within 1.05x of A's
DynamicHypergraph::deviceBytes(). The record is then M8b-h2h-<set>-<backend>[-t<N>].json.

--legacy-generator (M8b step 4, criterion 16; ESCHER-GPU's sets): generators::legacy draws A's
synthetic hypergraph and, before every batch, the batch from the by-id container's live state with
one std::mt19937_64(--seed of the run); each must equal A's. The record is then
M8b-legacy-<set>-<backend>[-t<N>].json. --hg FILE (criterion 13): the .hg file the runs loaded is
read with the original's option set and must give its loaded rows (record M8b-hg-<set>-...).

--record writes M8b-hypergraph-ids-<set>-<backend>[-t<N>].json (the totals, every case's report
and the tool's commit). Exit status: 0 every case matched, 1 a mismatch, 2 a usage error, 77 no
goldens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cbst_trace_goldens  # noqa: E402

# The sets whose batch records the replay reads: A's (build / update records) and B's
# hyperedge-kind sets (bulk_load / begin_batch records; its vertex-kind set is M11's).
SETS = {
    "cbst_trace_escher_gpu_test": "ESCHER-GPU",
    "cbst_trace_escher_gpu_coauth50k": "ESCHER-GPU",
    "cbst_trace_escher_gpu_syn1m": "ESCHER-GPU",
    "cbst_trace_mosp_escher_test": "MOSP_ESCHER",
    "cbst_trace_mosp_escher_dblp50k_hyperedge": "MOSP_ESCHER",
    "committed": "both",
}
COMMITTED = [
    "cpp/tests/data/goldens/cbst_trace/escher_gpu_test/001-main.cbt",
    "cpp/tests/data/goldens/hypergraph_batches/escher_gpu_016-main.cbt",
    "cpp/tests/data/goldens/hypergraph_batches/mosp_escher_014-test_h2h_delta.cbt",
]
TOTALS = (
    "sessions",
    "batches",
    "vertex_batches",
    "ids_compared",
    "rows_compared",
    "filtered_virtual",
    "skipped_compared",
    "skipped_ops",
    "dropped_members",
    "ambiguous",
    "content_states",
    "content_states_equal",
    "generated_graphs",
    "generated_batches",
    "generated_batches_equal",
    "hg_hyperedges",
    "original_exceptions",
    "h2h_states",
    "h2h_layouts_equal",
    "h2h_unrecorded",
    "h2h_payload_entries",
    "h2h_key_hashes",
    "h2h_full_rows",
    "mismatch_count",
)


def full_name(name: str) -> str:
    return name if name in SETS or name.startswith("cbst_trace_") else "cbst_trace_" + name


def find_exe(cuda: bool) -> Path | None:
    """The tool: $DYNG_HYPERGRAPH_IDS, else the first build tree that has it (with `cuda`, only
    the CUDA builds)."""
    env = os.environ.get("DYNG_HYPERGRAPH_IDS")
    if env:
        return Path(env)
    presets = (
        ("release-cuda", "parity-cuda", "dev-cuda")
        if cuda
        else ("release", "cpu-only", "parity", "dev", "release-cuda", "parity-cuda", "dev-cuda")
    )
    for preset in presets:
        exe = REPO / "build" / preset / "cpp" / "tests" / "dyng_hypergraph_ids"
        if exe.is_file():
            return exe
    return None


def cases_of(name: str, scratch: Path) -> list[Path]:
    if name == "committed":
        return [REPO / c for c in COMMITTED]
    return sorted(cbst_trace_goldens.set_dir(scratch, name).glob("*/trace.cbt.zst"))


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_case(exe: Path, trace: Path, args, tmp: Path) -> dict:
    out = tmp / (trace.parent.name + "-" + trace.name + ".json")
    cmd = [str(exe), "--backend", args.backend, "--mode", args.mode, "--json", str(out)]
    cmd += ["--tmp", str(tmp)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if args.no_model:
        cmd.append("--no-model")
    if args.line_graph:
        cmd.append("--line-graph")
    if args.legacy_generator:
        cmd.append("--legacy-generator")
    if args.hg:
        cmd += ["--hg", str(args.hg)]
    cmd.append(str(trace))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1) or not out.is_file():
        return {
            "trace": cbst_trace_goldens.portable(trace),
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip()[-2000:],
            "mismatch_count": 1,
        }
    report = json.loads(out.read_text())[0]
    report["trace"] = cbst_trace_goldens.portable(trace)
    return report


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="compare.py hypergraph_ids",
        description="Container-level id parity of dyng::hypergraph against the batch records of "
        "both originals' CBST traces (M8b, criterion 4).",
    )
    p.add_argument("--set", action="append", required=True, help="a set (repeatable)")
    p.add_argument("--backend", choices=["sequential", "openmp", "cuda"], required=True)
    p.add_argument("--threads", type=int, default=0, help="openmp threads (0: the default)")
    p.add_argument("--mode", choices=["by_id", "by_content", "both"], default="both")
    p.add_argument("--no-model", action="store_true", help="without the host-model lockstep")
    p.add_argument(
        "--line-graph",
        action="store_true",
        help="keep the line graph and compare it (and the device bytes) with ESCHER-GPU's",
    )
    p.add_argument(
        "--legacy-generator",
        action="store_true",
        help="draw A's hypergraph and batches with generators::legacy and compare them",
    )
    p.add_argument("--hg", type=Path, help="the .hg file the runs loaded (read and compared)")
    p.add_argument("--scratch", type=Path, default=cbst_trace_goldens.scratch_default())
    p.add_argument("--exe", type=Path, help="dyng_hypergraph_ids (default: find it in build/)")
    p.add_argument("--jobs", type=int, default=1, help="cases replayed at once")
    p.add_argument("--gpu", type=int, default=1, help="CUDA_VISIBLE_DEVICES unless set (default 1)")
    p.add_argument("--record", type=Path, help="write M8b-hypergraph-ids-<set>-<backend>.json here")
    p.add_argument("--skip-verify", action="store_true", help="do not re-hash the goldens")
    args = p.parse_args(argv)
    names = [full_name(n) for n in args.set]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        print(f"unknown sets {unknown}; the sets are {', '.join(SETS)}", file=sys.stderr)
        return 2
    if args.backend == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    exe = args.exe or find_exe(cuda=args.backend == "cuda")
    if exe is None or not exe.is_file():
        print(
            "dyng_hypergraph_ids not found: build it (cmake --build <build> --target "
            "dyng_hypergraph_ids_tool) or pass --exe",
            file=sys.stderr,
        )
        return 2
    missing = [
        n
        for n in names
        if n != "committed"
        and not (cbst_trace_goldens.set_dir(args.scratch, n) / "MANIFEST.sha256").is_file()
    ]
    if missing and len(missing) == len(names):
        print(f"no goldens for {missing} under {args.scratch}; skipping", file=sys.stderr)
        return 77
    bad = 0
    runs = args.scratch / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for n in names:
        problems: list[str] = []
        if n != "committed" and not args.skip_verify:
            problems += cbst_trace_goldens.verify_set(n, args.scratch)
        cases = cases_of(n, args.scratch)
        with tempfile.TemporaryDirectory(dir=runs) as t:
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                reports = list(pool.map(lambda c: run_case(exe, c, args, Path(t)), cases))
        totals = {k: sum(int(r.get(k, 0)) for r in reports) for k in TOTALS}
        totals["memory_ratio_max"] = max(
            (float(r.get("memory_ratio_max", 0)) for r in reports), default=0.0
        )
        if args.line_graph and SETS[n] == "ESCHER-GPU":
            compared = totals["h2h_states"] - totals["h2h_unrecorded"]
            if totals["h2h_states"] == 0 or totals["h2h_layouts_equal"] != compared:
                problems.append(
                    f"h2h: {totals['h2h_layouts_equal']} of {totals['h2h_states']} states equal "
                    "ESCHER-GPU's H2H"
                )
        if args.legacy_generator and SETS[n] == "ESCHER-GPU":
            if totals["generated_batches_equal"] != totals["generated_batches"] or (
                totals["generated_batches"] != totals["batches"]
            ):
                problems.append(
                    f"legacy generator: {totals['generated_batches_equal']} of "
                    f"{totals['generated_batches']} drawn batches equal A's "
                    f"({totals['batches']} batches)"
                )
        if args.hg and totals["hg_hyperedges"] == 0:
            problems.append(f"--hg {args.hg}: no hyperedge was compared")
        for r in reports:
            if not r.get("ok", False):
                first = r.get("error") or (r.get("mismatches") or ["?"])[0]
                problems.append(f"{r['trace']}: {r.get('mismatch_count', 1)} mismatches ({first})")
        record = {
            "set": n,
            "reference": SETS[n],
            "backend": args.backend,
            "threads": args.threads,
            "mode": args.mode,
            "model": not args.no_model,
            "line_graph": args.line_graph,
            "legacy_generator": args.legacy_generator,
            "hg": cbst_trace_goldens.portable(args.hg) if args.hg else None,
            "commit": git_head(),
            "tool": cbst_trace_goldens.portable(exe),
            "passed": not problems,
            "totals": totals,
            "cases": reports,
            "problems": problems,
        }
        print(
            f"{n} on {args.backend}"
            + (f" ({args.threads} threads)" if args.threads else "")
            + f": {len(reports)} traces, {totals['sessions']} sessions, {totals['batches']} "
            f"batches, {totals['ids_compared']} ids, {totals['rows_compared']} rows, "
            f"{totals['vertex_batches']} vertex-kind batches not replayed, "
            f"{totals['filtered_virtual']} virtual ops filtered, {totals['ambiguous']} ambiguous "
            f"by-content resolutions, "
            + (
                f"h2h {totals['h2h_layouts_equal']}/{totals['h2h_states']} layouts "
                f"({totals['h2h_unrecorded']} without A's input) "
                f"({totals['h2h_payload_entries']} entries), {totals['h2h_key_hashes']} key "
                f"hashes, {totals['h2h_full_rows']} full rows, memory ratio <= "
                f"{totals['memory_ratio_max']:.4f}, "
                if args.line_graph
                else ""
            )
            + (
                f"generator: {totals['generated_graphs']} hypergraphs, "
                f"{totals['generated_batches_equal']}/{totals['generated_batches']} batches equal, "
                if args.legacy_generator
                else ""
            )
            + (f"{totals['hg_hyperedges']} .hg hyperedges compared, " if args.hg else "")
            + (
                f"by content {totals['content_states_equal']}/{totals['content_states']} states "
                "with A's live sets, "
                if args.mode != "by_id"
                else ""
            )
            + f"{totals['mismatch_count']} mismatches: "
            + ("OK" if not problems else "FAILED")
        )
        for pr in problems[:20]:
            print("  " + pr)
        if args.record:
            args.record.mkdir(parents=True, exist_ok=True)
            short = n[len("cbst_trace_") :] if n.startswith("cbst_trace_") else n
            kind = (
                "h2h"
                if args.line_graph
                else "legacy"
                if args.legacy_generator
                else "hg"
                if args.hg
                else "hypergraph-ids"
            )
            name = f"M8b-{kind}-{short}-{args.backend}"
            if args.threads:
                name += f"-t{args.threads}"
            (args.record / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(compare_main(sys.argv[1:]))
