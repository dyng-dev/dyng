#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The replays of label_propagation's legacy generator and oracles against
LabelPropagation-CUDA@a276a3a's (O's) goldens (M10a step 2; ADRs 0067, 0068; the author's decisions
AD7, AD8 and AD12 of 2026-10-28).

    compare.py label_propagation generator --exe EXE [--threads 1,8,56] [--cases ...] [--json F]
    compare.py label_propagation model --exe EXE [--cases ...] [--json F]
    compare.py label_propagation check-runs --gpu 1 [--copy DIR] [--cases ...]   (O's --check)
    compare.py label_propagation oracle --exe EXE [--threads 8] [--json F]
    compare.py label_propagation margin --exe EXE [--json F]
    compare.py label_propagation subset    (writes cpp/tests/data/goldens/label_propagation)

EXE is dyng_lp_replay (cpp/tests/parity/label_propagation/lp_replay_main.cpp) of a build.

generator  For every case of the corpus G (label_propagation_goldens.GEN_CASES): the generator's
           arrays against the golden's recorded ones (raw: byte for byte; 5m, 5m1, 50m: the
           SHA-256 the recorder's digest mode stored), and for the bytes cases O's four
           graph/csr*.txt files against the library's writers; once per thread count. The
           arrays' digests are combined into one stream digest per case, which must be the same
           for every thread count and every build given (--exe may be repeated).
model      For every case of P+ (P_CASES): lp_host_model fed O's own recorded batches (5m: the
           generator's, whose digests are checked first) against every deterministic field of
           the golden trace (statuses mapped by name, ADR 0068), and the generator's tau against
           the recorded one.
check-runs O's recording twin (a patched copy built with the step-2 recorder: --wrap=snprintf
           records the arguments of its label-oracle line) with --check on C = {5k5, 50k} at
           delta 1e-6 (--validate at 5k5) and L = {500k, 5m} at delta 1e-4, both rules,
           OMP_NUM_THREADS=8, GPU 1, recording only propagation_end and check_labels
           (DYNG_DYNLP_TRACE_TAGS). The traces go to $DYNG_SCRATCH/runs/m10a-oracles-generator-
           writers/check/<case>/ (deleted by `oracle --clean`).
oracle     For every check run: harmonic_solution on O's labels of every batch against O's own
           recorded mean |F - F*| (to 1e-9 absolute or 1e-6 relative), the compared and excluded
           counts and the skip verdicts.
margin     The class margin m (AD7): the largest |F* - 0.5| of a vertex whose class (>= 0.5f)
           differs from F*'s, over O's delta 1e-6 runs: damped per batch (the twin's label bits of
           the damped golden) and in_place final labels (step 1's runs, label_propagation_inplace/
           tight). It also reports the in_place per-batch values of the check runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace as dlt  # noqa: E402
import label_propagation_goldens as g  # noqa: E402

STEP = "m10a-oracles-generator-writers"
CHECK_TAGS = "propagation_end,check_labels"
C_CASES = [
    g.Case(s, delta="1e-6", schedule=r) for s in ("5k5", "50k") for r in ("damped", "inplace")
]
L_CASES = [
    g.Case(s, delta="1e-4", schedule=r) for s in ("500k", "5m") for r in ("damped", "inplace")
]
SUBSET_DIR = REPO / "cpp" / "tests" / "data" / "goldens" / "label_propagation"
# AD7: the per-batch comparison of harmonic_solution with O's checker.
ABS_TOL = 1e-9
REL_TOL = 1e-6


def runs_dir() -> Path:
    d = g.scratch() / "runs" / STEP
    d.mkdir(parents=True, exist_ok=True)
    return d


def tool(exe: Path, args: list) -> tuple[int, dict]:
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if p.returncode not in (0, 1):
        raise g.GoldenError(f"{exe.name} {args[0]} failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.returncode, json.loads(p.stdout)


def config_args(case: g.Case) -> list:
    return [
        "--nodes",
        case.n,
        "--batches",
        case.batches,
        "--degree",
        case.avg_degree,
        "--seed",
        case.seed,
    ]


def decompressed(src: Path, tmp: Path) -> Path:
    out = tmp / (src.name[:-4] if src.name.endswith(".zst") else src.name)
    out.write_bytes(g.zstd_read(src))
    return out


def stream_digest(digests: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(digests):
        h.update(f"{k} {digests[k]}\n".encode())
    return h.hexdigest()


def select(cases: list, names: list | None) -> list:
    if not names:
        return cases
    known = {c.name: c for c in cases}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise g.GoldenError(f"unknown cases {unknown} (known: {sorted(known)})")
    return [known[n] for n in names]


# --- generator -----------------------------------------------------------------------------------


def cmd_generator(args: argparse.Namespace) -> int:
    root = g.set_dir(g.scratch() / "goldens", "generator")
    results = []
    ok = True
    for case in select(g.GEN_CASES, args.cases):
        gold = root / "cases" / case.name
        with tempfile.TemporaryDirectory(prefix="lp-gen-", dir=runs_dir()) as t:
            tmp = Path(t)
            golden = decompressed(gold / "generator.dlt.zst", tmp)
            texts = (
                {}
                if case.digest
                else {
                    f: g.sha256_bytes(g.zstd_read(gold / "graph" / f"{f}.zst")) for f in g.CSR_FILES
                }
            )
            row = {"case": case.name, "digest_mode": case.digest, "runs": []}
            for exe in args.exe:
                for threads in args.threads:
                    extra = [] if case.digest else ["--write-graph", tmp / "out" / "csr"]
                    rc, out = tool(
                        exe,
                        [
                            "generator",
                            *config_args(case),
                            "--threads",
                            threads,
                            "--golden",
                            golden,
                            *extra,
                        ],
                    )
                    text_equal = None
                    if not case.digest:
                        text_equal = all(
                            g.sha256_file(tmp / "out" / f) == texts[f] for f in g.CSR_FILES
                        )
                        shutil.rmtree(tmp / "out")
                    run = {
                        "exe": str(exe),
                        "threads": threads,
                        "arrays_equal": rc == 0 and out["equal"],
                        "arrays_compared": out.get("compared"),
                        "mismatches": out.get("mismatches"),
                        "text_files_equal": text_equal,
                        "generate_seconds": out["generate_seconds"],
                        "stream_sha256": stream_digest(out["digests"]),
                        "tau_bits": out["tau_bits"],
                    }
                    row["runs"].append(run)
                    good = run["arrays_equal"] and text_equal is not False
                    ok = ok and good
                    print(
                        f"{case.name} {Path(str(exe)).parents[2].name} t={threads}: "
                        f"{'equal' if good else 'DIFFERENT'} ({run['arrays_compared']} arrays"
                        f"{'' if case.digest else ', 4 text files'}; "
                        f"{run['generate_seconds']:.2f} s)",
                        flush=True,
                    )
            same = len({r["stream_sha256"] for r in row["runs"]}) == 1
            row["same_stream_for_every_run"] = same
            ok = ok and same
            results.append(row)
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "generator replay (criterion 8)",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print("generator replay:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- model ---------------------------------------------------------------------------------------


def cmd_model(args: argparse.Namespace) -> int:
    root = g.set_dir(g.scratch() / "goldens", "trace")
    results = []
    ok = True
    for case in select(g.P_CASES, args.cases):
        with tempfile.TemporaryDirectory(prefix="lp-model-", dir=runs_dir()) as t:
            trace = decompressed(root / "cases" / case.name / "trace.dlt.zst", Path(t))
            rc, out = tool(args.exe, ["model", "--trace", trace, "--threads", args.threads])
        good = rc == 0 and out["equal"]
        ok = ok and good
        results.append({"case": case.name, **out})
        print(
            f"{case.name}: {'exact' if good else 'DIFFERENT'} ({out['compared']} fields, "
            f"{out['mismatched']} mismatched, inputs {out['inputs']}, tau "
            f"{'equal' if out['generator_tau_equal'] else 'DIFFERENT'}; {out['seconds']:.1f} s)"
            + (f": {out['problem']} {out['mismatches'][:5]}" if not good else ""),
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "lp_host_model replay (criterion 7)",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print("model replay:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- O's --check runs with the recorder ----------------------------------------------------------


def check_case_args(case: g.Case) -> list[str]:
    validate = case.size == "5k5" and case.delta == "1e-6"
    c = g.Case(case.size, delta=case.delta, schedule=case.schedule, validate=validate)
    return c.argv(Path("<out>"))[:-2]


def cmd_check_runs(args: argparse.Namespace) -> int:
    """Runs the twin with --check (the caller holds the shared lock; GPU 1)."""
    copy = args.copy
    twin = copy / "bin" / "dynLP"
    if not (copy / "parity_export" / "bin" / "dynLP").is_file():
        raise g.GoldenError(f"{copy} is not a patched copy")
    if b"__wrap_snprintf" not in twin.read_bytes():
        raise g.GoldenError(f"{twin} was built without the step-2 recorder (no snprintf wrap)")
    before = g.origin_status()
    for case in select(C_CASES + L_CASES, args.cases):
        g.require_free(g.scratch())
        out = runs_dir() / "check" / case.name
        if (out / "dynLP.dlt").is_file() and args.skip_existing:
            continue
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        env = dict(
            g.clean_env(args.gpu),
            OMP_NUM_THREADS="8",
            DYNG_DYNLP_TRACE_DIR=str(out),
            DYNG_DYNLP_TRACE_TAGS=CHECK_TAGS,
        )
        cmd = [str(twin), *check_case_args(case), "--check", "--output", str(out / "o")]
        import time

        t0 = time.monotonic()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (out / "stdout.txt").write_text(p.stdout + p.stderr)
        shutil.rmtree(out / "o", ignore_errors=True)
        meta = {
            "case": case.name,
            "argv": cmd[1:],
            "exit": p.returncode,
            "seconds": time.monotonic() - t0,
            "gpu": g.gpu_record(args.gpu),
            "omp_num_threads": 8,
            "tags": CHECK_TAGS,
            "copy": str(copy),
        }
        g.write_json(out / "run.json", meta)
        print(f"{case.name}: exit {p.returncode}, {meta['seconds']:.0f} s", flush=True)
        if p.returncode != 0:
            raise g.GoldenError(f"{case.name}: O's --check exited {p.returncode}")
    if g.origin_status() != before:
        raise g.GoldenError(f"{g.REFERENCE}'s git state changed")
    return 0


# --- oracle --------------------------------------------------------------------------------------


def close(ours: float, theirs: float) -> bool:
    return abs(ours - theirs) <= ABS_TOL or abs(ours - theirs) <= REL_TOL * abs(theirs)


def cmd_oracle(args: argparse.Namespace) -> int:
    results = []
    ok = True
    for case in select(C_CASES + L_CASES, args.cases):
        d = runs_dir() / "check" / case.name
        trace_path = d / "dynLP.dlt"
        if not trace_path.is_file():
            raise g.GoldenError(f"{case.name}: no check run (run check-runs first)")
        t = dlt.read(trace_path)
        checks = {r.batch: r for r in t.records("check_labels")}
        _, out = tool(
            args.exe,
            ["oracle", *config_args(case), "--threads", args.threads, "--labels-trace", trace_path],
        )
        per = []
        for b in out["batches"]:
            o = checks.get(b["batch"])
            if o is None:
                per.append({"batch": b["batch"], "ok": False, "why": "no O record"})
                ok = False
                continue
            o_skipped = o.scalar("skipped") == 1
            row = {
                "batch": b["batch"],
                "o_skipped": o_skipped,
                "ours_converged": b["converged"],
                "o_pcg_iterations": o.scalar("pcg_iterations"),
                "ours_pcg_iterations": b["pcg_iterations"],
                "o_relative_residual": o.scalar("relative_residual"),
                "ours_relative_residual": b["relative_residual"],
                "ours_seconds": b["seconds"],
            }
            good = o_skipped == (not b["converged"])
            if not o_skipped:
                om = o.scalar("mean")
                row.update(
                    o_mean=om,
                    ours_mean=b["mean"],
                    abs_diff=abs(b["mean"] - om),
                    rel_diff=abs(b["mean"] - om) / om if om else 0.0,
                    o_compared=o.scalar("compared"),
                    ours_compared=b["compared"],
                    o_excluded=o.scalar("excluded"),
                    ours_excluded=b["excluded"],
                    o_flips=o.scalar("flips"),
                    ours_flips=b["flips"],
                    o_max=o.scalar("max"),
                    ours_max=b["max"],
                    tolerance=o.scalar("tolerance"),
                    margin=o.scalar("tolerance") / om if om > 0 else None,
                )
                good = (
                    good
                    and close(b["mean"], om)
                    and b["compared"] == row["o_compared"]
                    and b["excluded"] == row["o_excluded"]
                )
            row["ok"] = good
            ok = ok and good
            per.append(row)
        worst = max((r.get("abs_diff", 0.0) for r in per), default=0.0)
        print(
            f"{case.name}: {'reproduced' if all(r['ok'] for r in per) else 'DIFFERENT'} "
            f"({len(per)} batches, largest |mean - O's mean| {worst:.3g})",
            flush=True,
        )
        results.append(
            {"case": case.name, "batches": per, "o_run": json.loads((d / "run.json").read_text())}
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "harmonic_solution against O's checker (criterion 7c)",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "abs_tol": ABS_TOL,
                "rel_tol": REL_TOL,
                "date": g.now(),
                "threads": args.threads,
                "reproduced": ok,
                "cases": results,
            },
        )
    print("oracle replay:", "PASSED" if ok else "FAILED")
    if args.clean and ok:
        shutil.rmtree(runs_dir() / "check", ignore_errors=True)
    return 0 if ok else 1


# --- margin --------------------------------------------------------------------------------------


def cmd_margin(args: argparse.Namespace) -> int:
    goldens = g.scratch() / "goldens"
    damped_root = g.set_dir(goldens, "damped") / "cases"
    tight = g.set_dir(goldens, "inplace") / "tight"
    rows = []
    with tempfile.TemporaryDirectory(prefix="lp-margin-", dir=runs_dir()) as t:
        tmp = Path(t)
        for size in ("5k5", "50k"):
            case = g.Case(size, delta="1e-6")
            bits = decompressed(damped_root / case.name / "label_bits.bin.zst", tmp)
            _, out = tool(
                args.exe,
                ["oracle", *config_args(case), "--threads", args.threads, "--labels-bits", bits],
            )
            for b in out["batches"]:
                rows.append(
                    {
                        "source": f"damped {case.name} (golden label bits)",
                        "batch": b["batch"],
                        "flips": b["flips"],
                        "largest_flip_distance": b["largest_flip_distance"],
                        "compared": b["compared"],
                    }
                )
            inplace = g.Case(size, delta="1e-6", schedule="inplace")
            for run in sorted((tight / inplace.name).glob("*.labels.txt.zst")):
                txt = decompressed(run, tmp)
                _, out = tool(
                    args.exe,
                    [
                        "oracle",
                        *config_args(case),
                        "--threads",
                        args.threads,
                        "--final-labels",
                        txt,
                    ],
                )
                last = out["batches"][-1]
                rows.append(
                    {
                        "source": f"in_place {inplace.name} {run.name} (final labels, 6 decimals)",
                        "batch": last["batch"],
                        "flips": last["flips"],
                        "largest_flip_distance": last["largest_flip_distance"],
                        "compared": last["compared"],
                    }
                )
                txt.unlink()
        extra = []
        for case in C_CASES:
            trace_path = runs_dir() / "check" / case.name / "dynLP.dlt"
            if case.schedule != "inplace" or not trace_path.is_file():
                continue
            _, out = tool(
                args.exe,
                [
                    "oracle",
                    *config_args(case),
                    "--threads",
                    args.threads,
                    "--labels-trace",
                    trace_path,
                ],
            )
            for b in out["batches"]:
                extra.append(
                    {
                        "source": f"in_place {case.name} (check run, per batch)",
                        "batch": b["batch"],
                        "flips": b["flips"],
                        "largest_flip_distance": b["largest_flip_distance"],
                    }
                )
    m = max(r["largest_flip_distance"] for r in rows)
    beyond = [r for r in extra if r["largest_flip_distance"] > 3 * m]
    print(
        f"class margin m = {m:.6g} over {len(rows)} label sets; 3m = {3 * m:.6g}; "
        f"in_place per-batch values beyond 3m: {len(beyond)}"
    )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the class margin m (AD7, criterion 7)",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "date": g.now(),
                "m": m,
                "three_m": 3 * m,
                "rule": "no flip where |F* - 0.5| > 3m",
                "label_sets": rows,
                "inplace_per_batch_check_runs": extra,
                "inplace_per_batch_beyond_3m": beyond,
            },
        )
    return 0


# --- the committed subset ------------------------------------------------------------------------


def field_sha(f: dlt.Field) -> str:
    if f.digest:
        return f.sha256
    return hashlib.sha256(f.raw).hexdigest()


MODEL_FIELDS = {
    "apply_vertex_changes": ("status", "gt_class", "inserted", "num_inserted"),
    "append_edges": ("row_end", "slot_col_ind", "slot_values", "error_flags"),
    "sparsify": ("tau", "num_sub_edges", "sub_edges_sorted"),
    "connected_components": ("num_components", "component_of_inserted"),
    "label_init": ("num_components", "w0", "w1", "initial_labels_of_inserted"),
    "propagation_begin": ("frontier_size", "initial_frontier"),
}


def cmd_subset(args: argparse.Namespace) -> int:
    """cpp/tests/data/goldens/label_propagation: SHA-256 digests of O's own recorded arrays and
    files (never of dynG's output), for the CPU suite (dyng_lp_replay_tests)."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    goldens = g.scratch() / "goldens"
    head = (
        "# SHA-256 of LabelPropagation-CUDA@a276a3a's recorded {what} ({src}), written by\n"
        "# parity/label_propagation_replay.py subset; checked by cpp/tests/parity/\n"
        "# label_propagation/lp_replay_test.cpp. '<sha256>  <name>' per line.\n"
    )
    written = []
    for size in ("5k5", "50k"):
        case = g.Case(size)
        gold = g.set_dir(goldens, "generator") / "cases" / case.name
        t = dlt.read_bytes(g.zstd_read(gold / "generator.dlt.zst"))
        rec = t.one("generator")
        lines = [
            f"{field_sha(rec[f])}  {f}" for f in ("row_ptr", "col_ind", "values", "true_labels")
        ]
        for r in t.records("batch_data"):
            for f in (
                "inserted",
                "gt_ids",
                "gt_labels",
                "deleted",
                "edge_src",
                "edge_dst",
                "edge_weight",
            ):
                lines.append(f"{field_sha(r[f])}  b{r.batch}.{f}")
        for f in g.CSR_FILES:
            lines.append(f"{g.sha256_bytes(g.zstd_read(gold / 'graph' / f'{f}.zst'))}  {f}")
        p = SUBSET_DIR / f"generator_{case.name}.sha256"
        p.write_text(
            head.format(
                what="generator arrays and graph text files",
                src=f"label_propagation_generator/{case.name}",
            )
            + "\n".join(lines)
            + "\n"
        )
        written.append(p)
    case = g.Case("5k5")
    t = dlt.read_bytes(
        g.zstd_read(g.set_dir(goldens, "trace") / "cases" / case.name / "trace.dlt.zst")
    )
    lines = []
    for b, stages in sorted(t.batches().items()):
        for tag, names in MODEL_FIELDS.items():
            r = stages.get(tag)
            if r is None:
                continue
            for f in names:
                lines.append(f"{field_sha(r[f])}  b{b}.{tag}.{f}")
            if tag == "append_edges" and b == 0:
                lines.append(f"{field_sha(r['row_ptr'])}  b{b}.{tag}.row_ptr")
    p = SUBSET_DIR / f"model_{case.name}.sha256"
    p.write_text(
        head.format(
            what="stage-trace fields that lp_host_model predicts",
            src=f"label_propagation_trace/{case.name}",
        )
        + "\n".join(sorted(lines))
        + "\n"
    )
    written.append(p)
    for p in written:
        print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size} bytes)")
    return 0


# --- command line --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="compare.py label_propagation", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    names = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    gp = sub.add_parser("generator")
    gp.add_argument("--exe", type=Path, action="append", required=True)
    gp.add_argument("--threads", type=ints, default=[1, 8, 56])
    gp.add_argument("--cases", type=names)
    gp.add_argument("--json", type=Path)
    mp = sub.add_parser("model")
    mp.add_argument("--exe", type=Path, required=True)
    mp.add_argument("--threads", type=int, default=8)
    mp.add_argument("--cases", type=names)
    mp.add_argument("--json", type=Path)
    cp = sub.add_parser("check-runs")
    cp.add_argument("--copy", type=Path, required=True, help="a patched copy (step-2 recorder)")
    cp.add_argument("--gpu", type=int, default=1)
    cp.add_argument("--cases", type=names)
    cp.add_argument("--skip-existing", action="store_true")
    op = sub.add_parser("oracle")
    op.add_argument("--exe", type=Path, required=True)
    op.add_argument("--threads", type=int, default=8)
    op.add_argument("--cases", type=names)
    op.add_argument("--json", type=Path)
    op.add_argument("--clean", action="store_true", help="delete the check runs when reproduced")
    gm = sub.add_parser("margin")
    gm.add_argument("--exe", type=Path, required=True)
    gm.add_argument("--threads", type=int, default=8)
    gm.add_argument("--json", type=Path)
    sub.add_parser("subset")
    args = ap.parse_args(argv)
    try:
        return {
            "generator": cmd_generator,
            "model": cmd_model,
            "check-runs": cmd_check_runs,
            "oracle": cmd_oracle,
            "margin": cmd_margin,
            "subset": cmd_subset,
        }[args.command](args)
    except g.GoldenError as err:
        print(f"compare.py label_propagation {args.command}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
