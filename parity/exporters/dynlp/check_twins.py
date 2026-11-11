#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that the DynLP recording twins behave as the unpatched original (ADRs 0036, 0065).

    check_twins.py [--scratch DIR] [--gpu 1] [--json OUT] [--keep]

For damped 5k5 (--totalNodes 5000 --numBatches 5) and 50k (50000 / 10), '--avgDegree 5 --delta
0.0001 --seed 42 --update damped --quiet --noValidate', it runs
  - the UNPATCHED copy's bin/dynLP (the reference),
  - the PATCHED copy's bin/dynLP (the twin) with recording off (no DYNG_DYNLP_* variables), and
  - the twin with recording on (DYNG_DYNLP_TRACE_DIR set),
each in a fresh output directory, and requires dynlp_labels.txt to be byte-identical and the
results.csv columns iterations, frontier_visits, components and accuracy to be equal (time_ms
and eval_ms are timings). With recording on it also requires the trace to be complete (its end
record), one apply_vertex_changes marker per batch and the trace's final label bits to print
O's dynlp_labels.txt bytes ('%d %.6f' through std::to_chars, utils.cu:194-240). For
bin/propagation_test it compares the unpatched test's output lines of the damped rule with the
twin's, off and on (the in-place rule is not reproducible: its lines are compared on the verdict
only), and with recording on requires one hub_propagation record per call.

Exit status 0 when every comparison holds.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "parity"))
import dynlp_trace  # noqa: E402

REF = "LabelPropagation-CUDA@a276a3a"
CASES = {
    "5k5": ["--totalNodes", "5000", "--numBatches", "5"],
    "50k": ["--totalNodes", "50000", "--numBatches", "10"],
}
COMMON = ["--avgDegree", "5", "--delta", "0.0001", "--seed", "42", "--update", "damped",
          "--quiet", "--noValidate"]  # fmt: skip
COLUMNS = ("iterations", "frontier_visits", "components", "accuracy")


def clean_env(gpu: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYNG_DYNLP_", "OMP_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return env


def run(exe: Path, args: list[str], out: Path, env: dict) -> subprocess.CompletedProcess:
    out.mkdir(parents=True)
    p = subprocess.run([str(exe), *args, "--output", str(out)], env=env, capture_output=True,
                       text=True)  # fmt: skip
    if p.returncode != 0:
        raise SystemExit(f"{exe} {' '.join(args)} failed ({p.returncode}):\n{p.stdout[-1500:]}")
    return p


def columns(path: Path) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    return [{c: r[c] for c in COLUMNS} for r in rows]


def label_text(labels) -> bytes:
    """O's dynlp_labels.txt from label floats: 'i label' with the label as std::to_chars(double,
    fixed, 6) (utils.cu writeLabelsToFile / TextWriter::putFixed6). Python's '%.6f' of the double
    is correctly rounded, as to_chars is."""
    return "".join(f"{i} {float(x):.6f}\n" for i, x in enumerate(labels)).encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    scratch = os.environ.get("DYNG_SCRATCH", Path.home() / "Projects/dyng-work")
    ap.add_argument("--scratch", type=Path, default=Path(scratch))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)
    base = args.scratch / "ref" / REF
    orig, twin = base / "unpatched" / "bin" / "dynLP", base / "patched" / "bin" / "dynLP"
    marker = b"dyng-dynlp-recorder-twin format 1"
    if marker in orig.read_bytes() or marker not in twin.read_bytes():
        raise SystemExit("the unpatched bin/dynLP must be the original, the patched one the twin")
    env = clean_env(args.gpu)
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dynlp-twins-", dir=args.scratch / "runs"))
    results: dict = {"reference": REF, "gpu": args.gpu, "cases": {}}
    ok = True
    try:
        for case, case_args in CASES.items():
            a = case_args + COMMON
            run(orig, a, work / case / "orig", env)
            run(twin, a, work / case / "off", env)
            tenv = dict(env, DYNG_DYNLP_TRACE_DIR=str(work / case / "trace"))
            run(twin, a, work / case / "on", tenv)
            ref_labels = (work / case / "orig" / "dynlp_labels.txt").read_bytes()
            ref_cols = columns(work / case / "orig" / "results.csv")
            res: dict = {"args": a}
            for mode in ("off", "on"):
                d = work / case / mode
                same_labels = (d / "dynlp_labels.txt").read_bytes() == ref_labels
                same_cols = columns(d / "results.csv") == ref_cols
                res[mode] = {"labels_identical": same_labels, "columns_equal": same_cols}
                ok &= same_labels and same_cols
            t = dynlp_trace.read(work / case / "trace" / "dynLP.dlt")
            batches = t.batches()
            nb = int(case_args[3])
            markers = sorted(b for b, st in batches.items() if "apply_vertex_changes" in st)
            final = t.one("run_end")["final_labels"].values()
            res["trace"] = {
                "records": len(t.all),
                "batches_with_marker": markers,
                "one_marker_per_batch": markers == list(range(nb)),
                "final_label_bits_print_dynlp_labels": label_text(final) == ref_labels,
            }
            ok &= res["trace"]["one_marker_per_batch"]
            ok &= res["trace"]["final_label_bits_print_dynlp_labels"]
            results["cases"][case] = res
            print(f"{case}: {json.dumps(res)}", flush=True)
        # bin/propagation_test: the damped lines (deterministic) and the verdict.
        porig = base / "unpatched" / "bin" / "propagation_test"
        ptwin = base / "patched" / "bin" / "propagation_test"
        outs = {}
        (work / "hub").mkdir()
        for mode, exe, e in [
            ("orig", porig, env),
            ("off", ptwin, env),
            ("on", ptwin, dict(env, DYNG_DYNLP_TRACE_DIR=str(work / "hub" / "trace"))),
        ]:
            p = subprocess.run([str(exe)], env=e, capture_output=True, text=True)
            outs[mode] = (p.returncode, p.stdout.splitlines())

        def damped(lines: list[str]) -> list[str]:
            return [x for x in lines if not x.startswith("inplace")]

        def verdict(lines: list[str]) -> list[str]:
            return [x.rsplit(":", 1)[-1].strip() for x in lines if x.startswith("inplace")]

        hub = {}
        for mode in ("off", "on"):
            hub[mode] = {
                "exit_equal": outs[mode][0] == outs["orig"][0] == 0,
                "damped_lines_equal": damped(outs[mode][1]) == damped(outs["orig"][1]),
                "inplace_verdict_equal": verdict(outs[mode][1]) == verdict(outs["orig"][1]),
            }
            ok &= all(hub[mode].values())
        ht = dynlp_trace.read(work / "hub" / "trace" / "propagation_test.dlt")
        hub["trace_calls"] = len(ht.records("hub_propagation"))
        ok &= hub["trace_calls"] == 3
        results["propagation_test"] = hub
        print(f"propagation_test: {json.dumps(hub)}", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    results["passed"] = ok
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    print("check_twins:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
