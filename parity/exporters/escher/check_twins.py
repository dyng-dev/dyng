#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that the CBST recording twins behave as the original executables (ADR 0036).

    check_twins.py ESCHER-GPU|MOSP_ESCHER [--scratch DIR] [--json OUT] [--keep]

1. Runs the original's own test suite (references.toml `smoke_test`, e.g. `make test`) in the
   PATCHED copy, whose executables are the twins, with DYNG_CBST_INVOCATION_LOG set and
   DYNG_CBST_TRACE_DIR unset: every twin invocation of the suite is logged (working directory,
   arguments, its input files, the number of CBST operations it called).
2. Replays every logged invocation twice, in a fresh working directory each: with the UNPATCHED
   copy's executable and with the twin (both without DYNG_CBST_* variables, the same arguments
   and the same input files), and compares their exit statuses and their output (stdout and
   stderr) with the timing lines left out (lines with a time in ms or s).

Exit status 0 when the suite passed in the patched copy and every invocation matched. The JSON
record lists every invocation with both statuses and the number of timing lines left out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFS = REPO / "parity" / "references.toml"
# A line that reports a time ("12.3 ms", "0.5 s", "[timing] ...", "built in 12 ms", time columns).
TIMING = re.compile(
    r"\d(\.\d+)?\s*(ms|s|sec|secs|seconds|us|µs)\b|\[timing\]|\btime[sd]?\b|_ms\b|elapsed",
    re.IGNORECASE,
)


# Executables whose output is not reproducible in the ORIGINAL itself (the unpatched copy prints
# different bytes on every run): compared on the exit status and the verdict lines only.
NONDETERMINISTIC = {
    ("MOSP_ESCHER", "bin/main"): (
        "MOSP's demo pipeline draws its graph from std::random_device "
        "(mosp/src/generateGraph.cu:89-90): two runs of the unpatched copy differ",
        re.compile(r"PASS|FAIL|passed|failed|complete"),
    ),
}


def reference(name: str) -> dict:
    for r in tomllib.loads(REFS.read_text())["reference"]:
        if r["name"] == name:
            return r
    raise SystemExit(f"unknown reference {name}")


def copy_dir(name: str, variant: str, scratch: Path) -> Path:
    out = subprocess.check_output(
        [
            str(REPO / "parity" / "build_reference.sh"),
            "--variant",
            variant,
            "--scratch",
            str(scratch),
            "--print-dir",
            name,
        ],
        text=True,
    )
    return Path(out.strip().splitlines()[-1])


def read_invocation(path: Path) -> dict:
    inv = {"file": path.name, "args": [], "inputs": {}, "calls": None, "exit": None}
    for line in path.read_text(errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key == "cwd":
            inv["cwd"] = value
        elif key == "arg":
            inv["args"].append(value)
        elif key == "input":
            i, stored = value.split(" ", 1)
            inv["inputs"][int(i)] = stored
        elif key == "calls":
            inv["calls"] = int(value)
        elif key == "exit":
            inv["exit"] = value
    return inv


def filtered(text: str) -> tuple[list[str], int]:
    kept, dropped = [], 0
    for line in text.splitlines():
        if TIMING.search(line):
            dropped += 1
        else:
            kept.append(line)
    return kept, dropped


def run(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout
        )
        return p.returncode, p.stdout.decode(errors="replace")
    except subprocess.TimeoutExpired as e:
        return -999, (e.stdout or b"").decode(errors="replace") + "\n<timeout>"


def clean_env(ref: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYNG_CBST_")}
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    for item in ref.get("run_env", []):
        k, _, v = item.partition("=")
        env[k] = v
    return env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("name", choices=["ESCHER-GPU", "MOSP_ESCHER"])
    p.add_argument(
        "--scratch",
        default=os.environ.get("DYNG_SCRATCH", str(Path.home() / "Projects" / "dyng-work")),
    )
    p.add_argument("--json", help="write the record here")
    p.add_argument("--keep", action="store_true", help="keep the invocation log")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per replayed invocation")
    args = p.parse_args(argv)
    scratch = Path(args.scratch)
    ref = reference(args.name)
    patched = copy_dir(args.name, "patched", scratch)
    unpatched = copy_dir(args.name, "unpatched", scratch)
    env = clean_env(ref)
    work = Path(tempfile.mkdtemp(prefix=f"check_twins_{args.name}_", dir=scratch / "runs"))
    log_dir = work / "invocations"
    record: dict = {
        "reference": args.name,
        "commit": ref["commit"],
        "patched": str(patched),
        "unpatched": str(unpatched),
        "invocations": [],
    }
    try:
        suite_env = dict(env, DYNG_CBST_INVOCATION_LOG=str(log_dir))
        status, out = run(["bash", "-c", ref["smoke_test"]], patched, suite_env, 4 * 3600)
        (work / "suite.log").write_text(out)
        record["suite"] = {
            "command": ref["smoke_test"],
            "status": status,
            "pass": len(re.findall(r"^(PASS|XFAIL)\b", out, re.M)),
            "fail": len(re.findall(r"^(FAIL|XPASS)\b", out, re.M)),
        }
        print(
            f"{args.name}: the suite in the patched copy: exit {status}, "
            f"{record['suite']['pass']} PASS, {record['suite']['fail']} FAIL",
            flush=True,
        )
        problems = [] if status == 0 else [f"the suite failed in the patched copy (exit {status})"]
        invs = [read_invocation(f) for f in sorted(log_dir.glob("*.inv"))]
        for inv in invs:
            exe = Path(inv["args"][0])
            exe = exe if exe.is_absolute() else Path(inv["cwd"]) / exe
            exe = Path(os.path.normpath(exe))
            try:
                rel = exe.relative_to(patched)
            except ValueError:
                problems.append(f"{inv['file']}: {exe} is not in the patched copy")
                continue
            cmd_args = list(inv["args"][1:])
            for i, stored in inv["inputs"].items():
                if stored != "-":
                    cmd_args[i - 1] = str(log_dir / stored)
            results = {}
            nondet = NONDETERMINISTIC.get((args.name, str(rel)))
            for side, root in (("unpatched", unpatched), ("twin", patched)):
                cwd = Path(tempfile.mkdtemp(prefix="run_", dir=work))
                status, out = run([str(root / rel), *cmd_args], cwd, env, args.timeout)
                lines, dropped = filtered(out)
                if nondet:
                    lines = [line for line in lines if nondet[1].search(line)]
                results[side] = {"status": status, "lines": lines, "dropped": dropped}
                shutil.rmtree(cwd, ignore_errors=True)
            same = (
                results["unpatched"]["status"] == results["twin"]["status"]
                and results["unpatched"]["lines"] == results["twin"]["lines"]
            )
            entry = {
                "invocation": inv["file"],
                "exe": str(rel),
                "args": [
                    a if not a.startswith("/") else "<abs>/" + Path(a).name for a in inv["args"][1:]
                ],
                "cbst_calls": inv["calls"],
                "logged_exit": inv["exit"],
                "status_unpatched": results["unpatched"]["status"],
                "status_twin": results["twin"]["status"],
                "timing_lines_dropped": results["twin"]["dropped"],
                "same": same,
            }
            if nondet:
                entry["nondeterministic_original"] = nondet[0]
                entry["compared"] = "exit status and verdict lines (" + nondet[1].pattern + ")"
            record["invocations"].append(entry)
            if not same:
                a, b = results["unpatched"]["lines"], results["twin"]["lines"]
                first = next(
                    (i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b))
                )
                entry["first_difference"] = {
                    "line": first,
                    "unpatched": a[first] if first < len(a) else None,
                    "twin": b[first] if first < len(b) else None,
                }
                problems.append(
                    f"{inv['file']} {rel} {' '.join(entry['args'])}: differs "
                    f"({entry['first_difference']})"
                )
            print(
                f"  {'same' if same else 'DIFFERENT'}: {rel} {' '.join(entry['args'])} "
                f"(exit {entry['status_twin']}, {inv['calls']} CBST calls)",
                flush=True,
            )
        record["problems"] = problems
        record["passed"] = not problems and bool(invs)
        print(
            f"{args.name}: {len(invs)} invocations replayed, "
            f"{sum(e['same'] for e in record['invocations'])} same; "
            f"{'OK' if record['passed'] else 'FAILED'}"
        )
        for pr in problems:
            print("  " + pr)
    finally:
        if args.json:
            Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept {work}")
    return 0 if record.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
