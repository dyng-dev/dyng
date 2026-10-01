#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Mutation checks of the golden parity (PLAN Section 8.4): each recorded bug, put back into a
scratch copy of dynG, must make the golden replay fail.

    parity/mutate.py list
    parity/mutate.py run [--mutations sssp_roots_only_sequential,...] [--json out.json] [--keep]

`run` exports the current commit with `git archive HEAD` into $DYNG_SCRATCH/mutate/<commit7>/
(never the working tree), and for the control and every mutation: copies it, applies the
mutation (a textual replacement that must match exactly once), builds dyng-compat-mosp with the
`parity-cuda` preset (sequential, OpenMP and CUDA backends in one build), and replays the sssp
golden corpus of MOSP-OpenMP@c352151 on the mutation's configuration through parity/compare.py.
The control must pass on every configuration the mutations use; every mutation must fail (at
least one case differs). Builds and replays take the shared perf lock and run niced; CUDA
replays run on GPU $DYNG_TEST_GPU (default 1). The scratch copies are deleted unless --keep.
Exit 0 only if the control passes and every mutation is detected.

The mutations of cycle_count (CycleEnumeration-GPU's double-counted 5-cycles and weakened
ownership rule, and dynG's skipped workspace resize; host and CUDA) are CTests
(`ctest -L mutation`, cpp/tests/CMakeLists.txt, option DYNG_MUTATION_TESTS); the ones here need
the golden corpus, which CTest has only in the parity preset.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))

# PLAN 8.4, sssp: "a skipped subtree invalidation (the count-to-infinity revert)": only the roots
# (the heads of deleted or weight-increased tree edges) are invalidated, their descendants keep
# distances that may route through them, and the update converges to wrong distances (the
# count-to-infinity regressions of the original's tests, n = 6, seeds 621705 / 250813).
MUTATIONS = [
    {
        "name": "sssp_roots_only_sequential",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (sequential backend)",
        "file": "cpp/src/algorithms/sssp/sequential.cpp",
        "old": "        if (!is_invalid(child)) {",
        "new": "        if (false && !is_invalid(child)) {  // MUTATION: roots only",
        "config": "sequential",
    },
    {
        "name": "sssp_roots_only_openmp",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (OpenMP backend, chain walk)",
        "file": "cpp/src/algorithms/sssp/openmp.cpp",
        "old": "        const char result = load_state(u) == 0 ? char{1} : load_state(u);",
        "new": "        const char result = char{1};  // MUTATION: roots only",
        "config": "openmp:4",
    },
    {
        "name": "sssp_roots_only_cuda",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (CUDA fused kernel)",
        "file": "cpp/src/algorithms/sssp/fused.cuh",
        "old": "    if (p.changes.number_of_changed > 0) {\n      for (int round = 0;",
        "new": "    if (false && p.changes.number_of_changed > 0) {  // MUTATION: roots only\n"
        "      for (int round = 0;",
        "config": "cuda",
    },
]


def heavy(argv: list[str]) -> list[str]:
    lock = os.environ.get("DYNG_PERF_LOCK", str(SCRATCH / "perf.lock"))
    return (["flock", "-s", lock] if lock else []) + ["nice", "-n", "10", *argv]


def sh(argv: list[str], log: Path, cwd: Path | None = None, env: dict | None = None) -> int:
    with open(log, "a") as f:
        f.write(f"== {' '.join(argv)}\n")
        f.flush()
        return subprocess.run(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def apply(copy: Path, mutation: dict) -> None:
    path = copy / mutation["file"]
    text = path.read_text()
    count = text.count(mutation["old"])
    if count != 1:
        raise SystemExit(
            f"{mutation['name']}: the mutation point matches {count} times in {mutation['file']} "
            "(it must match once; update MUTATIONS with the code)"
        )
    path.write_text(text.replace(mutation["old"], mutation["new"]))


def build(copy: Path, log: Path) -> Path | None:
    exe = copy / "build" / "parity-cuda" / "tools" / "compat" / "dyng-compat-mosp"
    if sh(heavy(["cmake", "--preset", "parity-cuda"]), log, cwd=copy):
        return None
    if sh(
        heavy(["cmake", "--build", "--preset", "parity-cuda", "--target", "dyng-compat-mosp"]),
        log,
        cwd=copy,
    ):
        return None
    return exe if exe.is_file() else None


def replay(exe: Path, configs: list[str], out: Path, log: Path) -> dict:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DYNG_TEST_GPU", "1"))
    argv = [sys.executable, str(REPO / "parity" / "compare.py"), "--exe", str(exe)]
    argv += ["--configs", ",".join(configs), "--json", str(out)]
    rc = sh(heavy(argv), log, env=env)
    if not out.is_file():
        return {"rc": rc, "passed": False, "failed_cases": None}
    doc = json.loads(out.read_text())
    per_config = {c: 0 for c in configs}
    for groups in doc["matrix"].values():
        for config, cell in groups.items():
            per_config[config] = per_config.get(config, 0) + len(cell.get("fail", []))
    return {
        "rc": rc,
        "passed": bool(doc.get("passed")),
        "failed_cases": per_config,
        "goldens": doc["goldens"],
    }


def run(args: argparse.Namespace) -> int:
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    chosen = [m for m in MUTATIONS if not args.mutations or m["name"] in args.mutations]
    if args.mutations and len(chosen) != len(args.mutations):
        raise SystemExit(f"unknown mutations; known: {[m['name'] for m in MUTATIONS]}")
    work = args.work or SCRATCH / "mutate" / head[:7]
    work.mkdir(parents=True, exist_ok=True)
    log = work / "mutate.log"
    src = work / "src"
    if not src.is_dir():
        src.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", "HEAD"], check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    configs = sorted({m["config"] for m in chosen})
    results = []
    for entry in [{"name": "control", "plan": "no mutation", "config": ",".join(configs)}, *chosen]:
        copy = work / entry["name"]
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(src, copy, symlinks=True)
        if entry["name"] != "control":
            apply(copy, entry)
        print(f"[mutate] {entry['name']}: build", flush=True)
        exe = build(copy, log)
        if exe is None:
            result = {"build": "FAILED", "detected": False}
        else:
            print(f"[mutate] {entry['name']}: replay on {entry['config']}", flush=True)
            result = replay(exe, entry["config"].split(","), work / f"{entry['name']}.json", log)
            result["build"] = "ok"
            if entry["name"] == "control":
                result["detected"] = None
                result["verdict"] = "passed" if result["passed"] and result["rc"] == 0 else "FAILED"
            else:
                failed = sum((result.get("failed_cases") or {}).values())
                result["detected"] = (not result["passed"]) and failed > 0
                result["verdict"] = "detected" if result["detected"] else "NOT DETECTED"
        results.append({k: v for k, v in entry.items() if k not in ("old", "new")} | result)
        print(f"[mutate] {entry['name']}: {result.get('verdict', result['build'])}", flush=True)
        if not args.keep:
            shutil.rmtree(copy, ignore_errors=True)
    if not args.keep:
        shutil.rmtree(src, ignore_errors=True)
    control_ok = results[0].get("verdict") == "passed"
    detected = all(r.get("detected") for r in results[1:])
    doc = {
        "schema": 1,
        "what": "mutation checks of the sssp golden parity (PLAN 8.4)",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": head,
        "goldens": "sssp (MOSP-OpenMP@c352151), parity/goldens.toml",
        "build": "parity-cuda preset, target dyng-compat-mosp, from git archive HEAD",
        "results": results,
        "passed": control_ok and detected,
    }
    if args.json:
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        f"[mutate] control {'passed' if control_ok else 'FAILED'}; "
        f"{sum(1 for r in results[1:] if r.get('detected'))} / {len(results) - 1} "
        "mutations detected"
    )
    return 0 if doc["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--mutations", type=lambda s: [x for x in s.split(",") if x], default=None)
    r.add_argument("--work", type=Path, default=None, help="default $DYNG_SCRATCH/mutate/<commit7>")
    r.add_argument("--json", type=Path)
    r.add_argument("--keep", action="store_true", help="keep the scratch copies")
    args = parser.parse_args(argv)
    if args.command == "list":
        for m in MUTATIONS:
            print(f"{m['name']:32} {m['config']:10} {m['plan']}")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
