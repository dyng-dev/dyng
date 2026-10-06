#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Mutation checks of the golden parity (PLAN Section 8.4): each recorded bug, put back into a
scratch copy of dynG, must make its golden suite fail.

    parity/mutate.py list
    parity/mutate.py run [--mutations sssp_roots_only_sequential,...] [--suites sssp,mosp,...]
                         [--json out.json] [--keep]

`run` exports the current commit with `git archive HEAD` into $DYNG_SCRATCH/mutate/<commit7>/
(never the working tree), and for the control and every mutation: copies it, applies the
mutation, builds the compatibility driver of the mutation's suite with the `parity-cuda` preset
(sequential, OpenMP and CUDA backends in one build), and replays that suite's golden corpus on
the mutation's configuration through parity/compare.py. A mutation is either a textual
replacement (`old` -> `new`, which must match exactly once) or a list of `defines`: the
recorded mutation hooks that are already in the sources (`#if defined(DYNG_MUTATION_...)`, the
same ones the CTests `*.mutation.*` compile in), passed to the host and the CUDA compiler.

The suites (each a golden corpus of parity/goldens.toml and its replay):

  sssp         the sssp corpus of MOSP-OpenMP@c352151 through dyng-compat-mosp (`compare.py`):
               the K initial and updated SOSP trees of every case
  mosp         the same corpus and replay, whose cases also hold the MOSP outputs of the
               original: the combined graph's tree and distances and the MOSP path costs
               (combined/{distancesCsr,SSSPTreeCsr,mospCosts}.txt); a mosp mutation is detected
               by those files (the SOSP trees are computed before the combined graph)
  cycle_count  the cycle_count corpus of CycleEnumeration-GPU@0a976ad through
               dyng-compat-cycle-enum (`compare.py cycle_count`): histograms and generated
               batches; the `cuda` configuration replays the set `cycle_count_cuda` (the
               original's CUDA backend)

The control (no mutation) must pass on every (suite, configuration) the chosen mutations use;
every mutation must fail (at least one case differs). Builds and replays take the shared perf
lock and run niced; CUDA replays run on GPU $DYNG_TEST_GPU (default 1). The scratch copies are
deleted unless --keep. Exit 0 only if the control passes and every mutation is detected.

The CTests `ctest -L mutation` (cpp/tests/CMakeLists.txt, option DYNG_MUTATION_TESTS) check the
cycle_count mutations against the unit and randomized suites as well, on every build; the runs
here need the golden corpus, which lives outside the repository (docs/developer/robustness.md).
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

# The golden suites: the driver each one builds and the arguments of its replay
# (parity/compare.py [<prefix>] --exe <driver> --configs <config> --json <out>).
SUITES = {
    "sssp": {"target": "dyng-compat-mosp", "compare": [], "goldens": "sssp"},
    "mosp": {"target": "dyng-compat-mosp", "compare": [], "goldens": "sssp"},
    "cycle_count": {"target": "dyng-compat-cycle-enum", "compare": ["cycle_count"]},
}

MUTATIONS = [
    # PLAN 8.4, sssp: "a skipped subtree invalidation (the count-to-infinity revert)": only the
    # roots (the heads of deleted or weight-increased tree edges) are invalidated, their
    # descendants keep distances that may route through them, and the update converges to wrong
    # distances (the count-to-infinity regressions of the original's tests, n = 6, seeds 621705 /
    # 250813; MOSP-OpenMP's CHANGES.md, M-a).
    {
        "name": "sssp_roots_only_sequential",
        "suite": "sssp",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (sequential backend)",
        "file": "cpp/src/algorithms/sssp/sequential.cpp",
        "old": "        if (!is_invalid(child)) {",
        "new": "        if (false && !is_invalid(child)) {  // MUTATION: roots only",
        "config": "sequential",
    },
    {
        "name": "sssp_roots_only_openmp",
        "suite": "sssp",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (OpenMP backend, chain walk)",
        "file": "cpp/src/algorithms/sssp/openmp.cpp",
        "old": "        const char result = load_state(u) == 0 ? char{1} : load_state(u);",
        "new": "        const char result = char{1};  // MUTATION: roots only",
        "config": "openmp:4",
    },
    {
        "name": "sssp_roots_only_cuda",
        "suite": "sssp",
        "plan": "PLAN 8.4 sssp: a skipped subtree invalidation (CUDA fused kernel)",
        "file": "cpp/src/algorithms/sssp/fused.cuh",
        "old": "    if (p.changes.number_of_changed > 0) {\n      for (int round = 0;",
        "new": "    if (false && p.changes.number_of_changed > 0) {  // MUTATION: roots only\n"
        "      for (int round = 0;",
        "config": "cuda",
    },
    # PLAN 8.4, CycleEnum: "double counting 5-cycles; weakened ownership rule", the two bugs
    # CycleEnumeration-GPU recorded. The hooks are in the sources (cycle_count/problem.hpp: the
    # host searches of the sequential and OpenMP backends; dfs.cuh and cuda.cu: the device
    # kernels, where the original recorded them).
    {
        "name": "cycle_count_double_count_5",
        "suite": "cycle_count",
        "plan": "PLAN 8.4 CycleEnum: double counting 5-cycles (host searches)",
        "defines": ["DYNG_MUTATION_DOUBLE_COUNT_5"],
        "config": "sequential",
    },
    {
        "name": "cycle_count_weak_ownership",
        "suite": "cycle_count",
        "plan": "PLAN 8.4 CycleEnum: weakened ownership rule (host update searches)",
        "defines": ["DYNG_MUTATION_WEAK_OWNERSHIP"],
        "config": "openmp:4",
    },
    {
        "name": "cycle_count_double_count_5_cuda",
        "suite": "cycle_count",
        "plan": "PLAN 8.4 CycleEnum: double counting 5-cycles (CUDA work-queue kernel)",
        "defines": ["DYNG_MUTATION_CUDA_DOUBLE_COUNT_5"],
        "config": "cuda",
    },
    {
        "name": "cycle_count_weak_ownership_cuda",
        "suite": "cycle_count",
        "plan": "PLAN 8.4 CycleEnum: weakened ownership rule (CUDA update kernel)",
        "defines": ["DYNG_MUTATION_CUDA_WEAK_OWNERSHIP"],
        "config": "cuda",
    },
    # mosp (PLAN 8.4 lists none; these put back the two parts of the combined-graph step that the
    # originals' preference vector fixed, MOSP-OpenMP's CHANGES.md M-d): the weight of a combined
    # edge counts only the first tree that holds it (W(e) = K + 1 - sum over every tree holding e
    # is the rule), on each backend; and the path costs read the first objective's weights for
    # every objective (mospPathCosts; shared by every backend, run on the sequential one).
    {
        "name": "mosp_combined_first_tree_sequential",
        "suite": "mosp",
        "plan": "mosp: combined-edge weight from the first tree only (sequential backend)",
        "file": "cpp/src/algorithms/mosp/sequential.cpp",
        "old": "      w -= in.terms[j];",
        "new": "      w -= 0;  // MUTATION: first tree only",
        "config": "sequential",
    },
    {
        "name": "mosp_combined_first_tree_openmp",
        "suite": "mosp",
        "plan": "mosp: combined-edge weight from the first tree only (OpenMP backend)",
        "file": "cpp/src/algorithms/mosp/openmp.cpp",
        "old": "      w -= in.terms[j];",
        "new": "      w -= 0;  // MUTATION: first tree only",
        "config": "openmp:4",
    },
    {
        "name": "mosp_combined_first_tree_cuda",
        "suite": "mosp",
        "plan": "mosp: combined-edge weight from the first tree only (CUDA backend)",
        "file": "cpp/src/algorithms/mosp/cuda.cu",
        "old": "      w -= in.terms[j];",
        "new": "      w -= 0;  // MUTATION: first tree only",
        "config": "cuda",
    },
    {
        "name": "mosp_path_costs_first_objective",
        "suite": "mosp",
        "plan": "mosp: the path costs of every objective from the first objective's weights",
        "file": "cpp/src/algorithms/mosp/sequential.cpp",
        "old": "static_cast<std::int64_t>(weights[j * m + static_cast<std::size_t>(edge)]);",
        "new": "static_cast<std::int64_t>(weights[static_cast<std::size_t>(edge)]);  // MUTATION",
        "config": "sequential",
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
    """Applies a textual mutation (old -> new, exactly one match) to the copy; a mutation by
    `defines` changes no file (build() passes them to the compilers)."""
    if "file" not in mutation:
        return
    path = copy / mutation["file"]
    text = path.read_text()
    count = text.count(mutation["old"])
    if count != 1:
        raise SystemExit(
            f"{mutation['name']}: the mutation point matches {count} times in {mutation['file']} "
            "(it must match once; update MUTATIONS with the code)"
        )
    path.write_text(text.replace(mutation["old"], mutation["new"]))


def configure_args(defines: list[str]) -> list[str]:
    """The cache variables that compile the mutation hooks `defines` in (host and CUDA code)."""
    if not defines:
        return []
    flags = " ".join(f"-D{d}" for d in defines)
    return [f"-DCMAKE_CXX_FLAGS={flags}", f"-DCMAKE_CUDA_FLAGS={flags}"]


def build(copy: Path, targets: list[str], defines: list[str], log: Path) -> dict | None:
    """Builds `targets` in the copy (parity-cuda preset); their paths, or None on a failure."""
    out = copy / "build" / "parity-cuda" / "tools" / "compat"
    if sh(heavy(["cmake", "--preset", "parity-cuda", *configure_args(defines)]), log, cwd=copy):
        return None
    argv = ["cmake", "--build", "--preset", "parity-cuda", "--target", *targets]
    if sh(heavy(argv), log, cwd=copy):
        return None
    exes = {t: out / t for t in targets}
    return exes if all(e.is_file() for e in exes.values()) else None


def failures(doc: dict, configs: list[str]) -> dict:
    """Failed cases per configuration in a compare.py JSON record (both matrix layouts: the sssp
    corpus has {group: {config: {"fail": [...]}}}, the cycle_count corpus {case: {config:
    "equal" | "skipped" | [problems]}})."""
    per_config = {c: 0 for c in configs}
    for row in doc["matrix"].values():
        for config, cell in row.items():
            if isinstance(cell, dict):
                bad = len(cell.get("fail", []))
            else:
                bad = 1 if isinstance(cell, list) and cell else 0
            per_config[config] = per_config.get(config, 0) + bad
    return per_config


def replay(suite: str, exe: Path, config: str, out: Path, log: Path) -> dict:
    """Replays the golden corpus of `suite` with `exe` on one configuration."""
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DYNG_TEST_GPU", "1"))
    argv = [sys.executable, str(REPO / "parity" / "compare.py"), *SUITES[suite]["compare"]]
    argv += ["--exe", str(exe), "--configs", config, "--json", str(out)]
    rc = sh(heavy(argv), log, env=env)
    if not out.is_file():
        return {"rc": rc, "passed": False, "failed_cases": None}
    doc = json.loads(out.read_text())
    return {
        "rc": rc,
        "passed": bool(doc.get("passed")) and rc == 0,
        "failed_cases": failures(doc, [config]),
        "goldens": doc["goldens"],
    }


def replay_key(mutation: dict) -> tuple:
    """Mutations with the same key are replayed the same way (the control replays each once)."""
    suite = SUITES[mutation["suite"]]
    return (suite["target"], tuple(suite["compare"]), mutation["config"])


def run(args: argparse.Namespace) -> int:
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    chosen = [
        m
        for m in MUTATIONS
        if (not args.mutations or m["name"] in args.mutations)
        and (not args.suites or m["suite"] in args.suites)
    ]
    if args.mutations and len(chosen) != len(args.mutations):
        raise SystemExit(f"unknown mutations; known: {[m['name'] for m in MUTATIONS]}")
    if args.suites and not set(args.suites) <= set(SUITES):
        raise SystemExit(f"unknown suites; known: {sorted(SUITES)}")
    if not chosen:
        raise SystemExit("no mutation chosen")
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
    # The control replays every (driver, corpus, configuration) that a chosen mutation uses.
    control_replays: dict[tuple, dict] = {}
    for m in chosen:
        control_replays.setdefault(replay_key(m), m)
    control = {
        "name": "control",
        "plan": "no mutation",
        "suite": ",".join(sorted({m["suite"] for m in chosen})),
        "config": ",".join(sorted({m["config"] for m in chosen})),
    }
    results = []
    for entry in [control, *chosen]:
        is_control = entry["name"] == "control"
        copy = work / entry["name"]
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(src, copy, symlinks=True)
        apply(copy, entry)
        replays = list(control_replays.values()) if is_control else [entry]
        targets = sorted({SUITES[r["suite"]]["target"] for r in replays})
        print(f"[mutate] {entry['name']}: build {' '.join(targets)}", flush=True)
        exes = build(copy, targets, entry.get("defines", []), log)
        if exes is None:
            result: dict = {"build": "FAILED", "detected": False, "verdict": "BUILD FAILED"}
        else:
            runs = []
            for r in replays:
                print(f"[mutate] {entry['name']}: replay {r['suite']} on {r['config']}", flush=True)
                tag = f"{r['suite']}-{r['config'].replace(':', '')}" if is_control else ""
                out = work / (f"control-{tag}.json" if is_control else f"{entry['name']}.json")
                one = replay(r["suite"], exes[SUITES[r["suite"]]["target"]], r["config"], out, log)
                runs.append({"suite": r["suite"], "config": r["config"], **one})
            result = {"build": "ok"}
            if is_control:
                result["replays"] = runs
                result["detected"] = None
                result["verdict"] = "passed" if all(x["passed"] for x in runs) else "FAILED"
            else:
                result |= {k: v for k, v in runs[0].items() if k not in ("suite", "config")}
                failed = sum((result.get("failed_cases") or {}).values())
                result["detected"] = (not result["passed"]) and failed > 0
                result["verdict"] = "detected" if result["detected"] else "NOT DETECTED"
        results.append({k: v for k, v in entry.items() if k not in ("old", "new")} | result)
        print(f"[mutate] {entry['name']}: {result['verdict']}", flush=True)
        if not args.keep:
            shutil.rmtree(copy, ignore_errors=True)
    if not args.keep:
        shutil.rmtree(src, ignore_errors=True)
    control_ok = results[0].get("verdict") == "passed"
    detected = all(r.get("detected") for r in results[1:])
    suites = sorted({m["suite"] for m in chosen})
    doc = {
        "schema": 2,
        "what": f"mutation checks of the golden parity (PLAN 8.4): {', '.join(suites)}",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": head,
        "goldens": {
            "sssp": "sssp (MOSP-OpenMP@c352151; the mosp outputs included), parity/goldens.toml",
            "cycle_count": "cycle_count and cycle_count_cuda (CycleEnumeration-GPU@0a976ad), "
            "parity/goldens.toml",
        },
        "build": "parity-cuda preset, targets dyng-compat-mosp / dyng-compat-cycle-enum, "
        "from git archive HEAD",
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
    r.add_argument(
        "--suites",
        type=lambda s: [x for x in s.split(",") if x],
        default=None,
        help=f"only the mutations of these suites ({', '.join(SUITES)})",
    )
    r.add_argument("--work", type=Path, default=None, help="default $DYNG_SCRATCH/mutate/<commit7>")
    r.add_argument("--json", type=Path)
    r.add_argument("--keep", action="store_true", help="keep the scratch copies")
    args = parser.parse_args(argv)
    if args.command == "list":
        for m in MUTATIONS:
            print(f"{m['name']:36} {m['suite']:12} {m['config']:10} {m['plan']}")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
