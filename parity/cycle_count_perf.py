#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Performance A/B of cycle_count (OpenMP) against CycleEnumeration-GPU@0a976ad (PLAN 6.4.3, 8.6).

    flock $DYNG_SCRATCH/perf.lock parity/perf_ab.py cycle_count run \\
        --exe build/parity/tools/compat/dyng-compat-cycle-enum [--runs 11] [--threads 56] \\
        [--cases DD_k3,...,twitch_k4_25000_25000_s1] [--json parity/results/M2a-perf-....json]

(or `parity/cycle_count_perf.py run ...` directly).

run   rebuilds (idempotently) and verifies the UNPATCHED copy of CycleEnumeration-GPU@0a976ad
      (parity/build_reference.sh), checks that --exe comes from a parity-preset build tree, and
      for every case alternates the original (A: build/cycle-enum) and the port (B:
      dyng-compat-cycle-enum --timing) A/B/A/B for --runs rounds, both with --backend openmp
      --openmp-threads 56 (the original's docs/RESULTS.md) and the same environment (no OMP_*
      variables unless --env sets them for both), under the exclusive lock
      $DYNG_SCRATCH/perf.lock. One untimed round per side comes first (it warms the page cache;
      the standard outputs must be byte-identical, and equal to the golden histogram when the
      corpus of parity/cycle_count_goldens.py is present). In every timed round the two
      histograms must be identical as well (a correctness guard: a difference exits non-zero).

      The regions are LOADED from parity/timed_regions/cycle_count.toml
      ([[reference.cycle_enum_openmp.region]]): "process wall time" is the harness's clock around
      the process (time.perf_counter around fork/exec/wait, the same on both sides), the
      original's update_seconds and the port's RESULT fields are parsed from standard error, and
      the port's profiler stages from its --timing CSV. Medians are compared region by region;
      "compute" gates are <= 1.05x (<= 1.10x when the original's median is below 10 ms, which
      then needs >= 20 runs, else the verdict is provisional), "end_to_end" gates <= 1.10x. The
      spread (max - min) / median above 10 % is flagged (PLAN 8.6). The contamination monitor
      (parity/contamination.py) records, for every timed process, the foreign CPU use of the
      machine (busy CPU time from /proc/stat minus the harness's own and its children's), and
      flags runs above 2 cores; load averages are recorded as well.
      --enforce-gates exits non-zero on an exceeded gate.

      Experiments (PLAN 8.6: improvements reported in their own table): --baseline-exe replaces
      side A by another binary, either a variant of the original (--baseline-kind original, e.g.
      parity/experiments/cycle_enum/build_variant.sh) or another dynG build
      (--baseline-kind port, e.g. the compat driver of an earlier commit). The record then names
      the baseline (--baseline-label) and its verdicts are not gates.

The default cases are the gate of acceptance criterion 4 of M2a: the static count DD k = 3..7,
GitHub and Twitch k = 3, 4 and the update 25K + 25K k = 4 (seed 1) on DD, GitHub and Twitch.
COLLAB k = 3 (about 45 s per process) is run with `--cases collab_k3 --runs 5`.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import contamination  # noqa: E402
import cycle_count_goldens as goldens  # noqa: E402
import perf_ab  # noqa: E402  (perf_lock, port_build, build_reference, portable_path, cpu_model)

REGION_MAP = REPO / "parity" / "timed_regions" / "cycle_count.toml"
MAP_KEY = "cycle_enum_openmp"
WALL = "process wall time"
ORIGINAL_REPORT = {
    key: re.compile(rf"^{key}=([0-9.eE+-]+)$", re.M)
    for key in ["update_seconds", "count_seconds", "read_seconds"]
}
PORT_REPORT = ["update_ms", "compute_ms", "read_ms", "build_ms", "prior_ms", "generate_ms"]
RESULT = re.compile(r"^RESULT task=(\w+) (.*)$", re.M)
SHORT_REGION_MS = 10.0
SHORT_REGION_RUNS = 20
DEFAULT_CASES = [
    "DD_k3",
    "DD_k4",
    "DD_k5",
    "DD_k6",
    "DD_k7",
    "github_k3",
    "github_k4",
    "twitch_k3",
    "twitch_k4",
    "DD_k4_25000_25000_s1",
    "github_k4_25000_25000_s1",
    "twitch_k4_25000_25000_s1",
]


def load_regions() -> list[dict]:
    doc = tomllib.loads(REGION_MAP.read_text())
    regions = doc["reference"][MAP_KEY]["region"]
    names = [r["name"] for r in regions]
    for required in ["static_end_to_end", "update"]:
        if required not in names:
            raise SystemExit(f"{REGION_MAP}: region '{required}' is missing")
    for r in regions:
        if r.get("task") not in ("count", "update"):
            raise SystemExit(f"{REGION_MAP}: region {r['name']}: task must be count or update")
        if r["gate"] not in ("compute", "end_to_end", "none"):
            raise SystemExit(f"{REGION_MAP}: region {r['name']}: unknown gate '{r['gate']}'")
        for key in r.get("original_report", []) + r.get("original_report_optional", []):
            if key != WALL and key not in ORIGINAL_REPORT:
                raise SystemExit(f"{REGION_MAP}: region {r['name']}: unknown original key {key}")
        for key in r.get("port_report", []):
            if key != WALL and key not in PORT_REPORT:
                raise SystemExit(f"{REGION_MAP}: region {r['name']}: unknown port key {key}")
        if r["gate"] != "none" and not (r.get("original_report") and r.get("port_report")):
            raise SystemExit(f"{REGION_MAP}: gated region {r['name']} needs both sides")
    return regions


def timed_run(cmd: list, env: dict) -> tuple[float, str, str, dict]:
    """(wall ms, stdout, stderr, contamination) of one process; the clock covers fork, exec and
    wait, the contamination monitor the same span."""
    with contamination.Monitor() as monitor:
        t0 = time.perf_counter()
        proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
        wall = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stderr}")
    return wall, proc.stdout, proc.stderr, monitor.result


def parse_original(wall: float, stderr: str) -> dict:
    values = {WALL: wall}
    for key, pattern in ORIGINAL_REPORT.items():
        m = pattern.search(stderr)
        if m:
            values[key] = float(m.group(1)) * 1000.0  # seconds -> ms
    return values


def parse_port(wall: float, stderr: str, timing: Path) -> tuple[dict, dict]:
    values = {WALL: wall}
    m = RESULT.search(stderr)
    if m is None:
        raise SystemExit(f"no RESULT line in the port's standard error:\n{stderr}")
    for item in m.group(2).split():
        key, _, value = item.partition("=")
        if key in PORT_REPORT:
            values[key] = float(value)
    stages: dict[str, float] = {}
    for row in csv.reader(timing.read_text().splitlines()[1:]):
        if len(row) == 3 and row[0] == "stage":
            stages[row[1]] = stages.get(row[1], 0.0) + float(row[2])
    return values, stages


def side_value(values: dict, keys: list[str]) -> float:
    total = 0.0
    for key in keys:
        if key not in values:
            raise SystemExit(f"the run reported no '{key}' ({sorted(values)})")
        total += values[key]
    return total


def spread(xs: list[float]) -> float:
    m = statistics.median(xs)
    return (max(xs) - min(xs)) / m if m > 0 else 0.0


def summarize(
    regions: list[dict], task: str, samples: dict, runs: int, baseline_kind: str = "original"
) -> list[dict]:
    out = []
    for r in regions:
        if r["task"] != task:
            continue
        e: dict = {"region": r["name"], "gate_kind": r["gate"]}
        a_keys = r.get("original_report") if baseline_kind == "original" else r.get("port_report")
        optional = r.get("original_report_optional", [])
        if baseline_kind == "original" and not a_keys and optional:
            # Reported only by an instrumented experiment copy (parity/experiments/cycle_enum).
            if all(all(k in s for k in optional) for s in samples["original"]):
                a_keys = optional
        if a_keys:
            a = [side_value(s, a_keys) for s in samples["original"]]
            e.update(
                original_ms=statistics.median(a), original_samples=a, original_spread=spread(a)
            )
        b = [side_value(s, r["port_report"]) for s in samples["port"]]
        e.update(port_ms=statistics.median(b), port_samples=b, port_spread=spread(b))
        if r.get("port"):
            st = [sum(s.get(n, float("nan")) for n in r["port"]) for s in samples["stages"]]
            e["port_stage_ms"] = statistics.median(st)
        if "original_ms" in e:
            e["ratio"] = e["port_ms"] / e["original_ms"] if e["original_ms"] > 0 else float("nan")
            if r["gate"] != "none":
                short = r["gate"] == "compute" and e["original_ms"] < SHORT_REGION_MS
                e["gate"] = 1.10 if (r["gate"] == "end_to_end" or short) else 1.05
                e["within_gate"] = e["ratio"] <= e["gate"]
                e["provisional"] = short and runs < SHORT_REGION_RUNS
        e["noisy"] = e.get("original_spread", 0.0) > 0.10 or e["port_spread"] > 0.10
        out.append(e)
    return out


def report(results: dict, baseline: str = "original") -> None:
    print(
        f"\n| case | region | {baseline} (ms) | dynG (ms) | ratio | gate | spread A / B |\n"
        "|---|---|---:|---:|---:|---|---|"
    )
    for case, res in results.items():
        for e in res["regions"]:
            if "gate" in e:
                verdict = "ok" if e["within_gate"] else "EXCEEDED"
                if e["provisional"]:
                    verdict += " (provisional: < 20 runs)"
                g = f"<= {e['gate']:.2f} {verdict}"
            else:
                g = "-"
            orig = f"{e['original_ms']:.1f}" if "original_ms" in e else "-"
            ratio = f"{e['ratio']:.3f}" if "ratio" in e else "-"
            sa = f"{e['original_spread'] * 100:.0f} %" if "original_spread" in e else "-"
            flag = " (noisy)" if e["noisy"] else ""
            print(
                f"| {case} | {e['region']} | {orig} | {e['port_ms']:.1f} | {ratio} | {g} | "
                f"{sa} / {e['port_spread'] * 100:.0f} %{flag} |"
            )
    print(
        "\n| case | foreign cores A: median / max | B: median / max | flagged runs A / B "
        f"(> {contamination.FLAG_CORES:.0f} cores) |\n|---|---|---|---|"
    )
    for case, res in results.items():
        c = res["contamination"]
        a, b = c["original"], c["port"]
        print(
            f"| {case} | {a['foreign_cores_median']:.2f} / {a['foreign_cores_max']:.2f} | "
            f"{b['foreign_cores_median']:.2f} / {b['foreign_cores_max']:.2f} | "
            f"{a['flagged_runs']} / {b['flagged_runs']} of {a['runs']} |"
        )


def run(args: argparse.Namespace) -> int:
    regions = load_regions()
    exe = args.exe.resolve()
    build = perf_ab.port_build(exe)
    if not build["parity_preset"] and not args.allow_non_parity_build:
        raise SystemExit(
            f"{exe} is not from a parity-preset build tree ({build}); pass "
            "--allow-non-parity-build for an experiment (its record says so)"
        )
    name = goldens.REFERENCE
    perf_ab.build_reference(name)
    ref = perf_ab.reference_copy(name)
    original = ref / "build" / "cycle-enum"
    marker = ref / ".dyng-reference"
    if args.baseline_exe:
        original = args.baseline_exe.resolve()
        if not original.is_file():
            raise SystemExit(f"--baseline-exe {original} does not exist")
    cases = goldens.select(goldens.all_cases(), ",".join(args.cases))
    env = dict(os.environ)
    for var in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC"]:
        env.pop(var, None)
    env.pop("GOMP_SPINCOUNT", None)
    for item in args.env:  # extra settings for BOTH sides
        key, _, value = item.partition("=")
        env[key] = value
    omp = ["--backend", "openmp", "--openmp-threads", str(args.threads)]
    golden_root = args.goldens
    results: dict = {}
    failures: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="dyng-perf-cc-", dir=perf_ab.SCRATCH / "runs"))
    timing = work / "timing.csv"
    timing_a = work / "timing-a.csv"
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for case in cases:
                task = "update" if case.update else "count"
                cli = [*case.cli_args(args.datasets), *omp]
                a_cmd = [original, *cli]
                if args.baseline_kind == "port":
                    a_cmd += ["--timing", timing_a]
                b_cmd = [exe, *cli, "--timing", timing]
                # Untimed first round: page cache, and the outputs must be identical.
                _, out_a, _, _ = timed_run(a_cmd, env)
                _, out_b, _, _ = timed_run(b_cmd, env)
                if out_a != out_b:
                    raise SystemExit(f"{case.rel}: the histograms differ:\n{out_a}\n{out_b}")
                golden = golden_root / case.rel / "histogram.csv"
                checked = golden.is_file()
                if checked and golden.read_text() != out_a:
                    raise SystemExit(f"{case.rel}: the histogram differs from the golden {golden}")
                print(
                    f"{case.rel}: histograms identical"
                    + (" and equal to the golden" if checked else ""),
                    flush=True,
                )
                samples: dict[str, list] = {"original": [], "port": [], "stages": []}
                loads = []
                foreign: dict[str, list] = {"original": [], "port": []}
                for r in range(args.runs):
                    before = os.getloadavg()[0]
                    wall_a, out_a, err_a, cont_a = timed_run(a_cmd, env)
                    wall_b, out_b, err_b, cont_b = timed_run(b_cmd, env)
                    loads.append((before, os.getloadavg()[0]))
                    foreign["original"].append(cont_a)
                    foreign["port"].append(cont_b)
                    if out_a != out_b:
                        failures.append(f"{case.rel} round {r + 1}: the histograms differ")
                    values_b, stages = parse_port(wall_b, err_b, timing)
                    if args.baseline_kind == "port":
                        samples["original"].append(parse_port(wall_a, err_a, timing_a)[0])
                    else:
                        samples["original"].append(parse_original(wall_a, err_a))
                    samples["port"].append(values_b)
                    samples["stages"].append(stages)
                    port_key = "update_ms" if case.update else WALL
                    key = (
                        port_key
                        if args.baseline_kind == "port"
                        else ("update_seconds" if case.update else WALL)
                    )
                    print(
                        f"{case.rel} round {r + 1}/{args.runs}: {key} {args.baseline_label} "
                        f"{samples['original'][-1][key]:.1f} ms, port "
                        f"{values_b[port_key]:.1f} ms (foreign cores "
                        f"{cont_a['foreign_cores']:.2f} / {cont_b['foreign_cores']:.2f})",
                        flush=True,
                    )
                stage_names = sorted({n for s in samples["stages"] for n in s})
                results[case.rel] = {
                    "task": task,
                    "golden_checked": checked,
                    "regions": summarize(regions, task, samples, args.runs, args.baseline_kind),
                    "port_stages_median_ms": {
                        n: statistics.median(s.get(n, 0.0) for s in samples["stages"])
                        for n in stage_names
                    },
                    "port_report_median_ms": {
                        k: statistics.median(s[k] for s in samples["port"])
                        for k in PORT_REPORT
                        if all(k in s for s in samples["port"])
                    },
                    "load_average": {
                        "min": min(min(p) for p in loads),
                        "max": max(max(p) for p in loads),
                    },
                    "contamination": {
                        side: {
                            **contamination.summarize(runs),
                            "foreign_cores_per_run": [round(r["foreign_cores"], 3) for r in runs],
                        }
                        for side, runs in foreign.items()
                    },
                }
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results, args.baseline_label)
    exceeded = [
        f"{c}: {e['region']} {e['ratio']:.3f} > {e['gate']:.2f}"
        for c, res in results.items()
        for e in res["regions"]
        if "gate" in e and not e["within_gate"]
    ]
    if args.json:
        write_json(args, results, build, ref, marker, regions)
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    if failures:
        return 1
    if exceeded:
        print("gate exceeded: " + "; ".join(exceeded))
        if args.enforce_gates and not args.baseline_exe:
            return 1
    return 0


def write_json(args, results, build, ref, marker, regions) -> None:
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = (
        subprocess.run(
            ["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", ".", ":(exclude)parity/results"]
        ).returncode
        != 0
    )
    doc = {
        "schema": 1,
        "algorithm": "cycle_count",
        "backend": "openmp",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": {
            "name": goldens.REFERENCE,
            "commit": goldens.COMMIT,
            "variant": "unpatched",
            "binary": perf_ab.portable_path(ref / "build" / "cycle-enum"),
            "build": marker.read_text() if marker.is_file() else None,
        },
        "baseline": (
            {
                "experiment": True,
                "label": args.baseline_label,
                "kind": args.baseline_kind,
                "binary": perf_ab.portable_path(args.baseline_exe),
                "note": "side A is this binary, not the unpatched original: the verdicts are "
                "not gates (PLAN 8.6: improvements reported separately)",
            }
            if args.baseline_exe
            else {"experiment": False, "label": "original", "kind": "original"}
        ),
        "port": {
            "commit": head + ("+dirty" if dirty else ""),
            "binary": perf_ab.portable_path(args.exe),
            "build": build,
        },
        "region_map": {
            "file": perf_ab.portable_path(REGION_MAP),
            "regions": [r["name"] for r in regions],
        },
        "protocol": {
            "runs": args.runs,
            "order": "A/B/A/B (original first), one untimed round per case first",
            "threads": args.threads,
            "backend": "openmp (--backend openmp --openmp-threads N on both)",
            "env": " ".join(args.env) or "no OMP_* variables (the libgomp defaults) on both",
            "lock": "perf.lock (exclusive)",
            "statistic": "median",
            "outputs": "standard output captured (the histogram CSV), compared every round",
            "short_regions": f"< {SHORT_REGION_MS} ms need >= {SHORT_REGION_RUNS} runs",
        },
        "host": {
            "cpu": perf_ab.cpu_model(),
            "logical_cpus": os.cpu_count(),
            "kernel": platform.release(),
        },
        "datasets_sha256": {
            g: goldens.sha256_file(args.datasets / p)
            for g, p in goldens.GRAPHS.items()
            if any(c.split("/")[1].startswith(f"{g}_") for c in results)
        },
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.json}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="perf_ab.py cycle_count", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exe", type=Path, required=True, help="dyng-compat-cycle-enum (parity preset)")
    r.add_argument("--cases", type=lambda s: [c for c in s.split(",") if c], default=DEFAULT_CASES)
    r.add_argument("--runs", type=int, default=11)
    r.add_argument("--threads", type=int, default=goldens.THREADS)
    r.add_argument("--datasets", type=Path, default=perf_ab.SCRATCH / "datasets" / "cycle")
    r.add_argument("--goldens", type=Path, default=perf_ab.SCRATCH / "goldens" / goldens.SET)
    r.add_argument("--json", type=Path)
    r.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="extra environment for both sides (repeatable)",
    )
    r.add_argument("--no-lock", action="store_true", help="the caller holds perf.lock")
    r.add_argument("--lock-timeout", type=float, default=3 * 3600.0, metavar="SECONDS")
    r.add_argument("--allow-non-parity-build", action="store_true")
    r.add_argument("--enforce-gates", action="store_true", help="exit 1 on an exceeded gate")
    r.add_argument(
        "--baseline-exe",
        type=Path,
        help="an experiment: this binary replaces the unpatched original as side A",
    )
    r.add_argument("--baseline-kind", choices=["original", "port"], default="original")
    r.add_argument("--baseline-label", default="original")
    args = parser.parse_args(argv)
    if args.runs < 5:
        parser.error("--runs must be >= 5 (PLAN Section 6.3 step 7)")
    if args.baseline_exe and args.baseline_label == "original":
        parser.error("--baseline-exe needs a --baseline-label naming the experiment")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
