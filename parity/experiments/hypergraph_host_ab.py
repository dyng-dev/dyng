#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""A dynG-against-dynG A/B of the hypergraph container's host apply and of triad_count's host
update (the refactor bar of ADR 0024; R030's D13 fix, ADR 0061). Not a PLAN 8.6 gate.

    parity/experiments/hypergraph_host_ab.py run --baseline-exe A --exe B
        [--cases coauth_b50k_s1,syn1m_b50k_s1] [--backends sequential,openmp:8]
        [--modes apply,update:h_motif] [--layouts 17] [--rounds 51] [--json FILE]

Both sides are `dyng-compat-escher run` (built from two commits) with the golden case's argv of
the triad_count_escher_gpu set (coauth-DBLP from $DYNG_SCRATCH/datasets/hyper/coauth.hg; the
synthetic inputs generated in-process from A's parameters and the seed), `--backend <b>` (and
`--threads N`), `--timing FILE`; mode `apply` adds `--no-count` (hypergraph::apply() alone,
timed by the driver's `apply_ms`), mode `update[:<pattern>]` runs triad_count's update of the
pattern (`--pattern`: h_motif, the default, or type1, type2, type3; the `triad_count.update`
stage and the `hypergraph.apply` stage inside it). Rounds alternate A/B; round
r gives both sides a --timing file name 8 * (r mod --layouts) characters longer (ADR 0024 rule
1: the heap layouts); a round whose window the contamination monitor (perf_ab.MachineMonitor,
foreign CPU load) flags is repeated, at most --rounds times in all. A guard: every run's stdout
(the batch lines and the counts) equals the first run's, on both sides.

The record has the regions of each (case, backend, mode) per batch (`<stage>[b<N>]`) and summed
over the batches (`<stage>`), and the build before them (`hypergraph.build`, which builds the line
graph), with the samples of both sides as `original_samples` (A, the
baseline) and `port_samples` (B), so `parity/ab_modes.py --bar 1.02 FILE` reads it (ADR 0024 rules
2 and 3). Run it under the EXCLUSIVE perf lock (`flock $DYNG_SCRATCH/perf.lock ...`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "parity"))

import perf_ab  # noqa: E402

GOLDENS = perf_ab.SCRATCH / "goldens" / "triad_count" / "escher_gpu" / "cases"
COAUTH = perf_ab.SCRATCH / "datasets" / "hyper" / "coauth.hg"
# The stages each mode reads: the driver's timer and the profiler's stages.
STAGES = {
    "apply": ("apply_ms", "hypergraph.apply", "hypergraph.apply.insert"),
    "update": ("update_ms", "triad_count.update", "hypergraph.apply", "hypergraph.apply.insert"),
}
# Read once, before the batches: the build (the line graph's row build among it).
BUILD_STAGES = ("hypergraph.build",)


def case_argv(case: str) -> list[str]:
    """The golden case's argv (A's arguments without the program name), coauth-DBLP resolved."""
    group = GOLDENS / case / "group.json"
    if not group.is_file():
        raise SystemExit(f"no golden case {group} (parity/export_goldens.py triad_count)")
    argv = json.loads(group.read_text())["argv"][1:]
    return [str(COAUTH) if a == "<input:coauth>" else a for a in argv]


def backend_args(spec: str) -> list[str]:
    name, _, threads = spec.partition(":")
    return ["--backend", name, *(["--threads", threads] if threads else [])]


def stage_sums(timing: Path, stages: tuple[str, ...]) -> dict[str, float]:
    """Per stage: the ms of every batch (`<stage>[b<N>]`) and their sum (`<stage>`)."""
    doc = json.loads(timing.read_text())
    out: dict[str, float] = {}
    for rec in doc["stages"]:
        if rec["stage"] in BUILD_STAGES and rec["batch"] == 0:
            out[rec["stage"]] = out.get(rec["stage"], 0.0) + rec["ms"]
        if rec["stage"] in stages and rec["batch"] > 0:
            key = f"{rec['stage']}[b{rec['batch']}]"
            out[key] = out.get(key, 0.0) + rec["ms"]
            out[rec["stage"]] = out.get(rec["stage"], 0.0) + rec["ms"]
    return out


def run(args: argparse.Namespace) -> int:
    sides = {"A": Path(args.baseline_exe).resolve(), "B": Path(args.exe).resolve()}
    for exe in sides.values():
        if not exe.is_file():
            raise SystemExit(f"no executable {exe}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMP_")}
    record: dict = {
        "tool": "parity/experiments/hypergraph_host_ab.py",
        "started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "protocol": {
            "rule": "ADR 0024 (layouts, modes, the bar)",
            "layouts": args.layouts,
            "rounds": args.rounds,
            "alternation": "A/B per round",
            "lock": "exclusive perf lock (the caller's flock)",
        },
        "sides": {k: str(v) for k, v in sides.items()},
        "machine": {"cpu": platform.processor(), "load_start": os.getloadavg()},
        "results": {},
    }
    with (
        tempfile.TemporaryDirectory(prefix="dyng-hab-") as tmp,
        perf_ab.MachineMonitor(None) as monitor,
    ):
        for case in args.cases.split(","):
            argv = case_argv(case)
            for backend in args.backends.split(","):
                for mode in args.modes.split(","):
                    key = f"{case}/{backend}/{mode}"
                    kind, _, pattern = mode.partition(":")
                    if kind not in STAGES:
                        raise SystemExit(f"--modes: {mode} is not apply or update[:<pattern>]")
                    extra = (
                        ["--no-count"]
                        if kind == "apply"
                        else (["--pattern", pattern] if pattern else [])
                    )
                    samples: dict[str, dict[str, list[float]]] = {"A": {}, "B": {}}
                    windows = []
                    expected = None
                    done = repeated = 0
                    while done < args.rounds:
                        if repeated > args.rounds:
                            raise SystemExit(f"{key}: too many contaminated rounds")
                        timing = Path(tmp) / ("t" + "x" * (8 * (done % args.layouts)) + ".json")
                        got: dict[str, dict[str, float]] = {}
                        contaminated = False
                        for side, exe in sides.items():
                            cmd = [
                                exe,
                                "run",
                                *argv,
                                *backend_args(backend),
                                *extra,
                                "--timing",
                                timing,
                            ]
                            out, window = monitor.run(cmd, env)
                            lines = [line for line in out.splitlines() if "ms" not in line]
                            if expected is None:
                                expected = lines
                            elif lines != expected:
                                raise SystemExit(f"{key}: side {side}'s output differs (guard)")
                            got[side] = stage_sums(timing, STAGES[kind])
                            windows.append({"round": done, "side": side, **window})
                            contaminated = contaminated or window["contaminated"]
                        if contaminated:
                            repeated += 1
                            continue
                        for side in sides:
                            for region, ms in got[side].items():
                                samples[side].setdefault(region, []).append(ms)
                        done += 1
                    regions = [
                        {
                            "region": region,
                            "original_samples": samples["A"][region],
                            "port_samples": samples["B"][region],
                            "original_ms": statistics.median(samples["A"][region]),
                            "port_ms": statistics.median(samples["B"][region]),
                            "ratio": statistics.median(samples["B"][region])
                            / statistics.median(samples["A"][region]),
                        }
                        for region in sorted(samples["A"])
                    ]
                    record["results"][key] = {
                        "argv": argv,
                        "backend": backend,
                        "mode": mode,
                        "regions": regions,
                        "repeated_rounds": repeated,
                        "windows": windows,
                    }
                    summary = ", ".join(
                        f"{r['region']} {r['ratio']:.3f}" for r in regions if "[" not in r["region"]
                    )
                    print(f"{key}: {summary} ({repeated} repeated)", flush=True)
                    if args.json:
                        Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    record["finished"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    record["machine"]["load_end"] = os.getloadavg()
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="the A/B rounds")
    p.add_argument("--baseline-exe", required=True, help="side A: dyng-compat-escher (baseline)")
    p.add_argument("--exe", required=True, help="side B: dyng-compat-escher (the change)")
    p.add_argument("--cases", default="coauth_b50k_s1,syn1m_b50k_s1")
    p.add_argument("--backends", default="sequential,openmp:8")
    p.add_argument("--modes", default="apply,update:h_motif")
    p.add_argument("--layouts", type=int, default=17)
    p.add_argument("--rounds", type=int, default=51)
    p.add_argument("--json", help="the record")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
