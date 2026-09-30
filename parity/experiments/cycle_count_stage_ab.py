#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage A/B of two dyng-compat-cycle-enum builds on one update case (an experiment, not
a gate; the M3 acceptance fixes, parity/results/M3.md section 6).

    flock $DYNG_SCRATCH/perf.lock parity/experiments/cycle_count_stage_ab.py \\
        --a <earlier build>/tools/compat/dyng-compat-cycle-enum \\
        --b build/parity-cuda/tools/compat/dyng-compat-cycle-enum \\
        --case DD_k4_25000_25000_s1 --backend cuda --scope resident --runs 25 --json <record>

Both builds run alternately (A, B, A, B, ...) on the case of parity/cycle_count_goldens.py, with
their output discarded and the parent idle while they run (no machine monitor and no clock lock:
take the exclusive perf lock and check the machine yourself). Every process writes its profiler
stages (--timing); the stages of the timed update (the rows after the prior's
`cycle_count.compute`) are compared by their medians, a stage opened twice per update (under set
semantics `cycle_count.normalize`: the framework's Step 0, then the hook) as `<name>#2`, next to
the RESULT line's update_ms (the gated region).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cycle_count_goldens  # noqa: E402


def run_once(exe: str, args: list, env: dict, timing: Path) -> dict[str, float]:
    proc = subprocess.run(
        [exe, *args, "--timing", timing], env=env, check=True, capture_output=True, text=True
    )
    stages: dict[str, float] = {}
    seen: dict[str, int] = {}
    in_update = False
    for row in csv.reader(timing.read_text().splitlines()[1:]):
        if not row or row[0] != "stage":
            continue
        if row[1] == "cycle_count.compute":
            in_update = True  # the prior's count ends here; the timed update follows
            continue
        if not in_update:
            continue
        seen[row[1]] = seen.get(row[1], 0) + 1
        name = row[1] if seen[row[1]] == 1 else f"{row[1]}#{seen[row[1]]}"
        stages[name] = float(row[2])
    for line in proc.stderr.splitlines():
        if line.startswith("RESULT"):
            fields = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            stages["RESULT.update_ms"] = float(fields["update_ms"])
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, help="side A: dyng-compat-cycle-enum")
    parser.add_argument("--b", required=True, help="side B: dyng-compat-cycle-enum")
    parser.add_argument("--case", default="DD_k4_25000_25000_s1")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "openmp"])
    parser.add_argument("--scope", default="resident", choices=["original", "resident"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--datasets", type=Path, default=SCRATCH / "datasets" / "cycle")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    case = cycle_count_goldens.select(cycle_count_goldens.all_cases(), args.case)[0]
    if not case.update:
        parser.error(f"{args.case} is not an update case")
    cli = [str(a) for a in case.cli_args(args.datasets)]
    env = dict(os.environ)
    if args.backend == "cuda":
        cli += ["--backend", "cuda", "--scope", args.scope]
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_MODULE_LOADING="EAGER")
    else:
        cli += ["--backend", "openmp", "--openmp-threads", "56"]
    timing = args.json.with_suffix(".timing.csv")
    rounds: dict[str, list] = {"A": [], "B": []}
    for _ in range(args.runs):
        for side, exe in (("A", args.a), ("B", args.b)):
            rounds[side].append(run_once(exe, cli, env, timing))
    timing.unlink(missing_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    record = {"a": args.a, "b": args.b, "case": args.case, "backend": args.backend,
              "scope": args.scope, "rounds": rounds}  # fmt: skip
    args.json.write_text(json.dumps(record) + "\n")
    names = list(dict.fromkeys(k for p in rounds["A"] + rounds["B"] for k in p))
    print(f"{'stage':34s} {'A (ms)':>9s} {'B (ms)':>9s} {'B - A':>8s} {'B / A':>6s}")
    for name in names:
        med = {s: [p[name] for p in rounds[s] if name in p] for s in rounds}
        a = statistics.median(med["A"]) if med["A"] else None
        b = statistics.median(med["B"]) if med["B"] else None
        text = f"{name:34s} {a if a is not None else float('nan'):9.4f} "
        text += f"{b if b is not None else float('nan'):9.4f}"
        if a is not None and b is not None:
            text += f" {b - a:+8.4f} {b / a if a else float('nan'):6.3f}"
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
