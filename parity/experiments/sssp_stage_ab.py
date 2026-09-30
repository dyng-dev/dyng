#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage A/B of two dyng-compat-mosp builds on the OpenMP backend (an experiment, not a
gate; the M3 review, parity/results/M3.md section 5).

    flock $DYNG_SCRATCH/perf.lock parity/experiments/sssp_stage_ab.py \\
        --a <earlier build>/tools/compat/dyng-compat-mosp \\
        --b build/parity/tools/compat/dyng-compat-mosp \\
        --graph rgg --batch local10k --runs 41 --json <record>

Both builds run alternately (A, B, A, B, ...) on the inputs of `parity/perf_ab.py prepare`, with
the OpenMP settings of perf_ab.py (28 threads, OMP_PROC_BIND=close, OMP_PLACES=cores), their
output discarded and the parent idle while they run (no machine monitor: take the exclusive
perf lock and check the machine yourself). Every process writes its profiler stages
(--timing); per objective, the gated stages of the per-objective update (sssp.identify_affected,
sssp.seed, sssp.loop, sssp.finalize) are split at the largest gap of their sums, as the "modes"
of M3.md section 4.2, and compared stage by stage within each mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
BATCHES = {
    "local10k": "changes_local_10000_50_safe",
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
}
GATED = ["sssp.identify_affected", "sssp.seed", "sssp.loop", "sssp.finalize"]


def run_once(exe: str, common: list, env: dict, timing: Path) -> dict[str, list[float]]:
    cmd = [exe, *common, "--timing", timing]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
    stages: dict[str, list[float]] = {}
    for row in csv.reader(timing.read_text().splitlines()[1:]):
        if row and row[0] == "stage":
            stages.setdefault(row[1], []).append(float(row[2]))
    return stages


def split(values: list[float]) -> float | None:
    """The threshold between two modes (the largest gap, if above 4 % of the median)."""
    s = sorted(values)
    gap, i = max((s[j + 1] - s[j], j) for j in range(len(s) - 1))
    return s[i] + gap / 2 if gap > 0.04 * statistics.median(s) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, help="side A: dyng-compat-mosp")
    parser.add_argument("--b", required=True, help="side B: dyng-compat-mosp")
    parser.add_argument("--graph", default="rgg")
    parser.add_argument("--batch", default="local10k", choices=sorted(BATCHES))
    parser.add_argument("--runs", type=int, default=41)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    data = SCRATCH / "bench" / "mosp" / args.graph
    common = [
        "--graph", data / "csr" / "graphCsr", "--changes", data / BATCHES[args.batch],
        "--init", data / "init", "--no-output",
    ]  # fmt: skip
    env = dict(os.environ, OMP_NUM_THREADS="28", OMP_PROC_BIND="close", OMP_PLACES="cores")
    for var in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_DYNAMIC"]:
        env.pop(var, None)
    timing = args.json.with_suffix(".timing.csv")
    rounds: dict[str, list] = {"A": [], "B": []}
    for _ in range(args.runs):
        for side, exe in (("A", args.a), ("B", args.b)):
            rounds[side].append(run_once(exe, common, env, timing))
    timing.unlink(missing_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {"a": args.a, "b": args.b, "graph": args.graph, "batch": args.batch, "rounds": rounds}
        )  # fmt: skip
    )
    objectives = len(rounds["A"][0]["sssp.loop"])
    for o in range(objectives):
        total = {s: [sum(p[g][o] for g in GATED if g in p) for p in rounds[s]] for s in rounds}
        t = split(total["A"] + total["B"])
        slow = {s: sum(1 for x in total[s] if t is not None and x >= t) for s in total}
        ratio = statistics.median(total["B"]) / statistics.median(total["A"])
        print(f"objective {o}: slow rounds A {slow['A']}/{args.runs} B {slow['B']}/{args.runs}; "
              f"median B/A {ratio:.3f}")  # fmt: skip
        for mode in ["one"] if t is None else ["fast", "slow"]:
            fast = mode == "fast"
            ia, ib = (
                [j for j, x in enumerate(total[side]) if mode == "one" or (x < t) == fast]
                for side in ("A", "B")
            )
            if not ia or not ib:
                continue
            parts = []
            for g in GATED:
                va = statistics.median(rounds["A"][j][g][o] for j in ia)
                vb = statistics.median(rounds["B"][j][g][o] for j in ib)
                parts.append(f"{g.split('.')[1]} {va:.3f}/{vb:.3f} ({vb / va:.3f})")
            ga = statistics.median(total["A"][j] for j in ia)
            gb = statistics.median(total["B"][j] for j in ib)
            print(f"  {mode} (A {len(ia)}, B {len(ib)}): gated {ga:.3f}/{gb:.3f} "
                  f"({gb / ga:.3f}); " + "; ".join(parts))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
