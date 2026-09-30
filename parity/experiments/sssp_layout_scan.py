#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The heap-layout scan of two dyng-compat-mosp builds on the OpenMP backend (an experiment, not a
gate; the M3 acceptance fixes, parity/results/M3.md section 6.3).

    flock $DYNG_SCRATCH/perf.lock parity/experiments/sssp_layout_scan.py \\
        --a <earlier build>/tools/compat/dyng-compat-mosp \\
        --b build/parity/tools/compat/dyng-compat-mosp \\
        --graph roadNet-CA --batch local10k --reps 5 --json <record>

The only difference between the layouts is the length of the --timing file name (0, 8, ..., 128
extra characters): the program keeps the name on its heap, so every later allocation moves. At
each layout both builds run alternately (A, B, A, B, ...) --reps times with the OpenMP settings of
perf_ab.py, their output discarded and the parent idle (take the exclusive perf lock and check
the machine yourself). Per objective the gated stages of the per-objective update
(sssp.identify_affected, sssp.seed, sssp.loop, sssp.finalize) are summed; the ratio of medians is
printed per layout, with the geometric mean over the layouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
BATCHES = {
    "local10k": "changes_local_10000_50_safe",
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
}
GATED = ["sssp.identify_affected", "sssp.seed", "sssp.loop", "sssp.finalize"]


def gated(timing: Path) -> list[float]:
    stages: dict[str, list[float]] = {}
    for row in csv.reader(timing.read_text().splitlines()[1:]):
        if row and row[0] == "stage":
            stages.setdefault(row[1], []).append(float(row[2]))
    objectives = len(stages["sssp.loop"])
    return [sum(stages[g][o] for g in GATED if g in stages) for o in range(objectives)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, help="side A: dyng-compat-mosp")
    parser.add_argument("--b", required=True, help="side B: dyng-compat-mosp")
    parser.add_argument("--graph", default="roadNet-CA")
    parser.add_argument("--batch", default="local10k", choices=sorted(BATCHES))
    parser.add_argument("--reps", type=int, default=5, help="rounds per layout")
    parser.add_argument("--layouts", type=int, default=17)
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
    work = Path(tempfile.mkdtemp(prefix="dyng-layout-", dir=SCRATCH / "runs"))
    record: dict = {"a": args.a, "b": args.b, "graph": args.graph, "batch": args.batch}
    record["layouts"] = []
    try:
        for layout in range(args.layouts):
            timing = work / ("timing" + "x" * (8 * layout) + ".csv")
            samples: dict[str, list] = {"A": [], "B": []}
            for _ in range(args.reps):
                for side, exe in (("A", args.a), ("B", args.b)):
                    cmd = [str(c) for c in [exe, *common, "--timing", timing]]
                    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
                    samples[side].append(gated(timing))
            record["layouts"].append({"extra_chars": 8 * layout, "samples": samples})
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record) + "\n")
    objectives = len(record["layouts"][0]["samples"]["A"][0])
    logs: list[list[float]] = [[] for _ in range(objectives)]
    for entry in record["layouts"]:
        parts = []
        for o in range(objectives):
            a = statistics.median(s[o] for s in entry["samples"]["A"])
            b = statistics.median(s[o] for s in entry["samples"]["B"])
            logs[o].append(math.log(b / a))
            parts.append(f"obj{o} {a:.2f}/{b:.2f} = {b / a:.3f}")
        print(f"+{entry['extra_chars']:3d} chars: " + "; ".join(parts))
    for o in range(objectives):
        ratios = [math.exp(x) for x in logs[o]]
        print(f"objective {o}: geometric mean of the per-layout ratios "
              f"{math.exp(statistics.mean(logs[o])):.3f} (per layout {min(ratios):.3f}-"
              f"{max(ratios):.3f})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
