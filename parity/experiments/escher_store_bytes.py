#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The escher store's device bytes of two builds (M11a criterion 20: M8's store_stats().device_bytes
byte-identical to the merge base). Not a timed reading.

    parity/experiments/escher_store_bytes.py --baseline-exe A --exe B
        [--cases coauth_b50k_s1,syn1m_b50k_s1,syn2m_b50k_s1] [--gpu 1] [--json FILE]

Both sides are `dyng-compat-escher run` with the golden case's argv of the triad_count_escher_gpu
set, `--backend cuda --no-count` (the container alone, escher storage): its "[memory]" lines, the
store's store_stats().device_bytes after the build and after every batch (the accounting the
store_memory gates of 0.3 read), must be equal byte for byte, and its batch lines (the ids and
rows the container printed) too. Run it under the shared lock on the development GPU, after
tools/locklog.py quiet-ok.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "parity"))
sys.path.insert(0, str(REPO / "parity" / "experiments"))

import hypergraph_host_ab as hab  # noqa: E402


def run(exe: Path, argv: list[str], env: dict) -> tuple[list[str], list[str]]:
    """(the [memory] lines, the other output lines without timings) of one process."""
    cmd = [str(exe), "run", *argv, "--backend", "cuda", "--no-count"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{exe} exited {p.returncode}:\n{p.stderr[-1500:]}")
    memory = [x for x in p.stderr.splitlines() if x.startswith("[memory]")]
    lines = [x for x in p.stdout.splitlines() if "ms" not in x]
    return memory, lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline-exe", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--baseline-commit", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--cases", default="coauth_b50k_s1,syn1m_b50k_s1,syn2m_b50k_s1")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMP_")}
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CUDA_MODULE_LOADING"] = "EAGER"
    sides = {
        "A": hab.side_record(args.baseline_exe.resolve(), args.baseline_commit),
        "B": hab.side_record(args.exe.resolve(), args.commit),
    }
    results = {}
    for case in args.cases.split(","):
        a_mem, a_lines = run(args.baseline_exe, hab.case_argv(case), env)
        b_mem, b_lines = run(args.exe, hab.case_argv(case), env)
        results[case] = {
            "memory_lines": {"A": a_mem, "B": b_mem},
            "device_bytes_identical": a_mem == b_mem and bool(a_mem),
            "batch_lines_identical": a_lines == b_lines,
            "batch_lines_sha256": hashlib.sha256("\n".join(b_lines).encode()).hexdigest(),
        }
        r = results[case]
        print(f"{case}: device bytes {'identical' if r['device_bytes_identical'] else 'DIFFER'} "
              f"({len(a_mem)} points); batch lines "
              f"{'identical' if r['batch_lines_identical'] else 'DIFFER'}")  # fmt: skip
    ok = all(r["device_bytes_identical"] and r["batch_lines_identical"] for r in results.values())
    if args.json:
        doc = {
            "record": "the escher store's store_stats().device_bytes of the final library against "
            "the merge base (M11a criterion 20)",
            "tool": "parity/experiments/escher_store_bytes.py",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gpu": args.gpu,
            "sides": sides,
            "results": results,
            "identical": ok,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
