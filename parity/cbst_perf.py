#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Device memory of the ESCHER CBST store against ESCHER-GPU@abcf9b2 (M8a, acceptance criterion
15; PLAN 6.4.5, 8.6).

    parity/perf_ab.py cbst memory [--gpu 0] [--json parity/results/M8a-memory.json]

(or `parity/cbst_perf.py memory ...`). The author's decision of 2026-10-07: the memory gate of M8
is read as STORE ACCOUNTING (each tree's device_bytes() against ESCHER-GPU's deviceBytes() on
coauth-DBLP after the build and after each of 3 x 50K batches, seed 1: the test
dyng_cbst_memory_tests, whose table this command records with --accounting); the Nsight Systems
peak of the live device allocations of both processes is recorded but not gated in M8 (it is in
M9). The two processes, each under `nsys profile --cuda-memory-usage=true` on GPU --gpu (memory
kind Device: cudaMalloc and cudaMallocAsync):

- the original: the UNPATCHED copy's `build/main --input coauth.hg --batches 3 --batch-size 50000
  --seed 1 --check` (the command of the golden set escher_gpu_coauth50k; the whole program:
  the CBST store, its row views and the motif counting);
- the port: the store replay of the same run (dyng_cbst_memory_tests: every operation of the
  trace on dynG stores under escher_gpu(), the H2H rebuilds with arrays of the recorded sizes).

Under the EXCLUSIVE perf lock ($DYNG_SCRATCH/perf.lock), as every measurement.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import cycle_count_perf  # noqa: E402
import perf_ab  # noqa: E402

SET = "escher_gpu_coauth50k"
REFERENCE = "ESCHER-GPU"


def find_tests(build: Path | None) -> Path:
    candidates = [build] if build else [REPO / "build" / p for p in ("parity-cuda", "release-cuda")]
    for b in candidates:
        exe = b / "cpp" / "tests" / "dyng_cbst_memory_tests"
        if exe.is_file():
            return exe
    raise SystemExit(
        "dyng_cbst_memory_tests not found: build it in a Release CUDA tree (cmake --build "
        "build/parity-cuda --target dyng_cbst_memory_tests) or pass --build"
    )


def golden_case() -> tuple[Path, list[str]]:
    root = perf_ab.SCRATCH / "goldens" / "cbst_trace" / SET
    for case in sorted(root.glob("*/case.json")):
        argv = json.loads(case.read_text())["argv"]
        return case.parent, argv
    raise SystemExit(f"no golden case under {root} (parity/export_goldens.py cbst_trace {SET})")


def memory(args: argparse.Namespace) -> int:
    tests = find_tests(args.build).resolve()
    _, argv = golden_case()
    ref = perf_ab.reference_copy(REFERENCE)  # the unpatched copy
    main = ref / "build" / "main"
    if not main.is_file():
        raise SystemExit(f"{main} missing: parity/build_reference.sh {REFERENCE}")
    dataset = perf_ab.SCRATCH / "datasets" / "hyper"
    cmd = [str(main)] + [a.replace("<abs>/coauth.hg", str(dataset / "coauth.hg")) for a in argv[1:]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_MODULE_LOADING="EAGER")
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-mem-cbst-", dir=perf_ab.SCRATCH / "runs"))
    accounting = work / "accounting.json"
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            original = cycle_count_perf.device_memory(cmd, env, work / "a")
            port_env = dict(env, DYNG_CBST_MEMORY_JSON=str(accounting))
            port = cycle_count_perf.device_memory(
                [str(tests), "--gtest_filter=CbstEarlyMemory.*"], port_env, work / "b"
            )
        table = json.loads(accounting.read_text())["readings"] if accounting.is_file() else []
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    worst = max((r["ratio"] for r in table), default=float("nan"))
    print(
        f"{SET}: nsys peak of live device allocations: original {original['peak_live_mib']} MiB, "
        f"store replay {port['peak_live_mib']} MiB (pool {port['pool_mib']} MiB); store "
        f"accounting: {len(table)} readings, worst device_bytes / deviceBytes() {worst:.4f}"
    )
    for r in table:
        print(
            f"  {r['tree']:4} {r['point']:16} {r['op']:44} A {r['escher_gpu_device_bytes']:>11} "
            f"dynG {r['dyng_device_bytes']:>11} held {r['dyng_allocated_bytes']:>11} "
            f"ratio {r['ratio']:.4f}"
        )
    if args.json:
        doc = {
            "schema": 1,
            "what": "M8a criterion 15: the store accounting (gated, <= 1.05x per tree and point) "
            "and the Nsight Systems peak of live device allocations of both processes "
            "(recorded, not gated in M8; the author's decision of 2026-10-07)",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "set": SET,
            "original": {
                "command": [perf_ab.portable_path(c) if "/" in c else c for c in cmd],
                "nsys": original,
            },
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(tests),
                "nsys": port,
            },
            "gpu": args.gpu,
            "driver": perf_ab.driver_versions(),
            "accounting": table,
            "worst_ratio": worst,
            "passed": bool(table) and worst <= 1.05,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0 if table and worst <= 1.05 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("memory", help="the store accounting and the nsys peaks (criterion 15)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--build", type=Path, help="the CUDA build tree of dyng_cbst_memory_tests")
    p.add_argument("--json", type=Path)
    p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
    p.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)
    return memory(args)


if __name__ == "__main__":
    sys.exit(main())
