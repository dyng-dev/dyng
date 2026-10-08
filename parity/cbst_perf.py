#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Device memory of the ESCHER CBST store against ESCHER-GPU@abcf9b2 (M8a, acceptance criterion
15; PLAN 6.4.5, 8.6), and the early-warning table of M8a step 5.

    parity/perf_ab.py cbst memory [--gpu 0] [--preset escher_gpu|dyng_default]
                                  [--json parity/results/M8a-memory.json]
    parity/perf_ab.py cbst early [--gpu 0] [--json parity/results/M8a-early-warning.json]

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

`early` (reported, not gated; M8a step 5): both processes once more under `nsys profile -t cuda`
(no memory trace). Both bracket every store operation with cudaDeviceSynchronize (A's
ScopedTimer stages maintain.h2v_delete, v2h_unfill, h2v_insert and v2h_fill; the replay after
every operation), so the timeline splits into one segment per operation; a segment is classed by
the kernels it ran (erase: the delete kernels; unfill: unfill_*; insert: the best-fit and rebuild
kernels; fill: insert_node_* / allocate_space). Per class: the operations, the device time of
their kernels per operation, the blocking runtime calls per operation (cudaMemcpy and
cudaMemcpyToSymbol without Async, cudaStreamSynchronize, cudaEventSynchronize; the brackets
excluded) and the allocation calls per operation (cudaMalloc, cudaMallocAsync, cudaMallocHost,
cudaHostAlloc); and the registers per thread of the hot kernels (CUPTI's kernel records).

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
    cmd, env = original_command(argv, args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-mem-cbst-", dir=perf_ab.SCRATCH / "runs"))
    accounting = work / "accounting.json"
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            original = cycle_count_perf.device_memory(cmd, env, work / "a")
            port_env = dict(env, DYNG_CBST_MEMORY_JSON=str(accounting))
            port_env["DYNG_CBST_MEMORY_PRESET"] = args.preset
            port = cycle_count_perf.device_memory(
                [str(tests), "--gtest_filter=CbstEarlyMemory.*"], port_env, work / "b"
            )
        table = json.loads(accounting.read_text())["readings"] if accounting.is_file() else []
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    return report_memory(args, cmd, tests, original, port, table)


def original_command(argv: list[str], gpu: int) -> tuple[list[str], dict]:
    """ESCHER-GPU's command of the golden set (the unpatched copy) and its environment."""
    ref = perf_ab.reference_copy(REFERENCE)  # the unpatched copy
    main = ref / "build" / "main"
    if not main.is_file():
        raise SystemExit(f"{main} missing: parity/build_reference.sh {REFERENCE}")
    dataset = perf_ab.SCRATCH / "datasets" / "hyper"
    cmd = [str(main)] + [a.replace("<abs>/coauth.hg", str(dataset / "coauth.hg")) for a in argv[1:]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return cmd, env


def report_memory(args, cmd, tests, original, port, table) -> int:
    worst = max((r["ratio"] for r in table), default=float("nan"))
    print(
        f"{SET} ({args.preset}): nsys peak of live device allocations: original "
        f"{original['peak_live_mib']} MiB, "
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
            "preset": args.preset,
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


# The operation classes of the early-warning table: the kernels that mark a segment (A's names and
# dynG's, parity/escher_port/rename_map.toml), tried in this order.
OP_CLASSES = [
    (
        "h2v insert",
        {"locateReusableSlots", "computeItemSizes", "applyReuse", "rankOrderNodes"},
        {
            "locate_reusable_slots_kernel",
            "compute_item_sizes_kernel",
            "apply_reuse_thread_kernel",
            "rank_order_nodes_kernel",
        },
    ),
    ("h2v erase", {"locateDeleteTargets"}, {"locate_delete_targets_kernel"}),
    (
        "v2h unfill",
        {"unfill_thread", "unfill_warp", "unfill_block"},
        {"unfill_thread_kernel", "unfill_warp_kernel", "unfill_block_kernel"},
    ),
    (
        "v2h fill",
        {"insertNode_thread", "insertNode_warp", "insertNode_block", "computeNextMultipleOf4"},
        {
            "insert_node_thread_kernel",
            "insert_node_warp_kernel",
            "insert_node_block_kernel",
            "compute_next_multiple_of_4_kernel",
        },
    ),
]
# The hot kernels whose registers are reported (A's name -> dynG's).
HOT_KERNELS = {
    "insertNode_thread": "insert_node_thread_kernel",
    "insertNode_warp": "insert_node_warp_kernel",
    "insertNode_block": "insert_node_block_kernel",
    "applyReuse": "apply_reuse_thread_kernel",
    "applyReuse_warp": "apply_reuse_warp_kernel",
    "applyReuse_block": "apply_reuse_block_kernel",
    "unfill_thread": "unfill_thread_kernel",
    "unfill_warp": "unfill_warp_kernel",
    "unfill_block": "unfill_block_kernel",
    "locateDeleteTargets": "locate_delete_targets_kernel",
    "applyDeletes": "apply_deletes_kernel",
    "reduceAvailLevel": "reduce_avail_level_kernel",
    "allocateSpace": "allocate_space_kernel",
    "fixupOverflowMetadata": "fixup_overflow_metadata_kernel",
}
BLOCKING = {
    "cudaMemcpy",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemset",
}
ALLOCATING = {
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaMallocHost",
    "cudaHostAlloc",
    "cudaMallocManaged",
}


def runtime_name(name: str) -> str:
    """cudaMemcpy_v3020 -> cudaMemcpy (the runtime API's versioned names in the CUPTI records)."""
    return name.split("_v")[0] if "_v" in name else name


def segments(cmd: list[str], env: dict, report: Path, dyng_side: bool) -> dict:
    """The operations of one process under nsys (see the module description): per class the
    operations, kernel time, blocking and allocating calls; and every kernel's registers."""
    import sqlite3

    nsys = os.environ.get("NSYS", "/usr/local/cuda-13.1/bin/nsys")
    subprocess.run(
        [
            nsys,
            "profile",
            "-t",
            "cuda",
            "-o",
            str(report),
            "-f",
            "true",
            "--export=sqlite",
            *map(str, cmd),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    database = Path(str(report) + ".sqlite")
    try:
        db = sqlite3.connect(str(database))
        strings = dict(db.execute("select id, value from StringIds").fetchall())
        kernels = [
            (start, end, strings.get(name, "?"), regs)
            for start, end, name, regs in db.execute(
                "select start, end, shortName, registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL"
            )
        ]
        calls = [
            (start, end, runtime_name(strings.get(name, "?")))
            for start, end, name in db.execute(
                "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        ]
        db.close()
    finally:
        for f in [database, Path(str(report) + ".nsys-rep")]:
            f.unlink(missing_ok=True)
    calls.sort()
    kernels.sort()
    syncs = [c for c in calls if c[2] == "cudaDeviceSynchronize"]
    bounds = [0] + [c[1] for c in syncs] + [float("inf")]
    starts = [0] + [c[0] for c in syncs] + [float("inf")]
    import bisect

    seg_kernels: dict[int, list] = {}
    for k in kernels:
        seg_kernels.setdefault(bisect.bisect_left(bounds, k[0]), []).append(k)
    seg_calls: dict[int, list] = {}
    for c in calls:
        i = bisect.bisect_left(bounds, c[0])
        if c[2] != "cudaDeviceSynchronize" and c[0] < starts[i]:
            seg_calls.setdefault(i, []).append(c)
    classes = {
        name: {"ops": 0, "kernel_ns": 0, "blocking": 0, "allocations": 0}
        for name, _, _ in OP_CLASSES
    }
    for i, ks in seg_kernels.items():
        names = {k[2] for k in ks}
        for cls, a_names, d_names in OP_CLASSES:
            if names & (d_names if dyng_side else a_names):
                c = classes[cls]
                c["ops"] += 1
                c["kernel_ns"] += sum(k[1] - k[0] for k in ks)
                for call in seg_calls.get(i, []):
                    c["blocking"] += call[2] in BLOCKING
                    c["allocations"] += call[2] in ALLOCATING
                break
    regs: dict[str, int] = {}
    for _, _, name, r in kernels:
        regs[name] = max(regs.get(name, 0), int(r or 0))
    return {"classes": classes, "registers": regs}


def early(args: argparse.Namespace) -> int:
    tests = find_tests(args.build).resolve()
    _, argv = golden_case()
    cmd, env = original_command(argv, args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-early-cbst-", dir=perf_ab.SCRATCH / "runs"))
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            a = segments(cmd, env, work / "a", dyng_side=False)
            d = segments(
                [str(tests), "--gtest_filter=CbstEarlyMemory.*"],
                dict(env, DYNG_CBST_MEMORY_PRESET="escher_gpu"),
                work / "b",
                dyng_side=True,
            )
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    rows = []
    for cls, _, _ in OP_CLASSES:
        ca, cd = a["classes"][cls], d["classes"][cls]

        def per(c: dict, k: str) -> float | None:
            return round(c[k] / c["ops"], 3) if c["ops"] else None

        row = {
            "op": cls,
            "ops": [ca["ops"], cd["ops"]],
            "device_ms_per_op": [
                round(ca["kernel_ns"] / 1e6 / ca["ops"], 3) if ca["ops"] else None,
                round(cd["kernel_ns"] / 1e6 / cd["ops"], 3) if cd["ops"] else None,
            ],
            "blocking_per_op": [per(ca, "blocking"), per(cd, "blocking")],
            "allocations_per_op": [per(ca, "allocations"), per(cd, "allocations")],
        }
        dm = row["device_ms_per_op"]
        row["ratio"] = round(dm[1] / dm[0], 3) if dm[0] and dm[1] else None
        rows.append(row)
        print(
            f"{cls:11} ops {ca['ops']}/{cd['ops']}  device ms/op A {dm[0]} dynG {dm[1]} "
            f"(x{row['ratio']})  blocking/op {row['blocking_per_op']}  allocations/op "
            f"{row['allocations_per_op']}"
        )
    registers = {k: [a["registers"].get(k), d["registers"].get(v)] for k, v in HOT_KERNELS.items()}
    for k, (ra, rd) in registers.items():
        print(f"  registers {k:24} A {ra}  dynG {rd}")
    if args.json:
        doc = {
            "schema": 1,
            "what": "M8a step 5 early-warning table (reported, not gated): per store operation of "
            "escher_gpu_coauth50k, ESCHER-GPU's unpatched build/main against the dynG store replay "
            "under escher_gpu(): device time of the kernels, blocking and allocating runtime "
            "calls, registers of the hot kernels (values are [ESCHER-GPU, dynG])",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "set": SET,
            "original": [perf_ab.portable_path(c) if "/" in c else c for c in cmd],
            "port": {"commit": perf_ab.port_commit(), "binary": perf_ab.portable_path(tests)},
            "gpu": args.gpu,
            "driver": perf_ab.driver_versions(),
            "ops": rows,
            "registers": registers,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("memory", help="the store accounting and the nsys peaks (criterion 15)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--build", type=Path, help="the CUDA build tree of dyng_cbst_memory_tests")
    p.add_argument("--json", type=Path)
    p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--preset", default="escher_gpu", choices=["escher_gpu", "dyng_default"])
    e = sub.add_parser("early", help="the early-warning table of M8a step 5 (not gated)")
    e.add_argument("--gpu", type=int, default=0)
    e.add_argument("--build", type=Path, help="the CUDA build tree of dyng_cbst_memory_tests")
    e.add_argument("--json", type=Path)
    e.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
    e.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)
    return early(args) if args.command == "early" else memory(args)


if __name__ == "__main__":
    sys.exit(main())
