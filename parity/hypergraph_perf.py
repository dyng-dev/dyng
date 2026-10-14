#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Device memory and the maintain reading of the hypergraph container against ESCHER-GPU@abcf9b2
(M8b step 2: acceptance criteria 8 and 9; PLAN 6.4.5, 6.4.6, 8.6).

    parity/perf_ab.py hypergraph memory   [--gpu 0] [--json parity/results/M8b-memory.json]
    parity/perf_ab.py hypergraph maintain [--gpu 0] [--runs 5] [--lock-clocks boost]
                                          [--sets coauth50k,syn1m] [--json ...]

(or `parity/hypergraph_perf.py memory ...`). Both sides replay the same batches: ESCHER-GPU's
UNPATCHED `build/main` generates them (the command of the golden case, seed 1), and the port
replays that run's batch records (the golden set's trace, decompressed once) through
dyng::hypergraph with `dyng_hypergraph_ids --backend cuda --line-graph --apply-only` (the
container under {erase_first, escher_gpu(), sorted_merge}, the hypergraph loaded from the dump's
construct rows; the inserted ids are still compared with A's, nothing else).

`memory` (criterion 8; the gate is the author's: store accounting, <= 1.05x, decided 2026-10-07):

- the accounting: `store_stats().device_bytes` (A's DynamicHypergraph::deviceBytes() scope: the
  H2V, V2H and H2H trees and the H2V and H2H row views) after the build and after every batch,
  against A's recorded deviceBytes() at the same points (the golden set's batch records);
- that the UNPATCHED copy prints the same values at its two print points (src/driver.cu:196 and
  :239, "[memory] ESCHER device allocations: <init> MB after init, <last> MB after the last batch",
  in MiB with one decimal) as the trace recorded;
- the Nsight Systems peak of the live device allocations of both processes (recorded, not gated
  in M8; A's process also counts its motif counting, gated with it in M9).

`maintain` (criterion 9; reported, not gated): A/B/A/B with --runs rounds per side, the GPU
clocks locked for the whole reading (perf_ab.ClockLock, ADR 0018), the contamination monitor
(perf_ab.MachineMonitor), one heap layout (the --timing file names of a fixed length; ADR 0024):
A's `update.maintain` and its `maintain.*` timers (ScopedTimer: device-synchronized at both ends)
against the container's `hypergraph.apply` stages (`--sync-stages`: the stream synchronized at the
stage boundaries), mapped through parity/timed_regions/hypergraph.toml, per batch and summed over
the 3 batches; then one Nsight Systems CUDA API trace per side (A's NVTX ranges of its
ScopedTimers, the container's NVTX stage ranges; dynG without stage synchronization) counts the
blocking calls inside each batch's maintain region (cudaStreamSynchronize, cudaDeviceSynchronize,
cudaEventSynchronize, cudaMemcpy, cudaMemset, cudaMemcpyToSymbol / FromSymbol).

Every reading takes the EXCLUSIVE perf lock ($DYNG_SCRATCH/perf.lock).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import cycle_count_perf  # noqa: E402
import perf_ab  # noqa: E402

REFERENCE = "ESCHER-GPU"
# The sets of the readings: the golden case whose trace the port replays (A's batch records) and
# whose command A runs.
SETS = {
    "coauth50k": ("escher_gpu_coauth50k", "coauth_b50k_s1"),
    "syn1m": ("escher_gpu_syn1m", "syn1m_b50k_s1"),
}
BLOCKING = {
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMemset",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
}
# The maintain region and its parts: A's ScopedTimer names and dynG's stages (the region map
# parity/timed_regions/hypergraph.toml, loaded and checked by perf_ab.load_regions).
MAINTAIN_REGIONS = (
    "maintain",
    "maintain.delete",
    "maintain.insert",
    "maintain.line_graph",
    "maintain.views",
)


def find_tool(build: Path | None) -> Path:
    candidates = [build] if build else [REPO / "build" / p for p in ("parity-cuda", "release-cuda")]
    for b in candidates:
        exe = b / "cpp" / "tests" / "dyng_hypergraph_ids"
        if exe.is_file():
            return exe
    raise SystemExit(
        "dyng_hypergraph_ids not found: build it in a Release CUDA tree (cmake --build "
        "build/parity-cuda --target dyng_hypergraph_ids_tool) or pass --build"
    )


def golden_case(name: str) -> tuple[Path, list[str]]:
    set_dir, case = SETS[name]
    d = perf_ab.SCRATCH / "goldens" / "cbst_trace" / set_dir / case
    if not (d / "case.json").is_file():
        raise SystemExit(f"no golden case {d} (parity/export_goldens.py cbst_trace {set_dir})")
    return d, json.loads((d / "case.json").read_text())["argv"]


def decompress(case_dir: Path, work: Path) -> Path:
    """The case's trace (and side file) decompressed into `work` once."""
    out = work / f"{case_dir.name}.cbt"
    for src, dst in [("trace.cbt.zst", out), ("trace.blob.zst", Path(str(out) + ".blob"))]:
        if (case_dir / src).is_file() and not dst.is_file():
            subprocess.run(
                ["zstd", "-q", "-d", "--long=31", "-f", str(case_dir / src), "-o", str(dst)],
                check=True,
            )
    return out


def original_command(argv: list[str], gpu: int, check: bool = False) -> tuple[list[str], dict]:
    """ESCHER-GPU's command of the golden case on the UNPATCHED copy (without --check unless
    `check`: the integrity checks run outside the timed regions but not in the paper's runs)."""
    ref = perf_ab.reference_copy(REFERENCE)
    main = ref / "build" / "main"
    if not main.is_file():
        raise SystemExit(f"{main} missing: parity/build_reference.sh {REFERENCE}")
    dataset = perf_ab.SCRATCH / "datasets" / "hyper"
    args = [a.replace("<abs>/coauth.hg", str(dataset / "coauth.hg")) for a in argv[1:]]
    if not check:
        args = [a for a in args if a != "--check"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return [str(main), *args], env


def port_command(tool: Path, trace: Path, *extra: str) -> list[str]:
    return [
        str(tool),
        "--backend",
        "cuda",
        "--line-graph",
        "--apply-only",
        "--mode",
        "by_id",
        *extra,
        str(trace),
    ]


def run(cmd: list[str], env: dict) -> str:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return proc.stdout + proc.stderr


# --- memory ----------------------------------------------------------------------------------


def memory(args: argparse.Namespace) -> int:
    tool = find_tool(args.build).resolve()
    case_dir, argv = golden_case("coauth50k")
    cmd, env = original_command(argv, args.gpu, check=True)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-mem-hg-", dir=perf_ab.SCRATCH / "runs"))
    try:
        trace = decompress(case_dir, work)
        report_json = work / "report.json"
        port = port_command(tool, trace, "--json", str(report_json))
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            # The accounting (and the ids) of the replay, then A's own print points.
            run(port, env)
            printed = run(cmd, env)
            original = cycle_count_perf.device_memory(cmd, env, work / "a")
            port_peak = cycle_count_perf.device_memory(port, env, work / "b")
        rep = json.loads(report_json.read_text())[0]
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    points = ["build"] + [f"batch {b}" for b in range(1, len(rep["device_bytes"]))]
    table = [
        {
            "point": p,
            "dyng_device_bytes": mine,
            "escher_gpu_device_bytes": theirs,
            "ratio": mine / theirs if theirs > 0 else float("nan"),
        }
        for p, (mine, theirs) in zip(points, rep["device_bytes"], strict=True)
    ]
    worst = max((r["ratio"] for r in table), default=float("nan"))
    # A's two print points against the recorded values (MiB, one decimal; driver.cu:30, 238-240).
    line = next((x for x in printed.splitlines() if "ESCHER device allocations" in x), "")
    want = (
        f"{rep['device_bytes'][0][1] / 2**20:.1f} MB after init, "
        f"{rep['device_bytes'][-1][1] / 2**20:.1f} MB after the last batch"
    )
    prints_agree = want in line
    print(
        f"coauth50k (escher_gpu, erase_first, sorted_merge): {len(table)} readings, worst "
        f"device_bytes / deviceBytes() {worst:.4f}; ids {rep['ids_compared']} compared, "
        f"{rep['mismatch_count']} mismatches"
    )
    for r in table:
        print(
            f"  {r['point']:8} A {r['escher_gpu_device_bytes']:>13} dynG "
            f"{r['dyng_device_bytes']:>13} ratio {r['ratio']:.6f}"
        )
    print(
        f"  the unpatched copy prints: {line.strip()!r} ({'agrees' if prints_agree else 'DIFFERS'})"
    )
    print(
        f"  nsys peak of live device allocations: original {original['peak_live_mib']} MiB "
        f"(store, row views and motif counting), port {port_peak['peak_live_mib']} MiB (pool "
        f"{port_peak['pool_mib']} MiB)"
    )
    passed = bool(table) and worst <= 1.05 and prints_agree and rep["mismatch_count"] == 0
    if args.json:
        doc = {
            "schema": 1,
            "what": "M8b criterion 8: the store accounting (gated, <= 1.05x at every point) and "
            "the Nsight Systems peak of live device allocations of both processes (recorded, not "
            "gated in M8; the author's decision of 2026-10-07)",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "set": SETS["coauth50k"][0],
            "properties": "{erase_first, escher_gpu(), sorted_merge}",
            "original": {
                "command": [perf_ab.portable_path(c) if "/" in c else c for c in cmd],
                "printed": line.strip(),
                "printed_agrees_with_the_trace": prints_agree,
                "nsys": original,
            },
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(tool),
                "nsys": port_peak,
                "ids_compared": rep["ids_compared"],
                "mismatches": rep["mismatch_count"],
            },
            "gpu": args.gpu,
            "driver": perf_ab.driver_versions(),
            "accounting": table,
            "worst_ratio": worst,
            "passed": passed,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0 if passed else 1


# --- maintain --------------------------------------------------------------------------------


def stage_times(path: Path) -> dict[tuple[str, int], float]:
    """A --timing JSON file ({"stages": [{"stage", "batch", "ms"}]}, A's format, which the replay
    tool also writes): summed per (stage, batch)."""
    out: dict[tuple[str, int], float] = {}
    for rec in json.loads(path.read_text())["stages"]:
        key = (rec["stage"], int(rec["batch"]))
        out[key] = out.get(key, 0.0) + float(rec["ms"])
    return out


def region_value(times: dict, names: list[str], batch: int) -> float:
    return sum(times.get((n, batch), 0.0) for n in names)


def batches_sum(times: dict, stage: str) -> float:
    return sum(region_value(times, [stage], k) for k in (1, 2, 3))


def original_names(region: dict) -> list[str]:
    return [k.split(":", 1)[1] for k in region["original_report"]]


def api_trace(cmd: list[str], env: dict, report: Path, region: str) -> list[dict]:
    """One process under `nsys profile -t cuda,nvtx`: per NVTX range named `region` (one per
    batch, in order), the blocking runtime calls of its thread inside it, by name."""
    import sqlite3

    nsys = os.environ.get("NSYS", "/usr/local/cuda-13.1/bin/nsys")
    subprocess.run(
        [
            nsys,
            "profile",
            "-t",
            "cuda,nvtx",
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
        ranges = []
        for start, end, text, text_id, tid in db.execute(
            "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
        ):
            name = text if text is not None else strings.get(text_id, "")
            if name == region:
                ranges.append((start, end, tid))
        calls = [
            (start, tid, strings.get(name, "?").split("_v")[0])
            for start, name, tid in db.execute(
                "select start, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        ]
        db.close()
    finally:
        for f in [database, Path(str(report) + ".nsys-rep")]:
            f.unlink(missing_ok=True)
    ranges.sort()
    out = []
    for start, end, tid in ranges:
        counts: dict[str, int] = {}
        for t, ctid, name in calls:
            if ctid == tid and start <= t <= end and name in BLOCKING:
                counts[name] = counts.get(name, 0) + 1
        out.append({"total": sum(counts.values()), "by_name": counts})
    return out


def maintain(args: argparse.Namespace) -> int:
    tool = find_tool(args.build).resolve()
    regions = [
        r
        for r in perf_ab.load_regions(
            "escher_gpu", perf_ab.HYPERGRAPH_REGION_MAP, required=MAINTAIN_REGIONS
        )
        if r["name"] in MAINTAIN_REGIONS
    ]
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-maint-hg-", dir=perf_ab.SCRATCH / "runs"))
    clocks = perf_ab.ClockLock(args.gpu, args.lock_clocks)
    results: dict = {}
    try:
        traces = {}
        for name in args.sets:
            case_dir, argv = golden_case(name)
            traces[name] = (decompress(case_dir, work), argv)
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            with clocks:
                for name in args.sets:
                    trace, argv = traces[name]
                    cmd, env = original_command(argv, args.gpu)
                    timing = work / "timing.json"  # one layout: one file name for every round
                    samples: dict[str, list] = {"original": [], "port": []}
                    windows, rejected = [], []
                    with perf_ab.MachineMonitor(
                        args.gpu,
                        args.max_foreign_cpu,
                        allowed_pids={clocks.pid} if clocks.pid else None,
                        locked=clocks.locked,
                    ) as monitor:
                        r = 0
                        while r < args.runs:
                            _, win_a = monitor.run([*cmd, "--timing", str(timing)], env)
                            a = stage_times(timing)
                            _, win_b = monitor.run(
                                port_command(tool, trace, "--sync-stages", "--timing", str(timing)),
                                env,
                            )
                            b = stage_times(timing)
                            window = {"original": win_a, "port": win_b}
                            bad = [x for w in window.values() for x in w.get("reasons", [])]
                            if bad:
                                rejected.append({"before_round": r + 1, "reasons": bad})
                                print(f"{name}: round {r + 1} rejected ({'; '.join(bad)})")
                                if len(rejected) > 2 * args.runs:
                                    raise SystemExit(
                                        f"{name}: {len(rejected)} rounds rejected by the "
                                        "contamination monitor; the machine is too busy"
                                    )
                                continue
                            r += 1
                            windows.append(window)
                            samples["original"].append(a)
                            samples["port"].append(b)
                            print(
                                f"{name} round {r}/{args.runs}: maintain original "
                                f"{batches_sum(a, 'update.maintain'):.1f} ms, port "
                                f"{batches_sum(b, 'hypergraph.apply'):.1f} ms (3 batches)",
                                flush=True,
                            )
                    results[name] = summarize(regions, samples)
                    results[name]["rejected_rounds"] = rejected
                    results[name]["monitor"] = windows
                    # The host synchronizations per batch, from one CUDA API trace per side.
                    results[name]["host_syncs"] = {
                        "original": api_trace(cmd, env, work / "api-a", "update.maintain"),
                        "port": api_trace(
                            port_command(tool, trace, "--nvtx"),
                            env,
                            work / "api-b",
                            "hypergraph.apply",
                        ),
                    }
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    for name, res in results.items():
        print(
            f"{name}: medians of {args.runs} rounds (ms), GPU {args.gpu}, clocks "
            f"{clocks.record.get('locked')}"
        )
        for e in res["regions"]:
            print(
                f"  {e['region']:20} batch {e['batch']:>5}: original {e['original_ms']:9.2f} "
                f"port {e['port_ms']:9.2f} ratio {e['ratio']:.3f}"
            )
        for side in ("original", "port"):
            print(
                f"  host syncs per batch ({side}): "
                + ", ".join(f"{b['total']} {b['by_name']}" for b in res["host_syncs"][side])
            )
    if args.json:
        doc = {
            "schema": 1,
            "what": "M8b criterion 9: A's update.maintain and maintain.* against the container's "
            "hypergraph.apply stages (parity/timed_regions/hypergraph.toml); reported, not gated "
            "(M9 gates `update`)",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "protocol": {
                "runs": args.runs,
                "order": "A/B/A/B",
                "gpu": args.gpu,
                "clocks": clocks.record,
                "layout": "one (fixed --timing file name)",
                "original": "the UNPATCHED copy's build/main (ScopedTimer: device-synchronized)",
                "port": "dyng_hypergraph_ids --apply-only --line-graph --sync-stages",
            },
            "port": {"commit": perf_ab.port_commit(), "binary": perf_ab.portable_path(tool)},
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def summarize(regions: list[dict], samples: dict) -> dict:
    out = []
    for region in regions:
        a_names, b_names = original_names(region), region["port"]
        for batch in ("1", "2", "3", "sum"):
            ks = (1, 2, 3) if batch == "sum" else (int(batch),)
            a = [sum(region_value(s, a_names, k) for k in ks) for s in samples["original"]]
            b = [sum(region_value(s, b_names, k) for k in ks) for s in samples["port"]]
            ma, mb = statistics.median(a), statistics.median(b)
            out.append(
                {
                    "region": region["name"],
                    "batch": batch,
                    "original_ms": ma,
                    "port_ms": mb,
                    "ratio": mb / ma if ma > 0 else float("nan"),
                    "original_runs": a,
                    "port_runs": b,
                }
            )
    return {"regions": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("memory", "the store accounting, A's print points and the nsys peaks (criterion 8)"),
        ("maintain", "A's update.maintain against hypergraph.apply, A/B/A/B (criterion 9)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--build", type=Path, help="the CUDA build tree of dyng_hypergraph_ids")
        p.add_argument("--json", type=Path)
        p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
        p.add_argument("--no-lock", action="store_true")
        if name == "maintain":
            p.add_argument("--runs", type=int, default=5, help="rounds per side (at least 5)")
            p.add_argument("--lock-clocks", default="boost", choices=["boost", "base", "none"])
            p.add_argument("--sets", type=lambda s: s.split(","), default=["coauth50k", "syn1m"])
            p.add_argument("--max-foreign-cpu", type=float, default=perf_ab.MAX_FOREIGN_CPU)
    args = parser.parse_args(argv)
    if args.command == "maintain":
        if args.runs < 5:
            parser.error("--runs must be at least 5 (PLAN 8.6)")
        unknown = [s for s in args.sets if s not in SETS]
        if unknown:
            parser.error(f"unknown sets {unknown}; the sets are {', '.join(SETS)}")
        return maintain(args)
    return memory(args)


if __name__ == "__main__":
    sys.exit(main())
