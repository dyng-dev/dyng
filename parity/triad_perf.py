#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""triad_count against ESCHER-GPU@abcf9b2 (A): the timed A/B rounds and the device-memory readings
(M9a; PLAN 6.4.6, 8.6; ADRs 0018, 0021, 0048; the author's decisions of 2026-10-07, 2026-10-08
and 2026-10-15).

    parity/perf_ab.py triad_count run    [--cases coauth_b50k_s1,...] [--pattern main]
                                         [--rounds 11] [--lock-clocks base|boost|none]
                                         [--gpu 0] [--no-count] [--no-api-trace] [--json FILE]
    parity/perf_ab.py triad_count memory [--cases coauth_b50k_s1,...] [--pattern main]
                                         [--gpu 1] [--no-count] [--json FILE]

(or `parity/triad_perf.py run ...`). A case is a group of the golden set triad_count_escher_gpu
(its argv; coauth-DBLP read from $DYNG_SCRATCH/datasets/hyper/coauth.hg).

`run --probe` (the clock probe of ADR 0021: N rounds at a lock, none repeated) records per round
and side whether every busy GPU sample ran at the locked clocks; the lock "holds" when all did.

`run` (the timing gates; criterion 12): A's UNPATCHED build/main (or build/type<N>), built by its
own Makefile, with --timing, against `dyng-compat-escher run` (the parity-cuda preset) with A's
arguments, --pattern, --backend cuda, --sync-stages and --timing, under the EXCLUSIVE perf lock,
on GPU --gpu (default 0), the clocks locked for the whole reading (perf_ab.ClockLock; ADR 0018;
coauth-DBLP at base, ADR 0021; `ncu --clock-control reset` at the end), CUDA_MODULE_LOADING=EAGER
on both sides, no OMP_* variables, one heap layout (one --timing file name of a fixed length for
every round of both sides; ADR 0024), A/B alternation, the contamination monitor
(perf_ab.MachineMonitor: a contaminated round is repeated; it gives up after 2 x rounds
rejections), and per round the load average and procs_blocked before and after each side. A
per-round guard: the counts A prints equal the golden's (and the port's, once it counts), or,
with --no-count, the port's batch lines equal the golden's. The regions are those of
parity/timed_regions/triad_count.toml (loaded and checked by perf_ab.load_regions); medians of
the rounds. Then one Nsight Systems CUDA API trace per side counts the blocking calls inside each
batch's update region (A: NVTX `update`; the port: `triad_count.update`, or `hypergraph.apply`
with --no-count).

With --no-count (M9a step 2's go/no-go pre-read; the port does not count yet) the record has the
end-to-end budget: A_noncount = A's total - init.full_count - sum over the batches of
(update.subtract + update.add), against the port's total; the target is port <= A_noncount +
0.04 x A_total (a margin target, not a tolerance), and the breakdown of A's init.load,
v2h_build, h2h_build, flatten, construct and row_views against the port's load, hypergraph.build
and hypergraph.upload.

`memory` (criterion 13; D1): both whole processes (no --check), each under
`/usr/local/cuda-13.1/bin/nsys profile -t cuda,nvtx --cuda-memory-usage=true`, on GPU --gpu
(default 1) under the shared lock with no foreign process on that GPU: the peak of live device
allocations (cudaMalloc + cudaMallocAsync, kind Device, A's Thrust temporaries included), the
NVTX ranges containing each peak, the peak per range, live / pool / slack / held bytes at the end
of every batch's range, and the footprint (allocations outside the pool + the pool) at its peak
and at the live peak; the gate is port / A <= 1.05 on the peak of live allocations (with
--no-count the step-2 pre-read: target <= 1.03, and the triad budget for step 3 = 1.05 x A_peak
- port_peak - 1% of A_peak). Beside it the store accounting: the port's store_stats().device_bytes
after the build and after every batch (the driver's [memory] lines) against A's
DynamicHypergraph::deviceBytes() at the same points (the golden set's recording, group.json).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import hypergraph_perf  # noqa: E402
import perf_ab  # noqa: E402
import triad_goldens as tg  # noqa: E402

EXES = {"main": "h_motif", "type1": "type1", "type2": "type2", "type3": "type3"}
# One heap layout: every round of both sides writes its --timing file under one name of this
# length (ADR 0024).
TIMING_NAME = "timing-ab.json"
# A's region of a batch (its NVTX range, ScopedTimer "update") and the port's.
A_BATCH_RANGE = "update"
PORT_BATCH_RANGE = {True: "triad_count.update", False: "hypergraph.apply"}
STORE_LINE = re.compile(r"^\[memory\] batch (\d+): store device bytes (\d+)$")
STORE_INIT = re.compile(r"^\[memory\] dynG store device bytes: (\d+) after init, (\d+) after")
A_PRINT = re.compile(r"\[memory\] ESCHER device allocations: ([0-9.]+) MB after init, ([0-9.]+) MB")
A_INIT_STAGES = ("init.load", "init.v2h_build", "init.h2h_build", "init.v2v_build",
                 "init.flatten", "init.construct", "init.row_views")  # fmt: skip


def find_driver(build: Path | None) -> Path:
    for b in [build] if build else [REPO / "build" / p for p in ("parity-cuda", "release-cuda")]:
        exe = b / "tools" / "compat" / "dyng-compat-escher"
        if exe.is_file():
            return exe
    raise SystemExit(
        "dyng-compat-escher not found: cmake --build build/parity-cuda --target dyng-compat-escher"
        " (the parity-cuda preset), or pass --build"
    )


def group_of(name: str) -> tg.Group:
    groups = {g.name: g for g in tg.all_groups()}
    if name not in groups:
        raise SystemExit(f"unknown case {name} (a group of {tg.SET})")
    return groups[name]


def clean_env(gpu: int) -> dict:
    """CUDA_MODULE_LOADING=EAGER on both sides, no OMP_* variables (and no recorder variables)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "DYNG_CBST_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return env


def args_of(group: tg.Group) -> list[str]:
    """A's arguments of the group (no --check / --reference), the input as an absolute path."""
    out = []
    for a in group.count_args:
        m = re.fullmatch(r"\{(\w+)\}", a)
        if m and m.group(1) == "coauth":
            out.append(str(perf_ab.SCRATCH / tg.COAUTH))
        elif m:
            raise SystemExit(f"{group.name}: the timed cases read coauth-DBLP or are synthetic")
        else:
            out.append(a)
    return out


def commands(group: tg.Group, exe: str, driver: Path, timing: Path, count: bool,
             nvtx: bool = False, sync: bool = True) -> tuple[list[str], list[str]]:  # fmt: skip
    a_exe = perf_ab.reference_copy(tg.REFERENCE) / "build" / exe
    if not a_exe.is_file():
        raise SystemExit(f"{a_exe} missing: parity/build_reference.sh {tg.REFERENCE}")
    args = args_of(group)
    a = [str(a_exe), *args] + (["--timing", str(timing)] if timing else [])
    b = [str(driver), "run", *args, "--pattern", EXES[exe], "--backend", "cuda"]
    b += ["--sync-stages"] if sync else []
    b += ["--timing", str(timing)] if timing else []
    b += [] if count else ["--no-count"]
    b += ["--nvtx"] if nvtx else []
    return a, b


def stage_times(path: Path) -> dict[tuple[str, int], float]:
    return hypergraph_perf.stage_times(path)


def total(times: dict, stage: str, batches: range | None = None) -> float:
    return sum(v for (s, b), v in times.items() if s == stage and (batches is None or b in batches))


def guard(out: str, golden: dict, exe: str, side: str, count: bool) -> list[str]:
    """The round's guard: A's (or a counting port's) count lines equal the golden's; a port
    without counts prints the golden's batch lines."""
    problems = []
    if side == "original" or count:
        try:
            parsed = tg.parse_counts(out, exe)
        except tg.GoldenError as e:
            return [f"{side}: {e}"]
        if parsed["full"] != golden["full_count"]:
            problems.append(f"{side}: full count differs from the golden")
        deltas = [b["delta"] for b in parsed["batches"]]
        if deltas != [b["delta"] for b in golden["batches"]]:
            problems.append(f"{side}: batch deltas differ from the golden")
    lines = [x for x in out.splitlines() if tg.BATCH_LINE.match(x)]
    want = [
        f"Batch {b['index']} ({b['deletions']} deletions, {b['insertions']} insertions)"
        for b in golden["batches"]
    ]
    if lines != want:
        problems.append(f"{side}: batch lines {lines[:1]} differ from the golden's {want[:1]}")
    return problems


def blocked() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("procs_blocked "):
            return int(line.split()[1])
    return -1


def noncount_reading(a: dict, b: dict, batches: int) -> dict:
    """The end-to-end budget of step 2 (the port without counting) from one round's timers."""
    rng = range(1, batches + 1)
    a_total = total(a, "total", range(0, 1))
    a_full = total(a, "init.full_count")
    a_sub_add = total(a, "update.subtract", rng) + total(a, "update.add", rng)
    out = {
        "a_total": a_total,
        "a_full_count": a_full,
        "a_subtract_add": a_sub_add,
        "a_noncount": a_total - a_full - a_sub_add,
        "port_total": total(b, "total_ms", range(0, 1)),
        "a_init": {s: total(a, s) for s in A_INIT_STAGES},
        "port_init": {
            "load_ms": total(b, "load_ms"),
            "hypergraph.build": total(b, "hypergraph.build", range(0, 1)),
            "hypergraph.upload": total(b, "hypergraph.upload", range(0, 1)),
        },
        "a_maintain": [total(a, "update.maintain", range(k, k + 1)) for k in rng],
        "port_apply": [total(b, "apply_ms", range(k, k + 1)) for k in rng],
        "a_generate": [total(a, "batch.generate", range(k, k + 1)) for k in rng],
        "port_generate": [total(b, "generate_ms", range(k, k + 1)) for k in rng],
    }
    out["port_untimed_rest"] = out["port_total"] - out["port_init"]["load_ms"] - out["port_init"][
        "hypergraph.build"] - out["port_init"]["hypergraph.upload"] - sum(out["port_apply"]) - sum(
        out["port_generate"])  # fmt: skip
    out["a_untimed_rest"] = (a_total - sum(out["a_init"].values()) - a_full
                             - total(a, "update", rng) - sum(out["a_generate"]))  # fmt: skip
    return out


def median_of(rounds: list[dict], key) -> float:
    return statistics.median(key(r) for r in rounds)


def summarize_noncount(readings: list[dict]) -> dict:
    keys = ["a_total", "a_full_count", "a_subtract_add", "a_noncount", "port_total",
            "port_untimed_rest", "a_untimed_rest"]  # fmt: skip
    med = {k: median_of(readings, lambda r, k=k: r[k]) for k in keys}
    med["a_init"] = {s: median_of(readings, lambda r, s=s: r["a_init"][s]) for s in A_INIT_STAGES}
    med["port_init"] = {
        s: median_of(readings, lambda r, s=s: r["port_init"][s]) for s in readings[0]["port_init"]
    }
    n = len(readings[0]["a_maintain"])
    for k in ("a_maintain", "port_apply", "a_generate", "port_generate"):
        med[k] = [median_of(readings, lambda r, k=k, i=i: r[k][i]) for i in range(n)]
    med["target"] = med["a_noncount"] + 0.04 * med["a_total"]
    med["met"] = med["port_total"] <= med["target"]
    med["port_over_a_noncount_ms"] = med["port_total"] - med["a_noncount"]
    return med


def summarize_regions(regions: list[dict], a_rounds: list[dict], b_rounds: list[dict]) -> list:
    """The rows of the region map whose port stages the port recorded (medians)."""
    out = []
    for region in regions:
        a_names = [k.split(":", 1)[1] for k in region["original_report"] if k.startswith("timing:")]
        b_names = region.get("port_report") or region["port"]
        if not a_names or not b_names:
            continue
        rows = perf_ab.region_rows(region)
        for row in rows:
            _, batch = perf_ab.batch_of_row(row)
            sel = range(batch, batch + 1) if batch is not None else None
            a = [sum(total(r, n, sel) for n in a_names) for r in a_rounds]
            b = [sum(total(r, n, sel) for n in b_names) for r in b_rounds]
            if not any(b):
                continue  # the port does not record it (yet)
            ma, mb = statistics.median(a), statistics.median(b)
            out.append({"row": row, "gate": region["gate"], "original_ms": ma, "port_ms": mb,
                        "ratio": mb / ma if ma > 0 else None, "original_runs": a,
                        "port_runs": b})  # fmt: skip
    return out


def run(args: argparse.Namespace) -> int:
    driver = find_driver(args.build).resolve()
    regions = perf_ab.load_regions("escher_gpu", perf_ab.TRIAD_REGION_MAP,
                                   required=("update", "full_count", "end_to_end"))  # fmt: skip
    count = not args.no_count
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-ab-", dir=perf_ab.SCRATCH / "runs"))
    timing = work / TIMING_NAME
    env = clean_env(args.gpu)
    clocks = perf_ab.ClockLock(args.gpu, args.lock_clocks)
    results: dict = {}
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            with clocks:
                for case in args.cases:
                    group = group_of(case)
                    golden = json.loads(
                        (
                            tg.set_dir(perf_ab.SCRATCH) / "cases" / case / f"{args.pattern}.json"
                        ).read_text()
                    )
                    a_cmd, b_cmd = commands(group, args.pattern, driver, timing, count)
                    a_rounds, b_rounds, windows, rejected, loads = [], [], [], [], []
                    probe: list[dict] = []
                    with perf_ab.MachineMonitor(
                        args.gpu, args.max_foreign_cpu,
                        allowed_pids={clocks.pid} if clocks.pid else None, locked=clocks.locked,
                    ) as monitor:  # fmt: skip
                        r = 0
                        while r < args.rounds:
                            load = {"before_a": os.getloadavg()[0], "blocked_before_a": blocked()}
                            out_a, win_a = monitor.run(a_cmd, env)
                            ta = stage_times(timing)
                            load.update(after_a=os.getloadavg()[0], blocked_after_a=blocked())
                            out_b, win_b = monitor.run(b_cmd, env)
                            tb = stage_times(timing)
                            load.update(after_b=os.getloadavg()[0], blocked_after_b=blocked())
                            bad = guard(out_a, golden, args.pattern, "original", True) + guard(
                                out_b, golden, args.pattern, "port", count
                            )
                            if bad:
                                raise SystemExit(f"{case}: round {r + 1}: the guard failed: {bad}")
                            reasons = win_a.get("reasons", []) + win_b.get("reasons", [])
                            if args.probe:
                                # The clock probe keeps every round: whether the lock held.
                                probe.append({
                                    "round": r + 1,
                                    "original": {k: win_a.get("gpu", {}).get(k) for k in
                                                 ("clocks_locked", "sm_mhz", "pstates_busy")},
                                    "port": {k: win_b.get("gpu", {}).get(k) for k in
                                             ("clocks_locked", "sm_mhz", "pstates_busy")},
                                    "reasons": reasons,
                                })  # fmt: skip
                                print(f"{case} probe round {r + 1}: original clocks locked "
                                      f"{probe[-1]['original']['clocks_locked']}, port "
                                      f"{probe[-1]['port']['clocks_locked']}"
                                      f"{' (' + '; '.join(reasons) + ')' if reasons else ''}",
                                      flush=True)  # fmt: skip
                                reasons = []
                            if reasons:
                                rejected.append({"before_round": r + 1, "reasons": reasons})
                                print(f"{case}: round {r + 1} rejected ({'; '.join(reasons)})",
                                      flush=True)  # fmt: skip
                                if len(rejected) > 2 * args.rounds:
                                    raise SystemExit(
                                        f"{case}: {len(rejected)} rounds rejected by "
                                        "the contamination monitor; giving up"
                                    )
                                continue
                            r += 1
                            a_rounds.append(ta)
                            b_rounds.append(tb)
                            windows.append({"original": win_a, "port": win_b})
                            loads.append(load)
                            note = ""
                            if not count:
                                nc = noncount_reading(ta, tb, len(golden["batches"]))
                                note = (
                                    f"A total {nc['a_total']:.0f} (noncount "
                                    f"{nc['a_noncount']:.0f}) port {nc['port_total']:.0f} ms"
                                )
                            print(f"{case} round {r}/{args.rounds}: {note}", flush=True)
                    res: dict = {
                        "rounds": args.rounds,
                        "regions": summarize_regions(regions, a_rounds, b_rounds),
                        "monitor": perf_ab.monitor_summary(windows, rejected, args.max_foreign_cpu),
                        "loads": loads,
                        "original_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()}
                                            for t in a_rounds],
                        "port_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()}
                                        for t in b_rounds],
                    }  # fmt: skip
                    if args.probe:
                        res["probe"] = {
                            "rounds": probe,
                            "holds": all(x["original"]["clocks_locked"] and x["port"][
                                "clocks_locked"] and not x["reasons"] for x in probe),
                            "rule": "the lock holds when every busy GPU sample of both sides in "
                            "every round ran at the locked clocks and no round was contaminated",
                        }  # fmt: skip
                    if not count:
                        res["noncount"] = summarize_noncount(
                            [noncount_reading(a, b, len(golden["batches"]))
                             for a, b in zip(a_rounds, b_rounds, strict=True)])  # fmt: skip
                    if not args.no_api_trace:
                        a_api, b_api = commands(group, args.pattern, driver, None, count, nvtx=True,
                                                sync=False)  # fmt: skip
                        res["host_syncs"] = {
                            "original": hypergraph_perf.api_trace(a_api, env, work / "api-a",
                                                                  A_BATCH_RANGE),
                            "port": hypergraph_perf.api_trace(b_api, env, work / "api-b",
                                                              PORT_BATCH_RANGE[count]),
                        }  # fmt: skip
                    results[case] = res
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    for case, res in results.items():
        print(f"{case} ({args.pattern}; GPU {args.gpu}; clocks {clocks.record.get('locked')}):")
        for e in res["regions"]:
            print(
                f"  {e['row']:22} original {e['original_ms']:10.2f} port {e['port_ms']:10.2f} "
                f"ratio {e['ratio'] if e['ratio'] is None else round(e['ratio'], 3)}"
            )
        if "probe" in res:
            print(
                f"  clock probe ({clocks.record.get('control')}): "
                f"{'HOLDS' if res['probe']['holds'] else 'does not hold'}"
            )
        if "noncount" in res:
            nc = res["noncount"]
            print(
                f"  end-to-end budget: A total {nc['a_total']:.1f}, A_noncount "
                f"{nc['a_noncount']:.1f}, port {nc['port_total']:.1f}, target "
                f"{nc['target']:.1f} -> {'met' if nc['met'] else 'MISSED'}"
            )
        for side, rows in res.get("host_syncs", {}).items():
            print(
                f"  host syncs per batch ({side}): "
                + ", ".join(f"{b['total']} {b['by_name']}" for b in rows)
            )
    if args.json:
        doc = {
            "schema": 1,
            "what": "triad_count against ESCHER-GPU@abcf9b2: A/B rounds (parity/timed_regions/"
            "triad_count.toml)"
            + (
                "; the port with --no-count (M9a step 2's end-to-end pre-read)" if not count else ""
            ),
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pattern": args.pattern,
            "protocol": {
                "rounds": args.rounds,
                "order": "A/B alternation (A first in every round)",
                "gpu": args.gpu,
                "clocks": clocks.record,
                "lock": perf_ab.LOCK_STATUS,
                "layout": f"one (--timing <work>/{TIMING_NAME} on both sides)",
                "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
                "original": "the UNPATCHED copy's build/" + args.pattern + " (its Makefile)",
                "port": "dyng-compat-escher run --backend cuda --sync-stages"
                + ("" if count else " --no-count"),
            },
            "port": {"commit": perf_ab.port_commit(), "binary": perf_ab.portable_path(driver)},
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def foreign_gpu_processes(gpu: int) -> list[str]:
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i",
                           str(gpu)], capture_output=True, text=True).stdout.strip()  # fmt: skip
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    return [x.strip() for x in out.splitlines() if uuid and uuid in x]


def memory(args: argparse.Namespace) -> int:
    driver = find_driver(args.build).resolve()
    count = not args.no_count
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-mem-", dir=perf_ab.SCRATCH / "runs"))
    results: dict = {}
    try:
        for case in args.cases:
            group = group_of(case)
            gdoc = json.loads((tg.set_dir(perf_ab.SCRATCH) / "cases" / case / "group.json")
                              .read_text())  # fmt: skip
            a_cmd, b_cmd = commands(group, args.pattern, driver, None, count, nvtx=True, sync=False)
            before = foreign_gpu_processes(args.gpu)
            if before:
                raise SystemExit(f"GPU {args.gpu} has foreign compute processes: {before}")
            # The store accounting and A's print points (plain runs, no profiler).
            pa = subprocess.run(a_cmd, env=env, capture_output=True, text=True)
            pb = subprocess.run(b_cmd, env=env, capture_output=True, text=True)
            if pa.returncode or pb.returncode:
                raise SystemExit(f"{case}: a run failed:\n{pa.stderr[-1500:]}\n{pb.stderr[-1500:]}")
            store = {int(m.group(1)): int(m.group(2))
                     for x in pb.stderr.splitlines() if (m := STORE_LINE.match(x))}  # fmt: skip
            init = next((int(m.group(1)) for x in pb.stderr.splitlines()
                         if (m := STORE_INIT.match(x))), None)  # fmt: skip
            port_points = [init] + [store.get(k) for k in range(1, len(gdoc["device_bytes"]))]
            points = []
            for k, (mine, theirs) in enumerate(zip(port_points, gdoc["device_bytes"], strict=True)):
                points.append({"point": "build" if k == 0 else f"batch {k}",
                               "dyng_device_bytes": mine, "escher_gpu_device_bytes": theirs,
                               "ratio": mine / theirs if mine and theirs else None})  # fmt: skip
            a_line = next((m.group(0) for x in pa.stderr.splitlines() if (m := A_PRINT.search(x))),
                          "")  # fmt: skip
            want = (f"{gdoc['device_bytes'][0] / 2**20:.1f} MB after init, "
                    f"{gdoc['device_bytes'][-1] / 2**20:.1f} MB")  # fmt: skip
            # The Nsight Systems peaks of both whole processes.
            a_nsys = hypergraph_perf.memory_by_range(a_cmd, env, work / "a", A_BATCH_RANGE)
            b_nsys = hypergraph_perf.memory_by_range(b_cmd, env, work / "b",
                                                     PORT_BATCH_RANGE[count])  # fmt: skip
            after = foreign_gpu_processes(args.gpu)
            ratio = b_nsys["peak_live_bytes"] / a_nsys["peak_live_bytes"]
            res = {
                "store_accounting": points,
                "store_worst_ratio": max((p["ratio"] for p in points if p["ratio"]), default=None),
                "a_print_points": a_line,
                "a_print_points_agree_with_the_golden": want in a_line,
                "original_nsys": a_nsys,
                "port_nsys": b_nsys,
                "peak_ratio": ratio,
                "footprint_ratio_at_peak": b_nsys["footprint_at_live_peak_mib"]
                / a_nsys["footprint_at_live_peak_mib"],
                "footprint_ratio_max": b_nsys["footprint_peak_mib"] / a_nsys["footprint_peak_mib"],
                "footprint_ratio_between_batches": [
                    pb_["held_mib_end"] / pa_["held_mib_end"]
                    for pa_, pb_ in zip(a_nsys["batches"], b_nsys["batches"], strict=False)
                ],
                "foreign_gpu_processes": {"before": before, "after": after},
            }
            if not count:
                a_peak = a_nsys["peak_live_bytes"]
                res["preread"] = {
                    "target": 1.03,
                    "met": ratio <= 1.03,
                    "triad_budget_bytes": int(
                        1.05 * a_peak - b_nsys["peak_live_bytes"] - 0.01 * a_peak
                    ),  # fmt: skip
                    "rule": "1.05 x A_peak - port_peak (no counting) - 1% of A_peak: the triad "
                    "bytes step 3 may hold live during hypergraph.apply",
                }
            results[case] = res
            print(f"{case}: nsys peak A {a_nsys['peak_live_mib']} MiB in "
                  f"{' > '.join(a_nsys['peak_in']) or '-'}; port {b_nsys['peak_live_mib']} MiB in "
                  f"{' > '.join(b_nsys['peak_in']) or '-'}; ratio {ratio:.4f}; footprint A "
                  f"{a_nsys['footprint_peak_mib']} port {b_nsys['footprint_peak_mib']} MiB; store "
                  f"worst {res['store_worst_ratio']}", flush=True)  # fmt: skip
            if "preread" in res:
                print(
                    f"  pre-read: target 1.03 {'met' if res['preread']['met'] else 'MISSED'}; "
                    f"triad budget {res['preread']['triad_budget_bytes'] / 2**20:.1f} MiB"
                )
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    if args.json:
        doc = {
            "schema": 1,
            "what": "triad_count against ESCHER-GPU@abcf9b2: the Nsight Systems peak of live "
            "device allocations of both whole processes (D1) and the store accounting"
            + ("" if count else "; the port with --no-count (M9a step 2's memory pre-read)"),
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pattern": args.pattern,
            "protocol": {
                "nsys": "nsys profile -t cuda,nvtx --cuda-memory-usage=true (CUDA 13.1), "
                "memKind Device (cudaMalloc + cudaMallocAsync)",
                "gpu": args.gpu,
                "lock": "shared (flock -s by the caller); no foreign process on the GPU",
                "original": f"the UNPATCHED build/{args.pattern}, no --check",
                "port": "dyng-compat-escher run --backend cuda --nvtx"
                + ("" if count else " --no-count"),
            },
            "port": {"commit": perf_ab.port_commit(), "binary": perf_ab.portable_path(driver)},
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        (
            "run",
            "A/B rounds of A against the port (the timing gates; --no-count: step 2's pre-read)",
        ),
        ("memory", "the nsys peaks of both processes and the store accounting"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--cases", type=lambda s: s.split(","),
                       default=["coauth_b50k_s1", "syn1m_b50k_s1"])  # fmt: skip
        p.add_argument("--pattern", default="main", choices=sorted(EXES))
        p.add_argument("--gpu", type=int, default=0 if name == "run" else 1)
        p.add_argument("--no-count", action="store_true", help="the port applies batches only")
        p.add_argument("--build", type=Path, help="the parity-cuda build tree of the driver")
        p.add_argument("--json", type=Path)
        if name == "run":
            p.add_argument("--rounds", type=int, default=11)
            p.add_argument("--lock-clocks", default="base", choices=["base", "boost", "none"])
            p.add_argument("--max-foreign-cpu", type=float, default=perf_ab.MAX_FOREIGN_CPU)
            p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
            p.add_argument("--no-lock", action="store_true")
            p.add_argument("--no-api-trace", action="store_true")
            p.add_argument(
                "--probe",
                action="store_true",
                help="the clock probe: keep every round and record whether the lock "
                "held (no round is repeated)",
            )
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.rounds < 3:
            parser.error("--rounds must be at least 3 (5 for a reading, PLAN 8.6)")
        return run(args)
    return memory(args)


if __name__ == "__main__":
    sys.exit(main())
