#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""triad_count against ESCHER-GPU@abcf9b2 (A): the timed A/B rounds, the device-memory readings
and the clock-controlled kernel reading (M9a; PLAN 6.4.6, 8.6; ADRs 0018, 0021, 0048, 0056; the
author's decisions of 2026-10-07, 2026-10-08 and 2026-10-15).

    parity/perf_ab.py triad_count run     [--cases coauth_b50k_s1,...] [--patterns main,...]
                                          [--rounds 11] [--lock-clocks base|boost|none]
                                          [--gpu 0] [--no-count] [--no-api-trace] [--json FILE]
    parity/perf_ab.py triad_count memory  [--cases coauth_b50k_s1,...] [--patterns main]
                                          [--gpu 1] [--no-count] [--json FILE]
    parity/perf_ab.py triad_count kernels [--cases syn1m_b50k_s1] [--rounds 11] [--gpu 0]
                                          [--json FILE]

(or `parity/triad_perf.py run ...`). A case is a group of the golden set triad_count_escher_gpu
(its argv; coauth-DBLP read from $DYNG_SCRATCH/datasets/hyper/coauth.hg); a record keys its
results by `<case>/<pattern>` (pattern = A's executable: main, type1, type2, type3).

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
rejections), and per round the load average and procs_blocked before and after each side (the
contamination audit: a reading with a round above AUDIT_LOAD or AUDIT_BLOCKED is flagged, and
parity/bench_suite.py refuses it as a gate reading: it is taken again). A per-round guard: the
counts A prints equal the golden's (and the port's, once it counts), or, with --no-count, the
port's batch lines equal the golden's; a guard failure stops the reading. The regions are those
of parity/timed_regions/triad_count.toml (loaded and checked by perf_ab.load_regions); medians of
the rounds; every row of a gated region carries its gate (PLAN 8.6: compute 1.05, 1.10 below
10 ms; end to end 1.10) and its verdict. Then one Nsight Systems CUDA API trace per side counts the
blocking calls inside each batch's update region (A: NVTX `update`; the port:
`triad_count.update`, or `hypergraph.apply` with --no-count). The input's digest is taken when
the reading starts: coauth-DBLP's file; a synthetic input's rows as the golden set stores them
(both sides generate it in-process from A's parameters and the seed; generators::legacy
reproduces A's rows byte for byte, M9a step 1, and the guard requires the golden's full count).

With --no-count (M9a step 2's go/no-go pre-read; the port does not count) the record has the
end-to-end budget: A_noncount = A's total - init.full_count - sum over the batches of
(update.subtract + update.add), against the port's total; the target is port <= A_noncount +
0.04 x A_total (a margin target, not a tolerance), and the breakdown of A's init.load,
v2h_build, h2h_build, flatten, construct and row_views against the port's load, hypergraph.build
and hypergraph.upload.

`memory` (criterion 13; D1): both whole processes (no --check), each under
`/usr/local/cuda-13.1/bin/nsys profile -t cuda,nvtx --cuda-memory-usage=true`, on GPU --gpu
(default 1) under the shared lock with no foreign process on that GPU (checked before and after,
and recorded): the peak of live device allocations (cudaMalloc + cudaMallocAsync, kind Device,
A's Thrust temporaries included), the NVTX ranges containing each peak, the peak per range,
live / pool / slack / held bytes at the end of every batch's range, and the footprint
(allocations outside the pool + the pool) at its peak and at the live peak; the gate
`device_memory` is port / A <= 1.05 on the peak of live allocations. The port runs a third time
under nsys with --no-count: the triad bytes live during hypergraph.apply are the difference of
the two port runs' peaks inside hypergraph.apply. Beside it the store accounting (the author's
decision of 2026-10-07, carried into M9): the port's store_stats().device_bytes after the build
and after every batch (the driver's [memory] lines) against A's
DynamicHypergraph::deviceBytes() at the same points (the golden set's recording, group.json);
the gate `store_memory` is <= 1.05 at every point on coauth-DBLP and at A's two print points
(after the build, after the last batch) on the synthetic inputs, whose values are cross-checked
against A's own [memory] print of the same run. With --no-count: M9a step 2's pre-read (target
<= 1.03 and the triad budget = 1.05 x A_peak - port_peak - 1% of A_peak).

`kernels` (recorded, not gated; ADR 0018 rule 1): the update's subtract and add kernels of both
sides (A: wedge::countWedges and the counting kernel of the pattern; the port: the same kernels
of the transplanted engine), each process under `ncu --clock-control base --cache-control none`
with --kernel-name and --launch-skip / --launch-count selecting the launches of the three
batches (the full count's launches skipped), A/B alternation, medians of the per-batch subtract
and add kernel times (gpu__time_duration), DRAM bytes, registers and launch geometry.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib
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
APPLY_RANGE = "hypergraph.apply"
STORE_LINE = re.compile(r"^\[memory\] batch (\d+): store device bytes (\d+)$")
STORE_INIT = re.compile(r"^\[memory\] dynG store device bytes: (\d+) after init, (\d+) after")
A_PRINT = re.compile(r"\[memory\] ESCHER device allocations: ([0-9.]+) MB after init, ([0-9.]+) MB")
A_INIT_STAGES = ("init.load", "init.v2h_build", "init.h2h_build", "init.v2v_build",
                 "init.flatten", "init.construct", "init.row_views")  # fmt: skip
GATED_KINDS = ("compute", "end_to_end")
# The contamination audit (ADR 0056 section 4): the machine's load average is 2-9 while both
# programs run (A's host build is parallel); R020's storm showed 27-30 with every round passing
# the monitor. A round whose load average or procs_blocked exceeds these flags the reading.
AUDIT_LOAD = 16.0
AUDIT_BLOCKED = 4
# The store-accounting gate's points (ADR 0056 section 3): every point on coauth-DBLP (A's values
# after each batch from the golden trace, the end points cross-checked with A's print points),
# A's two print points (after the build, after the last batch) on the synthetic inputs.
STORE_EVERY_POINT = ("coauth",)
# The kernels of the update's subtract and add (ESCHER-GPU src/HMotifCount.cu etc.: countWedges
# and the counting kernel; the port: the transplanted engine's names, parity/triad_sass.py).
NCU_KERNELS = {
    "original": {
        "main": "regex:^(countWedges|triangleKernel)",
        "type2": "regex:^(countWedges|pairKernel)",
        "type3": "regex:^(countWedges|triangleKernel)",
        "type1": "regex:^keyKernel",
    },
    "port": {
        "main": "regex:^(count_wedges_kernel|triangle_kernel)",
        "type2": "regex:^(count_wedges_kernel|pair_kernel)",
        "type3": "regex:^(count_wedges_kernel|triangle_kernel)",
        "type1": "regex:^key_kernel",
    },
}
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "sm__cycles_elapsed.avg.per_second",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
]


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


def case_key(case: str, pattern: str) -> str:
    return f"{case}/{pattern}"


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


def reference_exe(exe: str) -> Path:
    a_exe = perf_ab.reference_copy(tg.REFERENCE) / "build" / exe
    if not a_exe.is_file():
        raise SystemExit(f"{a_exe} missing: parity/build_reference.sh {tg.REFERENCE}")
    return a_exe


def commands(group: tg.Group, exe: str, driver: Path, timing: Path | None, count: bool,
             nvtx: bool = False, sync: bool = True) -> tuple[list[str], list[str]]:  # fmt: skip
    a_exe = reference_exe(exe)
    args = args_of(group)
    a = [str(a_exe), *args] + (["--timing", str(timing)] if timing else [])
    b = [str(driver), "run", *args, "--pattern", EXES[exe], "--backend", "cuda"]
    b += ["--sync-stages"] if sync else []
    b += ["--timing", str(timing)] if timing else []
    b += [] if count else ["--no-count"]
    b += ["--nvtx"] if nvtx else []
    return a, b


def reference_record() -> dict:
    return {
        "name": tg.REFERENCE,
        "commit": tg.COMMIT,
        "variant": "unpatched",
        "binary": perf_ab.portable_path(perf_ab.reference_copy(tg.REFERENCE) / "build"),
        "build": "its own Makefile (parity/references.toml: make all-types build/cbst_test "
        "ARCH=sm_86)",
    }


_DIGESTS: dict[str, dict] = {}


def input_digest(group: tg.Group) -> dict:
    """The digest of the case's input, taken by this process (once per input): coauth-DBLP's
    file; a synthetic input's rows as the golden set stores them."""
    name = group.input_name
    if name in _DIGESTS:
        return _DIGESTS[name]
    if name == "coauth":
        path = perf_ab.SCRATCH / tg.COAUTH
        out = {
            "input": name,
            "sha256": tg.sha256_file(path),
            "how": f"the dataset file $DYNG_SCRATCH/{tg.COAUTH}, hashed by the measuring process",
        }
    else:
        data = tg.zstd_read(tg.set_dir(perf_ab.SCRATCH) / "inputs" / f"{name}.hg.zst")
        n, m, r1, r2 = group.options.positional[:4]
        out = {
            "input": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "generator": {"parameters": [n, m, r1, r2], "seed": group.options.seed},
            "how": "generated in-process by both sides from A's parameters and the seed; the "
            "golden set's stored rows of this input (generators::legacy reproduces A's rows "
            "byte for byte, M9a step 1) hashed by the measuring process; the per-round guard "
            "requires the golden's full count",
        }
    _DIGESTS[name] = out
    return out


def datasets_sha256(results: dict) -> dict[str, str]:
    return {r["input"]["input"]: r["input"]["sha256"] for r in results.values()}


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


def audit(loads: list[dict], windows: list[dict]) -> dict:
    """The contamination audit of a reading's accepted rounds: the load average and
    procs_blocked at every side's start and end, and foreign GPU processes in the windows."""
    flagged = []
    for i, (load, window) in enumerate(zip(loads, windows, strict=True)):
        why = []
        averages = [v for k, v in load.items() if not k.startswith("blocked")]
        blocks = [v for k, v in load.items() if k.startswith("blocked")]
        if max(averages) > AUDIT_LOAD:
            why.append(f"load average {max(averages):.1f} > {AUDIT_LOAD}")
        if max(blocks) > AUDIT_BLOCKED:
            why.append(f"procs_blocked {max(blocks)} > {AUDIT_BLOCKED}")
        foreign = [p for side in window.values() for p in (side.get("gpu") or {}).get(
            "foreign_processes", [])]  # fmt: skip
        if foreign:
            why.append(f"foreign GPU processes {foreign}")
        if why:
            flagged.append({"round": i + 1, "why": why})
    averages = [v for x in loads for k, v in x.items() if not k.startswith("blocked")]
    blocks = [v for x in loads for k, v in x.items() if k.startswith("blocked")]
    return {
        "rule": f"a round is flagged when the 1-minute load average exceeds {AUDIT_LOAD} or "
        f"procs_blocked exceeds {AUDIT_BLOCKED} at the start or end of either side, or a foreign "
        "compute process was on the timed GPU; a flagged reading is taken again (ADR 0056)",
        "load_average_max": max(averages, default=None),
        "procs_blocked_max": max(blocks, default=None),
        "flagged_rounds": flagged,
        "flagged": bool(flagged),
    }


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


def gate_fields(kind: str, original_ms: float, port_ms: float, a: list, b: list) -> dict:
    """The gate of a row of a gated region (PLAN 8.6; perf_ab.gate_limit: 1.05, 1.10 below
    10 ms, end to end 1.10) and its verdict; nothing for an ungated row."""
    if kind not in GATED_KINDS or original_ms <= 0:
        return {}
    ratio = port_ms / original_ms
    gate = perf_ab.gate_limit(kind, original_ms)
    return {
        "gate": gate,
        "within_gate": ratio <= gate,
        "provisional": kind == "compute"
        and original_ms < perf_ab.SHORT_REGION_MS
        and len(a) < perf_ab.SHORT_REGION_RUNS,
        "original_spread": perf_ab.spread(a),
        "port_spread": perf_ab.spread(b),
        "noisy": perf_ab.spread(a) > 0.10 or perf_ab.spread(b) > 0.10,
    }


def summarize_regions(regions: list[dict], a_rounds: list[dict], b_rounds: list[dict]) -> list:
    """The rows of the region map whose port stages the port recorded (medians), each with its
    gate kind and, for a gated region, its gate and verdict."""
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
                continue  # the port does not record it (with --no-count)
            ma, mb = statistics.median(a), statistics.median(b)
            entry = {"region": row, "gate_kind": region["gate"], "original_ms": ma,
                     "port_ms": mb, "ratio": mb / ma if ma > 0 else None, "original_runs": a,
                     "port_runs": b}  # fmt: skip
            entry |= gate_fields(region["gate"], ma, mb, a, b)
            out.append(entry)
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
    started = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            with clocks:
                for case in args.cases:
                    group = group_of(case)
                    digest = input_digest(group)
                    for pattern in args.patterns:
                        key = case_key(case, pattern)
                        golden = json.loads(
                            (tg.set_dir(perf_ab.SCRATCH) / "cases" / case / f"{pattern}.json")
                            .read_text()
                        )  # fmt: skip
                        res = run_case(args, group, pattern, golden, driver, timing, env, clocks,
                                       regions, count, work)  # fmt: skip
                        res["input"] = digest
                        results[key] = res
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    for key, res in results.items():
        print(f"{key} (GPU {args.gpu}; clocks {clocks.record.get('locked')}):")
        for e in res["regions"]:
            verdict = ""
            if "gate" in e:
                verdict = f" <= {e['gate']:.2f} {'ok' if e['within_gate'] else 'EXCEEDED'}"
            print(
                f"  {e['region']:22} original {e['original_ms']:10.2f} port {e['port_ms']:10.2f} "
                f"ratio {e['ratio'] if e['ratio'] is None else round(e['ratio'], 3)}{verdict}"
            )
        if res["audit"]["flagged"]:
            print(f"  contamination audit: FLAGGED {res['audit']['flagged_rounds']}")
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
            "schema": 2,
            "algorithm": "triad_count",
            "what": "triad_count against ESCHER-GPU@abcf9b2: A/B rounds (parity/timed_regions/"
            "triad_count.toml)"
            + (
                "; the port with --no-count (M9a step 2's end-to-end pre-read)" if not count else ""
            ),
            "date": started,
            "date_end": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "patterns": args.patterns,
            "reference": reference_record(),
            "protocol": {
                "rounds": args.rounds,
                "order": "A/B alternation (A first in every round)",
                "gpu": args.gpu,
                "clocks": clocks.record,
                "lock": perf_ab.LOCK_STATUS,
                "layout": f"one (--timing <work>/{TIMING_NAME} on both sides)",
                "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
                "original": "the UNPATCHED copy's build/<pattern> (its Makefile)",
                "port": "dyng-compat-escher run --backend cuda --sync-stages"
                + ("" if count else " --no-count"),
                "statistic": "median",
                "contamination_monitor": "contaminated rounds repeated (perf_ab.MachineMonitor)",
                "guard": "every round (accepted or repeated): the counts A prints"
                + (" and the port prints" if count else "")
                + " equal the golden case's; a failure stops the reading",
                "probe": bool(args.probe),
            },
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(driver),
                "build": perf_ab.port_build(driver, "cuda"),
            },
            "driver": perf_ab.driver_versions(),
            "datasets_sha256": datasets_sha256(results),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    exceeded = [
        f"{k} {e['region']}" for k, r in results.items() for e in r["regions"]
        if "gate" in e and not e["within_gate"]
    ]  # fmt: skip
    return 1 if exceeded and args.enforce_gates else 0


def run_case(args, group, pattern, golden, driver, timing, env, clocks, regions, count, work):
    """The rounds of one case and pattern (A first in every round), repeated while the monitor
    rejects them; then the API traces."""
    key = case_key(group.name, pattern)
    a_cmd, b_cmd = commands(group, pattern, driver, timing, count)
    a_rounds, b_rounds, windows, rejected, loads = [], [], [], [], []
    probe: list[dict] = []
    guarded = 0
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
            bad = guard(out_a, golden, pattern, "original", True) + guard(
                out_b, golden, pattern, "port", count
            )
            if bad:
                raise SystemExit(f"{key}: round {r + 1}: the guard failed: {bad}")
            guarded += 1
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
                print(f"{key} probe round {r + 1}: original clocks locked "
                      f"{probe[-1]['original']['clocks_locked']}, port "
                      f"{probe[-1]['port']['clocks_locked']}"
                      f"{' (' + '; '.join(reasons) + ')' if reasons else ''}",
                      flush=True)  # fmt: skip
                reasons = []
            if reasons:
                rejected.append({"before_round": r + 1, "reasons": reasons,
                                 "windows": {"original": win_a, "port": win_b}})  # fmt: skip
                print(f"{key}: round {r + 1} rejected ({'; '.join(reasons)})", flush=True)
                if len(rejected) > 2 * args.rounds:
                    raise SystemExit(
                        f"{key}: {len(rejected)} rounds rejected by the contamination monitor; "
                        "giving up (no record)"
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
                    f"A total {nc['a_total']:.0f} (noncount {nc['a_noncount']:.0f}) port "
                    f"{nc['port_total']:.0f} ms"
                )
            else:
                ks = range(1, len(golden["batches"]) + 1)
                ua = [round(total(ta, "update", range(k, k + 1)), 1) for k in ks]
                ub = [round(total(tb, "update_ms", range(k, k + 1)), 1) for k in ks]
                note = f"update A {ua} port {ub}"
            print(f"{key} round {r}/{args.rounds}: {note}", flush=True)
    res: dict = {
        "case": group.name,
        "pattern": pattern,
        "dataset": group.dataset,
        "seed": group.options.seed,
        "batches": len(golden["batches"]),
        "rounds": len(a_rounds),
        "complete": len(a_rounds) >= args.rounds,
        "regions": summarize_regions(regions, a_rounds, b_rounds),
        "monitor": perf_ab.monitor_summary(windows, rejected, args.max_foreign_cpu),
        "contaminated_rounds_kept": sum(
            1 for w in windows for side in w.values() if side.get("reasons")
        ),
        "loads": loads,
        "audit": audit(loads, windows),
        "guard": {"rounds_checked": guarded, "passed": True},
        "original_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()} for t in a_rounds],
        "port_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()} for t in b_rounds],
    }
    if args.probe:
        res["probe"] = {
            "rounds": probe,
            "holds": all(
                x["original"]["clocks_locked"] and x["port"]["clocks_locked"] and not x["reasons"]
                for x in probe
            ),  # fmt: skip
            "rule": "the lock holds when every busy GPU sample of both sides in every round ran "
            "at the locked clocks and no round was contaminated",
        }
    if not count:
        res["noncount"] = summarize_noncount(
            [noncount_reading(a, b, len(golden["batches"]))
             for a, b in zip(a_rounds, b_rounds, strict=True)])  # fmt: skip
    if not args.no_api_trace:
        a_api, b_api = commands(group, pattern, driver, None, count, nvtx=True, sync=False)
        res["host_syncs"] = {
            "original": hypergraph_perf.api_trace(a_api, env, work / "api-a", A_BATCH_RANGE),
            "port": hypergraph_perf.api_trace(
                b_api, env, work / "api-b", PORT_BATCH_RANGE[count]
            ),  # fmt: skip
        }
    return res


def foreign_gpu_processes(gpu: int) -> list[str]:
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i",
                           str(gpu)], capture_output=True, text=True).stdout.strip()  # fmt: skip
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    return [x.strip() for x in out.splitlines() if uuid and uuid in x]


def store_points(group: tg.Group, gdoc: dict, port_points: list) -> list[dict]:
    """The store accounting at every point, each marked whether the store_memory gate reads it
    (every point on coauth-DBLP; A's two print points on the synthetic inputs)."""
    every = group.dataset in STORE_EVERY_POINT
    last = len(gdoc["device_bytes"]) - 1
    points = []
    for k, (mine, theirs) in enumerate(zip(port_points, gdoc["device_bytes"], strict=True)):
        points.append({
            "point": "build" if k == 0 else f"batch {k}",
            "row": "store_memory[build]" if k == 0 else f"store_memory[b{k}]",
            "dyng_device_bytes": mine,
            "escher_gpu_device_bytes": theirs,
            "ratio": mine / theirs if mine and theirs else None,
            "print_point": k in (0, last),
            "gated": every or k in (0, last),
        })  # fmt: skip
    return points


def memory(args: argparse.Namespace) -> int:
    driver = find_driver(args.build).resolve()
    count = not args.no_count
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-mem-", dir=perf_ab.SCRATCH / "runs"))
    results: dict = {}
    started = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        for case in args.cases:
            group = group_of(case)
            gdoc = json.loads((tg.set_dir(perf_ab.SCRATCH) / "cases" / case / "group.json")
                              .read_text())  # fmt: skip
            for pattern in args.patterns:
                key = case_key(case, pattern)
                golden = json.loads((tg.set_dir(perf_ab.SCRATCH) / "cases" / case /
                                     f"{pattern}.json").read_text())  # fmt: skip
                a_cmd, b_cmd = commands(group, pattern, driver, None, count, nvtx=True, sync=False)
                before = foreign_gpu_processes(args.gpu)
                if before:
                    raise SystemExit(f"GPU {args.gpu} has foreign compute processes: {before}")
                # The store accounting and A's print points (plain runs, no profiler).
                pa = subprocess.run(a_cmd, env=env, capture_output=True, text=True)
                pb = subprocess.run(b_cmd, env=env, capture_output=True, text=True)
                if pa.returncode or pb.returncode:
                    raise SystemExit(
                        f"{key}: a run failed:\n{pa.stderr[-1500:]}\n{pb.stderr[-1500:]}"
                    )
                # The guard of run(): the counts both measured programs print are the golden's
                # (M9a review: a port that skipped counting would have a lower peak).
                runs_guarded = 0
                bad = guard(pa.stdout, golden, pattern, "original", True) + guard(
                    pb.stdout, golden, pattern, "port", count
                )
                if bad:
                    raise SystemExit(f"{key}: the plain runs' guard failed: {bad}")
                runs_guarded += 2
                store = {int(m.group(1)): int(m.group(2))
                         for x in pb.stderr.splitlines() if (m := STORE_LINE.match(x))}  # fmt: skip
                init = next((int(m.group(1)) for x in pb.stderr.splitlines()
                             if (m := STORE_INIT.match(x))), None)  # fmt: skip
                port_points = [init] + [store.get(k) for k in range(1, len(gdoc["device_bytes"]))]
                points = store_points(group, gdoc, port_points)
                a_line = next((m.group(0) for x in pa.stderr.splitlines()
                               if (m := A_PRINT.search(x))), "")  # fmt: skip
                want = (f"{gdoc['device_bytes'][0] / 2**20:.1f} MB after init, "
                        f"{gdoc['device_bytes'][-1] / 2**20:.1f} MB")  # fmt: skip
                if want not in a_line:
                    raise SystemExit(
                        f"{key}: A's print points '{a_line}' disagree with the golden "
                        f"recording ({want}): the store gate's reference values are not A's"
                    )
                # The Nsight Systems peaks of both whole processes (and the port without counting:
                # the triad bytes live during hypergraph.apply).
                a_nsys = hypergraph_perf.memory_by_range(a_cmd, env, work / "a", A_BATCH_RANGE,
                                                         capture=True)  # fmt: skip
                b_nsys = hypergraph_perf.memory_by_range(b_cmd, env, work / "b",
                                                         PORT_BATCH_RANGE[count],
                                                         capture=True)  # fmt: skip
                # The same guard on the profiled processes, whose peaks are the gate.
                bad = guard(a_nsys.pop("stdout"), golden, pattern, "original", True) + guard(
                    b_nsys.pop("stdout"), golden, pattern, "port", count
                )
                if bad:
                    raise SystemExit(f"{key}: the profiled runs' guard failed: {bad}")
                runs_guarded += 2
                triad_apply = None
                c_nsys = None
                if count:
                    _, c_cmd = commands(group, pattern, driver, None, False, nvtx=True, sync=False)
                    c_nsys = hypergraph_perf.memory_by_range(c_cmd, env, work / "c",
                                                             PORT_BATCH_RANGE[False])  # fmt: skip
                    with_count = b_nsys["range_peaks_mib"].get(APPLY_RANGE)
                    without = c_nsys["range_peaks_mib"].get(APPLY_RANGE)
                    if with_count is not None and without is not None:
                        triad_apply = round(with_count - without, 1)
                after = foreign_gpu_processes(args.gpu)
                if after:
                    raise SystemExit(f"GPU {args.gpu} had foreign compute processes: {after}")
                ratio = b_nsys["peak_live_bytes"] / a_nsys["peak_live_bytes"]
                res = {
                    "case": case,
                    "pattern": pattern,
                    "dataset": group.dataset,
                    "seed": group.options.seed,
                    "batches": len(gdoc["device_bytes"]) - 1,
                    "input": input_digest(group),
                    # The gate device_memory (D1): the peak of live device allocations.
                    "original": {
                        "peak_live_mib": a_nsys["peak_live_mib"],
                        "peak_live_bytes": a_nsys["peak_live_bytes"],
                        "peak_in": a_nsys["peak_in"],
                    },
                    "port": {
                        "peak_live_mib": b_nsys["peak_live_mib"],
                        "peak_live_bytes": b_nsys["peak_live_bytes"],
                        "peak_in": b_nsys["peak_in"],
                        "ratio": ratio,
                    },
                    "ratio": ratio,
                    # The gate store_memory (2026-10-07): store accounting per point.
                    "store_accounting": points,
                    "store_gated_points": "every point"
                    if group.dataset in STORE_EVERY_POINT
                    else "A's two print points (after the build, after the last batch)",
                    "store_worst_ratio": max(
                        (p["ratio"] for p in points if p["ratio"]), default=None
                    ),  # fmt: skip
                    "a_print_points": a_line,
                    "a_print_points_agree_with_the_golden": True,
                    # Recorded beside the gate (D1): the ranges, the footprint, the triad bytes.
                    "nsys": {"original": a_nsys, "port": b_nsys, "port_without_counting": c_nsys},
                    "triad_mib_during_apply": triad_apply,
                    "footprint_ratio_at_peak": b_nsys["footprint_at_live_peak_mib"]
                    / a_nsys["footprint_at_live_peak_mib"],
                    "footprint_ratio_max": b_nsys["footprint_peak_mib"]
                    / a_nsys["footprint_peak_mib"],
                    "footprint_ratio_between_batches": [
                        pb_["held_mib_end"] / pa_["held_mib_end"]
                        for pa_, pb_ in zip(a_nsys["batches"], b_nsys["batches"], strict=False)
                    ],
                    "foreign_gpu_processes": {"before": before, "after": after},
                    # Every measured process printed the golden's counts (and batch lines).
                    "guard": {"runs_checked": runs_guarded, "passed": True},
                }
                footprint = [
                    res["footprint_ratio_at_peak"],
                    *res["footprint_ratio_between_batches"],
                ]
                res["footprint_flagged"] = any(x > 1.05 for x in footprint)
                if not count:
                    a_peak = a_nsys["peak_live_bytes"]
                    res["preread"] = {
                        "target": 1.03,
                        "met": ratio <= 1.03,
                        "triad_budget_bytes": int(
                            1.05 * a_peak - b_nsys["peak_live_bytes"] - 0.01 * a_peak
                        ),  # fmt: skip
                        "rule": "1.05 x A_peak - port_peak (no counting) - 1% of A_peak: the "
                        "triad bytes step 3 may hold live during hypergraph.apply",
                    }
                results[key] = res
                print(f"{key}: nsys peak A {a_nsys['peak_live_mib']} MiB in "
                      f"{' > '.join(a_nsys['peak_in']) or '-'}; port {b_nsys['peak_live_mib']} "
                      f"MiB in {' > '.join(b_nsys['peak_in']) or '-'}; ratio {ratio:.4f}; "
                      f"footprint A {a_nsys['footprint_peak_mib']} port "
                      f"{b_nsys['footprint_peak_mib']} MiB; store worst "
                      f"{res['store_worst_ratio']}; triad MiB during apply {triad_apply}",
                      flush=True)  # fmt: skip
                if "preread" in res:
                    print(
                        f"  pre-read: target 1.03 {'met' if res['preread']['met'] else 'MISSED'}; "
                        f"triad budget {res['preread']['triad_budget_bytes'] / 2**20:.1f} MiB"
                    )
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    if args.json:
        doc = {
            "schema": 2,
            "algorithm": "triad_count",
            "what": "triad_count against ESCHER-GPU@abcf9b2: the Nsight Systems peak of live "
            "device allocations of both whole processes (D1) and the store accounting"
            + ("" if count else "; the port with --no-count (M9a step 2's memory pre-read)"),
            "date": started,
            "date_end": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "patterns": args.patterns,
            "reference": reference_record(),
            "protocol": {
                "nsys": "nsys profile -t cuda,nvtx --cuda-memory-usage=true (CUDA 13.1), "
                "memKind Device (cudaMalloc + cudaMallocAsync)",
                "gpu": args.gpu,
                "lock": "shared (flock -s by the caller); no foreign process on the GPU (checked "
                "before and after each case)",
                "original": "the UNPATCHED build/<pattern>, no --check",
                "port": "dyng-compat-escher run --backend cuda --nvtx"
                + ("" if count else " --no-count"),
                "guard": "every measured process (the two plain runs and the two profiled ones): "
                "the counts and batch lines it prints equal the golden case's; a failure stops "
                "the reading",
            },
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(driver),
                "build": perf_ab.port_build(driver, "cuda"),
            },
            "driver": perf_ab.driver_versions(),
            "datasets_sha256": datasets_sha256(results),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


# --- kernels (the clock-controlled kernel reading) ---------------------------------------------


def ncu_launches(cmd: list, env: dict, kernels: str, skip: int, count: int, log: Path) -> list:
    """One process under Nsight Compute at the base clocks: the metrics of `count` launches of the
    kernels matching `kernels` after the first `skip` of them, in launch order."""
    full = [
        str(perf_ab.NCU), "--clock-control", "base", "--cache-control", "none",
        "--kernel-name-base", "function", "--kernel-name", kernels,
        "--launch-skip", str(skip), "--launch-count", str(count),
        "--metrics", ",".join(NCU_METRICS), "--csv", "--log-file", str(log), *map(str, cmd),
    ]  # fmt: skip
    proc = subprocess.run(full, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stdout[-1500:] + proc.stderr[-1500:]
        raise SystemExit(f"ncu failed ({proc.returncode}):\n{tail}")
    return perf_ab.parse_ncu_csv(log.read_text(), count, str(cmd[0]))


def phase_layout(golden: dict, pattern: str) -> tuple[int, list[tuple[int, str]]]:
    """(full-count launches to skip, [(batch, phase)] of each update launch). Every phase of the
    wedge patterns launches the wedge count then the counting kernel; type 1 one key kernel per
    phase (an early out without sources launches nothing: the reading needs batches with both)."""
    per_phase = 1 if pattern == "type1" else 2
    order = []
    for b in golden["batches"]:
        if not (b["deletions"] and b["insertions"]):
            raise SystemExit(f"kernels: batch {b['index']} has no deletions or insertions")
        for phase in ("subtract", "add"):
            order += [(b["index"], phase)] * per_phase
    return per_phase, order


def kernels(args: argparse.Namespace) -> int:
    driver = find_driver(args.build).resolve()
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-ncu-", dir=perf_ab.SCRATCH / "runs"))
    results: dict = {}
    started = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for case in args.cases:
                group = group_of(case)
                for pattern in args.patterns:
                    key = case_key(case, pattern)
                    golden = json.loads((tg.set_dir(perf_ab.SCRATCH) / "cases" / case /
                                         f"{pattern}.json").read_text())  # fmt: skip
                    skip, order = phase_layout(golden, pattern)
                    a_cmd, b_cmd = commands(group, pattern, driver, None, True, sync=False)
                    sides: dict[str, list] = {"original": [], "port": []}
                    for r in range(args.rounds):
                        for side, cmd in (("original", a_cmd), ("port", b_cmd)):
                            launches = ncu_launches(cmd, env, NCU_KERNELS[side][pattern], skip,
                                                    len(order), work / f"{side}.csv")  # fmt: skip
                            sides[side].append(launches)
                        msg = []
                        for side in sides:
                            ms = sum(x["gpu__time_duration.sum"] for x in sides[side][-1]) / 1e6
                            msg.append(f"{side} {ms:.2f} ms")
                        print(f"{key} round {r + 1}/{args.rounds}: update kernels "
                              + ", ".join(msg), flush=True)  # fmt: skip
                    rows = []
                    batches = sorted({b for b, _ in order})
                    for b in batches:
                        for phase in ("subtract", "add"):
                            idx = [i for i, x in enumerate(order) if x == (b, phase)]
                            per = {}
                            for side in sides:
                                t = [sum(run[i]["gpu__time_duration.sum"] for i in idx) / 1e6
                                     for run in sides[side]]  # fmt: skip
                                d = [sum(run[i]["dram__bytes.sum"] for i in idx)
                                     for run in sides[side]]  # fmt: skip
                                per[side] = (t, d)
                            (ta, da), (tb, db) = per["original"], per["port"]
                            ma, mb = statistics.median(ta), statistics.median(tb)
                            rows.append({
                                "region": f"update.{phase}.kernels[b{b}]",
                                "reading": "kernel time, clocks locked to base (ncu)",
                                "gate_kind": "none",
                                "original_ms": ma, "port_ms": mb, "ratio": mb / ma,
                                "original_samples": ta, "port_samples": tb,
                                "original_spread": perf_ab.spread(ta),
                                "port_spread": perf_ab.spread(tb),
                                "original_dram_bytes": statistics.median(da),
                                "port_dram_bytes": statistics.median(db),
                            })  # fmt: skip
                    launch = {}
                    for side in sides:
                        launch[side] = [
                            {m: x.get(m) for m in ["name", *NCU_METRICS[3:]]}
                            | {"sm_clock_hz": x.get("sm__cycles_elapsed.avg.per_second")}
                            for x in sides[side][0][: (1 if pattern == "type1" else 2)]
                        ]
                    results[key] = {"case": case, "pattern": pattern, "rounds": args.rounds,
                                    "regions": rows, "launch": launch,
                                    "launch_order": [f"b{b}.{p}" for b, p in order]}  # fmt: skip
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    for key, res in results.items():
        print(f"{key}:")
        for e in res["regions"]:
            print(f"  {e['region']:28} original {e['original_ms']:9.2f} port {e['port_ms']:9.2f} "
                  f"ratio {e['ratio']:.3f} DRAM B/A "
                  f"{e['port_dram_bytes'] / max(1.0, e['original_dram_bytes']):.3f}")  # fmt: skip
        print(f"  launch {res['launch']}")
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "triad_count",
            "what": "the update's subtract and add kernels at locked clocks (Nsight Compute "
            "--clock-control base), recorded, not gated (ADR 0018 rule 1)",
            "date": started,
            "date_end": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference": reference_record(),
            "protocol": {
                "rounds": args.rounds,
                "order": "A/B alternation (A first in every round)",
                "gpu": args.gpu,
                "lock": perf_ab.LOCK_STATUS,
                "tool": subprocess.run(
                    [str(perf_ab.NCU), "--version"], capture_output=True, text=True
                )
                .stdout.strip()
                .splitlines()[-1],  # fmt: skip
                "ncu": "--clock-control base --cache-control none --kernel-name-base function "
                "--kernel-name <the pattern's kernels> --launch-skip <the full count's launches> "
                "--launch-count <the batches' launches>, metrics " + ",".join(NCU_METRICS),
                "kernels": NCU_KERNELS,
                "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
                "statistic": "median",
            },
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(driver),
                "build": perf_ab.port_build(driver, "cuda"),
            },
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def golden_input_digests() -> dict[str, str]:
    """{input name: loaded-rows SHA-256} of the golden set (parity/goldens.toml)."""
    doc = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    inputs = doc["sets"][tg.SET].get("inputs", {})
    return {name: spec["loaded_rows_sha256"] for name, spec in inputs.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        (
            "run",
            "A/B rounds of A against the port (the timing gates; --no-count: step 2's pre-read)",
        ),
        ("memory", "the nsys peaks of both processes and the store accounting"),
        ("kernels", "the update's kernels of both sides under ncu at the base clocks"),
    ]:
        p = sub.add_parser(name, help=help_text)
        default_cases = (
            ["syn1m_b50k_s1"] if name == "kernels" else ["coauth_b50k_s1", "syn1m_b50k_s1"]
        )
        p.add_argument("--cases", type=lambda s: s.split(","), default=default_cases)
        p.add_argument(
            "--patterns",
            "--pattern",
            dest="patterns",
            type=lambda s: [x for x in s.split(",") if x],
            default=["main"],
            help="A's executables: main (h_motif), type1, type2, type3 (comma-separated)",
        )
        p.add_argument("--gpu", type=int, default=1 if name == "memory" else 0)
        if name != "kernels":
            p.add_argument("--no-count", action="store_true", help="the port applies batches only")
        p.add_argument("--build", type=Path, help="the parity-cuda build tree of the driver")
        p.add_argument("--json", type=Path)
        if name in ("run", "kernels"):
            p.add_argument("--rounds", type=int, default=11)
            p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)
            p.add_argument("--no-lock", action="store_true")
        if name == "run":
            p.add_argument("--lock-clocks", default="base", choices=["base", "boost", "none"])
            p.add_argument("--max-foreign-cpu", type=float, default=perf_ab.MAX_FOREIGN_CPU)
            p.add_argument("--no-api-trace", action="store_true")
            p.add_argument("--enforce-gates", action="store_true",
                           help="exit 1 when a gated row exceeds its gate")  # fmt: skip
            p.add_argument(
                "--probe",
                action="store_true",
                help="the clock probe: keep every round and record whether the lock "
                "held (no round is repeated)",
            )
    args = parser.parse_args(argv)
    for p in args.patterns:
        if p not in EXES:
            parser.error(f"unknown pattern {p} (one of {sorted(EXES)})")
    if args.command in ("run", "kernels") and args.rounds < 3:
        parser.error("--rounds must be at least 3 (5 for a reading, PLAN 8.6)")
    if args.command == "run":
        return run(args)
    if args.command == "kernels":
        return kernels(args)
    return memory(args)


if __name__ == "__main__":
    sys.exit(main())
