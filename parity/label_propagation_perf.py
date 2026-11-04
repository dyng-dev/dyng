#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""label_propagation against LabelPropagation-CUDA@a276a3a (O): the pre-port baseline, the A/B
gates, the device-memory readings, the clock-controlled kernels and the engine comparison (M10a;
PLAN 6.4.7, 8.6; ADRs 0018, 0021, 0024, 0064; the author's decisions AD2-AD4, B5 of 2026-10-28).

    parity/perf_ab.py label_propagation baseline [--cases 50k,500k,5m,5m1]
                      [--schedules damped,inplace] [--rounds N] [--probe-rounds 3]
                      [--default-rounds 11] [--lock-clocks probe|boost|base|none]
                      [--no-default] [--gpu 0] [--json FILE] [--skip-existing]
    parity/perf_ab.py label_propagation memory --side original|port|both [--exe ...]
                      [--cases ...] [--schedules ...] [--gpu 1] [--json FILE] [--skip-existing]
    parity/perf_ab.py label_propagation run      --exe <dyng-compat-dynlp> ...   (M10a step 4)
    parity/perf_ab.py label_propagation kernels  --exe <dyng-compat-dynlp> ...   (M10a step 5)
    parity/perf_ab.py label_propagation engines  --exe <dyng-compat-dynlp> ...   (M10b)

(or `parity/label_propagation_perf.py <command> ...`). A case is one of CASES (O's argv
'--totalNodes N --numBatches B --avgDegree 5 --delta 0.0001 --seed 42'); a schedule is damped or
inplace (O's --update); a record keys its results by '<case>/<schedule>'.

`baseline` (PLAN 6.4.7: "the damped-rule baseline for the stable gate is measured before the
port"; AD3): the UNPATCHED copy's bin/dynLP ALONE (never a gate reference: the gates re-measure O
in the same session as the port, AD3), on GPU --gpu (default 0) under the EXCLUSIVE perf lock (held
by the caller's `locklog.py run EXCL` window: DYNG_PERF_LOCK_HELD=1, else taken here), one process
per round with bench/run.sh's argv (--quiet --noValidate) and its output (the graph files and the
label file) in a fresh per-round scratch directory on the same filesystem, deleted after the round
(--noGraphFiles only at 50m), CUDA_MODULE_LOADING=EAGER, no OMP_* variables, the contamination
monitor (perf_ab.MachineMonitor; a contaminated round is repeated, at most 2 x rounds rejections)
and triad_perf's audit. Per case and schedule:
  1. with --lock-clocks probe (the default): the 3-round boost probe (ADR 0021; every round kept)
     and the lock decision: boost when every busy GPU sample of every round ran at the boost lock
     and no round was contaminated, else base;
  2. the reading at that lock: 21 rounds at 50k and 500k, 11 at 5m and 5m1 (--rounds overrides);
  3. unless --no-default, a default-clock reading of 11 rounds (recorded, not gated; ADR 0018).
Per round and batch it records time_ms, iterations, components, frontier_visits and eval_ms from
results.csv, the process wall time (end_to_end), the host peak RSS (wait4) and the disk use of the
output; per reading the medians of update[b<k>], update[sum] (the gated row of the map), eval and
end_to_end, and the short-region rule: a damped update[sum] median under 10 ms is gated at 1.10x
with at least 20 runs (PLAN 8.6). A guard checks every round: damped rounds must print the same
per-batch iterations, components, frontier_visits, accuracy and the same dynlp_labels.txt bytes
(SHA-256) as the first round; in-place rounds the same components. A guard failure stops the
reading (an exact check that is not exact). df: no round starts with less than 20 GB free on the
output's filesystem (the reading stops). --skip-existing keeps a complete '<case>/<schedule>' of
an existing --json file; a reading adds its keys to the file.

`memory --side original` (D1, B5): O's whole process under `nsys profile -t cuda,nvtx
--cuda-memory-usage=true` with the timed argv, on GPU --gpu (default 1) under the shared lock, in
the caller's GPU-1 quiet window: `nvidia-smi -i <gpu> --query-compute-apps=pid,process_name,
used_memory` before and after the reading and every 0.25 s during it (GPU_SAMPLE_PERIOD_S); a
reading during which a process other than the reading's own was on the GPU is discarded (marked
not counted) and taken again later. The peak of live device allocations (cudaMalloc +
cudaMallocAsync, kind Device; the gated metric), the NVTX ranges containing it and the live bytes
at the end of every batch's range ("batch"). `--side port` reads dyng-compat-dynlp (M10a step 4);
`--side both` (the suite's gated reading, M10a step 5) reads O's process and then the port's in the
same window, the result "<case>/<schedule>" holding both sides and the ratio, counted only when
neither saw a foreign process.

`run`, `kernels` and `engines` have their command lines and record shapes fixed here (RECORD
below; tests in parity/tests/test_label_propagation_perf.py); `run` since M10a step 4 (with the
port's host time of every update() beside its CUDA-event time per batch, a disagreement above 2 %
flagged), `kernels` since step 5, `engines` in M10b:
  run      A/B/A/B of O and dyng-compat-dynlp on GPU 0 (the gates of the map: update[sum] damped,
           end_to_end; in_place reported), at the recorded lock per case and schedule
  kernels  both sides under `ncu --clock-control base` (the first NCU_LAUNCHES launches of each
           process): per kernel the launches, the time, registers and occupancy (reported)
  engines  the port's fused against its operators engine (AD6; reported)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import io
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import perf_ab  # noqa: E402

REFERENCE = "LabelPropagation-CUDA"
COMMIT = "a276a3a4e00db1f4b6d24d2bc7ba4d9bc71b6d87"
# O's argv per case (bench/run.sh configArgs, plus 5k5 of tests/run_tests.sh).
CASES = {
    "5k5": (5000, 5),
    "50k": (50000, 10),
    "500k": (500000, 10),
    "5m": (5000000, 10),
    "5m1": (5000000, 2),
    "50m": (50000000, 11),
}
GATE_CASES = ("50k", "500k", "5m", "5m1")
SCHEDULES = {"damped": "damped", "inplace": "inplace"}  # schedule -> O's --update
# AD3: rounds of the reading at the lock (21 at 50k and 500k, 11 at 5m and 5m1); 50m: 3 (AD4).
ROUNDS = {"5k5": 21, "50k": 21, "500k": 21, "5m": 11, "5m1": 11, "50m": 3}
PROBE_ROUNDS = 3
DEFAULT_CLOCK_ROUNDS = 11
NO_GRAPH_FILES = ("50m",)  # AD2: --noGraphFiles only at 50M, on both sides
MIN_FREE_BYTES = 20 * 10**9
SHORT_REGION_MS = perf_ab.SHORT_REGION_MS
SHORT_REGION_RUNS = perf_ab.SHORT_REGION_RUNS
MAX_ROWS = 11  # update[b0..b10] in label_propagation.toml
CSV_COLUMNS = (
    "batch",
    "time_ms",
    "iterations",
    "frontier_visits",
    "components",
    "accuracy",
    "eval_ms",
)
# The record shapes of every command (schema 1): the keys of a record and of one of its results.
RECORD = {
    "baseline": {
        "record": [
            "schema",
            "algorithm",
            "command",
            "what",
            "date",
            "date_end",
            "reference",
            "harness_commit",
            "protocol",
            "driver",
            "host",
            "results",
        ],
        "result": ["case", "schedule", "argv", "batches", "probe", "lock_decision", "readings"],
        "reading": [
            "clocks",
            "rounds",
            "complete",
            "rows",
            "per_batch",
            "guard",
            "monitor",
            "audit",
            "rejected",
            "short_region",
            "rounds_data",
        ],
    },
    "memory": {
        "record": [
            "schema",
            "algorithm",
            "command",
            "what",
            "date",
            "date_end",
            "reference",
            "harness_commit",
            "protocol",
            "driver",
            "results",
        ],
        "result": [
            "case",
            "schedule",
            "side",
            "argv",
            "counted",
            "peak_live_bytes",
            "peak_live_mib",
            "peak_in",
            "batches",
            "gpu_processes",
        ],
    },
    "run": {
        "record": [
            "schema",
            "algorithm",
            "command",
            "what",
            "date",
            "date_end",
            "reference",
            "port",
            "protocol",
            "driver",
            "host",
            "results",
        ],
        "result": [
            "case",
            "schedule",
            "argv",
            "lock",
            "rounds",
            "complete",
            "regions",
            "per_batch",
            "guard",
            "monitor",
            "audit",
            "rejected",
        ],
    },
    "kernels": {
        "record": [
            "schema",
            "algorithm",
            "command",
            "what",
            "date",
            "date_end",
            "reference",
            "port",
            "protocol",
            "driver",
            "results",
        ],
        "result": ["case", "schedule", "rounds", "kernels"],
    },
    "engines": {
        "record": [
            "schema",
            "algorithm",
            "command",
            "what",
            "date",
            "date_end",
            "port",
            "protocol",
            "driver",
            "results",
        ],
        "result": ["case", "schedule", "engines", "rounds", "regions", "max_abs_diff_ulps"],
    },
}


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def case_argv(
    case: str, schedule: str, delta: str = "0.0001", seed: int = 42, avg_degree: int = 5
) -> list[str]:
    """O's argv of a case without the output flags (bench/run.sh's configArgs + the schedule)."""
    if case not in CASES:
        raise SystemExit(f"unknown case '{case}' (known: {', '.join(CASES)})")
    if schedule not in SCHEDULES:
        raise SystemExit(f"unknown schedule '{schedule}' (known: {', '.join(SCHEDULES)})")
    n, b = CASES[case]
    return [
        "--totalNodes",
        str(n),
        "--numBatches",
        str(b),
        "--avgDegree",
        str(avg_degree),
        "--delta",
        delta,
        "--seed",
        str(seed),
        "--update",
        SCHEDULES[schedule],
    ]


def timed_argv(case: str, schedule: str, out: Path) -> list[str]:
    """The timed argv (bench/run.sh: --quiet --noValidate; --noGraphFiles only at 50m)."""
    extra = ["--noGraphFiles"] if case in NO_GRAPH_FILES else []
    return [*case_argv(case, schedule), "--quiet", "--noValidate", *extra, "--output", str(out)]


def original_exe() -> Path:
    exe = perf_ab.reference_copy(REFERENCE) / "bin" / "dynLP"
    if not exe.is_file():
        raise SystemExit(f"{exe} missing: build_reference.sh --variant unpatched {REFERENCE}")
    if b"dyng-dynlp-recorder-twin" in exe.read_bytes():
        raise SystemExit(f"{exe} is a recording twin, not the unpatched original")
    return exe


def reference_record() -> dict:
    return {
        "name": REFERENCE,
        "commit": COMMIT,
        "variant": "unpatched",
        "binary": perf_ab.portable_path(original_exe()),
        "build": "its own Makefile with the defaults (parity/references.toml: make all "
        "bin/propagation_test bin/input_contract_test; ARCH=sm_86, nvcc 13.1, g++ 12.2)",
    }


def clean_env(gpu: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "DYNG_DYNLP_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return env


def parse_results(text: str) -> list[dict]:
    """O's results.csv (utils.cu:265-300) as one dict per batch; the header must be O's (without
    --validate)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != CSV_COLUMNS:
        raise SystemExit(f"results.csv: unexpected header {list(rows[0]) if rows else None}")
    out = []
    for i, r in enumerate(rows):
        if int(r["batch"]) != i:
            raise SystemExit(f"results.csv: row {i} is batch {r['batch']}")
        out.append(
            {
                "time_ms": float(r["time_ms"]),
                "iterations": int(r["iterations"]),
                "frontier_visits": int(r["frontier_visits"]),
                "components": int(r["components"]),
                "accuracy": r["accuracy"],  # text: compared exactly
                "eval_ms": float(r["eval_ms"]),
            }
        )
    return out


def disk_use(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        with contextlib.suppress(OSError):
            if p.is_file():
                total += p.stat().st_size
    return total


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


class Monitor(perf_ab.MachineMonitor):
    """perf_ab.MachineMonitor whose run() reaps the program with wait4, so that the window also
    has the program's own peak resident set (ru_maxrss of that child alone)."""

    def run(self, cmd: list, env: dict) -> tuple[str, dict]:
        cpu0 = perf_ab.cpu_busy_seconds()
        self0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.time()
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                [str(c) for c in cmd], env=env, stdout=out, stderr=subprocess.STDOUT
            )
            self.child = proc.pid
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            self.child = None
            t1 = time.time()
            out.seek(0)
            text = out.read().decode(errors="replace")
        cpu1 = perf_ab.cpu_busy_seconds()
        self1 = resource.getrusage(resource.RUSAGE_SELF)
        if proc.returncode != 0:
            what = " ".join(map(str, cmd))
            raise SystemExit(f"{what} failed ({proc.returncode}):\n{text[-2000:]}")
        mine = (self1.ru_utime + self1.ru_stime) - (self0.ru_utime + self0.ru_stime)
        program = usage.ru_utime + usage.ru_stime
        window = self.window(t0, t1, t1 - t0, cpu1 - cpu0, mine, program)
        window["max_rss_kib"] = usage.ru_maxrss
        return text, window


def one_round(
    monitor: Monitor, exe: Path, case: str, schedule: str, env: dict, work: Path, r: int
) -> tuple[dict, dict]:
    """One process of O: (its per-batch rows and facts, the monitor window)."""
    free = free_bytes(work)
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"STOP: {free / 1e9:.1f} GB free on the output's filesystem, below 20 GB: no round "
            "is started (report it)"
        )
    out = work / f"round-{r}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        text, window = monitor.run([exe, *timed_argv(case, schedule, out)], env)
        rows = parse_results((out / "results.csv").read_text())
        labels = out / "dynlp_labels.txt"
        digest = hashlib.sha256(labels.read_bytes()).hexdigest() if labels.is_file() else None
        facts = {
            "batches": rows,
            "update_sum_ms": sum(x["time_ms"] for x in rows),
            "eval_ms": sum(x["eval_ms"] for x in rows),
            "end_to_end_ms": window["wall_s"] * 1000.0,
            "max_rss_kib": window["max_rss_kib"],
            "disk_bytes": disk_use(out),
            "labels_sha256": digest,
            "tau": next(
                (x.split(":", 1)[1].strip() for x in text.splitlines() if x.startswith("Auto tau")),
                None,
            ),
        }
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return facts, window


def guard(schedule: str, first: dict, facts: dict) -> list[str]:
    """The per-round guard: damped rounds are reproducible (per-batch iterations, components,
    frontier_visits, accuracy and the label bytes); in-place rounds have the same components."""
    a, b = first["batches"], facts["batches"]
    if len(a) != len(b):
        return [f"{len(b)} batches, the first round had {len(a)}"]
    keys = ("iterations", "components", "frontier_visits", "accuracy")
    if schedule == "inplace":
        keys = ("components",)
    bad = [
        f"batch {i} {k}: {y[k]} != {x[k]}"
        for i, (x, y) in enumerate(zip(a, b, strict=True))
        for k in keys
        if x[k] != y[k]
    ]
    if schedule == "damped" and facts["labels_sha256"] != first["labels_sha256"]:
        bad.append("dynlp_labels.txt differs from the first round's")
    if facts["tau"] != first["tau"]:
        bad.append(f"tau {facts['tau']} != {first['tau']}")
    return bad


def describe(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": (max(values) - min(values)) / statistics.median(values)
        if statistics.median(values)
        else None,
        "samples": values,
    }


def summarize_rows(rounds: list[dict]) -> dict:
    """The map's rows (label_propagation.toml) over the rounds: update[b<k>] for the case's
    batches, update[sum] (gated on damped), eval (summed over the batches), end_to_end."""
    nb = len(rounds[0]["batches"])
    if nb > MAX_ROWS:
        raise SystemExit(f"{nb} batches: the map has rows update[b0..b{MAX_ROWS - 1}] only")
    rows = {
        f"update[b{k}]": describe([x["batches"][k]["time_ms"] for x in rounds]) for k in range(nb)
    }
    rows["update[sum]"] = describe([x["update_sum_ms"] for x in rounds])
    rows["eval"] = describe([x["eval_ms"] for x in rounds])
    rows["end_to_end"] = describe([x["end_to_end_ms"] for x in rounds])
    return rows


def per_batch(rounds: list[dict]) -> list[dict]:
    """Per batch: time_ms, and the counts (one value when every round has it, else all)."""
    out = []
    for k in range(len(rounds[0]["batches"])):
        e: dict = {"batch": k, "time_ms": describe([x["batches"][k]["time_ms"] for x in rounds])}
        for key in ("iterations", "components", "frontier_visits", "accuracy"):
            vals = [x["batches"][k][key] for x in rounds]
            e[key] = vals[0] if len(set(vals)) == 1 else {"per_round": vals}
            if key == "iterations" and len(set(vals)) > 1:
                e[key].update(median=statistics.median(vals), min=min(vals), max=max(vals))
        out.append(e)
    return out


def short_region(schedule: str, rows: dict) -> dict:
    """PLAN 8.6: a gated region of 10 ms or more is read at 1.05x; below 10 ms at 1.10x with at
    least 20 runs (CUDA events). Recorded per case (the gate binds on damped only, AD2)."""
    median = rows["update[sum]"]["median"]
    if median >= SHORT_REGION_MS:
        return {"update_sum_median_ms": median, "at_least_10ms": True, "rule": "compute: <= 1.05x"}
    return {
        "update_sum_median_ms": median,
        "at_least_10ms": False,
        "rule": f"short region: <= 1.10x with at least {SHORT_REGION_RUNS} runs",
    }


def reading(
    args,
    exe: Path,
    case: str,
    schedule: str,
    env: dict,
    work: Path,
    mode: str,
    rounds: int,
    probe: bool = False,
) -> dict:
    """`rounds` accepted rounds of O at `mode` (boost / base / none); with `probe` every round
    is kept and the record says whether the lock held."""
    clocks = perf_ab.ClockLock(args.gpu, mode)
    accepted, windows, rejected, loads, probe_rounds = [], [], [], [], []
    first = None
    key = f"{case}/{schedule}"
    what = "probe" if probe else f"{mode} reading"
    with clocks:
        allowed = {clocks.pid} if clocks.pid else None
        with Monitor(
            args.gpu, args.max_foreign_cpu, allowed_pids=allowed, locked=clocks.locked
        ) as monitor:
            r = 0
            while r < rounds:
                load = {"before": os.getloadavg()[0], "blocked_before": blocked()}
                facts, window = one_round(monitor, exe, case, schedule, env, work, r)
                load.update(after=os.getloadavg()[0], blocked_after=blocked())
                if first is None:
                    first = facts
                bad = guard(schedule, first, facts)
                if bad:
                    raise SystemExit(f"STOP: {key} {what} round {r + 1}: the guard failed: {bad}")
                reasons = window.get("reasons", [])
                if probe:
                    probe_rounds.append(
                        {
                            "round": r + 1,
                            "clocks_locked": (window.get("gpu") or {}).get("clocks_locked"),
                            "sm_mhz": (window.get("gpu") or {}).get("sm_mhz"),
                            "pstates_busy": (window.get("gpu") or {}).get("pstates_busy"),
                            "reasons": reasons,
                            "update_sum_ms": facts["update_sum_ms"],
                        }
                    )
                    reasons = []
                if reasons:
                    rejected.append({"before_round": r + 1, "reasons": reasons})
                    print(
                        f"{key} {what}: round {r + 1} rejected ({'; '.join(reasons)})", flush=True
                    )
                    if len(rejected) > 2 * rounds:
                        raise SystemExit(f"{key}: {len(rejected)} rounds rejected; giving up")
                    continue
                r += 1
                accepted.append(facts)
                windows.append({"original": window})
                loads.append(load)
                print(
                    f"{key} {what} {r}/{rounds}: update[sum] {facts['update_sum_ms']:.2f} ms, "
                    f"end_to_end {facts['end_to_end_ms']:.0f} ms",
                    flush=True,
                )
    rows = summarize_rows(accepted)
    out = {
        "clocks": clocks.record,
        "rounds": len(accepted),
        "complete": len(accepted) >= rounds,
        "rows": rows,
        "per_batch": per_batch(accepted),
        "guard": {
            "rounds_checked": len(accepted) + len(rejected),
            "passed": True,
            "rule": "damped: per-batch counts, accuracy and the label bytes equal the first "
            "round's; in_place: the components",
        },
        "monitor": monitor_summary(windows, rejected, args.max_foreign_cpu),
        "audit": audit(loads, windows),
        "rejected": rejected,
        "short_region": short_region(schedule, rows),
        "rounds_data": [{k: v for k, v in x.items() if k != "batches"} for x in accepted],
    }
    if probe:
        out["probe_rounds"] = probe_rounds
        out["holds"] = all(x["clocks_locked"] and not x["reasons"] for x in probe_rounds)
    return out


def monitor_summary(windows: list[dict], rejected: list[dict], limit: float) -> dict:
    """perf_ab.monitor_summary for one side (O alone): the rule, every accepted round's window,
    the rejected rounds, the foreign load, the busy P-states and SM clocks, and whether every
    busy sample ran at the lock."""
    sides = [w["original"] for w in windows]
    out: dict = {
        "rule": f"a round is repeated if the CPUs were busy outside the harness for more than "
        f"{limit} cores on average, a foreign compute process was on the timed GPU, or (locked) "
        "a busy GPU sample ran at other clocks",
        "max_foreign_cpu_cores": limit,
        "rounds": windows,
        "rejected": rejected,
        "foreign_cpu_cores_max": max((w["foreign_cpu_cores"] for w in sides), default=None),
    }
    gpus = [w["gpu"] for w in sides if "gpu" in w]
    if gpus:
        states: dict[str, int] = {}
        for g in gpus:
            for state, n in (g.get("pstates_busy") or g.get("pstates") or {}).items():
                states[state] = states.get(state, 0) + n
        sm = [g["sm_mhz"]["median"] for g in gpus if g.get("sm_mhz")]
        out["original_pstates_busy"] = states
        out["original_sm_mhz_busy_median"] = statistics.median(sm) if sm else None
        if "clocks_locked" in gpus[0]:
            out["clocks_locked_in_every_round"] = all(g["clocks_locked"] for g in gpus)
    return out


def blocked() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("procs_blocked "):
            return int(line.split()[1])
    return -1


def audit(loads: list[dict], windows: list[dict]) -> dict:
    """triad_perf's contamination audit (ADR 0056 section 4) of one side's rounds."""
    import triad_perf

    shaped = [
        {
            "before_a": x["before"],
            "after_a": x["after"],
            "blocked_before_a": x["blocked_before"],
            "blocked_after_a": x["blocked_after"],
        }
        for x in loads
    ]
    return triad_perf.audit(shaped, windows)


def load_record(path: Path | None) -> dict:
    if path is not None and path.is_file():
        return json.loads(path.read_text())
    return {}


def baseline(args: argparse.Namespace) -> int:
    exe = original_exe()
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lp-baseline-", dir=perf_ab.SCRATCH / "runs"))
    old = load_record(args.json)
    results: dict = dict(old.get("results", {}))
    started = now()
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for case in args.cases:
                for schedule in args.schedules:
                    key = f"{case}/{schedule}"
                    done = results.get(key, {})
                    if (
                        args.skip_existing
                        and done
                        and all(x["complete"] for x in done["readings"].values())
                    ):
                        print(f"{key}: skipped (complete in {args.json})")
                        continue
                    res: dict = {
                        "case": case,
                        "schedule": schedule,
                        "argv": case_argv(case, schedule)
                        + ["--quiet", "--noValidate"]
                        + (["--noGraphFiles"] if case in NO_GRAPH_FILES else []),
                        "batches": CASES[case][1],
                        "readings": {},
                    }
                    if args.lock_clocks == "probe":
                        res["probe"] = reading(
                            args,
                            exe,
                            case,
                            schedule,
                            env,
                            work,
                            "boost",
                            args.probe_rounds,
                            probe=True,
                        )
                        lock = "boost" if res["probe"]["holds"] else "base"
                        res["lock_decision"] = {
                            "lock": lock,
                            "rule": "boost when every busy GPU sample of every probe round ran "
                            "at the boost lock and no round was contaminated, else base "
                            "(ADR 0021; AD3)",
                        }
                    else:
                        res["probe"] = None
                        lock = args.lock_clocks
                        res["lock_decision"] = {"lock": lock, "rule": f"--lock-clocks {lock}"}
                    rounds = args.rounds or ROUNDS[case]
                    res["readings"]["locked" if lock != "none" else "default"] = reading(
                        args, exe, case, schedule, env, work, lock, rounds
                    )
                    if not args.no_default and lock != "none":
                        res["readings"]["default"] = reading(
                            args, exe, case, schedule, env, work, "none", args.default_rounds
                        )
                    results[key] = res
                    write_baseline(args, old, results, started)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for key, res in results.items():
        for name, rd in res["readings"].items():
            row = rd["rows"]["update[sum]"]
            print(
                f"{key} {name} ({rd['clocks'].get('control')}): update[sum] median "
                f"{row['median']:.2f} ms, end_to_end {rd['rows']['end_to_end']['median']:.0f} "
                f"ms, {rd['short_region']['rule']}"
            )
    return 0


def write_baseline(args, old: dict, results: dict, started: str) -> None:
    if not args.json:
        return
    doc = {
        "schema": 1,
        "algorithm": "label_propagation",
        "command": "baseline",
        "what": "the pre-port baseline of LabelPropagation-CUDA@a276a3a alone (PLAN 6.4.7; AD3): "
        "never a gate reference (the gates re-measure O in the same session as the port)",
        "date": old.get("date", started),
        "date_end": now(),
        "reference": reference_record(),
        "harness_commit": perf_ab.port_commit(),
        "protocol": {
            "gpu": args.gpu,
            "lock": perf_ab.LOCK_STATUS,
            "order": "O alone, one process per round",
            "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
            "argv": "bench/run.sh's: --quiet --noValidate, the graph files and the labels written "
            "into a fresh per-round scratch directory on the same filesystem and deleted after "
            "the round (--noGraphFiles only at 50m)",
            "clocks": "the 3-round boost probe's decision per case and schedule (boost if it "
            "holds, else base; ADRs 0018, 0021), then 11 rounds at default clocks (recorded)",
            "rounds": {c: ROUNDS[c] for c in CASES},
            "statistic": "median",
            "contamination_monitor": "contaminated rounds repeated (perf_ab.MachineMonitor); "
            "triad_perf's audit per reading",
            "region_map": "parity/timed_regions/label_propagation.toml",
        },
        "driver": perf_ab.driver_versions(),
        "host": {"cpu": perf_ab.cpu_model()},
        "results": dict(sorted(results.items())),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.json.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(args.json)
    print(f"wrote {args.json}", flush=True)


# --- Device memory (D1) -----------------------------------------------------------------------


# The sampling period of GpuProcesses during a reading (M10a review: at 5 s the 50k and 500k
# memory readings, a few seconds each, had no sample during them, so their sampler never saw the
# reading's own process and could not have seen a foreign one either).
GPU_SAMPLE_PERIOD_S = 0.25
# The attempts of one memory side until a sample during it saw its own process.
MEMORY_ATTEMPTS = 3


class GpuProcesses:
    """nvidia-smi --query-compute-apps on one GPU before, every GPU_SAMPLE_PERIOD_S during and
    after a reading; `foreign` lists every process seen that is not the reading's own (a
    descendant of this harness); `own_seen_during` counts the samples during the reading that saw
    its own process (a side is counted only when it is at least 1: a sampler that saw nothing
    would also have seen no foreign process)."""

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.uuid = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True,
            text=True,
        ).stdout.strip()

    def sample(self, when: str) -> dict:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout
        me = os.getpid()
        procs = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4 or parts[1] != self.uuid:
                continue
            with contextlib.suppress(ValueError):
                pid = int(parts[0])
                procs.append(
                    {
                        "pid": pid,
                        "name": parts[2],
                        "used_memory": parts[3],
                        "own": perf_ab.descends_from(pid, me),
                    }
                )
        s = {"when": when, "time": now(), "processes": procs}
        self.samples.append(s)
        return s

    def _loop(self) -> None:
        while not self._stop.wait(GPU_SAMPLE_PERIOD_S):
            self.sample("during")

    def __enter__(self) -> GpuProcesses:
        self.sample("before")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample("after")

    @property
    def foreign(self) -> list[dict]:
        return [p for s in self.samples for p in s["processes"] if not p["own"]]

    @property
    def own_seen_during(self) -> int:
        return sum(
            1
            for x in self.samples
            if x["when"] == "during" and any(p["own"] for p in x["processes"])
        )

    @property
    def counted(self) -> bool:
        return not self.foreign and self.own_seen_during > 0

    def record(self, attempts: int = 1) -> dict:
        return {
            "foreign": self.foreign,
            "samples": len(self.samples),
            "period_s": GPU_SAMPLE_PERIOD_S,
            # every sample, so the record shows the sampler saw the reading's own process
            "own_seen_in_samples": sum(
                1 for x in self.samples if any(p["own"] for p in x["processes"])
            ),
            "own_seen_during": self.own_seen_during,
            "attempts": attempts,
            "all": self.samples,
        }


def memory(args: argparse.Namespace) -> int:
    if args.side == "both":
        return memory_both(args)
    import hypergraph_perf

    if args.side == "port":
        if args.exe is None:
            raise SystemExit("memory --side port needs --exe <dyng-compat-dynlp>")
        exe = port_exe(args.exe)
        extra = ["--backend", "cuda"]
    else:
        exe = original_exe()
        extra = []
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lp-memory-", dir=perf_ab.SCRATCH / "runs"))
    old = load_record(args.json)
    results: dict = dict(old.get("results", {}))
    started = now()
    try:
        for case in args.cases:
            for schedule in args.schedules:
                key = f"{case}/{schedule}/{args.side}"
                if args.skip_existing and results.get(key, {}).get("counted"):
                    print(f"{key}: skipped (counted in {args.json})")
                    continue
                if free_bytes(work) < MIN_FREE_BYTES:
                    raise SystemExit("STOP: below 20 GB free on the output's filesystem")
                out = work / "out"
                shutil.rmtree(out, ignore_errors=True)
                out.mkdir()
                cmd = [str(exe), *timed_argv(case, schedule, out), *extra]
                attempts = 0
                while attempts < MEMORY_ATTEMPTS:
                    attempts += 1
                    with GpuProcesses(args.gpu) as gp:
                        nsys = hypergraph_perf.memory_by_range(
                            cmd, env, work / "nsys", "batch", capture=True
                        )
                    if gp.foreign or gp.own_seen_during > 0:
                        break  # a foreign process is never retried away
                    shutil.rmtree(out, ignore_errors=True)
                    out.mkdir()
                shutil.rmtree(out, ignore_errors=True)
                nsys.pop("stdout", None)
                counted = gp.counted
                results[key] = {
                    "case": case,
                    "schedule": schedule,
                    "side": args.side,
                    "argv": timed_argv(case, schedule, Path("<scratch>")) + extra,
                    "counted": counted,
                    "peak_live_bytes": nsys["peak_live_bytes"],
                    "peak_live_mib": nsys["peak_live_mib"],
                    "peak_in": nsys["peak_in"],
                    "batches": nsys["batches"],
                    "footprint_peak_mib": nsys["footprint_peak_mib"],
                    "gpu_processes": gp.record(attempts),
                }
                note = "" if counted else (
                    f" (NOT COUNTED: foreign processes {gp.foreign})" if gp.foreign
                    else " (NOT COUNTED: no sample during the reading saw its process)"
                )  # fmt: skip
                print(
                    f"{key}: peak of live device allocations {nsys['peak_live_mib']} MiB{note}",
                    flush=True,
                )
                write_memory(args, old, results, started)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all(r["counted"] for r in results.values()) else 1


def memory_side(exe: Path, extra: list[str], case: str, schedule: str, env: dict, gpu: int,
                work: Path) -> dict:  # fmt: skip
    """One profiled process of one side: its D1 peak and the GPU processes seen meanwhile."""
    import hypergraph_perf

    if free_bytes(work) < MIN_FREE_BYTES:
        raise SystemExit("STOP: below 20 GB free on the output's filesystem")
    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    cmd = [str(exe), *timed_argv(case, schedule, out), *extra]
    attempts = 0
    while attempts < MEMORY_ATTEMPTS:
        attempts += 1
        with GpuProcesses(gpu) as gp:
            nsys = hypergraph_perf.memory_by_range(cmd, env, work / "nsys", "batch", capture=True)
        if gp.foreign or gp.own_seen_during > 0:
            break  # a foreign process is never retried away
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
    shutil.rmtree(out, ignore_errors=True)
    return {
        "argv": timed_argv(case, schedule, Path("<scratch>")) + extra,
        "counted": gp.counted,
        "peak_live_bytes": nsys["peak_live_bytes"],
        "peak_live_mib": nsys["peak_live_mib"],
        "peak_in": nsys["peak_in"],
        "batches": nsys["batches"],
        "footprint_peak_mib": nsys["footprint_peak_mib"],
        "gpu_processes": gp.record(attempts),
    }


def memory_both(args: argparse.Namespace) -> int:
    """memory --side both (the suite's gated reading, M10a step 5): per case and schedule O's
    process, then the port's, each profiled alone in the caller's GPU-1 quiet window; the result
    "<case>/<schedule>" holds both sides ("original", "port") and the ratio of their peaks,
    counted only when neither side saw a foreign process."""
    if args.exe is None:
        raise SystemExit("memory --side both needs --exe <dyng-compat-dynlp>")
    p_exe = port_exe(args.exe)
    o_exe = original_exe()
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lp-memory-", dir=perf_ab.SCRATCH / "runs"))
    old = load_record(args.json)
    results: dict = dict(old.get("results", {}))
    started = now()
    try:
        for case in args.cases:
            for schedule in args.schedules:
                key = f"{case}/{schedule}"
                if args.skip_existing and results.get(key, {}).get("counted"):
                    print(f"{key}: skipped (counted in {args.json})")
                    continue
                o = memory_side(o_exe, [], case, schedule, env, args.gpu, work)
                p = memory_side(p_exe, ["--backend", "cuda"], case, schedule, env, args.gpu, work)
                counted = o["counted"] and p["counted"]
                peak = o["peak_live_bytes"]
                ratio = p["peak_live_bytes"] / peak if peak else None
                results[key] = {
                    "case": case,
                    "schedule": schedule,
                    "side": "both",
                    "argv": o["argv"],
                    "counted": counted,
                    "ratio": ratio,
                    "original": o,
                    "port": p,
                }
                note = "" if counted else " (NOT COUNTED: a foreign process on the GPU)"
                print(
                    f"{key}: peak of live device allocations O {o['peak_live_mib']} MiB, port "
                    f"{p['peak_live_mib']} MiB, ratio {ratio:.4f}{note}",
                    flush=True,
                )
                write_memory(args, old, results, started)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all(r.get("counted") for r in results.values()) else 1


def write_memory(args, old: dict, results: dict, started: str) -> None:
    if not args.json:
        return
    doc = {
        "schema": 1,
        "algorithm": "label_propagation",
        "command": "memory",
        "what": "the Nsight Systems peak of live device allocations (D1) of the whole process "
        "(O's, or dyng-compat-dynlp's with --side port)",
        "date": old.get("date", started),
        "date_end": now(),
        "reference": reference_record(),
        **(
            {"port": port_record(port_exe(args.exe))}
            if getattr(args, "side", "original") in ("port", "both")
            else {}
        ),
        "harness_commit": perf_ab.port_commit(),
        "protocol": {
            "gpu": args.gpu,
            "tool": "nsys profile -t cuda,nvtx --cuda-memory-usage=true",
            "metric": "peak of live allocations, cudaMalloc + cudaMallocAsync, kind Device, the "
            "whole process (GOVERNANCE D1)",
            "lock": "the shared lock, inside a GPU-1 quiet window (locklog.py run GPU1-QUIET; B5)",
            "foreign_processes": f"nvidia-smi before, every {GPU_SAMPLE_PERIOD_S} s during, after; "
            "a reading with a foreign process is not counted",
            "argv": "the timed argv (bench/run.sh's; --noGraphFiles at 50m only)",
        },
        "driver": perf_ab.driver_versions(),
        "results": dict(sorted(results.items())),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.json.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(args.json)
    print(f"wrote {args.json}", flush=True)


def port_side(args: argparse.Namespace) -> int:
    raise SystemExit(
        f"label_propagation {args.command}: its port side arrives in M10b (engines: the "
        "operators engine); the command line and the record shape (RECORD) are fixed here"
    )


# --- The kernels under Nsight Compute (kernels; reported) ----------------------------------------

NCU_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size",
    "launch__block_size",
)
NCU_LAUNCHES = 4000  # the first launches of a process (the setup and the first batches)


def ncu_side(exe: Path, extra: list[str], case: str, schedule: str, env: dict, work: Path,
             launches: int) -> list[dict]:  # fmt: skip
    """One process under `ncu --clock-control base` (its first `launches` kernel launches): per
    kernel name the launches, the time (sum, median), the registers per thread, the theoretical
    occupancy (the limits by registers, warps, shared memory and blocks, and the theoretical
    warps per cycle) and the achieved occupancy (median)."""
    ncu = os.environ.get("NCU", "/usr/local/cuda-13.1/bin/ncu")
    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    log = work / "ncu.csv"
    log.unlink(missing_ok=True)
    cmd = [ncu, "--clock-control", "base", "--launch-count", str(launches), "--csv", "--page",
           "raw", "--log-file", str(log), "--metrics", ",".join(NCU_METRICS),
           str(exe), *timed_argv(case, schedule, out), *extra]  # fmt: skip
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(out, ignore_errors=True)
    rows = list(csv.reader(line for line in log.read_text().splitlines() if line.startswith('"')))
    head, body = rows[0], rows[2:]
    col = {name: head.index(name) for name in ("Kernel Name", *NCU_METRICS)}

    def num(row: list[str], name: str) -> float:
        return float(row[col[name]].replace(",", ""))

    by: dict[str, list] = {}
    for row in body:
        by.setdefault(row[col["Kernel Name"]], []).append(row)
    kernels = []
    for name, launches_of in sorted(by.items()):
        times = [num(r, "gpu__time_duration.sum") / 1e3 for r in launches_of]  # us
        first = launches_of[0]
        kernels.append(
            {
                "kernel": name,
                "launches": len(launches_of),
                "time_us_sum": round(sum(times), 3),
                "time_us_median": round(statistics.median(times), 3),
                "registers_per_thread": int(num(first, "launch__registers_per_thread")),
                "block_size": int(num(first, "launch__block_size")),
                "occupancy_limit_blocks": {
                    k: int(num(first, f"launch__occupancy_limit_{k}"))
                    for k in ("registers", "warps", "shared_mem", "blocks")
                },
                "theoretical_occupancy_pct": num(first, "sm__maximum_warps_per_active_cycle_pct"),
                "achieved_occupancy_pct_median": round(
                    statistics.median(
                        num(r, "sm__warps_active.avg.pct_of_peak_sustained_active")
                        for r in launches_of
                    ),
                    2,
                ),
            }
        )
    return kernels


def kernels(args: argparse.Namespace) -> int:
    """`kernels` (reported, never gated): both sides under `ncu --clock-control base` on GPU
    --gpu (the caller's exclusive-lock window), the timed argv, the first NCU_LAUNCHES launches
    of each process; per kernel the registers and the occupancy. The fp32 hot-loop SASS of every
    ported kernel is compared with O's objects by parity/label_propagation_sass.py port (its
    record is named here)."""
    p_exe = port_exe(args.exe)
    o_exe = original_exe()
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lp-ncu-", dir=perf_ab.SCRATCH / "runs"))
    results: dict = {}
    started = now()
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for case in args.cases:
                for schedule in args.schedules:
                    kernels_of_case(args, results, case, schedule, o_exe, p_exe, env, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    write_kernels(args, results, started, p_exe)
    return 0


def kernels_of_case(args, results: dict, case: str, schedule: str, o_exe: Path, p_exe: Path,
                    env: dict, work: Path) -> None:  # fmt: skip
    key = f"{case}/{schedule}"
    o = ncu_side(o_exe, [], case, schedule, env, work, args.launches)
    p = ncu_side(p_exe, ["--backend", "cuda"], case, schedule, env, work, args.launches)
    results[key] = {
        "case": case,
        "schedule": schedule,
        "argv": timed_argv(case, schedule, Path("<scratch>")),
        "rounds": 1,
        "kernels": {"original": o, "port": p},
    }
    print(f"{key}: {len(o)} kernels of the original, {len(p)} of the port", flush=True)


def write_kernels(args, results: dict, started: str, p_exe: Path) -> None:
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "label_propagation",
            "command": "kernels",
            "what": "both sides' kernels under Nsight Compute at the base clocks (reported): per "
            "kernel the launches, the time, the registers per thread and the occupancy",
            "date": started,
            "date_end": now(),
            "reference": reference_record(),
            "port": port_record(p_exe),
            "protocol": {
                "gpu": args.gpu,
                "tool": f"ncu --clock-control base --launch-count {args.launches} --metrics "
                + ",".join(NCU_METRICS),
                "lock": perf_ab.LOCK_STATUS,
                "argv": "the timed argv (bench/run.sh's); the port adds --backend cuda",
                "sass": "parity/label_propagation_sass.py port (the ported kernels' SASS against "
                "the original's objects; parity/results/M10a-sass-port*.json)",
            },
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}", flush=True)


# --- The A/B gates against O (run; AD2, AD3) -------------------------------------------------

GATES = {"update[sum]": 1.05, "end_to_end": 1.10}  # AD2's gated rows (damped); memory: `memory`


def port_exe(path: Path) -> Path:
    exe = Path(path).resolve()
    if not exe.is_file():
        raise SystemExit(f"{exe}: no such dyng-compat-dynlp")
    return exe


def port_record(exe: Path) -> dict:
    preset = {}
    for d in [exe.parent, *exe.parents]:
        cache = d / "CMakeCache.txt"
        if cache.exists():
            for line in cache.read_text().splitlines():
                for key in (
                    "CMAKE_BUILD_TYPE",
                    "CMAKE_CUDA_FLAGS_RELEASE",
                    "DYNG_CUDA_ARCHITECTURES",
                ):
                    if line.startswith(key + ":"):
                        preset[key] = line.split("=", 1)[1]
            preset["build"] = perf_ab.portable_path(d)
            break
    return {
        "name": "dyng-compat-dynlp",
        "binary": perf_ab.portable_path(exe),
        "commit": perf_ab.port_commit(),
        "argv": "the original's timed argv + --backend cuda",
        "build": preset,
        "time_ms": "a pair of CUDA events on the resources' stream around each public "
        "label_propagation::update() call (AD2); the host wall time of the call in dyng_times.csv",
    }


def port_round(
    monitor: Monitor, exe: Path, case: str, schedule: str, env: dict, work: Path, r: int
) -> tuple[dict, dict]:
    """One process of the port: as one_round(), with --backend cuda, plus the host time of every
    update() call (dyng_times.csv)."""
    free = free_bytes(work)
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"STOP: {free / 1e9:.1f} GB free on the output's filesystem, below 20 GB: no round "
            "is started (report it)"
        )
    out = work / f"port-round-{r}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        text, window = monitor.run(
            [exe, *timed_argv(case, schedule, out), "--backend", "cuda"], env
        )
        rows = parse_results((out / "results.csv").read_text())
        host = [
            float(x["host_ms"])
            for x in csv.DictReader(io.StringIO((out / "dyng_times.csv").read_text()))
        ]
        labels = out / "dynlp_labels.txt"
        digest = hashlib.sha256(labels.read_bytes()).hexdigest() if labels.is_file() else None
        facts = {
            "batches": rows,
            "host_ms": host,
            "update_sum_ms": sum(x["time_ms"] for x in rows),
            "update_host_sum_ms": sum(host),
            "eval_ms": sum(x["eval_ms"] for x in rows),
            "end_to_end_ms": window["wall_s"] * 1000.0,
            "max_rss_kib": window["max_rss_kib"],
            "disk_bytes": disk_use(out),
            "labels_sha256": digest,
            "tau": next(
                (x.split(":", 1)[1].strip() for x in text.splitlines() if x.startswith("Auto tau")),
                None,
            ),
        }
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return facts, window


def recorded_lock(baseline_path: Path | None, key: str) -> str:
    path = baseline_path or (REPO / "parity" / "results" / "M10-baseline-2026-10-29.json")
    doc = json.loads(Path(path).read_text())
    res = doc["results"].get(key)
    if res is None or "lock_decision" not in res:
        raise SystemExit(f"{path}: no recorded lock for {key}")
    return res["lock_decision"]["lock"]


def region_ratios(o_rows: dict, p_rows: dict) -> dict:
    out = {}
    for name, orow in o_rows.items():
        prow = p_rows.get(name)
        if prow is None:
            continue
        ratio = prow["median"] / orow["median"] if orow["median"] else None
        out[name] = {
            "original_median_ms": orow["median"],
            "port_median_ms": prow["median"],
            "ratio": ratio,
            "original": orow,
            "port": prow,
        }
    return out


def ab_reading(args, o_exe: Path, p_exe: Path, case: str, schedule: str, env: dict, work: Path,
               mode: str, rounds: int) -> dict:  # fmt: skip
    """`rounds` accepted pairs at `mode`: O then the port in every round (A/B/A/B), one process
    each, both in the same session under the same lock; a pair with a contaminated side is
    repeated (at most 2 x rounds rejections)."""
    clocks = perf_ab.ClockLock(args.gpu, mode)
    key = f"{case}/{schedule}"
    o_acc, p_acc, windows, rejected, loads = [], [], [], [], []
    o_first = p_first = None
    with clocks:
        allowed = {clocks.pid} if clocks.pid else None
        with Monitor(
            args.gpu, args.max_foreign_cpu, allowed_pids=allowed, locked=clocks.locked
        ) as monitor:
            r = 0
            while r < rounds:
                load = {"before": os.getloadavg()[0], "blocked_before": blocked()}
                of, ow = one_round(monitor, o_exe, case, schedule, env, work, r)
                pf, pw = port_round(monitor, p_exe, case, schedule, env, work, r)
                load.update(after=os.getloadavg()[0], blocked_after=blocked())
                o_first = o_first or of
                p_first = p_first or pf
                bad = guard(schedule, o_first, of) + [
                    f"port: {x}" for x in guard(schedule, p_first, pf)
                ]
                if schedule == "damped" and pf["labels_sha256"] != of["labels_sha256"]:
                    bad.append("the port's dynlp_labels.txt differs from the original's (AD6)")
                if schedule == "damped":
                    for k, (x, y) in enumerate(zip(of["batches"], pf["batches"], strict=True)):
                        for c in ("iterations", "components", "frontier_visits", "accuracy"):
                            if x[c] != y[c]:
                                bad.append(f"batch {k} {c}: port {y[c]} != original {x[c]}")
                if bad:
                    raise SystemExit(f"STOP: {key} round {r + 1}: the guard failed: {bad[:5]}")
                reasons = ow.get("reasons", []) + [f"port: {x}" for x in pw.get("reasons", [])]
                if reasons:
                    rejected.append({"before_round": r + 1, "reasons": reasons})
                    print(f"{key}: round {r + 1} rejected ({'; '.join(reasons)})", flush=True)
                    if len(rejected) > 2 * rounds:
                        raise SystemExit(f"{key}: {len(rejected)} rounds rejected; giving up")
                    continue
                r += 1
                o_acc.append(of)
                p_acc.append(pf)
                windows.append({"original": ow, "port": pw})
                loads.append(load)
                print(
                    f"{key} {mode} {r}/{rounds}: update[sum] O {of['update_sum_ms']:.2f} ms, port "
                    f"{pf['update_sum_ms']:.2f} ms (host {pf['update_host_sum_ms']:.2f}); "
                    f"end_to_end O {of['end_to_end_ms']:.0f} ms, port {pf['end_to_end_ms']:.0f} ms",
                    flush=True,
                )
    o_rows, p_rows = summarize_rows(o_acc), summarize_rows(p_acc)
    p_rows["update_host[sum]"] = describe([x["update_host_sum_ms"] for x in p_acc])
    regions = region_ratios(o_rows, p_rows)
    gated = schedule == "damped"
    for name, bar in GATES.items():
        if name in regions:
            reg = regions[name]
            if name == "update[sum]" and o_rows[name]["median"] < SHORT_REGION_MS:
                bar = 1.10  # PLAN 8.6's short regions (with at least 20 runs)
            reg["bar"] = bar
            reg["gated"] = gated
            reg["within"] = reg["ratio"] is not None and reg["ratio"] <= bar
    per = []
    for k in range(len(o_acc[0]["batches"])):
        e = {
            "batch": k,
            "original": {"time_ms": describe([x["batches"][k]["time_ms"] for x in o_acc])},
            "port": {"time_ms": describe([x["batches"][k]["time_ms"] for x in p_acc])},
        }
        if all(len(x.get("host_ms", [])) > k for x in p_acc):
            # The port's host wall time of the update() call beside its CUDA-event time; a
            # median that disagrees by more than 2 % is flagged (reported, never gated).
            host = describe([x["host_ms"][k] for x in p_acc])
            event = e["port"]["time_ms"]["median"]
            e["port"]["host_ms"] = host
            e["port"]["host_vs_event"] = host["median"] / event if event else None
            e["port"]["host_disagrees"] = bool(event) and abs(host["median"] / event - 1) > 0.02
        for side, acc in (("original", o_acc), ("port", p_acc)):
            for c in ("iterations", "components", "frontier_visits"):
                vals = [x["batches"][k][c] for x in acc]
                spread = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
                e[side][c] = vals[0] if len(set(vals)) == 1 else spread
        per.append(e)
    out = {
        "clocks": clocks.record,
        "rounds": len(o_acc),
        "complete": len(o_acc) >= rounds,
        "regions": regions,
        "per_batch": per,
        "guard": {
            "rounds_checked": len(o_acc) + len(rejected),
            "passed": True,
            "rule": "damped: the port's per-batch counts, accuracy and label bytes equal the "
            "original's and their own first round's; in_place: the components",
        },
        "monitor": {
            "rule": f"a pair is repeated if either side's window had a foreign CPU load above "
            f"{args.max_foreign_cpu} cores, a foreign GPU process, or (locked) a busy sample at "
            "other clocks",
            "rounds": windows,
            "rejected": rejected,
        },
        "audit": audit(loads, [{"original": w["original"]} for w in windows]),
        "rejected": rejected,
    }
    if schedule == "inplace":
        o_it = [sum(b["iterations"] for b in x["batches"]) for x in o_acc]
        p_it = [sum(b["iterations"] for b in x["batches"]) for x in p_acc]
        out["inplace"] = {
            "original_iterations_median": statistics.median(o_it),
            "port_iterations_median": statistics.median(p_it),
            "original_ms_per_iteration": o_rows["update[sum]"]["median"] / statistics.median(o_it),
            "port_ms_per_iteration": p_rows["update[sum]"]["median"] / statistics.median(p_it),
        }
    return out


def run_gates(args: argparse.Namespace) -> int:
    o_exe = original_exe()
    p_exe = port_exe(args.exe)
    env = clean_env(args.gpu)
    (perf_ab.SCRATCH / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lp-run-", dir=perf_ab.SCRATCH / "runs"))
    old = load_record(args.json)
    results: dict = dict(old.get("results", {}))
    started = now()
    failed = []
    try:
        with perf_ab.perf_lock(perf_ab.SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for case in args.cases:
                for schedule in args.schedules:
                    key = f"{case}/{schedule}"
                    lock = (
                        recorded_lock(args.baseline, key)
                        if args.lock_clocks == "recorded"
                        else args.lock_clocks
                    )
                    rounds = args.rounds or ROUNDS[case]
                    rd = ab_reading(args, o_exe, p_exe, case, schedule, env, work, lock, rounds)
                    res = {
                        "case": case,
                        "schedule": schedule,
                        "argv": timed_argv(case, schedule, Path("<scratch>")),
                        "lock": {"lock": lock, "source": args.lock_clocks},
                        **rd,
                    }
                    results[key] = res
                    write_run(args, old, results, started)
                    for name, reg in rd["regions"].items():
                        if reg.get("gated") and not reg.get("within"):
                            failed.append(f"{key} {name} {reg['ratio']:.3f} > {reg['bar']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for key, res in results.items():
        regs = res["regions"]
        print(
            f"{key} ({res['lock']['lock']}, {res['rounds']} rounds): update[sum] "
            f"{regs['update[sum]']['ratio']:.3f}, end_to_end {regs['end_to_end']['ratio']:.3f}"
        )
    if failed:
        print("above the bar: " + "; ".join(failed))
    return 1 if (failed and args.enforce_gates) else 0


def write_run(args, old: dict, results: dict, started: str) -> None:
    if not args.json:
        return
    doc = {
        "schema": 1,
        "algorithm": "label_propagation",
        "command": "run",
        "what": "the A/B readings of dyng-compat-dynlp against LabelPropagation-CUDA@a276a3a "
        "(AD2, AD3): update[sum] and end_to_end on damped (gated), in_place reported",
        "date": old.get("date", started),
        "date_end": now(),
        "reference": reference_record(),
        "port": port_record(port_exe(args.exe)),
        "protocol": {
            "gpu": args.gpu,
            "lock": perf_ab.LOCK_STATUS,
            "order": "A/B/A/B: in every round O, then the port, one process each, the same session",
            "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
            "argv": "bench/run.sh's: --quiet --noValidate, the graph files and the labels written "
            "into a fresh per-round scratch directory on the same filesystem and deleted after "
            "the round (--noGraphFiles only at 50m); the port adds --backend cuda",
            "clocks": "the lock the baseline recorded per case and schedule (ADRs 0018, 0021)",
            "rounds": {c: ROUNDS[c] for c in CASES},
            "statistic": "median; ratio = port / original",
            "gates": GATES,
            "contamination_monitor": "contaminated pairs repeated (perf_ab.MachineMonitor); "
            "never --keep-contaminated",
            "region_map": "parity/timed_regions/label_propagation.toml",
        },
        "driver": perf_ab.driver_versions(),
        "host": {"cpu": perf_ab.cpu_model()},
        "results": dict(sorted(results.items())),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.json.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(args.json)
    print(f"wrote {args.json}", flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="perf_ab.py label_propagation", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, gpu: int, cases: str = ",".join(GATE_CASES)) -> None:
        p.add_argument("--cases", default=cases, type=lambda s: s.split(","))
        p.add_argument("--schedules", default="damped,inplace", type=lambda s: s.split(","))
        p.add_argument("--gpu", type=int, default=gpu)
        p.add_argument("--json", type=Path)

    def timed(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rounds", type=int, help="rounds of the reading (default ROUNDS[case])")
        p.add_argument("--max-foreign-cpu", type=float, default=perf_ab.MAX_FOREIGN_CPU)
        p.add_argument(
            "--no-lock",
            action="store_true",
            help="the caller holds the exclusive lock (= DYNG_PERF_LOCK_HELD=1)",
        )
        p.add_argument("--lock-timeout", type=float, default=3 * 3600.0)

    b = sub.add_parser("baseline", help="O alone: probe, the reading at the lock, default clocks")
    common(b, 0)
    timed(b)
    b.add_argument("--lock-clocks", choices=["probe", "boost", "base", "none"], default="probe")
    b.add_argument("--probe-rounds", type=int, default=PROBE_ROUNDS)
    b.add_argument("--default-rounds", type=int, default=DEFAULT_CLOCK_ROUNDS)
    b.add_argument("--no-default", action="store_true", help="no default-clock reading")
    b.add_argument("--skip-existing", action="store_true")
    m = sub.add_parser("memory", help="the nsys peak of live device allocations (D1), GPU 1")
    common(m, 1)
    m.add_argument("--side", choices=["original", "port", "both"], default="original")
    m.add_argument("--exe", type=Path, help="--side port: dyng-compat-dynlp")
    m.add_argument("--skip-existing", action="store_true")
    for name, what in [
        ("run", "the A/B gates against O (the port side: M10a steps 4-5)"),
        ("kernels", "both sides under ncu at base clocks (reported)"),
        ("engines", "the port's fused against its operators engine (reported)"),
    ]:
        r = sub.add_parser(name, help=what)
        r.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp (parity-cuda)")
        common(r, 0, ",".join(GATE_CASES) if name == "run" else "50k,500k")
        timed(r)
        if name == "run":
            r.add_argument(
                "--lock-clocks",
                choices=["recorded", "boost", "base", "none"],
                default="recorded",
                help="recorded: the baseline's lock per case and schedule",
            )
            r.add_argument("--baseline", type=Path, help="the baseline record 'recorded' reads")
            r.add_argument("--enforce-gates", action="store_true")
        if name == "engines":
            r.add_argument("--engines", default="fused,operators", type=lambda s: s.split(","))
        if name == "kernels":
            r.add_argument("--launches", type=int, default=NCU_LAUNCHES)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for case in args.cases:
        if case not in CASES:
            raise SystemExit(f"unknown case '{case}' (known: {', '.join(CASES)})")
    for schedule in args.schedules:
        if schedule not in SCHEDULES:
            raise SystemExit(f"unknown schedule '{schedule}' (known: {', '.join(SCHEDULES)})")
    # At least 3 rounds (a median), but for run's single recorded process (AD4's one in_place
    # run at 50m, M10a step 5).
    least = 1 if args.command == "run" else 3
    if getattr(args, "rounds", None) is not None and args.rounds < least:
        raise SystemExit(f"--rounds must be >= {least}")
    if os.environ.get("DYNG_PERF_LOCK_HELD") == "1" and hasattr(args, "no_lock"):
        args.no_lock = True
    return {
        "baseline": baseline,
        "memory": memory,
        "run": run_gates,
        "kernels": kernels,
        "engines": port_side,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
