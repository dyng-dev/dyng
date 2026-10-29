#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""hyper_sssp's performance harness against MOSP_ESCHER@4b86159 (M11a; ADR 0080, HD3 and HD4).

    parity/hsosp_perf.py probe [--gpu 0] [--rounds 3] [--json OUT]

`probe` (M11a step 1, the quiet pre-read; records only, no gate): with the UNPATCHED copy of B
(parity/build_reference.sh, `make -j16 all NVCC=/usr/local/cuda-13.1/bin/nvcc CUDA_ARCH=sm_86`)
on the timed GPU, CUDA_MODULE_LOADING=EAGER and no OMP_* variable (B runs its host work on 56
unpinned threads):

1. the clock probe of ADR 0021 per dataset (HD4): `--rounds` runs (3) of B's heaviest
   configuration of the dataset, <dataset>_hyperedge_200000, under `--clock-control boost`
   (parity/perf_ab.py ClockLock) with the contamination monitor (MachineMonitor); the lock holds
   when every busy GPU sample of every round ran at the locked clocks and no round was
   contaminated. A dataset whose probe does not hold is read at base (`--clock-control base`);
2. B's quiet pre-read (HD4): every P configuration (ADR 0080 section 3) once at the base clock
   lock, the machine's 1-minute load average and procs_blocked sampled every 0.5 s during each run
   (and at its start and end): B's own quiet maxima. The H-SOSP audit keeps the thresholds of
   ADR 0056 (load average 16, procs_blocked 4) unless B's quiet maximum of the load average
   exceeds 8; then it is 2x that maximum (or an audit of foreign_runnable instead).

    parity/hsosp_perf.py run --configs <name> ... [--rounds 11] [--gpu 0] [--clocks base]
        --maintain-only [--exe build/parity-cuda/tools/compat/dyng-compat-hsosp] [--json OUT]

`run` (M11a step 3: the early maintain reading, criterion 12; the gate readings of
parity/timed_regions/hyper_sssp.toml follow in step 5): per configuration, `--rounds` alternating
rounds (B first in every round) of the UNPATCHED B (hsospBench --hg with B's defaults and --verify
none) and the port (dyng-compat-hsosp bench, the golden's exported batches replayed, --sync-stages,
--maintain-only), both with one heap layout (the CSV, timing and work directory names of fixed
length), CUDA_MODULE_LOADING=EAGER and no OMP_* variable, the GPU clocks locked (`--clocks`: the
probe's choice of step 1, base for both datasets), the contamination monitor (a rejected round is
repeated) and the audit of ADR 0056 with step 1's thresholds. Every round is guarded: B's h2h_pairs
and overflow_rebuilds of every batch equal the golden's and its mismatch_static is 0; the port's
pairs (the SHA-256 of their canonical encoding), pair count and CSR rebuilds of every batch equal
the golden's. A failed guard stops the reading (no record). The rows: maintain[b1..b3] (B's
t_delta_ms + t_csr_apply_ms against the port's stage hypergraph.apply), delta and csr_apply
(reported). Verdict of criterion 12 per row: at most 1.03 GO, 1.03-1.05 recorded with a fix plan,
above 1.05 fixed before the acceptance.

Run them only in an exclusive-lock window of the shared calendar on GPU 0, as
`tools/locklog.py run --reset-gpu 0 EXCL M11a <step key> <what> <minutes> -- parity/hsosp_perf.py
<probe|run> ...` (the clock reset after the window runs inside the lock); `perf_ab.py hyper_sssp
...` dispatches here. Step 5 adds the gated rows of the full update, `memory` and `kernels`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import hsosp_goldens as hg  # noqa: E402
import hypergraph_perf  # noqa: E402
import perf_ab  # noqa: E402

# One heap layout: every run writes its CSV under one name of this length (ADR 0024).
CSV_NAME = "hsosp-ab.csv"
TIMING_NAME = "hsosp-ab.json"
# Criterion 12's verdicts of the early maintain reading.
EARLY_GO = 1.03
EARLY_FIX = 1.05
PORT_BATCH = re.compile(
    r"^\[batch (\d+)\] pairs ([0-9a-f]{64}) deleted (\d+) inserted (\d+) h2h_pairs (-?\d+) "
    r"csr_rebuilds (\d+)$"
)
# The contamination audit of ADR 0056 section 4 and HD4's rule for H-SOSP.
AUDIT_LOAD = 16.0
AUDIT_BLOCKED = 4
QUIET_LIMIT = 8.0  # B's own quiet maximum above which the load threshold becomes 2x it
PROBE_SIZE = 200000  # the heaviest configuration of a dataset


def blocked() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("procs_blocked "):
            return int(line.split()[1])
    return -1


class LoadSampler:
    """The machine's 1-minute load average and procs_blocked, every `period` seconds."""

    def __init__(self, period: float = 0.5) -> None:
        self.period = period
        self.samples: list[tuple[float, float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            self.samples.append((time.time(), os.getloadavg()[0], blocked()))
            if self._stop.wait(self.period):
                return

    def __enter__(self) -> LoadSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def between(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        return {
            "samples": len(s),
            "load_average_max": max((x[1] for x in s), default=None),
            "procs_blocked_max": max((x[2] for x in s), default=None),
        }


def audit_rule(quiet_max_load: float | None) -> dict:
    """HD4's audit thresholds from B's quiet maximum of the 1-minute load average."""
    if quiet_max_load is not None and quiet_max_load > QUIET_LIMIT:
        return {
            "load_average": round(2 * quiet_max_load, 2),
            "procs_blocked": AUDIT_BLOCKED,
            "why": f"B's quiet maximum {quiet_max_load:.2f} exceeds {QUIET_LIMIT}: 2x it (HD4)",
        }
    return {
        "load_average": AUDIT_LOAD,
        "procs_blocked": AUDIT_BLOCKED,
        "why": f"B's quiet maximum {quiet_max_load} is at most {QUIET_LIMIT}: ADR 0056's "
        f"thresholds kept (HD4)",
    }


def probe_holds(rounds: list[dict]) -> bool:
    """The lock holds when every round's window ran at the locked clocks with no reason."""
    return bool(rounds) and all(
        r["window"].get("gpu", {}).get("clocks_locked") is True and not r["window"].get("reasons")
        for r in rounds
    )


def env_for(gpu: int) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMP_", "DYNG_CBST_", "DYNG_HSOSP_"))
    }
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_MODULE_LOADING"] = "EAGER"
    return env


def run_b(
    monitor: perf_ab.MachineMonitor,
    sampler: LoadSampler,
    exe: Path,
    path: Path,
    kind: str,
    size: int,
    work: Path,
    env: dict,
) -> dict:
    csv = work / CSV_NAME
    csv.unlink(missing_ok=True)
    cmd = [exe, *hg.p_argv(path, kind, size, str(csv))]
    t0 = time.time()
    load0 = (os.getloadavg()[0], blocked())
    out, window = monitor.run(cmd, env)
    load1 = (os.getloadavg()[0], blocked())
    t1 = time.time()
    rows = perf_ab.parse_hsosp_csv(csv.read_text())
    e2e = perf_ab.parse_hsosp_e2e(out)
    return {
        "window": window,
        "load": {"start": load0, "end": load1, "during": sampler.between(t0, t1)},
        "csv": [
            {k: r[k] for k in sorted(perf_ab.HSOSP_CSV_TIMES)}
            | {
                "fallback": r["fallback"],
                "h2h_pairs": r["h2h_pairs"],
                "seeds": r["seeds"],
                "overflow_rebuilds": r["overflow_rebuilds"],
                "mismatch_static": r["mismatch_static"],
            }
            for r in rows
        ],
        "e2e": e2e,
    }


def probe(args: argparse.Namespace) -> int:
    exe = hg.reference_dir(args.scratch, "unpatched") / "bin" / "hsospBench"
    env = env_for(args.gpu)
    work = Path(tempfile.mkdtemp(prefix="dyng-hsosp-probe-", dir=args.scratch / "runs"))
    record: dict = {
        "what": "M11a step 1: the clock probe per dataset and B's quiet pre-read "
        "(HD4; records only)",
        "reference": hg.REFERENCE,
        "commit": hg.COMMIT,
        "exe": hg.portable(exe),
        "gpu": args.gpu,
        "env": {"CUDA_MODULE_LOADING": "EAGER", "OMP_*": "unset"},
        "lock": {k: os.environ.get(k) for k in ("DYNG_PERF_LOCK", "DYNG_PERF_LOCK_HELD")},
        "layout": f"one (--out <work>/{CSV_NAME} on every run)",
        "versions": perf_ab.driver_versions(),
        "probe": {},
        "pre_read": {},
    }
    try:
        with LoadSampler() as sampler:
            for dataset in [d for d in hg.DATASETS if d in args.datasets]:
                path = hg.check_dataset(args.scratch, dataset)
                with (
                    perf_ab.ClockLock(args.gpu, args.probe_lock) as clocks,
                    perf_ab.MachineMonitor(
                        args.gpu,
                        allowed_pids={clocks.pid} if clocks.pid else None,
                        locked=clocks.locked,
                    ) as monitor,
                ):
                    rounds = []
                    for r in range(args.rounds):
                        res = run_b(monitor, sampler, exe, path, "hyperedge", PROBE_SIZE, work, env)
                        rounds.append({"round": r + 1, **res})
                        w = res["window"]
                        print(
                            f"probe {dataset}_hyperedge_{PROBE_SIZE} round {r + 1}: clocks locked "
                            f"{w.get('gpu', {}).get('clocks_locked')}, sm "
                            f"{w.get('gpu', {}).get('sm_mhz')} {w.get('reasons')}",
                            flush=True,
                        )
                    record["probe"][dataset] = {
                        "config": hg.config_name(dataset, "hyperedge", PROBE_SIZE),
                        "clock_lock": clocks.record,
                        "rounds": rounds,
                        "holds": probe_holds(rounds),
                        "clock": "boost" if probe_holds(rounds) else "base",
                    }
            with (
                perf_ab.ClockLock(args.gpu, args.base_lock) as clocks,
                perf_ab.MachineMonitor(
                    args.gpu,
                    allowed_pids={clocks.pid} if clocks.pid else None,
                    locked=clocks.locked,
                ) as monitor,
            ):
                record["pre_read_clock_lock"] = clocks.record
                for name, dataset, kind, size in hg.p_configs():
                    if args.configs and name not in args.configs:
                        continue
                    path = hg.check_dataset(args.scratch, dataset)
                    res = run_b(monitor, sampler, exe, path, kind, size, work, env)
                    record["pre_read"][name] = res
                    d = res["load"]["during"]
                    print(
                        f"pre-read {name}: load average max {d['load_average_max']}, "
                        f"procs_blocked max {d['procs_blocked_max']}, dynamic "
                        f"{[round(r['t_dynamic_total_ms'], 1) for r in res['csv']]} ms",
                        flush=True,
                    )
        loads = [
            x
            for r in record["pre_read"].values()
            for x in [
                r["load"]["during"]["load_average_max"],
                r["load"]["start"][0],
                r["load"]["end"][0],
            ]
            if x is not None
        ]
        blocks = [
            x
            for r in record["pre_read"].values()
            for x in [
                r["load"]["during"]["procs_blocked_max"],
                r["load"]["start"][1],
                r["load"]["end"][1],
            ]
            if x is not None
        ]
        quiet = max(loads, default=None)
        record["quiet_maxima"] = {
            "load_average": quiet,
            "procs_blocked": max(blocks, default=None),
            "load_average_median": statistics.median(loads) if loads else None,
        }
        record["audit"] = audit_rule(quiet)
        record["clocks"] = {d: p["clock"] for d, p in record["probe"].items()}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if args.json:
            args.json.write_text(json.dumps(record, indent=1) + "\n")
    print(
        json.dumps(
            {
                "clocks": record.get("clocks"),
                "quiet_maxima": record.get("quiet_maxima"),
                "audit": record.get("audit"),
            },
            indent=1,
        )
    )
    return 0


# --- run: the A/B rounds ------------------------------------------------------------------------


def golden_case(scratch: Path, name: str) -> tuple[dict, Path]:
    """The golden case of a P configuration: its case.json and its dump."""
    d = hg.set_dir(scratch) / "paper" / name
    return json.loads((d / "case.json").read_text()), d / "dump.hsd.zst"


def golden_batches(case: dict) -> list[dict]:
    """Per batch what the guards compare: B's pair count and pairs digest, and the CSR rebuilds so
    far (overflow_rebuilds counts them over the run)."""
    out, rebuilds = [], 0
    for a in case["sessions"][0]["apply"]:
        rebuilds += int(a["counters"]["bt.csr_rebuilt"])
        out.append(
            {
                "h2h_pairs": int(a["h2h_pair_count"]),
                "pairs_sha256": a["pairs.sha256"],
                "pairs_deleted": int(a["pairs.deleted"]),
                "pairs_inserted": int(a["pairs.inserted"]),
                "overflow_rebuilds": rebuilds,
            }
        )
    return out


def guard_original(rows: list[dict], golden: list[dict]) -> list[str]:
    """B's round: its h2h_pairs and overflow_rebuilds of every batch are the golden's, and its
    update equals its static recompute (mismatch_static 0)."""
    problems = []
    if len(rows) != len(golden):
        return [f"original: {len(rows)} CSV rows, the golden has {len(golden)} batches"]
    for k, (r, g) in enumerate(zip(rows, golden, strict=True), 1):
        if int(r["h2h_pairs"]) != g["h2h_pairs"]:
            problems.append(f"original b{k}: h2h_pairs {r['h2h_pairs']} != {g['h2h_pairs']}")
        if int(r["overflow_rebuilds"]) != g["overflow_rebuilds"]:
            problems.append(
                f"original b{k}: overflow_rebuilds {r['overflow_rebuilds']} != "
                f"{g['overflow_rebuilds']}"
            )
        if int(r["mismatch_static"]) != 0:
            problems.append(f"original b{k}: mismatch_static {r['mismatch_static']}")
    return problems


def parse_port_batches(out: str) -> list[dict]:
    rows = []
    for line in out.splitlines():
        m = PORT_BATCH.match(line.strip())
        if m:
            rows.append(
                {
                    "batch": int(m.group(1)),
                    "pairs_sha256": m.group(2),
                    "pairs_deleted": int(m.group(3)),
                    "pairs_inserted": int(m.group(4)),
                    "h2h_pairs": int(m.group(5)),
                    "csr_rebuilds": int(m.group(6)),
                }
            )
    return rows


def guard_port(out: str, golden: list[dict]) -> list[str]:
    """The port's round: its pairs (by digest), pair count and CSR rebuilds of every batch are the
    golden's."""
    rows = parse_port_batches(out)
    if len(rows) != len(golden):
        return [f"port: {len(rows)} batch lines, the golden has {len(golden)} batches"]
    problems = []
    for k, (r, g) in enumerate(zip(rows, golden, strict=True), 1):
        for key, want in (
            ("pairs_sha256", g["pairs_sha256"]),
            ("pairs_deleted", g["pairs_deleted"]),
            ("pairs_inserted", g["pairs_inserted"]),
            ("h2h_pairs", g["h2h_pairs"]),
            ("csr_rebuilds", g["overflow_rebuilds"]),
        ):
            if r[key] != want:
                problems.append(f"port b{k}: {key} {r[key]} != the golden's {want}")
    return problems


def port_argv(exe: Path, path: Path, kind: str, size: int, golden: Path, work: Path) -> list:
    return [
        str(exe),
        "bench",
        *hg.p_argv(path, kind, size, str(work / CSV_NAME)),
        "--golden",
        str(golden),
        "--timing",
        str(work / TIMING_NAME),
        "--maintain-only",
        "--sync-stages",
        "--digests",
        "--threads",
        str(os.cpu_count() or 1),
    ]


def original_rows(text: str) -> list[dict]:
    return perf_ab.parse_hsosp_csv(text)


def maintain_rows(b_rounds: list[list[dict]], p_rounds: list[dict], batches: int) -> list[dict]:
    """maintain[bk] (gated by criterion 12's early verdict) and the reported delta[bk],
    csr_apply[bk]: medians over the rounds, ratio port / original."""
    out = []
    for k in range(1, batches + 1):
        for name, original, port in (
            ("maintain", ("t_delta_ms", "t_csr_apply_ms"), ["hypergraph.apply"]),
            (
                "delta",
                ("t_delta_ms",),
                [
                    "hypergraph.apply.normalize",
                    "hypergraph.apply.ids",
                    "hypergraph.apply.incidence",
                    "hypergraph.apply.translate",
                ],
            ),
            ("csr_apply", ("t_csr_apply_ms",), ["hypergraph.apply.line_graph"]),
        ):
            a = [sum(float(r[k - 1][c]) for c in original) for r in b_rounds]
            b = [hypergraph_perf.region_value(t, port, k) for t in p_rounds]
            ma, mb = statistics.median(a), statistics.median(b)
            row = {
                "row": f"{name}[b{k}]",
                "original": "+".join(original),
                "port": port,
                "original_ms": ma,
                "port_ms": mb,
                "ratio": mb / ma if ma > 0 else None,
                "original_rounds": a,
                "port_rounds": b,
            }
            if name == "maintain":
                r = row["ratio"]
                row["verdict"] = (
                    "GO"
                    if r is not None and r <= EARLY_GO
                    else "recorded, fix plan"
                    if r is not None and r <= EARLY_FIX
                    else "above 1.05: fix"
                )
            out.append(row)
    return out


def audit(loads: list[dict], rule: dict) -> dict:
    flagged = [
        i + 1
        for i, x in enumerate(loads)
        if (x.get("load_average_max") or 0) > rule["load_average"]
        or (x.get("procs_blocked_max") or 0) > rule["procs_blocked"]
    ]
    return {"rule": rule, "flagged_rounds": flagged, "flagged": bool(flagged)}


def run_config(args, name, dataset, kind, size, exe, env, clocks, sampler, work, b_exe) -> dict:
    path = hg.check_dataset(args.scratch, dataset)
    case, dump = golden_case(args.scratch, name)
    golden = golden_batches(case)
    b_cmd = [str(b_exe), *hg.p_argv(path, kind, size, str(work / CSV_NAME))]
    p_cmd = port_argv(exe, path, kind, size, dump, work)
    b_rounds, p_rounds, windows, rejected, loads = [], [], [], [], []
    guarded = 0
    with perf_ab.MachineMonitor(
        args.gpu,
        args.max_foreign_cpu,
        allowed_pids={clocks.pid} if clocks.pid else None,
        locked=clocks.locked,
    ) as monitor:
        r = 0
        while r < args.rounds:
            (work / CSV_NAME).unlink(missing_ok=True)
            t0 = time.time()
            out_b, win_b = monitor.run(b_cmd, env)
            rows = original_rows((work / CSV_NAME).read_text())
            (work / CSV_NAME).unlink(missing_ok=True)
            (work / TIMING_NAME).unlink(missing_ok=True)
            out_p, win_p = monitor.run(p_cmd, env)
            t1 = time.time()
            timing = hypergraph_perf.stage_times(work / TIMING_NAME)
            bad = guard_original(rows, golden) + guard_port(out_p, golden)
            if bad:
                raise SystemExit(f"{name}: round {r + 1}: the guard failed: {bad}")
            guarded += 1
            reasons = win_b.get("reasons", []) + win_p.get("reasons", [])
            load = sampler.between(t0, t1)
            if reasons:
                rejected.append({"before_round": r + 1, "reasons": reasons})
                print(f"{name}: round {r + 1} rejected ({'; '.join(reasons)})", flush=True)
                if len(rejected) > 2 * args.rounds:
                    raise SystemExit(f"{name}: {len(rejected)} rounds rejected; giving up")
                continue
            r += 1
            b_rounds.append(rows)
            p_rounds.append(timing)
            windows.append({"original": win_b, "port": win_p})
            loads.append(load)
            m = [
                (
                    round(
                        float(rows[k - 1]["t_delta_ms"]) + float(rows[k - 1]["t_csr_apply_ms"]), 1
                    ),
                    round(hypergraph_perf.region_value(timing, ["hypergraph.apply"], k), 1),
                )
                for k in range(1, len(golden) + 1)
            ]
            print(f"{name} round {r}/{args.rounds}: maintain (B, port) {m}", flush=True)
    return {
        "config": name,
        "dataset": dataset,
        "kind": kind,
        "batch": size,
        "rounds": len(b_rounds),
        "rows": maintain_rows(b_rounds, p_rounds, len(golden)),
        "monitor": perf_ab.monitor_summary(windows, rejected, args.max_foreign_cpu),
        "loads": loads,
        "audit": audit(loads, audit_rule(args.quiet_max)),
        "guard": {"rounds_checked": guarded, "passed": True},
        "original_csv": [
            [{k: r[k] for k in ("t_delta_ms", "t_csr_apply_ms")} for r in x] for x in b_rounds
        ],
        "port_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()} for t in p_rounds],
    }


def run(args: argparse.Namespace) -> int:
    if not args.maintain_only:
        print("hsosp_perf.py run: the full rows arrive with hyper_sssp (M11a step 5); use "
              "--maintain-only", file=sys.stderr)  # fmt: skip
        return 2
    exe = args.exe.resolve()
    b_exe = hg.reference_dir(args.scratch, "unpatched") / "bin" / "hsospBench"
    env = env_for(args.gpu)
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-hsosp-run-", dir=args.scratch / "runs"))
    started = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}
    clocks = perf_ab.ClockLock(args.gpu, args.clocks)
    try:
        with LoadSampler() as sampler, clocks:
            for name, dataset, kind, size in hg.p_configs():
                if name not in args.configs:
                    continue
                results[name] = run_config(
                    args, name, dataset, kind, size, exe, env, clocks, sampler, work, b_exe
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for name, res in results.items():
        print(f"{name} (GPU {args.gpu}; clocks {clocks.record.get('locked')}):")
        for row in res["rows"]:
            ratio = row["ratio"]
            print(
                f"  {row['row']:16} original {row['original_ms']:9.2f} port {row['port_ms']:9.2f} "
                f"ratio {ratio if ratio is None else round(ratio, 3)} {row.get('verdict', '')}"
            )
        if res["audit"]["flagged"]:
            print(f"  contamination audit: FLAGGED {res['audit']['flagged_rounds']}")
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "hyper_sssp (the container's maintain row)",
            "what": "M11a criterion 12: the early maintain reading against MOSP_ESCHER@4b86159 "
            "(dyng-compat-hsosp bench --maintain-only; a go/no-go on the host costs, not the "
            "binding gate)",
            "date": started,
            "date_end": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "protocol": {
                "rounds": args.rounds,
                "order": "A/B alternation (B first in every round)",
                "gpu": args.gpu,
                "clocks": clocks.record,
                "lock": {k: os.environ.get(k) for k in ("DYNG_PERF_LOCK", "DYNG_PERF_LOCK_HELD")},
                "layout": f"one (--out <work>/{CSV_NAME}, --timing <work>/{TIMING_NAME})",
                "env": "CUDA_MODULE_LOADING=EAGER, no OMP_* variables",
                "original": "the UNPATCHED copy's bin/hsospBench --hg, B's defaults, --verify none",
                "port": "dyng-compat-hsosp bench --maintain-only --sync-stages --digests "
                "(the golden's batches; --threads = the machine's hardware threads, as B's "
                "OpenMP default)",
                "statistic": "median",
                "contamination_monitor": "rejected rounds repeated (perf_ab.MachineMonitor)",
                "guard": "every round: B's h2h_pairs, overflow_rebuilds and mismatch_static "
                "and the port's pairs digest, pair count and CSR rebuilds of every batch against "
                "the golden; a failure stops the reading",
                "verdict": "maintain[bk] <= 1.03 GO; 1.03-1.05 recorded with a fix plan; above "
                "1.05 fixed (criterion 12)",
            },
            "reference": {"repo": hg.REFERENCE, "commit": hg.COMMIT, "exe": hg.portable(b_exe)},
            "port": {
                "commit": perf_ab.port_commit(),
                "binary": perf_ab.portable_path(exe),
                "build": perf_ab.port_build(exe, "cuda"),
            },
            "driver": perf_ab.driver_versions(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="the clock probe per dataset and B's quiet pre-read")
    pr.add_argument("--gpu", type=int, default=0)
    pr.add_argument("--rounds", type=int, default=3)
    pr.add_argument("--scratch", type=Path, default=hg.scratch_default())
    pr.add_argument("--json", type=Path)
    # A dry run (outside the reading: another GPU, the default clocks, fewer cases).
    pr.add_argument("--datasets", nargs="+", default=list(hg.DATASETS))
    pr.add_argument("--configs", nargs="*", help="the pre-read's configurations (default: all 16)")
    pr.add_argument("--probe-lock", default="boost", choices=["boost", "base", "none"])
    pr.add_argument("--base-lock", default="base", choices=["base", "none"])
    pr.add_argument(
        "--dry-run",
        action="store_true",
        help="a check of the harness outside any window: GPU 1, default clocks, no "
        "record of a reading",
    )
    ru = sub.add_parser("run", help="the A/B rounds (M11a step 3: --maintain-only)")
    ru.add_argument("--configs", nargs="+", required=True)
    ru.add_argument("--rounds", type=int, default=11)
    ru.add_argument("--gpu", type=int, default=0)
    ru.add_argument("--clocks", default="base", choices=["base", "boost", "none"])
    ru.add_argument("--maintain-only", action="store_true")
    ru.add_argument(
        "--exe",
        type=Path,
        default=REPO / "build" / "parity-cuda" / "tools" / "compat" / "dyng-compat-hsosp",
    )
    ru.add_argument("--scratch", type=Path, default=hg.scratch_default())
    ru.add_argument("--json", type=Path)
    ru.add_argument("--max-foreign-cpu", type=float, default=perf_ab.MAX_FOREIGN_CPU)
    ru.add_argument(
        "--quiet-max",
        type=float,
        default=7.03,
        help="B's quiet maximum load average of step 1's pre-read (M11a-quiet-pre-read.json)",
    )
    ru.add_argument(
        "--dry-run",
        action="store_true",
        help="a check of the harness outside any window: GPU 1, default clocks, no record",
    )
    args = p.parse_args(argv)
    if args.cmd == "run":
        if args.dry_run:
            if args.gpu == 0:
                print("hsosp_perf.py: a dry run never uses GPU 0 (the timing GPU)", file=sys.stderr)
                return 2
            args.clocks = "none"
            return run(args)
        if os.environ.get("DYNG_PERF_LOCK_HELD") != "1":
            print(
                "hsosp_perf.py: run it inside an exclusive-lock window (tools/locklog.py run EXCL)",
                file=sys.stderr,
            )
            return 2
        return run(args)
    if args.dry_run:
        if args.gpu == 0:
            print("hsosp_perf.py: a dry run never uses GPU 0 (the timing GPU)", file=sys.stderr)
            return 2
        args.probe_lock = args.base_lock = "none"
        return probe(args)
    if os.environ.get("DYNG_PERF_LOCK_HELD") != "1":
        print(
            "hsosp_perf.py: run it inside an exclusive-lock window (tools/locklog.py run EXCL)",
            file=sys.stderr,
        )
        return 2
    return probe(args)


if __name__ == "__main__":
    sys.exit(main())
