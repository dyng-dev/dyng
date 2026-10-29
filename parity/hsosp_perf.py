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

Run it only in an exclusive-lock window of the shared calendar on GPU 0, as
`tools/locklog.py run --reset-gpu 0 EXCL M11a <step key> <what> <minutes> -- parity/hsosp_perf.py
probe ...` (the clock reset after the window runs inside the lock). M11a step 3 adds `run`
(the A/B gate readings of hyper_sssp.toml), `memory` and `kernels` to this file.
"""

from __future__ import annotations

import argparse
import json
import os
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
import perf_ab  # noqa: E402

# One heap layout: every run writes its CSV under one name of this length (ADR 0024).
CSV_NAME = "hsosp-ab.csv"
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
    args = p.parse_args(argv)
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
