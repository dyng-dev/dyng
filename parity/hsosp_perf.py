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

Without --maintain-only (M11a step 4) every row of the region map: the port runs the full bench
(the update and static rows, `--digests`), each round also guarded by the distances' and parents'
digests of every batch against the golden's, and the rows of HD3 (sosp_update, static,
dynamic_comparable, dynamic per batch, load_plus_dynamic) carry their bars (1.05, 1.10 below
10 ms, 1.10 end to end).

Run them only in an exclusive-lock window of the shared calendar on GPU 0, as
`tools/locklog.py run --reset-gpu 0 EXCL M11a <step key> <what> <minutes> -- parity/hsosp_perf.py
<probe|run> ...` (the clock reset after the window runs inside the lock); `perf_ab.py hyper_sssp
...` dispatches here.

    parity/hsosp_perf.py sass [--object engine.cu.o] [--original hsospDevice.cu.o] [--json OUT]

`sass` (M11a step 4, the as-is record; no GPU): `cuobjdump -res-usage` and `-sass` of every
kernel of hyper_sssp's CUDA engine (build/parity-cuda) and of B's unpatched hsospDevice.cu.o,
paired by name (parity/triad_sass.py's reading: registers, stack, shared and local memory, the
instruction count, the hot loops and the digest of the normalized instruction stream), and the
kernels only one side has.

    parity/hsosp_perf.py launches [--configs ...] [--gpu 1] [--json OUT]

`launches` (M11a step 4, the as-is record; not a timed reading): both processes under `nsys
profile -t cuda` on the development GPU (shared lock, after tools/locklog.py quiet-ok): the
launches of every kernel and the launch, copy, memset and synchronization API calls.
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
    r"csr_rebuilds (\d+)(?: distances ([0-9a-f]{64}) parents ([0-9a-f]{64}) fallback (\d) "
    r"mismatch_static (-?\d+))?$"
)
# The gate bars of HD3 (parity/timed_regions/hyper_sssp.toml): a compute row at 1.05 where B's
# median is at least 10 ms, else 1.10 (with at least 21 rounds); an end-to-end row at 1.10.
COMPUTE_BAR = 1.05
SHORT_BAR = 1.10
SHORT_MS = 10.0
END_TO_END_BAR = 1.10
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
    far (overflow_rebuilds counts them over the run); and of its update of state A after the batch
    the digests of the distances and parents, the invalidated count (seeds) and the fallback
    decision (HD7)."""
    out, rebuilds = [], 0
    session = case["sessions"][0]
    updates = {int(c["batches"]): c for c in session.get("sssp", []) if c.get("kind") == "update"}
    for k, a in enumerate(session["apply"], 1):
        rebuilds += int(a["counters"]["bt.csr_rebuilt"])
        u = updates.get(k, {})
        out.append(
            {
                "h2h_pairs": int(a["h2h_pair_count"]),
                "pairs_sha256": a["pairs.sha256"],
                "pairs_deleted": int(a["pairs.deleted"]),
                "pairs_inserted": int(a["pairs.inserted"]),
                "overflow_rebuilds": rebuilds,
                "distances_sha256": u.get("distances.sha256"),
                "parents_sha256": u.get("parents.sha256"),
                "seeds": int(u["seed_count"]) if "seed_count" in u else None,
                "fallback": int(u["fallback"]) if "fallback" in u else None,
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
        if "seeds" in r and g.get("seeds") is not None and int(r["seeds"]) != g["seeds"]:
            problems.append(f"original b{k}: seeds {r['seeds']} != the golden's {g['seeds']}")
    return problems


def decisions_differ(rows: list[dict], port: list[dict], golden: list[dict]) -> list[str]:
    """The per-batch fallback decisions of both sides against the golden's (HD7: a differing round
    is repeated, not a guard failure)."""
    out = []
    for k, g in enumerate(golden, 1):
        if g.get("fallback") is None:
            continue
        if (
            k <= len(rows)
            and "fallback" in rows[k - 1]
            and int(rows[k - 1]["fallback"]) != g["fallback"]
        ):
            out.append(f"original b{k}: fallback {rows[k - 1]['fallback']} != {g['fallback']}")
        if (
            k <= len(port)
            and port[k - 1].get("fallback") is not None
            and port[k - 1]["fallback"] != g["fallback"]
        ):
            out.append(f"port b{k}: fallback {port[k - 1]['fallback']} != {g['fallback']}")
    return out


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
                    "distances_sha256": m.group(7),
                    "parents_sha256": m.group(8),
                    "fallback": int(m.group(9)) if m.group(9) is not None else None,
                    "mismatch_static": int(m.group(10)) if m.group(10) is not None else None,
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
        # The full update (not --maintain-only): its result's digests and update == static.
        if r.get("distances_sha256") is not None:
            for key in ("distances_sha256", "parents_sha256"):
                if r[key] != g.get(key):
                    problems.append(f"port b{k}: {key} {r[key]} != the golden's {g.get(key)}")
            if r.get("mismatch_static"):
                problems.append(f"port b{k}: mismatch_static {r['mismatch_static']}")
    return problems


def port_argv(
    exe: Path,
    path: Path,
    kind: str,
    size: int,
    golden: Path,
    work: Path,
    maintain_only: bool = True,
) -> list:
    return [
        str(exe),
        "bench",
        *hg.p_argv(path, kind, size, str(work / CSV_NAME)),
        "--golden",
        str(golden),
        "--timing",
        str(work / TIMING_NAME),
        *(["--maintain-only"] if maintain_only else []),
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


def bar_of(region: dict, original_ms: float, rounds: int) -> float | None:
    """The gate bar of a gated row (HD3), None for a reported one."""
    gate = region.get("gate", "none")
    if gate == "end_to_end":
        return END_TO_END_BAR
    if gate == "compute":
        return COMPUTE_BAR if original_ms >= SHORT_MS else SHORT_BAR
    return None


def port_value(region: dict, times: dict, batch: int) -> float:
    """The port's value of a region in one round: its driver timers (port_report: steady_clock
    around the call after a synchronization) where the map names them, else its stages."""
    names = [n for n in region.get("port_report", []) if n != perf_ab.PROCESS_WALL_TIME]
    if not names:
        names = list(region.get("port", []))
    return hypergraph_perf.region_value(times, names, batch)


def full_rows(
    regions: list[dict],
    b_rounds: list[list[dict]],
    b_e2e: list[dict],
    p_rounds: list[dict],
    b_wall: list[float],
    p_wall: list[float],
    batches: int,
) -> list[dict]:
    """Every row of parity/timed_regions/hyper_sssp.toml (ADR 0080): medians over the rounds, ratio
    port / original, the bar of the gated rows (HD3) and the verdict."""
    out = []
    for region in regions:
        keys = [(f"{region['name']}[b{k}]", k) for k in range(1, batches + 1)]
        if not region.get("per_batch"):
            keys = [(region["name"], None)]
        for row, k in keys:
            if region["name"] == "process":
                a, b = b_wall, p_wall
            else:
                a = [
                    perf_ab.hsosp_original_value(rows, e2e, region, k)
                    for rows, e2e in zip(b_rounds, b_e2e, strict=True)
                ]
                b = [port_value(region, t, 0 if k is None else k) for t in p_rounds]
            if not region.get("original_report") and region["name"] != "process":
                a = []
            ma = statistics.median(a) if a else None
            mb = statistics.median(b) if b else None
            ratio = mb / ma if ma and mb is not None else None
            bar = bar_of(region, ma or 0.0, len(b_rounds)) if ma is not None else None
            out.append(
                {
                    "row": row,
                    "gate": region.get("gate", "none"),
                    "original": region.get("original_report", []),
                    "port": region.get("port_report") or region.get("port", []),
                    "original_ms": ma,
                    "port_ms": mb,
                    "ratio": ratio,
                    "bar": bar,
                    "within": None if bar is None or ratio is None else ratio <= bar,
                    "original_rounds": a,
                    "port_rounds": b,
                }
            )
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
    p_cmd = port_argv(exe, path, kind, size, dump, work, args.maintain_only)
    b_rounds, p_rounds, windows, rejected, loads = [], [], [], [], []
    b_e2e, b_wall, p_wall, differing = [], [], [], []
    guarded = 0
    attempts = 0
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
            attempts += 1
            if not args.maintain_only:
                # HD7: a round whose fallback decisions differ from the golden's is repeated (at
                # most doubling the rounds); more than a third of differing rounds: scope differs.
                diff = decisions_differ(rows, parse_port_batches(out_p), golden)
                if diff:
                    differing.append({"attempt": attempts, "decisions": diff})
                    print(f"{name}: attempt {attempts}: decisions differ ({'; '.join(diff)})",
                          flush=True)  # fmt: skip
                    if attempts >= 2 * args.rounds:
                        break
                    continue
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
            if not args.maintain_only:
                b_e2e.append(perf_ab.parse_hsosp_e2e(out_b))
                b_wall.append(float(win_b.get("wall_s", 0.0)) * 1000.0)
                p_wall.append(float(win_p.get("wall_s", 0.0)) * 1000.0)
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
    if not args.maintain_only:
        scope = len(differing) > attempts / 3
        return {
            "config": name,
            "dataset": dataset,
            "kind": kind,
            "batch": size,
            "rounds": len(b_rounds),
            "attempts": attempts,
            "decisions_differ": differing,
            "scope_differs": scope,
            "rows": full_rows(
                perf_ab.load_hsosp_regions(),
                b_rounds,
                b_e2e,
                p_rounds,
                b_wall,
                p_wall,
                len(golden),
            ),
            "monitor": perf_ab.monitor_summary(windows, rejected, args.max_foreign_cpu),
            "loads": loads,
            "audit": audit(loads, audit_rule(args.quiet_max)),
            "guard": {"rounds_checked": guarded, "passed": True},
            "original_csv": [
                [{k: r[k] for k in perf_ab.HSOSP_CSV_TIMES} for r in x] for x in b_rounds
            ],
            "original_e2e": b_e2e,
            "port_rounds": [{f"{s}@{b}": v for (s, b), v in t.items()} for t in p_rounds],
        }
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
            om = row["original_ms"]
            verdict = row.get("verdict", "")
            if row.get("bar") is not None:
                verdict = f"bar {row['bar']}: " + ("within" if row["within"] else "ABOVE")
            print(
                f"  {row['row']:28} original {'-' if om is None else f'{om:9.2f}'} port "
                f"{row['port_ms']:9.2f} ratio {ratio if ratio is None else round(ratio, 3)} "
                f"{verdict}"
            )
        if res.get("scope_differs"):
            print(f"  scope differs: {len(res['decisions_differ'])} of {res['attempts']} attempts")
        if res["audit"]["flagged"]:
            print(f"  contamination audit: FLAGGED {res['audit']['flagged_rounds']}")
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "hyper_sssp"
            + (" (the container's maintain row)" if args.maintain_only else ""),
            "what": (
                "M11a criterion 12: the early maintain reading against MOSP_ESCHER@4b86159 "
                "(dyng-compat-hsosp bench --maintain-only; a go/no-go on the host costs, not the "
                "binding gate)"
                if args.maintain_only
                else args.what
                or "hyper_sssp against MOSP_ESCHER@4b86159: every row of "
                "parity/timed_regions/hyper_sssp.toml (dyng-compat-hsosp bench)"
            ),
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
                "port": "dyng-compat-hsosp bench"
                + (" --maintain-only" if args.maintain_only else "")
                + " --sync-stages --digests "
                "(the golden's batches; --threads = the machine's hardware threads, as B's "
                "OpenMP default)",
                "statistic": "median",
                "contamination_monitor": "rejected rounds repeated (perf_ab.MachineMonitor)",
                "guard": "every round: B's h2h_pairs, overflow_rebuilds, seeds and "
                "mismatch_static and the port's pairs digest, pair count, CSR rebuilds and (the "
                "full update) its distances' and parents' digests and mismatch_static of every "
                "batch against the golden; a failure stops the reading. The fallback decisions "
                "of both sides against the golden's: a differing round is repeated (at most "
                "doubling the rounds), more than a third of differing rounds records 'scope "
                "differs' (HD7)",
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


# hyper_sssp's kernels (cpp/src/algorithms/hyper_sssp/kernels.cuh) and B's of the same code
# (hsosp/src/hsospDevice.cu): label -> (the port's mangled-name pattern, B's).
SASS_KERNELS = {
    "fill_ll": (r"\d+fill_ll_kernel", r"\d+fillLLKernel"),
    "fill_int": (r"\d+fill_int_kernel", r"\d+fillIntKernel"),
    "pull_distances_warp": (r"\d+pull_distances_warp_kernel", r"\d+pullDistancesWarp"),
    "collect_candidates_warp": (r"\d+collect_candidates_warp_kernel", r"\d+collectCandidatesWarp"),
    "pack_state": (r"\d+pack_state_kernel", r"\d+packStateKernel"),
    "unpack_state": (r"\d+unpack_state_kernel", r"\d+unpackStateKernel"),
    "mark_deleted_tree_edges": (
        r"\d+mark_deleted_tree_edges_kernel",
        r"\d+markDeletedTreeEdgesKernel",
    ),
    "mark_ids": (r"\d+mark_ids_kernel", r"\d+markIdsKernel"),
    "init_jump": (r"\d+init_jump_kernel", r"\d+initJumpKernel"),
    "jump": (r"\d+jump_kernel", r"\d+jumpKernel"),
    "invalidate": (r"\d+invalidate_kernel", r"\d+invalidateKernel"),
    "pull_invalidated": (r"\d+pull_invalidated_kernel", r"\d+pullInvalidatedKernel"),
    "relax_inserted": (r"\d+relax_inserted_kernel", r"\d+relaxInsertedKernel"),
    "push": (r"\d+push_kernel", r"\d+pushKernel"),
    # The public mode's (dynG only: B has a virtual source row instead).
    "mark_lost_source": (r"\d+mark_lost_source_kernel", None),
    "relax_base": (r"\d+relax_base_kernel", None),
    "seed_base": (r"\d+seed_base_kernel", None),
    # B's that the port does not have: the dead degrees are zeroed by the CUDA store's apply
    # (ADR 0089), the bench's mismatch count is the compat tool's host comparison.
    "zero_degree": (None, r"\d+zeroDegreeKernel"),
    "count_mismatches": (None, r"\d+countMismatchesKernel"),
}


def sass_side(obj: Path, index: int) -> dict[str, dict]:
    import triad_sass  # noqa: PLC0415

    res = triad_sass.resource_usage(obj)
    code = triad_sass.sass(obj)
    out: dict[str, dict] = {}
    for label, patterns in SASS_KERNELS.items():
        pattern = patterns[index]
        if pattern is None:
            continue
        for name, body in code.items():
            if re.search(pattern, name):
                entry = triad_sass.analyse(name, body)
                entry.update({"function": name, "resources": res.get(name, {})})
                out[label] = entry
                break
    return out


def sass_main(args: argparse.Namespace) -> int:
    import triad_sass  # noqa: PLC0415

    port = sass_side(args.object, 0)
    orig = sass_side(args.original, 1)
    rows = []
    for label in SASS_KERNELS:
        a, b = port.get(label), orig.get(label)
        row = {"kernel": label, "port": a, "original": b}
        if a and b:
            row["same_resources"] = a["resources"] == b["resources"]
            row["same_code"] = a["digest"] == b["digest"]
        rows.append(row)
        ra = a["resources"] if a else {}
        rb = b["resources"] if b else {}
        print(
            f"{label:24} REG {ra.get('REG', '-'):>3} / {rb.get('REG', '-'):>3}  instructions "
            f"{a['instructions'] if a else '-':>5} / {b['instructions'] if b else '-':>5}"
        )
    prov = triad_sass.provenance([args.object, args.original], False)
    prov["what"] = "hyper_sssp's CUDA engine (port) and MOSP_ESCHER@4b86159's hsospDevice.cu.o"
    doc = {"record": "hyper_sssp's kernels against B's (as-is port)", "provenance": prov,
           "kernels": rows}  # fmt: skip
    if args.json:
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


# The CUDA API calls the launch record counts: the launches, the copies and memsets, and every
# host synchronization (a synchronous cudaMemcpy blocks the host as well).
LAUNCH_API = re.compile(r"LaunchKernel|Memcpy|Memset|Synchronize")


def kernel_label(name: str) -> str:
    """A demangled kernel name of nsys reduced to its identifier: no return type, no namespace,
    no template arguments, no parameters ("void cub::...::DeviceScanKernel<...>(...)" ->
    "DeviceScanKernel", "dyng::detail::hsosp::<unnamed>::push_kernel(...)" -> "push_kernel")."""
    s = name.removeprefix("void ").replace("<unnamed>", "anon")
    while True:
        t = re.sub(r"<[^<>]*>", "", s)
        if t == s:
            break
        s = t
    return s.split("(", 1)[0].split("::")[-1].strip()


def nsys_sums(nsys: str, rep: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Kernel launches by kernel_label() and the counted API calls of one nsys report."""
    import csv  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    out: list[dict[str, int]] = []
    for report, count_col in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        cmd = [nsys, "stats", "-q", "--force-export=true", "--report", report]
        r = subprocess.run(
            [*cmd, "--format", "csv", "--output", "-", str(rep)],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        start = next(i for i, ln in enumerate(lines) if "Name" in ln and count_col in ln)
        sums: dict[str, int] = {}
        for row in csv.DictReader(lines[start:]):
            label = kernel_label(row["Name"]) if report == "cuda_gpu_kern_sum" else row["Name"]
            if report == "cuda_api_sum" and not LAUNCH_API.search(label):
                continue
            sums[label] = sums.get(label, 0) + int(row[count_col])
        out.append(dict(sorted(sums.items())))
    return out[0], out[1]


def launches_main(args: argparse.Namespace) -> int:
    """`launches` (M11a step 4, the as-is record; not a timed reading): per configuration, B's
    unpatched hsospBench (B's defaults, --verify none) and the port's dyng-compat-hsosp bench (the
    golden's batches, --digests, no --sync-stages) under `nsys profile -t cuda` on the development
    GPU, CUDA_MODULE_LOADING=EAGER: the launches of every kernel (by its identifier; the H-SOSP
    kernels paired as in SASS_KERNELS) and the counts of the launch, copy, memset and
    synchronization API calls of the whole process (the load, the initial solve, the three
    batches and their static recomputes). Run it under the shared lock; every GPU-1 job first
    checks the shared calendar (tools/locklog.py quiet-ok)."""
    import subprocess  # noqa: PLC0415

    nsys = shutil.which("nsys") or "/usr/local/cuda-13.1/bin/nsys"
    exe = args.exe.resolve()
    b_exe = hg.reference_dir(args.scratch, "unpatched") / "bin" / "hsospBench"
    env = env_for(args.gpu)
    version = subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout.strip()
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dyng-hsosp-launches-", dir=args.scratch / "runs"))
    configs: dict[str, dict] = {}
    try:
        for name, dataset, kind, size in hg.p_configs():
            if name not in args.configs:
                continue
            path = hg.check_dataset(args.scratch, dataset)
            _, dump = golden_case(args.scratch, name)
            sides = {
                "original": [str(b_exe), *hg.p_argv(path, kind, size, str(work / CSV_NAME))],
                "port": [
                    str(exe),
                    "bench",
                    *hg.p_argv(path, kind, size, str(work / CSV_NAME)),
                    "--golden",
                    str(dump),
                    "--digests",
                    "--threads",
                    str(os.cpu_count() or 1),
                ],
            }
            configs[name] = {}
            for side, cmd in sides.items():
                if args.gpu == 1:
                    hg.wait_gpu1_quiet()
                rep = work / f"{side}-{name}"
                subprocess.run(
                    [nsys, "profile", "-t", "cuda", "-f", "true", "-o", str(rep), *cmd],
                    check=True,
                    env=env,
                    capture_output=True,
                )
                kernels, api = nsys_sums(nsys, rep.with_suffix(".nsys-rep"))
                configs[name][side] = {"kernels": kernels, "api": api}
            port_k, orig_k = configs[name]["port"]["kernels"], configs[name]["original"]["kernels"]
            pairs = {}
            for label, (pp, bp) in SASS_KERNELS.items():
                pk = pp.removeprefix(r"\d+") if pp else None
                bk = bp.removeprefix(r"\d+") if bp else None
                pairs[label] = {
                    "port": port_k.get(pk, 0) if pk else None,
                    "original": orig_k.get(bk, 0) if bk else None,
                }
            configs[name]["hsosp_kernels"] = pairs
            for label, c in pairs.items():
                print(f"{name} {label:24} port {c['port']} original {c['original']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    doc = {
        "record": "Kernel launches and synchronizations of hyper_sssp's bench against "
        "MOSP_ESCHER's hsospBench (nsys, the development GPU, not a timed reading)",
        "tool": "parity/hsosp_perf.py launches",
        "commit": hg.git_head(),
        "nsys": version,
        "gpu": args.gpu,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "the whole process of each side (load, initial solve, 3 batches with their "
        "static recomputes); B's other kernels are its device CBST, the port's its slack_csr "
        "store",
        "commands": {
            "original": "nsys profile -t cuda hsospBench --hg <hg> --kind <kind> --batch <size> "
            "--batches 3 --del 50 --verify none --out <csv> (ref/MOSP_ESCHER@4b86159/unpatched)",
            "port": "nsys profile -t cuda dyng-compat-hsosp bench <the same flags> --golden "
            "<case dump> --digests --threads <cores> (" + hg.portable(exe) + ")",
        },
        "configs": configs,
    }
    if args.json:
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
    ru = sub.add_parser("run", help="the A/B rounds (every row; --maintain-only: criterion 12's)")
    ru.add_argument("--configs", nargs="+", required=True)
    ru.add_argument("--rounds", type=int, default=11)
    ru.add_argument("--gpu", type=int, default=0)
    ru.add_argument("--clocks", default="base", choices=["base", "boost", "none"])
    ru.add_argument("--maintain-only", action="store_true")
    ru.add_argument("--what", help="the record's description (the reading it is)")
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
    sa = sub.add_parser("sass", help="the kernels' SASS and resources against B's (no GPU)")
    sa.add_argument(
        "--object",
        type=Path,
        default=REPO
        / "build/parity-cuda/cpp/src/algorithms/hyper_sssp/CMakeFiles/dyng_hyper_sssp.dir"
        / "engine.cu.o",
    )
    sa.add_argument(
        "--original",
        type=Path,
        default=hg.scratch_default()
        / "ref/MOSP_ESCHER@4b86159/unpatched/build/hsosp/src/hsospDevice.cu.o",
    )
    sa.add_argument("--json", type=Path)
    la = sub.add_parser("launches", help="kernel launches and API calls against B's (nsys)")
    la.add_argument(
        "--configs", nargs="+", default=["dblp_hyperedge_50000", "geology_vertex_50000"]
    )
    la.add_argument("--gpu", type=int, default=1, help="the development GPU (not a timed reading)")
    la.add_argument(
        "--exe",
        type=Path,
        default=REPO / "build" / "parity-cuda" / "tools" / "compat" / "dyng-compat-hsosp",
    )
    la.add_argument("--scratch", type=Path, default=hg.scratch_default())
    la.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "sass":
        return sass_main(args)
    if args.cmd == "launches":
        return launches_main(args)
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
