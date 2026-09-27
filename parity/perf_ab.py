#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Performance A/B of the sssp port against the original (PLAN Sections 6.3 step 7, 8.5, 8.6).

    parity/perf_ab.py prepare [--graph roadNet-CA] [--hops 160]
    parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp [--runs 7]
                          [--graph roadNet-CA] [--batches safe50k,unsafe50k,local10k]
                          [--json parity/results/M1a-perf-openmp-roadNet-CA.json]

prepare  builds the benchmark inputs with the UNPATCHED original's own tool, exactly as the
         original's bench/prepare.sh does (MOSP-OpenMP@c352151):
           init/                         mospPrep init (Dijkstra, source 0)
           changes_50000_50/             mospPrep changes --changes 50000 --ins 50 --seed 777
           changes_50000_50_safe/        ... --safe
           changes_local_10000_50_safe/  mospPrep changes --changes 10000 --ins 50 --seed 777
                                         --local HOPS --safe (HOPS = 160 on roadNet-CA)
         into $DYNG_SCRATCH/bench/mosp/<graph>/. The CSR (K = 3 weights in [1, 100], seed 12345;
         mospPrep mtx2csr) is taken from $DYNG_SCRATCH/datasets/mosp/<graph>/csr by a symlink and
         is not copied.

run      alternates the original (A: the unpatched copy's bin/mosp) and the port (B:
         dyng-compat-mosp, parity preset) A/B/A/B for --runs rounds per batch, under the exclusive
         lock $DYNG_SCRATCH/perf.lock (flock(2); do NOT wrap this script in flock(1), it takes the
         lock itself), with OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores for both, and
         compares medians. Timed regions (parity/timed_regions/sssp.toml):
           sosp obj<k>  original "obj<k> SOSP update" (sospUpdateCpu) vs the port's
                        identify_affected + seed + loop + finalize for objective k
           sosp total   the sum over the K objectives
           apply        original apply_batch + prepare (reverse graph) vs the port's update.commit
           end to end   the RESULT lines (reading inputs to writing outputs; --no-output on both)
         Before the timed rounds of each batch, both write their outputs once and the files must be
         byte-identical (a correctness guard; not timed). Load average and the run-to-run spread
         are recorded; a spread above 10 % is flagged (PLAN Section 8.6: flagged, not failed).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import filecmp
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
COMMIT = "c35215135341d5b5d1553458afe4b2226edc38fb"
BATCHES = {
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
    "local10k": "changes_local_10000_50_safe",
}
OBJ = re.compile(r"^obj(\d+)\s+SOSP update ([0-9.]+) ms \(invalidated (\d+),", re.M)
ORIG_HOST = re.compile(r"apply batch ([0-9.]+) ms, prepare ([0-9.]+) ms")
PORT_HOST = re.compile(r"apply batch ([0-9.]+) ms")
E2E = re.compile(r"end_to_end_ms=([0-9.]+)")
THREADS = re.compile(r"threads[= ](\d+)")


def reference_copy() -> Path:
    out = subprocess.check_output([REPO / "parity" / "build_reference.sh", "--variant",
                                   "unpatched", "--print-dir", "MOSP-OpenMP"], text=True)
    return Path(out.strip().splitlines()[-1])


def check_call(cmd: list, **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def prepare(args: argparse.Namespace) -> int:
    ref = reference_copy()
    check_call([REPO / "parity" / "build_reference.sh", "--variant", "unpatched", "MOSP-OpenMP"])
    prep = ref / "bin" / "mospPrep"
    src = SCRATCH / "datasets" / "mosp" / args.graph / "csr"
    dst = SCRATCH / "bench" / "mosp" / args.graph
    dst.mkdir(parents=True, exist_ok=True)
    csr = dst / "csr"
    if not csr.exists():
        csr.symlink_to(src, target_is_directory=True)
    prefix = csr / "graphCsr"
    env = dict(os.environ, OMP_NUM_THREADS="28", OMP_PROC_BIND="close", OMP_PLACES="cores")
    steps = [
        ("init", ["init", prefix, dst / "init"]),
        (BATCHES["unsafe50k"], ["changes", prefix, dst / BATCHES["unsafe50k"], "--changes",
                                "50000", "--ins", "50", "--seed", "777"]),
        (BATCHES["safe50k"], ["changes", prefix, dst / BATCHES["safe50k"], "--changes", "50000",
                              "--ins", "50", "--seed", "777", "--safe"]),
        (BATCHES["local10k"], ["changes", prefix, dst / BATCHES["local10k"], "--changes", "10000",
                               "--ins", "50", "--seed", "777", "--local", str(args.hops), "--safe"]),
    ]
    for name, cmd in steps:
        if (dst / name).is_dir() and not args.force:
            print(f"{dst / name}: exists (use --force to regenerate)")
            continue
        print(f"mospPrep {' '.join(map(str, cmd))}", flush=True)
        check_call([prep, *cmd], env=env)
    # Record what was prepared (sizes and SHA-256 of the inputs) for the results file.
    lines = []
    for p in sorted(set(dst.rglob("*.txt")) | set(csr.glob("*.txt"))):
        digest = subprocess.check_output(["sha256sum", p], text=True).split()[0]
        lines.append(f"{digest}  {p.relative_to(dst)}")
    (dst / "INPUTS.sha256").write_text("\n".join(lines) + "\n")
    print(f"inputs in {dst} (INPUTS.sha256 written)")
    return 0


@contextlib.contextmanager
def perf_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        print(f"waiting for {path} ...", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def loadavg() -> float:
    return os.getloadavg()[0]


def run_one(cmd: list, env: dict) -> str:
    proc = subprocess.run([str(c) for c in cmd], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed:\n{proc.stdout[-2000:]}")
    return proc.stdout


def parse(log: str, side: str, k: int) -> dict:
    objs = {int(o): (float(ms), int(inv)) for o, ms, inv in OBJ.findall(log)}
    if sorted(objs) != list(range(k)):
        raise SystemExit(f"cannot parse the per-objective lines of {side}:\n{log}")
    if side == "original":
        m = ORIG_HOST.search(log)
        apply_ms = float(m.group(1)) + float(m.group(2))
    else:
        apply_ms = float(PORT_HOST.search(log).group(1))
    threads = [int(t) for t in THREADS.findall(log)]
    return {
        "sosp": [objs[o][0] for o in range(k)],
        "invalidated": [objs[o][1] for o in range(k)],
        "apply": apply_ms,
        "end_to_end": float(E2E.search(log).group(1)),
        "threads": threads[-1] if threads else None,
    }


def median(xs: list[float]) -> float:
    return statistics.median(xs)


def spread(xs: list[float]) -> float:
    """Relative spread: (max - min) / median."""
    m = median(xs)
    return (max(xs) - min(xs)) / m if m > 0 else 0.0


def gate(original_ms: float) -> float:
    return 1.05 if original_ms >= 10.0 else 1.10


def run(args: argparse.Namespace) -> int:
    ref = reference_copy()
    mosp = ref / "bin" / "mosp"
    exe = args.exe.resolve()
    data = SCRATCH / "bench" / "mosp" / args.graph
    if not (data / "init").is_dir():
        raise SystemExit(f"{data}: run `parity/perf_ab.py prepare --graph {args.graph}` first")
    k = len(list((data / "init").glob("obj*")))
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND="close",
               OMP_PLACES="cores")
    for var in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_DYNAMIC"]:
        env.pop(var, None)
    for item in args.env:  # extra settings for BOTH sides, e.g. OMP_WAIT_POLICY=active
        key, _, value = item.partition("=")
        env[key] = value
    batches = [b for b in args.batches.split(",") if b]
    results = {}
    with perf_lock(SCRATCH / "perf.lock"):
        for batch in batches:
            changes = data / BATCHES[batch]
            common = ["--graph", data / "csr" / "graphCsr", "--changes", changes, "--init",
                      data / "init"]
            # Correctness guard: both write their outputs once; the files must be identical.
            with tempfile.TemporaryDirectory(prefix="dyng-perf-", dir=SCRATCH / "runs") as t:
                t = Path(t)
                run_one([mosp, *common, "--out", t / "A"], env)
                run_one([exe, *common, "--out", t / "B"], env)
                for o in range(k):
                    for f in ["distancesUpdated.txt", "SSSPTreeUpdated.txt"]:
                        if not filecmp.cmp(t / "A" / f"obj{o}" / f, t / "B" / f"obj{o}" / f,
                                           shallow=False):
                            raise SystemExit(f"{batch}: obj{o}/{f} differs between the original "
                                             "and the port")
            print(f"{batch}: outputs byte-identical ({k} objectives)", flush=True)
            samples = {"original": [], "port": []}
            loads = []
            for r in range(args.runs):
                for side, cmd in [("original", [mosp, *common, "--no-output"]),
                                  ("port", [exe, *common, "--no-output"])]:
                    before = loadavg()
                    log = run_one(cmd, env)
                    loads.append((before, loadavg()))
                    samples[side].append(parse(log, side, k))
                print(f"{batch} round {r + 1}/{args.runs}: original sosp "
                      f"{sum(samples['original'][-1]['sosp']):.1f} ms, port "
                      f"{sum(samples['port'][-1]['sosp']):.1f} ms", flush=True)
            results[batch] = summarize(samples, k, loads)
    report(results, k, args)
    return 0


def summarize(samples: dict, k: int, loads: list) -> dict:
    out = {"regions": [], "invalidated": {}, "threads": {},
           "load_average": {"min": min(min(p) for p in loads),
                            "max": max(max(p) for p in loads)}}
    for side in ["original", "port"]:
        out["invalidated"][side] = samples[side][0]["invalidated"]
        out["threads"][side] = samples[side][0]["threads"]

    def region(name: str, pick, gated: bool) -> None:
        a = [pick(s) for s in samples["original"]]
        b = [pick(s) for s in samples["port"]]
        ma, mb = median(a), median(b)
        entry = {"region": name, "original_ms": ma, "port_ms": mb, "ratio": mb / ma,
                 "original_spread": spread(a), "port_spread": spread(b),
                 "original_samples": a, "port_samples": b}
        if gated:
            entry["gate"] = gate(ma)
            entry["within_gate"] = mb / ma <= gate(ma)
        entry["noisy"] = spread(a) > 0.10 or spread(b) > 0.10
        out["regions"].append(entry)

    for o in range(k):
        region(f"sosp obj{o}", lambda s, o=o: s["sosp"][o], True)
    region("sosp total", lambda s: sum(s["sosp"]), True)
    region("apply (batch + reverse graph)", lambda s: s["apply"], False)
    region("end to end", lambda s: s["end_to_end"], False)
    return out


def report(results: dict, k: int, args: argparse.Namespace) -> None:
    lines = [f"| batch | region | original (ms) | dynG (ms) | ratio | gate | spread A / B |",
             "|---|---|---:|---:|---:|---|---|"]
    for batch, res in results.items():
        for e in res["regions"]:
            g = (f"<= {e['gate']:.2f} {'ok' if e['within_gate'] else 'EXCEEDED'}"
                 if "gate" in e else "-")
            flag = " (noisy)" if e["noisy"] else ""
            lines.append(f"| {batch} | {e['region']} | {e['original_ms']:.2f} | "
                         f"{e['port_ms']:.2f} | {e['ratio']:.3f} | {g} | "
                         f"{e['original_spread'] * 100:.0f} % / {e['port_spread'] * 100:.0f} %"
                         f"{flag} |")
    print("\n".join(lines))
    for batch, res in results.items():
        inv = res["invalidated"]
        same = inv["original"] == inv["port"]
        print(f"{batch}: invalidated original {inv['original']} port {inv['port']} "
              f"({'equal' if same else 'DIFFERENT'}); threads {res['threads']}; "
              f"load average {res['load_average']['min']:.1f}-{res['load_average']['max']:.1f}")
    if args.json:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD"]).returncode != 0
        doc = {
            "schema": 1,
            "algorithm": "sssp",
            "backend": "openmp",
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "graph": args.graph,
            "reference": {"name": "MOSP-OpenMP", "commit": COMMIT, "variant": "unpatched",
                          "binary": str(reference_copy() / "bin" / "mosp")},
            "port": {"commit": head + ("+dirty" if dirty else ""), "binary": str(args.exe)},
            "protocol": {"runs": args.runs, "order": "A/B/A/B (original first)",
                         "threads": args.threads,
                         "env": " ".join(["OMP_PROC_BIND=close", "OMP_PLACES=cores", *args.env]),
                         "lock": "perf.lock",
                         "statistic": "median", "outputs": "--no-output on both"},
            "host": {"machine": platform.node(), "cpu": cpu_model(),
                     "kernel": platform.release()},
            "inputs_sha256": (SCRATCH / "bench" / "mosp" / args.graph / "INPUTS.sha256")
            .read_text().splitlines(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")


def cpu_model() -> str:
    with contextlib.suppress(OSError):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--graph", default="roadNet-CA")
    p.add_argument("--hops", type=int, default=160, help="local batch radius (roadNet-CA: 160)")
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--exe", type=Path, required=True, help="dyng-compat-mosp (parity preset)")
    r.add_argument("--graph", default="roadNet-CA")
    r.add_argument("--batches", default="safe50k,unsafe50k,local10k")
    r.add_argument("--runs", type=int, default=7)
    r.add_argument("--threads", type=int, default=28)
    r.add_argument("--json", type=Path)
    r.add_argument("--env", action="append", default=[], metavar="VAR=VALUE",
                   help="extra environment for both sides (repeatable)")
    args = parser.parse_args()
    if args.command == "run" and args.runs < 5:
        parser.error("--runs must be >= 5 (PLAN Section 6.3 step 7)")
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
