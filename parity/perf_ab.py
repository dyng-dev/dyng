#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Performance A/B of the sssp port against the original (PLAN Sections 6.3 step 7, 8.5, 8.6).

    parity/perf_ab.py prepare [--graph roadNet-CA] [--hops 160]
    parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp [--runs 21]
                          [--graph roadNet-CA] [--batches safe50k,unsafe50k,local10k]
                          [--json parity/results/M1b-perf-openmp-roadNet-CA.json]
    parity/perf_ab.py run --backend cuda --exe build/parity-cuda/tools/compat/dyng-compat-mosp
                          [--gpu 0] [--runs 21] [--graph roadNet-CA] [--json ...]

The graphs of the PLAN 6.4.2 gate are the directories of $DYNG_SCRATCH/datasets/mosp: roadNet-PA,
roadNet-CA, rgg (rgg_n_2_20_s0) and road_usa_g (road_usa); --hops defaults to the local-batch
radius the original's bench/prepare.sh names for each (110, 160, 110, 200).

prepare  builds the benchmark inputs with the UNPATCHED original's own tool, exactly as the
         original's bench/prepare.sh does (MOSP-OpenMP@c352151):
           init/                         mospPrep init (Dijkstra, source 0)
           changes_50000_50/             mospPrep changes --changes 50000 --ins 50 --seed 777
           changes_50000_50_safe/        ... --safe
           changes_local_10000_50_safe/  mospPrep changes --changes 10000 --ins 50 --seed 777
                                         --local HOPS --safe (HOPS: see HOPS below)
         into $DYNG_SCRATCH/bench/mosp/<graph>/. The CSR (K = 3 weights in [1, 100], seed 12345;
         mospPrep mtx2csr) is taken from $DYNG_SCRATCH/datasets/mosp/<graph>/csr by a symlink and
         is not copied.

run      rebuilds (idempotently) and verifies the unpatched copy, checks that --exe comes from a
         parity-preset build tree, then alternates the original (A: bin/mosp) and the port (B:
         dyng-compat-mosp --timing) A/B/A/B for --runs rounds per batch under the exclusive lock
         $DYNG_SCRATCH/perf.lock, with OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores
         for both, and compares medians region by region. The regions are LOADED from
         parity/timed_regions/sssp.toml ([[reference.mosp_openmp.region]]): the original's side
         from its report lines, the port's side from the profiler stages of its --timing CSV.
         Before the timed rounds of each batch, both write their outputs once and the files must
         be byte-identical; in every timed round the invalidated counters of every objective must
         be equal (both are correctness guards: a failure exits non-zero). Load average and the
         run-to-run spread are recorded; a spread above 10 % is flagged (PLAN Section 8.6). A
         gated region whose original median is below 10 ms needs >= 20 runs; with fewer, its
         verdict is marked provisional. Gates (from the map): "compute" <= 1.05x (<= 1.10x when
         the original's median is below 10 ms), "end_to_end" <= 1.10x. --enforce-gates exits
         non-zero on an exceeded gate.

--backend cuda compares dynG's cuda backend (`dyng-compat-mosp --backend cuda`, parity-cuda
         preset) with the UNPATCHED MOSP-CUDA@e220ee2 (bin/mosp) on the same prepared inputs,
         with the regions of [[reference.mosp_cuda.region]]. Both run on GPU --gpu (default 0, the
         performance GPU; CUDA_VISIBLE_DEVICES), MOSP-CUDA with CUDA_MODULE_LOADING=EAGER (its own
         default), dynG after resources::warm_up(). The per-objective region is the host time of
         the original's timer (sospUpdateGpu, up to its final synchronization) against the port's
         stage sssp.enact_fused, which has the same scope; the port's device time (CUDA events)
         of that stage is recorded next to it (the original has no device timer). The regions
         under 10 ms need >= 20 runs (PLAN 8.6).

The perf lock. The machine's convention is `flock $DYNG_SCRATCH/perf.lock <command>`, and this
script also takes the lock itself. Both work: the script sees in /proc/locks that an ancestor
process (flock(1)) holds the lock and runs under it; DYNG_PERF_LOCK_HELD=1 or --no-lock say so
explicitly. Otherwise it waits at most --lock-timeout seconds (default 3 hours) and then fails
with a message, instead of waiting forever.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import fcntl
import filecmp
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
REGION_MAP = REPO / "parity" / "timed_regions" / "sssp.toml"
# The original each backend of the port is compared with, and its section of sssp.toml.
REFERENCES = {
    "openmp": {
        "name": "MOSP-OpenMP",
        "commit": "c35215135341d5b5d1553458afe4b2226edc38fb",
        "map": "mosp_openmp",
    },
    "cuda": {
        "name": "MOSP-CUDA",
        "commit": "e220ee20d1b0948ece3df135a02d1b898264c22f",
        "map": "mosp_cuda",
    },
}
BATCHES = {
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
    "local10k": "changes_local_10000_50_safe",
}
# The report lines of the original's bin/mosp that sssp.toml may name in original_report.
OBJ = re.compile(r"^obj(\d+)\s+SOSP update ([0-9.]+) ms \(invalidated (\d+),", re.M)
REPORT = {
    "apply batch": re.compile(r"apply batch ([0-9.]+) ms"),
    "prepare": re.compile(r"prepare ([0-9.]+) ms"),
    "upload": re.compile(r"upload ([0-9.]+) ms"),
    "download": re.compile(r"download ([0-9.]+) ms"),
    "context": re.compile(r"context ([0-9.]+) ms"),
    "end_to_end_ms": re.compile(r"end_to_end_ms=([0-9.]+)"),
    "comb combined graph + SOSP": re.compile(r"^comb\s+combined graph \+ SOSP ([0-9.]+) ms", re.M),
}
PER_OBJECTIVE_REPORT = "obj<k> SOSP update"
PORT_INVALIDATED = re.compile(r"^counter,sssp\.invalidated\.obj(\d+),(\d+)$", re.M)
THREADS = re.compile(r"threads[= ](\d+)")
PARITY_PRESET = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_CXX_FLAGS_RELEASE": "-O3",
    "DYNG_BUILD_PARITY_TESTS": "ON",
    "DYNG_ENABLE_OPENMP": "ON",
}
# The parity-cuda preset: the parity preset plus MOSP-CUDA's nvcc flags for sm_86.
PARITY_CUDA_PRESET = {
    **PARITY_PRESET,
    "DYNG_ENABLE_CUDA": "ON",
    "DYNG_CUDA_ARCHITECTURES": "86",
    "CMAKE_CUDA_FLAGS_RELEASE": "-O3 -lineinfo -fmad=true",
}
PORT_DEVICE = "device"  # the --timing CSV rows with CUDA-event times (dyng-compat-mosp)
SHORT_REGION_MS = 10.0
SHORT_REGION_RUNS = 20
END_TO_END_GATE = 1.10
# Local-batch radius per graph, as the original's bench/prepare.sh names it (MOSP-OpenMP@c352151).
HOPS = {"roadNet-PA": 110, "roadNet-CA": 160, "rgg": 110, "road_usa_g": 200}


# --- Helpers ------------------------------------------------------------------------------------


def portable_path(path: Path | str) -> str:
    """Relative to the repository or $DYNG_SCRATCH; never a personal absolute path."""
    p = Path(path).resolve()
    for base, label in [(REPO, None), (SCRATCH.resolve(), "$DYNG_SCRATCH")]:
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        return rel if label is None else f"{label}/{rel}"
    return f"<outside the repository>/{p.name}" if p.is_relative_to(Path.home()) else str(p)


def build_reference(name: str, *extra: str) -> str:
    cmd = [REPO / "parity" / "build_reference.sh", "--variant", "unpatched", *extra, name]
    return subprocess.check_output([str(c) for c in cmd], text=True)


def reference_copy(name: str) -> Path:
    return Path(build_reference(name, "--print-dir").strip().splitlines()[-1])


def check_call(cmd: list, **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def cpu_model() -> str:
    with contextlib.suppress(OSError):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def cmake_cache(exe: Path) -> tuple[Path | None, dict]:
    """The CMakeCache.txt entries of the build tree that contains exe."""
    for d in exe.resolve().parents:
        cache = d / "CMakeCache.txt"
        if cache.is_file():
            entries = {}
            for line in cache.read_text().splitlines():
                if line and not line.startswith(("#", "//")) and "=" in line and ":" in line:
                    key, value = line.split("=", 1)
                    entries[key.split(":", 1)[0]] = value
            return d, entries
    return None, {}


def port_build(exe: Path, backend: str = "openmp") -> dict:
    tree, cache = cmake_cache(exe)
    keys = [
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_CXX_COMPILER",
        "DYNG_BUILD_PARITY_TESTS",
        "DYNG_ENABLE_OPENMP",
        "DYNG_ENABLE_CUDA",
        "DYNG_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_FLAGS_RELEASE",
        "CMAKE_CUDA_COMPILER",
    ]
    info = {k: cache.get(k) for k in keys}
    info["build_dir"] = portable_path(tree) if tree else None
    if cache.get("CMAKE_CXX_COMPILER"):
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            version = subprocess.check_output([cache["CMAKE_CXX_COMPILER"], "--version"], text=True)
            info["compiler"] = version.splitlines()[0]
    if cache.get("CMAKE_CUDA_COMPILER"):
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            nvcc = cache["CMAKE_CUDA_COMPILER"]
            version = subprocess.check_output([nvcc, "--version"], text=True)
            info["cuda_compiler"] = version.strip().splitlines()[-1]
    preset = PARITY_CUDA_PRESET if backend == "cuda" else PARITY_PRESET
    info["parity_preset"] = all(cache.get(k) == v for k, v in preset.items())
    return info


# --- The perf lock -------------------------------------------------------------------------------


def lock_holders(path: Path) -> set[int]:
    """PIDs holding a flock on `path` (from /proc/locks; empty if unknown)."""
    try:
        st = path.stat()
        text = Path("/proc/locks").read_text()
    except OSError:
        return set()
    holders = set()
    for line in text.splitlines():
        parts = line.split()
        # "1: FLOCK  ADVISORY  WRITE 12345 08:02:1234567 0 EOF" (blocked waiters have "->").
        if len(parts) >= 6 and parts[1] == "FLOCK" and "->" not in parts[1]:
            try:
                major, minor, inode = parts[5].split(":")
                pid = int(parts[4])
            except ValueError:
                continue
            if int(inode) == st.st_ino and os.makedev(int(major, 16), int(minor, 16)) == st.st_dev:
                holders.add(pid)
    return holders


def ancestors() -> set[int]:
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


@contextlib.contextmanager
def perf_lock(path: Path, timeout: float = 3 * 3600.0, skip: bool = False):
    """Hold the exclusive perf lock, or run under an ancestor's (flock(1)) hold of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip or os.environ.get("DYNG_PERF_LOCK_HELD") == "1":
        print(f"{path}: held by the caller (DYNG_PERF_LOCK_HELD=1 / --no-lock)", flush=True)
        yield
        return
    with open(path, "a") as f:
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                held_by_ancestor = lock_holders(path) & ancestors()
                if held_by_ancestor:
                    print(
                        f"{path}: held by an ancestor process {sorted(held_by_ancestor)} "
                        "(flock(1)); running under that lock",
                        flush=True,
                    )
                    yield
                    return
                if time.monotonic() >= deadline:
                    raise SystemExit(
                        f"{path}: still locked by another measurement after {timeout:.0f} s; "
                        "try again later (if flock(1) around this script holds it, set "
                        "DYNG_PERF_LOCK_HELD=1 or pass --no-lock)"
                    ) from None
                if not announced:
                    print(
                        f"waiting for {path} (held by {sorted(lock_holders(path))}) ...", flush=True
                    )
                    announced = True
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- The region map ------------------------------------------------------------------------------


def load_regions(backend: str = "openmp") -> list[dict]:
    """The regions of sssp.toml for a backend, checked against what this script can measure."""
    doc = tomllib.loads(REGION_MAP.read_text())
    regions = doc["reference"][REFERENCES[backend]["map"]]["region"]
    known = set(REPORT) | {PER_OBJECTIVE_REPORT}
    names = [r["name"] for r in regions]
    for required in ["sosp_update", "apply", "end_to_end"]:
        if required not in names:
            raise SystemExit(f"{REGION_MAP}: region '{required}' is missing")
    for r in regions:
        for key in r.get("original_report", []) + r.get("original_report_subtract", []):
            if key not in known:
                raise SystemExit(
                    f"{REGION_MAP}: region {r['name']}: perf_ab.py cannot parse the "
                    f"original's report line '{key}' (known: {sorted(known)})"
                )
        if r["gate"] not in ("compute", "end_to_end", "none"):
            raise SystemExit(f"{REGION_MAP}: region {r['name']}: unknown gate '{r['gate']}'")
        if not r.get("port"):
            raise SystemExit(f"{REGION_MAP}: region {r['name']}: no port stages")
    return regions


# --- One run of each side ----------------------------------------------------------------------


def run_one(cmd: list, env: dict) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed:\n{proc.stdout[-2000:]}")
    return proc.stdout


def report_keys(regions: list[dict]) -> list[str]:
    """The report lines of the original that the regions read."""
    keys = []
    for r in regions:
        for key in r.get("original_report", []) + r.get("original_report_subtract", []):
            if key in REPORT and key not in keys:
                keys.append(key)
    return keys


def parse_original(log: str, k: int, keys: list[str] | None = None) -> dict:
    objs = {int(o): (float(ms), int(inv)) for o, ms, inv in OBJ.findall(log)}
    if sorted(objs) != list(range(k)):
        raise SystemExit(f"cannot parse the per-objective lines of the original:\n{log}")
    values = {}
    for key in keys if keys is not None else list(REPORT):
        pattern = REPORT[key]
        m = pattern.search(log)
        if m is None:
            raise SystemExit(f"cannot find '{key}' in the original's report:\n{log}")
        values[key] = float(m.group(1))
    threads = [int(t) for t in THREADS.findall(log)]
    return {
        "objectives": [objs[o][0] for o in range(k)],
        "invalidated": [objs[o][1] for o in range(k)],
        "report": values,
        "threads": threads[-1] if threads else None,
    }


def parse_port(log: str, timing: Path, k: int) -> dict:
    stages: dict[str, list[float]] = {}
    device: dict[str, list[float]] = {}
    text = timing.read_text()
    for row in csv.reader(text.splitlines()[1:]):
        if len(row) == 3 and row[0] == "stage":
            stages.setdefault(row[1], []).append(float(row[2]))
        elif len(row) == 3 and row[0] == PORT_DEVICE:
            device.setdefault(row[1], []).append(float(row[2]))
    invalidated = {int(o): int(v) for o, v in PORT_INVALIDATED.findall(text)}
    if sorted(invalidated) != list(range(k)):
        raise SystemExit(f"cannot find the port's invalidated counters in {timing}")
    threads = [int(t) for t in THREADS.findall(log)]
    return {
        "stages": stages,
        "device": device,
        "invalidated": [invalidated[o] for o in range(k)],
        "threads": threads[-1] if threads else None,
    }


def stage_sum(port: dict, names: list[str], sample: int | None = None) -> float:
    total = 0.0
    for name in names:
        samples = port["stages"].get(name)
        if not samples:
            raise SystemExit(f"the port recorded no stage '{name}' (sssp.toml)")
        total += samples[sample] if sample is not None else sum(samples)
    return total


def original_value(orig: dict, region: dict, k: int | None) -> float:
    total = 0.0
    for key in region.get("original_report", []):
        if key == PER_OBJECTIVE_REPORT:
            total += orig["objectives"][k] if k is not None else sum(orig["objectives"])
        else:
            total += orig["report"][key]
    for key in region.get("original_report_subtract", []):
        total -= orig["report"][key]
    return total


def port_device_value(port: dict, region: dict, k: int | None) -> float | None:
    """The port's device time (CUDA events) of a region's port_device stages, if recorded."""
    names = region.get("port_device")
    if not names:
        return None
    total = 0.0
    for name in names:
        samples = port.get("device", {}).get(name)
        if not samples:
            return None
        total += samples[k] if k is not None else sum(samples)
    return total


def port_value(port: dict, region: dict, k: int | None) -> float:
    total = stage_sum(port, region["port"], k)
    if region.get("port_all_results"):
        total += stage_sum(port, region["port_all_results"])
    if region.get("port_result0"):
        total += stage_sum(port, region["port_result0"], 0)
    return total


# --- Statistics and report -------------------------------------------------------------------


def spread(xs: list[float]) -> float:
    """Relative spread: (max - min) / median."""
    m = statistics.median(xs)
    return (max(xs) - min(xs)) / m if m > 0 else 0.0


def gate_limit(kind: str, original_ms: float) -> float:
    if kind == "end_to_end":
        return END_TO_END_GATE
    return 1.05 if original_ms >= SHORT_REGION_MS else 1.10


def summarize(regions: list[dict], samples: dict, k: int, runs: int, loads: list) -> dict:
    out = {
        "regions": [],
        "invalidated": {},
        "threads": {},
        "load_average": {"min": min(min(p) for p in loads), "max": max(max(p) for p in loads)},
    }
    out["invalidated"]["original"] = samples["original"][0]["invalidated"]
    out["invalidated"]["port"] = samples["port"][0]["invalidated"]
    out["invalidated"]["equal_in_every_sample"] = all(
        a["invalidated"] == b["invalidated"]
        for a, b in zip(samples["original"], samples["port"], strict=True)
    )
    for side in ["original", "port"]:
        out["threads"][side] = samples[side][0]["threads"]

    def entry(
        name: str, a: list[float], b: list[float], gate: str, reading: str, dev: list | None = None
    ) -> None:
        ma, mb = statistics.median(a), statistics.median(b)
        e = {
            "region": name,
            "reading": reading,
            "original_ms": ma,
            "port_ms": mb,
            "ratio": mb / ma if ma > 0 else float("nan"),
            "original_spread": spread(a),
            "port_spread": spread(b),
            "original_samples": a,
            "port_samples": b,
            "gate_kind": gate,
        }
        if dev and all(x is not None for x in dev):
            e["port_device_ms"] = statistics.median(dev)
            e["port_device_samples"] = dev
        if gate in ("compute", "end_to_end"):
            e["gate"] = gate_limit(gate, ma)
            e["within_gate"] = e["ratio"] <= e["gate"]
            short = gate == "compute" and ma < SHORT_REGION_MS
            e["provisional"] = short and runs < SHORT_REGION_RUNS
        e["noisy"] = e["original_spread"] > 0.10 or e["port_spread"] > 0.10
        out["regions"].append(e)

    for region in regions:
        gate = region["gate"]
        if region.get("per_objective"):
            for o in range(k):
                a = [original_value(s, region, o) for s in samples["original"]]
                b = [port_value(s, region, o) for s in samples["port"]]
                dev = [port_device_value(s, region, o) for s in samples["port"]]
                entry(f"{region['name']} obj{o}", a, b, gate, "as measured", dev)
        else:
            a = [original_value(s, region, None) for s in samples["original"]]
            b = [port_value(s, region, None) for s in samples["port"]]
            entry(region["name"], a, b, gate, "as measured")
    return out


def report(results: dict) -> list[str]:
    lines = [
        "| batch | region | reading | original (ms) | dynG (ms) | ratio | gate | spread A / B |"
        " dynG device (ms) |",
        "|---|---|---|---:|---:|---:|---|---|---:|",
    ]
    for batch, res in results.items():
        for e in res["regions"]:
            if "gate" in e:
                verdict = "ok" if e["within_gate"] else "EXCEEDED"
                if e.get("provisional"):
                    verdict += " (provisional: < 20 runs)"
                g = f"<= {e['gate']:.2f} {verdict}"
            else:
                g = "-"
            flag = " (noisy)" if e["noisy"] else ""
            lines.append(
                f"| {batch} | {e['region']} | {e['reading']} | {e['original_ms']:.2f} | "
                f"{e['port_ms']:.2f} | {e['ratio']:.3f} | {g} | "
                f"{e['original_spread'] * 100:.0f} % / {e['port_spread'] * 100:.0f} %"
                f"{flag} | "
                + (f"{e['port_device_ms']:.2f}" if "port_device_ms" in e else "-")
                + " |"
            )
    print("\n".join(lines))
    for batch, res in results.items():
        inv = res["invalidated"]
        print(
            f"{batch}: invalidated original {inv['original']} port {inv['port']} "
            f"({'equal in every sample' if inv['equal_in_every_sample'] else 'DIFFERENT'}); "
            f"threads {res['threads']}; load average {res['load_average']['min']:.1f}-"
            f"{res['load_average']['max']:.1f}"
        )
    return lines


# --- Commands ----------------------------------------------------------------------------------


def prepare(args: argparse.Namespace) -> int:
    # The inputs are always made by MOSP-OpenMP's mospPrep (its bench/prepare.sh); MOSP-CUDA reads
    # the same files, and its generator is the same code (see generators::legacy).
    name = REFERENCES["openmp"]["name"]
    ref = reference_copy(name)
    build_reference(name)
    prep = ref / "bin" / "mospPrep"
    src = SCRATCH / "datasets" / "mosp" / args.graph / "csr"
    dst = SCRATCH / "bench" / "mosp" / args.graph
    dst.mkdir(parents=True, exist_ok=True)
    csr_dir = dst / "csr"
    if not csr_dir.exists():
        csr_dir.symlink_to(src, target_is_directory=True)
    prefix = csr_dir / "graphCsr"
    env = dict(os.environ, OMP_NUM_THREADS="28", OMP_PROC_BIND="close", OMP_PLACES="cores")
    common = ["--ins", "50", "--seed", "777"]
    steps = [
        ("init", ["init", prefix, dst / "init"]),
        (
            BATCHES["unsafe50k"],
            ["changes", prefix, dst / BATCHES["unsafe50k"], "--changes", "50000", *common],
        ),
        (
            BATCHES["safe50k"],
            ["changes", prefix, dst / BATCHES["safe50k"], "--changes", "50000", *common, "--safe"],
        ),
        (
            BATCHES["local10k"],
            [
                "changes",
                prefix,
                dst / BATCHES["local10k"],
                "--changes",
                "10000",
                *common,
                "--local",
                str(args.hops),
                "--safe",
            ],
        ),
    ]
    for name, cmd in steps:
        if (dst / name).is_dir() and not args.force:
            print(f"{dst / name}: exists (use --force to regenerate)")
            continue
        print(f"mospPrep {' '.join(map(str, cmd))}", flush=True)
        check_call([prep, *cmd], env=env)
    # Record what was prepared (sizes and SHA-256 of the inputs) for the results file.
    lines = []
    for p in sorted(set(dst.rglob("*.txt")) | set(csr_dir.glob("*.txt"))):
        digest = subprocess.check_output(["sha256sum", p], text=True).split()[0]
        lines.append(f"{digest}  {p.relative_to(dst)}")
    (dst / "INPUTS.sha256").write_text("\n".join(lines) + "\n")
    print(f"inputs in {dst} (INPUTS.sha256 written)")
    return 0


def run(args: argparse.Namespace) -> int:
    regions = load_regions(args.backend)
    keys = report_keys(regions)
    reference = REFERENCES[args.backend]
    exe = args.exe.resolve()
    build = port_build(exe, args.backend)
    if not build["parity_preset"] and not args.allow_non_parity_build:
        preset = "parity-cuda" if args.backend == "cuda" else "parity"
        raise SystemExit(
            f"{exe} is not from a {preset}-preset build tree ({build}); the gates are "
            f"defined on the {preset} preset (pass --allow-non-parity-build for an "
            "experiment, whose record says so)"
        )
    # Rebuild (idempotent) and verify the unpatched copy before timing it.
    build_reference(reference["name"])
    ref = reference_copy(reference["name"])
    mosp = ref / "bin" / "mosp"
    marker = ref / ".dyng-reference"
    data = SCRATCH / "bench" / "mosp" / args.graph
    if not (data / "init").is_dir():
        raise SystemExit(f"{data}: run `parity/perf_ab.py prepare --graph {args.graph}` first")
    k = len(list((data / "init").glob("obj*")))
    env = dict(
        os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND="close", OMP_PLACES="cores"
    )
    for var in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_DYNAMIC"]:
        env.pop(var, None)
    port_args: list = []
    if args.backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["CUDA_MODULE_LOADING"] = "EAGER"  # MOSP-CUDA's own default (setenv, not overwriting)
        port_args = ["--backend", "cuda"]
    for item in args.env:  # extra settings for BOTH sides, e.g. OMP_WAIT_POLICY=active
        key, _, value = item.partition("=")
        env[key] = value
    batches = [b for b in args.batches.split(",") if b]
    unknown = [b for b in batches if b not in BATCHES]
    if unknown or not batches:
        raise SystemExit(f"--batches: unknown or empty ({unknown}); known: {sorted(BATCHES)}")
    results = {}
    failures = []
    work = Path(tempfile.mkdtemp(prefix="dyng-perf-", dir=SCRATCH / "runs"))
    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for batch in batches:
                changes = data / BATCHES[batch]
                common = [
                    "--graph",
                    data / "csr" / "graphCsr",
                    "--changes",
                    changes,
                    "--init",
                    data / "init",
                ]
                # Correctness guard: both write their outputs once; the files must be identical.
                run_one([mosp, *common, "--out", work / "A"], env)
                run_one([exe, *common, *port_args, "--out", work / "B"], env)
                for o in range(k):
                    for f in ["distancesUpdated.txt", "SSSPTreeUpdated.txt"]:
                        if not filecmp.cmp(
                            work / "A" / f"obj{o}" / f, work / "B" / f"obj{o}" / f, shallow=False
                        ):
                            raise SystemExit(
                                f"{batch}: obj{o}/{f} differs between the original and the port"
                            )
                print(f"{batch}: outputs byte-identical ({k} objectives)", flush=True)
                samples: dict[str, list] = {"original": [], "port": []}
                loads = []
                timing = work / "timing.csv"
                for r in range(args.runs):
                    before = os.getloadavg()[0]
                    orig = parse_original(run_one([mosp, *common, "--no-output"], env), k, keys)
                    log = run_one(
                        [exe, *common, *port_args, "--no-output", "--timing", timing], env
                    )
                    port = parse_port(log, timing, k)
                    loads.append((before, os.getloadavg()[0]))
                    samples["original"].append(orig)
                    samples["port"].append(port)
                    if orig["invalidated"] != port["invalidated"]:
                        failures.append(
                            f"{batch} round {r + 1}: invalidated original "
                            f"{orig['invalidated']} != port {port['invalidated']}"
                        )
                    print(
                        f"{batch} round {r + 1}/{args.runs}: SOSP original "
                        f"{sum(orig['objectives']):.1f} ms, port "
                        f"{sum(stage_sum(port, regions[0]['port'], o) for o in range(k)):.1f}"
                        " ms",
                        flush=True,
                    )
                results[batch] = summarize(regions, samples, k, args.runs, loads)
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results)
    exceeded = [
        f"{b}: {e['region']} ({e['reading']}) {e['ratio']:.3f} > {e['gate']:.2f}"
        for b, res in results.items()
        for e in res["regions"]
        if "gate" in e and not e["within_gate"]
    ]
    if args.json:
        write_json(args, results, build, ref, marker, regions, reference)
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    if failures:
        return 1
    if exceeded:
        print("gate exceeded: " + "; ".join(exceeded))
        if args.enforce_gates:
            return 1
    return 0


def write_json(args, results, build, ref, marker, regions, reference) -> None:
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    # The records this script writes (parity/results/) do not make the measured code dirty.
    dirty = (
        subprocess.run(
            ["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", ".", ":(exclude)parity/results"]
        ).returncode
        != 0
    )
    doc = {
        "schema": 3,
        "algorithm": "sssp",
        "backend": args.backend,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "graph": args.graph,
        "reference": {
            "name": reference["name"],
            "commit": reference["commit"],
            "variant": "unpatched",
            "binary": portable_path(ref / "bin" / "mosp"),
            "build": marker.read_text() if marker.is_file() else None,
        },
        "port": {
            "commit": head + ("+dirty" if dirty else ""),
            "binary": portable_path(args.exe),
            "build": build,
        },
        "region_map": {"file": portable_path(REGION_MAP), "regions": [r["name"] for r in regions]},
        "protocol": {
            "runs": args.runs,
            "order": "A/B/A/B (original first)",
            "threads": args.threads,
            "env": " ".join(
                ["OMP_PROC_BIND=close", "OMP_PLACES=cores"]
                + (
                    [f"CUDA_VISIBLE_DEVICES={args.gpu}", "CUDA_MODULE_LOADING=EAGER"]
                    if args.backend == "cuda"
                    else []
                )
                + args.env
            ),
            "lock": "perf.lock",
            "statistic": "median",
            "outputs": "--no-output on both",
            "short_regions": f"< {SHORT_REGION_MS} ms need >= {SHORT_REGION_RUNS} runs",
        },
        "host": {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "kernel": platform.release()},
        "inputs_sha256": (SCRATCH / "bench" / "mosp" / args.graph / "INPUTS.sha256")
        .read_text()
        .splitlines(),
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--graph", default="roadNet-CA")
    p.add_argument(
        "--hops", type=int, help="local batch radius (default: HOPS[graph]; roadNet-CA: 160)"
    )
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--exe", type=Path, required=True, help="dyng-compat-mosp (parity preset)")
    r.add_argument(
        "--backend",
        choices=sorted(REFERENCES),
        default="openmp",
        help="openmp: against MOSP-OpenMP c352151; cuda: against MOSP-CUDA e220ee2",
    )
    r.add_argument("--gpu", type=int, default=0, help="--backend cuda: the GPU of both sides")
    r.add_argument("--graph", default="roadNet-CA")
    r.add_argument("--batches", default="safe50k,unsafe50k,local10k")
    r.add_argument("--runs", type=int, default=21)
    r.add_argument("--threads", type=int, default=28)
    r.add_argument("--json", type=Path)
    r.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="extra environment for both sides (repeatable)",
    )
    r.add_argument(
        "--no-lock",
        action="store_true",
        help="the caller holds $DYNG_SCRATCH/perf.lock (same as DYNG_PERF_LOCK_HELD=1)",
    )
    r.add_argument("--lock-timeout", type=float, default=3 * 3600.0, metavar="SECONDS")
    r.add_argument(
        "--allow-non-parity-build",
        action="store_true",
        help="time an --exe that is not from the parity preset (an experiment)",
    )
    r.add_argument(
        "--enforce-gates",
        action="store_true",
        help="exit 1 if a gated region exceeds its limit (the gates bind from M1b)",
    )
    args = parser.parse_args()
    if args.command == "run" and args.runs < 5:
        parser.error("--runs must be >= 5 (PLAN Section 6.3 step 7)")
    if args.command == "prepare" and args.hops is None:
        if args.graph not in HOPS:
            parser.error(f"--hops is required for {args.graph} (known: {sorted(HOPS)})")
        args.hops = HOPS[args.graph]
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
