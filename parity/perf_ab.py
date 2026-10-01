#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Performance A/B of the sssp port against the original (PLAN Sections 6.3 step 7, 8.5, 8.6).

`parity/perf_ab.py cycle_count run ...` runs the cycle_count A/B against
CycleEnumeration-GPU@0a976ad instead (parity/cycle_count_perf.py).

    parity/perf_ab.py prepare [--graph roadNet-CA] [--hops 160]
    parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp [--runs 21]
                          [--graph roadNet-CA] [--batches safe50k,unsafe50k,local10k]
                          [--json parity/results/M1b-perf-openmp-roadNet-CA.json]
    parity/perf_ab.py run --backend cuda --exe build/parity-cuda/tools/compat/dyng-compat-mosp
                          [--gpu 0] [--runs 21] [--graph roadNet-CA] [--json ...]
    parity/perf_ab.py kernels --exe build/parity-cuda/tools/compat/dyng-compat-mosp [--runs 21]
                          [--graph roadNet-CA] [--json ...]
    parity/perf_ab.py edge-type --backend openmp|cuda --exe <parity build> [--runs 21] ...
    parity/perf_ab.py engines --exe build/parity-cuda/tools/compat/dyng-compat-mosp [--gpu 0]
                          [--runs 21] [--graph roadNet-CA] [--json ...]
    parity/perf_ab.py memory --exe build/parity-cuda/tools/compat/dyng-compat-mosp [--gpu 0]
                          [--graph roadNet-CA] [--batches ...] [--json ...]
    parity/perf_ab.py run --baseline-exe <earlier dynG build> --baseline-label <commit> --exe ...
    parity/perf_ab.py mosp --backend openmp|cuda --exe <parity build>/dyng-compat-mosp [--gpu 0]
                          [--runs 21] [--graph roadNet-CA] [-k K] [--pref p1,..,pK] [--json ...]
    parity/perf_ab.py prepare --graph roadNet-CA-K4 --widen roadNet-CA:4

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

--baseline-exe replaces side A (the original) by an earlier dynG build's dyng-compat-mosp (for
         example the code before a refactor, PLAN 6.3 step 8, or main before a milestone), run
         with the same arguments and read through the same profiler stages as side B. The
         correctness guards stay (byte-identical outputs, equal invalidated counters); the region
         ratios are the change of the refactor and are reported, not gated. The record's
         "reference" names the baseline (--baseline-label).

--backend cuda compares dynG's cuda backend (`dyng-compat-mosp --backend cuda`, parity-cuda
         preset) with the UNPATCHED MOSP-CUDA@e220ee2 (bin/mosp) on the same prepared inputs,
         with the regions of [[reference.mosp_cuda.region]]. Both run on GPU --gpu (default 0, the
         performance GPU; CUDA_VISIBLE_DEVICES), MOSP-CUDA with CUDA_MODULE_LOADING=EAGER (its own
         default), dynG after resources::warm_up(). The per-objective region is the host time of
         the original's timer (sospUpdateGpu, up to its final synchronization) against the port's
         stage sssp.enact_fused, which has the same scope; the port's device time (CUDA events)
         of that stage is recorded next to it (the original has no device timer). The regions
         under 10 ms need >= 20 runs (PLAN 8.6).
         --lock-clocks (cuda; default boost) locks GPU --gpu's SM and memory clocks for the
         whole A/B without root (ADR 0018): Nsight Compute profiles the idle helper
         parity/clock_lock/clock_holder.cu under `ncu --clock-control boost|base`, which holds
         the lock for every process on the GPU while the helper lives; neither timed program is
         profiled. The monitor then requires every busy GPU sample of both sides to be at the
         locked clocks (otherwise the round is repeated), and `ncu --clock-control reset` runs
         at the end in any case. --lock-clocks none keeps the default clocks (DVFS): the
         as-measured reading, in which a program's own GPU work before the timed region decides
         the P-state of its kernels.

memory   (cuda) the device-memory gate of PLAN 8.6 (<= 1.05x): each side (MOSP-CUDA's bin/mosp and
         dyng-compat-mosp --backend cuda) runs once per batch under `nsys profile
         --cuda-memory-usage=true`; the peak of its live device allocations (cudaMalloc and
         cudaMallocAsync; memory kind Device) is compared, with the largest stream-ordered pool
         size nsys reports for the port (as `perf_ab.py cycle_count memory`). The original runs the
         whole MOSP update (the K SOSP updates and the combined graph); the port runs the K sssp
         updates only, unless --mosp (M7: dyng-compat-mosp --mosp, the same scope as the original).

kernels  (cuda) runs both sides A/B/A/B under Nsight Compute with the GPU clocks locked to base
         (`ncu --clock-control base --cache-control none`, no root needed) and compares the
         per-objective fused kernels (gpu__time_duration of the first K launches of
         sospPersistentKernel / sssp_persistent_kernel; MOSP-CUDA's K+1-th launch is its combined
         graph), with their DRAM bytes, registers, grid and occupancy limits. It takes the GPU's
         clock state (the DVFS P-state, which a program's own GPU work before the timed region
         decides) out of the comparison; it is the controlled-clock reading next to `run`'s
         as-measured one (parity/results/M1b.md).

edge-type  the edge_t benchmark of ADR 0009: the port with 32-bit (A) against 64-bit (B) edge
         offsets (`dyng-compat-mosp --edge-type`), A/B/A/B, same inputs, regions and guards
         (outputs byte-identical, invalidated counters equal); the ratio is int64 / int32.

engines  (cuda) dynG's fused engine (A, `dyng-compat-mosp --cuda-engine fused`) against its
         operators engine (B, `--cuda-engine operators`; M7, decision O24), A/B/A/B at locked
         clocks (--lock-clocks, default boost) with the guards of `run`; the per-objective SOSP
         region is sssp.enact_fused on A and identify_affected + seed + loop + finalize on B
         (the pseudo stage sssp.engine), host and device times. Reported, never gated.

mosp     (M7, PLAN 6.4.4) dynG's MOSP update (`dyng-compat-mosp --mosp`, the parity or
         parity-cuda preset) against the UNPATCHED original's bin/mosp (MOSP-OpenMP@c352151 with
         --backend openmp, MOSP-CUDA@e220ee2 with --backend cuda), A/B/A/B with the protocol of
         `run` (perf lock, contamination monitor, GPU clocks locked to boost on cuda). Both sides
         WRITE every output file (the K trees and combinedGraph/), as the original's bench/run.sh
         does, into a directory deleted after the run; before the timed rounds of a batch every
         output file of both is compared byte for byte (and the invalidated counters in every
         round). Regions from parity/timed_regions/mosp.toml: "(a) compute" (the K SOSP updates and
         the combined step; <= 1.05x) and "(b) end to end" (<= 1.10x) gated, the per-objective
         updates, the combined step and dynG's path costs reported. -k and --pref are passed to
         both (`mosp -k K --pref ...`); --graph roadNet-CA-K4 with -k 2..4 is the K sweep.

prepare --widen BASE:K  the K sweep's input: the CSR of BASE widened to K objectives (mospPrep
         widen, weights [1, 100], seed 12345), its initial trees and its 50K safe batch (seed 777).

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
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
REGION_MAP = REPO / "parity" / "timed_regions" / "sssp.toml"
MOSP_REGION_MAP = REPO / "parity" / "timed_regions" / "mosp.toml"  # `perf_ab.py mosp` (M7)
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
    # MospTimings::gpuCompute() (MOSP-CUDA) / compute() (MOSP-OpenMP): the "(a) compute" region.
    "compute_ms": re.compile(r"^RESULT (?:gpu_)?compute_ms=([0-9.]+)", re.M),
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


def driver_versions() -> dict:
    """The NVIDIA driver and the CUDA driver API version (nvidia-smi) when the record is taken;
    None for each where no driver is visible (PLAN 8.5 item 2: the record names the driver)."""
    out = {"nvidia": None, "cuda_driver_api": None}
    if not shutil.which("nvidia-smi"):
        return out
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
        out["nvidia"] = rows[0] if rows else None
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        out["cuda_driver_api"] = m.group(1) if m else None
    return out


def hash_inputs(data: Path) -> list[str]:
    """SHA-256 of every input file that `prepare` listed in <data>/INPUTS.sha256, computed now
    (when a measurement starts), in the same "<digest>  <path>" lines; exit if a file differs
    from what `prepare` recorded or is missing."""
    listed = (data / "INPUTS.sha256").read_text().splitlines()
    lines = []
    for line in listed:
        if not line.strip():
            continue
        want, rel = line.split(None, 1)
        rel = rel.strip()
        path = data / rel
        if not path.is_file():
            raise SystemExit(f"{path}: missing (listed in INPUTS.sha256)")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise SystemExit(f"{path}: SHA-256 {h.hexdigest()}, INPUTS.sha256 says {want}")
        lines.append(f"{h.hexdigest()}  {rel}")
    return lines


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


LOCK_STATUS = "perf.lock: not taken"


def blocking_flock(f, seconds: float) -> bool:
    """Take an exclusive flock on f, waiting in the kernel for at most `seconds` (SIGALRM; the
    main thread only). False on the timeout."""
    if seconds <= 0:
        return False

    class _Timeout(Exception):
        pass

    def on_alarm(signum, frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, on_alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            return True
        except _Timeout:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def perf_lock(path: Path, timeout: float = 3 * 3600.0, skip: bool = False):
    """Hold the exclusive perf lock, or run under an ancestor's (flock(1)) hold of it."""
    global LOCK_STATUS
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip or os.environ.get("DYNG_PERF_LOCK_HELD") == "1":
        print(f"{path}: held by the caller (DYNG_PERF_LOCK_HELD=1 / --no-lock)", flush=True)
        LOCK_STATUS = "perf.lock: declared held by the caller (--no-lock / DYNG_PERF_LOCK_HELD=1)"
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
                    LOCK_STATUS = "perf.lock: held exclusively by an ancestor (flock(1))"
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
                # Wait in the kernel, not by polling: a LOCK_NB poll every second starved behind
                # back-to-back shared holders (another agent's `flock -s` chain of builds and
                # suites), which take the lock again in the gap between two polls.
                if blocking_flock(f, deadline - time.monotonic()):
                    break
        LOCK_STATUS = "perf.lock: held exclusively by this process"
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- The region map ------------------------------------------------------------------------------


def load_regions(
    backend: str = "openmp",
    path: Path = REGION_MAP,
    required: tuple[str, ...] = ("sosp_update", "apply", "end_to_end"),
) -> list[dict]:
    """The regions of a region map (sssp.toml unless `path`) for a backend, checked against what
    this script can measure."""
    doc = tomllib.loads(path.read_text())
    regions = doc["reference"][REFERENCES[backend]["map"]]["region"]
    known = set(REPORT) | {PER_OBJECTIVE_REPORT}
    names = [r["name"] for r in regions]
    for name in required:
        if name not in names:
            raise SystemExit(f"{path}: region '{name}' is missing")
    for r in regions:
        for key in r.get("original_report", []) + r.get("original_report_subtract", []):
            if key not in known:
                raise SystemExit(
                    f"{path}: region {r['name']}: perf_ab.py cannot parse the "
                    f"original's report line '{key}' (known: {sorted(known)})"
                )
        if r["gate"] not in ("compute", "end_to_end", "none"):
            raise SystemExit(f"{path}: region {r['name']}: unknown gate '{r['gate']}'")
        if not r.get("port"):
            raise SystemExit(f"{path}: region {r['name']}: no port stages")
    return regions


# --- One run of each side ----------------------------------------------------------------------


def run_one(cmd: list, env: dict) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed:\n{proc.stdout[-2000:]}")
    return proc.stdout


# --- The contamination monitor (PLAN Sections 8.5 item 2 and 8.6) -------------------------------

CLK_TCK = os.sysconf("SC_CLK_TCK")
# A round is rejected (and repeated) when, during either side's run, the CPUs were busy outside this
# harness and its programs for more than this many cores on average, or a compute process that is
# not the timed program was on the timed GPU.
MAX_FOREIGN_CPU = 2.0
NVSMI_TIME = "%Y/%m/%d %H:%M:%S.%f"


def cpu_busy_seconds() -> float:
    """Busy CPU time of the whole machine (all CPUs; /proc/stat, without idle and iowait)."""
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:9]]
    return (sum(fields) - fields[3] - fields[4]) / CLK_TCK


def procs_running() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("procs_running "):
            return int(line.split()[1])
    return -1


def running_tasks(pid: int) -> int:
    """Threads of `pid` in state R (they are counted in procs_running)."""
    count = 0
    with contextlib.suppress(OSError):
        for task in os.scandir(f"/proc/{pid}/task"):
            with contextlib.suppress(OSError, IndexError):
                stat = Path(task.path, "stat").read_text()
                count += stat.rsplit(")", 1)[1].split()[0] == "R"
    return count


def descends_from(pid: int, root: int) -> bool:
    while pid > 1:
        if pid == root:
            return True
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


class MachineMonitor:
    """What the machine did while the timed programs ran (PLAN 8.5 item 2: "the contamination
    monitor"), sampled for a whole batch and cut into one window per side and round:

    - the CPUs: busy CPU time of the machine minus that of this harness and its children (the
      timed program), as average cores busy with foreign work; and the run queue (procs_running
      of /proc/stat, every 0.1 s) with the timed program's own running threads taken out;
    - the timed GPU (cuda only): its P-state, SM and memory clocks and utilization every 50 ms
      (nvidia-smi -lms, NVML), and every second the compute processes on it that are neither this
      harness nor its descendants (the originals' bench/gpumon.sh rule).

    A window is contaminated when the foreign CPU load exceeds max_foreign_cpu cores, a foreign
    compute process was on the timed GPU, or (with `locked`, the (SM, memory) MHz of a
    ClockLock) a busy GPU sample ran at other clocks; `run` and `edge-type` then repeat the round.
    `allowed_pids` are the harness's own GPU processes that do not descend from it (the clock
    holder, a child of ncu).
    """

    def __init__(
        self,
        gpu: int | None,
        max_foreign_cpu: float = MAX_FOREIGN_CPU,
        allowed_pids: set[int] | None = None,
        locked: tuple[int, int] | None = None,
    ) -> None:
        self.gpu = gpu
        self.max_foreign_cpu = max_foreign_cpu
        self.allowed_pids = set(allowed_pids or ())
        self.locked = locked
        self.gpu_samples: list[tuple] = []  # (time, pstate, sm MHz, memory MHz, utilization %)
        self.queue_samples: list[tuple] = []  # (time, procs_running, the program's running threads)
        self.foreign_gpu: list[tuple] = []  # (time, pid, process name)
        self.child: int | None = None
        self.gpu_uuid = None
        self.nvsmi = shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi: subprocess.Popen | None = None

    def __enter__(self) -> MachineMonitor:
        if self.gpu is not None and self.nvsmi:
            with contextlib.suppress(OSError, subprocess.CalledProcessError):
                self.gpu_uuid = subprocess.check_output(
                    [self.nvsmi, "--query-gpu=uuid", "--format=csv,noheader", "-i", str(self.gpu)],
                    text=True,
                ).strip()
            query = "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,utilization.gpu"
            cmd = [self.nvsmi, "-i", str(self.gpu), query, "--format=csv,noheader,nounits"]
            stdbuf = shutil.which("stdbuf")
            self._smi = subprocess.Popen(
                ([stdbuf, "-oL"] if stdbuf else []) + [*cmd, "-lms", "50"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._threads.append(threading.Thread(target=self._read_gpu, daemon=True))
            self._threads.append(threading.Thread(target=self._poll_gpu_processes, daemon=True))
        self._threads.append(threading.Thread(target=self._poll_queue, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._smi.wait(timeout=5)
        for t in self._threads:
            t.join(timeout=5)

    def _read_gpu(self) -> None:
        assert self._smi is not None and self._smi.stdout is not None
        for line in self._smi.stdout:
            parts = [x.strip() for x in line.split(",")]
            with contextlib.suppress(ValueError, IndexError):
                t = datetime.datetime.strptime(parts[0], NVSMI_TIME).timestamp()
                self.gpu_samples.append((t, parts[1], int(parts[2]), int(parts[3]), int(parts[4])))

    def _poll_gpu_processes(self) -> None:
        me = os.getpid()
        query = "--query-compute-apps=pid,gpu_uuid,process_name"
        while not self._stop.wait(1.0):
            with contextlib.suppress(OSError, subprocess.CalledProcessError):
                out = subprocess.check_output(
                    [self.nvsmi, query, "--format=csv,noheader"], text=True
                )
                now = time.time()
                for line in out.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) < 3 or (self.gpu_uuid and parts[1] != self.gpu_uuid):
                        continue
                    with contextlib.suppress(ValueError):
                        pid = int(parts[0])
                        if pid not in self.allowed_pids and not descends_from(pid, me):
                            self.foreign_gpu.append((now, pid, parts[2]))

    def _poll_queue(self) -> None:
        while not self._stop.wait(0.1):
            with contextlib.suppress(OSError, ValueError):
                child = self.child
                own = running_tasks(child) if child is not None else 0
                self.queue_samples.append((time.time(), procs_running(), own))

    def run(self, cmd: list, env: dict) -> tuple[str, dict]:
        """One side's run with its window: (stdout, the window's record)."""
        cpu0, self0, kids0 = (
            cpu_busy_seconds(),
            resource.getrusage(resource.RUSAGE_SELF),
            (resource.getrusage(resource.RUSAGE_CHILDREN)),
        )
        t0 = time.time()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.child = proc.pid
        out, _ = proc.communicate()
        self.child = None
        t1 = time.time()
        cpu1, self1, kids1 = (
            cpu_busy_seconds(),
            resource.getrusage(resource.RUSAGE_SELF),
            (resource.getrusage(resource.RUSAGE_CHILDREN)),
        )
        if proc.returncode != 0:
            raise SystemExit(f"{' '.join(map(str, cmd))} failed:\n{out[-2000:]}")
        mine = (self1.ru_utime + self1.ru_stime) - (self0.ru_utime + self0.ru_stime)
        program = (kids1.ru_utime + kids1.ru_stime) - (kids0.ru_utime + kids0.ru_stime)
        wall = t1 - t0
        return out, self.window(t0, t1, wall, cpu1 - cpu0, mine, program)

    def window(
        self, t0: float, t1: float, wall: float, busy: float, mine: float, program: float
    ) -> dict:
        # Wait (at most 1 s) for the GPU samples that cover the end of the window.
        deadline = time.time() + 1.0
        while self._smi is not None and time.time() < deadline:
            if self.gpu_samples and self.gpu_samples[-1][0] >= t1:
                break
            time.sleep(0.05)
        foreign_cpu = max(0.0, busy - mine - program) / wall if wall > 0 else 0.0
        queue = [(q, own) for t, q, own in list(self.queue_samples) if t0 <= t <= t1]
        rec: dict = {
            "wall_s": wall,
            "program_cpu_s": program,
            "foreign_cpu_cores": foreign_cpu,
            "procs_running": describe([q for q, _ in queue]),
            "foreign_runnable": describe([max(0, q - own - 1) for q, own in queue]),
        }
        reasons = []
        if foreign_cpu > self.max_foreign_cpu:
            reasons.append(f"foreign CPU load {foreign_cpu:.2f} cores > {self.max_foreign_cpu}")
        if self.gpu is not None:
            gpu = [s for s in list(self.gpu_samples) if t0 <= s[0] <= t1]
            busy_gpu = [s for s in gpu if s[4] > 0]
            use = busy_gpu or gpu
            pstates: dict[str, int] = {}
            for s in use:
                pstates[s[1]] = pstates.get(s[1], 0) + 1
            foreign = sorted({(pid, name) for t, pid, name in self.foreign_gpu if t0 <= t <= t1})
            rec["gpu"] = {
                "samples": len(gpu),
                "busy_samples": len(busy_gpu),
                "pstates_busy" if busy_gpu else "pstates": pstates,
                "sm_mhz": describe([s[2] for s in use]),
                "memory_mhz": describe([s[3] for s in use]),
                "foreign_processes": [{"pid": pid, "name": name} for pid, name in foreign],
            }
            if foreign:
                reasons.append(f"foreign compute processes on GPU {self.gpu}: {foreign}")
            if self.locked is not None:
                off = sorted({(s[2], s[3]) for s in busy_gpu if (s[2], s[3]) != self.locked})
                rec["gpu"]["clocks_locked"] = not off
                if off:
                    reasons.append(
                        f"GPU {self.gpu} busy at (SM, memory) MHz {off}, not at the locked "
                        f"{self.locked}"
                    )
        rec["contaminated"] = bool(reasons)
        rec["reasons"] = reasons
        return rec


class ClockLock:
    """The GPU clocks locked for the whole A/B without root (ADR 0018, rule 4): Nsight Compute
    profiles parity/clock_lock/clock_holder.cu under `--clock-control <mode>` and keeps the
    GPU's SM and memory clocks locked while that process lives, for every process on the GPU;
    neither timed program is profiled. On exit the holder ends normally (ncu restores the
    clocks) and `ncu --clock-control reset` resets them in any case. `mode` none does nothing.

    `record` describes the lock for the JSON; `locked` is the (SM, memory) MHz the GPU reports
    under it, which the monitor requires of every busy sample; `pid` is the holder's process."""

    SOURCE = REPO / "parity" / "clock_lock" / "clock_holder.cu"
    READY = re.compile(r"clock_holder ready pid=(\d+)")

    def __init__(self, gpu: int, mode: str) -> None:
        self.gpu = gpu
        self.mode = mode
        self.pid: int | None = None
        self.locked: tuple[int, int] | None = None
        self.record: dict = {"control": mode}
        self._proc: subprocess.Popen | None = None
        self._lines: list[str] = []
        self._reader: threading.Thread | None = None

    def _env(self) -> dict:
        return dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu))

    def _clocks(self) -> tuple[str, int, int]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu),
                "--query-gpu=pstate,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        state, sm, mem = [x.strip() for x in out.strip().split(",")]
        return state, int(sm), int(mem)

    def _build(self) -> Path:
        import hashlib

        digest = hashlib.sha256(self.SOURCE.read_bytes()).hexdigest()[:12]
        exe = SCRATCH / "tools" / "clock_holder" / digest / "clock_holder"
        if not exe.is_file():
            nvcc = os.environ.get("NVCC") or next(
                (
                    str(c)
                    for c in [Path("/usr/local/cuda-13.1/bin/nvcc"), shutil.which("nvcc")]
                    if c and Path(c).is_file()
                ),
                None,
            )
            if nvcc is None:
                raise SystemExit("--lock-clocks: nvcc not found (set NVCC)")
            exe.parent.mkdir(parents=True, exist_ok=True)
            check_call([nvcc, "-O2", "-arch=native", self.SOURCE, "-o", exe])
        return exe

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.append(line)

    def __enter__(self) -> ClockLock:
        if self.mode == "none":
            self.record["note"] = "default GPU clocks (DVFS), as measured"
            return self
        before = self._clocks()
        exe = self._build()
        self._proc = subprocess.Popen(
            [str(NCU), "--clock-control", self.mode, "--metrics", "gpu__time_duration.sum", exe],
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        deadline = time.time() + 120.0
        while self.pid is None:
            for line in list(self._lines):
                m = self.READY.search(line)
                if m:
                    self.pid = int(m.group(1))
            if self.pid is None:
                if self._proc.poll() is not None or time.time() > deadline:
                    self.__exit__()
                    raise SystemExit(
                        "--lock-clocks: the clock holder failed:\n" + "".join(self._lines)
                    )
                time.sleep(0.1)
        time.sleep(1.0)
        state, sm, mem = self._clocks()
        self.locked = (sm, mem)
        version = subprocess.run([str(NCU), "--version"], capture_output=True, text=True)
        self.record.update(
            {
                "tool": (version.stdout.strip().splitlines() or ["ncu"])[-1],
                "how": f"ncu --clock-control {self.mode} on parity/clock_lock/clock_holder.cu "
                "(idle, one kernel) for the whole A/B; the timed programs are not profiled",
                "before": {"pstate": before[0], "sm_mhz": before[1], "memory_mhz": before[2]},
                "locked": {"pstate": state, "sm_mhz": sm, "memory_mhz": mem},
                "check": "every busy GPU sample of every accepted round at the locked clocks",
            }
        )
        print(
            f"GPU {self.gpu}: clocks locked ({self.mode}): SM {sm} MHz, memory {mem} MHz",
            flush=True,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self.mode == "none":
            return
        if self._proc is not None:
            with contextlib.suppress(OSError, ValueError):
                assert self._proc.stdin is not None
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        # A killed ncu session leaves the clocks locked; reset them in every case.
        subprocess.run(
            [str(NCU), "--clock-control", "reset"],
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        with contextlib.suppress(OSError, ValueError, subprocess.CalledProcessError):
            state, sm, mem = self._clocks()
            self.record["after_reset"] = {"pstate": state, "sm_mhz": sm, "memory_mhz": mem}


def monitor_summary(rounds: list[dict], rejected: list[dict], limit: float) -> dict:
    """The monitor's record of a batch (kept per round in the JSON)."""
    sides = ["original", "port"]
    out: dict = {
        "rule": (
            f"a round is repeated if, during either side, the CPUs were busy outside the harness "
            f"for more than {limit} cores on average or a foreign compute process was on the "
            "timed GPU"
        ),
        "max_foreign_cpu_cores": limit,
        "rounds": rounds,
        "rejected": rejected,
        "foreign_cpu_cores_max": max(r[s]["foreign_cpu_cores"] for r in rounds for s in sides),
    }
    if rounds and "gpu" in rounds[0]["original"]:
        for side in sides:
            states: dict[str, int] = {}
            sm = []
            for r in rounds:
                g = r[side]["gpu"]
                for state, n in (g.get("pstates_busy") or g.get("pstates") or {}).items():
                    states[state] = states.get(state, 0) + n
                if g["sm_mhz"]:
                    sm.append(g["sm_mhz"]["median"])
            out[f"{side}_pstates_busy"] = states
            out[f"{side}_sm_mhz_busy_median"] = statistics.median(sm) if sm else None
        if "clocks_locked" in rounds[0]["original"]["gpu"]:
            out["clocks_locked_in_every_round"] = all(
                r[s]["gpu"]["clocks_locked"] for r in rounds for s in sides
            )
    return out


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


def summarize(
    regions: list[dict],
    samples: dict,
    k: int,
    runs: int,
    loads: list,
    a_value=None,
    monitor: dict | None = None,
) -> dict:
    """Medians and gate verdicts per region; side A is the original (its report lines) unless
    a_value reads A's samples otherwise (the edge-type A/B: both sides are the port)."""
    a_value = a_value or original_value
    out = {
        "regions": [],
        "invalidated": {},
        "threads": {},
        "load_average": {"min": min(min(p) for p in loads), "max": max(max(p) for p in loads)},
    }
    if monitor is not None:
        out["monitor"] = monitor
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
            "ratio": mb / ma if ma > 0 else None,  # None: the original has no such timer
            "original_spread": spread(a),
            "port_spread": spread(b),
            "original_samples": a,
            "port_samples": b,
            "gate_kind": gate,
        }
        if dev and all(x is not None for x in dev):
            e["port_device_ms"] = statistics.median(dev)
            e["port_device_samples"] = dev
        if gate in ("compute", "end_to_end") and e["ratio"] is not None:
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
                a = [a_value(s, region, o) for s in samples["original"]]
                b = [port_value(s, region, o) for s in samples["port"]]
                dev = [port_device_value(s, region, o) for s in samples["port"]]
                entry(f"{region['name']} obj{o}", a, b, gate, "as measured", dev)
        else:
            a = [a_value(s, region, None) for s in samples["original"]]
            b = [port_value(s, region, None) for s in samples["port"]]
            entry(region["name"], a, b, gate, "as measured")
    return out


def report(results: dict, labels: tuple[str, str] | None = None) -> list[str]:
    labels = labels or ("original", "dynG")
    lines = [
        f"| batch | region | reading | {labels[0]} (ms) | {labels[1]} (ms) | ratio | gate | "
        f"spread A / B | {labels[1]} device (ms) |",
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
                f"{e['port_ms']:.2f} | "
                + (f"{e['ratio']:.3f}" if e["ratio"] is not None else "-")
                + f" | {g} | "
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
        mon = res.get("monitor")
        if mon:
            line = (
                f"{batch}: monitor: {len(mon['rejected'])} round(s) rejected; foreign CPU load "
                f"<= {mon['foreign_cpu_cores_max']:.2f} cores "
                f"(limit {mon['max_foreign_cpu_cores']})"
            )
            if "original_pstates_busy" in mon:
                line += (
                    f"; GPU busy P-states {labels[0]} {mon['original_pstates_busy']} "
                    f"(SM {mon['original_sm_mhz_busy_median']} MHz), {labels[1]} "
                    f"{mon['port_pstates_busy']} (SM {mon['port_sm_mhz_busy_median']} MHz)"
                )
            print(line)
    return lines


# --- Commands ----------------------------------------------------------------------------------


def prepare(args: argparse.Namespace) -> int:
    # The inputs are always made by MOSP-OpenMP's mospPrep (its bench/prepare.sh); MOSP-CUDA reads
    # the same files, and its generator is the same code (see generators::legacy).
    name = REFERENCES["openmp"]["name"]
    ref = reference_copy(name)
    build_reference(name)
    prep = ref / "bin" / "mospPrep"
    dst = SCRATCH / "bench" / "mosp" / args.graph
    dst.mkdir(parents=True, exist_ok=True)
    csr_dir = dst / "csr"
    env = dict(os.environ, OMP_NUM_THREADS="28", OMP_PROC_BIND="close", OMP_PLACES="cores")
    if args.widen:
        # The K sweep (MOSP-CUDA@e220ee2 results/README.md: "roadNet-CA widened to 4 objectives
        # (mospPrep widen), 50K safe batch generated for it"): a real CSR (not a symlink).
        base, _, widen_k = args.widen.partition(":")
        if not widen_k.isdigit():
            raise SystemExit(f"--widen {args.widen}: expected BASE:K, e.g. roadNet-CA:4")
        if not (csr_dir / "graphCsrValues.txt").is_file() or args.force:
            csr_dir.mkdir(exist_ok=True)
            src_prefix = SCRATCH / "datasets" / "mosp" / base / "csr" / "graphCsr"
            cmd = ["widen", src_prefix, csr_dir / "graphCsr", widen_k, "1", "100", "12345"]
            print(f"mospPrep {' '.join(map(str, cmd))}", flush=True)
            check_call([prep, *cmd], env=env)
    elif not csr_dir.exists():
        csr_dir.symlink_to(
            SCRATCH / "datasets" / "mosp" / args.graph / "csr", target_is_directory=True
        )
    prefix = csr_dir / "graphCsr"
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
    if args.widen:
        steps = [x for x in steps if x[0] in ("init", BATCHES["safe50k"])]
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


def check_port_build(args: argparse.Namespace) -> tuple[Path, dict]:
    """The port's executable and its build record; refuses a non-parity build tree."""
    exe = args.exe.resolve()
    build = port_build(exe, args.backend)
    if not build["parity_preset"] and not args.allow_non_parity_build:
        preset = "parity-cuda" if args.backend == "cuda" else "parity"
        raise SystemExit(
            f"{exe} is not from a {preset}-preset build tree ({build}); the gates are "
            f"defined on the {preset} preset (pass --allow-non-parity-build for an "
            "experiment, whose record says so)"
        )
    return exe, build


def bench_inputs(graph: str) -> tuple[Path, int]:
    """The prepared inputs of a graph and its number of objectives."""
    data = SCRATCH / "bench" / "mosp" / graph
    if not (data / "init").is_dir():
        raise SystemExit(f"{data}: run `parity/perf_ab.py prepare --graph {graph}` first")
    return data, len(list((data / "init").glob("obj*")))


def run_env(args: argparse.Namespace) -> tuple[dict, list]:
    """The environment of both sides and the port's backend arguments."""
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
    return env, port_args


def batch_list(args: argparse.Namespace) -> list[str]:
    batches = [b for b in args.batches.split(",") if b]
    unknown = [b for b in batches if b not in BATCHES]
    if unknown or not batches:
        raise SystemExit(f"--batches: unknown or empty ({unknown}); known: {sorted(BATCHES)}")
    return batches


LAYOUT_STEP = 8  # bytes of file name added per layout (--layouts)


def layout_timing(work: Path, stem: str, args: argparse.Namespace, r: int) -> Path:
    """The --timing file of round r (0-based). With --layouts N > 1 its name grows by
    LAYOUT_STEP * (r mod N) characters, the same on both sides of the round: the argument's length
    moves the program's heap layout (parity/results/M3.md section 6.3), so the rounds of a
    dynG-against-dynG A/B cycle through N layouts instead of reading a single one."""
    layouts = getattr(args, "layouts", 1)
    extra = LAYOUT_STEP * (r % layouts) if layouts > 1 else 0
    # Directory components of at most 200 characters carry what does not fit in the file name.
    folder = work
    while extra > 200:
        folder = folder / ("x" * 200)
        extra -= 200
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}{'x' * extra}.csv"


def batch_args(data: Path, batch: str) -> list:
    return [
        "--graph",
        data / "csr" / "graphCsr",
        "--changes",
        data / BATCHES[batch],
        "--init",
        data / "init",
    ]


def same_outputs(a: Path, b: Path, k: int, what: str) -> None:
    """Correctness guard: the updated trees of two runs must be byte-identical."""
    for o in range(k):
        for f in ["distancesUpdated.txt", "SSSPTreeUpdated.txt"]:
            if not filecmp.cmp(a / f"obj{o}" / f, b / f"obj{o}" / f, shallow=False):
                raise SystemExit(f"{what}: obj{o}/{f} differs")


def monitored_gpu(args: argparse.Namespace) -> int | None:
    """The GPU whose state the monitor samples (cuda runs only)."""
    return args.gpu if args.backend == "cuda" else None


def rejects(args: argparse.Namespace, batch: str, r: int, window: dict, rejected: list) -> bool:
    """Record and reject a contaminated round (it is repeated); fail after too many."""
    reasons = [f"{side}: {x}" for side in ["original", "port"] for x in window[side]["reasons"]]
    if not reasons or args.keep_contaminated:
        return False
    rejected.append({"before_round": r + 1, "reasons": reasons, "windows": window})
    print(f"{batch}: round {r + 1} rejected ({'; '.join(reasons)}); repeating it", flush=True)
    if len(rejected) > args.runs:
        raise SystemExit(
            f"{batch}: {len(rejected)} rounds rejected by the contamination monitor; the machine "
            "is too busy for a gate run (see --max-foreign-cpu)"
        )
    time.sleep(5.0)
    return True


def gpu_note(window: dict) -> str:
    """P-state and SM clock of both sides while the GPU was busy, for the progress lines."""
    notes = []
    for side in ["original", "port"]:
        g = window[side].get("gpu")
        if g and g["sm_mhz"]:
            states = "/".join(sorted(g.get("pstates_busy") or g.get("pstates") or {}))
            notes.append(f"{states} {g['sm_mhz']['median']:.0f} MHz")
    return f"; GPU {' vs '.join(notes)}" if notes else ""


def run(args: argparse.Namespace) -> int:
    regions = load_regions(args.backend)
    keys = report_keys(regions)
    reference = REFERENCES[args.backend]
    exe, build = check_port_build(args)
    data, k = bench_inputs(args.graph)
    # The inputs as measured: hashed now, before the first round (not prepare's digests).
    args.inputs_sha256 = hash_inputs(data)
    env, port_args = run_env(args)
    baseline = args.baseline_exe.resolve() if args.baseline_exe else None
    if baseline is not None:
        # Side A is an earlier dynG build (PLAN 6.3 step 8: a refactor against the code before
        # it), read through the same profiler stages as side B; the ratios are reported, not gated.
        if not baseline.is_file():
            raise SystemExit(f"--baseline-exe {baseline} does not exist")
        regions = [dict(r, gate="none") for r in regions]
        ref, marker = baseline.parent, baseline.parent / ".no-reference-marker"
        side_a = [baseline, *port_args]
    else:
        # Rebuild (idempotent) and verify the unpatched copy before timing it.
        build_reference(reference["name"])
        ref = reference_copy(reference["name"])
        marker = ref / ".dyng-reference"
        side_a = [ref / "bin" / "mosp"]
    batches = batch_list(args)
    results = {}
    failures = []
    work = Path(tempfile.mkdtemp(prefix="dyng-perf-", dir=SCRATCH / "runs"))
    mode = args.lock_clocks if args.backend == "cuda" else "none"
    clocks = ClockLock(args.gpu, mode)
    try:
        with (
            perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock),
            clocks,
        ):
            for batch in batches:
                common = batch_args(data, batch)
                # Correctness guard: both write their outputs once; the files must be identical.
                run_one([*side_a, *common, "--out", work / "A"], env)
                run_one([exe, *common, *port_args, "--out", work / "B"], env)
                same_outputs(work / "A", work / "B", k, f"{batch}: side A and the port")
                print(f"{batch}: outputs byte-identical ({k} objectives)", flush=True)
                samples: dict[str, list] = {"original": [], "port": []}
                loads = []
                timing = work / "timing.csv"
                watched, rejected = [], []
                with MachineMonitor(
                    monitored_gpu(args),
                    args.max_foreign_cpu,
                    allowed_pids={clocks.pid} if clocks.pid else None,
                    locked=clocks.locked,
                ) as monitor:
                    r = 0
                    while r < args.runs:
                        before = os.getloadavg()[0]
                        timing = layout_timing(work, "timing", args, r)
                        if baseline is not None:
                            log_a, win_a = monitor.run(
                                [*side_a, *common, "--no-output", "--timing", timing], env
                            )
                            orig = parse_port(log_a, timing, k)
                            orig["objectives"] = [
                                stage_sum(orig, regions[0]["port"], o) for o in range(k)
                            ]
                        else:
                            log_a, win_a = monitor.run([*side_a, *common, "--no-output"], env)
                            orig = parse_original(log_a, k, keys)
                        log, win_b = monitor.run(
                            [exe, *common, *port_args, "--no-output", "--timing", timing], env
                        )
                        port = parse_port(log, timing, k)
                        window = {"original": win_a, "port": win_b}
                        if rejects(args, batch, r, window, rejected):
                            continue
                        r += 1
                        window["round"] = r
                        watched.append(window)
                        loads.append((before, os.getloadavg()[0]))
                        samples["original"].append(orig)
                        samples["port"].append(port)
                        if orig["invalidated"] != port["invalidated"]:
                            failures.append(
                                f"{batch} round {r}: invalidated original "
                                f"{orig['invalidated']} != port {port['invalidated']}"
                            )
                        print(
                            f"{batch} round {r}/{args.runs}: SOSP original "
                            f"{sum(orig['objectives']):.1f} ms, port "
                            f"{sum(stage_sum(port, regions[0]['port'], o) for o in range(k)):.1f}"
                            f" ms; foreign CPU {win_a['foreign_cpu_cores']:.2f} / "
                            f"{win_b['foreign_cpu_cores']:.2f} cores" + gpu_note(window),
                            flush=True,
                        )
                results[batch] = summarize(
                    regions,
                    samples,
                    k,
                    args.runs,
                    loads,
                    a_value=port_value if baseline is not None else None,
                    monitor=monitor_summary(watched, rejected, args.max_foreign_cpu),
                )
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results, labels=(args.baseline_label, "dynG") if baseline is not None else None)
    exceeded = [
        f"{b}: {e['region']} ({e['reading']}) {e['ratio']:.3f} > {e['gate']:.2f}"
        for b, res in results.items()
        for e in res["regions"]
        if "gate" in e and not e["within_gate"]
    ]
    if args.backend == "cuda":
        print(f"GPU clocks: {json.dumps(clocks.record)}")
    if args.json:
        write_json(args, results, build, ref, marker, regions, reference, clocks.record)
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    if failures:
        return 1
    if exceeded:
        print("gate exceeded: " + "; ".join(exceeded))
        if args.enforce_gates:
            return 1
    return 0


# --- edge-type: the edge_t benchmark (ADR 0009) ---------------------------------------------


def edge_type(args: argparse.Namespace) -> int:
    """dynG with 32-bit (A) against 64-bit (B) edge offsets, A/B/A/B, on the same inputs and
    regions as `run` (both sides read through the port's stages)."""
    regions = [dict(r, gate="none") for r in load_regions(args.backend)]
    exe, build = check_port_build(args)
    data, k = bench_inputs(args.graph)
    env, port_args = run_env(args)
    batches = batch_list(args)
    sides = ["int32", "int64"]
    results, failures = {}, []
    work = Path(tempfile.mkdtemp(prefix="dyng-edge-", dir=SCRATCH / "runs"))
    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for batch in batches:
                common = batch_args(data, batch)
                for side in sides:
                    run_one(
                        [exe, *common, *port_args, "--edge-type", side, "--out", work / side], env
                    )
                same_outputs(work / "int32", work / "int64", k, f"{batch}: int32 vs int64 offsets")
                print(f"{batch}: outputs byte-identical ({k} objectives)", flush=True)
                samples: dict[str, list] = {"original": [], "port": []}
                loads = []
                timing = work / "timing.csv"
                watched, rejected = [], []
                with MachineMonitor(monitored_gpu(args), args.max_foreign_cpu) as monitor:
                    r = 0
                    while r < args.runs:
                        before = os.getloadavg()[0]
                        got, window = {}, {}
                        for side, key in zip(sides, ["original", "port"], strict=True):
                            log, window[key] = monitor.run(
                                [
                                    exe,
                                    *common,
                                    *port_args,
                                    "--edge-type",
                                    side,
                                    "--no-output",
                                    "--timing",
                                    timing,
                                ],
                                env,
                            )
                            got[key] = parse_port(log, timing, k)
                        if rejects(args, batch, r, window, rejected):
                            continue
                        r += 1
                        window["round"] = r
                        watched.append(window)
                        loads.append((before, os.getloadavg()[0]))
                        for key in ["original", "port"]:
                            samples[key].append(got[key])
                        if got["original"]["invalidated"] != got["port"]["invalidated"]:
                            failures.append(f"{batch} round {r}: invalidated differ")
                        print(f"{batch} round {r}/{args.runs}" + gpu_note(window), flush=True)
                results[batch] = summarize(
                    regions,
                    samples,
                    k,
                    args.runs,
                    loads,
                    a_value=port_value,
                    monitor=monitor_summary(watched, rejected, args.max_foreign_cpu),
                )
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results, labels=("int32", "int64"))
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "sssp",
            "benchmark": "edge_t (ADR 0009): dynG int32 (A) vs int64 (B) edge offsets",
            "backend": args.backend,
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "graph": args.graph,
            "port": {"commit": port_commit(), "binary": portable_path(args.exe), "build": build},
            "protocol": {
                "runs": args.runs,
                "order": "A/B/A/B (int32 first)",
                "threads": args.threads,
                "gpu": args.gpu if args.backend == "cuda" else None,
                "lock": LOCK_STATUS,
                "contamination_monitor": f"as `run`; limit {args.max_foreign_cpu} cores",
                "statistic": "median",
            },
            "host": {"cpu": cpu_model(), "logical_cpus": os.cpu_count()},
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    return 1 if failures else 0


# --- engines: the CUDA operators engine against the fused engine (M7, informational) ----------

ENGINE_STAGES = [
    "sssp.enact_fused",
    "sssp.identify_affected",
    "sssp.seed",
    "sssp.loop",
    "sssp.finalize",
]
ENGINE_STAGE = "sssp.engine"  # the stages of whichever engine ran, summed per objective


def with_engine_stage(port: dict, k: int) -> dict:
    """Add the pseudo stage sssp.engine: per objective, the sum of the engine stages that were
    recorded (sssp.enact_fused for the fused engine, the four Tier A hooks for the operators
    engine), host and device times."""
    for kind in ["stages", "device"]:
        rows = port.get(kind, {})
        rows[ENGINE_STAGE] = [
            sum(rows[s][o] for s in ENGINE_STAGES if s in rows and o < len(rows[s]))
            for o in range(k)
        ]
    return port


def engines(args: argparse.Namespace) -> int:
    """dynG's fused CUDA engine (A) against its operators engine (B), A/B/A/B at locked clocks,
    on the inputs and regions of `run --backend cuda` (both sides read through the port's
    stages; the per-objective SOSP region is sssp.enact_fused on A and the four Tier A stages on
    B). Reported, never gated (PLAN 4.5.4: the operators engine is the fallback of a device
    without cooperative launch)."""
    if args.backend != "cuda":
        raise SystemExit("engines: the two engines are CUDA engines (--backend cuda)")
    regions = []
    for r in load_regions("cuda"):
        r = dict(r, gate="none")
        if "sssp.enact_fused" in r.get("port", []):
            r["port"] = [ENGINE_STAGE if s == "sssp.enact_fused" else s for s in r["port"]]
            r["port_device"] = [ENGINE_STAGE]
        regions.append(r)
    exe, build = check_port_build(args)
    data, k = bench_inputs(args.graph)
    env, port_args = run_env(args)
    batches = batch_list(args)
    sides = ["fused", "operators"]
    results, failures = {}, []
    work = Path(tempfile.mkdtemp(prefix="dyng-engines-", dir=SCRATCH / "runs"))
    clocks = ClockLock(args.gpu, args.lock_clocks)
    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock), clocks:
            for batch in batches:
                common = batch_args(data, batch)
                for side in sides:
                    run_one(
                        [exe, *common, *port_args, "--cuda-engine", side, "--out", work / side],
                        env,
                    )
                same_outputs(work / "fused", work / "operators", k, f"{batch}: fused vs operators")
                print(f"{batch}: outputs byte-identical ({k} objectives)", flush=True)
                samples: dict[str, list] = {"original": [], "port": []}
                loads = []
                watched, rejected = [], []
                with MachineMonitor(
                    monitored_gpu(args),
                    args.max_foreign_cpu,
                    allowed_pids={clocks.pid} if clocks.pid else None,
                    locked=clocks.locked,
                ) as monitor:
                    r = 0
                    while r < args.runs:
                        before = os.getloadavg()[0]
                        timing = work / "timing.csv"
                        got, window = {}, {}
                        for side, key in zip(sides, ["original", "port"], strict=True):
                            log, window[key] = monitor.run(
                                [
                                    exe,
                                    *common,
                                    *port_args,
                                    "--cuda-engine",
                                    side,
                                    "--no-output",
                                    "--timing",
                                    timing,
                                ],
                                env,
                            )
                            got[key] = with_engine_stage(parse_port(log, timing, k), k)
                        if rejects(args, batch, r, window, rejected):
                            continue
                        r += 1
                        window["round"] = r
                        watched.append(window)
                        loads.append((before, os.getloadavg()[0]))
                        for key in ["original", "port"]:
                            samples[key].append(got[key])
                        if got["original"]["invalidated"] != got["port"]["invalidated"]:
                            failures.append(f"{batch} round {r}: invalidated differ")
                        print(
                            f"{batch} round {r}/{args.runs}: SOSP fused "
                            f"{sum(got['original']['stages'][ENGINE_STAGE]):.2f} ms, operators "
                            f"{sum(got['port']['stages'][ENGINE_STAGE]):.2f} ms" + gpu_note(window),
                            flush=True,
                        )
                results[batch] = summarize(
                    regions,
                    samples,
                    k,
                    args.runs,
                    loads,
                    a_value=port_value,
                    monitor=monitor_summary(watched, rejected, args.max_foreign_cpu),
                )
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results, labels=("fused", "operators"))
    print(f"GPU clocks: {json.dumps(clocks.record)}")
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "sssp",
            "benchmark": "M7: dynG's CUDA engines, fused (A) vs operators (B); informational",
            "backend": "cuda",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "graph": args.graph,
            "port": {"commit": port_commit(), "binary": portable_path(args.exe), "build": build},
            "protocol": {
                "runs": args.runs,
                "order": "A/B/A/B (fused first)",
                "gpu": args.gpu,
                "clocks": clocks.record,
                "lock": LOCK_STATUS,
                "contamination_monitor": f"as `run`; limit {args.max_foreign_cpu} cores",
                "statistic": "median",
                "region": "per objective: sssp.enact_fused (A) / identify_affected + seed + "
                "loop + finalize (B); host times, device times (CUDA events) next to them",
            },
            "host": {"cpu": cpu_model(), "logical_cpus": os.cpu_count()},
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    return 1 if failures else 0


# --- mosp: the MOSP update against the originals (M7, PLAN 6.4.4) ------------------------------

MOSP_COMBINED = ["distancesCsr.txt", "SSSPTreeCsr.txt", "mospCosts.txt"]


def same_mosp_outputs(a: Path, b: Path, k: int, what: str) -> None:
    """Correctness guard of `mosp`: the K updated trees and the three combinedGraph/ files of two
    runs must be byte-identical."""
    same_outputs(a, b, k, what)
    for f in MOSP_COMBINED:
        if not filecmp.cmp(a / "combinedGraph" / f, b / "combinedGraph" / f, shallow=False):
            raise SystemExit(f"{what}: combinedGraph/{f} differs")


def mosp_options(args: argparse.Namespace) -> list:
    """The MOSP options both sides take (`mosp` and `dyng-compat-mosp --mosp` share them)."""
    out: list = []
    if args.objectives:
        out += ["-k", str(args.objectives)]
    if args.pref:
        out += ["--pref", args.pref]
    return out


def mosp_ab(args: argparse.Namespace) -> int:
    """The MOSP update of dynG (`dyng-compat-mosp --mosp`) against the unpatched original
    (`bin/mosp` of MOSP-OpenMP@c352151 or MOSP-CUDA@e220ee2), A/B/A/B under the perf lock (GPU
    clocks locked on cuda), both sides writing every output file as the original's bench/run.sh
    does; regions from parity/timed_regions/mosp.toml: "(a) compute" and "(b) end to end" gated,
    the per-objective updates, the combined step and the path costs reported."""
    regions = load_regions(args.backend, MOSP_REGION_MAP, ("compute", "end_to_end"))
    keys = report_keys(regions)
    reference = REFERENCES[args.backend]
    exe, build = check_port_build(args)
    data, k_graph = bench_inputs(args.graph)
    # The inputs as measured: hashed now, before the first round (as `run`).
    args.inputs_sha256 = hash_inputs(data)
    k = min(args.objectives, k_graph) if args.objectives else k_graph
    if args.no_output or k < k_graph:
        # mosp::update() computes no path costs then (dyng-compat-mosp: --no-output, or -k below
        # the graph's columns, whose mospCosts.txt is written by the host reference while writing,
        # as the original does): the reported region has no samples and is left out.
        regions = [r for r in regions if r["name"] != "path_costs"]
    env, port_args = run_env(args)
    if args.backend == "cuda" and args.cuda_engine != "automatic":
        port_args = [*port_args, "--cuda-engine", args.cuda_engine]
    build_reference(reference["name"])
    ref = reference_copy(reference["name"])
    marker = ref / ".dyng-reference"
    side_a = [ref / "bin" / "mosp"]
    side_b = [exe, "--mosp", *port_args]
    extra = mosp_options(args)
    batches = batch_list(args)
    results, failures = {}, []
    work = Path(tempfile.mkdtemp(prefix="dyng-perf-mosp-", dir=SCRATCH / "runs"))
    mode = args.lock_clocks if args.backend == "cuda" else "none"
    clocks = ClockLock(args.gpu, mode)

    def clear(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock), clocks:
            for batch in batches:
                common = [*batch_args(data, batch), *extra]
                # Correctness guard: every output file of both sides, byte for byte.
                run_one([*side_a, *common, "--out", work / "A"], env)
                run_one([*side_b, *common, "--out", work / "B"], env)
                same_mosp_outputs(work / "A", work / "B", k, f"{batch}: the original and the port")
                clear(work / "A")
                clear(work / "B")
                print(f"{batch}: every output file byte-identical ({k} objectives)", flush=True)
                samples: dict[str, list] = {"original": [], "port": []}
                loads, watched, rejected = [], [], []
                with MachineMonitor(
                    monitored_gpu(args),
                    args.max_foreign_cpu,
                    allowed_pids={clocks.pid} if clocks.pid else None,
                    locked=clocks.locked,
                ) as monitor:
                    r = 0
                    while r < args.runs:
                        before = os.getloadavg()[0]
                        timing = work / "timing.csv"
                        # Both write their outputs (bench/run.sh), into a directory deleted after
                        # the run, outside the timed program (or neither: --no-output).
                        out_a = ["--no-output"] if args.no_output else ["--out", work / "A"]
                        out_b = ["--no-output"] if args.no_output else ["--out", work / "B"]
                        log_a, win_a = monitor.run([*side_a, *common, *out_a], env)
                        clear(work / "A")
                        orig = parse_original(log_a, k, keys)
                        log_b, win_b = monitor.run(
                            [*side_b, *common, *out_b, "--timing", timing], env
                        )
                        clear(work / "B")
                        port = with_engine_stage(parse_port(log_b, timing, k), k)
                        window = {"original": win_a, "port": win_b}
                        if rejects(args, batch, r, window, rejected):
                            continue
                        r += 1
                        window["round"] = r
                        watched.append(window)
                        loads.append((before, os.getloadavg()[0]))
                        samples["original"].append(orig)
                        samples["port"].append(port)
                        if orig["invalidated"] != port["invalidated"]:
                            failures.append(
                                f"{batch} round {r}: invalidated original "
                                f"{orig['invalidated']} != port {port['invalidated']}"
                            )
                        compute = next(x for x in regions if x["name"] == "compute")
                        print(
                            f"{batch} round {r}/{args.runs}: (a) original "
                            f"{original_value(orig, compute, None):.1f} ms, port "
                            f"{port_value(port, compute, None):.1f} ms; (b) original "
                            f"{orig['report']['end_to_end_ms']:.0f} ms, port "
                            f"{stage_sum(port, ['total.end_to_end']):.0f} ms; foreign CPU "
                            f"{win_a['foreign_cpu_cores']:.2f} / {win_b['foreign_cpu_cores']:.2f}"
                            " cores" + gpu_note(window),
                            flush=True,
                        )
                results[batch] = summarize(
                    regions,
                    samples,
                    k,
                    args.runs,
                    loads,
                    monitor=monitor_summary(watched, rejected, args.max_foreign_cpu),
                )
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    report(results)
    exceeded = [
        f"{b}: {e['region']} {e['ratio']:.3f} > {e['gate']:.2f}"
        for b, res in results.items()
        for e in res["regions"]
        if "gate" in e and not e["within_gate"]
    ]
    if args.backend == "cuda":
        print(f"GPU clocks: {json.dumps(clocks.record)}")
    if args.json:
        write_json(
            args,
            results,
            build,
            ref,
            marker,
            regions,
            reference,
            clocks.record,
            algorithm="mosp",
            region_map=MOSP_REGION_MAP,
            outputs=(
                "--no-output on both (neither computes the path costs)"
                if args.no_output
                else "written by both (bench/run.sh), into a directory deleted after each run"
            ),
            extra={
                "objectives": k,
                "preferences": args.pref or "default (all 1)",
                "cuda_engine": args.cuda_engine if args.backend == "cuda" else None,
            },
        )
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    if failures:
        return 1
    if exceeded:
        print("gate exceeded: " + "; ".join(exceeded))
        if args.enforce_gates:
            return 1
    return 0


# --- kernels: the fused kernels at locked clocks (Nsight Compute) ---------------------------

NCU = Path(os.environ.get("NCU", "/usr/local/cuda-13.1/bin/ncu"))
NCU_KERNELS = "regex:^(sospPersistentKernel|sssp_persistent_kernel)$"
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "sm__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg.per_second",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_warps",
]


def ncu_run(cmd: list, env: dict, k: int, log: Path) -> tuple[list[dict], list[int]]:
    """One run under Nsight Compute with the clocks locked to base (--clock-control base): the
    metrics of the first k fused-kernel launches (the K objectives; MOSP-CUDA launches the kernel
    once more for the combined graph) and the invalidated counters of the program's report."""
    full = [
        NCU,
        "--clock-control",
        "base",
        "--cache-control",
        "none",
        "--kernel-name-base",
        "function",
        "-k",
        NCU_KERNELS,
        "--metrics",
        ",".join(NCU_METRICS),
        "--csv",
        "--log-file",
        log,
        *cmd,
    ]
    out = run_one(full, env)
    ids = parse_ncu_csv(log.read_text(), k, str(cmd[0]))
    inv = {int(o): int(v) for o, _, v in OBJ.findall(out)}
    return ids, [inv.get(o, -1) for o in range(k)]


def parse_ncu_csv(text: str, k: int, what: str) -> list[dict]:
    """The metrics of the first k kernel launches of an `ncu --csv` log (one row per launch and
    metric; launches in ID order)."""
    kernels: dict[int, dict] = {}
    rows = list(csv.reader(text.splitlines()))
    header = next((r for r in rows if r and r[0] == "ID"), None)
    if header is None:
        raise SystemExit(f"{what}: no ncu CSV:\n{text[-2000:]}")
    col = {name: i for i, name in enumerate(header)}
    for r in rows:
        if len(r) != len(header) or r[0] == "ID":
            continue
        raw = r[col["Metric Value"]].replace(",", "")
        try:
            value: float | str | None = float(raw) if raw else None
        except ValueError:
            value = raw
        entry = kernels.setdefault(int(r[col["ID"]]), {"name": r[col["Kernel Name"]]})
        entry[r[col["Metric Name"]]] = value
    ids = sorted(kernels)[:k]
    if len(ids) != k:
        raise SystemExit(f"{what}: {len(kernels)} fused-kernel launches under ncu, need {k}")
    return [kernels[i] for i in ids]


def kernels(args: argparse.Namespace) -> int:
    """The per-objective fused kernels of MOSP-CUDA (A) and dynG (B), A/B/A/B, each run under
    Nsight Compute with the GPU clocks locked to base, which removes the GPU's clock state
    (DVFS P-states) from the comparison: kernel time only (gpu__time_duration)."""
    if args.backend != "cuda":
        raise SystemExit("kernels: --backend cuda only")
    reference = REFERENCES["cuda"]
    exe, build = check_port_build(args)
    build_reference(reference["name"])
    ref = reference_copy(reference["name"])
    mosp = ref / "bin" / "mosp"
    data, k = bench_inputs(args.graph)
    env, port_args = run_env(args)
    batches = batch_list(args)
    results, failures = {}, []
    work = Path(tempfile.mkdtemp(prefix="dyng-ncu-", dir=SCRATCH / "runs"))
    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for batch in batches:
                common = batch_args(data, batch)
                side: dict[str, list] = {"original": [], "port": []}
                for r in range(args.runs):
                    for key, cmd in [
                        ("original", [mosp, *common, "--no-output"]),
                        ("port", [exe, *common, *port_args, "--no-output"]),
                    ]:
                        side[key].append(ncu_run(cmd, env, k, work / f"{key}.csv"))
                    (ka, ia), (kb, ib) = side["original"][-1], side["port"][-1]
                    if ia != ib:
                        failures.append(f"{batch} round {r + 1}: invalidated {ia} != {ib}")
                    print(
                        f"{batch} round {r + 1}/{args.runs}: kernels original "
                        f"{sum(x['gpu__time_duration.sum'] for x in ka) / 1e6:.2f} ms, port "
                        f"{sum(x['gpu__time_duration.sum'] for x in kb) / 1e6:.2f} ms",
                        flush=True,
                    )
                res = {"regions": [], "launch": {}}
                for key in ["original", "port"]:
                    first = side[key][0][0][0]
                    res["launch"][key] = {m: first.get(m) for m in ["name", *NCU_METRICS[4:]]} | {
                        "sm_clock_hz": first.get("sm__cycles_elapsed.avg.per_second"),
                        "dram_clock_hz": first.get("dram__cycles_elapsed.avg.per_second"),
                    }
                for o in range(k):
                    a = [run[0][o]["gpu__time_duration.sum"] / 1e6 for run in side["original"]]
                    b = [run[0][o]["gpu__time_duration.sum"] / 1e6 for run in side["port"]]
                    da = [run[0][o]["dram__bytes.sum"] for run in side["original"]]
                    db = [run[0][o]["dram__bytes.sum"] for run in side["port"]]
                    ma, mb = statistics.median(a), statistics.median(b)
                    gate = gate_limit("compute", ma)
                    res["regions"].append(
                        {
                            "region": f"fused kernel obj{o}",
                            "reading": "kernel time, clocks locked to base (ncu)",
                            "original_ms": ma,
                            "port_ms": mb,
                            "ratio": mb / ma,
                            "gate": gate,
                            "within_gate": mb / ma <= gate,
                            "provisional": ma < SHORT_REGION_MS and args.runs < SHORT_REGION_RUNS,
                            "original_spread": spread(a),
                            "port_spread": spread(b),
                            "noisy": spread(a) > 0.10 or spread(b) > 0.10,
                            "original_samples": a,
                            "port_samples": b,
                            "original_dram_bytes": statistics.median(da),
                            "port_dram_bytes": statistics.median(db),
                        }
                    )
                results[batch] = res
    finally:
        with contextlib.suppress(OSError):
            subprocess.run(["rm", "-rf", str(work)], check=False)
    print("| batch | kernel | original (ms) | dynG (ms) | ratio | gate | spread A / B | DRAM B/A |")
    print("|---|---|---:|---:|---:|---|---|---:|")
    for batch, res in results.items():
        for e in res["regions"]:
            verdict = "ok" if e["within_gate"] else "EXCEEDED"
            print(
                f"| {batch} | {e['region']} | {e['original_ms']:.2f} | {e['port_ms']:.2f} | "
                f"{e['ratio']:.3f} | <= {e['gate']:.2f} {verdict} | "
                f"{e['original_spread'] * 100:.0f} % / {e['port_spread'] * 100:.0f} % | "
                f"{e['port_dram_bytes'] / e['original_dram_bytes']:.3f} |"
            )
        print(f"{batch}: launch {res['launch']}")
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "sssp",
            "benchmark": "fused kernels at locked clocks (Nsight Compute --clock-control base)",
            "backend": "cuda",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "graph": args.graph,
            "reference": {
                "name": reference["name"],
                "commit": reference["commit"],
                "variant": "unpatched",
                "binary": portable_path(mosp),
            },
            "port": {"commit": port_commit(), "binary": portable_path(args.exe), "build": build},
            "protocol": {
                "runs": args.runs,
                "order": "A/B/A/B (side A first: "
                + ("the dynG baseline" if getattr(args, "baseline_exe", None) else "the original")
                + ")",
                "tool": subprocess.run([NCU, "--version"], capture_output=True, text=True)
                .stdout.strip()
                .splitlines()[-1],
                "ncu": "--clock-control base --cache-control none, metrics "
                + ",".join(NCU_METRICS),
                "gpu": args.gpu,
                "lock": LOCK_STATUS,
                "statistic": "median",
            },
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    for f in failures:
        print(f"CORRECTNESS: {f}", file=sys.stderr)
    exceeded = [e for res in results.values() for e in res["regions"] if not e["within_gate"]]
    if failures:
        return 1
    return 1 if exceeded and args.enforce_gates else 0


def port_commit() -> str:
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = (
        subprocess.run(
            ["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", ".", ":(exclude)parity/results"]
        ).returncode
        != 0
    )
    return head + ("+dirty" if dirty else "")


def inputs_fields(args) -> dict:
    """The record's input digests: hashed when the run started (hash_inputs), else (a record of
    an older kind) the digests `prepare` wrote, marked as such."""
    hashed = getattr(args, "inputs_sha256", None)
    if hashed is not None:
        return {"inputs_sha256": hashed, "inputs_hashed": "run start"}
    listed = (SCRATCH / "bench" / "mosp" / args.graph / "INPUTS.sha256").read_text().splitlines()
    return {"inputs_sha256": listed, "inputs_hashed": "prepare"}


def write_json(
    args,
    results,
    build,
    ref,
    marker,
    regions,
    reference,
    clocks=None,
    algorithm: str = "sssp",
    region_map: Path = REGION_MAP,
    outputs: str = "--no-output on both",
    extra: dict | None = None,
) -> None:
    # The records this script writes (parity/results/) do not make the measured code dirty.
    doc = {
        "schema": 3,
        "algorithm": algorithm,
        "backend": args.backend,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "graph": args.graph,
        "reference": (
            {
                "name": reference["name"],
                "commit": reference["commit"],
                "variant": "unpatched",
                "binary": portable_path(ref / "bin" / "mosp"),
                "build": marker.read_text() if marker.is_file() else None,
            }
            if not getattr(args, "baseline_exe", None)
            else {
                "name": "dynG baseline (--baseline-exe; not a gate)",
                "label": args.baseline_label,
                "binary": portable_path(args.baseline_exe),
                "build": port_build(args.baseline_exe.resolve(), args.backend),
            }
        ),
        "port": {
            "commit": port_commit(),
            "binary": portable_path(args.exe),
            "build": build,
        },
        "region_map": {"file": portable_path(region_map), "regions": [r["name"] for r in regions]},
        "protocol": {
            "runs": args.runs,
            "order": "A/B/A/B (side A first: "
            + ("the dynG baseline" if getattr(args, "baseline_exe", None) else "the original")
            + ")",
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
            "lock": LOCK_STATUS,
            "gpu_clocks": clocks if args.backend == "cuda" else "not applicable",
            "contamination_monitor": (
                "per round and side: foreign CPU load (machine busy time minus the harness and "
                "its programs), run queue, GPU P-state / clocks / utilization every 50 ms and "
                "foreign compute processes on the GPU every second; limit "
                f"{args.max_foreign_cpu} cores; contaminated rounds "
                + ("kept and flagged" if args.keep_contaminated else "repeated")
            ),
            "statistic": "median",
            "layouts": (
                f"{args.layouts} heap layouts, round r in layout r mod {args.layouts} (the "
                f"--timing file name {LAYOUT_STEP} * (r mod {args.layouts}) characters longer, "
                "the same on both sides)"
                if getattr(args, "layouts", 1) > 1
                else "one (a fixed --timing file name)"
            ),
            "outputs": outputs,
            "short_regions": f"< {SHORT_REGION_MS} ms need >= {SHORT_REGION_RUNS} runs",
        },
        "host": {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "kernel": platform.release()},
        "driver": driver_versions(),
        **inputs_fields(args),
        "results": results,
    }
    if extra:
        doc.update(extra)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.json}")


def memory(args: argparse.Namespace) -> int:
    """PLAN 8.6 device memory of sssp (or, with --mosp, of mosp) on CUDA: the peak live device
    allocations of MOSP-CUDA and of the port per batch, each measured in one process under Nsight
    Systems."""
    sys.path.insert(0, str(REPO / "parity"))
    import cycle_count_perf

    exe, build = check_port_build(args)
    data, k = bench_inputs(args.graph)
    args.inputs_sha256 = hash_inputs(data)
    reference = REFERENCES["cuda"]
    build_reference(reference["name"])
    ref = reference_copy(reference["name"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_MODULE_LOADING="EAGER")
    work = Path(tempfile.mkdtemp(prefix="dyng-mem-sssp-", dir=SCRATCH / "runs"))
    results: dict = {}
    try:
        with perf_lock(SCRATCH / "perf.lock", args.lock_timeout, args.no_lock):
            for batch in batch_list(args):
                common = [*batch_args(data, batch), "--no-output"]
                row = {
                    "original": cycle_count_perf.device_memory(
                        [ref / "bin" / "mosp", *common], env, work / "a"
                    ),
                    "port": cycle_count_perf.device_memory(
                        [exe, *common, "--backend", "cuda", *(["--mosp"] if args.mosp else [])],
                        env,
                        work / "b",
                    ),
                }
                base = row["original"]["peak_live_mib"]
                row["ratio"] = row["port"]["peak_live_mib"] / base if base else float("nan")
                results[batch] = row
                print(
                    f"{args.graph} {batch}: original {base} MiB, port "
                    f"{row['port']['peak_live_mib']} MiB (pool {row['port']['pool_mib']} MiB), "
                    f"ratio {row['ratio']:.3f}",
                    flush=True,
                )
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": "mosp" if args.mosp else "sssp",
            "backend": "cuda",
            "what": "device memory (PLAN 8.6): the peak of live device allocations per process "
            "(nsys --cuda-memory-usage, memory kind Device: cudaMalloc and cudaMallocAsync), "
            "and the largest stream-ordered pool size nsys reported for the port",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "graph": args.graph,
            "objectives": k,
            "reference": {"name": reference["name"], "binary": portable_path(ref / "bin" / "mosp")},
            "port": {"commit": port_commit(), "binary": portable_path(exe), "build": build},
            "gpu": args.gpu,
            "driver": driver_versions(),
            **inputs_fields(args),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


def main() -> int:
    if sys.argv[1:2] == ["cycle_count"]:  # cycle_count against CycleEnumeration-GPU
        sys.path.insert(0, str(REPO / "parity"))
        import cycle_count_perf

        return cycle_count_perf.main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--graph", default="roadNet-CA")
    p.add_argument(
        "--hops", type=int, help="local batch radius (default: HOPS[graph]; roadNet-CA: 160)"
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--widen",
        metavar="BASE:K",
        help="the K sweep input (M7): the CSR of BASE widened to K objectives with "
        "`mospPrep widen <BASE csr> <out> K 1 100 12345` (the existing columns unchanged), its "
        "initial trees and its 50K safe batch of seed 777, into bench/mosp/<--graph>/",
    )
    for command, help_text in [
        ("run", "the port against the unpatched original (the gates)"),
        ("edge-type", "the edge_t benchmark: the port with int32 (A) vs int64 (B) edge offsets"),
        ("engines", "cuda: the port's fused (A) vs operators (B) engine, reported (M7)"),
        ("kernels", "cuda: the fused kernels of both under Nsight Compute, clocks locked to base"),
        ("memory", "cuda: the peak device memory of both under Nsight Systems (PLAN 8.6)"),
        ("mosp", "the MOSP update against the original's bin/mosp, outputs written (M7 gates)"),
    ]:
        r = sub.add_parser(command, help=help_text)
        r.add_argument("--exe", type=Path, required=True, help="dyng-compat-mosp (parity preset)")
        r.add_argument(
            "--backend",
            choices=sorted(REFERENCES),
            default="cuda" if command in ("kernels", "memory", "engines") else "openmp",
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
            "--max-foreign-cpu",
            type=float,
            default=MAX_FOREIGN_CPU,
            metavar="CORES",
            help="contamination monitor: repeat a round when the CPUs were busy outside the "
            "harness for more than CORES on average during either side "
            f"(default {MAX_FOREIGN_CPU})",
        )
        r.add_argument(
            "--keep-contaminated",
            action="store_true",
            help="record contaminated rounds instead of repeating them (flagged in the JSON)",
        )
        if command == "mosp":
            r.add_argument(
                "-k",
                "--objectives",
                type=int,
                default=0,
                help="objectives used (`mosp -k`; default all of the graph's)",
            )
            r.add_argument("--pref", default="", help="preferences p1,..,pK (`mosp --pref`)")
            r.add_argument(
                "--cuda-engine",
                choices=["automatic", "fused", "operators"],
                default="automatic",
                help="--backend cuda: dyng-compat-mosp --cuda-engine (default automatic)",
            )
            r.add_argument(
                "--no-output",
                action="store_true",
                help="time both with --no-output (no files written, no path costs) instead of "
                "writing every output file as bench/run.sh does (the published (b) scope)",
            )
        if command == "memory":
            r.add_argument(
                "--mosp",
                action="store_true",
                help="run the port as dyng-compat-mosp --mosp (the whole MOSP update, the "
                "original's scope; M7) instead of the K sssp updates",
            )
        if command in ("engines", "mosp"):
            r.add_argument(
                "--lock-clocks",
                choices=["boost", "base", "none"],
                default="boost",
                help="lock the GPU's clocks for the whole A/B (ADR 0018; as `run`)",
            )
        if command == "run":
            r.add_argument(
                "--baseline-exe",
                type=Path,
                help="replace side A (the original) by an earlier dynG build's dyng-compat-mosp, "
                "read through the same profiler stages (PLAN 6.3 step 8: a refactor against the "
                "code before it); the ratios are reported, not gated",
            )
            r.add_argument(
                "--baseline-label",
                default="baseline",
                help="the name of side A in the report with --baseline-exe (e.g. a commit)",
            )
            r.add_argument(
                "--layouts",
                type=int,
                default=1,
                metavar="N",
                help="with --baseline-exe: cycle the rounds through N heap layouts (the --timing "
                "file name grows by 8 characters per layout, the same on both sides; M3.md "
                "section 6.3); 1 (the default) reads one layout",
            )
            r.add_argument(
                "--lock-clocks",
                choices=["boost", "base", "none"],
                default="boost",
                help="--backend cuda: lock the GPU's clocks for the whole A/B with Nsight "
                "Compute (no root; ADR 0018): boost (the default: the highest lockable clocks, "
                "on the RTX A5000 SM 1695 MHz / memory 7601 MHz), base, or none (default clocks, "
                "DVFS: the as-measured reading)",
            )
        r.add_argument(
            "--enforce-gates",
            action="store_true",
            help="exit 1 if a gated region exceeds its limit (the gates bind from M1b)",
        )
    args = parser.parse_args()
    if args.command != "prepare" and args.runs < 5:
        parser.error("--runs must be >= 5 (PLAN Section 6.3 step 7)")
    if getattr(args, "layouts", 1) < 1:
        parser.error("--layouts must be >= 1")
    if getattr(args, "layouts", 1) > 1 and not getattr(args, "baseline_exe", None):
        parser.error(
            "--layouts is for a dynG-against-dynG A/B (--baseline-exe): the originals "
            "take no --timing file, so only the port's layout would move"
        )
    if args.command == "prepare" and args.widen:
        args.hops = 0  # the widened graph gets the 50K safe batch only
    if args.command == "prepare" and args.hops is None:
        if args.graph not in HOPS:
            parser.error(f"--hops is required for {args.graph} (known: {sorted(HOPS)})")
        args.hops = HOPS[args.graph]
    # SIGTERM / SIGHUP unwind like Ctrl-C, so that a clock lock is always released.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
    commands = {
        "prepare": prepare,
        "run": run,
        "edge-type": edge_type,
        "engines": engines,
        "kernels": kernels,
        "memory": memory,
        "mosp": mosp_ab,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
