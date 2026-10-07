#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The informational "wheel vs parity build" row (PLAN 7.7; ADR 0032): the same CUDA work, timed
from Python through the CUDA plugin wheel and through the `parity-cuda` build.

PLAN 8.6's gates apply to the `parity` presets; a wheel differs (another host compiler, the
release architecture list instead of `-arch=sm_86`, the static CUDA runtime, libdyng linked
into the module, lazy module loading), so the benchmark report shows how far a wheel is from
the parity build, without gating on it. Both sides run the SAME driver code in this file
(`worker`), each in its own interpreter, with the inputs of the gates:

- **sssp** (`ipdps25_dynamosp_sosp`, CUDA reading): roadNet-CA, K = 3 objectives, the initial
  trees of `mospPrep init` (`Result.from_arrays`), one batch of the suite (default `unsafe50k`,
  50K changes) applied with `dyng.update(g, batch, [r0, r1, r2])` as dyng-compat-mosp does;
  timed: the stage `sssp.enact_fused` per objective (the CUDA region `sosp_update` of
  parity/timed_regions/sssp.toml) and the whole update (`update.commit` included);
- **cycle_count** (`ieee_tc_dyntrucy`, CUDA reading): DD (`DD_A.txt`, read as
  dyng-compat-cycle-enum reads it), k = 4: the count (stage `cycle_count.count`, the region
  `static_kernel`) and an update of 25K deletions + 25K insertions (stage `cycle_count.update`).
  The update's batch is drawn here (numpy, seed 1) and shared by both sides; it is not the
  original's generator (generators::legacy::cycle_enum_batch is not bound to Python).

Protocol (PLAN 8.6 / ADR 0018, as far as an informational row needs it): the exclusive perf
lock, GPU 0, the clocks locked with Nsight Compute (parity/perf_ab.py's ClockLock, `boost`), one
long-lived worker per side, an untimed warm-up round, then alternating rounds (A, B, A, B, ...);
medians and the ratio of each side to the first (`--side`). Host times of the profiler (with
CUDA events on, as the compat tools record them); device times are reported next to them.

Usage::

    # the parity side: the parity-cuda preset with the Python bindings, as an importable overlay
    parity/wheel_vs_parity.py build-parity --build-dir $DYNG_SCRATCH/build/wvp-parity-cuda \\
        --overlay $DYNG_SCRATCH/wvp/parity-overlay
    W=$DYNG_SCRATCH/wvp
    parity/wheel_vs_parity.py run \\
        --side parity-cuda=$W/venv-parity/bin/python:$W/parity-overlay \\
        --side wheel-cu13=$W/venv-wheel/bin/python \\
        --json parity/results/M6a-wheel-vs-parity.json

A side is ``name=<python>[:<PYTHONPATH>]``; the wheel side's interpreter has the core and plugin
wheels installed, the parity side's has NumPy only and imports ``dyng`` from the overlay.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
MOSP_INPUTS = SCRATCH / "bench" / "mosp"
CYCLE_INPUTS = SCRATCH / "datasets" / "cycle"
BATCHES = {
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
    "local10k": "changes_local_10000_50_safe",
}


# -------------------------------------------------------------------------------------------------
# The worker: one per side, in that side's interpreter
# -------------------------------------------------------------------------------------------------


def _stage_samples(prof, name: str) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    return [(s.host_ms, s.device_ms) for s in prof.samples if s.name == name]


def worker(args: argparse.Namespace) -> int:
    import dyng
    import dyng._backend as backend
    import numpy as np

    cuda = dyng.Resources.cuda(0)
    cuda.warm_up()
    host = dyng.Resources.sequential()
    build = dict(dyng.config()["build"])
    info = {
        "module": backend.active_module_name,
        "dyng": dyng.__version__,
        "dyng_file": dyng.__file__,
        "build": {k: build.get(k) for k in sorted(build)},
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    if args.task == "sssp":
        data = MOSP_INPUTS / args.graph
        base = dyng.io.read_csr_triplet(
            data / "csr" / "graphCsr", properties="mosp_compatible", resources=host
        )
        k = base.num_weights
        batch = dyng.io.read_legacy_batch(
            data / BATCHES[args.batch] / "insert.txt",
            data / BATCHES[args.batch] / "delete.txt",
            num_weights=k,
            num_vertices=base.num_vertices,
        )
        init = [
            (
                dyng.io.read_distances(
                    data / "init" / f"obj{j}" / "distancesOriginal.txt", base.num_vertices
                ),
                dyng.io.read_parents(
                    data / "init" / f"obj{j}" / "SSSPTreeOriginal.txt", base.num_vertices
                ),
            )
            for j in range(k)
        ]
        info["inputs"] = {
            "graph": args.graph,
            "vertices": base.num_vertices,
            "edges": base.num_edges,
            "objectives": k,
            "batch": args.batch,
        }

        def one_round() -> dict:
            g = base.to_backend(cuda)
            results = [
                dyng.sssp.Result.from_arrays(g, 0, d, p, canonicalize=False, objective=j)
                for j, (d, p) in enumerate(init)
            ]
            t0 = time.perf_counter()
            with dyng.profile(g.resources, cuda_events=True) as prof:
                dyng.update(g, batch, *results)
            wall = (time.perf_counter() - t0) * 1e3
            enact = _stage_samples(prof, "sssp.enact_fused")
            assert len(enact) == k, [s.name for s in prof.samples]
            checksum = int(sum(int(r.distances.to_numpy().sum()) for r in results))
            return {
                "sosp_update_host_ms": [h for h, _ in enact],
                "sosp_update_device_ms": [d for _, d in enact],
                "sosp_total_host_ms": sum(h for h, _ in enact),
                "update_host_ms": sum(s.host_ms for s in prof.samples if s.depth == 0),
                "commit_host_ms": prof.total_host_ms("update.commit"),
                "call_wall_ms": wall,
                "checksum": checksum,
            }

    else:
        path = CYCLE_INPUTS / "DD" / "DD_A.txt"
        base = dyng.io.read_edge_list(path, properties="cycle_enum_compatible", resources=host)
        rng = np.random.default_rng(1)
        csr = base.to_csr()
        row_ptr = np.asarray(csr.row_ptr)
        src = np.repeat(np.arange(base.num_vertices, dtype=np.int32), np.diff(row_ptr))
        dst = np.asarray(csr.col_ind, dtype=np.int32)
        pick = rng.choice(len(src), size=args.deletes, replace=False)
        existing = set(zip(src.tolist(), dst.tolist(), strict=True))
        ins: list[tuple[int, int]] = []
        while len(ins) < args.inserts:
            u, v = (int(x) for x in rng.integers(0, base.num_vertices, 2))
            if u != v and (u, v) not in existing:
                existing.add((u, v))
                ins.append((u, v))
        ins_a = np.array(ins, dtype=np.int32)
        batch = dyng.EdgeBatch(insert=(ins_a[:, 0], ins_a[:, 1]), delete=(src[pick], dst[pick]))
        info["inputs"] = {
            "graph": "DD",
            "vertices": base.num_vertices,
            "edges": base.num_edges,
            "k": args.k,
            "batch": f"{args.deletes} deletions + {args.inserts} insertions (numpy, seed 1)",
        }

        def one_round() -> dict:
            g = base.to_backend(cuda)
            with dyng.profile(g.resources, cuda_events=True) as prof:
                h = dyng.cycle_count.compute(g, max_length=args.k)
            count = _stage_samples(prof, "cycle_count.count")
            g2 = base.to_backend(cuda)
            prior = dyng.cycle_count.compute(g2, max_length=args.k)
            with dyng.profile(g2.resources, cuda_events=True) as prof2:
                dyng.cycle_count.update(g2, batch, prior)
            upd = _stage_samples(prof2, "cycle_count.update")
            assert len(count) == 1 and len(upd) == 1
            return {
                "count_host_ms": count[0][0],
                "count_device_ms": count[0][1],
                "update_host_ms": upd[0][0],
                "update_device_ms": upd[0][1],
                "checksum": [int(h.total), int(prior.total)],
            }

    print(json.dumps({"ready": info}), flush=True)
    for line in sys.stdin:
        if line.strip() != "round":
            break
        print(json.dumps({"round": one_round()}), flush=True)
    return 0


# -------------------------------------------------------------------------------------------------
# The parity side's module
# -------------------------------------------------------------------------------------------------


def build_parity(args: argparse.Namespace) -> int:
    """Configure and build the parity-cuda preset with the Python bindings (module `_core`
    only), then assemble an overlay directory `<overlay>/dyng` = python/dyng + that module."""
    build = args.build_dir
    cmd = [
        "cmake",
        "--preset",
        "parity-cuda",
        "-B",
        str(build),
        "-DDYNG_BUILD_PYTHON=ON",
        "-DDYNG_BUILD_TESTS=OFF",
        "-DDYNG_BUILD_PARITY_TESTS=OFF",
        "-DDYNG_BUILD_COMPAT_TOOLS=OFF",
        f"-DPython_EXECUTABLE={sys.executable}",
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "_core", "-j", str(args.jobs)],
        cwd=REPO,
        check=True,
    )
    modules = sorted(build.rglob("_core*.so"))
    if len(modules) != 1:
        raise SystemExit(f"expected one _core module under {build}, found {modules}")
    overlay = args.overlay / "dyng"
    if overlay.exists():
        shutil.rmtree(overlay)
    shutil.copytree(REPO / "python" / "dyng", overlay, ignore=shutil.ignore_patterns("_core*.so"))
    shutil.copy2(modules[0], overlay / modules[0].name)
    print(f"parity-cuda module: {overlay / modules[0].name} (libdyng from {build})")
    return 0


# -------------------------------------------------------------------------------------------------
# The A/B driver
# -------------------------------------------------------------------------------------------------


class Side:
    def __init__(self, spec: str, task_args: list[str], gpu: int) -> None:
        self.name, _, rest = spec.partition("=")
        python, _, pythonpath = rest.partition(":")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        env.pop("PYTHONPATH", None)
        env.pop("DYNG_CPU_ONLY", None)
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        self.proc = subprocess.Popen(
            [python, str(Path(__file__).resolve()), "worker", *task_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            cwd=SCRATCH,  # not the source tree: the side's own dyng is imported
        )
        self.info = self._read()["ready"]
        self.rounds: list[dict] = []

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"{self.name}: the worker ended ({self.proc.wait()})")
        return dict(json.loads(line))

    def round(self) -> dict:
        assert self.proc.stdin is not None
        self.proc.stdin.write("round\n")
        self.proc.stdin.flush()
        return dict(self._read()["round"])

    def close(self) -> None:
        if self.proc.stdin is not None:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
        self.proc.wait(timeout=120)


def _metrics(task: str) -> list[str]:
    if task == "sssp":
        return [
            "sosp_update_host_ms[0]",
            "sosp_update_host_ms[1]",
            "sosp_update_host_ms[2]",
            "sosp_total_host_ms",
            "update_host_ms",
            "sosp_update_device_ms[0]",
        ]
    return ["count_host_ms", "count_device_ms", "update_host_ms", "update_device_ms"]


def _value(r: dict, metric: str) -> float:
    if metric.endswith("]"):
        key, _, idx = metric[:-1].partition("[")
        return float(r[key][int(idx)])
    return float(r[metric])


def run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "parity"))
    import perf_ab  # the lock and the clock lock of the gates' harness

    results: dict = {
        "what": "informational wheel vs parity build row (PLAN 7.7); not a gate",
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "gpu": args.gpu,
        "rounds": args.rounds,
        "clocks": args.lock_clocks,
        "driver": perf_ab.driver_versions(),
        "tasks": {},
    }
    if args.no_lock and args.json:
        raise SystemExit("--no-lock is for trying the script; a recorded row takes the lock")
    with perf_ab.perf_lock(SCRATCH / "perf.lock", skip=args.no_lock):
        lock = perf_ab.ClockLock(args.gpu, args.lock_clocks)
        with lock:
            results["clock_lock"] = lock.record
            for task in args.tasks:
                task_args = ["--task", task, "--graph", args.graph, "--batch", args.batch]
                task_args += ["--k", str(args.k)]
                sides = [Side(s, task_args, args.gpu) for s in args.side]
                try:
                    for s in sides:
                        s.round()  # warm-up, untimed
                    for _ in range(args.rounds):
                        for s in sides:
                            s.rounds.append(s.round())
                finally:
                    for s in sides:
                        s.close()
                checks = {json.dumps(s.rounds[0]["checksum"]) for s in sides}
                if len(checks) != 1:
                    raise SystemExit(f"{task}: the sides computed different results: {checks}")
                table = {}
                for m in _metrics(task):
                    row = {s.name: statistics.median(_value(r, m) for r in s.rounds) for s in sides}
                    base = row[sides[0].name]
                    table[m] = {
                        "median_ms": row,
                        "ratio_to_" + sides[0].name: {
                            n: (v / base if base else None) for n, v in row.items()
                        },
                    }
                results["tasks"][task] = {
                    "sides": {s.name: s.info for s in sides},
                    "metrics": table,
                    "samples": {s.name: s.rounds for s in sides},
                }
    for task, t in results["tasks"].items():
        names = list(t["sides"])
        print(
            f"\n{task}: medians of {args.rounds} alternating rounds (ms), GPU {args.gpu}, "
            f"clocks {args.lock_clocks}"
        )
        print("| metric | " + " | ".join(names) + f" | {names[-1]} / {names[0]} |")
        print("|---|" + "---|" * (len(names) + 1))
        for m, row in t["metrics"].items():
            vals = [row["median_ms"][n] for n in names]
            ratio = vals[-1] / vals[0] if vals[0] else float("nan")
            print(f"| {m} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {ratio:.3f} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nrecord: {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--task", choices=["sssp", "cycle_count"], required=True)
    r = sub.add_parser("run")
    for q in (w, r):
        q.add_argument("--graph", default="roadNet-CA")
        q.add_argument("--batch", default="unsafe50k", choices=sorted(BATCHES))
        q.add_argument("--k", type=int, default=4)
    w.add_argument("--deletes", type=int, default=25000)
    w.add_argument("--inserts", type=int, default=25000)
    r.add_argument("--side", action="append", required=True, help="name=<python>[:<PYTHONPATH>]")
    r.add_argument("--tasks", nargs="+", default=["sssp", "cycle_count"])
    r.add_argument("--rounds", type=int, default=21)
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--lock-clocks", default="boost", choices=["boost", "base", "none"])
    r.add_argument("--json", type=Path)
    r.add_argument("--no-lock", action="store_true", help="trial runs only (no --json)")
    b = sub.add_parser("build-parity")
    b.add_argument("--build-dir", type=Path, required=True)
    b.add_argument("--overlay", type=Path, required=True)
    b.add_argument("--jobs", type=int, default=16)
    args = p.parse_args(argv)
    if args.command == "worker":
        return worker(args)
    if args.command == "build-parity":
        return build_parity(args)
    if len(args.side) < 2:
        p.error("run needs two --side")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
