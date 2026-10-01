#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Run a benchmark suite of benchmarks/paper/ through the A/B harness (PLAN Sections 8.5, 8.6).

    parity/bench_suite.py validate benchmarks/paper/*.yaml
    parity/bench_suite.py plan     benchmarks/paper/ieee_tc_dyntrucy.yaml [--readings cuda]
    parity/bench_suite.py run      benchmarks/paper/ieee_tc_dyntrucy.yaml --version 0.1.0rc1
                                   [--readings openmp,cuda] [--datasets ...] [--runs N]
                                   [--build-root build] [--skip-existing] [--no-verify-inputs]
    parity/bench_suite.py summarize benchmarks/paper/ieee_tc_dyntrucy.yaml --version 0.1.0rc1

A suite file (docs/developer/benchmarks.md) is machine-readable: datasets with their SHA-256
digests and the recipe that makes them, batches with their seeds, backends, the number of runs,
the metrics as regions of parity/timed_regions/<algorithm>.toml with the dynG profiler stages they
sum, the baselines (the pinned originals of parity/references.toml, unpatched copies), the
tolerances of PLAN 8.6, and the readings: one reading is one harness invocation per dataset (sssp)
or per case list (cycle_count), gated or recorded only (the default-clock readings of ADR 0018).

validate   checks a suite file against itself and against the harness it drives: the region map
           (every metric is a region with the same gate and the same profiler stages), the
           pinned commits (parity/references.toml), the harness's case lists and batch names, and
           the tolerances the harness applies. Exit 1 on any finding.
plan       prints the harness commands one execution of the suite runs (nothing is run).
run        verifies the inputs against the suite's digests (SHA-256 of every file; skip with
           --no-verify-inputs), then runs every planned command: parity/perf_ab.py run|memory for
           sssp, parity/perf_ab.py cycle_count run|memory for cycle_count. Each timing command
           takes the exclusive perf lock $DYNG_SCRATCH/perf.lock itself and runs only on a
           parity-preset build (--build-root/<preset>/tools/compat/...); the CUDA readings lock
           the GPU clocks for the whole A/B (--lock-clocks of the reading: boost, base, or none
           for the recorded default-clock reading). The cycle_count memory reading takes the
           shared lock (it times nothing). The full records go to --records (default
           $DYNG_SCRATCH/runs/bench/<version>); then `summarize` runs.
summarize  reads the records of every planned command and writes, under
           benchmarks/results/<version>/ (--out), a compacted copy of each record (the per-round
           machine-monitor windows dropped; verdicts, samples, rejected rounds and clock checks
           kept) and <suite>.json: the readings with every region's medians, ratio, gate and
           verdict, the inputs check and the overall verdict. It re-derives every gate from the
           suite's tolerances and refuses a record whose gate, reference commit, inputs, clock
           lock or build preset differs from the suite. Exit 1 if a gated region exceeds its
           gate, a gated reading is missing or incomplete, or a check fails.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
sys.path.insert(0, str(REPO / "parity"))

SCHEMA = 1
TOOLS = {"sssp": "perf_ab.sssp", "cycle_count": "perf_ab.cycle_count"}
REGION_MAPS = {
    "sssp": "parity/timed_regions/sssp.toml",
    "cycle_count": "parity/timed_regions/cycle_count.toml",
}
# The region map's section per backend.
MAP_KEYS = {
    "sssp": {"openmp": "mosp_openmp", "cuda": "mosp_cuda"},
    "cycle_count": {"openmp": "cycle_enum_openmp", "cuda": "cycle_enum_cuda"},
}
EXECUTABLES = {"sssp": "dyng-compat-mosp", "cycle_count": "dyng-compat-cycle-enum"}
PRESETS = {"openmp": "parity", "cuda": "parity-cuda"}
CLOCKS = ("boost", "base", "none")
KINDS = ("run", "memory")
TOLERANCE_KEYS = ("compute", "compute_short", "short_region_ms", "end_to_end", "memory")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
CASE_NAME = re.compile(
    r"^(?P<dataset>[A-Za-z0-9-]+)_k(?P<k>\d+)(?:_(?P<d>\d+)_(?P<i>\d+)_s(?P<s>\d+))?$"
)
COMPACTED = (
    "the per-round machine-monitor windows (results.*.monitor.rounds, and the windows of the "
    "rejected rounds) were dropped to keep the repository small; the verdicts, samples, rejected "
    "rounds with their reasons, clock check and GPU summary are unchanged; the full record: {full}"
)


class SuiteError(Exception):
    """A suite file, a record or an input that does not match what the suite says."""


# --- Loading and validation ----------------------------------------------------------------------


def load_suite(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - the dev env and lint.yml install PyYAML
        raise SystemExit(
            "bench_suite.py needs PyYAML (conda env dyng-dev, or pip install pyyaml)"
        ) from exc
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise SuiteError(f"{path}: not a mapping")
    doc["_path"] = str(Path(path))
    return doc


def references() -> dict[str, dict]:
    doc = tomllib.loads((REPO / "parity" / "references.toml").read_text())
    return {r["name"]: r for r in doc["reference"]}


def region_map(algorithm: str, backend: str) -> dict[str, dict]:
    doc = tomllib.loads((REPO / REGION_MAPS[algorithm]).read_text())
    return {r["name"]: r for r in doc["reference"][MAP_KEYS[algorithm][backend]]["region"]}


def metric_gate(metric: dict, backend: str) -> str:
    gate = metric.get("gate")
    return gate.get(backend) if isinstance(gate, dict) else gate


def harness_cases(backend: str) -> set[str]:
    import cycle_count_goldens as goldens

    cases = goldens.cuda_cases() if backend == "cuda" else goldens.all_cases()
    return {c.name for c in cases}


def case_name(case: dict, batches: dict[str, dict]) -> str:
    if case.get("task") == "count":
        return f"{case['dataset']}_k{case['k']}"
    b = batches[case["batch"]]
    return f"{case['dataset']}_k{case['k']}_{b['deletions']}_{b['insertions']}_s{b['seed']}"


def validate(suite: dict) -> list[str]:
    """Every finding about a suite file (empty: valid)."""
    errors: list[str] = []
    path = suite.get("_path", "<suite>")

    def need(cond: bool, message: str) -> bool:
        if not cond:
            errors.append(f"{path}: {message}")
        return cond

    need(suite.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    name = suite.get("suite")
    need(isinstance(name, str) and name == Path(path).stem, "suite must equal the file name")
    algorithm = suite.get("algorithm")
    if not need(algorithm in TOOLS, f"algorithm must be one of {sorted(TOOLS)}"):
        return errors
    harness = suite.get("harness") or {}
    need(harness.get("tool") == TOOLS[algorithm], f"harness.tool must be {TOOLS[algorithm]}")
    need(
        harness.get("region_map") == REGION_MAPS[algorithm],
        f"harness.region_map must be {REGION_MAPS[algorithm]}",
    )
    need(
        harness.get("executable") == f"tools/compat/{EXECUTABLES[algorithm]}",
        f"harness.executable must be tools/compat/{EXECUTABLES[algorithm]}",
    )
    need(harness.get("presets") == PRESETS, f"harness.presets must be {PRESETS}")
    for key in (
        "title",
        "paper",
        "datasets",
        "batches",
        "backends",
        "metrics",
        "baselines",
        "tolerance",
        "readings",
    ):
        need(key in suite, f"missing key {key!r}")
    if errors:
        return errors

    # Datasets: a name, where it comes from, and its digests.
    datasets = {}
    for d in suite["datasets"]:
        dname = d.get("name")
        if not need(
            isinstance(dname, str) and dname not in datasets,
            f"dataset name {dname!r} missing or repeated",
        ):
            continue
        datasets[dname] = d
        need("source" in d and "recipe" in d, f"dataset {dname}: needs source and recipe")
        digest = d.get("sha256")
        digests = digest.values() if isinstance(digest, dict) else [digest]
        need(
            bool(digests) and all(isinstance(x, str) and HEX64.match(x) for x in digests),
            f"dataset {dname}: sha256 must be a SHA-256 or a mapping file -> SHA-256",
        )
    batches = {}
    for b in suite["batches"]:
        bname = b.get("name")
        if need(
            isinstance(bname, str) and bname not in batches,
            f"batch name {bname!r} missing or repeated",
        ):
            batches[bname] = b
            need(isinstance(b.get("seed"), int), f"batch {bname}: needs an integer seed")
            need(isinstance(b.get("generator"), str), f"batch {bname}: needs a generator")

    # Backends and baselines: the pinned originals of references.toml.
    refs = references()
    backends = suite["backends"]
    for backend, spec in backends.items():
        if not need(backend in PRESETS, f"unknown backend {backend!r}"):
            continue
        base = spec.get("baseline")
        need(
            base in suite["baselines"], f"backend {backend}: baseline {base!r} is not in baselines"
        )
    for bname, spec in suite["baselines"].items():
        if not need(bname in refs, f"baseline {bname} is not in parity/references.toml"):
            continue
        need(
            spec.get("commit") == refs[bname]["commit"],
            f"baseline {bname}: commit is not the pinned {refs[bname]['commit']}",
        )
        need(
            spec.get("variant") == "unpatched",
            f"baseline {bname}: performance baselines are the unpatched copy",
        )

    # Tolerances: the ones PLAN 8.6 names and the harness applies.
    import perf_ab

    tol = suite["tolerance"]
    for key in TOLERANCE_KEYS:
        need(isinstance(tol.get(key), int | float), f"tolerance.{key} missing")
    if not errors:
        need(
            tol["compute"] == 1.05 and tol["compute_short"] == 1.10,
            "tolerance: compute 1.05, compute_short 1.10 (PLAN 8.6)",
        )
        need(
            tol["end_to_end"] == perf_ab.END_TO_END_GATE,
            f"tolerance.end_to_end must be {perf_ab.END_TO_END_GATE}",
        )
        need(
            tol["short_region_ms"] == perf_ab.SHORT_REGION_MS,
            f"tolerance.short_region_ms must be {perf_ab.SHORT_REGION_MS}",
        )
        need(tol["memory"] == 1.05, "tolerance.memory must be 1.05 (PLAN 8.6)")

    # Metrics: regions of the map with the same gate and stages.
    for mname, metric in suite["metrics"].items():
        for backend in backends:
            if backend not in metric:
                continue
            if metric_gate(metric, backend) == "memory":
                need(
                    mname == "device_memory",
                    f"metric {mname}: only device_memory has the memory gate",
                )
                continue
            regions = region_map(algorithm, backend)
            if not need(
                mname in regions, f"metric {mname}: no region {mname} in the {backend} region map"
            ):
                continue
            region = regions[mname]
            need(
                metric_gate(metric, backend) == region["gate"],
                f"metric {mname} ({backend}): gate {metric_gate(metric, backend)!r}, "
                f"the region map says {region['gate']!r}",
            )
            stages = list(region.get("port", [])) + list(region.get("port_all_results", []))
            stages = stages or list(region.get("port_report", []))
            need(
                metric[backend].get("port") == stages,
                f"metric {mname} ({backend}): port stages {metric[backend].get('port')} "
                f"!= region map {stages}",
            )

    # Every gated region of the map is a metric of the suite.
    for backend in backends:
        if backend not in PRESETS:
            continue
        for rname, region in region_map(algorithm, backend).items():
            if region["gate"] != "none":
                need(
                    backend in suite["metrics"].get(rname, {}),
                    f"the gated region {rname} ({backend}) of the region map is not a metric",
                )

    # Algorithm-specific: the harness's names.
    if algorithm == "sssp":
        for dname, d in datasets.items():
            if need(
                dname in perf_ab.HOPS, f"dataset {dname}: perf_ab.py knows {sorted(perf_ab.HOPS)}"
            ):
                need(
                    d.get("hops") == perf_ab.HOPS[dname],
                    f"dataset {dname}: hops must be {perf_ab.HOPS[dname]}",
                )
        for bname, b in batches.items():
            if need(
                bname in perf_ab.BATCHES,
                f"batch {bname}: perf_ab.py knows {sorted(perf_ab.BATCHES)}",
            ):
                need(
                    b.get("directory") == perf_ab.BATCHES[bname],
                    f"batch {bname}: directory must be {perf_ab.BATCHES[bname]}",
                )
        runs = suite.get("runs") or {}
        need(isinstance(runs.get("default"), int), "runs.default missing")
    else:
        cases = {
            case_name(c, batches)
            for c in suite.get("cases", [])
            if c.get("task") == "count" or c.get("batch") in batches
        }
        need(
            len(cases) == len(suite.get("cases", [])),
            "cases: a case names an unknown batch or repeats",
        )
        for c in suite.get("cases", []):
            need(c.get("dataset") in datasets, f"case {c}: unknown dataset")

    # Readings.
    names = set()
    for r in suite["readings"]:
        rname = r.get("name")
        if not need(
            isinstance(rname, str) and rname not in names,
            f"reading name {rname!r} missing or repeated",
        ):
            continue
        names.add(rname)
        need(r.get("kind") in KINDS, f"reading {rname}: kind must be one of {KINDS}")
        need(
            r.get("backend") in backends,
            f"reading {rname}: backend {r.get('backend')!r} is not in backends",
        )
        need(isinstance(r.get("gated"), bool), f"reading {rname}: gated must be true or false")
        if r.get("kind") == "run" and r.get("backend") == "cuda":
            need(r.get("clocks") in CLOCKS, f"reading {rname}: a CUDA run needs clocks in {CLOCKS}")
            if r.get("clocks") == "none":
                need(
                    r.get("gated") is False,
                    f"reading {rname}: a default-clock reading is not gated (ADR 0018)",
                )
        if r.get("kind") == "memory":
            need(r.get("backend") == "cuda", f"reading {rname}: memory readings are CUDA")
        runs = r.get("runs")
        need(
            runs is None or (isinstance(runs, int) and runs >= 5),
            f"reading {rname}: runs must be >= 5",
        )
        if algorithm == "cycle_count":
            known = harness_cases(r.get("backend", "openmp"))
            listed = r.get("cases") or []
            need(bool(listed), f"reading {rname}: needs cases")
            for c in listed:
                need(c in cases, f"reading {rname}: case {c} is not in the suite's cases")
                need(
                    c in known,
                    f"reading {rname}: case {c} is not a case of the harness ({r.get('backend')})",
                )
            if r.get("kind") == "run":
                need(isinstance(runs, int), f"reading {rname}: needs runs")
    return errors


# --- The plan ------------------------------------------------------------------------------------


def runs_for(suite: dict, reading: dict, dataset: str | None, override: int | None) -> int:
    if override is not None:
        return override
    if reading.get("runs") is not None:
        return int(reading["runs"])
    runs = suite.get("runs") or {}
    return int(runs.get(dataset, runs.get("default", 21)))


def record_name(suite: dict, reading: dict, dataset: str | None) -> str:
    parts = [suite["suite"], reading["name"]] + ([dataset] if dataset else [])
    return "-".join(parts) + ".json"


def plan(
    suite: dict,
    *,
    records: Path,
    build_root: Path,
    readings: list[str] | None = None,
    datasets: list[str] | None = None,
    runs: int | None = None,
    lock_timeout: float = 4 * 3600.0,
) -> list[dict]:
    """The harness commands of one execution of the suite."""
    algorithm = suite["algorithm"]
    selected = [r for r in suite["readings"] if not readings or r["name"] in readings]
    if readings:
        unknown = set(readings) - {r["name"] for r in suite["readings"]}
        if unknown:
            raise SuiteError(f"unknown readings {sorted(unknown)}")
    names = [d["name"] for d in suite["datasets"]]
    if datasets:
        unknown = set(datasets) - set(names)
        if unknown:
            raise SuiteError(f"unknown datasets {sorted(unknown)}")
    py = sys.executable
    harness = str(REPO / "parity" / "perf_ab.py")
    jobs = []
    for r in selected:
        backend = r["backend"]
        exe = build_root / PRESETS[backend] / "tools" / "compat" / EXECUTABLES[algorithm]
        if algorithm == "sssp":
            for dataset in names:
                if datasets and dataset not in datasets:
                    continue
                out = records / record_name(suite, r, dataset)
                if r["kind"] == "run":
                    argv = [
                        py,
                        harness,
                        "run",
                        "--backend",
                        backend,
                        "--exe",
                        str(exe),
                        "--graph",
                        dataset,
                    ]
                    argv += ["--runs", str(runs_for(suite, r, dataset, runs))]
                    if backend == "cuda":
                        argv += [
                            "--gpu",
                            str(r.get("gpu", suite["backends"]["cuda"].get("gpu", 0))),
                        ]
                        argv += ["--lock-clocks", r["clocks"]]
                    argv += ["--threads", str(suite["backends"]["openmp"].get("threads", 28))]
                else:
                    argv = [
                        py,
                        harness,
                        "memory",
                        "--backend",
                        "cuda",
                        "--exe",
                        str(exe),
                        "--graph",
                        dataset,
                    ]
                    argv += ["--gpu", str(r.get("gpu", 1))]
                argv += ["--lock-timeout", str(int(lock_timeout)), "--json", str(out)]
                jobs.append(
                    {
                        "reading": r["name"],
                        "dataset": dataset,
                        "kind": r["kind"],
                        "gated": r["gated"],
                        "record": out,
                        "argv": argv,
                        "lock": "exclusive (the harness)",
                    }
                )
        else:
            cases = list(r["cases"])
            if datasets:
                cases = [
                    c
                    for c in cases
                    if CASE_NAME.match(c) and CASE_NAME.match(c)["dataset"] in datasets
                ]
                if not cases:
                    continue
            out = records / record_name(suite, r, None)
            if r["kind"] == "run":
                argv = [py, harness, "cycle_count", "run", "--backend", backend, "--exe", str(exe)]
                argv += ["--cases", ",".join(cases), "--runs", str(runs_for(suite, r, None, runs))]
                if backend == "cuda":
                    argv += ["--gpu", str(r.get("gpu", suite["backends"]["cuda"].get("gpu", 0)))]
                    argv += ["--lock-clocks", r["clocks"]]
                    argv += [
                        "--scopes",
                        ",".join(suite["backends"]["cuda"].get("scopes", ["original", "resident"])),
                    ]
                else:
                    argv += ["--threads", str(suite["backends"]["openmp"].get("threads", 56))]
                argv += ["--lock-timeout", str(int(lock_timeout)), "--json", str(out)]
                lock = "exclusive (the harness)"
            else:
                argv = [
                    py,
                    harness,
                    "cycle_count",
                    "memory",
                    "--exe",
                    str(exe),
                    "--cases",
                    ",".join(cases),
                ]
                argv += ["--gpu", str(r.get("gpu", 1)), "--json", str(out)]
                lock = "shared"
            jobs.append(
                {
                    "reading": r["name"],
                    "dataset": None,
                    "kind": r["kind"],
                    "gated": r["gated"],
                    "record": out,
                    "argv": argv,
                    "lock": lock,
                }
            )
    return jobs


# --- Inputs --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_inputs(suite: dict, dataset: dict) -> dict[str, str]:
    digest = dataset["sha256"]
    if isinstance(digest, dict):
        return dict(digest)
    return {dataset.get("file", dataset["name"]): digest}


def input_root(suite: dict, dataset: dict) -> Path:
    root = suite["harness"].get("inputs", "")
    root = root.replace("$DYNG_SCRATCH", str(SCRATCH)).replace("{dataset}", dataset["name"])
    return Path(root)


def verify_inputs(suite: dict, datasets: list[str] | None = None) -> dict:
    """SHA-256 of every input file the suite names; raise on a mismatch or a missing file."""
    report = {}
    for d in suite["datasets"]:
        if datasets and d["name"] not in datasets:
            continue
        root = input_root(suite, d)
        for rel, want in expected_inputs(suite, d).items():
            path = root / rel
            if not path.is_file():
                raise SuiteError(f"{d['name']}: missing input {path} (see the dataset's recipe)")
            got = sha256_file(path)
            if got != want:
                raise SuiteError(f"{d['name']}: {rel} has SHA-256 {got}, the suite says {want}")
        report[d["name"]] = "all digests equal"
    return report


# --- Running -------------------------------------------------------------------------------------


def lock_held_by_ancestor() -> bool:
    import perf_ab

    return bool(perf_ab.lock_holders(SCRATCH / "perf.lock") & perf_ab.ancestors())


def run_jobs(jobs: list[dict], *, skip_existing: bool, log) -> list[dict]:
    done = []
    for job in jobs:
        record = Path(job["record"])
        if skip_existing and record.is_file():
            print(f"[bench_suite] {record.name}: exists, skipped", flush=True)
            job["rc"] = 0
            job["skipped"] = True
            done.append(job)
            continue
        argv = list(job["argv"])
        if job["lock"] == "shared" and not lock_held_by_ancestor():
            argv = ["flock", "-s", str(SCRATCH / "perf.lock"), "nice", "-n", "10", *argv]
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"== {stamp} {shlex.join(argv)}", flush=True)
        log.write(f"== {stamp} {shlex.join(argv)}\n")
        log.flush()
        record.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(argv, cwd=REPO)
        job["rc"] = proc.returncode
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        log.write(f"rc={proc.returncode} {stamp}\n")
        log.flush()
        done.append(job)
    return done


# --- Summary -------------------------------------------------------------------------------------


def compact(record: dict, full: str) -> dict:
    out = copy.deepcopy(record)
    for result in out.get("results", {}).values():
        monitor = result.get("monitor") if isinstance(result, dict) else None
        if not isinstance(monitor, dict):
            continue
        monitor.pop("rounds", None)
        for rejected in monitor.get("rejected", []) or []:
            if isinstance(rejected, dict):
                rejected.pop("windows", None)
    if any(isinstance(r, dict) and "monitor" in r for r in out.get("results", {}).values()):
        out["compacted"] = COMPACTED.format(full=full)
    return out


def expected_gate(tol: dict, kind: str, original_ms: float) -> float | None:
    if kind == "compute":
        return tol["compute_short"] if original_ms < tol["short_region_ms"] else tol["compute"]
    if kind == "end_to_end":
        return tol["end_to_end"]
    return None


def clock_control(record: dict) -> str | None:
    protocol = record.get("protocol", {})
    clocks = protocol.get("gpu_clocks") or protocol.get("clocks") or {}
    return clocks.get("control") if isinstance(clocks, dict) else None


def strip_dirty(commit: str) -> str:
    return commit.split("+", 1)[0]


def read_record(suite: dict, job: dict, tol: dict, problems: list[str]) -> dict:
    """One reading's regions with their verdicts, checked against the suite."""
    path = Path(job["record"])
    reading = next(r for r in suite["readings"] if r["name"] == job["reading"])
    where = path.name
    if not path.is_file():
        problems.append(f"{where}: missing (the command did not write its record)")
        return {
            "reading": job["reading"],
            "dataset": job["dataset"],
            "record": None,
            "complete": False,
        }
    record = json.loads(path.read_text())
    backend = reading["backend"]
    base_name = suite["backends"][backend]["baseline"]
    base = suite["baselines"][base_name]
    ref = record.get("reference", {})
    if ref.get("name") != base_name or (ref.get("commit") and ref["commit"] != base["commit"]):
        problems.append(
            f"{where}: reference {ref.get('name')}@{ref.get('commit')} "
            f"is not {base_name}@{base['commit']}"
        )
    if (
        ref.get("variant", "unpatched") != "unpatched"
        or "/patched/" in str(ref.get("binary", ""))
        or (record.get("baseline") or {}).get("experiment")
    ):
        problems.append(f"{where}: not the unpatched original (an experiment)")
    port = record.get("port", {})
    build = port.get("build") or {}
    if job["kind"] == "run" and build.get("parity_preset") is False:
        problems.append(f"{where}: the port was not a parity-preset build")
    if job["kind"] == "run" and backend == "cuda" and clock_control(record) != reading["clocks"]:
        problems.append(
            f"{where}: clocks {clock_control(record)!r}, the reading says {reading['clocks']!r}"
        )
    # Inputs: the record's digests must be the suite's.
    if suite["algorithm"] == "sssp" and job["kind"] == "run":
        dataset = next(d for d in suite["datasets"] if d["name"] == job["dataset"])
        got = {}
        for line in record.get("inputs_sha256", []):
            sha, rel = line.split(None, 1)
            got[rel.strip()] = sha
        if got != expected_inputs(suite, dataset):
            problems.append(f"{where}: the inputs' SHA-256 differ from the suite's")
    elif suite["algorithm"] == "cycle_count" and job["kind"] == "run":
        for name, sha in (record.get("datasets_sha256") or {}).items():
            want = next((d["sha256"] for d in suite["datasets"] if d["name"] == name), None)
            if want != sha:
                problems.append(f"{where}: dataset {name} SHA-256 {sha} is not the suite's {want}")
    regions = []
    complete = True
    if job["kind"] == "run":
        for case, result in record.get("results", {}).items():
            if result.get("complete") is False:
                complete = False
                problems.append(f"{where}: {case} is incomplete (more rejected rounds than runs)")
            for reg in result.get("regions", []):
                row = {
                    "case": case
                    if suite["algorithm"] == "cycle_count"
                    else f"{job['dataset']}/{case}",
                    "region": reg["region"],
                    "gate_kind": reg.get("gate_kind", "none"),
                    "original_ms": reg.get("original_ms"),
                    "port_ms": reg.get("port_ms"),
                    "ratio": reg.get("ratio"),
                }
                if reg.get("reading"):
                    row["clock_reading"] = reg["reading"]
                if row["gate_kind"] in ("compute", "end_to_end") and "gate" in reg:
                    want = expected_gate(tol, row["gate_kind"], reg["original_ms"])
                    if want is None or abs(reg["gate"] - want) > 1e-9:
                        problems.append(
                            f"{where}: {case} {reg['region']}: the harness gated at "
                            f"{reg['gate']}, the suite says {want}"
                        )
                    row["gate"] = reg["gate"]
                    row["within_gate"] = bool(reg["ratio"] <= reg["gate"])
                    if row["within_gate"] != bool(reg.get("within_gate")):
                        problems.append(
                            f"{where}: {case} {reg['region']}: the harness's verdict "
                            "disagrees with ratio <= gate"
                        )
                    row["provisional"] = bool(reg.get("provisional"))
                    row["noisy"] = bool(reg.get("noisy"))
                regions.append(row)
    else:  # memory
        for case, result in record.get("results", {}).items():
            original = result.get("original", {}).get("peak_live_mib")
            sides = {k: v for k, v in result.items() if k.startswith("port")}
            for side, value in sides.items():
                ratio = value.get("ratio", result.get("ratio")) if isinstance(value, dict) else None
                port_mib = value.get("peak_live_mib") if isinstance(value, dict) else None
                if ratio is None and original and port_mib is not None:
                    ratio = port_mib / original
                label = case if suite["algorithm"] == "cycle_count" else f"{job['dataset']}/{case}"
                regions.append(
                    {
                        "case": label,
                        "region": "device_memory" + side[len("port") :],
                        "gate_kind": "memory",
                        "original_mib": original,
                        "port_mib": port_mib,
                        "ratio": ratio,
                        "gate": tol["memory"],
                        "within_gate": ratio is not None and ratio <= tol["memory"],
                        "provisional": False,
                        "noisy": False,
                    }
                )
    return {
        "reading": job["reading"],
        "dataset": job["dataset"],
        "kind": job["kind"],
        "backend": backend,
        "clocks": reading.get("clocks"),
        "gated": reading["gated"],
        "record": path.name,
        "date": record.get("date"),
        "port_commit": port.get("commit"),
        "reference": {
            "name": ref.get("name"),
            "commit": ref.get("commit") or base["commit"],
            "variant": ref.get("variant", "unpatched"),
        },
        "protocol": {
            k: v
            for k, v in (record.get("protocol") or {}).items()
            if k in ("runs", "threads", "env", "lock", "statistic", "order", "scopes")
        },
        "host": record.get("host"),
        "complete": complete,
        "regions": regions,
        "_raw": record,
    }


def summarize(
    suite: dict, jobs: list[dict], *, out: Path, version: str, records: Path, inputs: dict | None
) -> dict:
    tol = suite["tolerance"]
    problems: list[str] = []
    readings = []
    out.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if job.get("rc", 0) != 0:
            problems.append(f"{Path(job['record']).name}: the harness exited {job['rc']}")
        entry = read_record(suite, job, tol, problems)
        raw = entry.pop("_raw", None)
        if raw is not None:
            full = (
                f"$DYNG_SCRATCH/{Path(job['record']).resolve().relative_to(SCRATCH.resolve())}"
                if Path(job["record"]).resolve().is_relative_to(SCRATCH.resolve())
                else str(job["record"])
            )
            (out / Path(job["record"]).name).write_text(
                json.dumps(compact(raw, full), indent=1) + "\n"
            )
        readings.append(entry)
    gated = [
        (r, g) for r in readings if r.get("gated") for g in r.get("regions", []) if "gate" in g
    ]
    exceeded = [
        f"{r['reading']}: {g['case']} {g['region']} {g['ratio']:.3f} > {g['gate']}"
        for r, g in gated
        if not g["within_gate"]
    ]
    provisional = [
        f"{r['reading']}: {g['case']} {g['region']}" for r, g in gated if g.get("provisional")
    ]
    missing = [
        f"{r['reading']}{'/' + r['dataset'] if r['dataset'] else ''}"
        for r in readings
        if r.get("gated") and (r.get("record") is None or not r.get("complete"))
    ]
    commits = sorted({r["port_commit"] for r in readings if r.get("port_commit")})
    ratios = [g["ratio"] for _, g in gated]
    summary = {
        "schema": SCHEMA,
        "suite": suite["suite"],
        "algorithm": suite["algorithm"],
        "title": suite["title"],
        "suite_file": str(Path(suite["_path"]).resolve().relative_to(REPO))
        if Path(suite["_path"]).resolve().is_relative_to(REPO)
        else suite["_path"],
        "suite_sha256": sha256_file(Path(suite["_path"])),
        "version": version,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "port_commits": commits,
        "baselines": suite["baselines"],
        "tolerance": tol,
        "inputs": inputs or "not verified by this summary (bench_suite.py run verifies them)",
        "readings": readings,
        "verdict": {
            "gated_regions": len(gated),
            "within_gate": len(gated) - len(exceeded),
            "ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "exceeded": exceeded,
            "provisional": provisional,
            "missing_or_incomplete": missing,
            "problems": problems,
            "passed": not (exceeded or provisional or missing or problems),
        },
    }
    (out / f"{suite['suite']}.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def print_summary(summary: dict) -> None:
    v = summary["verdict"]
    rng = v["ratio_range"]
    print(
        f"{summary['suite']}: {v['within_gate']} / {v['gated_regions']} gated regions "
        "within their gates"
        + (f" (ratios {rng[0]:.3f}-{rng[1]:.3f})" if rng else "")
        + (" PASSED" if v["passed"] else " FAILED")
    )
    for key in ("exceeded", "provisional", "missing_or_incomplete", "problems"):
        for item in v[key]:
            print(f"  {key}: {item}")


# --- Command line --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="check suite files")
    v.add_argument("suites", nargs="+", type=Path)
    for name in ("plan", "run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("suite", type=Path)
        p.add_argument(
            "--version", default=None, help="the results directory name (default: VERSION)"
        )
        p.add_argument("--readings", type=lambda s: [x for x in s.split(",") if x], default=None)
        p.add_argument("--datasets", type=lambda s: [x for x in s.split(",") if x], default=None)
        p.add_argument(
            "--runs", type=int, default=None, help="override every reading's runs (>= 5)"
        )
        p.add_argument(
            "--build-root",
            type=Path,
            default=REPO / "build",
            help="the parity presets' parent (default build/)",
        )
        p.add_argument(
            "--records",
            type=Path,
            default=None,
            help="full records (default $DYNG_SCRATCH/runs/bench/<version>)",
        )
        p.add_argument(
            "--out", type=Path, default=None, help="default benchmarks/results/<version>"
        )
        p.add_argument("--lock-timeout", type=float, default=4 * 3600.0, metavar="SECONDS")
        if name == "run":
            p.add_argument(
                "--skip-existing", action="store_true", help="keep records that exist already"
            )
            p.add_argument("--no-verify-inputs", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        failed = False
        for path in args.suites:
            errors = validate(load_suite(path))
            for e in errors:
                print(e)
            print(f"{path}: {'INVALID' if errors else 'valid'}")
            failed |= bool(errors)
        return 1 if failed else 0

    suite = load_suite(args.suite)
    errors = validate(suite)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    if args.runs is not None and args.runs < 5:
        parser.error("--runs must be >= 5 (PLAN Section 6.3 step 7)")
    version = args.version or (REPO / "VERSION").read_text().strip()
    records = args.records or SCRATCH / "runs" / "bench" / version
    out = args.out or REPO / "benchmarks" / "results" / version
    jobs = plan(
        suite,
        records=records,
        build_root=args.build_root.resolve(),
        readings=args.readings,
        datasets=args.datasets,
        runs=args.runs,
        lock_timeout=args.lock_timeout,
    )
    if args.command == "plan":
        for job in jobs:
            print(
                f"# {job['reading']}{' ' + job['dataset'] if job['dataset'] else ''} "
                f"({'gated' if job['gated'] else 'recorded, not gated'}; lock: {job['lock']})"
            )
            print(shlex.join(job["argv"]))
        return 0
    inputs = None
    if args.command == "run":
        if not args.no_verify_inputs:
            inputs = verify_inputs(suite, args.datasets)
            print(f"[bench_suite] inputs: {inputs}", flush=True)
        records.mkdir(parents=True, exist_ok=True)
        with open(records / f"{suite['suite']}.log", "a") as log:
            jobs = run_jobs(jobs, skip_existing=args.skip_existing, log=log)
    summary = summarize(suite, jobs, out=out, version=version, records=records, inputs=inputs)
    print_summary(summary)
    return 0 if summary["verdict"]["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SuiteError as exc:
        print(f"bench_suite.py: {exc}", file=sys.stderr)
        sys.exit(2)
