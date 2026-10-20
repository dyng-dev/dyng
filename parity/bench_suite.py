#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Run a benchmark suite of benchmarks/paper/ through the A/B harness (PLAN Sections 8.5, 8.6).

    parity/bench_suite.py validate benchmarks/paper/*.yaml
    parity/bench_suite.py plan     benchmarks/paper/ieee_tc_dyntrucy.yaml [--readings cuda]
    parity/bench_suite.py run      benchmarks/paper/ieee_tc_dyntrucy.yaml --version 0.1.0rc1
                                   [--readings openmp,cuda] [--datasets ...] [--runs N]
                                   [--batches safe50k]
                                   [--build-root build] [--skip-existing | --force]
                                   [--no-verify-inputs] [--records DIR] [--out DIR]
    parity/bench_suite.py summarize benchmarks/paper/ieee_tc_dyntrucy.yaml --version 0.1.0rc1

A suite file (docs/developer/benchmarks.md) is machine-readable: datasets with their SHA-256
digests and the recipe that makes them, batches with their seeds, backends, the number of runs,
the metrics as regions of parity/timed_regions/<algorithm>.toml with the dynG profiler stages they
sum, the baselines (the pinned originals of parity/references.toml, unpatched copies), the
tolerances of PLAN 8.6, and the readings: one reading is one harness invocation per dataset (sssp,
mosp) or per case list (cycle_count), gated or recorded only (the default-clock readings of ADR
0018).

validate   checks a suite file against itself and against the harness it drives: the region map
           (every metric is a region with the same gate and the same profiler stages), the
           pinned commits (parity/references.toml), the harness's case lists and batch names, and
           the tolerances the harness applies. Exit 1 on any finding.
plan       prints the harness commands one execution of the suite runs (nothing is run).
run        verifies the inputs against the suite's digests (SHA-256 of every file; skip with
           --no-verify-inputs), then runs every planned command: parity/perf_ab.py run|memory for
           sssp, parity/perf_ab.py mosp and memory --mosp for mosp, parity/perf_ab.py cycle_count
           run|memory for cycle_count, parity/perf_ab.py triad_count run|memory for triad_count
           (cases <dataset>_b<size>_s<seed> of the golden set triad_count_escher_gpu; a
           synthetic dataset is pinned by its generator parameters and the SHA-256 of its rows
           per seed). Each timing command
           takes the exclusive perf lock $DYNG_SCRATCH/perf.lock itself and runs only on a
           parity-preset build (--build-root/<preset>/tools/compat/...); the CUDA readings lock
           the GPU clocks for the whole A/B (--lock-clocks of the reading: boost, base, or none
           for the recorded default-clock reading). The cycle_count memory reading takes the
           shared lock (it times nothing). The full records go to --records (default
           $DYNG_SCRATCH/runs/bench/<version>), never over existing ones (--skip-existing keeps
           them, --force replaces them); then `summarize` runs.
summarize  reads the records of every planned command and writes, under
           benchmarks/results/<version>/ (--out), a compacted copy of each record (the per-round
           machine-monitor windows dropped; verdicts, samples, rejected rounds and clock checks
           kept) and <suite>.json: the readings with every region's medians, ratio, gate and
           verdict, the NVIDIA driver each record names, the inputs check and the overall
           verdict. It re-derives every gate from the suite's tolerances and refuses a record
           whose gate, reference commit, inputs, clock lock or build preset differs from the
           suite (a mosp record must also be of mosp, with the suite's K, batches and CUDA
           engine, every output written and the invalidated counts equal in every round). The
           inputs of every record must be verified: by the digests the harness took
           while measuring, or, for a record without them, by hashing the files again, each
           unchanged since the execution started (the first command of <records>/<suite>.log).
           Exit 1 if a gated region exceeds its gate, a gated reading is missing or incomplete,
           or a check fails.

--readings, --datasets, --batches (sssp, mosp) and --runs narrow the suite, and
--keep-contaminated (sssp, mosp: a diagnostic execution on a busy machine, whose rounds with
foreign load are kept and flagged instead of repeated) makes it a diagnostic one: such an
execution is written as
<suite>.partial.json and only to an --out outside benchmarks/results/ (a release's summary covers
the whole suite, and parity/certify.py checks that it does).
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
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
sys.path.insert(0, str(REPO / "parity"))

SCHEMA = 1
TOOLS = {
    "sssp": "perf_ab.sssp",
    "mosp": "perf_ab.mosp",
    "cycle_count": "perf_ab.cycle_count",
    "triad_count": "perf_ab.triad_count",  # parity/triad_perf.py (M9a step 2)
}
REGION_MAPS = {
    "sssp": "parity/timed_regions/sssp.toml",
    "mosp": "parity/timed_regions/mosp.toml",
    "cycle_count": "parity/timed_regions/cycle_count.toml",
    "triad_count": "parity/timed_regions/triad_count.toml",
}
# The region map's section per backend.
MAP_KEYS = {
    "sssp": {"openmp": "mosp_openmp", "cuda": "mosp_cuda"},
    "mosp": {"openmp": "mosp_openmp", "cuda": "mosp_cuda"},
    "cycle_count": {"openmp": "cycle_enum_openmp", "cuda": "cycle_enum_cuda"},
    "triad_count": {"cuda": "escher_gpu"},
}
EXECUTABLES = {
    "sssp": "dyng-compat-mosp",
    "mosp": "dyng-compat-mosp",
    "cycle_count": "dyng-compat-cycle-enum",
    "triad_count": "dyng-compat-escher",  # `dyng-compat-escher run` (M9a step 2)
}
# The algorithms whose harness runs one command per dataset over the MOSP bench inputs
# ($DYNG_SCRATCH/bench/mosp/<graph>, parity/perf_ab.py prepare): sssp (`perf_ab.py run|memory`)
# and mosp (`perf_ab.py mosp`, `perf_ab.py memory --mosp`, M7).
PER_DATASET = ("sssp", "mosp")
CUDA_ENGINES = ("automatic", "fused", "operators")  # perf_ab.py mosp --cuda-engine
PRESETS = {"openmp": "parity", "cuda": "parity-cuda"}
CLOCKS = ("boost", "base", "none")
KINDS = ("run", "memory")
TOLERANCE_KEYS = ("compute", "compute_short", "short_region_ms", "end_to_end", "memory")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
CASE_NAME = re.compile(
    r"^(?P<dataset>[A-Za-z0-9-]+)_k(?P<k>\d+)(?:_(?P<d>\d+)_(?P<i>\d+)_s(?P<s>\d+))?$"
)
# triad_count's cases (M9a): a group of the golden set triad_count_escher_gpu,
# <dataset>_b<size label>_s<seed> (coauth_b50k_s1); a record keys them <case>/<pattern>.
TRIAD_CASE = re.compile(r"^(?P<dataset>[A-Za-z0-9]+)_b(?P<size>\d+k?)_s(?P<seed>\d+)$")
TRIAD_PATTERNS = ("main", "type1", "type2", "type3")
# The memory metrics of an algorithm's suites (the gate "memory"): the Nsight Systems peak, and
# for triad_count also the store accounting (the author's decision of 2026-10-07).
MEMORY_METRICS = {"triad_count": ("device_memory", "store_memory")}
COMPACTED = (
    "the per-round machine-monitor windows (results.*.monitor.rounds, and the windows of the "
    "rejected rounds) and triad_count's per-round stage timers (results.*.original_rounds and "
    "port_rounds; every row's samples stay in its region) were dropped to keep the repository "
    "small; the verdicts, samples, rejected rounds with their reasons, clock check and GPU "
    "summary are unchanged; the full record: {full}"
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
    """The regions of the algorithm's map for a backend by gate row: a region read once per batch
    (per_batch = true, triad_count.toml) contributes one entry per row ("update[b1]", ...), each
    with the region's gate and stages, instead of its own name."""
    import perf_ab

    doc = tomllib.loads((REPO / REGION_MAPS[algorithm]).read_text())
    out: dict[str, dict] = {}
    for r in doc["reference"][MAP_KEYS[algorithm][backend]]["region"]:
        perf_ab.check_batch_rows(REPO / REGION_MAPS[algorithm], r)
        for row in perf_ab.region_rows(r):
            out[row] = r if row == r["name"] else dict(r, name=row, row_of=r["name"])
    return out


def metric_gate(metric: dict, backend: str) -> str:
    gate = metric.get("gate")
    return gate.get(backend) if isinstance(gate, dict) else gate


def triad_goldens():
    import triad_goldens

    return triad_goldens


def triad_case_dataset(key: str) -> str | None:
    """The dataset of a triad_count case or record key (coauth_b50k_s1[/main])."""
    m = TRIAD_CASE.match(key.split("/", 1)[0])
    return m["dataset"] if m else None


def triad_digest(suite: dict, case: str) -> str | None:
    """The suite's SHA-256 of a triad_count case's input: the dataset file's, or a synthetic
    dataset's rows of the case's seed."""
    m = TRIAD_CASE.match(case)
    if not m:
        return None
    d = next((x for x in suite["datasets"] if x["name"] == m["dataset"]), None)
    if d is None:
        return None
    digest = d.get("sha256")
    return digest.get(f"s{m['seed']}") if isinstance(digest, dict) else digest


def validate_triad(suite: dict, datasets: dict, batches: dict, need) -> None:
    """triad_count's suites: the datasets pinned (coauth-DBLP by its file's SHA-256; a synthetic
    dataset by A's generator parameters and the SHA-256 of its rows per seed, both those of the
    golden set), and every reading's cases groups of the golden set."""
    tg = triad_goldens()
    pinned = {
        name: spec["loaded_rows_sha256"]
        for name, spec in tomllib.loads((REPO / "parity" / "goldens.toml").read_text())["sets"][
            tg.SET
        ]
        .get("inputs", {})
        .items()
    }
    for dname, d in datasets.items():
        if dname == "coauth":
            need(
                d.get("sha256") == tg.COAUTH_SHA256,
                f"dataset {dname}: sha256 must be the pinned file's",
            )
            need(d.get("file") == tg.COAUTH, f"dataset {dname}: file must be {tg.COAUTH}")
        elif need(
            dname in tg.SYNTHETIC, f"dataset {dname}: coauth or one of {sorted(tg.SYNTHETIC)}"
        ):
            gen = d.get("generator") or {}
            need(
                gen.get("parameters") == list(tg.SYNTHETIC[dname]),
                f"dataset {dname}: generator.parameters must be {list(tg.SYNTHETIC[dname])}",
            )
            digest = d.get("sha256")
            if need(isinstance(digest, dict), f"dataset {dname}: sha256 per seed (s1, s2, ...)"):
                for seed_label, sha in digest.items():
                    want = pinned.get(f"{dname}_{seed_label}")
                    need(
                        sha == want,
                        f"dataset {dname}: sha256 {seed_label} is not the golden set's {want}",
                    )
    groups = {g.name for g in tg.all_groups()}
    for r in suite["readings"]:
        rname = r.get("name")
        listed = r.get("cases") or []
        need(bool(listed), f"reading {rname}: needs cases")
        patterns = r.get("patterns", ["main"])
        need(
            bool(patterns) and set(patterns) <= set(TRIAD_PATTERNS),
            f"reading {rname}: patterns must be among {TRIAD_PATTERNS}",
        )
        for c in listed:
            m = TRIAD_CASE.match(c)
            if not need(m is not None and c in groups, f"reading {rname}: case {c} is not a case "
                        f"of the golden set {tg.SET}"):  # fmt: skip
                continue
            need(m["dataset"] in datasets, f"reading {rname}: case {c}: unknown dataset")
            need(
                f"b{m['size']}" in batches, f"reading {rname}: case {c}: unknown batch b{m['size']}"
            )
            need(
                triad_digest(suite, c) is not None,
                f"reading {rname}: case {c}: no digest of its input",
            )
            if r.get("kind") == "run" and r.get("gated") and m["dataset"] == "coauth":
                need(
                    r.get("clocks") == "base",
                    f"reading {rname}: coauth-DBLP is read at the base lock (ADR 0021)",
                )
        if r.get("kind") == "run":
            need(isinstance(r.get("runs"), int), f"reading {rname}: needs runs")
            if r.get("gated"):
                need(patterns == ["main"], f"reading {rname}: the gated readings are h_motif's")
                need(
                    r.get("clocks") in ("base", "boost"),
                    f"reading {rname}: a gated reading is at a clock lock (ADRs 0018, 0021)",
                )


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
            if algorithm == "triad_count":  # the seed is the case's (--seed s<N>)
                seeds = b.get("seeds")
                need(
                    isinstance(seeds, list) and all(isinstance(x, int) for x in seeds),
                    f"batch {bname}: needs the integer seeds of its cases",
                )
            else:
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
                allowed = MEMORY_METRICS.get(algorithm, ("device_memory",))
                need(
                    mname in allowed,
                    f"metric {mname}: only {' and '.join(allowed)} have the memory gate",
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

    # Every gated region of the map is a metric of the suite that times it (a gated run reading
    # on the backend; tkde26_escher_plus gates memory only).
    for backend in backends:
        if backend not in PRESETS:
            continue
        if not any(
            r.get("kind") == "run" and r.get("gated") and r.get("backend") == backend
            for r in suite["readings"]
        ):
            continue
        for rname, region in region_map(algorithm, backend).items():
            if region["gate"] != "none":
                need(
                    backend in suite["metrics"].get(rname, {}),
                    f"the gated region {rname} ({backend}) of the region map is not a metric",
                )

    # Algorithm-specific: the harness's names.
    if algorithm in PER_DATASET:
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
        if algorithm == "mosp":
            # perf_ab.py mosp runs every column of the prepared CSR (no -k) with the default
            # preferences (no --pref): the record's `objectives` must be the suite's K.
            graph = suite.get("graph") or {}
            need(
                isinstance(graph.get("num_weights"), int) and graph["num_weights"] >= 1,
                "graph.num_weights (K) missing",
            )
            need(
                graph.get("preferences", "default") == "default",
                "graph.preferences: the harness runs the default preferences (all 1)",
            )
            engine = (backends.get("cuda") or {}).get("engine", "automatic")
            need(
                engine in CUDA_ENGINES,
                f"backends.cuda.engine must be one of {CUDA_ENGINES} (perf_ab.py --cuda-engine)",
            )
    elif algorithm == "triad_count":
        validate_triad(suite, datasets, batches, need)
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
    batches: list[str] | None = None,
    runs: int | None = None,
    lock_timeout: float = 4 * 3600.0,
    keep_contaminated: bool = False,
) -> list[dict]:
    """The harness commands of one execution of the suite. keep_contaminated (sssp, mosp: a
    diagnostic execution on a busy machine) keeps the rounds the contamination monitor would
    repeat, flagged, instead of repeating them."""
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
    batch_names = [b["name"] for b in suite["batches"]]
    if batches:
        if algorithm not in PER_DATASET:
            raise SuiteError(f"--batches narrows the suites of {PER_DATASET}, not {algorithm}")
        unknown = set(batches) - set(batch_names)
        if unknown:
            raise SuiteError(f"unknown batches {sorted(unknown)}")
        batch_names = [b for b in batch_names if b in batches]
    if keep_contaminated and algorithm not in PER_DATASET:
        raise SuiteError(f"--keep-contaminated is for the suites of {PER_DATASET}")
    py = sys.executable
    harness = str(REPO / "parity" / "perf_ab.py")
    jobs = []
    for r in selected:
        backend = r["backend"]
        exe = build_root / PRESETS[backend] / "tools" / "compat" / EXECUTABLES[algorithm]
        if algorithm in PER_DATASET:
            for dataset in names:
                if datasets and dataset not in datasets:
                    continue
                out = records / record_name(suite, r, dataset)
                if r["kind"] == "run":
                    command = "mosp" if algorithm == "mosp" else "run"
                    argv = [
                        py,
                        harness,
                        command,
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
                        if algorithm == "mosp":
                            engine = suite["backends"]["cuda"].get("engine", "automatic")
                            argv += ["--cuda-engine", engine]
                    argv += ["--threads", str(suite["backends"]["openmp"].get("threads", 28))]
                    if keep_contaminated:
                        argv.append("--keep-contaminated")
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
                    if algorithm == "mosp":
                        argv.append("--mosp")  # the whole MOSP update, the original's scope
                    argv += ["--gpu", str(r.get("gpu", 1))]
                if algorithm == "mosp" or batches:
                    argv += ["--batches", ",".join(batch_names)]
                argv += ["--lock-timeout", str(int(lock_timeout)), "--json", str(out)]
                jobs.append(
                    {
                        "reading": r["name"],
                        "dataset": dataset,
                        "batches": batch_names,
                        "diagnostic": keep_contaminated,
                        "kind": r["kind"],
                        "gated": r["gated"],
                        "record": out,
                        "argv": argv,
                        "lock": "exclusive (the harness)",
                    }
                )
        elif algorithm == "triad_count":
            cases = list(r["cases"])
            if datasets:
                cases = [c for c in cases if triad_case_dataset(c) in datasets]
                if not cases:
                    continue
            patterns = list(r.get("patterns", ["main"]))
            out = records / record_name(suite, r, None)
            build = build_root / PRESETS[backend]
            common = ["--cases", ",".join(cases), "--patterns", ",".join(patterns)]
            common += ["--build", str(build)]
            n = runs_for(suite, r, None, runs)
            if r["kind"] == "run":
                gpu = r.get("gpu", suite["backends"]["cuda"].get("gpu", 0))
                argv = [py, harness, "triad_count", "run", *common, "--rounds", str(n)]
                argv += ["--gpu", str(gpu), "--lock-clocks", r["clocks"]]
                argv += ["--lock-timeout", str(int(lock_timeout)), "--json", str(out)]
                lock = "exclusive (the harness)"
            else:
                argv = [py, harness, "triad_count", "memory", *common]
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
                    "cases": cases,
                    "patterns": patterns,
                    "runs": n if r["kind"] == "run" else None,
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
        if suite["algorithm"] == "triad_count" and isinstance(d.get("sha256"), dict):
            # A synthetic input of ESCHER-GPU: both sides generate it in-process; the golden
            # set stores its rows per seed (M9a step 1), which must have the suite's digests.
            tg = triad_goldens()
            for seed_label, want in d["sha256"].items():
                stored = tg.set_dir(SCRATCH) / "inputs" / f"{d['name']}_{seed_label}.hg.zst"
                if not stored.is_file():
                    raise SuiteError(f"{d['name']}: missing {stored} (the golden set's rows)")
                got = hashlib.sha256(tg.zstd_read(stored)).hexdigest()
                if got != want:
                    raise SuiteError(
                        f"{d['name']}: the rows of {seed_label} have SHA-256 {got}, the suite "
                        f"says {want}"
                    )
            report[d["name"]] = "the golden set's stored rows of every seed have the digests"
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


def record_datasets(suite: dict, job: dict, record: dict) -> list[str]:
    """The suite datasets a record measured: the job's (sssp, mosp), or those its cases read."""
    if suite["algorithm"] in PER_DATASET:
        return [job["dataset"]]
    if suite["algorithm"] == "triad_count":
        return sorted({d for k in record.get("results", {}) if (d := triad_case_dataset(k))})
    names = set()
    for key in record.get("results", {}):
        m = CASE_NAME.match(key.split("/", 1)[-1])
        if m:
            names.add(m["dataset"])
    return sorted(names)


def parse_stamp(text: str) -> datetime.datetime | None:
    try:
        stamp = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.UTC)).astimezone(datetime.UTC)


def execution_started(records: Path, suite: dict) -> datetime.datetime | None:
    """When the first command of the suite's executions in `records` started (the "== <time>"
    lines `run` writes to <records>/<suite>.log); None without a log."""
    log = records / f"{suite['suite']}.log"
    if not log.is_file():
        return None
    stamps = [
        parse_stamp(line.split()[1])
        for line in log.read_text(errors="replace").splitlines()
        if line.startswith("== ") and len(line.split()) > 1
    ]
    stamps = [s for s in stamps if s is not None]
    return min(stamps) if stamps else None


def unchanged_since(path: Path, want: str, start: datetime.datetime) -> str | None:
    """None if `path` hashes to `want` now and was neither written nor replaced since `start`
    (its modification and status-change times are earlier); else the reason."""
    if not path.is_file():
        return f"{path}: missing"
    st = path.stat()
    changed = datetime.datetime.fromtimestamp(max(st.st_mtime, st.st_ctime), datetime.UTC)
    if changed >= start:
        return f"{path}: changed at {changed:%Y-%m-%dT%H:%M:%SZ}, after {start:%Y-%m-%dT%H:%M:%SZ}"
    got = sha256_file(path)
    if got != want:
        return f"{path}: SHA-256 {got}, the suite says {want}"
    return None


def check_record_inputs(
    suite: dict, job: dict, record: dict, started: datetime.datetime | None
) -> dict:
    """How the inputs a record measured are known to be the suite's: digests the harness took
    while measuring (sssp: `inputs_hashed: run start`; cycle_count: `datasets_sha256`, hashed by
    the measuring process), or, for a record without them, the files hashed again now, each
    unchanged since the execution started (modification and status-change times). Returns
    {verified, how, datasets[, problems]}."""
    names = record_datasets(suite, job, record)
    by_name = {d["name"]: d for d in suite["datasets"]}
    out: dict = {"datasets": names}
    if suite["algorithm"] in PER_DATASET and record.get("inputs_hashed") == "run start":
        got = {}
        for line in record.get("inputs_sha256", []):
            sha, rel = line.split(None, 1)
            got[rel.strip()] = sha
        ok = got == expected_inputs(suite, by_name[job["dataset"]])
        return {**out, "verified": ok, "how": "hashed by the harness when the measurement started"}
    if suite["algorithm"] == "cycle_count":
        digests = record.get("datasets_sha256") or {}
        if names and all(n in digests for n in names):
            ok = all(digests[n] == by_name[n]["sha256"] for n in names)
            return {**out, "verified": ok, "how": "hashed by the measuring process"}
    if suite["algorithm"] == "triad_count":
        # Every case's input digest, taken by the measuring process (triad_perf.input_digest),
        # must be the suite's: the dataset file's, or a synthetic dataset's rows of its seed.
        results = record.get("results") or {}
        problems = [
            f"{key}: input SHA-256 {(r.get('input') or {}).get('sha256')} is not the suite's "
            f"{triad_digest(suite, key.split('/', 1)[0])}"
            for key, r in results.items()
            if (r.get("input") or {}).get("sha256") != triad_digest(suite, key.split("/", 1)[0])
        ]
        # The synthetic inputs are generated by both measured programs: what makes the digest
        # theirs is the guard on the counts every measured process printed (each round of a timed
        # reading; each plain and profiled run of a memory reading).
        unguarded = [
            key for key, r in results.items() if (r.get("guard") or {}).get("passed") is not True
        ]
        problems += [f"{key}: no count guard on its measured processes" for key in unguarded]
        out |= {
            "verified": bool(results) and not problems,
            "how": "hashed by the measuring process (coauth-DBLP's file; a synthetic input's rows "
            "as the golden set stores them, which both sides generate from the suite's "
            "parameters), with the guard requiring the golden's counts from every measured "
            "process (each round of a timed reading; each plain and profiled run of a memory "
            "reading)",
        }
        if problems:
            out["problems"] = problems
        return out
    # A record without digests taken while measuring (sssp records before 0.1.0rc1's review:
    # their digests are the ones `prepare` wrote; memory records): hash the files again.
    start = started or parse_stamp(record.get("date", ""))
    if start is None:
        return {**out, "verified": False, "how": "no digests and no measurement time"}
    problems = []
    for name in names:
        root = input_root(suite, by_name[name])
        for rel, want in expected_inputs(suite, by_name[name]).items():
            reason = unchanged_since(root / rel, want, start)
            if reason:
                problems.append(reason)
    out["how"] = "hashed again by summarize; every file unchanged since " + (
        f"the execution started ({start:%Y-%m-%dT%H:%M:%SZ}, the suite's run log)"
        if started
        else f"the record's date ({start:%Y-%m-%dT%H:%M:%SZ}, no run log)"
    )
    out["verified"] = not problems
    if problems:
        out["problems"] = problems
    return out


# --- Running -------------------------------------------------------------------------------------


def lock_held_by_ancestor() -> bool:
    import perf_ab

    return bool(perf_ab.lock_holders(SCRATCH / "perf.lock") & perf_ab.ancestors())


def run_jobs(jobs: list[dict], *, skip_existing: bool, log, force: bool = False) -> list[dict]:
    existing = [Path(j["record"]).name for j in jobs if Path(j["record"]).is_file()]
    if existing and not (skip_existing or force):
        raise SuiteError(
            f"records exist already ({', '.join(existing)}): --skip-existing keeps them, "
            "--force replaces them (move a record to replace to superseded/ first)"
        )
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
        if isinstance(result, dict):
            result.pop("original_rounds", None)
            result.pop("port_rounds", None)
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


def read_record(
    suite: dict,
    job: dict,
    tol: dict,
    problems: list[str],
    started: datetime.datetime | None = None,
) -> dict:
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
    # The record is of the suite's algorithm: an sssp record (`perf_ab.py run`, `memory` without
    # --mosp) has the shape of a mosp one, so a mosp suite needs the field; the records of the
    # 0.1 suites carry it as well, and a record of another algorithm is refused.
    algorithm = suite["algorithm"]
    named = record.get("algorithm", None if algorithm == "mosp" else algorithm)
    if named != algorithm:
        problems.append(f"{where}: a record of algorithm {named!r}, the suite is {algorithm!r}")
    if algorithm in PER_DATASET:
        # Every batch of the suite, and no other (the harness's record keys are batch names).
        want = job.get("batches") or [b["name"] for b in suite["batches"]]
        got = sorted(record.get("results", {}))
        if got != sorted(want):
            problems.append(f"{where}: batches {got}, the suite's are {sorted(want)}")
    if algorithm == "mosp":
        objectives = record.get("objectives")
        if objectives != (suite.get("graph") or {}).get("num_weights"):
            problems.append(
                f"{where}: {objectives} objectives, the suite's K is "
                f"{(suite.get('graph') or {}).get('num_weights')}"
            )
        if job["kind"] == "run":
            preferences = record.get("preferences")
            if preferences != "default (all 1)":
                problems.append(
                    f"{where}: preferences {preferences!r}, the suite's are the default"
                )
            outputs = (record.get("protocol") or {}).get("outputs", "")
            if not str(outputs).startswith("written by both"):
                problems.append(
                    f"{where}: outputs {outputs!r}: the suite's scope writes every output file "
                    "on both sides (bench/run.sh)"
                )
            if backend == "cuda":
                engine = suite["backends"]["cuda"].get("engine", "automatic")
                if record.get("cuda_engine") != engine:
                    problems.append(
                        f"{where}: CUDA engine {record.get('cuda_engine')!r}, the suite's is "
                        f"{engine!r}"
                    )
    if algorithm == "triad_count":
        check_triad_record(suite, job, record, where, problems)
    port = record.get("port", {})
    build = port.get("build") or {}
    if job["kind"] == "run" and build.get("parity_preset") is False:
        problems.append(f"{where}: the port was not a parity-preset build")
    if job["kind"] == "run" and backend == "cuda" and clock_control(record) != reading["clocks"]:
        problems.append(
            f"{where}: clocks {clock_control(record)!r}, the reading says {reading['clocks']!r}"
        )
    # Inputs: every digest a record carries must be the suite's, and the inputs it measured must
    # be verified (check_record_inputs).
    if suite["algorithm"] in PER_DATASET and record.get("inputs_sha256"):
        dataset = next(d for d in suite["datasets"] if d["name"] == job["dataset"])
        got = {}
        for line in record.get("inputs_sha256", []):
            sha, rel = line.split(None, 1)
            got[rel.strip()] = sha
        if got != expected_inputs(suite, dataset):
            problems.append(f"{where}: the inputs' SHA-256 differ from the suite's")
    elif suite["algorithm"] == "cycle_count":
        for name, sha in (record.get("datasets_sha256") or {}).items():
            want = next((d["sha256"] for d in suite["datasets"] if d["name"] == name), None)
            if want != sha:
                problems.append(f"{where}: dataset {name} SHA-256 {sha} is not the suite's {want}")
    inputs = check_record_inputs(suite, job, record, started)
    if not inputs["verified"]:
        problems.append(
            f"{where}: the inputs are not verified ({inputs['how']}"
            + (f": {'; '.join(inputs['problems'][:3])}" if inputs.get("problems") else "")
            + ")"
        )
    regions = []
    complete = True
    contaminated: dict[str, int] = {}
    if job["kind"] == "run":
        for case, result in record.get("results", {}).items():
            # Flagged, not repeated, by the OpenMP cycle_count harness (the others repeat them).
            flagged = sum(
                int(side.get("flagged_runs", 0))
                for side in (result.get("contamination") or {}).values()
                if isinstance(side, dict)
            )
            # Kept by perf_ab.py --keep-contaminated (sssp, mosp): only in a diagnostic
            # execution (bench_suite.py --keep-contaminated, always partial), where they are
            # flagged; a record that kept them is never a gate reading of the suite.
            monitor = result.get("monitor") if isinstance(result.get("monitor"), dict) else {}
            if "kept and flagged" in str(
                (record.get("protocol") or {}).get("contamination_monitor")
            ):
                kept = sum(
                    1
                    for rnd in monitor.get("rounds", []) or []
                    for side in ("original", "port")
                    if isinstance(rnd, dict) and (rnd.get(side) or {}).get("reasons")
                )
                if not job.get("diagnostic"):
                    problems.append(
                        f"{where}: {case}: contaminated rounds were kept (--keep-contaminated), "
                        "not repeated: not a gate reading"
                    )
                flagged += kept
            if flagged:
                contaminated[case] = flagged
            if result.get("complete") is False:
                complete = False
                problems.append(f"{where}: {case} is incomplete (more rejected rounds than runs)")
            # The deterministic counter (PLAN 8.3; mosp.toml [counters]): the `invalidated` of
            # every objective equal on both sides in every round (a mosp record must say so).
            inv = result.get("invalidated")
            if algorithm in PER_DATASET and (
                (algorithm == "mosp" and not isinstance(inv, dict))
                or (isinstance(inv, dict) and inv.get("equal_in_every_sample") is not True)
            ):
                problems.append(
                    f"{where}: {case}: the invalidated counts are not equal in every round"
                )
            for reg in result.get("regions", []):
                row = {
                    "case": case if job.get("dataset") is None else f"{job['dataset']}/{case}",
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
            original = peak_mib(result.get("original"))
            sides = {k: v for k, v in result.items() if k.startswith("port")}
            for side, value in sides.items():
                ratio = value.get("ratio", result.get("ratio")) if isinstance(value, dict) else None
                port_mib = peak_mib(value)
                if ratio is None and original and port_mib is not None:
                    ratio = port_mib / original
                label = case if job.get("dataset") is None else f"{job['dataset']}/{case}"
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
            # triad_count's store accounting (bytes per point; the gated points only).
            for point in result.get("store_accounting") or []:
                mine, theirs = point.get("dyng_device_bytes"), point.get("escher_gpu_device_bytes")
                if not point.get("gated") or not mine or not theirs:
                    continue
                ratio = mine / theirs
                regions.append(
                    {
                        "case": case,
                        "region": point["row"],
                        "gate_kind": "memory",
                        "original_bytes": theirs,
                        "port_bytes": mine,
                        "ratio": ratio,
                        "gate": tol["memory"],
                        "within_gate": ratio <= tol["memory"],
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
        "driver": record.get("driver"),
        "inputs": inputs,
        "complete": complete,
        "contaminated_runs": contaminated,
        "regions": regions,
        "_raw": record,
    }


def peak_mib(side) -> float | None:
    """A side's peak of live allocations in MiB: exact from its bytes when the record has them
    (triad_count's records), else the MiB it gives (rounded to 0.1 by the older harnesses)."""
    if not isinstance(side, dict):
        return None
    if isinstance(side.get("peak_live_bytes"), int):
        return side["peak_live_bytes"] / 2**20
    return side.get("peak_live_mib")


def triad_reading_gpu(suite: dict, job: dict) -> int:
    """The GPU a triad_count reading runs on (its own, else the backend's; memory: 1)."""
    reading = next((r for r in suite.get("readings", []) if r["name"] == job["reading"]), {})
    if job["kind"] == "run":
        return int(reading.get("gpu", (suite.get("backends") or {}).get("cuda", {}).get("gpu", 0)))
    return int(reading.get("gpu", 1))


def triad_required_rows(suite: dict, job: dict, record: dict) -> dict[str, list[str]]:
    """The rows a triad_count reading must give per case of its record: every batch's update,
    full_count and end_to_end (a timed reading); device_memory and the store accounting's gated
    points, every point on coauth-DBLP and A's two print points (after the build, after the
    last batch) on the synthetic inputs (a memory reading). The batches are the suite's."""
    sizes = {b["name"]: b for b in suite.get("batches", [])}
    out = {}
    for key in record.get("results", {}):
        m = TRIAD_CASE.match(key.split("/", 1)[0])
        spec = sizes.get(f"b{m['size']}") if m else None
        nb = int((spec or {}).get("batches") or 0)
        if job["kind"] == "run":
            out[key] = [f"update[b{k}]" for k in range(1, nb + 1)] + ["full_count", "end_to_end"]
        else:
            every = triad_case_dataset(key) == "coauth"
            points = (
                [f"store_memory[b{k}]" for k in range(1, nb + 1)]
                if every
                else [f"store_memory[b{nb}]"]
            )
            out[key] = ["device_memory", "store_memory[build]", *points]
    return out


def triad_sample_problems(result: dict, need_rounds: int | None) -> list[str]:
    """What is wrong with a timed triad_count case's samples: every row's medians are the medians
    of its samples (original_runs, port_runs), each row has one sample per round, the rounds are
    at least the reading's, and the per-round stage timers (when the record keeps them) have one
    entry per round. The records are not trusted for their medians or their round counts (M9a
    review)."""
    out = []
    n = result.get("rounds")
    if not isinstance(n, int) or n < 1:
        return [f"rounds {n!r} is not a count"]
    if need_rounds and n < need_rounds:
        out.append(f"{n} rounds, the reading needs {need_rounds}")
    for side in ("original_rounds", "port_rounds"):
        if side in result and len(result[side] or []) != n:
            out.append(f"{side}: {len(result[side] or [])} rounds, the record says {n}")
    for reg in result.get("regions", []) or []:
        name = reg.get("region")
        for side in ("original", "port"):
            runs = reg.get(f"{side}_runs")
            if not isinstance(runs, list) or len(runs) != n:
                out.append(f"{name}: {len(runs) if isinstance(runs, list) else 'no'} {side} "
                           f"samples for {n} rounds")  # fmt: skip
                continue
            med = statistics.median(runs)
            got = reg.get(f"{side}_ms")
            if not isinstance(got, int | float) or abs(med - got) > 1e-6 * max(1.0, abs(med)):
                out.append(f"{name}: the {side} median {got} is not its samples' {med}")
    return out


# The protocol of the triad_count readings (the author's: PLAN 8.6, ADRs 0018 / 0021, the one
# layout of 2026-10-08, D1 of 2026-10-15), checked field by field in every record.
TRIAD_NSYS = "nsys profile -t cuda,nvtx --cuda-memory-usage=true"
TRIAD_ENV = "CUDA_MODULE_LOADING=EAGER, no OMP_* variables"


def triad_protocol_problems(suite: dict, job: dict, record: dict) -> list[str]:
    """The protocol fields of a triad_count record against its reading: a timed reading on its
    GPU under the exclusive lock, CUDA_MODULE_LOADING=EAGER and no OMP_* variables, one heap
    layout, the unpatched original and the port with --sync-stages; a memory reading on its GPU
    (1) under the shared lock with D1's nsys command."""
    proto = record.get("protocol") or {}
    out = []
    gpu = triad_reading_gpu(suite, job)
    if proto.get("gpu") != gpu:
        out.append(f"GPU {proto.get('gpu')!r}, the reading's is {gpu}")
    lock = str(proto.get("lock", ""))
    original = str(proto.get("original", ""))
    if "UNPATCHED" not in original:
        out.append(f"original {original!r}: not the unpatched build")
    if job["kind"] == "run":
        if "exclusively" not in lock:
            out.append(f"lock {lock!r}: a timed reading holds the exclusive perf lock")
        if proto.get("env") != TRIAD_ENV:
            out.append(f"environment {proto.get('env')!r}, the protocol's is {TRIAD_ENV!r}")
        if not str(proto.get("layout", "")).startswith("one"):
            out.append(f"layout {proto.get('layout')!r}: the protocol runs one heap layout")
        if "--sync-stages" not in str(proto.get("port", "")):
            out.append(f"port {proto.get('port')!r}: the port runs with --sync-stages")
    else:
        if not lock.startswith("shared"):
            out.append(f"lock {lock!r}: a memory reading runs under the shared lock")
        if not str(proto.get("nsys", "")).startswith(TRIAD_NSYS):
            out.append(f"nsys {proto.get('nsys')!r}: D1's command is {TRIAD_NSYS!r}")
    return out


def check_triad_record(suite: dict, job: dict, record: dict, where: str, problems: list) -> None:
    """triad_count's record checks: the reading's cases and patterns exactly; the protocol
    fields (triad_protocol_problems); per timed case at least the reading's rounds, the medians
    and the sample counts recomputed (triad_sample_problems), the per-round guard passed in every
    round, no contaminated round kept, the contamination audit clean (a gated reading), and every
    gated row (update of every batch, full_count, end_to_end); per memory case no foreign GPU
    process, the guard of every measured process, both peaks, and every gated point of the store
    accounting (triad_required_rows)."""
    want = sorted(f"{c}/{p}" for c in job.get("cases") or [] for p in job.get("patterns") or [])
    got = sorted(record.get("results", {}))
    if want and got != want:
        problems.append(f"{where}: cases {got}, the reading's are {want}")
    for problem in triad_protocol_problems(suite, job, record):
        problems.append(f"{where}: {problem}")
    required = triad_required_rows(suite, job, record)
    sizes = {b["name"]: b for b in suite.get("batches", [])}
    for key, result in record.get("results", {}).items():
        case = key.split("/", 1)[0]
        m = TRIAD_CASE.match(case)
        spec = sizes.get(f"b{m['size']}") if m else None
        nb = result.get("batches")
        if spec is None or nb != spec.get("batches"):
            problems.append(
                f"{where}: {key}: {nb} batches, the suite's batch "
                f"{'b' + m['size'] if m else '?'} has {spec and spec.get('batches')}"
            )
            nb = nb or 0
        if job["kind"] == "run":
            n = result.get("rounds", 0)
            for problem in triad_sample_problems(result, job.get("runs")):
                problems.append(f"{where}: {key}: {problem}")
            g = result.get("guard") or {}
            if g.get("passed") is not True or g.get("rounds_checked", 0) < n:
                problems.append(f"{where}: {key}: the per-round guard did not pass in every round")
            monitor = result.get("monitor") if isinstance(result.get("monitor"), dict) else {}
            kept = int(result.get("contaminated_rounds_kept", 0)) + sum(
                1
                for rnd in monitor.get("rounds", []) or []
                for side in ("original", "port")
                if isinstance(rnd, dict) and (rnd.get(side) or {}).get("reasons")
            )
            if kept:
                problems.append(
                    f"{where}: {key}: {kept} contaminated rounds were kept, not repeated: not a "
                    "reading"
                )
            audit = result.get("audit") or {}
            if job.get("gated") and (audit.get("flagged") or "flagged" not in audit):
                problems.append(
                    f"{where}: {key}: the contamination audit flags rounds "
                    f"{audit.get('flagged_rounds')}: take the reading again"
                )
            rows = {reg.get("region") for reg in result.get("regions", [])}
            missing = [r for r in required[key] if r not in rows]
            if missing:
                problems.append(f"{where}: {key}: no row {', '.join(missing)}")
        else:
            foreign = result.get("foreign_gpu_processes") or {}
            if foreign.get("before") or foreign.get("after") or "before" not in foreign:
                problems.append(f"{where}: {key}: a foreign process on the GPU (or not checked)")
            g = result.get("guard") or {}
            if g.get("passed") is not True or g.get("runs_checked", 0) < 4:
                problems.append(
                    f"{where}: {key}: no count guard on the measured processes (the plain and the "
                    "profiled runs must print the golden's counts)"
                )
            have = {
                p.get("row")
                for p in result.get("store_accounting") or []
                if p.get("gated")
                and p.get("dyng_device_bytes")
                and p.get("escher_gpu_device_bytes")
            }
            if all(
                isinstance((result.get(side) or {}).get("peak_live_bytes"), int)
                for side in ("original", "port")
            ):
                have.add("device_memory")
            missing = [r for r in required[key] if r not in have]
            if "device_memory" in missing:
                problems.append(f"{where}: {key}: no device_memory peak (both sides' bytes)")
            stores = [r for r in missing if r != "device_memory"]
            if stores:
                problems.append(f"{where}: {key}: no store accounting at {', '.join(stores)}")


def summary_name(suite: dict, partial: dict | None) -> str:
    """<suite>.json for a whole execution, <suite>.partial.json for a narrowed one."""
    return f"{suite['suite']}{'.partial' if partial else ''}.json"


def summarize(
    suite: dict,
    jobs: list[dict],
    *,
    out: Path,
    version: str,
    records: Path,
    inputs: dict | None,
    partial: dict | None = None,
) -> dict:
    tol = suite["tolerance"]
    problems: list[str] = []
    readings = []
    out.mkdir(parents=True, exist_ok=True)
    started = execution_started(records, suite)
    for job in jobs:
        if job.get("rc", 0) != 0:
            problems.append(f"{Path(job['record']).name}: the harness exited {job['rc']}")
        entry = read_record(suite, job, tol, problems, started)
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
    contaminated = [
        f"{r['reading']}: {case} ({n} runs above the foreign-load threshold)"
        for r in readings
        for case, n in (r.get("contaminated_runs") or {}).items()
    ]
    missing = [
        f"{r['reading']}{'/' + r['dataset'] if r['dataset'] else ''}"
        for r in readings
        if r.get("gated") and (r.get("record") is None or not r.get("complete"))
    ]
    commits = sorted({r["port_commit"] for r in readings if r.get("port_commit")})
    ratios = [g["ratio"] for _, g in gated]
    by_record = {r["record"]: r["inputs"] for r in readings if r.get("record")}
    drivers = sorted({json.dumps(r["driver"], sort_keys=True) for r in readings if r.get("driver")})
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
        "partial": partial,
        "execution_started": started.strftime("%Y-%m-%dT%H:%M:%SZ") if started else None,
        "inputs": {
            "verified": bool(by_record) and all(v["verified"] for v in by_record.values()),
            "run_check": inputs
            or "not run by this summary (bench_suite.py run checks every input first)",
            "by_record": by_record,
        },
        "drivers": [json.loads(d) for d in drivers],
        "readings": readings,
        "verdict": {
            "gated_regions": len(gated),
            "within_gate": len(gated) - len(exceeded),
            "ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "exceeded": exceeded,
            "provisional": provisional,
            "contaminated": contaminated,  # flagged, not failed (PLAN 8.6): read again
            "missing_or_incomplete": missing,
            "problems": problems,
            "passed": not (exceeded or provisional or missing or problems),
        },
    }
    (out / summary_name(suite, partial)).write_text(json.dumps(summary, indent=1) + "\n")
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
    for key in ("exceeded", "provisional", "contaminated", "missing_or_incomplete", "problems"):
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
            "--batches",
            type=lambda s: [x for x in s.split(",") if x],
            default=None,
            help="sssp, mosp: only these batches (a narrowed, partial execution)",
        )
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
        p.add_argument(
            "--keep-contaminated",
            action="store_true",
            help="sssp, mosp: a diagnostic execution on a busy machine (always partial): keep "
            "and flag the rounds the contamination monitor would repeat (perf_ab.py "
            "--keep-contaminated); never a gate reading",
        )
        if name == "run":
            p.add_argument(
                "--skip-existing", action="store_true", help="keep records that exist already"
            )
            p.add_argument(
                "--force", action="store_true", help="replace records that exist already"
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
    # A narrowed execution (some readings or datasets, other runs) is never the suite's summary:
    # it is written as <suite>.partial.json, and never into a release's results directory, whose
    # full summary and compacted records it would otherwise replace.
    partial = {
        k: v
        for k, v in (
            ("readings", args.readings),
            ("datasets", args.datasets),
            ("batches", args.batches),
            ("runs", args.runs),
            ("keep_contaminated", args.keep_contaminated),
        )
        if v
    } or None
    results_root = (REPO / "benchmarks" / "results").resolve()
    if partial and args.command != "plan" and out.resolve().is_relative_to(results_root):
        parser.error(
            "--readings, --datasets, --batches, --runs or --keep-contaminated narrow the suite: "
            "give --out outside "
            "benchmarks/results/ (the release's summary covers the whole suite)"
        )
    jobs = plan(
        suite,
        records=records,
        build_root=args.build_root.resolve(),
        readings=args.readings,
        datasets=args.datasets,
        batches=args.batches,
        keep_contaminated=args.keep_contaminated,
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
            jobs = run_jobs(jobs, skip_existing=args.skip_existing, log=log, force=args.force)
    summary = summarize(
        suite, jobs, out=out, version=version, records=records, inputs=inputs, partial=partial
    )
    print_summary(summary)
    return 0 if summary["verdict"]["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SuiteError as exc:
        print(f"bench_suite.py: {exc}", file=sys.stderr)
        sys.exit(2)
