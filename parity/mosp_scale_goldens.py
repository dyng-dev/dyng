#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The paper-scale mosp goldens (M7, PLAN Sections 6.4.4 and 8.3): export and compare.

    parity/export_goldens.py mosp_scale [--cases ...] [--no-toml]
    parity/compare.py mosp_scale --exe build/parity/tools/compat/dyng-compat-mosp
                                 [--configs openmp:28,sequential] [--cases ...] [--json ...]
    parity/compare.py mosp_scale --exe build/parity-cuda/tools/compat/dyng-compat-mosp
                                 --configs cuda,cuda-operators

(or `parity/mosp_scale_goldens.py export|compare ...` directly).

The cases are the benchmark inputs of the original's bench/prepare.sh, prepared by
`parity/perf_ab.py prepare` under $DYNG_SCRATCH/bench/mosp/<graph> (MOSP-OpenMP@c352151's
mospPrep: K = 3 weights in [1, 100] of seed 12345, Dijkstra trees from source 0, the 50K safe
and unsafe batches and the local 10K batch of seed 777):

  group    cases
  gate        12  roadNet-PA, roadNet-CA, rgg (rgg_n_2_20_s0), road_usa_g x safe50k, unsafe50k,
                  local10k; the default preferences (all 1), K = 3: the inputs of the gates
  pref         2  roadNet-CA x safe50k, local10k with Pref {4, 1, 4} (L = 4)
  ksweep       6  roadNet-CA-K4 (`perf_ab.py prepare --graph roadNet-CA-K4 --widen roadNet-CA:4`:
                  `mospPrep widen` to 4 objectives, weights [1, 100], seed 12345; its 50K safe
                  batch of seed 777) with K = 1, 2, 3, 4 (`-k`) and the default preferences, K = 2
                  with Pref {1, 3} and K = 4 with Pref {4, 1, 4, 2} (MOSP-CUDA@e220ee2
                  results/README.md, "Validation")

Paper-scale outputs are too large to keep (road_usa: about 1.8 GB per case), so a case stores
only its command line, the SHA-256 of every input file it reads and of every output file of the
original's `mosp`, and its invalidated counters (PLAN 8.3: "Paper-scale cases store only the
command lines and the SHA-256 of each output file"). They live in the [sets.mosp_scale] section
of parity/goldens.toml (and in $DYNG_SCRATCH/goldens/mosp_scale/<case>/case.json).

export  runs bin/mosp of the PATCHED scratch copy of MOSP-OpenMP@c352151 (28 threads pinned;
        the copy's sources are unchanged) on every case and records the digests; then, as the
        cross-check of the two originals (PLAN 6.3 step 2), bin/mosp of the PATCHED
        MOSP-CUDA@e220ee2 copy (GPU $CUDA_VISIBLE_DEVICES, default 1) must write files with the
        same SHA-256 and report the same invalidated counters. Run it under the shared perf lock.
compare verifies the SHA-256 of the inputs, then runs `dyng-compat-mosp --mosp` with the case's
        options in every configuration (sequential, openmp[:threads], cuda[:device],
        cuda-fused[:device], cuda-operators[:device]; cuda on GPU 1 unless CUDA_VISIBLE_DEVICES is
        set), writes the outputs into a scratch directory, and requires the SHA-256 of every
        output file and the invalidated counters to equal the golden's. Exit status: 0 all equal,
        1 a mismatch or an error, 77 an input or the set is missing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import compare as sssp_compare  # noqa: E402  (run, host_info, build_info, git_head, ...)
import cycle_count_goldens  # noqa: E402  (write_toml: the sets section writer)

SET = "mosp_scale"
SKIP = 77
SCRATCH = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
COMMITS = {
    "MOSP-OpenMP": "c35215135341d5b5d1553458afe4b2226edc38fb",
    "MOSP-CUDA": "e220ee20d1b0948ece3df135a02d1b898264c22f",
}
BATCHES = {
    "safe50k": "changes_50000_50_safe",
    "unsafe50k": "changes_50000_50",
    "local10k": "changes_local_10000_50_safe",
}
GROUPS = ["gate", "pref", "ksweep"]
INVALIDATED = re.compile(r"^obj(\d+)\s+SOSP update .*\(invalidated (\d+),", re.M)
COMBINED = ["distancesCsr.txt", "SSSPTreeCsr.txt", "mospCosts.txt"]
CSR_PARTS = ["RowPtr", "ColInd", "Values"]


@dataclass(frozen=True)
class Case:
    group: str
    graph: str
    batch: str
    k: int = 0  # 0: every objective of the graph
    pref: str = ""

    @property
    def name(self) -> str:
        tail = f"_k{self.k}" if self.k else ""
        tail += f"_pref{self.pref.replace(',', '-')}" if self.pref else ""
        return f"{self.group}/{self.graph}_{self.batch}{tail}"

    def options(self) -> list[str]:
        return (["-k", str(self.k)] if self.k else []) + (
            ["--pref", self.pref] if self.pref else []
        )


def all_cases() -> list[Case]:
    out = [
        Case("gate", g, b)
        for g in ["roadNet-PA", "roadNet-CA", "rgg", "road_usa_g"]
        for b in ["safe50k", "unsafe50k", "local10k"]
    ]
    out += [Case("pref", "roadNet-CA", b, 0, "4,1,4") for b in ["safe50k", "local10k"]]
    out += [Case("ksweep", "roadNet-CA-K4", "safe50k", k) for k in [1, 2, 3, 4]]
    out += [
        Case("ksweep", "roadNet-CA-K4", "safe50k", 2, "1,3"),
        Case("ksweep", "roadNet-CA-K4", "safe50k", 4, "4,1,4,2"),
    ]
    return out


def select(cases: list[Case], spec: str) -> list[Case]:
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    if not wanted:
        return cases
    out = [c for c in cases if c.group in wanted or c.name in wanted or c.graph in wanted]
    if not out:
        raise SystemExit(f"--cases {spec}: selects no case")
    return out


def sha256_file(path: Path) -> str:
    return sssp_compare.sha256_file(path)


def input_files(case: Case) -> dict[str, Path]:
    """The files the case reads, by their name relative to the graph's bench directory."""
    data = SCRATCH / "bench" / "mosp" / case.graph
    files = {f"csr/graphCsr{x}.txt": data / "csr" / f"graphCsr{x}.txt" for x in CSR_PARTS}
    for f in ["insert.txt", "delete.txt"]:
        files[f"{BATCHES[case.batch]}/{f}"] = data / BATCHES[case.batch] / f
    k = case.k or len(list((data / "init").glob("obj*")))
    for o in range(k):
        for f in ["distancesOriginal.txt", "SSSPTreeOriginal.txt"]:
            files[f"init/obj{o}/{f}"] = data / "init" / f"obj{o}" / f
    return files


def case_args(case: Case) -> list:
    data = SCRATCH / "bench" / "mosp" / case.graph
    return [
        "--graph",
        data / "csr" / "graphCsr",
        "--changes",
        data / BATCHES[case.batch],
        "--init",
        data / "init",
        *case.options(),
    ]


def output_digests(out: Path, k: int) -> dict[str, str]:
    names = [
        f"obj{o}/{f}" for o in range(k) for f in ["distancesUpdated.txt", "SSSPTreeUpdated.txt"]
    ]
    names += [f"combinedGraph/{f}" for f in COMBINED]
    digests = {}
    for name in names:
        path = out / name
        digests[name] = sha256_file(path) if path.is_file() else "missing"
    return digests


def counters(log: str, k: int) -> list[int | None]:
    got = {int(o): int(v) for o, v in INVALIDATED.findall(log)}
    return [got.get(o) for o in range(k)]


def objectives_of(case: Case) -> int:
    data = SCRATCH / "bench" / "mosp" / case.graph
    return case.k or len(list((data / "init").glob("obj*")))


# --- export --------------------------------------------------------------------------------------


def reference_dir(name: str) -> Path:
    tool = REPO / "parity" / "build_reference.sh"
    base = [tool, "--variant", "patched", "--scratch", SCRATCH]
    subprocess.run([str(c) for c in [*base, name]], check=True)
    out = subprocess.check_output([str(c) for c in [*base, "--print-dir", name]], text=True)
    return Path(out.strip().splitlines()[-1])


def run_original(ref: Path, case: Case, out: Path, env: dict) -> tuple[dict, list]:
    shutil.rmtree(out, ignore_errors=True)
    rc, log = sssp_compare.run([ref / "bin" / "mosp", *case_args(case), "--out", out], env)
    if rc != 0:
        raise SystemExit(f"{case.name}: {ref.name} bin/mosp failed ({rc}): {log[-800:]}")
    k = objectives_of(case)
    digests = output_digests(out, k)
    shutil.rmtree(out, ignore_errors=True)
    return digests, counters(log, k)


def toml_section(metas: list[dict]) -> str:
    groups = {g: 0 for g in GROUPS}
    rows = []
    for meta in sorted(metas, key=lambda m: m["case"]):
        groups[meta["case"].split("/")[0]] += 1
        rows.append(f'\n[sets.{SET}.cases."{meta["case"]}"]\n')
        rows.append(f'graph = "{meta["graph"]}"\nbatch = "{meta["batch"]}"\n')
        rows.append(f'k = {meta["k"]}\npref = "{meta["pref"]}"\n')
        rows.append(f'command = "{meta["command"]}"\n')
        rows.append(f"invalidated = {json.dumps(meta['invalidated'])}\n")
        rows.append(f'[sets.{SET}.cases."{meta["case"]}".inputs]\n')
        rows += [f'"{n}" = "{d}"\n' for n, d in sorted(meta["inputs"].items())]
        rows.append(f'[sets.{SET}.cases."{meta["case"]}".outputs]\n')
        rows += [f'"{n}" = "{d}"\n' for n, d in sorted(meta["outputs"].items())]
    head = f"""[sets.{SET}]
reference = "MOSP-OpenMP"
commit = "{COMMITS["MOSP-OpenMP"]}"
cross_checked_with = "MOSP-CUDA {COMMITS["MOSP-CUDA"]}"
location = "$DYNG_SCRATCH/bench/mosp/<graph> (parity/perf_ab.py prepare); outputs: SHA-256"
generated_by = "parity/export_goldens.py mosp_scale (parity/mosp_scale_goldens.py)"
num_cases = {len(metas)}

[sets.{SET}.groups]
""" + "".join(f"{g} = {n}\n" for g, n in groups.items() if n)
    return head + "".join(rows)


def export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="export_goldens.py mosp_scale",
        description="Record the paper-scale mosp goldens (SHA-256) of MOSP-OpenMP@c352151, "
        "cross-checked with MOSP-CUDA@e220ee2.",
    )
    p.add_argument("--cases", default="", help="restrict to these groups, graphs or cases")
    p.add_argument("--no-toml", action="store_true", help="do not write parity/goldens.toml")
    p.add_argument("--no-cuda", action="store_true", help="skip the MOSP-CUDA cross-check")
    args = p.parse_args(argv)
    cases = select(all_cases(), args.cases)
    missing = [
        c.graph for c in cases if not (SCRATCH / "bench" / "mosp" / c.graph / "init").is_dir()
    ]
    if missing:
        print(
            f"inputs missing for {sorted(set(missing))}: run parity/perf_ab.py prepare",
            file=sys.stderr,
        )
        return SKIP
    omp = reference_dir("MOSP-OpenMP")
    cuda = None if args.no_cuda else reference_dir("MOSP-CUDA")
    env = dict(os.environ, OMP_NUM_THREADS="28", OMP_PROC_BIND="close", OMP_PLACES="cores")
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env.setdefault("CUDA_MODULE_LOADING", "EAGER")
    work = Path(tempfile.mkdtemp(prefix="dyng-mosp-scale-", dir=SCRATCH / "runs"))
    metas = []
    try:
        for case in cases:
            inputs = {n: sha256_file(f) for n, f in input_files(case).items()}
            outputs, invalidated = run_original(omp, case, work / "omp", env)
            if "missing" in outputs.values() or None in invalidated:
                raise SystemExit(f"{case.name}: MOSP-OpenMP wrote no complete output: {outputs}")
            line = f"{case.name}: MOSP-OpenMP {len(outputs)} files, invalidated {invalidated}"
            if cuda is not None:
                got, got_inv = run_original(cuda, case, work / "cuda", env)
                if got != outputs or got_inv != invalidated:
                    diff = [n for n in outputs if got.get(n) != outputs[n]]
                    raise SystemExit(
                        f"{case.name}: MOSP-CUDA differs from MOSP-OpenMP: files {diff}, "
                        f"invalidated {got_inv} vs {invalidated}"
                    )
                line += "; MOSP-CUDA identical"
            print(line, flush=True)
            opts = " ".join(case.options())
            meta = {
                "case": case.name,
                "graph": case.graph,
                "batch": BATCHES[case.batch],
                "k": objectives_of(case),
                "pref": case.pref,
                "command": (
                    f"mosp --graph <bench>/{case.graph}/csr/graphCsr --changes "
                    f"<bench>/{case.graph}/{BATCHES[case.batch]} --init <bench>/{case.graph}/init"
                    + (f" {opts}" if opts else "")
                ),
                "invalidated": invalidated,
                "inputs": inputs,
                "outputs": outputs,
            }
            d = SCRATCH / "goldens" / SET / case.name
            d.mkdir(parents=True, exist_ok=True)
            (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
            metas.append(meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not args.no_toml and len(cases) == len(all_cases()):
        cycle_count_goldens.write_toml(toml_section(metas), SET)
        print(f"wrote the [sets.{SET}] section of {REPO / 'parity' / 'goldens.toml'}")
    return 0


# --- compare -------------------------------------------------------------------------------------


def config_args(config: str) -> list[str]:
    backend, _, number = config.partition(":")
    if backend.startswith("cuda"):
        out = ["--backend", "cuda"] + (["--device", number] if number else [])
        if backend != "cuda":
            out += ["--cuda-engine", backend.removeprefix("cuda-")]
        return out
    return ["--backend", backend] + (["--threads", number] if number else [])


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="compare.py mosp_scale",
        description="Replay the paper-scale mosp goldens with dyng-compat-mosp --mosp.",
    )
    p.add_argument("--exe", type=Path, required=True, help="dyng-compat-mosp")
    p.add_argument("--configs", default="openmp:28,sequential")
    p.add_argument("--cases", default="", help="restrict to these groups, graphs or cases")
    p.add_argument("--json", type=Path, help="write the result matrix here")
    p.add_argument("--label", default="")
    args = p.parse_args(argv)
    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    if SET not in toml.get("sets", {}):
        print(f"parity/goldens.toml has no [sets.{SET}] (skipping)", file=sys.stderr)
        return SKIP
    golden = toml["sets"][SET]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        backend, _, number = c.partition(":")
        if backend not in ("sequential", "openmp", "cuda", "cuda-fused", "cuda-operators") or (
            number and not number.isdigit()
        ):
            p.error(f"--configs: '{c}' is not a configuration")
    by_name = {c.name: c for c in all_cases()}
    cases = [by_name[n] for n in sorted(golden["cases"]) if n in by_name]
    cases = select(cases, args.cases)
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores")
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    # Verify the inputs once per file (several cases share them).
    hashed: dict[Path, str] = {}
    for case in cases:
        want = golden["cases"][case.name]["inputs"]
        files = input_files(case)
        if not all(f.is_file() for f in files.values()):
            print(f"{case.name}: inputs missing (parity/perf_ab.py prepare)", file=sys.stderr)
            return SKIP
        for name, f in files.items():
            if f not in hashed:
                hashed[f] = sha256_file(f)
            if hashed[f] != want.get(name):
                print(f"{case.name}: input {name} has another SHA-256 than the golden's")
                return 1
    print(f"inputs verified: {len(hashed)} files of {len(cases)} cases")
    work = Path(tempfile.mkdtemp(prefix="dyng-mosp-scale-cmp-", dir=SCRATCH / "runs"))
    matrix: dict[str, dict[str, dict]] = {}
    failures = []
    try:
        for case in cases:
            meta = golden["cases"][case.name]
            for config in configs:
                out = work / "out"
                shutil.rmtree(out, ignore_errors=True)
                cmd = [args.exe, *case_args(case), "--mosp", *config_args(config), "--out", out]
                rc, log = sssp_compare.run(cmd, env)
                bad = []
                if rc != 0:
                    bad.append(f"dyng-compat-mosp failed ({rc}): {log[-400:]}")
                else:
                    got = output_digests(out, meta["k"])
                    bad += [f"{n} differs" for n, d in meta["outputs"].items() if got.get(n) != d]
                    inv = counters(log, meta["k"])
                    if inv != meta["invalidated"]:
                        bad.append(f"invalidated {inv} != {meta['invalidated']}")
                shutil.rmtree(out, ignore_errors=True)
                cell = matrix.setdefault(case.group, {}).setdefault(
                    config, {"cases": 0, "pass": 0, "fail": []}
                )
                cell["cases"] += 1
                if bad:
                    cell["fail"].append({"case": case.name, "problems": bad[:10]})
                    failures.append((case.name, config, bad))
                else:
                    cell["pass"] += 1
                print(
                    f"{case.name} [{config}]: {'equal' if not bad else '; '.join(bad[:3])}",
                    flush=True,
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    order = [g for g in GROUPS if g in matrix]
    print("\n| group | cases | " + " | ".join(configs) + " |")
    print("|---|---:|" + "---:|" * len(configs))
    for g in order:
        n = matrix[g][configs[0]]["cases"]
        print(f"| {g} | {n} | " + " | ".join(f"{matrix[g][c]['pass']}/{n}" for c in configs) + " |")
    if args.json:
        result = {
            "schema": 1,
            "algorithm": "mosp",
            "set": SET,
            "label": args.label,
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference": {
                "name": golden["reference"],
                "commit": golden["commit"],
                "cross_checked_with": golden["cross_checked_with"],
            },
            "port": {"repository": "dyng", "commit": sssp_compare.git_head()},
            "executable": sssp_compare.portable_path(args.exe),
            "build": sssp_compare.build_info(args.exe),
            "host": sssp_compare.host_info(),
            "tolerance": "none: SHA-256 of every output file and the invalidated counters equal",
            "configs": configs,
            "cases": [c.name for c in cases],
            "matrix": {g: matrix[g] for g in order},
            "passed": not failures,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"\n{'ALL EQUAL' if not failures else f'{len(failures)} MISMATCHES'}: "
        f"{len(cases)} cases x {len(configs)} configurations"
    )
    return 0 if not failures else 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "compare"):
        print(__doc__)
        return 2
    return export_main(sys.argv[2:]) if sys.argv[1] == "export" else compare_main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
