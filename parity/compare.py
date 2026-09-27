#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Replay the sssp golden corpus and compare byte for byte (PLAN Sections 6.3 step 6 and 8.3).

    parity/compare.py --exe build/parity/tools/compat/dyng-compat-mosp [options]
    parity/compare.py --driver original --ref <scratch copy of an original> [options]

First the goldens are verified: MANIFEST.sha256 must have the SHA-256 recorded in
parity/goldens.toml, every file must have the SHA-256 listed in MANIFEST.sha256, and every case
digest must equal the one in parity/goldens.toml. Then every case is replayed with every
configuration:

driver "compat" (dynG, through tools/compat/dyng-compat-mosp), per configuration
(sequential, openmp:<threads>):
  * `dyng-compat-mosp init`   == init/obj<k>/{distancesOriginal,SSSPTreeOriginal}.txt (compute;
    init_canonical/ for the noncanonical group, whose init/ holds perturbed tie parents)
  * `dyng-compat-mosp` update == updated/obj<k>/{distancesUpdated,SSSPTreeUpdated}.txt, from the
    golden initial trees (without --canonicalize, like `mosp`), and its `--write-graph` output == applied/graphCsr{RowPtr,ColInd,Values}.txt
    (graph::apply under mosp_compatible() vs applyChangeBatch + writeCsrGraph)
  * the invalidated counter of every objective == case.json (the original's counter)

driver "original" (another original, e.g. MOSP-CUDA@e220ee2 for the one-off cross-check of PLAN
Section 6.3 step 2): its bin/mospPrep init, bin/mosp (updated trees AND the combined graph) and
parity_export/bin/export_graph_io apply (updated CSR and the weight-increase bits) must produce
the golden files; its invalidated counters must be equal as well.

Comparisons are exact (no tolerance). Exit status: 0 all equal, 1 a mismatch or an error,
77 the goldens are missing (CTest skips the test).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import filecmp
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
INVALIDATED = re.compile(r"^obj(\d+)\s+SOSP update .*\(invalidated (\d+),", re.M)
CSR = ["graphCsrRowPtr.txt", "graphCsrColInd.txt", "graphCsrValues.txt"]
SKIP = 77


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_goldens(root: Path, golden_set: dict) -> None:
    """Checks the manifest, every file and every case digest; raises on any difference."""
    manifest = root / "MANIFEST.sha256"
    text = manifest.read_text()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != golden_set["manifest_sha256"]:
        raise SystemExit(f"{manifest}: SHA-256 {got} != parity/goldens.toml "
                         f"{golden_set['manifest_sha256']} (re-export or update the toml)")
    per_case: dict[str, list[tuple[PurePosixPath, str]]] = {}
    listed = set()
    for line in text.splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        if sha256_file(root / rel) != digest:
            raise SystemExit(f"golden file changed: {root / rel}")
        parts = PurePosixPath(rel).parts
        per_case.setdefault("/".join(parts[:2]), []).append((PurePosixPath(*parts[2:]), digest))
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and p.name != "MANIFEST.sha256"}
    if on_disk != listed:
        extra = sorted(on_disk ^ listed)[:5]
        raise SystemExit(f"goldens and MANIFEST.sha256 disagree on the file list: {extra}")
    cases = golden_set["cases"]
    if set(cases) != set(per_case):
        raise SystemExit("the cases of parity/goldens.toml and MANIFEST.sha256 differ")
    for name, files in per_case.items():
        lines = "".join(f"{d}  {p.as_posix()}\n" for p, d in sorted(files))
        if hashlib.sha256(lines.encode()).hexdigest() != cases[name]["sha256"]:
            raise SystemExit(f"case digest of {name} differs from parity/goldens.toml")


def run(cmd: list, env: dict) -> tuple[int, str]:
    proc = subprocess.run([str(c) for c in cmd], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def compare_files(pairs: list[tuple[Path, Path, str]]) -> list[str]:
    bad = []
    for got, want, label in pairs:
        if not got.is_file():
            bad.append(f"{label}: missing")
        elif not filecmp.cmp(got, want, shallow=False):
            bad.append(f"{label}: differs")
    return bad


def check_counters(log: str, want: list[int]) -> list[str]:
    got = {int(o): int(v) for o, v in INVALIDATED.findall(log)}
    have = [got.get(o) for o in range(len(want))]
    return [] if have == want else [f"invalidated {have} != {want}"]


def tree_pairs(out: Path, golden: Path, sub: str, names: list[str], k: int) -> list:
    return [(out / f"obj{o}" / f, golden / sub / f"obj{o}" / f, f"{sub}/obj{o}/{f}")
            for o in range(k) for f in names]


def canonical_init(golden: Path) -> str:
    """The golden directory of the canonical initial trees (compute / mospPrep init)."""
    return "init_canonical" if (golden / "init_canonical").is_dir() else "init"


def replay_compat(exe: Path, golden: Path, meta: dict, config: str, tmp: Path, env: dict) -> list[str]:
    k = meta["num_objectives"]
    backend, _, threads = config.partition(":")
    extra = ["--backend", backend] + (["--threads", threads] if threads else [])
    bad = []
    rc, log = run([exe, "init", golden / "input" / "graphCsr", tmp / "init", "-k", k, *extra], env)
    if rc != 0:
        return [f"init failed ({rc}): {log[-500:]}"]
    bad += compare_files(tree_pairs(tmp / "init", golden, canonical_init(golden),
                                    ["distancesOriginal.txt", "SSSPTreeOriginal.txt"], k))
    rc, log = run([exe, "--graph", golden / "input" / "graphCsr", "--changes", golden / "input",
                   "--init", golden / "init", "-k", k, "--out", tmp / "updated",
                   "--write-graph", tmp / "applied" / "graphCsr", *extra], env)
    if rc != 0:
        return bad + [f"update failed ({rc}): {log[-500:]}"]
    bad += compare_files(tree_pairs(tmp / "updated", golden, "updated",
                                    ["distancesUpdated.txt", "SSSPTreeUpdated.txt"], k))
    bad += compare_files([(tmp / "applied" / f, golden / "applied" / f, f"applied/{f}")
                          for f in CSR])
    bad += check_counters(log, meta["invalidated"])
    return bad


def replay_original(ref: Path, golden: Path, meta: dict, tmp: Path, env: dict) -> list[str]:
    k = meta["num_objectives"]
    bad = []
    # Without -k (MOSP-CUDA's mospPrep has no -k; every graph of the corpus has edges, so K is
    # the graph's number of weight columns).
    rc, log = run([ref / "bin" / "mospPrep", "init", golden / "input" / "graphCsr", tmp / "init"],
                  env)
    if rc != 0:
        return [f"mospPrep init failed ({rc}): {log[-500:]}"]
    bad += compare_files(tree_pairs(tmp / "init", golden, canonical_init(golden),
                                    ["distancesOriginal.txt", "SSSPTreeOriginal.txt"], k))
    rc, log = run([ref / "bin" / "mosp", "--graph", golden / "input" / "graphCsr", "--changes",
                   golden / "input", "--init", golden / "init", "-k", k, "--out",
                   tmp / "updated"], env)
    if rc != 0:
        return bad + [f"mosp failed ({rc}): {log[-500:]}"]
    bad += compare_files(tree_pairs(tmp / "updated", golden, "updated",
                                    ["distancesUpdated.txt", "SSSPTreeUpdated.txt"], k))
    bad += compare_files([(tmp / "updated" / "combinedGraph" / f, golden / "combined" / f,
                           f"combined/{f}")
                          for f in ["distancesCsr.txt", "SSSPTreeCsr.txt", "mospCosts.txt"]])
    bad += check_counters(log, meta["invalidated"])
    rc, log = run([ref / "parity_export" / "bin" / "export_graph_io", "apply",
                   golden / "input" / "graphCsr", golden / "input" / "insert.txt",
                   golden / "input" / "delete.txt", tmp / "applied" / "graphCsr"], env)
    if rc != 0:
        return bad + [f"export_graph_io apply failed ({rc}): {log[-500:]}"]
    bad += compare_files([(tmp / "applied" / f, golden / "applied" / f, f"applied/{f}")
                          for f in CSR + ["graphCsrWeightIncrease.txt"]])
    return bad


def portable_path(path: Path | str) -> str:
    """A path for the committed records: relative to the repository, or under $DYNG_SCRATCH,
    never an absolute personal path (PLAN Section 3.4: no personal paths in the repository)."""
    p = Path(path).resolve()
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    for base, label in [(REPO, None), (scratch.resolve(), "$DYNG_SCRATCH")]:
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        return rel if label is None else f"{label}/{rel}"
    if p.is_relative_to(Path.home()):
        return f"<outside the repository>/{p.name}"
    return p.as_posix()


def host_info() -> dict:
    """The machine, without its host name: CPU model and logical CPU count."""
    model = platform.processor()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": model, "logical_cpus": os.cpu_count(), "kernel": platform.release(),
            "python": platform.python_version()}


def build_info(exe: Path) -> dict:
    """Build type and compiler of the CMake tree that contains exe (walks up to CMakeCache.txt)."""
    info: dict = {}
    for d in exe.resolve().parents:
        cache = d / "CMakeCache.txt"
        if cache.is_file():
            keys = {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                    "CMAKE_CXX_FLAGS_RELEASE", "DYNG_ENABLE_OPENMP", "CMAKE_CXX_COMPILER_VERSION"}
            for line in cache.read_text().splitlines():
                name = line.split(":", 1)[0]
                if name in keys and "=" in line:
                    info[name] = line.split("=", 1)[1]
            info["build_dir"] = portable_path(d)
            break
    if "CMAKE_CXX_COMPILER" in info:
        rc, out = run([info["CMAKE_CXX_COMPILER"], "--version"], dict(os.environ))
        if rc == 0:
            info["compiler"] = out.splitlines()[0]
    return info


def git_head() -> str:
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD"]).returncode != 0
        return head + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--goldens", type=Path, default=scratch / "goldens" / "sssp")
    parser.add_argument("--driver", choices=["compat", "original"], default="compat")
    parser.add_argument("--exe", type=Path, help="dyng-compat-mosp (driver compat)")
    parser.add_argument("--ref", type=Path, help="scratch copy of an original (driver original)")
    parser.add_argument("--configs", default="sequential,openmp:1,openmp:4,openmp:16",
                        help="driver compat: backend[:threads] list")
    parser.add_argument("--groups", default="", help="restrict to these groups (comma list)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--json", type=Path, help="write the result matrix here")
    parser.add_argument("--label", default="", help="free text stored in the JSON")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not re-hash the goldens (they were verified by an earlier run)")
    args = parser.parse_args()

    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    golden_set = toml["sets"]["sssp"]
    if not (args.goldens / "MANIFEST.sha256").is_file():
        print(f"goldens not found under {args.goldens}; run parity/export_goldens.py "
              "(skipping)", file=sys.stderr)
        return SKIP
    if not args.skip_verify:
        verify_goldens(args.goldens, golden_set)
        print(f"goldens verified: {golden_set['num_cases']} cases, {golden_set['num_files']} files, "
              f"manifest {golden_set['manifest_sha256'][:16]}")

    env = dict(os.environ)
    env.setdefault("OMP_WAIT_POLICY", "passive")  # many small processes share the cores
    if args.driver == "compat":
        if args.exe is None or not args.exe.is_file():
            parser.error("--exe <dyng-compat-mosp> is required for driver compat")
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        if not configs:
            parser.error("--configs selects no configuration")
        for c in configs:
            backend, _, threads = c.partition(":")
            if backend not in ("sequential", "openmp") or (threads and not threads.isdigit()):
                parser.error(f"--configs: '{c}' is not sequential or openmp[:<threads>]")
    else:
        if args.ref is None or not (args.ref / "bin" / "mosp").is_file():
            parser.error("--ref <scratch copy with bin/mosp> is required for driver original")
        if not (args.ref / "parity_export" / "bin" / "export_graph_io").is_file():
            parser.error("--ref needs the export tools: use the PATCHED copy of build_reference.sh")
        configs = ["original"]
        env.setdefault("CUDA_MODULE_LOADING", "EAGER")
        env.setdefault("CUDA_VISIBLE_DEVICES", "1")
        env.setdefault("OMP_NUM_THREADS", "4")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = sorted(set(groups) - set(golden_set["groups"]))
    if unknown:
        parser.error(f"--groups: unknown group(s) {', '.join(unknown)}; the corpus has "
                     f"{', '.join(golden_set['groups'])}")
    cases = sorted(c for c in golden_set["cases"] if not groups or c.split("/")[0] in groups)
    if not cases:
        # A filter that selects nothing must never read as "ALL EQUAL".
        print("compare.py: no case selected", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="dyng-compare-", dir=scratch / "runs"
                                 if (scratch / "runs").is_dir() else None))

    def job(case: str, config: str) -> tuple[str, str, list[str]]:
        golden = args.goldens / case
        meta = json.loads((golden / "case.json").read_text())
        tmp = work / config.replace(":", "_") / case
        try:
            if args.driver == "compat":
                bad = replay_compat(args.exe.resolve(), golden, meta, config, tmp, env)
            else:
                bad = replay_original(args.ref.resolve(), golden, meta, tmp, env)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return case, config, bad

    matrix: dict[str, dict[str, dict]] = {}
    failures = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(job, c, cfg) for c in cases for cfg in configs]
            for fut in concurrent.futures.as_completed(futures):
                case, config, bad = fut.result()
                cell = matrix.setdefault(case.split("/")[0], {}).setdefault(
                    config, {"cases": 0, "pass": 0, "fail": []})
                cell["cases"] += 1
                if bad:
                    cell["fail"].append({"case": case, "problems": bad[:10]})
                    failures.append((case, config, bad))
                else:
                    cell["pass"] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Report.
    order = [g for g in golden_set["groups"] if g in matrix]
    print(f"\n| group | cases | " + " | ".join(configs) + " |")
    print("|---|---:|" + "---:|" * len(configs))
    for g in order:
        n = matrix[g][configs[0]]["cases"]
        cells = [f"{matrix[g][c]['pass']}/{n}" for c in configs]
        print(f"| {g} | {n} | " + " | ".join(cells) + " |")
    total = len(cases)
    cells = [f"{sum(matrix[g][c]['pass'] for g in order)}/{total}" for c in configs]
    print(f"| **all** | {total} | " + " | ".join(cells) + " |")
    for case, config, bad in sorted(failures)[:30]:
        print(f"MISMATCH {case} [{config}]: {'; '.join(bad[:4])}")

    if args.json:
        result = {
            "schema": 1,
            "algorithm": "sssp",
            "label": args.label,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference": {"name": golden_set["reference"], "commit": golden_set["commit"]},
            "port": {"repository": "dyng", "commit": git_head()},
            "driver": args.driver,
            "executable": portable_path(args.exe or args.ref),
            "build": (build_info(args.exe) if args.exe
                      else {"reference_copy": portable_path(args.ref)}),
            "host": host_info(),
            "goldens": {"manifest_sha256": golden_set["manifest_sha256"],
                        "cases": golden_set["num_cases"], "files": golden_set["num_files"]},
            "tolerance": "none: byte equality of every compared file; invalidated counters equal",
            "configs": configs,
            "matrix": {g: matrix[g] for g in order},
            "passed": not failures,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\n{'ALL EQUAL' if not failures else f'{len(failures)} MISMATCHES'}: "
          f"{total} cases x {len(configs)} configurations")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
