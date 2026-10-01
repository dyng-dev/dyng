#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Write the parity certificate of a release (PLAN Sections 8.3 and 10.3, steps 1-3).

    parity/certify.py check --version 0.1.0rc1 --name asan --command "ctest --preset asan" \\
                            --ctest-log asan-ctest.log [--commit <sha>]
    parity/certify.py check --version 0.1.0rc1 --name gpu_local --command ci/gpu_local.sh \\
                            --gpu-summary build/dev-cuda/gpu_local_summary.md
    parity/certify.py check --version 0.1.0rc1 --name mutation-sssp \\
                            --command "parity/mutate.py run" --json-verdict mutation-sssp.json
    parity/certify.py equivalence --version 0.1.0rc1 --measured d13d393 --clone <clean clone>
    parity/certify.py write --version 0.1.0rc1 [--results benchmarks/results/0.1.0rc1]

Everything the certificate states is read from files under benchmarks/results/<version>/, which
are committed with it:

  <suite>.json      the summaries parity/bench_suite.py writes (the performance gates of
                    benchmarks/paper/<suite>.yaml against the unpatched originals) and the
                    compacted harness records next to them;
  parity-*.json     the golden replays (parity/compare.py and compare.py cycle_count --json):
                    each golden set's manifest digest, the case x configuration matrix, the
                    tolerance and the verdict;
  checks.json       the other release checks (sanitizer presets, compute-sanitizer, mutation
                    CTests, check scripts), each with its command, commit, result and counts,
                    added one at a time by `certify.py check`;
  equivalence.json  (only when needed) builds of a measured commit and of the release that
                    differ only in METADATA_PATHS, compared by `certify.py equivalence`.

`write` assembles benchmarks/results/<version>/parity.json: the commit of the release and the
commits whose builds were measured (each must have the release commit's sources: `git diff
<measured> <release> -- <paths>` is empty, with the library paths of CODE_PATHS for the gates, the
replays and `check --scope library`, and the C++ tests too for a test-suite check, or the
certificate fails; the one exception is a library-scope difference confined to METADATA_PATHS for
which equivalence.json shows the measured programs unchanged), the hardware, driver, CUDA and
compiler versions, the build presets of the measured builds, the pinned originals (commit,
baseline SHA, upstream, which copy serves what), every golden set of parity/goldens.toml with its
manifest SHA-256, the SHA-256 of every case and every replay's pass/fail matrix (case x backend x
configuration), the tolerances used, the performance-gate table (every gated region: medians,
ratio, gate, verdict; the recorded default-clock readings separately) and the checks. It also
rewrites the generated part of README.md (between the certify markers) from the certificate. Exit
1 if any part failed or is missing; the certificate is written either way, with the reason.
"""

from __future__ import annotations

import argparse
import datetime
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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMA = 1
# What a measured build is made of (the library, the compat tools, the build system), and what a
# test-suite check runs in addition (the C++ tests). The performance gates, the golden replays and
# the golden mutations need the release's library sources; a test-suite check also its tests.
LIBRARY_PATHS = [
    "cpp/include",
    "cpp/src",
    "cpp/CMakeLists.txt",
    "tools",
    "cmake",
    "CMakeLists.txt",
    "CMakePresets.json",
]
CODE_PATHS = {"library": LIBRARY_PATHS, "tests": [*LIBRARY_PATHS, "cpp/tests"]}
# Library sources that no measured program reads: the algorithm registry's metadata (the manifests
# and the table scripts/regen.py generates from them). A measured commit whose
# library differs from the release's only in these files still certifies the release's
# performance gates, golden replays and golden mutations (never a test-suite check, whose tests
# read them) when equivalence.json shows, for both commits built in the same place with the same
# VERSION, that the compat tools are byte-identical and libdyng differs only inside `functions`,
# with every section and every symbol at the same address (`certify.py equivalence`).
METADATA_PATHS = {
    "cpp/src/core/registry_table.inc": {
        "what": "the algorithm registry table generated from the manifests by scripts/regen.py "
        "(names, titles, maturity, backends, citation keys), read only by dyng::algorithms()",
        "functions": ["dyng::algorithms()", "dyng::algorithms() [clone .cold]"],
    },
    **{
        f"cpp/src/algorithms/{name}/manifest.toml": {
            "what": f"the manifest of {name} (PLAN 4.8), read only by scripts/regen.py, which "
            "generates the registry table from it; no build reads it",
            "functions": [],
        }
        for name in ("sssp", "cycle_count")
    },
}
# What `equivalence` builds and compares: the measured programs and the library they load.
EQUIVALENCE_PRESETS = ["parity", "parity-cuda"]
EQUIVALENCE_TARGETS = ["dyng-compat-mosp", "dyng-compat-cycle-enum"]
EQUIVALENCE_FILES = [
    "cpp/libdyng.so",
    "tools/compat/dyng-compat-mosp",
    "tools/compat/dyng-compat-cycle-enum",
]
# Sections that may differ besides the listed functions' code: the build id, the unwind tables
# (a listed function's frame range and call-site table) and the symbol tables (its size, which the
# symbol comparison restricts to the listed functions).
EQUIVALENCE_FREE_SECTIONS = {
    ".note.gnu.build-id",
    ".eh_frame",
    ".eh_frame_hdr",
    ".gcc_except_table",
    ".symtab",
    ".dynsym",
}
# The golden sets of parity/goldens.toml every 0.1 certificate must replay, and the backends
# each must cover (a replay record names its configurations; "cuda" matches cuda, cuda/int64, ...).
REQUIRED_REPLAYS = {
    "sssp": ["sequential", "openmp", "cuda"],
    "cycle_count": ["sequential", "openmp"],
    "cycle_count_cuda": ["cuda"],
}
# "100% tests passed out of 595" (CMake 4) or "95% tests passed, 1 tests failed out of 20".
CTEST = re.compile(r"(\d+)% tests passed(?:, (\d+) tests? failed)? out of (\d+)")
BEGIN = "<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->"
END = "<!-- certify:end -->"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def head_commit() -> tuple[str, bool]:
    head = git("rev-parse", "HEAD")
    dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]).returncode != 0
    return head, dirty


def resolve(commit: str) -> str | None:
    """The full SHA of `commit` (a "+dirty" suffix ignored); None if it is not a known commit."""
    base = commit.split("+", 1)[0]
    if not re.fullmatch(r"[0-9a-f]{7,40}", base):
        return None
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def code_diff(commit: str, head: str, scope: str = "library") -> list[str] | None:
    """The files of CODE_PATHS[scope] that differ between `commit` and `head`; None if unknown."""
    base = resolve(commit)
    if base is None:
        return None
    out = git("diff", "--name-only", base, head, "--", *CODE_PATHS[scope])
    return sorted(out.splitlines())


def equivalent(commit: str, head: str, diff: list[str], results: Path) -> dict | None:
    """The passed equivalence.json pair that covers `commit` -> `head` with exactly `diff`."""
    path = results / "equivalence.json"
    base = resolve(commit)
    if not diff or base is None or not set(diff) <= set(METADATA_PATHS) or not path.is_file():
        return None
    for pair in json.loads(path.read_text()).get("pairs", []):
        if (
            pair.get("passed") is True
            and resolve(pair.get("measured", "")) == base
            and sorted(pair.get("differing_paths", [])) == diff
            and code_diff(pair.get("release", ""), head) == []
        ):
            return pair
    return None


def same_code(
    commit: str, head: str, scope: str = "library", results: Path | None = None
) -> bool | None:
    """True if `commit` has the sources of `head` in CODE_PATHS[scope] (or, in the library scope
    with `results`, differs only in METADATA_PATHS with a passed equivalence record there);
    None if unknown."""
    diff = code_diff(commit, head, scope)
    if diff is None:
        return None
    if not diff:
        return True
    return (
        scope == "library"
        and results is not None
        and equivalent(commit, head, diff, results) is not None
    )


def tool_version(argv: list[str], line: int = 0) -> str | None:
    if not shutil.which(argv[0]) and not Path(argv[0]).is_file():
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.splitlines()
    return lines[line] if lines else None


def environment() -> dict:
    """The machine and the toolchain, without host or user names."""
    cpu = platform.processor()
    try:
        for row in open("/proc/cpuinfo"):
            if row.startswith("model name"):
                cpu = row.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory_gb = None
    try:
        for row in open("/proc/meminfo"):
            if row.startswith("MemTotal"):
                memory_gb = round(int(row.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    gpus, driver, cuda_driver = [], None, None
    if shutil.which("nvidia-smi"):
        query = "--query-gpu=index,name,driver_version,memory.total,compute_cap"
        out = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True
        ).stdout
        for row in out.strip().splitlines():
            idx, name, drv, mem, cap = [x.strip() for x in row.split(",")]
            gpus.append({"index": int(idx), "name": name, "memory": mem, "compute_capability": cap})
            driver = drv
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        cuda_driver = m.group(1) if m else None
    nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc") or "/usr/local/cuda-13.1/bin/nvcc"
    os_name = None
    try:
        for row in open("/etc/os-release"):
            if row.startswith("PRETTY_NAME="):
                os_name = row.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return {
        "hardware": {
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "memory_gb": memory_gb,
            "gpus": gpus,
        },
        "os": {"name": os_name, "kernel": platform.release()},
        "driver": {"nvidia": driver, "cuda_driver_api": cuda_driver},
        "toolchain": {
            "cxx": tool_version(["/usr/bin/c++", "--version"]),
            "nvcc": tool_version([nvcc, "--version"], -1),
            "cmake": tool_version(["cmake", "--version"]),
            "python": platform.python_version(),
        },
    }


def originals() -> list[dict]:
    doc = tomllib.loads((REPO / "parity" / "references.toml").read_text())
    return [
        {
            "name": r["name"],
            "commit": r["commit"],
            "baseline_tag": r.get("baseline_tag"),
            "baseline_commit": r.get("baseline_commit"),
            "upstream": r.get("upstream"),
            "export_patch": r.get("export_patch") or None,
            "toolchain": r.get("toolchain"),
            "use": "performance baselines from the unpatched git-archive copy; "
            "goldens from the patched copy",
        }
        for r in doc["reference"]
    ]


# --- Golden replays ------------------------------------------------------------------------------


def replay_set(record: dict) -> str:
    return record.get("set") or record.get("algorithm")


def replay_matrix(record: dict) -> tuple[dict, int, int]:
    """(matrix, compared, failed) of a compare.py record, as {row: {config: verdict}}."""
    matrix, compared, failed = {}, 0, 0
    for row, configs in record.get("matrix", {}).items():
        matrix[row] = {}
        for config, value in configs.items():
            if isinstance(value, dict):  # sssp: per group {cases, pass, fail}
                n, ok = value.get("cases", 0), value.get("pass", 0)
                compared += n
                failed += n - ok
                matrix[row][config] = f"{ok}/{n}" + (
                    f" failed: {value['fail']}" if value.get("fail") else ""
                )
            else:  # cycle_count: per case "equal" | "DIFFERENT" | ...
                compared += 1
                failed += value != "equal"
                matrix[row][config] = value
    return matrix, compared, failed


def golden_suites(results: Path, head: str, problems: list[str]) -> list[dict]:
    doc = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    replays: dict[str, list[dict]] = {}
    for path in sorted(results.glob("parity-*.json")):
        record = json.loads(path.read_text())
        replays.setdefault(replay_set(record), []).append({"file": path.name, "record": record})
    out = []
    for name, spec in doc["sets"].items():
        entry = {
            "set": name,
            "reference": spec["reference"],
            "commit": spec["commit"],
            "generated_by": spec.get("generated_by"),
            "cases": spec["num_cases"],
            "files": spec["num_files"],
            "manifest_sha256": spec["manifest_sha256"],
            "case_sha256": {case: c["sha256"] for case, c in spec.get("cases", {}).items()},
            "replays": [],
        }
        covered = set()
        for rep in replays.get(name, []):
            record = rep["record"]
            matrix, compared, failed = replay_matrix(record)
            goldens = record.get("goldens", {})
            ok = bool(record.get("passed")) and failed == 0
            if goldens.get("manifest_sha256") != spec["manifest_sha256"]:
                problems.append(
                    f"{rep['file']}: replayed manifest {goldens.get('manifest_sha256')} "
                    f"is not goldens.toml's {spec['manifest_sha256']}"
                )
                ok = False
            commit = (record.get("port") or {}).get("commit", "unknown")
            code = same_code(commit, head, "library", results)
            if code is not True:
                problems.append(
                    f"{rep['file']}: measured commit {commit} does not have the release's code "
                    f"({'differs' if code is False else 'unknown'})"
                )
                ok = False
            if commit.endswith("+dirty"):
                problems.append(f"{rep['file']}: measured from a dirty tree ({commit})")
                ok = False
            configs = record.get("configs", [])
            covered |= {c.split(":")[0].split("/")[0] for c in configs}
            entry["replays"].append(
                {
                    "record": rep["file"],
                    "date": record.get("date"),
                    "port_commit": commit,
                    "executable": record.get("executable"),
                    "build": record.get("build"),
                    "reference": record.get("reference"),
                    "configs": configs,
                    "tolerance": record.get("tolerance"),
                    "compared": compared,
                    "failed": failed,
                    "matrix": matrix,
                    "passed": ok,
                }
            )
        missing = [b for b in REQUIRED_REPLAYS.get(name, []) if b not in covered]
        if missing:
            problems.append(f"golden set {name}: no replay on {', '.join(missing)}")
        entry["required_backends"] = REQUIRED_REPLAYS.get(name, [])
        entry["passed"] = (
            bool(entry["replays"]) and not missing and all(r["passed"] for r in entry["replays"])
        )
        out.append(entry)
    return out


# --- Performance ---------------------------------------------------------------------------------


def performance(results: Path, head: str, problems: list[str]) -> list[dict]:
    suites = []
    for path in sorted(results.glob("*.json")):
        if path.name in ("parity.json", "checks.json", "equivalence.json") or path.name.startswith(
            "parity-"
        ):
            continue
        doc = json.loads(path.read_text())
        if (
            not isinstance(doc, dict)
            or "suite" not in doc
            or "verdict" not in doc
            or "readings" not in doc
        ):
            continue  # a harness record (the summaries name them)
        verdict = doc["verdict"]
        ok = bool(verdict.get("passed"))
        for commit in doc.get("port_commits", []):
            code = same_code(commit, head, "library", results)
            if code is not True:
                problems.append(
                    f"{path.name}: measured commit {commit} does not have the release's code"
                )
                ok = False
            if commit.endswith("+dirty"):
                problems.append(f"{path.name}: measured from a dirty tree ({commit})")
                ok = False
        for reading in doc["readings"]:
            rec = reading.get("record")
            if rec and not (results / rec).is_file():
                problems.append(f"{path.name}: the record {rec} is not next to it")
                ok = False
        if not verdict.get("passed"):
            reasons = [
                *verdict.get("exceeded", []),
                *verdict.get("problems", []),
                *verdict.get("missing_or_incomplete", []),
                *verdict.get("provisional", []),
            ]
            problems.append(f"{path.name}: the suite did not pass ({'; '.join(reasons)})")
        gates, recorded = [], []
        for reading in doc["readings"]:
            for region in reading.get("regions", []):
                row = {
                    "reading": reading["reading"],
                    "backend": reading.get("backend"),
                    "clocks": reading.get("clocks"),
                    "case": region["case"],
                    "region": region["region"],
                    "ratio": region.get("ratio"),
                }
                for key in (
                    "original_ms",
                    "port_ms",
                    "original_mib",
                    "port_mib",
                    "gate",
                    "within_gate",
                    "noisy",
                ):
                    if key in region:
                        row[key] = region[key]
                if reading.get("gated") and "gate" in region:
                    gates.append(row)
                elif "gate" in region:
                    recorded.append(row)
        suites.append(
            {
                "suite": doc["suite"],
                "algorithm": doc["algorithm"],
                "summary": path.name,
                "suite_file": doc.get("suite_file"),
                "suite_sha256": doc.get("suite_sha256"),
                "port_commits": doc.get("port_commits"),
                "baselines": doc.get("baselines"),
                "tolerance": doc.get("tolerance"),
                "inputs": doc.get("inputs"),
                "readings": [
                    {
                        k: r.get(k)
                        for k in (
                            "reading",
                            "dataset",
                            "kind",
                            "backend",
                            "clocks",
                            "gated",
                            "record",
                            "date",
                            "protocol",
                            "complete",
                        )
                    }
                    for r in doc["readings"]
                ],
                "gated_regions": verdict.get("gated_regions"),
                "within_gate": verdict.get("within_gate"),
                "ratio_range": verdict.get("ratio_range"),
                "gate_table": gates,
                "recorded_not_gated": recorded,
                "passed": ok,
            }
        )
    return suites


# --- Checks --------------------------------------------------------------------------------------


def parse_ctest(text: str) -> dict | None:
    matches = CTEST.findall(text)
    if not matches:
        return None
    pct, failed, total = matches[-1]
    skipped = 0
    if "The following tests did not run:" in text:
        tail = text.split("The following tests did not run:", 1)[1]
        skipped = len(re.findall(r"^\s*\d+ - .*\(Skipped\)\s*$", tail, re.M))
    return {
        "tests": int(total),
        "failed": int(failed or 0),
        "skipped": skipped,
        "percent_passed": int(pct),
    }


def parse_gpu_summary(text: str) -> dict:
    steps = {}
    for row in text.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] not in ("step", "---") and not set(cells[0]) <= {"-"}:
            steps[cells[0]] = cells[1]
    return steps


def cmd_check(args: argparse.Namespace) -> int:
    results = args.results or REPO / "benchmarks" / "results" / args.version
    results.mkdir(parents=True, exist_ok=True)
    path = results / "checks.json"
    doc = json.loads(path.read_text()) if path.is_file() else {"schema": SCHEMA, "checks": []}
    entry: dict = {
        "name": args.name,
        "command": args.command,
        "commit": args.commit or head_commit()[0],
        "scope": args.scope,
    }
    entry["date"] = args.date or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    passed = True
    if args.ctest_log:
        counts = parse_ctest(Path(args.ctest_log).read_text(errors="replace"))
        if counts is None:
            raise SystemExit(f"{args.ctest_log}: no ctest summary line")
        entry["ctest"] = counts
        passed &= counts["failed"] == 0 and counts["tests"] > 0
    if args.gpu_summary:
        steps = parse_gpu_summary(Path(args.gpu_summary).read_text())
        if not steps:
            raise SystemExit(f"{args.gpu_summary}: no step table")
        entry["steps"] = steps
        passed &= all(v in ("passed", "skipped") for v in steps.values())
        if args.require_steps:
            for step in args.require_steps.split(","):
                if steps.get(step) != "passed":
                    entry.setdefault("missing_steps", []).append(step)
                    passed = False
    if args.json_verdict:
        verdict = json.loads(Path(args.json_verdict).read_text())
        entry["record"] = Path(args.json_verdict).name
        passed &= verdict.get("passed") is True
    if args.result:
        passed &= args.result == "passed"
    if not (args.ctest_log or args.gpu_summary or args.json_verdict or args.result):
        raise SystemExit("check: give --ctest-log, --gpu-summary, --json-verdict or --result")
    if args.details:
        entry["details"] = args.details
    entry["result"] = "passed" if passed else "FAILED"
    doc["checks"] = [c for c in doc["checks"] if c["name"] != args.name] + [entry]
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{args.name}: {entry['result']}")
    return 0 if passed else 1


def checks(results: Path, head: str, problems: list[str]) -> list[dict]:
    path = results / "checks.json"
    if not path.is_file():
        problems.append("checks.json is missing (sanitizers, mutation checks: certify.py check)")
        return []
    out = json.loads(path.read_text())["checks"]
    for c in out:
        code = same_code(c.get("commit", ""), head, c.get("scope", "tests"), results)
        c["release_code"] = code is True
        if c.get("result") != "passed":
            problems.append(f"check {c['name']}: {c.get('result')}")
        if code is not True:
            problems.append(
                f"check {c['name']}: commit {c.get('commit')} does not have the release's "
                f"{c.get('scope', 'tests')} sources"
            )
    return out


# --- Equivalence of builds ----------------------------------------------------------------------


def elf_sections(path: Path) -> list[dict]:
    """The section headers of an ELF file (readelf -SW): name, type, address, offset, size,
    flags."""
    out = subprocess.run(["readelf", "-SW", str(path)], capture_output=True, text=True, check=True)
    rows = []
    pattern = (
        r"\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+[0-9a-f]+"
        r"\s+([A-Za-z]*)\s"
    )
    for line in out.stdout.splitlines():
        m = re.match(pattern, line)
        if m and m.group(1) != "0":
            rows.append(
                {
                    "name": m.group(2),
                    "type": m.group(3),
                    "addr": int(m.group(4), 16),
                    "offset": int(m.group(5), 16),
                    "size": int(m.group(6), 16),
                    "flags": m.group(7),
                }
            )
    return rows


def elf_symbols(path: Path) -> list[tuple[int, int, str, str]]:
    """The defined symbols with a size (nm -C -S): (address, size, type, demangled name)."""
    out = subprocess.run(
        ["nm", "-C", "-S", "--defined-only", str(path)], capture_output=True, text=True, check=True
    )
    rows = []
    for line in out.stdout.splitlines():
        m = re.match(r"([0-9a-f]+) ([0-9a-f]+) (\S) (.*)$", line)
        if m:
            rows.append((int(m.group(1), 16), int(m.group(2), 16), m.group(3), m.group(4)))
    return sorted(rows)


def differing_offsets(a: bytes, b: bytes, chunk: int = 4096) -> list[int]:
    """The offsets at which two equally long byte strings differ."""
    out = []
    for start in range(0, len(a), chunk):
        if a[start : start + chunk] != b[start : start + chunk]:
            out += [
                start + i
                for i, (x, y) in enumerate(
                    zip(a[start : start + chunk], b[start : start + chunk], strict=True)
                )
                if x != y
            ]
    return out


def compare_elf(
    a: Path,
    b: Path,
    allowed: set[str],
    *,
    sections=elf_sections,
    symbols=elf_symbols,
) -> dict:
    """Compare a measured build's file `a` with the release build's `b`: "identical" (the same
    bytes), "equivalent" (every section and symbol at the same address, symbol sizes changed only
    for `allowed` functions, and every differing byte inside an allowed function's code or a
    section of EQUIVALENCE_FREE_SECTIONS) or "different" with the reasons."""
    da, db = a.read_bytes(), b.read_bytes()
    entry: dict = {
        "sha256_measured": hashlib.sha256(da).hexdigest(),
        "sha256_release": hashlib.sha256(db).hexdigest(),
    }
    if da == db:
        return {**entry, "result": "identical"}
    reasons: list[str] = []
    if len(da) != len(db):
        reasons.append(f"sizes differ ({len(da)} / {len(db)} bytes)")
    sa, sb = sections(a), sections(b)
    key = [(s["name"], s["addr"], s["offset"], s["size"]) for s in sa]
    if key != [(s["name"], s["addr"], s["offset"], s["size"]) for s in sb]:
        reasons.append("the section layout differs")
    ya, yb = symbols(a), symbols(b)
    if [(x[0], x[2], x[3]) for x in ya] != [(x[0], x[2], x[3]) for x in yb]:
        reasons.append("the symbols differ in name, address or kind")
    else:
        resized = sorted({x[3] for x, y in zip(ya, yb, strict=True) if x[1] != y[1]} - allowed)
        if resized:
            reasons.append(f"functions changed size: {', '.join(resized[:5])}")
    functions: set[str] = set()
    counts: dict[str, int] = {}
    if not reasons:
        code = [x for x in ya + yb if x[2] in "tTwWi" and x[1] > 0]
        for off in differing_offsets(da, db):
            sec = next(
                (
                    s
                    for s in sa
                    if s["type"] != "NOBITS" and s["offset"] <= off < s["offset"] + s["size"]
                ),
                None,
            )
            name = sec["name"] if sec else "(outside every section)"
            counts[name] = counts.get(name, 0) + 1
            if sec and "X" in sec["flags"]:
                vaddr = sec["addr"] + off - sec["offset"]
                owner = next((x[3] for x in code if x[0] <= vaddr < x[0] + x[1]), None)
                functions.add(owner or f"(no function at {vaddr:#x})")
            elif name not in EQUIVALENCE_FREE_SECTIONS:
                functions.add(f"(data in {name})")
        outside = sorted(functions - allowed)
        if outside:
            reasons.append(f"bytes differ outside the allowed functions: {', '.join(outside[:5])}")
    entry.update(
        {
            "result": "different" if reasons else "equivalent",
            "differing_bytes_by_section": counts,
            "differing_functions": sorted(functions),
        }
    )
    if reasons:
        entry["reasons"] = reasons
    return entry


def cmd_equivalence(args: argparse.Namespace) -> int:
    """Build the measured commit and, in the same clone, the release's library paths over it
    (everything else, VERSION included, stays the measured commit's), then compare the files."""
    results = args.results or REPO / "benchmarks" / "results" / args.version
    results.mkdir(parents=True, exist_ok=True)
    head, dirty = head_commit()
    measured = resolve(args.measured)
    if measured is None:
        raise SystemExit(f"{args.measured}: not a commit of this repository")
    diff = code_diff(measured, head)
    assert diff is not None
    unlisted = [d for d in diff if d not in METADATA_PATHS]
    if unlisted:
        raise SystemExit(
            f"the library differs outside METADATA_PATHS ({', '.join(unlisted)}): measure again"
        )
    clone = args.clone.resolve()

    def run(*cmd: str) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=clone, check=True)

    if subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise SystemExit(f"{clone}: not a clean clone")
    allowed = {f for path in diff for f in METADATA_PATHS[path]["functions"]}
    files: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for state in ("measured", "release"):
            run("git", "checkout", "-q", "--detach", measured)
            if state == "release":
                run("git", "checkout", "-q", head, "--", *LIBRARY_PATHS)
            for preset in args.presets.split(","):
                run("cmake", "--preset", preset)
                run("cmake", "--build", "--preset", preset, "--target", *EQUIVALENCE_TARGETS)
                for rel in EQUIVALENCE_FILES:
                    src = (clone / "build" / preset / rel).resolve()
                    dst = Path(tmp) / state / preset / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        run("git", "checkout", "-q", "--detach", measured)
        run("git", "checkout", "-q", measured, "--", ".")
        for preset in args.presets.split(","):
            for rel in EQUIVALENCE_FILES:
                a = Path(tmp) / "measured" / preset / rel
                b = Path(tmp) / "release" / preset / rel
                files.append({"preset": preset, "file": rel, **compare_elf(a, b, allowed)})
    passed = all(f["result"] in ("identical", "equivalent") for f in files)
    pair = {
        "measured": measured,
        "release": head + ("+dirty" if dirty else ""),
        "differing_paths": diff,
        "metadata": {path: METADATA_PATHS[path] for path in diff},
        "presets": args.presets.split(","),
        "how": "both built in one clone with the measured commit's VERSION and build system; the "
        "release state is the measured commit with the release's library paths checked out",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "passed": passed and not dirty,
    }
    path = results / "equivalence.json"
    doc = json.loads(path.read_text()) if path.is_file() else {"schema": SCHEMA, "pairs": []}
    doc["pairs"] = [
        p for p in doc["pairs"] if (p["measured"], p["release"]) != (measured, pair["release"])
    ] + [pair]
    path.write_text(json.dumps(doc, indent=1) + "\n")
    for f in files:
        extra = (
            f" ({', '.join(f.get('differing_functions', []))})"
            if f["result"] != "identical"
            else ""
        )
        print(f"{f['result']:10} {f['preset']}/{f['file']}{extra}")
        for r in f.get("reasons", []):
            print(f"           - {r}")
    if dirty:
        print("the release tree is dirty: the pair is recorded as not passed")
    print(f"equivalence {measured[:12]} -> {head[:12]}: {'passed' if pair['passed'] else 'FAILED'}")
    return 0 if pair["passed"] else 1


# --- README --------------------------------------------------------------------------------------


def fmt_ratio(rows: list[dict]) -> str:
    vals = [r["ratio"] for r in rows if r.get("ratio") is not None]
    return f"{min(vals):.3f}-{max(vals):.3f}" if vals else "-"


def readme_block(cert: dict) -> str:
    lines = [BEGIN, ""]
    v = cert["verdict"]
    lines.append(
        f"**Certificate:** `parity.json` (dynG {cert['version']}, commit `{cert['commit'][:12]}`): "
        + ("**all parts passed**." if v["passed"] else "**FAILED**: " + "; ".join(v["problems"]))
    )
    lines.append("")
    env = cert["environment"]
    gpu = env["hardware"]["gpus"][0]["name"] if env["hardware"]["gpus"] else "no GPU"
    lines.append(
        f"Machine: {env['hardware']['cpu']} ({env['hardware']['logical_cpus']} threads), "
        f"{len(env['hardware']['gpus'])}x {gpu}; driver {env['driver']['nvidia']}; "
        f"{env['toolchain']['nvcc']}; {env['toolchain']['cxx']}."
    )
    for e in cert.get("equivalences", []):
        files = ", ".join(f"{k} {v}" for k, v in e["files"].items())
        lines += [
            "",
            f"Measured at `{e['measured'][:12]}`, whose library differs from the release's only in "
            f"generated metadata ({', '.join(f'`{p}`' for p in e['differing_paths'])}); "
            f"`equivalence.json` compares the builds of both: {files}.",
        ]
    lines += [
        "",
        "### Golden parity",
        "",
        "| Golden set | Original | Cases | Replays (configurations: compared / failed) | Result |",
        "|---|---|---:|---|---|",
    ]
    for g in cert["golden_suites"]:
        reps = (
            "; ".join(
                f"{', '.join(r['configs'])}: {r['compared']} / {r['failed']}" for r in g["replays"]
            )
            or "none"
        )
        lines.append(
            f"| `{g['set']}` | {g['reference']}@{g['commit'][:7]} | {g['cases']} | {reps} "
            f"| {'passed' if g['passed'] else 'FAILED'} |"
        )
    lines += ["", "### Performance gates (PLAN 8.6, against the unpatched originals)", ""]
    lines += [
        "| Suite | Reading | Backend (clocks) | Gated regions | Ratio range (port / original) "
        "| Result |",
        "|---|---|---|---:|---|---|",
    ]
    for s in cert["performance"]:
        by_reading: dict[str, list[dict]] = {}
        for row in s["gate_table"]:
            by_reading.setdefault(row["reading"], []).append(row)
        for reading, rows in by_reading.items():
            within = sum(1 for r in rows if r.get("within_gate"))
            clocks = f" ({rows[0]['clocks']})" if rows[0].get("clocks") else ""
            lines.append(
                f"| `{s['suite']}` | {reading} | {rows[0]['backend']}{clocks} "
                f"| {within} / {len(rows)} | {fmt_ratio(rows)} "
                f"| {'passed' if within == len(rows) else 'FAILED'} |"
            )
        rec: dict[str, list[dict]] = {}
        for row in s["recorded_not_gated"]:
            rec.setdefault(row["reading"], []).append(row)
        for reading, rows in rec.items():
            clocks = f" ({rows[0]['clocks']})" if rows[0].get("clocks") else ""
            lines.append(
                f"| `{s['suite']}` | {reading} | {rows[0]['backend']}{clocks} "
                f"| recorded, not gated ({len(rows)}) | {fmt_ratio(rows)} | - |"
            )
    lines += ["", "### Checks", "", "| Check | Command | Result |", "|---|---|---|"]
    for c in cert["checks"]:
        detail = ""
        if "ctest" in c:
            t = c["ctest"]
            skipped = f", {t['skipped']} skipped" if t.get("skipped") else ""
            detail = f" ({t['tests'] - t['failed']} / {t['tests']} tests{skipped})"
        elif "steps" in c:
            detail = " (" + ", ".join(f"{k} {v}" for k, v in c["steps"].items()) + ")"
        lines.append(f"| {c['name']} | `{c['command']}` | {c['result']}{detail} |")
    lines += ["", END]
    return "\n".join(lines)


def update_readme(path: Path, block: str) -> None:
    text = path.read_text() if path.is_file() else ""
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = head + block + tail
    else:
        text = (text.rstrip() + "\n\n" if text else "") + block + "\n"
    path.write_text(text)


# --- write ---------------------------------------------------------------------------------------


def cmd_write(args: argparse.Namespace) -> int:
    results = args.results or REPO / "benchmarks" / "results" / args.version
    if not results.is_dir():
        raise SystemExit(f"{results}: no results directory")
    head, dirty = head_commit()
    problems: list[str] = []
    if dirty and not args.allow_dirty:
        problems.append("the release tree is dirty (commit first, or --allow-dirty for a draft)")
    goldens = golden_suites(results, head, problems)
    perf = performance(results, head, problems)
    if not perf:
        problems.append("no performance-suite summary (parity/bench_suite.py run)")
    algorithms = {s["algorithm"] for s in perf}
    for algo in ("sssp", "cycle_count"):
        if algo not in algorithms:
            problems.append(f"no performance suite of {algo}")
    checklist = checks(results, head, problems)
    measured = sorted(
        {r["port_commit"] for g in goldens for r in g["replays"]}
        | {c for s in perf for c in (s["port_commits"] or [])}
        | {c["commit"] for c in checklist}
    )
    version_file = (REPO / "VERSION").read_text().strip()
    equivalences = []
    for commit in measured:
        diff = code_diff(commit, head)
        pair = equivalent(commit, head, diff or [], results)
        if pair is not None:
            equivalences.append(
                {
                    "measured": pair["measured"],
                    "release": pair["release"],
                    "differing_paths": pair["differing_paths"],
                    "files": {f"{f['preset']}/{f['file']}": f["result"] for f in pair["files"]},
                    "record": "equivalence.json",
                }
            )
    cert = {
        "schema": SCHEMA,
        "what": "dynG parity certificate (PLAN 8.3): golden parity with the pinned originals, "
        "the performance gates of PLAN 8.6 and the release checks",
        "version": args.version,
        "version_file": version_file,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": head + ("+dirty" if dirty else ""),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "measured_commits": measured,
        "code_paths": CODE_PATHS,
        "code_rule": "every measured commit has the release commit's sources: the library scope "
        "for the performance gates, the golden replays and the golden mutations, the tests scope "
        "for the test-suite checks (git diff <measured> <release> -- code_paths[scope] is empty); "
        "in the library scope, a difference confined to metadata_paths is accepted when "
        "equivalence.json shows the measured programs unchanged (equivalences)",
        "metadata_paths": METADATA_PATHS,
        "equivalences": equivalences,
        "environment": environment(),
        "originals": originals(),
        "golden_suites": goldens,
        "performance": perf,
        "checks": checklist,
    }
    cert["verdict"] = {
        "golden_parity": bool(goldens) and all(g["passed"] for g in goldens),
        "performance_gates": bool(perf) and all(s["passed"] for s in perf),
        "checks": bool(checklist) and all(c.get("result") == "passed" for c in checklist),
        "problems": problems,
    }
    cert["verdict"]["passed"] = not problems and all(
        cert["verdict"][k] for k in ("golden_parity", "performance_gates", "checks")
    )
    (results / "parity.json").write_text(json.dumps(cert, indent=1) + "\n")
    update_readme(results / "README.md", readme_block(cert))
    print(f"{results / 'parity.json'}: {'PASSED' if cert['verdict']['passed'] else 'FAILED'}")
    for p in problems:
        print(f"  {p}")
    return 0 if cert["verdict"]["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command_name", required=True)
    w = sub.add_parser("write", help="assemble parity.json and the README tables")
    w.add_argument("--version", default=(REPO / "VERSION").read_text().strip())
    w.add_argument("--results", type=Path, default=None)
    w.add_argument("--allow-dirty", action="store_true", help="a draft from an uncommitted tree")
    e = sub.add_parser(
        "equivalence",
        help="compare the builds of a measured commit and of the release (METADATA_PATHS only)",
    )
    e.add_argument("--version", default=(REPO / "VERSION").read_text().strip())
    e.add_argument("--results", type=Path, default=None)
    e.add_argument("--measured", required=True, help="the commit whose builds were measured")
    e.add_argument(
        "--clone", required=True, type=Path, help="a clean clone with both commits (it is built)"
    )
    e.add_argument("--presets", default=",".join(EQUIVALENCE_PRESETS))
    c = sub.add_parser("check", help="record one release check in checks.json")
    c.add_argument("--version", default=(REPO / "VERSION").read_text().strip())
    c.add_argument("--results", type=Path, default=None)
    c.add_argument("--name", required=True)
    c.add_argument("--command", required=True, help="the command that was run (recorded)")
    c.add_argument("--commit", help="the commit whose build was checked (default HEAD)")
    c.add_argument("--date", help="when it ran (ISO 8601; default now)")
    c.add_argument(
        "--scope",
        choices=sorted(CODE_PATHS),
        default="tests",
        help="what the check depends on: tests (a test suite: library and C++ tests) or library "
        "(a build of the library and tools, e.g. mutate.py)",
    )
    c.add_argument("--ctest-log", type=Path)
    c.add_argument("--gpu-summary", type=Path, help="ci/gpu_local.sh's gpu_local_summary.md")
    c.add_argument("--require-steps", help="with --gpu-summary: steps that must have passed")
    c.add_argument(
        "--json-verdict", type=Path, help='a JSON record with a top-level "passed" (mutate.py)'
    )
    c.add_argument("--result", choices=["passed", "failed"])
    c.add_argument("--details")
    args = parser.parse_args(argv)
    commands = {"write": cmd_write, "check": cmd_check, "equivalence": cmd_equivalence}
    return commands[args.command_name](args)


if __name__ == "__main__":
    sys.exit(main())
