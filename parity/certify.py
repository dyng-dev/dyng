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
from fnmatch import fnmatchcase
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMA = 1
# What a measured build is made of (the library, the compat tools, the build system, the
# bibliography compiled in for dyng::citation()), and what a test-suite check runs in addition
# (the C++ tests). The performance gates, the golden replays and the golden mutations need the
# release's library sources; a test-suite check also its tests. VERSION is deliberately not a
# library path: it only names the build (dyng::version(), the package version), and a release
# candidate's measurements certify the final release whose VERSION differs only there
# (docs/developer/release.md, step 9).
LIBRARY_PATHS = [
    "cpp/include",
    "cpp/src",
    "cpp/CMakeLists.txt",
    "tools",
    "cmake",
    "CMakeLists.txt",
    "CMakePresets.json",
    "docs/references.bib",
]
TEST_PATHS = [*LIBRARY_PATHS, "cpp/tests"]
# What the distributions are built from and checked with: the sdist's sources (the library, the
# tests, the Python package, the metadata files and the licences) and the scripts and workflows
# that build and check them (release.yml's select, wheels.yml, ci/wheels.sh, ci/wheel_check.py).
PACKAGING_PATHS = [
    *TEST_PATHS,
    "python",
    "pyproject.toml",
    "VERSION",
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSES",
    "NOTICE",
    "THIRD_PARTY_LICENSES.txt",
    "ci/wheels.sh",
    "ci/wheel_check.py",
    ".github/workflows/release.yml",
    ".github/workflows/wheels.yml",
]
# A check of the whole tree (ci/check.sh, ci/gpu_local.sh: the scripts, the docs, the harness):
# every tracked file but the results of the releases, which are committed after the checks.
REPO_PATHS = [".", ":(exclude)benchmarks/results"]
CODE_PATHS = {
    "library": LIBRARY_PATHS,
    "tests": TEST_PATHS,
    "packaging": PACKAGING_PATHS,
    "repo": REPO_PATHS,
}
SCOPE_HELP = {
    "library": "a build of the library and the tools (the gates, the replays, mutate.py)",
    "tests": "a test suite of the C++ tests (the sanitizer presets, ctest -L mutation)",
    "packaging": "the distributions (select, ci/wheels.sh, twine, wheel_check, the venvs)",
    "repo": "a check of the whole tree (ci/check.sh, ci/gpu_local.sh)",
}
# The committed fixtures of cpp/tests/data: where each set comes from and the tests comparing it.
FIXTURES = REPO / "parity" / "fixtures" / "fixtures.toml"
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
# One test of ctest's progress output: "  3/595 Test   #1: Name ......   Passed    0.09 sec"
# (or "***Failed", "***Skipped", "***Not Run", "***Timeout", "***Exception: ...").
CTEST_TEST = re.compile(
    r"^\s*\d+/\d+ Test\s+#\d+: (?P<name>.+?)\s(?:\.+\s*)?(?:\*\*\*)?(?P<status>Passed|Failed|"
    r"Skipped|Not Run|Timeout|Exception[^\d]*?|SEGFAULT|Subprocess aborted|Child aborted)"
    r"\s+[\d.]+ sec\s*$",
    re.M,
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The lines of an evidence file kept in its committed excerpt: step headers and result lines.
EXCERPT = re.compile(
    r"^(==>|== |###|\| |ok |rc=|Checking|.*\btests passed\b|.*\d+ passed\b|.*: OK\b|.*\bPASSED\b|"
    r".*\bFAILED\b|.*\bfailed\b|.*all (checks|steps) passed|kind=|version=|prerelease=|.*refused)"
)
EXCERPT_LINES = 400
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


def file_changed(path: Path) -> datetime.datetime:
    """The last time `path` (not following a symbolic link) was written or replaced: the later
    of its modification and status-change times (a package install sets the latter)."""
    st = path.lstat()
    return datetime.datetime.fromtimestamp(max(st.st_mtime, st.st_ctime), datetime.UTC)


def driver_unchanged_since(start: datetime.datetime, version: str | None) -> dict:
    """Whether the NVIDIA driver `version` seen now is the one that ran at `start`: the loaded
    kernel module has that version, the machine booted before `start`, the module file on disk
    (the only one any load since boot can have used) has that version and was installed before
    `start`, and so was the user-space driver library libcuda.so.1 (the CUDA driver API), whose
    file name carries the version."""
    found: dict = {"since": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "driver": version}
    reasons: list[str] = []

    def stamp(t: datetime.datetime) -> str:
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        loaded = Path("/sys/module/nvidia/version").read_text().strip()
    except OSError:
        loaded = None
    found["loaded_module"] = loaded
    if version is None or loaded != version:
        reasons.append(f"the loaded module is {loaded}, nvidia-smi reports {version}")
    try:
        btime = next(int(row.split()[1]) for row in open("/proc/stat") if row.startswith("btime"))
        boot = datetime.datetime.fromtimestamp(btime, datetime.UTC)
        found["boot"] = stamp(boot)
        if boot >= start:
            reasons.append(f"the machine booted at {stamp(boot)}, after {stamp(start)}")
    except (OSError, StopIteration, ValueError):
        reasons.append("no boot time (/proc/stat)")
    modules = sorted(Path("/lib/modules", platform.release()).rglob("nvidia.ko"))
    if len(modules) != 1:
        reasons.append(f"{len(modules)} nvidia.ko files under /lib/modules/{platform.release()}")
    for module in modules[:1]:
        m = re.search(rb"(?:^|\0)version=([0-9][0-9.]*)\0", module.read_bytes())
        on_disk = m.group(1).decode() if m else None
        found["module_file"] = {
            "path": str(module),
            "version": on_disk,
            "installed": stamp(file_changed(module)),
        }
        if on_disk != loaded:
            reasons.append(f"{module} is version {on_disk}, the loaded module {loaded}")
        if file_changed(module) >= start:
            reasons.append(f"{module} was installed after {stamp(start)}")
    lib = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if lib.exists():
        target = lib.resolve()
        found["libcuda"] = {
            "link": str(lib),
            "target": str(target),
            "installed": stamp(max(file_changed(lib), file_changed(target))),
        }
        if version is None or not target.name.endswith(version):
            reasons.append(f"{target.name} is not the library of driver {version}")
        if max(file_changed(lib), file_changed(target)) >= start:
            reasons.append(f"{lib} or {target} was installed after {stamp(start)}")
    else:
        reasons.append(f"no {lib}")
    found["verified"] = not reasons
    if reasons:
        found["reasons"] = reasons
    return found


def measured_driver(perf: list[dict], now: dict, problems: list[str]) -> dict:
    """The driver of the measurements: the one the records name (all the same, and the one seen
    now when a driver is visible), or, for CUDA records that predate the field, the driver seen
    now when driver_unchanged_since shows it ran since before the first of them started."""
    named: dict[str, list[str]] = {}
    unnamed: list[tuple[str, str]] = []
    for s in perf:
        for r in s["readings"]:
            if r.get("driver"):
                named.setdefault(json.dumps(r["driver"], sort_keys=True), []).append(r["record"])
            elif r.get("backend") == "cuda" or r.get("kind") == "memory":
                start = s.get("execution_started") or r.get("date")
                unnamed.append((start, r.get("record")))
    out: dict = {"write_time": now}
    if len(named) > 1:
        problems.append(f"the records name different drivers: {sorted(named)}")
    if named:
        driver = json.loads(next(iter(named)))
        out.update(driver)
        out["source"] = f"named by {sum(len(v) for v in named.values())} records"
        if now.get("nvidia") and driver != now:
            problems.append(f"the records' driver {driver} is not the one seen now ({now})")
    if unnamed:
        start = min(parse_time(t) for t, _ in unnamed if t)
        found = driver_unchanged_since(start, now.get("nvidia"))
        out["records_without_driver"] = len(unnamed)
        out["unchanged_since_first_measurement"] = found
        if not found["verified"]:
            problems.append(
                f"{len(unnamed)} CUDA records name no driver and the driver seen now is not shown "
                f"to have run since {found['since']}: {'; '.join(found.get('reasons', []))}"
            )
        if named and json.loads(next(iter(named))) != now:
            problems.append("records with and without a driver: the named one is not today's")
        if not named:
            out.update(now)
            out["source"] = (
                "seen when the certificate was written; the records predate the per-record "
                "driver field, and the driver is shown unchanged since before the first of them "
                "(unchanged_since_first_measurement)"
            )
    if not named and not unnamed:
        out.update(now)
        out["source"] = "seen when the certificate was written (no CUDA records)"
    return out


def parse_time(text: str) -> datetime.datetime:
    stamp = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.UTC)).astimezone(datetime.UTC)


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


# --- Committed fixtures --------------------------------------------------------------------------


def tree_digest(directory: str) -> tuple[str, int]:
    """(SHA-256 of the sorted "<sha256>  <path>" lines of the tracked files under `directory`,
    number of files), the digest rule of parity/export_goldens.py's case digests."""
    files = sorted(git("ls-files", "-z", "--", directory).split("\0"))
    lines = []
    for rel in (f for f in files if f):
        sha = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        lines.append(f"{sha}  {Path(rel).relative_to(directory).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines)


def committed_fixtures(checklist: list[dict], problems: list[str]) -> list[dict]:
    """Every fixture set of cpp/tests/data (fixtures.toml) with its digest, the original it was
    exported from and the results of the tests that compare it in the release checks."""
    doc = tomllib.loads(FIXTURES.read_text())
    refs = {
        r["name"]: r
        for r in tomllib.loads((REPO / "parity" / "references.toml").read_text())["reference"]
    }
    listed = {Path(spec["directory"]).as_posix() for spec in doc["sets"].values()}
    tracked = {
        Path(f).parts[3]
        for f in git("ls-files", "--", "cpp/tests/data").splitlines()
        if len(Path(f).parts) > 4
    }
    for name in sorted(tracked):
        if f"cpp/tests/data/{name}" not in listed:
            problems.append(f"cpp/tests/data/{name}: a fixture set fixtures.toml does not list")
    out = []
    for name, spec in doc["sets"].items():
        digest, count = tree_digest(spec["directory"])
        ref = refs.get(spec["reference"], {})
        entry = {
            "set": name,
            "directory": spec["directory"],
            "reference": spec["reference"],
            "commit": ref.get("commit"),
            "generated_by": spec["generated_by"],
            "what": spec["what"],
            "files": count,
            "sha256": digest,
            "tests": [],
        }
        ok = count > 0 and bool(ref) and (REPO / spec["generated_by"]).is_file()
        if not ok:
            problems.append(f"fixture set {name}: no files, unknown original or no generator")
        for pattern in spec["tests"]:
            runs = []
            for c in checklist:
                matched = {
                    n: st for n, st in c.get("fixture_tests", {}).items() if fnmatchcase(n, pattern)
                }
                if matched:
                    runs.append(
                        {
                            "check": c["name"],
                            "release_code": c.get("release_code", False),
                            "tests": matched,
                        }
                    )
            passed_somewhere = any(
                r["release_code"] and "Passed" in r["tests"].values() for r in runs
            )
            failed = [
                f"{r['check']}: {n} {st}"
                for r in runs
                for n, st in r["tests"].items()
                if st not in ("Passed", "Skipped", "Not Run")
            ]
            result = "passed" if passed_somewhere and not failed else "FAILED"
            if not passed_somewhere:
                problems.append(
                    f"fixture set {name}: no release check ran and passed a test {pattern}"
                )
            for f in failed:
                problems.append(f"fixture set {name}: {f}")
            entry["tests"].append({"pattern": pattern, "result": result, "runs": runs})
            ok &= result == "passed"
        entry["passed"] = ok
        out.append(entry)
    return out


# --- Performance ---------------------------------------------------------------------------------


def suite_coverage(doc: dict, where: str) -> list[str]:
    """What keeps a summary from covering its whole suite: a suite file that is not the
    committed one, a partial execution, or a planned reading x dataset that is missing,
    repeated, incomplete or without its record."""
    sys.path.insert(0, str(REPO / "parity"))
    import bench_suite

    rel = doc.get("suite_file")
    path = REPO / rel if rel else None
    if path is None or not path.is_file():
        return [f"{where}: its suite file {rel} is not in the repository"]
    if bench_suite.sha256_file(path) != doc.get("suite_sha256"):
        return [f"{where}: {rel} is not the suite that was summarized (SHA-256 differs)"]
    out = []
    if doc.get("partial"):
        out.append(f"{where}: a partial execution ({doc['partial']}), not the whole suite")
    suite = bench_suite.load_suite(path)
    if suite.get("suite") != doc.get("suite") or suite.get("algorithm") != doc.get("algorithm"):
        out.append(f"{where}: names suite {doc.get('suite')}, the file is {suite.get('suite')}")
    planned = [
        (j["reading"], j["dataset"])
        for j in bench_suite.plan(suite, records=Path("records"), build_root=Path("build"))
    ]
    got = [(r.get("reading"), r.get("dataset")) for r in doc.get("readings", [])]
    for key in planned:
        label = key[0] + (f"/{key[1]}" if key[1] else "")
        n = got.count(key)
        if n == 0:
            out.append(f"{where}: the planned reading {label} is missing")
        elif n > 1:
            out.append(f"{where}: the reading {label} appears {n} times")
    for key in sorted(set(got) - set(planned), key=str):
        out.append(f"{where}: the reading {key} is not in the suite's plan")
    for r in doc.get("readings", []):
        label = f"{r.get('reading')}" + (f"/{r['dataset']}" if r.get("dataset") else "")
        if not r.get("record"):
            out.append(f"{where}: the reading {label} has no record")
        if r.get("complete") is not True:
            out.append(f"{where}: the reading {label} is not complete")
    return out


def performance(results: Path, head: str, problems: list[str]) -> list[dict]:
    suites = []
    for path in sorted(results.glob("*.json")):
        if path.name in ("parity.json", "checks.json", "equivalence.json") or path.name.startswith(
            "parity-"
        ):
            continue
        if path.name.endswith(".partial.json"):
            problems.append(f"{path.name}: a partial execution's summary in the release results")
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
        for problem in suite_coverage(doc, path.name):
            problems.append(problem)
            ok = False
        inputs = doc.get("inputs")
        if not (isinstance(inputs, dict) and inputs.get("verified") is True):
            problems.append(f"{path.name}: the inputs of the readings are not verified")
            ok = False
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
                            "driver",
                        )
                    }
                    | {"inputs": (r.get("inputs") or {}).get("how")}
                    for r in doc["readings"]
                ],
                "execution_started": doc.get("execution_started"),
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


def parse_ctest_tests(text: str) -> dict[str, str]:
    """{test name: status} of every test in ctest's progress output (the last run of a name
    wins); the status is Passed, Failed, Skipped, Not Run, Timeout or the failure kind."""
    out: dict[str, str] = {}
    for m in CTEST_TEST.finditer(ANSI.sub("", text)):
        out[m["name"].strip()] = m["status"].strip()
    return out


def fixture_patterns() -> list[str]:
    doc = tomllib.loads(FIXTURES.read_text())
    return sorted({t for spec in doc["sets"].values() for t in spec["tests"]})


def fixture_tests(tests: dict[str, str]) -> dict[str, str]:
    """The tests of `tests` that compare a committed fixture set (fixtures.toml)."""
    patterns = fixture_patterns()
    return {n: st for n, st in sorted(tests.items()) if any(fnmatchcase(n, p) for p in patterns)}


def portable(path: Path) -> str:
    """A path relative to the repository or $DYNG_SCRATCH; never a personal absolute path."""
    p = path.resolve()
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    for base, label in [(REPO, None), (scratch.resolve(), "$DYNG_SCRATCH")]:
        if p.is_relative_to(base):
            rel = p.relative_to(base).as_posix()
            return rel if label is None else f"{label}/{rel}"
    return p.name


def evidence(path: Path, results: Path, name: str, index: int) -> dict:
    """The record of one evidence file of a check: where it is, its SHA-256 and size, and a
    committed excerpt (the step headers and result lines, at most EXCERPT_LINES) next to
    checks.json under checks/."""
    data = path.read_bytes()
    text = ANSI.sub("", data.decode(errors="replace"))
    lines = [line.rstrip() for line in text.splitlines() if EXCERPT.match(line.strip())]
    rel = Path("checks") / f"{name}{'' if index == 0 else f'-{index}'}.txt"
    (results / rel).parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Excerpt of {portable(path)} (sha256 {hashlib.sha256(data).hexdigest()}, "
        f"{len(data)} bytes): its step headers and result lines, written by "
        "parity/certify.py check.",
    ]
    body = lines[:EXCERPT_LINES]
    if len(lines) > EXCERPT_LINES:
        body.append(f"# ... {len(lines) - EXCERPT_LINES} more lines in the full file")
    (results / rel).write_text("\n".join(header + body) + "\n")
    return {
        "file": portable(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "excerpt": rel.as_posix(),
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
    if not (args.ctest_log or args.gpu_summary or args.json_verdict or args.result):
        raise SystemExit("check: give --ctest-log, --gpu-summary, --json-verdict or --result")
    if args.result and not args.evidence:
        raise SystemExit("check: --result needs --evidence (the log of what was run)")
    passed = True
    files = list(args.evidence or [])
    tests: dict[str, str] = {}
    if args.ctest_log:
        text = Path(args.ctest_log).read_text(errors="replace")
        counts = parse_ctest(text)
        if counts is None:
            raise SystemExit(f"{args.ctest_log}: no ctest summary line")
        entry["ctest"] = counts
        passed &= counts["failed"] == 0 and counts["tests"] > 0
        tests.update(parse_ctest_tests(text))
        files.insert(0, args.ctest_log)
    for log in args.tests_log or []:
        tests.update(parse_ctest_tests(Path(log).read_text(errors="replace")))
        files.append(log)
    if args.gpu_summary:
        steps = parse_gpu_summary(Path(args.gpu_summary).read_text())
        if not steps:
            raise SystemExit(f"{args.gpu_summary}: no step table")
        entry["steps"] = steps
        required = [x for x in (args.require_steps or "").split(",") if x]
        entry["required_steps"] = required
        passed &= all(v in ("passed", "skipped") for v in steps.values())
        for step in required:
            if steps.get(step) != "passed":
                entry.setdefault("missing_steps", []).append(step)
                passed = False
        files.append(args.gpu_summary)
    if args.json_verdict:
        verdict = json.loads(Path(args.json_verdict).read_text())
        entry["record"] = Path(args.json_verdict).name
        passed &= verdict.get("passed") is True
    if args.result:
        passed &= args.result == "passed"
    if tests:
        failed = sorted(n for n, st in tests.items() if st not in ("Passed", "Skipped", "Not Run"))
        passed &= not failed
        if failed:
            entry["failed_tests"] = failed
        entry["fixture_tests"] = fixture_tests(tests)
    seen: list[Path] = []
    for f in files:
        if Path(f).resolve() not in seen:
            seen.append(Path(f).resolve())
    entry["evidence"] = [evidence(f, results, args.name, i) for i, f in enumerate(seen)]
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
        f"{len(env['hardware']['gpus'])}x {gpu}; driver {env['driver'].get('nvidia')} "
        f"(CUDA driver API {env['driver'].get('cuda_driver_api')}; "
        f"{env['driver'].get('source', 'seen when the certificate was written')}); "
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
    lines += [
        "",
        "### Committed fixtures (`cpp/tests/data`, `parity/fixtures/fixtures.toml`)",
        "",
        "| Fixture set | Original | Files | SHA-256 | Tests (checks) | Result |",
        "|---|---|---:|---|---|---|",
    ]
    for f in cert.get("committed_fixtures", []):
        names = sorted({n for t in f["tests"] for r in t["runs"] for n in r["tests"]})
        where = sorted({r["check"] for t in f["tests"] for r in t["runs"]})
        lines.append(
            f"| `{f['set']}` | {f['reference']}@{(f['commit'] or '')[:7]} | {f['files']} "
            f"| `{f['sha256'][:16]}...` | {len(f['tests'])} patterns, {len(names)} tests "
            f"({', '.join(where)}) | {'passed' if f['passed'] else 'FAILED'} |"
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
    lines += ["", "Inputs of the readings:", ""]
    for s in cert["performance"]:
        hows = sorted({r.get("inputs") or "not verified" for r in s["readings"]})
        lines.append(f"- `{s['suite']}`: {'; '.join(hows)}.")
    lines += [
        "",
        "### Checks",
        "",
        "| Check | Command | Scope | Result |",
        "|---|---|---|---|",
    ]
    for c in cert["checks"]:
        detail = ""
        if "ctest" in c:
            t = c["ctest"]
            skipped = f", {t['skipped']} skipped" if t.get("skipped") else ""
            detail = f" ({t['tests'] - t['failed']} / {t['tests']} tests{skipped})"
        elif "steps" in c:
            detail = " (" + ", ".join(f"{k} {v}" for k, v in c["steps"].items()) + ")"
        scope = c.get("scope", "tests")
        lines.append(f"| {c['name']} | `{c['command']}` | {scope} | {c['result']}{detail} |")
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
    fixtures = committed_fixtures(checklist, problems)
    env = environment()
    env["driver"] = measured_driver(perf, env["driver"], problems)
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
        "code_rule": "every measured commit has the release commit's sources (git diff "
        "<measured> <release> -- code_paths[scope] is empty): the library scope for the "
        "performance gates, the golden replays and the golden mutations; for a check, the scope "
        "it was recorded with (tests: a C++ test suite; packaging: the distributions; repo: a "
        "check of the whole tree but benchmarks/results). In the library scope only, a "
        "difference confined to metadata_paths is accepted when equivalence.json shows the "
        "measured programs unchanged (equivalences)",
        "metadata_paths": METADATA_PATHS,
        "equivalences": equivalences,
        "environment": env,
        "originals": originals(),
        "golden_suites": goldens,
        "committed_fixtures": fixtures,
        "performance": perf,
        "checks": checklist,
    }
    cert["verdict"] = {
        "golden_parity": bool(goldens) and all(g["passed"] for g in goldens),
        "committed_fixtures": bool(fixtures) and all(f["passed"] for f in fixtures),
        "performance_gates": bool(perf) and all(s["passed"] for s in perf),
        "checks": bool(checklist) and all(c.get("result") == "passed" for c in checklist),
        "problems": problems,
    }
    cert["verdict"]["passed"] = not problems and all(
        cert["verdict"][k]
        for k in ("golden_parity", "committed_fixtures", "performance_gates", "checks")
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
        help="what the check depends on (the paths of CODE_PATHS that must be the release's): "
        + "; ".join(f"{k}: {v}" for k, v in SCOPE_HELP.items()),
    )
    c.add_argument("--ctest-log", type=Path, help="ctest's output: counts and per-test results")
    c.add_argument(
        "--tests-log",
        type=Path,
        action="append",
        help="another log with ctest output (e.g. ci/gpu_local.sh's): per-test results",
    )
    c.add_argument(
        "--evidence",
        type=Path,
        action="append",
        help="a log of what was run (required with --result): its SHA-256 is recorded and an "
        "excerpt committed under checks/",
    )
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
