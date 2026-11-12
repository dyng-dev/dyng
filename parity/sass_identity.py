#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The SASS identity of the 0.3 CUDA objects between two builds (M11a criterion 20; the R030
placement lesson; parity/triad_sass.py's reading).

    parity/sass_identity.py --base <build tree> --head <build tree> [--base-commit SHA]
        [--all] [--json OUT]

For every object of OBJECTS (the escher store's kernels, escher_store.cu, the sorted-merge line
graph and triad_count's kernels) in both build trees (the same preset): every kernel function of
`cuobjdump -sass`, its normalized instruction stream (addresses, comments and its own name
removed: parity/triad_sass.py's digest) and `cuobjdump -res-usage`'s registers, stack, shared and
local memory. Identical when both objects hold the same functions with the same digests and
resources. Exit 1 if an object or a function differs or is missing.

A base built from a `git archive` copy (no .git: its commit cannot be read) names its commit with
--base-commit: the tool checks that the build tree's source tree (CMAKE_HOME_DIRECTORY) holds
exactly that commit's files under ARCHIVED (every archived file byte-identical, no other file
under its directories) and records the commit and the check (exit 2 if it fails). The M11a review
found the record's base commit null.

--all (M11b step 1, ADR 0094) compares every CUDA object of the library and the compat tools
(every `*.cu.o` of the base tree outside the tests) instead of OBJECTS, and each object's host
code as well: `objdump -d` of its sections, with the addresses and the names nvcc derives from
the build (`tmpxft_<pid>_...`, `_INTERNAL_<hash>_`, the anonymous namespaces' hashes) normalized.
Two builds of the same sources in two directories differ in those names only, so a CUDA object is
unchanged when its kernels and its host code are.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import triad_sass  # noqa: E402

# The objects the 0.3 release measured (relative to a build tree; the parity-cuda preset).
HG = "cpp/CMakeFiles/dyng_hypergraph.dir/src/hypergraph"
TRIAD = "cpp/src/algorithms/triad_count/CMakeFiles/dyng_triad_count.dir"
OBJECTS = [
    *(
        f"{HG}/escher/{name}.cu.o"
        for name in (
            "build_tree",
            "cbst_store",
            "compact",
            "delete_avail",
            "find",
            "insert_reuse",
            "integrity",
            "locator",
            "payload",
            "unfill",
        )
    ),
    f"{HG}/escher_store.cu.o",
    f"{HG}/line_graph/sorted_merge.cu.o",
    f"{TRIAD}/cuda.cu.o",
    f"{TRIAD}/engine/counters.cu.o",
]


# The paths of a base's git-archive copy that its build reads (the library's sources and build).
ARCHIVED = ("cpp", "cmake", "CMakeLists.txt", "CMakePresets.json")


def source_tree(build: Path) -> Path:
    """The source tree of a CMake build tree (CMAKE_HOME_DIRECTORY of its cache)."""
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:"):
            return Path(line.split("=", 1)[1])
    raise SystemExit(f"{build}: no CMAKE_HOME_DIRECTORY in CMakeCache.txt")


def check_archive(commit: str, tree: Path) -> dict:
    """Whether `tree` holds exactly `commit`'s files under ARCHIVED (the copy git archive gives)."""
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    data = subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", sha, *ARCHIVED],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    want: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if m.isfile():
                want[m.name] = tar.extractfile(m).read()
    differ = sorted(
        n for n, b in want.items() if not (tree / n).is_file() or (tree / n).read_bytes() != b
    )
    have = {
        str(f.relative_to(tree))
        for d in ARCHIVED
        if (tree / d).is_dir()
        for f in (tree / d).rglob("*")
        if f.is_file()
    }
    extra = sorted(have - set(want))
    return {
        "commit": sha,
        "paths": list(ARCHIVED),
        "files": len(want),
        "identical": not differ and not extra,
        "differ": differ[:20],
        "extra": extra[:20],
    }


# An anonymous namespace's mangled name carries a hash of the translation unit's path
# (_GLOBAL__N__<hash>_<file>_<hash>): two source trees give two names for the same code.
ANONYMOUS = re.compile(r"_GLOBAL__N__[0-9a-f]+_")


def kernels(obj: Path) -> dict[str, dict]:
    """{function: {digest, instructions, resources}} of every kernel function of an object, the
    names with their anonymous-namespace hash removed (in the instructions too)."""
    res = triad_sass.resource_usage(obj)
    out = {}
    for name, code in triad_sass.sass(obj).items():
        stream = "\n".join(
            strip_build_names(ANONYMOUS.sub("_GLOBAL__N__", triad_sass.normalize(ins, name)))
            for _, ins in code
        )
        out[strip_build_names(ANONYMOUS.sub("_GLOBAL__N__", name))] = {
            "digest": hashlib.sha256(stream.encode()).hexdigest(),
            "instructions": len(code),
            "resources": res.get(name, {}),
        }
    return out


# Names nvcc derives from the build (its temporary file's process id, a hash of the path).
BUILD_NAMES = (
    (re.compile(r"tmpxft_[0-9a-f]+_[0-9a-f]+"), "tmpxft"),
    (re.compile(r"_INTERNAL_[0-9a-f]+_"), "_INTERNAL_"),
    (re.compile(r"_GLOBAL__N__[0-9a-f]+_"), "_GLOBAL__N__"),
    # CUDA 13's anonymous namespaces: _GLOBAL__N__<n>_<file>_cu_<hash>_<digits><name>
    (re.compile(r"_cu_[0-9a-f]{8}_\d+"), "_cu_HASH_"),
    (re.compile(r"_[0-9a-f]{8}(_\d+)?(?=\w*cudafe)"), "_HASH"),
)


def strip_build_names(text: str) -> str:
    """`text` with the names nvcc derives from the build normalized (BUILD_NAMES)."""
    for pattern, repl in BUILD_NAMES:
        text = pattern.sub(repl, text)
    return text


def host_code(obj: Path) -> dict:
    """The digest of an object's host instructions (objdump -d), addresses and build-derived
    names normalized."""
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = []
    for line in out.splitlines()[2:]:  # the file name line
        line = re.sub(r"^\s*[0-9a-f]+:\s*", "", line)
        line = re.sub(r"\b[0-9a-f]{6,16}\b", "ADDR", line)
        lines.append(strip_build_names(line))
    stream = "\n".join(lines)
    return {"digest": hashlib.sha256(stream.encode()).hexdigest(), "lines": len(lines)}


def all_cuda_objects(tree: Path) -> list[str]:
    """Every *.cu.o of a build tree outside the tests and the dependencies."""
    return sorted(
        str(o.relative_to(tree))
        for o in tree.rglob("*.cu.o")
        if "/tests/" not in str(o) and "/_deps/" not in str(o)
    )


def compare(base: Path, head: Path, objects_of=None, host=False) -> dict:
    objects = {}
    for rel in objects_of if objects_of is not None else OBJECTS:
        a, b = base / rel, head / rel
        if not a.is_file() or not b.is_file():
            objects[rel] = {
                "identical": False,
                "missing": [str(p) for p in (a, b) if not p.is_file()],
            }
            continue
        ka, kb = kernels(a), kernels(b)
        differ = sorted(n for n in set(ka) | set(kb) if ka.get(n) != kb.get(n))
        host_same = True
        if host:
            ha, hb = host_code(a), host_code(b)
            host_same = ha == hb
            if not host_same:
                differ.append("(host code)")
        objects[rel] = {
            "identical": not differ,
            "functions": len(ka),
            "instructions": sum(k["instructions"] for k in ka.values()),
            "differ": differ,
            "base_sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
            "head_sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
            "digests": {n: k["digest"] for n, k in sorted(ka.items())},
        }
    return objects


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True, help="the merge base's build tree")
    p.add_argument("--head", type=Path, required=True, help="the final library's build tree")
    p.add_argument(
        "--base-commit",
        help="the commit the base's source tree was archived from (checked against it)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="every CUDA object of the base tree (outside the tests), kernels and host code",
    )
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    archive = None
    if args.base_commit:
        archive = check_archive(args.base_commit, source_tree(args.base))
        print(
            f"base source tree {'holds' if archive['identical'] else 'DOES NOT hold'} "
            f"{archive['commit']} ({archive['files']} files)"
        )
        if not archive["identical"]:
            print(f"  differ {archive['differ']} extra {archive['extra']}")
            return 2
    if args.all:
        objects = compare(args.base, args.head, all_cuda_objects(args.base), host=True)
    else:
        objects = compare(args.base, args.head)
    for rel, o in objects.items():
        why = "" if o["identical"] else f" {o.get('differ') or o.get('missing')}"
        verdict = "identical" if o["identical"] else "DIFFERS"
        print(f"{verdict}  {rel} ({o.get('functions', 0)} kernels){why}")
    ok = all(o["identical"] for o in objects.values())
    if args.json:
        doc = {
            "record": (
                "SASS and host-code identity of every CUDA object against the base (M11b, ADR 0094)"
                if args.all
                else "SASS identity of the 0.3 CUDA objects against the merge base (M11a "
                "criterion 20)"
            ),
            "tool": "parity/sass_identity.py",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provenance": {
                "base": triad_sass.provenance([args.base / OBJECTS[0]], False)["objects"][0],
                "head": triad_sass.provenance([args.head / OBJECTS[0]], False)["objects"][0],
                "cuobjdump": triad_sass.provenance([args.head / OBJECTS[0]], False)["cuobjdump"],
            },
            "objects": objects,
            "identical": ok,
        }
        if archive is not None:
            doc["provenance"]["base"]["commit"] = archive["commit"]
            doc["provenance"]["base"]["archive_check"] = archive
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    print("identical" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
