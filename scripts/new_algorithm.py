#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Scaffold a new algorithm from cpp/src/algorithms/_template (PLAN Sections 4.8 and 9.4).

    python3 scripts/new_algorithm.py dynamic_kcore --family fixed_point --backends seq,omp
    python3 scripts/new_algorithm.py dynamic_kcore --remove     # undo a scaffold

The new algorithm compiles and passes the conformance kit on the first build ("start green"): its
update() applies the batch and recomputes from scratch (stats.fallback_used = true). The script
writes the public header, the algorithm folder (manifest, CMakeLists.txt, <name>.cpp, problem.hpp,
sequential.cpp, openmp.cpp with --backends seq,omp), the tests (test_traits, the one-line
conformance suite, hand cases), the docs page, the add_subdirectory() line and a CHANGELOG entry,
then runs scripts/regen.py (the algorithm tables, CODEOWNERS and the registries). Next:

    cmake --preset dev && cmake --build --preset dev && ctest --preset dev -L <name>

In 0.1 the scaffold covers graphs (--container graph) and the host backends; a CUDA backend is
added by hand (the operators arrive with their second user; see sssp and cycle_count), and the
hypergraph container arrives in 0.2.
"""

from __future__ import annotations

import argparse
import keyword
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = "algorithm_template"
BACKEND_NAMES = {
    "seq": "sequential",
    "sequential": "sequential",
    "omp": "openmp",
    "openmp": "openmp",
}
BACKEND_DOC = {"sequential": "sequential", "openmp": "openmp"}
MARKER = re.compile(r"^\s*(?://|#)@@\s+(\S+)\s*$")
RESERVED = {
    "update",
    "compute",
    "detail",
    "framework",
    "testing",
    "core",
    "graph",
    "hypergraph",
    "io",
    "generators",
    "operators",
    "conformance",
    "algorithms",
    "template",
}


def destinations(name: str) -> dict[str, str]:
    """Template file (relative to _template) -> destination (relative to the repository)."""
    tests = f"cpp/tests/algorithms/{name}"
    return {
        f"public/{TEMPLATE}.hpp": f"cpp/include/dyng/{name}.hpp",
        "manifest.toml": f"cpp/src/algorithms/{name}/manifest.toml",
        "CMakeLists.txt": f"cpp/src/algorithms/{name}/CMakeLists.txt",
        f"{TEMPLATE}.cpp": f"cpp/src/algorithms/{name}/{name}.cpp",
        "problem.hpp": f"cpp/src/algorithms/{name}/problem.hpp",
        "sequential.cpp": f"cpp/src/algorithms/{name}/sequential.cpp",
        "openmp.cpp": f"cpp/src/algorithms/{name}/openmp.cpp",
        f"tests/{TEMPLATE}_traits.hpp": f"{tests}/{name}_traits.hpp",
        f"tests/{TEMPLATE}_conformance_test.cpp": f"{tests}/{name}_conformance_test.cpp",
        f"tests/{TEMPLATE}_test.cpp": f"{tests}/{name}_test.cpp",
        "tests/CMakeLists.txt": f"{tests}/CMakeLists.txt",
        f"docs/{TEMPLATE}.md": f"docs/algorithms/{name}.md",
    }


def camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def select(text: str, enabled: set[str], where: str) -> str:
    """Keep the lines of the enabled `//@@ <selector>` ... `//@@ end` blocks; drop the markers."""
    out = []
    stack: list[str] = []
    for line in text.splitlines(keepends=True):
        m = MARKER.match(line)
        if m:
            if m.group(1) == "end":
                if not stack:
                    raise SystemExit(f"new_algorithm.py: {where}: unbalanced `@@ end`")
                stack.pop()
            else:
                stack.append(m.group(1))
            continue
        if all(s in enabled for s in stack):
            out.append(line)
    if stack:
        raise SystemExit(f"new_algorithm.py: {where}: unclosed `@@ {stack[-1]}`")
    return "".join(out)


def render(text: str, name: str, fields: dict[str, str], enabled: set[str], where: str) -> str:
    text = select(text, enabled, where)
    for key, value in fields.items():
        text = text.replace("{{" + key + "}}", value)
    text = text.replace("ALGORITHM_TEMPLATE", name.upper())
    text = text.replace("AlgorithmTemplate", camel(name))
    return text.replace(TEMPLATE, name)


def add_subdirectory(root: Path, name: str) -> None:
    path = root / "cpp/src/algorithms/CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    line = f"add_subdirectory({name})\n"
    if line not in text:
        path.write_text(text.rstrip("\n") + "\n" + line, encoding="utf-8")


def remove_subdirectory(root: Path, name: str) -> None:
    path = root / "cpp/src/algorithms/CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(f"add_subdirectory({name})\n", ""), encoding="utf-8")


def changelog_line(name: str, family: str) -> str:
    return (
        f"- `{name}` (experimental, {family.replace('_', ' ')}): scaffolded with "
        "`scripts/new_algorithm.py`; its update recomputes from scratch until it is made "
        "incremental.\n"
    )


def add_changelog(root: Path, line: str) -> None:
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    anchor = "## [Unreleased]\n\n### Added\n\n"
    if anchor in text and line not in text:
        path.write_text(text.replace(anchor, anchor + line, 1), encoding="utf-8")


def drop_planned(root: Path, name: str) -> bool:
    """Remove the [[planned]] entry of `name` (its port starts now); True if there was one."""
    path = root / "cpp/src/algorithms/planned.toml"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^(?=\[\[planned\]\])", text)
    kept = [b for b in blocks if not re.search(rf'(?m)^name\s*=\s*"{re.escape(name)}"', b)]
    if len(kept) == len(blocks):
        return False
    path.write_text("".join(kept).rstrip("\n") + "\n", encoding="utf-8")
    return True


def format_sources(root: Path, files: list[Path]) -> None:
    """clang-format the generated C++ files (the markers' removal changes their alignment)."""
    tool = shutil.which("clang-format")
    if tool is None:
        print("new_algorithm.py: clang-format not found; run pre-commit on the new files")
        return
    subprocess.check_call([tool, "-i", "--style=file", *map(str, files)], cwd=root)


def run_regen(root: Path) -> int:
    return subprocess.call([sys.executable, str(root / "scripts/regen.py"), "--root", str(root)])


def create(args: argparse.Namespace, root: Path) -> int:
    name = args.name
    if args.container != "graph":
        print(
            "new_algorithm.py: the hypergraph container arrives in 0.2 (PLAN Section 11.3); "
            "scaffold on --container graph",
            file=sys.stderr,
        )
        return 2
    backends: list[str] = []
    for b in args.backends.split(","):
        b = b.strip()
        if b == "cuda":
            print(
                "new_algorithm.py: the scaffold writes the host backends only in 0.1; add "
                "cuda.cu by hand after the host backends pass the kit (docs/developer/"
                "conformance.md)",
                file=sys.stderr,
            )
            return 2
        if b not in BACKEND_NAMES:
            print(f"new_algorithm.py: unknown backend `{b}` (seq, omp)", file=sys.stderr)
            return 2
        if BACKEND_NAMES[b] not in backends:
            backends.append(BACKEND_NAMES[b])
    if "sequential" not in backends:
        backends.insert(0, "sequential")  # invariant I8: the sequential reference is mandatory
    backends.sort(key=["sequential", "openmp"].index)
    template = root / "cpp/src/algorithms/_template"
    targets = {src: root / dst for src, dst in destinations(name).items()}
    if "openmp" not in backends:
        del targets["openmp.cpp"]
    clash = [str(p.relative_to(root)) for p in targets.values() if p.exists()]
    if clash or (root / f"cpp/src/algorithms/{name}").exists():
        print(f"new_algorithm.py: `{name}` exists already: {', '.join(clash)}", file=sys.stderr)
        return 2
    enabled = {args.family} | set(backends)
    title = args.title or f"TODO: the title of {name}"
    fields = {
        "title": title,
        "computes": args.computes or f"TODO: what {name} computes",
        "family": args.family,
        "backends": ", ".join(f'"{b}"' for b in backends),
        "backends_doc": ", ".join(BACKEND_DOC[b] for b in backends),
        "maintainer": args.maintainer,
        "since": args.since,
    }
    for src, dst in targets.items():
        text = (template / src).read_text(encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(render(text, name, fields, enabled, src), encoding="utf-8")
    format_sources(root, [p for p in targets.values() if p.suffix in (".hpp", ".cpp")])
    add_subdirectory(root, name)
    if not args.no_changelog:
        add_changelog(root, changelog_line(name, args.family))
    planned = drop_planned(root, name)
    status = 0 if args.no_regen else run_regen(root)
    if status != 0:
        return status
    print(f"new_algorithm.py: created {name} ({args.family}; {', '.join(backends)})")
    if planned:
        print("  removed its entry from cpp/src/algorithms/planned.toml")
    for dst in targets.values():
        print(f"  {dst.relative_to(root)}")
    print(
        "next: cmake --preset dev && cmake --build --preset dev && "
        f"ctest --preset dev -L {name}   (docs/developer/conformance.md)"
    )
    return 0


def remove(args: argparse.Namespace, root: Path) -> int:
    name = args.name
    manifest = root / f"cpp/src/algorithms/{name}/manifest.toml"
    if not manifest.is_file():
        print(f"new_algorithm.py: `{name}` has no manifest", file=sys.stderr)
        return 2
    family = tomllib.loads(manifest.read_text(encoding="utf-8")).get("family", "")
    for dst in destinations(name).values():
        path = root / dst
        if path.is_file():
            path.unlink()
    for folder in (f"cpp/src/algorithms/{name}", f"cpp/tests/algorithms/{name}"):
        if (root / folder).is_dir():
            shutil.rmtree(root / folder)
    remove_subdirectory(root, name)
    changelog = root / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    changelog.write_text(text.replace(changelog_line(name, family), ""), encoding="utf-8")
    status = 0 if args.no_regen else run_regen(root)
    print(f"new_algorithm.py: removed {name}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="the algorithm's name (snake_case: namespace, header, folder)")
    parser.add_argument("--family", choices=["fixed_point", "aggregate_delta"])
    parser.add_argument("--container", default="graph", choices=["graph", "hypergraph"])
    parser.add_argument("--backends", default="seq", help="seq, or seq,omp (default: seq)")
    parser.add_argument("--title", help="the manifest's title")
    parser.add_argument("--computes", help='the "Computes" column of the algorithm tables')
    parser.add_argument("--maintainer", default="SMShovan", help="GitHub handle (CODEOWNERS)")
    parser.add_argument("--since", default="0.1", help="the first release that contains it")
    parser.add_argument("--remove", action="store_true", help="remove a scaffolded algorithm")
    parser.add_argument("--no-regen", action="store_true", help="do not run scripts/regen.py")
    parser.add_argument("--no-changelog", action="store_true", help="no CHANGELOG entry")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (tests)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if (
        not re.fullmatch(r"[a-z][a-z0-9_]*", args.name)
        or keyword.iskeyword(args.name)
        or (args.name in RESERVED)
    ):
        parser.error(f"`{args.name}` is not a valid algorithm name (snake_case, not reserved)")
    if args.remove:
        return remove(args, root)
    if args.family is None:
        parser.error("--family is required (fixed_point or aggregate_delta; PLAN Section 9.4)")
    return create(args, root)


if __name__ == "__main__":
    sys.exit(main())
