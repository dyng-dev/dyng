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
the API reference page docs/api/cpp/<name>.md, then runs scripts/regen.py (the algorithm tables,
CODEOWNERS and the registries). Next:

    cmake --preset dev && cmake --build --preset dev && ctest --preset dev -L <name>
    ci/docs.sh --update-api       # the new header joins the public API baseline (as `tracked`)

`--remove` undoes a scaffold: it deletes only an algorithm whose manifest still carries the
scaffold line (the template's "scaffolded by scripts/new_algorithm.py"; `--force` overrides), and
it puts back the [[planned]] entry of cpp/src/algorithms/planned.toml that the scaffold replaced.

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
# The scaffold line of the template's manifest.toml (remove() deletes only such algorithms).
SCAFFOLD_LINE = "# scaffolded by scripts/new_algorithm.py"
# The [[planned]] entry a scaffold replaced, kept as comments at the end of its manifest.toml.
PLANNED_MARK = "# planned entry replaced by this scaffold (restored by --remove), index "
# C++ keywords and alternative tokens (a namespace dyng::<name> must compile).
CPP_KEYWORDS = set(
    """alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t
    char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue
    co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit
    export extern false float for friend goto if inline int long mutable namespace new noexcept not
    not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires
    return short signed sizeof static static_assert static_cast struct switch template this
    thread_local throw true try typedef typeid typename union unsigned using virtual void volatile
    wchar_t while xor xor_eq""".split()
)
# Headers generated at configure time (cpp/include/dyng/<stem>.hpp.in): a header of that name would
# shadow them.
GENERATED_HEADERS = {"version", "config"}
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
        f"docs/api_{TEMPLATE}.md": f"docs/api/cpp/{name}.md",
    }


def dyng_entities(root: Path) -> set[str]:
    """Names declared at namespace scope of dyng (types, enumerations, aliases, namespaces) and the
    header stems of cpp/include/dyng: a namespace dyng::<name> cannot reuse them."""
    names: set[str] = set()
    include = root / "cpp/include/dyng"
    declaration = re.compile(
        r"^(?:template\s*<[^;{]*>\s*)?(?:class|struct|union|enum(?:\s+class)?|using)\s+"
        r"([a-z_][a-z0-9_]*)\b",
        re.MULTILINE,
    )
    namespace = re.compile(r"^namespace\s+dyng::([a-z_][a-z0-9_]*)", re.MULTILINE)
    for header in sorted(include.rglob("*.hpp*")):
        text = header.read_text(encoding="utf-8")
        names.update(declaration.findall(text))
        names.update(namespace.findall(text))
        if header.parent == include:
            names.add(header.name.split(".")[0])
    return names


def name_problem(root: Path, name: str, removing: bool) -> str | None:
    """Why `name` cannot be an algorithm's name (None if it can)."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return "not snake_case"
    if keyword.iskeyword(name):
        return "a Python keyword"
    if name in RESERVED:
        return "reserved by the library"
    if name in CPP_KEYWORDS:
        return "a C++ keyword"
    if name in GENERATED_HEADERS:
        return f"the generated header <dyng/{name}.hpp>"
    if not removing and name in dyng_entities(root):
        return "already the name of a declaration or header of namespace dyng"
    return None


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


UNRELEASED = "## [Unreleased]\n\n"
ADDED = "### Added\n\n"


def add_changelog(root: Path, line: str) -> None:
    """Add `line` under "### Added" of the CHANGELOG's "## [Unreleased]" section, creating the
    subsection when the section has none (as after a release); exit if there is no such
    section."""
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    if line in text:
        return
    if UNRELEASED not in text:
        raise SystemExit(f"{path}: no '## [Unreleased]' section for the entry (--no-changelog)")
    head, rest = text.split(UNRELEASED, 1)
    if rest.startswith(ADDED):
        rest = ADDED + line + rest[len(ADDED) :]
    else:
        rest = ADDED + line + "\n" + rest
    path.write_text(head + UNRELEASED + rest, encoding="utf-8")


def remove_changelog(root: Path, line: str) -> None:
    """Remove `line`, and the "### Added" subsection of "## [Unreleased]" if it is left empty."""
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8").replace(line, "")
    text = re.sub(
        r"(?m)^(## \[Unreleased\]\n\n)### Added\n\n(?:\n(?=## |### )|(?=## |### ))",
        r"\1",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def _planned_blocks(text: str) -> list[str]:
    """planned.toml as [header, block, block, ...]."""
    return re.split(r"(?m)^(?=\[\[planned\]\])", text)


def _join_planned(blocks: list[str]) -> str:
    """planned.toml from its blocks: the header, then the entries separated by one blank line."""
    entries = [b.rstrip("\n") + "\n" for b in blocks[1:]]
    return blocks[0] + "\n".join(entries)


def drop_planned(root: Path, name: str) -> tuple[int, str] | None:
    """Remove the [[planned]] entry of `name` (its port starts now); its index and text, or None."""
    path = root / "cpp/src/algorithms/planned.toml"
    if not path.is_file():
        return None
    blocks = _planned_blocks(path.read_text(encoding="utf-8"))
    for i, block in enumerate(blocks[1:], start=1):
        if re.search(rf'(?m)^name\s*=\s*"{re.escape(name)}"', block):
            del blocks[i]
            path.write_text(_join_planned(blocks), encoding="utf-8")
            return i, block.rstrip("\n") + "\n"
    return None


def keep_planned(manifest: Path, entry: tuple[int, str]) -> None:
    """Record a replaced [[planned]] entry as comments at the end of the new manifest."""
    index, block = entry
    lines = [f"{PLANNED_MARK}{index}:\n"] + [
        f"# {line}".rstrip() + "\n" for line in block.splitlines()
    ]
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.rstrip("\n") + "\n" + "".join(lines), encoding="utf-8")


def restore_planned(root: Path, manifest_text: str) -> bool:
    """Put back the [[planned]] entry a manifest recorded (keep_planned); True if there was one."""
    lines = manifest_text.splitlines()
    for at, line in enumerate(lines):
        if line.startswith(PLANNED_MARK):
            index = int(line[len(PLANNED_MARK) :].rstrip(":"))
            block = "".join(entry[2:] + "\n" for entry in lines[at + 1 :] if entry.startswith("# "))
            path = root / "cpp/src/algorithms/planned.toml"
            blocks = _planned_blocks(path.read_text(encoding="utf-8"))
            blocks.insert(min(index, len(blocks)), block)
            path.write_text(_join_planned(blocks), encoding="utf-8")
            return True
    return False


def format_sources(root: Path, files: list[Path]) -> None:
    """clang-format the generated C++ files (the markers' removal changes their alignment)."""
    tool = shutil.which("clang-format")
    if tool is None:
        print("new_algorithm.py: clang-format not found; run pre-commit on the new files")
        return
    subprocess.check_call([tool, "-i", "--style=file", *map(str, files)], cwd=root)


API_INDEX = "docs/api/cpp/index.md"


def add_api_page(root: Path, name: str) -> None:
    """List docs/api/cpp/<name>.md in the C++ API toctree (after the last algorithm page)."""
    path = root / API_INDEX
    text = path.read_text(encoding="utf-8")
    if f"\n{name}\n" in text:
        return
    anchor = "\ngenerators\n"  # the algorithm pages come before the generators page
    if anchor not in text:
        raise SystemExit(f"new_algorithm.py: {API_INDEX} has no `generators` toctree entry")
    path.write_text(text.replace(anchor, f"\n{name}{anchor}", 1), encoding="utf-8")


def remove_api_page(root: Path, name: str) -> None:
    path = root / API_INDEX
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(f"\n{name}\n", "\n", 1), encoding="utf-8")


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
    add_api_page(root, name)
    if not args.no_changelog:
        add_changelog(root, changelog_line(name, args.family))
    planned = drop_planned(root, name)
    if planned is not None:
        keep_planned(targets["manifest.toml"], planned)
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
    print(
        f"then: ci/docs.sh --update-api   (<dyng/{name}.hpp> joins the public API baseline "
        "cpp/tests/api/api_snapshot/public_api.txt; commit it with the algorithm)"
    )
    return 0


def remove(args: argparse.Namespace, root: Path) -> int:
    name = args.name
    manifest = root / f"cpp/src/algorithms/{name}/manifest.toml"
    if not manifest.is_file():
        print(f"new_algorithm.py: `{name}` has no manifest", file=sys.stderr)
        return 2
    manifest_text = manifest.read_text(encoding="utf-8")
    if SCAFFOLD_LINE not in manifest_text and not args.force:
        print(
            f"new_algorithm.py: `{name}` is not a scaffold (its manifest has no "
            f"`{SCAFFOLD_LINE}` line); refusing to delete it (--force overrides)",
            file=sys.stderr,
        )
        return 2
    family = tomllib.loads(manifest_text).get("family", "")
    for dst in destinations(name).values():
        path = root / dst
        if path.is_file():
            path.unlink()
    for folder in (f"cpp/src/algorithms/{name}", f"cpp/tests/algorithms/{name}"):
        if (root / folder).is_dir():
            shutil.rmtree(root / folder)
    remove_subdirectory(root, name)
    remove_api_page(root, name)
    restored = restore_planned(root, manifest_text)
    remove_changelog(root, changelog_line(name, family))
    status = 0 if args.no_regen else run_regen(root)
    print(f"new_algorithm.py: removed {name}")
    if restored:
        print("  restored its entry in cpp/src/algorithms/planned.toml")
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
    parser.add_argument(
        "--force", action="store_true", help="--remove also an algorithm that is not a scaffold"
    )
    parser.add_argument("--no-regen", action="store_true", help="do not run scripts/regen.py")
    parser.add_argument("--no-changelog", action="store_true", help="no CHANGELOG entry")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (tests)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    # An existing algorithm's name is reported by create() ("exists already") or removed.
    existing = (root / f"cpp/src/algorithms/{args.name}").exists()
    problem = name_problem(root, args.name, removing=args.remove or existing)
    if problem is not None:
        parser.error(f"`{args.name}` is not a valid algorithm name: {problem}")
    if args.remove:
        return remove(args, root)
    if args.family is None:
        parser.error("--family is required (fixed_point or aggregate_delta; PLAN Section 9.4)")
    return create(args, root)


if __name__ == "__main__":
    sys.exit(main())
