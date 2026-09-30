#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The executable code snippets of the README files and the documentation (PLAN Section 9.6).

A snippet is a fenced code block preceded by a marker comment; an optional second block, preceded
by an output marker, holds what the snippet prints:

    <!-- snippet: quickstart-python -->
    ```python
    import dyng
    ...
    ```
    <!-- snippet-output: quickstart-python -->
    ```text
    [0, 4, 1, 2] 2
    ```

The comments are invisible on GitHub, on PyPI and in the Sphinx site. Who runs the snippets:

- the Python snippets: python/tests/test_doc_snippets.py (pytest, so python.yml and ci/python.sh)
  executes every one and compares its standard output with the output block, if there is one;
- the C++ quickstart of README.md: examples/cpp/CMakeLists.txt extracts it at configure time
  (cmake/dyng_doc_snippet.cmake) and CTest builds, runs and compares it (the cpu workflow and
  ci/check.sh);
- this script, from ci/docs.sh: `--check` verifies that the README quickstarts exist and keep
  PLAN 9.6's size (10 lines of Python; 10 lines in the body of the C++ `main()`), that every
  snippet has a language the runners know, and that every output block belongs to a snippet.

Usage:
    ci/doc_snippets.py --check              # the structural check (no dyng needed)
    ci/doc_snippets.py --list               # every snippet with its file and line
    ci/doc_snippets.py --run-python         # run the Python snippets (needs `import dyng`)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Files searched for snippets: the two READMEs and every documentation page.
SOURCES = ("README.md", "python/README.md", "docs/**/*.md")
#: The quickstarts PLAN 9.6 requires, with their language and line budget.
QUICKSTARTS = {"quickstart-python": ("python", 10), "quickstart-cpp": ("cpp", 10)}
RUNNABLE = ("python", "cpp")

_MARKER = re.compile(r"^<!--\s*snippet(-output)?:\s*([A-Za-z0-9_.-]+)\s*-->\s*$")
_FENCE = re.compile(r"^(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)")


@dataclass
class Snippet:
    """One marked code block."""

    name: str
    lang: str
    code: str
    path: Path
    line: int  # 1-based line of the marker
    expected: str | None = None


def _fenced_block(lines: list[str], start: int, path: Path) -> tuple[str, str, int]:
    """The fenced block that starts at lines[start] (after blank lines): (lang, text, end)."""
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    m = _FENCE.match(lines[i]) if i < len(lines) else None
    if not m:
        raise SystemExit(f"{path}:{start}: a snippet marker must be followed by a fenced block")
    fence, lang = m.group(1), m.group(2)
    body: list[str] = []
    for j in range(i + 1, len(lines)):
        if lines[j].startswith(fence) and not lines[j][len(fence) :].strip():
            return lang, "\n".join(body) + "\n", j
        body.append(lines[j])
    raise SystemExit(f"{path}:{i + 1}: unterminated code block")


def snippets_of(path: Path) -> list[Snippet]:
    """The snippets of one file, in order, with their output blocks attached."""
    lines = path.read_text(encoding="utf-8").splitlines()
    found: dict[str, Snippet] = {}
    i = 0
    while i < len(lines):
        m = _MARKER.match(lines[i])
        if not m:
            i += 1
            continue
        is_output, name = bool(m.group(1)), m.group(2)
        lang, text, end = _fenced_block(lines, i + 1, path)
        if is_output:
            if name not in found:
                raise SystemExit(f"{path}:{i + 1}: output block of unknown snippet '{name}'")
            found[name].expected = text
        else:
            if name in found:
                raise SystemExit(f"{path}:{i + 1}: snippet '{name}' is defined twice")
            found[name] = Snippet(name, lang, text, path, i + 1)
        i = end + 1
    return list(found.values())


def all_snippets(root: Path = REPO_ROOT) -> list[Snippet]:
    """Every snippet of the README files and the documentation."""
    out: list[Snippet] = []
    for pattern in SOURCES:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                out += snippets_of(path)
    return out


def find(name: str, root: Path = REPO_ROOT) -> Snippet:
    """The snippet called `name` (names are unique per file; README.md is searched first)."""
    for s in all_snippets(root):
        if s.name == name:
            return s
    raise KeyError(name)


def code_lines(snippet: Snippet) -> int:
    """The size PLAN 9.6 counts: non-blank lines (Python); the body of main() (C++)."""
    lines = [ln for ln in snippet.code.splitlines() if ln.strip()]
    if snippet.lang != "cpp":
        return len(lines)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("int main("))
    except StopIteration:
        return len(lines)
    body = lines[start + 1 :]
    if body and body[-1].strip() == "}":
        body = body[:-1]
    return len(body)


def check(root: Path = REPO_ROOT) -> list[str]:
    """Problems with the snippets (an empty list when all is well)."""
    problems: list[str] = []
    snippets = all_snippets(root)
    readme = [s for s in snippets if s.path == root / "README.md"]
    for name, (lang, budget) in QUICKSTARTS.items():
        match = [s for s in readme if s.name == name]
        if not match:
            problems.append(f"README.md: the quickstart snippet '{name}' is missing")
            continue
        s = match[0]
        if s.lang != lang:
            problems.append(f"README.md:{s.line}: '{name}' must be a {lang} block, not {s.lang}")
        if code_lines(s) > budget:
            problems.append(
                f"README.md:{s.line}: '{name}' has {code_lines(s)} lines; PLAN 9.6 allows {budget}"
            )
    for s in snippets:
        if s.lang not in RUNNABLE:
            problems.append(
                f"{s.path.relative_to(root)}:{s.line}: snippet '{s.name}' is {s.lang!r}"
            )
        if s.lang == "cpp" and s.name != "quickstart-cpp":
            problems.append(
                f"{s.path.relative_to(root)}:{s.line}: only README.md's quickstart-cpp is built; "
                "quote other C++ code from examples/cpp with literalinclude"
            )
    return problems


def run_python(snippet: Snippet, *, python: str = sys.executable, timeout: float = 120) -> str:
    """Run a Python snippet in a fresh interpreter in an empty directory; its standard output."""
    with tempfile.TemporaryDirectory(prefix="dyng-snippet-") as tmp:
        proc = subprocess.run(
            [python, "-c", snippet.code],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{snippet.path}:{snippet.line}: snippet '{snippet.name}' failed "
            f"(exit {proc.returncode}):\n{proc.stderr}"
        )
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="structural check (ci/docs.sh)")
    g.add_argument("--list", action="store_true", help="list the snippets")
    g.add_argument("--run-python", action="store_true", help="run the Python snippets")
    args = ap.parse_args(argv)

    if args.list:
        for s in all_snippets():
            out = " (with output)" if s.expected is not None else ""
            print(f"{s.path.relative_to(REPO_ROOT)}:{s.line}: {s.name} [{s.lang}]{out}")
        return 0
    if args.check:
        problems = check()
        for p in problems:
            print(f"doc-snippets: {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"doc-snippets: OK ({len(all_snippets())} snippets)")
        return 0
    failed = 0
    for s in all_snippets():
        if s.lang != "python":
            continue
        try:
            out = run_python(s)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            failed += 1
            continue
        if s.expected is not None and out != s.expected:
            print(
                f"{s.path}:{s.line}: '{s.name}' printed\n{out}expected\n{s.expected}",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"ok  {s.path.relative_to(REPO_ROOT)}:{s.line} {s.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
