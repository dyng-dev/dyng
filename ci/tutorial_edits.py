#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Apply the edits of the tutorial "Your first dynamic algorithm" to a scaffolded my_bfs.

    python3 ci/tutorial_edits.py [--root DIR]

The tutorial (docs/tutorials/your_first_dynamic_algorithm.md, section 4) has the reader change
the files that `scripts/new_algorithm.py my_bfs ...` wrote: some edits paste a range of the
reference solution (examples/tutorial_algorithms/my_bfs/, the `// [tutorial: <name>]` markers
that the page quotes with literalinclude), the others are spelled out in the prose (replace
`reads_prepared_graph()`, use the frontier type, declare `offer` and `applied_`, assign
`applied_` in `resume`, delete the `fallback_used` line, the hand test). This script does each
edit at the place the prose names, in the scaffold's own files, instead of copying the finished
reference files over them: ci/scaffold_check.sh then builds the result and runs its kit, so a step
the prose gets wrong (a duplicate definition, a missing declaration) fails the check. It does not
update the doc comments that section 4.7 asks the reader to reword (they do not affect the build).

When the tutorial's prose changes, change the matching edit here. Each edit fails loudly when its
anchor in the scaffold is missing or not unique.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REFERENCE = Path("examples/tutorial_algorithms/my_bfs")
ALGORITHM = Path("cpp/src/algorithms/my_bfs")
TEST = Path("cpp/tests/algorithms/my_bfs/my_bfs_test.cpp")


class EditError(RuntimeError):
    """An anchor of an edit is missing or not unique."""


def snippet(path: Path, marker: str) -> str:
    """The lines between `// [tutorial: <marker>]` and `// [tutorial: <marker> end]`."""
    text = path.read_text(encoding="utf-8")
    begin = f"// [tutorial: {marker}]"
    end = f"// [tutorial: {marker} end]"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise EditError(f"{path}: marker {begin!r} missing or repeated")
    start = text.index("\n", text.index(begin)) + 1
    return text[start : text.index(end)]


def one(pattern: str, text: str, what: str) -> re.Match[str]:
    """The single match of `pattern` in `text`."""
    found = list(re.finditer(pattern, text, re.MULTILINE))
    if len(found) != 1:
        raise EditError(f"{what}: {len(found)} matches of the anchor {pattern!r}")
    return found[0]


def function_end(text: str, start: int) -> int:
    """The index just past the closing line `}` of the definition that starts at `start`."""
    return text.index("\n}\n", start) + 3


def edit_problem(problem: str, ref: Path) -> str:
    # 4.1: the workspace members, after `before`.
    m = one(r"^  std::vector<std::int64_t> before;.*\n", problem, "4.1 workspace")
    problem = problem[: m.end()] + snippet(ref / "problem.hpp", "workspace") + problem[m.end() :]
    # 4.2: the frontier, before the problem class; the class uses it.
    m = one(
        r"^/// The update family of my_bfs\.\n",
        problem,
        "4.2 frontier (before the problem class)",
    )
    problem = (
        problem[: m.start()]
        + snippet(ref / "problem.hpp", "frontier")
        + "\n"
        + problem[m.start() :]
    )
    m = one(
        r"^  using frontier_type = framework::internal_frontier;.*\n", problem, "4.2 frontier_type"
    )
    problem = (
        problem[: m.start()]
        + "  using frontier_type = my_bfs_frontier;  "
        + "///< the loop's frontier (the workspace's lists)\n"
        + problem[m.end() :]
    )
    # 4.2: replace the scaffold's reads_prepared_graph() (which returns false).
    m = one(
        r"^  ///.*\n  \[\[nodiscard\]\] bool reads_prepared_graph\(\) const noexcept \{\n"
        r"    return false;\n  \}\n",
        problem,
        "4.2 reads_prepared_graph",
    )
    problem = (
        problem[: m.start()]
        + "  /// The seed reads the in-edges: the commit builds them once for every result.\n"
        + "  [[nodiscard]] bool reads_prepared_graph() const noexcept {\n"
        + "    return true;\n  }\n"
        + problem[m.end() :]
    )
    # 4.3: the hooks replace the declaration of loop under `// ---- Step 2 ----`.
    m = one(r"^  // ---- Step 2 ----\n\n  ///.*\n  void loop\([^;]*;\n", problem, "4.3 loop")
    problem = problem[: m.start()] + snippet(ref / "problem.hpp", "hooks") + problem[m.end() :]
    # 4.3: the private helper and the commit's record.
    m = one(r"^ private:\n", problem, "4.3 private part")
    problem = (
        problem[: m.end()]
        + "  /// Offer the level `from + 1` to `v` (`from`: the level of an in-neighbour, -1 if "
        + "unreached).\n"
        + "  void offer(std::int64_t from, std::int64_t v, frontier_type& f);\n\n"
        + "  const applied* applied_ = nullptr;  "
        + "///< what the commit did (identify_affected, seed)\n"
        + problem[m.end() :]
    )
    return problem


def edit_source(source: str, ref: Path) -> str:
    cpp = ref / "my_bfs.cpp"
    # 4.1: reserve() and bytes() size and count the new members.
    m = one(r"^void my_bfs_workspace::reserve\(", source, "4.1 reserve")
    end = function_end(source, source.index("std::size_t my_bfs_workspace::bytes()", m.start()))
    source = source[: m.start()] + snippet(cpp, "reserve") + source[end:]
    # 4.4: resume names its parameter and remembers it.
    m = one(r"const applied& /\*applied\*/\) \{\n", source, "4.4 resume")
    source = (
        source[: m.start()]
        + "const applied& applied) {\n"
        + "  applied_ = &applied;  // what the commit did: the insertions and deletions the batch "
        + "requested\n"
        + source[m.end() :]
    )
    # 4.4-4.6: identify_affected, offer and seed go before loop; loop is replaced.
    m = one(
        r"^template <typename vertex_t, typename edge_t, typename weight_t>\n"
        r"void my_bfs_problem<vertex_t, edge_t, weight_t>::loop\(",
        source,
        "4.6 loop",
    )
    end = function_end(source, m.start())
    hooks = [snippet(cpp, name) for name in ("identify_affected", "offer", "seed", "loop")]
    source = source[: m.start()] + "\n".join(hooks) + source[end:]
    # 4.7: seed_static runs the static solve.
    m = one(
        r"^template <typename vertex_t, typename edge_t, typename weight_t>\n"
        r"void my_bfs_problem<vertex_t, edge_t, weight_t>::seed_static\(",
        source,
        "4.7 seed_static",
    )
    end = source.index("}\n", source.index("{", m.start())) + 2
    source = source[: m.start()] + snippet(cpp, "seed_static") + source[end:]
    # 4.7: finalize no longer reports a recomputation.
    m = one(r"^  stats\.fallback_used = true;.*\n", source, "4.7 finalize")
    source = source[: m.start()] + source[m.end() :]
    # 4.7: the requirement, at the end of begin_update.
    begin = one(r"::begin_update\(", source, "4.7 begin_update").start()
    end = source.index("\n}\n", begin)
    source = source[:end] + "\n" + snippet(cpp, "requirement").rstrip("\n") + source[end:]
    return source


def edit_test(test: str, ref: Path) -> str:
    # 4.8: the update no longer recomputes; the TODO gives way; the new case.
    m = one(r"^  EXPECT_TRUE\(s\.fallback_used\);.*\n", test, "4.8 fallback_used")
    test = (
        test[: m.start()]
        + "  EXPECT_FALSE(s.fallback_used);  // incremental: no recomputation\n"
        + test[m.end() :]
    )
    m = one(r"^// TODO\(my_bfs\).*\n", test, "4.8 TODO")
    test = (
        test[: m.start()]
        + "// A deleted shortest-path edge cuts 2 and 3 off; "
        + "the inserted 3 -> 2 does not reach them.\n"
        + test[m.end() :]
    )
    m = one(r"^\}  // namespace\n", test, "4.8 the end of the test file")
    return (
        test[: m.start()] + snippet(ref / "my_bfs_test.cpp", "hand test") + "\n" + test[m.start() :]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    ref = root / REFERENCE
    files = {
        root / ALGORITHM / "problem.hpp": edit_problem,
        root / ALGORITHM / "my_bfs.cpp": edit_source,
        root / TEST: edit_test,
    }
    try:
        for path, edit in files.items():
            if not path.is_file():
                raise EditError(f"{path} is missing: scaffold my_bfs first (the tutorial, step 3)")
            path.write_text(edit(path.read_text(encoding="utf-8"), ref), encoding="utf-8")
    except EditError as e:
        print(f"tutorial_edits.py: {e}", file=sys.stderr)
        return 1
    formatter = shutil.which("clang-format")
    if formatter is not None:  # the reader's editor or pre-commit would format the files
        subprocess.run([formatter, "-i", *map(str, files)], check=True)
    print("tutorial_edits.py: applied the edits of section 4 to my_bfs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
