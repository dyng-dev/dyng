# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/tutorial_edits.py without a build (ci/scaffold_check.sh builds and tests the
result)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# What new_algorithm.py and tutorial_edits.py read and write.
PATHS = (
    ".clang-format",  # the scaffold and the edits are formatted when clang-format is installed
    "README.md",
    "CHANGELOG.md",
    "VERSION",
    ".github/CODEOWNERS",
    "docs",
    "scripts",
    "examples/tutorial_algorithms",
    "cpp/include/dyng",
    "cpp/src/algorithms",
    "cpp/src/core/registry_table.inc",
    "cpp/tests/algorithms",
    "cpp/tests/conformance",
    "python/dyng/_algorithms.py",
)


def _load(name: str, folder: str):
    spec = importlib.util.spec_from_file_location(name, REPO / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new_algorithm = _load("new_algorithm", "scripts")
tutorial_edits = _load("tutorial_edits", "ci")


def _scaffolded(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in PATHS:
        src, dst = REPO / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    command = ["my_bfs", "--family", "fixed_point", "--backends", "seq,omp", "--root", str(root)]
    assert new_algorithm.main([*command, "--title", "My first dynamic BFS", "--no-regen"]) == 0
    return root


def test_the_edits_apply_once_each(tmp_path: Path) -> None:
    root = _scaffolded(tmp_path)
    assert tutorial_edits.main(["--root", str(root)]) == 0
    problem = (root / "cpp/src/algorithms/my_bfs/problem.hpp").read_text()
    source = (root / "cpp/src/algorithms/my_bfs/my_bfs.cpp").read_text()
    test = (root / "cpp/tests/algorithms/my_bfs/my_bfs_test.cpp").read_text()
    assert problem.count("bool reads_prepared_graph() const noexcept") == 1
    assert "internal_frontier" not in problem
    assert problem.count("struct my_bfs_frontier") == 1
    assert "fallback_used = true" not in source
    assert source.count("::seed(framework::context& ctx") == 1
    assert source.count("::loop(framework::context&") == 1
    assert "EXPECT_FALSE(s.fallback_used)" in test
    assert "TODO(my_bfs)" not in test
    assert "AnEdgeInsertedThenDeletedOffersNoLevel" in test


def test_a_missing_anchor_is_reported(tmp_path: Path) -> None:
    root = _scaffolded(tmp_path)
    assert tutorial_edits.main(["--root", str(root)]) == 0
    # A second run finds the scaffold's anchors gone (the edits are not idempotent by design).
    assert tutorial_edits.main(["--root", str(root)]) == 1


def test_the_reference_markers_are_complete() -> None:
    reference = REPO / "examples/tutorial_algorithms/my_bfs"
    for path, marker in [
        ("problem.hpp", "workspace"),
        ("problem.hpp", "frontier"),
        ("problem.hpp", "hooks"),
        ("my_bfs.cpp", "reserve"),
        ("my_bfs.cpp", "identify_affected"),
        ("my_bfs.cpp", "offer"),
        ("my_bfs.cpp", "seed"),
        ("my_bfs.cpp", "loop"),
        ("my_bfs.cpp", "seed_static"),
        ("my_bfs.cpp", "requirement"),
        ("my_bfs_test.cpp", "hand test"),
    ]:
        assert tutorial_edits.snippet(reference / path, marker).strip()
    with pytest.raises(tutorial_edits.EditError):
        tutorial_edits.snippet(reference / "my_bfs.cpp", "no such marker")
