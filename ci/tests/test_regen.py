# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of scripts/regen.py and scripts/new_algorithm.py (without a build: the build of a
scaffolded algorithm is ci/scaffold_check.sh)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


regen = _load("regen")
new_algorithm = _load("new_algorithm")

# What regen.py and new_algorithm.py read and write.
PATHS = (
    "README.md",
    "CHANGELOG.md",
    ".github/CODEOWNERS",
    "docs",
    "scripts",
    "cpp/include/dyng",
    "cpp/src/algorithms",
    "cpp/src/core/registry_table.inc",
    "cpp/tests/algorithms",
    "cpp/tests/conformance",
)


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in PATHS:
        src = REPO / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def test_the_repository_is_up_to_date() -> None:
    assert regen.main(["--check"]) == 0


def test_check_reports_a_stale_table_and_regen_fixes_it(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("dynamic single-source", "static"))
    assert regen.main(["--check", "--root", str(root)]) == 1
    assert regen.main(["--root", str(root)]) == 0
    assert regen.main(["--check", "--root", str(root)]) == 0
    assert "dynamic single-source shortest paths" in readme.read_text()


def test_registration_rules_are_enforced(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    manifest = root / "cpp/src/algorithms/sssp/manifest.toml"
    text = manifest.read_text()
    manifest.write_text(
        text.replace('backends    = ["sequential", "openmp", "cuda"]', 'backends = ["openmp"]')
        .replace('"dynamosp2025", ', '"nosuchkey2099", ')
        .replace('oracle      = "compute"', 'oracle = "guess"')
    )
    try:
        regen.load(root)
    except regen.ManifestError as e:
        message = str(e)
    else:
        raise AssertionError("an invalid manifest was accepted")
    assert "must start with `sequential`" in message
    assert "nosuchkey2099" in message
    assert "`oracle` is `guess`" in message
    assert regen.main(["--check", "--root", str(root)]) == 2


def test_a_manifest_needs_its_add_subdirectory_line(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    cmake = root / "cpp/src/algorithms/CMakeLists.txt"
    cmake.write_text(cmake.read_text().replace("add_subdirectory(cycle_count)\n", ""))
    assert regen.main(["--check", "--root", str(root)]) == 2


def test_markers_select_the_family_and_the_backends() -> None:
    text = (
        "a\n//@@ fixed_point\nfp\n//@@ openmp\nfp-omp\n//@@ end\n//@@ end\n"
        "  #@@ aggregate_delta\nad\n  #@@ end\nz\n"
    )
    assert new_algorithm.select(text, {"fixed_point"}, "t") == "a\nfp\nz\n"
    assert new_algorithm.select(text, {"fixed_point", "openmp"}, "t") == "a\nfp\nfp-omp\nz\n"
    assert new_algorithm.select(text, {"aggregate_delta"}, "t") == "a\nad\nz\n"


def test_render_replaces_the_names_and_the_fields() -> None:
    text = "algorithm_template ALGORITHM_TEMPLATE AlgorithmTemplate {{title}}\n"
    out = new_algorithm.render(text, "dynamic_kcore", {"title": "K-cores"}, set(), "t")
    assert out == "dynamic_kcore DYNAMIC_KCORE DynamicKcore K-cores\n"


def test_scaffold_and_remove_round_trip(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    before = _snapshot(root)
    args = ["dynamic_kcore", "--family", "fixed_point", "--backends", "seq,omp"]
    args += ["--root", str(root)]
    assert new_algorithm.main(args) == 0
    assert regen.main(["--check", "--root", str(root)]) == 0
    header = (root / "cpp/include/dyng/dynamic_kcore.hpp").read_text()
    assert "namespace dyng::dynamic_kcore" in header and "algorithm_template" not in header
    assert "@@" not in (root / "cpp/src/algorithms/dynamic_kcore/problem.hpp").read_text()
    assert (root / "cpp/src/algorithms/dynamic_kcore/openmp.cpp").is_file()
    assert "`dynamic_kcore`" in (root / "README.md").read_text()
    assert "struct dynamic_kcore;" in (root / "cpp/tests/conformance/registry.hpp").read_text()
    assert "/cpp/src/algorithms/dynamic_kcore/" in (root / ".github/CODEOWNERS").read_text()
    # A second scaffold of the same name is refused.
    assert new_algorithm.main(args) == 2
    assert new_algorithm.main(["dynamic_kcore", "--remove", "--root", str(root)]) == 0
    after = _snapshot(root)
    assert after == before


def test_a_planned_algorithm_leaves_the_planned_list(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    args = ["mosp", "--family", "fixed_point", "--no-changelog", "--root", str(root)]
    assert new_algorithm.main(args) == 0
    planned = (root / "cpp/src/algorithms/planned.toml").read_text()
    assert 'name      = "mosp"' not in planned
    assert 'name      = "triad_count"' in planned
    assert regen.main(["--check", "--root", str(root)]) == 0


def test_cuda_and_hypergraph_scaffolds_are_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    base = ["x_algo", "--family", "aggregate_delta", "--root", str(root)]
    assert new_algorithm.main([*base, "--backends", "seq,cuda"]) == 2
    assert new_algorithm.main([*base, "--container", "hypergraph"]) == 2
