# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python half of the API check (ci/api_check.sh: griffe) covers dyng.mosp, frozen in 0.2
(ADR 0035): a deliberate change of a mosp signature, default or public name is a breaking change.

griffe compares two static copies of the typed layer python/dyng here (ci/api_check.sh compares
the working tree with a base revision of the history); nothing is built or imported.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# CI (lint.yml's harness job, api-check.yml) sets DYNG_REQUIRE_GRIFFE=1: there a missing griffe
# fails these tests instead of skipping them.
if os.environ.get("DYNG_REQUIRE_GRIFFE") == "1":
    import griffe
else:
    griffe = pytest.importorskip("griffe")

PACKAGE = Path(__file__).resolve().parent.parent.parent / "python" / "dyng"


def _load(root: Path):
    return griffe.load(
        "dyng",
        search_paths=[str(root)],
        allow_inspection=False,
        modules_collection=griffe.ModulesCollection(),
        lines_collection=griffe.LinesCollection(),
    )


def _breakages(tmp_path: Path, edit) -> list:
    old = tmp_path / "old"
    new = tmp_path / "new"
    for root in (old, new):
        shutil.copytree(
            PACKAGE, root / "dyng", ignore=shutil.ignore_patterns("*.so", "__pycache__")
        )
    mosp = new / "dyng" / "mosp.py"
    text = mosp.read_text(encoding="utf-8")
    changed = edit(text)
    assert changed != text, "the edit did not apply"
    mosp.write_text(changed, encoding="utf-8")
    return list(griffe.find_breaking_changes(_load(old), _load(new)))


def _in_mosp(breakages: list) -> list[str]:
    return [b.explain() for b in breakages if b.obj.path.startswith("dyng.mosp")]


def test_an_unchanged_package_has_no_breaking_change(tmp_path: Path) -> None:
    assert _breakages(tmp_path, lambda t: t + "\n") == []


def test_a_renamed_parameter_of_mosp_compute_is_breaking(tmp_path: Path) -> None:
    def edit(text: str) -> str:
        return text.replace(
            "def compute(\n    graph: Graph,\n    source: int,",
            "def compute(\n    graph: Graph,\n    root: int,",
        )

    assert _in_mosp(_breakages(tmp_path, edit))


def test_a_changed_default_of_mosp_options_is_breaking(tmp_path: Path) -> None:
    def edit(text: str) -> str:
        return text.replace(
            "    compute_path_costs: bool = True\n", "    compute_path_costs: bool = False\n"
        )

    assert _in_mosp(_breakages(tmp_path, edit))


def test_a_removed_mosp_result_accessor_is_breaking(tmp_path: Path) -> None:
    def edit(text: str) -> str:
        return text.replace(
            "    def preference_scale(self) -> int:", "    def _preference_scale(self) -> int:"
        )

    assert _in_mosp(_breakages(tmp_path, edit))
