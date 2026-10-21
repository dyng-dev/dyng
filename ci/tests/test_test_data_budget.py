# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/test_data_budget.py (the committed test-data budget, ADR 0013, decision D5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_data_budget as tdb  # noqa: E402


def test_the_budget_is_one_and_a_quarter_megabytes() -> None:
    assert tdb.BUDGET == 1_250_000


def test_sizes_within_and_over_the_budget() -> None:
    ok, line = tdb.check({"a": 600_000, "b": 650_000})
    assert ok and "OK" in line
    ok, line = tdb.check({"a": 600_000, "b": 650_001})
    assert not ok and "OVER BUDGET" in line


def test_untracked_files_do_not_count(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    data = tmp_path / "cpp" / "tests" / "data"
    data.mkdir(parents=True)
    (data / "tracked.txt").write_bytes(b"x" * 10)
    (data / "untracked.bin").write_bytes(b"y" * 2_000_000)
    subprocess.run(["git", "-C", str(tmp_path), "add", "cpp/tests/data/tracked.txt"], check=True)
    sizes = tdb.tracked_sizes(tmp_path)
    assert sizes == {"cpp/tests/data/tracked.txt": 10}
    assert tdb.check(sizes)[0]


def test_the_repository_is_within_the_budget() -> None:
    ok, line = tdb.check(tdb.tracked_sizes())
    assert ok, line
