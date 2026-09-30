# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/wheel_check.py rejects broken distributions and accepts a good one."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wheel_check  # noqa: E402


def test_self_test() -> None:
    assert wheel_check.main(["--self-test"]) == 0


def test_no_files_is_an_error() -> None:
    assert wheel_check.main([]) == 1


def test_budget_is_90_mb() -> None:
    assert wheel_check.MAX_WHEEL_BYTES == 90_000_000
