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


def test_licence_expression_is_the_authors_decision() -> None:
    # GOVERNANCE.md, approvals log, 2026-09-30; pyproject.toml must agree.
    assert wheel_check.LICENSE_EXPRESSION == (
        "Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1"
    )
    assert wheel_check.check_pyproject() == []


def test_metadata_licence_fields() -> None:
    files = "".join(f"License-File: {f}\n" for f in wheel_check.LICENSE_FILES)
    good = f"Metadata-Version: 2.4\nLicense-Expression: {wheel_check.LICENSE_EXPRESSION}\n{files}"
    assert wheel_check.check_metadata(good, "METADATA") == []
    assert wheel_check.check_metadata(good.replace(" AND BSD-3-Clause", ""), "METADATA")
    assert wheel_check.check_metadata(good.replace("License-File: NOTICE\n", ""), "METADATA")
    assert wheel_check.check_metadata(good + "License: Apache-2.0\n", "METADATA")
