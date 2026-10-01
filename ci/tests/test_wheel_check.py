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


def test_the_repository_release_metadata_agrees() -> None:
    # VERSION, CHANGELOG.md and CITATION.cff (docs/developer/release.md steps 5, 6 and 9).
    assert wheel_check.check_release_metadata() == []


def _release_tree(tmp_path: Path, version: str = "0.2.0rc1", date: str = "2027-01-02") -> Path:
    (tmp_path / "VERSION").write_text(version + "\n")
    (tmp_path / "CITATION.cff").write_text(
        f'cff-version: 1.2.0\nversion: {version}\ndate-released: "{date}"\n'
        "preferred-citation:\n  version: 9.9.9\n"
    )
    url = wheel_check.REPOSITORY_URL
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{version}] - {date}\n\n- x\n\n"
        "## [0.1.0] - 2026-10-10\n\n- y\n\n"
        f"[Unreleased]: {url}/compare/v{version}...main\n"
        f"[{version}]: {url}/compare/v0.1.0...v{version}\n"
    )
    return tmp_path


def test_release_metadata_finds_what_disagrees(tmp_path: Path) -> None:
    root = _release_tree(tmp_path)
    assert wheel_check.check_release_metadata(root) == []
    assert wheel_check.check_release_metadata(root, "2027-01-02") == []
    assert "day of the tag" in " ".join(wheel_check.check_release_metadata(root, "2027-01-03"))
    cff = (root / "CITATION.cff").read_text()
    (root / "CITATION.cff").write_text(cff.replace('"2027-01-02"', '"2027-01-03"'))
    assert "date-released 2027-01-03" in " ".join(wheel_check.check_release_metadata(root))
    (root / "CITATION.cff").write_text(cff.replace("version: 0.2.0rc1", "version: 0.2.0"))
    assert "CITATION.cff: version 0.2.0" in " ".join(wheel_check.check_release_metadata(root))
    (root / "CITATION.cff").write_text(cff)
    text = (root / "CHANGELOG.md").read_text()
    (root / "CHANGELOG.md").write_text(text.replace("## [Unreleased]\n\n", ""))
    assert "Unreleased" in " ".join(wheel_check.check_release_metadata(root))
    (root / "CHANGELOG.md").write_text(text.replace("compare/v0.2.0rc1...main", "compare/x"))
    assert "link reference" in " ".join(wheel_check.check_release_metadata(root))
    (root / "CHANGELOG.md").write_text(text)
    (root / "VERSION").write_text("0.2.0\n")  # the final release, CHANGELOG not renamed
    errors = " ".join(wheel_check.check_release_metadata(root))
    assert "0 sections '## [0.2.0]" in errors and "CITATION.cff: version" in errors
    # Between releases: the last release must still be consistent.
    (root / "VERSION").write_text("0.2.1.dev0\n")
    assert wheel_check.check_release_metadata(root) == []
    assert wheel_check.main(["--release-metadata"]) == 0
