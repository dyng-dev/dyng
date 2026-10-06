# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/wheel_check.py rejects broken distributions and accepts a good one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_plugin_constants() -> None:
    # ADR 0030: the plugins, their extra licence files, and their licence expression, which stays
    # the CPU wheel's until the author decides (GOVERNANCE.md, open decisions).
    assert wheel_check.PLUGINS == ("cu12", "cu13")
    assert wheel_check.PLUGIN_EXTRA_LICENSE_FILES == (
        "THIRD_PARTY_LICENSES_CUDA.txt",
        "NVIDIA_CUDA_EULA.txt",
    )
    assert wheel_check.PLUGIN_LICENSE_EXPRESSION == wheel_check.LICENSE_EXPRESSION
    assert (wheel_check.ROOT / "THIRD_PARTY_LICENSES_CUDA.txt").is_file()


def test_forbidden_libraries() -> None:
    match = wheel_check.PLUGIN_FORBIDDEN_LIBRARY.match
    for bad in ("libcuda.so.1", "libcuda.so", "libcudart.so.13", "libcudart-1a2b3c4d.so.13.1"):
        assert match(bad), bad
    assert match("libnvidia-ml.so.1")
    for good in ("libgomp-a25fd822.so.1.0.0", "libc.so.6", "libm.so.6", "libpthread.so.0"):
        assert not match(good), good


@pytest.mark.parametrize(
    ("v", "names"),
    [
        ("0.1.0", ["dyng"]),
        ("0.1.2", ["dyng"]),
        ("0.1.1rc1", ["dyng"]),
        ("0.2.0rc1", ["dyng", "dyng-cu12", "dyng-cu13"]),
        ("0.2.0", ["dyng", "dyng-cu12", "dyng-cu13"]),
        ("0.2.0.dev0", ["dyng", "dyng-cu12", "dyng-cu13"]),
        ("1.0.0", ["dyng", "dyng-cu12", "dyng-cu13"]),
    ],
)
def test_release_distributions(v: str, names: list[str]) -> None:
    # PLAN Appendix F: the plugins are released from 0.2.0 (its candidates included); each
    # distribution publishes through its own environments (testpypi<suffix>, pypi<suffix>).
    dists = wheel_check.release_distributions(v)
    assert [d["name"] for d in dists] == names
    env = {d["name"]: d["suffix"] for d in dists}
    assert env["dyng"] == ""  # the trusted publishers of dyng: testpypi / pypi (never renamed)
    for p in names[1:]:
        assert env[p] == "-" + p.removeprefix("dyng-")
    assert all(d["prefix"].endswith(f"-{v}") for d in dists)


def test_release_set_and_split(tmp_path: Path) -> None:
    v = "0.2.0rc1"
    tag = "cp312-abi3-manylinux_2_28_x86_64"
    files = [
        tmp_path / f"dyng-{v}.tar.gz",
        tmp_path / f"dyng-{v}-{tag}.whl",
        tmp_path / f"dyng_cu12-{v}-{tag}.whl",
        tmp_path / f"dyng_cu13-{v}-{tag}.whl",
    ]
    for f in files:
        f.write_bytes(b"x")
    assert wheel_check.check_release_set(files, v) == []
    assert any("dyng-cu13: no wheel" in e for e in wheel_check.check_release_set(files[:3], v))
    assert any("no sdist" in e for e in wheel_check.check_release_set(files[1:], v))
    stray = tmp_path / "dyng_cu11-0.2.0rc1-x.whl"
    stray.write_bytes(b"x")
    assert wheel_check.check_release_set([*files, stray], v)
    plugin_sdist = tmp_path / f"dyng_cu12-{v}.tar.gz"
    plugin_sdist.write_bytes(b"x")
    assert any("an sdist" in e for e in wheel_check.check_release_set([*files, plugin_sdist], v))
    # a 0.1.x release has no plugins
    old = [tmp_path / "dyng-0.1.1.tar.gz", tmp_path / f"dyng-0.1.1-{tag}.whl"]
    for f in old:
        f.write_bytes(b"x")
    assert wheel_check.check_release_set(old, "0.1.1") == []
    assert wheel_check.check_release_set([*old, files[2]], "0.1.1")
    split = wheel_check.split_release(files, tmp_path / "out", v)
    assert {k: sorted(p.name for p in ps) for k, ps in split.items()} == {
        "dyng": sorted([files[0].name, files[1].name]),
        "dyng-cu12": [files[2].name],
        "dyng-cu13": [files[3].name],
    }
    assert (tmp_path / "out" / "dyng-cu13" / files[3].name).is_file()
    with pytest.raises(ValueError):
        wheel_check.split_release(files[:2], tmp_path / "out2", v)
