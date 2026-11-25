# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/cibw_test.py: cibuildwheel's test step from the unpacked sdist (R040's review).

A test source that the sdist leaves out fails before any test runs, as in cibuildwheel; with
every source present the test command runs from a directory holding only the test sources, with
the test environment set and the given interpreter first on PATH.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
import cibw_test  # noqa: E402

PYPROJECT = """\
[tool.cibuildwheel]
test-sources = ["pyproject.toml", "tests", "data/digest.sha256"]
test-environment = {{ DYNG_CIBW_PROBE = "set" }}
test-command = "{command}"
"""
# Runs in the copied directory: every test source is there, nothing else of the sdist is, the
# environment is set, and `python` is the interpreter given (here: this one).
PROBE = (
    'python -c \\"import os, pathlib, sys; '
    "assert sorted(p.name for p in pathlib.Path().iterdir()) == ['data', 'pyproject.toml', "
    "'tests'], list(pathlib.Path().iterdir()); "
    "assert pathlib.Path('data/digest.sha256').read_text() == 'abc', 'digest'; "
    "assert os.environ['DYNG_CIBW_PROBE'] == 'set'; "
    'sys.exit(7)\\"'
)


def _sdist(tmp: Path, files: dict[str, str]) -> Path:
    path = tmp / "pkg-1.0.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"pkg-1.0/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _files(**drop: bool) -> dict[str, str]:
    files = {
        "pyproject.toml": PYPROJECT.format(command=PROBE),
        "tests/test_x.py": "",
        "data/digest.sha256": "abc",
        "src/other.txt": "not a test source",
    }
    return {k: v for k, v in files.items() if not drop.get(k.replace("/", "_").replace(".", "_"))}


def test_a_test_source_missing_from_the_sdist_fails(tmp_path: Path, capsys) -> None:
    sdist = _sdist(tmp_path, _files(data_digest_sha256=True))
    assert cibw_test.main(["--sdist", str(sdist), "--check"]) == 1
    err = capsys.readouterr().err
    assert "data/digest.sha256" in err and "does not exist" in err


def test_the_check_passes_with_every_source(tmp_path: Path, capsys) -> None:
    sdist = _sdist(tmp_path, _files())
    assert cibw_test.main(["--sdist", str(sdist), "--check"]) == 0
    assert "3 test sources" in capsys.readouterr().out


def test_the_command_runs_from_the_copied_sources(tmp_path: Path) -> None:
    sdist = _sdist(tmp_path, _files())
    # The probe exits 7 after its assertions pass (an assertion failure would exit 1).
    assert cibw_test.main(["--sdist", str(sdist), "--python", sys.executable]) == 7


def test_an_sdist_without_a_test_command_is_refused(tmp_path: Path) -> None:
    sdist = _sdist(tmp_path, {"pyproject.toml": "[project]\nname = 'pkg'\n"})
    with pytest.raises(cibw_test.CibwTestError):
        cibw_test.run(sdist, None)


def test_the_repository_test_sources_are_sdist_members() -> None:
    # Every [tool.cibuildwheel] test-sources entry of the repository is a member of the sdist (a
    # file) or a directory holding members (parity/certify.py's sdist_members(): the tracked
    # files that pyproject.toml's sdist.exclude keeps in, plus sdist.include).
    sys.path.insert(0, str(ROOT / "parity"))
    import certify

    sources = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["cibuildwheel"][
        "test-sources"
    ]
    members = certify.sdist_members()
    absent = [
        s for s in sources if s not in members and not any(m.startswith(s + "/") for m in members)
    ]
    assert absent == [], f"test-sources entries the sdist leaves out: {absent}"
