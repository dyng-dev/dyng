#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Run cibuildwheel's test step the way wheels.yml runs it: from the unpacked sdist.

`.github/workflows/wheels.yml` builds the CPU wheel with cibuildwheel from the sdist
(`package-dir` is the sdist). cibuildwheel unpacks it, copies every `[tool.cibuildwheel]
test-sources` entry of the sdist's pyproject.toml into an empty directory (a missing entry is a
fatal error before any test runs: "Test source ... does not exist") and runs `test-command` there
with `test-environment` set. ci/wheels.sh runs the full pytest suite from the repository checkout,
which has every file; this script repeats cibuildwheel's step from the sdist itself, so that a
test source or a test input the sdist leaves out fails locally too (R040's review: the digest of
examples/data named in test-sources was not in the sdist).

    ci/cibw_test.py --sdist dist/dyng-X.tar.gz --check        # the test sources only
    ci/cibw_test.py --sdist dist/dyng-X.tar.gz --python venv/bin/python   # and the test command

`--python` is the interpreter of a venv with the wheel and the test requirements installed; its
directory goes first on PATH, so the command's `python` is that interpreter (as in cibuildwheel's
test venv).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path


class CibwTestError(Exception):
    """A test source is missing from the sdist, or the sdist has no cibuildwheel test step."""


def unpack(sdist: Path, into: Path) -> Path:
    """Unpack `sdist` into `into` and return its top directory (the project, as cibuildwheel's
    package dir)."""
    with tarfile.open(sdist) as tar:
        tar.extractall(into, filter="data")
    tops = [p for p in into.iterdir() if p.is_dir()]
    if len(tops) != 1 or not (tops[0] / "pyproject.toml").is_file():
        raise CibwTestError(f"{sdist}: not an sdist with one top directory holding pyproject.toml")
    return tops[0]


def test_settings(project: Path) -> dict:
    """The [tool.cibuildwheel] table of the project's pyproject.toml."""
    table = tomllib.loads((project / "pyproject.toml").read_text())
    cibw = table.get("tool", {}).get("cibuildwheel", {})
    if not cibw.get("test-command"):
        raise CibwTestError(f"{project}/pyproject.toml has no [tool.cibuildwheel] test-command")
    return cibw


def copy_test_sources(project: Path, sources: list[str], dest: Path) -> None:
    """Copy each test source of `project` to the same relative path under `dest`; a missing one
    fails as in cibuildwheel (every missing source is named)."""
    missing = [s for s in sources if not (project / s).exists()]
    if missing:
        raise CibwTestError(
            "test sources not in the sdist (cibuildwheel stops with 'Test source ... does not "
            "exist'): " + ", ".join(missing)
        )
    for source in sources:
        src, dst = project / source, dest / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def run(sdist: Path, python: Path | None) -> int:
    with tempfile.TemporaryDirectory(prefix="dyng-cibw-test-") as tmp:
        project = unpack(sdist, Path(tmp) / "sdist")
        cibw = test_settings(project)
        sources = list(cibw.get("test-sources", []))
        dest = Path(tmp) / "test"
        dest.mkdir()
        copy_test_sources(project, sources, dest)
        print(
            f"cibw_test: {len(sources)} test sources of the sdist copied: {', '.join(sources)}",
            flush=True,
        )
        if python is None:
            return 0
        env = dict(os.environ)
        env.update({k: str(v) for k, v in cibw.get("test-environment", {}).items()})
        env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"
        env.pop("PYTHONPATH", None)
        command = cibw["test-command"]
        print(f"cibw_test: running the test command in {dest}: {command}", flush=True)
        return subprocess.run(command, shell=True, cwd=dest, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sdist", type=Path, required=True, help="the sdist (.tar.gz)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="check the test sources only")
    group.add_argument("--python", type=Path, help="the test venv's interpreter")
    args = parser.parse_args(argv)
    try:
        return run(args.sdist, None if args.check else args.python.absolute())
    except CibwTestError as e:
        print(f"ci/cibw_test.py: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
