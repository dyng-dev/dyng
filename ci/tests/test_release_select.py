# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The `select` job of .github/workflows/release.yml, run for the kinds of tags it decides.

The step's script is read from the workflow (so the test follows every edit of it) and run with
bash in a temporary tree that holds what it reads: ``ci/wheel_check.py`` (copied),
``tools/name_reservation/pyproject.toml`` (copied), and ``VERSION``, ``CHANGELOG.md`` and
``CITATION.cff`` written for the tag. The outputs (``GITHUB_OUTPUT``) are checked: the kind, the
version, whether it is a pre-release, whether the CUDA plugins are built, every distribution
with its environment suffix, and the plugins' matrix (the core ``dyng`` is published by its own
jobs, after the plugins).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
RESERVATION = ROOT / "tools" / "name_reservation" / "pyproject.toml"
URL = "https://github.com/dyng-dev/dyng"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _select_script() -> str:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["select"]["steps"]
    (step,) = [s for s in steps if s.get("id") == "select"]
    assert step.get("shell") == "bash"
    script: str = step["run"]
    return script


def _tree(tmp: Path, version: str) -> Path:
    (tmp / "ci").mkdir(parents=True)
    shutil.copy(ROOT / "ci" / "wheel_check.py", tmp / "ci" / "wheel_check.py")
    (tmp / "tools" / "name_reservation").mkdir(parents=True)
    shutil.copy(RESERVATION, tmp / "tools" / "name_reservation" / "pyproject.toml")
    (tmp / "VERSION").write_text(version + "\n")
    (tmp / "CITATION.cff").write_text(
        f'cff-version: 1.2.0\ntitle: dynG\nversion: "{version}"\ndate-released: "2026-10-06"\n'
    )
    (tmp / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n"
        f"## [{version}] - 2026-10-06\n\n- A release.\n\n"
        f"[Unreleased]: {URL}/compare/v{version}...main\n"
        f"[{version}]: {URL}/releases/tag/v{version}\n"
    )
    return tmp


def _run(tmp: Path, tag: str, version: str) -> tuple[int, dict[str, str], str]:
    tree = _tree(tmp, version)
    output = tmp / "github_output"
    output.write_text("")
    script = tmp / "select.sh"
    script.write_text(_select_script())
    env = {**os.environ, "GITHUB_REF_NAME": tag, "GITHUB_OUTPUT": str(output)}
    out = subprocess.run(
        ["bash", str(script)], cwd=tree, env=env, capture_output=True, text=True, check=False
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    return out.returncode, values, out.stdout + out.stderr


def _suffixes(text: str) -> dict[str, str]:
    return {d["name"]: d["suffix"] for d in json.loads(text)}


PLUGINS = {"dyng-cu12": "-cu12", "dyng-cu13": "-cu13"}


@pytest.mark.parametrize(
    ("tag", "version", "kind", "prerelease", "distributions"),
    [
        ("v0.1.1", "0.1.1", "package", "false", {"dyng": ""}),
        ("v0.1.2rc1", "0.1.2rc1", "package", "true", {"dyng": ""}),
        ("v0.2.0rc1", "0.2.0rc1", "package", "true", {"dyng": "", **PLUGINS}),
        ("v0.2.0", "0.2.0", "package", "false", {"dyng": "", **PLUGINS}),
        ("v1.0.0", "1.0.0", "package", "false", {"dyng": "", **PLUGINS}),
    ],
)
def test_package_tags(
    tmp_path: Path,
    tag: str,
    version: str,
    kind: str,
    prerelease: str,
    distributions: dict[str, str],
) -> None:
    rc, out, log = _run(tmp_path, tag, version)
    assert rc == 0, log
    assert (out["kind"], out["version"], out["prerelease"]) == (kind, version, prerelease)
    assert _suffixes(out["distributions"]) == distributions
    plugins = {k: v for k, v in distributions.items() if k != "dyng"}
    assert out["plugins"] == ("true" if plugins else "false")
    assert _suffixes(out["plugin_distributions"]) == plugins
    for d in json.loads(out["distributions"]):  # the file prefixes collect sorts by
        package = d["name"].replace("-", "_") if d["name"] != "dyng" else "dyng"
        assert d["prefix"] == f"{package}-{version}"


def test_the_reservation_tag(tmp_path: Path) -> None:
    import tomllib

    reserved = tomllib.loads(RESERVATION.read_text())["project"]["version"]
    rc, out, log = _run(tmp_path, f"v{reserved}", "0.2.0.dev0")
    assert rc == 0, log
    assert (out["kind"], out["version"], out["prerelease"]) == ("reservation", reserved, "false")
    assert json.loads(out["distributions"]) == [
        {"name": "dyng", "prefix": f"dyng-{reserved}", "suffix": ""}
    ]
    assert out["plugins"] == "false" and out["plugin_distributions"] == "[]"


@pytest.mark.parametrize(
    ("tag", "version", "message"),
    [
        ("v0.0.2", "0.0.2", "0.0.x tags are the name reservation"),
        ("v0.2.0", "0.2.0rc1", "does not match VERSION"),
        ("v0.2.0rc1", "0.2.0", "does not match VERSION"),
        ("v0.2.0-rc.1", "0.2.0-rc.1", "not a canonical PEP 440 version"),
    ],
)
def test_refused_tags(tmp_path: Path, tag: str, version: str, message: str) -> None:
    rc, out, log = _run(tmp_path, tag, version)
    assert rc != 0 and message in log, log
    assert "distributions" not in out


def test_metadata_that_disagrees_is_refused(tmp_path: Path) -> None:
    tree = _tree(tmp_path / "t", "0.2.0")
    (tree / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "0.1.0"\ndate-released: "2026-10-06"\n'
    )
    output = tmp_path / "out"
    output.write_text("")
    script = tmp_path / "select.sh"
    script.write_text(_select_script())
    env = {**os.environ, "GITHUB_REF_NAME": "v0.2.0", "GITHUB_OUTPUT": str(output)}
    out = subprocess.run(
        ["bash", str(script)], cwd=tree, env=env, capture_output=True, text=True, check=False
    )
    assert out.returncode != 0 and "disagree" in out.stdout + out.stderr
