# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/plugin_pyproject.py renders the CUDA plugins' pyproject.toml from the root one (ADR 0030)."""

from __future__ import annotations

import json
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plugin_pyproject as pp  # noqa: E402
import wheel_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VERSION = (ROOT / "VERSION").read_text().strip()
EULA_TEXT = "End User License Agreement\n--------------------------\nNVIDIA CUDA Toolkit ...\n"


def rendered(plugin: str) -> dict:
    return tomllib.loads(pp.render((ROOT / "pyproject.toml").read_text(), plugin, VERSION))


@pytest.mark.parametrize("plugin", wheel_check.PLUGINS)
def test_the_plugin_metadata(plugin: str) -> None:
    d = rendered(plugin)
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    p = d["project"]
    major = plugin[2:]
    assert p["name"] == f"dyng-{plugin}"
    assert p["dynamic"] == ["version"]
    assert p["dependencies"] == [f"dyng=={VERSION}"]
    assert p["entry-points"] == {"dyng.backends": {plugin: f"dyng_{plugin}"}}
    assert "optional-dependencies" not in p and "scripts" not in p
    assert p["license"] == wheel_check.PLUGIN_LICENSE_EXPRESSION
    assert p["license-files"] == [*root["project"]["license-files"], *pp.PLUGIN_EXTRA_LICENSE_FILES]
    assert f"Environment :: GPU :: NVIDIA CUDA :: {major}" in p["classifiers"]
    assert not any(c.startswith("Typing ::") for c in p["classifiers"])
    assert p["readme"] == "python/plugin/README.md"
    for same in ("requires-python", "authors", "urls"):
        assert p[same] == root["project"][same]
    assert d["build-system"] == root["build-system"]
    assert d["tool"]["dynamic-metadata"] == [
        m for m in root["tool"]["dynamic-metadata"] if m["field"] == "version"
    ]


@pytest.mark.parametrize("plugin", wheel_check.PLUGINS)
def test_the_plugin_build(plugin: str) -> None:
    skb = rendered(plugin)["tool"]["scikit-build"]
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["scikit-build"]
    assert skb["wheel"]["packages"] == [f"python/plugin/dyng_{plugin}"]
    assert skb["wheel"]["py-api"] == root["wheel"]["py-api"] == "cp312"
    assert skb["build"]["targets"] == ["_core"]
    assert skb["install"]["components"] == ["python"]
    assert skb["minimum-version"] == root["minimum-version"]
    assert skb["cmake"]["version"] == root["cmake"]["version"]
    assert "sdist" not in skb
    defines = skb["cmake"]["define"]
    assert defines == {
        **root["cmake"]["define"],
        "DYNG_ENABLE_CUDA": "ON",
        "DYNG_ENABLE_OPENMP": "ON",
        "DYNG_PYTHON_PLUGIN": plugin,
        "CMAKE_CUDA_RUNTIME_LIBRARY": "Static",
        "DYNG_CUDA_ARCHITECTURES": "release",
    }


def test_unknown_plugins_and_other_projects_are_refused() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    with pytest.raises(ValueError, match="unknown plugin"):
        pp.render(text, "cu11", VERSION)
    with pytest.raises(ValueError, match="not the one of dyng"):
        pp.render(text.replace('name = "dyng"', 'name = "other"', 1), "cu13", VERSION)


def test_the_core_extras_pin_the_plugins() -> None:
    # The root pyproject.toml fills the extras cu12 / cu13 from the version (template provider).
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "optional-dependencies" in root["project"]["dynamic"]
    (extras,) = [m for m in root["tool"]["dynamic-metadata"] if m["field"] != "version"]
    assert extras["field"] == "optional-dependencies"
    assert extras["provider"] == "scikit_build_core.metadata.template"
    for plugin in wheel_check.PLUGINS:
        assert extras["result"][plugin] == [f"dyng-{plugin}=={{project[version]}}"]


def _tree(tmp_path: Path) -> Path:
    tree = tmp_path / "src"
    (tree / "python" / "plugin").mkdir(parents=True)
    shutil.copy(ROOT / "pyproject.toml", tree / "pyproject.toml")
    shutil.copy(ROOT / "VERSION", tree / "VERSION")
    shutil.copy(ROOT / pp.PLUGIN_README, tree / pp.PLUGIN_README)
    shutil.copytree(ROOT / pp.PLUGIN_PACKAGE_SOURCE, tree / pp.PLUGIN_PACKAGE_SOURCE)
    return tree


def test_prepare_turns_a_tree_into_the_plugins(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    eula = tmp_path / "EULA.txt"
    eula.write_text(EULA_TEXT)
    pp.prepare(tree, "cu13", eula)
    assert tomllib.loads((tree / "pyproject.toml").read_text())["project"]["name"] == "dyng-cu13"
    assert (tree / pp.EULA_NAME).read_text() == EULA_TEXT
    package = tree / "python" / "plugin" / "dyng_cu13" / "__init__.py"
    assert package.read_text() == (ROOT / pp.PLUGIN_PACKAGE_SOURCE / "__init__.py").read_text()
    # main() with --project-dir and --cuda-eula does the same
    tree2 = _tree(tmp_path / "2")
    args = ["--plugin", "cu12", "--project-dir", str(tree2), "--cuda-eula", str(eula)]
    assert pp.main(args) == 0
    assert (tree2 / "python" / "plugin" / "dyng_cu12" / "__init__.py").is_file()


def test_prepare_wants_the_eula(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    not_eula = tmp_path / "README"
    not_eula.write_text("hello")
    with pytest.raises(ValueError, match="EULA"):
        pp.prepare(tree, "cu13", not_eula)
    assert tomllib.loads((tree / "pyproject.toml").read_text())["project"]["name"] == "dyng"


def test_find_cuda_eula(tmp_path: Path) -> None:
    nvidia = tmp_path / "cuda-13.1"
    nvidia.mkdir()
    (nvidia / "EULA.txt").write_text(EULA_TEXT)
    assert pp.find_cuda_eula(nvidia) == nvidia / "EULA.txt"
    # a conda environment of conda-forge's packages: the cudart package's licence file
    conda = tmp_path / "env"
    (conda / "conda-meta").mkdir(parents=True)
    pkg = tmp_path / "pkgs" / "cuda-cudart-static-12.9.79-0"
    (pkg / "info" / "licenses").mkdir(parents=True)
    (pkg / "info" / "licenses" / "LICENSE").write_text(EULA_TEXT)
    (conda / "conda-meta" / "cuda-cudart-static-12.9.79-0.json").write_text(
        json.dumps({"extracted_package_dir": str(pkg)})
    )
    assert pp.find_cuda_eula(conda) == pkg / "info" / "licenses" / "LICENSE"
    (conda / "LICENSE").write_text("MIT License ...")  # not the EULA: skipped
    assert pp.find_cuda_eula(conda) == pkg / "info" / "licenses" / "LICENSE"
    with pytest.raises(FileNotFoundError, match="--cuda-eula"):
        pp.find_cuda_eula(tmp_path / "nothing")


def test_print(capsys: pytest.CaptureFixture[str]) -> None:
    assert pp.main(["--plugin", "cu13", "--print"]) == 0
    assert tomllib.loads(capsys.readouterr().out)["project"]["name"] == "dyng-cu13"
