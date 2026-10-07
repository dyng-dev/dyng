#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Turn a source tree of `dyng` into the source tree of a CUDA plugin (PLAN 5.4, 7.7; ADR 0030).

The CUDA plugins ``dyng-cu12`` and ``dyng-cu13`` are built from the same sources as the CPU
package: this script replaces the root ``pyproject.toml`` of a source tree (an unpacked sdist of
``dyng``, or a disposable checkout) with the plugin's, rendered from the root one, so the
metadata the two share (licence, authors, URLs, build requirements, scikit-build-core settings)
has one source. What changes for the plugin ``cu<N>``:

- ``name = "dyng-cu<N>"``, its own description, readme (``python/plugin/README.md``), keywords
  and classifiers; ``dependencies = ["dyng==<VERSION>"]`` (the plugin is useless without the
  ``dyng`` of exactly its version, which the typed layer comes from); no extras and no console
  script; the entry point ``[project.entry-points."dyng.backends"] cu<N> = "dyng_cu<N>"``;
- ``wheel.packages = ["python/plugin/dyng_cu<N>"]``, a copy of ``python/plugin/dyng_plugin`` made
  in the tree: the wheel holds only the plugin's import package (never ``dyng/``) and the
  extension module ``dyng_cu<N>/_core.abi3.so``;
- the CMake defines ``DYNG_ENABLE_CUDA=ON``, ``DYNG_PYTHON_PLUGIN=cu<N>``,
  ``CMAKE_CUDA_RUNTIME_LIBRARY=Static`` and ``DYNG_CUDA_ARCHITECTURES=release`` (the release list
  of ``cmake/cuda_architectures.cmake`` for the toolkit in use: SASS per architecture, PTX for
  the highest), ``DYNG_ENABLE_OPENMP=ON``;
- the licence files add ``THIRD_PARTY_LICENSES_CUDA.txt`` and ``NVIDIA_CUDA_EULA.txt``, which
  this script copies into the tree from the CUDA toolkit the wheel is built with (``--cuda-eula``,
  or found under ``--cuda-root`` / ``$CUDA_HOME`` / the directory of ``$CUDACXX``); the licence
  expression is ``ci/wheel_check.py``'s ``PLUGIN_LICENSE_EXPRESSION``.

Usage::

    python3 ci/plugin_pyproject.py --plugin cu13 --project-dir <unpacked dyng sdist> \\
        [--cuda-root /usr/local/cuda-13.1 | --cuda-eula <path>]
    python3 ci/plugin_pyproject.py --plugin cu13 --print      # the rendered file, to stdout

Then build the wheel in that directory with the CUDA toolkit of the plugin's major
(``CUDACXX=<toolkit>/bin/nvcc pip wheel <dir>``; ``ci/plugin_wheels.sh`` does all of it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wheel_check import (  # noqa: E402  (the constants of the checks)
    PLUGIN_EXTRA_LICENSE_FILES,
    PLUGIN_LICENSE_EXPRESSION,
    PLUGINS,
)

#: Where the plugin's import package and its PyPI description live in the source tree.
PLUGIN_PACKAGE_DIR = "python/plugin"
PLUGIN_PACKAGE_SOURCE = f"{PLUGIN_PACKAGE_DIR}/dyng_plugin"
PLUGIN_README = "python/plugin/README.md"
#: The file name of the CUDA Toolkit's licence in the plugin's source tree and wheel.
EULA_NAME = "NVIDIA_CUDA_EULA.txt"


def _toml(value: Any) -> str:
    """A TOML value (strings, booleans, integers, lists and inline tables of those)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # TOML basic strings share JSON's escapes
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_toml(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _key(k: str) -> str:
    """A key: bare, or dotted (``cmake.version``: nested tables), quoted otherwise."""
    return k if re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", k) else json.dumps(k)


def _table(name: str, items: dict[str, Any]) -> list[str]:
    return [f"[{name}]", *(f"{_key(k)} = {_toml(v)}" for k, v in items.items()), ""]


def render(root_pyproject: str, plugin: str, version: str) -> str:
    """The plugin's pyproject.toml, rendered from the text of the root one."""
    if plugin not in PLUGINS:
        raise ValueError(f"unknown plugin {plugin!r}; the plugins are {', '.join(PLUGINS)}")
    major = int(plugin[2:])
    package = f"dyng_{plugin}"
    data = tomllib.loads(root_pyproject)
    core = data["project"]
    if core["name"] != "dyng":
        raise ValueError("the source tree's pyproject.toml is not the one of dyng")
    skb = data["tool"]["scikit-build"]
    defines = dict(skb.get("cmake", {}).get("define", {}))
    version_meta = [m for m in data["tool"].get("dynamic-metadata", []) if m["field"] == "version"]
    if len(version_meta) != 1:
        raise ValueError("the root pyproject.toml has no [[tool.dynamic-metadata]] for version")

    project: dict[str, Any] = {
        "name": f"dyng-{plugin}",
        "dynamic": ["version"],
        "description": (
            f"dynG's CUDA {major} backends: the CUDA plugin of the dyng package "
            f'(pip install "dyng[{plugin}]")'
        ),
        "readme": PLUGIN_README,
        "requires-python": core["requires-python"],
        "license": PLUGIN_LICENSE_EXPRESSION,
        "license-files": [*core["license-files"], *PLUGIN_EXTRA_LICENSE_FILES],
        "authors": core["authors"],
        "keywords": [*core["keywords"], "CUDA", f"CUDA {major}"],
        # The plugin package is private and untyped (no py.typed): not "Typing :: Typed".
        "classifiers": [
            *(c for c in core["classifiers"] if not c.startswith("Typing ::")),
            "Environment :: GPU :: NVIDIA CUDA",
            f"Environment :: GPU :: NVIDIA CUDA :: {major}",
        ],
        # The typed layer comes from dyng; the native module must match it exactly.
        "dependencies": [f"dyng=={version}"],
    }
    # The scikit-build-core settings of the CPU package that the plugin shares (parsed as nested
    # tables, written back as dotted keys); not the sdist settings (no plugin sdist is built).
    tool: dict[str, Any] = {"minimum-version": skb["minimum-version"]}
    for section, keys in (
        ("cmake", ("version", "build-type")),
        ("ninja", ("version",)),
        ("build", ("targets",)),
        ("install", ("components",)),
        ("wheel", ("py-api",)),
    ):
        for k in keys:
            if k in skb.get(section, {}):
                tool[f"{section}.{k}"] = skb[section][k]
    # scikit-build-core names a package after its directory: prepare() copies the package source
    # to python/plugin/dyng_cu<N> in the (disposable) plugin tree.
    tool["wheel.packages"] = [f"{PLUGIN_PACKAGE_DIR}/{package}"]
    tool["editable.verbose"] = False
    defines.update(
        {
            "DYNG_ENABLE_CUDA": "ON",
            "DYNG_ENABLE_OPENMP": "ON",
            "DYNG_PYTHON_PLUGIN": plugin,
            "CMAKE_CUDA_RUNTIME_LIBRARY": "Static",
            "DYNG_CUDA_ARCHITECTURES": "release",
        }
    )

    out = [
        # REUSE-IgnoreStart (the header of the generated file, not of this one)
        "# SPDX-FileCopyrightText: 2026 The dynG Authors",
        "# SPDX-License-Identifier: Apache-2.0",
        # REUSE-IgnoreEnd
        "#",
        f"# GENERATED by ci/plugin_pyproject.py --plugin {plugin} from the pyproject.toml of dyng "
        f"{version}:",
        f"# the CUDA {major} plugin distribution dyng-{plugin} (PLAN 5.4 and 7.7; ADR 0030). "
        "Do not edit.",
        "",
        *_table("build-system", data["build-system"]),
        *_table("project", project),
        *_table('project.entry-points."dyng.backends"', {plugin: package}),
        *_table("project.urls", core["urls"]),
        *_table("tool.scikit-build", tool),
        *_table("tool.scikit-build.cmake.define", defines),
        "[[tool.dynamic-metadata]]",
        *(f"{_key(k)} = {_toml(v)}" for k, v in version_meta[0].items()),
        "",
    ]
    return "\n".join(out)


def find_cuda_eula(cuda_root: Path | None) -> Path:
    """The EULA of a CUDA toolkit: ``<root>/EULA.txt`` (NVIDIA's installers and packages),
    ``<root>/LICENSE`` when it is that EULA (a conda environment of the ``nvidia`` channel), or
    the licence of the ``cuda-cudart-static`` package that ``<root>/conda-meta`` names (a conda
    environment of conda-forge's CUDA packages)."""
    roots: list[Path] = []
    if cuda_root is not None:
        roots.append(cuda_root)
    else:
        for var in ("CUDA_HOME", "CUDA_PATH"):
            if os.environ.get(var):
                roots.append(Path(os.environ[var]))
        if os.environ.get("CUDACXX"):
            roots.append(Path(os.environ["CUDACXX"]).resolve().parent.parent)
    for root in roots:
        candidates = [root / "EULA.txt", root / "LICENSE"]
        # A conda environment of conda-forge's CUDA packages: the EULA is the licence file of the
        # cudart package, kept in the package cache that conda-meta names.
        for meta in sorted((root / "conda-meta").glob("cuda-cudart-static*.json")):
            pkg = json.loads(meta.read_text()).get("extracted_package_dir")
            if pkg:
                candidates.append(Path(pkg) / "info" / "licenses" / "LICENSE")
        for candidate in candidates:
            if candidate.is_file() and _is_cuda_eula(candidate):
                return candidate
    where = ", ".join(str(r) for r in roots) or "(no --cuda-root, CUDA_HOME or CUDACXX)"
    raise FileNotFoundError(
        f"no CUDA Toolkit EULA (EULA.txt) under {where}; pass --cuda-eula <file> (NVIDIA's "
        "packages ship it as <toolkit>/EULA.txt; conda's as <prefix>/LICENSE or in the package "
        "cache, pkgs/cuda-cudart-static*/info/licenses/LICENSE)"
    )


def _is_cuda_eula(path: Path) -> bool:
    head = path.read_text(errors="replace")[:4000]
    return "End User License Agreement" in head and "CUDA" in head


def prepare(project_dir: Path, plugin: str, eula: Path) -> Path:
    """Render the plugin's pyproject.toml into ``project_dir`` and copy the EULA next to it."""
    pyproject = project_dir / "pyproject.toml"
    version = (project_dir / "VERSION").read_text().strip()
    text = render(pyproject.read_text(), plugin, version)
    for required in (PLUGIN_PACKAGE_SOURCE + "/__init__.py", PLUGIN_README):
        if not (project_dir / required).is_file():
            raise FileNotFoundError(f"{project_dir / required} is missing")
    if not _is_cuda_eula(eula):
        raise ValueError(f"{eula} does not look like the CUDA Toolkit's EULA")
    shutil.copyfile(eula, project_dir / EULA_NAME)
    package = project_dir / PLUGIN_PACKAGE_DIR / f"dyng_{plugin}"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(
        project_dir / PLUGIN_PACKAGE_SOURCE, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    pyproject.write_text(text)
    return pyproject


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plugin", required=True, choices=PLUGINS)
    p.add_argument("--project-dir", type=Path, help="the source tree to turn into the plugin's")
    p.add_argument("--cuda-root", type=Path, help="the CUDA toolkit the wheel is built with")
    p.add_argument("--cuda-eula", type=Path, help="the toolkit's EULA (default: found in it)")
    p.add_argument("--print", action="store_true", help="print the rendered file and exit")
    args = p.parse_args(argv)
    if args.print:
        version = (ROOT / "VERSION").read_text().strip()
        sys.stdout.write(render((ROOT / "pyproject.toml").read_text(), args.plugin, version))
        return 0
    if args.project_dir is None:
        p.error("--project-dir is required (or --print)")
    eula = args.cuda_eula or find_cuda_eula(args.cuda_root)
    out = prepare(args.project_dir, args.plugin, eula)
    print(f"plugin_pyproject: {out} is the pyproject.toml of dyng-{args.plugin}; EULA from {eula}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
