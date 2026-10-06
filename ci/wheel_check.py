#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Check built distributions of the `dyng` package (PLAN Sections 7.7 and 8.8; ADR 0025).

For every wheel: the size budget (PLAN 7.7: fail above 90 MB; PyPI's per-file limit is 100 MB),
the file name (``dyng-<VERSION>-cp312-abi3-<platform>.whl``, with ``--platform`` the expected
platform tag, e.g. ``manylinux_2_28_x86_64``), the contents (the extension module
``dyng/_core.abi3.so``, the typed layer with ``py.typed`` and the stubs, the licence files, the
``dyng`` console script in ``entry_points.txt``) and, with ``--require-libgomp``, the OpenMP
runtime bundled by auditwheel under ``dyng.libs/``. For every sdist: the files a source build
needs (``VERSION``, ``pyproject.toml``, ``CMakeLists.txt``, ``cpp/``, ``python/``) and none of the
excluded trees (``parity/``, ``.github/``, ...) or repository-only files (the CC-BY-SA-4.0 Code
of Conduct, governance and tool configuration), and its size. Both kinds carry the licence
metadata the author decided on 2026-09-30 (GOVERNANCE.md): ``License-Expression:``
:data:`LICENSE_EXPRESSION` in ``METADATA`` / ``PKG-INFO`` (which ``pyproject.toml``'s ``license``
must equal) and a ``License-File:`` line for every licence file. ``VERSION`` must be a canonical
PEP 440 version (the distributions carry the normalised form, so any other spelling would give
file names that no check expects). ``--release-metadata`` checks ``VERSION``, ``CHANGELOG.md``
and ``CITATION.cff`` against each other (:func:`check_release_metadata`).

The CUDA plugin wheels (``dyng_cu12-*.whl``, ``dyng_cu13-*.whl``; PLAN 5.4 and 7.7, ADR 0030) are
recognised by their names and checked for what makes a plugin (:func:`check_plugin_wheel`): only
the import package ``dyng_cu<N>/`` with its module ``_core.abi3.so`` (no ``dyng/``), the entry
point ``cu<N> = dyng_cu<N>`` in the group ``dyng.backends``, ``Requires-Dist: dyng==<VERSION>``,
the licence files of the CPU wheel plus ``THIRD_PARTY_LICENSES_CUDA.txt`` and
``NVIDIA_CUDA_EULA.txt``, :data:`PLUGIN_LICENSE_EXPRESSION`, the same size budget, and a module
that needs neither ``libcuda`` (the user's driver) nor ``libcudart`` (linked statically): no
``DT_NEEDED`` entry and no bundled copy of either. The CPU wheel must offer the extras ``cu12``
and ``cu13`` pinned to its own version.

A release (``release.yml``) publishes :func:`release_distributions`: the core ``dyng`` alone up
to v0.1.x, and from 0.2.0 (:data:`PLUGINS_SINCE`) also ``dyng-cu12`` and ``dyng-cu13``, each
through its own pair of publishing environments; ``--release-set`` checks that the files are
exactly those, and ``--split`` sorts them into one directory per distribution.

Usage::

    python3 ci/wheel_check.py dist/*.whl dist/*.tar.gz --platform manylinux_2_28_x86_64 \\
        --require-libgomp
    python3 ci/wheel_check.py dist/dyng_cu13-*.whl --platform manylinux_2_28_x86_64 \\
        --require-libgomp                  # a CUDA plugin wheel (recognised by its name)
    python3 ci/wheel_check.py --version-info   # "<VERSION> <pre-release: true|false>"
    python3 ci/wheel_check.py --release-metadata [--release-date today]
    python3 ci/wheel_check.py --release-distributions   # release.yml's publishing matrix (JSON)
    python3 ci/wheel_check.py dist/* --release-set [--split dists] ...   # the whole release
    python3 ci/wheel_check.py --self-test

Exit status 0 when every file passes; the report lists each check.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import re
import struct
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: PLAN Section 7.7: CI fails a wheel above 90 MB.
MAX_WHEEL_BYTES = 90 * 1000 * 1000
#: An sdist far above this has picked up build trees or data.
MAX_SDIST_BYTES = 20 * 1000 * 1000

SDIST_REQUIRED = (
    "VERSION",
    "pyproject.toml",
    "CMakeLists.txt",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.txt",
    # the CUDA plugins are built from the sdist (ci/plugin_pyproject.py; ADR 0030)
    "THIRD_PARTY_LICENSES_CUDA.txt",
    "python/plugin/README.md",
    "python/plugin/dyng_plugin/__init__.py",
)
SDIST_REQUIRED_DIRS = ("cpp/include/dyng/", "cpp/src/", "python/dyng/", "python/bindings/")
SDIST_FORBIDDEN_DIRS = (".github/", "parity/", "build/", "docs/adr/", "tools/", "ci/")
#: Repository-only files (pyproject.toml's sdist.exclude): the sdist is Apache-2.0 only.
SDIST_FORBIDDEN_FILES = (
    "CODE_OF_CONDUCT.md",
    "LICENSES/CC-BY-SA-4.0.txt",
    "GOVERNANCE.md",
    "MAINTAINERS.md",
    "SUPPORT.md",
    "CMakePresets.json",
    ".pre-commit-config.yaml",
    ".readthedocs.yaml",
    ".clang-tidy",
)
#: The licence files every distribution carries (pyproject.toml's license-files).
LICENSE_FILES = ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.txt")
#: The SPDX licence expression of the distributions (core metadata 2.4, PEP 639): dynG's
#: Apache-2.0 and the licences of what the wheel bundles or links statically
#: (THIRD_PARTY_LICENSES.txt: nanobind, robin-map, the GCC runtime). The author's decision of
#: 2026-09-30 (GOVERNANCE.md, approvals log); pyproject.toml's `license` must be this string.
LICENSE_EXPRESSION = (
    "Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1"
)

#: The CUDA plugins (PLAN 5.4 and 7.7; ADR 0030): distributions dyng-cu<N>, packages dyng_cu<N>.
PLUGINS = ("cu12", "cu13")
#: The licence files a plugin wheel carries besides :data:`LICENSE_FILES`: the licences of the
#: CUDA Toolkit parts in its module (THIRD_PARTY_LICENSES_CUDA.txt) and the toolkit's EULA, copied
#: from the toolkit the wheel is built with (ci/plugin_pyproject.py).
PLUGIN_EXTRA_LICENSE_FILES = ("THIRD_PARTY_LICENSES_CUDA.txt", "NVIDIA_CUDA_EULA.txt")
#: The SPDX licence expression of the plugin wheels. Until the author decides otherwise it is the
#: CPU wheel's expression: whether the plugins' expression should also name what they add (the
#: CUDA runtime under NVIDIA's EULA, e.g. LicenseRef-NVIDIA-CUDA-EULA, and CCCL's
#: Apache-2.0 WITH LLVM-exception) is a licensing decision of the author (GOVERNANCE.md, open
#: decisions; ADR 0030). The licence files are complete either way.
PLUGIN_LICENSE_EXPRESSION = LICENSE_EXPRESSION
#: The first release (major, minor) whose distributions include the CUDA plugin wheels (PLAN
#: Appendix F: 0.2.0, its release candidates included); v0.1.x releases keep the core alone.
PLUGINS_SINCE = (0, 2)


def release_distributions(v: str) -> list[dict[str, str]]:
    """The distributions a release of version ``v`` publishes (release.yml's matrix): for each,
    the distribution's ``name``, the ``prefix`` of its file names and the ``suffix`` of its
    publishing environments (``testpypi<suffix>`` and ``pypi<suffix>``: the trusted publishers of
    ``dyng`` are bound to ``testpypi`` / ``pypi``, those of ``dyng-cu<N>`` to ``testpypi-cu<N>`` /
    ``pypi-cu<N>``; GOVERNANCE.md, approvals log)."""
    canonical_version(v)
    m = re.match(r"(\d+)\.(\d+)", v)
    release = (int(m.group(1)), int(m.group(2))) if m else (int(v), 0)
    out = [{"name": "dyng", "prefix": f"dyng-{v}", "suffix": ""}]
    if release >= PLUGINS_SINCE:
        out += [
            {"name": f"dyng-{p}", "prefix": f"dyng_{p}-{v}", "suffix": f"-{p}"} for p in PLUGINS
        ]
    return out


def distribution_of(filename: str) -> str | None:
    """The distribution a file name belongs to (``dyng``, ``dyng-cu12``, ...), or None."""
    m = re.match(r"(dyng(?:_cu\d+)?)-", filename)
    return m.group(1).replace("_", "-") if m else None


def check_release_set(paths: list[Path], v: str) -> list[str]:
    """The files of a release are exactly those :func:`release_distributions` names: the core's
    sdist and wheel(s), and one or more wheels of each plugin, all of version ``v``."""
    errors = []
    wanted = {d["name"]: d["prefix"] for d in release_distributions(v)}
    found: dict[str, list[str]] = {}
    for p in paths:
        dist = distribution_of(p.name)
        if dist not in wanted:
            errors.append(f"{p.name}: not a distribution of this release ({', '.join(wanted)})")
            continue
        if not p.name.startswith(wanted[dist] + (".tar.gz" if p.name.endswith(".gz") else "-")):
            errors.append(f"{p.name}: not version {v}")
        found.setdefault(dist, []).append(p.name)
    for dist in wanted:
        names = found.get(dist, [])
        if not any(n.endswith(".whl") for n in names):
            errors.append(f"{dist}: no wheel")
        if dist == "dyng" and not any(n.endswith(".tar.gz") for n in names):
            errors.append("dyng: no sdist")
        if dist != "dyng" and any(n.endswith(".tar.gz") for n in names):
            errors.append(f"{dist}: an sdist (the plugins are built from the core sdist)")
    return errors


def split_release(paths: list[Path], out: Path, v: str) -> dict[str, list[Path]]:
    """Copy each file of a checked release set to ``out/<distribution>/`` (one directory per
    publishing job of release.yml); returns the files per distribution."""
    errors = check_release_set(paths, v)
    if errors:
        raise ValueError("; ".join(errors))
    result: dict[str, list[Path]] = {}
    for p in paths:
        dist = distribution_of(p.name)
        assert dist is not None
        target = out / dist / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        result.setdefault(dist, []).append(target)
    return result


#: Shared libraries a plugin's module must neither need nor bundle (a file name matching this,
#: also as renamed by auditwheel, libcudart-1a2b3c4d.so.13): the driver (libcuda, libnvidia-*)
#: is the user's, the CUDA runtime (libcudart) is linked statically.
PLUGIN_FORBIDDEN_LIBRARY = re.compile(r"^(?:libcuda[.-]|libcudart|libnvidia-)")

#: A canonical PEP 440 public version (what packaging.version.Version(v) prints back unchanged),
#: without epoch and local part: 0.1.0, 0.1.0rc1, 0.1.0.post1, 0.1.0.dev0.
_CANONICAL = re.compile(
    r"(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*))*"
    r"(?P<pre>(?:a|b|rc)(?:0|[1-9]\d*))?(?:\.post(?:0|[1-9]\d*))?(?P<dev>\.dev(?:0|[1-9]\d*))?"
)


def canonical_version(v: str) -> tuple[str, bool]:
    """``(v, is_prerelease)`` for a canonical PEP 440 version; ValueError otherwise."""
    m = _CANONICAL.fullmatch(v)
    if m is None:
        raise ValueError(
            f"VERSION {v!r} is not a canonical PEP 440 version (e.g. 0.1.0, 0.1.0rc1, "
            "0.1.0.dev0; SemVer spellings such as 0.1.0-rc.1 are not)"
        )
    return v, bool(m.group("pre") or m.group("dev"))


def version() -> str:
    """The version of VERSION (canonical PEP 440; ValueError otherwise)."""
    return canonical_version((ROOT / "VERSION").read_text().strip())[0]


#: The repository whose CHANGELOG link references are checked.
REPOSITORY_URL = "https://github.com/dyng-dev/dyng"


def _cff_field(text: str, key: str) -> str | None:
    """A top-level scalar of CITATION.cff (``key: value`` at column 0, quotes removed)."""
    m = re.search(rf"(?m)^{re.escape(key)}:\s*\"?([^\"\n]*?)\"?\s*$", text)
    return m.group(1) if m else None


def check_release_metadata(root: Path = ROOT, release_date: str | None = None) -> list[str]:
    """The failed checks of VERSION, CHANGELOG.md and CITATION.cff against each other
    (docs/developer/release.md, steps 5, 6 and 9). For a release or a release candidate:
    CITATION.cff's ``version`` is VERSION; CHANGELOG.md has an ``## [Unreleased]`` section and
    under it ``## [VERSION] - <date>`` with ``<date>`` CITATION.cff's ``date-released``; the link
    references ``[Unreleased]: .../compare/vVERSION...main`` and ``[VERSION]: .../vVERSION``
    exist. With ``release_date`` (YYYY-MM-DD; the day of the tag), the date must be that day.
    Between releases (a ``.devN`` VERSION) the last release is checked the same way: CITATION.cff
    names it, and the CHANGELOG has its section with the same date."""
    errors: list[str] = []
    try:
        v, _ = canonical_version((root / "VERSION").read_text().strip())
    except ValueError as e:
        return [str(e)]
    cff = (root / "CITATION.cff").read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    cited, date = _cff_field(cff, "version"), _cff_field(cff, "date-released")
    dev = re.search(r"\.dev\d+$", v) is not None
    release = cited if dev else v
    if not dev and cited != v:
        errors.append(f"CITATION.cff: version {cited}, VERSION is {v}")
    if date is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        errors.append(f"CITATION.cff: date-released {date!r} is not YYYY-MM-DD")
    headings = re.findall(r"(?m)^## \[([^\]]+)\](?: - (\S+))?\s*$", changelog)
    names = [h[0] for h in headings]
    if not names or names[0] != "Unreleased":
        errors.append("CHANGELOG.md: the first section must be '## [Unreleased]'")
    found = [d for name, d in headings if name == release]
    if len(found) != 1:
        errors.append(f"CHANGELOG.md: {len(found)} sections '## [{release}] - <date>' (want 1)")
    elif found[0] != date:
        errors.append(
            f"CHANGELOG.md: '## [{release}] - {found[0]}', CITATION.cff date-released {date}"
        )
    elif names.index(release) != 1:
        errors.append(f"CHANGELOG.md: '## [{release}]' must follow '## [Unreleased]'")
    if not dev:
        links = {
            "Unreleased": f"[Unreleased]: {REPOSITORY_URL}/compare/v{v}...main",
            v: f"[{v}]: {REPOSITORY_URL}/",
        }
        for name, prefix in links.items():
            line = next((x for x in changelog.splitlines() if x.startswith(f"[{name}]: ")), None)
            if (
                line is None
                or not line.startswith(prefix)
                or (name == v and not line.rstrip().endswith(f"v{v}"))
            ):
                errors.append(f"CHANGELOG.md: link reference {line!r}, want {prefix}...")
    if release_date is not None and date != release_date:
        errors.append(
            f"the release date {date} (CHANGELOG.md, CITATION.cff) is not the day of the tag "
            f"{release_date}: fix both in a small pull request first"
        )
    return errors


def check_pyproject(path: Path = ROOT / "pyproject.toml") -> list[str]:
    """The failed checks of pyproject.toml's licence metadata (empty if it passes)."""
    project = tomllib.loads(path.read_text())["project"]
    errors: list[str] = []
    if project.get("license") != LICENSE_EXPRESSION:
        errors.append(
            f"pyproject.toml: license = {project.get('license')!r}, expected {LICENSE_EXPRESSION!r}"
        )
    return errors


def check_metadata(
    text: str,
    where: str,
    *,
    expression: str = LICENSE_EXPRESSION,
    license_files: tuple[str, ...] = LICENSE_FILES,
) -> list[str]:
    """The failed checks of a core-metadata file (a wheel's METADATA, an sdist's PKG-INFO)."""
    errors: list[str] = []
    headers = text.split("\n\n", 1)[0]
    expressions = re.findall(r"^License-Expression: (.*)$", headers, re.M)
    if expressions != [expression]:
        errors.append(f"{where}: License-Expression {expressions} != [{expression!r}]")
    if re.search(r"^License: ", headers, re.M):
        errors.append(f"{where}: a legacy License: field next to License-Expression")
    files = set(re.findall(r"^License-File: (.*)$", headers, re.M))
    for lic in license_files:
        if lic not in files:
            errors.append(f"{where}: no License-File: {lic}")
    return errors


def check_extras(text: str, version: str) -> list[str]:
    """The CPU wheel's extras cu12 / cu13 pin the plugin of exactly its version (ADR 0030)."""
    headers = text.split("\n\n", 1)[0]
    errors: list[str] = []
    extras = set(re.findall(r"^Provides-Extra: (.*)$", headers, re.M))
    requires = set(re.findall(r"^Requires-Dist: (.*)$", headers, re.M))
    for plugin in PLUGINS:
        want = f'dyng-{plugin}=={version}; extra == "{plugin}"'
        if plugin not in extras or want not in requires:
            errors.append(f"METADATA: the extra {plugin} does not require dyng-{plugin}=={version}")
    return errors


def elf_needed(data: bytes) -> list[str]:
    """The DT_NEEDED entries of a 64-bit little-endian ELF shared object."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF file")
    e_phoff, e_phentsize, e_phnum = (
        struct.unpack_from("<Q", data, 0x20)[0],
        struct.unpack_from("<H", data, 0x36)[0],
        struct.unpack_from("<H", data, 0x38)[0],
    )
    loads: list[tuple[int, int, int]] = []  # (vaddr, offset, filesz)
    dynamic: tuple[int, int] | None = None
    for i in range(e_phnum):
        p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz = struct.unpack_from(
            "<IIQQQQ", data, e_phoff + i * e_phentsize
        )
        if p_type == 1:  # PT_LOAD
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == 2:  # PT_DYNAMIC
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        return []
    needed: list[int] = []
    strtab = None
    for off in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, val = struct.unpack_from("<qQ", data, off)
        if tag == 0:  # DT_NULL
            break
        if tag == 1:  # DT_NEEDED
            needed.append(val)
        elif tag == 5:  # DT_STRTAB (a virtual address)
            strtab = val
    if strtab is None:
        raise ValueError("no DT_STRTAB")
    base = next((o + strtab - v for v, o, n in loads if v <= strtab < v + n), None)
    if base is None:
        raise ValueError("DT_STRTAB outside every PT_LOAD segment")
    return [data[base + i : data.index(b"\0", base + i)].decode() for i in needed]


def check_wheel(
    path: Path, *, platform: str | None, require_libgomp: bool, expect_version: str
) -> list[str]:
    """The failed checks of one wheel (empty if it passes)."""
    errors: list[str] = []
    size = path.stat().st_size
    if size > MAX_WHEEL_BYTES:
        errors.append(f"{size} bytes: above the 90 MB budget of PLAN 7.7")
    m = re.fullmatch(r"dyng-(?P<v>[^-]+)-cp312-abi3-(?P<plat>[\w.]+)\.whl", path.name)
    if not m:
        errors.append("the file name is not dyng-<version>-cp312-abi3-<platform>.whl")
    else:
        if m.group("v") != expect_version:
            errors.append(f"version {m.group('v')} != VERSION {expect_version}")
        if platform is not None and platform not in m.group("plat").split("."):
            errors.append(f"platform tag {m.group('plat')} does not include {platform}")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        entry = next((n for n in names if n.endswith(".dist-info/entry_points.txt")), None)
        entry_text = z.read(entry).decode() if entry else ""
        wheel_meta = next((n for n in names if n.endswith(".dist-info/WHEEL")), None)
        wheel_text = z.read(wheel_meta).decode() if wheel_meta else ""
        meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        meta_text = z.read(meta).decode() if meta else ""
    for required in (
        "dyng/_core.abi3.so",
        "dyng/__init__.py",
        "dyng/py.typed",
        "dyng/_core.pyi",
        "dyng/cli/__init__.py",
        "dyng/__main__.py",
    ):
        if required not in names:
            errors.append(f"missing {required}")
    for lic in LICENSE_FILES:
        if not any(n.endswith(f".dist-info/licenses/{lic}") for n in names):
            errors.append(f"missing the licence file {lic} in .dist-info/licenses")
    if not re.search(r"^dyng\s*=\s*dyng\.cli:main\s*$", entry_text, re.M):
        errors.append("entry_points.txt has no console script dyng = dyng.cli:main")
    if "Root-Is-Purelib: false" not in wheel_text:
        errors.append("WHEEL: expected Root-Is-Purelib: false (a platform wheel)")
    if meta is None:
        errors.append("missing .dist-info/METADATA")
    else:
        errors += check_metadata(meta_text, "METADATA")
        errors += check_extras(meta_text, expect_version)
    libgomp = [n for n in names if re.match(r"dyng\.libs/libgomp[-.]", n)]
    if require_libgomp and not libgomp:
        errors.append("the OpenMP runtime is not bundled (dyng.libs/libgomp-*.so*)")
    stray = [n for n in names if n.endswith((".a", ".o", ".cpp", ".hpp", ".cmake"))]
    if stray:
        errors.append(f"build files in the wheel: {', '.join(sorted(stray)[:5])}")
    return errors


def check_plugin_wheel(
    path: Path, *, platform: str | None, require_libgomp: bool, expect_version: str
) -> list[str]:
    """The failed checks of one CUDA plugin wheel, ``dyng_cu<N>-...whl`` (empty if it passes)."""
    errors: list[str] = []
    size = path.stat().st_size
    if size > MAX_WHEEL_BYTES:
        errors.append(f"{size} bytes: above the 90 MB budget of PLAN 7.7")
    m = re.fullmatch(
        r"dyng_(?P<plugin>cu\d+)-(?P<v>[^-]+)-cp312-abi3-(?P<plat>[\w.]+)\.whl", path.name
    )
    if not m or m.group("plugin") not in PLUGINS:
        return [
            *errors,
            f"the file name is not dyng_<{'|'.join(PLUGINS)}>-<version>-cp312-abi3-<platform>.whl",
        ]
    plugin = m.group("plugin")
    package = f"dyng_{plugin}"
    if m.group("v") != expect_version:
        errors.append(f"version {m.group('v')} != VERSION {expect_version}")
    if platform is not None and platform not in m.group("plat").split("."):
        errors.append(f"platform tag {m.group('plat')} does not include {platform}")
    module = f"{package}/_core.abi3.so"
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())

        def read(suffix: str) -> str:
            n = next((n for n in names if n.endswith(suffix)), None)
            return z.read(n).decode() if n else ""

        entry_text = read(".dist-info/entry_points.txt")
        wheel_text = read(".dist-info/WHEEL")
        meta_text = read(".dist-info/METADATA")
        module_data = z.read(module) if module in names else b""
    for required in (module, f"{package}/__init__.py"):
        if required not in names:
            errors.append(f"missing {required}")
    foreign = sorted(
        n for n in names if not n.startswith((f"{package}/", f"{package}.libs/", f"{package}-"))
    )
    if foreign:
        errors.append(f"files outside the plugin's package: {', '.join(foreign[:5])}")
    for lic in (*LICENSE_FILES, *PLUGIN_EXTRA_LICENSE_FILES):
        if not any(n.endswith(f".dist-info/licenses/{lic}") for n in names):
            errors.append(f"missing the licence file {lic} in .dist-info/licenses")
    if not re.search(rf"^\[dyng\.backends\]\s*^{plugin}\s*=\s*{package}\s*$", entry_text, re.M):
        errors.append(f"entry_points.txt has no [dyng.backends] {plugin} = {package}")
    if "console_scripts" in entry_text:
        errors.append("a plugin has no console scripts (the dyng command belongs to dyng)")
    if "Root-Is-Purelib: false" not in wheel_text:
        errors.append("WHEEL: expected Root-Is-Purelib: false (a platform wheel)")
    if not meta_text:
        errors.append("missing .dist-info/METADATA")
    else:
        errors += check_metadata(
            meta_text,
            "METADATA",
            expression=PLUGIN_LICENSE_EXPRESSION,
            license_files=(*LICENSE_FILES, *PLUGIN_EXTRA_LICENSE_FILES),
        )
        headers = meta_text.split("\n\n", 1)[0]
        if not re.search(rf"^Name: dyng-{plugin}$", headers, re.M):
            errors.append(f"METADATA: Name is not dyng-{plugin}")
        requires = re.findall(r"^Requires-Dist: (.*)$", headers, re.M)
        if requires != [f"dyng=={expect_version}"]:
            errors.append(f"METADATA: Requires-Dist {requires} != ['dyng=={expect_version}']")
    libs = [n for n in names if n.startswith(f"{package}.libs/")]
    bundled = [n for n in libs if PLUGIN_FORBIDDEN_LIBRARY.match(n.rsplit("/", 1)[-1])]
    if bundled:
        errors.append(f"bundles the CUDA driver or runtime: {', '.join(bundled)}")
    if require_libgomp and not any(re.match(rf"{package}\.libs/libgomp[-.]", n) for n in libs):
        errors.append(f"the OpenMP runtime is not bundled ({package}.libs/libgomp-*.so*)")
    if module_data:
        try:
            needed = elf_needed(module_data)
        except ValueError as e:
            errors.append(f"{module}: {e}")
        else:
            bad = [n for n in needed if PLUGIN_FORBIDDEN_LIBRARY.match(n)]
            if bad:
                errors.append(
                    f"{module} needs {', '.join(bad)} (the driver is loaded at run time, "
                    "the CUDA runtime must be linked statically)"
                )
    stray = [n for n in names if n.endswith((".a", ".o", ".cpp", ".hpp", ".cu", ".cmake"))]
    if stray:
        errors.append(f"build files in the wheel: {', '.join(sorted(stray)[:5])}")
    return errors


def check_sdist(path: Path, *, expect_version: str) -> list[str]:
    """The failed checks of one sdist."""
    errors: list[str] = []
    size = path.stat().st_size
    if size > MAX_SDIST_BYTES:
        errors.append(f"{size} bytes: above {MAX_SDIST_BYTES} (build trees or data included?)")
    top = f"dyng-{expect_version}/"
    if not path.name == f"dyng-{expect_version}.tar.gz":
        errors.append(f"the file name is not dyng-{expect_version}.tar.gz")
    with tarfile.open(path) as t:
        names = [n[len(top) :] for n in t.getnames() if n.startswith(top)]
        pkg_info = t.extractfile(f"{top}PKG-INFO") if f"{top}PKG-INFO" in t.getnames() else None
        pkg_info_text = pkg_info.read().decode() if pkg_info else ""
    present = set(names)
    if pkg_info is None:
        errors.append("missing PKG-INFO")
    else:
        errors += check_metadata(pkg_info_text, "PKG-INFO")
    for required in SDIST_REQUIRED:
        if required not in present:
            errors.append(f"missing {required}")
    for d in SDIST_REQUIRED_DIRS:
        if not any(n.startswith(d) for n in names):
            errors.append(f"missing the directory {d}")
    for d in SDIST_FORBIDDEN_DIRS:
        bad = [n for n in names if n.startswith(d)]
        if bad:
            errors.append(f"excluded directory {d} is in the sdist ({len(bad)} files)")
    for f in SDIST_FORBIDDEN_FILES:
        if f in present:
            errors.append(f"repository-only file {f} is in the sdist")
    return errors


def run(paths: list[Path], *, platform: str | None, require_libgomp: bool) -> int:
    """Check every file; print a report; return the exit status."""
    try:
        v = version()
    except ValueError as e:
        print(f"wheel_check: {e}", file=sys.stderr)
        return 1
    failed = 0
    if not paths:
        print("wheel_check: no files given", file=sys.stderr)
        return 1
    for e in check_pyproject():
        print(f"wheel_check: {e}", file=sys.stderr)
        failed += 1
    for p in paths:
        if p.name.startswith("dyng_cu") and p.name.endswith(".whl"):
            errors = check_plugin_wheel(
                p, platform=platform, require_libgomp=require_libgomp, expect_version=v
            )
        elif p.name.endswith(".whl"):
            errors = check_wheel(
                p, platform=platform, require_libgomp=require_libgomp, expect_version=v
            )
        elif p.name.endswith(".tar.gz"):
            errors = check_sdist(p, expect_version=v)
        else:
            errors = [f"not a wheel or an sdist: {p.name}"]
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0.0
        status = "ok" if not errors else "FAILED"
        print(f"{status:6} {p.name} ({size_mb:.2f} MB)")
        for e in errors:
            print(f"       - {e}")
        failed += bool(errors)
    return 1 if failed else 0


def _fake_elf(needed: list[str]) -> bytes:
    """A minimal ELF64 shared object whose dynamic section lists ``needed`` (for the self-test)."""
    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in needed)
    str_off = 64 + 2 * 56
    dyn_off = (str_off + len(strtab) + 7) // 8 * 8
    offsets, pos = [], 1
    for n in needed:
        offsets.append(pos)
        pos += len(n) + 1
    dyn = b"".join(struct.pack("<qQ", 1, o) for o in offsets)
    dyn += struct.pack("<qQ", 5, str_off) + struct.pack("<qQ", 0, 0)
    total = dyn_off + len(dyn)
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header += struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0, 0)
    ph = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, total, total, 0x1000)
    ph += struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn), len(dyn), 8)
    body = header + ph + strtab
    return body + bytes(dyn_off - len(body)) + dyn


def _self_test() -> int:
    """Broken distributions are rejected, a good one passes."""
    v = version()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        def wheel(name: str, files: dict[str, str] | dict[str, str | bytes]) -> Path:
            p = d / name
            with zipfile.ZipFile(p, "w") as z:
                for n, text in files.items():
                    z.writestr(n, text)
            return p

        info = f"dyng-{v}.dist-info"
        good_meta = (
            f"Metadata-Version: 2.4\nName: dyng\nVersion: {v}\n"
            f"License-Expression: {LICENSE_EXPRESSION}\n"
            + "".join(f"License-File: {lic}\n" for lic in LICENSE_FILES)
            + "".join(
                f'Provides-Extra: {x}\nRequires-Dist: dyng-{x}=={v}; extra == "{x}"\n'
                for x in PLUGINS
            )
            + "\n# dyng\n\nLicense: words in the description do not count\n"
        )
        good_files = {
            "dyng/_core.abi3.so": "x",
            "dyng/__init__.py": "",
            "dyng/py.typed": "",
            "dyng/_core.pyi": "",
            "dyng/cli/__init__.py": "",
            "dyng/__main__.py": "",
            "dyng.libs/libgomp-1234abcd.so.1.0.0": "x",
            f"{info}/licenses/LICENSE": "",
            f"{info}/licenses/NOTICE": "",
            f"{info}/licenses/THIRD_PARTY_LICENSES.txt": "",
            f"{info}/entry_points.txt": "[console_scripts]\ndyng = dyng.cli:main\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
            f"{info}/METADATA": good_meta,
        }
        plat = "manylinux_2_28_x86_64"
        good = wheel(f"dyng-{v}-cp312-abi3-{plat}.whl", good_files)
        assert not check_wheel(good, platform=plat, require_libgomp=True, expect_version=v)
        cases = {
            "no module": {k: x for k, x in good_files.items() if k != "dyng/_core.abi3.so"},
            "no libgomp": {k: x for k, x in good_files.items() if "libgomp" not in k},
            "no script": {**good_files, f"{info}/entry_points.txt": ""},
            "stray": {**good_files, "dyng/x.cpp": ""},
            "no third-party licences": {
                k: x for k, x in good_files.items() if "THIRD_PARTY" not in k
            },
            "no METADATA": {k: x for k, x in good_files.items() if not k.endswith("METADATA")},
            "Apache-2.0 only": {
                **good_files,
                f"{info}/METADATA": good_meta.replace(LICENSE_EXPRESSION, "Apache-2.0"),
            },
            "legacy License field": {
                **good_files,
                f"{info}/METADATA": good_meta.replace("Name:", "License: Apache\nName:"),
            },
            "no License-File of the third-party licences": {
                **good_files,
                f"{info}/METADATA": good_meta.replace(
                    "License-File: THIRD_PARTY_LICENSES.txt\n", ""
                ),
            },
            "an extra not pinned": {
                **good_files,
                f"{info}/METADATA": good_meta.replace(f"dyng-cu13=={v}", "dyng-cu13"),
            },
        }
        for label, files in cases.items():
            bad = wheel(f"dyng-{v}-cp312-abi3-{plat}.whl", files)
            assert check_wheel(bad, platform=plat, require_libgomp=True, expect_version=v), label
        wrong_tag = wheel(f"dyng-{v}-cp312-abi3-linux_x86_64.whl", good_files)
        assert check_wheel(wrong_tag, platform=plat, require_libgomp=True, expect_version=v)
        wrong_abi = wheel(f"dyng-{v}-cp313-cp313-{plat}.whl", good_files)
        assert check_wheel(wrong_abi, platform=plat, require_libgomp=True, expect_version=v)

        # A CUDA plugin wheel (ADR 0030).
        pinfo = f"dyng_cu13-{v}.dist-info"
        plugin_licenses = (*LICENSE_FILES, *PLUGIN_EXTRA_LICENSE_FILES)
        plugin_meta = (
            f"Metadata-Version: 2.4\nName: dyng-cu13\nVersion: {v}\n"
            f"License-Expression: {PLUGIN_LICENSE_EXPRESSION}\n"
            + "".join(f"License-File: {lic}\n" for lic in plugin_licenses)
            + f"Requires-Dist: dyng=={v}\n\n# plugin\n"
        )
        good_elf = _fake_elf(["libgomp-1234abcd.so.1.0.0", "libstdc++.so.6", "libc.so.6"])
        assert elf_needed(good_elf) == ["libgomp-1234abcd.so.1.0.0", "libstdc++.so.6", "libc.so.6"]
        plugin_files: dict[str, str | bytes] = {
            "dyng_cu13/_core.abi3.so": good_elf,
            "dyng_cu13/__init__.py": "",
            "dyng_cu13.libs/libgomp-1234abcd.so.1.0.0": "x",
            **{f"{pinfo}/licenses/{lic}": "" for lic in plugin_licenses},
            f"{pinfo}/entry_points.txt": "[dyng.backends]\ncu13 = dyng_cu13\n",
            f"{pinfo}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
            f"{pinfo}/METADATA": plugin_meta,
        }
        pname = f"dyng_cu13-{v}-cp312-abi3-{plat}.whl"
        good_plugin = wheel(pname, plugin_files)
        assert not check_plugin_wheel(
            good_plugin, platform=plat, require_libgomp=True, expect_version=v
        )
        plugin_cases: dict[str, dict[str, str | bytes]] = {
            "no module": {k: x for k, x in plugin_files.items() if not k.endswith(".so")},
            "the dyng package": {**plugin_files, "dyng/__init__.py": ""},
            "no entry point": {**plugin_files, f"{pinfo}/entry_points.txt": ""},
            "a console script": {
                **plugin_files,
                f"{pinfo}/entry_points.txt": "[console_scripts]\ndyng = dyng.cli:main\n"
                "[dyng.backends]\ncu13 = dyng_cu13\n",
            },
            "no EULA": {k: x for k, x in plugin_files.items() if "EULA" not in k},
            "unpinned dyng": {
                **plugin_files,
                f"{pinfo}/METADATA": plugin_meta.replace(f"dyng=={v}", "dyng"),
            },
            "bundled cudart": {**plugin_files, "dyng_cu13.libs/libcudart-ab12.so.13": "x"},
            "needs libcuda": {
                **plugin_files,
                "dyng_cu13/_core.abi3.so": _fake_elf(["libcuda.so.1", "libc.so.6"]),
            },
            "needs libcudart": {
                **plugin_files,
                "dyng_cu13/_core.abi3.so": _fake_elf(["libcudart.so.13"]),
            },
            "no libgomp": {k: x for k, x in plugin_files.items() if "libgomp" not in k},
        }
        for label, pfiles in plugin_cases.items():
            bad = wheel(pname, pfiles)
            assert check_plugin_wheel(bad, platform=plat, require_libgomp=True, expect_version=v), (
                label
            )
        bad_name = wheel(f"dyng_cu99-{v}-cp312-abi3-{plat}.whl", plugin_files)
        assert check_plugin_wheel(bad_name, platform=plat, require_libgomp=True, expect_version=v)

        def sdist(files: list[str], pkg_info: str | None = good_meta) -> Path:
            p = d / f"dyng-{v}.tar.gz"
            contents = {n: "x" for n in files}
            if pkg_info is not None:
                contents["PKG-INFO"] = pkg_info
            with tarfile.open(p, "w:gz") as t:
                for n, text in contents.items():
                    data = text.encode()
                    ti = tarfile.TarInfo(f"dyng-{v}/{n}")
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            return p

        good_sdist = [
            *SDIST_REQUIRED,
            "cpp/include/dyng/dyng.hpp",
            "cpp/src/x.cpp",
            "python/dyng/__init__.py",
            "python/bindings/module.cpp",
        ]
        assert not check_sdist(sdist(good_sdist), expect_version=v)
        assert check_sdist(sdist([*good_sdist, "parity/compare.py"]), expect_version=v)
        assert check_sdist(sdist(good_sdist[1:]), expect_version=v)
        assert check_sdist(sdist([*good_sdist, "CODE_OF_CONDUCT.md"]), expect_version=v)
        assert check_sdist(sdist(good_sdist, pkg_info=None), expect_version=v)
        apache_only = good_meta.replace(LICENSE_EXPRESSION, "Apache-2.0")
        assert check_sdist(sdist(good_sdist, pkg_info=apache_only), expect_version=v)
        assert not check_pyproject(), "pyproject.toml's license differs from LICENSE_EXPRESSION"
        bad_pyproject = d / "pyproject.toml"
        bad_pyproject.write_text('[project]\nname = "dyng"\nlicense = "Apache-2.0"\n')
        assert check_pyproject(bad_pyproject)
        for good_v, pre in (("0.1.0", False), ("0.1.0rc1", True), ("0.1.0.dev0", True)):
            assert canonical_version(good_v) == (good_v, pre), good_v
        for bad_v in ("0.1.0-rc.1", "0.1.0RC1", "v0.1.0", "0.1.0-dev", "0.01.0", "0.1.0rc"):
            try:
                canonical_version(bad_v)
            except ValueError:
                continue
            raise AssertionError(f"{bad_v} accepted as canonical")
        # The release set: v0.1.x is the core alone, 0.2.0 (and its candidates) adds the plugins.
        assert [x["name"] for x in release_distributions("0.1.1")] == ["dyng"]
        assert [x["name"] for x in release_distributions("0.2.0rc1")] == [
            "dyng",
            "dyng-cu12",
            "dyng-cu13",
        ]
        assert [x["suffix"] for x in release_distributions("1.0.0")] == ["", "-cu12", "-cu13"]
        files = [
            d / "dyng-0.2.0.tar.gz",
            d / "dyng-0.2.0-cp312-abi3-manylinux_2_28_x86_64.whl",
            d / "dyng_cu12-0.2.0-cp312-abi3-manylinux_2_28_x86_64.whl",
            d / "dyng_cu13-0.2.0-cp312-abi3-manylinux_2_28_x86_64.whl",
        ]
        for f in files:
            f.write_bytes(b"x")
        assert not check_release_set(files, "0.2.0")
        assert check_release_set(files[:3], "0.2.0")  # a plugin missing
        assert check_release_set(files[1:], "0.2.0")  # the sdist missing
        assert check_release_set(files, "0.2.1")  # another version
        assert check_release_set(files, "0.1.1")  # plugins in a 0.1.x release
        split = split_release(files, d / "split", "0.2.0")
        assert sorted(split) == ["dyng", "dyng-cu12", "dyng-cu13"] and len(split["dyng"]) == 2
    print("wheel_check self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="*", type=Path, help="wheels (.whl) and sdists (.tar.gz)")
    p.add_argument("--platform", default=None, help="a platform tag every wheel must carry")
    p.add_argument(
        "--require-libgomp", action="store_true", help="the OpenMP runtime must be bundled"
    )
    p.add_argument("--self-test", action="store_true", help="check the checks, then exit")
    p.add_argument(
        "--version-info",
        action="store_true",
        help="print '<VERSION> <true|false>' (a pre-release?) after checking that VERSION is "
        "canonical PEP 440 (release.yml's select job)",
    )
    p.add_argument(
        "--release-metadata",
        action="store_true",
        help="check VERSION, CHANGELOG.md and CITATION.cff against each other (release.yml's "
        "select job, ci/tests; docs/developer/release.md before a tag)",
    )
    p.add_argument(
        "--release-distributions",
        action="store_true",
        help="print the distributions of a release of VERSION as one line of JSON (release.yml's "
        "publishing matrix: name, file prefix, environment suffix)",
    )
    p.add_argument(
        "--release-set",
        action="store_true",
        help="with files: they must be exactly the distributions of a release of VERSION",
    )
    p.add_argument(
        "--split",
        type=Path,
        metavar="DIR",
        help="with --release-set: copy the files to DIR/<distribution>/ (release.yml's collect)",
    )
    p.add_argument(
        "--release-date",
        metavar="YYYY-MM-DD|today",
        help="with --release-metadata: the release date must be this day (today: UTC)",
    )
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.release_metadata:
        day = args.release_date
        if day == "today":
            day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        errors = check_release_metadata(ROOT, day)
        for e in errors:
            print(f"wheel_check: {e}", file=sys.stderr)
        if not errors:
            print(f"release metadata: VERSION, CHANGELOG.md and CITATION.cff agree ({version()})")
        return 1 if errors else 0
    if args.release_distributions:
        print(json.dumps(release_distributions(version()), separators=(",", ":")))
        return 0
    if args.version_info:
        try:
            v, pre = canonical_version((ROOT / "VERSION").read_text().strip())
        except ValueError as e:
            print(f"wheel_check: {e}", file=sys.stderr)
            return 1
        print(v, "true" if pre else "false")
        return 0
    status = run(args.files, platform=args.platform, require_libgomp=args.require_libgomp)
    if args.release_set and status == 0:
        errors = check_release_set(args.files, version())
        for e in errors:
            print(f"wheel_check: release set: {e}", file=sys.stderr)
        if errors:
            return 1
        names = ", ".join(d["name"] for d in release_distributions(version()))
        print(f"release set: the distributions of {version()} ({names})")
        if args.split is not None:
            for dist, files in sorted(split_release(args.files, args.split, version()).items()):
                print(f"  {args.split / dist}: {', '.join(f.name for f in files)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
