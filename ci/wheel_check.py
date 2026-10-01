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
of Conduct, governance and tool configuration), and its size. ``VERSION`` must be a canonical
PEP 440 version (the distributions carry the normalised form, so any other spelling would give
file names that no check expects).

Usage::

    python3 ci/wheel_check.py dist/*.whl dist/*.tar.gz --platform manylinux_2_28_x86_64 \\
        --require-libgomp
    python3 ci/wheel_check.py --version-info   # "<VERSION> <pre-release: true|false>"
    python3 ci/wheel_check.py --self-test

Exit status 0 when every file passes; the report lists each check.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import tempfile
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
    libgomp = [n for n in names if re.match(r"dyng\.libs/libgomp[-.]", n)]
    if require_libgomp and not libgomp:
        errors.append("the OpenMP runtime is not bundled (dyng.libs/libgomp-*.so*)")
    stray = [n for n in names if n.endswith((".a", ".o", ".cpp", ".hpp", ".cmake"))]
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
    present = set(names)
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
    for p in paths:
        if p.name.endswith(".whl"):
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


def _self_test() -> int:
    """Broken distributions are rejected, a good one passes."""
    v = version()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        def wheel(name: str, files: dict[str, str]) -> Path:
            p = d / name
            with zipfile.ZipFile(p, "w") as z:
                for n, text in files.items():
                    z.writestr(n, text)
            return p

        info = f"dyng-{v}.dist-info"
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
        }
        for label, files in cases.items():
            bad = wheel(f"dyng-{v}-cp312-abi3-{plat}.whl", files)
            assert check_wheel(bad, platform=plat, require_libgomp=True, expect_version=v), label
        wrong_tag = wheel(f"dyng-{v}-cp312-abi3-linux_x86_64.whl", good_files)
        assert check_wheel(wrong_tag, platform=plat, require_libgomp=True, expect_version=v)
        wrong_abi = wheel(f"dyng-{v}-cp313-cp313-{plat}.whl", good_files)
        assert check_wheel(wrong_abi, platform=plat, require_libgomp=True, expect_version=v)

        def sdist(files: list[str]) -> Path:
            p = d / f"dyng-{v}.tar.gz"
            with tarfile.open(p, "w:gz") as t:
                for n in files:
                    data = b"x"
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
        for good_v, pre in (("0.1.0", False), ("0.1.0rc1", True), ("0.1.0.dev0", True)):
            assert canonical_version(good_v) == (good_v, pre), good_v
        for bad_v in ("0.1.0-rc.1", "0.1.0RC1", "v0.1.0", "0.1.0-dev", "0.01.0", "0.1.0rc"):
            try:
                canonical_version(bad_v)
            except ValueError:
                continue
            raise AssertionError(f"{bad_v} accepted as canonical")
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
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.version_info:
        try:
            v, pre = canonical_version((ROOT / "VERSION").read_text().strip())
        except ValueError as e:
            print(f"wheel_check: {e}", file=sys.stderr)
            return 1
        print(v, "true" if pre else "false")
        return 0
    return run(args.files, platform=args.platform, require_libgomp=args.require_libgomp)


if __name__ == "__main__":
    sys.exit(main())
