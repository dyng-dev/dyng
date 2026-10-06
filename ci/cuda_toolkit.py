#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The CUDA toolkits of the plugin wheels' CI builds, pinned file by file (PLAN 7.7, 7.8; ADR 0032).

``wheels.yml`` builds ``dyng-cu12`` and ``dyng-cu13`` with cibuildwheel in the manylinux_2_28
image (AlmaLinux 8), with the CUDA toolkit installed in ``before-all`` from NVIDIA's RHEL 8
packages (PLAN 7.7's table). The packages are pinned in ``ci/cuda_toolkits.toml``: for each
plugin, the toolkit release (the latest 12.x and the latest 13.x, PLAN 7.8) and every RPM file
of the minimal set the build needs (``nvcc``, the CUDA runtime with its static library, CCCL,
NVTX 3, the documentation package that carries ``EULA.txt``, and what they require from NVIDIA's
repository), each with its SHA-256 and size. The runner downloads them on the host (cached
between runs) and checks every digest; the container installs exactly those files.

Commands::

    python3 ci/cuda_toolkit.py lock --plugin cu13 --release 13.4   # re-pin (needs network)
    python3 ci/cuda_toolkit.py download --plugin cu13 --dest .cuda-rpms   # fetch + verify
    python3 ci/cuda_toolkit.py verify --plugin cu13 --dest .cuda-rpms     # digests only
    python3 ci/cuda_toolkit.py root --plugin cu13        # /usr/local/cuda-13.4 (for scripts)
    python3 ci/cuda_toolkit.py show                      # the pinned toolkits
    python3 ci/cuda_toolkit.py extract --plugin cu13 --dest <dir>   # unpack without root

``lock`` resolves the dependency closure of the root packages inside NVIDIA's repository
metadata (``repodata/primary.xml``); requirements that the repository does not provide (the
host compiler ``gcc-c++``, ``/sbin/ldconfig``) are recorded as ``external`` and come from the
image's own repositories when ``dnf`` installs the files. ``extract`` (bsdtar needed) unpacks
the pinned RPMs into a directory, which gives the CI toolkit on a machine without root
(``<dir>/usr/local/cuda-X.Y``; ``ci/plugin_wheels.sh`` can build with it, docs/developer/wheels.md).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "ci" / "cuda_toolkits.toml"
#: NVIDIA's RHEL 8 repository: the packages of PLAN 7.7's table (manylinux_2_28 is AlmaLinux 8).
REPOSITORY = "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64"
#: The plugins and their CUDA majors (ci/wheel_check.py's PLUGINS).
PLUGINS = {"cu12": 12, "cu13": 13}
#: The root packages of a toolkit release ``{s}`` = ``13-4``: the compiler, the runtime (with
#: libcudart_static.a, which pulls CCCL), NVTX 3 (the profiler's ranges) and the documentation
#: package, which carries the toolkit's EULA.txt (the plugin wheels ship it; ADR 0030 item 9).
ROOT_PACKAGES = (
    "cuda-nvcc-{s}",
    "cuda-cudart-devel-{s}",
    "cuda-nvtx-{s}",
    "cuda-documentation-{s}",
)
SCHEMA = 1

_NS = {"c": "http://linux.duke.edu/metadata/common", "rpm": "http://linux.duke.edu/metadata/rpm"}


# -------------------------------------------------------------------------------------------------
# RPM versions and the repository metadata
# -------------------------------------------------------------------------------------------------


def _segments(v: str) -> list[tuple[int, int | str]]:
    """rpmvercmp's segments: runs of digits (compared as numbers) or letters."""
    return [(1, int(s)) if s.isdigit() else (0, s) for s in re.findall(r"\d+|[A-Za-z]+", v)]


def vercmp(a: str, b: str) -> int:
    """Compare two RPM versions (rpmvercmp without tilde / caret): -1, 0 or 1."""
    sa, sb = _segments(a), _segments(b)
    for x, y in zip(sa, sb, strict=False):
        if x != y:
            # a numeric segment is newer than an alphabetic one
            return 1 if x > y else -1
    return (len(sa) > len(sb)) - (len(sa) < len(sb))


@dataclass(frozen=True)
class Requirement:
    """One ``rpm:requires`` entry: a capability, optionally with a version constraint."""

    name: str
    flags: str | None = None  # EQ, GE, LE, GT, LT
    ver: str | None = None
    rel: str | None = None

    def admits(self, ver: str, rel: str) -> bool:
        if self.flags is None or self.ver is None:
            return True
        c = vercmp(ver, self.ver)
        if c == 0 and self.rel is not None:
            c = vercmp(rel, self.rel)
        return {
            "EQ": c == 0,
            "GE": c >= 0,
            "LE": c <= 0,
            "GT": c > 0,
            "LT": c < 0,
        }.get(self.flags, True)


@dataclass
class Package:
    """A package of the repository metadata."""

    name: str
    ver: str
    rel: str
    arch: str
    href: str
    sha256: str
    size: int
    requires: list[Requirement] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)

    @property
    def evr(self) -> str:
        return f"{self.ver}-{self.rel}"

    def newer_than(self, other: Package) -> bool:
        c = vercmp(self.ver, other.ver)
        return c > 0 or (c == 0 and vercmp(self.rel, other.rel) > 0)


def parse_primary(xml_bytes: bytes) -> list[Package]:
    """The packages of a repository's ``primary.xml``."""
    out: list[Package] = []
    for p in ET.fromstring(xml_bytes).findall("c:package", _NS):
        checksum = p.find("c:checksum", _NS)
        version = p.find("c:version", _NS)
        location = p.find("c:location", _NS)
        size = p.find("c:size", _NS)
        if checksum is None or version is None or location is None or size is None:
            continue
        if checksum.get("type") != "sha256":
            continue  # NVIDIA's repository publishes SHA-256 for every package
        reqs = [
            Requirement(e.get("name", ""), e.get("flags"), e.get("ver"), e.get("rel"))
            for e in p.findall("c:format/rpm:requires/rpm:entry", _NS)
        ]
        provides = [e.get("name", "") for e in p.findall("c:format/rpm:provides/rpm:entry", _NS)]
        files = [f.text or "" for f in p.findall("c:format/c:file", _NS)]
        out.append(
            Package(
                name=(p.findtext("c:name", "", _NS)),
                ver=version.get("ver", ""),
                rel=version.get("rel", ""),
                arch=p.findtext("c:arch", "", _NS),
                href=location.get("href", ""),
                sha256=(checksum.text or "").strip(),
                size=int(size.get("package", "0")),
                requires=reqs,
                provides=[*provides, *files],
            )
        )
    return out


def resolve(packages: list[Package], roots: Iterable[str]) -> tuple[list[Package], list[str]]:
    """The newest package of each root name and the closure of their requirements inside the
    repository; returns ``(packages sorted by name, external requirements)``."""
    by_name: dict[str, list[Package]] = {}
    providers: dict[str, set[str]] = {}
    for p in packages:
        if p.arch not in ("x86_64", "noarch"):
            continue
        by_name.setdefault(p.name, []).append(p)
        for cap in p.provides:
            providers.setdefault(cap, set()).add(p.name)

    def newest(name: str, req: Requirement | None = None) -> Package | None:
        best: Package | None = None
        for p in by_name.get(name, []):
            if req is not None and req.name == name and not req.admits(p.ver, p.rel):
                continue
            if best is None or p.newer_than(best):
                best = p
        return best

    chosen: dict[str, Package] = {}
    external: set[str] = set()
    queue: list[tuple[str, Requirement | None]] = []
    for r in roots:
        if r not in by_name:
            raise LookupError(f"{r}: no such package in the repository")
        queue.append((r, None))
    while queue:
        name, req = queue.pop()
        if name in chosen:
            continue
        p = newest(name, req)
        if p is None:
            raise LookupError(f"{name}: no version satisfies {req}")
        chosen[name] = p
        for q in p.requires:
            if q.name.startswith("rpmlib("):
                continue
            if q.name in by_name:
                queue.append((q.name, q))
                continue
            names = sorted(providers.get(q.name, ()))
            if not names:
                external.add(q.name)
                continue
            if any(n in chosen for n in names):
                continue
            # several packages provide it: the newest of them (by its own version)
            cands = [c for c in (newest(n) for n in names) if c is not None]
            best = cands[0]
            for c in cands[1:]:
                if c.newer_than(best):
                    best = c
            queue.append((best.name, None))
    return sorted(chosen.values(), key=lambda p: p.name), sorted(external)


def fetch(url: str, attempts: int = 4) -> bytes:
    """GET a URL (a few attempts with back-off)."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (https only)
                data: bytes = r.read()
                return data
        except OSError as e:
            if i + 1 == attempts:
                raise
            print(f"cuda_toolkit: {url}: {e}; retrying", file=sys.stderr)
            time.sleep(2.0 * (i + 1))
    raise AssertionError("unreachable")


def repository_packages(repository: str = REPOSITORY) -> list[Package]:
    """Read the repository's metadata (``repodata/repomd.xml`` -> ``primary.xml.gz``)."""
    repomd = ET.fromstring(fetch(f"{repository}/repodata/repomd.xml"))
    ns = {"r": "http://linux.duke.edu/metadata/repo"}
    for data in repomd.findall("r:data", ns):
        if data.get("type") == "primary":
            loc = data.find("r:location", ns)
            if loc is None:
                break
            raw = fetch(f"{repository}/{loc.get('href')}")
            return parse_primary(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    raise LookupError(f"{repository}: no primary metadata in repomd.xml")


# -------------------------------------------------------------------------------------------------
# The lock file
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Toolkit:
    """One plugin's pinned toolkit (a table of ``ci/cuda_toolkits.toml``)."""

    plugin: str
    release: str
    nvcc: str
    root: str
    repository: str
    packages: tuple[dict[str, object], ...]
    external: tuple[str, ...]


def load(path: Path = LOCK, *, complete: bool = True) -> dict[str, Toolkit]:
    """The pinned toolkits by plugin; ValueError if the file is malformed (or, with
    ``complete``, does not pin every plugin)."""
    data = tomllib.loads(path.read_text())
    errors: list[str] = []
    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    out: dict[str, Toolkit] = {}
    toolkits = data.get("toolkits", {})
    if set(toolkits) - set(PLUGINS) or (complete and set(toolkits) != set(PLUGINS)):
        errors.append(f"toolkits must be exactly {sorted(PLUGINS)}, not {sorted(toolkits)}")
    for plugin, t in sorted(toolkits.items()):
        major = PLUGINS.get(plugin)
        release = str(t.get("release", ""))
        m = re.fullmatch(r"(\d+)\.(\d+)", release)
        if m is None or int(m.group(1)) != major:
            errors.append(f"{plugin}: release {release!r} is not CUDA {major}.x")
            continue
        series = release.replace(".", "-")
        if t.get("root") != f"/usr/local/cuda-{release}":
            errors.append(f"{plugin}: root must be /usr/local/cuda-{release}")
        if not str(t.get("nvcc", "")).startswith(release + "."):
            errors.append(f"{plugin}: nvcc {t.get('nvcc')!r} is not of CUDA {release}")
        pkgs = t.get("packages", [])
        names = {p.get("name") for p in pkgs}
        for r in ROOT_PACKAGES:
            if r.format(s=series) not in names:
                errors.append(f"{plugin}: the root package {r.format(s=series)} is missing")
        for p in pkgs:
            href = str(p.get("href", ""))
            if not re.fullmatch(r"[A-Za-z0-9._+-]+\.rpm", href):
                errors.append(f"{plugin}: {p.get('name')}: bad href {href!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(p.get("sha256", ""))):
                errors.append(f"{plugin}: {p.get('name')}: sha256 is not 64 hex digits")
            if not isinstance(p.get("size"), int) or int(p["size"]) <= 0:
                errors.append(f"{plugin}: {p.get('name')}: size must be a positive integer")
            # every series-versioned package belongs to this release or to the major's common one
            n = str(p.get("name", ""))
            s = re.search(r"-(\d+)-(\d+)$", n)
            if s and f"{s.group(1)}-{s.group(2)}" != series:
                errors.append(f"{plugin}: {n} is not of CUDA {release}")
        if not str(t.get("repository", "")).startswith("https://developer.download.nvidia.com/"):
            errors.append(f"{plugin}: repository must be NVIDIA's (https)")
        out[plugin] = Toolkit(
            plugin,
            release,
            str(t.get("nvcc", "")),
            str(t.get("root", "")),
            str(t.get("repository", "")),
            tuple(pkgs),
            tuple(t.get("external", [])),
        )
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return out


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(toolkits: dict[str, Toolkit]) -> str:
    """The text of ``ci/cuda_toolkits.toml``."""
    lines = [
        # REUSE-IgnoreStart (the header of the generated file)
        "# SPDX-FileCopyrightText: 2026 The dynG Authors",
        "# SPDX-License-Identifier: Apache-2.0",
        # REUSE-IgnoreEnd
        "#",
        "# GENERATED by `python3 ci/cuda_toolkit.py lock --plugin <p> --release <X.Y>`:",
        "# the CUDA toolkits of the plugin wheels' CI builds (wheels.yml), every RPM",
        "# file pinned by its SHA-256 (PLAN 7.7, 7.8; ADR 0032). cu12 is built with the",
        "# latest 12.x, cu13 with the latest 13.x. `external` lists what the files need",
        "# from the image's own repositories. Re-pin with the command above; do not edit.",
        "",
        f"schema = {SCHEMA}",
    ]
    for plugin in sorted(toolkits):
        t = toolkits[plugin]
        lines += [
            "",
            f"[toolkits.{plugin}]",
            f"release = {_toml_str(t.release)}",
            f"nvcc = {_toml_str(t.nvcc)}",
            f"root = {_toml_str(t.root)}",
            f"repository = {_toml_str(t.repository)}",
            "external = [" + ", ".join(_toml_str(e) for e in t.external) + "]",
        ]
        for p in t.packages:
            lines += [
                "",
                f"[[toolkits.{plugin}.packages]]",
                f"name = {_toml_str(str(p['name']))}",
                f"version = {_toml_str(str(p['version']))}",
                f"href = {_toml_str(str(p['href']))}",
                f"sha256 = {_toml_str(str(p['sha256']))}",
                f"size = {int(str(p['size']))}",
            ]
    return "\n".join(lines) + "\n"


def lock_toolkit(plugin: str, release: str, packages: list[Package], repository: str) -> Toolkit:
    """Pin the toolkit ``release`` (``13.4``) of ``plugin`` from the repository's packages."""
    major = PLUGINS[plugin]
    if not re.fullmatch(rf"{major}\.\d+", release):
        raise ValueError(f"{plugin} needs a CUDA {major}.x release, not {release}")
    series = release.replace(".", "-")
    chosen, external = resolve(packages, [r.format(s=series) for r in ROOT_PACKAGES])
    nvcc = next(p for p in chosen if p.name == f"cuda-nvcc-{series}")
    return Toolkit(
        plugin,
        release,
        nvcc.ver,
        f"/usr/local/cuda-{release}",
        repository,
        tuple(
            {"name": p.name, "version": p.evr, "href": p.href, "sha256": p.sha256, "size": p.size}
            for p in chosen
        ),
        tuple(external),
    )


# -------------------------------------------------------------------------------------------------
# Download, verify, extract
# -------------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(t: Toolkit, dest: Path) -> list[str]:
    """Problems of the files in ``dest`` against the lock (empty: every file is right)."""
    problems = []
    for p in t.packages:
        f = dest / str(p["href"])
        if not f.is_file():
            problems.append(f"{f.name}: missing")
        elif f.stat().st_size != p["size"] or _sha256(f) != p["sha256"]:
            problems.append(f"{f.name}: size or SHA-256 differs from ci/cuda_toolkits.toml")
    return problems


def download(t: Toolkit, dest: Path) -> None:
    """Fetch the pinned files into ``dest`` (keeping the right ones already there), remove any
    other ``.rpm``, check every digest and write ``SHA256SUMS`` (``sha256sum -c`` format)."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {str(p["href"]) for p in t.packages}
    for stale in dest.glob("*.rpm"):
        if stale.name not in wanted:
            stale.unlink()
    for p in t.packages:
        f = dest / str(p["href"])
        if f.is_file() and f.stat().st_size == p["size"] and _sha256(f) == p["sha256"]:
            continue
        data = fetch(f"{t.repository}/{p['href']}")
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != p["size"] or digest != p["sha256"]:
            raise ValueError(
                f"{p['href']}: downloaded {len(data)} bytes with SHA-256 {digest}; "
                f"ci/cuda_toolkits.toml pins {p['size']} bytes, {p['sha256']}"
            )
        tmp = f.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, f)
    problems = verify(t, dest)
    if problems:
        raise ValueError("; ".join(problems))
    (dest / "SHA256SUMS").write_text("".join(f"{p['sha256']}  {p['href']}\n" for p in t.packages))


def extract(t: Toolkit, rpms: Path, dest: Path) -> Path:
    """Unpack the pinned RPMs (from ``rpms``, verified first) under ``dest``; returns the
    toolkit's root there (``<dest>/usr/local/cuda-X.Y``). Needs ``bsdtar`` (libarchive)."""
    problems = verify(t, rpms)
    if problems:
        raise ValueError("; ".join(problems))
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise FileNotFoundError("extract needs bsdtar (libarchive-tools, or conda's libarchive)")
    dest.mkdir(parents=True, exist_ok=True)
    for p in t.packages:
        subprocess.run([bsdtar, "-xf", str(rpms / str(p["href"])), "-C", str(dest)], check=True)
    root = dest / t.root.lstrip("/")
    if not (root / "bin" / "nvcc").is_file():
        raise FileNotFoundError(f"{root}/bin/nvcc is missing after the extraction")
    return root


# -------------------------------------------------------------------------------------------------
# Command line
# -------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("lock", "download", "verify", "root", "extract"):
        s = sub.add_parser(name)
        s.add_argument("--plugin", required=True, choices=sorted(PLUGINS))
        s.add_argument("--lock-file", type=Path, default=LOCK)
        if name == "lock":
            s.add_argument("--release", required=True, help="the toolkit release, e.g. 13.4")
            s.add_argument("--repository", default=REPOSITORY)
        if name in ("download", "verify", "extract"):
            s.add_argument("--dest", type=Path, required=True)
        if name == "extract":
            s.add_argument("--rpms", type=Path, required=True, help="the downloaded RPMs")
    s = sub.add_parser("show")
    s.add_argument("--lock-file", type=Path, default=LOCK)
    args = p.parse_args(argv)

    if args.command == "lock":
        current = load(args.lock_file, complete=False) if args.lock_file.is_file() else {}
        pkgs = repository_packages(args.repository)
        current[args.plugin] = lock_toolkit(args.plugin, args.release, pkgs, args.repository)
        if set(current) != set(PLUGINS):
            missing = sorted(set(PLUGINS) - set(current))
            print(f"cuda_toolkit: lock {', '.join(missing)} too before committing", file=sys.stderr)
        args.lock_file.write_text(render(current))
        t = current[args.plugin]
        print(
            f"{args.plugin}: CUDA {t.release} (nvcc {t.nvcc}), {len(t.packages)} packages, "
            f"{sum(int(str(x['size'])) for x in t.packages) / 1e6:.0f} MB"
        )
        return 0
    toolkits = load(args.lock_file)
    if args.command == "show":
        for name, t in sorted(toolkits.items()):
            size = sum(int(str(x["size"])) for x in t.packages) / 1e6
            print(
                f"{name}: CUDA {t.release} (nvcc {t.nvcc}) in {t.root}, "
                f"{len(t.packages)} RPMs, {size:.0f} MB"
            )
        return 0
    t = toolkits[args.plugin]
    if args.command == "root":
        print(t.root)
    elif args.command == "download":
        download(t, args.dest)
        print(f"cuda_toolkit: {len(t.packages)} RPMs of CUDA {t.release} in {args.dest}, verified")
    elif args.command == "verify":
        problems = verify(t, args.dest)
        for pr in problems:
            print(f"cuda_toolkit: {pr}", file=sys.stderr)
        return 1 if problems else 0
    elif args.command == "extract":
        print(extract(t, args.rpms, args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
