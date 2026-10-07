# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/cuda_toolkit.py pins the CI toolkits of the plugin wheels file by file (ADR 0032)."""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cuda_toolkit as ct  # noqa: E402
import wheel_check  # noqa: E402

COMMON = "http://linux.duke.edu/metadata/common"
RPM = "http://linux.duke.edu/metadata/rpm"


def _pkg(name: str, ver: str, requires: list[str] | None = None, provides: list[str] = ()) -> str:
    reqs = "".join(
        f'<rpm:entry name="{r.split()[0]}"'
        + (f' flags="{r.split()[1]}" ver="{r.split()[2]}"' if len(r.split()) == 3 else "")
        + "/>"
        for r in requires or []
    )
    prov = "".join(f'<rpm:entry name="{p}"/>' for p in [name, *provides])
    digest = hashlib.sha256(f"{name}-{ver}".encode()).hexdigest()
    return (
        f"<package type='rpm'><name>{name}</name><arch>x86_64</arch>"
        f'<version epoch="0" ver="{ver}" rel="1"/>'
        f'<checksum type="sha256" pkgid="YES">{digest}</checksum>'
        f'<size package="{len(name) * 1000}" installed="1" archive="1"/>'
        f'<location href="{name}-{ver}-1.x86_64.rpm"/>'
        f"<format><rpm:provides>{prov}</rpm:provides><rpm:requires>{reqs}</rpm:requires>"
        "</format></package>"
    )


PRIMARY = (
    f'<metadata xmlns="{COMMON}" xmlns:rpm="{RPM}" packages="9">'
    + _pkg("cuda-nvcc-13-4", "13.4.49", ["cuda-crt-13-4", "gcc-c++", "libnvvm-13-4"])
    + _pkg("cuda-nvcc-13-4", "13.4.92", ["cuda-crt-13-4", "gcc-c++", "libnvvm-13-4"])
    + _pkg("cuda-crt-13-4", "13.4.92")
    + _pkg("libnvvm-13-4", "13.4.92")
    + _pkg("cuda-cudart-devel-13-4", "13.4.92", ["cuda-cudart-13-4 EQ 13.4.92", "cccl-13-4"])
    + _pkg("cuda-cudart-13-4", "13.4.92", ["/sbin/ldconfig", "cuda-toolkit-config-common"])
    + _pkg("cuda-cudart-13-4", "13.4.99", ["/sbin/ldconfig"])  # excluded by the EQ constraint
    + _pkg("cccl-13-4", "13.3.4.3.1")
    + _pkg("cuda-toolkit-config-common", "13.4.92")
    + _pkg("cuda-toolkit-config-common", "13.10.1")
    + _pkg("cuda-nvtx-13-4", "13.4.92")
    + _pkg("cuda-documentation-13-4", "13.4.92")
    + _pkg("cuda-cuobjdump-13-4", "13.4.92")
    + "</metadata>"
).encode()


def test_vercmp() -> None:
    assert ct.vercmp("13.4.92", "13.4.49") == 1
    assert ct.vercmp("13.10.1", "13.4.92") == 1  # numeric, not lexicographic
    assert ct.vercmp("1.0", "1.0") == 0
    assert ct.vercmp("1.0", "1.0.1") == -1
    assert ct.vercmp("1.0a", "1.0.1") == -1  # a number is newer than letters
    req = ct.Requirement("x", "EQ", "13.4.92")
    assert req.admits("13.4.92", "1") and not req.admits("13.4.99", "1")
    assert ct.Requirement("x", "GE", "2.0").admits("2.1", "1")


def test_resolve_takes_the_newest_closure() -> None:
    pkgs = ct.parse_primary(PRIMARY)
    roots = [r.format(s="13-4") for r in ct.ROOT_PACKAGES]
    chosen, external = ct.resolve(pkgs, roots)
    versions = {p.name: p.ver for p in chosen}
    assert versions == {
        "cccl-13-4": "13.3.4.3.1",
        "cuda-crt-13-4": "13.4.92",
        "cuda-cuobjdump-13-4": "13.4.92",
        "cuda-cudart-13-4": "13.4.92",  # EQ beats newer
        "cuda-cudart-devel-13-4": "13.4.92",
        "cuda-documentation-13-4": "13.4.92",
        "cuda-nvcc-13-4": "13.4.92",
        "cuda-nvtx-13-4": "13.4.92",
        "cuda-toolkit-config-common": "13.10.1",
        "libnvvm-13-4": "13.4.92",
    }
    assert external == ["/sbin/ldconfig", "gcc-c++"]
    with pytest.raises(LookupError):
        ct.resolve(pkgs, ["cuda-nvcc-13-9"])


def test_lock_render_load_round_trip(tmp_path: Path) -> None:
    pkgs = ct.parse_primary(PRIMARY)
    t13 = ct.lock_toolkit("cu13", "13.4", pkgs, ct.REPOSITORY)
    assert t13.nvcc == "13.4.92" and t13.root == "/usr/local/cuda-13.4"
    with pytest.raises(ValueError):
        ct.lock_toolkit("cu12", "13.4", pkgs, ct.REPOSITORY)
    f = tmp_path / "lock.toml"
    f.write_text(ct.render({"cu13": t13}))
    assert ct.load(f, complete=False)["cu13"] == t13
    with pytest.raises(ValueError, match="exactly"):
        ct.load(f)  # cu12 is missing
    text = f.read_text()
    f.write_text(text.replace('release = "13.4"', 'release = "12.9"'))
    with pytest.raises(ValueError, match="is not CUDA 13.x"):
        ct.load(f, complete=False)
    f.write_text(text.replace('root = "/usr/local/cuda-13.4"', 'root = "/opt/cuda"'))
    with pytest.raises(ValueError, match="root must be"):
        ct.load(f, complete=False)
    f.write_text(text.replace('href = "cuda-nvtx', 'href = "../cuda-nvtx'))
    with pytest.raises(ValueError, match="bad href"):
        ct.load(f, complete=False)


def test_the_committed_lock() -> None:
    # Both plugins pinned: the latest 12.x and the latest 13.x (PLAN 7.8), with nvcc's release
    # equal to the toolkit's, every root package, and NVIDIA's repository.
    toolkits = ct.load()
    assert sorted(toolkits) == list(wheel_check.PLUGINS)
    for plugin, t in toolkits.items():
        assert t.release.split(".")[0] == plugin[2:]
        names = {p["name"] for p in t.packages}
        assert {r.format(s=t.release.replace(".", "-")) for r in ct.ROOT_PACKAGES} <= names
        assert t.repository == ct.REPOSITORY
        assert "gcc-c++" in t.external
    data = tomllib.loads(ct.LOCK.read_text())
    assert ct.render(ct.load()) == ct.LOCK.read_text(), "edit the lock only with `lock`"
    assert data["schema"] == ct.SCHEMA


def test_verify_and_download_from_a_local_mirror(tmp_path: Path, monkeypatch) -> None:
    pkgs = ct.parse_primary(PRIMARY)
    t = ct.lock_toolkit("cu13", "13.4", pkgs, "https://example.invalid/repo")
    blobs = {}
    fixed = []
    for p in t.packages:
        data = f"rpm {p['name']}".encode()
        blobs[f"https://example.invalid/repo/{p['href']}"] = data
        fixed.append({**p, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    t = ct.Toolkit(t.plugin, t.release, t.nvcc, t.root, t.repository, tuple(fixed), t.external)
    calls: list[str] = []

    def fake_fetch(url: str, attempts: int = 4) -> bytes:
        calls.append(url)
        return blobs[url]

    monkeypatch.setattr(ct, "fetch", fake_fetch)
    dest = tmp_path / "rpms"
    (dest).mkdir()
    (dest / "stale-1.0-1.x86_64.rpm").write_bytes(b"old")
    assert len(ct.verify(t, dest)) == len(fixed)
    ct.download(t, dest, signatures=False)  # the blobs are not RPMs (test_signatures below)
    assert ct.verify(t, dest) == []
    assert not (dest / "stale-1.0-1.x86_64.rpm").exists()
    sums = (dest / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == len(fixed) and all("  " in line for line in sums)
    n = len(calls)
    ct.download(t, dest, signatures=False)  # a cache hit downloads nothing
    assert len(calls) == n
    first = dest / str(fixed[0]["href"])
    first.write_bytes(b"tampered")
    assert ct.verify(t, dest) == [
        f"{first.name}: size or SHA-256 differs from ci/cuda_toolkits.toml"
    ]
    blobs[f"https://example.invalid/repo/{fixed[0]['href']}"] = b"evil"
    with pytest.raises(ValueError, match="pins"):
        ct.download(t, dest, signatures=False)


# -------------------------------------------------------------------------------------------------
# The packages' OpenPGP signatures (a key made for the test signs a synthetic RPM)
# -------------------------------------------------------------------------------------------------


def _header(entries: list[tuple[int, int, bytes]]) -> bytes:
    """An RPM header: (tag, type, data) entries, each datum 8-byte aligned in the store."""
    index, store = b"", b""
    for tag, kind, data in entries:
        store += bytes(-len(store) % 8)
        index += struct.pack(">IIII", tag, kind, len(store), 1 if kind != 7 else len(data))
        store += data
    return (
        b"\x8e\xad\xe8\x01"
        + bytes(4)
        + struct.pack(">II", len(entries), len(store))
        + (index + store)
    )


def _rpm(sign: object, payload: bytes, digest: str | None = None) -> bytes:
    main = _header(
        [
            (5092, 8, (digest or hashlib.sha256(payload).hexdigest()).encode() + b"\0"),
            (5093, 4, struct.pack(">I", 8)),
        ]
    )
    sig = _header([(268, 7, sign(main))])  # type: ignore[operator]
    lead = b"\xed\xab\xee\xdb" + bytes(92)
    return lead + sig + bytes(-len(sig) % 8) + main + payload


@pytest.fixture(scope="module")
def signer(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path, str]:
    gpg = shutil.which("gpg")
    if gpg is None:
        pytest.skip("needs gpg")
    home = tmp_path_factory.mktemp("gnupg")
    run = [
        gpg,
        "--homedir",
        str(home),
        "--batch",
        "--pinentry-mode",
        "loopback",
        "--passphrase",
        "",
    ]
    subprocess.run(
        [*run, "--quick-gen-key", "dynG test <test@example.invalid>", "rsa2048", "sign", "never"],
        check=True,
        capture_output=True,
    )
    listing = subprocess.run(
        [*run, "--with-colons", "--list-keys"], check=True, capture_output=True, text=True
    ).stdout
    records = [line.split(":") for line in listing.splitlines()]
    fingerprint = next(r[9] for r in records if r[0] == "fpr")  # codespell:ignore fpr
    key = home / "key.pub"
    key.write_bytes(subprocess.run([*run, "--export"], check=True, capture_output=True).stdout)

    def sign(data: bytes) -> bytes:
        out = subprocess.run([*run, "--detach-sign"], input=data, check=True, capture_output=True)
        return out.stdout

    return sign, key, fingerprint


def _one(dest: Path, data: bytes) -> ct.Toolkit:
    href = "cuda-test-13-4-13.4.92-1.x86_64.rpm"
    (dest / href).write_bytes(data)
    package = {"name": "cuda-test-13-4", "version": "13.4.92-1", "href": href}
    return ct.Toolkit("cu13", "13.4", "13.4.92", "/usr/local/cuda-13.4", "x", (package,), ())


def test_signatures(tmp_path: Path, signer: tuple[object, Path, str]) -> None:
    sign, key, fingerprint = signer
    good = _rpm(sign, b"the payload")
    signature, header, digest, payload = ct.rpm_signed_parts(good)
    assert digest == payload == hashlib.sha256(b"the payload").hexdigest()
    assert good.index(header) > 96 and signature
    t = _one(tmp_path, good)
    assert ct.verify_signatures(t, tmp_path, key, fingerprint) == []
    # Another key's fingerprint, a changed header, a changed payload, an unsigned file.
    assert "fingerprint" in ct.verify_signatures(t, tmp_path, key, "0" * 40)[0]
    start = good.index(header)
    bad = bytearray(good)
    bad[start + 20] ^= 1
    _one(tmp_path, bytes(bad))
    assert "no valid signature" in ct.verify_signatures(t, tmp_path, key, fingerprint)[0]
    _one(tmp_path, good[:-1] + b"X")
    assert "payload" in ct.verify_signatures(t, tmp_path, key, fingerprint)[0]
    _one(tmp_path, _rpm(lambda data: b"not a signature", b"p"))
    assert "no valid signature" in ct.verify_signatures(t, tmp_path, key, fingerprint)[0]
    with pytest.raises(ValueError, match="not an RPM"):
        ct.rpm_signed_parts(b"plain bytes")
    assert ct.verify_signatures(t, tmp_path / "missing", None, fingerprint)[0].endswith(
        "the signing key is missing"
    )


def test_the_pinned_signing_key() -> None:
    assert ct.SIGNING_KEY == ct.SIGNING_KEY_FINGERPRINT[-8:] + ".pub"
    assert len(ct.SIGNING_KEY_FINGERPRINT) == 40
