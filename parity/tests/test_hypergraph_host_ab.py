# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/experiments/hypergraph_host_ab.py (the ADR 0024 reading of R030's D13 fix)
that need no build or dataset: each side's record names its commit, digest and build settings,
a side whose sources are not its commit's is refused, and the lock is the one perf_lock found
(R030 review: the tool wrote "exclusive perf lock" without checking it)."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity" / "experiments"))
sys.path.insert(0, str(REPO / "parity"))

import hypergraph_host_ab as hab  # noqa: E402


def archive(commit: str, dest: Path) -> None:
    """`git archive` of the side's source paths into `dest` (as the reading's sides are made)."""
    tar = subprocess.run(
        ["git", "-C", str(REPO), "archive", commit, "--", *hab.SOURCE_PATHS],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")


def fake_build(tmp_path: Path, commit: str = "HEAD") -> Path:
    """A source copy of `commit` and a build tree with a CMakeCache.txt and an 'executable'."""
    src = tmp_path / "src"
    archive(commit, src)
    build = tmp_path / "build"
    (build / "tools" / "compat").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CXX_FLAGS:STRING=-O3 -march=x\n"
        f"CMAKE_HOME_DIRECTORY:INTERNAL={src}\nUNRELATED:STRING=1\n"
    )
    exe = build / "tools" / "compat" / "dyng-compat-escher"
    exe.write_bytes(b"\x7fELF not really")
    return exe


def test_a_side_records_its_commit_digest_and_build(tmp_path: Path) -> None:
    exe = fake_build(tmp_path)
    info = hab.side_record(exe, "HEAD")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert info["source"]["commit"] == head and info["source"]["matches"] is True
    assert info["source"]["files"] > 100
    assert info["build"]["CMAKE_BUILD_TYPE"] == "Release"
    assert info["build"]["CMAKE_CXX_FLAGS"] == "-O3 -march=x"
    assert "UNRELATED" not in info["build"]
    assert len(info["sha256"]) == 64 and info["build_tree"] == str(tmp_path / "build")


def test_a_side_whose_sources_differ_from_its_commit_is_refused(tmp_path: Path) -> None:
    exe = fake_build(tmp_path)
    changed = tmp_path / "src" / "cpp" / "src" / "util" / "region_exceptions.hpp"
    changed.write_text(changed.read_text() + "// changed\n")
    (tmp_path / "src" / "cpp" / "src" / "extra.cpp").write_text("int x;\n")
    info = hab.side_record(exe, "HEAD")
    assert info["source"]["matches"] is False
    assert info["source"]["differ"] == ["cpp/src/extra.cpp", "cpp/src/util/region_exceptions.hpp"]
    assert hab.side_record(exe, "no-such-commit")["source"]["matches"] is False
    args = hab.argparse.Namespace(
        baseline_exe=str(exe),
        exe=str(exe),
        baseline_commit="HEAD",
        commit="HEAD",
        lock_timeout=1.0,
        no_lock=True,
    )
    with pytest.raises(SystemExit, match="side A: its sources are not those of"):
        hab.run(args)


FAKE_DRIVER = """#!/usr/bin/env python3
import json, sys
argv = sys.argv
timing = argv[argv.index("--timing") + 1]
stages = [{"stage": "hypergraph.build", "batch": 0, "ms": 5.0}]
for b in (1, 2):
    stages += [{"stage": s, "batch": b, "ms": 1.0 + b}
               for s in ("apply_ms", "hypergraph.apply", "hypergraph.apply.insert")]
open(timing, "w").write(json.dumps({"stages": stages}))
print("batch 1 removed 0 added 1")
print("apply 2.0 ms")
"""


class QuietMonitor:
    """perf_ab.MachineMonitor without the machine: every window clean."""

    def __init__(self, *_args: object) -> None:
        pass

    def __enter__(self) -> QuietMonitor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def run(self, cmd: list, env: dict) -> tuple[str, dict]:
        out = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
        return out.stdout, {"contaminated": False}


def test_the_lock_field_is_what_perf_lock_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole reading with a stand-in driver: the record's lock is perf_ab.LOCK_STATUS of the
    lock the tool took itself, and the sides carry their commits."""
    exe = fake_build(tmp_path)
    exe.write_text(FAKE_DRIVER)
    exe.chmod(0o755)
    case = tmp_path / "goldens" / "c1"
    case.mkdir(parents=True)
    (case / "group.json").write_text('{"argv": ["escher", "--input", "x.hg"]}')
    monkeypatch.setattr(hab, "GOLDENS", tmp_path / "goldens")
    monkeypatch.setattr(hab.perf_ab, "SCRATCH", tmp_path)
    monkeypatch.setattr(hab.perf_ab, "MachineMonitor", QuietMonitor)
    monkeypatch.delenv("DYNG_PERF_LOCK_HELD", raising=False)
    record_path = tmp_path / "r.json"
    args = hab.argparse.Namespace(
        baseline_exe=str(exe),
        exe=str(exe),
        baseline_commit="HEAD",
        commit="HEAD",
        cases="c1",
        backends="sequential",
        modes="apply",
        layouts=2,
        rounds=3,
        json=str(record_path),
        lock_timeout=5.0,
        no_lock=False,
    )
    assert hab.run(args) == 0
    doc = hab.json.loads(record_path.read_text())
    assert doc["protocol"]["lock"] == "perf.lock: held exclusively by this process"
    assert (tmp_path / "perf.lock").exists()
    assert doc["sides"]["A"]["source"]["matches"] and doc["sides"]["B"]["source"]["matches"]
    regions = {r["region"]: r for r in doc["results"]["c1/sequential/apply"]["regions"]}
    assert regions["hypergraph.build"]["port_samples"] == [5.0, 5.0, 5.0]
    assert regions["apply_ms"]["ratio"] == 1.0
