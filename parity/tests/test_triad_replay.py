# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/triad_replay.py (M9a step 2, ADR 0050): the case selection, the .dgb record
rewriter, the comparator on synthetic reports, and the planted defects of the harness (one
changed bin, one perturbed deleted row, a dropped batch, swapped batches, a wrong inserted id),
each of which must FAIL a replay of a make-test case (those need the golden set in
$DYNG_SCRATCH and a built dyng_triad_replay; they skip otherwise)."""

from __future__ import annotations

import copy
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "parity" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load("triad_goldens")
tr = _load("triad_replay")
SET_DIR = tg.set_dir(tg.scratch_default())
needs_goldens = pytest.mark.skipif(
    not (SET_DIR / "MANIFEST.sha256").is_file() or shutil.which("zstd") is None,
    reason="the golden set triad_count_escher_gpu is not in $DYNG_SCRATCH",
)


def _tool() -> Path | None:
    try:
        return tr.find_tool("sequential")
    except tr.ReplayError:
        return None


needs_tool = pytest.mark.skipif(_tool() is None, reason="dyng_triad_replay is not built")


def test_selection_follows_the_coverage_table():
    seq = tr.select("sequential", None)
    assert sum(len(v) for v in seq.values()) == 120
    assert "syn1m_b50k_s1" not in seq and "test/14-skewed" in seq
    omp = tr.select("openmp", None)
    assert omp["coauth_b50k_s1"] == ["type1", "type2"]
    assert sum(len(v) for v in omp.values()) == 142
    cuda = tr.select("cuda", ["syn1m_s1"])
    assert sorted(cuda) == [f"syn1m_b{s}_s1" for s in ("100k", "200k", "20", "50k", "5k")]
    assert tr.select("cuda", ["coauth_b50k_s1/type1"]) == {"coauth_b50k_s1": ["type1"]}
    assert sum(len(v) for v in tr.select("cuda", None).values()) == 300
    # A case outside the backend's coverage is not selected.
    assert tr.select("sequential", ["coauth_b50k_s1"]) == {}


def test_dgb_records_round_trip_and_rewrite(tmp_path: Path):
    fields = {
        "batch_id": 4,
        "deleted_ids": [3, 9],
        "insert_offsets": [0, 2, 5],
        "members": [1, 2, 0, 4, 7],
        "deleted_offsets": [0, 1, 4],
        "deleted_members": [5, 1, 2, 3],
        "expected_ids": [3, 11],
    }
    rec = tr.dgb_record(fields)
    assert tr.dgb_fields(rec) == fields
    header = bytearray(64)
    header[:8] = tg.DGB_MAGIC
    path = tmp_path / "b.dgb"
    path.write_bytes(tg.dgb_single(bytes(header), rec))

    def bump(f):
        f["expected_ids"][1] += 1

    tr.rewrite_dgb(path, bump)
    _, records = tg.dgb_records(path.read_bytes())
    assert tr.dgb_fields(records[0])["expected_ids"] == [3, 12]
    # The CRC32 of the payload is the record's.
    length, crc = struct.unpack_from("<QI", records[0], 0)
    import zlib

    assert crc == zlib.crc32(records[0][16:]) & 0xFFFFFFFF and length == len(records[0]) - 16


def _report(full, deltas, mode_names=("by_key", "by_content")):
    after, cur = [], list(full)
    for d in deltas:
        cur = [c + x for c, x in zip(cur, d, strict=True)]
        after.append(list(cur))
    modes = []
    for m in mode_names:
        modes.append(
            {
                "mode": m,
                "full": {"stathyper_type1": list(full)},
                "batches": [
                    {
                        "index": k + 1,
                        "deletions": 2,
                        "insertions": 3,
                        "delta": {"stathyper_type1": d},
                        "counts_after": {"stathyper_type1": a},
                        "integrity_ok": True,
                        "integrity_error": "",
                        "ids_as_expected": True,
                        "first_id_mismatch": -1,
                        "unresolved": 0,
                        "ambiguous": 0,
                    }
                    for k, (d, a) in enumerate(zip(deltas, after, strict=True))
                ],
            }
        )
    return {"modes": modes}


def test_compare_case_finds_each_difference(tmp_path: Path):
    files = []
    for k in range(2):
        f = tmp_path / f"batch{k + 1}.dgb"
        f.write_bytes(bytes([k]))
        files.append(f)
    golden = {
        "full_count": [10],
        "batches": [
            {"deletions": 2, "insertions": 3, "delta": [1], "counts_after": [11],
             "batch_sha256": tg.sha256_file(files[0])},
            {"deletions": 2, "insertions": 3, "delta": [-4], "counts_after": [7],
             "batch_sha256": tg.sha256_file(files[1])},
        ],
    }  # fmt: skip
    prep = {"batches": files}
    good = _report([10], [[1], [-4]])
    assert tr.compare_case(golden, prep, good, "stathyper_type1", ["by_key", "by_content"]) == []
    cases = {
        "full": lambda r: r["modes"][0]["full"]["stathyper_type1"].__setitem__(0, 9),
        "delta": lambda r: r["modes"][1]["batches"][1]["delta"]["stathyper_type1"].__setitem__(
            0, -3
        ),
        "integrity": lambda r: r["modes"][0]["batches"][0].update(integrity_ok=False),
        "ids": lambda r: r["modes"][0]["batches"][1].update(ids_as_expected=False),
        "unresolved": lambda r: r["modes"][1]["batches"][0].update(unresolved=1),
        "drop": lambda r: r["modes"][0]["batches"].pop(),
        "composition": lambda r: r["modes"][1]["batches"][0].update(insertions=4),
    }
    for name, plant in cases.items():
        r = copy.deepcopy(good)
        plant(r)
        problems = tr.compare_case(golden, prep, r, "stathyper_type1", ["by_key", "by_content"])
        assert problems, name
    # A batch file that is not the golden's (swapped files) is found by its SHA-256.
    problems = tr.compare_case(golden, {"batches": files[::-1]}, good, "stathyper_type1",
                               ["by_key"])  # fmt: skip
    assert any("SHA-256" in x for x in problems)
    # A missing mode is a failure.
    assert tr.compare_case(golden, prep, _report([10], [[1], [-4]], ("by_key",)), "stathyper_type1",
                           ["by_key", "by_content"])  # fmt: skip


def _compare(*extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, DYNG_PERF_LOCK="")
    return subprocess.run(
        [sys.executable, str(REPO / "parity" / "compare.py"), "triad_count", "--backend",
         "sequential", "--cases", "test/04-small_b20/type1,test/04-small_b20/main", "--jobs", "1",
         "--oracle-threads", "2", *extra],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip


@needs_goldens
@needs_tool
def test_the_replay_of_a_make_test_case_passes():
    p = _compare()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "2/2 cases passed" in p.stdout


@needs_goldens
@needs_tool
@pytest.mark.parametrize(
    ("defect", "mode", "expect"),
    [
        ("bin", "both", "full count"),
        ("row", "by_content", "by_content batch 1"),
        ("drop", "both", "batches replayed"),
        ("swap", "both", "SHA-256"),
        ("id", "by_key", "inserted ids differ"),
    ],
)
def test_each_planted_defect_fails_the_replay(defect: str, mode: str, expect: str):
    p = _compare("--plant", defect, "--mode", mode)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "FAILED" in p.stdout and expect in p.stdout, p.stdout


@needs_goldens
@needs_tool
def test_the_algorithm_counter_is_not_available_before_triad_count():
    p = _compare("--counter", "algorithm")
    assert p.returncode == 2, p.stdout + p.stderr
