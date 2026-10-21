# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/triad_goldens.py (M9a step 1, ADR 0048): the case list, the parsing of
ESCHER-GPU's output, the per-batch .dgb split, and that every assertion of `verify` fails on a
planted defect (a comparison that cannot fail proves nothing; the M8a/M8b lesson)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "triad_goldens", REPO / "parity" / "triad_goldens.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["triad_goldens"] = module
    spec.loader.exec_module(module)
    return module


tg = _load()
needs_zstd = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is not installed")


# --- The cases ---------------------------------------------------------------------------------


def test_the_grid_and_make_test_give_300_cases():
    grid = [c for c in tg.case_names() if not c.startswith("test/")]
    test = [c for c in tg.case_names() if c.startswith("test/")]
    assert len(grid) == 240 and len(test) == 60
    assert len(set(tg.case_names())) == 300
    assert "coauth_b50k_s1/main" in grid and "syn2m_b200k_s3/type3" in grid
    assert "syn100k_b20_s2/type1" in grid
    g = {x.name: x for x in tg.grid_groups()}["coauth_b50k_s1"]
    assert g.argv("main") == [
        "main",
        "--input",
        "coauth.hg",
        "--batches",
        "3",
        "--batch-size",
        "50000",
        "--seed",
        "1",
    ]
    assert g.input_name == "coauth" and g.options.mode() == "sample"
    s = {x.name: x for x in tg.grid_groups()}["syn1m_b5k_s3"]
    assert s.input_name == "syn1m_s3" and s.options.mode() == "random"
    assert s.options.composition() == {"batch_size": 5000, "deletions": 2500, "insertions": 2500}


def test_make_test_is_the_suites_counting_invocations():
    inv = tg.suite_invocations()
    assert len(inv) == 60
    assert (
        "build/type2 --input <abs>/skewed.hg --batches 2 --batch-size 2800 --check "
        "--reference" in inv
    )
    assert "build/main 8 5 1 100 4096" in inv
    smoke = tg.test_groups()[0]
    assert smoke.input_name == "syn8_s1" and smoke.options.positional == [8, 5, 1, 100, 4096]
    assert smoke.options.composition()["default"].startswith("min(2, live)")
    ref = tg.test_groups()[2]
    assert ref.count_args == ["100000", "10", "1", "33333", "--batches", "0", "--seed", "5"]
    assert ref.input_name == "syn100k_s5"


def test_the_coverage_table_of_d6():
    cuda = tg.coverage_cases("cuda")
    assert len(cuda) == 300
    seq = tg.coverage_cases("sequential")
    assert len(seq) == 60 + 60  # make-test plus the syn100k grid
    omp = tg.coverage_cases("openmp")
    assert len(omp) == 60 + 60 + 20 + 2
    assert "coauth_b50k_s1/type2" in omp and "coauth_b50k_s1/main" not in omp
    assert "syn1m_b200k_s1/main" in omp and "syn1m_b200k_s2/main" not in omp


# --- Parsing A's output ------------------------------------------------------------------------

FULL = "1 241 7850 89258 0 2 0 117 45 2825 0 0 0 0 0 36 0 0 0 0 0 0 0 0 0 0 0 5 152 4873"
DELTA = "0 13 63 -154 10 0 237 0 0 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -15"
MAIN_OUT = (
    "Hypergraph: 3000 hyperedges, max vertex id 2000, 13503 pins\n"
    f"Motif counts (30 bins): {FULL}\n"
    "Batch 1 (10 deletions, 10 insertions)\n"
    f"Motif delta counts (30 bins): {DELTA}\n"
)
TYPE1_OUT = """Hypergraph: 3000 hyperedges, max vertex id 2000, 13503 pins
[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): 47093
[GPU] Type1 motif delta count: -12
"""


def test_parse_counts_of_main_and_type1():
    p = tg.parse_counts(MAIN_OUT, "main")
    assert p["full"][:4] == [1, 241, 7850, 89258] and len(p["full"]) == 30
    assert p["batches"][0]["printed"] == (1, 10, 10)
    assert p["batches"][0]["delta"][6] == 237
    assert p["hypergraph"] == (3000, 2000, 13503)
    t = tg.parse_counts(TYPE1_OUT, "type1")
    assert t["full"] == [47093] and t["batches"][0] == {"printed": None, "delta": [-12]}
    # the type2 counter's lines are not type1's
    with pytest.raises(tg.GoldenError):
        tg.parse_counts(TYPE1_OUT, "type2")
    assert tg.count_lines(MAIN_OUT) == MAIN_OUT.splitlines()[1:]


def test_parse_counts_refuses_short_lines_and_misnumbered_batches():
    with pytest.raises(tg.GoldenError, match="30 values"):
        tg.parse_counts("Motif counts (30 bins): 1 2 3\n", "main")
    with pytest.raises(tg.GoldenError, match="batch line 2"):
        tg.parse_counts(MAIN_OUT.replace("Batch 1 (", "Batch 2 ("), "main")


# --- .dgb ----------------------------------------------------------------------------------------


def _dgb(records: list[tuple[int, int]]) -> bytes:
    """A hypergraph .dgb (flags deleted_members | expected_ids) with (deletions, insertions) per
    record: deleted ids 0..d-1 with member sets {0}, insertions {1, 2}."""
    head = bytearray(64)
    head[:8] = b"DYNGBAT1"
    struct.pack_into("<II", head, 8, 1, 64)
    head[16:20] = bytes([1, 1, 0, 2])
    struct.pack_into("<QQ", head, 24, 3, len(records))
    struct.pack_into("<I", head, 60, zlib.crc32(bytes(head[:60])))
    body = b""
    for n, (d, i) in enumerate(records):
        m, q = 2 * i, d
        payload = struct.pack("<5Q", n, d, i, m, q)
        payload += struct.pack(f"<{d}q", *range(d))
        payload += struct.pack(f"<{i + 1}q", *range(0, m + 1, 2))
        members = struct.pack(f"<{m}i", *([1, 2] * i))
        payload += members + b"\0" * (-len(members) % 8)
        payload += struct.pack(f"<{d + 1}q", *range(d + 1))
        dm = struct.pack(f"<{q}i", *([0] * q))
        payload += dm + b"\0" * (-len(dm) % 8)
        payload += struct.pack(f"<{i}q", *range(100, 100 + i))
        body += struct.pack("<QI4x", len(payload), zlib.crc32(payload)) + payload
    return bytes(head) + body


def test_the_per_batch_split_keeps_valid_files():
    data = _dgb([(1, 2), (3, 0), (0, 4)])
    header, records = tg.dgb_records(data)
    assert len(records) == 3
    for n, rec in enumerate(records):
        single = tg.dgb_single(header, rec)
        h2, r2 = tg.dgb_records(single)
        assert r2 == [rec]
        assert struct.unpack_from("<Q", single, 32)[0] == 1
        assert struct.unpack_from("<I", single, 60)[0] == zlib.crc32(single[:60])
        assert tg.dgb_counts(rec)["batch_id"] == n
    assert [tg.dgb_counts(r)["deletions"] for r in records] == [1, 3, 0]
    with pytest.raises(tg.GoldenError, match="after the last record"):
        tg.dgb_records(data + b"x")
    with pytest.raises(tg.GoldenError, match="not a .dgb"):
        tg.dgb_records(b"DYNGBAT0" + data[8:])


def test_type1_sum_counts_choose3_of_every_row():
    rows = b"1 2 3\n# a comment\n2 3 4 5\n\n1\n"
    assert tg.type1_sum(rows) == (1 + 4 + 0, 3, 8, 4)


# --- verify fails on planted defects -----------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tiny_set(tmp_path: Path, monkeypatch) -> Path:
    """A one-group set built like the exporter's: input `small` (3 rows), batch 1 deleting one
    hyperedge and inserting one; consistent counts for the four executables."""
    rows = b"1 2 3\n2 3 4\n1 2 3 4\n"
    out = tmp_path / "set"
    (out / "inputs").mkdir(parents=True)
    hg = tmp_path / "small.hg"
    hg.write_bytes(rows)
    tg.zstd_compress(hg, out / "inputs" / "small.hg.zst")
    t1 = sum(math.comb(len(r.split()), 3) for r in rows.splitlines())
    tg.write_json(
        out / "inputs" / "small.json",
        {
            "name": "small",
            "loaded_rows_sha256": _sha(rows),
            "vertices": 4,
            "hyperedges": 3,
            "pins": 10,
            "max_cardinality": 4,
            "type1_sum_choose3": t1,
            "stored": "inputs/small.hg.zst",
        },
    )
    group = tg.Group(
        "test/99-tiny",
        "test",
        ["--input", "{small}", "--batches", "1", "--batch-size", "2", "--seed", "3"],
        "make-test",
    )
    monkeypatch.setattr(tg, "all_groups", lambda: [group])
    gdir = out / "cases" / group.name
    gdir.mkdir(parents=True)
    header, records = tg.dgb_records(_dgb([(1, 1)]))
    single = tg.dgb_single(header, records[0])
    raw = tmp_path / "batch1.dgb"
    raw.write_bytes(single)
    tg.zstd_compress(raw, gdir / "batch1.dgb.zst")
    batch = {
        "index": 1,
        "file": "batch1.dgb.zst",
        "sha256": _sha(single),
        "deletions": 1,
        "insertions": 1,
    }
    tg.write_json(
        gdir / "group.json",
        {
            "group": group.name,
            "input": "small",
            "loaded_rows_sha256": _sha(rows),
            "batches": [batch],
        },
    )
    for exe in tg.EXES:
        nb = tg.NUM_BINS[exe]
        full = [0] * nb
        full[0] = t1 if exe == "type1" else 5
        delta = [0] * nb
        delta[nb - 1] = 1
        after = [f + d for f, d in zip(full, delta, strict=True)]
        tg.write_json(
            gdir / f"{exe}.json",
            {
                "case": f"{group.name}/{exe}",
                "input": "small",
                "batches_requested": 1,
                "full_count": full,
                "batches": [
                    {
                        "index": 1,
                        "deletions": 1,
                        "insertions": 1,
                        "delta": delta,
                        "counts_after": after,
                        "batch_sha256": _sha(single),
                    }
                ],
                "check": {
                    "exit": 0,
                    "failed_lines": 0,
                    "count_lines_identical": True,
                    "integrity_ok": 2,
                    "recount_ok": 1,
                },
            },
        )
    return out


def _edit(path: Path, change) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


@needs_zstd
def test_verify_passes_on_a_consistent_set(tiny_set: Path, tmp_path: Path):
    assert tg.check_contents(tiny_set, tmp_path) == []


@needs_zstd
@pytest.mark.parametrize(
    ("what", "file", "change", "message"),
    [
        (
            "an open-triad bin",
            "main.json",
            lambda d: d["full_count"].__setitem__(22, 1),
            "open-triad bins in the full count",
        ),
        (
            "an open-triad delta",
            "main.json",
            lambda d: (
                d["batches"][0]["delta"].__setitem__(20, 1),
                d["batches"][0]["counts_after"].__setitem__(20, 1),
            ),
            "open-triad bins in the delta",
        ),
        (
            "one changed bin",
            "main.json",
            lambda d: d["batches"][0]["delta"].__setitem__(3, 2),
            "counts after differ",
        ),
        (
            "a negative count",
            "type2.json",
            lambda d: (
                d["batches"][0].__setitem__("delta", [-9]),
                d["batches"][0].__setitem__("counts_after", [-4]),
            ),
            "counts after differ or < 0",
        ),
        (
            "a type1 count off the rows",
            "type1.json",
            lambda d: (
                d["full_count"].__setitem__(0, 99),
                d["batches"][0].__setitem__("counts_after", [100]),
            ),
            "sum C(|e|, 3)",
        ),
        (
            "a wrong batch file",
            "type3.json",
            lambda d: d["batches"][0].__setitem__("batch_sha256", "0" * 64),
            "the batch file differs",
        ),
        (
            "other batch lines",
            "type3.json",
            lambda d: d["batches"][0].__setitem__("insertions", 2),
            "batch lines differ",
        ),
        (
            "a failed --check",
            "type2.json",
            lambda d: d["check"].__setitem__("recount_ok", 0),
            "--check run did not agree",
        ),
    ],
)
def test_verify_fails_on_a_planted_defect(tiny_set, tmp_path, what, file, change, message):
    _edit(tiny_set / "cases" / "test/99-tiny" / file, change)
    problems = tg.check_contents(tiny_set, tmp_path)
    assert any(message in p for p in problems), (what, problems)


@needs_zstd
def test_verify_fails_on_damaged_inputs_and_batch_files(tiny_set: Path, tmp_path: Path):
    hg = tmp_path / "other.hg"
    hg.write_bytes(b"1 2 3\n2 3 4\n1 2 3 5\n")
    tg.zstd_compress(hg, tiny_set / "inputs" / "small.hg.zst")
    assert any("loaded-rows hash" in p for p in tg.check_contents(tiny_set, tmp_path))
    raw = tmp_path / "b.dgb"
    header, records = tg.dgb_records(_dgb([(2, 1)]))
    raw.write_bytes(tg.dgb_single(header, records[0]))
    tg.zstd_compress(raw, tiny_set / "cases" / "test/99-tiny" / "batch1.dgb.zst")
    problems = tg.check_contents(tiny_set, tmp_path)
    assert any("SHA-256 differs from group.json" in p for p in problems)
    assert any("record counts differ" in p for p in problems)


@needs_zstd
def test_verify_set_checks_the_manifest_and_the_case_digests(tiny_set, tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "case_names", lambda: [f"test/99-tiny/{e}" for e in tg.EXES])
    sha, _, _ = tg.write_manifest(tiny_set)
    spec = {
        "manifest_sha256": sha,
        "cases": {c: {"sha256": tg.case_digest(tiny_set, c)} for c in tg.case_names()},
    }
    assert tg.verify_set(tiny_set, tmp_path, spec) == []
    _edit(tiny_set / "cases" / "test/99-tiny" / "main.json", lambda d: d.__setitem__("x", 1))
    problems = tg.verify_set(tiny_set, tmp_path, spec)
    assert any("differs from MANIFEST.sha256" in p for p in problems)
    assert any("case test/99-tiny/main differs" in p for p in problems)
    (tiny_set / "stray.txt").write_text("x")
    assert any("not in the manifest" in p for p in tg.verify_set(tiny_set, tmp_path, spec))


@needs_zstd
def test_run_parameters_are_not_hashed(tiny_set, tmp_path, monkeypatch):
    """The GPU, the raw trace's digest and the zstd version are provenance (PROVENANCE.json,
    recording.json): a re-export on another GPU or with another zstd reproduces the manifest and
    every case digest (M9a review)."""
    monkeypatch.setattr(tg, "case_names", lambda: [f"test/99-tiny/{e}" for e in tg.EXES])
    gdir = tiny_set / "cases" / "test/99-tiny"
    (tiny_set / tg.PROVENANCE).write_text('{"gpu": "1", "zstd": "v1.5.5"}\n')
    (gdir / tg.RECORDING).write_text('{"gpu": "1", "trace_sha256": "aa", "trace_bytes": 1}\n')
    sha, _, _ = tg.write_manifest(tiny_set)
    digests = {c: tg.case_digest(tiny_set, c) for c in tg.case_names()}
    assert tg.PROVENANCE not in (tiny_set / "MANIFEST.sha256").read_text()
    assert tg.RECORDING not in (tiny_set / "MANIFEST.sha256").read_text()
    # Another GPU and another trace: the same manifest and case digests, and verify passes.
    (tiny_set / tg.PROVENANCE).write_text('{"gpu": "GPU-e145f857", "zstd": "v1.5.6"}\n')
    (gdir / tg.RECORDING).write_text('{"gpu": "0", "trace_sha256": "bb", "trace_bytes": 2}\n')
    assert tg.write_manifest(tiny_set)[0] == sha
    assert {c: tg.case_digest(tiny_set, c) for c in tg.case_names()} == digests
    spec = {"manifest_sha256": sha, "cases": {c: {"sha256": d} for c, d in digests.items()}}
    assert tg.verify_set(tiny_set, tmp_path, spec) == []
    # A .zst file is hashed by its content: recompressed at another level, the same digests.
    batch = sorted(gdir.glob("batch*.dgb.zst"))[0]
    raw = tmp_path / "raw.dgb"
    raw.write_bytes(tg.zstd_read(batch))
    subprocess.run(["zstd", "-q", "-1", "-f", str(raw), "-o", str(batch)], check=True)
    assert tg.content_sha256(batch) == tg.sha256_bytes(raw.read_bytes())
    assert tg.write_manifest(tiny_set)[0] == sha
    assert tg.verify_set(tiny_set, tmp_path, spec) == []
