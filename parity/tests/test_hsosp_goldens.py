# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of the H-SOSP dump reader and writer (parity/hsosp_dump.py) and of the golden kind
hyper_sssp (parity/hsosp_goldens.py; M11a criteria 2 and 3, ADR 0081): the field encoding round
trip, the canonical encodings of ADR 0080, the rewrite of a dump into a case (the run-varying
fields left out, the digest mode, the loads), the case selection, B's deterministic output, and
that verify_case finds a planted change."""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
from array import array
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


hd = _load("hsosp_dump")
hg = _load("hsosp_goldens")
needs_zstd = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is not installed")


def arr(values, elem=4):
    return hd.ints_array(values, elem)


def rows(name, rs):
    off, vals = [0], []
    for r in rs:
        vals += r
        off.append(len(vals))
    return {f"{name}.offsets": arr(off, 8), f"{name}.values": arr(vals)}


def synthetic_dump(path: Path, work: int = 77, w2: int = 7) -> Path:
    """A small dump in the recorder's layout: one session, one batch, an update and a recompute."""
    recs = [
        (
            "header",
            {
                "format_version": 1,
                "writer_version": "1.0.0",
                "repo": "MOSP_ESCHER",
                "commit": hg.COMMIT,
                "rows": "all",
                "argv": ["<abs>/hsospBench", "--hg", "x"],
                "inputs": [],
                "env": ["CUDA_VISIBLE_DEVICES=1"],
                "seq": 0,
            },
        ),
        (
            "session",
            {
                "session": 1,
                "num_vertices": 4,
                "caps.max_hyperedges": 16,
                "caps.headroom_factor": "1.3",
                "caps.extra_payload_ints": 99,
                "exception": "",
                "seq": 1,
            },
        ),
        (
            "load",
            {
                "session": 1,
                "phase": "enter",
                **rows("rows", [[0], [0, 1], [1, 2], [3]]),
                "weights": arr([0, 5, w2, 0], 8),
                "source_he": 1,
                "target_he": 4,
                "seq": 2,
            },
        ),
        (
            "load",
            {
                "session": 1,
                "phase": "exit",
                "exception": "",
                "max_id": 4,
                "num_vertices": 4,
                "h2h_pair_count": 2,
                "alive_count": 4,
                "source_he": 1,
                "target_he": 4,
                "free_ids": 0,
                "line_graph_entries": 4,
                "seq": 3,
            },
        ),
        (
            "begin_batch",
            {
                "session": 1,
                "phase": "enter",
                "index": 0,
                "he_delete": arr([]),
                "vtx_delete.he": arr([]),
                "vtx_delete.vertex": arr([]),
                "vtx_insert.he": arr([4]),
                "vtx_insert.vertex": arr([2]),
                **rows("he_insert", []),
                "he_insert.weight": arr([], 8),
                "seq": 4,
            },
        ),
        (
            "begin_batch",
            {
                "session": 1,
                "phase": "exit",
                "index": 0,
                "exception": "",
                "final_ids": arr([]),
                "final_weights": arr([], 8),
                "dead_he": arr([]),
                "skipped_ops": 0,
                "max_id": 4,
                "seq": 5,
            },
        ),
        (
            "apply_batch",
            {
                "session": 1,
                "index": 0,
                "entry_headroom": "1.6",
                "exception": "",
                "bt.candidates": 2,
                "delta.num_del": 0,
                "delta.num_ins": 1,
                "pairs.deleted.a": arr([]),
                "pairs.deleted.b": arr([]),
                "pairs.inserted.a": arr([2]),
                "pairs.inserted.b": arr([3]),
                "max_id": 4,
                "h2h_pair_count": 3,
                "alive_count": 4,
                "dev.num_nodes": 4,
                "rows.kind": "all",
                "rows.ids": arr([1, 2, 3, 4]),
                **rows("rows", [[0], [0, 1], [1, 2], [2, 3]]),
                "rows.weight": arr([0, 5, 7, 0], 8),
                "rows.alive": arr([1, 1, 1, 1]),
                "seq": 6,
            },
        ),
        (
            "sssp",
            {
                "session": 1,
                "kind": "update",
                "state": 0,
                "batches": 1,
                "source": 1,
                "cfg.max_iterations": 4096,
                "cfg.work_budget": "1",
                "cfg.block_size": 256,
                "exception": "",
                "stats.iterations": 3,
                "stats.fallback_recompute": 0,
                "stats.fallback_iterations": 0,
                "stats.max_frontier": 2,
                "stats.seed_count": 1,
                "stats.jump_rounds": 1,
                "stats.work": work,
                "stats.num_entries": 6,
                "n": 4,
                "distances": arr([0, 5, 12, 12], 8),
                "parents": arr([-1, 0, 1, 2]),
                "seq": 7,
            },
        ),
        ("compare", {"session": 1, "state_a": 0, "state_b": 1, "n": 4, "mismatches": 0, "seq": 8}),
        ("end", {"exit": "normal", "records": 9, "seq": 9}),
    ]
    body = bytearray(hd.file_head())
    for kind, rec in recs:
        body += hd.encode_record(kind, rec)
    path.write_bytes(bytes(body))
    return path


def test_records_round_trip(tmp_path: Path) -> None:
    p = synthetic_dump(tmp_path / "a.hsd")
    recs = list(hd.Dump(p).records())
    assert [k for k, _ in recs][:3] == ["header", "session", "load"]
    body = bytearray(hd.file_head())
    for kind, rec in recs:
        body += hd.encode_record(kind, rec)
    assert bytes(body) == p.read_bytes()  # reader and writer agree on every byte
    assert hd.rows_of(recs[2][1], "rows") == [[0], [0, 1], [1, 2], [3]]


def test_a_foreign_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "x.hsd").write_bytes(b"DYNGCBT1" + struct.pack("<II", 1, 0))
    with pytest.raises(hd.DumpError):
        hd.Dump(tmp_path / "x.hsd")


def test_canonical_encodings() -> None:
    assert hd.canonical_distances([0, 5]) == struct.pack("<2q", 0, 5)
    assert hd.canonical_distances(array("q", [0]), 3) == struct.pack("<3q", 0, hd.INF, hd.INF)
    assert hd.canonical_parents([-1, 0], 3) == struct.pack("<3i", -1, 0, -1)
    rec = {
        "pairs.deleted.a": arr([3, 1]),
        "pairs.deleted.b": arr([4, 2]),
        "pairs.inserted.a": arr([0]),
        "pairs.inserted.b": arr([5]),
    }
    data, nd, ni = hd.canonical_pairs(rec)
    assert (nd, ni) == (2, 1)
    assert data == struct.pack("<6i", 1, 2, 3, 4, 0, 5)  # sorted, deleted first
    rec["pairs.inserted.a"] = arr([5])
    rec["pairs.inserted.b"] = arr([0])
    with pytest.raises(hd.DumpError):
        hd.canonical_pairs(rec)


def test_bs_deterministic_output() -> None:
    text = "timestamp,t_escher_ms,h2h_pairs,fallback,iters\n2026,1.5,10,0,7\n"
    assert hd.csv_deterministic(text) == [{"h2h_pairs": "10", "fallback": "0"}]
    log = "[load] x: built in 9000 ms; h2h pairs=3\n[row] a dyn=1.0ms\n[warn] x: overflow\n"
    assert hg.deterministic_log(log) == [
        "[load] x: built in <ms>; h2h pairs=3",
        "[warn] x: overflow",
    ]


def test_case_selection() -> None:
    assert hg.select("t") == (True, [])
    t, names = hg.select("p50")
    assert not t and names == list(hg.P50)
    t, names = hg.select("all")
    assert t and len(names) == 16
    assert hg.select("t,dblp_vertex_50000") == (True, ["dblp_vertex_50000"])
    with pytest.raises(SystemExit):
        hg.select("dblp_vertex_123")


@needs_zstd
def test_rewrite_leaves_the_run_varying_fields_out(tmp_path: Path) -> None:
    a = synthetic_dump(tmp_path / "a.hsd", work=77)
    b = synthetic_dump(tmp_path / "b.hsd", work=78)  # another run: another work count
    (tmp_path / "ca").mkdir()
    (tmp_path / "cb").mkdir()
    sa, la = hg.rewrite_dump(a, tmp_path / "ca" / "dump.hsd.zst", False, False, None, None)
    sb, lb = hg.rewrite_dump(b, tmp_path / "cb" / "dump.hsd.zst", False, False, None, None)
    assert sa == sb
    assert (tmp_path / "ca" / "dump.hsd.zst").read_bytes() == (
        tmp_path / "cb" / "dump.hsd.zst"
    ).read_bytes()
    assert la[0]["stats.work"] == 77 and lb[0]["stats.work"] == 78
    rec = next(r for k, r in hd.Dump(tmp_path / "ca" / "dump.hsd.zst").records() if k == "sssp")
    assert not any(k in rec for k in hd.LOGGED_FIELDS)
    s = sa["sessions"][0]
    assert s["apply"][0]["pairs.inserted"] == 1
    assert s["sssp"][0]["distances.sha256"] == hd.sha256(struct.pack("<4q", 0, 5, 12, 12))


@needs_zstd
def test_digest_mode_and_loads(tmp_path: Path) -> None:
    raw = synthetic_dump(tmp_path / "a.hsd")
    out = tmp_path / "set"
    (out / "paper" / "c").mkdir(parents=True)
    s, _ = hg.rewrite_dump(
        raw, out / "paper" / "c" / "dump.hsd.zst", True, True, out / "loads", "d"
    )
    recs = dict(
        (k, r)
        for k, r in hd.Dump(out / "paper" / "c" / "dump.hsd.zst").records()
        if k in ("sssp", "apply_batch")
    )
    assert "distances" not in recs["sssp"] and recs["sssp"]["distances.count"] == 4
    assert "pairs.inserted.a" not in recs["apply_batch"]
    assert recs["apply_batch"]["pairs.sha256"] == s["sessions"][0]["apply"][0]["pairs.sha256"]
    load = next(r for k, r in hd.Dump(out / "loads" / "d.hsd.zst").records() if k == "load")
    assert hd.rows_of(load, "rows") == [[0], [0, 1], [1, 2], [3]]
    # A second case of the dataset must load the same rows.
    other = synthetic_dump(tmp_path / "b.hsd", w2=8)
    (out / "paper" / "d").mkdir()
    with pytest.raises(hg.ExportError, match="different rows"):
        hg.rewrite_dump(other, out / "paper" / "d" / "dump.hsd.zst", True, True, out / "loads", "d")


@needs_zstd
def test_verify_case_finds_a_planted_change(tmp_path: Path) -> None:
    out = tmp_path / "set"
    raw = synthetic_dump(tmp_path / "a.hsd")
    case = "test/01-x"
    (out / case).mkdir(parents=True)
    s, _ = hg.rewrite_dump(raw, out / case / "dump.hsd.zst", False, False, None, None)
    c = {"case": case, "digest_mode": False, **s}
    (out / case / "case.json").write_text(json.dumps(c))
    assert hg.verify_case(out, case) == []
    for key, value in (
        ("distances.sha256", "0" * 64),
        ("parents.sha256", "1" * 64),
        ("fallback", 1),
    ):
        bad = json.loads(json.dumps(c))
        bad["sessions"][0]["sssp"][0][key] = value
        (out / case / "case.json").write_text(json.dumps(bad))
        assert hg.verify_case(out, case), key
    bad = json.loads(json.dumps(c))
    bad["sessions"][0]["apply"][0]["pairs.sha256"] = "2" * 64
    (out / case / "case.json").write_text(json.dumps(bad))
    assert any("pairs" in p for p in hg.verify_case(out, case))
