# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_goldens.py (M10a step 1, ADR 0065) that need no GPU, build or
original: the corpora and their argv, O's host status of the batches, O's label text, the
manifest and case digests (the hub's distribution outside them), the comparison of two exports,
the in_place thresholds (AD5) and their recomputation from a record, the iteration window and
the iteration envelope that replaced it (the author, 2026-10-29)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace  # noqa: E402
import label_propagation_goldens as lg  # noqa: E402


def test_the_corpora_are_the_steps() -> None:
    names = [c.name for c in lg.GEN_CASES]
    assert names[:6] == ["5k5_d1e-4_s42", "50k_d1e-4_s42", "500k_d1e-4_s42", "5m_d1e-4_s42",
                         "5m1_d1e-4_s42", "50m_d1e-4_s42"]  # fmt: skip
    assert "50k_deg3_d1e-4_s42" in names and "50k_deg7_d1e-4_s42" in names
    assert {c.seed for c in lg.GEN_CASES if c.size == "5k5"} == {42, 1, 2, 3}
    assert [c.digest for c in lg.GEN_CASES if c.size in ("5m", "5m1", "50m")] == [True] * 3
    p = [c.name for c in lg.P_CASES]
    assert p == ["5k5_d1e-4_s42", "50k_d1e-4_s42", "500k_d1e-4_s42", "5k5_d1e-6_s42",
                 "50k_d1e-6_s42", "5k5_d1e-4_s1", "5k5_d1e-4_s2", "50k_d1e-4_s1", "50k_d1e-4_s2",
                 "5m_d1e-4_s42"]  # fmt: skip
    argv = lg.Case("50k", delta="1e-6", schedule="inplace").argv(Path("/o"))
    assert argv == ["--totalNodes", "50000", "--numBatches", "10", "--avgDegree", "5", "--delta",
                    "0.000001", "--seed", "42", "--update", "inplace", "--quiet", "--noValidate",
                    "--noGraphFiles", "--output", "/o"]  # fmt: skip
    assert "--noGraphFiles" not in lg.Case("5k5").argv(Path("/o"), graph_files=True)
    assert "--validate" in lg.IRLP_CASES[0].argv(Path("/o"))


def record(tag: str, batch: int, **arrays) -> dynlp_trace.Record:
    r = dynlp_trace.Record(tag, batch, batch)
    for k, v in arrays.items():
        raw = struct.pack(f"<{len(v)}i", *v)
        r.fields[k] = dynlp_trace.Field(k, 1, False, len(v), raw, lg.sha256_bytes(raw))
    return r


def test_final_status_is_o_host_pass() -> None:
    pytest.importorskip("numpy")
    batches = [
        record("batch_data", 0, inserted=[0, 1], gt_ids=[0], deleted=[]),
        # 3 is inserted and deleted in the same batch (never part of the graph), 1 is deleted
        record("batch_data", 1, inserted=[2, 3, 4], gt_ids=[4], deleted=[1, 3]),
    ]
    status = lg.final_status(6, batches)
    gt, dl, ac, ia = (lg.VERTEX_GROUND_TRUTH, lg.VERTEX_DELETED, lg.VERTEX_ACTIVE,
                      lg.VERTEX_INACTIVE)  # fmt: skip
    assert list(status) == [gt, dl, ac, dl, gt, ia]


def test_label_text_is_o_format() -> None:
    f32 = struct.unpack("<f", struct.pack("<f", 0.1234565))[0]  # a float32 label, as O prints it
    assert lg.label_text([0.5, 1.0, f32]) == b"0 0.500000\n1 1.000000\n2 0.123457\n"


def test_manifest_and_case_digests(tmp_path: Path) -> None:
    s = tmp_path / "hub"
    (s / "cases" / "a").mkdir(parents=True)
    (s / "cases" / "a" / "x.json").write_text("{}\n")
    (s / "inplace").mkdir()
    (s / "inplace" / "run-0.json").write_text("{}\n")
    (s / "PROVENANCE.json").write_text("{}\n")
    digest, files, _ = lg.write_manifest(s, "hub")
    text = (s / "MANIFEST.sha256").read_text()
    assert files == 1 and "inplace/" not in text and "PROVENANCE" not in text
    assert lg.case_digests(s)["a"]["files"] == 1
    other = tmp_path / "other" / "label_propagation_hub"
    other.mkdir(parents=True)
    (other / "MANIFEST.sha256").write_text(text)
    a, b = tmp_path / "A", tmp_path / "B"
    for root in (a, b):
        (root / "label_propagation_hub").mkdir(parents=True)
        (root / "label_propagation_hub" / "MANIFEST.sha256").write_text(text)
    assert lg.compare_exports(a, b, ["hub"]) == []
    (b / "label_propagation_hub" / "MANIFEST.sha256").write_text(text + "00  extra\n")
    assert lg.compare_exports(a, b, ["hub"])


def test_thresholds_follow_ad5_and_are_recomputed() -> None:
    rec = {"rule": lg.CALIBRATION, "cases": {"5k5_d1e-4_s42_inplace": {"pairs": [
        {"a": 0, "b": 1, "mean": 1e-7, "max": 2e-7, "flips": 0},
        {"a": 0, "b": 2, "mean": 4e-5, "max": 3e-3, "flips": 2},
    ]}}}  # fmt: skip
    t = lg.recompute_thresholds(rec)["5k5_d1e-4_s42_inplace"]
    assert t == {"t_mean": pytest.approx(1.2e-4), "t_max": pytest.approx(9e-3), "t_flips": 6}
    rec["cases"]["5k5_d1e-4_s42_inplace"]["pairs"] = [{"a": 0, "b": 1, "mean": 0.0, "max": 0.0,
                                                      "flips": 0}]  # fmt: skip
    t = lg.recompute_thresholds(rec)["5k5_d1e-4_s42_inplace"]
    assert t == {"t_mean": 1e-6, "t_max": 1e-6, "t_flips": 0}  # the floors


def test_the_iteration_window_is_ten_percent_with_a_floor_of_two() -> None:
    assert lg.iteration_window(1000) == (900, 1100)
    assert lg.iteration_window(10) == (8, 12)


def test_the_iteration_envelope_is_the_reference_spread_with_a_floor_of_two() -> None:
    # the author's rule of 2026-10-29: [0.9 x min, 1.1 x max] of the reference runs, floor 2
    assert lg.iteration_envelope(411, 872) == (369.9, 959.2)
    assert lg.iteration_envelope(57, 71) == (51.3, 78.1)
    assert lg.iteration_envelope(0, 0) == (0.0, 2.0)
    assert lg.iteration_envelope(10, 10) == (8.0, 12.0)
    env = {"iterations_lo": [0.0, 8.0], "iterations_hi": [2.0, 12.0],
           "iterations_sum_lo": 9.0, "iterations_sum_hi": 13.0}  # fmt: skip
    assert lg.iterations_inside(env, [0, 10])["within"]
    assert lg.iterations_inside(env, [0, 13]) == {"batches_outside": [1], "sum": 13,
                                                  "sum_within": True, "within": False}  # fmt: skip
    assert not lg.iterations_inside(env, [2, 7])["within"]
    with pytest.raises(lg.GoldenError):
        lg.iterations_inside(env, [0])


def test_the_frozen_envelope_is_the_committed_calibration_records() -> None:
    import json
    import tomllib

    rec = json.loads((REPO / "parity" / "results" / "M10a-inplace-calibration.json").read_text())
    spec = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())["sets"][
        "label_propagation_inplace"
    ]
    assert spec["rule"]["iterations_rule"] == "envelope"
    assert set(spec["thresholds"]) == set(rec["cases"])
    for name, case in rec["cases"].items():
        env = lg.envelope_of(case)
        assert {key: spec["thresholds"][name][key] for key in env} == env
        per = case["iterations"]["per_batch"]
        assert len(env["iterations_lo"]) == len(per) == case["batches"]
        for e, lo, hi in zip(per, env["iterations_lo"], env["iterations_hi"], strict=True):
            assert lo <= e["min"] <= e["median"] <= e["max"] <= hi


def test_pair_stats_compare_the_masked_vertices() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([0.1, 0.49, 0.9, 0.0])
    b = np.array([0.1, 0.51, 0.8, 1.0])
    mask = np.array([True, True, True, False])
    s = lg.pair_stats(a, b, mask)
    assert s["flips"] == 1 and s["max"] == pytest.approx(0.1)
    assert s["mean"] == pytest.approx((0 + 0.02 + 0.1) / 3)
