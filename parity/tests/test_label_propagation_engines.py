# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_engines.py that need no build (M10b step 1; AD6, ADR 0101):
c from the calibration's maximum, the per-batch |dF| and digests of the label dumps, the count
comparison, the first diverging batch and vertex, the verdict and the frozen bound in
goldens.toml."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import label_propagation_engines as lpe  # noqa: E402

EPS = 2.0**-23


def dump(*batches: list[float]) -> bytes:
    return b"".join(struct.pack(f"<{len(b)}f", *b) for b in batches)


def test_constant_c_is_three_times_the_maximum_rounded_up_to_a_power_of_two() -> None:
    assert lpe.constant_c(0.0) == 1
    assert lpe.constant_c(EPS / 3) == 1
    assert lpe.constant_c(EPS) == 4  # 3 -> 4
    assert lpe.constant_c(4 * EPS) == 16  # 12 -> 16
    assert lpe.constant_c(16 * EPS / 3) == 16  # exactly 16 stays 16
    assert lpe.constant_c(1000 * EPS) == 4096  # 3000 -> 4096
    with pytest.raises(lpe.EngineError):
        lpe.constant_c(float("nan"))
    with pytest.raises(lpe.EngineError):
        lpe.constant_c(-1.0)


def test_per_batch_maximum_and_first_divergence() -> None:
    pytest.importorskip("numpy")  # per_batch_max_abs reads the dumps with NumPy
    gold = dump([0.5, 0.25, 1.0], [0.5, 0.75, 0.0])
    same = lpe.per_batch_max_abs(gold, gold, 3)
    assert same == [0.0, 0.0]
    assert lpe.first_divergence(gold, gold, 3) is None
    mine = dump([0.5, 0.25, 1.0], [0.5, 0.75 + 2 * EPS, 0.0])
    assert lpe.per_batch_max_abs(mine, gold, 3) == [0.0, pytest.approx(2 * EPS, rel=0.5)]
    first = lpe.first_divergence(mine, gold, 3)
    assert first is not None and (first["batch"], first["vertex"]) == (1, 1)
    with pytest.raises(lpe.EngineError):
        lpe.per_batch_max_abs(mine[:-4], gold, 3)
    with pytest.raises(lpe.EngineError):
        lpe.per_batch_max_abs(dump([float("nan"), 0.0, 0.0], [0.0] * 3), gold, 3)


def test_per_batch_digests_split_by_batch() -> None:
    d = lpe.per_batch_digests(dump([0.5, 0.5], [1.0, 0.0]), 2)
    assert len(d) == 2 and d[0] != d[1]


def test_counts_must_be_identical_per_batch() -> None:
    gold = [{"iterations": "3", "components": "5", "frontier_visits": "9", "accuracy": "1.0"}]
    assert lpe.count_mismatches([dict(gold[0], accuracy="0.5")], gold) == []
    bad = lpe.count_mismatches([dict(gold[0], iterations="4")], gold)
    assert bad == [{"batch": 0, "field": "iterations", "mine": "4", "golden": "3"}]
    assert lpe.count_mismatches([], gold)[0]["field"] == "batches"


def test_the_verdict_never_loosens_the_bound() -> None:
    bound = 4 * EPS
    assert lpe.judge_batches([0.0, 4 * EPS], [], bound)["equal"]
    over = lpe.judge_batches([0.0, 4.5 * EPS], [], bound)
    assert not over["equal"] and over["over"][0]["batch"] == 1
    counted = lpe.judge_batches([0.0], [{"batch": 0, "field": "iterations"}], bound)
    assert counted["within"] and not counted["equal"]


def test_variant_flags_perturb_only_association_and_contraction() -> None:
    assert lpe.variant_flags("control") == ""
    assert lpe.variant_flags("no_fma") == "-fmad=false"
    assert lpe.variant_flags("permuted") == "-DDYNG_CALIBRATION_LP_PERMUTED_REDUCTION"
    assert lpe.variant_flags("both") == "-fmad=false -DDYNG_CALIBRATION_LP_PERMUTED_REDUCTION"


def test_the_frozen_bound_is_read_and_checked(tmp_path: Path) -> None:
    toml = tmp_path / "goldens.toml"
    toml.write_text("[sets.label_propagation_damped]\nkind = 'x'\n")
    with pytest.raises(lpe.EngineError):
        lpe.frozen_bound(toml)
    toml.write_text(
        "[sets.label_propagation_damped]\nkind = 'x'\n"
        "[sets.label_propagation_damped.refactored]\nepsilon = 1.1920928955078125e-07\nc = 6\n"
    )
    with pytest.raises(lpe.EngineError):
        lpe.frozen_bound(toml)  # not a power of two
    toml.write_text(
        "[sets.label_propagation_damped]\nkind = 'x'\n"
        "[sets.label_propagation_damped.refactored]\nepsilon = 1.1920928955078125e-07\nc = 8\n"
    )
    b = lpe.frozen_bound(toml)
    assert b["c"] == 8 and b["bound"] == 8 * EPS


def test_the_repository_bound_is_the_calibrations() -> None:
    b = lpe.frozen_bound()
    assert b["c"] & (b["c"] - 1) == 0 and b["bound"] == b["c"] * EPS


def test_a_frozen_c_that_is_not_the_calibrations_is_refused(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import json

    rec = tmp_path / "cal.json"
    rec.write_text(json.dumps({"control_exact": True, "max_df": 4 * EPS}))  # c = 16
    sha = hashlib.sha256(rec.read_bytes()).hexdigest()
    monkeypatch.setattr(lpe, "REPO", tmp_path)
    toml = tmp_path / "goldens.toml"
    body = (
        "[sets.label_propagation_damped]\nkind = 'x'\n"
        "[sets.label_propagation_damped.refactored]\nepsilon = 1.1920928955078125e-07\n"
        "calibration_record = 'cal.json'\n"
    )
    toml.write_text(body + f"c = 16\ncalibration_sha256 = '{sha}'\n")
    assert lpe.frozen_bound(toml)["c"] == 16
    toml.write_text(body + f"c = 32\ncalibration_sha256 = '{sha}'\n")
    with pytest.raises(lpe.EngineError):
        lpe.frozen_bound(toml)  # looser than the calibration
    toml.write_text(body + "c = 16\ncalibration_sha256 = 'x'\n")
    with pytest.raises(lpe.EngineError):
        lpe.frozen_bound(toml)  # the record changed
