# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/triad_perf.py (M9a step 5, ADR 0056) that need no GPU, build or original: the
gate of every row (PLAN 8.6), the contamination audit, the store-accounting points the gate
reads, the launches the kernel reading selects, and the record's input digests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import triad_goldens as tg  # noqa: E402
import triad_perf  # noqa: E402


def test_gate_fields_follow_plan_8_6() -> None:
    fields = triad_perf.gate_fields("compute", 500.0, 520.0, [500.0] * 11, [520.0] * 11)
    assert fields["gate"] == 1.05 and fields["within_gate"] is True
    assert fields["provisional"] is False and fields["noisy"] is False
    over = triad_perf.gate_fields("compute", 500.0, 530.0, [500.0] * 11, [530.0] * 11)
    assert over["within_gate"] is False
    short = triad_perf.gate_fields("compute", 8.0, 8.5, [8.0] * 11, [8.5] * 11)
    assert short["gate"] == 1.10 and short["provisional"] is True  # < 10 ms, < 20 rounds
    e2e = triad_perf.gate_fields("end_to_end", 9000.0, 9800.0, [9000.0] * 3, [9800.0] * 3)
    assert e2e["gate"] == 1.10 and e2e["within_gate"] is True
    noisy = triad_perf.gate_fields("compute", 100.0, 100.0, [90.0, 100.0, 112.0], [100.0] * 3)
    assert noisy["noisy"] is True
    assert triad_perf.gate_fields("none", 100.0, 200.0, [100.0], [200.0]) == {}


def test_summarize_regions_reads_every_row() -> None:
    import perf_ab

    regions = perf_ab.load_regions(
        "escher_gpu", perf_ab.TRIAD_REGION_MAP, required=("update", "full_count", "end_to_end")
    )

    def side(update: float, full: float, total: float, a: bool) -> dict:
        times = {("init.full_count" if a else "full_count_ms", 0): full}
        times[("total" if a else "total_ms", 0)] = total
        for k in (1, 2, 3):
            times[("update" if a else "update_ms", k)] = update * k
        return times

    rows = triad_perf.summarize_regions(
        regions, [side(100, 500, 9000, True)] * 5, [side(103, 520, 9100, False)] * 5
    )
    by = {r["region"]: r for r in rows}
    assert set(by) == {"update[b1]", "update[b2]", "update[b3]", "full_count", "end_to_end"}
    assert by["update[b3]"]["original_ms"] == 300 and by["update[b3]"]["port_ms"] == 309
    assert by["update[b2]"]["gate_kind"] == "compute" and by["update[b2]"]["within_gate"]
    assert by["full_count"]["ratio"] == pytest.approx(520 / 500)
    assert by["end_to_end"]["gate"] == 1.10


def test_the_host_model_rows_beside_the_gated_update() -> None:
    """A's update.maintain holds its host model, the port keeps it after its update timer: the
    reported rows add host_model_ms to the port's update and show A's untimed maintain work."""
    a = {("update", k): 150.0 for k in (1, 2, 3)}
    a |= {("update.maintain", k): 50.0 for k in (1, 2, 3)}
    a |= {("maintain.h2h_merge", k): 30.0 for k in (1, 2, 3)}
    a |= {("maintain.row_views", k): 18.0 for k in (1, 2, 3)}
    b = {("update_ms", k): 148.0 for k in (1, 2, 3)} | {
        ("host_model_ms", k): 2.5 for k in (1, 2, 3)
    }
    rows = {r["region"]: r for r in triad_perf.host_model_rows([a] * 5, [b] * 5, 3)}
    assert rows["update_with_host_model[b2]"]["port_ms"] == 150.5
    assert rows["update_with_host_model[b2]"]["ratio"] == pytest.approx(150.5 / 150.0)
    assert rows["update_with_host_model[b2]"]["gate_kind"] == "none"
    assert rows["update.maintain_untimed[b3]"]["original_ms"] == pytest.approx(2.0)
    assert rows["update.maintain_untimed[b3]"]["port_ms"] == 2.5
    # A driver without the timer: no rows.
    assert triad_perf.host_model_rows([a], [{("update_ms", 1): 1.0}], 3) == []


def test_the_contamination_audit() -> None:
    quiet = {"before_a": 3.0, "blocked_before_a": 0, "after_a": 7.9, "blocked_after_a": 0,
             "after_b": 7.0, "blocked_after_b": 1}  # fmt: skip
    window = {"original": {"gpu": {"foreign_processes": []}}, "port": {"gpu": {}}}
    out = triad_perf.audit([quiet] * 3, [window] * 3)
    assert out["flagged"] is False and out["load_average_max"] == 7.9
    loaded = dict(quiet, after_b=27.0)
    blocked = dict(quiet, blocked_after_a=9)
    foreign = {"original": {"gpu": {"foreign_processes": [{"pid": 1, "name": "x"}]}}, "port": {}}
    out = triad_perf.audit([quiet, loaded, blocked, quiet], [window, window, window, foreign])
    assert out["flagged"] is True
    assert [f["round"] for f in out["flagged_rounds"]] == [2, 3, 4]
    assert "load average 27.0" in out["flagged_rounds"][0]["why"][0]
    assert "procs_blocked 9" in out["flagged_rounds"][1]["why"][0]
    assert "foreign GPU processes" in out["flagged_rounds"][2]["why"][0]


def test_the_store_points_the_gate_reads() -> None:
    gdoc = {"device_bytes": [100, 101, 102, 103]}
    groups = {g.name: g for g in tg.all_groups()}
    coauth = triad_perf.store_points(groups["coauth_b50k_s1"], gdoc, [100, 101, 102, 104])
    assert [p["row"] for p in coauth] == [
        "store_memory[build]",
        "store_memory[b1]",
        "store_memory[b2]",
        "store_memory[b3]",
    ]
    assert all(p["gated"] for p in coauth) and coauth[3]["ratio"] == pytest.approx(104 / 103)
    syn = triad_perf.store_points(groups["syn1m_b50k_s1"], gdoc, [100, 101, 102, 103])
    assert [p["gated"] for p in syn] == [True, False, False, True]  # A's two print points
    assert [p["print_point"] for p in syn] == [True, False, False, True]
    missing = triad_perf.store_points(groups["syn2m_b50k_s1"], gdoc, [None, 101, 102, 103])
    assert missing[0]["ratio"] is None


def test_the_kernel_reading_selects_the_update_launches() -> None:
    golden = {"batches": [{"index": k, "deletions": 5, "insertions": 5} for k in (1, 2, 3)]}
    skip, order = triad_perf.phase_layout(golden, "main")
    assert skip == 2 and len(order) == 12
    assert order[:4] == [(1, "subtract"), (1, "subtract"), (1, "add"), (1, "add")]
    skip, order = triad_perf.phase_layout(golden, "type1")
    assert skip == 1 and order == [(b, p) for b in (1, 2, 3) for p in ("subtract", "add")]
    golden["batches"][1]["insertions"] = 0
    with pytest.raises(SystemExit):
        triad_perf.phase_layout(golden, "main")
    assert set(triad_perf.NCU_KERNELS["original"]) == set(triad_perf.EXES)


def test_the_suite_digests_are_the_golden_inputs() -> None:
    digests = triad_perf.golden_input_digests()
    assert digests["coauth"] == tg.COAUTH_SHA256
    assert {f"syn{n}_s{s}" for n in ("1m", "2m") for s in (1, 2, 3)} <= set(digests)
