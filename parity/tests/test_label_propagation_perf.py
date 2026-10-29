# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_perf.py (M10a step 1, ADR 0064) that need no GPU, build or
original: O's argv per case (bench/run.sh's, --noGraphFiles only at 50m), the results.csv parser,
the per-round guard, the rows of the map a reading summarizes, the short-region rule, the record
shapes and the command lines of run, kernels and engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import label_propagation_perf as lp  # noqa: E402
import perf_ab  # noqa: E402

CSV = (
    "batch,time_ms,iterations,frontier_visits,components,accuracy,eval_ms\n"
    "0,0.744,0,0,50,1.000000,0.018\n"
    "1,11.133,1217,49918,829,0.508520,0.029\n"
)


def facts(iterations=(0, 1217), components=(50, 829), digest="a", tau="0.5") -> dict:
    rows = lp.parse_results(CSV)
    for r, it, c in zip(rows, iterations, components, strict=True):
        r["iterations"], r["components"] = it, c
    return {"batches": rows, "labels_sha256": digest, "tau": tau,
            "update_sum_ms": sum(r["time_ms"] for r in rows), "eval_ms": 0.047,
            "end_to_end_ms": 300.0}  # fmt: skip


def test_case_argv_is_bench_run_sh() -> None:
    assert lp.case_argv("50k", "damped") == [
        "--totalNodes", "50000", "--numBatches", "10", "--avgDegree", "5", "--delta", "0.0001",
        "--seed", "42", "--update", "damped",
    ]  # fmt: skip
    assert lp.case_argv("5m1", "inplace")[:4] == ["--totalNodes", "5000000", "--numBatches", "2"]
    t = lp.timed_argv("500k", "damped", Path("/x"))
    assert t[-2:] == ["--output", "/x"]
    assert "--quiet" in t and "--noValidate" in t and "--noGraphFiles" not in t
    assert "--noGraphFiles" in lp.timed_argv("50m", "damped", Path("/x"))
    with pytest.raises(SystemExit, match="unknown case"):
        lp.case_argv("5k", "damped")
    with pytest.raises(SystemExit, match="unknown schedule"):
        lp.case_argv("50k", "jacobi")
    assert lp.ROUNDS["50k"] == lp.ROUNDS["500k"] == 21 and lp.ROUNDS["5m"] == lp.ROUNDS["5m1"] == 11


def test_results_csv_is_read_with_o_header_only() -> None:
    rows = lp.parse_results(CSV)
    assert rows[1] == {"time_ms": 11.133, "iterations": 1217, "frontier_visits": 49918,
                       "components": 829, "accuracy": "0.508520", "eval_ms": 0.029}  # fmt: skip
    with pytest.raises(SystemExit, match="header"):
        lp.parse_results(CSV.replace("eval_ms", "irlp_time_ms"))
    with pytest.raises(SystemExit, match="batch"):
        lp.parse_results(CSV.replace("\n1,", "\n2,"))


def test_the_guard_can_fail() -> None:
    first = facts()
    assert lp.guard("damped", first, facts()) == []
    assert lp.guard("damped", first, facts(iterations=(0, 1218)))
    assert lp.guard("damped", first, facts(digest="b"))
    assert lp.guard("damped", first, facts(tau="0.6"))
    # in_place: iterations and labels vary between runs; the components may not
    assert lp.guard("inplace", first, facts(iterations=(0, 1300), digest="b")) == []
    assert lp.guard("inplace", first, facts(components=(50, 830)))


def test_the_rows_are_the_maps() -> None:
    rows = lp.summarize_rows([facts(), facts()])
    assert list(rows) == ["update[b0]", "update[b1]", "update[sum]", "eval", "end_to_end"]
    assert rows["update[sum]"]["median"] == pytest.approx(11.877)
    regions = perf_ab.load_regions("dynlp_cuda", perf_ab.LP_REGION_MAP,
                                   ("update", "end_to_end", "device_memory"))  # fmt: skip
    update = next(r for r in regions if r["name"] == "update")
    assert len(update["rows"]) == lp.MAX_ROWS == lp.CASES["50m"][1]
    many = facts()
    many["batches"] = many["batches"] * 6
    with pytest.raises(SystemExit, match="rows"):
        lp.summarize_rows([many])


def test_the_short_region_rule_is_plan_8_6() -> None:
    assert lp.short_region("damped", {"update[sum]": {"median": 31.0}})["at_least_10ms"]
    short = lp.short_region("damped", {"update[sum]": {"median": 8.0}})
    assert not short["at_least_10ms"] and "1.10x with at least 20 runs" in short["rule"]


def test_record_shapes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "reference_record", lambda: {"name": lp.REFERENCE})
    out = tmp_path / "b.json"
    args = argparse.Namespace(json=out, gpu=0)
    lp.write_baseline(args, {}, {"50k/damped": {}}, "2026-10-28T00:00:00Z")
    assert list(json.loads(out.read_text())) == lp.RECORD["baseline"]["record"]
    out2 = tmp_path / "m.json"
    lp.write_memory(argparse.Namespace(json=out2, gpu=1), {}, {}, "2026-10-28T00:00:00Z")
    assert list(json.loads(out2.read_text())) == lp.RECORD["memory"]["record"]
    for command in ("run", "kernels", "engines"):
        assert {"record", "result"} <= set(lp.RECORD[command])


@pytest.mark.parametrize("command", ["run", "kernels", "engines"])
def test_the_port_commands_parse_and_wait_for_the_port(command: str) -> None:
    args = lp.parser().parse_args([command, "--exe", "x", "--cases", "50k"])
    assert args.cases == ["50k"] and args.exe == Path("x")
    with pytest.raises(SystemExit, match="arrives in M10a steps 4-5"):
        lp.main([command, "--exe", "x", "--cases", "50k"])


def test_the_command_line_refuses_unknown_cases_and_few_rounds() -> None:
    with pytest.raises(SystemExit, match="unknown case"):
        lp.main(["baseline", "--cases", "60k"])
    with pytest.raises(SystemExit, match="rounds"):
        lp.main(["baseline", "--cases", "50k", "--rounds", "2"])
    with pytest.raises(SystemExit):
        lp.parser().parse_args(["run"])  # --exe is required
