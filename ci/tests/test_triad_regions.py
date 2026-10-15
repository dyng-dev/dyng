# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of triad_count's timed-region map (parity/timed_regions/triad_count.toml, M9a step 1,
ADR 0048) and of the per-batch rows that parity/perf_ab.py and parity/bench_suite.py read."""

from __future__ import annotations

import importlib.util
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


perf = _load("perf_ab")
suite = _load("bench_suite")
REQUIRED = ("update", "full_count", "end_to_end")


def _regions(path: Path | None = None) -> dict[str, dict]:
    path = path or perf.TRIAD_REGION_MAP
    return {r["name"]: r for r in perf.load_regions("escher_gpu", path, REQUIRED)}


def _write(tmp_path: Path, old: str, new: str) -> Path:
    text = perf.TRIAD_REGION_MAP.read_text()
    assert old in text, old
    path = tmp_path / "triad_count.toml"
    path.write_text(text.replace(old, new, 1))
    return path


def test_the_map_loads_with_the_gated_rows_of_plan_8_6():
    regions = _regions()
    gated = {name: r for name, r in regions.items() if r["gate"] != "none"}
    assert set(gated) == {"update", "full_count", "end_to_end"}
    update = gated["update"]
    assert update["per_batch"] and update["report_sum"]
    assert perf.region_rows(update) == ["update[b1]", "update[b2]", "update[b3]"]
    assert update["original_report"] == ["timing:update"]
    assert update["port"] == ["triad_count.update"]
    assert update["gate"] == "compute"
    assert gated["full_count"]["original_report"] == ["timing:init.full_count"]
    assert gated["full_count"]["port"] == ["triad_count.compute"]
    assert gated["full_count"]["gate"] == "compute"
    assert gated["end_to_end"]["original_report"] == ["timing:total"]
    assert gated["end_to_end"]["gate"] == "end_to_end"
    assert perf.region_rows(gated["full_count"]) == ["full_count"]
    assert perf.batch_of_row("update[b3]") == ("update", 3)
    assert perf.batch_of_row("full_count") == ("full_count", None)


def test_the_sub_stages_add_up_to_the_gated_update():
    """subtract + maintain + add on A's side is update; the port stages of the three sub-regions
    are exactly the stages of triad_count.update (ADR 0048 section 1)."""
    regions = _regions()
    assert regions["update.subtract"]["original_report"] == ["timing:update.subtract"]
    assert regions["update.maintain"]["original_report"] == ["timing:update.maintain"]
    assert regions["update.add"]["original_report"] == ["timing:update.add"]
    stages = [s for n in ("subtract", "maintain", "add") for s in regions[f"update.{n}"]["port"]]
    assert stages == [
        "triad_count.normalize",
        "triad_count.count_minus",
        "triad_count.commit",
        "triad_count.identify_affected",
        "triad_count.count_plus",
        "triad_count.finalize",
    ]
    for name in ("update.subtract", "update.maintain", "update.add", "build", "upload", "load"):
        assert regions[name]["gate"] == "none"
    assert regions["process"]["original_report"] == [perf.PROCESS_WALL_TIME]


def test_an_unknown_timer_is_refused(tmp_path: Path):
    path = _write(
        tmp_path,
        'original_report = ["timing:init.full_count"]',
        'original_report = ["timing:nope"]',
    )
    with pytest.raises(SystemExit, match="cannot parse"):
        _regions(path)


def test_an_unknown_key_is_refused(tmp_path: Path):
    path = _write(tmp_path, "per_batch = true\nrows", "per_bach = true\nrows")
    with pytest.raises(SystemExit, match="unknown key"):
        _regions(path)


def test_per_batch_rows_must_be_consecutive_and_named_after_the_region(tmp_path: Path):
    for rows in (
        '["update[b1]", "update[b3]"]',
        '["update[b2]", "update[b1]", "update[b3]"]',
        '["upd[b1]", "upd[b2]", "upd[b3]"]',
        "[]",
    ):
        path = _write(
            tmp_path, 'rows = ["update[b1]", "update[b2]", "update[b3]"]', f"rows = {rows}"
        )
        with pytest.raises(SystemExit, match="per_batch rows"):
            _regions(path)
    path = _write(tmp_path, 'rows = ["update[b1]", "update[b2]", "update[b3]"]\n', "")
    with pytest.raises(SystemExit, match="per_batch rows"):
        _regions(path)


def test_rows_without_per_batch_and_both_kinds_of_rows_are_refused(tmp_path: Path):
    path = _write(tmp_path, 'name = "full_count"\n', 'name = "full_count"\nreport_sum = true\n')
    with pytest.raises(SystemExit, match="needs per_batch"):
        _regions(path)
    path = _write(
        tmp_path,
        'name = "update"\nper_objective = false',
        'name = "update"\nper_objective = true',
    )
    with pytest.raises(SystemExit, match="per_objective and per_batch"):
        _regions(path)


def test_a_missing_gated_region_is_refused(tmp_path: Path):
    path = _write(tmp_path, 'name = "full_count"', 'name = "full_count_renamed"')
    with pytest.raises(SystemExit, match="'full_count' is missing"):
        _regions(path)


def test_bench_suite_reads_one_entry_per_gate_row():
    regions = suite.region_map("triad_count", "cuda")
    gated = sorted(name for name, r in regions.items() if r["gate"] != "none")
    assert gated == ["end_to_end", "full_count", "update[b1]", "update[b2]", "update[b3]"]
    assert "update" not in regions
    assert regions["update[b2]"]["row_of"] == "update"
    assert regions["update[b2]"]["port"] == ["triad_count.update"]
    # the maps without per-batch regions read as before
    assert "sosp_update" in suite.region_map("sssp", "cuda")


def test_the_certificate_folds_the_rows_back_into_their_region():
    certify = _load("certify")
    assert certify.REGION_SUFFIX.sub("", "update[b2]") == "update"
    assert certify.REGION_SUFFIX.sub("", "full_count") == "full_count"
