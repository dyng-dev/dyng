# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of label_propagation's timed-region map (parity/timed_regions/label_propagation.toml,
M10a step 1, ADR 0064), of M10's region keys first_batch and sum_gate and the report keys of
LabelPropagation-CUDA@a276a3a in parity/perf_ab.py and parity/bench_suite.py, and that every
earlier map and every suite under benchmarks/paper reads to the same numbers as before them."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
REQUIRED = ("update", "end_to_end", "device_memory")
# The rows and gates every earlier map read at B0 (b2b1524, before M10's keys): the SHA-256 of
# earlier_maps() below, taken with B0's parity/perf_ab.py and parity/bench_suite.py.
EARLIER_MAPS_AT_B0 = "ba585eef213eba7f57a90ef42f0e029888b5df6e153921187e7955493b0fe2a2"


def _regions(path: Path | None = None) -> dict[str, dict]:
    path = path or perf.LP_REGION_MAP
    return {r["name"]: r for r in perf.load_regions("dynlp_cuda", path, REQUIRED)}


def _write(tmp_path: Path, old: str, new: str) -> Path:
    text = perf.LP_REGION_MAP.read_text()
    assert old in text, old
    path = tmp_path / "label_propagation.toml"
    path.write_text(text.replace(old, new, 1))
    return path


def test_the_map_has_the_gated_rows_of_ad2():
    regions = _regions()
    update = regions["update"]
    rows = perf.region_rows(update)
    assert rows[0] == "update[b0]" and rows[-2] == "update[b10]" and rows[-1] == "update[sum]"
    assert update["gate"] == "none" and update["sum_gate"] == "compute"
    assert perf.row_gate(update, "update[sum]") == "compute"
    assert perf.row_gate(update, "update[b3]") == "none"
    assert update["original_report"] == ["csv:time_ms"]
    assert update["port"] == ["label_propagation.update"]
    assert regions["end_to_end"]["gate"] == "end_to_end"
    assert regions["end_to_end"]["original_report"] == [perf.PROCESS_WALL_TIME]
    assert regions["device_memory"]["gate"] == "memory"
    assert regions["device_memory"]["original_report"] == [perf.NSYS_PEAK_LIVE]
    gated = sorted(n for n, r in suite.region_map("label_propagation", "cuda").items()
                   if r["gate"] != "none")  # fmt: skip
    assert gated == ["device_memory", "end_to_end", "update[sum]"]
    for name, r in regions.items():
        if name.startswith("stage.") or name == "eval":
            assert r["gate"] == "none", name
    assert perf.batch_of_row("update[b0]") == ("update", 0)


def test_b0_rows_need_first_batch_0(tmp_path: Path):
    path = _write(tmp_path, "first_batch = 0\n", "")
    with pytest.raises(SystemExit, match="per_batch rows"):
        _regions(path)
    path = _write(tmp_path, "first_batch = 0\n", "first_batch = 2\n")
    with pytest.raises(SystemExit, match="first_batch must be 0 or 1"):
        _regions(path)
    path = _write(tmp_path, 'name = "eval"\n', 'name = "eval"\nfirst_batch = 0\n')
    with pytest.raises(SystemExit, match="needs per_batch"):
        _regions(path)
    path = _write(tmp_path, 'name = "eval"\n', 'name = "eval"\nsum_gate = "none"\n')
    with pytest.raises(SystemExit, match="needs per_batch"):
        _regions(path)


def test_sum_gate_is_compute_or_none_and_needs_report_sum(tmp_path: Path):
    path = _write(
        tmp_path,
        'report_sum = true\nsum_gate = "compute"',
        'report_sum = true\nsum_gate = "end_to_end"',
    )
    with pytest.raises(SystemExit, match="sum_gate must be"):
        _regions(path)
    path = _write(tmp_path, 'report_sum = true\nsum_gate = "compute"', 'sum_gate = "compute"')
    with pytest.raises(SystemExit, match="sum_gate needs report_sum"):
        _regions(path)
    path = _write(tmp_path, 'report_sum = true\nsum_gate = "compute"\n', "report_sum = true\n")
    assert "update[sum]" not in perf.region_rows(_regions(path)["update"])


def test_unknown_keys_report_keys_and_gates_are_refused(tmp_path: Path):
    path = _write(tmp_path, "report_sum = true\nsum_gate = ", "report_sum = true\nsum_gat = ")
    with pytest.raises(SystemExit, match="unknown key"):
        _regions(path)
    path = _write(tmp_path, 'original_report = ["csv:eval_ms"]', 'original_report = ["csv:x"]')
    with pytest.raises(SystemExit, match="cannot parse"):
        _regions(path)
    path = _write(tmp_path, 'gate = "end_to_end"', 'gate = "memory"')
    with pytest.raises(SystemExit, match="unknown gate"):  # memory only on the nsys peak row
        _regions(path)
    # the memory gate is label_propagation's only: the earlier maps still refuse it
    text = perf.TRIAD_REGION_MAP.read_text().replace('gate = "end_to_end"', 'gate = "memory"')
    other = tmp_path / "triad_count.toml"
    other.write_text(text)
    with pytest.raises(SystemExit, match="unknown gate"):
        perf.load_regions("escher_gpu", other, ("update", "full_count", "end_to_end"))


def earlier_maps() -> dict:
    out: dict = {}
    for alg, keys in suite.MAP_KEYS.items():
        if alg == "label_propagation":
            continue
        for be in keys:
            rm = suite.region_map(alg, be)
            out[f"{alg}/{be}"] = sorted(
                (row, r.get("gate"), r.get("row_of")) for row, r in rm.items()
            )
    for be, req, path in [
        ("openmp", ("sosp_update", "apply", "end_to_end"), perf.REGION_MAP),
        ("cuda", ("sosp_update", "apply", "end_to_end"), perf.REGION_MAP),
        ("openmp", ("compute", "end_to_end"), perf.MOSP_REGION_MAP),
        ("cuda", ("compute", "end_to_end"), perf.MOSP_REGION_MAP),
        ("escher_gpu", ("update", "full_count", "end_to_end"), perf.TRIAD_REGION_MAP),
        ("escher_gpu", (), perf.HYPERGRAPH_REGION_MAP),
        ("mosp_escher", (), perf.HYPERGRAPH_REGION_MAP),
    ]:
        regs = perf.load_regions(be, path, req)
        out[f"load_regions/{path.name}/{be}"] = [(r["name"], r["gate"], perf.region_rows(r))
                                                 for r in regs]  # fmt: skip
    return json.loads(json.dumps(out))


def test_every_earlier_map_reads_as_at_b0():
    text = json.dumps(earlier_maps(), sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == EARLIER_MAPS_AT_B0


def test_every_suite_validates_and_reads_the_same_numbers():
    """Every suite under benchmarks/paper validates, and each reading of its committed 0.3.0
    summary reads, from the committed record, to the same regions (ratios, verdicts)."""
    results = REPO / "benchmarks" / "results" / "0.3.0"
    checked = 0
    for path in sorted((REPO / "benchmarks" / "paper").glob("*.yaml")):
        s = suite.load_suite(path)
        assert suite.validate(s) == [], path
        summary = results / f"{s['suite']}.json"
        if not summary.is_file():
            continue
        for reading in json.loads(summary.read_text())["readings"]:
            job = dict(reading, record=str(results / reading["record"]))
            entry = suite.read_record(s, job, s["tolerance"], [])
            assert entry.get("regions") == reading.get("regions"), (path.name, reading["reading"])
            checked += 1
    assert checked >= 45
