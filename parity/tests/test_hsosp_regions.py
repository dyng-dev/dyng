# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of hyper_sssp's timed-region map (parity/timed_regions/hyper_sssp.toml, M11a step 1,
ADR 0080) and of M11's half of the joint region-key specification in parity/perf_ab.py: the
list-valued (summed) original keys, t_dynamic_total_ms, the e2e: keys and HD3's gated rows."""

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

# hsospBench's CSV header (hsospBench.cu:155-164) and two rows as B writes them.
CSV_HEADER = (
    "timestamp,suite,experiment,dataset,m_hyperedges,n_vertices,c_min,c_max,pool_size,"
    "bridge_frac,avg_h2h_deg,h2h_pairs,batch_kind,batch_size,del_pct,placement,rep,seed,"
    "t_escher_ms,t_delta_ms,t_csr_apply_ms,t_sosp_update_ms,t_dynamic_total_ms,t_static_ms,"
    "speedup,iters,fallback,seeds,max_frontier,overflow_rebuilds,dev_mem_mb,escher_mb,graph_mb,"
    "reachable_frac,verified,correct,oracle,mismatch_static,mismatch_oracle,parent_errors,"
    "fallback_iters,invalidated,update_work,static_iters"
)
CSV_ROWS = [
    "2026-10-28T10:00:00,real,paper,coauth.hg,2466792,1924991,1,25,0,0,7.1,8791234,hyperedge,"
    "50000,50,random,0,20260725,80.5,40.25,10.5,20.0,151.25,30.0,0.2,12,0,1234,99,0,900.1,"
    "500.2,300.3,0.9,0,1,none,0,-1,-1,0,1234,5678,40",
    "2026-10-28T10:00:01,real,paper,coauth.hg,2466792,1924991,1,25,0,0,7.1,8791001,hyperedge,"
    "50000,50,random,1,20260725,81.5,41.25,11.5,21.0,155.25,31.0,0.2,12,1,1234,99,0,900.1,"
    "500.2,300.3,0.9,0,1,none,0,-1,-1,7,1234,5678,40",
]
E2E = (
    "[load] coauth.hg: built in 9000 ms\n"
    "[e2e] coauth.hg: load 21345 ms, 2 batches: dynamic 307 ms, static 61 ms, wall 23456 ms\n"
)


def _regions(path: Path | None = None) -> dict[str, dict]:
    return {r["name"]: r for r in perf.load_hsosp_regions(path or perf.HSOSP_REGION_MAP)}


def _write(tmp_path: Path, old: str, new: str) -> Path:
    text = perf.HSOSP_REGION_MAP.read_text()
    assert old in text, old
    path = tmp_path / "hyper_sssp.toml"
    path.write_text(text.replace(old, new, 1))
    return path


def test_the_map_loads_with_hd3s_gates() -> None:
    regions = _regions()
    gated = {n: r["gate"] for n, r in regions.items() if r["gate"] != "none"}
    assert gated == perf.HSOSP_GATED_REGIONS
    assert perf.region_rows(regions["maintain"]) == ["maintain[b1]", "maintain[b2]", "maintain[b3]"]
    assert regions["maintain"]["original_report"] == ["csv:t_delta_ms", "csv:t_csr_apply_ms"]
    assert regions["maintain"]["port"] == ["hyper_sssp.normalize", "hypergraph.apply"]
    assert regions["dynamic"]["original_report"] == ["csv:t_dynamic_total_ms"]
    assert regions["dynamic_comparable"]["original_report"] == [
        "csv:t_delta_ms",
        "csv:t_csr_apply_ms",
        "csv:t_sosp_update_ms",
    ]
    assert regions["load_plus_dynamic"]["original_report"] == ["e2e:load", "e2e:dynamic"]
    assert regions["load_plus_dynamic"]["gate"] == "end_to_end"
    for name in ("escher", "delta", "csr_apply", "load", "wall", "static_sum", "process"):
        assert regions[name]["gate"] == "none"
    # Every per-batch gate reports its 3-batch sum, never as a gate reading.
    for name, kind in perf.HSOSP_GATED_REGIONS.items():
        if kind == "compute":
            assert regions[name]["report_sum"] is True


def test_the_stage_names_the_port_must_emit() -> None:
    stages = {s for r in _regions().values() for s in r.get("port", [])}
    for s in ("normalize", "ids", "incidence", "translate", "line_graph"):
        assert f"hypergraph.apply.{s}" in stages
    for s in ("update", "identify_affected", "seed", "loop", "recompute", "finalize", "compute"):
        assert f"hyper_sssp.{s}" in stages


def test_the_hypergraph_stub_points_at_the_map() -> None:
    text = perf.HYPERGRAPH_REGION_MAP.read_text()
    assert "hyper_sssp.toml" in text
    regions = perf.load_regions("mosp_escher", perf.HYPERGRAPH_REGION_MAP, ("escher",))
    assert [r["name"] for r in regions] == ["escher"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # an unknown timer of the original
        (
            'original_report = ["csv:t_escher_ms"]',
            'original_report = ["csv:t_bogus_ms"]',
            "t_bogus_ms",
        ),
        ('original_report = ["e2e:wall"]', 'original_report = ["e2e:walltime"]', "e2e:walltime"),
        # a timer listed twice in a summed original
        (
            'original_report = ["csv:t_delta_ms", "csv:t_csr_apply_ms"]',
            'original_report = ["csv:t_delta_ms", "csv:t_delta_ms"]',
            "listed twice",
        ),
        # a summed original that is not a list
        (
            'original_report = ["csv:t_static_ms"]',
            'original_report = "csv:t_static_ms"',
            "must be a list",
        ),
        # an e2e: key in a per-batch region
        (
            'original_report = ["csv:t_static_ms"]',
            'original_report = ["e2e:static"]',
            "not per batch",
        ),
        # e2e: and csv: keys summed together
        (
            'original_report = ["e2e:load", "e2e:dynamic"]',
            'original_report = ["e2e:load", "csv:t_dynamic_total_ms"]',
            "summed together",
        ),
        # an unknown gated row name
        (
            'name = "escher"\nper_objective = false',
            'name = "escher_gated"\nper_objective = false',
            None,
        ),
        # a missing gated row
        ('name = "dynamic_comparable"', 'name = "dynamic_comparable_x"', None),
        # a misspelt key
        ('per_batch = true\nrows = ["static[b1]"', 'per_bach = true\nrows = ["static[b1]"', None),
    ],
)
def test_the_loader_refuses(tmp_path: Path, old: str, new: str, message: str | None) -> None:
    path = _write(tmp_path, old, new)
    if "escher_gated" in new:  # make the renamed region gated
        text = path.read_text().replace(
            'port = ["hypergraph.apply.ids"]\ngate = "none"',
            'port = ["hypergraph.apply.ids"]\ngate = "compute"',
        )
        text = text.replace(
            'rows = ["escher[b1]", "escher[b2]", "escher[b3]"]',
            'rows = ["escher_gated[b1]", "escher_gated[b2]", "escher_gated[b3]"]',
        )
        path.write_text(text)
    if "dynamic_comparable_x" in new:
        path.write_text(
            path.read_text().replace(
                'rows = ["dynamic_comparable[b1]", "dynamic_comparable[b2]", '
                '"dynamic_comparable[b3]"]',
                'rows = ["dynamic_comparable_x[b1]", "dynamic_comparable_x[b2]", '
                '"dynamic_comparable_x[b3]"]',
            )
        )
    with pytest.raises(SystemExit) as e:
        perf.load_hsosp_regions(path)
    if message:
        assert message in str(e.value)


def test_a_gate_of_the_wrong_kind_is_refused(tmp_path: Path) -> None:
    old = 'port_report = ["load_ms", "dynamic_ms"]\ngate = "end_to_end"'
    path = _write(tmp_path, old, old.replace('gate = "end_to_end"', 'gate = "compute"'))
    with pytest.raises(SystemExit, match="HD3 decided 'end_to_end'"):
        perf.load_hsosp_regions(path)


def test_every_earlier_map_reads_as_before() -> None:
    # sssp.toml, mosp.toml, hypergraph.toml and triad_count.toml load with the new checks.
    for backend in ("openmp", "cuda"):
        assert perf.load_regions(backend)
        assert perf.load_regions(backend, perf.MOSP_REGION_MAP, ("compute", "end_to_end"))
    assert perf.load_regions("escher_gpu", perf.HYPERGRAPH_REGION_MAP, ("maintain",))
    triad = perf.load_regions(
        "escher_gpu", perf.TRIAD_REGION_MAP, ("update", "full_count", "end_to_end")
    )
    assert perf.region_rows({r["name"]: r for r in triad}["update"]) == [
        "update[b1]",
        "update[b2]",
        "update[b3]",
    ]


def test_bs_outputs_parse_and_sum() -> None:
    rows = perf.parse_hsosp_csv("\n".join([CSV_HEADER, *CSV_ROWS]) + "\n")
    e2e = perf.parse_hsosp_e2e(E2E)
    assert e2e == {"load": 21345.0, "dynamic": 307.0, "static": 61.0, "wall": 23456.0, "batches": 2}
    regions = _regions()
    assert perf.hsosp_original_value(rows, e2e, regions["maintain"], 1) == pytest.approx(50.75)
    assert perf.hsosp_original_value(rows, e2e, regions["maintain"], 2) == pytest.approx(52.75)
    assert perf.hsosp_original_value(rows, e2e, regions["maintain"], None) == pytest.approx(103.5)
    assert perf.hsosp_original_value(rows, e2e, regions["dynamic_comparable"], 2) == pytest.approx(
        73.75
    )
    assert perf.hsosp_original_value(rows, e2e, regions["dynamic"], 1) == pytest.approx(151.25)
    assert perf.hsosp_original_value(rows, e2e, regions["load_plus_dynamic"], None) == 21652.0
    with pytest.raises(SystemExit, match="one \\[e2e\\] line"):
        perf.parse_hsosp_e2e(E2E + E2E)
    with pytest.raises(SystemExit, match="no column"):
        perf.parse_hsosp_csv("a,b\n1,2\n")


# The leaf stages of one hyper_sssp::update on the port (cpp/src/hypergraph/hyper_update.cpp and the
# engine's stages): Step 0 runs under hyper_sssp.normalize BEFORE and outside hypergraph.apply, the
# commit holds hypergraph.apply, then the engine's stages. A plain apply (the maintain-only reading)
# emits hypergraph.apply with hypergraph.apply.normalize inside it.
UPDATE_STAGES = (
    "hyper_sssp.normalize",
    "hypergraph.apply",
    "hyper_sssp.identify_affected",
    "hyper_sssp.seed",
    "hyper_sssp.loop",
    "hyper_sssp.recompute",
    "hyper_sssp.finalize",
)


def test_every_stage_of_an_update_moves_exactly_one_gated_row_by_stages() -> None:
    """A planted delay in each stage of an update (M11a review: Step 0 was outside maintain) moves
    maintain or sosp_update, exactly one of them, by exactly the delay; dynamic_comparable and
    dynamic read the whole update (the driver's timer) and move with every stage."""
    regions = _regions()
    ports = {name: regions[name]["port"] for name in ("maintain", "sosp_update")}
    for stage in UPDATE_STAGES:
        moved = [
            name
            for name, port in ports.items()
            if sum(1.0 for n in port if n == stage) == 1.0
        ]
        assert len(moved) == 1, (stage, moved)
    # Nothing is counted twice: the two rows name disjoint stages, and a plain apply's Step 0
    # (hypergraph.apply.normalize) is inside hypergraph.apply, not named again by maintain.
    assert not set(ports["maintain"]) & set(ports["sosp_update"])
    assert "hypergraph.apply.normalize" not in ports["maintain"]
    # The reported delta row holds both spellings of Step 0 (at most one is emitted per round).
    assert {"hyper_sssp.normalize", "hypergraph.apply.normalize"} <= set(regions["delta"]["port"])
