# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/hsosp_perf.py probe (M11a step 1; ADR 0080 section 3, HD4): the audit
thresholds from B's quiet maximum, the probe's verdict, the environment of B's runs, the load
sampler, and the refusal outside an exclusive-lock window."""

from __future__ import annotations

import importlib.util
import sys
import time
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


perf = _load("hsosp_perf")


def test_the_audit_keeps_adr_0056s_thresholds_on_a_quiet_b() -> None:
    for quiet in (None, 2.5, 8.0):
        rule = perf.audit_rule(quiet)
        assert (rule["load_average"], rule["procs_blocked"]) == (16.0, 4)


def test_the_audit_names_where_its_quiet_maximum_comes_from() -> None:
    """A missing quiet maximum is said so (never 'None is at most 8'); run reads it from step 1's
    pre-read record by default (M11a review: a hard-coded 7.03 named a file that did not exist)."""
    assert "no quiet maximum" in perf.audit_rule(None)["why"]
    assert "None" not in perf.audit_rule(None)["why"]
    value, source = perf.quiet_max_of()
    assert 0 < value <= 8.0 and source.endswith("M11a-hsosp-preread.json")
    rule = perf.audit_rule(value, source)
    assert rule["quiet_max"] == value and rule["quiet_max_source"] == source
    assert (rule["load_average"], rule["procs_blocked"]) == (16.0, 4)


def test_the_audit_doubles_bs_quiet_maximum_above_8() -> None:
    rule = perf.audit_rule(9.25)
    assert rule["load_average"] == 18.5
    assert rule["procs_blocked"] == 4
    assert "2x" in rule["why"]


def _round(locked: bool | None, reasons: list[str]) -> dict:
    return {"window": {"gpu": {"clocks_locked": locked}, "reasons": reasons}}


def test_the_probe_holds_only_when_every_round_held() -> None:
    assert perf.probe_holds([_round(True, []), _round(True, []), _round(True, [])])
    assert not perf.probe_holds([_round(True, []), _round(False, []), _round(True, [])])
    assert not perf.probe_holds([_round(True, []), _round(True, ["foreign CPU load"])])
    assert not perf.probe_holds([_round(None, [])])
    assert not perf.probe_holds([])


def test_bs_environment_has_no_omp_and_eager_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("DYNG_HSOSP_DUMP_DIR", "/tmp/x")
    env = perf.env_for(0)
    assert "OMP_NUM_THREADS" not in env and "DYNG_HSOSP_DUMP_DIR" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_MODULE_LOADING"] == "EAGER"


def test_the_sampler_reads_the_load_average_and_procs_blocked() -> None:
    t0 = time.time()
    with perf.LoadSampler(period=0.01) as s:
        time.sleep(0.5)  # at least one sample even on a loaded machine (the first is at start)
    t1 = time.time()
    got = s.between(t0, t1)
    assert got["samples"] >= 1
    assert got["load_average_max"] >= 0.0
    assert got["procs_blocked_max"] >= 0


def test_the_probe_refuses_to_run_outside_an_exclusive_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("DYNG_PERF_LOCK_HELD", raising=False)
    assert perf.main(["probe"]) == 2
    assert "exclusive-lock window" in capsys.readouterr().err


# --- run (M11a step 3): the per-round guards, the rows, the verdicts ---------------------------

GOLDEN_CASE = {
    "sessions": [
        {
            "apply": [
                {
                    "h2h_pair_count": 100 + k,
                    "pairs.sha256": f"{k:064x}",
                    "pairs.deleted": 3,
                    "pairs.inserted": 4,
                    "counters": {"bt.csr_rebuilt": 1 if k == 1 else 0},
                }
                for k in range(3)
            ]
        }
    ]
}


def _port_out(
    golden: list[dict],
    sha_of: dict | None = None,
    full: bool = False,
    plant: dict | None = None,
) -> str:
    """The port's batch lines; `full` adds the result group of a full reading (the golden's
    digests, fallback 0, mismatch_static 0), `plant` = {batch: {field: value}} plants a value of
    it (None drops the whole group of that batch)."""
    sha_of, plant = sha_of or {}, plant or {}
    lines = []
    for k, g in enumerate(golden, 1):
        line = (
            f"[batch {k}] pairs {sha_of.get(k, g['pairs_sha256'])} deleted {g['pairs_deleted']} "
            f"inserted {g['pairs_inserted']} h2h_pairs {g['h2h_pairs']} "
            f"csr_rebuilds {g['overflow_rebuilds']}"
        )
        if full and plant.get(k, {}) is not None:
            r = {
                "distances": g.get("distances_sha256") or "a" * 64,
                "parents": g.get("parents_sha256") or "b" * 64,
                "fallback": 0,
                "mismatch_static": 0,
                **plant.get(k, {}),
            }
            line += (
                f" distances {r['distances']} parents {r['parents']} fallback {r['fallback']} "
                f"mismatch_static {r['mismatch_static']}"
            )
        lines.append(line)
    return "\n".join(lines)


def _b_rows(golden: list[dict]) -> list[dict]:
    return [
        {
            "h2h_pairs": str(g["h2h_pairs"]),
            "overflow_rebuilds": str(g["overflow_rebuilds"]),
            "mismatch_static": "0",
            "t_delta_ms": "10.0",
            "t_csr_apply_ms": "2.0",
        }
        for g in golden
    ]


def test_the_golden_batches_count_the_rebuilds_so_far() -> None:
    golden = perf.golden_batches(GOLDEN_CASE)
    assert [g["overflow_rebuilds"] for g in golden] == [0, 1, 1]
    assert [g["h2h_pairs"] for g in golden] == [100, 101, 102]


def test_the_guards_pass_on_the_goldens_values() -> None:
    golden = perf.golden_batches(GOLDEN_CASE)
    assert perf.guard_original(_b_rows(golden), golden) == []
    assert perf.guard_port(_port_out(golden), golden, maintain_only=True) == []
    full = [dict(g, distances_sha256="a" * 64, parents_sha256="b" * 64) for g in golden]
    assert perf.guard_port(_port_out(full, full=True), full, maintain_only=False) == []


def test_a_full_readings_guard_needs_and_checks_the_result_of_every_batch() -> None:
    """In a full reading a wrong distances or parents digest, a nonzero mismatch_static and a
    batch line without the result group each fail the guard (M11a review: the group was
    optional and a line without it skipped the digest checks and HD7's decision check)."""
    golden = [
        dict(g, distances_sha256="a" * 64, parents_sha256="b" * 64)
        for g in perf.golden_batches(GOLDEN_CASE)
    ]
    for plant in (
        {2: {"distances": "c" * 64}},
        {1: {"parents": "c" * 64}},
        {3: {"mismatch_static": 2}},
        {2: None},
    ):
        out = _port_out(golden, full=True, plant=plant)
        assert perf.guard_port(out, golden, maintain_only=False), plant
    # The maintain-only reading has no result group, and needs none.
    assert perf.guard_port(_port_out(golden), golden, maintain_only=True) == []
    assert perf.guard_port(_port_out(golden), golden, maintain_only=False)


def test_a_planted_failing_guard_is_caught() -> None:
    """Every guard can fail: a wrong pairs digest of the port, a wrong pair count or rebuild
    count of B, a nonzero mismatch_static, a missing batch."""
    golden = perf.golden_batches(GOLDEN_CASE)
    assert perf.guard_port(_port_out(golden, {2: "f" * 64}), golden, maintain_only=True)
    assert perf.guard_port(_port_out(golden[:2]), golden, maintain_only=True)
    rows = _b_rows(golden)
    rows[0]["h2h_pairs"] = "99"
    assert perf.guard_original(rows, golden)
    rows = _b_rows(golden)
    rows[2]["overflow_rebuilds"] = "2"
    assert perf.guard_original(rows, golden)
    rows = _b_rows(golden)
    rows[1]["mismatch_static"] = "1"
    assert perf.guard_original(rows, golden)


def test_a_failed_guard_stops_the_reading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run_config raises (no record) on the first round whose guard fails: here the port's digest of
    batch 2 is planted wrong."""
    golden = perf.golden_batches(GOLDEN_CASE)
    monkeypatch.setattr(perf.hg, "check_dataset", lambda scratch, name: tmp_path / "x.hg")
    monkeypatch.setattr(perf, "golden_case", lambda scratch, name: (GOLDEN_CASE, tmp_path / "d"))

    class Monitor:
        def __init__(self, *a, **k) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            pass

        def run(self, cmd, env):
            if "bench" in cmd:
                (tmp_path / perf.TIMING_NAME).write_text('{"stages": []}')
                return _port_out(golden, {2: "e" * 64}), {}
            header = "h2h_pairs,overflow_rebuilds,mismatch_static,t_delta_ms,t_csr_apply_ms"
            rows = [f"{g['h2h_pairs']},{g['overflow_rebuilds']},0,10.0,2.0" for g in golden]
            (tmp_path / perf.CSV_NAME).write_text(header + "\n" + "\n".join(rows) + "\n")
            return "", {}

    monkeypatch.setattr(perf.perf_ab, "MachineMonitor", Monitor)
    monkeypatch.setattr(perf, "original_rows", lambda text: _b_rows(golden))

    class Clocks:
        pid = None
        locked = None

    class Args:
        gpu = 1
        rounds = 3
        max_foreign_cpu = 2.0
        scratch = tmp_path
        quiet_max = 7.03
        maintain_only = True

    with perf.LoadSampler() as sampler, pytest.raises(SystemExit, match="the guard failed"):
        perf.run_config(
            Args(), "dblp_hyperedge_50000", "dblp", "hyperedge", 50000, Path("port"), {},
            Clocks(), sampler, tmp_path, Path("b"),
        )  # fmt: skip


def test_the_early_verdicts_of_criterion_12() -> None:
    def rows(port_ms: float) -> list[dict]:
        timing = {("hypergraph.apply", k): port_ms for k in (1, 2, 3)}
        b = [[{"t_delta_ms": 10.0, "t_csr_apply_ms": 0.0}] * 3]
        return [r for r in perf.maintain_rows(b, [timing], 3) if r["row"].startswith("maintain")]

    assert {r["verdict"] for r in rows(10.3)} == {"GO"}
    assert {r["verdict"] for r in rows(10.4)} == {"recorded, fix plan"}
    assert {r["verdict"] for r in rows(10.6)} == {"above 1.05: fix"}


def test_run_refuses_outside_a_window_and_a_dry_run_on_gpu_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DYNG_PERF_LOCK_HELD", raising=False)
    argv = ["run", "--configs", "dblp_hyperedge_50000", "--maintain-only"]
    assert perf.main(argv) == 2
    assert perf.main([*argv, "--dry-run", "--gpu", "0"]) == 2


def test_the_launch_record_names_a_kernel_by_its_identifier() -> None:
    cub = (
        "void cub::CUB_300104_SM_860::detail::radix_sort::DeviceRadixSortOnesweepKernel<cub::"
        "CUB_300104_SM_860::detail::radix::policy_hub<unsigned long long, int, unsigned int>::"
        "Policy1000, (bool)0, unsigned long long>(T1 *, int)"
    )
    assert perf.kernel_label(cub) == "DeviceRadixSortOnesweepKernel"
    assert perf.kernel_label("dyng::detail::hsosp::<unnamed>::push_kernel(const int *, int)") == (
        "push_kernel"
    )
    assert perf.kernel_label("escher_mosp::hsosp::<unnamed>::pushKernel(const int *)") == (
        "pushKernel"
    )
    assert perf.kernel_label("allocateSpace(CBSTNode *, int)") == "allocateSpace"


def test_the_launch_record_counts_only_launches_copies_memsets_and_syncs() -> None:
    kept = [n for n in ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaMemset",
                        "cudaStreamSynchronize", "cudaEventSynchronize", "cudaMalloc",
                        "cudaFree", "cuLibraryLoadData") if perf.LAUNCH_API.search(n)]  # fmt: skip
    assert kept == [
        "cudaLaunchKernel",
        "cudaMemcpyAsync",
        "cudaMemset",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
    ]


def test_launches_is_a_subcommand() -> None:
    with pytest.raises(SystemExit) as exc:
        perf.main(["launches", "--help"])
    assert exc.value.code == 0


# --- memory and kernels (M11a step 5; criteria 18 and 19, HD6) --------------------------------


def test_bs_graph_scope_is_its_device_bytes_and_two_states() -> None:
    # HsospState::deviceBytes(): 68 bytes a node, 8 + 64 bytes besides (kUpdCounters = 8).
    assert perf.b_state_bytes(0) == 72
    assert perf.b_state_bytes(10) == 10 * 68 + 72
    case = {
        "sessions": [
            {
                "caps": {"caps.max_hyperedges": 100},
                "device_build": {"dev.device_bytes": 5000},
                "apply": [
                    {"device": {"dev.device_bytes": 5000}},
                    {"device": {"dev.device_bytes": 7000}},
                ],
            }
        ]
    }
    states = 2 * (100 * 68 + 72)
    assert perf.b_graph_points(case) == [5000 + states, 5000 + states, 7000 + states]


def test_bs_printed_megabytes_are_the_streams_default_format() -> None:
    # std::ostream << double with precision 6, as hsospBench writes graph_mb.
    assert perf.printed_mb(2615957924) == "2494.77"
    assert perf.printed_mb(3 * 2**20) == "3"
    assert perf.printed_mb(2**20 // 2) == "0.5"


def test_the_ports_graph_scope_lines() -> None:
    err = (
        "[load] x\n[memory] load graph_scope_bytes 100\n[memory] batch 1 graph_scope_bytes 110\n"
        "noise\n[memory] batch 2 graph_scope_bytes 120\n"
    )
    assert perf.port_points(err) == [100, 110, 120]
    assert perf.port_points("[memory] batch 2 graph_scope_bytes 5\n") == [None, None, 5]


def test_a_foreign_gpu_process_is_found_and_the_readings_own_are_not() -> None:
    import os

    assert perf.own_process(os.getpid(), "python3")
    assert not perf.own_process(1, "init")
    # a process that has gone is the reading's only if it was seen alive as its descendant,
    # never by its name (other agents run the same executables on GPU 1)
    assert not perf.own_process(2**22 + 12345, "/usr/bin/hsospBench")
    assert not perf.own_process(2**22 + 12345, "other")
    perf.OWN_SEEN.add(2**22 + 12346)
    try:
        assert perf.own_process(2**22 + 12346, "/usr/bin/hsospBench")
    finally:
        perf.OWN_SEEN.discard(2**22 + 12346)
    samples = [
        {"time": "t0", "apps": [{"pid": 7, "name": "hsospBench", "own": True}]},
        {"time": "t1", "apps": [{"pid": 8, "name": "python", "own": False}]},
    ]
    assert perf.foreign_in(samples) == [{"time": "t1", "pid": 8, "name": "python", "own": False}]


def test_the_kernel_table_splits_the_launches_at_every_updates_first_kernel() -> None:
    port, original = perf.kernel_names(0), perf.kernel_names(1)
    assert port["pack_state_kernel"] == "pack_state" and original["packStateKernel"] == "pack_state"
    assert "zeroDegreeKernel" in original and "zero_degree" not in port.values()
    launches = [{"label": x} for x in ("fill_ll", "pull_distances_warp", "pack_state", "push",
                                        "pack_state", "jump", "push")]  # fmt: skip
    assert [[x["label"] for x in b] for b in perf.split_batches(launches)] == [
        ["pack_state", "push"],
        ["pack_state", "jump", "push"],
    ]


def test_a_full_readings_rows_have_the_suites_region_form() -> None:
    rows = [
        {"row": "maintain[b1]", "gate": "compute", "original_ms": 20.0, "port_ms": 10.0,
         "ratio": 0.5, "bar": 1.05, "within": True, "original_rounds": [20.0],
         "port_rounds": [10.0]},
        {"row": "escher[b1]", "gate": "none", "original_ms": 2.0, "port_ms": 1.0, "ratio": 0.5,
         "bar": None, "within": None},
    ]  # fmt: skip
    regions = perf.bench_regions(rows)
    assert regions[0]["region"] == "maintain[b1]" and regions[0]["gate"] == 1.05
    assert regions[0]["within_gate"] is True and regions[0]["original_runs"] == [20.0]
    assert "gate" not in regions[1] and regions[1]["gate_kind"] == "none"


def test_memory_and_kernels_are_subcommands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYNG_PERF_LOCK_HELD", raising=False)
    # kernels on the timing GPU refuses outside an exclusive window
    assert perf.main(["kernels", "--gpu", "0"]) == 2
    assert perf.MEMORY_CONFIGS == (
        "dblp_hyperedge_50000",
        "dblp_vertex_50000",
        "geology_hyperedge_50000",
    )


def test_ab_reads_the_region_maps_update_rows() -> None:
    """M11b criterion 3 (ADR 0024 shared vs as-is): `ab`'s sosp_update and dynamic_comparable rows
    sum the region map's port stages, and its default sides are the two CUDA engines."""
    import tomllib

    doc = tomllib.loads((REPO / "parity" / "timed_regions" / "hyper_sssp.toml").read_text())
    regions = {r["name"]: r for r in doc["reference"]["mosp_escher"]["region"]}
    rows = dict(perf.AB_ROWS)
    assert rows["sosp_update"] == regions["sosp_update"]["port"]
    assert rows["dynamic_comparable"] == regions["dynamic_comparable"]["port"]
    assert rows["static"] == regions["static"]["port"]
