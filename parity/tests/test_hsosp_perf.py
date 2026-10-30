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


def _port_out(golden: list[dict], sha_of: dict | None = None) -> str:
    sha_of = sha_of or {}
    return "\n".join(
        f"[batch {k}] pairs {sha_of.get(k, g['pairs_sha256'])} deleted {g['pairs_deleted']} "
        f"inserted {g['pairs_inserted']} h2h_pairs {g['h2h_pairs']} "
        f"csr_rebuilds {g['overflow_rebuilds']}"
        for k, g in enumerate(golden, 1)
    )


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
    assert perf.guard_port(_port_out(golden), golden) == []


def test_a_planted_failing_guard_is_caught() -> None:
    """Every guard can fail: a wrong pairs digest of the port, a wrong pair count or rebuild
    count of B, a nonzero mismatch_static, a missing batch."""
    golden = perf.golden_batches(GOLDEN_CASE)
    assert perf.guard_port(_port_out(golden, {2: "f" * 64}), golden)
    assert perf.guard_port(_port_out(golden[:2]), golden)
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
