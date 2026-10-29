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
    with perf.LoadSampler(period=0.05) as s:
        t0 = time.time()
        time.sleep(0.3)
        t1 = time.time()
    got = s.between(t0, t1)
    assert got["samples"] >= 2
    assert got["load_average_max"] >= 0.0
    assert got["procs_blocked_max"] >= 0


def test_the_probe_refuses_to_run_outside_an_exclusive_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("DYNG_PERF_LOCK_HELD", raising=False)
    assert perf.main(["probe"]) == 2
    assert "exclusive-lock window" in capsys.readouterr().err
