# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests of the parity harness scripts (run in lint.yml and ci/check.sh; no goldens, no
builds, no originals needed): every script imports and parses its options, compare.py refuses
selections that would compare nothing, and the pure helpers behave."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "parity/compare.py",
    "parity/export_goldens.py",
    "parity/perf_ab.py",
    "parity/cycle_count_goldens.py",
]
# The cycle_count entry points behind the sssp scripts' first argument.
CYCLE_COUNT = [
    ["parity/compare.py", "cycle_count"],
    ["parity/export_goldens.py", "cycle_count"],
    ["parity/perf_ab.py", "cycle_count", "run"],
]


def load(rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script: str) -> None:
    proc = subprocess.run([sys.executable, REPO / script, "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout


def test_compare_skips_without_goldens(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            REPO / "parity/compare.py",
            "--goldens",
            tmp_path,
            "--exe",
            sys.executable,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 77, proc.stdout + proc.stderr


def test_compare_rejects_empty_selections(tmp_path: Path) -> None:
    # A fake corpus with a manifest, so the selection is checked (--skip-verify: not hashed).
    (tmp_path / "MANIFEST.sha256").write_text("")
    base = [sys.executable, REPO / "parity/compare.py", "--goldens", tmp_path, "--skip-verify"]
    unknown = subprocess.run(
        [*base, "--exe", sys.executable, "--groups", "sosp_typo"], capture_output=True, text=True
    )
    assert unknown.returncode == 2 and "unknown group" in unknown.stderr
    empty = subprocess.run(
        [*base, "--exe", sys.executable, "--configs", ","], capture_output=True, text=True
    )
    assert empty.returncode == 2 and "no configuration" in empty.stderr
    for config in ["gpu:2", "cuda:x", "openmp:four"]:
        bad = subprocess.run(
            [*base, "--exe", sys.executable, "--configs", config], capture_output=True, text=True
        )
        assert bad.returncode == 2 and "is not sequential" in bad.stderr, config


def test_perf_ab_regions_of_both_backends() -> None:
    perf = load("parity/perf_ab.py")
    for backend in ["openmp", "cuda"]:
        regions = perf.load_regions(backend)
        names = [r["name"] for r in regions]
        assert {"sosp_update", "apply", "end_to_end"} <= set(names), backend
        assert set(perf.report_keys(regions)) <= set(perf.REPORT)
    cuda = {r["name"]: r for r in perf.load_regions("cuda")}
    assert cuda["sosp_update"]["port"] == ["sssp.enact_fused"]
    assert perf.report_keys(list(cuda.values())) == [
        "apply batch",
        "upload",
        "end_to_end_ms",
        "comb combined graph + SOSP",
    ]
    log = (
        "host   context 80.1 ms, read inputs 1.0 ms, canonicalize 0.0 ms, apply batch 12.5 ms, "
        "upload 30.0 ms, download 4.0 ms, write 0.0 ms\n"
        "obj0   SOSP update 7.900 ms (invalidated 12, jump rounds 3, iterations 4, epochs 1, "
        "pushes 9)\n"
        "comb   combined graph + SOSP 3.000 ms (1 edges, L=1, delta 1, iterations 1, pushes 1)\n"
        "RESULT gpu_compute_ms=10.900 end_to_end_ms=200.000\n"
    )
    parsed = perf.parse_original(log, 1, perf.report_keys(list(cuda.values())))
    assert parsed["objectives"] == [7.9] and parsed["invalidated"] == [12]
    assert parsed["report"]["upload"] == 30.0
    assert perf.original_value(parsed, cuda["apply"], None) == 42.5
    assert perf.original_value(parsed, cuda["end_to_end"], None) == 197.0


def test_portable_path_hides_personal_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = load("parity/compare.py")
    assert compare.portable_path(REPO / "build" / "dev") == "build/dev"
    monkeypatch.setenv("DYNG_SCRATCH", "/data/work")
    assert compare.portable_path("/data/work/ref/X@1/patched") == "$DYNG_SCRATCH/ref/X@1/patched"
    host = compare.host_info()
    assert "machine" not in host and host["logical_cpus"]


def test_perturbed_trees_are_valid_and_seeded(tmp_path: Path) -> None:
    exporter = load("parity/export_goldens.py")
    # 0 -> 1 (1), 0 -> 2 (1), 1 -> 3 (1), 2 -> 3 (1): vertex 3 has two tight parents.
    prefix = tmp_path / "graphCsr"
    Path(f"{prefix}RowPtr.txt").write_text("0\n2\n3\n4\n4\n")
    Path(f"{prefix}ColInd.txt").write_text("1\n2\n3\n3\n")
    Path(f"{prefix}Values.txt").write_text("1\n1\n1\n1\n")
    canonical = tmp_path / "canonical" / "obj0"
    canonical.mkdir(parents=True)
    (canonical / "distancesOriginal.txt").write_text("0 0\n1 1\n2 1\n3 2\n")
    (canonical / "SSSPTreeOriginal.txt").write_text("0 -1\n1 0\n2 0\n3 1\n")
    outs = []
    for name in ["a", "b", "c", "d", "e", "f"]:
        out = tmp_path / name
        changed = exporter.perturb_initial_trees(prefix, tmp_path / "canonical", out, 1, name)
        tree = (out / "obj0" / "SSSPTreeOriginal.txt").read_text()
        assert tree.startswith("0 -1\n1 0\n2 0\n3 ")  # only the tie of vertex 3 may change
        assert tree.splitlines()[3] in ("3 1", "3 2")
        assert changed == [0 if tree.endswith("3 1\n") else 1]
        again = exporter.perturb_initial_trees(prefix, tmp_path / "canonical", out, 1, name)
        assert again == changed  # seeded by the case name
        outs.append(tree)
    assert len(set(outs)) == 2  # both tight parents occur


def _lock_script(lock: Path) -> str:
    return (
        f"import sys; sys.path.insert(0, {str(REPO / 'parity')!r}); import perf_ab\n"
        "from pathlib import Path\n"
        f"with perf_ab.perf_lock(Path({str(lock)!r}), timeout=2): print('ran')\n"
    )


def test_perf_lock_runs_under_flock1(tmp_path: Path) -> None:
    # The machine's convention: `flock perf.lock <command>`; the script must not wait for the
    # lock its own ancestor holds (it used to hang forever).
    lock = tmp_path / "perf.lock"
    proc = subprocess.run(
        ["flock", lock, sys.executable, "-c", _lock_script(lock)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0 and "ran" in proc.stdout, proc.stdout + proc.stderr
    assert "ancestor" in proc.stdout


def test_perf_lock_times_out_on_another_holder(tmp_path: Path) -> None:
    lock = tmp_path / "perf.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", lock, "sleep", "30"])
    try:
        for _ in range(100):  # wait until the other process holds the lock
            probe = subprocess.run(["flock", "-n", lock, "true"])
            if probe.returncode != 0:
                break
        proc = subprocess.run(
            [sys.executable, "-c", _lock_script(lock)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode != 0 and "still locked" in proc.stderr, proc.stdout + proc.stderr
        held = subprocess.run(
            [sys.executable, "-c", _lock_script(lock)],
            capture_output=True,
            text=True,
            timeout=60,
            env={"DYNG_PERF_LOCK_HELD": "1", "PATH": "/usr/bin:/bin"},
        )
        assert held.returncode == 0 and "ran" in held.stdout
    finally:
        holder.kill()
        holder.wait()


def test_region_map_loads() -> None:
    perf = load("parity/perf_ab.py")
    regions = perf.load_regions()
    names = [r["name"] for r in regions]
    assert names[:2] == ["sosp_update", "sosp_total"] and "end_to_end" in names
    by_name = {r["name"]: r for r in regions}
    assert by_name["sosp_update"]["gate"] == "compute" and by_name["sosp_total"]["gate"] == "none"
    assert by_name["end_to_end"]["original_report_subtract"] == ["comb combined graph + SOSP"]


# --- cycle_count (CycleEnumeration-GPU) -----------------------------------------------------------


@pytest.mark.parametrize("command", CYCLE_COUNT)
def test_cycle_count_help(command: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, REPO / command[0], *command[1:], "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout and "cycle_count" in proc.stdout


def test_cycle_count_compare_skips_without_goldens(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            REPO / "parity/compare.py",
            "cycle_count",
            "--goldens",
            tmp_path,
            "--exe",
            sys.executable,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 77, proc.stdout + proc.stderr


def test_cycle_count_cases_and_histograms() -> None:
    g = load("parity/cycle_count_goldens.py")
    cases = g.all_cases()
    names = [c.rel for c in cases]
    assert len(set(names)) == len(names)
    for want in ["count/DD_k7", "count/collab_k3", "update/twitch_k4_50000_50000_s1"]:
        assert want in names
    assert "update/DD_k4_25000_25000_s1_w10000" in names
    assert [c.heavy for c in cases if c.graph == "collab"] == [True]
    upd = g.select(cases, "DD_k4_1000_1000_s1")[0]
    assert upd.cli_args(Path("/d"))[-6:] == [
        "--deletes",
        "1000",
        "--inserts",
        "1000",
        "--batch-seed",
        "1",
    ]
    with pytest.raises(SystemExit):
        g.select(cases, "DD_k99")
    text = "# cycle_size, num_of_cycles\n2, 5\n3, 7\nTotal, 12\n"
    assert g.parse_histogram(text) == {2: 5, 3: 7}
    with pytest.raises(ValueError):
        g.parse_histogram("2, 5\nTotal, 6\n")
    assert g.delta_csv({2: 5, 3: 7}, {2: 4, 3: 9, 4: 1}, 4) == (
        "# cycle_size, delta\n2, -1\n3, 2\n4, 1\nTotal, 2\n"
    )
    # Every plan total belongs to a count case.
    assert {(c.graph, c.k) for c in cases if not c.update} == set(g.PLAN_TOTALS)


def test_cycle_count_section_survives_both_writers(tmp_path: Path, monkeypatch) -> None:
    g = load("parity/cycle_count_goldens.py")
    toml = tmp_path / "parity" / "goldens.toml"
    toml.parent.mkdir()
    toml.write_text("schema = 1\n\n[sets.sssp]\nnum_cases = 0\n\n[sets.sssp.cases]\n")
    g.write_toml("[sets.cycle_count]\nnum_cases = 1\n", toml)
    g.write_toml("[sets.cycle_count]\nnum_cases = 2\n", toml)  # replaced, not appended
    assert toml.read_text().count("[sets.cycle_count]") == 1
    exporter = load("parity/export_goldens.py")
    monkeypatch.setattr(exporter, "REPO", tmp_path)
    exporter.write_toml(tmp_path, [], "0" * 64, 0)  # the sssp writer keeps the other set
    import tomllib

    doc = tomllib.loads(toml.read_text())
    assert doc["sets"]["cycle_count"]["num_cases"] == 2 and "sssp" in doc["sets"]


def test_cycle_count_region_map_loads() -> None:
    perf = load("parity/cycle_count_perf.py")
    regions = {r["name"]: r for r in perf.load_regions()}
    assert regions["static_end_to_end"]["gate"] == "compute"
    assert regions["update"]["original_report"] == ["update_seconds"]
    assert regions["update"]["port_report"] == ["update_ms"]
    assert regions["update_end_to_end"]["gate"] == "end_to_end"
    assert perf.parse_original(12.0, "deletions=1 insertions=1\nupdate_seconds=0.0221\n") == {
        perf.WALL: 12.0,
        "update_seconds": pytest.approx(22.1),
    }
    samples = {
        "original": [{perf.WALL: 100.0, "update_seconds": 20.0}] * 5,
        "port": [{perf.WALL: 104.0, "update_ms": 21.5, "compute_ms": 1.0}] * 5,
        "stages": [{"cycle_count.update": 21.4}] * 5,
    }
    out = {e["region"]: e for e in perf.summarize(list(regions.values()), "update", samples, 5)}
    assert out["update"]["ratio"] == pytest.approx(1.075) and not out["update"]["within_gate"]
    assert out["update"]["gate"] == 1.05 and out["update"]["port_stage_ms"] == 21.4
    assert out["update_end_to_end"]["within_gate"] and out["update_end_to_end"]["gate"] == 1.10


def test_cycle_count_summarize_with_a_port_baseline() -> None:
    perf = load("parity/cycle_count_perf.py")
    regions = list(perf.load_regions())
    samples = {
        "original": [{perf.WALL: 100.0, "update_ms": 30.0}] * 5,  # another dynG build
        "port": [{perf.WALL: 90.0, "update_ms": 15.0}] * 5,
        "stages": [{}] * 5,
    }
    out = {e["region"]: e for e in perf.summarize(regions, "update", samples, 5, "port")}
    assert out["update"]["original_ms"] == 30.0 and out["update"]["ratio"] == pytest.approx(0.5)


def test_contamination_monitor() -> None:
    cont = load("parity/contamination.py")
    with cont.Monitor() as m:
        subprocess.run([sys.executable, "-c", "sum(range(3000000))"], check=True)
    r = m.result
    assert r["wall_s"] > 0 and r["own_cores"] > 0 and r["foreign_cores"] >= 0
    assert r["busy_cores"] >= 0
    s = cont.summarize([{"foreign_cores": 0.1}, {"foreign_cores": 3.0}, {"foreign_cores": 0.5}])
    assert s["foreign_cores_median"] == 0.5 and s["foreign_cores_max"] == 3.0
    assert s["flagged_runs"] == 1 and s["runs"] == 3
    assert cont.summarize([])["runs"] == 0
