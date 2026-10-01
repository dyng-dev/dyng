# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests of the parity harness scripts (run in lint.yml and ci/check.sh; no goldens, no
builds, no originals needed): every script imports and parses its options, compare.py refuses
selections that would compare nothing, and the pure helpers behave."""

from __future__ import annotations

import importlib.util
import json
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
    # MOSP-CUDA's "upload" copies the K trees on the host first; dynG's copy is sssp.import.
    assert "sssp.import" in cuda["apply"]["port_all_results"]
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


def test_compare_edge_type_configs(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.sha256").write_text("")
    base = [sys.executable, REPO / "parity/compare.py", "--goldens", tmp_path, "--skip-verify"]
    for config in ["cuda/int16", "openmp:4/", "sequential/int64x"]:
        bad = subprocess.run(
            [*base, "--exe", sys.executable, "--configs", config], capture_output=True, text=True
        )
        assert bad.returncode == 2 and "is not sequential" in bad.stderr, config


def test_compare_cuda_engine_configs(tmp_path: Path) -> None:
    # cuda-fused and cuda-operators force one CUDA engine (--cuda-engine); other names are refused.
    compare = load("parity/compare.py")
    calls = []

    def fake_run(cmd: list, env: dict) -> tuple[int, str]:
        calls.append([str(c) for c in cmd])
        return 1, "stop"

    compare.run = fake_run
    golden = tmp_path / "case"
    for config, engine in [("cuda-operators:1", "operators"), ("cuda-fused/int64", "fused")]:
        calls.clear()
        compare.replay_compat(Path("exe"), golden, {"num_objectives": 1}, config, tmp_path, {})
        args = calls[0]
        assert args[args.index("--backend") + 1] == "cuda", config
        assert args[args.index("--cuda-engine") + 1] == engine, config
    calls.clear()
    compare.replay_compat(Path("exe"), golden, {"num_objectives": 1}, "cuda:1", tmp_path, {})
    assert "--cuda-engine" not in calls[0]
    (tmp_path / "MANIFEST.sha256").write_text("")
    base = [sys.executable, REPO / "parity/compare.py", "--goldens", tmp_path, "--skip-verify"]
    for config in ["cuda-warp", "cuda-operators:x"]:
        bad = subprocess.run(
            [*base, "--exe", sys.executable, "--configs", config], capture_output=True, text=True
        )
        assert bad.returncode == 2 and "is not sequential" in bad.stderr, config


def test_export_compare_corpora_ignores_only_the_reference(tmp_path: Path) -> None:
    export = load("parity/export_goldens.py")

    def corpus(root: Path, reference: str, tree: str) -> Path:
        case = root / "sosp" / "c0"
        (case / "updated").mkdir(parents=True)
        (case / "updated" / "tree.txt").write_text(tree)
        meta = {"case": "sosp/c0", "reference": reference, "invalidated": [3]}
        (case / "case.json").write_text(json.dumps(meta))
        export.write_manifest(root, [])
        return root

    a = corpus(tmp_path / "a", "MOSP-OpenMP@c", "0 0\n1 0\n")
    b = corpus(tmp_path / "b", "MOSP-CUDA@e", "0 0\n1 0\n")
    assert export.compare_corpora(a, b) == []
    c = corpus(tmp_path / "c", "MOSP-CUDA@e", "0 0\n1 1\n")
    assert export.compare_corpora(a, c) == ["differs: sosp/c0/updated/tree.txt"]
    meta = json.loads((c / "sosp" / "c0" / "case.json").read_text())
    meta["invalidated"] = [4]
    (c / "sosp" / "c0" / "case.json").write_text(json.dumps(meta))
    (c / "sosp" / "c0" / "updated" / "tree.txt").write_text("0 0\n1 0\n")
    export.write_manifest(c, [])
    assert export.compare_corpora(a, c) == ["differs: sosp/c0/case.json"]


def test_perf_ab_parses_ncu_csv() -> None:
    perf = load("parity/perf_ab.py")
    head = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value"\n'
    )
    rows = "".join(
        f'"{i}","1","mosp","h","sospPersistentKernel","1","7","(256, 1, 1)","(256, 1, 1)","0",'
        f'"8.6","Command line profiler metrics","{m}","ns","{v}"\n'
        for i in range(3)
        for m, v in [("gpu__time_duration.sum", f"{29 + i},141,120"), ("launch__grid_size", "256")]
    )
    kernels = perf.parse_ncu_csv("==PROF== note\n" + head + rows, 2, "mosp")
    assert [x["gpu__time_duration.sum"] for x in kernels] == [29141120.0, 30141120.0]
    assert kernels[0]["launch__grid_size"] == 256.0 and kernels[0]["name"] == "sospPersistentKernel"
    with pytest.raises(SystemExit):
        perf.parse_ncu_csv(head + rows, 4, "mosp")


def test_perf_ab_edge_type_summary_reads_the_port_on_both_sides() -> None:
    perf = load("parity/perf_ab.py")
    regions = [dict(r, gate="none") for r in perf.load_regions("cuda")]

    def sample(ms: float) -> dict:
        stages = {
            "sssp.enact_fused": [ms, ms],
            "update.commit": [10.0],
            "sssp.import": [0.3, 0.3],
            "sssp.upload": [1.0, 1.0],
            "sssp.workspace": [0.5, 0.0, 0.0, 0.0],
            "sssp.changes": [0.1, 0.1],
            "total.end_to_end": [100.0],
        }
        return {"stages": stages, "device": {}, "invalidated": [1, 2], "threads": 1}

    samples = {"original": [sample(4.0)] * 5, "port": [sample(4.2)] * 5}
    out = perf.summarize(regions, samples, 2, 5, [(1.0, 1.0)], a_value=perf.port_value)
    by_name = {e["region"]: e for e in out["regions"]}
    assert by_name["sosp_update obj0"]["ratio"] == pytest.approx(1.05)
    assert "gate" not in by_name["sosp_update obj0"]
    # update.commit + every sssp.import, sssp.upload, sssp.workspace and sssp.changes sample
    assert by_name["apply"]["original_ms"] == pytest.approx(13.3)


def test_perf_ab_run_takes_a_dyng_baseline() -> None:
    proc = subprocess.run(
        [sys.executable, REPO / "parity/perf_ab.py", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--baseline-exe" in proc.stdout and "--baseline-label" in proc.stdout
    perf = load("parity/perf_ab.py")
    lines = perf.report({}, labels=("f664f96", "dynG"))
    assert "f664f96 (ms)" in lines[0]


def test_perf_ab_layouts_cycle_the_timing_file_name(tmp_path: Path) -> None:
    perf = load("parity/perf_ab.py")
    args = type("Args", (), {"layouts": 3})()
    names = [perf.layout_timing(tmp_path, "timing", args, r).name for r in range(4)]
    assert names == ["timing.csv", "timing" + "x" * 8 + ".csv", "timing" + "x" * 16 + ".csv",
                     "timing.csv"]  # fmt: skip
    one = type("Args", (), {"layouts": 1})()
    assert perf.layout_timing(tmp_path, "timing", one, 5) == tmp_path / "timing.csv"
    # Long pads go into directories (a file name has at most 255 characters).
    many = type("Args", (), {"layouts": 100})()
    deep = perf.layout_timing(tmp_path, "timing", many, 99)
    assert len(str(deep)) - len(str(tmp_path / "timing.csv")) == 8 * 99 + 3  # three separators
    assert deep.parent.is_dir() and max(len(p) for p in deep.parts) <= 255
    # Only for a dynG-against-dynG A/B: the originals take no --timing file.
    proc = subprocess.run(
        [sys.executable, REPO / "parity/perf_ab.py", "run", "--exe", "x", "--layouts", "3"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "--baseline-exe" in proc.stderr


def test_perf_ab_monitor_sees_foreign_cpu_load() -> None:
    perf = load("parity/perf_ab.py")
    # A spinning process that is not the timed program: foreign load of about one core.
    spinner = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        with perf.MachineMonitor(None, max_foreign_cpu=0.5) as monitor:
            out, window = monitor.run([sys.executable, "-c", "import time; time.sleep(0.6)"], {})
    finally:
        spinner.kill()
        spinner.wait()
    assert out == ""
    assert window["foreign_cpu_cores"] >= 0.7 and window["contaminated"], window
    assert window["procs_running"] is not None and "gpu" not in window
    assert window["wall_s"] >= 0.6 and window["program_cpu_s"] < 0.5


def test_perf_ab_rejects_contaminated_rounds() -> None:
    perf = load("parity/perf_ab.py")

    class Args:
        keep_contaminated = False
        runs = 2

    clean = {"reasons": [], "foreign_cpu_cores": 0.1}
    busy = {"reasons": ["foreign CPU load 3.00 cores > 2.0"], "foreign_cpu_cores": 3.0}
    rejected: list = []
    assert not perf.rejects(Args(), "b", 0, {"original": clean, "port": clean}, rejected)
    perf.time.sleep = lambda _s: None  # no pause between repetitions in the test
    assert perf.rejects(Args(), "b", 0, {"original": clean, "port": busy}, rejected)
    assert rejected[0]["reasons"] == ["port: foreign CPU load 3.00 cores > 2.0"]
    perf.rejects(Args(), "b", 0, {"original": busy, "port": busy}, rejected)
    with pytest.raises(SystemExit):  # more rejections than rounds: the machine is too busy
        perf.rejects(Args(), "b", 0, {"original": busy, "port": clean}, rejected)
    rounds = [{"original": clean, "port": dict(clean, foreign_cpu_cores=0.4), "round": 1}]
    summary = perf.monitor_summary(rounds, rejected, 2.0)
    assert summary["foreign_cpu_cores_max"] == 0.4 and len(summary["rejected"]) == 3


def test_perf_ab_monitor_requires_the_locked_clocks() -> None:
    perf = load("parity/perf_ab.py")
    monitor = perf.MachineMonitor(0, locked=(1695, 7601))  # not entered: no sampling threads
    now = perf.time.time()
    # (time, P-state, SM MHz, memory MHz, utilization %); idle samples are not checked.
    monitor.gpu_samples = [
        (now + 0.1, "P2", 1695, 7601, 80),
        (now + 0.2, "P8", 210, 405, 0),
    ]
    window = monitor.window(now, now + 0.3, 0.3, 0.0, 0.0, 0.0)
    assert window["gpu"]["clocks_locked"] and not window["contaminated"], window
    monitor.gpu_samples.append((now + 0.25, "P0", 1905, 8001, 90))
    window = monitor.window(now, now + 0.3, 0.3, 0.0, 0.0, 0.0)
    assert not window["gpu"]["clocks_locked"] and window["contaminated"], window
    assert "not at the locked (1695, 7601)" in window["reasons"][0]


def test_perf_ab_monitor_allows_the_clock_holder() -> None:
    perf = load("parity/perf_ab.py")
    monitor = perf.MachineMonitor(0, allowed_pids={4242})
    assert 4242 in monitor.allowed_pids and monitor.locked is None


def test_perf_ab_clock_lock_none_does_nothing() -> None:
    perf = load("parity/perf_ab.py")
    with perf.ClockLock(0, "none") as clocks:
        assert clocks.pid is None and clocks.locked is None
    assert clocks.record["control"] == "none"
    assert perf.ClockLock.SOURCE.is_file()


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
    # The CUDA set (M2b acceptance criterion 3).
    cuda = [c.rel for c in g.cuda_cases()]
    assert len(set(cuda)) == len(cuda)
    for want in [
        "count/DD_k3",
        "count/DD_k7",
        "count/github_k4",
        "count/twitch_k4",
        "count/collab_k3",
        "update/DD_k4_100000_100000_s1",
        "update/github_k4_100000_100000_s1",
        "update/twitch_k4_25000_25000_s1",
        "update/collab_k4_25000_25000_s1",
    ]:
        assert want in cuda
    assert [c.rel for c in g.cuda_cases() if g.cuda_only(c)] == ["update/collab_k4_25000_25000_s1"]
    assert g.backend_args(g.SET_CUDA)[:2] == ["--backend", "cuda"]


def test_cycle_count_section_survives_both_writers(tmp_path: Path, monkeypatch) -> None:
    g = load("parity/cycle_count_goldens.py")
    toml = tmp_path / "parity" / "goldens.toml"
    toml.parent.mkdir()
    toml.write_text("schema = 1\n\n[sets.sssp]\nnum_cases = 0\n\n[sets.sssp.cases]\n")
    g.write_toml("[sets.cycle_count]\nnum_cases = 1\n", g.SET, toml)
    g.write_toml("[sets.cycle_count]\nnum_cases = 2\n", g.SET, toml)  # replaced, not appended
    assert toml.read_text().count("[sets.cycle_count]") == 1
    # The CUDA set (M2b) is written next to it; either writer keeps the other set.
    cuda = "[sets.cycle_count_cuda]\nnum_cases = 3\n\n[sets.cycle_count_cuda.cases]\n"
    g.write_toml(cuda, g.SET_CUDA, toml)
    g.write_toml("[sets.cycle_count]\nnum_cases = 4\n", g.SET, toml)
    g.write_toml(cuda.replace("3", "5"), g.SET_CUDA, toml)
    assert toml.read_text().count("[sets.cycle_count_cuda]") == 1
    exporter = load("parity/export_goldens.py")
    monkeypatch.setattr(exporter, "REPO", tmp_path)
    exporter.write_toml(tmp_path, [], "0" * 64, 0)  # the sssp writer keeps the other sets
    import tomllib

    doc = tomllib.loads(toml.read_text())
    assert doc["sets"]["cycle_count"]["num_cases"] == 4 and "sssp" in doc["sets"]
    assert doc["sets"]["cycle_count_cuda"]["num_cases"] == 5


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


def test_cycle_count_cuda_gpu_summary_of_a_port_baseline() -> None:
    perf = load("parity/cycle_count_perf.py")
    gpu = {"gpu": {"sm_mhz": {"min": 1695}, "busy_samples": 3}}
    windows = [
        {
            "original": {"reasons": [], "baseline": {"original": gpu, "resident": gpu}},
            "port": {"original": gpu, "resident": gpu},
        }
    ]
    out = perf.gpu_summary(windows, ["original", "resident"])
    assert set(out) == {
        "baseline[original]",
        "baseline[resident]",
        "port[original]",
        "port[resident]",
    }
    assert out["baseline[resident]"]["sm_mhz_min"] == 1695
    assert out["port[original]"]["busy_samples"] == 3


def test_cycle_count_cuda_extra_processes() -> None:
    perf = load("parity/cycle_count_perf.py")
    regions = perf.load_regions("cuda")
    main = {r["name"] for r in perf.scoped_regions(regions, "resident")}
    chain = {r["name"] for r in perf.scoped_regions(regions, "resident", "chain")}
    events = {r["name"] for r in perf.scoped_regions(regions, "original", "events")}
    assert "update[resident]" in main and "update_chain_steady[resident]" not in main
    assert chain == {"update_chain_steady[resident]", "update_chain_worst[resident]"}
    assert events == {"update_device[original]"}
    assert not perf.scoped_regions(regions, "original", "chain")  # resident only
    values: dict = {}
    err = "RESULT task=update update_ms=9.4 chain_ms=9.4,5.9,6.3,5.8 chain_match=yes\n"
    assert perf.parse_chain(values, err)
    assert values["chain_steady_ms"] == pytest.approx(5.9) and values["chain_worst_ms"] == 9.4
    assert not perf.parse_chain({}, "chain_ms=1,2 chain_match=no\n")
    samples = {
        "original": [{perf.WALL: 1.0, "update_seconds": 20.0}] * 21,
        "port": [values] * 21,
        "stages": [{}] * 21,
    }
    out = {
        e["region"]: e
        for e in perf.summarize(
            perf.scoped_regions(regions, "resident", "chain"), "update", samples, 21
        )
    }
    assert out["update_chain_worst[resident]"]["ratio"] == pytest.approx(0.47)
    assert out["update_chain_steady[resident]"]["within_gate"]


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


def test_cycle_count_optional_original_keys() -> None:
    perf = load("parity/cycle_count_perf.py")
    regions = list(perf.load_regions())
    err = "read_seconds=0.39\ncount_seconds=1.5\n"
    timed = perf.parse_original(2000.0, err)
    assert timed["count_seconds"] == pytest.approx(1500.0)
    assert timed["read_seconds"] == pytest.approx(390.0)
    port = {perf.WALL: 1500.0, "compute_ms": 1000.0, "read_ms": 380.0, "build_ms": 40.0}
    samples = {"original": [timed] * 5, "port": [port] * 5, "stages": [{}] * 5}
    out = {e["region"]: e for e in perf.summarize(regions, "count", samples, 5)}
    assert out["static_count"]["ratio"] == pytest.approx(1000.0 / 1500.0)
    assert out["static_read"]["ratio"] == pytest.approx(420.0 / 390.0)
    assert "gate" not in out["static_count"]  # never a gate
    plain = {"original": [{perf.WALL: 2000.0}] * 5, "port": [port] * 5, "stages": [{}] * 5}
    out = {e["region"]: e for e in perf.summarize(regions, "count", plain, 5)}
    assert "ratio" not in out["static_count"] and out["static_end_to_end"]["ratio"] == 0.75


def test_cycle_count_kernel_occupancy_and_names() -> None:
    perf = load("parity/cycle_count_perf.py")
    # sm_86, 128 threads (4 warps): up to 40 registers the warp limit (12 blocks) binds;
    # 48 registers: 1536 per warp, 42 warps, 10 blocks (83 %); 64: 8 blocks (67 %).
    assert perf.occupancy(40, 128)["occupancy"] == 1.0
    assert perf.occupancy(40, 128)["limited_by"] == "warps"
    got = perf.occupancy(48, 128)
    assert got["blocks_per_sm"] == 10 and got["limited_by"] == "registers"
    assert perf.occupancy(64, 128)["occupancy"] == pytest.approx(8 * 4 / 48)
    assert perf.occupancy(16, 256)["blocks_per_sm"] == 6
    name = (
        "void dyng::detail::(anonymous namespace)::count_edge_items_kernel<16, unsigned long>"
        "(dyng::detail::device_csr<unsigned long>, int)"
    )
    assert perf.kernel_key(name) == ("count_edge_items", 16, "unsigned long")
    original = (
        "void cycle_enum::cuda::detail::(anonymous namespace)::count_roots_queue_kernel<8>"
        "(cycle_enum::cuda::CsrView, int, unsigned long long*, unsigned long long*)"
    )
    assert perf.kernel_key(original) == ("count_roots_queue", 8, "unsigned int")
    assert perf.kernel_key("void x::change_rows_kernel(unsigned int)") == (
        "change_rows",
        None,
        "unsigned int",
    )
    row = {
        "base": "count_roots",
        "cap": 4,
        "fused": True,
        "registers": 26,
        "stack": 96,
        "shared": 0,
        "block_size": 128,
        "occupancy": 1.0,
    }
    doc = {
        "original": [dict(row, offsets="unsigned int")],
        "port": [dict(row, offsets="unsigned int"), dict(row, offsets="unsigned long", stack=128)],
    }
    (pair,) = perf.pair_kernels(doc)
    assert pair["kernel"] == "count_roots<4>" and pair["equal"]
    doc["port"][0]["registers"] = 28
    assert not perf.pair_kernels(doc)[0]["equal"]


def test_cycle_count_cuda_gpu_summary_keeps_both_sides() -> None:
    perf = load("parity/cycle_count_perf.py")

    def win(low: int, busy: int) -> dict:
        return {"gpu": {"sm_mhz": {"min": low}, "busy_samples": busy, "clocks_locked": True}}

    windows = [
        {"round": 1, "original": win(1695, 2), "port": {"original": win(1500, 0)}},
        {"round": 2, "original": win(1680, 0), "port": {"original": win(1695, 1)}},
    ]
    got = perf.gpu_summary(windows, ["original"])
    assert got["original"] == {
        "sm_mhz_min": 1680,
        "busy_samples": 2,
        "rounds_with_busy_samples": 1,
    }
    assert got["port[original]"]["sm_mhz_min"] == 1500


def test_cycle_count_peak_device_bytes() -> None:
    perf = load("parity/cycle_count_perf.py")
    # (start, bytes, operation 0 = allocate / 1 = free, address), out of order on purpose
    events = [(3, 0, 1, 0xA), (1, 100, 0, 0xA), (2, 50, 0, 0xB), (4, 70, 0, 0xC), (5, 0, 1, 0xB)]
    assert perf.peak_device_bytes(events) == 150
    assert perf.peak_device_bytes([]) == 0


def test_perf_ab_engine_stage_sums_whichever_engine_ran() -> None:
    perf = load("parity/perf_ab.py")
    fused = {"stages": {"sssp.enact_fused": [2.0, 3.0]}, "device": {"sssp.enact_fused": [1.5, 2.5]}}
    ops = {
        "stages": {
            "sssp.identify_affected": [1.0, 1.0],
            "sssp.seed": [0.5, 0.5],
            "sssp.loop": [2.0, 4.0],
            "sssp.finalize": [0.25, 0.25],
        },
        "device": {"sssp.loop": [1.0, 2.0]},
    }
    assert perf.with_engine_stage(fused, 2)["stages"]["sssp.engine"] == [2.0, 3.0]
    assert perf.with_engine_stage(fused, 2)["device"]["sssp.engine"] == [1.5, 2.5]
    assert perf.with_engine_stage(ops, 2)["stages"]["sssp.engine"] == [3.75, 5.75]
    assert perf.with_engine_stage(ops, 2)["device"]["sssp.engine"] == [1.0, 2.0]
