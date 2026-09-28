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
SCRIPTS = ["parity/compare.py", "parity/export_goldens.py", "parity/perf_ab.py"]


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


def test_compare_edge_type_configs(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.sha256").write_text("")
    base = [sys.executable, REPO / "parity/compare.py", "--goldens", tmp_path, "--skip-verify"]
    for config in ["cuda/int16", "openmp:4/", "sequential/int64x"]:
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
    assert by_name["apply"]["original_ms"] == pytest.approx(12.7)
