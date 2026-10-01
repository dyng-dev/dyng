# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of the benchmark-suite runner (parity/bench_suite.py) and the certificate writer
(parity/certify.py): the committed suites validate, the plan drives the harness with the suite's
protocol, the summary re-derives every gate and refuses records that do not match the suite, and
the certificate assembles and checks what it is given. No builds, GPUs or originals needed."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = sorted((REPO / "benchmarks" / "paper").glob("*.yaml"))


def load(rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return load("parity/bench_suite.py")


@pytest.fixture(scope="module")
def cert():
    return load("parity/certify.py")


def head() -> str:
    return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()


# --- bench_suite.py ------------------------------------------------------------------------------


def test_the_release_suites_exist_and_validate(bench) -> None:
    names = {p.stem for p in SUITES}
    assert {"ieee_tc_dyntrucy", "ipdps25_dynamosp_sosp"} <= names
    for path in SUITES:
        assert bench.validate(bench.load_suite(path)) == [], path


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s["baselines"]["CycleEnumeration-GPU"].update(commit="0" * 40), "pinned"),
        (lambda s: s["baselines"]["CycleEnumeration-GPU"].update(variant="patched"), "unpatched"),
        (lambda s: s["metrics"]["update"].update(gate="end_to_end"), "region map says"),
        (lambda s: s["metrics"]["update"]["openmp"].update(port=["x"]), "port stages"),
        (lambda s: s["metrics"].pop("update_end_to_end"), "is not a metric"),
        (lambda s: s["tolerance"].update(compute=1.2), "PLAN 8.6"),
        (lambda s: s["readings"][2].update(clocks="fast"), "clocks"),
        (lambda s: s["readings"][4].update(gated=True), "not gated"),
        (lambda s: s["readings"][0]["cases"].append("DD_k9"), "DD_k9"),
        (lambda s: s["readings"][0].update(runs=3), ">= 5"),
        (lambda s: s["datasets"][0].update(sha256="abc"), "SHA-256"),
        (lambda s: s.update(suite="other"), "file name"),
    ],
)
def test_validate_finds_what_differs(bench, edit, message: str) -> None:
    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    edit(suite)
    errors = bench.validate(suite)
    assert any(message in e for e in errors), errors


def test_plan_follows_the_suite(bench, tmp_path: Path) -> None:
    cc = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    jobs = bench.plan(cc, records=tmp_path, build_root=Path("/b"))
    by = {j["reading"]: j for j in jobs}
    assert set(by) == {r["name"] for r in cc["readings"]}
    cuda = by["cuda"]["argv"]
    assert cuda[cuda.index("--lock-clocks") + 1] == "boost" and "cycle_count" in cuda
    assert "/b/parity-cuda/tools/compat/dyng-compat-cycle-enum" in cuda
    base = by["cuda-collab-update"]["argv"]
    assert base[base.index("--lock-clocks") + 1] == "base"  # ADR 0021
    assert by["cuda-default-clocks"]["gated"] is False
    omp = by["openmp"]["argv"]
    assert omp[omp.index("--runs") + 1] == "11" and omp[omp.index("--threads") + 1] == "56"
    assert by["cuda-memory"]["lock"] == "shared" and "memory" in by["cuda-memory"]["argv"]

    sssp = bench.load_suite(REPO / "benchmarks/paper/ipdps25_dynamosp_sosp.yaml")
    jobs = bench.plan(sssp, records=tmp_path, build_root=Path("/b"), readings=["cuda"])
    assert [j["dataset"] for j in jobs] == ["roadNet-PA", "roadNet-CA", "rgg", "road_usa_g"]
    runs = {j["dataset"]: j["argv"][j["argv"].index("--runs") + 1] for j in jobs}
    assert runs == {"roadNet-PA": "21", "roadNet-CA": "21", "rgg": "21", "road_usa_g": "11"}
    assert all(j["argv"][j["argv"].index("--lock-clocks") + 1] == "boost" for j in jobs)
    with pytest.raises(bench.SuiteError):
        bench.plan(sssp, records=tmp_path, build_root=Path("/b"), readings=["nope"])


def _region(name: str, original: float, port: float, kind: str = "compute", gate=None) -> dict:
    ratio = port / original
    if gate is None:
        gate = 1.10 if kind == "end_to_end" or original < 10 else 1.05
    return {
        "region": name,
        "gate_kind": kind,
        "original_ms": original,
        "port_ms": port,
        "ratio": ratio,
        "gate": gate,
        "within_gate": ratio <= gate,
        "provisional": False,
        "noisy": False,
    }


def _cc_record(cases: list[str], commit: str, clocks: str | None = None, ratio: float = 0.9):
    record = {
        "schema": 1,
        "algorithm": "cycle_count",
        "date": "2026-10-01T00:00:00Z",
        "reference": {
            "name": "CycleEnumeration-GPU",
            "commit": "0a976adfa801a712135bf1adb51a228f353a0751",
            "variant": "unpatched",
        },
        "baseline": {"experiment": False, "label": "original", "kind": "original"},
        "port": {"commit": commit, "build": {"parity_preset": True}},
        "protocol": {"runs": 11},
        "datasets_sha256": {
            "DD": "e538ec13a262b84a0664db2f22e6ff720fb5bb74965536c37f4b8e962d156e4a"
        },
        "results": {},
    }
    if clocks:
        record["protocol"]["clocks"] = {"control": clocks}
    for case in cases:
        record["results"][f"count/{case}"] = {
            "regions": [_region("static_end_to_end", 100.0, 100.0 * ratio)],
            "monitor": {"rounds": [1, 2, 3], "rejected": [{"reasons": ["x"], "windows": {}}]},
        }
    return record


def _write_cc_records(bench, suite, records: Path, commit: str, **kw) -> list[dict]:
    records.mkdir(parents=True, exist_ok=True)
    jobs = bench.plan(suite, records=records, build_root=Path("/b"))
    for job in jobs:
        reading = next(r for r in suite["readings"] if r["name"] == job["reading"])
        if job["kind"] == "memory":
            rec = {
                "reference": {"name": "CycleEnumeration-GPU", "binary": "$X/unpatched/cycle-enum"},
                "port": {"commit": commit},
                "results": {
                    c: {
                        "original": {"peak_live_mib": 18.0},
                        "port[original]": {"peak_live_mib": 18.0, "ratio": 1.0},
                    }
                    for c in reading["cases"]
                },
            }
        else:
            rec = _cc_record(reading["cases"], commit, reading.get("clocks"), **kw)
        Path(job["record"]).write_text(json.dumps(rec))
    return jobs


def test_summary_passes_and_compacts(bench, tmp_path: Path) -> None:
    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    jobs = _write_cc_records(bench, suite, tmp_path / "rec", head(), ratio=0.9)
    out = tmp_path / "out"
    summary = bench.summarize(suite, jobs, out=out, version="t", records=tmp_path, inputs=None)
    assert summary["verdict"]["passed"], summary["verdict"]
    assert summary["verdict"]["gated_regions"] > 0
    record = json.loads((out / "ieee_tc_dyntrucy-cuda.json").read_text())
    monitor = record["results"]["count/DD_k4"]["monitor"]
    assert "rounds" not in monitor and "windows" not in monitor["rejected"][0]
    assert "compacted" in record
    assert (out / "ieee_tc_dyntrucy.json").is_file()


def test_summary_fails_on_exceeded_or_mismatched_records(bench, tmp_path: Path) -> None:
    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    jobs = _write_cc_records(bench, suite, tmp_path / "rec", head(), ratio=1.2)
    summary = bench.summarize(
        suite, jobs, out=tmp_path / "o1", version="t", records=tmp_path, inputs=None
    )
    assert not summary["verdict"]["passed"] and summary["verdict"]["exceeded"]

    # A record measured at other clocks than the reading's, a patched reference, other inputs.
    jobs = _write_cc_records(bench, suite, tmp_path / "rec2", head(), ratio=0.9)
    cuda = next(j for j in jobs if j["reading"] == "cuda")
    rec = json.loads(Path(cuda["record"]).read_text())
    bad = copy.deepcopy(rec)
    bad["protocol"]["clocks"] = {"control": "none"}
    bad["reference"]["variant"] = "patched"
    bad["datasets_sha256"]["DD"] = "0" * 64
    bad["results"]["count/DD_k4"]["regions"][0]["gate"] = 1.5
    Path(cuda["record"]).write_text(json.dumps(bad))
    summary = bench.summarize(
        suite, jobs, out=tmp_path / "o2", version="t", records=tmp_path, inputs=None
    )
    problems = " ".join(summary["verdict"]["problems"])
    for text in ["clocks 'none'", "not the unpatched", "SHA-256", "gated at 1.5"]:
        assert text in problems, problems
    # A missing record and a failed harness run are failures too.
    Path(jobs[0]["record"]).unlink()
    jobs[1]["rc"] = 3
    summary = bench.summarize(
        suite, jobs, out=tmp_path / "o3", version="t", records=tmp_path, inputs=None
    )
    problems = " ".join(summary["verdict"]["problems"])
    assert "missing" in problems and "exited 3" in problems


def test_verify_inputs(bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    suite["harness"]["inputs"] = str(tmp_path)
    suite["datasets"] = [dict(suite["datasets"][0])]
    (tmp_path / "DD").mkdir()
    data = tmp_path / "DD" / "DD_A.txt"
    data.write_text("1, 2\n")
    with pytest.raises(bench.SuiteError, match="SHA-256"):
        bench.verify_inputs(suite)
    suite["datasets"][0]["sha256"] = bench.sha256_file(data)
    assert bench.verify_inputs(suite) == {"DD": "all digests equal"}
    data.unlink()
    with pytest.raises(bench.SuiteError, match="missing input"):
        bench.verify_inputs(suite)


def test_bench_suite_cli(bench) -> None:
    proc = subprocess.run(
        [sys.executable, REPO / "parity/bench_suite.py", "validate", *SUITES],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = subprocess.run(
        [sys.executable, REPO / "parity/bench_suite.py", "plan", SUITES[0], "--runs", "3"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2 and ">= 5" in proc.stderr


# --- certify.py ----------------------------------------------------------------------------------


def test_parsers(cert) -> None:
    log = "...\n100% tests passed, 0 tests failed out of 596\n\nTotal Test time (real) = 1 sec\n"
    assert cert.parse_ctest(log) == {"tests": 596, "failed": 0, "percent_passed": 100}
    assert cert.parse_ctest("99% tests passed, 1 test failed out of 120")["failed"] == 1
    assert cert.parse_ctest("nothing") is None
    md = "### x\n\n| step | result |\n|---|---|\n| build | passed |\n| memcheck | FAILED |\n"
    assert cert.parse_gpu_summary(md) == {"build": "passed", "memcheck": "FAILED"}
    matrix, compared, failed = cert.replay_matrix(
        {"matrix": {"sosp": {"cuda": {"cases": 3, "pass": 2, "fail": ["a"]}}}}
    )
    assert (compared, failed) == (3, 1) and "failed" in matrix["sosp"]["cuda"]
    matrix, compared, failed = cert.replay_matrix(
        {"matrix": {"count/DD_k3": {"openmp:4": "equal"}}}
    )
    assert (compared, failed) == (1, 0)


def _replays(cert, results: Path, commit: str) -> None:
    import tomllib

    sets = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"]
    configs = {
        "sssp": [["sequential", "openmp:4"], ["cuda"]],
        "cycle_count": [["sequential", "openmp:56"]],
        "cycle_count_cuda": [["cuda", "cuda:resident"]],
    }
    for name, lists in configs.items():
        for i, cfg in enumerate(lists):
            record = {
                "algorithm": "sssp" if name == "sssp" else "cycle_count",
                "port": {"commit": commit},
                "goldens": {"manifest_sha256": sets[name]["manifest_sha256"]},
                "configs": cfg,
                "matrix": {"g": {c: "equal" for c in cfg}},
                "passed": True,
            }
            if name != "sssp":
                record["set"] = name
            (results / f"parity-{name}-{i}.json").write_text(json.dumps(record))


def test_certificate_from_results(cert, bench, tmp_path: Path) -> None:
    commit = head()
    results = tmp_path / "results"
    results.mkdir()
    _replays(cert, results, commit)
    for path in SUITES:
        suite = bench.load_suite(path)
        if suite["algorithm"] == "cycle_count":
            jobs = _write_cc_records(bench, suite, tmp_path / "rec", commit)
            bench.summarize(suite, jobs, out=results, version="t", records=tmp_path, inputs=None)
        else:  # a minimal passing summary of the sssp suite
            summary = {
                "suite": suite["suite"],
                "algorithm": "sssp",
                "port_commits": [commit],
                "readings": [
                    {
                        "reading": "cuda",
                        "backend": "cuda",
                        "clocks": "boost",
                        "gated": True,
                        "record": None,
                        "regions": [
                            {
                                "case": "rgg/safe50k",
                                "region": "apply",
                                "ratio": 0.8,
                                "gate": 1.1,
                                "within_gate": True,
                            }
                        ],
                    }
                ],
                "verdict": {
                    "passed": True,
                    "gated_regions": 1,
                    "within_gate": 1,
                    "ratio_range": [0.8, 0.8],
                },
            }
            (results / f"{suite['suite']}.json").write_text(json.dumps(summary))
    log = tmp_path / "asan.log"
    log.write_text("100% tests passed, 0 tests failed out of 12\n")
    args = [
        "check",
        "--results",
        str(results),
        "--name",
        "asan",
        "--command",
        "ctest --preset asan",
    ]
    assert cert.main([*args, "--ctest-log", str(log)]) == 0
    assert (
        cert.main(["write", "--results", str(results), "--version", "t", "--allow-dirty"]) == 0
    ), json.loads((results / "parity.json").read_text())["verdict"]
    doc = json.loads((results / "parity.json").read_text())
    assert doc["verdict"]["passed"] and doc["commit"].startswith(commit)
    assert {g["set"] for g in doc["golden_suites"]} >= {"sssp", "cycle_count", "cycle_count_cuda"}
    sssp = next(g for g in doc["golden_suites"] if g["set"] == "sssp")
    assert len(sssp["case_sha256"]) == sssp["cases"]
    assert {o["name"] for o in doc["originals"]} >= {
        "MOSP-OpenMP",
        "MOSP-CUDA",
        "CycleEnumeration-GPU",
    }
    readme = (results / "README.md").read_text()
    assert cert.BEGIN in readme and "all parts passed" in readme
    # Rewriting keeps the hand-written text around the generated block.
    (results / "README.md").write_text("# Title\n\n" + readme + "\nNotes.\n")
    assert cert.main(["write", "--results", str(results), "--version", "t", "--allow-dirty"]) == 0
    text = (results / "README.md").read_text()
    assert (
        text.startswith("# Title")
        and text.rstrip().endswith("Notes.")
        and text.count(cert.BEGIN) == 1
    )

    # A failed check, a replay of another manifest and a missing backend fail the certificate.
    log.write_text("95% tests passed, 1 test failed out of 20\n")
    assert cert.main([*args, "--ctest-log", str(log)]) == 1
    bad = json.loads((results / "parity-cycle_count_cuda-0.json").read_text())
    bad["goldens"]["manifest_sha256"] = "0" * 64
    (results / "parity-cycle_count_cuda-0.json").write_text(json.dumps(bad))
    (results / "parity-sssp-1.json").unlink()
    assert cert.main(["write", "--results", str(results), "--version", "t", "--allow-dirty"]) == 1
    problems = " ".join(json.loads((results / "parity.json").read_text())["verdict"]["problems"])
    assert (
        "check asan" in problems
        and "replayed manifest" in problems
        and "no replay on cuda" in problems
    )


def test_measured_commit_must_have_the_release_code(cert) -> None:
    commit = head()
    assert cert.same_code(commit, commit) is True
    assert cert.same_code(commit + "+dirty", commit) is True
    assert cert.same_code("unknown", commit) is None
    assert cert.same_code("f" * 40, commit) is None
