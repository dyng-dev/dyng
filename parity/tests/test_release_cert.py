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
    assert {"ieee_tc_dyntrucy", "ipdps25_dynamosp_sosp", "ipdps25_dynamosp_mosp"} <= names
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


DRIVER = {"nvidia": "590.48.01", "cuda_driver_api": "13.1"}
CC_DIGESTS = {
    "DD": "e538ec13a262b84a0664db2f22e6ff720fb5bb74965536c37f4b8e962d156e4a",
    "github": "a5a43e92b9a22054882017a2b5dea267535c8d94705ac4987d967e517448c9d4",
    "twitch": "c2270e018af70321c8f6d1517abc097204105ec7bafe71527555b4f954fbea82",
    "collab": "043c465d207992054d2567a277bfba9fad24d994040241987e61f9e9daa12ae4",
}


def _gated_metrics(bench, suite: dict, backend: str) -> list[tuple[str, str]]:
    """(region name, gate kind) of every timed metric the suite gates on `backend` (a
    per-objective metric as objective 0's region)."""
    out = []
    for name, metric in suite["metrics"].items():
        gate = bench.metric_gate(metric, backend) if backend in metric else None
        if gate in (None, "none", "memory"):
            continue
        out.append((f"{name} obj0" if metric.get("per_objective") else name, gate))
    return out


def _cc_record(
    cases: list[str],
    commit: str,
    clocks: str | None = None,
    ratio: float = 0.9,
    regions: list[tuple[str, str]] | None = None,
):
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
        "driver": DRIVER,
        "datasets_sha256": dict(CC_DIGESTS),
        "results": {},
    }
    if clocks:
        record["protocol"]["clocks"] = {"control": clocks}
    regions = regions or [("static_end_to_end", "compute")]
    for case in cases:
        record["results"][f"count/{case}"] = {
            "regions": [_region(name, 100.0, 100.0 * ratio, kind) for name, kind in regions],
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
                "driver": DRIVER,
                "datasets_sha256": dict(CC_DIGESTS),
                "results": {
                    c: {
                        "original": {"peak_live_mib": 18.0},
                        "port[original]": {"peak_live_mib": 18.0, "ratio": 1.0},
                    }
                    for c in reading["cases"]
                },
            }
        else:
            regions = _gated_metrics(bench, suite, reading["backend"])
            rec = _cc_record(reading["cases"], commit, reading.get("clocks"), regions=regions, **kw)
        Path(job["record"]).write_text(json.dumps(rec))
    return jobs


def _write_sssp_records(bench, suite, records: Path, commit: str, ratio: float = 0.9):
    """Records of the suites over the MOSP bench inputs (sssp: perf_ab.py run|memory; mosp:
    perf_ab.py mosp, memory --mosp), with a region of every gated metric."""
    mosp = suite["algorithm"] == "mosp"
    records.mkdir(parents=True, exist_ok=True)
    jobs = bench.plan(suite, records=records, build_root=Path("/b"))
    for job in jobs:
        reading = next(r for r in suite["readings"] if r["name"] == job["reading"])
        dataset = next(d for d in suite["datasets"] if d["name"] == job["dataset"])
        backend = reading["backend"]
        base = suite["backends"][backend]["baseline"]
        rec = {
            "algorithm": suite["algorithm"],
            "reference": {
                "name": base,
                "commit": suite["baselines"][base]["commit"],
                "variant": "unpatched",
            },
            "port": {"commit": commit, "build": {"parity_preset": True}},
            "date": "2026-10-01T00:00:00Z",
            "driver": DRIVER,
            "inputs_hashed": "run start",
            "inputs_sha256": [
                f"{sha}  {rel}" for rel, sha in bench.expected_inputs(suite, dataset).items()
            ],
            "results": {},
        }
        regions = _gated_metrics(bench, suite, backend)
        for batch in ("safe50k", "unsafe50k", "local10k"):
            if job["kind"] == "memory":
                rec["results"][batch] = {
                    "original": {"peak_live_mib": 100.0},
                    "port": {"peak_live_mib": 100.0 * ratio},
                    "ratio": ratio,
                }
            else:
                rec["results"][batch] = {
                    "regions": [_region(n, 50.0, 50.0 * ratio, kind) for n, kind in regions],
                    "invalidated": {"original": 7, "port": 7, "equal_in_every_sample": True},
                }
        rec["protocol"] = {}
        if backend == "cuda" and job["kind"] == "run":
            rec["protocol"]["gpu_clocks"] = {"control": reading["clocks"]}
        if mosp:
            rec["objectives"] = suite["graph"]["num_weights"]
            if job["kind"] == "run":
                rec["preferences"] = "default (all 1)"
                rec["protocol"]["outputs"] = (
                    "written by both (bench/run.sh), into a directory deleted after each run"
                )
                rec["cuda_engine"] = "automatic" if backend == "cuda" else None
        Path(job["record"]).write_text(json.dumps(rec))
    return jobs


def _triad_case_result(bench, suite: dict, key: str, kind: str, runs: int, ratio: float) -> dict:
    case, pattern = key.split("/")
    m = bench.TRIAD_CASE.match(case)
    out = {
        "case": case,
        "pattern": pattern,
        "dataset": m["dataset"],
        "seed": int(m["seed"]),
        "batches": 3,
        "input": {"input": m["dataset"], "sha256": bench.triad_digest(suite, case)},
    }
    if kind == "memory":
        every = m["dataset"] == "coauth"
        points = []
        for k in range(4):
            row = "store_memory[build]" if k == 0 else f"store_memory[b{k}]"
            points.append(
                {
                    "point": "build" if k == 0 else f"batch {k}",
                    "row": row,
                    "dyng_device_bytes": 1000 + k,
                    "escher_gpu_device_bytes": 1000 + k,
                    "ratio": 1.0,
                    "gated": every or k in (0, 3),
                }
            )
        peak = 100 * 2**20
        return out | {
            "original": {"peak_live_mib": 100.0, "peak_live_bytes": peak},
            "port": {
                "peak_live_mib": 100.0 * ratio,
                "peak_live_bytes": round(peak * ratio),
                "ratio": round(peak * ratio) / peak,
            },
            "ratio": round(peak * ratio) / peak,
            "store_accounting": points,
            "foreign_gpu_processes": {"before": [], "after": []},
            "guard": {"runs_checked": 4, "passed": True},
        }
    rows = [("update[b1]", "compute"), ("update[b2]", "compute"), ("update[b3]", "compute")]
    rows += [("full_count", "compute"), ("end_to_end", "end_to_end")]
    return out | {
        "rounds": runs,
        "complete": True,
        "guard": {"rounds_checked": runs, "passed": True},
        "contaminated_rounds_kept": 0,
        "audit": {"flagged": False, "flagged_rounds": []},
        "monitor": {"rounds": [{"original": {"reasons": []}, "port": {"reasons": []}}] * runs},
        "regions": [
            _region(n, 500.0, 500.0 * ratio, kind)
            | {"original_runs": [500.0] * runs, "port_runs": [500.0 * ratio] * runs}
            for n, kind in rows
        ]
        + [
            {
                "region": "update.subtract[b1]",
                "gate_kind": "none",
                "original_ms": 2.0,
                "port_ms": 2.0,
                "ratio": 1.0,
                "original_runs": [2.0] * runs,
                "port_runs": [2.0] * runs,
            }
        ],  # fmt: skip
    }


def triad_protocol(bench, suite: dict, job: dict) -> dict:
    """The protocol fields parity/triad_perf.py writes for a reading (bench_suite checks them)."""
    if job["kind"] == "run":
        return {
            "rounds": job.get("runs"),
            "gpu": bench.triad_reading_gpu(suite, job),
            "lock": "perf.lock: held exclusively by this process",
            "layout": "one (--timing <work>/timing-ab.json on both sides)",
            "env": bench.TRIAD_ENV,
            "original": "the UNPATCHED copy's build/<pattern> (its Makefile)",
            "port": "dyng-compat-escher run --backend cuda --sync-stages",
        }
    return {
        "nsys": bench.TRIAD_NSYS + " (CUDA 13.1), memKind Device (cudaMalloc + cudaMallocAsync)",
        "gpu": bench.triad_reading_gpu(suite, job),
        "lock": "shared (flock -s by the caller); no foreign process on the GPU",
        "original": "the UNPATCHED build/<pattern>, no --check",
        "port": "dyng-compat-escher run --backend cuda --nvtx",
    }


def _write_triad_records(bench, suite, records: Path, commit: str, ratio: float = 0.9):
    """Records of parity/perf_ab.py triad_count run|memory (parity/triad_perf.py's schema 2)."""
    records.mkdir(parents=True, exist_ok=True)
    jobs = bench.plan(suite, records=records, build_root=Path("/b"))
    for job in jobs:
        reading = next(r for r in suite["readings"] if r["name"] == job["reading"])
        rec = {
            "schema": 2,
            "algorithm": "triad_count",
            "date": "2026-10-16T00:00:00Z",
            "reference": {
                "name": "ESCHER-GPU",
                "commit": suite["baselines"]["ESCHER-GPU"]["commit"],
                "variant": "unpatched",
            },
            "port": {"commit": commit, "build": {"parity_preset": True}},
            "driver": DRIVER,
            "protocol": triad_protocol(bench, suite, job),
            "results": {
                f"{c}/{p}": _triad_case_result(
                    bench, suite, f"{c}/{p}", job["kind"], job.get("runs") or 0, ratio
                )
                for c in job["cases"]
                for p in job["patterns"]
            },
        }
        if job["kind"] == "run":
            rec["protocol"]["clocks"] = {"control": reading["clocks"]}
        Path(job["record"]).write_text(json.dumps(rec))
    return jobs


def _full_results(cert, bench, tmp_path: Path, commit: str) -> Path:
    """A results directory with replays, both suites' whole summaries and a passed check that
    ran every fixture test."""
    results = tmp_path / "results"
    results.mkdir()
    _replays(cert, results, commit)
    for path in SUITES:
        suite = bench.load_suite(path)
        write = {
            "cycle_count": _write_cc_records,
            "triad_count": _write_triad_records,
        }.get(suite["algorithm"], _write_sssp_records)
        records = tmp_path / f"rec-{suite['suite']}"
        jobs = write(bench, suite, records, commit)
        summary = bench.summarize(
            suite, jobs, out=results, version="t", records=records, inputs=None
        )
        assert summary["verdict"]["passed"], summary["verdict"]
    return results


def _ctest_log(names: list[str], failed: tuple[str, ...] = ()) -> str:
    lines = [
        f"{i + 1:3}/{len(names)} Test #{i + 1}: {n} ......   "
        + ("***Failed" if n in failed else "Passed")
        + "    0.01 sec"
        for i, n in enumerate(names)
    ]
    ok = len(names) - len(failed)
    pct = 100 * ok // len(names)
    tail = f"{pct}% tests passed, {len(failed)} tests failed out of {len(names)}"
    return "\n".join(lines) + "\n\n" + tail + "\n"


def _fixture_test_names(cert) -> list[str]:
    """One ctest name per fixtures.toml pattern (a pattern's '*' replaced by 'X')."""
    return [p.replace("*", "X") for p in cert.fixture_patterns()]


def _golden_mutation_record(commit: str) -> dict:
    """A passed record of parity/mutate.py run: the control and every mutation detected, each
    replay of a manifest of parity/goldens.toml."""
    import tomllib

    mutate = load("parity/mutate.py")
    sets = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"]
    sssp = {"manifest_sha256": sets["sssp"]["manifest_sha256"]}
    # The GoogleTest suites (cbst, M8a) replay no golden corpus: mutate.py records no manifest.
    gtest = {name for name, spec in mutate.SUITES.items() if spec.get("runner") == "gtest"}

    def goldens(suite: str) -> dict:
        return {} if suite in gtest else {"goldens": sssp}

    return {
        "schema": 2,
        "commit": commit,
        "results": [
            {
                "name": "control",
                "verdict": "passed",
                "replays": [
                    {"suite": "sssp", "config": "sequential", "goldens": sssp},
                    *({"suite": s, "config": "cuda"} for s in sorted(gtest)),
                ],
            },
            *(
                {
                    "name": m["name"],
                    "suite": m["suite"],
                    "config": m["config"],
                    **goldens(m["suite"]),
                    "detected": True,
                    "verdict": "detected",
                }
                for m in mutate.MUTATIONS
            ),
        ],
        "passed": True,
    }


def _release_checks(cert, results: Path, tmp_path: Path, commit: str, skip=()) -> None:
    """Records every check of REQUIRED_CHECKS but asan and those in `skip`, each passed."""
    log = tmp_path / "check.log"
    log.write_text(_ctest_log(_fixture_test_names(cert)))
    gpu = tmp_path / "gpu_local_summary.md"
    gpu.write_text(
        "| step | result |\n|---|---|\n| memcheck | passed |\n| synccheck | passed |\n"
        "| racecheck | passed |\n"
    )
    (results / "mutation-goldens.json").write_text(json.dumps(_golden_mutation_record(commit)))
    base = ["check", "--results", str(results), "--commit", commit, "--command", "x"]
    recorded = {
        "tsan": ["--ctest-log", str(log)],
        "tsan-openmp": ["--ctest-log", str(log)],
        "mutation-ctests": ["--ctest-log", str(log)],
        "mutation-sssp": ["--scope", "library", "--result", "passed", "--evidence", str(log)],
        "mutation-goldens": [
            "--scope",
            "library",
            "--json-verdict",
            str(results / "mutation-goldens.json"),
        ],
        "distributions": ["--scope", "packaging", "--result", "passed", "--evidence", str(log)],
        "check-parity": ["--scope", "repo", "--result", "passed", "--evidence", str(log)],
        "api-check": ["--scope", "repo", "--result", "passed", "--evidence", str(log)],
        "gpu_local": ["--scope", "repo", "--gpu-summary", str(gpu)],
    }
    assert set(recorded) | {"asan"} == set(cert.REQUIRED_CHECKS)
    for name, extra in recorded.items():
        if name not in skip:
            assert cert.main([*base, "--name", name, *extra]) == 0, name


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
    assert cert.parse_ctest(log) == {"tests": 596, "failed": 0, "skipped": 0, "percent_passed": 100}
    cmake4 = (
        "100% tests passed out of 595\n\nThe following tests did not run:\n"
        "\t161 - AllocationFailure.InsertEdgeIsStrong (Skipped)\n"
        "\t162 - A.B (Skipped)\n"
    )
    assert cert.parse_ctest(cmake4) == {
        "tests": 595,
        "failed": 0,
        "skipped": 2,
        "percent_passed": 100,
    }
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
    # The paper-scale mosp goldens (M7): SHA-256 digests, no manifest; every case replayed.
    for i, cfg in enumerate([["sequential", "openmp:28"], ["cuda-fused", "cuda-operators"]]):
        record = {
            "algorithm": "mosp",
            "set": "mosp_scale",
            "port": {"commit": commit},
            "configs": cfg,
            "cases": sorted(sets["mosp_scale"]["cases"]),
            "matrix": {"gate": {c: {"cases": 12, "pass": 12, "fail": []} for c in cfg}},
            "passed": True,
        }
        (results / f"parity-mosp_scale-{i}.json").write_text(json.dumps(record))
    # The CBST trace goldens (M8a): every set replayed through the dynG store under its original's
    # preset and dyng_default (parity/certify.py, CBST_REQUIRED_PRESETS).
    for name, spec in sets.items():
        if spec.get("kind") != "cbst_trace":
            continue
        for preset in (cert.CBST_ORIGINAL_PRESET[spec["reference"]], "dyng_default"):
            config = f"device:{preset}/exact"
            record = {
                "set": name,
                "port": {"commit": commit},
                "goldens": {"manifest_sha256": spec["manifest_sha256"]},
                "configs": [config],
                "matrix": {"case": {config: "equal"}},
                "passed": True,
            }
            (results / f"parity-{name}-device-{preset}.json").write_text(json.dumps(record))
        # The hypergraph container's replays of the set's batch records (M8b, from 0.3).
        for feature, backends in cert.HYPERGRAPH_ID_REPLAYS.get(name, {}).items():
            mode = "both" if feature == "ids" else feature
            for backend in backends:
                _hypergraph_id_record(results, name, spec, commit, backend, mode)
    # ESCHER-GPU's triad goldens (M9a, from 0.3): every backend of the coverage table, both modes.
    _triad_replays(cert, results, commit)


TRIAD_SET = "triad_count_escher_gpu"


def _triad_replays(cert, results: Path, commit: str, skip: tuple = ()) -> None:
    """Passed certificate records of `compare.py triad_count --counter algorithm --mode both
    --certificate` for every backend of the coverage table (parity/triad_replay.py's format),
    but those in `skip`."""
    import tomllib

    replay = load("parity/triad_replay.py")
    spec = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"][TRIAD_SET]
    for backend, cases in cert.triad_required_cases(spec).items():
        if backend in skip:
            continue
        doc = {
            "set": TRIAD_SET,
            "manifest_sha256": spec["manifest_sha256"],
            "backend": backend,
            "threads": 8 if backend == "openmp" else None,
            "mode": "both",
            "counter": "algorithm",
            "plant": None,
            "coverage": True,
            "commit": commit,
            "cases": {c: {"passed": True, "batches": 3} for c in sorted(cases)},
            "passed": True,
        }
        replay.write_certificate(doc, results)


def _hypergraph_id_record(
    results: Path,
    name: str,
    spec: dict,
    commit: str,
    backend: str,
    feature: str,
    checks: dict | None = None,
) -> None:
    """A passed record of `compare.py hypergraph_ids --certificate` (the hypergraph-id schema)."""
    config = f"{backend}:hypergraph_ids/{feature}"
    record = {
        "set": name,
        "kind": "hypergraph_ids",
        "port": {"commit": commit},
        "goldens": {"manifest_sha256": spec["manifest_sha256"]},
        "configs": [config],
        "checks": checks
        if checks is not None
        else {"model": True, "verified": True, "line_graph_oracle": True},
        "matrix": {"case": {config: "equal"}},
        "passed": True,
    }
    path = results / f"parity-{name}-hypergraph_ids-{backend}-{feature}.json"
    path.write_text(json.dumps(record))


@pytest.fixture()
def fixed_driver(cert, monkeypatch: pytest.MonkeyPatch):
    """The driver seen "now" is the records' (the certificate compares the two)."""
    real = cert.environment
    # The certificates of these tests are written for several series on one tree; the rule that
    # a label follows its tree's series is tested on its own (test_the_label_follows_the_tree).
    monkeypatch.setattr(cert, "tree_version", lambda: "")

    def environment() -> dict:
        env = real()
        env["driver"] = dict(DRIVER)
        return env

    monkeypatch.setattr(cert, "environment", environment)


def test_certificate_from_results(cert, bench, tmp_path: Path, fixed_driver) -> None:
    commit = head()
    results = _full_results(cert, bench, tmp_path, commit)
    log = tmp_path / "asan.log"
    log.write_text(
        _ctest_log(_fixture_test_names(cert))
        + "\nThe following tests did not run:\n\t7 - A.B (Skipped)\n"
    )
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
    _release_checks(cert, results, tmp_path, commit)
    assert (
        cert.main(["write", "--results", str(results), "--version", "t", "--allow-dirty"]) == 0
    ), json.loads((results / "parity.json").read_text())["verdict"]
    doc = json.loads((results / "parity.json").read_text())
    assert doc["verdict"]["passed"] and doc["commit"].startswith(commit)
    assert {g["set"] for g in doc["golden_suites"]} >= {
        "sssp",
        "cycle_count",
        "cycle_count_cuda",
        "mosp_scale",
    }
    sssp = next(g for g in doc["golden_suites"] if g["set"] == "sssp")
    assert len(sssp["case_sha256"]) == sssp["cases"]
    scale = next(g for g in doc["golden_suites"] if g["set"] == "mosp_scale")
    assert scale["passed"] and len(scale["case_sha256"]) == scale["cases"] == 20
    assert {o["name"] for o in doc["originals"]} >= {
        "MOSP-OpenMP",
        "MOSP-CUDA",
        "CycleEnumeration-GPU",
    }
    n = len(_fixture_test_names(cert))
    assert all(f["passed"] for f in doc["committed_fixtures"])
    assert {f["set"] for f in doc["committed_fixtures"]} == {
        "mosp_changes",
        "mosp_graph_io",
        "mosp_sssp",
        "mosp_combined",
        "cycle_enum",
        "cbst_trace",
        "triad_counts",
    }
    assert doc["environment"]["driver"]["nvidia"] == DRIVER["nvidia"]
    assert "named by" in doc["environment"]["driver"]["source"]
    asan = next(c for c in doc["checks"] if c["name"] == "asan")
    assert asan["evidence"][0]["sha256"] and (results / asan["evidence"][0]["excerpt"]).is_file()
    readme = (results / "README.md").read_text()
    assert cert.BEGIN in readme and "all parts passed" in readme
    assert (
        f"| asan | `ctest --preset asan` | tests | passed ({n} / {n} tests, 1 skipped) |" in readme
    )
    assert "### Committed fixtures" in readme and "hashed by the" in readme
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
    names = _fixture_test_names(cert)
    log.write_text(_ctest_log(names, failed=(names[0],)))
    assert cert.main([*args, "--ctest-log", str(log)]) == 1
    bad = json.loads((results / "parity-cycle_count_cuda-0.json").read_text())
    bad["goldens"]["manifest_sha256"] = "0" * 64
    (results / "parity-cycle_count_cuda-0.json").write_text(json.dumps(bad))
    (results / "parity-sssp-1.json").unlink()
    # A paper-scale mosp replay of part of the set (no manifest to compare) fails as well.
    partial = json.loads((results / "parity-mosp_scale-0.json").read_text())
    partial["cases"] = partial["cases"][:-1]
    (results / "parity-mosp_scale-0.json").write_text(json.dumps(partial))
    assert cert.main(["write", "--results", str(results), "--version", "t", "--allow-dirty"]) == 1
    problems = " ".join(json.loads((results / "parity.json").read_text())["verdict"]["problems"])
    assert (
        "check asan" in problems
        and "replayed manifest" in problems
        and "no replay on cuda" in problems
        and "replayed 19 cases, not the 20 cases of goldens.toml's [sets.mosp_scale]" in problems
    )


def test_measured_commit_must_have_the_release_code(cert) -> None:
    commit = head()
    assert cert.same_code(commit, commit) is True
    assert cert.same_code(commit + "+dirty", commit) is True
    assert cert.same_code("unknown", commit) is None
    assert cert.same_code("f" * 40, commit) is None


def _fake_elf(cert, tmp_path: Path, a_bytes: bytes, b_bytes: bytes, *, b_symbols=None):
    a, b = tmp_path / "a.so", tmp_path / "b.so"
    a.write_bytes(a_bytes)
    b.write_bytes(b_bytes)
    secs = [
        {
            "name": ".text",
            "type": "PROGBITS",
            "addr": 0x1000,
            "offset": 0,
            "size": 64,
            "flags": "AX",
        },
        {
            "name": ".rodata",
            "type": "PROGBITS",
            "addr": 0x2000,
            "offset": 64,
            "size": 32,
            "flags": "A",
        },
        {
            "name": ".eh_frame",
            "type": "PROGBITS",
            "addr": 0x3000,
            "offset": 96,
            "size": 32,
            "flags": "A",
        },
    ]
    syms = [(0x1000, 32, "T", "dyng::algorithms()"), (0x1020, 32, "T", "dyng::sssp_kernel()")]
    table = {a: syms, b: b_symbols or syms}
    return lambda allowed: cert.compare_elf(
        a, b, allowed, sections=lambda _: secs, symbols=lambda path: table[path]
    )


def test_equivalent_builds_differ_only_in_allowed_functions(cert, tmp_path: Path) -> None:
    allowed = {"dyng::algorithms()"}
    base = bytes(128)

    def flip(*offsets: int) -> bytes:
        out = bytearray(base)
        for o in offsets:
            out[o] = 1
        return bytes(out)

    assert _fake_elf(cert, tmp_path, base, base)(allowed)["result"] == "identical"
    ok = _fake_elf(cert, tmp_path, base, flip(5, 100))(allowed)
    assert ok["result"] == "equivalent", ok
    assert ok["differing_functions"] == ["dyng::algorithms()"]
    assert ok["differing_bytes_by_section"] == {".text": 1, ".eh_frame": 1}
    # Code of another function, data, a moved symbol, another function resized, another size.
    assert _fake_elf(cert, tmp_path, base, flip(40))(allowed)["result"] == "different"
    assert _fake_elf(cert, tmp_path, base, flip(70))(allowed)["result"] == "different"
    moved = [(0x1000, 32, "T", "dyng::algorithms()"), (0x1028, 24, "T", "dyng::sssp_kernel()")]
    assert (
        _fake_elf(cert, tmp_path, base, flip(5), b_symbols=moved)(allowed)["result"] == "different"
    )
    grown = [(0x1000, 32, "T", "dyng::algorithms()"), (0x1020, 30, "T", "dyng::sssp_kernel()")]
    assert (
        _fake_elf(cert, tmp_path, base, flip(5), b_symbols=grown)(allowed)["result"] == "different"
    )
    resized = [(0x1000, 30, "T", "dyng::algorithms()"), (0x1020, 32, "T", "dyng::sssp_kernel()")]
    assert (
        _fake_elf(cert, tmp_path, base, flip(5), b_symbols=resized)(allowed)["result"]
        == "equivalent"
    )
    assert _fake_elf(cert, tmp_path, base, base + b"x")(allowed)["result"] == "different"


def test_elf_readers_on_a_real_binary(cert, tmp_path: Path) -> None:
    import shutil

    if not (shutil.which("readelf") and shutil.which("nm")):
        pytest.skip("binutils not installed")
    exe = Path(sys.executable).resolve()
    sections = cert.elf_sections(exe)
    assert any(s["name"] == ".text" and "X" in s["flags"] for s in sections)
    copy_ = tmp_path / "copy"
    shutil.copy2(exe, copy_)
    assert cert.compare_elf(exe, copy_, set())["result"] == "identical"


def test_metadata_only_difference_needs_an_equivalence_record(
    cert, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured, release = "a" * 40, "b" * 40
    meta = sorted(cert.METADATA_PATHS)
    diffs = {(measured, "library"): meta, (measured, "tests"): meta, (release, "library"): []}
    monkeypatch.setattr(cert, "resolve", lambda c: c.split("+")[0] if c else None)
    monkeypatch.setattr(cert, "code_diff", lambda c, h, scope="library": diffs.get((c, scope)))
    assert cert.same_code(measured, release, "library", tmp_path) is False  # no record yet
    pair = {"measured": measured, "release": release, "differing_paths": meta, "passed": True}
    (tmp_path / "equivalence.json").write_text(json.dumps({"schema": 1, "pairs": [pair]}))
    assert cert.same_code(measured, release, "library", tmp_path) is True
    assert cert.same_code(measured, release, "library") is False  # without the results
    assert cert.same_code(measured, release, "tests", tmp_path) is False  # tests read metadata
    (tmp_path / "equivalence.json").write_text(json.dumps({"pairs": [{**pair, "passed": False}]}))
    assert cert.same_code(measured, release, "library", tmp_path) is False
    (tmp_path / "equivalence.json").write_text(json.dumps({"pairs": [pair]}))
    diffs[(measured, "library")] = [*meta, "cpp/src/algorithms/sssp/sssp.cpp"]
    assert cert.same_code(measured, release, "library", tmp_path) is False  # beyond metadata


def test_metadata_paths_cover_every_algorithm_manifest(cert) -> None:
    # A maturity change rewrites a manifest and the registry table; no build reads a manifest.
    for name in ("sssp", "cycle_count", "mosp", "dynamic_bfs", "triangle_delta"):
        assert f"cpp/src/algorithms/{name}/manifest.toml" in cert.METADATA_PATHS
    assert not any("_template" in path for path in cert.METADATA_PATHS)
    assert "cpp/src/core/registry_table.inc" in cert.METADATA_PATHS


# --- mutate.py -----------------------------------------------------------------------------------


def test_mutation_points_match_the_code_once() -> None:
    mutate = load("parity/mutate.py")
    assert {m["config"] for m in mutate.MUTATIONS} == {"sequential", "openmp:4", "cuda"}
    # Every suite has a mutation on each backend (PLAN 8.4; mosp's path costs are shared).
    for suite in ("sssp", "mosp", "cycle_count"):
        assert {m["config"] for m in mutate.MUTATIONS if m["suite"] == suite} == {
            "sequential",
            "openmp:4",
            "cuda",
        }, suite
    assert len({m["name"] for m in mutate.MUTATIONS}) == len(mutate.MUTATIONS)
    hooks = "\n".join(
        p.read_text() for p in sorted((REPO / "cpp/src/algorithms/cycle_count").iterdir())
    )
    # The CBST mutation hooks (M8a): the store's sources and the graph-on-CBST test.
    hooks += "\n".join(
        p.read_text() for p in sorted((REPO / "cpp/src/hypergraph/escher").iterdir())
    )
    hooks += (REPO / "cpp/tests/parity/escher/cbst_graph_test.cpp").read_text()
    # The hypergraph container's hooks (M8b): its sources outside the store (the line graph's in
    # cpp/src/hypergraph/line_graph).
    hooks += "\n".join(p.read_text() for p in sorted((REPO / "cpp/src/hypergraph").glob("*.cpp")))
    hooks += "\n".join(
        p.read_text() for p in sorted((REPO / "cpp/src/hypergraph/line_graph").glob("*.c*"))
    )
    # The kit's planted bug (M8b, criterion 11): in the test-only hypergraph fake.
    hooks += (REPO / "cpp/tests/framework/fake_hyper_problems.hpp").read_text()
    assert {m["name"] for m in mutate.MUTATIONS if m["suite"] == "kit"} == {
        "kit_frontier_requested_ids"
    }
    # The triad oracle's planted mutation (M9a, criterion 4): in its source.
    hooks += (REPO / "cpp/src/testing/triad_reference.cpp").read_text()
    assert {m["name"] for m in mutate.MUTATIONS if m["suite"] == "triad_reference"} == {
        "triad_reference_class_map"
    }
    assert mutate.suite_targets("triad_reference") == ["dyng_triad_reference_tests"]
    # triad_count's mutations (M9a, criterion 11): in the code its two engines share (since M9b
    # the exactly-once predicates are the local-pattern engine's installed header, ADR 0058).
    for path in sorted((REPO / "cpp/src/algorithms/triad_count").rglob("*")):
        if path.is_file():
            hooks += path.read_text()
    hooks += (REPO / "cpp/include/dyng/experimental/detail/local_patterns_engine.hpp").read_text()
    assert {m["name"] for m in mutate.MUTATIONS if m["suite"] == "triad_count"} == {
        "triad_wrong_bin_sign",
        "triad_skip_frontier_filter_b",
        "triad_skip_frontier_filter_k",
        "triad_inserted_requested_ids",
        "triad_type1_unbounded_keys",
        "cbst_unfill_drops_last_triad",
    }
    assert mutate.suite_targets("triad_count") == [
        "dyng_triad_count_tests",
        "dyng_triad_count_cuda_tests",
    ]
    assert mutate.suite_targets("kit") == [
        "dyng_hypergraph_kit_tests",
        "dyng_hypergraph_kit_cuda_tests",
    ]
    assert {m["name"] for m in mutate.MUTATIONS if m["suite"] == "hypergraph"} == {
        "hypergraph_h2v_not_deduplicated",
        "hypergraph_allocator_ties_reversed",
        "hypergraph_mirror_additions_dropped",
    }
    assert {m["name"] for m in mutate.MUTATIONS if m["suite"] == "cbst"} == {
        "cbst_unfill_drops_last",
        "cbst_graph_own_ids",
        "cbst_reversed_ties",
        "cbst_surplus_rule1",
        "cbst_keep_dead_slots",
        "cbst_compaction_padding",
        "cbst_requested_keys",
    }
    assert mutate.suite_targets("cbst") == [
        "dyng_cbst_device_tests",
        "dyng_cbst_policy_tests",
        "dyng_cbst_graph_tests",
    ]
    assert mutate.suite_targets("sssp") == ["dyng-compat-mosp"]
    for m in mutate.MUTATIONS:
        assert m["suite"] in mutate.SUITES, m["name"]
        if "defines" in m:
            # a recorded hook of the sources, compiled in by the define
            assert "file" not in m and m["defines"], m["name"]
            for define in m["defines"]:
                assert f"defined({define})" in hooks, (m["name"], define)
            continue
        text = (REPO / m["file"]).read_text()
        assert text.count(m["old"]) == 1, m["name"]
        assert m["new"] != m["old"]


def test_mutation_defines_reach_both_compilers() -> None:
    mutate = load("parity/mutate.py")
    assert mutate.configure_args([]) == []
    assert mutate.configure_args(["A", "B"]) == [
        "-DCMAKE_CXX_FLAGS=-DA -DB",
        "-DCMAKE_CUDA_FLAGS=-DA -DB",
    ]


def test_mutation_failures_read_both_matrix_layouts() -> None:
    mutate = load("parity/mutate.py")
    sssp = {"matrix": {"g1": {"cuda": {"fail": ["a", "b"]}}, "g2": {"cuda": {"fail": []}}}}
    assert mutate.failures(sssp, ["cuda"]) == {"cuda": 2}
    cycle = {
        "matrix": {
            "c1": {"sequential": "equal"},
            "c2": {"sequential": ["differs"]},
            "c3": {"sequential": "skipped"},
        }
    }
    assert mutate.failures(cycle, ["sequential"]) == {"sequential": 1}


def test_mutation_apply_refuses_an_ambiguous_point(tmp_path: Path) -> None:
    mutate = load("parity/mutate.py")
    (tmp_path / "f.cpp").write_text("a\na\n")
    with pytest.raises(SystemExit, match="matches 2 times"):
        mutate.apply(tmp_path, {"name": "x", "file": "f.cpp", "old": "a", "new": "b"})
    (tmp_path / "f.cpp").write_text("a\n")
    mutate.apply(tmp_path, {"name": "x", "file": "f.cpp", "old": "a", "new": "b"})
    assert (tmp_path / "f.cpp").read_text() == "b\n"


def test_summary_lists_contaminated_runs_without_failing(bench, tmp_path: Path) -> None:
    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    jobs = _write_cc_records(bench, suite, tmp_path / "rec", head(), ratio=0.9)
    omp = next(j for j in jobs if j["reading"] == "openmp")
    rec = json.loads(Path(omp["record"]).read_text())
    rec["results"]["count/DD_k3"]["contamination"] = {
        "original": {"flagged_runs": 2},
        "port": {"flagged_runs": 1},
    }
    Path(omp["record"]).write_text(json.dumps(rec))
    out = tmp_path / "out"
    summary = bench.summarize(suite, jobs, out=out, version="t", records=tmp_path, inputs=None)
    assert summary["verdict"]["passed"]
    assert summary["verdict"]["contaminated"] == [
        "openmp: count/DD_k3 (3 runs above the foreign-load threshold)"
    ]


# --- Review of R010: whole suites, verified inputs, the driver, evidence, scopes, fixtures --------


def test_certificate_refuses_a_partial_or_changed_suite(
    cert, bench, tmp_path: Path, fixed_driver
) -> None:
    commit = head()
    results = _full_results(cert, bench, tmp_path, commit)
    log = tmp_path / "asan.log"
    log.write_text(_ctest_log(_fixture_test_names(cert)))
    base = ["--results", str(results)]
    check = ["check", *base, "--name", "asan", "--command", "ctest", "--ctest-log", str(log)]
    assert cert.main(check) == 0
    _release_checks(cert, results, tmp_path, commit)
    write = ["write", *base, "--version", "t", "--allow-dirty"]
    assert cert.main(write) == 0

    def problems() -> str:
        return " ".join(json.loads((results / "parity.json").read_text())["verdict"]["problems"])

    path = results / "ieee_tc_dyntrucy.json"
    full = json.loads(path.read_text())
    # One reading of six: a narrowed run's summary in place of the whole suite's.
    cut = copy.deepcopy(full)
    cut["readings"] = [r for r in cut["readings"] if r["reading"] == "cuda"]
    path.write_text(json.dumps(cut))
    assert cert.main(write) == 1
    assert "the planned reading openmp is missing" in problems()
    # An incomplete reading, a repeated one, a partial marker.
    cut = copy.deepcopy(full)
    cut["readings"][0]["complete"] = False
    cut["readings"].append(cut["readings"][1])
    cut["partial"] = {"readings": ["cuda"]}
    path.write_text(json.dumps(cut))
    assert cert.main(write) == 1
    text = problems()
    assert "is not complete" in text and "appears 2 times" in text and "partial" in text
    # Another suite file than the one summarized; inputs that are not verified.
    cut = copy.deepcopy(full)
    cut["suite_sha256"] = "0" * 64
    path.write_text(json.dumps(cut))
    assert cert.main(write) == 1
    assert "SHA-256 differs" in problems()
    cut = copy.deepcopy(full)
    cut["inputs"]["verified"] = False
    path.write_text(json.dumps(cut))
    assert cert.main(write) == 1
    assert "inputs of the readings are not verified" in problems()
    # A partial summary next to the whole one is refused too.
    path.write_text(json.dumps(full))
    (results / "ieee_tc_dyntrucy.partial.json").write_text(json.dumps(full))
    assert cert.main(write) == 1
    assert "partial execution's summary" in problems()


def test_bench_suite_never_writes_a_narrowed_summary_into_the_results(
    bench, tmp_path: Path
) -> None:
    suite_file = REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml"
    with pytest.raises(SystemExit) as exc:
        bench.main(["summarize", str(suite_file), "--readings", "cuda", "--version", "t"])
    assert exc.value.code == 2
    suite = bench.load_suite(suite_file)
    jobs = _write_cc_records(bench, suite, tmp_path / "rec", head())
    cuda = [j for j in jobs if j["reading"] == "cuda"]
    out = tmp_path / "out"
    summary = bench.summarize(
        suite,
        cuda,
        out=out,
        version="t",
        records=tmp_path / "rec",
        inputs=None,
        partial={"readings": ["cuda"]},
    )
    assert summary["partial"] and (out / "ieee_tc_dyntrucy.partial.json").is_file()
    assert not (out / "ieee_tc_dyntrucy.json").exists()
    # run never replaces a record unless asked to.
    with pytest.raises(bench.SuiteError, match="records exist already"):
        bench.run_jobs(cuda, skip_existing=False, log=None)
    done = bench.run_jobs(cuda, skip_existing=True, log=None)
    assert done[0]["skipped"]


def test_inputs_of_a_record_without_digests_are_hashed_again(bench, tmp_path: Path) -> None:
    import datetime
    import os

    suite = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    suite["harness"]["inputs"] = str(tmp_path)
    data = tmp_path / "DD" / "DD_A.txt"
    data.parent.mkdir()
    data.write_text("1, 2\n")
    suite["datasets"] = [dict(suite["datasets"][0], sha256=bench.sha256_file(data))]
    job = {"reading": "cuda-memory", "dataset": None}
    record = {"date": "2030-01-01T00:00:00Z", "results": {"count/DD_k4": {}}}
    later = datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC)
    got = bench.check_record_inputs(suite, job, record, later)
    assert got["verified"] and "unchanged since the execution started" in got["how"]
    assert bench.check_record_inputs(suite, job, record, None)["verified"]  # the record's date
    # A file written after the execution started, or with other content, is not verified.
    earlier = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
    got = bench.check_record_inputs(suite, job, record, earlier)
    assert not got["verified"] and "changed at" in got["problems"][0]
    data.write_text("1, 3\n")
    os.utime(data, (0, 0))
    got = bench.check_record_inputs(suite, job, record, later)
    assert not got["verified"] and "SHA-256" in got["problems"][0]
    # Digests the measuring process took are used as they are.
    record["datasets_sha256"] = {"DD": suite["datasets"][0]["sha256"]}
    got = bench.check_record_inputs(suite, job, record, earlier)
    assert got["verified"] and got["how"] == "hashed by the measuring process"
    # The run log gives the start of the execution.
    (tmp_path / "ieee_tc_dyntrucy.log").write_text(
        "== 2026-09-30T21:28:39-05:00 a\nrc=0 x\n== 2026-10-01T01:00:00-05:00 b\n"
    )
    start = bench.execution_started(tmp_path, suite)
    assert start == datetime.datetime(2026, 10, 1, 2, 28, 39, tzinfo=datetime.UTC)


def test_the_measured_driver(cert, monkeypatch: pytest.MonkeyPatch) -> None:
    other = {"nvidia": "580.1", "cuda_driver_api": "13.0"}

    def suite(*drivers):
        return [
            {
                "execution_started": "2026-09-30T00:00:00Z",
                "readings": [
                    {"record": f"r{i}", "backend": "cuda", "kind": "run", "driver": d}
                    for i, d in enumerate(drivers)
                ],
            }
        ]

    problems: list[str] = []
    got = cert.measured_driver(suite(DRIVER, DRIVER), dict(DRIVER), problems)
    assert not problems and got["nvidia"] == DRIVER["nvidia"]
    cert.measured_driver(suite(DRIVER, other), dict(DRIVER), problems)
    assert any("different drivers" in p for p in problems)
    problems.clear()
    cert.measured_driver(suite(other), dict(DRIVER), problems)
    assert any("not the one seen now" in p for p in problems)
    # Records without a driver: the driver seen now must be shown unchanged since they started.
    problems.clear()
    monkeypatch.setattr(
        cert, "driver_unchanged_since", lambda start, v: {"verified": False, "since": "x"}
    )
    cert.measured_driver(suite(None), dict(DRIVER), problems)
    assert any("name no driver" in p for p in problems)
    problems.clear()
    monkeypatch.setattr(
        cert, "driver_unchanged_since", lambda start, v: {"verified": True, "since": "x"}
    )
    got = cert.measured_driver(suite(None), dict(DRIVER), problems)
    assert not problems and "unchanged since before" in got["source"]


def test_driver_unchanged_since_on_this_machine(cert) -> None:
    import datetime

    future = datetime.datetime(2100, 1, 1, tzinfo=datetime.UTC)
    got = cert.driver_unchanged_since(future, None)
    assert got["verified"] is False  # no driver version given: never verified
    assert (
        cert.driver_unchanged_since(datetime.datetime(1971, 1, 1, tzinfo=datetime.UTC), "1")[
            "verified"
        ]
        is False
    )


def test_check_needs_evidence_and_records_it(cert, tmp_path: Path) -> None:
    results = tmp_path / "r"
    args = ["check", "--results", str(results), "--name", "dist", "--command", "ci/wheels.sh"]
    with pytest.raises(SystemExit, match="needs --evidence"):
        cert.main([*args, "--result", "passed", "--scope", "packaging"])
    log = tmp_path / "wheels.log"
    log.write_text("==> sdist\nnoise\n\x1b[32mPASSED\x1b[0m\nok     dyng.whl\n" + "x\n" * 400)
    assert (
        cert.main([*args, "--result", "passed", "--scope", "packaging", "--evidence", str(log)])
        == 0
    )
    entry = json.loads((results / "checks.json").read_text())["checks"][0]
    assert entry["scope"] == "packaging"
    assert entry["evidence"][0]["bytes"] == log.stat().st_size
    excerpt = (results / entry["evidence"][0]["excerpt"]).read_text()
    assert "==> sdist" in excerpt and "PASSED" in excerpt and "noise" not in excerpt
    short = tmp_path / "short.log"
    short.write_text("[mutate] control passed\n")  # a short log is copied whole
    assert cert.main([*args, "--result", "passed", "--evidence", str(short)]) == 0
    entry = json.loads((results / "checks.json").read_text())["checks"][0]
    assert "[mutate] control passed" in (results / entry["evidence"][0]["excerpt"]).read_text()
    summary = tmp_path / "gpu.md"
    summary.write_text(
        "| step | result |\n|---|---|\n| memcheck | passed |\n| racecheck | skipped |\n"
    )
    gpu = ["check", "--results", str(results), "--name", "gpu", "--command", "ci/gpu_local.sh"]
    assert cert.main([*gpu, "--gpu-summary", str(summary)]) == 0
    entry = json.loads((results / "checks.json").read_text())["checks"][-1]
    assert entry["required_steps"] == []
    assert cert.main([*gpu, "--gpu-summary", str(summary), "--require-steps", "racecheck"]) == 1
    entry = json.loads((results / "checks.json").read_text())["checks"][-1]
    assert entry["required_steps"] == ["racecheck"] and entry["missing_steps"] == ["racecheck"]


def test_scopes_cover_what_each_check_depends_on(cert) -> None:
    commit = head()
    for scope in ("library", "tests", "packaging", "repo"):
        assert cert.same_code(commit, commit, scope) is True
    assert "pyproject.toml" in cert.CODE_PATHS["packaging"]
    assert ".github/workflows/release.yml" in cert.CODE_PATHS["packaging"]
    assert cert.CODE_PATHS["repo"] == [".", ":(exclude)benchmarks/results"]
    parent = subprocess.check_output(
        ["git", "-C", REPO, "rev-list", "--max-count=1", "--skip=40", "HEAD"], text=True
    ).strip()
    if parent:
        changed = cert.code_diff(parent, commit, "repo")
        assert changed is not None and not any(c.startswith("benchmarks/results/") for c in changed)


def test_committed_fixtures(cert) -> None:
    names = _fixture_test_names(cert)
    ok = [{"name": "asan", "release_code": True, "fixture_tests": dict.fromkeys(names, "Passed")}]
    problems: list[str] = []
    sets = cert.committed_fixtures(ok, problems)
    assert not problems and all(s["passed"] for s in sets)
    for s in sets:
        digest, count = cert.tree_digest(s["directory"])
        assert (digest, count) == (s["sha256"], s["files"]) and count > 0
        assert s["commit"] and len(s["commit"]) == 40
    # A test that failed somewhere, a pattern no check ran, a check of other code.
    bad = [{**ok[0], "fixture_tests": {**ok[0]["fixture_tests"], names[0]: "Failed"}}]
    problems.clear()
    cert.committed_fixtures(bad, problems)
    assert any("Failed" in p for p in problems)
    problems.clear()
    cert.committed_fixtures([{**ok[0], "release_code": False}], problems)
    assert any("no release check ran and passed" in p for p in problems)


def test_ctest_progress_lines(cert) -> None:
    text = (
        "  1/3 Test #585: example.sssp ....   Passed    0.08 sec\n"
        " 39/3 Test  #40: Host.Huge ......***Skipped   0.07 sec\n"
        "  3/3 Test   #9: L.R<(anonymous namespace)::t<int,int>> ...***Failed  1.20 sec\n"
        "\x1b[32m  2/2 Test #1: A.B ..   Passed    0.01 sec\x1b[0m\n"
    )
    assert cert.parse_ctest_tests(text) == {
        "example.sssp": "Passed",
        "Host.Huge": "Skipped",
        "L.R<(anonymous namespace)::t<int,int>>": "Failed",
        "A.B": "Passed",
    }


def test_cbst_trace_sets_are_gated(cert, tmp_path: Path) -> None:
    """The CBST trace goldens of M8a are gated (M8a review): a set without a replay of the dynG
    store under its original's preset and under dynG's default fails the certificate; with both
    records it passes; a record of one is checked like any other. (Read as a 0.2 certificate:
    from 0.3 the hypergraph-id replays are required too, the next test.)"""
    import tomllib

    sets = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"]
    traces = [n for n, s in sets.items() if s.get("kind") == "cbst_trace"]
    assert traces
    results = tmp_path / "results"
    results.mkdir()
    problems: list[str] = []
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.2.0")}
    for n in traces:
        assert listed[n]["kind"] == "cbst_trace" and listed[n]["passed"] is False
        assert any(n in p and "no replay of the CBST store" in p for p in problems)
    assert listed["sssp"]["passed"] is False  # an algorithm set without replays still fails
    name = traces[0]
    original = cert.CBST_ORIGINAL_PRESET[sets[name]["reference"]]

    def write(preset: str, manifest: str) -> None:
        record = {
            "set": name,
            "port": {"commit": head()},
            "goldens": {"manifest_sha256": manifest},
            "configs": [f"device:{preset}/exact"],
            "matrix": {"g": {f"device:{preset}/exact": "equal"}},
            "passed": True,
        }
        (results / f"parity-{name}-device-{preset}.json").write_text(json.dumps(record))

    # Only the original's preset: the default's replay is missing.
    write(original, sets[name]["manifest_sha256"])
    problems = []
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.2.0")}
    assert listed[name]["passed"] is False
    assert any("dyng_default" in p for p in problems if name in p)
    # Both presets: the set passes.
    write("dyng_default", sets[name]["manifest_sha256"])
    problems = []
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.2.0")}
    assert listed[name]["passed"] is True, [p for p in problems if name in p]
    # A record of another manifest fails like any other.
    write("dyng_default", "0" * 64)
    problems = []
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.2.0")}
    assert listed[name]["passed"] is False
    assert any("replayed manifest" in p for p in problems)


def test_hypergraph_id_replays_are_required_from_0_3(cert, tmp_path: Path) -> None:
    """M8b: a 0.3 certificate needs the hypergraph container's replays of the CBST trace sets'
    batch records (the hypergraph-id schema) on every backend, and the line graph of
    ESCHER-GPU's sets on cuda; a 0.2 certificate does not; a failed replay does not count."""
    import tomllib

    sets = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"]
    assert set(cert.HYPERGRAPH_ID_REPLAYS) <= set(sets)
    results = tmp_path / "results"
    results.mkdir()
    commit = head()
    # Every CBST set has its store replays (M8a), none has a hypergraph-id replay yet.
    for name, spec in sets.items():
        if spec.get("kind") != "cbst_trace":
            continue
        for preset in (cert.CBST_ORIGINAL_PRESET[spec["reference"]], "dyng_default"):
            config = f"device:{preset}/exact"
            (results / f"parity-{name}-device-{preset}.json").write_text(
                json.dumps(
                    {
                        "set": name,
                        "port": {"commit": commit},
                        "goldens": {"manifest_sha256": spec["manifest_sha256"]},
                        "configs": [config],
                        "matrix": {"c": {config: "equal"}},
                        "passed": True,
                    }
                )
            )
    name = "cbst_trace_escher_gpu_test"

    def listed(version: str) -> tuple[dict, list[str]]:
        problems: list[str] = []
        out = {g["set"]: g for g in cert.golden_suites(results, commit, problems, version)}
        return out[name], [p for p in problems if name in p]

    entry, problems = listed("0.2.0")
    assert entry["passed"] is True and problems == []  # not required before 0.3
    entry, problems = listed("0.3.0")
    assert entry["passed"] is False
    assert any("hypergraph-id replay (ids) on sequential, openmp, cuda" in p for p in problems)
    assert any("(line_graph) on cuda" in p for p in problems)
    for backend in ("sequential", "openmp", "cuda"):
        _hypergraph_id_record(results, name, sets[name], commit, backend, "both")
    entry, problems = listed("0.3.0")
    assert entry["passed"] is False and len(problems) == 1 and "line_graph" in problems[0]
    _hypergraph_id_record(results, name, sets[name], commit, "cuda", "line_graph")
    entry, problems = listed("0.3.0rc1")
    assert entry["passed"] is True, problems
    assert entry["hypergraph_ids"] == {
        "ids": ["cuda", "openmp", "sequential"],
        "line_graph": ["cuda"],
    }
    # A replay without the host model, without verified goldens or (line graph) without the
    # oracle does not count (M8b review).
    for checks, feature, backend in (
        ({"model": False, "verified": True, "line_graph_oracle": True}, "both", "openmp"),
        ({"model": True, "verified": False, "line_graph_oracle": True}, "both", "openmp"),
        ({"model": True, "verified": True, "line_graph_oracle": False}, "line_graph", "cuda"),
    ):
        _hypergraph_id_record(results, name, sets[name], commit, backend, feature, checks)
        entry, problems = listed("0.3.0")
        assert entry["passed"] is False, checks
        assert any(f"on {backend}" in p for p in problems), (checks, problems)
        _hypergraph_id_record(results, name, sets[name], commit, backend, feature)
    # A failed replay does not count (and fails the set).
    failed = results / f"parity-{name}-hypergraph_ids-openmp-both.json"
    doc = json.loads(failed.read_text())
    doc["passed"] = False
    failed.write_text(json.dumps(doc))
    entry, problems = listed("0.3.0")
    assert entry["passed"] is False
    assert any("(ids) on openmp" in p for p in problems)


# --- R020: the mosp suite (ipdps25_dynamosp_mosp) and its gates in the certificate ----------------

MOSP_SUITE = REPO / "benchmarks/paper/ipdps25_dynamosp_mosp.yaml"


def test_the_mosp_suite_covers_the_m7_gates(bench) -> None:
    suite = bench.load_suite(MOSP_SUITE)
    assert bench.validate(suite) == []
    assert suite["algorithm"] == "mosp" and suite["harness"]["tool"] == "perf_ab.mosp"
    assert suite["harness"]["region_map"] == "parity/timed_regions/mosp.toml"
    # The graphs and batches of M7's gates (parity/results/M7.md sections 6 and 8), K = 3.
    assert [d["name"] for d in suite["datasets"]] == [
        "roadNet-PA",
        "roadNet-CA",
        "rgg",
        "road_usa_g",
    ]
    assert [b["name"] for b in suite["batches"]] == ["safe50k", "unsafe50k", "local10k"]
    assert all(b["seed"] == 777 for b in suite["batches"])
    assert suite["graph"]["num_weights"] == 3
    # Its inputs are the sssp suite's (the same prepared directories).
    sosp = bench.load_suite(REPO / "benchmarks/paper/ipdps25_dynamosp_sosp.yaml")
    assert {d["name"]: d["sha256"] for d in suite["datasets"]} == {
        d["name"]: d["sha256"] for d in sosp["datasets"]
    }
    # The gates: "(a) compute" and "(b) end to end" on both backends, the device memory on cuda.
    gated = {
        (name, backend)
        for name, metric in suite["metrics"].items()
        for backend in ("openmp", "cuda")
        if backend in metric and bench.metric_gate(metric, backend) != "none"
    }
    assert gated == {
        ("compute", "openmp"),
        ("compute", "cuda"),
        ("end_to_end", "openmp"),
        ("end_to_end", "cuda"),
        ("device_memory", "cuda"),
    }
    readings = {r["name"]: r for r in suite["readings"]}
    assert readings["cuda"]["clocks"] == "boost" and readings["cuda"]["gated"]  # ADR 0018
    assert readings["cuda-default-clocks"]["clocks"] == "none"
    assert not readings["cuda-default-clocks"]["gated"]
    assert {"MOSP-OpenMP", "MOSP-CUDA"} == set(suite["baselines"])


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s["harness"].update(tool="perf_ab.sssp"), "harness.tool"),
        (lambda s: s["harness"].update(region_map="parity/timed_regions/sssp.toml"), "region_map"),
        (lambda s: s["metrics"]["compute"].update(gate="none"), "region map says"),
        (lambda s: s["metrics"]["compute"]["cuda"].update(port=["sssp.engine"]), "port stages"),
        (lambda s: s["metrics"].pop("end_to_end"), "is not a metric"),
        (lambda s: s["graph"].pop("num_weights"), "num_weights"),
        (lambda s: s["graph"].update(preferences=[1, 2, 3]), "preferences"),
        (lambda s: s["backends"]["cuda"].update(engine="fast"), "engine"),
        (lambda s: s["batches"][0].update(directory="x"), "directory"),
        (lambda s: s["datasets"][0].update(hops=1), "hops"),
        (lambda s: s["baselines"]["MOSP-CUDA"].update(variant="patched"), "unpatched"),
        (lambda s: s["readings"][2].update(gated=True), "not gated"),
    ],
)
def test_validate_finds_what_differs_in_the_mosp_suite(bench, edit, message: str) -> None:
    suite = bench.load_suite(MOSP_SUITE)
    edit(suite)
    errors = bench.validate(suite)
    assert any(message in e for e in errors), errors


def test_plan_of_the_mosp_suite(bench, tmp_path: Path) -> None:
    suite = bench.load_suite(MOSP_SUITE)
    jobs = bench.plan(suite, records=tmp_path, build_root=Path("/b"))
    assert len(jobs) == 4 * 4  # four readings x four graphs
    by = {(j["reading"], j["dataset"]): j["argv"] for j in jobs}
    omp = by[("openmp", "roadNet-PA")]
    assert omp[2:4] == ["mosp", "--backend"] and omp[omp.index("--backend") + 1] == "openmp"
    assert "/b/parity/tools/compat/dyng-compat-mosp" in omp
    assert omp[omp.index("--threads") + 1] == "28" and "--lock-clocks" not in omp
    assert omp[omp.index("--batches") + 1] == "safe50k,unsafe50k,local10k"
    cuda = by[("cuda", "road_usa_g")]
    assert cuda[2] == "mosp" and "/b/parity-cuda/tools/compat/dyng-compat-mosp" in cuda
    assert cuda[cuda.index("--runs") + 1] == "11" and cuda[cuda.index("--gpu") + 1] == "0"
    assert cuda[cuda.index("--lock-clocks") + 1] == "boost"
    assert cuda[cuda.index("--cuda-engine") + 1] == "automatic"
    default = by[("cuda-default-clocks", "rgg")]
    assert default[default.index("--lock-clocks") + 1] == "none"
    assert default[default.index("--runs") + 1] == "21"
    memory = by[("cuda-memory", "roadNet-CA")]
    assert memory[2] == "memory" and "--mosp" in memory
    assert memory[memory.index("--gpu") + 1] == "1"
    assert all(j["lock"] == "exclusive (the harness)" for j in jobs)
    # perf_ab.py accepts every planned command line (parsing only: --help of each subcommand
    # lists the options the plan uses).
    for sub, options in (
        ("mosp", ["--cuda-engine", "--lock-clocks", "--batches", "--threads", "--json"]),
        ("memory", ["--mosp", "--batches", "--gpu", "--json"]),
    ):
        proc = subprocess.run(
            [sys.executable, REPO / "parity/perf_ab.py", sub, "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        for option in options:
            assert option in proc.stdout, (sub, option)


def test_summary_of_the_mosp_suite(bench, tmp_path: Path) -> None:
    suite = bench.load_suite(MOSP_SUITE)
    jobs = _write_sssp_records(bench, suite, tmp_path / "rec", head())
    out = tmp_path / "out"
    summary = bench.summarize(suite, jobs, out=out, version="t", records=tmp_path, inputs=None)
    assert summary["verdict"]["passed"], summary["verdict"]
    assert summary["algorithm"] == "mosp" and (out / "ipdps25_dynamosp_mosp.json").is_file()
    regions = {g["region"] for r in summary["readings"] for g in r["regions"]}
    assert {"compute", "end_to_end", "device_memory"} <= regions

    # Records that are not the suite's are refused, one finding each.
    def refused(job: dict, edit, text: str) -> None:
        path = Path(job["record"])
        good = path.read_text()
        rec = json.loads(good)
        edit(rec)
        path.write_text(json.dumps(rec))
        got = bench.summarize(
            suite, jobs, out=tmp_path / "o", version="t", records=tmp_path, inputs=None
        )
        path.write_text(good)
        problems = " ".join(got["verdict"]["problems"])
        assert not got["verdict"]["passed"] and text in problems, problems

    cuda = next(j for j in jobs if j["reading"] == "cuda")
    memory = next(j for j in jobs if j["reading"] == "cuda-memory")
    omp = next(j for j in jobs if j["reading"] == "openmp")
    refused(cuda, lambda r: r.update(algorithm="sssp"), "a record of algorithm 'sssp'")
    refused(memory, lambda r: r.pop("algorithm"), "a record of algorithm None")  # no --mosp
    refused(cuda, lambda r: r.update(objectives=2), "2 objectives")
    refused(omp, lambda r: r.update(preferences="1,2,3"), "preferences '1,2,3'")
    refused(
        omp,
        lambda r: r["protocol"].update(outputs="--no-output on both"),
        "writes every output file",
    )
    refused(cuda, lambda r: r.update(cuda_engine="operators"), "CUDA engine 'operators'")
    refused(cuda, lambda r: r["results"].pop("local10k"), "the suite's are")
    refused(
        omp,
        lambda r: r["results"]["safe50k"]["invalidated"].update(equal_in_every_sample=False),
        "invalidated counts are not equal",
    )
    refused(
        cuda,
        lambda r: r["protocol"].update(gpu_clocks={"control": "none"}),
        "clocks 'none'",
    )


def test_release_series_and_the_required_suites(cert) -> None:
    assert cert.release_series("0.2.0rc1") == (0, 2)
    assert cert.release_series("0.1.0") == (0, 1)
    assert cert.release_series("0.10.3.dev0") == (0, 10)
    assert cert.release_series("v1.0") == (1, 0)
    assert cert.release_series("t") is None
    assert set(cert.required_suites("0.1.0")) == {"ipdps25_dynamosp_sosp", "ieee_tc_dyntrucy"}
    for version in ("0.2.0rc1", "0.2.0", "0.3.0", "t"):
        assert "ipdps25_dynamosp_mosp" in cert.required_suites(version), version
    # The ESCHER suites of triad_count from 0.3 (M9a): a 0.2 certificate does not need them.
    escher = {"ipdps26_escher", "tkde26_escher_plus"}
    assert not escher & set(cert.required_suites("0.2.0"))
    for version in ("0.3.0rc1", "0.3.0", "t"):
        assert escher <= set(cert.required_suites(version)), version
    # Every required suite is a committed suite of its algorithm.
    for name, spec in cert.REQUIRED_SUITES.items():
        path = REPO / "benchmarks" / "paper" / f"{name}.yaml"
        assert path.is_file() and f"algorithm: {spec['algorithm']}\n" in path.read_text()


def test_the_committed_summaries_read_every_gated_metric(cert) -> None:
    """The coverage rule holds for the summaries the 0.1.0 certificate was written from."""
    results = REPO / "benchmarks" / "results" / "0.1.0"
    for name in ("ipdps25_dynamosp_sosp", "ieee_tc_dyntrucy"):
        doc = json.loads((results / f"{name}.json").read_text())
        rows = [
            {"backend": r["backend"], "region": g["region"]}
            for r in doc["readings"]
            if r["gated"]
            for g in r["regions"]
            if "gate" in g
        ]
        assert cert.missing_gates(REPO / doc["suite_file"], rows) == [], name
        assert cert.missing_gates(REPO / doc["suite_file"], []) != []


def test_a_02_certificate_needs_the_mosp_gates(cert, bench, tmp_path: Path, fixed_driver) -> None:
    commit = head()
    results = _full_results(cert, bench, tmp_path, commit)
    log = tmp_path / "asan.log"
    log.write_text(_ctest_log(_fixture_test_names(cert)))
    base = ["--results", str(results)]
    check = ["check", *base, "--name", "asan", "--command", "ctest", "--ctest-log", str(log)]
    assert cert.main(check) == 0
    _release_checks(cert, results, tmp_path, commit)

    def write(version: str) -> int:
        return cert.main(["write", *base, "--version", version, "--allow-dirty"])

    def problems() -> str:
        return " ".join(json.loads((results / "parity.json").read_text())["verdict"]["problems"])

    assert write("0.2.0rc1") == 0, problems()
    doc = json.loads((results / "parity.json").read_text())
    assert doc["required_suites"]["ipdps25_dynamosp_mosp"] == {"algorithm": "mosp", "since": "0.2"}
    mosp = next(s for s in doc["performance"] if s["suite"] == "ipdps25_dynamosp_mosp")
    assert mosp["passed"] and {r["region"] for r in mosp["gate_table"]} == {
        "compute",
        "end_to_end",
        "device_memory",
    }
    assert "`ipdps25_dynamosp_mosp`" in (results / "README.md").read_text()

    # A gated metric without a reading: the memory reading's regions dropped from the summary.
    path = results / "ipdps25_dynamosp_mosp.json"
    full = path.read_text()
    cut = json.loads(full)
    for r in cut["readings"]:
        if r["reading"] == "cuda-memory":
            r["regions"] = []
    path.write_text(json.dumps(cut))
    assert write("0.2.0rc1") == 1
    assert "no gated reading of device_memory (cuda)" in problems()

    # Without the mosp suite: a 0.2 certificate fails, a 0.1 certificate does not need it.
    path.unlink()
    for record in results.glob("ipdps25_dynamosp_mosp-*.json"):
        record.unlink()
    assert write("0.2.0rc1") == 1
    assert "no performance suite of mosp" in problems()
    assert write("0.1.0") == 0, problems()
    path.write_text(full)


def test_a_narrowed_mosp_execution(bench, tmp_path: Path) -> None:
    """--datasets and --batches narrow a mosp execution (the plumbing check of R020): the
    harness gets the batches, the summary is partial and checks the records against them."""
    suite = bench.load_suite(MOSP_SUITE)
    jobs = bench.plan(
        suite,
        records=tmp_path / "rec",
        build_root=Path("/b"),
        readings=["openmp", "cuda"],
        datasets=["roadNet-PA"],
        batches=["safe50k"],
    )
    assert [(j["reading"], j["dataset"]) for j in jobs] == [
        ("openmp", "roadNet-PA"),
        ("cuda", "roadNet-PA"),
    ]
    assert all(j["argv"][j["argv"].index("--batches") + 1] == "safe50k" for j in jobs)
    with pytest.raises(bench.SuiteError, match="unknown batches"):
        bench.plan(suite, records=tmp_path, build_root=Path("/b"), batches=["big"])
    cc = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
    with pytest.raises(bench.SuiteError, match="--batches"):
        bench.plan(cc, records=tmp_path, build_root=Path("/b"), batches=["u25k"])
    full = _write_sssp_records(bench, suite, tmp_path / "rec", head())
    for job in full:
        path = Path(job["record"])
        rec = json.loads(path.read_text())
        rec["results"] = {"safe50k": rec["results"]["safe50k"]}
        path.write_text(json.dumps(rec))
    out = tmp_path / "out"
    narrowed = {"readings": ["openmp", "cuda"], "datasets": ["roadNet-PA"], "batches": ["safe50k"]}
    summary = bench.summarize(
        suite, jobs, out=out, version="t", records=tmp_path / "rec", inputs=None, partial=narrowed
    )
    assert summary["verdict"]["passed"], summary["verdict"]
    assert (out / "ipdps25_dynamosp_mosp.partial.json").is_file()
    # The same one-batch records do not make the whole suite's summary.
    whole = bench.summarize(
        suite, full, out=tmp_path / "o2", version="t", records=tmp_path / "rec", inputs=None
    )
    assert not whole["verdict"]["passed"]
    assert "the suite's are" in " ".join(whole["verdict"]["problems"])
    with pytest.raises(SystemExit) as exc:
        bench.main(["summarize", str(MOSP_SUITE), "--batches", "safe50k", "--version", "t"])
    assert exc.value.code == 2


def test_a_diagnostic_execution_keeps_contaminated_rounds_flagged(bench, tmp_path: Path) -> None:
    """--keep-contaminated (a busy machine): the harness keeps the rounds it would repeat; the
    summary flags them in a diagnostic execution and refuses such a record otherwise."""
    suite = bench.load_suite(MOSP_SUITE)
    narrowed = {"datasets": ["roadNet-PA"], "keep_contaminated": True}
    jobs = bench.plan(
        suite,
        records=tmp_path / "rec",
        build_root=Path("/b"),
        datasets=["roadNet-PA"],
        keep_contaminated=True,
    )
    for job in jobs:
        assert job["diagnostic"]
        assert ("--keep-contaminated" in job["argv"]) == (job["kind"] == "run")
    with pytest.raises(bench.SuiteError, match="--keep-contaminated"):
        cc = bench.load_suite(REPO / "benchmarks/paper/ieee_tc_dyntrucy.yaml")
        bench.plan(cc, records=tmp_path, build_root=Path("/b"), keep_contaminated=True)
    _write_sssp_records(bench, suite, tmp_path / "rec", head())
    omp = next(j for j in jobs if j["reading"] == "openmp")
    rec = json.loads(Path(omp["record"]).read_text())
    rec["protocol"]["contamination_monitor"] = "...; contaminated rounds kept and flagged"
    rec["results"]["safe50k"]["monitor"] = {
        "rounds": [{"original": {"reasons": ["foreign CPU load 24 cores > 2.0"]}, "port": {}}],
        "rejected": [],
    }
    Path(omp["record"]).write_text(json.dumps(rec))
    out = tmp_path / "out"
    summary = bench.summarize(
        suite, jobs, out=out, version="t", records=tmp_path / "rec", inputs=None, partial=narrowed
    )
    assert summary["verdict"]["passed"], summary["verdict"]
    assert summary["verdict"]["contaminated"] == [
        "openmp: safe50k (1 runs above the foreign-load threshold)"
    ]
    plain = bench.plan(suite, records=tmp_path / "rec", build_root=Path("/b"))
    summary = bench.summarize(
        suite, plain, out=tmp_path / "o2", version="t", records=tmp_path / "rec", inputs=None
    )
    assert not summary["verdict"]["passed"]
    assert "not a gate reading" in " ".join(summary["verdict"]["problems"])


# --- The release checks, the gate rows, the packaging scope (R020 review) ------------------------


@pytest.fixture()
def certified(cert, bench, tmp_path: Path, fixed_driver):
    """A results directory that certifies, and helpers to write it and read its problems."""
    commit = head()
    results = _full_results(cert, bench, tmp_path, commit)
    log = tmp_path / "asan.log"
    log.write_text(_ctest_log(_fixture_test_names(cert)))
    base = ["--results", str(results)]
    asan = ["check", *base, "--name", "asan", "--command", "x", "--ctest-log", str(log)]
    assert cert.main(asan) == 0
    _release_checks(cert, results, tmp_path, commit)

    def write(version: str = "0.2.0rc1") -> int:
        return cert.main(["write", *base, "--version", version, "--allow-dirty"])

    def problems() -> str:
        return " ".join(json.loads((results / "parity.json").read_text())["verdict"]["problems"])

    assert write() == 0, problems()
    return results, write, problems


def _edit_checks(results: Path, edit) -> str:
    path = results / "checks.json"
    full = path.read_text()
    doc = json.loads(full)
    edit(doc["checks"])
    path.write_text(json.dumps(doc))
    return full


def test_a_certificate_needs_every_release_check(cert, certified) -> None:
    results, write, problems = certified
    for name in cert.required_checks("0.2.0rc1"):

        def drop(cs: list[dict], n: str = name) -> None:
            cs.remove(next(c for c in cs if c["name"] == n))

        full = _edit_checks(results, drop)
        assert write() == 1, name
        assert f"check {name} is missing (required from" in problems()
        (results / "checks.json").write_text(full)
    # 0.1 needed mutation-sssp, not tsan-openmp, mutation-goldens or api-check.
    assert set(cert.required_checks("0.1.0")) == {
        "asan",
        "tsan",
        "mutation-ctests",
        "mutation-sssp",
        "distributions",
        "check-parity",
        "gpu_local",
    }
    assert "mutation-goldens" in cert.required_checks("0.2.0") and "mutation-sssp" not in (
        cert.required_checks("0.3.0")
    )

    # A record of a narrower scope than its check needs, and a gpu_local without racecheck.
    def narrow(cs: list[dict]) -> None:
        next(c for c in cs if c["name"] == "distributions")["scope"] = "tests"
        del next(c for c in cs if c["name"] == "gpu_local")["steps"]["racecheck"]

    full = _edit_checks(results, narrow)
    assert write() == 1
    text = problems()
    assert "check distributions: recorded with scope tests, it needs at least packaging" in text
    assert "check gpu_local: the step racecheck did not pass or was not run" in text
    (results / "checks.json").write_text(full)
    assert write() == 0, problems()


def test_the_golden_mutation_record_is_read(cert, certified) -> None:
    results, write, problems = certified
    path = results / "mutation-goldens.json"
    full = path.read_text()
    cases = [
        (lambda d: d["results"].pop(), "is missing"),  # the last mutation dropped
        (lambda d: d["results"][1].update(detected=False, verdict="NOT DETECTED"), "NOT DETECTED"),
        (lambda d: d["results"][0].update(verdict="FAILED"), "the control did not pass"),
        (
            lambda d: d["results"][2]["goldens"].update(manifest_sha256="0" * 64),
            "is not one of parity/goldens.toml",
        ),
        (lambda d: d.update(commit="0" * 40), "does not have the release's library sources"),
        # a replay of a golden suite without its manifest (only the GoogleTest suites have none)
        (lambda d: d["results"][2].pop("goldens"), "golden manifest None is not one of"),
    ]
    for edit, message in cases:
        doc = json.loads(full)
        edit(doc)
        path.write_text(json.dumps(doc))
        assert write() == 1, message
        assert "check mutation-goldens: " in problems() and message in problems(), problems()
    path.write_text(full)
    assert write() == 0, problems()


def test_the_gtest_mutation_rows_need_the_release_tests(
    cert, certified, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR 0061 section 4: the rows of a GoogleTest suite of mutate.py (cbst and the suites after
    it) are detected by the C++ tests, so a golden-mutation record with such rows needs the
    release's cpp/tests as well as its library; a record without them is bound to the library
    only, and a difference outside cpp/tests (README.md, the examples) does not matter."""
    results, write, problems = certified
    path = results / "mutation-goldens.json"
    commit = head()
    mutate = load("parity/mutate.py")
    gtest = {name for name, spec in mutate.SUITES.items() if spec.get("runner") == "gtest"}
    assert "cbst" in gtest and cert.GTEST_MUTATION_PATHS == ["cpp/tests"]
    # The real diff: the last commit that changed cpp/tests differs from its parent there only.
    last = subprocess.check_output(
        ["git", "-C", REPO, "log", "-1", "--format=%H", "--", "cpp/tests"], text=True
    ).strip()
    parent = subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", f"{last}~1"], text=True
    ).strip()
    changed = cert.paths_diff(parent, last, cert.GTEST_MUTATION_PATHS)
    assert changed and all(f.startswith("cpp/tests/") for f in changed)
    assert cert.paths_diff(commit, commit, cert.GTEST_MUTATION_PATHS) == []
    real = cert.paths_diff

    def other_tests(c: str, h: str, paths: list[str]) -> list[str] | None:
        if paths == cert.GTEST_MUTATION_PATHS:
            return ["cpp/tests/hypergraph/escher/cbst_host_test.cpp"]
        return real(c, h, paths)

    monkeypatch.setattr(cert, "paths_diff", other_tests)
    message = "does not have the release's C++ test suites (cpp/tests)"
    assert write() == 1 and message in problems() and "cbst" in problems(), problems()
    doc = json.loads(path.read_text())
    doc["results"] = [r for r in doc["results"] if r.get("suite") not in gtest]
    doc["results"][0]["replays"] = [
        r for r in doc["results"][0]["replays"] if r.get("suite") not in gtest
    ]
    stripped = results / "mutation-goldens-no-gtest.json"
    stripped.write_text(json.dumps(doc))
    found = cert.golden_mutations(stripped, commit, results)
    assert found and not any(message in p for p in found), found  # missing rows, not the tests
    monkeypatch.setattr(cert, "paths_diff", real)
    stripped.unlink()
    assert write() == 0, problems()


def test_the_gate_rows_are_recomputed(cert, certified) -> None:
    results, write, problems = certified
    path = results / "ipdps25_dynamosp_mosp.json"
    full = path.read_text()

    def first_gated(doc: dict) -> dict:
        return next(g for r in doc["readings"] if r["gated"] for g in r["regions"] if "gate" in g)

    # A ratio above its gate, the flag and the verdict left as they were.
    doc = json.loads(full)
    row = first_gated(doc)
    row["port_ms"] = row["original_ms"] * 1.3
    row["ratio"] = 1.3
    path.write_text(json.dumps(doc))
    assert write() == 1
    text = problems()
    assert "differ from its record's" in text and "within_gate True but 1.3000" in text
    assert "the verdict says passed with 1 gated rows above the gate" in text
    perf = json.loads((results / "parity.json").read_text())["performance"]
    mosp = next(s for s in perf if s["suite"] == "ipdps25_dynamosp_mosp")
    assert not mosp["passed"] and any(r["within_gate"] is False for r in mosp["gate_table"])
    # A ratio that is not port / original; a gate that is not the suite's.
    doc = json.loads(full)
    first_gated(doc)["ratio"] *= 0.5
    path.write_text(json.dumps(doc))
    assert write() == 1 and "is not" in problems() and "ratio" in problems()
    doc = json.loads(full)
    first_gated(doc)["gate"] = 2.0
    path.write_text(json.dumps(doc))
    assert write() == 1 and "the suite's tolerance says" in problems()
    path.write_text(full)
    assert write() == 0, problems()


def test_measured_commits_are_full_and_once(cert, certified) -> None:
    results, write, problems = certified
    commit = head()
    checks = json.loads((results / "checks.json").read_text())
    checks["checks"][0]["commit"] = commit[:7]  # a short SHA of the same commit
    (results / "checks.json").write_text(json.dumps(checks))
    assert write() == 0, problems()
    assert json.loads((results / "parity.json").read_text())["measured_commits"] == [commit]


PLUGIN_INPUTS = [
    "ci/plugin_wheels.sh",
    "ci/plugin_pyproject.py",
    "ci/plugin_smoke.py",
    "ci/cuda_toolkit.py",
    "ci/cuda_toolkits.toml",
    "ci/cibw_plugin.sh",
    "ci/cibw_plugin_repair.sh",
    "ci/cibuildwheel-plugin.toml",
    "THIRD_PARTY_LICENSES_CUDA.txt",
]


def test_the_packaging_scope_covers_the_sdist_and_the_plugins(cert) -> None:
    members = cert.sdist_members()
    assert "pyproject.toml" in members and "python/dyng/__init__.py" in members
    assert "docs/references.bib" in members and not any(m.startswith("docs/adr") for m in members)
    assert not any(m.startswith(("ci/", "parity/", ".github/")) for m in members)
    outside = [m for m in members if not cert.in_paths(m, cert.PACKAGING_PATHS)]
    assert outside == [], f"sdist members outside the packaging scope: {outside}"
    tracked = set(subprocess.check_output(["git", "-C", REPO, "ls-files"], text=True).split())
    for path in PLUGIN_INPUTS:
        assert path in tracked and cert.in_paths(path, cert.PACKAGING_PATHS), path
    # A change of a plugin input invalidates a distributions record: M6a's toolkit pins and the
    # plugin_wheels.sh preflight touch no other file of the scope.
    for commit in ("d670ada", "1be9f2c"):
        if cert.resolve(commit) is None:
            continue  # a shallow clone
        parent = cert.git("rev-parse", commit + "~1")
        changed = cert.code_diff(parent, commit, "packaging")
        assert changed and set(changed) <= set(PLUGIN_INPUTS), changed
        assert cert.same_code(parent, commit, "packaging") is False


# --- M9a: the ESCHER suites (triad_count) and their gates in the 0.3 certificate -----------------

ESCHER_SUITE = REPO / "benchmarks/paper/ipdps26_escher.yaml"
ESCHER_PLUS_SUITE = REPO / "benchmarks/paper/tkde26_escher_plus.yaml"


def test_the_escher_suites_cover_the_m9_gates(bench) -> None:
    import tomllib

    timing = bench.load_suite(ESCHER_SUITE)
    memory = bench.load_suite(ESCHER_PLUS_SUITE)
    assert bench.validate(timing) == [] and bench.validate(memory) == []
    for suite in (timing, memory):
        assert (
            suite["algorithm"] == "triad_count"
            and suite["harness"]["tool"] == "perf_ab.triad_count"
        )
        assert suite["baselines"] == {
            "ESCHER-GPU": {
                "commit": "abcf9b2bd42f896ec3103a6835edf713ed5e416c",
                "variant": "unpatched",
                "backend": "cuda",
            }
        }
        # coauth-DBLP by its file's SHA-256, syn1M / syn2M by A's parameters and their rows' digest
        # per seed: those of the golden set (step 1).
        data = {d["name"]: d for d in suite["datasets"]}
        assert data["coauth"]["sha256"] == (
            "65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6"
        )
        inputs = tomllib.loads((REPO / "parity/goldens.toml").read_text())["sets"][TRIAD_SET][
            "inputs"
        ]
        for name, params in (
            ("syn1m", [1000000, 10, 1, 333333]),
            ("syn2m", [2000000, 10, 1, 666666]),
        ):
            assert data[name]["generator"]["parameters"] == params
            for s in (1, 2, 3):
                assert data[name]["sha256"][f"s{s}"] == inputs[f"{name}_s{s}"]["loaded_rows_sha256"]

    def gated(suite: dict) -> set[str]:
        return {n for n, m in suite["metrics"].items() if bench.metric_gate(m, "cuda") != "none"}

    # ipdps26: update of every batch, the full count and end to end (criterion 12).
    assert gated(timing) == {"update[b1]", "update[b2]", "update[b3]", "full_count", "end_to_end"}
    readings = {r["name"]: r for r in timing["readings"]}
    assert readings["cuda-coauth"]["cases"] == [f"coauth_b50k_s{s}" for s in (1, 2, 3)]
    assert readings["cuda-coauth"]["clocks"] == "base" and readings["cuda-coauth"]["runs"] >= 11
    assert readings["cuda"]["cases"] == [
        f"{d}_b50k_s{s}" for d in ("syn1m", "syn2m") for s in (1, 2, 3)
    ]
    assert readings["cuda"]["runs"] >= 21 and readings["cuda"]["clocks"] in ("base", "boost")
    assert all(readings[n]["gated"] for n in ("cuda-coauth", "cuda"))
    assert readings["cuda-default-clocks"]["clocks"] == "none"
    assert not readings["cuda-default-clocks"]["gated"]
    assert readings["cuda-batch-sizes"]["cases"] == ["coauth_b100k_s1", "coauth_b200k_s1"]
    assert not readings["cuda-batch-sizes"]["gated"]
    ref = timing["paper_reference"]
    assert ref["update_per_batch_s"] == {
        "coauth_b50k": 0.546,
        "coauth_b100k": 1.033,
        "coauth_b200k": 1.936,
        "syn1m_b50k": 0.127,
        "syn2m_b50k": 0.151,
    }
    assert ref["full_count_s"] == {"coauth": 5.22, "syn1m": 0.47, "syn2m": 0.95}
    assert {k: ref["end_to_end_s"][k] for k in ("coauth_b50k", "syn1m_b50k", "syn2m_b50k")} == {
        "coauth_b50k": 9.1,
        "syn1m_b50k": 2.2,
        "syn2m_b50k": 4.1,
    }
    # tkde26: the store accounting and the nsys peak (criterion 13); types and batch sizes recorded.
    assert gated(memory) == {"store_memory", "device_memory"}
    readings = {r["name"]: r for r in memory["readings"]}
    assert readings["cuda-memory"]["gated"] and readings["cuda-memory"]["gpu"] == 1
    assert readings["cuda-memory"]["cases"] == [
        f"{d}_b50k_s1" for d in ("coauth", "syn1m", "syn2m")
    ]
    assert readings["cuda-types"]["patterns"] == ["type1", "type2", "type3"]
    assert not readings["cuda-types"]["gated"] and not readings["cuda-memory-batch-sizes"]["gated"]
    assert memory["paper_reference"]["store_gb"] == {"coauth": 1.83, "syn1m": 0.67, "syn2m": 1.34}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s["datasets"][0].update(sha256="0" * 64), "pinned file's"),
        (lambda s: s["datasets"][1]["sha256"].update(s1="0" * 64), "golden set's"),
        (lambda s: s["datasets"][1]["generator"].update(parameters=[1, 2, 3, 4]), "parameters"),
        (lambda s: s["readings"][0]["cases"].append("coauth_b50k_s9"), "not a case of the golden"),
        (lambda s: s["readings"][0].update(clocks="boost"), "base lock (ADR 0021)"),
        (lambda s: s["readings"][1].update(patterns=["type1"]), "h_motif"),
        (lambda s: s["readings"][1].update(clocks="none"), "not gated"),
        (lambda s: s["metrics"].pop("update[b3]"), "update[b3] (cuda) of the region map is not"),
        (lambda s: s["metrics"]["full_count"].update(gate="none"), "region map says"),
        (lambda s: s["metrics"]["update[b1]"]["cuda"].update(port=["update_ms"]), "port stages"),
        (lambda s: s["batches"][0].update(seeds=None), "seeds"),
        (lambda s: s["harness"].update(tool="perf_ab.cycle_count"), "harness.tool"),
    ],
)
def test_validate_finds_what_differs_in_the_escher_suite(bench, edit, message: str) -> None:
    suite = bench.load_suite(ESCHER_SUITE)
    edit(suite)
    errors = bench.validate(suite)
    assert any(message in e for e in errors), errors


def test_the_memory_suite_gates_memory_only(bench) -> None:
    suite = bench.load_suite(ESCHER_PLUS_SUITE)
    for metric in ("compute", "update[b1]"):
        bad = copy.deepcopy(suite)
        bad["metrics"][metric] = {"gate": "memory", "cuda": {"original": "x", "port": "y"}}
        assert any("have the memory gate" in e for e in bench.validate(bad))
    # A gated timing reading makes every gated region of the map a required metric.
    bad = copy.deepcopy(suite)
    bad["readings"][2].update(gated=True, patterns=["main"])
    assert any("is not a metric" in e for e in bench.validate(bad))


def test_plan_of_the_escher_suites(bench, tmp_path: Path) -> None:
    timing = bench.load_suite(ESCHER_SUITE)
    jobs = bench.plan(timing, records=tmp_path, build_root=Path("/b"))
    by = {j["reading"]: j for j in jobs}
    assert set(by) == {"cuda-coauth", "cuda", "cuda-default-clocks", "cuda-batch-sizes"}
    coauth = by["cuda-coauth"]["argv"]
    assert coauth[2:4] == ["triad_count", "run"] and "/b/parity-cuda" in coauth
    assert coauth[coauth.index("--lock-clocks") + 1] == "base"
    assert coauth[coauth.index("--rounds") + 1] == "11" and coauth[coauth.index("--gpu") + 1] == "0"
    assert coauth[coauth.index("--patterns") + 1] == "main"
    syn = by["cuda"]["argv"]
    assert syn[syn.index("--rounds") + 1] == "21"
    assert by["cuda"]["cases"] == [f"{d}_b50k_s{s}" for d in ("syn1m", "syn2m") for s in (1, 2, 3)]
    assert all(j["lock"] == "exclusive (the harness)" for j in jobs)
    narrowed = bench.plan(timing, records=tmp_path, build_root=Path("/b"), datasets=["syn2m"])
    assert [j["cases"] for j in narrowed] == [
        ["syn2m_b50k_s1", "syn2m_b50k_s2", "syn2m_b50k_s3"],
        ["syn2m_b50k_s1"],
    ]
    with pytest.raises(bench.SuiteError):
        bench.plan(timing, records=tmp_path, build_root=Path("/b"), batches=["b50k"])
    memory = bench.load_suite(ESCHER_PLUS_SUITE)
    jobs = {j["reading"]: j for j in bench.plan(memory, records=tmp_path, build_root=Path("/b"))}
    mem = jobs["cuda-memory"]
    assert mem["argv"][2:4] == ["triad_count", "memory"] and mem["lock"] == "shared"
    assert mem["argv"][mem["argv"].index("--gpu") + 1] == "1"
    types = jobs["cuda-types"]["argv"]
    assert types[types.index("--patterns") + 1] == "type1,type2,type3"
    assert types[types.index("--gpu") + 1] == "0"
    # perf_ab.py triad_count accepts every planned command line (--help lists the options).
    for sub, options in (
        ("run", ["--cases", "--patterns", "--rounds", "--lock-clocks", "--gpu", "--build"]),
        ("memory", ["--cases", "--patterns", "--gpu", "--build", "--json"]),
        ("kernels", ["--cases", "--rounds", "--gpu", "--json"]),
    ):
        proc = subprocess.run(
            [sys.executable, REPO / "parity/perf_ab.py", "triad_count", sub, "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        for option in options:
            assert option in proc.stdout, (sub, option)


def test_summary_of_the_escher_suites(bench, tmp_path: Path) -> None:
    """The summaries of both suites pass on good records, and a record is refused when a batch is
    missing, it has too few rounds, a contaminated round was kept, the contamination audit flags
    it, the guard did not pass, the clock lock is not the reading's, an input digest is not the
    suite's, or the store accounting misses a gated point."""
    timing = bench.load_suite(ESCHER_SUITE)
    memory = bench.load_suite(ESCHER_PLUS_SUITE)
    jobs = _write_triad_records(bench, timing, tmp_path / "rec", head())
    out = tmp_path / "out"
    summary = bench.summarize(timing, jobs, out=out, version="t", records=tmp_path, inputs=None)
    assert summary["verdict"]["passed"], summary["verdict"]
    rows = {
        (r["reading"], g["case"], g["region"]) for r in summary["readings"] for g in r["regions"]
    }
    assert ("cuda-coauth", "coauth_b50k_s2/main", "update[b3]") in rows
    assert summary["verdict"]["gated_regions"] == (3 + 6) * 5
    mjobs = _write_triad_records(bench, memory, tmp_path / "mrec", head())
    msum = bench.summarize(memory, mjobs, out=out, version="t", records=tmp_path, inputs=None)
    assert msum["verdict"]["passed"], msum["verdict"]
    gated = {(g["case"], g["region"]) for r in msum["readings"] if r["gated"] for g in r["regions"]}
    assert ("coauth_b50k_s1/main", "store_memory[b2]") in gated
    assert ("syn1m_b50k_s1/main", "store_memory[b3]") in gated
    assert ("syn1m_b50k_s1/main", "store_memory[b2]") not in gated  # A's print points only
    assert ("syn2m_b50k_s1/main", "device_memory") in gated

    def refused(suite: dict, js: list, reading: str, edit, text: str) -> None:
        job = next(j for j in js if j["reading"] == reading)
        path = Path(job["record"])
        good = path.read_text()
        rec = json.loads(good)
        edit(rec)
        path.write_text(json.dumps(rec))
        got = bench.summarize(
            suite, js, out=tmp_path / "o", version="t", records=tmp_path, inputs=None
        )
        path.write_text(good)
        problems = " ".join(got["verdict"]["problems"])
        assert not got["verdict"]["passed"] and text in problems, problems

    first = "coauth_b50k_s1/main"

    def result(rec: dict, key: str = first) -> dict:
        return rec["results"][key]

    refused(timing, jobs, "cuda-coauth", lambda r: result(r)["regions"].pop(2), "no row update[b3]")
    refused(timing, jobs, "cuda-coauth", lambda r: result(r).update(batches=2), "2 batches")
    refused(timing, jobs, "cuda-coauth", lambda r: result(r).update(rounds=7), "7 rounds")
    refused(
        timing,
        jobs,
        "cuda-coauth",
        lambda r: result(r)["monitor"]["rounds"].__setitem__(0, {"original": {"reasons": ["x"]}}),
        "contaminated rounds were kept",
    )
    refused(
        timing,
        jobs,
        "cuda-coauth",
        lambda r: result(r).update(audit={"flagged": True, "flagged_rounds": [{"round": 3}]}),
        "the contamination audit flags rounds",
    )
    refused(
        timing,
        jobs,
        "cuda-coauth",
        lambda r: result(r).update(guard={"rounds_checked": 11, "passed": False}),
        "the per-round guard",
    )
    refused(
        timing,
        jobs,
        "cuda-coauth",
        lambda r: r["protocol"].update(clocks={"control": "boost"}),
        "clocks 'boost'",
    )
    refused(
        timing,
        jobs,
        "cuda",
        lambda r: result(r, "syn1m_b50k_s2/main")["input"].update(sha256="0" * 64),
        "the inputs are not verified",
    )
    refused(
        timing, jobs, "cuda", lambda r: r["results"].pop("syn2m_b50k_s3/main"), "the reading's are"
    )

    # The medians and the round counts are recomputed from the samples (M9a review).
    def scaled(r):
        reg = result(r)["regions"][0]
        reg["port_runs"] = [x * 1.3 for x in reg["port_runs"]]

    refused(timing, jobs, "cuda-coauth", scaled, "is not its samples'")
    refused(
        timing,
        jobs,
        "cuda-coauth",
        lambda r: result(r)["regions"][1].update(original_runs=[500.0] * 3, port_runs=[450.0] * 3),
        "samples for 11 rounds",
    )
    # The protocol fields of a timed record (the author's protocol).
    for field, value, text in [
        ("layout", "randomized", "one heap layout"),
        ("env", "CUDA_MODULE_LOADING=LAZY, OMP_NUM_THREADS=8", "environment"),
        ("port", "dyng-compat-escher run --backend cuda", "--sync-stages"),
        ("gpu", 1, "GPU 1"),
        ("lock", "none", "exclusive perf lock"),
        ("original", "the patched copy", "not the unpatched build"),
    ]:
        refused(
            timing, jobs, "cuda-coauth", lambda r, f=field, v=value: r["protocol"].update({f: v}),
            text,
        )  # fmt: skip
    # The protocol fields of a memory record (D1), its count guard and both peaks' bytes.
    for field, value, text in [
        ("gpu", 0, "GPU 0"),
        ("lock", "none", "shared lock"),
        ("nsys", "nsys profile -t cuda", "D1's command"),
    ]:
        refused(
            memory, mjobs, "cuda-memory", lambda r, f=field, v=value: r["protocol"].update({f: v}),
            text,
        )  # fmt: skip
    refused(memory, mjobs, "cuda-memory", lambda r: result(r).pop("guard"), "no count guard")
    refused(
        memory,
        mjobs,
        "cuda-memory",
        lambda r: result(r, "syn2m_b50k_s1/main")["port"].pop("peak_live_bytes"),
        "no device_memory peak",
    )
    refused(
        memory,
        mjobs,
        "cuda-memory",
        lambda r: result(r)["store_accounting"].pop(2),
        "no store accounting at store_memory[b2]",
    )
    refused(
        memory,
        mjobs,
        "cuda-memory",
        lambda r: result(r, "syn2m_b50k_s1/main")["store_accounting"][3].update(gated=False),
        "no store accounting at store_memory[b3]",
    )
    refused(
        memory,
        mjobs,
        "cuda-memory",
        lambda r: result(r)["foreign_gpu_processes"].update(after=["1234, x"]),
        "a foreign process on the GPU",
    )
    # A store ratio above the gate is a gated row outside its gate.
    job = next(j for j in mjobs if j["reading"] == "cuda-memory")
    rec = json.loads(Path(job["record"]).read_text())
    rec["results"][first]["store_accounting"][1]["dyng_device_bytes"] = 2000
    Path(job["record"]).write_text(json.dumps(rec))
    got = bench.summarize(
        memory, mjobs, out=tmp_path / "o", version="t", records=tmp_path, inputs=None
    )
    assert not got["verdict"]["passed"]
    assert any("store_memory[b1]" in e for e in got["verdict"]["exceeded"]), got["verdict"]


def test_a_03_certificate_needs_the_escher_gates(cert, certified) -> None:
    """From 0.3 on a certificate needs both ESCHER suites (every gated row rechecked, the
    store_memory rows point by point) and the triad replays of every backend of the coverage
    table, by key and by content; 0.1 and 0.2 certificates do not."""
    results, write, problems = certified
    assert write("0.3.0") == 0, problems()
    doc = json.loads((results / "parity.json").read_text())
    assert doc["required_suites"]["ipdps26_escher"] == {"algorithm": "triad_count", "since": "0.3"}
    assert doc["required_suites"]["tkde26_escher_plus"] == {
        "algorithm": "triad_count",
        "since": "0.3",
    }
    plus = next(s for s in doc["performance"] if s["suite"] == "tkde26_escher_plus")
    regions = {r["region"] for r in plus["gate_table"]}
    assert {"device_memory", "store_memory[build]", "store_memory[b3]"} <= regions
    triad = next(g for g in doc["golden_suites"] if g["set"] == TRIAD_SET)
    assert triad["passed"] is True and triad["required_coverage"] == {
        "cuda": 300,
        "openmp": 142,
        "sequential": 120,
    }

    # A store_memory row whose bytes differ from its record's is caught by recheck_gates.
    path = results / "tkde26_escher_plus.json"
    full = path.read_text()
    cut = json.loads(full)
    row = next(
        g
        for r in cut["readings"]
        if r["gated"]
        for g in r["regions"]
        if g["region"] == "store_memory[b1]"
    )
    row["port_bytes"] += 1
    row["ratio"] = row["port_bytes"] / row["original_bytes"]
    path.write_text(json.dumps(cut))
    assert write("0.3.0") == 1 and "differ from its record's" in problems()
    path.write_text(full)

    # A triad gate row removed from the summary: the gate is not read.
    tpath = results / "ipdps26_escher.json"
    tfull = tpath.read_text()
    cut = json.loads(tfull)
    for r in cut["readings"]:
        r["regions"] = [g for g in r["regions"] if g["region"] != "update[b2]"]
    tpath.write_text(json.dumps(cut))
    assert write("0.3.0") == 1 and "no gated reading of update[b2] (cuda)" in problems()
    tpath.write_text(tfull)

    # A triad replay removed (one backend, or one mode): a 0.3 certificate fails, 0.2 does not.
    for drop in ("openmp", "by_content"):
        saved = {p: p.read_text() for p in results.glob(f"parity-{TRIAD_SET}-*.json")}
        for p, text in saved.items():
            rec = json.loads(text)
            if drop == rec["backend"]:
                p.unlink()
            elif drop == "by_content" and rec["backend"] == "cuda":
                rec["configs"] = [c for c in rec["configs"] if not c.endswith(drop)]
                rec["matrix"] = {
                    k: {c: v[c] for c in rec["configs"]} for k, v in rec["matrix"].items()
                }
                p.write_text(json.dumps(rec))
        assert write("0.3.0") == 1, drop
        want = "have no passed replay by_content" if drop == "by_content" else "gives openmp have"
        assert want in problems(), problems()
        assert write("0.2.0rc1") == 0, problems()
        for p, text in saved.items():
            p.write_text(text)
    # A replay of the oracle (the go/no-go) is no evidence of the algorithm.
    p = next(results.glob(f"parity-{TRIAD_SET}-sequential-*.json"))
    good = p.read_text()
    rec = json.loads(good)
    rec["checks"]["counter"] = "oracle"
    p.write_text(json.dumps(rec))
    assert write("0.3.0") == 1 and "gives sequential have" in problems()
    p.write_text(good)
    # Without the ESCHER suites: 0.3 fails, 0.1 and 0.2 still pass.
    for name in ("ipdps26_escher", "tkde26_escher_plus"):
        (results / f"{name}.json").unlink()
        for record in results.glob(f"{name}-*.json"):
            record.unlink()
    assert write("0.3.0") == 1
    assert "no performance suite of triad_count" in problems()
    assert write("0.2.0rc1") == 0, problems()
    assert write("0.1.0") == 0, problems()


def test_triad_replays_listed_before_0_3(cert, tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    problems: list[str] = []
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.2.0")}
    assert listed[TRIAD_SET]["passed"] is None and not any(TRIAD_SET in p for p in problems)
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.3.0rc1")}
    assert listed[TRIAD_SET]["passed"] is False
    assert sum(TRIAD_SET in p for p in problems) == 2 * 3  # 3 backends x 2 modes
    problems.clear()
    _triad_replays(cert, results, head())
    listed = {g["set"]: g for g in cert.golden_suites(results, head(), problems, "0.3.0")}
    assert listed[TRIAD_SET]["passed"] is True, [p for p in problems if TRIAD_SET in p]
    assert listed[TRIAD_SET]["triad_coverage"]["openmp"] == {"by_content": 142, "by_key": 142}


def test_the_triad_replay_certificate_records(tmp_path: Path) -> None:
    replay = load("parity/triad_replay.py")
    doc = {
        "set": TRIAD_SET,
        "manifest_sha256": "a" * 64,
        "backend": "openmp",
        "threads": 1,
        "mode": "both",
        "counter": "algorithm",
        "plant": None,
        "coverage": True,
        "commit": "c" * 40,
        "cases": {"x/main": {"passed": True}, "y/type1": {"passed": False}},
        "passed": False,
    }
    rec = replay.certificate_record(doc)
    assert rec["configs"] == ["openmp/t1:triad/by_key", "openmp/t1:triad/by_content"]
    assert rec["matrix"]["y/type1"]["openmp/t1:triad/by_key"] == "DIFFERENT"
    assert rec["passed"] is False and rec["goldens"]["manifest_sha256"] == "a" * 64
    path = replay.write_certificate(doc, tmp_path)
    assert path.name == f"parity-{TRIAD_SET}-openmp-t1-both-algorithm.json"
    for edit in ({"plant": "bin"}, {"coverage": False}):
        with pytest.raises(replay.ReplayError):
            replay.certificate_record(doc | edit)


def test_the_dry_run_certificate(cert, bench, tmp_path: Path, fixed_driver) -> None:
    """A dry 0.3 run of triad_count: its suites and golden set judged as in a release, every
    other part listed as outside; it fails when a triad gate row or replay is removed, and it is
    never written into benchmarks/results/."""
    commit = head()
    results = tmp_path / "dry"
    results.mkdir()
    _triad_replays(cert, results, commit)
    for path in (ESCHER_SUITE, ESCHER_PLUS_SUITE):
        suite = bench.load_suite(path)
        records = tmp_path / f"rec-{suite['suite']}"
        jobs = _write_triad_records(bench, suite, records, commit)
        summary = bench.summarize(
            suite, jobs, out=results, version="t", records=records, inputs=None
        )
        assert summary["verdict"]["passed"], summary["verdict"]

    def write() -> int:
        argv = ["write", "--results", str(results), "--version", "0.3.0", "--allow-dirty"]
        return cert.main([*argv, "--dry-run", "triad_count"])

    def doc() -> dict:
        return json.loads((results / "parity.json").read_text())

    assert write() == 0, doc()["verdict"]
    d = doc()
    assert d["dry_run"]["algorithm"] == "triad_count"
    assert "suite ipdps25_dynamosp_mosp (mosp)" in d["dry_run"]["outside"]
    assert "golden set sssp" in d["dry_run"]["outside"]
    assert [g["set"] for g in d["golden_suites"]] == [TRIAD_SET]
    assert {s["suite"] for s in d["performance"]} == {"ipdps26_escher", "tkde26_escher_plus"}
    assert "**Dry run** of `triad_count`" in (results / "README.md").read_text()
    # A gated row removed: fails.
    path = results / "ipdps26_escher.json"
    full = path.read_text()
    cut = json.loads(full)
    for r in cut["readings"]:
        r["regions"] = [g for g in r["regions"] if g["region"] != "full_count"]
    path.write_text(json.dumps(cut))
    assert write() == 1 and any("full_count" in p for p in doc()["verdict"]["problems"])
    path.write_text(full)

    # One gated row of one case dropped from a summary, its verdict counts adjusted: fails, for
    # every kind of row (M9a review: each (backend, metric) was read by some case, so it passed).
    def drop_one(suite: str, case: str, region: str) -> str:
        path = results / f"{suite}.json"
        full = path.read_text()
        cut = json.loads(full)
        n = 0
        for r in cut["readings"]:
            if not r["gated"]:
                continue
            keep = [g for g in r["regions"] if not (g["case"] == case and g["region"] == region)]
            n += len(r["regions"]) - len(keep)
            r["regions"] = keep
        assert n == 1, (suite, case, region)
        cut["verdict"]["gated_regions"] -= 1
        cut["verdict"]["within_gate"] -= 1
        path.write_text(json.dumps(cut))
        return full

    for suite, case, region in [
        ("tkde26_escher_plus", "coauth_b50k_s1/main", "store_memory[b2]"),
        ("tkde26_escher_plus", "syn1m_b50k_s1/main", "store_memory[build]"),
        ("tkde26_escher_plus", "syn2m_b50k_s1/main", "device_memory"),
        ("ipdps26_escher", "syn1m_b50k_s2/main", "update[b2]"),
        ("ipdps26_escher", "coauth_b50k_s3/main", "full_count"),
        ("ipdps26_escher", "syn2m_b50k_s3/main", "end_to_end"),
    ]:
        full = drop_one(suite, case, region)
        assert write() == 1, (case, region)
        assert any(f"no gated row {region} of {case}" in p for p in doc()["verdict"]["problems"])
        (results / f"{suite}.json").write_text(full)
    assert write() == 0, doc()["verdict"]

    # The record's samples are the medians' source: scaled or truncated samples fail.
    summary = json.loads((results / "ipdps26_escher.json").read_text())
    rec_name = next(r["record"] for r in summary["readings"] if r["reading"] == "cuda-coauth")
    rec_path = results / rec_name
    rec_full = rec_path.read_text()
    for edit, text in [
        (lambda reg: reg.update(port_runs=[x * 1.5 for x in reg["port_runs"]]), "samples'"),
        (lambda reg: reg.update(original_runs=reg["original_runs"][:3]), "samples for"),
    ]:
        rec = json.loads(rec_full)
        edit(rec["results"]["coauth_b50k_s1/main"]["regions"][0])
        rec_path.write_text(json.dumps(rec))
        assert write() == 1 and any(text in p for p in doc()["verdict"]["problems"]), text
        rec_path.write_text(rec_full)
    assert write() == 0, doc()["verdict"]
    # A replay removed: fails.
    next(results.glob(f"parity-{TRIAD_SET}-cuda-*.json")).unlink()
    assert write() == 1 and any("gives cuda have" in p for p in doc()["verdict"]["problems"])
    # Never into a release's results.
    argv = ["write", "--results", str(REPO / "benchmarks/results/0.1.0"), "--version", "0.3.0"]
    with pytest.raises(SystemExit):
        cert.main([*argv, "--dry-run", "triad_count"])


def test_the_triad_mutations_reach_the_golden_mutation_record(cert, certified) -> None:
    mutate = load("parity/mutate.py")
    names = {m["name"] for m in mutate.MUTATIONS}
    assert {
        "triad_wrong_bin_sign",
        "triad_skip_frontier_filter_b",
        "triad_skip_frontier_filter_k",
        "triad_inserted_requested_ids",
        "triad_type1_unbounded_keys",
        "cbst_unfill_drops_last_triad",
    } <= names
    results, write, problems = certified
    path = results / "mutation-goldens.json"
    full = path.read_text()
    doc = json.loads(full)
    doc["results"] = [r for r in doc["results"] if r.get("name") != "triad_skip_frontier_filter_k"]
    path.write_text(json.dumps(doc))
    assert write() == 1 and "mutation triad_skip_frontier_filter_k is missing" in problems()
    path.write_text(full)


def test_memory_rows_use_the_peaks_bytes(cert, bench) -> None:
    """A record with the peaks' bytes (triad_count) gives its MiB exactly, so the summary's ratio
    is the record's and recheck_gates recomputes it (the MiB rounded to 0.1 would not)."""
    side_a = {"peak_live_mib": 3018.9, "peak_live_bytes": 3165530112}
    side_b = {"peak_live_mib": 3021.9, "peak_live_bytes": 3168724992}
    assert bench.peak_mib(side_a) == 3165530112 / 2**20
    assert bench.peak_mib({"peak_live_mib": 18.0}) == 18.0 and bench.peak_mib(None) is None
    record = {"results": {"c/main": {"original": side_a, "port": side_b}}}
    got = cert.record_region(
        record, {"case": "c/main", "region": "device_memory", "gate_kind": "memory"}
    )
    assert got["port"] / got["original"] == pytest.approx(3168724992 / 3165530112, rel=1e-12)


def test_the_label_follows_the_tree(cert, tmp_path: Path) -> None:
    """A certificate's --version picks its series' rules, so it must be of the tree's series:
    '0.2.1' on a 0.3 tree would skip the triad requirements (M9a review)."""
    assert cert.series_problem("0.2.1", "0.3.0.dev0")
    assert cert.series_problem("0.1.0", "0.3.0")
    assert cert.series_problem("0.3.0", "0.3.0.dev0") is None
    assert cert.series_problem("0.3.0rc1", "0.3.0") is None
    assert cert.series_problem("t", "0.3.0.dev0") is None  # an unparsable label requires all
    results = tmp_path / "results"
    results.mkdir()
    own = cert.tree_version()
    series = cert.release_series(own)
    older = f"{series[0]}.{series[1] - 1}.9" if series and series[1] > 0 else "9.9.9"
    with pytest.raises(SystemExit, match="follows its tree's release series"):
        cert.main(["write", "--results", str(results), "--version", older, "--allow-dirty"])
