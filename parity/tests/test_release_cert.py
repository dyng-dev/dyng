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


DRIVER = {"nvidia": "590.48.01", "cuda_driver_api": "13.1"}
CC_DIGESTS = {
    "DD": "e538ec13a262b84a0664db2f22e6ff720fb5bb74965536c37f4b8e962d156e4a",
    "github": "a5a43e92b9a22054882017a2b5dea267535c8d94705ac4987d967e517448c9d4",
    "twitch": "c2270e018af70321c8f6d1517abc097204105ec7bafe71527555b4f954fbea82",
    "collab": "043c465d207992054d2567a277bfba9fad24d994040241987e61f9e9daa12ae4",
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
        "driver": DRIVER,
        "datasets_sha256": dict(CC_DIGESTS),
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
            rec = _cc_record(reading["cases"], commit, reading.get("clocks"), **kw)
        Path(job["record"]).write_text(json.dumps(rec))
    return jobs


def _write_sssp_records(bench, suite, records: Path, commit: str, ratio: float = 0.9):
    records.mkdir(parents=True, exist_ok=True)
    jobs = bench.plan(suite, records=records, build_root=Path("/b"))
    for job in jobs:
        reading = next(r for r in suite["readings"] if r["name"] == job["reading"])
        dataset = next(d for d in suite["datasets"] if d["name"] == job["dataset"])
        backend = reading["backend"]
        base = suite["backends"][backend]["baseline"]
        rec = {
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
        for batch in ("safe50k", "unsafe50k", "local10k"):
            if job["kind"] == "memory":
                rec["results"][batch] = {
                    "original": {"peak_live_mib": 100.0},
                    "port": {"peak_live_mib": 100.0 * ratio},
                    "ratio": ratio,
                }
            else:
                rec["results"][batch] = {"regions": [_region("apply", 50.0, 50.0 * ratio)]}
        if backend == "cuda" and job["kind"] == "run":
            rec["protocol"] = {"gpu_clocks": {"control": reading["clocks"]}}
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
        write = _write_cc_records if suite["algorithm"] == "cycle_count" else _write_sssp_records
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


@pytest.fixture()
def fixed_driver(cert, monkeypatch: pytest.MonkeyPatch):
    """The driver seen "now" is the records' (the certificate compares the two)."""
    real = cert.environment

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


# --- mutate.py -----------------------------------------------------------------------------------


def test_mutation_points_match_the_code_once() -> None:
    mutate = load("parity/mutate.py")
    assert {m["config"] for m in mutate.MUTATIONS} == {"sequential", "openmp:4", "cuda"}
    for m in mutate.MUTATIONS:
        text = (REPO / m["file"]).read_text()
        assert text.count(m["old"]) == 1, m["name"]
        assert m["new"] != m["old"]


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
