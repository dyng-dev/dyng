# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of the CBST trace tools of M8a (ADR 0036): the Python reader on the committed traces
(and a changed byte), the nm census on small objects built here, the export dispatch, the twin
check's timing filter and the cross-check comparator."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUBSET = REPO / "cpp" / "tests" / "data" / "goldens" / "cbst_trace"


def load(rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def committed() -> list[Path]:
    return sorted(SUBSET.rglob("*.cbt"))


def test_committed_traces_verify() -> None:
    et = load("parity/escher_trace.py")
    traces = committed()
    assert traces, f"no committed traces under {SUBSET}"
    repos = set()
    for path in traces:
        trees: dict = {}
        ended = False
        for kind, rec in et.Trace(path).records():
            if kind == "header":
                repos.add(rec["repo"])
                assert rec["format_version"] == et.FORMAT_VERSION
            if kind == "end":
                ended = rec["exit"] == "normal"
            if kind == "op":
                tree = trees.setdefault(rec["tree"], et.TreeState())
                et.apply_digest(tree, rec, keep_rows=True)
                assert et.verify_op(tree, rec) == [], (path, rec["seq"])
        assert ended, path
    assert repos == {"ESCHER-GPU", "MOSP_ESCHER"}


def test_subset_listing_matches_the_files() -> None:
    import hashlib

    listing = (SUBSET / "SUBSET.txt").read_text().splitlines()
    rows = [line.split("  ") for line in listing if line and not line.startswith("#")]
    assert len(rows) == len(committed())
    for digest, rel, *_ in rows:
        assert hashlib.sha256((SUBSET / rel).read_bytes()).hexdigest() == digest, rel
    total = sum(p.stat().st_size for p in committed())
    assert total <= 320_000


def test_a_changed_byte_fails_verification(tmp_path: Path) -> None:
    et = load("parity/escher_trace.py")
    src = committed()[0]
    data = bytearray(src.read_bytes())
    at = data.find(b"key.rows")
    assert at > 0
    first_value = at + len(b"key.rows") + 1 + 1 + 8
    data[first_value] ^= 1
    bad = tmp_path / "bad.cbt"
    bad.write_bytes(bytes(data))
    assert et.main(["verify", str(bad)]) == 1


def test_summary_covers_the_committed_paths() -> None:
    """The committed subset lets hosted CI replay reuse, surplus with rebuild, fill overflow,
    unfill on chained rows and MOSP_ESCHER's compaction inside fill and inside insert."""
    et = load("parity/escher_trace.py")
    s = [et.summarize(p) for p in committed()]
    assert sum(x["insert"]["reuse"] for x in s) > 0
    assert sum(x["insert"]["rebuild"] for x in s) > 0
    assert sum(x["fill"]["overflow"] for x in s) > 0
    assert sum(x["unfill"]["chained"] for x in s) > 0
    b = [x for x in s if x.get("repo") == "MOSP_ESCHER"]
    assert sum(x["fill"]["compaction"] for x in b) > 0
    assert sum(x["insert_compaction"] for x in b) > 0


def test_fnv1a_reference_values() -> None:
    et = load("parity/escher_trace.py")
    assert et.fnv1a(b"") == 0xCBF29CE484222325
    assert et.fnv1a(b"a") == 0xAF63DC4C8601EC8C
    assert et.fnv1a(b"foobar") == 0x85944171F73967E8


@pytest.mark.skipif(
    shutil.which("g++") is None or shutil.which("nm") is None, reason="needs g++ and nm"
)
def test_census_finds_an_unwrapped_reference(tmp_path: Path, monkeypatch) -> None:
    census = load("parity/exporters/escher/cbst_census.py")
    copy = tmp_path / "copy"  # noqa: F811 (the module is not used in this test)
    (copy / "build").mkdir(parents=True)
    (copy / "parity_export" / "link").mkdir(parents=True)
    (copy / "parity_export" / "bin").mkdir(parents=True)
    (copy / "ops.cpp").write_text(
        textwrap.dedent("""
        #include <vector>
        struct CBSTContext { int n; };
        struct CBSTOperations {
          void fill(const std::vector<int>&);
          void erase(const std::vector<int>&);
          const CBSTContext& context() const;
          CBSTContext ctx_;
        };
        void CBSTOperations::fill(const std::vector<int>&) {}
        void CBSTOperations::erase(const std::vector<int>&) {}
        const CBSTContext& CBSTOperations::context() const { return ctx_; }
        void unfillCBST(const std::vector<int>&, CBSTContext&) {}
    """)
    )
    (copy / "caller.cpp").write_text(
        textwrap.dedent("""
        #include <vector>
        struct CBSTContext { int n; };
        struct CBSTOperations {
          void fill(const std::vector<int>&);
          void erase(const std::vector<int>&);
          const CBSTContext& context() const;
          CBSTContext ctx_;
        };
        void unfillCBST(const std::vector<int>&, CBSTContext&);
        void use(CBSTOperations& o) {
          o.fill({1}); o.erase({1}); unfillCBST({1}, const_cast<CBSTContext&>(o.context()));
        }
    """)
    )
    for name in ("ops", "caller"):
        subprocess.run(
            ["g++", "-c", str(copy / f"{name}.cpp"), "-o", str(copy / "build" / f"{name}.o")],
            check=True,
        )
    (copy / "parity_export" / "link" / "t.cmd").write_text(
        "g++ -o build/t build/ops.o build/caller.o\n"
    )
    shutil.copyfile(copy / "build" / "caller.o", copy / "parity_export" / "bin" / "t")
    exports = tmp_path / "exports" / "X"
    exports.mkdir(parents=True)
    (exports / "wrap_hooks.txt").write_text("")
    monkeypatch.setattr(census, "REPO", tmp_path / "repo")
    (tmp_path / "repo" / "parity" / "export_patches").mkdir(parents=True)
    (tmp_path / "repo" / "parity" / "export_patches" / "X").symlink_to(exports)
    monkeypatch.setitem(census.HOOK_OBJECT, "X", "nohooks.o")
    monkeypatch.setitem(census.NOT_TWINS, "X", [])
    # unfillCBST is not wrapped: the census fails and names it
    (exports / "wrap_symbols.txt").write_text(
        "_ZN14CBSTOperations4fillERKSt6vectorIiSaIiEE\n_ZN14CBSTOperations5eraseERKSt6vectorIiSaIiEE\n"
    )
    report = census.census(copy, "X")
    assert not report["passed"]
    assert any("unfillCBST" in p for p in report["problems"])
    # wrapped (but the twin lacks the wrappers): still fails, for that reason
    (exports / "wrap_symbols.txt").write_text(
        "_ZN14CBSTOperations4fillERKSt6vectorIiSaIiEE\n_ZN14CBSTOperations5eraseERKSt6vectorIiSaIiEE\n"
        "_Z10unfillCBSTRKSt6vectorIiSaIiEER11CBSTContext\n"
    )
    report = census.census(copy, "X")
    assert all("unwrapped" not in p for p in report["problems"])
    assert any("no __wrap_" in p for p in report["problems"])
    # context() is a read-only accessor
    kinds = {r["symbol"].split("(")[0]: r["kind"] for r in report["twins"][0]["references"]}
    assert kinds["CBSTOperations::context"] == "accessor"


def test_export_and_compare_dispatch() -> None:
    for cmd in (
        ["parity/export_goldens.py", "cbst_trace", "--help"],
        ["parity/compare.py", "cbst_trace", "--help"],
        ["parity/escher_trace.py", "--help"],
        ["parity/escher_crosscheck.py", "--help"],
        ["parity/exporters/escher/check_twins.py", "--help"],
        ["parity/exporters/escher/cbst_census.py", "--help"],
    ):
        p = subprocess.run(
            [sys.executable, str(REPO / cmd[0]), *cmd[1:]], capture_output=True, text=True
        )
        assert p.returncode == 0, (cmd, p.stderr)
    g = load("parity/cbst_trace_goldens.py")
    for name, spec in g.SETS.items():
        assert name.startswith("cbst_trace_") and spec["reference"] in g.REFS
        assert spec["mode"] in ("full", "digest", "auto", "batches")


def test_trace_sections_survive_the_goldens_writers(tmp_path: Path, monkeypatch) -> None:
    """Set names with digits (cbst_trace_escher_gpu_coauth50k) are kept by both writers."""
    g = load("parity/cycle_count_goldens.py")
    toml = tmp_path / "goldens.toml"
    toml.write_text("schema = 1\n\n[sets.sssp]\nnum_cases = 0\n\n[sets.sssp.cases]\n")
    g.write_toml(
        "[sets.cbst_trace_escher_gpu_coauth50k]\nnum_cases = 1\n\n"
        '[sets.cbst_trace_escher_gpu_coauth50k.cases]\n"a" = { ops = 1 }\n',
        "cbst_trace_escher_gpu_coauth50k",
        toml,
    )
    g.write_toml("[sets.cycle_count]\nnum_cases = 2\n", g.SET, toml)
    import tomllib

    doc = tomllib.loads(toml.read_text())
    assert doc["sets"]["cbst_trace_escher_gpu_coauth50k"]["num_cases"] == 1
    assert doc["sets"]["cycle_count"]["num_cases"] == 2


def test_twin_check_filters_only_timing_lines() -> None:
    ct = load("parity/exporters/escher/check_twins.py")
    kept, dropped = ct.filtered(
        "[timing] batch 1: update 13.0 ms\n[check] H2V: 8 keys, 0 wrong rows\n"
        "[load] built in 12 ms\nPASS  x\nhsosp t_escher_ms=3.2\n"
    )
    assert kept == ["[check] H2V: 8 keys, 0 wrong rows", "PASS  x"]
    assert dropped == 3


def test_crosscheck_classifies_the_documented_differences() -> None:
    cc = load("parity/escher_crosscheck.py")
    a = {"fixed_size": 100, "fixed_before": 50, "bump": 80, "bump_before": 40}
    b = {"fixed_size": 60, "fixed_before": 60, "bump": 30, "bump_before": 55}
    assert cc.classify("value", "fill", a, b, "ESCHER-GPU", "") == "grows_vs_compacts"
    assert (
        cc.classify(
            "not replayed",
            "rebuild",
            {},
            {},
            "ESCHER-GPU",
            "stop: MOSP_ESCHER has no rebuildFromDevice",
        )
        == "rebuild_only_in_a"
    )
    assert cc.classify("exception", "construct", {}, {}, "ESCHER-GPU", "") == "b_validates"
    assert cc.classify("occupancy", "construct", {}, {}, "ESCHER-GPU", "") == "caller_counts"
    same = {
        "status": 0,
        "exception": None,
        "mapping": [1],
        "keys": [1],
        "subtree_avail": [0],
        "records": {1: (1, 0, 4, 0, 1, 3, 0, 0, 1, 1, 1)},
        "rows": {1: [5]},
        "fixed_size": 10,
        "bump": 4,
        "bump_before": 4,
        "fixed_before": 10,
    }
    assert cc.first_difference(same, copy.deepcopy(same)) is None
    other = dict(same, mapping=[2])
    assert cc.first_difference(same, other)[0] == "mapping"


def test_hash_only_arrays_are_unknown_not_errors() -> None:
    """The ops of a batches-mode trace record their inputs as hashes (escher_gpu_syn1m): the
    summary skips what it cannot read instead of failing."""
    et = load("parity/escher_trace.py")
    from array import array

    rec = {
        "in.keys": et.Array("hash", 4, 3, hash=0x1234),
        "in.prefix": et.Array("inline", 4, 2, data=array("i", [2, 5])),
        "in.empty": et.Array("null", 4, 0),
    }
    assert et.known(rec, "in.keys") is None
    assert list(et.known(rec, "in.prefix")) == [2, 5]
    assert list(et.known(rec, "in.empty")) == []
    assert et.known(rec, "in.absent") is None
    with pytest.raises(et.TraceError):
        rec["in.keys"].values()


def test_crosscheck_classifies_a_compacting_op() -> None:
    """A divergence at an op that compacted on one side (rows it does not name moved) is the
    documented grows_vs_compacts difference, whatever field differs first."""
    cc = load("parity/escher_crosscheck.py")
    a = {"compacted": False, "what": ""}
    b = {"compacted": True, "what": ""}
    assert cc.classify("occupancy", "fill", a, b, "ESCHER-GPU", "") == "grows_vs_compacts"
    thrown = {"compacted": False, "what": "payload overflow: Increase payloadCapacity"}
    assert cc.classify("exception", "fill", a, thrown, "ESCHER-GPU", "") == "grows_vs_compacts"


def test_model_replay_rules_of_the_sets() -> None:
    """compare.py cbst_trace --store: only the documented ops may stay unreplayable (ADR 0037)."""
    g = load("parity/cbst_trace_goldens.py")
    h2h = {"dataset": "H2H", "unreplayable_reason": "op 1 (rebuild) has hash-only inputs: in.x"}
    h2v = {"dataset": "H2V", "unreplayable_reason": "op 1 (rebuild) has hash-only inputs: in.x"}
    assert g.allowed_unreplayable("cbst_trace_escher_gpu_coauth50k", h2h)
    assert not g.allowed_unreplayable("cbst_trace_escher_gpu_coauth50k", h2v)
    assert g.allowed_unreplayable("cbst_trace_escher_gpu_syn1m", h2v)
    assert not g.allowed_unreplayable("cbst_trace_escher_gpu_test", h2h)
    assert not g.allowed_unreplayable("cbst_trace_mosp_escher_test", h2h)
    # The device store's replay is reserved for M8a step 3.
    assert g.compare_main(["--store", "device", "--set", "cbst_trace_escher_gpu_targeted"]) == 2


def test_model_replay_tool_on_the_committed_subset(tmp_path: Path) -> None:
    """dyng_cbst_replay, when built, replays the committed traces with 100% agreement."""
    g = load("parity/cbst_trace_goldens.py")
    exe = g.find_replay_exe()
    if exe is None or not exe.is_file():
        pytest.skip("dyng_cbst_replay is not built")
    for trace in committed():
        r = g.replay_case(exe, trace, "", "exact", tmp_path)
        assert "error" not in r, r
        assert r["ok"] and r["mismatch_count"] == 0 and r["compared"] == r["ops"], trace
