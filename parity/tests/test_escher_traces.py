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
    # The device store has the presets escher_gpu, mosp_escher and dyng_default from M8a step 5
    # (and the replay tools of M8a step 4 the staging preset mosp_escher_staged).
    assert g.DEVICE_PRESETS == ("escher_gpu", "mosp_escher", "dyng_default", "mosp_escher_staged")
    assert g.rule_counts({"documented": [{"rule": "b"}, {"rule": "a"}, {"rule": "b"}]}) == {
        "a": 1,
        "b": 2,
    }


def test_device_replay_judged_against_the_model() -> None:
    """compare.py cbst_trace --store device: MOSP_ESCHER's traces under escher_gpu() must diverge
    where the host model under the same preset does (ADR 0038); a construct below its payload is
    one class whether reported as a status mismatch or as the documented divergence."""
    g = load("parity/cbst_trace_goldens.py")

    class args:  # noqa: N801 - a namespace
        policy = ""

    # Without --policy every set runs under its original's preset (M8a step 5).
    assert not g.against_model("cbst_trace_mosp_escher_test", args)
    assert not g.against_model("cbst_trace_escher_gpu_test", args)
    args.policy = "escher_gpu"
    assert g.against_model("cbst_trace_mosp_escher_test", args)
    args.policy = "dyng_default"  # judged against the trace (field set policy, criterion 10(c))
    assert not g.against_model("cbst_trace_mosp_escher_test", args)
    assert not g.against_model("cbst_trace_escher_gpu_test", args)
    args.policy = "mosp_escher_staged"  # MOSP_ESCHER's behaviours merged so far (M8a step 4)
    assert g.against_model("cbst_trace_mosp_escher_test", args)
    args.policy = "mosp_escher"  # MOSP_ESCHER's preset (M8a step 5): judged against the trace
    assert not g.against_model("cbst_trace_mosp_escher_test", args)
    assert g.against_model("cbst_trace_escher_gpu_test", args)
    args.policy = ""
    model = {
        "trees": [{"tree": 1}, {"tree": 2}, {"tree": 3}],
        "mismatches": [
            {"tree": 1, "tree_op": 0, "op": "construct", "field": "status"},
            {"tree": 2, "tree_op": 7, "op": "fill", "field": "event.compaction"},
            {"tree": 2, "tree_op": 7, "op": "fill", "field": "key.value"},
        ],
    }
    store = {
        "trees": [{"tree": 1}, {"tree": 2}, {"tree": 3}],
        "mismatches": [{"tree": 2, "tree_op": 7, "op": "fill", "field": "event.compaction"}],
        "documented": [{"tree": 1, "tree_op": 0, "rule": "construct_below_payload"}],
    }
    assert g.first_divergences(model) == g.first_divergences(store)
    assert g.first_divergences(store) == {
        1: (0, "construct validation"),
        2: (7, "event.compaction"),
        3: None,
    }
    store["mismatches"][0]["tree_op"] = 8
    assert g.first_divergences(model) != g.first_divergences(store)
    # The store's own integrity verdict is not the field that names a divergence.
    store["mismatches"] = [
        {"tree": 2, "tree_op": 7, "op": "fill", "field": "check.rows"},
        {"tree": 2, "tree_op": 7, "op": "fill", "field": "event.compaction"},
    ]
    assert g.first_divergences(store)[2] == (7, "event.compaction")
    # construct_without_row_counts exempts fields of an op; it does not end a tree (M8a step 5).
    store["documented"] = [{"tree": 3, "tree_op": 0, "rule": "construct_without_row_counts"}]
    assert g.first_divergences(store)[3] is None


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


def test_device_replay_certificate_record() -> None:
    """compare.py cbst_trace --store device --certificate writes the release certificate's replay
    record (parity/certify.py reads its matrix: one config, a verdict per case)."""
    g = load("parity/cbst_trace_goldens.py")
    certify = load("parity/certify.py")

    class args:  # noqa: N801 - a namespace
        policy = ""
        fields = "exact"

    record = {
        "head": "0" * 40,
        "date": "2026-10-07T00:00:00+00:00",
        "manifest_sha256": "ab" * 32,
        "exe": "build/parity-cuda/cpp/tests/dyng_cbst_replay",
        "against_model": False,
        "passed": False,
        "cases": [{"case": "001-main", "ok": True}, {"case": "002-type1", "ok": False}],
    }
    doc = g.certificate_record("cbst_trace_escher_gpu_test", record, args)
    assert doc["set"] == "cbst_trace_escher_gpu_test"
    assert doc["configs"] == ["device:escher_gpu/exact"]
    assert doc["goldens"]["manifest_sha256"] == "ab" * 32
    matrix, compared, failed = certify.replay_matrix(doc)
    assert (compared, failed) == (2, 1)
    assert matrix["001-main"] == {"device:escher_gpu/exact": "equal"}


def test_whole_first_divergence_against_the_model() -> None:
    """M8a review: a store diverges like the model only when every field, key and value of the
    first divergent op agrees, not only the op and the first field name."""
    g = load("parity/cbst_trace_goldens.py")

    def report(mismatches, documented=()):
        return {
            "trees": [{"tree": 1}, {"tree": 2}],
            "mismatches": mismatches,
            "documented": list(documented),
        }

    m = {"tree": 1, "tree_op": 5, "op": "fill", "key": -1}
    model = report([dict(m, field="after.scratch_payload", expected="4", actual="8")])
    same = report(
        [
            dict(m, field="after.scratch_payload", expected="4", actual="8"),
            dict(m, field="check.rows", expected="0", actual="1"),
        ]
    )
    assert g.first_divergence_sets(model) == g.first_divergence_sets(same)
    # The same op and first field, but a row of the store differs too: not as the model.
    worse = report(
        [
            dict(m, field="after.scratch_payload", expected="4", actual="8"),
            dict(m, field="key.row", key=3, expected="[1]", actual="[2]"),
        ]
    )
    assert g.first_divergences(model) == g.first_divergences(worse)
    assert g.first_divergence_sets(model) != g.first_divergence_sets(worse)
    # Another value of the same field: not as the model either.
    other = report([dict(m, field="after.scratch_payload", expected="4", actual="16")])
    assert g.first_divergence_sets(model) != g.first_divergence_sets(other)
    # A construct below its payload is one class, reported either way.
    c = {"tree": 2, "tree_op": 0, "op": "construct", "key": -1}
    as_status = report(
        [
            dict(c, field="status", expected="error", actual="ok"),
            dict(c, field="after.fixed_size", expected="8", actual="0"),
        ]
    )
    as_documented = report([], [{"tree": 2, "tree_op": 0, "rule": "construct_below_payload"}])
    assert g.first_divergence_sets(as_status)[2] == g.first_divergence_sets(as_documented)[2]


def test_first_divergence_of_a_thrown_op() -> None:
    """An op that both the store and the model throw at: the exception type is compared, not the
    message each words differently, and a thrown insert's InsertMapping is none on either side
    (the merge-step replays of the staging preset, M8a review fixes). Every other value still
    counts."""
    g = load("parity/cbst_trace_goldens.py")

    def report(*mismatches):
        return {"trees": [{"tree": 8}], "mismatches": list(mismatches), "documented": []}

    m = {"tree": 8, "tree_op": 3, "op": "insert", "key": -1}
    store = report(
        dict(m, field="out.mapping", expected="item 0: 13 of [13,14]", actual="- of []"),
        dict(
            m,
            field="status",
            expected="ok",
            actual="exception dyng::capacity_error: dyng: CBST t: 106 payload entries do not fit "
            "its capacity of 96 (policy mosp_escher_staged does not grow the payload)",
        ),
        dict(m, field="after.num_records", expected="14", actual="12"),
    )
    model = report(
        dict(m, field="out.mapping", expected="item 0: 13 of [13,14]", actual="0 of [0,0]"),
        dict(
            m,
            field="status",
            expected="ok",
            actual="exception dyng::capacity_error: insertCBST: payload overflow (106 > 96)",
        ),
        dict(m, field="after.num_records", expected="14", actual="12"),
    )
    assert g.first_divergence_sets(store) == g.first_divergence_sets(model)
    assert ("status", -1, "ok", "exception dyng::capacity_error") in g.first_divergence_sets(store)[
        8
    ][1]
    # Another exception type, or another state value at the op: not as the model.
    other_type = report(
        dict(store["mismatches"][0]),
        dict(store["mismatches"][1], actual="exception dyng::internal_error: x"),
        dict(store["mismatches"][2]),
    )
    assert g.first_divergence_sets(other_type) != g.first_divergence_sets(model)
    other_state = report(
        dict(store["mismatches"][0]),
        dict(store["mismatches"][1]),
        dict(store["mismatches"][2], actual="13"),
    )
    assert g.first_divergence_sets(other_state) != g.first_divergence_sets(model)
    # An op that did not throw keeps its mapping: a different mapping is not as the model.
    kept = report(dict(m, field="out.mapping", expected="item 0: 13 of [13,14]", actual="- of []"))
    kept_model = report(
        dict(m, field="out.mapping", expected="item 0: 13 of [13,14]", actual="0 of [0,0]")
    )
    assert g.first_divergence_sets(kept) != g.first_divergence_sets(kept_model)


def test_documented_divergence_counts_are_pinned(tmp_path: Path) -> None:
    """M8a review: a replay whose documented divergences differ from the pinned counts fails."""
    g = load("parity/cbst_trace_goldens.py")
    table = tmp_path / "counts.toml"
    table.write_text(
        "[cbst_trace_mosp_escher_test.dyng_default]\npolicy_difference = 4\npolicy_exception = 1\n"
        "[cbst_trace_escher_gpu_test.escher_gpu]\n"
    )
    ok = {"policy_difference": 4, "policy_exception": 1}
    assert (
        g.documented_count_problems("cbst_trace_mosp_escher_test", "dyng_default", ok, table) == []
    )
    assert g.documented_count_problems("cbst_trace_escher_gpu_test", "escher_gpu", {}, table) == []
    planted = {"policy_difference": 2400, "policy_exception": 1}
    assert g.documented_count_problems(
        "cbst_trace_mosp_escher_test", "dyng_default", planted, table
    )
    assert g.documented_count_problems(
        "cbst_trace_escher_gpu_test", "escher_gpu", {"policy_difference": 1}, table
    )
    # A set and preset without pinned counts fails too.
    assert g.documented_count_problems("cbst_trace_escher_gpu_test", "dyng_default", {}, table)
    # The committed table pins every set of the M8a goldens under its original's preset.
    import tomllib

    pinned = tomllib.loads((REPO / "parity" / "cbst_documented.toml").read_text())
    for name in g.SETS:  # the presets parity/certify.py requires
        assert g.original_preset(name) in pinned.get(name, {}), name
        assert "dyng_default" in pinned.get(name, {}), name

