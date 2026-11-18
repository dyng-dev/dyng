# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_port.py's comparators that need no build (M10a review): the
fused engine's trace replay compares every field the golden records but the batch inputs, so a
planted batch_stamp, capacity, update_rule, delta or num_new_edges change is caught; the
deterministic replay's field list holds them too; the IrLP replay refuses a snapshot count that
is not the batch count on either side."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace as dlt  # noqa: E402
import label_propagation_port as lpp  # noqa: E402


def f(name: str, value: int, dtype: int = 1) -> dlt.Field:
    raw = struct.pack("<i" if dtype == 1 else "<f", value)
    return dlt.Field(name, dtype, False, 1, raw, hashlib.sha256(raw).hexdigest())


def golden_trace(stamp: int = 7, extra_port_field: bool = False, golden: bool = True) -> dlt.Trace:
    """Two batches with every stage of the golden's records; the batch inputs only if golden."""
    recs = []
    for b in range(2):
        for tag, names in lpp.TRACE_FIELDS.items():
            fields = {n: f(n, 10 * b + i) for i, n in enumerate(names)}
            if tag == "sparsify":
                fields["batch_stamp"] = f("batch_stamp", stamp + b)
            if tag == "append_edges" and b == 0:
                fields["row_ptr"] = f("row_ptr", 3)
            if tag == "apply_vertex_changes" and golden:
                for n in sorted(lpp.GOLDEN_ONLY_FIELDS["apply_vertex_changes"]):
                    fields[n] = f(n, 1)
            if tag == "propagation_begin":
                fields["labels"] = f("labels", 5)
                fields["next_tag"] = f("next_tag", 2)
            recs.append(dlt.Record(tag, b, len(recs), fields))
        end = {n: f(n, b) for n in ("iterations", "frontier_visits", "persistent_iterations",
                                     "next_tag", "labels")}  # fmt: skip
        if extra_port_field:
            end["port_only"] = f("port_only", 1)
        recs.append(dlt.Record("propagation_end", b, len(recs), end))
    return dlt.Trace(Path("t"), False, recs)


def test_the_deterministic_fields_include_the_stamps_and_the_parameters() -> None:
    have = {(t, n) for t, names in lpp.TRACE_FIELDS.items() for n in names}
    for want in [
        ("sparsify", "batch_stamp"),
        ("sparsify", "num_new_edges"),
        ("append_edges", "capacity"),
        ("append_edges", "num_new_edges"),
        ("propagation_begin", "delta"),
        ("propagation_begin", "update_rule"),
        ("propagation_begin", "max_iterations"),
        ("propagation_begin", "num_nodes"),
        ("connected_components", "num_inserted"),
        ("apply_vertex_changes", "num_deleted"),
    ]:
        assert want in have, want


def test_every_field_catches_a_planted_batch_stamp_and_ignores_the_batch_inputs() -> None:
    gold = golden_trace()
    port = golden_trace(golden=False)  # the port does not record the batch inputs
    compared, bad = lpp.compare_traces(port, gold, every_field=True)
    assert bad == [] and compared > 0
    # A planted batch_stamp change (the fields the old field list skipped): caught.
    _, bad = lpp.compare_traces(golden_trace(stamp=8, golden=False), gold, every_field=True)
    assert {m["field"] for m in bad} == {"sparsify.batch_stamp"}
    _, bad = lpp.compare_traces(golden_trace(stamp=8, golden=False), gold)
    assert {m["field"] for m in bad} == {"sparsify.batch_stamp"}
    # A field the golden records and the port lacks, and a stage only the port records.
    port = golden_trace(golden=False)
    del port.all[0].fields["status"]
    _, bad = lpp.compare_traces(port, gold, every_field=True)
    assert [m["field"] for m in bad] == ["apply_vertex_changes.status"]
    port = golden_trace(golden=False)
    port.all.append(dlt.Record("label_init_extra", 0, 99, {}))
    _, bad = lpp.compare_traces(port, gold, every_field=True)
    assert bad == []  # not a batch stage: the reader's batches() ignores it


def test_the_irlp_counts_need_one_snapshot_per_batch() -> None:
    n, batches = 10, 4
    expected = {"snapshots": [{"snapshot": i} for i in range(batches)]}
    assert lpp.irlp_counts(batches, expected, 4 * n * batches, n, batches)[1]
    # A shortened golden, an extra port snapshot, a partial one, an extra results.csv row.
    short = {"snapshots": expected["snapshots"][:-1]}
    assert not lpp.irlp_counts(batches, short, 4 * n * batches, n, batches)[1]
    assert not lpp.irlp_counts(batches, expected, 4 * n * (batches + 1), n, batches)[1]
    assert not lpp.irlp_counts(batches, expected, 4 * n * batches + 4, n, batches)[1]
    assert not lpp.irlp_counts(batches, expected, 4 * n * batches, n, batches + 1)[1]


def test_an_in_place_run_skips_only_the_schedule_field() -> None:
    """The in_place replay compares its deterministic stages with the damped trace golden: the
    schedule (propagation_begin.update_rule) is the one field that differs by construction."""
    gold = golden_trace()
    port = golden_trace(golden=False)
    for r in port.all:
        if r.tag == "propagation_begin":
            r.fields["update_rule"] = f("update_rule", 99)
    _, bad = lpp.compare_traces(port, gold)
    assert {m["field"] for m in bad} == {"propagation_begin.update_rule"}
    _, bad = lpp.compare_traces(port, gold, skip=lpp.SCHEDULE_FIELD)
    assert bad == []


# --- the original's value sets of connected_components.hook_rounds (ADR 0100) ---------------------


def i64(name: str, value: int) -> dlt.Field:
    raw = struct.pack("<q", value)
    return dlt.Field(name, 2, False, 1, raw, hashlib.sha256(raw).hexdigest())


def cc_trace(hook_rounds: dict[int, int], components: int = 4) -> dlt.Trace:
    """Three batches whose connected_components stage records the hook rounds (i64, as the
    recorder does) beside the component count."""
    recs = []
    for b in range(3):
        fields = {
            "num_components": f("num_components", components if b == 1 else 4),
            "hook_rounds": i64("hook_rounds", hook_rounds.get(b, 4)),
        }
        recs.append(dlt.Record("connected_components", b, len(recs), fields))
    return dlt.Trace(Path("t"), False, recs)


def test_the_hook_rounds_are_judged_against_the_originals_values() -> None:
    """A hook-round count the original's own runs produce at that case and batch is accepted
    (and recorded), one they do not produce is a mismatch, and every other field stays exact."""
    gold = cc_trace({1: 4})
    original = {0: [4], 1: [3, 4], 2: [4]}
    accepted: list[dict] = []
    _, bad = lpp.compare_traces(
        cc_trace({1: 3}), gold, every_field=True, original=original, accepted=accepted
    )
    assert bad == []
    assert accepted == [{"batch": 1, "field": "connected_components.hook_rounds", "port": 3,
                         "golden": 4, "original_values": [3, 4]}]  # fmt: skip
    # A value outside the original's set, at the batch or at another batch: a mismatch.
    for port, orig in [({1: 5}, original), ({2: 3}, original), ({1: 3}, {1: [3]})]:
        accepted = []
        _, bad = lpp.compare_traces(
            cc_trace(port), gold, every_field=True, original=orig, accepted=accepted
        )
        assert [m["field"] for m in bad] == ["connected_components.hook_rounds"], port
        assert accepted == []
    # Without the original's values (every other replay; a case the evidence does not list):
    # exact.
    _, bad = lpp.compare_traces(cc_trace({1: 3}), gold, every_field=True)
    assert [m["field"] for m in bad] == ["connected_components.hook_rounds"]
    _, bad = lpp.compare_traces(cc_trace({1: 3}), gold, every_field=True, original={})
    assert [m["field"] for m in bad] == ["connected_components.hook_rounds"]
    # Every other field is still exact, also beside an accepted hook-round count.
    accepted = []
    _, bad = lpp.compare_traces(
        cc_trace({1: 3}, components=5), gold, every_field=True, original=original,
        accepted=accepted,
    )  # fmt: skip
    assert [m["field"] for m in bad] == ["connected_components.num_components"]
    assert len(accepted) == 1
    # Only the fields of ORIGINAL_VALUE_FIELDS take the rule.
    assert lpp.ORIGINAL_VALUE_FIELDS == frozenset({("connected_components", "hook_rounds")})
    other = cc_trace({1: 4}, components=3)
    _, bad = lpp.compare_traces(other, gold, every_field=True, original={1: [3, 4]})
    assert [m["field"] for m in bad] == ["connected_components.num_components"]


def test_the_evidence_is_the_original_twins_runs(tmp_path: Path) -> None:
    """The committed evidence holds, per batch of 500k, the golden's count and every count of
    the original's twin (3 or 4 at batch 2), derived from its committed source; a changed set or
    a changed source fails its check."""
    import json

    assert lpp.original_values_problems() == []
    assert lpp.original_values("500k_d1e-4_s42")[2] == [3, 4]
    assert all(len(v) == 1 for b, v in lpp.original_values("500k_d1e-4_s42").items() if b != 2)
    assert lpp.original_values("50k_d1e-4_s42") == {}
    doc = json.loads(lpp.ORIGINAL_VALUES_RECORD.read_text())
    doc["cases"]["500k_d1e-4_s42"]["values"]["3"] = [3, 4]
    widened = tmp_path / "widened.json"
    widened.write_text(json.dumps(doc))
    assert any("not its source's" in p for p in lpp.original_values_problems(widened))
    doc = json.loads(lpp.ORIGINAL_VALUES_RECORD.read_text())
    doc["cases"]["500k_d1e-4_s42"]["source_sha256"] = "0" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps(doc))
    assert any("not the committed one" in p for p in lpp.original_values_problems(other))
    doc = json.loads(lpp.ORIGINAL_VALUES_RECORD.read_text())
    doc["fields"].append("connected_components.num_components")
    more = tmp_path / "more.json"
    more.write_text(json.dumps(doc))
    assert any("fields" in p for p in lpp.original_values_problems(more))


def damped_record(mismatches: list[dict], **checks: bool) -> dict:
    case = {
        "case": "500k_d1e-4_s42",
        "checks": {"dynlp_labels.txt": True, "label_bits": True, "results.csv": True,
                   "trace": not mismatches, **checks},
        "equal": False,
        "trace_mismatches": mismatches,
    }  # fmt: skip
    return {"equal": False, "cases": [case]}


def hook_mismatch(port: int, golden: int = 4, batch: int = 2) -> dict:
    return {
        "batch": batch,
        "field": "connected_components.hook_rounds",
        "detail": f"port i64[1] {lpp.i64_sha256(port)[:12]} vs golden i64[1] "
        f"{lpp.i64_sha256(golden)[:12]}",
    }


def test_a_damped_record_judged_again_under_the_rule() -> None:
    """The existing record is judged again without a new run: the 500k hook-round difference
    the original itself produces passes; any other recorded difference keeps the case failed."""
    plain = lambda case: False  # noqa: E731 (500k's golden is not a digest trace)
    out = lpp.rejudge_damped(damped_record([hook_mismatch(3)]), golden_mode=plain)
    assert out["equal"] is True and out["cases"][0]["checks"]["trace"] is True
    assert out["cases"][0]["original_values_accepted"][0]["port"] == 3
    assert out["original_value_sets"]["fields"] == ["connected_components.hook_rounds"]
    for doc in [
        damped_record([hook_mismatch(5)]),  # not a value of the original
        damped_record([hook_mismatch(3, batch=3)]),  # not at that batch
        damped_record([hook_mismatch(3), {"batch": 2, "field": "label_init.w0", "detail": "x"}]),
        damped_record([hook_mismatch(3)], label_bits=False),
        damped_record([hook_mismatch(3)] * 20),  # a truncated list: not judged
    ]:
        out = lpp.rejudge_damped(doc, golden_mode=plain)
        assert out["equal"] is False and out["cases"][0]["equal"] is False
    # A golden of the other digest mode: the recorded trace verdict may have been that.
    out = lpp.rejudge_damped(damped_record([hook_mismatch(3)]), golden_mode=lambda c: True)
    assert out["equal"] is False


# --- The rate rule of the in_place certificate (the author's decision of 2026-11-18, ADR 0111) ---


def _fake_inplace(monkeypatch, tmp_path: Path, port_iterations, original_iterations,
                  port_label=0.25):  # fmt: skip
    """cmd_inplace on one synthetic case without a build, a GPU or the goldens: the port's and the
    original's k-th runs return the k-th entry of their iteration lists (a callable of k), the
    port's labels `port_label` (the reference runs' are 0.25), the trace equal."""
    import argparse

    import numpy as np

    n, batches = 4, 3
    th = {"t_mean": 0.01, "t_max": 0.01, "t_flips": 0, "components": [1, 2, 2],
          "iterations_lo": [0, 8, 8], "iterations_hi": [2, 12, 12],
          "iterations_sum_lo": 15, "iterations_sum_hi": 27}  # fmt: skip
    name = "5k5_d1e-4_s42_inplace"
    frozen = {"rule": {"n_gpu1": 2}, "thresholds": {name: th},
              "rate_rule": dict(lpp.g.RATE_RULE)}  # fmt: skip
    monkeypatch.setattr(lpp, "frozen_inplace", lambda: frozen)
    monkeypatch.setattr(lpp, "patched_copy", lambda: tmp_path)
    monkeypatch.setattr(lpp, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(lpp.g, "scratch", lambda: tmp_path)
    monkeypatch.setattr(lpp.g, "set_dir", lambda root, short: tmp_path / short)
    monkeypatch.setattr(lpp.g, "status_of", lambda case, copy, gpu: np.zeros(n, dtype=np.int64))
    ref = "".join(f"{v} 0.250000\n" for v in range(n)).encode()
    monkeypatch.setattr(lpp.g, "run_record", lambda d, stem: {"labels": ref})
    monkeypatch.setattr(lpp, "decompressed", lambda src, tmp: src)
    monkeypatch.setattr(lpp.dlt, "read", lambda path: None)
    monkeypatch.setattr(lpp, "compare_traces", lambda ours, golden, skip: (281, []))
    count = {"port": 0, "original": 0}
    original_exe = tmp_path / "dynLP"
    original_exe.write_bytes(b"\x7fELF the unpatched original")

    def run_compat(exe, argv, env):
        out = Path(argv[argv.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if "--trace" in argv:
            return None
        side = "original" if Path(exe) == original_exe else "port"
        k = count[side]
        count[side] += 1
        its = (port_iterations if side == "port" else original_iterations)(k)
        rows = ["batch,iterations,frontier_visits,components,accuracy"]
        rows += [f"{b},{its[b]},1,{th['components'][b]},0.5" for b in range(batches)]
        (out / "results.csv").write_text("\n".join(rows) + "\n")
        label = port_label(k) if callable(port_label) else port_label
        (out / "dynlp_labels.txt").write_text("".join(f"{v} {label:.6f}\n" for v in range(n)))
        return None

    monkeypatch.setattr(lpp, "run_compat", run_compat)
    exe = tmp_path / "dyng-compat-dynlp"
    exe.write_bytes(b"port")
    args = argparse.Namespace(exe=exe, runs=None, rate=True, original=original_exe,
                              cases=["5k5"], json=tmp_path / "rec.json")  # fmt: skip
    monkeypatch.setattr(lpp, "build_preset", lambda exe: {})
    rc = lpp.cmd_inplace(args)
    import json

    return rc, json.loads((tmp_path / "rec.json").read_text()), count


INSIDE = [1, 10, 10]
OUTSIDE = [1, 10, 13]  # batch 2 above the envelope's 12, the sum (24) inside


def test_the_rate_rule_passes_the_originals_own_rate(monkeypatch, tmp_path: Path) -> None:
    """30 runs each; the port outside in 2, the original in 2 (the 2026-11-14 evidence at 50k):
    within; both sides ran the rule's 30 runs, alternated."""
    two = lambda k: OUTSIDE if k in (3, 17) else INSIDE  # noqa: E731
    rc, doc, count = _fake_inplace(monkeypatch, tmp_path, two, two)
    case = doc["cases"][0]
    assert rc == 0 and doc["within"] and doc["rule"] == "rate", case.get("rate")
    assert count == {"port": 30, "original": 30}
    rate = case["rate"]
    assert rate["port"]["outside"] == 2 and rate["original"]["outside"] == 2
    assert abs(rate["p_value"] - 0.6940231935771632) < 1e-12
    assert rate["rule"] == {"runs": 30, "alpha": 0.05, "test": "fisher_exact_one_sided"}
    assert len(case["original_runs"]) == 30 and len(case["runs"]) == 30


def test_the_plain_jacobi_mutation_is_still_detected_under_the_rate_rule(
    monkeypatch, tmp_path: Path
) -> None:
    """AD9's in_place mutation (plain Jacobi in place of the half-sweeps) takes more iterations
    on every batch: every port run outside the envelope against the original's 2 of 30, so the
    rate rule fails the case (p ~ 4e-15), as the every-run rule did."""
    jacobi = lambda k: [1, 13, 13]  # noqa: E731
    two = lambda k: OUTSIDE if k in (3, 17) else INSIDE  # noqa: E731
    rc, doc, _ = _fake_inplace(monkeypatch, tmp_path, jacobi, two)
    rate = doc["cases"][0]["rate"]
    assert rc == 1 and not doc["within"]
    assert rate["port"]["outside"] == 30 and rate["significantly_higher"]
    assert rate["p_value"] < 1e-14


def test_the_rate_rule_keeps_the_run_by_run_checks(monkeypatch, tmp_path: Path) -> None:
    """Under the rate rule the labels, the components and the iteration sum are still judged on
    every port run: one run's labels beyond T_max, or one run's sum outside the envelope, fails
    the case although the rate is the original's."""
    rc, doc, _ = _fake_inplace(monkeypatch, tmp_path, lambda k: INSIDE, lambda k: INSIDE,
                               port_label=lambda k: 0.5 if k == 7 else 0.25)  # fmt: skip
    case = doc["cases"][0]
    assert rc == 1 and case["rate"]["within"] and not case["within"]
    assert not case["runs"][7]["checked_every_run"]
    rc, doc, _ = _fake_inplace(monkeypatch, tmp_path, lambda k: [1, 13, 13] if k == 4 else INSIDE,
                               lambda k: INSIDE)  # fmt: skip
    assert rc == 0, doc["cases"][0]["rate"]  # one per-batch excursion: 1 of 30 against 0, p 0.5
    rc, doc, _ = _fake_inplace(monkeypatch, tmp_path, lambda k: [2, 12, 14] if k == 4 else INSIDE,
                               lambda k: INSIDE)  # fmt: skip
    assert rc == 1 and not doc["cases"][0]["runs"][4]["iteration_check"]["sum_within"]


def test_the_rate_rule_refuses_other_run_counts_and_rules(monkeypatch, tmp_path: Path) -> None:
    """--rate takes the frozen rule's 30 runs (never --runs), and refuses a goldens.toml rate rule
    that is not ADR 0111's."""
    import argparse

    import pytest

    args = argparse.Namespace(exe=tmp_path, runs=10, rate=True, original=tmp_path / "x",
                              cases=[], json=None)  # fmt: skip
    frozen = {"rule": {}, "thresholds": {}, "rate_rule": dict(lpp.g.RATE_RULE)}
    monkeypatch.setattr(lpp, "frozen_inplace", lambda: frozen)
    with pytest.raises(lpp.g.GoldenError, match="rule's 30 runs"):
        lpp.cmd_inplace(args)
    frozen["rate_rule"]["alpha"] = 0.01
    args.runs = None
    with pytest.raises(lpp.g.GoldenError, match="not ADR 0111's"):
        lpp.cmd_inplace(args)
