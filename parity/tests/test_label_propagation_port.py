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
