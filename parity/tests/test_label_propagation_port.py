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
