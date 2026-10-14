# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Container-level id parity from Python: one small dump of each original replayed through
dyng.Hypergraph gives the original's ids (M8b, acceptance criterion 17).

The dumps are the committed batch extracts of the CBST trace goldens
(cpp/tests/data/goldens/hypergraph_batches, SUBSET.txt), decoded with the repository's trace
reader (parity/escher_trace.py) by the rules of the C++ replay
(cpp/tests/parity/escher/hypergraph_ids.cpp):

- ESCHER-GPU@abcf9b2 (A), 016-main: ``main --input small.hg --batches 3 --batch-size 20 --seed 3``
  under ``id_reuse="erase_first"`` and ``CapacityPolicy.escher_gpu(capacity)``: every batch's
  ``inserted_ids`` equal A's ``insertedKeys - 1`` and the h2v rows by id equal A's model; and
  ``generators.legacy.EscherBatchGenerator`` draws A's batches again from the container.
- MOSP_ESCHER@4b86159 (B), the first 9 sessions of 014-test_h2h_delta under
  ``id_reuse="insert_first"`` and ``CapacityPolicy.mosp_escher(extra)`` with B's headroom
  factor: every hyperedge-kind batch's ids equal B's ``finalIds - 1`` (a session stops at its
  first vertex-kind batch, which is 0.4's).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import dyng
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXTRACTS = REPO / "cpp" / "tests" / "data" / "goldens" / "hypergraph_batches"


def _trace_module() -> Any:
    reader = REPO / "parity" / "escher_trace.py"
    if not reader.is_file():  # pragma: no cover - an sdist without the parity harness
        pytest.skip("parity/escher_trace.py (the trace reader) is not in this tree")
    sys.path.insert(0, str(reader.parent))
    try:
        import escher_trace
    finally:
        sys.path.pop(0)
    return escher_trace


def _ints(trace: Any, value: Any) -> list[int]:
    return list(value.values(trace))


def _rows(trace: Any, fields: dict[str, Any], name: str, shift: int) -> list[list[int]]:
    values = _ints(trace, fields[f"{name}.values"])
    prefix = _ints(trace, fields[f"{name}.prefix"])
    out, begin = [], 0
    for end in prefix:
        out.append([v - shift for v in values[begin:end]])
        begin = end
    return out


def _h2v_rows(h: dyng.Hypergraph) -> list[list[int]]:
    v = h.h2v()
    return [v.row(i).tolist() for i in range(v.num_rows)]


def _same_rows(got: list[list[int]], want: list[list[int]]) -> bool:
    size = max(len(got), len(want))  # the bound may differ by trailing dead ids
    return got + [[]] * (size - len(got)) == want + [[]] * (size - len(want))


@pytest.mark.parametrize("backend", ["sequential", "openmp"])
def test_escher_gpu_dump_gives_its_ids(backend: str) -> None:
    et = _trace_module()
    trace = et.Trace(EXTRACTS / "escher_gpu_016-main.cbt")
    res = dyng.Resources(backend)
    capacity: dict[str, int] = {}
    h: dyng.Hypergraph | None = None
    model: list[list[int]] = []
    pending: dict[str, Any] = {}
    engine = generator = None
    batches = ids = 0
    for tag, f in trace.records():
        if tag == "tree":
            capacity[f["dataset_name"]] = int(f["payload_capacity"])
        if tag != "batch":
            continue
        if f["hook"] == "escher_gpu.build" and f["phase"] == "exit":
            model = [sorted(set(r)) for r in _rows(trace, f, "loaded_rows", 1)]
            n = int(f["max_vertex_id"])
            # ESCHER-GPU's payloadCapacity: max(--payload-capacity, size + size / 8 + 1024).
            flat = sum((len(r) + 4) // 4 * 4 for r in model)
            formula = flat + flat // 8 + 1024
            requested = capacity.get("H2V", formula)
            h = dyng.Hypergraph.from_lists(
                model,
                num_vertices=n,
                id_reuse="erase_first",
                capacity=dyng.CapacityPolicy.escher_gpu(requested if requested > formula else 0),
                line_graph="none",
                resources=res,
            )
            assert h.store_stats().payload_capacity == requested
            # A loaded small.hg (--input, so sample mode) and drew its batches from one
            # std::mt19937_64(--seed 3); --batch-size 20.
            engine = dyng.generators.legacy.EscherEngine(3)
            generator = dyng.generators.legacy.EscherBatchGenerator(
                engine,
                batch_size=20,
                insert_mode="sample",
                max_cardinality=max(len(r) for r in model),
                min_vertex=0,
                max_vertex=n - 1,
            )
        elif f["hook"] == "escher_gpu.update" and f["phase"] == "enter":
            pending = {
                "deleted": [k - 1 for k in _ints(trace, f["deleted_keys"])],
                "inserted": _rows(trace, f, "inserted_rows", 1),
            }
        elif f["hook"] == "escher_gpu.update" and f["phase"] == "exit":
            assert h is not None and generator is not None
            drawn = generator.next(h)
            assert drawn.delete_hyperedges.tolist() == pending["deleted"]
            assert drawn.insertions() == pending["inserted"]
            s = h.apply(
                dyng.HyperedgeBatch(
                    delete_hyperedges=pending["deleted"], insert_hyperedges=pending["inserted"]
                )
            )
            want = [k - 1 for k in _ints(trace, f["inserted_keys"])]
            assert s.inserted_ids.dtype == np.int64
            assert s.inserted_ids.tolist() == want
            for i in pending["deleted"]:
                model[i] = []
            for i, r in zip(want, pending["inserted"], strict=True):
                model.extend([] for _ in range(i + 1 - len(model)))
                model[i] = sorted(set(r))
            assert _same_rows(_h2v_rows(h), model)
            h.check_integrity()
            batches += 1
            ids += len(want)
    assert (batches, ids) == (3, 30)


def _caps(sizing: list[tuple[int, int]]) -> tuple[float, int]:
    """MOSP_ESCHER's Caps of a session: the factor (1.3 by default; 2.0, 1.25, 1.5 in its tests)
    on which the trees agree on one extra, and that extra (hypergraph_ids.cpp, derive_caps)."""
    for factor in (1.3, 2.0, 1.25, 1.5):
        extras = {cap - int(flat * factor) for flat, cap in sizing if cap < 2_000_000_000}
        if len(extras) == 1 and min(extras) >= 0:
            return factor, extras.pop()
    raise AssertionError(f"no Caps of MOSP_ESCHER fits {sizing}")


@pytest.mark.parametrize("backend", ["sequential", "openmp"])
def test_mosp_escher_dump_gives_its_ids(backend: str) -> None:
    et = _trace_module()
    trace = et.Trace(EXTRACTS / "mosp_escher_014-test_h2h_delta.cbt")
    res = dyng.Resources(backend)
    names: dict[int, str] = {}
    h: dyng.Hypergraph | None = None
    session: dict[str, Any] = {}
    pending: dict[str, Any] | None = None
    sessions = batches = ids = 0
    for tag, f in trace.records():
        if tag == "tree":
            names[int(f["tree"])] = f["dataset_name"]
            continue
        if tag == "op" and session.get("loading") and f["op"] == "construct":
            if names[int(f["tree"])] == "v2h":
                session["n"] = int(f["in.num_records"])
            session["sizing"].append((int(f["in.flat_payload_size"]), int(f["in.capacity"])))
            continue
        if tag != "batch":
            continue
        hook, phase = f["hook"], f["phase"]
        if hook == "mosp_escher.bulk_load" and phase == "enter":
            session = {
                "loading": True,
                "rows": _rows(trace, f, "rows", 0),
                "weights": _ints(trace, f["weights"]),
                "virtual": {int(f["source_he"]), int(f["target_he"])},
                "sizing": [],
            }
            h = None
        elif hook == "mosp_escher.bulk_load" and phase == "exit":
            session["loading"] = False
            factor, extra = _caps(session["sizing"])
            capacity = dyng.CapacityPolicy.mosp_escher(extra)
            # 1.3 - 1.0 is not 0.3 in binary; the preset's 0.3 is the value proven equal.
            capacity.initial_headroom = 0.3 if factor == 1.3 else factor - 1.0
            # B's virtual source and target are ordinary hyperedges here (weight 0 loaded as 1).
            h = dyng.Hypergraph.from_lists(
                session["rows"],
                [max(1, w) for w in session["weights"]],
                num_vertices=session["n"],
                id_reuse="insert_first",
                capacity=capacity,
                line_graph="none",
                resources=res,
            )
            sessions += 1
        elif hook == "mosp_escher.begin_batch" and phase == "enter":
            pending = None
            if h is None:
                continue
            if _ints(trace, f["vtx_delete.he"]) or _ints(trace, f["vtx_insert.he"]):
                h = None  # a vertex-kind batch (incidence edits, 0.4): the session stops here
                continue
            pending = {
                # B skips the ops on its virtual hyperedges (the harness filters them)
                "deleted": [
                    i - 1 for i in _ints(trace, f["he_delete"]) if i not in session["virtual"]
                ],
                "inserted": _rows(trace, f, "he_insert", 0),
                "weights": _ints(trace, f["he_insert.weight"]),
            }
        elif hook == "mosp_escher.begin_batch" and phase == "exit" and pending is not None:
            assert h is not None
            s = h.apply(
                dyng.HyperedgeBatch(
                    pending["deleted"], pending["inserted"], pending["weights"] or None
                )
            )
            want = [i - 1 for i in _ints(trace, f["final_ids"])]
            assert s.inserted_ids.tolist() == want
            weights = h.weights
            assert weights is not None
            assert [int(weights[i]) for i in want] == _ints(trace, f["final_weights"])[: len(want)]
            h.check_integrity()
            batches += 1
            ids += len(want)
            pending = None
    assert (sessions, batches, ids) == (9, 10, 258)
