# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The algorithms of integer or no weights refuse a float32 graph by name (R040's review).

label_propagation's graphs of 0.4 hold float32 weights (Graph.with_capacity, or Graph.from_edges
with float32 weights). sssp and mosp read them as "unweighted" before, and cycle_count,
dynamic_bfs and triangle_delta failed with a bare TypeError from the native overloads; each now
raises NotSupportedError naming the algorithm, the float32 weights and the fix.
"""

from __future__ import annotations

import dyng
import numpy as np
import pytest

SEQ = dyng.Resources("sequential")


def _slotted() -> dyng.Graph:
    return dyng.Graph.with_capacity([2, 2, 2], resources=SEQ)


def _compact() -> dyng.Graph:
    return dyng.Graph.from_edges(
        [0, 1, 2],
        [1, 2, 0],
        np.array([0.5, 1.0, 0.25], dtype=np.float32),
        directed=False,
        resources=SEQ,
    )


CALLS = {
    "sssp": lambda g: dyng.sssp.compute(g, 0, resources=SEQ),
    "mosp": lambda g: dyng.mosp.compute(g, 0, resources=SEQ),
    "cycle_count": lambda g: dyng.cycle_count.compute(g, resources=SEQ),
    "dynamic_bfs": lambda g: dyng.dynamic_bfs.compute(g, 0, resources=SEQ),
    "triangle_delta": lambda g: dyng.triangle_delta.compute(g, resources=SEQ),
}


@pytest.mark.parametrize("build", [_slotted, _compact], ids=["with_capacity", "from_edges"])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_float32_graph_is_refused_by_name(name: str, build) -> None:
    g = build()
    assert g.weight_dtype == np.float32 and not g.weighted
    with pytest.raises(dyng.NotSupportedError) as e:
        CALLS[name](g)
    message = str(e.value)
    assert f"dyng.{name} does not read float32-weighted graphs" in message
    assert "int32 weights" in message and "label_propagation" in message
    assert "unweighted (build it with weights=)" not in message


def test_an_unweighted_graph_keeps_its_message() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], resources=SEQ)
    with pytest.raises(dyng.NotSupportedError, match="this graph is unweighted"):
        dyng.sssp.compute(g, 0, resources=SEQ)
