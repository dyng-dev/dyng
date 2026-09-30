# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.EdgeBatch: construction and checked conversion to the graph's types."""

from __future__ import annotations

import dyng
import numpy as np
import pytest


def test_construction() -> None:
    b = dyng.EdgeBatch(insert=([0, 5], [7, 9], [3, 2]), delete=([1], [2]))
    assert (b.num_insertions, b.num_deletions, b.num_weights) == (2, 1, 1)
    assert b.insert_weights is not None and b.insert_weights.shape == (2, 1)
    assert not b.empty and dyng.EdgeBatch().empty
    assert "insertions=2" in repr(b)
    unweighted = dyng.EdgeBatch(insert=([0], [1]))
    assert unweighted.num_weights == 0 and unweighted.insert_weights is None
    multi = dyng.EdgeBatch(insert=([0], [1], [[1, 2, 3]]))
    assert multi.num_weights == 3


def test_invalid_batches() -> None:
    with pytest.raises(dyng.InvalidArgumentError, match="differ in length"):
        dyng.EdgeBatch(insert=([0, 1], [1]))
    with pytest.raises(dyng.InvalidArgumentError, match="differ in length"):
        dyng.EdgeBatch(delete=([0], [1, 2]))
    with pytest.raises(dyng.InvalidArgumentError, match="must be"):
        dyng.EdgeBatch(insert=[0, 1, 2, 3])
    with pytest.raises(dyng.InvalidArgumentError, match="delete must be"):
        dyng.EdgeBatch(delete=([0], [1], [2]))
    with pytest.raises(dyng.InvalidArgumentError, match="integers"):
        dyng.EdgeBatch(insert=([0.5], [1]))
    with pytest.raises(dyng.InvalidArgumentError, match="weights must be integers"):
        dyng.EdgeBatch(insert=([0], [1], [0.5]))


def test_int64_batch_ids_are_checked_for_an_int32_graph() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    tree = dyng.sssp.compute(g, 0)
    b = dyng.EdgeBatch(insert=(np.array([0], np.int64), np.array([2], np.int64), [1]))
    dyng.sssp.update(g, b, tree)  # values fit: converted
    assert tree.distances.tolist() == [0, 1, 1]
    too_big = dyng.EdgeBatch(insert=(np.array([0], np.int64), np.array([2**40], np.int64), [1]))
    with pytest.raises(dyng.InvalidArgumentError, match="does not fit int32"):
        dyng.sssp.update(g, too_big, tree)
    assert g.version == 1  # nothing changed


def test_weights_for_an_unweighted_graph() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 0])
    with pytest.raises(dyng.InvalidArgumentError, match="unweighted"):
        g.apply(dyng.EdgeBatch(insert=([0], [2], [5])))
    g.apply(dyng.EdgeBatch(insert=([0], [2])))
    assert g.num_edges == 3


def test_missing_weights_for_a_weighted_graph() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 0], [1, 1])
    with pytest.raises(dyng.InvalidArgumentError):
        g.apply(dyng.EdgeBatch(insert=([0], [2])))
    g.apply(dyng.EdgeBatch(delete=([0], [1])))  # no insertions: no weights needed
    assert g.num_edges == 1


def test_zero_copy_for_matching_arrays() -> None:
    s = np.array([0], np.int32)
    b = dyng.EdgeBatch(insert=(s, np.array([1], np.int32), np.array([1], np.int32)))
    assert b.insert_src is s


def test_dlpack_inputs() -> None:
    torch = pytest.importorskip("torch")
    s = torch.tensor([0, 1], dtype=torch.int32)
    d = torch.tensor([1, 2], dtype=torch.int32)
    g = dyng.Graph.from_edges(s, d, torch.tensor([1, 1], dtype=torch.int32))
    assert g.vertex_dtype == np.int32 and g.num_edges == 2
