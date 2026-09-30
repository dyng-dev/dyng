# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""copy and pickle: value objects round-trip, native containers clone or explain themselves."""

from __future__ import annotations

import copy
import pickle

import dyng
import numpy as np
import pytest


def test_a_used_batch_pickles_and_deep_copies() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    b = dyng.EdgeBatch(insert=(np.array([0], np.int64), np.array([2], np.int64), [3]))
    g.apply(b)  # a used batch must stay a plain value object
    for twin in (pickle.loads(pickle.dumps(b)), copy.deepcopy(b)):
        assert isinstance(twin, dyng.EdgeBatch)
        assert twin.insert_src.tolist() == [0] and twin.insert_dst.tolist() == [2]
        assert twin.insert_weights is not None and twin.insert_weights.tolist() == [[3]]
        g2 = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
        g2.apply(twin)
        assert g2.num_edges == 3


@pytest.mark.parametrize("backend", ["sequential", "openmp"])
def test_resources_pickle_to_equal_resources(backend: str) -> None:
    r = dyng.Resources(backend, num_threads=2) if backend == "openmp" else dyng.Resources(backend)
    r.copy_policy = "warn"
    twin = pickle.loads(pickle.dumps(r))
    assert (twin.backend, twin.num_threads, twin.copy_policy) == (backend, r.num_threads, "warn")
    assert twin._native is not r._native
    assert copy.copy(r)._native is r._native and copy.deepcopy(r)._native is r._native


def test_graphs_and_results_deep_copy_as_clones() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    for twin in (copy.copy(g), copy.deepcopy(g)):
        assert twin._native is not g._native and twin.num_edges == 2
        twin.apply(dyng.EdgeBatch(delete=([0], [1])))
        assert g.num_edges == 2
    tree = dyng.sssp.compute(g, 0)
    assert copy.deepcopy(tree).distances.tolist() == [0, 1, 2]


def test_native_containers_explain_how_to_send_them() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 0])
    tree = dyng.sssp.compute(dyng.Graph.from_edges([0], [1], [1]), 0)
    hist = dyng.cycle_count.compute(g, max_length=2)
    for obj, hint in ((g, "from_csr"), (tree, "from_arrays"), (hist, "counts")):
        with pytest.raises(TypeError, match=hint):
            pickle.dumps(obj)


def test_arrays_pickle_as_numpy_copies() -> None:
    tree = dyng.sssp.compute(dyng.Graph.from_edges([0, 1], [1, 2], [2, 3]), 0)
    a = tree.distances
    out = pickle.loads(pickle.dumps(a))
    assert isinstance(out, np.ndarray) and out.tolist() == [0, 2, 5]
    assert isinstance(copy.deepcopy(a), np.ndarray)
    assert copy.copy(a) is a
