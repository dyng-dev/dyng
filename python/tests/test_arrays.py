# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Array: zero-copy DLPack / NumPy views that keep their owner alive, and staleness."""

from __future__ import annotations

import gc
import weakref

import dyng
import numpy as np
import pytest


@pytest.fixture
def tree() -> dyng.sssp.Result:
    g = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], [4, 1, 1, 5])
    return dyng.sssp.compute(g, 0)


def test_array_attributes(tree: dyng.sssp.Result) -> None:
    d = tree.distances
    assert isinstance(d, dyng.Array)
    assert d.dtype == np.int64 and d.shape == (4,) and d.ndim == 1 and d.size == 4
    assert len(d) == 4 and list(d) == [0, 4, 1, 6] and d[3] == 6
    assert d[1:3].tolist() == [4, 1]
    assert d.device == "cpu" and d.is_current
    assert d.__dlpack_device__() == (1, 0)
    assert "dtype=int64" in repr(d)
    assert (d == np.array([0, 4, 1, 6])).all()
    assert tree.parents.dtype == np.int32 and tree.parents.tolist() == [-1, 0, 0, 2]


def test_numpy_round_trips_are_zero_copy(tree: dyng.sssp.Result) -> None:
    d = tree.distances
    via_interface = np.asarray(d)
    via_dlpack = np.from_dlpack(d)
    view = d.to_numpy(copy=False)
    ptr = via_interface.__array_interface__["data"][0]
    assert via_dlpack.__array_interface__["data"][0] == ptr
    assert view.__array_interface__["data"][0] == ptr
    assert not via_interface.flags.writeable and not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 1
    copy = d.to_numpy()
    assert copy.flags.writeable and copy.__array_interface__["data"][0] != ptr
    assert np.array_equal(copy, via_dlpack)
    assert np.array(d, dtype=np.float64).tolist() == [0.0, 4.0, 1.0, 6.0]


def test_views_keep_their_owner_alive() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [2, 3])
    tree = dyng.sssp.compute(g, 0)
    owner = weakref.ref(tree._native)
    arr = tree.distances
    view = np.from_dlpack(arr)
    del tree, arr
    gc.collect()
    assert owner() is not None  # the NumPy view keeps the native result alive
    assert view.tolist() == [0, 2, 5]
    del view
    gc.collect()
    assert owner() is None  # and releases it


def test_arrays_of_an_updated_result_are_stale(tree: dyng.sssp.Result) -> None:
    g_arr = tree.distances
    g = dyng.Graph.from_edges([0], [1], [1])
    t2 = dyng.sssp.compute(g, 0)
    old = t2.distances
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([1], [2], [1])), t2)
    assert not old.is_current
    for use in (
        lambda: old.to_numpy(),
        lambda: np.asarray(old),
        lambda: old.__dlpack__(),
        lambda: len(old),
    ):
        with pytest.raises(dyng.StaleResultError, match="updated since"):
            use()
    assert "stale" in repr(old)
    assert t2.distances.tolist() == [0, 1, 2]  # read again: current
    assert g_arr.is_current  # another result is not affected


def test_cycle_counts_array() -> None:
    g = dyng.Graph.from_edges([0, 1, 2, 1], [1, 2, 0, 0])
    h = dyng.cycle_count.compute(g, max_length=3)
    c = h.counts
    assert c.dtype == np.uint64 and c.tolist() == [0, 0, 1, 1]
    assert np.from_dlpack(c).tolist() == [0, 0, 1, 1]


def test_torch_round_trip(tree: dyng.sssp.Result) -> None:
    torch = pytest.importorskip("torch")
    t = torch.from_dlpack(tree.distances)
    assert t.tolist() == [0, 4, 1, 6]
    assert tree.distances.to_torch().dtype == torch.int64


def test_cupy_round_trip(tree: dyng.sssp.Result) -> None:
    cupy = pytest.importorskip("cupy")
    try:
        c = tree.distances.to_cupy()
    except Exception as e:  # pragma: no cover - no device
        pytest.skip(f"cupy without a device: {e}")
    assert cupy.asnumpy(c).tolist() == [0, 4, 1, 6]
