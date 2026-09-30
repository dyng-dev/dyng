# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Array: zero-copy DLPack / NumPy views that keep their owner alive, and staleness."""

from __future__ import annotations

import copy
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


def test_views_keep_their_memory_alive() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [2, 3])
    tree = dyng.sssp.compute(g, 0)
    owner = weakref.ref(tree._native)
    arr = tree.distances
    view = np.from_dlpack(arr)
    del tree, arr
    gc.collect()
    # The view holds the result's state (not the Python object): its memory stays valid.
    assert owner() is None
    junk = [np.full(3, 777) for _ in range(200)]
    assert view.tolist() == [0, 2, 5]
    del junk


def _grow(n: int, extra: int) -> dyng.EdgeBatch:
    src = np.arange(n - 1, n - 1 + extra, dtype=np.int64)
    return dyng.EdgeBatch(insert=(src, src + 1, np.ones(extra, dtype=np.int32)))


EXPORTS = {
    "from_dlpack": np.from_dlpack,
    "asarray": np.asarray,
    "to_numpy(copy=False)": lambda a: a.to_numpy(copy=False),
    "memoryview": lambda a: np.asarray(memoryview(a._nd)),
}


@pytest.mark.parametrize("how", sorted(EXPORTS))
def test_exports_survive_an_update_that_grows_the_graph(how: str) -> None:
    # An update that adds vertices reallocates the result's arrays; views exported before it
    # must keep showing the old state (the update copies the state while a view is alive).
    n = 20_000
    src = np.arange(n - 1)
    g = dyng.Graph.from_edges(src, src + 1, np.ones(n - 1, dtype=np.int32))
    tree = dyng.sssp.compute(g, 0)
    arr = tree.distances
    view = EXPORTS[how](arr)
    expected = np.arange(n, dtype=np.int64)
    assert np.array_equal(view, expected)
    dyng.sssp.update(g, _grow(n, 30_000), tree)
    junk = [np.full(64, 777) for _ in range(2000)]  # reuse freed heap memory, if any
    assert np.array_equal(view, expected)
    assert not arr.is_current
    with pytest.raises(dyng.StaleResultError):
        arr.tolist()
    now = tree.distances.to_numpy()
    assert now.size == n + 30_000 and now[-1] == n + 30_000 - 1
    del junk


def test_update_copies_the_state_only_while_a_view_is_alive() -> None:
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 3], [5, 5, 5])
    tree = dyng.sssp.compute(g, 0)

    def data_pointer() -> int:
        return int(np.from_dlpack(tree.distances).__array_interface__["data"][0])

    before = data_pointer()
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([0], [2], [1])), tree)  # no view alive
    assert data_pointer() == before  # updated in place
    view = np.from_dlpack(tree.distances)
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([0], [3], [1])), tree)
    assert view.tolist() == [0, 5, 1, 6]  # the old state, unchanged
    assert tree.distances.tolist() == [0, 5, 1, 1]
    assert data_pointer() != int(view.__array_interface__["data"][0])


def test_cycle_count_exports_survive_updates() -> None:
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0])
    hist = dyng.cycle_count.compute(g, max_length=3)
    arr = hist.counts
    view = np.from_dlpack(arr)
    dyng.cycle_count.update(g, dyng.EdgeBatch(delete=([2], [0])), hist)
    assert view.tolist() == [0, 0, 0, 1] and hist.counts.tolist() == [0, 0, 0, 0]
    assert not arr.is_current


def test_copies_of_a_result_are_independent() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    tree = dyng.sssp.compute(g, 0)
    arr = tree.distances
    for twin in (copy.copy(tree), copy.deepcopy(tree)):
        assert twin._native is not tree._native
        assert twin.distances.tolist() == [0, 1, 2]
    alias = copy.copy(tree)
    dyng.sssp.update(g, _grow(3, 5_000), alias)  # updating the copy leaves the original stale
    assert arr.is_current and arr.tolist() == [0, 1, 2]
    assert len(alias.distances) == 5_003
    hist = dyng.cycle_count.compute(dyng.Graph.from_edges([0, 1], [1, 0]), max_length=2)
    assert copy.deepcopy(hist).counts.tolist() == hist.counts.tolist()
    assert copy.copy(hist)._native is not hist._native


def test_generation_lives_in_the_native_result() -> None:
    # Any wrapper of the same native result sees the update (no per-wrapper counter).
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    tree = dyng.sssp.compute(g, 0)
    arr = tree.distances
    twin = dyng.sssp.Result._wrap(tree._native, tree.vertex_dtype, tree._resources)
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([0], [2], [1])), twin)
    assert not arr.is_current
    assert tree._native.generation == 1


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
