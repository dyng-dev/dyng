# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.update with several results, and stale results (PLAN Section 5.5)."""

from __future__ import annotations

import dyng
import pytest


def graph(res: dyng.Resources | None = None) -> dyng.Graph:
    return dyng.Graph.from_edges([0, 1, 2, 2], [1, 2, 0, 3], [1, 1, 1, 2], resources=res)


def test_update_several_results(res: dyng.Resources) -> None:
    g = graph(res)
    tree, hist = dyng.sssp.compute(g, 0), dyng.cycle_count.compute(g, max_length=3)
    b = dyng.EdgeBatch(insert=([3], [0], [1]), delete=([2], [0]))
    st_tree, st_hist = dyng.update(g, b, tree, hist)
    assert isinstance(st_tree, dyng.sssp.Stats) and isinstance(st_hist, dyng.cycle_count.Stats)
    assert g.version == 1 and tree.graph_version == 1 and hist.graph_version == 1
    assert tree.distances.tolist() == dyng.sssp.compute(g, 0).distances.tolist()
    assert hist.counts.tolist() == dyng.cycle_count.compute(g, max_length=3).counts.tolist()
    assert st_tree.batch == st_hist.batch  # one apply


def test_one_update_leaves_the_other_stale(res: dyng.Resources) -> None:
    g = graph(res)
    tree, hist = dyng.sssp.compute(g, 0), dyng.cycle_count.compute(g, max_length=3)
    b = dyng.EdgeBatch(delete=([2], [0]))
    dyng.sssp.update(g, b, tree)
    with pytest.raises(dyng.StaleResultError):
        dyng.cycle_count.update(g, b, hist)
    with pytest.raises(dyng.StaleResultError):
        dyng.update(g, b, tree, hist)
    assert g.version == 1  # nothing changed by the failed calls


def test_apply_makes_results_stale() -> None:
    g = graph()
    tree = dyng.sssp.compute(g, 0)
    g.apply(dyng.EdgeBatch(delete=([0], [1])))
    with pytest.raises(dyng.StaleResultError):
        dyng.sssp.update(g, dyng.EdgeBatch(), tree)


def test_result_of_another_graph_is_stale() -> None:
    g1, g2 = graph(), graph()
    tree = dyng.sssp.compute(g1, 0)
    with pytest.raises(dyng.StaleResultError):
        dyng.sssp.update(g2, dyng.EdgeBatch(), tree)


def test_invalid_calls() -> None:
    g = graph()
    tree = dyng.sssp.compute(g, 0)
    with pytest.raises(dyng.InvalidArgumentError, match="no results"):
        dyng.update(g, dyng.EdgeBatch())
    with pytest.raises(dyng.InvalidArgumentError, match="more than once"):
        dyng.update(g, dyng.EdgeBatch(), tree, tree)
    with pytest.raises(TypeError):
        dyng.update(g, dyng.EdgeBatch(), "tree")
    with pytest.raises(dyng.InvalidArgumentError):  # an invalid batch: nothing is changed
        dyng.update(g, dyng.EdgeBatch(insert=([0], [1], [0])), tree)
    assert g.version == 0 and tree.graph_version == 0


def test_same_type_results() -> None:
    g = graph()
    a, b = dyng.sssp.compute(g, 0), dyng.sssp.compute(g, 2)
    stats = dyng.update(g, dyng.EdgeBatch(insert=([3], [1], [1])), a, b)
    assert len(stats) == 2
    assert b.distances.tolist() == dyng.sssp.compute(g, 2).distances.tolist()


def _mosp_graph(res: dyng.Resources) -> dyng.Graph:
    return dyng.Graph.from_edges(
        [0, 1, 2, 2], [1, 2, 0, 3], [[1, 2], [1, 1], [1, 3], [2, 1]], resources=res
    )


def test_results_keep_the_resources_of_their_last_writer() -> None:
    # The native result names its last writer's stream by its raw handle; the Python result keeps
    # that writer's Resources (and with it a stream object of Resources.cuda(stream=...)).
    made, other, third = (dyng.Resources.sequential() for _ in range(3))
    g = graph(made)
    tree, hist = dyng.sssp.compute(g, 0), dyng.cycle_count.compute(g, max_length=3)
    assert tree._writer is made and hist._writer is made
    dyng.update(g, dyng.EdgeBatch(insert=([3], [0], [1])), tree, hist, resources=third)
    assert tree._writer is third and hist._writer is third
    dyng.sssp.update(g, dyng.EdgeBatch(delete=([2], [3])), tree, resources=other)
    assert tree._writer is other
    keep = tree.distances._is_current.keep
    assert keep[0] is made and keep[1] is other  # the Array keeps both
    clone = tree.clone(third)
    assert clone._resources is third and clone._writer is third
    g2 = graph(made)
    hist2 = dyng.cycle_count.compute(g2, max_length=3)
    dyng.cycle_count.update(g2, dyng.EdgeBatch(delete=([2], [0])), hist2, resources=other)
    assert hist2._writer is other and hist2.counts._is_current.keep[1] is other
    mg = _mosp_graph(made)
    mr = dyng.mosp.compute(mg, 0)
    dyng.mosp.update(mg, dyng.EdgeBatch(delete=([2], [3])), mr, resources=third)
    assert mr._writer is third and mr.combined_distances._is_current.keep[1] is third


def test_a_failed_update_follows_the_native_writer() -> None:
    # The native holder records the writer when it starts writing the result (it advances the
    # generation in the same step); the Python side follows the generation.
    made, other = dyng.Resources.sequential(), dyng.Resources.sequential()
    g = graph(made)
    tree = dyng.sssp.compute(g, 0)
    generation = tree._native.generation
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.sssp.update(g, dyng.EdgeBatch(insert=([0], [3], [0])), tree, resources=other)
    assert (tree._writer is other) == (tree._native.generation != generation)
    g.apply(dyng.EdgeBatch(delete=([2], [3])))  # the tree is stale
    generation = tree._native.generation
    with pytest.raises(dyng.StaleResultError):
        dyng.sssp.update(g, dyng.EdgeBatch(delete=([0], [1])), tree, resources=made)
    assert (tree._writer is made) == (tree._native.generation != generation)
    writer, generation = tree._writer, tree._native.generation
    with pytest.raises(TypeError):  # refused in Python: the native result is not touched
        dyng.sssp.update(g, "not a batch", tree, resources=other)  # type: ignore[arg-type]
    assert tree._writer is writer and tree._native.generation == generation


def test_graphs_and_results_keep_the_streams_of_every_resources_used_on_them() -> None:
    # Memory a call adds to a graph or a result is released on that call's stream, maybe much
    # later, so every stream object used on them is kept (dyng._writer.StreamKeep), not only the
    # last writer's. Fake stream objects stand in for CuPy streams (the CPU module has no CUDA).
    import gc
    import weakref

    from dyng import resources as resources_module

    class Stream:
        pass

    made, a, b = (dyng.Resources.sequential() for _ in range(3))
    streams = {r: Stream() for r in (made, a, b)}
    resources_module._streams.update(streams)
    alive = {name: weakref.ref(streams[r]) for name, r in (("made", made), ("a", a), ("b", b))}
    g = graph(made)
    tree = dyng.sssp.compute(g, 0)
    dyng.sssp.update(g, dyng.EdgeBatch(insert=([3], [0], [1])), tree, resources=a)
    dyng.update(g, dyng.EdgeBatch(delete=([2], [3])), tree, resources=b)
    assert len(g._streams) == 3 and len(tree._streams) == 3
    arr = tree.distances
    del streams, made, a, b
    gc.collect()
    assert all(ref() is not None for ref in alive.values())
    del g, tree
    gc.collect()
    assert all(ref() is not None for ref in alive.values())  # the Array keeps the result's
    del arr
    gc.collect()
    assert all(ref() is None for ref in alive.values())


def test_a_graph_keeps_the_stream_of_apply_and_compute() -> None:
    from dyng import resources as resources_module

    made, other, third = (dyng.Resources.sequential() for _ in range(3))
    s_other, s_third = object(), object()
    resources_module._streams[other] = s_other
    resources_module._streams[third] = s_third
    g = graph(made)
    assert len(g._streams) == 0  # made with no stream object
    g.apply(dyng.EdgeBatch(insert=([3], [0], [1])), resources=other)
    dyng.cycle_count.compute(g, max_length=3, resources=third)
    assert set(map(id, g._streams.streams())) == {id(s_other), id(s_third)}
    g.apply(dyng.EdgeBatch(delete=([3], [0])), resources=other)
    assert len(g._streams) == 2
