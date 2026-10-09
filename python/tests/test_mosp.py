# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.mosp: the thesis example, compute, update, options, imported trees, composition, and a
property-based comparison with a pure-Python oracle (Dijkstra per objective, the combined graph,
Dijkstra on it, the path costs along the MOSP tree)."""

from __future__ import annotations

import copy
import doctest
import math
import pickle
from collections.abc import Sequence
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.testing import oracles

INF = dyng.mosp.INFINITE_DISTANCE

# The graph of thesis Chapter 4, Fig. "Finding a single MOSP" (MOSP-CUDA's mospTest
# thesis-example): u1..u7 are 0..6, source u1, three objectives.
THESIS_EDGES = [
    (0, 1, (2, 1, 5)),
    (0, 2, (4, 1, 1)),
    (1, 3, (2, 4, 2)),
    (2, 1, (10, 15, 2)),
    (2, 3, (5, 16, 3)),
    (3, 4, (1, 1, 1)),
    (4, 1, (4, 3, 2)),
    (4, 5, (1, 2, 2)),
    (4, 6, (5, 6, 2)),
    (5, 6, (1, 1, 1)),
]


def thesis_graph(res: dyng.Resources | None = None) -> dyng.Graph:
    src = [e[0] for e in THESIS_EDGES]
    dst = [e[1] for e in THESIS_EDGES]
    w = np.array([e[2] for e in THESIS_EDGES], dtype=np.int32)
    return dyng.Graph.from_edges(src, dst, w, properties="mosp_compatible", resources=res)


def thesis_batch() -> dyng.EdgeBatch:
    """Delete (u2,u4), (u5,u2); insert (u4,u6):(10,2,12), (u2,u6):(12,1,14)."""
    return dyng.EdgeBatch(
        insert=(np.array([3, 1]), np.array([5, 5]), np.array([[10, 2, 12], [12, 1, 14]])),
        delete=(np.array([1, 4]), np.array([3, 1])),
    )


def path_to(paths: dyng.mosp.Result, v: int) -> list[int]:
    parents = paths.combined_parents.tolist()
    out = [v]
    while parents[out[-1]] != -1:
        out.append(parents[out[-1]])
    return out


# -------------------------------------------------------------------------------------------------
# A pure-Python oracle (shares no code with the library)
# -------------------------------------------------------------------------------------------------


def _dijkstra(
    n: int, edges: dict[tuple[int, int], int], source: int
) -> tuple[list[int], list[int]]:
    """Distances and canonical parents (lowest-id tight in-neighbour)."""
    model = oracles.GraphModel(n, [], [])
    model.edges = dict(edges)
    return oracles.dijkstra(model, source)


def mosp_oracle(
    n: int,
    columns: Sequence[dict[tuple[int, int], int]],
    source: int,
    preferences: Sequence[int] = (),
) -> tuple[list[list[int]], list[list[int]], list[int], list[int], list[list[int]]]:
    """(distances per objective, parents per objective, combined distances, MOSP tree, costs)."""
    k = len(columns)
    pref = list(preferences) or [1] * k
    scale = math.lcm(*pref)
    trees = [_dijkstra(n, c, source) for c in columns]
    member: dict[tuple[int, int], int] = {}
    for i, (_, parents) in enumerate(trees):
        for v, p in enumerate(parents):
            if p != -1:
                member[(p, v)] = member.get((p, v), 0) + scale // pref[i]
    combined = {e: scale * (k + 1) - m for e, m in member.items()}
    cd, cp = _dijkstra(n, combined, source)
    costs = [[INF] * k for _ in range(n)]
    for v in range(n):
        if cd[v] >= INF:
            continue
        path, x = [], v
        while x != source:
            path.append((cp[x], x))
            x = cp[x]
        costs[v] = [sum(c[e] for e in path) for c in columns]
    return [t[0] for t in trees], [t[1] for t in trees], cd, cp, costs


def assert_matches_oracle(
    paths: dyng.mosp.Result,
    n: int,
    columns: Sequence[dict[tuple[int, int], int]],
    source: int,
    preferences: Sequence[int] = (),
) -> None:
    d, p, cd, cp, costs = mosp_oracle(n, columns, source, preferences)
    for k in range(len(columns)):
        assert paths.distances(k).tolist() == d[k], f"objective {k}"
        assert paths.parents(k).tolist() == p[k], f"objective {k}"
    assert paths.combined_distances.tolist() == cd
    assert paths.combined_parents.tolist() == cp
    assert paths.path_costs.tolist() == costs


# -------------------------------------------------------------------------------------------------
# The thesis' worked example
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pref", "path", "cost"),
    [
        ([4, 1, 4], [6, 5, 1, 0], [15, 3, 20]),  # sub-figures (d), (e)
        ([4, 4, 1], [6, 4, 3, 2, 0], [15, 24, 7]),  # sub-figures (f), (g)
    ],
)
def test_the_thesis_worked_example(
    res: dyng.Resources, pref: list[int], path: list[int], cost: list[int]
) -> None:
    g = thesis_graph(res)
    paths = dyng.mosp.compute(g, source=0, preferences=pref)
    assert paths.preference_scale == 4 and paths.num_objectives == 3
    st = dyng.mosp.update(g, thesis_batch(), paths)
    # the updated SOSP trees of sub-figures (a), (b), (c)
    trees = [[-1, 0, 0, 2, 3, 4, 5], [-1, 0, 0, 2, 3, 1, 5], [-1, 2, 0, 2, 3, 4, 4]]
    for k in range(3):
        assert paths.parents(k).tolist() == trees[k]
    assert path_to(paths, 6) == path
    costs = paths.path_costs
    assert costs.shape == (7, 3) and costs.ndim == 2 and costs.dtype == np.int64
    assert costs.to_numpy()[6].tolist() == cost
    assert st.preference_scale == 4 and len(st.objectives) == 3
    assert all(isinstance(o, dyng.sssp.Stats) for o in st.objectives)
    assert st.batch.inserted_edges == 2 and st.batch.deleted_edges == 2
    fresh = dyng.mosp.compute(g, 0, preferences=pref)
    assert paths.combined_parents.tolist() == fresh.combined_parents.tolist()
    assert paths.path_costs.tolist() == fresh.path_costs.tolist()
    columns = [{(u, v): w[k] for u, v, w in _thesis_after()} for k in range(3)]
    assert_matches_oracle(paths, 7, columns, 0, pref)


def _thesis_after() -> list[tuple[int, int, tuple[int, ...]]]:
    edges = [e for e in THESIS_EDGES if (e[0], e[1]) not in ((1, 3), (4, 1))]
    return edges + [(3, 5, (10, 2, 12)), (1, 5, (12, 1, 14))]


def test_the_module_example_runs() -> None:
    failures, tests = doctest.testmod(dyng.mosp, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tests > 0 and failures == 0


# -------------------------------------------------------------------------------------------------
# compute, update, options
# -------------------------------------------------------------------------------------------------


def two_objectives(res: dyng.Resources | None = None, **kw: object) -> dyng.Graph:
    w = np.array([[4, 1], [1, 5], [1, 1], [5, 2]], dtype=np.int32)
    return dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], w, resources=res, **kw)


def test_compute_and_accessors(res: dyng.Resources) -> None:
    g = two_objectives(res)
    paths = dyng.mosp.compute(g, 0)
    assert paths.source == 0 and paths.graph_version == 0 and paths.space == "host"
    assert paths.vertex_dtype == np.int32 and paths.num_objectives == 2
    assert paths.preference_scale == 1
    assert paths.distances(0).tolist() == [0, 4, 1, 6]
    assert paths.distances(1).tolist() == [0, 1, 2, 4]
    assert paths.parents(1).tolist() == [-1, 0, 1, 2]
    # combined (K + 1 - m): (0,1) and (2,3) in both trees -> 1; (0,2), (1,2) in one -> 2
    assert paths.combined_distances.tolist() == [0, 1, 2, 3]
    assert paths.combined_parents.tolist() == [-1, 0, 0, 2]
    assert paths.path_costs.tolist() == [[0, 0], [4, 1], [1, 5], [6, 7]]
    assert paths.combined_parents.device == "cpu" and len(paths.path_costs) == 4
    assert "num_objectives=2" in repr(paths)
    for bad in (-1, 2):
        with pytest.raises(dyng.InvalidArgumentError, match="out of range"):
            paths.distances(bad)
    with pytest.raises(dyng.InvalidArgumentError):
        paths.parents(True)  # type: ignore[arg-type]


def test_update_matches_compute_and_arrays_go_stale(res: dyng.Resources) -> None:
    g = two_objectives(res)
    paths = dyng.mosp.compute(g, 0, preferences=[1, 3])
    before = paths.combined_parents
    kept = paths.path_costs.to_numpy(copy=False)
    b = dyng.EdgeBatch(insert=([1, 3], [3, 4], np.array([[1, 1], [2, 9]])), delete=([0], [2]))
    st = dyng.mosp.update(g, b, paths)
    assert isinstance(st, dyng.mosp.Stats) and st.preference_scale == 3
    assert st.combined_edges > 0 and st.affected >= 1 and isinstance(st.engine_used, str)
    with pytest.raises(dyng.StaleResultError):
        before.tolist()
    assert kept.tolist() == [[0, 0], [4, 1], [1, 5], [6, 7]]  # views keep the old state
    fresh = dyng.mosp.compute(g, 0, preferences=[1, 3])
    for k in range(2):
        assert paths.parents(k).tolist() == fresh.parents(k).tolist()
    assert paths.combined_distances.tolist() == fresh.combined_distances.tolist()
    assert paths.path_costs.tolist() == fresh.path_costs.tolist()
    assert paths.graph_version == g.version == 1
    with pytest.raises(dyng.StaleResultError):
        dyng.mosp.update(dyng.Graph.from_edges([0], [1], [[1, 1]]), b, paths)


def test_compute_path_costs_takes_effect_at_the_next_update() -> None:
    # Result.path_costs follows the option of the last compute() / update(), not the current
    # one: set_options() changes it for the next update() (the documented rule, ADR 0035).
    g = two_objectives()
    paths = dyng.mosp.compute(g, 0)
    before = paths.path_costs.tolist()
    paths.set_options(compute_path_costs=False)
    assert paths.path_costs.tolist() == before  # still readable until the next update
    dyng.mosp.update(g, dyng.EdgeBatch(delete=([0], [1])), paths)
    with pytest.raises(dyng.InvalidArgumentError, match="compute_path_costs"):
        paths.path_costs.tolist()
    paths.set_options(compute_path_costs=True)
    with pytest.raises(dyng.InvalidArgumentError, match="compute_path_costs"):
        paths.path_costs.tolist()  # not computed before the next update
    grow = dyng.EdgeBatch(insert=([0], [3], np.array([[9, 9]], dtype=np.int32)))
    dyng.mosp.update(g, grow, paths)
    assert paths.path_costs.tolist() == dyng.mosp.compute(g, 0).path_costs.tolist()


def test_options_and_their_checks() -> None:
    g = two_objectives()
    paths = dyng.mosp.compute(g, 0, preferences=[2, 3], delta=4, num_objectives=2)
    o = paths.options
    assert o.preferences == [2, 3] and o.delta == 4 and paths.preference_scale == 6
    paths.set_options(delta=7, compute_path_costs=False, cuda_engine="operators")
    assert paths.options.delta == 7 and paths.options.cuda_engine == "operators"
    with pytest.raises(dyng.InvalidArgumentError):
        paths.set_options(preferences=[1, 1])  # fixed at compute
    dyng.mosp.update(g, dyng.EdgeBatch(delete=([0], [1])), paths)
    with pytest.raises(dyng.InvalidArgumentError, match="compute_path_costs"):
        paths.path_costs.tolist()
    opts = dyng.mosp.Options(preferences=[1, 2], num_objectives=1)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.mosp.compute(g, 0, options=opts)  # two preferences, one objective
    one = dyng.mosp.compute(g, 0, num_objectives=1)
    assert one.num_objectives == 1 and one.path_costs.shape == (4, 1)
    bad = [
        {"preferences": [0, 1]},
        {"preferences": [1, 2, 3]},
        {"preferences": [2**20 + 1, 1]},  # lcm above 2^20
        {"preferences": [1024, 1025]},  # lcm 1049600, above 2^20
        {"preferences": "12"},
        {"num_objectives": 3},
        {"delta": -1},
        {"cuda_engine": "warp"},
    ]
    for kw in bad:
        with pytest.raises(dyng.InvalidArgumentError):
            dyng.mosp.compute(g, 0, **kw)  # type: ignore[arg-type]
    assert math.lcm(1024, 1025) > dyng.mosp.MAX_PREFERENCE_SCALE
    edge = dyng.mosp.compute(g, 0, preferences=[1023, 1024])  # lcm 1047552: allowed
    assert edge.preference_scale == 1023 * 1024
    with pytest.raises(TypeError):
        dyng.mosp.compute(g, 0, objective=1)  # sssp's option, not mosp's
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.mosp.compute(g, 9)
    with pytest.raises(dyng.NotSupportedError, match="unweighted"):
        dyng.mosp.compute(dyng.Graph.from_edges([0], [1]), 0)


def test_constants_equal_the_cpp_constants() -> None:
    from dyng._backend import native

    assert dyng.mosp.MAX_OBJECTIVES == native.MOSP_MAX_OBJECTIVES
    assert dyng.mosp.MAX_PREFERENCE_SCALE == native.MOSP_MAX_PREFERENCE_SCALE
    assert dyng.mosp.INFINITE_DISTANCE == dyng.sssp.INFINITE_DISTANCE


def test_unreachable_and_int64_ids(res: dyng.Resources) -> None:
    w = np.array([[3, 1], [1, 1]], dtype=np.int32)
    g = dyng.Graph.from_edges(
        np.array([0, 1], np.int64), np.array([1, 2], np.int64), w, num_vertices=4, resources=res
    )
    assert g.vertex_dtype == np.int64
    paths = dyng.mosp.compute(g, 0)
    assert paths.vertex_dtype == np.int64 and paths.combined_parents.dtype == np.int64
    assert paths.combined_distances.tolist() == [0, 1, 2, INF]
    assert paths.path_costs.tolist()[3] == [INF, INF]
    assert paths.path_costs.tolist()[2] == [4, 2]


def test_from_arrays_equals_compute(res: dyng.Resources) -> None:
    g = thesis_graph(res)
    ref = dyng.mosp.compute(g, 0, preferences=[4, 1, 4])
    d = [ref.distances(k).to_numpy() for k in range(3)]
    p = [ref.parents(k).to_numpy() for k in range(3)]
    paths = dyng.mosp.Result.from_arrays(g, 0, d, p, preferences=[4, 1, 4])
    assert paths.combined_parents.tolist() == ref.combined_parents.tolist()
    assert paths.path_costs.tolist() == ref.path_costs.tolist()
    # lists of int64 parents are converted with a range check; K must match the options
    again = dyng.mosp.Result.from_arrays(g, 0, d, [x.astype(np.int64) for x in p])
    assert again.num_objectives == 3
    st = dyng.mosp.update(g, thesis_batch(), paths)
    assert len(st.objectives) == 3
    assert path_to(paths, 6) == [6, 5, 1, 0]
    g = thesis_graph(res)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.mosp.Result.from_arrays(g, 0, d[:2], p[:2])
    broken = [x.copy() for x in p]
    broken[1][3] = 6  # not an in-neighbour (canonicalize would repair it)
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.mosp.Result.from_arrays(g, 0, d, broken, canonicalize=False)
    fixed = dyng.mosp.Result.from_arrays(g, 0, d, broken)
    assert fixed.parents(1).tolist() == p[1].tolist()
    with pytest.raises(TypeError):
        dyng.mosp.Result()


def test_clone_copy_and_pickle(res: dyng.Resources) -> None:
    g = thesis_graph(res)
    paths = dyng.mosp.compute(g, 0, preferences=[4, 4, 1])
    for twin in (paths.clone(), copy.copy(paths), copy.deepcopy(paths)):
        assert twin._native is not paths._native
        assert twin.path_costs.tolist() == paths.path_costs.tolist()
    twin = paths.clone(dyng.Resources.sequential())
    dyng.mosp.update(g, thesis_batch(), paths)
    assert twin.graph_version == 0 and paths.graph_version == 1
    with pytest.raises(TypeError, match="from_arrays"):
        pickle.dumps(paths)


def test_dyng_update_composes_mosp_with_other_results(res: dyng.Resources) -> None:
    g = thesis_graph(res)
    paths = dyng.mosp.compute(g, 0, preferences=[4, 1, 4])
    tree = dyng.sssp.compute(g, 0, objective=2)
    st_paths, st_tree = dyng.update(g, thesis_batch(), paths, tree)
    assert isinstance(st_paths, dyng.mosp.Stats) and isinstance(st_tree, dyng.sssp.Stats)
    assert st_paths.batch == st_tree.batch and g.version == 1
    assert tree.parents.tolist() == paths.parents(2).tolist()
    assert path_to(paths, 6) == [6, 5, 1, 0]
    (only,) = dyng.update(g, dyng.EdgeBatch(delete=([5], [6])), paths)
    assert isinstance(only, dyng.mosp.Stats)
    with pytest.raises(dyng.StaleResultError):
        dyng.mosp.update(g, dyng.EdgeBatch(), dyng.mosp.compute(thesis_graph(), 0))


def test_testing_references() -> None:
    g = thesis_graph()
    paths = dyng.mosp.compute(g, 0, preferences=[4, 1, 4])
    parents = [paths.parents(k).to_numpy() for k in range(3)]
    c = dyng.testing.combined_graph(parents, 0, [4, 1, 4])
    assert c.weights is not None and c.weights.shape == (len(c.col_ind), 1)
    cg = dyng.Graph.from_csr(c.row_ptr, c.col_ind, c.weights)
    d, p = dyng.testing.dijkstra(cg, 0)
    assert d.tolist() == paths.combined_distances.tolist()
    assert p.tolist() == paths.combined_parents.tolist()
    costs = dyng.testing.mosp_path_costs(g, paths.combined_parents.to_numpy(), 0)
    assert costs.tolist() == paths.path_costs.tolist()
    two = dyng.testing.mosp_path_costs(g, p, 0, num_objectives=2)
    assert two.shape == (7, 2) and two.tolist() == costs[:, :2].tolist()


def test_write_path_costs_writes_mosp_costs_txt(tmp_path: Path) -> None:
    out = tmp_path / "c" / "mospCosts.txt"
    g = dyng.Graph.from_edges([0], [1], [[2, 3]], num_vertices=3)
    dyng.io.write_path_costs(out, dyng.mosp.compute(g, 0).path_costs)
    assert out.read_text() == "0 0 0\n1 2 3\n2 INF INF\n"
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.io.write_path_costs(out, np.zeros(3, np.int64))


def test_backends_agree_bitwise() -> None:
    rng = np.random.default_rng(11)
    n, m, k = 80, 500, 3
    src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(1, 30, (m, k)).astype(np.int32)
    ins = (rng.integers(0, n, 40), rng.integers(0, n, 40), rng.integers(1, 30, (40, k)))
    batch = dyng.EdgeBatch(insert=ins, delete=(src[:60], dst[:60]))
    out = []
    for backend in ("sequential", "openmp"):
        if not dyng.config()["backends"][backend]:
            continue
        r = dyng.Resources(backend)
        g = dyng.Graph.from_edges(src, dst, w, properties="mosp_compatible", resources=r)
        paths = dyng.mosp.compute(g, 5, preferences=[3, 1, 2])
        st = dyng.mosp.update(g, batch, paths)
        out.append(
            (
                paths.combined_distances.to_numpy(),
                paths.combined_parents.to_numpy(),
                paths.path_costs.to_numpy(),
                st.affected,
                st.combined_edges,
            )
        )
    for o in out[1:]:
        for a, b in zip(o, out[0], strict=True):
            assert np.array_equal(a, b)


# -------------------------------------------------------------------------------------------------
# Property-based: random graphs, batches and preferences against the pure-Python oracle
# -------------------------------------------------------------------------------------------------

hypothesis = pytest.importorskip("hypothesis")

from conftest import host_backends  # noqa: E402
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@st.composite
def mosp_case(draw: st.DrawFn) -> dict[str, object]:
    n = draw(st.integers(1, 10))
    k = draw(st.integers(1, 4))
    pair = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1))
    edges = draw(st.lists(pair, max_size=30, unique=True))
    weights = draw(
        st.lists(
            st.lists(st.integers(1, 9), min_size=k, max_size=k),
            min_size=len(edges),
            max_size=len(edges),
        )
    )
    ins = draw(st.lists(st.tuples(st.integers(0, n), st.integers(0, n)), max_size=8, unique=True))
    ins_w = draw(
        st.lists(
            st.lists(st.integers(1, 9), min_size=k, max_size=k),
            min_size=len(ins),
            max_size=len(ins),
        )
    )
    dele = draw(st.lists(st.sampled_from(edges) if edges else pair, max_size=8))
    pref = draw(st.sampled_from([[], [draw(st.integers(1, 6)) for _ in range(k)]]))
    source = draw(st.integers(0, n - 1))
    backend = draw(st.sampled_from(host_backends()))
    return {
        "n": n,
        "k": k,
        "edges": edges,
        "weights": weights,
        "ins": ins,
        "ins_w": ins_w,
        "dele": dele,
        "pref": pref,
        "source": source,
        "backend": backend,
    }


@given(mosp_case())
def test_compute_and_update_match_the_oracle(case: dict[str, object]) -> None:
    n, k, source = int(case["n"]), int(case["k"]), int(case["source"])  # type: ignore[call-overload]
    edges = list(case["edges"])  # type: ignore[call-overload]
    weights = list(case["weights"])  # type: ignore[call-overload]
    pref = list(case["pref"])  # type: ignore[call-overload]
    res = dyng.Resources(str(case["backend"]))
    w = np.array(weights, dtype=np.int32).reshape(-1, k)
    g = dyng.Graph.from_edges(
        [e[0] for e in edges], [e[1] for e in edges], w, num_vertices=n, resources=res
    )
    paths = dyng.mosp.compute(g, source, preferences=pref)
    columns = [{e: wt[i] for e, wt in zip(edges, weights, strict=True)} for i in range(k)]
    assert_matches_oracle(paths, n, columns, source, pref)
    # the batch under the default semantics (deletions first, then upserts, vertex growth)
    ins = list(case["ins"])  # type: ignore[call-overload]
    ins_w = list(case["ins_w"])  # type: ignore[call-overload]
    dele = list(case["dele"])  # type: ignore[call-overload]
    batch = dyng.EdgeBatch(
        insert=(
            [e[0] for e in ins],
            [e[1] for e in ins],
            np.array(ins_w, dtype=np.int32).reshape(-1, k),
        ),
        delete=([e[0] for e in dele], [e[1] for e in dele]),
    )
    st_ = dyng.mosp.update(g, batch, paths)
    for e in dele:
        for c in columns:
            c.pop(e, None)
    for e, wt in zip(ins, ins_w, strict=True):
        for i, c in enumerate(columns):
            c[e] = wt[i]
    n2 = max([n] + [max(e) + 1 for e in ins])
    assert g.num_vertices == n2
    assert_matches_oracle(paths, n2, columns, source, pref)
    assert st_.combined_edges == len(distinct_tree_edges(paths))
    fresh = dyng.mosp.compute(g, source, preferences=pref)
    assert paths.combined_parents.tolist() == fresh.combined_parents.tolist()


def distinct_tree_edges(paths: dyng.mosp.Result) -> set[tuple[int, int]]:
    """The edges of the combined graph: the distinct (parent, vertex) pairs of the K trees."""
    return {
        (p, v)
        for k in range(paths.num_objectives)
        for v, p in enumerate(paths.parents(k).tolist())
        if p != -1
    }


def test_combined_edges_count_distinct_tree_edges() -> None:
    g = thesis_graph()
    paths = dyng.mosp.compute(g, 0)
    st_ = dyng.mosp.update(g, thesis_batch(), paths)
    assert st_.combined_edges == len(distinct_tree_edges(paths)) == 9
