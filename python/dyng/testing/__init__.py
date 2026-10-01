# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.testing: oracles to check results (C++ ``dyng::testing``) and Hypothesis strategies.

The native oracles (:func:`dijkstra`, :func:`check_sssp_tree`, :func:`simple_cycles`,
:func:`combined_graph`, :func:`mosp_path_costs`) share no code with the algorithms they check.
:mod:`dyng.testing.oracles` holds small pure-Python oracles (Dijkstra with lowest-id ties, a
brute-force cycle count and a model of batch semantics), and
:mod:`dyng.testing.strategies` Hypothesis strategies for random graphs and batches (it needs the
``hypothesis`` package).

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 1], [1, 2], [2, 3])
    >>> bool(dyng.testing.check_sssp_tree(g, dyng.sssp.compute(g, 0)))
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _dtypes
from .._backend import native
from .._convert import as_bool, as_int
from ..graph import CSR, Graph
from ..sssp import Result as SsspResult
from ..sssp import _check_graph

__all__ = [
    "SsspTreeCheck",
    "dijkstra",
    "check_sssp_tree",
    "simple_cycles",
    "combined_graph",
    "mosp_path_costs",
]


@dataclass(frozen=True)
class SsspTreeCheck:
    """The outcome of :func:`check_sssp_tree`; true when the tree is correct."""

    distance_mismatches: int  #: distances that differ from Dijkstra's
    inconsistent_parents: int  #: parents p without an edge (p, v) with d[p] + w == d[v]
    non_canonical_parents: int  #: consistent parents that are not the lowest id
    parent_mismatches: int  #: parents that differ from Dijkstra's (canonical mode)
    require_canonical: bool  #: the mode of the check
    ok: bool  #: whether the tree passed
    summary: str  #: a one-line description

    def __bool__(self) -> bool:
        return self.ok


def dijkstra(graph: Graph, source: int, *, objective: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra's shortest-path tree with lowest-id ties (the library's oracle).

    Returns:
        ``(distances, parents)``: int64 distances (:data:`dyng.sssp.INFINITE_DISTANCE` if
        unreachable) and parents (-1 for the source and unreachable vertices).
    """
    _check_graph(graph)
    d, p = native.testing_dijkstra(
        graph._native,
        as_int(source, "testing.dijkstra: source"),
        as_int(objective, "testing.dijkstra: objective"),
    )
    return d, p


def check_sssp_tree(
    graph: Graph, tree: SsspResult, *, require_canonical: bool = True
) -> SsspTreeCheck:
    """Check a tree of :mod:`dyng.sssp` against Dijkstra on ``graph``'s current state."""
    _check_graph(graph)
    c = native.testing_check_sssp_tree(
        graph._native,
        tree._native,
        as_bool(require_canonical, "check_sssp_tree: require_canonical"),
    )
    return SsspTreeCheck(
        int(c.distance_mismatches),
        int(c.inconsistent_parents),
        int(c.non_canonical_parents),
        int(c.parent_mismatches),
        bool(c.require_canonical),
        bool(c.ok()),
        str(c.summary()),
    )


def simple_cycles(graph: Graph, max_length: int = -1, *, brute_force: bool = False) -> np.ndarray:
    """The directed simple-cycle histogram of ``graph``'s current state by the library's oracle
    (a subset DP; ``brute_force=True``: plain enumeration). counts[len] is the number of cycles of
    length ``len``; unlike :class:`dyng.cycle_count.Result` the array is not trimmed to the
    vertex count."""
    if not isinstance(graph, Graph):
        raise TypeError("simple_cycles: graph must be a dyng.Graph")
    out: np.ndarray = native.testing_simple_cycles(
        graph._native,
        as_int(max_length, "testing: max_length"),
        as_bool(brute_force, "testing: brute_force"),
    )
    return out


def combined_graph(
    parents: Sequence[Any], source: int, preferences: Sequence[int] | None = None
) -> CSR:
    """The combined graph of K shortest-path trees (C++ ``testing::combined_graph_reference``, a
    port of MOSP's combinedGraphReference()): the union of the K trees' edges, the edge (p, v)
    weighted ``L (K + 1) - sum L / Pref_i`` over the trees i containing it (``L = lcm(Pref)``).

    Args:
        parents: K parent arrays of n entries each (-1: none), all of one integer dtype.
        source: The source (it has no in-edge).
        preferences: K preferences >= 1, or None (all ones).

    Returns:
        The combined graph as a :class:`dyng.CSR` (int64 offsets, the parents' vertex dtype,
        weights of shape (num_edges, 1), int32).
    """
    arrays = [_dtypes.to_numpy(p, f"combined_graph: parents[{k}]") for k, p in enumerate(parents)]
    vertex = _dtypes.infer_id_dtype(arrays, None, "combined_graph") if arrays else _dtypes.INT32
    arrays = [_dtypes.checked_cast(a, vertex, "combined_graph: parents") for a in arrays]
    code = "i64" if vertex == np.int64 else "i32"
    prefs = [as_int(p, "combined_graph: preferences") for p in (preferences or [])]
    row_ptr, col_ind, weights = getattr(native, f"testing_combined_graph_{code}")(
        arrays, as_int(source, "combined_graph: source"), prefs
    )
    return CSR(row_ptr, col_ind, weights.reshape(-1, 1))


def mosp_path_costs(
    graph: Graph, parents: Any, source: int, *, num_objectives: int = 0
) -> np.ndarray:
    """The objective values of the path from ``source`` to every vertex along a tree (C++
    ``testing::mosp_path_costs_reference``, MOSP's mospPathCosts() written independently): the
    weights of the tree edge (p, v) are those of the first edge from p to v in p's row.

    Args:
        graph: The graph (its current state).
        parents: The tree: one parent per vertex, -1 for the source and unreachable vertices.
        source: The source.
        num_objectives: The weight columns to sum (0: every column of the graph).

    Returns:
        An (n, K) int64 array; :data:`dyng.sssp.INFINITE_DISTANCE` for vertices the tree does
        not connect to the source.
    """
    _check_graph(graph)
    p = _dtypes.checked_cast(
        _dtypes.to_numpy(parents, "mosp_path_costs: parents"),
        graph.vertex_dtype,
        "mosp_path_costs: parents",
    )
    out: np.ndarray = native.testing_mosp_path_costs(
        graph._native,
        p,
        as_int(source, "mosp_path_costs: source"),
        as_int(num_objectives, "mosp_path_costs: num_objectives"),
    )
    return out
