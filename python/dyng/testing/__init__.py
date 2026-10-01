# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.testing: oracles to check results (C++ ``dyng::testing``) and Hypothesis strategies.

The native oracles (:func:`dijkstra`, :func:`check_sssp_tree`, :func:`simple_cycles`) share no
code with the algorithms they check. :mod:`dyng.testing.oracles` holds small pure-Python oracles
(Dijkstra with lowest-id ties, a brute-force cycle count and a model of batch semantics), and
:mod:`dyng.testing.strategies` Hypothesis strategies for random graphs and batches (it needs the
``hypothesis`` package).

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 1], [1, 2], [2, 3])
    >>> bool(dyng.testing.check_sssp_tree(g, dyng.sssp.compute(g, 0)))
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._backend import native
from .._convert import as_bool, as_int
from ..graph import Graph
from ..sssp import Result as SsspResult
from ..sssp import _check_graph

__all__ = ["SsspTreeCheck", "dijkstra", "check_sssp_tree", "simple_cycles"]


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
