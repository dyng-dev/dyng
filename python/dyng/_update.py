# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.update(graph, batch, r1, r2, ...): one batch, several results (C++ ``dyng::update``)."""

from __future__ import annotations

from typing import Any

from ._backend import native
from .batch import EdgeBatch
from .errors import InvalidArgumentError
from .graph import Graph
from .resources import Resources, resolve

__all__ = ["update"]


def update(
    graph: Graph, batch: EdgeBatch, *results: Any, resources: Resources | None = None
) -> tuple[Any, ...]:
    """Apply ``batch`` to ``graph`` once and bring every result up to date.

    All before-apply work of every result runs on the graph before the batch, then the batch is
    applied once, then every result is updated on the new graph. Updating one result alone
    (``dyng.sssp.update(g, b, tree)``) would leave the others stale: using them next raises
    :class:`~dyng.StaleResultError`.

    Args:
        graph: The graph; its version grows by one.
        batch: The batch.
        *results: One or more results of this graph (``dyng.sssp.Result``,
            ``dyng.cycle_count.Result``), each at most once.
        resources: Default: the resources the graph was built with.

    Returns:
        One stats object per result, in order (``dyng.sssp.Stats``, ``dyng.cycle_count.Stats``).

    Raises:
        StaleResultError: a result does not match the graph.
        InvalidArgumentError: no result, a result passed twice, or an invalid batch (nothing is
            changed).

    Example:
        >>> import dyng
        >>> g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0], [1, 1, 1])
        >>> tree, hist = dyng.sssp.compute(g, 0), dyng.cycle_count.compute(g, max_length=3)
        >>> st_tree, st_hist = dyng.update(g, dyng.EdgeBatch(delete=([2], [0])), tree, hist)
        >>> hist.total, tree.distances.tolist()
        (0, [0, 1, 2])
    """
    from . import cycle_count, sssp

    if not isinstance(graph, Graph):
        raise TypeError("dyng.update: graph must be a dyng.Graph")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.update: batch must be a dyng.EdgeBatch")
    if not results:
        raise InvalidArgumentError(
            "dyng.update: no results (applying a batch without updating any result would leave "
            "every result of the graph stale; use Graph.apply())"
        )
    kinds = []
    for i, r in enumerate(results):
        if isinstance(r, sssp.Result):
            sssp._check_graph(graph)
            kinds.append(sssp.Stats)
        elif isinstance(r, cycle_count.Result):
            cycle_count._check_graph(graph)
            kinds.append(cycle_count.Stats)
        else:
            raise TypeError(
                f"dyng.update: argument {i + 2} is not a dynG result ({type(r).__name__})"
            )
    if len({id(r) for r in results}) != len(results):
        raise InvalidArgumentError("dyng.update: a result is passed more than once")
    for r in results:
        if isinstance(r, sssp.Result) and r.vertex_dtype != graph.vertex_dtype:
            from .errors import StaleResultError

            raise StaleResultError("dyng.update: an sssp result was computed on another graph")
    res = resolve(resources, graph._resources)
    nb = batch._native_for(graph)
    out = native.update(res._native, graph._native, nb, [r._native for r in results])
    return tuple(kind._from_native(s) for kind, s in zip(kinds, out, strict=True))
