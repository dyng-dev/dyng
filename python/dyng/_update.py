# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.update(graph, batch, r1, r2, ...): one batch, several results (C++ ``dyng::update``),
over a :class:`~dyng.Graph` or (0.3) a :class:`~dyng.Hypergraph`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from ._backend import native
from ._writer import resolve_on, writing
from .batch import EdgeBatch
from .errors import InvalidArgumentError
from .graph import Graph
from .hypergraph import HyperedgeBatch, Hypergraph
from .resources import Resources

if TYPE_CHECKING:
    from . import cycle_count, mosp, sssp, triad_count

__all__ = ["update"]


@overload
def update(
    graph: Graph, batch: EdgeBatch, r1: sssp.Result, /, *, resources: Resources | None = None
) -> tuple[sssp.Stats]: ...
@overload
def update(
    graph: Graph,
    batch: EdgeBatch,
    r1: cycle_count.Result,
    /,
    *,
    resources: Resources | None = None,
) -> tuple[cycle_count.Stats]: ...
@overload
def update(
    graph: Graph,
    batch: EdgeBatch,
    r1: sssp.Result,
    r2: cycle_count.Result,
    /,
    *,
    resources: Resources | None = None,
) -> tuple[sssp.Stats, cycle_count.Stats]: ...
@overload
def update(
    graph: Graph,
    batch: EdgeBatch,
    r1: cycle_count.Result,
    r2: sssp.Result,
    /,
    *,
    resources: Resources | None = None,
) -> tuple[cycle_count.Stats, sssp.Stats]: ...
@overload
def update(
    graph: Graph, batch: EdgeBatch, *results: sssp.Result, resources: Resources | None = None
) -> tuple[sssp.Stats, ...]: ...
@overload
def update(
    graph: Graph,
    batch: EdgeBatch,
    *results: cycle_count.Result,
    resources: Resources | None = None,
) -> tuple[cycle_count.Stats, ...]: ...
@overload
def update(
    graph: Graph, batch: EdgeBatch, r1: mosp.Result, /, *, resources: Resources | None = None
) -> tuple[mosp.Stats]: ...
@overload
def update(
    graph: Graph,
    batch: EdgeBatch,
    *results: sssp.Result | cycle_count.Result | mosp.Result,
    resources: Resources | None = None,
) -> tuple[sssp.Stats | cycle_count.Stats | mosp.Stats, ...]: ...
@overload
def update(
    graph: Hypergraph,
    batch: HyperedgeBatch,
    *results: triad_count.Result,
    resources: Resources | None = None,
) -> tuple[triad_count.Stats, ...]: ...
def update(
    graph: Graph | Hypergraph,
    batch: EdgeBatch | HyperedgeBatch,
    *results: Any,
    resources: Resources | None = None,
) -> tuple[Any, ...]:
    """Apply ``batch`` to ``graph`` once and bring every result up to date.

    All before-apply work of every result runs on the graph before the batch, then the batch is
    applied once, then every result is updated on the new graph. Updating one result alone
    (``dyng.sssp.update(g, b, tree)``) would leave the others stale: using them next raises
    :class:`~dyng.StaleResultError`.

    Args:
        graph: The graph or hypergraph; its version grows by one.
        batch: The batch (an :class:`~dyng.EdgeBatch` for a graph, a
            :class:`~dyng.HyperedgeBatch` for a hypergraph).
        *results: One or more results of this graph (``dyng.sssp.Result``,
            ``dyng.cycle_count.Result``, ``dyng.mosp.Result``; of a hypergraph
            ``dyng.triad_count.Result``), each at most once.
        resources: Default: the resources the graph was built with.

    Returns:
        One stats object per result, in order (``dyng.sssp.Stats``, ``dyng.cycle_count.Stats``,
        ``dyng.mosp.Stats``, ``dyng.triad_count.Stats``).

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
        >>> hg = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]])
        >>> motifs = dyng.triad_count.compute(hg)
        >>> type1 = dyng.triad_count.compute(hg, kind="stathyper_type1")
        >>> st_motifs, st_type1 = dyng.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[3]),
        ...                                   motifs, type1)
        >>> motifs.total, type1.total, st_type1.batch.deleted_hyperedges
        (1, 3, 1)
    """
    from . import cycle_count, mosp, sssp, triad_count

    if isinstance(graph, Hypergraph):
        return _update_hypergraph(graph, batch, results, resources)
    if not isinstance(graph, Graph):
        raise TypeError("dyng.update: graph must be a dyng.Graph or a dyng.Hypergraph")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.update: batch must be a dyng.EdgeBatch")
    if not results:
        raise InvalidArgumentError(
            "dyng.update: no results (applying a batch without updating any result would leave "
            "every result of the graph stale; use Graph.apply())"
        )
    kinds: list[type[sssp.Stats] | type[cycle_count.Stats] | type[mosp.Stats]] = []
    for i, r in enumerate(results):
        if isinstance(r, sssp.Result):
            sssp._check_graph(graph)
            kinds.append(sssp.Stats)
        elif isinstance(r, cycle_count.Result):
            cycle_count._check_graph(graph)
            kinds.append(cycle_count.Stats)
        elif isinstance(r, mosp.Result):
            mosp._check_graph(graph)
            kinds.append(mosp.Stats)
        elif isinstance(r, triad_count.Result):
            # DYNG_TRIAD_COUNT_CONTAINER_MESSAGE of <dyng/triad_count.hpp>, in Python's names
            raise TypeError(
                f"dyng.update: argument {i + 2} is a dyng.triad_count.Result, a result of a "
                "hypergraph algorithm; dyng.update() takes a dyng.Hypergraph for it"
            )
        else:
            raise TypeError(
                f"dyng.update: argument {i + 2} is not a dynG result ({type(r).__name__})"
            )
    if len({id(r) for r in results}) != len(results):
        raise InvalidArgumentError("dyng.update: a result is passed more than once")
    for r in results:
        if isinstance(r, (sssp.Result, mosp.Result)) and r.vertex_dtype != graph.vertex_dtype:
            from .errors import StaleResultError

            name = "an sssp" if isinstance(r, sssp.Result) else "a mosp"
            raise StaleResultError(f"dyng.update: {name} result was computed on another graph")
    res = resolve_on(graph, resources)
    nb = batch._native_for(graph)
    with writing(results, res):
        out = native.update(res._native, graph._native, nb, [r._native for r in results])
    return tuple(kind._from_native(s) for kind, s in zip(kinds, out, strict=True))


def _update_hypergraph(
    graph: Hypergraph, batch: Any, results: tuple[Any, ...], resources: Resources | None
) -> tuple[Any, ...]:
    """dyng.update over a hypergraph: every result is a ``dyng.triad_count.Result``."""
    from . import triad_count

    if not isinstance(batch, HyperedgeBatch):
        raise TypeError("dyng.update: the batch of a hypergraph must be a dyng.HyperedgeBatch")
    if not results:
        raise InvalidArgumentError(
            "dyng.update: no results (applying a batch without updating any result would leave "
            "every result of the hypergraph stale; use Hypergraph.apply())"
        )
    for i, r in enumerate(results):
        if not isinstance(r, triad_count.Result):
            raise TypeError(
                f"dyng.update: argument {i + 2} is not a result of a hypergraph algorithm "
                f"(dyng.triad_count.Result), got {type(r).__name__}"
            )
    if len({id(r) for r in results}) != len(results):
        raise InvalidArgumentError("dyng.update: a result is passed more than once")
    res = resolve_on(graph, resources)
    nb = batch._native_for(graph.weighted)
    bins = [r.num_bins for r in results]
    with writing(results, res):
        out = native.update(res._native, graph._native, nb, [r._native for r in results])
    return tuple(triad_count.Stats._from_native(s, n) for s, n in zip(out, bins, strict=True))
