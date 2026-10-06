# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.triangle_delta: the triangle count of an undirected graph under edge batches (a tutorial
algorithm).

Teaching material (maturity ``"tutorial"``), not a research algorithm: the aggregate-delta
companion of the tutorial "Your first dynamic algorithm". :func:`compute` counts the triangles of
an undirected graph (``directed=False``, sorted rows, no parallel edges; self-loops and weights
are ignored). :func:`update` applies a batch and changes the count by the triangles the deleted
edges destroy (counted on the old graph) and the inserted edges create (counted on the new
graph), each triangle by its smallest changed edge; afterwards the count equals :func:`compute`
exactly.

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 0, 0, 1, 1], [1, 2, 3, 2, 3], directed=False)
    >>> tri = dyng.triangle_delta.compute(g)
    >>> tri.count
    2
    >>> stats = dyng.triangle_delta.update(g, dyng.EdgeBatch(insert=([2], [3])), tri)
    >>> tri.count, stats.triangles_added
    (4, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ._backend import native
from ._convert import copy_fields, enum_name
from .batch import EdgeBatch
from .graph import ApplySummary, Graph
from .resources import Resources, resolve

__all__ = ["Options", "Stats", "Result", "compute", "update"]


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::triangle_delta::options``;
    none yet)."""

    def _to_native(self) -> Any:
        return native.TriangleDeltaOptions()

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        del o
        return cls()


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::triangle_delta::stats``); all deterministic.

    Attributes:
        affected: 1 if the count changed, else 0.
        iterations: Always 0.
        frontier_visits: Changed edges counted (deletions + insertions).
        fallback_used: Always False.
        converged: Always True.
        engine_used: ``"operators"`` (the framework's operators, Tier A).
        batch: What applying the batch did to the graph.
        deletions: Undirected edges the batch removed.
        insertions: Undirected edges the batch added.
        triangles_removed: Triangles the batch destroyed.
        triangles_added: Triangles the batch created.
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: ApplySummary
    deletions: int
    insertions: int
    triangles_removed: int
    triangles_added: int

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(f.name for f in fields(cls) if f.name != "batch")
        return cls(**copy_fields(s, names), batch=ApplySummary._from_native(s.batch))


class Result:
    """A triangle count kept up to date by :func:`update` (C++ ``dyng::triangle_delta::result``)."""

    __slots__ = ("_native", "_resources", "__weakref__")
    _native: Any
    _resources: Resources

    def __init__(self) -> None:
        raise TypeError("use dyng.triangle_delta.compute()")

    @classmethod
    def _wrap(cls, handle: Any, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        return self

    @property
    def count(self) -> int:
        """The number of triangles."""
        return int(self._native.count)

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """``"host"`` (the count is a host value on every backend)."""
        return enum_name(self._native.space)

    def clone(self, resources: Resources | None = None) -> Result:
        """A deep copy for ``resources``."""
        res = resolve(resources, self._resources)
        return Result._wrap(self._native.clone(res._native), res)

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.triangle_delta.Result cannot be pickled (it belongs to one state of one "
            "graph); send result.count, or use copy.deepcopy(result) / result.clone()"
        )

    def __repr__(self) -> str:
        return f"dyng.triangle_delta.Result({self.count}, graph_version={self.graph_version})"


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")


def compute(
    graph: Graph,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
) -> Result:
    """Count the triangles of ``graph``.

    Args:
        graph: An undirected graph (``directed=False``) with sorted rows and no parallel edges;
            it is not modified.
        options: Options (none yet).
        resources: Default: the resources the graph was built with.

    Returns:
        The count, matching ``graph.version``.

    Raises:
        InvalidArgumentError: the graph is directed, has unsorted rows or allows parallel edges.
        NotSupportedError: an unsupported graph type or an unavailable backend.
    """
    _check_graph(graph)
    opt = options if options is not None else Options()
    if not isinstance(opt, Options):
        raise TypeError(
            "dyng.triangle_delta.compute: options must be a dyng.triangle_delta.Options"
        )
    res = resolve(resources, graph._resources)
    return Result._wrap(
        native.triangle_delta_compute(res._native, graph._native, opt._to_native()), res
    )


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply ``batch`` to ``graph`` and update the count ``result``.

    Postcondition: ``result`` equals ``compute(graph)`` exactly.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state.
        InvalidArgumentError: the batch is invalid for the graph's semantics (nothing changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.triangle_delta.update: result must be a dyng.triangle_delta.Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.triangle_delta.update: batch must be a dyng.EdgeBatch")
    res = resolve(resources, graph._resources)
    nb = batch._native_for(graph)
    return Stats._from_native(
        native.triangle_delta_update(res._native, graph._native, nb, result._native)
    )
