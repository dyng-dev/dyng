# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.dynamic_bfs: BFS levels kept up to date under edge batches (a tutorial algorithm).

Teaching material (maturity ``"tutorial"``), not a research algorithm: the subject of the
tutorial "Your first dynamic algorithm". :func:`compute` gives the BFS level (hop distance) of
every vertex from ``source`` in a directed graph, -1 for the vertices the source does not reach
(weights are ignored). :func:`update` applies a batch and repairs the levels: the subtrees below
the deleted BFS-tree edges are invalidated and re-seeded, the inserted edges offer shorter
levels, and a frontier propagates the improvements; afterwards the levels equal :func:`compute`
on the new graph exactly.

The names mirror the C++ API (``dyng::dynamic_bfs``). Supported graphs: those of
:mod:`dyng.sssp` and the unweighted graphs with int32 ids; the in-edges must be stored (the
default).

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 0, 1, 2, 3], [1, 2, 3, 3, 4])
    >>> bfs = dyng.dynamic_bfs.compute(g, 0)
    >>> bfs.levels.to_numpy().tolist()
    [0, 1, 1, 2, 3]
    >>> stats = dyng.dynamic_bfs.update(g, dyng.EdgeBatch(insert=([0], [4])), bfs)
    >>> bfs.levels.to_numpy().tolist(), stats.affected
    ([0, 1, 1, 2, 1], 1)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from . import _writer as _writer_state
from ._backend import native
from ._convert import as_int, copy_fields, enum_name, with_options
from ._writer import resolve_on, writing
from .array import Array
from .batch import EdgeBatch
from .graph import ApplySummary, Graph
from .resources import Resources, resolve

__all__ = ["Options", "Stats", "Result", "compute", "update"]


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::dynamic_bfs::options``).

    Attributes:
        source: The source vertex; fixed at :func:`compute`.
    """

    source: int = 0

    def _to_native(self) -> Any:
        o = native.DynamicBfsOptions()
        o.source = as_int(self.source, "dynamic_bfs.Options.source")
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        return cls(**copy_fields(o, tuple(f.name for f in dataclasses.fields(cls))))


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::dynamic_bfs::stats``).

    Attributes:
        affected: Vertices whose level changed (deterministic).
        iterations: Rounds of the frontier loop.
        frontier_visits: Frontier vertices expanded.
        fallback_used: Always False.
        converged: Always True.
        engine_used: ``"operators"`` (the framework's operators, Tier A).
        batch: What applying the batch did to the graph.
        invalidated: Vertices invalidated below the deleted tree edges (deterministic).
        invalidation_rounds: Passes of the invalidation (deterministic).
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: ApplySummary
    invalidated: int
    invalidation_rounds: int

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "batch")
        return cls(**copy_fields(s, names), batch=ApplySummary._from_native(s.batch))


class Result:
    """BFS levels kept up to date by :func:`update` (C++ ``dyng::dynamic_bfs::result``)."""

    __slots__ = ("_native", "_resources", "__weakref__")
    _native: Any
    _resources: Resources
    # the bookkeeping of dyng._writer (a side table, as for the other results)
    _writer = _writer_state.WRITER
    _streams = _writer_state.STREAMS
    _write_lock = _writer_state.WRITE_LOCK
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError("use dyng.dynamic_bfs.compute()")

    @classmethod
    def _wrap(cls, handle: Any, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        _writer_state.track(self, resources)
        return self

    @property
    def levels(self) -> Array:
        """One level per vertex (int64), -1 for the unreached; a zero-copy view of the current
        state (device memory for a result of the cuda backend)."""
        from .sssp import _result_array

        return _result_array(
            self._native,
            self._native.levels,
            "dynamic_bfs.Result.levels",
            (self._resources, self._writer, self._streams),
        )

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the options; ``source`` must stay the same."""
        opt = with_options(
            Options, options or self.options, kwargs, "dynamic_bfs.Result.set_options"
        )
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """``"host"`` or ``"device"`` (where the levels live)."""
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
            "dyng.dynamic_bfs.Result cannot be pickled (it belongs to one state of one graph); "
            "send result.levels.to_numpy(), or use copy.deepcopy(result) / result.clone()"
        )

    def __repr__(self) -> str:
        return f"dyng.dynamic_bfs.Result(graph_version={self.graph_version})"


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")


def compute(
    graph: Graph,
    source: int | None = None,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """The BFS levels of ``graph`` from ``source``.

    Args:
        graph: The graph (its in-edges stored); it is not modified.
        source: The source vertex (default: ``options.source``, 0 without options).
        options: Options; the keyword ``source`` overrides its field.
        resources: Default: the resources the graph was built with.
        **kwargs: Option fields.

    Returns:
        The levels, matching ``graph.version``.

    Raises:
        InvalidArgumentError: the source is not a vertex, or the graph does not store its
            in-edges.
        NotSupportedError: an unsupported graph type or an unavailable backend.
    """
    _check_graph(graph)
    if source is not None:
        kwargs["source"] = source
    opt = with_options(Options, options, kwargs, "dynamic_bfs.compute")
    res = resolve_on(graph, resources)
    return Result._wrap(
        native.dynamic_bfs_compute(res._native, graph._native, opt._to_native()), res
    )


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply ``batch`` to ``graph`` and update the levels ``result``.

    Postcondition: ``result`` equals ``compute(graph, options=result.options)`` exactly.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state.
        InvalidArgumentError: the batch is invalid for the graph's semantics (nothing changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.dynamic_bfs.update: result must be a dyng.dynamic_bfs.Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.dynamic_bfs.update: batch must be a dyng.EdgeBatch")
    res = resolve_on(graph, resources)
    nb = batch._native_for(graph)
    with writing([result], res):
        out = native.dynamic_bfs_update(res._native, graph._native, nb, result._native)
    return Stats._from_native(out)
