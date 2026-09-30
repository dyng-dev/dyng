# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.sssp: dynamic single-source shortest paths (the SOSP update of DynaMOSP).

:func:`compute` builds the canonical shortest-path tree of a graph (ties go to the lowest parent
id); :func:`update` applies a batch of edge insertions, deletions and weight changes to the graph
and brings the tree up to date incrementally. Every backend returns the same tree, bit for bit.

The names mirror the C++ API (``dyng::sssp``): :class:`Options`, :class:`Result`, :class:`Stats`,
:func:`compute`, :func:`update`. Supported graphs: int32 weights with (int32, int32), (int32,
int64) or (int64, int64) vertex ids and edge offsets, in-edges stored (the default), weights of
the objective in [1, 2^31 - 1]. Cite with ``dyng.citation("sssp")``.

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 0, 1], [1, 2, 2], [4, 1, 1])
    >>> tree = dyng.sssp.compute(g, source=0)
    >>> tree.distances.tolist()
    [0, 4, 1]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _dtypes
from ._backend import native
from ._convert import as_bool, as_int, copy_fields, enum_member, enum_name, with_options
from .array import Array
from .batch import EdgeBatch
from .errors import NotSupportedError, StaleResultError
from .graph import ApplySummary, Graph
from .resources import Resources, resolve

__all__ = ["Options", "Stats", "Result", "compute", "update", "INFINITE_DISTANCE"]

INFINITE_DISTANCE: int = int(np.iinfo(np.int64).max) // 4
"""The "unreachable" distance, max(int64) // 4 = 2**61 - 1 (MOSP's DISTANCE_INF)."""

EngineName = Literal["automatic", "fused", "operators"]


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::sssp::options``).

    Attributes:
        delta: Near-far bucket width; 0 = automatic (MOSP's defaultDelta). Changes the schedule,
            never the result.
        objective: Which weight column is the edge length; fixed at :func:`compute`.
        cuda_engine: The CUDA engine (``"automatic"``, ``"fused"``, ``"operators"``); ignored
            by the host backends.
        validate_inputs: O(n) checks on trees imported with :meth:`Result.from_arrays`.
    """

    delta: int = 0
    objective: int = 0
    cuda_engine: EngineName = "automatic"
    validate_inputs: bool = True

    def _to_native(self) -> Any:
        o = native.SsspOptions()
        o.delta = as_int(self.delta, "sssp.Options.delta")
        o.objective = as_int(self.objective, "sssp.Options.objective")
        o.cuda_engine = enum_member(native.Engine, self.cuda_engine, "sssp.Options.cuda_engine")
        o.validate_inputs = as_bool(self.validate_inputs, "sssp.Options.validate_inputs")
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        return cls(**copy_fields(o, tuple(f.name for f in dataclasses.fields(cls))))


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::sssp::stats``).

    ``invalidated``, ``affected``, ``batch`` and ``packed_parents`` are deterministic;
    ``iterations``, ``frontier_visits``, ``epochs`` and ``pushes`` depend on the schedule.

    Attributes:
        affected: Vertices whose distance or parent changed.
        iterations: Rounds of the Step 2 loop.
        frontier_visits: Frontier entries processed.
        fallback_used: A fallback path ran.
        converged: The iteration converged.
        engine_used: ``"fused"`` or ``"operators"``.
        batch: What applying the batch did to the graph.
        invalidated: Vertices of invalidated subtrees.
        epochs: Raises of the near-far threshold.
        pushes: Vertex expansions of Step 2.
        packed_parents: Distances and parents packed in 64-bit words.
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: ApplySummary
    invalidated: int
    epochs: int
    pushes: int
    packed_parents: bool

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "batch")
        return cls(**copy_fields(s, names), batch=ApplySummary._from_native(s.batch))


class Result:
    """A shortest-path tree kept up to date by :func:`update` (C++ ``dyng::sssp::result``).

    Arrays are read-only, zero-copy :class:`dyng.Array` views that belong to the current state
    of the result (see :class:`dyng.Array` for their lifetime).
    """

    __slots__ = ("_native", "_resources", "_vertex", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("use dyng.sssp.compute() or dyng.sssp.Result.from_arrays()")

    @classmethod
    def _wrap(cls, handle: Any, vertex: np.dtype, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        self._vertex = vertex
        return self

    def _array(self, getter: Any, what: str) -> Array:
        return _result_array(self._native, getter, f"sssp.Result.{what}")

    @property
    def source(self) -> int:
        """The source vertex."""
        return int(self._native.source)

    @property
    def distances(self) -> Array:
        """One distance per vertex (int64), :data:`INFINITE_DISTANCE` for unreachable vertices."""
        return self._array(self._native.distances, "distances")

    @property
    def parents(self) -> Array:
        """One parent per vertex (the graph's vertex dtype), the lowest-id in-neighbour on a
        shortest path (unless kept from a non-canonical imported tree); -1 for the source and
        unreachable vertices."""
        return self._array(self._native.parents, "parents")

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the tunables (``delta``, ``cuda_engine``, ``validate_inputs``); ``objective``
        must stay the same."""
        opt = with_options(Options, options or self.options, kwargs, "sssp.Result.set_options")
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """Where the arrays live, ``"host"`` or ``"device"``."""
        return enum_name(self._native.space)

    @property
    def vertex_dtype(self) -> np.dtype:
        """The vertex id type of the graph the result belongs to."""
        return self._vertex

    def clone(self, resources: Resources | None = None) -> Result:
        """A deep copy (arrays, options, version) for ``resources``."""
        res = resolve(resources, self._resources)
        return Result._wrap(self._native.clone(res._native), self._vertex, res)

    @classmethod
    def from_arrays(
        cls,
        graph: Graph,
        source: int,
        distances: Any,
        parents: Any,
        *,
        canonicalize: bool = True,
        options: Options | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> Result:
        """Adopt an existing tree of ``graph`` (e.g. read from MOSP's distance and tree files).

        Args:
            graph: The graph the tree belongs to.
            source: The source vertex.
            distances: One distance per vertex (int64; values >= INFINITE_DISTANCE / 2 mean
                unreachable).
            parents: One parent per vertex, -1 for none (converted to the graph's vertex dtype
                with a range check).
            canonicalize: Replace every parent by the lowest-id tight in-neighbour (MOSP's
                canonicalizeTree()), so later updates equal :func:`compute`.
            options: Options (``objective`` = the weight column the tree belongs to).
            resources: Default: the graph's resources.
            **kwargs: Option fields, overriding ``options``.

        Raises:
            InvalidArgumentError: a check of the tree fails (with ``validate_inputs``).
        """
        _check_graph(graph)
        opt = with_options(Options, options, kwargs, "sssp.Result.from_arrays")
        res = resolve(resources, graph._resources)
        d = _dtypes.checked_cast(
            _dtypes.to_numpy(distances, "from_arrays: distances"),
            _dtypes.INT64,
            "sssp.Result.from_arrays: distances",
        )
        p = _dtypes.checked_cast(
            _dtypes.to_numpy(parents, "from_arrays: parents"),
            graph.vertex_dtype,
            "sssp.Result.from_arrays: parents",
        )
        handle = native.sssp_from_arrays(
            res._native,
            graph._native,
            as_int(source, "sssp.Result.from_arrays: source"),
            d,
            p,
            as_bool(canonicalize, "sssp.Result.from_arrays: canonicalize"),
            opt._to_native(),
        )
        return cls._wrap(handle, graph.vertex_dtype, res)

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.sssp.Result cannot be pickled (it belongs to one state of one graph); send "
            "result.distances.to_numpy() and result.parents.to_numpy() and rebuild it with "
            "dyng.sssp.Result.from_arrays(graph, source, distances, parents), or use "
            "copy.deepcopy(result) / result.clone() within a process"
        )

    def __repr__(self) -> str:
        return (
            f"dyng.sssp.Result(source={self.source}, num_vertices={len(self.distances)}, "
            f"graph_version={self.graph_version})"
        )


def _result_array(handle: Any, getter: Any, what: str) -> Array:
    """An Array of a native result's current state, stale once the result is updated.

    The generation is read before the array, so an update in between makes the Array stale
    (never current with an older state's memory).
    """
    generation = handle.generation
    return Array(getter(), lambda: bool(handle.generation == generation), what)


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")
    if not graph.weighted:
        raise NotSupportedError(
            "dyng.sssp needs integer weights: graphs of types (int32, int32 or int64, int32) or "
            "(int64, int64, int32); this graph is unweighted (build it with weights=)"
        )


def compute(
    graph: Graph,
    source: int,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Compute the canonical shortest-path tree of ``graph`` from ``source``.

    Args:
        graph: The graph (with in-edges stored); it is not modified.
        source: The source vertex.
        options: Options; the keywords ``delta``, ``objective``, ``cuda_engine`` and
            ``validate_inputs`` override its fields.
        resources: Default: the resources the graph was built with.
        **kwargs: Option fields.

    Returns:
        The tree, matching ``graph.version``.

    Raises:
        InvalidArgumentError: the source or objective is out of range, a weight is below 1, the
            graph stores no in-edges, or it belongs to another backend.
        NotSupportedError: an unweighted graph, or a backend or engine that is not available.
    """
    _check_graph(graph)
    opt = with_options(Options, options, kwargs, "sssp.compute")
    res = resolve(resources, graph._resources)
    src = as_int(source, "sssp.compute: source")
    handle = native.sssp_compute(res._native, graph._native, src, opt._to_native())
    return Result._wrap(handle, graph.vertex_dtype, res)


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply ``batch`` to ``graph`` and update the tree ``result``.

    Postcondition: ``result`` equals ``compute(graph, result.source, options=result.options)``
    on the new graph (for a canonical tree; see the C++ documentation of the tie rule).

    Args:
        graph: The graph; the batch is applied to it and its version grows by one.
        batch: Insertions (upserts), deletions and weight changes.
        result: A result of :func:`compute` (or of an earlier update) on ``graph``.
        resources: Default: the resources the graph was built with.

    Returns:
        The counters of this update.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state (it was computed
            on another graph, or the graph was changed without it).
        InvalidArgumentError: an invalid id or weight in the batch (nothing is changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.sssp.update: result must be a dyng.sssp.Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.sssp.update: batch must be a dyng.EdgeBatch")
    if result.vertex_dtype != graph.vertex_dtype:
        raise StaleResultError(
            "dyng.sssp.update: the result was computed on another graph (its vertex ids are "
            f"{result.vertex_dtype.name}, the graph's {graph.vertex_dtype.name})"
        )
    res = resolve(resources, graph._resources)
    nb = batch._native_for(graph)
    return Stats._from_native(native.sssp_update(res._native, graph._native, nb, result._native))
