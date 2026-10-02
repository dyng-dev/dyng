# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.mosp: dynamic multi-objective shortest paths (the MOSP update of DynaMOSP).

A graph with K weight columns (objectives) and a source. :func:`compute` builds, for every
objective k, the canonical shortest-path tree (as :mod:`dyng.sssp` computes it), then the
*combined graph* of the K trees, whose edge (p, v) weighs ``L (K + 1) - sum L / Pref_i`` over the
trees i containing it (``Pref`` the preference vector, a lower value a higher priority, and
``L = lcm(Pref)``), its canonical shortest-path tree (the **MOSP tree**) and the **path costs**:
the K objective values of the MOSP path to every vertex. :func:`update` applies a batch once,
updates the K trees incrementally and rebuilds the combined graph, the MOSP tree and the costs.
Every backend returns the same bytes.

The names mirror the C++ API (``dyng::mosp``): :class:`Options`, :class:`Result`, :class:`Stats`,
:func:`compute`, :func:`update`. Supported graphs: those of :mod:`dyng.sssp` (int32 weights,
in-edges stored), with at least one weight column. Cite with ``dyng.citation("mosp")``.

Example:
    The worked example of thesis Chapter 4 (vertices u1..u7 are 0..6): with Pref (4, 1, 4) the
    MOSP path to u7 costs (15, 3, 20) after the batch.

    >>> import dyng
    >>> src = [0, 0, 1, 2, 2, 3, 4, 4, 4, 5]
    >>> dst = [1, 2, 3, 1, 3, 4, 1, 5, 6, 6]
    >>> w = [[2, 1, 5], [4, 1, 1], [2, 4, 2], [10, 15, 2], [5, 16, 3], [1, 1, 1], [4, 3, 2],
    ...      [1, 2, 2], [5, 6, 2], [1, 1, 1]]
    >>> g = dyng.Graph.from_edges(src, dst, w, properties="mosp_compatible")
    >>> paths = dyng.mosp.compute(g, source=0, preferences=[4, 1, 4])
    >>> batch = dyng.EdgeBatch(insert=([3, 1], [5, 5], [[10, 2, 12], [12, 1, 14]]),
    ...                        delete=([1, 4], [3, 1]))
    >>> st = dyng.mosp.update(g, batch, paths)
    >>> paths.path_costs.to_numpy()[6].tolist()
    [15, 3, 20]
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _dtypes
from . import sssp as _sssp
from ._backend import native
from ._convert import as_bool, as_int, copy_fields, enum_member, enum_name, with_options
from .array import Array
from .batch import EdgeBatch
from .errors import InvalidArgumentError, NotSupportedError, StaleResultError
from .graph import ApplySummary, Graph
from .resources import Resources, resolve
from .sssp import INFINITE_DISTANCE, EngineName

__all__ = [
    "Options",
    "Stats",
    "Result",
    "compute",
    "update",
    "INFINITE_DISTANCE",
    "MAX_OBJECTIVES",
    "MAX_PREFERENCE_SCALE",
]

MAX_OBJECTIVES: int = 64
"""The largest number of objectives (C++ ``mosp::max_objectives``; MOSP's originals allow 32)."""

MAX_PREFERENCE_SCALE: int = 1 << 20
"""The largest preference scale L = lcm(preferences) (C++ ``mosp::max_preference_scale``)."""

_INT32_MAX = 2**31 - 1


def _preferences(value: Any, what: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise InvalidArgumentError(f"{what}: expected a sequence of integers, got {value!r}")
    out = [as_int(p, what) for p in value]
    for p in out:
        if not 1 <= p <= _INT32_MAX:
            raise InvalidArgumentError(f"{what}: preferences must be in [1, 2^31 - 1], got {p}")
    return out


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::mosp::options``).

    Attributes:
        preferences: One preference per objective, each >= 1 (a lower value is a higher
            priority); empty = all ones. lcm(preferences) must not exceed 2^20. Fixed at
            :func:`compute`.
        delta: Near-far bucket width of the K sssp updates; 0 = automatic per objective. The
            combined solve always uses the automatic width of the combined graph. Changes the
            schedule, never the result.
        cuda_engine: The CUDA engine of the K updates and the combined solve (``"automatic"``,
            ``"fused"``, ``"operators"``); ignored by the host backends.
        compute_path_costs: Compute the path costs in :func:`compute` and :func:`update`; while
            false, :attr:`Result.path_costs` raises.
        validate_inputs: O(K n) checks of trees imported with :meth:`Result.from_arrays`.
        num_objectives: The objectives are the first num_objectives weight columns; 0 = every
            column (MOSP's -k). Fixed at :func:`compute`.
    """

    preferences: list[int] = field(default_factory=list)
    delta: int = 0
    cuda_engine: EngineName = "automatic"
    compute_path_costs: bool = True
    validate_inputs: bool = True
    num_objectives: int = 0

    def _to_native(self) -> Any:
        o = native.MospOptions()
        o.preferences = _preferences(self.preferences, "mosp.Options.preferences")
        o.delta = as_int(self.delta, "mosp.Options.delta")
        o.cuda_engine = enum_member(native.Engine, self.cuda_engine, "mosp.Options.cuda_engine")
        o.compute_path_costs = as_bool(self.compute_path_costs, "mosp.Options.compute_path_costs")
        o.validate_inputs = as_bool(self.validate_inputs, "mosp.Options.validate_inputs")
        o.num_objectives = as_int(self.num_objectives, "mosp.Options.num_objectives")
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "preferences")
        return cls(preferences=[int(p) for p in o.preferences], **copy_fields(o, names))


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::mosp::stats``).

    ``affected``, ``batch``, ``combined_edges``, ``preference_scale`` and the objectives'
    deterministic counters are identical on every backend; ``iterations``, ``frontier_visits``
    and the objectives' schedule counters depend on the schedule.

    Attributes:
        affected: Vertices whose combined distance or MOSP parent changed.
        iterations: Rounds of the update (summed over the stages that report them).
        frontier_visits: Frontier entries processed.
        fallback_used: A fallback path ran.
        converged: The iteration converged.
        engine_used: The engine of the update (``"fused"`` or ``"operators"``).
        batch: What applying the batch did to the graph.
        objectives: The K sssp updates, in objective order (:class:`dyng.sssp.Stats`).
        combined_edges: Edges of the combined graph.
        preference_scale: L = lcm(preferences); the combined distances are in units of 1/L.
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: ApplySummary
    objectives: tuple[_sssp.Stats, ...]
    combined_edges: int
    preference_scale: int

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(
            f.name for f in dataclasses.fields(cls) if f.name not in ("batch", "objectives")
        )
        return cls(
            **copy_fields(s, names),
            batch=ApplySummary._from_native(s.batch),
            objectives=tuple(_sssp.Stats._from_native(o) for o in s.objectives),
        )


class Result:
    """The K shortest-path trees, the MOSP tree and the path costs kept up to date by
    :func:`update` (C++ ``dyng::mosp::result``).

    Arrays are read-only, zero-copy :class:`dyng.Array` views that belong to the current state
    of the result (see :class:`dyng.Array` for their lifetime). The trees and the combined arrays
    live where the backend computes (device memory on cuda); the path costs are host memory on
    every backend.
    """

    __slots__ = ("_native", "_resources", "_vertex", "__weakref__")
    _native: Any
    _resources: Resources
    _vertex: np.dtype

    def __init__(self) -> None:
        raise TypeError("use dyng.mosp.compute() or dyng.mosp.Result.from_arrays()")

    @classmethod
    def _wrap(cls, handle: Any, vertex: np.dtype, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        self._vertex = vertex
        return self

    def _array(self, getter: Any, what: str) -> Array:
        return _sssp._result_array(self._native, getter, f"mosp.Result.{what}")

    def _objective(self, objective: int, what: str) -> int:
        k = as_int(objective, f"mosp.Result.{what}: objective")
        if not 0 <= k < self.num_objectives:
            raise InvalidArgumentError(
                f"mosp.Result.{what}: the objective {k} is out of range [0, {self.num_objectives})"
            )
        return k

    @property
    def source(self) -> int:
        """The source vertex."""
        return int(self._native.source)

    @property
    def num_objectives(self) -> int:
        """The number of objectives K."""
        return int(self._native.num_objectives)

    def distances(self, objective: int) -> Array:
        """The distances of one objective's tree (int64; :data:`INFINITE_DISTANCE` if
        unreachable)."""
        k = self._objective(objective, "distances")
        return self._array(lambda: self._native.distances(k), f"distances({k})")

    def parents(self, objective: int) -> Array:
        """The parents of one objective's tree (the graph's vertex dtype; -1 for the source and
        unreachable vertices)."""
        k = self._objective(objective, "parents")
        return self._array(lambda: self._native.parents(k), f"parents({k})")

    @property
    def combined_distances(self) -> Array:
        """The distances in the combined graph (int64, in units of 1/L, L =
        :attr:`preference_scale`; :data:`INFINITE_DISTANCE` if unreachable)."""
        return self._array(self._native.combined_distances, "combined_distances")

    @property
    def combined_parents(self) -> Array:
        """The MOSP tree, the canonical shortest-path tree of the combined graph (-1 for the
        source and unreachable vertices)."""
        return self._array(self._native.combined_parents, "combined_parents")

    @property
    def path_costs(self) -> Array:
        """The K objective values of the MOSP path to every vertex, an (n, K) int64 array in host
        memory (:data:`INFINITE_DISTANCE` for unreachable vertices).

        Raises:
            InvalidArgumentError: the options had ``compute_path_costs=False`` at the last
                :func:`compute` or :func:`update`.
        """
        return self._array(self._native.path_costs, "path_costs")

    @property
    def preference_scale(self) -> int:
        """The preference scale L = lcm(preferences); the unit of the combined distances is
        1/L."""
        return int(self._native.preference_scale)

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the tunables (``delta``, ``cuda_engine``, ``compute_path_costs``,
        ``validate_inputs``); ``preferences`` and ``num_objectives`` must stay the same."""
        opt = with_options(Options, options or self.options, kwargs, "mosp.Result.set_options")
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """Where the trees and the combined arrays live, ``"host"`` or ``"device"`` (the path
        costs are host memory on every backend)."""
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
        distances: Sequence[Any],
        parents: Sequence[Any],
        *,
        canonicalize: bool = True,
        options: Options | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> Result:
        """Adopt K existing trees of ``graph`` (e.g. MOSP's ``obj<k>/distancesOriginal.txt`` and
        ``SSSPTreeOriginal.txt``) and build the combined graph, the MOSP tree and the path costs.

        Args:
            graph: The graph the trees belong to.
            source: The source vertex.
            distances: K arrays of one distance per vertex (int64; values >= INFINITE_DISTANCE / 2
                mean unreachable).
            parents: K arrays of one parent per vertex, -1 for none (converted to the graph's
                vertex dtype with a range check).
            canonicalize: Apply the lowest-id tie rule to every tree (MOSP's canonicalizeTree()).
            options: Options; K must equal the objectives they select (``num_objectives``, or
                every weight column).
            resources: Default: the graph's resources.
            **kwargs: Option fields, overriding ``options``.

        Raises:
            InvalidArgumentError: a check fails (sssp's checks of a tree with
                ``validate_inputs``, K, the options).
        """
        _check_graph(graph)
        opt = with_options(Options, options, kwargs, "mosp.Result.from_arrays")
        res = resolve(resources, graph._resources)
        if isinstance(distances, (str, bytes)) or isinstance(parents, (str, bytes)):
            raise TypeError("mosp.Result.from_arrays: distances and parents are lists of arrays")
        ds = [
            _dtypes.checked_cast(
                _dtypes.to_numpy(d, f"from_arrays: distances[{k}]"),
                _dtypes.INT64,
                f"mosp.Result.from_arrays: distances[{k}]",
            )
            for k, d in enumerate(distances)
        ]
        ps = [
            _dtypes.checked_cast(
                _dtypes.to_numpy(p, f"from_arrays: parents[{k}]"),
                graph.vertex_dtype,
                f"mosp.Result.from_arrays: parents[{k}]",
            )
            for k, p in enumerate(parents)
        ]
        handle = native.mosp_from_arrays(
            res._native,
            graph._native,
            as_int(source, "mosp.Result.from_arrays: source"),
            ds,
            ps,
            as_bool(canonicalize, "mosp.Result.from_arrays: canonicalize"),
            opt._to_native(),
        )
        return cls._wrap(handle, graph.vertex_dtype, res)

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.mosp.Result cannot be pickled (it belongs to one state of one graph); send the "
            "trees (result.distances(k).to_numpy(), result.parents(k).to_numpy() for every k) "
            "and rebuild it with dyng.mosp.Result.from_arrays(graph, source, distances, "
            "parents), or use copy.deepcopy(result) / result.clone() within a process"
        )

    def __repr__(self) -> str:
        return (
            f"dyng.mosp.Result(source={self.source}, num_objectives={self.num_objectives}, "
            f"num_vertices={len(self.combined_parents)}, preference_scale="
            f"{self.preference_scale}, graph_version={self.graph_version})"
        )


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")
    if not graph.weighted:
        raise NotSupportedError(
            "dyng.mosp needs integer weights: graphs of types (int32, int32 or int64, int32) or "
            "(int64, int64, int32) with one weight column per objective; this graph is "
            "unweighted (build it with weights= of shape (m, K))"
        )


def compute(
    graph: Graph,
    source: int,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Compute the K canonical shortest-path trees of ``graph`` from ``source``, the combined
    graph, the MOSP tree and the path costs.

    Args:
        graph: The graph (with in-edges stored, at least one weight column); it is not modified.
        source: The source vertex.
        options: Options; the keywords ``preferences``, ``delta``, ``cuda_engine``,
            ``compute_path_costs``, ``validate_inputs`` and ``num_objectives`` override its
            fields.
        resources: Default: the resources the graph was built with.
        **kwargs: Option fields.

    Returns:
        The result, matching ``graph.version``.

    Raises:
        InvalidArgumentError: the source is out of range, the options are invalid (a preference
            below 1, not one preference per objective, lcm above 2^20, num_objectives out of
            range, more than 64 objectives, a negative delta), a weight is below 1, or the graph
            stores no in-edges.
        NotSupportedError: an unweighted graph, or a backend or engine that is not available.
    """
    _check_graph(graph)
    opt = with_options(Options, options, kwargs, "mosp.compute")
    res = resolve(resources, graph._resources)
    src = as_int(source, "mosp.compute: source")
    handle = native.mosp_compute(res._native, graph._native, src, opt._to_native())
    return Result._wrap(handle, graph.vertex_dtype, res)


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply ``batch`` to ``graph`` once and update the K trees, the MOSP tree and the path
    costs of ``result``.

    Postcondition: ``result`` equals ``compute(graph, result.source, options=result.options)``
    on the new graph (for canonical trees; see the C++ documentation of the tie rule).

    Args:
        graph: The graph; the batch is applied to it and its version grows by one.
        batch: Insertions (with the graph's number of weights each), deletions and weight
            changes.
        result: A result of :func:`compute` (or of an earlier update) on ``graph``.
        resources: Default: the resources the graph was built with.

    Returns:
        The counters of this update.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state.
        InvalidArgumentError: an invalid id or weight in the batch, or insertions with another
            number of weights than the graph's (a batch without insertions is accepted whatever
            its number of weights; nothing is changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.mosp.update: result must be a dyng.mosp.Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.mosp.update: batch must be a dyng.EdgeBatch")
    if result.vertex_dtype != graph.vertex_dtype:
        raise StaleResultError(
            "dyng.mosp.update: the result was computed on another graph (its vertex ids are "
            f"{result.vertex_dtype.name}, the graph's {graph.vertex_dtype.name})"
        )
    res = resolve(resources, graph._resources)
    nb = batch._native_for(graph)
    return Stats._from_native(native.mosp_update(res._native, graph._native, nb, result._native))
