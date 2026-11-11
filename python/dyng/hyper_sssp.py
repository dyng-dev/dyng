# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.hyper_sssp: dynamic single-source shortest hyperpaths (H-SOSP, IA3 2026).

On the line graph of a weighted hypergraph (one node per live hyperedge; two hyperedges are
adjacent when they share a vertex), entering hyperedge h costs its weight w(h). From a source
vertex s, the distance D(h) of a live hyperedge is the weight of the cheapest walk
h_1, ..., h_k = h with s in h_1, so D(h) = w(h) for every hyperedge that contains s. Its parent is
-1 when it contains s and D(h) = w(h), and otherwise the lowest-id neighbour on a shortest walk
(the canonical tree). The distance of a vertex is the smallest D(h) over its live hyperedges, and
0 for the source (:meth:`Result.distance_to`, :meth:`Result.vertex_distances`).

:func:`compute` solves it; :func:`update` applies a batch of hyperedge deletions and insertions
and incidence edits (a vertex removed from or added to a live hyperedge) to the hypergraph and
brings the result up to date; afterwards it equals :func:`compute` on the new hypergraph bit for
bit, on every backend. :func:`dyng.update` updates several results (several sources, say) with
one batch.

The names mirror the C++ API (``dyng::hyper_sssp``, ``<dyng/hyper_sssp.hpp>``). Requirements: a
weighted hypergraph (int32 weights in [1, 2^28]) built with ``storage="slack_csr"`` and its line
graph (``line_graph="slack_csr"`` or ``"automatic"``) and ``store_v2h=True``. Backends: the
sequential backend and CUDA (a plugin wheel); there is no OpenMP backend in 0.4, so pass
``resources=dyng.Resources("sequential")`` where the default resources are OpenMP's (the CPU
wheel's default). Cite with ``dyng.citation("hyper_sssp")``.

Example:
    >>> import dyng
    >>> seq = dyng.Resources("sequential")
    >>> hg = dyng.Hypergraph.from_lists(
    ...     [[0, 1], [1, 2], [2, 3], [0, 3]], [2, 3, 1, 9],
    ...     storage="slack_csr", line_graph="slack_csr", resources=seq)
    >>> sp = dyng.hyper_sssp.compute(hg, source=0)
    >>> sp.hyperedge_distances.tolist(), sp.hyperedge_parents.tolist()
    ([2, 5, 6, 9], [-1, 0, 1, -1])
    >>> sp.distance_to(2), sp.vertex_distances().tolist()
    (5, [0, 2, 5, 6])
    >>> st = dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(insert_incidences=([0], [3])), sp)
    >>> sp.hyperedge_distances.tolist(), st.line_graph_insertions
    ([2, 5, 3, 9], 1)
"""

from __future__ import annotations

import dataclasses
import enum
import weakref
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _writer as _writer_state
from ._backend import native
from ._convert import as_bool, as_int, copy_fields, enum_member, enum_name, with_options
from ._writer import resolve_on, writing
from .array import Array
from .errors import InvalidArgumentError, NotSupportedError, StaleResultError
from .hypergraph import HyperApplySummary, HyperedgeBatch, Hypergraph
from .resources import Resources, resolve

__all__ = [
    "INFINITE_DISTANCE",
    "BudgetAction",
    "FallbackReason",
    "Options",
    "Stats",
    "Result",
    "compute",
    "update",
]

INFINITE_DISTANCE: int = int(np.iinfo(np.int64).max) // 4
"""The distance of a dead or unreachable hyperedge or vertex, max(int64) // 4 = 2**61 - 1 (C++
``infinite_distance<std::int64_t>()``, MOSP_ESCHER's INF)."""

EngineName = Literal["automatic", "fused", "operators"]


class BudgetAction(enum.StrEnum):
    """What :func:`update` does when its work or iteration budget is spent (C++
    ``dyng::hyper_sssp::budget_action``). A member is also its name as a string."""

    recompute = "recompute"  #: recompute from scratch (``Stats.fallback_used``); exact
    error = "error"  #: raise :class:`~dyng.ConvergenceError` after the commit


class FallbackReason(enum.StrEnum):
    """Why an update fell back to a recompute (C++ ``dyng::hyper_sssp::fallback_reason``)."""

    none = "none"  #: no fallback
    work_budget = "work_budget"  #: the relaxations exceeded ``work_budget`` x the live entries
    iterations = "iterations"  #: the push rounds reached ``max_iterations`` (cuda)
    distance_width = "distance_width"  #: a distance did not fit the engine's packed word (cuda)
    invalidation_depth = "invalidation_depth"  #: the invalidation's pointer jumping (cuda)


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::hyper_sssp::options``).

    Every field is a tunable: it changes how an update runs, never its result
    (:meth:`Result.set_options`).

    Attributes:
        work_budget: The work budget of an update, a multiple of the live line-graph entries
            (MOSP_ESCHER's ``workBudget``): an update whose relaxations exceed
            ``work_budget * max(1, live entries)`` before a push round (cuda) or an expansion
            (sequential) takes the ``on_budget`` action. The work before the first check (the
            invalidation, the seed's pulls and offers) is never refused: an update whose work
            all lies in its seed completes even at 0, and 0 acts at the first push round or
            expansion; ``float("inf")`` never acts. Tune it from :attr:`Stats.relaxations`
            divided by the live line-graph entries (``len(hypergraph.h2h().col_ind)``).
        max_iterations: Push rounds an update may run before the ``on_budget`` action
            (MOSP_ESCHER's ``maxIterations``; cuda only). At least 1.
        on_budget: ``"recompute"`` (fall back to a recompute: exact) or ``"error"`` (raise
            :class:`~dyng.ConvergenceError`; the hypergraph keeps the batch and the result is
            left unusable).
        cuda_engine: ``"automatic"`` and ``"fused"`` run MOSP_ESCHER's kernels as they are;
            ``"operators"`` runs the shared sssp engine (the same results). Ignored by the
            sequential backend.
        validate_inputs: Check the weights against [1, 2^28] (:func:`compute`: every live
            hyperedge; :func:`update`: the batch's insertions, before the commit).
    """

    work_budget: float = 1.0
    max_iterations: int = 4096
    on_budget: BudgetAction | Literal["recompute", "error"] = "recompute"
    cuda_engine: EngineName = "automatic"
    validate_inputs: bool = True

    def _to_native(self) -> Any:
        o = native.HyperSsspOptions()
        if isinstance(self.work_budget, bool) or not isinstance(
            self.work_budget, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                "hyper_sssp.Options.work_budget: expected a number, got "
                f"{type(self.work_budget).__name__}"
            )
        o.work_budget = float(self.work_budget)
        o.max_iterations = as_int(self.max_iterations, "hyper_sssp.Options.max_iterations")
        o.on_budget = enum_member(
            native.HyperSsspBudgetAction, str(self.on_budget), "hyper_sssp.Options.on_budget"
        )
        o.cuda_engine = enum_member(
            native.Engine, self.cuda_engine, "hyper_sssp.Options.cuda_engine"
        )
        o.validate_inputs = as_bool(self.validate_inputs, "hyper_sssp.Options.validate_inputs")
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        f = copy_fields(o, tuple(f.name for f in dataclasses.fields(cls)))
        f["on_budget"] = BudgetAction(f["on_budget"])
        return cls(**f)


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::hyper_sssp::stats``).

    ``batch``, ``line_graph_insertions``, ``line_graph_deletions``, ``invalidated`` and
    ``affected`` are deterministic; ``iterations``, ``frontier_visits`` and ``relaxations``
    depend on the schedule on cuda. ``fallback_used`` and ``fallback_reason`` are deterministic on
    the sequential backend and schedule-dependent on cuda near the budget (a batch within a few
    percent of its budget may fall back in one run and not in another, with identical results).
    ``relaxations`` and ``iterations`` also depend on the CUDA engine: ``"fused"`` (MOSP_ESCHER's
    relax) pushes a hyperedge again when its parent id drops at an equal distance,
    ``"operators"`` only when its distance drops, so on inputs with equal-distance ties the
    shared engine counts fewer, and the two engines' decisions near the budget can differ.

    Attributes:
        affected: Hyperedges whose subtree was invalidated (= ``invalidated``).
        iterations: Push rounds (cuda) or expansions (sequential) of the update.
        frontier_visits: The largest frontier (cuda) or the expansions (sequential).
        fallback_used: The update fell back to a recompute.
        converged: Always True.
        engine_used: The CUDA engine that ran (``"fused"`` or ``"operators"``);
            ``"operators"`` on the sequential backend.
        batch: What applying the batch did to the hypergraph (``batch.inserted_ids``: the id of
            every inserted hyperedge; ``batch.inserted_incidences`` / ``deleted_incidences``:
            the applied incidence edits).
        line_graph_insertions: Line-graph pairs the batch added.
        line_graph_deletions: Line-graph pairs the batch removed.
        invalidated: Hyperedges in invalidated subtrees.
        relaxations: The relaxations the work budget counted (the neighbours read by the pulls
            and pushes): the quantity to tune ``Options.work_budget`` from, divided by the
            line graph's live entries.
        fallback_reason: Why the update fell back (``"none"`` if it did not).
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: HyperApplySummary
    line_graph_insertions: int
    line_graph_deletions: int
    invalidated: int
    relaxations: int
    fallback_reason: FallbackReason

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(
            f.name for f in dataclasses.fields(cls) if f.name not in ("batch", "fallback_reason")
        )
        return cls(
            **copy_fields(s, names),
            batch=HyperApplySummary._from_native(s.batch),
            fallback_reason=FallbackReason(enum_name(s.fallback_reason)),
        )


class Result:
    """Shortest hyperpaths from one source vertex, kept up to date by :func:`update` (C++
    ``dyng::hyper_sssp::result``).

    The arrays are read-only, zero-copy :class:`dyng.Array` views of the current state (device
    memory for a result of the cuda backend; see :class:`dyng.Array` for their lifetime).
    """

    __slots__ = ("_native", "_resources", "_hypergraph", "__weakref__")
    _native: Any
    _resources: Resources
    _hypergraph: weakref.ReferenceType[Hypergraph]
    # the bookkeeping of dyng._writer (a side table)
    _writer = _writer_state.WRITER
    _streams = _writer_state.STREAMS
    _write_lock = _writer_state.WRITE_LOCK
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError("use dyng.hyper_sssp.compute()")

    @classmethod
    def _wrap(cls, handle: Any, hypergraph: Hypergraph, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        self._hypergraph = weakref.ref(hypergraph)
        _writer_state.track(self, resources)
        return self

    def _array(self, getter: Any, what: str) -> Array:
        from .sssp import _result_array

        return _result_array(
            self._native,
            getter,
            f"hyper_sssp.Result.{what}",
            (self._resources, self._writer, self._streams),
        )

    def _link(self, hypergraph: Hypergraph) -> None:
        """The result now matches ``hypergraph`` (after a successful update; the C++ result's
        weak reference behind :meth:`distance_to` points to it too)."""
        self._hypergraph = weakref.ref(hypergraph)

    def _graph(self) -> Any:
        """The native hypergraph the result matches (None once it is gone)."""
        h = self._hypergraph()
        return None if h is None else h._native

    @property
    def source(self) -> int:
        """The source vertex."""
        return int(self._native.source)

    @property
    def hyperedge_distances(self) -> Array:
        """D(h) by hyperedge id (int64, ``hyperedge_id_bound`` values);
        :data:`INFINITE_DISTANCE` for dead or unreachable ids."""
        return self._array(self._native.hyperedge_distances, "hyperedge_distances")

    @property
    def hyperedge_parents(self) -> Array:
        """The canonical parent of every hyperedge id, int32, and -1 for a hyperedge reached
        directly from the source vertex and for dead or unreachable ids."""
        return self._array(self._native.hyperedge_parents, "hyperedge_parents")

    def distance_to(self, target: int) -> int:
        """The distance of a vertex: the smallest D(h) over its live hyperedges, 0 for the
        source, :data:`INFINITE_DISTANCE` if none of them is reachable.

        Raises:
            InvalidArgumentError: ``target`` is not a vertex of the hypergraph.
            StaleResultError: the hypergraph the result matches is gone or has changed since
                (update the result with the batch).
        """
        v = as_int(target, "hyper_sssp.Result.distance_to: target")
        if v < 0:
            raise InvalidArgumentError(
                f"hyper_sssp.Result.distance_to: the vertex {v} is outside [0, num_vertices)"
            )
        return int(self._native.distance_to(self._graph(), v))

    def vertex_distances(self) -> Array:
        """The distance of every vertex (:meth:`distance_to` of each; int64, host memory),
        computed on request once per hypergraph version.

        Raises:
            StaleResultError: as :meth:`distance_to`.
        """
        handle = self._native
        generation = handle.generation
        return Array(
            handle.vertex_distances(self._graph()),
            lambda: bool(handle.generation == generation),
            "hyper_sssp.Result.vertex_distances",
            None,
            (self._resources, self._writer, self._streams),
        )

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the tunables (every field of :class:`Options`)."""
        opt = with_options(
            Options, options or self.options, kwargs, "hyper_sssp.Result.set_options"
        )
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The hypergraph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """Where the hyperedge arrays live, ``"host"`` or ``"device"``."""
        return enum_name(self._native.space)

    def clone(self, resources: Resources | None = None) -> Result:
        """A deep copy (arrays, options, version, the engine's state) for ``resources``
        (sequential or cuda): how a result moves between backends, with its hypergraph
        (:meth:`dyng.Hypergraph.to_backend`).

        The copy matches the moved hypergraph at once (:func:`update` accepts it), but
        :meth:`distance_to` and :meth:`vertex_distances` keep reading the hypergraph this result
        matches until the copy's first :func:`update`, which links it to the hypergraph it is
        given. Once the original hypergraph is gone or changed they raise
        :class:`~dyng.StaleResultError`, so update the copy on the moved hypergraph first (an
        empty :class:`~dyng.HyperedgeBatch` is enough) when the original may go."""
        res = resolve(resources, self._resources)
        out = object.__new__(Result)
        out._native = self._native.clone(res._native)
        out._resources = res
        out._hypergraph = self._hypergraph
        _writer_state.track(out, res)
        return out

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.hyper_sssp.Result cannot be pickled (it belongs to one state of one "
            "hypergraph); send result.hyperedge_distances.to_numpy() and "
            "result.hyperedge_parents.to_numpy(), or use copy.deepcopy(result) / result.clone() "
            "within a process"
        )

    def __repr__(self) -> str:
        # Never raises: a result a failed update left unusable shows as poisoned.
        try:
            ids = f"num_hyperedge_ids={len(self.hyperedge_distances)}"
        except StaleResultError:
            ids = "poisoned"
        return (
            f"dyng.hyper_sssp.Result(source={self.source}, {ids}, "
            f"graph_version={self.graph_version})"
        )


def _check_hypergraph(hypergraph: Hypergraph) -> None:
    if not isinstance(hypergraph, Hypergraph):
        raise TypeError("dyng.hyper_sssp: expected a dyng.Hypergraph")
    if not hypergraph.weighted:
        raise NotSupportedError(
            "dyng.hyper_sssp needs a weighted hypergraph (int32 weights in [1, 2^28], the "
            "hyperedge lengths); this hypergraph is unweighted (build it with weights)"
        )


def compute(
    hypergraph: Hypergraph,
    source: int,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Compute the shortest hyperpaths of ``hypergraph`` from the source vertex ``source``.

    Args:
        hypergraph: A weighted hypergraph with ``storage="slack_csr"``, its line graph and
            ``store_v2h=True``; it is not modified (``line_graph="automatic"`` builds the line
            graph here, once).
        source: The source vertex.
        options: Options; the keywords (``work_budget``, ``max_iterations``, ``on_budget``,
            ``cuda_engine``, ``validate_inputs``) override its fields.
        resources: Default: the resources the hypergraph was built with (sequential or cuda;
            ``dyng.Resources("sequential")`` for a hypergraph of the OpenMP backend).
        **kwargs: Option fields.

    Returns:
        The result, matching ``hypergraph.version``.

    Raises:
        InvalidArgumentError: the source is out of range, an option is invalid, a live weight is
            outside [1, 2^28] (``validate_inputs``), the hypergraph keeps no vertex-to-hyperedge
            rows, or it belongs to another backend.
        NotSupportedError: an unweighted hypergraph; storage other than ``"slack_csr"`` or
            ``line_graph="none"``; OpenMP resources (hyper_sssp runs on ``"sequential"`` and
            ``"cuda"`` in 0.4: use ``dyng.Resources("sequential")``); a backend that is not
            available.
        OutOfMemoryError: host or device memory cannot be allocated (also a ``MemoryError``).
    """
    _check_hypergraph(hypergraph)
    opt = with_options(Options, options, kwargs, "hyper_sssp.compute")
    res = resolve_on(hypergraph, resources)
    src = as_int(source, "hyper_sssp.compute: source")
    handle = native.hyper_sssp_compute(res._native, hypergraph._native, src, opt._to_native())
    return Result._wrap(handle, hypergraph, res)


def update(
    hypergraph: Hypergraph,
    batch: HyperedgeBatch,
    result: Result,
    *,
    resources: Resources | None = None,
) -> Stats:
    """Apply ``batch`` to ``hypergraph`` and update ``result``.

    Postcondition: ``result`` equals ``compute(hypergraph, result.source,
    options=result.options)`` exactly, distances and parents.

    Args:
        hypergraph: The hypergraph; the batch is applied to it and its version grows by one.
        batch: Hyperedge deletions and insertions (with ``insert_weights``) and incidence edits
            (``delete_incidences`` / ``insert_incidences``).
        result: A result of :func:`compute` (or of an earlier update) on ``hypergraph``.
        resources: Default: the resources the hypergraph was built with.

    Returns:
        The counters of this update.

    Raises:
        StaleResultError: ``result`` does not match the hypergraph's current state (it was
            computed on another hypergraph, or the hypergraph was changed without it), or a
            failed update left it unusable.
        InvalidArgumentError: the batch is invalid (an inserted weight outside [1, 2^28] with
            ``validate_inputs``, a member out of range); nothing is changed.
        NotSupportedError: as :func:`compute` (nothing is changed).
        ConvergenceError: under ``on_budget="error"``, the update exceeded its work or iteration
            budget (after the commit: the hypergraph holds the new version and ``result`` is left
            unusable; recompute it).
        CapacityError: the hypergraph's storage cannot hold the batch.
        OutOfMemoryError: host or device memory cannot be allocated (also a ``MemoryError``):
            before the commit nothing is changed; inside it the hypergraph is unusable; after it
            ``result`` is left unusable.
    """
    _check_hypergraph(hypergraph)
    if not isinstance(result, Result):
        raise TypeError("dyng.hyper_sssp.update: result must be a dyng.hyper_sssp.Result")
    if not isinstance(batch, HyperedgeBatch):
        raise TypeError("dyng.hyper_sssp.update: batch must be a dyng.HyperedgeBatch")
    res = resolve_on(hypergraph, resources)
    nb = batch._native_for(True)
    with writing([result], res):
        out = native.hyper_sssp_update(res._native, hypergraph._native, nb, result._native)
    result._link(hypergraph)
    return Stats._from_native(out)
