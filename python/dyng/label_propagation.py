# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.label_propagation: dynamic binary label propagation under vertex batches (DynLP).

:func:`compute` solves the labels of a similarity graph from scratch (IrLP, the published
baseline); :func:`update` applies one vertex batch (arriving and departing vertices with their
edges, the ground truth of the arrivals) to the graph and brings the labels up to date
incrementally (DynLP: the batch's subgraph is sparsified, its components seeded, and the labels
propagated from the affected vertices only). A label is a float32 in [0, 1]: the probability of
class 1; a seed's label is exactly its class.

The names mirror the C++ API (``dyng::label_propagation``): :class:`Options`, :class:`Stats`,
:class:`Result`, :func:`compute`, :func:`update`, the schedule names ``"in_place"`` (the
published rule) and ``"damped"`` (1/2-damped Jacobi, bitwise per backend) and the limit actions
``"error"``, ``"report"`` and ``"fallback_recompute"``. Supported graphs: undirected float32
graphs with int32 vertex ids (int32 or int64 edge offsets); :func:`update` needs slot rows
(:meth:`dyng.Graph.with_capacity`, ``semantics="net_effect"``). Backends: sequential and cuda
(there is no OpenMP backend in 0.4: OpenMP resources raise :class:`~dyng.NotSupportedError`).
Cite with ``dyng.citation("label_propagation")``.

**The published setup** (the documented default; ADR 0071): build the graph with
``Graph.with_capacity(stream.row_capacity)`` from a vertex-arrival stream
(:func:`dyng.generators.vertex_arrival_stream`), call :func:`compute` on the empty graph, then
:func:`update` once per batch with ``similarity_threshold=stream.similarity_threshold``. The
**streaming mode** (``layout="slack"`` rows and ``similarity_threshold=None``: the average
weight of the current graph) is an advanced mode, checked by the oracles but not gated.

**Zero-copy labels and copy-on-write.** :attr:`Result.labels` is a read-only float32
:class:`dyng.Array` that views the result's memory (host or device): ``torch.from_dlpack``,
``numpy.from_dlpack`` and CuPy read it without a copy, ordered on the result's stream (ADR
0031). An export (a tensor, a NumPy array, the Array itself) keeps showing the labels it was
read from: the next :func:`update` then copies the result's per-vertex state once (the labels,
the seed labels and the propagation tags: 13 bytes per vertex on the host, 17 on cuda) and
writes the copy, so the export never changes under its reader. Drop the
exports before the next update (``del t``) to update in place without the copy. A stale
:class:`dyng.Array` (read before an update) raises :class:`~dyng.StaleResultError`; the tensor
made from it stays valid with the old labels.

Example:
    >>> import dyng
    >>> res = dyng.Resources("sequential")
    >>> s = dyng.generators.vertex_arrival_stream(total_vertices=400, num_batches=3, seed=7,
    ...                                           resources=res)
    >>> g = dyng.Graph.with_capacity(s.row_capacity, resources=res)
    >>> r = dyng.label_propagation.compute(g, update_schedule="damped",
    ...                                    similarity_threshold=s.similarity_threshold)
    >>> stats = [dyng.label_propagation.update(g, b, r) for b in s.batches]
    >>> r.labels.dtype, len(r.labels), g.num_active_vertices > 0
    (dtype('float32'), 400, True)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _dtypes
from . import _writer as _writer_state
from ._backend import native
from ._convert import as_int, copy_fields, enum_member, enum_name, with_options
from ._writer import resolve_on, writing
from .array import Array
from .batch import EdgeBatch
from .errors import NotSupportedError
from .graph import ApplySummary, Graph
from .resources import Resources, resolve
from .sssp import _result_array

__all__ = ["Options", "Stats", "Result", "compute", "update"]

ScheduleName = Literal["in_place", "damped"]
OnLimitName = Literal["error", "report", "fallback_recompute"]
EngineName = Literal["automatic", "fused", "operators"]


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::label_propagation::options``).

    Every field is a tunable (:meth:`Result.set_options`). ``similarity_threshold`` and
    ``tolerance`` are float32 in C++; :attr:`Result.options` reads them back as the shortest
    decimal of that float32.

    Attributes:
        similarity_threshold: tau, the weight above which an edge of a batch's subgraph is kept
            (sparsification). A number: the published setup passes the stream's
            ``similarity_threshold`` (the average weight of the full graph). None: the average
            weight of the current graph's live entries (the streaming mode; 0.5 without edges).
        tolerance: delta, a label changes "by more than delta" (> 0).
        max_iterations: The cap of the propagation per update and of compute's iterations (> 0).
        on_iteration_limit: What reaching the cap does: ``"report"`` (keep the labels, report
            ``converged=False``), ``"error"`` (raise ConvergenceError) or
            ``"fallback_recompute"`` (recompute every label with IrLP).
        update_schedule: ``"in_place"`` (the published rule; a tolerance on CUDA) or
            ``"damped"`` (1/2-damped Jacobi; bitwise per backend).
        cuda_engine: The CUDA engine of :func:`update`: ``"automatic"`` (the fused engine),
            ``"fused"`` or ``"operators"`` (damped only); ignored by the sequential backend.
    """

    similarity_threshold: float | None = None
    tolerance: float = 1e-4
    max_iterations: int = 100000
    on_iteration_limit: OnLimitName = "report"
    update_schedule: ScheduleName = "in_place"
    cuda_engine: EngineName = "automatic"

    def _to_native(self) -> Any:
        o = native.LpOptions()
        tau = self.similarity_threshold
        o.similarity_threshold = None if tau is None else float(tau)
        o.tolerance = float(self.tolerance)
        o.max_iterations = as_int(self.max_iterations, "label_propagation.Options.max_iterations")
        o.on_iteration_limit = enum_member(
            native.LpOnLimit,
            self.on_iteration_limit,
            "label_propagation.Options.on_iteration_limit",
        )
        o.update_schedule = enum_member(
            native.LpSchedule, self.update_schedule, "label_propagation.Options.update_schedule"
        )
        o.cuda_engine = enum_member(
            native.Engine, self.cuda_engine, "label_propagation.Options.cuda_engine"
        )
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        out = cls(**copy_fields(o, tuple(f.name for f in dataclasses.fields(cls))))
        # The C++ fields are float: read back as the shortest decimal of the float (equal to the
        # value given, which passes the same float back)
        if out.similarity_threshold is not None:
            out.similarity_threshold = _dtypes.float32_value(out.similarity_threshold)
        out.tolerance = _dtypes.float32_value(out.tolerance)
        return out


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::label_propagation::stats``).

    Under ``"damped"`` every field is deterministic per backend; under ``"in_place"``
    ``affected``, ``converged``, ``fallback_used``, ``iterations`` and ``frontier_visits``
    depend on the schedule (``batch``, ``components``, ``sparsified_edges`` and
    ``initial_frontier`` stay deterministic).

    Attributes:
        affected: Vertices whose label bits changed.
        iterations: Propagation iterations.
        frontier_visits: Frontier entries processed.
        fallback_used: ``on_iteration_limit="fallback_recompute"`` recomputed the labels.
        converged: The propagation converged within ``max_iterations``.
        engine_used: ``"fused"`` or ``"operators"`` on cuda, the backend's engine otherwise.
        batch: What applying the batch did to the graph.
        components: Connected components of the batch's sparsified subgraph.
        sparsified_edges: Edges of the sparsified subgraph.
        initial_frontier: The affected vertices the propagation started from (``|V_aff|``).
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: ApplySummary
    components: int
    sparsified_edges: int
    initial_frontier: int

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "batch")
        return cls(**copy_fields(s, names), batch=ApplySummary._from_native(s.batch))


class Result:
    """The labels of a graph, kept up to date by :func:`update` (C++
    ``dyng::label_propagation::result``).

    Arrays are read-only, zero-copy :class:`dyng.Array` views of the current state (see the
    module description for copy-on-write).
    """

    __slots__ = ("_native", "_resources", "__weakref__")
    _native: Any
    _resources: Resources
    # the bookkeeping of dyng._writer (a side table)
    _writer = _writer_state.WRITER
    _streams = _writer_state.STREAMS
    _write_lock = _writer_state.WRITE_LOCK
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError("use dyng.label_propagation.compute()")

    @classmethod
    def _wrap(cls, handle: Any, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        _writer_state.track(self, resources)
        return self

    def _array(self, getter: Any, what: str) -> Array:
        return _result_array(
            self._native,
            getter,
            f"label_propagation.Result.{what}",
            (self._resources, self._writer, self._streams),
        )

    @property
    def labels(self) -> Array:
        """One label per vertex (float32 in [0, 1]; host or device memory, read-only)."""
        return self._array(self._native.labels, "labels")

    @property
    def seed_labels(self) -> Array:
        """One seed label per vertex (int8; -1 none, else the class 0 or 1), from compute's
        seeds and the labels of the batches' arrivals."""
        return self._array(self._native.seed_labels, "seed_labels")

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the tunables (every field of :class:`Options`)."""
        opt = with_options(
            Options, options or self.options, kwargs, "label_propagation.Result.set_options"
        )
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """Where the arrays live, ``"host"`` or ``"device"``."""
        return enum_name(self._native.space)

    def clone(self, resources: Resources | None = None) -> Result:
        """A deep copy (labels, seed labels, options, version) for ``resources``."""
        res = resolve(resources, self._resources)
        _check_backend(res, "label_propagation.Result.clone")
        return Result._wrap(self._native.clone(res._native), res)

    def reserve(
        self, max_batch_vertices: int, max_batch_edges: int, *, resources: Resources | None = None
    ) -> None:
        """Size the workspaces for batches of up to ``max_batch_vertices`` arrivals and
        ``max_batch_edges`` edges (the original's setup), so that :func:`update` allocates
        nothing for them (a stream's ``max_batch_vertices`` and ``max_batch_edges``). The labels
        are kept."""
        res = resolve_on(self, resources)
        _check_backend(res, "label_propagation.Result.reserve")
        self._native.reserve(
            res._native,
            as_int(max_batch_vertices, "reserve: max_batch_vertices"),
            as_int(max_batch_edges, "reserve: max_batch_edges"),
        )

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.label_propagation.Result cannot be pickled (it belongs to one state of one "
            "graph); send result.labels.to_numpy(), or use copy.deepcopy(result) / "
            "result.clone() within a process"
        )

    def __repr__(self) -> str:
        return (
            f"dyng.label_propagation.Result(num_vertices={len(self.labels)}, "
            f"graph_version={self.graph_version}, space={self.space!r})"
        )


def _check_backend(res: Resources, what: str) -> None:
    """AD19: there is no OpenMP backend of label_propagation in 0.4."""
    if res.backend == "openmp":
        raise NotSupportedError(
            f"{what}: label_propagation has no OpenMP backend in 0.4; its backends are "
            "'sequential' and 'cuda': pass resources=dyng.Resources(\"sequential\") (or "
            "dyng.Resources.cuda() with a CUDA plugin)"
        )


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")
    if graph.weight_dtype != _dtypes.FLOAT32 or graph.vertex_dtype != _dtypes.INT32:
        w = "unweighted" if graph.weight_dtype is None else graph.weight_dtype.name
        raise NotSupportedError(
            "dyng.label_propagation supports float32 graphs with int32 vertex ids (an undirected "
            "similarity graph: dyng.Graph.with_capacity(...), or float32 weights); this graph is "
            f"({graph.vertex_dtype.name}, {graph.edge_dtype.name}, {w})"
        )


def _seeds(seed_vertices: Any, seed_labels: Any) -> tuple[np.ndarray, np.ndarray]:
    if seed_vertices is None and seed_labels is None:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int8)
    if seed_vertices is None or seed_labels is None:
        raise TypeError(
            "label_propagation.compute: seed_vertices and seed_labels go together (both or neither)"
        )
    v = _dtypes.checked_cast(
        _dtypes.to_numpy(seed_vertices, "label_propagation.compute: seed_vertices"),
        _dtypes.INT32,
        "label_propagation.compute: seed_vertices",
    )
    lab = _dtypes.checked_cast(
        _dtypes.to_numpy(seed_labels, "label_propagation.compute: seed_labels"),
        _dtypes.INT8,
        "label_propagation.compute: seed_labels",
    )
    return v, lab


def compute(
    graph: Graph,
    seed_vertices: Any = None,
    seed_labels: Any = None,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Compute the labels of ``graph`` from scratch (IrLP, the published baseline).

    The published setup calls it on the empty graph of :meth:`dyng.Graph.with_capacity` (no
    seeds: every label 0.5) and lets :func:`update` bring in the batches; on a graph with edges
    it propagates the seeds' classes to every vertex of their components (a vertex that is not
    active is 0.5).

    Args:
        graph: An undirected float32 graph (compact or slot rows); it is not modified.
        seed_vertices: Active vertices with a known class, each once (integers), or None.
        seed_labels: Their classes, 0 or 1 (int8), or None.
        options: Options; the keywords (the fields of :class:`Options`) override its fields.
        resources: Default: the resources the graph was built with (sequential or cuda).
        **kwargs: Option fields.

    Returns:
        The labels, matching ``graph.version``.

    Raises:
        InvalidArgumentError: a seed is not an active vertex, listed twice, or its class is not
            0 or 1; an invalid option.
        NotSupportedError: a graph of another type, or OpenMP resources (use
            ``dyng.Resources("sequential")``).
        ConvergenceError: the iteration cap was reached under ``on_iteration_limit`` other than
            ``"report"``.
    """
    _check_graph(graph)
    opt = with_options(Options, options, kwargs, "label_propagation.compute")
    res = resolve_on(graph, resources)
    _check_backend(res, "label_propagation.compute")
    v, lab = _seeds(seed_vertices, seed_labels)
    handle = native.label_propagation_compute(res._native, graph._native, v, lab, opt._to_native())
    return Result._wrap(handle, res)


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply the vertex batch ``batch`` to ``graph`` and update the labels ``result`` (DynLP).

    Args:
        graph: A float32 graph with slot rows (:meth:`dyng.Graph.with_capacity`); the batch is
            applied to it and its version grows by one.
        batch: Arriving vertices with their labels (``insert_vertices``, ``vertex_labels``),
            their edges (insertions with float32 weights) and departing vertices
            (``delete_vertices``).
        result: A result of :func:`compute` (or of an earlier update) on ``graph``.
        resources: Default: the resources the graph was built with.

    Returns:
        The counters of this update.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state.
        InvalidArgumentError: an invalid id, label or weight in the batch (nothing is changed).
        NotSupportedError: compact rows, OpenMP resources, or an engine the schedule does not
            have.
        CapacityError: a slotted row overflows (nothing is changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.label_propagation.update: result must be a Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.label_propagation.update: batch must be a dyng.EdgeBatch")
    res = resolve_on(graph, resources)
    _check_backend(res, "label_propagation.update")
    nb = batch._native_for(graph)
    with writing([result], res):
        out = native.label_propagation_update(res._native, graph._native, nb, result._native)
    return Stats._from_native(out)
