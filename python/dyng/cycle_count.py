# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.cycle_count: exact histograms of directed simple cycles by length (TruCy / DynTruCy).

:func:`compute` counts every directed simple cycle of length 2..max_length (each cycle once, from
its smallest vertex; a 2-cycle is a pair of opposite edges; self-loops lie on no cycle).
:func:`update` applies a batch to the graph and changes the histogram by the cycles the batch
destroys and creates, without recounting the rest; afterwards the histogram equals
:func:`compute` on the new graph exactly.

The names mirror the C++ API (``dyng::cycle_count``). Supported graphs: int32 vertex ids, int32
or int64 edge offsets, unweighted or int32 weights (ignored), rows sorted by neighbour without
parallel edges (the default properties, or ``properties="cycle_enum_compatible"``). Cite with
``dyng.citation("cycle_count")``.

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 1, 2, 1], [1, 2, 0, 0])
    >>> hist = dyng.cycle_count.compute(g, max_length=3)
    >>> hist.counts.tolist(), hist.total
    ([0, 0, 1, 1], 2)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _dtypes
from ._backend import native
from ._convert import copy_fields, enum_member, enum_name, with_options
from .array import Array
from .batch import EdgeBatch
from .errors import NotSupportedError
from .graph import ApplySummary, Graph
from .resources import Resources, resolve

__all__ = ["Options", "Stats", "Result", "compute", "update"]


@dataclass
class Options:
    """Options of :func:`compute` and :func:`update` (C++ ``dyng::cycle_count::options``).

    ``max_length``, ``method`` and ``mode`` are fixed at :func:`compute`; the others are CUDA
    tunables that never change what is counted.

    Attributes:
        max_length: The longest counted cycle (>= 2), or -1 for no bound. Without a bound only
            the sequential compute is Johnson's algorithm; the OpenMP compute and every update
            enumerate simple paths (exponential time on some graphs): set a bound for large
            graphs. On cuda the effective bound must be at most 64.
        method: ``"johnson"``.
        mode: ``"simple"``.
        cuda_engine: ``"automatic"``, ``"fused"`` or ``"operators"`` (cuda only).
        scheduler: ``"work_queue"`` or ``"naive"`` (cuda only).
        work_items: ``"automatic"``, ``"roots"``, ``"edges"`` or ``"two_hop"`` (cuda only).
    """

    max_length: int = -1
    method: Literal["johnson"] = "johnson"
    mode: Literal["simple"] = "simple"
    cuda_engine: Literal["automatic", "fused", "operators"] = "automatic"
    scheduler: Literal["work_queue", "naive"] = "work_queue"
    work_items: Literal["automatic", "roots", "edges", "two_hop"] = "automatic"

    def _to_native(self) -> Any:
        o = native.CycleCountOptions()
        o.max_length = int(self.max_length)
        o.method = enum_member(native.SearchMethod, self.method, "cycle_count.Options.method")
        o.mode = enum_member(native.CycleMode, self.mode, "cycle_count.Options.mode")
        o.cuda_engine = enum_member(
            native.Engine, self.cuda_engine, "cycle_count.Options.cuda_engine"
        )
        o.scheduler = enum_member(
            native.CudaScheduler, self.scheduler, "cycle_count.Options.scheduler"
        )
        o.work_items = enum_member(
            native.CudaWorkItems, self.work_items, "cycle_count.Options.work_items"
        )
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        return cls(**copy_fields(o, tuple(f.name for f in dataclasses.fields(cls))))


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::cycle_count::stats``); all deterministic.

    Attributes:
        affected: Lengths whose count changed.
        iterations: Always 0.
        frontier_visits: Change edges searched (deletions + insertions).
        fallback_used: Always False.
        converged: Always True.
        engine_used: ``"operators"`` on the host backends, ``"fused"`` on cuda.
        batch: What applying the batch did to the graph.
        deletions: Edges of the net change that were deleted.
        insertions: Edges of the net change that were inserted.
        cycles_removed: Cycles of length <= the bound the batch destroyed.
        cycles_added: Cycles of length <= the bound the batch created.
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
    cycles_removed: int
    cycles_added: int

    @classmethod
    def _from_native(cls, s: Any) -> Stats:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "batch")
        return cls(**copy_fields(s, names), batch=ApplySummary._from_native(s.batch))


class Result:
    """A cycle histogram kept up to date by :func:`update` (C++ ``dyng::cycle_count::result``).

    ``counts[len]`` is the number of directed simple cycles of length ``len`` (entries 0 and 1
    are 0); the array has :attr:`bound` + 1 entries.
    """

    __slots__ = ("_native", "_resources", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("use dyng.cycle_count.compute()")

    @classmethod
    def _wrap(cls, handle: Any, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        return self

    @property
    def counts(self) -> Array:
        """The histogram (uint64), a zero-copy view of the current state."""
        from .sssp import _result_array

        return _result_array(self._native, self._native.counts, "cycle_count.Result.counts")

    def count(self, length: int) -> int:
        """The number of cycles of one length (0 outside [2, bound])."""
        return int(self._native.count(int(length)))

    @property
    def total(self) -> int:
        """The number of cycles of every counted length."""
        return int(self._native.total)

    @property
    def bound(self) -> int:
        """The longest length the histogram covers, min(max_length, max(n, 2)) or max(n, 2)."""
        return int(self._native.bound)

    def to_dict(self) -> dict[int, int]:
        """``{length: count}`` for the lengths with a non-zero count."""
        c = self._native.counts()
        return {i: int(v) for i, v in enumerate(np.from_dlpack(c).tolist()) if v}

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the CUDA tunables; ``max_length``, ``method`` and ``mode`` must stay the same."""
        opt = with_options(
            Options, options or self.options, kwargs, "cycle_count.Result.set_options"
        )
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The graph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """``"host"`` (the histogram is a host array on every backend)."""
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
            "dyng.cycle_count.Result cannot be pickled (it belongs to one state of one graph); "
            "send result.counts.to_numpy() (the histogram), or use copy.deepcopy(result) / "
            "result.clone() within a process"
        )

    def __repr__(self) -> str:
        return f"dyng.cycle_count.Result({self.to_dict()}, graph_version={self.graph_version})"


def _check_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("expected a dyng.Graph")
    if graph.vertex_dtype != _dtypes.INT32:
        raise NotSupportedError(
            "dyng.cycle_count supports int32 vertex ids only (graphs (int32, int32 or int64, "
            "unweighted or int32)); this graph has int64 ids: build it with vertex_dtype='int32'"
        )


def compute(
    graph: Graph,
    *,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Count the directed simple cycles of ``graph`` by length.

    Args:
        graph: The graph (sorted rows, no parallel edges); it is not modified.
        options: Options; the keywords ``max_length``, ``method``, ``mode``, ``cuda_engine``,
            ``scheduler`` and ``work_items`` override its fields.
        resources: Default: the resources the graph was built with.
        **kwargs: Option fields.

    Returns:
        The histogram, matching ``graph.version``.

    Raises:
        InvalidArgumentError: the rows are not sorted, parallel edges are allowed, or
            ``max_length`` is neither -1 nor >= 2.
        NotSupportedError: int64 vertex ids, or an unavailable backend or engine.
    """
    _check_graph(graph)
    opt = with_options(Options, options, kwargs, "cycle_count.compute")
    res = resolve(resources, graph._resources)
    return Result._wrap(
        native.cycle_count_compute(res._native, graph._native, opt._to_native()), res
    )


def update(
    graph: Graph, batch: EdgeBatch, result: Result, *, resources: Resources | None = None
) -> Stats:
    """Apply ``batch`` to ``graph`` and update the histogram ``result``.

    Postcondition: ``result`` equals ``compute(graph, options=result.options)`` exactly.

    Raises:
        StaleResultError: ``result`` does not match the graph's current state.
        InvalidArgumentError: the batch is invalid for the graph's semantics (nothing changed).
    """
    _check_graph(graph)
    if not isinstance(result, Result):
        raise TypeError("dyng.cycle_count.update: result must be a dyng.cycle_count.Result")
    if not isinstance(batch, EdgeBatch):
        raise TypeError("dyng.cycle_count.update: batch must be a dyng.EdgeBatch")
    res = resolve(resources, graph._resources)
    nb = batch._native_for(graph)
    return Stats._from_native(
        native.cycle_count_update(res._native, graph._native, nb, result._native)
    )
