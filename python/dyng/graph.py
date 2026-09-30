# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Graph: the dynamic graph container, its properties and what applying a batch did."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from . import _dtypes
from ._backend import native
from ._convert import copy_fields, enum_member, enum_name
from .errors import InvalidArgumentError
from .resources import Resources, resolve

if TYPE_CHECKING:
    from .batch import EdgeBatch

__all__ = ["BatchSemantics", "GraphProperties", "ApplySummary", "CSR", "Graph"]

PresetName = Literal["default", "mosp_compatible", "cycle_enum_compatible"]
SemanticsName = Literal["upsert_last_wins", "set"]

_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass
class BatchSemantics:
    """How a batch of edge changes is interpreted (C++ ``dyng::batch_semantics``).

    The default is :meth:`upsert_last_wins` (MOSP's ``applyChangeBatch()``): deletions first, each
    removing the first remaining (u, v); then insertions, an insertion of an existing edge
    overwriting its weights. :meth:`set` is CycleEnumeration-GPU's: the two lists are sets.

    Attributes:
        on_existing_insert: ``"upsert"``, ``"error"`` or ``"ignore"``.
        on_missing_delete: ``"ignore"`` or ``"error"``.
        on_self_loop: ``"keep"``, ``"drop"`` or ``"error"``.
        deletions_first: Deletions, then insertions (False: the reverse).
        allow_vertex_growth: An insertion naming v >= num_vertices grows the vertex set.
        as_sets: Read the deletions and the insertions as two sets (Step 0 of :meth:`set`).
    """

    on_existing_insert: Literal["upsert", "error", "ignore"] = "upsert"
    on_missing_delete: Literal["ignore", "error"] = "ignore"
    on_self_loop: Literal["keep", "drop", "error"] = "keep"
    deletions_first: bool = True
    allow_vertex_growth: bool = True
    as_sets: bool = False

    @classmethod
    def upsert_last_wins(cls) -> BatchSemantics:
        """MOSP's semantics (the default)."""
        return cls()

    @classmethod
    def set(cls) -> BatchSemantics:
        """CycleEnumeration-GPU's semantics: a batch is two sets."""
        return cls._from_native(native.BatchSemantics.set())

    @classmethod
    def _from_native(cls, s: Any) -> BatchSemantics:
        return cls(**copy_fields(s, tuple(f.name for f in dataclasses.fields(cls))))

    def _to_native(self) -> Any:
        s = native.BatchSemantics()
        s.on_existing_insert = enum_member(
            native.ExistingInsert, self.on_existing_insert, "on_existing_insert"
        )
        s.on_missing_delete = enum_member(
            native.MissingDelete, self.on_missing_delete, "on_missing_delete"
        )
        s.on_self_loop = enum_member(native.SelfLoop, self.on_self_loop, "on_self_loop")
        s.deletions_first = bool(self.deletions_first)
        s.allow_vertex_growth = bool(self.allow_vertex_growth)
        s.as_sets = bool(self.as_sets)
        return s


@dataclass
class GraphProperties:
    """The properties of a graph (C++ ``dyng::graph_properties``): direction, stored in-edges,
    layout, row order, parallel edges and batch semantics. The field names are the C++ ones.

    Attributes:
        directed: False: stored symmetric; a batch edge (u, v) changes both directions.
        store_transposed: Keep the in-edges (sssp needs them).
        num_weights: Weight columns (taken from the input when a graph is built).
        layout: ``"compact"`` (the only layout of 0.1).
        headroom: Spare capacity of the ``"slack"`` layout (0.3).
        order: ``"sorted"`` (rows sorted by neighbour) or ``"append"`` (MOSP).
        parallel_edges: ``"forbid"`` (a simple graph) or ``"allow"``.
        semantics: How batches are applied.
    """

    directed: bool = True
    store_transposed: bool = True
    num_weights: int = 1
    layout: Literal["compact", "slotted", "slack"] = "compact"
    headroom: float = 0.125
    order: Literal["sorted", "append"] = "sorted"
    parallel_edges: Literal["forbid", "allow"] = "forbid"
    semantics: BatchSemantics = field(default_factory=BatchSemantics)

    @classmethod
    def mosp_compatible(cls) -> GraphProperties:
        """Byte parity with MOSP's CSR: append order, parallel edges, upsert_last_wins."""
        return cls._from_native(native.GraphProperties.mosp_compatible())

    @classmethod
    def cycle_enum_compatible(cls) -> GraphProperties:
        """Parity with CycleEnumeration-GPU: sorted rows, a simple graph, set semantics."""
        return cls._from_native(native.GraphProperties.cycle_enum_compatible())

    @classmethod
    def preset(cls, name: PresetName) -> GraphProperties:
        """A preset by name: ``"default"``, ``"mosp_compatible"`` or ``"cycle_enum_compatible"``."""
        if name == "default":
            return cls()
        if name == "mosp_compatible":
            return cls.mosp_compatible()
        if name == "cycle_enum_compatible":
            return cls.cycle_enum_compatible()
        raise InvalidArgumentError(
            f"properties: {name!r} is not a preset ('default', 'mosp_compatible', "
            "'cycle_enum_compatible')"
        )

    @classmethod
    def _from_native(cls, p: Any) -> GraphProperties:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "semantics")
        return cls(**copy_fields(p, names), semantics=BatchSemantics._from_native(p.semantics))

    def _to_native(self) -> Any:
        p = native.GraphProperties()
        p.directed = bool(self.directed)
        p.store_transposed = bool(self.store_transposed)
        p.num_weights = int(self.num_weights)
        p.layout = enum_member(native.RowLayout, self.layout, "layout")
        p.headroom = float(self.headroom)
        p.order = enum_member(native.RowOrder, self.order, "order")
        p.parallel_edges = enum_member(native.MultiEdges, self.parallel_edges, "parallel_edges")
        semantics = self.semantics
        if isinstance(semantics, str):
            semantics = _semantics(semantics)
        p.semantics = semantics._to_native()
        return p


def _semantics(value: BatchSemantics | SemanticsName) -> BatchSemantics:
    if isinstance(value, BatchSemantics):
        return value
    if value == "upsert_last_wins":
        return BatchSemantics.upsert_last_wins()
    if value == "set":
        return BatchSemantics.set()
    raise InvalidArgumentError(
        f"semantics: {value!r} is not 'upsert_last_wins', 'set' or a dyng.BatchSemantics"
    )


def make_properties(
    properties: GraphProperties | PresetName | None,
    *,
    directed: bool | None = None,
    store_transposed: bool | None = None,
    layout: str | None = None,
    row_order: str | None = None,
    multi_edges: str | None = None,
    semantics: BatchSemantics | SemanticsName | None = None,
) -> GraphProperties:
    """The properties of a new graph: a preset or object, with keyword overrides (internal)."""
    if properties is None:
        props = GraphProperties()
    elif isinstance(properties, str):
        props = GraphProperties.preset(properties)  # type: ignore[arg-type]
    elif isinstance(properties, GraphProperties):
        props = dataclasses.replace(properties)
    else:
        raise TypeError("properties must be a preset name or a dyng.GraphProperties")
    if directed is not None:
        props.directed = bool(directed)
    if store_transposed is not None:
        props.store_transposed = bool(store_transposed)
    if layout is not None:
        props.layout = layout  # type: ignore[assignment]
    if row_order is not None:
        props.order = row_order  # type: ignore[assignment]
    if multi_edges is not None:
        props.parallel_edges = multi_edges  # type: ignore[assignment]
    if semantics is not None:
        props.semantics = _semantics(semantics)
    return props


@dataclass(frozen=True)
class ApplySummary:
    """What applying a batch did to a graph (C++ ``dyng::apply_summary``).

    Attributes:
        inserted_edges: Insertions that added a new edge.
        updated_edges: Insertions that overwrote an existing edge (upsert).
        deleted_edges: Deletions that removed an edge.
        ignored_deletions: Deletions of edges that did not exist.
        dropped_self_loops: Self-loop operations skipped.
        cancelled_pairs: Insert/delete pairs of one edge (set semantics).
        inserted_vertices: Vertices added (by vertex growth).
        deleted_vertices: Vertices removed.
        num_vertices_after: The vertex count after the batch.
        ignored_insertions: Insertions of existing edges left unchanged (ignore).
    """

    inserted_edges: int = 0
    updated_edges: int = 0
    deleted_edges: int = 0
    ignored_deletions: int = 0
    dropped_self_loops: int = 0
    cancelled_pairs: int = 0
    inserted_vertices: int = 0
    deleted_vertices: int = 0
    num_vertices_after: int = 0
    ignored_insertions: int = 0

    @classmethod
    def _from_native(cls, s: Any) -> ApplySummary:
        return cls(**copy_fields(s, tuple(f.name for f in dataclasses.fields(cls))))


class CSR(NamedTuple):
    """A host copy of the out-edges of a graph.

    Attributes:
        row_ptr: The num_vertices + 1 row offsets (the edge offset dtype).
        col_ind: The neighbour of each edge, in row order (the vertex dtype).
        weights: Shape (num_edges, K) int32 (a view), or None if unweighted.
    """

    row_ptr: np.ndarray
    col_ind: np.ndarray
    weights: np.ndarray | None


def _edge_dtype(requested: Any, vertex: np.dtype, stored_edges: int) -> np.dtype:
    if requested is not None:
        return _dtypes.id_dtype(requested, "edge_dtype")
    if vertex == _dtypes.INT64 or stored_edges > _INT32_MAX:
        return _dtypes.INT64
    return _dtypes.INT32


class Graph:
    """A dynamic directed (or symmetric) graph with K integer weight columns.

    Build one with :meth:`from_edges`, :meth:`from_csr` or a reader of :mod:`dyng.io`. The graph
    holds a version counter that grows by one with every applied batch; results remember the
    version (and the graph state) they match. The instantiation (id and weight types) is chosen
    from the input dtypes (the dtype rules of the Python API guide in the documentation);
    :attr:`vertex_dtype`, :attr:`edge_dtype` and :attr:`weighted` report it.

    A graph is not thread-safe in C++; the Python binding serializes calls that change it
    (:meth:`apply`, the algorithms' ``update``) against every other call on it.
    """

    __slots__ = ("_native", "_type", "_resources", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("use dyng.Graph.from_edges(), dyng.Graph.from_csr() or a reader of dyng.io")

    @classmethod
    def _wrap(cls, handle: Any, gtype: _dtypes.GraphType, resources: Resources) -> Graph:
        self = object.__new__(cls)
        self._native = handle
        self._type = gtype
        self._resources = resources
        return self

    # -- construction --------------------------------------------------------------------------
    @classmethod
    def from_edges(
        cls,
        src: Any,
        dst: Any,
        weights: Any = None,
        *,
        num_vertices: int | None = None,
        properties: GraphProperties | PresetName | None = None,
        directed: bool | None = None,
        store_transposed: bool | None = None,
        layout: str | None = None,
        row_order: str | None = None,
        multi_edges: str | None = None,
        semantics: BatchSemantics | SemanticsName | None = None,
        vertex_dtype: Any = None,
        edge_dtype: Any = None,
        resources: Resources | None = None,
    ) -> Graph:
        """Build a graph from an edge list.

        Edges are grouped by source; with ``row_order="sorted"`` (the default) rows are sorted by
        destination, with ``"append"`` they keep the input order. With ``multi_edges="forbid"``
        (the default) repeated (u, v) are merged (first position kept, last weights win). For an
        undirected graph every edge is stored in both directions.

        Args:
            src: Source of each edge (any integer array: NumPy, DLPack, buffer protocol, list).
            dst: Destination of each edge.
            weights: None (an unweighted graph), shape (m,) (one weight column) or (m, K);
                integers in [1, 2^31 - 1] for sssp.
            num_vertices: The vertex count (default: the largest id + 1).
            properties: A preset name (``"default"``, ``"mosp_compatible"``,
                ``"cycle_enum_compatible"``) or a :class:`GraphProperties`; the keywords below
                override its fields.
            directed: See :class:`GraphProperties`.
            store_transposed: See :class:`GraphProperties`.
            layout: See :class:`GraphProperties` (``layout``).
            row_order: :class:`GraphProperties` ``order``.
            multi_edges: :class:`GraphProperties` ``parallel_edges``.
            semantics: ``"upsert_last_wins"``, ``"set"`` or a :class:`BatchSemantics`.
            vertex_dtype: Force the vertex id type (``"int32"`` / ``"int64"``; values are
                checked). Default: from the dtypes of ``src`` and ``dst``.
            edge_dtype: Force the edge offset type. Default: int32 when the edge count fits
                (int64 with int64 ids).
            resources: The resources the graph belongs to (default: the default resources).

        Returns:
            The graph at version 0.

        Raises:
            InvalidArgumentError: an id out of range, arrays of different lengths, non-integer
                ids or weights.
            NotSupportedError: an unsupported type combination (listing the supported ones) or
                property.
            CapacityError: more edges than the edge offset type holds.
        """
        res = resolve(resources)
        s = _dtypes.to_numpy(src, "Graph.from_edges: src")
        d = _dtypes.to_numpy(dst, "Graph.from_edges: dst")
        if s.shape != d.shape:
            raise InvalidArgumentError(
                f"Graph.from_edges: src and dst differ in length ({s.size} vs {d.size})"
            )
        vertex = _dtypes.infer_id_dtype([src, dst], vertex_dtype, "Graph.from_edges")
        s = _dtypes.checked_cast(s, vertex, "Graph.from_edges: src")
        d = _dtypes.checked_cast(d, vertex, "Graph.from_edges: dst")
        props = make_properties(
            properties,
            directed=directed,
            store_transposed=store_transposed,
            layout=layout,
            row_order=row_order,
            multi_edges=multi_edges,
            semantics=semantics,
        )
        m = int(s.size)
        stored = m if props.directed else 2 * m
        edge = _edge_dtype(edge_dtype, vertex, stored)
        w = None
        k = 0
        if weights is not None:
            w2 = _dtypes.weights_matrix(weights, m, "Graph.from_edges: weights")
            k = int(w2.shape[1])
            w = w2.reshape(-1)
        gtype = _dtypes.find_graph_type(vertex, edge, weights is not None)
        if num_vertices is None:
            num_vertices = int(max(s.max(initial=-1), d.max(initial=-1))) + 1
        props.num_weights = k
        handle = gtype.native_class.from_edges(
            res._native, int(num_vertices), s, d, w, k, props._to_native()
        )
        return cls._wrap(handle, gtype, res)

    @classmethod
    def from_csr(
        cls,
        row_ptr: Any,
        col_ind: Any,
        weights: Any = None,
        *,
        properties: GraphProperties | PresetName | None = None,
        directed: bool | None = None,
        store_transposed: bool | None = None,
        layout: str | None = None,
        row_order: str | None = None,
        multi_edges: str | None = None,
        semantics: BatchSemantics | SemanticsName | None = None,
        vertex_dtype: Any = None,
        edge_dtype: Any = None,
        resources: Resources | None = None,
    ) -> Graph:
        """Build a graph from a CSR of the out-edges (e.g. ``scipy.sparse.csr_array``'s
        ``indptr``, ``indices`` and ``data``).

        The CSR is validated; with ``row_order="sorted"`` rows are sorted, with
        ``multi_edges="forbid"`` duplicates merged. For an undirected graph the CSR must already
        be symmetric.

        Args:
            row_ptr: num_vertices + 1 offsets; its dtype selects the edge offset type (int32 or
                int64) unless ``edge_dtype`` is given.
            col_ind: The neighbour of each edge; its dtype selects the vertex id type.
            weights: None, shape (m,) or (m, K) integers.
            properties: As for :meth:`from_edges` (and the keywords that override it).
            directed: See :meth:`from_edges`.
            store_transposed: See :meth:`from_edges`.
            layout: See :meth:`from_edges`.
            row_order: See :meth:`from_edges`.
            multi_edges: See :meth:`from_edges`.
            semantics: See :meth:`from_edges`.
            vertex_dtype: Force the vertex id type (values are checked).
            edge_dtype: Force the edge offset type (values are checked).
            resources: The resources the graph belongs to.

        Returns:
            The graph at version 0.
        """
        c = _dtypes.to_numpy(col_ind, "Graph.from_csr: col_ind")
        m = int(c.size)
        w = None
        k = 0
        if weights is not None:
            w2 = _dtypes.weights_matrix(weights, m, "Graph.from_csr: weights")
            k = int(w2.shape[1])
            w = np.ascontiguousarray(w2.T)  # objective-major, as the library stores weights
        return cls._from_csr_objective_major(
            row_ptr,
            c,
            w,
            k,
            weights is not None,
            make_properties(
                properties,
                directed=directed,
                store_transposed=store_transposed,
                layout=layout,
                row_order=row_order,
                multi_edges=multi_edges,
                semantics=semantics,
            ),
            vertex_dtype,
            edge_dtype,
            col_ind,
            resolve(resources),
        )

    @classmethod
    def _from_csr_objective_major(
        cls,
        row_ptr: Any,
        col_ind: np.ndarray,
        weights_km: np.ndarray | None,
        num_weights: int,
        weighted: bool,
        props: GraphProperties,
        vertex_dtype: Any,
        edge_dtype: Any,
        declared_col_ind: Any,
        res: Resources,
    ) -> Graph:
        r = _dtypes.to_numpy(row_ptr, "Graph.from_csr: row_ptr")
        vertex = _dtypes.infer_id_dtype([declared_col_ind], vertex_dtype, "Graph.from_csr")
        if edge_dtype is None:
            declared = _dtypes._declared_dtype(row_ptr)
            if vertex == _dtypes.INT64:
                edge = _dtypes.INT64
            elif declared is not None and declared.kind in "iu" and declared.itemsize >= 8:
                edge = _dtypes.INT64
            else:
                edge = _edge_dtype(None, vertex, int(col_ind.size))
        else:
            edge = _dtypes.id_dtype(edge_dtype, "Graph.from_csr: edge_dtype")
        gtype = _dtypes.find_graph_type(vertex, edge, weighted)
        r = _dtypes.checked_cast(r, edge, "Graph.from_csr: row_ptr")
        c = _dtypes.checked_cast(col_ind, vertex, "Graph.from_csr: col_ind")
        w = None if weights_km is None else np.ascontiguousarray(weights_km, np.int32).reshape(-1)
        props.num_weights = num_weights
        handle = gtype.native_class.from_csr(res._native, r, c, w, num_weights, props._to_native())
        return cls._wrap(handle, gtype, res)

    # -- properties ----------------------------------------------------------------------------
    @property
    def num_vertices(self) -> int:
        """The vertex count."""
        return int(self._native.num_vertices)

    @property
    def num_edges(self) -> int:
        """The number of stored (directed) edges (an undirected edge counts twice)."""
        return int(self._native.num_edges)

    @property
    def num_weights(self) -> int:
        """The number of weight columns K."""
        return int(self._native.num_weights)

    @property
    def directed(self) -> bool:
        """Whether the graph is directed."""
        return bool(self._native.directed)

    @property
    def has_transposed(self) -> bool:
        """Whether the in-edges are stored."""
        return bool(self._native.has_transposed)

    @property
    def version(self) -> int:
        """0 after construction; +1 per applied batch."""
        return int(self._native.version)

    @property
    def space(self) -> str:
        """Where the storage lives, ``"host"`` (or ``"device"`` for a graph of CUDA resources)."""
        return enum_name(self._native.space)

    @property
    def properties(self) -> GraphProperties:
        """The properties (a copy; ``num_weights`` as stored)."""
        return GraphProperties._from_native(self._native.properties)

    @property
    def vertex_dtype(self) -> np.dtype:
        """The vertex id type (int32 or int64)."""
        return self._type.vertex

    @property
    def edge_dtype(self) -> np.dtype:
        """The edge offset type (int32 or int64)."""
        return self._type.edge

    @property
    def weighted(self) -> bool:
        """Whether the weight type is int32 (otherwise the unweighted instantiation)."""
        return self._type.weighted

    @property
    def resources(self) -> Resources:
        """The resources the graph was built with (the default of every call on it)."""
        return self._resources

    # -- operations ----------------------------------------------------------------------------
    def apply(self, batch: EdgeBatch, *, resources: Resources | None = None) -> ApplySummary:
        """Apply a batch to the structure (and weights) of the graph.

        Results computed on the previous version become stale (their ``update`` raises
        :class:`~dyng.StaleResultError`); use the algorithms' ``update`` (or :func:`dyng.update`)
        to apply a batch and keep results current.

        Raises:
            InvalidArgumentError: an invalid id or weight, or a semantics rule says error.
            CapacityError: the edge count after the batch does not fit the edge offset type.
            NotSupportedError: vertex insertions or deletions (planned for 0.3).
        """
        res = resolve(resources, self._resources)
        nb = batch._native_for(self)
        return ApplySummary._from_native(self._native.apply(res._native, nb))

    def to_csr(self, *, resources: Resources | None = None) -> CSR:
        """A host copy of the out-edges (weights as a (m, K) view of the objective-major copy)."""
        res = resolve(resources, self._resources)
        row_ptr, col_ind, w = self._native.to_csr(res._native)
        return CSR(row_ptr, col_ind, None if w is None else w.T)

    def edges(
        self, *, resources: Resources | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The stored edges as (src, dst, weights of shape (m, K) or None), in row order."""
        csr = self.to_csr(resources=resources)
        src = np.repeat(np.arange(self.num_vertices, dtype=self.vertex_dtype), np.diff(csr.row_ptr))
        return src, csr.col_ind, csr.weights

    def clone(self, resources: Resources | None = None) -> Graph:
        """A deep copy (same properties, storage, version and state) for ``resources``."""
        res = resolve(resources, self._resources)
        return Graph._wrap(self._native.clone(res._native), self._type, res)

    def to_backend(self, resources: Resources) -> Graph:
        """The graph for the backend of ``resources`` (host backends versus cuda; a copy)."""
        res = resolve(resources)
        return Graph._wrap(self._native.to_backend(res._native), self._type, res)

    def reserve(self, edge_capacity: int, *, resources: Resources | None = None) -> None:
        """Pre-size the current storage for ``edge_capacity`` edges."""
        res = resolve(resources, self._resources)
        self._native.reserve(res._native, int(edge_capacity))

    def check_integrity(self, *, resources: Resources | None = None) -> None:
        """Check the storage invariants; raises :class:`~dyng.InternalError` if one is broken."""
        res = resolve(resources, self._resources)
        self._native.check_integrity(res._native)

    def __copy__(self) -> Graph:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Graph:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.Graph cannot be pickled (it may live in device memory and carries a version); "
            "send g.to_csr() and g.properties and rebuild it with dyng.Graph.from_csr(), or use "
            "copy.deepcopy(g) / g.clone() within a process"
        )

    def __repr__(self) -> str:
        w = "int32" if self.weighted else "unweighted"
        return (
            f"dyng.Graph(num_vertices={self.num_vertices}, num_edges={self.num_edges}, "
            f"num_weights={self.num_weights}, directed={self.directed}, version={self.version}, "
            f"types=({self.vertex_dtype.name}, {self.edge_dtype.name}, {w}), "
            f"backend={self._resources.backend!r})"
        )
