# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Graph: the dynamic graph container, its properties and what applying a batch did."""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from . import _dtypes
from . import _writer as _writer_state
from ._backend import native
from ._convert import as_bool, as_int, copy_fields, enum_member, enum_name
from ._writer import resolve_on
from .errors import InvalidArgumentError
from .resources import Resources, resolve

if TYPE_CHECKING:
    from .batch import EdgeBatch

__all__ = ["BatchSemantics", "GraphProperties", "ApplySummary", "CSR", "Graph", "VertexStatus"]

PresetName = Literal["default", "mosp_compatible", "cycle_enum_compatible"]
SemanticsName = Literal["upsert_last_wins", "set", "net_effect"]
LayoutName = Literal["slotted", "slack"]

_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass
class BatchSemantics:
    """How a batch of edge changes is interpreted (C++ ``dyng::batch_semantics``).

    The default is :meth:`upsert_last_wins` (MOSP's ``applyChangeBatch()``): deletions first, each
    removing the first remaining (u, v); then insertions, an insertion of an existing edge
    overwriting its weights. :meth:`set` is CycleEnumeration-GPU's: the two lists are sets.
    :meth:`net_effect` (0.4) is LabelPropagation-CUDA's, for graphs with slot rows
    (:meth:`Graph.with_capacity`): the net effect of each batch on the vertex set, its edges
    appended.

    Attributes:
        on_existing_insert: ``"upsert"``, ``"error"`` or ``"ignore"``.
        on_missing_delete: ``"ignore"`` or ``"error"``.
        on_self_loop: ``"keep"``, ``"drop"`` or ``"error"``.
        deletions_first: Deletions, then insertions (False: the reverse).
        allow_vertex_growth: An insertion naming v >= num_vertices grows the vertex set.
        as_sets: Read the deletions and the insertions as two sets (Step 0 of :meth:`set`).
        as_net_effect: Apply the net effect of each batch on the vertex set, its edges appended
            (Step 0 of :meth:`net_effect`; slot rows only; 0.4).
    """

    on_existing_insert: Literal["upsert", "error", "ignore"] = "upsert"
    on_missing_delete: Literal["ignore", "error"] = "ignore"
    on_self_loop: Literal["keep", "drop", "error"] = "keep"
    deletions_first: bool = True
    allow_vertex_growth: bool = True
    as_sets: bool = False
    as_net_effect: bool = False

    @classmethod
    def upsert_last_wins(cls) -> BatchSemantics:
        """MOSP's semantics (the default)."""
        return cls()

    @classmethod
    def set(cls) -> BatchSemantics:
        """CycleEnumeration-GPU's semantics: a batch is two sets."""
        return cls._from_native(native.BatchSemantics.set())

    @classmethod
    def net_effect(cls) -> BatchSemantics:
        """LabelPropagation-CUDA's semantics (0.4): a batch's vertex insertions and deletions by
        their net effect (an id in both lists ends deleted), its edges appended to the rows of
        both endpoints; self-loops kept, no vertex growth. Only graphs with slot rows
        (:meth:`Graph.with_capacity`) take it; edge deletions are not supported."""
        return cls._from_native(native.BatchSemantics.net_effect())

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
        s.deletions_first = as_bool(self.deletions_first, "BatchSemantics.deletions_first")
        s.allow_vertex_growth = as_bool(
            self.allow_vertex_growth, "BatchSemantics.allow_vertex_growth"
        )
        s.as_sets = as_bool(self.as_sets, "BatchSemantics.as_sets")
        s.as_net_effect = as_bool(self.as_net_effect, "BatchSemantics.as_net_effect")
        return s


@dataclass
class GraphProperties:
    """The properties of a graph (C++ ``dyng::graph_properties``): direction, stored in-edges,
    layout, row order, parallel edges and batch semantics. The field names are the C++ ones.

    Attributes:
        directed: False: stored symmetric; a batch edge (u, v) changes both directions.
        store_transposed: Keep the in-edges (sssp needs them).
        num_weights: Weight columns (taken from the input when a graph is built).
        layout: ``"compact"`` (the default; the only layout of 0.1), or the slot rows of 0.4
            built by :meth:`Graph.with_capacity`: ``"slotted"`` (fixed rows) or ``"slack"``
            (rows that move when they overflow).
        headroom: Spare capacity of the ``"slack"`` layout (0.4; default 0.125).
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
        p.directed = as_bool(self.directed, "GraphProperties.directed")
        p.store_transposed = as_bool(self.store_transposed, "GraphProperties.store_transposed")
        p.num_weights = as_int(self.num_weights, "GraphProperties.num_weights")
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
    if value == "net_effect":
        return BatchSemantics.net_effect()
    raise InvalidArgumentError(
        f"semantics: {value!r} is not 'upsert_last_wins', 'set', 'net_effect' or a "
        "dyng.BatchSemantics"
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
        props = GraphProperties.preset(properties)
    elif isinstance(properties, GraphProperties):
        props = dataclasses.replace(properties)
    else:
        raise TypeError("properties must be a preset name or a dyng.GraphProperties")
    if directed is not None:
        props.directed = as_bool(directed, "directed")
    if store_transposed is not None:
        props.store_transposed = as_bool(store_transposed, "store_transposed")
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
        dropped_insertions: Insertions not applied because an endpoint is not active after the
            batch (``net_effect``; 0.4).
        relocated_rows: Rows moved to the tail of the slot arena because the batch overflowed
            them (``"slack"`` rows; 0.4).
        compactions: Compactions of the rows the batch ran (``"slack"`` rows: 0 or 1; 0.4).
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
    dropped_insertions: int = 0
    relocated_rows: int = 0
    compactions: int = 0

    @classmethod
    def _from_native(cls, s: Any) -> ApplySummary:
        return cls(**copy_fields(s, tuple(f.name for f in dataclasses.fields(cls))))


class CSR(NamedTuple):
    """A host copy of the out-edges of a graph.

    Attributes:
        row_ptr: The num_vertices + 1 row offsets (the edge offset dtype).
        col_ind: The neighbour of each edge, in row order (the vertex dtype).
        weights: Shape (num_edges, K) int32 or float32 (0.4; a view), or None if unweighted.
    """

    row_ptr: np.ndarray
    col_ind: np.ndarray
    weights: np.ndarray | None


class VertexStatus(enum.IntEnum):
    """The state of a vertex of a graph with slot rows (C++ ``dyng::vertex_status``; 0.4).

    :meth:`Graph.vertex_status` returns these codes. A vertex of :meth:`Graph.with_capacity`
    starts ``inactive``; a batch's vertex insertion makes it ``active``, a deletion ``deleted``.
    The codes are this enumeration's, mapped from the library's by name.
    """

    active = 0
    deleted = 1
    inactive = 2


def _status_codes() -> np.ndarray:
    """The Python code (VertexStatus) of each native vertex_status value, by name."""
    table = np.zeros(256, dtype=np.uint8)
    for name in ("active", "deleted", "inactive"):
        table[int(getattr(native.VertexStatus, name).value)] = int(VertexStatus[name])
    return table


def _weights_for(
    weights: Any, m: int, weight_dtype: Any, what: str
) -> tuple[np.ndarray | None, int, np.dtype | None]:
    """(flat weights or None, K, weight dtype or None) of a graph's weights (AD14, ADR 0108)."""
    if weights is None:
        if weight_dtype is not None:
            raise InvalidArgumentError(f"{what}: weight_dtype without weights")
        return None, 0, None
    wdt = None if weight_dtype is None else _dtypes.weight_dtype_of(weight_dtype, what)
    if wdt == _dtypes.FLOAT32 or (wdt is None and _dtypes.is_float_weights(weights)):
        if wdt is None:
            view = _dtypes.weights_view(weights, m, what, allow_float=True)
            if view.dtype != _dtypes.FLOAT32:
                # float64 (or another floating type) without weight_dtype: never narrowed
                raise InvalidArgumentError(
                    f"{what}: {view.dtype.name} weights; the float graphs hold float32 weights: "
                    "pass weights.astype(numpy.float32), or weight_dtype='float32' to convert "
                    "them (checked: finite and within float32's range)"
                )
            weights = view
        w2 = _dtypes.float_weights_matrix(weights, m, what, explicit=True)
        return w2.reshape(-1), int(w2.shape[1]), _dtypes.FLOAT32
    w2 = _dtypes.weights_matrix(weights, m, what)
    return w2.reshape(-1), int(w2.shape[1]), _dtypes.INT32


def _edge_dtype(requested: Any, vertex: np.dtype, stored_edges: int) -> np.dtype:
    if requested is not None:
        return _dtypes.id_dtype(requested, "edge_dtype")
    if vertex == _dtypes.INT64 or stored_edges > _INT32_MAX:
        return _dtypes.INT64
    return _dtypes.INT32


class Graph:
    """A dynamic directed (or symmetric) graph with K integer weight columns, or (0.4) one
    float32 weight column.

    Build one with :meth:`from_edges`, :meth:`from_csr`, :meth:`with_capacity` (slot rows, 0.4)
    or a reader of :mod:`dyng.io`. The graph holds a version counter that grows by one with every
    applied batch; results remember the version (and the graph state) they match. The
    instantiation (id and weight types) is chosen from the input dtypes (the dtype rules of the
    Python API guide in the documentation): float32 weights select the float graphs of
    :mod:`dyng.label_propagation`, float64 weights only with ``weight_dtype="float32"``.
    :attr:`vertex_dtype`, :attr:`edge_dtype`, :attr:`weight_dtype` and :attr:`weighted` report
    it.

    A graph is not thread-safe in C++; the Python binding serializes calls that change it
    (:meth:`apply`, the algorithms' ``update``) against every other call on it.
    """

    __slots__ = ("_native", "_type", "_resources", "__weakref__")
    _native: Any
    _type: _dtypes.GraphType
    _resources: Resources
    # the bookkeeping of dyng._writer (a side table: the slots stay as in 0.1)
    _streams = _writer_state.STREAMS
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError("use dyng.Graph.from_edges(), dyng.Graph.from_csr() or a reader of dyng.io")

    @classmethod
    def _wrap(cls, handle: Any, gtype: _dtypes.GraphType, resources: Resources) -> Graph:
        self = object.__new__(cls)
        self._native = handle
        self._type = gtype
        self._resources = resources
        _writer_state.track(self, resources)
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
        weight_dtype: Any = None,
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
                integers in [1, 2^31 - 1] for sssp; float32 (one column) selects a float graph
                (0.4, :mod:`dyng.label_propagation`).
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
            weight_dtype: ``"float32"``: a float graph from floating-point (e.g. float64) or
                integer weights, converted and checked (finite, within float32's range); None:
                from the weights' dtype (float32 or integers; float64 is refused rather than
                narrowed silently). 0.4.
            resources: The resources the graph belongs to (default: the default resources).

        Returns:
            The graph at version 0.

        Raises:
            InvalidArgumentError: an id out of range, arrays of different lengths, non-integer
                ids, invalid weights (NaN or infinite floats; float64 without
                ``weight_dtype``).
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
        w, k, wdt = _weights_for(weights, m, weight_dtype, "Graph.from_edges: weights")
        gtype = _dtypes.find_graph_type(vertex, edge, wdt)
        if num_vertices is None:
            num_vertices = int(max(s.max(initial=-1), d.max(initial=-1))) + 1
        else:
            num_vertices = as_int(num_vertices, "Graph.from_edges: num_vertices")
        props.num_weights = k
        handle = gtype.native_class.from_edges(
            res._native, num_vertices, s, d, w, k, props._to_native()
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
        weight_dtype: Any = None,
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
            weights: None, shape (m,) or (m, K) integers, or float32 (0.4: a float graph).
            properties: As for :meth:`from_edges` (and the keywords that override it).
            directed: See :meth:`from_edges`.
            store_transposed: See :meth:`from_edges`.
            layout: See :meth:`from_edges`.
            row_order: See :meth:`from_edges`.
            multi_edges: See :meth:`from_edges`.
            semantics: See :meth:`from_edges`.
            vertex_dtype: Force the vertex id type (values are checked).
            edge_dtype: Force the edge offset type (values are checked).
            weight_dtype: As for :meth:`from_edges` (0.4).
            resources: The resources the graph belongs to.

        Returns:
            The graph at version 0.
        """
        c = _dtypes.to_numpy(col_ind, "Graph.from_csr: col_ind")
        m = int(c.size)
        flat, k, wdt = _weights_for(weights, m, weight_dtype, "Graph.from_csr: weights")
        w = None
        if flat is not None:
            # objective-major, as the library stores weights
            w = np.ascontiguousarray(flat.reshape(m, k).T)
        return cls._from_csr_objective_major(
            row_ptr,
            c,
            w,
            k,
            wdt,
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
        weighted: bool | np.dtype | None,
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
        wdt = np.int32 if gtype.weight is None else gtype.weight
        w = None if weights_km is None else np.ascontiguousarray(weights_km, wdt).reshape(-1)
        props.num_weights = num_weights
        handle = gtype.native_class.from_csr(res._native, r, c, w, num_weights, props._to_native())
        return cls._wrap(handle, gtype, res)

    @classmethod
    def with_capacity(
        cls,
        row_capacity: Any,
        *,
        vertex_capacity: int | None = None,
        directed: bool = False,
        layout: LayoutName = "slotted",
        semantics: BatchSemantics | SemanticsName = "net_effect",
        headroom: float = 0.125,
        edge_dtype: Any = None,
        weight_dtype: Any = "float32",
        resources: Resources | None = None,
    ) -> Graph:
        """An empty graph with slot rows (C++ ``graph::with_capacity``; 0.4): ``vertex_capacity``
        vertices, all inactive, vertex v owning ``row_capacity[v]`` slots.

        Vertices arrive and leave with the vertex arrays of the batches
        (``EdgeBatch(insert_vertices=..., vertex_labels=..., delete_vertices=...)``) under the
        ``"net_effect"`` semantics; a batch's edges are appended to the rows of both endpoints.
        This is LabelPropagation-CUDA's device graph and the published setup of
        :mod:`dyng.label_propagation` (``row_capacity``: a vertex-arrival stream's
        :attr:`~dyng.generators.VertexArrivalStream.row_capacity`, the full graph's degrees).

        ``layout="slotted"`` rows are fixed: an append past a row's capacity raises
        :class:`~dyng.CapacityError`; on the host backends before anything changes, on cuda after
        the batch was applied on the device (the graph is then updated and poisoned, and a result
        of it unusable; ADR 0074). ``layout="slack"`` rows (the
        streaming mode) get ``ceil(row_capacity[v] * (1 + headroom))`` slots; a row that
        overflows moves to the end of the slot arena with its order kept, and the rows are
        compacted when the slots left by moved rows outnumber the slots the rows own; there
        ``row_capacity`` is a hint that may be shorter than ``vertex_capacity`` (or empty).

        Args:
            row_capacity: The slots of each vertex (integers >= 0; the edge offset dtype).
            vertex_capacity: The vertex count (default: ``len(row_capacity)``).
            directed: Must be False in 0.4 (the rows are symmetric).
            layout: ``"slotted"`` (the default) or ``"slack"``.
            semantics: ``"net_effect"`` (the default; required in 0.4) or a
                :class:`BatchSemantics` equal to :meth:`BatchSemantics.net_effect` (with
                ``allow_vertex_growth`` set or not for ``"slack"`` rows).
            headroom: The spare share of the ``"slack"`` rows (in [0, 16]).
            edge_dtype: The edge offset type (default: int32 when the sum of the capacities
                fits it, for ``"slack"`` rows twice the sum with the headroom; else int64).
            weight_dtype: ``"float32"`` (the default: label_propagation's graphs), ``"int32"``,
                or None for an unweighted graph.
            resources: The resources the graph belongs to (default: the default resources).

        Returns:
            The graph at version 0, every vertex inactive and every row empty.

        Raises:
            InvalidArgumentError: a negative capacity, ``row_capacity`` of another length
                (``"slotted"``), or a headroom outside [0, 16].
            CapacityError: the sum of the capacities does not fit the edge offset type.
            NotSupportedError: a property the 0.4 slot rows do not support (named).

        Example:
            >>> import numpy as np, dyng
            >>> g = dyng.Graph.with_capacity([1, 1, 0], resources=dyng.Resources("sequential"))
            >>> b = dyng.EdgeBatch(insert=([0], [1], np.array([0.5], dtype=np.float32)),
            ...                    insert_vertices=[0, 1], vertex_labels=[1, -1])
            >>> g.apply(b).inserted_vertices, g.num_active_vertices, g.num_edges
            (2, 2, 2)
        """
        res = resolve(resources)
        cap = _dtypes.to_numpy(row_capacity, "Graph.with_capacity: row_capacity")
        if cap.size and (cap.dtype.kind == "b" or cap.dtype.kind not in "iu"):
            raise InvalidArgumentError(
                f"Graph.with_capacity: row_capacity must be integers, got {cap.dtype.name}"
            )
        n = (
            int(cap.size)
            if vertex_capacity is None
            else as_int(vertex_capacity, "Graph.with_capacity: vertex_capacity")
        )
        if edge_dtype is None:
            # int32 when the slots fit it; slack rows keep room for moved rows (twice the slots)
            total = int(cap.astype(np.int64).sum()) if cap.size else 0
            if layout == "slack":
                total = 2 * (int(total * (1.0 + float(headroom))) + int(cap.size))
            edge = _dtypes.INT32 if total <= _INT32_MAX else _dtypes.INT64
        else:
            edge = _dtypes.id_dtype(edge_dtype, "Graph.with_capacity: edge_dtype")
        cap = _dtypes.checked_cast(cap, edge, "Graph.with_capacity: row_capacity")
        if weight_dtype is None:
            wdt: np.dtype | None = None
        else:
            wdt = _dtypes.weight_dtype_of(weight_dtype, "Graph.with_capacity: weight_dtype")
        gtype = _dtypes.find_graph_type(_dtypes.INT32, edge, wdt)
        props = GraphProperties(
            directed=as_bool(directed, "Graph.with_capacity: directed"),
            store_transposed=False,
            num_weights=0 if wdt is None else 1,
            layout=layout,
            headroom=float(headroom),
            order="append",
            parallel_edges="allow",
            semantics=_semantics(semantics),
        )
        handle = gtype.native_class.with_capacity(res._native, n, cap, props._to_native())
        return cls._wrap(handle, gtype, res)

    # -- properties ----------------------------------------------------------------------------
    @property
    def num_vertices(self) -> int:
        """The vertex count (for slot rows the vertex capacity, whatever the states)."""
        return int(self._native.num_vertices)

    @property
    def num_active_vertices(self) -> int:
        """The number of active vertices (0.4), which is :attr:`num_vertices` for compact rows
        (every vertex is active) and the vertices in ``VertexStatus.active`` for slot rows."""
        return int(self._native.num_active_vertices)

    def vertex_status(self) -> np.ndarray:
        """A host copy of the vertex states (0.4): one :class:`VertexStatus` code per vertex
        (uint8); every vertex ``active`` for compact rows."""
        raw = self._native.vertex_status()
        if raw.size == 0:
            return np.full(self.num_vertices, int(VertexStatus.active), dtype=np.uint8)
        out: np.ndarray = _status_codes()[raw]
        return out

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
        """Whether the weight type is int32 (otherwise the unweighted instantiation, or a float32
        graph of 0.4: see :attr:`weight_dtype`)."""
        return self._type.weighted

    @property
    def weight_dtype(self) -> np.dtype | None:
        """The weight type (int32, float32 for label_propagation's graphs of 0.4, or None for an
        unweighted graph)."""
        return self._type.weight

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
            NotSupportedError: vertex insertions or deletions (planned for 0.4).
        """
        res = resolve_on(self, resources)
        nb = batch._native_for(self)
        return ApplySummary._from_native(self._native.apply(res._native, nb))

    def to_csr(self, *, resources: Resources | None = None) -> CSR:
        """A host copy of the out-edges (weights as a (m, K) view of the objective-major copy)."""
        res = resolve_on(self, resources)
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
        res = resolve_on(self, resources)
        return Graph._wrap(self._native.clone(res._native), self._type, res)

    def to_backend(self, resources: Resources) -> Graph:
        """The graph for the backend of ``resources`` (host backends versus cuda; a copy)."""
        res = resolve(resources)
        return Graph._wrap(self._native.to_backend(res._native), self._type, res)

    def reserve(self, edge_capacity: int, *, resources: Resources | None = None) -> None:
        """Pre-size the current storage for ``edge_capacity`` edges."""
        res = resolve_on(self, resources)
        self._native.reserve(res._native, as_int(edge_capacity, "Graph.reserve: edge_capacity"))

    def check_integrity(self, *, resources: Resources | None = None) -> None:
        """Check the storage invariants; raises :class:`~dyng.InternalError` if one is broken."""
        res = resolve_on(self, resources)
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
        w = "unweighted" if self.weight_dtype is None else self.weight_dtype.name
        return (
            f"dyng.Graph(num_vertices={self.num_vertices}, num_edges={self.num_edges}, "
            f"num_weights={self.num_weights}, directed={self.directed}, version={self.version}, "
            f"types=({self.vertex_dtype.name}, {self.edge_dtype.name}, {w}), "
            f"backend={self._resources.backend!r})"
        )
