# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Hypergraph: the dynamic hypergraph container, its properties, batches and summaries.

A hypergraph holds hyperedges (sets of vertices) identified by 0-based int64 ids. Hyperedge i of
:meth:`Hypergraph.from_hyperedges` gets id i; a batch (:class:`HyperedgeBatch`) deletes
hyperedges by id and inserts new ones, whose ids the store's best-fit reuse chooses and
:attr:`HyperApplySummary.inserted_ids` reports (read them there, never predict them). The C++
API is ``dyng::hypergraph<V, W>`` (``<dyng/hypergraph/hypergraph.hpp>``); the concepts page
"Hypergraphs and hyperedge ids" of the documentation explains ids, ``id_reuse`` and the
capacity presets.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np

from . import _dtypes
from . import _writer as _writer_state
from ._backend import native
from ._convert import as_bool, as_int, copy_fields, enum_member, enum_name
from ._writer import resolve_on
from .errors import InvalidArgumentError, NotSupportedError
from .resources import Resources, resolve

__all__ = [
    "CapacityPolicy",
    "HypergraphProperties",
    "HyperApplySummary",
    "HyperedgeBatch",
    "HyperedgeList",
    "Incidence",
    "LineGraph",
    "StoragePathCounters",
    "StorageStats",
    "Hypergraph",
]

StorageName = Literal["escher", "slack_csr"]
LineGraphName = Literal["none", "automatic", "sorted_merge", "slack_csr"]
IdReuseName = Literal["erase_first", "insert_first"]
CapacityPresetName = Literal["default", "escher_gpu", "mosp_escher"]

INT32 = _dtypes.INT32
INT64 = _dtypes.INT64


@dataclass
class CapacityPolicy:
    """How a hypergraph's storage is sized and grows (C++ ``dyng::capacity_policy``).

    A tree starts with ``min(max(min_initial_payload, size + floor(size * initial_headroom) +
    initial_slack), max_payload)`` payload entries. A full payload is compacted first
    (``compact_before_grow``), then grows to ``max(needed + needed / 4, floor(capacity *
    grow_factor))`` entries, at most ``max_payload``, or raises :class:`~dyng.CapacityError`
    when ``grow_factor`` is 0. :meth:`escher_gpu` and :meth:`mosp_escher` reproduce the two
    originals exactly; the default is dynG's policy.

    Attributes:
        initial_headroom: Slack of a new tree as a fraction of its size.
        initial_slack: Slack entries added to every new tree.
        min_initial_payload: Lower bound of the initial payload (0: none).
        grow_factor: Growth of a full payload (1.5), or 0: never grow.
        compact_before_grow: Compact a full payload before growing it.
        max_payload: Largest payload per tree (0: 2^31 - 1, the int32 offsets of the store).
        row_alignment: An inserted row takes a multiple of this, plus a terminator.
        compaction_headroom: After a compaction, grow while fewer than capacity /
            compaction_headroom entries are free (0: never).
        exact_scratch: Operation buffers sized exactly (ESCHER-GPU) or grown geometrically.
    """

    initial_headroom: float = 0.125
    initial_slack: int = 1024
    min_initial_payload: int = 0
    grow_factor: float = 1.5
    compact_before_grow: bool = True
    max_payload: int = 0
    row_alignment: int = 4
    compaction_headroom: int = 8
    exact_scratch: bool = False

    @classmethod
    def escher_gpu(cls, min_initial_payload: int = 0) -> CapacityPolicy:
        """ESCHER-GPU@abcf9b2's policy: trees start at ``size + size / 8 + 1024`` entries (at
        least ``min_initial_payload``, its ``--payload-capacity``), never compact, and grow by
        its rule up to 2^31 - 1."""
        return cls._from_native(
            native.CapacityPolicy.escher_gpu(
                as_int(min_initial_payload, "CapacityPolicy.escher_gpu: min_initial_payload")
            )
        )

    @classmethod
    def mosp_escher(cls, extra: int = 1 << 20) -> CapacityPolicy:
        """MOSP_ESCHER@4b86159's policy: trees start at ``floor(size * 1.3) + extra`` entries (at
        most 2.0e9), compact at its trigger points and never grow (:class:`~dyng.CapacityError`).
        """
        return cls._from_native(
            native.CapacityPolicy.mosp_escher(as_int(extra, "CapacityPolicy.mosp_escher: extra"))
        )

    @classmethod
    def preset(cls, name: CapacityPresetName) -> CapacityPolicy:
        """A preset by name: ``"default"``, ``"escher_gpu"`` or ``"mosp_escher"`` (with their
        default arguments)."""
        if name == "default":
            return cls()
        if name == "escher_gpu":
            return cls.escher_gpu()
        if name == "mosp_escher":
            return cls.mosp_escher()
        raise InvalidArgumentError(
            f"capacity: {name!r} is not a preset ('default', 'escher_gpu', 'mosp_escher')"
        )

    @classmethod
    def _from_native(cls, p: Any) -> CapacityPolicy:
        return cls(**copy_fields(p, tuple(f.name for f in dataclasses.fields(cls))))

    def _to_native(self) -> Any:
        p = native.CapacityPolicy()
        p.initial_headroom = float(self.initial_headroom)
        p.initial_slack = as_int(self.initial_slack, "CapacityPolicy.initial_slack")
        p.min_initial_payload = as_int(
            self.min_initial_payload, "CapacityPolicy.min_initial_payload"
        )
        p.grow_factor = float(self.grow_factor)
        p.compact_before_grow = as_bool(
            self.compact_before_grow, "CapacityPolicy.compact_before_grow"
        )
        p.max_payload = as_int(self.max_payload, "CapacityPolicy.max_payload")
        p.row_alignment = as_int(self.row_alignment, "CapacityPolicy.row_alignment")
        p.compaction_headroom = as_int(
            self.compaction_headroom, "CapacityPolicy.compaction_headroom"
        )
        p.exact_scratch = as_bool(self.exact_scratch, "CapacityPolicy.exact_scratch")
        return p


def _capacity(value: CapacityPolicy | CapacityPresetName) -> CapacityPolicy:
    if isinstance(value, CapacityPolicy):
        return dataclasses.replace(value)
    if isinstance(value, str):
        return CapacityPolicy.preset(value)
    raise TypeError("capacity must be a preset name or a dyng.CapacityPolicy")


@dataclass
class HypergraphProperties:
    """The properties of a hypergraph, fixed at construction (C++ ``dyng::hypergraph_properties``).

    Attributes:
        storage: ``"escher"`` (the ESCHER store; ``"slack_csr"`` is planned for 0.4).
        line_graph: ``"none"`` (:meth:`Hypergraph.h2h` raises), ``"automatic"`` (built when
            first read, then maintained), ``"sorted_merge"`` (built with the hypergraph and
            maintained by every batch); ``"slack_csr"`` is planned for 0.4.
        reuse: The order of a batch's deletions and insertions, which decides the reused ids:
            ``"erase_first"`` (ESCHER's order, the default) or ``"insert_first"`` (H-SOSP's).
        capacity: Sizing and growth of the storage (:class:`CapacityPolicy`).
        store_v2h: Keep vertex -> hyperedges (:meth:`Hypergraph.v2h`, the line graph).
    """

    storage: StorageName = "escher"
    line_graph: LineGraphName = "automatic"
    reuse: IdReuseName = "erase_first"
    capacity: CapacityPolicy = field(default_factory=CapacityPolicy)
    store_v2h: bool = True

    @classmethod
    def _from_native(cls, p: Any) -> HypergraphProperties:
        return cls(
            storage=enum_name(p.storage),  # type: ignore[arg-type]
            line_graph=enum_name(p.line_graph),  # type: ignore[arg-type]
            reuse=enum_name(p.reuse),  # type: ignore[arg-type]
            capacity=CapacityPolicy._from_native(p.capacity),
            store_v2h=bool(p.store_v2h),
        )

    def _to_native(self) -> Any:
        p = native.HypergraphProperties()
        p.storage = enum_member(native.HypergraphStorage, self.storage, "storage")
        p.line_graph = enum_member(native.LineGraphPolicy, self.line_graph, "line_graph")
        p.reuse = enum_member(native.IdReuse, self.reuse, "id_reuse")
        p.capacity = _capacity(self.capacity)._to_native()
        p.store_v2h = as_bool(self.store_v2h, "HypergraphProperties.store_v2h")
        return p


def make_hypergraph_properties(
    properties: HypergraphProperties | None,
    *,
    storage: StorageName | None = None,
    line_graph: LineGraphName | None = None,
    id_reuse: IdReuseName | None = None,
    capacity: CapacityPolicy | CapacityPresetName | None = None,
    store_v2h: bool | None = None,
) -> HypergraphProperties:
    """The properties of a new hypergraph with keyword overrides (internal)."""
    if properties is None:
        props = HypergraphProperties()
    elif isinstance(properties, HypergraphProperties):
        props = dataclasses.replace(properties)
    else:
        raise TypeError("properties must be a dyng.HypergraphProperties")
    if storage is not None:
        props.storage = storage
    if line_graph is not None:
        props.line_graph = line_graph
    if id_reuse is not None:
        props.reuse = id_reuse
    if capacity is not None:
        props.capacity = _capacity(capacity)
    if store_v2h is not None:
        props.store_v2h = as_bool(store_v2h, "store_v2h")
    return props


@dataclass(frozen=True)
class HyperApplySummary:
    """What applying a batch did to a hypergraph (C++ ``dyng::hyper_apply_summary``).

    Attributes:
        inserted_hyperedges: Hyperedges inserted.
        deleted_hyperedges: Live hyperedges deleted.
        inserted_incidences: Incidences added by incidence edits (0.4).
        deleted_incidences: Incidences removed by incidence edits (0.4).
        skipped_ops: ``ignored_deletions + dropped_members``.
        ignored_deletions: Deletions of dead, out-of-range or repeated ids.
        dropped_members: Repeated members of inserted hyperedges dropped.
        num_hyperedges_after: Live hyperedges after the batch.
        id_bound_after: :attr:`Hypergraph.hyperedge_id_bound` after the batch.
        inserted_ids: The id of every inserted hyperedge, in batch order (int64; the store's
            best-fit choice, authoritative).
    """

    inserted_hyperedges: int = 0
    deleted_hyperedges: int = 0
    inserted_incidences: int = 0
    deleted_incidences: int = 0
    skipped_ops: int = 0
    ignored_deletions: int = 0
    dropped_members: int = 0
    num_hyperedges_after: int = 0
    id_bound_after: int = 0
    inserted_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def _from_native(cls, s: Any) -> HyperApplySummary:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name != "inserted_ids")
        ids = np.asarray(s.inserted_ids, dtype=np.int64)
        ids.flags.writeable = False
        return cls(**copy_fields(s, names), inserted_ids=ids)


@dataclass(frozen=True)
class StoragePathCounters:
    """Which kernel paths an operation family took (degree-binned thread, warp, block)."""

    thread: int = 0
    warp: int = 0
    block: int = 0


@dataclass(frozen=True)
class StorageStats:
    """The storage counters of a hypergraph (C++ ``dyng::storage_stats``).

    Attributes:
        device_bytes: Device memory of the stores and kept row views (ESCHER-GPU's
            ``deviceBytes()`` scope).
        view_bytes: Memory of the views built on request.
        host_bytes: Host memory of the storage (CPU backends).
        payload_capacity: h2v payload entries allocated.
        payload_used: h2v payload entries in use.
        growths: Payload growths (all trees).
        compactions: Payload compactions (all trees).
        fill: Appends of values to rows (v2h).
        reuse: Inserted rows that reused a deleted slot (h2v).
        unfill: Removals of values from rows (v2h).
    """

    device_bytes: int = 0
    view_bytes: int = 0
    host_bytes: int = 0
    payload_capacity: int = 0
    payload_used: int = 0
    growths: int = 0
    compactions: int = 0
    fill: StoragePathCounters = field(default_factory=StoragePathCounters)
    reuse: StoragePathCounters = field(default_factory=StoragePathCounters)
    unfill: StoragePathCounters = field(default_factory=StoragePathCounters)

    @classmethod
    def _from_native(cls, s: Any) -> StorageStats:
        def paths(p: Any) -> StoragePathCounters:
            return StoragePathCounters(int(p.thread), int(p.warp), int(p.block))

        names = tuple(
            f.name for f in dataclasses.fields(cls) if f.name not in ("fill", "reuse", "unfill")
        )
        return cls(
            **copy_fields(s, names),
            fill=paths(s.fill),
            reuse=paths(s.reuse),
            unfill=paths(s.unfill),
        )


class Incidence(NamedTuple):
    """A host copy of one incidence relation (h2v or v2h) at one version: row r lists
    ``col_ind[row_ptr[r]:row_ptr[r + 1]]`` in increasing order.

    Attributes:
        row_ptr: num_rows + 1 offsets (int64).
        col_ind: The ids of every row (vertices for h2v, int64 hyperedge ids for v2h).
        version: The hypergraph version the copy describes.
    """

    row_ptr: np.ndarray
    col_ind: np.ndarray
    version: int

    def row(self, r: int) -> np.ndarray:
        """The ids of row ``r``."""
        return self.col_ind[int(self.row_ptr[r]) : int(self.row_ptr[r + 1])]

    @property
    def num_rows(self) -> int:
        """The number of rows."""
        return int(self.row_ptr.size) - 1


class LineGraph(NamedTuple):
    """A host copy of the line graph (h2h) at one version: row i lists the ids of the live
    hyperedges that share a vertex with hyperedge i, sorted; a dead id has an empty row.

    Attributes:
        row_ptr: num_rows + 1 offsets (int64).
        col_ind: The neighbour ids (int64).
        out: Per row, the offset of the first neighbour with an id greater than the row's.
        version: The hypergraph version the copy describes.
    """

    row_ptr: np.ndarray
    col_ind: np.ndarray
    out: np.ndarray
    version: int

    def row(self, r: int) -> np.ndarray:
        """The neighbours of hyperedge ``r``."""
        return self.col_ind[int(self.row_ptr[r]) : int(self.row_ptr[r + 1])]

    @property
    def num_rows(self) -> int:
        """The number of rows."""
        return int(self.row_ptr.size) - 1


class HyperedgeList(NamedTuple):
    """Hyperedges as a CSR of member lists (the members of hyperedge i are
    ``members[offsets[i]:offsets[i + 1]]``), with the vertex count and the weights.

    Attributes:
        num_vertices: Vertex ids lie in [0, num_vertices).
        offsets: num_hyperedges + 1 offsets (int64).
        members: The members, back to back (int32).
        weights: One int32 weight per hyperedge, or None.
    """

    num_vertices: int
    offsets: np.ndarray
    members: np.ndarray
    weights: np.ndarray | None

    @property
    def num_hyperedges(self) -> int:
        """The number of hyperedges."""
        return int(self.offsets.size) - 1

    def to_lists(self) -> list[list[int]]:
        """The member lists as Python lists."""
        o = self.offsets
        return [self.members[int(o[i]) : int(o[i + 1])].tolist() for i in range(o.size - 1)]


def _csr_of_lists(lists: Iterable[Any], what: str) -> tuple[np.ndarray, np.ndarray]:
    """(offsets int64, members as given) of an iterable of member lists."""
    offsets = [0]
    parts: list[np.ndarray] = []
    for i, members in enumerate(lists):
        a = _dtypes.to_numpy(members, f"{what}: hyperedge {i}")
        if a.size and (a.dtype.kind == "b" or a.dtype.kind not in "iu"):
            raise InvalidArgumentError(f"{what}: hyperedge {i}: members must be integers")
        parts.append(a)
        offsets.append(offsets[-1] + int(a.size))
    if parts:
        members_all = np.concatenate([p.astype(np.int64, copy=False) for p in parts])
    else:
        members_all = np.zeros(0, dtype=np.int64)
    return np.asarray(offsets, dtype=np.int64), members_all


def _vertex_dtype(arrays: list[Any], requested: Any, what: str) -> np.dtype:
    """The vertex type of a hypergraph built from ``arrays``: int32, the only one compiled
    (PLAN 4.4.3); an int64 input is never narrowed without ``vertex_dtype="int32"``."""
    vertex = _dtypes.infer_id_dtype(arrays, requested, what)
    if vertex != INT32:
        raise NotSupportedError(
            f"{what}: the members are {vertex.name}, and hypergraphs have int32 vertex ids (the "
            "supported combinations are (vertex int32, weight int32) and (vertex int32, "
            "unweighted)); pass vertex_dtype='int32' to convert them with a range check"
        )
    return vertex


def _native_class(weighted: bool) -> Any:
    return native.HypergraphI32I32 if weighted else native.HypergraphI32U


class HyperedgeBatch:
    """A batch of hypergraph changes: hyperedge deletions (by id) and insertions (member lists).

    The order in which deletions and insertions reach the store is a property of the hypergraph
    (:attr:`HypergraphProperties.reuse`); the ids of the inserted hyperedges are reported by
    :attr:`HyperApplySummary.inserted_ids`. Deletions of dead, out-of-range or repeated ids are
    skipped and counted, as are repeated members of an inserted hyperedge.

    The arrays are kept as given and converted, with range checks, to the hypergraph's types at
    every use (int64 ids are never narrowed silently).

    Args:
        delete_hyperedges: Ids of the hyperedges to delete (int64).
        insert_hyperedges: The member lists of the hyperedges to insert (e.g.
            ``[[3, 9, 12], [9, 12, 40, 41]]``); or give ``insert_offsets`` and
            ``insert_members`` (a CSR) instead.
        insert_weights: One weight per inserted hyperedge (a weighted hypergraph).
        insert_offsets: CSR offsets of the inserted hyperedges (insertions + 1, from 0).
        insert_members: CSR members of the inserted hyperedges.
        delete_incidences: ``(hyperedges, vertices)``: vertices removed from live hyperedges
            (planned for 0.4: applying such a batch raises :class:`~dyng.NotSupportedError`).
        insert_incidences: ``(hyperedges, vertices)``: vertices added to live hyperedges (0.4).

    Examples:
        >>> import dyng
        >>> b = dyng.HyperedgeBatch(
        ...     delete_hyperedges=[17, 42], insert_hyperedges=[[3, 9], [9, 12, 40]]
        ... )
        >>> b.num_deletions, b.num_insertions
        (2, 2)
    """

    __slots__ = ("_del", "_offsets", "_members", "_weights", "_del_inc", "_ins_inc")

    def __init__(
        self,
        delete_hyperedges: Any = None,
        insert_hyperedges: Iterable[Any] | None = None,
        insert_weights: Any = None,
        *,
        insert_offsets: Any = None,
        insert_members: Any = None,
        delete_incidences: tuple[Any, Any] | None = None,
        insert_incidences: tuple[Any, Any] | None = None,
    ) -> None:
        self._del = self._ids(delete_hyperedges, "delete_hyperedges")
        if insert_hyperedges is not None:
            if insert_offsets is not None or insert_members is not None:
                raise InvalidArgumentError(
                    "HyperedgeBatch: give insert_hyperedges or insert_offsets / insert_members, "
                    "not both"
                )
            self._offsets, self._members = _csr_of_lists(
                insert_hyperedges, "HyperedgeBatch: insert_hyperedges"
            )
        elif insert_offsets is not None or insert_members is not None:
            if insert_offsets is None or insert_members is None:
                raise InvalidArgumentError(
                    "HyperedgeBatch: insert_offsets and insert_members go together"
                )
            self._offsets = _dtypes.checked_cast(
                _dtypes.to_numpy(insert_offsets, "HyperedgeBatch: insert_offsets"),
                INT64,
                "HyperedgeBatch: insert_offsets",
            )
            self._members = self._ids(insert_members, "insert_members")
        else:
            self._offsets = np.zeros(1, dtype=np.int64)
            self._members = np.zeros(0, dtype=np.int64)
        if self._offsets.size == 0:
            self._offsets = np.zeros(1, dtype=np.int64)
        n = int(self._offsets.size) - 1
        self._weights: np.ndarray | None = None
        if insert_weights is not None:
            w = _dtypes.to_numpy(insert_weights, "HyperedgeBatch: insert_weights")
            if w.size != n:
                raise InvalidArgumentError(
                    f"HyperedgeBatch: {w.size} weights for {n} inserted hyperedges"
                )
            self._weights = _dtypes.checked_cast(w, INT32, "HyperedgeBatch: insert_weights")
        self._del_inc = self._pair(delete_incidences, "delete_incidences")
        self._ins_inc = self._pair(insert_incidences, "insert_incidences")

    @staticmethod
    def _ids(x: Any, what: str) -> np.ndarray:
        if x is None:
            return np.zeros(0, dtype=np.int64)
        a = _dtypes.to_numpy(x, f"HyperedgeBatch: {what}")
        if a.size and (a.dtype.kind == "b" or a.dtype.kind not in "iu"):
            raise InvalidArgumentError(
                f"HyperedgeBatch: {what} must be integers, got {a.dtype.name}"
            )
        return a

    @classmethod
    def _pair(cls, value: Any, what: str) -> tuple[np.ndarray, np.ndarray] | None:
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise InvalidArgumentError(f"HyperedgeBatch: {what} must be (hyperedges, vertices)")
        h = cls._ids(value[0], f"{what} hyperedges")
        v = cls._ids(value[1], f"{what} vertices")
        if h.size != v.size:
            raise InvalidArgumentError(f"HyperedgeBatch: {what}: the two arrays differ in length")
        return h, v

    @classmethod
    def _from_arrays(cls, parts: tuple[Any, ...]) -> HyperedgeBatch:
        """A batch from a native (delete ids, offsets, members, weights or None) tuple."""
        deletions, offsets, members, weights = parts
        return cls(deletions, None, weights, insert_offsets=offsets, insert_members=members)

    # -- properties ----------------------------------------------------------------------------
    @property
    def delete_hyperedges(self) -> np.ndarray:
        """Ids of the deleted hyperedges."""
        return self._del

    @property
    def insert_offsets(self) -> np.ndarray:
        """Offsets of the inserted hyperedges' members (int64, insertions + 1)."""
        return self._offsets

    @property
    def insert_members(self) -> np.ndarray:
        """Members of the inserted hyperedges, back to back."""
        return self._members

    @property
    def insert_weights(self) -> np.ndarray | None:
        """Weights of the inserted hyperedges (int32), or None."""
        return self._weights

    @property
    def num_insertions(self) -> int:
        """The number of inserted hyperedges."""
        return int(self._offsets.size) - 1

    @property
    def num_deletions(self) -> int:
        """The number of deleted hyperedges."""
        return int(self._del.size)

    @property
    def empty(self) -> bool:
        """Whether the batch changes nothing."""
        return (
            self.num_insertions == 0
            and self.num_deletions == 0
            and self._del_inc is None
            and self._ins_inc is None
        )

    def insertions(self) -> list[list[int]]:
        """The member lists of the inserted hyperedges."""
        o = self._offsets
        return [self._members[int(o[i]) : int(o[i + 1])].tolist() for i in range(o.size - 1)]

    def __repr__(self) -> str:
        return (
            f"dyng.HyperedgeBatch(deletions={self.num_deletions}, "
            f"insertions={self.num_insertions}, weighted={self._weights is not None})"
        )

    # -- native --------------------------------------------------------------------------------
    def _native_for(self, weighted: bool) -> Any:
        """The native batch for a hypergraph type (int32 vertices; weighted or not), built from
        the arrays' current contents."""

        def vertices(a: np.ndarray, what: str) -> np.ndarray:
            return _dtypes.checked_cast(a, INT32, f"HyperedgeBatch: {what}")

        def ids(a: np.ndarray, what: str) -> np.ndarray:
            return _dtypes.checked_cast(a, INT64, f"HyperedgeBatch: {what}")

        w: np.ndarray | None = None
        if weighted:
            if self._weights is not None:
                w = np.ascontiguousarray(self._weights, dtype=np.int32)
            elif self.num_insertions > 0:
                raise InvalidArgumentError(
                    "HyperedgeBatch: the hypergraph is weighted; give insert_weights"
                )
            else:
                w = np.zeros(0, dtype=np.int32)
        elif self._weights is not None and self.num_insertions > 0:
            raise InvalidArgumentError(
                "HyperedgeBatch: the batch carries weights, but the hypergraph is unweighted"
            )
        cls = native.HyperedgeBatchI32I32 if weighted else native.HyperedgeBatchI32U
        di = self._del_inc
        ii = self._ins_inc
        return cls(
            ids(self._del, "delete_hyperedges"),
            ids(self._offsets, "insert_offsets"),
            vertices(self._members, "insert_members"),
            w,
            None if di is None else ids(di[0], "delete_incidences"),
            None if di is None else vertices(di[1], "delete_incidences"),
            None if ii is None else ids(ii[0], "insert_incidences"),
            None if ii is None else vertices(ii[1], "insert_incidences"),
        )


class Hypergraph:
    """A dynamic hypergraph: hyperedges (sets of vertices) with 0-based int64 ids.

    Build one with :meth:`from_hyperedges`, :meth:`from_lists` or :func:`dyng.io.read_hypergraph`;
    change it with :meth:`apply` and a :class:`HyperedgeBatch`. The hypergraph has a version
    counter (+1 per batch). Vertex ids are int32; a weighted hypergraph has one int32 weight in
    [1, 2^31 - 1] per hyperedge.

    The storage belongs to the backend of the resources that built it (a host store for
    ``"sequential"`` and ``"openmp"``, the device store for ``"cuda"``); :meth:`to_backend`
    moves it. The views (:meth:`h2v`, :meth:`v2h`, :meth:`h2h`) are host copies.

    A hypergraph is not thread-safe in C++; the Python binding serializes :meth:`apply` against
    every other call on it.
    """

    __slots__ = ("_native", "_weighted", "_resources", "__weakref__")
    _native: Any
    _weighted: bool
    _resources: Resources
    _streams = _writer_state.STREAMS
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError(
            "use dyng.Hypergraph.from_hyperedges(), dyng.Hypergraph.from_lists() or "
            "dyng.io.read_hypergraph()"
        )

    @classmethod
    def _wrap(cls, handle: Any, weighted: bool, resources: Resources) -> Hypergraph:
        self = object.__new__(cls)
        self._native = handle
        self._weighted = weighted
        self._resources = resources
        _writer_state.track(self, resources)
        return self

    # -- construction --------------------------------------------------------------------------
    @classmethod
    def from_hyperedges(
        cls,
        offsets: Any,
        members: Any,
        weights: Any = None,
        *,
        num_vertices: int | None = None,
        properties: HypergraphProperties | None = None,
        storage: StorageName | None = None,
        line_graph: LineGraphName | None = None,
        id_reuse: IdReuseName | None = None,
        capacity: CapacityPolicy | CapacityPresetName | None = None,
        store_v2h: bool | None = None,
        vertex_dtype: Any = None,
        resources: Resources | None = None,
    ) -> Hypergraph:
        """Build a hypergraph from a CSR of member lists: hyperedge i has the members
        ``members[offsets[i]:offsets[i + 1]]`` and gets id i.

        Each hyperedge's members are sorted and repeated members dropped. The list may be empty.

        Args:
            offsets: num_hyperedges + 1 offsets from 0 (int64; any integer array).
            members: The members (int32 vertex ids; an int64 array needs ``vertex_dtype="int32"``,
                then every value is checked).
            weights: None (unweighted) or one integer weight in [1, 2^31 - 1] per hyperedge.
            num_vertices: The vertex count (default: the largest member + 1).
            properties: A :class:`HypergraphProperties`; the keywords below override its fields.
            storage: :attr:`HypergraphProperties.storage`.
            line_graph: :attr:`HypergraphProperties.line_graph`.
            id_reuse: :attr:`HypergraphProperties.reuse`.
            capacity: A :class:`CapacityPolicy` or a preset name (``"escher_gpu"``,
                ``"mosp_escher"``, ``"default"``).
            store_v2h: :attr:`HypergraphProperties.store_v2h`.
            vertex_dtype: ``"int32"`` to convert int64 members (checked).
            resources: The resources the hypergraph belongs to (default: the default resources).

        Returns:
            The hypergraph at version 0.

        Raises:
            InvalidArgumentError: a member out of range, an empty hyperedge, malformed offsets,
                a weight outside [1, 2^31 - 1].
            NotSupportedError: int64 members without ``vertex_dtype="int32"``, ``"slack_csr"``
                (planned for 0.4), or a CUDA backend without a CUDA build.
            CapacityError: the storage exceeds the capacity policy.
        """
        res = resolve(resources)
        o = _dtypes.checked_cast(
            _dtypes.to_numpy(offsets, "Hypergraph.from_hyperedges: offsets"),
            INT64,
            "Hypergraph.from_hyperedges: offsets",
        )
        m_raw = _dtypes.to_numpy(members, "Hypergraph.from_hyperedges: members")
        _vertex_dtype([members], vertex_dtype, "Hypergraph.from_hyperedges")
        props = make_hypergraph_properties(
            properties,
            storage=storage,
            line_graph=line_graph,
            id_reuse=id_reuse,
            capacity=capacity,
            store_v2h=store_v2h,
        )
        return cls._build(o, m_raw, weights, num_vertices, props, res)

    @classmethod
    def from_lists(
        cls,
        hyperedges: Iterable[Any],
        weights: Any = None,
        *,
        num_vertices: int | None = None,
        properties: HypergraphProperties | None = None,
        storage: StorageName | None = None,
        line_graph: LineGraphName | None = None,
        id_reuse: IdReuseName | None = None,
        capacity: CapacityPolicy | CapacityPresetName | None = None,
        store_v2h: bool | None = None,
        resources: Resources | None = None,
    ) -> Hypergraph:
        """Build a hypergraph from member lists: ``from_lists([[0, 1, 2], [1, 3]])``.

        Hyperedge i (the i-th list) gets id i. The members are checked against int32.

        Args:
            hyperedges: An iterable of member lists (lists or integer arrays).
            weights: None or one weight per hyperedge.
            num_vertices: The vertex count (default: the largest member + 1).
            properties: As for :meth:`from_hyperedges` (and the keywords that override it).
            storage: See :meth:`from_hyperedges`.
            line_graph: See :meth:`from_hyperedges`.
            id_reuse: See :meth:`from_hyperedges`.
            capacity: See :meth:`from_hyperedges`.
            store_v2h: See :meth:`from_hyperedges`.
            resources: The resources the hypergraph belongs to.

        Returns:
            The hypergraph at version 0.
        """
        res = resolve(resources)
        o, m = _csr_of_lists(hyperedges, "Hypergraph.from_lists")
        props = make_hypergraph_properties(
            properties,
            storage=storage,
            line_graph=line_graph,
            id_reuse=id_reuse,
            capacity=capacity,
            store_v2h=store_v2h,
        )
        return cls._build(o, m, weights, num_vertices, props, res)

    @classmethod
    def _build(
        cls,
        offsets: np.ndarray,
        members: np.ndarray,
        weights: Any,
        num_vertices: int | None,
        props: HypergraphProperties,
        res: Resources,
    ) -> Hypergraph:
        what = "Hypergraph"
        m = _dtypes.checked_cast(members, INT32, f"{what}: members")
        w = None
        if weights is not None:
            w = _dtypes.checked_cast(
                _dtypes.to_numpy(weights, f"{what}: weights"), INT32, f"{what}: weights"
            )
        if num_vertices is None:
            num_vertices = int(m.max(initial=-1)) + 1
        else:
            num_vertices = as_int(num_vertices, f"{what}: num_vertices")
        handle = _native_class(w is not None).from_hyperedges(
            res._native, num_vertices, offsets, m, w, props._to_native()
        )
        return cls._wrap(handle, w is not None, res)

    @classmethod
    def _from_list(
        cls, parts: tuple[Any, ...], props: HypergraphProperties, res: Resources
    ) -> Hypergraph:
        """A hypergraph from a native (num_vertices, offsets, members, weights) tuple."""
        n, offsets, members, weights = parts
        handle = _native_class(weights is not None).from_hyperedges(
            res._native, int(n), offsets, members, weights, props._to_native()
        )
        return cls._wrap(handle, weights is not None, res)

    # -- properties ----------------------------------------------------------------------------
    @property
    def num_vertices(self) -> int:
        """The vertex count; vertex ids lie in [0, num_vertices)."""
        return int(self._native.num_vertices)

    @property
    def num_hyperedges(self) -> int:
        """The number of live hyperedges."""
        return int(self._native.num_hyperedges)

    @property
    def hyperedge_id_bound(self) -> int:
        """The bound of the hyperedge ids.

        Live ids lie in [0, bound) and dead ids may exist below it. The bound can go down after a
        batch whose insertions append new rows, which drops the deleted slots it did not reuse.
        """
        return int(self._native.hyperedge_id_bound)

    @property
    def version(self) -> int:
        """0 after construction; +1 per applied batch."""
        return int(self._native.version)

    @property
    def space(self) -> str:
        """The memory space of the storage, ``"host"`` or ``"device"``."""
        return enum_name(self._native.space)

    @property
    def properties(self) -> HypergraphProperties:
        """The properties (a copy)."""
        return HypergraphProperties._from_native(self._native.properties)

    @property
    def weighted(self) -> bool:
        """Whether the hyperedges carry int32 weights."""
        return self._weighted

    @property
    def vertex_dtype(self) -> np.dtype:
        """The vertex id type (int32)."""
        return INT32

    @property
    def resources(self) -> Resources:
        """The resources the hypergraph was built with (the default of every call on it)."""
        return self._resources

    # -- views (host copies) -------------------------------------------------------------------
    def h2v(self, *, resources: Resources | None = None) -> Incidence:
        """Hyperedge -> sorted members, one row per id in [0, hyperedge_id_bound) (empty for a
        dead id); a host copy."""
        res = resolve_on(self, resources)
        return Incidence(*self._native.h2v(res._native))

    def v2h(self, *, resources: Resources | None = None) -> Incidence:
        """Vertex -> the ids of its live hyperedges, increasing; a host copy.

        Raises:
            InvalidArgumentError: the hypergraph was built with ``store_v2h=False``.
        """
        res = resolve_on(self, resources)
        return Incidence(*self._native.v2h(res._native))

    def h2h(self, *, resources: Resources | None = None) -> LineGraph:
        """The line graph: every live hyperedge to the hyperedges it overlaps; a host copy.

        Under ``line_graph="automatic"`` the first call builds it, and every batch keeps it from
        then on.

        Raises:
            InvalidArgumentError: ``line_graph="none"`` or ``store_v2h=False``.
            CapacityError: the line graph exceeds the capacity policy.
        """
        res = resolve_on(self, resources)
        return LineGraph(*self._native.h2h(res._native))

    @property
    def weights(self) -> np.ndarray | None:
        """The weights by id (0 for a dead id; a host copy), or None if unweighted."""
        w: np.ndarray | None = self._native.weights()
        return w

    # -- operations ----------------------------------------------------------------------------
    def apply(
        self, batch: HyperedgeBatch, *, resources: Resources | None = None
    ) -> HyperApplySummary:
        """Apply a batch: delete hyperedges by id and insert new ones, in the order of
        :attr:`HypergraphProperties.reuse`.

        Returns:
            What the batch did; ``inserted_ids`` holds the id of every inserted hyperedge.

        Raises:
            InvalidArgumentError: a member out of range, an empty inserted hyperedge, a weight
                outside [1, 2^31 - 1]; nothing is changed.
            NotSupportedError: incidence edits (planned for 0.4); nothing is changed.
            CapacityError: the storage exceeds its policy (the hypergraph is then unusable).
        """
        if not isinstance(batch, HyperedgeBatch):
            raise TypeError("Hypergraph.apply: batch must be a dyng.HyperedgeBatch")
        res = resolve_on(self, resources)
        nb = batch._native_for(self._weighted)
        return HyperApplySummary._from_native(self._native.apply(res._native, nb))

    def to_hyperedge_list(self, *, resources: Resources | None = None) -> HyperedgeList:
        """The live hyperedges in increasing id order (ids are not kept: hyperedge i of the list
        is the i-th live id), with the weights; a host copy."""
        res = resolve_on(self, resources)
        n, offsets, members, weights = self._native.to_hyperedge_list(res._native)
        return HyperedgeList(int(n), offsets, members, weights)

    def clone(self, resources: Resources | None = None) -> Hypergraph:
        """A deep copy (same properties, ids, rows, version and state; the deleted slots too, so
        later batches give the copy the same ids) for ``resources``."""
        res = resolve_on(self, resources)
        return Hypergraph._wrap(self._native.clone(res._native), self._weighted, res)

    def to_backend(self, resources: Resources) -> Hypergraph:
        """The hypergraph for the backend of ``resources`` (host backends versus cuda; a copy)."""
        res = resolve(resources)
        return Hypergraph._wrap(self._native.to_backend(res._native), self._weighted, res)

    def store_stats(self) -> StorageStats:
        """The storage counters (device bytes, payload, growths, compactions, kernel paths)."""
        return StorageStats._from_native(self._native.store_stats())

    def check_integrity(self, *, resources: Resources | None = None) -> None:
        """Check the storage invariants; raises :class:`~dyng.InternalError` if one is broken."""
        res = resolve_on(self, resources)
        self._native.check_integrity(res._native)

    def __copy__(self) -> Hypergraph:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Hypergraph:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.Hypergraph cannot be pickled (it may live in device memory and its ids depend "
            "on its history); use copy.deepcopy(h) / h.clone() within a process"
        )

    def __repr__(self) -> str:
        return (
            f"dyng.Hypergraph(num_vertices={self.num_vertices}, "
            f"num_hyperedges={self.num_hyperedges}, id_bound={self.hyperedge_id_bound}, "
            f"weighted={self._weighted}, version={self.version}, "
            f"backend={self._resources.backend!r})"
        )
