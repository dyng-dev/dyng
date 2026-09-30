# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.EdgeBatch: a batch of edge insertions (with weights) and deletions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import _dtypes
from .errors import InvalidArgumentError

if TYPE_CHECKING:
    from .graph import Graph

__all__ = ["EdgeBatch"]


def _pair(value: Any, what: str) -> tuple[Any, Any, Any]:
    if value is None:
        return None, None, None
    if not isinstance(value, (tuple, list)) or len(value) not in (2, 3):
        raise InvalidArgumentError(f"EdgeBatch: {what} must be (src, dst) or (src, dst, weights)")
    if len(value) == 2:
        return value[0], value[1], None
    return value[0], value[1], value[2]


class EdgeBatch:
    """A batch of edge changes: insertions (u, v, weights) and deletions (u, v).

    How a batch changes a graph is decided by the graph's batch semantics
    (:class:`dyng.BatchSemantics`; by default deletions first, then insertions, an insertion of an
    existing edge overwriting its weights).

    The arrays are kept as given (no copy when they are C-contiguous arrays of the graph's id and
    weight types) and converted, with range checks, to the types of the graph a batch is used
    with. The batch reads its arrays at every use: do not change them while a call runs.

    Args:
        insert: ``(src, dst)`` or ``(src, dst, weights)``; weights of shape (n,) or (n, K).
        delete: ``(src, dst)``.
        insert_vertices: Vertices to insert (0.3; applying such a batch raises
            :class:`~dyng.NotSupportedError` in 0.1).
        vertex_labels: Labels of the inserted vertices (int8; 0.3).
        delete_vertices: Vertices to delete (0.3).

    Examples:
        >>> import numpy as np, dyng
        >>> b = dyng.EdgeBatch(insert=([0, 5], [7, 9], [3, 2]), delete=([1], [2]))
        >>> b.num_insertions, b.num_deletions, b.num_weights
        (2, 1, 1)
    """

    __slots__ = (
        "_ins_src",
        "_ins_dst",
        "_ins_w",
        "_del_src",
        "_del_dst",
        "_ins_v",
        "_labels",
        "_del_v",
        "_cache",
    )

    def __init__(
        self,
        insert: Any = None,
        delete: Any = None,
        *,
        insert_vertices: Any = None,
        vertex_labels: Any = None,
        delete_vertices: Any = None,
    ) -> None:
        s, d, w = _pair(insert, "insert")
        ds, dd, extra = _pair(delete, "delete")
        if extra is not None:
            raise InvalidArgumentError("EdgeBatch: delete must be (src, dst)")
        self._ins_src = self._ids(s, "insert src")
        self._ins_dst = self._ids(d, "insert dst")
        if self._ins_src.size != self._ins_dst.size:
            raise InvalidArgumentError("EdgeBatch: the insert src and dst differ in length")
        self._ins_w: np.ndarray | None = None
        if w is not None:
            self._ins_w = _dtypes.weights_matrix(w, int(self._ins_src.size), "EdgeBatch: weights")
        self._del_src = self._ids(ds, "delete src")
        self._del_dst = self._ids(dd, "delete dst")
        if self._del_src.size != self._del_dst.size:
            raise InvalidArgumentError("EdgeBatch: the delete src and dst differ in length")
        self._ins_v = (
            None if insert_vertices is None else self._ids(insert_vertices, "insert_vertices")
        )
        self._labels = (
            None
            if vertex_labels is None
            else _dtypes.checked_cast(
                _dtypes.to_numpy(vertex_labels, "vertex_labels"), np.dtype(np.int8), "vertex_labels"
            )
        )
        self._del_v = (
            None if delete_vertices is None else self._ids(delete_vertices, "delete_vertices")
        )
        self._cache: dict[tuple[str, int], Any] = {}

    @staticmethod
    def _ids(x: Any, what: str) -> np.ndarray:
        if x is None:
            return np.zeros(0, dtype=np.int32)
        a = _dtypes.to_numpy(x, f"EdgeBatch: {what}")
        if a.size and (a.dtype.kind == "b" or a.dtype.kind not in "iu"):
            raise InvalidArgumentError(f"EdgeBatch: {what} must be integers, got {a.dtype.name}")
        return a

    @classmethod
    def _from_arrays(cls, parts: tuple[Any, ...]) -> EdgeBatch:
        """A batch from a native reader's or generator's (src, dst, w, dsrc, ddst, K) tuple."""
        ins_src, ins_dst, w, del_src, del_dst, _k = parts
        insert = (ins_src, ins_dst) if w is None else (ins_src, ins_dst, w)
        return cls(insert=insert, delete=(del_src, del_dst))

    # -- properties ----------------------------------------------------------------------------
    @property
    def insert_src(self) -> np.ndarray:
        """Sources of the insertions."""
        return self._ins_src

    @property
    def insert_dst(self) -> np.ndarray:
        """Destinations of the insertions."""
        return self._ins_dst

    @property
    def insert_weights(self) -> np.ndarray | None:
        """Weights of the insertions, shape (n, K) int32, or None."""
        return self._ins_w

    @property
    def delete_src(self) -> np.ndarray:
        """Sources of the deletions."""
        return self._del_src

    @property
    def delete_dst(self) -> np.ndarray:
        """Destinations of the deletions."""
        return self._del_dst

    @property
    def num_insertions(self) -> int:
        """The number of edge insertions."""
        return int(self._ins_src.size)

    @property
    def num_deletions(self) -> int:
        """The number of edge deletions."""
        return int(self._del_src.size)

    @property
    def num_weights(self) -> int:
        """Weights per insertion (0 without weights)."""
        return 0 if self._ins_w is None else int(self._ins_w.shape[1])

    @property
    def empty(self) -> bool:
        """Whether the batch changes nothing."""
        return (
            self.num_insertions == 0
            and self.num_deletions == 0
            and (self._ins_v is None or self._ins_v.size == 0)
            and (self._del_v is None or self._del_v.size == 0)
        )

    def __repr__(self) -> str:
        return (
            f"dyng.EdgeBatch(insertions={self.num_insertions}, deletions={self.num_deletions}, "
            f"num_weights={self.num_weights})"
        )

    # -- native --------------------------------------------------------------------------------
    def _native_for(self, graph: Graph) -> Any:
        """The native batch for the id and weight types of ``graph`` (cached per type)."""
        return self._native_for_type(graph._type, graph.num_weights)

    def _native_for_type(self, gtype: _dtypes.GraphType, k_graph: int) -> Any:
        """The native batch for a graph type with ``k_graph`` weight columns (cached)."""
        key = (gtype.batch_code, k_graph)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        v = gtype.vertex

        def ids(a: np.ndarray, what: str) -> np.ndarray:
            return _dtypes.checked_cast(a, v, f"EdgeBatch: {what}")

        w: np.ndarray | None = None
        k = self.num_weights
        if gtype.weighted:
            if self._ins_w is not None:
                w = np.ascontiguousarray(self._ins_w).reshape(-1)
            elif self.num_insertions == 0:
                k = k_graph  # no insertions: nothing to weigh
        elif self._ins_w is not None and self.num_insertions > 0:
            raise InvalidArgumentError(
                "EdgeBatch: the batch carries weights, but the graph is unweighted"
            )
        else:
            k = 0
        native_batch = gtype.native_batch_class(
            ids(self._ins_src, "insert src"),
            ids(self._ins_dst, "insert dst"),
            w,
            ids(self._del_src, "delete src"),
            ids(self._del_dst, "delete dst"),
            k,
            None if self._ins_v is None else ids(self._ins_v, "insert_vertices"),
            self._labels,
            None if self._del_v is None else ids(self._del_v, "delete_vertices"),
        )
        self._cache[key] = native_batch
        return native_batch
