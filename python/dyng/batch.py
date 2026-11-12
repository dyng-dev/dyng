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

    The arrays are kept as given, not copied, and converted, with range checks, to the id and
    weight types of the graph at **every** use (no copy when they already are C-contiguous
    arrays of those types). A batch therefore always applies the current contents of its arrays,
    whatever their dtypes: refilling the arrays between two uses applies the new values. Do not
    change them while a call that uses the batch runs.

    Weights are integers (int32 graphs) or, since 0.4, floating point (the float32 graphs of
    :mod:`dyng.label_propagation`; ADR 0108, AD14): float32 weights are used as they are, and
    float64 weights only when every value is exactly a float32, unless ``weight_dtype="float32"``
    asks for the conversion (then checked: finite and within float32's range). Nothing is
    narrowed silently.

    Vertex batches (0.4): ``insert_vertices`` (with optional ``vertex_labels``, -1 for none, else
    the class 0 or 1) and ``delete_vertices`` are applied by graphs with slot rows
    (:meth:`dyng.Graph.with_capacity`, ``semantics="net_effect"``); a graph with compact rows
    raises :class:`~dyng.NotSupportedError` for them.

    Args:
        insert: ``(src, dst)`` or ``(src, dst, weights)``; weights of shape (n,) or (n, K).
        delete: ``(src, dst)``.
        insert_vertices: Vertices to insert (0.4; slot rows only).
        vertex_labels: Labels of the inserted vertices (int8: -1 none, 0 or 1; 0.4).
        delete_vertices: Vertices to delete (0.4; slot rows only).
        weight_dtype: ``"float32"``: convert floating-point weights to float32 now (checked);
            None (the default): keep the weights as given (0.4).

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
    )
    _ins_w: np.ndarray | None

    def __init__(
        self,
        insert: Any = None,
        delete: Any = None,
        *,
        insert_vertices: Any = None,
        vertex_labels: Any = None,
        delete_vertices: Any = None,
        weight_dtype: Any = None,
    ) -> None:
        s, d, w = _pair(insert, "insert")
        ds, dd, extra = _pair(delete, "delete")
        if extra is not None:
            raise InvalidArgumentError("EdgeBatch: delete must be (src, dst)")
        self._ins_src = self._ids(s, "insert src")
        self._ins_dst = self._ids(d, "insert dst")
        if self._ins_src.size != self._ins_dst.size:
            raise InvalidArgumentError("EdgeBatch: the insert src and dst differ in length")
        self._ins_w = None
        if weight_dtype is not None:
            wdt = _dtypes.weight_dtype_of(weight_dtype, "EdgeBatch: weight_dtype")
            if wdt != _dtypes.FLOAT32:
                raise InvalidArgumentError(
                    "EdgeBatch: weight_dtype is 'float32' or None (integer weights keep their "
                    "rule: int32, checked)"
                )
        if w is not None:
            n = int(self._ins_src.size)
            view = _dtypes.weights_view(w, n, "EdgeBatch: weights", allow_float=True)
            if weight_dtype is not None:
                # the conversion the caller asked for, checked (AD14)
                view = _dtypes.float_weights_matrix(view, n, "EdgeBatch: weights", explicit=True)
            elif view.size and view.dtype.kind == "f":
                _dtypes.float_weights_matrix(view, n, "EdgeBatch: weights", explicit=True)  # finite
            else:
                _dtypes.weights_matrix(view, n, "EdgeBatch: weights")  # early range check
            self._ins_w = view
        self._del_src = self._ids(ds, "delete src")
        self._del_dst = self._ids(dd, "delete dst")
        if self._del_src.size != self._del_dst.size:
            raise InvalidArgumentError("EdgeBatch: the delete src and dst differ in length")
        self._ins_v = (
            None if insert_vertices is None else self._ids(insert_vertices, "insert_vertices")
        )
        self._labels = (
            None if vertex_labels is None else _dtypes.to_numpy(vertex_labels, "vertex_labels")
        )
        if self._labels is not None:
            _dtypes.checked_cast(self._labels, np.dtype(np.int8), "vertex_labels")
        self._del_v = (
            None if delete_vertices is None else self._ids(delete_vertices, "delete_vertices")
        )

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
        """A batch from a native reader's or generator's (src, dst, w, dsrc, ddst, K) tuple, or
        its 9-tuple with (insert_vertices, vertex_labels, delete_vertices) (None: no vertex
        operations; 0.4)."""
        ins_src, ins_dst, w, del_src, del_dst, k = parts[:6]
        if k == 0:
            w = None
        insert = (ins_src, ins_dst) if w is None else (ins_src, ins_dst, w)
        if len(parts) == 9 and parts[6] is not None:
            return cls(
                insert=insert,
                delete=(del_src, del_dst),
                insert_vertices=parts[6],
                vertex_labels=parts[7],
                delete_vertices=parts[8],
            )
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
        """Weights of the insertions, shape (n, K), or None: int32 for integer weights (a view
        when given as a C-contiguous int32 array); floating-point weights (0.4) as given (a
        float32 array after ``weight_dtype="float32"``)."""
        if self._ins_w is None:
            return None
        if self._ins_w.dtype.kind == "f":
            return self._ins_w
        return _dtypes.weights_matrix(self._ins_w, self.num_insertions, "EdgeBatch: weights")

    @property
    def insert_vertices(self) -> np.ndarray | None:
        """The vertices the batch inserts (0.4), or None."""
        return self._ins_v

    @property
    def vertex_labels(self) -> np.ndarray | None:
        """The labels of the inserted vertices (int8: -1 none, 0 or 1; 0.4), or None."""
        if self._labels is None:
            return None
        return _dtypes.checked_cast(self._labels, np.dtype(np.int8), "vertex_labels")

    @property
    def delete_vertices(self) -> np.ndarray | None:
        """The vertices the batch deletes (0.4), or None."""
        return self._del_v

    def _has_vertex_operations(self) -> bool:
        """Whether the batch inserts or deletes vertices (0.4)."""
        return bool(
            (self._ins_v is not None and self._ins_v.size)
            or (self._del_v is not None and self._del_v.size)
        )

    def _has_float_weights(self) -> bool:
        """Whether the batch's weights are floating point (0.4: for the float32 graphs)."""
        return self._ins_w is not None and self._ins_w.dtype.kind == "f"

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
        extra = ""
        if self._has_vertex_operations():
            ni = 0 if self._ins_v is None else int(self._ins_v.size)
            nd = 0 if self._del_v is None else int(self._del_v.size)
            extra = f", inserted_vertices={ni}, deleted_vertices={nd}"
        return (
            f"dyng.EdgeBatch(insertions={self.num_insertions}, deletions={self.num_deletions}, "
            f"num_weights={self.num_weights}{extra})"
        )

    # -- native --------------------------------------------------------------------------------
    def _native_for(self, graph: Graph) -> Any:
        """The native batch for the id and weight types of ``graph``."""
        return self._native_for_type(graph._type, graph.num_weights)

    def _native_for_type(self, gtype: _dtypes.GraphType, k_graph: int) -> Any:
        """The native batch for a graph type with ``k_graph`` weight columns.

        Built from the arrays' current contents at every call (never cached): a batch applies
        the same values whether or not its arrays need a conversion.
        """
        v = gtype.vertex

        def ids(a: np.ndarray, what: str) -> np.ndarray:
            return _dtypes.checked_cast(a, v, f"EdgeBatch: {what}")

        w: np.ndarray | None = None
        k = self.num_weights
        if gtype.is_float:
            if self._ins_w is not None:
                w = _dtypes.float_weights_matrix(
                    self._ins_w, self.num_insertions, "EdgeBatch: weights"
                ).reshape(-1)
            elif self.num_insertions == 0:
                k = k_graph  # no insertions: nothing to weigh
        elif gtype.weighted:
            if self._ins_w is not None:
                w = _dtypes.weights_matrix(
                    self._ins_w, self.num_insertions, "EdgeBatch: weights"
                ).reshape(-1)
            elif self.num_insertions == 0:
                k = k_graph  # no insertions: nothing to weigh
        elif self._ins_w is not None and self.num_insertions > 0:
            raise InvalidArgumentError(
                "EdgeBatch: the batch carries weights, but the graph is unweighted"
            )
        else:
            k = 0
        labels = (
            None
            if self._labels is None
            else _dtypes.checked_cast(self._labels, np.dtype(np.int8), "vertex_labels")
        )
        return gtype.native_batch_class(
            ids(self._ins_src, "insert src"),
            ids(self._ins_dst, "insert dst"),
            w,
            ids(self._del_src, "delete src"),
            ids(self._del_dst, "delete dst"),
            k,
            None if self._ins_v is None else ids(self._ins_v, "insert_vertices"),
            labels,
            None if self._del_v is None else ids(self._del_v, "delete_vertices"),
        )
