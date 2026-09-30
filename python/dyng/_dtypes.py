# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Dtype dispatch: which C++ instantiation an input selects (PLAN Section 5.4, rule 2).

The compiled library holds five graph types (PLAN Section 4.4.3): ``(vertex, edge offset,
weight)`` = ``(int32, int32, int32)``, ``(int32, int64, int32)``, ``(int64, int64, int32)``,
``(int32, int32, unweighted)`` and ``(int32, int64, unweighted)``.

The rules (ADR 0025):

- The vertex id type comes from the declared dtype of the id arrays: int32 selects int32 and
  int64 selects int64; narrower integer types (int8, int16, uint8, uint16) are widened to int32
  and uint32 to int64. An int64 array is **never narrowed** to int32 on its own: pass
  ``vertex_dtype="int32"`` to ask for it, and every value is then checked. Inputs without a dtype
  (Python lists, ranges) take int32 when every value fits, else int64.
- Where an instantiation is already chosen (a batch or a tree for an existing graph), ids are
  converted to the graph's type with a range check: a value that does not fit raises
  :class:`~dyng.InvalidArgumentError` instead of wrapping.
- Weights are integers in the graph's weight type (int32); floating-point or boolean weights are
  rejected.
- An unsupported combination raises :class:`~dyng.NotSupportedError` listing the supported ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._backend import native
from .errors import InvalidArgumentError, NotSupportedError

__all__ = ["GraphType", "GRAPH_TYPES", "find_graph_type", "supported_graph_types"]

INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class GraphType:
    """One compiled graph instantiation."""

    vertex: np.dtype
    edge: np.dtype
    weighted: bool

    @property
    def code(self) -> str:
        """The suffix of the native names: ``i32_i64_u``."""
        return "_".join((_code(self.vertex), _code(self.edge), "i32" if self.weighted else "u"))

    @property
    def batch_code(self) -> str:
        """The suffix of the native batch class: ``I32I32``, ``I32U``."""
        return _code(self.vertex).upper() + ("I32" if self.weighted else "U")

    @property
    def native_class(self) -> Any:
        """The native graph class."""
        name = "Graph" + _code(self.vertex).upper() + _code(self.edge).upper()
        return getattr(native, name + ("I32" if self.weighted else "U"))

    @property
    def native_batch_class(self) -> Any:
        """The native batch class of this graph type."""
        return getattr(native, "EdgeBatch" + self.batch_code)

    def describe(self) -> str:
        """``(int32, int64, unweighted)``."""
        w = "int32" if self.weighted else "unweighted"
        return f"(vertex {self.vertex.name}, edge {self.edge.name}, weight {w})"


def _code(dt: np.dtype) -> str:
    return "i32" if dt == INT32 else "i64"


GRAPH_TYPES: tuple[GraphType, ...] = (
    GraphType(INT32, INT32, True),
    GraphType(INT32, INT64, True),
    GraphType(INT64, INT64, True),
    GraphType(INT32, INT32, False),
    GraphType(INT32, INT64, False),
)


def supported_graph_types() -> str:
    """The supported combinations, one per line (for error messages)."""
    return "; ".join(t.describe() for t in GRAPH_TYPES)


def find_graph_type(vertex: np.dtype, edge: np.dtype, weighted: bool) -> GraphType:
    """The instantiation of (vertex, edge, weighted), or NotSupportedError listing the others."""
    for t in GRAPH_TYPES:
        if t.vertex == vertex and t.edge == edge and t.weighted == weighted:
            return t
    w = "int32" if weighted else "unweighted"
    raise NotSupportedError(
        f"dyng: no graph type (vertex {np.dtype(vertex).name}, edge {np.dtype(edge).name}, "
        f"weight {w}) in this build; the supported combinations are {supported_graph_types()}"
    )


def id_dtype(name: Any, what: str) -> np.dtype:
    """``"int32"`` / ``"int64"`` / a NumPy dtype as a supported id dtype."""
    try:
        dt = np.dtype(name)
    except TypeError as e:
        raise InvalidArgumentError(f"{what}: {name!r} is not a dtype") from e
    if dt not in (INT32, INT64):
        raise NotSupportedError(f"{what}: {dt.name} is not supported (int32 or int64)")
    return dt


def _declared_dtype(x: Any) -> np.dtype | None:
    """The dtype an input declares (NumPy, and anything with a NumPy-compatible dtype)."""
    dt = getattr(x, "dtype", None)
    if dt is None:
        return None
    try:
        return np.dtype(dt)
    except TypeError:
        # e.g. torch.int32: go through NumPy
        return np.asarray(x).dtype


def to_numpy(x: Any, what: str) -> np.ndarray:
    """``x`` as a 1-D NumPy array (zero-copy for NumPy arrays, DLPack and the buffer protocol)."""
    if isinstance(x, np.ndarray):
        a = x
    elif hasattr(x, "__dlpack__") and not isinstance(x, (list, tuple, range)):
        try:
            a = np.from_dlpack(x)
        except (BufferError, TypeError, RuntimeError):
            a = np.asarray(x)
    else:
        a = np.asarray(x)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1:
        raise InvalidArgumentError(f"{what}: expected a 1-D array, got shape {a.shape}")
    return a


def infer_id_dtype(arrays: list[Any], requested: Any, what: str) -> np.dtype:
    """The vertex id dtype selected by ``arrays`` (see the module description)."""
    if requested is not None:
        return id_dtype(requested, f"{what}: vertex_dtype")
    chosen: np.dtype | None = None
    undeclared: list[np.ndarray] = []
    for x in arrays:
        dt = _declared_dtype(x)
        if dt is None:
            undeclared.append(np.asarray(x))
            continue
        if dt.kind == "b" or dt.kind not in "iu":
            raise InvalidArgumentError(f"{what}: ids must be integers, got dtype {dt.name}")
        if dt.kind == "u" and dt.itemsize >= 8:
            raise NotSupportedError(
                f"{what}: uint64 ids are not supported; pass vertex_dtype='int64' (checked) or "
                "convert the arrays"
            )
        wide = INT64 if dt.itemsize > 4 or (dt.kind == "u" and dt.itemsize == 4) else INT32
        chosen = wide if chosen is None else (INT64 if INT64 in (chosen, wide) else INT32)
    if chosen is None:
        chosen = INT32
        for a in undeclared:
            if a.size and a.dtype.kind not in "iu":
                raise InvalidArgumentError(f"{what}: ids must be integers")
            if a.size and (int(a.max()) > _INT32_MAX or int(a.min()) < -_INT32_MAX - 1):
                chosen = INT64
    return chosen


def checked_cast(a: np.ndarray, dtype: np.dtype, what: str) -> np.ndarray:
    """``a`` as a C-contiguous array of ``dtype``; raises if a value does not fit (no wrapping).

    Zero-copy when ``a`` already is a C-contiguous array of ``dtype``.
    """
    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return np.ascontiguousarray(a)
    if a.dtype.kind == "b" or a.dtype.kind not in "iu":
        if a.size == 0:
            return np.zeros(0, dtype=dtype)
        raise InvalidArgumentError(f"{what}: expected integers, got dtype {a.dtype.name}")
    if a.size:
        info = np.iinfo(dtype)
        lo, hi = int(a.min()), int(a.max())
        if lo < info.min or hi > info.max:
            bad = lo if lo < info.min else hi
            raise InvalidArgumentError(
                f"{what}: the value {bad} does not fit {dtype.name} (the graph's type); "
                "nothing is narrowed silently"
            )
    return np.ascontiguousarray(a, dtype=dtype)


def weights_view(w: Any, count: int, what: str) -> np.ndarray:
    """Weights as a matrix of shape (count, K) (1-D input: K = 1), without converting them.

    Checks the shape and that the values are integers; the conversion to int32 (with a range
    check) is :func:`weights_matrix`. For NumPy input the result views the caller's array.
    """
    a = np.asarray(w) if not isinstance(w, np.ndarray) else w
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] != count:
        raise InvalidArgumentError(
            f"{what}: expected {count} weights (shape ({count},) or ({count}, K)), got shape "
            f"{a.shape}"
        )
    if a.size and (a.dtype.kind == "b" or a.dtype.kind not in "iu"):
        raise InvalidArgumentError(
            f"{what}: weights must be integers (int32 in 0.1), got dtype {a.dtype.name}"
        )
    return a


def weights_matrix(w: Any, count: int, what: str) -> np.ndarray:
    """Weights as a C-contiguous int32 matrix of shape (count, K) (1-D input: K = 1)."""
    a = weights_view(w, count, what)
    return checked_cast(a.reshape(-1), INT32, what).reshape(a.shape)
