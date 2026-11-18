# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Dtype dispatch: which C++ instantiation an input selects (PLAN Section 5.4, rule 2).

The compiled library holds seven graph types (PLAN Section 4.4.3): ``(vertex, edge offset,
weight)`` = ``(int32, int32, int32)``, ``(int32, int64, int32)``, ``(int64, int64, int32)``,
``(int32, int32, unweighted)``, ``(int32, int64, unweighted)`` and, since 0.4,
``(int32, int32, float32)`` and ``(int32, int64, float32)`` (label_propagation's similarity
graphs).

The rules (ADR 0011, item 3):

- The vertex id type comes from the declared dtype of the id arrays: int32 selects int32 and
  int64 selects int64; narrower integer types (int8, int16, uint8, uint16) are widened to int32
  and uint32 to int64. An int64 array is **never narrowed** to int32 on its own: pass
  ``vertex_dtype="int32"`` to ask for it, and every value is then checked. Objects with the
  buffer protocol (``array.array``, ``memoryview``) declare their element format the same way.
  Inputs without a dtype (Python lists, ranges) take int32 when every value fits, else int64.
- Where an instantiation is already chosen (a batch or a tree for an existing graph), ids are
  converted to the graph's type with a range check: a value that does not fit raises
  :class:`~dyng.InvalidArgumentError` instead of wrapping.
- Weights select the weight type (0.4, ADR 0108; decided by the author on 2026-10-28, AD14):
  integer weights select int32 (checked, as before); **float32** weights select the float
  graphs; float64 weights are accepted only with an explicit ``weight_dtype="float32"`` and are
  then checked (finite and within float32's range; rounded to the nearest float32, the
  conversion the caller asked for), never narrowed silently. Boolean weights are rejected.
  Where the graph is already chosen (a batch applied to a float graph), float32 weights are used
  as they are and float64 or integer weights only when every value is exactly a float32;
  otherwise :class:`~dyng.InvalidArgumentError` asks for ``astype(numpy.float32)``.
- An unsupported combination raises :class:`~dyng.NotSupportedError` listing the supported ones.

Inputs in CUDA device memory (CuPy, PyTorch, JAX, any DLPack producer; a :class:`dyng.Array` of
the CUDA backend) are copied to the host once, ordered after the producer's stream through
``__dlpack__(stream=...)`` (ADR 0031): graphs and batches are built from host arrays in this
release. A device input needs a CUDA plugin's module and must be C-contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._backend import _dyng_version, native, plugin_install_hint
from .errors import InvalidArgumentError, NotSupportedError

__all__ = ["GraphType", "GRAPH_TYPES", "find_graph_type", "supported_graph_types"]

INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
INT8 = np.dtype(np.int8)
FLOAT32 = np.dtype(np.float32)
_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class GraphType:
    """One compiled graph instantiation.

    ``weight`` is the weight dtype: int32, float32 (0.4), or None for the unweighted graphs.
    """

    vertex: np.dtype
    edge: np.dtype
    weight: np.dtype | None

    @property
    def weighted(self) -> bool:
        """Whether the weight type is int32 (the meaning of :attr:`dyng.Graph.weighted`)."""
        return self.weight == INT32

    @property
    def has_weights(self) -> bool:
        """Whether the graph has weight columns (int32 or float32)."""
        return self.weight is not None

    @property
    def is_float(self) -> bool:
        """Whether the weight type is float32 (label_propagation's graphs; 0.4)."""
        return self.weight == FLOAT32

    @property
    def _wcode(self) -> str:
        return "u" if self.weight is None else ("f32" if self.weight == FLOAT32 else "i32")

    @property
    def code(self) -> str:
        """The suffix of the native names: ``i32_i64_u``, ``i32_i32_f32``."""
        return "_".join((_code(self.vertex), _code(self.edge), self._wcode))

    @property
    def batch_code(self) -> str:
        """The suffix of the native batch class: ``I32I32``, ``I32U``, ``I32F32``."""
        return _code(self.vertex).upper() + self._wcode.upper()

    @property
    def native_class(self) -> Any:
        """The native graph class."""
        name = "Graph" + _code(self.vertex).upper() + _code(self.edge).upper()
        return getattr(native, name + self._wcode.upper())

    @property
    def native_batch_class(self) -> Any:
        """The native batch class of this graph type."""
        return getattr(native, "EdgeBatch" + self.batch_code)

    def describe(self) -> str:
        """``(int32, int64, unweighted)``."""
        w = "unweighted" if self.weight is None else self.weight.name
        return f"(vertex {self.vertex.name}, edge {self.edge.name}, weight {w})"


def _code(dt: np.dtype) -> str:
    return "i32" if dt == INT32 else "i64"


GRAPH_TYPES: tuple[GraphType, ...] = (
    GraphType(INT32, INT32, INT32),
    GraphType(INT32, INT64, INT32),
    GraphType(INT64, INT64, INT32),
    GraphType(INT32, INT32, None),
    GraphType(INT32, INT64, None),
    GraphType(INT32, INT32, FLOAT32),
    GraphType(INT32, INT64, FLOAT32),
)


def supported_graph_types() -> str:
    """The supported combinations, one per line (for error messages)."""
    return "; ".join(t.describe() for t in GRAPH_TYPES)


def find_graph_type(vertex: np.dtype, edge: np.dtype, weight: bool | np.dtype | None) -> GraphType:
    """The instantiation of (vertex, edge, weight), or NotSupportedError listing the others.

    ``weight``: a weight dtype (int32 or float32), None for unweighted, or a bool (True: int32).
    """
    if isinstance(weight, bool):
        wdt: np.dtype | None = INT32 if weight else None
    else:
        wdt = None if weight is None else np.dtype(weight)
    for t in GRAPH_TYPES:
        if t.vertex == vertex and t.edge == edge and t.weight == wdt:
            return t
    w = "unweighted" if wdt is None else wdt.name
    raise NotSupportedError(
        f"dyng: no graph type (vertex {np.dtype(vertex).name}, edge {np.dtype(edge).name}, "
        f"weight {w}) in this build; the supported combinations are {supported_graph_types()}"
    )


def float32_value(x: float) -> float:
    """The shortest decimal that is the float32 nearest to ``x``, as a Python float (a float32
    field read back from C++: ``float32_value(1e-4) == 1e-4``, and passing it back gives the same
    float32)."""
    return float(str(np.float32(x)))


def weight_dtype_of(name: Any, what: str) -> np.dtype:
    """``"int32"`` / ``"float32"`` / a NumPy dtype as a supported weight dtype."""
    try:
        dt = np.dtype(name)
    except TypeError as e:
        raise InvalidArgumentError(f"{what}: {name!r} is not a dtype") from e
    if dt not in (INT32, FLOAT32):
        raise NotSupportedError(f"{what}: {dt.name} is not supported (int32 or float32)")
    out: np.dtype = dt
    return out


def id_dtype(name: Any, what: str) -> np.dtype:
    """``"int32"`` / ``"int64"`` / a NumPy dtype as a supported id dtype."""
    try:
        dt = np.dtype(name)
    except TypeError as e:
        raise InvalidArgumentError(f"{what}: {name!r} is not a dtype") from e
    if dt not in (INT32, INT64):
        raise NotSupportedError(f"{what}: {dt.name} is not supported (int32 or int64)")
    out: np.dtype = dt
    return out


def _device_of(x: Any) -> tuple[int, int] | None:
    """The DLPack device of ``x`` when it is in CUDA device memory, else None."""
    if isinstance(x, np.ndarray | list | tuple | range):
        return None
    probe = getattr(x, "__dlpack_device__", None)
    if probe is None:
        return None
    try:
        kind, dev = probe()
    except Exception:
        return None
    from .array import is_device

    return (int(kind), int(dev)) if is_device(int(kind)) else None


def device_to_host(x: Any, what: str) -> np.ndarray | None:
    """A host copy of ``x`` if it is in CUDA device memory (a dyng.Array: its host copy), else
    None. The copy is ordered after the producer's stream (``__dlpack__(stream=...)``)."""
    from .array import Array, cuda_resources

    if isinstance(x, Array):
        return x.to_numpy(copy=None)
    device = _device_of(x)
    if device is None:
        return None
    if not native.build_config["cuda"]:
        hint = plugin_install_hint(_dyng_version(), "cu13")
        raise NotSupportedError(
            f"{what}: the input is in CUDA device memory (device {device[1]}), but the active "
            f"native module is the CPU module dyng._core; install a CUDA plugin ({hint}) or "
            "pass a host array"
        )
    res = cuda_resources(device[1])
    stream = int(res.stream)  # 1: the legacy default stream (DLPack's value for it)
    try:
        capsule = x.__dlpack__(stream=stream, max_version=(1, 0))
    except TypeError:  # a producer older than the 2023 array API
        capsule = x.__dlpack__(stream=stream)
    out: np.ndarray = native.array_to_host(res, capsule)
    return out


def _declared_dtype(x: Any) -> np.dtype | None:
    """The dtype an input declares: its ``dtype`` (NumPy, and anything with a NumPy-compatible
    dtype), or the element format of an object with the buffer protocol (``array.array``,
    ``memoryview``); None for Python sequences, which declare nothing."""
    dt = getattr(x, "dtype", None)
    if dt is None:
        if isinstance(x, list | tuple | range):
            return None
        try:
            view = memoryview(x)
        except TypeError:
            return None
        return np.dtype(np.asarray(view).dtype)
    try:
        declared: np.dtype = np.dtype(dt)
        return declared
    except TypeError:
        pass
    try:
        # e.g. torch.int32 (str: "torch.int32"); without reading the elements
        named: np.dtype = np.dtype(str(dt).rsplit(".", 1)[-1])
        return named
    except TypeError:
        pass
    host = device_to_host(x, "dtype")
    return np.dtype((host if host is not None else np.asarray(x)).dtype)


def to_numpy(x: Any, what: str) -> np.ndarray:
    """``x`` as a 1-D NumPy array (zero-copy for NumPy arrays, DLPack and the buffer protocol in
    host memory; one copy of device memory)."""
    host = None if isinstance(x, np.ndarray) else device_to_host(x, what)
    if isinstance(x, np.ndarray):
        a = x
    elif host is not None:
        a = host
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


def weights_view(w: Any, count: int, what: str, *, allow_float: bool = False) -> np.ndarray:
    """Weights as a matrix of shape (count, K) (1-D input: K = 1), without converting them.

    Checks the shape and that the values are integers (with ``allow_float``: integers or
    floating point, 0.4); the conversions are :func:`weights_matrix` (int32, with a range check)
    and :func:`float_weights_matrix` (float32). For NumPy input the result views the caller's
    array.
    """
    if isinstance(w, np.ndarray):
        a = w
    else:
        host = device_to_host(w, what)
        a = host if host is not None else np.asarray(w)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] != count:
        raise InvalidArgumentError(
            f"{what}: expected {count} weights (shape ({count},) or ({count}, K)), got shape "
            f"{a.shape}"
        )
    kinds = "iuf" if allow_float else "iu"
    if a.size and (a.dtype.kind == "b" or a.dtype.kind not in kinds):
        if allow_float:
            raise InvalidArgumentError(
                f"{what}: weights must be integers or floating point, got dtype {a.dtype.name}"
            )
        raise InvalidArgumentError(
            f"{what}: weights must be integers (int32; float32 weights select the float graphs "
            f"of 0.4), got dtype {a.dtype.name}"
        )
    return a


def weights_matrix(w: Any, count: int, what: str) -> np.ndarray:
    """Weights as a C-contiguous int32 matrix of shape (count, K) (1-D input: K = 1)."""
    a = weights_view(w, count, what, allow_float=True)
    if a.size and a.dtype.kind == "f":
        raise InvalidArgumentError(
            f"{what}: the graph's weights are int32, got dtype {a.dtype.name} (float32 weights "
            "select the float graphs of label_propagation; nothing is narrowed silently)"
        )
    return checked_cast(a.reshape(-1), INT32, what).reshape(a.shape)


def is_float_weights(w: Any) -> bool:
    """Whether a weight input declares a floating-point dtype (it then selects a float graph)."""
    dt = _declared_dtype(w)
    if dt is None:
        a = np.asarray(w)
        return a.size > 0 and a.dtype.kind == "f"
    return dt.kind == "f"


def to_float32(a: np.ndarray, what: str, *, explicit: bool) -> np.ndarray:
    """``a`` as a C-contiguous float32 array (AD14, ADR 0108).

    float32 input is used as it is (zero-copy when C-contiguous). Other input: with ``explicit``
    (the caller passed ``weight_dtype="float32"``) every value must be finite and within
    float32's range, and is rounded to the nearest float32; without it every value must be
    exactly a float32 (no rounding), else :class:`~dyng.InvalidArgumentError`. NaN and
    infinities are always rejected.
    """
    if a.dtype == FLOAT32:
        out = np.ascontiguousarray(a)
        if out.size and not bool(np.isfinite(out).all()):
            raise InvalidArgumentError(f"{what}: a weight is NaN or infinite")
        return out
    if a.size == 0:
        return np.zeros(a.shape, dtype=FLOAT32)
    if a.dtype.kind == "b" or a.dtype.kind not in "iuf":
        raise InvalidArgumentError(f"{what}: expected numbers, got dtype {a.dtype.name}")
    wide = a.astype(np.float64)
    if a.dtype.kind == "f" and not bool(np.isfinite(wide).all()):
        raise InvalidArgumentError(f"{what}: a weight is NaN or infinite")
    if bool((np.abs(wide) > _FLOAT32_MAX).any()):
        raise InvalidArgumentError(
            f"{what}: a weight is outside float32's range (|w| > {_FLOAT32_MAX:.6g})"
        )
    narrow = wide.astype(FLOAT32)
    if not explicit and not np.array_equal(narrow.astype(np.float64), wide):
        raise InvalidArgumentError(
            f"{what}: {a.dtype.name} weights that are not exactly float32 values; the graph's "
            "weights are float32 and nothing is narrowed silently: pass weights.astype("
            "numpy.float32) (or weight_dtype='float32' where a function takes it)"
        )
    return np.ascontiguousarray(narrow)


def float_weights_matrix(w: Any, count: int, what: str, *, explicit: bool = False) -> np.ndarray:
    """Weights as a C-contiguous float32 matrix of shape (count, K) (see :func:`to_float32`)."""
    a = weights_view(w, count, what, allow_float=True)
    return to_float32(a.reshape(-1), what, explicit=explicit).reshape(a.shape)
