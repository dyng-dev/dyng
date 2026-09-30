# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Array: a zero-copy, read-only view of an array owned by a result (PLAN Section 5.4).

The result arrays of the library (``Result.distances``, ``Result.parents``, ``Result.counts``) are
returned as :class:`Array`. An Array does not copy: ``np.asarray(a)`` (through
``__array_interface__``), ``np.from_dlpack(a)`` and ``torch.from_dlpack(a)`` (through
``__dlpack__``) view the library's memory, and the view keeps the result alive.

**Lifetime** (as in C++, where these views are "valid until the next update"): an Array belongs to
one state of its result. Once the result is updated (``update``, ``dyng.update``), using the
Array raises :class:`~dyng.StaleResultError`; read the property again for the new state. Views
that NumPy or another library already made follow the C++ rule: they show the result's memory,
which an update changes in place (or reallocates when the graph grows), so take a copy
(:meth:`Array.to_numpy`, the default) of anything that must outlive the next update.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .errors import StaleResultError

__all__ = ["Array"]


class Array:
    """A read-only view of a result's array, with DLPack and the NumPy array interface.

    Attributes are those of a 1-D array: :attr:`shape`, :attr:`dtype`, :attr:`size`,
    :attr:`ndim`, ``len(a)``, iteration and indexing (which read through a copy-free NumPy
    view).
    """

    __slots__ = ("_nd", "_view", "_is_current", "_what", "__weakref__")

    def __init__(self, nd: Any, is_current: Callable[[], bool], what: str) -> None:
        self._nd = nd  # nanobind's array-API object: owns a reference to the result
        self._view: np.ndarray | None = None
        self._is_current = is_current
        self._what = what

    # -- state ---------------------------------------------------------------------------------
    def _check(self) -> None:
        if not self._is_current():
            raise StaleResultError(
                f"dyng.Array ({self._what}): the result was updated since this array was read; "
                "read the property of the result again"
            )

    def _numpy_view(self) -> np.ndarray:
        self._check()
        if self._view is None:
            v = np.from_dlpack(self._nd)
            v.flags.writeable = False
            self._view = v
        return self._view

    @property
    def is_current(self) -> bool:
        """Whether the result is still in the state this array was read from."""
        return bool(self._is_current())

    # -- array attributes ----------------------------------------------------------------------
    @property
    def dtype(self) -> np.dtype:
        """The element type."""
        return self._numpy_view().dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape (1-D)."""
        return self._numpy_view().shape

    @property
    def ndim(self) -> int:
        """1."""
        return 1

    @property
    def size(self) -> int:
        """The number of elements."""
        return int(self._numpy_view().size)

    @property
    def device(self) -> str:
        """``"cpu"`` for host memory, ``"cuda:<n>"`` for device memory."""
        kind, dev = self.__dlpack_device__()
        return "cpu" if kind in (1, 3) else f"cuda:{dev}"

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_numpy(copy=True).tolist())

    def __getitem__(self, index: Any) -> Any:
        out = self._numpy_view()[index]
        return out.copy() if isinstance(out, np.ndarray) else out.item()

    def __eq__(self, other: object) -> Any:
        return np.asarray(self._numpy_view()) == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        if not self._is_current():
            return f"dyng.Array({self._what}, stale)"
        v = self._numpy_view()
        return f"dyng.Array({np.array2string(v, threshold=20)}, dtype={v.dtype.name})"

    # -- interop -------------------------------------------------------------------------------
    @property
    def __array_interface__(self) -> dict[str, Any]:
        """The NumPy array interface (host memory; read-only, zero-copy)."""
        return self._numpy_view().__array_interface__

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        v = self._numpy_view()
        if copy:
            return np.array(v, dtype=dtype, copy=True)
        if dtype is not None and np.dtype(dtype) != v.dtype:
            if copy is False:
                raise ValueError("dyng.Array: a dtype conversion needs a copy")
            return v.astype(dtype)
        return v

    def __dlpack__(self, **kwargs: Any) -> Any:
        """Export through DLPack (zero-copy; ``stream``, ``max_version``, ``dl_device`` and
        ``copy`` as in the Python array API)."""
        self._check()
        return self._nd.__dlpack__(**kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        """The DLPack device: (1, 0) for host memory, (2, n) for CUDA device n."""
        return tuple(self._nd.__dlpack_device__())  # type: ignore[return-value]

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """The elements as a NumPy array.

        Args:
            copy: True (the default): an independent copy, safe to keep across updates.
                False: a read-only view of the library's memory (valid until the next update of
                the result).
        """
        v = self._numpy_view()
        return v.copy() if copy else v

    def tolist(self) -> list[Any]:
        """The elements as a Python list."""
        return self._numpy_view().tolist()

    def to_torch(self) -> Any:
        """A PyTorch tensor viewing the elements (zero-copy through DLPack; needs torch)."""
        import torch  # lazy: an optional dependency

        self._check()
        return torch.from_dlpack(self._nd)

    def to_cupy(self) -> Any:
        """A CuPy array of the elements (zero-copy for device memory; host memory is copied to
        the current device; needs cupy)."""
        import cupy  # lazy: an optional dependency

        self._check()
        kind, _ = self.__dlpack_device__()
        if kind == 2:
            return cupy.from_dlpack(self._nd)
        return cupy.asarray(self._numpy_view())
