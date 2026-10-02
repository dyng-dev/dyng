# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Array: a zero-copy, read-only view of an array owned by a result (PLAN Section 5.4).

The result arrays of the library (``Result.distances``, ``Result.parents``, ``Result.counts``,
``dyng.mosp.Result.path_costs``) are returned as :class:`Array`. An Array does not copy:
``np.asarray(a)`` (through ``__array_interface__``), ``np.from_dlpack(a)`` and
``torch.from_dlpack(a)`` (through ``__dlpack__``) view the library's memory, and the view keeps
the result alive.

**Lifetime.** An Array belongs to one state of its result. Once the result is updated
(``update``, ``dyng.update``), using the Array raises :class:`~dyng.StaleResultError`; read the
property again for the new state. Views that NumPy or another library made from an Array
(``np.asarray``, ``np.from_dlpack``, ``to_numpy(copy=False)``, ...) keep showing the state they
were made from, unchanged: while any such view (or the Array itself) is alive, an update first
copies the result's state and changes the copy (copy-on-write), so exported memory is never
changed or freed under a view. The price is one copy of the result's arrays per update while a
view is alive; drop views you no longer need (``del``) to update in place.

Exports are read-only: DLPack consumers that ask for DLPack >= 1.0 get a zero-copy capsule with
the read-only flag, and a consumer that asks for the unversioned (legacy) capsule, which cannot
carry the flag, gets a copy. PyTorch and CuPy have no read-only arrays: tensors from
:meth:`Array.to_torch` / ``torch.from_dlpack`` alias the result and must not be written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .errors import StaleResultError

__all__ = ["Array"]


class Array:
    """A read-only view of a result's array, with DLPack and the NumPy array interface.

    Attributes are those of a NumPy array: :attr:`shape`, :attr:`dtype`, :attr:`size`,
    :attr:`ndim`, ``len(a)``, iteration and indexing (which read through a copy-free NumPy
    view). Every array is 1-D except ``dyng.mosp.Result.path_costs``, which is 2-D (n, K).
    ``np.asarray(a)`` (through ``__array_interface__``), ``np.from_dlpack(a)`` and
    ``torch.from_dlpack(a)`` (through ``__dlpack__``) view the library's memory without a copy,
    and the view keeps the result alive.

    **Lifetime**: once the result is updated, using the Array raises
    :class:`~dyng.StaleResultError`; read the property again for the new state. Views already
    made from it keep showing the old state, unchanged (the update copies the result's state
    while a view is alive; see the module description).
    """

    __slots__ = ("_nd", "_view", "_is_current", "_what", "__weakref__")

    def __init__(self, nd: Any, is_current: Callable[[], bool], what: str) -> None:
        self._nd = nd  # nanobind's array-API object: holds a reference to the result's state
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
        """The shape ((n,) for 1-D arrays, (n, K) for path costs)."""
        return self._numpy_view().shape

    @property
    def ndim(self) -> int:
        """The number of dimensions (1, or 2 for path costs)."""
        return int(self._numpy_view().ndim)

    @property
    def size(self) -> int:
        """The number of elements."""
        return int(self._numpy_view().size)

    @property
    def device(self) -> str:
        """The device, ``"cpu"`` for host memory or ``"cuda:<n>"`` for device memory."""
        kind, dev = self.__dlpack_device__()
        return "cpu" if kind in (1, 3) else f"cuda:{dev}"

    def __len__(self) -> int:
        return int(self.shape[0])

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

    # -- copying and pickling ------------------------------------------------------------------
    def __copy__(self) -> Array:
        return self  # a read-only view: a shallow copy is the same view

    def __deepcopy__(self, memo: dict[int, Any]) -> np.ndarray:
        return self.to_numpy()

    def __reduce__(self) -> Any:
        # Pickles (and deep-copies) as an independent NumPy array of the elements.
        return (np.asarray, (self.to_numpy(),))

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

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: Any = None,
        copy: bool | None = None,
    ) -> Any:
        """Export through DLPack (``stream``, ``max_version``, ``dl_device`` and ``copy`` as in
        the Python array API).

        A consumer that asks for DLPack >= 1.0 (``max_version=(1, 0)`` or later: NumPy >= 2.1,
        recent PyTorch, CuPy and JAX) gets a zero-copy capsule carrying the read-only flag. The
        unversioned (legacy) capsule, which older consumers ask for, cannot carry that flag:
        they get an independent copy (``copy=False`` then raises ``BufferError``), so no
        consumer can write into the result's state.
        """
        self._check()
        legacy = max_version is None or tuple(max_version) < (1, 0)
        if not legacy:
            kwargs: dict[str, Any] = {"max_version": max_version}
            if stream is not None:
                kwargs["stream"] = stream
            if dl_device is not None:
                kwargs["dl_device"] = dl_device
            if copy is not None:
                kwargs["copy"] = copy
            return self._nd.__dlpack__(**kwargs)
        if copy is False:
            raise BufferError(
                f"dyng.Array ({self._what}): a zero-copy export is read-only, which the "
                "unversioned DLPack capsule cannot express; ask for max_version=(1, 0) or allow "
                "a copy"
            )
        kind, _ = self.__dlpack_device__()
        if kind == 2:
            raise BufferError(
                f"dyng.Array ({self._what}): device memory is exported through DLPack >= 1.0 "
                "only (max_version=(1, 0)), whose capsule is read-only"
            )
        return self._numpy_view().copy().__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        """The DLPack device: (1, 0) for host memory, (2, n) for CUDA device n."""
        kind, dev = self._nd.__dlpack_device__()
        return int(kind), int(dev)

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """The elements as a NumPy array.

        Args:
            copy: True (the default): an independent copy, safe to keep across updates.
                False: a read-only view of the library's memory; it keeps showing this state of
                the result after an update (which then copies the state; see the module
                description).
        """
        v = self._numpy_view()
        return v.copy() if copy else v

    def tolist(self) -> list[Any]:
        """The elements as a Python list."""
        out: list[Any] = self._numpy_view().tolist()
        return out

    def to_torch(self) -> Any:
        """A PyTorch tensor viewing the elements (zero-copy through DLPack; needs torch).

        PyTorch has no read-only tensors: the tensor aliases the result's state and **must not
        be written** (an in-place change would corrupt what later updates compute from). Use
        ``torch.from_numpy(a.to_numpy())`` for a tensor you may change.
        """
        import torch  # lazy: an optional dependency

        self._check()
        return torch.from_dlpack(self._nd)

    def to_cupy(self) -> Any:
        """A CuPy array of the elements (zero-copy for device memory; host memory is copied to
        the current device; needs cupy).

        A zero-copy CuPy array aliases the result's state and must not be written (CuPy has no
        read-only arrays).
        """
        import cupy  # lazy: an optional dependency

        self._check()
        kind, _ = self.__dlpack_device__()
        if kind == 2:
            return cupy.from_dlpack(self._nd)
        return cupy.asarray(self._numpy_view())
