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

**Device memory** (results of the CUDA backend, with a CUDA plugin wheel; ADR 0031). ``device``
is ``"cuda:<n>"`` and ``__dlpack_device__()`` is ``(2, n)``. The array exports
``__cuda_array_interface__`` (version 3, read-only, with the stream of the call that wrote the
result) and ``__dlpack__`` (DLPack >= 1.0 only; zero-copy), so ``torch.from_dlpack(a)``,
``cupy.from_dlpack(a)`` and ``cupy.asarray(a)`` view it on the device. A DLPack consumer passes its
stream (``__dlpack__(stream=...)``) and the array orders it after the stream that wrote the result
with a CUDA event, as the DLPack Python specification prescribes (PLAN Section 5.4, rule 4);
the host does not wait. ``shape``, ``dtype``, ``size`` and ``ndim`` read no element.
:meth:`Array.to_numpy` copies to the host (ordered on that stream); ``to_numpy(copy=False)``
raises, since host code cannot view device memory. Indexing, iteration, comparison, ``repr``
and ``np.asarray(a)`` (``__array__``) read through one host copy that the Array keeps.
"""

from __future__ import annotations

import atexit
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ._backend import native
from .errors import StaleResultError

__all__ = ["Array"]

#: DLPack device types: host, CUDA device memory, pinned host memory, CUDA managed memory.
_KDL_CPU, _KDL_CUDA, _KDL_CUDA_HOST, _KDL_CUDA_MANAGED = 1, 2, 3, 13
#: The CUDA stream values of the DLPack and CUDA array interface protocols.
_LEGACY_STREAM, _PER_THREAD_STREAM = 1, 2

_cuda_resources_lock = threading.Lock()
_cuda_resources: dict[int, Any] = {}


def cuda_resources(device: int) -> Any:
    """Native CUDA resources of ``device`` on the legacy default stream (cached; internal).

    They copy device memory that dynG did not write (inputs from CuPy, PyTorch, ...) to the host.
    The legacy default stream (DLPack's stream 1) is the one every producer can order its work
    before (PyTorch refuses the per-thread default stream as a DLPack consumer stream).
    """
    with _cuda_resources_lock:
        res = _cuda_resources.get(device)
        if res is None:
            res = native.Resources.cuda(device, _LEGACY_STREAM, 0)
            _cuda_resources[device] = res
        return res


@atexit.register
def _drop_cuda_resources() -> None:
    # Module globals may be cleared after the native module is finalized: drop the cached
    # handles first, so nanobind does not report them as leaked (as resources.py does).
    with _cuda_resources_lock:
        _cuda_resources.clear()


def is_device(kind: int) -> bool:
    """Whether a DLPack device type is CUDA memory the host cannot read directly."""
    return kind in (_KDL_CUDA, _KDL_CUDA_MANAGED)


def consumer_stream(stream: Any) -> int:
    """A DLPack ``stream`` argument as a CUDA stream handle (None: the legacy default stream)."""
    if stream is None:
        return _LEGACY_STREAM
    if isinstance(stream, bool) or not isinstance(stream, int):
        raise TypeError(f"__dlpack__: stream must be None or an int, got {type(stream).__name__}")
    if stream == 0:
        raise ValueError(
            "__dlpack__: stream=0 is ambiguous for CUDA (DLPack); pass 1 for the legacy default "
            "stream, 2 for the per-thread default stream, or a stream handle"
        )
    return int(stream)


class _Owner:
    """What an Array knows about its result: whether the state it views is current, the
    resources of the call that last wrote the result (their stream orders device memory), and the
    result's dyng.Resources, kept alive with the Array (and with them a stream object)."""

    __slots__ = ("is_current", "writer", "keep")

    def __init__(
        self, is_current: Callable[[], bool], writer: Callable[[], Any] | None, keep: Any
    ) -> None:
        self.is_current, self.writer, self.keep = is_current, writer, keep

    def __call__(self) -> bool:
        return bool(self.is_current())

    def __del__(self) -> None:
        # The callables hold the native result (and so its memory) and go before ``keep``, which
        # keeps the stream objects that memory is released on (CPython would clear the slots in
        # sorted order: is_current, keep, writer).
        self.is_current = self.writer = None  # type: ignore[assignment]


class Array:
    """A read-only view of a result's array, with DLPack and the NumPy or CUDA array interface.

    Attributes are those of a NumPy array: :attr:`shape`, :attr:`dtype`, :attr:`size`,
    :attr:`ndim`, ``len(a)``, iteration and indexing (which read through a copy-free NumPy
    view in host memory, and through one host copy in device memory). Every array is 1-D except
    ``dyng.mosp.Result.path_costs``, which is 2-D (n, K). Host memory: ``np.asarray(a)``
    (through ``__array_interface__``), ``np.from_dlpack(a)`` and ``torch.from_dlpack(a)``
    (through ``__dlpack__``) view the library's memory without a copy, and the view keeps the
    result alive. Device memory (:attr:`device` ``"cuda:<n>"``): ``torch.from_dlpack(a)``,
    ``cupy.from_dlpack(a)`` and ``cupy.asarray(a)`` (through ``__cuda_array_interface__``) view it
    on the device; :meth:`to_numpy` copies it to the host.

    **Lifetime**: once the result is updated, using the Array raises
    :class:`~dyng.StaleResultError`; read the property again for the new state. Views already
    made from it keep showing the old state, unchanged (the update copies the result's state
    while a view is alive; see the module description).
    """

    __slots__ = ("_nd", "_view", "_is_current", "_what", "__weakref__")

    def __init__(
        self,
        nd: Any,
        is_current: Callable[[], bool],
        what: str,
        writer: Callable[[], Any] | None = None,
        keep: Any = None,
    ) -> None:
        self._nd = nd  # nanobind's array-API object: holds a reference to the result's state
        self._view: np.ndarray | None = None  # host: a NumPy view; device: a read-only host copy
        self._is_current = _Owner(is_current, writer, keep)
        self._what = what

    def __del__(self) -> None:
        # The result's memory (held by _nd) goes first, then the owner, which may hold the last
        # reference to the stream objects that memory is released on (dyng._writer.StreamKeep);
        # CPython would clear the slots in sorted order, _is_current before _nd.
        self._nd = None

    # -- state ---------------------------------------------------------------------------------
    def _check(self) -> None:
        if not self._is_current():
            raise StaleResultError(
                f"dyng.Array ({self._what}): the result was updated since this array was read; "
                "read the property of the result again"
            )

    def _on_device(self) -> bool:
        return is_device(self.__dlpack_device__()[0])

    def _meta(self) -> tuple[Any, ...]:
        """(pointer, shape, type string, device type, device id, C-contiguous); no element read."""
        return tuple(native.array_info(self._nd))

    def _resources(self) -> Any:
        """The native resources whose stream orders this array's device memory."""
        writer = self._is_current.writer
        res = writer() if writer is not None else None
        if res is None or res.device != self.__dlpack_device__()[1]:
            res = cuda_resources(self.__dlpack_device__()[1])
        return res

    def _to_host(self) -> np.ndarray:
        out: np.ndarray = native.array_to_host(self._resources(), self._nd)
        return out

    def _numpy_view(self) -> np.ndarray:
        self._check()
        if self._view is None:
            v = self._to_host() if self._on_device() else np.from_dlpack(self._nd)
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
        if self._on_device():
            self._check()
            out: np.dtype = np.dtype(self._meta()[2])
            return out
        return self._numpy_view().dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape ((n,) for 1-D arrays, (n, K) for path costs)."""
        if self._on_device():
            self._check()
            return tuple(int(x) for x in self._meta()[1])
        return self._numpy_view().shape

    @property
    def ndim(self) -> int:
        """The number of dimensions (1, or 2 for path costs)."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        n = 1
        for x in self.shape:
            n *= x
        return n

    @property
    def device(self) -> str:
        """The device, ``"cpu"`` for host memory or ``"cuda:<n>"`` for device memory."""
        kind, dev = self.__dlpack_device__()
        return "cpu" if kind in (_KDL_CPU, _KDL_CUDA_HOST) else f"cuda:{dev}"

    def __len__(self) -> int:
        return int(self.shape[0])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._numpy_view().tolist())

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
        where = "" if not self._on_device() else f", device={self.device!r}"
        return f"dyng.Array({np.array2string(v, threshold=20)}, dtype={v.dtype.name}{where})"

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
        """The NumPy array interface (host memory only; read-only, zero-copy)."""
        if self._on_device():
            raise AttributeError(
                f"dyng.Array ({self._what}) is in device memory ({self.device}): it has "
                "__cuda_array_interface__, not __array_interface__"
            )
        return self._numpy_view().__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """The CUDA array interface, version 3 (device memory only; read-only, zero-copy).

        ``stream`` is the stream of the call that last wrote the result (1 for the legacy
        default stream, 2 for the per-thread default stream); a consumer orders its work after
        it, as the protocol prescribes.
        """
        if not self._on_device():
            raise AttributeError(
                f"dyng.Array ({self._what}) is in host memory: it has __array_interface__, not "
                "__cuda_array_interface__"
            )
        self._check()
        pointer, shape, typestr, _, _, _ = self._meta()
        handle = int(self._resources().stream)
        return {
            "shape": tuple(int(x) for x in shape),
            "typestr": typestr,
            "data": (int(pointer), True),
            "version": 3,
            "strides": None,
            "stream": _LEGACY_STREAM if handle == 0 else handle,
        }

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and self._on_device():
            raise ValueError(
                f"dyng.Array ({self._what}) is in device memory ({self.device}): a NumPy array "
                "of it needs a copy (to_numpy())"
            )
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
        they get an independent copy of host memory (``copy=False`` then raises
        ``BufferError``), so no consumer can write into the result's state; device memory is
        exported through DLPack >= 1.0 only.

        Device memory: ``stream`` is the consumer's stream (None: the legacy default stream, 1
        the legacy default stream, 2 the per-thread default stream, another int a stream handle,
        -1 no ordering); the consumer's later work on it is ordered after the stream that wrote
        the result (a CUDA event; the host does not wait). ``dl_device=(1, 0)`` with ``copy``
        not False exports a host copy.
        """
        self._check()
        legacy = max_version is None or tuple(max_version) < (1, 0)
        kind, dev = self.__dlpack_device__()
        if dl_device is not None and tuple(dl_device) != (kind, dev):
            if tuple(dl_device) == (_KDL_CPU, 0) and copy is not False:
                host = self.to_numpy()
                return host.__dlpack__() if legacy else host.__dlpack__(max_version=max_version)
            raise BufferError(
                f"dyng.Array ({self._what}): cannot export {self.device} memory to the DLPack "
                f"device {tuple(dl_device)}"
            )
        if copy is True:
            if is_device(kind):
                raise BufferError(
                    f"dyng.Array ({self._what}): copy=True of device memory is not supported "
                    "(export it zero-copy and copy it in the consumer, or use to_numpy())"
                )
            host = self.to_numpy()
            return host.__dlpack__() if legacy else host.__dlpack__(max_version=max_version)
        if is_device(kind):
            if legacy:
                raise BufferError(
                    f"dyng.Array ({self._what}): device memory is exported through DLPack >= 1.0 "
                    "only (max_version=(1, 0)), whose capsule is read-only"
                )
            if stream != -1:
                native.order_stream(self._resources(), consumer_stream(stream))
            return self._nd.__dlpack__(max_version=max_version)
        if not legacy:
            kwargs: dict[str, Any] = {"max_version": max_version}
            if stream is not None:
                kwargs["stream"] = stream
            if copy is not None:
                kwargs["copy"] = copy
            return self._nd.__dlpack__(**kwargs)
        if copy is False:
            raise BufferError(
                f"dyng.Array ({self._what}): a zero-copy export is read-only, which the "
                "unversioned DLPack capsule cannot express; ask for max_version=(1, 0) or allow "
                "a copy"
            )
        return self._numpy_view().copy().__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        """The DLPack device: (1, 0) for host memory, (2, n) for CUDA device n."""
        kind, dev = self._nd.__dlpack_device__()
        return int(kind), int(dev)

    def to_numpy(self, *, copy: bool | None = True) -> np.ndarray:
        """The elements as a NumPy array.

        Args:
            copy: True (the default): an independent copy, safe to keep across updates.
                False: a read-only view of the library's memory; it keeps showing this state of
                the result after an update (which then copies the state; see the module
                description); raises ``ValueError`` for device memory, which the host cannot
                view. None: the read-only view of host memory, or a read-only host copy of
                device memory (made once per Array).
        """
        if self._on_device():
            if copy is False:
                raise ValueError(
                    f"dyng.Array ({self._what}) is in device memory ({self.device}): the host "
                    "cannot view it; to_numpy() (or copy=None) copies it to the host"
                )
            if copy and self._view is None:
                self._check()
                return self._to_host()
        v = self._numpy_view()
        return v.copy() if copy else v

    def tolist(self) -> list[Any]:
        """The elements as a Python list."""
        out: list[Any] = self._numpy_view().tolist()
        return out

    def to_torch(self) -> Any:
        """A PyTorch tensor viewing the elements (zero-copy through DLPack, on the array's
        device; needs torch).

        PyTorch has no read-only tensors: the tensor aliases the result's state and **must not
        be written** (an in-place change would corrupt what later updates compute from). Use
        ``torch.from_numpy(a.to_numpy())`` for a tensor you may change (or ``.clone()`` it).
        On a device, PyTorch passes its current stream, which is ordered after the result's.
        """
        import torch  # lazy: an optional dependency

        self._check()
        if self._on_device():
            return torch.from_dlpack(self)
        return torch.from_dlpack(self._nd)

    def to_cupy(self) -> Any:
        """A CuPy array of the elements (zero-copy for device memory, ordered on CuPy's current
        stream; host memory is copied to the current device; needs cupy).

        A zero-copy CuPy array aliases the result's state and must not be written (CuPy has no
        read-only arrays).
        """
        import cupy  # lazy: an optional dependency

        self._check()
        if self._on_device():
            return cupy.from_dlpack(self)
        return cupy.asarray(self._numpy_view())
