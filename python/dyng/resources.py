# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Execution resources: the backend (sequential, OpenMP, CUDA), threads, stream and profiler.

Every function of dynG takes a keyword-only ``resources=None``. For the algorithms and
:meth:`dyng.Graph.apply`, ``None`` means the resources the graph was built with; for everything
else (building graphs, reading files) it means the process-wide default,
:func:`get_default_resources` (the one documented exception to "no hidden global state", PLAN
Section 5.4).
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from ._backend import native
from ._convert import enum_member, enum_name
from .errors import NotSupportedError

__all__ = ["Resources", "get_default_resources", "set_default_resources"]

BackendName = Literal["sequential", "openmp", "cuda"]
CopyPolicyName = Literal["allow", "warn", "error"]

_CUDA_PLUGIN_MESSAGE = (
    "this dyng installation is the CPU wheel (sequential and OpenMP backends); the CUDA backends "
    "come as the plugin wheels dyng-cu12 and dyng-cu13 of the 0.1.x releases "
    '(pip install "dyng[cu13]" once they are published), or build dyng from source with CUDA '
    "(docs: getting started, 'Building from source')"
)


def _stream_handle(stream: Any) -> int:
    """A CUDA stream as an integer handle: None/0, an int, ``__cuda_stream__``, CuPy or PyTorch."""
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    protocol = getattr(stream, "__cuda_stream__", None)
    if protocol is not None:
        value = protocol() if callable(protocol) else protocol
        return int(value[1])
    for attr in ("ptr", "cuda_stream", "handle"):
        value = getattr(stream, attr, None)
        if isinstance(value, int):
            return value
    raise TypeError(
        "stream must be None, an int handle, an object with __cuda_stream__, or a CuPy or "
        "PyTorch stream"
    )


class Resources:
    """Execution resources: backend, device, stream, thread count and profiler.

    Cheap to copy: copies share one handle, so :attr:`copy_policy` and :func:`dyng.profile`
    affect every copy (as in C++, ``dyng::resources``).

    Args:
        backend: ``"sequential"``, ``"openmp"``, ``"cuda"``, or None for the default backend
            (CUDA if available, else OpenMP, else sequential).
        num_threads: OpenMP threads (``openmp``; 0 = the OpenMP default, which honours
            ``OMP_NUM_THREADS``) or host threads (``cuda``).
        device: The CUDA device (``cuda`` only).

    Examples:
        >>> import dyng
        >>> dyng.Resources("sequential").backend
        'sequential'
    """

    __slots__ = ("_native", "_profiler", "__weakref__")

    def __init__(
        self, backend: BackendName | None = None, *, num_threads: int = 0, device: int = 0
    ) -> None:
        self._profiler: Any = None
        if backend is None:
            self._native = native.Resources()
        elif backend == "sequential":
            self._native = native.Resources.sequential()
        elif backend == "openmp":
            self._native = native.Resources.openmp(num_threads)
        elif backend == "cuda":
            self._native = Resources.cuda(device=device, host_threads=num_threads)._native
        else:
            enum_member(native.Backend, backend, "dyng.Resources: backend")
            raise AssertionError  # pragma: no cover

    @classmethod
    def _wrap(cls, handle: Any) -> Resources:
        self = cls.__new__(cls)
        self._native = handle
        self._profiler = None
        return self

    @classmethod
    def sequential(cls) -> Resources:
        """The sequential (reference) backend: one host thread."""
        return cls._wrap(native.Resources.sequential())

    @classmethod
    def openmp(cls, num_threads: int = 0) -> Resources:
        """The OpenMP backend.

        Args:
            num_threads: Threads of the library's parallel regions; 0 = the OpenMP default at the
                time of the call (``omp_get_max_threads()``, which honours ``OMP_NUM_THREADS``).

        Raises:
            NotSupportedError: if the library was built without OpenMP.
        """
        return cls._wrap(native.Resources.openmp(num_threads))

    @classmethod
    def cuda(cls, device: int = 0, stream: Any = None, host_threads: int = 0) -> Resources:
        """The CUDA backend on one device and stream.

        Args:
            device: The CUDA device ordinal.
            stream: The stream all work is ordered on: None (the per-thread default stream), an
                integer ``cudaStream_t``, a CuPy or PyTorch stream, or any object with
                ``__cuda_stream__``.
            host_threads: OpenMP threads of the host-side work (0 = the OpenMP default).

        Raises:
            NotSupportedError: in the CPU wheel (see the message: the CUDA plugins of 0.1.x), or
                when no device is visible.
        """
        handle = _stream_handle(stream)
        if not native.build_config["cuda"]:
            raise NotSupportedError(f"dyng.Resources.cuda: {_CUDA_PLUGIN_MESSAGE}")
        return cls._wrap(native.Resources.cuda(device, handle, host_threads))

    @property
    def backend(self) -> BackendName:
        """The backend: ``"sequential"``, ``"openmp"`` or ``"cuda"``."""
        return enum_name(self._native.backend)  # type: ignore[return-value]

    @property
    def device(self) -> int:
        """The CUDA device, -1 for the host backends."""
        return int(self._native.device)

    @property
    def num_threads(self) -> int:
        """Host threads: 1 (sequential), the OpenMP team size, or the host threads of cuda."""
        return int(self._native.num_threads)

    @property
    def default_space(self) -> str:
        """Where results are placed: ``"host"`` or ``"device"``."""
        return enum_name(self._native.default_space)

    @property
    def copy_policy(self) -> CopyPolicyName:
        """What an implicit copy of an input between memory spaces does: ``"allow"`` (the
        default), ``"warn"`` or ``"error"`` (raise instead of copying). Shared by every copy."""
        return enum_name(self._native.copy_policy)  # type: ignore[return-value]

    @copy_policy.setter
    def copy_policy(self, value: CopyPolicyName) -> None:
        self._native.copy_policy = enum_member(native.CopyPolicy, value, "copy_policy")

    @property
    def workspace_bytes(self) -> int:
        """Bytes of scratch memory the handle caches for the engines."""
        return int(self._native.workspace_bytes)

    def release_workspaces(self) -> None:
        """Free the cached scratch memory that is not in use."""
        self._native.release_workspaces()

    def warm_up(self) -> None:
        """Initialise the backend ahead of timed work (CUDA: context and kernels)."""
        self._native.warm_up()

    def synchronize(self) -> None:
        """Wait for the work enqueued on the handle's stream (a no-op on the host backends)."""
        self._native.synchronize()

    def __repr__(self) -> str:
        extra = f", device={self.device}" if self.backend == "cuda" else ""
        return f"dyng.Resources({self.backend!r}, num_threads={self.num_threads}{extra})"


_default_lock = threading.Lock()
_default: Resources | None = None


def get_default_resources() -> Resources:
    """The process-wide default resources (created on first use for the default backend)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Resources()
        return _default


def set_default_resources(resources: Resources | BackendName | None) -> None:
    """Replace the process-wide default resources.

    Args:
        resources: The new default, a backend name, or None to go back to the default backend.
    """
    global _default
    if isinstance(resources, str):
        resources = Resources(resources)
    if resources is not None and not isinstance(resources, Resources):
        raise TypeError("set_default_resources: expected a dyng.Resources, a backend name or None")
    with _default_lock:
        _default = resources


def resolve(resources: Resources | None, fallback: Resources | None = None) -> Resources:
    """``resources``, else ``fallback``, else the default resources (internal)."""
    if resources is not None:
        if not isinstance(resources, Resources):
            raise TypeError("resources must be a dyng.Resources")
        return resources
    if fallback is not None:
        return fallback
    return get_default_resources()
