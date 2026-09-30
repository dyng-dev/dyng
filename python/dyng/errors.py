# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The exceptions of dynG (PLAN Section 5.4).

Every exception the library raises derives from :class:`Error`. Each class also derives from the
builtin exception a Python user would expect, so ``except ValueError`` catches an
:class:`InvalidArgumentError` and ``except OSError`` a :class:`FileFormatError`.

======================== ======================================== ==============================
dynG                     Bases                                    C++
======================== ======================================== ==============================
Error                    Exception                                dyng::error
InvalidArgumentError     Error, ValueError                        invalid_argument_error
StaleResultError         InvalidArgumentError                     stale_result_error
FileFormatError          Error, OSError (``.path``, ``.line``)    io_error
CapacityError            Error, MemoryError                       capacity_error
NotSupportedError        Error, NotImplementedError               not_supported_error
ConvergenceError         Error, RuntimeError                      convergence_error
CudaError                Error, RuntimeError (``.code``)          cuda_error
OutOfMemoryError         Error, MemoryError                       out_of_memory_error
InternalError            Error, RuntimeError                      internal_error
======================== ======================================== ==============================
"""

from __future__ import annotations

from ._backend import native

__all__ = [
    "Error",
    "InvalidArgumentError",
    "StaleResultError",
    "FileFormatError",
    "CapacityError",
    "NotSupportedError",
    "ConvergenceError",
    "CudaError",
    "OutOfMemoryError",
    "InternalError",
]


class Error(Exception):
    """Base class of every exception raised by dynG."""


class InvalidArgumentError(Error, ValueError):
    """Bad input: ids out of range, wrong types or backend, violated preconditions."""


class StaleResultError(InvalidArgumentError):
    """A result was used with a graph whose state no longer matches it.

    Raised when a result is updated with a graph that was changed without it (for example by
    another algorithm's ``update``), with another graph, after a failed update left it unusable,
    and when a :class:`dyng.Array` of a result is read after the result was updated.
    """


class FileFormatError(Error, OSError):
    """A file could not be read or written, or its content is malformed.

    Attributes:
        path: The file concerned, or None.
        line: The 1-based line of the problem, or None.
        column: The 1-based column of the problem, or None.
    """

    path: str | None
    line: int | None
    column: int | None

    def __init__(
        self,
        message: str,
        path: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""

    def __reduce__(self):  # keeps the attributes through pickling (multiprocessing)
        return (type(self), (str(self), self.path, self.line, self.column))


class CapacityError(Error, MemoryError):
    """A fixed capacity was exceeded (for example more edges than the edge offset type holds)."""


class NotSupportedError(Error, NotImplementedError):
    """A backend, type combination, engine or feature is not built or not implemented."""


class ConvergenceError(Error, RuntimeError):
    """An iterative algorithm hit its iteration cap and was asked to fail."""


class CudaError(Error, RuntimeError):
    """A CUDA runtime or driver call failed.

    Attributes:
        code: The numeric ``cudaError_t`` value, or None.
    """

    code: int | None

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code

    def __reduce__(self):
        return (type(self), (str(self), self.code))


class OutOfMemoryError(Error, MemoryError):
    """A host or device allocation failed."""


class InternalError(Error, RuntimeError):
    """A broken internal invariant: a library bug. Please report it."""


native._set_error_types(
    {
        "Error": Error,
        "InvalidArgumentError": InvalidArgumentError,
        "StaleResultError": StaleResultError,
        "FileFormatError": FileFormatError,
        "CapacityError": CapacityError,
        "NotSupportedError": NotSupportedError,
        "ConvergenceError": ConvergenceError,
        "CudaError": CudaError,
        "OutOfMemoryError": OutOfMemoryError,
        "InternalError": InternalError,
    }
)
