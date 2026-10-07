# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""What graphs and results keep of the resources used on them (internal; ADR 0031).

A native result holder remembers the native resources of the call that last wrote it
(``handle.writer``): their stream orders the result's device memory for every later reader
(``to_numpy()``, ``__cuda_array_interface__``, ``__dlpack__``). Those native resources refer to
the stream by its raw handle only; the stream object given to ``Resources.cuda(stream=...)`` is
kept alive by the Python :class:`~dyng.Resources`. So every result keeps the Python resources of
its last writer (``_writer``), and every graph and result keeps the resources of every stream
object used on it (``_streams``, :class:`StreamKeep`): the memory a call adds is released on the
call's stream, possibly long after it. The Arrays read from a result keep the same.

The bookkeeping lives in a side table, not in the classes' ``__slots__`` (the Python API check
reports any change of a public class's ``__slots__``); the classes read it through the
properties :data:`WRITER`, :data:`STREAMS` and :data:`WRITE_LOCK`, call :func:`track` when an
object is made, and use :func:`release` as ``__del__``: it frees the native object first, then
the bookkeeping, so the streams outlive the memory released on them.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

__all__ = [
    "STREAMS",
    "WRITER",
    "WRITE_LOCK",
    "StreamKeep",
    "release",
    "resolve_on",
    "track",
    "writing",
]


class StreamKeep:
    """Keeps the stream objects of every resources that ran a call on one graph or result.

    A call allocates the memory it adds to a graph or a result on the stream of its resources,
    and that memory is released on the same stream (``dyng::buffer``), possibly long after the
    call: by a later update that replaces it, or when the graph or result is freed. So the stream
    object of every resources used on the object (``Resources.cuda(stream=...)``) must live as
    long as the object, not only that of its last writer. Integer handles and the default
    streams have no object: the caller keeps such a stream alive (documented).

    It keeps one :class:`~dyng.Resources` per stream object, not the stream itself: the
    resources' native handle also releases memory on the stream (its cached workspaces), and
    ``Resources.__del__`` releases the handle before the stream. Graphs and results keep one
    (:data:`STREAMS`, released after their native object by :func:`release`); Arrays keep it too
    (:class:`dyng.Array` releases its memory first).
    """

    __slots__ = ("_objects",)

    def __init__(self, resources: Any = None) -> None:
        self._objects: dict[int, Any] = {}
        if resources is not None:
            self.add(resources)

    def add(self, resources: Any) -> None:
        """Keep ``resources`` if they were made with a stream object not kept yet."""
        from .resources import _stream_object

        stream = _stream_object(resources)
        if stream is not None:
            # The id is unique while the stream lives, and the kept resources keep it alive.
            self._objects.setdefault(id(stream), resources)

    def streams(self) -> list[Any]:
        """The stream objects kept."""
        from .resources import _stream_object

        return [_stream_object(r) for r in self._objects.values()]

    def __len__(self) -> int:
        return len(self._objects)


def resolve_on(obj: Any, resources: Any) -> Any:
    """``resources``, else the resources of ``obj`` (a graph or a result), kept by ``obj``."""
    from .resources import resolve

    res = resolve(resources, obj._resources)
    obj._streams.add(res)
    return res


class _Book:
    """The bookkeeping of one graph or result."""

    __slots__ = ("writer", "lock", "streams")

    def __init__(self, resources: Any) -> None:
        self.writer = resources  # a result's last writer
        self.lock = threading.Lock()  # a result's writer bookkeeping (writing())
        self.streams = StreamKeep(resources)


_books: weakref.WeakKeyDictionary[Any, _Book] = weakref.WeakKeyDictionary()
_books_lock = threading.Lock()


def track(obj: Any, resources: Any) -> None:
    """Start the bookkeeping of a new graph or result made with ``resources``."""
    _books[obj] = _Book(resources)


def _book(obj: Any) -> _Book:
    book = _books.get(obj)
    if book is None:  # not made through track() (cannot happen through the public API)
        with _books_lock:
            book = _books.get(obj)
            if book is None:
                book = _books[obj] = _Book(obj._resources)
    return book


def _set_writer(obj: Any, resources: Any) -> None:
    _book(obj).writer = resources


#: ``_writer``: the Python resources of the call that last wrote a result.
WRITER = property(lambda obj: _book(obj).writer, _set_writer)
#: ``_streams``: the :class:`StreamKeep` of a graph or result.
STREAMS = property(lambda obj: _book(obj).streams)
#: ``_write_lock``: the lock of a result's writer bookkeeping.
WRITE_LOCK = property(lambda obj: _book(obj).lock)


def release(obj: Any) -> None:
    """``__del__`` of graphs and results: the native object (and its memory, released on the
    streams of the resources used) first, then the bookkeeping that keeps those streams."""
    try:
        obj._native = None
        _books.pop(obj, None)
    except Exception:  # pragma: no cover - interpreter shutdown (module globals gone)
        pass


@contextmanager
def writing(results: Sequence[Any], resources: Any) -> Iterator[None]:
    """Run a native update of ``results`` with ``resources`` (inside the ``with``).

    Afterwards, every result the update wrote keeps ``resources`` as its writer, and every result
    keeps their stream object (:class:`StreamKeep`). A result was
    written when its generation moved (the native holder advances the generation and records the
    writer in one step, also when the update fails after that point). The results' locks, taken
    in a fixed order, make the native call and this bookkeeping one step per result, so two
    threads updating one result cannot leave the Python writer different from the native one.
    """
    with ExitStack() as stack:
        for r in sorted(results, key=id):
            stack.enter_context(r._write_lock)
        before = [int(r._native.generation) for r in results]
        try:
            yield
        finally:
            for r, generation in zip(results, before, strict=True):
                r._streams.add(resources)
                if int(r._native.generation) != generation:
                    r._writer = resources
