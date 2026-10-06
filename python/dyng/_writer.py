# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python resources of the call that last wrote a result (internal; ADR 0031).

A native result holder remembers the native resources of the call that last wrote it
(``handle.writer``): their stream orders the result's device memory for every later reader
(``to_numpy()``, ``__cuda_array_interface__``, ``__dlpack__``). Those native resources refer to
the stream by its raw handle only; the stream object given to ``Resources.cuda(stream=...)`` is
kept alive by the Python :class:`~dyng.Resources`. So every result keeps, next to the resources
it was made with, the Python resources of its last writer (``_writer``), and the Arrays read from
it keep both: an update with other resources (another stream) cannot leave the result naming a
stream that was destroyed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

__all__ = ["writing", "new_lock"]


def new_lock() -> threading.Lock:
    """The lock of one result's writer bookkeeping."""
    return threading.Lock()


@contextmanager
def writing(results: Sequence[Any], resources: Any) -> Iterator[None]:
    """Run a native update of ``results`` with ``resources`` (inside the ``with``).

    Afterwards, every result the update wrote keeps ``resources`` as its writer. A result was
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
                if int(r._native.generation) != generation:
                    r._writer = resources
