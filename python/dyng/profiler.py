# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The profiler: where the library spends its time, stage by stage (C++ ``dyng::profiler``).

Example:
    >>> import dyng
    >>> g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    >>> with dyng.profile(g.resources) as p:
    ...     tree = dyng.sssp.compute(g, 0)
    >>> any(s.name.startswith("sssp.") for s in p.stages)
    True
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ._backend import native
from .resources import Resources, resolve

__all__ = ["Profiler", "StageRecord", "StageSample", "profile"]


@dataclass(frozen=True)
class StageRecord:
    """One stage: its totals over every call.

    Attributes:
        name: The stage name, ``<algo>.<hook>[.<sub>]``.
        depth: Nesting depth at the first call (0 = outermost).
        calls: Completed calls.
        host_ms: Total host wall time (ms).
        device_ms: Total device time (ms; 0 without CUDA events).
    """

    name: str
    depth: int
    calls: int
    host_ms: float
    device_ms: float


@dataclass(frozen=True)
class StageSample:
    """One completed call of a stage.

    Attributes:
        name: The stage.
        depth: Nesting depth of this call.
        host_ms: Host wall time (ms).
        device_ms: Device time (ms; 0 without CUDA events).
    """

    name: str
    depth: int
    host_ms: float
    device_ms: float


class Profiler:
    """Records the stages and counters of the library's calls through the resources it is
    attached to (use :func:`profile`).

    The records are read as copies, while no native call is running (the library writes them
    during calls, which run without the GIL).

    Args:
        sync_stages: Synchronize the stream at every stage boundary (CUDA; exact stage times).
        nvtx: Emit NVTX ranges (CUDA builds with NVTX).
        cuda_events: Time stages with CUDA events (device time).
    """

    __slots__ = ("_native",)
    _native: Any

    def __init__(
        self, *, sync_stages: bool = False, nvtx: bool = False, cuda_events: bool = False
    ) -> None:
        self._native = native.Profiler(sync_stages, nvtx, cuda_events)

    @property
    def stages(self) -> list[StageRecord]:
        """Every stage, in first-call order."""
        return [StageRecord(*row) for row in self._native.stages()]

    @property
    def samples(self) -> list[StageSample]:
        """Every completed call, in completion order."""
        return [StageSample(*row) for row in self._native.samples()]

    @property
    def counters(self) -> dict[str, int]:
        """The counters (name -> sum of the added values)."""
        return dict(self._native.counters())

    def total_host_ms(self, name: str) -> float:
        """The total host time of a stage (0 if it never ran)."""
        return sum(s.host_ms for s in self.stages if s.name == name)

    def reset(self) -> None:
        """Forget every record."""
        self._native.reset()

    def to_csv(self) -> str:
        """The records as CSV (the C++ ``write_csv``)."""
        return str(self._native.to_csv())

    def to_json(self) -> str:
        """The records as JSON (the C++ ``write_json``)."""
        return str(self._native.to_json())

    def to_dataframe(self) -> Any:
        """The stages as a pandas DataFrame (needs pandas)."""
        try:
            import pandas as pd  # lazy: an optional dependency
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("Profiler.to_dataframe() needs pandas (pip install pandas)") from e
        return pd.DataFrame([s.__dict__ for s in self.stages])

    def __repr__(self) -> str:
        return f"dyng.Profiler({len(self.stages)} stages)"


@contextmanager
def profile(
    resources: Resources | None = None,
    *,
    sync_stages: bool = False,
    nvtx: bool = False,
    cuda_events: bool = False,
) -> Iterator[Profiler]:
    """Attach a new :class:`Profiler` to ``resources`` for the ``with`` block.

    Every copy of the handle records into it (graphs remember their resources, so pass the
    graph's: ``dyng.profile(g.resources)``). The previous profiler, if any, is attached again on
    exit. Attaching and detaching wait until no native call is running in any thread, so a call
    that runs while another thread enters or leaves the block records wholly into one profiler.

    Args:
        resources: Default: the default resources.
        sync_stages: See :class:`Profiler`.
        nvtx: See :class:`Profiler`.
        cuda_events: See :class:`Profiler`.
    """
    res = resolve(resources)
    p = Profiler(sync_stages=sync_stages, nvtx=nvtx, cuda_events=cuda_events)
    previous = res._profiler
    res._native.attach_profiler(p._native)
    res._profiler = p
    try:
        yield p
    finally:
        res._native.attach_profiler(None if previous is None else previous._native)
        res._profiler = previous
