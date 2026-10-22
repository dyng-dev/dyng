# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.triad_count: exact triad counts of a dynamic hypergraph (ESCHER / ESCHER+).

:func:`compute` counts the closed triads of a :class:`dyng.Hypergraph` (three hyperedges that
pairwise share a vertex: the triangles of the line graph) by one of four patterns:

- ``"h_motif"``: the 30 h-motif classes of the triads' Venn regions (bin ``k`` is ESCHER-GPU's
  h-motif ``k + 1``; bins 20-25, ESCHER-GPU's open triads 21-26, are always 0);
- ``"stathyper_type1"``: one bin, the sum of ``C(|e|, 3)`` over the live hyperedges;
- ``"stathyper_type2"``: one bin, the sum of C(a, 2) * b * c over the adjacent pairs;
- ``"stathyper_type3"``: one bin, the sum of x * y * z over the closed triads.

:func:`update` applies a batch of hyperedge deletions and insertions and changes every bin by
the triads the batch destroys and creates (the subtract phase on the old hypergraph, the add
phase on the new one), without recounting the rest; afterwards the counts equal :func:`compute`
on the new hypergraph exactly. :func:`dyng.update` updates several results (one per pattern, say)
with one batch.

The names mirror the C++ API (``dyng::triad_count``, ``<dyng/triad_count.hpp>``). Supported
hypergraphs: int32 vertex ids, unweighted or int32 weights (the weights are ignored). The
h-motifs and types 2 and 3 read the line graph and the vertex-to-hyperedge rows: the hypergraph
needs ``line_graph`` other than ``"none"`` and ``store_v2h=True`` (the defaults); type 1 needs
neither. Cite with ``dyng.citation("triad_count")``.

Example:
    >>> import dyng
    >>> hg = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]])
    >>> tri = dyng.triad_count.compute(hg)
    >>> tri.total, tri.num_bins
    (4, 30)
    >>> st = dyng.triad_count.update(hg, dyng.HyperedgeBatch(delete_hyperedges=[3]), tri)
    >>> tri.total, int(st.delta.sum()), st.triads_removed
    (1, -3, 3)
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import _writer as _writer_state
from ._backend import native
from ._convert import as_int, copy_fields, enum_member, enum_name, with_options
from ._writer import resolve_on, writing
from .hypergraph import HyperApplySummary, HyperedgeBatch, Hypergraph
from .resources import Resources, resolve

__all__ = ["MAX_BINS", "Pattern", "PatternName", "Options", "Stats", "Result", "compute", "update"]

#: The bins of ``"h_motif"`` and the most any pattern has (C++ ``triad_count::max_bins``).
MAX_BINS = 30

#: The names of the patterns (the C++ enumerators of ``triad_count::pattern``).
PatternName = Literal["h_motif", "stathyper_type1", "stathyper_type2", "stathyper_type3"]


class Pattern(enum.StrEnum):
    """What a result counts (C++ ``dyng::triad_count::pattern``).

    A member is also its name as a string (``Pattern.h_motif == "h_motif"``), so every function
    that takes a pattern takes either.
    """

    h_motif = "h_motif"  #: the 30 h-motif classes (bins 20-25, the open triads, are always 0)
    stathyper_type1 = "stathyper_type1"  #: one bin: the sum of ``C(|e|, 3)``
    stathyper_type2 = "stathyper_type2"  #: one bin: the sum of C(a, 2) * b * c
    stathyper_type3 = "stathyper_type3"  #: one bin: the sum of x * y * z


@dataclass
class Options:
    """Options of :func:`compute` (C++ ``dyng::triad_count::options``).

    ``kind`` is fixed at :func:`compute`; ``cuda_engine`` is a tunable that never changes what is
    counted (:meth:`Result.set_options`).

    Attributes:
        kind: The pattern: ``"h_motif"`` (30 bins), ``"stathyper_type1"``,
            ``"stathyper_type2"`` or ``"stathyper_type3"`` (one bin each), or a
            :class:`Pattern`.
        cuda_engine: ``"automatic"`` or ``"fused"`` (the wedge engine); ``"operators"`` raises
            :class:`~dyng.NotSupportedError` on cuda (triad_count has no operators engine).
            Ignored by the host backends.
    """

    kind: PatternName | Pattern = "h_motif"
    cuda_engine: Literal["automatic", "fused", "operators"] = "automatic"

    def _to_native(self) -> Any:
        o = native.TriadCountOptions()
        o.kind = enum_member(native.TriadPattern, self.kind, "triad_count.Options.kind")
        o.cuda_engine = enum_member(
            native.Engine, self.cuda_engine, "triad_count.Options.cuda_engine"
        )
        return o

    @classmethod
    def _from_native(cls, o: Any) -> Options:
        return cls(**copy_fields(o, tuple(f.name for f in dataclasses.fields(cls))))


def _delta(values: Any, num_bins: int) -> np.ndarray:
    d = np.array(values, dtype=np.int64)[:num_bins]
    d.flags.writeable = False
    return d


@dataclass(frozen=True)
class Stats:
    """Counters of one :func:`update` (C++ ``dyng::triad_count::stats``); all deterministic and
    equal on every backend.

    Attributes:
        affected: Bins whose count changed.
        iterations: Always 0.
        frontier_visits: Hyperedges of the two phases (deleted_frontier + inserted_frontier).
        fallback_used: Always False (every backend updates incrementally).
        converged: Always True.
        engine_used: ``"operators"`` on the host backends, ``"fused"`` on cuda.
        batch: What applying the batch did to the hypergraph (``batch.inserted_ids``: the id of
            every inserted hyperedge, int64).
        deleted_frontier: The live hyperedges the batch deleted (the subtract phase).
        inserted_frontier: The hyperedges the batch inserted (the add phase).
        triads_removed: The triads the batch destroyed, summed over the bins (types 1-3, whose
            one bin is a sum of contributions: the contributions removed; the hyperedge weights
            are ignored).
        triads_added: The triads the batch created, summed over the bins (types 1-3: the
            contributions added).
        delta: The change of every bin, ``counts`` after minus ``counts`` before (a read-only
            int64 array of the result's ``num_bins`` values).
    """

    affected: int
    iterations: int
    frontier_visits: int
    fallback_used: bool
    converged: bool
    engine_used: str
    batch: HyperApplySummary
    deleted_frontier: int
    inserted_frontier: int
    triads_removed: int
    triads_added: int
    delta: np.ndarray

    @classmethod
    def _from_native(cls, s: Any, num_bins: int) -> Stats:
        names = tuple(f.name for f in dataclasses.fields(cls) if f.name not in ("batch", "delta"))
        return cls(
            **copy_fields(s, names),
            batch=HyperApplySummary._from_native(s.batch),
            delta=_delta(s.delta, num_bins),
        )


class Result:
    """Triad counts kept up to date by :func:`update` (C++ ``dyng::triad_count::result``).

    ``counts[bin]`` is the number of triads in that bin (for the StatHyper types the one bin
    holds the pattern's weighted sum); :attr:`num_bins` is 30 for ``"h_motif"`` and 1 otherwise.
    """

    __slots__ = ("_native", "_resources", "__weakref__")
    _native: Any
    _resources: Resources
    # the bookkeeping of dyng._writer (a side table)
    _writer = _writer_state.WRITER
    _streams = _writer_state.STREAMS
    _write_lock = _writer_state.WRITE_LOCK
    __del__ = _writer_state.release

    def __init__(self) -> None:
        raise TypeError("use dyng.triad_count.compute()")

    @classmethod
    def _wrap(cls, handle: Any, resources: Resources) -> Result:
        self = object.__new__(cls)
        self._native = handle
        self._resources = resources
        _writer_state.track(self, resources)
        return self

    @property
    def counts(self) -> np.ndarray:
        """The bins (a read-only uint64 NumPy copy of the current state, host memory on every
        backend; read it again after an update)."""
        c: np.ndarray = self._native.counts()
        c.flags.writeable = False
        return c

    def count(self, bin: int) -> int:
        """The count of one bin.

        Raises:
            InvalidArgumentError: ``bin`` is outside ``[0, num_bins)``.
        """
        b = as_int(bin, "triad_count.Result.count: bin", lo=-(2**31), hi=2**31 - 1)
        return int(self._native.count(b))

    @property
    def total(self) -> int:
        """The sum of the bins (:class:`~dyng.CapacityError` past 2^64 - 1)."""
        return int(self._native.total)

    @property
    def num_bins(self) -> int:
        """30 for ``"h_motif"``, 1 for the StatHyper types."""
        return int(self._native.num_bins)

    @property
    def kind(self) -> Pattern:
        """The pattern this result counts (fixed at :func:`compute`)."""
        return Pattern(enum_name(self._native.kind))

    @property
    def options(self) -> Options:
        """The options (a copy)."""
        return Options._from_native(self._native.options)

    def set_options(self, options: Options | None = None, **kwargs: Any) -> None:
        """Change the tunables (``cuda_engine``); ``kind`` must stay the same."""
        opt = with_options(
            Options, options or self.options, kwargs, "triad_count.Result.set_options"
        )
        self._native.set_options(opt._to_native())

    @property
    def graph_version(self) -> int:
        """The hypergraph version this result matches."""
        return int(self._native.graph_version)

    @property
    def space(self) -> str:
        """``"host"`` (the counts are a host array on every backend)."""
        return enum_name(self._native.space)

    def clone(self, resources: Resources | None = None) -> Result:
        """A deep copy for ``resources``."""
        res = resolve(resources, self._resources)
        return Result._wrap(self._native.clone(res._native), res)

    def __copy__(self) -> Result:
        return self.clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> Result:
        return self.clone()

    def __reduce__(self) -> Any:
        raise TypeError(
            "dyng.triad_count.Result cannot be pickled (it belongs to one state of one "
            "hypergraph); send result.counts (a NumPy array), or use copy.deepcopy(result) / "
            "result.clone() within a process"
        )

    def __repr__(self) -> str:
        nonzero = {i: int(v) for i, v in enumerate(self.counts.tolist()) if v}
        return (
            f"dyng.triad_count.Result(kind={str(self.kind)!r}, {nonzero}, "
            f"graph_version={self.graph_version})"
        )


def _check_hypergraph(hypergraph: Hypergraph) -> None:
    if not isinstance(hypergraph, Hypergraph):
        raise TypeError("dyng.triad_count: expected a dyng.Hypergraph")


def compute(
    hypergraph: Hypergraph,
    *,
    kind: PatternName | Pattern | None = None,
    options: Options | None = None,
    resources: Resources | None = None,
    **kwargs: Any,
) -> Result:
    """Count the closed triads of ``hypergraph`` by a pattern.

    Args:
        hypergraph: The hypergraph; it is not modified (``line_graph="automatic"`` builds its
            line graph here, once).
        kind: The pattern (:attr:`Options.kind`; default ``"h_motif"``).
        options: Options; ``kind`` and the keywords (``cuda_engine``) override its fields.
        resources: Default: the resources the hypergraph was built with.
        **kwargs: Option fields.

    Returns:
        The counts, matching ``hypergraph.version``.

    Raises:
        InvalidArgumentError: an unknown pattern or engine; a pattern that reads the line graph
            on a hypergraph built with ``line_graph="none"`` or ``store_v2h=False`` (the message
            names the property); ``"h_motif"`` or type 3 on a hypergraph of 2,147,483,647
            vertices.
        NotSupportedError: ``cuda_engine="operators"`` on cuda, or an unavailable backend.
        CapacityError: a count exceeds 2^64 - 1.
        OutOfMemoryError: host or device memory cannot be allocated (also a ``MemoryError``).
    """
    _check_hypergraph(hypergraph)
    if kind is not None:
        kwargs = {**kwargs, "kind": kind}
    opt = with_options(Options, options, kwargs, "triad_count.compute")
    res = resolve_on(hypergraph, resources)
    return Result._wrap(
        native.triad_count_compute(res._native, hypergraph._native, opt._to_native()), res
    )


def update(
    hypergraph: Hypergraph,
    batch: HyperedgeBatch,
    result: Result,
    *,
    resources: Resources | None = None,
) -> Stats:
    """Apply ``batch`` to ``hypergraph`` and update the counts ``result``.

    Postcondition: ``result`` equals ``compute(hypergraph, options=result.options)`` exactly.

    Raises:
        StaleResultError: ``result`` does not match the hypergraph's current state, or a failed
            update left it unusable.
        InvalidArgumentError: the batch is invalid for the hypergraph (nothing changed).
        NotSupportedError: incidence edits (0.4; nothing changed).
        CapacityError: a count exceeds 2^64 - 1, or a bin changes by 2^63 or more (after the
            commit: ``result`` is left unusable; recompute it).
        OutOfMemoryError: host or device memory cannot be allocated (also a ``MemoryError``),
            with one of three outcomes: before the commit, ``hypergraph`` and ``result`` are
            unchanged; inside the commit, ``hypergraph`` is unusable (as for
            :meth:`Hypergraph.apply`) and ``result`` keeps the counts before the batch (recompute
            it once the hypergraph is built again); after the commit, ``hypergraph`` holds the
            new version and ``result`` is left unusable (recompute it).
        InternalError: a bin would become negative (a library bug; ``result`` is left unusable).
    """
    _check_hypergraph(hypergraph)
    if not isinstance(result, Result):
        raise TypeError("dyng.triad_count.update: result must be a dyng.triad_count.Result")
    if not isinstance(batch, HyperedgeBatch):
        raise TypeError("dyng.triad_count.update: batch must be a dyng.HyperedgeBatch")
    res = resolve_on(hypergraph, resources)
    nb = batch._native_for(hypergraph.weighted)
    num_bins = result.num_bins
    with writing([result], res):
        out = native.triad_count_update(res._native, hypergraph._native, nb, result._native)
    return Stats._from_native(out, num_bins)
