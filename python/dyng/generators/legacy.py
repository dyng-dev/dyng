# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.generators.legacy: the original repositories' generators, bit-exact for a fixed seed.

- :func:`mosp_changes`: MOSP's change generator (``mospPrep changes`` of MOSP-OpenMP@c352151 and
  MOSP-CUDA@e220ee2).
- :func:`cycle_enum_batch`: CycleEnumeration-GPU@0a976ad's batch generator (``cycle-enum --task
  update``).
- :func:`escher_hypergraph` and :class:`EscherBatchGenerator` (0.3): ESCHER-GPU@abcf9b2's
  synthetic hypergraph (``hyperedge2vertex``) and batch generator (``BatchGenerator``), drawing
  from one shared :class:`EscherEngine` (its ``std::mt19937_64``) in ESCHER-GPU's order.

The streams depend on ``std::mt19937`` / ``std::mt19937_64`` and on libstdc++'s distributions,
which the library reproduces itself; the modes that shuffle (``"targeted"``, ``"increase"``)
need a build against libstdc++ (the Linux wheels are).
:func:`standard_library_matches_legacy_streams` checks the standard library at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .._backend import native
from .._convert import as_bool, as_int, as_seed, copy_fields, enum_member
from ..batch import EdgeBatch
from ..errors import NotSupportedError
from ..graph import Graph
from ..hypergraph import (
    CapacityPolicy,
    HyperedgeBatch,
    HyperedgeList,
    Hypergraph,
    HypergraphProperties,
    make_hypergraph_properties,
)
from ..resources import Resources, resolve

__all__ = [
    "MospChangeReport",
    "mosp_changes",
    "cycle_enum_batch",
    "EscherEngine",
    "escher_hypergraph",
    "escher_hypergraph_arrays",
    "EscherBatchGenerator",
    "standard_library_matches_legacy_streams",
]


@dataclass(frozen=True)
class MospChangeReport:
    """What :func:`mosp_changes` did (the report line of ``mospPrep changes``).

    Attributes:
        inserts: Insertions of new edges.
        reweights: Insertions that overwrite existing edges.
        deletes: Deletions kept.
        requested_deletes: Deletions before the safe filter.
        safe_rounds: Rounds of the safe filter.
        safe: The safe filter ran.
        centre: Centre of a local batch, -1 otherwise.
        local_hops: Radius of a local batch.
        region: Vertices of the local region.
        summary: The report in ``mospPrep changes`` format.
    """

    inserts: int
    reweights: int
    deletes: int
    requested_deletes: int
    safe_rounds: int
    safe: bool
    centre: int
    local_hops: int
    region: int
    summary: str


def mosp_changes(
    graph: Graph,
    *,
    num_changes: int,
    insertion_percentage: float = 50.0,
    mode: Literal["uniform", "targeted", "reweight", "increase"] = "uniform",
    weight_min: int = 1,
    weight_max: int = 100,
    seed: int = 1,
    local_hops: int = 0,
    safe_deletions: bool = False,
    source: int = 0,
) -> tuple[EdgeBatch, MospChangeReport]:
    """MOSP's change generator (``generateChangeBatch``), bit-exact for a fixed seed.

    The batch has one weight per objective of ``graph`` and follows MOSP's file order
    (insertions, re-weighted edges, deletions); written with
    :func:`dyng.io.write_legacy_batch` it is byte-identical to ``mospPrep changes`` on the same
    CSR files.

    Args:
        graph: The graph (weighted; e.g. from :func:`dyng.io.read_csr_triplet`).
        num_changes: ``--changes``.
        insertion_percentage: ``--ins``.
        mode: ``--mode``.
        weight_min: ``--wmin``.
        weight_max: ``--wmax``.
        seed: ``--seed``.
        local_hops: ``--local``.
        safe_deletions: ``--safe``.
        source: ``--source``.

    Returns:
        The batch and the report.
    """
    if not isinstance(graph, Graph):
        raise TypeError("mosp_changes: graph must be a dyng.Graph")
    if not graph.weighted:
        raise NotSupportedError("mosp_changes: MOSP's generator needs a weighted graph")
    opt = native.MospChangeOptions()
    opt.num_changes = as_int(num_changes, "mosp_changes: num_changes")
    opt.insertion_percentage = float(insertion_percentage)
    opt.mode = enum_member(native.MospChangeMode, mode, "mosp_changes: mode")
    opt.weight_min = as_int(weight_min, "mosp_changes: weight_min")
    opt.weight_max = as_int(weight_max, "mosp_changes: weight_max")
    opt.seed = as_seed(seed, "mosp_changes: seed")
    opt.local_hops = as_int(local_hops, "mosp_changes: local_hops")
    opt.safe_deletions = as_bool(safe_deletions, "mosp_changes: safe_deletions")
    opt.source = as_int(source, "mosp_changes: source")
    parts, report = native.mosp_changes(graph._native, opt)
    names = (
        "inserts",
        "reweights",
        "deletes",
        "requested_deletes",
        "safe_rounds",
        "safe",
        "centre",
        "local_hops",
        "region",
    )
    return EdgeBatch._from_arrays(parts), MospChangeReport(
        **copy_fields(report, names), summary=str(report.summary())
    )


def cycle_enum_batch(
    graph: Graph,
    *,
    num_deletions: int = 0,
    num_insertions: int = 0,
    seed: int = 0,
    locality_window: int | None = None,
) -> EdgeBatch:
    """CycleEnumeration-GPU's batch generator (``generate_batch``), bit-exact for a fixed seed.

    Deletions are sampled without replacement from the existing edges, insertions from the
    non-edges; the batch is normalized (deletions, then insertions, each sorted by (source,
    destination)). For a weighted graph every insertion carries the weight 1.

    Args:
        graph: The graph (sorted rows, no parallel edges).
        num_deletions: ``--deletes``.
        num_insertions: ``--inserts``.
        seed: ``--batch-seed``.
        locality_window: ``--batch-locality`` (None: no window).
    """
    if not isinstance(graph, Graph):
        raise TypeError("cycle_enum_batch: graph must be a dyng.Graph")
    opt = native.CycleEnumBatchOptions()
    opt.num_deletions = as_int(num_deletions, "cycle_enum_batch: num_deletions")
    opt.num_insertions = as_int(num_insertions, "cycle_enum_batch: num_insertions")
    opt.seed = as_seed(seed, "cycle_enum_batch: seed")
    opt.locality_window = (
        -1
        if locality_window is None
        else as_int(locality_window, "cycle_enum_batch: locality_window")
    )
    parts: Any = native.cycle_enum_batch(graph._native, opt)
    return EdgeBatch._from_arrays(parts)


class EscherEngine:
    """The single ``std::mt19937_64`` of an ESCHER-GPU run (``std::mt19937_64(seed)`` for its
    ``--seed``): :func:`escher_hypergraph` and :class:`EscherBatchGenerator` draw from it in
    ESCHER-GPU's order, so the batches continue the hypergraph's stream as A's do.

    Args:
        seed: The seed (ESCHER-GPU's ``--seed``).
    """

    __slots__ = ("_native",)

    def __init__(self, seed: int) -> None:
        self._native = native.Mt19937_64(as_seed(seed, "EscherEngine: seed"))

    def __call__(self) -> int:
        """The engine's next 64-bit draw (advances it)."""
        return int(self._native())


def _escher_options(
    num_hyperedges: int, max_cardinality: int, min_vertex: int, max_vertex: int, what: str
) -> Any:
    o = native.EscherHypergraphOptions()
    o.num_hyperedges = as_int(num_hyperedges, f"{what}: num_hyperedges")
    o.max_cardinality = as_int(max_cardinality, f"{what}: max_cardinality")
    o.min_vertex = as_int(min_vertex, f"{what}: min_vertex")
    o.max_vertex = as_int(max_vertex, f"{what}: max_vertex")
    return o


def escher_hypergraph_arrays(
    engine: EscherEngine,
    *,
    num_hyperedges: int,
    max_cardinality: int,
    min_vertex: int,
    max_vertex: int,
    weighted: bool = False,
) -> HyperedgeList:
    """ESCHER-GPU's synthetic hypergraph (its positional arguments ``n m r1 r2``, with
    ``min_vertex = r1 - 1`` and ``max_vertex = r2 - 1``) as arrays, bit-exact for the engine's
    state: hyperedge i has a size drawn from [1, max_cardinality] and distinct members drawn
    from [min_vertex, max_vertex], sorted. The hypergraph has max_vertex + 1 vertices; hyperedge
    i is ESCHER-GPU's key i + 1 and vertex v its vertex v + 1.

    Args:
        engine: The shared engine (advanced by the draws).
        num_hyperedges: n.
        max_cardinality: m.
        min_vertex: r1 - 1.
        max_vertex: r2 - 1.
        weighted: Give every hyperedge the weight 1 (a weighted list).
    """
    if not isinstance(engine, EscherEngine):
        raise TypeError("escher_hypergraph: engine must be a dyng.generators.legacy.EscherEngine")
    opt = _escher_options(
        num_hyperedges, max_cardinality, min_vertex, max_vertex, "escher_hypergraph"
    )
    gen = native.escher_hypergraph_i32_i32 if weighted else native.escher_hypergraph_i32_u
    n, offsets, members, weights = gen(engine._native, opt)
    return HyperedgeList(int(n), offsets, members, weights)


def escher_hypergraph(
    engine: EscherEngine,
    *,
    num_hyperedges: int,
    max_cardinality: int,
    min_vertex: int,
    max_vertex: int,
    weighted: bool = False,
    properties: HypergraphProperties | None = None,
    line_graph: Literal["none", "automatic", "sorted_merge", "slack_csr"] | None = None,
    capacity: CapacityPolicy | Literal["default", "escher_gpu", "mosp_escher"] | None = None,
    resources: Resources | None = None,
) -> Hypergraph:
    """ESCHER-GPU's synthetic hypergraph (:func:`escher_hypergraph_arrays`) as a
    :class:`dyng.Hypergraph` under ESCHER-GPU's settings (``id_reuse="erase_first"``,
    ``capacity="escher_gpu"`` unless given), so that :class:`EscherBatchGenerator` reproduces
    ESCHER-GPU's batches on it.

    Args:
        engine: The shared engine.
        num_hyperedges: See :func:`escher_hypergraph_arrays`.
        max_cardinality: See :func:`escher_hypergraph_arrays`.
        min_vertex: See :func:`escher_hypergraph_arrays`.
        max_vertex: See :func:`escher_hypergraph_arrays`.
        weighted: See :func:`escher_hypergraph_arrays`.
        properties: The hypergraph's properties (default: erase_first, escher_gpu()).
        line_graph: :attr:`dyng.HypergraphProperties.line_graph`.
        capacity: A :class:`dyng.CapacityPolicy` or a preset name.
        resources: The resources the hypergraph belongs to.
    """
    lst = escher_hypergraph_arrays(
        engine,
        num_hyperedges=num_hyperedges,
        max_cardinality=max_cardinality,
        min_vertex=min_vertex,
        max_vertex=max_vertex,
        weighted=weighted,
    )
    if properties is None:
        properties = HypergraphProperties(reuse="erase_first", capacity=CapacityPolicy.escher_gpu())
    props = make_hypergraph_properties(properties, line_graph=line_graph, capacity=capacity)
    return Hypergraph._from_list(tuple(lst), props, resolve(resources))


class EscherBatchGenerator:
    """ESCHER-GPU's batch generator (``BatchGenerator::next``), bit-exact for the engine's state,
    reading the live state of a :class:`dyng.Hypergraph`.

    Each :meth:`next` reads the live ids in increasing order and their rows, chooses the
    deletions by a partial Fisher-Yates shuffle of the live ids (sorted afterwards) and the
    insertions as random hyperedges (``insert_mode="random"``, :func:`escher_hypergraph`'s
    distribution) or as copies of live hyperedges (``"sample"``). Driven against a hypergraph
    under ``id_reuse="erase_first"`` and ``capacity="escher_gpu"`` that applies every batch, it
    reproduces ESCHER-GPU's batches.

    Args:
        engine: The shared engine (kept alive by the generator).
        num_deletions: ``--deletions`` (-1 with ``num_insertions=-1``: ESCHER-GPU's demo batch).
        num_insertions: ``--insertions``.
        insert_mode: ``--insert-mode``: ``"random"`` or ``"sample"``.
        max_cardinality: The distribution of random insertions: largest size.
        min_vertex: Smallest vertex drawn.
        max_vertex: Largest vertex drawn (raised to the hypergraph's last vertex, as A does).
        batch_size: ``--batch-size N``: N / 2 deletions and N - N / 2 insertions (instead of
            ``num_deletions`` / ``num_insertions``).
    """

    __slots__ = ("_native", "_engine")

    def __init__(
        self,
        engine: EscherEngine,
        *,
        num_deletions: int = -1,
        num_insertions: int = -1,
        insert_mode: Literal["random", "sample"] = "random",
        max_cardinality: int = 1,
        min_vertex: int = 0,
        max_vertex: int = 0,
        batch_size: int | None = None,
    ) -> None:
        if not isinstance(engine, EscherEngine):
            raise TypeError(
                "EscherBatchGenerator: engine must be a dyng.generators.legacy.EscherEngine"
            )
        graph = _escher_options(0, max_cardinality, min_vertex, max_vertex, "EscherBatchGenerator")
        mode = enum_member(
            native.EscherInsertMode, insert_mode, "EscherBatchGenerator: insert_mode"
        )
        if batch_size is not None:
            opt = native.EscherBatchOptions.batch_size(
                as_int(batch_size, "EscherBatchGenerator: batch_size"), mode, graph
            )
        else:
            opt = native.EscherBatchOptions()
            opt.num_deletions = as_int(num_deletions, "EscherBatchGenerator: num_deletions")
            opt.num_insertions = as_int(num_insertions, "EscherBatchGenerator: num_insertions")
            opt.insert_mode = mode
            opt.graph = graph
        self._engine = engine
        self._native = native.EscherBatchGenerator(engine._native, opt)

    def next(self, hypergraph: Hypergraph, *, resources: Resources | None = None) -> HyperedgeBatch:
        """The next batch for the hypergraph's live state: the deletions (sorted ids), then the
        insertions in ESCHER-GPU's order."""
        if not isinstance(hypergraph, Hypergraph):
            raise TypeError("EscherBatchGenerator.next: hypergraph must be a dyng.Hypergraph")
        res = resolve(resources, hypergraph.resources)
        return HyperedgeBatch._from_arrays(self._native.next(res._native, hypergraph._native))


def standard_library_matches_legacy_streams() -> bool:
    """Whether the standard library the native module was built with draws the legacy streams
    (libstdc++ 11 or later: True; the streams themselves are the library's own reproduction)."""
    return bool(native.standard_library_matches_legacy_streams())
