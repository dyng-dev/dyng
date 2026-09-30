# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.generators.legacy: the original repositories' generators, bit-exact for a fixed seed.

- :func:`mosp_changes`: MOSP's change generator (``mospPrep changes`` of MOSP-OpenMP@c352151 and
  MOSP-CUDA@e220ee2).
- :func:`cycle_enum_batch`: CycleEnumeration-GPU@0a976ad's batch generator (``cycle-enum --task
  update``).

The streams depend on ``std::mt19937`` and on libstdc++'s distributions, which the library
reproduces itself; the modes that shuffle (``"targeted"``, ``"increase"``) need a build against
libstdc++ (the Linux wheels are).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .._backend import native
from .._convert import copy_fields, enum_member
from ..batch import EdgeBatch
from ..errors import NotSupportedError
from ..graph import Graph

__all__ = ["MospChangeReport", "mosp_changes", "cycle_enum_batch"]


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
    opt.num_changes = int(num_changes)
    opt.insertion_percentage = float(insertion_percentage)
    opt.mode = enum_member(native.MospChangeMode, mode, "mosp_changes: mode")
    opt.weight_min = int(weight_min)
    opt.weight_max = int(weight_max)
    opt.seed = int(seed)
    opt.local_hops = int(local_hops)
    opt.safe_deletions = bool(safe_deletions)
    opt.source = int(source)
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
    opt.num_deletions = int(num_deletions)
    opt.num_insertions = int(num_insertions)
    opt.seed = int(seed)
    opt.locality_window = -1 if locality_window is None else int(locality_window)
    parts: Any = native.cycle_enum_batch(graph._native, opt)
    return EdgeBatch._from_arrays(parts)
