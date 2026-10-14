# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Seeded generators of hypergraphs and hyperedge batches
(C++ ``<dyng/generators/hypergraphs.hpp>``).

The streams use distributions owned by the library (``std::mt19937_64`` with the library's own
uniform-integer and sampling algorithms), so they are bit-exact per (seed, major version) with
every compiler and standard library; golden hashes in the tests pin them.
"""

from __future__ import annotations

from typing import Literal

from .._backend import native
from .._convert import as_bool, as_int, enum_member
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
    "HyperedgeBatchMixName",
    "uniform_hypergraph",
    "uniform_hypergraph_arrays",
    "random_hyperedge_batch",
]

HyperedgeBatchMixName = Literal[
    "insert_only",
    "delete_only",
    "mixed",
    "reuse",
    "heavy_delete",
    "delete_all",
    "reinsert",
    "larger",
    "duplicates",
    "dead_ids",
    "skewed",
    "limits",
    "empty",
]
"""The kinds of batches :func:`random_hyperedge_batch` draws (C++ ``hyperedge_batch_mix``):
``insert_only``, ``delete_only``, ``mixed``, ``reuse`` (delete-then-reuse within one batch),
``heavy_delete``, ``delete_all``, ``reinsert``, ``larger`` (a batch larger than the hypergraph),
``duplicates`` (repeated members and parallel hyperedges), ``dead_ids`` (deletions apply()
skips), ``skewed`` (one hyperedge of ``skewed_cardinality`` members), ``limits`` (ids at the index
limits) and ``empty``."""


def uniform_hypergraph_arrays(
    *,
    num_vertices: int,
    num_hyperedges: int,
    min_cardinality: int = 1,
    max_cardinality: int = 4,
    weighted: bool = False,
    min_weight: int = 1,
    max_weight: int = 100,
    seed: int = 1,
) -> HyperedgeList:
    """A uniform random hypergraph as arrays: hyperedge i has a cardinality uniform in
    [min_cardinality, max_cardinality] (clamped to n) and uniformly drawn distinct members,
    sorted (C++ ``generators::uniform_hypergraph``).

    Args:
        num_vertices: n.
        num_hyperedges: Hyperedges to draw.
        min_cardinality: Smallest cardinality (>= 1).
        max_cardinality: Largest cardinality.
        weighted: Draw one weight per hyperedge in [min_weight, max_weight].
        min_weight: Smallest weight (>= 1).
        max_weight: Largest weight.
        seed: The seed of the stream.
    """
    opt = native.UniformHypergraphOptions()
    opt.num_vertices = as_int(num_vertices, "uniform_hypergraph: num_vertices")
    opt.num_hyperedges = as_int(num_hyperedges, "uniform_hypergraph: num_hyperedges")
    opt.min_cardinality = as_int(min_cardinality, "uniform_hypergraph: min_cardinality")
    opt.max_cardinality = as_int(max_cardinality, "uniform_hypergraph: max_cardinality")
    opt.min_weight = as_int(min_weight, "uniform_hypergraph: min_weight")
    opt.max_weight = as_int(max_weight, "uniform_hypergraph: max_weight")
    opt.seed = as_int(seed, "uniform_hypergraph: seed")
    gen = (
        native.uniform_hypergraph_i32_i32
        if as_bool(weighted, "uniform_hypergraph: weighted")
        else native.uniform_hypergraph_i32_u
    )
    n, offsets, members, weights = gen(opt)
    return HyperedgeList(int(n), offsets, members, weights)


def uniform_hypergraph(
    *,
    num_vertices: int,
    num_hyperedges: int,
    min_cardinality: int = 1,
    max_cardinality: int = 4,
    weighted: bool = False,
    min_weight: int = 1,
    max_weight: int = 100,
    seed: int = 1,
    properties: HypergraphProperties | None = None,
    line_graph: Literal["none", "automatic", "sorted_merge", "slack_csr"] | None = None,
    id_reuse: Literal["erase_first", "insert_first"] | None = None,
    capacity: CapacityPolicy | Literal["default", "escher_gpu", "mosp_escher"] | None = None,
    resources: Resources | None = None,
) -> Hypergraph:
    """A uniform random hypergraph (:func:`uniform_hypergraph_arrays`) as a
    :class:`dyng.Hypergraph` (hyperedge i gets id i).

    Args:
        num_vertices: See :func:`uniform_hypergraph_arrays`.
        num_hyperedges: See :func:`uniform_hypergraph_arrays`.
        min_cardinality: See :func:`uniform_hypergraph_arrays`.
        max_cardinality: See :func:`uniform_hypergraph_arrays`.
        weighted: See :func:`uniform_hypergraph_arrays`.
        min_weight: See :func:`uniform_hypergraph_arrays`.
        max_weight: See :func:`uniform_hypergraph_arrays`.
        seed: See :func:`uniform_hypergraph_arrays`.
        properties: The hypergraph's properties (and the keywords that override them).
        line_graph: :attr:`dyng.HypergraphProperties.line_graph`.
        id_reuse: :attr:`dyng.HypergraphProperties.reuse`.
        capacity: A :class:`dyng.CapacityPolicy` or a preset name.
        resources: The resources the hypergraph belongs to.
    """
    lst = uniform_hypergraph_arrays(
        num_vertices=num_vertices,
        num_hyperedges=num_hyperedges,
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        weighted=weighted,
        min_weight=min_weight,
        max_weight=max_weight,
        seed=seed,
    )
    props = make_hypergraph_properties(
        properties, line_graph=line_graph, id_reuse=id_reuse, capacity=capacity
    )
    return Hypergraph._from_list(tuple(lst), props, resolve(resources))


def random_hyperedge_batch(
    hypergraph: Hypergraph,
    *,
    mix: HyperedgeBatchMixName = "mixed",
    size: int = -1,
    min_cardinality: int = 1,
    max_cardinality: int = 4,
    skewed_cardinality: int = 1100,
    min_weight: int = 1,
    max_weight: int = 100,
    seed: int = 1,
    resources: Resources | None = None,
) -> HyperedgeBatch:
    """A random hyperedge batch for the current state of ``hypergraph`` (its rows and weights).

    Live hyperedges are chosen uniformly; deleted ids appear in the batch first, insertions after
    them. The batch names ids of this state, so apply it to this hypergraph; its inserted ids are
    the hypergraph's choice (:attr:`dyng.HyperApplySummary.inserted_ids`).

    Args:
        hypergraph: The hypergraph.
        mix: The kind of batch (:data:`HyperedgeBatchMixName`).
        size: k of the mix (-1: max(1, live hyperedges / 10)).
        min_cardinality: Fresh hyperedges' smallest cardinality.
        max_cardinality: Fresh hyperedges' largest cardinality.
        skewed_cardinality: The large hyperedge of ``mix="skewed"``.
        min_weight: Weights of fresh hyperedges (weighted hypergraphs).
        max_weight: See ``min_weight``.
        seed: The seed of the stream.
        resources: Resources of the hypergraph's backend (default: the hypergraph's).
    """
    if not isinstance(hypergraph, Hypergraph):
        raise TypeError("random_hyperedge_batch: hypergraph must be a dyng.Hypergraph")
    opt = native.HyperedgeBatchOptions()
    opt.mix = enum_member(native.HyperedgeBatchMix, mix, "random_hyperedge_batch: mix")
    opt.size = as_int(size, "random_hyperedge_batch: size")
    opt.min_cardinality = as_int(min_cardinality, "random_hyperedge_batch: min_cardinality")
    opt.max_cardinality = as_int(max_cardinality, "random_hyperedge_batch: max_cardinality")
    opt.skewed_cardinality = as_int(
        skewed_cardinality, "random_hyperedge_batch: skewed_cardinality"
    )
    opt.min_weight = as_int(min_weight, "random_hyperedge_batch: min_weight")
    opt.max_weight = as_int(max_weight, "random_hyperedge_batch: max_weight")
    opt.seed = as_int(seed, "random_hyperedge_batch: seed")
    res = resolve(resources, hypergraph.resources)
    parts = native.random_hyperedge_batch(res._native, hypergraph._native, opt)
    return HyperedgeBatch._from_arrays(parts)
