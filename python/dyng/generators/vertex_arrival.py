# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Vertex-arrival streams for :mod:`dyng.label_propagation` (C++
``dyng::generators::generate_vertex_arrival_stream`` and
``generators::legacy::dynlp_vertex_arrival_stream``; 0.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _dtypes
from .._backend import native
from .._convert import as_int, as_seed
from ..batch import EdgeBatch
from ..errors import InvalidArgumentError
from ..resources import Resources, resolve

__all__ = ["VertexArrivalStream", "vertex_arrival_stream"]


@dataclass(frozen=True)
class VertexArrivalStream:
    """A vertex-arrival stream: a similarity graph and the batches that bring its vertices in
    (C++ ``generators::vertex_arrival_stream``; 0.4).

    The full graph's edges are not in any graph yet: the batches carry them. Batch b0 inserts the
    first ground truth (its vertices with their classes) and the edges among it; every later
    batch inserts arrivals (some with a ground-truth class), every full-graph edge whose later
    endpoint arrives in it, and deletes earlier vertices.

    The published setup (ADR 0071) is
    ``g = dyng.Graph.with_capacity(stream.row_capacity)``, then
    :func:`dyng.label_propagation.compute` on the empty graph, then one
    :func:`dyng.label_propagation.update` per batch with
    ``similarity_threshold=stream.similarity_threshold``.

    Attributes:
        row_ptr: The full graph's CSR row offsets (the edge offset dtype).
        col_ind: Its neighbours (int32; both directions of every edge).
        weights: Its edge weights (float32 in (0, 1]).
        true_labels: The class (0 or 1) of every vertex (int8).
        batches: The batches b0 .. b(B-1) (:class:`dyng.EdgeBatch` objects with the vertex
            arrays and float32 weights).
        row_capacity: The degree of every vertex in the full graph (the slots of
            :meth:`dyng.Graph.with_capacity`).
        similarity_threshold: The published tau: the average edge weight of the full graph
            (the original's ``computeAvgEdgeWeight``), a float32 given as its shortest decimal.
        max_batch_vertices: The largest number of arrivals of a batch.
        max_batch_edges: The largest number of edges of a batch.
    """

    row_ptr: np.ndarray
    col_ind: np.ndarray
    weights: np.ndarray
    true_labels: np.ndarray
    batches: list[EdgeBatch] = field(repr=False)
    row_capacity: np.ndarray = field(repr=False)
    similarity_threshold: float = 0.5
    max_batch_vertices: int = 0
    max_batch_edges: int = 0

    @property
    def num_vertices(self) -> int:
        """The vertices of the full graph."""
        return int(self.row_ptr.size) - 1

    @property
    def num_batches(self) -> int:
        """The number of batches."""
        return len(self.batches)

    def write_graph_files(self, directory: str | os.PathLike[str]) -> None:
        """Write the full graph as LabelPropagation-CUDA's text files (its ``writeGraphFiles``):
        ``<directory>/csrRowPtr.txt``, ``csrColInd.txt``, ``csrValues.txt`` (6 decimals) and
        ``csrTrueLabels.txt`` (``i class``), byte for byte as the original writes them into its
        ``output/graph/`` directory. The directory is created if missing."""
        prefix = os.path.join(os.fspath(directory), "csr")
        os.makedirs(os.fspath(directory), exist_ok=True)
        code = "i32" if self.row_ptr.dtype == _dtypes.INT32 else "i64"
        getattr(native, f"write_dynlp_graph_files_{code}")(
            prefix,
            np.ascontiguousarray(self.row_ptr),
            np.ascontiguousarray(self.col_ind, dtype=np.int32),
            np.ascontiguousarray(self.weights, dtype=np.float32),
            np.ascontiguousarray(self.true_labels, dtype=np.int8),
        )

    def __repr__(self) -> str:
        return (
            f"dyng.generators.VertexArrivalStream(num_vertices={self.num_vertices}, "
            f"num_edges={int(self.col_ind.size)}, num_batches={self.num_batches}, "
            f"similarity_threshold={self.similarity_threshold!r})"
        )


def _options(
    total_vertices: int,
    average_degree: int,
    num_batches: int,
    seed: int,
    ground_truth_fraction: float,
    deleted_fraction: float,
    what: str,
) -> Any:
    o = native.VertexArrivalOptions()
    o.total_vertices = as_int(total_vertices, f"{what}: total_vertices")
    o.average_degree = as_int(average_degree, f"{what}: average_degree")
    o.num_batches = as_int(num_batches, f"{what}: num_batches")
    o.seed = as_seed(seed, f"{what}: seed")
    o.ground_truth_fraction = float(ground_truth_fraction)
    o.deleted_fraction = float(deleted_fraction)
    return o


def _stream(parts: tuple[Any, ...]) -> VertexArrivalStream:
    row_ptr, col_ind, weights, true_labels, batches, cap, tau, max_v, max_e = parts
    return VertexArrivalStream(
        row_ptr=row_ptr,
        col_ind=col_ind,
        weights=weights,
        true_labels=true_labels,
        batches=[EdgeBatch._from_arrays(tuple(b)) for b in batches],
        row_capacity=cap,
        similarity_threshold=_dtypes.float32_value(tau),
        max_batch_vertices=int(max_v),
        max_batch_edges=int(max_e),
    )


def _edge_code(edge_dtype: Any, what: str) -> str:
    if edge_dtype is None:
        return "i32"
    dt = _dtypes.id_dtype(edge_dtype, f"{what}: edge_dtype")
    return "i32" if dt == _dtypes.INT32 else "i64"


def make_stream(
    kind: str,
    total_vertices: int,
    average_degree: int,
    num_batches: int,
    seed: int,
    ground_truth_fraction: float,
    deleted_fraction: float,
    edge_dtype: Any,
    resources: Resources | None,
    what: str,
) -> VertexArrivalStream:
    """The stream of ``kind`` (``"vertex_arrival_stream"`` or ``"dynlp_vertex_arrival_stream"``;
    internal)."""
    if not 0.0 < float(ground_truth_fraction) <= 1.0 or not 0.0 <= float(deleted_fraction) <= 1.0:
        raise InvalidArgumentError(
            f"{what}: ground_truth_fraction must be in (0, 1] and deleted_fraction in [0, 1]"
        )
    res = resolve(resources)
    opt = _options(
        total_vertices,
        average_degree,
        num_batches,
        seed,
        ground_truth_fraction,
        deleted_fraction,
        what,
    )
    fn = getattr(native, f"{kind}_{_edge_code(edge_dtype, what)}")
    return _stream(fn(res._native, opt))


def vertex_arrival_stream(
    *,
    total_vertices: int = 50000,
    average_degree: int = 5,
    num_batches: int = 10,
    seed: int = 42,
    ground_truth_fraction: float = 0.01,
    deleted_fraction: float = 0.09,
    edge_dtype: Any = None,
    resources: Resources | None = None,
) -> VertexArrivalStream:
    """A seeded vertex-arrival stream with the library's own distributions (C++
    ``generators::generate_vertex_arrival_stream``; 0.4).

    The model and structure of LabelPropagation-CUDA's generator (an Erdos-Renyi graph with
    similarity weights and two classes; vertex arrivals with a ground-truth share, and
    departures; :func:`dyng.generators.legacy.dynlp_vertex_arrival_stream` reproduces the
    original's stream bit for bit), drawn from distributions the library owns (ADR 0107):
    bit-exact per (seed, major version) on every platform and compiler, and independent of the
    thread count.

    Args:
        total_vertices: --totalNodes: vertices of the full graph (> 1).
        average_degree: --avgDegree: target average degree (> 0).
        num_batches: --numBatches: batches b0 .. b(num_batches - 1) (> 0).
        seed: --seed: the seed (0 .. 2^32 - 1).
        ground_truth_fraction: --gtFraction: the labelled share of the arrivals (b0: of all
            vertices).
        deleted_fraction: departures per batch, as a share of its arrivals.
        edge_dtype: The edge offset type of the full graph and of ``row_capacity`` (``"int32"``,
            the default, or ``"int64"``).
        resources: Host threads of the generation (the result does not depend on them).

    Returns:
        The stream.
    """
    return make_stream(
        "vertex_arrival_stream",
        total_vertices,
        average_degree,
        num_batches,
        seed,
        ground_truth_fraction,
        deleted_fraction,
        edge_dtype,
        resources,
        "vertex_arrival_stream",
    )
