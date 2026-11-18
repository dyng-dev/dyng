# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.testing's harmonic solution (0.4): label_propagation's oracle, re-exported by
:mod:`dyng.testing` (the module itself is private)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _dtypes
from .._backend import native
from .._convert import as_int
from ..graph import Graph

__all__ = ["HarmonicResult", "harmonic_solution"]


@dataclass(frozen=True)
class HarmonicResult:
    """The converged harmonic solution F* of a float graph (C++ ``testing::harmonic_result``;
    0.4): label_propagation's oracle.

    Attributes:
        solution: F* per vertex (float64): a present seed holds its class, a determined vertex
            the solution, every other vertex 0.5.
        determined: Per vertex, whether it is present, unlabelled and in a component with a
            present seed (bool).
        converged: The preconditioned conjugate gradient reached ``relative_tolerance``.
        relative_residual: ``|r| / |b|`` at the end.
        iterations: PCG iterations.
        num_seeds: Present vertices with a seed label.
        num_unlabelled: Present vertices without one.
        num_determined: Unlabelled vertices of seeded components.
        num_excluded: Unlabelled vertices of seed-free components.
    """

    solution: np.ndarray
    determined: np.ndarray
    converged: bool
    relative_residual: float
    iterations: int
    num_seeds: int
    num_unlabelled: int
    num_determined: int
    num_excluded: int

    def error(self, labels: Any) -> tuple[float, float]:
        """(mean, max) of ``|F - F*|`` over the determined vertices (0.0 when there are none)."""
        from ..array import Array

        f = labels.to_numpy() if isinstance(labels, Array) else np.asarray(labels)
        if f.shape != self.solution.shape:
            raise ValueError(f"error: {f.size} labels for {self.solution.size} vertices")
        d = np.abs(f.astype(np.float64) - self.solution)[self.determined]
        return (float(d.mean()), float(d.max())) if d.size else (0.0, 0.0)


def harmonic_solution(
    graph: Graph,
    seed_labels: Any,
    *,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 100000,
    num_threads: int = 1,
    resources: Any = None,
) -> HarmonicResult:
    """The harmonic solution of a float graph's current edges (C++ ``testing::harmonic_solution``,
    LabelPropagation-CUDA's checker; 0.4): label_propagation's oracle.

    The present vertices are the active ones (every vertex of a graph with compact rows); on each
    component with a present seed, F* solves the harmonic equations (each unlabelled vertex the
    weighted mean of its neighbours) by a preconditioned conjugate gradient in double precision.
    It shares no code with :mod:`dyng.label_propagation`.

    Args:
        graph: A float32 graph.
        seed_labels: One seed label per vertex (int8: -1 none, else 0 or 1), e.g.
            ``result.seed_labels`` of :mod:`dyng.label_propagation`.
        relative_tolerance: Stop when ``|r| <= relative_tolerance * |b|``.
        max_iterations: The PCG iteration cap (> 0).
        num_threads: Host threads (>= 1); the result does not depend on it.
        resources: The resources of the graph's host copy (default: the graph's).

    Returns:
        F*, with the counts.

    Example:
        >>> import numpy as np, dyng
        >>> g = dyng.Graph.from_edges([0, 1], [1, 2], np.array([1.0, 1.0], dtype=np.float32),
        ...                           directed=False)
        >>> f = dyng.testing.harmonic_solution(g, np.array([1, -1, 0], dtype=np.int8))
        >>> [round(float(x), 6) for x in f.solution]
        [1.0, 0.5, 0.0]
    """
    from ..array import Array
    from ..resources import resolve

    if not isinstance(graph, Graph):
        raise TypeError("harmonic_solution: expected a dyng.Graph")
    if graph.weight_dtype != _dtypes.FLOAT32:
        raise TypeError("harmonic_solution: expected a float32 graph (label_propagation's)")
    res = resolve(resources, graph.resources)
    host = seed_labels.to_numpy() if isinstance(seed_labels, Array) else seed_labels
    seeds = _dtypes.checked_cast(
        _dtypes.to_numpy(host, "harmonic_solution: seed_labels"),
        _dtypes.INT8,
        "harmonic_solution: seed_labels",
    )
    solution, determined, d = native.testing_harmonic_solution(
        res._native,
        graph._native,
        seeds,
        float(relative_tolerance),
        as_int(max_iterations, "harmonic_solution: max_iterations"),
        as_int(num_threads, "harmonic_solution: num_threads"),
    )
    return HarmonicResult(
        solution=solution,
        determined=determined.astype(bool),
        converged=bool(d["converged"]),
        relative_residual=float(d["relative_residual"]),
        iterations=int(d["iterations"]),
        num_seeds=int(d["num_seeds"]),
        num_unlabelled=int(d["num_unlabelled"]),
        num_determined=int(d["num_determined"]),
        num_excluded=int(d["num_excluded"]),
    )
