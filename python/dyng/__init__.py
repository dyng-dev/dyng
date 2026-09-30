# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dynG: dynamic graph and hypergraph algorithms that update their results under batches.

A graph (:class:`Graph`) changes by batches of edge insertions and deletions
(:class:`EdgeBatch`); each algorithm computes a result once (``compute``) and keeps it current
batch by batch (``update``), faster than recomputing and with the same answer.

Quick start::

    import dyng
    g = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], [4, 1, 1, 5])
    tree = dyng.sssp.compute(g, source=0)
    batch = dyng.EdgeBatch(insert=([1], [3], [1]), delete=([0], [2]))
    stats = dyng.sssp.update(g, batch, tree)
    print(tree.distances.to_numpy(), stats.invalidated)

The algorithms of 0.1 are :mod:`dyng.sssp` (dynamic single-source shortest paths) and
:mod:`dyng.cycle_count` (exact directed simple-cycle histograms); :func:`update` updates several
results with one batch. Backends: ``Resources("sequential")`` and ``Resources("openmp")`` in this
wheel (the CUDA plugins follow in 0.1.x). Cite with :func:`citation`.
"""

from __future__ import annotations

from typing import Any

from . import cycle_count, generators, io, sssp, testing  # noqa: E402  (after the core names)
from . import errors as _errors  # registers the exception classes for the native module
from ._registry import AlgorithmInfo, algorithms, citation, citation_keys
from ._update import update
from .array import Array
from .batch import EdgeBatch
from .config import config, get_log_level, set_log_level, show_config, use_cpu_only
from .errors import (
    CapacityError,
    ConvergenceError,
    CudaError,
    Error,
    FileFormatError,
    InternalError,
    InvalidArgumentError,
    NotSupportedError,
    OutOfMemoryError,
    StaleResultError,
)
from .graph import CSR, ApplySummary, BatchSemantics, Graph, GraphProperties
from .profiler import Profiler, StageRecord, StageSample, profile
from .resources import Resources, get_default_resources, set_default_resources

del _errors


def __getattr__(name: str) -> Any:
    # dyng.__version__ comes from the native module, which is chosen on first use (not at
    # import: dyng.use_cpu_only() must still be able to choose it).
    if name == "__version__":
        from . import _version

        return _version.__getattr__("__version__")
    raise AttributeError(f"module 'dyng' has no attribute {name!r}")


__version__: str  # resolved by __getattr__ (the version string, PEP 440)

__all__ = [
    "__version__",
    # core
    "Resources",
    "get_default_resources",
    "set_default_resources",
    "Array",
    "Graph",
    "GraphProperties",
    "BatchSemantics",
    "ApplySummary",
    "CSR",
    "EdgeBatch",
    "update",
    # algorithms
    "sssp",
    "cycle_count",
    # modules
    "io",
    "generators",
    "testing",
    # profiling, registry, configuration
    "profile",
    "Profiler",
    "StageRecord",
    "StageSample",
    "algorithms",
    "AlgorithmInfo",
    "citation",
    "citation_keys",
    "show_config",
    "config",
    "set_log_level",
    "get_log_level",
    "use_cpu_only",
    # errors
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
