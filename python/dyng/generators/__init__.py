# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.generators: seeded generators of graphs, hypergraphs and batches (C++ ``dyng::generators``).

- :func:`uniform_hypergraph` and :func:`random_hyperedge_batch` (0.3): hypergraphs and hyperedge
  batches with distributions owned by the library, bit-exact per (seed, major version).
- :mod:`dyng.generators.legacy`: the bit-exact reproductions of the original tools' random
  streams, so the inputs of published experiments can be regenerated.
"""

from __future__ import annotations

from . import legacy
from .hypergraphs import (
    HyperedgeBatchMix,
    random_hyperedge_batch,
    uniform_hypergraph,
    uniform_hypergraph_arrays,
)

__all__ = [
    "legacy",
    "HyperedgeBatchMix",
    "uniform_hypergraph",
    "uniform_hypergraph_arrays",
    "random_hyperedge_batch",
]
