# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""first_update: the program of docs/getting_started/first_update_python.md.

It builds a four-vertex graph, computes shortest paths from vertex 0, applies one batch with
dyng.sssp.update() and prints the statistics and the repaired tree, exactly as the C++ example
examples/cpp/first_update.cpp does. python/tests/test_examples.py runs it on both CPU backends.

    python first_update.py [sequential|openmp]
"""

import sys

import dyng


def main(backend: str = "openmp") -> None:
    res = dyng.Resources(backend)  # "sequential" or "openmp" in the CPU wheel

    # Edges 0->1 (4), 0->2 (1), 2->1 (2), 1->3 (1): sources, destinations, weights.
    g = dyng.Graph.from_edges([0, 0, 2, 1], [1, 2, 1, 3], [4, 1, 2, 1], resources=res)
    tree = dyng.sssp.compute(g, source=0)  # the canonical tree: lowest-id ties

    batch = dyng.EdgeBatch(insert=([2], [3], [1]), delete=([2], [1]))  # +2->3 (1), -2->1
    st = dyng.sssp.update(g, batch, tree)  # applies the batch to g and repairs the tree

    print(f"invalidated {st.invalidated}, affected {st.affected}")
    print("distances", *tree.distances.tolist())
    print("parents", *tree.parents.tolist())  # distances 0 4 1 2, parents -1 0 0 2


if __name__ == "__main__":
    main(*sys.argv[1:2])
