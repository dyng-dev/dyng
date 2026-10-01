# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""mosp_update: multi-objective shortest paths on MOSP's text files, updated by a batch.

Reads a graph with K weight columns in MOSP's CSR text format, computes the K trees, the MOSP
tree and the path costs from vertex 0 with a preference vector, applies the batch of MOSP's
insert.txt / delete.txt files with dyng.mosp.update() and writes the combined graph's outputs in
MOSP's file formats (byte-compatible with the original `mosp --pref`), like
examples/cpp/mosp_update.cpp. python/tests/test_examples.py compares the files with the
original's.

    python mosp_update.py <csrPrefix> <insert.txt> <delete.txt> <outDir> <p1,..,pK> [backend]
"""

import sys
from pathlib import Path

import dyng


def main(prefix: str, insert: str, delete: str, out: str, pref: str, backend: str = "sequential"):
    res = dyng.Resources(backend)
    g = dyng.io.read_csr_triplet(prefix, properties="mosp_compatible", resources=res)
    preferences = [int(p) for p in pref.split(",")]
    paths = dyng.mosp.compute(g, source=0, preferences=preferences)

    batch = dyng.io.read_legacy_batch(
        insert, delete, num_weights=g.num_weights, num_vertices=g.num_vertices
    )
    st = dyng.mosp.update(g, batch, paths)
    print(
        f"K={paths.num_objectives} L={paths.preference_scale}: invalidated "
        f"{[o.invalidated for o in st.objectives]}, combined edges {st.combined_edges}, "
        f"affected {st.affected}"
    )
    last = g.num_vertices - 1
    print(f"vertex {last}: path costs {paths.path_costs.to_numpy()[last].tolist()}")

    comb = Path(out) / "combinedGraph"
    comb.mkdir(parents=True, exist_ok=True)
    dyng.io.write_distances(comb / "distancesCsr.txt", paths.combined_distances)
    dyng.io.write_parents(comb / "SSSPTreeCsr.txt", paths.combined_parents)
    dyng.io.write_path_costs(comb / "mospCosts.txt", paths.path_costs)  # (n, K) int64


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit("usage: mosp_update.py <csrPrefix> <insert.txt> <delete.txt> <outDir> <p1,..,pK>")
    main(*sys.argv[1:7])
