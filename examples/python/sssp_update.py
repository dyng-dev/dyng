# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""sssp_update: shortest paths on MOSP's text files, updated by a batch.

Reads a graph in MOSP's CSR text format, computes a shortest-path tree from vertex 0, applies the
batch of MOSP's insert.txt / delete.txt files with dyng.sssp.update() and writes the updated tree
in MOSP's file format (byte-compatible with the original's output), like
examples/cpp/sssp_update.cpp. python/tests/test_examples.py compares the files with the original's.

    python sssp_update.py <csrPrefix> <insert.txt> <delete.txt> <outDir> [sequential|openmp]
"""

import sys
from pathlib import Path

import dyng


def main(prefix: str, insert: str, delete: str, out: str, backend: str = "sequential") -> None:
    res = dyng.Resources(backend)
    # MOSP's graph as it is: rows in file order, parallel edges kept (MOSP's applyChangeBatch).
    g = dyng.io.read_csr_triplet(prefix, properties="mosp_compatible", resources=res)
    tree = dyng.sssp.compute(g, source=0)

    batch = dyng.io.read_legacy_batch(
        insert, delete, num_weights=g.num_weights, num_vertices=g.num_vertices
    )
    with dyng.profile(res) as prof:
        st = dyng.sssp.update(g, batch, tree)
    print(
        f"+{st.batch.inserted_edges} -{st.batch.deleted_edges} edges, "
        f"invalidated {st.invalidated}, affected {st.affected}, engine {st.engine_used}"
    )

    Path(out).mkdir(parents=True, exist_ok=True)
    dyng.io.write_distances(Path(out) / "distancesUpdated.txt", tree.distances)
    dyng.io.write_parents(Path(out) / "SSSPTreeUpdated.txt", tree.parents)
    for stage in prof.stages:  # sssp.update, sssp.prepare, sssp.commit, sssp.loop, ...
        print(f"  {'  ' * stage.depth}{stage.name}: {stage.calls} call(s)")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("usage: sssp_update.py <csrPrefix> <insert.txt> <delete.txt> <outDir> [backend]")
    main(*sys.argv[1:6])
