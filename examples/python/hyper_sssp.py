# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""hyper_sssp: shortest hyperpaths of a weighted .hg hypergraph from two source vertices, kept up
to date by one dyng.update() per batch of a .dgt file (hyperedges and incidence edits).

    python hyper_sssp.py <hypergraph.hg> <batches.dgt> [sequential|cuda]

It prints what examples/cpp/hyper_sssp_update.cpp prints (the same lines, byte for byte).
"""

import sys

import dyng

INF = dyng.hyper_sssp.INFINITE_DISTANCE


def show(r):
    """The reached hyperedge ids as id:D/parent, and the distance of vertex 11."""
    d, p = r.hyperedge_distances.tolist(), r.hyperedge_parents.tolist()
    ids = "".join(f" {h}:{d[h]}/{p[h]}" for h in range(len(d)) if d[h] < INF)
    print(f"  from {r.source}:{ids} | vertex 11: {r.distance_to(11)}")


hg_path, dgt_path = sys.argv[1], sys.argv[2]
backend = sys.argv[3] if len(sys.argv) > 3 else "sequential"  # no openmp backend in 0.4
res = {"sequential": dyng.Resources.sequential, "cuda": dyng.Resources.cuda}[backend]()

# [tutorial: read]
hg = dyng.io.read_hypergraph(
    hg_path, weighted=True, storage="slack_csr", line_graph="slack_csr", resources=res
)  # '; w' on every line; file ids 1..n become 0..n-1
# [tutorial: read end]
# [tutorial: compute]
a = dyng.hyper_sssp.compute(hg, 0)  # from vertex 0
c = dyng.hyper_sssp.compute(hg, 9, work_budget=0.5)  # from vertex 9, a smaller work budget
# [tutorial: compute end]
print(f"{hg.num_hyperedges} hyperedges, {hg.num_vertices} vertices")
show(a)
show(c)
batches = dyng.io.read_hyperedge_batches(dgt_path, num_vertices=hg.num_vertices, weighted=True)
for i, batch in enumerate(batches):
    # [tutorial: update]
    sa, sc = dyng.update(hg, batch, a, c)  # one commit, both results updated
    s = sa.batch
    print(
        f"batch {i}: -{s.deleted_hyperedges} +{s.inserted_hyperedges} hyperedges, incidences "
        f"-{s.deleted_incidences} +{s.inserted_incidences}, line graph -{sa.line_graph_deletions} "
        f"+{sa.line_graph_insertions}, invalidated {sa.invalidated} and {sc.invalidated}"
    )
    # [tutorial: update end]
    show(a)
    show(c)
