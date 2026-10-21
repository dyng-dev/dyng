# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""triad_count: count the triads of a .hg hypergraph by h-motif (30 bins) and StatHyper's type 1,
then apply the batches of a .dgt file to both results with one dyng.update() each.

    python triad_count.py <hypergraph.hg> <batches.dgt> [sequential|openmp|cuda]

It prints what examples/cpp/triad_count_update.cpp prints (the same lines, byte for byte).
"""

import sys

import dyng


def print_bins(label, values):
    """The non-zero bins as bin:value (bin k is ESCHER-GPU's h-motif k + 1)."""
    print(label + "".join(f" {k}:{v}" for k, v in enumerate(values.tolist()) if v))


hg_path, dgt_path = sys.argv[1], sys.argv[2]
backend = sys.argv[3] if len(sys.argv) > 3 else "sequential"
res = {
    "sequential": dyng.Resources.sequential,
    "openmp": dyng.Resources.openmp,
    "cuda": dyng.Resources.cuda,
}[backend]()

# [tutorial: read]
hg = dyng.io.read_hypergraph(
    hg_path, dedupe_hyperedges=True, line_graph="sorted_merge", resources=res
)  # file ids 1..n become 0..n-1
# [tutorial: read end]
# [tutorial: compute]
motifs = dyng.triad_count.compute(hg)  # kind="h_motif": 30 bins
type1 = dyng.triad_count.compute(hg, kind="stathyper_type1")  # one bin: the sum of C(|e|, 3)
# [tutorial: compute end]
print(f"{hg.num_hyperedges} hyperedges: {motifs.total} triads, type1 {type1.total}")
print_bins("  bins", motifs.counts)
batches = dyng.io.read_hyperedge_batches(dgt_path, num_vertices=hg.num_vertices)
for i, batch in enumerate(batches):
    # [tutorial: update]
    st, st1 = dyng.update(hg, batch, motifs, type1)  # one commit, both results updated
    ids = "".join(f" {x}" for x in st.batch.inserted_ids.tolist())
    print(f"batch {i}: {st.deleted_frontier} deleted, inserted ids{ids}")
    print(
        f"  h_motif -{st.triads_removed} +{st.triads_added} = {motifs.total}, "
        f"type1 -{st1.triads_removed} +{st1.triads_added} = {type1.total}"
    )
    print_bins("  delta", st.delta)
    # [tutorial: update end]
