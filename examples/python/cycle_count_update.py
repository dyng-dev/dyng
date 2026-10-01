# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""cycle_count_update: directed cycle histograms of a changing graph.

Counts the directed simple cycles of length 2..k of an edge-list graph (a TUDataset *_A.txt file
loads directly), applies a batch made by CycleEnumeration-GPU's generator with
dyng.cycle_count.update() and prints the updated histogram in the original's CSV format, like
examples/cpp/cycle_count_update.cpp. python/tests/test_examples.py compares the output with the
original `cycle-enum --task update`.

    python cycle_count_update.py <edges.txt> <k> <deletions> <insertions> <seed> [backend]
"""

import sys

import dyng


def main(
    path: str, k: str, deletions: str, insertions: str, seed: str, backend: str = "sequential"
) -> None:
    res = dyng.Resources(backend)
    # CycleEnumeration-GPU's graph: sorted rows, no parallel edges, set semantics for batches.
    g = dyng.io.read_edge_list(path, properties="cycle_enum_compatible", resources=res)
    hist = dyng.cycle_count.compute(g, max_length=int(k))  # exact static count

    batch = dyng.generators.legacy.cycle_enum_batch(
        g, num_deletions=int(deletions), num_insertions=int(insertions), seed=int(seed)
    )
    # Subtract the cycles through the deletions on G_t, apply, add those through the insertions.
    st = dyng.cycle_count.update(g, batch, hist)
    print(
        f"-{st.deletions} +{st.insertions} edges: {st.cycles_removed} cycles removed, "
        f"{st.cycles_added} added",
        file=sys.stderr,
    )
    print(dyng.io.histogram_csv(hist), end="")  # "# cycle_size, num_of_cycles" ... "Total, N"


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(
            "usage: cycle_count_update.py <edges.txt> <k> <deletions> <insertions> <seed> [backend]"
        )
    main(*sys.argv[1:7])
