# Tutorials

Lessons that take you through a complete task, step by step.

```{toctree}
:maxdepth: 1

sssp_mosp_files
your_first_dynamic_algorithm
hypergraph_triads
count_a_new_hypergraph_pattern
```

{doc}`hypergraph_triads` (0.3) does its task in C++ and in Python. For the graph algorithms,
until their Python tutorials exist, {doc}`../getting_started/first_update_python` and the Python
sections of {doc}`../algorithms/sssp` and {doc}`../algorithms/cycle_count` show the steps.

Planned, each written when its code exists:

| Tutorial | Release |
|---|---|
| Dynamic SSSP in Python in 10 minutes | later (planned for 0.2.x, which was not made) |
| Cycle counting on a changing graph | later (planned for 0.2.x, which was not made) |
| **Aggregate deltas** (`triangle_delta`: signed recounts with an ownership rule; the code and its page exist since 0.2) | later |
| MOSP with preferences | later (planned for 0.2.x, which was not made) |
| **Hypergraph triads** (`triad_count`, in C++ and Python): delivered, {doc}`hypergraph_triads` | 0.3 |
| **Count a new hypergraph pattern (an Op)** (`experimental::count_local_patterns`): delivered, {doc}`count_a_new_hypergraph_pattern` | 0.3 |
| Streaming label propagation | 0.4 |

{doc}`your_first_dynamic_algorithm` (0.2) builds `my_bfs`, a dynamic BFS, from a fresh clone to a
green conformance kit with `scripts/new_algorithm.py` and the fixed-point template. Its reference
solution is in `examples/tutorial_algorithms/my_bfs/`; the tutorial quotes it by marked ranges (a
change that removes a marker breaks the documentation build), and `ci/scaffold_check.sh` scaffolds
`my_bfs`, makes the tutorial's edits in the scaffold's files where the prose puts them
(`ci/tutorial_edits.py`) and runs its kit, so a framework or scaffold change that breaks the
tutorial, or a step whose prose does not build, breaks CI. The two teaching algorithms of the library, {doc}`../algorithms/dynamic_bfs`
and {doc}`../algorithms/triangle_delta` (maturity `tutorial`), are the finished versions on every
backend (ADR 0033).

{doc}`hypergraph_triads` (0.3) counts the triads of a `.hg` hypergraph by h-motif and by
StatHyper's type 1, applies the hyperedge batches of a `.dgt` file to both results with one
`dyng::update()`, and reads the deltas and the inserted ids, in C++ and in Python with the same
output; its code is `examples/cpp/triad_count_update.cpp` and `examples/python/triad_count.py`,
which CI runs on every backend (the C++ one built against the installed package) against the
committed expected output.

{doc}`count_a_new_hypergraph_pattern` (0.3) writes four Ops for
`experimental::count_local_patterns`, the extension point of `triad_count`'s wedge engine (the
line graph's triangles, the pairs of hyperedges sharing `k` vertices, triangles by the size of
their core, and pairs through a vertex), checks them against a brute force and `triad_count`, and
keeps them exact under hyperedge batches on the GPU and the CPU. Its code is in
`examples/tutorial_algorithms/local_patterns/`, built and run on every backend by the local GPU
gate (`ci/gpu_local.sh`: it needs nvcc and a GPU), while hosted CI runs its host half (the Ops'
callbacks driven by hand, without nvcc); the page quotes it by marked ranges, and `ci/tutorial_quotes.py`
(run by `ci/docs.sh`) pins every quote of the tutorials, so a quoted line that moves or changes
without its page fails the documentation check.
