# Tutorials

Lessons that take you through a complete task, step by step.

```{toctree}
:maxdepth: 1

sssp_mosp_files
your_first_dynamic_algorithm
```

Until the Python tutorials exist, {doc}`../getting_started/first_update_python` and the Python
sections of {doc}`../algorithms/sssp` and {doc}`../algorithms/cycle_count` show the same steps.

Planned, each written when its code exists:

| Tutorial | Release |
|---|---|
| Dynamic SSSP in Python in 10 minutes | 0.2.x |
| Cycle counting on a changing graph | 0.2.x |
| **Aggregate deltas** (`triangle_delta`: signed recounts with an ownership rule; the code and its page exist since 0.2) | later |
| MOSP with preferences | 0.2.x |
| Hypergraph triads, and counting a new hypergraph pattern | 0.3 |
| Streaming label propagation | 0.4 |

{doc}`your_first_dynamic_algorithm` (0.2) builds `my_bfs`, a dynamic BFS, from a fresh clone to a
green conformance kit with `scripts/new_algorithm.py` and the fixed-point template. Its reference
solution is in `examples/tutorial_algorithms/my_bfs/`; the tutorial quotes it by marked ranges (a
change that removes a marker breaks the documentation build), and `ci/scaffold_check.sh` scaffolds
`my_bfs`, puts the reference files in and runs its kit, so a framework change that breaks the
tutorial breaks CI. The two teaching algorithms of the library, {doc}`../algorithms/dynamic_bfs`
and {doc}`../algorithms/triangle_delta` (maturity `tutorial`), are the finished versions on every
backend (ADR 0033).
