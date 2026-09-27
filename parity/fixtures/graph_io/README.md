# graph/io fixtures from MOSP-OpenMP@c352151

`make_graph_io_fixtures.sh` regenerates `cpp/tests/data/mosp_graph_io/` from the pinned
original. The original repository is only read with `git archive`; the copy is built out of
tree under `$DYNG_SCRATCH/runs/graph-io-fixtures/` (default `~/Projects/dyng-work`).
`export_graph_io.cpp` is a small driver compiled against the original sources; it calls the
original functions unchanged (`generateGraphCSR`, `generateChangedEdges`, `readCsrGraph`,
`readChangeBatch`, `applyChangeBatch`, `writeCsrGraph`, `transposeCsrGraph`).

| Fixture | Inputs | Expected output (from the original) |
|---|---|---|
| `testCase0` .. `testCase9` | the 10 `generateTestCases` cases tracked in the pinned commit | `applied/` (applyChangeBatch + writeCsrGraph, weightIncreaseMask bits, transposeCsrGraph); `updateGraphCSR/` (the tracked updateGraphCSR output) |
| `r00` .. `r11` | `generateGraphCSR` + `generateChangedEdges` with duplicates, existing edges, self-loops, K = 1..5 and 32 | `applied/` |
| `h0` .. `h4` | hand-written: parallel edges, self-loops and unsorted rows; delete-all; an empty batch; rows without out-edges; blank lines | `applied/` |
| `mtx/` | two small Matrix Market files | `mospPrep mtx2csr` output for K = 3 / 1 (weights [1, 100], seed 12345) and K = 4 ([1, 2^31 - 1], seed 7) |

The consumer is `cpp/tests/graph/mosp_apply_fixture_test.cpp` (CTest labels `cpu graph`). The
script is deterministic: running it again leaves `git status` clean. The full parity harness
(`parity/build_reference.sh`, golden export, comparison script) supersedes the ad-hoc archive
directory used here; the fixtures themselves stay as small committed test data.
