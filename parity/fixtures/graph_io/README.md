# graph/io fixtures from MOSP-OpenMP@c352151

`make_graph_io_fixtures.sh` regenerates `cpp/tests/data/mosp_graph_io/` from the pinned
original. It uses the patched scratch copy that `parity/build_reference.sh` builds under
`$DYNG_SCRATCH/ref/MOSP-OpenMP@c352151/patched` (default `~/Projects/dyng-work`; the original
repository is only read with `git archive`). Its exporter `parity_export/bin/export_graph_io`
(source `parity/exporters/mosp/export_graph_io.cpp`) calls the original functions unchanged
(`generateGraphCSR`, `generateChangedEdges`, `readCsrGraph`, `readChangeBatch`,
`applyChangeBatch`, `writeCsrGraph`, `transposeCsrGraph`).

| Fixture | Inputs | Expected output (from the original) |
|---|---|---|
| `testCase0` .. `testCase9` | the 10 `generateTestCases` cases tracked in the pinned commit | `applied/` (applyChangeBatch + writeCsrGraph, weightIncreaseMask bits, transposeCsrGraph); `updateGraphCSR/` (the tracked updateGraphCSR output) |
| `r00` .. `r11` | `generateGraphCSR` + `generateChangedEdges` with duplicates, existing edges, self-loops, K = 1..5 and 32 | `applied/` |
| `h0` .. `h4` | hand-written: parallel edges, self-loops and unsorted rows; delete-all; an empty batch; rows without out-edges; blank lines | `applied/` |
| `mtx/` | two small Matrix Market files | `mospPrep mtx2csr` output for K = 3 / 1 (weights [1, 100], seed 12345) and K = 4 ([1, 2^31 - 1], seed 7) |

The consumer is `cpp/tests/graph/mosp_apply_fixture_test.cpp` (CTest labels `cpu graph`). The
script is deterministic: running it again leaves `git status` clean. The fixtures are the small
committed subset of the parity harness; the full golden corpus (388 cases, outside the
repository) is described in `parity/README.md`.
