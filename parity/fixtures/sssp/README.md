# sssp fixtures from MOSP-OpenMP@c352151

`make_sssp_fixtures.sh` regenerates `cpp/tests/data/mosp_sssp/` from the pinned original. It uses
the patched scratch copy that `parity/build_reference.sh` builds under
`$DYNG_SCRATCH/ref/MOSP-OpenMP@c352151/patched` (default `~/Projects/dyng-work`; the original
repository is only read with `git archive`). Its exporter `parity_export/bin/export_sssp` (source
`parity/exporters/mosp/export_sssp.cpp`) calls the original file-based updates unchanged
(`sequentialSOSPUpdate`, `parallelSOSPUpdate`).

For every case and objective the script writes:

| File | Produced by the original |
|---|---|
| `<case>/init/obj<k>/distancesOriginal.txt`, `SSSPTreeOriginal.txt` | `mospPrep init` (Dijkstra, lowest-id ties) |
| `<case>/updated/obj<k>/distancesUpdated.txt`, `SSSPTreeUpdated.txt` | `mosp` (mospUpdate -> sospUpdateCpu), run with `--validate` |
| `<case>/stats.txt` | the `invalidated` counter per objective, from the `mosp` report |

Before anything is written, four original implementations must agree byte for byte on the
updated trees: `mosp`, `parallelSOSPUpdate`, `sequentialSOSPUpdate` and `mospPrep expected`
(Dijkstra on the updated graph). This is the "cross-check the originals once" step of PLAN
Section 6.3 for the CPU implementations.

| Cases | Inputs |
|---|---|
| `testCase0` .. `testCase9`, `r00` .. `r11`, `h0` .. `h4` | the graph/io fixtures in `cpp/tests/data/mosp_graph_io/` (the 10 `generateTestCases` cases; generated cases with duplicates, existing edges, self-loops, K = 1..5 and 32; hand-written cases with parallel edges, unsorted rows, delete-all, an empty batch, blank lines) |
| `c2i_0` .. `c2i_2` | the count-to-infinity regressions of `mospTest` (`runRegressions`): `generateGraphCSR` + `generateChangedEdges` with the stress-test seeds (621705 / 250813: d(1) = 90, not 60) |
| `escher_disconnect`, `escher_delete_all`, `escher_ties` | the three cases of MOSP_ESCHER `tests/unit/test_mosp_update.cu` (d = 100, 101; delete-all; a new tight path with a lower parent id) |
| `ties_k2` | hand-written, K = 2: equal-distance parents, a tree-edge weight increase, a parallel edge, a re-inserted deleted edge |

`cases.txt` lists `<case> <input directory relative to cpp/tests/data> <K>`. The consumers are
`cpp/tests/algorithms/sssp/sssp_fixture_test.cpp` (every case, objective, backend and index type:
`compute` equals the init files, `update` from the init files equals the updated files, and
`invalidated` equals the original's counter) and the `compat_mosp.*` CTest cases of
`tools/compat` (the drop-in driver `dyng-compat-mosp` writes the same files).

The script is deterministic: running it again leaves `git status` clean. The fixtures (about
39 KB) are the small committed subset of the parity harness; the full golden corpus (495 cases,
outside the repository) is described in `parity/README.md`.
