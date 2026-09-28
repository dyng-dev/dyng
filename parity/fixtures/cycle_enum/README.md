# CycleEnum fixtures from CycleEnumeration-GPU@0a976ad

`make_cycle_enum_fixtures.sh` regenerates `cpp/tests/data/cycle_enum/` from the pinned original.
It uses the patched scratch copy that `parity/build_reference.sh` builds under
`$DYNG_SCRATCH/ref/CycleEnumeration-GPU@0a976ad/patched` (default `~/Projects/dyng-work`; the
original repository is only read with `git archive`). Its exporter
`parity_export/bin/export_cycle_enum` (source `parity/exporters/cycle_enum/export_cycle_enum.cpp`)
calls the original functions unchanged.

| Fixture | Inputs | Expected output (from the original) |
|---|---|---|
| `parser/` | the original's two test files, a TUDataset-style `*_A.txt`, a file with comments, CRLF, tabs, commas, signs, timestamps, duplicates, self-loops and 10^12 ids, Matrix Market files of every symmetry, a sparse-id file | `<file>.parse` (`read_temporal_graph`: compact ids, grouped edges, timestamps) and `<file>.csr` (`build_directed_graph(read_graph_view)`) |
| `errors/` | 15 malformed or borderline files | `ok`, or `error <line>` of the original's `GraphParseError` |
| `cases/` | 80 random graphs (2..11 vertices) with arbitrary batches: absent deletions, present insertions, self-loops, duplicates, delete-then-reinsert pairs, ids past the graph | `prepare_batch`, `apply_batch` (checked equal for the raw and the prepared batch), and the subset-DP oracle and brute-force histograms before and after |
| `generator/` | 4 graphs and 22 (deletions, insertions, seed, window) cases | `generate_batch`, or `error` for its `std::invalid_argument` |
| `datasets.txt` (`--datasets`) | DD, GitHub, Twitch, COLLAB and the timestamped CollegeMsg, email-Eu-core-temporal, sx-mathoverflow under `$DYNG_SCRATCH/datasets/cycle` | FNV-1a digests of the parser output, the CSR (every dataset), the batches (TUDataset graphs) of seed 1 (1K+1K, 25K+25K, 50K+50K; windows) and the normalized batch and CSR after them |

The consumers are `cpp/tests/io/edge_list_io_test.cpp`, `cpp/tests/graph/set_semantics_test.cpp`,
`cpp/tests/generators/cycle_enum_batch_test.cpp`, `cpp/tests/testing/cycle_oracle_test.cpp` (label
`cpu`) and `cpp/tests/parity/cycle_enum_datasets_test.cpp` (label `parity`; it skips the datasets
it cannot find). The script is deterministic: running it again leaves `git status` clean.
