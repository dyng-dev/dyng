# Reader fuzzers

libFuzzer targets for every file reader of the library (PLAN Sections 5.7 and 8.1). The design,
the CI jobs and what to do with a finding are on the developer page
[Robustness checks](../../docs/developer/robustness.md).

| Target | Reader | Options (the bytes of the input's first line) | Files |
|---|---|---|---|
| `fuzz_matrix_market` | `io::read_matrix_market()` | weight source, self-loops, sort and dedupe, random K, vertex type | 1 |
| `fuzz_edge_list` | `io::read_edge_list()` (also its Matrix Market path) | K, id mapping, index base, symmetrize, self-loops, duplicates, threads, types | 1 |
| `fuzz_csr_triplet` | `io::read_csr_triplet()` | K, graph type | 3: RowPtr, ColInd, Values |
| `fuzz_legacy_batch` | `io::read_legacy_batch()` (MOSP `insert.txt` / `delete.txt`) | K, vertex count, `mosp_lenient`, vertex type | 2: insert, delete |
| `fuzz_batch_text` | `io::read_batches()` (`.dgt`) | K, vertex count, batch type | 1 |
| `fuzz_result_io` | `io::read_distances()`, `io::read_parents()` | reader, vertex count | 1 |

An input is `<option line>\n<file 1>[\n@@\n<file 2>...]`: byte i of the first line selects
option i (a digit counts as its value), and lines that are exactly `@@` separate the files of a
reader that opens several (`fuzz_input.hpp`). Each target checks that a rejected input raises only
the exceptions the reader documents, that an accepted one is consistent, and, where a writer
exists, that writing and reading it again gives the same data (any exception in that round trip
is a finding; CTest `fuzz.selftest.<target>` plants a writer bug and checks that the replay
reports it).

```bash
ci/fuzz.sh                       # build (preset fuzz, Clang), replay the corpus, 60 s per target
ci/fuzz.sh --time 600 edge_list  # one target for 10 minutes
```

- `corpus/<target>/`: the committed seed corpus (small, readable inputs; runs start from a copy in
  `build/fuzz-runs`, so new inputs never land in the repository).
- `regressions/<target>/`: the reproducers of fixed findings; every test build replays them
  (`ctest -L fuzz`), and `cpp/tests/io/fuzz_regression_test.cpp` states what the reader must do.
- `replay_main.cpp`: the driver of `dyng_fuzz_replay_<target>`, the same targets built with any
  compiler and run once over `corpus/` and `regressions/` (CTest `fuzz.replay.<target>`).

The binary batch files (`.dgb`) and the hypergraph files (`.hg`) come with M8 (0.3); each reader
added then gets its target here.
