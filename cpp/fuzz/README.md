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
| `fuzz_batch_text` | `io::read_batches()`, `io::read_hyperedge_batches()` (`.dgt`) | K, vertex count, batch type (four graph, two hypergraph) | 1 |
| `fuzz_result_io` | `io::read_distances()`, `io::read_parents()` | reader, vertex count | 1 |
| `fuzz_hg` | `io::read_hypergraph()` (`.hg`) | relabelling, index base, dedupe, max_cardinality, weights (none, file, random, H-SOSP's option set), vertex count | 1 |
| `fuzz_batch_binary` | `io::batch_reader` (`.dgb`; the types its header names) | vertex count | 1 (binary) |

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

The `.dgb` seeds are written from the format's specification by `make_dgb_seeds.py` (Python's
`struct` and `zlib.crc32`, independent of the library's writer; `python3 cpp/fuzz/make_dgb_seeds.py`
rewrites `corpus/batch_binary` and `regressions/batch_binary`), plus one file the converter
`dyng-compat-escher` wrote from a committed trace; the `.hg` seeds are small hand-written files
covering the line rules and every option, plus the converter's `.hg` of the same trace. The
`regressions/` of these two targets hold inputs whose size fields lie (a record length of 2^40
bytes, a batch count of 2^50, an insertion count of 2^60 with a valid CRC32) and an `.hg` vertex
count beyond the vertex type: the class of finding of M6b, which the readers were written against
(`fuzz_regression_test.cpp` bounds their allocations). Every reader of 0.3 has its target here;
each reader added later gets one in the same pull request.
