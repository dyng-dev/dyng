# Retrospective: M2a (`cycle_count` on the CPU backends)

Status: **in progress.** Each implementation step appends its section; the close-out adds the
milestone summary, the acceptance record, the lessons and the re-estimate.

## Step 1: graph, I/O, generator and oracle pieces (graph-io-gen, 2026-09-27)

Goal: the CycleEnumeration-GPU@0a976ad pieces that `cycle_count` stands on (PLAN Sections 5.2,
5.7, 5.8, 6.4.1 CycleEnum rows, 6.4.3): the unweighted graph, `graph_properties::
cycle_enum_compatible()` with `batch_semantics::set()`, the host sorted-row apply, the parser as
`io::read_edge_list`, `generate_batch` as a legacy generator, and the test oracles, each with
parity evidence against the pinned original.

### Done

- **Unweighted graphs.** `graph<int32, int32 | int64, unweighted>` instantiated
  (`DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE`: graph, host apply, device copy); `unweighted` compares
  equal to itself and `is_unweighted_v<W>` names it; `edge_batch<V, unweighted>` defaults to no
  weights and has `insert_edge(u, v)`; the constructors reject weight columns for an unweighted
  graph. `graph::from_edges` takes an edge list sorted by (source, destination) without repeats
  (what `read_edge_list` returns) as the CSR in one pass.
- **Set semantics** (ADR 0010 amendment). `batch_semantics::as_sets`, `batch_semantics::set()`,
  `graph_properties::cycle_enum_compatible()`. `cpp/src/graph/apply_set_host.cpp` ports
  `prepare_batch()` (Step 0: sorted, deduplicated lists without no-ops; a delete-then-reinsert
  pair stays) and `apply_batch()` (the merge of every touched row) straight; the new CSR is
  assembled in parallel blocks with the same bytes as the original's sequential loop. The
  normalized batch reaches the algorithms as `detail::apply_delta` (sorted lists: a change's
  position is its ownership id). The shared checks of both host applies moved to
  `graph/apply_common.hpp`.
- **`io::read_edge_list` / `write_edge_list`** (`cpp/src/io/edge_list_io.cpp`,
  `cpp/src/util/parallel_parts.hpp`): the original's parser (whole-file read, line-aligned parts
  on std::thread, `from_chars`, the first malformed line in file order, dense or sparse id
  compaction, one parallel sort) with the PLAN 5.7 generalizations (weight columns before the
  optional timestamp, `vertex_ids::as_is` with an index base, symmetrization, kept duplicates
  with their timestamps, kept self-loops, a thread cap).
- **`generators::legacy::cycle_enum_batch()`** (the original's `generate_batch`) with
  `detail::legacy_uniform_int64` and `detail::legacy_shuffle`, library-owned reproductions of
  libstdc++'s `uniform_int_distribution<size_t>` and `std::shuffle` for `std::mt19937_64`
  (checked against the real ones when the tests are built with libstdc++), so the batches do not
  depend on the standard library.
- **`dyng::testing`**: `oracle_simple_cycles` (the subset DP of the original's
  `tests/support/cycle_oracles.hpp`), `brute_force_simple_cycles` (its
  `count_simple_cycles_bruteforce`, with an explicit DFS stack) and `edge_set_after_batch` (the
  edge-set recount of its arbitrary-batch tests); histograms as `testing::cycle_histogram`
  (`counts[len]`).
- **Parity harness.** `references.toml` pins CycleEnumeration-GPU@0a976ad (baseline tag SHA
  recorded) with the build of its RESULTS.md (CMake Release, OpenMP and CUDA, sm_86);
  `build_reference.sh` builds both copies (the original's tests fetch googletest v1.14.0 from
  GitHub at configure time: recorded as an external input). The additive export patch compiles
  `parity/exporters/cycle_enum/export_cycle_enum.cpp` against the unchanged sources;
  `parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh` writes `cpp/tests/data/cycle_enum`
  (112 KB) and, with `--datasets`, the digests of `datasets.txt`.
- **Tests** (all green on `cpu-only` and `dev`; `ci/check.sh` passes): new suites in the graph,
  io, generators and testing executables (label `cpu`; those executables now hold 81, 38, 13 and
  6 cases), plus the dataset test (label `parity`, about 100 s in Debug; it skips missing
  datasets). The original's DirectedGraphTest, UpdateBatchValidationTest cases, GraphParserTest
  and BatchGeneratorTest are re-expressed.

### Parity evidence

- Normalized batch and CSR byte-equal to `prepare_batch()` + `apply_batch()` on 80 random
  fixture cases (arbitrary batches) for four graph types on both host backends, and on the
  generated batches of seed 1 (1K+1K, 25K+25K, 50K+50K) on DD, GitHub and Twitch (digests).
  The parallel assembly equals the sequential one and the edge-set model on 150K-vertex graphs.
- `read_edge_list` equals `read_temporal_graph` (vertices, their order, grouped edges,
  timestamps) and the CSR equals `build_directed_graph(read_graph_view)` on 9 fixture files and
  on DD, GitHub, Twitch and COLLAB (digests); the accept/reject decision and the error line equal
  the original's on 15 malformed or borderline files.
- `cycle_enum_batch` equals `generate_batch` on 22 fixture cases (windows 0, 1, 2, 6, 17, 30,
  39, 40, 50, 299; seed 2^64 - 1; the original's two errors) and on the 9 dataset batches and 3
  windowed DD batches (a window of 100 is the original's error there as well).
- The oracles equal the original's subset DP and brute force on the 80 cases before and after
  the batch, and each other on 300 random graphs with and without a length bound.

### Measured (informal; not a gate)

Release builds, exclusive perf lock, medians of 5 (9 for apply) on the development machine;
original = its functions compiled with its Release flags:

| Graph | parse (orig: + CSR/CSC view) | CSR build | apply 25K+25K (orig: prepare + apply) |
|---|---:|---:|---:|
| DD | 147 ms (184) | 9.1 ms (11.2) | 4.4 ms at 28 threads, 6.1 ms sequential (8.1) |
| GitHub | 365 ms (437) | 34 ms (46) | 9.5 ms at 28 threads, 19.8 ms sequential (28.8) |
| Twitch | 1,179 ms (1,428) | 115 ms (149) | 24 ms at 56 threads (69) |

`cycle_enum_batch` on `g.view().out` measured slower than `generate_batch` (DD 95 vs 38 ms)
because `graph::view()` builds the stored in-edges on first use; the generator itself is not
slower (the shuffle of 1.7M edges takes 22 ms in both). The later steps' gated regions exclude
batch generation.

### Deviations from the plan (and where they are recorded)

1. `set()` keeps a delete-then-reinsert pair in both lists instead of cancelling it, as the
   original's `prepare_batch()` does (PLAN 5.2 says "cancel pairs"; parity of normalized batches
   wins): ADR 0010 amendment, point 2. `apply_summary::cancelled_pairs` counts the pairs.
2. `as_sets` with `on_existing_insert = upsert`, `deletions_first = false` or unsorted /
   multigraph rows throws `not_supported_error`: ADR 0010 amendment, point 3.
3. The legacy generator is named `generators::legacy::cycle_enum_batch()` (after the existing
   `legacy::mosp_changes()`), with `cycle_enum_batch_options::locality_window = -1` for the
   original's `std::nullopt`; on a weighted graph its insertions weigh 1.
4. Only the simple-cycle oracle is ported now; the time-window and temporal oracles follow with
   those modes (0.4). The histogram type of the oracles is a dense `std::vector<std::uint64_t>`.
5. `read_edge_list` accepts `+` and commas (the original's tokens), unlike dynG's other readers
   (documented in `docs/api/file_formats.md`); `edge_list` has no timestamp column, so the
   timestamps of `duplicate_edges::keep` come back in `edge_list_info`; weight columns are not
   read from Matrix Market files (`read_matrix_market` does that). The parser takes a thread
   count in its options (0 = the hardware concurrency, the original's only choice) because the
   io functions take no `resources`.
6. `batch_semantics::net_effect()` stays with the DynLP vertex batches (0.3).

### Notes for the next steps

- The normalized batch is `detail::apply_delta` from `graph_access::apply(res, g, batch,
  &delta)`; read G_t before the call (count(−)) and G_{t+1} after it (count(+)).
- `graph::view()` transposes on first use; the static counter does not need the in-edges (the
  original's DirectedGraph has none): read the out-edges with `graph_access::out_view()`.
- The dataset digests (`cpp/tests/data/cycle_enum/datasets.txt`) cover the inputs of the
  histogram parity runs: the same graphs and batches reach both codes.
