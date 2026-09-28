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

## Step 2: cycle_count on the CPU backends (cycle-cpu, 2026-09-27)

Goal: `dyng::cycle_count` on the sequential and OpenMP backends (PLAN 5.1, 6.3, 6.4.3, 4.5
hooks), ported straight from CycleEnumeration-GPU@0a976ad, with the original's suites, the
randomized parity suites, the two recorded mutations as tests that must fail, profiler stages,
an example and a compat driver.

### Done

- **Public API** `<dyng/cycle_count.hpp>`: `options{max_length, method, mode}` (enums
  `search_method::johnson`, `cycle_mode::simple`), `stats` (the apply summary, `deletions`,
  `insertions`, `cycles_removed`, `cycles_added`; all deterministic), the opaque `result`
  (`counts()` indexed by length, `count(len)`, `total()`, `bound()`, `clone()`), `compute()`,
  `update()`, and `dyng::update()` support (`update_traits<cycle_count::result>`). Instantiated for
  int32 vertices, int32 / int64 offsets, `unweighted` / int32 weights (weights ignored).
  `compute()` rejects graphs without `row_order::sorted` and `multi_edges::forbid` with an
  `invalid_argument_error` naming both properties and the fix; cuda throws `not_supported_error`
  (M2b).
- **The port** (`cpp/src/algorithms/cycle_count/`): `static_sequential.cpp` (JohnsonSearch:
  `circuit_bounded` with a bound, Johnson's blocked lists without one), `static_openmp.cpp`
  (`count_root`, roots with `schedule(dynamic)`, per-thread histograms), `cycles_through_edge.hpp`
  (`count_cycles_through_edge`, `search`) with `changed_edge_index` (ChangedEdgeIndex, the
  original's `unordered_map` keyed by the two 32-bit ids), `sequential.cpp` / `openmp.cpp` (the
  phases of `update_static_histogram[_openmp]`; the OpenMP backend with one thread runs the
  sequential phases, as the original), `cycle_count.cpp` (validation, dispatch, the update
  participant, `apply_histogram_delta` in the finalize hook, the result), `problem.hpp` (state,
  workspace, the template card). The mechanical changes: names, templates on the index types,
  exceptions (`internal_error` for a negative bucket, `capacity_error` for a count beyond 2^64 - 1),
  threads from `resources`, no `static thread_local` (the per-thread marks, histograms and the
  ownership table live in a pooled workspace, ADR 0015; the marks are sized by their own thread,
  first touch in parallel as the original's thread-local buffers), dense histograms instead of the
  `CycleHistogram` map.
- **Hooks and stages**: `cycle_count.update` > `normalize` (Step 0 on G_t), `count_minus` (G_t),
  `commit` > `graph.apply`, `identify_affected`, `count_plus` (G_{t+1}), `finalize`;
  `cycle_count.compute` > `reset`, `count`, `finalize`. The update never transposes the graph.
- **Graph / framework (additive)**: `detail::compute_structural_change()`
  (`cpp/src/graph/structural_change.{hpp,cpp}`): `prepare_batch()` on G_t for every batch
  semantics (under `set()` equal to the commit's `apply_delta`, checked in Debug builds on every
  update); `update_participant::reads_prepared_graph()` (default true) so `run_update()` builds the
  in-edges or the device copy only for results that read them; `run_update()` instantiated for
  unweighted graphs; `dyng_add_module` records the module targets and sources (for the mutation
  copies). `io::format_histogram_csv` (the original's `to_csv`).
- **dfs.hpp**: the exact pruned DFS with the lower_bound closure (`extend_prefix`, `find_edge`,
  `dispatch_capacity`) of the original's CUDA kernels, ported for the host and tested against the
  oracle with root, edge and two-hop prefixes (as the work queue forms them), ready for M2b. The
  CPU backends do not use it (neither does the original).
- **Tests** (`dyng_cycle_count_tests`, 82 cases, label `cpu`): the original's OptionsTest
  (defaults), JohnsonStaticTest, OpenMPJohnsonTest, SequentialParityTest (simple part), HistogramEngineTest, HistogramTest,
  CyclesThroughEdgeTest, DynamicUpdateParityTest, DynamicUpdateOpenMPTest,
  RandomizedStaticParityTest (CPU part, 400 graphs), RandomizedUpdateParityTest (valid and
  arbitrary batches, UpdateBatchValidationTest) re-expressed; dynG's own: every batch semantics
  (upsert, ignore with kept self-loops, insertions first, undirected set and upsert, weighted
  graphs) over chains of three batches against `compute()` and the oracle (conformance C2),
  inverse batches (C5), thread counts 1/2/4/7 against the sequential backend (C3/C6), stale and
  poisoned results, invalid batches that change nothing, profiler stages, workspace reuse,
  composition with sssp through `dyng::update` (C10), unbounded results on growing graphs.
  `structural_change_test.cpp` (graph tests): equal to `prepare_batch` on the 80 fixture cases and
  to the edge-set change under six semantics.
- **Which of the original's 218 TEST macros apply.** Ported or re-expressed (step 1 and this
  step): the simple-cycle CPU suites above, CyclesThroughEdgeTest, HistogramTest (as the result
  accessors and `format_histogram_csv`; the map's increment/equality/clear API has no counterpart),
  BruteForceTest and OracleSelfTest (`cycle_oracle_test.cpp`), the graph, parser, batch generator
  and edge-change suites (step 1), and the simple-mode CLI cases of CliSequentialTest /
  CliUpdateTest (the 19 compat CLI runs). Not applicable in M2a: the CUDA suites (cuda_*,
  CliCudaTest, CudaCountersMatchOracle, dynamic_update_cuda; M2b), Read-Tarjan, time-window,
  temporal, cycle-union and path-bundling suites (modes not in dynG before 0.4), OpenMPConfigTest
  and ParallelPartsTest (replaced by `resources`, tested in core), ResultTest / OptionsTest name
  tables (no status or name strings in the API), ProjectSmokeTest.
- **Mutation tests** (PLAN 8.4): `DYNG_MUTATION_TESTS` (ON with the tests) builds three copies of
  libdyng in which the cycle_count module is recompiled: `double_count_5`
  (`DYNG_MUTATION_DOUBLE_COUNT_5`: the static counters count 5-cycles twice), `weak_ownership`
  (`DYNG_MUTATION_WEAK_OWNERSHIP`: the changed edge with the id just below the anchor no longer
  owns the cycle) and a `control` copy without a mutation. `cycle_count_random_test.cpp` is linked
  against each; `cpp/tests/mutation/run_mutant.cmake` requires the suite to run to its end and to
  fail for a mutation (18 and 19 failing cases) and to pass for the control copy. CTests
  `cycle_count.mutation.{control,double_count_5,weak_ownership}`, labels `cpu;mutation`.
- **Parity data** from the pinned original (the exporter now links its sequential Johnson, OpenMP
  counter and `update_static_histogram[_openmp]`): `cases/case_NNN.counts` (static and updated
  histograms, k = 2..7 and unbounded, both backends), `counts/` (7 fixture graphs, 16 counts and 7
  generated-batch updates), `cli/` (19 runs of the original `cycle-enum`: standard output and exit
  status), and with `--datasets` `datasets_counts.txt` (DD k = 3..7, GitHub and Twitch k = 3, 4,
  COLLAB k = 3, and the seed-1 updates 1K/25K/50K on DD, GitHub, Twitch; the original's
  sequential and OpenMP results checked equal while generating). Tracked fixtures of cycle_enum:
  92 KB -> 166 KB.
- **Tools**: `dyng-compat-cycle-enum` (`tools/compat/cycle_enum`: the original's options, aliases,
  validation messages, `update_seconds=`, `--compare-recompute`, histogram CSV; plus `--timing`
  and a `RESULT` line), its CTest `compat_cycle_enum.cli`; `examples/cpp/cycle_count_update.cpp`
  with CTests comparing its output to the original CLI's; `parity/timed_regions/cycle_count.toml`;
  `docs/algorithms/cycle_count.md`.

### Parity evidence

- The 80 random fixture cases, k = 2..7 and unbounded, sequential and OpenMP: the static
  histogram before and after the batch and the updated histogram equal the original's
  (`CycleCountFixtures.RandomCasesEqualTheOriginal`).
- Fixture graphs read with `io::read_edge_list` (TUDataset-style, Matrix Market symmetric and
  hermitian, sparse ids, comments), 16 static counts up to k = 40 and unbounded, 7 updates with
  generated batches (windows 299): equal.
- The compat driver reproduces the original CLI's standard output and exit status on 19 runs
  (count and update, both backends, `--compare-recompute`, locality, and 9 error cases).
- Datasets (cpu-only Release, `dyng_cycle_enum_parity_tests --gtest_filter=CycleCountDatasets*`,
  6.5 min): OpenMP (56 threads) on all 19 lines (DD k = 3..7, GitHub k = 3, 4, Twitch k = 3, 4,
  COLLAB k = 3; the 9 updates) and the sequential backend on DD k = 3..6 and the 9 updates: equal
  to the original. The totals are the plan's numbers (DD 2,020,240 / 4,396,674 / 9,476,048 /
  21,485,606 / 54,966,172; GitHub k = 4 75,872,845; Twitch k = 4 389,362,369; COLLAB k = 3
  1,257,799,573). `DYNG_CYCLE_PARITY_FULL=1` runs the sequential backend on every line.

### Measured (informal; not a gate)

`build/parity` (-O3) against the UNPATCHED original's `cycle-enum`, exclusive perf lock, OpenMP
56 threads, default wait policy for the updates (median of 11 / 7 / 7), `OMP_WAIT_POLICY=PASSIVE`
for the static runs (3 runs each; process wall time):

| Graph | static k = 3, orig / port (s) | static k = 4, orig / port (s) | update 25K+25K k = 4, orig / port (ms) |
|---|---:|---:|---:|
| DD | 0.28 / 0.25 | 0.30 / 0.25 | 25.9 / 25.2 |
| GitHub | 0.72 / 0.60 | 2.78 / 2.25 | 292.6 / 278.1 |
| Twitch | 1.84 / 1.57 | 3.63 / 3.05 | 160.6 / 119.3 |

With `OMP_WAIT_POLICY=PASSIVE` the DD update measured 28-30 ms on both sides (noisy, 3 runs); the
gate protocol of step 3 fixes the environment. Stage breakdown of one DD update (ms): normalize
2.5, count_minus 8.5-10, commit 5.7-6.4, identify_affected 1.6, count_plus 8.5-10.

### Deviations from the plan (and where they are recorded)

1. **Step 0 twice, as the original.** `before_apply` needs the normalized deletions on G_t, but the
   graph's normalization happens inside the commit. `compute_structural_change()` computes them on
   G_t, and the commit normalizes again (the original's `apply_batch` also sorts and deduplicates
   again). Under `set()` Debug builds check both lists equal on every update.
2. **Every batch semantics** (PLAN 5.1) is supported through the structural change, including
   `deletions_first = false` and undirected graphs; parity is defined under `set()`.
3. **Unbounded updates.** The original's update requires a length cap; dynG's `update()` of a
   result computed without a bound enumerates every cycle through the change edges (bound
   max(n, 2), the histogram grows with the graph).
4. **The result's histogram is a dense array** (`counts()[len]`, size bound + 1) instead of the
   `CycleHistogram` map; `count(len)` returns 0 for lengths the map would reject (< 2); overflow is
   checked when histograms are merged, summed or updated (the original checks every increment).
   `stats::affected` counts the lengths whose count changed.
5. **Mutation tests**: a CMake option `DYNG_MUTATION_TESTS` builds all recorded mutations and a
   control copy (instead of `DYNG_TEST_MUTATION=<name>` building one); the mutation points are in
   the CPU code (the static counters and the ownership index), because the original's were in its
   CUDA kernels, which M2b ports; M2b adds its CUDA mutation points to the same list.
   `parity/mutate.py` (PLAN 8.4, before minor releases) is not written; the CTests replace it for
   cycle_count.
6. **The exact pruned DFS with the lower_bound closure** is CUDA code in the original; M2a ports it
   for the host (`dfs.hpp`, tested) but no CPU backend uses it.
7. **`graph_properties::cycle_enum_compatible()` keeps `store_transposed = true`** (step 1); the
   update does not build the in-edges (`reads_prepared_graph() == false`), so the flag costs nothing.
8. **Compat driver**: `--algorithm read-tarjan|brute-force`, the time-window and temporal modes and
   `--backend cuda` exit with status 1 and a message (not ported in M2a); `--version` is not
   reproduced. The graph type is `graph<int32, int64, unweighted>`.

### Notes for the next steps

- Gates: `dyng-compat-cycle-enum --task count` against `cycle-enum` (process wall time), and
  `update_seconds=` on both sides (`parity/timed_regions/cycle_count.toml`). Set the OpenMP wait
  policy identically on both sides.
- The full sequential dataset parity: `DYNG_CYCLE_PARITY_FULL=1 build/<preset>/cpp/tests/
  dyng_cycle_enum_parity_tests --gtest_filter=CycleCountDatasets*` (COLLAB k = 3 sequential takes
  tens of minutes).
