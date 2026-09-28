# Retrospective: M2a (`cycle_count` on the CPU backends)

Status: **complete (2026-09-28)** on branch `m2-cycle`, pending the orchestrator's merge into
`main` (after M1b). Steps 1-3 each appended their section. Step 4 (close-out) added the milestone
summary, the acceptance record, the consolidated deviations, the lessons, the work M2b must do for
CUDA and the re-estimate. Step 5 fixed the 16 confirmed findings of the independent review and
updated those sections.

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

- Re-run on the final code of this step (after the per-thread marks moved into the phase):
  dataset parity (parity preset, 7.3 min) passed; `ci/check.sh` passed in a fresh clone of
  m2-cycle; `ci/gpu_local.sh` (dev-cuda, GPU 1: gpu, cpu, cuda goldens, memcheck, tidy) passed.

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

## Step 3: parity harness, goldens and the OpenMP gates (parity-perf, 2026-09-28)

Goal: the harness pieces for CycleEnumeration-GPU@0a976ad (golden export, replay, timed regions,
performance A/B), the full parity run and the OpenMP gates of acceptance criterion 4, recorded in
`parity/results/M2a.md`.

### Done

- `references.toml` and `build_reference.sh` already covered the reference since step 1 (CMake
  Release, OpenMP and CUDA sm_86 as its RESULTS.md; the CUDA build stays possible for M2b).
- **Golden corpus** (`parity/cycle_count_goldens.py`, reached as `parity/export_goldens.py
  cycle_count`): 24 cases from the original `cycle-enum` of the patched copy (OpenMP, 56
  threads): DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB k = 3; the seed-1 updates 1K+1K,
  25K+25K and 50K+50K (k = 4) on DD, GitHub and Twitch with `prior.csv`, `delta.csv` and the
  generated `batch.txt`; a DD sweep (locality windows 1000 / 10000 / 100000, k = 3 and 5).
  Cross-checks before writing: the original's sequential backend (23 cases), its
  `--compare-recompute`, the committed exporter counts, the plan's totals, the batch sizes.
  `--twice` exported again from a fresh archive copy: identical manifest `e40fa03b...`.
  `goldens.toml` gains a `[sets.cycle_count]` section (the sssp writer keeps it).
- **Replay** (`parity/compare.py cycle_count`, CTest `parity.cycle_count.cycle_enum_0a976ad`,
  label `parity`): the histogram CSV and the generated batch byte for byte; `dyng-compat-cycle-enum
  --write-batch` writes the batch in the exporter's text. Result: 72 of 72 replays equal
  (sequential including COLLAB, OpenMP 4 and 56 threads). This closes the step-2 open issue: the
  sequential backend now ran DD k = 7, GitHub, Twitch and COLLAB as well.
- **Timed regions** (`parity/timed_regions/cycle_count.toml`): regions now carry the task and the
  keys the harness reads: `static_end_to_end` (gated), `static_count` (reported), `update`
  (`update_seconds` vs `update_ms`, gated) and `update_end_to_end` (gate 1.10).
- **Performance A/B** (`parity/cycle_count_perf.py`, reached as `parity/perf_ab.py cycle_count
  run`): alternating A/B under the exclusive perf lock, 56 threads on both sides, histograms
  compared in every round and against the goldens, medians, spread flags, JSON records.
- **Results** (`parity/results/M2a.md` and three JSON files): every gate met. Static end to end
  0.64-0.96x (DD k = 3..7, GitHub and Twitch k = 3, 4), COLLAB k = 3 0.65x (44.5 s vs 29.0 s),
  update 25K+25K k = 4 DD 0.96x (26.7 / 25.6 ms; 0.96x again with 31 runs), GitHub 0.96x (294 /
  282 ms), Twitch 0.76x (161 / 122 ms); update end to end 0.68-0.89x.
- Harness smoke tests (`parity/tests/test_harness.py`): the new entry points, case names,
  histogram parsing and deltas, the goldens.toml section surviving both writers, the region map.

### Measured

See `parity/results/M2a.md`. The port was nowhere slower than the original, so no profiling and
no fix were needed for the gates. The likely cause of the gap is recorded there (the original's
`std::map` histogram increment per cycle vs dynG's dense arrays; not profiled).

### Deviations from the plan (and where they are recorded)

1. **Harness layout.** The cycle_count code of the harness lives in two new modules
   (`parity/cycle_count_goldens.py`, `parity/cycle_count_perf.py`) that `export_goldens.py`,
   `compare.py` and `perf_ab.py` dispatch to on a first argument `cycle_count`, instead of
   extending the sssp code paths: main (M1b) was changing those three files at the same time, and
   the sssp logic (objectives, trees, CSR files) shares little with histograms. The edits to the
   shared scripts are a few lines each.
2. **The static gate is end to end.** The original has no CPU timer for the count; its RESULTS.md
   reports process wall times, so the paper-timed region is the process (gate 1.05, every case
   >= 250 ms). The port's count alone is reported next to it.
3. **Environment of the gate.** Both sides ran with the libgomp defaults (no `OMP_*` variables),
   as the original's RESULTS.md, not with the pinned 28-thread environment of the sssp gate
   (PLAN 8.6: "56 where the original used 56").
4. **Golden locality sweep.** A window of 1000 cannot supply 25K deletions on DD (the original
   rejects it), so the window-1000 case uses 1K+1K.
5. **Load average** during the runs was 5.6-44, mostly the measured 56-thread processes
   themselves; DD's 25 ms update had outliers (spread above 10 %, flagged, not failed); the
   31-run repeat gave the same verdict.

### Final verification (commit 9f78c00)

- `ci/check.sh --parity` in a fresh clone of `m2-cycle`: every step OK (clang-format, cpu-only
  377/377 and dev 384/384 `ctest -L cpu`, clang-tidy, reuse, provenance, harness, doxygen,
  pre-commit) and `ctest -L parity` 4/4 (the dataset digests, the dataset histograms, the sssp
  replay and the new `parity.cycle_count.cycle_enum_0a976ad`, 439 s).
- `ci/gpu_local.sh` (dev-cuda, GPU 1): build, `ctest -L gpu` 79/79, `ctest -L cpu` 384/384, the
  sssp cuda goldens, memcheck and clang-tidy all passed.

### Notes for M2b

- The CUDA regions (kernel_ms / memcpy_ms, the device update, the resident scope) go into
  `timed_regions/cycle_count.toml` as `[reference.cycle_enum_cuda]`; `cycle_count_perf.py`
  keeps the backend fixed to OpenMP today and needs a `--backend cuda` path (CUDA events or
  nsys kernel sums over >= 20 runs for DD's 1.44 ms kernel).
- The goldens are backend independent: `compare.py cycle_count --configs cuda` only needs the
  config parser extended once the compat driver accepts `--backend cuda`.
- `dyng-compat-cycle-enum` generates the batch from `g.to_csr(res)` (a copy; Twitch 778 ms,
  outside every gated region); reading the out-edges directly would bring the update task's end
  to end closer to the original's structure.

## Step 4: close-out (finish, 2026-09-28)

### Done

- **`docs/algorithms/cycle_count.md`** completed against PLAN 9.5: what it computes, a graph
  requirements table (sorted rows, no parallel edges, weights ignored, no in-edges needed, int32
  vertices; `cycle_enum_compatible()` and the other batch semantics), backends, the determinism
  level `exact_value` and why, complexity, the timed-region table and the measured performance
  table of step 3, limitations, **Differences from the paper** (exact k-bounded enumeration, not
  the approximate kappa-truncated TruCy, which is not implemented; TruCy / DynTruCy is
  **submitted** to IEEE TC; what that means for counts, timings and the kappa experiments) and a
  **Paper vs fixed code** table (C1, C4, C6, H1, H3, C9, C10, K1/K2, K3/C5/C7 and the
  time-window items of the original's `CHANGES.md`, each with what dynG takes), the mapping from
  the original code and how to cite. The header's `@paper` lines say "submitted" as well.
- **README**: the status table lists `cycle_count` (sequential and OpenMP, bit-identical to
  CycleEnumeration-GPU@0a976ad, linking the parity certificate) and `cycle_count` on CUDA as
  planned for M2b; one paragraph on the example and the compat driver. **CHANGELOG**: one
  additive entry.
- **Dataset coverage of the parser** (acceptance criterion 2 names every dataset under
  `$DYNG_SCRATCH/datasets/cycle`): the digests now include the three timestamped SNAP edge lists
  (CollegeMsg, email-Eu-core-temporal, sx-mathoverflow; parser output and CSR). Their digests
  came from the original's exporter; the test passed on them and failed when one digest was
  altered (negative check). Before, only the four TUDataset graphs were compared.

### Deviations from the plan

None new. The algorithm page has no Python snippet (the binding is M5; the page says so).

## Step 5: review fixes (review-fix, 2026-09-28)

An independent review of M2a looked through four lenses: parity and correctness, API design,
performance, and tests. It confirmed 16 findings, each checked by a skeptic. They are 13 distinct
defects: the stack overflow was found twice and the cost of unbounded updates three times. Every
one is fixed on `m2-cycle` (commits `7bdd483..HEAD`), and none changes a count: the goldens, the
fixtures and the randomized suites are unchanged and pass, and the 72 golden replays were run again.

### Findings and fixes

| # | Finding (severity) | Fix | Commit |
|---|---|---|---|
| 1 | The default options crash. Without a bound, every search recursed once per path vertex: the update overflowed the stack (SIGSEGV) on a 70,000-vertex ring, the static counts at 200,000 vertices (sequential) and 1,000,000 (OpenMP). (high, found twice) | Every search keeps its path on an explicit stack: the update search, one shared static root search (`root_search.hpp`, for the OpenMP counter and the bounded sequential Johnson), and the unbounded Johnson (`circuit()` and `unblock()`). Same exploration order, same counts. This is option (b) of the finding; the default stays unbounded, as in the original (deviation 1). Tests close a 300,000-vertex ring by an update and count it statically, on both backends | `d6d9c05`, `0679ed1` |
| 2 | An unbounded update cost O(changes x n) time and 2 x threads x n x 8 bytes: the original's per-edge counts array was filled and summed over max_length + 1 entries for every change edge, and unbounded meant max_length = n. A large bound cost the same (a 10-vertex graph with k = 2e8 took 7.6 GB). (high and medium, found three times) | Histograms and engines are sized by min(k, max(n, 2)). The per-edge array is gone: each cycle is added to the thread's phase counters, which grow with the longest cycle found, and the reduction reads and clears only the lengths reached. The unbounded OpenMP compute grows its per-thread histograms the same way, and `apply_histogram_delta` visits only the reached lengths. The compat driver no longer clamps k to 1,000,000 (its comment was false beyond 1M vertices). Test: an unbounded update of 100,000 triangles costs about as much as the bounded one | `d6d9c05` |
| 3 | The OpenMP phase sized the marks in one parallel region and used them unchecked in a second one: out of bounds if the two teams differ in size, e.g. with `OMP_DYNAMIC=true`. (low) | One region: each thread sizes its scratch (marks, stack, counters), then runs the `omp for`. Allocation failures are reported after the region, and the scratch is reset | `d6d9c05` |
| 4 | The docs said "without a bound the sequential search is Johnson", which hid that the OpenMP compute and every update enumerate simple paths: exponential on a DAG. (low) | Stated in the `max_length` and `update()` Doxygen and on the algorithm page: a table per search, and the advice to use the sequential backend or a bound | `d6d9c05`, `3cb6fac` |
| 5 | `as_sets` with upsert, with insertions first, or with unsorted or multigraph rows was accepted at construction, and the doc said upsert works. (medium) | One check (`expect_supported_semantics`) in every graph constructor and in the set apply throws `not_supported_error` at construction; the doc names the exception | `7bdd483` |
| 6 | `io::format_histogram_csv` used a verb the plan does not have (PLAN 5.7 names `write_histogram_csv`). (low) | `io::write_histogram_csv(std::ostream&, counts, include_total)`; the formatter is private | `43938c0` |
| 7 | Unsupported vertex types failed at link time. (low) | A `static_assert` with a message in `compute`, `update` and `update_traits::make_participant` (inline wrappers over `detail::cycle_count_compute` / `cycle_count_update`). CTests `cycle_count.static_assert.{compute,update}` require the message | `f99ad3d` |
| 8 | The `stats` doc said only "All are deterministic". (low) | It now gives the meaning of every inherited counter for cycle_count and says why they are deterministic here | `f99ad3d` |
| 9 | The improvements were not isolated (PLAN 8.6). (medium) | Experiment copies of the original (`parity/experiments/cycle_enum`: dense histogram only; stage timers; both), the pre-fix port as another baseline, `--baseline-exe` in the harness, and the isolation section of `parity/results/M2a.md` | `2ae134c`, `eef6889` |
| 10 | There was no contamination monitor (PLAN 8.6). (medium) | `parity/contamination.py` measures the foreign CPU cores for each timed process: `/proc/stat` busy time minus the harness's and its children's rusage. Recorded per run in the JSON, flagged above 2 cores | `2ae134c` |
| 11 | "The steady-state update allocates nothing" was false: the ownership `unordered_map` allocated a node per change edge per phase, against invariant I9. (low) | A flat open-addressing table that reuses its arrays; a test counts the `operator new` calls of steady-state phases (0) | `fb08c20`, `d6d9c05` |
| 12 | M2a had no sanitizer run, and an out-of-bounds mutation survived the dev suite. (medium) | ASan+UBSan and TSan runs (below); `DYNG_STDLIB_ASSERTIONS` (`_GLIBCXX_ASSERTIONS`, ON in Debug builds); the phases check their workspace; the mutation `skip_workspace_resize` must fail the suite, and it does (12 failing cases) | `581ca99`, `d6d9c05` |
| 13 | The randomized suites had no seed replay (PLAN 8.1). (medium) | `cpp/tests/support/test_seeds.hpp`, shared with sssp: a seed per trial, `DYNG_TEST_SEED` / `DYNG_TEST_SEEDS`, and the replay command in every trace | `aa37f95` |

### Measured and verified

- **Parity**: the 72 golden replays (24 cases x sequential, OpenMP 4 and 56 threads) are
  byte-identical at `bce07a6` and again at `0679ed1`, after every change to a search. The
  randomized suites, the 80 fixture cases, the dataset histograms (`ctest -L cycle_count`, dev)
  and the mutation tests pass: `double_count_5` is killed by 20 failing cases, `weak_ownership`
  by 20 and `skip_workspace_resize` by 12, and the control copy passes.
- **Performance** (`parity/results/M2a.md`, port `0679ed1`, 11 A/B rounds, OpenMP 56,
  contamination monitor on): every gate is met. Static end to end is 0.53-0.93 of the original
  (GitHub k = 4 0.525, DD k = 7 0.600), the update 0.44-0.65, the update end to end 0.54-0.84,
  and COLLAB k = 3 0.291 (44.6 s against 13.0 s). One of the 264 timed processes ran with more
  than 2 foreign cores; it was an original run, and its case's verdict does not depend on it.
- **Isolation** (the new Section 3.4 of the certificate): with the same dense histogram on both
  sides, the straight port's count is 0.70-0.94 of the original's. The ported search is
  therefore not slower, and no regression is masked. The dense histogram saves about a fifth of
  the original's count at k = 4 and nothing measurable at k = 3, so the first certificate's
  unprofiled guess that it was the main cause was wrong. The review fixes bring the count to
  0.76-0.82 and the update to 0.48-0.69 of the straight port's. Separately built copies of the
  same code differ by up to about 20 % (code layout); the certificate states this.
- **A regression the isolation caught.** The first explicit-stack searches (`d6d9c05`) kept the
  expanded vertex's cursor in the stack array. That made the OpenMP static count up to 33 %
  slower than the recursive port: GitHub k = 4 count 1,344 -> 1,788 ms, which was still 0.79 of
  the original and inside the gate. Only the comparison with the straight port showed it.
  `0679ed1` keeps the expanded vertex in locals and scans a row in an inner loop; the count is
  now 0.76-0.82 of the recursive port's.
- **Sanitizers** (local runs, as in M1a and M1b; presets `asan` and `tsan`, Debug with
  `_GLIBCXX_ASSERTIONS`, only the three suites built):
  - ASan+UBSan passed `dyng_cycle_count_tests` (91 passed, 3 skipped: two CUDA cases and the
    allocation count, which is off under the sanitizers), `dyng_graph_tests` (84) and
    `dyng_io_tests` (38) with no report, at `3cb6fac`. It passed again on the cycle_count suite
    at `1881907`.
  - TSan (OpenMP off, as the preset documents: libgomp is not instrumented) passed 52 (5
    skipped), 78 (6 skipped) and 38, with no report.
- **Intermediate commits**: each commit of this step was built with `cpu-only` (`-Werror`) in a
  separate worktree, and its cycle_count, graph and io suites were run. All pass except
  `d6d9c05`: at -O3, GCC's `-Wmismatched-new-delete` on the test's counting `operator new` stops
  one test TU there. `bce07a6` fixes it; the library itself builds at `d6d9c05`.
- **Unbounded cost** (the review's probes, now tests):
  - an unbounded update of 100,000 disjoint triangles (2,000 + 2,000 changes) takes about as
    long as the bounded one (`UnboundedUpdateCostsAboutTheBoundedOne`);
  - a bound of 2e9 on graphs of 3-11 vertices costs nothing (`LargeBoundsMatchOracle`);
  - a 300,000-vertex ring is counted and updated without using the thread's stack
    (`LongRingsNeedNoThreadStack`);
  - the steady-state phases make no allocation (`SteadyStatePhasesAllocateNothing`).

### Deviations from the plan (and where they are recorded)

1. **The searches are iterative** (finding 1, option (b)). The original recurses; the port keeps
   the path on an explicit stack in the same order, so the counts and the order of exploration
   are the original's. The depth is bounded by memory (24 bytes per level), not by the thread's
   stack. Option (a), making the bound required, was not taken: the original's default is
   unbounded and its OptionsTest checks that, and the CUDA question (k <= 64) belongs to M2b.
   Recorded here and on the algorithm page.
2. **The update's per-edge counts array is gone**, and the per-thread counters grow with the
   cycles found. This changes the form of `accumulate_phase[_parallel]`, not the sums; the
   original's overflow check on each edge's sum cannot trigger before 2^64 cycles. Reported in
   the certificate's isolation section as an improvement, not as parity.
3. **The histogram has min(k, max(n, 2)) + 1 entries** (with a bound it had k + 1): no length
   past the vertex count can hold a cycle. `bound()` returns that length and
   `get_options().max_length` the requested bound. The CSV output is unchanged, as it prints only
   the non-zero lengths.
4. **The ownership index is a flat open-addressing table**, not the original's
   `std::unordered_map`. It gives the same answers and allocates nothing per update.
5. **`io::write_histogram_csv(std::ostream&, array_view<const std::uint64_t>, bool)`** takes the
   counts, not the result as in the PLAN 5.3 example, so that the io module does not depend on
   the algorithms: `write_histogram_csv(std::cout, hist.counts())`. Steps 2 and 4 above still
   call it `format_histogram_csv`, its name then.
6. **A third mutation, `skip_workspace_resize`**, is dynG's own; the two recorded ones come from
   the original. The phases' workspace check catches it in every build type.
7. **Sanitizers**: the ASan/UBSan and TSan CI jobs stay M6 deliverables (PLAN 11). M2a records
   local runs, as M1a and M1b did, and Debug builds now check standard containers.
8. **The contamination monitor** exists only in the cycle_count harness. M1b is changing
   `parity/perf_ab.py` (sssp) on `main`, so adopting `parity/contamination.py` there is left to
   the merge (open item).
9. **A certificate made of experiments.** The isolation needs builds of the original that are not
   the unpatched reference: one changes the counter, one only adds timers. They live under
   `parity/experiments/`, are named in every record (`baseline`), and never gate.

### Notes for M2b

- CUDA supports k <= 64 (PLAN 6.4.3), but the default `max_length` is -1 (unbounded, as in the
  original). M2b must decide, and record, what `compute(res_cuda, g)` does with the default
  options: throw `not_supported_error` naming the limit and the CPU backends, or clamp to
  max(n, 2) when n <= 64. The M3 conformance kit must then run the defaults per backend
  accordingly.
- The per-thread scratch of the update (`cycle_count_thread`: marks, explicit stack, counters
  grown on demand) is the host design. The device update keeps the original's path membership
  and `owner[]` array (K3 / C5 / C7).
- Use the stage-timer experiment copy for the CUDA isolation as well: the original already prints
  `kernel_ms`, and the host-side stages need the same treatment.

## Milestone summary

M2a was done in four steps on 2026-09-27/28 on branch `m2-cycle` (from `main` at `b59de86`):
graph, I/O, generator and oracle pieces (step 1), `cycle_count` on the sequential and OpenMP
backends (step 2), the parity harness, goldens and OpenMP gates (step 3), the close-out (step 4)
and the fixes of the independent review (step 5). 33 commits up to the close-out, 16
more for the review fixes; nothing pushed (the orchestrator pushes). Size of the branch against
`b59de86` at the close-out (added lines, tracked files; the review fixes added about 2,800 more,
most of them tests, harness and documentation):

| Area | Lines added |
|---|---:|
| Library: public headers and sources (`cpp/include`, `cpp/src`) | 4,650 |
| Tests (`cpp/tests`, without the fixture data) | 3,750 |
| Parity harness, compat driver, CI and scripts (`parity`, `tools`, `ci`) | 5,000 |
| Documentation (Markdown) | 800 |
| Committed fixtures (`cpp/tests/data/cycle_enum`: 389 files, 200 KB) | 12,400 |

### Acceptance record

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | A fresh clone of `m2-cycle` configures, builds every preset (`-Werror`), passes all tests via `ci/check.sh`; `ci/gpu_local.sh` still passes | final verification below (fresh clone, again after the review fixes) | met |
| 2 | `cycle_enum_compatible()` (sorted rows, no parallel edges, `set()`, Step 0 = net structural change) and the unweighted graph; sorted-row apply byte-equal to `apply_batch` / `prepare_batch` on the fixtures and random batches; `io::read_edge_list` = the original's parser (TUDataset, comments, separators, Matrix Market symmetries both ways) on the datasets | step 1: 80 random fixture cases x four graph types x both host backends, the seed-1 dataset batches (digests), 9 parser fixtures and 15 malformed files; step 4: the digests cover all seven files under `datasets/cycle` (the four TUDataset graphs and the three timestamped edge lists) | met |
| 3 | Histograms bit-identical on sequential AND OpenMP: fixtures; DD k = 3..7, GitHub k = 4, Twitch k = 4, COLLAB k = 3; update deltas seed 1 1K/25K/50K on DD, GitHub, Twitch; `generate_batch` identity; randomized differential tests vs subset DP, brute force, edge-set recount in CI; the two recorded mutations fail the suite (tested) | step 2: fixtures (80 random cases, 16 counts, 7 updates, 19 CLI runs), randomized suites (label `cpu`, seed replay since step 5), `cycle_count.mutation.{control,double_count_5,weak_ownership}` (and dynG's `skip_workspace_resize`, step 5); step 3: 72/72 golden replays (24 cases x sequential, OpenMP 4, OpenMP 56; the plan's totals; batches byte-equal) (`parity/results/M2a.md`), again 72/72 at `0679ed1` after the review fixes | met |
| 4 | OpenMP-56 gates (static end to end DD k = 3..7, GitHub / Twitch k = 3, 4; update 25K+25K k = 4 on DD, GitHub, Twitch) <= 1.05x / 1.10x; COLLAB k = 3 reported; recorded with methodology | step 5 (port `0679ed1`, replacing step 3's record): every gate met, ratios 0.53-0.93 (static), 0.44-0.65 (update), 0.54-0.84 (update end to end); COLLAB k = 3 0.29 (gated, met); unpatched original, exclusive lock, 11 A/B rounds (31 for the noisy DD regions, 5 for COLLAB), contamination monitor, improvements isolated in their own section | met |
| 5 | `references.toml`, `build_reference.sh`, export / compare scripts, `timed_regions/cycle_count.toml`; the algorithm page with 'Differences from the paper' and 'Paper vs fixed code'; this retrospective | steps 1, 3 and 4 | met |

### Final verification

After the review fixes (step 5), all in a fresh `git clone` of `m2-cycle` (heavy steps under the
shared perf lock, niced):

| Command | Commit | Result |
|---|---|---|
| `ci/check.sh --parity` (13 min) | `1881907` | clang-format, `cpu-only` 392/392 and `dev` 399/399 (`ctest -L cpu`, `-Werror`, including `cycle_count.mutation.{control,double_count_5,weak_ownership,skip_workspace_resize}` and `cycle_count.static_assert.{compute,update}`), clang-tidy naming, REUSE, provenance (106 files), harness smoke tests, Doxygen + convention check (116 compounds) and the parity preset `ctest -L parity` 4/4 (616 s) passed. `pre-commit` failed: its trailing-whitespace hook stripped the blank context lines of the two experiment patches. `09e190b` excludes `*.patch` from that hook |
| `ci/gpu_local.sh` (dev-cuda, GPU 1, 8 min) | `1881907` | build, `ctest -L gpu` 79/79, `ctest -L cpu` 399/399, sssp cuda goldens, memcheck, clang-tidy: all passed. The "dirty tree" in its report is the two patches rewritten by the failed hook above |
| ASan+UBSan `dyng_cycle_count_tests` (asan preset) | `1881907` | 91 passed, 3 skipped, no report |
| `ci/check.sh` quick steps (`DYNG_CHECK_SKIP="build tidy docs"`: format, REUSE, provenance, harness, pre-commit) | HEAD | passed; the commits after `1881907` change only `.pre-commit-config.yaml` and this retrospective |

At the close-out (step 4), all in a fresh `git clone` of `m2-cycle`:

| Command | Commit | Result |
|---|---|---|
| `ci/check.sh --parity` (18 min) | `0e9c46d` | every step OK: clang-format; `cpu-only` 377/377 and `dev` 384/384 (`ctest -L cpu`, `-Werror`, including `cycle_count.mutation.{control,double_count_5,weak_ownership}`); clang-tidy naming; REUSE; provenance (105 files); harness smoke tests; Doxygen + convention check (116 compounds); pre-commit; parity preset `ctest -L parity` 4/4 (dataset digests, dataset histograms 412 s, the sssp replay, `parity.cycle_count.cycle_enum_0a976ad` 420 s) |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | `0e9c46d` | build, `ctest -L gpu` 79/79, `ctest -L cpu` 384/384, sssp cuda goldens, memcheck, clang-tidy: all passed |
| `dyng_cycle_enum_parity_tests --gtest_filter='CycleEnumDatasets.*'` (parity preset, after `git pull`) | `b7fb8d5` | passed on all seven dataset files (25 s) |

The commits after `b7fb8d5` change only this retrospective; `pre-commit` (REUSE, codespell,
whitespace) passed on it.

### Measured parity and performance (steps 3 and 5)

- **Parity:** 72 of 72 golden replays byte-identical (sequential including COLLAB k = 3, OpenMP
  4 and 56 threads), at `1148d15` and again at `0679ed1`; the golden corpus was exported twice
  from fresh archive copies with the same manifest (`e40fa03b...`) and cross-checked against the
  original's own sequential backend, its `--compare-recompute`, the committed exporter counts and
  the plan's totals.
- **Performance (OpenMP 56 threads, libgomp defaults, the original's RESULTS.md setup):** the
  port is nowhere slower than the original (tables in `docs/algorithms/cycle_count.md` Section 5
  and `parity/results/M2a.md`). Step 5 measured where the gap comes from with copies of the
  original that differ in one change each: the straight-ported search is faster than the
  original's with the same histogram (it scans 4-byte column ids, the original 24-byte adjacency
  entries), the dense histogram saves about a fifth at k = 4, and the review fixes add the rest.

### Deviations, consolidated

Every deviation is in the table of the step that made it; the ones that matter beyond M2a:

1. **Batch semantics** (ADR 0010 amendment): `set()` keeps a delete-then-reinsert pair in both
   lists (the original's `prepare_batch`), not cancelled as PLAN 5.2 says; `as_sets` rejects
   upsert, insertions-first and unsorted or multigraph rows; every other semantics reaches
   cycle_count through Step 0 on G_t, which therefore runs twice (on G_t and in the commit), as
   in the original.
2. **API:** `generators::legacy::cycle_enum_batch()` (name after `legacy::mosp_changes()`); the
   histogram is a dense array (`counts()[len]`) instead of the `CycleHistogram` map; unbounded
   updates are allowed; `stats::affected` counts the lengths whose count changed; the parser
   takes a thread count in its options.
3. **Tests:** one CMake option `DYNG_MUTATION_TESTS` builds every recorded mutation plus a
   control copy (instead of `DYNG_TEST_MUTATION=<name>`), with the mutation points in the CPU
   code; `parity/mutate.py` is not written. Only the simple-cycle oracle is ported (time-window
   and temporal oracles with those modes, 0.4).
4. **Ported ahead of use:** the pruned lower_bound DFS of the original's CUDA kernels
   (`dfs.hpp`) is ported and tested on the host but used by no CPU backend (the original's CPU
   backends do not use it either).
5. **Harness:** the cycle_count harness is in two new modules reached from `export_goldens.py`,
   `compare.py` and `perf_ab.py` by a first argument `cycle_count` (main was editing those files
   for M1b); the static gate is on end-to-end process wall time (the original has no CPU count
   timer); the gate environment is the original's (56 threads, libgomp defaults, no pinning).
6. **Compat driver:** Read-Tarjan, brute force, the time-window and temporal modes, `--backend
   cuda` and `--version` are not reproduced.
7. `graph_properties::cycle_enum_compatible()` keeps `store_transposed = true`; cycle_count never
   builds the in-edges, so it costs nothing.
8. **Review fixes (step 5):** iterative searches (the original recurses), no per-edge counts
   array in the update, histograms of min(k, max(n, 2)) + 1 entries, a flat ownership table,
   `io::write_histogram_csv(std::ostream&, counts)` taking the counts, a third (dynG) mutation,
   sanitizers as local runs until M6, the contamination monitor in the cycle_count harness only,
   and experiment copies of the original for the isolation (never gates).

### Lessons

1. **Pin the inputs first, then port.** Step 1 made the parser, the CSR, the generated batches
   and the normalized batches byte-equal before any counting code existed; every later histogram
   mismatch could only have been in the counters, and there were none on the datasets.
2. **Cross-check the original against itself before trusting a golden.** The export ran the
   original's sequential backend, its `--compare-recompute`, a second export path and the plan's
   totals before writing a case, and exported twice from fresh copies. This costs minutes and
   removes a whole class of false mismatches.
3. **Ownership bugs need the right inputs; mutations prove the inputs are there.** A weakened
   ownership rule changes a count only on cycles through two or more change edges of one phase.
   Building the recorded mutations as copies of the library and requiring the randomized suite to
   fail on each (18-19 failing cases per mutation), with a control copy that must pass, keeps the
   suite's power tested instead of assumed.
4. **Additive edits to shared files merge cheaply.** With M1b editing the harness scripts on
   `main` at the same time, dispatching to new modules kept the shared scripts' diffs to a few
   lines each.
5. **Name what a region contains on both sides.** The original has no CPU count timer, so the
   gate is the process; writing that down in `timed_regions/cycle_count.toml` before measuring
   avoided comparing a port-only count against an end-to-end number.
6. **A shared machine still needs many rounds.** DD's 25 ms update had single outliers of 37-48
   ms; 31 rounds gave the same verdict as 11. Record spreads; flag, do not fail.
7. **The acceptance text is a checklist, not a summary.** Re-reading criterion 2 word by word at
   close-out showed that the parser digests covered only four of the seven dataset files; the
   gap was cheap to close.
8. **Measure the straight port as its own baseline.** A gate against the original cannot see a
   regression that an improvement elsewhere pays for. The first explicit-stack searches were 33 %
   slower than the recursive port and still passed the gate; timing them against the pre-fix
   build found it in minutes. Keep a build of the straight port for every milestone that changes
   a hot path.
9. **An unprofiled explanation is a hypothesis.** The first certificate attributed the speed-up
   to the dense histogram; isolating it showed it explains a fifth at k = 4 and nothing at k = 3.
10. **Two builds of the same code are not the same binary.** Copies of the original that differ
    only in the CLI's `main` ran the same counter up to 20 % apart. Ratios within one experiment
    are exact; differences across separately built copies need that margin.
11. **Defaults are API.** The original's unbounded default crashed on long paths in both codes;
    porting it straight exported the crash. A default deserves the same tests as any option
    (long rings, huge bounds, growth past the workspace).

### What M2b must do for CUDA

1. **Static CUDA counter** (`static_cuda.cu`, `dfs.cuh`, `work_queue.cuh`): the original's exact
   pruned DFS (K1: thread-local path and cursors, `lower_bound` closure, per-thread histogram
   reduced across the warp, grid by the occupancy calculator) and the prefix work items (K2: roots,
   edges, two-hop items built on the device with CUB prefix sums; the `auto` heuristic: edges for
   k <= 3, at k = 4 two-hop on graphs with >= 16 edges per vertex, two-hop for k >= 5). Reuse the
   host-tested `dfs.hpp` logic; CUDA k <= 64.
2. **CUDA update** (`cuda.cu`): K3 / C5 / C7 of the original: path membership instead of a
   visited array, `mark_owners_kernel` (the `owner[]` array, one entry per CSR position),
   `item_counts_kernel`, `count_owned_cycles_kernel`, (change, first hop) work items, and G_{t+1}
   built on the device (`next_degree_kernel`, `build_next_rows_kernel`, the device sorted-row
   merge). The device graph must stay **resident across batches** (PLAN 6.4.3); report the
   original's scope (with upload) and the resident scope separately (PLAN 8.6: improvements in
   their own table).
3. **Mutations:** add the CUDA mutation points (5-cycles counted twice in the work-queue kernel;
   weakened ownership in the update kernel) to `DYNG_MUTATION_TESTS`, with a CUDA control copy.
4. **Harness:** `[reference.cycle_enum_cuda]` regions in `timed_regions/cycle_count.toml`
   (`kernel_ms`, `memcpy_ms`, the device update; CUDA events or nsys kernel sums over >= 20 runs
   for regions under 10 ms, e.g. DD's 1.44 ms static kernel and 4.4 ms update); a `--backend
   cuda` path in `cycle_count_perf.py` (GPU 0, `CUDA_MODULE_LOADING=EAGER` or
   `resources::warm_up()`); cuda configurations in `compare.py cycle_count` (the goldens are
   backend independent); `dyng-compat-cycle-enum --backend cuda` and `--cuda-work-items`.
5. **Gates** (PLAN 6.4.3): static CUDA k = 4 kernel DD 1.44, GitHub 58.0, Twitch 38.1 ms, COLLAB
   k = 3 51.4 ms; update 25K+25K k = 4 DD 4.4, GitHub 14.3, Twitch 27.3, COLLAB 197 ms; DD 50K+50K
   6.0 and 100K+100K 9.6 ms; COLLAB k = 4 (199,739,028,717) nightly.
6. **Tests:** the original's CUDA suites (cuda_*, CliCudaTest, CudaCountersMatchOracle,
   dynamic_update_cuda), the randomized suites on the CUDA backend (label `gpu`),
   compute-sanitizer memcheck / racecheck / initcheck on the new kernels in `ci/gpu_local.sh`.
7. **Small items:** the compat driver generates the batch from `g.to_csr(res)` (a copy; Twitch
   778 ms, outside every gated region); read the out-edges directly. Update the manifest's
   `backends` and the maturity note of the algorithm page.

## Re-estimate

M2a (the CPU half of the plan's M2, estimated with M2 at 2-3 working weeks and re-estimated after
M1a at 4-6 days for all three backends) took four implementation steps over two days, including
the full dataset parity and the OpenMP gates. The CUDA half is the larger one: two kernel
families, the device batch application, the resident graph, two measurement scopes and the
short-region timing protocol. Estimate for M2b: 3-5 days of focused work plus a review-and-fix
step, after M1b has merged (it needs M1b's CUDA core, streams and the GPU timing harness). The
0.1 estimate of the M1a retrospective is unchanged.

## Open items carried forward

1. ~~Independent review of M2a~~: done; its 16 confirmed findings are fixed (step 5).
2. **Merge:** `m2-cycle` edits a few shared files additively (CMake module lists and options,
   `cmake/sanitizers.cmake`, `graph` apply paths and constructors, `io/result_io`,
   `update_participant::reads_prepared_graph()`, the parity entry scripts, `goldens.toml`, the
   sssp randomized test's seeds helper, CHANGELOG, README); the orchestrator merges after M1b.
   With the merge, `parity/perf_ab.py` (sssp) should adopt `parity/contamination.py` (PLAN 8.6);
   M1b is changing that file on `main`.
3. ~~The explanation of the port's speed advantage is not profiled~~: measured (step 5,
   `parity/results/M2a.md` Section 3.4).
4. The 9,000-graph fuzz campaign of the original (PLAN 6.4.3: nightly) is not set up; the
   randomized CTest suites run in CI (and `DYNG_TEST_SEEDS` widens them).
5. The CUDA work of M2b (above), including what the default unbounded options do on CUDA
   (k <= 64; step 5, notes for M2b).
6. ASan/UBSan and TSan as CI jobs (M6); M2a records local runs.
