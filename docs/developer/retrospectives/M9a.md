# Retrospective: M9a (triad_count against ESCHER-GPU)

Status: **done** on the branch `m9a-triad-count` (all five steps; ready for the pull request). M9a is the first half of PLAN 11.3's
M9 row: `triad_count` exact against ESCHER-GPU@abcf9b2 on the sequential, OpenMP and CUDA
backends, the independent oracle, the conformance kit and the mutations, and the paper gates
(the Nsight Systems peak included); M9b follows with `experimental::count_local_patterns`,
Python, the CLI, the docs and the freeze. It runs in five steps, and each step adds its section
here. Release 0.3.0 = M8a + M8b + M9a + M9b (PLAN Appendix F; the split is the author's decision
D2).

**The author's decisions for M9 (2026-10-15; recorded in GOVERNANCE.md's approvals log by step
1).** D1: the gated Nsight Systems peak is the peak of live device allocations (`cudaMalloc` +
`cudaMallocAsync`, kind Device) over the whole timed process, on coauth_b50k_s1, syn1m_b50k_s1 and
syn2m_b50k_s1, port / A <= 1.05; the footprint is reported, not gated. D2: the M9a / M9b split.
D3: the host backends are incremental host engines (an untuned transliteration of A's wedge
engine), `testing::triad_reference` the independent oracle. D5: the committed test-data budget is
1.25 MB, at most about 16 KB of make-test counts. D4 and D6-D10 are taken on the plan's defaults,
pending the author's confirmation at the 0.3.0 release.

## Step 1: the reference side (regions-goldens-crosscheck)

### What was built

| Commit | What |
|---|---|
| `2f54bce` | **First commit of the branch**: `parity/timed_regions/triad_count.toml` (the region map: `update[b1..b3]`, `full_count`, `end_to_end` gated; the sub-rows reported; the scope rules), the per-batch rows and the refusal of unknown keys in `perf_ab.load_regions` and `bench_suite.region_map`, `ci/tests/test_triad_regions.py`, ADR 0048 and its GOVERNANCE line |
| `23d0c03` | the author's M9 decisions in GOVERNANCE.md; `plan.md` (the M9 row split into M9a / M9b) and the roadmap (D3: the 0.5 OpenMP item becomes tuning) |
| `903b3aa` | `parity/triad_goldens.py` (export, verify, crosscheck, list), registered in `export_goldens.py` and `compare.py`; `parity/tests/test_triad_goldens.py` |
| `f91b226` | the set `triad_count_escher_gpu` in `parity/goldens.toml` (with the inputs and the coverage table), named in `references.toml` |
| `7eb7e61` | `certify.py` lists a `triad_count` set without replays without a verdict (deviation 13) |
| `07ee804` | `parity/results/M9a.md` section 1, `parity/results/M9a-crosscheck-escher_gpu.json`, this section, ADR 0048's figures, plan.md, CHANGELOG |

Results: the set exported twice from two fresh patched copies with byte-identical manifests
(300 cases, 367.7 MB); every case's `--check` run agrees; `generators::legacy` reproduces all 12
synthetic inputs; A agrees with its own CPU reference on all 90 cross-check runs, whose counts
equal the goldens' (`parity/results/M9a.md` section 1). Nothing was found against the original.

### Deviations from the plan and the step's text (each with the same intent)

1. **The branch starts at `origin/main` `9cde147`**, not at `m8b-hypergraph` `65a62c9`: the
   author had merged PR #15 (M8b) when M9a started, so the step's own rule applies ("if
   origin/main already contains PR #15's merge, branch from origin/main instead"). The frozen-path
   diff of the acceptance (`git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,
   graph}`) is empty on `9cde147` too.
2. **The unpatched counts of an export come from its own fresh copy.** The export patch keeps A's
   Makefile-built executables unchanged as `parity_export/orig/<exe>` before it installs the
   recording twins (ADR 0036); each of the two exports counts with its copy's originals, so the
   double export compares two independent builds of A end to end. The persistent unpatched copy
   (the performance baseline) is the one the cross-check runs, and its counts equal the goldens'
   on all 90 runs, so both builds print the same counts.
3. **`update.add` includes `identify_affected`.** The step names `update.add = count_plus +
   finalize`, but `update` contains `identify_affected` (A builds the inserted keys' frontier
   inside `update.add`), and the three sub-rows should add up to the gated row on both sides.
4. **`full_count` maps to `triad_count.compute`**, the stage the framework emits for the whole
   `compute()` call (`hook_stages::compute`); the sketch's `triad_count.count` is the aggregate
   count inside it. A's `init.full_count` includes allocating and reading back its bins, so the
   whole call is the matching scope (the step asked to use the framework's name and record it).
5. **One `.dgb` per batch.** `to-dgb` writes one file with all batches; the acceptance asks for the
   SHA-256 of each batch's file, so the exporter splits the file into valid one-record `.dgb`
   files (`batch<N>.dgb`, header count 1, CRC recomputed; dyng's `BatchReader` reads them). The
   SHA-256 in the records is that of the uncompressed file; the stored files are zstd-compressed.
6. **Case names use the size labels of the gate cases** (`coauth_b50k_s1`, `syn1m_b5k_s3`,
   `syn2m_b200k_s2`, `syn100k_b20_s1`), as M8a's sets and the acceptance's gate rows name them;
   `info.json` maps the labels to the sizes. make-test's groups are `test/<NN>-<label>`.
7. **make-test's seven input-rejection invocations are not cases** (they print no counts). The
   60 counting invocations are checked against M8a's set, invocation by invocation.
8. **The `--check` run of a case is its count run plus `--check`.** The suite's own
   `--reference` is exercised by the cross-check, which runs every make-test case with
   `--check --reference` (also the cases the suite runs with `--reference` only).
9. **The region loader is stricter**: it refuses an unknown key of a region (a misspelt
   `per_bach` would otherwise be silently ignored) and a repeated region name; a region with only
   a `port_report` (the process wall time) counts as naming its port side. All existing maps
   load unchanged.
10. **No batches-only recorder mode was added**: the recorder's existing batches mode records
    coauth-DBLP in about 15 s per run (alone), and its exit record already carries `inserted_keys`.
11. **The legacy-generator proof runs through the Python binding** (`dyng.generators.legacy.
    escher_hypergraph_arrays` + `dyng.io.write_hypergraph`), which calls the C++ generator; no new
    C++ tool was needed, so this step changed no C++ code.
12. **`bench_suite.py` knows triad_count's region map** (`REGION_MAPS`, `MAP_KEYS`) but not yet a
    harness tool (`TOOLS`): `perf_ab.py triad_count` arrives in step 2, and a suite file naming it
    before then would not validate.
13. **`certify.py` lists the triad set without a verdict until it has replays.** A golden set
    without replay records fails every certificate (`golden_parity`), and the triad replays
    arrive with the port; until M9a step 5 makes them required from the 0.3 series, a
    `triad_count` set without replays is listed with `passed = None` (shown as "listed"), and a
    replay record, once there, is judged as usual. The 0.1 / 0.2 certificate tests pass
    unchanged.
14. **The author's M9 decisions were recorded in this step** (GOVERNANCE.md, plan.md, roadmap), as
    the first step of the milestone.

### Machine time

| What | Time |
|---|---|
| Double export (two copies built, two exports at once, 3 groups each, GPU 1, shared lock) | 62 min wall (06:28-07:30); 10,569 s / 10,472 s of group time; coauth 61%, syn2m 23% |
| Cross-check (90 runs, 2 at a time) | about 4 min wall, 401.7 s of run time; coauth main's CPU reference 143.5 s (time box 2 h) |
| Development exports (subsets, the existing patched copy) and the first coauth reference run | about 10 min |
| `verify` | 16 s |

### Notes for step 2 (oracle, replay harness, driver, pre-reads)

- **Reading the set** (`$DYNG_SCRATCH/goldens/triad_count/escher_gpu`; `triad_goldens.py`):
  `cases/<group>/group.json` has the input name, the vertex count (`vertices`, A's
  maxVertexId: pass it as `hypergraph_read_options::num_vertices`), the H2V `payload_capacity`
  for `capacity_policy::escher_gpu(capacity)` (0 everywhere but `test/01-smoke`: 4096), and the
  batch files `batch<N>.dgb.zst` (decompress with `zstd -dc`; flags `deleted_members` and
  `expected_ids`, ids = A's keys - 1). `cases/<group>/<exe>.json` has `full_count`, and per batch
  `delta`, `counts_after`, `deletions`, `insertions` and `batch_sha256`. Inputs:
  `inputs/<name>.json` + `inputs/<name>.hg.zst` (A's loaded rows, `index_base` 1), coauth-DBLP
  from `$DYNG_SCRATCH/datasets/hyper/coauth.hg` (identical bytes).
- **Coverage**: `triad_goldens.coverage_cases(backend)` (and `compare.py triad_count --list
  --backend B`) gives the cases per backend: cuda 300, sequential 120, openmp 142.
- **`compare.py triad_count`** today verifies (`--verify`) and lists (`--list`); step 2 adds the
  replay modes and counters in `triad_goldens.compare_main`. Do not change the export functions
  without exporting again (`export --twice`): the committed manifest was produced by `903b3aa`'s
  exporter.
- **Driver names the map expects**: `update_ms`, `full_count_ms`, `total_ms`, `load_ms`,
  `generate_ms` (driver timers, A's `--timing` JSON format), stages `triad_count.update`,
  `triad_count.compute`, `driver.total`, `driver.batch.generate`, `io.read_hypergraph`,
  `hypergraph.build`, `hypergraph.upload`.
- **Memory points for store_memory**: `group.json` `device_bytes` holds A's `deviceBytes()` after
  the build and after every batch for every group (coauth_b50k_s1: 1,922,784,204 after the build,
  as M8a's trace).
- **For step 5**: `certify.golden_suites` lists `triad_count` sets without replays as "not yet
  required" (deviation 13); the step-5 rule (required from (0,3), all backends per the coverage
  table, both modes) replaces that branch. `bench_suite.TOOLS` needs `triad_count` once
  `perf_ab.py triad_count` exists.
- **make-test stdout** (`cases/test/*/<exe>.stdout`) is A's unpatched stdout, for M9b's CLI;
  the hosted-CI exact-parity test of D5 may commit at most about 16 KB of these counts (tracked
  `cpp/tests/data` is 998,934 bytes; the 1.25 MB budget and its check in `ci/check.sh` are not in
  place yet).

## Step 2: the oracle, the replays, the driver, the pre-reads (oracle-harness-driver-prereads)

**GO for step 3.** The oracle reproduces ESCHER-GPU (A) exactly on all 423 go/no-go case replays,
by key and by content, on the sequential, openmp and cuda containers; the driver draws A's batches
byte for byte; both pre-read targets are met (end to end and memory); the boost lock does not
hold on the synthetic cases, so their gates run at base (`parity/results/M9a.md` section 2).

### What was built

| Commit | What |
|---|---|
| `bcf3f59` | `dyng::testing::triad_reference` (tracked header, its own object library, OpenMP), the class map derived from the Venn-region orbits, `dyng_triad_reference_tests` with the test-only fixture of A's table, `triad_reference_isolation`, the planted mutation (CTest `testing.mutation.triad_reference_class_map`, `parity/mutate.py` suite `triad_reference`), ADR 0049, the API baseline |
| `062b330` | `generators::legacy::escher_host_model` and `escher_batch_generator::next(model)` (one shared draw with `next(res, hypergraph)`), their tests, the API baseline |
| `282f3c7` | `dyng_triad_replay` (`triad_replay.{hpp,cpp}`, `triad_replay_main.cpp`, `dyng_triad_replay_tests`), `parity/triad_replay.py` (`compare.py triad_count` replays, `--plant`, `singles`, `draws`), `parity/tests/test_triad_replay.py`, ADR 0050 |
| `1e6e933` | `dyng-compat-escher run` (`tools/compat/escher/escher_run.cpp`), `parity/triad_perf.py` (`perf_ab.py triad_count run`, `memory`, `--probe`), the footprint figures of `hypergraph_perf.memory_by_range`, `bench_suite` knows the tool, ADR 0051 |
| `729bb79` | two template parameter names for clang-tidy's naming rules; the mutation-point test reads the oracle's hook |
| `eeb03ec` | `parity/results/M9a.md` section 2 and its JSON records, this section, ADRs 0049-0051 in GOVERNANCE.md and the ADR index, plan.md, CHANGELOG |

### Deviations from the plan and the step's text (each with the same intent)

1. **The oracle's API is wider than one function.** Besides `triad_reference(...)` it has
   `triad_reference_all` (the requested patterns from one line graph, so a replay counts all four
   with one rebuild), the class-map functions (so the tests can show the map is the permutation
   orbits) and its own `triad_pattern` enumeration (it may not include `dyng/triad_count.hpp`). Its
   errors are `std::invalid_argument` / `std::bad_alloc`, like `rebuild_line_graph`'s.
2. **The class representatives are the canonical (smallest) ids of the orbits**, so a test can
   prove that the list is a list of orbit representatives; "classes with fewer than two
   intersecting pairs" counts a pair as intersecting when its pairwise-only region or the triple
   region is non-empty (A's table agrees on all 128 ids).
3. **The single-triad `.hg` files are not committed.** The C++ test builds the 30 triples itself;
   the record `M9a-oracle-singles-escher_gpu.json` keeps their lines and SHA-256 (the test-data
   budget of D5 is left to the make-test counts of a later step).
4. **The tool reports, the script judges.** `dyng_triad_replay` writes the counts, ids, integrity
   and resolution facts as JSON; the comparison with the goldens, the coverage table, the records
   and the planted defects are in `parity/triad_replay.py` (as `dyng_hypergraph_ids` /
   `hypergraph_ids.py` do). The planted defects are a `--plant` option of the harness, each tested
   to fail; the comparison also checks every batch file's SHA-256 against the golden's.
5. **The go/no-go runs the syn1m seed-1 grid on sequential and coauth_b50k_s1/type1 on sequential
   with `--no-coverage`**: the step asks for them, the coverage table (D6) does not list them.
6. **The oracle evaluation is reused between the two modes** when their live lists are equal (the
   oracle is a function of the list); a differing state is evaluated again.
7. **The driver's own additions**: `--dump-batches FILE` (the batch-hash proof; never in a timed
   run), `apply_ms` as the per-batch timer of `--no-count` (the region map's `update_ms` is the
   counted update of step 3), `driver.total` beside `total_ms`, the stage `driver.batch.generate`
   from a `scoped_stage`, and the stage `io.read_hypergraph` for a file input (a synthetic input's
   load is `load_ms` only). `--reference` and an alignment other than A's default 4 are refused.
8. **`perf_ab.py triad_count run --probe`**: the gate harness repeats a contaminated round and gives
   up after 2 x rounds rejections without a record; the clock probe needs the record of every
   round (whether the lock held per side), so `--probe` keeps the rounds.
9. **`hypergraph_perf.memory_by_range` gained the footprint figures** (additive keys: the held
   bytes at their peak and at the live peak, and their ranges); M8b's records are unchanged.
10. **The oracle mutation is shown detected by its CTest** (the mutant copy of the tests fails 4 of
    8); `parity/mutate.py` lists it in the suite `triad_reference`, whose run builds a whole
    `parity-cuda` copy and belongs to the mutation campaign of step 5 (criterion 11).
11. **`bench_suite.TOOLS` knows `triad_count`** (`perf_ab.triad_count`, executable
    `dyng-compat-escher`), step 1's deviation 12 resolved; the suites are step 5's.
12. **The records were re-run from the committed code**: the oracle, the replays, the draws and the
    memory pre-read from `1e6e933`, the timed pre-reads from `729bb79` (a later commit renamed two
    template parameters only); an earlier set of the same runs from the working tree gave the
    same verdicts and is not kept. A record's commit `+dirty` means only the records and docs were
    uncommitted.

### Machine time

| What | Time |
|---|---|
| go/no-go oracle replays (423 case replays in 9 runs; GPU 1 and the CPUs, shared lock) | about 20 min wall per full set (sum of group times: cuda 726 s, openmp 736 s, sequential 966 s); run twice (working tree, then the committed code) |
| `singles` (30 runs of A's unpatched main and of the oracle) | under 1 min |
| `draws` (22 groups, cuda, 2 at a time) | under 1 min |
| memory pre-read (3 cases x 2 processes under nsys, plus the plain runs) | about 5 min, twice |
| end-to-end pre-read (5 rounds x 2 cases, base lock, exclusive, plus 4 API traces) | about 3 min, twice |
| boost probe (3 rounds x 2 cases) | about 2 min, twice (the first attempt gave up before `--probe` existed) |
| `ci/check.sh` (all steps, then tidy / harness / format again after the fix) and the portability pre-checks (Clang 18 syntax over 366 files, the fortified cpu-only build) | about 35 min |

### Notes for step 3 (the CUDA port)

- **Budgets** (internal margin targets from 2.4, not tolerances): end to end, the counting may
  cost at most A's `init.full_count + sum(update.subtract + update.add)` plus 892 ms on
  coauth_b50k_s1 and **109 ms on syn1m_b50k_s1**; memory, the triad bytes live during
  `hypergraph.apply` at most 120.6 MiB (coauth_b50k_s1), **45.1 MiB (syn1m_b50k_s1)** and 88.4 MiB
  (syn2m_b50k_s1).
- **The syn gates run at base** (the boost probe does not hold; ADR 0021's rule).
- `--counter algorithm` is the stub in `cpp/tests/parity/escher/triad_replay.cpp`
  (`make_counter`): a counter class with `build()` (compute), `apply()` (`dyng::update` with the
  triad results; it must return the summary), `counts()` (baseline + deltas; also compare with a
  fresh `compute()`), and `launches()` (the launch counters per mode and shape, criterion 9).
- The driver's counting path: `run()` in `tools/compat/escher/escher_run.cpp` exits 2 without
  `--no-count`; step 3 adds compute (`full_count_ms`), the update (`update_ms`, between device
  synchronizations, the batch a host batch), A's count lines, and the port's NVTX batch range
  `triad_count.update` (`triad_perf.PORT_BATCH_RANGE`). The memory reading's batch range follows.
- The host syncs of the port's apply alone are 16-19 per batch (`cudaStreamSynchronize`), A's
  update with counting 90-96: the budget for the triad stages is the difference.
- The footprint between batches is the pool's reserve (1.63-1.69 x A's held bytes; reported,
  flagged above 1.05, not gated).

## Step 3: the CUDA port, the API and the early gate reading (cuda-port-api-early-gate)

**GO for step 4.** ESCHER-GPU's wedge engine is ported by a reproducible transplant and six
mechanical classes, each replayed exactly; the Op form (c6) leaves the counting kernels' machine
code identical; `triad_count` on cuda is exact on all 300 golden cases by key and by content; the
driver prints A's stdout byte for byte; the early gate reading meets every margin target (update
<= 1.011, full count <= 1.003, end to end <= 0.984; the Nsight Systems peak <= 1.0015)
(`parity/results/M9a.md` section 3). Nothing needs the author but the new public names (ADR 0052,
D9).

### What was built

| Commit | What |
|---|---|
| `be82dec` | `scripts/new_algorithm.py triad_count --container hypergraph --family aggregate_delta --backends seq,omp`; `ci/tests/test_regen.py` checks `hyper_sssp` as the planned algorithm |
| `645d6dc` | c1: the transplant (`engine/`, `rename_map_triad.toml`, `manual_hunks_triad.patch`), the public header `dyng/triad_count.hpp` (in `dyng.hpp`), the problem (`problem.hpp`, `triad_count.cpp`), the host recount, the CUDA glue (`cuda.cu`), `--counter algorithm` of `dyng_triad_replay`, the counting of `dyng-compat-escher run`, the retargeted fitness file, the manifest |
| `58b0895` | the test CMakeLists that c1 formatted by mistake, restored |
| `64d3b7c`, `29a02ad`, `679cec9`, `84f894b`, `c0793d6` | c2 errors, c3 memory, c4 streams, c5 registration and counters, c6 the Op form (`parity/triad_sass.py`) |
| `860fb1b` | the host recount on the engine's operations |
| `85efa0b` | the engine's per-call state without a heap allocation; compute()'s own offsets; the tests (host and cuda) and the kit for the four patterns |
| `7c78e8b` | the driver's `printHypergraphParams` lines; `triad_replay.py stdout`; the gpu_local step `triad`; the API baseline; the algorithm page |
| `7c75a45`, `40efdba`, `a7a53cc`, `091e667`, `a39b0bf` | the kit's type suites in their own translation unit (a GCC 12 false positive under `_FORTIFY_SOURCE=3`); the provenance symbols; the harness and replay tests of the algorithm counter; the manifest's `python` flag and its message's line length (ruff) |
| (the records commit) | `parity/results/M9a.md` section 3 and its records, ADRs 0052 and 0053, GOVERNANCE.md, plan.md, CHANGELOG, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The transplant commit carries the infrastructure** (as M8b's `3b73335`): the replay after c1
   needs the API, the problem and the glue. The transplanted files are the generated transplant
   (`transplant.py --check --rev 645d6dc`: identical). c1 also ran clang-format over the triad
   test CMakeLists by mistake; `58b0895` restored it (no history was rewritten).
2. **The counters stay classes in c1** (behind a shim of A's driver interface, a manual hunk) and
   become `count_full()` / `count_phase()` in c3: their only state was device memory, which c3
   moves to the workspace. The subtract and add bins are kept apart from c3 on; at c1-c2 the stats'
   `triads_removed` / `triads_added` were split from the net delta (transitional; the replays
   check counts only).
3. **The frontier is built by a kernel from c3 on** (the deleted ids uploaded once; the inserted
   ids from the applied record's device copy), not A's host flag vector; type 1 builds no flags
   (its key kernel never reads them). Type 1's full count runs over the h2v view's keys, types 2
   and 3 over the line graph's (A's `numHyperedgeKeys`); at c1-c5 the glue gave every pattern the
   h2v view's key count (the same results: the rows above the bound are empty).
4. **c6's bins handle tests the bin's sign only**: testing the upper bound too cost one
   instruction per triangle in MotifOp's hot loop, against the c6 rule (machine code equal to
   c5's). M9b's user operations may take a handle that tests both.
5. **The host recount uses the engine's operations** (`860fb1b`), not the scaffold's placeholder
   nor a copy of the set primitives: still a recount (`fallback_used`), as the step asks. Its
   `triads_removed` / `triads_added` are every triad before and after (ADR 0052 section 2.3), so
   the kit's cross-backend counters are the frontier sizes and the deltas until step 4.
6. **`stats::delta` and `max_bins`** are in the header (the sketch's open question 1, PLAN 5.5's
   `st.delta`); `count(bin)` throws outside `[0, num_bins())`; the vertex limit of the classifying
   merge is refused by `compute()` (h_motif, type 3) rather than counted (ADR 0052 section 2.5).
   The new names await the author (D9).
7. **compute()'s offsets have scratch arrays of their own**, released when it returns: released
   from the update's arrays, an update after a compute() reserved again and C8 failed. The
   engine's per-call state lives in the problem (a `new` per update was a host allocation C8
   counts).
8. **The kit runs the four patterns as four suites**: h_motif under the registered tag, the types
   under test-only tags with the manifest's registry entry (C0 reads the entry); their suites are in
   a second translation unit (GCC 12 -O3 with `_FORTIFY_SOURCE=3` reported a false `-Wnonnull` in
   `std::vector`'s assignment with all four in one).
9. **The compat tools need `triad_count` in the build** (the root CMakeLists): `dyng-compat-escher
   run` counts with it, so a build of a subset of the algorithms leaves the compat tools out, as
   it already did without mosp.
10. **The driver prints A's `printHypergraphParams` lines** for a synthetic input (A prints them
    inside its load stage), so its whole stdout is A's.
11. **The memory reading ran on all three gate cases** (the step names coauth_b50k_s1): the tight
    syn1m budget of step 2 is checked too.
12. **No container fix and no profiling were needed**: every margin target was met at the first
    reading, so `cpp/src/hypergraph` is unchanged and no ADR 0024 bar is due.
13. **The CBST source rules cover `cpp/src/algorithms/triad_count`** from c4 on (CTest
    `triad_count_source_rules`; the script takes `SOURCE_SUBDIRS`).
14. **A manifest says whether its Python module exists** (`python = false`, experimental only;
    `091e667`): the Python registry listed `triad_count` from the scaffold on, and the
    uniform-contract test of the Python layer imported `dyng.triad_count`, which M9b writes. The
    scaffold templates start with `python = false`, since they generate no bindings.
15. **`dyng_triad_replay_tests`** now checks the algorithm counter against the oracle on its own
    case (`a7a53cc`); the case that the counter is not available stays for builds without
    triad_count.

### Machine time

| What | Time |
|---|---|
| per-commit replays (make-test, syn1m_b50k_s1, coauth_b50k_s1; c2-c6 twice each: before the commit and recorded after it; c1 once from a worktree) | about 75 s each, 15 min in all; the c1 worktree build and replay 6 min |
| the full cuda parity (300 cases, both modes, recount per batch; GPU 1, 3 jobs) | 40 min wall (7,107 s of group time) |
| host make-test replays (dev build, both modes) | sequential 8 min, openmp 1.3 min |
| the kit: host (dev) / cuda (dev-cuda) | 8.4 min / 2.4 min |
| `triad_replay.py stdout`, 10 repeated runs x 4 patterns | 3 min, 12 min |
| early timing reading (5 rounds x 2 cases, GPU 0 exclusive) and memory reading (3 cases under nsys, GPU 1) | 2.7 min and 1.4 min |
| builds: parity-cuda (full), dev-cuda, sanitize-cuda, the fortified cpu-only build, Clang 18 syntax over 373 files | about 40 min in all (much of it shared with the parity run) |
| `ci/gpu_local.sh` (dev-cuda, GPU 1; the `triad` step's racecheck of make-test 12 min, the coauth batch under memcheck, synccheck and initcheck about 20 min) | about 3.5 h, all 20 steps passed (one earlier run stopped by me after the replay test change) |
| `ci/check.sh` (full), then python/regen/harness and scaffold/api again after `091e667` | 28 min, 2 min, 1.5 min |

### Notes for step 4 (the incremental host engines)

- The operations are host-compilable (`engine/count_ops.hpp`, `motif_ops.hpp`,
  `stathyper_ops.hpp`, `sets.hpp`); the host full mode is `host_count.hpp`. The frontier mode is the
  same loop over the frontier keys with all neighbours from `loc[f]`, closing `k > b` through
  `out[b]`, and the predicates of `engine/ownership.hpp` (`min_member_skips_wedge`,
  `min_member_skips_closing`); the flags cover `views.h2h.max_key + 1` keys.
- The deleted frontier is `normalized_->deletions` (begin_update keeps it); the inserted one
  `applied.record.inserted_ids` (host). count(-) must run in `count(old)` (it is a no-op on the
  host backends now) and count(+) in `count(new)` with `computing_` false; finalize then calls
  `apply_delta(minus, plus, stats)` like cuda, and `fallback_used` becomes false.
- Then add `triads_removed` / `triads_added` to the kit's `deterministic()` and drop the host
  branch of `TriadCount.UpdatesMatchTheOracleAndTheirStats` (the oracle's `removed()` / `added()`
  then apply on every backend); `ABinThatWouldGoNegativePoisonsTheResult` too.
- The host replays of the coverage table (sequential: make-test + the syn100k grid; openmp at 1
  and 8 threads: also the syn1m s1 grid and coauth_b50k_s1 types 1-2) take
  `--counter algorithm`.

### Notes for step 5 (gates, mutations, suites, certificate)

- The early readings: coauth_b50k_s1 update 0.949-0.980, full count 0.999, end to end 0.942;
  syn1m_b50k_s1 update 0.872-1.011, full count 1.003, end to end 0.984; nsys peak 1.0010-1.0015.
  syn1m's `update.subtract` is 1.015-1.027 of A's (inside update, which is gated).
- Mutation points: `engine/ownership.hpp` (the two filters: `triad_skip_frontier_filter_b` / `_k`,
  shared with the host engine of step 4), `apply_delta` and the phases' sign
  (`triad_wrong_bin_sign`), `triad_cuda_identify_affected` / the host's inserted frontier
  (`triad_inserted_requested_ids`), `count_full`'s type-1 key bound and the host's `num_sources`
  (`triad_type1_unbounded_keys`).
- `cpp/src/hypergraph` is unchanged: no ADR 0024 bar.
- The records of this step that a certificate could read are in `parity/results` (the cuda
  parity record `M9a-parity-cuda-escher_gpu.json` is the replay record format of compare.py,
  compacted).

## Step 4: the incremental host engines, the kit and the mutations (host-engine-kit-mutations)

**GO for step 5.** The sequential and OpenMP backends of `triad_count` are incremental (the
author's decision D3): the host engine transliterates ESCHER-GPU's wedge engine with the CUDA
engine's operations and exactly-once predicates and runs through `operators::count_delta`, which
enters `cpp/src/operators` with triangle_delta as its other user. The host backends are exact
against ESCHER-GPU on their coverage table by key and by content (OpenMP identical at 1 and 8
threads), every backend's stats match the oracle's phases, the kit passes on all three backends,
and the five triad mutations plus cbst_unfill_drops_last are detected on both engines
(`parity/results/M9a.md` section 4).

### What was built

| Commit | What |
|---|---|
| `164bfa5` | `operators::count_delta` / `count_owned` (ADR 0055): a required ownership rule (compile-failure CTests), the sign once per bin, private per-thread bins added in thread order on the host, atomic device bins read once on cuda; `download()` and `threads()` on the executors; triangle_delta's update counts through it; `dyng_operators_tests`, `dyng_operators_cuda_tests` |
| `779dcfd` | the host engine (`host_engine.hpp`, ADR 0054; `engine/phases.hpp` shared with the CUDA engine: the phase signs, the full count's keys, the launch outcome), the Tier A hooks with `identify_affected`, `triad_inserted_ids()` shared by both engines, the mutation hooks (inert); the tests (every backend against the oracle's phases, new hand cases, random multi-batch sequences under both id_reuse orders, OpenMP at 1 / 8 / 56 threads, the 128 Venn ids through the host bins handle, the kernel paths on every backend), the kit's deterministic counters and its OpenMP at 8 threads (`kit_openmp_threads`), the public header's and the algorithm page's text |
| `93ef4a1` | the mutation CTests `triad_count.mutation.*` (host and cuda, cbst_unfill_drops_last on the CUDA suite) and the suite `triad_count` of `parity/mutate.py` |
| `f20f5b3` | D5: `ci/test_data_budget.py` (1.25 MB; `ci/check.sh` step `testdata`, a pre-commit hook), ADR 0013 amended, `parity/triad_goldens.py fixture`, the fixture of A's make-test counts (3,737 bytes), `dyng_triad_count_make_test_tests` (cpu) and `..._large_tests` (parity, slow) |
| `c8b6e68` | `parity/mutate.py` finds the test executables of `cpp/tests/algorithms/<algorithm>` (its first triad run reported the control as a failed build) |
| (the records commit) | ADRs 0054 and 0055, GOVERNANCE.md, `parity/results/M9a.md` section 4 and its records, plan.md, CHANGELOG, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **count_delta has two entry points** (`count_delta`, signed; `count_owned`, unsigned): the
   full count of compute() keeps the unsigned 2^64 - 1 limit of the public contract, while a
   phase of an update is limited to 2^63 - 1 per bin on the host (signed bins; the documented
   `capacity_error`). The step names only `count_delta`.
2. **triangle_delta is count_delta's second user** (the plan's table names it; the tutorial
   algorithms count for the rule of two): its engine's `count_owned` became `count_delta(...,
   sign, ...)`. cycle_count is not moved (it needs an ADR 0024 timing bar; the step takes no
   timed readings) and `device_frontier` is not promoted (ADR 0055).
3. **A static partition dealt in chunks** (chunks of 64 sources dealt to the threads in turn), not
   contiguous ranges: still static and independent of the schedule; the per-thread bins are on
   each thread's stack and stored to the workspace for the thread-order sum.
4. **The host engine records launch counters too** (passes and early outs per mode and shape), so
   `KernelPathsAndEarlyOuts` runs on every backend; `triad_launch_counters` names passes.
5. **A frontier id outside the h2v view is an `internal_error`** (a cheap host check of every
   frontier id), added when `triad_inserted_requested_ids` crashed the host suite instead of
   failing it; it never fires on valid input.
6. **A new hand case** (the bound goes down in the batch that deletes the top ids) was needed for
   `triad_type1_unbounded_keys`: with the bound going down later, the rows past it are already
   empty and the mutation reads zeros.
7. **The mutation hooks sit in the host-engine commit** (`779dcfd`, inert without their defines);
   the commit after it adds the CTests and the mutate.py suite.
8. **`cbst_unfill_drops_last` against the triad suite** is the CTest
   `triad_count.mutation.cuda.cbst_unfill_drops_last` and the mutate.py entry
   `cbst_unfill_drops_last_triad` (a second name for the same define in the suite triad_count).
9. **The kit's OpenMP thread count is a trait** (`test_traits::kit_openmp_threads`, default 4;
   triad_count 8): `resources_for` takes the case type. Other algorithms' kits are unchanged.
10. **D5 is carried out in this step** (the step allowed it "only if the author raises the
    budget", which D5 did on 2026-10-15): the fixture holds the synthetic make-test groups (01,
    03, 07, 12; 3,737 bytes); the file-input groups run on stand-ins against the oracle (A's awk
    inputs are not reproduced). The hosted test runs types 1 and 2 on the syn100k groups and every
    pattern on the small ones; the triangle patterns on the heavy groups are a `parity`/`slow`
    test (5 min in Release).
11. **`parity/mutate.py` looked for test executables in `cpp/tests` only** (the triad suites are
    in `cpp/tests/algorithms/triad_count`): fixed in `c8b6e68` after its first run reported the
    control as a failed build.
12. **The host parity records were re-run from the committed build** (`c8b6e68`): the first runs
    used the uncommitted tree and recorded the old HEAD. The openmp 1-thread run of that first
    pass was stopped and only the rerun counts.
13. **Sanitizers on the triad, operator and triangle tests** (`ctest -L
    'triad_count|operators|triangle_delta'` in each sanitizer preset), not every CPU test;
    `tsan-openmp` with R020's local Clang 18 wrapper (recreated; it adds the rpath only when
    linking) and the conda Archer.
14. **gpu_local ran its `build` and `triad` steps only**, plus compute-sanitizer memcheck on the
    triad and operator CUDA tests and `ctest -L 'operators|triad_count|triangle_delta'` on dev-cuda;
    the full gpu_local run is step 5's.

### Machine time

| What | Time |
|---|---|
| host parity at `c8b6e68`: sequential (6 jobs), openmp 8 threads (3 jobs), openmp 1 thread (10 jobs) | 15, 10 and 32 min wall; the first pass from the uncommitted tree about the same |
| `parity/mutate.py run --suites triad_count` (7 parity-cuda copies) | about 60 min (one aborted first run of 10 min) |
| `ci/check.sh` (full) | about 60 min, concurrent with the sanitizers and the mutations |
| sanitizer presets (builds and the triad tests): asan, tsan, tsan-openmp | about 25, 15 and 20 min |
| gpu_local `triad` (the coauth batch under memcheck, synccheck, initcheck; the make-test racecheck 14 min) | about 75 min |
| dev, dev-cuda, parity-cuda builds and the kits | about 40 min |

### Notes for step 5 (gates, mutations, suites, certificate)

- `certify.py`'s interim rule (step 1) can now require the triad replays of all three backends per
  the coverage table, both modes: the records are `M9a-parity-cuda-escher_gpu.json` (step 3) and
  `M9a-parity-{sequential,openmp8,openmp1}-escher_gpu.json` (this step).
- The mutation record `M9a-mutate-triad_count.json` is the triad part of the 0.3 mutation
  evidence; `ctest -L mutation` covers the same mutations in every build.
- Nothing changed in `cpp/src/hypergraph` or the CUDA kernels' machine code: the step-3 gate
  readings stand; no ADR 0024 bar is due.

## Step 5: the gates, the ESCHER suites and the 0.3 certificate (gates-suites-certificate)

**M9a is complete; nothing stops at a gate.** Every gated row is within its bar against
ESCHER-GPU@abcf9b2 re-measured in the same session: the timing gates 45 / 45 (update per batch
0.858-1.028, full count 0.999-1.002, end to end 0.941-0.987; coauth-DBLP 11 rounds and syn1M /
syn2M 21 rounds, all at the base lock), the memory gates 11 / 11 (store accounting 1.0000 at every
gated point, the Nsight Systems peak 1.0010-1.0015). The final replays are exact on the coverage
table, the two ESCHER suites validate and pass, and the dry 0.3 certificate of `triad_count`
passes `recheck_gates` (`parity/results/M9a.md` section 5). **No C++ changed in this step.**

### What was built

| Commit | What |
|---|---|
| `33dcca4` | `perf_ab.py triad_count` records a suite can read (schema 2): every gated row with its gate and verdict, results by `<case>/<pattern>`, the reference and the port's build, the input digest at run start, the per-round guard, the kept contaminated rounds and the contamination audit; the memory reading's `device_memory` and `store_memory` (the gated points), the triad bytes during the apply (a non-counting third run), the footprint; `kernels` (ncu at the base clocks, the update's launches selected by name, skip and count); `parity/tests/test_triad_perf.py` |
| `faaf4e9` | `compare.py triad_count --certificate` and `triad_replay.py certificate --from` (the replays in the certificate's format) |
| `5ef6674` | `benchmarks/paper/ipdps26_escher.yaml`, `tkde26_escher_plus.yaml`; `bench_suite.py` for triad_count (case names, plan, dataset pins against the golden set, record checks); `benchmarks/README.md`, `docs/developer/release.md` |
| `4036351` | `certify.py`: both suites from (0, 3), the store rows rechecked point by point, a per-batch metric read by its row, the triad replays of every backend of the coverage table by key and by content from (0, 3), `write --dry-run <algorithm>`; `fixtures.toml` lists the committed make-test triad counts; the tests |
| `7f65284` | the memory rows from the peaks' bytes (found by the dry run), the compacted records drop the per-round stage timers |
| (the records commit) | ADR 0056 and its GOVERNANCE line, `parity/results/M9a.md` section 5 and `parity/results/M9a-gates/` (the summaries, the compacted records, the replays, the dry certificate), the kernel, maintain and SASS records, plan.md, CHANGELOG, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The gates ran from `4036351`, the final parity replays from the build of `7a2393f` /
   `4036351`**: the same library (`git diff 7a2393f 4036351 -- cpp tools cmake CMakeLists.txt
   CMakePresets.json` is empty); the commits after `4036351` change only Python, YAML and records.
   The certificate's rule (the library scope) is what makes them the release's readings.
2. **The suites' split is ours** (ADR 0056 section 2): ESCHER-GPU's repository does not tell
   ESCHER's readings from ESCHER+'s. The tkde suite's "batch sizes" reading is a memory reading
   (coauth 100K and 200K), so that the timing reading of the same batch sizes, in the ipdps
   suite, is not run twice.
3. **The ungated sub-stages are suite metrics row by row** (`update.subtract[b1]` ...), because a
   per-batch region's rows are its metric names (ADR 0048 section 1.5); `tkde26_escher_plus`
   lists only the memory metrics: a suite gates timing only where it has a gated run reading
   (`bench_suite.validate`).
4. **A contamination audit was added** (ADR 0056 section 4: load average above 16 or
   procs_blocked above 4 at a side's start or end, or a foreign GPU process, flags a gated
   reading, which the summary refuses). No reading was flagged; the monitor repeated two rounds.
5. **The triad bytes during `hypergraph.apply` are measured, not estimated**: the difference of
   the peaks inside `hypergraph.apply` of the counting port and of a third, non-counting run of
   the same process (2.8 / 1.4 / 2.4 MiB).
6. **The store gate on the synthetic inputs reads A's two print points**, the values the golden
   recording and A's own `[memory]` print agree on; the middle points are recorded (equal).
7. **`write --dry-run <algorithm>`** is how the dry certificate marks the other suites, golden
   sets, checks and fixtures as outside it; it refuses a results directory inside
   `benchmarks/results/`.
8. **The dry run found a harness bug and it was fixed before the records were summarized again**:
   the summary had put the record's byte ratio next to MiB rounded to 0.1, so `recheck_gates`
   could not recompute the `device_memory` rows. No reading was repeated (the records hold the
   bytes).
9. **The committed fixture of A's make-test counts is listed in `fixtures.toml`** (step 4 had
   committed it without the entry), so a certificate reports it with its hosted test.
10. **The compacted records drop the per-round stage timers** of triad_count (every row keeps its
    samples), to keep the committed records small (about 0.8 MB with the dry certificate).
11. **The kernel reading is per batch and phase** (12 launches of each process after the full
    count's 2), recorded, not gated; the step's "11 rounds" are kept.
12. **The mutation campaign ran the suites triad_count, cbst, hypergraph and kit** as the step
    asks; `triad_reference` (step 2's oracle mutation) is shown by its CTest in every build.
13. **The ADR index (`docs/adr/README.md`) gains the rows of ADRs 0054 and 0055**, which step 4
    left out, beside 0056's.
14. **The step was resumed after a pause** (the orchestrator stopped it at about 03:10 UTC on
    2026-10-16, at its end): the gate readings, committed code and records were kept (they were
    measured on the step's final library code); `ci/gpu_local.sh`, started at `7f65284` before the
    pause, ran to completion and is used as it is (nothing but docs and records changed after
    `7f65284`); the records were committed after the resume, with section 5.7-5.8 of
    `parity/results/M9a.md` (the checks and the acceptance criteria one by one).

### Measurements and checks

| Check | Tree | Result |
|---|---|---|
| the timing gates (`ipdps26_escher`, 9 cases, 45 rows) | `4036351` | **45 / 45 within**, 0.858-1.028 (results 5.1); no audit flag (5.2) |
| the memory gates (`tkde26_escher_plus`, 3 cases, 11 rows) | `4036351` | **11 / 11 within**: store 1.0000, nsys peak 1.0010-1.0015 (5.4) |
| recorded readings: default clocks, 100K / 200K batches, types 1-3, the kernels (ncu), SASS, the M8b maintain reading | `4036351` | 5.3 (kernels 1.001-1.006; SASS as step 4; maintain 0.814 / 0.851) |
| the final replays (`compare.py triad_count --counter algorithm --mode both`) | `7a2393f` / `4036351` (one library) | cuda 300/300, sequential 120/120, openmp 142/142 at 8 and at 1 thread (identical) (5.5) |
| the dry 0.3 certificate (`certify.py write --dry-run triad_count`) | `7f65284` | **PASSED** (after the fix of deviation 8); fails with a gate row or a replay removed (5.6) |
| `parity/mutate.py run --suites triad_count,cbst,hypergraph,kit` (one parity-cuda copy per mutation, sm_86) | `7f65284` | **control passed, 17 / 17 detected** (the five triad mutations and cbst_unfill_drops_last on the triad suite, 7 CBST, 3 hypergraph, the kit's planted bug); `M9a-mutate-step5.json`; 41 min |
| `ci/check.sh` (all steps) | `7f65284` + the uncommitted records | **all checks passed**: cpu-only 1,244 / 1,244, dev 1,277 / 1,277, consumer, tidy, reuse, provenance, testdata (1,002,671 of 1,250,000 bytes), regen, harness, python, api, scaffold (80 + 3 + 54 + 41), docs, pre-commit |
| sanitizers, `ctest --preset <p> -L 'triad_count\|operators\|triangle_delta' -LE gpu` | `7f65284` | asan (ASan + UBSan) **247 / 247**; tsan-openmp (Clang 18 + Archer) **247 / 247** (the `parity`/`slow` make-test tests included) |
| a fresh clone of `7f65284` | clone | cpu-only with `-D_FORTIFY_SOURCE=3` and no nvcc on PATH: built without warnings, `ctest -L cpu` **1,244 / 1,244**; parity-cuda: built, `ctest -L triad_count` **297 / 297** (GPU 1); the clone was removed |
| Clang 18 `-fsyntax-only` (`clang_syntax.py`, the dev database) | `7f65284` | 401 files, **0 failures** |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | `7f65284` | **all 20 steps passed** (`dyng-work/runs/m9a-step5/gpu_local.log`): `ctest -L gpu` 812 / 812, `ctest -L cpu` 1,277 / 1,277, the cuda golden corpora, compute-sanitizer memcheck / synccheck / initcheck 0 errors (the triad CUDA tests and one coauth_b50k_s1 batch included), racecheck 0 hazards (the triad kit's C1, C11 and C12, and the make-test replays: 15 groups x 4 patterns, both modes), the CBST and hypergraph replays, the plugin wheel (`pytest -m gpu` 61 passed), clang-tidy |
| frozen headers: `git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,graph}`, and against `main` (`9cde147`) | HEAD | **empty** (both) |
| `cpp/src/hypergraph` against M8b (`git diff 65a62c9 -- cpp/src/hypergraph`) | HEAD | empty: no ADR 0024 bar due |
| ESCHER-GPU's working tree (`git -C ESCHER-GPU status --porcelain`) | | empty |

### Machine time

| What | Time |
|---|---|
| the final replays: cuda (3 jobs), sequential (6), openmp 8 threads (3), openmp 1 thread (10) | 36, 11, 9 and 26 min wall (GPU 1 and the host, shared lock, concurrent) |
| the gate readings (both suites, the kernels, the maintain reading; exclusive lock on GPU 0, memory on GPU 1) | 22:57-00:15 UTC (78 min, after waiting 32 min for the replays' shared locks) |
| `parity/mutate.py` (18 parity-cuda copies) | 41 min (19:15-19:56 CDT) |
| `ci/check.sh`, the sanitizers, the fresh clone (concurrent with `ci/gpu_local.sh`) | about 70 min, 60 min and 25 min |
| `ci/gpu_local.sh` | 3 h 50 min (19:18-23:08 CDT; started one minute after `7f65284`, which changed only Python) |

## The review fixes (2026-10-20)

Independent reviewers confirmed 26 findings against `fe9c56c` (four lenses: parity and ownership,
gates and memory protocol, GPU robustness and sanitizers, integration and records). **All are
fixed in this branch; none needed a frozen header, a relaxed gate or a changed protocol, and no
stop condition was met.** The library changed (ADR 0057, the scratch headroom, the host model's
growth), so the replays and every gate reading were taken again on the final library
(`parity/results/M9a.md` section 6): exact on the whole coverage table, the timing gates 45 / 45
(update per batch 0.853-1.019), the memory gates 11 / 11 (store 1.0000, the Nsight Systems peak
1.0015-1.0021), the dry 0.3 certificate passed over the new records, and the step-5 records are
now refused by it (stale library, no count guard on the memory runs).

### The findings and their fixes

| # | Finding (severity) | Fix |
|---|---|---|
| 1, 14 | type 1: C(\|e\|, 3) in signed 64-bit, wrong from 2,097,152 members (UB on the host, a wrapped value and a poisoned update on cuda; the oracle wrapped from 2,642,246) (blocker / high) | `e1b0205` (ADR 0057): C(n, 3) divided before it is multiplied, exact up to n = 4,801,280 and capacity_error past 2^64 - 1; `8aa1b4b`: the oracle in 128-bit, `std::overflow_error` past 2^64 - 1; `dyng_triad_count_limits_tests` / `_cuda_tests` (2.1M, 2.2M, 3M, 4.8M members, compute and update, every backend) |
| 2 | no overflow detection on cuda (a type-1 total past 2^64 - 1 wrapped) (medium) | `e1b0205`: the capacity bit of the device error word, set by the checked operations and the carry-tested block and global bins, raised as capacity_error; the 14 x 2M case in the limits tests |
| 3, 15 | type-2 / type-3 products in signed 64-bit and unchecked sums on every backend and in the oracle (medium / high) | `e1b0205`: unsigned checked products (`__builtin_mul_overflow`, `__umul64hi`) and checked sums; `8aa1b4b`: the oracle in 128-bit; tests at 1.2e19 (exact) and 5e19 / 1.97e19 (capacity_error) |
| 16 | the signed phase bins refused a valid update of 2^63 or more triads of a bin (internal_error on cuda, a false capacity_error on the host) (medium) | `e1b0205`: unsigned phase magnitudes with their signs (`triad_phase_bins`), `count - removed + added` without an intermediate past 2^64 - 1; a net change outside `stats::delta`'s signed range is capacity_error with one message on every backend; tests of 1.2e19 removed, added and swapped |
| 17 | the frontier flags reserved exactly: a free and a malloc in count(-) of nearly every batch (criterion 7) (medium) | `b153ff3`: `scratch_buffer::reserve_with_headroom` (max(1.5 x need, 2 x capacity)) for the update's scratch on both engines; 50 equal-shape batches reserve in at most 5 updates; the nsys API trace of coauth_b50k_s1 at HEAD: 5 pool allocations in batch 1, none in batches 2 and 3 (ADR 0053 amendment) |
| 18 | the update guarantee left out a failure inside the commit (medium) | `b700e2e`: the header names the third outcome (h poisoned, r unchanged with G_t's counts); the OOM sweep test fails every device allocation of an update in turn; `update.hpp` (frozen) and `cpp/src/hypergraph` untouched |
| 4 | the golden set hashed the GPU (recipe, raw-trace digest) and `.zst` bytes (medium) | `98cfd8d`: run parameters in unhashed `recording.json` / `PROVENANCE.json`, content hashed decompressed, `export --twice --second-gpu`; `e015a12`: exported again on GPU 1 and GPU 0 from two fresh patched copies, identical `MANIFEST.sha256` (`4fb7dbcf...`), the counts and batches unchanged |
| 5 | the comparator never compared what the container deleted with A's batch (low) | `20840db`: `deleted == deletions` and `ignored == 0` in every mode; planted in the self-test |
| 6 | the oracle's isolation check ran `-MM` without the build's flags (low) | `285646e`: the OpenMP flags and the target's definitions passed; a planted include under `#ifdef _OPENMP` now fails |
| 7 | certify.py accepted a 0.3 summary missing one case's gated row (high) | `778f5b9`: every required row of every case of a triad reading (`triad_required_rows`), with tests dropping one row |
| 8 | the medians and `rounds` were trusted (medium) | `778f5b9`: the medians recomputed from the samples, one sample per round, at least the reading's rounds |
| 9 | the host-model asymmetry of the gated update (medium) | `756fb68`: recorded in the region map and ADRs 0048 / 0051, `update_with_host_model[b<N>]` and `update.maintain_untimed[b<N>]` reported beside the gate; `escher_host_model::apply` grows geometrically (see the note below and D12) |
| 10 | `--version` chose the rules (a 0.2.x label skipped the triad requirements) (medium) | `778f5b9`: a label of another release series than `VERSION` is refused |
| 11 | the memory readings had no count guard (medium) | `778f5b9`: the plain and the profiled runs of both sides pass the guard; a memory record without it is refused; the input-verification text corrected; the memory readings taken again |
| 12 | the protocol fields of the triad records were not validated (low) | `778f5b9`: GPU, lock, EAGER / no OMP_*, one layout, the unpatched original, `--sync-stages`; memory: GPU 1, the shared lock, D1's nsys command |
| 13, 23 | ADR 0056 named records that do not exist; the 5-round re-take in a quiet window (low) | `9e59d0f`: the paths corrected (the rounding claim was refuted by the verifier); the re-take: every reading taken again in full (section 6 of the results) |
| 19 | the build impact was not recorded (low) | recorded below (ci-cuda13 at `9cde147` and at the review's library) |
| 20 | the triad tests were likely to push the hosted cpu and sanitizer jobs past 60 minutes (high) | `eaf4299`: the host triad mutants run on their killing cases (0.3-3 s each in Debug), `cpu.yml` runs ctest on 4 jobs; `13f305e`: the triad kit's executable carries `slow` and leaves the hosted sanitizer jobs (it runs in `cpu.yml`, `ci/check.sh` and locally under the sanitizers); measured below |
| 21 | D4, D6-D10 (and D11) missing from GOVERNANCE.md's Open decisions (medium) | `9e59d0f` |
| 22 | the names of ADRs 0049 and 0051 accepted under delegation (medium) | `9e59d0f`: D9 lists them; ADRs 0049 and 0051 marked "technical parts accepted; names pending the author" |
| 24 | ADR 0052 still described recounting host backends (low) | `9e59d0f`: "amended by ADR 0054 and 0057", the recount text marked historical |
| 25 | the compile-fail test matched any diagnostic naming `count_phase` (low) | `20b320d`: a `static_assert` overload and its message matched |
| 26 | a stale comment above `kit_openmp_threads` (low) | `20b320d` |

### Deviations (each with the same intent)

1. **M9a criterion 6's "Device bins are signed 64-bit, with minus and plus kept apart"** becomes
   (ADR 0057): the device bins are unsigned 64-bit magnitudes, minus and plus kept apart, the
   signs recorded and applied in `apply_delta`; a bin that would go negative still raises
   internal_error and poisons the result. The signed bins could not hold a valid phase of 2^63 or
   more triads.
2. **ESCHER-GPU has the same overflows** (`Choose3Op`, `Type2Op`, `Type3Op`, plain `atomicAdd`s):
   dynG is exact where A wraps or is undefined. No golden case comes near the limits (coauth's
   largest hyperedge has 280 members), so the parity stays exact; the deviation from A is
   recorded here and in ADR 0057.
3. **The machine code of the counting kernels changed** (`M9a-review-sass-escher_gpu.json`):
   count_wedges identical to A; keys<choose3> 152 -> 288 instructions, pairs<type2> 456 -> 512,
   triangles<type3> 512 -> 576 (the checked arithmetic); triangles<motif> 536 -> 560 (step 5: 552;
   motif_op's per-triangle adds stay unchecked, ADR 0057 section 1, the global flush is
   carry-tested). The gated h_motif kernels read 1.002-1.006 of A under ncu (step 5:
   1.001-1.006). StatHyper types 1-3 stay below A on coauth (update 0.72-0.96).
4. **The triad conformance kit left the hosted sanitizer jobs** (label `slow`): under ASan and
   TSan-OpenMP its four C2 cases had not finished after 13 minutes in a hosted-runner emulation.
   It still runs in the hosted `cpu.yml` jobs, in `ci/check.sh`, and under both sanitizers
   locally at each milestone (as here: 258 / 258 each).
5. **The reviewers' "re-take two 5-round readings in a quiet window"** was met by taking every
   reading again in full (the gated ones at 11 and 21 rounds), because the library had changed.

### Notes

- **The host model in the gated update (D12, for the author).** The gate compares the library's
  `update` with A's `update`, as approved; A's timer contains its host model (1.5-2 ms a batch,
  7-13 ms where its vector grows), the port's driver keeps its host model after its timer. The
  driver's host model (`escher_host_model::apply`: it validates, sorts and copies each inserted
  row) costs 5.7-6.6 ms a batch (11-18 ms where the vector grows), so the reported
  `update_with_host_model` row reaches 1.057-1.061 on three syn2m rows at the base lock (1.070 on
  syn2m b2 at default clocks), while the library-only comparison (A's update minus its untimed
  in-maintain time against the port's update) is at most 1.031. Nothing in the gate changed; a
  cheaper driver host model (it sorts rows that the generator already gives sorted) is a library
  change (`cpp/src/generators`) that would make every reading stale again, so it is left for the
  author's decision rather than made at the end of the review.
- **The footprint between batches** is 1.63-1.80x A's held bytes (D11, unchanged in kind; the
  scratch headroom adds 1.4 MiB at most to the triad bytes during the apply: 4.2 / 2.1 / 3.6 MiB
  against 2.8 / 1.4 / 2.4).

### Build impact (finding 19)

The compile-only `ci-cuda13` preset (CUDA 13.1, release architectures, every instantiation and
mutant), built from `git archive` copies at the M9a base `9cde147` and at the review's library
(`e015a12`), `-j 28` on the shared machine (load 40-95, so the step times are upper bounds;
CPU-minutes are the sums of `.ninja_log` step times):

| | `9cde147` (M8b merged) | M9a (review fixes) |
|---|---:|---:|
| wall | 355 s | 314 s |
| CPU-minutes, the whole preset (885 / 1,032 steps) | 153.8 | 138.3 |
| of which the six triad host mutants and cbst_unfill_drops_last's copy (one architecture each) | - | 12.4 |
| of which the triad library units (`counters.cu` 51 s, `cuda.cu` 52 s, the host units 8 s) | - | 1.8 |
| `libdyng.so` unstripped / stripped | 40,244,088 / 39,149,600 B | 41,659,800 / 40,516,800 B (+3.5 %) |

The two totals differ by less than the machine's noise (the base build ran at a higher load);
what M9a adds is about 14 CPU-minutes, inside the hosted `cuda-build`'s 120-minute timeout. The
plugin wheels are built and measured in M9b's packaging step; at +3.5 % of libdyng they stay far
below the 90 MB budget (M8b: `dyng-cu13` 8.19 MB, `dyng-cu12` 8.27 MB).

### Hosted-CI margins (finding 20)

`ctest -L cpu` on 4 jobs pinned to 4 hardware threads of the loaded machine (a hosted-runner
emulation; the hosted jobs also build first): dev preset (Debug) 1,280 / 1,280 in 703 s; asan and
tsan-openmp with `-LE slow` (what `sanitizers.yml` runs after `13f305e`) 1,139 / 1,139 in 515 s
and 981 s. Before `13f305e` the asan run was stopped after 821 s with the kit's C2 cases still
running. The 60-minute job timeouts keep their margin; no timeout was raised.

### Checks of the review fixes

| Check | Tree | Result |
|---|---|---|
| the replays (`compare.py triad_count --counter algorithm --mode both`) | `e015a12` (the final library) | cuda 300/300, sequential 120/120, openmp 142/142 at 8 and 1 threads (identical), both modes |
| the gate readings (both suites, the kernels, the maintain reading) | `2eda830` (clean; the final library) | timing 45 / 45 within (0.853-1.019, 0.998-1.003, 0.915-0.983), memory 11 / 11 (1.0000; 1.0015-1.0021); the dry 0.3 certificate **PASSED** |
| `parity/mutate.py run --suites triad_count,cbst,hypergraph,kit` (one parity-cuda copy per mutation, GPU 1) | `18a4a75` | **control passed, 17 / 17 detected** (`M9a-review-mutate.json`; 34 min) |
| `ci/check.sh` (all steps) | `18a4a75` | every step passed but provenance (and pre-commit's provenance hook): the new `engine/exact.hpp` names ESCHER-GPU's operations; listed in NOT_DERIVED (`0e6e6fb`, `ci/` only), then `DYNG_CHECK_ONLY="provenance precommit" ci/check.sh` **passed**; cpu-only 1,247 / 1,247, dev 1,280 / 1,280, consumer, tidy, reuse, testdata (1,002,671 of 1,250,000 bytes), regen, harness (337 passed), python, api (1,168 declarations), scaffold, docs |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | `18a4a75` | **all 20 steps passed** in 3 h 43 min: `ctest -L gpu` 819 / 819 (the CUDA limits tests of ADR 0057 and the OOM sweep included), `ctest -L cpu` 1,280 / 1,280, the cuda golden corpora, memcheck on every gpu executable, synccheck, racecheck, the CBST and hypergraph sanitizers and replays, the triad step (synccheck and initcheck on the triad CUDA tests, memcheck / synccheck / initcheck on one coauth_b50k_s1 batch, racecheck on C1, C11, C12 and the make-test replays in both modes), the plugin wheel (`pytest -m gpu` 61 passed; `dyng-cu13` 8.21 MB for the native architecture), clang-tidy |
| ASan + UBSan and TSan-OpenMP (Clang 18 + Archer), `ctest --preset <p> -L 'triad_count\|operators\|triangle_delta' -LE gpu -j 8` | the final library (built at 16:30 after `756fb68`) | **258 / 258** each (the host limits tests included) |
| hosted-CI emulation, `ctest -L cpu -j 4` on 4 hardware threads | dev, asan, tsan-openmp | 1,280 / 1,280 in 703 s; 1,139 / 1,139 in 515 s and 981 s (`-LE slow`) |
| Clang 18 `-fsyntax-only` (`clang_syntax.py`, the dev database) | `e015a12` | 420 files, **0 failures** |
| cpu-only with `-D_FORTIFY_SOURCE=3`, no nvcc on PATH | `e015a12` | built without warnings |
| frozen headers against `65a62c9`; `cpp/src/hypergraph` against `65a62c9` | HEAD | empty (no ADR 0024 bar due) |
| ESCHER-GPU's working tree | | `git status --porcelain` empty |
| signatures | `9cde147..HEAD` | every commit SSH-signed (`G`), identity S M Shovan |

### Machine time

| What | Time |
|---|---|
| the replays (cuda 3 jobs, sequential 6, openmp 8 threads 3, then 1 thread 10) | 36, 15, 13 and 30 min wall (concurrent, the shared lock) |
| the gate readings (both suites, the kernels, the maintain reading) | 18:48-19:43 CDT (55 min, exclusive lock on GPU 0; memory on GPU 1) |
| the golden export twice (two fresh patched copies, GPUs 1 and 0) | ended 17:41 CDT (a first attempt failed when a concurrent rebuild replaced libdyng under it, and was run again) |
| the build impact (two ci-cuda13 builds) | 12 min |
| `parity/mutate.py`, `ci/check.sh` and `ci/gpu_local.sh` (concurrent, the shared lock) | 34 min, 45 min and 3 h 43 min (19:47-23:31 CDT) |

## The milestone

### Deviations of M9a from PLAN 6.4.6, 4.6 and 11.3 (the milestone)

- **PLAN 6.4.6**: the parity runs by content **and** by key on every backend from the start (the
  M8 id parity held); the coverage per backend is the plan's default (D6, pending the author);
  the gates are read against A re-measured in the same session, never against the README's
  figures (which are recorded as `paper_reference`); the paper's coauth gate is read per batch
  (`update[b1..b3]`) rather than as one figure; the device memory is read twice, as store
  accounting (2026-10-07) and as the Nsight Systems peak (D1). The open bins 21-26 stay 0
  (verified on every golden). The `experimental::count_local_patterns` exposure and its tutorial
  are M9b's (D2).
- **PLAN 4.6**: the host backends are incremental host engines, not recount-based references (D3,
  ADR 0054); the independent oracle is `testing::triad_reference`.
- **PLAN 11.3**: M9 runs as M9a and M9b (D2); `triad_count` stays experimental and not frozen
  until M9b (D4); the 0.3 certificate's plumbing (both suites, the triad replays from 0.3) is in
  M9a, the release's certificate itself at 0.3.0.

### Effort

M9a's five steps ran on 2026-10-15 (CDT), from the region map (step 1, early morning) to this
step's records (evening): one day of wall-clock time for the milestone, against the plan's
re-baselined "M9 in 3-5 days plus the review and fix steps" (M8b's retrospective). Machine time
dominated the later steps: the golden export (62 min), the replays (about 1.5 h per full set of
backends), the mutation campaigns (about 1 h in step 4, 41 min here), `ci/gpu_local.sh`
(about 3.5 h in step 3, 3 h 50 min here) and the gate readings (78 min). The review fixes took
one more day (2026-10-20), most of it machine time: the replays, the golden export twice, every
gate reading again and the final checks ("The review fixes" above).

### The author's pending decisions (collected for the 0.3.0 release)

| # | Decision | State |
|---|---|---|
| D1 | the gated Nsight Systems peak (live allocations, whole process) | the author's (2026-10-15); read in 5.4: 1.0010-1.0015, at the review's library (results 6.4): 1.0015-1.0021 |
| D2 | M9 as M9a and M9b | the author's (2026-10-15) |
| D3 | incremental host engines, the oracle independent | the author's (2026-10-15); ADR 0054 |
| D4 | `triad_count` frozen and stable at 0.3.0 at the end of M9b; `count_local_patterns` experimental | **pending**: taken on the plan's default |
| D5 | the 1.25 MB test-data budget | the author's (2026-10-15); 1,002,671 bytes used |
| D6 | the parity coverage per backend (cuda every case; sequential make-test + syn100k; openmp + syn1m s1 + coauth_b50k_s1 types 1-2) | **pending**: the plan's default; the 0.3 certificate requires exactly this table |
| D7 | the gate rows, cases and rounds (ADR 0048 section 1.4) | **pending**: the plan's default; read as such in 5.1 |
| D8 | the suites `ipdps26_escher` and `tkde26_escher_plus` and their split | **pending**: the plan's default; the split is ours (ADR 0056) |
| D9 | the new public names of `triad_count` (ADR 0052), of the oracle `testing::triad_*` (ADR 0049) and of the generators' host model and the port driver (ADR 0051) | **pending**: naming is not delegated (the review found ADRs 0049 and 0051 logged as fully accepted; corrected) |
| D10 | the "paper vs fixed code" wording | **pending**: the plan's default (M9b's docs) |
| D11 | the footprint between batches (the stream-ordered pool's reserve, 1.63-1.80x A's held bytes on every reading; reported and flagged above 1.05 under D1, not gated) | **new, for the author**: whether a later release trims the pool between updates (a library change outside M9a's scope) or the footprint stays reported only |
| D12 | the host model in the gated update (A's `update` contains its host model, the port's driver keeps its own after the timer; `update_with_host_model` reported beside the gate reaches 1.057-1.061 on three syn2m rows, results 6.2) | **new, for the author** (the review): the gate as approved and the asymmetry reported, or a protocol counting a host model on both sides, or a cheaper driver host model |

Also for the author from earlier steps: the manifest key `python = false` (step 3, deviation 14)
and the test trait `kit_openmp_threads` (step 4) were taken under delegation as technical items;
merging the M9a pull request (a merge commit, D2) is the author's.

### Ready-to-paste pull request summary

```markdown
## M9a: triad_count exact against ESCHER-GPU, with its gates (0.3)

`triad_count` (experimental; `<dyng/triad_count.hpp>`): ESCHER-GPU@abcf9b2's 30-bin h-motif
counts and StatHyper's triad types 1-3, kept exact under hyperedge batches on the sequential,
OpenMP and CUDA backends (PLAN 6.4.6; the author's decisions D1-D3 and D5 of 2026-10-15).

**Parity (exact).** The golden set `triad_count_escher_gpu` (PLAN 6.4.6's grid of 240 cases plus
ESCHER-GPU's `make test`, 300 cases, exported twice with identical manifests; ESCHER-GPU agrees
with its own CPU reference on 90 runs) replays exactly by key and by content: cuda 300/300,
sequential 120/120, OpenMP 142/142 at 1 and 8 threads (identical). An independent oracle
(`dyng::testing::triad_reference`, its own class map) reproduces ESCHER-GPU everywhere and checks
every backend's stats. Kit (C1-C12 but C4) on every backend; the five triad mutations of PLAN 8.4
and cbst_unfill_drops_last detected on both engines. Counts are exact up to 2^64 - 1 on every
backend and refused past it with capacity_error (ADR 0057; ESCHER-GPU's StatHyper arithmetic
overflows from 2,097,152 members per hyperedge, found by the review).

**The CUDA port** is a reproducible transplant of ESCHER-GPU's wedge engine in six mechanical
commits; count_wedges' machine code equals ESCHER-GPU's, the h-motif triangle kernel adds an error
word and a carry-tested flush (+24 instructions), the StatHyper kernels checked arithmetic.
**The host engines** are incremental (D3), through `operators::count_delta` (ADR 0055).

**Gates** (against ESCHER-GPU re-measured in the same session, base clock lock, GPU 0, exclusive
lock; `benchmarks/paper/ipdps26_escher.yaml`, `tkde26_escher_plus.yaml`):

| | coauth-DBLP 50K s1-s3 (11 rounds) | syn1M / syn2M 50K s1-s3 (21 rounds) | bar |
|---|---|---|---|
| update, each of 3 batches | 0.949-0.971 | 0.853-1.019 | 1.05 |
| full count | 0.998-0.999 | 1.002-1.003 | 1.05 |
| end to end | 0.915-0.916 | 0.967-0.983 | 1.10 |
| store accounting (every gated point) | 1.0000 | 1.0000 | 1.05 |
| Nsight Systems peak of live allocations (D1) | 1.0015 | 1.0021 / 1.0017 | 1.05 |

Read on the final library, after the review fixes (`parity/results/M9a.md` section 6). Host
synchronizations per batch 19-22 against ESCHER-GPU's 90-96. The footprint between batches (the
stream-ordered pool's reserve, 1.63-1.69x) is reported, not gated (D1). Reported beside the update
gate: ESCHER-GPU's update contains its host model and the port's driver keeps its own after the
timer; with the driver's host model added the port is up to 1.061x on three syn2m rows (D12).

**Certificate.** From 0.3 `parity/certify.py` requires both ESCHER suites (the store rows
rechecked point by point) and the triad replays of every backend of the coverage table by key and
by content, every gated row of every case with its medians recomputed from the samples, and the
protocol fields; the dry 0.3 certificate of `triad_count` passes
(`parity/results/M9a-review-gates/`).

**Checks** (at the final library, after the review fixes). `ci/check.sh` all passed; `ci/gpu_local.sh` all 20 steps passed on GPU 1 (819 GPU tests; memcheck, synccheck and initcheck on the triad CUDA tests and a coauth batch; racecheck on C1, C11, C12 and the make-test replays); ASan/UBSan and
TSan-OpenMP on the triad tests; a fresh clone (CPU-only fortified, CUDA); Clang 18 syntax;
frozen headers unchanged (`git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,graph}`
empty); the mutation campaign 17 / 17 detected.

**For the author**: D4 and D6-D10 are taken on the plan's defaults pending the 0.3.0 release
(GOVERNANCE.md); the new public names (D9: ADRs 0049, 0051, 0052); D11, whether a later release
trims the pool between updates; D12, the host model in the gated update. ADRs 0048-0057. Records: `parity/results/M9a.md`,
`docs/developer/retrospectives/M9a.md`. M9b (count_local_patterns, Python, the CLI, the freeze)
follows on a stacked branch. Merge with a merge commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
