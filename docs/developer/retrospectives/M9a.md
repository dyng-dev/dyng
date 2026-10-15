# Retrospective: M9a (triad_count against ESCHER-GPU)

Status: **in progress** on the branch `m9a-triad-count`. M9a is the first half of PLAN 11.3's
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
