# ADR 0048: The triad_count parity and gate protocol (the timed regions, the golden set, the original's cross-check)

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M9a step 1 (PLAN 6.3 steps 1-3, 6.4.6, 8.3, 8.6;
  the M9a acceptance criteria 1, 2 and 3). The protocol itself is the author's: PLAN 8.6 (the
  tolerances, the A/B protocol), the decisions of 2026-10-07 (M8's memory gate read as store
  accounting; O16 `erase_first`), 2026-10-08 (layout control: the one-layout protocol of ADRs
  0018 / 0021 against the originals, the originals built exactly as their Makefile builds them,
  no alignment flags) and 2026-10-15 (D1, the gated Nsight Systems peak). This ADR changes none
  of them: it writes down which timer of the original is compared with which stage of the port,
  and how the reference side (the goldens and the original's own cross-check) is produced.
  **Pending the author** (taken on the plan's defaults until the 0.3.0 release, decisions D6 and
  D7 of 2026-10-15): the coverage table of section 2.6 and the gate rows, cases and rounds of
  section 1.4.
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step regions-goldens-crosscheck)
- **Builds on:** ADR 0013 (parity and goldens), ADR 0018 and ADR 0021 (clocks), ADR 0024 (dynG
  against dynG), ADR 0036 (the CBST trace recorder and its batch dumps), ADR 0041-0045 (the
  hypergraph container, its line graph and row views, its update path, its files and
  generators).

## Context

`triad_count` is the port of ESCHER-GPU@abcf9b2's (A's) dynamic triad counters: `build/main`
(the 30-bin h-motif counts) and `build/type1`, `type2`, `type3` (StatHyper's three triad types),
four executables around one driver (`src/driver.cu`, `runDynamicCounting`). PLAN 6.3 asks for
three things before any port code: the timed-region map (step 3; without it "<= 1.05x of the
paper-timed region" is not defined), the goldens exported from the pinned original (step 1,
PLAN 8.3), and the original checked against its own second implementation (step 2: A's CPU
reference, `utils/reference.cpp`, behind `--reference`). PLAN 6.4.6 fixes the grid (synthetic
100K / 1M / 2M and coauth-DBLP; seeds 1-3; batch sizes 20, 5K, 50K, 100K, 200K; 3 consecutive
batches) and the replay rule (deletions replayed by content, by key once the M8 id parity
holds; both must agree, D.1 T6).

## Decision

### 1. The timed-region map (`parity/timed_regions/triad_count.toml`)

1. **Gated rows** (PLAN 8.6 tolerances only):
   - `update`, read **per batch**: one gate row per batch, `update[b1]`, `update[b2]`,
     `update[b3]` (`per_batch = true`, `rows`). A's `timing:update` (subtract + maintain + add,
     a synchronized ScopedTimer) against the port's `triad_count.update` (normalize + count_minus
     + commit, which contains `hypergraph.apply`, + identify_affected + count_plus + finalize),
     reported by the driver as `update_ms`: steady_clock around `triad_count::update()`, started
     after a device synchronization and ended when the host holds the delta. Gate `compute`
     (1.05). The sum of the three batches is reported (`report_sum = true`), never gated.
   - `full_count`: A's `timing:init.full_count` against the stage of `compute()`. The framework
     emits `triad_count.compute` for the whole call (`hook_stages::compute`); the sketch's name
     `triad_count.count` is the aggregate-delta count inside it (`hook_stages::count`) and is
     recorded as a sub-stage. The gated stage is `triad_count.compute`, because A's
     `counter.count()` also allocates its bins and reads them back (`full_count_ms`, after a
     device synchronization). Gate `compute`.
   - `end_to_end`: A's `timing:total` (from the start of `run()` to after the last batch)
     against the driver's total from the start of its `main()` (warm_up, the CUDA context, the
     load, the build and upload, compute, the generation and the updates). Gate `end_to_end`
     (1.10).
2. **Reported rows:** `update.subtract` = normalize + count_minus; `update.maintain` = commit;
   `update.add` = identify_affected + count_plus + finalize (the step text names count_plus +
   finalize; identify_affected, the frontier of the assigned inserted ids, is A's
   `DeviceFrontier` inside `update.add`, so it is counted there and the three sub-rows add up to
   `update` on both sides); `build`, `upload`, `load`, `batch.generate` (per batch) and the
   process wall time.
3. **Scope rules**, the same on both sides:
   - the line graph and the row views are built at construction (`line_graph_policy::
     sorted_merge`, as A's `DynamicHypergraph::build` builds H2H and its row views), never inside
     a gated region;
   - the batch is a **host** batch when `update` starts: Step 0's staging is inside the region,
     as A uploads the deleted keys inside `update.subtract`; no prestaged `.dgb` or device batch;
   - batch generation is outside the region on both sides (A: `batch.generate`; the port draws
     with `generators::legacy` before its timer starts);
   - no `--check` and no `--reference` on either side in a timed round;
   - the port runs with `--sync-stages` (the profiler synchronizes at stage boundaries, as A's
     ScopedTimer synchronizes at its start and end);
   - **one asymmetry** (recorded in the M9a review, 2026-10-20): A's `update.maintain`
     (`DynamicHypergraph::update`) also maintains A's host model of the rows (`h2v_`: the
     deleted keys' rows cleared, the vector resized, the inserted rows assigned) inside the gated
     `update`, while the port's driver maintains its host model (`escher_host_model`, ADR 0051)
     after its update timer, timed apart as `host_model_ms`. The gate compares `update` as each
     side times it (the region map does not change); `parity/triad_perf.py` reports beside it,
     never gated, `update_with_host_model[b<N>]` (A's update against the port's update plus
     `host_model_ms`) and `update.maintain_untimed[b<N>]` (A's maintain minus its `maintain.*`
     sub-stages, about 1.5-2 ms per batch and 7-13 ms in a batch where `h2v_` grows, against the
     port's `host_model_ms`). The M9a reading at HEAD records both (`parity/results/M9a.md`
     section 6).
4. **The gate readings** (PLAN 6.4.6, 8.6; D7 pending the author): coauth_b50k_s1-s3 at the base
   clock lock (ADR 0021; the boost lock does not hold for A's process on coauth-DBLP, M8b),
   at least 11 rounds; syn1m_b50k_s1-s3 and syn2m_b50k_s1-s3 at least 21 rounds, at boost when a
   recorded 3-round probe holds the lock, otherwise at base; each against A re-measured in the
   same session (exclusive perf lock, GPU 0, `CUDA_MODULE_LOADING=EAGER` on both sides, no
   `OMP_*`, the contamination monitor, one heap layout, A/B alternation, medians). The memory
   gates are the author's: store accounting (2026-10-07) and the Nsight Systems peak of live
   device allocations of both whole timed processes (D1, 2026-10-15).
5. **The loaders.** `parity/perf_ab.py load_regions` refuses an unknown original timer, a region
   without port stages, **an unknown key** (new: a misspelt `per_bach` would otherwise be ignored)
   a repeated region name, and per-batch rows that are not `<name>[b1]`, `<name>[b2]`, ... in order; `region_rows()` and
   `batch_of_row()` expand and fold the rows. `parity/bench_suite.py region_map()` lists a per-batch
   region by its rows, so a suite names `update[b1..b3]` as metrics; `parity/certify.py`'s
   `REGION_SUFFIX` already folds `update[bN]` back into `update`. Tests:
   `ci/tests/test_triad_regions.py`.

### 2. The golden set `triad_count_escher_gpu`

1. **Cases.** (a) **make-test**: every `build/{main,type1,type2,type3}` invocation of A's
   `tests/run_tests.sh` that counts (60 invocations in 15 groups of four executables with the same
   arguments); the seven input-rejection invocations of `build/main` print no counts and are not
   cases. Their gawk-generated inputs (`small.hg`, `skewed.hg`, `medium.hg`) are taken from M8a's
   set `cbst_trace_escher_gpu_test` (checked against its recorded SHA-256), and the exporter
   checks that its 60 argument lists are exactly the set's main/type1/type2/type3 invocations.
   (b) **the grid** of PLAN 6.4.6: syn100k `100000 10 1 33333`, syn1m `1000000 10 1 333333`,
   syn2m `2000000 10 1 666666` (A's default random inserts) and coauth `--input coauth.hg`
   (A's default sample inserts), x `--seed` 1, 2, 3 x `--batch-size` 20, 5000, 50000, 100000,
   200000 x `--batches 3` x the four executables: 240 cases named
   `<dataset>_b<size>_s<seed>/<executable>`.
2. **Counts** come from A's **unpatched** executables, run without `--check` and `--reference`,
   parsed from stdout (`Motif counts (30 bins):`, `Batch b (D deletions, I insertions)`,
   `Motif delta counts (30 bins):`, `[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): X`,
   `[GPU] TypeN baseline motif count: X`, `[GPU] TypeN motif delta count: X`). Each case runs a
   second time with `--check`; it must exit 0, report no `FAILED`, and print byte-identical count
   lines. The unpatched executables of an export are those of its own fresh patched copy: the
   export patch keeps A's Makefile-built executables unchanged as `parity_export/orig/<exe>`
   before it installs the recording twins (ADR 0036), so every export is self-contained and the
   double export compares two independent builds. (The persistent unpatched copy, the performance
   baseline, is run by the cross-check of section 3 and must print the goldens' count lines.)
3. **Batches** come from one run of the **patched** recording twin of `build/main` per (dataset,
   seed, size), in the recorder's batches mode (ADR 0036; enough for coauth-DBLP, 15 s per run, so
   no new mode was needed; the exit record already carries `inserted_keys`), converted by
   `dyng-compat-escher to-dgb` and split into one `.dgb` per batch (`batch<N>.dgb`: the file
   header with a batch count of 1 and that batch's record, flags `deleted_members` and
   `expected_ids`, id = key - 1), so that every batch has the SHA-256 of its own batch file. The
   same RNG stream drives the four executables (the counters draw nothing), so the batch lines
   of type1-3 must equal main's on every case, and once per dataset (seed 1, size 5000; for
   make-test the group `small_b20`) the twins of type1, type2 and type3 record their batches too,
   whose files must be byte-identical to main's.
4. **Inputs**, each distinct one stored once under `inputs/`: A's loaded rows (the build record)
   written as a `.hg` file (`index_base` 1; hyperedge i is A's key i + 1), whose SHA-256 is the
   input's **loaded-rows hash**. coauth-DBLP is not stored: its loaded rows are byte-identical to
   `datasets/hyper/coauth.hg` (SHA-256 `65fc83b2...`), which the set names. Synthetic inputs are
   recorded as legacy-generator parameters (A's `n m r1 r2` and the seed) plus the loaded-rows
   hash, and `generators::legacy::escher_hypergraph` (through the Python binding) must write a
   `.hg` with the same hash for every (dataset, seed); the stored `.hg` files of syn1m and syn2m
   are the per-seed files the paper suites pin. The make-test file inputs are stored as given to
   A as well (`<name>.source.hg`).
5. **Records.** One JSON per case: the reference and its commit, the executable, the argument
   list (input paths written as `<input:name>`), the input's digest or generator parameters, the
   seed, the batch composition and insert mode, `full_count` (30 values or 1) and per batch
   {index, deletions, insertions, delta, counts_after (derived: the full count plus the deltas so
   far), the batch file's SHA-256}, and the `--check` run's verdict. A group JSON holds the
   recording run's facts (vertex count, H2V payload capacity, A's `numHyperedgeKeys` and
   `deviceBytes()` after the build and after every batch, from the trace's batch records). A's
   unpatched stdout of the make-test cases is kept for M9b's CLI. Nothing that varies between
   runs is hashed: timings, the `[memory]` print points of stderr, paths and the GPU's name are
   not in any hashed file. Large files are compressed with zstd (`-12 --long=27`, the version is
   recorded); raw traces are dropped after conversion, the group JSON keeping their SHA-256 and
   the recipe.
6. **Coverage** (pending the author, D6): cuda replays every case; sequential make-test plus the
   syn100k grid; openmp the same plus the syn1m seed-1 grid and coauth_b50k_s1 for type1 and type2.
   Recorded in `parity/goldens.toml` next to the set.
7. **Export twice.** `parity/triad_goldens.py export --twice` builds two FRESH `git archive
   abcf9b2` patched copies the M8a way (`parity/build_reference.sh --variant patched --fresh
   --scratch $DYNG_SCRATCH/ref/m9a-export-<1|2>`), exports from each, requires byte-identical
   `MANIFEST.sha256` files and removes the copies.
8. **verify** (`parity/triad_goldens.py verify`, also `parity/compare.py triad_count --verify`):
   the manifest against `goldens.toml`, every file against the manifest, every case digest; bins
   20-25 (A's open triads 21-26) are 0 in every count and delta; the coauth-DBLP h_motif full
   count sums to 8,354,454,505; every type1 full count equals the sum of C(|e|, 3) computed from
   the input's rows; the derived counts after every batch are non-negative and equal to the
   recorded ones; the four executables' batch lines agree; every batch file's record has the
   case's deletions and insertions.

### 3. A against its own CPU reference (PLAN 6.3 step 2)

A's unpatched executables (the persistent copy) run with `--check --reference`,
`OMP_NUM_THREADS=16`, `nice -n 10`, on GPU 1 under the shared lock, on make-test, syn100k s1-s3
at 5000 and 50000, syn1m s1 at 50000 (all four executables), coauth type1 b50k_s1 and coauth main
`--batches 1 --batch-size 20` (time-boxed to 2 h). `--check` compares every CBST row with A's host
model and "baseline + deltas" with a full GPU recount; `--reference` compares the counts with the
CPU reference after the build and after every batch. Any disagreement stops the milestone (it is
a finding against the original, never worked around). The runs' count lines must also equal the
goldens'. `parity/triad_goldens.py crosscheck` runs and records them
(`parity/results/M9a.md` section 1).

## Outcome (M9a step 1)

The set was exported twice from two fresh patched copies with byte-identical `MANIFEST.sha256`
(`2d8e49b5...`; 75 groups, 300 cases, 686 files, 367.7 MB), `verify` passes, all 300 `--check`
runs agree, the twins of type1-3 record main's batch files, `generators::legacy` reproduces the
12 synthetic inputs, and A agrees with its CPU reference on all 90 cross-check runs, whose counts
equal the goldens' (`parity/results/M9a.md` section 1).

## Consequences

- The port (M9a steps 2-5) is measured and replayed against a fixed reference: the region map
  is committed before any file under `cpp/src/algorithms/triad_count/` or
  `cpp/include/dyng/triad_count.hpp` exists (the git log shows the order).
- No committed fixtures in this step: PLAN 8.3's budget for tracked test data (< 1 MB) is nearly
  used (cpp/tests/data holds 998,934 bytes); the author raised it to 1.25 MB on 2026-10-15 (D5),
  and M9a's later steps commit at most about 16 KB of make-test counts under that amendment.
- The golden set is outside the repository ($DYNG_SCRATCH/goldens/triad_count/escher_gpu); its
  manifest and per-case digests are in `parity/goldens.toml`.

## Alternatives considered

- **Counts from the patched copy's twins.** Rejected: PLAN 8.3 takes goldens from the patched
  copy only for what the export patch adds; the counts are A's own output, and the twins' extra
  synchronizations and hashing are not part of A. The originals the patch keeps aside are A's
  unpatched build.
- **One `.dgb` per case with all three batches.** Rejected for the record: the acceptance asks for
  the SHA-256 of each batch's file. Splitting keeps every batch a valid `.dgb` (the reader checks
  each record's CRC and the header's count), and a replay reads them in order.
- **Storing coauth-DBLP's loaded rows.** Unnecessary: they are byte-identical to the dataset
  file, which is pinned by its SHA-256 in `datasets/registry.toml`.
