# M9a parity record: triad_count against ESCHER-GPU

Date: 2026-10-15 onwards. M9a ports ESCHER-GPU@abcf9b2's (A's) dynamic triad counters
(`build/main`: the 30-bin h-motif counts; `build/type1`, `type2`, `type3`: StatHyper's triad
types) as `triad_count`, exact on the sequential, OpenMP and CUDA backends, with the gates of
PLAN 8.6 (PLAN 6.3, 6.4.6, 8.3, 11.3; the author's decisions of 2026-10-15 in GOVERNANCE.md).
Each step adds its section.

| Item | Value |
|---|---|
| Branch | `m9a-triad-count`, created by the orchestrator from `origin/main` at `9cde147` (the merge of PR #15, M8b; M8a is in `main` too; `main` is 0.3.0.dev0). `origin/main` already contained PR #15, so the branch did not start from `m8b-hypergraph` at `65a62c9`; `9cde147`'s tree is that branch's plus `main`. |
| Reference | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A) |
| Inputs | coauth-DBLP `$DYNG_SCRATCH/datasets/hyper/coauth.hg` (SHA-256 `65fc83b2...7cb6`, `datasets/registry.toml`); A's synthetic hypergraphs (`n m r1 r2`, `--seed`); make-test's gawk files from M8a's set `cbst_trace_escher_gpu_test` |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86; A built by its own Makefile (`make all-types build/cbst_test ARCH=sm_86`) |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1; GPU 1 for every untimed run (no timed reading in step 1) |

## 1. Step regions-goldens-crosscheck: the reference side

### 1.1 The timed-region map (criterion 1; ADR 0048 section 1)

`parity/timed_regions/triad_count.toml` was committed in `2f54bce`, the branch's first commit,
before anything exists under `cpp/src/algorithms/triad_count/` or
`cpp/include/dyng/triad_count.hpp` (`git log --diff-filter=A --format=%h -- cpp/src/algorithms/triad_count cpp/include/dyng/triad_count.hpp`
is empty at the end of step 1). Its gated rows:

| Row | A's timer | Port | Gate |
|---|---|---|---|
| `update[b1]`, `update[b2]`, `update[b3]` | `timing:update` (subtract + maintain + add) per batch | `triad_count.update` (normalize + count_minus + commit + identify_affected + count_plus + finalize), driver `update_ms` | compute (1.05) |
| `full_count` | `timing:init.full_count` | `triad_count.compute` (the whole `compute()` call; `triad_count.count` inside it is a sub-stage), driver `full_count_ms` | compute (1.05) |
| `end_to_end` | `timing:total` | the driver's total from the start of `main()` | end_to_end (1.10) |

Reported only: `update.subtract` = normalize + count_minus, `update.maintain` = commit,
`update.add` = identify_affected + count_plus + finalize (per batch), `build`, `upload`, `load`,
`batch.generate` (per batch), `process`; the 3-batch sum of `update`. The scope rules (line graph
and row views built at construction; a host batch when `update` starts; generation outside;
no `--check`; `--sync-stages`) are in the file's notes and the ADR. `perf_ab.load_regions` and
`bench_suite.region_map` read the per-batch rows; `ci/tests/test_triad_regions.py` (9 tests: the
map loads, an unknown timer, an unknown key, malformed rows, a missing gated region are refused).

### 1.2 The golden set `triad_count_escher_gpu` (criterion 2; ADR 0048 section 2)

`flock -s $DYNG_SCRATCH/perf.lock nice -n 10 parity/triad_goldens.py export --twice --jobs 3`
built two FRESH `git archive abcf9b2` patched copies (`parity/build_reference.sh --variant patched
--fresh --scratch $DYNG_SCRATCH/ref/m9a-export-<1|2>`), exported from both at once, compared the
manifests and removed the copies:

| Item | Value |
|---|---|
| Groups / cases | 75 groups (15 make-test, 60 grid) x 4 executables = **300 cases** (60 make-test, 240 grid) |
| Files / size | 686 files, 367,663,624 bytes (inputs 157.7 MB, cases 209.9 MB; zstd -12) |
| `MANIFEST.sha256` | `2d8e49b515a65b1141eeb1dec3a401cc7fad73cc399157a97121752ad08bb9d1`, **byte-identical in both exports** |
| Machine time | 62 min wall for both exports (06:28-07:30, GPU 1, 6 processes at a time); 10,569 s and 10,472 s of group time per export (coauth 6,421 s, syn2m 2,441 s, syn1m 1,251 s, syn100k 280 s, make-test 176 s); building the two copies took about 1 min of it |
| Copies removed | `$DYNG_SCRATCH/ref/m9a-export-1`, `-2` (gone) |
| Goldens budget | the set adds 0.37 GB (budget 1.5 GB); `$DYNG_SCRATCH/goldens` holds 7.1 GB (budget 10 GB) |
| ESCHER-GPU's working tree | `git -C /home/sskg8/Projects/ESCHER-GPU status --porcelain` empty before and after (and HEAD unchanged, checked by the exporter and by `build_reference.sh`) |

What each export checked while it ran (any failure stops it):

- **Counts** from A's unpatched executables (the patched copy's `parity_export/orig/<exe>`, the
  Makefile-built executables the export patch keeps aside), without `--check`/`--reference`; each
  of the 300 cases ran again with `--check`: exit 0, no `FAILED`, integrity OK after the build and
  every batch, "baseline + deltas == full GPU recount" OK after every batch, **count lines
  byte-identical** (the whole stdout was identical in every case).
- **Batches**: one run of the recording twin of `build/main` per group (batches mode; coauth-DBLP
  needed no new mode, the exit record already carries `inserted_keys`), `dyng-compat-escher
  to-dgb`, one `.dgb` per batch (flags `deleted_members`, `expected_ids`); the inserted keys of
  every batch record number its insertions. On every case of every executable the batch lines and
  the deletions/insertions equal the batch files'; on `test/04-small_b20`, `syn100k_b5k_s1`,
  `syn1m_b5k_s1`, `syn2m_b5k_s1` and `coauth_b5k_s1` the twins of type1, type2 and type3 recorded
  **byte-identical batch files** to main's.
- **make-test**: its 60 argument lists equal the main/type1/type2/type3 invocations of M8a's set
  `cbst_trace_escher_gpu_test`; the gawk inputs (`small.hg`, `skewed.hg`, `medium.hg`) have the
  SHA-256 that set recorded. A's unpatched stdout of the 60 cases is kept (`<exe>.stdout`).
- **Inputs** (16, each stored once): coauth-DBLP's loaded rows are byte-identical to the dataset
  file (not stored); the 12 synthetic inputs (syn100k s1-s3 and s5, syn1m s1-s3, syn2m s1-s3,
  syn3k s1, syn8 s1) are **reproduced byte for byte by `generators::legacy::escher_hypergraph`**
  (through `dyng.generators.legacy`, written with `index_base` 1); the per-seed syn1m / syn2m `.hg`
  files are stored with their SHA-256 for the paper suites:

| Input | Hyperedges | Pins | A's vertex count | sum C(\|e\|, 3) | loaded-rows SHA-256 |
|---|---:|---:|---:|---:|---|
| coauth | 2,467,389 | 7,748,737 | 1,930,378 | 32,051,412 | `65fc83b2...7cb6` (= the file) |
| syn1m_s1 / s2 / s3 | 1,000,000 | 5,492,345 / 5,501,414 / 5,499,896 | 333,333 | 32,873,885 / 33,066,347 / 33,043,484 | `664dc706...`, `6103d6fa...`, `101ba09f...` |
| syn2m_s1 / s2 / s3 | 2,000,000 | 10,989,640 / 11,001,035 / 11,001,680 | 666,666 | 65,862,478 / 66,035,612 / 66,063,489 | `c7a7ad18...`, `05c223c0...`, `10abbf5f...` |
| syn100k_s1 / s2 / s3 | 100,000 | 548,192 / 550,756 / 549,008 | 33,333 | 3,273,878 / 3,319,546 / 3,290,743 | `parity/goldens.toml` |

`parity/triad_goldens.py verify` (16 s): OK. It checks the manifest against `goldens.toml`, every
file and the 300 case digests; **bins 20-25 are 0** in every count and delta; the coauth-DBLP
h_motif full count sums to **8,354,454,505** (in all 15 coauth groups); every **type1 full count
equals sum C(|e|, 3)** computed from the input's rows; the counts after every batch are the
recorded ones and non-negative; the executables' batch lines agree; every batch file's record
counts equal the case's; the full counts of one input and executable agree across its groups.
`parity/tests/test_triad_goldens.py` plants a defect for each of these assertions (an open-triad
bin, a changed bin, a negative count, a type1 count off the rows, a wrong batch file, other batch
lines, a failed `--check`, damaged rows, a damaged batch file, a changed or stray file) and
requires `verify` to report it.

The gate cases (seed 1, 3 x 50K), for the later steps:

| Case | full count total (main / type1 / type2 / type3) | A's `deviceBytes()` after the build, b1, b2, b3 | `numHyperedgeKeys` |
|---|---|---|---|
| coauth_b50k_s1 | 8,354,454,505 / 32,051,412 / 123,987,342 / 40,211,269 | 1,922,784,204; 1,923,300,764; 1,923,713,052; 1,924,170,844 | 2,467,389 -> 2,467,512 |
| syn1m_b50k_s1 | 248,643,562 / 32,873,885 / 90,430 / 159,950 | 703,051,032; 703,183,432; 703,130,968; 703,185,928 | 1,000,000 -> 1,000,326 |
| syn2m_b50k_s1 | 497,652,211 / 65,862,478 / 88,268 / 161,614 | 1,406,745,268; 1,406,769,300; 1,406,707,652; 1,406,784,388 | 2,000,000 -> 2,000,161 |

(coauth's `deviceBytes()` equals M8a's coauth50k trace, 1,922,784,204 after the build.)

### 1.3 A against its own CPU reference (criterion 3; PLAN 6.3 step 2)

`flock -s $DYNG_SCRATCH/perf.lock parity/triad_goldens.py crosscheck --jobs 2 --record
parity/results/M9a-crosscheck-escher_gpu.json`: A's unpatched executables of the persistent copy
(`$DYNG_SCRATCH/ref/ESCHER-GPU@abcf9b2/unpatched/build`, the performance baseline) with `--check
--reference`, `OMP_NUM_THREADS=16`, `nice -n 10`, GPU 1. A case agrees when it exits 0, prints no
`FAILED`, and reports "counts == CPU reference: OK" and the integrity check OK after the build and
every batch and "baseline + deltas == full GPU recount: OK" after every batch. Its count lines
were compared with the goldens as well.

| Cases | Runs | Agree | Counts = goldens | Run time (sum) |
|---|---:|---:|---:|---:|
| make-test (15 groups x 4) | 60 | 60 | 60 | 43.3 s |
| syn100k s1-s3 at 5000 and 50000, x 4 | 24 | 24 | 24 | 72.5 s |
| syn1m s1 at 50000, x 4 | 4 | 4 | 4 | 115.8 s (main 46.3, type1 10.6, type2 14.3, type3 44.6) |
| coauth type1 b50k_s1 | 1 | 1 | 1 | 26.6 s |
| coauth main `--batches 1 --batch-size 20 --seed 1` (2 h time box) | 1 | 1 | 1 (the full count against coauth_b50k_s1/main) | 143.5 s |
| **total** | **90** | **90** | **90** | 401.7 s |

**A agrees with its own CPU reference on every case**, and the persistent unpatched build prints
the goldens' counts on all 90 runs (the goldens came from the fresh copies' builds). The coauth
h_motif reference finished well inside its time box (143.5 s; it ran once more by hand before the
script, 144.1 s, with the same result), so the fallback syn2m s1 b50000 main was not needed.

## 2. Step oracle-harness-driver-prereads: the oracle, the replays, the driver, the pre-reads

Code: `bcf3f59` (the oracle), `062b330` (the generators' host model), `282f3c7` (the replays),
`1e6e933` (the driver and the A/B harness), `729bb79` (two template parameter names for
clang-tidy). The records below ran from `1e6e933` (oracle, replays, draws, memory) and `729bb79`
(the timed pre-reads), built with the `parity-cuda` preset; a record's `+dirty` means that only
these records and the docs were uncommitted. No engine code exists yet: nothing under
`cpp/src/algorithms/triad_count/` and no `cpp/include/dyng/triad_count.hpp`.

### 2.1 The independent oracle (criterion 4; ADR 0049)

`dyng::testing::triad_reference` (`dyng/testing/triad_reference.hpp`, tracked; its own object
library `dyng_testing_triad_reference`) transliterates A's CPU reference (`utils/reference.cpp`:
the line graph rebuilt per hyperedge, every triangle i < j < k, 64-bit counts, sequential or
OpenMP) and derives the 30-bin class map itself (the orbits of the 128 Venn ids under the six
permutations, numbered by the documented representatives 64, 65, 67, 71, 72, 73, 76, 75, 77,
79, 88, 90, 89, 91, 93, 95, 120, 121, 123, 127, 24, 26, 25, 27, 29, 31, 56, 57, 59, 63).

| Check | Result |
|---|---|
| derived map == A's `ESCHER_MOTIF_ID_TO_INDEX` (test-only fixture) on all 128 ids | equal |
| classes are the permutation orbits; 32 ids in no class, 96 in 30 classes | yes |
| one triple per class through the oracle: a single 1 in bin k - 1 for the 24 reachable classes, nothing for the 6 open triads | yes |
| the same 30 triples (`.hg`, one vertex per Venn region) through A's **unpatched** `build/main --input F --batches 0` (`parity/triad_replay.py singles`, `M9a-oracle-singles-escher_gpu.json`) | **30/30 agree with the oracle; A reports the same single bin for all 24 reachable bins** (0-19, 26-29) |
| hand cases (types 1-3, duplicate hyperedges, dead rows, repeated members, a hyperedge of 1,024 members, vertex 2^31 - 2), a brute force over all triples of 40 random hypergraphs on 1, 3 and all threads, `removed()` / `added()` against the brute force, malformed input | `dyng_triad_reference_tests`: 8 tests pass |
| isolation: no project include but its header, no undefined `dyng::` symbol in its object (`triad_reference_isolation`) | passes |
| the planted mutation `DYNG_MUTATION_TRIAD_REFERENCE_CLASS_MAP` (classes 2 and 3 swapped) | CTest `testing.mutation.triad_reference_class_map` passes: the mutant fails 4 of its 8 tests (the class map, the single triples, the brute force, removed / added); listed by `parity/mutate.py list` (suite `triad_reference`) |

### 2.2 The replays and the go/no-go (criteria 4 and 8, the harness; ADR 0050)

`parity/compare.py triad_count --counter oracle --mode both` (the tool `dyng_triad_replay`): per
group the container from A's loaded rows under {erase_first, escher_gpu(capacity),
sorted_merge}; by key (A's ids, `inserted_ids` == the exported expected ids) and by content
(every exported deleted row resolved to a live hyperedge with that member set, A's rows inserted
in A's order); `check_integrity()` after every batch; the oracle on `to_hyperedge_list()` after
the build and every batch, all four patterns; the full count, every net delta and counts-after,
the deletions / insertions and every batch file's SHA-256 compared with the goldens; both modes
equal. Record: `M9a-gonogo-escher_gpu.json` (per run and group; the run logs are in
`$DYNG_SCRATCH/runs/m9a-step2/final`).

| Backend | make-test + syn100k grid | syn1m seed-1 grid | coauth_b50k_s1/type1 | Run time (sum of groups) |
|---|---:|---:|---:|---|
| cuda (GPU 1) | 120/120 | 20/20 | 1/1 | 240 s + 435 s + 51 s |
| openmp, 8 threads | 120/120 | 20/20 | 1/1 | 303 s + 404 s + 30 s |
| sequential | 120/120 | 20/20 (`--no-coverage`) | 1/1 (`--no-coverage`) | 367 s + 523 s + 77 s |

**GO: the oracle reproduces A exactly on all 423 case replays, by key and by content, on every
backend** (the full count, every delta and every count after; the expected ids by key in every
batch; no unresolved deletion by content; integrity after every batch). By content, deleted rows
whose member set several live hyperedges share are common on the synthetic sets (duplicate
small hyperedges), and resolving them to another id than A's changes no count, as PLAN 6.4.6
expects. The oracle evaluated a list again only when the two modes' states differed (the cache
of ADR 0050 section 2).

The harness's planted defects (`parity/tests/test_triad_replay.py`, on `test/04-small_b20` main
and type1 with the goldens): one changed bin of the full count, one perturbed deleted row (by
content), batch 2 dropped, batches 1 and 2 swapped, one wrong expected id (by key): **each fails
the replay** with its own finding, the unmodified replay passes, and `--counter algorithm` exits
2 (not available before step 3). The C++ test `dyng_triad_replay_tests` checks the tool's reports
on a case it writes itself (hosted CI).

### 2.3 The port driver and the generators' host model (criterion 12, the driver; ADR 0051)

`dyng-compat-escher run` takes A's argv; in this step it runs with `--no-count` only. Its batches
come from `generators::legacy::escher_batch_generator::next(escher_host_model)`, so the gated
process never builds a public view (`h2v()` is not called). `parity/triad_replay.py draws`
(`M9a-draws-escher_gpu.json`, cuda): **22/22 groups draw the goldens' batch files byte for byte**
(SHA-256 of every batch file, split one record per file as the exporter does): the 15 make-test
groups, syn1m_b50k_s1-s3, syn2m_b50k_s1 and coauth_b50k_s1-s3; the batch lines, and on make-test
the `Hypergraph:` line, are A's. The existing `next(res, hypergraph)` draws unchanged (the M8b
tests `dyng_generators_tests` and `dyng_hypergraph_ids_tests` with `--legacy-generator` pass),
and the new test draws both overloads in lockstep over 6 batches, both reuse orders, both insert
modes and three sizes.

### 2.4 The go/no-go pre-reads (not gates)

**(a) End to end** (`parity/perf_ab.py triad_count run --no-count --rounds 5 --lock-clocks base
--gpu 0`, `M9a-preread-e2e-escher_gpu.json`): exclusive perf lock, GPU 0 at the base lock (SM 1170
/ memory 7601 MHz, every busy sample of every round at the lock), `CUDA_MODULE_LOADING=EAGER`, no
`OMP_*`, one `--timing` file name, A/B alternation, no round rejected (foreign CPU at most 1.2
cores; load average 2-7, `procs_blocked` 0 in every round); medians of 5 rounds (ms):

| Case | A total | A init.full_count | A sum(subtract + add) | A_noncount | port (`--no-count`) total | target (A_noncount + 0.04 x A total) | |
|---|---:|---:|---:|---:|---:|---:|---|
| coauth_b50k_s1 | 9,417.1 | 5,292.8 | 1,402.6 | 2,720.7 | 2,204.9 | 3,097.4 | **met** (892 ms margin) |
| syn1m_b50k_s1 | 2,482.7 | 635.3 | 298.4 | 1,548.9 | 1,539.1 | 1,648.2 | **met** (109 ms margin) |

The breakdown (medians, ms; A's timers against the port's):

| Case | A init.load / port load | A v2h + h2h + flatten (host) / port hypergraph.build | A construct + row_views / port hypergraph.upload | A update.maintain b1-b3 / port apply b1-b3 | A batch.generate b1-b3 / port | the rest (A untimed / port untimed) |
|---|---|---|---|---|---|---|
| coauth_b50k_s1 | 356.6 / 357.0 | 1,394.4 / 756.5 | 176.8 / 514.1 | 70.6, 68.1, 68.9 / 55.5, 60.1, 60.9 | 175.9, 16.2, 16.1 / 15.1, 14.2, 15.7 | 333.2 / 369.9 |
| syn1m_b50k_s1 | 359.5 / 428.2 | 674.6 / 422.8 | 67.2 / 175.9 | 65.3, 45.0, 55.9 / 47.9, 49.8, 51.3 | 16.3, 15.2, 15.2 / 16.6, 15.7, 16.2 | 234.6 / 312.9 |

(The rest: A's context creation and `fitsInMemory`, its host model, the outputs; the port's
`warm_up()` (context and the EAGER load of libdyng's modules), the host model built from the list,
the outputs.) Host synchronizations per batch (one Nsight Systems API trace per side, blocking
calls in the batch's range): A's `update` (with its counting) 96, 96, 96 (coauth) and 96, 90, 96
(syn1m); the port's `hypergraph.apply` 19, 19, 19 and 19, 16, 19 (`cudaStreamSynchronize` only).
Nothing needed fixing in the driver, the reader or the generators.

**(b) Memory** (`parity/perf_ab.py triad_count memory --no-count --gpu 1`,
`M9a-preread-memory-escher_gpu.json`; both whole processes under `nsys profile -t cuda,nvtx
--cuda-memory-usage=true`, no foreign process on GPU 1 before and after; A without `--check`):

| Case | A peak (MiB) and range | port peak (MiB) and range | port / A | target <= 1.03 | triad budget for step 3 (1.05 A - port - 0.01 A) | footprint at the live peak A / port (MiB) | held between batches A / port (MiB) |
|---|---|---|---:|---|---:|---|---|
| coauth_b50k_s1 | 3,018.9, `update > update.maintain > maintain.h2h_merge` | 3,019.1, `hypergraph.apply > hypergraph.apply.line_graph` | 1.0001 | met | **120.6 MiB** | ratio 1.017 | 1,866 / 3,040-3,104 (1.63-1.66) |
| syn1m_b50k_s1 | 1,131.4, the same | 1,131.6, the same | 1.0002 | met | **45.1 MiB** | ratio 1.018 | 703 / 1,152-1,184 (1.64-1.69) |
| syn2m_b50k_s1 | 2,211.7, the same | 2,211.8, the same | 1.0000 | met | **88.4 MiB** | ratio 0.998 | 1,374 / 2,240 (1.63) |

The store accounting (`store_stats().device_bytes` against A's `deviceBytes()` after the build and
every batch) is equal at every point of the three cases, and A's two print points agree with the
golden recording. The footprint between batches is the stream-ordered pool's reserve (M8b: the
pool keeps what it reached); it is reported and flagged above 1.05, not gated (D1).

**(c) The boost probe** (`--lock-clocks boost --rounds 3 --probe`,
`M9a-preread-boost-probe-escher_gpu.json`): the boost lock (SM 1695 / memory 7601 MHz) **does not
hold** on syn1m_b50k_s1 or syn2m_b50k_s1: in every round A's process ran busy samples at 1,470-1,665
MHz, and the port's in 3 of the 6 rounds. **The syn gates therefore run at the base lock**, as
coauth-DBLP's (ADR 0021; criterion 12's rule "at boost if a recorded 3-round probe holds,
otherwise base").

### 2.5 For step 3

- **GO for step 3**: the oracle reproduces A everywhere (2.2), the driver draws A's batches
  (2.3), and both pre-read targets are met (2.4).
- **End-to-end budget**: the counting of the port must cost no more than A's `init.full_count +
  sum(update.subtract + update.add)` plus the margins of 2.4(a): 892 ms on coauth_b50k_s1, **109 ms
  on syn1m_b50k_s1** (the tight one: the port's synthetic load is 69 ms slower than A's
  generator, its untimed rest 78 ms longer, while its build is faster).
- **Memory budget**: the triad bytes live during `hypergraph.apply` (the gated peak lies in the
  H2H merge on both sides) must stay below 120.6 MiB (coauth_b50k_s1), **45.1 MiB
  (syn1m_b50k_s1)** and 88.4 MiB (syn2m_b50k_s1).
- **Host syncs**: A's update makes 90-96 blocking calls per batch; the port's apply 16-19 so far.
