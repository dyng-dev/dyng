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

## 3. Step cuda-port-api-early-gate: the CUDA port, the API and the early gate reading

Code: `be82dec` (the scaffold), `645d6dc` (c1, the transplant with the API, the problem, the host
recount, the replay counter and the driver's counting), `58b0895`, `64d3b7c` (c2), `29a02ad` (c3),
`679cec9` (c4), `84f894b` (c5), `c0793d6` (c6), `860fb1b` (the host recount on the engine's
operations), `85efa0b` (the tests and the kit), `7c78e8b` (the driver's stdout, `triad_replay.py
stdout`, the gpu_local step), `7c75a45`, `40efdba`, `a7a53cc`, `091e667` (the manifest's `python` flag), `a39b0bf` (its message under the line limit: pre-commit, which ran before `091e667`). ADRs 0052 (the API) and 0053 (the engine port).
The parity records ran from `85efa0b` (the replay tool of the reduced build) and `7c78e8b` (the
driver); the gate readings from `40efdba` (`+dirty`: only records and docs uncommitted).

### 3.1 The transplant and the commit table (criterion 7; ADR 0053)

`python3 parity/escher_port/transplant.py --map parity/escher_port/rename_map_triad.toml --check
--rev 645d6dc`: **10 files, identical**. After every commit of the port, `parity/compare.py
triad_count --backend cuda --mode by_key --counter algorithm --cases
make-test,syn1m_b50k_s1,coauth_b50k_s1` on a reduced Release build (`cmake --preset parity-cuda
-DDYNG_ALGORITHMS=triad_count`, GPU 1, shared lock; `M9a-transplant-replays-escher_gpu.json`):

| Commit | Class | Cases | make-test / syn1m_b50k_s1 / coauth_b50k_s1 (s, group time) |
|---|---|---:|---|
| `645d6dc` c1 | transplant (A's counters run as A's driver runs them) | 68/68 | 9 / 18.5 / 169.8 (replayed later from a worktree, beside other GPU work) |
| `64d3b7c` c2 | errors (`DYNG_CUDA_TRY`; the unclassified triangle raises the error word) | 68/68 | 7 / 17.0 / 69.3 |
| `29a02ad` c3 | memory (the pooled workspace, CUB's scan, the frontier by a kernel, the phases' bins apart) | 68/68 | 8 / 16.9 / 68.8 |
| `679cec9` c4 | streams (the resources' stream; syncs at A's read points only) | 68/68 | 8 / 16.8 / 69.1 |
| `84f894b` c5 | registration and counters (`DYNG_REGISTER_KERNEL`, launch counters) | 68/68 | 7 / 16.9 / 68.6 |
| `c0793d6` c6 | the Op form (host-and-device operations, rows and bins handles, `ownership::min_member`) | 68/68 | 7 / 16.9 / 68.8 |

(Each group's time includes the recount after every batch that the replay makes for "baseline +
deltas == recount".)

### 3.2 SASS and resources (criterion 7; `M9a-sass-escher_gpu.json`, `parity/triad_sass.py`)

The five counting kernels of A's unpatched objects and of the parity-cuda `engine/counters.cu.o`
(`cuobjdump -res-usage`, `cuobjdump -sass`; a digest of the normalized instruction stream):

| Kernel | REG A / c5 / c6 | stack, local | shared (B) | instructions A / c5 / c6 | largest loops A / c5 = c6 | machine code |
|---|---|---|---:|---|---|---|
| count_wedges | 14 / 14 / 14 | 0, 0 | 0 | 48 / 48 / 48 | 1 / 1 | A = c5 = c6 |
| triangle_kernel<motif_op> | 39 / 39 / 39 | 0, 0 | 240 | 536 / 552 / 552 | 461, 438, 435 / 479, 456, 453 | c5 = c6; +16 against A: the error word of a pattern of no class (c2) |
| triangle_kernel<type3_op> | 39 / 39 / 39 | 0, 0 | 8 | 512 / 512 / 512 | 419, 409, 406 / the same | A = c5 = c6 |
| pair_kernel<type2_op> | 38 / 38 / 38 | 0, 0 | 8 | 456 / 456 / 456 | 364, 360, 192 / the same | A = c5 = c6 |
| key_kernel<choose3_op> | 16 / 16 / 16 | 0, 0 | 8 | 152 / 152 / 152 | 41, 40, 17 / the same | A = c5 = c6 |

**c6 go/no-go: GO** (`triad_sass.py compare`: the five kernels identical to c5's). Theoretical
occupancy is the same everywhere (256-thread blocks; at 38-39 registers 6 blocks per SM, 1,536
threads: 100 % on sm_86). A first c6 whose bins handle also tested the upper bound of the bin had
one IADD3 more per triangle in MotifOp's loop (479 -> 481); the handle tests the sign only (ADR
0053 section 1). Compile time: `engine/counters.cu` and `cuda.cu` take 30 s and 31 s of nvcc each
for the release architectures (75-real ... 120), negligible against the hosted cuda-build's 120
minutes.

### 3.3 Exact parity on cuda (criterion 8; `M9a-parity-cuda-escher_gpu.json`)

`parity/compare.py triad_count --backend cuda --mode both --counter algorithm --jobs 3` (all 300
cases of the set: the 240 grid cases and make-test's 60; GPU 1): **300/300 cases exact**: the
full count, every net delta and count after equal to the golden's in both modes, the expected ids
by key in every batch, every deleted row resolved by content, `check_integrity()` after every
batch, both modes equal, and per batch the result equal to a fresh `compute()` and its stats
consistent (delta, frontier sizes, removed / added). 2,373 s wall (7,107 s of group time;
coauth's groups 282-452 s each).

| Backend | Cases | Modes | Counter | Result |
|---|---|---|---|---|
| cuda | 300 (the grid and make-test) | by key and by content | algorithm (the wedge engine) | **300/300 exact** |
| sequential (dev build) | make-test (60) | both | algorithm (the host recount, step 3) | 60/60 |
| openmp, 8 threads (dev build) | make-test (60) | both | algorithm (the host recount) | 60/60 |

The host backends' grids (the syn100k grid, the openmp syn1m s1 grid and coauth types 1-2) are
step 4's, with the incremental host engines.

- **The driver's stdout** (`parity/triad_replay.py stdout`, `M9a-stdout-escher_gpu.json`): `dyng-compat-escher
  run --pattern P` with A's arguments prints **A's unpatched stdout byte for byte on all 60
  make-test cases**, and the count and batch lines of coauth_b50k_s1, syn1m_b50k_s1 and
  syn2m_b50k_s1 for all four patterns: 72/72.
- **10 repeated coauth_b50k_s1 runs** (the driver, each pattern, `M9a-repeat-escher_gpu.json`):
  one stdout per pattern over its 10 runs.
- The harness's planted defects fail the algorithm counter too (a changed bin, swapped batches, a
  wrong expected id; `parity/tests/test_triad_replay.py`).

### 3.4 Kernel paths (criterion 9)

The result keeps launch counters per (mode, shape) (`triad_launch_counters`; the replay reports
them). On every case each pattern ran its kernels in both modes: full and frontier x triangles
(h_motif, type 3), pairs (type 2), keys (type 1). The store's kernel paths of make-test's skewed and
medium cases (the replay's container, fill / reuse / unfill as thread, warp, block launches):
**skewed 2, 0, 2 each; medium 3, 3, 0 each: the thread, warp and block paths all ran**.
`dyng_triad_count_cuda_tests` (GPU 1): `KernelPathsAndEarlyOuts` (both early outs: an insert-only
batch has no subtract sources, the deletion of an isolated hyperedge has sources without wedges;
every (mode, shape) launched), insert-only, delete-only, delete-all, a batch larger than the
hypergraph, the id bound going down; `EveryVennPatternGetsItsClassOrRaisesTheErrorWord`: the 128
Venn ids through MotifOp and the bins handle on the device, 96 classified as the oracle's derived
map says, the 32 others raise the error word and write neither `bins[-1]` nor `bins[30]`.

### 3.5 Tests and the kit (criteria 5, 6, 10)

| Executable | Build | Result |
|---|---|---|
| `dyng_triad_count_tests` (host backends) | dev | 20/20 |
| `dyng_triad_count_conformance_tests` (h_motif and types 1-3; both instantiations; every preset, mix and shape) | dev (budgets) | 88 passed, 16 skipped (C4: one engine; C12: cuda only); C10 with another pattern and the fake |
| `dyng_triad_count_cuda_tests` | dev-cuda | 14/14 |
| `dyng_triad_count_conformance_cuda_tests` | dev-cuda (budgets) | 96 passed, 8 skipped (C4: one engine), C8 strict, C10 with another pattern and the fake, C12 on cuda |

C8 on cuda: no allocation after the commit once reserved, 3 host synchronizations per update (1 for
type 1), the problem's budget equal to the traits'. An update measured directly
(`AnUpdateSynchronizesAtTheEnginesReadPointsOnly`): 3 (type 1: 1). In the gated process (Nsight
Systems API trace, 3.6): the port's update makes 22, 19, 22 blocking calls per batch
(`cudaStreamSynchronize`: the container's apply 19 / 16 / 19, the counting 3), A's 96, 90-96.

### 3.6 The early gate reading (criteria 12 and 13, not the gates)

**Timing** (`parity/perf_ab.py triad_count run --cases coauth_b50k_s1,syn1m_b50k_s1 --rounds 5
--lock-clocks base --gpu 0`, `M9a-early-run-escher_gpu.json`): exclusive perf lock, GPU 0 at the
base lock (SM 1170 / memory 7601 MHz), EAGER, one heap layout, A/B alternation, no round rejected;
medians of 5 rounds (ms):

| Row | coauth_b50k_s1 A / port | ratio | syn1m_b50k_s1 A / port | ratio | target |
|---|---|---:|---|---:|---|
| update[b1] | 510.7 / 484.8 | 0.949 | 164.6 / 143.6 | 0.872 | <= 1.03 |
| update[b2] | 548.7 / 533.4 | 0.972 | 144.3 / 145.9 | 1.011 | <= 1.03 |
| update[b3] | 549.5 / 538.3 | 0.980 | 155.0 / 147.2 | 0.950 | <= 1.03 |
| full_count | 5,293.8 / 5,286.6 | 0.999 | 635.2 / 636.9 | 1.003 | <= 1.03 |
| end_to_end | 9,393.0 / 8,850.2 | 0.942 | 2,495.6 / 2,455.9 | 0.984 | <= 1.07 |

Reported sub-rows (ratios b1-b3): coauth `update.subtract` 1.002, 1.010, 0.996; `update.maintain`
0.767, 0.879, 0.866; `update.add` 0.958, 0.964, 0.990; syn1m `update.subtract` 1.015, 1.027, 1.026;
`update.maintain` 0.724, 1.069, 0.896; `update.add` 0.933, 0.941, 0.937. **Every target met**; no
profiling or container fix was needed (nothing changed in `cpp/src/hypergraph`).

**Memory** (`parity/perf_ab.py triad_count memory --cases coauth_b50k_s1,syn1m_b50k_s1,syn2m_b50k_s1
--gpu 1`, `M9a-early-memory-escher_gpu.json`; both whole processes under `nsys profile -t cuda,nvtx
--cuda-memory-usage=true`, no `--check`, no foreign process on GPU 1):

| Case | A peak (MiB), range | port peak (MiB), range | port / A | target <= 1.04 | triad bytes during `hypergraph.apply` (budget of step 2) | footprint at the peak A / port | held between batches A / port (MiB) |
|---|---|---|---:|---|---|---|---|
| coauth_b50k_s1 | 3,018.9, `maintain.h2h_merge` | 3,021.9, `triad_count.update > triad_count.commit > hypergraph.apply > hypergraph.apply.line_graph` | 1.0010 | met | 3.0 MiB (120.6) | 3,050.9 / 3,136.0 (1.028) | 1,866 / 3,040-3,136 |
| syn1m_b50k_s1 | 1,131.4, the same | 1,133.0, the same | 1.0014 | met | 1.6 MiB (45.1) | 1,163.4 / 1,184.0 (1.018) | 703 / 1,152-1,184 |
| syn2m_b50k_s1 | 2,211.7, the same | 2,214.2, the same | 1.0011 | met | 2.5 MiB (88.4) | 2,243.7 / 2,240.0 (0.998) | 1,374 / 2,240 |

Peaks per NVTX range of the port (coauth_b50k_s1, MiB): `triad_count.compute` 1,852.6 (A's
`init.full_count` 1,852.6), `triad_count.count_minus` 1,837.6 (A's `update.subtract` 1,837.2),
`triad_count.enact_fused` 1,852.6 within the batches' `triad_count.finalize` 1,833.8 - 1,838.1
(A's `update.add` 1,837.7), `hypergraph.apply.line_graph` 3,021.9. The store accounting equals A's
`deviceBytes()` after the build and after every batch (ratio 1.0 at every point, A's two print
points agree). The footprint between batches is the pool's reserve (1.63-1.69 x A's held bytes,
as in step 2): reported, flagged above 1.05, not gated (D1).

### 3.7 The checks of the step

| Check | Tree | Result |
|---|---|---|
| `transplant.py --check --rev 645d6dc` | `645d6dc` | 10 files identical |
| reduced replays after every commit (3.1) | c1-c6 | 68/68 each |
| builds: `parity-cuda` (all algorithms; `85efa0b`), `sanitize-cuda` (`40efdba`), `dev-cuda` (in gpu_local, `a7a53cc`) | as named | built, warnings as errors |
| `ci/check.sh` (format, cpu-only and dev with `ctest -L cpu`, consumer, tidy, reuse, provenance, regen, harness, docs with the API baseline, api, scaffold, precommit) | `a7a53cc` | all passed except `python` (`dyng.triad_count` absent: fixed in `091e667`) |
| `DYNG_CHECK_ONLY="python regen harness" ci/check.sh` | `091e667` | passed (python: 637 passed, 72 skipped) |
| `DYNG_CHECK_ONLY="scaffold api" ci/check.sh` (the templates changed in `091e667`) | `091e667` | passed (scaffold_check: 80 + 54 + 41 tests; api-snapshot 1,275 declarations) |
| `DYNG_CHECK_ONLY="reuse provenance docs regen precommit" ci/check.sh` | `091e667` + the records | all passed except ruff's line length in `scripts/regen.py` (fixed in `a39b0bf`; `pre-commit run --all-files` then passed) |
| Clang 18 syntax (`clang_syntax.py`, the fortified tree's database) | `091e667` | 386 files, 0 failures |
| cpu-only with `-D_FORTIFY_SOURCE=3` (GCC 12, `-Werror`), `ctest -L cpu -R triad` | `091e667` | built; 107/107 |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | `a7a53cc` (the summary names `091e667`, which changed only Python, regen and the manifests) | all 20 steps passed in about 3.5 h (RTX A5000, driver 590.48.01, CUDA 13.1): `ctest -L gpu` 797/797, `ctest -L cpu` 1,243/1,243, the sssp corpus ALL EQUAL (495 x 2), the cycle_count corpus ALL EQUAL (48), memcheck on every gpu executable, the cbst and hypergraph steps; the `triad` step: synccheck and initcheck on `dyng_triad_count_cuda_tests`, memcheck, synccheck and initcheck on one coauth_b50k_s1 batch, racecheck on the kit's C1, C11 and C12 (0 hazards) and on make-test in both modes (60/60 cases) |
| frozen headers: `git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,graph}` | head | empty |

### 3.8 For steps 4 and 5

- **GO for step 4**: exact on cuda everywhere, the early readings inside every margin target.
- The host backends recount in `update()` (`fallback_used`); step 4 replaces `host_count.hpp`'s full
  mode with the incremental host engine (frontier mode over the same operations, the predicates of
  `engine/ownership.hpp`), then `triads_removed` / `triads_added` join the kit's deterministic
  counters (`triad_count_traits.hpp`).
- Step 5: `cpp/src/hypergraph` did not change, so no ADR 0024 bar is needed; the gpu_local `triad`
  step exists; the mutations act on `ownership.hpp`'s predicates, the bins' sign
  (`cuda.cu` / `triad_count.cpp` `apply_delta`), the frontier of the inserted ids
  (`triad_cuda_identify_affected`) and type 1's key bound (`count_full`).

## 4. Step host-engine-kit-mutations: the incremental host engines, the kit, the mutations

The sequential and OpenMP backends run the incremental host engine (`host_engine.hpp`; ADR 0054,
the author's decision D3), `operators::count_delta` enters `cpp/src/operators` with triangle_delta
and triad_count as its users (ADR 0055), ESCHER-GPU's make test runs in hosted CI under the
1.25 MB data budget (D5, ADR 0013 amended), and the triad mutations of PLAN 8.4 are recorded on
both engines. Commits: `164bfa5` (count_delta), `779dcfd` (the host engine and its tests),
`93ef4a1` (the mutations), `f20f5b3` (make test in hosted CI, D5), `c8b6e68` (mutate.py finds an
algorithm's test executables), and the records commit.

### 4.1 CPU parity (criterion 8, host; `M9a-parity-{sequential,openmp8,openmp1}-escher_gpu.json`)

`parity/compare.py triad_count --backend B --counter algorithm --mode both` against the golden set
`triad_count_escher_gpu` (manifest `2d8e49b5...`), from the parity-cuda build at `c8b6e68`: by key
(the exported ids deleted, `inserted_ids == expected_ids` per batch) and by content (each exported
deleted row resolved to a live hyperedge with those members), `check_integrity()` and a fresh
`compute()` ("baseline + deltas == recount") after every batch, full counts, net deltas and counts
after compared with A:

| Backend | Cases (the coverage table, D6 pending) | Result | Wall (group time) |
|---|---|---|---|
| sequential | make-test (60) + the syn100k grid (60) | **120/120 exact**, both modes | 914 s (4,021 s; 6 jobs) |
| openmp, 8 threads | make-test + the syn100k grid + the syn1m s1 grid (20) + coauth_b50k_s1 types 1 and 2 | **142/142 exact**, both modes | 612 s (1,754 s; 3 jobs) |
| openmp, 1 thread | the same | **142/142 exact**, both modes | 1,901 s (12,575 s; 10 jobs) |

The full records of 1 and 8 threads are identical case by case (counts, batches, launch counters;
only the times differ). An earlier run of all three from the uncommitted tree that became
`779dcfd` (the same code before clang-format) gave the same verdicts (120/120 sequential, 142/142
openmp 8); the records are the reruns at `c8b6e68`.

### 4.2 The host engine against the oracle and the kit (criteria 6 and 10)

| Executable | Build | Result |
|---|---|---|
| `dyng_triad_count_tests` (sequential, openmp) | dev | 33 tests: all pass (the thread-count test is skipped on sequential) |
| `dyng_triad_count_make_test_tests` (hosted CI; the synthetic make-test groups exact against A's committed counts, the file-input shapes against the oracle) | dev / cpu-only | 4/4 (68 s in Debug on the loaded machine) |
| `dyng_triad_count_make_test_large_tests` (labels parity, slow; the triangle patterns on syn100k, skewed, medium) | parity-cuda (Release) | 4/4 (293 s) |
| `dyng_triad_count_conformance_tests` (four patterns, both instantiations, every preset, mix and shape; openmp at 8 threads) | dev (budgets) | 104: 88 passed, 16 skipped (C4: one engine per backend; C12: cuda only); 8.2 min |
| `dyng_triad_count_cuda_tests`, `dyng_triad_count_conformance_cuda_tests` | dev-cuda | 19 passed, 1 skipped (the OpenMP thread test); kit 104: 96 passed, 8 skipped (C4), C8 strict, C12 on cuda |
| `dyng_operators_tests`, `dyng_operators_cuda_tests` | dev, dev-cuda | 6/6, 2/2 |
| compile-failure CTests `operators.count_delta.static_assert.*` | every build | 5/5 (the control compiles; no rule, a value that is not a rule, the triad phase without a rule fail with the message) |

What the host tests show (criterion 6 and 10): on every backend the counts equal the oracle and a
fresh `compute()` after every batch; `triads_removed` / `triads_added` equal the oracle's
`removed(G_t, deleted)` / `added(G_{t+1}, inserted)` and `delta` the per-bin difference (so the
frontier sizes are the batch's and the phases are exactly-once); `fallback_used` is false; a bin
that would go negative raises `internal_error` and poisons the result; every closed class (bins
0-19, 26-29) moves by one through an update and the open bins 20-25 stay 0; reused ids (an
inserted id equal to a deleted one under erase_first, none under insert_first), duplicates,
single-member hyperedges and hyperedges of 1,030-1,100 members; the bound going down (also in the
batch that deletes the top ids); type 1 on `line_graph_policy::none`; requirement errors naming
the property; random multi-batch sequences under both id_reuse orders; OpenMP identical at 1, 8
and 56 threads; the kernel paths (both early outs, every (mode, shape)) on every backend; the 128
Venn ids through the host bins handle (96 classified, 32 reported and never written). The kit's
deterministic counters now include `triads_removed` / `triads_added` (C3, C9), C4 is skipped (one
engine per backend), C8 holds in the budget build on the host (no allocation once reserved, no
host synchronization), C10 runs two triad results and the fake partner, the planted bug of the
kit (`conformance.mutation.frontier_requested_ids`) is still detected.

### 4.3 Mutations (criterion 11; `M9a-mutate-triad_count.json`)

`parity/mutate.py run --suites triad_count` (git archive of `c8b6e68`, one parity-cuda copy per
mutation, built for sm_86 only; the suite: `dyng_triad_count_tests` on the host backends and
`dyng_triad_count_cuda_tests` on GPU 1): **the control passed; 6 / 6 mutations detected**:

| Mutation | Where (shared by both engines) | Failed tests |
|---|---|---:|
| `triad_wrong_bin_sign` | `engine/phases.hpp` `phase_sign` (count(-) adds) | 32 |
| `triad_skip_frontier_filter_b` | `engine/ownership.hpp` `min_member_skips_wedge` (A's wedge.cuh:101) | 24 |
| `triad_skip_frontier_filter_k` | `engine/ownership.hpp` `min_member_skips_closing` (A's wedge.cuh:129) | 18 |
| `triad_inserted_requested_ids` | `triad_count.cpp` `triad_inserted_ids` (the requested ids) | 31 |
| `triad_type1_unbounded_keys` | `engine/phases.hpp` `full_count_keys` (the largest bound so far) | 3 |
| `cbst_unfill_drops_last_triad` | the CBST store (M8a's mutation) against the triad suite | 9 |

The same mutations are CTests of every build: `ctest -L mutation -R triad_count` passed 6/6 in the
dev build (each mutant killed on the host backends; the control passes) and 13/13 in dev-cuda (the
host and CUDA suites, `cbst_unfill_drops_last` on the CUDA suite). `triad_type1_unbounded_keys` is
caught by the test whose bound goes down in the batch that deletes the top ids (the rows past the
new bound still hold the deleted three-member rows); with the bound going down in a later batch
the stale rows are already empty and the mutation was invisible (found by the first mutant run,
which added that case). `triad_inserted_requested_ids` first crashed the host suite (rows read
past the view's bound); the frontier now refuses an id outside the view (`internal_error`).

### 4.4 The rule of two (ADR 0055)

- `operators::count_delta` / `count_owned` (`cpp/src/operators/count_delta.hpp`): users
  triangle_delta (sequential, OpenMP, CUDA; its tests and kit unchanged in result) and
  triad_count's host engine. The required ownership argument is shown by the compile-failure
  CTests `operators.count_delta.static_assert.*` (no rule, a value that is not a rule, the triad
  host engine's phase without a rule; the control compiles).
- **cycle_count is not moved**: the move needs an ADR 0024 timing bar of 1.02 and this step takes
  no timed readings; deferred (ADR 0055 section 3).
- **device_frontier is not promoted**: sharing it would change the line-graph merge's code
  (`cpp/src/hypergraph/line_graph/device_frontier.cuh`) or add a per-call allocation and an upload
  to triad_count's CUDA frontier (ADR 0055 section 4). `cpp/src/hypergraph` is unchanged in this
  step.
- The CUDA engine's counting kernels are machine-code identical to c6 (`parity/triad_sass.py
  compare`, `M9a-sass-step4-escher_gpu.json`, built at `c8b6e68`: instructions 48 / 152 / 456 /
  552 / 512, registers 14 / 16 / 38 / 39 / 39, no stack, no local memory).

### 4.5 The checks of the step

| Check | Tree | Result |
|---|---|---|
| `ci/check.sh` (format, cpu-only and dev with `ctest -L cpu`, consumer, tidy, reuse, provenance, testdata, regen, harness, python, api, scaffold, docs, precommit) | `c8b6e68` + the draft records | **all checks passed** (cpu-only 1,244/1,244, dev 1,277/1,277) |
| the records' own checks (`DYNG_CHECK_ONLY="reuse provenance testdata harness docs precommit"`) | the records commit | passed |
| Clang 18 syntax (`clang_syntax.py`, the dev database incl. the mutants) | the step's tree | 401 files, 0 failures |
| cpu-only with `-D_FORTIFY_SOURCE=3` (GCC 12, `-Werror`), configured without the CUDA toolkit on PATH; `ctest -L 'triad_count|operators|triangle_delta|testing'` | the step's tree | built; 241/241 |
| ASan/UBSan, TSan (OpenMP off), TSan with OpenMP (Clang 18 and Archer, R020's local wrapper): `ctest -L 'triad_count|operators|triangle_delta'` | `c8b6e68` | 235/235, 219/219, 235/235 |
| `ci/gpu_local.sh` steps `build` and `triad` (dev-cuda, GPU 1): synccheck and initcheck on `dyng_triad_count_cuda_tests`; memcheck, synccheck, initcheck on one coauth_b50k_s1 batch; racecheck on the kit's C1, C11, C12 and on the make-test replays (60/60 in both modes) | `c8b6e68` | all passed |
| compute-sanitizer memcheck (`--leak-check full`) on `dyng_triad_count_cuda_tests` and `dyng_operators_cuda_tests` | `c8b6e68` | 0 errors |
| `ctest -L 'operators|triad_count|triangle_delta'` | dev-cuda (GPU 1), the tree of `93ef4a1` | 441/441 (the triad and triangle kits, the mutation CTests) |
| `parity/triad_sass.py compare` (c6 against `c8b6e68`) | parity-cuda | the five counting kernels identical |
| frozen headers: `git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,graph}` | head | empty |
| `ci/test_data_budget.py` | head | 1,002,671 bytes of 1,250,000 |

### 4.6 For step 5

- The host backends are exact on the coverage table (D6 pending); step 5's certificate requires
  the triad replays of every backend per that table, both modes (`certify.py`'s interim rule of
  step 1 to be replaced).
- The host engine is untuned (no gate): the sequential replays of the syn100k grid take 90-260 s
  per group with a recount after every batch; the OpenMP syn1m s1 grid 3-4.5 min per group at 8
  threads.
- `parity/mutate.py run --suites triad_count` builds one parity-cuda copy per mutation (about 8
  min each on the shared machine).
