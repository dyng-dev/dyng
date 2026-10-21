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

## 5. Step gates-suites-certificate: the gates, the ESCHER suites, the 0.3 certificate

Code: `33dcca4` (`perf_ab.py triad_count`: records with the gates, the store points, the
contamination audit and the kernel reading), `faaf4e9` (`compare.py triad_count --certificate`),
`5ef6674` (the suites `ipdps26_escher.yaml` and `tkde26_escher_plus.yaml`, `bench_suite.py`),
`4036351` (`certify.py`: the ESCHER suites and the triad replays from 0.3, the dry run), `7f65284`
(the memory rows from the peaks' bytes, found by the dry run; the compacted records). ADR 0056.
**No C++ changed in this step** (the library at `4036351` is the one of `7a2393f`, step 4's
records commit, and of `c8b6e68`); `cpp/src/hypergraph` is unchanged since M8b (`git diff 65a62c9
-- cpp/src/hypergraph` is empty), so no ADR 0024 bar was due. Every gate reading ran from the
clean commit `4036351` (no `+dirty`), the parity-cuda build of that tree.

| Item | Value |
|---|---|
| Timed readings | GPU 0, the exclusive perf lock (taken by the harness), the base clock lock (SM 1170 / memory 7601 MHz; ADR 0021) for every gated and recorded-at-lock reading, `CUDA_MODULE_LOADING=EAGER`, no `OMP_*`, one heap layout, A/B alternation (A first), the contamination monitor and the audit, the per-round guard; 2026-10-15 22:57 UTC to 2026-10-16 00:15 UTC |
| Memory readings | GPU 1, the shared lock, no foreign process on GPU 1 before and after each case (recorded: none); 2026-10-16 00:02-00:06 UTC |
| Records | `parity/results/M9a-gates/` (the two suite summaries, their compacted records, the triad replays in the certificate's format, the dry certificate `parity.json` and its `README.md`); `M9a-gates-kernels-escher_gpu.json`; `M9a-maintain-head.json`; `M9a-sass-step5-escher_gpu.json`. The full records: `$DYNG_SCRATCH/runs/bench/m9a`, `$DYNG_SCRATCH/runs/m9a-step5` |
| Boost probe | M9a step 2 (`M9a-preread-boost-probe-escher_gpu.json`): the boost lock does not hold on syn1m / syn2m (A's process drops clocks in every round), so the synthetic gates run at base, as coauth-DBLP's (ADR 0021; criterion 12's rule) |

### 5.1 The timing gates (criterion 12; `ipdps26_escher.yaml`, `M9a-gates/ipdps26_escher.json`)

`parity/bench_suite.py run benchmarks/paper/ipdps26_escher.yaml` (`perf_ab.py triad_count run`):
**45 / 45 gated rows within their gates, ratios 0.858-1.028; the suite PASSED** (re-derived by the
summary and recomputed by the dry certificate's `recheck_gates`). Medians (ms), A / port, ratio:

| Case (rounds; rejected and repeated) | update[b1] | update[b2] | update[b3] | full_count | end_to_end |
|---|---|---|---|---|---|
| coauth_b50k_s1 (11; 1) | 511.3 / 485.5 **0.950** | 549.8 / 532.4 **0.968** | 549.9 / 536.5 **0.976** | 5,292.9 / 5,286.4 **0.999** | 9,390.4 / 8,839.2 **0.941** |
| coauth_b50k_s2 (11; 1) | 539.6 / 515.4 **0.955** | 540.8 / 530.8 **0.981** | 546.4 / 524.3 **0.960** | 5,293.4 / 5,286.7 **0.999** | 9,402.8 / 8,855.9 **0.942** |
| coauth_b50k_s3 (11; 0) | 524.0 / 503.5 **0.961** | 537.8 / 526.3 **0.979** | 533.5 / 521.4 **0.977** | 5,292.6 / 5,286.1 **0.999** | 9,371.2 / 8,843.1 **0.944** |
| syn1m_b50k_s1 (21; 0) | 164.3 / 143.5 **0.873** | 144.1 / 145.6 **1.010** | 155.1 / 147.0 **0.948** | 635.2 / 636.7 **1.002** | 2,492.3 / 2,440.4 **0.979** |
| syn1m_b50k_s2 (21; 0) | 164.2 / 143.3 **0.873** | 155.4 / 147.7 **0.951** | 144.8 / 146.3 **1.010** | 640.2 / 641.7 **1.002** | 2,480.1 / 2,447.3 **0.987** |
| syn1m_b50k_s3 (21; 0) | 148.4 / 142.3 **0.959** | 161.2 / 147.0 **0.912** | 155.8 / 147.7 **0.948** | 639.5 / 640.9 **1.002** | 2,492.3 / 2,454.0 **0.985** |
| syn2m_b50k_s1 (21; 0) | 184.3 / 158.2 **0.858** | 154.8 / 158.6 **1.025** | 169.0 / 161.1 **0.953** | 1,273.4 / 1,276.6 **1.002** | 4,433.6 / 4,250.6 **0.959** |
| syn2m_b50k_s2 (21; 0) | 157.1 / 153.3 **0.976** | 181.4 / 163.2 **0.899** | 155.5 / 159.6 **1.027** | 1,279.0 / 1,282.1 **1.002** | 4,415.5 / 4,238.3 **0.960** |
| syn2m_b50k_s3 (21; 0) | 157.6 / 154.1 **0.977** | 180.7 / 162.9 **0.902** | 154.6 / 159.0 **1.028** | 1,280.2 / 1,283.4 **1.002** | 4,417.4 / 4,249.8 **0.962** |
| gate | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.10 |

**Every gated row is within its bar, per batch** (no case passes only on its mean). The port's
update is flat across batches (142-163 ms on the synthetic inputs) while A's alternates between
about 145 and 165-185 ms by batch: A's `update.maintain` of a batch is 45-65 ms (its row views
and V2H fill rebuild more or less of the trees), the port's 46-52 ms. The highest synthetic
ratios (1.010-1.028) are the batches where A's maintain is at its low, and there the port's
`update.subtract` (1.014-1.029, the Step 0 normalization plus count(-)) is what is left; the
rows are 0.022-0.040 below the bar, unimodal over the 21 rounds (relative spread, max - min
over the median, 1-5 % on A's side and 1-2 % on the port's), and the same
batches sit at the same ratios at default clocks (5.3). The reading is not near a bar in the
sense of the step's stop rule, so it is reported here, not escalated.

Reported, not gated (ratios b1 / b2 / b3, seed 1): coauth `update.subtract` 1.002 / 1.000 /
0.996, `update.maintain` 0.770 / 0.889 / 0.868, `update.add` 0.957 / 0.959 / 0.987; syn1m
`update.subtract` 1.020 / 1.027 / 1.029, `update.maintain` 0.715 / 1.067 / 0.889, `update.add`
0.934 / 0.942 / 0.935; syn2m `update.subtract` 1.017 / 1.026 / 1.024, `update.maintain` 0.715 /
1.113 / 0.920, `update.add` 0.927 / 0.928 / 0.930. Outside the gated regions (A / port, ms,
seed 1): `build` 1,414 / 746 (coauth), 686 / 448 (syn1m), 1,411 / 899 (syn2m); `upload` 182 /
472, 68 / 175, 132 / 354 (the port uploads the stores, the line graph and the row views in
`hypergraph.upload`; A builds H2H on the host and uploads less); `load` 354 / 356, 361 / 424,
724 / 849. **Host synchronizations per batch** (Nsight Systems API trace, blocking calls inside
each batch's update region): A 96 / 96 / 96 (coauth) and 90-96 (syn), the port **22 / 22 / 22**
and 19-22 (the container's apply 16-19, the counting 3; the traits declare 3 for the counting),
so the port is at or below A's on every batch.

### 5.2 The contamination audit (criterion 12; ADR 0056 section 4)

Every accepted round of every reading (gated and recorded): 1-minute load average at most 10.3
(coauth s2; 6.5-9.6 elsewhere; the audit's limit 16), `procs_blocked` 0 in every round of the
gated readings (1 in one round of the default-clock and batch-size readings), no foreign compute
process on GPU 0 in any window, foreign CPU load at most 1.30 cores. The monitor rejected and
repeated two rounds of the gated readings: coauth_b50k_s1 round 4 (one busy GPU sample at memory
7600 MHz instead of the locked 7601) and coauth_b50k_s2 round 4 (foreign CPU load 2.38 cores).
No reading was flagged, none was taken again, `--keep-contaminated` does not exist for
triad_count. The machine's other users were idle in the window (GPU 1 carried this step's own
parity replays before the timed readings started; the timed readings waited for the exclusive
lock until they ended).

### 5.3 Recorded, not gated (criterion 12)

| Reading | Case | update b1 / b2 / b3 | full_count | end_to_end |
|---|---|---|---|---|
| default clocks (11 rounds) | coauth_b50k_s1 | 0.962 / 0.971 / 0.966 | 1.003 | 0.947 |
| | syn1m_b50k_s1 | 0.879 / 1.017 / 0.976 | 0.999 | 0.996 |
| | syn2m_b50k_s1 | 0.847 / 1.023 / 0.952 | 1.001 | 0.964 |
| batch sizes (base, 5 rounds) | coauth_b100k_s1 | 0.993 / 0.996 / 0.989 (A 943-991 ms) | 0.999 | 0.952 |
| | coauth_b200k_s1 | 0.990 / 1.012 / 1.014 (A 1,837-1,848 ms) | 0.999 | 0.970 |
| StatHyper types (base, 5 rounds; `tkde26_escher_plus.yaml`) | coauth_b50k_s1 type1 | 0.742 / 0.775 / 0.766 | 0.134 (3.8 / 0.5 ms) | 0.868 |
| | type2 | 0.751 / 0.804 / 0.790 | 0.825 | 0.878 |
| | type3 | 0.951 / 0.959 / 0.988 | 0.992 | 0.955 |

(On coauth_b200k_s1 the port's `update.maintain` is 1.04-1.25 of A's, a recorded sub-stage: the
larger deletions exercise the container's delete path, M8b's known 1.7-4x `maintain.delete`; the
gated scope, update, stays at 0.99-1.01.)

**The kernel reading** (`perf_ab.py triad_count kernels --cases syn1m_b50k_s1 --rounds 11`,
`M9a-gates-kernels-escher_gpu.json`; ncu at the base clocks, the update's `countWedges` /
`count_wedges_kernel` and `triangleKernel<MotifOp>` / `triangle_kernel<motif_op>` launches of the
three batches, the full count's skipped): subtract b1-b3 1.006 / 1.001 / 1.002, add 1.004 /
1.001 / 1.004 (about 47.5 ms of kernels per phase), DRAM bytes 1.000-1.004, both kernels with
the same registers (16 and 39), grid (98 and 8,815 blocks of 256) and occupancy limit.

**SASS and resources at HEAD** (`parity/triad_sass.py`, `M9a-sass-step5-escher_gpu.json`):
against A's unpatched objects, four counting kernels identical (count_wedges, keys<choose3>,
pairs<type2>, triangles<type3>), triangles<motif> +16 instructions (536 / 552: the error word of a
pattern of no class, ADR 0053), registers 14 / 16 / 38 / 39 / 39 on both sides, no stack, no
local memory.

**The M8b maintain reading at HEAD** (`perf_ab.py hypergraph maintain --gpu 0 --runs 5
--lock-clocks base`, `M9a-maintain-head.json`; sums of 3 batches): coauth50k `maintain` 0.814
(207.7 / 169.1 ms; M8b review 0.786), syn1m 0.851 (165.5 / 140.8; 0.833); sub-stages coauth
delete 1.769, insert 0.694, line_graph 0.822, views 0.300; host syncs per batch A 76 (70), dynG
19 (16). `cpp/src/hypergraph` is unchanged since M8b; the difference is the session (the same
code read again), well inside the bar ADR 0024 would apply.

The papers' figures beside the re-measured originals (informational; A's README: default clocks,
medians of 3 runs over seeds 1-3; this table: medians of the gated rounds of seed 1 at the base
lock, and the default-clock reading):

| Figure | A's README | A re-measured (base / default clocks) | port (base / default clocks) |
|---|---:|---:|---:|
| coauth 50K update per batch (s) | 0.546 | 0.511-0.550 / 0.515-0.564 | 0.486-0.537 / 0.496-0.545 |
| coauth 100K / 200K update per batch (s) | 1.033 / 1.936 | 0.943-0.991 / 1.837-1.848 (base) | 0.936-0.980 / 1.819-1.875 (base) |
| syn1M / syn2M update per batch (s) | 0.127 / 0.151 | 0.144-0.164 / 0.155-0.184 (base); 0.118-0.133 / 0.131-0.158 (default) | 0.143-0.147 / 0.158-0.161 (base); 0.117-0.120 / 0.134-0.137 (default) |
| full count coauth / syn1M / syn2M (s) | 5.22 / 0.47 / 0.95 | 5.29 / 0.64 / 1.27 (base); 5.03 / 0.48 / 0.97 (default) | 5.29 / 0.64 / 1.28 (base); 5.05 / 0.48 / 0.97 (default) |
| end to end coauth 50K / syn1M / syn2M (s) | 9.1 / 2.2 / 4.1 | 9.39 / 2.49 / 4.43 (base); 9.07 / 2.19 / 3.97 (default) | 8.84 / 2.44 / 4.25 (base); 8.59 / 2.18 / 3.82 (default) |
| store, ESCHER allocations (GB) | 1.83 / 0.67 / 1.34 | 1.92 / 0.70 / 1.41 (GB = 10^9 bytes: 1,833.7 / 670.5 / 1,341.6 MiB) | equal at every point |

**Improvements, reported apart from parity** (PLAN 8.6): the port's `update.maintain` (the
container's apply) is 0.72-0.89 of A's on every coauth batch and on the synthetic batches where
A rebuilds more (1.07-1.12 on those where A's maintain is at its low; M8b's container: fewer host
synchronizations, row views kept per commit), `build` 0.53-0.65 of A's (the parallel sorted
build), and host synchronizations per batch 19-22 against 90-96. They do not enter the gated
rows' verdicts, which compare the whole update.

### 5.4 The memory gates (criterion 13; `tkde26_escher_plus.yaml`, `M9a-gates/tkde26_escher_plus.json`)

`parity/bench_suite.py run benchmarks/paper/tkde26_escher_plus.yaml` (`perf_ab.py triad_count
memory`, GPU 1, shared lock, no foreign process before or after any case): **11 / 11 gated rows
within 1.05, ratios 1.000-1.0015; the suite PASSED.**

| Case | store_memory (dynG / A's deviceBytes(), every gated point) | device_memory: A peak (range) | port peak (range) | port / A | triad MiB during `hypergraph.apply` |
|---|---|---|---|---:|---:|
| coauth_b50k_s1 | build, b1, b2, b3: 1,922,784,204 / 1,923,300,764 / 1,923,713,052 / 1,924,170,844 bytes on both sides: **1.0000** (A's print 1,833.7 / 1,835.0 MB agrees) | 3,018.9 MiB (`update > update.maintain > maintain.h2h_merge`) | 3,021.9 MiB (`triad_count.update > triad_count.commit > hypergraph.apply > hypergraph.apply.line_graph`) | **1.0010** | 2.8 |
| syn1m_b50k_s1 | build, b3 (A's print points): 703,051,032 / 703,185,928: **1.0000** (b1, b2 equal too, recorded) | 1,131.4 MiB (the same range) | 1,133.0 MiB (the same range) | **1.0015** | 1.4 |
| syn2m_b50k_s1 | build, b3: 1,406,745,268 / 1,406,784,388: **1.0000** (b1, b2 equal too) | 2,211.7 MiB | 2,214.2 MiB | **1.0011** | 2.4 |

Recorded beside the gate (D1; reported, not gated):

| Case | footprint at the live peak A / port (MiB) | ratio | held between batches A / port (MiB) | ratios | live / pool / slack at the end of b3, port (MiB) |
|---|---|---:|---|---|---|
| coauth_b50k_s1 | 3,050.9 / 3,136.0 | 1.028 | 1,866-1,867 / 3,040-3,136 | 1.629 / 1.663 / 1.680 **(flagged > 1.05)** | 1,838.1 / 3,136.0 / 1,297.9 |
| syn1m_b50k_s1 | 1,163.4 / 1,184.0 | 1.018 | 702.6 / 1,152-1,184 | 1.640 / 1.685 / 1.685 **(flagged)** | 672.3 / 1,184.0 / 511.7 |
| syn2m_b50k_s1 | 2,243.7 / 2,240.0 | 0.998 | 1,373.6 / 2,240 | 1.631 / 1.631 / 1.631 **(flagged)** | 1,344.2 / 2,240.0 / 895.8 |

The peak per NVTX range (coauth_b50k_s1, MiB): `triad_count.compute` 1,852.6 (A's
`init.full_count` 1,852.6), `triad_count.count_minus` 1,837.6 (A's `update.subtract` 1,837.2),
`triad_count.finalize` 1,833.8 (A's `update.add` 1,837.7), `hypergraph.apply.line_graph` 3,021.9
(A's `maintain.h2h_merge` 3,018.9). The footprint between batches is the stream-ordered pool's
reserve (M8b: the pool keeps what the H2H merge reached; A's process holds its allocations plus
a 32 MiB Thrust pool), flagged above 1.05 as the step asks, not a gate (D1). Also recorded
(`cuda-memory-batch-sizes`): coauth_b100k_s1 and b200k_s1, store accounting 1.0000 at every
point, nsys peak 1.0012 and 1.0016, triad MiB during the apply 3.3 and 4.4, footprint at the peak
1.039 and 1.028, between batches 1.66-1.78 (flagged). The gated process keeps no public view and
no device-side by-content resolution (the driver draws its batches from the host model, ADR
0051; criterion 13's last line).

### 5.5 Exact parity at HEAD (criterion 8; the final replays)

`parity/compare.py triad_count --counter algorithm --mode both` from the parity-cuda build of
`7a2393f` / `4036351` (the same library; GPU 1 and the host, the shared lock), each converted to
the certificate's format (`triad_replay.py certificate`; `M9a-gates/parity-triad_count_escher_gpu-*.json`):

| Backend | Cases (the coverage table, D6 pending) | Result | Wall |
|---|---|---|---|
| cuda | the 240 grid cases + make-test's 60 | **300/300 exact**, both modes | 2,147 s (3 jobs) |
| sequential | make-test + the syn100k grid | **120/120 exact**, both modes | 634 s (6 jobs) |
| openmp, 8 threads | + the syn1m s1 grid and coauth_b50k_s1 types 1 and 2 | **142/142 exact**, both modes | 511 s (3 jobs) |
| openmp, 1 thread | the same | **142/142 exact**, both modes; the record identical to 8 threads case by case but for the times | 1,553 s (10 jobs) |

### 5.6 The dry 0.3 certificate (criterion 14; `M9a-gates/parity.json`)

`parity/certify.py write --version 0.3.0 --results <the records> --dry-run triad_count`:
**PASSED** (the dry run of `triad_count`; 17 parts outside it: the other golden sets, the
sssp, cycle_count and mosp suites, the release checks and the committed fixtures). It judged the
golden set `triad_count_escher_gpu` (manifest `2d8e49b5...`) with the replays of every backend
of the coverage table by key and by content (cuda 300 / 300, openmp 142 / 142, sequential 120 /
120 cases per mode) and both suites with every gated row recomputed by `recheck_gates` (45 + 11
rows; the store_memory rows point by point in bytes). Its first run **failed** on the five
`device_memory` rows: the summary had taken the record's ratio (from the peaks' bytes) next to
MiB rounded to 0.1, so port / original did not recompute; fixed in the summary and the
certificate (the MiB from the bytes, `7f65284`) and summarized again from the same records (no
reading repeated). The same directory with `update[b3]` removed from the timing summary fails
("no gated reading of update[b3] (cuda)"), and with the sequential replay removed fails ("120 of
the 120 cases the coverage table gives sequential have no passed replay by_key ..."); the tests
`parity/tests/test_release_cert.py` (a 0.3 certificate without the ESCHER suites, a store row,
a per-batch row, a backend's or a mode's replay fails; 0.1 and 0.2 certificates pass; records
with a missing batch, too few rounds, a kept contaminated round, a flagged audit, a failed guard,
another clock lock, another input digest or a missing store point are refused) and
`parity/tests/test_triad_perf.py` cover the rules.

### 5.7 The checks of the step

| Check | Tree | Result |
|---|---|---|
| `ci/check.sh` (all steps) | `7f65284` + the uncommitted records | all passed: cpu-only 1,244 / 1,244, dev 1,277 / 1,277, consumer, tidy, reuse, provenance, testdata (1,002,671 of 1,250,000 bytes), regen, harness, python, api, scaffold, docs, pre-commit |
| `ci/gpu_local.sh` (dev-cuda, GPU 1; `dyng-work/runs/m9a-step5/gpu_local.log`) | `7f65284` (started one minute after it; nothing but docs and records changed since) | **all 20 steps passed** in 3 h 50 min (19:18-23:08 CDT, RTX A5000, driver 590.48.01, CUDA 13.1): `ctest -L gpu` 812 / 812, `ctest -L cpu` 1,277 / 1,277, the sssp and cycle_count cuda corpora, memcheck on every gpu executable (0 errors), synccheck (sssp, mosp, cycle_count), racecheck, the cbst and hypergraph sanitizers and replays; the `triad` step: synccheck and initcheck on `dyng_triad_count_cuda_tests`, memcheck, synccheck and initcheck on one coauth_b50k_s1 batch (0 errors), racecheck on the kit's C1, C11 and C12 (0 hazards) and on the make-test replays in both modes (15 groups x 4 patterns); the plugin wheel (`pytest -m gpu` 61 passed); clang-tidy |
| ASan + UBSan, TSan-OpenMP (Clang 18 + Archer) on `triad_count\|operators\|triangle_delta` | `7f65284` | 247 / 247 each |
| a fresh clone: cpu-only with `-D_FORTIFY_SOURCE=3`, no nvcc on PATH; parity-cuda | `7f65284` | built without warnings; `ctest -L cpu` 1,244 / 1,244; `ctest -L triad_count` 297 / 297 (GPU 1) |
| Clang 18 `-fsyntax-only` (`clang_syntax.py`) | `7f65284` | 401 files, 0 failures |
| `parity/mutate.py run --suites triad_count,cbst,hypergraph,kit` | `7f65284` | control passed, 17 / 17 detected (`M9a-mutate-step5.json`) |
| frozen headers against `65a62c9` and `main` (`9cde147`); `cpp/src/hypergraph` against `65a62c9` | HEAD | empty (all three) |
| ESCHER-GPU's working tree | | `git status --porcelain` empty |
| signatures | `9cde147..HEAD` | every commit SSH-signed (`G`), identity S M Shovan |

### 5.8 The acceptance criteria of M9a, checked at the end

| # | Criterion | Where | State |
|---|---|---|---|
| 1 | the timed-region map, committed first (`2f54bce` before `be82dec`, 14 commits apart); per-batch rows, the scope rules, `load_regions` refuses unknown timers | 1.1; ADR 0048 | met |
| 2 | the golden set (G + make-test, 300 cases), counts unpatched, `--check` exit 0, batches from the patched recorder, exported twice with identical manifests, `verify`'s three assertions, 367.7 MB new (7.1 GB total) | 1.2 | met |
| 3 | A against its CPU reference: 90 runs agree (coauth main at `--batches 1 --batch-size 20` finished in 143.5 s) | 1.3 | met |
| 4 | the independent oracle, its derived class map (128 ids), isolation, the 24 single-triad files, the go/no-go (423 replays) | 2.1, 2.2 | met (GO) |
| 5 | the public header (host-only, in `dyng.hpp`, api_snapshot, not frozen; no CBST key, +1 or sentinel) | 3.5; ADR 0052 | met; the names await the author (D9) |
| 6 | delta semantics and hazards I1, I2, I7, I9, each tested; baseline + deltas == compute() on every backend | 3.5, 4.2; ADRs 0053, 0054 | met |
| 7 | the CUDA port: reproducible transplant, six mechanical commits each green, provenance headers, internal row views, syncs 22 against A's 96, resources and SASS recorded | 3.1, 3.2, 5.3 | met |
| 8 | exact parity, both modes: cuda 300 / 300, sequential 120 / 120, openmp 142 / 142 at 1 and 8 threads (identical); the comparator's planted defects fail; 10 repeated coauth runs identical | 2.2, 3.3, 4.1, 5.5 | met |
| 9 | kernel paths: counters per (mode, shape), both early outs, insert-only, delete-only, delete-all, a batch larger than the graph, skewed / medium on thread, warp and block paths, the 128 Venn ids and the error word | 3.4, 4.2 | met |
| 10 | the conformance kit on sequential, openmp(8) and cuda, both instantiations, presets, mixes, shapes; C4 skipped with its reason; C8 strict; C10 two patterns and the fake; the kit's planted bug fails; property tests | 3.5, 4.2 | met |
| 11 | the five triad mutations and cbst_unfill_drops_last on the triad suites of both engines, the control, `mutate.py list` | 4.3, 5.7 | met |
| 12 | the timing gates: 45 / 45 within (coauth 11 rounds, syn 21 rounds, base lock after the boost probe), the recorded readings, the contamination audit | 5.1-5.3 | met |
| 13 | the memory gates: store accounting 1.0000 at every gated point, the Nsight Systems peak 1.0010-1.0015 (D1); the recorded peaks, ranges, triad bytes, footprint (flagged, not gated) | 5.4 | met |
| 14 | the suites (both validate), the 0.3 certificate plumbing and its tests, the dry 0.3 certificate passes `recheck_gates` and fails without a gate row or a replay | 5.6 | met |
| 15 | the frozen surface, no hypergraph change (no ADR 0024 bar due), ADRs 0048-0056 logged, the records, the checks green, `ci/tests/test_regen.py` fixed (step 3), the test data within the 1.25 MB budget (D5) | 5.7 | met |

## 6. The review fixes: the replays and the gates again on the final library

The confirmed review findings against `fe9c56c` are fixed in `8aa1b4b`..`2eda830` (the list and
the commits: `docs/developer/retrospectives/M9a.md`, "The review fixes"). Three of them changed
the library (ADR 0057's exact 64-bit arithmetic, `e1b0205`; the update scratch's headroom,
`b153ff3`; the driver host model's geometric growth, `756fb68`), so by `certify.py`'s library
scope every step-5 replay and reading was stale; the step-5 memory records also lack the count
guard the review added. Everything below was taken again on that final library (`e015a12` for the
replays, the clean tree `2eda830` for the readings; `git diff e015a12 HEAD -- cpp tools cmake
CMakeLists.txt CMakePresets.json` is empty). Records: `parity/results/M9a-review-gates/` (the
replays, both suites' summaries and compacted records, the dry certificate `parity.json` and its
`README.md`), `M9a-review-kernels-escher_gpu.json`, `M9a-review-maintain-head.json`,
`M9a-review-sass-escher_gpu.json`; the full records `$DYNG_SCRATCH/runs/bench/m9a-review`,
`$DYNG_SCRATCH/runs/m9a-review`, `$DYNG_SCRATCH/runs/m9a-review2`.

### 6.1 Exact parity (criterion 8)

The golden set was exported again first (`98cfd8d`, `e015a12`: no run parameter hashed; two fresh
patched copies on GPU 1 and GPU 0, identical `MANIFEST.sha256` `4fb7dbcf...`; the counts and
batches unchanged). `compare.py triad_count --counter algorithm --mode both`, the comparator now
also requiring `deleted == ` A's deletions and `ignored == 0` per batch:

| Backend | Cases | Result | Wall |
|---|---|---|---|
| cuda (GPU 1) | the 240 grid cases + make-test's 60 | **300/300 exact**, both modes | 2,138 s (3 jobs) |
| sequential | make-test + the syn100k grid | **120/120 exact**, both modes | 912 s (6 jobs) |
| openmp, 8 threads | + the syn1m s1 grid and coauth_b50k_s1 types 1 and 2 | **142/142 exact**, both modes | 770 s (3 jobs) |
| openmp, 1 thread | the same | **142/142 exact**, both modes; the per-case reports identical to 8 threads but for the times | 1,789 s (10 jobs) |

### 6.2 The timing gates (criterion 12)

`bench_suite.py run benchmarks/paper/ipdps26_escher.yaml` (`perf_ab.py triad_count run`; GPU 0,
the exclusive lock, the base clock lock SM 1170 / memory 7601 MHz, `CUDA_MODULE_LOADING=EAGER`,
no `OMP_*`, one heap layout, A/B alternation, the per-round guard, the contamination audit;
18:48-19:30 CDT on 2026-10-20): **45 / 45 gated rows within their gates, ratios 0.853-1.019;
PASSED.** Medians (ms), A / port, ratio:

| Case (rounds; rejected and repeated) | update[b1] | update[b2] | update[b3] | full_count | end_to_end |
|---|---|---|---|---|---|
| coauth_b50k_s1 (11; 0) | 510.3 / 485.7 **0.952** | 549.7 / 531.1 **0.966** | 549.4 / 531.7 **0.968** | 5,293.7 / 5,286.4 **0.999** | 9,827.4 / 9,006.2 **0.916** |
| coauth_b50k_s2 (11; 0) | 540.3 / 516.1 **0.955** | 539.6 / 524.0 **0.971** | 547.6 / 519.7 **0.949** | 5,294.6 / 5,285.8 **0.998** | 9,851.7 / 9,013.9 **0.915** |
| coauth_b50k_s3 (11; 1) | 524.8 / 499.9 **0.953** | 537.4 / 518.2 **0.964** | 533.2 / 515.5 **0.967** | 5,292.9 / 5,286.8 **0.999** | 9,820.7 / 8,990.1 **0.915** |
| syn1m_b50k_s1 (21; 0) | 164.5 / 142.7 **0.868** | 144.4 / 144.7 **1.002** | 155.2 / 146.2 **0.942** | 635.3 / 636.7 **1.002** | 2,483.3 / 2,441.5 **0.983** |
| syn1m_b50k_s2 (21; 0) | 164.2 / 142.9 **0.870** | 155.5 / 146.7 **0.943** | 144.9 / 145.2 **1.002** | 640.1 / 641.8 **1.003** | 2,502.7 / 2,458.3 **0.982** |
| syn1m_b50k_s3 (21; 0) | 148.2 / 141.5 **0.954** | 161.3 / 146.2 **0.907** | 156.2 / 147.0 **0.941** | 639.5 / 641.0 **1.002** | 2,505.0 / 2,459.1 **0.982** |
| syn2m_b50k_s1 (21; 0) | 184.3 / 157.1 **0.853** | 154.5 / 157.2 **1.018** | 170.3 / 159.4 **0.936** | 1,273.6 / 1,276.7 **1.002** | 4,503.1 / 4,353.8 **0.967** |
| syn2m_b50k_s2 (21; 0) | 157.5 / 151.7 **0.963** | 181.3 / 161.4 **0.890** | 155.4 / 158.3 **1.019** | 1,279.1 / 1,282.2 **1.002** | 4,501.5 / 4,423.3 **0.983** |
| syn2m_b50k_s3 (21; 0) | 157.9 / 153.0 **0.969** | 181.4 / 161.3 **0.889** | 155.2 / 157.7 **1.016** | 1,280.2 / 1,283.4 **1.003** | 4,495.9 / 4,350.1 **0.968** |
| gate | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.10 |

Against step 5 the update rows are as before or lower (the highest synthetic rows 1.016-1.019
against 1.025-1.028); ADR 0057's checked arithmetic does not touch the h_motif hot loop
(motif_op's adds stay unchecked; the flush is carry-tested once per block and bin). A's end to
end on coauth is 9.82-9.85 s here against 9.37-9.40 s in step 5 (the port 8.99-9.01 against
8.84-8.86), both sides in the same session. The audit: 1-minute load at most 9.6, procs_blocked
at most 1, no foreign process on GPU 0, no round flagged; the monitor rejected and repeated one
round (coauth_b50k_s3 round 3, foreign CPU load 2.27 cores). Host synchronizations per batch:
A 96 (coauth) and 90-96 (syn), the port 22 and 19-22, as in step 5.

**The host model beside the gate (reported, not gated; `756fb68`, D12).** A's `update` contains
its host model (`update.maintain_untimed`: 1.5-2.1 ms a batch, 7.3-13.1 ms in the batch where its
`h2v_` grows); the port's driver keeps its host model after the update timer, 5.7-6.6 ms a batch
(11.4-18.2 ms where the row vector grows). `update_with_host_model[b<N>]` (A's update against the
port's update plus its host model) is 0.937-1.061 at the base lock, above 1.05 on three rows
(syn2m_b50k_s1 b2 1.060, s2 b3 1.061, s3 b3 1.057; syn1m 1.042-1.043 on its highest rows), and
1.070 on syn2m_b50k_s1 b2 at default clocks. The library-only comparison (the port's update
against A's update minus its untimed in-maintain time) is 0.910-1.031. The gated rows compare the
library's update with A's, as approved (ADR 0048 section 3 records the asymmetry); the driver's
host model, which validates, sorts and copies each inserted row, is the part that costs more than
A's assignment. Left for the author (D12, GOVERNANCE.md): a protocol that counts a host model on
both sides, or a cheaper driver host model (a library change that would make the readings stale
again).

### 6.3 Recorded, not gated

| Reading | Case | update b1 / b2 / b3 | full_count | end_to_end |
|---|---|---|---|---|
| default clocks (11 rounds) | coauth_b50k_s1 | 0.951 / 0.966 / 0.951 | 1.004 | 0.919 |
| | syn1m_b50k_s1 | 0.886 / 1.000 / 0.979 | 0.998 | 0.993 |
| | syn2m_b50k_s1 | 0.841 / 1.022 / 0.944 | 1.001 | 0.962 |
| batch sizes (base, 5 rounds) | coauth_b100k_s1 | 0.985 / 0.990 / 0.975 | 0.999 | 0.929 |
| | coauth_b200k_s1 | 0.986 / 1.015 / 1.018 | 0.999 | 0.949 |
| StatHyper types (base, 5 rounds; `tkde26_escher_plus.yaml`) | coauth_b50k_s1 type1 | 0.724 / 0.751 / 0.731 | 0.138 (3.9 / 0.5 ms) | 0.791 |
| | type2 | 0.736 / 0.789 / 0.769 | 0.824 | 0.798 |
| | type3 | 0.950 / 0.958 / 0.963 | 0.992 | 0.927 |

The kernel reading (ncu at the base clocks, syn1m_b50k_s1, 11 rounds): subtract 1.005 / 1.003 /
1.003, add 1.002 / 1.005 / 1.005, DRAM bytes 0.999-1.004. SASS and resources against A
(`M9a-review-sass-escher_gpu.json`): count_wedges identical; keys<choose3> 152 -> 288
instructions, pairs<type2> 456 -> 512, triangles<type3> 512 -> 576 (the checked products and
sums of ADR 0057), triangles<motif> 536 -> 560 (step 5: 552); registers 14 / 22 / 37 / 39 / 40
against A's 14 / 16 / 38 / 39 / 39; no stack, no local memory. The allocation calls per update
(nsys API trace, coauth_b50k_s1, GPU 1): 5 pool allocations in count(-) of batch 1, **none in
batches 2 and 3** (step 5's build freed and reallocated the flags in each). The M8b maintain
reading at HEAD (`perf_ab.py hypergraph maintain --runs 5 --lock-clocks base`): coauth50k
`maintain` 0.788 (208.5 / 164.4 ms), syn1m 0.833 (165.5 / 137.8); `cpp/src/hypergraph` unchanged
since M8b.

### 6.4 The memory gates (criterion 13)

`bench_suite.py run benchmarks/paper/tkde26_escher_plus.yaml` (GPU 1, the shared lock, no foreign
process before or after any case; the count guard on the plain and the profiled runs of both
sides, 4 processes a case): **11 / 11 gated rows within 1.05, ratios 1.000-1.002; PASSED.**

| Case | store_memory (every gated point) | device_memory: A peak | port peak (range) | port / A | triad MiB during the apply |
|---|---|---|---|---:|---:|
| coauth_b50k_s1 | build, b1, b2, b3 equal in bytes: **1.0000** | 3,018.9 MiB (3,165,509,863 B) | 3,023.3 MiB (3,170,201,054 B; `hypergraph.apply.line_graph`) | **1.0015** | 4.2 |
| syn1m_b50k_s1 | build, b3: **1.0000** | 1,131.4 MiB | 1,133.7 MiB (the same range) | **1.0021** | 2.1 |
| syn2m_b50k_s1 | build, b3: **1.0000** | 2,211.7 MiB | 2,215.4 MiB | **1.0017** | 3.6 |

The triad bytes live during the apply grew by 0.7-1.4 MiB against step 5 (the headroom of the
update's scratch, `b153ff3`); the peaks stay in the H2H merge on both sides. Footprint at the live
peak 1.028 / 1.018 / 0.998; between batches 1.63-1.68 (coauth), 1.64-1.69 (syn1m), 1.63 (syn2m)
**(flagged above 1.05, not gated; D11)**. Recorded: coauth_b100k_s1 and b200k_s1, store 1.0000,
nsys peak 1.0018 and 1.0023, between batches 1.66-1.80.

### 6.5 The dry 0.3 certificate (criterion 14)

`certify.py write --version 0.3.0 --results <the records> --dry-run triad_count` over the replays
of 6.1 and the readings of 6.2-6.4: **PASSED** (17 parts outside it), with the review's new rules
(every gated row of every case of a reading, the medians recomputed from the samples, the round
counts, the protocol fields, the memory count guard, the release series of `VERSION`). The step-5
directory `parity/results/M9a-gates/` given to the same command now **fails**: its gate records
were measured on `4036351` ("does not have the release's code"), its memory records have no count
guard, and its replays are of the earlier golden manifest. It stays as step 5's record.

### 6.6 The checks of the review fixes

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
