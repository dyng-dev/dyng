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
