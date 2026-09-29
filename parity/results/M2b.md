# M2b parity certificate: `cycle_count` (CUDA) against CycleEnumeration-GPU@0a976ad

Date: 2026-09-28. Format: PLAN Section 8.3 ("Parity certificate"). M2b merges the accepted M2a
branch (`cycle_count` on the CPU backends) and ports `cycle_count` to CUDA. This file starts with
the re-verification after the merge (section 1), then the parity of the CUDA port (section 2);
the performance gates follow.

## 1. Merge re-verification

The branch `m2b-cycle-cuda` (from `main` at `eda8b8b`, INT1) merges `m2-cycle` (M2a) with the
merge commit `c8d5d5d`. Everything below was measured on `fbeb58a` (the merge plus three
documentation commits; no library code changed after the merge commit). The performance records
say `fbeb58a+dirty` because `CHANGELOG.md` was being edited while they ran; the timed binaries
were built from `fbeb58a` and the only uncommitted change was that file.

| Item | Value |
|---|---|
| Port | dynG `fbeb58a97513dbbcf1548c660fe49a8ff944e3ff` |
| References | MOSP-OpenMP `c352151`, MOSP-CUDA `e220ee2` (unpatched copies for performance, rebuilt and verified by the harness), CycleEnumeration-GPU `0a976ad` (unpatched copy's `build/cycle-enum`) |
| Toolchain | GCC 12.2.0, CUDA 13.1, driver 590.48.01; `parity` and `parity-cuda` presets (`-O3`), `dev` / `dev-cuda` for the test suites |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1 |

### 1.1 Gates and test suites

| Check | Result |
|---|---|
| `ci/check.sh --parity` (clang-format, `cpu-only` and `dev` builds with `ctest -L cpu` (401 and 414 tests), clang-tidy naming, REUSE, provenance, harness tests, docs, pre-commit, `ctest -L parity` on the `parity` preset) | all steps passed; the parity label ran `parity.sssp.mosp_openmp_c352151`, `parity.cycle_count.cycle_enum_0a976ad` and the dataset tests `CycleEnumDatasets.DigestsEqualTheOriginal`, `CycleCountDatasets.HistogramsEqualTheOriginal` (4 / 4) |
| `ci/gpu_local.sh` (`dev-cuda`, GPU 1: build, `ctest -L gpu` 92 tests, `ctest -L cpu` 414 tests in the CUDA build, the sssp golden corpus on `cuda`, compute-sanitizer memcheck and synccheck, clang-tidy on the CUDA branches) | all steps passed; memcheck and synccheck 0 errors |
| `ci/docs.sh` | passed (after `174707d`: the API page of the `cycle_count` group, which the M4 site requires for every Doxygen group) |

### 1.2 Byte parity (no tolerance)

| Corpus | Configurations | Result | Record |
|---|---|---|---|
| `sssp`, 495 cases (MOSP-OpenMP@c352151 goldens) | sequential, openmp:1, openmp:4, openmp:16, openmp:28, sequential/int64, openmp:4/int64 (`parity` preset) | **495 / 495 in every configuration** | `M2b-merge-sssp-parity-preset.json` |
| `sssp`, 495 cases | cuda, cuda/int64 (`parity-cuda` preset, GPU 1) | **495 / 495 in both** | `M2b-merge-sssp-cuda-parity-cuda-preset.json` |
| `sssp`, 495 cases | cuda (`dev-cuda`, inside `ci/gpu_local.sh`) | **495 / 495** | the gate's summary |
| `cycle_count`, 24 cases (CycleEnumeration-GPU@0a976ad goldens: fixtures, DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB k = 3, the seed-1 updates 1K/25K/50K on DD, GitHub, Twitch, the DD locality and bound sweep) | the three configurations of `compare.py cycle_count --full` (sequential, openmp:4, openmp:56) | **72 / 72 replays equal** (histograms and generated batches) | `M2b-merge-cycle_count-parity-preset.json` |

The CPU `cycle_count` parity is unchanged from M2a (`M2a-cycle_count-parity-preset.json`: the
same 24 cases, the same verdicts).

### 1.3 Performance spot checks (PLAN 8.6; exclusive perf lock)

Protocol as the M1b certificate (sections 13 and 14) and the M2a certificate (section 3.1):
unpatched originals, A/B/A/B alternating rounds with the original first, medians, contamination
monitor (a round is repeated when either side had more than 2 foreign cores), 28 OpenMP threads
pinned to cores for `sssp`, 56 for `cycle_count`, `CUDA_MODULE_LOADING=EAGER`. The CUDA gate is
read at locked clocks (ADR 0018, accepted option B: `--lock-clocks boost`, SM 1695 MHz / memory
7601 MHz; every busy GPU sample of both sides in every accepted round was at the locked clocks).
roadNet-CA has regions under 10 ms, so it ran 21 rounds; road_usa (every gated region >= 10 ms)
ran 11.

**`sssp` on CUDA against MOSP-CUDA@e220ee2, clocks locked** (ratio dynG / original;
`M2b-merge-perf-cuda-<graph>.json`):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds | rejected | foreign CPU max (cores) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-CA | safe50k | 0.985 | 0.997 | 0.997 | 8.76 | 0.871 | 0.876 | 21 | 0 | 1.33 |
| roadNet-CA | unsafe50k | 0.990 | 1.000 | 0.996 | 8.75 | 0.872 | 0.846 | 21 | 0 | 1.30 |
| roadNet-CA | local10k | 0.978 | 1.000 | 0.999 | 3.49 | 0.844 | 0.877 | 21 | 0 | 1.68 |
| road_usa | safe50k | 1.000 | 0.999 | 1.000 | 102.83 | 0.841 | 0.816 | 11 | 0 | 1.42 |
| road_usa | unsafe50k | 1.000 | 1.001 | 1.001 | 102.84 | 0.848 | 0.768 | 11 | 1 | 1.37 |
| road_usa | local10k | 0.994 | 0.994 | 0.996 | 22.70 | 0.833 | 0.770 | 11 | 0 | 1.27 |

Gates: per objective <= 1.10x on roadNet-CA (regions of 3.1-8.9 ms, 21 runs), <= 1.05x on
road_usa; apply and end to end <= 1.10x. **18 / 18 per-objective readings within the gate**
(0.978-1.001x), apply 0.83-0.87x, end to end 0.77-0.88x: the same as the M1b acceptance record
(section 14 of `M1b.md`: roadNet-CA 0.977-1.000x, road_usa 0.993-1.001x).

**`sssp` on OpenMP (28 threads) against MOSP-OpenMP@c352151** (`M2b-merge-perf-openmp-<graph>.json`):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds | rejected | foreign CPU max (cores) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-CA | safe50k | 0.803 | 0.910 | 0.925 | 31.09 | 0.893 | 0.806 | 21 | 0 | 0.44 |
| roadNet-CA | unsafe50k | 0.822 | 0.904 | 0.920 | 29.86 | 0.878 | 0.864 | 21 | 0 | 0.32 |
| roadNet-CA | local10k | 0.699 | 0.671 | 0.767 | 14.21 | 0.897 | 0.867 | 21 | 0 | 0.40 |
| road_usa | safe50k | 0.876 | 0.924 | 0.920 | 317.31 | 0.736 | 0.845 | 11 | 0 | 0.37 |
| road_usa | unsafe50k | 0.865 | 0.915 | 0.911 | 320.34 | 0.743 | 0.796 | 11 | 0 | 0.19 |
| road_usa | local10k | 0.737 | 0.839 | 0.841 | 74.51 | 0.736 | 0.815 | 11 | 0 | 0.29 |

**18 / 18 per-objective readings within the gate** (0.67-0.93x; roadNet-CA local10k obj2, 9.4 ms,
is gated at 1.10x), apply 0.74-0.90x, end to end 0.80-0.87x. The `invalidated` counters were
equal to the original's in every sample of every batch on both backends.

**`cycle_count` OpenMP update gate (56 threads) against CycleEnumeration-GPU@0a976ad** (11
rounds; `M2b-merge-perf-openmp-cycle_count.json`; the histograms of both sides were identical and
equal to the golden in every round):

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | M2a ratio | Foreign cores max A / B |
|---|---|---:|---:|---:|---|---:|---|
| DD 25K+25K k = 4 | update | 26.2 | 16.1 | 0.615 | <= 1.05 ok | 0.646 | 0.80 / 0.82 |
| DD 25K+25K k = 4 | update_end_to_end | 364.9 | 306.3 | 0.839 | <= 1.10 ok | 0.838 | |
| GitHub 25K+25K k = 4 | update | 294.6 | 125.5 | 0.426 | <= 1.05 ok | 0.444 | 0.16 / 0.17 |
| GitHub 25K+25K k = 4 | update_end_to_end | 3,262.6 | 1,844.2 | 0.565 | <= 1.10 ok | 0.539 | |

No run was flagged (> 2 foreign cores).

**Verdict.** The merge introduced no regression: byte parity of `sssp` (495 / 495 on sequential,
OpenMP and CUDA, both edge types) and of `cycle_count` (24 cases x 3 configurations) holds, every
test suite and sanitizer passes, and the re-checked gates read as before the merge.

### 1.4 Reproducing

```bash
source scripts/dev_env.sh
ci/check.sh --parity
ci/gpu_local.sh
ci/docs.sh
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py \
    --exe build/parity/tools/compat/dyng-compat-mosp \
    --configs sequential,openmp:1,openmp:4,openmp:16,openmp:28,sequential/int64,openmp:4/int64 \
    --json parity/results/M2b-merge-sssp-parity-preset.json
CUDA_VISIBLE_DEVICES=1 flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py \
    --exe build/parity-cuda/tools/compat/dyng-compat-mosp --configs cuda,cuda/int64 \
    --json parity/results/M2b-merge-sssp-cuda-parity-cuda-preset.json
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py cycle_count \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --full \
    --json parity/results/M2b-merge-cycle_count-parity-preset.json
# performance (perf_ab.py takes the exclusive lock itself); roadNet-CA with 21 runs, road_usa_g with 11
for gr in roadNet-CA:21 road_usa_g:11; do
  g=${gr%:*} runs=${gr#*:}
  parity/perf_ab.py run --backend cuda --gpu 0 --lock-clocks boost \
      --exe build/parity-cuda/tools/compat/dyng-compat-mosp --graph $g --runs $runs \
      --json parity/results/M2b-merge-perf-cuda-$g.json
  parity/perf_ab.py run --backend openmp --exe build/parity/tools/compat/dyng-compat-mosp \
      --graph $g --runs $runs --json parity/results/M2b-merge-perf-openmp-$g.json
done
parity/perf_ab.py cycle_count run --exe build/parity/tools/compat/dyng-compat-cycle-enum \
    --cases DD_k4_25000_25000_s1,github_k4_25000_25000_s1 --runs 11 \
    --json parity/results/M2b-merge-perf-openmp-cycle_count.json
```

## 2. The CUDA port: byte parity (step cycle-cuda)

The CUDA backend of `cycle_count` (commits `8fdec6e`..`4a1a8b5`; ADR 0020 for the resident
graph) against CycleEnumeration-GPU@0a976ad's **CUDA** backend, built from a `git archive` of the
pinned commit with `-DCYCLE_ENUM_ENABLE_CUDA=ON` by `parity/build_reference.sh` (the patched copy
for the goldens and fixtures: its sources unchanged, the exporter added; the unpatched copy for
performance). GPU 1 (the development GPU), CUDA 13.1.

| Item | Value |
|---|---|
| Port | dynG `7d3e116` (CUDA replays; `+dirty`: the harness change `fdc0f3a` was being written, the timed library and tool are those of `40a0000`), `fdc0f3a` (host replays; `+dirty`: documentation) |
| Reference | CycleEnumeration-GPU `0a976ad`, `cycle-enum --backend cuda --cuda-device 0` (CUDA_VISIBLE_DEVICES=1, CUDA_MODULE_LOADING=EAGER) and its exporter built against the copy's CUDA libraries (`export_cycle_enum_cuda`) |

### 2.1 The originals cross-checked (PLAN 6.3 step 2)

Before a CUDA golden or fixture was written, the original's CUDA output was compared with its
CPU output of the same case: every one of the 80 random fixture cases (k = 2..7 and no bound;
static counts before and after the batch with the work queue, the naive counter and every kind of
work item; `update_static_histogram_cuda`) and the 23 fixture-graph jobs equal the sequential
histograms (`make_cycle_enum_fixtures.sh`, section 5b); the 21 dataset cases that the OpenMP
golden set also has are byte-identical to it (`histogram.csv`, `prior.csv`, `batch.txt`; the
other three, the 100K + 100K updates of DD and GitHub and the COLLAB update, are new), and every
update case's own `--compare-recompute` says `match=yes`. The original's CUDA and CPU backends agree
everywhere they were compared.

### 2.2 Byte parity (no tolerance)

| Corpus | Configurations | Result | Record |
|---|---|---|---|
| `cycle_count_cuda`, 24 cases (the original's CUDA backend: DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB k = 3; the seed-1 updates k = 4 of 1K+1K, 25K+25K, 50K+50K, 100K+100K on DD and GitHub, 1K..50K on Twitch, 25K+25K on COLLAB, and DD 25K+25K at k = 3 and 5) | cuda (the graph uploaded per call, `int32_t` offsets), cuda:resident, cuda:int64 (`parity-cuda` preset) | **72 / 72 replays equal** (histograms and generated batches) | `M2b-cycle_count-cuda-set-parity-cuda-preset.json` |
| `cycle_count_cuda` | openmp:56 (`parity` preset; COLLAB's update is replayed on cuda only: its prior is the k = 4 count of COLLAB) | **23 / 23 equal** | `M2b-cycle_count-cuda-set-openmp-parity-preset.json` |
| `cycle_count`, 24 cases (the M2a OpenMP corpus), after the change of the host commit (Step 0 once, ADR 0020) | sequential, openmp:4, openmp:56 (`--full`) | **72 / 72 replays equal** | `M2b-cycle_count-parity-preset.json` |
| fixtures: 80 random cases x 7 bounds (before, after, update, and every scheduler and kind of work item), 23 fixture-graph jobs | cuda, both offset types (`CycleCountFixtures.*` in `dyng_cycle_count_cuda_tests`) | equal to `cases/*.cuda`, `counts/*.cuda` | `ctest -L gpu` |
| the original CLI on cuda: 18 runs (schedulers, work items, bounds 64 and 65, updates, errors) | `dyng-compat-cycle-enum --backend cuda` | standard output and exit status equal (`compat_cycle_enum.cli.cuda`) | `ctest -L gpu` |

Cross-backend equality (CUDA = OpenMP = sequential) on randomized graphs and chains of batches,
the device set apply against the host apply (byte-equal CSR, apply summaries, normalized lists
and insertion ids), the bounds 2..64 and the rejection beyond, and the corner cases are the CUDA
cases of `dyng_cycle_count_cuda_tests` (`cycle_count_cuda_test.cpp`).

## 3. Performance: the harness checked, and the CPU update after the change of the commit

The full CUDA gate of M2b acceptance criterion 4 (all ten cases, both scopes) is section 4. In
that step the CUDA harness was run end to end on DD, and the OpenMP update gate was re-checked
because the host commit changed (Step 0 once, ADR 0020). Port `6123e1d` plus the fix of
`dyng-compat-cycle-enum` that reads only the timed call's stages in the resident scope (the
records say `+dirty`; the fix is committed right after this section); unpatched original;
exclusive perf lock.

**CUDA, DD** (`parity/cycle_count_perf.py run --backend cuda --gpu 0 --runs 21`, clocks locked
(boost: SM 1695 MHz, memory 7601 MHz, P2) for the whole A/B and checked in every busy sample;
`M2b-cuda-perf-cycle_count-DD-smoke.json`):

| Case | Region | Original (ms) | dynG original scope (ms) | ratio | dynG resident scope (ms) | ratio | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| DD k = 4 | static kernel | 1.401 | 0.963 | 0.688 | 0.922 | 0.658 | <= 1.10 ok |
| DD k = 4 | static memcpy (reported) | 1.181 | 0.934 | 0.791 | 0.006 | - | - |
| DD k = 4 | static total (reported) | 2.600 | 2.854 | 1.098 | 0.962 | 0.370 | - |
| DD k = 4 | static end to end | 436.7 | 427.5 | 0.979 | | | <= 1.10 ok |
| DD 25K+25K k = 4 | update | 5.596 | 3.999 | 0.715 | 3.657 | 0.653 | <= 1.10 ok |
| DD 25K+25K k = 4 | update end to end | 497.5 | 479.6 | 0.964 | | | <= 1.10 ok |

The original's update reads 5.6 ms at these clocks (its RESULTS.md: 4.4 ms at the default,
DVFS-raised clocks; ADR 0018). The static total of dynG's original scope includes the first
allocation of the pinned scalar buffer of the workspace (about 0.9 ms, stage `cycle_count.reset`),
which a second call does not pay; the gated kernel region excludes it on both sides. Register
counts (`cycle_count_perf.py kernels`, `cuobjdump`) of the `uint32_t` instantiations equal the
original's (for example 19 for `count_edge_items_kernel<4>`, 26 for `count_roots_kernel<4>`, 22
for `count_owned_cycles_kernel<4>`, 96 bytes of stack each).

**OpenMP update, 56 threads** (11 rounds, `M2b-cuda-step-perf-openmp-cycle_count.json`):

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | Section 1.3 ratio |
|---|---|---:|---:|---:|---|---:|
| DD 25K+25K k = 4 | update | 26.3 | 15.1 | 0.574 | <= 1.05 ok | 0.615 |
| DD 25K+25K k = 4 | update end to end | 366.2 | 304.6 | 0.832 | <= 1.10 ok | 0.839 |
| GitHub 25K+25K k = 4 | update | 297.6 | 122.1 | 0.410 | <= 1.05 ok | 0.426 |
| GitHub 25K+25K k = 4 | update end to end | 3,266.6 | 1,757.0 | 0.538 | <= 1.10 ok | 0.565 |

No run flagged (> 2 foreign cores). The update got faster, as expected: the commit no longer
normalizes the batch a second time.

### 3.1 Sanitizers

compute-sanitizer on `dyng_cycle_count_cuda_tests` (GPU 1, `DYNG_TEST_SEEDS=2`, every test of
the suite): memcheck with leak checking 0 errors and 0 bytes leaked, racecheck 0 hazards,
synccheck 0 errors (`ci/gpu_local.sh` runs the same three).

## 4. The CUDA performance gates (step cuda-parity-perf)

M2b acceptance criterion 4, PLAN 6.4.3 and 8.6: the static kernel (k = 4 on DD, GitHub and
Twitch, k = 3 on COLLAB) and the update 25K + 25K (k = 4) on DD, GitHub, Twitch and COLLAB, plus
DD 50K + 50K and 100K + 100K, against CycleEnumeration-GPU@0a976ad's CUDA backend, in both scopes.

### 4.1 Methodology

| Item | Value |
|---|---|
| Port | dynG `94523c5` (`parity-cuda` preset, `-O3`; every performance and replay record says `94523c5`, no uncommitted change; the device memory record says `11a9b2f`, a harness-only commit on top of it). A complete campaign on `fe753d2` (before the device-memory change of section 4.8) read the same, within 0.02 in every ratio; it is superseded and not committed |
| Reference | the **unpatched** copy of CycleEnumeration-GPU `0a976ad` (`git archive`, built by `parity/build_reference.sh` with the original's documented Release build: `-DCYCLE_ENUM_ENABLE_CUDA=ON -DCYCLE_ENUM_ENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=86`, nvcc 13.1, GCC 12.2), `build/cycle-enum` |
| Harness | `parity/cycle_count_perf.py run --backend cuda` (`parity/perf_ab.py cycle_count run`), the regions of `[reference.cycle_enum_cuda]` in `parity/timed_regions/cycle_count.toml` |
| GPU | GPU 0 (RTX A5000, sm_86, driver 590.48.01); both programs see only it (`CUDA_VISIBLE_DEVICES=0`, `--cuda-device 0`), `CUDA_MODULE_LOADING=EAGER` |
| Clocks | locked for the whole A/B (ADR 0018, option B): `--lock-clocks boost` (SM 1695 MHz, memory 7601 MHz), the COLLAB update `--lock-clocks base` (SM 1170 MHz, memory 7601 MHz; section 4.5). Every busy GPU sample of every side in every accepted round was at the locked clocks |
| Lock and monitor | exclusive `perf.lock`; the machine monitor rejects and repeats a round with a foreign GPU process, a busy sample off the locked clocks, or more than 2 foreign CPU cores on either side |
| Order | per case one untimed round (page cache; the three histograms must be identical and equal to the golden), then rounds of A (original), B (dynG, original scope), B (dynG, resident scope) |
| Runs | 21 accepted rounds per case (11 for the COLLAB update, whose region is >= 10 ms: PLAN 8.6 asks for >= 5; and for the default-clock readings); medians compared |
| Timers | static: `kernel_ms` of both programs' `--report-timing` (CUDA events: building the work items, the counting kernel, the synchronization); update: `update_seconds` of the original (host clock around `update_histogram`) and dynG's `update_ms` (host clock around `cycle_count::update()`); end to end: the harness's clock around each process |
| Gates | >= 10 ms: <= 1.05x; < 10 ms (DD static and DD 25K/50K updates): <= 1.10x with >= 20 runs; end to end <= 1.10x |
| Correctness guard | in every timed round the three histograms were compared with each other (and, before the first round, with the golden of `cycle_count_cuda`); none differed |

The two scopes (PLAN 6.4.3; `dyng-compat-cycle-enum --scope`):

- **original scope**: what the original times. The count task's timed `compute()` is the graph's
  first (the upload is outside `kernel_ms` on both sides, inside `total_ms`); the update's G_t has
  no device copy (a clone), so the timed `update()` uploads it, as the original uploads G_t per
  call.
- **resident scope**: the graph is on the device before the timed call. For the count task the
  compat tool uploads it with an untimed count of the 2-cycles (about a millisecond; section
  4.4); for the update task it is resident after the prior, which both programs compute right
  before the update.

### 4.2 Gate table (clocks locked)

Ratio = dynG / original, medians. PLAN = the original's numbers quoted by PLAN 6.4.3 (its
RESULTS.md, default clocks). Records: `M2b-cuda-perf-cycle_count.json` (boost, eight cases),
`M2b-cuda-perf-cycle_count-repeat.json` (boost: COLLAB k = 3 and DD 25K+25K, repeated because
another user's GPU job ran on GPU 0 during the campaign and the monitor rejected more rounds
than `--runs`; the campaign's record marks both cases incomplete) and
`M2b-cuda-perf-cycle_count-collab-update-base.json` (base).

| Case | Region | PLAN (ms) | Original (ms) | dynG, original scope (ms) | ratio | dynG, resident scope (ms) | ratio | Gate | End to end (ratio) | Rounds / rejected |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| DD k = 4 | kernel | 1.44 | 1.396 | 0.967 | 0.692 | 0.948 | 0.679 | <= 1.10 ok / ok | 0.955 | 21 / 0 |
| GitHub k = 4 | kernel | 58.0 | 58.183 | 57.523 | 0.989 | 58.048 | 0.998 | <= 1.05 ok / ok | 0.909 | 21 / 5 |
| Twitch k = 4 | kernel | 38.1 | 39.083 | 38.997 | 0.998 | 38.222 | 0.978 | <= 1.05 ok / ok | 0.883 | 21 / 8 |
| COLLAB k = 3 (repeat) | kernel | 51.4 | 52.112 | 51.786 | 0.994 | 52.286 | 1.003 | <= 1.05 ok / ok | 0.931 | 21 / 17 |
| DD 25K+25K (repeat) | update | 4.4 | 5.609 | 3.272 | 0.583 | 2.918 | 0.520 | <= 1.10 ok / ok | 0.954 | 21 / 0 |
| DD 50K+50K | update | 6.0 | 7.092 | 4.794 | 0.676 | 4.553 | 0.642 | <= 1.10 ok / ok | 0.967 | 21 / 3 |
| DD 100K+100K | update | 9.6 | 10.867 | 8.058 | 0.742 | 7.510 | 0.691 | <= 1.05 ok / ok | 0.945 | 21 / 0 |
| GitHub 25K+25K | update | 14.3 | 15.013 | 11.675 | 0.778 | 8.499 | 0.566 | <= 1.05 ok / ok | 0.922 | 21 / 7 |
| Twitch 25K+25K | update | 27.3 | 25.203 | 20.663 | 0.820 | 8.636 | 0.343 | <= 1.05 ok / ok | 0.920 | 21 / 5 |
| COLLAB 25K+25K (base lock) | update | 197 | 218.1 | 215.3 | 0.987 | 205.7 | 0.943 | <= 1.05 ok / ok | 0.986 | 11 / 0 |

**Every gated region is within its gate in both scopes: 20 / 20 readings (0.34-1.003x), and all
ten end-to-end readings (0.88-0.99x).** The first campaign's partial reading of COLLAB k = 3 (15
rounds before it was stopped) was 1.005x / 1.007x, the same as the repeat. The DD update reads
above PLAN's 4.4 ms on the original's side because PLAN quotes default, DVFS-raised clocks
(ADR 0018); at the same locked clocks both programs ran the same kernels.

Where the ratios come from:

- **Static kernels, >= 38 ms (GitHub, Twitch, COLLAB): equal** (0.978-1.003x). The counting
  kernels are the original's, with the same registers, stack and occupancy (section 4.6).
- **DD static kernel: 0.69x.** DD's kernel is short (about 0.5 ms of counting); the original's
  `kernel_ms` region also allocates and frees its item arrays with `cudaMalloc` / `cudaFree`
  (`DeviceBuffer`) and reads its two scalars back with pageable `cudaMemcpy`, which dynG leases
  from its workspace pool and reads through pinned memory.
- **Updates: 0.58-0.99x in the original scope, 0.34-0.94x resident.** Step 0 on the host is
  cheaper (section 4.4), and in the resident scope G_t is not uploaded again (the upload is
  11.9 ms of Twitch's 20.7 ms). Stage medians of the update (ms; `cycle_count.count_minus`
  includes the upload of G_t in the original scope):

| Case | Scope | normalize (Step 0) | count_minus | commit (device apply) | count_plus | update |
|---|---|---:|---:|---:|---:|---:|
| DD 25K+25K | original | 1.33 | 1.09 | 0.44 | 0.23 | 3.27 |
| DD 25K+25K | resident | 1.89 | 0.21 | 0.45 | 0.23 | 2.91 |
| DD 100K+100K | original | 4.74 | 1.28 | 0.73 | 0.76 | 8.05 |
| DD 100K+100K | resident | 5.06 | 0.49 | 0.73 | 0.75 | 7.51 |
| GitHub 25K+25K | original | 3.11 | 3.75 | 2.76 | 1.89 | 11.67 |
| GitHub 25K+25K | resident | 3.49 | 0.24 | 2.75 | 1.85 | 8.49 |
| Twitch 25K+25K | original | 3.74 | 12.33 | 3.45 | 0.93 | 20.66 |
| Twitch 25K+25K | resident | 3.77 | 0.34 | 3.45 | 0.91 | 8.63 |
| COLLAB 25K+25K (base) | original | 4.38 | 11.83 | 76.28 | 122.66 | 215.25 |
| COLLAB 25K+25K (base) | resident | 4.43 | 0.36 | 76.34 | 124.34 | 205.68 |

COLLAB's update is bound by its G_{t+1} build (the device merge of the batch into 24.6 million
edges: `commit` 76.3 ms) and the insert phase (123 ms); both are the original's kernels, which the
original runs as well, so the ratio is 0.99 in the original scope and the resident scope saves
only the upload.

Reported regions (not gated; from the same records):

| Case | Region | Original (ms) | dynG (ms) | ratio |
|---|---|---:|---:|---:|
| DD k = 4 | static_memcpy (original scope) | 1.182 | 0.957 | 0.810 |
| DD k = 4 | static_total (original scope) | 2.601 | 2.862 | 1.101 |
| DD k = 4 | static_total (resident scope) | 2.601 | 0.989 | 0.380 |
| GitHub k = 4 | static_memcpy (original scope) | 3.523 | 3.459 | 0.982 |
| GitHub k = 4 | static_total (original scope) | 61.768 | 61.937 | 1.003 |
| GitHub k = 4 | static_total (resident scope) | 61.768 | 58.094 | 0.941 |
| Twitch k = 4 | static_memcpy (original scope) | 12.021 | 11.899 | 0.990 |
| Twitch k = 4 | static_total (original scope) | 51.249 | 51.724 | 1.009 |
| Twitch k = 4 | static_total (resident scope) | 51.249 | 38.268 | 0.747 |
| COLLAB k = 3 | static_memcpy (original scope) | 11.532 | 11.692 | 1.014 |
| COLLAB k = 3 | static_total (original scope) | 63.915 | 64.461 | 1.009 |
| COLLAB k = 3 | static_total (resident scope) | 63.915 | 52.350 | 0.819 |

DD's `static_total` in the original scope (1.10x, not a gated region) includes the first
allocation of the workspace's pinned scalar buffer (`cudaHostAlloc`, about 0.85 ms in the stage
`cycle_count.reset`), which a second call through the same resources does not pay; without it the
region would read about 0.77x.

### 4.3 Clock record

Per side, over the accepted rounds (`gpu_summary` in the records): every **busy** sample was at
the locked clocks (the rejection rule). Samples that were not busy (the 50 ms sampler saw no
utilization) went below the lock in a few cases, on dynG's side of the long static kernels
(GitHub 1455-1470 MHz, COLLAB k = 3 1545 MHz) and of the GitHub update's resident scope
(1395 MHz), and in the superseded `fe753d2` campaign on both sides of the same cases; the original's
side ran fewer rounds with busy samples at all (7-16 of 21, against 12-18 for dynG), so a dip in
its kernel window is less likely to be seen. This is the GPU's power management under sustained
load (section 4.5). Rounds with a busy sample off the lock were repeated (GitHub static 5, Twitch
static 8, COLLAB static 17 in the repeat, DD 50K 3, GitHub update 7, Twitch update 5; the COLLAB
k = 3 and DD 25K+25K cases of the first campaign were also hit by another user's process on GPU 0
and more than 2 foreign CPU cores).

### 4.4 Two misses found by the first gate campaign, and their fixes

The first full campaign (port `bba3017`, clocks locked, 21 rounds; the harness stopped at the
COLLAB update before writing its record, see section 4.5, so these medians come from its run log)
exceeded the gate in two places:

| Case | Region | Original (ms) | dynG original scope | dynG resident scope |
|---|---|---:|---:|---:|
| GitHub k = 4 | kernel | 58.515 | 58.128 (0.993) | 62.565 (**1.069**) |
| Twitch k = 4 | kernel | 39.831 | 38.461 (0.966) | 43.587 (**1.094**) |
| COLLAB k = 3 | kernel | 52.087 | 51.876 (0.996) | 55.130 (**1.058**) |
| DD 100K+100K | update | 10.898 | 12.330 (**1.131**) | 11.853 (**1.088**) |

**The resident static kernel.** In the resident scope the compat tool made the graph resident
with a full untimed count right before the timed one. Profiled with Nsight Systems (default
clocks), the same kernel took 56.0 ms in the first call and 61.3 ms in the second; a scratch driver
(`compute()` five times on one graph, clocks locked at boost) gave 58.5 ms for the first call and
61.0-62.5 ms for every call that followed another at once, and 54-60 ms with a one-second
pause between calls (at default clocks: 58.2, then 59.4-61.3; with two-second pauses 57.9-59.6).
The kernel, its launch and its memory are the same; the GPU is in the state of a long busy
period, which slows the next kernel even at a locked clock (the driver's power management: the
GPU's throttle reasons show SW power capping during these runs). The original's process never
has that state before its kernel: its GPU is idle while it reads the file. So the resident scope
now uploads with an untimed count of the 2-cycles (root items, about a millisecond) instead
(`c0c100e`), and the resident kernel reads 0.98-1.00x like the original scope. Nothing changed
in the library.

**DD 100K+100K.** The stage breakdown showed Step 0 (`cycle_count.normalize`, on the host, as the
original's `prepare_batch`) at 8.8 ms of the 12.2 ms update. `sort_and_dedup` used
`std::stable_sort` (to keep the first of equal pairs in batch order), 1.5x the cost of the
original's `std::sort` on 100K changes and with a merge buffer. It now sorts with `std::sort` by
(source, target, position in the batch), which gives the same lists because the positions are
unique, and checks first whether a list is already in that order (generated batches are): Step 0
8.8 -> 4.7 ms, the update 12.2 -> 8.0 ms (0.74x; `8e4d453`). The histograms and every normalized
list are unchanged (section 5); the CPU backends use the same Step 0 and get the same saving.

### 4.5 The COLLAB update and the power cap

The COLLAB update's prior is the k = 4 static count of COLLAB (199.7 billion cycles, 6.5 s of
kernels on both sides). At the boost lock the GPU cannot hold 1695 MHz through it: the driver
lowers the SM clock to 1350-1680 MHz under its power cap (230 W, the default) in every process of
both programs. The first campaign rejected 22 consecutive rounds and stopped (the harness then
exited without a record; it now records such a case as incomplete and continues, `778ebf5`). At
the base lock (1170 MHz) the clocks hold for the whole process on both sides, so this case is
gated at base: 11 rounds, none rejected, 0.987x (original scope) and 0.943x (resident). ADR 0018
records the rule (a case whose GPU cannot hold the boost lock is read at the base lock, applied
equally to both programs). At default clocks the case reads 0.989x / 0.937x (section 4.7).

### 4.6 Kernels: registers, stack and occupancy

`parity/cycle_count_perf.py kernels` (`cuobjdump --dump-resource-usage` of the original's
`libcycle_enum_cuda.a` and `libcycle_enum_dynamic.a` and of dynG's `libdyng.so`; the theoretical
occupancy on sm_86 at the launch block size: 48 warps and 16 blocks per SM, 65,536 registers in
units of 256 per warp; no kernel uses shared memory). Record: `M2b-cuda-kernels-cycle_count.json`.

| Kernel (cap = compile-time path capacity) | Block | Registers original / dynG | Stack (bytes) original / dynG | Occupancy |
|---|---:|---|---|---|
| `count_roots<4, 8, 16, 32, 64>` (naive) | 128 | 26, 34, 40, 40, 40 / same | 96, 176, 336, 656, 1296 / same | 100 % |
| `count_roots_queue<4..64>` | 128 | 20, 22, 30, 40, 40 / same | as above / same | 100 % |
| `count_edge_items<4..64>` | 128 | 19, 20, 30, 40, 40 / same | as above / same | 100 % |
| `count_two_hop_items<4..64>` | 128 | 20, 22, 29, 40, 40 / same | as above / same | 100 % |
| `count_owned_cycles<4..64>` (update) | 128 | 22, 22, 30, 36, 36 / same | as above / same | 100 % |
| `forward_rows`, `fill_edge_items`, `target_degree` | 256 | 16, 18, 14 / same | 0 / 0 | 100 % |
| `mark_owners`, `item_counts`, `change_rows`, `next_degree`, `build_next_rows` | 128 | 12, 14, 10, 18, 22 / same | 0 / 0 | 100 % |

**33 / 33 kernels of the original have a dynG counterpart (the `uint32_t`-offset instantiation)
with the same registers, stack, shared memory and occupancy.** Every kernel is limited by the
warp limit (12 blocks of 128 threads), not by registers, so the persistent work-queue grids are
the same size on both sides (12 x 64 blocks). The `uint64_t`-offset instantiations (graphs with
`int64_t` edge offsets, not the original's) use 21-40 registers and 128-1808 bytes of stack, also
at 100 % occupancy.

### 4.7 Default clocks (recorded, ungated: ADR 0018)

The same A/B without a clock lock, 11 rounds per case (`M2b-cuda-perf-cycle_count-default-clocks.json`):

| Case | Region | Original (ms) | dynG original scope (ms) | ratio | dynG resident scope (ms) | ratio | End to end |
|---|---|---:|---:|---:|---:|---:|---:|
| DD k = 4 | kernel | 1.392 | 0.975 | 0.701 | 0.959 | 0.689 | 0.953 |
| GitHub k = 4 | kernel | 59.476 | 57.634 | 0.969 | 57.782 | 0.972 | 0.917 |
| Twitch k = 4 | kernel | 38.681 | 38.376 | 0.992 | 38.010 | 0.983 | 0.878 |
| COLLAB k = 3 | kernel | 51.306 | 51.644 | 1.007 | 51.539 | 1.005 | 0.912 |
| DD 25K+25K | update | 5.528 | 3.224 | 0.583 | 2.926 | 0.529 | 0.929 |
| DD 50K+50K | update | 7.013 | 4.797 | 0.684 | 4.486 | 0.640 | 0.956 |
| DD 100K+100K | update | 10.710 | 7.944 | 0.742 | 7.451 | 0.696 | 0.942 |
| GitHub 25K+25K | update | 14.753 | 11.595 | 0.786 | 8.425 | 0.571 | 0.934 |
| Twitch 25K+25K | update | 25.295 | 20.294 | 0.802 | 8.556 | 0.338 | 0.921 |
| COLLAB 25K+25K | update | 199.4 | 197.2 | 0.989 | 186.9 | 0.937 | 0.977 |

At default clocks every reading is also within the gate's bounds (0.34-1.007x; end to end
0.88-0.98x). The original's COLLAB update reads 199.4 ms (PLAN: 197 ms); its DD kernel 1.39 ms
(PLAN: 1.44 ms). Seven rounds each of the Twitch and COLLAB updates were repeated (another GPU
process or foreign CPU load).

### 4.8 Device memory (PLAN 8.6)

The peak of live device allocations per process, from Nsight Systems' CUDA memory trace
(`parity/cycle_count_perf.py memory`: `nsys profile --cuda-memory-usage=true`, memory kind Device,
`cudaMalloc` and `cudaMallocAsync`; GPU 1; `M2b-cuda-memory-cycle_count.json`, taken at `11a9b2f`,
which adds the command to the harness; the library is that of `94523c5`). Both scopes of dynG give
the same peak.

| Case | Original (MiB) | dynG (MiB) | ratio | dynG before `94523c5` (MiB) |
|---|---:|---:|---:|---:|
| DD k = 4 | 18.0 | 18.0 | 1.000 | |
| GitHub k = 4 | 67.7 | 67.7 | 1.000 | |
| Twitch k = 4 | 225.5 | 225.5 | 1.000 | |
| COLLAB k = 3 | 193.2 | 193.2 | 1.000 | |
| DD 25K+25K | 32.9 | 32.9 | 1.000 | |
| DD 50K+50K | 33.6 | 33.6 | 1.000 | |
| DD 100K+100K | 35.2 | 35.2 | 1.000 | 51.9 (1.47x) |
| GitHub 25K+25K | 119.5 | 119.5 | 1.000 | 181.6 (1.52x) |
| Twitch 25K+25K | 408.6 | 408.6 | 1.000 | 619.7 (1.52x) |
| COLLAB 25K+25K | 382.8 | 383.0 | 1.001 | 762.2 (1.99x) |

**Within PLAN 8.6's <= 1.05x on every case after `94523c5`**, which this step added after
measuring (M2b's criteria do not list device memory, and M1b did not measure it). Before it, the
update kept the prior's static-count work items in the workspace (on COLLAB k = 4 about 300 MB of
two-hop items) and marked the deleted positions of G_t twice (cycle_count's delete phase and the
device apply, one m-int array each); the last column is the same measurement of `fe753d2`, built
in a temporary worktree. Now the update returns the work items when it begins and the marks are
computed once into the normalized batch (ADR 0020, point 6).

What `nvidia-smi` shows per process is larger on dynG's side by a fixed amount: its
stream-ordered pool reserves in 32 MB granules (the largest pool size nsys reported is 9-65 MB
above the live peak), and its process loads every kernel of the library at `warm_up()` (about
34 MB more than the original's on a three-edge graph: 240 against 206 MiB).

### 4.9 Reproducing

```bash
source scripts/dev_env.sh
cmake --build --preset parity-cuda
parity/build_reference.sh CycleEnumeration-GPU          # the patched (exporter) and unpatched copies
E=build/parity-cuda/tools/compat/dyng-compat-cycle-enum
# the gate at locked clocks (the default case list is M2b's; perf_ab.py takes the exclusive lock)
parity/perf_ab.py cycle_count run --backend cuda --gpu 0 --runs 21 --exe $E \
    --cases DD_k4,github_k4,twitch_k4,collab_k3,DD_k4_25000_25000_s1,github_k4_25000_25000_s1,twitch_k4_25000_25000_s1,DD_k4_50000_50000_s1,DD_k4_100000_100000_s1 \
    --json parity/results/M2b-cuda-perf-cycle_count.json
parity/perf_ab.py cycle_count run --backend cuda --gpu 0 --runs 11 --lock-clocks base --exe $E \
    --cases collab_k4_25000_25000_s1 --json parity/results/M2b-cuda-perf-cycle_count-collab-update-base.json
parity/perf_ab.py cycle_count run --backend cuda --gpu 0 --runs 11 --lock-clocks none --exe $E \
    --json parity/results/M2b-cuda-perf-cycle_count-default-clocks.json
# the two cases hit by another user's GPU job in the first run, repeated
parity/perf_ab.py cycle_count run --backend cuda --gpu 0 --runs 21 --exe $E \
    --cases collab_k3,DD_k4_25000_25000_s1 --json parity/results/M2b-cuda-perf-cycle_count-repeat.json
python3 parity/cycle_count_perf.py kernels --library build/parity-cuda/cpp/libdyng.so \
    --json parity/results/M2b-cuda-kernels-cycle_count.json
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 python3 parity/cycle_count_perf.py memory --exe $E \
    --gpu 1 --json parity/results/M2b-cuda-memory-cycle_count.json
# parity on the final code
CUDA_VISIBLE_DEVICES=1 flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py cycle_count \
    --exe $E --set cycle_count_cuda --configs cuda,cuda:resident,cuda:int64 \
    --json parity/results/M2b-final-cycle_count-cuda-set-parity-cuda-preset.json
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py cycle_count \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --set cycle_count_cuda --configs openmp:56 \
    --json parity/results/M2b-final-cycle_count-cuda-set-openmp-parity-preset.json
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py cycle_count \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --full \
    --json parity/results/M2b-final-cycle_count-parity-preset.json
```

## 5. Parity at the final code (step cuda-parity-perf)

Three changes of this step touch the update on every backend (Step 0's sort, `8e4d453`; the shared
deletion marks and the released work items, `94523c5`) or the compat tool (`c0c100e`), so the
corpora were replayed again on the final code `94523c5` (clean tree; `parity` and `parity-cuda`
presets rebuilt at that commit):

| Corpus | Configurations | Result | Record |
|---|---|---|---|
| `cycle_count_cuda`, 24 cases (the original's CUDA backend) | cuda, cuda:resident, cuda:int64 (GPU 1) | **72 / 72 replays equal** (histograms and generated batches) | `M2b-final-cycle_count-cuda-set-parity-cuda-preset.json` |
| `cycle_count_cuda` | openmp:56 (COLLAB's update is replayed on cuda only) | **23 / 23 equal** | `M2b-final-cycle_count-cuda-set-openmp-parity-preset.json` |
| `cycle_count`, 24 cases (the M2a corpus of the original's OpenMP backend) | sequential, openmp:4, openmp:56 (`--full`) | **72 / 72 replays equal** | `M2b-final-cycle_count-parity-preset.json` |
| every timed round of section 4 | cuda, both scopes | the three histograms identical in every round and equal to the golden before the first | the performance records |

`ci/check.sh --parity` (all steps, including `ctest -L parity` 4 / 4), `ci/gpu_local.sh` (all ten
steps: 155 gpu tests, 414 cpu tests in the CUDA build, the CUDA corpora of sssp and cycle_count,
memcheck, synccheck and racecheck) and `ci/docs.sh` passed on `98a0dd3` (this step's records and
documents on top of `94523c5`) in the working tree; the fresh-clone runs of acceptance criterion 1
are the milestone's close-out.

## 6. Certificate

M2b acceptance criteria 3 and 4 against CycleEnumeration-GPU@0a976ad's CUDA backend:

| Criterion | Evidence | Verdict |
|---|---|---|
| 3: histograms bit-identical on the fixtures, DD k = 3..7, GitHub k = 4, Twitch k = 4, COLLAB k = 3 and the seed-1 update deltas (1K, 25K, 50K, 100K on DD and GitHub; 25K on Twitch and COLLAB) | the golden set `cycle_count_cuda` (24 cases, written from `cycle-enum --backend cuda`, each checked against the original's CPU output) replayed on cuda, cuda:resident and cuda:int64: 72 / 72 (section 5); the CUDA fixtures in `dyng_cycle_count_cuda_tests` | **pass** |
| 3: cross-backend equality (CUDA = OpenMP = sequential) | the CUDA set on openmp:56 (23 / 23), the CPU corpus on sequential and OpenMP (72 / 72), the randomized chains of `cycle_count_cuda_test.cpp` | **pass** |
| 3: `max_length <= 64` on cuda with a clear error | `CycleCountCuda.EveryBoundUpToSixtyFourMatchesTheSequentialCount`, `BoundsBeyondSixtyFourAreRejectedWithAClearError` (`invalid_argument_error` before anything changes) and `UpdatesAtEveryCapacityBoundaryMatchTheSequentialUpdate`, the CLI cases with bounds 64 and 65 | **pass** |
| 3: compute-sanitizer clean | memcheck (0 errors, 0 leaks), racecheck (0 hazards), synccheck (0 errors) on `dyng_cycle_count_cuda_tests` (`ci/gpu_local.sh`, section 5) | **pass** |
| 4: CUDA gates, both scopes, clocks locked, >= 20 runs with CUDA events for regions < 10 ms | section 4.2: 20 / 20 gated readings within the gate (0.34-1.003x), end to end 0.88-0.99x | **pass** (the COLLAB update at the base lock, section 4.5) |
| 4: default-clock readings; registers and occupancy | sections 4.7 and 4.6 (33 / 33 kernels equal to the original's) | recorded |
| PLAN 8.6: device memory <= 1.05x | section 4.8: the peak of live allocations 1.000-1.001x on all ten cases | **pass** (not an M2b criterion; fixed in this step) |
