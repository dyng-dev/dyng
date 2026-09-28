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
