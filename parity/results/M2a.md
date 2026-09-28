# M2a parity certificate: `cycle_count` (CPU) against CycleEnumeration-GPU@0a976ad

Date: 2026-09-28 (updated after the review fixes of M2a, port `0679ed1`; the first version of
this certificate measured `1148d15`). Format: PLAN Section 8.3 ("Parity certificate"). The
machine-readable records are next to this file:

| File | Content |
|---|---|
| `M2a-cycle_count-parity-preset.json` | golden replay, `parity` preset (Release, `-O3`): 24 cases x sequential, OpenMP 4 and 56 threads |
| `M2a-perf-openmp-cycle_count.json` | OpenMP-56 performance A/B, 11 runs per case (the gate of acceptance criterion 4) |
| `M2a-perf-openmp-cycle_count-DD-31runs.json` | the two noisiest DD regions again with 31 runs |
| `M2a-perf-openmp-cycle_count-collab.json` | COLLAB k = 3 static, 5 runs |
| `M2a-experiment-*.json` | the isolation experiments of Section 3.4 (not gates): the original with only a dense histogram, with stage timers, both; the port before the review fixes (`a975865`) |

## 1. What was compared

| Item | Value |
|---|---|
| Original (reference) | CycleEnumeration-GPU `0a976adfa801a712135bf1adb51a228f353a0751` (baseline tag `baseline-2026-09` = `da2067d62c2ce8f9234908089fa15bc53b5979c0`), built from a `git archive` copy by `parity/build_reference.sh`: CMake Release (`-O3 -DNDEBUG`), `-DCYCLE_ENUM_ENABLE_OPENMP=ON -DCYCLE_ENUM_ENABLE_CUDA=ON`, sm_86, as its `docs/RESULTS.md`. Unpatched copy for performance; the patched copy adds only `parity_export/bin/export_cycle_enum` (no original file changes) and exported the goldens with the original's own `build/cycle-enum` |
| Port | dynG `0679ed1` (`cycle_count` sequential and OpenMP backends, the graph under `graph_properties::cycle_enum_compatible()`, `io::read_edge_list`, `generators::legacy::cycle_enum_batch()`), driven by `tools/compat/dyng-compat-cycle-enum`; the review fixes of M2a included (explicit-stack searches, counters sized by the cycles found, the flat ownership index) |
| Toolchain | GCC 12.2.0 (Debian 12.2.0-14+deb12u1) for both; dynG with the `parity` preset (`-O3`, OpenMP on) |
| Host | Intel Xeon Gold 6258R (28 cores, 56 hardware threads, one NUMA node), 124 GB, Debian 12, Linux 6.1 |
| Inputs | TUDataset `DD_A.txt`, `github_stargazers_A.txt`, `twitch_egos_A.txt`, `COLLAB_A.txt` under `$DYNG_SCRATCH/datasets/cycle` (SHA-256 in every `case.json` and in the performance JSON) |
| Goldens | `$DYNG_SCRATCH/goldens/cycle_count`: 24 cases, 90 files, 10.5 MB; `MANIFEST.sha256` = `e40fa03ba3300376866362b9e7e737799a0ae99ac9f08f89c660d47b5fde91eb`; one SHA-256 per case in the `[sets.cycle_count]` section of `parity/goldens.toml`. `compare.py` re-hashed every file before comparing |
| Tolerance | none: byte equality of the histogram CSV (every length and the total) and of the generated batch (PLAN Section 8.3). Work-item counts are not compared |

### The golden corpus (`parity/export_goldens.py cycle_count`)

| Group | Cases | Produced by the original (0a976ad) |
|---|---:|---|
| `count` | 10 | `cycle-enum --backend openmp --openmp-threads 56 --max-cycle-length k`: DD k = 3..7, GitHub k = 3, 4, Twitch k = 3, 4, COLLAB k = 3 |
| `update` | 14 | `cycle-enum --task update --deletes d --inserts i --batch-seed 1 --compare-recompute`, k = 4: 1K+1K, 25K+25K, 50K+50K on DD, GitHub and Twitch (9); a DD sweep: locality windows 1000 (1K+1K; a window of 1000 has too few edges for 25K deletions, the original rejects it), 10000 and 100000 (25K+25K), and k = 3, 5 (25K+25K) |

Per `count` case: `histogram.csv` (standard output). Per `update` case: `histogram.csv` (the updated
histogram), `prior.csv` (the static histogram of the same graph and k), `delta.csv` (updated -
prior per length and in total) and `batch.txt` (the generated batch, `export_cycle_enum
generate`: `- u v` per deletion, then `+ u v`). `case.json` holds the command, the input's
SHA-256, the requested and generated batch sizes and the cross-checks.

**Proof that the export is honest** (PLAN Section 8.3):

- The original checked against itself before a case was written: the sequential backend printed
  the same bytes as the OpenMP backend on the 23 cases other than COLLAB (sequential up to 74 s
  per case); every update's `--compare-recompute` printed `match=yes`; the histograms equal
  `cpp/tests/data/cycle_enum/datasets_counts.txt` (made from the original's functions through the
  exporter, a second path; 19 cases have an entry there); the batch files have the sizes the CLI
  reported; the ten static totals are the plan's numbers (PLAN 6.4.3).
- `--twice`: the corpus was exported a second time from a FRESH scratch copy (a new `git archive`
  built from scratch in a temporary work area) and the manifest was identical
  (`e40fa03b...`).

| Case | Total (all lengths) | Case | Total after the update |
|---|---:|---|---:|
| DD k = 3 | 2,020,240 | DD 1K+1K | 4,388,002 |
| DD k = 4 | 4,396,674 | DD 25K+25K | 4,182,398 |
| DD k = 5 | 9,476,048 | DD 50K+50K | 3,975,602 |
| DD k = 6 | 21,485,606 | GitHub 1K+1K | 75,819,437 |
| DD k = 7 | 54,966,172 | GitHub 25K+25K | 74,644,109 |
| GitHub k = 3 | 7,112,157 | GitHub 50K+50K | 73,432,075 |
| GitHub k = 4 | 75,872,845 | Twitch 1K+1K | 389,293,913 |
| Twitch k = 3 | 48,891,585 | Twitch 25K+25K | 387,651,174 |
| Twitch k = 4 | 389,362,369 | Twitch 50K+50K | 385,936,907 |
| COLLAB k = 3 | 1,257,799,573 | | |

## 2. Result matrix (dynG against the goldens)

`parity/compare.py cycle_count --exe build/parity/tools/compat/dyng-compat-cycle-enum --full`
(the CTest `parity.cycle_count.cycle_enum_0a976ad` runs the same without `--full`, that is without
COLLAB's sequential replay). Each cell: byte-identical histogram CSV and, for updates, a
byte-identical generated batch and equal batch sizes.

| Group | Cases | sequential | openmp:4 | openmp:56 |
|---|---:|---:|---:|---:|
| count | 10 | 10/10 | 10/10 | 10/10 |
| update | 14 | 14/14 | 14/14 | 14/14 |
| **all** | 24 | 24/24 | 24/24 | 24/24 |

**ALL EQUAL: 72 replays** at `0679ed1` (the whole replay took 11 min; at `1148d15` it took
18 min, the sequential COLLAB k = 3 alone 444 s: the bounded sequential count now shares the
OpenMP counter's explicit-stack search). The replay was run again after every change of the
review fixes that touched a search (`bce07a6`: 72 of 72; `0679ed1`: 72 of 72). The deltas follow: the updated histograms equal the goldens and the port's prior is its
static count, which equals `count/` for k = 4.

Other parity evidence of M2a (in-repo, CTest; see the retrospective): the 80 random fixture cases
(static, updated, k = 2..7 and unbounded, both backends), 16 fixture-graph counts and 7 updates,
19 CLI runs of the original (`compat_cycle_enum.cli`), the batch-generator identity on 22 fixture
cases and the seed-1 dataset batches, the subset-DP oracle / brute force / edge-set recount
randomized suites, and the mutation tests `cycle_count.mutation.{control,double_count_5,
weak_ownership}` (the two recorded mutations make the suite fail; the control copy passes).

## 3. Performance (PLAN Section 8.6; acceptance criterion 4)

### 3.1 Methodology

- `parity/perf_ab.py cycle_count run` (`parity/cycle_count_perf.py`) under `flock
  $DYNG_SCRATCH/perf.lock` (exclusive: the builds, tests and measurements of the other agents on
  the machine take the same lock, shared or exclusive).
- Original: the UNPATCHED copy's `build/cycle-enum` (rebuilt idempotently and verified against
  the archive before timing). Port: `build/parity/tools/compat/dyng-compat-cycle-enum` at
  `0679ed1`, `parity` preset (checked from its CMakeCache).
- Both sides: `--backend openmp --openmp-threads 56` (the original's RESULTS.md), the same
  environment with no `OMP_*` / `GOMP_*` variables (the libgomp defaults, as RESULTS.md), the
  same input file, standard output captured by the harness on both sides.
- Per case: one untimed round per side first (page cache; the two histograms must be byte-identical
  and equal to the golden), then A/B/A/B (original first) for 11 rounds (31 for the DD re-run,
  5 for COLLAB); the histograms are compared in every round. Medians are compared.
- Regions (`parity/timed_regions/cycle_count.toml`): **static_end_to_end** = the process wall time
  of `--task count` (the only CPU timing the original has; RESULTS.md's "OpenMP (56 threads), end
  to end"), gated as the paper-timed region; **update** = the original's `update_seconds`
  (steady_clock around `update_histogram`) against the port's `update_ms` (steady_clock around
  `cycle_count::update()`; the profiler stage `cycle_count.update` agrees within 0.1 ms);
  **update_end_to_end** = the process wall time of `--task update` (gate 1.10). Every gated region
  is >= 10 ms on the original's side, so every compute gate is <= 1.05x. The "original scope" is
  the only scope on the CPU (the resident-graph scope of PLAN 6.4.3 concerns the CUDA backend,
  M2b). **static_count** and **static_read** (never gates) compare the port's stages with the
  instrumented experiment copy of Section 3.4.
- **Contamination monitor** (PLAN 8.6; `parity/contamination.py`): for every timed process, the
  busy CPU time of the whole machine (`/proc/stat`) minus the CPU time of the harness and the
  processes it waited for (`getrusage`), divided by the wall time: the cores that foreign work kept
  busy while the process ran. A run above 2 foreign cores is flagged (recorded, not dropped). The
  load average is recorded as well but cannot separate foreign load from the measured 56-thread
  process itself (it reached 17-44 here with almost no foreign work). The records keep, per case
  and side, the median, the maximum, the flagged count and the foreign cores of every run
  (`foreign_cores_per_run`; the harness wrote the busy and own components per run as well, and
  those were dropped from the committed records to keep them small, as the harness now does).
- Spread = (max - min) / median; above 10 % is flagged (PLAN 8.6: flagged, not failed).

### 3.2 Gate table (11 runs; `M2a-perf-openmp-cycle_count.json`)

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| DD k = 3 | static_end_to_end | 256.7 | 238.7 | 0.930 | <= 1.05 ok | 13 % / 11 % (noisy) |
| DD k = 4 | static_end_to_end | 251.2 | 229.6 | 0.914 | <= 1.05 ok | 9 % / 12 % (noisy) |
| DD k = 5 | static_end_to_end | 293.3 | 262.9 | 0.896 | <= 1.05 ok | 8 % / 9 % |
| DD k = 6 | static_end_to_end | 514.5 | 373.1 | 0.725 | <= 1.05 ok | 4 % / 8 % |
| DD k = 7 | static_end_to_end | 1,488.7 | 892.7 | 0.600 | <= 1.05 ok | 2 % / 3 % |
| GitHub k = 3 | static_end_to_end | 719.2 | 562.4 | 0.782 | <= 1.05 ok | 7 % / 3 % |
| GitHub k = 4 | static_end_to_end | 2,806.5 | 1,472.0 | 0.525 | <= 1.05 ok | 2 % / 3 % |
| Twitch k = 3 | static_end_to_end | 1,812.5 | 1,537.2 | 0.848 | <= 1.05 ok | 4 % / 3 % |
| Twitch k = 4 | static_end_to_end | 3,608.7 | 2,474.1 | 0.686 | <= 1.05 ok | 1 % / 1 % |
| DD 25K+25K k = 4 | update | 26.1 | 16.9 | 0.646 | <= 1.05 ok | 79 % / 109 % (noisy) |
| GitHub 25K+25K k = 4 | update | 294.3 | 130.7 | 0.444 | <= 1.05 ok | 2 % / 8 % |
| Twitch 25K+25K k = 4 | update | 157.8 | 76.1 | 0.483 | <= 1.05 ok | 7 % / 17 % (noisy) |
| DD 25K+25K k = 4 | update_end_to_end | 368.1 | 308.5 | 0.838 | <= 1.10 ok | 12 % / 10 % (noisy) |
| GitHub 25K+25K k = 4 | update_end_to_end | 3,283.7 | 1,769.2 | 0.539 | <= 1.10 ok | 1 % / 2 % |
| Twitch 25K+25K k = 4 | update_end_to_end | 4,576.7 | 3,329.1 | 0.727 | <= 1.10 ok | 2 % / 2 % |

**Every gate is met.** Contamination: the median foreign load per case was 0.00-0.92 cores (of
56), the maximum 3.54; one of the 264 timed processes was flagged (an original run of GitHub k =
3; its case's verdict does not depend on it: 0.782). The noisy DD regions (a 25 ms update with
single outliers of 40-60 ms) were measured again with 31 runs
(`M2a-perf-openmp-cycle_count-DD-31runs.json`): DD k = 3 static 254.4 / 246.8 ms (0.970), DD
update 26.3 / 16.6 ms (0.631), DD update end to end 361.6 / 307.8 ms (0.851), no run flagged;
same verdicts.

| Case | Foreign cores, original: median / max | dynG: median / max | Flagged runs (> 2 cores) |
|---|---|---|---|
| DD k = 3 | 0.00 / 0.19 | 0.06 / 0.57 | 0 / 0 of 11 |
| DD k = 4 | 0.39 / 1.32 | 0.00 / 1.41 | 0 / 0 of 11 |
| DD k = 5 | 0.00 / 1.01 | 0.20 / 0.90 | 0 / 0 of 11 |
| DD k = 6 | 0.07 / 0.32 | 0.14 / 0.70 | 0 / 0 of 11 |
| DD k = 7 | 0.11 / 0.24 | 0.11 / 0.60 | 0 / 0 of 11 |
| GitHub k = 3 | 0.92 / 3.54 | 0.77 / 1.29 | 1 / 0 of 11 |
| GitHub k = 4 | 0.43 / 1.12 | 0.40 / 0.91 | 0 / 0 of 11 |
| Twitch k = 3 | 0.03 / 0.17 | 0.07 / 0.25 | 0 / 0 of 11 |
| Twitch k = 4 | 0.07 / 0.15 | 0.01 / 0.10 | 0 / 0 of 11 |
| DD 25K+25K | 0.43 / 0.55 | 0.08 / 0.57 | 0 / 0 of 11 |
| GitHub 25K+25K | 0.00 / 0.10 | 0.07 / 0.18 | 0 / 0 of 11 |
| Twitch 25K+25K | 0.05 / 0.16 | 0.04 / 0.19 | 0 / 0 of 11 |

The first certificate (port `1148d15`, before the review fixes) measured the same gates at
0.64-0.96 (static) and 0.76-0.96 (update); its load average was the only record of the machine's
state then (5.6-44).

### 3.3 COLLAB k = 3 (5 runs; `M2a-perf-openmp-cycle_count-collab.json`)

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| COLLAB k = 3 | static_end_to_end | 44,565.1 | 12,961.1 | 0.291 | <= 1.05 ok | 1 % / 0 % |

(RESULTS.md: 44.5 s for the new code, reproduced here within 0.2 %.) Foreign load: median 0.05
cores, no run flagged. The port's count stage is 10.6 s (26.6 s at `1148d15`).

### 3.4 Improvements, isolated (PLAN 8.6: reported separately, never a gate)

The port is faster than the original in every gated region. PLAN 8.6 requires the reasons to be
measured, so that an improvement never hides a regression (for example a straight-ported search
that became slower). Four configurations of the static count were built and timed. The experiment
copies of the original come from `parity/experiments/cycle_enum/build_variant.sh`: a fresh
`git archive` of `0a976ad`, the listed patches applied, built with the reference's own build
command.

- **original** (`stage_timers.patch`): instrumentation only. It prints `count_seconds` around
  `run_backend` and `read_seconds` around `read_graph_view`. The unpatched original times only
  updates.
- **original + dense histogram** (`dense_histogram.patch` and `stage_timers.patch`): the OpenMP
  counter counts into a dense per-thread array, as the port does, instead of calling
  `CycleHistogram::increment()` (a `std::map` lookup and an overflow check in another translation
  unit) once per cycle.
- **dynG `a975865`**: the straight port before the review fixes (recursive searches, the per-edge
  counts array, the `unordered_map` ownership index), built with the `parity` preset in a separate
  worktree.
- **dynG `0679ed1`**: this certificate's port (explicit-stack searches that keep the expanded vertex
  in registers, counters sized by the cycles found, the flat ownership index).

Each pair below ran as its own A/B experiment: 11 rounds, OpenMP 56, exclusive lock, medians, no
run flagged by the contamination monitor except one DD k = 6 run of the original. The records
are `M2a-experiment-*.json`. **Caveat.** Two builds of the same source can differ by up to about
20 %. The stage-timers copy took 36.2 s end to end on COLLAB k = 3 against 44.6 s for the
unpatched copy, but 3.02 s against 2.81 s on GitHub k = 4. The patch touches only the CLI's
`main`, so the difference is how the build places the unchanged counter in the binary (code
alignment). Ratios measured in one experiment are exact for its two binaries. Differences between
separately built copies below about 20 % are within this build-to-build variation.

**The count stage** (`static_count`: the original's `count_seconds`, the port's
`cycle_count.compute`; ms). (a) = `M2a-experiment-stage-timers[-collab].json`; (b) =
`M2a-experiment-dense-timers-vs-a975865.json`.

| Case | original (a) | original + dense histogram (b) | dynG `a975865`, straight port (b) | dynG `0679ed1` (a) | straight port / original + dense (b) | `0679ed1` / original (a) |
|---|---:|---:|---:|---:|---:|---:|
| DD k = 3 | 47.3 | 33.3 | 31.4 | 42.3 | 0.942 | 0.895 |
| DD k = 4 | 57.2 | 44.5 | 41.0 | 49.2 | 0.921 | 0.860 |
| DD k = 5 | 106.9 | 86.2 | 78.4 | 76.7 | 0.910 | 0.717 |
| DD k = 6 | 267.5 | 289.8 | 232.3 | 190.5 | 0.801 | 0.712 |
| DD k = 7 | 997.9 | 999.4 | 834.1 | 682.2 | 0.835 | 0.684 |
| GitHub k = 3 | 196.5 | 223.8 | 167.0 | 118.5 | 0.746 | 0.603 |
| GitHub k = 4 | 2,538.1 | 1,934.9 | 1,350.4 | 1,027.7 | 0.698 | 0.405 |
| Twitch k = 3 | 292.0 | 313.9 | 238.3 | 179.7 | 0.759 | 0.615 |
| Twitch k = 4 | 2,299.3 | 1,848.0 | 1,383.6 | 1,131.4 | 0.749 | 0.492 |
| COLLAB k = 3 | 33,635.5 | - | - | 10,583.8 | - | 0.315 |

(DD k = 3 and 4 count in 30-60 ms and are noisy; `M2a-experiment-dense-timers.json` measured
`0679ed1` there at 30.8 and 35.8 ms against 34.1 and 44.7 ms for original + dense.)

**Reading the graph** (`static_read`: the original's `read_graph_view`, which parses the file and
builds the CSR and the CSC, against the port's `read_ms + build_ms`, which build the CSR only;
run (a)): 0.89-0.97 (DD 0.90-0.97, GitHub 0.89-0.90, Twitch 0.92, COLLAB 0.96).

**Whole processes** (`static_end_to_end`, ms; each ratio from one experiment: e =
`M2a-experiment-dense-histogram-vs-a975865.json`, f = `M2a-experiment-a975865.json`):

| Case | original (gate run) | original + dense (`M2a-experiment-dense-histogram.json`) | dynG `a975865` (e) | dynG `0679ed1` (gate run) | `a975865` / original + dense (e) | `0679ed1` / `a975865` (f) |
|---|---:|---:|---:|---:|---:|---:|
| DD k = 3 | 256.7 | 265.1 | 243.8 | 238.7 | 0.942 | 1.006 |
| DD k = 4 | 251.2 | 248.6 | 240.2 | 229.6 | 0.922 | 1.000 |
| DD k = 5 | 293.3 | 305.5 | 266.3 | 262.9 | 0.871 | 0.958 |
| DD k = 6 | 514.5 | 463.0 | 426.5 | 373.1 | 0.906 | 0.883 |
| DD k = 7 | 1,488.7 | 1,263.2 | 1,047.8 | 892.7 | 0.819 | 0.847 |
| GitHub k = 3 | 719.2 | 665.4 | 600.9 | 562.4 | 0.895 | 0.930 |
| GitHub k = 4 | 2,806.5 | 2,871.5 | 1,786.4 | 1,472.0 | 0.621 | 0.822 |
| Twitch k = 3 | 1,812.5 | 1,744.6 | 1,597.8 | 1,537.2 | 0.909 | 0.972 |
| Twitch k = 4 | 3,608.7 | 3,580.3 | 2,751.3 | 2,474.1 | 0.768 | 0.907 |

**The update** (ms; `M2a-experiment-a975865.json`, f):

| Case | Region | original (gate run) | dynG `a975865` (f) | dynG `0679ed1` (gate run) | dynG `0679ed1` (f) | `0679ed1` / `a975865` (f) |
|---|---|---:|---:|---:|---:|---:|
| DD 25K+25K | update | 26.1 | 24.6 | 16.9 | 16.9 | 0.686 |
| DD 25K+25K | update_end_to_end | 368.1 | 327.9 | 308.5 | 302.1 | 0.921 |
| GitHub 25K+25K | update | 294.3 | 271.0 | 130.7 | 130.7 | 0.482 |
| GitHub 25K+25K | update_end_to_end | 3,283.7 | 2,220.8 | 1,769.2 | 1,761.6 | 0.793 |
| Twitch 25K+25K | update | 157.8 | 119.7 | 76.1 | 76.1 | 0.636 |
| Twitch 25K+25K | update_end_to_end | 4,576.7 | 3,655.0 | 3,329.1 | 3,338.8 | 0.913 |

**What this shows.**

1. **The straight-ported DFS is not slower than the original's.** With the same dense histogram on
   both sides, the straight port's count is 0.70-0.94 of the original's in the same experiment.
   So no regression hides behind the other gains: a slowdown of the ported search or of the
   ownership lookup would show here. The remaining difference is structural: the port scans
   plain int32 column arrays (4 bytes per edge), where the original scans `GraphView`'s
   `AdjacencyEntry` (vertex, edge id and two timestamp offsets: 24 bytes per edge), so it moves
   six times the memory for each edge a search reads.
2. **The dense histogram** is not the main cause, as the first certificate assumed without
   profiling. Against the stage-timers copy it saves about a fifth of the count at k = 4 (GitHub
   2,538 -> 1,935 ms, Twitch 2,299 -> 1,848 ms) and nothing measurable at k = 3 or on DD k = 7.
   Those three cases lie within the build-to-build variation above.
3. **The review fixes** make the static count 0.76-0.82 of the straight port's on the large cases
   (GitHub k = 4 1,350 -> 1,028 ms, Twitch k = 4 1,384 -> 1,131 ms, DD k = 7 834 -> 682 ms). The
   explicit-stack search scans a row in a tight inner loop. The first explicit-stack version kept
   the expanded vertex's cursor in memory instead, and it was up to 33 % slower than the recursive
   port (GitHub k = 4 count 1,788 ms, `bce07a6`, still 0.79 of the original). That was measured,
   and fixed in `0679ed1`, before this certificate. The update is 0.48-0.69 of the straight port's:
   each cycle goes straight into the thread's counters instead of a per-edge array of k + 1
   entries filled and summed for every change edge, and the flat ownership index answers the
   lookup of every scanned edge without a hash-node chase or allocation (identify_affected: 2.0
   -> 0.3 ms on DD).
4. **Reading the graph** is 0.89-0.97 of the original's `read_graph_view`. The parser is the same
   algorithm (`io::read_edge_list` is its port), and the port does not build the CSC.

### 3.5 Where the time goes in the port (medians of the profiler stages, ms)

| Case | read | build | count | update: normalize / count_minus / commit / identify / count_plus |
|---|---:|---:|---:|---|
| DD k = 3 / 7 | 190 / 196 | 10.9 | 27.7 / 678 | - |
| GitHub k = 3 / 4 | 383 / 384 | 43.2 | 123 / 1,038 | - |
| Twitch k = 3 / 4 | 1,190 / 1,185 | 142 | 185 / 1,129 | - |
| COLLAB k = 3 | 2,268 | 136 | 10,551 | - |
| DD 25K+25K | 190 | 10.4 | (prior 35.1) | 2.5 / 3.6 / 6.3 / 0.3 / 3.7 (total 16.9) |
| GitHub 25K+25K | 387 | 42.7 | (prior 1,029) | 4.4 / 95.5 / 8.2 / 0.3 / 22.5 (total 131) |
| Twitch 25K+25K | 1,198 | 141 | (prior 1,133) | 4.2 / 35.0 / 25.1 / 0.2 / 11.2 (total 76.1) |

At `1148d15` the update stages were: DD 2.6 / 8.8 / 5.8 / 2.0 / 5.3 (25.6), GitHub 4.4 / 208 /
12.5 / 2.1 / 54.5 (282), Twitch 4.3 / 64.7 / 28.1 / 2.2 / 21.1 (122). identify_affected builds
the ownership index; count_minus and count_plus are the searches (and, before the fixes, the
per-edge fill and sum of max_length + 1 counters and one hash-node allocation per change edge).

## 4. Reproducing

```bash
source scripts/dev_env.sh
parity/build_reference.sh CycleEnumeration-GPU
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/export_goldens.py cycle_count --twice
cmake --preset parity && cmake --build --preset parity
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 parity/compare.py cycle_count \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --full \
    --json parity/results/M2a-cycle_count-parity-preset.json
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py cycle_count run \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --runs 11 \
    --json parity/results/M2a-perf-openmp-cycle_count.json
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py cycle_count run \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum \
    --cases DD_k4_25000_25000_s1,DD_k3 --runs 31 \
    --json parity/results/M2a-perf-openmp-cycle_count-DD-31runs.json
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py cycle_count run \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --cases collab_k3 --runs 5 \
    --json parity/results/M2a-perf-openmp-cycle_count-collab.json
# The isolation experiments (Section 3.4); perf_ab.py takes the exclusive lock itself.
T=$(flock -s "$DYNG_SCRATCH/perf.lock" parity/experiments/cycle_enum/build_variant.sh stage_timers)
DT=$(flock -s "$DYNG_SCRATCH/perf.lock" \
    parity/experiments/cycle_enum/build_variant.sh dense_histogram stage_timers)
S=DD_k3,DD_k4,DD_k5,DD_k6,DD_k7,github_k3,github_k4,twitch_k3,twitch_k4
parity/perf_ab.py cycle_count run --exe build/parity/tools/compat/dyng-compat-cycle-enum \
    --cases $S --runs 11 --baseline-exe "$T" --baseline-label original+stage-timers \
    --json parity/results/M2a-experiment-stage-timers.json
parity/perf_ab.py cycle_count run --exe <parity build of a975865>/tools/compat/dyng-compat-cycle-enum \
    --cases $S --runs 11 --baseline-exe "$DT" --baseline-label original+dense-histogram+stage-timers \
    --json parity/results/M2a-experiment-dense-timers-vs-a975865.json
parity/perf_ab.py cycle_count run --exe build/parity/tools/compat/dyng-compat-cycle-enum \
    --runs 11 --baseline-exe <parity build of a975865>/tools/compat/dyng-compat-cycle-enum \
    --baseline-kind port --baseline-label dynG-a975865 \
    --json parity/results/M2a-experiment-a975865.json
```

The experiment records name their side A in `baseline` (label, kind, binary); `port.commit` is the
repository's HEAD when the record was written, and `port.binary` the side-B executable (for the
`-vs-a975865` records a parity build of `a975865` in a separate worktree).
