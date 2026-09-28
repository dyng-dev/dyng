# M2a parity certificate: `cycle_count` (CPU) against CycleEnumeration-GPU@0a976ad

Date: 2026-09-28. Format: PLAN Section 8.3 ("Parity certificate"). The machine-readable records
are next to this file:

| File | Content |
|---|---|
| `M2a-cycle_count-parity-preset.json` | golden replay, `parity` preset (Release, `-O3`): 24 cases x sequential, OpenMP 4 and 56 threads |
| `M2a-perf-openmp-cycle_count.json` | OpenMP-56 performance A/B, 11 runs per case (the gate of acceptance criterion 4) |
| `M2a-perf-openmp-cycle_count-DD-31runs.json` | the two noisiest DD regions again with 31 runs |
| `M2a-perf-openmp-cycle_count-collab.json` | COLLAB k = 3 static, 5 runs |

## 1. What was compared

| Item | Value |
|---|---|
| Original (reference) | CycleEnumeration-GPU `0a976adfa801a712135bf1adb51a228f353a0751` (baseline tag `baseline-2026-09` = `da2067d62c2ce8f9234908089fa15bc53b5979c0`), built from a `git archive` copy by `parity/build_reference.sh`: CMake Release (`-O3 -DNDEBUG`), `-DCYCLE_ENUM_ENABLE_OPENMP=ON -DCYCLE_ENUM_ENABLE_CUDA=ON`, sm_86, as its `docs/RESULTS.md`. Unpatched copy for performance; the patched copy adds only `parity_export/bin/export_cycle_enum` (no original file changes) and exported the goldens with the original's own `build/cycle-enum` |
| Port | dynG `1148d15` (`cycle_count` sequential and OpenMP backends, the graph under `graph_properties::cycle_enum_compatible()`, `io::read_edge_list`, `generators::legacy::cycle_enum_batch()`), driven by `tools/compat/dyng-compat-cycle-enum` |
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

**ALL EQUAL: 72 replays** (sequential COLLAB k = 3 took 444 s; GitHub / Twitch k = 4 about 60 s
each). The deltas follow: the updated histograms equal the goldens and the port's prior is its
static count, which equals `count/` for k = 4.

Other parity evidence of M2a (in-repo, CTest; see the retrospective): the 80 random fixture cases
(static, updated, k = 2..7 and unbounded, both backends), 16 fixture-graph counts and 7 updates,
19 CLI runs of the original (`compat_cycle_enum.cli`), the batch-generator identity on 22 fixture
cases and the seed-1 dataset batches, the subset-DP oracle / brute force / edge-set recount
randomized suites, and the mutation tests `cycle_count.mutation.{control,double_count_5,
weak_ownership}` (the two recorded mutations make the suite fail; the control copy passes).

## 3. Performance (PLAN Section 8.6; acceptance criterion 4)

### Methodology

- `parity/perf_ab.py cycle_count run` (`parity/cycle_count_perf.py`) under `flock
  $DYNG_SCRATCH/perf.lock` (exclusive: the builds, tests and measurements of the other agents on
  the machine take the same lock, shared or exclusive).
- Original: the UNPATCHED copy's `build/cycle-enum` (rebuilt idempotently and verified against
  the archive before timing). Port: `build/parity/tools/compat/dyng-compat-cycle-enum` at
  `1148d15`, `parity` preset (checked from its CMakeCache).
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
  M2b).
- Load average: 5.6-44 during the runs; it is dominated by the measured processes themselves
  (56 threads each). Spread = (max - min) / median; above 10 % is flagged (PLAN 8.6: flagged, not
  failed).

### Gate table (11 runs; `M2a-perf-openmp-cycle_count.json`)

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| DD k = 3 | static_end_to_end | 256.0 | 245.3 | 0.958 | <= 1.05 ok | 17 % / 20 % (noisy) |
| DD k = 4 | static_end_to_end | 251.7 | 235.3 | 0.935 | <= 1.05 ok | 6 % / 8 % |
| DD k = 5 | static_end_to_end | 302.0 | 270.1 | 0.895 | <= 1.05 ok | 6 % / 8 % |
| DD k = 6 | static_end_to_end | 516.4 | 425.4 | 0.824 | <= 1.05 ok | 4 % / 8 % |
| DD k = 7 | static_end_to_end | 1,479.5 | 1,038.8 | 0.702 | <= 1.05 ok | 3 % / 3 % |
| GitHub k = 3 | static_end_to_end | 713.9 | 603.9 | 0.846 | <= 1.05 ok | 4 % / 4 % |
| GitHub k = 4 | static_end_to_end | 2,800.2 | 1,785.3 | 0.638 | <= 1.05 ok | 1 % / 10 % |
| Twitch k = 3 | static_end_to_end | 1,817.9 | 1,595.8 | 0.878 | <= 1.05 ok | 3 % / 2 % |
| Twitch k = 4 | static_end_to_end | 3,624.1 | 2,746.9 | 0.758 | <= 1.05 ok | 2 % / 1 % |
| DD 25K+25K k = 4 | update | 26.7 | 25.6 | 0.959 | <= 1.05 ok | 84 % / 32 % (noisy) |
| GitHub 25K+25K k = 4 | update | 294.0 | 281.9 | 0.959 | <= 1.05 ok | 5 % / 9 % |
| Twitch 25K+25K k = 4 | update | 160.7 | 121.9 | 0.759 | <= 1.05 ok | 6 % / 13 % (noisy) |
| DD 25K+25K k = 4 | update_end_to_end | 355.0 | 316.9 | 0.893 | <= 1.10 ok | 17 % / 12 % (noisy) |
| GitHub 25K+25K k = 4 | update_end_to_end | 3,292.2 | 2,241.4 | 0.681 | <= 1.10 ok | 1 % / 8 % |
| Twitch 25K+25K k = 4 | update_end_to_end | 4,596.2 | 3,659.3 | 0.796 | <= 1.10 ok | 1 % / 1 % |

**Every gate is met.** The two noisy DD regions (single outliers of 37-48 ms in a 25 ms region;
the machine is shared) were measured again with 31 runs
(`M2a-perf-openmp-cycle_count-DD-31runs.json`): DD k = 3 static 259.5 / 241.8 ms (0.932), DD
update 26.0 / 25.0 ms (0.960), DD update end to end 356.4 / 325.4 ms (0.913); same verdicts.

### COLLAB k = 3 (5 runs; `M2a-perf-openmp-cycle_count-collab.json`)

| Case | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| COLLAB k = 3 | static_end_to_end | 44,513.4 | 29,028.0 | 0.652 | <= 1.05 ok | 0 % / 0 % |

(RESULTS.md: 44.5 s for the new code, reproduced here within 0.1 %.)

### Where the time goes in the port (medians of the profiler stages, ms)

| Case | read | build | count | update: normalize / count_minus / commit / identify / count_plus |
|---|---:|---:|---:|---|
| DD k = 3 / 7 | 196 / 189 | 10.8 | 30.8 / 831 | - |
| GitHub k = 3 / 4 | 384 / 387 | 42.7 | 168 / 1,344 | - |
| Twitch k = 3 / 4 | 1,198 / 1,200 | 143 | 234 / 1,383 | - |
| COLLAB k = 3 | 2,295 | 138 | 26,587 | - |
| DD 25K+25K | 185 | 10.8 | (prior 39.8) | 2.6 / 8.8 / 5.8 / 2.0 / 5.3 (total 25.6) |
| GitHub 25K+25K | 386 | 42.6 | (prior 1,345) | 4.4 / 208 / 12.5 / 2.1 / 54.5 (total 282) |
| Twitch 25K+25K | 1,202 | 143 | (prior 1,381) | 4.3 / 64.7 / 28.1 / 2.2 / 21.1 (total 122) |

### Why the port is faster (not an intended improvement; reported, not gated)

No part of the port is slower than the original, so nothing had to be profiled and fixed for the
gate. The known structural differences, all recorded deviations of M2a:

- **Histogram increments.** The original's counters call `CycleHistogram::increment()` (a
  `std::map` lookup plus an overflow check, out of line in another library) once per cycle found;
  the port adds to a dense per-thread array and checks overflow when histograms are merged
  (deviation 4 of step 2). This is most likely the main difference (not profiled: the port is
  not slower anywhere) where cycles are many per search step: COLLAB k = 3
  (1.26 billion cycles; count 26.6 s vs about 42 s), GitHub and Twitch k = 4, DD k = 7.
- **Graph layout.** The original's DFS reads `GraphView` edges (a struct per edge); the port reads
  plain int32 column arrays of the CSR.
- **Graph build.** dynG builds only the out-edge CSR for the count (the original also builds the
  CSC), and the commit of the update builds the new CSR in parallel blocks (step 1).

The update's compute gate (`update_seconds`) compares the same scope on both sides, and the
end-to-end regions include the untimed parts (read, build, prior, generation). One caveat of the
compat driver (outside every gated region, reported for completeness): it generates the batch
from `g.to_csr(res)`, a copy of the graph (Twitch `generate_ms` 778 ms, the original's
`generate_batch` reads its `GraphView` directly); the update's end-to-end ratio is 0.80
nevertheless.

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
```
