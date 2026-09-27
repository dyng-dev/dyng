# M1a parity certificate: `sssp` (CPU) against MOSP-OpenMP@c352151

Date: 2026-09-27 (re-recorded after the independent review of M1a; see the retrospective, Step 6). Format: PLAN Section 8.3 ("Parity certificate"). The machine-readable records
are next to this file:

| File | Content |
|---|---|
| `M1a-sssp-parity-preset.json` | golden replay, `parity` preset (Release, `-O3`) |
| `M1a-sssp-dev-preset.json` | golden replay, `dev` preset (Debug) |
| `M1a-crosscheck-mosp-cuda-e220ee2.json` | one-off cross-check of the two originals |
| `M1a-perf-openmp-roadNet-CA.json` | OpenMP performance A/B (default OpenMP wait policy; schema 2, after the review) |
| `M1a-perf-openmp-roadNet-CA-active-wait.json` | the same with `OMP_WAIT_POLICY=active` on both sides |

## 1. What was compared

| Item | Value |
|---|---|
| Original (reference) | MOSP-OpenMP `c35215135341d5b5d1553458afe4b2226edc38fb` (baseline tag `baseline-2026-09` = `7284f50`), built from a `git archive` copy by `parity/build_reference.sh` (unpatched for performance, patched = unpatched + additive exporters for goldens) |
| Second original | MOSP-CUDA `e220ee20d1b0948ece3df135a02d1b898264c22f` (`baseline-2026-09` = `ac29545`), CUDA 13.1 (V13.1.115), driver 590.48.01, sm_86 |
| Port | dynG at the commit recorded in each JSON (golden replays `e8ff5f5`, performance `06d1fc1`; the sssp engines are unchanged since `416c171`) (`sssp` sequential and OpenMP backends, `graph::apply` under `graph_properties::mosp_compatible()`), driven by `tools/compat/dyng-compat-mosp` |
| Toolchain | GCC 12.2.0 (Debian 12.2.0-14+deb12u1) for both; the original with its Makefile (`-std=c++17 -O3 -fopenmp`), dynG with the `parity` preset (`-O3`) and the `dev` preset (Debug, `-Werror`) |
| Goldens | `$DYNG_SCRATCH/goldens/sssp`: 495 cases, 11,799 files, 184 MB; `MANIFEST.sha256` = `668145c6fddb4b2a408a108d3433ccb3ecf6cdc020d1e67b06a9ec8bd28daddd`; one SHA-256 per case in `parity/goldens.toml`. `compare.py` re-hashed every file before comparing |
| Tolerance | none: byte equality of every compared file; `invalidated` counters equal (PLAN Section 8.3). `iterations`, `epochs`, `pushes` are logged only |

### The golden corpus (`parity/export_goldens.py`)

| Group | Cases | Produced by the original (c352151) |
|---|---:|---|
| `testcases` | 10 | `tests/testCase0..9` (generateTestCases, tracked in the commit) |
| `regressions` | 3 | `mospTest --only regressions`: count-to-infinity, n = 6, seeds 621705/250813 (d(1) = 90, not 60), and seeds 770968/694580, 115080/943676 |
| `thesis` | 1 | `mospTest --only thesis-example` (thesis Ch. 4 worked example, K = 3) |
| `escher` | 4 | the MOSP_ESCHER 4-vertex case (d = 100, 101), delete-all, ties; ties with K = 2 |
| `fixtures` | 17 | the graph/io fixture inputs: duplicates, existing edges, self-loops, parallel edges, unsorted rows, delete-all, an empty batch, blank lines, K = 1..5 and 32 |
| `sosp` | 148 | `mospTest --seed 1 --only sosp` (5 graphs x 3 seeds x 10 change sets = 150; the original's generator yields no batch for 2 of them, grid-medium_1 tree-weight-increases and targeted, and mospTest skips them) |
| `large_weights` | 2 | `mospTest --only large-weights`: 320 x 320 grid, weights up to 2^31 - 1 (the distance-only fallback), safe and unsafe |
| `packing` | 3 | the packing boundaries of both originals' `mospTest` (n = 2^16 + 1 with weights 2^31 - 1; n = 2^17 - 1 pull and push) |
| `stress` | 200 | `stressTest 1` and `parallelStressTest 2` (100 cases each) |
| `noncanonical` | 107 | the inputs of `testcases`, `regressions`, `escher`, `fixtures` and `sosp` with **non-canonical** initial trees: `mospPrep init`, then every vertex with several tight in-neighbours takes a random one (seeded by the case name; 75 base cases without any tie are skipped); the goldens are what `mosp` (no `--canonicalize`) computes. Added after the review: the 388 canonical cases could not detect a tie-rule difference between the backends |

Per case: the initial trees (`mospPrep init`; for `noncanonical` the perturbed trees, with `mospPrep init` in `init_canonical/` for the `compute` comparison), the updated trees and the `invalidated` counters
(`mosp --validate`), the updated CSR and weight-increase bits (`applyChangeBatch` +
`writeCsrGraph`), and the combined graph (kept for `mosp`, 0.2).

## 2. Result matrix (dynG against the goldens)

Each cell: cases with byte-identical `compute` output (= `mospPrep init`), byte-identical `update`
output (= `mosp`), byte-identical updated CSR (= `applyChangeBatch`) and equal `invalidated`
counters, for every objective.

| Group | Cases | sequential | openmp:1 | openmp:4 | openmp:16 |
|---|---:|---:|---:|---:|---:|
| testcases | 10 | 10/10 | 10/10 | 10/10 | 10/10 |
| regressions | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| thesis | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| escher | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| fixtures | 17 | 17/17 | 17/17 | 17/17 | 17/17 |
| sosp | 148 | 148/148 | 148/148 | 148/148 | 148/148 |
| large_weights | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| packing | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| stress | 200 | 200/200 | 200/200 | 200/200 | 200/200 |
| noncanonical | 107 | 107/107 | 107/107 | 107/107 | 107/107 |
| **all** | **495** | **495/495** | **495/495** | **495/495** | **495/495** |

The same matrix holds for the `parity` preset and for the `dev` preset. The committed test
suite (`ctest -L cpu`) additionally runs 34 fixture cases on 3 index-type combinations and
randomized cross-checks against `testing::dijkstra` + `check_sssp_tree(require_canonical)`.

## 3. The originals among themselves (PLAN Section 6.3 step 2)

- **Inside MOSP-OpenMP** (every case, before it was written): `mosp` (mospUpdate ->
  sospUpdateCpu) == `parallelSOSPUpdate` == `sequentialSOSPUpdate` == `mospPrep expected`
  (Dijkstra on the updated graph), byte for byte, every objective; the initial trees that
  `mospTest` wrote == `mospPrep init`; for `tests/testCase0..9` the expected files tracked in the
  commit (`distances/SSSPTree{Original,Updated,SospUpdate}.txt`) == the goldens. For the
  `noncanonical` group only `mosp` == `parallelSOSPUpdate` (both `sospUpdateCpu`) and the
  distances of `mospPrep expected` are required: `sequentialSOSPUpdate` and Dijkstra choose
  lowest-id tie parents where `sospUpdateCpu` keeps the input's; the original
  `sequentialSOSPUpdate` differs from `mosp` on 59 of the group's 298 objectives (ADR 0006).
- **Old MOSP-CUDA == old MOSP-OpenMP** (one-off, 495/495 cases including the 107 non-canonical ones, RTX A5000 GPU 1): MOSP-CUDA's
  `mospPrep init`, `mosp` (updated trees, combined graph and MOSP costs, `invalidated` counters)
  and its `applyChangeBatch` (updated CSR and weight-increase bits, through the CUDA build of the
  graph/io exporter) are byte-identical to the MOSP-OpenMP goldens. No difference between the two
  originals exists on this corpus, so any later CUDA mismatch belongs to the port (M1b).

## 4. Proof that the harness is honest (PLAN Section 8.3)

| Check | Result |
|---|---|
| Goldens exported twice from two fresh archive copies (`export_goldens.py --twice`) | identical `MANIFEST.sha256` (388 cases); after the review both reference copies were rebuilt `--fresh` and a new export of all 495 cases reproduced the committed manifest |
| The originals' repositories after all builds and exports | `git status` clean, HEAD at c352151 / e220ee2 (checked by `build_reference.sh` before and after each build) |
| Scratch copies equal to the archive (content and executable bit of every tracked file) | yes, for all four copies; the patched copies differ only by the added `parity_export/` directory |
| Archive-build check | both originals build from the archive alone: no gitlinks, no absolute paths needed (MOSP-CUDA's README mentions `/usr/local/cuda` as an example only); external inputs listed in `references.toml` |
| The originals' own test suites in the unpatched copies (`build_reference.sh --test`) | MOSP-OpenMP `make test`: 10/10, 100/100, 100/100, mospTest all passed. MOSP-CUDA `make test`: 10/10, 100/100, 100/100, mospTest all passed, end-to-end test passed |
| Mutation: OpenMP subtree invalidation reduced to the roots (the count-to-infinity revert of PLAN Section 8.4), replayed with sequential and openmp:4 | 266 mismatching cases, all on OpenMP (the sequential backend was not mutated and stayed equal) |
| A golden file changed by one byte | `compare.py` stops with "golden file changed"; with `--skip-verify` the byte comparison reports the case |
| The committed fixtures regenerated through the new harness | `git status` clean |

## 5. Performance A/B (OpenMP, roadNet-CA; recorded, gated from M1b)

This section was re-measured after the M1a review (see the retrospective, Step 6): the first
record read the gate on the sum over the objectives, left a moved first-touch cost outside the
port's region, compared the original's whole `prepare` with the port's graph work only, and
included the original's combined graph in its end-to-end figure. Its claims "SOSP at parity or
faster (0.87-0.95x)" and "apply 23-33 % faster" are withdrawn.

Protocol: `flock $DYNG_SCRATCH/perf.lock parity/perf_ab.py run --runs 21` at dynG `06d1fc1`
(clean tree); the unpatched original's `bin/mosp` (A, rebuilt and verified before timing; build
fingerprint in the JSON) and `dyng-compat-mosp` of the `parity` preset (B) alternate A/B/A/B;
28 threads, `OMP_PROC_BIND=close OMP_PLACES=cores` on both; medians of 21 runs (so the regions
under 10 ms are not provisional); `--no-output` on both. Before the timed rounds of each batch
both wrote their trees once: byte-identical; the `invalidated` counters were equal in every one
of the 63 rounds. Inputs: `perf_ab.py prepare`, i.e. the original's `bench/prepare.sh` with its
own `mospPrep`: the roadNet-CA CSR (n = 1,971,281, m = 5,533,214, K = 3 weights in [1, 100],
seed 12345), `mospPrep init`, 50K batches with 50 % insertions and seed 777 (unsafe; safe =
23,319 of 25,000 deletions kept), and the 10K local batch (160 hops, 120,604 vertices, safe).
The SHA-256 of every input is in the JSON. Load average during the runs: 3.0-5.3.

Timed regions: loaded by `perf_ab.py` from `parity/timed_regions/sssp.toml`:

- `sosp_update` (gated per objective, PLAN 6.4.2): the original's `obj<k> SOSP update`
  (`sospUpdateCpu`) against dynG's `sssp.identify_affected + seed + loop + finalize` for
  objective k. Objective 0 twice: *as measured*, and *first touch counted*: + the
  `sssp.workspace.pretouch` stage of objective 0's result. MOSP's objective 0 pays the first
  touch of its shared frontier lists inside the region; each dynG result touches its own lists
  when it is created. The gate uses the counted value.
- `sosp_total`: the sum over the objectives, including objective 0's first touch on both sides;
  reported, **not a gate**.
- `apply`: the original's `apply batch` + `prepare` against dynG's `update.commit` + the K
  `sssp.import` stages (the tree copies, which `mospUpdate` does inside `prepare`) + objective
  0's `sssp.workspace` (the one workspace reserve of `prepare`).
- `end_to_end`: the original's `end_to_end_ms` minus its `comb combined graph + SOSP` (the
  `mosp` step, 0.2) against dynG's whole run.

| Batch | Region | Reading | Original (ms) | dynG (ms) | Ratio | Gate (from M1b) | Spread A / B |
|---|---|---|---:|---:|---:|---|---|
| safe 50K | sosp_update obj0 | as measured (first touch outside the port's region) | 31.72 | 25.80 | 0.813 | - | 24 % / 8 % |
|  | sosp_update obj0 | first touch counted | 31.72 | 34.71 | **1.094** | <= 1.05 **exceeded** | 24 % / 7 % |
|  | sosp_update obj1 | as measured | 21.03 | 21.50 | 1.023 | <= 1.05 ok | 10 % / 2 % |
|  | sosp_update obj2 | as measured | 20.44 | 21.21 | 1.038 | <= 1.05 ok | 9 % / 3 % |
|  | sosp_total | as measured | 74.38 | 77.30 | 1.039 | - | 13 % / 3 % |
|  | apply | as measured | 138.30 | 128.82 | 0.931 | - | 9 % / 5 % |
|  | end_to_end | as measured | 678.34 | 884.47 | 1.304 | - | 31 % / 1 % |
| unsafe 50K | sosp_update obj0 | as measured (first touch outside the port's region) | 31.21 | 24.52 | 0.786 | - | 28 % / 10 % |
|  | sosp_update obj0 | first touch counted | 31.21 | 33.52 | **1.074** | <= 1.05 **exceeded** | 28 % / 8 % |
|  | sosp_update obj1 | as measured | 21.19 | 20.19 | 0.953 | <= 1.05 ok | 9 % / 4 % |
|  | sosp_update obj2 | as measured | 20.36 | 20.06 | 0.986 | <= 1.05 ok | 9 % / 3 % |
|  | sosp_total | as measured | 73.46 | 73.88 | 1.006 | - | 16 % / 3 % |
|  | apply | as measured | 140.30 | 131.57 | 0.938 | - | 9 % / 6 % |
|  | end_to_end | as measured | 645.82 | 879.46 | 1.362 | - | 35 % / 2 % |
| local 10K | sosp_update obj0 | as measured (first touch outside the port's region) | 13.78 | 10.81 | 0.784 | - | 37 % / 7 % |
|  | sosp_update obj0 | first touch counted | 13.78 | 19.79 | **1.436** | <= 1.05 **exceeded** | 37 % / 8 % |
|  | sosp_update obj1 | as measured | 10.15 | 8.24 | 0.812 | <= 1.05 ok | 22 % / 5 % |
|  | sosp_update obj2 | as measured | 9.34 | 8.76 | 0.938 | <= 1.10 ok | 21 % / 4 % |
|  | sosp_total | as measured | 33.78 | 36.76 | 1.088 | - | 25 % / 6 % |
|  | apply | as measured | 126.10 | 108.58 | 0.861 | - | 11 % / 7 % |
|  | end_to_end | as measured | 581.12 | 816.62 | 1.405 | - | 38 % / 1 % |

`invalidated` was equal on every batch, objective and round (safe 1,651,121 / 1,105,632 /
1,022,434; unsafe 1,651,536 / 1,106,633 / 1,023,496; local 97,792 / 65,160 / 89,277).

### Findings

1. **The per-objective gate is not met for objective 0 once its first-touch cost is counted**
   (1.07-1.09x on the 50K batches, 1.44x on the local batch), and objectives 1 and 2 are within
   the gate in this run (0.81-1.04x; earlier runs measured up to 1.063x for objective 2 of the
   50K batches, so they are close to the limit). The kernel itself is not slower: objective 0 as
   measured is 0.79-0.81x. The difference is where scratch memory is touched: MOSP shares one
   workspace across the K objectives and objective 0 touches only the pages its frontiers
   actually use; each dynG result owns a workspace and touches all six lists (about 47 MB on
   roadNet-CA) when it is created, about 9 ms per result, measured while the three results are
   built concurrently. The counted figure is therefore an upper bound, and a large one for the
   local batch, whose small frontiers touch few pages in the original. The fix is not a timing
   convention but results that share one workspace (an M1b blocker, ADR 0013); until then no
   parity claim is made for the SOSP region.
2. **The sum over the objectives** (not a gate) is 1.01-1.09x with the first touch on both sides.
3. **Apply is 0.86-0.94x** once the original's `prepare` is mapped completely (the reviewers'
   independent estimate from a stand-alone reproduction: 0.91-0.96x). The combined-graph resize
   left inside the original's `prepare` (a few ms) is the only known remaining asymmetry.
4. **End to end is 1.30-1.41x** with the original's combined graph subtracted (gate <= 1.10 from
   M1b). The causes found in M1a stand: `graph::from_csr` builds a transposition of the loaded
   graph (about 210 ms) that `apply` rebuilds anyway, and `from_arrays` validates and imports the
   trees; both are M1b items.
5. The original's own phases vary much more between runs than dynG's (spreads above 10 % are
   flagged in the JSON, not failed; PLAN Section 8.6).
6. With `OMP_WAIT_POLICY=active` on both sides (second JSON): objective 0 as measured is 0.82-0.92x and 1.28-1.65x with the
   first touch counted (the original's objective 0 is faster when its threads spin, so the moved
   cost weighs more); objectives 1 and 2 are 0.85-1.00x; apply 0.89-0.97x; end to end
   1.31-1.37x. Load average 3.5-10.9.

## 6. Reproduce

```bash
source scripts/dev_env.sh
parity/build_reference.sh --test                    # both originals, unpatched + patched
parity/export_goldens.py --twice                    # goldens + second export from a fresh copy
cmake --preset parity && cmake --build --preset parity
ctest --preset parity -L parity                     # or: parity/compare.py --exe ...
parity/compare.py --driver original --ref "$DYNG_SCRATCH/ref/MOSP-CUDA@e220ee2/patched"
parity/perf_ab.py prepare
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp --runs 21
```
