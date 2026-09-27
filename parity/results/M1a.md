# M1a parity certificate: `sssp` (CPU) against MOSP-OpenMP@c352151

Date: 2026-09-27. Format: PLAN Section 8.3 ("Parity certificate"). The machine-readable records
are next to this file:

| File | Content |
|---|---|
| `M1a-sssp-parity-preset.json` | golden replay, `parity` preset (Release, `-O3`) |
| `M1a-sssp-dev-preset.json` | golden replay, `dev` preset (Debug) |
| `M1a-crosscheck-mosp-cuda-e220ee2.json` | one-off cross-check of the two originals |
| `M1a-perf-openmp-roadNet-CA.json` | OpenMP performance A/B (default OpenMP wait policy) |
| `M1a-perf-openmp-roadNet-CA-active-wait.json` | the same with `OMP_WAIT_POLICY=active` on both sides |

## 1. What was compared

| Item | Value |
|---|---|
| Original (reference) | MOSP-OpenMP `c35215135341d5b5d1553458afe4b2226edc38fb` (baseline tag `baseline-2026-09` = `7284f50`), built from a `git archive` copy by `parity/build_reference.sh` (unpatched for performance, patched = unpatched + additive exporters for goldens) |
| Second original | MOSP-CUDA `e220ee20d1b0948ece3df135a02d1b898264c22f` (`baseline-2026-09` = `ac29545`), CUDA 13.1 (V13.1.115), driver 590.48.01, sm_86 |
| Port | dynG `93e60f8` (`sssp` sequential and OpenMP backends, `graph::apply` under `graph_properties::mosp_compatible()`), driven by `tools/compat/dyng-compat-mosp` |
| Toolchain | GCC 12.2.0 (Debian 12.2.0-14+deb12u1) for both; the original with its Makefile (`-std=c++17 -O3 -fopenmp`), dynG with the `parity` preset (`-O3`) and the `dev` preset (Debug, `-Werror`) |
| Goldens | `$DYNG_SCRATCH/goldens/sssp`: 388 cases, 8,620 files, 144 MB; `MANIFEST.sha256` = `78b658553362f77686571dc5f21c28667809580a33983354a982912c54591c26`; one SHA-256 per case in `parity/goldens.toml`. `compare.py` re-hashed every file before comparing |
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

Per case: the initial trees (`mospPrep init`), the updated trees and the `invalidated` counters
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
| **all** | **388** | **388/388** | **388/388** | **388/388** | **388/388** |

The same matrix holds for the `parity` preset and for the `dev` preset. The committed test
suite (`ctest -L cpu`) additionally runs 34 fixture cases on 3 index-type combinations and
randomized cross-checks against `testing::dijkstra` + `check_sssp_tree(require_canonical)`.

## 3. The originals among themselves (PLAN Section 6.3 step 2)

- **Inside MOSP-OpenMP** (every case, before it was written): `mosp` (mospUpdate ->
  sospUpdateCpu) == `parallelSOSPUpdate` == `sequentialSOSPUpdate` == `mospPrep expected`
  (Dijkstra on the updated graph), byte for byte, every objective; the initial trees that
  `mospTest` wrote == `mospPrep init`; for `tests/testCase0..9` the expected files tracked in the
  commit (`distances/SSSPTree{Original,Updated,SospUpdate}.txt`) == the goldens.
- **Old MOSP-CUDA == old MOSP-OpenMP** (one-off, 388/388 cases, RTX A5000 GPU 1): MOSP-CUDA's
  `mospPrep init`, `mosp` (updated trees, combined graph and MOSP costs, `invalidated` counters)
  and its `applyChangeBatch` (updated CSR and weight-increase bits, through the CUDA build of the
  graph/io exporter) are byte-identical to the MOSP-OpenMP goldens. No difference between the two
  originals exists on this corpus, so any later CUDA mismatch belongs to the port (M1b).

## 4. Proof that the harness is honest (PLAN Section 8.3)

| Check | Result |
|---|---|
| Goldens exported twice from two fresh archive copies (`export_goldens.py --twice`) | identical `MANIFEST.sha256` |
| The originals' repositories after all builds and exports | `git status` clean, HEAD at c352151 / e220ee2 (checked by `build_reference.sh` before and after each build) |
| Scratch copies equal to the archive (content and executable bit of every tracked file) | yes, for all four copies; the patched copies differ only by the added `parity_export/` directory |
| Archive-build check | both originals build from the archive alone: no gitlinks, no absolute paths needed (MOSP-CUDA's README mentions `/usr/local/cuda` as an example only); external inputs listed in `references.toml` |
| The originals' own test suites in the unpatched copies (`build_reference.sh --test`) | MOSP-OpenMP `make test`: 10/10, 100/100, 100/100, mospTest all passed. MOSP-CUDA `make test`: 10/10, 100/100, 100/100, mospTest all passed, end-to-end test passed |
| Mutation: OpenMP subtree invalidation reduced to the roots (the count-to-infinity revert of PLAN Section 8.4), replayed with sequential and openmp:4 | 266 mismatching cases, all on OpenMP (the sequential backend was not mutated and stayed equal) |
| A golden file changed by one byte | `compare.py` stops with "golden file changed"; with `--skip-verify` the byte comparison reports the case |
| The committed fixtures regenerated through the new harness | `git status` clean |

## 5. Performance A/B (OpenMP, roadNet-CA; recorded, gated from M1b)

Protocol: `parity/perf_ab.py run --runs 9`; the unpatched original's `bin/mosp` (A) and
`dyng-compat-mosp` of the `parity` preset (B) alternate A/B/A/B under the exclusive
`perf.lock`; 28 threads, `OMP_PROC_BIND=close OMP_PLACES=cores` on both; medians; `--no-output`
on both. Before the timed rounds of each batch both wrote their trees once: byte-identical.
Inputs: `perf_ab.py prepare`, i.e. the original's `bench/prepare.sh` with its own `mospPrep`:
the roadNet-CA CSR (n = 1,971,281, m = 5,533,214, K = 3 weights in [1, 100], seed 12345),
`mospPrep init`, 50K batches with 50 % insertions and seed 777 (unsafe; safe = 23,319 of 25,000
deletions kept), and the 10K local batch (160 hops, 120,604 vertices, safe). The SHA-256 of
every input is in the JSON. Load average during the runs: 7.6-10.8 (a shared lab machine).

Timed regions (`parity/timed_regions/sssp.toml`): per objective, the original's
`obj<k> SOSP update` (`sospUpdateCpu`) against dynG's `sssp.identify_affected + seed + loop +
finalize`.

| Batch | Region | Original (ms) | dynG (ms) | Ratio | Gate (from M1b) | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe 50K | SOSP obj0 | 31.50 | 25.42 | 0.807 | <= 1.05 | 22 % / 7 % |
| | SOSP obj1 | 20.87 | 21.60 | 1.035 | <= 1.05 | 10 % / 7 % |
| | SOSP obj2 | 20.23 | 21.45 | **1.060** | <= 1.05 | 7 % / 8 % |
| | **SOSP, 3 objectives** | 74.31 | 68.53 | **0.922** | <= 1.05 | 10 % / 6 % |
| | apply (+ reverse graph) | 139.20 | 107.20 | 0.770 | - | 4 % / 3 % |
| | end to end | 692.21 | 876.62 | 1.266 | <= 1.10 (M1b) | 27 % / 1 % |
| unsafe 50K | SOSP obj0 | 30.00 | 24.75 | 0.825 | <= 1.05 | 28 % / 6 % |
| | SOSP obj1 | 21.01 | 21.69 | 1.033 | <= 1.05 | 11 % / 1 % |
| | SOSP obj2 | 20.32 | 21.48 | **1.057** | <= 1.05 | 7 % / 1 % |
| | **SOSP, 3 objectives** | 71.32 | 68.03 | **0.954** | <= 1.05 | 15 % / 2 % |
| | apply (+ reverse graph) | 138.20 | 103.60 | 0.750 | - | 3 % / 5 % |
| | end to end | 686.74 | 872.98 | 1.271 | <= 1.10 (M1b) | 26 % / 1 % |
| local 10K | SOSP obj0 | 12.97 | 11.04 | 0.851 | <= 1.05 | 26 % / 27 % |
| | SOSP obj1 | 10.06 | 8.31 | 0.826 | <= 1.05 | 17 % / 24 % |
| | SOSP obj2 | 9.15 | 8.86 | 0.968 | <= 1.10 | 10 % / 23 % |
| | **SOSP, 3 objectives** | 32.42 | 28.11 | **0.867** | <= 1.05 | 16 % / 25 % |
| | apply (+ reverse graph) | 126.00 | 84.90 | 0.674 | - | 4 % / 4 % |
| | end to end | 595.71 | 814.25 | 1.367 | <= 1.10 (M1b) | 35 % / 2 % |

`invalidated` was equal on every batch and objective (safe 1,651,121 / 1,105,632 / 1,022,434;
unsafe 1,651,536 / 1,106,633 / 1,023,496; local 97,792 / 65,160 / 89,277).

### Findings

1. **The SOSP compute is at parity or faster**: the sum over the three objectives is
   0.87-0.95x of the original on all three batches (and 0.96 / 1.01 / 0.87 with
   `OMP_WAIT_POLICY=active`, the second JSON).
2. **Objectives 1 and 2 of the 50K batches are 3-6 % slower, objective 0 is 15-20 % faster.** The
   original shares ONE workspace (about 38 bytes per vertex) across its K SOSP updates; dynG gives
   each `sssp::result` its own, touched at creation (Step 3). The original's first objective
   therefore pays for touching its workspace and the later ones reuse warm pages; dynG's
   objectives each start on a workspace of their own. Experiment (not committed): with the three
   results sharing one workspace, dynG's objectives 1 and 2 dropped by 0.3-0.5 ms to
   0.97-1.03x of the original (unsafe 21.24 / 21.17 ms vs 21.89 / 21.41 ms) and objective 0 to
   0.70-0.73x. The per-objective gate of M1b is therefore sensitive to the workspace-sharing
   decision; sharing one workspace among results (as `mosp`, 0.2, needs anyway) is recorded as an
   open item in the retrospective. No change to the engine was needed.
3. **Apply is 23-33 % faster** (parallel transposition, Step 3).
4. **End to end is 1.27-1.37x of the original** (gate <= 1.10 from M1b; the original's figure also
   includes its combined-graph step, about 24 ms, which the `sssp` port does not run). Phase
   breakdown of one unsafe run (ms, original vs dynG): reading the CSR 230-380 vs 300; batch and
   trees 75-81 vs 122; building the graph object `graph::from_csr` (copy + transposition of the
   *loaded* graph, which the original never builds) - vs 213; importing and validating the three
   trees (`from_arrays` with `validate_inputs`) - vs 66; apply 137 vs 103; SOSP 72 vs 65. This
   step already removed the largest cause: reading was sequential in dynG (1.27 s -> 0.63 s by
   reading the three CSR files and the batch and tree files concurrently, like the original's
   `runConcurrently`, see the retrospective). The rest is `from_csr` building a transposition that
   `apply` rebuilds anyway, and the tree validation; both are M1b items.
5. The original's own phases vary much more between runs than dynG's (e.g. reading the CSR
   230-380 ms), which is why several spreads are flagged above 10 %. Such runs are flagged, not
   failed (PLAN Section 8.6).

## 6. Reproduce

```bash
source scripts/dev_env.sh
parity/build_reference.sh --test                    # both originals, unpatched + patched
parity/export_goldens.py --twice                    # goldens + second export from a fresh copy
cmake --preset parity && cmake --build --preset parity
ctest --preset parity -L parity                     # or: parity/compare.py --exe ...
parity/compare.py --driver original --ref "$DYNG_SCRATCH/ref/MOSP-CUDA@e220ee2/patched"
parity/perf_ab.py prepare && parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp --runs 9
```
