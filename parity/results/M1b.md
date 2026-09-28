# M1b parity record: `sssp` (OpenMP and CUDA) against MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2

Date: 2026-09-27/28. Format: PLAN Section 8.3 ("Parity certificate") and ADR 0013. The record
grew with the steps of M1b: sections 1-5 are the OpenMP gate of step cpu-gates, section 6 the
first CUDA readings of step sssp-cuda, **sections 7-12 the M1b certificate** (step
cuda-parity-perf): byte parity on every backend, the CUDA and OpenMP performance gates on the four
graphs of PLAN 6.4.2, the fused kernel's resources, and the `edge_t` benchmark; **section 13**
re-measures parity and the gates on the code fixed after the M1b review; **section 14** reads
the CUDA gate with the GPU clocks locked for the whole A/B (ADR 0018 rule 4, without root), the
protocol of the gate from the M1b acceptance fix on. Where the sections disagree, sections 7-12
supersede 3 and 6.3, section 13 supersedes the gate readings of 7-9 (it measures the final code),
and section 14 supersedes section 13's CUDA verdict (section 13's default-clock readings stay as
the ungated as-measured record).

**Certificate summary** (details and every number below; machine-readable records next to this
file):

| Check (PLAN 6.4.2, 8.3, 8.6) | Result |
|---|---|
| Golden corpus, 495 cases (incl. the M1b cases), byte equality | **pass**: sequential, OpenMP 1/4/16/28 threads, CUDA; int32 and int64 edge offsets (section 7) |
| Goldens re-exported with MOSP-CUDA@e220ee2's own tools | **identical** to the committed MOSP-OpenMP corpus, file for file (section 7.2) |
| Paper-scale outputs (4 graphs x 3 batches, both backends) | byte-identical trees, `invalidated` equal in every timed round (sections 8, 9) |
| CUDA per-objective SOSP region at **default** clocks (recorded, not gated: ADR 0018, option B accepted 2026-09-28) | 34 / 36 within 1.05x (0.98-1.01x); road_usa local 10K objectives 0 and 1 read 1.065x and 1.063x (objective 2 1.037x), caused by the GPU's clock state, not the kernel (section 8.2) |
| CUDA per-objective fused kernel at locked clocks | 36 / 36 within 0.992-1.014x (road_usa local 10K: 0.992-0.995x); DRAM bytes 0.99-1.03x (section 8.4) |
| CUDA apply and end to end (<= 1.10x) | **pass**: apply 0.55-0.74x, end to end 0.83-0.92x |
| OpenMP per-objective SOSP region (<= 1.05x / 1.10x) | **36 / 36 within the gate** (0.63-0.99x), after the barrier change of section 9 (the straight port exceeded on road_usa local 10K in two campaigns) |
| OpenMP apply and end to end (<= 1.10x) | **pass**: apply 0.78-1.02x, end to end 0.75-0.91x |
| Fused kernel registers / occupancy vs the original | **equal**: 59 registers, 616 bytes of parameters, 4 blocks of 256 threads per SM, grid 256 x 256 (section 11) |
| **M1b review** (section 13): the gates re-measured on the fixed code `674fc3b` with the contamination monitor | byte parity 495 / 495 in 9 configurations; CUDA 33 / 36 per objective within the gate, road_usa local 10K **1.060 / 1.060 / 1.054x (default clocks; not gated per ADR 0018)**, 0.989-0.991x at locked clocks; OpenMP 36 / 36; apply and end to end pass on both; no contamination in the gated rounds |
| **M1b acceptance fix** (section 14): the CUDA gate with the clocks locked for the whole A/B (`--lock-clocks boost`, no root), code `3ecb8fd` | **36 / 36 per-objective regions within the gate** (0.977-1.028x; road_usa local 10K **0.993 / 0.994 / 0.995x**); apply 0.67-0.87x, end to end 0.76-0.90x; clocks at the locked values in every busy sample of every round; at base clocks road_usa 0.993-1.002x; at default clocks road_usa local 10K still 1.06x (P-state, recorded, ungated) |
| `edge_t` benchmark (ADR 0009) | int64 costs 3-4.5 % on the CUDA SOSP region (50K batches), up to 4.4 % on OpenMP; default int32 with checked construction (section 10) |

| File | Content |
|---|---|
| `M1b-final-sssp-parity-preset.json` | final golden replay, host backends (sequential, OpenMP 1/4/16/28; int64: sequential, OpenMP 4), `parity` preset |
| `M1b-final-sssp-cuda-parity-cuda-preset.json` | final golden replay, CUDA (int32, int64), `parity-cuda` preset, GPU 1 |
| `M1b-goldens-mosp-cuda-export.json` | the corpus exported with MOSP-CUDA@e220ee2's tools, compared with the committed corpus |
| `M1b-final-perf-cuda-<graph>.json` | CUDA A/B against MOSP-CUDA@e220ee2 (the gate record) |
| `M1b-final-kernels-cuda-<graph>.json` | the fused kernels of both under Nsight Compute, clocks locked to base |
| `M1b-final-perf-openmp-<graph>.json` | OpenMP A/B against MOSP-OpenMP@c352151 (the gate record) |
| `M1b-edge-type-{openmp,cuda}-<graph>.json` | the `edge_t` benchmark: int32 against int64 edge offsets |
| `M1b-diag-openmp-road_usa_g-*.json` | the OpenMP road_usa readings before the barrier change (section 9) |
| `M1b-sssp-parity-preset.json`, `M1b-sssp-dev-preset.json`, `M1b-perf-openmp-*.json` | step cpu-gates (sections 2, 3) |
| `M1b-sssp-cuda-parity-cuda-preset.json` | step sssp-cuda (section 6.1) |
| `M1b-accept-perf-cuda-<graph>.json` | the CUDA gate at locked clocks (boost), section 14 |
| `M1b-accept-perf-cuda-base-road_usa_g.json`, `M1b-accept-perf-cuda-unlocked-road_usa_g.json` | road_usa at base clocks; road_usa's local batch at default clocks (section 14) |
| `M1b-review-*.json` | the M1b review's records on the fixed code (section 13): golden replays, both A/Bs with the contamination monitor, the locked-clock kernels of road_usa |

`<graph>` is `roadNet-PA`, `roadNet-CA`, `rgg` (rgg_n_2_20_s0) or `road_usa_g` (road_usa).

## 1. What was compared (step cpu-gates)

| Item | Value |
|---|---|
| Original (reference) | MOSP-OpenMP `c35215135341d5b5d1553458afe4b2226edc38fb`, built by `parity/build_reference.sh` from a `git archive` copy (unpatched for performance; the build fingerprint is in every JSON) |
| Port | dynG `79400d2` (clean tree; every JSON records it), `tools/compat/dyng-compat-mosp` of the `parity` preset (`-O3`, GCC 12.2.0, OpenMP) |
| Goldens | `$DYNG_SCRATCH/goldens/sssp`: 495 cases, 11,799 files, `MANIFEST.sha256` = `668145c6fddb4b2a408a108d3433ccb3ecf6cdc020d1e67b06a9ec8bd28daddd` (unchanged since M1a) |
| Tolerance | none: byte equality of every compared file; `invalidated` counters equal |
| Host | Xeon Gold 6258R (28 cores, 56 threads, one socket), 124 GB, Debian 12, kernel 6.1 |

## 2. Byte parity (golden replay)

`parity/compare.py` over all 495 cases of the M1a corpus (testcases, regressions, thesis,
escher, fixtures, the 148 `mospTest sosp` cases, large weights, packing boundaries, 200 stress
cases, 107 non-canonical cases) on sequential and OpenMP with 1, 4 and 16 threads: **495/495
byte-identical in every configuration**, in the `parity` and in the `dev` preset (compute, update,
updated CSR, `invalidated`). The replay also ran during the step, after the workspace change and
after the I/O and apply changes (`ctest -L parity` inside `ci/check.sh --parity`). The OpenMP paths added in this step (the parallel CSR
assembly of `apply`, the `from_csr` fast path, the parallel import and validation) are exercised
by the corpus: the packing and large-weight cases have 102,400-131,071 rows, above the parallel
thresholds.

The A/B below adds 12 paper-scale checks: before the timed rounds of each graph x batch, the
original and the port write their updated trees once, and the files were byte-identical in all
12; the `invalidated` counters were equal for every objective in every one of the 252 timed
rounds.

## 3. Performance A/B (OpenMP; the PLAN 8.6 gates)

**Protocol.** `flock $DYNG_SCRATCH/perf.lock parity/perf_ab.py run --exe
build/parity/tools/compat/dyng-compat-mosp --graph <g> --runs 21 --json ...` at `79400d2`: the
unpatched original's `bin/mosp` (A, rebuilt and verified before timing) and `dyng-compat-mosp`
(B) alternate A/B/A/B, 21 rounds per batch, medians compared; 28 threads with
`OMP_PROC_BIND=close OMP_PLACES=cores` on both sides (the default wait policy); `--no-output` on
both. Regions from `parity/timed_regions/sssp.toml` (loaded by the script):

- `sosp_update objK` (gated per objective, PLAN 6.4.2: <= 1.05x, <= 1.10x when the original's
  median is below 10 ms, which needs >= 20 runs): the original's `obj<k> SOSP update`
  (`sospUpdateCpu`) against dynG's `sssp.identify_affected + seed + loop + finalize` of objective
  k. **As measured on both sides**: since this step the K results share one workspace (ADR 0015),
  exactly as `mospUpdate()` shares its `SospWorkspace`, so objective 0 pays the first touch of
  the frontier-list pages it uses inside its region on both sides and no cost is moved (M1a's
  "first touch counted" reading is withdrawn with its cause).
- `apply` (<= 1.10x): the original's `apply batch` + `prepare` (reverse graph, workspace reserve,
  tree copies) against dynG's `update.commit` (apply + transposition of the updated graph) + the
  K tree imports + every `sssp.workspace` lease.
- `end_to_end` (<= 1.10x, PLAN 8.6): the original's `end_to_end_ms` minus its combined-graph step
  (`mosp`, 0.2) against dynG's whole run (read, build, validate the input trees, update).
- `sosp_total` (the sum over the objectives) is in the JSON for orientation only; it is not a
  gate reading.

Inputs: `parity/perf_ab.py prepare --graph <g>`, i.e. the original's `bench/prepare.sh` with its
own `mospPrep` (MOSP-OpenMP@c352151, unpatched) on the prepared CSR of
`$DYNG_SCRATCH/datasets/mosp/<g>/csr` (K = 3 weights in [1, 100], seed 12345): `mospPrep init`,
50K batches with 50 % insertions and seed 777 (unsafe; safe = deletions filtered so that no
vertex becomes unreachable) and the 10K local batch (seed 777, `--local` radius 110 / 160 / 110
/ 200 hops on roadNet-PA / roadNet-CA / rgg_n_2_20_s0 / road_usa, the radii the original's
bench/prepare.sh names). The SHA-256 of every input file is in each JSON.

**Result: every gate is met** (36 per-objective SOSP readings, 12 apply and 12 end-to-end
readings; `--enforce-gates` would pass). Spreads above 10 % are flagged in the JSON, not failed
(PLAN 8.6); they come almost entirely from the original's input phase (its `read_graph` stage
alone varied between 2.1 and 3.9 s on road_usa in the development runs), which the medians of 21
alternating rounds absorb. The load average during the runs (mostly the runs themselves) is in
each JSON.

### 3.1 roadNet-PA (`M1b-perf-openmp-roadNet-PA.json`; port `79400d2`)

| Batch | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 16.39 | 14.73 | 0.898 | <= 1.05 ok | 30 % / 7 % |
|  | sosp_update obj1 | 12.63 | 11.80 | 0.934 | <= 1.05 ok | 14 % / 4 % |
|  | sosp_update obj2 | 12.40 | 11.77 | 0.949 | <= 1.05 ok | 12 % / 5 % |
|  | sosp_total | 41.67 | 38.36 | 0.920 | - | 18 % / 4 % |
|  | apply | 77.00 | 69.78 | 0.906 | <= 1.10 ok | 29 % / 20 % |
|  | end_to_end | 373.84 | 308.56 | 0.825 | <= 1.10 ok | 37 % / 7 % |
| unsafe50k | sosp_update obj0 | 18.45 | 14.40 | 0.781 | <= 1.05 ok | 26 % / 11 % |
|  | sosp_update obj1 | 12.97 | 11.71 | 0.903 | <= 1.05 ok | 14 % / 2 % |
|  | sosp_update obj2 | 12.67 | 11.71 | 0.924 | <= 1.05 ok | 11 % / 4 % |
|  | sosp_total | 43.27 | 37.76 | 0.872 | - | 18 % / 3 % |
|  | apply | 87.70 | 68.78 | 0.784 | <= 1.10 ok | 22 % / 20 % |
|  | end_to_end | 379.49 | 303.55 | 0.800 | <= 1.10 ok | 36 % / 5 % |
| local10k | sosp_update obj0 | 28.14 | 22.60 | 0.803 | <= 1.05 ok | 30 % / 25 % |
|  | sosp_update obj1 | 25.11 | 20.37 | 0.812 | <= 1.05 ok | 22 % / 27 % |
|  | sosp_update obj2 | 24.74 | 20.04 | 0.810 | <= 1.05 ok | 21 % / 27 % |
|  | sosp_total | 78.11 | 62.84 | 0.804 | - | 24 % / 26 % |
|  | apply | 65.70 | 56.28 | 0.857 | <= 1.10 ok | 27 % / 27 % |
|  | end_to_end | 390.46 | 319.43 | 0.818 | <= 1.10 ok | 34 % / 11 % |

safe50k: invalidated [1049304, 1071591, 1001669] (equal in every sample: True); load 10.6-13.3; runs 21
unsafe50k: invalidated [1049383, 1071640, 1001814] (equal in every sample: True); load 12.4-14.8; runs 21
local10k: invalidated [532189, 482776, 371509] (equal in every sample: True); load 12.9-14.3; runs 21

### 3.2 roadNet-CA (`M1b-perf-openmp-roadNet-CA.json`; port `79400d2`)

| Batch | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 31.59 | 25.50 | 0.807 | <= 1.05 ok | 27 % / 9 % |
|  | sosp_update obj1 | 20.90 | 20.56 | 0.984 | <= 1.05 ok | 7 % / 10 % |
|  | sosp_update obj2 | 20.30 | 20.10 | 0.990 | <= 1.05 ok | 7 % / 8 % |
|  | sosp_total | 72.91 | 66.09 | 0.906 | - | 15 % / 9 % |
|  | apply | 139.40 | 119.60 | 0.858 | <= 1.10 ok | 10 % / 10 % |
|  | end_to_end | 676.47 | 517.37 | 0.765 | <= 1.10 ok | 34 % / 5 % |
| unsafe50k | sosp_update obj0 | 31.39 | 25.28 | 0.805 | <= 1.05 ok | 30 % / 11 % |
|  | sosp_update obj1 | 20.96 | 20.31 | 0.969 | <= 1.05 ok | 9 % / 6 % |
|  | sosp_update obj2 | 20.47 | 20.05 | 0.980 | <= 1.05 ok | 8 % / 2 % |
|  | sosp_total | 74.25 | 65.68 | 0.885 | - | 16 % / 4 % |
|  | apply | 139.80 | 122.54 | 0.877 | <= 1.10 ok | 10 % / 11 % |
|  | end_to_end | 686.85 | 520.22 | 0.757 | <= 1.10 ok | 32 % / 6 % |
| local10k | sosp_update obj0 | 13.25 | 11.23 | 0.847 | <= 1.05 ok | 45 % / 18 % |
|  | sosp_update obj1 | 9.82 | 7.97 | 0.812 | <= 1.10 ok | 28 % / 19 % |
|  | sosp_update obj2 | 9.17 | 8.58 | 0.936 | <= 1.10 ok | 24 % / 21 % |
|  | sosp_total | 32.71 | 27.82 | 0.851 | - | 33 % / 19 % |
|  | apply | 124.80 | 105.31 | 0.844 | <= 1.10 ok | 10 % / 10 % |
|  | end_to_end | 609.78 | 463.69 | 0.760 | <= 1.10 ok | 35 % / 7 % |

safe50k: invalidated [1651121, 1105632, 1022434] (equal in every sample: True); load 11.1-14.1; runs 21
unsafe50k: invalidated [1651536, 1106633, 1023496] (equal in every sample: True); load 11.0-12.8; runs 21
local10k: invalidated [97792, 65160, 89277] (equal in every sample: True); load 11.2-14.0; runs 21

### 3.3 rgg_n_2_20_s0 (`M1b-perf-openmp-rgg.json`; port `79400d2`)

| Batch | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 52.13 | 52.48 | 1.007 | <= 1.05 ok | 6 % / 8 % |
|  | sosp_update obj1 | 40.14 | 39.78 | 0.991 | <= 1.05 ok | 2 % / 4 % |
|  | sosp_update obj2 | 44.54 | 44.16 | 0.991 | <= 1.05 ok | 3 % / 7 % |
|  | sosp_total | 137.05 | 136.59 | 0.997 | - | 3 % / 5 % |
|  | apply | 229.60 | 179.98 | 0.784 | <= 1.10 ok | 8 % / 8 % |
|  | end_to_end | 1162.08 | 974.88 | 0.839 | <= 1.10 ok | 35 % / 7 % |
| unsafe50k | sosp_update obj0 | 51.67 | 52.27 | 1.012 | <= 1.05 ok | 6 % / 7 % |
|  | sosp_update obj1 | 40.10 | 39.71 | 0.990 | <= 1.05 ok | 1 % / 2 % |
|  | sosp_update obj2 | 44.43 | 43.92 | 0.989 | <= 1.05 ok | 2 % / 4 % |
|  | sosp_total | 136.24 | 135.96 | 0.998 | - | 3 % / 3 % |
|  | apply | 227.40 | 182.96 | 0.805 | <= 1.10 ok | 8 % / 6 % |
|  | end_to_end | 1160.58 | 962.09 | 0.829 | <= 1.10 ok | 37 % / 9 % |
| local10k | sosp_update obj0 | 79.84 | 65.32 | 0.818 | <= 1.05 ok | 18 % / 18 % |
|  | sosp_update obj1 | 69.72 | 57.05 | 0.818 | <= 1.05 ok | 18 % / 19 % |
|  | sosp_update obj2 | 77.79 | 64.83 | 0.833 | <= 1.05 ok | 18 % / 18 % |
|  | sosp_total | 228.34 | 186.83 | 0.818 | - | 16 % / 18 % |
|  | apply | 201.30 | 154.17 | 0.766 | <= 1.10 ok | 11 % / 5 % |
|  | end_to_end | 1183.20 | 1003.30 | 0.848 | <= 1.10 ok | 40 % / 8 % |

safe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 9.7-12.9; runs 21
unsafe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 9.7-13.4; runs 21
local10k: invalidated [274665, 184671, 284403] (equal in every sample: True); load 11.0-13.7; runs 21

### 3.4 road_usa (`M1b-perf-openmp-road_usa_g.json`; port `79400d2`)

| Batch | Region | Original (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 314.82 | 280.00 | 0.889 | <= 1.05 ok | 8 % / 7 % |
|  | sosp_update obj1 | 273.59 | 263.56 | 0.963 | <= 1.05 ok | 8 % / 6 % |
|  | sosp_update obj2 | 275.46 | 264.15 | 0.959 | <= 1.05 ok | 4 % / 6 % |
|  | sosp_total | 866.16 | 807.24 | 0.932 | - | 6 % / 6 % |
|  | apply | 1573.10 | 1248.38 | 0.794 | <= 1.10 ok | 6 % / 3 % |
|  | end_to_end | 7153.71 | 5479.48 | 0.766 | <= 1.10 ok | 33 % / 4 % |
| unsafe50k | sosp_update obj0 | 319.23 | 280.00 | 0.877 | <= 1.05 ok | 10 % / 4 % |
|  | sosp_update obj1 | 279.48 | 264.37 | 0.946 | <= 1.05 ok | 8 % / 3 % |
|  | sosp_update obj2 | 278.71 | 264.13 | 0.948 | <= 1.05 ok | 8 % / 3 % |
|  | sosp_total | 877.23 | 807.66 | 0.921 | - | 9 % / 3 % |
|  | apply | 1576.80 | 1250.64 | 0.793 | <= 1.10 ok | 7 % / 3 % |
|  | end_to_end | 7070.29 | 5503.99 | 0.778 | <= 1.10 ok | 36 % / 4 % |
| local10k | sosp_update obj0 | 73.60 | 61.94 | 0.842 | <= 1.05 ok | 18 % / 3 % |
|  | sosp_update obj1 | 62.29 | 59.61 | 0.957 | <= 1.05 ok | 15 % / 4 % |
|  | sosp_update obj2 | 62.49 | 59.94 | 0.959 | <= 1.05 ok | 16 % / 5 % |
|  | sosp_total | 199.08 | 181.49 | 0.912 | - | 16 % / 4 % |
|  | apply | 1551.90 | 1226.73 | 0.790 | <= 1.10 ok | 3 % / 2 % |
|  | end_to_end | 5977.29 | 4849.66 | 0.811 | <= 1.10 ok | 40 % / 4 % |

safe50k: invalidated [17043767, 15989663, 16311741] (equal in every sample: True); load 5.6-14.6; runs 21
unsafe50k: invalidated [17046218, 15994408, 16314393] (equal in every sample: True); load 5.2-12.7; runs 21
local10k: invalidated [261871, 254274, 230241] (equal in every sample: True); load 3.6-12.7; runs 21

## 4. Findings

1. **Per objective, the port is at parity or faster everywhere** (0.78-1.01x; the largest ratio
   is objective 0 of rgg_n_2_20_s0, 1.007-1.012x, where both sides' objective 0 includes the
   first touch). Objectives 1..K-1 compare warm pages on both sides (0.90-0.99x on the 50K
   batches). The kernel is the same code as in M1a; what changed is that the objectives share one
   workspace, as in the original (ADR 0015).
2. **End to end 0.76-0.85x** (M1a: 1.30-1.41x on roadNet-CA). The M1a gap was in the
   surroundings, not in the engines: the load-time transposition of the input graph (never used:
   the update needs the in-edges of the updated graph), file reads that grew a string in 64 KB
   chunks (several copies of every file), a two-pass token parser for the tree files, a
   sequential edge-major to objective-major conversion of the weights, and sequential validation
   of the imported trees. dynG still validates the K input trees (`validate_inputs`, which the
   original does not do) and still wins.
3. **Apply 0.77-0.91x.** The CSR assembly of `apply` and the transposition run in parallel with
   identical output (the original's `applyChangeBatch` is sequential; its `buildHostGraph` is
   parallel).
4. Memory: one workspace for the K objectives instead of K (about 38 bytes per vertex once).

## 5. Reproduce

```bash
source scripts/dev_env.sh
parity/build_reference.sh
cmake --preset parity && cmake --build --preset parity
parity/compare.py --exe build/parity/tools/compat/dyng-compat-mosp --json /tmp/replay.json
for g in roadNet-PA roadNet-CA rgg road_usa_g; do
  parity/perf_ab.py prepare --graph $g
  flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py run \
      --exe build/parity/tools/compat/dyng-compat-mosp --graph $g --runs 21 --enforce-gates
done
```

## 6. Step sssp-cuda: `sssp` (CUDA) against MOSP-CUDA@e220ee2

### 6.1 Byte parity (golden replay on cuda; `M1b-sssp-cuda-parity-cuda-preset.json`)

`parity/compare.py --configs cuda` through `dyng-compat-mosp --backend cuda` (parity-cuda preset:
Release, host `-O3`, nvcc `-O3 -lineinfo -fmad=true`, sm_86; CUDA 13.1; RTX A5000, GPU 1), port
`492eba0`: **495 / 495 cases byte-identical** (`compute` = `mospPrep init`, `update` = `mosp`
from the golden initial trees incl. the 107 non-canonical ones, the updated CSR, and the
`invalidated` counter of every objective). The goldens are MOSP-OpenMP@c352151's; MOSP-CUDA
e220ee2 produces the same files on every case (the one-off cross-check,
`M1a-crosscheck-mosp-cuda-e220ee2.json`), so this is byte parity with MOSP-CUDA.

### 6.2 The fused kernel's resources (PLAN 8.6 kernel checks)

`cuobjdump --dump-resource-usage` (sm_86):

| Kernel | REG | STACK | SHARED | CONSTANT[0] (parameters) |
|---|---:|---:|---:|---:|
| MOSP-CUDA `sospPersistentKernel` (unpatched `bin/mosp`) | 59 | 0 | 0 | 616 |
| dynG `sssp_persistent_kernel<int32, int32, int32>` (parity-cuda `libdyng.so`) | 59 | 0 | 0 | 616 |
| dynG `sssp_persistent_kernel<int32, int64, int32>` | 60 | 0 | 0 | 616 |
| dynG `sssp_persistent_kernel<int64, int64, int32>` | 64 | 0 | 0 | 624 |

Same registers for the original's types, so the same occupancy and the same cooperative grid
(256 blocks of 256 threads on the RTX A5000 on both sides, nsys launch records). The SASS of the
int32 instantiation has 224 more instructions than the original's (2,592 vs 2,368), all in the
unpack pass (the `affected` count, ADR 0017 item 1); a build without that count produces SASS
byte-identical to the original's (opcodes, registers and constant offsets). The int64
instantiations stay within 64 registers (4 blocks of 256 threads per SM on sm_86).

### 6.3 A first A/B (not the gate record)

roadNet-PA, `perf_ab.py run --backend cuda --runs 5` (GPU 0, exclusive perf lock), ratio dynG /
MOSP-CUDA of the medians: SOSP region per objective 0.98-1.00x (4.6-9.4 ms; host times of the same
scope on both sides, dynG's CUDA-event time within 0.01 ms of its host time), apply 0.55-0.60x,
end to end 0.82-0.84x, for the 50K safe, 50K unsafe and 10K local batches; outputs byte-identical
and `invalidated` equal in every run. The gate record (four graphs, >= 20 runs because the
per-objective regions are under 10 ms on roadNet-PA) follows in the next step.

### 6.4 Finding: GPU clocks, not code, separate the per-objective times on road_usa local 10K

A 5-run A/B on road_usa (`--batches safe50k,local10k`) read the 50K safe batch at 1.00-1.01x per
objective (about 100 ms) but the local 10K batch at 1.05-1.09x (about 21 ms). Investigation:

- The kernels are the same code (6.2), launched with the same grid, on identical inputs (the
  `invalidated`, `iterations` and `epochs` counters match).
- The first version of the `affected` count read the old pair next to the unconditional write
  (+12 bytes per vertex): about 0.4 ms of the gap. The unpack now writes only the pairs that
  change (commit `522cf0d`), which costs nothing measurable (22.62 vs 22.65 ms without the count).
- Under Nsight Compute, which locks the clocks, the kernels take the same time: 29.03 / 28.15 /
  28.44 ms (dynG) vs 29.17 / 28.32 / 28.59 ms (MOSP-CUDA), with equal instruction counts and DRAM
  traffic within 1 %.
- nsys GPU metrics during the three kernels: GPC clock 1.69 / 1.69 / 1.77 GHz for dynG, 1.90 /
  1.90 / 1.89 GHz for MOSP-CUDA (SYS clock 1.42 vs 1.60 GHz). The boost clock is still ramping
  when dynG's first kernels start: dynG's GPU work right before objective 0 is the graph upload
  (about 150 ms), MOSP-CUDA's is its whole "upload" stage (about 490 ms: graph, the K trees,
  workspace), after seconds of host-only file reading on both sides.

So the remaining difference is a boost-clock state (GPU idle before the timed region, dynG's
shorter GPU lead-in), not the port. The gate step has to choose a protocol that controls for it
(locked clocks if available, or the same GPU lead-in on both sides, recorded with the clock
during the kernels); it must not be hidden by moving work.

## 7. Step cuda-parity-perf: byte parity (the certificate, PLAN 8.3)

### 7.1 What was compared

| Item | Value |
|---|---|
| Originals | MOSP-CUDA `e220ee20d1b0948ece3df135a02d1b898264c22f` and MOSP-OpenMP `c35215135341d5b5d1553458afe4b2226edc38fb`, built by `parity/build_reference.sh` from `git archive` copies (CUDA 13.1 nvcc, `make -j16 all stressTest parallelStressTest`; goldens from the patched copies, whose patches only add export tools; timings from the unpatched copies; the build fingerprints are in every JSON) |
| Port | dynG `0f0fba9` (clean tree; recorded in every JSON). Later commits of the step change names and documentation only |
| Presets | `parity` (Release, `-O3`, GCC 12.2.0, OpenMP) and `parity-cuda` (plus nvcc `-O3 -lineinfo -fmad=true`, sm_86) |
| Goldens | `$DYNG_SCRATCH/goldens/sssp`: 495 cases, 11,799 files, `MANIFEST.sha256` = `668145c6fddb4b2a408a108d3433ccb3ecf6cdc020d1e67b06a9ec8bd28daddd` (unchanged since M1a) |
| Tolerance | none: byte equality of every compared file (distances, parents, the updated CSR); `invalidated` counters equal; `iterations`, `epochs`, `pushes` logged only |
| Hardware | Xeon Gold 6258R (28 cores, 56 threads), 124 GB; 2x RTX A5000 (sm_86, 24 GB), driver 590.48.01, CUDA 13.1; Debian 12, kernel 6.1 |

### 7.2 The corpus, exported by each original

The corpus was exported in M1a with MOSP-OpenMP@c352151's tools, and cross-checked once against
MOSP-CUDA (`M1a-crosscheck-mosp-cuda-e220ee2.json`). In this step `parity/export_goldens.py
--reference MOSP-CUDA --compare-to` exported the **whole corpus again with MOSP-CUDA@e220ee2's own
tools** (its `bin/main` test-case generator, `mospTest` with the packing-boundary oracle,
`stressTest 1` and `parallelStressTest 2` (100/100 passed each), `mospPrep`, `bin/mosp --validate`
running `sospUpdateGpu`, and the file-based `sospUpdateGpu` / sequential export tool built against
the CUDA sources) on GPU 1, and compared it with the committed corpus: **the same 11,799 files with
the same bytes** (manifest of the CUDA export `a68e1abc...`; `case.json` differs only in the
descriptive fields that name the reference), and `bin/mosp`'s `invalidated` counters equal
MOSP-OpenMP's on every case (`M1b-goldens-mosp-cuda-export.json`). So "byte-identical to the
goldens" is byte parity with MOSP-CUDA e220ee2 and with MOSP-OpenMP c352151 at once.

The M1b cases of PLAN 6.4.2 are groups of this corpus:

| PLAN 6.4.2 case | Group (M1a.md section 1 has the details) | Cases |
|---|---|---:|
| the 10 `generateTestCases` cases (MOSP-CUDA: from its `bin/main`) | `testcases` | 10 |
| count-to-infinity regressions (n = 6, seeds 621705 / 250813: d(1) = 90, not 60; two more seed pairs) | `regressions` | 3 |
| the thesis example; the ESCHER 4-vertex case (d = 100, 101), delete-all and ties; the graph/io fixtures | `thesis`, `escher`, `fixtures` | 1 + 4 + 17 |
| the `mospTest sosp` group | `sosp` | 148 |
| the large-weight distance-only fallback (320 x 320 grid, weights up to 2^31 - 1, safe and unsafe) | `large_weights` | 2 |
| the packing boundaries: n = 2^17 - 1 (47 distance bits) pull and push (MOSP-CUDA's `mospTest`), n = 2^16 + 1 with weights 2^31 - 1 (MOSP-OpenMP's) | `packing` | 3 |
| 100 `stressTest` + 100 `parallelStressTest` seeds | `stress` | 200 |
| the same inputs with non-canonical (perturbed tie-parent) initial trees | `noncanonical` | 107 |

### 7.3 Replay matrix (`compare.py`, port `0f0fba9`)

| Group | Cases | sequential | OpenMP 1 | OpenMP 4 | OpenMP 16 | OpenMP 28 | sequential int64 | OpenMP 4 int64 | CUDA | CUDA int64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| testcases | 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| regressions | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| thesis | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| escher | 4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| fixtures | 17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 |
| sosp | 148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 |
| large_weights | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| packing | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| stress | 200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 |
| noncanonical | 107 | 107/107 | 107/107 | 107/107 | 107/107 | 107/107 | 107/107 | 107/107 | 107/107 | 107/107 |
| **all** | **495** | **495/495** | **495/495** | **495/495** | **495/495** | **495/495** | **495/495** | **495/495** | **495/495** | **495/495** |

`int64` runs `dyng-compat-mosp --edge-type int64` (64-bit edge offsets), the other instantiation of
PLAN 4.4.3; its outputs are the same bytes. The CUDA columns ran on GPU 1 with the `parity-cuda`
build; `ci/gpu_local.sh` replays the `cuda` column again at every GPU gate (it did, green, at
`0f0fba9`, next to `ctest -L gpu` and compute-sanitizer memcheck).

**Cross-backend equality beyond the corpus** (PLAN 6.4.2): `dyng_sssp_cuda_tests` (label `gpu`)
compares cuda = openmp = sequential on randomized graphs and batches for all three index types,
from canonical and perturbed input trees, against `testing::dijkstra` and
`check_sssp_tree(require_canonical)`, and runs the hand cases and the 34 MOSP fixtures on cuda;
`ci/gpu_local.sh` ran it (and the CPU suites of the CUDA build) green on the final code.

**Paper-scale outputs.** Before the timed rounds of every graph x batch, `perf_ab.py run` writes
the updated trees of the original and of the port once and requires the same bytes: 12 / 12 on
CUDA and 12 / 12 on OpenMP; the `edge-type` runs require the same of int32 and int64 (24 / 24).
The `invalidated` counters were equal in every timed round (21 x 12 per backend).

## 8. The CUDA performance gate (PLAN 8.6, 6.4.2)

### 8.1 Protocol

`flock $DYNG_SCRATCH/perf.lock parity/perf_ab.py run --backend cuda --gpu 0 --exe
build/parity-cuda/tools/compat/dyng-compat-mosp --graph <g> --runs 21`: the unpatched
MOSP-CUDA@e220ee2 `bin/mosp` (A, rebuilt and verified before timing) and `dyng-compat-mosp
--backend cuda` (B) alternate A/B/A/B, one process per run as the original's bench runs it, 21
rounds per batch, medians compared; GPU 0 (nothing else on it), `CUDA_MODULE_LOADING=EAGER` on
both sides (dynG also calls `resources::warm_up()`), `--no-output` on both. Same inputs as the
OpenMP gate (MOSP's own `mospPrep`, K = 3 weights in [1, 100] seed 12345, 50K safe / 50K unsafe /
10K local batches with seed 777, the local radii of the original's `bench/prepare.sh`). Regions
(`parity/timed_regions/sssp.toml`, `[reference.mosp_cuda]`):

- `sosp_update objK` (gated per objective: <= 1.05x, <= 1.10x when the original's median is below
  10 ms; every run has >= 20 rounds): MOSP-CUDA's `obj<k> SOSP update` timer (`sospUpdateGpu()`,
  host time up to its final synchronization) against dynG's `sssp.enact_fused` of objective k,
  the same scope (host time; dynG's CUDA-event time of the same stage is the last column, for
  orientation; PLAN 8.6 names CUDA events or kernel sums for regions under 10 ms, but the
  original has no device timer, so both sides are read as host times, ADR 0017 deviation table).
- `apply` (<= 1.10x): the original's `apply batch` + `upload` stages (host apply, the device
  graph with its reverse CSR built on the device, the K trees, the change lists, the workspace
  allocations) against dynG's `update.commit` (host apply + the device copy of the updated graph)
  plus every result's tree upload, workspace lease and change-list upload.
- `end_to_end` (<= 1.10x): the original's `end_to_end_ms` minus its combined-graph step (`mosp`,
  0.2) against dynG's whole run.

Next to it, the **controlled-clock reading** of ADR 0018: `perf_ab.py kernels` runs both programs
A/B/A/B under Nsight Compute 2025.4.1 with `--clock-control base --cache-control none` (SM clock
locked to 1.17 GHz, memory 7.59 GHz, on both sides) and compares the per-objective fused kernels
(`gpu__time_duration`, DRAM bytes, registers, grid, occupancy limits), 21 rounds.

### 8.2 The GPU clock state (ADR 0018)

The RTX A5000 on GPU 0 picks its performance state by itself (nvidia-smi, this step): idle **P8**
(SM 210 MHz, memory 405 MHz); a process with a context and little GPU work **P2** (SM up to
1695 MHz, memory 7601 MHz); after about half a second of sustained work **P0** (SM 1920 MHz,
memory 8001 MHz), back to P2 within about a second of idling. Both programs spend seconds on the
host (reading the text inputs, the host apply) with the GPU idle, so the state of the timed
kernels depends on the GPU work right before them. Nsight Systems with GPU metrics (20 kHz),
road_usa's 10K local batch, three alternating runs of each program (`nsys profile --trace=cuda
--gpu-metrics-devices=0`; the reports are in `$DYNG_SCRATCH/runs/m1b-final/nsys`), the average
GPC clock during each fused kernel:

| Run | GPU busy in the 600 ms before objective 0 | obj0 | obj1 | obj2 |
|---|---:|---|---|---|
| MOSP-CUDA 1 | 256 ms | 22.51 ms @ 1.694 GHz | 20.63 ms @ 1.842 GHz | 20.65 ms @ 1.893 GHz |
| dynG 1 | 153 ms | 22.49 ms @ 1.694 GHz | 21.90 ms @ 1.694 GHz | 22.00 ms @ 1.694 GHz |
| MOSP-CUDA 2 | 256 ms | 20.97 ms @ 1.896 GHz | 20.52 ms @ 1.890 GHz | 20.78 ms @ 1.887 GHz |
| dynG 2 | 139 ms | 22.48 ms @ 1.694 GHz | 21.82 ms @ 1.694 GHz | 22.06 ms @ 1.694 GHz |
| MOSP-CUDA 3 | 258 ms | 20.97 ms @ 1.896 GHz | 20.50 ms @ 1.895 GHz | 20.65 ms @ 1.895 GHz |
| dynG 3 | 140 ms | 22.20 ms @ 1.726 GHz | 21.18 ms @ 1.762 GHz | 20.90 ms @ 1.830 GHz |

At the same clock the kernels take the same time (MOSP-CUDA's objective 0 of run 1 at 1.694 GHz:
22.51 ms; dynG's at 1.694 GHz: 22.49 ms; dynG's objective 2 of run 3 at 1.83 GHz: 20.90 ms). The
original runs its whole "upload" stage (the device graph with the reverse CSR built on the
device, the K trees, the change lists, the workspace; about 470 ms, 256 ms of it GPU work in the
last 600 ms) right before objective 0, so its kernels usually start in P0; dynG uploads the K
trees earlier (in `result::from_arrays`, before the host apply) and its graph upload inside the
commit is shorter (140-150 ms of GPU work), so its kernels usually start in P2. On the 50K
batches the kernels are long enough (25-113 ms each) to reach P0 early and the effect disappears
in the medians; on road_usa's local batch (about 21 ms per kernel after a long idle host phase) it
is the whole difference. ADR 0018 records why nothing is moved to change it and the open
decision; the locked-clock reading (8.4) is the kernel comparison without it.

### 8.3 Results, as measured (the gate)

#### roadNet-PA (`M1b-final-perf-cuda-roadNet-PA.json`; port `0f0fba9`)

| Batch | Region | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | Spread A / B | dynG device (ms) |
|---|---|---:|---:|---:|---|---|---:|
| safe50k | sosp_update obj0 | 4.74 | 4.64 | 0.979 | <= 1.10 ok | 3 % / 2 % | 4.63 |
|  | sosp_update obj1 | 4.71 | 4.69 | 0.994 | <= 1.10 ok | 2 % / 2 % | 4.68 |
|  | sosp_update obj2 | 4.80 | 4.77 | 0.994 | <= 1.10 ok | 2 % / 2 % | 4.76 |
|  | sosp_total | 14.26 | 14.10 | 0.988 | - | 2 % / 2 % | - |
|  | apply | 80.90 | 49.33 | 0.610 | <= 1.10 ok | 5 % / 12 % | - |
|  | end_to_end | 560.04 | 479.13 | 0.856 | <= 1.10 ok | 24 % / 4 % | - |
| unsafe50k | sosp_update obj0 | 4.71 | 4.61 | 0.980 | <= 1.10 ok | 2 % / 2 % | 4.61 |
|  | sosp_update obj1 | 4.68 | 4.67 | 0.997 | <= 1.10 ok | 2 % / 2 % | 4.66 |
|  | sosp_update obj2 | 4.80 | 4.78 | 0.994 | <= 1.10 ok | 2 % / 2 % | 4.77 |
|  | sosp_total | 14.19 | 14.07 | 0.992 | - | 2 % / 2 % | - |
|  | apply | 81.00 | 50.53 | 0.624 | <= 1.10 ok | 8 % / 11 % | - |
|  | end_to_end | 559.63 | 482.45 | 0.862 | <= 1.10 ok | 23 % / 6 % | - |
| local10k | sosp_update obj0 | 9.30 | 9.28 | 0.999 | <= 1.10 ok | 4 % / 4 % | 9.28 |
|  | sosp_update obj1 | 9.42 | 9.42 | 1.000 | <= 1.10 ok | 4 % / 4 % | 9.41 |
|  | sosp_update obj2 | 9.19 | 9.20 | 1.001 | <= 1.10 ok | 4 % / 4 % | 9.19 |
|  | sosp_total | 27.91 | 27.91 | 1.000 | - | 4 % / 4 % | - |
|  | apply | 67.40 | 37.24 | 0.553 | <= 1.10 ok | 8 % / 12 % | - |
|  | end_to_end | 553.88 | 480.38 | 0.867 | <= 1.10 ok | 14 % / 5 % | - |

safe50k: invalidated [1049304, 1071591, 1001669] (equal in every sample: True); load 14.4-18.1; runs 21
unsafe50k: invalidated [1049383, 1071640, 1001814] (equal in every sample: True); load 9.3-13.3; runs 21
local10k: invalidated [532189, 482776, 371509] (equal in every sample: True); load 8.0-9.6; runs 21

#### roadNet-CA (`M1b-final-perf-cuda-roadNet-CA.json`; port `0f0fba9`)

| Batch | Region | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | Spread A / B | dynG device (ms) |
|---|---|---:|---:|---:|---|---|---:|
| safe50k | sosp_update obj0 | 8.64 | 8.64 | 1.000 | <= 1.10 ok | 2 % / 1 % | 8.64 |
|  | sosp_update obj1 | 8.71 | 8.77 | 1.006 | <= 1.10 ok | 1 % / 1 % | 8.76 |
|  | sosp_update obj2 | 8.61 | 8.66 | 1.006 | <= 1.10 ok | 2 % / 1 % | 8.65 |
|  | sosp_total | 25.97 | 26.08 | 1.004 | - | 1 % / 1 % | - |
|  | apply | 113.90 | 79.54 | 0.698 | <= 1.10 ok | 5 % / 4 % | - |
|  | end_to_end | 753.97 | 679.24 | 0.901 | <= 1.10 ok | 26 % / 12 % | - |
| unsafe50k | sosp_update obj0 | 8.65 | 8.68 | 1.004 | <= 1.10 ok | 1 % / 1 % | 8.68 |
|  | sosp_update obj1 | 8.73 | 8.79 | 1.008 | <= 1.10 ok | 4 % / 1 % | 8.79 |
|  | sosp_update obj2 | 8.57 | 8.67 | 1.012 | <= 1.10 ok | 1 % / 1 % | 8.67 |
|  | sosp_total | 25.95 | 26.15 | 1.008 | - | 1 % / 1 % | - |
|  | apply | 112.20 | 80.33 | 0.716 | <= 1.10 ok | 5 % / 7 % | - |
|  | end_to_end | 793.89 | 679.71 | 0.856 | <= 1.10 ok | 41 % / 6 % | - |
| local10k | sosp_update obj0 | 3.49 | 3.44 | 0.985 | <= 1.10 ok | 3 % / 3 % | 3.43 |
|  | sosp_update obj1 | 3.08 | 3.10 | 1.006 | <= 1.10 ok | 3 % / 3 % | 3.09 |
|  | sosp_update obj2 | 3.43 | 3.45 | 1.005 | <= 1.10 ok | 3 % / 3 % | 3.44 |
|  | sosp_total | 10.00 | 9.99 | 0.999 | - | 3 % / 3 % | - |
|  | apply | 99.70 | 64.98 | 0.652 | <= 1.10 ok | 6 % / 20 % | - |
|  | end_to_end | 774.63 | 648.10 | 0.837 | <= 1.10 ok | 26 % / 5 % | - |

safe50k: invalidated [1651121, 1105632, 1022434] (equal in every sample: True); load 3.9-5.7; runs 21
unsafe50k: invalidated [1651536, 1106633, 1023496] (equal in every sample: True); load 3.3-4.3; runs 21
local10k: invalidated [97792, 65160, 89277] (equal in every sample: True); load 3.1-4.1; runs 21

#### rgg_n_2_20_s0 (`M1b-final-perf-cuda-rgg.json`; port `0f0fba9`)

| Batch | Region | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | Spread A / B | dynG device (ms) |
|---|---|---:|---:|---:|---|---|---:|
| safe50k | sosp_update obj0 | 25.09 | 24.98 | 0.996 | <= 1.05 ok | 3 % / 3 % | 24.98 |
|  | sosp_update obj1 | 24.70 | 24.62 | 0.997 | <= 1.05 ok | 3 % / 3 % | 24.61 |
|  | sosp_update obj2 | 25.37 | 25.22 | 0.994 | <= 1.05 ok | 3 % / 3 % | 25.22 |
|  | sosp_total | 75.13 | 74.81 | 0.996 | - | 3 % / 3 % | - |
|  | apply | 169.10 | 124.57 | 0.737 | <= 1.10 ok | 4 % / 4 % | - |
|  | end_to_end | 1240.69 | 1091.99 | 0.880 | <= 1.10 ok | 35 % / 2 % | - |
| unsafe50k | sosp_update obj0 | 25.15 | 25.00 | 0.994 | <= 1.05 ok | 2 % / 2 % | 24.99 |
|  | sosp_update obj1 | 24.76 | 24.66 | 0.996 | <= 1.05 ok | 3 % / 2 % | 24.65 |
|  | sosp_update obj2 | 25.42 | 25.28 | 0.994 | <= 1.05 ok | 2 % / 2 % | 25.27 |
|  | sosp_total | 75.28 | 74.94 | 0.996 | - | 2 % / 2 % | - |
|  | apply | 168.90 | 125.11 | 0.741 | <= 1.10 ok | 4 % / 3 % | - |
|  | end_to_end | 1232.21 | 1094.21 | 0.888 | <= 1.10 ok | 34 % / 2 % | - |
| local10k | sosp_update obj0 | 56.16 | 56.01 | 0.997 | <= 1.05 ok | 0 % / 0 % | 56.00 |
|  | sosp_update obj1 | 58.33 | 58.19 | 0.998 | <= 1.05 ok | 0 % / 0 % | 58.18 |
|  | sosp_update obj2 | 56.25 | 56.15 | 0.998 | <= 1.05 ok | 1 % / 0 % | 56.15 |
|  | sosp_total | 170.74 | 170.35 | 0.998 | - | 0 % / 0 % | - |
|  | apply | 141.50 | 98.65 | 0.697 | <= 1.10 ok | 4 % / 4 % | - |
|  | end_to_end | 1263.92 | 1165.45 | 0.922 | <= 1.10 ok | 31 % / 2 % | - |

safe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 2.9-5.5; runs 21
unsafe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 1.9-2.9; runs 21
local10k: invalidated [274665, 184671, 284403] (equal in every sample: True); load 1.7-1.9; runs 21

#### road_usa (`M1b-final-perf-cuda-road_usa_g.json`; port `0f0fba9`)

| Batch | Region | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | Spread A / B | dynG device (ms) |
|---|---|---:|---:|---:|---|---|---:|
| safe50k | sosp_update obj0 | 101.36 | 102.49 | 1.011 | <= 1.05 ok | 1 % / 1 % | 102.49 |
|  | sosp_update obj1 | 100.37 | 100.63 | 1.003 | <= 1.05 ok | 1 % / 1 % | 100.62 |
|  | sosp_update obj2 | 100.70 | 100.62 | 0.999 | <= 1.05 ok | 1 % / 1 % | 100.62 |
|  | sosp_total | 302.47 | 303.53 | 1.004 | - | 0 % / 1 % | - |
|  | apply | 1056.40 | 756.16 | 0.716 | <= 1.10 ok | 5 % / 5 % | - |
|  | end_to_end | 6294.15 | 5253.93 | 0.835 | <= 1.10 ok | 34 % / 9 % | - |
| unsafe50k | sosp_update obj0 | 101.23 | 102.09 | 1.008 | <= 1.05 ok | 1 % / 1 % | 102.08 |
|  | sosp_update obj1 | 100.36 | 100.42 | 1.001 | <= 1.05 ok | 1 % / 1 % | 100.42 |
|  | sosp_update obj2 | 100.74 | 100.54 | 0.998 | <= 1.05 ok | 1 % / 1 % | 100.53 |
|  | sosp_total | 302.30 | 303.12 | 1.003 | - | 1 % / 1 % | - |
|  | apply | 1065.20 | 757.72 | 0.711 | <= 1.10 ok | 13 % / 20 % | - |
|  | end_to_end | 6361.26 | 5255.58 | 0.826 | <= 1.10 ok | 32 % / 5 % | - |
| local10k | sosp_update obj0 | 21.24 | 22.62 | 1.065 | <= 1.05 **EXCEEDED** | 8 % / 6 % | 22.61 |
|  | sosp_update obj1 | 20.64 | 21.94 | 1.063 | <= 1.05 **EXCEEDED** | 7 % / 5 % | 21.93 |
|  | sosp_update obj2 | 20.77 | 21.54 | 1.037 | <= 1.05 ok | 1 % / 6 % | 21.53 |
|  | sosp_total | 62.63 | 66.13 | 1.056 | - | 5 % / 6 % | - |
|  | apply | 1032.90 | 736.21 | 0.713 | <= 1.10 ok | 6 % / 47 % | - |
|  | end_to_end | 5924.93 | 4993.50 | 0.843 | <= 1.10 ok | 39 % / 21 % | - |

safe50k: invalidated [17043767, 15989663, 16311741] (equal in every sample: True); load 3.6-18.6; runs 21
unsafe50k: invalidated [17046218, 15994408, 16314393] (equal in every sample: True); load 4.2-75.5; runs 21
local10k: invalidated [261871, 254274, 230241] (equal in every sample: True); load 20.7-244.0; runs 21

### 8.4 Results at locked clocks (the fused kernels, Nsight Compute)

#### roadNet-PA (`M1b-final-kernels-cuda-roadNet-PA.json`)

| Batch | Kernel | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | DRAM bytes dynG / MOSP-CUDA | Spread A / B |
|---|---|---:|---:|---:|---|---:|---|
| safe50k | obj0 | 5.34 | 5.35 | 1.001 | <= 1.10 ok | 1.010 | 1.0 % / 0.9 % |
|  | obj1 | 5.47 | 5.48 | 1.002 | <= 1.10 ok | 1.009 | 0.6 % / 0.8 % |
|  | obj2 | 5.58 | 5.56 | 0.997 | <= 1.10 ok | 1.010 | 1.3 % / 1.4 % |
| unsafe50k | obj0 | 5.33 | 5.34 | 1.002 | <= 1.10 ok | 1.010 | 1.1 % / 1.0 % |
|  | obj1 | 5.46 | 5.46 | 0.999 | <= 1.10 ok | 1.010 | 0.7 % / 0.8 % |
|  | obj2 | 5.56 | 5.55 | 0.998 | <= 1.10 ok | 1.010 | 1.2 % / 0.9 % |
| local10k | obj0 | 12.65 | 12.69 | 1.003 | <= 1.05 ok | 1.027 | 0.5 % / 0.3 % |
|  | obj1 | 12.93 | 12.97 | 1.003 | <= 1.05 ok | 1.028 | 0.7 % / 0.5 % |
|  | obj2 | 12.61 | 12.64 | 1.003 | <= 1.05 ok | 1.025 | 0.3 % / 0.4 % |

#### roadNet-CA (`M1b-final-kernels-cuda-roadNet-CA.json`)

| Batch | Kernel | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | DRAM bytes dynG / MOSP-CUDA | Spread A / B |
|---|---|---:|---:|---:|---|---:|---|
| safe50k | obj0 | 9.73 | 9.81 | 1.008 | <= 1.10 ok | 1.009 | 0.7 % / 0.9 % |
|  | obj1 | 9.93 | 10.00 | 1.008 | <= 1.10 ok | 1.009 | 1.1 % / 0.8 % |
|  | obj2 | 9.80 | 9.87 | 1.007 | <= 1.10 ok | 1.009 | 0.8 % / 1.1 % |
| unsafe50k | obj0 | 9.71 | 9.79 | 1.008 | <= 1.10 ok | 1.008 | 0.8 % / 0.8 % |
|  | obj1 | 9.90 | 10.03 | 1.014 | <= 1.10 ok | 1.009 | 1.0 % / 0.9 % |
|  | obj2 | 9.77 | 9.86 | 1.010 | <= 1.10 ok | 1.003 | 1.0 % / 0.8 % |
| local10k | obj0 | 4.61 | 4.58 | 0.993 | <= 1.10 ok | 1.016 | 0.6 % / 0.9 % |
|  | obj1 | 4.19 | 4.16 | 0.992 | <= 1.10 ok | 1.005 | 1.0 % / 0.6 % |
|  | obj2 | 4.71 | 4.67 | 0.993 | <= 1.10 ok | 1.006 | 0.8 % / 0.9 % |

#### rgg_n_2_20_s0 (`M1b-final-kernels-cuda-rgg.json`)

| Batch | Kernel | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | DRAM bytes dynG / MOSP-CUDA | Spread A / B |
|---|---|---:|---:|---:|---|---:|---|
| safe50k | obj0 | 29.01 | 28.89 | 0.996 | <= 1.05 ok | 1.002 | 0.6 % / 0.7 % |
|  | obj1 | 28.70 | 28.47 | 0.992 | <= 1.05 ok | 1.003 | 0.5 % / 0.7 % |
|  | obj2 | 29.36 | 29.14 | 0.993 | <= 1.05 ok | 1.001 | 0.6 % / 0.4 % |
| unsafe50k | obj0 | 29.01 | 28.86 | 0.995 | <= 1.05 ok | 1.002 | 0.5 % / 0.8 % |
|  | obj1 | 28.71 | 28.47 | 0.992 | <= 1.05 ok | 1.002 | 0.7 % / 0.7 % |
|  | obj2 | 29.33 | 29.11 | 0.993 | <= 1.05 ok | 1.000 | 0.6 % / 0.6 % |
| local10k | obj0 | 84.59 | 84.64 | 1.001 | <= 1.05 ok | 1.007 | 0.3 % / 0.3 % |
|  | obj1 | 87.88 | 88.02 | 1.002 | <= 1.05 ok | 1.006 | 0.2 % / 0.2 % |
|  | obj2 | 84.78 | 84.89 | 1.001 | <= 1.05 ok | 1.009 | 0.2 % / 0.2 % |

#### road_usa (`M1b-final-kernels-cuda-road_usa_g.json`)

| Batch | Kernel | MOSP-CUDA (ms) | dynG (ms) | Ratio | Gate | DRAM bytes dynG / MOSP-CUDA | Spread A / B |
|---|---|---:|---:|---:|---|---:|---|
| safe50k | obj0 | 112.43 | 113.64 | 1.011 | <= 1.05 ok | 1.007 | 0.5 % / 0.6 % |
|  | obj1 | 111.38 | 112.53 | 1.010 | <= 1.05 ok | 1.004 | 0.5 % / 0.5 % |
|  | obj2 | 111.69 | 112.93 | 1.011 | <= 1.05 ok | 1.005 | 0.4 % / 0.6 % |
| unsafe50k | obj0 | 112.47 | 113.58 | 1.010 | <= 1.05 ok | 1.007 | 0.5 % / 0.5 % |
|  | obj1 | 111.38 | 112.58 | 1.011 | <= 1.05 ok | 1.004 | 0.5 % / 0.4 % |
|  | obj2 | 111.76 | 112.90 | 1.010 | <= 1.05 ok | 1.005 | 0.5 % / 0.4 % |
| local10k | obj0 | 29.15 | 29.02 | 0.995 | <= 1.05 ok | 1.000 | 0.9 % / 0.4 % |
|  | obj1 | 28.34 | 28.12 | 0.992 | <= 1.05 ok | 0.987 | 0.3 % / 0.4 % |
|  | obj2 | 28.59 | 28.40 | 0.993 | <= 1.05 ok | 0.989 | 0.4 % / 0.4 % |

### 8.5 Verdict (CUDA)

- **Per-objective SOSP region, as measured: 34 of 36 readings within the gate** (0.979-1.012x on
  roadNet-PA, roadNet-CA and rgg_n_2_20_s0 and on road_usa's 50K batches, 21 rounds each).
  **road_usa's local 10K batch exceeds on objectives 0 and 1 (1.065x, 1.063x; objective 2
  1.037x): FAIL** under PLAN 8.6 as written. The cause is the GPU's performance state during
  the timed kernels (section 8.2), not the kernel: at locked clocks the same kernels read
  0.992-0.995x (8.4), and at equal clocks the nsys-profiled kernels take the same time. ADR 0018
  proposes a verdict rule that would pass this reading (different P-states recorded, the
  controlled reading within the gate, end to end within the gate: 0.84x) and leaves the choice
  to the author: (A) one A/B with the clocks locked by root, (B) accept the rule, or (C) keep the
  miss as a known M1b gate miss. Until then this certificate reports it as a FAIL.
- **Fused kernels at locked clocks: 36 of 36 within the gate**, 0.992-1.014x, DRAM bytes
  0.987-1.028x (the `affected` count), the same registers, grid and occupancy limits (section
  11).
- **apply 0.55-0.74x and end to end 0.83-0.92x**: within the 1.10x gates everywhere (dynG's host
  apply is parallel; MOSP-CUDA's `applyChangeBatch` is sequential).
- Load averages above 20 in some records (up to 244) come from kernel CIFS threads of a network
  mount in uninterruptible sleep (`cifsd` in state D), which count toward the load average but use
  no CPU; the run queue showed no other runnable work.

## 9. The OpenMP gate on the final code, and the barrier finding

### 9.1 What moved: road_usa's 10K local batch

PLAN 6.4.2 names the watch item: "the local 10K batch (sensitive to barrier count)". In the
first gate campaign of this step (`2b39fac`, the straight port of step cpu-gates plus the CUDA
work; that record was superseded and is not kept) road_usa's local batch read **1.12x** on
objectives 1 and 2 (69.7 / 70.4 ms against 62.3 / 62.4 ms; objective 0 0.97x), and again at
`3315aba` (1.115 / 1.125x, `M1b-diag-openmp-road_usa_g-before-gather-pair.json`), while the same
code had read 0.96x in step cpu-gates (section 3.4). Every other OpenMP reading was within its
gate. Investigation (road_usa,
10K local batch, 28 threads, `dyng-compat-mosp --timing` and the original's report lines):

- **Per process, the port's loop ran in one of two modes**: `sssp.loop` 28-29 ms or 37-39 ms per
  objective, the same mode for all three objectives of a process; `identify_affected` (27 ms),
  `seed` and `finalize` (5 ms) did not move. The original has the same two modes (62 or 72-74 ms
  per objective); in the A/B campaigns it was mostly in the fast one, the port mostly in the slow
  one, and the share drifted over hours with the machine's state.
- Not the memory layout: the modes persist with ASLR off (`setarch -R`), with transparent huge
  pages off for the process (`PR_SET_THP_DISABLE`), and whatever the huge-page backing of the big
  arrays was (`/proc/<pid>/smaps`). Not the power budget: idling 3 or 10 s before the update (so
  that the 10 s RAPL window is fresh) did not choose the mode. Keeping the per-thread lists on
  their own cache lines (`9439733`, util/thread_list.hpp) did not change it either
  (1.118 / 1.119x, `M1b-diag-openmp-road_usa_g-thread-lists-only.json`).
- `OMP_WAIT_POLICY=active` put **both** programs in the slow mode in every run (port 68-69 ms,
  original 73-74 ms per objective); `passive` made both 4x slower (futex wake-ups).
- **Where the time goes** (the loop instrumented per thread, a scratch build): of 36 ms, the
  work-sharing loops took 3-5 ms and the two list gathers 19-20 ms. A local batch leaves about
  455 vertices per near-far round (500K pushes in about 1,100 rounds per objective), i.e. about
  seven chunks of 64 for 28 threads, and each round passed **six barriers** (the `omp for`, two in
  each `list_gather::gather()`, the region's): about 6,600 barriers per objective, whose latency
  (a few microseconds, and higher in the slow mode) is most of the loop.

**The change** (`0f0fba9`): the work-sharing loops that feed a gather are `nowait` (the gather's
first barrier already waits for every thread's part), and the regions that gather two lists (the
near/far split, every near-far round, the far re-split) use the new `list_gather::gather_pair()`,
which does both prefix sums between one pair of barriers: **three barriers per round**. The lists,
their order and every output are unchanged (the golden replay, section 7.3). Result on the same
workload: `sssp.loop` 19-20 ms per objective in every run (no slow mode seen since), SOSP region
50-52 ms against the original's 62 ms (6 alternating runs; the 21-round gate record is below). As
PLAN 8.6 asks, this improvement is reported separately from the straight port: the straight
port's gate records are section 3 (`79400d2`, within every gate) and the two diagnostic records
above (road_usa local 10K exceeded); the gate record below is the shipped code.

### 9.2 Results (the gate; port `0f0fba9`)

Protocol as in section 3 (`perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp
--graph <g> --runs 21`, 28 threads with `OMP_PROC_BIND=close OMP_PLACES=cores` on both sides,
the default wait policy, the unpatched MOSP-OpenMP@c352151).

#### roadNet-PA (`M1b-final-perf-openmp-roadNet-PA.json`; port `0f0fba9`)

| Batch | Region | MOSP-OpenMP (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 16.46 | 14.15 | 0.859 | <= 1.05 ok | 25 % / 13 % |
|  | sosp_update obj1 | 12.64 | 10.66 | 0.843 | <= 1.05 ok | 14 % / 5 % |
|  | sosp_update obj2 | 12.58 | 10.79 | 0.857 | <= 1.05 ok | 9 % / 4 % |
|  | sosp_total | 41.74 | 35.67 | 0.855 | - | 16 % / 6 % |
|  | apply | 76.10 | 70.10 | 0.921 | <= 1.10 ok | 24 % / 22 % |
|  | end_to_end | 376.46 | 315.64 | 0.838 | <= 1.10 ok | 37 % / 5 % |
| unsafe50k | sosp_update obj0 | 16.62 | 14.06 | 0.846 | <= 1.05 ok | 24 % / 8 % |
|  | sosp_update obj1 | 12.65 | 10.47 | 0.828 | <= 1.05 ok | 9 % / 3 % |
|  | sosp_update obj2 | 12.66 | 10.66 | 0.842 | <= 1.05 ok | 10 % / 5 % |
|  | sosp_total | 42.00 | 35.12 | 0.836 | - | 11 % / 4 % |
|  | apply | 76.70 | 78.39 | 1.022 | <= 1.10 ok | 33 % / 19 % |
|  | end_to_end | 353.20 | 320.26 | 0.907 | <= 1.10 ok | 37 % / 10 % |
| local10k | sosp_update obj0 | 27.98 | 18.45 | 0.659 | <= 1.05 ok | 22 % / 19 % |
|  | sosp_update obj1 | 25.15 | 15.90 | 0.632 | <= 1.05 ok | 21 % / 18 % |
|  | sosp_update obj2 | 24.81 | 15.62 | 0.630 | <= 1.05 ok | 20 % / 18 % |
|  | sosp_total | 78.42 | 49.70 | 0.634 | - | 20 % / 16 % |
|  | apply | 66.10 | 56.19 | 0.850 | <= 1.10 ok | 25 % / 28 % |
|  | end_to_end | 381.42 | 317.86 | 0.833 | <= 1.10 ok | 28 % / 8 % |

safe50k: invalidated [1049304, 1071591, 1001669] (equal in every sample: True); load 7.1-8.8; runs 21
unsafe50k: invalidated [1049383, 1071640, 1001814] (equal in every sample: True); load 6.9-7.8; runs 21
local10k: invalidated [532189, 482776, 371509] (equal in every sample: True); load 5.6-6.9; runs 21

#### roadNet-CA (`M1b-final-perf-openmp-roadNet-CA.json`; port `0f0fba9`)

| Batch | Region | MOSP-OpenMP (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 33.96 | 24.55 | 0.723 | <= 1.05 ok | 25 % / 9 % |
|  | sosp_update obj1 | 20.92 | 19.05 | 0.910 | <= 1.05 ok | 11 % / 5 % |
|  | sosp_update obj2 | 20.39 | 18.79 | 0.921 | <= 1.05 ok | 8 % / 5 % |
|  | sosp_total | 75.07 | 62.28 | 0.830 | - | 15 % / 5 % |
|  | apply | 139.00 | 124.58 | 0.896 | <= 1.10 ok | 8 % / 11 % |
|  | end_to_end | 638.00 | 537.51 | 0.842 | <= 1.10 ok | 34 % / 7 % |
| unsafe50k | sosp_update obj0 | 31.38 | 24.52 | 0.781 | <= 1.05 ok | 22 % / 8 % |
|  | sosp_update obj1 | 21.08 | 18.95 | 0.899 | <= 1.05 ok | 8 % / 5 % |
|  | sosp_update obj2 | 20.45 | 18.63 | 0.911 | <= 1.05 ok | 9 % / 5 % |
|  | sosp_total | 73.31 | 62.22 | 0.849 | - | 13 % / 5 % |
|  | apply | 139.30 | 127.11 | 0.913 | <= 1.10 ok | 10 % / 9 % |
|  | end_to_end | 665.69 | 539.23 | 0.810 | <= 1.10 ok | 32 % / 4 % |
| local10k | sosp_update obj0 | 14.44 | 9.89 | 0.685 | <= 1.05 ok | 38 % / 19 % |
|  | sosp_update obj1 | 10.16 | 6.89 | 0.678 | <= 1.05 ok | 24 % / 17 % |
|  | sosp_update obj2 | 9.32 | 7.14 | 0.766 | <= 1.10 ok | 23 % / 17 % |
|  | sosp_total | 33.66 | 23.89 | 0.710 | - | 28 % / 16 % |
|  | apply | 124.90 | 111.06 | 0.889 | <= 1.10 ok | 12 % / 11 % |
|  | end_to_end | 590.09 | 481.64 | 0.816 | <= 1.10 ok | 33 % / 3 % |

safe50k: invalidated [1651121, 1105632, 1022434] (equal in every sample: True); load 17.3-21.7; runs 21
unsafe50k: invalidated [1651536, 1106633, 1023496] (equal in every sample: True); load 12.8-18.2; runs 21
local10k: invalidated [97792, 65160, 89277] (equal in every sample: True); load 11.0-13.3; runs 21

#### rgg_n_2_20_s0 (`M1b-final-perf-openmp-rgg.json`; port `0f0fba9`)

| Batch | Region | MOSP-OpenMP (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 51.72 | 51.09 | 0.988 | <= 1.05 ok | 6 % / 5 % |
|  | sosp_update obj1 | 40.13 | 38.04 | 0.948 | <= 1.05 ok | 2 % / 5 % |
|  | sosp_update obj2 | 44.50 | 42.23 | 0.949 | <= 1.05 ok | 3 % / 4 % |
|  | sosp_total | 136.36 | 131.51 | 0.964 | - | 4 % / 4 % |
|  | apply | 227.70 | 184.52 | 0.810 | <= 1.10 ok | 10 % / 6 % |
|  | end_to_end | 1129.27 | 1019.43 | 0.903 | <= 1.10 ok | 39 % / 2 % |
| unsafe50k | sosp_update obj0 | 51.68 | 51.13 | 0.989 | <= 1.05 ok | 3 % / 7 % |
|  | sosp_update obj1 | 40.17 | 38.02 | 0.946 | <= 1.05 ok | 2 % / 3 % |
|  | sosp_update obj2 | 44.59 | 42.16 | 0.946 | <= 1.05 ok | 2 % / 3 % |
|  | sosp_total | 136.66 | 131.59 | 0.963 | - | 2 % / 4 % |
|  | apply | 228.20 | 182.53 | 0.800 | <= 1.10 ok | 9 % / 6 % |
|  | end_to_end | 1231.52 | 1016.54 | 0.825 | <= 1.10 ok | 39 % / 1 % |
| local10k | sosp_update obj0 | 79.23 | 55.55 | 0.701 | <= 1.05 ok | 24 % / 11 % |
|  | sosp_update obj1 | 69.16 | 46.77 | 0.676 | <= 1.05 ok | 19 % / 15 % |
|  | sosp_update obj2 | 77.79 | 54.72 | 0.704 | <= 1.05 ok | 24 % / 11 % |
|  | sosp_total | 226.61 | 156.82 | 0.692 | - | 22 % / 12 % |
|  | apply | 202.50 | 156.84 | 0.775 | <= 1.10 ok | 11 % / 9 % |
|  | end_to_end | 1177.12 | 1020.09 | 0.867 | <= 1.10 ok | 41 % / 2 % |

safe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 1.7-6.2; runs 21
unsafe50k: invalidated [878682, 549027, 690545] (equal in every sample: True); load 4.6-10.8; runs 21
local10k: invalidated [274665, 184671, 284403] (equal in every sample: True); load 9.4-13.8; runs 21

#### road_usa (`M1b-final-perf-openmp-road_usa_g.json`; port `0f0fba9`)

| Batch | Region | MOSP-OpenMP (ms) | dynG (ms) | Ratio | Gate | Spread A / B |
|---|---|---:|---:|---:|---|---|
| safe50k | sosp_update obj0 | 311.92 | 272.00 | 0.872 | <= 1.05 ok | 100 % / 8 % |
|  | sosp_update obj1 | 273.62 | 252.58 | 0.923 | <= 1.05 ok | 6 % / 9 % |
|  | sosp_update obj2 | 271.40 | 254.32 | 0.937 | <= 1.05 ok | 6 % / 9 % |
|  | sosp_total | 856.97 | 778.52 | 0.908 | - | 38 % / 9 % |
|  | apply | 1546.50 | 1218.33 | 0.788 | <= 1.10 ok | 51 % / 4 % |
|  | end_to_end | 7237.77 | 5562.84 | 0.769 | <= 1.10 ok | 43 % / 2 % |
| unsafe50k | sosp_update obj0 | 317.10 | 272.14 | 0.858 | <= 1.05 ok | 4 % / 4 % |
|  | sosp_update obj1 | 278.50 | 253.34 | 0.910 | <= 1.05 ok | 3 % / 6 % |
|  | sosp_update obj2 | 277.95 | 256.03 | 0.921 | <= 1.05 ok | 3 % / 7 % |
|  | sosp_total | 874.70 | 784.12 | 0.896 | - | 3 % / 6 % |
|  | apply | 1544.00 | 1216.72 | 0.788 | <= 1.10 ok | 4 % / 2 % |
|  | end_to_end | 6911.39 | 5560.25 | 0.805 | <= 1.10 ok | 24 % / 1 % |
| local10k | sosp_update obj0 | 75.61 | 57.81 | 0.765 | <= 1.05 ok | 22 % / 11 % |
|  | sosp_update obj1 | 62.36 | 55.35 | 0.888 | <= 1.05 ok | 18 % / 9 % |
|  | sosp_update obj2 | 62.74 | 55.68 | 0.887 | <= 1.05 ok | 19 % / 10 % |
|  | sosp_total | 199.98 | 169.20 | 0.846 | - | 19 % / 10 % |
|  | apply | 1529.30 | 1196.01 | 0.782 | <= 1.10 ok | 6 % / 1 % |
|  | end_to_end | 6533.07 | 4924.60 | 0.754 | <= 1.10 ok | 37 % / 2 % |

safe50k: invalidated [17043767, 15989663, 16311741] (equal in every sample: True); load 9.7-119.7; runs 21
unsafe50k: invalidated [17046218, 15994408, 16314393] (equal in every sample: True); load 9.7-116.0; runs 21
local10k: invalidated [261871, 254274, 230241] (equal in every sample: True); load 4.7-10.0; runs 21

**Verdict (OpenMP): every gate is met.** 36 of 36 per-objective SOSP readings within the gate
(0.630-0.989x; the local batches 0.63-0.89x), apply 0.775-1.022x, end to end 0.754-0.907x;
outputs byte-identical and `invalidated` equal in every round. Spreads above 10 % (flagged in the
JSON, not failed: PLAN 8.6) are mostly the original's (its input phase, one 100 % outlier round of
objective 0 on road_usa's safe batch) and the short, barrier-bound local batches on both sides;
the medians of 21 alternating rounds absorb them.

## 10. The `edge_t` benchmark (ADR 0009)

`parity/perf_ab.py edge-type --backend openmp|cuda --graph <g> --runs 11`: `dyng-compat-mosp
--edge-type int32` (A) against `--edge-type int64` (B), A/B/A/B on the inputs and regions of the
gate (same presets, lock, threads, GPU 0), medians of 11 rounds; outputs byte-identical (checked
once per batch) and `invalidated` equal in every round. The decision (int32 by default, checked
construction, int64 a first-class instantiation) is ADR 0009.

Ratio int64 / int32 per region (the summary table is in ADR 0009):

#### roadNet-PA, cuda (`M1b-edge-type-cuda-roadNet-PA.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 4.64 | 4.80 | 1.035 | 2 % / 2 % |
|  | sosp_update obj1 | 4.68 | 4.87 | 1.041 | 2 % / 2 % |
|  | sosp_update obj2 | 4.77 | 4.96 | 1.039 | 2 % / 2 % |
|  | sosp_total | 14.11 | 14.63 | 1.037 | 2 % / 2 % |
|  | apply | 49.93 | 49.05 | 0.982 | 11 % / 12 % |
|  | end_to_end | 470.53 | 466.70 | 0.992 | 3 % / 7 % |
| unsafe50k | sosp_update obj0 | 4.62 | 4.81 | 1.041 | 2 % / 2 % |
|  | sosp_update obj1 | 4.67 | 4.86 | 1.042 | 2 % / 1 % |
|  | sosp_update obj2 | 4.76 | 4.97 | 1.044 | 2 % / 2 % |
|  | sosp_total | 14.05 | 14.65 | 1.043 | 2 % / 2 % |
|  | apply | 50.25 | 49.61 | 0.987 | 12 % / 9 % |
|  | end_to_end | 475.65 | 468.55 | 0.985 | 5 % / 5 % |
| local10k | sosp_update obj0 | 9.16 | 9.28 | 1.014 | 5 % / 4 % |
|  | sosp_update obj1 | 9.28 | 9.41 | 1.014 | 5 % / 4 % |
|  | sosp_update obj2 | 9.07 | 9.19 | 1.013 | 4 % / 4 % |
|  | sosp_total | 27.50 | 27.88 | 1.014 | 5 % / 4 % |
|  | apply | 37.54 | 38.98 | 1.038 | 11 % / 9 % |
|  | end_to_end | 467.14 | 470.71 | 1.008 | 5 % / 5 % |

#### roadNet-CA, cuda (`M1b-edge-type-cuda-roadNet-CA.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 8.64 | 8.94 | 1.036 | 1 % / 1 % |
|  | sosp_update obj1 | 8.77 | 9.06 | 1.033 | 1 % / 2 % |
|  | sosp_update obj2 | 8.67 | 8.91 | 1.028 | 2 % / 1 % |
|  | sosp_total | 26.10 | 26.90 | 1.031 | 1 % / 1 % |
|  | apply | 80.87 | 77.88 | 0.963 | 7 % / 5 % |
|  | end_to_end | 672.27 | 665.08 | 0.989 | 3 % / 2 % |
| unsafe50k | sosp_update obj0 | 8.67 | 8.94 | 1.031 | 1 % / 2 % |
|  | sosp_update obj1 | 8.74 | 9.04 | 1.035 | 1 % / 2 % |
|  | sosp_update obj2 | 8.64 | 8.89 | 1.029 | 2 % / 2 % |
|  | sosp_total | 26.03 | 26.89 | 1.033 | 1 % / 1 % |
|  | apply | 80.81 | 78.10 | 0.967 | 5 % / 4 % |
|  | end_to_end | 668.94 | 663.48 | 0.992 | 3 % / 2 % |
| local10k | sosp_update obj0 | 3.36 | 3.39 | 1.009 | 8 % / 8 % |
|  | sosp_update obj1 | 3.03 | 3.05 | 1.005 | 8 % / 8 % |
|  | sosp_update obj2 | 3.37 | 3.40 | 1.010 | 8 % / 8 % |
|  | sosp_total | 9.76 | 9.84 | 1.008 | 8 % / 8 % |
|  | apply | 64.49 | 63.54 | 0.985 | 3 % / 3 % |
|  | end_to_end | 634.41 | 629.20 | 0.992 | 2 % / 3 % |

#### rgg_n_2_20_s0, cuda (`M1b-edge-type-cuda-rgg.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 24.97 | 25.99 | 1.041 | 2 % / 3 % |
|  | sosp_update obj1 | 24.60 | 25.71 | 1.045 | 3 % / 2 % |
|  | sosp_update obj2 | 25.26 | 26.33 | 1.042 | 2 % / 2 % |
|  | sosp_total | 74.82 | 78.00 | 1.042 | 2 % / 2 % |
|  | apply | 126.33 | 124.53 | 0.986 | 4 % / 3 % |
|  | end_to_end | 1088.27 | 1087.91 | 1.000 | 2 % / 2 % |
| unsafe50k | sosp_update obj0 | 25.05 | 26.01 | 1.038 | 3 % / 2 % |
|  | sosp_update obj1 | 24.67 | 25.73 | 1.043 | 2 % / 2 % |
|  | sosp_update obj2 | 25.35 | 26.34 | 1.039 | 2 % / 2 % |
|  | sosp_total | 75.06 | 78.10 | 1.040 | 2 % / 2 % |
|  | apply | 125.19 | 125.98 | 1.006 | 3 % / 4 % |
|  | end_to_end | 1089.11 | 1090.80 | 1.002 | 1 % / 3 % |
| local10k | sosp_update obj0 | 56.01 | 56.46 | 1.008 | 0 % / 1 % |
|  | sosp_update obj1 | 58.23 | 58.68 | 1.008 | 1 % / 1 % |
|  | sosp_update obj2 | 56.14 | 56.62 | 1.009 | 1 % / 1 % |
|  | sosp_total | 170.39 | 171.79 | 1.008 | 1 % / 0 % |
|  | apply | 96.19 | 96.84 | 1.007 | 5 % / 9 % |
|  | end_to_end | 1154.34 | 1153.24 | 0.999 | 3 % / 2 % |

#### road_usa, cuda (`M1b-edge-type-cuda-road_usa_g.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 102.30 | 103.84 | 1.015 | 2 % / 1 % |
|  | sosp_update obj1 | 100.26 | 101.95 | 1.017 | 1 % / 1 % |
|  | sosp_update obj2 | 100.38 | 101.94 | 1.016 | 0 % / 0 % |
|  | sosp_total | 303.21 | 307.64 | 1.015 | 1 % / 1 % |
|  | apply | 750.85 | 757.86 | 1.009 | 2 % / 2 % |
|  | end_to_end | 5218.56 | 5225.86 | 1.001 | 5 % / 1 % |
| unsafe50k | sosp_update obj0 | 102.29 | 103.69 | 1.014 | 1 % / 1 % |
|  | sosp_update obj1 | 100.55 | 102.21 | 1.017 | 1 % / 1 % |
|  | sosp_update obj2 | 100.52 | 102.39 | 1.019 | 0 % / 1 % |
|  | sosp_total | 303.64 | 308.27 | 1.015 | 1 % / 1 % |
|  | apply | 750.30 | 756.60 | 1.008 | 1 % / 1 % |
|  | end_to_end | 5204.71 | 5217.46 | 1.002 | 1 % / 1 % |
| local10k | sosp_update obj0 | 22.66 | 22.74 | 1.004 | 0 % / 5 % |
|  | sosp_update obj1 | 21.98 | 22.02 | 1.002 | 0 % / 5 % |
|  | sosp_update obj2 | 22.13 | 22.22 | 1.004 | 0 % / 6 % |
|  | sosp_total | 66.76 | 66.97 | 1.003 | 0 % / 5 % |
|  | apply | 728.35 | 734.51 | 1.008 | 1 % / 1 % |
|  | end_to_end | 4949.41 | 4955.83 | 1.001 | 1 % / 1 % |

#### roadNet-PA, openmp (`M1b-edge-type-openmp-roadNet-PA.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 14.14 | 14.33 | 1.013 | 9 % / 10 % |
|  | sosp_update obj1 | 10.60 | 10.75 | 1.013 | 2 % / 2 % |
|  | sosp_update obj2 | 10.68 | 10.87 | 1.018 | 2 % / 3 % |
|  | sosp_total | 35.39 | 35.97 | 1.017 | 3 % / 5 % |
|  | apply | 69.50 | 80.29 | 1.155 | 19 % / 18 % |
|  | end_to_end | 324.06 | 319.25 | 0.985 | 4 % / 7 % |
| unsafe50k | sosp_update obj0 | 13.89 | 13.80 | 0.994 | 7 % / 11 % |
|  | sosp_update obj1 | 10.45 | 10.60 | 1.015 | 4 % / 2 % |
|  | sosp_update obj2 | 10.62 | 10.79 | 1.016 | 2 % / 3 % |
|  | sosp_total | 35.05 | 35.38 | 1.009 | 4 % / 4 % |
|  | apply | 79.66 | 71.30 | 0.895 | 18 % / 18 % |
|  | end_to_end | 323.68 | 318.29 | 0.983 | 7 % / 4 % |
| local10k | sosp_update obj0 | 19.88 | 20.11 | 1.012 | 13 % / 15 % |
|  | sosp_update obj1 | 17.91 | 17.91 | 1.000 | 12 % / 13 % |
|  | sosp_update obj2 | 17.62 | 17.73 | 1.006 | 13 % / 14 % |
|  | sosp_total | 55.53 | 55.94 | 1.007 | 12 % / 13 % |
|  | apply | 67.91 | 65.99 | 0.972 | 22 % / 21 % |
|  | end_to_end | 330.44 | 328.81 | 0.995 | 7 % / 8 % |

#### roadNet-CA, openmp (`M1b-edge-type-openmp-roadNet-CA.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 24.31 | 24.49 | 1.007 | 7 % / 14 % |
|  | sosp_update obj1 | 19.01 | 19.84 | 1.044 | 5 % / 7 % |
|  | sosp_update obj2 | 18.67 | 19.19 | 1.028 | 4 % / 10 % |
|  | sosp_total | 62.09 | 64.46 | 1.038 | 5 % / 8 % |
|  | apply | 126.38 | 127.46 | 1.009 | 9 % / 10 % |
|  | end_to_end | 540.50 | 540.61 | 1.000 | 1 % / 2 % |
| unsafe50k | sosp_update obj0 | 24.91 | 25.70 | 1.032 | 7 % / 13 % |
|  | sosp_update obj1 | 19.03 | 19.44 | 1.021 | 7 % / 3 % |
|  | sosp_update obj2 | 18.62 | 19.20 | 1.031 | 5 % / 3 % |
|  | sosp_total | 62.24 | 64.28 | 1.033 | 7 % / 5 % |
|  | apply | 119.83 | 129.54 | 1.081 | 8 % / 10 % |
|  | end_to_end | 535.00 | 541.80 | 1.013 | 4 % / 2 % |
| local10k | sosp_update obj0 | 10.03 | 10.65 | 1.062 | 16 % / 24 % |
|  | sosp_update obj1 | 6.92 | 7.28 | 1.051 | 17 % / 16 % |
|  | sosp_update obj2 | 7.11 | 7.23 | 1.016 | 15 % / 17 % |
|  | sosp_total | 23.79 | 25.18 | 1.058 | 14 % / 19 % |
|  | apply | 104.51 | 112.31 | 1.075 | 11 % / 9 % |
|  | end_to_end | 477.99 | 485.52 | 1.016 | 5 % / 2 % |

#### rgg_n_2_20_s0, openmp (`M1b-edge-type-openmp-rgg.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 51.34 | 52.17 | 1.016 | 3 % / 3 % |
|  | sosp_update obj1 | 37.88 | 39.09 | 1.032 | 4 % / 2 % |
|  | sosp_update obj2 | 42.38 | 43.28 | 1.021 | 3 % / 2 % |
|  | sosp_total | 131.81 | 134.28 | 1.019 | 2 % / 1 % |
|  | apply | 181.26 | 190.43 | 1.051 | 6 % / 5 % |
|  | end_to_end | 1023.42 | 1029.89 | 1.006 | 2 % / 1 % |
| unsafe50k | sosp_update obj0 | 51.35 | 51.84 | 1.009 | 8 % / 4 % |
|  | sosp_update obj1 | 38.19 | 38.84 | 1.017 | 2 % / 3 % |
|  | sosp_update obj2 | 42.34 | 42.93 | 1.014 | 3 % / 2 % |
|  | sosp_total | 131.56 | 133.58 | 1.015 | 4 % / 2 % |
|  | apply | 179.46 | 190.34 | 1.061 | 6 % / 6 % |
|  | end_to_end | 1020.16 | 1027.78 | 1.007 | 5 % / 1 % |
| local10k | sosp_update obj0 | 59.54 | 57.13 | 0.960 | 9 % / 10 % |
|  | sosp_update obj1 | 51.22 | 46.78 | 0.913 | 12 % / 13 % |
|  | sosp_update obj2 | 58.89 | 54.47 | 0.925 | 10 % / 11 % |
|  | sosp_total | 170.25 | 157.66 | 0.926 | 10 % / 12 % |
|  | apply | 156.91 | 161.72 | 1.031 | 6 % / 8 % |
|  | end_to_end | 1026.08 | 1023.96 | 0.998 | 2 % / 1 % |

#### road_usa, openmp (`M1b-edge-type-openmp-road_usa_g.json`)

| Batch | Region | int32 (ms) | int64 (ms) | int64 / int32 | Spread int32 / int64 |
|---|---|---:|---:|---:|---|
| safe50k | sosp_update obj0 | 271.69 | 276.48 | 1.018 | 5 % / 2 % |
|  | sosp_update obj1 | 252.43 | 256.57 | 1.016 | 2 % / 2 % |
|  | sosp_update obj2 | 253.39 | 257.51 | 1.016 | 2 % / 2 % |
|  | sosp_total | 778.55 | 790.70 | 1.016 | 3 % / 2 % |
|  | apply | 1205.33 | 1295.62 | 1.075 | 5 % / 3 % |
|  | end_to_end | 5527.29 | 5629.17 | 1.018 | 2 % / 2 % |
| unsafe50k | sosp_update obj0 | 273.58 | 277.07 | 1.013 | 2 % / 3 % |
|  | sosp_update obj1 | 253.64 | 257.61 | 1.016 | 1 % / 2 % |
|  | sosp_update obj2 | 254.28 | 258.54 | 1.017 | 2 % / 2 % |
|  | sosp_total | 781.70 | 793.26 | 1.015 | 1 % / 2 % |
|  | apply | 1195.51 | 1297.74 | 1.086 | 2 % / 1 % |
|  | end_to_end | 5520.11 | 5624.52 | 1.019 | 1 % / 1 % |
| local10k | sosp_update obj0 | 58.23 | 56.24 | 0.966 | 8 % / 14 % |
|  | sosp_update obj1 | 55.37 | 53.00 | 0.957 | 8 % / 17 % |
|  | sosp_update obj2 | 55.62 | 53.53 | 0.962 | 8 % / 15 % |
|  | sosp_total | 169.25 | 162.58 | 0.961 | 8 % / 15 % |
|  | apply | 1161.11 | 1263.97 | 1.089 | 2 % / 3 % |
|  | end_to_end | 4859.00 | 4979.85 | 1.025 | 1 % / 1 % |

## 11. The fused kernel's resources (PLAN 8.6 kernel checks, M1b criterion 4)

`cuobjdump --dump-resource-usage` of the unpatched MOSP-CUDA `bin/mosp` and of the final
`parity-cuda` `libdyng.so` (sm_86), and Nsight Compute's launch metrics from section 8.4 (every
batch and graph, both sides):

| Kernel | REG | STACK | SHARED | LOCAL | CONSTANT[0] (parameters) | Blocks per SM (register limit) | Grid x block on the RTX A5000 |
|---|---:|---:|---:|---:|---:|---:|---|
| MOSP-CUDA `sospPersistentKernel` | 59 | 0 | 0 | 0 | 616 | 4 | 256 x 256 |
| dynG `sssp_persistent_kernel<int32, int32, int32>` (the default and parity instantiation) | 59 | 0 | 0 | 0 | 616 | 4 | 256 x 256 |
| dynG `sssp_persistent_kernel<int32, int64, int32>` | 60 | 0 | 0 | 0 | 616 | 4 | 256 x 256 |
| dynG `sssp_persistent_kernel<int64, int64, int32>` | 64 | 0 | 0 | 0 | 624 | 4 | 256 x 256 |

(Blocks per SM and grid of the int64 instantiations follow from their registers: 60 and 64
registers per thread are both allocated as 64, so 4 blocks of 256 threads per SM.) The occupancy
of the default instantiation is the original's (register-limited at 4 blocks of 256
threads, 32 warps of 48 per SM; the block and warp limits are 16 and 6 blocks on both sides), and
the occupancy API gives both the same cooperative grid on the RTX A5000 (64 SMs x 4 = 256 blocks).
The SASS of the int32 instantiation differs from the original's only in the unpack pass (the
`affected` count, section 6.2); the controlled-clock kernel times of section 8.4 and the DRAM
bytes (0.99-1.03x of the original's) are the measured cost of that addition.

## 12. Verdict and reproduction

| M1b criterion | Status |
|---|---|
| 2. CUDA byte-identical to MOSP-CUDA e220ee2 on the 495 cases incl. the M1b cases; CUDA = OpenMP = sequential on the corpus and randomized | **met** (sections 7.2, 7.3; `dyng_sssp_cuda_tests`) |
| 3a. CUDA per-objective SOSP region <= 1.05x / 1.10x | **met** with the clocks locked for the whole A/B (section 14: 36 / 36, 0.977-1.028x); at default clocks road_usa local 10K reads 1.06x (sections 8, 13, 14; the GPU's P-state, ADR 0018) |
| 3b. CUDA end to end <= 1.10x | **met** (0.83-0.92x; apply 0.55-0.74x; section 14: 0.76-0.90x, apply 0.67-0.87x) |
| 3c. OpenMP per-objective SOSP region and end to end | **met** (0.63-0.99x; end to end 0.75-0.91x; apply 0.78-1.02x) |
| 4. Fused kernel registers and occupancy no worse than the original's | **met** (59 = 59 registers, same occupancy and grid; section 11) |
| 7. `edge_t` benchmark and ADR 0009; workspace-sharing ADR | **met** (section 10, ADR 0009: int32 with checked construction; ADR 0015) |

```bash
source scripts/dev_env.sh
parity/build_reference.sh    # unpatched and patched copies (nvcc of references.toml: CUDA 13.1)
cmake --preset parity && cmake --build --preset parity
cmake --preset parity-cuda && cmake --build --preset parity-cuda
# byte parity
parity/compare.py --exe build/parity/tools/compat/dyng-compat-mosp \
    --configs sequential,openmp:1,openmp:4,openmp:16,openmp:28,sequential/int64,openmp:4/int64
CUDA_VISIBLE_DEVICES=1 parity/compare.py --exe build/parity-cuda/tools/compat/dyng-compat-mosp \
    --configs cuda,cuda/int64
parity/export_goldens.py --reference MOSP-CUDA --compare-to      # the corpus from MOSP-CUDA's tools
# performance (each command takes the exclusive perf lock itself)
for g in roadNet-PA roadNet-CA rgg road_usa_g; do
  parity/perf_ab.py prepare --graph $g
  parity/perf_ab.py run --backend cuda --gpu 0 \
      --exe build/parity-cuda/tools/compat/dyng-compat-mosp --graph $g --runs 21
  # (--lock-clocks boost is the default since section 14; --lock-clocks none: default clocks)
  parity/perf_ab.py kernels --gpu 0 --exe build/parity-cuda/tools/compat/dyng-compat-mosp \
      --graph $g --runs 21
  parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp --graph $g --runs 21
  parity/perf_ab.py edge-type --backend openmp --exe build/parity/tools/compat/dyng-compat-mosp \
      --graph $g --runs 11
  parity/perf_ab.py edge-type --backend cuda --gpu 0 \
      --exe build/parity-cuda/tools/compat/dyng-compat-mosp --graph $g --runs 11
done
```

## 13. M1b review: the gates re-measured on the fixed code, with the contamination monitor

After the independent review of M1b (retrospective, "Review and fix step") the gates were read
again on the final code, port `674fc3b` (clean), because the review changed the fused kernel (the
`invalidated` count, ADR 0017 item 1), the workspace leases (a CUDA event per lease, ADR 0015
update), the batch path (the copy-policy staging, ADR 0016 item 11) and the CUDA `apply` region
(now with `sssp.import`), and because PLAN 8.5 / 8.6 ask for a contamination monitor that the
step-4 records did not have.

**Protocol.** As section 8.1 (parity and parity-cuda presets, unpatched originals, exclusive perf
lock, GPU 0, 21 alternating rounds per batch, medians), plus `perf_ab.py`'s monitor: per round and
side the foreign CPU load (machine busy time minus the harness and its programs), the run queue
without the timed program's threads, and for cuda the GPU's P-state, SM and memory clocks and
utilization every 50 ms and foreign compute processes on GPU 0 every second. A round with more
than 2 cores of foreign load or a foreign GPU process is repeated (recorded as rejected). One round
was rejected in the whole campaign (OpenMP roadNet-CA 50K unsafe, 2.16 cores); the largest
foreign load of an accepted round was 1.46 cores, no foreign GPU process was seen. Records:
`M1b-review-perf-{cuda,openmp}-<graph>.json`, `M1b-review-kernels-cuda-road_usa_g.json`,
`M1b-review-sssp-{parity,cuda-parity-cuda}-preset.json`.

**Byte parity.** The golden corpus replays 495 / 495 in all nine configurations on the final code
(sequential, OpenMP 1/4/16/28, sequential and OpenMP 4 with int64 offsets, CUDA, CUDA with int64
offsets); in the A/B every batch's trees are byte-identical and the `invalidated` counters equal in
every timed round, on both backends.

**CUDA against MOSP-CUDA@e220ee2** (as measured; ratio dynG / original, medians of 21 alternating rounds):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds rejected | foreign CPU max (cores) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | safe50k | 0.980 | 0.993 | 0.989 | 4.74 | 0.684 | 0.885 | 0 | 1.24 |
| roadNet-PA | unsafe50k | 0.979 | 0.994 | 0.989 | 4.72 | 0.679 | 0.858 | 0 | 0.41 |
| roadNet-PA | local10k | 0.988 | 0.996 | 0.997 | 9.36 | 0.655 | 0.856 | 0 | 0.31 |
| roadNet-CA | safe50k | 1.001 | 1.006 | 1.005 | 8.64 | 0.855 | 0.874 | 0 | 1.20 |
| roadNet-CA | unsafe50k | 1.003 | 1.007 | 1.009 | 8.66 | 0.823 | 0.844 | 0 | 0.54 |
| roadNet-CA | local10k | 0.979 | 1.001 | 1.001 | 3.50 | 0.802 | 0.826 | 0 | 0.34 |
| rgg_n_2_20_s0 | safe50k | 0.994 | 0.995 | 0.993 | 25.12 | 0.791 | 0.880 | 0 | 0.56 |
| rgg_n_2_20_s0 | unsafe50k | 0.994 | 0.997 | 0.994 | 25.11 | 0.777 | 0.860 | 0 | 0.25 |
| rgg_n_2_20_s0 | local10k | 0.998 | 1.000 | 1.000 | 56.07 | 0.766 | 0.895 | 0 | 0.17 |
| road_usa | safe50k | 1.010 | 1.002 | 1.000 | 101.19 | 0.941 | 0.802 | 0 | 1.46 |
| road_usa | unsafe50k | 1.006 | 1.001 | 0.999 | 101.64 | 0.957 | 0.870 | 0 | 0.18 |
| road_usa | local10k | **1.060** (> 1.05) | **1.060** (> 1.05) | **1.054** (> 1.05) | 21.25 | 0.954 | 0.844 | 0 | 0.17 |

**OpenMP (28 threads pinned) against MOSP-OpenMP@c352151** (as measured; ratio dynG / original, medians of 21 alternating rounds):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds rejected | foreign CPU max (cores) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | safe50k | 0.846 | 0.835 | 0.850 | 16.62 | 1.031 | 0.898 | 0 | 0.67 |
| roadNet-PA | unsafe50k | 0.793 | 0.814 | 0.836 | 17.40 | 0.898 | 0.855 | 0 | 0.72 |
| roadNet-PA | local10k | 0.652 | 0.629 | 0.631 | 28.23 | 0.857 | 0.785 | 0 | 1.01 |
| roadNet-CA | safe50k | 0.812 | 0.925 | 0.929 | 30.50 | 0.898 | 0.783 | 0 | 0.59 |
| roadNet-CA | unsafe50k | 0.770 | 0.903 | 0.922 | 31.83 | 0.865 | 0.817 | 1 | 0.64 |
| roadNet-CA | local10k | 0.734 | 0.697 | 0.782 | 13.42 | 0.850 | 0.769 | 0 | 0.44 |
| rgg_n_2_20_s0 | safe50k | 0.990 | 0.946 | 0.951 | 51.84 | 0.808 | 0.884 | 0 | 0.23 |
| rgg_n_2_20_s0 | unsafe50k | 0.994 | 0.954 | 0.951 | 51.82 | 0.813 | 0.819 | 0 | 0.30 |
| rgg_n_2_20_s0 | local10k | 0.700 | 0.675 | 0.700 | 79.24 | 0.791 | 0.786 | 0 | 0.25 |
| road_usa | safe50k | 0.888 | 0.940 | 0.950 | 307.94 | 0.791 | 0.854 | 0 | 0.33 |
| road_usa | unsafe50k | 0.874 | 0.922 | 0.921 | 312.80 | 0.790 | 0.800 | 0 | 0.22 |
| road_usa | local10k | 0.786 | 0.893 | 0.900 | 74.63 | 0.786 | 0.847 | 0 | 0.20 |


**The CUDA miss on road_usa's local 10K batch persists** (1.060 / 1.060 / 1.054x; step 4:
1.065 / 1.063 / 1.037x), now on all three objectives by a hair on objective 2. What the new data
show:

- *Not contamination.* Foreign CPU load at most 0.17 cores in every round of that batch, no
  foreign GPU process, no rejected round.
- *Bimodal per-round times on both sides.* Objective 0 of the original took 21.2-21.4 ms in 16 of
  21 rounds and 21.8-22.7 ms in 5; dynG took 22.5-22.6 ms in 18 rounds and 21.5-22.0 ms in 3. The
  two modes are the same on both sides (about 21.3 and 22.6 ms); the medians differ because the
  sides fall into them in different proportions, as in step 4.
- *The kernel at locked clocks is the original's.* Nsight Compute at base clocks, 21 rounds:
  road_usa local 10K 0.991 / 0.989 / 0.990x, the 50K batches 1.009-1.011x (the review's
  per-thread `invalidated` count included), DRAM bytes 0.986-1.007x, 59 registers and a 256 x 256
  grid on both sides.
- *What the monitor cannot show.* Sampled every 50 ms over each whole process, both sides reach
  the same maximum SM clock (1905 MHz) in every round and both spend some samples in P0; the three
  kernels of a round take about 65 ms together, one or two samples, and cannot be told apart from
  the upload and apply work around them. So the gated samples do **not** establish ADR 0018's
  condition (a) ("the two sides' kernels ran in different performance states"): at this
  resolution the recorded states are the same. Under the proposed rule 3 the reading would stay a
  FAIL; under the strict reading it is a FAIL. ADR 0018 records this; the author's decision (A,
  B or C) is still open, and option A (locked clocks for the whole A/B) remains the one that would
  settle it.

**Everything else passes as measured:** CUDA 33 / 36 per-objective readings within the gate
(0.979-1.010x), apply 0.66-0.96x (higher than step 4's 0.55-0.74x because `sssp.import` is now
counted on dynG's side), end to end 0.80-0.90x; OpenMP 36 / 36 per-objective readings (0.63-0.99x),
apply 0.79-1.03x, end to end 0.77-0.90x. The fused kernel keeps the original's 59 registers (60 and
64 for the int64 instantiations, as before).

## 14. M1b acceptance fix: the CUDA gate with the GPU clocks locked for the whole A/B

The acceptance verifier re-measured section 13's CUDA gate at default clocks and read road_usa's
local 10K batch at 1.061 / 1.059 / 1.014x (FAIL of criterion 3(a)), naming option A of ADR 0018
(locked clocks for the whole A/B, thought to need root) as the way forward. It does not need
root: Nsight Compute's clock control holds a lock for every process on the GPU while the process
it profiles lives (`RmProfilingAdminOnly: 0` on this machine). ADR 0018's update has the checks.

**Protocol.** As section 13 (parity and parity-cuda presets built from `3ecb8fd`, the unpatched
MOSP-CUDA@e220ee2 rebuilt and verified by `build_reference.sh`, exclusive perf lock, GPU 0, 21
alternating A/B rounds per batch, medians, contamination monitor, `CUDA_MODULE_LOADING=EAGER`),
plus `--lock-clocks boost` (now the default of `perf_ab.py run --backend cuda`): before the
first round the harness starts `ncu --clock-control boost parity/clock_lock/clock_holder.cu`
on GPU 0 (one kernel, then idle until the harness closes its input); the GPU then reports SM
1695 MHz and memory 7601 MHz (P2) under any load. Neither timed program is profiled or changed.
The monitor accepts the helper's idle context and **requires every busy GPU sample (50 ms) of both
sides in every accepted round to be at the locked clocks** (column "clocks locked"); at the end
the helper exits and `ncu --clock-control reset` restores the default clocks (recorded in each
JSON, `protocol.gpu_clocks`). Rounds with more than 2 cores of foreign CPU load were repeated;
the machine carried about 1.1 cores of foreign load (desktop daemons reacting to network
mounts) throughout, below the limit.

**CUDA against MOSP-CUDA@e220ee2, clocks locked (boost)** (ratio dynG / original, medians of 21
alternating rounds; records `M1b-accept-perf-cuda-<graph>.json`):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds rejected | foreign CPU max (cores) | clocks locked in every round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| roadNet-PA | safe50k | 1.010 | 1.027 | 1.028 | 4.75 | 0.693 | 0.902 | 0 | 1.40 | yes |
| roadNet-PA | unsafe50k | 1.011 | 1.023 | 1.024 | 4.74 | 0.690 | 0.881 | 0 | 1.41 | yes |
| roadNet-PA | local10k | 0.994 | 1.002 | 1.002 | 9.36 | 0.675 | 0.879 | 0 | 1.37 | yes |
| roadNet-CA | safe50k | 0.985 | 0.994 | 0.997 | 8.76 | 0.867 | 0.835 | 0 | 1.66 | yes |
| roadNet-CA | unsafe50k | 0.989 | 0.999 | 0.996 | 8.74 | 0.816 | 0.883 | 0 | 1.31 | yes |
| roadNet-CA | local10k | 0.977 | 0.999 | 1.000 | 3.50 | 0.796 | 0.837 | 0 | 1.36 | yes |
| rgg_n_2_20_s0 | safe50k | 0.988 | 0.991 | 0.991 | 25.90 | 0.805 | 0.889 | 2 | 1.28 | yes |
| rgg_n_2_20_s0 | unsafe50k | 0.990 | 0.991 | 0.991 | 25.88 | 0.776 | 0.832 | 0 | 1.29 | yes |
| rgg_n_2_20_s0 | local10k | 1.001 | 1.003 | 1.002 | 61.49 | 0.734 | 0.901 | 0 | 1.69 | yes |
| road_usa | safe50k | 1.000 | 0.999 | 1.000 | 102.80 | 0.805 | 0.813 | 1 | 1.40 | yes |
| road_usa | unsafe50k | 0.999 | 1.000 | 1.001 | 102.80 | 0.810 | 0.756 | 3 | 1.97 | yes |
| road_usa | local10k | 0.993 | 0.994 | 0.995 | 22.72 | 0.798 | 0.816 | 1 | 1.32 | yes |

**The same A/B at base clocks** (SM 1170 MHz, memory 7601 MHz; road_usa,
`M1b-accept-perf-cuda-base-road_usa_g.json`):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds rejected | foreign CPU max (cores) | clocks locked in every round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| road_usa | safe50k | 1.002 | 1.002 | 1.001 | 112.56 | 0.810 | 0.787 | 0 | 1.41 | yes |
| road_usa | unsafe50k | 1.002 | 1.002 | 1.002 | 112.53 | 0.816 | 0.745 | 0 | 1.26 | yes |
| road_usa | local10k | 0.993 | 0.994 | 0.995 | 29.38 | 0.816 | 0.776 | 0 | 1.29 | yes |

**At default clocks** (`--lock-clocks none`, the as-measured reading of sections 8 and 13;
road_usa's local batch, `M1b-accept-perf-cuda-unlocked-road_usa_g.json`; ungated, see below):

| Graph | Batch | obj0 | obj1 | obj2 | original obj0 (ms) | apply | end to end | rounds rejected | foreign CPU max (cores) | clocks locked in every round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| road_usa | local10k | **1.063** (> 1.05) | **1.060** (> 1.05) | **1.062** (> 1.05) | 21.22 | 0.882 | 0.781 | 0 | 1.70 | n/a |

**What the three readings show.**

- With the clocks fixed, road_usa's local batch reads 0.993-0.995x at both lock levels, like the
  other 33 per-objective readings (0.977-1.028x; the short roadNet-PA regions, 4.7-9.4 ms, are
  gated at 1.10x). The per-round times become unimodal: objective 0 of the original took
  22.65-22.81 ms in all 21 rounds, dynG's 22.53-22.63 ms.
- At default clocks the two modes of section 13 are back (original objective 0: 18 of 21 rounds at
  21.17-21.26 ms; dynG: 16 of 21 at 22.49-22.65 ms), and so is the 1.06x. The fast mode of the
  original (21.2 ms) is faster than its own time at the 1695 MHz lock (22.7 ms): its kernels ran
  above 1695 MHz, in P0, which the monitor saw more often for the original (118 of 480 busy
  samples in P0) than for dynG (68 of 525). The difference is the GPU's DVFS response to the work
  each program runs before its kernels, not the port (ADR 0018).
- apply (0.67-0.88x) and end to end (0.75-0.90x) pass under every clock setting.

**Verdict (criterion 3).** With the gate protocol of ADR 0018 rule 4 (clocks locked for the whole
A/B), every CUDA reading is within its gate: per objective 36 / 36 (<= 1.05x, <= 1.10x under 10
ms), apply and end to end <= 1.10x. The OpenMP gate is unchanged from section 13 (36 / 36 per
objective, apply and end to end within 1.10x; the code has not changed since). The default-clock
reading is kept as a separate, ungated record, as ADR 0018 decision 1 requires; whether it should
be gated as well is the one question left to the author in ADR 0018.

## 15. Author decision on the CUDA gate (2026-09-28)

The author accepted ADR 0018 option B: the CUDA per-objective gate is read with the clocks locked for
the whole A/B (section 14, `--lock-clocks boost`), and default-clock readings are recorded, not gated.
Under that rule every M1b gate passes: 36 / 36 CUDA per-objective regions (section 14), CUDA apply
and end to end, and all OpenMP gates.
