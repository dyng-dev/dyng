# M1b parity record, step cpu-gates: `sssp` (OpenMP) against MOSP-OpenMP@c352151

Date: 2026-09-27. Format: PLAN Section 8.3 ("Parity certificate") and ADR 0013. This record
closes the M1a carry-over (ADR 0013, "Open M1b blocker"): the OpenMP gates of PLAN 8.6 / 6.4.2
on the four graphs of the `sssp` suite. The CUDA sections are added by the later M1b steps. The
machine-readable records are next to this file:

| File | Content |
|---|---|
| `M1b-sssp-parity-preset.json` | golden replay, `parity` preset (Release, `-O3`) |
| `M1b-sssp-dev-preset.json` | golden replay, `dev` preset (Debug) |
| `M1b-perf-openmp-roadNet-PA.json` | OpenMP A/B, roadNet-PA (schema 3) |
| `M1b-perf-openmp-roadNet-CA.json` | OpenMP A/B, roadNet-CA |
| `M1b-perf-openmp-rgg.json` | OpenMP A/B, rgg_n_2_20_s0 |
| `M1b-perf-openmp-road_usa_g.json` | OpenMP A/B, road_usa |

## 1. What was compared

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
`b59de86`: **495 / 495 cases byte-identical** (`compute` = `mospPrep init`, `update` = `mosp`
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
(the blocks per SM come from the occupancy API on both sides; 256 threads per block). The SASS of
the int32 instantiation has about 190 more instructions than the original's (2,560 vs 2,368),
from the `affected` count in the unpack pass (ADR 0017 item 1); the search loops are unchanged.
The int64 instantiations stay within 64 registers (4 blocks of 256 threads per SM on sm_86).

### 6.3 A first A/B (not the gate record)

roadNet-PA, `perf_ab.py run --backend cuda --runs 5` (GPU 0, exclusive perf lock), ratio dynG /
MOSP-CUDA of the medians: SOSP region per objective 0.98-1.00x (4.6-9.4 ms; host times of the same
scope on both sides, dynG's CUDA-event time within 0.01 ms of its host time), apply 0.55-0.60x,
end to end 0.82-0.84x, for the 50K safe, 50K unsafe and 10K local batches; outputs byte-identical
and `invalidated` equal in every run. The gate record (four graphs, >= 20 runs because the
per-objective regions are under 10 ms on roadNet-PA) follows in the next step.
