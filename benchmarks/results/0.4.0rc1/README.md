# dynG 0.4.0rc1: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.4.0rc1 (`sssp`, `cycle_count`, `mosp`, the dynamic hypergraph and `triad_count`, and,
for the first time, `label_propagation` against LabelPropagation-CUDA and `hyper_sssp` against
MOSP_ESCHER with the `slack_csr` store, stable from 0.4.0), with the release checks of PLAN
Section 10.3 steps 1-4. `parity.json` is the machine-readable certificate; the tables at the end
of this page are generated from it by `parity/certify.py write`. `benchmarks/README.md` (the
developer guide's "Benchmark suites" page) describes the tools and the file formats;
`docs/developer/retrospectives/R040.md` (step 3) tells how this certificate was made, with ADR
0024's reading of the stable algorithms against 0.3.0, which is not part of this certificate (a
dynG-against-dynG reading, `parity/results/R040/adr0024/`; its open STOP for the author is in
GOVERNANCE.md, Open decisions).

## What was run, and how

- **Code.** The release branch `release-0.4.0` (from `main` at `d3375da`, the merge of the ADR
  0120 fix, which contains M10b's merge `c67dd40`). Every measured build comes from a clean `git
  clone` of the branch (the commit each record names), with the presets `parity` and
  `parity-cuda`. The gates, the golden replays and the golden mutations ran at `f60a2f3`
  (2026-11-20 06:59 UTC to 2026-11-21; its library is the release's final one); the sanitizer
  presets, the mutation CTests, the portability pre-checks, the distributions,
  `ci/check.sh --parity`, `ci/api_check.sh d3375da` (against `main` as pushed) and one run of
  `ci/gpu_local.sh --preset parity-cuda` at `b285c97`, the last commit outside this directory.
  `parity.json` checks every measured commit against the release commit in the paths of its
  scope (`library` for the gates, the replays and the golden mutations, plus `cpp/tests` for the
  golden mutations of the GoogleTest suites, ADR 0061; `tests` for the C++ test suites;
  `packaging` for the distributions; `repo`, the whole tree but this directory, for
  `ci/check.sh`, `ci/api_check.sh` and `ci/gpu_local.sh`). No equivalence record is needed: no
  measured commit differs from the release in its scope.
- **Golden parity** (`parity-*.json`; CUDA replays on GPU 1): the sssp corpus of
  MOSP-OpenMP@c352151 (495 cases, the whole MOSP update with the combined graph through
  `dyng-compat-mosp`) on sequential, OpenMP 1 / 4 / 16 / 28 / 56 threads and int64 offsets, and on
  CUDA with the automatic, the fused and the operators engine and int64 offsets (6,435 / 6,435);
  the cycle_count CPU corpus of CycleEnumeration-GPU@0a976ad with `--full` and its CUDA corpus (72
  + 72); the paper-scale mosp set `mosp_scale` on sequential, OpenMP 28 and CUDA with both engines
  (80 / 80).
- **The CBST trace goldens** (`parity-cbst_trace_<set>-device-<preset>.json`): all eight sets of
  ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159 through the dynG store on the device under the
  original's preset (the exact field set) and under `dyng_default` (the policy field set), each
  checked against `parity/cbst_documented.toml`: 16 / 16 OK.
- **The hypergraph container's replays** (`parity-<set>-hypergraph_ids-*.json`): the five CBST
  trace sets with hypergraph batches with the originals' ids and rows (`--mode both`) on
  sequential, OpenMP (8 threads) and CUDA, ESCHER-GPU's three sets with the line graph on CUDA:
  18 / 18 runs, 0 mismatches; and (from 0.4) MOSP_ESCHER's sessions of `test`,
  `dblp50k_hyperedge` and `dblp50k_vertex` replayed **whole on `slack_csr` storage**, incidence
  edits included, on the three backends (`-slack_csr.json`): 9 / 9 runs, 0 mismatches.
- **The triad goldens** (`parity-triad_count_escher_gpu-*.json`): by key and by content with the
  algorithm counter on every backend of the coverage table: CUDA 300 / 300, sequential 120 / 120,
  OpenMP at 8 and at 1 thread 142 / 142 each.
- **The label_propagation golden sets** (from 0.4; `parity-label_propagation_<kind>-*.json`,
  each written by `compare.py label_propagation certificate` from its replay record): the
  generator on sequential; the stage trace of P+ (5m included) on sequential and CUDA; the damped
  byte parity and the hub golden on the fused engine of the parity-cuda build; IrLP on CUDA; the
  in_place set under **ADR 0111's rate rule** (30 port and 30 original runs per case on GPU 1;
  port / original runs outside the frozen envelope 2 / 0, 3 / 0, 0 / 0 and 1 / 0 at 5k5, 50k, 500k
  and 5m, one-sided Fisher p 0.246, 0.119, 1 and 0.5: every case within; every port run's labels
  within every reference run and its components exact); the harmonic oracle's port replay on
  sequential (C) and on CUDA over the eight C and L cases, L the release replay at delta 1e-4
  (AD7). Every kind passed.
- **hyper_sssp's replays** (from 0.4; `parity-hyper_sssp_mosp_escher-*.json`): every mode of the
  set's `[modes]` table, by_id, by_id_public and by_key, on CUDA with both engines (`automatic`,
  the as-is kernels, and `operators`, the shared sssp engine) and on sequential, over every case
  the table names; and the `slack_csr` store check on CUDA over every make-test session and every
  paper configuration, with the line-graph oracle after every batch: 0 mismatches.
- **Performance gates** (the seven suites of `benchmarks/paper/` that `certify.py` requires from
  0.4: `ipdps25_dynamosp_sosp`, `ieee_tc_dyntrucy`, `ipdps25_dynamosp_mosp`, `ipdps26_escher`,
  `tkde26_escher_plus`, `ics26_dynlp` and `ia3_hsosp`; every gated row recomputed from the
  compacted records): against the unpatched originals, every timing reading in its own exclusive
  window of the perf lock, alternating rounds, medians; OpenMP sssp and mosp with 28 threads,
  OpenMP cycle_count with 56; CUDA on GPU 0 with the clocks locked for the whole A/B (boost for
  sssp, mosp, cycle_count and label_propagation; the COLLAB update, triad_count, hyper_sssp and
  AD4's 50m reading at base; ADR 0021), the default-clock readings recorded and not gated (ADR
  0018); device memory on GPU 1 under Nsight Systems in GPU1-QUIET windows (D1). **611 / 611**
  gated regions within their gates. **AD4's release-only reading** `release-damped-50m` (5 rounds
  at base, `--noGraphFiles` on both sides, `CUDA_MODULE_LOADING=EAGER`, one exclusive window of 17
  minutes, 0 of 5 rounds rejected; its `release_gate` state `gated`): `update[sum]` 1.012
  (63,309.6 against 64,064.8 ms), `end_to_end` 1.059. `ia3_hsosp`'s `dblp_hyperedge_25000` first
  batch (the author's decision of 2026-11-13: read once here, and this reading decides):
  `maintain[b1]` 0.834. The monitors rejected and repeated 70 rounds (foreign CPU load above 2
  cores, or a GPU clock below its lock); no accepted reading kept a contaminated round. Three
  readings were taken again as the protocol says, their first records kept under `superseded/` of
  the full records: `ics26_dynlp`'s `memory-damped` 500k (a foreign process on GPU 1 during the
  original's reading, not counted) and `ipdps26_escher`'s `cuda-coauth` and `cuda` (the suite's
  audit wants the harness to hold the lock itself). The records here are compacted; each names its
  full copy under `$DYNG_SCRATCH/runs/bench/0.4.0rc1/`. Every record names the NVIDIA driver and
  carries the digests of its inputs taken by the measuring harness, all equal to the suites'.
- **rgg's unsafe50k batch** is byte-identical to its safe50k batch (`benchmarks/README.md`): its
  rows of the sssp and mosp suites are a second reading of the same input.
- **Committed fixtures** (`cpp/tests/data`, `parity/fixtures/fixtures.toml`): each set with the
  SHA-256 of its files and the results of the CTests that compare it, taken from the per-test
  lines of the checks' ctest logs.
- **Checks** (`checks.json`; `certify.py` requires each for a 0.4 certificate): the golden
  mutations (`mutation-goldens.json`, `parity/mutate.py run` without `--suites` or
  `--mutations`: the control and every mutation of `parity/mutate.py list`, the suites sssp,
  mosp, cycle_count, label_propagation, triad_reference, cbst, hypergraph, triad_count, hyper_sssp
  with the `slack_csr` container's and the kit: the control and **50 / 50** detected); the
  `asan`, `tsan` and `tsan-openmp` presets (`ci/sanitizers.sh`: 1,486 / 1,486, 1,274 / 1,274 and
  1,486 / 1,486; `tsan-openmp` with the conda Clang 18.1.8 through a local wrapper and the
  environment's Archer, as for 0.2 and 0.3); the mutation CTests (`ctest --preset dev-cuda -L
  mutation`, GPU 1: 72 / 72); the distributions as `release.yml` builds them for
  `v0.4.0rc1` (the core sdist and abi3 wheel and the plugins `dyng-cu12` / `dyng-cu13` with CI's
  CUDA toolkits 12.9 and 13.5, ADR 0113; the `collect` checks; fresh 3.12 / 3.13 venvs including
  `dyng[cu13]==0.4.0rc1` and `dyng[cu12]==0.4.0rc1` from a local index laid out like TestPyPI and
  the documented release-candidate forms; the wheel's test subset (471-472 passed), the README
  quickstart, every Python snippet of the docs, the label_propagation and hyper_sssp tutorials
  and the label_propagation example's labels against their committed digests, `pytest -m gpu`
  on GPU 1, 80 passed; SHA-256 of the local builds, whose tag's files are CI's: sdist
  `de1c60f3...`, CPU wheel `7580a694...`, cu12 `a75306e9...`, cu13 `904f0227...`);
  `ci/check.sh --parity` (cpu-only 1,656 / 1,656, dev 1,701 / 1,701, the harness 552 passed,
  `ci/python.sh` 816 passed, the parity preset's golden replay 92 / 92); `ci/api_check.sh d3375da` (the frozen C++ API and the Python API
  against `main` as pushed); and **one** run of `ci/gpu_local.sh --preset parity-cuda` with
  nothing skipped (every step of `GPU_LOCAL_STEPS` passed, ADR 0062: the CUDA tests and goldens,
  compute-sanitizer, the CBST store, the hypergraph and triad_count steps, count_local_patterns,
  label_propagation with the in_place cloud under ADR 0111's rate rule, ADR 0110, slack_csr,
  hyper_sssp, the plugin wheel, clang-tidy; its racecheck runs under ADR 0112's bounded retry).
  Every check names its logs with their SHA-256; an excerpt of each is in `checks/`.
- **Files.** `ia3_hsosp.json`, the suite summary, is written without indentation (the same
  document; the repository's limit of 256 KB on an added file, `.pre-commit-config.yaml`).

<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->

**Certificate:** `parity.json` (dynG 0.4.0rc1, commit `b285c974b9ca`): **all parts passed**.

Machine: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz (56 threads), 2x NVIDIA RTX A5000; driver 590.48.01 (CUDA driver API 13.1; named by 89 records); Build cuda_13.1.r13.1/compiler.37061995_0; c++ (Debian 12.2.0-14+deb12u1) 12.2.0.

### Golden parity

| Golden set | Original | Cases | Replays (configurations: compared / failed) | Result |
|---|---|---:|---|---|
| `sssp` | MOSP-OpenMP@c352151 | 495 | sequential, openmp:1, openmp:4, openmp:16, openmp:28, openmp:56, sequential/int64, openmp:4/int64: 3960 / 0; cuda, cuda-fused, cuda-operators, cuda/int64, cuda-operators/int64: 2475 / 0 | passed |
| `cycle_count` | CycleEnumeration-GPU@0a976ad | 24 | sequential, openmp:4, openmp:56: 72 / 0 | passed |
| `cycle_count_cuda` | CycleEnumeration-GPU@0a976ad | 24 | cuda, cuda:resident, cuda:int64: 72 / 0 | passed |
| `mosp_scale` | MOSP-OpenMP@c352151 | 20 | sequential, openmp:28: 40 / 0; cuda, cuda-operators: 40 / 0 | passed |
| `cbst_trace_escher_gpu_test` | ESCHER-GPU@abcf9b2 | 75 | device:dyng_default/policy: 75 / 0; device:escher_gpu/exact: 75 / 0; cuda:hypergraph_ids/both: 75 / 0; cuda:hypergraph_ids/line_graph: 75 / 0; openmp/t8:hypergraph_ids/both: 75 / 0; sequential:hypergraph_ids/both: 75 / 0 | passed |
| `cbst_trace_escher_gpu_targeted` | ESCHER-GPU@abcf9b2 | 2 | device:dyng_default/policy: 2 / 0; device:escher_gpu/exact: 2 / 0 | passed |
| `cbst_trace_mosp_escher_targeted` | MOSP_ESCHER@4b86159 | 2 | device:dyng_default/policy: 2 / 0; device:mosp_escher/exact: 2 / 0 | passed |
| `cbst_trace_escher_gpu_coauth50k` | ESCHER-GPU@abcf9b2 | 1 | device:dyng_default/policy: 1 / 0; device:escher_gpu/exact: 1 / 0; cuda:hypergraph_ids/both: 1 / 0; cuda:hypergraph_ids/line_graph: 1 / 0; openmp/t8:hypergraph_ids/both: 1 / 0; sequential:hypergraph_ids/both: 1 / 0 | passed |
| `cbst_trace_mosp_escher_dblp50k_hyperedge` | MOSP_ESCHER@4b86159 | 1 | device:dyng_default/policy: 1 / 0; device:mosp_escher/exact: 1 / 0; cuda:hypergraph_ids/both: 1 / 0; cuda:hypergraph_ids/slack_csr: 1 / 0; openmp/t8:hypergraph_ids/both: 1 / 0; openmp/t8:hypergraph_ids/slack_csr: 1 / 0; sequential:hypergraph_ids/both: 1 / 0; sequential:hypergraph_ids/slack_csr: 1 / 0 | passed |
| `cbst_trace_mosp_escher_dblp50k_vertex` | MOSP_ESCHER@4b86159 | 1 | device:dyng_default/policy: 1 / 0; device:mosp_escher/exact: 1 / 0; cuda:hypergraph_ids/slack_csr: 1 / 0; openmp/t8:hypergraph_ids/slack_csr: 1 / 0; sequential:hypergraph_ids/slack_csr: 1 / 0 | passed |
| `cbst_trace_escher_gpu_syn1m` | ESCHER-GPU@abcf9b2 | 3 | device:dyng_default/policy: 3 / 0; device:escher_gpu/exact: 3 / 0; cuda:hypergraph_ids/both: 3 / 0; cuda:hypergraph_ids/line_graph: 3 / 0; openmp/t8:hypergraph_ids/both: 3 / 0; sequential:hypergraph_ids/both: 3 / 0 | passed |
| `cbst_trace_mosp_escher_test` | MOSP_ESCHER@4b86159 | 24 | device:dyng_default/policy: 24 / 0; device:mosp_escher/exact: 24 / 0; cuda:hypergraph_ids/both: 24 / 0; cuda:hypergraph_ids/slack_csr: 24 / 0; openmp/t8:hypergraph_ids/both: 24 / 0; openmp/t8:hypergraph_ids/slack_csr: 24 / 0; sequential:hypergraph_ids/both: 24 / 0; sequential:hypergraph_ids/slack_csr: 24 / 0 | passed |
| `triad_count_escher_gpu` | ESCHER-GPU@abcf9b2 | 300 | cuda:triad/by_key, cuda:triad/by_content: 600 / 0; openmp/t1:triad/by_key, openmp/t1:triad/by_content: 284 / 0; openmp/t8:triad/by_key, openmp/t8:triad/by_content: 284 / 0; sequential:triad/by_key, sequential:triad/by_content: 240 / 0 | passed |
| `label_propagation_generator` | LabelPropagation-CUDA@a276a3a | 14 | sequential:lp/generator: 14 / 0 | passed |
| `label_propagation_damped` | LabelPropagation-CUDA@a276a3a | 10 | cuda-fused:lp/damped: 10 / 0 | passed |
| `label_propagation_trace` | LabelPropagation-CUDA@a276a3a | 10 | cuda:lp/trace: 10 / 0; sequential:lp/trace: 10 / 0 | passed |
| `label_propagation_hub` | LabelPropagation-CUDA@a276a3a | 1 | cuda-fused:lp/hub: 1 / 0 | passed |
| `label_propagation_irlp` | LabelPropagation-CUDA@a276a3a | 2 | cuda:lp/irlp: 2 / 0 | passed |
| `label_propagation_inplace` | LabelPropagation-CUDA@a276a3a | 6 | cuda:lp/inplace: 4 / 0 | passed |
| `label_propagation_oracle` | LabelPropagation-CUDA@a276a3a | 8 | cuda:lp/oracle: 8 / 0; sequential:lp/oracle: 4 / 0 | passed |
| `hyper_sssp_mosp_escher` | MOSP_ESCHER@4b86159 | 25 | cuda:hyper_sssp/by_id: 25 / 0; cuda:hyper_sssp/by_id_public: 13 / 0; cuda:hyper_sssp/by_key: 10 / 0; cuda:hyper_sssp/store: 25 / 0; cuda-operators:hyper_sssp/by_id: 25 / 0; cuda-operators:hyper_sssp/by_id_public: 13 / 0; cuda-operators:hyper_sssp/by_key: 10 / 0; sequential:hyper_sssp/by_id: 12 / 0; sequential:hyper_sssp/by_id_public: 9 / 0; sequential:hyper_sssp/by_key: 10 / 0 | passed |

### Committed fixtures (`cpp/tests/data`, `parity/fixtures/fixtures.toml`)

| Fixture set | Original | Files | SHA-256 | Tests (checks) | Result |
|---|---|---:|---|---|---|
| `mosp_changes` | MOSP-OpenMP@c352151 | 49 | `989f8b4b7d69f88e...` | 1 patterns, 3 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_graph_io` | MOSP-OpenMP@c352151 | 372 | `664b54743f786a9b...` | 2 patterns, 7 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_sssp` | MOSP-OpenMP@c352151 | 482 | `3fd1f76321898904...` | 3 patterns, 48 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `cycle_enum` | CycleEnumeration-GPU@0a976ad | 529 | `5ba52e2cfda6afcd...` | 13 patterns, 20 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_combined` | MOSP-OpenMP@c352151 | 36 | `3e3b73b6822b7a3e...` | 2 patterns, 27 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `cbst_trace` | ESCHER-GPU@abcf9b2 | 9 | `e474e47264a189ee...` | 1 patterns, 6 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `triad_counts` | ESCHER-GPU@abcf9b2 | 1 | `83fd19ff8684d9b3...` | 1 patterns, 2 tests (check-parity, gpu_local) | passed |

### Performance gates (PLAN 8.6, against the unpatched originals)

| Suite | Reading | Backend (clocks) | Gated regions | Ratio range (port / original) | Result |
|---|---|---|---:|---|---|
| `ia3_hsosp` | cuda-dblp | cuda (base) | 128 / 128 | 0.148-1.026 | passed |
| `ia3_hsosp` | cuda-geology | cuda (base) | 128 / 128 | 0.200-1.034 | passed |
| `ia3_hsosp` | cuda-memory | cuda | 15 / 15 | 0.477-1.000 | passed |
| `ia3_hsosp` | cuda-default-clocks | cuda (none) | recorded, not gated (64) | 0.216-1.023 | - |
| `ics26_dynlp` | damped-50k | cuda (boost) | 2 / 2 | 1.008-1.063 | passed |
| `ics26_dynlp` | damped-500k | cuda (boost) | 2 / 2 | 1.007-1.020 | passed |
| `ics26_dynlp` | damped-5m | cuda (boost) | 2 / 2 | 0.987-1.008 | passed |
| `ics26_dynlp` | damped-5m1 | cuda (boost) | 2 / 2 | 1.010-1.019 | passed |
| `ics26_dynlp` | memory-damped | cuda | 4 / 4 | 1.002-1.012 | passed |
| `ics26_dynlp` | release-damped-50m | cuda (base) | 2 / 2 | 1.012-1.059 | passed |
| `ieee_tc_dyntrucy` | openmp | openmp | 15 / 15 | 0.440-0.938 | passed |
| `ieee_tc_dyntrucy` | openmp-collab | openmp | 1 / 1 | 0.329-0.329 | passed |
| `ieee_tc_dyntrucy` | cuda | cuda (boost) | 37 / 37 | 0.220-1.003 | passed |
| `ieee_tc_dyntrucy` | cuda-collab-update | cuda (base) | 5 / 5 | 0.945-0.999 | passed |
| `ieee_tc_dyntrucy` | cuda-memory | cuda | 20 / 20 | 1.000-1.001 | passed |
| `ieee_tc_dyntrucy` | cuda-default-clocks | cuda (none) | recorded, not gated (42) | 0.220-1.006 | - |
| `ipdps25_dynamosp_mosp` | openmp | openmp | 24 / 24 | 0.621-0.926 | passed |
| `ipdps25_dynamosp_mosp` | cuda | cuda (boost) | 24 / 24 | 0.749-1.027 | passed |
| `ipdps25_dynamosp_mosp` | cuda-memory | cuda | 12 / 12 | 0.931-1.019 | passed |
| `ipdps25_dynamosp_mosp` | cuda-default-clocks | cuda (none) | recorded, not gated (24) | 0.713-1.006 | - |
| `ipdps25_dynamosp_sosp` | openmp | openmp | 60 / 60 | 0.688-0.989 | passed |
| `ipdps25_dynamosp_sosp` | cuda | cuda (boost) | 60 / 60 | 0.658-1.029 | passed |
| `ipdps25_dynamosp_sosp` | cuda-memory | cuda | 12 / 12 | 0.812-0.858 | passed |
| `ipdps25_dynamosp_sosp` | cuda-default-clocks | cuda (none) | recorded, not gated (60) | 0.647-1.060 | - |
| `ipdps26_escher` | cuda-coauth | cuda (base) | 15 / 15 | 0.942-0.999 | passed |
| `ipdps26_escher` | cuda | cuda (base) | 30 / 30 | 0.855-1.024 | passed |
| `ipdps26_escher` | cuda-default-clocks | cuda (none) | recorded, not gated (15) | 0.840-1.021 | - |
| `ipdps26_escher` | cuda-batch-sizes | cuda (base) | recorded, not gated (10) | 0.954-1.017 | - |
| `tkde26_escher_plus` | cuda-memory | cuda | 11 / 11 | 1.000-1.002 | passed |
| `tkde26_escher_plus` | cuda-memory-batch-sizes | cuda | recorded, not gated (10) | 1.000-1.002 | - |
| `tkde26_escher_plus` | cuda-types | cuda (base) | recorded, not gated (15) | 0.142-0.993 | - |

Inputs of the readings:

- `ia3_hsosp`: hashed by the measuring process (hsosp_goldens.check_dataset before each configuration; the probe's datasets likewise).
- `ics26_dynlp`: generated in-process by both sides from the configuration (the golden set's digest; dynG's generator byte-identical to the original's), the per-round guard on what both computed.
- `ieee_tc_dyntrucy`: hashed by the measuring process.
- `ipdps25_dynamosp_mosp`: hashed by the harness when the measurement started.
- `ipdps25_dynamosp_sosp`: hashed by the harness when the measurement started.
- `ipdps26_escher`: hashed by the measuring process (coauth-DBLP's file; a synthetic input's rows as the golden set stores them, which both sides generate from the suite's parameters), with the guard requiring the golden's counts from every measured process (each round of a timed reading; each plain and profiled run of a memory reading).
- `tkde26_escher_plus`: hashed by the measuring process (coauth-DBLP's file; a synthetic input's rows as the golden set stores them, which both sides generate from the suite's parameters), with the guard requiring the golden's counts from every measured process (each round of a timed reading; each plain and profiled run of a memory reading).

### Checks

| Check | Command | Scope | Result |
|---|---|---|---|
| asan | `ci/sanitizers.sh asan (ctest -L cpu -LE slow of the asan preset), from a clean clone` | tests | passed (1486 / 1486 tests, 112 skipped) |
| tsan | `ci/sanitizers.sh tsan (ctest -L cpu -LE slow of the tsan preset), from a clean clone` | tests | passed (1274 / 1274 tests, 131 skipped) |
| tsan-openmp | `ci/sanitizers.sh tsan-openmp (Clang 18.1.8 of the conda env dyng-clang through a local wrapper: --gcc-install-dir GCC 12, --sysroot=/, -Wno-deprecated-declarations; Archer DYNG_ARCHER_LIBRARY = the env's libarcher.so.bak; ctest -L cpu -LE slow), from a clean clone` | tests | passed (1486 / 1486 tests, 112 skipped) |
| mutation-goldens | `parity/mutate.py run (every mutation of parity/mutate.py list: sssp, mosp, cycle_count, label_propagation, triad_reference, cbst, hypergraph, triad_count, hyper_sssp with the slack_csr container's, kit; parity-cuda copies, CUDA on GPU 1), from a clean clone` | library | passed |
| check-parity | `ci/check.sh --parity, from a clean clone (origin/main = d3375da, main as pushed)` | repo | passed |
| api-check | `ci/api_check.sh d3375da (the frozen C++ API with ci/api_snapshot.py --against and the Python API with griffe, against main as pushed), from a clean clone` | repo | passed |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1, after locklog.py quiet-ok R040), from a clean clone` | tests | passed (72 / 72 tests) |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.4.0rc1 (v0.4.0, v0.4.0-rc.1, v0.4.0rc2 and v0.3.0 refused); ci/wheel_check.py --release-metadata (and --release-date 2026-11-20); ci/wheels.sh; ci/plugin_wheels.sh with DYNG_PLUGINS='cu12 cu13' and the CI toolkits of ci/cuda_toolkits.toml (CUDA 12.9 and 13.5, the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-11-20, ADR 0113; unpacked by ci/cuda_toolkit.py extract; verify --signatures passed); twine check --strict; ci/wheel_check.py on each file and --release-set --split; fresh Python 3.12 and 3.13 venvs: the core wheel alone, core + dyng-cu12, core + dyng-cu13, dyng[cu13]==0.4.0rc1 and dyng[cu12]==0.4.0rc1 from a local simple index laid out like TestPyPI (--extra-index-url PyPI), and the documented release-candidate forms (the CPU package with PyPI as the extra index; NumPy from PyPI, then --no-deps dyng and the plugin from the index): the wheel's test subset (cibuildwheel's test-command), the README quickstart, every Python snippet of the docs (label_propagation's and hyper_sssp's quickstarts among them), the tutorials 'Streaming label propagation' and 'Shortest hyperpaths' (their tests) and the label_propagation example against its committed digests, dyng --version, ci/plugin_smoke.py --expect cuda and pytest -m gpu on GPU 1; from a clean clone, the whole run under the shared lock` | packaging | passed |
| gpu_local | `ci/gpu_local.sh --preset parity-cuda (GPU 1, every step, one run; DYNG_LOCKLOG_MILESTONE=R040), from a clean clone` | repo | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (mosp) passed, synccheck (cycle_count) passed, racecheck passed, cbst sanitizers (racecheck, synccheck, initcheck) passed, cbst replays (device, small sets) passed, cbst replays (device, default preset) passed, cbst bin/main recorded and replayed once passed, cbst lockstep with the host model (compact after every op; every preset) passed, hypergraph sanitizers (racecheck, synccheck, initcheck) passed, hypergraph replays (cuda, small sets; ids, rows, line graph, legacy generator) passed, triad_count sanitizers (synccheck, initcheck; a coauth batch; racecheck) passed, count_local_patterns (sanitizers on the tutorial and API tests; nvcc consumer) passed, label_propagation (CUDA tests, kit, replays of P, in_place cloud, oracle; sanitizers) passed, slack_csr sanitizers (all four tools; a forced-overflow session) passed, slack_csr replays (cuda, make test; ids, pairs, counters, device layout) passed, hyper_sssp sanitizers (all four tools; the engine's tests, forced fallbacks) passed, hyper_sssp replays (cuda, make test; three modes, both engines, update == compute) passed, plugin wheel (pytest -m gpu) passed, clang-tidy passed) |

<!-- certify:end -->
