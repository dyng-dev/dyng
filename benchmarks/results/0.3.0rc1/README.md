# dynG 0.3.0rc1: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.3.0rc1 (`sssp`, `cycle_count` and `mosp`, and, for the first time, the dynamic
hypergraph and `triad_count`, stable from 0.3.0), with the release checks of PLAN Section 10.3
steps 1-4. `parity.json` is the machine-readable certificate; the tables at the end of this page
are generated from it by `parity/certify.py write`. `benchmarks/README.md` (the developer guide's
"Benchmark suites" page) describes the tools and the file formats;
`docs/developer/retrospectives/R030.md` (step 4) tells how this certificate was made, on the
release's final library (the D13 fix of R030 step 1, ADR 0061, and the `.hg` reader's refusals of
the review fixes, ADR 0062, included), and step 5 how it was brought to the tag day's tree (the
release date 2026-10-28). It replaces the first certificate of 0.3.0rc1 (`75d6940`, R030 step 3),
measured before the review's library change.

## What was run, and how

- **Code.** The release branch `release-0.3.0` (from `main` at `e6f87ed`, the merge of M9b).
  Every measured build comes from a clean `git clone` of the branch (the commit each record
  names), with the presets `parity` and `parity-cuda`. The gates, the golden replays and the
  golden mutations ran at `d803284` (2026-10-23, 04:32-13:56 UTC; its library is the release's
  final one, `c9de41d`); the sanitizer presets, the mutation CTests and the portability
  pre-checks at `a0f7177`; on the tag day (R030 step 5, the release date moved to 2026-10-28) the
  distributions, `ci/check.sh --parity`, `ci/api_check.sh e6f87ed` (against `main` as pushed) and
  `ci/gpu_local.sh` again at `7e918a2`, the last commit outside this directory; the hosted
  `sanitizers.yml` jobs of the release pull request #18 at its head `177df79`.
  Three readings were taken again alone at the end (R030.md step 4, deviation 1): the coauth
  reading of `ipdps26_escher`, whose contamination audit flagged a round, and the two memory
  readings of `tkde26_escher_plus`, which refused to run beside another process on GPU 1; the
  flagged record is kept under `superseded/` of the full records. `parity.json` checks every measured commit against the release commit in the paths
  of its scope (`library` for the gates, the replays and the golden mutations, plus `cpp/tests`
  for the golden mutations of the GoogleTest suites, ADR 0061; `tests` for the C++ test suites;
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
- **The CBST trace goldens** (M8a; `parity-cbst_trace_<set>-device-<preset>.json`): all eight
  sets of ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159 replayed through the dynG store on the
  device under the original's preset (`escher_gpu` / `mosp_escher`, the exact field set) and under
  `dyng_default` (the policy field set), each checked against `parity/cbst_documented.toml`: 16 /
  16 replays OK; the only unreplayable operations are the documented ones (the three hash-only
  H2H rebuilds of `escher_gpu_coauth50k`; the 54 batch-mode operations of `escher_gpu_syn1m`,
  whose batches the hypergraph replays below cover).
- **The hypergraph container's replays** (M8b; `parity-<set>-hypergraph_ids-*.json`): the batch
  records of the five CBST trace sets with hypergraph batches replayed through `dyng::hypergraph`
  with the originals' ids and rows (`--mode both`) on sequential, OpenMP (8 threads) and CUDA, and
  ESCHER-GPU's three sets with the line graph against A's H2H after every batch on CUDA: 18 / 18
  runs, 0 mismatches.
- **The triad goldens** (M9a; `parity-triad_count_escher_gpu-*.json`): `parity/compare.py
  triad_count --counter algorithm --mode both` (by key and by content) on every backend of the
  coverage table of `parity/goldens.toml` (D6, accepted by the author on 2026-10-22): CUDA every
  case (300 / 300); sequential make-test and the syn100k grid (120 / 120); OpenMP the same plus
  the syn1m seed-1 grid and coauth_b50k_s1 for types 1 and 2, at 8 threads and at 1 thread (142 /
  142 each); all four patterns (h_motif and StatHyper's types 1-3).
- **Performance gates** (the five suites of `benchmarks/paper/` that `certify.py` requires from
  0.3: `ipdps25_dynamosp_sosp`, `ieee_tc_dyntrucy`, `ipdps25_dynamosp_mosp`, `ipdps26_escher` and
  `tkde26_escher_plus`; `parity/bench_suite.py run`, every gated row recomputed from the compacted
  records): against the unpatched originals, every timing under the exclusive perf lock,
  alternating rounds, medians; OpenMP sssp and mosp with 28 threads pinned to the cores, OpenMP
  cycle_count with 56; CUDA on GPU 0 with the clocks locked for the whole A/B (boost for sssp,
  mosp and cycle_count, the COLLAB update and triad_count at base, ADR 0021), the default-clock
  readings recorded and not gated (ADR 0018); device memory on GPU 1 (sssp, mosp and cycle_count
  under Nsight Systems; triad_count's store accounting against A's `deviceBytes()` and the Nsight
  Systems peak of live allocations of both whole processes, decision D1). **326 / 326** gated
  regions within their gates. 47 rounds were rejected by the monitors and repeated (foreign CPU
  load above 2 cores, at most 2.74; or a GPU clock below the lock in a cycle_count CUDA update);
  no accepted reading kept or flagged a contaminated round, and the triad contamination audit
  flags nothing in the records here. Against the first certificate of 0.3.0rc1 (the same suites
  and protocol, the library before the review's change), dynG's median gated rows moved -2.4 %
  to +2.8 % per suite, backend and gate kind (one row -11.2 %: cycle_count's OpenMP COLLAB count),
  the originals' -0.8 % to +2.0 %. The records here are compacted; each names its full copy under
  `$DYNG_SCRATCH/runs/bench/0.3.0rc1/`. Every record names the NVIDIA driver and carries the
  digests of its inputs taken by the measuring harness, all equal to the suites' digests.
- **rgg's unsafe50k batch** is byte-identical to its safe50k batch (`benchmarks/README.md`): its
  rows of the sssp and mosp suites are a second reading of the same input.
- **dynG against dynG since 0.2.0** (informational; the first certificate's comparison, which the
  medians above leave as it was). Against 0.2.0's certificate, dynG's median
  end-to-end rows moved +1.4 % to +4.8 % per suite and backend, the originals' -0.3 % to +2.4 %,
  the compute rows not (median -0.4 % to +2.2 %); road_usa read slower for both sides in this
  session. A stage A/B of
  `dyng-compat-mosp` (v0.2.0 against `8a7a43a`, sssp OpenMP, 11
  alternated processes per side) puts all of it in reading the graph (rgg 499.8 -> 555.9 ms,
  roadNet-PA 115.0 -> 128.4, roadNet-CA 202.8 -> 225.2), whose sources are 0.2.0's; with the
  JCC-erratum mitigation flags on both sides the difference is gone (rgg 455.1 / 454.1 ms,
  roadNet-CA 184.9 / 185.2). It is code placement on this CPU, the sensitivity of R020 (finding 12,
  investigation (b)), and a reading for the deferred layout option (`docs/roadmap.md`), not a
  library change. The worst gated end-to-end ratio of sssp and mosp is 1.024 here (sssp OpenMP,
  the apply of roadNet-PA safe50k; gate 1.10; 0.909 in the first certificate).
- **The D13 fix inside the triad_count gates.** In the first certificate, against M9b's records,
  dynG's construction (`build`) was 0.7-1.2 % faster on syn1M and syn2M, and the end-to-end ratios
  0.919-0.991 (M9b 0.920-0.992); here the end-to-end ratios are 0.935-0.990. The dynG-against-dynG
  reading of the fix on this library is `parity/results/R030.md` section 4 (ADR 0062).
- **Committed fixtures** (`cpp/tests/data`, `parity/fixtures/fixtures.toml`): each set with the
  SHA-256 of its files and the results of the CTests that compare it, taken from the per-test
  lines of the checks' ctest logs.
- **Checks** (`checks.json`; `certify.py` requires each for a 0.3 certificate): the golden
  mutations (`mutation-goldens.json`, `parity/mutate.py run`: the control and all 29 mutations of
  sssp, cycle_count, mosp, the CBST store, the hypergraph container, the conformance kit,
  triad_count and the triad oracle, each detected, read against `parity/mutate.py list`), the
  `asan`, `tsan` and `tsan-openmp` presets (`ci/sanitizers.sh`; `tsan-openmp` with the conda Clang
  18.1.8 through a local wrapper and the environment's Archer, as for 0.2.0), the hosted
  `sanitizers.yml` jobs of the release pull request #18 (`sanitizers-hosted`: run 40568804466 on
  its head `177df79`, `asan / gcc-13`, `tsan / gcc-13` and `tsan-openmp / clang-18`), the mutation
  CTests (`ctest --preset dev-cuda -L mutation`), the distributions as `release.yml` builds them
  for `v0.3.0rc1` (the core sdist and abi3 wheel and the plugins `dyng-cu12` / `dyng-cu13` with
  CI's CUDA toolkits 12.9 and 13.4, the `collect` checks, fresh 3.12 / 3.13 venvs including
  `dyng[cu13]==0.3.0rc1` and `dyng[cu12]==0.3.0rc1` from a local index laid out like TestPyPI and
  the documented release-candidate forms, the wheel's test subset, the README quickstart and every
  Python snippet of the docs, the GPU tests on GPU 1), `ci/check.sh --parity`, `ci/api_check.sh
  e6f87ed` (the frozen C++ API and the Python API against `main` as pushed) and `ci/gpu_local.sh`
  (every step, each of the series' steps passed as `certify.py` now requires, ADR 0062: the CUDA
  tests and goldens, compute-sanitizer memcheck, synccheck and racecheck, the CBST store, the
  hypergraph and triad_count steps, count_local_patterns, the plugin wheel, clang-tidy). Every check names its logs with their SHA-256; an excerpt of each is in `checks/`.

<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->

**Certificate:** `parity.json` (dynG 0.3.0rc1, commit `7e918a244f0d`): **all parts passed**.

Machine: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz (56 threads), 2x NVIDIA RTX A5000; driver 590.48.01 (CUDA driver API 13.1; named by 45 records); Build cuda_13.1.r13.1/compiler.37061995_0; c++ (Debian 12.2.0-14+deb12u1) 12.2.0.

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
| `cbst_trace_mosp_escher_dblp50k_hyperedge` | MOSP_ESCHER@4b86159 | 1 | device:dyng_default/policy: 1 / 0; device:mosp_escher/exact: 1 / 0; cuda:hypergraph_ids/both: 1 / 0; openmp/t8:hypergraph_ids/both: 1 / 0; sequential:hypergraph_ids/both: 1 / 0 | passed |
| `cbst_trace_mosp_escher_dblp50k_vertex` | MOSP_ESCHER@4b86159 | 1 | device:dyng_default/policy: 1 / 0; device:mosp_escher/exact: 1 / 0 | passed |
| `cbst_trace_escher_gpu_syn1m` | ESCHER-GPU@abcf9b2 | 3 | device:dyng_default/policy: 3 / 0; device:escher_gpu/exact: 3 / 0; cuda:hypergraph_ids/both: 3 / 0; cuda:hypergraph_ids/line_graph: 3 / 0; openmp/t8:hypergraph_ids/both: 3 / 0; sequential:hypergraph_ids/both: 3 / 0 | passed |
| `cbst_trace_mosp_escher_test` | MOSP_ESCHER@4b86159 | 24 | device:dyng_default/policy: 24 / 0; device:mosp_escher/exact: 24 / 0; cuda:hypergraph_ids/both: 24 / 0; openmp/t8:hypergraph_ids/both: 24 / 0; sequential:hypergraph_ids/both: 24 / 0 | passed |
| `triad_count_escher_gpu` | ESCHER-GPU@abcf9b2 | 300 | cuda:triad/by_key, cuda:triad/by_content: 600 / 0; openmp/t1:triad/by_key, openmp/t1:triad/by_content: 284 / 0; openmp/t8:triad/by_key, openmp/t8:triad/by_content: 284 / 0; sequential:triad/by_key, sequential:triad/by_content: 240 / 0 | passed |

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
| `ieee_tc_dyntrucy` | openmp | openmp | 15 / 15 | 0.423-0.944 | passed |
| `ieee_tc_dyntrucy` | openmp-collab | openmp | 1 / 1 | 0.296-0.296 | passed |
| `ieee_tc_dyntrucy` | cuda | cuda (boost) | 37 / 37 | 0.220-1.002 | passed |
| `ieee_tc_dyntrucy` | cuda-collab-update | cuda (base) | 5 / 5 | 0.945-0.996 | passed |
| `ieee_tc_dyntrucy` | cuda-memory | cuda | 20 / 20 | 1.000-1.001 | passed |
| `ieee_tc_dyntrucy` | cuda-default-clocks | cuda (none) | recorded, not gated (42) | 0.223-1.001 | - |
| `ipdps25_dynamosp_mosp` | openmp | openmp | 24 / 24 | 0.652-0.924 | passed |
| `ipdps25_dynamosp_mosp` | cuda | cuda (boost) | 24 / 24 | 0.681-1.006 | passed |
| `ipdps25_dynamosp_mosp` | cuda-memory | cuda | 12 / 12 | 0.931-1.019 | passed |
| `ipdps25_dynamosp_mosp` | cuda-default-clocks | cuda (none) | recorded, not gated (24) | 0.672-1.033 | - |
| `ipdps25_dynamosp_sosp` | openmp | openmp | 60 / 60 | 0.679-1.024 | passed |
| `ipdps25_dynamosp_sosp` | cuda | cuda (boost) | 60 / 60 | 0.673-1.001 | passed |
| `ipdps25_dynamosp_sosp` | cuda-memory | cuda | 12 / 12 | 0.812-0.858 | passed |
| `ipdps25_dynamosp_sosp` | cuda-default-clocks | cuda (none) | recorded, not gated (60) | 0.658-1.059 | - |
| `ipdps26_escher` | cuda-coauth | cuda (base) | 15 / 15 | 0.935-0.999 | passed |
| `ipdps26_escher` | cuda | cuda (base) | 30 / 30 | 0.852-1.015 | passed |
| `ipdps26_escher` | cuda-default-clocks | cuda (none) | recorded, not gated (15) | 0.839-1.021 | - |
| `ipdps26_escher` | cuda-batch-sizes | cuda (base) | recorded, not gated (10) | 0.944-1.016 | - |
| `tkde26_escher_plus` | cuda-memory | cuda | 11 / 11 | 1.000-1.002 | passed |
| `tkde26_escher_plus` | cuda-memory-batch-sizes | cuda | recorded, not gated (10) | 1.000-1.002 | - |
| `tkde26_escher_plus` | cuda-types | cuda (base) | recorded, not gated (15) | 0.117-0.993 | - |

Inputs of the readings:

- `ieee_tc_dyntrucy`: hashed by the measuring process.
- `ipdps25_dynamosp_mosp`: hashed by the harness when the measurement started.
- `ipdps25_dynamosp_sosp`: hashed by the harness when the measurement started.
- `ipdps26_escher`: hashed by the measuring process (coauth-DBLP's file; a synthetic input's rows as the golden set stores them, which both sides generate from the suite's parameters), with the guard requiring the golden's counts from every measured process (each round of a timed reading; each plain and profiled run of a memory reading).
- `tkde26_escher_plus`: hashed by the measuring process (coauth-DBLP's file; a synthetic input's rows as the golden set stores them, which both sides generate from the suite's parameters), with the guard requiring the golden's counts from every measured process (each round of a timed reading; each plain and profiled run of a memory reading).

### Checks

| Check | Command | Scope | Result |
|---|---|---|---|
| asan | `ci/sanitizers.sh asan (ctest -L cpu of the asan preset), from a clean clone` | tests | passed (1179 / 1179 tests, 65 skipped) |
| tsan | `ci/sanitizers.sh tsan (ctest -L cpu of the tsan preset), from a clean clone` | tests | passed (1024 / 1024 tests, 103 skipped) |
| tsan-openmp | `ci/sanitizers.sh tsan-openmp (Clang 18.1.8 of the conda env dyng-clang through a local wrapper: --gcc-install-dir GCC 12, --sysroot=/, -Wno-deprecated-declarations; Archer DYNG_ARCHER_LIBRARY = the env's libarcher.so.bak; ctest -L cpu), from a clean clone` | tests | passed (1179 / 1179 tests, 65 skipped) |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1), from a clean clone` | tests | passed (34 / 34 tests) |
| mutation-goldens | `parity/mutate.py run (every mutation of sssp, cycle_count, mosp, cbst, hypergraph, kit, triad_count and triad_reference; parity-cuda copies, CUDA on GPU 1), from a clean clone` | library | passed |
| sanitizers-hosted | `the hosted sanitizers.yml jobs of pull request #18 (the release candidate's), run 40568804466 (event pull_request, head 177df79, tested as its merge 8b6fedc into e6f87ed, an ancestor of the head, so the same tree): asan / gcc-13, tsan / gcc-13, tsan-openmp / clang-18; the job logs read with gh run view --log (read-only). Scope tests: no path of the tests scope changed from 177df79 to the release commit (docs/developer/release.md step 3)` | tests | passed |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.3.0rc1 (v0.3.0, v0.3.0-rc.1, v0.3.0rc2 and v0.2.0 refused); ci/wheel_check.py --release-metadata (and --release-date 2026-10-28, the tag day); ci/wheels.sh; ci/plugin_wheels.sh with DYNG_PLUGINS='cu12 cu13' and the CI toolkits of ci/cuda_toolkits.toml (CUDA 12.9 and 13.4, still the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-10-27, a fresh lock identical, unpacked by ci/cuda_toolkit.py extract; verify --signatures passed); twine check --strict; ci/wheel_check.py on each file and --release-set --split; fresh Python 3.12 and 3.13 venvs: the core wheel alone, core + dyng-cu12, core + dyng-cu13, dyng[cu13]==0.3.0rc1 and dyng[cu12]==0.3.0rc1 from a local simple index laid out like TestPyPI (--extra-index-url PyPI), and the documented release-candidate forms (the CPU package with PyPI as the extra index; NumPy from PyPI, then --no-deps dyng and the plugin from the index): the wheel's test subset (cibuildwheel's test-command), the README quickstart, every Python snippet of the docs (the hypergraph and triad_count quickstarts), dyng --version, ci/plugin_smoke.py --expect cuda and pytest -m gpu on GPU 1; from a clean clone of the tag day's commit, the whole run under the shared lock` | packaging | passed |
| check-parity | `ci/check.sh --parity, from a clean clone of the tag day's commit (origin/main = e6f87ed, main as pushed)` | repo | passed |
| api-check | `ci/api_check.sh e6f87ed (the frozen C++ API with ci/api_snapshot.py --against and the Python API with griffe, against main as pushed), from a clean clone of the tag day's commit` | repo | passed |
| gpu_local | `ci/gpu_local.sh (dev-cuda, GPU 1, every step), from a clean clone of the tag day's commit` | repo | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (mosp) passed, synccheck (cycle_count) passed, racecheck passed, cbst sanitizers (racecheck, synccheck, initcheck) passed, cbst replays (device, small sets) passed, cbst replays (device, default preset) passed, cbst bin/main recorded and replayed once passed, cbst lockstep with the host model (compact after every op; every preset) passed, hypergraph sanitizers (racecheck, synccheck, initcheck) passed, hypergraph replays (cuda, small sets; ids, rows, line graph, legacy generator) passed, triad_count sanitizers (synccheck, initcheck; a coauth batch; racecheck) passed, count_local_patterns (sanitizers on the tutorial and API tests; nvcc consumer) passed, plugin wheel (pytest -m gpu) passed, clang-tidy passed) |

<!-- certify:end -->
