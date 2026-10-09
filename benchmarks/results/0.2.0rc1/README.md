# dynG 0.2.0rc1: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.2.0rc1 (`sssp`, `cycle_count` and, for the first time, `mosp`, stable from 0.2.0),
with the release checks of PLAN Section 10.3 steps 1-4. `parity.json` is the machine-readable
certificate; the tables at the end of this page are generated from it by `parity/certify.py
write`. `benchmarks/README.md` (the developer guide's "Benchmark suites" page) describes the tools
and the file formats; `docs/developer/retrospectives/R020.md` (steps 4 and 5) tells how this
certificate was made. It replaces the first certificate of 0.2.0rc1 (commit `f90fb9c`), measured
before the independent reviews of the branch: one review fix changed the library (the
CSR-triplet reader, below), so every part was measured again on the final code.

## What was run, and how

- **Code.** The release branch `release-0.2.0` (from `main` at `8b9067e`, the merge of M6b).
  Every measured build comes from a clean `git clone` of the branch (the commit each record
  names). The gates, the golden replays, the golden mutations, the C++ test suites (the sanitizer
  presets, the mutation CTests) and the portability pre-checks ran at `b1a4869`, the last commit
  of the `library` scope; the distributions, `ci/check.sh --parity`, `ci/api_check.sh 8b9067e`
  and `ci/gpu_local.sh` at the last commit outside this directory (`616718a`: the release date
  was moved to the tag day, 2026-10-09, and these four were run again, `release.md`, "A tag day
  later than the final-release pull request"; they replace the records at `b1a4869` and
  `866ba5f`). `parity.json` checks every measured commit against the release commit in the
  paths of its scope (`library` for the gates, the replays and the golden mutations; `tests` for
  the C++ test suites; `packaging` for the distributions; `repo`, the whole tree but this
  directory, for `ci/check.sh`, `ci/api_check.sh` and `ci/gpu_local.sh`). No equivalence record
  is needed: no measured commit differs from the release in its scope.
- **Golden parity** (`parity-*.json`, `parity/compare.py`, the `parity` and `parity-cuda` presets;
  CUDA replays on GPU 1): the sssp corpus of MOSP-OpenMP@c352151 (495 cases, each the whole MOSP
  update with the combined graph through `dyng-compat-mosp`) on sequential, OpenMP 1 / 4 / 16 /
  28 / 56 threads and int64 offsets, and on CUDA with the automatic, the fused and the operators
  engine and int64 offsets; the cycle_count CPU corpus of CycleEnumeration-GPU@0a976ad with
  `--full` (24 cases on sequential, OpenMP 4 and 56 threads) and its CUDA corpus (24 cases on
  cuda, cuda:resident, cuda:int64); the paper-scale mosp set `mosp_scale` (20 cases: the four gate
  graphs x three batches, Pref {4, 1, 4}, the K sweep; every output file's SHA-256 and the
  `invalidated` counters) on sequential, OpenMP 28, CUDA with the automatic engine (`cuda`) and
  CUDA with the operators engine (`cuda-operators`).
- **Performance gates** (`ipdps25_dynamosp_sosp*.json`, `ieee_tc_dyntrucy*.json`,
  `ipdps25_dynamosp_mosp*.json`; `parity/bench_suite.py run` on the three suites of
  `benchmarks/paper/`, which `certify.py` requires from 0.2, and which recomputes every gated row
  from the compacted records): against the unpatched originals, every timing under the exclusive
  perf lock, alternating rounds, medians; OpenMP sssp and mosp with 28 threads pinned to the
  cores, OpenMP cycle_count with 56; CUDA on GPU 0 with the clocks locked for the whole A/B (boost;
  the COLLAB update at base, ADR 0021), the default-clock readings recorded and not gated (ADR
  0018); device memory under Nsight Systems on GPU 1. The records here are compacted; each names
  its full copy under `$DYNG_SCRATCH/runs/bench/0.2.0rc1/`. Every record names the NVIDIA driver
  and carries the digests of its inputs taken by the measuring harness, all equal to the suites'
  digests.
- **rgg's unsafe50k batch** is byte-identical to its safe50k batch (the original's `--safe` filter
  drops only the deletions that disconnect a vertex from the source, and none of the seed-777
  batch's deletions on rgg_n_2_20_s0 does): its rows of the three suites are a second reading of
  the same input, not independent coverage of unsafe deletions.
- **One reading taken again.** Of the mosp suite's first pass, the OpenMP reading of roadNet-PA
  was stopped by the contamination monitor without a record (22 rejected rounds, while the
  unlocked venv tests of this certificate's own distributions check ran). It was taken again
  alone from the same clone and build with `parity/bench_suite.py run ... --skip-existing`
  (20:04-20:06 UTC on 2026-10-08, no round rejected, foreign load at most 1.23 cores); the
  summary covers the whole suite. Every other reading is from the first pass, without kept or
  flagged contaminated rounds (R020.md, step 5, deviation 5).
- **dynG against dynG since 0.1.0.** The first certificate of 0.2.0rc1 showed dynG's end to end
  5-12 % slower than in 0.1.0's certificate on every sssp case (and mosp's 4-9 % slower than in
  M7), all of it in reading the graph (a dynG-vs-dynG stage A/B; the reader's success path was the
  same source as 0.1.0's, so most likely code placement). `b1a4869` makes the CSR-triplet reader
  parse a plain number while it scans it: the reader alone on rgg 578 ms (0.1.0) / 692 ms (first
  certificate) / 507 ms (this one). End to end against 0.1.0's certificate (the same suite file and protocol), dynG's 24 sssp
  readings now move by a median of -3.6 % (-11.8 to -1.0 %; the first certificate: +8.2 %, +5.3
  to +12.2 %) and the originals' by -0.6 %, with `sosp_total` and `apply` unchanged (median 0.0 %
  and -0.8 %); for example OpenMP rgg safe50k 967.5 ms (0.1.0) / 1079.9 ms (first certificate) /
  890.4 ms (this one). mosp's 24 end-to-end readings move by -9.2 % against the first
  certificate and by -4.4 % against M7's readings (`parity/results/M7.md`), its compute by 0.0 % (R020.md, step 5).
- **Committed fixtures** (`cpp/tests/data`, `parity/fixtures/fixtures.toml`): each set with the
  SHA-256 of its files and the results of the CTests that compare it, taken from the per-test
  lines of the checks' ctest logs.
- **Checks** (`checks.json`; `certify.py` requires each for a 0.2 certificate): the golden
  mutations (`mutation-goldens.json`, `parity/mutate.py run`: the control and all eleven mutations
  of sssp, cycle_count and mosp on the sequential, OpenMP and CUDA backends, each detected, read
  against `parity/mutate.py list`), the `asan`, `tsan` and `tsan-openmp` presets
  (`ci/sanitizers.sh`; `tsan-openmp` with the conda Clang 18.1.8 through a local wrapper and
  Archer, R020.md step 4, deviation 2; the hosted `sanitizers.yml` jobs of the release pull
  request are recorded once it is open, R020.md step 5, deviation 2), the mutation CTests (`ctest
  --preset dev-cuda -L mutation`), the distributions as `release.yml` builds them for
  `v0.2.0rc1` (the core sdist and wheel and the plugins `dyng-cu12` / `dyng-cu13` with CI's CUDA
  toolkits 12.9 and 13.4, the `collect` checks, fresh 3.12 / 3.13 venvs including
  `dyng[cu13]==0.2.0rc1` from a local index laid out like TestPyPI and the documented
  `--no-deps` form, the GPU tests on GPU 1), `ci/check.sh --parity`, `ci/api_check.sh 8b9067e`
  (the frozen C++ API and the Python API against `main` as pushed) and `ci/gpu_local.sh` (the CUDA
  tests and goldens, compute-sanitizer memcheck, synccheck and racecheck, the plugin wheel,
  clang-tidy). Every check names its logs with their SHA-256; an excerpt of each is in `checks/`.

<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->

**Certificate:** `parity.json` (dynG 0.2.0rc1, commit `616718aa4a62`): **all parts passed**.

Machine: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz (56 threads), 2x NVIDIA RTX A5000; driver 590.48.01 (CUDA driver API 13.1; named by 38 records); Build cuda_13.1.r13.1/compiler.37061995_0; c++ (Debian 12.2.0-14+deb12u1) 12.2.0.

### Golden parity

| Golden set | Original | Cases | Replays (configurations: compared / failed) | Result |
|---|---|---:|---|---|
| `sssp` | MOSP-OpenMP@c352151 | 495 | sequential, openmp:1, openmp:4, openmp:16, openmp:28, openmp:56, sequential/int64, openmp:4/int64: 3960 / 0; cuda, cuda-fused, cuda-operators, cuda/int64, cuda-operators/int64: 2475 / 0 | passed |
| `cycle_count` | CycleEnumeration-GPU@0a976ad | 24 | sequential, openmp:4, openmp:56: 72 / 0 | passed |
| `cycle_count_cuda` | CycleEnumeration-GPU@0a976ad | 24 | cuda, cuda:resident, cuda:int64: 72 / 0 | passed |
| `mosp_scale` | MOSP-OpenMP@c352151 | 20 | sequential, openmp:28: 40 / 0; cuda, cuda-operators: 40 / 0 | passed |

### Committed fixtures (`cpp/tests/data`, `parity/fixtures/fixtures.toml`)

| Fixture set | Original | Files | SHA-256 | Tests (checks) | Result |
|---|---|---:|---|---|---|
| `mosp_changes` | MOSP-OpenMP@c352151 | 49 | `989f8b4b7d69f88e...` | 1 patterns, 3 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_graph_io` | MOSP-OpenMP@c352151 | 372 | `664b54743f786a9b...` | 2 patterns, 7 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_sssp` | MOSP-OpenMP@c352151 | 482 | `3fd1f76321898904...` | 3 patterns, 48 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `cycle_enum` | CycleEnumeration-GPU@0a976ad | 529 | `5ba52e2cfda6afcd...` | 13 patterns, 20 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |
| `mosp_combined` | MOSP-OpenMP@c352151 | 36 | `3e3b73b6822b7a3e...` | 2 patterns, 27 tests (asan, check-parity, gpu_local, tsan, tsan-openmp) | passed |

### Performance gates (PLAN 8.6, against the unpatched originals)

| Suite | Reading | Backend (clocks) | Gated regions | Ratio range (port / original) | Result |
|---|---|---|---:|---|---|
| `ieee_tc_dyntrucy` | openmp | openmp | 15 / 15 | 0.440-0.951 | passed |
| `ieee_tc_dyntrucy` | openmp-collab | openmp | 1 / 1 | 0.328-0.328 | passed |
| `ieee_tc_dyntrucy` | cuda | cuda (boost) | 37 / 37 | 0.220-1.004 | passed |
| `ieee_tc_dyntrucy` | cuda-collab-update | cuda (base) | 5 / 5 | 0.945-0.999 | passed |
| `ieee_tc_dyntrucy` | cuda-memory | cuda | 20 / 20 | 1.000-1.001 | passed |
| `ieee_tc_dyntrucy` | cuda-default-clocks | cuda (none) | recorded, not gated (42) | 0.226-1.003 | - |
| `ipdps25_dynamosp_mosp` | openmp | openmp | 24 / 24 | 0.597-0.927 | passed |
| `ipdps25_dynamosp_mosp` | cuda | cuda (boost) | 24 / 24 | 0.710-1.005 | passed |
| `ipdps25_dynamosp_mosp` | cuda-memory | cuda | 12 / 12 | 0.931-1.019 | passed |
| `ipdps25_dynamosp_mosp` | cuda-default-clocks | cuda (none) | recorded, not gated (24) | 0.712-1.008 | - |
| `ipdps25_dynamosp_sosp` | openmp | openmp | 60 / 60 | 0.703-1.036 | passed |
| `ipdps25_dynamosp_sosp` | cuda | cuda (boost) | 60 / 60 | 0.643-1.007 | passed |
| `ipdps25_dynamosp_sosp` | cuda-memory | cuda | 12 / 12 | 0.812-0.858 | passed |
| `ipdps25_dynamosp_sosp` | cuda-default-clocks | cuda (none) | recorded, not gated (60) | 0.639-1.061 | - |

Inputs of the readings:

- `ieee_tc_dyntrucy`: hashed by the measuring process.
- `ipdps25_dynamosp_mosp`: hashed by the harness when the measurement started.
- `ipdps25_dynamosp_sosp`: hashed by the harness when the measurement started.

### Checks

| Check | Command | Scope | Result |
|---|---|---|---|
| mutation-goldens | `parity/mutate.py run (every mutation of sssp, cycle_count and mosp; parity-cuda copies, CUDA on GPU 1), from a clean clone` | library | passed |
| asan | `ci/sanitizers.sh asan (ctest -L cpu of the asan preset), from a clean clone` | tests | passed (839 / 839 tests, 50 skipped) |
| tsan | `ci/sanitizers.sh tsan (ctest -L cpu of the tsan preset), from a clean clone` | tests | passed (745 / 745 tests, 85 skipped) |
| tsan-openmp | `ci/sanitizers.sh tsan-openmp (Clang 18.1.8 of the conda env dyng-clang through a local wrapper: --gcc-install-dir GCC 12, --sysroot=/, the rpath of its libomp, -Wno-error=deprecated-declarations; Archer DYNG_ARCHER_LIBRARY; ctest -L cpu), from a clean clone` | tests | passed (839 / 839 tests, 50 skipped) |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1), from a clean clone` | tests | passed (7 / 7 tests) |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.2.0rc1 (v0.2.0 and v0.2.0-rc.1 refused); ci/wheel_check.py --release-metadata (and --release-date today, the tag day 2026-10-09); ci/wheels.sh; ci/plugin_wheels.sh with DYNG_PLUGINS='cu12 cu13' and the CI toolkits of ci/cuda_toolkits.toml (CUDA 12.9 and 13.4, still the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-10-08, unpacked by ci/cuda_toolkit.py extract; verify --signatures passed); twine check --strict; ci/wheel_check.py on each file and --release-set --split; fresh Python 3.12 and 3.13 venvs: the core wheel alone, core + dyng-cu12, core + dyng-cu13, dyng[cu13]==0.2.0rc1 from a local simple index laid out like TestPyPI (--extra-index-url PyPI), and the documented form (NumPy from PyPI, then --no-deps dyng dyng-cu13 from the index): the wheel's test subset (test_mosp.py included), the README quickstart, dyng --version, ci/plugin_smoke.py --expect cuda and pytest -m gpu on GPU 1; from a clean clone, the whole run under the shared lock` | packaging | passed |
| check-parity | `ci/check.sh --parity, from a clean clone (origin/main set to 8b9067e, main as pushed)` | repo | passed |
| api-check | `ci/api_check.sh 8b9067e (the frozen C++ API with ci/api_snapshot.py --against and the Python API with griffe, against main as pushed), from a clean clone` | repo | passed |
| gpu_local | `ci/gpu_local.sh (dev-cuda, GPU 1, every step), from a clean clone` | repo | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (mosp) passed, synccheck (cycle_count) passed, racecheck passed, plugin wheel (pytest -m gpu) passed, clang-tidy passed) |

<!-- certify:end -->
