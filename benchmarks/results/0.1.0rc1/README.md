# dynG 0.1.0rc1: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.1.0rc1 (`sssp` and `cycle_count`), with the release checks of PLAN Section 10.3
steps 1-3. `parity.json` is the machine-readable certificate; the tables at the end of this page
are generated from it by `parity/certify.py write`. `benchmarks/README.md` (the developer guide's
"Benchmark suites" page) describes the tools and the file formats.

## What was run, and how

- **Code.** The release branch `release-0.1.0`. Its library, compat tools and build system are
  those of `main` at `8842acf` (the merge of M5); R010 adds harness scripts, documents and two
  test fixes. Every measured build comes from a clean `git clone` of the branch (the commit each
  record names); `parity.json` checks that every measured commit has the release commit's
  sources in the paths of its scope (`library` for the gates, the replays and the golden
  mutations; `tests` for the C++ test suites; `packaging` for the distributions; `repo`, the
  whole tree but this directory, for `ci/check.sh` and `ci/gpu_local.sh`).
- **Golden parity** (`parity-*.json`, `parity/compare.py`, the `parity` and `parity-cuda`
  presets; CUDA replays on GPU 1): the sssp corpus of MOSP-OpenMP@c352151 (495 cases) on
  sequential, OpenMP 1 / 4 / 16 / 28 / 56 threads, int64 offsets, CUDA and CUDA int64 (byte
  equality with the originals, which MOSP-CUDA@e220ee2 equals); the cycle_count CPU corpus of
  CycleEnumeration-GPU@0a976ad with `--full` (24 cases on sequential, OpenMP 4 and 56 threads) and
  its CUDA corpus (24 cases on cuda, cuda:resident, cuda:int64).
- **Performance gates** (`ipdps25_dynamosp_sosp*.json`, `ieee_tc_dyntrucy*.json`,
  `parity/bench_suite.py run` on the suites of `benchmarks/paper/`): against the unpatched
  originals, every timing under the exclusive perf lock, alternating rounds, medians; OpenMP sssp
  with 28 threads pinned to the cores, OpenMP cycle_count with 56 threads; CUDA on GPU 0 with the
  clocks locked for the whole A/B (boost; the COLLAB update at base, ADR 0021), the default-clock
  readings recorded and not gated (ADR 0018); device memory under Nsight Systems on GPU 1. The
  records here are compacted (the per-round machine-monitor windows are dropped); each names its
  full copy under `$DYNG_SCRATCH/runs/bench/0.1.0rc1/`. Each summary covers its whole suite
  (`certify.py` checks it against the suite's plan and the suite file's SHA-256).
- **Inputs and driver** (R010 review). The cycle_count run records carry the datasets' digests
  their measuring process took. The sssp records carry the digests `perf_ab.py prepare` wrote
  (the harness hashes at run start only since the review), and the memory records none: for
  these, `bench_suite.py summarize` hashed every input file again and found each equal to the
  suite's digest and unchanged (modification and status-change times) since the execution
  started (sssp 2026-10-01 02:28 UTC, cycle_count 05:40 UTC; the files were last changed
  between 2026-09-23 and 2026-09-27). The records predate the per-record driver field; `parity.json` shows the driver
  seen when the certificate was written (590.48.01, CUDA driver API 13.1) unchanged since before
  the first measurement (the loaded kernel module, boot 2026-08-10, the module file installed
  2026-08-05, `libcuda.so.590.48.01` installed 2026-02-05).
- **Committed fixtures** (`cpp/tests/data`, `parity/fixtures/fixtures.toml`): the four sets
  exported from the originals (`mosp_changes`, the generator outputs of `mospPrep changes`;
  `mosp_graph_io`; `mosp_sssp`; `cycle_enum`), each with the SHA-256 of its files and the results
  of the CTests that compare it, taken from the per-test lines of the checks' ctest logs.
- **Checks** (`checks.json`): the `asan` and `tsan` presets' test suites, `ci/gpu_local.sh` (the
  CUDA tests, the CUDA goldens, compute-sanitizer memcheck, synccheck and racecheck, clang-tidy on
  the CUDA branches), the mutation CTests of cycle_count (`ctest -L mutation`, host and CUDA),
  the golden mutations of sssp (`mutation-sssp.json`, `parity/mutate.py`), `ci/check.sh
  --parity` and the distributions as `release.yml` builds them for `v0.1.0rc1`. Every check
  names its logs with their SHA-256; an excerpt (or, for a short log, a copy) of each is in
  `checks/`. `ci/gpu_local.sh`, `ci/check.sh --parity` and the distributions were run again
  after the R010 review on its final code (scopes `repo` and `packaging`); the C++ test suites
  of `07d46e2` stand, since no test path changed after them.
- **Readings taken again.** The first pass of the gate (2026-09-30 21:28 to 2026-10-01 01:51
  CDT, one clone at `d13d393`) ran partly under a foreign CPU load of about 27 cores on the shared
  machine. The CUDA sssp reading of `road_usa_g` at locked clocks and the default-clock reading of
  `roadNet-PA` stopped after 12 rejected rounds (the harness exits 1 by design); the OpenMP
  cycle_count reading, whose harness flags such runs instead of repeating them, had almost every
  run flagged (median foreign load 16-28 cores); the default-clock cycle_count reading was
  interrupted, so the first pass never reached the cycle_count CUDA memory reading planned after
  it. The four readings were taken again, and the memory reading was taken for the first time,
  from the same clone and builds on 2026-10-01 between 02:04 and 02:58 CDT on a quiet machine:
  five readings of the second session (the memory reading ran last, 02:55 to 02:58). The
  flagged and the partial records are kept under `$DYNG_SCRATCH/runs/bench/0.1.0rc1/superseded/`
  and are not used. In the OpenMP cycle_count reading that replaced the flagged one, one run of
  11 is flagged (`count/DD_k3`, original side), which the median absorbs; the suite summary lists
  it under `contaminated`. Every other reading comes from the first pass, taken without rejected
  or flagged rounds beyond the harness's repeats, and is kept.
- **The release commit and the measured builds.** Making `sssp` and `cycle_count` stable (R010,
  step 2) changes three files of `cpp/src` after the measurements: the two manifests, which no
  build reads, and the registry table `cpp/src/core/registry_table.inc` that `scripts/regen.py`
  generates from them, read only by `dyng::algorithms()`. The gates, the golden replays and the
  sssp golden mutations were not measured again: `equivalence.json` (`parity/certify.py
  equivalence`) builds `d13d393` and, in the same clone, the release's library paths over it
  (presets `parity` and `parity-cuda`), and finds both compat tools byte-identical and `libdyng`
  differing only inside `dyng::algorithms()` (and the build id, unwind and symbol tables), with
  every section and symbol at the same address. The test-suite checks (the `asan` and `tsan`
  presets, `ci/gpu_local.sh`, the mutation CTests) read the registry, so they were run again on
  the release code (`07d46e2`, a clean clone). `VERSION` (0.1.0rc1 in the release, 0.1.0.dev0 in
  the measured builds) is not a certified path, as before.

<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->

**Certificate:** `parity.json` (dynG 0.1.0rc1, commit `26a330b28af4`): **all parts passed**.

Machine: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz (56 threads), 2x NVIDIA RTX A5000; driver 590.48.01; Build cuda_13.1.r13.1/compiler.37061995_0; c++ (Debian 12.2.0-14+deb12u1) 12.2.0.

Measured at `d13d393f8ea5`, whose library differs from the release's only in generated metadata (`cpp/src/algorithms/cycle_count/manifest.toml`, `cpp/src/algorithms/sssp/manifest.toml`, `cpp/src/core/registry_table.inc`); `equivalence.json` compares the builds of both: parity/cpp/libdyng.so equivalent, parity/tools/compat/dyng-compat-mosp identical, parity/tools/compat/dyng-compat-cycle-enum identical, parity-cuda/cpp/libdyng.so equivalent, parity-cuda/tools/compat/dyng-compat-mosp identical, parity-cuda/tools/compat/dyng-compat-cycle-enum identical.

### Golden parity

| Golden set | Original | Cases | Replays (configurations: compared / failed) | Result |
|---|---|---:|---|---|
| `sssp` | MOSP-OpenMP@c352151 | 495 | sequential, openmp:1, openmp:4, openmp:16, openmp:28, openmp:56, sequential/int64, openmp:4/int64: 3960 / 0; cuda, cuda/int64: 990 / 0 | passed |
| `cycle_count` | CycleEnumeration-GPU@0a976ad | 24 | sequential, openmp:4, openmp:56: 72 / 0 | passed |
| `cycle_count_cuda` | CycleEnumeration-GPU@0a976ad | 24 | cuda, cuda:resident, cuda:int64: 72 / 0 | passed |

### Performance gates (PLAN 8.6, against the unpatched originals)

| Suite | Reading | Backend (clocks) | Gated regions | Ratio range (port / original) | Result |
|---|---|---|---:|---|---|
| `ieee_tc_dyntrucy` | openmp | openmp | 15 / 15 | 0.438-0.981 | passed |
| `ieee_tc_dyntrucy` | openmp-collab | openmp | 1 / 1 | 0.328-0.328 | passed |
| `ieee_tc_dyntrucy` | cuda | cuda (boost) | 37 / 37 | 0.229-1.005 | passed |
| `ieee_tc_dyntrucy` | cuda-collab-update | cuda (base) | 5 / 5 | 0.947-1.002 | passed |
| `ieee_tc_dyntrucy` | cuda-memory | cuda | 20 / 20 | 1.000-1.001 | passed |
| `ieee_tc_dyntrucy` | cuda-default-clocks | cuda (none) | recorded, not gated (42) | 0.222-1.003 | - |
| `ipdps25_dynamosp_sosp` | openmp | openmp | 60 / 60 | 0.631-1.046 | passed |
| `ipdps25_dynamosp_sosp` | cuda | cuda (boost) | 60 / 60 | 0.650-1.007 | passed |
| `ipdps25_dynamosp_sosp` | cuda-memory | cuda | 12 / 12 | 0.812-0.858 | passed |
| `ipdps25_dynamosp_sosp` | cuda-default-clocks | cuda (none) | recorded, not gated (60) | 0.666-1.062 | - |

### Checks

| Check | Command | Result |
|---|---|---|
| mutation-sssp | `parity/mutate.py run` | passed |
| tsan | `ctest --preset tsan -j 16 (tsan preset: Debug, DYNG_SANITIZE=thread, OpenMP off), from a clean clone` | passed (520 / 520 tests, 42 skipped) |
| asan | `ctest --preset asan -j 16 (asan preset: Debug, DYNG_SANITIZE=address;undefined, OpenMP on), from a clean clone` | passed (595 / 595 tests, 26 skipped) |
| gpu_local | `ci/gpu_local.sh (dev-cuda, GPU 1), from a clean clone` | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (cycle_count) passed, racecheck passed, clang-tidy passed) |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1), from a clean clone` | passed (7 / 7 tests) |
| check-parity | `ci/check.sh --parity, from a clean clone` | passed |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.1.0rc1; ci/wheels.sh; twine check --strict; ci/wheel_check.py --platform manylinux_2_28_x86_64 --require-libgomp; fresh Python 3.12 and 3.13 venvs, from a clean clone` | passed |

<!-- certify:end -->
