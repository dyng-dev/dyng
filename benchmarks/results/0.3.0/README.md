# dynG 0.3.0: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.3.0 (`sssp`, `cycle_count` and `mosp`, and, for the first time, the dynamic
hypergraph and `triad_count`, stable from 0.3.0), with the release checks of PLAN Section 10.3
steps 1-4. `parity.json` is the machine-readable certificate; the tables at the end of this page
are generated from it by `parity/certify.py write`. `benchmarks/README.md` (the developer guide's
"Benchmark suites" page) describes the tools and the file formats;
`docs/developer/retrospectives/R030.md` (steps 4 and 5) tells how the release candidate's
certificate was measured, `docs/developer/retrospectives/R031.md` how it was carried over to
0.3.0.

## Carried over from 0.3.0rc1

This certificate is the release candidate's (`benchmarks/results/0.3.0rc1/`), **carried over**
as `docs/developer/release.md` (step 9, "The final-release pull request") prescribes: the
library did not change since the release candidate's measured commits, which `certify.py write`
verified for every part in the paths of its scope. This certificate's measured commits are
`d803284` (the golden replays, the performance gates and the golden mutations, `library` scope,
plus `cpp/tests` for the golden mutations of the GoogleTest suites), `669fa60` (the hosted
sanitizer jobs of pull request #18, `library` scope), `d77620b` (the `tests` checks run again on
the final tree, below; the review's fixes and the acceptance review's fix after it changed no
path of the `tests` scope) and `9fff2a8` (the `packaging` and `repo` checks run a third time,
after the acceptance review's fix, whose diff to the final commit outside this directory is
empty); `git diff`
from each to the final commit touches no path of its scope. `a0f7177` (the release candidate's
sanitizer presets and mutation CTests), `7e918a2` (its tag day's re-run checks) and `177df79`
(the head of its first hosted sanitizer run) are not measured here: their records were replaced.

- **Carried over unchanged:** the golden replays (`parity-*.json`: sssp, cycle_count, mosp, the
  CBST trace goldens, the hypergraph container's replays and the triad goldens), the suite
  summaries and the compacted records of the performance gates of the five suites, and the golden
  mutations (`mutation-goldens.json`, measured at `d803284`), all described below as measured for
  0.3.0rc1. Only `VERSION` (0.3.0rc1, now 0.3.0), `CHANGELOG.md`, `CITATION.cff`, `README.md`,
  `python/README.md` (the PyPI project page), one Python test
  (`python/tests/test_triad_count_cli.py`, the documented `--format escher` output) and
  documents changed (R031, `docs/developer/retrospectives/R031.md`), none a library path.
- **Run again on the final tree** from fresh clones of the branch `release-0.3.0-final`: the
  `packaging` and `repo` checks at `9fff2a8`, the last commit outside this directory (first run at
  `d77620b`, again at `1a509a1` after the review's fixes of the pull request's branch, and a third
  time after the acceptance review's fix: `docs/developer/plan.md`'s status line dated
  2026-10-29, `c31bda5`, and `docs/developer/retrospectives/R031.md`; R031), the `tests`
  checks at `d77620b`; concurrently under the shared lock and never
  during a timing run, and recorded with `certify.py check --version 0.3.0` in place of the
  release candidate's records of the same names (logs under `$DYNG_SCRATCH/runs/r031/cert/`, the
  scripts in `$DYNG_SCRATCH/r031/cert/`):
  - the `packaging` check `distributions`: the distributions as `release.yml` builds them for the
    tag `v0.3.0` (`select` reports a final release: `prerelease=false`, `plugins=true`; the core
    sdist and wheel and the plugins `dyng-cu12` / `dyng-cu13` with CI's CUDA toolkits 12.9 and
    13.4, still the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-10-28, a fresh lock
    identical; the `collect` checks; fresh 3.12 / 3.13 venvs with the wheel's test
    subset, the README quickstart, every Python snippet of the docs and the GPU tests of both
    plugins on GPU 1; the pinned extras `dyng[cuN]==0.3.0` and the final release's documented
    install lines without a version, `pip install dyng`, `"dyng[cu13]"` and `"dyng[cu12]"`,
    against a local simple index of the three distributions with PyPI as the extra index; and the
    upgrade from 0.2.0: a venv with `dyng[cuN]==0.2.0` from PyPI keeps it for `pip install
    "dyng[cuN]"`, and `pip install -U "dyng[cuN]"`, the documented upgrade line, installs 0.3.0
    and the plugin of the same version, whose smoke test passes on GPU 1);
  - the `repo` checks `check-parity` (`ci/check.sh --parity`), `api-check` (`ci/api_check.sh
    20bc2d0`, against `main` as pushed, the merge of 0.3.0rc1) and `gpu_local`
    (`ci/gpu_local.sh`, every step, the compute-sanitizer steps and 0.3's CBST, hypergraph,
    triad_count and count_local_patterns steps included). The first `ci/gpu_local.sh` run at
    `1a509a1` (05:40-09:24 UTC) failed one step, transiently, as in R030 (deviation 7): every
    step passed except "cbst sanitizers", where compute-sanitizer's racecheck of
    `dyng_cbst_scenario_tests` ended with "process didn't terminate successfully", 0 hazards and
    no gtest output at all (the process ended before its buffered output was written), while the
    other six CBST executables passed under racecheck, synccheck and initcheck. The same command
    (`DYNG_CBST_REDUCED=1`, GPU 1) run alone three times from a fresh build of the same commit
    passed every time: 39 / 39 tests, 0 hazards. The test is unchanged since `d77620b`, where the
    step passed; without a reproduction and without access to the kernel log the cause is not
    known (another process on the shared machine or the sanitizer are the candidates). Nothing
    was changed: the whole of `ci/gpu_local.sh` was run again from a fresh clone of `1a509a1`
    (09:44-13:26 UTC, every step passed, with a 30-second monitor of the GPUs' processes), and
    that run was the `gpu_local` record at `1a509a1` (logs `gpu_local-1a509a1-run1-failed.log`,
    `gpu_local-1a509a1.log`, `cbst-scenario-racecheck-rerun1..3.log` and
    `gpu-monitor-1a509a1-run2.log` beside it; R031 deviation 7, an open item for 0.4.0). The
    run at `9fff2a8`, after the acceptance review's fix (14:29-18:17 UTC, concurrently with the
    `distributions` and `check-parity` runs under the shared lock, with the same monitor of GPU
    1, `gpu-monitor-9fff2a8.log`, whose samples show only dynG's own test processes), passed
    every step at its first attempt and is the `gpu_local` record; the logs of the `1a509a1` runs of the
    `packaging` and `repo` checks are kept beside the new ones with the suffix `-1a509a1`;
  - the `tests` checks `asan`, `tsan`, `tsan-openmp` and `mutation-ctests`, because `README.md`,
    whose C++ quickstart is a test, changed (its status block and install lines; the C++
    quickstart itself did not); at `d77620b`, whose `tests`-scope paths are the final commit's.
- **Replaced by a newer run:** `sanitizers-hosted`, the hosted `sanitizers.yml` jobs of the
  release candidate's pull request #18: run 40603416606 on its final head `669fa60` (2026-10-27,
  23:37-23:55 UTC; event `pull_request`, which tested the merge of `669fa60` into `e6f87ed`, the
  same tree: `asan / gcc-13` 1179 of 1179 tests, `tsan / gcc-13` 1024 of 1024, `tsan-openmp /
  clang-18` 1179 of 1179, all green), in place of the release candidate's record of run
  40568804466 on `177df79`; recorded with the scope `library`: its library is the release's, while
  `README.md` changed since (release.md step 3; R031, deviation 4).

## The smoke test of 0.3.0rc1 from TestPyPI (checklist step 8)

R031 step 1 (`docs/developer/retrospectives/R031.md` has the tables), 2026-10-28, 00:14-00:33 UTC
by the machine's clock (about 2.4 minutes behind real time), on the GPU machine (NVIDIA driver
590.48.01, CUDA 13.1; the GPU tests on GPU 1), every venv new (CPython 3.12.14 and 3.13.15), every
command in a clean environment (`env -i`, no dev environment on `PATH`, no pip cache), the tests,
examples and `ci/` scripts from the tag `v0.3.0rc1`:

- **The CPU package**, installed as the release candidate's README and install guide say
  (`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
  dyng==0.3.0rc1`): `import dyng`, `dyng.show_config()`, `sssp`, `cycle_count` and `mosp` compute
  and update and, new in 0.3, `dyng.Hypergraph` with `triad_count` compute and update (h-motif
  total 4 over 30 bins, after deleting hyperedge 3 total 1, delta sum -3; `stathyper_type1` total
  3) on `sequential` and `openmp` with equal results, `dyng --version` (`dyng 0.3.0rc1`), `dyng
  mosp compute` and `dyng triad_count compute` on both backends (byte-identical outputs), the
  example `examples/python/triad_count.py` on its data equal to
  `examples/data/triad_count_update.expected.txt` byte for byte on both backends; the licence
  metadata as decided; the tag's pytest suite: 735 passed, 82 skipped, on both Python versions.
- **Both CUDA plugins**, each in its own fresh venvs, installed as the docs say (NumPy and the
  test tools from PyPI, then `--no-deps dyng==0.3.0rc1 dyng-cuN==0.3.0rc1` from TestPyPI alone):
  `ci/plugin_smoke.py --expect cuda` on GPU 1 (sssp, cycle_count, mosp and triad_count on cuda
  equal to the sequential backend; cu12 built with CUDA 12.9, cu13 with 13.4), the fallback
  without a device and without a driver (`ci/without_cuda_driver.sh`), `pytest -m gpu` with
  `DYNG_REQUIRE_CUDA=1` (70 passed, 7 skipped: the CuPy / PyTorch round trips, not installed) and
  the suite with `DYNG_CPU_ONLY=1` (735 passed, 82 skipped): all passed for cu12 and cu13 on 3.12
  and 3.13. `pip install -i https://test.pypi.org/simple/ "dyng[cuN]==0.3.0rc1"` resolved to the
  plugin of the same version, and its smoke test passed.
- **The files tested are the published ones:** pip's installation reports name the same wheels in
  every venv, their SHA-256 equal to TestPyPI's digests and to the `dist` artifact of
  `release.yml` run 40606434243 (`dyng` `56565ce7...`, `dyng_cu12` `37dab82c...`, `dyng_cu13`
  `63f0e120...`, the sdist `1c602f90...`); pypi.org had 0.2.0 as the latest release of all three
  projects. The step-10 block of release.md ran once as written on that artifact (the plugin
  wheels' GPU tests on GPU 1), and passed.

The combined log is `$DYNG_SCRATCH/runs/r031/smoke-rc1.log` (copied to `checks/smoke-rc1.txt`;
sha256 `fea1115089f46f25...`), the scripts `$DYNG_SCRATCH/r031/run_smoke.sh`,
`smoke_rc_cpu.py` and `step10_dryrun.sh`. No separate CPU-only machine was available (R031,
deviation 2).

## What was run, and how (measured for 0.3.0rc1)

This section is the release candidate's, as its certificate described it. Its golden replays,
performance gates and golden mutations (at `d803284`) are the ones carried over here. Its
**check records** are not: the sanitizer presets and the mutation CTests at `a0f7177`, the
distributions of `v0.3.0rc1`, `ci/check.sh --parity`, `ci/api_check.sh e6f87ed` and
`ci/gpu_local.sh` at `7e918a2`, and the hosted `sanitizers.yml` run 40568804466 at `177df79` were
replaced in this directory's `checks.json` and `checks/` by the R031 runs at `d77620b` and
`9fff2a8` listed in "Carried over from 0.3.0rc1" above (the distributions of `v0.3.0`,
`ci/api_check.sh 20bc2d0`) and by the hosted run 40603416606 at `669fa60`. The statements below about which check records
stand, and at which commits, are the release candidate's; the generated tables at the end of this
page are this certificate's.

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

**Certificate:** `parity.json` (dynG 0.3.0, commit `ef00f8bd5b50`): **all parts passed**.

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
| mutation-goldens | `parity/mutate.py run (every mutation of sssp, cycle_count, mosp, cbst, hypergraph, kit, triad_count and triad_reference; parity-cuda copies, CUDA on GPU 1), from a clean clone` | library | passed |
| asan | `ci/sanitizers.sh asan (ctest -L cpu of the asan preset), from a clean clone of the final tree's commit` | tests | passed (1179 / 1179 tests, 65 skipped) |
| tsan | `ci/sanitizers.sh tsan (ctest -L cpu of the tsan preset), from a clean clone of the final tree's commit` | tests | passed (1024 / 1024 tests, 103 skipped) |
| tsan-openmp | `ci/sanitizers.sh tsan-openmp (Clang 18.1.8 of the conda env dyng-clang through a local wrapper: --gcc-install-dir GCC 12, --sysroot=/, -Wno-deprecated-declarations; Archer DYNG_ARCHER_LIBRARY = the env's libarcher.so.bak; ctest -L cpu), from a clean clone of the final tree's commit` | tests | passed (1179 / 1179 tests, 65 skipped) |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1), from a clean clone of the final tree's commit` | tests | passed (34 / 34 tests) |
| sanitizers-hosted | `the hosted sanitizers.yml jobs of pull request #18 (the release candidate's), run 40603416606 on its final head 669fa60 (event pull_request, tested as its merge 47df662 into e6f87ed, an ancestor of the head, so the same tree): asan / gcc-13, tsan / gcc-13, tsan-openmp / clang-18; the job logs read with gh run view --job <id> --log (read-only). It replaces the release candidate's record of run 40568804466 on 177df79. Scope library: README.md, a path of the tests scope, changed since 669fa60 (the status texts only); the tests scope is covered by the local presets run again on the final tree (docs/developer/release.md step 3)` | library | passed |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.3.0, a final release (kind=package, version=0.3.0, prerelease=false, plugins=true; v0.3.0rc1, v0.3.0-rc.1, v0.3.1 and v0.2.0 refused); ci/wheel_check.py --release-metadata (and --release-date 2026-10-29, the planned tag day); ci/wheels.sh; ci/plugin_wheels.sh with DYNG_PLUGINS='cu12 cu13' and the CI toolkits of ci/cuda_toolkits.toml (CUDA 12.9 and 13.4, still the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-10-28, a fresh lock identical, verify --signatures passed); twine check --strict; ci/wheel_check.py on each file and --release-set --split; fresh Python 3.12 and 3.13 venvs: the core wheel alone, core + dyng-cu12, core + dyng-cu13, and dyng[cu13]==0.3.0 and dyng[cu12]==0.3.0 from a local simple index of the three distributions with PyPI as the extra index: the wheel's test subset (cibuildwheel's test-command), the README quickstart, every Python snippet of the docs, dyng --version, ci/plugin_smoke.py --expect cuda (triad_count on cuda included) and pytest -m gpu on GPU 1; the final release's documented install lines without a version (pip install dyng, "dyng[cu13]", "dyng[cu12]") against that index: 0.3.0 and the plugin of the same version, the quickstart, the snippets, dyng --version and the plugin smoke test; the upgrade from 0.2.0: dyng[cuN]==0.2.0 from PyPI is kept by pip install "dyng[cuN]" and replaced by 0.3.0 and its plugin with pip install -U "dyng[cuN]", whose plugin smoke test passes on GPU 1; from a clean clone of the final tree's commit, the whole run under the shared lock` | packaging | passed |
| check-parity | `ci/check.sh --parity, from a clean clone of the final tree's commit (origin/main = 20bc2d0, main as pushed)` | repo | passed |
| api-check | `ci/api_check.sh 20bc2d0 (the frozen C++ API with ci/api_snapshot.py --against and the Python API with griffe, against main as pushed, the merge of 0.3.0rc1), from a clean clone of the final tree's commit` | repo | passed |
| gpu_local | `ci/gpu_local.sh (dev-cuda, GPU 1, every step), from a clean clone of the final tree's commit` | repo | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (mosp) passed, synccheck (cycle_count) passed, racecheck passed, cbst sanitizers (racecheck, synccheck, initcheck) passed, cbst replays (device, small sets) passed, cbst replays (device, default preset) passed, cbst bin/main recorded and replayed once passed, cbst lockstep with the host model (compact after every op; every preset) passed, hypergraph sanitizers (racecheck, synccheck, initcheck) passed, hypergraph replays (cuda, small sets; ids, rows, line graph, legacy generator) passed, triad_count sanitizers (synccheck, initcheck; a coauth batch; racecheck) passed, count_local_patterns (sanitizers on the tutorial and API tests; nvcc consumer) passed, plugin wheel (pytest -m gpu) passed, clang-tidy passed) |

<!-- certify:end -->
