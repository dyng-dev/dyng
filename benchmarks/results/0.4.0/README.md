# dynG 0.4.0: the release certificate

The parity certificate of PLAN Section 8.3 and the release performance gate of PLAN Section 8.6
for dynG 0.4.0 (`sssp`, `cycle_count`, `mosp`, the dynamic hypergraph and `triad_count`, and,
for the first time, `label_propagation` against LabelPropagation-CUDA and `hyper_sssp` against
MOSP_ESCHER with the `slack_csr` store, stable from 0.4.0), with the release checks of PLAN
Section 10.3 steps 1-4. `parity.json` is the machine-readable certificate; the tables at the end
of this page are generated from it by `parity/certify.py write`. `benchmarks/README.md` (the
developer guide's "Benchmark suites" page) describes the tools and the file formats;
`docs/developer/retrospectives/R040.md` (steps 3 and 4 and the tag day) tells how the release
candidate's certificate was measured, with ADR 0024's reading of the stable algorithms against
0.3.0, which is not part of this certificate (the author decided its two STOPs on 2026-11-21:
GOVERNANCE.md's approvals log), and `docs/developer/retrospectives/R041.md` how it was carried
over to 0.4.0.

## Carried over from 0.4.0rc1

This certificate is the release candidate's (`benchmarks/results/0.4.0rc1/`), **carried over** as
`docs/developer/release.md` (step 9, "The final-release pull request") prescribes: the library did
not change since the release candidate's measured commits, which `certify.py write` verified for
every part in the paths of its scope. This certificate's measured commits are `f60a2f3` (the golden
replays, the performance gates and the golden mutations, `library` scope, plus `cpp/tests` for the
golden mutations of the GoogleTest suites), `3b8c651` (the hosted sanitizer jobs of pull request
#27, `library` scope), `fe7ecc2` (the `tests` checks run again on the final tree, below) and
`91b966e` (the `packaging` and `repo` checks, run once more after R041 step 4, which changed
`CHANGELOG.md` and R041.md only: the last commit outside this directory); `git diff` from each to
the final commit touches no path of its scope. `2616af6` (the release candidate's sanitizer presets
and mutation CTests) and `77e7646` (its tag day's `packaging` and `repo` checks) are not measured
here: their records were replaced.

- **Carried over unchanged:** the golden replays (`parity-*.json`: sssp, cycle_count, mosp, the
  CBST trace goldens, the hypergraph container's replays with MOSP_ESCHER's whole `slack_csr`
  sessions, the triad goldens, the label_propagation golden sets with the in_place set under ADR
  0111's rate rule and the harmonic oracle's release replay, and hyper_sssp's modes on both CUDA
  engines and on sequential with the `slack_csr` store check), the suite summaries and the
  compacted records of the performance gates of the seven suites (`ics26_dynlp` with AD4's
  release-only 50m reading, `ia3_hsosp`), and the golden mutations (`mutation-goldens.json`,
  measured at `f60a2f3`), all described below as measured for 0.4.0rc1. Only `VERSION`
  (0.4.0rc1, now 0.4.0), `CHANGELOG.md`, `CITATION.cff`, `README.md`, `python/README.md` (the
  PyPI project page), one Python test with its marker check (`python/tests/test_examples.py`,
  `python/tests/test_gpu_marker.py`: the `gpu` marker of the hyper_sssp example's `cuda` case)
  and documents changed (R041), none a library path.
- **Run again on the final tree** from fresh clones of the branch `release-0.4.0-final` at `fe7ecc2`
  (the `tests` checks) and at `91b966e` (the `packaging` and `repo` checks, after R041's review
  fixes: ADRs 0069, 0071, 0090, 0096, 0097 and 0109, the developer plan, the tutorials "Streaming
  label propagation" and "Hypergraph triads" and R041.md; and after step 4's two text corrections in
  `CHANGELOG.md` and R041.md; no path of the `library` or `tests` scope), concurrently under the
  shared lock and never during a timing run, and recorded with `certify.py check --version 0.4.0` in
  place of the release candidate's records of the same names (logs under
  `$DYNG_SCRATCH/runs/r041/cert/` for the `tests` checks and `$DYNG_SCRATCH/runs/r041/textfix/` for
  the `packaging` and `repo` checks; the scripts in `$DYNG_SCRATCH/r041/`). The `distributions`
  record names the clone it ran from, `91b966e`, the final-release pull request's commit, not a tag
  day's:
  - the `packaging` check `distributions`: the distributions as `release.yml` builds them for the
    tag `v0.4.0` (`select` reports a final release: `kind=package`, `version=0.4.0`,
    `prerelease=false`, `plugins=true`, the three distributions; `v0.4.0rc1`, `v0.4.0-rc.1`,
    `v0.4.1` and `v0.3.0` refused; the core sdist and wheel and the plugins `dyng-cu12` /
    `dyng-cu13` with CI's CUDA toolkits 12.9 and 13.5, still the newest 12.x and 13.x of NVIDIA's
    RHEL 8 repository on 2026-11-26 (`cuda-nvcc` 12.9.86 and 13.5.37), a fresh lock identical,
    the RPMs' signatures verified; the `collect` checks: `twine check --strict`,
    `ci/wheel_check.py` on each file and `--release-set --split`; fresh 3.12 / 3.13 venvs with the
    wheel's test subset (473-474 passed), the README quickstart, every Python snippet of the docs,
    the tutorials' tests, the label_propagation example's labels against the committed digests and
    the GPU tests of both plugins on GPU 1 (`pytest -m gpu`: 82 passed, 8 skipped in
    `ci/plugin_wheels.sh`, one more than 0.4.0rc1, step 1's marker fix); the pinned extras
    `dyng[cuN]==0.4.0` and the final release's documented install lines without a version, `pip
    install dyng`, `"dyng[cu13]"` and `"dyng[cu12]"`, against a local simple index of the three
    distributions with PyPI as the extra index (0.4.0 and the plugin of the same version taken;
    PyPI has 0.3.0); and the upgrade from 0.3.0: a venv with `dyng[cuN]==0.3.0` from PyPI keeps
    it for `pip install "dyng[cuN]"`, `pip install -U dyng` alone gives `dyng` 0.4.0 beside the
    0.3.0 plugin, which dynG does not use (a `BackendWarning`, `cuda (no)`: what the docs warn
    about), and `pip install -U "dyng[cuN]"`, the documented upgrade line, installs 0.4.0 and the
    plugin of the same version, whose smoke test passes on GPU 1; 175 steps, none failed);
  - the `repo` checks, at `91b966e`: `check-parity` (`ci/check.sh --parity`, 2026-11-26 19:55-22:07
    UTC: cpu-only 1,656 / 1,656, dev 1,701 / 1,701, the harness 557 passed, `ci/python.sh` 839
    passed, `ci/docs.sh`, the parity preset's golden replay 92 / 92), `api-check` (`ci/api_check.sh
    4016b469`, against `main` as pushed, the merge of 0.4.0rc1: no breaking change of the frozen C++
    API or of the Python API) and `gpu_local` (**one** `ci/gpu_local.sh --preset parity-cuda` run on
    GPU 1, 2026-11-26 22:07 to 2026-11-27 03:54 UTC: every one of its 26 steps passed at the first
    attempt, label_propagation with the in_place cloud under ADR 0111's rate rule; no sanitizer run
    retried, ADR 0112); the earlier runs of the three at `fe7ecc2` (`gpu_local` 02:59-08:53 UTC) and
    at `928ee6b` (`gpu_local` 11:20-17:19 UTC, after the review fixes) passed as well, every step at
    the first attempt, and their records were replaced;
  - the `tests` checks `asan` (1,486 / 1,486), `tsan` (1,274 / 1,274), `tsan-openmp` (1,486 /
    1,486; the conda Clang 18.1.8 through R020's local wrapper and the environment's Archer, as
    before) and `mutation-ctests` (`ctest --preset dev-cuda -L mutation`, GPU 1: 72 / 72),
    because `README.md`, whose C++ quickstart is a test, changed (its status block and install
    lines; the C++ quickstart itself did not).
- **Added:** `sanitizers-hosted`, the hosted `sanitizers.yml` jobs of the release candidate's pull
  request #27: run 45598377457 on its final head `3b8c651` (2026-11-25, 10:47-11:20 UTC; event
  `pull_request`, which tested the merge `7e04f23` of `3b8c651` into `d3375da`, an ancestor of the
  head, so the same tree): `asan / gcc-13` 1486 of 1486 tests, `tsan / gcc-13` 1274 of 1274,
  `tsan-openmp / clang-18` 1486 of 1486, all green. 0.4.0rc1's certificate had none (the run
  ended after it was written). Recorded with the scope `library`: its library is the release's,
  while `README.md` changed since (release.md step 3).

## The smoke test of 0.4.0rc1 from TestPyPI (checklist step 8)

R041 step 1 (`docs/developer/retrospectives/R041.md` has the tables), 2026-11-26, 00:51-01:02 UTC
by the machine's clock (about 3 minutes behind real time; R041's own readings 01:25-01:33), on the
GPU machine (NVIDIA driver 590.48.01, CUDA 13.1; the GPU tests on GPU 1), every venv new (CPython
3.12.14 and 3.13.15), every command in a clean environment (`env -i`, no dev environment on
`PATH`, no pip cache), the tests, examples and `ci/` scripts from the tag `v0.4.0rc1`:

- **The CPU package**, installed as the release candidate's README and install guide say
  (`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
  dyng==0.4.0rc1`): `import dyng`, `dyng.show_config()`, `sssp`, `cycle_count` and `mosp` compute
  and update, `dyng.Hypergraph` with `triad_count` compute and update, on `sequential` and
  `openmp` with equal results; new in 0.4, `label_propagation` on a `dyng.Graph.with_capacity`
  graph fed by `dyng.generators.vertex_arrival_stream` (compute and one update per batch, damped
  and in_place, sequential: labels in [0, 1], every batch converged, two damped runs bitwise
  equal; `NotSupportedError` on openmp) and `hyper_sssp` on a `slack_csr` `dyng.Hypergraph`
  (compute and update with an incidence edit, sequential; a second update equal to a fresh
  compute; `NotSupportedError` on openmp); `dyng --version` (`dyng 0.4.0rc1`), `dyng mosp
  compute` on both backends (byte-identical outputs); the examples of `examples/python` on their
  data byte for byte: `triad_count.py` (both backends) and `hyper_sssp.py` equal to their
  `.expected.txt`, and `label_propagation_torch.py` and `dyng label_propagation update` giving
  the committed sequential digest `81f174fe...`; the tag's pytest suite: 835 passed, 99 skipped,
  on both Python versions.
- **Both CUDA plugins**, each in its own fresh venvs, installed as the docs say (NumPy and the
  test tools from PyPI, then `--no-deps dyng==0.4.0rc1 dyng-cuN==0.4.0rc1` from TestPyPI alone):
  `ci/plugin_smoke.py --expect cuda` on GPU 1 (sssp, cycle_count, mosp, triad_count and
  hyper_sssp on cuda equal to the sequential backend, label_propagation within the oracle's
  bound; cu12 built with CUDA 12.9, cu13 with 13.5), the fallback without a device and without a
  driver (`ci/without_cuda_driver.sh`), `pytest -m gpu` with `DYNG_REQUIRE_CUDA=1` (81 passed, 8
  skipped: the CuPy / PyTorch round trips, not installed) and the suite with `DYNG_CPU_ONLY=1`
  (835 passed, 99 skipped): all passed for cu12 and cu13 on 3.12 and 3.13. `pip install -i
  https://test.pypi.org/simple/ "dyng[cuN]==0.4.0rc1"` resolved to the plugin of the same
  version, and its smoke test passed.
- **One test case `pytest -m gpu` did not select:** the hyper_sssp example's `cuda` case of
  `python/tests/test_examples.py` had no `gpu` marker. Run on the published wheels in the four
  plugin venvs it passed; its marker is fixed in this release with a check that every `cuda`
  case carries it (`184042a`, tests only), so the GPU tests of the final tag's wheels select 82
  cases.
- **The files tested are the published ones:** pip's installation reports name the same wheels in
  every venv, their SHA-256 equal to TestPyPI's digests and to the `dist` artifact of
  `release.yml` run 45728140156 (`dyng` `14fb1a1f...`, `dyng_cu12` `9962b288...`, `dyng_cu13`
  `8a523bdf...`, the sdist `596fb2d4...`); pypi.org had 0.3.0 as the latest release of all three
  projects. The step-10 block of release.md ran once as written on that artifact (the plugin
  wheels' GPU tests on GPU 1), and passed.

The combined log is `$DYNG_SCRATCH/runs/r041/smoke-rc1.log` (copied to `checks/smoke-rc1.txt`;
sha256 `f64f30c4c8baaf91...`), the scripts `$DYNG_SCRATCH/runs/r040-smoke/run_smoke.sh`,
`smoke_rc_cpu.py` and `$DYNG_SCRATCH/r041/step10_dryrun.sh`. No separate CPU-only machine was
available (as for 0.1.0 to 0.3.0).

## What was run, and how (measured for 0.4.0rc1)

This section is the release candidate's, as its certificate described it. Its golden replays,
performance gates and golden mutations (at `f60a2f3`) are the ones carried over here. Its
**check records** are not: the sanitizer presets and the mutation CTests at `2616af6`, and the
distributions of `v0.4.0rc1`, `ci/check.sh --parity`, `ci/api_check.sh d3375da` and
`ci/gpu_local.sh --preset parity-cuda` at `77e7646` (its tag day) were replaced in this
directory's `checks.json` and `checks/` by the R041 runs at `fe7ecc2` and `91b966e` listed in
"Carried over from 0.4.0rc1" above, and the hosted `sanitizers.yml` record was added; the tables at the end of
this page are the carried-over certificate's.

- **Code.** The release branch `release-0.4.0` (from `main` at `d3375da`, the merge of the ADR
  0120 fix, which contains M10b's merge `c67dd40`). Every measured build comes from a clean `git
  clone` of the branch (the commit each record names), with the presets `parity` and
  `parity-cuda`. The gates, the golden replays and the golden mutations ran at `f60a2f3`
  (2026-11-20 06:59 UTC to 2026-11-21; its library is the release's final one). After the review
  fixes of R040 step 4 (no path of the `library` scope changed), the other checks were taken
  again, each from a fresh clone: the sanitizer presets, the mutation CTests and the portability
  pre-checks at `2616af6`, the last commit of the `tests` scope; the distributions,
  `ci/check.sh --parity`, `ci/api_check.sh d3375da` (against `main` as pushed) and one run of
  `ci/gpu_local.sh --preset parity-cuda` at `77e7646` (the tag day; first at `91943a5`; corrected
  in R041, as 0.4.0rc1's `checks.json` names `77e7646`).
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
  the documented release-candidate forms; cibuildwheel's test command from the unpacked sdist,
  every test source in it (`ci/cibw_test.py`); the wheel's test subset (473-474 passed), the README
  quickstart, every Python snippet of the docs, the label_propagation and hyper_sssp tutorials
  and the label_propagation example's labels against their committed digests, `pytest -m gpu`
  on GPU 1, 80-81 passed; SHA-256 of the local builds, whose tag's files are CI's: sdist
  `c518bced...`, CPU wheel `d03eb2b1...`, cu12 `01a502b1...`, cu13 `c00210be...`);
  `ci/check.sh --parity` (cpu-only 1,656 / 1,656, dev 1,701 / 1,701, the harness 557 passed,
  `ci/python.sh` 837 passed, the parity preset's golden replay 92 / 92); `ci/api_check.sh d3375da` (the frozen C++ API and the Python API
  against `main` as pushed); and **one** run of `ci/gpu_local.sh --preset parity-cuda` with
  nothing skipped (every step of `GPU_LOCAL_STEPS` passed, ADR 0062: the CUDA tests and goldens,
  compute-sanitizer, the CBST store, the hypergraph and triad_count steps, count_local_patterns,
  label_propagation with the in_place cloud under ADR 0111's rate rule, ADR 0110, slack_csr,
  hyper_sssp, the plugin wheel, clang-tidy; its racecheck runs under ADR 0112's bounded retry).
  Every check names its logs with their SHA-256; an excerpt of each is in `checks/`.
- **Files.** `ia3_hsosp.json`, the suite summary, is written without indentation (the same
  document; the repository's limit of 256 KB on an added file, `.pre-commit-config.yaml`).

<!-- certify:begin (generated by parity/certify.py write; do not edit by hand) -->

**Certificate:** `parity.json` (dynG 0.4.0, commit `958d11dc98ba`): **all parts passed**.

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
| mutation-goldens | `parity/mutate.py run (every mutation of parity/mutate.py list: sssp, mosp, cycle_count, label_propagation, triad_reference, cbst, hypergraph, triad_count, hyper_sssp with the slack_csr container's, kit; parity-cuda copies, CUDA on GPU 1), from a clean clone` | library | passed |
| sanitizers-hosted | `the hosted sanitizers.yml jobs of pull request #27 (the release candidate's), run 45598377457 on its final head 3b8c651 (event pull_request, tested as its merge 7e04f23 into d3375da, an ancestor of the head, so the same tree; 2026-11-25 10:47-11:20 UTC): asan / gcc-13, tsan / gcc-13, tsan-openmp / clang-18; the job logs read with gh run view --job <id> --log (read-only). Scope library: README.md, a path of the tests scope, changed since 3b8c651 (the status texts only); the tests scope is covered by the local presets run again on the final tree (docs/developer/release.md step 3)` | library | passed |
| asan | `ci/sanitizers.sh asan (ctest -L cpu -LE slow of the asan preset), from a clean clone` | tests | passed (1486 / 1486 tests, 112 skipped) |
| tsan | `ci/sanitizers.sh tsan (ctest -L cpu -LE slow of the tsan preset), from a clean clone` | tests | passed (1274 / 1274 tests, 131 skipped) |
| tsan-openmp | `ci/sanitizers.sh tsan-openmp (Clang 18.1.8 of the conda env dyng-clang through a local wrapper: --gcc-install-dir GCC 12, --sysroot=/, -Wno-deprecated-declarations; Archer DYNG_ARCHER_LIBRARY = the env's libarcher.so.bak; ctest -L cpu -LE slow), from a clean clone` | tests | passed (1486 / 1486 tests, 112 skipped) |
| mutation-ctests | `ctest --preset dev-cuda -L mutation (GPU 1, after locklog.py quiet-ok R041), from a clean clone` | tests | passed (72 / 72 tests) |
| distributions | `release.yml's select step for GITHUB_REF_NAME=v0.4.0 (kind=package, version=0.4.0, prerelease=false, plugins=true; v0.4.0rc1, v0.4.0-rc.1, v0.4.1 and v0.3.0 refused); ci/wheel_check.py --release-metadata (and --release-date 2026-11-28, the planned tag day); ci/wheels.sh (with cibuildwheel's test step from the unpacked sdist's test sources, ci/cibw_test.py); ci/plugin_wheels.sh with DYNG_PLUGINS='cu12 cu13' and the CI toolkits of ci/cuda_toolkits.toml (CUDA 12.9 and 13.5, the newest 12.x and 13.x of NVIDIA's RHEL 8 repository on 2026-11-20, 2026-11-21, 2026-11-25 and again in R041 on the day of this run, a fresh lock identical, ADR 0113; unpacked by ci/cuda_toolkit.py extract; verify --signatures passed); twine check --strict; ci/wheel_check.py on each file and --release-set --split; fresh Python 3.12 and 3.13 venvs: the core wheel alone, core + dyng-cu12, core + dyng-cu13, dyng[cu13]==0.4.0 and dyng[cu12]==0.4.0 from a local simple index with the three distributions and PyPI as the extra index (what PyPI serves after the release; PyPI has 0.3.0), the final release's documented install lines without a version (pip install dyng, dyng[cu13], dyng[cu12]: 0.4.0 and the plugin of the same version taken), and the upgrade from 0.3.0 (dyng[cuN]==0.3.0 from PyPI kept without -U; pip install -U dyng alone leaves the 0.3.0 plugin; the documented pip install -U dyng[cuN] installs 0.4.0 with its plugin, whose smoke test passes on GPU 1): the wheel's test subset (cibuildwheel's test-command), the README quickstart, every Python snippet of the docs (label_propagation's and hyper_sssp's quickstarts among them), the tutorials 'Streaming label propagation' and 'Shortest hyperpaths' (their tests) and the label_propagation example against its committed digests, dyng --version, ci/plugin_smoke.py --expect cuda and pytest -m gpu on GPU 1; from a clean clone of the final-release pull request's commit 91b966e (R041 step 4, the final tree), the whole run under the shared lock` | packaging | passed |
| check-parity | `ci/check.sh --parity, from a clean clone (origin/main = 4016b469, main as pushed: the merge of pull request #27, 0.4.0rc1)` | repo | passed |
| api-check | `ci/api_check.sh 4016b469 (the frozen C++ API with ci/api_snapshot.py --against and the Python API with griffe, against main as pushed, the merge of pull request #27), from a clean clone` | repo | passed |
| gpu_local | `ci/gpu_local.sh --preset parity-cuda (GPU 1, every step, one run; DYNG_LOCKLOG_MILESTONE=R041), from a clean clone` | repo | passed (build passed, ctest -L gpu passed, ctest -L cpu passed, parity (cuda goldens) passed, parity (cycle_count cuda goldens) passed, memcheck passed, synccheck passed, synccheck (mosp) passed, synccheck (cycle_count) passed, racecheck passed, cbst sanitizers (racecheck, synccheck, initcheck) passed, cbst replays (device, small sets) passed, cbst replays (device, default preset) passed, cbst bin/main recorded and replayed once passed, cbst lockstep with the host model (compact after every op; every preset) passed, hypergraph sanitizers (racecheck, synccheck, initcheck) passed, hypergraph replays (cuda, small sets; ids, rows, line graph, legacy generator) passed, triad_count sanitizers (synccheck, initcheck; a coauth batch; racecheck) passed, count_local_patterns (sanitizers on the tutorial and API tests; nvcc consumer) passed, label_propagation (CUDA tests, kit, replays of P, in_place cloud, oracle; sanitizers) passed, slack_csr sanitizers (all four tools; a forced-overflow session) passed, slack_csr replays (cuda, make test; ids, pairs, counters, device layout) passed, hyper_sssp sanitizers (all four tools; the engine's tests, forced fallbacks) passed, hyper_sssp replays (cuda, make test; three modes, both engines, update == compute) passed, plugin wheel (pytest -m gpu) passed, clang-tidy passed) |

<!-- certify:end -->
