# Changelog

All notable changes to dynG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 0.1.0.
Before 0.1.0 anything may change.

## [Unreleased]

## [0.2.0rc1] - 2026-10-07

### Summary

The release candidate of dynG 0.2.0: a third stable algorithm, the CUDA backend from Python, and
the hardening planned after 0.1.0. Release candidates are published on TestPyPI only:

```bash
pip install "numpy>=1.26"
pip install -i https://test.pypi.org/simple/ --no-deps "dyng==0.2.0rc1" "dyng-cu13==0.2.0rc1"
```

(`dyng-cu12==0.2.0rc1` for an NVIDIA driver of CUDA 12; `pip install -i
https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==0.2.0rc1` for the
CPU package alone; the plugins come from TestPyPI alone, because their names are not on PyPI
before 0.2.0); 0.2.0 follows on PyPI, with `pip install "dyng[cu13]"`. Nothing of 0.1.0 changes incompatibly: `sssp` and `cycle_count`
keep their API. The entries below the summary are the detailed record of milestones M7, M6a and
M6b and of the release preparation R020.

- **`mosp` is stable** (SemVer applies from 0.2.0): dynamic multi-objective shortest paths, the
  MOSP update of DynaMOSP (the K single-objective trees, the combined graph of the preferences,
  its shortest-path tree and the path costs), on the sequential, OpenMP and CUDA backends, in C++
  (`<dyng/mosp.hpp>`), Python (`dyng.mosp`) and on the command line (`dyng mosp`, the original
  tools' files byte for byte). It is byte-identical to MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2
  on their golden corpus and on 20 paper-scale cases, and its API was reviewed before the freeze
  (ADR 0035); `dyng.update()` keeps mosp, sssp and cycle_count results on one graph.
- **The `sssp` operators engine:** CUDA `sssp` (and so `mosp`) also runs on GPUs and drivers
  without cooperative launch, byte-identical to the fused engine; `cuda_engine` chooses
  (`automatic`, `fused`, `operators`).
- **CUDA from Python:** the plugin wheels `dyng-cu13` and `dyng-cu12`, installed with
  `pip install "dyng[cu13]"` (NVIDIA driver 580 or newer, CUDA 13) or `"dyng[cu12]"` (driver 525
  or newer, CUDA 12). They bring their own CUDA runtime (no toolkit needed) and code for sm_75 to
  sm_120 (Turing to Blackwell). `import dyng` picks the plugin of the driver's CUDA major and falls
  back to the CPU backends with one `dyng.BackendWarning` that says why; results stay in device
  memory and go to PyTorch or CuPy through DLPack or `__cuda_array_interface__` without a copy;
  `dyng.show_config()` says what was chosen.
- **The tutorial "Your first dynamic algorithm"** and the teaching algorithms `dynamic_bfs` and
  `triangle_delta` (maturity `tutorial`, not under the stability promise): from a fresh clone to a
  new algorithm that passes the conformance kit on every backend.
- **The documentation site** <https://dyng-dev.github.io/dyng/>, built from `main`.
- **Robustness:** libFuzzer targets for every file reader (the reader bugs they found are fixed),
  the recorded bug mutations of sssp, cycle_count and mosp checked against their golden suites,
  long Hypothesis runs, and ASan + UBSan and TSan (with OpenMP and Archer) on every pull request,
  now required checks.
- **The release certificate** `benchmarks/results/0.2.0rc1/` (PLAN 8.3 and 8.6): golden parity of
  sssp, cycle_count and mosp on every backend, the release performance gates of all three against
  the unpatched originals (mosp's for the first time, the new suite
  `benchmarks/paper/ipdps25_dynamosp_mosp.yaml`; CUDA at locked clocks, ADRs 0018 and 0021), the
  sanitizers and the mutation checks.

Known limitations of 0.2.0: Linux x86-64 only (aarch64 wheels later); the plugins need a GPU of
compute capability 7.5 or newer; `cycle_count` counts simple cycles only (no time-window or
temporal modes, no approximate TruCy mode); the hypergraph and `triad_count` come with 0.3.0,
`label_propagation` and `hyper_sssp` with 0.4.0; no DOI (Zenodo is not connected).

### R020: the release changes (0.2, branch `release-0.2.0`)

- Changed: `VERSION` 0.2.0rc1 (it was 0.2.0.dev0, the development version after 0.1.0);
  `CITATION.cff` `version: 0.2.0rc1`, `date-released: 2026-10-07`; this section `[0.2.0rc1]`
  with the summary above: everything that was under `Unreleased` (M7, M6a, M6b and R020) moved
  here, an empty `Unreleased` above it, and the link references.
- Docs: the status texts for the release candidate: the status blocks of `README.md` and the
  documentation's start page ("Alpha: 0.2 release candidate"), the install lines of the release
  candidate from TestPyPI (`"dyng[cu13]==0.2.0rc1"`, `"dyng[cu12]==0.2.0rc1"`, `dyng==0.2.0rc1`)
  with the plugins' driver and GPU requirements (README, `docs/getting_started`), `SECURITY.md`
  (0.2.0rc1 supported until 0.2.0), `SUPPORT.md`, `CONTRIBUTING.md`, the roadmap and the
  developer plan. The documentation site's banner names a release candidate as such (TestPyPI
  only; `docs/conf.py`).
- Governance: the author's decisions for 0.2.0 in the approvals log of `GOVERNANCE.md`: the AI
  assistant pushes the signed tags `v0.2.0rc1` / `v0.2.0` and creates the GitHub Release on the
  author's behalf (2026-10-06); the author merges the release pull requests and approves the
  three PyPI deployments, `pypi-cu12` and `pypi-cu13` together, then `pypi` (2026-10-07); the
  sanitizer checks `asan / gcc-13`, `tsan / gcc-13` and `tsan-openmp / clang-18` are required
  checks of `main` (2026-10-07; 27 required checks); and the repository state of 2026-10-07 (M6b
  merged, the first Pages deployment, the website field). `docs/developer/release.md` says who
  does what for 0.2.0; `docs/developer/repository_settings.md` lists the 27 required checks.
- CI: the pinned CUDA toolkits of the plugin wheels were checked against NVIDIA's RHEL 8
  repository on 2026-10-07: CUDA 12.9 (nvcc 12.9.86) and 13.4 (nvcc 13.4.92) are still the
  newest 12.x and 13.x, and a fresh `ci/cuda_toolkit.py lock` of both reproduces
  `ci/cuda_toolkits.toml` byte for byte (no re-pin; the cached RPMs pass `verify --signatures`).
- Docs: two informational investigations in `docs/developer/retrospectives/R020.md` (record
  `parity/results/R020-investigations.json`; no gate applies, nothing in the library changed):
  (a) the CUDA 13.4 cu13 wheel's cycle_count update is not slower than the CUDA 13.1 builds
  (0.965-0.986x over 8 processes per side; the update kernels within 1 % under Nsight Compute):
  M6a's 8.8 % was the process mode of the host stage `cycle_count.normalize`, so the wheel keeps
  CUDA 13.4; (b) a measured proposal for the open decision on layout control (function, loop and
  branch alignment against the originals and dynG against dynG), attached to the decision in
  `GOVERNANCE.md`, which stays open; the certificate uses the accepted one-layout protocol.

### R020: the mosp API review and freeze (0.2, branch `release-0.2.0`)

- Changed: **`mosp` is stable** from 0.2.0 (the author's decision of 2026-10-07, GOVERNANCE.md):
  `<dyng/mosp.hpp>`, the Python module `dyng.mosp` and the command line `dyng mosp` follow the
  stable tier of the stability policy, as `sssp` and `cycle_count` since 0.1.0. The API review
  before the freeze (ADR 0035) changed no signature, so existing code needs no change. The
  manifest, the registry (`dyng::algorithms()`, `dyng.algorithms()`), the algorithm tables and
  `docs/algorithms/mosp.md` say `stable`; `dynamic_bfs` and `triangle_delta` stay `tutorial`.
- Docs: the review's amendments of `<dyng/mosp.hpp>`: what the inherited `update_stats` counters
  mean for mosp and which are deterministic; the exceptions `result::from_arrays()` inherits from
  sssp's import; `clone()` sizes the pooled workspace; `path_costs()` stays host memory by
  default; `max_objectives` and `max_preference_scale` may be raised, never lowered;
  `set_options()` allocates nothing after its checks (its strong guarantee, pinned by the new
  `dyng_mosp_allocation_failure_tests`).
- CI: `ci/api_snapshot.py` marks an algorithm's header *frozen* exactly when its manifest says
  maturity `stable` (sssp, cycle_count, mosp) and *tracked* otherwise (the tutorial algorithms);
  the baseline's header comment says so. New tests: the label follows the manifest, a changed
  mosp signature fails the C++ check, and `ci/tests/test_api_check.py` shows that griffe (the
  Python check of `ci/api_check.sh`) reports changes of `dyng.mosp` signatures, defaults and names
  as breaking.

### R020: the mosp benchmark suite (0.2, branch `release-0.2.0`)

- Added: the benchmark-suite record **`benchmarks/paper/ipdps25_dynamosp_mosp.yaml`**: the MOSP
  update of DynaMOSP (IPDPS 2025) on roadNet-PA, roadNet-CA, rgg_n_2_20_s0 and road_usa (K = 3,
  the 50K safe, 50K unsafe and 10K local batches of seed 777, the sssp suite's inputs and
  digests), "(a) compute" and "(b) end to end" against the unpatched `bin/mosp` of
  MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2 (CUDA at the boost clock lock, the default-clock
  reading recorded, not gated), the device memory against MOSP-CUDA's; the gates M7 took.
- Parity tools: `parity/bench_suite.py` runs a mosp suite through `parity/perf_ab.py mosp` and
  `perf_ab.py memory --mosp`, and refuses a record that is not of mosp or not of the suite's K,
  batches, preferences, CUDA engine and output scope, or whose `invalidated` counts differ
  between the two sides; `--batches` narrows an sssp or mosp execution (a partial one), and
  `--keep-contaminated` makes a diagnostic one for a busy machine (rounds with foreign load kept
  and flagged, never a gate reading: a record that kept them is refused otherwise).
- Parity tools: `parity/certify.py write` requires the suites of each release series
  (`REQUIRED_SUITES`: the sssp and cycle_count suites from 0.1, the mosp suite from 0.2: a 0.2
  certificate without the mosp gates fails), and a gated reading of every gated metric of each
  suite on every backend it is gated on.

### R020: the release certificate (0.2, branch `release-0.2.0`)

- Added: the release certificate of 0.2.0rc1, `benchmarks/results/0.2.0rc1/` (`parity.json` and
  its `README.md`): the golden replays of sssp, cycle_count and mosp (the paper-scale
  `mosp_scale` set included) on every backend, the three performance suites against the
  unpatched originals, the sanitizer presets and compute-sanitizer, the mutation checks, the
  distributions as `release.yml` builds them for `v0.2.0rc1`, and `ci/check.sh --parity` and
  `ci/gpu_local.sh` on the release tree; the summary is in
  `docs/developer/retrospectives/R020.md`.
- Parity tools: the generated-metadata exception of `parity/certify.py` (`METADATA_PATHS`) covers
  the manifest of every algorithm (mosp and the tutorial algorithms too, not only sssp and
  cycle_count); no build reads a manifest.

### R020: review fixes (0.2, branch `release-0.2.0`)

- Changed (documentation of the frozen mosp API; no signature changed): `options::
  compute_path_costs` takes effect at the next `compute()` or `update()` (`set_options()` changes
  it for the next `update()`; `path_costs()` follows the option of the last one), as
  `path_costs()` documented; it is no longer called a tunable. `mosp::update()` documents the
  backend-mismatch and copy-policy errors it shares with `sssp::update()`. The Python module and
  `Stats` docstrings state the determinism exception for trees imported without
  canonicalization and the meaning of `converged`, `fallback_used` and `engine_used` (ADR 0035,
  amendment A1).
- Changed: the API check enforces the C++ freeze: `ci/api_check.sh` (`api-check.yml`) compares
  the frozen sections of the C++ API baseline with the base branch (`ci/api_snapshot.py
  --against`) and fails on a breaking change without the `api-change` label and a CHANGELOG
  entry, as it does for Python; the freeze tests of `ci/tests/test_api_check.py` run in CI
  (griffe installed, `DYNG_REQUIRE_GRIFFE=1`) (ADR 0035, amendments A2 and A3).
- Changed: `parity/certify.py` requires every release check of the version's series
  (`REQUIRED_CHECKS`: the sanitizer presets, the mutation CTests and golden mutations, the
  distributions, `check-parity`, `api-check`, `gpu_local` with its compute-sanitizer steps), reads
  the golden-mutation record against `parity/mutate.py list`, recomputes every gate row from the
  records and the suite's tolerance instead of copying its flags, names each measured commit once
  as a full SHA, and its `packaging` and `tests` scopes cover the CUDA plugins' build inputs, every
  file the sdist ships and the fuzzers' regressions.
- Fixed: `find_package(dyng 0.2 REQUIRED)` in README.md and the install guide (0.1 refuses an
  installed 0.2, `SameMinorVersion` before 1.0); `ci/cmake_consumer.sh` (`ci/check.sh`, `cpu.yml`)
  builds a consumer with the documented block against an installed build.
- Changed: the release candidate's plugin install takes `dyng` and the plugin from TestPyPI alone
  (`--no-deps`, NumPy from PyPI first), because the names `dyng-cu12` / `dyng-cu13` are not on
  PyPI before 0.2.0.
- Fixed: `dyng.show_config()` no longer suggests `pip install "dyng[cu13]"` when a plugin is
  installed but unusable, and the selection reason says whether a plugin of the driver's CUDA
  major is installed; the `dyng` command prints a `BackendWarning` as `dyng: warning: ...` instead
  of naming the console-script wrapper's line.
- Changed: the PyPI description (`python/README.md`) names mosp and runs it; the wheel's test
  subset runs `test_mosp.py`; the getting-started next steps name mosp and the tutorial
  algorithms. The benchmark suites and `benchmarks/README.md` say that rgg's unsafe50k batch is
  byte-identical to its safe50k batch (no deletion disconnects a vertex there).
- Changed: `docs/developer/release.md` for 0.2 (every packaging- and repo-scope check is re-run on
  a later tag day, `api-check` included; the record of the hosted sanitizer jobs; `wheels.yml`
  green before the merge; mosp in the smoke test; the final-release checklist in version-neutral
  terms); GOVERNANCE.md: the layout-control decision is due before `v0.2.0`, and the PyPI names of
  the plugins are a new open decision.
- The release certificate was measured again on the reviewed code (R020 retrospective, step 5).

### Added (M6b)

- M6b: the **tutorial algorithms** (teaching material, maturity `tutorial`; ADR 0033), created with
  `scripts/new_algorithm.py` and completed on the sequential, OpenMP and CUDA backends (Tier A),
  each with hand cases, an independent oracle and the conformance kit C0-C12 on every backend:
  `dynamic_bfs` (fixed point: BFS levels from a source; the subtrees under deleted BFS-tree edges
  are invalidated and re-seeded, a frontier propagates the improvements) and `triangle_delta`
  (aggregate delta: the triangle count of an undirected graph; count(-) on the old graph,
  count(+) on the new one, `ownership::min_member`). Python: `dyng.dynamic_bfs`,
  `dyng.triangle_delta` (not in `dyng.update()`, no CLI command).
- M6b: the tutorial **"Your first dynamic algorithm"** (`docs/tutorials`): from a fresh clone to
  a green conformance kit with `scripts/new_algorithm.py` and the fixed-point template. Its
  reference solution is `examples/tutorial_algorithms/my_bfs/`, which `ci/scaffold_check.sh`
  builds and tests.
- M6b: `maturity_level::tutorial` (appended; `"tutorial"` in the manifests, the registry and the
  generated tables).
- M6b: the framework operators' executors (`cpp/src/operators/`): `sequential_exec`,
  `openmp_exec` and `cuda_exec` run a `DYNG_HD` functor per element on their backend, with the
  atomics of such functors, so one source serves the three backends (the two tutorial algorithms
  use them; PLAN 4.5.3's rule of two).
- M6b: the conformance kit generates **undirected** graphs for an algorithm that requires them
  (`graph_properties::directed = false`); its edge and batch counters count each stored direction.
- M6b: **reader fuzzers** (`cpp/fuzz`, ADR 0034): libFuzzer targets for every file reader (Matrix
  Market, edge lists, the MOSP CSR triplet, MOSP batches, `.dgt` batches, distance and parent
  files) with ASan and UBSan, checking the documented exceptions, the consistency of accepted
  input and write/read round trips; the CMake option `DYNG_BUILD_FUZZERS` and the preset `fuzz`
  (Clang), `ci/fuzz.sh`, a seed corpus, and `.github/workflows/fuzz.yml` (pull requests that touch
  a reader: 60 s per target; weekly: 10 minutes). Every test build replays the seed corpus and the
  reproducers of fixed findings (`ctest -L fuzz`).
- M6b: `parity/mutate.py` covers the **cycle_count and mosp golden suites** (CycleEnumeration-GPU's
  double-counted 5-cycles and weakened ownership rule, host and device; a combined-graph edge
  weight from one tree only and path costs from one objective's weights, on each backend), next
  to sssp's; `--suites` selects them.
- M6b: the Hypothesis profile **`full`** (10,000 examples per property, not derandomized;
  `DYNG_HYPOTHESIS_EXAMPLES`) and `.github/workflows/property.yml` (weekly, on demand); property
  tests of `dynamic_bfs` and `triangle_delta` over chains of batches.
- M6b: the documentation is published on **GitHub Pages**, <https://dyng-dev.github.io/dyng/>:
  `docs.yml` deploys the site of every push to `main` (`actions/configure-pages`,
  `upload-pages-artifact`, `deploy-pages`; environment `github-pages`); README, `CITATION.cff`
  (`url`) and the Sphinx canonical URL name it. `docs/developer/robustness.md` describes the
  fuzzers, the mutation checks and the long property runs.
- M6b: **sanitizer jobs** on every pull request (`.github/workflows/sanitizers.yml`, not a required
  check yet): the CPU tests under ASan + UBSan (`asan`, GCC 13), under TSan with OpenMP off
  (`tsan`, GCC 13) and under TSan with the OpenMP backends (the new preset **`tsan-openmp`**:
  Clang and libomp, whose OMPT tool Archer tells TSan about OpenMP's synchronization). The script
  `ci/sanitizers.sh` runs the same presets locally.
- M6b: the author's re-grouping of the releases after 0.1.0 (2026-10-02) is recorded in
  `GOVERNANCE.md`, `docs/roadmap.md` and `docs/developer/plan.md`: 0.2.0 = `mosp` + the hardening
  planned as 0.1.x (CUDA plugin wheels, tutorials, GitHub Pages, fuzzers, mutation checks,
  sanitizers); 0.3.0 = the hypergraph and `triad_count`; 0.4.0 = `label_propagation` and
  `hyper_sssp`. The planned versions in the documentation, the docstrings and the messages of
  features that are not there yet follow it.

### Changed (M6b)

- M6b: configuring `DYNG_SANITIZE=thread` with `DYNG_ENABLE_OPENMP=ON` and a compiler other than
  Clang is an error (GCC's libgomp is not instrumented, so every parallel region would be reported
  as a race); use the preset `tsan` (OpenMP off) or `tsan-openmp` (Clang with Archer).
- M6b review: configuring with a `DYNG_ALGORITHMS` name that no algorithm has is an error (a stale
  name such as the tutorial's `my_bfs` after `new_algorithm.py --remove` used to skip every other
  algorithm quietly); `scripts/new_algorithm.py` takes the default of `--since` from `VERSION`;
  `scripts/regen.py --check` rejects the scaffold line in a manifest whose maturity is not
  `experimental`.
- M6b review: the conformance kit also runs every check with `deletions_first = false` (a third
  property preset) and the batch mix `cancel` (edges inserted and deleted, or deleted and
  re-inserted, in one batch) in C2, C3, C4, C9 and C10.
- M6b review: the reader fuzzers treat any exception in the write -> read round trip as a finding
  (a rejected copy used to count as a rejected input); `fuzz.selftest.<target>` plants a writer
  bug and checks that the replay reports it; the corpus replay of the libFuzzer build runs with a
  256 MB allocation limit and the third finding's reproducer; `fuzz.yml` sets
  `vm.mmap_rnd_bits=28`, accepts at most 1000 s per target on demand and uploads the reproducers
  of a cancelled run.
- M6b review: the documentation site (built from `main`) shows a banner naming the development
  version and the latest release; the README and the landing page name 0.1.0 as the latest
  release; the algorithm pages name releases instead of milestones; M6b's ADRs are 0033 and 0034
  (M6a has 0030-0032); ADR 0029 is accepted (the author, option A).

### Fixed (M6b)

- M6b, found by the reader fuzzers (and the review their memory use prompted):
  `io::read_matrix_market()` reserved memory for the number of entries the size line announced
  (up to 3 GB for a 74-byte file), and `io::read_csr_triplet()` for the edge count of
  `RowPtr.txt` (4 GB for a 13-byte `RowPtr.txt`) and for edges x weights of `Values.txt` before
  counting its lines. The reservations are now bounded by what the file can
  hold; the errors are unchanged (`cpp/tests/io/fuzz_regression_test.cpp`).
- M6b, found by `python/tests/test_reader_robustness.py`: the command line's text batches
  (`--batch`) raised `OverflowError` for an integer beyond 64 bits and `UnicodeDecodeError` for a
  file that is not UTF-8; both are now `dyng.FileFormatError` with the path and line.
- M6b: the cycle_count kernel tests did not build with Clang's ASan or TSan (their counting
  `operator new` collided with the sanitizer runtime's); they now detect Clang's sanitizers too.
- M6b review: `dynamic_bfs` (every backend) and the tutorial's `my_bfs` returned levels that were
  too low when `batch_semantics::deletions_first = false` and a batch inserted an edge and deleted
  it again: the seed offered a level along the requested insertion, which is not in the graph. It
  now offers only along an edge of G_{t+1}. `applied_batch::delta` is documented as what it is,
  the requested changes, not the net change.
- M6b review: `dynamic_bfs` and `triangle_delta` kept the scaffold line in their manifests, so
  `new_algorithm.py <name> --remove` deleted them without `--force`.
- M6b review: the tutorial "Your first dynamic algorithm": step 4.2 now replaces the scaffold's
  `reads_prepared_graph()` (adding a second one did not compile), the clean-up resets
  `DYNG_ALGORITHMS`, the prerequisites follow the install page (no compiler in the conda
  environment), and the reference solution's comments describe the incremental algorithm.

### M7: the sssp operators engine (0.2, branch `m7-mosp`)

- Added: the CUDA **operators engine** of `sssp` (decision O24, ADR 0026;
  `cpp/src/algorithms/sssp/operators.{cuh,cu}`): MOSP_ESCHER@4b86159's multi-kernel host loop with
  MOSP-CUDA@e220ee2's semantics, one framework hook per phase (Tier A), byte-identical to the fused
  engine on every input (the same trees, `invalidated` and `affected`; conformance check C4). It
  needs no cooperative launch and synchronizes 3 + iterations + epochs times per update.
- Changed (behaviour, no signature change): on CUDA, `sssp::options::cuda_engine =
  engine::automatic` runs the operators engine on a device without cooperative launch (it threw
  `not_supported_error`), and `engine::operators` runs the operators engine (it threw everywhere);
  `engine::fused` still throws `not_supported_error` without cooperative launch, and its message
  now names `engine::operators`. `stats::engine_used` reports `operators` for it. ADR 0017 item 2
  is superseded by ADR 0026.
- Changed: the device building blocks the two CUDA engines share (packed words, stamp claims, the
  warp-aggregated append, the warp reductions) moved unchanged from `fused.cuh` to `kernels.cuh`;
  the fused kernel's SASS is identical for all three instantiations.
- Changed: the conformance kit's C4 compares the two engines after `compute()` and after three
  consecutive batches for every preset, the small and medium sizes and every batch mix (it
  compared one batch); it runs for `sssp` on CUDA.
- Tests: the CUDA `sssp` suites run both engines: the MOSP fixtures, the packing boundary
  (n = 2^17 - 1), the distance-only fallback (320 x 320 grid, weights 2 * 10^9), the input checks
  (an out-of-range distance, a parent cycle), the steady-state allocations, the budget with strict
  budgets, and the randomized chains, where the CUDA executable compares the operators engine with
  the fused engine and the host backends on every chain; a larger engine-equality test (30K-40K
  vertices, many near-far rounds).
- Parity tools: `dyng-compat-mosp --cuda-engine automatic|fused|operators` (also in `init`; the
  per-objective report line names the engine), `parity/compare.py --configs
  cuda-fused[:d],cuda-operators[:d]`, and `parity/perf_ab.py engines` (the operators engine
  against the fused one, A/B/A/B at locked clocks; reported, not gated).

### M7: `mosp` (0.2, branch `m7-mosp`)

- Added: **`dyng::mosp`** (`<dyng/mosp.hpp>`, ADR 0027), dynamic multi-objective shortest paths
  (DynaMOSP; MOSP-CUDA@e220ee2 `mospUpdate.cu` / `combinedGraphGpu.cu`, MOSP-OpenMP@c352151
  `mospUpdate.cpp` / `combinedGraphCpu.cpp`) on the sequential, OpenMP and CUDA backends:
  `compute(res, g, source, options)`, `update(res, g, batch, result)`, `options{preferences (lcm
  <= 2^20), delta, cuda_engine, compute_path_costs, validate_inputs, num_objectives}`, `result`
  with the K sssp trees, the combined distances (units of 1/L), the MOSP tree, the path costs
  (host memory) and `from_arrays()` over K trees, `stats` with the K sssp stats,
  `combined_edges`, `preference_scale` and `affected` (vertices whose combined distance or MOSP
  parent changed). mosp is a composition: the batch is applied once, the K objectives are sssp
  problems updated one after the other on one workspace, then the combined graph (weights
  `L (K + 1) - sum L / Pref_i`; count, scan, fill) is solved with sssp's engine through the static
  enactor, and the path costs are recomputed; `dyng::update()` composes it with other results. The
  conformance kit passes C1-C12 on every backend (K = 3, preferences {2, 1, 3}).
- Added: `io::write_path_costs()` (MOSP's `mospCosts.txt`), `<dyng/testing/mosp_oracle.hpp>`
  (`testing::combined_graph_reference()`, `testing::mosp_path_costs_reference()`), and
  `<dyng/mosp.hpp>` in the umbrella header.
- Changed (behaviour, no signature change): `sssp::update()` accepts a batch without insertions
  whatever its `num_weights` (as `graph::apply()`); before, an empty or deletion-only batch built
  with the default one weight column was rejected on a graph with several weight columns.
- Changed (tests): the conformance kit's graphs can have several weight columns
  (`test_traits::num_weights`), C7 checks a batch with one column more than the graph, and C10
  builds its graph with the larger column count of the pair.
- Parity tools: `dyng-compat-mosp --mosp [--pref p1,..,pK] [--no-path-costs]`, the whole `mosp`
  driver (`combinedGraph/{distancesCsr,SSSPTreeCsr,mospCosts}.txt`, the `comb` report line,
  `RESULT compute_ms=`); the default mode (SOSP only, which the sssp gates measure) is unchanged.
  `parity/compare.py` replays the corpus with `--mosp` and compares `combined/` too;
  `parity/fixtures/mosp/make_mosp_fixtures.sh` writes the preference cases of
  `cpp/tests/data/mosp_combined` (the thesis example with Pref {4, 1, 4} and {4, 4, 1}, K = 2..4,
  `-k` below the graph's columns) after MOSP-OpenMP and MOSP-CUDA agreed on them.

### M7: parity and gates of `mosp` (0.2, branch `m7-mosp`)

- Changed (performance, same values): on the openmp and cuda backends `mosp`'s path costs run on
  the host threads of the resources handle (a level-synchronous traversal of the MOSP tree; the
  sequential port of `mospPathCosts` stays on the sequential backend and reports a missing tree
  edge for the other two). On cuda the download of the MOSP tree for the path costs is timed in
  `mosp.path_costs`, not `mosp.finalize`, and is one more synchronization of an update with
  `compute_path_costs` (ADR 0027, amendment).
- Parity tools: `parity/perf_ab.py mosp` (the MOSP update against MOSP-OpenMP@c352151's and
  MOSP-CUDA@e220ee2's `mosp`, both writing every output file as the originals' `bench/run.sh`;
  regions in the new `parity/timed_regions/mosp.toml`: "(a) compute" and "(b) end to end" gated,
  the per-objective updates, the combined step and the path costs reported; `-k`, `--pref`,
  `--cuda-engine`, `--no-output`), `parity/perf_ab.py prepare --widen BASE:K` (the K sweep's
  input: `mospPrep widen`, its trees and its 50K safe batch), and the paper-scale goldens
  `parity/export_goldens.py mosp_scale` / `parity/compare.py mosp_scale`
  (`parity/mosp_scale_goldens.py`: the SHA-256 of every output file of MOSP-OpenMP's `mosp` on
  20 benchmark cases, cross-checked with MOSP-CUDA, in `[sets.mosp_scale]` of
  `parity/goldens.toml`). `dyng-compat-mosp --mosp --timing` writes the setup's stages as `setup`
  rows.
- Changed (device memory, same values): on cuda the combined solve of `mosp` overwrites the
  previous MOSP tree in place and counts `affected` in its unpack pass (the fused and the
  operators engine count the changed vertices of a static solve on request, as an update's
  `affected`), so a cuda result no longer keeps a second pair of combined arrays (12 bytes per
  vertex): mosp's peak device memory went from 1.068-1.075x to 1.015-1.019x of MOSP-CUDA's on
  the road graphs (PLAN 8.6: <= 1.05x); one synchronization less per update (K + 3 with the path
  costs and the fused engine; ADR 0027, amendment).
- Parity tools: `parity/perf_ab.py memory --mosp` (the device memory of the whole MOSP update);
  `perf_ab.py` waits for the perf lock in the kernel instead of polling it once a second;
  `perf_ab.py mosp` leaves the path-cost region out when the port computes none (`-k` below the
  graph's columns, `--no-output`).
- Records: `parity/results/M7.md` sections 4-10 (the paper-scale parity, the K sweep, the mosp
  gates on both backends, the combined step, the device memory, the sssp gate re-check).

### M7: `mosp` in Python and on the command line (0.2, branch `m7-mosp`)

- Added: **`dyng.mosp`** (ADR 0028): `compute(graph, source, *, options=None, resources=None,
  **kwargs)`, `update(graph, batch, result, *, resources=None)`, `Options` (`preferences`,
  `delta`, `cuda_engine`, `compute_path_costs`, `validate_inputs`, `num_objectives`), `Stats`
  (with the K `dyng.sssp.Stats` in `objectives`, `combined_edges`, `preference_scale`) and
  `Result` (`distances(k)`, `parents(k)`, `combined_distances`, `combined_parents`,
  `path_costs` as a zero-copy (n, K) int64 array, `preference_scale`, `from_arrays()` over lists
  of K arrays, `clone()`), and `MAX_OBJECTIVES` / `MAX_PREFERENCE_SCALE`. `dyng.update()`
  accepts mosp results next to sssp and cycle_count results.
- Added: `dyng.io.write_path_costs()` (MOSP's `mospCosts.txt`), `dyng.testing.combined_graph()`
  and `dyng.testing.mosp_path_costs()` (the C++ references).
- Added: the command line **`dyng mosp compute|update`**, MOSP's files in and out byte for byte
  as the `mosp` driver writes them (`obj<k>/`, `combinedGraph/{distancesCsr,SSSPTreeCsr,
  mospCosts}.txt`); `--preferences 4,1,4` is the driver's `--pref`, `--num-objectives` its `-k`.
  List-valued option fields become comma-separated flags.
- Added: `examples/cpp/mosp_update.cpp` and `examples/python/mosp_update.py` (the thesis
  example's combined files, compared with the originals' in CTest and pytest).
- Changed: `dyng.Array.ndim` and `.shape` report the array's dimensions (2 for the path costs;
  every other result array is 1-D as before), and `len()` is the length of the first dimension.
- Changed (build): `DYNG_BUILD_PYTHON=ON` needs mosp in the build (`DYNG_ALGORITHMS=all`).
- Docs: the mosp page's Python and command-line sections and its performance table, the sssp
  page's engine table (operators against fused) and engine choice from Python and the command
  line, the command-line and Python API references, ADR 0028.
- CI: `ci/gpu_local.sh` replays the golden corpus with both CUDA engines (`--configs
  cuda,cuda-operators`, the whole MOSP update per case) and runs synccheck on the CUDA mosp
  suite.

### M7: review fixes (0.2, branch `m7-mosp`)

- Fixed: on CUDA, a `mosp::update()` that added vertices went one host synchronization over its
  I9 budget (the release of the old pinned copy of the MOSP tree), which a Debug build logged and
  strict budgets turned into `internal_error` and a poisoned result; the reserving run now counts
  it. The budget of mosp's combined solve on the operators engine is a static solve's,
  2 + iterations + epochs (it was the update's 3 + ..., one too many).
- Fixed (tests): conformance check C8 runs with `engine::automatic` and, where a backend has a
  second engine, `engine::operators`, against the budget of the engine that ran
  (`test_traits::host_sync_budget(backend, stats)`, optional); it failed on a device without
  cooperative launch, where `engine::automatic` runs the operators engine. The mosp test
  executables run every test with strict budgets.
- Docs: the packing window (ADR 0029, proposed, for the author): for non-canonical imported trees
  inside (n - 1) * W <= max_distance < n * W the host backends (MOSP-OpenMP's packing rule) and
  cuda (MOSP-CUDA's) return different tie parents, each byte-identical to its original; pinned by
  `SsspPackingWindow.NonCanonicalTreesFollowEachOriginalsPackingRule`. The `@sync` contracts of
  `sssp::compute()` / `update()` give the operators engine's counts; `mosp.hpp`'s determinism
  and error texts corrected; the gate record lists PLAN 8.6's noisy flags; the `.dgb` batches
  move to M8.
- Merged `main` at 0.1.0rc1 (pull request #5); the release certificate (`parity/certify.py`)
  knows mosp's paper-scale goldens and fixtures.

### M6a: the CUDA plugin wheels (0.2, branch `m6a-cuda-wheels`)

- Added: the CUDA plugin distributions **`dyng-cu12`** and **`dyng-cu13`** (ADR 0030; PLAN 5.4,
  7.7, 7.8), built from the core sdist: `ci/plugin_pyproject.py` renders a plugin's
  `pyproject.toml` from the root one (name, `dyng==<same version>`, the `dyng.backends` entry
  point, `DYNG_ENABLE_CUDA=ON`, the static CUDA runtime, the release architectures of
  `cmake/cuda_architectures.cmake`: SASS for sm_75-sm_120 and PTX for sm_120), and
  `ci/plugin_wheels.sh` builds, repairs (manylinux_2_28, libgomp bundled, `libcuda` never),
  checks and install-tests them locally. The import packages `dyng_cu12` / `dyng_cu13` come from
  one source, `python/plugin/dyng_plugin` (a ctypes probe of the driver: `status()`,
  `available()`; the extension module `native` imported on first access). About 5.5 MB per
  wheel.
- Added: the extras `cu12` / `cu13` of `dyng` (`pip install "dyng[cu13]"`), pinned to the same
  version through scikit-build-core's dynamic metadata, and `cupy-cu12` / `cupy-cu13`.
- Added: the CMake option `DYNG_PYTHON_PLUGIN` (empty, `cu12`, `cu13`): the Python module of a
  plugin (`dyng_cu<N>._core`, nanobind domain `dyng_cu<N>`); the module's `build_config` reports
  `plugin`, `cuda_toolkit`, `cuda_architectures` and `cuda_runtime`, and `dyng.show_config()`
  prints them for a CUDA module.
- Changed: `CMAKE_CUDA_RUNTIME_LIBRARY=Static` now links the static CUDA runtime everywhere
  (libdyng, its modules and the tests linked the shared `CUDA::cudart` explicitly) and needs
  `BUILD_SHARED_LIBS=OFF` (a shared libdyng would put two CUDA runtimes into one process);
  `Shared` stays the default.
- Changed: `ci/wheel_check.py` checks plugin wheels (contents, entry point, dependency, licence
  files, no `libcuda` / `libcudart` needed or bundled, read from the module's ELF dynamic
  section) and the CPU wheel's pinned extras; the sdist must contain the plugin sources.
- Changed: the messages of `Resources.cuda()` and `show_config()` in the CPU module name
  `pip install "dyng[cu13]"` instead of "the 0.1.x releases".
- Licences: `THIRD_PARTY_LICENSES_CUDA.txt` (the CUDA runtime, CUB, Thrust, libcu++, NVTX) and
  the CUDA toolkit's EULA (`NVIDIA_CUDA_EULA.txt`, copied from the toolkit at build time) are
  licence files of the plugin wheels. Their licence expression stays the CPU wheel's until the
  author decides (GOVERNANCE.md, open decisions).
- Added: the choice of the CUDA plugin in `dyng` (ADR 0031; PLAN 5.4): of the installed plugins
  of the same version that can run here, the one of the driver's CUDA major wins (else the
  newest older major); a plugin of another version than `dyng` is never loaded; a plugin whose
  module fails to load passes to the next. With plugins installed but none usable, dynG runs on
  `dyng._core` and issues one **`dyng.BackendWarning`** (new; a `UserWarning`) naming each
  plugin's reason and the remedy; without plugins nothing changes. `DYNG_CPU_ONLY=1` and
  `dyng.use_cpu_only()` choose the CPU module without the warning. `dyng.show_config()` prints
  why the module was chosen and every plugin's state; `dyng.config()` has `selection` and
  `plugins`; `Resources.cuda()` in the CPU module says why no plugin is used.
- Added: `dyng.Array` in device memory (ADR 0031; PLAN 5.4 rule 4): `__cuda_array_interface__`
  (version 3, read-only, with the stream of the call that last wrote the result), DLPack >= 1.0
  exports ordered on the consumer's stream with a CUDA event (`__dlpack__(stream=...)`;
  `to_torch()` / `to_cupy()` use it), `shape` / `dtype` without reading elements, `to_numpy()` as
  a host copy, and `to_numpy(copy=None)` (new: the read-only view of host memory or a kept
  read-only host copy of device memory). Inputs in CUDA device memory (CuPy, PyTorch, JAX, a
  device `dyng.Array`) are accepted and copied to the host once. A `Resources.cuda(stream=...)`
  keeps its stream object alive, and so do the graphs, results and arrays made with it.
- Added: the GPU tests of the Python package, `python/tests/test_cuda.py` (marker `gpu`): sssp,
  cycle_count and mosp on CUDA equal the sequential backend element by element, a subset of the
  goldens on CUDA (MOSP's files byte for byte, CycleEnumeration-GPU's CUDA histograms), device
  arrays, device inputs, PyTorch / CuPy round trips, the fallback; they run in
  `ci/plugin_wheels.sh` (and in a venv with PyTorch / CuPy via `DYNG_PLUGIN_INTEROP_PYTHON`) and
  in the new step `plugin` of `ci/gpu_local.sh`. `test_backend_selection.py` checks the choice
  with fake plugins on any machine.
- Docs: the install guide's section on the CUDA plugin wheels (which plugin for which driver,
  troubleshooting the fallback warning); the Python API page's rules for the native module and
  device arrays.
- Added: the CUDA plugin wheels in CI (ADR 0032; PLAN 7.7, 7.8, 8.8): `wheels.yml` builds
  `dyng-cu12` and `dyng-cu13` from the run's sdist with cibuildwheel in the manylinux_2_28 image
  (`ci/cibuildwheel-plugin.toml`; `before-all` = `ci/cibw_plugin.sh` installs the CUDA toolkit,
  checks it and renders the plugin's tree), checks them (`twine check --strict`,
  `ci/wheel_check.py` with the 90 MB budget), and install-tests each with the CPU wheel in fresh
  venvs on Python 3.12 and 3.13 without a GPU: `ci/plugin_smoke.py --expect fallback` (one
  `dyng.BackendWarning`, "no CUDA driver", the plugin's module importing without a driver) and
  the whole pytest suite. Artifacts `wheel-cu12-...` and `wheel-cu13-...`; the input `plugins`
  turns the plugin jobs off.
- Added: `ci/cuda_toolkits.toml`, the CUDA toolkits of the CI builds pinned file by file (every
  RPM of NVIDIA's RHEL 8 repository the build installs, with its SHA-256): CUDA 12.9 for `cu12`
  and CUDA 13.4 for `cu13`, the latest of each major. `ci/cuda_toolkit.py` re-pins them from the
  repository's metadata (`lock`), downloads and verifies them (`download`, `verify`; the runner
  caches the files), and unpacks them without root (`extract`), so the CI toolkits can be used
  locally (`DYNG_CUDA13_ROOT=... ci/plugin_wheels.sh`).
- Changed: `release.yml` publishes the CUDA plugins from v0.2.0 (release candidates included):
  `select` names the distributions of the tag (`ci/wheel_check.py --release-distributions`),
  `wheels.yml` builds the plugins too, `collect` checks that the files are exactly those
  distributions (`--release-set`) and sorts them into one directory each (`--split`), and the
  publish jobs run once per distribution, each in its own environment: `dyng` through
  `testpypi` / `pypi`, `dyng-cu12` through `testpypi-cu12` / `pypi-cu12`, `dyng-cu13` through
  `testpypi-cu13` / `pypi-cu13`. TestPyPI first; PyPI only for final versions, after every
  TestPyPI upload, with the author's approval of each `pypi*` environment (three for 0.2.0).
  v0.0.x and v0.1.x tags publish `dyng` alone, as before.
- Added: `ci/plugin_smoke.py` (the smoke test of an installed plugin wheel: `--expect cuda` on a
  GPU machine, `--expect fallback` without a driver or device), used by `wheels.yml` and
  `ci/plugin_wheels.sh`, and `ci/without_cuda_driver.sh` (runs a command with the NVIDIA driver
  hidden in a user and mount namespace, no root), with which `ci/plugin_wheels.sh` also runs the
  hosted runners' smoke test.
- Added: `parity/wheel_vs_parity.py`, the informational "wheel vs parity build" row of PLAN 7.7:
  the same CUDA work (sssp's update on roadNet-CA, cycle_count on DD) timed from Python through
  the plugin wheel and through the `parity-cuda` build, under the exclusive perf lock with locked
  clocks; recorded in `parity/results/M6a-wheel-vs-parity.json`, never gated.
- Docs: `docs/developer/wheels.md` (the plugins in CI, the pinned toolkits and how to re-pin
  them, the CI toolkits locally, CI vs local builds, the per-distribution environments),
  `docs/developer/release.md` (three approvals for a final release with the plugins, the local
  rehearsal with the plugins), `docs/developer/repository_settings.md` (the protection rules of
  the four plugin environments, for the author), and the install guide (which plugin for which
  driver, the requirements, a GPU-less machine).

### M6a: review fixes (0.2, branch `m6a-cuda-wheels`)

- Fixed: `DYNG_CPU_ONLY` is read as a boolean: `1`, `true`, `yes`, `on` (any case) choose the
  CPU module; `0`, `false`, `no`, `off` and empty leave the choice to dynG (before, every value
  but `0` and empty forced the CPU module, `false` included); any other value is ignored with a
  `dyng.BackendWarning`.
- Fixed: choosing the CUDA plugin no longer initializes CUDA in the process (ADR 0031,
  amendments): the plugin reads the driver's version with `cuDriverGetVersion` and the devices
  through NVML (or a short child process when NVML cannot tell which devices CUDA will see), so
  a process that used dynG (even `dyng.__version__` or CPU work) can still fork workers that use
  CUDA. Before, every first use called `cuInit`, and a forked child's CUDA calls failed with a
  misleading "no CUDA device is visible". A child forked after its parent really used CUDA now
  gets an error that names fork and the remedy (`spawn` / `forkserver`).
- Fixed: a plugin is used only when a visible GPU has compute capability 7.5 or newer (the
  plugins' oldest architecture); on Volta, Pascal and older GPUs dynG falls back to the CPU
  module with a `dyng.BackendWarning` that names the GPU's architecture, instead of choosing the
  plugin and failing every default call with `cudaErrorNoKernelImageForDevice`. That CUDA error
  now names the device's compute capability.
- Fixed: `Resources.cuda(stream=0)` is the legacy default stream, as `cudaStream_t` 0 is in CUDA
  and in C++ (`stream_ref(0)`), and so are `torch.cuda.default_stream()` and
  `cupy.cuda.Stream.null`; `stream=None` (the default) is the per-thread default stream. Before,
  0 and the frameworks' default streams were silently mapped to the per-thread default stream.
- Fixed: a graph or result used with other resources (`resources=Resources.cuda(stream=s)` in
  the algorithms' `compute` and `update`, `dyng.update` or `Graph.apply`) keeps the stream object
  of every such resources alive as long as itself and its Arrays (memory a call adds is released
  on the call's stream); before, dropping the stream left `__cuda_array_interface__` naming a
  destroyed stream, and `to_numpy()`, freeing the graph or dropping an Array failed with a CUDA
  error or crashed the process.
- Fixed: `dyng.config()` / `dyng.show_config()` no longer initialize CUDA when a plugin is chosen
  (the CUDA entry comes from the plugin's probe), so printing the configuration does not stop the
  process from forking workers that use CUDA.
- Changed (release): `release.yml` uploads the CUDA plugins before `dyng` on each index, and
  `dyng` only when every plugin upload succeeded, so `pip install "dyng[cu13]"` never meets a
  `dyng` whose plugin is missing (pip would fall back to an older `dyng` without the extra with
  only a warning). `ci/tests/test_release_select.py` tests the `select` step for every kind of
  tag. The release checklist runs the GPU tests on the CI-built wheels of both plugins before
  their PyPI environments are approved.
- Changed (CI): the plugin wheels are checked for exactly the SASS and PTX of their toolkit's
  release list (`ci/wheel_check.py --code-objects` with cuobjdump, in the container; the smoke
  test checks the architectures the module reports), their module is linked with
  `-Wl,--exclude-libs,ALL` as in the local build and may export only its init function and std /
  nanobind / type_info symbols; the pinned CUDA RPMs are checked against NVIDIA's OpenPGP
  signature (key fingerprint pinned; gpg on the runner, `rpm -K` and `localpkg_gpgcheck` in the
  container); `wheels.yml` builds the plugins for pull requests that change the library or the
  licence files too.

### M6a: acceptance fixes (0.2, branch `m6a-cuda-wheels`)

- Fixed: `ci/plugin_wheels.sh` checks every plugin's CUDA toolkit for `bin/nvcc` and
  `bin/cuobjdump` before anything is built (or the output directory cleared) and names what is
  missing; before, a toolkit without cuobjdump failed the code-object check only after the build.
- Docs: `docs/developer/wheels.md`: the conda-forge CUDA 12.9 recipe for the local cu12 build
  installs `cuda-cuobjdump` (it could not pass the code-object check without it).
  `docs/developer/repository_settings.md` and `release.md`: a final release with the plugins is
  approved in two "Review deployments" dialogs (`pypi-cu12` and `pypi-cu13`, then `pypi`), as
  `release.yml` orders them.

### M6a: the plugins' licence expression (0.2, branch `m6a-cuda-wheels`)

- Changed: the `License-Expression` of `dyng-cu12` / `dyng-cu13` names what the plugins add to
  the CPU wheel's contents: `... AND Apache-2.0 WITH LLVM-exception AND
  LicenseRef-NVIDIA-End-User-License-Agreement` (libcu++ and NVTX; the statically linked CUDA
  runtime under NVIDIA's CUDA Toolkit EULA). The author's decision of 2026-10-06 (GOVERNANCE.md;
  ADR 0030 item 9). The CPU wheel `dyng` and the sdist keep their expression.
  `ci/wheel_check.py` refuses a plugin wheel with any other expression.
- Docs: `docs/developer/repository_settings.md` item 11b is done: the four plugin environments
  have the protection rules of `testpypi` / `pypi`.

## [0.1.0] - 2026-10-02

### Summary

The first release of dynG: two dynamic algorithms, each ported from a pinned research code and
proved equal to it, with one C++ API, a Python package and a command line. Install it with `pip
install dyng` (Python >= 3.12, Linux x86-64). 0.1.0 is the release candidate 0.1.0rc1 (tag
`v0.1.0rc1`, published on TestPyPI only and smoke-tested from there) with the final-release
changes below and no change to the library. The entries below the summary are the detailed
record of milestones M1a to M5 and of the release preparation.

- **Algorithms.** `sssp` (dynamic single-source shortest paths: DynaMOSP's SOSP update,
  byte-identical to MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2 on their 495-case golden corpus)
  and `cycle_count` (exact k-bounded directed simple-cycle histograms: the TruCy / DynTruCy
  update, bit-identical to CycleEnumeration-GPU@0a976ad on its CPU and CUDA corpora), each on the
  sequential, OpenMP and CUDA backends, within the performance gates against the originals.
  Both are **stable** (SemVer applies from 0.1.0).
- **One update model** (the template of thesis Chapter 3) in an internal framework, with the
  conformance kit, the scaffold `scripts/new_algorithm.py`, and `dyng::update(res, g, batch, r1,
  r2)` for several results on one graph.
- **The C++ API** of `core/*`, `graph/*`, `update.hpp`, `sssp.hpp` and `cycle_count.hpp`, reviewed
  and frozen for 0.1 (ADR 0023); CMake package `dyng::dyng`.
- **The Python package** `dyng` (Python >= 3.12; one abi3 manylinux_2_28 x86_64 CPU wheel with the
  sequential and OpenMP backends; ADR 0011) and the **`dyng` command line** (ADR 0025), which
  reads and writes the originals' file formats.
- **Documentation:** getting started in Python and C++, the update model, the algorithm pages
  with "Differences from the paper" and "Paper vs fixed code", the C++, Python and CLI
  references, the history of the ported codes, and the developer guides.
- **The release certificate** `benchmarks/results/0.1.0/` (PLAN 8.3 and 8.6): golden parity
  on every backend (sssp 4950 of 4950 replays byte-equal, cycle_count 144 of 144), the release
  performance gate of both algorithms against the unpatched originals (sssp 132 of 132 gated
  regions, ratios 0.63-1.05; cycle_count 78 of 78, ratios 0.23-1.01; CUDA at locked clocks,
  ADRs 0018 and 0021), the sanitizers and the mutation checks; the benchmark-suite records of
  the papers in `benchmarks/paper/`.

Known limitations of 0.1.0: no CUDA backend in the Python wheel (the plugin wheels `dyng-cu12` /
`dyng-cu13` follow in 0.1.x); on CUDA, `sssp` needs cooperative launch (the operators engine
follows in 0.2); `cycle_count` counts simple cycles only (no time-window or temporal modes, no
approximate TruCy mode); Linux x86-64 only. `dyng` 0.0.1 on PyPI was only the name reservation.

### Final release (R011)

- Changed: `VERSION` 0.1.0, `CITATION.cff` `version: 0.1.0`; this section, the release
  candidate's `[0.1.0rc1]`, renamed `[0.1.0]` with its link references.
- Docs: the status texts of the release candidate brought to the release (README, the
  documentation's start page and install page, `SECURITY.md` with its supported versions,
  `CONTRIBUTING.md`, `SUPPORT.md`, the roadmap and the developer plan): `pip install dyng`
  from PyPI.
- Added: `benchmarks/results/0.1.0/`, the certificate of the release, carried over from
  0.1.0rc1 (`docs/developer/release.md`, step 9): the library did not change since the measured
  commits, so the gates, the golden replays and the golden mutations stand; the distributions,
  `ci/check.sh --parity`, `ci/gpu_local.sh` and the C++ test suites (the sanitizer presets, the
  mutation CTests; `README.md` changed) were run again on the final tree. The smoke test of
  0.1.0rc1 from TestPyPI is recorded there.

### Tag day (R012)

- Changed: the release date 2026-10-02, the day of the tag `v0.1.0` (UTC), in this heading and
  in `CITATION.cff`'s `date-released` (`docs/developer/release.md`, step 10).
- Docs: the author's decisions recorded in `GOVERNANCE.md`: the `NOTICE` wording confirmed as
  drafted (O3, 2026-10-01; `NOTICE` unchanged); Zenodo not connected for 0.1.0, so 0.1.0 has no
  DOI (2026-10-01); the licence expression of the distributions: no objection by the tag day,
  the approval of the `pypi` deployment is the final confirmation. The release guide, the
  repository settings (Zenodo), the citing page, the roadmap, the developer plan and the R011
  retrospective say so.
- Changed: the certificate's checks of the `packaging` and `repo` scopes (the distributions,
  `ci/check.sh --parity`, `ci/gpu_local.sh`) run again on the tag-day tree and recorded; the
  certificate written again (`benchmarks/results/0.1.0/`).

### Release preparation (R010)

- Added: the benchmark-suite records of PLAN 8.5, `benchmarks/paper/ieee_tc_dyntrucy.yaml`
  (cycle_count, the static and update cases of DynTruCy on the TU datasets DD, github_stargazers,
  twitch_egos and COLLAB; the temporal datasets wait for the time-window mode of 0.4) and `benchmarks/paper/ipdps25_dynamosp_sosp.yaml`
  (sssp, the per-objective regions of DynaMOSP on roadNet-PA, roadNet-CA, rgg and road_usa_g,
  three batches each), with datasets (SHA-256 or recipe), batches, backends, runs, metrics,
  baselines and tolerances; `parity/bench_suite.py` (`validate`, `run`, `summary`) runs a suite
  through the A/B harnesses and writes its results JSON, listing the runs flagged for foreign
  CPU load (`docs/developer/benchmarks.md`).
- Added: `parity/certify.py` writes the parity certificate `benchmarks/results/<version>/parity.json`
  and the tables of its `README.md` (commit, hardware, driver, CUDA, the originals' SHAs, every
  golden set with its result, the gate table, the checks), records the release checks
  (`certify.py check`) and compares the builds of a measured commit and of the release when they
  differ only in the registry's metadata (`certify.py equivalence`); `parity/mutate.py` checks
  that mutations of sssp fail its goldens on every backend.
- Added: `benchmarks/results/0.1.0rc1/`, the certificate of the release candidate.
- Changed: `sssp` and `cycle_count` are `stable` (their manifests, the registry and the tables
  generated by `scripts/regen.py`).
- Changed: the distributions' `License-Expression` names the licences of what the wheel bundles:
  `Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1` (dynG's
  Apache-2.0, nanobind, robin-map and the GCC runtime; the author's decision of 2026-09-30).
  `ci/wheel_check.py` checks it in `pyproject.toml` and in every distribution's metadata.
- Changed: `VERSION` 0.1.0rc1; `CITATION.cff` version 0.1.0rc1, released 2026-10-01; the README
  and the documentation's status line say "alpha: 0.1 release candidate".
- Governance: the author lets the AI assistant push the release tags and create the GitHub
  Release on the author's behalf, and keeps the approval of the `pypi` deployment (2026-09-30;
  `docs/developer/release.md` says who does what); the `main` ruleset requires 24 checks
  (the Python, sdist, scaffold and API checks of M5 added), and the Actions allow-list includes
  `pypa/cibuildwheel` (`docs/developer/repository_settings.md`).
- Fixed: the dataset parity tests run in builds without OpenMP.
- Fixed (R010 review): the certificate requires each suite summary to cover the whole committed
  suite (SHA-256 of the suite file, every reading of its plan once, complete) with verified
  inputs; it names the driver of the measurements (the harness records it now), every committed
  fixture set of `cpp/tests/data` with its digest, original and test results
  (`parity/fixtures/fixtures.toml`), and each check's scope (new: `packaging`, `repo`) and
  evidence (SHA-256 and a committed excerpt). `bench_suite.py` keeps a narrowed run out of the
  release results (`<suite>.partial.json`) and never replaces records without `--force`; the
  sssp harness hashes its inputs when a run starts. `ci/wheel_check.py --release-metadata`
  checks `VERSION`, this file and `CITATION.cff` against each other (in `release.yml`'s
  `select` job and `ci/tests`); `scripts/new_algorithm.py` writes its CHANGELOG entry again
  when `Unreleased` is empty; the PyPI description names the bundled licences.

### Added

- The batch text format `.dgt` of PLAN Section 5.7 (M5 review): `dyng::io::read_batches<V, W>()`
  / `write_batches<V, W>()` with `batch_file_options` in `<dyng/io/batch_io.hpp>` (a tracked
  header; the API baseline gains these three declarations), `dyng.io.read_batches()` /
  `write_batches()`, and `.dgt` files for `dyng cycle_count update --batch`. The format:
  `docs/api/file_formats.md`. Vertex and hypergraph operations are reserved (0.3, 0.2).
- Documentation for 0.1 (M5, PLAN Sections 9.2-9.6): the Python API reference generated by
  sphinx-autoapi from `python/dyng` and the committed stubs (no compiled module needed;
  `docs/api/python/`, with the package-wide rules of dtype dispatch, arrays, stale results,
  exceptions and threads), getting started in Python (`pip install dyng`, the ten-line
  quickstarts, "A first update in Python"), Python and CLI examples on the `sssp` and
  `cycle_count` pages, the "Port research code" how-to (PLAN 6.3), the history pages of
  MOSP-OpenMP, MOSP-CUDA and CycleEnumeration-GPU, and the release process
  (`docs/developer/release.md`, PLAN 10.3). `environment.yml` pins sphinx-autoapi 3.8.1.
- The README quickstarts (ten lines of Python and of C++, PLAN 9.6) are executed by the tests:
  `python/tests/test_doc_snippets.py` runs every marked Python snippet of the READMEs and the
  pages and compares its output, the CTest `example.readme_quickstart` builds and runs the C++
  one, and `ci/doc_snippets.py --check` (in `ci/docs.sh`) keeps them present and short.
- `examples/python` (`first_update.py`, `sssp_update.py`, `cycle_count_update.py`), tested against
  the originals' outputs by `python/tests/test_examples.py`.
- `.github/workflows/api-check.yml` and `ci/api_check.sh` (PLAN 5.9): griffe compares the Python
  API with the base branch and fails on a breaking change without the `api-change` label and a
  CHANGELOG entry; the `api` step of `ci/check.sh`. `environment.yml` pins griffe 2.3.0.
- Governance: `DCO` is a required check of the `main` ruleset (2026-09-30); the maintainer's
  commits are SSH-signed.

- The Python package `dyng` (M5, PLAN Sections 5.4 and 7.7; ADR 0011): a root `pyproject.toml`
  (scikit-build-core, nanobind 3.1.0, one abi3 wheel for CPython >= 3.12, the version from
  `VERSION`), the CMake option `DYNG_BUILD_PYTHON` building the extension module `dyng._core`
  (stable ABI, nanobind and `libdyng` linked statically, the sequential and OpenMP backends), and
  the typed layer in `python/dyng`: `Resources` (sequential, openmp; `cuda` raises
  `NotSupportedError` in the CPU wheel and names the CUDA plugins of 0.1.x) and the default
  resources, `Graph` (`from_edges` / `from_csr` with dtype dispatch that never narrows ids
  silently, property presets by name), `EdgeBatch`, `dyng.sssp` and `dyng.cycle_count`
  (`compute` / `update` / `Options` / `Result` / `Stats`), `dyng.update(graph, batch, *results)`,
  the exception hierarchy (`InvalidArgumentError` is a `ValueError`, `FileFormatError` an
  `OSError` with `.path` / `.line`, ...), `dyng.io` (edge lists, Matrix Market, MOSP's CSR,
  batches, distances, trees, histogram CSV), `dyng.generators.legacy`, `dyng.testing` (native
  oracles, pure-Python oracles, Hypothesis strategies), `profile()`, `citation()`,
  `show_config()`, `algorithms()`, `__version__`. Result arrays are zero-copy `dyng.Array` views
  (`__dlpack__`, `__array_interface__`, `to_numpy()`) that keep their result alive and raise
  `StaleResultError` once it is updated. The GIL is released around native work. The stubs
  `python/dyng/_core.pyi` are committed and checked with `python scripts/regen.py --stubs
  --check`; `scripts/regen.py` also generates `python/dyng/_algorithms.py`. What 0.1 does not
  bind is listed in `docs/developer/python_gaps.md`.
- The pytest suite `python/tests` (M5): API surface, dtype dispatch, DLPack / NumPy round trips,
  exception mapping, stale results, threads, a Hypothesis profile (random graphs and batches
  against pure-Python Dijkstra and brute-force cycle oracles), and Python-level parity with the
  committed goldens of MOSP-OpenMP (byte-identical files) and CycleEnumeration-GPU (identical
  histograms). `environment.yml` pins nanobind, scikit-build-core and Hypothesis.
- The `dyng` command line (M5, PLAN Section 5.6; ADR 0025), a console script of the Python
  package (also `python -m dyng`): `dyng sssp compute|update` (MOSP's distance and tree files,
  byte-identical to `mospPrep init` and `mosp`), `dyng cycle_count compute|update`
  (CycleEnumeration-GPU's histogram CSV), `dyng prep mtx2csr|widen|cache|changes|init|expected`
  (the `mospPrep` subcommands with their arguments and outputs, the binary cache included),
  `dyng convert` (MOSP CSR, Matrix Market, edge lists) and `dyng generate mosp_changes|cycle_enum_batch`.
  Option flags are the option fields in kebab case (`--max-length`). Reference:
  `docs/api/cli.md`; tests against the originals' fixtures in `python/tests/test_cli.py`.
- The distributions (M5, PLAN Section 7.7; ADR 0025): `[tool.cibuildwheel]` and
  `.github/workflows/wheels.yml` (the sdist and the manylinux_2_28 x86_64 abi3 CPU wheel with
  libgomp bundled, a pytest subset in the built wheel under 3.12 and 3.13, install tests of the
  wheel alone, artifacts), `.github/workflows/python.yml` (editable install, stubs, pytest; the
  suite against an installed sdist on 3.12 and 3.13), `ci/wheel_check.py` (the 90 MB budget,
  tags, contents), and the local build `ci/wheels.sh` / `ci/check.sh --wheels`
  (`docs/developer/wheels.md`). `release.yml` builds the real distributions for tags v0.1.0 and
  later through `wheels.yml` (TestPyPI first, PyPI for final versions after the author's
  approval); v0.0.1 still builds the name reservation. Nothing has been published.
- Framework (M3, internal-stable): the update template as code in `cpp/src/framework/`:
  `problem_base` (CRTP hooks with no-op defaults, families `fixed_point` and `aggregate_delta`),
  `update_enactor` and `static_enactor` (the fixed hook order, one profiler stage per implemented
  hook, the convergence cap with `on_limit`, the device error check), `old_view` / `new_view`
  (invariant I1), `context`, the policies, the budgets of the algorithm phase (invariant I9, with
  allocation and host-synchronization counters in `DYNG_DEBUG_BUDGETS` builds), compile-time
  conformance checks, and the participant adapter for `run_update` composition. The guide is
  `docs/developer/framework.md`.
- `sssp` runs through the framework (M3): `detail::sssp_problem` with the Tier A hooks on the
  sequential and OpenMP backends and the fused persistent kernel behind `enact_fused` (Tier B) on
  CUDA; migrated one backend per commit with the golden parity (495 / 495 on every backend) and
  the performance re-checked after each (`parity/results/M3.md`). No change to the public API,
  the profiler stages or the results. `parity/perf_ab.py run --baseline-exe` times dynG against
  an earlier dynG build.
- `cycle_count` runs through the framework (M3): `detail::cycle_count_problem`, an aggregate-delta
  problem with the ownership rule `ownership::min_member`, runs the Tier A hooks on the sequential
  and OpenMP backends (`count` on the old view subtracts on G_t, `count` on the new view adds on
  G_{t+1}, `finalize` applies the signed delta) and the ported CUDA kernels behind `enact_fused` /
  `compute_fused` (Tier B), with the resident device graph and Step 0 once per update (ADR 0020)
  unchanged; migrated one backend per commit with the golden parity (72 / 72 on the CPU and the
  CUDA corpus) and the performance re-checked (`parity/results/M3.md`). The public multi-result
  `dyng::update` now composes two framework problems. Profiles: under set semantics
  `cycle_count.normalize` is called twice per update (the framework's Step 0 and the hook that
  takes its lists), and on CUDA the ported code's stages sit inside a new `cycle_count.enact_fused`
  stage; every other stage is unchanged. The CUDA engines of both algorithms and the graph's device
  paths count their host synchronizations for the budgets (I9).
- The conformance kit (M3, PLAN Section 8.2; `cpp/tests/conformance/`,
  `docs/developer/conformance.md`): checks C0-C12 for every registered algorithm on every backend
  and graph type, from a `test_traits` specialisation and one line `DYNG_CONFORMANCE_SUITE(<name>)`
  (`dyng_<name>_conformance_tests`, labels `cpu;conformance;<name>`, and
  `dyng_<name>_conformance_cuda_tests` in CUDA builds); sssp and cycle_count pass it on
  sequential, OpenMP and CUDA. C8 runs in `DYNG_DEBUG_BUDGETS` builds (the dev presets) and counts
  host heap allocations too (a counting `operator new` in the conformance executables).
- The algorithm registry: `dyng::algorithms()` / `dyng::find_algorithm()` in
  `<dyng/core/registry.hpp>` (name, title, family, container, maturity, determinism, oracle kind,
  backends, cite keys of every algorithm built into the library), generated from the manifests;
  `dyng::citation()` knows every registered algorithm.
- `scripts/regen.py` (the algorithm tables of the README, the landing page and the algorithms
  index, the CODEOWNERS block, the registries; `--check` in pre-commit, `ci/check.sh` and
  `lint.yml`; it enforces the registration rules of invariant I8) and `scripts/new_algorithm.py`
  with `cpp/src/algorithms/_template` (a fixed-point or aggregate-delta algorithm on the host
  backends that builds and passes the kit on the first build; `ci/scaffold_check.sh`, CTest
  `scaffold.new_algorithm`, job `scaffold` of `cpu.yml`). The manifests gain `computes`, `paper`
  and `since`; the planned algorithms are listed in `cpp/src/algorithms/planned.toml`. The README
  has the generated algorithm table.
- The 0.1 API freeze (M3, ADR 0023, accepted under delegation; ADR 0006, the algorithm contract,
  accepted with it): the review of `core/*`, `graph/*`, `update.hpp`, `sssp.hpp`,
  `cycle_count.hpp` and the top-level headers, and the committed public-API listing
  `cpp/tests/api/api_snapshot/public_api.txt`, generated from the Doxygen XML by
  `ci/api_snapshot.py` and checked by `ci/docs.sh` (so by the `docs` workflow and `ci/check.sh`):
  a change of a public signature, default value, field or enumerator fails until it is reviewed
  and the baseline updated (`docs/developer/api_review_checklist.md`, "Updating the API
  baseline"). The frozen headers are marked `frozen`, `io/*`, `generators/*` and `testing/*`
  `tracked` (frozen in M5).
- `to_string()` for `engine`, `determinism`, `memory_space`, `copy_policy`, `algorithm_family`,
  `container_kind`, `maturity_level` and `oracle_kind` (the enumerators' own names, as in the
  manifests).
- Reviewed API sketches of the later algorithms, written against the frozen contract:
  `docs/design/sketches/` (`mosp`, the `hypergraph` with `hyperedge_batch`, `triad_count`,
  `label_propagation`, `hyper_sssp`).
- The "Add an algorithm" guide outline (`docs/how_to/add_an_algorithm.md`, PLAN Section 9.4) and
  the M3 retrospective with the re-estimate (`docs/developer/retrospectives/M3.md`); the M3 gate
  suites on the final code in `parity/results/M3.md` section 4.
- Governance: ADRs 0020 and 0021 accepted by the author (2026-09-29); technical ADRs may be
  accepted under delegation; the maintainer's commits are SSH-signed so that the DCO app exempts them.
- ADR 0018 accepted (option B): the CUDA performance gate is read with the GPU clocks locked for
  the whole A/B; default-clock readings are recorded, not gated.
- Repository bootstrap: Apache-2.0 license, NOTICE, CITATION.cff, AUTHORS, governance and
  contribution guides, ADRs 0001 (name), 0002 (license and credit), 0004 (naming) and 0014
  (approval checkpoints).
- Project infrastructure (M4): SECURITY.md with the disclosure process, SUPPORT.md,
  MAINTAINERS.md, a complete CONTRIBUTING.md, eight GitHub issue forms and the issue chooser,
  the pull request template with the review checklist, CODEOWNERS, the DCO app configuration,
  the label taxonomy (`.github/labels.yml`, applied by the `labels` workflow;
  `docs/developer/labels.md`), `ci/github_meta_check.py` (a pre-commit hook),
  `.readthedocs.yaml` (not connected yet) and the repository settings guide
  (`docs/developer/repository_settings.md`).
- Documentation site (M4): Sphinx + MyST + pydata-sphinx-theme + Breathe over the Doxygen XML,
  in the Diataxis sections (getting started, tutorials, how-to guides, explanation including
  the update model, reference with the curated C++ API and file formats, developer pages with
  the ADRs, retrospectives and the parity guide); a landing page, "How to cite" from
  `docs/references.bib`, the public roadmap (`docs/roadmap.md`) and the short plan
  (`docs/developer/plan.md`). `ci/docs.sh` builds it with warnings as errors and checks internal
  links (`--doxygen-only` for the Doxygen check alone); the `docs` workflow builds it on hosted
  runners from `environment.yml` (which now pins the Sphinx tools) and uploads the site.
- Workflow hardening (M4): every `actions/checkout` sets `persist-credentials: false`,
  `release.yml` passes step outputs to its scripts through `env:`, dependabot waits 7 days
  before proposing a new action release; pre-commit runs `actionlint` (with shellcheck) and
  `zizmor` on the workflows, and `ci/github_meta_check.py` enforces the least-privilege and
  SHA-pinning rules (`--verify-pins` checks each SHA against its version tag).
- Contributor infrastructure after review (M4): a hosted clang-tidy naming job (`tidy` in
  `lint.yml`; `DYNG_CHECK_ONLY` in `ci/check.sh`), the action pins verified in CI, timeouts on
  every job, the first-interaction `welcome` workflow; `ci/docs_links.py` checks links to
  repository files and the site's anchors; `examples/cpp/first_update.cpp` (run by CTest, quoted
  by the docs); `DYNG_ORIGINALS_DIR` for the parity harness; the API review checklist and the
  provenance record (`docs/developer/`); a Code of Conduct escalation path and GOVERNANCE
  contacts.
- Build system: CMake >= 3.30 with presets `dev`, `release`, `relwithdebinfo`, `cpu-only`,
  `asan`, `tsan` and `parity`; the `dyng-dev` conda environment (`environment.yml`) and
  `scripts/dev_env.sh`; install and export of `dyng::dyng` for `find_package(dyng)`.
- Core: error hierarchy with `DYNG_EXPECTS` / `DYNG_FAIL`; backends and `resources`
  (sequential, OpenMP with a thread count); `stream_ref`; `memory_resource_ref` and the host
  memory resource; `array_view`; `buffer`; host copies; logging with a replaceable sink; an
  instance-based profiler with `<algo>.<hook>` stage names; `citation()`.
- Quality: GoogleTest unit tests with CTest labels, the header self-containment check,
  `ci/check.sh`, the Doxygen check, pre-commit (clang-format, REUSE, codespell), and the
  `lint` and `cpu` GitHub workflows.
- Graph: `graph<V,E,W>` (pimpl, host storage, compact rows, objective-major weight columns,
  stored in-edges, version counter), `graph_view`, `csr` / `csr_view`, `edge_list` /
  `edge_list_view`, `edge_batch` / `edge_batch_view`, `apply_summary`, `graph_properties`
  with `row_order`, `multi_edges`, `batch_semantics` and the preset `mosp_compatible()`;
  `graph::apply` is MOSP's `applyChangeBatch()` ported straight (ADR 0010).
- I/O: Matrix Market reader and writer (seeded random weights bit-exact with `mospPrep
  mtx2csr`), the MOSP text CSR, `insert.txt` / `delete.txt` batches, distance and SSSP-tree
  files; strict parsing with `io_error` carrying path, line and column
  (`docs/api/file_formats.md`).
- Byte-parity fixtures for the updated CSR, its transposition and the weight-increase flags
  against MOSP-OpenMP@c352151 (`parity/fixtures/graph_io/`).
- `sssp`: `compute()` and `update()` on the sequential backend (MOSP's `sequentialSOSPUpdate`
  adapted into hooks) and the OpenMP backend (MOSP-OpenMP's `sospUpdateCpu` /
  `sospFromScratchCpu` ported straight); canonical trees identical across backends and
  byte-identical to MOSP-OpenMP@c352151 on the test corpus; `result::from_arrays` with
  canonicalization and `validate_inputs`; `stale_result_error` detection
  (`docs/algorithms/sssp.md`, ADR 0006).
- `update_stats`, and `dyng::update(res, g, batch, results...)` / `dyng::update_each()` to apply a
  batch once for several results.
- `dyng::testing` (installed target): `dijkstra()` with lowest-id ties and `check_sssp_tree()`.
- `tools/compat/dyng-compat-mosp` (drop-in clone of the per-objective part of MOSP's `mosp`
  driver and of `mospPrep init`), the `sssp_update` example, `parity/timed_regions/sssp.toml`
  and the sssp byte-parity fixtures (`parity/fixtures/sssp/`).
- The parity harness (`parity/`, ADR 0013): `references.toml` (MOSP-OpenMP c352151, MOSP-CUDA
  e220ee2 and their baseline SHAs), `build_reference.sh` (verified `git archive` scratch copies,
  unpatched and patched, archive-build check), additive export tools, `export_goldens.py` (the
  sssp golden corpus, now 495 cases, with the originals cross-checked, reproducible export),
  `goldens.toml`, `compare.py` (CTest label `parity`), `perf_ab.py` (A/B/A/B under the perf
  lock), and the M1a parity certificate `parity/results/M1a.md`.
- `dyng-compat-mosp --write-graph` writes the updated graph in MOSP's CSR text format.
- A pure-Python name-reservation package `dyng` 0.0.1 (`tools/name_reservation/`) and the
  Trusted Publishing workflow `release.yml`.
- `ci/check.sh --parity` (the golden replay as part of the local gate) and a clang-tidy step
  with the naming rules of ADR 0004; `ci/doxygen_coverage.py` (a `@brief` on every public
  entity, every entity in a group, `@backends` / `@determinism` / `@paper` on `compute` and
  `update`) run by `ci/docs.sh`, also as the CMake target `docs-doxygen`.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 3.0), `SECURITY.md` and `SUPPORT.md`.
- The M1a retrospective with the acceptance record and a re-estimate of the roadmap
  (`docs/developer/retrospectives/M1a.md`).
- `legacy_batch_options::mosp_lenient` (MOSP's accept/reject rules for batch files; used by
  `dyng-compat-mosp`), `sssp::stats::packed_parents` on both CPU backends, and the golden group
  `noncanonical` (107 cases with perturbed tie parents; 495 golden cases in total).
- `ci/provenance_check.py` (provenance headers of ported files), ruff lint and format for the
  Python harness, smoke tests of the harness (`parity/tests`), Clang 17/18 jobs in `cpu.yml`.
- `resources` owns a workspace pool: `compute()` / `update()` lease the engines' scratch memory
  from it, so results run through one handle share one workspace (ADR 0015);
  `resources::release_workspaces()` and `resources::workspace_bytes()`.
- `graph::from_csr(res, csr&&, props)`: takes over the arrays of a CSR that already has the
  requested form.
- The OpenMP A/B on roadNet-PA, roadNet-CA, rgg_n_2_20_s0 and road_usa with the PLAN 8.6 gates
  (`parity/results/M1b.md`); `perf_ab.py` gates `end_to_end` and `apply` at 1.10x.
- CUDA core (ADRs 0003 and 0016): `DYNG_ENABLE_CUDA=ON` builds (`cmake/cuda_architectures.cmake`:
  `native` for development, the release list per toolkit, CUDA >= 12.4), the presets `dev-cuda`,
  `release-cuda`, `parity-cuda`, `sanitize-cuda`, `ci-cuda12` and `ci-cuda13`;
  `resources::cuda(device, stream)` with the device's capabilities recorded once, a device guard
  on every call, `synchronize()` and `warm_up()` (context, every library kernel through the kernel
  registry, stream and pool); `cuda_async_memory_resource` (own stream-ordered pool, the default
  through `default_device_memory_resource()`) and `pinned_host_memory_resource`; device
  `buffer<T>` and copies between every pair of spaces; the private `DYNG_CUDA_TRY`,
  `DYNG_CUDA_TRY_NO_THROW`, `DYNG_CHECK_KERNEL`, the sticky device error word, the fill kernels,
  CCCL 3.x memory-resource adapters and device workspaces (`scratch_buffer`) in the pool of ADR
  0015.
- `ci/gpu_local.sh` (the local GPU gate: `ctest -L gpu` on GPU 1, the CPU tests of the CUDA build,
  compute-sanitizer memcheck, clang-tidy on the CUDA branches), `ci/build_cuda.sh` and the
  compile-only workflow `cuda-build.yml` (CUDA 13.1.1, 13.3.1 and 12.9.2 containers, register
  and library-size report).
- `sssp` on the CUDA backend (ADR 0017): MOSP-CUDA@e220ee2's persistent cooperative kernel ported
  verbatim as the fused engine (`options::cuda_engine`: `automatic` / `fused`;
  `not_supported_error` without cooperative launch, `operators` in 0.2); `compute()`,
  `update()`, `from_arrays()` (host or device arrays) and `clone()` across host and device;
  result arrays in device memory. Byte-identical to MOSP-CUDA on all 495 golden cases; the CUDA
  test executable `dyng_sssp_cuda_tests` (label `gpu`) runs the shared sssp suites on cuda and
  compares cuda with the host backends on randomized inputs.
- Graphs built with CUDA resources: a resident device copy per graph state (out- and in-edges,
  objective-major weight columns; the in-edges built on the device as MOSP-CUDA's
  `uploadDeviceGraph` does), uploaded on first use and inside the commit of `dyng::update`.
- `profiler_options::cuda_events`: device times of profiler stages from CUDA events.
- `generators::legacy::mosp_changes()`: MOSP's change generator (`mospPrep changes`), bit-exact for
  fixed seeds, with committed fixtures from both originals; `dyng-compat-mosp changes`.
- `dyng-compat-mosp --backend cuda [--device d]` (CUDA-event times in its `--timing` CSV),
  `compare.py --configs cuda`, the CTest golden replay `parity.sssp.mosp_cuda_e220ee2`, and
  `perf_ab.py run --backend cuda` against the unpatched MOSP-CUDA; `ci/gpu_local.sh` replays the
  golden corpus on cuda.
- The M1b parity certificate (`parity/results/M1b.md` sections 7-12): byte parity with
  MOSP-CUDA@e220ee2 and MOSP-OpenMP@c352151 on all 495 golden cases for every backend and both
  edge offset types, and the CUDA and OpenMP performance gates on roadNet-PA, roadNet-CA,
  rgg_n_2_20_s0 and road_usa; ADR 0018 (the GPU clock state in the CUDA gate, proposed).
- `parity/export_goldens.py --reference MOSP-CUDA --compare-to` (the corpus from MOSP-CUDA's own
  tools, compared file for file with the committed one); `compare.py` configurations with
  `/int32` or `/int64`; `dyng-compat-mosp --edge-type`; `perf_ab.py kernels` (the fused kernels
  of both programs under Nsight Compute at locked clocks) and `perf_ab.py edge-type`.
- The `sssp_update` example takes the `cuda` backend (CTest `example.sssp_update.cuda`, label
  `gpu`, skipped without a device); the README describes the CUDA build, the CUDA presets and the
  local GPU gate; the sssp page documents the CUDA engines, the determinism level, the
  performance against both originals and a "paper vs fixed code" section.
- The M1b retrospective with the acceptance record and a re-estimate of the roadmap
  (`docs/developer/retrospectives/M1b.md`).

- M2a, CycleEnumeration-GPU graph pieces: `graph<V, E, unweighted>` ((int32, int32) and
  (int32, int64)) and `is_unweighted_v`; `batch_semantics::as_sets` with the preset
  `batch_semantics::set()` and `graph_properties::cycle_enum_compatible()` (Step 0 is the
  original's `prepare_batch()`, the apply its sorted-row `apply_batch()`; byte-equal CSR and
  normalized batches on committed fixtures and on the TUDataset graphs; ADR 0010 amendment);
  `edge_batch::insert_edge(u, v)` without weights.
- `io::read_edge_list` / `io::write_edge_list`: the original's parallel `from_chars` parser
  (TUDataset `*_A.txt`, comments, commas, signs, timestamps, Matrix Market of every symmetry),
  generalized with weight columns, `vertex_ids::as_is`, symmetrization, kept duplicates and self-
  loops and a thread cap (`docs/api/file_formats.md`).
- `generators::legacy::cycle_enum_batch()`: the original's `generate_batch()`, bit-exact, with the
  library-owned reproductions of libstdc++'s 64-bit `uniform_int_distribution` and `shuffle`.
- `dyng::testing`: `oracle_simple_cycles` (subset DP), `brute_force_simple_cycles` and
  `edge_set_after_batch` (the recount of a batch).
- Parity harness: the CycleEnumeration-GPU@0a976ad reference (`references.toml`, OpenMP and CUDA
  build as in its RESULTS.md), its exporter `export_cycle_enum`, the fixture script
  `parity/fixtures/cycle_enum/`, and the dataset digest test (CTest label `parity`).
- `cycle_count` on the sequential and OpenMP backends (M2a, `<dyng/cycle_count.hpp>`,
  `docs/algorithms/cycle_count.md`): exact k-bounded directed simple-cycle histograms; `compute()`
  is CycleEnumeration-GPU's sequential Johnson and OpenMP counter, `update()` its DynTruCy-style
  `update_static_histogram[_openmp]` (count(-) on G_t, one apply, count(+) on G_{t+1}, edge-id
  ownership), ported straight; every batch semantics accepted through Step 0 on G_t
  (`detail::compute_structural_change`). Histograms bit-identical to the original's on the
  committed fixtures (80 random cases, k = 2..7 and unbounded; fixture graphs and generated
  batches) and, with `--datasets`, on the TUDataset graphs (CTest label `parity`). The two recorded
  mutations are built into copies of the library and must fail the randomized suite (CTest
  `cycle_count.mutation.*`, label `mutation`). The host port of the original's pruned
  lower_bound search of its CUDA kernels (`dfs.hpp`) is tested for M2b.
- `io::write_histogram_csv(std::ostream&, counts)` (the original's `# cycle_size,
  num_of_cycles` ... `Total, N`; PLAN 5.7's name),
  `tools/compat/dyng-compat-cycle-enum` (the original `cycle-enum` CLI, byte-equal standard output
  on the committed CLI cases), the `cycle_count_update` example and
  `parity/timed_regions/cycle_count.toml`.
- `update_participant::reads_prepared_graph()`: `dyng::update` builds the in-edges (or the device
  copy) in the commit only for results that read them.
- Parity harness for `cycle_count` (M2a): the golden corpus of CycleEnumeration-GPU@0a976ad
  (`parity/export_goldens.py cycle_count`: 24 cases, histograms, update priors, deltas and
  generated batches, exported twice identically), its replay (`parity/compare.py cycle_count`, CTest
  `parity.cycle_count.cycle_enum_0a976ad`), the OpenMP A/B (`parity/perf_ab.py cycle_count run`)
  and `dyng-compat-cycle-enum --write-batch`. Results in `parity/results/M2a.md`: 72 of 72 replays
  byte-identical (sequential, OpenMP 4 and 56 threads); every OpenMP-56 gate met (at `1148d15`:
  static end to end 0.64-0.96x, update 25K+25K 0.76-0.96x of the original).
- M2a close-out: `docs/algorithms/cycle_count.md` completed (graph requirements, determinism,
  the performance table, 'Differences from the paper': exact k-bounded enumeration, not the
  paper's approximate kappa-truncated TruCy, and the 'Paper vs fixed code' table of
  CycleEnumeration-GPU's fixes); the README lists `cycle_count` on the CPU backends (CUDA: M2b);
  the M2a retrospective. The TruCy / DynTruCy paper is cited as submitted.
- M2a review fixes (`cycle_count`, graph, io, tests, harness): the searches keep their paths on
  explicit stacks (the default unbounded options no longer overflow the thread's stack; a
  300,000-vertex ring is counted and updated in the tests); histograms and engines are sized by
  min(k, max(n, 2)) and the update's per-thread counters grow with the cycles found, so an
  unbounded update costs what its searches cost (it was O(changes x n)); the ownership index is a
  flat table that allocates nothing in a steady-state update (I9); the OpenMP phase sizes its
  scratch in the region that uses it; `static_assert` on unsupported graph types; `as_sets`
  combinations that cannot apply a batch are rejected at graph construction; the stats and the
  unbounded cost per backend are documented; seed replay (`DYNG_TEST_SEED`, `DYNG_TEST_SEEDS`)
  in the randomized cycle_count suites; a third mutation test (`skip_workspace_resize`);
  `DYNG_STDLIB_ASSERTIONS` (`_GLIBCXX_ASSERTIONS` in Debug builds); the contamination monitor
  and isolation experiments of the cycle_count performance harness
  (`parity/contamination.py`, `--baseline-exe`, `parity/experiments/cycle_enum`); the parity
  certificate `parity/results/M2a.md` re-measured at `0679ed1` (72 of 72 replays; static end to
  end 0.53-0.93x, update 0.44-0.65x of the original, COLLAB k = 3 0.29x) with the improvements
  isolated in their own section.

### Fixed (M5 review)

- Result arrays exported from Python (`np.from_dlpack`, `np.asarray`, `to_numpy(copy=False)`,
  DLPack consumers) read freed memory after an update that grew the vertex set: the result's
  state now lives in reference-counted storage, every export holds a reference to the state it
  views, and an update copies the state first while an export is alive (copy-on-write). The
  staleness counter of `dyng.Array` moved into the native result, so `copy.copy(result)` can no
  longer bypass `StaleResultError`; `Result.__copy__` / `__deepcopy__` clone (ADR 0011 item 4).
- `EdgeBatch` cached converted id arrays but read same-dtype arrays live, so a refilled batch
  applied a mix of old and new values depending on the id dtype; it now converts its arrays at
  every use.
- Scalar arguments and option fields were coerced with `int()` / `bool()` (`sssp.compute(g, 1.9)`
  ran from vertex 1, `max_length=2.7` counted with bound 2); they are checked with
  `operator.index` semantics and raise `InvalidArgumentError` naming the argument.
- `dyng.profile()` and `Resources.copy_policy` changed the shared resources handle while other
  threads' calls read it; they now wait until no native call runs (a writer-preferring
  native-call lock, so the wait ends under load).
- Buffer-protocol inputs (`array.array('q')`, `memoryview`) now declare their element type for the
  dtype dispatch (int64 buffers no longer select int32 ids).
- `pickle` / `copy` of a used `EdgeBatch` failed; batches, `Resources` (rebuilt), `Array` (a NumPy
  copy) now pickle, and `Graph` / results explain the supported path (ADR 0011 item 13).
- The native module is chosen on first use instead of at `import dyng`, so `dyng.use_cpu_only()`
  works as PLAN 5.4 says and a plugin process never loads `dyng._core` (ADR 0011 item 14).
- `dyng cycle_count update` without `--max-length` ran an unbounded update that did not finish;
  it is a usage error, as in the original. sssp CLI errors name the command line's flags
  (`--random-weights`), `--source` is range-checked with a readable message, and the library's
  `dyng:` prefix is no longer doubled.
- The typed layer passes `mypy --strict` (checked in `ci/python.sh` and `python.yml`), and
  `dyng.update` has overloads for its stats types.
- Packaging: the wheel carries `THIRD_PARTY_LICENSES.txt` (nanobind, robin-map, the GCC runtime);
  the sdist leaves out the repository-only files (the CC-BY-SA-4.0 Code of Conduct, governance,
  tool configuration); the version uses `[[tool.dynamic-metadata]]` and scikit-build-core is
  bounded below 2; `release.yml` requires a canonical PEP 440 `VERSION` and derives the
  pre-release flag from it; `wheels.yml` builds the wheel from the sdist; the local toolchain
  pins the GCC 12 runtime; `release.md` sets `VERSION` to `X.Y.ZrcN` for a candidate.
- `ci/python.sh` no longer re-points an environment whose `dyng` is an editable install of another
  checkout (it uses a throwaway venv); `python.yml` builds the module with Clang 18 too;
  `DYNG_BUILD_DOCS=ON` builds the site (target `docs`).
- Documentation: the "add an algorithm" guide's bindings step, the CLI in the algorithm pages'
  mapping tables, the cycle-enum flag table, the MOSP-OpenMP history (the `-O3` baseline, the
  packed-format boundary fix), the CUDA message of the CPU wheel, and a getting-started link.

### Changed (M5 review)

- `dyng.Array.__dlpack__()` without `max_version >= (1, 0)` (the unversioned capsule, which
  cannot mark an export read-only) returns a copy; `copy=False` then raises `BufferError`.
- `dyng sssp compute` no longer has `--validate-inputs` (it only checks trees read with
  `update --init`).

### Fixed (M3 review)

- Budgets (invariant I9) count per calling thread and exclude the profiler's `sync_stages`
  synchronizations; an excess is logged in a Debug build and throws only under strict budgets
  (`DYNG_STRICT_BUDGETS=1`, armed by the conformance kit), so a correct update on another thread
  no longer fails and poisons its result. The half before the commit is measured too (cycle_count's
  CUDA budget: 4 host synchronizations).
- The framework chooses and checks the engine before the commit (an engine that does not exist or
  cannot run, or `fallback_recompute` without a recompute hook, leaves the graph and the result
  unchanged); `engine::automatic` picks a fused engine only where the problem's new
  `fused_available(ctx)` says it runs.
- `dyng::update()` / `update_each()` dispatch through `detail::participant_of<container_t>::run()`,
  so a later container (the hypergraph) needs no change to `update.hpp`; passing the owning batch
  or a non-container stops at a plain-English `static_assert`; `update_each()` rejects an empty
  list and a list in device memory (`invalid_argument_error`, before anything changes).
- `edge_batch::insert_edge` / `delete_edge` and `edge_list::add_edge` are strong under allocation
  failure (a failed call leaves the arrays as they were); `to_vector()` reports allocation failure
  as `out_of_memory_error`. Every mutating member of the container, batch and result classes states
  its guarantee (`@guarantee`, checked by `ci/doxygen_coverage.py`).
- `edge_list`'s fields are in the order of `edge_list_view` (`num_weights` last) (**breaking** for
  positional brace-initialization of `edge_list`; the API baseline is updated).
- Tooling: `regen.py` writes a space between long CODEOWNERS paths and their owners and rejects a
  manifest that omits a backend whose source exists; `new_algorithm.py` refuses names that cannot
  compile, removes only scaffolds (restoring their `planned.toml` entry) and writes the API page;
  `ci/docs.sh --update-api` updates the API baseline (the documented `&&` command never could).
- The kit: C4 is exercised on a fake two-engine algorithm, the registration rules have
  compile-fail tests, and C0 catches a manifest that drops a backend.
- Measurement: `parity/perf_ab.py memory` measures sssp's device memory against MOSP-CUDA
  (0.81-0.86x; cycle_count 1.000x); the gate suites, parity replays and the readings against
  pre-M3 dynG were repeated on the final code (`parity/results/M3.md` section 5, which also
  corrects the conclusions of section 4.2); `parity/experiments/sssp_stage_ab.py` compares two
  builds stage by stage.
- `sssp::compute()` / `update()` stop at a plain-English `static_assert` for an unsupported graph
  type (they failed to link), as cycle_count's do; `sssp::result`'s `distance_t` must be
  `std::int64_t` (the only width in 0.1). The scaffold's template follows the same rule.
- `to_string()` for `row_layout`, `row_order`, `multi_edges`, `batch_semantics::existing_insert` /
  `missing_delete` / `self_loop` and `cycle_count::search_method` / `cycle_mode` /
  `cuda_scheduler` / `cuda_work_items`.
- `profiler` recording is thread-safe (copies of a `resources` handle share it and may run
  concurrently); graphs, results and `dyng::update()` document their thread safety.
- The API snapshot also lists the `dyng::detail` contract of the public signatures
  (`update_traits`, `participant_of`, `stats_of`, the `*_supported_v` traits), every macro, and
  the includes of `<dyng/dyng.hpp>`; macros need `@ingroup`.
- Documentation: no references to the unpublished plan in the public headers; "deterministic per
  backend" defined for `update_stats::engine_used` and `sssp::stats::packed_parents`; the sssp
  synchronization text and the umbrella header's description corrected; sketch fixes (`mosp`
  stage names, `hyper_sssp`'s budget guarantee, `triad_count`, the hypergraph's `hyperedge_list`).

### Fixed (M3 acceptance)

- cycle_count's CUDA DD 25K + 25K update read 1.02x of the pre-M3 dynG: the redundant host copy
  of the normalized lists is gone (see "Changed"); 0.989-0.998x now.
- `-DDYNG_ALGORITHMS=<one algorithm>` (PLAN 9.4) links: sssp is added to every subset.
- Measurement (ADR 0024, accepted under delegation): a dynG-against-dynG A/B cycles its rounds
  through heap layouts (`perf_ab.py run --layouts N`, `cycle_count_perf.py run --layouts N`),
  because the same two builds read 0.99x or 1.10x of each other depending on the length of the
  `--timing` file name; `parity/ab_modes.py` reads bimodal regions mode by mode with a bootstrap
  interval; `parity/experiments/cycle_count_stage_ab.py` and `sssp_layout_scan.py`. Every suite
  re-measured on the final code (`parity/results/M3.md` section 6): the gates against the
  originals hold, and no gated region regresses by more than 2 % against `019ef13`.

### Changed

- Exception guarantees (the 0.1 API review): `compute()`, `update()`, `from_arrays()`,
  `graph::apply()`, `dyng::update()` and `dyng::update_each()` state them in a new `@guarantee`
  paragraph (strong before the commit; basic after it, with the result poisoned; `graph::apply()`
  strong), and `ci/doxygen_coverage.py` requires it. The checker takes the algorithm namespaces
  from the manifests (it checked only `sssp` before). `dyng::update()`, `update_each()` and
  `dyng::algorithms()` report host allocation failures as `out_of_memory_error` instead of
  letting `std::bad_alloc` leave the library; `sssp::compute()` / `update()` document
  `cuda_error`.
- The header self-containment targets (`cpp/tests/api`) also fail when a public header includes a
  CUDA, CUB, Thrust or libcu++ header.
- Budgets (I9): a run that grows a reusable array on purpose (`detail::note_reservation()`: a new
  workspace, a grown scratch buffer or per-thread list, a grown result) is a reserving run whose
  allocations are reported, not failed; the graph's own materializations inside an update (its
  device copy uploaded on first use, `detail::container_scope`) are container work and never held
  against a problem's budget; `run_update()` records the commit's counts. sssp and cycle_count
  declare their budgets (no allocation once reserved; 0 host syncs on the host backends, 1 and 2
  on CUDA), checked by the update enactor in `DYNG_DEBUG_BUDGETS` builds.
- OpenMP `sssp`: `list_gather` keeps its offsets in an inline array (up to 256 threads) instead of
  a `std::vector` per gather, so the near-far rounds allocate nothing (found by C8); the same
  trees.
- A build of a subset of the algorithms (`-DDYNG_ALGORITHMS=...`) configures: the suites of an
  algorithm are built with it, and the examples and compat tools (sssp and cycle_count) only when
  both are built. sssp is part of every build (the library's MOSP batch generator and the shared
  suites call it): a list without it gets it added, so `-DDYNG_ALGORITHMS=<name>` builds and links
  (it failed to link before); `ci/scaffold_check.sh` builds every target of such a subset.
- CUDA `cycle_count` under set semantics reads the framework's normalized change lists (their
  device copy and their lengths) instead of copying them into its workspace first; the same
  histograms and stats (the DD 25K + 25K update: about 0.07 ms less of 3.4 ms).
- `.github/workflows/welcome.yml` no longer greets owners, organization members and repository
  collaborators (the event's `author_association`); first-time outside contributors are greeted
  as before.
- The default edge offset type is `int32`: `dyng::graph<>` is `graph<int32, int32, int32>`, and
  building or updating a graph past 2^31 - 1 edges throws `capacity_error` naming the int64
  instantiation (ADR 0009, from the `edge_t` benchmark).
- OpenMP `sssp`: a near-far round passes three barriers instead of six
  (`list_gather::gather_pair()`, `nowait` loops) and the per-thread lists live in the workspace on
  their own cache lines; the same trees, 0.63-0.89x of MOSP-OpenMP's time on the 10K local
  batches.
- The Doxygen convention check (`ci/doxygen_coverage.py`) requires `@sync` or `@async` on every
  CUDA-capable public function: those taking `resources`, a `stream_ref` or a
  `memory_resource_ref`, and the stream-ordered members of `buffer` and `resources`; the memory
  resources, `buffer` and `resources` document how they order their work.
- Placement (PLAN 4.6 rule 5): a graph belongs to the backend of the resources that built it;
  sssp on resources of the other kind (host backends versus cuda) throws `invalid_argument_error`
  instead of copying, and `graph::clone(res)` / `result::clone(res)` move a graph or a result.
  `graph::from_*`, `clone`, `reserve`, `apply`, `to_csr` and `check_integrity` accept CUDA
  resources; `graph::space()` is `device` for a CUDA graph.
- `dyng::testing::check_sssp_tree(g, r)` copies a device result to the host first.
- The host-side work of a call with CUDA resources uses the OpenMP threads; since the M1b review
  their count is fixed when the handle is created (`resources::cuda(device, stream,
  host_threads)`) and reported by `num_threads()`.
- The CPU presets (`dev`, `release`, `relwithdebinfo`, `parity`) pin `DYNG_ENABLE_CUDA=OFF`; a
  build without a preset enables CUDA when a CUDA compiler is found. With CUDA built and a device
  visible, `default_backend()` is `cuda`.
- `resources::set_memory_resource()` on a CUDA handle accepts device and managed resources; the
  host sanitizer flags apply to host code only.
- The host transposition runs in parallel on the OpenMP backend (same, deterministic result).
- `io::read_csr_triplet` reads its three files concurrently (same result and the same first
  error as reading them one after the other); `dyng-compat-mosp` reads and writes its files
  concurrently, like the original driver. dynG now links `Threads::Threads` privately.
- `sssp`: the sequential backend uses the tie rule of `sospUpdateCpu` (an improved vertex offers
  its (distance, id) pair) instead of the re-scan of `sequentialSOSPUpdate`, so both backends
  return the same tree also from non-canonical input trees (ADR 0006); the documentation states
  the tie rule.
- A failed (poisoned) `sssp::result` now throws on every use, not only on `update()`; both
  backends report a parent cycle of an imported tree.
- Results also record the identity of the graph state, so a result used with another graph (or
  a reassigned graph variable) throws `stale_result_error` (ADR 0006).
- `resources` moves share the handle like copies; the log sink is called without the logging
  lock; host allocation failures leave the library as `out_of_memory_error`.
- The OpenMP performance A/B loads its regions from `parity/timed_regions/sssp.toml`, counts the
  moved first-touch cost, maps the original's `prepare` and end-to-end timers completely, and
  works under `flock(1)`; `build_reference.sh` keeps its logs and fingerprints each build.
- Tests run with `OMP_WAIT_POLICY=PASSIVE`; the lint workflow uses Doxygen 1.18.0, as
  `environment.yml` (now pinned) does.
- `sssp::result` no longer owns a workspace; the M1a pre-touch of every result's frontier lists
  is removed, and the A/B compares every objective as measured (ADR 0015).
- Graphs build their in-edges on first use (not at construction, not in `graph::apply()`);
  `dyng::update()` builds them inside the commit, for the updated graph only.
- OpenMP backend: `graph::apply()` assembles the new CSR in parallel, `from_csr()` checks in
  parallel, `sssp::result::from_arrays()` imports and validates in parallel (same results and
  messages).
- I/O: files are read with one allocation and one read; distance and tree files are parsed in
  one pass (the strict reader reports errors); CSR weights are parsed straight into their
  objective-major columns.

### Fixed (M1b review)

- CUDA `sssp`: the fused kernel counts `invalidated` per thread instead of reading the
  candidate-list counter while other threads append insertion heads to it (a data race inherited
  from MOSP-CUDA@e220ee2; the trees were never affected). `ci/gpu_local.sh` runs the CUDA sssp
  suite under `compute-sanitizer --tool synccheck`.
- CUDA `sssp`: vertex growth releases the old result arrays on the updating stream, after the
  copies that read them (`buffer::set_stream()`).
- Pooled workspaces are ordered across streams (a CUDA event per workspace): copies of a
  default-stream handle used on two threads run on two per-thread streams, which the docs now
  state for `resources`, `stream_ref` and `buffer`; `release_workspaces()` waits for the last use.
- Inputs in device memory are accepted everywhere and copied to the host once under
  `copy_policy` (`error` refuses, `warn` logs, `allow` logs at debug and acts as `warn` while a
  profiler is attached): batches of `graph::apply()`, `dyng::update()`, `update_each()` and
  `sssp::update()`, graph builds and `sssp::result::from_arrays()`. A batch with only some arrays
  in device memory no longer crashes `sssp::update()`.
- `copy(res, src, dst)` and `update_each()` deduce their types from mutable views;
  `to_space()` allocates from the resource that matches the requested space;
  `graph::to_backend(res)` exists and the placement errors name it.
- `DYNG_WITH_NVTX`: `profiler_options::nvtx` emits NVTX ranges per stage.
- Documentation: `@sync` of `sssp::compute()` / `update()` names the graph upload; `@sync` and
  `@throws` on the synchronous allocation members and `buffer`; the Doxygen check covers them.
- Build and CI: `native` CUDA architectures without a visible GPU fall back to the release list
  with a warning; `cuda-build.yml` builds CUDA 13.4.1 (the latest 13.x) and caches ccache; a GCC
  13/14 `-Werror` false positive in a test is gone; the `compat_mosp` cuda tests skip without a
  device (`ci/gpu_local.sh` requires one); `ci/check.sh` and the clang-tidy steps run under the
  shared perf lock.
- Harness: `perf_ab.py` records a contamination monitor per round (foreign CPU load, run queue,
  GPU P-state and clocks, foreign GPU processes) and repeats contaminated rounds; the CUDA
  `apply` region includes the host tree copies (`sssp.import`).

### Fixed (M1b acceptance)

- Harness: `perf_ab.py run --backend cuda` locks the GPU's clocks for the whole A/B without root
  (`--lock-clocks boost|base|none`, default `boost`; Nsight Compute holds the lock through the idle
  helper `parity/clock_lock/clock_holder.cu`, neither timed program is profiled), checks every
  busy GPU sample against the locked clocks and resets the clocks at the end (ADR 0018, rule 4).
- Build: `DYNG_WITH_NVTX` defaults to ON only when the toolkit's `nvtx3/nvToolsExt.h` exists
  (`CUDA::nvtx3` alone does not guarantee the headers); `cuda-build.yml` documents the
  `LD_LIBRARY_PATH` a conda-forge toolkit needs for the local equivalent.

### Integration 1 (M1b and M4 merged)

- `main` merges the project infrastructure (M4, branch `m4-infra`) with a merge commit on top of
  the CUDA backend (M1b). `ci/check.sh` keeps M4's steps (clang-tidy naming, harness tests,
  `DYNG_CHECK_ONLY`, the Sphinx site) under M1b's shared-lock wrapper; CONTRIBUTING.md and the
  README describe the CUDA presets and the local GPU gate `ci/gpu_local.sh`.
- `cuda-build.yml` meets the workflow checks of M4 (a job timeout, no persisted credentials).
- Documentation site: the M1b ADRs (0003, 0009, 0015-0018) and retrospective, an API page for
  the `generators` group, the CUDA backend on the install, backend, getting-started, roadmap and
  tutorial pages, the CUDA parity replay and links to the M1a and M1b parity certificates;
  the algorithm tables list `sssp` on sequential, OpenMP and CUDA.
- `examples/cpp/first_update.cpp` uses `graph<>` and runs on the `cuda` backend
  (`example.first_update.cuda`, label `gpu`).
- Credit and citations (the author's facts of 2026-09-27): the ESCHER IPDPS 2026 title and
  authors; TruCy cited as a submitted manuscript; S M Ferdous's affiliation (PNNL); the
  placeholder-identity commits of the originals credited to S M Shovan; no funding line yet.
- Merge policy (ADR 0019): milestone and integration pull requests are merged with merge
  commits, external contributions squash-merged, rebase merging disabled. `GOVERNANCE.md` and
  the repository settings guide record it, mark the settings already applied, and list the
  `cuda-build` and `docs` jobs among the required checks of the `main` ruleset (pending the
  first pull request).
- The INT1 retrospective (`docs/developer/retrospectives/INT1.md`).

### M2b: M2a merged (branch `m2b-cycle-cuda`)

- The CPU `cycle_count` of M2a (branch `m2-cycle`) is merged with a merge commit on top of
  INT1. The graph keeps both sides: M1b's `int32` default `edge_t` with checked construction
  (ADR 0009), the device graph and the copy-policy staging, and M2a's `unweighted`
  instantiations, `graph_properties::cycle_enum_compatible()` / `batch_semantics::set()` and
  the sorted-row set apply. The host batch checks are one function (`validate_batch_shape`) for
  both applies; `checked_edge_count` exists once.
- Documentation site: an API page for the `cycle_count` group; the algorithm tables, landing
  page, README, roadmap and short plan list `cycle_count` as working on sequential and OpenMP,
  CUDA in progress. M2a added no ADR (it amended ADR 0010), so no ADR was renumbered.
- The `main` ruleset exists since pull request #1 (17 required checks, strict; the Repository
  admin role bypasses only through pull requests); `DCO` becomes required once the author's
  organization membership is public. `GOVERNANCE.md`, CONTRIBUTING.md and the repository
  settings guide say so.
- Re-verification after the merge: `parity/results/M2b.md`, section "Merge re-verification".

### M2b: `cycle_count` on CUDA (branch `m2b-cycle-cuda`)

- The CUDA backend of `cycle_count` (`<dyng/cycle_count.hpp>`): the straight port of
  CycleEnumeration-GPU@0a976ad's static counters (the work queue with root, edge and implicit
  two-hop prefix items and its automatic choice; the naive one-thread-per-root counter) and of its
  update (the delete phase on G_t, the insert phase on G_{t+1}: `mark_owners_kernel`,
  `item_counts_kernel`, `count_owned_cycles_kernel`), behind `compute()` and `update()`. New
  options: `cuda_engine` (`automatic` / `fused`; `operators` throws `not_supported_error`),
  `scheduler` (`cuda_scheduler::work_queue` / `naive`) and `work_items` (`cuda_work_items`). The
  effective bound is limited to 64 on cuda (`invalid_argument_error`). Histograms are
  bit-identical to the original's CUDA backend on the fixtures and the TUDataset corpus.
- The resident device graph (ADR 0020): a CUDA graph under `batch_semantics::as_sets` without
  weight columns is updated on the device (`build_next_rows_kernel` and friends,
  `graph/apply_set_device.cu`) and stays there across batches; its host CSR is downloaded when
  something reads it. The device in-edges are built on first use. Under set semantics a batch is
  normalized once per update (`<algo>.normalize`), for every result and the commit (the host
  commit no longer normalizes again).
- `dyng-compat-cycle-enum --backend cuda` with the original's CUDA flags (`--cuda-device`,
  `--cuda-scheduler`, `--cuda-work-items`, `--report-timing` from CUDA events) and `--scope
  original|resident`, `--edge-type`; the CLI test `compat_cycle_enum.cli.cuda` (label `gpu`).
- Parity harness: the exporter's CUDA build (`export_cycle_enum_cuda`) and the CUDA fixtures
  (`cases/*.cuda`, `counts/*.cuda`, `cli/cuda*`, each checked equal to the original's sequential
  backend when written); the golden set `cycle_count_cuda` (24 cases from `cycle-enum --backend
  cuda`, including the 100K + 100K updates of DD and GitHub and the COLLAB update), replayed by
  `compare.py cycle_count --configs cuda,cuda:resident,cuda:int64` (CTest
  `parity.cycle_count.cycle_enum_cuda_0a976ad`); `parity/cycle_count_perf.py run --backend cuda`
  (the regions of `[reference.cycle_enum_cuda]` in both scopes, the clocks locked for the whole
  A/B, ADR 0018) and `kernels` (register counts of both sides).
- Tests: the shared `cycle_count` suites on cuda (`dyng_cycle_count_cuda_tests`, label `gpu`)
  and the CUDA cases (bounds 2..64 and beyond, schedulers, cross-backend chains of batches, the
  device apply against the host apply, the lazy host copy, the host-commit fallbacks, corner
  cases). `ci/gpu_local.sh` replays the CUDA golden set and runs synccheck and racecheck on the
  CUDA `cycle_count` suite.

### M2b: the CUDA gates of `cycle_count` (branch `m2b-cycle-cuda`)

- The CUDA gate of `cycle_count` against CycleEnumeration-GPU@0a976ad's CUDA backend
  (`parity/results/M2b.md` sections 4-6): ten cases (static k = 4 on DD, GitHub, Twitch, k = 3
  on COLLAB; the updates 25K+25K on all four and DD 50K+50K, 100K+100K) in both scopes, clocks
  locked; every gated reading within its gate (the COLLAB update at the base lock, now pending
  ADR 0021, see the review entries below). Default-clock readings, the registers, stack and
  occupancy of all 33 kernels (equal to the original's) and the peak device memory of every case
  (equal to the original's) are recorded.
- Faster Step 0 under set semantics: `std::sort` by (source, target, position) instead of
  `std::stable_sort`, and no sort for lists already in order (every backend; the DD 100K+100K
  CUDA update 12.2 -> 8.0 ms). Results unchanged. (Replaced by the review's bucket sort: the skip
  helped only the gate's already sorted batches.)
- Less device memory in the CUDA `cycle_count` update: the static-count work items are returned
  when an update begins, and the deletion marks of G_t are computed once per update and shared
  by the cycle_count delete phase and the device apply (ADR 0020, point 6).
- `dyng-compat-cycle-enum --scope resident` (count task) makes the graph resident with a 2-cycle
  count, so the timed kernel does not follow a full count (which runs it 5-12 % slower even at
  locked clocks).
- Harness: `parity/cycle_count_perf.py kernels` covers every kernel with its sm_86 occupancy and
  pairs both sides; `run --backend cuda` records an unmeasurable case (more rejected rounds than
  `--runs`) as incomplete and continues, writes the JSON after every case, keeps the original's
  monitor window apart from the port's and summarizes the GPU clocks per side; `memory` records
  the peak device memory of both sides per case (Nsight Systems' memory trace).
- A case whose GPU cannot hold the boost lock under its power cap (the COLLAB update, whose prior
  counts for 6.5 s) is read at the base lock, applied equally to both sides: first written as an
  update of the accepted ADR 0018, now the Proposed ADR 0021, pending the author.

### M2b: close-out (branch `m2b-cycle-cuda`)

- `cycle_count` works on the sequential, OpenMP and CUDA backends (M2 done). The algorithm page
  describes the CUDA work items of every scheduler and update phase, determinism on cuda (the
  claim order of the work queue changes, the sums do not; the device sums are not checked for
  overflow, as in the original) and the default-clock readings next to the locked-clock gate
  table.
- The `cycle_count` manifest lists the cuda backend and the original's CUDA sources.
- The short plan's estimate is re-estimated after M2 (0.1.0 in about 1-2 weeks of focused work,
  3-5 weeks of calendar time); the M2b retrospective has the milestone summary, the M2 summary,
  the acceptance record and the final verification from a fresh clone.
- `parity/results/M2b.md` section 7: the three gate scripts pass in a fresh clone, and the sssp
  CUDA and OpenMP gates and the cycle_count OpenMP update gate hold on the final code.

### M2b: review fixes (branch `m2b-cycle-cuda`)

- Fixed: under `batch_semantics::set()` with `self_loop::keep` the `cycle_count` update counted
  a spurious 2-cycle through every self-loop of a batch on every backend (the normalized lists
  keep self-loops as change edges); the phases now skip them.
- Fixed: a chain of CUDA updates on a resident graph downloaded all of G_t in every update after
  the first (Step 0 read the stale host copy). Step 0 of such a graph now tests the membership of
  the changes in G_t on the device (`graph_access::normalize`); the graph is downloaded only when
  something reads it on the host, on the stream of the resources that built the state (no longer
  the legacy default stream), without copying stale content into a regrown vector.
- Changed: Step 0's sort under set semantics is a bucket sort (stable, no shortcut for sorted
  lists; 0.7x of the original's `std::sort` on sorted input, 0.3x on shuffled input). The
  sortedness skip of the gate campaign is gone.
- Added: `cycle_count::result::set_options()` for the tunables `cuda_engine`, `scheduler` and
  `work_items` (`max_length`, `method` and `mode` stay fixed at `compute()`).
- Changed: `bound()` and `get_options()` of a poisoned `cycle_count` result throw
  `stale_result_error`; `compute()` on cuda counts an edgeless graph of any size (the zero
  histogram before the 64-vertex check, as the original); the 32-bit change-id limit of the cuda
  update raises `capacity_error`; the documentation says that the device sums wrap at 2^64 on
  cuda, as the original's.
- Fixed: staging errors of `cycle_count::update` name it instead of `dyng::update`.
- The device set apply keeps its scratch in the pooled normalized batch: a steady CUDA update
  allocates only the arrays of G_{t+1}.
- Tests: the recorded mutations in the CUDA kernels (`cycle_count.mutation.cuda.*`, label
  `gpu`), the original's large-graph device test (300,000 vertices), device Step 0 against host
  Step 0, chained updates, steady-state allocations, kept self-loops on all backends; the
  host-only kernel tests no longer run in the gpu executable.
- Harness: `dyng-compat-cycle-enum --chain n`; the CUDA gate records chained updates on the
  resident graph (`update_chain_steady`, `update_chain_worst`), dynG's CUDA-event time of the
  short updates (`update_device`) and the CPU-load threshold.
- Docs: the public graph documentation covers the device merge; ADR 0020 is Proposed (it had
  been marked Accepted without an acceptance) with the review's amendments; the M2b clock rule
  that had been appended to the accepted ADR 0018 is ADR 0021 (Proposed), pending the author;
  README lists the M2b certificate.

### M2b: acceptance fixes (branch `m2b-cycle-cuda`)

- Docs: the status pages (the developer plan, the roadmap, README, the algorithm page) no longer
  say that every CUDA gate is met; they name the COLLAB update, read at the base clock lock and
  pending the author's decision on ADR 0021, which the developer plan also lists as an open
  decision. The certificate and the retrospective point at ADR 0021 instead of the removed
  "ADR 0018 update".

## [0.0.1] - 2026-09-27

### Added

- The PyPI name reservation: a pure-Python placeholder package `dyng` 0.0.1
  (`tools/name_reservation/`), published to PyPI and TestPyPI by `release.yml` through Trusted
  Publishing from tag `v0.0.1` (commit `15a6051`). It contains no library code.

[Unreleased]: https://github.com/dyng-dev/dyng/compare/v0.2.0rc1...main
[0.2.0rc1]: https://github.com/dyng-dev/dyng/compare/v0.1.0...v0.2.0rc1
[0.1.0]: https://github.com/dyng-dev/dyng/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dyng-dev/dyng/tree/v0.0.1
