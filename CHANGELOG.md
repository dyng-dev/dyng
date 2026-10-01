# Changelog

All notable changes to dynG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 0.1.0.
Before 0.1.0 anything may change.

## [Unreleased]

## [0.1.0] - 2026-10-01

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

[Unreleased]: https://github.com/dyng-dev/dyng/compare/v0.1.0...main
[0.1.0]: https://github.com/dyng-dev/dyng/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dyng-dev/dyng/tree/v0.0.1
