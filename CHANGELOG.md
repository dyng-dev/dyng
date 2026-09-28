# Changelog

All notable changes to dynG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 0.1.0.
Before 0.1.0 anything may change.

## [Unreleased]

### Added

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

### Changed

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

## [0.0.1] - 2026-09-27

### Added

- The PyPI name reservation: a pure-Python placeholder package `dyng` 0.0.1
  (`tools/name_reservation/`), published to PyPI and TestPyPI by `release.yml` through Trusted
  Publishing from tag `v0.0.1` (commit `15a6051`). It contains no library code.

[Unreleased]: https://github.com/dyng-dev/dyng/compare/v0.0.1...main
[0.0.1]: https://github.com/dyng-dev/dyng/tree/v0.0.1
