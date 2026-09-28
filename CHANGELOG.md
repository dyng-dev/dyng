# Changelog

All notable changes to dynG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 0.1.0.
Before 0.1.0 anything may change.

## [Unreleased]

### Added

- Repository bootstrap: Apache-2.0 license, NOTICE, CITATION.cff, AUTHORS, governance and
  contribution guides, ADRs 0001 (name), 0002 (license and credit), 0004 (naming) and 0014
  (approval checkpoints).
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

### Changed

- Placement (PLAN 4.6 rule 5): a graph belongs to the backend of the resources that built it;
  sssp on resources of the other kind (host backends versus cuda) throws `invalid_argument_error`
  instead of copying, and `graph::clone(res)` / `result::clone(res)` move a graph or a result.
  `graph::from_*`, `clone`, `reserve`, `apply`, `to_csr` and `check_integrity` accept CUDA
  resources; `graph::space()` is `device` for a CUDA graph.
- `dyng::testing::check_sssp_tree(g, r)` copies a device result to the host first.
- The host-side work of a call with CUDA resources uses the OpenMP threads.

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

## [0.0.1] - 2026-09-27

### Added

- The PyPI name reservation: a pure-Python placeholder package `dyng` 0.0.1
  (`tools/name_reservation/`), published to PyPI and TestPyPI by `release.yml` through Trusted
  Publishing from tag `v0.0.1` (commit `15a6051`). It contains no library code.

[Unreleased]: https://github.com/dyng-dev/dyng/compare/v0.0.1...main
[0.0.1]: https://github.com/dyng-dev/dyng/tree/v0.0.1
