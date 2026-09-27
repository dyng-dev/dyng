# Retrospective: M1a (walking skeleton, CPU `sssp`)

Status: **in progress.** Each implementation step appends its section; the final step adds the
milestone summary and the re-estimate of the roadmap.

## Step 1: repository bootstrap and M0 drafts (2026-09-27)

### Done

- Local repository `/home/sskg8/Projects/dyng` (branch `main`, no remote), local git identity,
  configuration files (`.clang-format`, `.clang-tidy`, `.editorconfig`, `.gitattributes`,
  `.pre-commit-config.yaml`, `.codespellrc`, `REUSE.toml`).
- `environment.yml` (conda env `dyng-dev`: Python 3.12, CMake 4.4, Ninja, clang-format and
  clang-tidy 22.1.8, Doxygen, pre-commit, reuse, pytest, numpy, build, twine) and
  `scripts/dev_env.sh`. The env was created on the development machine and the conda cache
  cleaned.
- CMake build (presets `dev`, `release`, `relwithdebinfo`, `cpu-only`, `asan`, `tsan`,
  `parity`), install/export (`find_package(dyng)` verified with a shared and a static build).
- Minimal core with unit tests (62 test cases; green on `dev`, `cpu-only`, `asan`, `tsan`, and
  with OpenMP off); the header self-containment check; the Doxygen check with warnings as
  errors.
- Community and legal drafts, ADRs 0001, 0002, 0004, 0014; the `dyng` 0.0.1 name-reservation
  package and `release.yml`; `ci/check.sh`, `lint.yml`, `cpu.yml`, `dependabot.yml`.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| `LICENSE` at the root | also `LICENSES/Apache-2.0.txt` | `reuse lint` requires the license text under `LICENSES/`; the root copy stays for GitHub and humans |
| `docs/Doxyfile.in` (configured by CMake) | `docs/Doxyfile`, driven by `ci/docs.sh` through environment variables | there is no `docs` CMake target before M5; the check is useful now |
| one `dyng_tests` executable | one GoogleTest executable per module (`dyng_core_tests`, ...) via `dyng_add_test()` | keeps rebuilds small and labels per module; CTest sees the same test cases |
| `DYNG_ENABLE_CUDA` ON if a CUDA compiler is found | OFF by default; ON fails at configure time with a clear message | the CUDA backend arrives in M1b; the option exists so presets and scripts keep their shape |
| `dyng::log()` | `dyng::log_message()` | an unqualified `log()` inside `namespace dyng` would hide the math function (ADR 0004) |
| `version_info{major, minor, patch}` | `major_version`, `minor_version`, `patch_version` | glibc may define macros `major` / `minor` (ADR 0004) |
| `resources::device()` "device 0" | -1 for host backends | a host backend has no device; documented |
| `cmake/CPM.cmake` | a small bootstrap that downloads the pinned CPM release with a SHA-256 check | the plan forbids vendoring third-party code |
| `VERSION` | `0.1.0.dev0` (PEP 440); CMake uses the numeric `0.1.0` | the next release is 0.1.0; the name-reservation package has its own version 0.0.1 |
| gersemi (CMake formatter) in pre-commit | not yet | not required by M1a ("a minimal pre-commit"); add with the M4 infrastructure |
| `dyng::testing` target | not yet created; `install.cmake` exports it once it exists | created with the first oracle (`testing::dijkstra`) in the `sssp` step |

### Notes for the next steps

- Add library code as a module with `dyng_add_module(NAME <module> SOURCES ...)` in
  `cpp/CMakeLists.txt`; tests with `dyng_add_test(NAME ... SOURCES ... LABELS cpu <module>)`.
- Stage timing: `dyng::scoped_stage stage(res, "sssp.loop");` (validated `<algo>.<hook>` names;
  CSV output is compatible with the MOSP `--timing` CSV).
- Preconditions: `DYNG_EXPECTS(cond, "message ", value, ...)` (at least one message argument).
- `ci/check.sh` is the gate; `DYNG_CHECK_SKIP="precommit"` skips steps that need the network.
- The `parity` preset sets `CMAKE_CXX_FLAGS_RELEASE=-O3` (MOSP-OpenMP's Makefile uses
  `-std=c++17 -Wall -Wextra -O3 -fopenmp`).

### Open items for the author

- Confirm who made the placeholder-identity commits in ESCHER-GPU, LabelPropagation-CUDA and
  MOSP-CUDA (credit in `AUTHORS.md`).
- Confirm funding acknowledgements, if any grant requires one (README "Acknowledgements").
- The ESCHER IPDPS 2026 title differs between the thesis publication list ("ESCHER: An
  Efficient and Scalable GPU Data Structure for Dynamic Hypergraph Triad Counting", used here)
  and the reference list of the TruCy manuscript ("ESCHER: Efficient and Scalable Hypergraph
  Evolution Representation with Application to Triad Counting", with S. Bhowmick as co-author);
  confirm the published title and author list.
- The TruCy/DynTruCy IEEE TC paper is recorded as submitted (2026); update when accepted.

### Process note

- Commit `dc56e9f` ("test: skip the impossible-allocation test under sanitizers") also contains
  `.github/dependabot.yml`, the `name-reservation` job of `lint.yml` and the first version of
  this retrospective, because they were staged together by mistake. History is not rewritten;
  this note records it.

## Step 2: graph container, batches and I/O (2026-09-27)

### Done

- `graph<V,E,W>` (pimpl, move-only, host storage behind `detail::graph_impl` so the device
  layout of M1b can be added without changing the public class): compact rows, objective-major
  weight columns (`weights[k * m + e]`), stored in-edges (the transposition of MOSP's
  `transposeCsrGraph()`, deterministic order), version counter, `view()` (`graph_view` with
  `out` / `in` as `csr_view`), `clone`, `reserve`, `to_csr`, `check_integrity`.
- `graph_properties` with `row_order`, `multi_edges`, `batch_semantics` and
  `mosp_compatible()`; `apply_summary`; `csr` / `csr_view`, `edge_list` / `edge_list_view`,
  `edge_batch` / `edge_batch_view`. ADR 0010 (draft) records the semantics.
- `graph::apply` = MOSP's `applyChangeBatch()` ported straight (a new CSR per batch; provenance
  header in `cpp/src/graph/apply_host.cpp`), with the batch_semantics switches around the
  unchanged core. `detail::graph_access::apply(res, g, batch, &delta)` also returns
  `detail::apply_delta`: the effective insertions and deletions and one weight-increase byte per
  (insertion, objective), i.e. MOSP's `weightIncreaseMask` without the K <= 32 limit.
- `io`: `read/write_csr_triplet`, `read/write_legacy_batch` (insert.txt / delete.txt),
  `read/write_distances`, `read/write_parents`, `read/write_matrix_market` with seeded random
  weights. Private `util/parser` (strict tokens, `io_error` with path:line:column),
  `util/text_writer` (MOSP's TextWriter), `util/rng` (the libstdc++ >= 11
  `uniform_int_distribution` algorithm reproduced, so `mtx2csr` weights do not depend on the
  standard library). `docs/api/file_formats.md` describes the formats.
- Tests (59 new cases; 121 in total): construction and semantics, a randomized comparison of
  `apply` with an independent row-list model (300 graphs x 4 batches x 3 type combinations, all
  semantics switches except the error policies, which have their own tests), the MOSP
  input-validation cases re-expressed, format and round-trip tests, and byte parity with
  MOSP-OpenMP@c352151 on 27 apply fixtures (updated CSR, transposed CSR and the weight-increase
  flags; plus `updateGraphCSR` as an edge set on the 10 generateTestCases cases) and 3
  `mospPrep mtx2csr` fixtures. The fixture tests were checked to fail on a mutated expected file.
- The fixtures (345 KB) are produced by `parity/fixtures/graph_io/make_graph_io_fixtures.sh`
  from a `git archive` of the pinned original under `$DYNG_SCRATCH/runs/graph-io-fixtures`; the
  script is deterministic (a second run leaves `git status` clean).
- One-off check on roadNet-CA (K = 3, 50K batch `changes_50000_50`): the updated CSR written by
  dynG is byte-identical to the original's `applyChangeBatch()` + `writeCsrGraph()`. Informal
  timings (not under the perf lock, cpu-only preset, one thread): dynG apply 58 ms + transpose
  59 ms vs the original's applyChangeBatch 67 ms + transposeCsrGraph 56 ms; reading the text CSR
  538 ms vs 394 ms (the original parses the three files concurrently).
- Green: `ci/check.sh` (cpu-only, dev, clang-format, REUSE, Doxygen, pre-commit), asan and tsan.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| `batch_semantics` struct default `on_self_loop = drop` | `keep`; `batch_semantics{}` == `upsert_last_wins()` | MOSP keeps self-loops; two different "defaults" would be a trap (ADR 0010) |
| `apply_summary` fields of Section 5.2 | plus `ignored_insertions` (appended) | `existing_insert::ignore` needs a counter |
| `graph_properties::num_weights` | `from_edges` / `from_csr` take K from their input and store it | avoids a mandatory, redundant setting for every multi-objective graph |
| presets `set()`, `net_effect()`, `cycle_enum_compatible()`; `edge_batch::normalized()`, vertex operations; `graph::with_capacity`, `to_backend` | not yet (vertex operations throw `not_supported_error`) | they belong to M2 (`cycle_count`), 0.3 (DynLP) and M1b (device); declaring them now would mean stubs with guessed semantics |
| `edge_list` weights layout (unspecified) | edge-major (file layout), like `edge_batch_view::insert_weights`; `csr` and the graph are objective-major | the text formats are edge-major; the engines read one column at a time |
| readers "re-express" MOSP's validation | same accept/reject decisions, plus stricter tokens: `12abc`, `1.5`, `+1` are errors; extra tokens on batch lines and delete lines with < 2 integers are errors (MOSP ignored them); `csr_triplet_options::num_weights`, when given, must match the file (MOSP ignores `-k` for graphs with edges); Matrix Market files must be well formed (mospPrep silently skipped bad or out-of-range entries and stopped early at EOF) | strict parsing was the goal; on well-formed files (everything the original tools write) the results are identical |
| Matrix Market `hermitian` | mirrored like `symmetric` | the Matrix Market specification; mospPrep treated it as general (no benchmark graph is hermitian) |
| `parity/` layout | added `parity/fixtures/graph_io/` (exporter + script + README) | the fixtures were needed before `build_reference.sh` exists; the harness step can fold the script in |
| `read_edge_list`, `read_legacy_mtx`, `.dgt`, graph cache | not in this step | not needed by `sssp` in M1a |

### Notes for the next steps

- `sssp::update` should call `detail::graph_access::apply(res, g, batch, &delta)` (private
  header `cpp/src/graph/graph_impl.hpp`); per objective k the MOSP "changed" edges are
  `delta.delete_*` plus the insertions with `delta.weight_increased[i * K + k] != 0`, and the
  insert heads are `delta.insert_dst`. `delta.delete_*` lists every requested in-range deletion,
  matched or not, exactly like MOSP's `changedFrom/To`, so `invalidated` stays comparable.
- MOSP's `maxWeight[k]` is the largest weight of the ORIGINAL graph and of the batch insertions,
  and `weightSum[k]` (for `defaultDelta`) is over the ORIGINAL graph: compute both from
  `g.view()` BEFORE applying the batch.
- `g.view().out.weight_column(k)` and `g.view().in.weight_column(k)` are MOSP's
  `HostCsr::weights` for objective k (no copy); the in-edge order differs from MOSP-OpenMP's
  atomics-built reverse graph, which is schedule-dependent anyway; results are canonical.
- The profiler stage `graph.apply` covers apply + transposition (MOSP times `applyBatch`
  separately and puts the reverse graph into `prepare`); map it in
  `parity/timed_regions/sssp.toml`.
- Distance / tree files: `io::read_distances<std::int64_t>`, `io::read_parents<vertex_t>`,
  `io::write_distances`, `io::write_parents` are byte-compatible with MOSP.

### Process note

- The fresh-clone run of `ci/check.sh` caught two fixture inputs that the repository-wide `*.mtx`
  ignore rule had hidden (commit `208a593` exempts `cpp/tests/data`). Always run the gate in a
  fresh clone before reporting a step as done.

### Open items

- The text CSR reader is ~35 % slower than MOSP's (sequential vs three files in parallel). It is
  outside every timed region; parallel parsing can come with the shared parser (CycleEnum
  `from_chars` + threads) in M2.

## Step 3: `sssp` on the CPU backends (2026-09-27)

### Done

- Public API (`<dyng/sssp.hpp>`, PLAN Section 5.1): `options`, `stats` (derived from the new
  `update_stats` in `<dyng/core/stats.hpp>`), the opaque move-only `result` (distances, parents,
  options, graph version, workspace; `clone`, `set_options`, `from_arrays` with canonicalization
  and `validate_inputs`), `compute()` and `update()`; `stale_result_error` when the result does
  not match the graph's version. `<dyng/update.hpp>`: `dyng::update(res, g, batch, results...)`
  and `dyng::update_each(res, g, batch, list)` over `detail::run_update()` (all before-apply work
  on G_t, one commit, all updates on G_{t+1}; a result that fails after the commit is poisoned).
- OpenMP backend (`cpp/src/algorithms/sssp/openmp.cpp`): MOSP-OpenMP@c352151 `sospUpdateCpu` /
  `sospFromScratchCpu` ported straight into the hooks identify_affected / seed / loop / finalize
  (provenance header; `util/list_gather.hpp` from `listGather.h`). Sequential backend
  (`sequential.cpp`): `sequentialSOSPUpdate` adapted into the same hooks; `compute()` runs the
  same loop from the source. Profiler stages `sssp.compute`, `sssp.reset`, `sssp.update`,
  `sssp.prepare`, `sssp.commit`, `sssp.identify_affected`, `sssp.seed`, `sssp.loop`,
  `sssp.finalize` (`update.commit` for the multi-result update).
- `dyng::testing` (installed target `dyng::testing`): `dijkstra()` and `check_sssp_tree()`
  (ports of `dijkstraCsrGraph` and `checkSospTree`; in-edges rebuilt locally).
- Tests (60 new CTest cases; 181 on `dev`): hand cases, the regressions (count-to-infinity
  n = 6 seeds 621705 / 250813: d(1) = 90; the ESCHER 4-vertex case d = 100, 101; ties;
  delete-all), randomized differential tests (5 shapes x 3 scenarios x 3 index types, 3-5
  batches, both backends bit-equal, canonical check against Dijkstra and against a fresh
  compute), stale-result and validation tests, and byte parity with MOSP-OpenMP@c352151 on 34
  fixture cases (every objective, sequential and OpenMP with 1/2/4 threads, 3 index types;
  `invalidated` equal to the original's counter). The fixtures (39 KB,
  `cpp/tests/data/mosp_sssp`) come from `parity/fixtures/sssp/make_sssp_fixtures.sh`, which first
  checks that four original implementations agree byte for byte (`mosp`, `parallelSOSPUpdate`,
  `sequentialSOSPUpdate`, `mospPrep expected`). Two mutations (no root marking in the OpenMP
  invalidation; no children walk in the sequential one) each fail dozens of tests.
- `tools/compat/dyng-compat-mosp` (drop-in clone of the per-objective part of MOSP's `mosp`
  driver and of `mospPrep init`; CTest compares its files with the fixtures),
  `examples/cpp/sssp_update.cpp` (run by CTest), `parity/timed_regions/sssp.toml`,
  ADR 0006 (draft), `docs/algorithms/sssp.md`, the sssp manifest.
- Green: `dev` 181/181, `cpu-only`, `asan` (all pass), `tsan` (OpenMP off, see below), a build
  with `DYNG_ENABLE_OPENMP=OFF` (162/162), Doxygen, REUSE, clang-format.

### Informal check on roadNet-CA (K = 3, `changes_50000_50`, 28 threads pinned, perf lock)

Not the M1a A/B deliverable (that belongs to the parity step), but a regression check of the port:

- `dyng-compat-mosp` (parity preset) and the original `mosp` (-O3, git-archive copy) write
  **byte-identical** `obj<k>/distancesUpdated.txt` and `SSSPTreeUpdated.txt`, with equal
  `invalidated` counters (1651102 / 1105903 / 1007744). `sssp::compute` on both backends writes
  files byte-identical to MOSP-OpenMP's `mospPrep init` for all three objectives.
- Per-objective SOSP region (median of 7 alternating runs, ms): original 35.3 / 23.5 / 22.5,
  dynG 27.8 / 23.5 / 22.6. Before the parallel transposition dynG's first objective was about 17 %
  slower (37.9 vs 32.5) although the kernel is at parity: with `OMP_WAIT_POLICY=active` both
  were equal (23.7 vs 25.5 / 22.4 vs 22.6 / 22.2 vs 22.0). The cause was idle cores: dynG
  transposed the graph sequentially inside `graph.apply`, MOSP builds its reverse graph in parallel
  right before the first objective. The transposition is now parallel on the OpenMP backend
  (deterministic, identical result), which also cut `graph.apply` from about 135 ms to about
  110 ms (MOSP: apply 82 ms + prepare 114 ms).
- Also found: the workspace lists of a fresh result took their first-touch page faults inside the
  timed region of every objective (MOSP shares one workspace across objectives, so only its first
  objective pays them). `reserve()` now touches the pages once, at compute / from_arrays / clone.
- The sequential compute takes about 0.55 s per objective on roadNet-CA (MOSP's Dijkstra in
  `mospPrep init`: about 0.55 s); the OpenMP compute about 60 ms.
- End to end, dynG is slower (about 1.7 s vs 1.2 s) because it reads the graph and the 2K tree
  files one after another (the original reads them concurrently; see Step 2's open item) and
  validates the imported trees.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| Section 5.1: `dyng::update(res, g, batch, results...)` only | also `dyng::update_each(res, g, batch, array_view<result_t* const>)` | the number of objectives is a run-time value (the compat driver, later `mosp`); a variadic pack cannot hold it |
| `result::get_options() const noexcept` | not `noexcept`: throws `invalid_argument_error` for a moved-from result | a moved-from result has no options to return |
| `update_stats::affected` (undefined) | vertices whose distance or parent changed (deterministic); the OpenMP unpack pass counts it (one reduction added to the straight port) | C1 needs `affected == 0` for an empty batch; the count is exact and backend-independent |
| `engine_used` | OpenMP reports `engine::fused` (the ported paper engine), sequential `engine::operators` (hook-by-hook) | the CPU backends have no engine option; the two values tell the tier apart |
| sequential backend = `sequentialSOSPUpdate` adapted | its classification of weight increases is the graph's (`apply_delta`, MOSP `applyChangeBatch` semantics) and per-round candidate flags are generation stamps | one classification for all backends keeps `invalidated` identical across backends and equal to `mospUpdate`; the original's per-round `vector<bool>(n)` is O(n) per round |
| sequential `compute` from `sospFromScratch*` | the static enactor: the same re-scan loop from the source (reset -> seed_static -> loop -> finalize) | it is the framework's compute path and stays independent of `testing::dijkstra`; 0.55 s on roadNet-CA is acceptable for the reference backend |
| `validate_inputs`: "rooted at the source, no parent cycle" | also distance range, unreachable vertices without parents, reachable vertices with reachable parents; parent ids are always range-checked (memory safety), with or without validation | MOSP tolerated some of these silently; imported trees are the only way a corrupt tree can enter |
| the graph's in-edges | sssp requires `store_transposed` (throws otherwise) | building a transposition per call would hide a large cost |
| `instantiate.{cpp,cu}` per algorithm (Section 4.8) | explicit instantiations at the end of `sssp.cpp`, `sequential.cpp`, `openmp.cpp` | each file instantiates what it defines; a separate file would need the definitions in a header |
| `@paper @cite dynamosp2025` | `@paper` with the keys in text | `@cite` needs `CITE_BIB_FILES` and `bibtex`, which the environment does not have |
| examples compiled against the installed package (Section 9.6) | compiled in-tree (`examples/cpp/CMakeLists.txt`) and run by CTest | the install test with `find_package` was done in Step 1; an installed-package example build belongs to the CI setup (M4) |
| `tsan` preset with OpenMP | `tsan` builds with `DYNG_ENABLE_OPENMP=OFF` | GCC's libgomp is not instrumented: every OpenMP barrier and reduction is reported as a race (verified). OpenMP race checking needs Archer (clang + llvm-openmp), not in the environment |
| sssp in the parity step | `parity/timed_regions/sssp.toml` and `parity/fixtures/sssp/` written here | the timing map must exist before the port (Section 6.3 step 3); the byte-parity fixtures were needed by the tests |
| graph module | parallel deterministic transposition for the OpenMP backend | found by the roadNet-CA check above |

### Notes for the next steps

- Parity harness: `dyng-compat-mosp` (built by the `dev` and `parity` presets) reads the `mosp`
  inputs and writes the same `obj<k>` files; its RESULT line and `--timing` CSV map to MOSP's stages
  as `parity/timed_regions/sssp.toml` describes. `parity/fixtures/sssp/make_sssp_fixtures.sh` can be
  folded into the golden export (like the graph/io script); its archive is
  `$DYNG_SCRATCH/runs/sssp-fixtures` (12 MB, with a built `mosp`, `mospPrep` and the two exporters).
- **The initial trees under `$DYNG_SCRATCH/datasets/mosp/*/init` are not canonical** (they differ
  from MOSP-OpenMP's `mospPrep init` in some parents, e.g. vertex 902 of roadNet-CA objective 0:
  959 vs 907; they were prepared with another tool). Without `--canonicalize` both `mosp` and
  `dyng-compat-mosp` keep the parents of untouched vertices, so their outputs still agree byte for
  byte, but golden trees that are meant to equal Dijkstra should be regenerated with
  `mospPrep init` (or both sides run with `--canonicalize`).
- The OpenMP A/B should keep the same thread placement for both sides
  (`OMP_PROC_BIND=close OMP_PLACES=cores`, 28 threads) and report the wait policy; the first
  objective is sensitive to how busy the cores were just before it.
- `mospTest sosp` (148 cases) is not in the committed fixtures; the parity step exports it.

### Open items

- End-to-end time on large graphs is dominated by text parsing done sequentially (graph and 2K
  tree files); parallel parsing is planned with the shared parser (M2).
- Each sssp result owns a workspace (about 38 bytes per vertex); K results on one graph use K
  workspaces, where MOSP shares one. `mosp` (0.2) should share.
- OpenMP race checking (Archer) is not set up; see the tsan row above.

## Step 4: parity harness and the M1a parity proof (2026-09-27)

### Done

- `parity/references.toml`: MOSP-OpenMP `c352151` (baseline-2026-09 = `7284f50`) and MOSP-CUDA
  `e220ee2` (baseline-2026-09 = `ac29545`), full SHAs, paths, build and test commands, run
  environment, external inputs, toolchains.
- `parity/build_reference.sh`: `git archive` into `$DYNG_SCRATCH/ref/<name>@<commit7>/{unpatched,
  patched}`, verification of every tracked file against the archive, build with the original
  Makefile (MOSP-CUDA with `NVCC=/usr/local/cuda-13.1/bin/nvcc`), additive export patch on the
  patched copy only, archive-build check report, and a before/after check that the original's
  HEAD and status did not change. Idempotent (a second run takes about 1.5 s). `--test` ran both
  originals' own `make test` in the unpatched copies: all passed.
- Export patches (`parity/export_patches/<repo>/build.sh`) build the exporters of
  `parity/exporters/mosp/` (moved from `parity/fixtures/`) against the copies' sources; the
  graph/io exporter also builds against MOSP-CUDA with nvcc. The two fixture scripts now use the
  patched copy; they regenerate identical fixtures, and their ad-hoc archives
  (`runs/graph-io-fixtures`, `runs/sssp-fixtures`, 24 MB) were deleted.
- `parity/export_goldens.py`: the sssp golden corpus, 388 cases in 9 groups (10 testcases, 3
  count-to-infinity regressions, the thesis example, 4 ESCHER/ties cases, 17 graph/io fixture
  inputs, the 148 `mospTest sosp` cases, 2 large-weight cases, 3 packing boundaries, 200 stress
  cases), 144 MB under `$DYNG_SCRATCH/goldens/sssp`, with the original implementations
  cross-checked before each case is written. `--twice` re-exported from a fresh archive copy:
  identical manifest. `parity/goldens.toml` (52 KB) is the committed manifest.
- `parity/compare.py` and the CTest test `parity.sssp.mosp_openmp_c352151` (label `parity`,
  `parity` preset; skipped without goldens). Result: 388/388 byte-identical on sequential and
  OpenMP 1/4/16 threads, in the `parity` and `dev` presets (compute, update, updated CSR,
  `invalidated`). A mutation (OpenMP invalidation reduced to the roots) gives 266 mismatches; a
  changed golden byte is detected.
- One-off cross-check: MOSP-CUDA@e220ee2 == MOSP-OpenMP@c352151 on all 388 cases (trees,
  combined graph and MOSP costs, updated CSR and weight-increase bits, counters).
- `parity/perf_ab.py` and the OpenMP A/B on roadNet-CA (safe 50K, unsafe 50K, local 10K with 160
  hops, seed 777, generated by the original's own `mospPrep` as its `bench/prepare.sh` does;
  28 threads pinned; 9 alternating rounds under the perf lock): SOSP compute summed over the
  objectives 0.92 / 0.95 / 0.87x of the original; details and explanations in
  `parity/results/M1a.md`.
- ADR 0013 (draft), `parity/README.md`, the timed-region map extended (workspace finding, draft
  MOSP-CUDA section for M1b), CHANGELOG, README.

### Found and fixed

- **Concurrent reading did not happen under thread pinning.** With `OMP_PROC_BIND` set, libgomp
  pins the initial thread to one core at start-up, and `std::thread` / `std::async` workers inherit
  that one-core mask: reading the batch and the six tree files "concurrently" with `std::async`
  took as long as reading them one after the other (1.19 s vs 1.27 s). The original's
  `runConcurrently` runs its jobs on OpenMP threads, which are placed on distinct cores. dynG now
  has `detail::run_concurrently` (OpenMP region when enabled, `std::async` otherwise), used by
  `io::read_csr_triplet` (RowPtr, ColInd and Values in parallel; any problem re-runs the strict
  sequential readers so the first error is unchanged), and `dyng-compat-mosp` reads, builds its
  K results and writes the same way. Input phase on roadNet-CA: 1.27 s -> 0.63 s; tree import
  150 -> 66 ms; end to end 1.60 s -> 0.87 s.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| `parity/export_patches/<repo>/*.patch` | `export_patches/<repo>/build.sh` compiles standalone exporter programs (`parity/exporters/mosp/`) into `<copy>/parity_export/` | the MOSP exports need no change to any original file; an added program is the most additive patch possible, and `build_reference.sh` proves the tracked files are untouched |
| `$DYNG_SCRATCH` default `~/scratch/dyng-parity/<repo>-<commit>-<variant>` | `$DYNG_SCRATCH/ref/<name>@<commit7>/<variant>` with `DYNG_SCRATCH=~/Projects/dyng-work` | the author's work area (Appendix E) |
| `goldens.toml`: "case -> files -> sha256" | one digest per case (SHA-256 of the case's sorted per-file lines) plus the SHA-256 of the full per-file `MANIFEST.sha256`, which lives next to the goldens | 8,620 per-file lines would make the committed file about 1 MB; the two-level scheme verifies the same bytes |
| goldens as a release asset (`goldens-vN.tar.zst`, `fetch_goldens.py`) | goldens only in `$DYNG_SCRATCH`, regenerated by `export_goldens.py` | no remote before M4; the asset and fetch script come when CI needs the goldens |
| golden replay under `cpp/tests/parity/<algo>/` | `parity/compare.py`, registered as a CTest test with the label `parity` in `tools/compat/CMakeLists.txt` | the replay drives the compat tool and the goldens outside the tree; a script is simpler than a C++ test and also replays another original (`--driver original`) |
| parity certificate `benchmarks/results/<version>/parity.json` by `certify.py` | `parity/results/M1a.md` + JSON records by `compare.py` / `perf_ab.py` | the certificate per release starts with 0.1; the M1a record has the same content |
| stress tests: 100 `stressTest` + 100 `parallelStressTest` seeds | `stressTest 1` and `parallelStressTest 2` | with the same seed the two programs draw the same 100 cases; seed 2 gives 100 more |
| packing boundary n = 2^17 - 1 | the MOSP-CUDA construction (pull and push) and the MOSP-OpenMP one (n = 2^16 + 1), written by the export script | `mospTest` checks them in memory only (no case files) |
| perf inputs from `$DYNG_SCRATCH/datasets/mosp` | the prepared CSR is reused (symlink); `init` and the three batches are regenerated with c352151's `mospPrep` as its `bench/prepare.sh` does | the dataset's trees and batches were made by an older MOSP-CUDA `mospPrep` (non-canonical trees, see Step 3); the SuiteSparse `.mtx` is not on the machine, so the CSR is identified by SHA-256 |
| — | `io::read_csr_triplet` and `dyng-compat-mosp` read concurrently; dynG links `Threads::Threads` | the end-to-end gap on roadNet-CA (see "Found and fixed") |

### Performance findings to carry into M1b

- **Per-objective gate vs workspace sharing.** Objectives 1 and 2 of the 50K batches are 3-6 %
  slower than the original, objective 0 15-20 % faster; the sum is 5-8 % faster. MOSP shares one
  `SospWorkspace` across objectives; each dynG result owns one. A throw-away experiment with one
  shared workspace put objectives 1 and 2 at 0.97-1.03x. Decide in M1b (before the CUDA gate)
  whether results on one graph can share scratch (e.g. through `resources` or an explicit
  workspace argument of `update_each`); `mosp` (0.2) needs it anyway. Until then the gate should
  be read on the per-objective sum.
- **End to end 1.27-1.37x** (gated at 1.10x from M1b): `graph::from_csr` builds a transposition of
  the loaded graph (213 ms on roadNet-CA) that `apply` rebuilds for the updated graph anyway, and
  `from_arrays` validates the trees (66 ms for three). Candidates: a `from_csr` that takes the
  arrays by value and builds the in-edges lazily, and a parallel `validate_tree`.
- The original's own run-to-run spread on this shared machine is large (reading the CSR
  230-380 ms, objective 0 22-28 %); use at least 9 rounds and report the spread.
- `OMP_WAIT_POLICY` changes the per-objective picture (both sides); `perf_ab.py --env` records
  such settings.

### Notes for the next steps

- M1b: add a `mosp_cuda` driver path to `compare.py` for dynG's CUDA backend (the compat tool
  gets `--backend cuda`), and a CUDA-events variant of `perf_ab.py` (>= 20 runs for regions under
  10 ms, GPU 0, `CUDA_MODULE_LOADING=EAGER`). The MOSP-CUDA reference copies are already built,
  and its timers are mapped (draft) in `parity/timed_regions/sssp.toml`.
- The golden set also holds MOSP's combined graph and MOSP costs per case, for `mosp` (0.2).
- Space in `$DYNG_SCRATCH`: ref 56 MB, goldens 144 MB, bench inputs 160 MB (+ the datasets).

### Open items

- The goldens are not yet published (release asset + `fetch_goldens.py`, M4), so the `parity`
  label only runs where `export_goldens.py` has run.
- The per-objective OpenMP gate and the end-to-end gap above (M1b).
- `perf_ab.py` measures the OpenMP backend only; the sequential backend has no original
  counterpart with timers (`sequentialSOSPUpdate` is a file-based function).
