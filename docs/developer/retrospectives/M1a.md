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

### Open items

- The text CSR reader is ~35 % slower than MOSP's (sequential vs three files in parallel). It is
  outside every timed region; parallel parsing can come with the shared parser (CycleEnum
  `from_chars` + threads) in M2.
