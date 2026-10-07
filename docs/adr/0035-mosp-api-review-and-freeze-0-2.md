# ADR 0035: The mosp API review and freeze (0.2)

- **Status:** Accepted under delegation (2026-10-07; GOVERNANCE.md, "Delegation of technical
  ADRs") for the review, its amendments and the freeze mechanism. The maturity itself, `mosp`
  **stable** from 0.2.0, is the author's decision of 2026-10-07 (GOVERNANCE.md, approvals log),
  taken on the condition that this review comes first and that the release certificate of 0.2.0
  carries mosp's full gate record. Nothing here changes a rule the author approved: the stability
  policy of PLAN Section 5.9, the performance gates and their protocol (ADRs 0018, 0021), the
  parity rules and the goldens are untouched; no public signature changed.
- **Date:** 2026-10-07
- **Deciders:** the AI assistant, on the author's behalf (R020, step mosp-freeze); the maturity:
  S M Shovan

## Context

`mosp` arrived in M7 (ADRs 0027, 0028, 0029) at maturity `experimental`, so PLAN 5.9 allowed its
API to change in any minor release. The author decided that it becomes `stable` in 0.2.0, as
`sssp` and `cycle_count` became stable in 0.1.0. From then on a change of `<dyng/mosp.hpp>`,
`dyng.mosp` or `dyng mosp` is a SemVer matter, so the API is reviewed first, the way ADR 0023
reviewed the 0.1 headers: against `docs/developer/api_review_checklist.md` (naming and the verb
vocabulary, the uniform contract of ADR 0006, Doxygen, exception guarantees and `dyng::error`,
host-compilability, ownership and ABI, the bindings), against `sssp` and `cycle_count`, and against
what the coming work could force to break: the hypergraph container and the ESCHER store of 0.3
(M8a / M8b), `triad_count` (M9), `hyper_sssp` (which builds on shortest paths, PLAN 6.4.8) and
`label_propagation` (0.4, PLAN 6.4.7, with the slotted row layout and vertex batches).

Reviewed: every public entity of `cpp/include/dyng/mosp.hpp` (the constants, `options`, `stats`,
`result`, `compute()`, `update()`, and the `detail` contract the baseline lists:
`mosp_supported_v`, `DYNG_MOSP_TYPES_MESSAGE`, `update_traits<mosp::result>`); the mosp parts of
shared headers (`graph_properties::mosp_compatible()`, `io::write_path_costs()`,
`<dyng/testing/mosp_oracle.hpp>`, `generators::legacy::mosp_changes()`, the include in
`<dyng/dyng.hpp>`); the Python module `dyng.mosp` (`python/dyng/mosp.py`,
`python/bindings/mosp.cpp`) with `dyng.update()` and the `dyng.io` / `dyng.testing` helpers; the
command line `dyng mosp compute|update` (`python/dyng/cli/_algorithms.py`); and
`docs/algorithms/mosp.md`.

## Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | The names follow ADR 0004 (clang-tidy's naming rules pass): `options`, `stats`, `result`, `max_objectives`, `max_preference_scale`; the accessors are nouns (`distances`, `parents`, `combined_distances`, `combined_parents`, `path_costs`, `preference_scale`), the count is `num_objectives`; the algorithm is named for what it computes, the paper (DynaMOSP) is in `@paper`. | none needed |
| 2 | The verb vocabulary | `compute`, `update`, `from_arrays` (`from_*`), `clone`, `set_options`; `get_options` is ADR 0023's accepted exception (the type `options` is in scope). | none needed |
| 3 | Contract uniformity with `sssp` and `cycle_count` | The skeleton is the same: an aggregate `options` (fields appended only), `stats : update_stats` with `batch` first, an opaque move-only pimpl `result` with `get_options` / `set_options` / `graph_version` / `space` / `clone`, `compute(res, g, source, opt)` (never mutates `g`), `update(res, g&, batch, r&)` (applies the batch, throws `stale_result_error`), `from_arrays()` as sssp's, `update_traits` for `dyng::update()`. Differences: the K trees are read with the objective as an argument (`distances(k)`, `parents(k)`), where sssp has `distances()`; `path_costs()` is host memory on every backend while `space()` describes the trees and the combined arrays; `result` is a template on the vertex type as sssp's (ADR 0023 finding 4: its arrays hold vertex ids). | Accepted as the rule: an indexed family of arrays is a method with the index (ADR 0028 for Python). |
| 4 | Fixed inputs | `preferences` and `num_objectives` are fixed at `compute()` but live in `options` (as `sssp::options::objective`); `set_options()` rejects a change. PLAN 5.1 allows "arguments or read-only afterwards". | Accepted. |
| 5 | `stats` | Every appended counter says *deterministic*, but the meaning of the inherited `update_stats` counters for mosp (`affected`, `iterations`, `frontier_visits`, `converged`, `fallback_used`, `engine_used`) was documented only in the Python docstring; `objectives` referred to "their deterministic counters". | Documented in the header (`stats`): which counters are deterministic, deterministic per backend or schedule-dependent, and what each sums (`a1d2815`). |
| 6 | Doxygen | `compute()` and `update()` carry `@backends`, `@determinism`, `@paper` (keys `dynamosp2025`, `dynamosptpds2025`) and `@guarantee`; `ci/doxygen_coverage.py` reads mosp's namespace from the manifest. `from_arrays()`'s `@throws` omitted the cases it inherits from `sssp::result::from_arrays()` (the copy policy, an array in device memory without CUDA, host lists); `clone()` did not say that it sizes the pooled workspace as sssp's does. | Documented (`a1d2815`). |
| 7 | Exception guarantees | `compute()` / `from_arrays()` strong; `update()` strong before the commit, basic after it (the result is poisoned), as sssp. `set_options()` promised "strong" and changes the K sssp results before its own options: checked that nothing after its checks can throw (the K sssp results accept options with `delta >= 0` and objective k; the preferences are equal, so the copy assignment reuses their storage and allocates nothing). | Kept, with the reason in a comment and `MospAllocationFailure.SetOptionsIsStrong` (a rejected change leaves every option; an accepted one succeeds while every allocation fails) (`a1d2815`). |
| 8 | Exceptions leaving the library | Every compiled entry point (`compute`, `update`, `from_arrays`, `clone`) ends in `DYNG_TRANSLATE_ALLOCATION_FAILURE`; the header-inline `compute()`, `update()` and `update_traits::make_participant()` only forward (the last inside `dyng::update()`, which translates). The Python layer raises the `dyng` errors and, for an argument of the wrong type, `TypeError`, as `dyng.sssp` and `dyng.cycle_count` do (ADR 0011). | none needed |
| 9 | Host-compilability | `<dyng/mosp.hpp>` includes no CUDA, CUB, Thrust or libcu++ header; its self-containment unit (`dyng_mosp_hpp.cpp`, ADR 0023 finding 9) builds in every preset. | none needed |
| 10 | Ownership and ABI | `result` is a unique pimpl. `options` (a `std::vector` of preferences) and `stats` (a `std::vector<sssp::stats>`) are aggregates whose size changes when a field is appended, also when `sssp::stats` gains one; PLAN 5.9 allows this before 1.0 (CMake `SameMinorVersion`), as for the 0.1 aggregates. | Accepted for 0.x (ADR 0023 finding 10). |
| 11 | The memory space of `path_costs()` | ADR 0027 computed the costs on the host "first, the device later". Moving them to device memory later would break every caller that reads them on the host. | Committed in the header: `path_costs()` returns host memory; device costs, if they come, are an appended option whose default keeps host memory (`a1d2815`). |
| 12 | Limits | `max_objectives = 64` and `max_preference_scale = 2^20` are in the baseline. | Documented: a later release may raise them (a relaxation, with a baseline update and a CHANGELOG entry), never lower them (`a1d2815`). |
| 13 | Composition | `dyng::update(res, g, b, paths, other...)` and `dyng.update(g, b, paths, other)` compose mosp with any graph result through one commit (conformance C10); `update_each()` is not needed by mosp. | none needed |
| 14 | Python | `dyng.mosp` mirrors the C++ API: `Options` (the C++ fields and defaults in order, tested equal to `native.MospOptions()`), the frozen dataclass `Stats` (`objectives` a tuple of `dyng.sssp.Stats`), `Result` with the methods `distances(k)` / `parents(k)` and properties for the rest (`get_options()` is the property `options`, as in `dyng.sssp`), `compute(graph, source, *, options, resources, **kwargs)`, `update(graph, batch, result, *, resources)`, `Result.from_arrays()`, the constants (tested equal to the native ones). Path costs are an (n, K) `dyng.Array` (ADR 0028). | none needed |
| 15 | Command line | `dyng mosp compute|update`: one flag per option field, generated from `dyng.mosp.Options` (`--preferences 4,1,4`, `--num-objectives`, `--delta`, `--cuda-engine`, `--[no-]compute-path-costs`, `--validate-inputs` on `update` only, as for `dyng sssp`), `--source`, `--init`, `--canonicalize`, `--write-graph`, the batch flags; MOSP's output files (ADR 0028). Under PLAN 5.9 these flags and files are now stable. | none needed |
| 16 | 0.3: the hypergraph, the ESCHER store, `triad_count`, `hyper_sssp` | mosp is a graph algorithm and keeps its own namespace; `hyper_sssp` is a separate algorithm with its own result (sketch `docs/design/sketches/hyper_sssp.md`) and reuses sssp's engine through the framework (internal-stable), not through `mosp` or `sssp`'s public API. A hypergraph overload of `mosp::compute()` / `update()` would be an overload on another container type, not ambiguous with the existing ones. `dyng::update()` reaches the hypergraph through `participant_of` (ADR 0023 A1), which mosp does not touch. One diagnostic: `update_traits<mosp::result>::make_participant` (like sssp's and cycle_count's) reads `container_t::edge_type`, which the sketched hypergraph does not have, so composing a mosp result with a hypergraph fails to compile without naming the reason. | No API change needed. The diagnostic is `detail` (no SemVer surface) and shared by the three algorithms: left to M8b, which adds the hypergraph's `participant_of` (open item). |
| 17 | 0.4: `label_propagation` | It brings `row_layout::slotted` / `slack`, vertex soft deletion and vertex batches. Today a graph with another layout than `compact` cannot be built (`graph` throws), so mosp sees compact graphs only; when 0.4 adds the layouts, mosp (through sssp) either supports them or `compute()` rejects them with `invalid_argument_error` naming the property (PLAN 5.1, graph requirements), which is not a break of 0.2 calls. A vertex batch is another batch type: an `update()` overload, if mosp ever accepts one. New `update_stats` / `apply_summary` fields are appended. | No API change needed. |
| 18 | Objective selection | `num_objectives` selects the first K weight columns (MOSP's `-k`); choosing arbitrary columns would need another field. | Accepted: such a selection would be an appended field; `num_objectives` keeps its meaning. |
| 19 | The mosp parts of shared headers | `graph_properties::mosp_compatible()` and the umbrella's `#include <dyng/mosp.hpp>` are in frozen headers. `io::write_path_costs()`, `<dyng/testing/mosp_oracle.hpp>` and `generators::legacy::mosp_changes()` are in `io/*`, `testing/*` and `generators/*`, which stay *tracked* for every algorithm (sssp's `write_distances()` and `testing::dijkstra()` too): listed and checked by the baseline, not frozen. | Unchanged; the review of those headers as a whole is still open (ADR 0023, Consequences). |
| 20 | The tutorial algorithms | `dynamic_bfs` and `triangle_delta` stay at maturity `tutorial` (ADR 0033): their headers stay *tracked*, their Python modules are not SemVer-covered and they have no CLI command. | Unchanged; not frozen by this ADR. |

## Decision

1. **The mosp API is frozen** as reviewed above, with the documentation amendments of findings 5,
   6, 7, 11 and 12: every declaration of `<dyng/mosp.hpp>` (the 39 lines of its baseline section, its `detail`
   contract included), the Python
   names of `dyng.mosp`, and the flags and output files of `dyng mosp`. From 0.2.0 they follow
   PLAN 5.9's stable tier: before 1.0 a break only in a minor release with a CHANGELOG "Changed" or
   "Removed" entry and a migration note; struct fields and enumerators only appended. No
   signature changed in the review, so the freeze needs no migration note.
2. **The maturity decides the freeze.** `ci/api_snapshot.py` marks `dyng/<name>.hpp` *frozen*
   exactly when the algorithm's manifest says `maturity = "stable"` (sssp and cycle_count since
   0.1, mosp since 0.2), next to the frozen core headers of ADR 0023; every other algorithm header
   (the tutorial algorithms) is *tracked*. Before, the list was written by hand and labelled
   `mosp.hpp` frozen while the algorithm was experimental. A test keeps the label and the manifest
   in step (`test_an_algorithm_header_is_frozen_exactly_when_its_manifest_says_stable`).
3. **Both API checks fail on a deliberate mosp change**, shown on 2026-10-07 in a throwaway clone
   of this branch (and kept as tests):
   - C++: `mosp::compute(..., const options& opt = {})` with the default removed: `ci/docs.sh
     --doxygen-only` exits 1 with

     ```text
     -  template <...> [[nodiscard]] result< vertex_t > dyng::mosp::compute(const resources &res, const graph< vertex_t, edge_t, weight_t > &g, vertex_t source, const options &opt={})
     +  template <...> [[nodiscard]] result< vertex_t > dyng::mosp::compute(const resources &res, const graph< vertex_t, edge_t, weight_t > &g, vertex_t source, const options &opt)
     api-snapshot: the public C++ API differs from the baseline (above: - baseline, + headers).
     ```

     (`test_a_mosp_signature_change_fails_in_the_frozen_mosp_section` pins the same on a
     synthetic Doxygen tree.)
   - Python: `dyng.mosp.compute(graph, source, ...)` with `source` renamed `root`:
     `ci/api_check.sh 8b9067e` exits 1 with `compute(source): Parameter was removed` and
     `compute(root): Parameter was added as required`; unchanged, it passes ("no breaking
     change"). `ci/tests/test_api_check.py` runs griffe on two copies of `python/dyng` and pins
     three breaking changes of `dyng.mosp` (a renamed parameter, a changed `Options` default, a
     removed `Result` accessor).
4. **What is not frozen:** `dyng::detail` (the participant, the solve), the profiler stage names
   other than those the benchmark suites use (PLAN 5.9), and the tracked headers of finding 19.

## Consequences

- From 0.2.0, `mosp` is covered like `sssp` and `cycle_count`: the algorithm tables, the
  registry (`dyng::algorithms()`, `dyng.algorithms()`) and `docs/algorithms/mosp.md` say
  `stable`.
- A later maturity change of any algorithm moves its header between *frozen* and *tracked* by
  its manifest alone (`scripts/regen.py`, then `ci/docs.sh --update-api` for the label).
- Open for M8b: the diagnostic of finding 16 (a static assertion that names the container when a
  graph result is composed with a hypergraph), for all three stable algorithms at once.
