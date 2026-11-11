# ADR 0096: The hyper_sssp API review and freeze (0.4)

- **Status:** Accepted under delegation (2026-11-11; GOVERNANCE.md, "Delegation of technical
  ADRs") for the review, its documentation amendments and the freeze mechanism. The names and the
  semantics it freezes are the author's, **decided by the author on 2026-10-28** (GOVERNANCE.md,
  approvals log, the rows of 2026-10-28): HD8 (the budget options and `budget_action`), HD11 (the
  distance semantics and the weight limit), HD12 (the new public names), HD13 (the engines), HD14
  (no openmp backend), HD7 / AD20 (the `fallback_used` determinism sentence) and HD17 (frozen at the
  end of M11b; `stable` from 0.4.0 only on the author's confirmation). **Pending the author:** the
  stability confirmation at the 0.4.0 release (HD17, GOVERNANCE.md, Open decisions, M11 (HD)) and
  the acknowledgement of the public names that HD12's list does not spell out (section "The new
  public names" below). It changes no rule the author approved: the stability policy of PLAN 5.9,
  the gates and their protocol, the parity rules and the frozen headers (`core/*`, `graph/*`,
  `hypergraph/*`, `update.hpp`) are untouched.
- **Date:** 2026-11-11
- **Deciders:** the AI assistant, on the author's behalf (M11b, step
  public-python-cli-docs-freeze); the names, semantics and maturity: S M Shovan (2026-10-28, HD7,
  HD8, HD11-HD14, HD17)
- **Builds on:** ADR 0023 (the method, the snapshot), ADR 0035 (the freeze enforced against the
  base branch; the maturity decides the freeze), ADR 0047 (the freeze's Doxygen rules), ADR 0060
  (the precedent: triad_count's review and freeze), ADRs 0090-0092 and 0095 (hyper_sssp's API, its
  port, its budget, its engines), ADR 0097 (the Python module and the command line).

## Context

`hyper_sssp` arrived in M11a as a draft (ADR 0090), its header tracked by the API baseline; M11b
added the shared sssp engine (`engine::operators`, ADR 0095), `dyng.hyper_sssp` and `dyng
hyper_sssp` (ADR 0097). HD17 freezes the C++ header, the Python module and the command line at
the end of M11b and makes them stable at 0.4.0 only on the author's confirmation. From the freeze
a change of them is a SemVer matter, so the API is reviewed first, as ADRs 0023, 0035, 0047 and
0060 reviewed the earlier APIs: against the checklist `docs/developer/api_review_checklist.md`
(naming and the verb vocabulary, the uniform contract of ADR 0006, Doxygen, exception guarantees
and `dyng::error`, host-compilability, ownership and ABI, the bindings), against its siblings
`sssp` (the same fixed-point update on a graph) and `triad_count` (the other hypergraph result),
and against the sketch `docs/design/sketches/hyper_sssp.md`.

Reviewed: every public entity of `cpp/include/dyng/hyper_sssp.hpp` (`budget_action`,
`fallback_reason`, the two `to_string()`, `options`, `stats`, `result`, `compute()`, `update()`,
and the `detail` contract the baseline lists: `hyper_sssp_supported_v`,
`update_traits<hyper_sssp::result>`, the two `DYNG_HYPER_SSSP_*` messages); M11a's container
additions against what hyper_sssp ended up needing; the Python module and the command line (ADR
0097); `docs/algorithms/hyper_sssp.md`. Out of scope, as tracked: `<dyng/testing/
hyper_sssp_reference.hpp>` (PLAN 5.9: io, generators and testing are frozen as a whole later).

## Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | ADR 0004's rules hold (clang-tidy's naming rules pass): `budget_action` {`recompute`, `error`}, `fallback_reason` {`none`, `work_budget`, `iterations`, `distance_width`, `invalidation_depth`}, `options::work_budget`, `max_iterations`, `on_budget`, `cuda_engine`, `validate_inputs`, `stats::line_graph_insertions`, `line_graph_deletions`, `invalidated`, `relaxations`, `fallback_reason`; the algorithm is named for what it computes, the paper (H-SOSP) is in `@paper`. The names are HD12's. `on_budget` stays separate from label_propagation's `on_limit` (HD12). | none needed |
| 2 | The verb vocabulary | `compute`, `update`, `clone`, `set_options`, `get_options` (ADR 0023's accepted exception), `to_string`; the accessors are nouns (`source`, `hyperedge_distances`, `hyperedge_parents`, `vertex_distances`, `graph_version`, `space`); `distance_to(v)` is sssp's PLAN 5.3 spelling of a per-vertex read. | none needed |
| 3 | The uniform contract (against `sssp` and `triad_count`) | The same skeleton: an aggregate `options` (fields appended only, every one a tunable through `set_options()`), `stats : update_stats` with `batch` (a `hyper_apply_summary`, as triad_count's) first, an opaque move-only pimpl `result<vertex_t, distance_t = std::int64_t>` (a template as sssp's: its arrays are by id and live on the device on cuda), `compute(res, h, source, opt)` never mutates `h` (the automatic line graph aside, documented), `update(res, h&, batch, r&)`, `stale_result_error` on a version mismatch. The source is an argument of `compute()` as sssp's. The sketch's `options::delta` is gone (HD8: MOSP_ESCHER's UpdateConfig has none; the shared engine chooses MOSP's defaultDelta internally) and its int64 parents are int32 (slack_csr ids are below 2^31; ADR 0090). | none needed (the sketch's two departures are ADR 0090's and HD8's) |
| 4 | `stats` and determinism | Every field states its level; the inherited fields are explained in the class comment, `fallback_used` with HD7's sentence (deterministic on sequential, schedule-dependent on cuda near the budget), and `<dyng/core/stats.hpp>` carries the joint per-field sentence (HD7 / AD20). The comments of `fallback_reason::distance_width` and `invalidation_depth` named only `engine::fused`'s limits (32 bits, 64 rounds), though `engine::operators` (M11b) has its own (the bits next to the parent; ceil(log2 n) + 1 rounds). | Reworded: each names both engines' limits |
| 5 | Thread safety | The result's comment said its accessors may run concurrently with each other. `distance_to()` and `vertex_distances()` fill caches of the result (the vertex distances; on cuda a host copy of the hyperedge distances) on their first call per version and read the hypergraph's rows, so two of them, or one and a change of the hypergraph, would race. | Documented: those two may not overlap another call on the result or a change of its hypergraph (the other accessors may run concurrently); the Python binding holds the result's lock exclusively and the hypergraph's shared during them (ADR 0097). No library change (a lock in the result would change libdyng's objects, whose gates step 4 reads; the documented rule is the C++ library's general one, ADR 0023 note 5) |
| 6 | Doxygen | `compute()`, `update()` carry `@backends`, `@determinism`, `@paper` (key `hsosp2026`) and `@guarantee`. `compute()`'s description named only MOSP_ESCHER's recompute, not `engine::operators`' static solve (sssp's near-far loop); `update()`'s `@sync` said the stream is synchronized once per pointer-jumping round, which holds for `engine::fused` only; `compute()`'s `@throws not_supported_error` ran two conditions together. The freeze's Doxygen rules (ADR 0047: every entity in the group, `@throws` or `noexcept` on every function, `@sync` / `@async` where a `cuda_error` is documented) were checked by hand only. | Reworded the three; `ci/doxygen_coverage.py` holds `dyng/hyper_sssp.hpp` to the freeze's rules (`FROZEN_GROUP_HEADERS`, group `hyper_sssp`), as it holds `triad_count.hpp`: OK on 217 compounds |
| 7 | Exceptions leaving the library | `compute()`, `update()`, `clone()` and `vertex_distances()` translate host allocation failures (`DYNG_TRANSLATE_ALLOCATION_FAILURE`, M11a; `hyper_sssp_allocation_failure_test.cpp`); every documented error is a `dyng::error`; the openmp refusal names sequential and cuda and the fix (HD14). | none needed |
| 8 | Exception guarantees | `compute()` strong; `update()` strong before the commit (weights, version, backend, storage, options), basic inside it (the hypergraph poisoned, the result unchanged), basic after it (a spent budget under `budget_action::error`, a CUDA or allocation failure: the result poisoned); `set_options()` strong. | none needed |
| 9 | Host-compilability | `<dyng/hyper_sssp.hpp>` includes no CUDA, CUB, Thrust or libcu++ header; its self-containment unit builds in every preset, and a CXX-only consumer of the installed package builds and runs the example (`ci/cmake_consumer.sh`). | none needed |
| 10 | Ownership and ABI | `result` is a unique pimpl; `options` and `stats` are aggregates whose size changes when a field is appended (PLAN 5.9 allows it before 1.0). | none needed |
| 11 | Limits | weights in [1, 2^28] (MOSP_ESCHER's), int32 vertex ids and parents, the `hypergraph<int32_t, int32_t>` instantiation only (`hyper_sssp_supported_v`). | Accepted: a later release may relax a limit (an api-change with a CHANGELOG entry), never tighten it |
| 12 | Composition | `dyng::update(res, h, b, r1, r2, ...)` composes hyper_sssp results with each other (several sources); with triad_count results it is refused by triad_count's requirement (slack_csr storage, HD9); a graph result with a hypergraph stops in the frozen `update_traits` (`DYNG_HYPER_SSSP_CONTAINER_MESSAGE`). | none needed |
| 13 | M11a's container additions against what hyper_sssp needed | hyper_sssp reads `hypergraph_storage::slack_csr`, `line_graph_policy::slack_csr` (and `automatic` under it), the vertex-to-hyperedge rows, the weights by id, `hyper_apply_summary`'s `inserted_ids`, `inserted_incidences`, `deleted_incidences` and `skipped_ops`, and the batch's incidence arrays; the `.dgb` flag `incidence_edits` and the `.dgt` lines carry them. Every addition is used; none is missing (the pair delta stays internal: `stats::line_graph_insertions` / `deletions` count it; `overlaps` stays empty, HD9; no capacity field, HD9). | none needed |
| 14 | The sketch's open questions | (1) per-vertex distances as an array: yes, `vertex_distances()`, computed on request once per version (not maintained by `update()`; HD11); (2) parity by id needs `id_reuse::insert_first`, and the documented default stays `erase_first` (O16): yes, as the sketch said, with by-key parity for the other order (HD2). | The sketch is marked frozen with both answers |
| 15 | After 0.4 | an OpenMP backend (HD14: later) adds a backend, not an API; escher-storage incidence edits and the composition with triad_count change requirements (documented behaviour; an api-change entry); the shared static solve inside the as-is engine (a roadmap follow-up of ADR 0095) is internal. | No API change needed |

## Decision

1. **The hyper_sssp API is frozen** as reviewed, with the documentation amendments of findings 4,
   5 and 6: every declaration of `<dyng/hyper_sssp.hpp>` (its baseline section, the `detail`
   contract included), the Python names of `dyng.hyper_sssp` (and `dyng.update()` over a
   `dyng.Hypergraph` with its results), and the flags, output files and exit statuses of `dyng
   hyper_sssp` (ADR 0097). No signature changed in the review, so the freeze needs no migration
   note.
2. **Maturity.** The manifest says `maturity = "stable"` (with its `origin` of MOSP_ESCHER@4b86159
   unchanged; the scaffold's line and its planned entry removed), so `ci/api_snapshot.py` lists
   `dyng/hyper_sssp.hpp` as frozen (ADR 0035 decision 2) and `scripts/regen.py` writes `stable`
   into the algorithm tables and the registry, as ADR 0060 did for triad_count. By HD17 this is
   **stable from 0.4.0 only on the author's confirmation** at the release (GOVERNANCE.md, Open
   decisions, M11 (HD)); without it the release says `experimental` again (the manifest's one
   field) while the header stays frozen. `slack_csr` storage and the incidence edits join the
   stable hypergraph surface on the same confirmation (they are documented behaviour of the frozen
   hypergraph headers, ADR 0085). The registry's maturity is a constant of libdyng's
   `registry.cpp`: its object differs from the step-2 library by that byte, which M11b's final
   identity proof (criterion 10) accounts for.
3. **The baselines.** `cpp/tests/api/api_snapshot/public_api.txt` lists `[dyng/hyper_sssp.hpp]
   frozen` (`ci/docs.sh --update-api`); from the merge of M11b a removed or changed declaration of
   it fails `ci/docs.sh` (against the baseline) and `ci/api_check.sh` (against the base branch)
   without the `api-change` label and a CHANGELOG entry. The Python half: `ci/api_check.sh` runs
   griffe against the base branch, so `dyng.hyper_sssp` is checked from the commit that merges it
   into `main`; the tests below pin the rule now.
4. **Both checks fail on a deliberate change** (shown on 2026-11-11 in a throwaway clone of this
   branch at `1987c2b`, the freeze commit, reverted and deleted; script and log
   `$DYNG_SCRATCH/runs/m11b-step3/freeze-demo.{sh,log}`), and the tests keep it:
   - `ci/tests/test_api_snapshot.py::test_the_hyper_sssp_header_is_frozen_and_a_change_is_breaking`:
     the section is frozen with its detail contract, the oracle tracked; a changed default of
     `options::work_budget`, a removed `stats::relaxations`, an int64 `hyperedge_parents()` and a
     removed enumerator `invalidation_depth` are breaking changes of `[dyng/hyper_sssp.hpp]`;
   - `ci/tests/test_api_check.py`: a renamed parameter of `dyng.hyper_sssp.update`, a changed
     default of `Options.work_budget` and a removed `Result.vertex_distances` are breaking changes
     for griffe.

## The new public names (for the author; HD12)

HD12 decided the names of M11 as the plans list them (GOVERNANCE.md, approvals log, "The new
public names of 0.4", 2026-10-28): hyper_sssp's `budget_action`, `fallback_reason`, `options`,
`stats`, `result` and its accessors; `testing::hyper_sssp_reference`, `check_hyper_sssp_tree`
and `line_graph_delta`; the appended `hyperedge_batch_mix` enumerators; the compat tool
`dyng-compat-hsosp`; Python `dyng.hyper_sssp` and the command line `dyng hyper_sssp`; the `.dgb`
names `incidence_edits`, `known_graph`, `known_hypergraph`. Every public name M11 adds, frozen by
this ADR or tracked, with what the decided list does not spell out marked **(ack)**, an
acknowledgement item under GOVERNANCE.md's Open decisions (M11 (HD)):

- **C++ `dyng::hyper_sssp`** (ADR 0090; frozen): `budget_action` {`recompute`, `error`} (the
  enumerators: already listed for acknowledgement in M11a), `fallback_reason` {`none`,
  `work_budget`, `iterations`, `distance_width`, `invalidation_depth`}, `to_string(budget_action)`,
  `to_string(fallback_reason)` (listed in M11a); `options` (`work_budget`, `max_iterations`,
  `on_budget`, `cuda_engine`, `validate_inputs`); `stats` (`batch`, `line_graph_insertions`,
  `line_graph_deletions`, `invalidated`, `relaxations`, `fallback_reason`); `result` (`source`,
  `hyperedge_distances`, `hyperedge_parents`, `distance_to`, `vertex_distances`, `get_options`,
  `set_options`, `graph_version`, `space`, `clone`: the last five listed in M11a); `compute`,
  `update`; the macros `DYNG_HYPER_SSSP_TYPES_MESSAGE`, `DYNG_HYPER_SSSP_CONTAINER_MESSAGE` (listed
  in M11a); the detail contract `hyper_sssp_supported_v`, `update_traits<hyper_sssp::result>`; the
  profiler stages `hyper_sssp.compute`, `.normalize`, `.identify_affected`, `.seed`, `.loop`,
  `.recompute`, `.finalize` **(ack)**.
- **The container (0.4, M11a; documented behaviour of the frozen hypergraph headers)**:
  `hypergraph_storage::slack_csr`, `line_graph_policy::slack_csr` (PLAN 4.4's names), the batch's
  `delete_incidence` / `insert_incidence` (0.3's reserved names); the seven `hyperedge_batch_mix`
  enumerators `incidence_removals` ... `incidence_out_of_range` (listed in M11a);
  `io::batch_file_flags::incidence_edits`, `known_graph`, `known_hypergraph` (HD10 / HD12).
- **Python** (ADR 0097): `dyng.hyper_sssp` with `BudgetAction`, `FallbackReason` (StrEnums of the
  C++ enumerators), `INFINITE_DISTANCE` **(ack;** sssp's constant, the same value**)**, `Options`,
  `Stats`, `Result` (`source`, `hyperedge_distances`, `hyperedge_parents`, `distance_to`,
  `vertex_distances`, `options`, `set_options`, `graph_version`, `space`, `clone`), `compute(
  hypergraph, source, *, options, resources, **kwargs)`, `update(hypergraph, batch, result, *,
  resources)`; `dyng.update()` over a `dyng.Hypergraph` with hyper_sssp results;
  `dyng.citation("hyper_sssp")`; the container's `HyperedgeBatch.delete_incidences`,
  `insert_incidences` (the 0.3 constructor keywords, now also properties) and
  `num_incidence_edits` **(ack)**, `dyng.io.BatchFileHeader.incidence_edits` **(ack;** the flag's
  name**)**, the seven incidence mixes in `dyng.generators.HyperedgeBatchMixName` (the C++
  enumerators); `dyng.testing.hyper_sssp_reference(hypergraph, source)` and
  `check_hyper_sssp_tree(hypergraph, result)` (the C++ oracle's names); the Hypothesis strategy
  keyword `max_incidence_edits` and the fields `RandomHyperedgeBatch.delete_incidences` /
  `insert_incidences`, the model keywords `HypergraphModel.apply(..., delete_incidences=,
  insert_incidences=)` **(ack;** tracked, `dyng.testing`**)**.
- **The command line** (ADR 0097): `dyng hyper_sssp compute|update` with the option flags
  `--work-budget`, `--max-iterations`, `--on-budget`, `--cuda-engine`, `--validate-inputs` (the
  fields, kebab case), and **(ack)** `--input` / `--hypergraph`, `--read-options`,
  `--random-weights`, the read flags of `dyng triad_count`, `--source`, `--out`,
  `--vertex-distances`, `--batches`, `--compare-recompute`, `--quiet`, the backend flags, and the
  output files `hyperedge_distances.txt`, `hyperedge_parents.txt`, `vertex_distances.txt`.
- **The compat tool** (M11a, HD12): `dyng-compat-hsosp to-dgb`, `replay`, `bench`.

## Consequences

- From the merge of M11b, `<dyng/hyper_sssp.hpp>`, `dyng.hyper_sssp` and `dyng hyper_sssp` change
  only as api-changes (additive, or with a migration note), and are SemVer-stable from 0.4.0 once
  the author confirms (HD17).
- The thread-safety rule of finding 5 is the documented contract; a lock inside the result (which
  would let the two accessors run concurrently) is a possible later change that removes a
  restriction, never one that adds one.
