# ADR 0023: The 0.1 API review and freeze

- **Status:** Accepted under delegation (2026-09-29; GOVERNANCE.md, "Delegation of technical
  ADRs"). It records the review of the 0.1 headers that PLAN Section 11.2 (M3) asks for, the
  fixes it led to, and the mechanism that freezes the result. It does not change a rule the author
  approved: the stability policy of PLAN Section 5.9, the performance gates, the measurement
  protocol (ADRs 0018, 0021) and the parity rules are untouched. With this ADR the algorithm
  contract of ADR 0006 is accepted as the frozen 0.1 contract.
- **Date:** 2026-09-29
- **Deciders:** the AI assistant, on the author's behalf (M3, step api-freeze)

## Context

PLAN Section 5 freezes the 0.1 headers in M3, after `sssp` and `cycle_count` run through the
framework on all three backends: `core/*`, `graph/*`, `update.hpp`, `sssp.hpp` and
`cycle_count.hpp`. Until then they evolved with the two ports. The freeze is preceded by a review
against `docs/developer/api_review_checklist.md`: naming (PLAN 4.4, ADR 0004), the uniformity of
the algorithm contract (PLAN 5.1, ADR 0006), Doxygen completeness (PLAN 9.1), host-compilability
(PLAN 4.1), ownership and ABI (pimpl), exception guarantees (PLAN 4.7.3), and anything that would
hurt the Python bindings of M5 (PLAN 5.4). After the freeze a change of these headers is an API
change under PLAN 5.9, so the freeze needs a check that notices one.

The review covered every header under `cpp/include/dyng` (the top-level `dyng.hpp`,
`citation.hpp`, `version.hpp`, `config.hpp` with the core ones). `io/*`, `generators/*` and
`testing/*` were read but are not frozen in 0.1: their shape is decided together with the CLI and
the Python layer in M5 (PLAN 5.6-5.8).

## Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | clang-tidy's naming rules (ADR 0004) pass on every header; no `_t` type names, `*_type` aliases, `enum class` everywhere, `DYNG_` macros only. | none needed |
| 2 | Naming: the verb vocabulary | Verbs outside the fixed list: `insert_edge` / `delete_edge` (the plan's builder names), `edge_list::add_edge`, `reserve` / `clear` / `resize` (the standard container verbs), `profiler::begin_stage` / `end_stage` / `add_counter` and `scoped_stage::stop` (recording), `resources::attach_profiler` / `release_workspaces`, `log_message`, `parse_log_level`, `find_algorithm`, `array_view::subview`, the factories `host_view` / `device_view` and the queries `backend_available` / `default_backend`. | Accepted: each is a container, builder or recording operation the vocabulary does not cover, or is named by the plan. `buffer::memory_resource()` next to `resources::memory()` is kept: `buffer` follows `rmm::device_uvector`, which names it so. |
| 3 | Naming: `get_` accessors | `get_backend`, `get_profiler`, `get_copy_policy`, `get_options`, `get_log_level` | Accepted: each would clash with a type of the same name (the rule's exception). |
| 4 | Contract uniformity | Both algorithms have `options` (aggregate, appended only), `stats : update_stats` with `batch`, an opaque move-only `result` with `get_options` / `set_options` / `graph_version` / `space` / `clone`, `compute(res, g, inputs..., opt)` and `update(res, g&, batch, r&)`. Differences: `sssp::result<vertex_t, distance_t>` is a template (its arrays hold vertex ids and distances), `cycle_count::result` is not (a histogram of `uint64` counts for every graph type); only sssp has `from_arrays()` (MOSP's trees are read from files); `cycle_count::result::space()` is host on every backend. | Accepted as the rule: a result is a template only when its arrays' types depend on the graph's; importing a result is per algorithm. |
| 5 | Printable enumerations | `to_string` existed for `backend` and `log_level` only; `stats::engine_used`, `algorithm_info`'s fields, `memory_space` and `copy_policy` had no name (the plan's example prints `dyng::to_string(st.engine_used)`; the CLI and the Python layer accept enumerators as strings). | Added `to_string` for `engine`, `determinism`, `memory_space`, `copy_policy`, `algorithm_family`, `container_kind`, `maturity_level`, `oracle_kind`: the enumerators' own names, the manifests' spelling (`09abf17`). |
| 6 | Doxygen | `WARN_AS_ERROR` and `ci/doxygen_coverage.py` passed, but the checker's list of algorithm namespaces had only `dyng::sssp`: `cycle_count`'s `compute()` / `update()` were not checked for `@backends` / `@determinism` / `@paper` (they had them). `sssp::compute()` / `update()` did not list `cuda_error`. | The checker reads the algorithm namespaces from the manifests; `@throws cuda_error` added (`c98f8e9`). |
| 7 | Exception guarantees | PLAN 4.7.3 asks each function to document strong or basic; the headers said "nothing is changed" on some `@throws` lines only. | A `@guarantee` paragraph (Doxygen alias "Exception safety:") on `compute()`, `update()`, `from_arrays()`, `graph::apply()` (strong: the next state is built before it replaces the current one), `dyng::update()` and `update_each()` (strong before the commit, basic after it: the graph holds the new version and the failing result is poisoned; the others are still updated and the first exception is rethrown). `ci/doxygen_coverage.py` requires it on those functions (`c98f8e9`). |
| 8 | Exceptions leaving the library | `dyng::algorithms()` documented `std::bad_alloc`; the header-inline bodies of `dyng::update()` / `update_each()` could let `std::bad_alloc` escape from their own containers. Every exception must derive from `dyng::error` (PLAN 4.7.3). | Both translate `std::bad_alloc` / `std::length_error` to `out_of_memory_error` (`09abf17`, `c98f8e9`). |
| 9 | Host-compilability | The self-containment targets compiled each header alone, but a CUDA build has the CUDA include directories on their command line, so a stray `#include <cuda_runtime.h>` would have passed there. | The generated units fail with `#error` if a CUDA, CUB, Thrust or libcu++ header was included (`6647519`). `stream.hpp` forward-declares `CUstream_st`; no header includes CUDA. |
| 10 | Ownership and ABI | `graph`, `sssp::result`, `cycle_count::result` (unique pimpl) and `resources` (shared handle) keep their layout across releases. The aggregates (`options`, `stats`, `graph_properties`, `batch_semantics`, `apply_summary`, `profiler_options`, `algorithm_info`) change size when a field is appended, which PLAN 5.9 allows before 1.0 (CMake compatibility `SameMinorVersion`). `profiler` holds its vectors inline, so its layout is part of the ABI. | Accepted for 0.x. Before 1.0 (when `abidiff` joins the check, PLAN 5.9): `profiler` moves behind a pimpl. |
| 11 | Macros in public headers | `DYNG_EXPECTS`, `DYNG_FAIL` (planned), `DYNG_CYCLE_COUNT_TYPES_MESSAGE` (the `static_assert` text of cycle_count's supported types; C++17 needs a literal there). | Accepted; all `DYNG_` prefixed. |
| 12 | The registry header (ADR 0022) | `algorithm_info::determinism_level` is not named `determinism` because the type of the same name is in scope; `cite` holds the keys of `docs/references.bib`. | Accepted as frozen. |

### Notes for the Python bindings (M5)

Nothing in the 0.1 headers blocks nanobind, but these need a deliberate binding, recorded here so
M5 does not rediscover them:

1. `dyng::update(res, g, batch, results...)` is variadic. Python's `dyng.update(g, b, r1, r2)`
   takes a run-time list of results of mixed types; the binding builds the type-erased
   participants itself and calls `detail::run_update()` (the same path `update()` takes).
2. The views returned by `result::distances()`, `parents()`, `counts()` and `graph::view()` are
   valid until the next update or apply. `dyng.Array` must keep its owner alive and either check
   the owner's version on access or copy (PLAN 5.4 rule 4 and `copy=False`).
3. `std::initializer_list` overloads (`edge_batch::insert_edge`, `edge_list::add_edge`) are not
   bound; the `array_view` overloads are.
4. `resources` copies share one handle, and a default CUDA stream is per thread: two Python
   threads using one `Resources` run on two streams (the class documentation says so).
5. `profiler` is not thread-safe; the binding holds the GIL while it reads the records.
6. The enumerations accept their `to_string` names (finding 5).

## Decision

1. **The 0.1 API is frozen** as reviewed above: every declaration of `core/*`, `graph/*`,
   `update.hpp`, `sssp.hpp`, `cycle_count.hpp` and the top-level `dyng.hpp`, `citation.hpp`,
   `version.hpp`, `config.hpp`. `io/*`, `generators/*` and `testing/*` are *tracked* (checked the
   same way) and are reviewed and frozen with the CLI and the Python layer in M5. The algorithm
   contract of ADR 0006 is accepted as part of the freeze.
2. **The freeze is a committed listing, not an ABI dump.** `ci/api_snapshot.py` generates a
   listing of every public declaration from the Doxygen XML that `ci/docs.sh` already produces
   (signatures with default arguments, template parameters, bases, public members with their
   initializers, enumerators, type aliases, macro names; no comments, line numbers or bodies) and
   compares it with `cpp/tests/api/api_snapshot/public_api.txt`. `ci/docs.sh` runs the check after
   Doxygen, so the `docs` workflow and the local gate fail on any change the baseline does not
   contain; a changed default value fails too (a breaking change under PLAN 5.9). This is the
   "C++ public-header snapshot (from Doxygen XML)" of PLAN 5.9. `abi-dumper` / libabigail are not
   used before 1.0: most of the API is templates instantiated in the user's code, which an ABI
   dump of `libdyng.so` does not see, and PLAN 5.9 adds `abidiff` at 1.0.
3. **Changing the API** follows "Updating the API baseline" in
   `docs/developer/api_review_checklist.md`: the checklist, a CHANGELOG entry and the `api-change`
   label, then `ci/docs.sh --doxygen-only && ci/api_snapshot.py --update` and the new baseline in
   the same pull request. The `api-check.yml` workflow of M5 adds the label check and `griffe`.
4. **Every function that changes a container or a result states its exception guarantee** with
   `@guarantee`, checked by `ci/doxygen_coverage.py`; host allocation failures leave the library as
   `out_of_memory_error`, also from header-inline code.
5. **Public headers include no CUDA header**, checked by the self-containment targets in every
   build.
6. **Reviewed sketches** of the later algorithms' APIs (`mosp`, the hypergraph with
   `hyperedge_batch`, `triad_count`, `label_propagation`, `hyper_sssp`) are in
   `docs/design/sketches/`, written against the frozen contract. They are not contracts; each is
   frozen by the milestone that implements it.

## Consequences

- An accidental change of a public signature, default, field or enumerator now fails CI with a
  diff instead of reaching a release; a deliberate one carries its own baseline diff.
- The listing depends on the Doxygen version (pinned in `environment.yml`, 1.18.0). A Doxygen
  upgrade that changes how signatures are printed needs one `--update` in its own commit.
- The frozen headers can still gain declarations (PLAN 5.9's additive rules); each addition
  updates the baseline, so the reviewers see it.
- The `profiler` pimpl, the Python items above and the review of `io/*`, `generators/*` and
  `testing/*` are open work for M5 and before 1.0 (the retrospective lists them).
