# ADR 0090: hyper_sssp's draft API, its names and the vertex-distance rule

- **Status:** Accepted. The names and the semantics are the author's: HD8 (the budget options and
  `budget_action`), HD11 (the distance semantics and the weight limit), HD12 (the new public names),
  HD13 (the engines), HD14 (no openmp backend) and HD17 (frozen at the end of M11b, stable at
  0.4.0 only on the author's confirmation), **decided by the author on 2026-10-28** (GOVERNANCE.md,
  approvals log, the rows of 2026-10-28). The technical parts (the header's layout, the result's
  internals, the files, the checks) are accepted under delegation (GOVERNANCE.md, "Delegation of
  technical ADRs"). The public names the decided list does not spell out are listed for the
  author's acknowledgement (section 6; GOVERNANCE.md, Open decisions, M11 (HD)). It carries out
  part of M11a step 4 (PLAN 4.8, 5.3; the M11a acceptance criteria 13 and 14's API parts).
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD8, HD11-HD14, HD17, 2026-10-28); the AI assistant, on the author's
  behalf, for the technical parts (M11a, step hyper-sssp-sequential-asis-cuda)
- **Builds on:** ADR 0006 (the algorithm contract), ADR 0041 (the hypergraph container), ADR
  0047 (the frozen hypergraph API), ADR 0060 (triad_count's API review: the result and the
  guarantees), ADR 0080 (the parity modes), ADR 0082 (the oracle), ADR 0086 and 0089 (slack_csr
  storage).

## Context

PLAN 5.3 sketches `hyper_sssp`: a source vertex, distances by hyperedge, `distance_to(v)` as "the
minimum over the hyperedges h of v of D(h) + w(h)", options with an int64 "frontier visits"
budget. MOSP_ESCHER@4b86159 (B) computes D(h) with w(h) already in it (`HostHypergraph.cpp:386`,
`HypergraphOracle.cpp:124`), runs from a virtual source row, and configures its update with
`UpdateConfig {workBudget, maxIterations, blockSize}`. The decision sheet of 2026-10-28 fixed the
semantics and the names; this ADR records the header that implements them. It is a draft: M11b
adds the shared sssp engine (`engine::operators` on cuda), the Python layer and the CLI, and
freezes the header at its end (HD17).

## Decision

### 1. The header `<dyng/hyper_sssp.hpp>` (namespace `dyng::hyper_sssp`)

- `compute(res, h, source, opt = {})` and `update(res, h, batch, r)` on
  `hypergraph<int32_t, int32_t>` (other types fail to compile with a static_assert naming the
  supported one); `dyng::update()` composes results through `update_traits`.
- `options` (HD8): `double work_budget = 1.0`, `std::int64_t max_iterations = 4096`,
  `budget_action on_budget = recompute`, `engine cuda_engine = automatic`,
  `bool validate_inputs = true`. Every field is a tunable (`result::set_options()`).
- `budget_action {recompute, error}` and `fallback_reason {none, work_budget, iterations,
  distance_width, invalidation_depth}` with their `to_string()` overloads.
- `stats : update_stats` (HD8, HD12): `batch`, `line_graph_insertions`, `line_graph_deletions`,
  `invalidated`, `relaxations`, `fallback_reason`; each field states its determinism level, and
  `fallback_used` is stated weaker on cuda (ADR 0092 section 4).
- `result<vertex_t, distance_t = std::int64_t>` (move-only, opaque): `source()`,
  `hyperedge_distances()` (int64), `hyperedge_parents()` (**int32**: slack_csr ids are below 2^31,
  ADR 0089; the parents' width is B's), `distance_to(v)`, `vertex_distances()`, `get_options()`,
  `set_options()`, `graph_version()`, `space()`, `clone(res)`: the accessors of the 0.3 results
  (ADR 0060), so a user moves between them without learning new names.

### 2. The vertex distances (HD11; supersedes PLAN 5.3's "+ w(h)")

`distance_to(v)` is the minimum of D(h) over the live hyperedges h that contain v, 0 for the
source, infinite when none is reachable. D(h) already includes w(h), so nothing is added: PLAN
5.3's "+ w(h)" would count the last hyperedge's weight twice. The rule is the sketch's and B's
(`distance_to(target == source) = 0`). `vertex_distances()` is computed on request, once per
hypergraph version, from the host v2h rows of the hypergraph the result matches (the sketch's open
question 1: not maintained by update()). The result keeps a weak reference to that hypergraph (a
lifetime token of the container's implementation, `hypergraph_impl::lifetime`); when it is gone or
at another version, both throw `stale_result_error` instead of reading freed or newer rows.

### 3. The requirements and their checks

Weights in [1, 2^28] (B's limit: the packed word's 32-bit distance with headroom; the container
keeps [1, 2^31 - 1]), checked by `compute()` over the live hyperedges and by `update()` over the
batch's insertions **before the commit** (strong guarantee; `validate_inputs = false` skips the
check, and heavier weights then fall back with `distance_width`). slack_csr storage with its line
graph and `store_v2h`, else `not_supported_error` / `invalid_argument_error` before anything
changes. The checks run in `hyper_sssp.cpp` (`expect_backend`, `expect_options`,
`expect_hypergraph`), shared by both engines.

### 4. The backends (HD13, HD14)

sequential (B's host emulation, `sequential.cpp`; `engine::operators`) and cuda (B's kernels,
`engine::fused`; ADR 0091). openmp resources raise `not_supported_error` whose message names
sequential and cuda and the fix `dyng.Resources("sequential")`. The rule is the one HD14 records
jointly with M10's AD19, written in M10a's guard ADR and cited here, not restated as a new rule:
"an algorithm accepts the backends its manifest lists; any other raises not_supported_error naming
them" (it is not an amendment of ADR 0010, whose line 54 concerns batch presets).
`options::cuda_engine = engine::operators` throws `not_supported_error` until M11b.

### 5. The files

`cpp/include/dyng/hyper_sssp.hpp` (the draft API, included by `<dyng/dyng.hpp>`, tracked by the
API snapshot), `cpp/src/algorithms/hyper_sssp/hyper_sssp.cpp` (the checks, the hooks, the result),
`problem.hpp` (the framework problem), `sequential.cpp`, `cuda.cu` (dynG's driver of the engine)
and the transplanted `engine.cu`, `kernels.cuh`, `state.cuh` (ADR 0091). The driver and the
transplant are separate files so that `transplant.py --check` compares only B's code with its
hunks.

### 6. Names for the author's acknowledgement

Beyond HD12's list, the header adds: `to_string(budget_action)` and `to_string(fallback_reason)`
(the convention of every dynG enumeration), the result's `get_options`, `set_options`,
`graph_version`, `space` and `clone` (the "..." of HD12's result entry, spelled out), the
enumerators `budget_action::recompute` / `error` (HD8 names the values, not the spelling), and
the macros `DYNG_HYPER_SSSP_TYPES_MESSAGE` and `DYNG_HYPER_SSSP_CONTAINER_MESSAGE` (the
static_assert texts, as `DYNG_TRIAD_*` in 0.3). They are listed under GOVERNANCE.md's Open
decisions (M11 (HD)); a rename before the freeze is an edit of the draft.

### 7. The determinism sentence of `<dyng/core/stats.hpp>` (HD7, joint with M10's AD20)

The last paragraph of `update_stats`'s class comment reads: "An algorithm's stats documentation
may state a weaker level (deterministic per backend, or schedule-dependent) for an inherited
field; that statement governs for that algorithm." M10a step 3 owns the sentence; M11a writes the
identical text at the same place because M10a had not merged, so the two branches agree on it
(CHANGELOG, api-change: documentation).

## Consequences

- A user of 0.3's results finds the same accessors; the new surface is the options, the stats
  and the two enumerations.
- The rule of section 2 is the documented one; the oracle (ADR 0082) and the tests use it.
- The API snapshot tracks the draft from this step; M11b freezes it.
