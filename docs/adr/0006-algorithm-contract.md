# ADR 0006: Algorithm contract

- **Status:** Proposed (draft, M1a); accepted with the 0.1 API freeze (M3)
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

The six originals expose their algorithms in six ways: file-path functions that print to `cout`
(`sequentialSOSPUpdate`, `parallelSOSPUpdate`), in-memory drivers with global timers
(`mospUpdate`), CLI-only entry points, `bool` + `cerr` error reporting and `exit()`. A user who
keeps several results on one changing graph, and the conformance kit that tests every algorithm
the same way, need one shape. `sssp` is the first algorithm that implements it (M1a); the
contract is frozen with `cycle_count` in M3.

## Decision

1. **One header per algorithm** (`<dyng/<algo>.hpp>`, namespace `dyng::<algo>`) with `options`,
   `stats`, `result`, `compute()` and `update()`.
2. **`options`** is an aggregate with defaults; fields are only appended. Inputs fixed at
   `compute()` (for sssp: the source and `objective`) cannot change later; tunables (for sssp:
   `delta`, `cuda_engine`, `validate_inputs`) change with `result::set_options()`.
3. **`result`** is opaque (pimpl) and move-only; `clone(res)` is the deep copy. It owns the
   property arrays, the options and the **graph version** it matches. (M1a: also the engines'
   workspace. Since M1b the scratch memory belongs to the `resources` handle and is leased per
   run, so results run through one handle share it; ADR 0015.) Accessors return `array_view`s. `result::from_arrays()` adopts an existing
   result (for sssp: trees read from MOSP's files), optionally canonicalizing it, and validates it
   when `options.validate_inputs` is set.
4. **`compute(res, container, inputs..., options) -> result`** never mutates the container.
5. **`update(res, container&, batch, result&) -> stats` applies the batch and updates the result
   in one call.** It throws `stale_result_error` if `result.graph_version() != g.version()`,
   or if the result was computed on another graph state (see *Graph identity* below).
   Validation (batch ids and weights, the result's version) happens before the container changes
   (strong guarantee). An error after the commit, which only corrupt imported inputs can cause,
   leaves the result *poisoned*: every later use of it (`update()`, `distances()`, `parents()`,
   the options, `clone()`) throws `stale_result_error` until it is recomputed (PLAN Section
   4.7.3); only the `noexcept` queries `source()` and `graph_version()` still answer. Both CPU
   backends detect the only such error, a parent cycle in an imported tree that no root of the
   batch breaks, with the same check and message.
6. **Several results on one graph** go through `dyng::update(res, g, batch, r1, r2, ...)` (a
   tuple of stats) or `dyng::update_each(res, g, batch, list)` (a run-time list, e.g. the K
   objectives of a multi-objective graph). Both run every result's before-apply work on G_t, one
   commit, then every result's update on G_{t+1} (`detail::run_update`); a result may appear only
   once.
7. **`stats`** derives from `update_stats` (`affected`, `iterations`, `frontier_visits`,
   `fallback_used`, `converged`, `engine_used`) and appends algorithm counters; each field is
   documented as *deterministic* (asserted by tests, compared with the original where it has one,
   e.g. sssp `invalidated`) or *schedule-dependent* (logged only). Timing is never a statistic;
   it goes through the profiler with the stage names `<algo>.<hook>`.
8. **Oracle kind per algorithm** (manifest field `oracle`): `compute` for exact algorithms
   (`update ∘ … ∘ update == compute`, bit for bit where the determinism level is `bitwise`), or
   `reference` for approximate ones. sssp is `compute` + `bitwise`: canonical trees (lowest-id
   parent ties) identical on every backend. The oracle holds from a canonical start (every
   `compute()` result, `from_arrays(..., canonicalize = true)`); see *Tie rule* below for trees
   adopted with `canonicalize = false`.
9. **Backends:** the sequential backend is mandatory; a missing backend throws
   `not_supported_error` naming the available ones.

### Tie rule of sssp (added after the M1a review)

`sssp::result::from_arrays(..., canonicalize = false)` adopts a valid shortest-path tree whose
tie parents need not be the lowest ids (MOSP's `mosp` driver works on such trees, and the
roadNet-CA dataset trees are such trees). The two update rules of the original then disagree:
`sospUpdateCpu` (MOSP-OpenMP) and `sospUpdateGpu` (MOSP-CUDA) only let a vertex adopt a smaller
(distance, parent id) pair offered by a vertex whose distance decreased, while the legacy
`sequentialSOSPUpdate` re-scans all in-neighbours of every candidate and also switches to a
lower-id tight parent that did not improve. The M1a sequential backend inherited the second rule,
so it disagreed with the OpenMP backend (and with `mosp`) on 26 of about 570 such random cases
and on roadNet-CA (1-2 parent lines per objective), and the documented postcondition
"update equals compute exactly" was false for such trees.

**Decision:** one rule for every backend, the push rule of `sospUpdateCpu` / `sospUpdateGpu`,
because that is what the paper engines and the `mosp` driver compute and what the CUDA port
(M1b) will reproduce. The sequential backend keeps the seed pull pass (identical in both
originals) and applies the push rule round by round in Step 2, and in the distance-only mode
(`stats::packed_parents == false`) it recovers every parent with the lowest-id rule, as the
OpenMP unpack does. Consequences: (a) every backend returns the same tree on every valid input
tree; (b) from a canonical tree, `update()` equals `compute()` exactly (the conformance oracle);
(c) from a non-canonical tree, the distances equal `compute()`'s and the tree is a valid
shortest-path tree, but kept tie parents can differ from `compute()`'s (documented in
`<dyng/sssp.hpp>` and `docs/algorithms/sssp.md`); (d) dynG's sequential backend no longer equals
the original `sequentialSOSPUpdate` on non-canonical trees (on canonical trees, and so on every
golden case, all three originals and both dynG backends agree). The rejected alternative, to
require canonical input trees (canonicalize always, or reject non-canonical trees), would have
broken byte parity with `mosp` on the MOSP datasets. Tests: `NonCanonicalInputTreesAgreeOnEveryBackend`
(random tie perturbations, canonicalize = false, both backends), the review's hand case, and a
distance-only case; the golden group `noncanonical` (parity/goldens.toml) replays perturbed trees
against `mosp`.

### Graph identity (added after the M1a review)

PLAN Sections 5.1 and 5.2 define the version as "+1 per applied batch", starting at 0. Every
graph therefore starts at version 0, and a check of the version alone accepted a result of graph
A in `update(res, B, batch, r)` whenever B had the same vertex count and version, and a result
kept across a reassignment `g = graph::from_edges(...)`; the update then returned a tree that was
silently wrong. The version keeps its plan meaning (a per-graph counter, +1 per applied batch,
reported by `graph::version()` and `result::graph_version()`), and the graph state additionally
carries a **process-wide unique state identifier** (`detail::graph_impl::state_id`, never 0):
every construction and every applied batch draws a new one, `clone()` copies it (the content is
identical), moves keep it. A result records the identifier of the state it matches and `update()`
checks both. So a result may be updated on a clone of its graph, but not on another graph, nor on
a reassigned graph variable, nor on a clone that has since diverged. The identifier is internal
(not part of the public API or the Python bindings); only the extra `stale_result_error` cases
are.

## Consequences

- `graph::apply()` stays public for users who maintain only the structure; results left behind
  are detected as stale instead of being silently wrong.
- The multi-result update needs an internal participant interface
  (`detail::update_participant`: before_apply / after_apply / poison); M3 turns it into the
  framework's composition (`run_update(ctx, g, batch, problems...)`).
- `update_each()` is an addition to the plan's Section 5.1 sketch, needed because the number of
  objectives is known only at run time.
- (M1a) Each sssp result owned its workspace (about 38 bytes per vertex for the OpenMP engine),
  so K results used K workspaces where MOSP-OpenMP shares one; the M1a OpenMP A/B measured the
  cost (`parity/results/M1a.md`). **Decided in M1b (ADR 0015):** the `resources` handle owns a
  workspace pool; `compute()` and `update()` lease a workspace per engine run, so the K
  objectives of `update_each()` share one, as in MOSP, and a steady-state update allocates no
  scratch memory.
- The contract is exercised by one algorithm so far (sssp, two CPU backends); M2 (`cycle_count`,
  an aggregate-delta algorithm with oracle kind `compute`) is the second user, and M3 accepts
  this ADR only after both run through the extracted framework.

## Alternatives rejected

Passing the parameters again on every `update()` (state and parameters could disagree); a
`session` class (the result already owns its state, and scratch memory belongs to `resources`,
ADR 0015); letting users call `g.apply()` and then
`update()` (the library could not enforce the G_t / G_{t+1} order that counting algorithms need).
