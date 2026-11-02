# ADR 0092: hyper_sssp's work budget, its fallback and the framework's over_budget hook

- **Status:** Accepted. The semantics are the author's: HD7 (the fallback decision is not a
  parity target; the weaker determinism of `fallback_used` on cuda, through the per-field sentence
  of `<dyng/core/stats.hpp>`) and HD8 (B's triggers kept exactly; `work_budget`,
  `max_iterations`, `on_budget`, `relaxations`, `fallback_reason`), **decided by the author on
  2026-10-28** (GOVERNANCE.md, approvals log, "The parity rules of 0.4 (sheet C1-C8)" and "API
  semantics of 0.4 (sheet E2-E10)"). The technical parts (the framework hook, the host engine's
  counting, the error path, the host-synchronization budget, the tests) are accepted under
  delegation (GOVERNANCE.md, "Delegation of technical ADRs"). Part of M11a step 4.
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD7, HD8, 2026-10-28); the AI assistant, on the author's behalf, for
  the technical parts (M11a, step hyper-sssp-sequential-asis-cuda)
- **Builds on:** ADR 0006 (the algorithm contract), ADR 0043 (the hypergraph's update path), ADR
  0061 (the guarantees after the commit), ADR 0090 (the API), ADR 0091 (the engines).

## Context

MOSP_ESCHER@4b86159 (B) falls back from `hsospUpdate` to `hsospRecompute` when, before a push
round, the relaxations so far exceed `workBudget x max(1, numEntries)` (the live line-graph
entries), when it has run `maxIterations` push rounds, when a distance does not fit the 32-bit
half of its packed word, or when the invalidation's pointer jumping needs more than 64 rounds.
The fallback is exact: only the time changes. Its relaxation count on the GPU depends on the
thread schedule (a push round's `atomicMin` winners decide which words are re-pushed), so near the
threshold the decision can flip between runs; update_stats documents `fallback_used` as
deterministic.

## Decision

### 1. The triggers (HD8)

- **cuda** keeps B's four triggers exactly, in B's order: `work > work_budget x max(1, live
  entries)` checked before each push round; `max_iterations` push rounds (default 4096, B's;
  64-bit); the 32-bit width; 64 jump rounds.
- **sequential** counts B's work units the same way (the rows the seed's pulls read, then the row
  of every expanded hyperedge) and checks the same limit before each expansion of its priority
  queue. It has no rounds, no packed words and no pointer jumping, so only the work trigger exists
  there: on make test the host engine falls back 71 times where B's CUDA engine falls back 73
  (two batches near B's iteration or width limits), with identical results (the decision is
  tabulated, not compared: HD7).
- `fallback_reason` names the trigger (`work_budget`, `iterations`, `distance_width`,
  `invalidation_depth`); `relaxations` is the count the budget saw.
- The recompute's own round cap is max(max_iterations, id bound + 1), so a fallback always
  converges.
- **Pinned by tests** (added in the M11a review, which found that a 1.5x work limit and an
  off-by-one iteration cap passed every test and replay): on a path whose batch lowers one
  hyperedge per push round, an update's work at about 1.25x the limit falls back and at about
  0.8x does not (both backends), and on cuda `max_iterations = k` push rounds (the rounds the
  update needs) does not fall back while `k - 1` does (`HyperSssp.TheWorkTrigger...`,
  `HyperSssp.TheIterationTrigger...`). The make-test sessions on cuda by id, whose work lies far
  from the budget, also fail the result replay if a fallback decision differs from B's twice
  (HD7's repeat; elsewhere the decision is tabulated).

### 2. The framework hook `over_budget(ctx)`

A Tier A problem may provide `over_budget(ctx) -> bool`, asked by `update_enactor` after every
`loop` call of Step 2 (never by `compute()`): true ends Step 2 through the `recompute` hook in the
stage `recompute`, as the convergence cap under `on_limit::fallback_recompute` does, and sets
`fallback_used`. A problem that provides it must provide `recompute` (a compile-time check in
`framework/conformance.hpp`). It is documented in `docs/developer/framework.md` with the fake of
ADR 0091 section 1 as its test (the budget after k loops, recompute and throw). The Tier B engine
(cuda) decides inside `enact_fused`, as B does, and opens the `recompute` stage itself on every
update (0 ms without a fallback), so the region map's sosp_update row is the same on both.

### 3. `budget_action::error`

`on_budget = error` governs the work and iteration triggers only (the width and the depth always
recompute, as B's): the update throws `convergence_error` after the commit, so the hypergraph
holds the new version and the result is poisoned (`stale_result_error` on every later use until
it is recomputed): the basic guarantee of ADR 0061, documented in the header. The tests force it
on both backends (`TheWorkBudgetRecomputesOrThrows`: work_budget 0 recomputes or throws,
+infinity never falls back, max_iterations 1 on cuda).

### 4. The determinism of `fallback_used` (HD7)

`<dyng/core/stats.hpp>` gains the per-field sentence of HD7 (ADR 0090 section 7), and
`hyper_sssp::stats` states that `fallback_used` and `fallback_reason` are deterministic on
sequential and schedule-dependent on cuda near the budget, with identical results; `relaxations`
and `iterations` are schedule-dependent on cuda. On the golden set the cuda decisions equal B's on
all 685 updates and on ten repeated runs; the gate rounds guard them (a round whose decisions
differ from the golden's is repeated, ADR 0080).

### 5. The host-synchronization budget (I9)

The cuda engine reads its counters once per push round and once per jump round. The problem's
`algorithm_budget` declares at most `jump_rounds + 2 + iterations + 1 + 3` synchronizations per
update (plus `2 + 2 x recompute rounds` after a fallback), counted by the engine's one
synchronization helper and checked by the conformance kit (C8) and under `DYNG_DEBUG_BUDGETS`.
The last 3 are a growth of the result's device state (when the store's node capacity grew, or on
the first update of a clone): the new state's fills, the copy of the old state, and the release
of the old state's pinned counters (the pinned resource synchronizes before it frees). The first
version allowed 2, so a growing update threw `internal_error` under strict budgets (M11a review);
`HyperSssp.AGrowthOfTheResultsStateStaysWithinTheBudget` grows the state under strict budgets.

## Consequences

- A fallback is never silent: `fallback_used` and `fallback_reason` say why.
- The sequential and cuda backends can disagree on the decision, never on the result.
- Any later Tier A algorithm with a work budget uses the same hook.
