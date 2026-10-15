# ADR 0043: The hypergraph's update path and the conformance kit's container axis

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8b step 3 (PLAN 4.5.1, 4.5.5 I1 and I9, 8.2, 8.4; the M8b acceptance
  criteria 10 and 11; ADR 0023 amendment A1) and changes no rule the author approved: the frozen
  `update.hpp` and the graph headers stay byte-identical to main, the parity rules, the gates and
  their protocol are untouched, and the graph suites of the kit keep their behaviour and test
  counts.
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step update-kit-scaffold)

## Context

ADR 0023 A1 made `dyng::update()` dispatch through the class template
`detail::participant_of<container_t>`, so that the hypergraph could join without editing the
frozen `update.hpp`. ADR 0041 left the applied record as container state "until dyng::update
(step 4) moves it into its transaction" (its deviation 12) and gave `hypergraph::apply()` one
body from validation to commit. The framework's adapter (`problem_participant`), its
`expect_current_result` / `stamp_result` and the conformance kit (C0-C12, ADR 0022) were written
for `graph<V,E,W>`: the kit's case names read `edge_type`, its registration needs an
`edge_batch_view` and an `apply_summary`, C0 asserts `container_kind::graph`, its presets, model,
eight batch mixes and C7/C9/C11 are graph-specific. M9's `triad_count` needs all of it on the
hypergraph: a problem that subtracts on G_t through the deleted ids and adds on G_{t+1} through the
inserted ids, run by `dyng::update()`, checked by the kit.

## Decision

### 1. The hypergraph transaction (`cpp/src/hypergraph/transaction.hpp`)

`detail::hyper_transaction<V,W>` is the container's internal transaction of PLAN 4.5.1:

| Phase | What it does | Guarantee |
|---|---|---|
| **begin** (constructor) | placement and poisoning checks; incidence edits refused (0.4); the batch staged to the host once under the copy policy; **Step 0 once**: inserted rows sorted and deduplicated (dropped members counted), weights checked, deletions checked against G_t (dead, out-of-range, negative and repeated ids skipped and counted) into `hyper_staged_batch` | strong: nothing changes |
| **before()** | an `old_view` of G_t (the internal row views, h2v, h2h of G_t stay valid) | – |
| **commit()** | the store phase in the order of `id_reuse`, the line graph, one read of the device error words, the new row views; then the no-throw commit (version, state identity, weights, the applied record). Refused if the hypergraph changed since begin (another apply) or the transaction committed already | basic: a store error poisons the hypergraph |
| **after()**, **applied()** | a `new_view` of G_{t+1}; the applied record | – |

`hypergraph::apply()` is a transaction without participants (its stages are unchanged:
`hypergraph.apply` around begin and commit). `before()` after the commit throws
`internal_error`; the `old_view` itself checks its version in Debug builds (invariant I1), so a
problem that keeps G_t past the commit fails there (tests
`HypergraphUpdate.AnOldViewUsedAfterTheCommitThrowsInDebugBuilds` and
`...AProblemThatKeepsItsOldViewPastTheCommitFailsInDebugBuilds`). The transaction copies the
resources handle (shared, cheap): a caller's temporary must not dangle.

**The applied record stays container state** (ADR 0041 deviation 12 is resolved this way, not by
moving it): `applied()` returns the record of the commit it made, which is the hypergraph's last
record. Moving it into the transaction would take it from `hypergraph_access::applied()`, which
the parity tools and M11's incidence-delta hook read after a plain `apply()`. It carries the
inserted ids on the host and (CUDA) the device, the deleted ids with their member sets, the
touched hyperedges and vertices, the skipped counts and the incidence-delta hook (unchanged from
ADR 0041 section 10).

### 2. `dyng::update()` over hypergraphs

- `<dyng/hypergraph/hypergraph.hpp>` includes `<dyng/update.hpp>` and specializes
  `detail::participant_of<hypergraph<V,W>>` with the participant interface
  `detail::hyper_update_participant<V,W>` (`target`, `use_normalized(const
  hyper_staged_batch<W>*)`, `before_apply(res, h, batch)`, `after_apply(res, h,
  hyper_apply_summary, hyper_applied)`, `poison`, `reads_prepared_graph`), its batch view
  `hyperedge_batch_view<V,W>` and `run()`. `update.hpp` and `cpp/include/dyng/graph/*` are
  unchanged.
- `detail::run_update(res, h, batch, participants, n, commit_stage)` for hypergraphs
  (`cpp/src/hypergraph/hyper_update.cpp`): duplicate targets rejected; the transaction's begin
  under the stage `<algo>.normalize` (`update.normalize`); the staged batch handed to every
  participant; the line graph made sure of when a participant `reads_prepared_graph()` (built on
  G_t under `line_graph_policy::automatic`, an `invalid_argument_error` under `none`, both before
  anything changes); every `before_apply` on G_t; **one** commit inside `budget_scope`
  (`<commit_stage>` around `hypergraph.apply`; container growth is reported through
  `last_commit_counts()`, never failed); every `after_apply` on G_{t+1}, a failing participant
  poisoned while the others still run, the first exception rethrown.
- **Preparing the views.** The internal row views (h2v and, when maintained, h2h) are rebuilt by
  every commit; the public views stay lazy (never built on the apply path, ADR 0042). So
  "prepare" for hypergraphs means the line graph only, and `reads_prepared_graph()` means "reads
  the line graph" (default false for hypergraph problems, true for graph problems as before).
- `detail::is_hypergraph_v<T>` (public, in `hypergraph.hpp`) lets a hypergraph algorithm's
  `update_traits` say in plain English that it takes a hypergraph (the scaffold's template does).

### 3. The framework

`framework/composition.hpp` serves both containers: `composition_detail::graph_types<container>`
names, per container, the participant interface, the batch view, what the hooks see before
(`hyper_requested_batch {hyperedges, normalized}`) and after (`hyper_applied_batch {summary,
record, normalized}`) the commit, Step 0's normalized batch, the summary type of `stats::batch`
(`has_batch_summary` takes it) and the record `after_apply` receives. `problem_participant`,
`make_participant` and `update_one` are unchanged in shape. `expect_current_result` and
`stamp_result` read the state identity through `state_identity()` overloads, and the messages
name the container ("hypergraph version 3"); the graph's messages are byte-identical.

### 4. The kit's container axis

`container_traits<container_t>` (`cpp/tests/conformance/containers.hpp`; the hypergraph's in
`hypergraph_generators.hpp`) gives the checks the container's properties and presets, its host
model and batch mixes, the batch and summary types, the summary's counters, what an empty batch
must leave at zero, the model check, the inverse of a batch (C5), the composition presets and
mixes (C10), and the mixes of C11 and C12. What is too different to be data, C7's invalid batches
and skip rules and C9's summary arithmetic, is `container_checks<case_t, container_t>`; the
graph's bodies moved there verbatim. A test_traits may name `container_types` (else
`graph_types`, the name of 0.1-0.2), `registry_entry()` (a test-only algorithm the library does
not register) and `test_partners` (C10's test-only partners).

- **Graph suites unchanged.** Same mixes, seeds, presets, checks and order of steps; the test lists
  of every graph conformance executable (host and CUDA) are identical to the branch base's, and
  they pass.
- **The hypergraph presets:** {erase_first, insert_first} x {escher_gpu(), mosp_escher(), the
  default}, each with the traits' `require()`, plus the traits' extras.
- **The hypergraph model** keeps the live hyperedges by id and the id bound. A batch's deletions
  name live ids (or, in the skip-rule mixes, dead, out-of-range, negative and repeated ones); the
  inserted ids are the container's, recorded from `hyper_apply_summary::inserted_ids` by
  `observe()` after the step, never predicted. The inverse of a batch (C5) is built from that
  summary and the deleted member sets: delete the inserted ids, insert the deleted hyperedges
  again.
- **The mixes** (criterion 11): insertions only, deletions only, mixed, delete-then-reuse within
  one batch, duplicate members (and copies of live hyperedges), dead-id deletions, empty,
  delete-all, re-insert, a batch larger than the hypergraph, ids at the index limits (members 0
  and n - 1, the largest live id, INT64_MIN / INT64_MAX deletions), skewed cardinality (>= 1,024
  on the medium shapes). `HypergraphKitGenerators.EveryMixMakesWhatItsNamePromises` checks that
  each mix makes what its name says (M6b: a kit tests only what its generator makes).
- **C7 (hypergraph):** a member below 0 or at n, an empty hyperedge, malformed offsets, a bad
  weight (weights on an unweighted batch) are rejected with `invalid_argument_error` and an
  incidence edit with `not_supported_error`, each before anything changes (version, live count,
  bound, result); dead, out-of-range, negative and repeated deletions and repeated members are
  counted.
- **C9 (hypergraph):** applied + skipped == requested; `ignored_deletions` and `dropped_members`
  equal what the generator put in; `skipped_ops` their sum; the inserted ids distinct, below the
  bound, each holding the inserted hyperedge (read back through `h2v()`).

### 5. The test-only fakes and the planted bug

`cpp/tests/framework/fake_hyper_problems.hpp`: `hyper_pairs_problem` (the overlapping pairs of
live hyperedges, the edges of h2h) and `hyper_squares_problem` (the sum of |e|^2), both
aggregate delta with `ownership::min_member`: count(-) on the OLD row views with frontier = the
deleted ids (Step 0's, in `before_apply`), count(+) on the NEW row views with frontier =
`inserted_ids` (on CUDA the applied record's device copy, read by a kernel). Their kit
(`dyng_hypergraph_kit_tests`, `..._cuda_tests`) runs every applicable check on sequential, openmp
and cuda: C5 with the inverse from the summary, C10 with the two fakes on one container, C11, C12
on two non-blocking streams, C8 with 0 synchronizations on the host and 2 on CUDA. C4 is skipped
(one engine). The planted bug `DYNG_MUTATION_KIT_FRONTIER_REQUESTED_IDS` (frontier =
`id_bound(G_t) + i`, the ids the request would get without reuse) fails C2, C5 and C7 (and C8 in
budgets builds) on every backend; it is the CTest `conformance.mutation.frontier_requested_ids`
and the suite `kit` of `parity/mutate.py`.

### 6. Gaps the kit exposed, fixed or recorded before the freeze

| # | Finding | Resolution |
|---|---|---|
| 1 | The host stores built the lazy public views (h2v, v2h, h2h) and the `automatic` line graph without `container_scope`, so an algorithm reading `h2v()` in its algorithm phase was charged the container's allocations (C8), while the CUDA store counted them as container work. | Fixed: the host store counts them as container work too (as ADR 0022's container work of graphs). |
| 2 | A hypergraph algorithm's result passed with a graph had no plain-English error. | `detail::is_hypergraph_v`; the scaffold's `update_traits` static_asserts it (compile-fail test `hypergraph.update.static_assert.hypergraph_result_with_graph`). |
| 3 | The frozen `update.hpp`'s static_assert text for a wrong batch names "edge_batch_view<V, W> of the graph's vertex and weight types" also for a hypergraph. | Recorded, not changed: `update.hpp` is frozen and must stay byte-identical; the first half of the message ("the batch must be the container's batch view") is right for both (compile-fail tests for the owning `hyperedge_batch` and for each container's batch view with the other). |
| 4 | A graph algorithm's result passed with a hypergraph stops inside the graph algorithm's frozen `update_traits` (`no type named 'edge_type'`), not at a plain-English message. | Recorded (sssp.hpp is frozen); a compile-fail test pins that it does not compile. |
| 5 | `hypergraph::apply()`'s validation, staging and commit were one function; Step 0 could not be shared. | The transaction (section 1). |
| 6 | The kit could not run an algorithm the library does not register. | `test_traits::registry_entry()`; C0 still compares the traits with that entry. |

## Consequences

- `dyng::update(res, h, batch, r1, r2)` and `update_each` work for hypergraph algorithms with one
  Step 0 and one commit; M9's `triad_count` plugs in through `make_participant` like a graph
  algorithm, and its traits through the kit's hypergraph axis with no kit change
  (`ci/scaffold_check.sh` proves it with two scaffolded hypergraph algorithms).
- `<dyng/hypergraph/hypergraph.hpp>` gains `detail::hyper_update_participant`, `run_update`,
  `participant_of<hypergraph>` and `is_hypergraph`; the API baseline records them (tracked until
  the freeze of step 5).
- The stage names of a hypergraph update: `update.normalize` (or `<algo>.normalize`),
  `update.commit` (or `<algo>.commit`) containing `hypergraph.apply` and its sub-stages.
