# ADR 0076: The conformance kit's float slotted axis, label_propagation's traits and compare()

- **Status:** Accepted under delegation (2026-10-30; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 5 (kit-mutations-gates-certificate; PLAN 8.1-8.2; the
  M10a acceptance criteria 14 and 15), in the ADR 0043 pattern (the container axis). The rules it
  applies are the author's, **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals
  log): AD7, the harmonic oracle at 2000 delta with delta 1e-6 in CI and the class rule of 3m (the
  rows "The parity rules of 0.4 (sheet C1-C8)" and "The AD7 margin on C"), AD20, the per-field
  determinism levels (the row "API semantics of 0.4 (sheet E2-E10)"), AD11's scope of vertex
  batches; no gate, tolerance or parity rule changes. The graph and hypergraph suites of the kit
  keep their behaviour and test lists.
- **Date:** 2026-10-30
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  kit-mutations-gates-certificate)

## Context

Until this step label_propagation was registered with the kit (C0 against the manifest) without
running it (ADR 0072): its graphs are `graph<int32, int32|int64, float>` objects with
`row_layout::slotted`, its batches vertex batches under `batch_semantics::net_effect()` (ADR
0069), which the graph axis (compact graphs, edge batches, eight edge mixes) does not make; its
oracle kind is `reference` (the harmonic solution), its determinism level `tolerance` with the
damped schedule bitwise per backend; its compute() needs seeds the graph does not hold. The kit's
graph axis is keyed by the container type, and the slotted graphs have the graph's type.

## Decision

### 1. The axis key (additive kit changes)

- A test_traits may name `template <typename graph_t> using container_axis = ...;`: the kit then
  reads `graph_t` through `container_traits<container_axis<graph_t>>` (`kit_detail::axis_key_of`,
  `axis_of`), and `container_checks` is selected by that key. Every existing traits names none, so
  its axis, case names and checks are unchanged.
- `slotted_axis<graph_t>` (`cpp/tests/conformance/slotted_generators.hpp`): the axis of
  label_propagation. The other optional hooks the kit gained, each defaulting to the old behaviour:
  `affected_deterministic` (false: `stats::affected` is left out of the counters, AD20),
  `compare_across_backends()` and `deterministic_across_backends()` (C3 and C12: a weaker level
  across backends than per backend), `container_traits::stale_probe()` (C11: a batch valid after
  the separate `g.apply()` where a batch cannot be given twice: vertex ids are single-use).
- C8's last line compares `compute()` with the update chain only for `oracle_kind::compute`; for
  `reference` it checks the chain against the reference (DynLP and IrLP are both near the harmonic
  solution, not equal to each other).

### 2. The slotted axis

| Part | Rule |
|---|---|
| Full graph | `generators::legacy::dynlp_vertex_arrival_stream` at average degree 5 (LabelPropagation-CUDA's graphs: weights in [0.01, 1], intra-class edges in [0.5, 1]); the graph is `graph::with_capacity()` of its row capacities |
| Shapes | 5,000 (tiny), 6,000 (small) and 8,000 (medium) vertices: C2's graphs come from the DynLP generator at 5,000 or more (M10a criterion 15) |
| Model | the state of every vertex (inactive, active, deleted) and of every full-graph edge (not delivered, delivered, pending, dropped); 70 % of the vertices present at the start (`build()`: with_capacity, one batch with every arrived vertex and every delivered edge, one with the departed vertices) |
| Seeds | `kit_seed_label(v)`: a hash of the id, about one vertex in 16, class 0 or 1; it depends on the id only, so a traits' compute() can name the active seeds of any graph of the axis without the model |
| Mixes | `arrival_with_labels` (4 % of n arrive with their edges to present and past vertices and among themselves, one in eight live edges withheld as pending, at least one labelled arrival), `departure` (1.4 % of n), `mixed`, `insert_and_delete_same_batch` (a quarter of the arrivals deleted in the same batch: they end deleted, their edges dropped), `edges_between_present` (half of the pending edges, no vertex operation) |
| Departures | active vertices without a seed label, the original generator's rule (its phase C): see section 5 |
| C7 | edge deletions (`not_supported_error`); ids and endpoints out of range, re-insertion of an active and of a deleted id, an id twice in the insertions and in the deletions, a labelled vertex deleted in the same batch, a label on a vertex not inserted, a label other than -1 / 0 / 1 (`invalid_argument_error`), each before anything changes; a row overflow (`capacity_error`: refused before anything changes on the host, found on the device after the commit on CUDA, which poisons the graph and refuses the next batch); bad options; an edge to a deleted vertex is dropped and counted twice |
| C9 | `inserted_vertices` = \|Ins \ Del\|, `deleted_vertices` = \|Del ∩ V_t\|, the appended and dropped entries from the model, applied + dropped = 2 x the requested edges, the active count |
| C10 | the one preset, mixes arrival, mixed, insert_and_delete |

`LabelPropagationKitGenerators.EveryMixMakesWhatItsNamePromises` checks each mix (a kit tests
only what its generator makes, M6b).

### 3. label_propagation's traits (`label_propagation_traits.hpp`)

Two suites: `label_propagation` (`schedule::in_place`, the default) and the test-only tag
`label_propagation_damped` (`schedule::damped`; `registry_entry()` is label_propagation's), each
the other's C10 partner, so one `dyng::update()` carries a damped and an in_place result.

| Item | Rule |
|---|---|
| options | delta 1e-6 (AD7 in CI), tau 0.5, the suite's schedule; compute() reserves the kit's batch sizes (`result::reserve`) |
| near_reference (C2, C8) | the harmonic solution of the current graph (`testing::harmonic_solution` over `to_csr()`, the active vertices and the active seeds): PCG converged (0 skips), mean \|F - F*\| <= 2000 delta, no class flip where \|F* - 0.5\| > 3m (m = 2.9217e-3, `parity/results/M10a-class-margin.json`), labels in [0, 1], seeds exact; for the update chain and for compute() (IrLP) |
| compare() (C1, C6, C10, C11) | damped: bitwise (the label bits and the seeds); in_place: the seeds exactly and mean \|dF\| <= 2 x 2000 delta over the unlabelled vertices either side moved from 0.5 |
| compare_across_backends() (C3, C12) | the in_place rule for both schedules (damped is bitwise per backend: the host and the device sum in other orders) |
| deterministic() | components, sparsified_edges, initial_frontier; damped also converged and fallback_used (and affected, `affected_deterministic`); never iterations or frontier_visits (AD20) |
| deterministic_across_backends() | components, sparsified_edges, initial_frontier and the batch summary (converged, fallback_used and affected are deterministic per backend only) |
| history_independent | false: C5 is skipped (ids are single-use, a departure cannot be undone, DynLP stops at a delta-approximate point that depends on the path) |
| host_sync_budget | 0 on sequential; `run_dependent_budget` on cuda (the fused engine's own count) |

### 4. A defect the kit found, fixed

**C10 on CUDA.** In one `dyng::update()` with two label_propagation results, the second result's
`stats::batch` reported 0 appended and 0 dropped entries: the device apply's report is settled
once, by the first participant whose engine synchronizes (ADR 0074), and the second read an
already-settled report as empty. The settled report is now kept with the device rows
(`device_slotted::last_settled`, reset by every device apply), so every participant of the update
reads the same counts. `cpp/src/graph/{device_slotted.hpp, apply_slotted_device.cu, graph.cpp}`;
no other path reads it.

### 5. What C2 needed, and the departures

On the first draft of the axis the damped suite's C2 failed one chain: a departure batch (deleting
any active vertex, seeds included) left 4 class flips with \|F* - 0.5\| up to 0.019 > 3m, while
mean \|F - F*\| stayed within 2000 delta. DynLP propagates from V_aff and stops where changes fall
below delta; after a seed's departure the harmonic solution of its component moves far from the
deleted vertex, and the damped schedule's wave (each update halves the change) dies before it
gets there. The original's generator never deletes a seed (its phase C deletes active unlabelled
vertices only), and AD7's bound and m were calibrated on that flow. The axis's departures follow
the original's rule; no bound, margin or tolerance was changed. A seed's departure stays covered
by the hand case `LabelPropagation.ASeedFreePairConverges` (the seed leaves, its neighbour and a
new vertex converge), not by the kit's oracle. **Decided by the author on 2026-10-30** (GOVERNANCE.md, approvals log, the row "Seed departures
and AD7's class rule"; first flagged as a new question in an author-reserved area, the parity
rules, taken on its most conservative option): AD7's class rule need not hold after a seed's
departure, which the original's flow never makes; the kit keeps the original's rule (departures
of active unlabelled vertices only), the hand case covers a seed's departure, and the algorithm
page (Limitations) and the update's doc comment in `<dyng/label_propagation.hpp>` document the
property: after a seed is deleted, labels far from it can stay stale until later batches reach
them.

The models start with 70 % of the vertices present: IrLP's Jacobi converges slowly on the long
chains of a sparse present subgraph (13 s for one 12,000-vertex model at 30 % present in Release,
0.03-0.17 s at 70 %), and the kit runs compute() after every batch of C2.

### 6. Times and labels

On the development machine (Xeon Gold 6258R): the host executable (sequential, both suites, both
offset types) 3 min in Release and 18 min in Debug; the CUDA executable 4.5 min (Debug). The host
executable carries the label `slow` (as triad_count's kit, M9a finding 20): the hosted sanitizer
jobs (`ctest -L cpu -LE slow`) leave it out, `cpu.yml` and `ci/check.sh` run it; the hosted times
are not measured in M10a (no workflow is dispatched from the branch).

## Consequences

- label_propagation passes C0-C3, C6-C12 on sequential and cuda (C4 skipped: one engine per backend
  until M10b's operators engine; C5 skipped as stated). C3 compares sequential and cuda in the CUDA
  executable.
- An algorithm on another container shape plugs into the kit through `container_axis` without
  touching the graph or hypergraph axes.
- The near_reference documentation in `test_traits.hpp` now gives its signature with the resources
  (it read `near_reference(const graph_t&, const snapshot&)`).
