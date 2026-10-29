# ADR 0071: The label_propagation API (provisional until M10b's freeze)

- **Status:** The technical parts accepted under delegation (2026-10-29; GOVERNANCE.md,
  "Delegation of technical ADRs"). The names and semantics are the author's: AD12 (the new public
  names of 0.4, with E1's changes; E2: `update_stats::affected` keeps its frozen meaning and
  label_propagation adds `stats::initial_frontier` = |V_aff|), AD20 (the per-field determinism
  sentence of `core/stats.hpp`), AD16 (the published setup is the documented default), AD19 (no
  OpenMP backend in 0.4) and the comparator rules of sheet C8, **decided by the author on
  2026-10-28** (GOVERNANCE.md, approvals log, the rows "The new public names of 0.4 (AD12, HD8,
  HD12; sheet E1)", "API semantics of 0.4 (sheet E2-E10)", "Licensing, data provenance, budgets
  and scope of 0.4 (sheet G1-G3, H1-H3)" and "The parity rules of 0.4 (sheet C1-C8)"). The names
  beyond AD12's list (section 5) are listed in GOVERNANCE.md's open-decision sub-block `M10 (AD)`,
  pending the author's acknowledgement. The header is `tracked`, not frozen: M10b freezes it.
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  host-container-sequential-driver)
- **Builds on:** ADR 0023 (the uniform contract), ADR 0052 (the API ADR precedent), ADR 0069 (the
  slotted rows), ADR 0070 (the backend sentence), the sketch `docs/design/sketches/
  label_propagation.md`.

## Context

The sketch fixed the shape and left three questions for M10: where `on_limit` lives, whether
`std::optional<float>` is acceptable in `options`, and whether the published setup or the
streaming mode is the documented default. PLAN 5.3's example applied the initial batch and then
called `compute()`; the original's run (runDynLP) starts from an empty device graph and treats the
first batch b0 (the first ground truth and its edges) as an ordinary batch.

## Decision

### 1. The header `cpp/include/dyng/label_propagation.hpp`

| Entity | Form |
|---|---|
| `enum class schedule : std::uint8_t` | `in_place` (the published rule), `damped` (1/2-damped Jacobi) |
| `enum class on_limit : std::uint8_t` | `error`, `report`, `fallback_recompute`, nested in `label_propagation` (sketch question 1: until a second algorithm needs the same enumeration; hyper_sssp's `budget_action` stays separate, E1) |
| `struct options` | `std::optional<float> similarity_threshold` (a value: the published setup passes `stream.similarity_threshold`; `std::nullopt`: the average weight of the current graph's live directed entries, a double sum in vertex and slot order rounded to float, 0.5 without edges; never NaN), `float tolerance = 1e-4f` (> 0), `std::int64_t max_iterations = 100000` (> 0), `on_limit on_iteration_limit = on_limit::report`, `schedule update_schedule = schedule::in_place`, `engine cuda_engine = engine::automatic`; every field a tunable (`set_options`) |
| `struct stats : update_stats` | `apply_summary batch`; `std::int64_t components` (of G' over Ins \ Del), `sparsified_edges` (of G'), `initial_frontier` (= \|V_aff\|); all deterministic |
| `template <vertex_t> class result` | `labels()` (floats in [0, 1]), `seed_labels()` (int8: -1, 0, 1), `get_options()`, `set_options()`, `graph_version()`, `space()`, `clone(res)`, `reserve(res, max_batch_vertices, max_batch_edges)`; move-only, pimpl |
| `compute(res, graph<V, E, float>, array_view<const V> seed_vertices, array_view<const int8_t> seed_labels, options = {})` | IrLP |
| `update(res, graph<V, E, float>&, edge_batch_view<V, float>, result<V>&)` | DynLP, one batch |
| `detail::label_propagation_supported_v`, `detail::make_label_propagation_participant`, `detail::update_traits<label_propagation::result<V>>` | `dyng::update()` with label_propagation results |

Instantiated for `vertex_t = int32` and `edge_t` int32 or int64, float weights; other types stop
at a `static_assert` (`DYNG_LABEL_PROPAGATION_TYPES_MESSAGE`). `std::optional<float>` is accepted
in `options` (sketch question 2: the C++ API is the contract; nanobind binds it).

### 2. The published setup as data, not a mode

The published setup (sketch question 3; AD16: the documented default) is data the caller passes:
the graph built by `graph::with_capacity(stream.row_capacity)` (the full graph's degrees) and
`options::similarity_threshold = stream.similarity_threshold` (the original's
`computeAvgEdgeWeight` of the full graph). There is no `published` mode: the streaming mode of
M10b (`row_layout::slack`, `std::nullopt` tau from the graph seen so far) uses the same functions
with other data.

### 3. The parity flow against PLAN 5.3

The original's run is reproduced as `graph::with_capacity(stream.row_capacity)`, then
`compute()` on the empty capacity graph (untimed setup: every label 0.5, no tag consumed), then
`update(b0)` ... `update(b(B-1))`. PLAN 5.3's `g.apply(stream.initial)` + `compute()` would make b0
part of a static solve, which the original never runs: its b0 is counted, timed and consumes a
propagation tag. Both forms are valid uses of the API; the parity runs, the gates and the docs use
the first.

### 4. Semantics and requirements

- **I4 (no oscillating schedules)** applies to `update()`'s schedules: there is no plain Jacobi
  (the `damped` rule removes the period-2 oscillation of the original's defect D-b). IrLP, the
  published baseline of `compute()`, is exempt (ADR 0072): it is the paper's comparison, kept
  unchanged.
- **Requirements:** `directed = false` and float weights for both; `update()` needs
  `row_layout::slotted` with `batch_semantics::net_effect()` (`not_supported_error` otherwise,
  naming `graph::with_capacity()`); `compute()` also reads compact float graphs (the same labels
  for the same edges: it reads the column-sorted snapshot of the live edges).
- **Labels:** a seed's label is exactly its class (after every update, deleted seeds included);
  a deleted vertex keeps its last label; a vertex that never entered the graph, or arrived and
  left in one batch, stays at 0.5 under `update()`; under `compute()` every vertex that is not
  active is 0.5.
- **Determinism:** `damped` is bitwise per backend and build; `in_place` is a tolerance (the CUDA
  backend's published rule is not reproducible; the sequential backend is, ADR 0072). The
  inherited `affected` (the vertices whose label bits changed; E2), `converged` and
  `fallback_used` are deterministic per backend under `damped` and schedule-dependent under
  `in_place`: AD20's sentence in `core/stats.hpp` ("An algorithm's stats documentation may state a
  weaker level ... for an inherited field; that statement governs for that algorithm"), written
  here as the last paragraph of the `update_stats` class comment, the placement both tracks use.
- **on_limit:** `report` keeps the labels and reports `converged = false`; `error` throws
  `convergence_error` after the commit (the result is poisoned); `fallback_recompute` recomputes
  every label with IrLP (`fallback_used = true`). `compute()` reports a cap under `report` and
  throws otherwise (it is the recomputation).
- **Backends:** AD19 and ADR 0070's sentence: the manifest lists sequential (and cuda from M10a
  step 4); OpenMP resources raise `not_supported_error` naming `sequential` and `cuda` and the fix
  `dyng.Resources("sequential")`.
- **Seeds of compute():** active vertices of `g`, each once, class 0 or 1 (`invalid_argument_error`
  otherwise). The batches' labels join the result's seed labels in `update()`.
- **Diagnostics:** the profiler counter `label_propagation.irlp_iterations` (IrLP's iterations of a
  `compute()`), and the stage trace of ADR 0073 (`DYNG_DYNLP_TRACE_DIR`).

### 5. The new public names

AD12's list (decided by the author on 2026-10-28): `label_propagation::schedule`, `on_limit`,
`options`, `stats` (with `initial_frontier`), `result`, `compute`, `update`; the graph's
`with_capacity`, `num_active_vertices`, `vertex_status`, `graph_view::status` and `row_end`,
`batch_semantics::net_effect()`, `edge_batch::insert_vertex` and `delete_vertex` (ADR 0069); the
vertex-arrival generators (`generators::legacy::dynlp_vertex_arrival_stream`, ADR 0067);
`testing::harmonic_solution`, `union_find`, `lp_host_model` (ADR 0068); the compat tool
`dyng-compat-dynlp` (ADR 0073).

**Beyond AD12's list, pending the author's acknowledgement** (GOVERNANCE.md, `M10 (AD)`):

| Name | Where | Why |
|---|---|---|
| `schedule::in_place`, `schedule::damped`; `on_limit::error`, `report`, `fallback_recompute` | the enumerators | the sketch's |
| `options::similarity_threshold`, `tolerance`, `max_iterations`, `on_iteration_limit`, `update_schedule`, `cuda_engine` | the fields | the sketch's |
| `stats::components`, `sparsified_edges` | the fields | the sketch's |
| `result::labels`, `seed_labels`, `reserve` | the members | the sketch's, and `reserve` (the original's setup) |
| `detail::label_propagation_supported_v`, `DYNG_LABEL_PROPAGATION_TYPES_MESSAGE`, `detail::make_label_propagation_participant` | the dispatch | the pattern of every algorithm |
| the profiler counter `label_propagation.irlp_iterations` | profiler | results.csv's `irlp_iterations` |
| `batch_semantics::as_net_effect`, `apply_summary::dropped_insertions`, `edge_batch::insert_vertices()`, `insert_vertex_labels()`, `delete_vertices()`, `vertex_arrival_batch::view()`, `vertex_status::{active, deleted, inactive}` | ADR 0069 | the container |
| `generators::vertex_arrival_options`, `vertex_arrival_batch`, `vertex_arrival_stream` | ADR 0067 | the generator's types |
| `testing::harmonic_options`, `harmonic_result`, `label_error`, `label_error_against`, `class_agreement`, `class_agreement_with`, `lp_model_batch`, `lp_model_report` | ADR 0068 | the oracles' types |

## Consequences

- M10b reviews and freezes the header (with Python and the command line); the names above are
  then on the stable list for the author's confirmation at 0.4.0 (AD15).
- `examples/` and the tutorials pass explicit resources (AD19).

## Alternatives considered

- **`on_limit` in `core/types.hpp`:** rejected until a second user (rule of two).
- **A `published` option or mode:** rejected; the setup is data (section 2).
- **`affected` = |V_aff| for label_propagation:** rejected by the author (E2); `initial_frontier`
  carries it.
