# ADR 0103: `row_layout::slack`: rows that move to the tail, compactions and vertex growth

- **Status:** Accepted under delegation (2026-11-11; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the streaming mode the author decided as AD16
  (sheet E7, **decided by the author on 2026-10-28**: the published setup is the documented
  default, streaming on slack rows an advanced mode with no gate; GOVERNANCE.md's approvals log,
  the row "API semantics of 0.4 (sheet E2-E10)") and PLAN 12.1's R13 and O9. M10b step 2 (the
  M10b acceptance criterion 4; PLAN 5.2). It implements the representation ADR 0069 section 8
  designed, with the changes recorded below.
- **Date:** 2026-11-11
- **Deciders:** the AI assistant, on the author's behalf

## Context

LabelPropagation-CUDA@a276a3a (O) needs the capacity of every row up front (its fixed slot
layout; `row_layout::slotted`, ADR 0069). A real stream does not know the degrees of the vertices
that will arrive (PLAN 12.1, R13). PLAN 5.2 reserves `row_layout::slack` ("per-row headroom;
overflowing rows move to a tail region; compaction when needed") and `graph_properties::headroom`
(0.125); ADR 0069 section 8 designed it as slotted rows whose slots move, and planned a
`graph_view::row_start` field.

## Decision

1. **Representation.** A slack graph is a slotted graph (the same `slotted_rows` state, the same
   vertex batches under `batch_semantics::net_effect()`, Step 0's checks, the queries) whose slot
   ranges move: row v owns `[row_ptr[v], row_limit[v])` of one slot arena (the graph's host CSR;
   `row_ptr[n]` is the arena's size), its entries are `[row_ptr[v], row_end[v])`. **No
   `graph_view::row_start` field** (a change from ADR 0069 section 8): `view().out.row_ptr[v]` is
   row v's start for both layouts, so every reader of slotted rows (the LP engines read rows as
   `(row_ptr[v], row_end[v])`) reads slack rows unchanged; for slack rows `out.row_ptr` is not
   monotone and `row_ptr[v + 1]` is not the end of row v's slots (documented in `graph_view.hpp`).
   This keeps a public name out of the API; the row limits stay internal.
2. **Capacities.** `graph::with_capacity(res, n, hints, props)` with `layout = row_layout::slack`
   takes empty, partial (fewer than n) or full hints; row v gets `slack_capacity(hint) =
   ceil(hint * (1 + headroom))` slots (0 without a hint), in vertex order. The headroom is
   `graph_properties::headroom` (default 0.125; refused outside [0, 16]).
3. **A row that overflows moves.** Step 0 (`normalize_vertex_batch_slack`) counts the entries
   each row receives (the edges whose endpoints are both active after the batch) and plans the
   batch's layout step: a row whose entries plus appends exceed its slots moves to the end of the
   arena with `slack_capacity(entries + appends)` slots, in increasing vertex order (a deterministic
   tail). **All** of its entries move, in order, also those to departed vertices (which every
   reader skips, as in slotted rows): a move never changes the order in which a reader meets a
   row's entries, so the engines sum a moved row exactly as before. The old slots become dead
   (-1 / 0), counted in `dead_slots`.
4. **Compaction.** When the dead slots after the planned moves would outnumber the owned slots
   (2 x dead > arena), the batch compacts instead: every row is rewritten in vertex order with
   the entries whose neighbour is not deleted (a deleted vertex's row is emptied), each row with
   `slack_capacity(kept + appends)` slots (a row never written keeps its hint; *amended after
   M10b's review*: every row without entries that is not a deleted vertex's keeps its capacity, an
   active vertex whose edges have not arrived included: before, only an inactive row kept it, so a
   compaction dropped the hint of an arrived vertex and its first edge moved the row;
   `Slack.ACompactionKeepsTheHintOfAnActiveEmptyRow`). It runs inside
   `apply()`, after the batch's vertex states and before its appends, outside any kernel, at
   most once per batch; it is amortized (a compaction follows at least as many dead slots as it
   rewrites). **It changes the summation order of a row on the device**: the kept entries sit at
   other slot positions than before, and the engines sum a row in lanes by slot position, so
   labels after a compaction may differ in the last bits from slotted rows' (the oracles hold:
   `LabelPropagationSlack.ForcedCompactionsAndTheOracles5k5`).
5. **Vertex growth.** With `batch_semantics::allow_vertex_growth` (slack rows only; slotted rows
   keep `net_effect()` exactly), an id at or beyond `num_vertices()` (up to the vertex type's
   limit) grows the graph to the largest id + 1, the new vertices inactive with rows of no slots
   at the arena's end. Step 0 grows only its scratch, so a refused batch leaves the graph as it
   was.
6. **capacity_error only at the edge_t limit.** Step 0 refuses a batch whose moves would pass
   `edge_t`'s maximum and whose compaction's bound (the owned slots plus every touched row's
   growth) would too; `with_capacity()` refuses hints whose capacities do not fit. Slotted rows
   keep their `capacity_error` on overflow.
7. **The strong guarantee on the host.** The record, the arena's growth (reserved) and a
   compaction's new arrays (laid out from a copy of the states after the batch) are allocated
   before anything changes.
8. **CUDA.** The device copy keeps the row limits next to the row starts and ends, and the
   device apply keeps the host mirror of the layout current (the states, starts, limits and ends:
   Step 0 reads them without a download): the growth on both sides, the states (host, then the
   device kernels of the slotted apply), then the moves (the rows copied to the arena's tail by a
   kernel per move, their old slots cleared by a second one; the arena grows by x 1.5 outside the
   kernels) or a compaction (the kept entries counted on the device and read back once, the
   layout computed on the host by the same function as the host apply's, the rows rewritten into
   a new arena by one thread per row), the host mirror's row ends advanced by the appends, then
   the slotted apply's append kernels against the row limits. The slotted apply is unchanged: the
   append kernels read the row limit as `row_ptr + 1` for slotted rows (the same address as
   `row_ptr[v + 1]`), and the apply was split into its two halves without reordering anything.
9. **Stats.** `apply_summary` gains `relocated_rows` and `compactions` (appended; every
   algorithm's `stats::batch` carries them).
10. **The guard.** Every compact-only algorithm refuses slack rows through ADR 0070's central
    check (`expect_compact_layout` refuses every layout but compact; `CompactOnlyAlgorithmsRefuse`
    tests sssp); `from_edges()` and `from_csr()` refuse them, naming `with_capacity()`.
11. **label_propagation on slack rows** reads them through `(row_ptr[v], row_end[v])` on both
    backends and both engines; its CUDA queues are sized for the rows' lengths after each batch
    (a bin is a row's length when the vertex is queued, and a compaction only shortens rows),
    outside the kernels, in before_apply: a capacity that grows takes headroom (x 1.5, at most the
    vertex count; the scratch buffers' `reserve_with_headroom` rule), and the per-vertex host
    scratch of that count lives in the leased CUDA workspace, so the steady state allocates
    nothing (the kit's C8 on the slack preset; invariant I9);
    `update()` refuses a batch that would grow the vertex set (a result keeps one label per vertex
    of its graph) before anything changes. compute() bins a slack row by its slots.
12. **Tests.** `graph/slack_test.cpp` (the headroom, moves with their order, hubs and repeated
    overflows, every row overflowing in one batch, deletions of moved rows, natural and forced
    compactions, growth, the strong guarantee, the guard, vertex-arrival streams with 0, 25, 50
    and 100 % of the capacities against the kit's model `support/slack_model.hpp`, which keeps
    each row as the sequence of its entries); `graph/slack_cuda_test.cpp` (the device apply's
    layout equals the host apply's after every batch, growth, clones);
    `label_propagation_slack_test.cpp` (bitwise slotted while no compaction ran, the streaming mode
    against the oracles on corpus C, forced compactions; after M10b's review also compactions
    inside `update()`, `NaturalCompactionsInsideUpdate5k24`: no hints, a headroom of 0, 24 batches,
    the states, the rows, the seeds and the label range every batch, and damped's labels against
    slotted rows, bitwise until the compaction and within 100 delta on average after it). The
    harmonic oracle's bound (AD7, ADR 0068) is calibrated on corpus C's short streams (5 and 10
    batches): on a 40-batch 5k stream O's incremental rule drifts past it from batch 9 on slotted
    rows as much as on slack rows (mean |F - F*| 5.2e-3 against 2000 delta = 2e-3; M10b's review),
    so the long-stream test compares with slotted rows, not with that bound; the kit's float axis gains the preset
    "slack, net_effect(), half capacities" on both backends (C7's row overflow becomes a move).

13. **M11a's slack-row accessor is not reused** (the step's instruction: reuse it if it is on
    `origin/main` and suits graph rows, else record why). `detail::hyper::slack_row_view`
    (`cpp/src/hypergraph/slack_csr/slack_rows.hpp`, ADR 0086; on `origin/main` since M11a's
    merge `64a5123`) describes the node rows of a hypergraph's `slack_csr` line graph: 64-bit row
    starts with 32-bit degrees and slot capacities, the neighbours of a row as a set in no
    particular order, and one weight per node (`node_weight`). label_propagation's rows are graph
    rows with a float weight per entry, a state per vertex and an entry order that a move must
    keep (the bitwise identity with slotted rows, point 3), and its engines read them through
    `graph_view`'s `(out.row_ptr[v], row_end[v])` exactly as they read slotted rows. Wrapping graph
    rows in `slack_row_view` would need per-entry weights, vertex states and an order guarantee it
    does not have, and a second read path in both engines; so slack graph rows keep the slotted
    rows' state (`slotted_rows`, with `row_limit` added) and the accessor stays the hypergraph
    store's. Its file comment's "label_propagation reuses it in M10b" is M11's text and is left to
    M11 (M10 does not edit M11's files).

## Consequences

- Damped on slack rows with the published threshold is bitwise slotted's on the same stream
  while no compaction ran (with the published capacities nothing moves; with half of them rows
  move and the labels stay bitwise).
- The published setup and its gates are untouched: slotted rows take the same code as before
  (the reading of ADR 0024 on the LP slotted path, parity/results/M10b.md).
- A slack graph's memory is its arena (live and dead slots, at most twice the owned ones) plus
  three arrays per vertex.

## Alternatives considered

- **Moving only a row's live entries.** Rejected: it would change the lane order of a moved row
  and break the bitwise identity with slotted rows before any compaction.
- **A host apply with a device mirror uploaded per batch.** Rejected: the CUDA engines read the
  batch the device apply uploads, and a whole upload per batch costs O(m) transfers.
- **M11a's `slack_row_view` for graph rows.** Rejected (point 13): a different row shape (node
  weights, unordered sets) and a second read path in the engines.
- **A public compaction call or a tunable threshold.** Not added (no new public names): the
  tests force a compaction through `detail::graph_access::compact_slack`.
