# ADR 0010: Batch and graph semantics

- **Status:** Proposed (draft, M1a); accepted with the 0.1 API freeze (M3)
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

The originals interpret a batch of edge changes in different ways:

- MOSP-OpenMP / MOSP-CUDA `applyChangeBatch()` (and the file-based `updateGraphCSR()`): rows are
  unsorted; deletions come first and each removes the FIRST remaining `(u,v)` of row `u`;
  insertions follow in file order, an insertion of an existing `(u,v)` overwrites the weights of
  the first remaining match (so the last insertion of a pair wins), otherwise the edge is
  appended to row `u`; parallel edges and self-loops in the input are kept; surviving edges keep
  their order. `applyChangeBatch()` also classifies every insertion per objective as a weight
  increase (the first `(u,v)` of row `u` after the batch against the first before it), which the
  SOSP update treats like a deletion. `updateGraphCSR()` writes sorted rows and collapses
  parallel edges; it agrees with `applyChangeBatch()` only as an edge set on simple graphs.
- CycleEnumeration-GPU `prepare_batch()` keeps sorted rows of a simple graph and normalizes a
  batch as a set (duplicates removed, no-ops dropped, insert/delete pairs cancelled).
- LabelPropagation-CUDA applies vertex batches (Ins \ Del, Del ∩ V_t) to slotted rows.

Byte parity with MOSP's CSR and with CycleEnum's CSR is only possible if one container can do
both row orders and both multigraph rules, and if every result on a graph agrees on how a batch
is read.

## Decision

1. **The semantics belong to the graph**, not to the batch or the algorithm:
   `graph_properties::semantics` (a `batch_semantics`) is fixed when the graph is created, so all
   results maintained on one graph see the same net change. `edge_batch` / `edge_batch_view`
   only carry the requested operations, in order.
2. `batch_semantics` has the switches of PLAN Section 5.2: `on_existing_insert` (upsert, error,
   ignore), `on_missing_delete` (ignore, error), `on_self_loop` (keep, drop, error),
   `deletions_first`, `allow_vertex_growth`. The default-constructed value **equals**
   `batch_semantics::upsert_last_wins()`, MOSP's `applyChangeBatch()`: upsert, ignore missing
   deletions, **keep self-loops**, deletions first, vertex growth.
3. `graph_properties` adds `row_order` (`sorted`, `append`) and `multi_edges` (`forbid`,
   `allow`). The general default is a sorted simple graph (the least surprising for new users).
   `graph_properties::mosp_compatible()` = append + allow + upsert_last_wins reproduces MOSP's
   CSR byte for byte, edge order and parallel edges included.
4. **Construction** (`from_edges`, `from_csr`) follows the same rules: rows are grouped by source
   in input order; `sorted` sorts rows stably; `forbid` merges duplicate `(u,v)` (first position,
   last weights); self-loops follow `on_self_loop`. An undirected graph stores both directions,
   and every batch edge of an undirected graph changes both.
5. **Application** on the compact layout is MOSP's `applyChangeBatch()` ported straight (a new
   CSR per batch), with the switches around the unchanged core. It returns an `apply_summary`
   (public) and, for the algorithms, a `detail::apply_delta` with the effective insertions and
   deletions and the per-(insertion, objective) weight-increase flags (one byte each, lifting
   MOSP's K <= 32 limit).
6. **Deferred:** the `set()` preset and `cycle_enum_compatible()` (M2, with `cycle_count`), and
   `net_effect()` with vertex batches (0.3, with `label_propagation`). Vertex operations in a
   batch throw `not_supported_error` until then. Every algorithm will accept every preset,
   because Step 0 reduces a batch to its net structural change.

## Consequences

- MOSP parity of the updated CSR is a byte comparison (the graph/io fixture tests compare the
  updated CSR, its transposition and the weight-increase flags with the pinned original on 27
  cases). CycleEnum parity will use `cycle_enum_compatible()`.
- Distances and parents of `sssp` are canonical, so they do not depend on the row order; only the
  CSR files do.
- The struct defaults in the PLAN sketch had `on_self_loop = drop`; that would have made
  `batch_semantics{}` differ from `upsert_last_wins()` and from MOSP. The default is `keep`
  (recorded in the M1a retrospective). Presets that need `drop` (for example `set()`) set it.
- `graph_properties::num_weights` is used only by constructors without input data
  (`with_capacity`, 0.3); `from_edges` / `from_csr` take the number of weight columns from their
  input and store it there.
