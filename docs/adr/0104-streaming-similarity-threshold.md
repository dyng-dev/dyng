# ADR 0104: The streaming mode's similarity threshold: a maintained double sum (`similarity_threshold = std::nullopt`)

- **Status:** Accepted under delegation (2026-11-11), technical scope only. The mode is the
  author's: **AD16 (sheet E7), decided by the author on 2026-10-28** (GOVERNANCE.md's approvals
  log, the row "API semantics of 0.4 (sheet E2-E10)": the published setup is the documented
  default; streaming, with slack rows and tau from the current graph, an advanced mode with no
  gate, checked by the kit and the host model and reported dynG against dynG). PLAN 6.4.7 ("tau:
  `nullopt` = average edge weight of the current graph"), O9. M10b step 2 (the M10b acceptance
  criterion 4).
- **Date:** 2026-11-11
- **Deciders:** S M Shovan (AD16); the AI assistant, on the author's behalf (the rest)

## Context

M10a computed `similarity_threshold = std::nullopt` as the average weight of the live directed
entries of the graph after each batch, summed afresh in every update (an O(m) pass; on CUDA a
download of the rows per update). The step asks for the average to be maintained in the result as
a double sum and a count (insertions add; deletions subtract the live entries walked before the
apply), with a documented accumulation order.

## Decision

1. **State.** The result keeps `tau_sum` (double), `tau_count` and `tau_current`. Nothing of it
   runs while the options carry a threshold (the published setup is unchanged; `tau_current`
   stays false).
2. **The accumulation order** (documented on `options::similarity_threshold`):
   - set-up: the first update that needs the sum after compute() or set_options() sums the graph
     before its batch (G_t) over the live directed entries in vertex order and slot order (the
     sum M10a's every update made);
   - deletions, before the apply (where the update already walks the deleted vertices' rows for
     the affected set): for each vertex of Del ∩ V_t in the batch's order, a partial double sum
     over its row in slot order of every live entry's weight, and of its mirror's (the same
     weight, a second addition) when the neighbour is neither the vertex itself nor deleted in the
     same batch (that neighbour's own walk counts it then); the partial sum and its count are
     subtracted from the totals;
   - insertions, after the apply: the weight of every appended entry in batch order (an edge
     whose two endpoints are active after the batch adds its weight twice, one addition per
     entry);
   - tau = float(sum / count), 0.5 for a graph without entries.
   A compaction drops only entries of deleted vertices (never live ones), so it changes nothing of
   the sum.
3. **Backends.** The host computes it on both backends from the same data, so the value is
   bitwise the same on sequential and cuda: on cuda the deleted rows are walked on the host copy
   of G_t (downloaded for it; the streaming mode only), the appended entries from the batch and
   the host's states.
4. **Checked independently.** `LabelPropagationSlack.Streaming*` recompute the threshold by this
   order from lp_host_model's rows and compare the sparsified edges and the components of G'
   (which depend on tau) with the library's in every batch.

## Consequences

- The streaming mode's tau differs in the last bits from M10a's fresh sum (the order is the
  maintained one); the published setup's tau is the stream's, unchanged.
- A result can switch modes with set_options(): the sum is set up again from the graph.

## Alternatives considered

- **A device kernel for the deleted rows' partial sums.** Deferred: it would avoid the download
  per update on cuda; the streaming mode has no gate (AD16), and the host walk is exact and
  simple.
- **Recounting the whole graph per update** (M10a's way). Rejected by the step: O(m) per update.
