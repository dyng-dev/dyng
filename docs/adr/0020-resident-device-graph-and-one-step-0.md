# ADR 0020: The resident device graph under set semantics, and Step 0 once per update

- **Status:** Accepted (M2b, step cycle-cuda)
- **Date:** 2026-09-28
- **Deciders:** S M Shovan (lead maintainer)

## Context

PLAN 6.4.3 ports CycleEnumeration-GPU@0a976ad's CUDA update behind `cycle_count::update` and asks
that "the device graph becomes resident across batches (the original re-uploads G_t per call)",
with "the device sorted row merge from `build_next_rows_kernel`" (PLAN 6.4.1, M2 deliverables).
Until M2b a CUDA graph kept its authoritative CSR on the host: every batch was applied on the host
(`graph::apply`, a new CSR) and the device copy was dropped and uploaded again on first use, as
MOSP-CUDA does (M1b, ADR 0017).

The original's CUDA update (`update_static_histogram_cuda`) times, as `update_seconds`:
`prepare_batch` on the host (sort, deduplicate, binary searches in G_t), the upload of G_t, both
device phases, the device build of G_{t+1} (`build_next_rows_kernel`, then discarded) and the copy
of the histograms: 4.4 ms on DD for 25K + 25K (its RESULTS.md). It never produces the graph after
the batch on the host. A dynG update must apply the batch to the graph, and the host set apply of
M2a costs about 6 ms on DD (56 threads; the stage `graph.apply` of the M2b merge records), because
it normalizes the batch a second time and writes a new O(n + m) CSR. A host apply in the update
would put the DD gate (<= 1.10x of 4.4 ms) out of reach, and would defeat the point of a resident
graph.

M2a also normalized every set batch twice: cycle_count's Step 0 (`compute_structural_change`,
2.5 ms of the DD update) and again inside the commit (`apply_set_batch_host`), which the original
avoids on CUDA (its device merge reads the prepared batch).

## Decision

1. **Step 0 of `batch_semantics::as_sets` is computed once per update.** `normalize_set_batch()`
   (graph/normalized_batch.hpp; the first half of the M2a set apply, unchanged) produces the
   normalized batch and the counters of the batch. `run_update()` computes it before the
   participants' `before_apply` (profiler stage `<algo>.normalize`, e.g. `cycle_count.normalize`,
   `update.normalize` for `dyng::update`), hands it to every participant through the new
   `update_participant::use_normalized()` (a default no-op) and to the commit
   (`graph_access::apply(..., normalized)`). cycle_count takes its change lists from it instead
   of computing its own. Graphs without set semantics are unchanged (cycle_count still reduces
   their batches itself). The CPU backends profit as well: their commit no longer normalizes
   again.

2. **A resident CUDA graph under set semantics is updated on the device.** `graph_access::apply`
   merges the normalized batch into the sorted rows on the device (`apply_set_batch_device`,
   graph/apply_set_device.cu: the straight port of `mark_owners_kernel`, `change_rows_kernel`,
   `next_degree_kernel`, the scan and `build_next_rows_kernel`) when all of these hold: the graph
   was built with CUDA resources, its device copy is resident, the resources are CUDA resources of
   its device, the semantics are `as_sets`, the graph has no weight columns and the vertex ids are
   32-bit. The new device copy becomes the graph's state; its `insertion_ids` (the kernel's
   `next_owner`) are what the cycle_count insert phase reads as its ownership array, as the
   original does. Otherwise the batch is applied on the host and the device copy is dropped, as
   before (M1b).

3. **The host CSR of such a graph is a lazily downloaded copy.** After a device apply the host
   CSR is stale; `graph_impl::host_edges()` downloads it (a synchronous copy on the graph's device)
   the first time anything reads it: `view()`, `to_csr()`, `check_integrity()`, `clone()`, a host
   apply, the host engines, Step 0 of the next batch. `num_vertices()` and `num_edges()` never
   download (they read the device copy's counts). The download is thread-safe like the lazy
   transposition; mutating calls happen only from the owner of the graph.

4. **The device in-edges are built on first use.** `device_edges(res, in_edges)` uploads the
   out-edges only; the in-edges (count, scan, fill on the device, as before) are built when an
   engine asks for them (`graph_access::device`: sssp, and the commit's `prepare` for participants
   that read the prepared graph). cycle_count reads `graph_access::device_out` and never pays the
   transposition.

5. **The device change lists are uploaded once per update**, from pinned staging
   (`upload_normalized_batch`), and shared by the device apply and the cycle_count phases.

## Consequences

- cycle_count's CUDA update in the original's scope (G_t uploaded inside the call) costs Step 0,
  the upload, the phases and the merge, as the original; in the resident scope the upload is gone.
  Both are measured (`parity/timed_regions/cycle_count.toml`, `[reference.cycle_enum_cuda]`).
- A chain of updates on the device downloads nothing as long as nobody reads the host CSR, except
  Step 0 of the next batch, which binary-searches the host rows of G_t: the second update of a
  chain downloads G_t once (about what the original's upload costs). A device Step 0 (sorting and
  classifying the batch on the device) would remove that download; it is not part of the port and
  is recorded as future work (docs/developer/retrospectives/M2b.md).
- Weighted graphs and graphs with other batch semantics keep the M1b behaviour (host apply,
  re-upload); the cycle_count update then builds its insert-phase owner array itself
  (`mark_owners_kernel` on G_{t+1}). A device merge of weight columns, and the append-order device
  apply of PLAN 6.4.1 (for `mosp_compatible()`), come later.
- The device apply synchronizes its stream at the end (one host synchronization in the commit), so
  the state is complete before the host copy can be downloaded on another stream.
- `update_participant` gains one virtual function with a default; the sssp participant ignores
  it. Code that reads `graph_impl::out` directly no longer compiles: the field is private and
  reached through `host_edges()` / `host_edges_for_write()`.
