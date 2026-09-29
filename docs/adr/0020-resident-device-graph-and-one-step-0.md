# ADR 0020: The resident device graph under set semantics, and Step 0 once per update

- **Status:** Proposed (M2b, step cycle-cuda; amended by the M2b review: points 7-10). Awaiting
  the author's acceptance, which GOVERNANCE.md's approvals log will record (the first version said
  "Accepted" although no acceptance had been given).
- **Date:** 2026-09-28 (amended 2026-09-29)
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
   CSR is stale; `graph_impl::host_edges()` downloads it (point 8) the first time anything reads
   it: `view()`, `to_csr()`, `check_integrity()`, `clone()`, a host apply, the host engines. Step 0
   of the next batch no longer reads it (point 7). `num_vertices()` and `num_edges()` never
   download (they read the device copy's counts). The download is thread-safe like the lazy
   transposition; mutating calls happen only from the owner of the graph.

4. **The device in-edges are built on first use.** `device_edges(res, in_edges)` uploads the
   out-edges only; the in-edges (count, scan, fill on the device, as before) are built when an
   engine asks for them (`graph_access::device`: sssp, and the commit's `prepare` for participants
   that read the prepared graph). cycle_count reads `graph_access::device_out` and never pays the
   transposition.

5. **The device change lists are uploaded once per update**, from pinned staging
   (`upload_normalized_batch`), and shared by the device apply and the cycle_count phases.

6. **The deletion marks of G_t are computed once per update** (amended in M2b, step
   cuda-parity-perf). `mark_normalized_deletions()` (graph/apply_set_device.cu: `mark_owners_kernel`
   on a `0x7f` array of m ints) writes them into the normalized batch, next to its device lists;
   the cycle_count delete phase reads them as its ownership array and the device apply as its
   deleted positions, as the original's single `owner` array serves both. The first version of
   this ADR computed them twice (cycle_count's workspace and a buffer of the apply), which kept a
   second m-int array alive during the merge. The graph module still owns the function; the
   algorithm only asks for it earlier.

7. **Step 0 of a stale-host graph runs against the device copy** (M2b review).
   `graph_access::normalize()` normalizes against the host CSR while it is current, and otherwise
   (a device apply produced the state) with `normalize_set_batch_device()`: the requested lists are
   built, sorted and deduplicated on the host exactly as before, and the only thing Step 0 reads of
   G_t, the membership of each requested change (`has_edge`), is answered on the device by
   `edge_membership_kernel` (one binary search per change in its sorted row), through the pinned
   staging and device buffers of the pooled normalized batch. Both paths share one core with a
   membership callback, so their lists and counters are identical
   (`CycleCountCuda.DeviceStepZeroEqualsTheHostStepZero`). A chain of updates therefore never
   downloads the graph: the first version downloaded all of G_t in every update after the first
   (15 + 88 MB per update on Twitch), and a growing graph reallocated the host arrays (a 26 ms
   spike), which the resident scope of the gate, measured on the first update only, did not see.
   The chain is now measured (`--chain`, regions `update_chain_steady` / `update_chain_worst`).

8. **The download is ordered on the state's own stream** (M2b review). `download_device_graph()`
   used a synchronous `cudaMemcpy` on the legacy default stream, which PLAN 4.7.1 rules out (it
   waits for every blocking stream of the device). It now enqueues `cudaMemcpyAsync` on the stream
   the state's buffers are ordered on (the stream of the resources that built it, which the
   resources contract keeps alive as long as the buffers) and synchronizes that stream. The copies
   go straight into the CSR's pageable vectors, not through pinned staging: a staged copy would add
   a host copy of the whole graph, and after point 7 a download happens only on an explicit host
   read. A vector that must grow is released first and reserved with room for growth (no copy of
   stale content).

9. **Self-loop change edges are skipped by the phases** (M2b review). Under `as_sets` with
   `self_loop::keep` the normalized batch keeps a self-loop as a change edge (the graph stores the
   loop), while `compute_structural_change()` and the original's `prepare_batch()` drop it. The
   phases counted a spurious 2-cycle through it on every backend. A self-loop lies on no simple
   cycle of length >= 2, so the phases skip it without renumbering the ids
   (`count_cycles_through_edge()` returns 0, `item_counts_kernel` gives it no work items); its id
   never decides an ownership, so the device insertion ids and deletion marks, indexed by the
   normalized positions, stay valid.

10. **The device apply's scratch is pooled** (M2b review). The change rows of both lists, the
    degrees and the scan's CUB scratch live in the pooled normalized batch and are reallocated only
    to grow, so a steady workload allocates only the three arrays of G_{t+1} (invariant I9,
    `CycleCountCuda.SteadyStateUpdatesAllocateOnlyTheNextGraph`). The peak device memory is
    unchanged: the scratch was live at the apply's peak before as well.

## Consequences

- cycle_count's CUDA update in the original's scope (G_t uploaded inside the call) costs Step 0,
  the upload, the phases and the merge, as the original; in the resident scope the upload is gone.
  Both are measured (`parity/timed_regions/cycle_count.toml`, `[reference.cycle_enum_cuda]`).
- A chain of updates on the device downloads nothing as long as nobody reads the host CSR (point
  7). The sort and deduplication of Step 0 stay on the host, as the original's `prepare_batch()`
  (a bucket sort that costs less than the original's `std::sort` on sorted and on shuffled input,
  with no shortcut for sorted lists; `sort_and_dedup`); a fully
  device Step 0 remains future work.
- Weighted graphs and graphs with other batch semantics keep the M1b behaviour (host apply,
  re-upload); the cycle_count update then builds its insert-phase owner array itself
  (`mark_owners_kernel` on G_{t+1}). A device merge of weight columns, and the append-order device
  apply of PLAN 6.4.1 (for `mosp_compatible()`), come later.
- Device memory (PLAN 8.6): with the marks shared and the static-count work items returned when
  an update begins (`cycle_count_cuda_begin_update`), the peak of live device allocations of the
  CUDA update equals the original's on every gate case (`parity/results/M2b.md` section 4.8). The
  stream-ordered pool keeps up to one 32 MB granule more reserved.
- The device apply synchronizes its stream at the end (one host synchronization in the commit), so
  the state is complete before the host copy can be downloaded on another stream.
- `update_participant` gains one virtual function with a default; the sssp participant ignores
  it. Code that reads `graph_impl::out` directly no longer compiles: the field is private and
  reached through `host_edges()` / `host_edges_for_write()`.
