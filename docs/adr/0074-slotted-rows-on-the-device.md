# ADR 0074: Slotted rows resident on the device, and the device apply of vertex batches

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 4 (cuda-device-apply-fused-engine-parity), its first
  commit and go/no-go (PLAN 5.2, 5.9, 4.7.3, 6.4.7; the M10a acceptance criterion 9, device path).
  It builds on ADR 0069 (row_layout::slotted on the host) and ADR 0020 (the resident device graph,
  the lazy host download). The rules it carries out are the author's: AD11 (vertex batches and
  `net_effect` on slotted rows only) and C8's status mapping, **decided by the author on
  2026-10-28** (GOVERNANCE.md, approvals log, the rows "API semantics of 0.4 (sheet E2-E10)" and
  "The parity rules of 0.4 (sheet C1-C8)").
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  cuda-device-apply-fused-engine-parity)

## Context

LabelPropagation-CUDA@a276a3a (O) keeps its dynamic graph on the device for the whole run
(`dynamicGraph.h`'s `DeviceGraph`): fixed slot ranges sized from the full graph, a row end per
vertex, the slot columns and weights, and a status per vertex; per batch only the ids and the new
edges cross PCIe, the vertex changes are applied by three kernels, and the edges are appended
through half-edge keys, a stable radix sort (CUB), an exclusive sum by key and an append that drops
and flags what does not fit; a sticky error word is read after the batch. ADR 0069 gave dynG the
host form of these rows; label_propagation's fused CUDA engine (ADR 0075) needs the device form,
with the rows never re-uploaded.

## Decision

### 1. The device copy (`cpp/src/graph/device_slotted.hpp`, `apply_slotted_device.cu`)

A slotted graph built with CUDA resources and 32-bit vertex ids (`device_slotted_supported_v`)
keeps `device_slotted<V, E, W>` in its `graph_impl`: `row_ptr` (n + 1, fixed), `row_end`, the slot
`col_ind` (-1 in unused slots) and `weights` (0), and `status` as int32 with `vertex_status`'s
values (active 0, deleted 1, inactive 2: O's active and deleted codes; O's ground truth is a seed
label of the algorithm's result, ADR 0069), plus the current batch's device arrays, the append
scratch (two key and two item buffers, 16 bytes per half-edge, and CUB's temporary storage), a
pinned staging buffer and a report block with its pinned mirror.

- `graph::with_capacity()` on CUDA builds the device copy directly (O's `createDeviceGraph`: the
  slots cleared, every row empty, every vertex inactive); the host keeps the slot ranges, the states
  and the counts, and its slot columns and weights stay unallocated and stale.
- The host copy of the rows is **stale after a device apply** and downloaded once by the first host
  read (`graph_impl::host_edges()`: view(), to_csr(), check_integrity(), clone()), ADR 0020's rule.
  `num_vertices()`, `num_active_vertices()` and `num_edges()` never download: the states and the
  counts stay current on the host.
- A host graph cloned to CUDA uploads its rows on first device use; a CUDA slotted graph applied
  with host resources is applied on the (downloaded) host copy and its device copy dropped.
- `graph::reserve(res, edge_capacity)` on a slotted CUDA graph sizes the append scratch and the edge
  arrays for batches of up to `edge_capacity` edges (O sizes them from the whole run's batches); the
  vertex arrays grow to the exact size of the largest batch seen. Both grow outside the kernels.

### 2. One batch, in O's order

1. **Step 0 stays on the host** (ADR 0069's `normalize_vertex_batch`, every check of the input
   contract with the strong guarantee) **except the row capacity**, which needs the rows: it is
   checked on the device. Then `upload_vertex_batch()` copies the deleted ids, Ins \ Del with the
   seed labels, then the edges (O's `idsReady` then `edgesReady` order): arrays the caller declares
   `memory_space::pinned_host` directly, pageable ones through the pinned staging (whose last copy
   is fenced before the next host write); Ins \ Del and Del_eff are the batch's own arrays when Step
   0 removed nothing from them. This happens in `graph_access::normalize`, before any participant
   reads G_t (label_propagation's `before_apply` marks the deletion neighbours on the device rows).
2. **The commit** (`graph_access::apply` -> `apply_vertex_batch_device`): the live entries Del_eff
   removes are counted first (Del_eff marked `deleting`, then each row summed as the host apply
   counts: 2 per entry to an active vertex, 1 per entry to another vertex being deleted or to
   itself; an addition: O keeps no count, dynG's `num_edges()` needs it), then O's kernels:
   `setStatusKernel` (every id of Del), `insertKernel` (Ins \ Del; a vertex that was not inactive is
   recorded), `halfEdgesKernel` (key = n when an endpoint is not active, **and when an endpoint is
   out of range: recorded instead of read out of bounds, a documented fix of O**, defence in depth
   after Step 0), CUB `DeviceRadixSort::SortPairs` (stable, `keyBits`) and
   `DeviceScan::ExclusiveSumByKey` over a constant 1, `appendKernel` (a half-edge past its row's end
   is dropped and recorded), `advanceRowEndKernel`, and a one-thread count of the dropped half-edges
   from the sorted keys; then the report is copied to its pinned mirror. The host states change as
   the host apply changes them; the applied record lists the vertex changes and marks
   `edges_on_device` (the appended edges stay on the device).
3. **The report is read once**, after the next synchronization the caller makes anyway:
   label_propagation's fused engine settles it after the propagation's first host synchronization
   (`graph_access::settle_device_apply`, no extra synchronization: O reads its flags once per batch
   too); `graph::apply()` and `run_update()` synchronize once and settle it when nobody did
   (`graph_access::finish_device_apply`). Settling updates the live entries and the summary's
   appended and dropped counts and maps the error word: an overflow to **capacity_error**, a
   re-insertion or an out-of-range endpoint to **invalid_argument_error** (unreachable after Step
   0). **A device overflow poisons the graph** (the basic guarantee, documented in `graph.hpp`): the
   batch is applied with the overflowing half-edges dropped, `check_integrity()` reports it and
   every later apply raises capacity_error.

The kernels move vertex ids and weight bits only: there is no floating-point instruction in the
object (`parity/label_propagation_sass.py port`: none in its 24 functions, CUB's included) and no
fast math (the graph module is not a FAST_MATH target).

## Consequences

- O's batches applied to a CUDA `with_capacity` graph reproduce the trace golden's row ends, slot
  bits and statuses (by name, C8) after every batch of P+ (`compare.py label_propagation container
  --backend cuda`; `parity/results/M10a.md` section 11.1); compute-sanitizer memcheck is clean on the
  5k5 and 50k replays and on the tests; the step-3 host tests have a device twin
  (`cpp/tests/graph/slotted_cuda_test.cpp`: hand batches, a DynLP stream with pinned and pageable
  arrays, 32- and 64-bit offsets, integer and unweighted graphs, Step 0's errors with the strong
  guarantee, the device overflow, O's nine input-contract cases, clones between the backends,
  reserve()).
- The capacity check moves from before the batch (host, strong) to after it (device, basic) on
  CUDA; the documentation of `graph::with_capacity()` and `graph::apply()` says so (a
  documentation-only api-change).
- `graph_impl.hpp` and `graph.cpp` changed (the device slotted members, the slotted branches of
  `num_vertices()`, `host_edges()`, `normalize()` and `apply()`): code compiled into the stable
  algorithms' paths. Step 5 re-reads every ADR 0024 region on the final library.

## Alternatives considered

- **Step 0's capacity check on the host against host row ends kept current**: the row ends would
  need the host to count every half-edge per row (a random-access pass over the batch per update,
  O(edges) host work O does not do); rejected for the gate (AD2's update[sum] at 1.05x).
- **A graph-level synchronization after every device apply**: simpler, but one more host
  synchronization per batch than O, against the step's "merged into an existing sync".
