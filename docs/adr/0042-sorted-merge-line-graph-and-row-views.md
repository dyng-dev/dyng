# ADR 0042: The sorted_merge line graph, the row views and the line-graph oracle

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8b step 2 (PLAN 5.2, 6.1 C2 and D6, 6.4.5, 6.4.6, 4.5.1, 4.7.5; the
  M8b acceptance criteria 6, 7, 8 (the reading; the gate itself is the author's and is not
  changed), 9 (the reading) and 19). It changes no rule the author approved: the parity rules
  (PLAN 8.3), the gates and their protocol (PLAN 8.6; the memory gate is read as store accounting,
  the author's decision of 2026-10-07), and the names fixed by the plan
  (`line_graph_policy::sorted_merge`, `hypergraph_storage::escher`) stay as they are.
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step line-graph-views-memory)
- **Builds on:** ADR 0041 (the container), ADR 0038-0040 (the store), ADR 0018 (the clock lock),
  ADR 0021 (performance protocol of the operator engines).

## Context

ESCHER-GPU@abcf9b2 (A) keeps the line graph H2H of its hypergraph as a third CBST and maintains it
per batch with an exact, out-of-place sorted delta merge (`kernel/delta_merge.cu`,
`structure/graph_maintenance.cu`, `updateLineGraph`): the rows of the inserted hyperedges (the
union of their vertices' V2H rows, after the batch), the removals (every surviving neighbour of a
deleted hyperedge loses it), the mirrored additions (every neighbour of an inserted hyperedge that
was not inserted gains it), merged row by row into a new payload over which the tree is rebuilt
(`rebuildFromDevice`). Its counting kernels read the trees through key-indexed row views
(`RowLocator`, `include/row_view.hpp`), rebuilt once per batch; the initial line graph is built on
the host (`flatHyperedgeAdjacency`, `utils/utils.cpp:143`). MOSP_ESCHER@4b86159 (B) checks its own
line graph (a slack CSR, M11's) against an independent rebuild (`HypergraphOracle.cpp`,
`rebuildLineGraph`: the pairs of every vertex, sorted and deduplicated).

PLAN 5.2 sketched `line_graph_policy { none, automatic, sorted_merge, slack_csr }` and a
`line_graph_view` (frozen in M8b step 1: `row_ptr`, `col_ind`, `out`, `overlaps`, `weights`,
`version`); M9's `triad_count` reads the old h2h/h2v row views with frontier = deleted ids, then
the new ones with frontier = inserted ids, and uses `out` for oriented wedges (PLAN 6.4.6).

## Decision

### 1. The policies

| policy | the line graph |
|---|---|
| `none` | not kept; `h2h()` throws `invalid_argument_error` |
| `automatic` (default) | built from the current state by the first `h2h()` or by the first algorithm that needs it (`detail::hypergraph_access::require_line_graph`), maintained by every later `apply()` with `sorted_merge` |
| `sorted_merge` | built with the hypergraph (A's `init.h2h_build`) and maintained by every `apply()`; needs v2h (`store_v2h = false` is an `invalid_argument_error` at construction) |
| `slack_csr` | `not_supported_error` naming 0.4 (unchanged) |

The line graph is a property of the store: `clone()` copies it, `to_backend()` rebuilds it from
the rows on the new backend (the result is the same layout, section 3).

### 2. The port (PLAN 6.3; staged as in M8a)

`cpp/src/hypergraph/line_graph/{sorted_merge.cuh, sorted_merge.cu, device_frontier.cuh}` are A's
H2H merge, transplanted by `parity/escher_port/transplant.py --map
parity/escher_port/rename_map_line_graph.toml` (the rename map, the manual hunks of
`manual_hunks_line_graph.patch`: the rebuild takes the store's `buffer`s, three device
synchronizations against the legacy default stream, a comment the renames changed), then changed
by one class of mechanical change per commit: errors (the device error word and `capacity_error`
instead of `exit(2)`, the 64-bit scan of the merged sizes checked against the policy's payload
limit before the merged payload is allocated, the chain walks through `chain.cuh`), buffers (the
resources' memory resource, Thrust with `thrust_allocator`), streams (every kernel, copy and Thrust
call on the resources' stream, the host synchronizations at A's read points only, counted), and
registration (`DYNG_REGISTER_KERNEL`). The V2V merge (`updateVertexGraph`, `PairTest`, the
vertex-pair kernels) belongs to the coarse triangles of 0.4 and is not ported; the row heads
kernel and the chained-row kernels of `delta_merge.cu` were ported with the locator in M8a.

The host backends keep the same line graph with a host form of the same merge
(`line_graph/sorted_merge_host.cpp`: the same delta, merged row by row, OpenMP over the rows), and
both backends build the initial line graph with the port of `flatHyperedgeAdjacency`
(`line_graph/flat_adjacency.cpp`, deterministic). Both give **A's layout entry by entry**: the rows
of keys 1..n back to back, each padded to `(count + 4) / 4 * 4` slots with zeros and `INT_MIN`
last, the row offsets in key order. The H2H tree is constructed (and rebuilt by the merge) with
its payload sized exactly, under every capacity preset (A's `payloadCapacity` of a merged tree);
a line graph larger than the policy's payload limit (by default the int32 offsets, where A ended
the process) raises `capacity_error` (at construction, with the strong guarantee; in `apply()`,
after the h2v and v2h phases, poisoning the hypergraph).

### 3. The key count of the line graph

The H2H tree has the keys `1..max_bound`, where `max_bound` is the largest `hyperedge_id_bound()`
the hypergraph has had: A's `numHyperedgeKeys` (`h2v_.size()`, which never shrinks), while dynG's
bound can go down after a surplus rebuild (ADR 0041, section 2). The rows above the bound are
empty. With it the layout, the memory and the frontier flags of the merge (A's `DeviceFrontier`
over `numKeys + 1` ids, which must cover the ids deleted by the batch even when the bound went
down) are A's exactly. `store_snapshot` carries `max_bound` and whether the line graph is kept.

### 4. The views

- **Internal row views** (`cpp/src/hypergraph/row_views.hpp`, `detail::hyper::row_views`, read by
  M9's and M11's kernels through `detail::hypergraph_access::views(h)`): A's `RowView` of H2V
  (keys `1..bound`, members `+ 1`, dead and absent keys `loc = len = 0`) and of H2H (keys
  `1..max_bound`, neighbour keys, `out` = the offset of the first entry greater than the key),
  rebuilt **once per commit** with A's locator (`row_locator::build`, one pass over each tree's
  nodes; stage `hypergraph.apply.views`). The CUDA store's arrays are on the device (the trees'
  payloads and the locator's arrays); the host stores give the same type over host arrays with
  the same layout, so a kernel and its host loop read the same encoding. `row_view` moved from
  `escher/row_view.cuh` to this host-only header for that.
- **Public views** (`h2v()`, `v2h()`, `h2h()`): decoded copies (0-based ids, `int64` offsets, no
  sentinel), in the hypergraph's memory space, built on request **at most once per version** and
  never by `apply()`; `v2h()` sorts (A's `buildSortedCopy`). `line_graph_view::out` holds offsets
  into `col_ind`; `overlaps` stays empty under `escher` (ADR 0041, section 7); `weights` is the
  host array of the weights by id. The stores count their view builds (`view_builds`), which the
  tests check.
- `storage_stats::device_bytes` is A's `DynamicHypergraph::deviceBytes()` scope: the H2V, V2H and
  H2H trees and the H2V and H2H row views (the public views are in `view_bytes`). Without the
  line graph (`none`, or `automatic` before the first read) the H2H tree and view are absent.

### 5. The order of an apply and the guarantees

`hypergraph.apply` = validation, the h2v/v2h phases of ADR 0041 (in the `id_reuse` order), the
line-graph phase (`hypergraph.apply.line_graph`, only when the line graph is kept; A's
`maintain.h2h_merge`), the one read of the stores' device error words, the row views
(`hypergraph.apply.views`), the commit. The merge reads the old H2H row view (kept until the
merge has written the new payload) and the updated V2H; `old_view`s of the line graph stay valid
until the merge rebuilds the tree. A failure in the line-graph phase poisons the hypergraph
(ADR 0041, section 8).

### 6. The oracle

`dyng::testing::rebuild_line_graph` (`cpp/include/dyng/testing/line_graph_oracle.hpp`, tracked)
is MOSP_ESCHER's `rebuildLineGraph` algorithm in the C++ standard library only: per vertex every
ordered pair of its hyperedges, sorted, deduplicated. It shares no code with the container: its
object library is checked by `line_graph_oracle_isolation` (no project include but its header, no
undefined dynG symbol), as `cbst_host_model` is.

### 7. Parity of the line graph (PLAN 6.1 C2; criterion 7)

`parity/compare.py hypergraph_ids --line-graph` replays A's batch records with the line graph
kept and compares, after the build and every batch, the container's H2H layout with the input of
A's H2H construct or rebuild at the same point (row offsets and payload, by the FNV-1a hashes the
trace records; every set records them, `batches` and hash-only sets included), and, where the
trace has state digests (A's `make test`, coauth-DBLP 50K), every key's row hash and the rows the
digests record in full. Equal layouts mean equal rows by id, offsets and padding included. The same
run records the device bytes against A's `DynamicHypergraph::deviceBytes()` at each point and
fails above 1.05x (the gate of criterion 8).

## Consequences

- The CPU backends and CUDA hold the same line graph entry by entry, so `to_backend()` and the
  conformance kit's cross-backend checks compare layouts, not only row sets.
- `max_bound` makes a hypergraph whose bound went down keep empty H2H rows above it (as A does);
  they cost 4 payload slots and a node each.
- `automatic` adds no cost to a hypergraph whose line graph is never read; the first read pays
  the build (on CUDA: a host copy of h2v, the host build, the upload).
- M9 reads the old views before `apply()` and the new ones after it, through `views()`; M11's
  `slack_csr` line graph will fill the same internal type.
