# Row layouts, vertex batches and capacities

A `dyng::graph` stores its rows in one of three layouts (`graph_properties::layout`, the
enumeration `row_layout`; 0.4 added the second and third). The layout decides how a batch changes
the rows, which batches the graph takes, and which algorithms can read it. This page explains the
three, the vertex status and the net-effect batches that go with the slot layouts, the
capacities, and when to use which; the exact contracts are in {doc}`../api/cpp/graph`.

| | `compact` | `slotted` | `slack` |
|---|---|---|---|
| Built by | `from_edges()`, `from_csr()` | `with_capacity()` (a capacity per vertex) | `with_capacity()` (no, partial or full capacity hints) |
| Rows | exact CSR: row v is `[row_ptr[v], row_ptr[v + 1])` | a fixed slot range per vertex and a row end: row v is `[row_ptr[v], row_end[v])` inside its slots | slot ranges that move: a row that outgrows its slots moves to the tail of the slot arena |
| Batches | edge insertions, deletions and weight changes under any `batch_semantics` but `net_effect()`; a batch rebuilds the touched rows | vertex batches under `batch_semantics::net_effect()`: vertices arrive and leave, edges are appended | the same vertex batches, and vertex growth with `allow_vertex_growth` |
| Read by | every algorithm | `label_propagation` | `label_propagation` |
| Since | 0.1 | 0.4 (LabelPropagation-CUDA's device graph; ADR 0069) | 0.4 (the streaming mode; ADR 0103) |

## Compact rows

The layout of every graph built from an edge list or a CSR, and the only one every algorithm
reads: `sssp`, `mosp`, `cycle_count`, the tutorial algorithms and the compositions of
`dyng::update()`. A batch is normalized under the graph's `batch_semantics` (upserts, sets,
MOSP's or CycleEnumeration-GPU's rules; ADR 0010) and the touched rows are rebuilt, so the rows
stay an exact CSR with no spare room.

## Slot rows: `row_layout::slotted`

DynLP's device graph (LabelPropagation-CUDA's `DeviceGraph`): every vertex owns a fixed range of
slots, sized when the graph is built, and a row end; a batch appends the new entries of a row at
its end, and nothing ever moves. `graph::with_capacity(res, num_vertices, capacities, props)`
builds it with `capacities[v]` slots for vertex v (the published setup passes the stream's full
degrees, `vertex_arrival_stream::row_capacity`). An append that does not fit its row is a
`capacity_error` (an edge delivered twice, or capacities that are too small); the rows never
reallocate, which is what lets the CUDA backend keep them on the device between updates
(ADR 0074) and what makes the published setup fast.

**Vertex status.** The vertices of a slotted graph exist from the start (`num_vertices()` is the
stream's final count) and each has a state, `vertex_status` (`graph::vertex_status()`,
`graph_view::status`): `inactive` (never inserted yet), `active` (in the graph) or `deleted`
(removed by a batch, or retired by a deletion before it was ever inserted).
`graph::num_active_vertices()` counts the active ones. The rows of a deleted vertex stay where
they are; every reader skips entries whose neighbour is not active.

**Net-effect batches.** `batch_semantics::net_effect()` reads a batch as its effect on the vertex
set: it inserts `insert_vertices` (each with an optional seed label in `insert_vertex_labels`) and
deletes `delete_vertices`, and its effect is Ins \ Del (in list order) and Del ∩ V_t: an id in
both never enters the graph. Ids are single-use: an id that was inserted, or named in a deletion,
cannot be inserted again. The batch's edges are appended to the rows of both endpoints, in batch
order, when both endpoints are active after the batch; any other edge is dropped and counted
(`apply_summary::dropped_insertions`). Edge deletions are not supported: an edge leaves with its
vertex. These are LabelPropagation-CUDA's semantics (its `runDynLP` host pass and
`launchAppendEdges`), and a graph built by `with_capacity()` takes exactly these.

## Slack rows: `row_layout::slack`

For streams whose degrees are not known up front (ADR 0103). `with_capacity()` takes capacity
hints (none, some or all vertices); a row with hint k gets `ceil(k * (1 + headroom))` slots
(`graph_properties::headroom`, default 0.125, in [0, 16]); a vertex without a hint starts with
none. Then:

- **A row that overflows moves.** When a batch's appends do not fit a row, the row moves to the
  tail of the slot arena with all of its entries, in order, and new headroom; the old slots become
  dead. A move never changes the order in which a reader meets a row's entries.
- **Compaction at a documented bound.** When the dead slots after a batch's moves would outnumber
  the owned ones (2 x dead > arena), the batch compacts instead: every row is rewritten in vertex
  order without its entries to deleted vertices. At most once per batch, amortized (a compaction
  follows at least as many dead slots as it rewrites). A compaction changes the slot positions
  inside rows, and so the order in which the CUDA engines sum a row: labels after a compaction may
  differ from slotted rows' in the last bits (the oracles still hold).
- **Vertex growth.** With `batch_semantics::allow_vertex_growth`, an id at or beyond
  `num_vertices()` grows the graph (the new vertices inactive, with no slots).
  `label_propagation::update()` refuses growth (build the graph with the stream's final count).
- **Limits.** A batch fails with `capacity_error` only when the arena would pass the edge offset
  type's maximum (int32 offsets: 2^31 - 1 slots; use the int64 instantiation).

`apply_summary` reports `relocated_rows` and `compactions` per batch.

## The published setup and the streaming mode

label_propagation runs in two setups (the author's decision AD16 of 2026-10-28; PLAN 12.1 O9):

| | The published setup (the default) | The streaming mode (advanced) |
|---|---|---|
| What is known up front | the whole stream (its full degrees and its average weight) | nothing: vertices and edges arrive as they come |
| Graph | `with_capacity(stream.row_capacity)`, `row_layout::slotted` | `with_capacity()` with no or partial hints, `row_layout::slack` |
| tau (`options::similarity_threshold`) | the stream's published threshold (the full graph's average edge weight) | `std::nullopt`: the average weight of the current graph, maintained per update |
| Labels | the original's bits under `damped` (LabelPropagation-CUDA's flow) | bitwise slotted's while no compaction has run; the harmonic oracle holds on every batch |
| Performance | gated against the original (AD2) | reported against the published setup, never gated (500k: 1.39x the update time, 1.48x with the current tau, 1.59x the memory) |

Use the published setup to reproduce the paper or when the full graph is known (a batch replay of
a fixed dataset); use the streaming mode when the data really arrives over time (your own kNN
graph growing batch by batch: {doc}`../tutorials/streaming_label_propagation`). Moving between
them is two lines (the layout and the threshold).

## The compact-only algorithms

Every algorithm but `label_propagation` reads compact rows only. One central check refuses a
slotted or slack graph in `sssp`, `mosp`, `cycle_count`, `dynamic_bfs`, `triangle_delta` and the
compositions with a `not_supported_error` that names the algorithm, the layout and the fix (AD11;
ADR 0070), before anything changes: copy the edges into a compact graph,

```cpp
auto props = g.properties();
props.layout = dyng::row_layout::compact;
props.semantics = dyng::batch_semantics::upsert_last_wins();
auto compact = decltype(g)::from_csr(res, g.to_csr(res).view(), props);
```

(`row_order::append` and `multi_edges::allow` keep every entry in slot order). `triad_count` and `hyper_sssp` take hypergraphs and are not
affected. In 0.4 vertex batches and `net_effect()` exist only on slotted and slack graphs.

## Determinism

The layouts do not change an algorithm's determinism level (`determinism::bitwise`,
`exact_value` or `tolerance`; {doc}`../concepts/update_model`): the rows are deterministic
functions of the batches on every backend. What a layout can change is the **order of a sum**:
label_propagation's engines sum a row in the order of its slots, so slotted rows give the
original's bits, slack rows the same bits until a compaction rewrites a row, and the bits after a
compaction are again deterministic per backend and build (`damped`).
