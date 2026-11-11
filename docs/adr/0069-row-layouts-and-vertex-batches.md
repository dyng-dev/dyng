# ADR 0069: Row layouts and vertex batches: float graphs, `row_layout::slotted` with `graph::with_capacity`, `batch_semantics::net_effect()` and the slack representation

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs") for its technical choices. It carries out M10a step 3, the host path (PLAN 5.2, 5.9,
  4.7.3, 6.4.7; the M10a acceptance criterion 9). The surface it adds follows the author's
  decisions of 2026-10-28: AD11 (vertex batches and `net_effect` exist only on slotted and slack
  graphs in 0.4; edge deletions under `net_effect` are not supported; one central check refuses
  the new layouts elsewhere, ADR 0070) and AD12 (the names: `vertex_status` with
  `graph_view::status` and `graph_view::row_end`, `graph::with_capacity`,
  `graph::num_active_vertices`, `batch_semantics::net_effect()`, `edge_batch::insert_vertex` and
  `delete_vertex`), **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, the
  rows "The new public names of 0.4 (AD12, HD8, HD12; sheet E1)" and "API semantics of 0.4
  (sheet E2-E10)", AD11 being sheet E5), and the comparator rules of sheet C8 (statuses map by
  name; O's GROUND_TRUTH is "active with a seed label"; the row "The parity rules of 0.4 (sheet
  C1-C8)"). The names this
  ADR adds beyond AD12's list are listed in section 9 and in GOVERNANCE.md's open-decision
  sub-block `M10 (AD)`, pending the author's acknowledgement.
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  host-container-sequential-driver)

## Context

LabelPropagation-CUDA@a276a3a (O) keeps its dynamic graph on the device as an append-only CSR with
a fixed slot layout (`headers/dynamicGraph.h`, `DeviceGraph`): vertex v owns the slots
`[rowPtr[v], rowPtr[v+1])`, sized for its degree in the full graph, and its current neighbours are
`[rowPtr[v], rowEnd[v])`. A batch inserts vertices (some of them ground truth, with a class),
deletes vertices and brings the new edges of its arrivals; nothing is re-uploaded, rows only grow,
a deleted vertex keeps its row and the entries pointing to it, and every kernel skips them by the
vertex status. Its input contract (single-use ids, every edge delivered once) is checked on the
device with sticky error flags and on the host for the ids of Ins ∩ Del.

dynG 0.1-0.3 has one layout (`row_layout::compact`: an exact CSR rebuilt per batch) and edge batches
only; the vertex arrays of `edge_batch_view` exist since 0.1 and were refused. PLAN 5.2 reserved
`row_layout::slotted` and `slack`, `graph::with_capacity` and `batch_semantics::net_effect()` for
DynLP. This ADR adds them on the host; the resident device path (O's stable radix-sort append)
comes in M10a step 4 and is recorded there.

## Decision

### 1. Float graphs

`graph<int32, int32, float>` and `graph<int32, int64, float>` are instantiated (with
`edge_batch<V, float>` / `edge_batch_view<V, float>`, `from_edges`, `from_csr`, `to_csr`,
`check_integrity`, `clone`, `with_capacity` and `run_update`): a new instantiation list
`DYNG_FOR_EACH_FLOAT_GRAPH_TYPE` (`cpp/src/graph/instantiate.hpp`) beside the existing ones, which
are unchanged; the algorithms of 0.1-0.3 do not instantiate float graphs. A float graph takes the
host apply on every backend; the device merge of `batch_semantics::set()` (unweighted graphs only)
is excluded for float weights at compile time, and the compact device copy is instantiated for
them (the upload of a compact float graph works as for the other weights).

### 2. `row_layout::slotted` on the host

- **Storage.** The slot ranges are the prefix sum of the row capacities (the graph's host CSR:
  `row_ptr`, and `col_ind` / `weights` over every slot, -1 / 0 in the unused ones, as O's
  `createDeviceGraph` clears them); next to it a row end and a state per vertex, the counts of
  active vertices and live entries (so the queries are O(1)), and two scratch arrays of Step 0 (a
  byte and a 32-bit counter per vertex, all zero between calls).
- **Requirements in 0.4** (anything else raises `not_supported_error` at construction, naming the
  property): `directed = false` (every edge is stored in both rows, as O does), `row_order::append`
  (rows receive their entries in batch order), `multi_edges::allow` (no duplicate check, O's
  contract: a duplicate is caught only when it overflows a row), and `semantics =
  batch_semantics::net_effect()` exactly. `num_weights` must be 1 (0 for unweighted graphs);
  `store_transposed` is set to false (the rows are symmetric; no in-edges are kept). Every graph
  type can be slotted (only label_propagation, on float graphs, reads it).
- **`graph::with_capacity(res, vertex_capacity, array_view<const edge_t> row_capacity, props)`**:
  an empty graph with every vertex inactive and every row empty. A negative or missing capacity
  raises `invalid_argument_error`; a sum of capacities beyond `edge_t` raises `capacity_error`
  (naming the int64 instantiation) before anything is allocated. `from_edges`, `from_csr` and the
  empty constructor refuse slotted rows (they name `with_capacity`), and `net_effect()` on compact
  rows.

### 3. `vertex_status`, `graph_view::status`, `graph_view::row_end`, `num_active_vertices`

`enum class vertex_status : std::uint8_t {active, deleted, inactive}` (in `graph_view.hpp`), and two
fields appended to `graph_view`: `status` and `row_end` (array views, empty for compact rows). The
type and the field cannot share one name; this pair is AD12's decided naming. For a slotted graph,
`view().out` describes the slot storage (row v is `[out.row_ptr[v], row_end[v])`; an entry is an
edge only when both endpoints are active; `in` is empty). `graph::num_active_vertices()` is the
number of active vertices (`num_vertices()` for compact rows, whose vertices are all active).

**The comparator mapping (sheet C8, decided).** O's statuses map by name, never by numeric value:
O `VERTEX_GROUND_TRUTH` (2) = (active, seed label = gtClass); O `VERTEX_ACTIVE` (0) = active with
no seed label; `VERTEX_DELETED` (1) = deleted and `VERTEX_INACTIVE` (3) = inactive. The graph keeps
no labels: a ground-truth vertex is active, and its class travels as a seed label in the batch
(`edge_batch_view::insert_vertex_labels`) and the applied record to the algorithm's result
(label_propagation's `seed_labels()`). Accuracy and the propagation frontier ("ACTIVE and not GT")
are therefore "active and without a seed label".

### 4. `batch_semantics::net_effect()` and the vertex operations of `edge_batch`

- `batch_semantics::net_effect()` is a factory; one field is appended after `as_sets`:
  `bool as_net_effect = false` (a member cannot share the factory's name). The factory sets
  `as_net_effect`, `self_loop::keep`, `deletions_first`, no vertex growth; the edge-level switches
  do not apply (every live edge is appended; edge deletions are refused).
- `edge_batch::insert_vertex(v, std::int8_t label = -1)` and `delete_vertex(v)`, with the
  accessors `insert_vertices()`, `insert_vertex_labels()` and `delete_vertices()`; `empty()`,
  `clear()` and `view()` cover them (the view's `insert_vertex_labels` holds one label per inserted
  vertex). `generators::vertex_arrival_batch` gains `view()`, the same view of a generated batch.

### 5. Applying a vertex batch (Step 0, then the commit)

Step 0 runs on the host, once per update for every participant (`run_update` computes it after the
staging, before any participant's `before_apply`, under the stage `<algo>.normalize`; it is stored
with the edge Step 0 of `as_sets` in the update's `normalized_batch` and passed to the commit), and
checks, in this order, before anything changes (strong guarantee):

1. edge deletions → `not_supported_error`; the shape (labels empty or one per inserted vertex: "a
   label on a vertex not being inserted"; the edge arrays and weights);
2. every id and edge endpoint in `[0, num_vertices())`;
3. labels in {-1, 0, 1};
4. an id listed twice in the deletions, or twice in the insertions;
5. an inserted id that is not inactive (single-use ids: an id ever inserted, or named in an earlier
   `delete_vertices`, cannot be inserted);
6. a labelled insertion that the batch also deletes (GT ∩ Del);
7. capacity: the entries each row receives (both endpoints of an edge active after the batch)
   against its free slots → `capacity_error`.

1-6 raise `invalid_argument_error`. Then the commit, which cannot fail any more (the record's arrays
are reserved first): the deletions (each active id of Del loses its live entries, and every id of
Del ends deleted: an id in Ins ∩ Del never enters the graph, a never-inserted id is retired;
exactly as O), the effective insertions Ins \ Del become active (their rows are empty: single-use),
then the appends in batch order: for edge e, its other endpoint into u's row, then u into v's row
(O's half-edges 2e and 2e+1 after the stable sort by source), for every edge whose two endpoints
are active after the batch; any other edge is dropped and counted (`apply_summary::
dropped_insertions`, appended, per stored direction). Edge-only insertions between present vertices
are accepted. `apply_summary` counts `inserted_vertices` = |Ins \ Del|, `deleted_vertices` = |Del ∩
V_t|, `inserted_edges` = the entries appended, `num_vertices_after` = the capacity.

**The applied record** (`apply_delta`, internal): `inserted_vertices` (Ins_eff in list order) with
`inserted_labels`, `deleted_vertices` (Del_eff in list order) and, as insertions, the appended edges
once each as (u, v) of the batch in batch order with `insert_index` (their position in the batch,
for the weights). This is the framework's `applied_batch::delta` for label_propagation's hooks.

### 6. Queries, and why

| Query | Slotted rows | Reason |
|---|---|---|
| `num_vertices()` | the vertex capacity | the ids are `[0, capacity)` whatever their state; it is what arrays indexed by vertex need |
| `num_active_vertices()` | the active vertices | the size of the current vertex set |
| `num_edges()` | the live entries (both endpoints active; an undirected edge twice) | it keeps the frozen meaning "stored directed edges" of the current graph and equals `to_csr().num_edges()`, as for compact rows |
| `to_csr()` | per row the live entries in slot order (inactive and deleted vertices have empty rows) | the compacted current graph, in the order the propagation reads it |
| `view()` | the slot storage with `status` and `row_end` | the zero-copy description the engines read |
| `check_integrity()` | rows inside their slots, row ends coherent, unused slots -1 / 0, no entry to a never-inserted vertex, inactive vertices without entries, the entries symmetric, the two counts, the scratch clear | O's contract as invariants |
| `clone()` | a deep copy of everything | as for compact rows |
| `reserve()` | checks the argument, changes nothing | the slots are fixed by `with_capacity` |

### 7. O's nine input-contract cases (`tests/input_contract_test.cu`), mapped

| Case | O | dynG | Difference |
|---|---|---|---|
| valid | accepted | accepted | none |
| insdel (inserted and deleted in one batch, once) | accepted | accepted; the id ends deleted | none |
| reinsert (deleted in b1, inserted in b2) | rejected after the batch (host check / sticky flag) | `invalid_argument_error`, graph unchanged | refused before anything changes |
| dupid (inserted in b0 and b1) | rejected (device flag, read after the batch) | `invalid_argument_error`, graph unchanged | refused before the batch is applied (O has applied it) |
| dupsame (listed twice in one batch) | rejected (device flag) | `invalid_argument_error`, graph unchanged | as dupid |
| range (a deleted id outside [0, N)) | rejected (host) | `invalid_argument_error`, graph unchanged | none |
| dupedge (the same edge twice in one batch) | rejected after the batch (row-overflow flag; the entry was dropped) | `capacity_error`, graph unchanged | refused before the batch is applied |
| insdelre (inserted in b0, inserted and deleted in b1) | rejected (host) | `invalid_argument_error`, graph unchanged | none |
| insdel2 (listed twice in a batch that deletes it) | rejected (host) | `invalid_argument_error`, graph unchanged | none |

Further differences, each a refusal of an input O accepts or treats otherwise: a deletion listed
twice in one batch (`invalid_argument_error`; O accepts it), a labelled vertex deleted in the same
batch (`invalid_argument_error`; O's device would set GROUND_TRUTH on a deleted id), a label other
than -1, 0, 1 (O thresholds a float at 0.5), and an edge endpoint out of range (O's kernels skip
or read out of bounds). No `exit()`: O's `launchAppendEdges` exits when a batch exceeds its
append scratch; dynG sizes the scratch per batch.

### 8. The slack representation (designed here, implemented in M10b)

So that M10b does not redesign `graph_view`: a slack graph is a slotted graph whose capacities
are not known up front. Each row gets `ceil(degree * (1 + headroom))` slots (at least 1;
`graph_properties::headroom`, default 0.125) when it is first written; a row that overflows moves
to a tail region appended after the last slot (its old slots become garbage, -1 / 0, its
`row_ptr[v]` / `row_end[v]` point into the tail, so `row_ptr` is no longer monotone: `view().out`
of a slack graph describes row v as `[row_start(v), row_end[v])` with the row start in a third
array, appended to `graph_view` in M10b as `row_start` (empty for compact and slotted rows, whose
row start is `out.row_ptr[v]`); compaction rewrites the rows contiguously when the garbage exceeds
the live slots (one pass, rows in vertex order, counted as container growth, I9). Vertex batches,
`net_effect()`, the states and the queries are those of slotted rows; edge deletions stay
unsupported in 0.4.

### 9. Names beyond AD12's list (pending the author's acknowledgement)

`batch_semantics::as_net_effect` (the field the factory sets), `apply_summary::dropped_insertions`,
`edge_batch::insert_vertices()`, `insert_vertex_labels()`, `delete_vertices()` (the accessors of the
new arrays, as `insert_src()` etc.), `generators::vertex_arrival_batch::view()`, and the enumerators
`vertex_status::{active, deleted, inactive}`. They are listed in GOVERNANCE.md's open-decision
sub-block `M10 (AD)` and in the API ADR's list (ADR 0071).

## Consequences

- The container replay (M10a step 3; `compare.py label_propagation container`): O's recorded
  batches applied to `with_capacity` reproduce the trace's `row_end`, slot columns and value bits
  and the statuses (by the mapping of section 3) after every batch of P+, and `check_integrity`
  agrees with `lp_host_model` (also a CPU test on a generated 5k5 stream).
- Every algorithm of 0.1-0.3 refuses slotted rows (ADR 0070); label_propagation reads them (ADR
  0071).
- The device path (step 4) keeps this host contract: the same Step 0 on the host, O's append on
  the device; its overflow (a contract violation that Step 0 does not see) is read from the sticky
  word once per update inside an existing synchronization and poisons the graph.

## Alternatives considered

- **Detecting the contract violations on the device after the batch, as O does.** Rejected for the
  host path: the strong guarantee costs one pass over the batch (O already makes one, b1).
- **`num_edges()` = undirected edges (each once).** Rejected: the frozen documentation counts an
  undirected edge twice, and `num_edges() == to_csr().num_edges()` holds for every layout.
- **A separate container type for DynLP.** Rejected by PLAN 5.2 (one container with layouts) and
  ADR 0010 (the batch semantics belong to the graph).
