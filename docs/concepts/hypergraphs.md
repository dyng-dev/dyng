# Hypergraphs and hyperedge ids

A hypergraph (`dyng::hypergraph`, `dyng.Hypergraph`) holds **hyperedges**: sets of vertices.
Like a graph it changes by batches, but a batch of a hypergraph deletes and inserts whole
hyperedges (and, from 0.4 on `slack_csr` storage, removes vertices from and adds vertices to live
hyperedges), and the **ids** of the inserted hyperedges are chosen by the store. This page explains
those ids, the order of a batch, the two storages and the incidence edits, how the storage is
sized, the limits of the index types and the materialized line graph. The exact contracts are on {doc}`../api/cpp/hypergraph`; the decisions
behind them are ADRs {doc}`../adr/0041-hypergraph-container`,
{doc}`../adr/0042-sorted-merge-line-graph-and-row-views`,
{doc}`../adr/0047-hypergraph-api-review-and-freeze-0-3` and, for 0.4's `slack_csr` storage and
incidence edits, {doc}`../adr/0085-incidence-edits-and-the-allocator` and
{doc}`../adr/0086-host-slack-csr-store-and-line-graph`.

## Ids are 0-based, and -1 means none

Vertex ids lie in `[0, num_vertices())`; hyperedge ids are `int64` and lie in
`[0, hyperedge_id_bound())`. Hyperedge `i` of `from_hyperedges()` (or of `from_lists()` in Python)
gets id `i`. A deleted hyperedge leaves a **dead id**: its row in `h2v()` is empty, and it is
nobody's neighbour in the line graph. Wherever an id may be absent, `-1` stands for "none" (as
for the parents of `sssp`). The store keeps its own keys internally (ESCHER's trees count from 1
and use sentinels in their payload); none of that appears in the API.

## The ids of a batch come from the store

A batch (`hyperedge_batch`, `dyng.HyperedgeBatch`) lists ids to delete and member lists to insert.
`apply()` (or `dyng::update()`) returns a `hyper_apply_summary` whose `inserted_ids` holds the id
of every inserted hyperedge, in batch order. **Read the ids there; never predict them.** The
ESCHER store reuses the slots of deleted hyperedges by **best fit** (ESCHER+, TKDE'26): an
inserted hyperedge goes into a dead slot whose capacity fits it, the smallest that does (equal
capacities in id order), and takes that slot's id. When some inserted hyperedges fit no dead slot
(they are **surplus**), the batch drops every dead slot it did not reuse and gives the surplus
hyperedges the ids after the last live id, in batch order. That is the **surplus drop rule**: it
is why `hyperedge_id_bound()` **can go down** after a batch (dead ids above the last live id
disappear), why a dead id below the bound may never be reused, and why an id that was dead before
a batch can be given out again.

Repeated members of an inserted hyperedge are dropped (`dropped_members`), and deletions of dead,
out-of-range or repeated ids are skipped (`ignored_deletions`); `skipped_ops` counts every
skipped operation: their sum, plus the incidence edits a batch skips (0.4; below).

## `id_reuse`: the order of a batch

Whether a slot freed by a batch can be reused by the same batch depends on the order in which the
batch's deletions and insertions reach the store, a property of the hypergraph:

- `id_reuse::erase_first` (the default; the ESCHER paper's order): deletions first, so an
  insertion can take the slot of a hyperedge the same batch deleted;
- `id_reuse::insert_first` (H-SOSP's order, MOSP_ESCHER): insertions first, so only slots that
  were already dead before the batch can be reused.

**A worked example.** Three hyperedges `{0, 1, 2, 3}` (id 0), `{4}` (id 1) and `{5, 6}` (id 2)
under `capacity_policy::escher_gpu()`; at construction a row of `k` members takes a slot of
`k + 1` entries rounded up to a multiple of 4 (always; `row_alignment` does not set it), so id 0
has a slot of 8 and ids 1 and 2 slots of 4 (room for 3 members). A surplus row inserted later by
a batch reserves its members rounded up to a multiple of `row_alignment`, plus one slot. One batch deletes id 1 and inserts `{7, 8}`:

| `id_reuse` | what the store sees | the id of `{7, 8}` |
|---|---|---|
| `erase_first` | id 1 is deleted first; its dead slot (room for 3) fits `{7, 8}` | **1** |
| `insert_first` | `{7, 8}` is inserted first; no slot is dead yet, so it is surplus and goes after the last live id; then id 1 is deleted | **3** |

<!-- snippet: hypergraph-id-reuse -->
```python
import dyng

for reuse in ("erase_first", "insert_first"):
    h = dyng.Hypergraph.from_lists(
        [[0, 1, 2, 3], [4], [5, 6]], num_vertices=9, id_reuse=reuse, capacity="escher_gpu"
    )
    s = h.apply(dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[7, 8]]))
    print(reuse, s.inserted_ids.tolist())
```
<!-- snippet-output: hypergraph-id-reuse -->
```text
erase_first [1]
insert_first [3]
```

The default is `erase_first` (the author's decision O16, the plan's default); the parity runs
against H-SOSP set `insert_first`. Results that depend on ids (shortest hyperpaths) are compared
across the two orders by content (the sorted member set and the weight), not by id.

## Two storages: `escher` and `slack_csr` (0.4)

`hypergraph_properties::storage` (Python `storage=`) chooses how the incidences are kept:

- `hypergraph_storage::escher` (the default): ESCHER's CBST store, rows sorted for ESCHER-GPU's
  line-graph merge; the storage of `triad_count` and of 0.3. It refuses incidence edits.
- `hypergraph_storage::slack_csr` (0.4): MOSP_ESCHER's H-SOSP storage: the ids of the ESCHER
  store's best-fit reuse (the same id allocator), the incidence lists, and a **slack CSR line
  graph**; it takes incidence edits, and it is the storage of `hyper_sssp`. It needs the
  vertex-to-hyperedge rows (`store_v2h = true`, else `invalid_argument_error`).

Each storage has its line graphs; the other two combinations raise `not_supported_error` at
construction, naming the fix:

| `storage` \ `line_graph` | `none` | `automatic` | `sorted_merge` | `slack_csr` |
|---|---|---|---|---|
| `escher` | no line graph | built on first read, kept as `sorted_merge` | built and kept | `not_supported_error` |
| `slack_csr` | no line graph | built on first read, kept as `slack_csr` | `not_supported_error` | built and kept |

**When to choose which.** `escher` for triad counts (`triad_count` refuses `slack_csr` storage in
0.4, so the two algorithms do not share a hypergraph) and for batches of whole hyperedges with
ESCHER-GPU's ids; `slack_csr` for shortest hyperpaths (`hyper_sssp` refuses `escher`) and whenever
a batch changes the members of existing hyperedges. Both run on every backend.

**The slack CSR line graph.** Each hyperedge id has a row of its neighbours' ids in a slot with
headroom (a multiple of 32 entries, at least a quarter more than its degree): a batch's pair
deletions swap-remove from the rows, its insertions append, a row that outgrows its slot moves to
the tail of the array with a new slot, and a tail that cannot take the moves rebuilds the whole
array from the incidence lists with the same headroom. The rows are sets but not
sorted: `h2h()` sorts a copy of each row on request (the same view as under `sorted_merge`), and
`hypergraph_view::overlaps` is empty under this storage. The sizes are internal (MOSP_ESCHER's
headroom of 1.6; no public capacity field, HD9); an entry count past 2^31 - 1 is a
`capacity_error`.

## Incidence edits (0.4, `slack_csr` storage)

A batch can remove a vertex from a live hyperedge (`hyperedge_batch::delete_incidence(h, v)`;
`.dgt` line `-i h v`; Python `delete_incidences=(hyperedges, vertices)`) and add one
(`insert_incidence(h, v)`; `+i h v`; `insert_incidences=`). The hyperedge keeps its id and weight.

**The order of a batch** is fixed, whatever the order of its arrays (MOSP_ESCHER's phases): the
hyperedge deletions, then the incidence removals, then the incidence additions (each in batch
order), then the insertions. Every edit is checked against the hypergraph as it is after the
deletions, and is **skipped and counted** in `skipped_ops` when:

- its hyperedge is negative, at or above the id bound, dead, or deleted by the same batch (an id
  the batch inserts is one of these: the insertions come last);
- its vertex is outside `[0, num_vertices())` (an out-of-range member of an *inserted* hyperedge
  is an `invalid_argument_error` instead);
- a removal names a vertex that is not a member (a second removal of the same vertex is one);
- a removal would empty its hyperedge (the last member stays; delete the hyperedge instead);
- an addition names a vertex that is already a member (a repeat is one).

The summary counts the applied ones (`inserted_incidences`, `deleted_incidences`), so applied plus
skipped equals requested. Under escher storage a batch with an edit raises `not_supported_error`
(naming `slack_csr`) before anything changes. A changed hyperedge changes the line graph: the pairs
that start or stop sharing a vertex form the batch's **pair delta**, computed inside the commit
from the incidence changes, exact whatever the order of the operations (`hyper_sssp`'s
`stats::line_graph_insertions` / `line_graph_deletions` count it).

**Ids with edits.** An edit never allocates an id, but it changes a row's length, and the id
allocator models it: a removal leaves room in the hyperedge's slot (an unfill), an addition fills
it or opens an overflow segment (a fill), and these decide where later compactions put the rows,
hence the slots that best fit reads, hence the ids of later insertions. Under `id_reuse` the store
sees the edits between the two halves of the batch: `erase_first` erases the deleted ids, applies
the removals and the additions, then inserts; `insert_first` (MOSP_ESCHER's order) inserts first,
then erases and applies the edits. Read the ids from `inserted_ids`; the parity harness replays
MOSP_ESCHER's sessions with their ids under `insert_first`.

## Capacity presets

The store keeps each incidence relation (h2v, v2h, and the line graph) in a payload array that is
sized when it is built and grows when it is full. `capacity_policy` says how:

| preset | start | when full |
|---|---|---|
| default `capacity_policy{}` | `size + size / 8 + 1024` entries | compact first (MOSP_ESCHER's trigger points), then grow by 1.5x, with headroom after a compaction |
| `capacity_policy::escher_gpu(min_initial_payload)` | ESCHER-GPU@abcf9b2: `size + size / 8 + 1024` (at least its `--payload-capacity`) | grow by its rule up to 2^31 - 1, never compact |
| `capacity_policy::mosp_escher(extra)` | MOSP_ESCHER@4b86159: `floor(size * 1.3) + extra` (at most 2.0e9) | compact at its trigger points, never grow: `capacity_error` |

The presets reproduce the originals' layouts and memory exactly, which is what the parity tests
and the memory gate need; the default is the policy to use otherwise. A policy decides memory,
speed and where `capacity_error` can occur; the rows of the live hyperedges are the same under
every policy.

## Index limits: int32 offsets and `capacity_error`

The ESCHER store addresses its payload with **int32 offsets**, so one relation holds at most
2^31 - 1 entries (`max_payload = 0` means exactly that limit; `mosp_escher()` stops at 2.0e9).
An `apply()` that would need more raises `capacity_error`, and so does the line graph when it
outgrows the limit. An error inside a store operation leaves the hypergraph **poisoned**: every
later call that reads or changes its storage (the views, `weights()`, `apply()`,
`to_hyperedge_list()`, `clone()`, `check_integrity()`) throws until it is destroyed or assigned,
and the `noexcept` accessors (`num_hyperedges()`, `version()`, `store_stats()`, ...) return
unspecified values, because a half-applied batch cannot be undone cheaply on the device. Errors found before the store changes (an out-of-range vertex, an
empty hyperedge, a bad weight, an incidence edit under escher storage, a batch of more than
`INT_MAX / 3` inserted members, host memory running out while the batch is checked) change nothing.

Host memory running out inside one of the container's OpenMP loops on the sequential or openmp
backend (the vertex-to-hyperedge rows that `apply()` changes, the line graph's build) is the same
`out_of_memory_error`, with the same guarantee: the loop stops, and the error is raised after it
on the calling thread (D13, fixed in 0.3.0rc1; {doc}`../adr/0061-exceptions-out-of-openmp-regions`).

On CUDA a hypergraph's storage works on the stream of the `resources` it was built with.
`apply()` (and `dyng::update()`) with resources on another stream of the same device is ordered
after the work already queued on that stream and before the work queued there later, so the
views of G_t may be read on either stream until the call.

The public offsets (`incidence_view::row_ptr`, the batch's `insert_offsets`) are `int64`
regardless, so the API does not change when a later storage lifts the limit.

## Vertex ids: sizing the vertex type

Hypergraphs are instantiated with `int32` vertex ids (with or without `int32` weights). Vertex ids
must lie below 2^31 - 1. A `.hg` file may name larger ids: read it with
`relabel = dense_sorted` or `first_appearance` (`io::hypergraph_read_options`), which numbers the
distinct ids 0, 1, ... and returns the original ids in `hypergraph_file_info::external_ids`. In
Python, member arrays of NumPy's default `int64` type are refused with `NotSupportedError` unless
`vertex_dtype="int32"` is given, under which every value is range-checked: ids are never
narrowed silently.

## The materialized line graph

The **line graph** (h2h) joins two hyperedges when they share a vertex. dynG keeps it
**materialized**: one sorted row per hyperedge id, maintained batch by batch, because the
algorithms on hypergraphs (triad counting, shortest hyperpaths) walk it over and over.

- `line_graph_policy::sorted_merge` builds it with the hypergraph and keeps it with every batch
  (ESCHER-GPU's out-of-place sorted delta merge on CUDA, a host form with the same layout on the
  CPU backends);
- `automatic` (the default) builds it the first time something reads it (`h2h()`, or an algorithm
  that needs it), then keeps it;
- `none` never builds it: `h2h()` throws;
- `slack_csr` (0.4, with `hypergraph_storage::slack_csr`) is MOSP_ESCHER's H-SOSP line graph: a
  slack CSR whose rows are sets kept by every batch from its order-free pair delta (the pairs that
  start or stop sharing a vertex), so incidence edits (a vertex removed from or added to a live
  hyperedge) keep it exact. `automatic` under slack_csr storage maintains it as `slack_csr`.
  Incidence edits need slack_csr storage in 0.4 (escher storage refuses them), and triad_count
  reads escher storage only (section "Two storages" above).

Each row of `h2h()` is sorted and contiguous, and `out[r]` is the offset of its first neighbour
with an id greater than `r`, so a loop over `col_ind[out[r] .. row_ptr[r + 1])` visits each pair
of overlapping hyperedges once, from its smaller id. The line graph needs the vertex-to-hyperedge
rows (`store_v2h = true`), and it costs memory: on coauth-DBLP its payload is about 258 million
entries (1 GB), about half of the container's device memory. `dyng::testing::rebuild_line_graph` recomputes it
from scratch, the oracle the tests compare the maintained one with.

The views (`h2v()`, `v2h()`, `h2h()`) are built from the store on first use per version (each
carries the `version` it describes) and are invalidated by the next `apply()`; `apply()` itself
never builds them. In Python they are host copies.
