# Hypergraphs and hyperedge ids

A hypergraph (`dyng::hypergraph`, `dyng.Hypergraph`) holds **hyperedges**: sets of vertices.
Like a graph it changes by batches, but a batch of a hypergraph deletes and inserts whole
hyperedges, and the **ids** of the inserted hyperedges are chosen by the store. This page explains
those ids, the order of a batch, how the storage is sized, the limits of the index types and the
materialized line graph. The exact contracts are on {doc}`../api/cpp/hypergraph`; the decisions
behind them are ADRs {doc}`../adr/0041-hypergraph-container`,
{doc}`../adr/0042-sorted-merge-line-graph-and-row-views` and
{doc}`../adr/0047-hypergraph-api-review-and-freeze-0-3`.

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
skipped operation, in 0.3 their sum (the incidence edits of 0.4 add the edits they skip).

## `id_reuse`: the order of a batch

Whether a slot freed by a batch can be reused by the same batch depends on the order in which the
batch's deletions and insertions reach the store, a property of the hypergraph:

- `id_reuse::erase_first` (the default; the ESCHER paper's order): deletions first, so an
  insertion can take the slot of a hyperedge the same batch deleted;
- `id_reuse::insert_first` (H-SOSP's order, MOSP_ESCHER): insertions first, so only slots that
  were already dead before the batch can be reused.

**A worked example.** Three hyperedges `{0, 1, 2, 3}` (id 0), `{4}` (id 1) and `{5, 6}` (id 2)
under `capacity_policy::escher_gpu()`; a row of `k` members takes a slot of `k + 1` entries
rounded up to a multiple of 4, so id 0 has a slot of 8 and ids 1 and 2 slots of 4 (room for 3
members). One batch deletes id 1 and inserts `{7, 8}`:

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
empty hyperedge, a bad weight, an incidence edit before 0.4, a batch of more than `INT_MAX / 3`
inserted members, host memory running out while the batch is checked) change nothing.

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
- `slack_csr` (H-SOSP's slack rows) is planned for 0.4.

Each row of `h2h()` is sorted and contiguous, and `out[r]` is the offset of its first neighbour
with an id greater than `r`, so a loop over `col_ind[out[r] .. row_ptr[r + 1])` visits each pair
of overlapping hyperedges once, from its smaller id. The line graph needs the vertex-to-hyperedge
rows (`store_v2h = true`), and it costs memory: on coauth-DBLP its payload is about 258 million
entries (1 GB), about half of the container's device memory. `dyng::testing::rebuild_line_graph` recomputes it
from scratch, the oracle the tests compare the maintained one with.

The views (`h2v()`, `v2h()`, `h2h()`) are built from the store on first use per version (each
carries the `version` it describes) and are invalidated by the next `apply()`; `apply()` itself
never builds them. In Python they are host copies.
