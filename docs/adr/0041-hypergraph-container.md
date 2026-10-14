# ADR 0041: The hypergraph container on the ESCHER store

- **Status:** Accepted under delegation (2026-10-13; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8b step 1 (PLAN 5.2, 4.4.2, 4.4.3, 4.6 rules 5 and 7, 4.7.3, 4.7.5,
  6.4.5, 11.3; the M8b acceptance criteria 1-6 and 12) and settles the three open questions of
  `docs/design/sketches/hypergraph.md`. It changes no rule the author approved: the parity rules
  (PLAN 8.3), the gates and their protocol (PLAN 8.6), the names fixed by the plan
  (`hypergraph`, `hypergraph_storage::escher`, `line_graph_policy::sorted_merge`, `id_reuse`,
  `capacity_policy::escher_gpu()` / `mosp_escher()`, `hyper_apply_summary`) and the author's
  decisions of 2026-10-07 (O16: `erase_first` is the default; the memory gate is read as store
  accounting) are kept as they are.
- **Date:** 2026-10-13
- **Deciders:** the AI assistant, on the author's behalf (M8b, step container-core)
- **Number:** 0036-0040 are M8a's; 0041 is the first of M8b.

## Context

M8a (ADRs 0036-0040) delivered the internal ESCHER CBST store `dyng::detail::escher::cbst_store`
(CUDA only, 1-based keys, `+1`-shifted values, sentinels, int32 offsets), its presets
`store_policy::escher_gpu()`, `mosp_escher(extra)` and `dyng_default()`, and the oracle
`dyng::testing::cbst_host_model`. M8b puts the public `hypergraph<vertex_t, weight_t>` on it, for
the three backends, and freezes its headers at the end of the milestone (step 5, ADR 0023's
method).

PLAN 5.2 sketched the container in 0.1; the sketch `docs/design/sketches/hypergraph.md` (reviewed
in M3 against the frozen 0.1 contract) refined it and left three questions open. The two
originals do not agree on the order of a batch: ESCHER-GPU@abcf9b2 (A) erases first
(`structure/dynamic_hypergraph.cu:193-274`: H2V erase, V2H unfill, H2V insert, V2H fill with the
keys the insert assigned), MOSP_ESCHER@4b86159 (B) inserts first
(`hypergraph/src/DynamicHypergraph.cpp:282-525`: the H2V insert in `beginBatch` decides the ids,
`finishBatch` then erases, unfills and fills). Because the store's best-fit reuse (ESCHER+) picks
the ids, the two orders give different ids for the same batch (section 9 has an example), and an
id cannot be predicted by the caller: the surplus rule of an insert drops the deleted slots that
were not reused and numbers the new rows after the last live key, so it can hand out a dead id
again, or a smaller id bound than before.

## Decision

### 1. PLAN 5.2 and the sketch, item by item

| Item | PLAN 5.2 | Sketch | Decided |
|---|---|---|---|
| headers | one sketch block | six headers under `dyng/hypergraph/` | the sketch's six: `hypergraph_properties.hpp`, `hyperedge_list.hpp`, `hypergraph_view.hpp`, `hyperedge_batch.hpp`, `hyper_apply_summary.hpp`, `hypergraph.hpp`; host-only, in `dyng.hpp`, `tracked` until the freeze of step 5 |
| `hypergraph_storage` | `escher`, `slack_csr` | same | same; `slack_csr` raises `not_supported_error` naming 0.4 |
| `line_graph_policy` | `none`, `automatic`, `sorted_merge`, `slack_csr` | same | same; `automatic` is the default (the line graph is built when it is read: `h2h()`); `sorted_merge` arrives in M8b step 2 (until then it raises `not_supported_error`), `slack_csr` raises `not_supported_error` naming 0.4 |
| `id_reuse` | `erase_first` (default), `insert_first` | same | same (O16, 2026-10-07) |
| `capacity_policy` | `initial_headroom`, `grow_factor`, `compact_before_grow`, `max_payload` | + the presets | PLAN's four fields kept, with the fields that make both originals one line each (section 6) |
| `hypergraph_properties` | storage, line graph, reuse, capacity | + `store_v2h` | the sketch's |
| `hyperedge_list(_view)` | named only | CSR of member lists, `int64` offsets, weights after members | the sketch's, with `add_hyperedge` giving the strong guarantee (as `edge_list::add_edge`) |
| `incidence_view<id_t>` | h2v, v2h | "as `csr_view`, dead rows empty" | `row_ptr` (`int64`) and `col_ind`, the `version` it describes; rows sorted; dead rows empty (section 2) |
| `v2h(bool sorted = false)` | a flag | always sorted, no flag | no flag: `v2h()` is sorted, built lazily, at most once per version, never on the apply path (section 4) |
| `line_graph_view<weight_t>` | named only | "+ overlap sizes, sorted rows" | `row_ptr`, `col_ind`, `out` (the first neighbour above the row's own id), `overlaps` (empty under `escher`: section 7), `weights` (the hyperedge weights by id), `version`; filled from M8b step 2 |
| `storage_stats` | named only | six counters | `device_bytes` in A's `DynamicHypergraph::deviceBytes()` scope (`dynamic_hypergraph.cu:175-180`: the trees and their row views), plus `view_bytes` (the lazily built public views, outside that scope), `host_bytes`, the H2V payload capacity and use, growths, compactions and the degree-binned kernel paths of fill, reuse and unfill (thread / warp / block) |
| `hyperedge_batch_view` | four op groups | same | same; the two incidence-edit groups raise `not_supported_error` naming 0.4 when not empty |
| `hyperedge_batch` (owning) | "builder"; PLAN 5.3's example constructs it from `res` and chains the builders | `explicit hyperedge_batch(bool weighted = !is_unweighted_v<weight_t>)`, `delete_hyperedge`, `insert_hyperedge`, incidence edits, `reserve`, `clear`, `view` | the sketch's builders, rolling back a failed append (`out_of_memory_error`, strong guarantee) as `edge_batch`; **differs** (M8b review, recorded here): a default constructor only (`noexcept`; weighted or not is the type's `weight_t`, decided at compile time, so the sketch's `bool weighted` argument has nothing to decide, and a host builder takes no `resources`), and the builders return `void` like `edge_batch`'s (no chaining) |
| `hyper_apply_summary` | ids, counts, `skipped_ops` | counts spelled out | the sketch's, and `skipped_ops` broken down into `ignored_deletions` and `dropped_members` (section 8) |
| members | `from_hyperedges`, counts, `version`, views, `apply`, `to_hyperedge_list`, `store_stats`, `check_integrity` | + `clone`, `to_backend`, `properties`, `space` | the sketch's, plus `weights()` (the id-indexed weights, section 3); the state identity is internal (`detail::next_graph_state_id()`, shared with `graph`) |

PLAN 5.3's `triad_count` example therefore reads, against the frozen API (M9's docs use this
form; `hyper_apply_summary::inserted_ids` is a host `std::vector`, section 3, so no `to_vector`):

```cpp
dyng::hyperedge_batch<std::int32_t, dyng::unweighted> b;
b.delete_hyperedge(17);
b.delete_hyperedge(42);
b.insert_hyperedge({3, 9, 12});
b.insert_hyperedge({9, 12, 40, 41});
auto st = dyng::triad_count::update(res, hg, b.view(), triads);
const auto& ids = st.batch.inserted_ids;  // best-fit reuse: likely {17, 42}
```

### 2. Dead rows are empty rows (open question 1)

A dead id (deleted, or below the bound and never assigned) has an empty row in `h2v()`; there is
no liveness bitmap. This is unambiguous because an empty hyperedge is rejected (by
`from_hyperedges()` and by `apply()`, `invalid_argument_error`), so every live row has at least
one member. `hyperedge_id_bound()` is the largest key of the H2V tree (dead slots that are still
in the tree included): ids lie in `[0, bound)`. It **can go down**: an insert with surplus rows
rebuilds the tree without the deleted slots it did not reuse, and numbers the surplus rows from
the last live key, so a tree whose highest keys were dead ends below its old bound.

### 3. inserted_ids and weights (open questions 2 and 3)

- `hyper_apply_summary::inserted_ids` is a host `std::vector<std::int64_t>` (the caller owns it;
  it outlives the container's state). The container also keeps a device copy in its internal
  applied record (CUDA), so the insert phase of `triad_count` (M9) reads its frontier on the
  device without an upload.
- Weights (`weight_t = std::int32_t`): one per hyperedge, in a host side array indexed by id,
  under `escher` storage too; validated in [1, 2^31 - 1] (PLAN 4.4.2; `hyper_sssp` checks its own
  stricter range). A dead id's weight is 0. `weights()` returns that host array; the device copy
  for `hyper_sssp` is added with it (M11).

### 4. Placement on each backend, and the views

| | sequential, openmp | cuda |
|---|---|---|
| authoritative | the host store: `incidence_model` (sorted H2V rows by id, sorted V2H rows by vertex) and the production best-fit `id_allocator` | the device store: two `cbst_store`s (H2V, V2H) of M8a; the row view (A's `RowLocator`) of H2V rebuilt once per commit |
| host state per batch | the store itself | only what validation and the bound need: one byte per id (absent, live, a dead slot still in the tree) and the live count; the members of the deleted ids are read from the device store (`cbst_store::rows`), never kept on the host |
| `h2v()`, `v2h()` | host CSR built lazily from the store (once per version) | device CSR built lazily from the store (once per version); `v2h()` sorts each row (A's `buildSortedCopy`) |
| placement | a host container refuses CUDA resources and the reverse (`invalid_argument_error` naming `to_backend(res)`); nothing is copied silently (PLAN 4.6 rule 5) | the same, and a CUDA container refuses CUDA resources of another device |

The host copies of a CUDA container are lazy (ADR 0020): `to_hyperedge_list()`,
`check_integrity()` and `clone()`/`to_backend()` to a host backend download the stores when they
are called. `apply()` reads the device error word once, at its end (`cbst_store::check_errors`
after the last store operation; the store's own read points inside an operation are A's).

**Streams** (M8b review): a CUDA container's store works on the stream of the resources it was
built with. `apply()` and `dyng::update()` take resources that may name another stream (of the
same device); the store phase then waits for the work queued so far on the call's stream (an
event) and the call's stream waits for the store's work at its end, so a reader of G_t's views
queued on either stream, and count(-) of an update, are ordered before the store frees or reuses
G_t's memory. Resources on the same stream cost nothing.

### 5. The host id allocator

`cpp/src/hypergraph/id_allocator.{hpp,cpp}` reproduces the ids of the H2V tree of the store on
the host. It is written from the store's documented contract (`cpp/src/hypergraph/escher/
README.md`) and from the originals, **not** from `testing::cbst_host_model`, so the
cross-backend check (C3) compares two implementations; the model then checks both in lockstep.
Its state per key is `{absent, live, dead}` and the slot `length`, and the H2V payload
accounting: the bump pointer, the capacity, growths and compactions under the store policy that
the `capacity_policy` maps to (section 6). It implements:

- the three padding rules: construct `((n + 4) / 4) * 4` slots (rule 1); a surplus row
  `ceil(n / alignment) * alignment + 1` (rule 2); an overflow segment of an append
  `((k + 4) / 4) * 4` (rule 3, with the tail accounting of a row: used by the incidence inserts of
  0.4 and by the allocator's own tests, not by the hyperedge batches of 0.3);
- eligibility: with `D` dead slots in the tree, only the first `min(K, D)` items may reuse one;
- best fit over the dead slots in ascending key order (the order-statistic walk), both sorts
  stable (equal sizes keep the item order, equal capacities the key order), capacity
  `length - 1`, `assigned[i] = i + max_{j <= i}(lo[j] - j)`;
- the surplus rule: when any item is not matched, the deleted slots that were not reused are
  dropped and the surplus rows get the keys after the last live key, in item order;
- the trigger points of the compacting presets (an insert whose surplus rows do not fit behind
  the bump pointer compacts after the reuse: live rows `(L + 4) / 4 * 4` slots, dead slots keep
  their length; an append whose overflow segments do not fit compacts and appends the rest once
  more), the growth rule, the hysteresis of `dyng_default()`, and `capacity_error` where the store
  throws one.

M11's `slack_csr` reuses it. **V2H needs no allocator state**: its keys are the vertices
(`1..n`), fixed for the container's life; it is never inserted into or erased from, so no id
depends on its layout. Its payload layout decides nothing a reader sees (its rows are sets); the
host store keeps its rows as sorted vectors. (A consequence, documented: a fixed-capacity preset
can make the CUDA store's V2H raise `capacity_error` where the host store has no limit; no parity
set comes near it.)

### 6. capacity_policy

```cpp
struct capacity_policy {
  double initial_headroom = 0.125;        // PLAN: a tree starts at size + floor(size * h) + initial_slack
  std::int64_t initial_slack = 1024;      // A: 1024; B: Caps::extraPayloadInts (`extra`)
  std::int64_t min_initial_payload = 0;   // A's --payload-capacity (a lower bound)
  double grow_factor = 1.5;               // PLAN: max(needed + needed / 4, capacity * 1.5); 0 = never
  bool compact_before_grow = true;        // PLAN: B's trigger points first
  std::int64_t max_payload = 0;           // PLAN: 0 = 2^31 - 1 (the int32 offsets)
  std::int32_t row_alignment = 4;         // rule 2's alignment
  std::int32_t compaction_headroom = 8;   // the hysteresis of ADR 0040 (0 = none)
  bool exact_scratch = false;             // A's scratch sizing (else B's geometric growth)
};
```

- `escher_gpu(min_initial_payload = 0)` = `{0.125, 1024, min, 1.5, false, 0, 4, 0, true}`: A's
  `payloadCapacity` (`size + size/8 + 1024`, at least the requested capacity) and
  `ensurePayloadCapacity`, never compacts; the store's `store_policy::escher_gpu()`.
- `mosp_escher(extra = 2^20)` = `{0.3, extra, 0, 0, true, 2.0e9, 4, 0, false}`: B's
  `constructFromRows` (`floor(flat * 1.3) + extra`, at most 2.0e9), a fixed capacity, compaction
  at B's trigger points, then `capacity_error`; the store's `store_policy::mosp_escher(extra)`.
- The default `{0.125, 1024, 0, 1.5, true, 0, 4, 8, false}` is `store_policy::dyng_default()`.

The initial capacity of a tree updated in place is `min(max(min_initial_payload, size +
floor(size * initial_headroom) + initial_slack), max_payload)`, and `capacity_error` if that is
below `size`. `size + floor(size * 0.3)` equals B's `floor(size * 1.3)` for every size below
2^31 (tested over the boundary values), and `size + floor(size * 0.125)` is A's `size + size / 8`
exactly. Where A ended the process (an initial capacity above 2^31 - 1) dynG clamps to
`max_payload`. In 0.3 `grow_factor` is 1.5 or 0 (another value raises `not_supported_error`);
for a policy that never grows `max_payload` bounds the initial capacity only. The conversion to
the store's policy (`detail::to_store_policy`) is tested to give the presets' growth, compaction,
scratch, alignment and hysteresis, and the container's initial sizing is tested against
`store_policy::initial_capacity()`.

### 7. Overlap sizes

The `sorted_merge` line graph stores no overlap sizes (A's H2H rows are neighbour ids only), and
`triad_count` recomputes intersections from H2V. `line_graph_view::overlaps` stays in the frozen
view as an optional array that is **empty under `escher`**, so `slack_csr` (0.4) can fill it
without a header change.

### 8. Batch semantics and the guarantees

- **erase_first** follows A exactly: validate; read the members of the deleted ids from the
  store; H2V erase; V2H remove (grouped by vertex, the deleted ids in batch order); H2V insert
  (the members sorted and deduplicated); V2H append of the **assigned** ids (grouped by vertex,
  in item order).
- **insert_first** follows B's `beginBatch` / `finishBatch` order: validate; H2V insert; H2V
  erase of the deleted ids; V2H remove; V2H append.
- A deletion is checked against the state **before** the batch (B's rules,
  `HostHypergraph.cpp:178-195`): an id that is negative, at or above the bound, dead, or deleted
  earlier in the same batch is skipped and counted in `ignored_deletions`. A member that repeats
  in an inserted hyperedge is dropped and counted in `dropped_members` (B deduplicates silently,
  `DynamicHypergraph.cpp:323-326`). `skipped_ops` counts every skipped operation; in 0.3 it is
  `ignored_deletions + dropped_members` (M8b review: the frozen field is defined as the total, not
  as that sum, so the incidence edits of 0.4 count their skips in it, as B's `skippedOps` does;
  see "Incidence edits in 0.4" below).
- **Incidence edits in 0.4** (the rule M11 implements; recorded at the freeze, M8b review):
  B's `HostHypergraph.cpp:197-235` skips and counts an incidence removal or addition on a dead,
  out-of-range or virtual hyperedge, of an out-of-range vertex, of a vertex that is not a member
  (removal), one that would empty its hyperedge, and an addition of an existing member. dynG
  follows B: each is skipped and counted in `skipped_ops` (B's `skippedOps` parity and the
  conformance check C9, applied + skipped = requested). An out-of-range vertex in an incidence
  edit is therefore **skipped**, unlike an out-of-range member of an inserted hyperedge, which
  stays an `invalid_argument_error` (an inserted hyperedge has no G_t state to be checked
  against). A per-kind counter, if M11 wants one, is an additive field.
- **Strong guarantee** for everything checked before the first store operation:
  `invalid_argument_error` for a member out of range, an empty inserted hyperedge, malformed
  offsets, a weight outside [1, 2^31 - 1] or weights on an unweighted batch;
  `not_supported_error` for non-empty incidence edits (0.4); an input copy refused by the copy
  policy; `capacity_error` for a batch of more than `INT_MAX / 3` inserted members (the int32
  payload of one store operation); an `out_of_memory_error` while the batch is staged and checked
  (M8b review: the public header had listed these two as poisoning). Then version, state
  identity, rows and `to_hyperedge_list()` are unchanged.
- **Basic guarantee** for the store phase: a `capacity_error`, an `out_of_memory_error`, a
  `cuda_error` or the device error word raised by a store operation poisons the container; every
  later call that reads or changes the storage (the views, `weights()`, `apply()`,
  `to_hyperedge_list()`, `clone()`/`to_backend()`, `check_integrity()`) throws
  `invalid_argument_error` until it is rebuilt (`from_hyperedges`); the destructor and the move
  operations work, and the `noexcept` accessors (counts, bound, version, `store_stats()`) return
  unspecified values (M8b review: `weights()` did not check, and the text said "every later
  call"). Nothing throws after the commit (version, state identity, the
  applied record and the dropped views are `noexcept` steps).
- **Ids are read from `inserted_ids`, never predicted**: best fit decides them, and the surplus
  rule can reissue an id that was dead before the batch (or one deleted in the same batch under
  `erase_first`).

### 9. The id_reuse example

Three hyperedges `{0,1,2,3}` (id 0, a slot of 8), `{4}` (id 1, a slot of 4) and `{5,6}` (id 2,
a slot of 4) under `escher_gpu()`; a batch deletes id 1 and inserts `{7,8}`. `erase_first`: the
dead slot of id 1 (capacity 3) fits the new row, so it gets id **1**. `insert_first`: the insert runs before
the erase, there is no dead slot, the row is surplus and gets id **3** (after the last live key);
then id 1 is erased. The example is a test (`HypergraphApply.IdReuseOrdersGiveDifferentIds`) and
is in the concepts page (step 5).

### 10. The internal applied record and the stages

`detail::hyper_applied` (read by M9 and M11 through `detail::hypergraph_access`): the inserted ids
on the host and, on CUDA, in a device buffer; the deleted ids with their member sets (CSR);
the touched hyperedges and vertices (sorted, unique); the skipped counts; and the incidence-delta
hook of M11 (`(vertex, id, sign)` triples, filled only when a caller asks for them). Profiler
stages: `hypergraph.build` (host side of the construction), `hypergraph.upload` (the device stores
and their views), `hypergraph.apply` with `.apply.delete`, `.apply.insert` and `.apply.views`
(the per-commit row view), named as in `parity/timed_regions/hypergraph.toml`;
`.apply.line_graph` comes with `sorted_merge`.

### 11. Container-level id parity

`parity/compare.py hypergraph_ids --set <name> --backend <b>` drives `dyng_hypergraph_ids` (a
parity tool of `cpp/tests/parity/escher`) over M8a's trace sets, reading their batch records
with M8a's reader:

- **A's sets** (`cbst_trace_escher_gpu_test`, `_coauth50k`, `_syn1m`) under `{erase_first,
  escher_gpu(capacity from the trace)}`, vertex `v` of A = dyng vertex `v - 1`, key `k` = id
  `k - 1`: `inserted_ids == insertedKeys - 1` for every batch; h2v rows by id equal A's model
  (`hyperedges()`, rebuilt from the batch records); v2h rows equal as multisets. A second run
  deletes **by content**: each recorded `deletedRows` set is resolved to the smallest live id with
  that member set not yet taken in the batch. Both runs must give the same live member-set
  multiset after every batch, and the same ids for every batch up to the first ambiguous
  resolution (a deleted set held by several live ids, where the smallest is not the one A
  deleted); the ambiguous resolutions are counted and reported.
- **B's hyperedge-kind sets** (the `make test` cases, `cbst_trace_mosp_escher_dblp50k_hyperedge`)
  under `{insert_first, mosp_escher(extra of the trace's tree)}`: B's rows are loaded as they are
  (B's virtual source and target hyperedges are ordinary hyperedges at B's positions, id = B id
  - 1; their weight 0 is loaded as 1, the only weight dynG accepts); ops naming a virtual id are
  filtered out by the harness and counted; `inserted_ids == finalIds - 1`; `ignored_deletions`
  plus the filtered ops equals `skippedOps`; `dropped_members` equals B's dropped duplicates
  (`he_insert` values minus the H2V insert's payload); h2v and v2h equal. A session (one
  `bulkLoad`) is replayed up to its first batch with incidence edits (B's vertex-kind batches
  belong to M11).
- Every set runs on `cuda`, `sequential` and `openmp` (1 and 8 threads), with `check_integrity()`
  and two `cbst_host_model`s (H2V, V2H) in lockstep after every batch.

## Consequences

- The container hides every CBST detail; `cbst_source_rules` keeps checking that no public header
  shows one. `testing/cbst_host_model.hpp` stays the documented exception.
- The CPU backends give the same ids as the CUDA store for every batch, by construction of the
  allocator and by test; a container moved between backends keeps its ids, its dead slots, their
  lengths and its payload accounting (`to_backend` writes the allocator state into a constructed
  device tree; the bump pointer is carried over with `cbst_store::set_used_payload`, a dynG
  addition).
- The frozen `line_graph_view` carries an `overlaps` array that the `escher` storage leaves empty.
- A deleted id's members are read back from the device (two host synchronizations per apply with
  deletions); that keeps the CUDA container free of a host mirror of its rows.
- The `insert_first` order reproduces B's ids only with B's virtual hyperedges loaded as ordinary
  ones; `hyper_sssp` (M11) makes them internal again.
