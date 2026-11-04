# ADR 0085: Incidence edits on slack_csr storage, and the id allocator's H2V fills and unfills

- **Status:** Accepted. The scope and the behaviour on the stable surfaces are the author's: HD9
  (incidence edits on `slack_csr` storage only in 0.4; escher storage keeps `not_supported_error`
  with the strong guarantee; triad_count refuses slack_csr storage) and HD12 (the names: the
  appended `hyperedge_batch_mix` enumerators), **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log, the rows of 2026-10-28); O16 (`erase_first` the default; H-SOSP
  parity on `insert_first`, 2026-10-07). The technical parts (Step 0's fields, the allocator's
  model of the tails, the lockstep tools, the kit's model) are accepted under delegation
  (GOVERNANCE.md, "Delegation of technical ADRs"). It carries out M11a step 2 (PLAN 6.4.8, 4.4.2,
  5.2; ADR 0041 section 8; the M11a acceptance criteria 7, 8 and 15).
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD9, HD12, 2026-10-28); the AI assistant, on the author's behalf, for
  the technical parts (M11a, step carried-host-incidence-slack)
- **Builds on:** ADR 0041 (the container: sections 5, 8 and 10; "Incidence edits in 0.4", recorded
  at the freeze), ADR 0043 (the transaction, the kit's hypergraph axis), ADR 0047 (the frozen
  0.3 surface), ADR 0086 (the slack_csr store that takes the edits).

## Context

A 0.3 hyperedge batch has four operation groups; the two incidence-edit groups (`delete_incidence`
/ `insert_incidence`: a vertex removed from or added to a live hyperedge) were declared and
refused (`not_supported_error`, ADR 0041). MOSP_ESCHER@4b86159 (B) applies them: its vertex-kind
paper batches are incidence edits only, and its host model (`HostHypergraph::applyBatch`,
`hypergraph/src/HostHypergraph.cpp:122-295`) runs every batch in a fixed phase order with skip
rules, while its ESCHER H2V tree (the ids) receives the removals and additions as unfills and
fills (`DynamicHypergraph::finishBatch`, `DynamicHypergraph.cpp:282-525`). M8b replayed B's
sessions only up to their first vertex-kind batch, because the host id allocator modelled neither
(M8b deviation 7): an unfill changes a row's tail occupancy, a fill appends to the tail or opens an
overflow segment, and both decide where later compactions put the rows, hence the slot lengths that
best fit reads, hence the ids.

## Decision

### 1. Scope (HD9)

Incidence edits are applied on `hypergraph_storage::slack_csr` (ADR 0086), on every host backend
in this step (the cuda backend refuses slack_csr storage until M11a step 3, `not_supported_error`
naming "M11a step 3"; a test flips with it). Under escher storage a batch with an edit raises
`not_supported_error` before anything changes, the message naming `hypergraph_storage::slack_csr`
(escher rows stay sorted for ESCHER-GPU's merge, ADR 0042). triad_count and the experimental
count_local_patterns refuse slack_csr storage in `compute()` and before the commit of an update
(`not_supported_error`, strong guarantee), so hyper_sssp and triad_count do not share a
hypergraph in 0.4; their kit traits accept the escher presets only (`test_traits::accepts`).

### 2. Step 0 (the strong guarantee) and B's semantics

The transaction's begin stages the four incidence arrays with the batch's other arrays (the copy
policy of the resources) and, after the deletions, evaluates every edit in batch order against
G_t, removals first, then additions (B's phases 2 and 3, whatever the order of the arrays), on
working copies of the rows of the hyperedges they name. An edit is **skipped and counted** in
`skipped_ops` when (B's rules, each its own unit test):

| Rule | B (`HostHypergraph.cpp`) |
|---|---|
| its hyperedge is negative, at or above the bound, dead, or deleted by the batch (phase 1 ran first); an id the batch inserts is one of these | `c.heId < 1 \|\| c.heId > maxId() \|\| !alive[c.heId - 1]` (`isVirtual` is the harness's filter, below) |
| its vertex is out of range | `c.vertex < 0 \|\| c.vertex >= numVertices` |
| a removal of a non-member (e.g. a second removal of the same vertex) | `it == verts.end() \|\| *it != c.vertex` |
| a removal that would empty its hyperedge (the second of two removals on a 2-vertex hyperedge) | `verts.size() == 1` |
| an addition of a member (e.g. a repeat) | `*it == c.vertex` |

An out-of-range vertex of an edit is skipped, while an out-of-range member of an inserted
hyperedge stays an `invalid_argument_error` (it has no G_t state to be checked against; a test
shows both). Arrays of unequal length are an `invalid_argument_error`. Step 0 records the
applied removals and additions in batch order, the edited ids and their rows before and after the
edits (new private fields of `hyper_staged_batch`); nothing is changed (strong guarantee). The
summary fills `inserted_incidences` / `deleted_incidences`, and `skipped_ops` = ignored deletions
+ dropped members + skipped edits, so C9 holds: applied + skipped = requested.

### 3. The order of the store's phases (id_reuse, documented)

- **insert_first** is B's `beginBatch` / `finishBatch`: H2V insert (the ids); then H2V erase of the
  deleted ids, H2V unfill (the removals, grouped by hyperedge in batch order), V2H unfill (the
  deleted rows' members, then the removed incidences), H2V fill (the additions, grouped by
  hyperedge in batch order), V2H fill (the added incidences, then the inserted rows).
- **erase_first** is erase, unfill, fill, then insert (and the V2H fill of the inserted rows).

`id_reuse`'s documentation says so (a documentation-only change of the frozen header). Without
edits both orders are 0.3's.

### 4. The id allocator models fills and unfills

`id_allocator::append(keys, prefix, values)` (the count-only form is replaced: the values decide
which of them land in an overflow segment) and `id_allocator::remove_values(keys, prefix, values)`:

- a fill writes into the tail segment and opens an overflow segment of `((k + 4) / 4) * 4` slots at
  the bump pointer for the rest (padding rule 3), compacting first under the compacting presets
  (MOSP_ESCHER's fillImpl), as before;
- an unfill drops the row's count by its removals and leaves in the tail occupancy **the values left
  in the tail segment** (`unfillRowSerial`, `escher/kernel/unfill.cu:18-23`; the warp and block
  kernels give the same on a single-segment row). The allocator therefore keeps the values of the
  tail segment of every row whose tail is an overflow segment (only those: a single-segment row's
  tail is the whole row), drops them on a compaction, an erase or a reuse, and carries them in
  `allocator_state` across backends.

The mutation **#4** of the hyper_sssp suite (PLAN 8.4; `DYNG_MUTATION_HSOSP_ALLOCATOR_UNFILL_DROPS_LAST`,
the unfill model drops the last value of every item) is planted here; the hypergraph suite catches
it (`hypergraph.mutation.hsosp_allocator_unfill_drops_last`).

### 5. The lockstep (before the replays, the M8a lesson)

- `IdAllocator.MatchesTheHostModelWithFillsAndUnfills`: random inserts, erases, fills of new values
  and unfills of members (rows stay sets) under every preset, the allocator against
  `testing::cbst_host_model` after every operation: per key the state, slot length, count, tail
  occupancy, tail capacity and whether the tail is an overflow segment; the bump pointer, the
  capacity, the compactions; the unfills of single- and multi-segment rows and of 32 or more values
  (the warp kernel) are all reached.
- `dyng_allocator_lockstep` (`cpp/tests/parity/escher/allocator_lockstep.{hpp,cpp}`, driven by
  `parity/hypergraph_ids.py allocator`): every operation of MOSP_ESCHER's recorded trees (the H2V
  tree of every DynamicHypergraph in `cbst_trace_mosp_escher_test`, `_dblp50k_hyperedge`,
  `_dblp50k_vertex`; every tree of `cbst_trace_mosp_escher_targeted`, which drives CBSTOperations
  directly) on the allocator and the host model, compared after every operation with the state
  digest B recorded and with the model, InsertMappings and exceptions included.

### 6. The applied record (what hyper_sssp reads)

`hyper_applied` gains, under slack_csr storage (`has_slack_record`; empty under escher): the
`(vertex, id, sign)` of every applied change (the deleted and inserted hyperedges' members and the
edits), the edited ids, the touched hyperedges' rows before and after the batch, the touched
vertices' lengths after it, the pair delta (ADR 0086), the new ids (at or above G_t's bound), the
recreated ids (below it) and the dead ids (deleted and not reused), and the edit counts. The
frozen `hyper_update_participant` is unchanged: a participant reads the record it already
receives (`after_apply`), and the staged batch (`use_normalized`) carries the edits.

### 7. The kit (ADRs 0043, 0044) and the generators (HD12)

`generators::hyperedge_batch_mix` gains seven **appended** enumerators (their names are in the
HD12 list for the author's acknowledgement as "the appended hyperedge_batch_mix enumerators"):
`incidence_removals`, `incidence_additions`, `incidence_mixed`, `incidence_emptying`,
`incidence_same_batch`, `incidence_repeats`, `incidence_out_of_range` (documented streams, drawn
after every earlier mix, so the 0.3 streams are unchanged: their golden hashes pass unchanged, and
the new mixes are pinned from 0.4). The kit's hypergraph axis gains the same mixes (they apply to
slack_csr presets only) and the slack_csr presets {insert_first, erase_first} x {mosp_escher(),
the default} (in the host test executables; the CUDA ones get them with M11a step 3); its model
evaluates the skip rules as the container does, and C9 checks the edit counts; C7 asserts the
refusal under the escher presets. The kit's hypergraph fakes treat an edited hyperedge as leaving
with its old row and coming back with its new one.

## Consequences

- B's sessions are replayed whole (`stopped_sessions` = 0), the ids of the vertex-kind batches
  compared (ADR 0086 has the replays and their numbers).
- The 0.3 escher-storage behaviour, its id replays and the 0.3 kit presets are unchanged.
- A per-kind skip counter remains possible as an additive field (ADR 0041 section 8); none is
  added.
