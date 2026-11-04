# ADR 0086: The host slack_csr store, its slack CSR line graph, and incidence_delta inside the commit

- **Status:** Accepted. The surface and its behaviour are the author's: HD9 (the storage
  `hypergraph_storage::slack_csr` and line graph `line_graph_policy::slack_csr` of 0.4 with their
  property matrix; no public capacity field for the CSR's headroom: internal knobs; the h2h view
  sorted on request; `overlaps` empty under slack_csr), **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log, the rows of 2026-10-28). The technical parts (the store's layout,
  the knobs' defaults, where the pair delta is computed, the views, the checks, the replays) are
  accepted under delegation (GOVERNANCE.md, "Delegation of technical ADRs"). It carries out M11a
  step 2 (PLAN 4.5.3, 6.4.8; the M11a acceptance criteria 6, 9, 10, 11 (host parts), 15 and 16
  (host sites)); the CUDA store is M11a step 3's.
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD9, 2026-10-28); the AI assistant, on the author's behalf, for the
  technical parts (M11a, step carried-host-incidence-slack)
- **Builds on:** ADR 0041 (the container), ADR 0042 (escher storage and the sorted merge), ADR
  0080 (the by_id parity mode, its knobs), ADR 0081 (the H-SOSP dumps), ADR 0082 (the oracle),
  ADR 0085 (the incidence edits and the allocator).

## Context

hyper_sssp (MOSP_ESCHER@4b86159's H-SOSP, B) runs on a line graph that is not ESCHER's sorted
merge: B keeps a slack CSR (`DeviceH2H`, `hsosp/include/hsosp.cuh`, `hsosp/src/hsospDevice.cu`):
per hyperedge a row of neighbour ids in a 32-aligned slot with headroom, deletions swap-removed,
insertions appended, a row that outgrows its slot moved to the tail, a full tail rebuilt from the
incidence lists. Its rows are unsorted after a batch, its node weights are scattered by id, and it
maintains its pair count. The batch's pair delta comes from the incidence changes alone
(`deriveDelta`, `hsosp/src/hsospDelta.cu`), exact and order-free. The 0.3 container had one
storage (escher) whose line graph is ESCHER-GPU's sorted merge (ADR 0042), and refused incidence
edits.

## Decision

### 1. The property matrix (HD9)

`validate_properties` (`cpp/src/hypergraph/capacity.cpp`), each cell a test
(`SlackBackends.ThePropertyMatrix`):

| storage \ line graph | none | automatic | sorted_merge | slack_csr |
|---|---|---|---|---|
| escher | ok | ok (the sorted merge) | ok | `not_supported_error`, naming `slack_csr` storage |
| slack_csr | ok | ok (the slack CSR, built on first read) | `not_supported_error`, naming `slack_csr` | ok (built at construction) |

slack_csr storage needs `store_v2h` (`invalid_argument_error` without it: the pair delta reads the
hyperedges of every changed vertex). The cuda backend refuses slack_csr storage until M11a step 3
(`not_supported_error` naming "M11a step 3"; `SlackBackends.SlackCsrOnCudaArrivesInStep3` flips
with it). The Python layer refuses it until M11b (a `NotSupportedError` from
`HypergraphProperties`; recorded as a deviation, the retrospective).

### 2. The host store (`cpp/src/hypergraph/slack_csr/host_store.{hpp,cpp}`)

`slack_host_store` is an `incidence_store` (the sequential and openmp backends) holding B's host
incidence model (`HostHypergraph`): the rows by id (sorted members, a dead id empty), the
hyperedges of every vertex (sorted ids), the id allocator (ESCHER's H2V tree with fills and
unfills, ADR 0085), the weights by id the line graph keeps, and the slack CSR line graph. The
incidence edits are new store operations (`h2v_remove`, `h2v_append`; the escher stores do not
support them, `supports_incidence_edits()`), applied in the order of ADR 0085 section 3. The
views of the public surface (`h2v()`, `v2h()`, `h2h()`, the escher-layout row views of
`hypergraph_view`) are built lazily, once per state, under `container_scope` (the budgets, I9);
`h2h()` sorts each row on request (HD9; the slack CSR itself stays unsorted, the build counted in
`view_builds`); `overlaps` is empty. `check_integrity()` checks the rows, the incidence lists, the
allocator and the CSR (every row a set, its degree within its slot, the slots disjoint, the
entries the exact line graph of the rows, the pair count, the node weights the weights by id).

### 3. The slack CSR (`slack_line_graph`, `slack_layout.hpp`)

B's DeviceH2H rules, one source on `operators::sequential_exec` and `openmp_exec`:

- build (`buildCsr`): row r has the sorted neighbours of id r in a slot of
  `row_capacity_for(deg) = align_up_32(deg + max(4, deg / 4))`; the entry capacity
  `max(cursor, cursor x entry_headroom) + 4096`; a dead id has degree 0 and a slot of
  `row_capacity_for(0)`; an entry count past 2^31 - 1 is a `capacity_error` (the device's int32
  offsets);
- apply (`applyDeltaPairs`, `applyRowsWarp`): per row, the deletions (swap-remove of the first
  occurrence), then the insertions appended; a row that outgrows its slot moves to the tail with a
  slot of `row_capacity_for(new degree)` (the relocations planned sequentially in row order, B's
  order); the entry count changes by 2 x (inserted - deleted), the pair count by inserted -
  deleted (`h2hPairCount`); the node weights of the new ids are scattered (a dead id keeps its
  last weight, as on the device);
- a tail that cannot take the relocations rebuilds the whole CSR from the incidence lists with the
  same headroom (counted: `csr_rebuilds`);
- the node-indexed arrays have `node_capacity` nodes and grow by half (at least to the bound)
  when an id passes it (counted: `node_growths`).

The incidence mirror's slot rules of B's DeviceIncidence (`he_slot_for`, `v_slot_for`, the
re-upload growth 1.5) are in `slack_layout.hpp` for the CUDA store of M11a step 3; the host store
has no mirror (its rows are the host model).

### 4. The knobs (internal; HD9: no public capacity field)

`slack::knobs` (`entry_headroom` = 1.6, B's hsospBench value; `node_capacity` = 0: the bound plus
max(1024, bound / 8); `max_candidates` = INT_MAX, B's limit) are set only through
`detail::hypergraph_access::build(res, list, props, knobs)` (the parity harness gives B's
headroom and `maxNodes`) and kept by clones and snapshots. An entry headroom below 1 or NaN is an
`invalid_argument_error` (B's buildCsr).

### 5. incidence_delta inside the commit (a deviation from the sketch)

PLAN 4.5.3 sketches the pair delta as hyper_sssp's `translate` stage. The framework has no
translate hook a participant could run before the container's commit point, and the container's
line graph must be exact after every commit whichever algorithm is attached (or none). The delta
is therefore computed by the container inside its commit, in the stage
`hypergraph.apply.translate`, between `hypergraph.apply.incidence` and `hypergraph.apply.line_graph`
(the stages are a test), and handed to hyper_sssp in the applied record (`pairs`; ADR 0085
section 6). The rule is B's (`incidence_delta.hpp`): the net sign per incidence; for every changed
incidence (v, h), every o in v's hyperedges before and after the batch gives the candidate
(min, max) (o == h counts as a candidate and gives none); the candidates sorted and deduplicated;
a pair inserted if the two rows overlap only after, deleted if only before. Its output depends only
on the two states (`SlackEdits.PermutingAPhasesOperationsGivesTheSamePairs`; the replays on 1 and
8 threads). More than `max_candidates` candidates is a `capacity_error` raised in the store phase:
the hypergraph is poisoned (the basic guarantee of every store-phase failure, ADR 0041 section 8;
`SlackEdits.MoreCandidatesThanTheLimitIsACapacityError`); B throws at the same count.

### 6. The mutations (PLAN 8.4, host sites)

`DYNG_MUTATION_HSOSP_CSR_SKIPS_LAST_DELETION` (#3, the host twin: the CSR apply skips the last
deletion of every row), `DYNG_MUTATION_HSOSP_PAIR_COUNT_ADDS_DELETIONS` (#7),
`DYNG_MUTATION_HSOSP_DELTA_NO_POST_PASS` (#8: the delta skips the candidates after the batch); each
fails the host hypergraph suite (CTests `hypergraph.mutation.hsosp_*`, with #4 of ADR 0085).

### 7. The replays (GO / NO-GO 1, host part)

`dyng_hsosp_replay` (`cpp/tests/parity/hsosp/hsosp_replay.{hpp,cpp}`, driven by
`parity/hsosp_goldens.py delta`) replays every session of the H-SOSP dumps of
`hyper_sssp_mosp_escher` (T and P50) whole on slack_csr storage, by id (ADR 0080's by_id mode:
virtual rows at B's positions, weight 1, id = B id - 1; insert_first; mosp_escher(extra) with B's
headroom; B's CSR knobs): after every batch the ids, the skip counts, the dead ids, the rows and
weights, the pairs and the pair count against B's, `check_integrity()`, and the oracle (ADR 0082)
on the line graph and the pairs. The CBST traces of MOSP_ESCHER are replayed whole on slack_csr
storage by `parity/hypergraph_ids.py` (`--storage slack_csr`). The numbers are in
`parity/results/M11a.md` (step 2).

## Consequences

- triad_count stays on escher storage (ADR 0085 section 1); the sorted merge and its readings are
  unchanged.
- B's candidate count is recorded beside dynG's on the host stores; equality is asserted on the
  CUDA store (M11a step 3), whose CSR is B's on the device.
- The CUDA store reuses `slack_layout.hpp` and `incidence_delta`'s rule; the host replays are its
  reference.
