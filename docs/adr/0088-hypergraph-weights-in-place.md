# ADR 0088: The weights by id are updated in place (no copy of every weight per batch)

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only: removing the copy is one of the delegated technical items of M11
  (the reconciled decision sheet of 2026-10-27, "Removing the next_weights copy"). No public
  header changes, no documented behaviour changes: `hypergraph::weights()` and the guarantees of
  `apply()` are as frozen in 0.3 (ADR 0047).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M11a, step
  carried-host-incidence-slack)
- **Builds on:** ADR 0041 (section 3: the id-indexed weights; section 8: the guarantees), ADR 0024
  (dynG-against-dynG readings).

## Context

A weighted hypergraph keeps one weight per id in a host array (`hypergraph_impl::weights`, ADR
0041 section 3), on every storage and backend. The commit of a batch (`hyper_transaction::commit`,
`hypergraph.cpp`) built the next array in a local `next_weights`: a copy of every weight
(O(id bound): 2.47 million on coauth-DBLP, per batch), the deleted ids zeroed, a resize to the new
bound, the inserted ids' weights, then a swap after the commit point. The copy made the update of
the weights trivially exception-safe, but the strong guarantee does not need it: every error
before the store phase (Step 0) changes nothing because nothing is written then, and an error
inside the store phase poisons the hypergraph (the basic guarantee), whose weights no call reads
afterwards (`weights()` refuses a poisoned hypergraph, ADR 0041 section 8). For `slack_csr`
storage and `hyper_sssp` (M11), whose batches are compared against MOSP_ESCHER's maintain time,
the copy was a fixed host cost per batch proportional to the whole hypergraph.

## Decision

1. Inside the store phase (after the store operations), the array only **reserves** room for the
   new id bound (an allocation failure there poisons, like any failure of the store phase).
2. After the commit point (no-throw), the array is written **in place**, O(batch): the deleted
   ids weigh 0, the array is resized to the new bound (a smaller bound drops the dead tail; a
   larger one fits the reserved room, so nothing is allocated), and the inserted ids get their
   weights (in this order: under `erase_first` an inserted id may be one the batch deleted).
3. The same code serves every storage (`escher` and `slack_csr`) and every backend: the weights
   are a host array on all of them.

## Tests

- `HypergraphBackends.ValidationErrorsLeaveTheHypergraphUnchanged` (every backend) now also
  compares `weights()` by id before and after each rejected batch (the strong guarantee).
- `HypergraphBackends.WeightsFollowEveryBatchInPlace` (new, every backend): 60 random batches
  under both `id_reuse` orders and the three capacity presets against a model of the weights by
  id, with batches that make the bound go down (a surplus insert after the highest ids were
  deleted) and inserted ids that reuse a deleted one; when the bound does not grow, the array's
  storage is the same object before and after the batch (no copy).
- The allocation-failure sweeps of `dyng_hypergraph_allocation_failure_tests` (ADR 0061) still
  find every outcome documented: unchanged, or poisoned.

## The reading

The change is on the escher path of the stable container (`hypergraph.cpp`), so its reading is
part of the ADR 0024 reading of the escher path in M11a step 5 (`hypergraph.apply` and its
sub-stages on escher storage, sequential and openmp-8, and cuda; the bar 1.02 of HD16, decided
by the author on 2026-10-28; a region above it is located and listed for the author, the D14
path). The removed work is a copy of the whole weight array per batch, so the expected direction
of the reading is down on weighted hypergraphs and unchanged on unweighted ones (the coauth-DBLP
and syn1M cases of `hypergraph_host_ab.py` are unweighted: no change is expected there).

## Consequences

- A batch of a weighted hypergraph costs O(batch) for its weights instead of O(id bound).
- The commit point's no-throw part now writes the weight array (into reserved room) instead of
  swapping it.
