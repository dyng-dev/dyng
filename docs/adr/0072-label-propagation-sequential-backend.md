# ADR 0072: label_propagation's sequential backend, the hooks of the problem, and IrLP's I4 exemption

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 3 (PLAN 4.5, 4.6, D.6; the M10a acceptance criterion 11).
  The rules it meets are the author's: AD7 (the harmonic oracle), AD19 (no OpenMP backend in 0.4)
  and the comparator rules of sheet C8, **decided by the author on 2026-10-28** (GOVERNANCE.md,
  approvals log, the rows "The parity rules of 0.4 (sheet C1-C8)" and "Licensing, data
  provenance, budgets and scope of 0.4 (sheet G1-G3, H1-H3)").
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  host-container-sequential-driver)

## Context

PLAN 4.6 requires a sequential backend for every algorithm; LabelPropagation-CUDA@a276a3a (O) has
none (its `--check` host model reproduces the graph, not the propagation). The backend must be
exact against O's trace where O is deterministic (the rows, V_aff, G', the components, the
supernode weights, the initial labels), meet the harmonic oracle where O is approximate, and be
reproducible.

## Decision

### 1. The hooks (both tiers; `cpp/src/algorithms/label_propagation/problem.hpp`)

- **before_apply** (G_t): the neighbours of Del ∩ V_t on the rows of G_t are affected (O's
  `markDeletionsKernel`, which walks every id of Del; an id deleted before has no live
  neighbours to mark, so Del ∩ V_t is the definition, as `lp_host_model`).
- **identify_affected** (G_{t+1}): the batch's seed labels join the result; the inserted vertices
  Ins \ Del and both endpoints of every new edge that was appended are affected; then, when
  vertices were inserted, `identify_affected.sparsify` (G': the appended edges of weight > tau
  whose two endpoints were inserted in this batch) and `identify_affected.components` (union-find
  over Ins \ Del, a root its smallest vertex, a component's id the rank of its root in insertion
  order: O's numbering, `connectedComponents.cu`). **A fix of O:** O marks the endpoints of the new
  edges inside its `if (numInserted > 0)` (dynLP.cu:561), so a batch of edges between present
  vertices would leave them unmarked; O's generator never makes such a batch (every new edge
  brings an arriving vertex), so the fix changes no corpus case. It is recorded as a documented
  difference; edges to a vertex that is not active after the batch are not appended (ADR 0069)
  and mark nothing (O marks the endpoints of an edge whose endpoints are not DELETED, which
  differs only for a never-inserted endpoint, which O's generator never produces).
- **seed:** the supernode initialization: per component, the sums over its vertices' entries to
  seeds of class 0 and 1 of `nearbyint(double(w) * 2^32)` as uint64 (O's `__double2ull_rn`), then
  a seed's class, or `float(W1' / (W0' + W1'))` with `Wk' = double(Wk) / 2^32` when the sum
  exceeds 1e-10, else 0.5: bit-identical to O (no floating-point reduction order is involved).
- **loop:** the propagation (section 2); **finalize:** every seed's label is its class (O's
  `enforceGroundTruthKernel`, deleted seeds included), then the stats.
- **Tags** are run-wide, as O's `ws.nextTag`: the result holds the next tag (1 after `compute()`)
  and the per-vertex stamps; every update consumes one tag for its first queue (b0 included) and
  one per iteration, and the next update starts at the last tag + 1.

### 2. The sequential propagation (`sequential.cpp`)

- The first queue: the affected vertices that are active and have no seed label (O's
  `initFrontierKernel`), in vertex order, stamped with the update's first tag.
- Every iteration splits the queue into O's two hashed half-sweeps (`halfSweep(v, tag)`, the
  same 32-bit hash) and updates each half in **ascending vertex order**. `in_place` writes each
  new label at once, so a vertex reads the labels written earlier in the sweep (Gauss-Seidel; the
  second half reads the first's, as on the device); `damped` computes `F' = 0.5f (F + average)` of
  every queued vertex from the iteration's labels into a side array and commits it after both
  halves (O's `UPDATE_DAMPED`).
- The arithmetic of `accumulateNeighbors` and `updateLabel` in single precision: the entries to
  vertices that are not deleted, the 1e-10 guard (`w_all > 1e-10f`, else the old label), the
  damping, the clamp to [0, 1], and a change `|Δ| > δ` queues the vertex's active unlabelled
  neighbours and the vertex itself for the next iteration, each once per iteration (the stamps).
  The host sources are compiled with `-ffp-contract=off`, so the bits are the same on every host
  build (no FMA contraction).
- Iterations and frontier visits are counted as O counts them (one iteration per queue; the visits
  the summed queue sizes); the loop stops when the queue is empty or at `max_iterations`.
- Both schedules are bitwise reproducible on this backend (the order is fixed); its labels differ
  from O's CUDA labels in the last bits (another summation order), and are judged by the harmonic
  oracle (AD7), not compared with O.

### 3. compute() = IrLP on the host, exempt from I4

`compute()` is O's `runIrLPSnapshot`: the snapshot of the current graph (the entries between
active vertices, every row sorted by column, stably: the rows of O's full graph filtered to the
present vertices), every label 0.5 and the active seeds their class, then double-buffered Jacobi
over every active vertex without a seed label (the same arithmetic as O's `irLPKernel`), the seeds
restored after every iteration, until no label changes by more than δ or `max_iterations`.
**Exempt from I4** (no oscillating schedules): IrLP is the paper's published baseline, which PLAN
6.4.7 keeps "deliberately unchanged"; its Jacobi does not oscillate on the generator's graphs
(every component's iteration contracts towards the seeds' harmonic values), and `update()` never
runs it except as `on_limit::fallback_recompute`.

### 4. No allocation after reserve

The per-update scratch is a pooled workspace of the resources handle (ADR 0015): per-vertex arrays
sized for the graph, per-batch arrays (the supernode weights, G') sized by `result::reserve()` for
the largest batch, growing with headroom (half again) when a batch is larger (a reservation, I9;
M9a finding 17). The result's arrays (labels, seeds, stamps) are sized at `compute()`. The
algorithm phase then allocates nothing and does not synchronize (budget `steady_state(0)`): a test
counts the host heap allocations of every update of a stream after `reserve()` (the kit's counting
`operator new`). The commit (the container) is budgeted separately.

### 5. OpenMP

None in 0.4 (AD19): OpenMP resources raise `not_supported_error` naming `sequential` and `cuda`
(ADR 0070's backend sentence).

## Consequences

- `compare.py label_propagation trace --backend sequential` (ADR 0073): every deterministic field of
  every batch of P+ equals O's trace golden.
- The harmonic oracle on C: `compare.py label_propagation oracle-port` and the CPU tests.

## Alternatives considered

- **A host port of O's queue order:** O's order inside a half-sweep is the order of its atomic
  appends, which is not reproducible; ascending order is the reproducible choice.
- **Double precision on the host:** rejected; the same single-precision arithmetic as O keeps the
  iteration counts close (they match O's damped counts on the corpus).
