# ADR 0079: Step 0 of vertex batches in fewer passes (the 5m1 gate miss of M10a step 5)

- **Status:** Accepted under delegation (2026-10-30; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 5 (PLAN 8.6; the M10a acceptance criteria 9 and 18).
  The gates are the author's (AD2, AD3, **decided by the author on 2026-10-28**; GOVERNANCE.md,
  approvals log, the row "The measurement protocol and gates of 0.4 (sheet B1-B6)"); nothing here
  changes a bar, a round count, the protocol or a check of Step 0.
- **Date:** 2026-10-30
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  kit-mutations-gates-certificate)

## Context

The first gate readings of M10a step 5 on the step-4 library (`0c815a1`) passed at 50k, 500k and
5m, and missed at 5m1: damped `update[sum]` **1.080** (11 rounds at the boost lock, the port
slower in every round: O 1,387 ms, the port 1,498 ms; `end_to_end` 1.032). 5m1 has two batches:
b0 (the 50,000 seeds) and b1, one batch of 4,950,000 arriving vertices (445,500 of them deleted
in the same batch) and 12,498,775 edges. The whole difference is in b1 (+110 ms; b0 +1.2 ms).

**The diagnosis** (GPU 1, outside the gate protocol; the records under
`$DYNG_SCRATCH/runs/m10a-kit-mutations-gates-certificate/diag5m1`):

- nsys timelines of both sides on 5m1: the device work of b1 is the same (O's stages
  `b4.sparsify` .. `b7.propagation` 1,306 ms, the port's `label_propagation.enact_fused`
  1,317 ms; the kernels are the original's SASS, ADR 0075); before b1's first CUDA call the
  original spends 53 ms on the host (`b1.host.batchIds`: its range scans and one pass over the
  batch that builds Ins \ Del into pinned memory and sets the host states), the port **137 ms**
  (no CUDA call in between), then 8 ms in `cudaHostAlloc` (the pinned staging of Ins \ Del,
  first used by this batch).
- The port's host part of b1, timed by section (a temporary instrumented build): the range
  checks 28 ms, the duplicate marks 36 ms, the net effect 57 ms, the staging 8.5 ms and its
  copies 2.6 ms, the host states 31 ms, the record 2.8 ms: 166 ms against the original's 54 ms.
  The graph's Step 0 (ADR 0069) makes five random passes over per-vertex arrays where the
  original makes two: the marks of the insertions, the net effect (the marks read again), the
  clearing of the marks (the scope guard) and the states, besides branchy range checks of every
  id and endpoint.
- The 5m case (ten batches of 500,000 vertices) has the same host work per vertex but reads 1.002:
  its device work per batch is slightly faster than the original's, which hides it; 5m1 puts all
  of it into one update of 1.4 s.

The cause is the cost of the port's host Step 0, not the single stream (AD21: the lost overlap is
0.03 % of the update, ADR 0075) and not the kernels.

## Decision

Step 0 of a vertex batch (`normalize_vertex_batch` and `apply_vertex_states_host` in
`cpp/src/graph/apply_slotted_host.cpp`, the only callers of these paths are slotted graphs) does
the same checks in fewer passes:

1. **Range checks** of the deletions, insertions, labels and edge endpoints are one branch-free
   pass each (a maximum, which vectorizes); only a failing check runs the former loop, which
   throws `invalid_argument_error` naming the first offender, in the former order.
2. **One pass over the insertions** marks them, checks them (listed twice, inserted or deleted
   before, a seed deleted in its batch) and collects Ins_eff = Ins \ Del with its labels in list
   order; if a check failed, the former checking loop runs again on fresh marks and throws at
   the first offender with the former message. The D-e mutation hook (ADR 0077) moves with the
   net effect into this pass.
3. **Tagged marks.** `slotted_rows::marks` holds, per vertex, the two flag bits and in the six
   high bits the tag of the Step 0 that set them, and one more byte with the last tag; each Step 0
   takes the next tag, so earlier flags (also those of a Step 0 that threw) read as clear without
   a pass over the batch, and every 63 tags the bytes are cleared once. `check_integrity` checks
   the array's size and the tag instead of all-zero marks.
4. The random passes (the marks, the states, `apply_vertex_states_host`) **prefetch** the bytes
   they read 16 ids ahead.

Nothing else changes: the checks, their order and messages, the strong guarantee of the host
path, the basic guarantee of the device path (ADR 0074), the record and every result. The graph
module's other code, sssp, mosp, cycle_count, triangle_delta, dynamic_bfs and triad_count do not
compile these functions.

`dyng-compat-dynlp`'s evaluation (after the timer; `end_to_end` only) reads the accuracy from the
pinned copy of the labels, as the original reads `h_labels`, with the original's OpenMP reduction
above 10^6 vertices, and fills the run's host vector only where it is read (the label dump,
`--validate`, `--check`, the label file).

**Measured** (the same instrumented runs, GPU 1): the port's host part of 5m1's b1 from 166 ms to
78 ms (ranges 12.8, marks and insertions 40.6, the rest of the net effect 1.4, the staging 8.4 and
its copies 2.4, the states 12.6); b1 from 1,503 ms to 1,402 ms (the original's 1,382 ms in its own
run on GPU 1); 5m's ten batches 5,495 ms (unchanged device work). The gate readings on the final
library are in `parity/results/M10a.md` section 12 and ADR 0078.

## Consequences

- Every reading of M10a step 5 is taken again on the final library (the gates, the replays, the
  in_place cloud, the oracles, the mutations, the sanitizers).
- Not done: the host threads of the resources are not used for Step 0 (the functions do not
  receive the resources, and giving them would change `graph.cpp`'s apply, which every algorithm
  compiles); the pinned staging of Ins \ Del is still allocated by the first batch that needs it.
  Either is open if a later reading needs it.
