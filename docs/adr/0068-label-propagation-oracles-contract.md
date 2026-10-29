# ADR 0068: The label_propagation oracles: harmonic_solution, union_find and lp_host_model, their contract and their validation against LabelPropagation-CUDA

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M10a step 2 (PLAN 5.8, 6.4.7, 8.2 C2, 8.3; O27;
  the M10a acceptance criterion 7). The rules it implements are the author's: AD7 (the harmonic
  oracle: mean |F - F*| <= 2000 delta on every batch, at delta = 1e-6 in CI; 0 PCG skips; no class
  flip where |F* - 0.5| > 3m, m from O's own delta 1e-6 runs), the comparator rules of sheet C8
  (O's GROUND_TRUTH is "active with a seed label"; statuses map by name), **decided by the author
  on 2026-10-28** (GOVERNANCE.md, approvals log, the row "The parity rules of 0.4 (sheet
  C1-C8)"), and the names of AD12 (the row "The new public names of 0.4": `testing::
  harmonic_solution`, `union_find` and `lp_host_model`, the latter keeping its name).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step oracles-generator-writers)

## Context

DynLP is approximate (PLAN T2, O27): its update chain and IrLP are each compared with the converged
harmonic solution F* of the current graph, not with each other; its deterministic stages (the
container's slot rows and statuses, the affected set, the sparsified subgraph G', its components,
the supernode weights and the initial labels) are exact and are compared with O's recorded traces
(ADR 0065). PLAN 5.8 requires oracles that share no code with what they check. O has its own
host model and harmonic solver in its `--check` oracle (`src/checker.cu`); M10a's oracles extend
that model to O's exact slot layout and make the solver independent of the thread count.

## Decision

### 1. The contract (PLAN 5.8)

`cpp/include/dyng/testing/{harmonic_solution,union_find,lp_host_model}.hpp`, host-only, in the
isolated object library `dyng_testing_label_propagation` (part of `dyng::testing`; union_find is
header-only). They include only the C++ standard library and each other's headers
(`lp_host_model` uses `union_find`), use `std::thread` for harmonic_solution's optional threads,
and throw the standard exceptions (`std::invalid_argument`, `std::out_of_range`). CTest's
`label_propagation_oracles_isolation_{harmonic_solution,lp_host_model}` (the M8a/M9a check
`check_model_isolation.cmake`, given the extra allowed header) proves it: the preprocessor's
dependency list names no project file but the oracle's own headers (nothing of `cpp/src`, of the
container or of an algorithm), and the object files reference no symbol of namespace `dyng`, so
nothing but the standard library is on the link line. They are not in `dyng/dyng.hpp` (testing
headers stay out of it).

### 2. harmonic_solution

O's `DynLPChecker::harmonicSolution`, with the same system and stopping rule: the unknowns are the
present unlabelled vertices of components with a present seed (multi-source BFS from the seeds over
unknown vertices; the others are excluded and counted); `(D - W_UU) x = W_UL y_L` with D the
weight to present neighbours; Jacobi-preconditioned CG in double from x0 = 0 until
|r| <= 1e-12 |b|, at most 100,000 iterations; `converged` iff the final relative residual is <=
1e-12 (O's skip verdict otherwise); seeds hold their label, excluded vertices 0.5. One change:
every dot product and norm sums fixed chunks of 4,096 vertices in order and then the chunks in
order, so the result is bitwise the same for every thread count (O's OpenMP reductions are not;
tested at 1 and 4 threads). The input is any CSR with a presence mask and seed labels (negative
columns, i.e. unused slots, and entries to absent vertices are ignored), so the full graph, as O
passes it, and slot rows can both be given. `label_error_against()` is O's checkLabels
comparison (mean and max |F - F*| in vertex order over the compared vertices, the flips of
`F >= 0.5f` against `F* >= 0.5`); `class_agreement_with(margin)` counts the flips with
|F* - 0.5| > margin and reports the largest |F* - 0.5| of a flip.

### 3. union_find

Deterministic disjoint sets over [0, n) whose root is the smallest member (union makes the smaller
root the parent), with path halving: the convention of O's Shiloach-Vishkin (hook the larger root
to the smaller) and of O's checker.

### 4. lp_host_model

O's checker model extended to the exact slot model of O's device graph (the header documents each
rule): single-use ids, the net effect (Ins_eff = Ins \ Del in Ins order, Del_eff = Del ∩ V_t);
every id named in Del is retired (an id of Ins ∩ Del ends deleted, a never-inserted id named in Del
too); the device's order of the status changes (deletions, then the effective insertions, whose id
raises O's sticky ERROR_REINSERTED = 1 + max id when it was not inactive, then the seeds); rows as
fixed slots of the full degrees, filled per batch in half-edge-id order when both endpoints are
present after the batch, an entry past the row's capacity dropped with the sticky
ERROR_ROW_OVERFLOW; unused slots -1 / 0.0f; V_aff = N_t(Del_eff) ∪ Ins_eff ∪ N_{t+1}(Ins_eff) on
the slot rows, restricted to present unlabelled vertices; G' (w > tau strictly, both endpoints in
Ins_eff) as sorted pairs; components with O's numbering (the rank of the root, the smallest id, in
Ins_eff order); W0 / W1 as uint64 sums of `nearbyint(double(w) * 2^32)` under round-to-nearest
(O's `__double2ull_rn`) over the row entries to seeded vertices of any status; the initial labels
(seeds exactly 0 / 1, else `float(W1' / (W0' + W1'))` when the sum exceeds 1e-10, else 0.5); tau as
computeAvgEdgeWeight. A seed label on a vertex deleted in the same batch, an id or endpoint out of
range and a label array of the wrong size are refused before any change. Propagation is not
modelled.

**The comparator mapping (sheet C8):** O's statuses map by name, never by number: VERTEX_INACTIVE
(3) = inactive, VERTEX_DELETED (1) = deleted, VERTEX_GROUND_TRUTH (2) = active with a seed label,
VERTEX_ACTIVE (0) = active without one; O's gtClass = the seed label (-1, 0, 1).

**Known limits** (no case of the corpus reaches them; recorded for M10a steps 3 and 5): O's device
marks as affected both endpoints of every new edge whose endpoints are not DELETED and the
neighbours of every id named in Del (not only Del_eff); the model follows the definition above, so
for a batch with an edge between two vertices present before it, or with a deletion of an id
deleted earlier, the model's V_aff can be smaller than O's device frontier (O's own checker would
report the same difference). O's device also sets a deleted id's GROUND_TRUTH status when the id is
both seeded and deleted; the model and (M10a step 3) the container refuse that input.

### 5. Validation against O (the step's go/no-go)

- **The model** (`compare.py label_propagation model`): fed O's own recorded batches (`run_begin`,
  `batch_data` of the trace golden; at 5m, digest mode, the generator's batches after their digests
  are checked equal to O's), it equals `label_propagation_trace` at every batch of P+ in every
  field it predicts: the status and gt_class arrays, Ins_eff, row_ptr (b0), row_end, every slot's
  column and weight bits, the two error flags, tau's bits, G' and its size, the component ids and
  count, W0 / W1, the initial label bits, the initial frontier and its size: 10 cases, 1,530
  fields, 0 mismatches. The 5k5 field digests are committed (`model_5k5_d1e-4_s42.sha256`) and
  checked by the CPU suite.
- **harmonic_solution against O's checker** (`compare.py label_propagation oracle`): the recording
  twin ran O's `--check` on C (5k5, 50k at delta 1e-6, both rules; `--validate` at 5k5) and L
  (500k, 5m at delta 1e-4, both rules; OMP_NUM_THREADS=8, GPU 1), recording the labels of every
  batch and, through `--wrap=snprintf` in checker.o, the full-precision arguments of O's label
  oracle line (the recorder's `check_labels` record; with `DYNG_DYNLP_TRACE_TAGS` only these two
  records are written; ADR 0065's recorder, otherwise unchanged: check_twins passes with it).
  harmonic_solution on O's labels reproduces O's per-batch mean |F - F*| to at most 3.6e-14
  absolute (the bar: 1e-9 absolute or 1e-6 relative), with the same compared and excluded counts,
  the same class flips and the same skip verdicts (none skipped) on all 70 batches; the PCG
  iteration counts agree on 68 of 70 (two 5k5 batches differ by the reduction order, both
  converged to the residual rule).
- **The class margin m** (`compare.py label_propagation margin`): the largest |F* - 0.5| of a
  vertex whose class differs from F*'s over O's delta 1e-6 runs (damped per batch from the twin's
  label bits; in_place final labels of step 1's three runs per case): **m = 2.9217e-3** (50k
  damped, batch 3), so the AD7 rule is: no flip where |F* - 0.5| > 3m = 8.765e-3. O's own in_place
  per-batch labels of the check runs stay inside it (largest 2.864e-3).

The independent oracle therefore confirms step 1's STOP: O's damped run at 5k5 batch 1 has mean
|F - F*| = 1.5144e-3 against 2000 delta = 2e-3 (margin 1.32), computed the same by both solvers.

## Consequences

- M10a steps 3-5 compare the container, the sequential backend and the CUDA engines with
  lp_host_model (exact) and harmonic_solution (AD7), and the kit's C2 uses harmonic_solution as
  `near_reference`.
- The step-2 recorder records `check_labels` only in `--check` runs, so the golden sets (exported
  without `--check`) are unchanged.

## Alternatives considered

- **Reading O's printed mean (`%.2e`)**: three significant digits cannot show agreement to 1e-9;
  the recorder takes the values from the formatting call itself.
- **O's thread-count-dependent reductions**: a test oracle whose answer depends on the machine's
  core count cannot be bitwise reproducible; the fixed-order sums differ from O's by roundoff only
  (3.6e-14 at most on C and L).
- **Sharing a union-find or CSR helper with the library:** forbidden by PLAN 5.8.
