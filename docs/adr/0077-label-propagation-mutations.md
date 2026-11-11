# ADR 0077: label_propagation's recorded mutations

- **Status:** Accepted under delegation (2026-10-30; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 5 (PLAN 8.4; the M10a acceptance criterion 16). The set
  of mutations is the author's, **decided by the author on 2026-10-28** (AD9, sheet C5; the row
  "The parity rules of 0.4 (sheet C1-C8)" of GOVERNANCE.md's approvals log): plain Jacobi in an
  in_place and a damped variant, plus LabelPropagation-CUDA@a276a3a's D-a, D-b, D-c and D-e (its
  CHANGES.md, section 2). The GoogleTest scope of the record follows ADR 0061 section 4.
- **Date:** 2026-10-30
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  kit-mutations-gates-certificate)

## Decision

### 1. The hooks

Each mutation is a hook in the sources, compiled in by its define (`#if
defined(DYNG_MUTATION_LP_<NAME>)`), on every backend where the code it changes runs. Without the
define the code is the code of step 4: the fused engine's SASS is checked again against the
original's objects on the final build (`parity/label_propagation_sass.py port`).

| Mutation (`parity/mutate.py`) | Define | What it puts back | Where |
|---|---|---|---|
| `inplace_plain_jacobi` | `DYNG_MUTATION_LP_INPLACE_PLAIN_JACOBI` | in_place as one sweep reading the previous labels: the side arrays and the commit of damped, without the damping (PLAN 8.4) | `sequential.cpp`; `propagation.cu` (mode 2 of the kernels' `damped`), `cuda.cu` (the side arrays sized) |
| `damped_no_damping` | `DYNG_MUTATION_LP_DAMPED_NO_DAMPING` | damped with F' = the average (O's D-d) | `sequential.cpp`; `propagation.cu` (`updateLabel`) |
| `lp_d_a_unfiltered_components` | `DYNG_MUTATION_LP_D_A_UNFILTERED_COMPONENTS` | G' and its components over the new edges between inserted vertices without the tau filter (O's D-a, `5dddc4e`; the original also passed the new-to-existing edges, whose `par = -1` made the device read out of bounds; the hook keeps the reads in bounds so that a test, not a crash, detects it) | `label_propagation.cpp`; `sparsify.cu` (`build_subgraph_kernel`) |
| `lp_d_b_affected_too_small` | `DYNG_MUTATION_LP_D_B_AFFECTED_TOO_SMALL` | the endpoints of new edges that were present before the batch are not marked (O's D-b, `4a8a631`) | `label_propagation.cpp`; `sparsify.cu` (`build_subgraph_kernel`, `mark_edge_endpoints_kernel`) |
| `lp_d_c_stale_buffer` | `DYNG_MUTATION_LP_D_C_STALE_BUFFER` | the damped commit goes to the other of two full label arrays, whose entries not updated in the iteration are two iterations old (O's D-c, `8419483`) | `sequential.cpp` (a second full array, swapped); `propagation.cu` (the head path with two device arrays, swapped, copied back; the persistent tail off) |
| `lp_d_e_resurrected` | `DYNG_MUTATION_LP_D_E_RESURRECTED` | an id inserted and deleted in one batch is re-activated (O's D-e, `a38ce8c`: the insertions applied after the deletions) | `cpp/src/graph/apply_slotted_host.cpp` (Step 0's net effect of vertex batches, which the host and the device apply share) |

### 2. The suites that detect them

`parity/mutate.py`'s suite `label_propagation` (a GoogleTest suite: `dyng_label_propagation_tests`
and `_cuda_tests`, the checker tests and the kit's executables of both backends); the CTests
`label_propagation.mutation.<name>` (label `cpu`) and `label_propagation.mutation.cuda.<name>`
(label `gpu`) of every build run the hand cases and the checker at 5k5 against a copy of libdyng
whose label_propagation module (D-e: the graph module) is rebuilt with the define, with the
module's own compile options (`--use_fast_math` on its CUDA sources, no contraction on its host
sources), so that the control reproduces the original's damped counts. The control copy passes on
both backends; each mutation fails at least one test on sequential and on cuda (dev-cuda,
2026-10-30):

| Mutation | Killed on sequential by | Killed on cuda by |
|---|---|---|
| `inplace_plain_jacobi` | `ASeedFreePairConverges` | `ASeedFreePairConverges`, `HubGolden` |
| `damped_no_damping` | `ASeedFreePairConverges`, `DampedIterationsAreTheOriginals` | the same and `HubGolden` |
| `lp_d_a_unfiltered_components` | the stream oracles (components against lp_host_model), `DampedIterationsAreTheOriginals`, the checker at 5k5 | the same |
| `lp_d_b_affected_too_small` | the stream oracles (the initial frontier against V_aff), `EdgeOnlyBatchMarksItsEndpoints`, the checker | the same |
| `lp_d_c_stale_buffer` | `DampedIterationsAreTheOriginals`, `NoHostAllocationInTheAlgorithmPhaseAfterReserve` | `DampedIterationsAreTheOriginals`, `HubGolden` |
| `lp_d_e_resurrected` | `InsertedAndDeletedInOneBatchEndsDeleted`, the stream oracles, the checker (statuses against lp_host_model) | the same and more |

The three hand cases were added for this (M10a step 5): a seed-free pair of adjacent vertices
with different labels converges (plain Jacobi swaps it until the cap), an id inserted and deleted
in one batch ends deleted, and the damped iteration counts at 5k5 are the original's
(0, 1217, 548, 327, 65; its results.csv in the damped golden).

## Consequences

- `parity/mutate.py list` holds the six mutations; a golden-mutation record with their rows needs
  the release's `cpp/tests` (`GTEST_MUTATION_PATHS`), as every GoogleTest suite's.
- The hooks change no default build: the step-5 SASS comparison and the replays are taken on the
  final library.
