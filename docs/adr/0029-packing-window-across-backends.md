# ADR 0029: The packing window, where the host and CUDA backends follow different originals

- **Status:** Proposed (2026-10-01, M7 review). A parity rule, so it is **not** delegated
  (GOVERNANCE.md, "Delegation of technical ADRs"): the lead maintainer decides between options A
  and B below. Until then the code keeps its behaviour since M1b (option A), and the documentation
  and a test state and pin it.
- **Date:** 2026-10-01
- **Deciders:** S M Shovan (lead maintainer); drafted by the AI assistant

## Context

`sssp` packs (distance, parent) into one 64-bit word when the distances fit next to the parent
bits, and falls back to distance-only words otherwise (every parent is then recovered after the
search with the lowest-id rule). The two pinned originals decide "fit" differently:

| Original | Packs when | dynG backends that copy it |
|---|---|---|
| MOSP-OpenMP@c352151 (`choosePacking` of `sospUpdateCpu`) | (n - 1) * W + W <= max_distance | sequential, openmp (`openmp.cpp` `choose_packing`, `sssp_packs_parents`) |
| MOSP-CUDA@e220ee2 (`choosePacking`, `sospUpdateGpu.cu:670-680`) | (n - 1) * W <= max_distance | cuda, both engines (`cuda.cu` `choose_packing`, `operators.cu`) |

Here n is the number of vertices, W the graph's largest weight and max_distance = 2^(64 - b) - 2
with b parent bits (the smallest b with 2^b - 1 >= n). Inside the **packing window**
(n - 1) * W <= max_distance < n * W the host backends run distance-only and the CUDA engines run
packed. The two modes differ on one input class only:

- from a **canonical** input tree (from `compute()`, or `from_arrays(..., canonicalize = true)`)
  both modes give `compute()`'s tree, so every backend returns the same bytes (only
  `stats::packed_parents` differs; recorded since M1b, M1b retrospective decision 8,
  `docs/history/mosp_cuda.md`);
- from a valid **non-canonical** input tree (`from_arrays(..., canonicalize = false)`, as MOSP's
  `mosp` driver keeps its dataset trees), the distance-only mode replaces every tie parent with
  the lowest id, while the packed mode keeps the tie parents of the vertices the batch does not
  re-evaluate (sssp's tie rule). The distances are equal; the parents differ.

The M7 review reproduced this with n = 65538, W = 2147450879 and 2147450880 (one heavy edge that
is never on a shortest path) and non-canonical Dijkstra trees: dynG's sequential and OpenMP
backends were byte-equal to MOSP-OpenMP, both CUDA engines byte-equal to MOSP-CUDA, and 2214 lines
of `obj0/SSSPTreeUpdated.txt` (and mosp's combined files) differed between the two groups. For
n = 65537 the window is W in [2147450881, 2147483647]; for roadNet-CA (n = 1,971,281) a largest
weight of 4462121 or 4462122. With int32 weights no graph below 65537 vertices reaches it. The
golden corpus, the paper-scale goldens and the randomized suites do not.

Until this review, three pages said the opposite: `docs/algorithms/sssp.md` ("All backends return
identical trees, from canonical and non-canonical input trees alike"), `sssp.hpp`'s
`from_arrays()` ("identically on every backend") and `docs/algorithms/mosp.md` ("identical on
every backend and engine"), against PLAN's C3 ("sequential == OpenMP == CUDA").

## Options

**A. Each backend follows its own original (the behaviour since M1b; recommended).** Byte parity
with both originals is kept on every input, the window included; the cross-backend equality of
C3 holds everywhere except for non-canonical input trees inside the window, which is documented
as the one exception (each backend then equals its original; the distances are equal on every
backend; every tree is a valid shortest-path tree).

**B. One rule on every backend** (MOSP-OpenMP's `bound + W` on cuda, or MOSP-CUDA's `bound` on the
host). C3 holds on every input; byte parity with the other original is given up for non-canonical
trees inside the window (a parity exception against MOSP-CUDA or MOSP-OpenMP instead of a
cross-backend exception). MOSP-OpenMP's rule is the safer one for the CUDA engines (it leaves room
for the candidate bound + W that a relaxation forms before the comparison; the CUDA engines drop
candidates above the bound first, so they need no room), so B would most likely mean "the host
rule everywhere".

## Decision (proposed)

Option A, pending the author's decision:

1. The code is unchanged: the host backends use MOSP-OpenMP's rule, both CUDA engines
   MOSP-CUDA's.
2. The documentation states the exception: `sssp.hpp` (the group text and `from_arrays()`),
   `docs/algorithms/sssp.md` section 4 and `docs/algorithms/mosp.md` (the K trees inherit it; the
   combined graph's weights are far below the window).
3. The test `SsspPackingWindow.NonCanonicalTreesFollowEachOriginalsPackingRule`
   (`cpp/tests/algorithms/sssp/sssp_random_test.cpp`, both the host and the CUDA executable) pins
   the behaviour on n = 65537 at both ends of the window: the host backends run distance-only and
   return the canonical tree; both CUDA engines run packed, keep the imported tie parents of the
   vertices the batch does not re-evaluate and agree with each other; the distances are equal on
   every backend.
4. The per-backend parity of the window was checked once against both originals on a generated
   case (`parity/results/M7.md`, section 11).

If the author chooses B instead, item 1 changes one line in `cuda.cu` and one in `operators.cu`
(or `openmp.cpp` and `sssp_packs_parents`), the test's expectation for the changed backends flips,
and `parity/results/` records the parity exception against the original whose rule is dropped.

## Consequences

- C3 ("cross-backend equality") is qualified by one documented exception under option A; C4
  (fused == operators) is unaffected (both CUDA engines use the same rule).
- `mosp` inherits the exception through its K trees, and hence its combined files, only for
  non-canonical imported trees inside the window; its combined solve is a static solve on weights
  of at most L * (K + 1), far below the window.
- Option A needs no change to the parity harness: every golden case is outside the window.
