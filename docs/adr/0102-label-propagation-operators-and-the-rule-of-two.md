# ADR 0102: label_propagation's operators and the rule of two

- **Status:** Accepted under delegation (2026-11-11; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10b step 1 (the M10b acceptance criterion 2; PLAN 4.5.3, 4.5.4,
  6.4.7, 11.4). It records a deviation from the M10 row of PLAN 11.4, which lists
  `binned_frontier`, `neighbor_reduce` and `connected_components` among the framework operators;
  it changes no rule the author approved.
- **Date:** 2026-11-11
- **Deciders:** the AI assistant, on the author's behalf (M10b, step operators-engine)

## Context

PLAN 4.5.3 names three operators whose first user is `label_propagation`: `binned_frontier<id_t,
num_bins>` (per-degree-bin queues, half-sweep sets, iteration stamps; DynLP's propagation),
`neighbor_reduce<direction>` (pull; sssp's pull with the packed argmin and label_propagation's
weighted average) and `connected_components` (Shiloach-Vishkin on CUDA, union-find on the CPU;
label_propagation, later a WCC algorithm). PLAN 11.4's M10 row adds them to the operator set.
The same section sets the rule of two: an operator enters `cpp/src/operators/` or
`cpp/src/framework/` only when at least two algorithms (the tutorial algorithms count) use it;
until then it stays in the algorithm's folder. In 0.4 each of the three has one user:
label_propagation's operators engine (ADR 0101). sssp's pull (the named second user of
`neighbor_reduce`) has its own packed-argmin operator in `cpp/src/algorithms/sssp/` (ADR 0026)
and is reworked by M11 (M10 does not touch it); no WCC algorithm exists.

## Decision

1. **Where they live.** The three operators are executor-templated code (ADR 0033's pattern) in
   `cpp/src/algorithms/label_propagation/operators/`, namespace `dyng::detail::lp_ops`: a host
   header per operator (`binned_frontier.hpp`, `neighbor_reduce.hpp`, `connected_components.hpp`:
   the contract, the data structures and the sequential implementation) and a CUDA header
   (`.cuh`: the cuda_exec implementation, CUDA translation units only). The caller names the
   executor and passes the resources (`neighbor_reduce<operators::cuda_exec>(res, ...)`): an
   operator that launches its own kernels needs the stream, which `operators::cuda_exec` keeps
   private (`cpp/src/operators/` is not changed in M10). This deviates from PLAN 11.4's M10 row
   (which lists them as framework operators) by the PLAN's own rule of two.
2. **Their contracts.**
   - `binned_frontier<id_t, num_bins>`: two sets that alternate between iterations, one queue per
     degree bin and half-sweep (half 0 from the front of the bin's array, half 1 from the back:
     the original's layout), run-wide stamps (a vertex enters a set once per iteration: the stamp
     holds the tag of the last set that holds it), the bin of the current degree and the half of
     a hash of (vertex, tag); `push` (on the device warp-aggregated, one atomic per group of lanes
     with the same bin and half: the original's appendToQueue), `clear`, `read_size` (one host
     synchronization on CUDA), and `expand` (the changed vertices of a list and their eligible
     neighbours into the next set, with 8 or 32 lanes or a block per vertex).
   - `neighbor_reduce`: for every vertex u of a list, the sums over u's live entries (a column
     >= 0 whose vertex is not deleted) of w and of w x[v], in the association order of a lane
     mapping (8 lanes, a warp, a block of 256 or 1024 threads: lane l sums the entries l, l + L,
     ... with one fused multiply-add per entry, then the original's shuffle tree; a block adds
     its warps' sums in warp order), and `apply(i, u, average)` once with the original's guard
     (`average = sum(w x) / sum(w)` when `sum(w) > 1e-10f`, else x[u]). The sequential
     implementation computes exactly that order, so its sums are the CUDA operator's bits; in
     label_propagation's fast-math target the CUDA division is the original's approximate one.
   - `connected_components`: the components of an edge list over a vertex list with canonical
     ids (a component's root is its smallest vertex; the roots numbered in list order: the
     original's numbering). Host: union-find (union by the smaller root, path halving). CUDA:
     Shiloach-Vishkin with cuda_exec's `for_each` (hook the larger root under the smaller with
     atomicMin, compress, until a round changes nothing: one host synchronization per round),
     then the root flags scanned with CUB and numbered (one more synchronization for the count).
3. **Their tests** (`cpp/tests/algorithms/label_propagation/label_propagation_operators_test.cpp`,
   `label_propagation_operators_cuda_test.cu`, inputs shared in
   `label_propagation_operators_cases.hpp`): on sequential_exec against independent models (a
   double-precision average over the live entries for every mapping; the association order by
   hand; the bins, halves, stamps and layout; expand's set by brute force; the canonical ids by a
   breadth-first search), and on cuda_exec against the sequential model: neighbor_reduce's
   averages bit for bit on every mapping (the test's translation unit has no fast math), the
   device pushes and expand on every lane count giving the host's sets per bin and half, and
   Shiloach-Vishkin giving the host's ids with one synchronization per round and one for the
   count. The inputs hold soft-deleted entries, unused slots, deleted vertices and hubs of the
   degrees 8, 9, 32, 33, 1024, 1025, 3000 and 6000 (the bins' edges and above).
4. **The promotion rule.** An operator moves to `cpp/src/operators/` (namespace
   `dyng::detail::operators`, with the host executors' `openmp_exec` implementation then owed)
   in the change that adds its second user, which also moves its tests to `cpp/tests/operators/`
   and runs them on every executor. The named candidates: a WCC algorithm
   (`connected_components`) and sssp's pull in 0.5 (`neighbor_reduce`, if its packed argmin is
   written as a reduction functor over the same mappings). `binned_frontier` has no named second
   user. Until then nothing else may include `algorithms/label_propagation/operators/`.

## Consequences

- No public API: the operators are `detail` code of one algorithm. `cpp/src/operators/` (and its
  `execution.hpp`, M11's carried item) is not changed.
- The operators engine (ADR 0101) is written with them; its parity with the fused engine is
  bitwise on P+ (parity/results/M10b.md section 1).
- The rule of two stays as PLAN 4.5.3 wrote it; PLAN 11.4's M10 row is read through it (the
  retrospective of M10b records the deviation).
