# ADR 0055: operators::count_delta enters cpp/src/operators (the rule of two)

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only: an internal-stable addition (PLAN 5.9: `cpp/src/operators/` changes
  need an ADR and must keep every in-tree algorithm green). M9a step 4 (PLAN 4.5.3, the rule of
  two; 4.5.5 I2 and I7; 8.5's micro-benchmark list names `count_delta`).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step host-engine-kit-mutations)
- **Builds on:** ADR 0033 (the executors and the tutorial algorithms), ADR 0054 (triad_count's
  host engine), ADR 0024 (the dynG-vs-dynG bar), ADR 0042 (the line-graph merge).

## Context

PLAN 4.5.3's operator table lists `count_delta<ownership>` (signed, 64-bit bins) with the users
cycle_count, triad_count and triangle_delta, under the rule of two: an operator enters
`cpp/src/operators/` once two algorithms (the tutorial algorithms count) use it. Until M9a
`triangle_delta` (a tutorial, ADR 0033) was its only user, and kept its count in its folder
(`docs/developer/framework.md`, "What is not in the framework yet"). triad_count's host engine
(ADR 0054) is the second: each phase counts the structures through its changed elements, each
counted once by its owner, with a sign.

## Decision

### 1. The operator (`cpp/src/operators/count_delta.hpp`)

```cpp
template <int num_bins, typename exec_t, typename ownership_t, typename function_t>
std::uint32_t count_delta(const exec_t& exec, framework::sign s, ownership_t rule,
                          std::int64_t n, const function_t& f, std::uint64_t* scratch,
                          std::int64_t* out);   // out[k] += s * (bin k's total)
template <int num_bins, typename exec_t, typename ownership_t, typename function_t>
std::uint32_t count_owned(const exec_t& exec, ownership_t rule, std::int64_t n,
                          const function_t& f, std::uint64_t* scratch, std::uint64_t* totals);
```

- The work of element i is `f(i, rule, bins)`, adding with `bins.add(bin, count)`.
- **The ownership rule is required, with no default** (I2): the overloads without it are
  `static_assert`s ("count_delta needs an ownership rule: pass framework::ownership::min_member{}
  ... (invariant I2: exactly-once counting has no default rule)"), and a type that is not a rule
  (`framework::is_ownership_policy_v`) stops at another. The compile-failure CTests
  `operators.count_delta.static_assert.*` show both, and that the triad host engine's phases
  cannot be called without a rule.
- **The sign** is applied once per bin to 64-bit totals (I7); `count_owned` is the unsigned count
  (a full count: compute()).
- **Host executors** (`sequential_exec`, `openmp_exec`): a static partition (chunks of 64 elements
  dealt to the threads in turn), private bins per thread on its stack, stored in `scratch`
  (`count_scratch_words<num_bins>(exec)`: one set of bins and a status word per thread, the
  caller's workspace, so a steady-state update allocates nothing) and added up **in thread order**
  after the parallel region: the totals do not depend on the thread count or the schedule. An
  exception inside the region is carried out of it and rethrown.
- **cuda_exec**: one `for_each` whose bins handle adds to device words with atomics, and one
  `download` of the bins (one host synchronization, counted); an empty phase launches and reads
  nothing.
- **Errors are returned** as bits (`count_invalid_bin`: a bin outside the range, never written;
  `count_overflow`: a total past 2^64 - 1, or a signed bin leaving the 64-bit range on the host),
  so each caller raises them with its own message.
- The executors gain `download(host, src, n)` (several scalars, one synchronization) and the host
  executors `threads()`.

### 2. The users

- **triangle_delta** (tutorial; sequential, OpenMP and CUDA): its engine's `count_owned` becomes
  `count_delta(res, g, changes, sign, ws)`, the update's two counts through the operator with the
  sign the enactor passes (the problem subtracts the minus phase's magnitude). Its functor takes
  `(i, rule, bins)`. The static count is unchanged (no ownership). Its results, budgets (one host
  synchronization per count on cuda) and kit are unchanged.
- **triad_count's host engine** (ADR 0054): every phase through `count_delta`, the full count
  through `count_owned`, on `sequential_exec` and `openmp_exec`. The CUDA wedge engine keeps its
  own kernels (Tier B; block bins in shared memory, machine code identical to ADR 0053's c6), and
  shares with the host engine the operations, the exactly-once predicates and the phase signs.

### 3. cycle_count is not moved (deferred)

PLAN's table names cycle_count as a user too. The step's rule allows the move only if cycle_count's
goldens on every backend, its CUDA SASS and resource usage and an ADR 0024 bar of 1.02 are
unchanged. The bar is a timing reading, and this step takes none (M9a step 4: "no timed readings");
cycle_count's host phases (per-thread marks and counters, a dynamic schedule over the changed
edges, `run_host_phase`) would change shape. The move is **deferred** to a milestone that can
measure it (for example the 0.5 tuning work); until then cycle_count keeps its phases, and the
operator has the two users the rule asks for.

### 4. device_frontier is not promoted

`dense_frontier` (PLAN 4.5.3: a flag per id and the list; A's `DeviceFrontier`) has two
implementations: the line-graph merge's `escher::device_frontier`
(`cpp/src/hypergraph/line_graph/device_frontier.cuh`: built from host id lists, with buffers of its
own per merge) and triad_count's CUDA frontier (keys and flags built by a kernel in the pooled
workspace, the inserted ids from the applied record's device copy, no upload; ADR 0053, c3).
Sharing one type means changing one of them: the merge's code (which the step forbids without the
container's replays, the maintain reading and an ADR 0024 bar) or triad_count's (a per-call
allocation and an upload, against I9). So it is **not promoted**; the two stay where they are.

## Consequences

- `cpp/src/operators/` holds the executors, the atomics and `count_delta`; framework.md's
  "Writing hooks with operators" and the operator table say so.
- Tests: `dyng_operators_tests` (host: the sign once per bin, the thread-count invariance at 1, 3,
  8 and 56 threads, the empty phase, an invalid bin, the 64-bit limits, the scratch size) and
  `dyng_operators_cuda_tests` (the device totals equal the host's, one synchronization, none for an
  empty phase, an invalid bin); the compile-failure CTests of I2.
- The micro-benchmark of PLAN 8.5 (`count_delta`) has its operator; the benchmark itself is not
  part of M9a.
