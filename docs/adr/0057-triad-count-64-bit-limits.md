# ADR 0057: triad_count's 64-bit limits (exact products, a capacity bit on the device, unsigned phases)

- **Status:** Accepted under delegation (2026-10-20; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It fixes four confirmed findings of the M9a review against the
  documented contract of `dyng/triad_count.hpp` (exact counts, capacity_error only past
  2^64 - 1, identical results on every backend) and amends ADRs 0053 (the device bins), 0054 (the
  host phases) and 0055 (triad_count's use of the operator). It changes no rule the author
  approved: the parity with ESCHER-GPU@abcf9b2 stays exact on every golden case (PLAN 8.3), no
  gate, tolerance or protocol changes, no frozen header (`hypergraph/*`, `update.hpp`, `core/*`,
  `graph/*`) is edited.
- **Date:** 2026-10-20
- **Deciders:** the AI assistant, on the author's behalf (M9a, the review fixes)
- **Builds on:** ADR 0049 (the oracle), ADR 0052 (the API), ADR 0053 (the wedge engine), ADR 0054
  (the host engine), ADR 0055 (`count_delta` / `count_owned`).

## Context

ESCHER-GPU@abcf9b2 computes the StatHyper terms in signed 64-bit and adds them without a check:
`Choose3Op` is `n * (n - 1) * (n - 2) / 6` in `long long`, `Type2Op` `a * (a - 1) / 2 * b * c`,
`Type3Op` `x * y * z`; the block bins and the global bins are plain `atomicAdd`s; the subtract
phase flushes `-bin` in two's complement. The port kept all of it, and the review found:

1. **C(|e|, 3) goes wrong from |e| = 2,097,152** (n (n - 1) (n - 2) passes 2^63 - 1 while
   C(n, 3) fits until n = 4,801,280): on cuda compute() returned 16,915,784,523,091,993,014 for one
   hyperedge of 2,100,000 members (exact: 1,543,497,795,000,700,000) and update() raised an
   internal_error and poisoned the result; on the host the value depended on undefined behaviour
   (`-fsanitize=undefined`: signed integer overflow). The oracle's `uint64_t` choose3 wrapped from
   n = 2,642,246, so past it every backend and the oracle agreed on a wrong value.
2. **No overflow detection on cuda**: a type-1 total past 2^64 - 1 (14 hyperedges of 2,000,000
   members) threw capacity_error on the host backends and came back wrapped on cuda.
3. **Type-2 and type-3 products** past 2^64 - 1 (two hyperedges sharing 100,000 of their 200,000
   members each; three pairwise-only overlaps of 2,700,000) came back wrapped on every backend,
   and the oracle wrapped the same way, so the property tests could not see it.
4. **The signed phase bins** refused a valid update that removes or adds 2^63 or more triads of
   one bin (a type-2 count of 1.2e19 deleted): internal_error ("a library bug") on cuda, a false
   capacity_error on the host, after the commit.

## Decision

### 1. Exact arithmetic in the operations (`engine/exact.hpp`, both engines)

- **C(n, 3)** is computed in unsigned 64-bit with the divisions first: the even one of n and
  n - 1 is halved, the multiple of 3 among the three factors divided by 3, so only the last
  product can pass 2^64 - 1 (n < 2^31: the first product is below 2^62). The value is the
  original's wherever the original does not overflow.
- **Every product** of the type-2 and type-3 operations is unsigned and checked
  (`__builtin_mul_overflow` on the host, `__umul64hi` on the device), and every addition to an
  operation's register sum is checked. An operation keeps an `over` flag beside its sum; its
  `flush()` calls `bins.overflow()` instead of adding a wrapped sum.
- The bins handles report it: the host handle (`operators::host_count_bins::overflow()`, a new
  member) sets `count_overflow`, which the host engine raises as capacity_error; the device handle
  (`wedge::block_bins`) sets the **capacity bit** of the call's device error word
  (`device_error::capacity`), which `triad_cuda_compute` / `triad_cuda_finalize` raise as
  capacity_error through `throw_device_errors` (internal_error still wins when both are set).
- **The device bins carry-test their additions**: `block_bins<num_bins, checked>::add` tests
  `old + value < old` on the `atomicAdd`'s return for the operations that add more than 1
  (types 1-3: once per thread, at their flush); motif_op adds 1 per triangle (`unit_counts`), which
  cannot carry a block's bin (one wedge closes fewer than 2^31 triangles, and a thread of the
  capped grid has at most 2^24 wedges, so a block sees fewer than 2^63), so its shared bins stay
  unchecked on the hot path. `flush_bins` carry-tests the global bins once per block and bin.

### 2. Unsigned phases (`triad_phase_bins`)

Each phase of an update is kept as **unsigned magnitudes per bin with the sign it counted with**
(`triad_phase_bins {sign, counts}`), on both engines: the kernels add magnitudes and no longer
take a sign; the CUDA engine records the signs `count(-)` and `count(+)` were called with; the host
engine's phases run `operators::count_owned` (the unsigned count of ADR 0055's operator; the sign
is kept beside it) instead of `count_delta`'s signed bins. `apply_delta` checks, before it writes
anything, that the subtract phase counted with `minus` and the add phase with `plus` (the
recorded mutation `triad_wrong_bin_sign` still fails there: internal_error), that no bin goes
negative (internal_error, as before), computes the count after as `count - removed + added` with
no intermediate past 2^64 - 1, and raises capacity_error when that count exceeds 2^64 - 1.

`stats::delta` stays signed 64-bit (PLAN I7, the M9a criterion 5 shape): a bin whose net change
is outside [-2^63, 2^63 - 1] raises **capacity_error** with the same message on every backend
(after the commit, so the result is poisoned, as for any count past 2^64 - 1). A batch that
removes and adds 2^63 or more triads of a bin with a small net change is now exact.

M9a criterion 6's "Device bins are signed 64-bit, with minus and plus kept apart" becomes: the
device bins are 64-bit unsigned magnitudes, minus and plus kept apart, the signs applied and
checked on the host, a bin that would go negative an internal_error that poisons the result. The
deviation keeps the criterion's purpose (no wrap, no mixing of the phases, the negative-bin
check) and removes the false refusal of finding 4.

### 3. The oracle (`testing::triad_reference`)

The oracle computes every product and sum in unsigned 128-bit integers and throws
`std::overflow_error` when a requested count exceeds 2^64 - 1. It shares none of the engines'
overflow tests (its own arithmetic, still standard library only: `triad_reference_isolation`
holds), so at the 64-bit limit it checks the engines instead of wrapping with them.

## Consequences

- `dyng/triad_count.hpp` documents the limits: every count exact up to 2^64 - 1 on every backend,
  capacity_error past it, and capacity_error for an update that changes a bin by 2^63 or more.
- Tests: `dyng_triad_count_limits_tests` (host; labels parity, slow) and
  `dyng_triad_count_limits_cuda_tests` (gpu): one hyperedge of 2,100,000 members (compute and
  update, inserting one of 3,000,000 and deleting the first), C(4,801,281, 3) and two
  hyperedges of 3,900,000 members (each in (2^63, 2^64), the sum past it) refused, an update that
  takes type 1 past 2^64 - 1 refused and poisoning the result, type 2 at k = 70,000 exact
  (1.2e19) and at k = 100,000 refused, a batch that removes and adds 1.2e19 exact with a delta of
  0, a net change of 1.2e19 in either direction refused on every backend, type 3 at
  x = 2,200,000 exact (1.06e19) and at 2,700,000 refused; the oracle's values and refusals at the
  same sizes. `dyng_triad_count_cuda_tests` checks the device arithmetic against the host's
  (C(n, 3) at the boundaries, the checked products and sums) and the capacity bit (a carry in a
  block bin, `overflow()`, a carry in the flush).
- ESCHER-GPU has the same overflows; the goldens never reach them (coauth-DBLP's largest
  hyperedge has 280 members), so every golden case is unchanged. The deviation from A is
  recorded in the M9a retrospective.
- The h-motif kernel's hot path is unchanged but for the flush (a carry test once per block and
  bin) and the dropped sign; the SASS reading and the gates are taken again at HEAD
  (parity/results/M9a.md, section 6).
