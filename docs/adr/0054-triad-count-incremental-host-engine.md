# ADR 0054: The incremental host engine of triad_count and the oracle's independence from it

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the author's decision D3 of 2026-10-15 (the
  sequential and OpenMP backends of `triad_count` are incremental host engines, an untuned
  transliteration of ESCHER-GPU's wedge engine with the same operations, no performance gate;
  `dyng::testing::triad_reference` is the independent oracle) in M9a step 4 (the M9a acceptance
  criteria 6, 8 (host), 10 and 11). It changes no rule the author approved: the parity is exact
  (PLAN 8.3), the gates and their protocol are untouched, no frozen header (`hypergraph/*`,
  `update.hpp`, `core/*`, `graph/*`) is edited, the CUDA engine's machine code is unchanged.
- **Amended by:** ADR 0057 (2026-10-20): the phases run `count_owned` as unsigned magnitudes with
  their signs beside them (the signed bins of `count_delta` refused a phase of 2^63 or more).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step host-engine-kit-mutations);
  the scope change against PLAN 4.6 is the author's (D3)
- **Builds on:** ADR 0033 (executors), ADR 0043 (the hypergraph's update path and the kit's axis),
  ADR 0049 (the oracle), ADR 0050 (the replays), ADR 0052 (the API), ADR 0053 (the CUDA port),
  ADR 0055 (`operators::count_delta`).

## Context

PLAN 4.6 (backend matrix) makes the host backends of `triad_count` reference backends ("R: ESCHER
CPU reference; recount-based delta", the OpenMP backend becoming incremental in 0.4) and the
roadmap's 0.5 lists "an OpenMP backend for incremental `triad_count`". M9a step 3 shipped that
plan: `update()` recounted the whole hypergraph on the host (`stats::fallback_used`). The author's
decision D3 replaced it: the host backends run the incremental algorithm, a transliteration of the
CUDA engine, with no performance gate, and the oracle stays independent of both engines.

## Decision

### 1. The host engine (`cpp/src/algorithms/triad_count/host_engine.hpp`)

ESCHER-GPU@abcf9b2's `kernel/wedge.cuh` as loops over the internal row views in host memory (the
host stores use A's layout, ADR 0042): one *source* is what the device engine's threads do for
that source's wedges.

- **Full mode** (compute()): the sources are the keys 1..`full_count_keys(kind, views)`
  (`engine/phases.hpp`, shared with the CUDA engine: `views.h2v.max_key` for type 1, the line
  graph's `views.h2h.max_key` otherwise); a source `a` takes its out-wedges `(a, b)`, `b > a`
  (from `out[a]`), and closes a wedge into the triangles `(a, b, k)`, `k > b`, by merging the rows
  of `a` after `b` and of `b` after `out[b]` (TriangleBody); nothing is filtered.
- **Frontier mode** (a phase of update()): the sources are the frontier's keys; a source `f` takes
  every wedge of its row (from `loc[f]`), skips a wedge `(f, b)` when `b` is in the frontier and
  `b < f`, and skips a closing `k` when `k` is in the frontier and `k < f`: the predicates of
  `engine/ownership.hpp` (`min_member_skips_wedge`, `min_member_skips_closing`), the same ones
  `wedge.cuh` calls, with `ownership::min_member` passed explicitly (invariant I2).
- **Shapes**: triangles (`motif_op`, 30 bins; `type3_op`), pairs (`type2_op`: every wedge is a
  pair), keys (`choose3_op`: C(|e|, 3) of each source). The operations are the CUDA engine's
  (`engine/*_ops.hpp`, the Op form of ADR 0053's c6, host-and-device structs) with the host bins
  handle of `operators::count_delta`.
- **Early outs as in A's countWedgeShape**: no sources (an empty frontier) and sources without a
  wedge (the wedges are counted first, as the device scan does); the launch counters of the
  result count them and the passes per (mode, shape) on every backend (criterion 9's tests run on
  the host backends too).
- **The pass**: `operators::count_delta` (ADR 0055) on `sequential_exec` (sequential.cpp) or
  `openmp_exec` (openmp.cpp): a static partition of the sources (chunks of 64 dealt to the threads
  in turn), private bins per thread, added in thread order after the pass; the sign of the phase
  applied once per bin into signed 64-bit bins. 1, 8 and 56 threads give identical counts and
  counters (tested).
- **Errors**: a pattern of no class (motif_op's bin -1) is reported by the bins handle and raised
  as `internal_error` (never written: tested on all 128 Venn ids); a count past the 64-bit range is
  a `capacity_error`. Both come after the commit and poison the result.

### 2. The problem's hooks on the host backends (Tier A)

The aggregate-delta template as on cuda, through the framework's Tier A hooks:

- **count(-)** (`triad_count.count_minus`, on G_t): lease the host workspace
  (`triad_host_workspace`, pooled, ADR 0015) and reserve what both phases need before the commit
  (frontier keys for the larger phase, flags for `views.h2h.max_key + insertions + 1` keys, the
  per-thread bins); the frontier of Step 0's live distinct deletions; the subtract phase into the
  subtract bins.
- **commit**: `hypergraph.apply`.
- **identify_affected** (`triad_count.identify_affected`, new on the host): the frontier of the
  ids the commit ASSIGNED (`triad_inserted_ids()`, shared with cuda: `hyper_apply_summary::
  inserted_ids`, on cuda their device copy). A frontier id outside `[0, views.h2v.max_key)` is an
  `internal_error` (every loop indexes the rows by these keys).
- **count(+)** (`triad_count.count_plus`, on G_{t+1}): the add phase into the add bins.
- **finalize** (`triad_count.finalize`): `apply_delta(minus, plus)`, the CUDA path's: the signs
  checked, a bin that would go negative an `internal_error` that poisons the result, the stats
  (`triads_removed`, `triads_added`, `delta`), `fallback_used` false.

compute() on the host backends runs the full mode (`triad_count.count`). Both engines take the
phase's sign from `triad::phase_sign()` and the add phase's ids from `triad_inserted_ids()`, so
the recorded mutations of PLAN 8.4 act on both (ADR 0055 is the operator; criterion 11's suite).

### 3. The oracle stays independent

`dyng::testing::triad_reference` (ADR 0049) shares no code with either engine: its own OBJECT
library, A's `utils/reference.cpp` algorithm (a brute force over the hyperedges' vertex rows, not
the line graph's wedges), its own class map derived from the permutation orbits; the isolation
check `triad_reference_isolation` (no include or link of `cpp/src/algorithms/triad_count`, the
engine or the container internals) is unchanged. The host engine is not an oracle: the tests
compare every backend with `triad_reference` (counts, `removed()` / `added()` of the phases), with
a fresh `compute()` and, on the golden set, with A.

### 4. Scope against PLAN 4.6 and the roadmap

- PLAN 4.6's backend matrix entry for `triad_count` ("R (ESCHER CPU reference; recount-based
  delta)", "R (same code, OpenMP) -> real incremental in 0.4") is superseded by D3: both host
  backends are incremental from 0.3.0. The plan's recount stays the documented fallback (a
  backend that cannot run the engine may recount and report `fallback_used`); no backend uses it.
- The roadmap's 0.5 item "an OpenMP backend for incremental `triad_count`" is performance tuning
  of this backend (D3; `docs/roadmap.md`): the static partition, the per-source merges and the
  absence of a wedge-parallel schedule are not tuned, and the host backends have no gate.
- PLAN 5.8 / 8.2's oracle requirement is met by `triad_reference`, not by the sequential backend.

## Consequences

- `stats::fallback_used` is false on every backend; `triads_removed` / `triads_added` are the
  phases' counts everywhere, so the kit's deterministic counters include them (C3, C9) and the
  tests check them against the oracle on every backend.
- The kit runs the four patterns on sequential, openmp (8 threads: the traits' new optional
  `kit_openmp_threads`) and cuda.
- CPU parity against A (criterion 8, host): `parity/results/M9a.md` section 4.
- A phase of an update on the host is limited to 2^63 - 1 triads per bin (signed bins); compute()
  keeps the unsigned 2^64 - 1 limit (`count_owned`).

## Outcome (M9a step 4)

Exact against ESCHER-GPU@abcf9b2 on the host coverage table, by key and by content, with
integrity and a fresh `compute()` after every batch: sequential 120/120 (make-test and the syn100k
grid), OpenMP 142/142 at 8 threads and 142/142 at 1 thread (also the syn1m s1 grid and
coauth_b50k_s1 types 1 and 2), the two OpenMP records identical case by case. The kit passes on
the three backends; the five triad mutations and `cbst_unfill_drops_last` are detected on both
engines; the CUDA kernels' machine code is unchanged (`parity/results/M9a.md` section 4).
