# ADR 0101: label_propagation's operators engine on CUDA and AD6's c

- **Status:** Accepted (2026-11-11). The parity rule is the author's: **AD6, decided by the
  author on 2026-10-28** (GOVERNANCE.md's approvals log, the row of 2026-10-28 that holds AD6:
  byte parity of the fused engine; a refactored engine within max |dF| <= c·ε per batch with
  ε = 2^-23 and c = 3 x the maximum from perturbing association and contraction only, rounded up
  to a power of two, frozen before the refactor; identical counts required; if identical counts
  are impossible the engine stays non-default and reported, never with a looser bound). The
  calibration's tool, the engine and its tests are accepted under delegation (GOVERNANCE.md,
  "Delegation of technical ADRs"), technical scope only. M10b step 1 (the M10b acceptance
  criteria 2 and 3; PLAN 4.5.3, 4.5.4, 6.4.7, 8.3). It follows ADR 0026's pattern (sssp's
  operators engine) and uses ADR 0102's operators.
- **Date:** 2026-11-11
- **Deciders:** S M Shovan (AD6); the AI assistant, on the author's behalf (the rest)

## Context

M10a ported LabelPropagation-CUDA@a276a3a's (O's) pipeline as label_propagation's fused CUDA
engine (Tier B, ADR 0075), bitwise O's damped labels. `options::cuda_engine = engine::operators`
threw `not_supported_error` until M10b. PLAN 6.4.7 asks for "the `binned_frontier` +
`neighbor_reduce` operators where parity allows" (Tier A), judged by AD6's refactored-engine
rule; PLAN 4.5.4 makes the operators twin optional for in-place label propagation.

## Decision

1. **c first (frozen before the engine's code).** `compare.py label_propagation calibrate`
   (`parity/label_propagation_engines.py`) builds `dyng-compat-dynlp` from `git archive HEAD`
   once per variant in one tree; a variant sets the cache variable
   `DYNG_LP_CALIBRATION_CUDA_FLAGS`, which only label_propagation's CUDA sources take (empty in
   every preset and release build): `control` (none), `no_fma` (`-fmad=false`, the contraction),
   `permuted` (`DYNG_CALIBRATION_LP_PERMUTED_REDUCTION` in `propagation.cu`: the lane and warp
   trees in the other association order, a block's warp sums from the last; compiled out of every
   library build) and `both`. Every batch of P (the nine non-digest cases of P+), damped, fused,
   against the twin's per-batch label bits. The control must be exact. Record
   `parity/results/M10b-calibration.json` (on `cc9641b`): the control exact; max |dF| = 2.411e-2
   (`permuted`, 500k, batch 1) = 202,210.75 ε; **c = 2^20, c·ε = 0.125**, frozen with the
   record's SHA-256 in `parity/goldens.toml` `[sets.label_propagation_damped.refactored]`
   (commit `242be2d`, before the engine). Every perturbed run changed a count (frontier_visits,
   some iterations): only an engine whose arithmetic is the fused engine's to the bit can keep
   AD6's identical counts on P, so the engine is built for that.
2. **The engine** (`operators_engine.hpp` / `.cu`, Tier A on the framework's hooks; ADR 0102's
   operators):
   - `before_apply` on G_t is the fused engine's (the affected flags cleared, the deleted
     vertices' neighbours marked on the rows before the batch);
   - `identify_affected`: the seeds' classes, the inserted vertices marked and stamped, G' (the
     fused engine's sparsify pass, or the endpoints of a batch without insertions, ADR 0072) and
     its components with `connected_components` (Shiloach-Vishkin; the same partition and ids);
     the labels before the update kept;
   - `seed`: the fused engine's supernode initialization (64-bit fixed point, exact);
   - `loop`: the damped propagation with `binned_frontier` and `neighbor_reduce`: the first set
     from the affected flags, then per iteration and half-sweep and bin one `neighbor_reduce`
     launch with the damped rule (F' = 0.5f (F + average), clamped, into the bin's side array;
     changed when |F' - F| > δ), one `expand` launch per list (the changed vertices and their
     active unlabelled neighbours into the next set), the commit after both half-sweeps, and one
     synchronization for the next set's size; the seeds' labels restored; at the cap
     `on_limit::fallback_recompute` runs IrLP (the fused engine's: the published baseline,
     unchanged) and `on_limit::error` throws `convergence_error`;
   - `finalize`: the changed labels counted and G''s size read in one synchronization, the
     graph's report settled after the first synchronization (as the fused engine), the stats.
3. **The arithmetic is the fused engine's.** Each bin's mapping is the fused engine's: 8 lanes
   (small bin), a warp (medium), and for the large bin a block of 1024 threads while the frontier
   holds at most 1024 vertices (the fused engine's persistent single-block kernel takes those
   iterations) and of 256 threads above (its multi-block kernels). The sums are one fused
   multiply-add per entry in lane order, the shuffle trees and the block sum are the original's,
   and the division and the damped rule are compiled in label_propagation's fast-math
   translation unit. The frontier sets (and so the iterations and frontier_visits) depend only
   on which labels change by more than δ, not on the order of the pushes. The persistent kernel
   itself has no operators twin: every iteration is one kernel per phase.
4. **The engine choice.** On CUDA, `update()` runs `options::cuda_engine`: `engine::automatic`
   stays the fused engine (unchanged), `engine::fused` the fused engine, `engine::operators`
   this engine. `schedule::in_place` has no operators twin (PLAN 4.5.4: non-deterministic; AD6's
   option not taken): `engine::operators` with `schedule::in_place` raises `not_supported_error`
   in update()'s entry checks, before anything changes. `compute()` is IrLP, the published
   static baseline, on every engine (`select_engine` returns the fused engine for compute()).
   `stats::engine_used` reports `operators`. The host backend has one engine (unchanged).
5. **Budget (I9).** No allocation once reserved (`result::reserve()` also sizes the engine's
   per-position flags when the result's engine is `operators`); host synchronizations per update:
   the Shiloach-Vishkin rounds + 1 when vertices were inserted, 1 + the iterations of the
   propagation, IrLP's (iterations + 1) under the fallback, and 1 in finalize; counted with
   `note_host_sync` and declared through `algorithm_budget` after the phase (the counts are
   known then), checked under strict budgets by its tests.
6. **The stage trace** (ADR 0073) records the fused engine and the sequential backend; the
   operators engine writes no stage records (its parity is the labels' bits and the counts).
7. **Mutations (AD9).** The engine shares the fused engine's sparsify pass (D-a, D-b) and the
   graph's Step 0 (D-e); its own damped rule and commit carry the damping and D-c hooks. The
   CTests `label_propagation.mutation.cuda_operators.<name>` run each mutant on the operators
   engine's cases alone; `inplace_plain_jacobi` acts on schedule::in_place, which this engine
   does not run, so it has no row there (the fused and host rows keep it).
8. **Tools.** `compare.py label_propagation operators` (P+: max |dF| <= c·ε per batch against
   the twin's bits, 5m by its per-batch digests, with the fused engine's own dump of the same
   case as the reference if a digest differed; identical per-batch iterations, components and
   frontier_visits), the kit's C4 (fused against operators after compute() and three batches,
   damped bitwise), `perf_ab.py label_propagation engines` (the fused against the operators
   engine, reported).

## Consequences

- **Outcome under AD6: identical counts, no fallback.** On P+ every batch's label bits equal the
  fused engine's (max |dF| = 0 <= c·ε, 5m by its digests) with identical iterations, components
  and frontier_visits (parity/results/M10b.md section 1); the hub golden holds on the operators
  engine too. AD6's pre-agreed fallback was not needed; `engine::automatic` stays the fused
  engine (the operators engine is slower, reported in parity/results/M10b.md).
- The fused engine's code is unchanged (the calibration hook is compiled out; the fused damped
  bytes on P stay exact).
- No public signature changes: `options::cuda_engine`'s documentation changes (operators on
  CUDA for update(), damped only; compute() IrLP on every engine), recorded in the CHANGELOG.
