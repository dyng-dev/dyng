# ADR 0026: The sssp operators engine on CUDA and the engine choice of 0.2

- **Status:** Accepted under delegation (2026-09-30; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out decision O24 (the `sssp` operators engine in 0.2, the fallback without
  cooperative launch) and changes item 2 of ADR 0017 accordingly; it does not change a rule the
  author approved: the PLAN 8.6 gates and their protocol (ADRs 0018, 0021), the parity rules and
  the public API's signatures are untouched. The operators engine has no performance gate (M7
  acceptance criterion 1: measured and reported against the fused engine).
- **Date:** 2026-09-30
- **Deciders:** the AI assistant, on the author's behalf (M7, step sssp-operators)

## Context

In 0.1 the CUDA backend of `sssp` has one engine, MOSP-CUDA@e220ee2's persistent cooperative
kernel (the fused engine, Tier B; ADR 0017). Without cooperative launch `engine::automatic` and
`engine::fused` throw `not_supported_error`, and `engine::operators` throws everywhere (ADR 0017
item 2). PLAN 4.5.4 and decision O24 require the operator twin from 0.2: a Tier A engine written
with the framework's operators, the fallback where the fused kernel cannot run, byte-identical to
it (conformance check C4). PLAN 6.4.2 names MOSP_ESCHER@4b86159's `mosp/src/sospUpdateGpu.cu`
(the MP1 multi-kernel host loop) as the structure. That file is an earlier state of the same
algorithm and differs from MOSP-CUDA@e220ee2 in details that change results or error reports: it
always runs ceil(log2 n) pointer-jumping rounds and cannot see a parent cycle, its pull and push
pack candidates above the distance bound, it has no distance-only fallback (it rejects such
graphs), and its unpack writes every entry (no `affected`).

## Decision

1. **Structure and semantics.** The operators engine (`cpp/src/algorithms/sssp/operators.cuh`,
   `operators.cu`) has MP1's shape: one kernel per phase (pack the old tree, roots, one kernel
   per pointer-jumping round, invalidation, insertion heads, the pull pass, then a host loop of
   push, minimum, threshold and split kernels, then unpack), launched on the stream of the
   resources handle, with MOSP-CUDA@e220ee2's semantics phase by phase: the same packing choice
   (including the distance-only words and the bound), the same stamp generations per list, the
   same in_far deduplication, the source never relaxed, candidates above the bound never packed,
   the same unpack (only changed entries written, `affected` summed per warp; the lowest-id parent
   recovery of the distance-only mode), and the same input checks (an input distance out of range
   is `device_error::invalid_input`, a vertex still jumping in round ceil(log2 n) + 1 is
   `device_error::parent_cycle`). MP1's simplifications listed above are not taken.
2. **Decisions on the device where the host needs no value.** The grid barriers of the fused
   kernel become kernel boundaries. A pointer-jumping round after a round in which no vertex
   jumped returns at once (the flags live in the control block), so the host enqueues all
   ceil(log2 n) + 1 rounds without waiting and the device ends them as early as the fused kernel
   does. The near-far threshold is computed on the device, and the first minimum and split read
   the frontier's length from the device. The host reads the control block back once after
   Step 1 (the input checks and the number of candidates), once after the first split, once per
   push iteration and per threshold raise (whether to go on, as MP1's `nearFar()`), and once after
   the unpack (`affected`; a result is complete when the call returns, ADR 0017 item 7). Every
   kernel is grid-stride, with at most as many blocks as the device holds at once.
3. **Byte equality with the fused engine.** Both engines run the same monotone algorithm; only the
   order of the atomic operations differs. Each final word is the minimum of its value after the
   invalidation and every candidate offered to it; a candidate that survives carries the vertex's
   final distance and comes from the pull pass (an in-neighbour whose distance did not change) or
   from a push of an in-neighbour at its own final distance, and every vertex whose distance
   decreased is pushed at its final distance. That set does not depend on the schedule, so the
   trees are the same bytes, for canonical and for non-canonical input trees; `invalidated` and
   `affected` count deterministic sets. `iterations`, `epochs` and `pushes` describe the schedule
   and may differ (they are logged, never compared). The tests check it rather than rely on the
   argument (Consequences).
4. **The engine choice (replaces ADR 0017 item 2).** On CUDA, `options::cuda_engine`:
   `engine::automatic` runs the fused engine where the device supports cooperative launch and the
   operators engine elsewhere; `engine::fused` runs the fused engine and throws
   `not_supported_error` on a device without cooperative launch (before anything changes; the
   message names `engine::operators`); `engine::operators` runs the operators engine on every
   device. The host backends ignore the option, as before. The framework's engine selection does
   it (`sssp_problem::select_engine` returns the option, `fused_available` the capability); the
   choice is still made before the commit. `stats::engine_used` reports `operators` for the
   operators engine.
5. **Budget (invariant I9).** The operators engine allocates nothing once reserved and
   synchronizes at most 3 + iterations + epochs times per update; `algorithm_budget` states it
   after the phase (the counts are known then). The fused engine's budget stays 1. The kit's C8
   runs the default engine (the fused one on the gate GPUs).
6. **Where the operators live (rule of two).** The device building blocks the two engines share
   (the packed words, the stamp claim, the warp-aggregated append, the warp reductions,
   `device_csr`) move unchanged from `fused.cuh` to `kernels.cuh`; the fused kernel's machine code
   is unchanged (identical SASS for all three instantiations, checked with `cuobjdump`). The
   operators (`invalidate_subtree` by pointer jumping, the pull `neighbor_reduce` with the packed
   argmin, the push `advance` with `packed_min`, the near / far frontiers) stay in the sssp
   folder: no second algorithm uses them yet (`cycle_count`'s CUDA code shares none of them; the
   planned second user is `hyper_sssp`). Nothing enters `cpp/src/operators` in M7.
7. **Tools.** `dyng-compat-mosp --cuda-engine automatic|fused|operators`; `parity/compare.py
   --configs cuda-fused[:d],cuda-operators[:d]` replays the golden corpus with a forced engine;
   `parity/perf_ab.py engines` measures the operators engine against the fused one (A/B/A/B at
   locked clocks, the protocol of `run`, reported only); the kit's C4 compares the two engines
   after compute() and after three batches for every preset, the small and medium sizes and every
   batch mix.

## Consequences

- No public signature changes (the API listing is unchanged); the behaviour of
  `engine::automatic` on a device without cooperative launch and of `engine::operators` on CUDA
  changes as item 4 says, recorded in the CHANGELOG.
- C4 runs for `sssp` on CUDA and passes; the golden corpus (495 cases), the MOSP fixtures, the
  randomized suites (the CUDA executable compares both engines with each other and with the host
  backends on every chain), the packing boundary and the distance-only fallback pass on both
  engines.
- The operators engine is slower than the fused one where the update is short (one host
  synchronization per near-far round instead of one per update); `parity/results/M7.md` records
  how much on the four gate graphs. It is the fallback, not the default.
