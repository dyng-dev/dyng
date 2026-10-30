# ADR 0075: label_propagation's fused CUDA engine, the verbatim port of LabelPropagation-CUDA

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 4 (cuda-device-apply-fused-engine-parity; PLAN 4.5.4
  Tier B, 4.7.1-4.7.5, 6.4.7, 7.2, 7.4, 8.3; the M10a acceptance criteria 12, 13 and 14 (CUDA)).
  The rules it carries out are the author's, **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log): AD21, the one stream of the resources (the row "API semantics of
  0.4 (sheet E2-E10)"); AD6, the fused engine's byte parity, AD5, the in_place calibration (as
  refined on 2026-10-29: the row "The in_place iteration rule"), AD7, the harmonic oracle (the rows
  "The parity rules of 0.4 (sheet C1-C8)" and "The AD7 margin on C"); AD2's scope of the timed
  region (the row "The measurement protocol and gates of 0.4 (sheet B1-B6)"); AD12's names and AD20's
  determinism levels (ADR 0071). It builds on ADR 0015 (pooled workspaces), ADR 0017 (sssp's
  verbatim fused engine), ADR 0053 (the transplant in mechanical classes), ADR 0066 (the build-flags
  go/no-go), ADR 0072 (the hooks and the sequential backend) and ADR 0074 (the device rows).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  cuda-device-apply-fused-engine-parity)

## Context

AD6 requires the CUDA damped labels to be the original's bit for bit, which only a straight port
compiled the original's way can give (PLAN 6.4.7); ADR 0066 showed that dynG's FAST_MATH nvcc
line under parity-cuda compiles O's six pipeline TUs to identical SASS.

## Decision

### 1. The transplant (`cpp/src/algorithms/label_propagation/`)

One CUDA TU per TU of O, each with its provenance header: `sparsify.cu` (sparsifySubgraph.cu),
`components.cu` (connectedComponents.cu), `label_init.cu` (labelInitialization.cu),
`propagation.cu` (iterativePropagation.cu and headers/iterativePropagation.h), `irlp.cu`
(validation.cu), and `cuda.cu` (dynLP.cu's batch on the framework's hooks); the host-only
declarations are `fused.hpp`. O's graph TU (dynamicGraph.cu) is the graph module's device apply
(ADR 0074). The kernels keep O's bodies, parameter lists, launch configurations (256-thread
blocks, 8-lane groups, warps, 256-thread blocks per large vertex, the persistent
`<<<1, 1024>>>` block), reductions (the 4/2/1 and 16..1 shuffle trees, the shared-memory block sum,
IrLP's tree) and the run-wide tags. The mechanical changes, each applied to the whole file:

| Class | What changed |
|---|---|
| names | snake_case kernels in `dyng::detail::lp_cuda` (rename map: `parity/label_propagation_sass.py` PORT_KERNELS); O's names stay in the comments |
| types | the row offsets templated (`edge_t`): the int32 instantiation is O's code; the statuses with `vertex_status`'s values (active 0 and deleted 1 are O's codes; O's ground-truth status is the result's seed label, read through `gtClass` as O's kernels already do) |
| stream | every kernel, copy and memset on the resources' one stream (AD21; below) |
| memory | the run-wide arrays (labels, `gtClass`, `newStamp`, the queue stamps, `nextTag`) in the result's state (`lp_cuda::state`, created by compute()); every other array in the pooled workspace (`lp_cuda::workspace`, ADR 0015): the queues and capacity-derived arrays sized from the row capacities, the per-batch ones (CC, label init, G') by `result::reserve()` exactly as O sizes them for the largest batch, with headroom only when an update finds a larger one; pinned mirrors from the staging resource |
| errors | CUDA errors thrown (`DYNG_CUDA_TRY`, `DYNG_CHECK_KERNEL`); O's `exit()` calls typed: the 64-round cap of Shiloach-Vishkin and the label-init workspace overflow are `internal_error` (both unreachable: malfunction guards), the graph's overflow and re-insertion the device apply's report (ADR 0074) |
| counters | O's `prof::` counters returned (`propagation_report`: iterations, frontier visits, persistent iterations, the first queue's size, the host synchronizations) and recorded as stats and the profiler counters `label_propagation.hook_rounds`, `label_propagation.persistent_iterations` (and `irlp_iterations` for compute() and the fallback) |

The class split is coarser than ADR 0053's one-class-per-commit: the classes above cannot run in
the library one at a time (the original's code allocates, streams and exits on its own), so they
land in one commit (`863fff7`), as ADR 0053's c1 included its glue, and the goldens were replayed
on it; the later commits (`6cbd245`, `04aed8e`) change no kernel. **The go/no-go of the transplant
is the SASS:** `parity/label_propagation_sass.py port` finds all 20 ported kernels (the int32
instantiations in a parity-cuda build) **instruction for instruction identical** to O's unpatched
objects, with the same registers, stack, shared and local memory; the floating-point sequences of
the hot loops (the bins' `accumulateNeighbors` / `updateLabel`, `smallFrontierKernel`, the damped
commit, `irLPKernel`, label init's phase 2) are therefore equal too (record:
`parity/results/M10a-sass-port.json`).

### 2. The pipeline on the hooks (AD21: one stream)

The library never creates a stream (`core/resources.hpp`:60): O's copy stream and its two events
become stream order on the resources' stream: Step 0 copies the ids then the edges (ADR 0074);
`before_apply` (G_t) leases and sizes the workspace, clears the affected flags and runs
`launchMarkDeletions` on Del_eff's rows before the batch; the commit applies the vertex changes and
appends; `enact_fused` sets the seeds' classes (`setGroundTruthKernel`'s `gtClass`),
`launchMarkInserted`, then, with insertions, `launchBuildSparsifiedSubgraph`,
`launchConnectedComponents` and `launchLabelInitialization` (without insertions: the endpoints of
the batch's edges are marked, ADR 0072's fix), `launchIterativePropagation` with
`enforceGroundTruthKernel`; the graph's report is settled after the propagation's first host
synchronization. Two additions: the labels before the update are kept (in the CC's root-id scratch,
free once the components are numbered) and the changed ones counted (`stats::affected`, AD12/E2:
the vertices whose label bits changed), in one more synchronization at the end, which also gives
the stats' counts. **The cost of the lost overlap** (AD21, measured): with nsys, O's host-to-device
copies overlap its kernels for 0.036 ms of its 106 ms at 50k and 0.163 ms of 482 ms at 500k
(0.03 %), the most the single stream can lose (`parity/results/M10a-early.json`, `ad21`).

The engine choice: `options::cuda_engine` on CUDA (`engine::automatic` and `engine::fused` select
this engine; `engine::operators` raises not_supported_error until M10b), the host engine on the
sequential backend. `on_limit::fallback_recompute` runs IrLP on the device. The budget of the
algorithm phase is run-dependent (no allocation once reserved; one host synchronization per
Shiloach-Vishkin round, the components' read, one per head iteration and per persistent launch, the
final read), checked under DYNG_DEBUG_BUDGETS; unchecked while the stage trace downloads the
stages. The stage trace (ADR 0073) is recorded from the device (the same records and fields as
the recorder's, the propagation's labels, counts and tags included).

### 3. compute() = IrLP on CUDA

O builds its IrLP snapshot on the host from the full graph (every row's entries between present
vertices in the full graph's row order: sorted by column, distinct columns). dynG builds it on the
device from the current rows: the live entries counted and scanned (CUB), copied in slot order and
sorted by column per row with CUB's stable segmented sort; then O's `launchIrLP` (one block per
vertex, the tree reduction, the 1e-10 guard, the clamp, the changed flag, ground truth per
iteration). An empty snapshot is skipped without leasing the scratch (the published flow's
compute() on the empty capacity graph: O starts every label at 0.5). A compact CUDA graph is read
through its compact device copy (every vertex active).

### 4. The driver

`dyng-compat-dynlp --backend cuda` page-locks the batch vectors in place (O's `pinBatches`) and
passes them as `memory_space::pinned_host`, reserves the graph's device scratch and the result's
workspace before the batch loop, times each `update()` with its own pair of CUDA events on the
resources' stream (AD2; no profiler is attached to the timed calls: the profiler's CUDA events
would synchronize every stage), copies the labels into pinned memory after the timer, and keeps
the vertex states for the evaluation as O's host pass does (no download of the rows). `--phases`
prints the run's phases (diagnostics). `perf_ab.py label_propagation run` (O then the port in every
round) and `memory --side port` are the port side of ADR 0064's harness.

## Consequences

- Damped byte parity on P+ against the goldens (`compare.py label_propagation damped --config
  cuda-fused`, parity-cuda checked): `dynlp_labels.txt` bytes, every batch's label bits (5m by
  digest), the results.csv columns and every field of the stage trace; IrLP's bits and iterations
  per snapshot at 5k5 and 50k; the hub golden bitwise, twice; the in_place cloud (every port run
  against every reference run within T_mean, T_max and T_flips, the iterations inside the frozen
  envelope, the components and the initial frontier exact); the harmonic oracle on C for both
  schedules and IrLP; the deterministic stages equal the sequential backend's
  (`parity/results/M10a.md` section 11).
- The early gate against unpatched O (`parity/results/M10a-early.json`): update[sum] and the
  device-memory peak within 1.03, end_to_end within 1.10. The fixed cost of the unified library at
  process start (its CUDA modules loaded eagerly with the context, AD2's CUDA_MODULE_LOADING=EAGER)
  is about 30 ms, which shows only in the 50k end_to_end (~400 ms runs; section 11.6).
- The plugin size: the LP CUDA objects for CUDA 13.1's release architecture list hold 11.9 MB of
  fatbin (7.6 MB of it IrLP's CUB segmented sort), about 3.3 MB compressed: a projected plugin wheel
  of about 12 MB against 0.3.0's 8.7 MB and the 90 MB limit.

## Alternatives considered

- **A caller-provided second stream** for O's overlap: a frozen-core API change, not chosen by the
  author (AD21).
- **The host snapshot for IrLP** (O's way): a download of the rows per compute(); rejected for the
  device sort (the same order, no download).
- **The profiler's CUDA events for the timed region** (step 3's driver): they synchronize at every
  stage end; replaced by the tool's own pair of events around the call.
