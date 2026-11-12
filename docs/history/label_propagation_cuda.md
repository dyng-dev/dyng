# LabelPropagation-CUDA

LabelPropagation-CUDA is the code of DynLP (ICS 2026; `dyng.citation("label_propagation")`, key
`dynlp2026`): binary harmonic label propagation on a similarity graph, kept up to date on the GPU
while vertices arrive in batches with their edges and leave. It also holds IrLP, the iterative
baseline the paper compares with, and a synthetic Erdős-Rényi stream generator. dynG's
`label_propagation`, the slot rows of `row_layout::slotted` and the vertex batches of
`batch_semantics::net_effect()` are ports of it.

| | |
|---|---|
| Repository | <https://github.com/SMShovan/LabelPropagation-CUDA> |
| Pinned commit | `a276a3a4e00db1f4b6d24d2bc7ba4d9bc71b6d87` (2026-09-24) |
| Paper snapshot | tag `baseline-2026-09` = `f0a665e` (the 2026-03 refactor) |
| History | 63 commits, 2024-07 to 2026-09, all by S M Shovan (56 under his identities, 7 under the placeholder `CUDA <user@example.com>`, {doc}`../developer/provenance`); 37 AI-assisted (the 2026 fixes) |
| Ported in | M10a (the fused CUDA engine verbatim, IrLP, the sequential backend, the slot rows and the vertex batches, the legacy generator, `dyng-compat-dynlp`, the gates), M10b (the operators engine, slack rows and the streaming mode, the library generator, the batch files' vertex records, the label files, Python, the command line, the kNN recipe) |
| Parity | the damped labels byte for byte (every batch's float bits, `dynlp_labels.txt`, the counts, every stage of the trace, IrLP's bits, the hub golden) on the CUDA backend; the generator byte for byte; `in_place` within the calibration frozen from the original's own runs; the harmonic oracle on every batch ([M10a](https://github.com/dyng-dev/dyng/blob/main/parity/results/M10a.md), [M10b](https://github.com/dyng-dev/dyng/blob/main/parity/results/M10b.md)) |
| Gates | damped `update[sum]` at most 1.05x, `end_to_end` at most 1.10x and the device memory at most 1.05x of the original, re-measured in the same session at 50k, 500k, 5m and 5m1 ({doc}`../algorithms/label_propagation`, section 5) |

The code was a CUDA research prototype: one executable `dynLP` with a Makefile (`--use_fast_math`,
`ARCH ?= sm_86`), a monolithic `runDynLP` over all batches, `exit()`-based error checks, and a
graph generator inside the program. Its 2026 fixes (the branch `fix/correctness-perf`) gave it
the test oracles of `checker.h` (`--check`: the harmonic solution by PCG in double precision, the
component partition by union-find, the exact affected set, the vertex status and the device rows
against a host model), stage timers (`--profile`), a `results.csv` per run and `make test`,
which dynG's parity harness builds on.

## What was ported, from which files

| LabelPropagation-CUDA@a276a3a | dynG |
|---|---|
| `src/dynLP.cu` (`runDynLP`: the batch loop, the host pass, the transfers) | `label_propagation::update()` (one batch), on the framework's hooks (`cpp/src/algorithms/label_propagation/label_propagation.cpp`; ADR 0071) |
| `src/iterativePropagation.cu` (`propagateLabelsKernel`, `markNeighborsKernel`, the degree bins, the half-sweeps) | the fused engine's `loop` (`propagation.cu`, `fused.hpp`; ADR 0075), the operators `binned_frontier` and `neighbor_reduce` of the operators engine (ADRs 0101, 0102) |
| `src/sparsifySubgraph.cu`, `src/connectedComponents.cu` (Shiloach-Vishkin hook and jump) | `identify_affected` (`sparsify.cu`, `components.cu`); the operator `connected_components` |
| `src/labelInitialization.cu` (the supernode initialization, 64-bit fixed point) | `seed` (`label_init.cu`) |
| `src/validation.cu` (`irLPKernel`, `runIrLPSnapshot`) | `label_propagation::compute()` (`irlp.cu`) |
| `src/dynamicGraph.cu` (`DeviceGraph`: fixed slots, `rowEnd`, `launchApplyVertexChanges`, `launchAppendEdges`) | `row_layout::slotted` and the vertex batches of `graph::apply()` (`cpp/src/graph/apply_slotted_device.cu`; ADRs 0069, 0074) |
| `src/generateSparseGraph.cu`, `src/generateBatches.cu` | `generators::legacy::dynlp_vertex_arrival_stream()`, byte for byte (libstdc++ 12's float draws and shuffle reimplemented, ADR 0067) |
| `headers/graphTypes.h` (`DynLPConfig`, `BatchData`, `VertexStatus`) | `label_propagation::options`, `generators::vertex_arrival_options`, `vertex_arrival_batch`, `vertex_status` |
| `src/checker.cu` (the oracles) | `dyng::testing::harmonic_solution`, `union_find`, `lp_host_model` (ADR 0068), independent of the algorithm |
| `src/utils.cu` (`parseArguments`, `writeLabelsToFile`), `src/dynLP.cu` (`computeAccuracy`), `src/main.cu` | `dyng-compat-dynlp` (the original's argv; ADR 0073), `io::write_labels()` |

Not ported: the Makefile, `exit()`-based `CUDA_CHECK`, the camelCase flags outside the
compatibility tool, the two-stream overlap of the batch copy with the first kernels (the library
creates no stream, AD21; it was worth 0.03 % of the update time) and the bench scripts.

## What was fixed in the original before the port

Recorded in the original's `CHANGES.md` (D-a to D-h: correctness; DP1 to DP3 and H-D1/H-D2:
performance), against `baseline-2026-09`:

- **D-a** the connected components ran on the wrong edge list (out-of-bounds reads; an
  illegal-address abort at 500K vertices and more); **D-b** the affected set dropped about two
  thirds of V_aff; **D-e** deletions were ineffective and a vertex inserted and deleted in the same
  batch stayed active forever.
- **D-c / D-d** a stale double buffer and plain Jacobi's period-2 oscillation sent the batches to
  the 100,000-iteration cap; the in-place update with two hashed half-sweeps (and the damped rule,
  `--update damped`) converges in every batch.
- The two divisions by zero of the paper's pseudo-code (a component without edges to either
  supernode, a vertex without active neighbours), the one-sided convergence check, **D-h** the
  generator's int32 overflow (and N·d < 2^31 checked), **D-f** the full re-upload of the graph per
  batch (now a device-resident append-only graph), **D-g** the metrics (per snapshot, against
  IrLP).
- **DP1-DP3** the 8-lane sub-warp mapping with degree bins, the frontier-resident loop and the
  per-batch overheads: 50K x 10 from 16.9 s to 0.085 s, 500K x 10 from 41.5 s to 0.35 s, 5M x 10
  from 185.8 s to 3.46 s of DynLP time against the correctness-equivalent baseline
  (`results/README.md`).

The paper's figures were measured on the code before these fixes; "Differences from the paper"
and "Paper vs fixed code" on the {doc}`../algorithms/label_propagation` page (sections 7 and 8)
quote the original's own words on what that changes (the accuracy of about 51 % against the
synthetic true labels, the delta-approximation, the IrLP speedups that do not reproduce).

## What differs in dynG

- **A library, not a run.** `runDynLP` ran a generated stream end to end; dynG has `compute()` and
  one `update()` per batch on the frozen graph container, with the seeds as inputs, the stream as
  a generator, and the original's run as the documented published setup (ADR 0071).
- **The rows are the graph's.** The original's `DeviceGraph` is `row_layout::slotted`, a layout of
  the general graph container (with `row_layout::slack` beside it for the streaming mode), and its
  status array is `vertex_status`; the other algorithms refuse those layouts by name (AD11).
- **Two schedules, two backends, two engines.** `damped` and `in_place` on a new sequential backend
  and on CUDA; on CUDA the verbatim fused engine (bitwise the original's damped labels) and an
  operators engine with the same bits (ADR 0101).
- **Errors instead of sticky flags.** A contract violation that the original recorded in a device
  error flag and reported at the end of the run is a named exception at the update that causes it
  (`capacity_error`, `invalid_argument_error`).
- **What the original accepted unchecked** (C8): `--tau 0`, a NaN `--tau` and `--maxIter <= 0`
  are refused by `dyng-compat-dynlp` and by the options' checks; a negative `--tau` keeps the
  original's automatic threshold.
- **Streaming.** Slack rows that grow, the current graph's tau, and float edge lists, `.dgt` /
  `.dgb` vertex records and a kNN recipe (`datasets/preprocess/knn_graph.py`) for your own data
  ({doc}`../tutorials/streaming_label_propagation`).

The mapping of every name and every `DynLPConfig` field is section 9 of
{doc}`../algorithms/label_propagation`.
