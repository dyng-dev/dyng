# label_propagation: binary harmonic label propagation under vertex batches

Maturity: **experimental** (provisional until the freeze of M10b; stable at 0.4.0 only on the
author's confirmation). Header: `<dyng/label_propagation.hpp>`. Oracle: `reference` (the
converged harmonic solution). Determinism: `tolerance` (the published `in_place` rule); the
`damped` schedule is bitwise per backend and build. Backends: sequential and cuda (there is no
OpenMP backend in 0.4). Paper: DynLP (ICS 2026); the port of LabelPropagation-CUDA@a276a3a
(ADRs 0069-0075).

:::{note}
This page describes the state of M10a step 4; M10b completes it (Python, the command line, the
formats, the streaming mode). The decisions behind it are the author's of 2026-10-28
(GOVERNANCE.md, approvals log).
:::

## 1. Problem

Every vertex gets a label F(v) in [0, 1] on an undirected graph with float similarity weights: a
seed (a vertex with a seed label, the ground truth of class 0 or 1) keeps its class, and every
other vertex tends to the weighted average of its neighbours' labels (the harmonic solution).
Vertices arrive in batches with their similarity edges, some with a seed label, and leave; each
`update()` brings the labels up to date until no label changes by more than
`options::tolerance`.

## 2. Template mapping

| Step | Hook (stage `label_propagation.<hook>`) | What it does (the original) |
|---|---|---|
| 0 | `normalize` (the framework's Step 0 of `batch_semantics::net_effect()`) | the batch checked against G_t, Ins \ Del and Del ∩ V_t (runDynLP's host pass) |
| 1a | `before_apply` (G_t) | the neighbours of the deleted vertices are affected (markDeletionsKernel) |
| commit | `commit` | the status changes and the appends of `graph::apply()` (launchApplyVertexChanges, launchAppendEdges) |
| 1b | `identify_affected` (`.sparsify`, `.components`) | the inserted vertices and the endpoints of every new edge are affected; G' (weight > tau between two inserted vertices) and its components |
| 1b | `seed` | the supernode initialization in 64-bit fixed point (labelInitialization.cu) |
| 2 | `loop` | the propagation: two hashed half-sweeps per iteration, run-wide tags (iterativePropagation.cu) |
| finish | `finalize` | the seeds' labels restored, the stats |

`compute()` is IrLP (validation.cu), the published static baseline: Jacobi iterations over every
active vertex without a seed label.

## 3. API

```cpp
#include <dyng/dyng.hpp>
#include <dyng/label_propagation.hpp>

using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
dyng::generators::vertex_arrival_options va;  // the original's defaults: 50000 vertices, 10 batches
auto stream = dyng::generators::legacy::dynlp_vertex_arrival_stream(res, va);
dyng::graph_properties props;
props.directed = false;
props.layout = dyng::row_layout::slotted;
props.order = dyng::row_order::append;
props.parallel_edges = dyng::multi_edges::allow;
props.semantics = dyng::batch_semantics::net_effect();
lp_graph g = lp_graph::with_capacity(res, va.total_vertices, dyng::host_view(stream.row_capacity),
                                     props);
dyng::label_propagation::options opt;
opt.similarity_threshold = stream.similarity_threshold;  // the published tau
auto r = dyng::label_propagation::compute(res, g, {}, {}, opt);  // the empty graph: every label 0.5
r.reserve(res, stream.max_batch_vertices, stream.max_batch_edges);
for (const auto& batch : stream.batches) {
  dyng::label_propagation::stats s = dyng::label_propagation::update(res, g, batch.view(), r);
}
```

## 4. Backends, engines and determinism

The sequential backend (ADR 0072) is new (the original has no host DynLP): its deterministic
stages (the rows, V_aff, G', the components, the supernode weights, the initial labels) equal the
original's trace exactly, and its labels meet the harmonic oracle (mean |F - F*| <= 2000 delta,
AD7). It updates each half-sweep in ascending vertex order, so both schedules are bitwise
reproducible on it.

The CUDA backend is the original's fused engine, ported verbatim (ADR 0075): every kernel's SASS is
the original's, compiled with `--use_fast_math` on this target only, and its damped labels are the
original's bit for bit on the whole corpus (the label files, every batch's bits, the counts and
every stage of the trace); `in_place` stays inside the calibration frozen from the original's own
runs (AD5), and both schedules and IrLP meet the harmonic oracle. The graph's rows stay on the
device (ADR 0074): a batch crosses PCIe once, and the host copy of the rows is downloaded only when
something reads it. The engine runs on the resources' one stream (the library creates none; AD21);
the original's overlap of the edge copy with its first kernels, which the single stream gives up,
was worth 0.03 % of its update time. `options::cuda_engine = engine::operators` (the refactored
engine) arrives in M10b.

## 5. Performance notes

The stages are listed in section 2; `parity/timed_regions/label_propagation.toml` maps them to
the original's timers. The gates against the original (AD2) are read on the CUDA backend
(M10a step 5).

## 6. Limitations

- `update()` needs `row_layout::slotted` (`graph::with_capacity()`, the full graph's degrees as
  capacities); the streaming mode (`row_layout::slack`) arrives in M10b.
- Edge deletions are not supported (vertices leave with their edges).
- 32-bit vertex ids and float weights only.

## 7. Differences from the paper

The fixed code of LabelPropagation-CUDA@a276a3a is ported, not the paper's pseudo-code; M10b's
page quotes the original's CHANGES.md (AD17).

## 8. Mapping from the original code

| Original (a276a3a) | dynG |
|---|---|
| `DeviceGraph` (fixed slots, `rowEnd`, sticky error flags) | `row_layout::slotted` (`graph::with_capacity`), refused contract violations (ADR 0069) |
| `VertexStatus` | `vertex_status` and the result's seed labels (GROUND_TRUTH = active with a seed label) |
| `runDynLP` (all batches) | one `update()` per batch |
| `launchIterativePropagation` | the `loop` hook |
| `runIrLPSnapshot` | `compute()` |
| `dynLP` command line | `dyng-compat-dynlp` (tools/compat/dynlp, ADR 0073) |

## 9. How to cite

`dyng::citation("label_propagation")` returns the library's BibTeX entry and the entry of the
manifest's `cite` key `dynlp2026`.
