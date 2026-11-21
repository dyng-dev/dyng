# label_propagation: binary harmonic label propagation under vertex batches

Maturity: **stable** from 0.4.0 (the author's decision of 2026-11-18 on AD15, GOVERNANCE.md; the
API was reviewed and frozen at the end of M10b, ADR 0109). The sequential and CUDA backends, the
Python module, the command line and the benchmark suite `benchmarks/paper/ics26_dynlp.yaml` are
new in 0.4.0. Header: `<dyng/label_propagation.hpp>`.
Python: `dyng.label_propagation`. Command line: `dyng label_propagation compute|update`. Oracle:
`reference` (the converged harmonic solution). Determinism: `tolerance` (the published `in_place`
rule); the `damped` schedule is bitwise per backend and build. Backends: sequential and cuda (no
OpenMP backend in 0.4, AD19). Parity: the port of LabelPropagation-CUDA@a276a3a, its damped labels
byte-identical to the original's ([M10a certificate](https://github.com/dyng-dev/dyng/blob/main/parity/results/M10a.md),
[M10b](https://github.com/dyng-dev/dyng/blob/main/parity/results/M10b.md)). History:
{doc}`../history/label_propagation_cuda`.

## 1. Problem

Input: an undirected graph with float similarity weights in (0, 1] (a kNN or similarity graph),
some of whose vertices are **seeds**: they carry a class, 0 or 1 (the ground truth). Output: a
label F(v) in [0, 1] for every vertex. A seed keeps its class; every other vertex tends to the
weighted average of its neighbours' labels, so that F is the **harmonic solution** of the graph
(Zhu, Ghahramani and Lafferty's semi-supervised label propagation): the unique fixed point of
F(v) = sum_u w(u, v) F(u) / sum_u w(u, v) with the seeds fixed. The class of a vertex is
`F(v) >= 0.5`.

**Weights must be positive** (the harmonic average needs them; the original's similarity
weights are in (0, 1], and weights above 1 are read the same way). dynG 0.4 does **not** check
this: as for every float graph, NaN and infinite weights are refused, but a weight <= 0 is taken
as given, and the labels are then not the harmonic solution of the input and may leave [0, 1].
Refusing such weights in `compute()` and in the batches is planned after 0.4.0
({doc}`../roadmap`).

**Update model.** Vertices arrive in batches with their similarity edges (some of them seeds) and
leave (with their edges); a batch is applied with its **net effect** (an id that arrives and
leaves in the same batch never enters the graph; `batch_semantics::net_effect()`), and `update()`
brings the labels up to date: it propagates from the vertices the batch affects until no label
changes by more than `options::tolerance` (delta). The result is a **delta-approximation** of the
harmonic solution, not the solution: only the neighbours of vertices that changed by more than
delta are revisited (section 7). `compute()` is the static baseline (IrLP: Jacobi iterations over
every unlabelled active vertex, stopped globally).

**The published setup** (PLAN 6.4.7; ADR 0071; the documented default, the author's decision AD16
of 2026-10-28): the full graph is known up front (a vertex-arrival stream), so the graph gets the
stream's full degrees as its row capacities (`graph::with_capacity(stream.row_capacity)`,
`row_layout::slotted`), `compute()` runs on the empty graph (every label 0.5), and every batch
b0, b1, ... is one `update()` with the published threshold tau (the full graph's average edge
weight). b0, the first ground truth and the edges among it, is an ordinary update. **The streaming
mode** (an advanced mode, no performance gate; ADRs 0103, 0104) drops the up-front knowledge:
`row_layout::slack` rows that grow, and tau from the current graph (section 3).

## 2. Template mapping

The fixed-point family (PLAN 4.5; {doc}`../concepts/update_model`).

| Step | Hook (stage `label_propagation.<hook>`) | What it does (the original's code) |
|---|---|---|
| 0 | `normalize` (the framework's Step 0 of `batch_semantics::net_effect()`) | the batch checked against G_t; Ins \ Del and Del ∩ V_t (runDynLP's host pass) |
| 1a | `before_apply` (on G_t) | the neighbours of the deleted vertices, on the rows before the batch, are affected (markDeletionsKernel) |
| commit | `commit` | the status changes and the appends of `graph::apply()` (launchApplyVertexChanges, launchAppendEdges) |
| 1b | `identify_affected` (`.sparsify`, `.components`) | the inserted vertices and both endpoints of every new edge are affected; G' (the edges of weight > tau between two inserted vertices) and its components (Shiloach-Vishkin hook and jump) |
| 1b | `seed` | each component's initial label W_L1 / (W_L0 + W_L1) from its supernode weights to the two classes, summed in 64-bit fixed point (labelInitialization.cu) |
| 2 | `loop` | the propagation over the frontier: two hashed half-sweeps per iteration, run-wide propagation tags (iterativePropagation.cu) |
| finish | `finalize` | the seeds' labels restored, the stats |

The template card's four challenges (thesis Chapter 3):

| Challenge | label_propagation |
|---|---|
| Affected-set rule | V_aff = N_t(Del) ∪ Ins ∪ N_{t+1}(Ins), with the adjacency before the deletions (Algorithm 2 of the paper, as the fixed code computes it) |
| Propagation scope | the frontier: a vertex whose label changed by more than delta adds its neighbours; the loop ends when the frontier is empty or at `max_iterations` |
| Correctness under parallelism | `in_place`: every vertex is updated once per iteration in one of two hashed half-sweeps (the second reads the first's labels), which breaks the period-2 oscillation of plain Jacobi; `damped`: F' = (F + average) / 2 from the previous iteration's labels, bitwise reproducible |
| Data structure | slot rows (`row_layout::slotted`: a fixed capacity per vertex, appends at `row_end`, deleted vertices kept as inactive rows), resident on the device between updates (ADRs 0069, 0074) |

## 3. API

```cpp
#include <dyng/dyng.hpp>

#include <iostream>

const auto res = dyng::resources::cuda();  // or dyng::resources::sequential(); no OpenMP backend
dyng::generators::vertex_arrival_options va;  // 50,000 vertices, 10 batches, degree 5, seed 42
const auto stream = dyng::generators::generate_vertex_arrival_stream(res, va);
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
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
opt.update_schedule = dyng::label_propagation::schedule::damped;
auto r = dyng::label_propagation::compute(res, g, {}, {}, opt);  // the empty graph: labels 0.5
r.reserve(res, stream.max_batch_vertices, stream.max_batch_edges);
for (const auto& batch : stream.batches) {
  const dyng::label_propagation::stats s = dyng::label_propagation::update(res, g, batch.view(), r);
  std::cout << s.iterations << ' ' << s.components << '\n';
}
const std::vector<float> labels = dyng::to_vector(res, r.labels());
```

The same in Python (labels as a zero-copy tensor; the whole example is
`examples/python/label_propagation_torch.py`):

```python
import dyng, torch

res = dyng.Resources.cuda()  # or dyng.Resources("sequential")
stream = dyng.generators.vertex_arrival_stream(total_vertices=50_000, num_batches=10, seed=42,
                                               resources=res)
g = dyng.Graph.with_capacity(stream.row_capacity, layout="slotted", semantics="net_effect",
                             resources=res)
lp = dyng.label_propagation.compute(g, update_schedule="damped",
                                    similarity_threshold=stream.similarity_threshold)
for batch in stream.batches:
    st = dyng.label_propagation.update(g, batch, lp)
F = torch.from_dlpack(lp.labels)  # float32 on cuda:0, no copy; read it, do not write it
```

A small run on the sequential backend, checked by the documentation tests (the deterministic
stages per batch: active vertices, components of G', its edges and the initial frontier):

<!-- snippet: label-propagation-python -->
```python
import dyng

res = dyng.Resources("sequential")  # or dyng.Resources.cuda(); no OpenMP backend
s = dyng.generators.vertex_arrival_stream(total_vertices=2000, num_batches=4, seed=7,
                                          resources=res)
g = dyng.Graph.with_capacity(s.row_capacity, layout="slotted", semantics="net_effect",
                             resources=res)
lp = dyng.label_propagation.compute(g, update_schedule="damped",
                                    similarity_threshold=s.similarity_threshold)
for b in s.batches:
    st = dyng.label_propagation.update(g, b, lp)
    print(g.num_active_vertices, st.components, st.sparsified_edges, st.initial_frontier)
F = lp.labels.to_numpy()  # lp.labels: a read-only zero-copy view; .to_numpy() copies
print(F.dtype, len(F), bool(((F >= 0) & (F <= 1)).all()))
```

<!-- snippet-output: label-propagation-python -->
```text
20 20 0 0
621 335 266 595
1222 388 247 1090
1823 388 248 1579
float32 2000 True
```

`compute(res, g, seed_vertices, seed_labels, opt)` takes the seeds of the graph's present
vertices (the seeds of later arrivals come with their batch: `edge_batch_view::insert_vertex_labels`,
-1 for an unlabelled vertex). `update(res, g, batch, r)` applies the batch to `g` and updates
`r`; `dyng::update(res, g, batch, r1, r2, ...)` updates several results through one commit.
`result::labels()` is F per vertex (seeds exactly 0 or 1, inactive vertices 0.5), `seed_labels()`
the seeds (-1 none). In Python `Result.labels` is a read-only float32 `dyng.Array` that views the
result's memory (DLPack, `__cuda_array_interface__` on cuda); while a view is alive an update
copies the result's per-vertex state first (copy-on-write), so a view keeps showing the labels it
was made from ({doc}`../api/python/index`). Copy-on-write protects the view from the result, not
the result from the view: PyTorch and CuPy have no read-only arrays, so a tensor from
`torch.from_dlpack(lp.labels)` aliases the result's labels and must not be written (a write
changes the state the next update starts from); `clone()` it to modify it.

| `options` field | Default | Meaning |
|---|---|---|
| `similarity_threshold` | `std::nullopt` | tau: an edge between two vertices inserted in the same batch joins G' when its weight is > tau. The published setup passes `stream.similarity_threshold`; `nullopt` is the streaming mode's average weight of the current graph's live entries, maintained per update, deterministic and bitwise the same on every backend (ADR 0104) |
| `tolerance` | `1e-4` | delta, the stopping rule (> 0) |
| `max_iterations` | `100000` | the cap of one update's propagation and of `compute()`'s iterations (the original's default) |
| `on_iteration_limit` | `on_limit::report` | at the cap: `report` (`stats::converged = false`), `error` (`convergence_error`) or `fallback_recompute` (IrLP; `stats::fallback_used`) |
| `update_schedule` | `schedule::in_place` | `in_place`, the published rule (not bitwise reproducible on cuda), or `damped` (1/2-damped Jacobi, bitwise per backend and build) |
| `cuda_engine` | `engine::automatic` | `automatic` / `fused` (the original's pipeline, ported verbatim) or `operators` (damped only; ADR 0101) |

`stats` (one per update) adds to `update_stats` the batch's `apply_summary`, `components` (of G'),
`sparsified_edges` (the edges of G') and `initial_frontier` (|V_aff| without the seeds and the
inactive vertices); `affected` keeps its frozen meaning (the vertices whose label bits changed;
AD12 E2).

**The streaming mode.** When the degrees are not known up front: `row_layout::slack` (rows with
`ceil(k (1 + headroom))` slots for a capacity hint k, headroom 0.125; a row that outgrows its slots
moves to the tail of the slot arena with its entries in order, and the arena is compacted when the
dead slots outnumber the live ones), `similarity_threshold = std::nullopt`, the same `update()`
({doc}`../concepts/row_layouts`, {doc}`../tutorials/streaming_label_propagation`).

## 4. Backends, engines and determinism

| Backend | `compute()` | `update()` | Determinism |
|---|---|---|---|
| sequential | IrLP (Jacobi) | the hooks on the host (ADR 0072); each half-sweep in ascending vertex order | bitwise per build, both schedules |
| cuda, `engine::fused` (the default) | IrLP, the original's kernel verbatim | the original's pipeline (ADR 0075): every kernel's SASS the original's, `--use_fast_math` on this target only | `damped`: bitwise per build, the original's bits; `in_place`: within the calibration frozen from the original's runs (AD5); the certificate reads the per-batch iteration envelope by the rate rule of ADR 0111 (30 runs of each side: the port's runs outside it not significantly more than the original's own), the labels, components and iteration sums on every run |
| cuda, `engine::operators` | IrLP | the hooks written with the operators `binned_frontier`, `neighbor_reduce`, `connected_components` (ADRs 0101, 0102), `damped` only | the fused engine's bits with the same per-batch counts (AD6's bound c·ε, c = 2^20, met at 0) |

The sequential backend is new (the original has no host DynLP): its deterministic stages (the rows,
V_aff, G', the components, the supernode weights, the initial labels) equal the original's trace
exactly, and its labels meet the harmonic oracle. The CUDA backend's graph rows stay on the device
(ADR 0074): a batch crosses PCIe once. The engine runs on the resources' one stream (the library
creates none; AD21); the original's overlap of the edge copy with its first kernels, which the
single stream gives up, was worth 0.03 % of its update time. `engine::automatic` always chooses the
fused engine; the operators engine is slower (1.25-1.83x at 50k-500k, reported) and exists as the
framework-shaped reference and the start of the operators' promotion (rule of two, ADR 0102).

The determinism of the stats (AD20, the per-field levels of `core/stats.hpp`): `affected`,
`converged` and `fallback_used` are deterministic per backend under `damped` and
schedule-dependent under `in_place`; `iterations` and `frontier_visits` are schedule-dependent
(repeatable under `damped`); `components`, `sparsified_edges`, `initial_frontier` and `batch` are
deterministic.

## 5. Complexity, memory and performance

One update costs O(|batch| + sum over the iterations of the frontier's degrees); the frontier
starts at V_aff and grows only through vertices that changed by more than delta. `compute()` costs
O(iterations x |E|). Memory: the slot rows (the full degrees of the stream: 2|E| slots), per vertex
a label, a seed label and a propagation tag, and the leased workspaces (`result::reserve()` sizes
them for the largest batch; on cuda `graph::reserve()` sizes the device scratch of a batch's edges).

The stages are listed in section 2; `parity/timed_regions/label_propagation.toml` maps them to the
original's timers: the original's `time_ms` per batch is the CUDA-event time around one public
`update()` (ADR 0064). The gates against the original (AD2, AD3; one-layout protocol of ADRs
0018 / 0021; the original re-measured in the same session; `benchmarks/paper/ics26_dynlp.yaml`):
damped `update[sum]` at most 1.05x at 50k, 500k (21 rounds), 5m and 5m1 (11 rounds); `end_to_end`
at most 1.10x (the whole process, graph files written); device memory at most 1.05x (the nsys peak
of live allocations, D1). The record of the final library is `parity/results/M10b.md` section 4
(M10a's is section 12 of `parity/results/M10a.md`); in short:

| Case | Rounds, clocks | damped `update[sum]` (bar 1.05) | `end_to_end` (bar 1.10) | device memory (bar 1.05) |
|---|---|---|---|---|
| 50k | 21, boost lock | 1.006 | 1.050 | 1.002 |
| 500k | 21, boost lock | 1.008 | 1.030 | 1.002 |
| 5m | 11, boost lock | 0.987 | 1.040 | 1.002 |
| 5m1 | 11, boost lock | 1.013 | 1.086 | 1.012 |
| 50m (recorded, AD4; gated once in the release reading at 5 rounds) | 3, base lock | 1.013 | 1.105 | |

(The ratios are dynG's over the original's, measured in the same session.)

`in_place`, the default clocks and the kernels are reported, not gated. Each algorithm's CUDA
kernels are registered with the CUDA runtime only when a process first runs that algorithm (ADR
0109 part B), so a label_propagation process no longer loads the kernels of the other algorithms
at its start-up (not with CUDA 12's static runtime, the `dyng-cu12` plugin, which registers every
module at load time as before 0.4).

**The streaming mode** against the published setup (dynG against dynG, reported, never gated;
`parity/results/M10b.md` section 2.8): at 500k 1.39x the update time (1.48x with the streaming
threshold) and 1.59x the device memory; damped on slack rows is bitwise slotted's while no
compaction has run.

## 6. Limitations

- `update()` needs `row_layout::slotted` or `row_layout::slack` (`graph::with_capacity()`); it
  refuses a batch that would grow the vertex set of a slack graph (`allow_vertex_growth`): build
  the graph with the stream's final vertex count. The other algorithms refuse slotted and slack
  graphs (`not_supported_error`, one central check; AD11): copy the edges into a compact graph.
- Edge deletions are not supported (vertices leave with their edges); edges arrive with their
  later endpoint, each at most once (a second delivery that does not fit its row is a
  `capacity_error`).
- After a seed is deleted, labels far from it can stay stale. The propagation starts at the
  deleted seed's neighbours and stops where the changes fall below `options::tolerance` (under
  `damped` each update halves the change, so the wave dies out within a few hops), and the
  harmonic solution of the seed's component moves further than the wave reaches; those labels
  catch up only when a later batch's affected vertices reach them. LabelPropagation-CUDA's flow
  never deletes a seed (its generator deletes active unlabelled vertices only), and the harmonic
  oracle's class rule (AD7) is checked on that flow; a seed's departure is covered by the hand
  case `LabelPropagation.ASeedFreePairConverges` (the author's decision of 2026-10-30, ADR 0076).
- Two classes (the paper's binary problem); 32-bit vertex ids and float weights only.
- `in_place` on cuda is not bitwise reproducible (a vertex reads whatever its neighbours hold at
  that instant); use `damped` where bits matter.
- The operators engine runs `damped` only.

## 7. Differences from the paper

dynG ports the fixed code of LabelPropagation-CUDA@a276a3a, not the paper's pseudo-code or the
code the paper measured. What the fixed code itself says about the paper (the author's decision
AD17 of 2026-10-28, jointly with M11's HD19; quoted from the original's `CHANGES.md` and
`README.md`):

- **Accuracy against the true labels is about 51 %.** "The accuracy vs the true labels is about
  51% for every version: the synthetic weights carry little recoverable signal with 1% seeds
  (IrLP reaches the same value); it is not a quality measure of the update" (`results/README.md`).
  dynG therefore measures quality against IrLP and against the harmonic solution
  (`dyng::testing::harmonic_solution`, PCG in double precision), never against the generator's
  true labels.
- **DynLP is delta-approximate.** "On these synthetic graphs DynLP's δ-stopped labels disagree
  with the harmonic optimum's class for 0.4–6% of the vertices after the last batch (5M vs 50K,
  in place; damped is about twice as far) ... This is a property of the published stopping rule
  (only neighbors of vertices that changed by more than δ are revisited, so sub-δ changes are
  never propagated), not of the GPU code; smaller δ closes the gap linearly. It matters for the
  paper's '~99% accuracy relative to the optimal solution' claim" (`CHANGES.md` §6). The oracle
  is mean |F - F*| <= 2000 delta (AD7).
- **The half-sweeps.** The paper's update is asynchronous with no ordering. "Each iteration now
  updates the frontier in two half-sweeps: a hash of (vertex, iteration) assigns every frontier
  vertex to one half, and the second half reads the labels written by the first" (`CHANGES.md`
  §2, D-c / D-d; PLAN's issue I4 cites it as "(D-b)", the commit list's label is D-c / D-d):
  without it a fine-grained thread mapping lets the update degenerate into a period-2 Jacobi
  oscillation. The `damped` schedule ("½-damped Jacobi, bitwise reproducible") is the fixed
  code's addition, with the same fixed point.
- **IrLP is the unchanged block-per-vertex kernel, so the speedups are overstated.** "The IrLP
  kernel is deliberately left as in the paper, so the speedups are against the paper's
  block-per-vertex IrLP. In an unpublished experiment the same 8-lane mapping made IrLP about 23×
  faster at 500K, which would shrink the reported speedups under a like-for-like comparison"
  (`CHANGES.md` §6). dynG's `compute()` is that IrLP kernel verbatim (it is the baseline the paper
  compares with).
- **tau is used only for G'.** "Algorithm 2's pseudo-code adds only edges with Sim > τ to
  E_{t+1}, while the implementation section describes τ as a temporary exclusion for connected
  components. The code keeps all edges in the graph and uses τ only for G'" (`CHANGES.md` §6).
- **Deletion candidates include same-batch insertions.** "The thesis samples deletions 'from the
  existing graph'; the generator also picks vertices inserted in the same batch (about 1/b of the
  pool in batch b, all of them in batch 1) ... DynLP applies the net effect of the batch"
  (`CHANGES.md` §6); dynG's `batch_semantics::net_effect()`.
- **Batch 0 is an update.** The first ground truth arrives as batch 0 and is applied, counted and
  timed like every other batch (it consumes a propagation tag), as in the original's `runDynLP`.
- **The timed scope includes the host bookkeeping and the transfers.** The fixed code's time per
  batch is "batch transfer, kernels, host bookkeeping up to the end of propagation; label download
  and accuracy are reported separately (`eval_ms`)" (`CHANGES.md` §1), where the paper reports the
  GPU computation; dynG's gates time the same scope as the fixed code (ADR 0064).

## 8. Paper vs fixed code

The DynLP paper (ICS 2026) was measured on the research code of the time (tag
`baseline-2026-09` = `f0a665e`, the 2026-03 refactor, and the earlier `f042d2f` it refactored).
The pinned original that dynG ports is the **fixed** code (`fix/correctness-perf`,
LabelPropagation-CUDA@a276a3a), whose `CHANGES.md` and `results/README.md` record what changed and
by how much. dynG is byte-identical to the fixed code's damped labels, not to the paper's code. The
paper's own figures were not re-derived by the fixed code or by dynG; the comparisons below are the
original's own records (RTX A5000, CUDA 13.1, synthetic Erdős-Rényi graphs of average degree 5,
delta 1e-4, seed 42, medians of 3).

Correctness (the fixed code's list, quoted from `CHANGES.md` §2 and the README):

| Paper's code (`baseline-2026-09`) | Fixed code and dynG |
|---|---|
| **Division by zero in the initialization** (Algorithm 2, line 22): "When a connected component has no edges to either ground-truth supernode, the formula's denominator is zero" | "Fixed by defaulting to 0.5 (neutral)" |
| **Division by zero in the propagation** (line 28): "When a vertex has no active neighbors, W_all = 0" | "Fixed by skipping the update for isolated vertices" |
| D-a: "connected components on the wrong edge list": out-of-bounds reads, wrong component ids, "at ≥ 500K vertices the run aborted with an illegal-address error" | the components of G' (oracle: the partition equals a host union-find) |
| D-b: "affected set too small" ("dropping about two thirds of V_aff") | both endpoints of every new edge and every inserted vertex are affected; tau only selects G' |
| D-e: "deletions ineffective, deleted vertices resurrected": a vertex inserted and deleted in the same batch "stayed ACTIVE in all later batches" | deletion neighbours read from the pre-deletion adjacency, "deletion wins" (**same-batch deletions**: the net effect, Ins \ Del and Del ∩ V_t) |
| D-c / D-d: a stale double buffer and a period-2 oscillation: "batches ran to `maxIterations = 100000` ... 5 of 10 batches at 50K and every batch at 500K and 5M" | the in-place update with two hashed half-sweeps, and the damped rule; every batch converges |
| One-sided convergence check (F'_u - F_u > delta) | \|F'_u - F_u\| > delta |
| D-h: "generator overflow": `int maxAttempts` "overflowed above 214.7M target edges" | a 64-bit counter, and "graphs whose directed entries exceed the 32-bit CSR offsets (**N·d > 2^31 − 1**) are rejected instead of overflowing" (dynG: `invalid_argument_error` from both generators) |
| D-f: "full re-upload instead of the thesis's partial transfer" | a device-resident append-only graph (dynG: slot rows resident on the device, ADR 0074) |

Performance (the fixed code's own summary, `CHANGES.md`; DynLP time in the original's scope):

| Config | Paper's code (+ fixes a-c) | Fixed code |
|---|---:|---:|
| 50K x 10 | 16.9 s, 501,625 iterations | 0.085 s, 4,927 iterations |
| 500K x 10 | 41.5 s, 900,000 iterations (the unmodified code crashes in batch 1) | 0.35 s, 6,628 iterations |
| 5M x 10 | 185.8 s, 900,000 iterations | 3.46 s, 9,072 iterations |
| 5M x 1 | 18.7 s, 100,000 iterations | 0.66 s, 394 iterations |
| 50M x 11 | not run | 36.3 s, 11,772 iterations |

"The baselines stop every batch at ≥ 500K at the 100,000-iteration cap; this branch converges in
every batch." The speedups the paper reports against IrLP **do not reproduce** as a like-for-like
comparison (section 7: IrLP keeps the block-per-vertex kernel, about 23x slower than the 8-lane
mapping at 500K), and the thesis's "9.7x" from overlapping transfers with kernels has little left
to hide in the fixed code ("Only the batch is transferred now (tens of MB at 5M), so the
transfer/kernel overlap the thesis credits with 9.7× has little left to hide in this code",
`CHANGES.md` §6; in dynG the overlap was worth 0.03 % of the update time, AD21). dynG's gates
against the fixed code are section 5.

## 9. Mapping from the original code

Every field of `DynLPConfig` (`headers/graphTypes.h` 137-172) and where it lives in dynG: the
algorithm's `options`, the generators' `vertex_arrival_options`, the command line and the parity
tool `dyng-compat-dynlp` (which takes the original's argv unchanged, ADR 0073).

| `DynLPConfig` field (original flag) | dynG | `dyng` command line | `dyng-compat-dynlp` |
|---|---|---|---|
| `totalNodes` (`--totalNodes`, 50000) | `vertex_arrival_options::total_vertices` | `--total-vertices` | `--totalNodes` |
| `avgDegree` (`--avgDegree`, 5) | `vertex_arrival_options::average_degree` | `--average-degree` | `--avgDegree` |
| `numBatches` (`--numBatches`, 10) | `vertex_arrival_options::num_batches` | `--num-batches` | `--numBatches` |
| `seed` (`--seed`, 42) | `vertex_arrival_options::seed` | `--seed` | `--seed` |
| `tau` (`--tau`, -1: the average edge weight) | `options::similarity_threshold` (the published setup passes `stream.similarity_threshold`) | `--similarity-threshold published\|current\|TAU` | `--tau` (negative: the original's automatic mode; 0 and NaN rejected, C8) |
| `delta` (`--delta`, 1e-4) | `options::tolerance` | `--tolerance` | `--delta` |
| `maxIterations` (`--maxIter`, 100000) | `options::max_iterations` (and `options::on_iteration_limit`, dynG's) | `--max-iterations`, `--on-iteration-limit` | `--maxIter` (<= 0 rejected, C8) |
| `updateRule` (`--update inplace\|damped`) | `options::update_schedule` (`schedule::in_place`, `damped`) | `--update-schedule` | `--update` |
| `groundTruthFraction` (`--gtFraction`, 0.01) | `vertex_arrival_options::ground_truth_fraction` | `--ground-truth-fraction` | `--gtFraction` |
| `unlabeledFraction` (0.90, no flag) | derived: the arrivals that are not ground truth | | |
| `deletedFraction` (0.09, no flag) | `vertex_arrival_options::deleted_fraction` | `--deleted-fraction` | |
| `outputDir` (`--output`) | the caller's (`io::write_labels`) | `--out` | `--output` |
| `validate` (`--validate`, `--noValidate`) | `compute()` per snapshot (IrLP) | | `--validate`, `--noValidate` |
| `writeGraphFiles` (`--noGraphFiles`) | `dyng generate dynlp_vertex_arrival_stream --graph-files` (Python `VertexArrivalStream.write_graph_files()`) | | `--noGraphFiles` |
| `verbose` (`--verbose`, `--quiet`) | | `--quiet` | `--verbose`, `--quiet` |
| `profile` (`--profile`) | `dyng::profiler` (stages `label_propagation.*`) | | `--profile` |
| `check` (`--check`) | `dyng::testing` (`harmonic_solution`, `union_find`, `lp_host_model`) and the conformance kit | | `--check` |
| (none) | `options::cuda_engine` | `--cuda-engine` | `--cuda-engine` (dynG's) |

The other names:

| LabelPropagation-CUDA@a276a3a | dynG |
|---|---|
| `runDynLP` (all batches of a run) | `label_propagation::compute()` once on the empty graph, then one `update()` per batch (the published setup, ADR 0071) |
| `runIrLPSnapshot`, `irLPKernel` | `compute()` (IrLP; the kernel verbatim on cuda) |
| `generateSparseGraph`, `generateBatches`, `computeAvgEdgeWeight` | `generators::legacy::dynlp_vertex_arrival_stream()` (byte for byte); the library's own `generators::generate_vertex_arrival_stream()` |
| `BatchData` | `vertex_arrival_batch` and its `view()`, an `edge_batch_view` (`insert_vertices`, `insert_vertex_labels`, the edges, `delete_vertices`) |
| `DeviceGraph` (fixed slots, `rowEnd`, sticky error flags) | `row_layout::slotted` (`graph::with_capacity`; `graph_view::row_end`), contract violations refused with named errors (ADR 0069) |
| `VertexStatus` (`INACTIVE`, `ACTIVE`, `GROUND_TRUTH`, `DELETED`) | `vertex_status` (`graph::vertex_status()`, `graph_view::status`; mapped by name) and the result's `seed_labels()` (the original's GROUND_TRUTH = active with a seed label, C8) |
| `markDeletionsKernel`, `launchApplyVertexChanges`, `launchAppendEdges` | `before_apply`, `graph::apply()` of a vertex batch under `batch_semantics::net_effect()` |
| `sparsifySubgraph`, `launchConnectedComponents` (hook, jump) | `identify_affected` (`.sparsify`, `.components`); `operators::connected_components` in the operators engine |
| `labelInitialization` | `seed` |
| `launchIterativePropagation`, `propagateLabelsKernel` | `loop` (the fused engine; `binned_frontier` and `neighbor_reduce` in the operators engine) |
| `dynlp_labels.txt` (`i %.6f`) | `io::write_labels()` (the same bytes), `io::read_labels()` |
| `results.csv` (`batch, time_ms, iterations, frontier_visits, components, accuracy, eval_ms`) | `label_propagation::stats` and the profiler (`time_ms` = the stage `label_propagation.update`); `dyng label_propagation update` writes the same columns to `stats.csv` |
| `checker.h` oracles | `dyng::testing::harmonic_solution` (PCG), `union_find`, `lp_host_model` (ADR 0068) |
| `dynLP` command line | `dyng-compat-dynlp` (tools/compat/dynlp, ADR 0073); `dyng label_propagation` |

## 10. How to cite

`dyng::citation("label_propagation")` (Python `dyng.citation("label_propagation")`): DynLP (ICS
2026), key `dynlp2026` in `docs/references.bib` and `CITATION.cff`:

```bibtex
@inproceedings{dynlp2026,
  author    = {Shovan, S. M. and Ferdous, S. M. and Khanda, Arindam and Das, Sajal K.},
  title     = {{DynLP}: Parallel Dynamic Batch Update for Graph-based Label Propagation in
               Semi-Supervised Learning},
  booktitle = {ACM International Conference on Supercomputing (ICS)},
  year      = {2026}
}
```

Please cite dynG itself as well (`dyng::citation()`, `CITATION.cff`).
