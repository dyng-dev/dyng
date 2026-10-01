# mosp: dynamic multi-objective shortest paths

Maturity: **experimental** (sequential, OpenMP and CUDA backends since M7, for 0.2).
Header: `<dyng/mosp.hpp>`. Oracle: `compute`. Determinism: `bitwise`. Parity: byte-identical to
MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2 on the MOSP golden corpus, combined graph included
([M7 record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M7.md)).

## 1. Problem

Input: a directed graph with K integer weight columns (objectives; weights in [1, 2^31 - 1]), a
source vertex and a preference vector Pref (one value >= 1 per objective; a lower value is a
higher priority). Output, for every vertex:

- the K canonical shortest-path trees T_0 .. T_{K-1}, one per objective (as {doc}`sssp` computes
  them: lowest parent id among equal distances);
- the **MOSP tree**: the canonical shortest-path tree of the *combined graph*. Its edges are the
  edges of the K trees; an edge (p, v) weighs `L (K + 1) - sum over the trees T_i containing it of
  L / Pref_i` (thesis Chapter 4, Algorithm MOSP_Update, Step 2), with `L = lcm(Pref)` (at most
  2^20) keeping the weights integral, so the combined distances are in units of 1/L. With the
  default Pref = (1, ..., 1), L = 1 and an edge in m trees weighs K + 1 - m: edges that many
  objectives agree on are cheap;
- the **path costs**: the K objective values of the MOSP path to every vertex (Step 3); the weights
  of a tree edge (p, v) are those of the first edge from p to v in p's row (MOSP's
  `mospPathCosts`; it matters only for parallel edges).

Update model: a batch is applied once to the graph (under its `batch_semantics`;
`graph_properties::mosp_compatible()` reproduces MOSP's `applyChangeBatch()`), the K trees are
updated incrementally (sssp's update, its tie rule included), the combined graph is rebuilt from
the K updated trees and solved from scratch, and the path costs are recomputed. Postcondition:
from canonical trees (every result of `compute()`, or `from_arrays(..., canonicalize = true)`) the
result equals `compute()` on the new graph bit for bit; from trees with non-lowest tie parents
(MOSP's dataset trees, adopted with `canonicalize = false`) each tree follows sssp's tie rule, as
the original `mosp` driver does, and the MOSP tree is the canonical tree of the combined graph of
those trees.

## 2. Template mapping

mosp is a **composition** (PLAN 4.5.2): K `sssp` problems over the objective views of one graph,
then a finalize step.

```
normalize -> translate -> prepare -> [before_apply] -> commit ->
identify_affected -> seed -> { FP: loop until is_converged } -> finalize
```

| Hook (profiler stage) | mosp |
|---|---|
| `mosp.normalize`, `mosp.commit` | the batch is normalized (set semantics) and applied once for the K objectives; the commit classifies every insertion per objective (weight increased or not) |
| translate | the objective projection: objective k is an sssp problem on weight column k |
| `mosp.objective` (K samples, objective order) | one sssp update each (its stages `sssp.prepare` .. `sssp.finalize`, or `sssp.enact_fused`, nest inside), one after the other on the handle's one sssp workspace (ADR 0015) |
| `mosp.combine` | Step 2: the combined graph of the K trees: one pass per vertex compares its K parents (the first occurrence of a parent carries the edge and its weight), count the out-degrees, prefix sum, fill |
| `mosp.combined_sssp` | Step 3: a static sssp solve of the combined graph from the source (the sssp engine of the backend, through the framework's static enactor: `sssp.reset`, `sssp.seed`, `sssp.loop`, `sssp.finalize`, or `sssp.enact_fused`) with the automatic near-far width of the combined graph |
| `mosp.finalize` | `affected`: the vertices whose combined distance or MOSP parent changed |
| `mosp.path_costs` | Step 3, last line: the path costs along the MOSP tree (on the host) |

`compute()` runs K `sssp::compute()` (each in `mosp.objective`) and the same finalize, inside
`mosp.compute`; `update()` runs inside `mosp.update`. The participant of a mosp result in
`dyng::update()` is `detail::mosp_problem` (`cpp/src/algorithms/mosp/`): it owns the K sssp
participants, so `dyng::update(res, g, batch, paths, other_result)` composes mosp with any other
result through one commit.

The four Chapter 3 challenges: (i) affected set: per objective, sssp's subtree invalidation; the
combined graph is rebuilt from all K trees (it is O(K n) and needs no affected set); (ii)
propagation scope: sssp's decrease-only propagation per objective, a full near-far solve of the
combined graph; (iii) correctness under parallelism: the combined graph is built with atomic
degree counts and fill cursors, whose order the solve does not see (lowest-id ties); (iv) data
structure: the compact CSR with objective-major weight columns; the combined CSR lives in the
handle's pooled workspace.

## 3. API

```cpp
dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
props.num_weights = 3;
auto g = dyng::graph<std::int32_t, std::int32_t, std::int32_t>::from_edges(res, edges.view(), props);
dyng::mosp::options opt;
opt.preferences = {4, 1, 4};                            // thesis Chapter 4 example; L = 4
auto paths = dyng::mosp::compute(res, g, /*source=*/0, opt);
dyng::mosp::stats st = dyng::mosp::update(res, g, batch.view(), paths);
auto costs = dyng::to_vector(res, paths.path_costs());          // n * K, vertex-major
auto tree = dyng::to_vector(res, paths.combined_parents());     // the MOSP tree
auto t1 = dyng::to_vector(res, paths.parents(1));               // objective 1's tree
```

On the thesis example (`cpp/tests/algorithms/mosp/mosp_test.cpp`, `TheThesisWorkedExample`) the
MOSP path to u7 costs (15, 3, 20) with Pref {4, 1, 4} and (15, 24, 7) with Pref {4, 4, 1}.

| Option | Default | Meaning |
|---|---|---|
| `preferences` | empty (all 1) | one value >= 1 per objective, lcm <= 2^20 (fixed at compute) |
| `delta` | 0 (automatic) | near-far width of the K sssp updates (sssp's rule per objective); the combined solve always uses the automatic width of the combined graph, as the originals do |
| `cuda_engine` | automatic | sssp's CUDA engine for the K updates and the combined solve |
| `compute_path_costs` | true | compute the path costs; while false, `path_costs()` throws |
| `validate_inputs` | true | sssp's O(n) checks of every tree adopted with `result::from_arrays()` |
| `num_objectives` | 0 (all) | the objectives are the first `num_objectives` weight columns (MOSP's `-k`; fixed at compute) |

`stats` holds the batch summary, `affected` (vertices whose combined distance or MOSP parent
changed), the K sssp stats in objective order, `combined_edges` and `preference_scale` (L).
`result::from_arrays(res, g, source, distances, parents, canonicalize, opt)` adopts K trees (lists
of K arrays) and builds the combined graph and the MOSP tree; `clone(res)` copies a result to any
backend.

## 4. Backends, engines and determinism

| Backend | K updates | Combined graph | Combined solve | Path costs |
|---|---|---|---|---|
| sequential | sssp sequential engine | count / scan / fill | sssp sequential engine | host |
| openmp | sssp OpenMP engine (`sospUpdateCpu`) | MOSP-OpenMP's `combinedGraphSospCpu` (relaxed atomics) | sssp OpenMP engine (`sospFromScratchCpu`) | host |
| cuda | sssp fused or operators engine | MOSP-CUDA's `combinedGraphSospGpu` kernels (count, CUB scan, fill) | sssp fused or operators engine (`sospFromScratchGpu`) | host (the MOSP tree is downloaded) |

Determinism: `bitwise`; every tree is canonical, so the trees, the combined arrays, the path costs,
`affected`, `combined_edges` and the objectives' deterministic counters are identical on every
backend and engine (conformance checks C3 and C4). On cuda the trees and the combined arrays are
device memory; the path costs are host memory on every backend in 0.2 (computed on the host as the
original computes them, after one download of the MOSP tree).

Host synchronizations of an update on cuda: one per objective with the fused engine (the
operators engine: sssp's budget), one for the combined graph's size, the combined solve's (its
unpack pass counts `affected`) and, with `compute_path_costs`, one for the download of the MOSP tree (timed in
`mosp.path_costs`, outside the "(a) compute" region, as in the originals). The path costs run on
the host threads of the resources handle (openmp and cuda; a level-synchronous traversal of the
MOSP tree, M7) and sequentially on the sequential backend.

## 5. Performance notes

The gates (PLAN 8.6, 6.4.4) compare the "(a) compute" region (the K updates and Steps 2-3:
`mosp.objective` and `mosp.combine` + `mosp.combined_sssp` + `mosp.finalize`) with the
originals' `gpu_compute_ms` / `compute_ms`, and "(b) end to end" with their `end_to_end_ms`;
`dyng-compat-mosp --mosp` reports both (its `RESULT compute_ms=` line). The records are in
`parity/results/M7.md`.

Memory: the result holds the K trees (12 bytes per vertex each), the MOSP tree (12 bytes per
vertex; on the host backends twice, the previous one being kept for `affected`; on cuda the
combined solve counts `affected` while it overwrites the previous tree) and the path costs (8 K
bytes per vertex, host).
The combined CSR (up to K n edges) is pooled scratch of the resources handle, shared by every mosp
result run through it.

## 6. Limitations

- At most 64 objectives (`max_objectives`) and L = lcm(Pref) <= 2^20.
- The combined graph is rebuilt and solved from scratch at every update (as in the originals);
  its size is at most K n edges, and with 32-bit edge offsets K n must fit them.
- The path costs are computed on the host.

## 7. Differences from the paper

- Step 2's weights `K + 1 - sum 1 / Pref_i` (thesis Chapter 4) are scaled by L = lcm(Pref) to
  stay integral, `L (K + 1) - sum L / Pref_i`, as in the fixed code; the combined distances are in
  units of 1/L (divide by `preference_scale()` for the thesis' values). The path costs are exact
  either way.
- Step 3 solves the combined graph from scratch with the near-far engine of the SOSP updates (the
  fixed code's `sospFromScratch*`).

## 8. Paper vs fixed code

The published code (MOSP_ESCHER's `parallelCombinedGraph`, the original design) built the combined
graph through a host `std::map` of tree-edge memberships with W = K + 1 - m (no preferences),
wrote it as temporary text files (an empty base graph plus every combined edge as an insertion)
and ran the file-based SOSP update on them. The fixed originals (MOSP-CUDA@e220ee2,
MOSP-OpenMP@c352151) build it in memory (one pass per vertex over its K parents: count, scan,
fill), support preference vectors and solve it with `sospFromScratch*`; dynG ports the fixed code
(the temporary-file path is not ported, PLAN 6.4.4). dynG's own changes: K up to 64 (the originals
stop at 32); the K trees are separate sssp results instead of one objective-major array;
`affected` (the originals report no such counter); the path costs are part of `update()` (the
originals compute them while writing `mospCosts.txt`) and cover the K objectives (MOSP's cover
every weight column of the graph; `dyng-compat-mosp -k` keeps the original's file); the in-edges
of the combined graph are built for the sequential engine's distance-only parent recovery.

## 9. Mapping from the original code

| Original | dynG |
|---|---|
| `mospUpdate()` | `mosp::update()` (the batch applied once by the framework's commit) |
| `MospOptions{source, numberOfObjectives, preferences, delta}` | `compute(res, g, source, options{preferences, delta, num_objectives, ...})` |
| `MospResult` | `mosp::result` (`distances(k)`, `parents(k)`, `combined_distances()`, `combined_parents()`) and `mosp::stats` |
| `preferenceScale()` | `result::preference_scale()`, `stats::preference_scale` |
| `combinedEdgeWeight()`, `combinedEdge` | the combine step (`cpp/src/algorithms/mosp/`) |
| `combinedGraphSospGpu()` / `combinedGraphSospCpu()` | `mosp.combine` + `mosp.combined_sssp` |
| `CombineWorkspace` | the pooled `detail::mosp_workspace` / `mosp_cuda_workspace` |
| `mospPathCosts()` | `result::path_costs()`; `testing::mosp_path_costs_reference()` |
| `combinedGraphReference()` | `testing::combined_graph_reference()` |
| `mosp` driver | `dyng-compat-mosp --mosp` (tools/compat) |

## 10. How to cite

`dyng::citation("mosp")` (keys `dynamosp2025` and `dynamosptpds2025` in `docs/references.bib`).
